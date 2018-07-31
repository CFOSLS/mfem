//                                MFEM(with 4D elements) CFOSLS for 3D/4D hyperbolic equation
//                                  with adaptive refinement involving a div-free formulation
//
// Compile with: make
//
// Description:  This example code solves a simple 3D/4D hyperbolic problem over [0,1]^3(4)
//               corresponding to the saddle point system
//                                  sigma_1 = u * b
//							 		sigma_2 - u        = 0
//                                  div_(x,t) sigma    = f
//                       with b = vector function (~velocity),
//						 NO boundary conditions (which work only in case when b * n = 0 pointwise at the domain space boundary)
//						 and initial condition:
//                                  u(x,0)            = 0
//               Here, we use a given exact solution
//                                  u(xt) = uFun_ex(xt)
//               and compute the corresponding r.h.s.
//               We discretize with Raviart-Thomas finite elements (sigma), continuous H1 elements (u) and
//					  discontinuous polynomials (mu) for the lagrange multiplier.
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>

// avoids elimination of the scalar unknown from L^2, used only temporarily for studying the problem
// currently, minsolver produces incorrect result for this formulation, but
// the saddle-point solve is fine, gives approximately the same error
// as the formulation with eliminated scalar unknown for CYLINDER_CUBE_TEST (same large!)
//#define HDIVL2L2

// A test with a rotating Gaussian hill in the cubic domain.
// The actual inflow for a rotation when the space domain is [-1,1]^2 is actually two corners
// and one has to split bdr attributes for the faces, which is quite a pain
// Instead, we prescribe homogeneous bdr conditions at the entire boundary except for the top,
// since the solution is 0 at the boundary anyway. This is overconstraining but works ok.
#define CYLINDER_CUBE_TEST

// if passive, the mesh is simply uniformly refined at each iteration
#define AMR

// activates using the solution at the previous mesh as a starting guess for the next problem
//#define CLEVER_STARTING_GUESS

// activates using a (simpler & cheaper) preconditioner for the problems, simple Gauss-Seidel
// never used now
//#define USE_GS_PREC

#define MULTILEVEL_PARTSOL

// used as a reference solution
#define APPROACH_0

// only the finest level consideration, 0 starting guess, solved by minimization solver
// (i.e., partsol finder is also used)
//#define APPROACH_1

//#define APPROACH_2

// the approach when we go back only for one level, i.e. we use the solution from the previous level
// to create a starting guess for the finest level
//#define APPROACH_3_2

// the full-recursive approach when we go back up to the coarsest level,
// we recoarsen the righthand side, solve from coarsest to finest level
// which time reusing the previous solution
//#define APPROACH_3

#ifdef APPROACH_0
//#define     DIVFREE_MINSOLVER
#endif

#ifdef APPROACH_1
#define     PARTSOL_SETUP
#define     DIVFREE_MINSOLVER
#endif

#ifdef APPROACH_2
#define     PARTSOL_SETUP
#define     DIVFREE_MINSOLVER
#define     CLEVER_STARTING_PARTSOL
#define     RECOARSENING_AMR
//#undef     CLEVER_STARTING_GUESS
#endif


#ifdef APPROACH_3_2
#define     PARTSOL_SETUP
#define     DIVFREE_MINSOLVER
#define     CLEVER_STARTING_PARTSOL
#define     RECOARSENING_AMR
//#undef     CLEVER_STARTING_GUESS
#endif

#ifdef APPROACH_3
#define     PARTSOL_SETUP
#define     DIVFREE_MINSOLVER
#define     RECOARSENING_AMR
#define     CLEVER_STARTING_PARTSOL
//#define     CLEVER_STARTING_GUESS
#endif

using namespace std;
using namespace mfem;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

void DefineEstimatorComponents(FOSLSProblem * problem, int fosls_func_version,
                               std::vector<std::pair<int,int> >& grfuns_descriptor,
                               Array<ParGridFunction*>& extra_grfuns,
                               Array2D<BilinearFormIntegrator *> & integs, bool verbose);

void PrintDefinedMacrosStats(bool verbose);

int main(int argc, char *argv[])
{
    int num_procs, myid;
    bool visualization = 1;
    bool output_solution = true;

    // 1. Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    bool verbose = (myid == 0);

    int nDimensions     = 3;
    int numsol          = -33;

#ifdef CYLINDER_CUBE_TEST
    numsol = 8;
#endif

    int ser_ref_levels  = 2;
    int par_ref_levels  = 0;

    const char *formulation = "cfosls"; // "cfosls" or "fosls"
    const char *space_for_S = "L2";     // "H1" or "L2"
    const char *space_for_sigma = "Hdiv"; // "Hdiv" or "H1"

    /*
    // Hdiv-H1 case
    using FormulType = CFOSLSFormulation_HdivH1Hyper;
    using FEFormulType = CFOSLSFEFormulation_HdivH1Hyper;
    using BdrCondsType = BdrConditions_CFOSLS_HdivH1_Hyper;
    using ProblemType = FOSLSProblem_HdivH1L2hyp;
    */

    // Hdiv-L2 case
#ifdef HDIVL2L2
    using FormulType = CFOSLSFormulation_HdivL2L2Hyper;
    using FEFormulType = CFOSLSFEFormulation_HdivL2L2Hyper;
    using BdrCondsType = BdrConditions_CFOSLS_HdivL2L2_Hyper;
    using ProblemType = FOSLSProblem_HdivL2L2hyp;
#else // then we eliminate the scalar unknown
    using FormulType = CFOSLSFormulation_HdivL2Hyper;
    using FEFormulType = CFOSLSFEFormulation_HdivL2Hyper;
    using BdrCondsType = BdrConditions_CFOSLS_HdivL2_Hyper;
    using ProblemType = FOSLSProblem_HdivL2hyp;
#endif

    // solver options
    int prec_option = 1; //defines whether to use preconditioner or not, and which one

    const char *mesh_file = "../data/cube_3d_moderate.mesh";
    //const char *mesh_file = "../data/square_2d_moderate.mesh";

    //const char *mesh_file = "../data/cube4d_low.MFEM";

    //const char *mesh_file = "../data/cube4d.MFEM";
    //const char *mesh_file = "dsadsad";
    //const char *mesh_file = "../data/orthotope3D_moderate.mesh";
    //const char *mesh_file = "../data/sphere3D_0.1to0.2.mesh";
    //const char * mesh_file = "../data/orthotope3D_fine.mesh";

    //const char * meshbase_file = "../data/sphere3D_0.1to0.2.mesh";
    //const char * meshbase_file = "../data/sphere3D_0.05to0.1.mesh";
    //const char * meshbase_file = "../data/sphere3D_veryfine.mesh";
    //const char * meshbase_file = "../data/beam-tet.mesh";
    //const char * meshbase_file = "../data/escher-p3.mesh";
    //const char * meshbase_file = "../data/orthotope3D_moderate.mesh";
    //const char * meshbase_file = "../data/orthotope3D_fine.mesh";
    //const char * meshbase_file = "../data/square_2d_moderate.mesh";
    //const char * meshbase_file = "../data/square_2d_fine.mesh";
    //const char * meshbase_file = "../data/square-disc.mesh";
    //const char *meshbase_file = "dsadsad";
    //const char * meshbase_file = "../data/circle_fine_0.1.mfem";
    //const char * meshbase_file = "../data/circle_moderate_0.2.mfem";

    int feorder         = 1;

    if (verbose)
        cout << "Solving (С)FOSLS Transport equation with MFEM & hypre \n";

    OptionsParser args(argc, argv);
    //args.AddOption(&mesh_file, "-m", "--mesh",
    //               "Mesh file to use.");
    args.AddOption(&feorder, "-o", "--feorder",
                   "Finite element order (polynomial degree).");
    args.AddOption(&ser_ref_levels, "-sref", "--sref",
                   "Number of serial refinements 4d mesh.");
    args.AddOption(&par_ref_levels, "-pref", "--pref",
                   "Number of parallel refinements 4d mesh.");
    args.AddOption(&nDimensions, "-dim", "--whichD",
                   "Dimension of the space-time problem.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&prec_option, "-precopt", "--prec-option",
                   "Preconditioner choice (0, 1 or 2 for now).");
    args.AddOption(&formulation, "-form", "--formul",
                   "Formulation to use (cfosls or fosls).");
    args.AddOption(&space_for_S, "-spaceS", "--spaceS",
                   "Space for S (H1 or L2).");
    args.AddOption(&space_for_sigma, "-spacesigma", "--spacesigma",
                   "Space for sigma (Hdiv or H1).");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");

    args.Parse();
    if (!args.Good())
    {
       if (verbose)
       {
          args.PrintUsage(cout);
       }
       MPI_Finalize();
       return 1;
    }
    if (verbose)
    {
       args.PrintOptions(cout);
    }

    if (verbose)
    {
        if (strcmp(formulation,"cfosls") == 0)
            std::cout << "formulation: CFOSLS \n";
        else
            std::cout << "formulation: FOSLS \n";

        if (strcmp(space_for_sigma,"Hdiv") == 0)
            std::cout << "Space for sigma: Hdiv \n";
        else
            std::cout << "Space for sigma: H1 \n";

        if (strcmp(space_for_S,"H1") == 0)
            std::cout << "Space for S: H1 \n";
        else
            std::cout << "Space for S: L2 \n";

#ifdef HDIVL2L2
        std::cout << "S: is not eliminated from the system \n";
#else
        if (strcmp(space_for_S,"L2") == 0)
            std::cout << "S: is eliminated from the system \n";
#endif
    }

    if (verbose)
        std::cout << "Running tests for the paper: \n";

    //mesh_file = "../data/netgen_cylinder_mesh_0.1to0.2.mesh";
    //mesh_file = "../data/pmesh_cylinder_moderate_0.2.mesh";
    //mesh_file = "../data/pmesh_cylinder_fine_0.1.mesh";

    //mesh_file = "../data/pmesh_check.mesh";
    mesh_file = "../data/cube_3d_moderate.mesh";

#ifdef CYLINDER_CUBE_TEST
    if (verbose)
        std::cout << "WARNING: CYLINDER_CUBE_TEST works only when the domain is a cube [0,1]! \n";
#endif

    if (verbose)
        std::cout << "For the records: numsol = " << numsol
                  << ", mesh_file = " << mesh_file << "\n";

    PrintDefinedMacrosStats(verbose);

#ifdef HDIVL2L2
    MFEM_ASSERT(strcmp(space_for_S,"L2") == 0, "Space for S must be H1 or L2!\n");
#endif

    MFEM_ASSERT(strcmp(formulation,"cfosls") == 0 || strcmp(formulation,"fosls") == 0, "Formulation must be cfosls or fosls!\n");
    MFEM_ASSERT(strcmp(space_for_S,"H1") == 0 || strcmp(space_for_S,"L2") == 0, "Space for S must be H1 or L2!\n");
    MFEM_ASSERT(strcmp(space_for_sigma,"Hdiv") == 0 || strcmp(space_for_sigma,"H1") == 0, "Space for sigma must be Hdiv or H1!\n");

    MFEM_ASSERT(!strcmp(space_for_sigma,"H1") == 0 || (strcmp(space_for_sigma,"H1") == 0 && strcmp(space_for_S,"H1") == 0), "Sigma from H1vec must be coupled with S from H1!\n");

    if (verbose)
        std::cout << "Number of mpi processes: " << num_procs << "\n";

    StopWatch chrono;

    Mesh *mesh = NULL;

    shared_ptr<ParMesh> pmesh;

    if (nDimensions == 3 || nDimensions == 4)
    {
        if (verbose)
            cout << "Reading a " << nDimensions << "d mesh from the file " << mesh_file << endl;
        ifstream imesh(mesh_file);
        if (!imesh)
        {
             std::cerr << "\nCan not open mesh file: " << mesh_file << '\n' << std::endl;
             MPI_Finalize();
             return -2;
        }
        else
        {
            mesh = new Mesh(imesh, 1, 1);
            imesh.close();
        }
    }
    else //if nDimensions is no 3 or 4
    {
        if (verbose)
            cerr << "Case nDimensions = " << nDimensions << " is not supported \n"
                 << flush;
        MPI_Finalize();
        return -1;

    }
    //mesh = new Mesh(2, 2, 2, Element::HEXAHEDRON, 1);

    if (mesh) // if only serial mesh was generated previously, parallel mesh is initialized here
    {
        for (int l = 0; l < ser_ref_levels; l++)
            mesh->UniformRefinement();

        if ( verbose )
            cout << "Creating parmesh(" << nDimensions <<
                    "d) from the serial mesh (" << nDimensions << "d)" << endl << flush;
        pmesh = make_shared<ParMesh>(comm, *mesh);
        delete mesh;
    }

    for (int l = 0; l < par_ref_levels; l++)
    {
       pmesh->UniformRefinement();
    }

    int dim = nDimensions;

#ifdef CYLINDER_CUBE_TEST
    Vector vert_coos;
    pmesh->GetVertices(vert_coos);
    int nv = pmesh->GetNV();
    for (int vind = 0; vind < nv; ++vind)
    {
        for (int j = 0; j < dim; ++j)
        {
            if (j < dim - 1) // shift only in space
            {
                // translation by -0.5 in space variables
                vert_coos(j*nv + vind) -= 0.5;
                // dilation so that the resulting mesh covers [-1,1] ^d in space
                vert_coos(j*nv + vind) *= 2.0;
            }
            // dilation in time so that final time interval is [0,2]
            if (j == dim - 1)
                vert_coos(j*nv + vind) *= 2.0;
        }
    }
    pmesh->SetVertices(vert_coos);

    /*
    std::stringstream fname;
    fname << "checkmesh.mesh";
    std::ofstream ofid(fname.str().c_str());
    ofid.precision(8);
    pmesh->Print(ofid);

    MPI_Finalize();
    return 0;
    */

#endif

    pmesh->PrintInfo(std::cout); if(verbose) cout << endl;

    // 6. Define a parallel finite element space on the parallel mesh. Here we
    //    use the Raviart-Thomas finite elements of the specified order.

    int numblocks = 1;

    if (strcmp(space_for_S,"H1") == 0)
        numblocks++;
    if (strcmp(formulation,"cfosls") == 0)
        numblocks++;

#ifdef HDIVL2L2
    numblocks = 3;
#endif

    if (verbose)
        std::cout << "Number of blocks in the formulation: " << numblocks << "\n";

   if (verbose)
       std::cout << "Running AMR ... \n";

   FormulType * formulat = new FormulType (dim, numsol, verbose);
   FEFormulType * fe_formulat = new FEFormulType(*formulat, feorder);
   BdrConditions * bdr_conds = new BdrCondsType(*pmesh);

#ifdef CYLINDER_CUBE_TEST
   delete bdr_conds;
   MFEM_ASSERT(pmesh->bdr_attributes.Max() == 6, "For CYLINDER_CUBE_TEST there must be"
                                                 " a bdr aittrbute for each face");

   std::vector<Array<int>* > bdr_attribs_data(formulat->Nblocks());
   for (int i = 0; i < formulat->Nblocks(); ++i)
       bdr_attribs_data[i] = new Array<int>(pmesh->bdr_attributes.Max());

   if (strcmp(space_for_S,"L2") == 0)
   {
       *bdr_attribs_data[0] = 1;
       (*bdr_attribs_data[0])[5] = 0;
   }
   else // S from H^1
   {
       *bdr_attribs_data[0] = 0;
       *bdr_attribs_data[1] = 1;
       (*bdr_attribs_data[1])[5] = 0;
   }
   *bdr_attribs_data[formulat->Nblocks() - 1] = 0;

   bdr_conds = new BdrConditions(*pmesh, formulat->Nblocks());
   bdr_conds->Set(bdr_attribs_data);
#endif

   /*
   // Hdiv-L2 case
   int numfoslsfuns = 1;

   std::vector<std::pair<int,int> > grfuns_descriptor(numfoslsfuns);
   // this works
   grfuns_descriptor[0] = std::make_pair<int,int>(1, 0);

   Array2D<BilinearFormIntegrator *> integs(numfoslsfuns, numfoslsfuns);
   for (int i = 0; i < integs.NumRows(); ++i)
       for (int j = 0; j < integs.NumCols(); ++j)
           integs(i,j) = NULL;

   integs(0,0) = new VectorFEMassIntegrator(*Mytest.Ktilda);

   FOSLSEstimator * estimator;

   estimator = new FOSLSEstimator(*problem, grfuns_descriptor, NULL, integs, verbose);
   */

    //#if 0
#ifdef DIVFREE_MINSOLVER
   bool with_hcurl = true;
#else
   bool with_hcurl = false;
#endif

   GeneralHierarchy * hierarchy = new GeneralHierarchy(1, *pmesh, feorder, verbose, with_hcurl);
   hierarchy->ConstructDofTrueDofs();

#ifdef DIVFREE_MINSOLVER
   hierarchy->ConstructDivfreeDops();
#endif
   FOSLSProblHierarchy<ProblemType, GeneralHierarchy> * prob_hierarchy = new
           FOSLSProblHierarchy<ProblemType, GeneralHierarchy>
           (*hierarchy, 1, *bdr_conds, *fe_formulat, prec_option, verbose);

   ProblemType * problem = prob_hierarchy->GetProblem(0);

#ifdef MULTILEVEL_PARTSOL
   const Array<SpaceName>* space_names_funct = problem->GetFEformulation().GetFormulation()->
           GetFunctSpacesDescriptor();
#endif

   FOSLSProblem* problem_mgtools = hierarchy->BuildDynamicProblem<ProblemType>
           (*bdr_conds, *fe_formulat, prec_option, verbose);
   hierarchy->AttachProblem(problem_mgtools);

#ifdef DIVFREE_MINSOLVER
   ComponentsDescriptor * descriptor;
   {
       bool with_Schwarz = true;
       bool optimized_Schwarz = true;
       bool with_Hcurl = true;
       bool with_coarsest_partfinder = true;
       bool with_coarsest_hcurl = false;
       bool with_monolithic_GS = false;
       descriptor = new ComponentsDescriptor(with_Schwarz, optimized_Schwarz,
                                                     with_Hcurl, with_coarsest_partfinder,
                                                     with_coarsest_hcurl, with_monolithic_GS);
   }
   MultigridToolsHierarchy * mgtools_hierarchy =
           new MultigridToolsHierarchy(*hierarchy, problem_mgtools->GetAttachedIndex(), *descriptor);

   GeneralMinConstrSolver * NewSolver;
   {
       bool with_local_smoothers = true;
       bool optimized_localsolvers = true;
       bool with_hcurl_smoothers = true;

       int stopcriteria_type = 3;

       int numblocks_funct = numblocks - 1;

       int size_funct = problem_mgtools->GetTrueOffsetsFunc()[numblocks_funct];
       NewSolver = new GeneralMinConstrSolver(size_funct, *mgtools_hierarchy, with_local_smoothers,
                                        optimized_localsolvers, with_hcurl_smoothers, stopcriteria_type, verbose);

       double newsolver_reltol = 1.0e-8;

       if (verbose)
           std::cout << "newsolver_reltol = " << newsolver_reltol << "\n";

       NewSolver->SetRelTol(newsolver_reltol);
       NewSolver->SetMaxIter(500);
       NewSolver->SetPrintLevel(0);
       //NewSolver->SetStopCriteriaType(0);
   }
#endif


#if defined(PARTSOL_SETUP) && defined(MULTILEVEL_PARTSOL)
   bool optimized_localsolvers = true;
   bool with_hcurl_smoothers = true;
   DivConstraintSolver * partsol_finder;

   partsol_finder = new DivConstraintSolver
           (*problem_mgtools, *hierarchy, optimized_localsolvers, with_hcurl_smoothers, verbose);

   bool report_funct = true;
#endif // for #ifdef PARTSOL_SETUP or MULTILEVEL_PARTSOL

   int fosls_func_version = 1;
   if (verbose)
    std::cout << "fosls_func_version = " << fosls_func_version << "\n";

   int numblocks_funct = 1;
   if (strcmp(space_for_S,"H1") == 0)
       ++numblocks_funct;

   /// The descriptor describes the grid functions used in the error estimator
   /// each pair (which corresponds to a grid function used in the estimator)
   /// has the form <a,b>, where:
   /// 1) a pair of the form <1,b> means that the corresponding grid function
   /// is one of the grid functions inside the FOSLSProblem, and b
   /// equals its index in grfuns array
   /// 2) a pair of the for <-1,b> means that the grid function is in the extra
   /// grid functions (additional argument in the estimator construction)
   /// and b is its index inside the extra grfuns array.
   /// (*) The user should take care of updating the extra grfuns, if they
   /// are not a part of the problem (e.g., defined on a different pfespace)

   std::vector<std::pair<int,int> > grfuns_descriptor;
   Array<ParGridFunction*> extra_grfuns;
   Array2D<BilinearFormIntegrator *> integs;

   DefineEstimatorComponents(problem_mgtools, fosls_func_version, grfuns_descriptor, extra_grfuns, integs, verbose);

   FOSLSEstimator * estimator;
   estimator = new FOSLSEstimator(*problem_mgtools, grfuns_descriptor, NULL, integs, verbose);
   problem_mgtools->AddEstimator(*estimator);

   //ThresholdSmooRefiner refiner(*estimator, 0.0001); // 0.1, 0.001
   ThresholdRefiner refiner(*estimator);

   refiner.SetTotalErrorFraction(0.9); // 0.5

#ifdef PARTSOL_SETUP
   Array<Vector*> div_rhs_lvls(0);
   Array<BlockVector*> partsol_lvls(0);
   Array<BlockVector*> partsol_funct_lvls(0);
   Array<BlockVector*> initguesses_funct_lvls(0);
#endif

#ifdef APPROACH_0
   Array<BlockVector*> problem_refsols_lvls(0);
#endif

   Array<BlockVector*> problem_sols_lvls(0);

   double saved_functvalue;


   // 12. The main AMR loop. In each iteration we solve the problem on the
   //     current mesh, visualize the solution, and refine the mesh.
#ifdef AMR
   const int max_dofs = 300000;//1600000; 400000;
#else // uniform refinement
   const int max_dofs = 600000;
#endif

   HYPRE_Int global_dofs = problem->GlobalTrueProblemSize();
   if (verbose)
       std::cout << "starting n_el = " << hierarchy->GetFinestParMesh()->GetNE() << "\n";

   double fixed_rtol = 1.0e-12; // 1.0e-10
   double fixed_atol = 1.0e-15;
   double initial_res_norm = -1.0;

   bool compute_error = true;

   // Main loop (with AMR or uniform refinement depending on the predefined macro AMR)
   int max_iter_amr = 21; // 21;
   int it_print_step = 5;
   for (int it = 0; it < max_iter_amr; it++)
   {
       if (verbose)
       {
          cout << "\nAMR iteration " << it << "\n";
          cout << "Number of unknowns: " << global_dofs << "\n\n";
       }

       problem_sols_lvls.Prepend(new BlockVector(problem->GetTrueOffsets()));
       *problem_sols_lvls[0] = 0.0;

#ifdef APPROACH_0
       problem_refsols_lvls.Prepend(new BlockVector(problem->GetTrueOffsets()));
       *problem_refsols_lvls[0] = 0.0;

       BlockVector saved_sol(problem_mgtools->GetTrueOffsets());
       saved_sol = 0.0;
       problem_mgtools->SolveProblem(problem_mgtools->GetRhs(), saved_sol, verbose, false);
       *problem_refsols_lvls[0] = saved_sol;

       // functional value for the initial guess
       BlockVector reduced_problem_sol(problem_mgtools->GetTrueOffsetsFunc());
       for (int blk = 0; blk < numblocks_funct; ++blk)
           reduced_problem_sol.GetBlock(blk) = saved_sol.GetBlock(blk);
#ifdef DIVFREE_MINSOLVER
       CheckFunctValueNew(comm,*NewSolver->GetFunctOp_nobnd(0), NULL, reduced_problem_sol,
                       "for the problem solution via saddle-point system ", verbose);
#endif
       // alternative check for studying the functional behavior
       /*
       BlockVector temp(problem->GetTrueOffsetsFunc());
       NewSolver->GetFunctOp_nobnd(0)->Mult(problem_refsols_lvls[0]->GetBlock(0), temp.GetBlock(0));
       //problem->GetOp_nobnd()->GetBlock(0,0).Mult(problem_refsols_lvls[0]->GetBlock(0), temp.GetBlock(0));
       if (verbose)
           std::cout << "temp euclidean norm = " << temp.GetBlock(0).Norml2() << "\n";
       double func_value_alt = sqrt( temp.GetBlock(0) * problem_refsols_lvls[0]->GetBlock(0)
               / temp.GetBlock(0).Size());
       if (verbose)
           std::cout << "func value alt = " << func_value_alt << "\n";
       */

       if (compute_error)
           problem_mgtools->ComputeError(saved_sol, verbose, true);

       //MPI_Finalize();
       //return 0;

       /*
       for (int l = 0; l < hierarchy->Nlevels(); ++l)
       {
           if (verbose)
               std::cout << "l = " << l << ", ref problem sol funct value \n";
           CheckFunctValueNew(comm,*NewSolver->GetFunctOp_nobnd(0), NULL, reduced_problem_sol,
                           "for the level problem solution via saddle-point system ", verbose);
       }
       */
#endif

#ifdef PARTSOL_SETUP
       initguesses_funct_lvls.Prepend(new BlockVector(problem->GetTrueOffsetsFunc()));
       *initguesses_funct_lvls[0] = 0.0;

       div_rhs_lvls.Prepend(new Vector(problem->GetRhs().GetBlock(numblocks - 1).Size()));
       problem_mgtools->ComputeRhsBlock(*div_rhs_lvls[0], numblocks - 1);

       partsol_funct_lvls.Prepend(new BlockVector(problem->GetTrueOffsetsFunc()));
#endif

#ifdef APPROACH_1
       int l = 0;
       ProblemType * problem_l = prob_hierarchy->GetProblem(l);

       // initguesses is 0 except for the exact solutions's bdr values
       *initguesses_funct_lvls[l] = 0.0;
       problem_l->SetExactBndValues(*initguesses_funct_lvls[l]);

       // computing funct rhside which absorbs the contribution of the non-homogeneous boundary conditions
       BlockVector padded_initguess(problem_l->GetTrueOffsets());
       padded_initguess = 0.0;
       for (int blk = 0; blk < numblocks_funct; ++blk)
           padded_initguess.GetBlock(blk) = initguesses_funct_lvls[l]->GetBlock(blk);

       BlockVector padded_rhs(problem_l->GetTrueOffsets());
       problem_l->GetOp_nobnd()->Mult(padded_initguess, padded_rhs);

       padded_rhs *= -1;

       BlockVector NewRhs(problem_l->GetTrueOffsetsFunc());
       NewRhs = 0.0;
       for (int blk = 0; blk < numblocks_funct; ++blk)
           NewRhs.GetBlock(blk) = padded_rhs.GetBlock(blk);
       problem_l->ZeroBndValues(NewRhs);

       partsol_finder->SetFunctRhs(NewRhs);

       HypreParMatrix & Constr_l = (HypreParMatrix&)(problem_l->GetOp_nobnd()->GetBlock(numblocks - 1, 0));

       Vector div_initguess(Constr_l.Height());

       Constr_l.Mult(initguesses_funct_lvls[0]->GetBlock(0), div_initguess);

       *div_rhs_lvls[0] -= div_initguess;

       BlockVector zero_vec(problem_l->GetTrueOffsetsFunc());
       zero_vec = 0.0;

       if (verbose)
           std::cout << "Finding a particular solution... \n";

       // only for debugging
       //BlockVector reduced_saved_sol(problem_l->GetTrueOffsetsFunc());
       //for (int blk = 0; blk < numblocks_funct; ++blk)
           //reduced_saved_sol.GetBlock(blk) = saved_sol.GetBlock(blk);
       //*div_rhs_lvls[0] += div_initguess;
       //partsol_finder->FindParticularSolution(reduced_saved_sol, *partsol_funct_lvls[0], *div_rhs_lvls[0], verbose, report_funct);

       partsol_finder->FindParticularSolution(zero_vec, *partsol_funct_lvls[0], *div_rhs_lvls[0], verbose, report_funct);

       *partsol_funct_lvls[0] += *initguesses_funct_lvls[0];
       *div_rhs_lvls[0] += div_initguess;

       // only for debugging, making partsol_funct[0] = solution of the saddle point system
       //for (int blk = 0; blk < numblocks_funct; ++blk)
           //partsol_funct_lvls[0]->GetBlock(blk) = saved_sol.GetBlock(blk);

       // functional value for the initial guess
       double starting_funct_value = CheckFunctValueNew(comm,*NewSolver->GetFunctOp_nobnd(l), NULL,
                                                     *partsol_funct_lvls[l], "for the initial guess ", verbose);

       // checking the initial guess
       {
           problem_l->ComputeBndError(*partsol_funct_lvls[0]);

           Vector tempc(Constr_l.Height());
           Constr_l.Mult(partsol_funct_lvls[0]->GetBlock(0), tempc);
           tempc -= *div_rhs_lvls[0];
           double res_constr_norm = ComputeMPIVecNorm(comm, tempc, "", false);
           if (!(res_constr_norm < 1.0e-10))
               std::cout << "res_constr_norm = " << res_constr_norm << "\n";
           MFEM_ASSERT (res_constr_norm < 1.0e-10, "");
       }

       zero_vec = 0.0;
       NewSolver->SetInitialGuess(l, zero_vec);

       //div_rhs_lvls[0]->Print();


       Vector div_partsol(Constr_l.Height());
       Constr_l.Mult(partsol_funct_lvls[0]->GetBlock(0), div_partsol);
       *div_rhs_lvls[0] -= div_partsol;

       NewSolver->SetConstrRhs(*div_rhs_lvls[0]);

       NewRhs = 0.0;

       {
           // computing rhs = - Funct_nobnd * init_guess at level l, with zero boundary conditions imposed
           BlockVector padded_initguess(problem_l->GetTrueOffsets());
           padded_initguess = 0.0;
           for (int blk = 0; blk < numblocks_funct; ++blk)
               padded_initguess.GetBlock(blk) = partsol_funct_lvls[0]->GetBlock(blk);

           BlockVector padded_rhs(problem_l->GetTrueOffsets());
           problem_l->GetOp_nobnd()->Mult(padded_initguess, padded_rhs);

           padded_rhs *= -1;
           for (int blk = 0; blk < numblocks_funct; ++blk)
               NewRhs.GetBlock(blk) = padded_rhs.GetBlock(blk);
           problem_l->ZeroBndValues(NewRhs);
       }

       NewSolver->SetFunctRhs(NewRhs);
       if (it == 0)
           NewSolver->SetBaseValue(starting_funct_value);
       else
           NewSolver->SetBaseValue(saved_functvalue);
       NewSolver->SetFunctAdditionalVector(*partsol_funct_lvls[l]);

       // solving for correction
       BlockVector correction(problem_l->GetTrueOffsetsFunc());
       correction = 0.0;

       NewSolver->SetPrintLevel(0);

       if (verbose)
           std::cout << "Solving the finest level problem... \n";

       NewSolver->Mult(l, &Constr_l, NewRhs, correction);

       *div_rhs_lvls[0] += div_partsol;


       for (int blk = 0; blk < numblocks_funct; ++blk)
       {
           problem_sols_lvls[l]->GetBlock(blk) = partsol_funct_lvls[l]->GetBlock(blk);
           problem_sols_lvls[l]->GetBlock(blk) += correction.GetBlock(blk);
       }

       if (l == 0)
       {
           BlockVector tmp1(problem_l->GetTrueOffsetsFunc());
           for (int blk = 0; blk < numblocks_funct; ++blk)
               tmp1.GetBlock(blk) = problem_sols_lvls[l]->GetBlock(blk);

           saved_functvalue = CheckFunctValueNew(comm,*NewSolver->GetFunctOp_nobnd(0), NULL, tmp1,
                           "for the finest level solution ", verbose);

           BlockVector tmp2(problem_l->GetTrueOffsetsFunc());
           for (int blk = 0; blk < numblocks_funct; ++blk)
               tmp2.GetBlock(blk) = problem_l->GetExactSolProj()->GetBlock(blk);

           CheckFunctValueNew(comm,*NewSolver->GetFunctOp_nobnd(0), NULL, tmp2,
                           "for the projection of the exact solution ", verbose);
       }

       //MPI_Finalize();
       //return 0;
#endif

#if defined(APPROACH_2) || defined(APPROACH_3) || defined (APPROACH_3_2)
       // recoarsening constraint rhsides from finest to coarsest level
       for (int l = 1; l < div_rhs_lvls.Size(); ++l)
           hierarchy->GetTruePspace(SpaceName::L2,l - 1)->MultTranspose(*div_rhs_lvls[l-1], *div_rhs_lvls[l]);

       // re-solving all the problems with coarsened rhs, from coarsest to finest
       // and using the previous soluition as a starting guess
#ifdef RECOARSENING_AMR
       int coarsest_lvl; // coarsest level to be considered
#ifdef APPROACH_2
       coarsest_lvl = 0;
#endif

#ifdef APPROACH_3_2
       coarsest_lvl = ( hierarchy->Nlevels() > 1 ? 1 : 0);
#endif

#ifdef APPROACH_3
       coarsest_lvl = hierarchy->Nlevels() - 1; // all levels from coarsest to finest (0)
#endif

       for (int l = coarsest_lvl; l >= 0; --l)
#else
       for (int l = 0; l >= 0; --l) // only l = 0
#endif
       {
           if (verbose)
               std::cout << "level " << l << "\n";
           ProblemType * problem_l = prob_hierarchy->GetProblem(l);

           // solving the problem at level l

           *initguesses_funct_lvls[l] = 0.0;

#ifdef CLEVER_STARTING_PARTSOL
           // create a better initial guess
           if (l < coarsest_lvl)
           {
               //std::cout << "size of problem_sols_lvls[l + 1] = " << problem_sols_lvls[l + 1]->Size() << "\n";
               //std::cout << "size of initguesses_funct_lvls[l] = " << initguesses_funct_lvls[l]->Size() << "\n";

               for (int blk = 0; blk < numblocks_funct; ++blk)
               {
                   //std::cout << "size of problem_sols_lvls[l + 1]->GetBlock(blk) = " << problem_sols_lvls[l + 1]->GetBlock(blk).Size() << "\n";
                   //std::cout << "size of initguesses_funct_lvls[l]->GetBlock(blk) = " << initguesses_funct_lvls[l]->GetBlock(blk).Size() << "\n";

                   hierarchy->GetTruePspace( (*space_names_funct)[blk], l)->Mult
                       (problem_sols_lvls[l + 1]->GetBlock(blk), initguesses_funct_lvls[l]->GetBlock(blk));
               }

               std::cout << "check init norm before bnd = " << initguesses_funct_lvls[l]->Norml2()
                             / sqrt (initguesses_funct_lvls[l]->Size()) << "\n";
           }
#endif
           // setting correct bdr values
           problem_l->SetExactBndValues(*initguesses_funct_lvls[l]);

           // functional value for the initial guess
           double starting_funct_value = CheckFunctValueNew(comm,*NewSolver->GetFunctOp_nobnd(l), NULL,
                                                         *partsol_funct_lvls[l], "for the initial guess ", verbose);


           // computing funct rhside which absorbs the contribution of the non-homogeneous boundary conditions
           BlockVector padded_initguess(problem_l->GetTrueOffsets());
           padded_initguess = 0.0;
           for (int blk = 0; blk < numblocks_funct; ++blk)
               padded_initguess.GetBlock(blk) = initguesses_funct_lvls[l]->GetBlock(blk);

           BlockVector padded_rhs(problem_l->GetTrueOffsets());
           problem_l->GetOp_nobnd()->Mult(padded_initguess, padded_rhs);

           padded_rhs *= -1;

           BlockVector NewRhs(problem_l->GetTrueOffsetsFunc());
           NewRhs = 0.0;
           for (int blk = 0; blk < numblocks_funct; ++blk)
               NewRhs.GetBlock(blk) = padded_rhs.GetBlock(blk);
           problem_l->ZeroBndValues(NewRhs);

           partsol_finder->SetFunctRhs(NewRhs);

           HypreParMatrix & Constr_l = (HypreParMatrix&)(problem_l->GetOp_nobnd()->GetBlock(numblocks - 1, 0));

           Vector div_initguess(Constr_l.Height());
           Constr_l.Mult(initguesses_funct_lvls[l]->GetBlock(0), div_initguess);

           *div_rhs_lvls[l] -= div_initguess;

           BlockVector zero_vec(problem_l->GetTrueOffsetsFunc());
           zero_vec = 0.0;

           if (verbose)
               std::cout << "Finding a particular solution... \n";

           // functional value for the initial guess for particular solution
           CheckFunctValueNew(comm,*NewSolver->GetFunctOp_nobnd(l), NULL, *initguesses_funct_lvls[l],
                           "for the particular solution ", verbose);


           partsol_finder->FindParticularSolution(l, Constr_l, zero_vec, *partsol_funct_lvls[l],
                                                  *div_rhs_lvls[l], verbose, report_funct);

           *partsol_funct_lvls[l] += *initguesses_funct_lvls[l];
           *div_rhs_lvls[l] += div_initguess;

           // checking the particular solution
           {
               problem_l->ComputeBndError(*partsol_funct_lvls[l]);

               HypreParMatrix & Constr = (HypreParMatrix&)(problem_l->GetOp_nobnd()->GetBlock(numblocks - 1, 0));
               Vector tempc(Constr.Height());
               Constr.Mult(partsol_funct_lvls[l]->GetBlock(0), tempc);
               tempc -= *div_rhs_lvls[l];
               double res_constr_norm = ComputeMPIVecNorm(comm, tempc, "", false);
               MFEM_ASSERT (res_constr_norm < 1.0e-10, "");
           }

           // functional value for the initial guess
           CheckFunctValueNew(comm,*NewSolver->GetFunctOp_nobnd(l), NULL, *partsol_funct_lvls[l],
                           "for the particular solution ", verbose);

           //BlockVector zero_vec(problem_l->GetTrueOffsetsFunc());
           zero_vec = 0.0;
           NewSolver->SetInitialGuess(l, zero_vec);
           NewSolver->SetConstrRhs(*div_rhs_lvls[l]);

           //if (verbose)
               //NewSolver->PrintAllOptions();

           //BlockVector NewRhs(problem_l->GetTrueOffsetsFunc());
           NewRhs = 0.0;

           // computing rhs = ...
           {
               BlockVector padded_initguess(problem_l->GetTrueOffsets());
               padded_initguess = 0.0;
               for (int blk = 0; blk < numblocks_funct; ++blk)
                   padded_initguess.GetBlock(blk) = partsol_funct_lvls[l]->GetBlock(blk);

               BlockVector padded_rhs(problem_l->GetTrueOffsets());
               problem_l->GetOp_nobnd()->Mult(padded_initguess, padded_rhs);

               padded_rhs *= -1;
               for (int blk = 0; blk < numblocks_funct; ++blk)
                   NewRhs.GetBlock(blk) = padded_rhs.GetBlock(blk);
               problem_l->ZeroBndValues(NewRhs);
           }

           NewSolver->SetFunctRhs(NewRhs);
           if (l == coarsest_lvl && it == 0)
               NewSolver->SetBaseValue(starting_funct_value);
           else
               NewSolver->SetBaseValue(saved_functvalue);
           NewSolver->SetFunctAdditionalVector(*partsol_funct_lvls[l]);


           // solving for correction
           BlockVector correction(problem_l->GetTrueOffsetsFunc());
           correction = 0.0;
           //std::cout << "NewSolver size = " << NewSolver->Size() << "\n";
           //std::cout << "NewRhs norm = " << NewRhs.Norml2() / sqrt (NewRhs.Size()) << "\n";
           //if (l == 0)
               //NewSolver->SetPrintLevel(1);
           //else
               NewSolver->SetPrintLevel(1);

           if (verbose && l == 0)
               std::cout << "Solving the finest level problem... \n";

           NewSolver->Mult(l, &Constr_l, NewRhs, correction);

           for (int blk = 0; blk < numblocks_funct; ++blk)
           {
               problem_sols_lvls[l]->GetBlock(blk) = partsol_funct_lvls[l]->GetBlock(blk);
               problem_sols_lvls[l]->GetBlock(blk) += correction.GetBlock(blk);
           }

           if (l == 0)
           {
               BlockVector tmp1(problem_l->GetTrueOffsetsFunc());
               for (int blk = 0; blk < numblocks_funct; ++blk)
                   tmp1.GetBlock(blk) = problem_sols_lvls[l]->GetBlock(blk);

               saved_functvalue = CheckFunctValueNew(comm,*NewSolver->GetFunctOp_nobnd(0), NULL, tmp1,
                               "for the finest level solution ", verbose);

               BlockVector tmp2(problem_l->GetTrueOffsetsFunc());
               for (int blk = 0; blk < numblocks_funct; ++blk)
                   tmp2.GetBlock(blk) = problem_l->GetExactSolProj()->GetBlock(blk);

               CheckFunctValueNew(comm,*NewSolver->GetFunctOp_nobnd(0), NULL, tmp2,
                               "for the projection of the exact solution ", verbose);
           }

       } // end of loop over levels

#ifdef RECOARSENING_AMR
       if (verbose)
           std::cout << "Re-coarsening (and re-solving if divfree problem in H(curl) is considered)"
                        " has been finished\n\n";
#endif
#endif

#if defined(APPROACH_0) && (!defined(APPROACH_1) && !defined(APPROACH_2) && !defined(APPROACH_3))
       *problem_sols_lvls[0] = *problem_refsols_lvls[0];
#endif

       if (compute_error)
           problem_mgtools->ComputeError(*problem_sols_lvls[0], verbose, true);

       // to make sure that problem has grfuns in correspondence with the problem_sol we compute here
       // though for now its coordination already happens in ComputeError()
       problem_mgtools->DistributeToGrfuns(*problem_sols_lvls[0]);


       // Send the solution by socket to a GLVis server.

       if (visualization && (it % 5 == 0 || it == max_iter_amr - 1)) //&& it % 4 == 0 ) //it == max_iter_amr - 1)
       {
           int ne = pmesh->GetNE();
           for (int elind = 0; elind < ne; ++elind)
               pmesh->SetAttribute(elind, elind);
           ParGridFunction * sigma = problem_mgtools->GetGrFun(0);
           ParGridFunction * S;

#ifdef HDIVL2L2
           S = problem_mgtools->GetGrFun(1);
#else
           if (problem_mgtools->GetFEformulation().Nunknowns() >= 2)
               S = problem_mgtools->GetGrFun(1);
           else // only sigma = Hdiv-L2 formulation with eliminated S
               S = (dynamic_cast<ProblemType*>(problem_mgtools))->RecoverS(problem_sols_lvls[0]->GetBlock(0));
#endif

           char vishost[] = "localhost";
           int  visport   = 19916;

           socketstream sigma_sock(vishost, visport);
           sigma_sock << "parallel " << num_procs << " " << myid << "\n";
           sigma_sock << "solution\n" << *pmesh << *sigma << "window_title 'sigma, AMR iter No."
                  << it <<"'" << flush;

           ParGridFunction * sigma_ex = new ParGridFunction(problem_mgtools->GetPfes(0));
           BlockVector * exactsol_proj = problem_mgtools->GetExactSolProj();
           sigma_ex->SetFromTrueDofs(exactsol_proj->GetBlock(0));

           socketstream sigmaex_sock(vishost, visport);
           sigmaex_sock << "parallel " << num_procs << " " << myid << "\n";
           sigmaex_sock << "solution\n" << *pmesh << *sigma_ex << "window_title 'sigma exact, AMR iter No."
                  << it <<"'" << flush;

           delete exactsol_proj;
           delete sigma_ex;

           socketstream s_sock(vishost, visport);
           s_sock << "parallel " << num_procs << " " << myid << "\n";
           s_sock << "solution\n" << *pmesh << *S << "window_title 'S, AMR iter No."
                  << it << "'" << flush;

           if (!(problem_mgtools->GetFEformulation().Nunknowns() >= 2))
               delete S;

       }

       if (output_solution && it % it_print_step == 0)
       {
           // don't know what exactly ref is used for
           int ref = 1;

           //std::ofstream fp_sigma("sigma_test_it0.vtk");
           std::string filename_sig;
           filename_sig = "sigma_it_";
           filename_sig.append(std::to_string(it));
           if (num_procs > 1)
           {
               filename_sig.append("_proc_");
               filename_sig.append(std::to_string(myid));
           }
           filename_sig.append(".vtk");
           std::ofstream fp_sigma(filename_sig);

           pmesh->PrintVTK(fp_sigma, ref, true);
           //pmesh->PrintVTK(fp_sigma);

           ParGridFunction * sigma = problem_mgtools->GetGrFun(0);
           ParGridFunction * S;

#ifdef HDIVL2L2
           S = problem_mgtools->GetGrFun(1);
#else
           if (problem_mgtools->GetFEformulation().Nunknowns() >= 2)
               S = problem_mgtools->GetGrFun(1);
           else // only sigma = Hdiv-L2 formulation with eliminated S
               S = (dynamic_cast<ProblemType*>(problem_mgtools))->RecoverS(problem_sols_lvls[0]->GetBlock(0));
#endif
           std::string field_name_sigma("sigma_h");
           sigma->SaveVTK(fp_sigma, field_name_sigma, ref);

           //std::ofstream fp_S("u_test_it0.vtk");
           std::string filename_S;
           filename_S = "u_it_";
           filename_S.append(std::to_string(it));
           if (num_procs > 1)
           {
               filename_S.append("_proc_");
               filename_S.append(std::to_string(myid));
           }
           filename_S.append(".vtk");
           std::ofstream fp_S(filename_S);

           pmesh->PrintVTK(fp_S, ref, true);
           //pmesh->PrintVTK(fp_S);

           std::string field_name_S("u_h");
           S->SaveVTK(fp_S, field_name_S, ref);

           //MPI_Finalize();
           //return 0;
       }

#ifdef AMR
       int nel_before = hierarchy->GetFinestParMesh()->GetNE();

       // testing with only 1 element marked for refinement
       //Array<int> els_to_refine(1);
       //els_to_refine = hierarchy->GetFinestParMesh()->GetNE() / 2;
       //hierarchy->GetFinestParMesh()->GeneralRefinement(els_to_refine);

       // true AMR
       refiner.Apply(*hierarchy->GetFinestParMesh());
       if (verbose)
           std::cout << "\nRefinement statistics: \n";

       int nmarked_el = refiner.GetNumMarkedElements(); // already makes a reduction over all processes

       int local_nel_before = nel_before;
       int global_nel_before;
       MPI_Reduce(&local_nel_before, &global_nel_before, 1, MPI_INT, MPI_SUM, 0, comm);

       int local_nel_after = hierarchy->GetFinestParMesh()->GetNE();
       int global_nel_after = 0;
       MPI_Reduce(&local_nel_after, &global_nel_after, 1, MPI_INT, MPI_SUM, 0, comm);

       if (verbose)
       {
           std::cout << "Marked elements percentage = " << 100 * nmarked_el * 1.0 / global_nel_before << " % \n";
           std::cout << "nmarked_el = " << nmarked_el << ", nel_before = " << global_nel_before << "\n";
           std::cout << "nel_after = " << global_nel_after << "\n";
           std::cout << "number of elements introduced = " << global_nel_after - global_nel_before << "\n";
           std::cout << "percentage (w.r.t to # before) of elements introduced = " <<
                        100.0 * (global_nel_after - global_nel_before) * 1.0 / global_nel_before << "% \n\n";
       }

       if (visualization && (it % 5 == 0 || it == max_iter_amr - 1) ) //it == max_iter_amr - 1)
       {
           const Vector& local_errors = estimator->GetLastLocalErrors();
           if (feorder == 0)
               MFEM_ASSERT(local_errors.Size() == problem_mgtools->GetPfes(numblocks_funct)->TrueVSize(), "");

           FiniteElementCollection * l2_coll;
           if (feorder > 0)
               l2_coll = new L2_FECollection(0, dim);

           ParFiniteElementSpace * L2_space;
           if (feorder == 0)
               L2_space = problem_mgtools->GetPfes(numblocks_funct);
           else
               L2_space = new ParFiniteElementSpace(problem->GetParMesh(), l2_coll);
           ParGridFunction * local_errors_pgfun = new ParGridFunction(L2_space);
           local_errors_pgfun->SetFromTrueDofs(local_errors);
           char vishost[] = "localhost";
           int  visport   = 19916;

           socketstream amr_sock(vishost, visport);
           amr_sock << "parallel " << num_procs << " " << myid << "\n";
           amr_sock << "solution\n" << *pmesh << *local_errors_pgfun <<
                         "window_title 'Local errors, AMR iter No." << it << "'" << flush;

           if (feorder > 0)
           {
               delete l2_coll;
               delete L2_space;
           }
       }

#else
       hierarchy->GetFinestParMesh()->UniformRefinement();
#endif

       if (refiner.Stop())
       {
          if (verbose)
             cout << "Stopping criterion satisfied. Stop. \n";
          break;
       }

       bool recoarsen = true;
       prob_hierarchy->Update(recoarsen);

       problem = prob_hierarchy->GetProblem(0);

       problem_mgtools->BuildSystem(verbose);
#ifdef DIVFREE_MINSOLVER
       mgtools_hierarchy->Update(recoarsen);
       NewSolver->UpdateProblem(*problem_mgtools);
       NewSolver->Update(recoarsen);
#endif

#ifdef PARTSOL_SETUP
       partsol_finder->UpdateProblem(*problem_mgtools);
       partsol_finder->Update(recoarsen);
#endif

       // checking #dofs after the refinement
       global_dofs = problem_mgtools->GlobalTrueProblemSize();

       if (global_dofs > max_dofs)
       {
          if (verbose)
             cout << "Reached the maximum number of dofs. Stop. \n";
          break;
       }

   } // end of the main AMR loop

   MPI_Finalize();
   return 0;
//#endif
}

void DefineEstimatorComponents(FOSLSProblem * problem, int fosls_func_version, std::vector<std::pair<int,int> >& grfuns_descriptor,
                          Array<ParGridFunction*>& extra_grfuns, Array2D<BilinearFormIntegrator *> & integs, bool verbose)
{
    int numfoslsfuns = -1;

    if (verbose)
        std::cout << "fosls_func_version = " << fosls_func_version << "\n";

    if (fosls_func_version == 1)
        numfoslsfuns = 2;
    else if (fosls_func_version == 2)
        numfoslsfuns = 3;

    // extra_grfuns.SetSize(0); // must come by default
    if (fosls_func_version == 2)
        extra_grfuns.SetSize(1);

    grfuns_descriptor.resize(numfoslsfuns);

    integs.SetSize(numfoslsfuns, numfoslsfuns);
    for (int i = 0; i < integs.NumRows(); ++i)
        for (int j = 0; j < integs.NumCols(); ++j)
            integs(i,j) = NULL;

    Hyper_test* Mytest = dynamic_cast<Hyper_test*>
            (problem->GetFEformulation().GetFormulation()->GetTest());
    MFEM_ASSERT(Mytest, "Unsuccessful cast into Hyper_test* \n");

    const Array<SpaceName>* space_names_funct = problem->GetFEformulation().GetFormulation()->
            GetFunctSpacesDescriptor();

    // version 1, only || sigma - b S ||^2, or || K sigma ||^2
    if (fosls_func_version == 1)
    {
        // this works
        grfuns_descriptor[0] = std::make_pair<int,int>(1, 0);
        grfuns_descriptor[1] = std::make_pair<int,int>(1, 1);

        if ( (*space_names_funct)[0] == SpaceName::HDIV) // sigma is from Hdiv
            integs(0,0) = new VectorFEMassIntegrator;
        else // sigma is from H1vec
            integs(0,0) = new ImproperVectorMassIntegrator;

        integs(1,1) = new MassIntegrator(*Mytest->GetBtB());

        if ( (*space_names_funct)[0] == SpaceName::HDIV) // sigma is from Hdiv
            integs(1,0) = new VectorFEMassIntegrator(*Mytest->GetMinB());
        else // sigma is from H1
            integs(1,0) = new MixedVectorScalarIntegrator(*Mytest->GetMinB());
    }
    else if (fosls_func_version == 2)
    {
        // version 2, only || sigma - b S ||^2 + || div bS - f ||^2
        MFEM_ASSERT(problem->GetFEformulation().Nunknowns() == 2 && (*space_names_funct)[1] == SpaceName::H1,
                "Version 2 works only if S is from H1 \n");

        // this works
        grfuns_descriptor[0] = std::make_pair<int,int>(1, 0);
        grfuns_descriptor[1] = std::make_pair<int,int>(1, 1);
        grfuns_descriptor[2] = std::make_pair<int,int>(-1, 0);

        int numblocks = problem->GetFEformulation().Nblocks();

        extra_grfuns[0] = new ParGridFunction(problem->GetPfes(numblocks - 1));
        extra_grfuns[0]->ProjectCoefficient(*problem->GetFEformulation().GetFormulation()->GetTest()->GetRhs());

        if ( (*space_names_funct)[0] == SpaceName::HDIV) // sigma is from Hdiv
            integs(0,0) = new VectorFEMassIntegrator;
        else // sigma is from H1vec
            integs(0,0) = new ImproperVectorMassIntegrator;

        integs(1,1) = new H1NormIntegrator(*Mytest->GetBBt(), *Mytest->GetBtB());

        integs(1,0) = new VectorFEMassIntegrator(*Mytest->GetMinB());

        // integrators related to f (rhs side)
        integs(2,2) = new MassIntegrator;
        integs(1,2) = new MixedDirectionalDerivativeIntegrator(*Mytest->GetMinB());
    }
    else
    {
        MFEM_ABORT("Unsupported version of fosls functional \n");
    }
}

void PrintDefinedMacrosStats(bool verbose)
{
#ifdef HDIVL2L2
    if (verbose)
        std::cout << "HDIVL2L2 active \n";
#else
    if (verbose)
        std::cout << "HDIVL2L2 passive \n";
#endif

#ifdef AMR
    if (verbose)
        std::cout << "AMR active \n";
#else
    if (verbose)
        std::cout << "AMR passive \n";
#endif

#ifdef PARTSOL_SETUP
    if (verbose)
        std::cout << "PARTSOL_SETUP active \n";
#else
    if (verbose)
        std::cout << "PARTSOL_SETUP passive \n";
#endif

#if defined(PARTSOL_SETUP) && (!(defined(DIVFREE_HCURLSETUP) || defined(DIVFREE_MINSOLVER)))
    MFEM_ABORT("For PARTSOL_SETUP one of the divfree options must be active");
#endif

#ifdef DIVFREE_MINSOLVER
    if (verbose)
        std::cout << "DIVFREE_MINSOLVER active \n";
#else
    if (verbose)
        std::cout << "DIVFREE_MINSOLVER passive \n";
#endif

#if defined(DIVFREE_MINSOLVER) && defined(DIVFREE_HCURLSETUP)
    MFEM_ABORT("Cannot have both \n");
#endif

#ifdef CLEVER_STARTING_GUESS
    if (verbose)
        std::cout << "CLEVER_STARTING_GUESS active \n";
#else
    if (verbose)
        std::cout << "CLEVER_STARTING_GUESS passive \n";
#endif

#if defined(CLEVER_STARTING_PARTSOL) && !defined(MULTILEVEL_PARTSOL)
    MFEM_ABORT("CLEVER_STARTING_PARTSOL cannot be active if MULTILEVEL_PARTSOL is not \n");
#endif

#ifdef CLEVER_STARTING_PARTSOL
    if (verbose)
        std::cout << "CLEVER_STARTING_PARTSOL active \n";
#else
    if (verbose)
        std::cout << "CLEVER_STARTING_PARTSOL passive \n";
#endif

#ifdef USE_GS_PREC
    if (verbose)
        std::cout << "USE_GS_PREC active (overwrites the prec_option) \n";
#else
    if (verbose)
        std::cout << "USE_GS_PREC passive \n";
#endif

#ifdef MULTILEVEL_PARTSOL
    if (verbose)
        std::cout << "MULTILEVEL_PARTSOL active \n";
#else
    if (verbose)
        std::cout << "MULTILEVEL_PARTSOL passive \n";
#endif

#ifdef CYLINDER_CUBE_TEST
    if (verbose)
        std::cout << "CYLINDER_CUBE_TEST active \n";
#else
    if (verbose)
        std::cout << "CYLINDER_CUBE_TEST passive \n";
#endif

#ifdef RECOARSENING_AMR
    if (verbose)
        std::cout << "RECOARSENING_AMR active \n";
#else
    if (verbose)
        std::cout << "RECOARSENING_AMR passive \n";
#endif
}



