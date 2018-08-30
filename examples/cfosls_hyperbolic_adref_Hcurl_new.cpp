///                           MFEM(with 4D elements) CFOSLS for 3D/4D transport equation
///                                      with adaptive mesh refinement,
///                                     solved by a minimization solver.
///
/// The problem considered in this example is
///                             du/dt + b * u = f (either 3D or 4D in space-time)
/// casted in the CFOSLS formulation
/// 1) either in Hdiv-L2 case:
///                             (K sigma, sigma) -> min
/// where sigma is from H(div), u is recovered (as an element of L^2) from sigma = b * u,
/// and K = (I - bbT / || b ||);
/// 2) or in Hdiv-L2-L2 case (recently added, not fully tested, if HDIVL2L2 is #defined)
///                             || sigma - b * u || ^2 -> min
/// where sigma is from H(div) and u is from L^2 *but not eliminated from the system as in 1));
/// 3) or in Hdiv-H1-L2 case
///                             || sigma - b * u || ^2 -> min
/// where sigma is from H(div) and u is from H^1;
/// minimizing in all cases under the constraint
///                             div sigma = f.
///
/// The problem is discretized using RT, linear Lagrange and discontinuous constants in 3D/4D.
/// The current 3D tests are either in cube (preferred) or in a cylinder, with a rotational velocity field b.
///
/// The problem is then solved with adaptive mesh refinement (AMR).
/// Different approaches can be used depending on the #defined macros, see their description near their
/// declaration below.
///
/// This example demonstrates usage of AMR related classes from mfem/cfosls/, such as
/// GeneralHierarchy, FOSLSProblem, FOSLSProblHierarchy, FOSLSEstimator, MultigridToolsHierarchy,
/// GeneralMinConstrSolver, DivConstraintSolver, ThresholdSmooRefiner, etc.
///
/// (**) This code was tested in serial and in parallel.
/// (***) The example was tested for memory leaks with valgrind, in 3D.
///
/// Typical run of this example: ./cfosls_hyperbolic_adref_Hcurl_new --whichD 3 --spaceS L2 -no-vis
/// If you want to use the Hdiv-H1-L2 formulation, you will need not only change --spaceS option but also
/// change the source code, around 4.
///
/// Other examples on adaptive mesh refinement, are cfosls_laplace_adref_Hcurl_new.cpp,
/// cfosls_laplace_adref_Hcurl.cpp and cfosls_hyperbolic_adref_Hcurl.cpp.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>

// Avoids elimination of the scalar unknown from L^2, used only temporarily for studying the problem
// currently, minsolver produces incorrect result for this formulation, but
// the saddle-point solve is fine, gives approximately the same error
// as the formulation with eliminated scalar unknown for CYLINDER_CUBE_TEST (same large!)
//#define HDIVL2L2

// A test with a rotating Gaussian hill in the cubic domain.
// Originally, the test was for the cylinder, hence "cylinder" test,
// but later, to avoid errors from circle boundary approximation
// the same test was considered in the cubic domain
#define CYLINDER_CUBE_TEST

// Defines whether boundary conditions for CYLINDER_CUBE_TEST are overconstraining (see below)
// The actual inflow for a rotation when the space domain is [-1,1]^2 is actually two corners
// and one has to split bdr attributes for the faces, which is quite a pain
// Instead, we prescribe homogeneous bdr conditions at the entire boundary except for the top,
// since the solution is 0 at the boundary anyway. This is overconstraining but works ok.
//#define OVERCONSTRAINED

// If passive, the mesh is simply uniformly refined at each iteration
//#define AMR

// When active, this macro makes the code create an instance of DivConstraintSolver
// whicha llows to search for a particular solution of the divergence constraint
// The flag gets activated within some of APPROACHes below
//#define PARTSOL_SETUP
// This macro, when #defined, activates usage of the minimization solver
// In this example it is coupled with PARTSOL_SETUP, cannot be used without that flag.
//#define DIVFREE_MINSOLVER

// Here the problem is solved by a preconditioned MINRES
// used as a reference solution
// This flag can be activated independently from other APPROACH_k, k > 0.
#define APPROACH_0

// NOT MORE THAN ONE OF THE APPROACH_K, K > 0, BELOW SHOULD BE #DEFINED

// Here the problem is solved by the minimization solver, but uses only the finest level,
// with zero starting guess, solved by minimization solver (i.e., partsol finder is also used)
//#define APPROACH_1

// Here the problem is solved by the minimization solver, but uses only the finest level, and takes
// as the initial guess zero vector
// It looks like it's the same as APPROACH_1, but implemented in a form closer to APPROACH_3
//#define APPROACH_2

// Here the problem is solved by the minimization solver,but uses one previous level
// so we go back only for one level, i.e. we use the solution from the previous level
// to create a starting guess for the finest level
//#define APPROACH_3_2

// Here the problem is solved by the minimization solver and exploits all the available levels,
// i.e, the full-recursive approach when we go back up to the coarsest level,
// we recoarsen the righthand side, solve from coarsest to finest level
// which time reusing the previous solution
//#define APPROACH_3

#ifdef APPROACH_0
#endif

#ifdef APPROACH_1
#define     PARTSOL_SETUP
#define     DIVFREE_MINSOLVER
#endif

#ifdef APPROACH_2
#define     PARTSOL_SETUP
#define     DIVFREE_MINSOLVER
// it never actually goes into that so it's the same as APPROACH_1
//#define     CLEVER_STARTING_PARTSOL
#define     RECOARSENING_AMR
#endif

#ifdef APPROACH_3_2
#define     PARTSOL_SETUP
#define     DIVFREE_MINSOLVER
#define     CLEVER_STARTING_PARTSOL
#define     RECOARSENING_AMR
#endif

#ifdef APPROACH_3
#define     PARTSOL_SETUP
#define     DIVFREE_MINSOLVER
#define     RECOARSENING_AMR
#define     CLEVER_STARTING_PARTSOL
#endif

using namespace std;
using namespace mfem;
using std::shared_ptr;
using std::make_shared;

// Defines required estimator components, such as integrators, grid function structure
// for a given problem and given version of the functional
void DefineEstimatorComponents(FOSLSProblem * problem, int fosls_func_version,
                               std::vector<std::pair<int,int> >& grfuns_descriptor,
                               Array<ParGridFunction*>& extra_grfuns,
                               Array2D<BilinearFormIntegrator *> & integs, bool verbose);

// Outputs which macros were #defined
void PrintDefinedMacrosStats(bool verbose);

// Rearranges boundary attributes for a rotational test in a cube so that essential boundary is
// exactly the inflow boundary, not all the bottom and lateral boundary (which would be overconstraining
// but work since the solution is localized  strictly inside the domain)
// Used when CYLINDER_CUBE_TEST is defined, but OVERCONSTRAINED is not
void ReArrangeBdrAttributes(Mesh* mesh4cube);

int main(int argc, char *argv[])
{
    // 1. Initialize MPI
    int num_procs, myid;

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

    bool visualization = 1;
    bool output_solution = true;

    int ser_ref_levels  = 1;
    int par_ref_levels  = 0;

    // This must be consistent with what formulation is used below.
    // Search for "using FormulType" below
    const char *space_for_S = "L2";     // "H1" or "L2"
    const char *space_for_sigma = "Hdiv"; // "Hdiv" or "H1" ("H1" needs to be fixed)

    // solver options
    int prec_option = 1; //defines whether to use preconditioner or not, and which one

    const char *mesh_file = "../data/cube_3d_moderate.mesh";
    //const char *mesh_file = "../data/square_2d_moderate.mesh";

    //const char *mesh_file = "../data/cube4d_low.MFEM";
    //const char *mesh_file = "../data/cube4d.MFEM";
    //const char *mesh_file = "../data/orthotope3D_moderate.mesh";
    //const char *mesh_file = "../data/sphere3D_0.1to0.2.mesh";
    //const char *mesh_file = "../data/orthotope3D_fine.mesh";
    //mesh_file = "../data/netgen_cylinder_mesh_0.1to0.2.mesh";
    //mesh_file = "../data/pmesh_cylinder_moderate_0.2.mesh";
    //mesh_file = "../data/pmesh_cylinder_fine_0.1.mesh";

    int feorder         = 1;

    if (verbose)
        cout << "Solving (С)FOSLS Transport equation in AMR setting \n";

    // 2. Parse command-line options.

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
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
            std::cout << "S is eliminated from the system \n";
#endif
    }

    // 3. Define the problem to be solved (CFOSLS Hdiv-L2 or Hdiv-H1 formulation, e.g.)
    /*
    // Hdiv-H1 case
    MFEM_ASSERT(strcmp(space_for_S,"H1") == 0, "Hdiv-H1-L2 formulation must have space_for_S = `H1` \n");
    using FormulType = CFOSLSFormulation_HdivH1Hyper;
    using FEFormulType = CFOSLSFEFormulation_HdivH1Hyper;
    using BdrCondsType = BdrConditions_CFOSLS_HdivH1_Hyper;
    using ProblemType = FOSLSProblem_HdivH1L2hyp;
    */

    // Hdiv-L2 case
#ifdef HDIVL2L2
    MFEM_ASSERT(strcmp(space_for_S,"L2") == 0, "Hdiv-L2-L2 formulation must have space_for_S = `L2` \n");
    using FormulType = CFOSLSFormulation_HdivL2L2Hyper;
    using FEFormulType = CFOSLSFEFormulation_HdivL2L2Hyper;
    using BdrCondsType = BdrConditions_CFOSLS_HdivL2L2_Hyper;
    using ProblemType = FOSLSProblem_HdivL2L2hyp;
#else // then we eliminate the scalar unknown
    MFEM_ASSERT(strcmp(space_for_S,"L2") == 0, "Hdiv-L2-L2 formulation must have space_for_S = `L2` \n");
    using FormulType = CFOSLSFormulation_HdivL2Hyper;
    using FEFormulType = CFOSLSFEFormulation_HdivL2Hyper;
    using BdrCondsType = BdrConditions_CFOSLS_HdivL2_Hyper;
    using ProblemType = FOSLSProblem_HdivL2hyp;
#endif

#ifdef CYLINDER_CUBE_TEST
    if (verbose)
        std::cout << "WARNING: CYLINDER_CUBE_TEST works only when the domain is a cube [0,1]! \n";
#endif

    if (verbose)
        std::cout << "For the records: numsol = " << numsol
                  << ", mesh_file = " << mesh_file << "\n";

    // Printing in the output which macros have been #defined
    PrintDefinedMacrosStats(verbose);

#ifdef HDIVL2L2
    MFEM_ASSERT(strcmp(space_for_S,"L2") == 0, "Space for S must be H1 or L2!\n");
#endif

    MFEM_ASSERT(strcmp(space_for_S,"H1") == 0 || strcmp(space_for_S,"L2") == 0, "Space for S must be H1 or L2!\n");
    MFEM_ASSERT(strcmp(space_for_sigma,"Hdiv") == 0 || strcmp(space_for_sigma,"H1") == 0, "Space for sigma must be Hdiv or H1!\n");

    MFEM_ASSERT(!strcmp(space_for_sigma,"H1") == 0 || (strcmp(space_for_sigma,"H1") == 0 && strcmp(space_for_S,"H1") == 0),
                "Sigma from H1vec must be coupled with S from H1!\n");

    if (verbose)
        std::cout << "Number of mpi processes: " << num_procs << "\n";

    // 4. Reading the mesh and performing a prescribed number of serial and parallel refinements
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
    int dim = nDimensions;

    // 4.1 Posprocessing the mesh in case of a "cylinder" test running in the cube
    // If the domain cube was [0,1]^2 x [0,T] before, here we want to stretch it
    // in (x,y) plane to cover [-1,1]^2 x [0,1] instead
#ifdef CYLINDER_CUBE_TEST

    Vector vert_coos;
    mesh->GetVertices(vert_coos);
    int nv = mesh->GetNV();
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
    mesh->SetVertices(vert_coos);

    // if we don't want to overconstrain the solution (see comments before
    // #define OVERCONSTRAINED), we need to manually rearrange boundary attributes
#ifndef OVERCONSTRAINED
    // rearranging boundary attributes which are now assigned to cube faces,
    // not to bot + square corners + top as we need
    ReArrangeBdrAttributes(mesh);

    /*
    std::string filename_mesh;
    filename_mesh = "checkmesh_cube4transport.mesh";
    std::ofstream ofid(filename_mesh);
    ofid.precision(8);
    mesh->Print(ofid);

    MPI_Finalize();
    return 0;
    */
#endif

#endif

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
       pmesh->UniformRefinement();

    pmesh->PrintInfo(std::cout); if(verbose) cout << endl;

    // 5. Creating an instance of the FOSLSProblem to be solved.

    // Define how many blocks there will be in the problem (loose way, correct one
    // would be to call Nblocks() for formulat which is defined below)
    int numblocks = 1;

    if (strcmp(space_for_S,"H1") == 0)
        numblocks++;
    numblocks++;

#ifdef HDIVL2L2
    numblocks = 3;
#endif

    if (verbose)
        std::cout << "Number of blocks in the formulation: " << numblocks << "\n";

   FormulType * formulat = new FormulType (dim, numsol, verbose);
   FEFormulType * fe_formulat = new FEFormulType(*formulat, feorder);
   BdrConditions * bdr_conds = new BdrCondsType(*pmesh);

   // for "cylinder" test in the cube, we redefine boundary attributes
   // since they require special treatment in this case
   // (either for overconstraining or for adjusting them fron general-purpose
   // conditions from BdrConditions_CFOSLS_HdivL2_Hype to our re-arranged bdr attributes
#ifdef CYLINDER_CUBE_TEST
   delete bdr_conds;
   MFEM_ASSERT(pmesh->bdr_attributes.Max() == 6, "For CYLINDER_CUBE_TEST there must be"
                                                 " a bdr aittrbute for each face");

   std::vector<Array<int>* > bdr_attribs_data(formulat->Nblocks());
   for (int i = 0; i < formulat->Nblocks(); ++i)
       bdr_attribs_data[i] = new Array<int>(pmesh->bdr_attributes.Max());

#ifdef OVERCONSTRAINED
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
#else // in this case we have rearranged manually the bdr attributes, see above
   if (strcmp(space_for_S,"L2") == 0)
   {
       *bdr_attribs_data[0] = 0;
       (*bdr_attribs_data[0])[0] = 1;
       (*bdr_attribs_data[0])[1] = 0;
       (*bdr_attribs_data[0])[2] = 1;
       (*bdr_attribs_data[0])[3] = 0;
       (*bdr_attribs_data[0])[4] = 1;
       (*bdr_attribs_data[0])[5] = 0;
   }
   else // S from H^1
   {
       *bdr_attribs_data[0] = 0;

       *bdr_attribs_data[1] = 0;
       (*bdr_attribs_data[1])[0] = 1;
       (*bdr_attribs_data[1])[1] = 0;
       (*bdr_attribs_data[1])[2] = 1;
       (*bdr_attribs_data[1])[3] = 0;
       (*bdr_attribs_data[1])[4] = 1;
       (*bdr_attribs_data[1])[5] = 0;
   }
   *bdr_attribs_data[formulat->Nblocks() - 1] = 0;
#endif
   bdr_conds = new BdrConditions(*pmesh, formulat->Nblocks());
   bdr_conds->Set(bdr_attribs_data);
#endif

   // for minimization solver we need to build H(curl)-related components in the GeneralHierarchy
#ifdef DIVFREE_MINSOLVER
   bool with_hcurl = true;
#else
   bool with_hcurl = false;
#endif

   // 5.1 Creating a hierarchy of meshes
   GeneralHierarchy * hierarchy = new GeneralHierarchy(1, *pmesh, feorder, verbose, with_hcurl);
   hierarchy->ConstructDofTrueDofs();
   hierarchy->ConstructEl2Dofs();

#ifdef DIVFREE_MINSOLVER
   hierarchy->ConstructDivfreeDops();
#endif

   // 5.2 Creating a problem hierarchy, which will be used in case when we ant to solve problems
   // at previous levels
   FOSLSProblHierarchy<ProblemType, GeneralHierarchy> * prob_hierarchy = new
           FOSLSProblHierarchy<ProblemType, GeneralHierarchy>
           (*hierarchy, 1, *bdr_conds, *fe_formulat, prec_option, verbose);

   ProblemType * problem = prob_hierarchy->GetProblem(0);

#ifdef PARTSOL_SETUP
   const Array<SpaceName>* space_names_funct = problem->GetFEformulation().GetFormulation()->
           GetFunctSpacesDescriptor();
#endif

   // 5.3 Creating a "dynamic" problem which always lives on the finest level of the hierarchy
   FOSLSProblem* dynamic_problem = hierarchy->BuildDynamicProblem<ProblemType>
           (*bdr_conds, *fe_formulat, prec_option, verbose);
   hierarchy->AttachProblem(dynamic_problem);

   // 5.4 In case we use minimization solver, here we create an instance of GeneralMinConstrSolver,
   // building it on MultigridToolsHierarchy
#ifdef DIVFREE_MINSOLVER
   ComponentsDescriptor * descriptor;
   {
       bool with_Schwarz = true;
       bool optimized_Schwarz = true;
       bool with_Hcurl = true;
       bool with_coarsest_partfinder = true;
       bool with_coarsest_hcurl = false;
       bool with_monolithic_GS = false;
       bool with_nobnd_op = true;
       descriptor = new ComponentsDescriptor(with_Schwarz, optimized_Schwarz,
                                             with_Hcurl, with_coarsest_partfinder,
                                             with_coarsest_hcurl, with_monolithic_GS,
                                             with_nobnd_op);
   }
   MultigridToolsHierarchy * mgtools_hierarchy =
           new MultigridToolsHierarchy(*hierarchy, dynamic_problem->GetAttachedIndex(), *descriptor);

   GeneralMinConstrSolver * NewSolver;
   {
       bool with_local_smoothers = true;
       bool optimized_localsolvers = true;
       bool with_hcurl_smoothers = true;

       int stopcriteria_type = 3;

       int numblocks_funct = numblocks - 1;

       int size_funct = dynamic_problem->GetTrueOffsetsFunc()[numblocks_funct];
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

   // 5.5 In case we solve for a particular solution of the divergence constraint,
   // we create an instance of DivConstraintSolver here, building it on
   // the dynamic problem and hierarchy
#ifdef PARTSOL_SETUP
   bool optimized_localsolvers = true;
   bool with_hcurl_smoothers = true;
   DivConstraintSolver * partsol_finder;

   partsol_finder = new DivConstraintSolver
           (*dynamic_problem, *hierarchy, optimized_localsolvers, with_hcurl_smoothers, verbose);

   bool report_funct = true;
#endif

   // 6. Creating the error estimator

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

   // Create components required for the FOSLSEstimator: bilinear form integrators and grid functions
   DefineEstimatorComponents(dynamic_problem, fosls_func_version, grfuns_descriptor, extra_grfuns, integs, verbose);

   FOSLSEstimator * estimator;
   estimator = new FOSLSEstimator(*dynamic_problem, grfuns_descriptor, NULL, integs, verbose);
   dynamic_problem->AddEstimator(*estimator);

   // ThresholdSmooRefiner is an extension of ThresholdRefiner which introduces
   // additional parameter to compute local errors using intermediate face error
   // indicators
   //ThresholdSmooRefiner refiner(*estimator, 0.0001); // 0.1, 0.001
   ThresholdRefiner refiner(*estimator);

   refiner.SetTotalErrorFraction(0.9); // 0.5

   // Some additional vector arrays for temporary storage
#ifdef PARTSOL_SETUP
   // constraint righthand sides
   Array<Vector*> div_rhs_lvls(0);
   // particular solution to the constraint, only blocks
   // related to the functional variables
   Array<BlockVector*> partsol_funct_lvls(0);
   // initial guesses for finding the particular solution
   Array<BlockVector*> initguesses_funct_lvls(0);
#endif

#ifdef APPROACH_0
   // reference (preconditioned MINRES for saddle-point systems) solutions
   Array<BlockVector*> problem_refsols_lvls(0);
#endif

   // solutions (at all levels)
   Array<BlockVector*> problem_sols_lvls(0);

   // used to store the functional value for the solution from the previous AMR iteration
   double saved_functvalue;

   // 7. The main AMR loop. At each iteration we solve the problem on the
   //     current mesh, visualize the solution, and refine the mesh.
#ifdef AMR
   const int max_dofs = 300000;//1600000; 400000;
#else // uniform refinement
   const int max_dofs = 600000;
#endif

   HYPRE_Int global_dofs = problem->GlobalTrueProblemSize();
   if (verbose)
       std::cout << "starting n_el = " << hierarchy->GetFinestParMesh()->GetNE() << "\n";

   bool compute_error = true;

   // Main loop (with AMR or uniform refinement depending on the predefined macro AMR)
   if (verbose)
       std::cout << "Running AMR ... \n";

   // upper limit on the number of AMR iterations
   int max_iter_amr = 21; // 21;

   // controls the print step of the solution into the output files (in terms of AMR iterations)
   int it_print_step = 5;
   // controls the visualization step of the solution (in terms of AMR iterations)
   int it_viz_step = 5;
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

       BlockVector saved_sol(dynamic_problem->GetTrueOffsets());
       saved_sol = 0.0;
       dynamic_problem->SolveProblem(dynamic_problem->GetRhs(), saved_sol, verbose, false);
       *problem_refsols_lvls[0] = saved_sol;

       // functional value for the reference solution at the finest level
       BlockVector reduced_problem_sol(dynamic_problem->GetTrueOffsetsFunc());
       for (int blk = 0; blk < numblocks_funct; ++blk)
           reduced_problem_sol.GetBlock(blk) = saved_sol.GetBlock(blk);

#ifdef DIVFREE_MINSOLVER
       CheckFunctValueNew(comm,*NewSolver->GetFunctOp_nobnd(0), NULL, reduced_problem_sol,
                       "for the problem solution via saddle-point system ", verbose);
#endif

       if (compute_error)
           dynamic_problem->ComputeError(saved_sol, verbose, true);
#endif

#ifdef PARTSOL_SETUP
       // future initial guess for finding the particular solution
       initguesses_funct_lvls.Prepend(new BlockVector(problem->GetTrueOffsetsFunc()));
       *initguesses_funct_lvls[0] = 0.0;

       // setting div_rhs_lvls[0] to be the constraint righthand side
       div_rhs_lvls.Prepend(new Vector(problem->GetRhs().GetBlock(numblocks - 1).Size()));
       dynamic_problem->ComputeRhsBlock(*div_rhs_lvls[0], numblocks - 1);

       // future particular solution
       partsol_funct_lvls.Prepend(new BlockVector(problem->GetTrueOffsetsFunc()));
#endif

#ifdef APPROACH_1
       // 7.1 Solving the problem at the finest level

       int l = 0;

       // initguess is zero except for the exact solutions's bdr values
       *initguesses_funct_lvls[l] = 0.0;
       dynamic_problem->SetExactBndValues(*initguesses_funct_lvls[l]);

       // Computing functional rhside which absorbs the contribution of the non-homogeneous boundary conditions
       // The reason we do this is that the DivConstraintSolver and GeneralMinConstrSolver
       // work correctly only if the boundary conditions are zero

       // we need padded vectors here which have the same number of blocks as the problem
       // but we will use only the blocks corresponding to the functional variables
       BlockVector padded_initguess(dynamic_problem->GetTrueOffsets());
       padded_initguess = 0.0;
       for (int blk = 0; blk < numblocks_funct; ++blk)
           padded_initguess.GetBlock(blk) = initguesses_funct_lvls[l]->GetBlock(blk);

       BlockVector padded_rhs(dynamic_problem->GetTrueOffsets());
       dynamic_problem->GetOp_nobnd()->Mult(padded_initguess, padded_rhs);

       padded_rhs *= -1;

       BlockVector NewRhs(dynamic_problem->GetTrueOffsetsFunc());
       NewRhs = 0.0;
       for (int blk = 0; blk < numblocks_funct; ++blk)
           NewRhs.GetBlock(blk) = padded_rhs.GetBlock(blk);
       dynamic_problem->ZeroBndValues(NewRhs);

       partsol_finder->SetFunctRhs(NewRhs);

       HypreParMatrix & Constr_l = (HypreParMatrix&)(dynamic_problem->GetOp_nobnd()->GetBlock(numblocks - 1, 0));

       // Modifying the constraint righthand side to account for the non-zero initial guess
       Vector div_initguess(Constr_l.Height());

       Constr_l.Mult(initguesses_funct_lvls[0]->GetBlock(0), div_initguess);

       *div_rhs_lvls[0] -= div_initguess;

       BlockVector zero_vec(dynamic_problem->GetTrueOffsetsFunc());
       zero_vec = 0.0;

       if (verbose)
           std::cout << "Finding a particular solution... \n";

       // Finding a correction to the initial guess as a particular solution
       partsol_finder->FindParticularSolution(zero_vec, *partsol_funct_lvls[0], *div_rhs_lvls[0], verbose, report_funct);

       // adding back the initial guess, so now partsol_funct_lvls[0] is a true particular
       // solution to the constraint
       *partsol_funct_lvls[0] += *initguesses_funct_lvls[0];

       // restoring the constraint righthand side (modified above)
       *div_rhs_lvls[0] += div_initguess;

       // functional value for the particular solution
       double starting_funct_value = CheckFunctValueNew(comm,*NewSolver->GetFunctOp_nobnd(l), NULL,
                                                     *partsol_funct_lvls[l], "for the initial guess ", verbose);

       // checking that the particular solution satisfies the constraint
       {
           dynamic_problem->ComputeBndError(*partsol_funct_lvls[0]);

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

       Vector div_partsol(Constr_l.Height());
       Constr_l.Mult(partsol_funct_lvls[0]->GetBlock(0), div_partsol);
       *div_rhs_lvls[0] -= div_partsol;

       NewSolver->SetConstrRhs(*div_rhs_lvls[0]);

       NewRhs = 0.0;

       {
           // computing rhs = - Funct_nobnd * init_guess at level l, with zero boundary conditions imposed
           BlockVector padded_initguess(dynamic_problem->GetTrueOffsets());
           padded_initguess = 0.0;
           for (int blk = 0; blk < numblocks_funct; ++blk)
               padded_initguess.GetBlock(blk) = partsol_funct_lvls[0]->GetBlock(blk);

           BlockVector padded_rhs(dynamic_problem->GetTrueOffsets());
           dynamic_problem->GetOp_nobnd()->Mult(padded_initguess, padded_rhs);

           padded_rhs *= -1;
           for (int blk = 0; blk < numblocks_funct; ++blk)
               NewRhs.GetBlock(blk) = padded_rhs.GetBlock(blk);
           dynamic_problem->ZeroBndValues(NewRhs);
       }

       NewSolver->SetFunctRhs(NewRhs);
       if (it == 0)
           NewSolver->SetBaseValue(starting_funct_value);
       else
           NewSolver->SetBaseValue(saved_functvalue);
       NewSolver->SetFunctAdditionalVector(*partsol_funct_lvls[l]);

       // solving for correction
       BlockVector correction(dynamic_problem->GetTrueOffsetsFunc());
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
           BlockVector tmp1(dynamic_problem->GetTrueOffsetsFunc());
           for (int blk = 0; blk < numblocks_funct; ++blk)
               tmp1.GetBlock(blk) = problem_sols_lvls[l]->GetBlock(blk);

           saved_functvalue = CheckFunctValueNew(comm,*NewSolver->GetFunctOp_nobnd(0), NULL, tmp1,
                           "for the finest level solution ", verbose);

           BlockVector * exactsol_proj = dynamic_problem->GetExactSolProj();
           BlockVector tmp2(dynamic_problem->GetTrueOffsetsFunc());
           for (int blk = 0; blk < numblocks_funct; ++blk)
               tmp2.GetBlock(blk) = exactsol_proj->GetBlock(blk);

           CheckFunctValueNew(comm,*NewSolver->GetFunctOp_nobnd(0), NULL, tmp2,
                           "for the projection of the exact solution ", verbose);

           delete exactsol_proj;
       }
#endif

       // all three approaches below are implemented in the same loop but with different parameters
#if defined(APPROACH_2) || defined(APPROACH_3) || defined (APPROACH_3_2)

       // recoarsening constraint rhsides from finest to coarsest level
       // recoarsened rhsides are used when we are computing particular solutions
       // on any of the previous levels
       for (int l = 1; l < div_rhs_lvls.Size(); ++l)
           hierarchy->GetTruePspace(SpaceName::L2,l - 1)->MultTranspose(*div_rhs_lvls[l-1], *div_rhs_lvls[l]);

       // 7.1 Re-solving all the problems with (coarsened) rhs, from coarsest to finest
       // and using the previous soluition as a starting guess
#ifdef RECOARSENING_AMR
       int coarsest_lvl; // coarsest level to be considered

#ifdef APPROACH_2
       coarsest_lvl = 0;
#endif

#ifdef APPROACH_3_2
       // if we have more tha one level, we step one level back, otherwise there is nowhere to go
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

           // Computing initial guess

           *initguesses_funct_lvls[l] = 0.0;

#ifdef CLEVER_STARTING_PARTSOL
           // create a better initial guess by taking the interpolant of the previous solution
           // which would be available if we consider levels finer than the coarsest
           if (l < coarsest_lvl)
           {
               for (int blk = 0; blk < numblocks_funct; ++blk)
               {
                   hierarchy->GetTruePspace( (*space_names_funct)[blk], l)->Mult
                       (problem_sols_lvls[l + 1]->GetBlock(blk), initguesses_funct_lvls[l]->GetBlock(blk));
               }
           }
#endif
           // setting correct bdr values for the initial guess
           problem_l->SetExactBndValues(*initguesses_funct_lvls[l]);

           // Computing funct rhside which absorbs the contribution of the non-homogeneous boundary conditions
           // The reason we do this is that the DivConstraintSolver and GeneralMinConstrSolver
           // work correctly only if the boundary conditions are zero

           // Here we have to create padded vectors which have the same number of blocks as the problem
           // although we actually use only blocks corresponding to the functional (not to the constraint)
           BlockVector padded_initguess(problem_l->GetTrueOffsets());
           padded_initguess = 0.0;
           for (int blk = 0; blk < numblocks_funct; ++blk)
               padded_initguess.GetBlock(blk) = initguesses_funct_lvls[l]->GetBlock(blk);

           BlockVector padded_rhs(problem_l->GetTrueOffsets());
           problem_l->GetOp_nobnd()->Mult(padded_initguess, padded_rhs);

           padded_rhs *= -1;

           BlockVector zero_vec(problem_l->GetTrueOffsetsFunc());
           zero_vec = 0.0;

           // Since we want to absorb the initial guess in the rhs of the solver,
           // we will have to move its contribution to the rhs corresponding to the functional
           BlockVector NewRhs(problem_l->GetTrueOffsetsFunc());
           NewRhs = 0.0;
           for (int blk = 0; blk < numblocks_funct; ++blk)
               NewRhs.GetBlock(blk) = padded_rhs.GetBlock(blk);
           problem_l->ZeroBndValues(NewRhs);

           partsol_finder->SetFunctRhs(NewRhs);

           // Modifying the constraint rhs for the particular solution to be found,
           // because we have initial guess with nonzero boundary conditions
           HypreParMatrix & Constr_l = (HypreParMatrix&)(problem_l->GetOp_nobnd()->GetBlock(numblocks - 1, 0));

           Vector div_initguess(Constr_l.Height());
           Constr_l.Mult(initguesses_funct_lvls[l]->GetBlock(0), div_initguess);

           // we subtract contribution of the initial guess to the constraint, but we will add it back
           // after we find the particular solution
           *div_rhs_lvls[l] -= div_initguess;

           if (verbose)
               std::cout << "Finding a particular solution... \n";

           // The reason we check the functional value is that in that the algorithm should never increase
           // the functional value (of the approximation, with initial guess absorbed)
           // If it doesn't, this is an indication of incorrect behavior

           // functional value for the initial guess for particular solution
           CheckFunctValueNew(comm,*NewSolver->GetFunctOp_nobnd(l), NULL, *initguesses_funct_lvls[l],
                           "for the particular solution ", verbose);


           // Finding the particular solution, or a correction to it (without initial guess)
           partsol_finder->FindParticularSolution(l, Constr_l, zero_vec, *partsol_funct_lvls[l],
                                                  *div_rhs_lvls[l], verbose, report_funct);

           // adding the initial guess, so that now partsol_funct_lvls[l] is a true particular solution
           *partsol_funct_lvls[l] += *initguesses_funct_lvls[l];

           // restoring the constraint rhs (modified above)
           *div_rhs_lvls[l] += div_initguess;

           // checking whether the particular solution satisfies the constraint
           {
               problem_l->ComputeBndError(*partsol_funct_lvls[l]);

               HypreParMatrix & Constr = (HypreParMatrix&)(problem_l->GetOp_nobnd()->GetBlock(numblocks - 1, 0));
               Vector tempc(Constr.Height());
               Constr.Mult(partsol_funct_lvls[l]->GetBlock(0), tempc);
               tempc -= *div_rhs_lvls[l];
               double res_constr_norm = ComputeMPIVecNorm(comm, tempc, "", false);
               MFEM_ASSERT (res_constr_norm < 1.0e-10, "");
           }

           // functional value for the particular solution
           double starting_funct_value =
                   CheckFunctValueNew(comm,*NewSolver->GetFunctOp_nobnd(l), NULL, *partsol_funct_lvls[l],
                                      "for the particular solution ", verbose);

           // Now we are going to setup the minimization solver, which will take particular solution
           // as starting guess and hence work in the divergence-free space (implicitly)

           // But again, as in the case for particular solution, we need to move the contribution of
           // particular solution to the rhs, and solve for a correction
           zero_vec = 0.0;
           NewSolver->SetInitialGuess(l, zero_vec);
           //NewSolver->SetConstrRhs(*div_rhs_lvls[l]);

           //if (verbose)
               //NewSolver->PrintAllOptions();

           NewRhs = 0.0;

           // Computing modified rhs for the functional part of the problem
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

           // Setting the stopping criteria tolerance
           // Taking into account that we might be solving (after re-using previous levels) a problem
           // with smaller starting functional value, we set the base value (value, with which the functional
           // value is compared to at any iteration to check if w should stop) accordingly below:

           // If we don't have a finer level, then we just take as the base value (see SetBaseValue())
           // the functional value for the particular solution
           if (l == coarsest_lvl && it == 0)
               NewSolver->SetBaseValue(starting_funct_value);
           else// otherwise, we use the functional value for the solution from the previous iteration
               // which is a good approximation of the functional value for the solution, since
               // it doesn't change in the test much from iteration to iteration (not with orders of magnitude)
               NewSolver->SetBaseValue(saved_functvalue);

           // This one makes it possible for the solver to monitor the true functional value, taking
           // into account the particular solution (so we add partsol_funct_lvls[l] to the correction
           // which is computed at every iteration of the solver)
           NewSolver->SetFunctAdditionalVector(*partsol_funct_lvls[l]);

           // Solving for correction to the particular solution, which will minimize the functional as well
           BlockVector correction(problem_l->GetTrueOffsetsFunc());
           correction = 0.0;

           NewSolver->SetPrintLevel(1);

           if (verbose && l == 0)
               std::cout << "Solving the finest level problem... \n";

           NewSolver->Mult(l, &Constr_l, NewRhs, correction);

           for (int blk = 0; blk < numblocks_funct; ++blk)
           {
               problem_sols_lvls[l]->GetBlock(blk) = partsol_funct_lvls[l]->GetBlock(blk);
               problem_sols_lvls[l]->GetBlock(blk) += correction.GetBlock(blk);
           }

           // Saving the functional value at the finest level, and reporting the functional value
           // for the projection of the exact solution. Since the projection is not w.r.t to the energy
           // scalar product defined by the functional, our computed solution is better (gives smaller
           // functional value) than the projection
           if (l == 0)
           {
               BlockVector tmp1(problem_l->GetTrueOffsetsFunc());
               for (int blk = 0; blk < numblocks_funct; ++blk)
                   tmp1.GetBlock(blk) = problem_sols_lvls[l]->GetBlock(blk);

               saved_functvalue = CheckFunctValueNew(comm,*NewSolver->GetFunctOp_nobnd(0), NULL, tmp1,
                               "for the finest level solution ", verbose);

               BlockVector * exactsol_proj = problem_l->GetExactSolProj();
               BlockVector tmp2(problem_l->GetTrueOffsetsFunc());
               for (int blk = 0; blk < numblocks_funct; ++blk)
                   tmp2.GetBlock(blk) = exactsol_proj->GetBlock(blk);

               CheckFunctValueNew(comm,*NewSolver->GetFunctOp_nobnd(0), NULL, tmp2,
                               "for the projection of the exact solution ", verbose);

               delete exactsol_proj;
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
           dynamic_problem->ComputeError(*problem_sols_lvls[0], verbose, true);

       // to make sure that problem has grfuns which correspond to the computed problem solution
       // we explicitly distribute the solution here
       // though for now the coordination already happens in ComputeError()
       dynamic_problem->DistributeToGrfuns(*problem_sols_lvls[0]);

       // 7.2 (optional) Send the solution by socket to a GLVis server.
       if (visualization && (it % it_viz_step == 0 || it == max_iter_amr - 1)) //&& it % 4 == 0 ) //it == max_iter_amr - 1)
       {
           int ne = pmesh->GetNE();
           for (int elind = 0; elind < ne; ++elind)
               pmesh->SetAttribute(elind, elind);
           ParGridFunction * sigma = dynamic_problem->GetGrFun(0);
           ParGridFunction * S;

#ifdef HDIVL2L2
           S = dynamic_problem->GetGrFun(1);
#else
           if (dynamic_problem->GetFEformulation().Nunknowns() >= 2)
               S = dynamic_problem->GetGrFun(1);
           else // only sigma = Hdiv-L2 formulation with eliminated S
               S = (dynamic_cast<ProblemType*>(dynamic_problem))->RecoverS(problem_sols_lvls[0]->GetBlock(0));
#endif

           char vishost[] = "localhost";
           int  visport   = 19916;

           socketstream sigma_sock(vishost, visport);
           sigma_sock << "parallel " << num_procs << " " << myid << "\n";
           sigma_sock << "solution\n" << *pmesh << *sigma << "window_title 'sigma, AMR iter No."
                  << it <<"'" << flush;

           ParGridFunction * sigma_ex = new ParGridFunction(dynamic_problem->GetPfes(0));
           BlockVector * exactsol_proj = dynamic_problem->GetExactSolProj();
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

           if (!(dynamic_problem->GetFEformulation().Nunknowns() >= 2))
               delete S;

       }

       // 7.3 (optional) Printing the solution ino files with VTK format
       // which can then be visualized be Paraview (hand-made) scripts
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

           ParGridFunction * sigma = dynamic_problem->GetGrFun(0);
           ParGridFunction * S;

#ifdef HDIVL2L2
           S = dynamic_problem->GetGrFun(1);
#else
           if (dynamic_problem->GetFEformulation().Nunknowns() >= 2)
               S = dynamic_problem->GetGrFun(1);
           else // only sigma = Hdiv-L2 formulation with eliminated S
               S = (dynamic_cast<ProblemType*>(dynamic_problem))->RecoverS(problem_sols_lvls[0]->GetBlock(0));
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

#ifndef HDIVL2L2
           if (dynamic_problem->GetFEformulation().Nunknowns() < 2)
               delete S;
#endif
       }

       // 7.4 Refine the mesh, either uniformly (if AMR was not #defined) or adaptively,
       // using the estimator-refiner
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

       // Visualizing the local errors
       if (visualization && (it % 5 == 0 || it == max_iter_amr - 1) ) //it == max_iter_amr - 1)
       {
           const Vector& local_errors = estimator->GetLastLocalErrors();
           if (feorder == 0)
               MFEM_ASSERT(local_errors.Size() == dynamic_problem->GetPfes(numblocks_funct)->TrueVSize(), "");

           FiniteElementCollection * l2_coll;
           if (feorder > 0)
               l2_coll = new L2_FECollection(0, dim);

           ParFiniteElementSpace * L2_space;
           if (feorder == 0)
               L2_space = dynamic_problem->GetPfes(numblocks_funct);
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

       // 7.5 After the refinement, updating the hierarchy and everything else
       bool recoarsen = true;
       // this also updates the underlying hierarchy
       prob_hierarchy->Update(recoarsen);

       problem = prob_hierarchy->GetProblem(0);

       // Since the hierarchy was updated, we need to rebuild the problem
       dynamic_problem->BuildSystem(verbose);

#ifdef DIVFREE_MINSOLVER
       mgtools_hierarchy->Update(recoarsen);
       NewSolver->UpdateProblem(*dynamic_problem);
       NewSolver->Update(recoarsen);
#endif

#ifdef PARTSOL_SETUP
       partsol_finder->UpdateProblem(*dynamic_problem);
       partsol_finder->Update(recoarsen);
#endif

       // checking #dofs after the refinement
       global_dofs = dynamic_problem->GlobalTrueProblemSize();

       if (global_dofs > max_dofs)
       {
          if (verbose)
             cout << "Reached the maximum number of dofs. Stop. \n";
          break;
       }

   } // end of the main AMR loop

   // 8. Deallocating memory
#ifdef DIVFREE_MINSOLVER
   delete NewSolver;
   delete mgtools_hierarchy;
   delete descriptor;
#endif

#ifdef PARTSOL_SETUP
   delete partsol_finder;
#endif

#ifdef PARTSOL_SETUP
   for (int i = 0; i < div_rhs_lvls.Size(); ++i)
       delete div_rhs_lvls[i];
   for (int i = 0; i < partsol_funct_lvls.Size(); ++i)
       delete partsol_funct_lvls[i];
   for (int i = 0; i < initguesses_funct_lvls.Size(); ++i)
       delete initguesses_funct_lvls[i];
#endif

#ifdef APPROACH_0
   for (int i = 0; i < problem_refsols_lvls.Size(); ++i)
       delete problem_refsols_lvls[i];
#endif

   for (int i = 0; i < problem_sols_lvls.Size(); ++i)
       delete problem_sols_lvls[i];
   for (int i = 0; i < formulat->Nblocks(); ++i)
       delete bdr_attribs_data[i];
   delete hierarchy;
   delete prob_hierarchy;

   for (int i = 0; i < extra_grfuns.Size(); ++i)
       if (extra_grfuns[i])
           delete extra_grfuns[i];
   for (int i = 0; i < integs.NumRows(); ++i)
       for (int j = 0; j < integs.NumCols(); ++j)
           if (integs(i,j))
               delete integs(i,j);

   delete dynamic_problem;
   delete estimator;

   delete bdr_conds;
   delete formulat;
   delete fe_formulat;

   MPI_Finalize();
   return 0;
}

// See it's declaration
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

// See it's declaration
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

#ifdef APPROACH_0
    if (verbose)
        std::cout << "APPROACH_0 active \n";
#else
    if (verbose)
        std::cout << "APPROACH_0 passive \n";
#endif

#ifdef APPROACH_1
    if (verbose)
        std::cout << "APPROACH_1 active \n";
#else
    if (verbose)
        std::cout << "APPROACH_1 passive \n";
#endif

#ifdef APPROACH_2
    if (verbose)
        std::cout << "APPROACH_2 active \n";
#else
    if (verbose)
        std::cout << "APPROACH_2 passive \n";
#endif

#ifdef APPROACH_3
    if (verbose)
        std::cout << "APPROACH_3 active \n";
#else
    if (verbose)
        std::cout << "APPROACH_3 passive \n";
#endif

#ifdef PARTSOL_SETUP
    if (verbose)
        std::cout << "PARTSOL_SETUP active \n";
#else
    if (verbose)
        std::cout << "PARTSOL_SETUP passive \n";
#endif

#if defined(PARTSOL_SETUP) && (!defined(DIVFREE_MINSOLVER))
    MFEM_ABORT("For PARTSOL_SETUP one of the divfree options must be active");
#endif

#ifdef DIVFREE_MINSOLVER
    if (verbose)
        std::cout << "DIVFREE_MINSOLVER active \n";
#else
    if (verbose)
        std::cout << "DIVFREE_MINSOLVER passive \n";
#endif

#if defined(CLEVER_STARTING_PARTSOL) && !defined(PARTSOL_SETUP)
    MFEM_ABORT("CLEVER_STARTING_PARTSOL cannot be active if PARTSOL_SETUP is not \n");
#endif

#ifdef CLEVER_STARTING_PARTSOL
    if (verbose)
        std::cout << "CLEVER_STARTING_PARTSOL active \n";
#else
    if (verbose)
        std::cout << "CLEVER_STARTING_PARTSOL passive \n";
#endif

#ifdef CYLINDER_CUBE_TEST
    if (verbose)
        std::cout << "CYLINDER_CUBE_TEST active \n";
#else
    if (verbose)
        std::cout << "CYLINDER_CUBE_TEST passive \n";
#endif

#ifdef OVERCONSTRAINED
    if (verbose)
        std::cout << "OVERCONSTRAINED active \n";
#else
    if (verbose)
        std::cout << "OVERCONSTRAINED passive \n";
#endif


#ifdef RECOARSENING_AMR
    if (verbose)
        std::cout << "RECOARSENING_AMR active \n";
#else
    if (verbose)
        std::cout << "RECOARSENING_AMR passive \n";
#endif
}

// See it's declaration and comments inside
void ReArrangeBdrAttributes(Mesh* mesh4cube)
{
    Mesh * mesh = mesh4cube;

    int dim = mesh->Dimension();

    int nbe = mesh->GetNBE();

    // Assume we have [-1,1] x [-1,1] x [0,2]
    // cube faces:
    // 0: bottom (z = 0)
    // 1: x = -1
    // 2: y = -1
    // 3: x = 1
    // 4: y = 1
    // 5: z = 2
    // final boundary parts (=attributes):
    // 1: bottom cube face
    // lateral boundary parts at the square corners:
    // 2: (face == 1 && y < 0) || (face == 2 && x < 0), here face = cube face
    // 3: (face == 2 && x >=0) || (face == 3 && y < 0)
    // 4: (face == 3 && y >=0) || (face == 4 && x >=0)
    // 5: (face == 4 && x < 0) || (face == 1 && y >=0)
    // and
    // 6: top cube face
    for (int beind = 0; beind < nbe; ++beind)
    {
        //std::cout << "beind = " << beind << "\n";
        // determine which cube face the be belongs to, via vertex coordinate dispersion
        int cubeface = -1;
        Element * bel = mesh->GetBdrElement(beind);
        int be_nv = bel->GetNVertices();
        Array<int> vinds(be_nv);
        bel->GetVertices(vinds);
        //vinds.Print();

        //std::cout << "mesh nv =  " << pmesh->GetNV() << "\n";

        Array<double> av_coords(dim);
        av_coords = 0.0;
        for (int vno = 0; vno < be_nv; ++vno)
        {
            //std::cout << "vinds[vno] = " << vinds[vno] << "\n";
            double * vcoos = mesh->GetVertex(vinds[vno]);
            for (int coo = 0; coo < dim; ++coo)
            {
                //std::cout << vcoos[coo] << " ";
                av_coords[coo] += vcoos[coo];
            }
            //std::cout << "\n";
        }
        //av_coords.Print();

        for (int coo = 0; coo < dim; ++coo)
        {
            av_coords[coo] /= 1.0 * be_nv;
        }

        //std::cout << "average be coordinates: \n";
        //av_coords.Print();

        int face_coo = -1;
        for (int coo = 0; coo < dim; ++coo)
        {
            bool coo_fixed = true;
            //std::cout << "coo = " << coo << "\n";
            for (int vno = 0; vno < be_nv; ++vno)
            {
                double * vcoos = mesh->GetVertex(vinds[vno]);
                //std::cout << "vcoos[coo] = " << vcoos[coo] << "\n";
                if (fabs(vcoos[coo] - av_coords[coo]) > 1.0e-13)
                    coo_fixed = false;
            }
            if (coo_fixed)
            {
                if (face_coo > -1)
                {
                    MFEM_ABORT("Found a second coordinate which is fixed \n");
                }
                else
                {
                    face_coo = coo;
                }
            }
        }

        MFEM_ASSERT(face_coo != -1,"Didn't find a fixed coordinate \n");

        double value = av_coords[face_coo];
        if (face_coo == 0 && fabs(value - (-1.0)) < 1.0e-13)
            cubeface = 1;
        if (face_coo== 1 && fabs(value - (-1.0)) < 1.0e-13)
            cubeface = 2;
        if (face_coo == 2 && fabs(value - (0.0)) < 1.0e-13)
            cubeface = 0;
        if (face_coo == 0 && fabs(value - (1.0)) < 1.0e-13)
            cubeface = 3;
        if (face_coo == 1 && fabs(value - (1.0)) < 1.0e-13)
            cubeface = 4;
        if (face_coo == 2 && fabs(value - (2.0)) < 1.0e-13)
            cubeface = 5;

        //std::cout << "cubeface = " << cubeface << "\n";

        // determine to which vertical stripe of the cube face the be belongs to
        Array<int> signs(dim);
        int x,y;
        if (cubeface != 0 && cubeface != 5)
        {
            for (int coo = 0; coo < dim; ++coo)
            {
                Array<int> coo_signs(be_nv);
                for (int vno = 0; vno < be_nv; ++vno)
                {
                    double * vcoos = mesh->GetVertex(vinds[vno]);
                    if (vcoos[coo] - 0.0 > 1.0e-13)
                        coo_signs[vno] = 1;
                    else if (vcoos[coo] - 0.0 < -1.0e-13)
                        coo_signs[vno] = -1;
                    else
                        coo_signs[vno] = 0;
                }

                int first_sign = -2; // anything which is not -1, 0 or 1
                for (int i = 0; i < be_nv; ++i)
                {
                    if (abs(coo_signs[i]) > 0)
                    {
                        first_sign = coo_signs[i];
                        break;
                    }
                }

                MFEM_ASSERT(first_sign != -2, "All signs were 0 for this coordinate, i.e. triangle"
                                              " is on the line");

                for (int i = 0; i < be_nv; ++i)
                {
                    if (coo_signs[i] != 0 && coo_signs[i] != first_sign)
                    {
                        MFEM_ABORT("Boundary element intersects the face middle line");
                    }
                }

                signs[coo] = first_sign;
            }
            x = signs[0];
            y = signs[1];
        }

        switch(cubeface)
        {
        case 0:
            bel->SetAttribute(1);
            break;
        case 1:
        {
            if (y < 0)
                bel->SetAttribute(2);
            else
                bel->SetAttribute(5);
        }
            break;
        case 2:
            if (x < 0)
                bel->SetAttribute(2);
            else
                bel->SetAttribute(3);
            break;
        case 3:
            if (y < 0)
                bel->SetAttribute(3);
            else
                bel->SetAttribute(4);
            break;
        case 4:
            if (x < 0)
                bel->SetAttribute(5);
            else
                bel->SetAttribute(4);
            break;
        case 5:
            bel->SetAttribute(6);
            break;
        default:
        {
            MFEM_ABORT("Could not find a cube face for the boundary element \n");
        }
            break;
        }
    } // end of loop ove be elements

}
