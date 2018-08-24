///
///                           MFEM(with 4D elements) CFOSLS for 3D/4D laplace equation
///                                      with standard adaptive refinement,
///                                       solved by a  minimization solver.
///
/// The problem considered in this example is
///                             laplace(u) = f
/// (either 3D or 4D, calling one of the variables time in space-time)
///
/// casted in the CFOSLS formulation in Hdiv-H1-L2:
///                             || sigma - b * u || ^2 -> min
/// where sigma is from H(div) and u is from H^1
/// minimizing the functional under the constraint
///                             div sigma = f.
///
/// The current 3D test is a regular solution in a cube.
///
/// The problem is discretized using RT, linear Lagrange and discontinuous constants in 3D/4D.
///
/// The problem is then solved with adaptive mesh refinement (AMR).
///
/// This example demonstrates usage of AMR related classes from mfem/cfosls/, such as
/// FOSLSProblem, GeneralHierarchy, MultigridToolsHierarchy, DivConstraintSolver,
/// GenMinConstrSolver, FOSLSEstimator, etc.
///
/// (**) This code was tested only in serial.
/// (***) The example was tested for memory leaks with valgrind, in 3D.
///
/// Typical run of this example: ./cfosls_laplace_adref_Hcurl_new --whichD 3 -no-vis
/// If you ant Hdiv-H1-L2 formulation, you will need not only change --spaceS option but also
/// change the source code, around 4.
///
/// Other examples on adaptive mesh refinement are cfosls_hyperbolic_adref.cpp,
/// cfosls_laplace_adref_Hcurl.cpp and cfosls_hyperbolic_adref_Hcurl_new.cpp.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>

// if not active, the mesh is simply uniformly refined at each iteration
#define AMR

// makes the algorithm go over the previous levels (meshes from the previous iterations)
// possibly improving the overall iteration counts by re-using information from the previous
// meshes
//#define RECOARSENING_AMR

// activates using the solution at the previous mesh as a starting guess for the next problem
// combined with RECOARSENING_AMR this leads to a variant of cascadic MG
#define CLEVER_STARTING_GUESS

// if active, leads to using the classical FEM formulation for Laplace equation with
// one scalar unknown from H1 rather than CFOSLS Hdiv-H1-L2 formulation (or Hdiv-L2
// from mixed FEM)
//#define H1FEMLAPLACE

// Here the problem is solved by a preconditioned MINRES, and its solution is
// used as a reference solution
#define REFERENCE_SOLUTION

using namespace std;
using namespace mfem;
using std::shared_ptr;
using std::make_shared;

int main(int argc, char *argv[])
{
    // 1. Initialize MPI
    int num_procs, myid;
    bool visualization = 1;

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    bool verbose = (myid == 0);

    int nDimensions     = 3;
    int numsol          = -3;

    int ser_ref_levels  = 1;
    int par_ref_levels  = 1;

    // This must be consistent with what formulation is used below.
    // Search for "using FormulType" below
    const char *space_for_S = "H1";     // "H1" or "L2"
    const char *space_for_sigma = "Hdiv"; // "Hdiv" or "H1"

    // solver options
    int prec_option = 1; //defines whether to use preconditioner or not, and which one

    // also might be changed after command-line arguments parsing
    const char *mesh_file = "../data/cube_3d_moderate.mesh";
    //mesh_file = "../data/netgen_cylinder_mesh_0.1to0.2.mesh";
    //mesh_file = "../data/pmesh_cylinder_moderate_0.2.mesh";
    //mesh_file = "../data/pmesh_cylinder_fine_0.1.mesh";
    mesh_file = "../data/cube_3d_moderate.mesh";

    int feorder         = 0;
#ifdef H1FEMLAPLACE
    ++feorder;
#endif

    if (verbose)
        cout << "Solving the Laplace equation in AMR setting with a minimization solver \n";

    // 2. Parse command-line options.

    OptionsParser args(argc, argv);
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
    }

    if (numsol == 11)
    {
        //mesh_file = "../data/netgen_lshape3D_onemoretry.netgen";
        mesh_file = "../data/netgen_lshape3D_onemoretry_coarsest.netgen";
    }

    if (verbose)
        std::cout << "For the records: numsol = " << numsol
                  << ", mesh_file = " << mesh_file << "\n";
    if (verbose)
        std::cout << "Number of mpi processes: " << num_procs << "\n";

    // 3. Define the problem to be solved (CFOSLS Hdiv-L2 or Hdiv-H1 formulation, e.g.)

#ifdef H1FEMLAPLACE
    MFEM_ASSERT(strcmp(space_for_S,"H1") == 0, "Classical FEM formulation for Laplace must "
                                               "have space_for_S = `H1` \n");
    using FormulType = FOSLSFormulation_Laplace;
    using FEFormulType = FOSLSFEFormulation_Laplace;
    using BdrCondsType = BdrConditions_Laplace;
    using ProblemType = FOSLSProblem_Laplace;
#else
    // Hdiv-H1 fomulation
    MFEM_ASSERT(strcmp(space_for_S,"H1") == 0, "Hdiv-H1-L2 CFOSLS formulation for Laplace must "
                                               "have space_for_S = `H1` \n");
    using FormulType = CFOSLSFormulation_Laplace;
    using FEFormulType = CFOSLSFEFormulation_HdivH1L2_Laplace;
    using BdrCondsType = BdrConditions_CFOSLS_HdivH1Laplace;
    using ProblemType = FOSLSProblem_HdivH1lapl;
#endif

    /*
    // for mixed formulation of Laplace equation, Hdiv-L2
    MFEM_ASSERT(strcmp(space_for_S,"L2") == 0, "Mixed formulation must have space_for_S = `L2` \n");
    using FormulType = CFOSLSFormulation_MixedLaplace;
    using FEFormulType = CFOSLSFEFormulation_MixedLaplace;
    using BdrCondsType = BdrConditions_MixedLaplace;
    using ProblemType = FOSLSProblem_MixedLaplace;
    */

    // Reporting which macros have been defined in the current configuration

#ifdef AMR
    if (verbose)
        std::cout << "AMR active \n";
#else
    if (verbose)
        std::cout << "AMR passive \n";
#endif

#ifdef RECOARSENING_AMR
    if (verbose)
        std::cout << "RECOARSENING_AMR active \n";
#else
    if (verbose)
        std::cout << "RECOARSENING_AMR passive \n";
#endif

#ifdef REFERENCE_SOLUTION
    if (verbose)
        std::cout << "REFERENCE_SOLUTION active \n";
#else
    if (verbose)
        std::cout << "REFERENCE_SOLUTION passive \n";
#endif

#ifdef CLEVER_STARTING_GUESS
    if (verbose)
        std::cout << "CLEVER_STARTING_GUESS active \n";
#else
    if (verbose)
        std::cout << "CLEVER_STARTING_GUESS passive \n";
#endif

    MFEM_ASSERT(strcmp(space_for_S,"H1") == 0 || strcmp(space_for_S,"L2") == 0,
                "Space for S must be H1 or L2!\n");
    MFEM_ASSERT(strcmp(space_for_sigma,"Hdiv") == 0 || strcmp(space_for_sigma,"H1") == 0,
                "Space for sigma must be Hdiv or H1!\n");

    MFEM_ASSERT(!strcmp(space_for_sigma,"H1") == 0 || (strcmp(space_for_sigma,"H1") == 0 &&
                                                       strcmp(space_for_S,"H1") == 0),
                "Sigma from H1vec must be coupled with S from H1!\n");

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

    int dim = nDimensions;

    pmesh->PrintInfo(std::cout); if(verbose) cout << endl;

    // 5. Creating an instance of the FOSLSProblem to be solved.

    // Define how many blocks there will be in the problem (loose way, correct one
    // would be to call Nblocks() for formulat which is defined below)
    int numblocks = 1;
    if (strcmp(space_for_S,"H1") == 0)
        numblocks++;
    numblocks++;

    if (verbose)
        std::cout << "Number of blocks in the formulation: " << numblocks << "\n";

   FormulType * formulat = new FormulType (dim, numsol, verbose);
   FEFormulType * fe_formulat = new FEFormulType(*formulat, feorder);
   BdrConditions * bdr_conds = new BdrCondsType(*pmesh);

   // 5.1. Creating a hierarchy of meshes and problems at those meshes
   bool with_hcurl = true;
   GeneralHierarchy * hierarchy = new GeneralHierarchy(1, *pmesh, feorder, verbose, with_hcurl);
   hierarchy->ConstructDofTrueDofs();
   hierarchy->ConstructDivfreeDops();
   hierarchy->ConstructEl2Dofs();

   FOSLSProblHierarchy<ProblemType, GeneralHierarchy> * prob_hierarchy = new
           FOSLSProblHierarchy<ProblemType, GeneralHierarchy>
           (*hierarchy, 1, *bdr_conds, *fe_formulat, prec_option, verbose);

   ProblemType * problem = prob_hierarchy->GetProblem(0);

   const Array<SpaceName>* space_names_funct = problem->GetFEformulation().GetFormulation()->
           GetFunctSpacesDescriptor();

   // 5.2 Creating a "dynamic" problem which always lives on the finest level of the hierarchy
   // and a DivConstraintSolver used for finding the particular solution
   FOSLSProblem* problem_mgtools = hierarchy->BuildDynamicProblem<ProblemType>
           (*bdr_conds, *fe_formulat, prec_option, verbose);
   hierarchy->AttachProblem(problem_mgtools);

   bool optimized_localsolvers = true;
   bool with_hcurl_smoothers = true;
   DivConstraintSolver * partsol_finder;

   partsol_finder = new DivConstraintSolver
           (*problem_mgtools, *hierarchy, optimized_localsolvers, with_hcurl_smoothers, verbose);

   bool report_funct = true;

   // 5.3. Creating multigrid tools hierarchy which is used to define GeneralMinConstrSolver
   // which is the minimization solver used for solving the problem
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

       double newsolver_reltol = 1.0e-6;

       if (verbose)
           std::cout << "newsolver_reltol = " << newsolver_reltol << "\n";

       NewSolver->SetRelTol(newsolver_reltol);
       NewSolver->SetMaxIter(200);
       NewSolver->SetPrintLevel(0);
       NewSolver->SetStopCriteriaType(stopcriteria_type);
   }

   // 6. Creating the error estimator

   Laplace_test* Mytest = dynamic_cast<Laplace_test*>
           (problem->GetFEformulation().GetFormulation()->GetTest());
   MFEM_ASSERT(Mytest, "Unsuccessful cast into Hyper_test* \n");

   int numfoslsfuns = -1;

   int fosls_func_version = 1;
   if (verbose)
    std::cout << "fosls_func_version = " << fosls_func_version << "\n";

   if (fosls_func_version == 1)
   {
#ifdef H1FEMLAPLACE
       numfoslsfuns = 1;
#else
       numfoslsfuns = 1;
       if (strcmp(space_for_S,"H1") == 0)
           ++numfoslsfuns;
#endif
   }

#ifdef H1FEMLAPLACE
   int numblocks_funct = 1;
#else
   int numblocks_funct = 1;
   if (strcmp(space_for_S,"H1") == 0)
       ++numblocks_funct;
#endif

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
   std::vector<std::pair<int,int> > grfuns_descriptor(numfoslsfuns);

   Array2D<BilinearFormIntegrator *> integs(numfoslsfuns, numfoslsfuns);
   for (int i = 0; i < integs.NumRows(); ++i)
       for (int j = 0; j < integs.NumCols(); ++j)
           integs(i,j) = NULL;

   // version 1, only || sigma + grad S ||^2, or || sigma ||^2
   if (fosls_func_version == 1)
   {
#ifdef H1FEMLAPLACE
       // this works
       grfuns_descriptor[0] = std::make_pair<int,int>(1, 0);
       integs(0,0) = new DiffusionIntegrator;
#else
       // this works
       grfuns_descriptor[0] = std::make_pair<int,int>(1, 0);
       integs(0,0) = new VectorFEMassIntegrator;

       if (strcmp(space_for_S,"H1") == 0)
       {
           grfuns_descriptor[1] = std::make_pair<int,int>(1, 1);
           integs(1,1) = new DiffusionIntegrator;
           integs(0,1) = new MixedVectorGradientIntegrator;
       }
#endif
   }
   else
   {
       MFEM_ABORT("Unsupported version of fosls functional \n");
   }

   FOSLSEstimator * estimator;
   estimator = new FOSLSEstimator(*problem_mgtools, grfuns_descriptor, NULL, integs, verbose);
   problem_mgtools->AddEstimator(*estimator);

   //ThresholdRefiner refiner(*estimator);
   ThresholdSmooRefiner refiner(*estimator); // 0.1, 0.001
   refiner.SetTotalErrorFraction(0.95); // 0.5

   if (verbose)
       std::cout << "beta for the face marking strategy: " << refiner.Beta() << "\n";

   // constraint righthand sides
   Array<Vector*> div_rhs_lvls(0);
   // particular solution to the constraint, only blocks
   // related to the functional variables
   Array<BlockVector*> partsol_funct_lvls(0);
   // initial guesses for finding the particular solution
   Array<BlockVector*> initguesses_funct_lvls(0);

   // solutions (at all levels)
   Array<BlockVector*> problem_sols_lvls(0);

#ifdef REFERENCE_SOLUTION
   // reference (preconditioned MINRES for saddle-point systems) solutions
   Array<BlockVector*> problem_refsols_lvls(0);
#endif

   double saved_functvalue;

   // 7. The main AMR loop. At each iteration we solve the problem on the
   //     current mesh, visualize the solution, and refine the mesh.
#ifdef AMR
   const int max_dofs = 200000;//1600000;
#else // uniform refinement
   const int max_dofs = 600000;
#endif

   HYPRE_Int global_dofs = problem->GlobalTrueProblemSize();
   std::cout << "starting n_el = " << problem_mgtools->GetParMesh()->GetNE() << "\n";

   // Main loop (with AMR or uniform refinement depending on the predefined macros)
   int max_iter_amr = 3;
   for (int it = 0; it < max_iter_amr; it++)
   {
       if (verbose)
       {
          cout << "\nAMR iteration " << it << "\n";
          cout << "Number of unknowns: " << global_dofs << "\n\n";
       }

       bool compute_error = true;

       initguesses_funct_lvls.Prepend(new BlockVector(problem->GetTrueOffsetsFunc()));
       *initguesses_funct_lvls[0] = 0.0;

       // future particular solution
       partsol_funct_lvls.Prepend(new BlockVector(problem->GetTrueOffsetsFunc()));

       problem_sols_lvls.Prepend(new BlockVector(problem->GetTrueOffsets()));
       *problem_sols_lvls[0] = 0.0;

#ifdef REFERENCE_SOLUTION
       problem_refsols_lvls.Prepend(new BlockVector(problem->GetTrueOffsets()));
       *problem_refsols_lvls[0] = 0.0;
#endif

       div_rhs_lvls.Prepend(new Vector(problem_mgtools->GetRhs().GetBlock(numblocks - 1).Size()));
       problem_mgtools->ComputeRhsBlock(*div_rhs_lvls[0], numblocks - 1);

#ifdef REFERENCE_SOLUTION
       if (verbose)
           std::cout << "Solving the saddle point system \n";
       problem_mgtools->SolveProblem(problem_mgtools->GetRhs(), *problem_refsols_lvls[0], verbose, false);

       // functional value for the initial guess
       BlockVector reduced_problem_sol(problem_mgtools->GetTrueOffsetsFunc());
       for (int blk = 0; blk < numblocks_funct; ++blk)
           reduced_problem_sol.GetBlock(blk) = problem_refsols_lvls[0]->GetBlock(blk);
       CheckFunctValueNew(comm,*NewSolver->GetFunctOp_nobnd(0), NULL, reduced_problem_sol,
                       "for the problem solution via saddle-point system ", verbose);
       if (compute_error)
           problem_mgtools->ComputeError(*problem_refsols_lvls[0], verbose, true);

       if (problem_refsols_lvls.Size() > 1)
       {
           BlockVector temp(problem->GetTrueOffsetsFunc());
           for (int blk = 0; blk < numblocks_funct; ++blk)
           {
               hierarchy->GetTruePspace( (*space_names_funct)[blk], 0)->Mult
                   (problem_refsols_lvls[1]->GetBlock(blk), temp.GetBlock(blk));
           }
           CheckFunctValueNew(comm,*NewSolver->GetFunctOp_nobnd(0), NULL, temp,
                           "for the previous level problem solution via saddle-point system ", verbose);
       }
#endif

#ifdef RECOARSENING_AMR
       if (verbose)
           std::cout << "Starting re-coarsening and re-solving part \n";
#endif
       // recoarsening constraint rhsides from finest to coarsest level
       // recoarsened rhsides are used when we are computing particular solutions
       // on any of the previous levels
       for (int l = 1; l < div_rhs_lvls.Size(); ++l)
           hierarchy->GetTruePspace(SpaceName::L2,l - 1)->MultTranspose(*div_rhs_lvls[l-1], *div_rhs_lvls[l]);

       // 7.1 Re-solving the problems with (coarsened) rhs
       // and using the previous soluition as a starting guess (if RECOARSENING_AMR
       // and/or CLEVER_STARTING_GUESS is used)

       int coarsest_lvl; // coarsest lvl to be considered in the loop below
#ifdef RECOARSENING_AMR
       coarsest_lvl = hierarchy->Nlevels() - 1;
#else
       coarsest_lvl = 0;
#endif
       for (int l = coarsest_lvl; l >= 0; --l) // all levels from coarsest to finest
       {
           if (verbose)
               std::cout << "level " << l << "\n";
           ProblemType * problem_l = prob_hierarchy->GetProblem(l);

           // Computing initial guess

           *initguesses_funct_lvls[l] = 0.0;

#ifdef CLEVER_STARTING_GUESS
           // create a better initial guess by taking the interpolant of the previous solution
           // which would be available if we consider levels finer than the coarsest
           if (l < problem_sols_lvls.Size() - 1)
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

       if (compute_error)
           problem_mgtools->ComputeError(*problem_sols_lvls[0], verbose, true);

       // to make sure that problem has grfuns in correspondence with the problem_sol we compute here
       // though for now its coordination already happens in ComputeError()
       // though for now the coordination already happens in ComputeError()
       problem_mgtools->DistributeToGrfuns(*problem_sols_lvls[0]);

       // 7.2 (optional) Send the solution by socket to a GLVis server.
       if (visualization && it == max_iter_amr - 1)
       {
           int ne = pmesh->GetNE();
           for (int elind = 0; elind < ne; ++elind)
               pmesh->SetAttribute(elind, elind);
           ParGridFunction * sigma = problem_mgtools->GetGrFun(0);
           ParGridFunction * S;
           S = problem_mgtools->GetGrFun(1);

           char vishost[] = "localhost";
           int  visport   = 19916;

           socketstream sigma_sock(vishost, visport);
           sigma_sock << "parallel " << num_procs << " " << myid << "\n";
           sigma_sock << "solution\n" << *pmesh << *sigma << "window_title 'sigma, AMR iter No."
                  << it <<"'" << flush;

           socketstream s_sock(vishost, visport);
           s_sock << "parallel " << num_procs << " " << myid << "\n";
           s_sock << "solution\n" << *pmesh << *S << "window_title 'S, AMR iter No."
                  << it <<"'" << flush;
       }

       // 7.3 Refine the mesh, either uniformly (if AMR was not #defined) or adaptively,
       // using the estimator-refiner
#ifdef AMR
       int nel_before = hierarchy->GetFinestParMesh()->GetNE();

       // testing with only 1 element marked for refinement
       //Array<int> els_to_refine(1);
       //els_to_refine = hierarchy->GetFinestParMesh()->GetNE() / 2;
       //hierarchy->GetFinestParMesh()->GeneralRefinement(els_to_refine);

       // true AMR
       refiner.Apply(*hierarchy->GetFinestParMesh());
       int nmarked_el = refiner.GetNumMarkedElements();
       if (verbose)
       {
           std::cout << "Marked elements percentage = " << 100 * nmarked_el * 1.0 / nel_before << " % \n";
           std::cout << "nmarked_el = " << nmarked_el << ", nel_before = " << nel_before << "\n";
           int nel_after = hierarchy->GetFinestParMesh()->GetNE();
           std::cout << "nel_after = " << nel_after << "\n";
           std::cout << "number of elements introduced = " << nel_after - nel_before << "\n";
           std::cout << "percentage (w.r.t to # before) of elements introduced = " <<
                        100.0 * (nel_after - nel_before) * 1.0 / nel_before << "% \n\n";
       }

       // (optional) Send the vector of local element errors by socket to a GLVis server.
       if (visualization && it == max_iter_amr - 1)
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
               L2_space = new ParFiniteElementSpace(problem_mgtools->GetParMesh(), l2_coll);
           ParGridFunction * local_errors_pgfun = new ParGridFunction(L2_space);
           local_errors_pgfun->SetFromTrueDofs(local_errors);
           char vishost[] = "localhost";
           int  visport   = 19916;

           socketstream amr_sock(vishost, visport);
           amr_sock << "parallel " << num_procs << " " << myid << "\n";
           amr_sock << "solution\n" << *pmesh << *local_errors_pgfun <<
                         "window_title 'local errors, AMR iter No." << it <<"'" << flush;

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

       // 7.4 After the refinement, updating the hierarchy and everything else
       bool recoarsen = true;
       prob_hierarchy->Update(recoarsen);
       problem = prob_hierarchy->GetProblem(0);

       problem_mgtools->BuildSystem(verbose);
       mgtools_hierarchy->Update(recoarsen);
       NewSolver->UpdateProblem(*problem_mgtools);
       NewSolver->Update(recoarsen);

       partsol_finder->UpdateProblem(*problem_mgtools);
       partsol_finder->Update(recoarsen);

       // checking #dofs after the refinement
       global_dofs = problem_mgtools->GlobalTrueProblemSize();

       if (global_dofs > max_dofs)
       {
          if (verbose)
             cout << "Reached the maximum number of dofs. Stop. \n";
          break;
       }

   }

   // 8. Deallocating memory
   delete NewSolver;
   delete mgtools_hierarchy;
   delete descriptor;

   delete partsol_finder;

   for (int i = 0; i < div_rhs_lvls.Size(); ++i)
       delete div_rhs_lvls[i];
   for (int i = 0; i < partsol_funct_lvls.Size(); ++i)
       delete partsol_funct_lvls[i];
   for (int i = 0; i < initguesses_funct_lvls.Size(); ++i)
       delete initguesses_funct_lvls[i];

   for (int i = 0; i < problem_refsols_lvls.Size(); ++i)
       delete problem_refsols_lvls[i];

   for (int i = 0; i < problem_sols_lvls.Size(); ++i)
       delete problem_sols_lvls[i];
   delete hierarchy;
   delete prob_hierarchy;

   for (int i = 0; i < integs.NumRows(); ++i)
       for (int j = 0; j < integs.NumCols(); ++j)
           if (integs(i,j))
               delete integs(i,j);

   delete problem_mgtools;
   delete estimator;

   delete bdr_conds;
   delete formulat;
   delete fe_formulat;

   MPI_Finalize();
   return 0;
}

