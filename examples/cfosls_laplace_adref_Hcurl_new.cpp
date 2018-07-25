//                                MFEM(with 4D elements) CFOSLS for 3D/4D laplace equation
//                                  with adaptive refinement involving a div-free formulation
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>

// if passive, the mesh is simply uniformly refined at each iteration
#define AMR

//#define RECOARSENING_AMR

// activates using the solution at the previous mesh as a starting guess for the next problem
// combined with RECOARSENING_AMR this leads to a variant of cascadic MG
#define CLEVER_STARTING_GUESS

//#define FOSLS

#define REFERENCE_SOLUTION

using namespace std;
using namespace mfem;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

int main(int argc, char *argv[])
{
    int num_procs, myid;
    bool visualization = 1;

    // 1. Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    bool verbose = (myid == 0);

    int nDimensions     = 3;
    int numsol          = 11;

    int ser_ref_levels  = 1;
    int par_ref_levels  = 1;

    const char *formulation = "cfosls"; // "cfosls" or "fosls"
    const char *space_for_S = "H1";     // "H1" or "L2"
#ifdef FOSLS
    if (strcmp(space_for_S,"L2") == 0)
    {
        MFEM_ABORT("FOSLS formulation of Laplace equation requires S from H^1");
    }
#endif
    const char *space_for_sigma = "Hdiv"; // "Hdiv" or "H1"

#ifdef FOSLS
    using FormulType = FOSLSFormulation_Laplace;
    using FEFormulType = FOSLSFEFormulation_Laplace;
    using BdrCondsType = BdrConditions_Laplace;
    using ProblemType = FOSLSProblem_Laplace;
#else
    // Hdiv-H1 fomulation
    using FormulType = CFOSLSFormulation_Laplace;
    using FEFormulType = CFOSLSFEFormulation_HdivH1L2_Laplace;
    using BdrCondsType = BdrConditions_CFOSLS_HdivH1Laplace;
    using ProblemType = FOSLSProblem_HdivH1lapl;


#endif

    /*
    using FormulType = CFOSLSFormulation_MixedLaplace;
    using FEFormulType = CFOSLSFEFormulation_MixedLaplace;
    using BdrCondsType = BdrConditions_MixedLaplace;
    using ProblemType = FOSLSProblem_MixedLaplace;
    */

    // solver options
    int prec_option = 1; //defines whether to use preconditioner or not, and which one

    const char *mesh_file = "../data/cube_3d_moderate.mesh";

    int feorder         = 0;
#ifdef FOSLS
    ++feorder;
#endif

    if (verbose)
        cout << "Solving (С)FOSLS laplace equation with MFEM & hypre \n";

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

        if (strcmp(space_for_S,"L2") == 0)
            std::cout << "S: is eliminated from the system \n";
    }

    if (verbose)
        std::cout << "Running tests for the paper: \n";

    //mesh_file = "../data/netgen_cylinder_mesh_0.1to0.2.mesh";
    //mesh_file = "../data/pmesh_cylinder_moderate_0.2.mesh";
    //mesh_file = "../data/pmesh_cylinder_fine_0.1.mesh";

    //mesh_file = "../data/pmesh_check.mesh";
    mesh_file = "../data/cube_3d_moderate.mesh";

    if (numsol == 11)
    {
        //mesh_file = "../data/netgen_lshape3D_onemoretry.netgen";
        mesh_file = "../data/netgen_lshape3D_onemoretry_coarsest.netgen";
    }

    if (verbose)
        std::cout << "For the records: numsol = " << numsol
                  << ", mesh_file = " << mesh_file << "\n";

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

    MFEM_ASSERT(strcmp(formulation,"cfosls") == 0 || strcmp(formulation,"fosls") == 0,
                "Formulation must be cfosls or fosls!\n");
    MFEM_ASSERT(strcmp(space_for_S,"H1") == 0 || strcmp(space_for_S,"L2") == 0,
                "Space for S must be H1 or L2!\n");
    MFEM_ASSERT(strcmp(space_for_sigma,"Hdiv") == 0 || strcmp(space_for_sigma,"H1") == 0,
                "Space for sigma must be Hdiv or H1!\n");

    MFEM_ASSERT(!strcmp(space_for_sigma,"H1") == 0 || (strcmp(space_for_sigma,"H1") == 0 &&
                                                       strcmp(space_for_S,"H1") == 0),
                "Sigma from H1vec must be coupled with S from H1!\n");

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

    pmesh->PrintInfo(std::cout); if(verbose) cout << endl;

    // 6. Define a parallel finite element space on the parallel mesh. Here we
    //    use the Raviart-Thomas finite elements of the specified order.

    int numblocks = 1;

    if (strcmp(space_for_S,"H1") == 0)
        numblocks++;
    if (strcmp(formulation,"cfosls") == 0)
        numblocks++;

    if (verbose)
        std::cout << "Number of blocks in the formulation: " << numblocks << "\n";

   if (verbose)
       std::cout << "Running AMR ... \n";

   FormulType * formulat = new FormulType (dim, numsol, verbose);
   FEFormulType * fe_formulat = new FEFormulType(*formulat, feorder);
   BdrConditions * bdr_conds = new BdrCondsType(*pmesh);


   bool with_hcurl = true;
   GeneralHierarchy * hierarchy = new GeneralHierarchy(1, *pmesh, feorder, verbose, with_hcurl);
   hierarchy->ConstructDofTrueDofs();
   hierarchy->ConstructDivfreeDops();

   FOSLSProblHierarchy<ProblemType, GeneralHierarchy> * prob_hierarchy = new
           FOSLSProblHierarchy<ProblemType, GeneralHierarchy>
           (*hierarchy, 1, *bdr_conds, *fe_formulat, prec_option, verbose);

   ///////////////////////////////////////////////////////
   ProblemType * problem = prob_hierarchy->GetProblem(0);


   //bool optimized_localsolvers = true;
   //bool with_hcurl_smoothers = true;
   //DivConstraintSolver * partsol_finder;
   //partsol_finder = new DivConstraintSolver
           //(*problem, *hierarchy, optimized_localsolvers, with_hcurl_smoothers, verbose);
   //bool report_funct = true;

   const Array<SpaceName>* space_names_funct = problem->GetFEformulation().GetFormulation()->
           GetFunctSpacesDescriptor();

   FOSLSProblem* problem_mgtools = hierarchy->BuildDynamicProblem<ProblemType>
           (*bdr_conds, *fe_formulat, prec_option, verbose);
   hierarchy->AttachProblem(problem_mgtools);

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

       double newsolver_reltol = 1.0e-6;

       if (verbose)
           std::cout << "newsolver_reltol = " << newsolver_reltol << "\n";

       NewSolver->SetRelTol(newsolver_reltol);
       NewSolver->SetMaxIter(200);
       NewSolver->SetPrintLevel(0);
       NewSolver->SetStopCriteriaType(stopcriteria_type);
   }

   ///////////////////////////////////////////////////////////////

   Laplace_test* Mytest = dynamic_cast<Laplace_test*>
           (problem->GetFEformulation().GetFormulation()->GetTest());
   MFEM_ASSERT(Mytest, "Unsuccessful cast into Hyper_test* \n");

   int numfoslsfuns = -1;

   int fosls_func_version = 1;
   if (verbose)
    std::cout << "fosls_func_version = " << fosls_func_version << "\n";

   if (fosls_func_version == 1)
   {
#ifdef FOSLS
       numfoslsfuns = 1;
#else
       numfoslsfuns = 1;
       if (strcmp(space_for_S,"H1") == 0)
           ++numfoslsfuns;
#endif
   }

#ifdef FOSLS
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
#ifdef FOSLS
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

   Array<Vector*> div_rhs_lvls(0);
   //Array<BlockVector*> partsol_lvls(0);
   //Array<BlockVector*> partsol_funct_lvls(0);
   Array<BlockVector*> initguesses_funct_lvls(0);

   Array<BlockVector*> problem_sols_lvls(0);

#ifdef REFERENCE_SOLUTION
   Array<BlockVector*> problem_refsols_lvls(0);
   Array<BlockVector*> problem_refsols_funct_lvls(0);
#endif

   double saved_functvalue;

   // 12. The main AMR loop. In each iteration we solve the problem on the
   //     current mesh, visualize the solution, and refine the mesh.
#ifdef AMR
   const int max_dofs = 200000;//1600000;
#else
   const int max_dofs = 600000;
#endif

   //const double fixed_rtol = 1.0e-15; // 1.0e-10; 1.0e-12;
   //const double fixed_atol = 1.0e-5;

   //const double initial_rtol = fixed_rtol;
   //const double initial_atol = fixed_atol;
   //double initial_res_norm = -1.0;

   //double adjusted_atol = -1.0;

   HYPRE_Int global_dofs = problem->GlobalTrueProblemSize();
   std::cout << "starting n_el = " << problem_mgtools->GetParMesh()->GetNE() << "\n";

   // Main loop (with AMR or uniform refinement depending on the predefined macros)
   int max_iter_amr = 20;
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

       problem_sols_lvls.Prepend(new BlockVector(problem->GetTrueOffsets()));
       *problem_sols_lvls[0] = 0.0;

#ifdef REFERENCE_SOLUTION
       problem_refsols_lvls.Prepend(new BlockVector(problem->GetTrueOffsets()));
       *problem_refsols_lvls[0] = 0.0;
       problem_refsols_funct_lvls.Prepend(new BlockVector(problem->GetTrueOffsetsFunc()));
       *problem_refsols_funct_lvls[0] = 0.0;
#endif
       //for (int i = 0; i < problem_sols_lvls.Size(); ++i)
           //std::cout << "problem_sols_lvls[" << i << "]'s' size = " << problem_sols_lvls[i]->Size() << "\n";

       div_rhs_lvls.Prepend(new Vector(problem_mgtools->GetRhs().GetBlock(numblocks - 1).Size()));
       //*div_rhs_lvls[0] = problem_mgtools->GetRhs().GetBlock(numblocks - 1); // incorrect but works for laplace with no bdr for sigma
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

       for (int blk = 0; blk < numblocks_funct; ++blk)
           problem_refsols_funct_lvls[0]->GetBlock(blk) = problem_refsols_lvls[0]->GetBlock(blk);

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

       /*
       for (int l = 0; l < hierarchy->Nlevels(); ++l)
       {
           if (verbose)
               std::cout << "l = " << l << ", ref problem sol funct value \n";
           Vector temp(NewSolver->GetFunctOp_nobnd(0)->Width());
           if (l == 0)
               temp = problem_refsols_funct_lvls[0];
           else
           {
               Vector temp2()
               for (int k = l - 1; k >=0; --k)
               {
                   for (int blk = 0; blk < numblocks_funct; ++blk)
                   {
                       hierarchy->GetTruePspace( (*space_names_funct)[blk], k)->Mult
                           (problem_refsols_lvls[l + 1]->GetBlock(blk), initguesses_funct_lvls[l]->GetBlock(blk));
                   }

               }

               temp = proj;
           }
           CheckFunctValueNew(comm,*NewSolver->GetFunctOp_nobnd(0), NULL, temp,
                           "for the level problem solution via saddle-point system ", verbose);
       }
       */

#endif
//#else

#ifdef RECOARSENING_AMR
       if (verbose)
           std::cout << "Starting re-coarsening and re-solving part \n";
#endif
       // recoarsening constraint rhsides from finest to coarsest level
       for (int l = 1; l < div_rhs_lvls.Size(); ++l)
           hierarchy->GetTruePspace(SpaceName::L2,l - 1)->MultTranspose(*div_rhs_lvls[l-1], *div_rhs_lvls[l]);

       // re-solving all the problems with coarsened rhs, from coarsest to finest
       // and using the previous soluition as a starting guess
       int coarsest_lvl; // coarsest lvl considered below
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

           // solving the problem at level l

           *initguesses_funct_lvls[l] = 0.0;

#ifdef CLEVER_STARTING_GUESS
           // create a better initial guess
           if (l < problem_sols_lvls.Size() - 1)
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

           // for debugging
           //for (int blk = 0; blk < numblocks_funct; ++blk)
               //initguesses_funct_lvls[l]->GetBlock(blk) = reduced_problem_sol.GetBlock(blk);

           //std::cout << "check init norm after bnd = " << initguesses_funct_lvls[l]->Norml2()
                        /// sqrt (initguesses_funct_lvls[l]->Size()) << "\n";

           // checking the initial guess
           {
               problem_l->ComputeBndError(*initguesses_funct_lvls[l]);

               HypreParMatrix & Constr = (HypreParMatrix&)(problem_l->GetOp_nobnd()->GetBlock(numblocks - 1, 0));
               Vector tempc(Constr.Height());
               Constr.Mult(initguesses_funct_lvls[l]->GetBlock(0), tempc);
               tempc -= *div_rhs_lvls[l];
               double res_constr_norm = ComputeMPIVecNorm(comm, tempc, "", false);
               MFEM_ASSERT (res_constr_norm < 1.0e-10, "");
           }

           // functional value for the initial guess
           double starting_funct_value = CheckFunctValueNew(comm,*NewSolver->GetFunctOp_nobnd(l), NULL,
                                                         *initguesses_funct_lvls[l], "for the initial guess ", verbose);

           BlockVector zero_vec(problem_l->GetTrueOffsetsFunc());
           zero_vec = 0.0;
           NewSolver->SetInitialGuess(l, zero_vec);

           //NewSolver->SetInitialGuess(l, *initguesses_funct_lvls[l]);

           NewSolver->SetConstrRhs(*div_rhs_lvls[l]);

           //if (verbose)
               //NewSolver->PrintAllOptions();

           BlockVector NewRhs(problem_l->GetTrueOffsetsFunc());
           NewRhs = 0.0;

           // computing rhs = - Funct_nobnd * init_guess at level l, with zero boundary conditions imposed
           BlockVector padded_initguess(problem_l->GetTrueOffsets());
           padded_initguess = 0.0;
           for (int blk = 0; blk < numblocks_funct; ++blk)
               padded_initguess.GetBlock(blk) = initguesses_funct_lvls[l]->GetBlock(blk);

           BlockVector padded_rhs(problem_l->GetTrueOffsets());
           problem_l->GetOp_nobnd()->Mult(padded_initguess, padded_rhs);

           padded_rhs *= -1;
           for (int blk = 0; blk < numblocks_funct; ++blk)
               NewRhs.GetBlock(blk) = padded_rhs.GetBlock(blk);
           problem_l->ZeroBndValues(NewRhs);

           NewSolver->SetFunctRhs(NewRhs);
           if (l == coarsest_lvl && it == 0)
               NewSolver->SetBaseValue(starting_funct_value);
           else
               NewSolver->SetBaseValue(saved_functvalue);
           NewSolver->SetFunctAdditionalVector(*initguesses_funct_lvls[l]);

           HypreParMatrix & Constr_l = (HypreParMatrix&)(problem_l->GetOp()->GetBlock(numblocks - 1, 0));

           // solving for correction
           BlockVector correction(problem_l->GetTrueOffsetsFunc());
           correction = 0.0;
           //std::cout << "NewSolver size = " << NewSolver->Size() << "\n";
           //std::cout << "NewRhs norm = " << NewRhs.Norml2() / sqrt (NewRhs.Size()) << "\n";
           //if (l == 0)
               //NewSolver->SetPrintLevel(1);
           //else
               NewSolver->SetPrintLevel(1);

           if (verbose)
               std::cout << "Solving the minimization problem at level " << l << "\n";

           NewSolver->Mult(l, &Constr_l, NewRhs, correction);

           for (int blk = 0; blk < numblocks_funct; ++blk)
           {
               problem_sols_lvls[l]->GetBlock(blk) = initguesses_funct_lvls[l]->GetBlock(blk);
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
//#endif // for #ifdef REFERENCE_SOLUTION

       if (compute_error)
           problem_mgtools->ComputeError(*problem_sols_lvls[0], verbose, true);

       // to make sure that problem has grfuns in correspondence with the problem_sol we compute here
       // though for now its coordination already happens in ComputeError()
       problem_mgtools->DistributeToGrfuns(*problem_sols_lvls[0]);


       // Send the solution by socket to a GLVis server.
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

       bool recoarsen = true;
       prob_hierarchy->Update(recoarsen);
       problem = prob_hierarchy->GetProblem(0);

       problem_mgtools->BuildSystem(verbose);
       mgtools_hierarchy->Update(recoarsen);
       NewSolver->UpdateProblem(*problem_mgtools);
       NewSolver->Update(recoarsen);

       //partsol_finder->UpdateProblem(*problem);
       //partsol_finder->Update(recoarsen);

       // checking #dofs after the refinement
       global_dofs = problem_mgtools->GlobalTrueProblemSize();

       if (global_dofs > max_dofs)
       {
          if (verbose)
             cout << "Reached the maximum number of dofs. Stop. \n";
          break;
       }

   }

   MPI_Finalize();
   return 0;
}

