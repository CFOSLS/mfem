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

// activates using the solution at the previous mesh as a starting guess for the next problem
#define CLEVER_STARTING_GUESS

// activates using a (simpler & cheaper) preconditioner for the problems, simple Gauss-Seidel
#define USE_GS_PREC

//#define FOSLS

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
        mesh_file = "../data/netgen_lshape3D_onemoretry.netgen";
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

#ifdef CLEVER_STARTING_GUESS
    if (verbose)
        std::cout << "CLEVER_STARTING_GUESS active \n";
#else
    if (verbose)
        std::cout << "CLEVER_STARTING_GUESS passive \n";
#endif

#ifdef USE_GS_PREC
    if (verbose)
        std::cout << "USE_GS_PREC active (overwrites the prec_option) \n";
#else
    if (verbose)
        std::cout << "USE_GS_PREC passive \n";
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

   FOSLSFormulation * formulat = new FormulType (dim, numsol, verbose);
   FOSLSFEFormulation * fe_formulat = new FEFormulType(*formulat, feorder);
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

       int stopcriteria_type = 1;

       int numblocks_funct = numblocks - 1;

       int size_funct = problem_mgtools->GetTrueOffsetsFunc()[numblocks_funct];
       NewSolver = new GeneralMinConstrSolver(size_funct, *mgtools_hierarchy, with_local_smoothers,
                                        optimized_localsolvers, with_hcurl_smoothers, stopcriteria_type, verbose);
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

   estimator = new FOSLSEstimatorOnHier<ProblemType, GeneralHierarchy>
           (*prob_hierarchy, 0, grfuns_descriptor, NULL, integs, verbose);

   problem->AddEstimator(*estimator);

   ThresholdRefiner refiner(*estimator);
   refiner.SetTotalErrorFraction(0.95); // 0.5

   Array<Vector*> div_rhs_lvls(0);
   //Array<BlockVector*> partsol_lvls(0);
   //Array<BlockVector*> partsol_funct_lvls(0);
   Array<BlockVector*> initguesses_funct_lvls(0);

   Array<BlockVector*> problem_sols_lvls(0);

   // 12. The main AMR loop. In each iteration we solve the problem on the
   //     current mesh, visualize the solution, and refine the mesh.
#ifdef AMR
   const int max_dofs = 200000;//1600000;
#else
   const int max_dofs = 400000;
#endif

   const double fixed_rtol = 1.0e-15; // 1.0e-10; 1.0e-12;
   const double fixed_atol = 1.0e-5;

   const double initial_rtol = fixed_rtol;
   const double initial_atol = fixed_atol;
   double initial_res_norm = -1.0;

   double adjusted_atol = -1.0;

   HYPRE_Int global_dofs = problem->GlobalTrueProblemSize();
   std::cout << "starting n_el = " << prob_hierarchy->GetHierarchy().GetFinestParMesh()->GetNE() << "\n";

   // Main loop (with AMR or uniform refinement depending on the predefined macros)
   for (int it = 0; ; it++)
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

       //partsol_lvls.Prepend(new BlockVector(problem->GetTrueOffsets()));
       //*partsol_lvls[0] = 0.0;

       //partsol_funct_lvls.Prepend(new BlockVector(problem->GetTrueOffsetsFunc()));
       //*partsol_funct_lvls[0] = 0.0;

       div_rhs_lvls.Prepend(new Vector(problem->GetRhs().GetBlock(numblocks - 1).Size()));
       *div_rhs_lvls[0] = problem->GetRhs().GetBlock(numblocks - 1);

       //if (verbose && it == 0)
           //std::cout << "div_rhs norm = " << div_rhs_lvls[0]->Norml2() / sqrt (div_rhs_lvls[0]->Size()) << "\n";

       if (verbose)
           std::cout << "Starting re-coarsening and re-solving part \n";

       // recoarsening constraint rhsides from finest to coarsest level
       for (int l = 1; l < div_rhs_lvls.Size(); ++l)
           hierarchy->GetTruePspace(SpaceName::L2,l - 1)->MultTranspose(*div_rhs_lvls[l-1], *div_rhs_lvls[l]);

       if (verbose)
       {
           //std::cout << "norms of partsol_lvls before: \n";
           //for (int l = 0; l < partsol_lvls.Size(); ++l)
               //std::cout << "partsol norm = " << partsol_lvls[l]->Norml2() / sqrt(partsol_lvls[l]->Size()) << "\n";;
           std::cout << "norms of rhs_lvls before: \n";
           for (int l = 0; l < div_rhs_lvls.Size(); ++l)
               std::cout << "rhs norm = " << div_rhs_lvls[l]->Norml2() / sqrt(div_rhs_lvls[l]->Size()) << "\n";;
       }

       // re-solving all the problems with coarsened rhs, from coarsest to finest
       // and using the previous soluition as a starting guess
       int coarsest_lvl = prob_hierarchy->Nlevels() - 1;
       for (int l = coarsest_lvl; l >= 1; --l) // l = 0 could be included actually after testing
       {
           if (verbose)
               std::cout << "level " << l << "\n";
           ProblemType * problem_l = prob_hierarchy->GetProblem(l);

           /*
           // finding a new particular solution for the new rhs
           Vector partsol_guess(partsol_funct_lvls[l]->Size());//partsol_finder->Size());
           partsol_guess = 0.0;

           if (l < coarsest_lvl)
           {
               BlockVector partsol_guess_viewer(partsol_guess.GetData(), problem_l->GetTrueOffsetsFunc());
               for (int blk = 0; blk < numblocks_funct; ++blk)
                   hierarchy->GetTruePspace((*space_names_funct)[blk], l)->Mult
                           (problem_sols_lvls[l + 1]->GetBlock(blk), partsol_guess_viewer.GetBlock(blk));
           }

           HypreParMatrix& Constr_l = (HypreParMatrix&)problem_l->GetOp_nobnd()->GetBlock(numblocks_funct,0);
           // full V-cycle
           //partsol_finder->FindParticularSolution(l, Constr_l, partsol_guess,
                                                  //*partsol_funct_lvls[l], *div_rhs_lvls[l], verbose, report_funct);

           // finest-available level update
           partsol_finder->UpdateParticularSolution(l, Constr_l, partsol_guess,
                                                  *partsol_funct_lvls[l], *div_rhs_lvls[l], verbose, report_funct);

           for (int blk = 0; blk < numblocks_funct; ++blk)
               partsol_lvls[l]->GetBlock(blk) = partsol_funct_lvls[l]->GetBlock(blk);

           // a check that the particular solution does satisfy the divergence constraint after all
           HypreParMatrix & Constr = (HypreParMatrix&)(problem_l->GetOp()->GetBlock(numblocks - 1, 0));
           Vector tempc(Constr.Height());
           Constr.Mult(partsol_lvls[l]->GetBlock(0), tempc);
           tempc -= *div_rhs_lvls[l];
           double res_constr_norm = ComputeMPIVecNorm(comm, tempc, "", false);
           MFEM_ASSERT (res_constr_norm < 1.0e-10, "");
           */

           // solving the problem at level l
           // ...
           MFEM_ABORT("Code is to be written here \n");
       }

       if (verbose)
           std::cout << "Re-coarsening (and re-solving if divfree problem in H(curl) is considered)"
                        " has been finished\n";

       /*
       // finding the particular solution for the finest level
       {
           ProblemType * problem_l = prob_hierarchy->GetProblem(0);

           // finding a new particular solution for the new rhs
           Vector partsol_guess(partsol_funct_lvls[0]->Size());//partsol_finder->Size());
           partsol_guess = 0.0;

           HypreParMatrix& Constr_l = (HypreParMatrix&)problem_l->GetOp_nobnd()->GetBlock(numblocks_funct,0);

           if (it > 0)
           {
               BlockVector partsol_guess_viewer(partsol_guess.GetData(), problem_l->GetTrueOffsetsFunc());
               for (int blk = 0; blk < numblocks_funct; ++blk)
                   hierarchy->GetTruePspace((*space_names_funct)[blk], 0)->Mult
                           (problem_sols_lvls[1]->GetBlock(blk), partsol_guess_viewer.GetBlock(blk));
           }

           // full V-cycle
           //partsol_finder->FindParticularSolution(l, Constr_l, partsol_guess,
                                                  //*partsol_funct_lvls[l], *div_rhs_lvls[l], verbose, report_funct);

           // finest-available level update
           partsol_finder->UpdateParticularSolution(0, Constr_l, partsol_guess,
                                                  *partsol_funct_lvls[0], *div_rhs_lvls[0], verbose, report_funct);

           for (int blk = 0; blk < numblocks_funct; ++blk)
               partsol_lvls[0]->GetBlock(blk) = partsol_funct_lvls[0]->GetBlock(blk);
       }
       */

       // solving at the finest level
       BlockVector& problem_sol = problem->GetSol();
       problem_sol = 0.0;

       double newsolver_reltol = 1.0e-6;

       if (verbose)
           std::cout << "newsolver_reltol = " << newsolver_reltol << "\n";

       NewSolver->SetRelTol(newsolver_reltol);
       NewSolver->SetMaxIter(200);
       NewSolver->SetPrintLevel(1);
       NewSolver->SetStopCriteriaType(0);

#ifdef CLEVER_STARTING_GUESS
       if (it > 0)
           for (int blk = 0; blk < numblocks_funct; ++blk)
               hierarchy->GetTruePspace( (*space_names_funct)[blk], 0)->Mult
                   (problem_sols_lvls[1]->GetBlock(blk), initguesses_funct_lvls[0]->GetBlock(blk));
#endif
       // setting correct bdr values
       BlockVector * bdr_cond = *problem->GetInitialCondition();
       MFEM_ABORT("Code fails at the next line");
       // it should be setting only tdofs at the boundary, so the semantics is wrong here
       *initguesses_funct_lvls[0] = *bdr_cond;

       //*initguesses_funct_lvls[0] += *partsol_funct_lvls[0];

       NewSolver->SetInitialGuess(*initguesses_funct_lvls[0]);
       NewSolver->SetConstrRhs(*div_rhs_lvls[0]);
       //NewSolver.SetUnSymmetric();

       if (verbose)
           NewSolver->PrintAllOptions();

       Vector NewRhs(NewSolver->Size());
       NewRhs = 0.0;

       BlockVector divfree_part(problem->GetTrueOffsetsFunc());
       NewSolver->Mult(NewRhs, divfree_part);

       for (int blk = 0; blk < numblocks_funct; ++blk)
           problem_sol.GetBlock(blk) = divfree_part.GetBlock(blk);

       //problem_sol += *partsol_lvls[0];

       *problem_sols_lvls[0] = problem_sol;

       if (compute_error)
           problem->ComputeError(problem_sol, verbose, false);

       // to make sure that problem has grfuns in correspondence with the problem_sol we compute here
       // though for now its coordination already happens in ComputeError()
       problem->DistributeToGrfuns(problem_sol);


       // Send the solution by socket to a GLVis server.
       if (visualization)
       {
           ParGridFunction * sigma = problem->GetGrFun(0);
           ParGridFunction * S;
           S = problem->GetGrFun(1);

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

       if (visualization)
       {
           const Vector& local_errors = estimator->GetLocalErrors();
           if (feorder == 0)
               MFEM_ASSERT(local_errors.Size() == problem->GetPfes(numblocks_funct)->TrueVSize(), "");

           FiniteElementCollection * l2_coll;
           if (feorder > 0)
               l2_coll = new L2_FECollection(0, dim);

           ParFiniteElementSpace * L2_space;
           if (feorder == 0)
               L2_space = problem->GetPfes(numblocks_funct);
           else
               L2_space = new ParFiniteElementSpace(problem->GetParMesh(), l2_coll);
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
       global_dofs = problem->GlobalTrueProblemSize();

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

