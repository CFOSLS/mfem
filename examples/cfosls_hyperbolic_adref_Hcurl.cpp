//                                MFEM(with 4D elements) CFOSLS with S from H1 for 3D/4D hyperbolic equation
//                                  with adaptive refinement in H(curl)
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
// Solver: MINRES preconditioned by boomerAMG or ADS

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>

#define DIVFREE_ESTIMATOR

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
    int numsol          = -3;

    int ser_ref_levels  = 0;
    int par_ref_levels  = 0;

    const char *formulation = "cfosls"; // "cfosls" or "fosls"
    const char *space_for_S = "H1";     // "H1" or "L2"
    const char *space_for_sigma = "Hdiv"; // "Hdiv" or "H1"
    bool eliminateS = true;            // in case space_for_S = "L2" defines whether we eliminate S from the system
    bool keep_divdiv = false;           // in case space_for_S = "L2" defines whether we keep div-div term in the system

    // solver options
    int prec_option = 1; //defines whether to use preconditioner or not, and which one
    bool use_ADS;

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

    int feorder         = 0;

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
    args.AddOption(&eliminateS, "-elims", "--eliminateS", "-no-elims",
                   "--no-eliminateS",
                   "Turn on/off elimination of S in L2 formulation.");
    args.AddOption(&keep_divdiv, "-divdiv", "--divdiv", "-no-divdiv",
                   "--no-divdiv",
                   "Defines if div-div term is/ is not kept in the system.");
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
        {
            std::cout << "S: is ";
            if (!eliminateS)
                std::cout << "not ";
            std::cout << "eliminated from the system \n";
        }

        std::cout << "div-div term: is ";
        if (keep_divdiv)
            std::cout << "not ";
        std::cout << "eliminated \n";
    }

    if (verbose)
        std::cout << "Running tests for the paper: \n";


    //mesh_file = "../data/netgen_cylinder_mesh_0.1to0.2.mesh";
    //mesh_file = "../data/pmesh_cylinder_moderate_0.2.mesh";
    //mesh_file = "../data/pmesh_cylinder_fine_0.1.mesh";

    mesh_file = "../data/pmesh_check.mesh";
    //mesh_file = "../data/cube_3d_moderate.mesh";


    if (verbose)
        std::cout << "For the records: numsol = " << numsol
                  << ", mesh_file = " << mesh_file << "\n";

    switch (prec_option)
    {
    case 1: // smth simple like AMS
        use_ADS = false;
        break;
    case 2: // MG
        use_ADS = true;
        break;
    default: // no preconditioner
        break;
    }

    if (verbose)
    {
        std::cout << "use_ADS = " << use_ADS << "\n";
    }

    //MFEM_ASSERT(numsol == 8 && nDimensions == 3, "Adaptive refinement is tested currently only for the older reports' problem in the cylinder! \n");

    MFEM_ASSERT(strcmp(formulation,"cfosls") == 0 || strcmp(formulation,"fosls") == 0, "Formulation must be cfosls or fosls!\n");
    MFEM_ASSERT(strcmp(space_for_S,"H1") == 0 || strcmp(space_for_S,"L2") == 0, "Space for S must be H1 or L2!\n");
    MFEM_ASSERT(strcmp(space_for_sigma,"Hdiv") == 0 || strcmp(space_for_sigma,"H1") == 0, "Space for sigma must be Hdiv or H1!\n");

    MFEM_ASSERT(!strcmp(space_for_sigma,"H1") == 0 || (strcmp(space_for_sigma,"H1") == 0 && strcmp(space_for_S,"H1") == 0), "Sigma from H1vec must be coupled with S from H1!\n");
    MFEM_ASSERT(!strcmp(space_for_sigma,"H1") == 0 || (strcmp(space_for_sigma,"H1") == 0 && use_ADS == false), "ADS cannot be used when sigma is from H1vec!\n");
    MFEM_ASSERT(!(strcmp(formulation,"fosls") == 0 && strcmp(space_for_S,"L2") == 0 && !keep_divdiv), "For FOSLS formulation with S from L2 div-div term must be present!\n");
    MFEM_ASSERT(!(strcmp(formulation,"cfosls") == 0 && strcmp(space_for_S,"H1") == 0 && keep_divdiv), "For CFOSLS formulation with S from H1 div-div term must not be present for sigma!\n");

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

    //if(dim==3) pmesh->ReorientTetMesh();

    pmesh->PrintInfo(std::cout); if(verbose) cout << endl;

    // 6. Define a parallel finite element space on the parallel mesh. Here we
    //    use the Raviart-Thomas finite elements of the specified order.
    int dim = nDimensions;

    int numblocks = 1;

    if (strcmp(space_for_S,"H1") == 0)
        numblocks++;
    else // "L2"
        if (!eliminateS)
            numblocks++;
    if (strcmp(formulation,"cfosls") == 0)
        numblocks++;

    if (verbose)
        std::cout << "Number of blocks in the formulation: " << numblocks << "\n";

   if (verbose)
       std::cout << "Running AMR ... \n";

   // Hdiv-H1 case
   using FormulType = CFOSLSFormulation_HdivH1Hyper;
   using FEFormulType = CFOSLSFEFormulation_HdivH1Hyper;
   using BdrCondsType = BdrConditions_CFOSLS_HdivH1_Hyper;
   using ProblemType = FOSLSProblem_HdivH1L2hyp;
#ifdef DIVFREE_ESTIMATOR
   using DivfreeFormulType = CFOSLSFormulation_HdivH1DivfreeHyp;
   using DivfreeFEFormulType = CFOSLSFEFormulation_HdivH1DivfreeHyper;
#endif

   /*
   // Hdiv-L2 case
   using FormulType = CFOSLSFormulation_HdivL2Hyper;
   using FEFormulType = CFOSLSFEFormulation_HdivL2Hyper;
   using BdrCondsType = BdrConditions_CFOSLS_HdivL2_Hyper;
   using ProblemType = FOSLSCylProblem_HdivL2L2hyp;
   */

   FOSLSFormulation * formulat = new FormulType (dim, numsol, verbose);
   FOSLSFEFormulation * fe_formulat = new FEFormulType(*formulat, feorder);
   BdrConditions * bdr_conds = new BdrCondsType(*pmesh);
#ifdef DIVFREE_ESTIMATOR
   DivfreeFormulType * formulat_divfree = new DivfreeFormulType (dim, numsol, verbose);
   DivfreeFEFormulType * fe_formulat_divfree = new DivfreeFEFormulType(*formulat_divfree, feorder);
#endif

   // now constructed via a hierarchy
   //ProblemType * problem = new ProblemType (*pmesh, *bdr_conds, *fe_formulat, prec_option, verbose);

   //FOSLSDivfreeProblem * problem_divfree = new FOSLSDivfreeProblem(*pmesh, *bdr_conds, *fe_formulat_divfree,
                                                                   //*problem->GetFEformulation().GetFeColl(0),
                                                                   //*problem->GetPfes(0), verbose);
   //FOSLSDivfreeProblem * problem_divfree = new FOSLSDivfreeProblem(*pmesh, *bdr_conds, *fe_formulat_divfree,
                                                                   //*problem->GetFEformulation().GetFeColl(0),
                                                                   //*problem->GetPfes(0), verbose);

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

   GeneralHierarchy * hierarchy = new GeneralHierarchy(1, *pmesh, feorder, verbose);
   FOSLSProblHierarchy<ProblemType, GeneralHierarchy> * prob_hierarchy = new
           FOSLSProblHierarchy<ProblemType, GeneralHierarchy>(*hierarchy, 1, *bdr_conds, *fe_formulat, prec_option, verbose);

   ProblemType * problem = prob_hierarchy->GetProblem(0);

#ifdef DIVFREE_ESTIMATOR
   FOSLSProblHierarchy<FOSLSDivfreeProblem, GeneralHierarchy> * divfreeprob_hierarchy = new
           FOSLSProblHierarchy<FOSLSDivfreeProblem, GeneralHierarchy>(*hierarchy, 1, *bdr_conds, *fe_formulat_divfree, prec_option, verbose);

   FOSLSDivfreeProblem * problem_divfree = divfreeprob_hierarchy->GetProblem(0);
#endif

   Hyper_test* Mytest = dynamic_cast<Hyper_test*>
           (problem->GetFEformulation().GetFormulation()->GetTest());
   MFEM_ASSERT(Mytest, "Unsuccessful cast into Hyper_test* \n");

   int numfoslsfuns = -1;

   int fosls_func_version = 2;
   if (fosls_func_version == 1)
       numfoslsfuns = 2;
   else if (fosls_func_version == 2)
       numfoslsfuns = 3;

   int numblocks_funct = 1;
   if (strcmp(space_for_S,"H1") == 0)
       ++numblocks_funct;

   Array<ParGridFunction*> extra_grfuns(0);
   if (fosls_func_version == 2)
   {
       extra_grfuns.SetSize(1);
   }

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

   // version 1, only || sigma - b S ||^2, or || K sigma ||^2
   if (fosls_func_version == 1)
   {
       // this works
       grfuns_descriptor[0] = std::make_pair<int,int>(1, 0);
       grfuns_descriptor[1] = std::make_pair<int,int>(1, 1);

       if (strcmp(space_for_sigma,"Hdiv") == 0) // sigma is from Hdiv
           integs(0,0) = new VectorFEMassIntegrator;
       else // sigma is from H1vec
           integs(0,0) = new ImproperVectorMassIntegrator;

       integs(1,1) = new MassIntegrator(*Mytest->GetBtB());

       if (strcmp(space_for_sigma,"Hdiv") == 0) // sigma is from Hdiv
           integs(1,0) = new VectorFEMassIntegrator(*Mytest->GetMinB());
       else // sigma is from H1
           integs(1,0) = new MixedVectorScalarIntegrator(*Mytest->GetMinB());
   }
   else if (fosls_func_version == 2)
   {
       // version 2, only || sigma - b S ||^2 + || div bS - f ||^2
       MFEM_ASSERT(strcmp(space_for_S,"H1") == 0, "Version 2 works only if S is from H1 \n");

       // this works
       grfuns_descriptor[0] = std::make_pair<int,int>(1, 0);
       grfuns_descriptor[1] = std::make_pair<int,int>(1, 1);
       grfuns_descriptor[2] = std::make_pair<int,int>(-1, 0);

       extra_grfuns[0] = new ParGridFunction(problem->GetPfes(numblocks - 1));
       extra_grfuns[0]->ProjectCoefficient(*problem->GetFEformulation().GetFormulation()->GetTest()->GetRhs());

       if (strcmp(space_for_sigma,"Hdiv") == 0) // sigma is from Hdiv
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

   FOSLSEstimator * estimator;

   // this works
   if (fosls_func_version == 2)
       estimator = new FOSLSEstimator(*problem, grfuns_descriptor, &extra_grfuns, integs, verbose);
   else
       estimator = new FOSLSEstimator(*problem, grfuns_descriptor, NULL, integs, verbose);

   problem->AddEstimator(*estimator);

   ThresholdRefiner refiner(*estimator);
   refiner.SetTotalErrorFraction(0.5);

#ifdef DIVFREE_ESTIMATOR
   std::vector<std::pair<int,int> > grfuns_descriptor_divfree(numblocks_funct);

   Array2D<BilinearFormIntegrator *> integs_divfree(numblocks_funct, numblocks_funct);
   for (int i = 0; i < integs_divfree.NumRows(); ++i)
       for (int j = 0; j < integs_divfree.NumCols(); ++j)
           integs_divfree(i,j) = NULL;

   /// Not needed after the discussion with Panayot. The error should
   /// be estimated by the initial error estimator for the problem
   /// involving the H(div) space
   FOSLSEstimator * estimator_divfree;

   // this works
   grfuns_descriptor_divfree[0] = std::make_pair<int,int>(1, 0);
   if (strcmp(space_for_S,"H1") == 0)
        grfuns_descriptor_divfree[1] = std::make_pair<int,int>(1, 1);

   if (strcmp(space_for_S,"H1") == 0)
   {
       integs_divfree(0,0) = new CurlCurlIntegrator;
       integs_divfree(1,1) = new H1NormIntegrator(*Mytest->GetBBt(), *Mytest->GetBtB());
       // untested integrator, actually
       integs_divfree(1,0) = new MixedVectorFECurlVQScalarIntegrator(*Mytest->GetMinB());
   }
   else
       integs_divfree(0,0) = new CurlCurlIntegrator(*Mytest->GetKtilda());

   estimator_divfree = new FOSLSEstimator(*problem_divfree, grfuns_descriptor_divfree, NULL, integs_divfree, verbose);

   problem_divfree->AddEstimator(*estimator_divfree);

   ThresholdRefiner refiner_divfree(*estimator_divfree);
   refiner_divfree.SetTotalErrorFraction(0.5);

   //MPI_Finalize();
   //return 0;

   //MPI_Finalize();
   //return 0;
#endif


   // 12. The main AMR loop. In each iteration we solve the problem on the
   //     current mesh, visualize the solution, and refine the mesh.
   const int max_dofs = 200000;//1600000;
   HYPRE_Int global_dofs = problem->GlobalTrueProblemSize();

   for (int it = 0; ; it++)
   {
       if (verbose)
       {
          cout << "\nAMR iteration " << it << "\n";
          cout << "Number of unknowns: " << global_dofs << "\n";
       }

       bool compute_error = true;

#ifdef DIVFREE_ESTIMATOR
       // finding a particular solution
       HypreParMatrix * B_hpmat = dynamic_cast<HypreParMatrix*>(&problem->GetOp()->GetBlock(2,0));
       Vector& div_rhs = problem->GetRhs().GetBlock(2);
       ParGridFunction * partsigma = FindParticularSolution(problem->GetPfes(0), *B_hpmat, div_rhs, verbose);
       BlockVector true_partsol(problem->GetTrueOffsets());
       true_partsol = 0.0;
       partsigma->ParallelProject(true_partsol.GetBlock(0));

       // creating the operator for the div-free problem
       problem_divfree->ConstructDivfreeHpMats();
       problem_divfree->CreateOffsetsRhsSol();
       const HypreParMatrix * divfree_hpmat = &problem_divfree->GetDivfreeHpMat();
       BlockOperator * problem_divfree_op = new BlockOperator(problem_divfree->GetTrueOffsets());
       HypreParMatrix * orig00 = dynamic_cast<HypreParMatrix*>(&problem->GetOp()->GetBlock(0,0));
       HypreParMatrix * blk00 = RAP(divfree_hpmat, orig00, divfree_hpmat);
       problem_divfree_op->SetBlock(0,0,blk00);

       HypreParMatrix * blk10, *blk01, *blk11;
       if (strcmp(space_for_S,"H1") == 0)
       {
           blk11 = CopyHypreParMatrix(*(dynamic_cast<HypreParMatrix*>(&problem->GetOp()->GetBlock(1,1))));

           HypreParMatrix * orig10 = dynamic_cast<HypreParMatrix*>(&problem->GetOp()->GetBlock(1,0));
           blk10 = ParMult(orig10, divfree_hpmat);

           blk01 = blk10->Transpose();

           problem_divfree_op->SetBlock(0,1,blk01);
           problem_divfree_op->SetBlock(1,0,blk10);
           problem_divfree_op->SetBlock(1,1,blk11);
       }
       problem_divfree_op->owns_blocks = true;

       problem_divfree->ResetOp(*problem_divfree_op);
       problem_divfree->InitSolver(verbose);

       // creating a preconditioner for the divfree problem
       problem_divfree->CreatePrec(*problem_divfree->GetOp(), 0, verbose);
       problem_divfree->UpdateSolverPrec();


       //  creating the solution and right hand side for the divfree problem
       BlockVector rhs(problem_divfree->GetTrueOffsets());

       BlockVector temp(problem->GetTrueOffsets());
       problem->GetOp()->Mult(true_partsol, temp);
       temp *= -1;
       temp += problem->GetRhs();

       divfree_hpmat->MultTranspose(temp.GetBlock(0), rhs.GetBlock(0));
       if (strcmp(space_for_S,"H1") == 0)
           rhs.GetBlock(1) = temp.GetBlock(1);

       problem_divfree->SolveProblem(rhs, verbose, false);

       /// converting the solution back into sigma from Hdiv inside the problem
       /// (adding a particular solution as a part of the process)
       /// and checking the accuracy of the resulting solution

       BlockVector& problem_sol = problem->GetSol();
       problem_sol = 0.0;

       BlockVector& problem_divfree_sol = problem_divfree->GetSol();

       problem_divfree->GetDivfreeHpMat().Mult(1.0, problem_divfree_sol.GetBlock(0), 1.0, problem_sol.GetBlock(0));
       if (strcmp(space_for_S,"H1") == 0)
           problem_sol.GetBlock(1) = problem_divfree_sol.GetBlock(1);

       problem_sol += true_partsol;

       if (compute_error)
           problem->ComputeError(problem_sol, verbose, true);

       // to make sure that problem has grfuns in correspondence with the problem_sol we compute here
       // though for now it coordination already happens in ComputeError()
       problem->DistributeToGrfuns(problem_sol);
#else
       if (compute_error)
           problem->Solve(verbose, compute_error);
#endif

       // 17. Send the solution by socket to a GLVis server.
       if (visualization)
       {
           ParGridFunction * sigma = problem->GetGrFun(0);
           ParGridFunction * S;
           if (strcmp(space_for_S,"H1") == 0)
               S = problem->GetGrFun(1);
           else
               S = (dynamic_cast<FOSLSProblem_HdivL2L2hyp*>(problem))->RecoverS();

           char vishost[] = "localhost";
           int  visport   = 19916;

           socketstream sigma_sock(vishost, visport);
           sigma_sock << "parallel " << num_procs << " " << myid << "\n";
           sigma_sock << "solution\n" << *pmesh << *sigma << "window_title 'sigma, AMR iter No."
                  << it <<"'" << flush;

           if (strcmp(space_for_S,"H1") == 0)
           {
               socketstream s_sock(vishost, visport);
               s_sock << "parallel " << num_procs << " " << myid << "\n";
               s_sock << "solution\n" << *pmesh << *S << "window_title 'S, AMR iter No."
                      << it <<"'" << flush;
           }
       }

       // 18. Call the refiner to modify the mesh. The refiner calls the error
       //     estimator to obtain element errors, then it selects elements to be
       //     refined and finally it modifies the mesh. The Stop() method can be
       //     used to determine if a stopping criterion was met.

       // new variant
       refiner.Apply(*prob_hierarchy->GetHierarchy().GetFinestParMesh());

       // old variant, before introducing the hierarchy stuff
       //refiner.Apply(*problem->GetParMesh());

       if (refiner.Stop())
       {
          if (verbose)
             cout << "Stopping criterion satisfied. Stop. \n";
          break;
       }

       bool recoarsen = true;
       MFEM_ABORT("The code fails.");
       prob_hierarchy->Update(recoarsen);

       // old variant
       //problem->Update();

#ifdef DIVFREE_ESTIMATOR
       // old variant
       //problem_divfree->Update();
       delete partsigma;
#endif

       MPI_Finalize();
       return 0;

       problem->BuildSystem(verbose);

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

