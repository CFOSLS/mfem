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

#define DEBUGGING_CASE

#define FOSLS

// doesn't work as designed, || f_H - P^T f_h || is still nonzero
//#define USEALWAYS_COARSE_RHS

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
    int numsol          = -9;

    int ser_ref_levels  = 3;
    int par_ref_levels  = 0;

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

   FormulType * formulat = new FormulType (dim, numsol, verbose);
   FEFormulType * fe_formulat = new FEFormulType(*formulat, feorder);
   BdrConditions * bdr_conds = new BdrCondsType(*pmesh);

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

   bool with_hcurl = false;

   GeneralHierarchy * hierarchy = new GeneralHierarchy(1, *pmesh, feorder, verbose, with_hcurl);
   FOSLSProblHierarchy<ProblemType, GeneralHierarchy> * prob_hierarchy = new
           FOSLSProblHierarchy<ProblemType, GeneralHierarchy>
           (*hierarchy, 1, *bdr_conds, *fe_formulat, prec_option, verbose);

   //ComponentsDescriptor descriptor(true, true, true, true, true, false);
   //MultigridToolsHierarchy * mgtools_hierarchy = new MultigridToolsHierarchy(*hierarchy, 0, descriptor);

   ProblemType * problem = prob_hierarchy->GetProblem(0);

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

#ifdef CLEVER_STARTING_GUESS
   BlockVector * coarse_guess;
#endif

#ifdef DEBUGGING_CASE
   Vector * rhs;
   Vector * checkdiff;
   Vector * coarse_rhs;
#endif

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

#ifdef DEBUGGING_CASE

#ifdef USEALWAYS_COARSE_RHS
       if (it == 0)
       {
           rhs = new Vector(problem->GetRhs().GetBlock(0).Size());
           *rhs = problem->GetRhs().GetBlock(0);
       }
#else
       rhs = &problem->GetRhs().GetBlock(0);
#endif

       if (verbose && it == 0)
           std::cout << "rhs norm = " << rhs->Norml2() / sqrt (rhs->Size()) << "\n";
       if (it == 0)
       {
           checkdiff = new Vector(rhs->Size());
           *checkdiff = *rhs;
       }

       if (it == 0)
       {
           coarse_rhs = new Vector(rhs->Size());
           *coarse_rhs = *rhs;
       }

       if (it > 0)
       {
           Vector temp(hierarchy->GetTruePspace(SpaceName::H1,0)->Width());
           hierarchy->GetTruePspace(SpaceName::H1,0)->MultTranspose(*rhs, temp);
           *checkdiff -= temp;

           //checkdiff->Print();

           if (verbose)
               std::cout << "|| f_H - P^T f_h || = " <<
                            checkdiff->Norml2() / sqrt (checkdiff->Size()) << "\n";

           delete checkdiff;
           checkdiff = new Vector(rhs->Size());
           *checkdiff = *rhs;

           /*
           *checkdiff += temp;
           Vector temp2;
           Vector finer_buff;
           // temp2 = Qh_1 f_h, living on the fine level
           partsol_finder->NewProjectFinerL2ToCoarser(0, div_rhs, temp2, finer_buff);

           // temp = P^T  * temp2
           hierarchy->GetTruePspace(SpaceName::L2,0)->MultTranspose(temp2, temp);
           *checkdiff -= temp;

           if (verbose)
               std::cout << "|| f_H - P^T Qh_1 f_h || " <<
                            checkdiff->Norml2() / sqrt (checkdiff->Size()) << "\n";
           */
       }

       //if (it == 2)
       //{
           //MPI_Finalize();
           //return 0;
       //}
#endif


#ifdef CLEVER_STARTING_GUESS
       // if it's not the first iteration we reuse the previous solution as a starting guess
       if (it > 0)
           prob_hierarchy->GetTrueP(0)->Mult(*coarse_guess, problem->GetSol());

       // checking the residual
       BlockVector res(problem->GetTrueOffsets());
       problem->GetOp()->Mult(problem->GetSol(), res);
       res -= *rhs;

       double res_norm = ComputeMPIVecNorm(comm, res, "", false);
       if (it == 0)
       {
           initial_res_norm = res_norm;
           //initial_atol = initial_res_norm * initial_rtol;
       }

       if (verbose)
       {
           std::cout << "Initial res norm at iteration # " << it << " = " << res_norm << "\n";
           // checking theoretical decomposition of the residual
#if 0
           if (it > 0)
           {
               int fine_size = res.Size();
               int coarse_size = prob_hierarchy->GetTrueP(0)->Width();

               Vector mass_fine_diag;
               ParBilinearForm mass_fine(hierarchy->GetSpace(SpaceName::H1,0));
               mass_fine.AddDomainIntegrator(new MassIntegrator);
               mass_fine.Assemble();
               mass_fine.Finalize();
               SparseMatrix * mass_fine_spmat = mass_fine.LoseMat();
               mass_fine_spmat->GetDiag(mass_fine_diag);

               SparseMatrix * PtWP = mfem::RAP(*hierarchy->GetPspace(SpaceName::H1,0), *mass_fine_spmat,
                                               *hierarchy->GetPspace(SpaceName::H1,0));
               Vector PtWP_diag;
               PtWP->GetDiag(PtWP_diag);

               SparseMatrix * Pt = Transpose(*hierarchy->GetPspace(SpaceName::H1,0));
               SparseMatrix * PtP = mfem::Mult(*Pt, *hierarchy->GetPspace(SpaceName::H1,0));
               //PtP->Print();
               Vector PtP_diag;
               PtP->GetDiag(PtP_diag);

               Vector tempc(coarse_size);
               Vector tempc2(coarse_size);
               Vector tempf1(fine_size);
               Vector tempf2(fine_size);

               // sub-check
               {
                   Vector tmp1(fine_size);

                   hierarchy->GetPspace(SpaceName::H1,0)->MultTranspose(res, tempc);

                   /*
                   for (int i = 0; i < tempc.Size(); ++i)
                       tempc[i] /= PtWP_diag[i];

                   // tempf1 = P * (Pt W P)^{-1} * P^T res
                   hierarchy->GetPspace(SpaceName::H1,0)->Mult(tempc, tempf1);

                   // tmp1 = W * P * (Pt W P)^{-1} * P^T * res
                   mass_fine_spmat->Mult(tempf1, tmp1);
                   */

                   // tempc = (Pt P)^{-1} * P * res
                   for (int i = 0; i < tempc.Size(); ++i)
                       tempc[i] /= PtP_diag[i];

                   // tmp1 = P * (Pt P)^{-1} * P^T res
                   hierarchy->GetPspace(SpaceName::H1,0)->Mult(tempc, tmp1);

                   Vector tmp2(fine_size);
                   tmp2 = res;
                   // tmp2 = (I - Q) res = res - tmp1
                   tmp2 -= tmp1;

                   double orto_check = tmp1 * tmp2;
                   if (fabs(orto_check) > 1.0e-12)
                       std::cout << "(tmp1, tmp2) = " << orto_check << "\n";
                   //MFEM_ASSERT(fabs(orto_check) < 1.0e-12, "Orthogonality sub-check failed");

                   Vector Wtmp1(fine_size);
                   mass_fine_spmat->Mult(tmp1, Wtmp1);

                   double orto_check2 = Wtmp1 * tmp2;
                   if (fabs(orto_check2) > 1.0e-12)
                       std::cout << "(W * tmp1, tmp2) = " << orto_check2 << "\n";
                   //MFEM_ASSERT(fabs(orto_check2) < 1.0e-12, "Orthogonality sub-check failed");

                   std::cout << "|| Q res || = " << tmp1.Norml2() / sqrt(tmp1.Size()) << "\n";
                   std::cout << "|| (I - Q) res || = " << tmp2.Norml2() / sqrt(tmp2.Size()) << "\n";
               }

               Vector term1(fine_size);
               tempc = *coarse_rhs;
               hierarchy->GetPspace(SpaceName::H1,0)->MultTranspose(*rhs, tempc2);
               // tempc = f_H - P^T f_h
               tempc -= tempc2;

               // tempc = (Pt W P)^{-1}  * (f_H - P^T f_h)
               for (int i = 0; i < tempc.Size(); ++i)
                   tempc[i] /= PtWP_diag[i];

               // tempf = P * (Pt W P)^{-1} * (f_H - P^T f_h)
               prob_hierarchy->GetTrueP(0)->Mult(tempc, tempf1);

               // tempf2 = W * P * (Pt W P)^{-1} * (f_H - P^T f_h)
               mass_fine_spmat->Mult(tempf1, tempf2);

               term1 = tempf2;

               /*
               // tempc = (Pt P)^{-1}  * (f_H - P^T f_h)
               for (int i = 0; i < tempc.Size(); ++i)
                   tempc[i] /= PtP_diag[i];

               // term1 = P * (Pt P)^{-1} * (f_H - P^T f_h)
               prob_hierarchy->GetTrueP(0)->Mult(tempc, term1);
               */

               std::cout << "Norm of term 1 = " << term1.Norml2() / sqrt (term1.Size()) << "\n";

               Vector term2(fine_size);
               term2 = res;
               // tempc = Pt * res
               hierarchy->GetPspace(SpaceName::H1,0)->MultTranspose(term2, tempc);

               // tempc = (Pt W P)^{-1} * Pt * res
               for (int i = 0; i < tempc.Size(); ++i)
                   tempc[i] /= PtWP_diag[i];

               // tempf = P * (Pt W P)^{-1} * Pt * res
               hierarchy->GetPspace(SpaceName::H1,0)->Mult(tempc, tempf1);

               // tempf2 = W * P * (Pt W P)^{-1} * Pt * res
               mass_fine_spmat->Mult(tempf1, tempf2);

               term2 -= tempf2;

               /*
               // tempc = (Pt P)^{-1} * Pt * res
               for (int i = 0; i < tempc.Size(); ++i)
                   tempc[i] /= PtP_diag[i];

               // tempf1 = P * (Pt P)^{-1} * Pt * res
               hierarchy->GetPspace(SpaceName::H1,0)->Mult(tempc, tempf1);

               term2 -= tempf1;
               */

               std::cout << "Norm of term 2 = " << term2.Norml2() / sqrt (term2.Size()) << "\n";

               double orto_check = term1 * term2;
               if (fabs(orto_check) > 1.0e-12)
                   std::cout << "(term1, term2) = " << orto_check << "\n";
               MFEM_ASSERT(fabs(orto_check) < 1.0e-12, "Orthogonality check failed");

               Vector check(fine_size);
               check = term1;
               check += term2;
               check -= res;
               MFEM_ASSERT(check.Norml2() / sqrt (check.Size()) < 1.0e-13, "Something went wrong");


               delete mass_fine_spmat;
               delete PtWP;
               delete Pt;
               delete PtP;
           }
#endif
           std::cout << "Initial relative tolerance: " << initial_rtol << "\n";
           std::cout << "Initial absolute tolerance: " << initial_atol << "\n";
       }

       //adjusted_atol = initial_atol / 2.0 ;
       adjusted_atol = initial_atol / pow(2, it);
       if (it == 0)
       {
           problem->SetRelTol(initial_rtol);
           problem->SetAbsTol(initial_atol);
       }
       else
       {
           problem->SetRelTol(1.0e-18);
           problem->SetAbsTol(adjusted_atol);

           if (verbose)
               std::cout << "adjusted atol = " << adjusted_atol << "\n";
       }

       //double adjusted_rtol = fixed_rtol * initial_res_norm / res_norm;
       //if (verbose)
           //std::cout << "adjusted rtol = " << adjusted_rtol << "\n";

#ifdef USE_GS_PREC
       if (it > 0)
       {
           prec_option = 100;
           std::cout << "Resetting prec with the Gauss-Seidel preconditioners \n";
           problem->ResetPrec(prec_option);
       }
#endif

       //std::cout << "checking rhs norm for the first solve: " <<
                    //problem->GetRhs().Norml2() /  sqrt (problem->GetRhs().Size()) << "\n";

       problem->ChangeSolver(initial_rtol, adjusted_atol);
       problem->SolveProblem(*rhs, problem->GetSol(), verbose, false);

       // checking the residual afterwards
       {
           BlockVector res(problem->GetTrueOffsets());
           problem->GetOp()->Mult(problem->GetSol(), res);
           res -= *rhs;

           double res_norm = ComputeMPIVecNorm(comm, res, "", false);
           if (verbose)
               std::cout << "Res norm after solving the problem at iteration # "
                         << it << " = " << res_norm << "\n";
       }

#else
       problem->Solve(verbose, false);
#endif

#ifdef DEBUGGING_CASE
       delete coarse_rhs;
       coarse_rhs = new Vector(rhs->Size());
       *coarse_rhs = *rhs;
#endif

      BlockVector& problem_sol = problem->GetSol();
      if (compute_error)
      {
          problem->ComputeError(problem_sol, verbose, true);
          problem->ComputeBndError(problem_sol);
      }

      // (special) testing cheaper preconditioners!
      /*
      if (verbose)
          std::cout << "Performing a special check for the preconditioners iteration counts! \n";

      prec_option = 100;
      problem->ResetPrec(prec_option);

      BlockVector special_guess(problem->GetTrueOffsets());
      special_guess = problem->GetSol();

      int special_num = 1;
      Array<int> el_indices(special_num);
      for (int i = 0; i < special_num; ++i)
          el_indices[i] = problem->GetParMesh()->GetNE() / 2 + i;

      std::cout << "Number of elements where the sol was changed: " <<
                   special_num << "(" <<  special_num * 100.0 /
                   problem->GetParMesh()->GetNE() << "%) \n";

      for (int blk = 0; blk < problem->GetFEformulation().Nblocks(); ++blk)
      {
          ParFiniteElementSpace * pfes = problem->GetPfes(blk);

          Array<int> dofs;
          MFEM_ASSERT(num_procs == 1, "This works only in serial");

          for (int elind = 0; elind < el_indices.Size(); ++elind)
          {
              //std::cout << "el index = " << el_indices[elind] << " (out of " <<
                           //pfes->GetParMesh()->GetNE() << ") \n";
              pfes->GetElementDofs(el_indices[elind], dofs);

              for (int i = 0; i < dofs.Size(); ++i)
              {
                  int ldof = dofs[i];
                  if (dofs[i] < 0)
                  {
                      ldof = -1 - dofs[i];
                      //std::cout << "corrected ldof = " << ldof << "\n";
                  }

                  //int dof_sign = pfes->GetDofSign(dofs[i]);
                  //int ldof = pfes->GetLocalTDofNumber(dofs[i]);
                  if (ldof > -1)
                  {
                      //special_guess.GetBlock(blk)[ldof] = 0.0;
                      special_guess.GetBlock(blk)[ldof] =
                        problem->GetSol().GetBlock(blk)[ldof] * 0.9;
                  }

              }
          }
      }

      BlockVector check_diff(problem->GetTrueOffsets());
      check_diff = special_guess;
      check_diff -= problem->GetSol();
      double check_diff_norm = ComputeMPIVecNorm(comm, check_diff, "", false);

      if (verbose)
          std::cout << "|| sol - special_guess || = " << check_diff_norm << "\n";

      int nnz_count = 0;
      for (int i = 0; i < check_diff.Size(); ++i)
          if (fabs(check_diff[i]) > 1.0e-8)
              ++nnz_count;

      if (verbose)
          std::cout << "nnz_count in the diff = " << nnz_count << "\n";

#if 0
      {
          std::cout << "2 check \n";
          BlockDiagonalPreconditioner * prec_pt = dynamic_cast<BlockDiagonalPreconditioner*>(problem->GetPrec());
          Vector tmp1(prec_pt->GetDiagonalBlock(0).Width());
          tmp1 = 1.0;
          Vector tmp2(prec_pt->GetDiagonalBlock(0).Height());
          prec_pt->GetDiagonalBlock(0).Mult(tmp1, tmp2);
          std::cout << "tmp2 norm = " << tmp2.Norml2() << "\n";

          Vector tmp3(prec_pt->GetDiagonalBlock(1).Width());
          tmp3 = 1.0;
          Vector tmp4(prec_pt->GetDiagonalBlock(1).Height());
          prec_pt->GetDiagonalBlock(1).Mult(tmp3, tmp4);
          std::cout << "tmp4 norm = " << tmp4.Norml2() << "\n";

      }
#endif

      {
          // checking the residual
          BlockVector res(problem->GetTrueOffsets());
          problem->GetOp()->Mult(special_guess, res);
          res -= problem->GetRhs();

          double res_norm = ComputeMPIVecNorm(comm, res, "", false);

          if (verbose)
              std::cout << "Initial res norm for the second solve = " << res_norm << "\n";

          double adjusted_rtol;
          // 1st: strategy of keeping the relative tolerance constant throughout levels
          adjusted_rtol = initial_rtol * initial_res_norm / res_norm;
          if (verbose)
              std::cout << "adjusted rtol due to the initial res change = " << adjusted_rtol << "\n";

          // 2nd: strategy of decreasing absolute tolerance with iteration count
          adjusted_atol = initial_atol * 1.0 / 2;
          if (verbose)
              std::cout << "adjusted atol = " << adjusted_atol << "\n";

          problem->SetRelTol(adjusted_rtol);
          problem->SetAbsTol(adjusted_atol);
      }

      std::cout << "checking rhs norm for the second solve: " <<
                   problem->GetRhs().Norml2() /  sqrt (problem->GetRhs().Size()) << "\n";
      problem->ChangeSolver();
      problem->SolveProblem(problem->GetRhs(), special_guess, verbose, false);

      //problem_sol = problem->GetSol();
      //if (compute_error)
          //problem->ComputeError(problem_sol, verbose, true);

      MPI_Finalize();
      return 0;
      */

#ifdef CLEVER_STARTING_GUESS
       if (it > 0)
           delete coarse_guess;
       coarse_guess = new BlockVector(problem->GetTrueOffsets());
       *coarse_guess = problem_sol;
#endif

       // 17. Send the solution by socket to a GLVis server.
       if (visualization)
       {
           ParGridFunction * sigma = problem->GetGrFun(0);

           char vishost[] = "localhost";
           int  visport   = 19916;

           socketstream sigma_sock(vishost, visport);
           sigma_sock << "parallel " << num_procs << " " << myid << "\n";
           sigma_sock << "solution\n" << *pmesh << *sigma << "window_title 'sigma, AMR iter No."
                  << it <<"'" << flush;
       }

       // 18. Call the refiner to modify the mesh. The refiner calls the error
       //     estimator to obtain element errors, then it selects elements to be
       //     refined and finally it modifies the mesh. The Stop() method can be
       //     used to determine if a stopping criterion was met.

#ifdef AMR
       int nel_before = prob_hierarchy->GetHierarchy().GetFinestParMesh()->GetNE();

       // testing with only 1 element marked for refinement
       //Array<int> els_to_refine(1);
       //els_to_refine = prob_hierarchy->GetHierarchy().GetFinestParMesh()->GetNE() / 2;
       //prob_hierarchy->GetHierarchy().GetFinestParMesh()->GeneralRefinement(els_to_refine);

       // true AMR
       refiner.Apply(*prob_hierarchy->GetHierarchy().GetFinestParMesh());
       int nmarked_el = refiner.GetNumMarkedElements();
       if (verbose)
       {
           std::cout << "Marked elements percentage = " << 100 * nmarked_el * 1.0 / nel_before << " % \n";
           std::cout << "nmarked_el = " << nmarked_el << ", nel_before = " << nel_before << "\n";
           int nel_after = prob_hierarchy->GetHierarchy().GetFinestParMesh()->GetNE();
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
       prob_hierarchy->GetHierarchy().GetFinestParMesh()->UniformRefinement();
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

#ifdef DEBUGGING_CASE
#ifdef USEALWAYS_COARSE_RHS
       Vector tempvec(rhs->Size());
       tempvec = *rhs;
       delete rhs;
       rhs = new Vector(problem->GetRhs().GetBlock(0).Size());
       Vector mass_coarse_diag;
       ParBilinearForm mass_coarse(hierarchy->GetSpace(SpaceName::H1,1));
       mass_coarse.AddDomainIntegrator(new MassIntegrator);
       mass_coarse.Assemble();
       mass_coarse.Finalize();
       SparseMatrix * mass_coarse_spmat = mass_coarse.LoseMat();
       mass_coarse_spmat->GetDiag(mass_coarse_diag);

       for (int i = 0; i < tempvec.Size(); ++i)
           tempvec[i] /= mass_coarse_diag[i];
       prob_hierarchy->GetHierarchy().GetTruePspace(SpaceName::H1, 0)->Mult(tempvec, *rhs);

       Vector mass_fine_diag;
       ParBilinearForm mass_fine(hierarchy->GetSpace(SpaceName::H1,0));
       mass_fine.AddDomainIntegrator(new MassIntegrator);
       mass_fine.Assemble();
       mass_fine.Finalize();
       SparseMatrix * mass_fine_spmat = mass_fine.LoseMat();
       mass_fine_spmat->GetDiag(mass_fine_diag);

       for (int i = 0; i < rhs->Size(); ++i)
           (*rhs)[i] *= mass_fine_diag[i];

       delete mass_coarse_spmat;
       delete mass_fine_spmat;
#endif
#endif


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

