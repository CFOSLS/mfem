///                           MFEM(with 4D elements) for 3D/4D laplace equation
///                              in standard adaptive mesh refinement setting,
///                                    solved by preconditioned MINRES.
///
/// The problem considered in this example is
///                             laplace(u) = f
/// (either 3D or 4D, calling one of the variables time in space-time)
///
/// approximated either by standard H1 FEM (if H1FEMLAPLACE is #defined)
/// or by the mixed FEM in Hdiv-L2
///
/// The current 3D test is a regular solution in a cube.
///
/// The problem is discretized using RT, linear Lagrange and discontinuous constants in 3D/4D.
///
/// The problem is then solved with adaptive mesh refinement (AMR).
///
/// This example demonstrates usage of AMR related classes from mfem/cfosls/, such as
/// FOSLSEstimatorOnHier, FOSLSEstimator, etc.
/// The estimator is created on FOSLSEstimatorOnHier which represents an out-of-dated, not optimal way.
/// The newer approach is to construct the estimator from a "dynamic" problem built on the
/// mesh hierarchy (GeneralHierarchy) instead.
///
/// (**) This code was tested only in serial.
/// (***) The example was tested for memory leaks with valgrind, in 3D. A lot of
/// ""Conditional jump depends on an uninitialized value" was reported but not fixed.
///
/// Typical run of this example: ./cfosls_laplace_adref_Hcurl --whichD 3 -no-vis
///
/// Another examples on adaptive mesh refinement, with more complicated solvers, are
/// cfosls_laplace_adref_Hcurl_new.cpp and cfosls_hyperbolic_adref_Hcurl_new.cpp.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>

// if passive, the mesh is simply uniformly refined at each iteration
#define AMR

// activates using the solution at the previous mesh (its interpolant) as a starting
// guess for the next problem
#define CLEVER_STARTING_GUESS

// activates using a (simpler & cheaper) preconditioner for the problems, simple Gauss-Seidel
// turned out that it doesn't bring any benefit
//#define USE_GS_PREC

// if active, standard conforming H1 formulation for Laplace is used
//#define H1FEMLAPLACE

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

    int ser_ref_levels  = 2;
    int par_ref_levels  = 0;

    // 2. Define the problem to be solved (mixed or standard Laplace)

#ifdef H1FEMLAPLACE
    using FormulType = FOSLSFormulation_Laplace;
    using FEFormulType = FOSLSFEFormulation_Laplace;
    using BdrCondsType = BdrConditions_Laplace;
    using ProblemType = FOSLSProblem_Laplace;
#else
    using FormulType = CFOSLSFormulation_MixedLaplace;
    using FEFormulType = CFOSLSFEFormulation_MixedLaplace;
    using BdrCondsType = BdrConditions_MixedLaplace;
    using ProblemType = FOSLSProblem_MixedLaplace;
#endif

    // solver options
    int prec_option = 1; //defines whether to use preconditioner or not, and which one

    const char *mesh_file = "../data/cube_3d_moderate.mesh";

    int feorder         = 0;
#ifdef H1FEMLAPLACE
    feorder = (feorder > 0 ? feorder : 1);
#endif

    if (verbose)
        cout << "Solving (С)FOSLS laplace equation with MFEM & hypre \n";

    // 3. Parse command-line options.

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

    // Reporting whether certain flags are active (#defined) or not
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

    if (verbose)
        std::cout << "Number of mpi processes: " << num_procs << "\n";

    // 4. Reading the starting mesh and performing a prescribed number
    // of serial and parallel refinements

    //mesh_file = "../data/netgen_cylinder_mesh_0.1to0.2.mesh";
    //mesh_file = "../data/pmesh_cylinder_moderate_0.2.mesh";
    //mesh_file = "../data/pmesh_cylinder_fine_0.1.mesh";

    //mesh_file = "../data/pmesh_check.mesh";
    mesh_file = "../data/cube_3d_moderate.mesh";

    if (verbose)
        std::cout << "For the records: numsol = " << numsol
                  << ", mesh_file = " << mesh_file << "\n";


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

    // 5. Creating the problem instance and corresponding hierarchies of meshes and problems.

   FormulType * formulat = new FormulType (dim, numsol, verbose);
   FEFormulType * fe_formulat = new FEFormulType(*formulat, feorder);
   BdrConditions * bdr_conds = new BdrCondsType(*pmesh);

   bool with_hcurl = false;

   GeneralHierarchy * hierarchy = new GeneralHierarchy(1, *pmesh, feorder, verbose, with_hcurl);

   // Remark: In this example an old way of creating estimator is considered.
   // The estimator will be built on the problem hierarchy.
   // An easier way is to use a "dynamic" problem built on the hierarchy of meshes
   // Look at the newer way in cfosls_hyperbolic_adref_Hcurl_new.cpp
   FOSLSProblHierarchy<ProblemType, GeneralHierarchy> * prob_hierarchy = new
           FOSLSProblHierarchy<ProblemType, GeneralHierarchy>
           (*hierarchy, 1, *bdr_conds, *fe_formulat, prec_option, verbose);

   ProblemType * problem = prob_hierarchy->GetProblem(0);

   Laplace_test* Mytest = dynamic_cast<Laplace_test*>
           (problem->GetFEformulation().GetFormulation()->GetTest());
   MFEM_ASSERT(Mytest, "Unsuccessful cast into Hyper_test* \n");

   int numfoslsfuns = -1;

#ifdef H1FEMLAPLACE
       numfoslsfuns = 1;
#else
       numfoslsfuns = 2;
#endif

   // 6. Creating the FOSLS error estimator, first creating its components

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

   // Integrators for bilinear forms in the integrator
   Array2D<BilinearFormIntegrator *> integs(numfoslsfuns, numfoslsfuns);
   for (int i = 0; i < integs.NumRows(); ++i)
       for (int j = 0; j < integs.NumCols(); ++j)
           integs(i,j) = NULL;

#ifdef H1FEMLAPLACE
       // this works
       grfuns_descriptor[0] = std::make_pair<int,int>(1, 0);
       integs(0,0) = new DiffusionIntegrator;
#else
       // this works
       grfuns_descriptor[0] = std::make_pair<int,int>(1, 0);
       integs(0,0) = new VectorFEMassIntegrator;

       grfuns_descriptor[1] = std::make_pair<int,int>(1, 1);
       integs(1,1) = new DiffusionIntegrator;
       integs(0,1) = new MixedVectorGradientIntegrator;
#endif

   FOSLSEstimator * estimator;

   // See remark above. This way to define an estimator is not optimal.
   estimator = new FOSLSEstimatorOnHier<ProblemType, GeneralHierarchy>
           (*prob_hierarchy, 0, grfuns_descriptor, NULL, integs, verbose);

   // Adding estimator to the problem so that it get's updated whenever the
   // problem is updated
   problem->AddEstimator(*estimator);

   ThresholdRefiner refiner(*estimator);
   refiner.SetTotalErrorFraction(0.95); // 0.5

#ifdef CLEVER_STARTING_GUESS
   BlockVector * coarse_guess;
#endif

   // 7. The main AMR loop. In each iteration we solve the problem on the
   //     current mesh, visualize the solution, and refine the mesh, then repeat.
#ifdef AMR
   const int max_dofs = 200000;//1600000;
#else
   const int max_dofs = 400000;
#endif

   // Tolerance for the problem solver is being adjusted at each iteration
   const double fixed_rtol = 1.0e-15; // 1.0e-10; 1.0e-12;
   const double fixed_atol = 1.0e-5;

   const double initial_rtol = fixed_rtol;
   const double initial_atol = fixed_atol;
   double initial_res_norm = -1.0;

   double adjusted_atol = -1.0;

   HYPRE_Int global_dofs = problem->GlobalTrueProblemSize();
   std::cout << "starting n_el = " << prob_hierarchy->GetHierarchy().GetFinestParMesh()->GetNE() << "\n";

   if (verbose)
       std::cout << "Running AMR ... \n";

   // Main loop (with AMR or uniform refinement depending on the predefined macros)
   int max_iter_amr = 2; // 21;
   for (int it = 0; it < max_iter_amr ; it++)
   {
       if (verbose)
       {
          cout << "\nAMR iteration " << it << "\n";
          cout << "Number of unknowns: " << global_dofs << "\n\n";
       }

       bool compute_error = true;

#ifdef CLEVER_STARTING_GUESS
       // if it's not the first iteration we reuse the previous solution
       // by setting the starting guess to be the interpolant of the prev. solution
       if (it > 0)
           prob_hierarchy->GetTrueP(0)->Mult(*coarse_guess, problem->GetSol());

       // checking the residual
       BlockVector res(problem->GetTrueOffsets());
       problem->GetOp()->Mult(problem->GetSol(), res);
       res -= problem->GetRhs();//*rhs;

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

#ifdef USE_GS_PREC
       if (it > 0)
       {
           prec_option = 100;
           std::cout << "Resetting prec with the Gauss-Seidel preconditioners \n";
           problem->ResetPrec(prec_option);
       }
#endif

       // 12.1 Solving the problem at current iteration
#ifdef H1FEMLAPLACE
       // chaning the solver from MINRES to CG for standard FEM for Laplace in H^1
       problem->ChangeSolver(initial_rtol, adjusted_atol);
#endif
       problem->SolveProblem(problem->GetRhs(), problem->GetSol(), verbose, false);

       // checking the residual afterwards
       {
           BlockVector res(problem->GetTrueOffsets());
           problem->GetOp()->Mult(problem->GetSol(), res);
           res -= problem->GetRhs();//*rhs;

           double res_norm = ComputeMPIVecNorm(comm, res, "", false);
           if (verbose)
               std::cout << "Res norm after solving the problem at iteration # "
                         << it << " = " << res_norm << "\n";
       }

#else
       problem->Solve(verbose, false);
#endif

      // Computing the error
      BlockVector& problem_sol = problem->GetSol();
      if (compute_error)
      {
          problem->ComputeError(problem_sol, verbose, true);
          problem->ComputeBndError(problem_sol);
      }

      // some older code which survived, maybe even doesn't compile
#if 0
      // (special) testing cheaper preconditioners!
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
#endif

#ifdef CLEVER_STARTING_GUESS
       if (it > 0)
           delete coarse_guess;
       coarse_guess = new BlockVector(problem->GetTrueOffsets());
       *coarse_guess = problem_sol;
#endif

       // 12.2 Send the solution by socket to a GLVis server.
       if (visualization)
       {
           ParGridFunction * sigma = problem->GetGrFun(0);

           char vishost[] = "localhost";
           int  visport   = 19916;

           socketstream sigma_sock(vishost, visport);
           sigma_sock << "parallel " << num_procs << " " << myid << "\n";
           sigma_sock << "solution\n" << *pmesh << *sigma << "window_title 'sigma, AMR iter No."
                  << it <<"'" << flush;

           ParGridFunction * sigma_ex = new ParGridFunction(problem->GetPfes(0));
           BlockVector * exactsol_proj = problem->GetExactSolProj();
           sigma_ex->SetFromTrueDofs(exactsol_proj->GetBlock(0));

           socketstream sigmaex_sock(vishost, visport);
           sigmaex_sock << "parallel " << num_procs << " " << myid << "\n";
           sigmaex_sock << "solution\n" << *pmesh << *sigma_ex << "window_title 'sigma exact, AMR iter No."
                  << it <<"'" << flush;

           delete exactsol_proj;
           delete sigma_ex;
       }

       // 12.3 Call the refiner to modify the mesh. The refiner calls the error
       // estimator to obtain element errors, then it selects elements to be
       // refined and finally it modifies the mesh. The Stop() method can be
       // used to determine if a stopping criterion was met.

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

       // visualizing the vector of local errors
       if (visualization)
       {
           const Vector& local_errors = estimator->GetLocalErrors();

           FiniteElementCollection * l2_coll = new L2_FECollection(0, dim);

           ParFiniteElementSpace * L2_space = new ParFiniteElementSpace
                   (problem->GetParMesh(), l2_coll);
           ParGridFunction * local_errors_pgfun = new ParGridFunction(L2_space);
           local_errors_pgfun->SetFromTrueDofs(local_errors);
           char vishost[] = "localhost";
           int  visport   = 19916;

           socketstream amr_sock(vishost, visport);
           amr_sock << "parallel " << num_procs << " " << myid << "\n";
           amr_sock << "solution\n" << *pmesh << *local_errors_pgfun <<
                         "window_title 'local errors, AMR iter No." << it <<"'" << flush;

           delete l2_coll;
           delete L2_space;
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

       // 12.4 Updating the problem hierarchy, and its finest level problem
       bool recoarsen = true;
       prob_hierarchy->Update(recoarsen);
       problem = prob_hierarchy->GetProblem(0);

       // checking #dofs after the refinement
       global_dofs = problem->GlobalTrueProblemSize();

       if (global_dofs > max_dofs)
       {
          if (verbose)
             cout << "Reached the maximum number of dofs. Stop. \n";
          break;
       }

   }

   // 13. Deallocating the memory

   delete coarse_guess;

   delete hierarchy;
   delete prob_hierarchy;

   for (int i = 0; i < integs.NumRows(); ++i)
       for (int j = 0; j < integs.NumCols(); ++j)
           if (integs(i,j))
               delete integs(i,j);

   delete estimator;

   delete bdr_conds;
   delete formulat;
   delete fe_formulat;


   MPI_Finalize();
   return 0;
}

