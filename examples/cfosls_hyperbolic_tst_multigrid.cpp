///                       CFOSLS formulation for transport equation in 3D/4D
///                         solved with a two-grid parallel-in-time multigrid
///                             (similar to the idea of parareal)
/// The problem considered in this example is
///                             du/dt + b * u = f (either 3D or 4D in space-time)
/// casted in the CFOSLS formulation
/// 1) either in Hdiv-L2 case:
///                             (K sigma, sigma) -> min
/// where sigma is from H(div), u is recovered (as an element of L^2) from sigma = b * u,
/// and K = (I - bbT / || b ||)
/// 2) or in Hdiv-H1-L2 case
///                             || sigma - b * u || ^2 -> min
/// where sigma is from H(div) and u is from H^1
/// minimizing in both cases under the constraint
///                             div sigma = f.
///
/// The problem is discretized using RT, Lagrange and discontinuous constants in 3D/4D.
///
/// The problem is then solved using a parallel-in-time two-grid method.
/// First, the entire domain is divided into non-overlapping time slabs.
/// In each time slab there are two space-time meshes, a fine and a coarse one.
///
/// The method can be considered as a multigrid (two-grid) with the following components:
///
/// 1) Smoother = parallel fine-level solve in each time slab.
/// It takes a vector in the entire domain, extracts the values at the interfaces between
/// time slabs and solves the problem within each time slab, using the interface value as initial
/// condition.

/// 2) Coarse solver = sequential time-stepping (time slab, after time slab, taking the initial condition
/// in each time slab from the previous one) at the coarse level.

/// 3) Standard interpolation (defined as a standard conforming interpolation within each time slab)

/// 4) Operator of the problem is a sequential fine-grid time-stepping.
///
/// Another point of view on the algorithm:
/// The fine-grid time-stepping can be written in the block two-diagonal matrix form:
///     ( L_0   & 0     &  0  & ... )
///     ( J_0,1 & L_1   &  0  & ... )
/// A = (   0   & J_1,2 & L_2 &  0  )
///     (                 ... & ... )
///
///                                     A * x = f,
///
/// where x and f are defined in the entire domain as an array of vector for each time slab
/// (which store values for the interfaces between time slabs twice).
///
/// Here L_i is the operator within i-th time slab, which doesn't touch initial condition
/// J_i,i+1 is the operator which accounts for the nonzero initial condition.
///
/// Then one can consider this matrix operator at two levels, a fine and a coarse one.
/// The coarse one is defined as P^T A P, where the interpolation operator is constructed as
/// P = diag (P_0, P_1, ... ) where P_i is the standrad interpolation operator for the problem
/// in the i-th time slab.
/// The smoother is then simply the diag(A) and we run a "geometric" multigrid with these
/// components.
///
/// AFAIK, I don't know results which justifies this method theoretically, especially for the
/// underlying transposrt equation. If you do, please let me know.
///
/// This example demonstrates usage of time-slabbing related classes from mfem/cfosls/, such as
/// ParMeshCyl, FOSLSCylProblem, TimeStepping<T>, GeneralCylHierarchy, FOSLSCylProblHierarchy,
/// TwoGridTimeStepping<T>, etc.
///
/// (*) MG cycle in this code means in fact a two-grid method. It's not a regular multigrid and
/// the current implementation is developed strictly for the two-grid case.
/// The extension would be straight-forward though.
///
/// (**) Mostly, the code was tested in serial, although in the end it was checked in parallel.
/// (***) The example was tested for memory leaks with valgrind, in Hdiv-L2 formulation, 3D/4D.
///
/// Typical run of this example: ./cfosls_hyperbolic_tst_multigrid --whichD 3 -no-vis
/// If you ant Hdiv-H1-L2 formulation, you will need not only change --spaceS option but also
/// change the source code, around 4.
///
/// Another example on time-slabbing technique, with a standard time-stepping is
/// cfosls_hyperbolic_timestepping.cpp.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>
#include <unistd.h>

// (if active) using a test with nonhomogeneous initial condition
#define NONHOMO_TEST

// must be active (activates construction of the mesh from the base (lower-dimensional mesh),
// making it a cylinder
#define USE_TSL

using namespace std;
using namespace mfem;

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
   int numsol          = 0;

   int ser_ref_levels  = 1;
   int par_ref_levels  = 0;

   // 2. Parse command-line options.

   // filename for the input mesh, is used only if USE_TSL is not defined
   const char *mesh_file = "../data/star.mesh";
#ifdef USE_TSL
   // filename for the input base mesh
   const char *meshbase_file = "../data/star.mesh";
   // number of time steps (which define the time slabs)
   int Nt = 4;
   double tau = 0.25;
#endif

   const char *space_for_S = "H1";     // "H1" or "L2"
   const char *space_for_sigma = "Hdiv"; // "Hdiv" or "H1" (H1 not tested a while)

   // defines whether to use preconditioner or not, and which one
   int prec_option = 1;

   int feorder = 0;

   bool visualization = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
#ifdef USE_TSL
   args.AddOption(&meshbase_file, "-mbase", "--meshbase",
                  "Mesh base file to use.");
   args.AddOption(&Nt, "-nt", "--nt",
                  "Number of time slabs.");
   args.AddOption(&tau, "-tau", "--tau",
                  "Height of each time slab ~ timestep.");
#endif
   args.AddOption(&ser_ref_levels, "-sref", "--sref",
                  "Number of serial refinements 4d mesh.");
   args.AddOption(&par_ref_levels, "-pref", "--pref",
                  "Number of parallel refinements 4d mesh.");
   args.AddOption(&nDimensions, "-dim", "--whichD",
                  "Dimension of the space-time problem.");
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
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
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

   MFEM_ASSERT(strcmp(space_for_S,"H1") == 0 || strcmp(space_for_S,"L2") == 0,
               "Space for S must be H1 or L2!\n");
   MFEM_ASSERT(strcmp(space_for_sigma,"Hdiv") == 0 || strcmp(space_for_sigma,"H1") == 0,
               "Space for sigma must be Hdiv or H1!\n");

   MFEM_ASSERT(!strcmp(space_for_sigma,"H1") == 0 || (strcmp(space_for_sigma,"H1") == 0 && strcmp(space_for_S,"H1") == 0),
               "Sigma from H1vec must be coupled with S from H1!\n");

#ifdef NONHOMO_TEST
   if (nDimensions == 3)
       numsol = -33;
   else // 4D case
       numsol = -44;
#else
   if (nDimensions == 3)
       numsol = -3;
   else // 4D case
       numsol = -4;
#endif

   // 3. Createing a cylinder mesh from the given base mesh
   // and perform a prescribed number of serial and parallel refinements
#ifdef USE_TSL
   if (verbose)
       std::cout << "USE_TSL is active (the cylinder mesh is constructed using mesh generator) \n";

   if (nDimensions == 3)
   {
       meshbase_file = "../data/square_2d_moderate.mesh";
   }
   else // 4D case
   {
       meshbase_file = "../data/cube_3d_moderate.mesh";
       //meshbase_file = "../data/cube_3d_small.mesh";
   }

   Mesh *meshbase = NULL;
   ifstream imesh(meshbase_file);
   if (!imesh)
   {
       std::cerr << "\nCan not open mesh base file: " << meshbase_file << '\n' << std::endl;
       MPI_Finalize();
       return -2;
   }
   else
   {
       if (verbose)
            std::cout << "meshbase_file: " << meshbase_file << "\n";
       meshbase = new Mesh(imesh, 1, 1);
       imesh.close();
   }

   for (int l = 0; l < ser_ref_levels; l++)
       meshbase->UniformRefinement();

   meshbase->CheckElementOrientation(true);

   ParMesh * pmeshbase = new ParMesh(comm, *meshbase);
   for (int l = 0; l < par_ref_levels; l++)
       pmeshbase->UniformRefinement();

   pmeshbase->CheckElementOrientation(true);

   //if (verbose)
       //std::cout << "pmeshbase shared structure \n";
   //pmeshbase->PrintSharedStructParMesh();

   delete meshbase;

   // Actually, pmesh is not used in the time-stepping code
   // What is used, it is timeslabs_pmeshcyls, see 5.
   ParMeshCyl * pmesh = new ParMeshCyl(comm, *pmeshbase, 0.0, tau, Nt);

   //if (verbose)
       //std::cout << "pmesh shared structure \n";
   //pmesh->PrintSharedStructParMesh();

#else
   if (verbose)
       std::cout << "USE_TSL is deactivated \n";
   if (nDimensions == 3)
   {
       mesh_file = "../data/cube_3d_moderate.mesh";
       //mesh_file = "../data/two_tets.mesh";
   }
   else // 4D case
   {
       mesh_file = "../data/pmesh_tsl_1proc.mesh";
       //mesh_file = "../data/cube4d_96.MFEM";
       //mesh_file = "../data/two_pentatops.MFEM";
       //mesh_file = "../data/two_pentatops_2.MFEM";
   }

   if (verbose)
   {
       std::cout << "mesh_file: " << mesh_file << "\n";
       std::cout << "Number of mpi processes: " << num_procs << "\n" << std::flush;
   }

   Mesh *mesh = NULL;

   ParMesh * pmesh;

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
   else //if nDimensions is not 3 or 4
   {
       if (verbose)
           cerr << "Case nDimensions = " << nDimensions << " is not supported \n" << std::flush;
       MPI_Finalize();
       return -1;
   }

   if (mesh) // if only serial mesh was generated previously, parallel mesh is initialized here
   {
       for (int l = 0; l < ser_ref_levels; l++)
           mesh->UniformRefinement();

       if (verbose)
           cout << "Creating parmesh(" << nDimensions <<
                   "d) from the serial mesh (" << nDimensions << "d)" << endl << flush;
       pmesh = new ParMesh(comm, *mesh);
       for (int l = 0; l < par_ref_levels; l++)
           pmesh->UniformRefinement();
       delete mesh;
   }

#endif

   int dim = nDimensions;

   MPI_Barrier(comm);
   std::cout << std::flush;
   MPI_Barrier(comm);

   // 4. Define the problem to be solved (CFOSLS Hdiv-L2 or Hdiv-H1 formulation, e.g., here)

   // Hdiv-H1 case
   using FormulType = CFOSLSFormulation_HdivH1Hyper;
   using FEFormulType = CFOSLSFEFormulation_HdivH1Hyper;
   using BdrCondsType = BdrConditions_CFOSLS_HdivH1_Hyper;
   using ProblemType = FOSLSCylProblem_HdivH1L2hyp;
   MFEM_ASSERT(strcmp(space_for_S,"H1") == 0, "Space for S must be H1 in this case!\n");

   /*
   // Hdiv-L2 case
   using FormulType = CFOSLSFormulation_HdivL2Hyper;
   using FEFormulType = CFOSLSFEFormulation_HdivL2Hyper;
   using BdrCondsType = BdrConditions_CFOSLS_HdivL2_Hyper;
   using ProblemType = FOSLSCylProblem_HdivL2hyp;
   MFEM_ASSERT(strcmp(space_for_S,"L2") == 0, "Space for S must be L2 in this case!\n");
   */

   FormulType * formulat = new FormulType (dim, numsol, verbose);
   FEFormulType * fe_formulat = new FEFormulType(*formulat, feorder);
   BdrConditions * bdr_conds = new BdrCondsType(*pmesh);

   // if we wanted to solve the problem in the entire domain, we could have used this
   /*
   ProblemType * problem = new ProblemType
           //(*pmesh, *bdr_conds, *fe_formulat, prec_option, verbose);

   problem->Solve(verbose);

   delete problem;
   */

   // if we wanted to solve the problem in the entire domain via a hierarchy of meshes,
   // we could have used this
   /*
   int nlevels = 2;
   GeneralCylHierarchy * hierarchy = new GeneralCylHierarchy(nlevels, *pmesh, 0, verbose);

   FOSLSProblHierarchy<ProblemType, GeneralHierarchy> * problems_hierarchy =
           new FOSLSProblHierarchy<ProblemType, GeneralHierarchy>(*hierarchy, nlevels, *bdr_conds, *fe_formulat, prec_option, verbose);

   for (int l = 0; l < nlevels; ++l)
   {
       ProblemType* problem = problems_hierarchy->GetProblem(l);
       problem->Solve(verbose);
   }
   */

   /*
   Array<ProblemType*> problems(nlevels);
   for (int l = 0; l < nlevels; ++l)
   {
       problems[l] = new ProblemType
               (*hierarchy, l, *bdr_conds, *fe_formulat, prec_option, verbose);
   }

   //problems[0]->Solve(verbose);
   for (int l = 0; l < nlevels; ++l)
       problems[l]->Solve(verbose);
   */

   // 5. Define the time slabs structure (how many, how many time steps within each
   // the time slab width in time steps
   int nslabs = 2;//4;//2;
   double slab_tau = 0.125;//1.0/16;//0.125;
   int slab_width = 4; // in time steps (as time intervals) withing a single time slab
   Array<ParMeshCyl*> timeslabs_pmeshcyls(nslabs);

   if (verbose)
   {
       std::cout << "Creating a sequence of time slabs: \n";
       std::cout << "# of slabs: " << nslabs << "\n";
       std::cout << "# of time intervals per slab: " << slab_width << "\n";
       std::cout << "time step within a time slab: " << slab_tau << "\n";
   }

   // 6. Creating time cylinder meshes for each time slab
   double tinit_tslab = 0.0;
   for (int tslab = 0; tslab < nslabs; ++tslab )
   {
       timeslabs_pmeshcyls[tslab] = new ParMeshCyl(comm, *pmeshbase, tinit_tslab, slab_tau, slab_width);

       tinit_tslab += slab_tau * slab_width;
   }

   /*
   for (int tslab = 0; tslab < nslabs; ++tslab )
       delete timeslabs_pmeshcyls[tslab];

   delete pmeshbase;
   delete pmesh;

   delete bdr_conds;
   delete formulat;
   delete fe_formulat;

   MPI_Finalize();
   return 0;
   */

   MFEM_ASSERT(fabs(tinit_tslab - 1.0) < 1.0e-14, "The slabs should cover the time interval "
                                                  "[0,1] but the upper bound doesn't match \n");

   // 7. Creating a set of problems hierarchies (a hierarchy per time slab)
   int two_grid = 2;
   Array<GeneralCylHierarchy*> cyl_hierarchies(nslabs);
   Array<FOSLSCylProblHierarchy<ProblemType, GeneralCylHierarchy>* > cyl_probhierarchies(nslabs);
   for (int tslab = 0; tslab < nslabs; ++tslab )
   {
       cyl_hierarchies[tslab] =
               new GeneralCylHierarchy(two_grid, *timeslabs_pmeshcyls[tslab], feorder, verbose);
       cyl_probhierarchies[tslab] =
               new FOSLSCylProblHierarchy<ProblemType, GeneralCylHierarchy>
               (*cyl_hierarchies[tslab], 2, *bdr_conds, *fe_formulat, prec_option, verbose);
   }

   // 8. Creating a two-grid time-stepping object which creates and gives access to
   // fine- and coarse- time-stepping over time slabs
   TwoGridTimeStepping<ProblemType> * twogrid_tstp =
           new TwoGridTimeStepping<ProblemType>(cyl_probhierarchies, verbose);

   TimeStepping<ProblemType> * fine_timestepping = twogrid_tstp->GetFineTimeStp();
   TimeStepping<ProblemType> * coarse_timestepping = twogrid_tstp->GetCoarseTimeStp();

   // some debugging part which survived, maybe even doesn't compile
#if 0
   /*
   // testing parallel solve vs separate subdomain solves
   Array<Vector*> exact_inputs(nslabs);
   for (int tslab = 0; tslab < fine_timestepping->Nslabs(); ++tslab)
       exact_inputs[tslab] = fine_timestepping->GetProblem(tslab)->GetExactBase("bot");
   if (verbose)
       std::cout << "\n Parallel solve with exact inputs: \n";
   fine_timestepping->ParallelSolve(exact_inputs, true);

   if (verbose)
       std::cout << "\n Separate problem sovles: \n";

   Array<Vector*> some_outputs(nslabs);
   for (int tslab = 0; tslab < fine_timestepping->Nslabs(); ++tslab)
   {
       some_outputs[tslab] = new Vector(fine_timestepping->GetInitCondSize());
       exact_inputs[tslab] = fine_timestepping->GetProblem(tslab)->GetExactBase("bot");
       fine_timestepping->GetProblem(tslab)->Solve(*exact_inputs[tslab], *some_outputs[tslab]);
   }

   MPI_Finalize();
   return 0;
   */

   /*
   // testing parallel solve vs sequential solves
   if (verbose)
       std::cout << "\n Sequential solve: \n";

   Array<Vector*> seq_outputs(nslabs);
   Vector * exact_input = fine_timestepping->GetProblem(0)->GetExactBase("bot");

   fine_timestepping->SequentialSolve(*exact_input, true);

   seq_outputs[0] = exact_input;
   for (int tslab = 0; tslab < fine_timestepping->Nslabs() - 1; ++tslab)
   {
       seq_outputs[tslab + 1] = &fine_timestepping->GetProblem(tslab)
               ->ExtractAtBase("top", fine_timestepping->GetProblem(tslab)->GetSol());
   }

   if (verbose)
       std::cout << "\n Parallel solve with inputs from seq. solve: \n";
   fine_timestepping->ParallelSolve(seq_outputs, true);

   MPI_Finalize();
   return 0;
   */
#endif

   // 9. Creating components of the parallel-in-time two grid algorithm
   // which will eventually be used in the form of a specific instance of
   // GeneralMultigrid

   // interpolation
   Array<Operator*> P_tstp(1);
   P_tstp[0] = twogrid_tstp->GetGlobalInterpolationOp();
   //P_tstp[0] = twogrid_tstp->GetGlobalInterpolationOpWithBnd();

   // creating fine-level operator, smoother and coarse-level operator

   // operator action, sequential time-stepping with zero initial value
   Array<Operator*> Ops_tstp(1);

   // smoother, parallel time-stepping with zero initial values
   Array<Operator*> Smoo_tstp(1);
   Array<Operator*> NullSmoo_tstp(1);

   // coarse operator
   // TODO: Describe this
   Operator* CoarseOp_tstp;

   Ops_tstp[0] =
           new TimeSteppingSeqOp<ProblemType>(*fine_timestepping, verbose);

   Smoo_tstp[0] =
           new TimeSteppingSmoother<ProblemType> (*fine_timestepping, verbose);

   CoarseOp_tstp =
           new TimeSteppingSolveOp<ProblemType>(*coarse_timestepping, verbose);
   //CoarseOp_tstp =
           //new TSTSpecialSolveOp<ProblemType>(*coarse_timestepping, verbose);
   NullSmoo_tstp[0] = NULL;

   // some debugging part which survived, maybe even doesn't compile
#if 0
   // checking SeqOp
   if (verbose)
       std::cout << "\n Sequential solve: \n";

   Vector input_tslab0(fine_timestepping->GetInitCondSize());
   input_tslab0 = 0.0;

   Vector testsol(fine_timestepping->GetGlobalProblemSize());
   Vector testAsol(fine_timestepping->GetGlobalProblemSize());
   Vector testrhs(fine_timestepping->GetGlobalProblemSize());
   testrhs = 2.0;
   fine_timestepping->ZeroBndValues(testrhs);
   fine_timestepping->SequentialSolve(testrhs, input_tslab0, testsol, true);

   BlockVector sol_viewer(testsol.GetData(), fine_timestepping->GetGlobalOffsets());

   Ops_tstp[0]->Mult(testsol, testAsol);

   testAsol -= testrhs;

   BlockVector diff_viewer(testAsol.GetData(), fine_timestepping->GetGlobalOffsets());
   for (int tslab = 0; tslab < fine_timestepping->Nslabs() ; ++tslab)
   {
       std::cout << "component diff norm = " << diff_viewer.GetBlock(tslab).Norml2()
                    / sqrt(diff_viewer.GetBlock(tslab).Size()) << "\n";
       for (int j = 0; j < diff_viewer.GetBlock(tslab).Size(); ++j)
           if (fabs(diff_viewer.GetBlock(tslab)[j]) > 1.0e-8)
               std::cout << j << ": diff = " << diff_viewer.GetBlock(tslab)[j] << "\n";
   }

   if (verbose)
       std::cout << "|| f - A * sol through SeqOp || = " <<
                    testAsol.Norml2() / sqrt (testAsol.Size()) << "\n";
   MPI_Finalize();
   return 0;

#endif

   // some debugging part which survived, maybe even doesn't compile
#if 0
   // checking SolveOp for the finest level (no coarsening)
   Operator * FineOp_tstp = new TimeSteppingSolveOp<ProblemType>(*fine_timestepping, verbose);

   Vector testsol(fine_timestepping->GetGlobalProblemSize());

   /*
   Vector input_tslab0(fine_timestepping->GetInitCondSize());
   input_tslab0 = 0.0;

   Vector testrhs(fine_timestepping->GetGlobalProblemSize());
   testrhs = 2.0;
   fine_timestepping->ZeroBndValues(testrhs);

   fine_timestepping->SequentialSolve(testrhs, input_tslab0, testsol, true);
   */

   testsol = 2.0;
   BlockVector testsol_viewer(testsol.GetData(), fine_timestepping->GetGlobalOffsets());
   fine_timestepping->GetProblem(0)->ZeroBndValues(testsol_viewer.GetBlock(0));

   Vector Atestsol(fine_timestepping->GetGlobalProblemSize());
   Ops_tstp[0]->Mult(testsol, Atestsol);

   Vector check_testsol(fine_timestepping->GetGlobalProblemSize());
   FineOp_tstp->Mult(Atestsol, check_testsol);

   Vector diff(fine_timestepping->GetGlobalProblemSize());
   diff = check_testsol;
   diff -= testsol;

   BlockVector Atestsol_viewer(Atestsol.GetData(), fine_timestepping->GetGlobalOffsets());
   BlockVector check_testsol_viewer(check_testsol.GetData(), fine_timestepping->GetGlobalOffsets());
   BlockVector diff_viewer(diff.GetData(), fine_timestepping->GetGlobalOffsets());
   for (int tslab = 0; tslab < fine_timestepping->Nslabs() ; ++tslab)
   {
       std::cout << "component diff norm = " << diff_viewer.GetBlock(tslab).Norml2()
                    / sqrt(diff_viewer.GetBlock(tslab).Size()) << "\n";
       for (int j = 0; j < diff_viewer.GetBlock(tslab).Size(); ++j)
           if (fabs(diff_viewer.GetBlock(tslab)[j]) > 1.0e-8)
               //std::cout << j << ": diff = " << diff_viewer.GetBlock(tslab)[j] << "\n";
               std::cout << j << ": diff = " << diff_viewer.GetBlock(tslab)[j]
                            << " val1 = " << testsol_viewer.GetBlock(tslab)[j]
                            << ", val2 = " << check_testsol_viewer.GetBlock(tslab)[j]
                            << ", Atestsol =  " << Atestsol_viewer.GetBlock(tslab)[j] << "\n";
   }

   if (verbose)
       std::cout << "|| testsol - A^(-1) * A * testsol through SeqOp and SolveOp || = " <<
                    diff.Norml2() / sqrt (diff.Size()) << "\n";

   MPI_Finalize();
   return 0;

#endif

   // 10. Finally, creating GeneralMultigrid instance which implements the algorithm described
   // in the beginning of this file
   GeneralMultigrid * spacetime_mg =
           new GeneralMultigrid(two_grid, P_tstp, Ops_tstp, *CoarseOp_tstp, Smoo_tstp, NullSmoo_tstp);

   ProblemType * problem0 = fine_timestepping->GetProblem(0);

   // creating rhs
   Vector mg_rhs(spacetime_mg->Width());
   fine_timestepping->ComputeGlobalRhs(mg_rhs);
   BlockVector mg_rhs_viewer(mg_rhs.GetData(), fine_timestepping->GetGlobalOffsets());
   //fine_timestepping->ZeroBndValues(mg_rhs);

   // input_tslab0 is the initial condition at the bottom of the entire domain cylinder
   Vector * input_tslab0 = problem0->GetExactBase("bot");

   // 10.5. First, to check, we solve with seq. solve on the finest level and compute the error

   if (verbose)
       std::cout << "\n\nSolving with a sequential time-stepping and checking the error \n";

   // checksol is the reference solution, the solution from the sequential
   // time-stepping at the fine level
   Vector checksol(spacetime_mg->Width());
   BlockVector checksol_viewer(checksol.GetData(), fine_timestepping->GetGlobalOffsets());

   // computing checksol
   fine_timestepping->SequentialSolve(mg_rhs, *input_tslab0, checksol, true);

   // checkres is the residual for the reference solution, computed within each time slab
   Vector checkres(spacetime_mg->Width());
   BlockVector checkres_viewer(checkres.GetData(), fine_timestepping->GetGlobalOffsets());

   fine_timestepping->SeqOp(checksol, input_tslab0, checkres);
   checkres -= mg_rhs;
   checkres *= -1;

   for (int tslab = 0; tslab < nslabs; ++tslab)
   {
       double tslabnorm = ComputeMPIVecNorm(comm, checkres_viewer.GetBlock(tslab),"", false);

       if (verbose)
           std::cout << "checkres, tslab = " << tslab << ", res norm = " <<
                        tslabnorm << "\n";
   }

   // checking the error for the reference solution
   fine_timestepping->ComputeError(checksol);
   fine_timestepping->ComputeBndError(checksol);

   // 11. Preparing righthand side for the multigrid solve
   // creating initial guess which satisfies given initial condition for the starting time slab
   Vector mg_x0(spacetime_mg->Width());
   mg_x0 = 0.0;
   BlockVector mg_x0_viewer(mg_x0.GetData(), fine_timestepping->GetGlobalOffsets());
   // unlike input_tslab0, exact_initcond0 will be Vector defined in the entire first
   // time slab, i.e., not only at the bottom base
   BlockVector * exact_initcond0 = problem0->GetTrueInitialCondition();
   mg_x0_viewer.GetBlock(0) = *exact_initcond0;

   fine_timestepping->ComputeGlobalRhs(mg_rhs);
   fine_timestepping->GetProblem(0)->CorrectFromInitCnd(*input_tslab0, mg_rhs_viewer.GetBlock(0));
   fine_timestepping->GetProblem(0)->ZeroBndValues(mg_rhs_viewer.GetBlock(0));

   // some debugging part which survived, maybe doesn't even compile
#if 0
   // second, to check, we solve with seq. solve on the finest level for the correction
   // and compute the error

   if (verbose)
       std::cout << "Solving for a correction with sequential solve and checking the final error \n";


   //Operator * FineOp_tstp = new TimeSteppingSolveOp<ProblemType>(*fine_timestepping, verbose);
   *input_tslab0 = 0.0;

   Vector checksol2(spacetime_mg->Width());
   BlockVector checksol2_viewer(checksol2.GetData(), fine_timestepping->GetGlobalOffsets());
   fine_timestepping->SequentialSolve(mg_rhs, *input_tslab0, checksol2, false);
   checksol2 += mg_x0;

   fine_timestepping->ComputeBndError(checksol2);

   fine_timestepping->ComputeError(checksol2);

   Vector diff(checksol2_viewer.GetBlock(0).Size());
   diff = checksol2_viewer.GetBlock(0);
   diff -= checksol_viewer.GetBlock(0);
   if (verbose)
       std::cout << "|| diff of checksols || = " << diff.Norml2() / sqrt(diff.Size()) << "\n";
   //diff.Print();

   MPI_Finalize();
   return 0;
#endif

   // some debugging part which survived, maybe even doesn't compile
#if 0
   if (verbose)
       std::cout << "Solving for a correction with 1 MG cycle \n";

   // solving for the correction, only one MG cycle
   Vector mg_corr(spacetime_mg->Width());
   mg_corr = 0.0;

   spacetime_mg->Mult(mg_rhs, mg_corr);

   Vector mg_finalsol(spacetime_mg->Width());
   mg_finalsol = mg_x0;
   mg_finalsol += mg_corr;

   fine_timestepping->ComputeError(mg_finalsol);

   MPI_Finalize();
   return 0;
#endif

   // 12. Running multiple MG cycles in a loop, to imitate
   // a parallel-in-time (though run in serial) algorithm
   if (verbose)
       std::cout << "Solving for a correction with multiple MG cycles \n";

   // tolerance used for stopping criteria below
   double eps = 1.0e-6;

   // residual in the MG algorithm
   Vector mg_res(spacetime_mg->Width());
   BlockVector mg_res_viewer(mg_res.GetData(), fine_timestepping->GetGlobalOffsets());
   mg_res = mg_rhs;

   double global_res0_norm = ComputeMPIVecNorm(comm, mg_res, "", false);
   if (verbose)
       std::cout << "res0 norm = " << global_res0_norm << "\n";

   // final MG solution
   Vector mg_finalsol(spacetime_mg->Width());
   BlockVector mg_finalsol_viewer(mg_finalsol.GetData(), fine_timestepping->GetGlobalOffsets());

   // Set final solution to the initial vector (will add the correction in the end, after MG cycles)
   mg_finalsol = mg_x0;

   // solving for the correction, only one MG cycle
   Vector mg_corr(spacetime_mg->Width());
   mg_corr = 0.0;

   Vector mg_temp(spacetime_mg->Width());

   bool converged = false;

   int iter = 0;
   // The method must not peform more than #timeslabs of MG cycles
   // Maximum number of iterations.
   int mg_max_iter = 10;

   // main loop, at each iteration a two-grid method is called
   while (!converged && iter < mg_max_iter)
   {
       ++iter;

       if (iter > 1)
       {
           for (int tslab = 0; tslab < nslabs; ++tslab)
           {
               double global_res_norm = ComputeMPIVecNorm(comm, mg_res_viewer.GetBlock(tslab), "", false);
               if (verbose)
                   std::cout << "mg_res full tslab = " << tslab << ", norm = "
                             << global_res_norm << "\n";

               std::cout << "mg_res full tslab = " << tslab << "\n";
               std::cout << "norm = " << mg_res_viewer.GetBlock(tslab).Norml2() /
                            sqrt (mg_res_viewer.GetBlock(tslab).Size()) << "\n";
           }
       }

       // solve for a correction with a current residual
       mg_corr = 0.0;
       spacetime_mg->Mult(mg_res, mg_corr);

       double global_corr_norm = ComputeMPIVecNorm(comm, mg_corr, "", false);

       if (verbose)
           std::cout << "Iteration " << iter << ": correction norm = " <<
                        global_corr_norm << "\n";

       // removing discrepancy at the interfaces between time slabs (taking values from below)
       fine_timestepping->UpdateInterfaceFromPrev(mg_corr);

       // update the solution by adding the correction to it
       mg_finalsol += mg_corr;

       /*
       std::cout << "Checking jump on the interface between time slabs \n";

       Vector& vec1 = fine_timestepping->GetProblem(0)->ExtractAtBase("top", mg_finalsol_viewer.GetBlock(0));
       Vector& vec2 = fine_timestepping->GetProblem(1)->ExtractAtBase("bot", mg_finalsol_viewer.GetBlock(1));

       Vector diff(vec1.Size());
       diff = vec1;
       diff -= vec2;

       std::cout << "Discrepancy at the interface, norm = " << diff.Norml2() / sqrt(diff.Size()) << "\n";
       */

       // update the residual
       fine_timestepping->SeqOp(mg_corr, mg_temp);
       mg_temp -= mg_res;
       mg_temp *= -1;
       // FIXME: This zeroing is too much on paper, but without it error at the boundary is reported
       fine_timestepping->ZeroBndValues(mg_temp);
       //mg_temp.Print();

       mg_res = mg_temp;

       for (int tslab = 0; tslab < nslabs; ++tslab)
       {
           double tslab_res_norm = ComputeMPIVecNorm(comm, mg_res_viewer.GetBlock(tslab), "", false);
           if (verbose)
                std::cout << "mg_res after iterate, tslab = " << tslab <<
                             ", norm = " << tslab_res_norm << "\n";
       }

       double global_res_norm = ComputeMPIVecNorm(comm, mg_res, "", false);

       // check convergence
       // stopping criteria: stop, if res_norm (current residual norm)
       //                               < eps * res0_norm (initial residual norm)
       if (global_res_norm < eps * global_res0_norm)
           converged = true;

       // reporting convergence status
       if (verbose)
           std::cout << "Iteration " << iter << ": res_norm = " << global_res_norm << "\n";

       //fine_timestepping->ComputeError(mg_finalsol);
       //fine_timestepping->ComputeBndError(mg_finalsol);
   }

   if (verbose)
   {
       if (converged)
            std::cout << "Convergence's been reached within " << iter << " iterations. \n";
       else
           std::cout << "Convergence has not been reached within " << iter << " iterations. \n";
   }

   fine_timestepping->ComputeError(mg_finalsol);
   fine_timestepping->ComputeBndError(mg_finalsol);

   // 13. Studying the difference between ref. solution and MG solution,
   // for each time slab, for each variable

   // difference between reference solution and final solution of the parallel-in-time algorithm
   Vector diff(spacetime_mg->Width());
   diff = mg_finalsol;
   diff -= checksol;

   BlockVector diff_viewer(diff.GetData(), fine_timestepping->GetGlobalOffsets());

   for (int tslab = 0; tslab < nslabs; ++tslab)
   {
       if (verbose)
           std::cout << "tslab = " << tslab << "\n";

       BlockVector diff_blk_viewer(diff_viewer.GetBlock(tslab).GetData(), fine_timestepping->GetProblem(tslab)->GetTrueOffsets());
       BlockVector checksol_blk_viewer(checksol_viewer.GetBlock(tslab).GetData(), fine_timestepping->GetProblem(tslab)->GetTrueOffsets());
       BlockVector finalsol_blk_viewer(mg_finalsol_viewer.GetBlock(tslab).GetData(), fine_timestepping->GetProblem(tslab)->GetTrueOffsets());
       for (int blk = 0; blk < fe_formulat->Nunknowns(); ++blk)
       {
           if (verbose)
           {
               std::cout << "|| diff of checksol and mg sol ||, blk = " << blk << " = " <<
                            diff_blk_viewer.GetBlock(blk).Norml2() / sqrt(diff_blk_viewer.GetBlock(blk).Size()) << "\n";
           }

           for (int i = 0; i < diff_blk_viewer.GetBlock(blk).Size(); ++i)
               if (fabs(diff_blk_viewer.GetBlock(blk)[i]) > 1.0e-9)
               {
                   std::cout << "entry " << i << ": checksol = " << checksol_blk_viewer.GetBlock(blk)[i] << ", "
                             << "finalsol = " << finalsol_blk_viewer.GetBlock(blk)[i] << ", "
                             << "diff = " << diff_blk_viewer.GetBlock(blk)[i] << "\n";
               }

       }
       //if (verbose)
           //std::cout << "|| diff of checksol and mg sol ||(tslab = " << tslab << ") = " <<
                        //diff_viewer.GetBlock(tslab).Norml2() / sqrt(diff_viewer.GetBlock(tslab).Size()) << "\n";
   }

   MPI_Barrier(comm);

   // 14. Deallocating memory

   for (int i = 0; i < cyl_probhierarchies.Size(); ++i)
       delete cyl_probhierarchies[i];

   for (int i = 0; i < cyl_hierarchies.Size(); ++i)
       delete cyl_hierarchies[i];

   delete twogrid_tstp;

   delete spacetime_mg;

   for (int i = 0; i < Ops_tstp.Size(); ++i)
       delete Ops_tstp[i];

   for (int i = 0; i < Smoo_tstp.Size(); ++i)
       delete Smoo_tstp[i];

   delete CoarseOp_tstp;

   for (int tslab = 0; tslab < nslabs; ++tslab )
       delete timeslabs_pmeshcyls[tslab];

   delete input_tslab0;
   delete exact_initcond0;

   delete pmeshbase;
   delete pmesh;

   delete bdr_conds;
   delete formulat;
   delete fe_formulat;

   MPI_Finalize();
   return 0;

}

