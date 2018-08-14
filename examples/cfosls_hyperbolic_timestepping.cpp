///                       CFOSLS formulation for transport equation in 3D/4D solved via
///                                      standard time-slabbing technique
///
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
/// Typical run of this example: ./cfosls_hyperbolic_timestepping --whichD 3 --spaceS "L2" -no-vis
/// If you ant Hdiv-H1-L2 formulation, you will need not only change --spaceS option but also
/// change the source code, around 4.
/// Another example on time-slabbing technique, with a more complicated two-grid method is
/// cfosls_hyperbolic_tst_multigrid.cpp.

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
   // 1. Initialize MPI.
   int num_procs, myid;

   MPI_Init(&argc, &argv);
   MPI_Comm comm = MPI_COMM_WORLD;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   bool verbose = (myid == 0);

   int nDimensions     = 3;
   int numsol          = 0;

   int ser_ref_levels  = 2;
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
   MFEM_ASSERT(!strcmp(space_for_sigma,"H1") == 0 || (strcmp(space_for_sigma,"H1") == 0
                                                      && strcmp(space_for_S,"H1") == 0),
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

   meshbase->CheckElementOrientation(true);

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

   /*
   // Hdiv-L2 case
   using FormulType = CFOSLSFormulation_HdivL2Hyper;
   using FEFormulType = CFOSLSFEFormulation_HdivL2Hyper;
   using BdrCondsType = BdrConditions_CFOSLS_HdivL2_Hyper;
   using ProblemType = FOSLSCylProblem_HdivL2hyp;
   */

   FormulType * formulat = new FormulType (dim, numsol, verbose);
   FEFormulType * fe_formulat = new FEFormulType(*formulat, feorder);
   BdrCondsType * bdr_conds = new BdrCondsType(*pmesh);

   // if we wanted to solve the problem in the entire domain, we could have used this
   /*
   ProblemType * problem = new ProblemType (*pmesh, *bdr_conds, *fe_formulat, prec_option, verbose);

   problem->Solve(verbose);

   delete problem;

   MPI_Finalize();
   return 0;
   */

   // 5. Define the time slabs structure (how many, how many time steps within each
   // the time slab width in time steps
   int nslabs = 2;//1;//2;//4;//2;
   double slab_tau = 0.125;//0.125;//1.0/16;//0.125;
       int slab_width = 4; // in time steps (as time intervals) withing a single time slab
   Array<ParMeshCyl*> timeslabs_pmeshcyls(nslabs);

   if (verbose)
   {
       std::cout << "Creating a sequence of time slabs: \n";
       std::cout << "# of slabs: " << nslabs << "\n";
       std::cout << "# of time intervals per slab: " << slab_width << "\n";
       std::cout << "time step within a time slab: " << slab_tau << "\n";
   }

   // 6. Creating a fine-level time-stepping instance from a series of problems
   // in the cylinders (time slabs)
   double tinit_tslab = 0.0;
   Array<ProblemType*> timeslabs_problems(nslabs);
   for (int tslab = 0; tslab < nslabs; ++tslab )
   {
       timeslabs_pmeshcyls[tslab] = new ParMeshCyl(comm, *pmeshbase, tinit_tslab, slab_tau, slab_width);
       // just for fun, refining each mesh once after creating (could be more, of course)
       timeslabs_pmeshcyls[tslab]->Refine(1);

       timeslabs_problems[tslab] = new ProblemType(*timeslabs_pmeshcyls[tslab], *bdr_conds, *fe_formulat, prec_option, verbose);

       tinit_tslab += slab_tau * slab_width;
   }

   MFEM_ASSERT(fabs(tinit_tslab - 1.0) < 1.0e-14, "The slabs should cover the time interval "
                                                 "[0,1] but the upper bound doesn't match \n");

   TimeStepping<ProblemType> * fine_timestepping = new TimeStepping<ProblemType>(timeslabs_problems, verbose);

   // creating fine-level time-stepping from a TwoGridTimeStepping instance
#if 0
   // if this doesn't work, look in cfosls_hyperbolic_tst_multigrid,
   // where a newer version is used
   double tinit_tslab = 0.0;
   for (int tslab = 0; tslab < nslabs; ++tslab )
   {
       timeslabs_pmeshcyls[tslab] = new ParMeshCyl(comm, *pmeshbase, tinit_tslab, slab_tau, slab_width);
       tinit_tslab += slab_tau * slab_width;
   }

   MFEM_ASSERT(fabs(tinit_tslab - 1.0) < 1.0e-14, "The slabs should cover the time interval "
                                                 "[0,1] but the upper bound doesn't match \n");

   // creating a set of problems hierarchies (a hierarchy per time slab)
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

   TwoGridTimeStepping<ProblemType> * twogrid_tstp =
           new TwoGridTimeStepping<ProblemType>(cyl_probhierarchies, verbose);

   // creating fine and coarse time-stepping and interpolation operator between them
   TimeStepping<ProblemType> * fine_timestepping = twogrid_tstp->GetFineTimeStp();

#endif // for creating fine-level time-stepping from a TwoGridTimeStepping instance

   // some older code which survived, maybe even doesn't compile
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

   int global_size = fine_timestepping->GetGlobalProblemSize();

   // 7. Creating an initial guess which satisfies given initial condition
   // for the starting time slab (since the current time-stepping implementation
   // is designed to work with zero initial guess at certain places, like computing residual)

   Vector xinit(global_size);
   xinit = 0.0;
   FOSLSCylProblem * problem0 = fine_timestepping->GetProblem(0);
   BlockVector xinit_viewer(xinit.GetData(), fine_timestepping->GetGlobalOffsets());
   BlockVector * exact_initcond0 = problem0->GetTrueInitialCondition();
   xinit_viewer.GetBlock(0) = *exact_initcond0;

   // computing rhs
   Vector rhs(global_size);
   fine_timestepping->ComputeGlobalRhs(rhs);
   fine_timestepping->ZeroBndValues(rhs);

   // computing initial data
   Vector input_tslab0(fine_timestepping->GetInitCondSize());
   input_tslab0 = *fine_timestepping->GetProblem(0)->GetExactBase("bot");

   // 8. Solving with a sequential time-stepping

   if (verbose)
       std::cout << "\n\nSolving via sequential time-stepping and checking the error \n";

   Vector checksol(global_size);
   fine_timestepping->SequentialSolve(rhs, input_tslab0, checksol, true);

   // 9. Checking the residual after the solve in each time slab
   Vector checkres(global_size);
   BlockVector checkres_viewer(checkres.GetData(), fine_timestepping->GetGlobalOffsets());

   fine_timestepping->SeqOp(checksol, &input_tslab0, checkres);
   checkres -= rhs;
   checkres *= -1;

   for (int tslab = 0; tslab < nslabs; ++tslab)
   {
       double tslabnorm = ComputeMPIVecNorm(comm, checkres_viewer.GetBlock(tslab),"", false);

       if (verbose)
           std::cout << "checkres, tslab = " << tslab << ", res norm = " <<
                        tslabnorm << "\n";
   }

   double global_res_norm = ComputeMPIVecNorm(comm, checkres, "", false);
   if (verbose)
       std::cout << "res norm = " << global_res_norm << "\n";

   // 9. Checking the error for the final solution
   fine_timestepping->ComputeError(checksol);
   fine_timestepping->ComputeBndError(checksol);

   // 17. Free the used memory.
   for (int tslab = 0; tslab < nslabs; ++tslab )
       delete timeslabs_pmeshcyls[tslab];

   delete exact_initcond0;

   delete pmeshbase;
   delete pmesh;

   delete bdr_conds;
   delete formulat;
   delete fe_formulat;

   MPI_Finalize();
   return 0;
}

