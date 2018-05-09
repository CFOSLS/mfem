//                       CFOSLS formulation for transport equation in 3D/4D with time-slabbing technique

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>
#include <unistd.h>

#define MYZEROTOL (1.0e-13)
#define ZEROTOL (1.0e-13)

//#include "cfosls/testhead.hpp"
//#include "divfree_solver_tools.hpp"

#define NONHOMO_TEST

// must be active
#define USE_TSL

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;

   // 1. Initialize MPI
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
   const char *mesh_file = "../data/star.mesh";
#ifdef USE_TSL
   const char *meshbase_file = "../data/star.mesh";
   int Nt = 4;
   double tau = 0.25;
#endif

   const char *formulation = "cfosls"; // "cfosls" or "fosls"
   const char *space_for_S = "H1";     // "H1" or "L2"
   const char *space_for_sigma = "Hdiv"; // "Hdiv" or "H1"
   bool eliminateS = true;            // in case space_for_S = "L2" defines whether we eliminate S from the system

   // solver options
   int prec_option = 1; //defines whether to use preconditioner or not, and which one
   bool use_ADS;

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
   args.AddOption(&formulation, "-form", "--formul",
                  "Formulation to use (cfosls or fosls).");
   args.AddOption(&space_for_S, "-spaceS", "--spaceS",
                  "Space for S (H1 or L2).");
   args.AddOption(&space_for_sigma, "-spacesigma", "--spacesigma",
                  "Space for sigma (Hdiv or H1).");
   args.AddOption(&eliminateS, "-elims", "--eliminateS", "-no-elims",
                  "--no-eliminateS",
                  "Turn on/off elimination of S in L2 formulation.");

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
   }

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

   MFEM_ASSERT(strcmp(formulation,"cfosls") == 0 || strcmp(formulation,"fosls") == 0, "Formulation must be cfosls or fosls!\n");
   MFEM_ASSERT(strcmp(space_for_S,"H1") == 0 || strcmp(space_for_S,"L2") == 0, "Space for S must be H1 or L2!\n");
   MFEM_ASSERT(strcmp(space_for_sigma,"Hdiv") == 0 || strcmp(space_for_sigma,"H1") == 0, "Space for sigma must be Hdiv or H1!\n");

   MFEM_ASSERT(!strcmp(space_for_sigma,"H1") == 0 || (strcmp(space_for_sigma,"H1") == 0 && strcmp(space_for_S,"H1") == 0), "Sigma from H1vec must be coupled with S from H1!\n");
   MFEM_ASSERT(!strcmp(space_for_sigma,"H1") == 0 || (strcmp(space_for_sigma,"H1") == 0 && use_ADS == false), "ADS cannot be used when sigma is from H1vec!\n");

   StopWatch chrono;

   //DEFAULTED LINEAR SOLVER OPTIONS
   //int max_iter = 150000;
   //double rtol = 1e-12;//1e-7;//1e-9;
   //double atol = 1e-14;//1e-9;//1e-12;

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

#ifdef USE_TSL
   if (verbose)
       std::cout << "USE_TSL is active (mesh is constructed using mesh generator) \n";
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

   /*
   std::stringstream fname;
   fname << "square_2d_moderate_corrected.mesh";
   std::ofstream ofid(fname.str().c_str());
   ofid.precision(8);
   meshbase->Print(ofid);
   */

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

   /*
   int nslabs = 3;
   Array<int> slabs_widths(nslabs);
   slabs_widths[0] = Nt / 2;
   slabs_widths[1] = Nt / 2 - 1;
   slabs_widths[2] = 1;
   ParMeshCyl * pmesh = new ParMeshCyl(comm, *pmeshbase, 0.0, tau, Nt, nslabs, &slabs_widths);
   */

   ParMeshCyl * pmesh = new ParMeshCyl(comm, *pmeshbase, 0.0, tau, Nt);
   //pmesh->Refine(1);

   //pmesh->PrintSlabsStruct();

   //delete pmeshbase;
   //delete pmesh;
   //MPI_Finalize();
   //return 0;

   /*
   if (num_procs == 1)
   {
       std::stringstream fname;
       fname << "pmesh_check.mesh";
       std::ofstream ofid(fname.str().c_str());
       ofid.precision(8);
       pmesh->Print(ofid);
   }
   */

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
   Transport_test Mytest(dim, numsol);

   MPI_Barrier(comm);
   std::cout << std::flush;
   MPI_Barrier(comm);

   if (verbose)
      std::cout << "Checking a single solve from a one TimeCylHyper instance "
                    "created for the entire domain \n";

   /*
   // Hdiv-L2 formulation
   FOSLSFormulation * formulat = new CFOSLSFormulation_HdivL2Hyper (dim, numsol, verbose);
   FOSLSFEFormulation * fe_formulat = new CFOSLSFEFormulation_HdivL2Hyper(*formulat, feorder);
   BdrConditions * bdr_conds = new BdrConditions_CFOSLS_HdivL2_Hyper(*pmesh);
   FOSLSCylProblem_HdivL2L2hyp * problem = new FOSLSCylProblem_HdivL2L2hyp
           (*pmesh, *bdr_conds, *fe_formulat, prec_option, verbose);
   */

   // Hdiv-H1 formulation
   FOSLSFormulation * formulat = new CFOSLSFormulation_HdivH1Hyper (dim, numsol, verbose);
   FOSLSFEFormulation * fe_formulat = new CFOSLSFEFormulation_HdivH1Hyper(*formulat, feorder);
   BdrConditions * bdr_conds = new BdrConditions_CFOSLS_HdivH1_Hyper(*pmesh);

   FOSLSCylProblem_HdivH1L2hyp * problem = new FOSLSCylProblem_HdivH1L2hyp
           (*pmesh, *bdr_conds, *fe_formulat, prec_option, verbose);

   //problem->Solve(verbose);

   //int length = 2;
   //mfem::D<C> * testtt = new mfem::D<C>(1,2, length);

   //MPI_Finalize();
   //return 0;

   /*
   int nlevels = 2;
   GeneralCylHierarchy * hierarchy = new GeneralCylHierarchy(nlevels, *pmesh, 0, verbose);

   FOSLSProblHierarchy<FOSLSProblem_CFOSLS_HdivL2_Hyper, GeneralHierarchy> * problems_hierarchy =
           new FOSLSProblHierarchy<FOSLSProblem_CFOSLS_HdivL2_Hyper, GeneralHierarchy>(*hierarchy, nlevels, *bdr_conds, *fe_formulat, prec_option, verbose);

   for (int l = 0; l < nlevels; ++l)
   {
       FOSLSProblem_CFOSLS_HdivL2_Hyper* problem = problems_hierarchy->GetProblem(l);
       problem->Solve(verbose);
   }
   */

   /*
   Array<FOSLSCylProblem_HdivL2L2hyp*> problems(nlevels);
   for (int l = 0; l < nlevels; ++l)
   {
       problems[l] = new FOSLSCylProblem_HdivL2L2hyp
               (*hierarchy, l, *bdr_conds, *fe_formulat, prec_option, verbose);
   }

   //problems[0]->Solve(verbose);
   for (int l = 0; l < nlevels; ++l)
       problems[l]->Solve(verbose);
   */

   int nslabs = 4;//4;//2;
   double slab_tau = 1.0/16;//1.0/16;//0.125;
   int slab_width = 4; // in time steps (as time intervals) withing a single time slab
   Array<ParMeshCyl*> timeslabs_pmeshcyls(nslabs);
   Array<FOSLSCylProblem_HdivH1L2hyp*> timeslabs_problems(nslabs);

   if (verbose)
   {
       std::cout << "Creating a sequence of time slabs: \n";
       std::cout << "# of slabs: " << nslabs << "\n";
       std::cout << "# of time intervals per slab: " << slab_width << "\n";
       std::cout << "time step within a time slab: " << slab_tau << "\n";
   }

   double tinit_tslab = 0.0;
   for (int tslab = 0; tslab < nslabs; ++tslab )
   {
       timeslabs_pmeshcyls[tslab] = new ParMeshCyl(comm, *pmeshbase, tinit_tslab, slab_tau, slab_width);

       //timeslabs_problems[tslab] = new FOSLSCylProblem_HdivL2L2hyp(*timeslabs_pmeshcyls[tslab], *bdr_conds, *fe_formulat, prec_option, verbose);
       timeslabs_problems[tslab] = new FOSLSCylProblem_HdivH1L2hyp(*timeslabs_pmeshcyls[tslab], *bdr_conds, *fe_formulat, prec_option, verbose);

       tinit_tslab += slab_tau * slab_width;
   }

   MFEM_ASSERT(fabs(tinit_tslab - 1.0) < 1.0e-14, "The slabs should cover the time interval "
                                                 "[0,1] but the upper bound doesn't match \n");

   /*
   TimeStepping<FOSLSCylProblem_HdivH1L2hyp> * time_stepping = new TimeStepping<FOSLSCylProblem_HdivH1L2hyp>(timeslabs_problems, verbose);

   Vector* init_vector = timeslabs_problems[0]->GetExactBase("bot");

   bool compute_error = true;
   time_stepping->SequentialSolve(*init_vector, compute_error);

   MPI_Finalize();
   return 0;
   */

   /*
   int nlevels = 2;
   GeneralCylHierarchy * hierarchy = new GeneralCylHierarchy(nlevels, *pmesh, 0, verbose);

   FOSLSCylProblHierarchy<FOSLSCylProblem_HdivL2L2hyp, GeneralCylHierarchy> * problems_hierarchy =
           new FOSLSCylProblHierarchy<FOSLSCylProblem_HdivL2L2hyp, GeneralCylHierarchy>
           (*hierarchy, nlevels, *bdr_conds, *fe_formulat, prec_option, verbose);
   */

   // creating a set of problems hierarchies (a hierarchy per time slab)
   int two_grid = 2;
   Array<GeneralCylHierarchy*> cyl_hierarchies(nslabs);
   Array<FOSLSCylProblHierarchy<FOSLSCylProblem_HdivH1L2hyp, GeneralCylHierarchy>* > cyl_probhierarchies(nslabs);
   for (int tslab = 0; tslab < nslabs; ++tslab )
   {
       cyl_hierarchies[tslab] =
               new GeneralCylHierarchy(two_grid, *timeslabs_pmeshcyls[tslab], feorder, verbose);
       cyl_probhierarchies[tslab] =
               new FOSLSCylProblHierarchy<FOSLSCylProblem_HdivH1L2hyp, GeneralCylHierarchy>
               (*cyl_hierarchies[tslab], 2, *bdr_conds, *fe_formulat, prec_option, verbose);
   }

   TwoGridTimeStepping<FOSLSCylProblem_HdivH1L2hyp> * twogrid_tstp =
           new TwoGridTimeStepping<FOSLSCylProblem_HdivH1L2hyp>(cyl_probhierarchies, verbose);

   // creating fine and coarse time-stepping and interpolation operator between them
   TimeStepping<FOSLSCylProblem_HdivH1L2hyp> * fine_timestepping = twogrid_tstp->GetFineTimeStp();

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

   TimeStepping<FOSLSCylProblem_HdivH1L2hyp> * coarse_timestepping = twogrid_tstp->GetCoarseTimeStp();

   Array<Operator*> P_tstp(1);
   P_tstp[0] = twogrid_tstp->GetGlobalInterpolationOp();
   //P_tstp[0] = twogrid_tstp->GetGlobalInterpolationOpWithBnd();

   // creating fine-level operator, smoother and coarse-level operator
   Array<Operator*> Ops_tstp(1);
   Array<Operator*> Smoo_tstp(1);
   Array<Operator*> NullSmoo_tstp(1);
   Operator* CoarseOp_tstp;

   Ops_tstp[0] =
           new TimeSteppingSeqOp<FOSLSCylProblem_HdivH1L2hyp>(*fine_timestepping, verbose);

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

#if 0
   // checking SolveOp for the finest level (no coarsening)
   Operator * FineOp_tstp = new TimeSteppingSolveOp<FOSLSCylProblem_HdivH1L2hyp>(*fine_timestepping, verbose);

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

   Smoo_tstp[0] =
           new TimeSteppingSmoother<FOSLSCylProblem_HdivH1L2hyp> (*fine_timestepping, verbose);

   CoarseOp_tstp =
           new TimeSteppingSolveOp<FOSLSCylProblem_HdivH1L2hyp>(*coarse_timestepping, verbose);
   //CoarseOp_tstp =
           //new TSTSpecialSolveOp<FOSLSCylProblem_HdivH1L2hyp>(*coarse_timestepping, verbose);
   NullSmoo_tstp[0] = NULL;

   // finally, creating general multigrid instance
   GeneralMultigrid * spacetime_mg =
           new GeneralMultigrid(P_tstp, Ops_tstp, *CoarseOp_tstp, Smoo_tstp, NullSmoo_tstp);

   // creating initial guess which satisfies given initial condition for the starting time slab
   Vector mg_x0(spacetime_mg->Width());
   mg_x0 = 0.0;
   FOSLSCylProblem_HdivH1L2hyp * problem0 = fine_timestepping->GetProblem(0);
   BlockVector mg_x0_viewer(mg_x0.GetData(), fine_timestepping->GetGlobalOffsets());
   BlockVector * exact_initcond0 = problem0->GetTrueInitialCondition();
   mg_x0_viewer.GetBlock(0) = *exact_initcond0;

   // creating rhs and computing the residual for the mg_x0
   Vector mg_rhs(spacetime_mg->Width());
   fine_timestepping->ComputeGlobalRhs(mg_rhs);
   BlockVector mg_rhs_viewer(mg_rhs.GetData(), fine_timestepping->GetGlobalOffsets());
   fine_timestepping->ZeroBndValues(mg_rhs);

   Vector input_tslab0(fine_timestepping->GetInitCondSize());
   input_tslab0 = *fine_timestepping->GetProblem(0)->GetExactBase("bot");
   //fine_timestepping->GetProblem(0)->SetAtBase("bot", input_tslab0, mg_rhs_viewer.GetBlock(0));

   // first, to check, we solve with seq. solve on the finest level and compute the error

   if (verbose)
       std::cout << "Solving with sequential solve and checking the error \n";


   //Vector tempvec(fine_timestepping->GetProblem(0)->GlobalTrueProblemSize());
   //fine_timestepping->GetProblem(0)->ConvertInitCndToFullVector(input_tslab0, temp_vec);

   Vector checksol(spacetime_mg->Width());
   BlockVector checksol_viewer(checksol.GetData(), fine_timestepping->GetGlobalOffsets());
   fine_timestepping->SequentialSolve(mg_rhs, input_tslab0, checksol, true);

   Vector checkres(spacetime_mg->Width());
   BlockVector checkres_viewer(checkres.GetData(), fine_timestepping->GetGlobalOffsets());

   fine_timestepping->SeqOp(checksol, &input_tslab0, checkres);
   checkres -= mg_rhs;
   checkres *= -1;

   for (int tslab = 0; tslab < nslabs; ++tslab)
   {
       std::cout << "checkres, tslab = " << tslab << "\n";
       std::cout << "norm = " << checkres_viewer.GetBlock(tslab).Norml2() /
                    sqrt (checkres_viewer.GetBlock(tslab).Size()) << "\n";

       //for (int i = 0; i < checkres_viewer.GetBlock(tslab).Size(); ++i)
           //if (fabs(checkres_viewer.GetBlock(tslab)[i]) > 1.0e-10)
               //std::cout << "entry " << i << ": res value = " << checkres_viewer.GetBlock(tslab)[i] << "\n";

       //std::cout << "values of res at bottom interface: \n";
       //Vector& vec = fine_timestepping->GetProblem(tslab)->ExtractAtBase("bot", checkres_viewer.GetBlock(tslab));
       //vec.Print();
   }

   fine_timestepping->ComputeError(checksol);
   fine_timestepping->ComputeBndError(checksol);

   //double checkres_norm = checkres.Norml2() / sqrt (checkres.Size());
   //if (verbose)
       //std::cout << "checkres norm = " << checkres_norm << "\n";

   //MPI_Finalize();
   //return 0;

   /*
   Array<Vector*> & debug_botbases = fine_timestepping->ExtractAtBases("bot", checksol);
   std::cout << "botbases of output from seq,. solve outside mg \n";
   for (int tslab = 0; tslab < nslabs; ++tslab)
   {
       std::cout << "tslab = " << tslab << "\n";
       debug_botbases[tslab]->Print();
   }
   */


   /*
   BlockVector checksol_viewer(checksol.GetData(), fine_timestepping->GetGlobalOffsets());
   //checksol_viewer.GetBlock(0).Print();

   fine_timestepping->ComputeBndError(checksol);

   MPI_Finalize();
   return 0;
   */

   fine_timestepping->ComputeGlobalRhs(mg_rhs);
   fine_timestepping->GetProblem(0)->CorrectFromInitCnd(input_tslab0, mg_rhs_viewer.GetBlock(0));
   fine_timestepping->GetProblem(0)->ZeroBndValues(mg_rhs_viewer.GetBlock(0));

#if 0
   // second, to check, we solve with seq. solve on the finest level for the correction
   // and compute the error

   if (verbose)
       std::cout << "Solving for a correction with sequential solve and checking the final error \n";


   //Operator * FineOp_tstp = new TimeSteppingSolveOp<FOSLSCylProblem_HdivH1L2hyp>(*fine_timestepping, verbose);
   input_tslab0 = 0.0;

   Vector checksol2(spacetime_mg->Width());
   BlockVector checksol2_viewer(checksol2.GetData(), fine_timestepping->GetGlobalOffsets());
   fine_timestepping->SequentialSolve(mg_rhs, input_tslab0, checksol2, false);
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

#if 0
   if (verbose)
       std::cout << "Solving for a correction with 1 MG cycle \n";

   // solving for the correction, only one MG cycle
   Vector mg_sol(spacetime_mg->Width());
   mg_sol = 0.0;

   spacetime_mg->Mult(mg_rhs, mg_sol);

   Vector mg_finalsol(spacetime_mg->Width());
   mg_finalsol = mg_x0;
   mg_finalsol += mg_sol;

   fine_timestepping->ComputeError(mg_finalsol);

   MPI_Finalize();
   return 0;
#endif

   if (verbose)
       std::cout << "Solving for a correction with multiple MG cycles \n";

   double eps = 1.0e-6;

   Vector mg_res(spacetime_mg->Width());
   BlockVector mg_res_viewer(mg_res.GetData(), fine_timestepping->GetGlobalOffsets());
   mg_res = mg_rhs;
   double res0_norm = mg_res.Norml2() / sqrt (mg_res.Size());
   if (verbose)
       std::cout << "res0 norm = " << res0_norm << "\n";

   Vector mg_finalsol(spacetime_mg->Width());
   BlockVector mg_finalsol_viewer(mg_finalsol.GetData(), fine_timestepping->GetGlobalOffsets());

   mg_finalsol = mg_x0;

   // solving for the correction, only one MG cycle
   Vector mg_sol(spacetime_mg->Width());
   mg_sol = 0.0;

   Vector mg_temp(spacetime_mg->Width());

   bool converged = false;

   double res_norm;
   int iter = 0;
   while (!converged && iter < 4)
   {
       ++iter;

       /*
       Array<Vector*> & debug_botbases = fine_timestepping->ExtractAtBases("bot", mg_res);
       std::cout << "botbases of input residual for MG Mult \n";
       for (int tslab = 0; tslab < nslabs; ++tslab)
       {
           std::cout << "tslab = " << tslab << "\n";
           debug_botbases[tslab]->Print();
       }
       */

       if (iter > 1)
           for (int tslab = 0; tslab < nslabs; ++tslab)
           {
               std::cout << "mg_res full tslab = " << tslab << "\n";
               std::cout << "norm = " << mg_res_viewer.GetBlock(tslab).Norml2() /
                            sqrt (mg_res_viewer.GetBlock(tslab).Size()) << "\n";
               //mg_res_viewer.GetBlock(tslab).Print();
           }


       // solve for a correction with a current residual
       mg_sol = 0.0;
       spacetime_mg->Mult(mg_res, mg_sol);

       if (verbose)
           std::cout << "Iteration " << iter << ": correction norm = " <<
                        mg_sol.Norml2() / sqrt(mg_sol.Size()) << "\n";

       // removing discrepancy at the interfaces between time slabs (taking values from below)
       fine_timestepping->UpdateInterfaceFromPrev(mg_sol);

       // update the solution
       mg_finalsol += mg_sol;

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
       fine_timestepping->SeqOp(mg_sol, mg_temp);
       mg_temp -= mg_res;
       mg_temp *= -1;
       // FIXME: This zeroing is too much on paper, but without it error at the boundary is reported
       fine_timestepping->ZeroBndValues(mg_temp);
       //mg_temp.Print();

       mg_res = mg_temp;

       for (int tslab = 0; tslab < nslabs; ++tslab)
       {
           std::cout << "mg_res after iterate, tslab = " << tslab << "\n";
           std::cout << "norm = " << mg_res_viewer.GetBlock(tslab).Norml2() /
                        sqrt (mg_res_viewer.GetBlock(tslab).Size()) << "\n";

           //for (int i = 0; i < mg_res_viewer.GetBlock(tslab).Size(); ++i)
               //if (fabs(mg_res_viewer.GetBlock(tslab)[i]) > 1.0e-10)
                   //std::cout << "entry " << i << ": res value = " << mg_res_viewer.GetBlock(tslab)[i] << "\n";

           //std::cout << "values of res at bottom interface: \n";
           //Vector& vec = fine_timestepping->GetProblem(tslab)->ExtractAtBase("bot", mg_res_viewer.GetBlock(tslab));
           //vec.Print();
       }

       res_norm = mg_res.Norml2() / sqrt (mg_res.Size());

       // check convergence
       if (res_norm < eps * res0_norm)
           converged = true;

       // output convergence status
       if (verbose)
       {
           std::cout << "Iteration " << iter << ": res_norm = " << res_norm << "\n";
           fine_timestepping->ComputeError(mg_finalsol);
           fine_timestepping->ComputeBndError(mg_finalsol);
       }
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

   MPI_Finalize();
   return 0;

   /*

   if (verbose)
      std::cout << "Checking a single solve from a hierarchy of problems "
                    "created for the entire domain \n";

//#if 0
  {
      int pref_lvls_tslab = 0;
      int solve_at_lvl = 0;
      //TimeCylHyper * timeslab_test = new TimeCylHyper (*pmeshbase, 0.0, tau, Nt, pref_lvls_tslab,
                                                         //formulation, space_for_S, space_for_sigma);
      TimeCylHyper * timeslab_test = new TimeCylHyper (*pmesh, pref_lvls_tslab,
                                                         formulation, space_for_S, space_for_sigma, numsol);

      int init_cond_size = timeslab_test->GetInitCondSize(solve_at_lvl);
      std::vector<std::pair<int,int> > * tdofs_link = timeslab_test->GetTdofsLink(solve_at_lvl);
      Vector Xinit(init_cond_size);
      ParFiniteElementSpace * testfespace;
      ParGridFunction * sol_exact;
      if (strcmp(space_for_S,"H1") == 0) // S is present
      {
          testfespace = timeslab_test->Get_S_space(solve_at_lvl);
          sol_exact = new ParGridFunction(testfespace);
          sol_exact->ProjectCoefficient(*Mytest.scalarS);
      }
      else
      {
          testfespace = timeslab_test->Get_Sigma_space(solve_at_lvl);
          sol_exact = new ParGridFunction(testfespace);
          sol_exact->ProjectCoefficient(*Mytest.sigma);
      }

      Vector sol_exact_truedofs(testfespace->TrueVSize());
      sol_exact->ParallelProject(sol_exact_truedofs);

      for (int i = 0; i < init_cond_size; ++i)
      {
          int tdof_bot = (*tdofs_link)[i].first;
          Xinit[i] = sol_exact_truedofs[tdof_bot];
      }

      //Xinit.Print();

      Vector Xout(init_cond_size);

      timeslab_test->Solve(solve_at_lvl, Xinit, Xout);

      timeslab_test->ComputeError(solve_at_lvl, *timeslab_test->GetSol(solve_at_lvl));

      //MPI_Finalize();
      //return 0;

      // checking the error at the top boundary
      Vector Xout_exact(init_cond_size);
      for (int i = 0; i < init_cond_size; ++i)
      {
          int tdof_top = (*tdofs_link)[i].second;
          Xout_exact[i] = sol_exact_truedofs[tdof_top];
      }

      Vector Xout_error(init_cond_size);
      Xout_error = Xout;
      Xout_error -= Xout_exact;
      if (verbose)
      {
          std::cout << "|| Xout  - Xout_exact || = " << Xout_error.Norml2() / sqrt (Xout_error.Size()) << "\n";
          std::cout << "|| Xout  - Xout_exact || / || Xout_exact || = " << (Xout_error.Norml2() / sqrt (Xout_error.Size())) /
                       (Xout_exact.Norml2() / sqrt (Xout_exact.Size()))<< "\n";
      }

      // testing InterpolateAtBase()
      if (solve_at_lvl == 1 && strcmp(space_for_S,"H1") == 0)
      {
          Vector Xout_fine(timeslab_test->GetInitCondSize(0));
          timeslab_test->InterpolateAtBase("top", 0, Xout, Xout_fine);

          Vector Xout_truedofs(timeslab_test->Get_S_space(solve_at_lvl)->TrueVSize());
          Xout_truedofs = 0.0;
          for (unsigned int i = 0; i < tdofs_link->size(); ++i)
          {
              Xout_truedofs[(*tdofs_link)[i].second] = Xout[i];
          }
          ParGridFunction * Xout_dofs = new ParGridFunction(timeslab_test->Get_S_space(solve_at_lvl));
          Xout_dofs->Distribute(&Xout_truedofs);

          std::vector<std::pair<int,int> > * tdofs_fine_link = timeslab_test->GetTdofsLink(0);
          Vector Xout_fine_truedofs(timeslab_test->Get_S_space(0)->TrueVSize());
          Xout_fine_truedofs = 0.0;
          for (unsigned int i = 0; i < tdofs_fine_link->size(); ++i)
          {
              Xout_fine_truedofs[(*tdofs_fine_link)[i].second] = Xout_fine[i];
          }
          ParGridFunction * Xout_fine_dofs = new ParGridFunction(timeslab_test->Get_S_space(0));
          Xout_fine_dofs->Distribute(&Xout_fine_truedofs);

          ParMeshCyl * pmeshcyl_coarse = timeslab_test->GetParMeshCyl(solve_at_lvl);

          //std::cout << "pmeshcyl_coarse ne = " << pmeshcyl_coarse->GetNE() << "\n";
          ParMeshCyl * pmeshcyl_fine = timeslab_test->GetParMeshCyl(0);

          char vishost[] = "localhost";
          int  visport   = 19916;

          socketstream uuu_sock(vishost, visport);
          uuu_sock << "parallel " << num_procs << " " << myid << "\n";
          uuu_sock.precision(8);
          uuu_sock << "solution\n" << *pmeshcyl_coarse <<
                      *Xout_dofs << "window_title 'Xout coarse'"
                 << endl;

          socketstream s_sock(vishost, visport);
          s_sock << "parallel " << num_procs << " " << myid << "\n";
          s_sock.precision(8);
          MPI_Barrier(comm);
          s_sock << "solution\n" << *pmeshcyl_fine <<
                    *Xout_fine_dofs << "window_title 'Xout fine'"
                  << endl;

      }


      delete timeslab_test;
  }
//#endif

   MPI_Finalize();
   return 0;

   */

   //CFOSLSHyperbolicFormulation problem_structure(dim, numsol, space_for_S, space_for_sigma, true, pmesh->bdr_attributes.Max(), verbose);
   //CFOSLSHyperbolicProblem problem2(*pmesh, problem_structure, feorder, prec_option, verbose);
   //problem2.Solve(verbose);

  if (verbose)
    std::cout << "Checking a sequential solve within several TimeCylHyper instances \n";

  {
      int pref_lvls_tslab = 0;
      int solve_at_lvl = 0;

      int nslabs = 2;
      std::vector<TimeCylHyper*> timeslabs(nslabs);
      double slab_tau = 0.125;
      int slab_width = 4; // in time steps (as time intervals) withing a single time slab
      double tinit_tslab = 0.0;

      if (verbose)
      {
          std::cout << "Creating a sequence of time slabs: \n";
          std::cout << "# of slabs: " << nslabs << "\n";
          std::cout << "# of time intervals per slab: " << slab_width << "\n";
          std::cout << "time step within a time slab: " << slab_tau << "\n";
          std::cout << "# of refinements: " << pref_lvls_tslab << "\n";
          if (solve_at_lvl == 0)
              std::cout << "solution level: " << solve_at_lvl << "\n";
      }

      for (int tslab = 0; tslab < nslabs; ++tslab )
      {
          timeslabs[tslab] = new TimeCylHyper (*pmeshbase, tinit_tslab, slab_tau, slab_width, pref_lvls_tslab,
                                                formulation, space_for_S, space_for_sigma, numsol);
          tinit_tslab += slab_tau * slab_width;
      }

      MFEM_ASSERT(fabs(tinit_tslab - 1.0) < 1.0e-14, "The slabs should cover the time interval "
                                                    "[0,1] but the upper bound doesn't match \n");

      Vector Xinit;

      int init_cond_size = timeslabs[0]->GetInitCondSize(solve_at_lvl);
      std::vector<std::pair<int,int> > * tdofs_link = timeslabs[0]->GetTdofsLink(solve_at_lvl);
      Xinit.SetSize(init_cond_size);

      ParFiniteElementSpace * testfespace;
      ParGridFunction * sol_exact;

      if (strcmp(space_for_S,"H1") == 0) // S is present
      {
          testfespace = timeslabs[0]->Get_S_space(solve_at_lvl);
          sol_exact = new ParGridFunction(testfespace);
          sol_exact->ProjectCoefficient(*Mytest.scalarS);
      }
      else
      {
          testfespace = timeslabs[0]->Get_Sigma_space(solve_at_lvl);
          sol_exact = new ParGridFunction(testfespace);
          sol_exact->ProjectCoefficient(*Mytest.sigma);
      }

      Vector sol_exact_truedofs(testfespace->TrueVSize());
      sol_exact->ParallelProject(sol_exact_truedofs);

      for (int i = 0; i < init_cond_size; ++i)
      {
          int tdof_bot = (*tdofs_link)[i].first;
          Xinit[i] = sol_exact_truedofs[tdof_bot];
      }

      // initializing the input boundary condition for the first vector

      Vector Xout(init_cond_size);

      for (int tslab = 0; tslab < nslabs; ++tslab )
      {
          //Xinit.Print();
          timeslabs[tslab]->Solve(solve_at_lvl, Xinit, Xout);

          timeslabs[tslab]->ComputeError(solve_at_lvl, *timeslabs[tslab]->GetSol(solve_at_lvl));

          Xinit = Xout;
          if (strcmp(space_for_S,"L2") == 0)
              Xinit *= -1.0;

          // checking the error at the top boundary
          Vector Xout_exact(init_cond_size);
          Vector sol_exact_truedofs;
          {
              ParFiniteElementSpace * testfespace;
              ParGridFunction * sol_exact;

              if (strcmp(space_for_S,"H1") == 0) // S is present
              {
                  testfespace = timeslabs[tslab]->Get_S_space(solve_at_lvl);
                  sol_exact = new ParGridFunction(testfespace);
                  sol_exact->ProjectCoefficient(*Mytest.scalarS);
              }
              else
              {
                  testfespace = timeslabs[tslab]->Get_Sigma_space(solve_at_lvl);
                  sol_exact = new ParGridFunction(testfespace);
                  sol_exact->ProjectCoefficient(*Mytest.sigma);
              }

              sol_exact_truedofs.SetSize(testfespace->TrueVSize());
              sol_exact->ParallelProject(sol_exact_truedofs);
          }

          for (int i = 0; i < init_cond_size; ++i)
          {
              int tdof_top = (*tdofs_link)[i].second;
              Xout_exact[i] = sol_exact_truedofs[tdof_top];
          }

          Vector Xout_error(init_cond_size);
          Xout_error = Xout;
          Xout_error -= Xout_exact;
          if (verbose)
          {
              std::cout << "|| Xout  - Xout_exact || = " << Xout_error.Norml2() / sqrt (Xout_error.Size()) << "\n";
              std::cout << "|| Xout  - Xout_exact || / || Xout_exact || = " << (Xout_error.Norml2() / sqrt (Xout_error.Size())) /
                           (Xout_exact.Norml2() / sqrt (Xout_exact.Size()))<< "\n";
          }

      }
  }

  MPI_Finalize();
  return 0;

  /*

  if (verbose)
    std::cout << "Checking a sequential solve within a TimeStepping instance \n";

  {
      int pref_lvls_tslab = 1;

      int nslabs = 2;
      std::vector<TimeCylHyper*> timeslabs(nslabs);
      double slab_tau = 0.125;
      int slab_width = 4; // in time steps (as time intervals) withing a single time slab
      double tinit_tslab = 0.0;

      int solve_at_lvl = 0;

      if (verbose)
      {
          std::cout << "Creating a sequence of time slabs: \n";
          std::cout << "# of slabs: " << nslabs << "\n";
          std::cout << "# of time intervals per slab: " << slab_width << "\n";
          std::cout << "time step within a time slab: " << slab_tau << "\n";
          std::cout << "# of refinements: " << pref_lvls_tslab << "\n";
          if (solve_at_lvl == 0)
              std::cout << "solution level: " << solve_at_lvl << "\n";
      }

      for (int tslab = 0; tslab < nslabs; ++tslab )
      {
          timeslabs[tslab] = new TimeCylHyper (*pmeshbase, tinit_tslab, slab_tau, slab_width, pref_lvls_tslab,
                                                formulation, space_for_S, space_for_sigma);
          tinit_tslab += slab_tau * slab_width;
      }

      MFEM_ASSERT(fabs(tinit_tslab - 1.0) < 1.0e-14, "The slabs should cover the time interval "
                                                    "[0,1] but the upper bound doesn't match \n");

      TimeSteppingScheme * timestepping = new TimeSteppingScheme(timeslabs);

      Vector Xinit;

      int init_cond_size = timestepping->GetTimeSlab(0)->GetInitCondSize(solve_at_lvl);
      std::vector<std::pair<int,int> > * tdofs_link = timestepping->GetTimeSlab(0)->GetTdofsLink(solve_at_lvl);
      Xinit.SetSize(init_cond_size);

      ParFiniteElementSpace * testfespace;
      ParGridFunction * sol_exact;

      if (strcmp(space_for_S,"H1") == 0) // S is present
      {
          testfespace = timestepping->GetTimeSlab(0)->Get_S_space(solve_at_lvl);
          sol_exact = new ParGridFunction(testfespace);
          sol_exact->ProjectCoefficient(*Mytest.scalarS);
      }
      else
      {
          testfespace = timestepping->GetTimeSlab(0)->Get_Sigma_space(solve_at_lvl);
          sol_exact = new ParGridFunction(testfespace);
          sol_exact->ProjectCoefficient(*Mytest.sigma);
      }

      Vector sol_exact_truedofs(testfespace->TrueVSize());
      sol_exact->ParallelProject(sol_exact_truedofs);

      for (int i = 0; i < init_cond_size; ++i)
      {
          int tdof_bot = (*tdofs_link)[i].first;
          Xinit[i] = sol_exact_truedofs[tdof_bot];
      }



      // initializing the input boundary condition for the first vector
      timestepping->SetInitialCondition(X_init, solve_at_lvl);

      timestepping->ComputeAnalyticalRhs(solve_at_lvl); ...

      timestepping->Solve("sequential", "regular", rhss, solve_at_lvl, true);
  }

  MPI_Finalize();
  return 0;
  */

  if (verbose)
    std::cout << "Checking a two-grid scheme with independent fine and sequential coarse solvers \n";
  {
      int fine_lvl = 0;
      int coarse_lvl = 1;

      int pref_lvls_tslab = 1;

      int nslabs = 2;
      std::vector<TimeCylHyper*> timeslabs(nslabs);
      double slab_tau = 0.125;
      int slab_width = 4; // in time steps (as time intervals) withing a single time slab

      if (verbose)
      {
          std::cout << "Creating a sequence of time slabs: \n";
          std::cout << "# of slabs: " << nslabs << "\n";
          std::cout << "# of time intervals per slab: " << slab_width << "\n";
          std::cout << "time step within a time slab: " << slab_tau << "\n";
          std::cout << "# of refinements: " << pref_lvls_tslab << "\n";
      }

      double tinit_tslab = 0.0;
      for (int tslab = 0; tslab < nslabs; ++tslab )
      {
          timeslabs[tslab] = new TimeCylHyper (*pmeshbase, tinit_tslab, slab_tau, slab_width, pref_lvls_tslab,
                                                formulation, space_for_S, space_for_sigma, numsol);
          tinit_tslab += slab_tau * slab_width;
      }

      MFEM_ASSERT(fabs(tinit_tslab - 1.0) < 1.0e-14, "The slabs should cover the time interval "
                                                    "[0,1] but the upper bound doesn't match \n");

      // getting some approximations for the first iteration from the coarse solver
      // sequential coarse solve
      if (verbose)
          std::cout << "Sequential coarse solve: \n";

      std::vector<Vector*> Xouts_coarse(nslabs + 1);
      int solve_at_lvl = 1;

      Vector Xinit;
      // initializing the input boundary condition for the first vector
      int init_cond_size = timeslabs[0]->GetInitCondSize(solve_at_lvl);
      std::vector<std::pair<int,int> > * tdofs_link = timeslabs[0]->GetTdofsLink(solve_at_lvl);
      Xinit.SetSize(init_cond_size);

      ParFiniteElementSpace * testfespace;
      ParGridFunction * sol_exact;

      if (strcmp(space_for_S,"H1") == 0) // S is present
      {
          testfespace = timeslabs[0]->Get_S_space(solve_at_lvl);
          sol_exact = new ParGridFunction(testfespace);
          sol_exact->ProjectCoefficient(*Mytest.scalarS);
      }
      else
      {
          testfespace = timeslabs[0]->Get_Sigma_space(solve_at_lvl);
          sol_exact = new ParGridFunction(testfespace);
          sol_exact->ProjectCoefficient(*Mytest.sigma);
      }

      Vector sol_exact_truedofs(testfespace->TrueVSize());
      sol_exact->ParallelProject(sol_exact_truedofs);

      for (int i = 0; i < init_cond_size; ++i)
      {
          int tdof_bot = (*tdofs_link)[i].first;
          Xinit[i] = sol_exact_truedofs[tdof_bot];
      }

      Vector Xout(init_cond_size);

      Xouts_coarse[0] = new Vector(init_cond_size);
      (*Xouts_coarse[0]) = Xinit;

      for (int tslab = 0; tslab < nslabs; ++tslab )
      {
          timeslabs[tslab]->Solve(solve_at_lvl, Xinit, Xout);
          Xinit = Xout;
          if (strcmp(space_for_S,"L2") == 0)
              Xinit *= -1.0;

          Xouts_coarse[tslab + 1] = new Vector(init_cond_size);
          (*Xouts_coarse[tslab]) = Xinit;


          Vector Xout_exact(init_cond_size);

          // checking the error at the top boundary
          ParFiniteElementSpace * testfespace;
          ParGridFunction * sol_exact;

          if (strcmp(space_for_S,"H1") == 0) // S is present
          {
              testfespace = timeslabs[0]->Get_S_space(solve_at_lvl);
              sol_exact = new ParGridFunction(testfespace);
              sol_exact->ProjectCoefficient(*Mytest.scalarS);
          }
          else
          {
              testfespace = timeslabs[0]->Get_Sigma_space(solve_at_lvl);
              sol_exact = new ParGridFunction(testfespace);
              sol_exact->ProjectCoefficient(*Mytest.sigma);
          }

          Vector sol_exact_truedofs(testfespace->TrueVSize());
          sol_exact->ParallelProject(sol_exact_truedofs);

          for (int i = 0; i < init_cond_size; ++i)
          {
              int tdof_top = (*tdofs_link)[i].second;
              Xout_exact[i] = sol_exact_truedofs[tdof_top];
          }

          Vector Xout_error(init_cond_size);
          Xout_error = Xout;
          Xout_error -= Xout_exact;
          if (verbose)
          {
              std::cout << "|| Xout  - Xout_exact || = " << Xout_error.Norml2() / sqrt (Xout_error.Size()) << "\n";
              std::cout << "|| Xout  - Xout_exact || / || Xout_exact || = " << (Xout_error.Norml2() / sqrt (Xout_error.Size())) /
                           (Xout_exact.Norml2() / sqrt (Xout_exact.Size()))<< "\n";
          }

      } // end of loop over all time slabs, performing a coarse solve

      if (verbose)
          std::cout << "Creating initial data for the two-grid method \n";
      solve_at_lvl = 0;

      std::vector<Vector*> Xinits_fine(nslabs + 1);
      std::vector<Vector*> Xouts_fine(nslabs + 1);

      int init_cond_size_fine = timeslabs[0]->GetInitCondSize(solve_at_lvl);

      Xinits_fine[0] = new Vector(init_cond_size_fine);
      timeslabs[0]->InterpolateAtBase("bot", solve_at_lvl, *Xouts_coarse[0], *Xinits_fine[0]);
      Xouts_fine[0] = new Vector(init_cond_size_fine);
      *Xouts_fine[0] = *Xinits_fine[0];

      for (int tslab = 0; tslab < nslabs; ++tslab )
      {
          Xinits_fine[tslab + 1] = new Vector(init_cond_size_fine);

          // interpolate Xouts_coarse on the finer mesh into Xinits_fine
          timeslabs[tslab]->InterpolateAtBase("top", solve_at_lvl, *Xouts_coarse[tslab + 1], *Xinits_fine[tslab + 1]);

          Xouts_fine[tslab + 1] = new Vector(init_cond_size_fine);
      }

      // now we have Xinits_fine as initial conditions for the fine grid solves

      if (verbose)
          std::cout << "Starting two-grid iterations ... \n";

      int numlvls = pref_lvls_tslab + 1;
      MFEM_ASSERT(numlvls == 2, "Current implementation allows only a two-grid scheme \n");
      std::vector<Array<Vector*>*> residuals_lvls(numlvls);
      std::vector<Array<Vector*>*> corr_lvls(numlvls);
      std::vector<Array<Vector*>*> sol_lvls(numlvls);
      for (unsigned int i = 0; i < residuals_lvls.size(); ++i)
      {
          residuals_lvls[i] = new Array<Vector*>(nslabs);
          corr_lvls[i] = new Array<Vector*>(nslabs);
          sol_lvls[i] = new Array<Vector*>(nslabs);
          for (int tslab = 0; tslab < nslabs; ++tslab )
          {
              (*residuals_lvls[i])[tslab] = new Vector(timeslabs[tslab]->ProblemSize(i));
              (*corr_lvls[i])[tslab] = new Vector(timeslabs[tslab]->ProblemSize(i));
              (*sol_lvls[i])[tslab] = new Vector(timeslabs[tslab]->ProblemSize(i));
              *(*sol_lvls[i])[tslab] = 0.0;
          }
      }

      for (int it = 0; it < 2; ++it)
      {
          // 1. parallel-in-time smoothing
          for (int tslab = 0; tslab < nslabs; ++tslab )
          {
              timeslabs[tslab]->Solve(fine_lvl, *Xinits_fine[tslab], *Xouts_fine[tslab]);
              *(*sol_lvls[fine_lvl])[tslab] = *(timeslabs[tslab]->GetSol(fine_lvl));
              if (tslab > 0)
                  timeslabs[tslab]->ComputeResidual(fine_lvl, *Xouts_fine[tslab - 1], *(*sol_lvls[fine_lvl])[tslab],
                          *(*residuals_lvls[fine_lvl])[tslab]);
              else
                  timeslabs[tslab]->ComputeResidual(fine_lvl, *Xinits_fine[0], *(*sol_lvls[fine_lvl])[0],
                          *(*residuals_lvls[fine_lvl])[0]);

              //(*residuals_lvls[fine_lvl])[tslab]->Print();
              //;
          }


          // 2. projecting onto coarse space
          for (int tslab = 0; tslab < nslabs; ++tslab )
          {
              timeslabs[tslab]->Restrict(fine_lvl, *(*residuals_lvls[fine_lvl])[tslab], *(*residuals_lvls[coarse_lvl])[tslab]);
          }

          // 3. coarse problem solve
          Xinit = 0.0;
          for (int tslab = 0; tslab < nslabs; ++tslab )
          {
              //(*residuals_lvls[coarse_lvl])[tslab]->Print();

              timeslabs[tslab]->Solve("coarsened", coarse_lvl, *(*residuals_lvls[coarse_lvl])[tslab],
                                      *(*corr_lvls[coarse_lvl])[tslab], Xinit, Xout);
              Xinit = Xout;
              if (strcmp(space_for_S,"L2") == 0)
                  Xinit *= -1.0;

              (*Xouts_coarse[tslab]) = Xinit;

          } // end of loop over all time slabs, performing a coarse solve

          // 4. interpolating back and updating the solution
          for (int tslab = 0; tslab < nslabs; ++tslab )
          {
              //for (int i = 0; i < (*corr_lvls[coarse_lvl])[tslab]->Size(); ++i)
                  //std::cout << "corr coarse = " << (*(*corr_lvls[coarse_lvl])[tslab])[i] << "\n";

              //(*corr_lvls[coarse_lvl])[tslab]->Print();

              timeslabs[tslab]->Interpolate(fine_lvl, *(*corr_lvls[coarse_lvl])[tslab], *(*corr_lvls[fine_lvl])[tslab]);
          }

          // computing error in each time slab before update
          if (verbose)
              std::cout << "Errors before adding the coarse grid corrections \n";
          for (int tslab = 0; tslab < nslabs; ++tslab )
          {
              timeslabs[tslab]->ComputeError(fine_lvl, *(*sol_lvls[fine_lvl])[tslab]);
          }

          for (int tslab = 0; tslab < nslabs; ++tslab )
          {
              //for (int i = 0; i < (*sol_lvls[fine_lvl])[tslab]->Size(); ++i)
                  //std::cout << "sol before = " << (*(*sol_lvls[fine_lvl])[tslab])[i] <<
                               //", corr = " << (*(*corr_lvls[fine_lvl])[tslab])[i] << "\n";
              *(*sol_lvls[fine_lvl])[tslab] += *(*corr_lvls[fine_lvl])[tslab];
          }

          // 4.5 computing error in each time slab
          if (verbose)
              std::cout << "Errors after adding the coarse grid corrections \n";
          for (int tslab = 0; tslab < nslabs; ++tslab )
          {
              timeslabs[tslab]->ComputeError(fine_lvl, *(*sol_lvls[fine_lvl])[tslab]);
          }


          // 5. update initial conditions to start the next iteration
          int bdrcond_block = -1;
          if (strcmp(space_for_S,"H1") == 0) // S is present
              bdrcond_block = 1;
          else
              bdrcond_block = 0;

          for (int tslab = 0; tslab < nslabs; ++tslab )
          {
              std::vector<std::pair<int,int> > * tdofs_link = timeslabs[tslab]->GetTdofsLink(fine_lvl);
              *Xinits_fine[tslab] = 0.0;

              BlockVector sol_viewer((*sol_lvls[fine_lvl])[tslab]->GetData(),
                                     *timeslabs[tslab]->GetBlockTrueOffsets(fine_lvl));

              // FIXME: We have actually two values at all interfaces from two time slabs
              // Here I simply chose the time slab which is above the interface
              for (int i = 0; i < init_cond_size_fine; ++i)
              {
                  int tdof_bot = (*tdofs_link)[i].first;
                  (*Xinits_fine[tslab])[i] = sol_viewer.GetBlock(bdrcond_block)[tdof_bot];
              }

          }

      }

  }



#if 0
  if (verbose)
    std::cout << "Checking a sequential coarse solve with following ~parallel fine solves \n";

  {
      int pref_lvls_tslab = 1;

      int nslabs = 2;
      std::vector<TimeCylHyper*> timeslabs(nslabs);
      double slab_tau = 0.125;
      int slab_width = 4; // in time steps (as time intervals) withing a single time slab

      if (verbose)
      {
          std::cout << "Creating a sequence of time slabs: \n";
          std::cout << "# of slabs: " << nslabs << "\n";
          std::cout << "# of time intervals per slab: " << slab_width << "\n";
          std::cout << "time step within a time slab: " << slab_tau << "\n";
          std::cout << "# of refinements: " << pref_lvls_tslab << "\n";
      }

      double tinit_tslab = 0.0;
      for (int tslab = 0; tslab < nslabs; ++tslab )
      {
          timeslabs[tslab] = new TimeCylHyper (*pmeshbase, tinit_tslab, slab_tau, slab_width, pref_lvls_tslab,
                                                formulation, space_for_S, space_for_sigma);
          tinit_tslab += slab_tau * slab_width;
      }

      MFEM_ASSERT(fabs(tinit_tslab - 1.0) < 1.0e-14, "The slabs should cover the time interval "
                                                    "[0,1] but the upper bound doesn't match \n");


      // sequential coarse solve
      if (verbose)
          std::cout << "Sequential coarse solve: \n";

      std::vector<Vector*> Xouts_coarse(nslabs + 1);
      int solve_at_lvl = 1;

      Vector Xinit;
      // initializing the input boundary condition for the first vector
      int init_cond_size = timeslabs[0]->GetInitCondSize(solve_at_lvl);
      std::vector<std::pair<int,int> > * tdofs_link = timeslabs[0]->GetTdofsLink(solve_at_lvl);
      Xinit.SetSize(init_cond_size);

      ParFiniteElementSpace * testfespace;
      ParGridFunction * sol_exact;

      if (strcmp(space_for_S,"H1") == 0) // S is present
      {
          testfespace = timeslabs[0]->Get_S_space(solve_at_lvl);
          sol_exact = new ParGridFunction(testfespace);
          sol_exact->ProjectCoefficient(*Mytest.scalarS);
      }
      else
      {
          testfespace = timeslabs[0]->Get_Sigma_space(solve_at_lvl);
          sol_exact = new ParGridFunction(testfespace);
          sol_exact->ProjectCoefficient(*Mytest.sigma);
      }

      Vector sol_exact_truedofs(testfespace->TrueVSize());
      sol_exact->ParallelProject(sol_exact_truedofs);

      for (int i = 0; i < init_cond_size; ++i)
      {
          int tdof_bot = (*tdofs_link)[i].first;
          Xinit[i] = sol_exact_truedofs[tdof_bot];
      }

      Vector Xout(init_cond_size);

      Xouts_coarse[0] = new Vector(init_cond_size);
      (*Xouts_coarse[0]) = Xinit;

      for (int tslab = 0; tslab < nslabs; ++tslab )
      {
          timeslabs[tslab]->Solve(solve_at_lvl, Xinit, Xout);
          Xinit = Xout;
          if (strcmp(space_for_S,"L2") == 0)
              Xinit *= -1.0;

          Xouts_coarse[tslab + 1] = new Vector(init_cond_size);
          (*Xouts_coarse[tslab]) = Xinit;


          Vector Xout_exact(init_cond_size);

          // checking the error at the top boundary
          ParFiniteElementSpace * testfespace;
          ParGridFunction * sol_exact;

          if (strcmp(space_for_S,"H1") == 0) // S is present
          {
              testfespace = timeslabs[0]->Get_S_space(solve_at_lvl);
              sol_exact = new ParGridFunction(testfespace);
              sol_exact->ProjectCoefficient(*Mytest.scalarS);
          }
          else
          {
              testfespace = timeslabs[0]->Get_Sigma_space(solve_at_lvl);
              sol_exact = new ParGridFunction(testfespace);
              sol_exact->ProjectCoefficient(*Mytest.sigma);
          }

          Vector sol_exact_truedofs(testfespace->TrueVSize());
          sol_exact->ParallelProject(sol_exact_truedofs);

          for (int i = 0; i < init_cond_size; ++i)
          {
              int tdof_top = (*tdofs_link)[i].second;
              Xout_exact[i] = sol_exact_truedofs[tdof_top];
          }

          Vector Xout_error(init_cond_size);
          Xout_error = Xout;
          Xout_error -= Xout_exact;
          if (verbose)
          {
              std::cout << "|| Xout  - Xout_exact || = " << Xout_error.Norml2() / sqrt (Xout_error.Size()) << "\n";
              std::cout << "|| Xout  - Xout_exact || / || Xout_exact || = " << (Xout_error.Norml2() / sqrt (Xout_error.Size())) /
                           (Xout_exact.Norml2() / sqrt (Xout_exact.Size()))<< "\n";
          }

      } // end of loop over all time slabs, performing a coarse solve

      if (verbose)
          std::cout << "Creating initial data for fine grid solves \n";
      solve_at_lvl = 0;

      std::vector<Vector*> Xinits_fine(nslabs + 1);
      std::vector<Vector*> Xouts_fine(nslabs + 1);

      int init_cond_size_fine = timeslabs[0]->GetInitCondSize(solve_at_lvl);

      Xinits_fine[0] = new Vector(init_cond_size_fine);
      timeslabs[0]->InterpolateAtBase("bot", solve_at_lvl, *Xouts_coarse[0], *Xinits_fine[0]);
      Xouts_fine[0] = new Vector(init_cond_size_fine);
      *Xouts_fine[0] = *Xinits_fine[0];

      for (int tslab = 0; tslab < nslabs; ++tslab )
      {
          Xinits_fine[tslab + 1] = new Vector(init_cond_size_fine);

          // interpolate Xouts_coarse on the finer mesh into Xinits_fine
          timeslabs[tslab]->InterpolateAtBase("top", solve_at_lvl, *Xouts_coarse[tslab + 1], *Xinits_fine[tslab + 1]);

          Xouts_fine[tslab + 1] = new Vector(init_cond_size_fine);
      }


      if (verbose)
          std::cout << "Solving fine grid problems \n";

      // can be done in parallel, instead of for loop since the fine grid problems are independent
      solve_at_lvl = 0;
      for (int tslab = 0; tslab < nslabs; ++tslab )
      {
          timeslabs[tslab]->Solve(solve_at_lvl, *Xinits_fine[tslab], *Xouts_fine[tslab]);
          /*
          Xinit = Xout;
          if (strcmp(space_for_S,"L2") == 0)
              Xinit *= -1.0;

          Vector Xout_exact(init_cond_size);

          // checking the error at the top boundary
          ParFiniteElementSpace * testfespace;
          ParGridFunction * sol_exact;

          if (strcmp(space_for_S,"H1") == 0) // S is present
          {
              testfespace = timeslabs[0]->Get_S_space(solve_at_lvl);
              sol_exact = new ParGridFunction(testfespace);
              sol_exact->ProjectCoefficient(*Mytest.scalarS);
          }
          else
          {
              testfespace = timeslabs[0]->Get_Sigma_space(solve_at_lvl);
              sol_exact = new ParGridFunction(testfespace);
              sol_exact->ProjectCoefficient(*Mytest.sigma);
          }

          Vector sol_exact_truedofs(testfespace->TrueVSize());
          sol_exact->ParallelProject(sol_exact_truedofs);

          for (int i = 0; i < init_cond_size; ++i)
          {
              int tdof_top = (*tdofs_link)[i].second;
              Xout_exact[i] = sol_exact_truedofs[tdof_top];
          }

          Vector Xout_error(init_cond_size);
          Xout_error = Xout;
          Xout_error -= Xout_exact;
          if (verbose)
          {
              std::cout << "|| Xout  - Xout_exact || = " << Xout_error.Norml2() / sqrt (Xout_error.Size()) << "\n";
              std::cout << "|| Xout  - Xout_exact || / || Xout_exact || = " << (Xout_error.Norml2() / sqrt (Xout_error.Size())) /
                           (Xout_exact.Norml2() / sqrt (Xout_exact.Size()))<< "\n";
          }
          */
      } // end of loop over all time slabs, performing fine solves

      // Computing the corrections as the coarse level initial conditions
      if (verbose)
          std::cout << "Computing corrections \n";

      std::vector<Vector*> Xcorrs_coarse(nslabs + 1);

      solve_at_lvl = 1;

      int init_cond_size_coarse = timeslabs[0]->GetInitCondSize(solve_at_lvl);

      Xcorrs_coarse[0] = new Vector(init_cond_size_coarse);
      timeslabs[0]->RestrictAtBase("bot", solve_at_lvl, *Xouts_fine[0], *Xcorrs_coarse[0]);

      for (int tslab = 0; tslab < nslabs; ++tslab )
      {
          Xcorrs_coarse[tslab + 1] = new Vector(init_cond_size_coarse);

          // restricts Xouts_fine to the coarser mesh into Xcorrs_coarse
          timeslabs[tslab]->RestrictAtBase("top", solve_at_lvl, *Xouts_fine[tslab], *Xcorrs_coarse[tslab + 1]);
      }

      for (int t = 0; t <= nslabs; ++t )
      {
          *Xcorrs_coarse[t] += *Xouts_coarse[t];
      }

      // prolongating correstiions in time
      if (verbose)
          std::cout << "Prolongating corrections \n";



  }// end of block of testing a parallel-in-time solver
#endif

   // 17. Free the used memory.
   MPI_Barrier(comm);

   MPI_Finalize();
   return 0;
}

