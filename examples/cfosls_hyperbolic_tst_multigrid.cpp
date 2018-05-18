///                       CFOSLS formulation for transport equation in 3D/4D
///                         solved with a parallel-in-time multigrid
///                             (similar to parareal and XBraid)

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>
#include <unistd.h>

#define MYZEROTOL (1.0e-13)
#define ZEROTOL (1.0e-13)

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

   //pmesh->PrintSlabsStruct();

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
   using ProblemType = FOSLSCylProblem_HdivL2L2hyp;
   */

   FOSLSFormulation * formulat = new FormulType (dim, numsol, verbose);
   FOSLSFEFormulation * fe_formulat = new FEFormulType(*formulat, feorder);
   BdrConditions * bdr_conds = new BdrCondsType(*pmesh);

   //ProblemType * problem = new ProblemType
           //(*pmesh, *bdr_conds, *fe_formulat, prec_option, verbose);

   //problem->Solve(verbose);

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

   int nslabs = 2;//4;//2;
   double slab_tau = 0.125;//1.0/16;//0.125;
   int slab_width = 4; // in time steps (as time intervals) withing a single time slab
   Array<ParMeshCyl*> timeslabs_pmeshcyls(nslabs);
   //Array<ProblemType*> timeslabs_problems(nslabs);

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

       //timeslabs_problems[tslab] = new ProblemType(*timeslabs_pmeshcyls[tslab], *bdr_conds, *fe_formulat, prec_option, verbose);
       //timeslabs_problems[tslab] = new ProblemType(*timeslabs_pmeshcyls[tslab], *bdr_conds, *fe_formulat, prec_option, verbose);

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

   TimeStepping<ProblemType> * coarse_timestepping = twogrid_tstp->GetCoarseTimeStp();

   Array<Operator*> P_tstp(1);
   P_tstp[0] = twogrid_tstp->GetGlobalInterpolationOp();
   //P_tstp[0] = twogrid_tstp->GetGlobalInterpolationOpWithBnd();

   // creating fine-level operator, smoother and coarse-level operator
   Array<Operator*> Ops_tstp(1);
   Array<Operator*> Smoo_tstp(1);
   Array<Operator*> NullSmoo_tstp(1);
   Operator* CoarseOp_tstp;

   Ops_tstp[0] =
           new TimeSteppingSeqOp<ProblemType>(*fine_timestepping, verbose);

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

   Smoo_tstp[0] =
           new TimeSteppingSmoother<ProblemType> (*fine_timestepping, verbose);

   CoarseOp_tstp =
           new TimeSteppingSolveOp<ProblemType>(*coarse_timestepping, verbose);
   //CoarseOp_tstp =
           //new TSTSpecialSolveOp<ProblemType>(*coarse_timestepping, verbose);
   NullSmoo_tstp[0] = NULL;

   // finally, creating general multigrid instance
   GeneralMultigrid * spacetime_mg =
           new GeneralMultigrid(two_grid, P_tstp, Ops_tstp, *CoarseOp_tstp, Smoo_tstp, NullSmoo_tstp);

   // creating initial guess which satisfies given initial condition for the starting time slab
   Vector mg_x0(spacetime_mg->Width());
   mg_x0 = 0.0;
   ProblemType * problem0 = fine_timestepping->GetProblem(0);
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
       std::cout << "\n\nSolving with sequential solve and checking the error \n";


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
       if (verbose)
       {
           std::cout << "checkres, tslab = " << tslab << "\n";
       }

       // FIXME: This is correct only in serial
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

   fine_timestepping->ComputeGlobalRhs(mg_rhs);
   fine_timestepping->GetProblem(0)->CorrectFromInitCnd(input_tslab0, mg_rhs_viewer.GetBlock(0));
   fine_timestepping->GetProblem(0)->ZeroBndValues(mg_rhs_viewer.GetBlock(0));

#if 0
   // second, to check, we solve with seq. solve on the finest level for the correction
   // and compute the error

   if (verbose)
       std::cout << "Solving for a correction with sequential solve and checking the final error \n";


   //Operator * FineOp_tstp = new TimeSteppingSolveOp<ProblemType>(*fine_timestepping, verbose);
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

   int local_size = mg_res.Size();
   int global_size = 0;
   MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, comm);

   double local_res0_norm_sq = mg_res.Norml2() * mg_res.Norml2();
   double global_res0_norm = 0.0;
   MPI_Allreduce(&local_res0_norm_sq, &global_res0_norm, 1, MPI_DOUBLE, MPI_SUM, comm);
   global_res0_norm = sqrt (global_res0_norm / global_size);

   if (verbose)
       std::cout << "res0 norm = " << global_res0_norm << "\n";

   Vector mg_finalsol(spacetime_mg->Width());
   BlockVector mg_finalsol_viewer(mg_finalsol.GetData(), fine_timestepping->GetGlobalOffsets());

   mg_finalsol = mg_x0;

   // solving for the correction, only one MG cycle
   Vector mg_sol(spacetime_mg->Width());
   mg_sol = 0.0;

   Vector mg_temp(spacetime_mg->Width());

   bool converged = false;

   int iter = 0;
   int mg_max_iter = 10;
   while (!converged && iter < mg_max_iter)
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


       double local_corr_norm_sq = mg_sol.Norml2() * mg_sol.Norml2();
       double global_corr_norm = 0;

       MPI_Allreduce(&local_corr_norm_sq, &global_corr_norm, 1, MPI_DOUBLE, MPI_SUM, comm);

       global_corr_norm = sqrt (global_corr_norm / global_size);

       if (verbose)
           std::cout << "Iteration " << iter << ": correction norm = " <<
                        global_corr_norm << "\n";

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
           if (verbose)
                std::cout << "mg_res after iterate, tslab = " << tslab << "\n";
           // FIXME: Works only in serial correctly, but it's only a debuggint print
           std::cout << "norm = " << mg_res_viewer.GetBlock(tslab).Norml2() /
                        sqrt (mg_res_viewer.GetBlock(tslab).Size()) << "\n";

           //for (int i = 0; i < mg_res_viewer.GetBlock(tslab).Size(); ++i)
               //if (fabs(mg_res_viewer.GetBlock(tslab)[i]) > 1.0e-10)
                   //std::cout << "entry " << i << ": res value = " << mg_res_viewer.GetBlock(tslab)[i] << "\n";

           //std::cout << "values of res at bottom interface: \n";
           //Vector& vec = fine_timestepping->GetProblem(tslab)->ExtractAtBase("bot", mg_res_viewer.GetBlock(tslab));
           //vec.Print();
       }

       double local_res_norm_sq = mg_res.Norml2() * mg_res.Norml2();
       double global_res_norm = 0.0;

       MPI_Allreduce(&local_res_norm_sq, &global_res_norm, 1, MPI_DOUBLE, MPI_SUM, comm);
       global_res_norm = sqrt (global_res_norm / global_size);

       // check convergence
       if (global_res_norm < eps * global_res0_norm)
           converged = true;

       // output convergence status
       if (verbose)
       {
           std::cout << "Iteration " << iter << ": res_norm = " << global_res_norm << "\n";
       }

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
   MPI_Finalize();
   return 0;

}

