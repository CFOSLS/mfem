///                       MFEM Example 1 - Parallel Version (modified for using geometric MG)
///
/// See description of Example 1 for MFEM at mfem.org.
///
/// This is a modified version in which the Laplace problem is considered.
/// The problem is sovled with a geomtric multigrid preconditioner.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>
#include <unistd.h>

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

   int ser_ref_levels  = 1;
   int par_ref_levels  = 1;

   // 2. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool visualization = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-sref", "--sref",
                  "Number of serial refinements 4d mesh.");
   args.AddOption(&par_ref_levels, "-pref", "--pref",
                  "Number of parallel refinements 4d mesh.");
   args.AddOption(&nDimensions, "-dim", "--whichD",
                  "Dimension of the space-time problem.");
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

   if (nDimensions == 3)
   {
       mesh_file = "../data/cube_3d_moderate.mesh";
   }
   else // 4D case
   {
       mesh_file = "../data/cube4d_96.MFEM";
   }

   // 3. Reading the mesh and performing a prescribed number of serial and parallel refinements
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
       delete mesh;
   }

   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;

   int dim = nDimensions;

   // 4. Define the setup for the geometric multigrid
   // Now it's not the optimal way, since the hierarchy of nested meshed
   // and corresponding f.e. structures are created explicitly.
   // A simpler way is to use GeneralHierarchy from mfem/cfosls/.
   // For example, look at cfosls_hyperbolic_multigrid.cpp.
   int num_levels = par_ref_levels + 1;
   Array<ParMesh*> pmesh_lvls(num_levels);
   Array<ParFiniteElementSpace*> H_space_lvls(num_levels);

   FiniteElementCollection *h1_coll;
   ParFiniteElementSpace *H_space;
   if (dim == 3)
       h1_coll = new H1_FECollection(order, nDimensions);
   else
   {
       if (order == 1)
           h1_coll = new LinearFECollection;
       else if (order == 2)
       {
           if (verbose)
               std::cout << "We have Quadratic FE for H1 in 4D, but are you sure? \n";
           h1_coll = new QuadraticFECollection;
       }
       else
           MFEM_ABORT("Higher-order H1 elements are not implemented in 4D \n");
   }
   H_space = new ParFiniteElementSpace(pmesh, h1_coll);
   Array<HypreParMatrix*> TrueP_H(par_ref_levels);
   std::vector<Array<int>* > EssBdrTrueDofs_H1(num_levels);
   Array< SparseMatrix* > P_H_lvls(num_levels - 1);
   const SparseMatrix* P_H_local;

   // 4.1 Creating hierarchy of everything needed for the geometric multigrid preconditioner
   for (int l = num_levels - 1; l >= 0; --l)
   {
       // creating pmesh for level l
       if (l == num_levels - 1)
       {
           pmesh_lvls[l] = new ParMesh(*pmesh);
       }
       else
       {
           pmesh->UniformRefinement();
           pmesh_lvls[l] = new ParMesh(*pmesh);
       }

       // creating pfespaces for level l
       H_space_lvls[l] = new ParFiniteElementSpace(pmesh_lvls[l], h1_coll);

       EssBdrTrueDofs_H1[l] = new Array<int>();
       H_space_lvls[l]->GetEssentialTrueDofs(ess_bdr, *EssBdrTrueDofs_H1[l]);

       // for all but one levels we create projection matrices between levels
       // and projectors assembled on true dofs if MG preconditioner is used
       if (l < num_levels - 1)
       {
           H_space->Update();
           P_H_local = (SparseMatrix *)H_space->GetUpdateOperator();
           P_H_lvls[l] = RemoveZeroEntries(*P_H_local);

           auto d_td_coarse_H = H_space_lvls[l + 1]->Dof_TrueDof_Matrix();
           SparseMatrix * RP_H_local = Mult(*H_space_lvls[l]->GetRestrictionMatrix(), *P_H_lvls[l]);
           TrueP_H[num_levels - 2 - l] = d_td_coarse_H->LeftDiagMult(
                       *RP_H_local, H_space_lvls[l]->GetTrueDofOffsets());
           TrueP_H[num_levels - 2 - l]->CopyColStarts();
           TrueP_H[num_levels - 2 - l]->CopyRowStarts();

           delete RP_H_local;
       }

   } // end of loop over all levels

   pmesh_lvls[0]->PrintInfo(std::cout); if(verbose) cout << endl;

   // 5. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (pmesh_lvls[0]->bdr_attributes.Size())
   {
      H_space_lvls[0]->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 6. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1,phi_i) where phi_i are the basis functions in fespace.
   ParLinearForm *b = new ParLinearForm(H_space_lvls[0]);
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   // 7. Define the solution vector x as a parallel finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   ParGridFunction x(H_space_lvls[0]);
   x = 0.0;

   // 8. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //     domain integrator.
   ParBilinearForm *a = new ParBilinearForm(H_space_lvls[0]);
   a->AddDomainIntegrator(new DiffusionIntegrator(one));

   // 9. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   a->Assemble();

   HypreParMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   if (myid == 0)
   {
      cout << "Size of linear system: " << A.GetGlobalNumRows() << "\n";
   }

   // 10. Define and apply a parallel CG solver for AX=B with the geometric multigrid preconditioner.

#ifdef BND_FOR_MULTIGRID
   Multigrid * prec = new Multigrid(A, TrueP_H, EssBdrTrueDofs_H1);
#else
   Multigrid * prec = new Multigrid(A, TrueP_H);
#endif
   CGSolver solver(comm);
   if (verbose)
       cout << "Linear solver: CG \n";

   solver.SetAbsTol(1e-12);
   solver.SetRelTol(1e-12);
   solver.SetMaxIter(10000);
   solver.SetOperator(A);
   solver.SetPreconditioner(*prec);

   solver.SetPrintLevel(1);

   StopWatch chrono;

   MPI_Barrier(comm);
   chrono.Clear();
   chrono.Start();
   solver.Mult(B, X);
   MPI_Barrier(comm);
   chrono.Stop();

   if (verbose)
   {
       if (solver.GetConverged())
           std::cout << "Linear solver converged in " << solver.GetNumIterations()
                     << " iterations with a residual norm of " << solver.GetFinalNorm() << ".\n";
       else
           std::cout << "Linear solver did not converge in " << solver.GetNumIterations()
                     << " iterations. Residual norm is " << solver.GetFinalNorm() << ".\n";
       std::cout << "Linear solver took " << chrono.RealTime() << "s. \n";
   }


   // 11. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a->RecoverFEMSolution(X, *b, x);

   // 12. Free the used memory.
   delete prec;
   for (int i = 0; i < H_space_lvls.Size(); ++i)
       delete H_space_lvls[i];

   for (int i = 0; i < TrueP_H.Size(); ++i)
       delete TrueP_H[i];

   for (unsigned int i = 0; i < EssBdrTrueDofs_H1.size(); ++i)
       delete EssBdrTrueDofs_H1[i];

   for (int i = 0; i < P_H_lvls.Size(); ++i)
       delete P_H_lvls[i];

   for (int i = 0; i < pmesh_lvls.Size(); ++i)
       delete pmesh_lvls[i];

   delete H_space;
   delete h1_coll;

   delete a;
   delete b;
   delete pmesh;

   MPI_Finalize();
   return 0;
}

