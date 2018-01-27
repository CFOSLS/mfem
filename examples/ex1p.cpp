//                       MFEM Example 1 - Parallel Version modified

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm comm = MPI_COMM_WORLD;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   const char *mesh_file = "../data/cube_3d_moderate.mesh";
   int order = 1;
   bool static_cond = false;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
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

   // 3. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   //mesh->UniformRefinement(); // this changes the failure to success

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   pmesh->UniformRefinement();
   //pmesh->RefineGroups();

   pmesh->PrintInfo(std::cout);
   if (myid == 0)
       std::cout << "\n";

   Array<int> ess_bdrSigma(pmesh->bdr_attributes.Max());
   ess_bdrSigma = 0;
   ess_bdrSigma[0] = 1;

   int feorder = 0;
   //int dim = 3;
   FiniteElementCollection *hdiv_coll_test = new RT_FECollection(feorder, dim);
   FiniteElementCollection *l2_coll_test = new L2_FECollection(feorder, dim);
   FiniteElementCollection *hdivfree_coll_test = new ND_FECollection(feorder + 1, dim);

   ParFiniteElementSpace *R_space_test = new ParFiniteElementSpace(pmesh, hdiv_coll_test);
   ParFiniteElementSpace *W_space_test = new ParFiniteElementSpace(pmesh, l2_coll_test);
   ParFiniteElementSpace *C_space_test = new ParFiniteElementSpace(pmesh, hdivfree_coll_test);

   if (myid == 0)
    ess_bdrSigma.Print();

   ParDiscreteLinearOperator Divfree_op(C_space_test, R_space_test); // from Hcurl or HDivSkew(C_space) to Hdiv(R_space)
   Divfree_op.AddDomainInterpolator(new CurlInterpolator);
   Divfree_op.Assemble();
   Vector tempsol(Divfree_op.Width());
   tempsol = 0.0;
   Vector temprhs(Divfree_op.Height());
   temprhs = 0.0;
   Divfree_op.EliminateTrialDofs(ess_bdrSigma, tempsol, temprhs);
   //Divfree_op.EliminateTestDofs(ess_bdrSigma);
   Divfree_op.Finalize();
   HypreParMatrix * Divfree_hpmat = Divfree_op.ParallelAssemble();

   ParMixedBilinearForm *Bblock = new ParMixedBilinearForm(R_space_test, W_space_test);
   Bblock->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   Bblock->Assemble();
   //Vector tempsol(Bblock->Width());
   //tempsol = 0.0;
   //Vector temprhs(Bblock->Height());
   //temprhs = 0.0;
   //Bblock->EliminateTrialDofs(ess_bdrSigma, tempsol, temprhs);
   //Bblock->EliminateTestDofs(ess_bdrSigma);
   Bblock->Finalize();
   HypreParMatrix * Constraint_global = Bblock->ParallelAssemble();

   delete Bblock;

   HypreParMatrix * checkprod = ParMult(Constraint_global, Divfree_hpmat);

   SparseMatrix diagg;
   checkprod->GetDiag(diagg);

   SparseMatrix offdiagg;
   HYPRE_Int * cmap_offd;
   checkprod->GetOffd(offdiagg, cmap_offd);

   for (int i = 0 ;i < num_procs; ++i)
   {
       if (myid == i)
       {
           std::cout << "I am " << myid << "\n" << std::flush;
           std::cout << "Constraint[0] * Curl[0] diag norm = " << diagg.MaxNorm() << "\n";
           std::cout << "Constraint[0] * Curl[0] offdiag norm = " << offdiagg.MaxNorm() << "\n";
       }
       MPI_Barrier(comm);
   }

   delete checkprod;

   MPI_Finalize();
   return 0;
}
