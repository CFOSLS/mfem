#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>
#include <unistd.h>

#define ZEROTOL (5.0e-14)

#define NONUNIFORM_REFINE

using namespace std;
using namespace mfem;

double u_test(const Vector& xt)
{
    double x = xt(0);
    double y;
    if (xt.Size() >= 3)
        y = xt(1);
    double z;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size()-1);

    double res = t*t*exp(t) * sin (3.0 * M_PI * x);
    if (xt.Size() >= 3)
        res *= sin (2.0 * M_PI * y);
    if (xt.Size() == 4)
        res *= sin (M_PI * z);

    return res;
}

void bFun_test(const Vector& xt, Vector& b )
{
    double x = xt(0);
    double y = xt(1);

    b.SetSize(xt.Size());

    b(0) = sin(x*M_PI)*cos(y*M_PI);
    b(1) = - sin(y*M_PI)*cos(x*M_PI);

    b(xt.Size()-1) = 1.;

}


void sigma_test(const Vector& xt, Vector& sigma)
{
    Vector b;
    bFun_test(xt, b);
    sigma.SetSize(xt.Size());
    sigma(xt.Size()-1) = u_test(xt);
    for (int i = 0; i < xt.Size()-1; i++)
        sigma(i) = b(i) * sigma(xt.Size()-1);

    return;
}


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

   int ser_ref_levels  = 1;
   int par_ref_levels  = 1;

   // 2. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";

   int feorder = 0;
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
       //mesh_file = "../data/two_tets.mesh";
       //mesh_file = "../data/pmesh_cylinder_moderate_0.2.mesh";
   }
   else // 4D case
   {
       mesh_file = "../data/pmesh_tsl_1proc.mesh";
       //mesh_file = "../data/cube4d_96.MFEM";
       //mesh_file = "../data/two_pentatops.MFEM";
       //mesh_file = "../data/two_pentatops_2.MFEM";
   }

   if (verbose)
       std::cout << "Number of mpi processes: " << num_procs << "\n" << std::flush;

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

   int dim = nDimensions;

   int num_levels = 2; //par_ref_levels + 1;
   Array<ParMesh*> pmesh_lvls(num_levels);
   Array<ParFiniteElementSpace*> Hdiv_space_lvls(num_levels);
   Array<ParFiniteElementSpace*> L2_space_lvls(num_levels);
   Array<ParFiniteElementSpace*> H1_space_lvls(num_levels);

   FiniteElementCollection *hdiv_coll;
   ParFiniteElementSpace *Hdiv_space;
   FiniteElementCollection *l2_coll;
   ParFiniteElementSpace *L2_space;

   if (dim == 4)
       hdiv_coll = new RT0_4DFECollection;
   else
       hdiv_coll = new RT_FECollection(feorder, dim);

   Hdiv_space = new ParFiniteElementSpace(pmesh, hdiv_coll);

   l2_coll = new L2_FECollection(feorder, nDimensions);
   L2_space = new ParFiniteElementSpace(pmesh, l2_coll);

   FiniteElementCollection *h1_coll;
   ParFiniteElementSpace *H1_space;
   if (dim == 3)
       h1_coll = new H1_FECollection(feorder + 1, nDimensions);
   else
   {
       if (feorder + 1 == 1)
           h1_coll = new LinearFECollection;
       else if (feorder + 1 == 2)
       {
           if (verbose)
               std::cout << "We have Quadratic FE for H1 in 4D, but are you sure? \n";
           h1_coll = new QuadraticFECollection;
       }
       else
           MFEM_ABORT("Higher-order H1 elements are not implemented in 4D \n");
   }
   H1_space = new ParFiniteElementSpace(pmesh, h1_coll);

   Array<HypreParMatrix*> TrueP_Hdiv(num_levels - 1);
   Array<HypreParMatrix*> TrueP_L2(num_levels - 1);
   Array<HypreParMatrix*> TrueP_H1(num_levels - 1);

   Array< SparseMatrix* > P_Hdiv_lvls(num_levels - 1);
   Array< SparseMatrix* > P_L2_lvls(num_levels - 1);
   Array< SparseMatrix* > P_H1_lvls(num_levels - 1);

   const SparseMatrix* P_Hdiv_local;
   const SparseMatrix* P_L2_local;
   const SparseMatrix* P_H1_local;

   // Creating hierarchy of everything needed for the geometric multigrid preconditioner
   for (int l = num_levels - 1; l >= 0; --l)
   {
       // creating pmesh for level l
       if (l == num_levels - 1)
       {
           pmesh_lvls[l] = new ParMesh(*pmesh);
       }
       else
       {
#ifdef NONUNIFORM_REFINE
           int nmarked = 5;
           Array<int> els_to_refine(nmarked);
           for (int i = 0; i < nmarked; ++i)
               els_to_refine[i] = pmesh->GetNE()/2 + i;
           pmesh->GeneralRefinement(els_to_refine);
#else
           pmesh->UniformRefinement();
#endif
           pmesh_lvls[l] = new ParMesh(*pmesh);
       }
       pmesh_lvls[l]->PrintInfo(std::cout); if(verbose) cout << endl;

       // creating pfespaces for level l
       Hdiv_space_lvls[l] = new ParFiniteElementSpace(pmesh_lvls[l], hdiv_coll);
       L2_space_lvls[l] = new ParFiniteElementSpace(pmesh_lvls[l], l2_coll);
       H1_space_lvls[l] = new ParFiniteElementSpace(pmesh_lvls[l], h1_coll);

       // for all but one levels we create projection matrices between levels
       // and projectors assembled on true dofs if MG preconditioner is used
       if (l < num_levels - 1)
       {
           Hdiv_space->Update();
           P_Hdiv_local = (SparseMatrix *)Hdiv_space->GetUpdateOperator();
           P_Hdiv_lvls[l] = RemoveZeroEntries(*P_Hdiv_local);

           auto d_td_coarse_Hdiv = Hdiv_space_lvls[l + 1]->Dof_TrueDof_Matrix();
           SparseMatrix * RP_Hdiv_local = Mult(*Hdiv_space_lvls[l]->GetRestrictionMatrix(), *P_Hdiv_lvls[l]);
           TrueP_Hdiv[num_levels - 2 - l] = d_td_coarse_Hdiv->LeftDiagMult(
                       *RP_Hdiv_local, Hdiv_space_lvls[l]->GetTrueDofOffsets());
           TrueP_Hdiv[num_levels - 2 - l]->CopyColStarts();
           TrueP_Hdiv[num_levels - 2 - l]->CopyRowStarts();

           delete RP_Hdiv_local;

           L2_space->Update();
           P_L2_local = (SparseMatrix *)L2_space->GetUpdateOperator();
           P_L2_lvls[l] = RemoveZeroEntries(*P_L2_local);

           auto d_td_coarse_L2 = L2_space_lvls[l + 1]->Dof_TrueDof_Matrix();
           SparseMatrix * RP_L2_local = Mult(*L2_space_lvls[l]->GetRestrictionMatrix(), *P_L2_lvls[l]);
           TrueP_L2[num_levels - 2 - l] = d_td_coarse_L2->LeftDiagMult(
                       *RP_L2_local, L2_space_lvls[l]->GetTrueDofOffsets());
           TrueP_L2[num_levels - 2 - l]->CopyColStarts();
           TrueP_L2[num_levels - 2 - l]->CopyRowStarts();

           delete RP_L2_local;

           H1_space->Update();
           P_H1_local = (SparseMatrix *)H1_space->GetUpdateOperator();
           P_H1_lvls[l] = RemoveZeroEntries(*P_H1_local);

           auto d_td_coarse_H = H1_space_lvls[l + 1]->Dof_TrueDof_Matrix();
           SparseMatrix * RP_H1_local = Mult(*H1_space_lvls[l]->GetRestrictionMatrix(), *P_H1_lvls[l]);
           TrueP_H1[num_levels - 2 - l] = d_td_coarse_H->LeftDiagMult(
                       *RP_H1_local, H1_space_lvls[l]->GetTrueDofOffsets());
           TrueP_H1[num_levels - 2 - l]->CopyColStarts();
           TrueP_H1[num_levels - 2 - l]->CopyRowStarts();

           delete RP_H1_local;
       }

   } // end of loop over all levels

   pmesh_lvls[0]->PrintInfo(std::cout); if(verbose) cout << endl;

   // the check: B * P_div sigma_coarse is constant within coarse elements

   /*
    * this is incorrect
   ParMixedBilinearForm div(Hdiv_space_lvls[0], L2_space_lvls[0]);
   div.AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   div.Assemble();
   div.Finalize();
   */

   // this is correct (check is passed)
   ParDiscreteLinearOperator div(Hdiv_space_lvls[0], L2_space_lvls[0]);
   div.AddDomainInterpolator(new DivergenceInterpolator);
   div.Assemble();
   div.Finalize();
   HypreParMatrix * Div = div.ParallelAssemble();

   ParGridFunction * sigma_coarse_pgfun = new ParGridFunction(Hdiv_space_lvls[1]);
   //VectorFunctionCoefficient testfun_coeff(dim, testHdivfun);
   VectorFunctionCoefficient testfun_coeff(dim, sigma_test);
   sigma_coarse_pgfun->ProjectCoefficient(testfun_coeff);
   Vector sigma_c(Hdiv_space_lvls[1]->TrueVSize());
   sigma_coarse_pgfun->ParallelProject(sigma_c);

   Vector Psigma_c(Div->Width());
   TrueP_Hdiv[0]->Mult(sigma_c, Psigma_c);
   Vector BPsigma_c(Div->Height());
   Div->Mult(Psigma_c, BPsigma_c);

   SparseMatrix * AE_e = Transpose(*P_L2_lvls[0]);

   //std::cout << "Look at AE_e \n";
   //AE_e->Print();

   for (int AE = 0; AE < pmesh_lvls[1]->GetNE(); ++AE)
   {
       double row_av = 0.0;
       double rowsum = 0.0;
       int row_length = AE_e->RowSize(AE);
       int * elinds = AE_e->GetRowColumns(AE);
       for (int j = 0; j < row_length; ++j)
           rowsum += BPsigma_c[elinds[j]];
       row_av = rowsum / row_length;

       if (rowsum > 1.0e-14)
       {
           bool bad_row = false;
           for (int j = 0; j < row_length; ++j)
               if (fabs(BPsigma_c[elinds[j]] - row_av) > 1.0e-13)
               {
                   bad_row = true;
                   break;
               }

           if (bad_row)
           {
               std::cout << "AE " << AE << ": \n";
               std::cout << "vertices: \n";
               const Element * AEl = pmesh_lvls[1]->GetElement(AE);
               int nverts = AEl->GetNVertices();
               Array<int> AElverts;
               AEl->GetVertices(AElverts);
               for (int vno = 0; vno < nverts; ++vno)
               {
                   double * vert_coos = pmesh_lvls[1]->GetVertex(AElverts[vno]);
                   std::cout << "(";
                   for (int coo = 0; coo < dim; ++coo)
                       if (coo < dim - 1)
                           std::cout << vert_coos[coo] << ", ";
                       else
                           std::cout << vert_coos[coo] << ") ";
               }
               std::cout << "\n";

               std::cout << "fine subelements: ";
               for (int j = 0; j < row_length; ++j)
                   std::cout << elinds[j] << " ";
               std::cout << "\n";
               for (int j = 0; j < row_length; ++j)
               {
                   std::cout << "fine subelement " << elinds[j] << "\n";
                   std::cout << "vertices: \n";
                   const Element * el = pmesh_lvls[0]->GetElement(elinds[j]);
                   int nverts = el->GetNVertices();
                   Array<int> elverts;
                   el->GetVertices(elverts);
                   for (int vno = 0; vno < nverts; ++vno)
                   {
                       double * vert_coos = pmesh_lvls[0]->GetVertex(elverts[vno]);
                       std::cout << "(";
                       for (int coo = 0; coo < dim; ++coo)
                           if (coo < dim - 1)
                               std::cout << vert_coos[coo] << ", ";
                           else
                               std::cout << vert_coos[coo] << ") ";
                   }
                   std::cout << "\n";
               }
               //std::cout << "rowsum = " << rowsum << "\n";
               std::cout << "row_av = " << row_av << "\n";
               for (int j = 0; j < row_length; ++j)
               {
                   std::cout << BPsigma_c[elinds[j]] << " ";
               }
               std::cout << "\n \n";

           }
       }

   }

   MPI_Finalize();
   return 0;
}

