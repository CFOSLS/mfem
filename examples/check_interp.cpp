//                       MFEM check for interpolation matrices in 4D

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>
#include <unistd.h>

#define WITH_HCURL

//#define WITH_HDIVSKEW

using namespace std;
using namespace mfem;

SparseMatrix * RemoveZeroEntries(const SparseMatrix& in);

void Compare_Offd_detailed(SparseMatrix& offd1, int * cmap1, SparseMatrix& offd2, int * cmap2);

// IDEA: Probably there is a bad orientation of boundary elements, there is not case dim = 4 in CheckBdrElementOrientation, no GetPentaOrientation
// Just try a mesh with two tets with sref = 4, pref = 1 in 3D, and see that there is a message about two boudnary element orientations being fixed
// In 4D there is no one to complain about it

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

   // sref = 3, pref = 1, mesh = two_penta crushes the check for Hdiv in 4D! (np = 2 > 1)
   // sref = 1, pref = 1, mesh = cube_96 crushes the check for Hdivskew and Hcurl in 4D! (np = 2 > 1)
   int ser_ref_levels  = 4;
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
       //mesh_file = "../data/cube_3d_moderate.mesh";
       mesh_file = "../data/two_tets.mesh";
   }
   else // 4D case
   {
       //mesh_file = "../data/cube4d_96.MFEM";
       mesh_file = "../data/two_pentatops.MFEM";
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
       delete mesh;
   }

   int dim = nDimensions;

   // For geometric multigrid
   int num_levels = par_ref_levels + 1;
   Array<ParMesh*> pmesh_lvls(num_levels);
   Array<ParFiniteElementSpace*> Hdiv_space_lvls(num_levels);
   Array<ParFiniteElementSpace*> L2_space_lvls(num_levels);
   Array<ParFiniteElementSpace*> Hdivskew_space_lvls(num_levels);
   Array<ParFiniteElementSpace*> Hcurl_space_lvls(num_levels);
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

#ifdef WITH_HDIVSKEW
   FiniteElementCollection *hdivskew_coll;
   if (dim == 4)
       hdivskew_coll = new DivSkew1_4DFECollection;
   ParFiniteElementSpace *Hdivskew_space;
   if (dim == 4)
       Hdivskew_space = new ParFiniteElementSpace(pmesh, hdivskew_coll);
#endif

#ifdef WITH_HCURL
   FiniteElementCollection *hcurl_coll;
   if (dim == 3)
       hcurl_coll = new ND_FECollection(feorder + 1, nDimensions);
   else // dim == 4
       hcurl_coll = new ND1_4DFECollection;

   ParFiniteElementSpace * Hcurl_space = new ParFiniteElementSpace(pmesh, hcurl_coll);
#endif

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

   Array<HypreParMatrix*> TrueP_Hdiv(par_ref_levels);
   Array<HypreParMatrix*> TrueP_L2(par_ref_levels);
   Array<HypreParMatrix*> TrueP_Hdivskew(par_ref_levels);
   Array<HypreParMatrix*> TrueP_Hcurl(par_ref_levels);
   Array<HypreParMatrix*> TrueP_H1(par_ref_levels);

   Array< SparseMatrix* > P_Hdiv_lvls(num_levels - 1);
   Array< SparseMatrix* > P_L2_lvls(num_levels - 1);
   Array< SparseMatrix* > P_Hdivskew_lvls(num_levels - 1);
   Array< SparseMatrix* > P_Hcurl_lvls(num_levels - 1);
   Array< SparseMatrix* > P_H1_lvls(num_levels - 1);

   const SparseMatrix* P_Hdiv_local;
   const SparseMatrix* P_L2_local;
   const SparseMatrix* P_Hdivskew_local;
   const SparseMatrix* P_Hcurl_local;
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
           pmesh->UniformRefinement();
           pmesh_lvls[l] = new ParMesh(*pmesh);
       }

       // creating pfespaces for level l
       Hdiv_space_lvls[l] = new ParFiniteElementSpace(pmesh_lvls[l], hdiv_coll);
       L2_space_lvls[l] = new ParFiniteElementSpace(pmesh_lvls[l], l2_coll);
#ifdef WITH_HDIVSKEW
       if (dim == 4)
        Hdivskew_space_lvls[l] = new ParFiniteElementSpace(pmesh_lvls[l], hdivskew_coll);
#endif
#ifdef WITH_HCURL
       Hcurl_space_lvls[l] = new ParFiniteElementSpace(pmesh_lvls[l], hcurl_coll);
#endif
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

#ifdef WITH_HDIVSKEW
           if (dim == 4)
           {
               Hdivskew_space->Update();
               P_Hdivskew_local = (SparseMatrix *)Hdivskew_space->GetUpdateOperator();
               P_Hdivskew_lvls[l] = RemoveZeroEntries(*P_Hdivskew_local);
               //P_Hdivskew_lvls[l] = (SparseMatrix *)Hdivskew_space->GetUpdateOperator();

               auto d_td_coarse_Hdivskew = Hdivskew_space_lvls[l + 1]->Dof_TrueDof_Matrix();
               SparseMatrix * RP_Hdivskew_local = Mult(*Hdivskew_space_lvls[l]->GetRestrictionMatrix(), *P_Hdivskew_lvls[l]);
               TrueP_Hdivskew[num_levels - 2 - l] = d_td_coarse_Hdivskew->LeftDiagMult(
                           *RP_Hdivskew_local, Hdivskew_space_lvls[l]->GetTrueDofOffsets());
               TrueP_Hdivskew[num_levels - 2 - l]->CopyColStarts();
               TrueP_Hdivskew[num_levels - 2 - l]->CopyRowStarts();

               delete RP_Hdivskew_local;
           }
#endif
#ifdef WITH_HCURL
           Hcurl_space->Update();
           P_Hcurl_local = (SparseMatrix *)Hcurl_space->GetUpdateOperator();
           P_Hcurl_lvls[l] = RemoveZeroEntries(*P_Hcurl_local);

           auto d_td_coarse_Hcurl = Hcurl_space_lvls[l + 1]->Dof_TrueDof_Matrix();
           SparseMatrix * RP_Hcurl_local = Mult(*Hcurl_space_lvls[l]->GetRestrictionMatrix(), *P_Hcurl_lvls[l]);
           TrueP_Hcurl[num_levels - 2 - l] = d_td_coarse_Hcurl->LeftDiagMult(
                       *RP_Hcurl_local, Hcurl_space_lvls[l]->GetTrueDofOffsets());
           TrueP_Hcurl[num_levels - 2 - l]->CopyColStarts();
           TrueP_Hcurl[num_levels - 2 - l]->CopyRowStarts();

           delete RP_Hcurl_local;
#endif
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

   // checking that P M_fine P = M_coarse for mass matrices for each of the spaces
   // H1
   {
       ParBilinearForm *mass_h1f = new ParBilinearForm(H1_space_lvls[0]);
       mass_h1f->AddDomainIntegrator(new MassIntegrator);
       mass_h1f->Assemble();
       mass_h1f->Finalize();
       HypreParMatrix * Mass_H1f = mass_h1f->ParallelAssemble();

       ParBilinearForm *mass_h1c = new ParBilinearForm(H1_space_lvls[1]);
       mass_h1c->AddDomainIntegrator(new MassIntegrator);
       mass_h1c->Assemble();
       mass_h1c->Finalize();
       HypreParMatrix * Mass_H1c = mass_h1c->ParallelAssemble();

       HypreParMatrix * PMP_H1 = RAP(TrueP_H1[num_levels - 1 - 1], Mass_H1f, TrueP_H1[num_levels - 1 -1]);

       // checking the difference
       SparseMatrix diag1;
       Mass_H1c->GetDiag(diag1);

       SparseMatrix diag2;
       PMP_H1->GetDiag(diag2);

       diag1.Add(-1.0, diag2);

       if (diag1.MaxNorm() > 1.0e-14)
           std::cout << "For H1 diagonal blocks are not equal, max norm = " << diag1.MaxNorm() << "! \n";

       SparseMatrix offd1;
       int * cmap1;
       Mass_H1c->GetOffd(offd1, cmap1);

       SparseMatrix offd2;
       int * cmap2;
       PMP_H1->GetOffd(offd2, cmap2);

       offd1.Add(-1.0, offd2);

       for (int i = 0; i < num_procs; ++i)
       {
           if (myid == i)
           {
               std::cout << "I am " << myid << "\n";
               if (offd1.MaxNorm() > 1.0e-14)
               {
                   std::cout << "For H1 off-diagonal blocks are not equal, max norm = " << offd1.MaxNorm() << "! \n";

                   Compare_Offd_detailed(offd1, cmap1, offd2, cmap2);
               }

               std::cout << "\n" << std::flush;
           }
           MPI_Barrier(comm);
       } // end fo loop over all processors, one after another
   }

   // Hdiv
   {
       ParBilinearForm *mass_hdivf = new ParBilinearForm(Hdiv_space_lvls[0]);
       mass_hdivf->AddDomainIntegrator(new VectorFEMassIntegrator);
       mass_hdivf->Assemble();
       mass_hdivf->Finalize();
       HypreParMatrix * Mass_Hdivf = mass_hdivf->ParallelAssemble();

       ParBilinearForm *mass_hdivc = new ParBilinearForm(Hdiv_space_lvls[1]);
       mass_hdivc->AddDomainIntegrator(new VectorFEMassIntegrator);
       mass_hdivc->Assemble();
       mass_hdivc->Finalize();
       HypreParMatrix * Mass_Hdivc = mass_hdivc->ParallelAssemble();

       HypreParMatrix * PMP_Hdiv = RAP(TrueP_Hdiv[num_levels - 1 - 1], Mass_Hdivf, TrueP_Hdiv[num_levels - 1 -1]);

       // checking the difference
       SparseMatrix diag1;
       Mass_Hdivc->GetDiag(diag1);

       SparseMatrix diag2;
       PMP_Hdiv->GetDiag(diag2);

       diag1.Add(-1.0, diag2);

       if (diag1.MaxNorm() > 1.0e-14)
           std::cout << "For Hdiv diagonal blocks are not equal, max norm = " << diag1.MaxNorm() << "! \n";

       SparseMatrix offd1;
       int * cmap1;
       Mass_Hdivc->GetOffd(offd1, cmap1);

       SparseMatrix offd2;
       int * cmap2;
       PMP_Hdiv->GetOffd(offd2, cmap2);

       offd1.Add(-1.0, offd2);

       for (int i = 0; i < num_procs; ++i)
       {
           if (myid == i)
           {
               std::cout << "I am " << myid << "\n";
               if (offd1.MaxNorm() > 1.0e-14)
               {
                   std::cout << "For Hdiv off-diagonal blocks are not equal, max norm = " << offd1.MaxNorm() << "! \n";

                   Compare_Offd_detailed(offd1, cmap1, offd2, cmap2);
               }

               std::cout << "\n" << std::flush;
           }
           MPI_Barrier(comm);
       } // end fo loop over all processors, one after another
   }

   // L2
   {
       ParBilinearForm *mass_l2f = new ParBilinearForm(L2_space_lvls[0]);
       mass_l2f->AddDomainIntegrator(new MassIntegrator);
       mass_l2f->Assemble();
       mass_l2f->Finalize();
       HypreParMatrix * Mass_L2f = mass_l2f->ParallelAssemble();

       ParBilinearForm *mass_l2c = new ParBilinearForm(L2_space_lvls[1]);
       mass_l2c->AddDomainIntegrator(new MassIntegrator);
       mass_l2c->Assemble();
       mass_l2c->Finalize();
       HypreParMatrix * Mass_L2c = mass_l2c->ParallelAssemble();

       HypreParMatrix * PMP_L2 = RAP(TrueP_L2[num_levels - 1 - 1], Mass_L2f, TrueP_L2[num_levels - 1 -1]);

       // checking the difference
       SparseMatrix diag1;
       Mass_L2c->GetDiag(diag1);

       SparseMatrix diag2;
       PMP_L2->GetDiag(diag2);

       diag1.Add(-1.0, diag2);

       if (diag1.MaxNorm() > 1.0e-14)
           std::cout << "For L2 diagonal blocks are not equal, max norm = " << diag1.MaxNorm() << "! \n";

       SparseMatrix offd1;
       int * cmap1;
       Mass_L2c->GetOffd(offd1, cmap1);

       SparseMatrix offd2;
       int * cmap2;
       PMP_L2->GetOffd(offd2, cmap2);

       offd1.Add(-1.0, offd2);

       for (int i = 0; i < num_procs; ++i)
       {
           if (myid == i)
           {
               std::cout << "I am " << myid << "\n";
               if (offd1.MaxNorm() > 1.0e-14)
               {
                   std::cout << "For L2 off-diagonal blocks are not equal, max norm = " << offd1.MaxNorm() << "! \n";

                   Compare_Offd_detailed(offd1, cmap1, offd2, cmap2);
               }

               std::cout << "\n" << std::flush;
           }
           MPI_Barrier(comm);
       } // end fo loop over all processors, one after another
   }

#ifdef WITH_HCURL
   // Hcurl
   {
       ParBilinearForm *mass_hcurlf = new ParBilinearForm(Hcurl_space_lvls[0]);
       mass_hcurlf->AddDomainIntegrator(new VectorFEMassIntegrator);
       mass_hcurlf->Assemble();
       mass_hcurlf->Finalize();
       HypreParMatrix * Mass_Hcurlf = mass_hcurlf->ParallelAssemble();

       ParBilinearForm *mass_hcurlc = new ParBilinearForm(Hcurl_space_lvls[1]);
       mass_hcurlc->AddDomainIntegrator(new VectorFEMassIntegrator);
       mass_hcurlc->Assemble();
       mass_hcurlc->Finalize();
       HypreParMatrix * Mass_Hcurlc = mass_hcurlc->ParallelAssemble();

       HypreParMatrix * PMP_Hcurl = RAP(TrueP_Hcurl[num_levels - 1 - 1], Mass_Hcurlf, TrueP_Hcurl[num_levels - 1 -1]);

       // checking the difference
       SparseMatrix diag1;
       Mass_Hcurlc->GetDiag(diag1);

       SparseMatrix diag2;
       PMP_Hcurl->GetDiag(diag2);

       diag1.Add(-1.0, diag2);

       if (diag1.MaxNorm() > 1.0e-14)
           std::cout << "For Hcurl diagonal blocks are not equal, max norm = " << diag1.MaxNorm() << "! \n";

       SparseMatrix offd1;
       int * cmap1;
       Mass_Hcurlc->GetOffd(offd1, cmap1);

       SparseMatrix offd2;
       int * cmap2;
       PMP_Hcurl->GetOffd(offd2, cmap2);

       offd1.Add(-1.0, offd2);

       for (int i = 0; i < num_procs; ++i)
       {
           if (myid == i)
           {
               std::cout << "I am " << myid << "\n";
               if (offd1.MaxNorm() > 1.0e-14)
               {
                   std::cout << "For Hcurl off-diagonal blocks are not equal, max norm = " << offd1.MaxNorm() << "! \n";

                   Compare_Offd_detailed(offd1, cmap1, offd2, cmap2);
               }

               std::cout << "\n" << std::flush;
           }
           MPI_Barrier(comm);
       } // end fo loop over all processors, one after another

   }
#endif

#ifdef WITH_HDIVSKEW
   // Hdivskew
   if (dim == 4)
   {
       ParBilinearForm *mass_hdivskewf = new ParBilinearForm(Hdivskew_space_lvls[0]);
       mass_hdivskewf->AddDomainIntegrator(new VectorFE_DivSkewMassIntegrator);
       mass_hdivskewf->Assemble();
       mass_hdivskewf->Finalize();
       HypreParMatrix * Mass_Hdivskewf = mass_hdivskewf->ParallelAssemble();

       ParBilinearForm *mass_hdivskewc = new ParBilinearForm(Hdivskew_space_lvls[1]);
       mass_hdivskewc->AddDomainIntegrator(new VectorFE_DivSkewMassIntegrator);
       mass_hdivskewc->Assemble();
       mass_hdivskewc->Finalize();
       HypreParMatrix * Mass_Hdivskewc = mass_hdivskewc->ParallelAssemble();

       HypreParMatrix * PMP_Hdivskew = RAP(TrueP_Hdivskew[num_levels - 1 - 1], Mass_Hdivskewf, TrueP_Hdivskew[num_levels - 1 -1]);

       // checking the difference
       SparseMatrix diag1;
       Mass_Hdivskewc->GetDiag(diag1);

       SparseMatrix diag2;
       PMP_Hdivskew->GetDiag(diag2);

       diag1.Add(-1.0, diag2);

       for (int i = 0; i < num_procs; ++i)
       {
           if (myid == i)
           {
               std::cout << "I am " << myid << "\n";
               if (diag1.MaxNorm() > 1.0e-14)
                   std::cout << "For Hdivskew diagonal blocks are not equal, max norm = " << diag1.MaxNorm() << "! \n";
               std::cout << "\n" << std::flush;
           }
           MPI_Barrier(comm);
       }


       SparseMatrix offd1;
       int * cmap1;
       Mass_Hdivskewc->GetOffd(offd1, cmap1);

       SparseMatrix offd2;
       int * cmap2;
       PMP_Hdivskew->GetOffd(offd2, cmap2);

       offd1.Add(-1.0, offd2);

       for (int i = 0; i < num_procs; ++i)
       {
           if (myid == i)
           {
               std::cout << "I am " << myid << "\n";
               if (offd1.MaxNorm() > 1.0e-14)
               {
                   std::cout << "For Hdivskew off-diagonal blocks are not equal, max norm = " << offd1.MaxNorm() << "! \n";

                   Compare_Offd_detailed(offd1, cmap1, offd2, cmap2);
               }

               // checking on a specific vector
               //Vector testort(offd1.Width());

               std::cout << "\n" << std::flush;
           }
           MPI_Barrier(comm);
       } // end fo loop over all processors, one after another

   }
#endif

   MPI_Barrier(comm);
   if (verbose)
       std::cout << "All checks were done, all failures should"
                    " have appeared above \n" << std::flush;
   MPI_Barrier(comm);

   MPI_Finalize();
   return 0;
}

SparseMatrix * RemoveZeroEntries(const SparseMatrix& in)
{
    int * I = in.GetI();
    int * J = in.GetJ();
    double * Data = in.GetData();
    double * End = Data+in.NumNonZeroElems();

    int nnz = 0;
    for (double * data_ptr = Data; data_ptr != End; data_ptr++)
    {
        if (*data_ptr != 0)
            nnz++;
    }

    int * outI = new int[in.Height()+1];
    int * outJ = new int[nnz];
    double * outData = new double[nnz];
    nnz = 0;
    for (int i = 0; i < in.Height(); i++)
    {
        outI[i] = nnz;
        for (int j = I[i]; j < I[i+1]; j++)
        {
            if (Data[j] !=0)
            {
                outJ[nnz] = J[j];
                outData[nnz++] = Data[j];
            }
        }
    }
    outI[in.Height()] = nnz;

    return new SparseMatrix(outI, outJ, outData, in.Height(), in.Width());
}

void Compare_Offd_detailed(SparseMatrix& offd1, int * cmap1, SparseMatrix& offd2, int * cmap2)
{
    /*
     * it is not a good idea to compare cmaps
    for ( int i = 0; i < offd1.Width(); ++i)
    {
        if (cmap1[i] != cmap2[i])
             std::cout << "cmap1 != cmap2 at " << i << ": cmap1 = " << cmap1[i] << ", cmap2 = " << cmap2[i] << "\n";
    }
    */

    for ( int row = 0; row < offd1.Height(); ++row)
    {
        std::cout << "row = " << row << "\n";

        int nnz_rowshift1 = offd1.GetI()[row];
        std::map<int,double> row_entries1;
        for (int colind = 0; colind < offd1.RowSize(row); ++colind)
        {
            int col1 = offd1.GetJ()[nnz_rowshift1 + colind];
            int truecol1 = cmap1[col1];
            double val1 = offd1.GetData()[nnz_rowshift1 + colind];
            //if (fabs(val1) > 1.0e-15)
                 row_entries1.insert(std::make_pair(truecol1, val1));
        }

        int nnz_rowshift2 = offd2.GetI()[row];
        std::map<int,double> row_entries2;
        for (int colind = 0; colind < offd2.RowSize(row); ++colind)
        {
            int col2 = offd2.GetJ()[nnz_rowshift2 + colind];
            int truecol2 = cmap2[col2];
            double val2 = offd2.GetData()[nnz_rowshift2 + colind];
            //if (fabs(val2) > 1.0e-15)
                 row_entries2.insert(std::make_pair(truecol2, val2));
        }

        if (row_entries1.size() != row_entries2.size())
            std::cout << "row_entries1.size() = " << row_entries1.size() << " != " << row_entries2.size() << " = row_entries2.size() \n";

        std::map<int, double>::iterator it;
        std::map<int, double>::iterator it2;
        for ( it = row_entries1.begin(); it != row_entries1.end(); it++ )
        {
            int truecol1 = it->first;
            double value1 = it->second;
            it2 = row_entries2.find(truecol1);
            if (it2 != row_entries2.end())
            {
                double value2 = it2->second;
                if ( fabs(value2 - value1) / fabs(value1) > 1.0e-14 &&  (fabs(value1) > 1.0e-15 || fabs(value2) > 1.0e-15 ) )
                    std::cout << "For truecol = " << truecol1 << " values are different: " << value1 << " != " << value2 << "\n";
                row_entries2.erase(it2);
            }
            else
            {
                std::cout << "Cannot find pair (" << truecol1 << ", " << value1 << ") from row_entries1 in row_eintries2 \n";
            }
        }

        if (row_entries1.size() != row_entries2.size())
        {
            for ( it2 = row_entries2.begin(); it2 != row_entries2.end(); it2++ )
            {
                int truecol2 = it2->first;
                double value2 = it2->second;
                std::cout << "additional item in row_entry2: (" << truecol2 << ", " << value2 << ") ";
            }
            std::cout << "\n";
        }

        //int nnz_rowshift = offd1.GetI()[row];
        if (offd1.GetI()[row] != offd2.GetI()[row])
            std::cout << "offd1.GetI()[row] = " << offd1.GetI()[row] << " != " << offd2.GetI()[row] << " = offd2.GetI()[row], row = " << row << "\n";
        if (offd1.RowSize(row) != offd2.RowSize(row))
            std::cout << "offd1.RowSize(row) = " << offd1.RowSize(row) << " != " << offd2.RowSize(row) << " = offd2.RowSize(row), row = " << row << "\n";

        for (int colind = 0; colind < offd1.RowSize(row); ++colind)
        {
            int col1 = offd1.GetJ()[nnz_rowshift1 + colind];
            int truecol1 = cmap1[col1];
            double val1 = offd1.GetData()[nnz_rowshift1 + colind];

            int col2 = offd2.GetJ()[nnz_rowshift2 + colind];
            int truecol2 = cmap2[col2];
            double val2 = offd2.GetData()[nnz_rowshift2 + colind];

            //if (col1 != col2)
                //std::cout << "colind = " << colind << ": " << "col1 = " << col1 << "!= " << col2 << " = col2 \n";
            if (truecol1 != truecol2)
            {
                std::cout << "colind = " << colind << ": " << "truecol1 = " << truecol1 << "!= " << truecol2 << " = truecol2 \n";
                std::cout << "colind = " << colind << ": " << "value1 = " << val1 << "!= " << val2 << " = value2 \n";
            }
            else
            {
                if ( fabs(val1 - val2) / fabs(val1) > 1.0e-14)
                     std::cout << "colind = " << colind << ": " << "value1 = " << val1 << "!= " << val2 << " = value2 \n";
            }
        }

        //int nnz_rowshift2 = offd2.GetI()[row];
        if (offd1.RowSize(row) != offd2.RowSize(row))
        {
            std::cout << "Additional columns in offd2 \n";
            for (int colind = offd1.RowSize(row); colind < offd2.RowSize(row); ++colind)
            {
                int col2 = offd2.GetJ()[nnz_rowshift2 + colind];
                int truecol2 = cmap2[col2];
                double val2 = offd2.GetData()[nnz_rowshift2 + colind];

                std::cout << "colind = " << colind << ": " << truecol2 << " = truecol2 \n";
                std::cout << "colind = " << colind << ": " << val2 << " = value2 \n";
            }

        }


        std::cout << "\n";
    }

    //std::cout << "offd1 \n";
    //offd1.Print();
    //std::cout << "offd2 \n";
    //offd2.Print();

}
