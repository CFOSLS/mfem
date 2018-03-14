//                       MFEM check for interpolation matrices in 3D and 4D

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>
#include <unistd.h>

#define WITH_HCURL

//#define WITH_HDIVSKEW

#define VERBOSE_OFFD

#define ZEROTOL (5.0e-14)

#define USE_TSL

using namespace std;
using namespace mfem;

void testVectorFun(const Vector& xt, Vector& res);

SparseMatrix * RemoveZeroEntries(const SparseMatrix& in);

void Compare_Offd_detailed(SparseMatrix& offd1, int * cmap1, SparseMatrix& offd2, int * cmap2);

std::set<std::pair<int,int> >* CreateBotToTopDofsLink(const char * eltype, FiniteElementSpace& fespace,
                                                         std::vector<std::pair<int,int> > & bot_to_top_bels, bool verbose = false);

double testH1fun(Vector& xt);
void testHdivfun(const Vector& xt, Vector& res);

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
   int par_ref_levels  = 0;

   // 2. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
#ifdef USE_TSL
   const char *meshbase_file = "../data/star.mesh";
   int Nt = 2;
   double tau = 0.5;
#endif

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

   for (int l = 0; l < ser_ref_levels; l++)
       meshbase->UniformRefinement();

   ParMesh * pmeshbase = new ParMesh(comm, *meshbase);
   for (int l = 0; l < par_ref_levels; l++)
       pmeshbase->UniformRefinement();

   //if (verbose)
       //std::cout << "pmeshbase shared structure \n";
   //pmeshbase->PrintSharedStructParMesh();

   delete meshbase;

   ParMeshTSL * pmesh = new ParMeshTSL(comm, *pmeshbase, tau, Nt);

   //MPI_Finalize();
   //return 0;

   /*
   if (num_procs == 1)
   {
       std::stringstream fname;
       fname << "pmesh_tsl_1proc.mesh";
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

#endif

   int dim = nDimensions;

   // For geometric multigrid
   int num_levels = 2; //par_ref_levels + 1;
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

   /*
   int ngroups = pmesh->GetNGroups();

   std::cout << std::flush;
   MPI_Barrier(comm);

   for (int i = 0; i < num_procs; ++i)
   {
       if (myid == i)
       {
           std::cout << "I am " << myid << "\n";

           std::set<int> shared_facetdofs;
           std::set<int> shared_facedofs;

           for (int grind = 1; grind < ngroups; ++grind)
           {
               std::cout << "group = " << grind << "\n";
               int ngroupfaces = pmesh->GroupNFaces(grind);
               std::cout << "ngroupfaces = " << ngroupfaces << "\n";
               Array<int> dofs;
               //if (ngroupfaces > 10)
                    //ngroupfaces = 10;
               for (int faceind = 0; faceind < ngroupfaces; ++faceind)
               {
                   std::cout << "shared face, faceind = " << faceind << " \n";

                   Hdiv_space->GetSharedFaceDofs(grind, faceind, dofs);
                   for (int dofind = 0; dofind < dofs.Size(); ++dofind)
                   {
                       shared_facetdofs.insert(Hdiv_space->GetGlobalTDofNumber(dofs[dofind]));
                       std::cout << "tdof = " << Hdiv_space->GetGlobalTDofNumber(dofs[dofind]) << "\n";
                       shared_facedofs.insert(dofs[dofind]);
                   }
                   if (dofs.Size() > 0)
                   {
                       std::cout << "dofs \n";
                       dofs.Print();
                   }
                   else
                   {
                       std::cout << "dofs are empty for this shared face \n";
                   }

                   int l_face, ori;
                   pmesh->GroupFace(grind, faceind, l_face, ori);
                   const Element * face = pmesh->GetFace(l_face);

                   const int * vertices = face->GetVertices();
                   int nv = face->GetNVertices();
                   std::cout << "its vertices: \n";
                   for (int vind = 0; vind < nv; ++vind)
                   {
                       double * vcoords = pmesh->GetVertex(vertices[vind]);
                       for (int i = 0; i < pmesh->Dimension(); ++i)
                           std::cout << vcoords[i] << " ";
                       std::cout << "\n";
                   }
                   std::cout << "\n";

               }


           }

           std::set<int>::iterator it;
           std::cout << "shared face tdofs \n";
           for ( it = shared_facetdofs.begin(); it != shared_facetdofs.end(); it++ )
           {
               std::cout << *it << " ";
           }
           std::cout << "\n";

           std::cout << "shared face dofs \n";
           for ( it = shared_facedofs.begin(); it != shared_facedofs.end(); it++ )
           {
               std::cout << *it << " ";
           }

           std::cout << "\n" << std::flush;
       }
       MPI_Barrier(comm);
   } // end fo loop over all processors, one after another


   std::cout << std::flush;
   MPI_Barrier(comm);
   MPI_Finalize();
   return 0;
*/

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

   std::set<std::pair<int,int> >::iterator it;

   /*

   std::set<std::pair<int,int> > * tdofs_link_H1 = new std::set<std::pair<int,int> >;
   for (int i = 0; i < num_procs; ++i)
   {
       if (myid == i)
       {
           std::set<std::pair<int,int> > * dofs_link_H1 =
                   CreateBotToTopDofsLink("linearH1",*H1_space, pmesh->bot_to_top_bels);
           std::cout << std::flush;

           std::cout << "dof pairs for H1: \n";
           std::set<std::pair<int,int> >::iterator it;
           for ( it = dofs_link_H1->begin(); it != dofs_link_H1->end(); it++ )
           {
               std::cout << "<" << it->first << ", " << it->second << "> \n";
               int tdof1 = H1_space->GetLocalTDofNumber(it->first);
               int tdof2 = H1_space->GetLocalTDofNumber(it->second);
               std::cout << "corr. tdof pair: <" << tdof1 << "," << tdof2 << ">\n";
               if (tdof1 * tdof2 < 0)
                   MFEM_ABORT( "unsupported case: tdof1 and tdof2 belong to different processors! \n");

               if (tdof1 > -1)
                   tdofs_link_H1->insert(std::pair<int,int>(tdof1, tdof2));
               else
                   std::cout << "Ignored dofs pair which are not own tdofs \n";
           }
       }
       MPI_Barrier(comm);
   } // end fo loop over all processors, one after another

   if (verbose)
        std::cout << "Drawing in H1 case \n";

   ParGridFunction * testfullH1 = new ParGridFunction(H1_space);
   FunctionCoefficient testH1_coeff(testH1fun);
   testfullH1->ProjectCoefficient(testH1_coeff);
   Vector testfullH1_tdofs(H1_space->TrueVSize());
   testfullH1->ParallelProject(testfullH1_tdofs);

   Vector testH1_bot_tdofs(H1_space->TrueVSize());
   testH1_bot_tdofs = 0.0;

   for ( it = tdofs_link_H1->begin(); it != tdofs_link_H1->end(); it++ )
   {
       testH1_bot_tdofs[it->first] = testfullH1_tdofs[it->first];
   }

   ParGridFunction * testH1_bot = new ParGridFunction(H1_space);
   testH1_bot->Distribute(&testH1_bot_tdofs);

   Vector testH1_top_tdofs(H1_space->TrueVSize());
   testH1_top_tdofs = 0.0;

   //std::set<std::pair<int,int> >::iterator it;
   for ( it = tdofs_link_H1->begin(); it != tdofs_link_H1->end(); it++ )
   {
       testH1_top_tdofs[it->second] = testfullH1_tdofs[it->second];
   }

   ParGridFunction * testH1_top = new ParGridFunction(H1_space);
   testH1_top->Distribute(&testH1_top_tdofs);

   if (verbose)
        std::cout << "Sending to GLVis in H1 case \n";

   //MPI_Finalize();
   //return 0;
   {
       char vishost[] = "localhost";
       int  visport   = 19916;
       socketstream u_sock(vishost, visport);
       u_sock << "parallel " << num_procs << " " << myid << "\n";
       u_sock.precision(8);
       u_sock << "solution\n" << *pmesh << *testfullH1 << "window_title 'testfullH1'"
              << endl;

       socketstream ubot_sock(vishost, visport);
       ubot_sock << "parallel " << num_procs << " " << myid << "\n";
       ubot_sock.precision(8);
       ubot_sock << "solution\n" << *pmesh << *testH1_bot << "window_title 'testH1bot'"
              << endl;

       socketstream utop_sock(vishost, visport);
       utop_sock << "parallel " << num_procs << " " << myid << "\n";
       utop_sock.precision(8);
       utop_sock << "solution\n" << *pmesh << *testH1_top << "window_title 'testH1top'"
              << endl;
   }

   */

   std::set<std::pair<int,int> > * tdofs_link_Hdiv = new std::set<std::pair<int,int> >;
   for (int i = 0; i < num_procs; ++i)
   {
       if (myid == i)
       {
           std::set<std::pair<int,int> > * dofs_link_RT0 =
                      CreateBotToTopDofsLink("RT0",*Hdiv_space, pmesh->bot_to_top_bels);
           std::cout << std::flush;

           std::cout << "dof pairs for Hdiv: \n";
           std::set<std::pair<int,int> >::iterator it;
           for ( it = dofs_link_RT0->begin(); it != dofs_link_RT0->end(); it++ )
           {
               std::cout << "<" << it->first << ", " << it->second << "> \n";
               int tdof1 = Hdiv_space->GetLocalTDofNumber(it->first);
               int tdof2 = Hdiv_space->GetLocalTDofNumber(it->second);
               std::cout << "corr. tdof pair: <" << tdof1 << "," << tdof2 << ">\n";
               if (tdof1 * tdof2 < 0)
                   MFEM_ABORT( "unsupported case: tdof1 and tdof2 belong to different processors! \n");

               if (tdof1 > -1)
                   tdofs_link_Hdiv->insert(std::pair<int,int>(tdof1, tdof2));
               else
                   std::cout << "Ignored a dofs pair which are not own tdofs \n";
           }
       }
       MPI_Barrier(comm);
   } // end fo loop over all processors, one after another

   if (verbose)
        std::cout << "Drawing in Hdiv case \n";

   ParGridFunction * testfullHdiv = new ParGridFunction(Hdiv_space);
   VectorFunctionCoefficient testHdiv_coeff(dim, testHdivfun);
   testfullHdiv->ProjectCoefficient(testHdiv_coeff);
   Vector testfullHdiv_tdofs(Hdiv_space->TrueVSize());
   testfullHdiv->ParallelProject(testfullHdiv_tdofs);

   Vector testHdiv_bot_tdofs(Hdiv_space->TrueVSize());
   testHdiv_bot_tdofs = 0.0;

   for ( it = tdofs_link_Hdiv->begin(); it != tdofs_link_Hdiv->end(); it++ )
   {
       testHdiv_bot_tdofs[it->first] = testfullHdiv_tdofs[it->first];
   }

   ParGridFunction * testHdiv_bot = new ParGridFunction(Hdiv_space);
   testHdiv_bot->Distribute(&testHdiv_bot_tdofs);

   Vector testHdiv_top_tdofs(Hdiv_space->TrueVSize());
   testHdiv_top_tdofs = 0.0;

   for ( it = tdofs_link_Hdiv->begin(); it != tdofs_link_Hdiv->end(); it++ )
   {
       testHdiv_top_tdofs[it->second] = testfullHdiv_tdofs[it->second];
   }

   ParGridFunction * testHdiv_top = new ParGridFunction(Hdiv_space);
   testHdiv_top->Distribute(&testHdiv_top_tdofs);

   if (verbose)
        std::cout << "Sending to GLVis in Hdiv case \n";

   //MPI_Finalize();
   //return 0;

   {
       char vishost[] = "localhost";
       int  visport   = 19916;
       socketstream u_sock(vishost, visport);
       u_sock << "parallel " << num_procs << " " << myid << "\n";
       u_sock.precision(8);
       u_sock << "solution\n" << *pmesh << *testfullHdiv << "window_title 'testfullHdiv'"
              << endl;

       socketstream ubot_sock(vishost, visport);
       ubot_sock << "parallel " << num_procs << " " << myid << "\n";
       ubot_sock.precision(8);
       ubot_sock << "solution\n" << *pmesh << *testHdiv_bot << "window_title 'testHdivbot'"
              << endl;

       socketstream utop_sock(vishost, visport);
       utop_sock << "parallel " << num_procs << " " << myid << "\n";
       utop_sock.precision(8);
       utop_sock << "solution\n" << *pmesh << *testHdiv_top << "window_title 'testHdivtop'"
              << endl;
   }


   MPI_Finalize();
   return 0;

   Array<HypreParMatrix*> TrueP_Hdiv(num_levels - 1);
   Array<HypreParMatrix*> TrueP_L2(num_levels - 1);
   Array<HypreParMatrix*> TrueP_Hdivskew(num_levels - 1);
   Array<HypreParMatrix*> TrueP_Hcurl(num_levels - 1);
   Array<HypreParMatrix*> TrueP_H1(num_levels - 1);

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
       pmesh_lvls[l]->PrintInfo(std::cout); if(verbose) cout << endl;

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

       SparseMatrix diag1_copy(diag1);

       SparseMatrix diag2;
       PMP_H1->GetDiag(diag2);

       diag1_copy.Add(-1.0, diag2);

       for (int i = 0; i < num_procs; ++i)
       {
           if (myid == i)
           {
               if (diag1_copy.MaxNorm() > ZEROTOL)
               {
                   std::cout << "I am " << myid << "\n";
                   std::cout << "For H1 diagonal blocks are not equal, max norm = " << diag1_copy.MaxNorm() << "! \n";
                   std::cout << "\n" << std::flush;
               }

           }
           MPI_Barrier(comm);

       } // end fo loop over all processors, one after another

       SparseMatrix offd1;
       int * cmap1;
       Mass_H1c->GetOffd(offd1, cmap1);

       SparseMatrix offd1_copy(offd1);

       SparseMatrix offd2;
       int * cmap2;
       PMP_H1->GetOffd(offd2, cmap2);

       offd1_copy.Add(-1.0, offd2);

       for (int i = 0; i < num_procs; ++i)
       {
           if (myid == i)
           {
               if (offd1_copy.MaxNorm() > ZEROTOL)
               {
                   //std::cout << "I am " << myid << "\n";
                   //std::cout << "For H1 off-diagonal blocks are not equal, max norm = " << offd1_copy.MaxNorm() << "! \n";

#ifdef VERBOSE_OFFD
                   Compare_Offd_detailed(offd1, cmap1, offd2, cmap2);
                   //std::cout << "\n" << std::flush;
#endif
               }

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

       SparseMatrix diag1_copy(diag1);

       SparseMatrix diag2;
       PMP_Hdiv->GetDiag(diag2);

       diag1_copy.Add(-1.0, diag2);

       /*
       Array<int> all_bdr(pmesh_lvls[1]->bdr_attributes.Max());
       all_bdr = 1;
       Array<int> bdrdofs;
       Hdiv_space_lvls[1]->GetEssentialTrueDofs(all_bdr, bdrdofs);

       int tdof_offset = Hdiv_space_lvls[1]->GetMyTDofOffset();

       int ngroups = pmesh_lvls[1]->GetNGroups();

       for (int i = 0; i < num_procs; ++i)
       {
           if (myid == i)
           {
               std::cout << "I am " << myid << "\n";
               std::set<int> shared_facetdofs;

               for (int grind = 0; grind < ngroups; ++grind)
               {
                   int ngroupfaces = pmesh_lvls[1]->GroupNFaces(grind);
                   std::cout << "ngroupfaces = " << ngroupfaces << "\n";
                   Array<int> dofs;
                   for (int faceind = 0; faceind < ngroupfaces; ++faceind)
                   {
                       Hdiv_space_lvls[1]->GetSharedFaceDofs(grind, faceind, dofs);
                       for (int dofind = 0; dofind < dofs.Size(); ++dofind)
                       {
                           shared_facetdofs.insert(Hdiv_space_lvls[1]->GetGlobalTDofNumber(dofs[dofind]));
                       }
                   }
               }

               std::cout << "shared face tdofs \n";
               std::set<int>::iterator it;
               for ( it = shared_facetdofs.begin(); it != shared_facetdofs.end(); it++ )
               {
                   std::cout << *it << " ";
               }

               std::cout << "my tdof offset = " << tdof_offset << "\n";
               if (diag1_copy.MaxNorm() > ZEROTOL)
               {
                   std::cout << "I am " << myid << "\n";
                   std::cout << "For Hdiv diagonal blocks are not equal, max norm = " << diag1_copy.MaxNorm() << "! \n";
                   std::cout << "\n" << std::flush;
               }

               std::cout << "\n" << std::flush;
           }
           MPI_Barrier(comm);
       } // end fo loop over all processors, one after another
       */

       SparseMatrix offd1;
       int * cmap1;
       Mass_Hdivc->GetOffd(offd1, cmap1);

       SparseMatrix offd1_copy(offd1);

       SparseMatrix offd2;
       int * cmap2;
       PMP_Hdiv->GetOffd(offd2, cmap2);

       offd1_copy.Add(-1.0, offd2);

       for (int i = 0; i < num_procs; ++i)
       {
           if (myid == i)
           {
               if (offd1_copy.MaxNorm() > ZEROTOL)
               {
                   //std::cout << "I am " << myid << "\n";
                   //std::cout << "For Hdiv off-diagonal blocks are not equal, max norm = " << offd1_copy.MaxNorm() << "! \n";
#ifdef VERBOSE_OFFD
                   Compare_Offd_detailed(offd1, cmap1, offd2, cmap2);

                   /*
                   std::cout << "bdrdofs \n";
                   for (int i = 0; i < bdrdofs.Size(); ++i )
                       std::cout << bdrdofs[i] << " ";
                   */

                   /*
                   std::set<int> bdr_columns;
                   for (int i = 0; i < bdrdofs.Size(); ++i )
                       bdr_columns.insert(bdrdofs[i]);
                   */

                   /*
                   std::cout << "bdr columns \n";
                   std::set<int>::iterator it;
                   for ( it = bdr_columns.begin(); it != bdr_columns.end(); it++ )
                   {
                       std::cout << *it << " ";
                   }

                   std::cout << "\n" << std::flush;
                   */
                   //std::cout << "\n" << std::flush;
#endif
               }

           }
           MPI_Barrier(comm);
       } // end fo loop over all processors, one after another

       std::cout << "\n" << std::flush;
       MPI_Barrier(comm);

       // checking on a particular vector
       /*
       Vector truevec_c(TrueP_Hdiv[0]->Width());
       truevec_c = 0.0;
       int ort_index;
       if (myid == 0)
       {
           ort_index = 48;// 4D, two_pentatops.MFEM, sref = 3, pref = 0
           //ort_index = 9;// 3D, cube_3d_moderate, sref = 0, pref = 0
           truevec_c[ort_index] = 1.0;
       }


       Vector Atruevec_c(TrueP_Hdiv[0]->Width());
       Mass_Hdivc->Mult(truevec_c, Atruevec_c);

       for (int i = 0; i < num_procs; ++i)
       {
           if (myid == i)
           {
               std::cout << "I am " << myid << "\n";
               std::cout << "My Atruvec_c \n";
               int nnz_found = 0.0;
               for (int i = 0; i < Atruevec_c.Size(); ++i)
                   if (fabs(Atruevec_c[i]) > ZEROTOL)
                   {
                        std::cout << "Atruevec_c[" << i << "] = " << Atruevec_c[i] << " ";
                        nnz_found ++;
                   }
               std::cout << "\n";
               std::cout << "nnz_found in Atruevec_c = " << nnz_found << "\n";
               std::cout << "\n" << std::flush;
           }
           MPI_Barrier(comm);
       } // end of the loop over all processors, one after another

       // checking the marked row of Mass_Hdiv_c for proc 0
       if (myid == 0)
       {
           int row = ort_index;
           std::cout << "row = " << row << "\n";
           int nnz_rowshift_diag = diag1.GetI()[row];
           for (int colind = 0; colind < diag1.RowSize(row); ++colind)
           {
               int col1 = diag1.GetJ()[nnz_rowshift_diag + colind];
               double val1 = diag1.GetData()[nnz_rowshift_diag + colind];
               std::cout << "col1 = " << col1 << ", value1 = " << val1 << "\n";
           }
           std::cout << "GetRowNorml1 for diag1 = " << diag1.GetRowNorml1(row) << "\n";

           int nnz_rowshift_offd = offd1.GetI()[row];
           for (int colind = 0; colind < offd1.RowSize(row); ++colind)
           {
               int col1 = offd1.GetJ()[nnz_rowshift_offd + colind];
               int truecol1 = cmap1[col1];
               double val1 = offd1.GetData()[nnz_rowshift_offd + colind];
               std::cout << "col1 = " << col1 << ", truecol1 = " << truecol1 << " value1 = " << val1 << "\n";
           }
           std::cout << "GetRowNorml1 for offd1 = " << offd1.GetRowNorml1(row) << "\n";

           std::cout << "\n" << std::flush;
       }
       MPI_Barrier(comm);

       Vector truevec_f(TrueP_Hdiv[0]->Height());
       TrueP_Hdiv[0]->Mult(truevec_c, truevec_f);
       for (int i = 0; i < num_procs; ++i)
       {
           if (myid == i)
           {
               std::cout << "I am " << myid << "\n";
               std::cout << "My truvec_f \n";
               int nnz_found = 0.0;
               for (int i = 0; i < truevec_f.Size(); ++i)
                   if (fabs(truevec_f[i]) > ZEROTOL)
                   {
                        std::cout << "truevec_f[" << i << "] = " << truevec_f[i] << " ";
                        nnz_found ++;
                   }
               std::cout << "\n";
               std::cout << "nnz_found in truevec_f = " << nnz_found << "\n";
               std::cout << "\n" << std::flush;
           }
           MPI_Barrier(comm);
       } // end of the loop over all processors, one after another

       Vector truevec_cback(TrueP_Hdiv[0]->Width());
       TrueP_Hdiv[0]->MultTranspose(truevec_f, truevec_cback);

       for (int i = 0; i < num_procs; ++i)
       {
           if (myid == i)
           {
               std::cout << "I am " << myid << "\n";
               std::cout << "My truvec_cback \n";
               int nnz_found = 0.0;
               for (int i = 0; i < truevec_cback.Size(); ++i)
                   if (fabs(truevec_cback[i]) > ZEROTOL)
                   {
                        std::cout << "truevec_cback[" << i << "] = " << truevec_cback[i] << " ";
                        nnz_found ++;
                   }
               std::cout << "\n";
               std::cout << "nnz_found in truevec_cback = " << nnz_found << "\n";
               std::cout << "\n" << std::flush;
           }
           MPI_Barrier(comm);
       } // end of the loop over all processors, one after another

       Vector Atruevec_f(TrueP_Hdiv[0]->Height());
       Mass_Hdivf->Mult(truevec_f, Atruevec_f);

       Vector Atruevec_cback(TrueP_Hdiv[0]->Width());
       TrueP_Hdiv[0]->MultTranspose(Atruevec_f, Atruevec_cback);

       for (int i = 0; i < num_procs; ++i)
       {
           if (myid == i)
           {
               std::cout << "I am " << myid << "\n";
               std::cout << "My Atruvec_cback \n";
               int nnz_found = 0.0;
               for (int i = 0; i < Atruevec_cback.Size(); ++i)
                   if (fabs(Atruevec_cback[i]) > ZEROTOL)
                   {
                        std::cout << "Atruevec_cback[" << i << "] = " << Atruevec_cback[i] << " ";
                        nnz_found ++;
                   }
               std::cout << "\n";
               std::cout << "nnz_found in Atruevec_cback = " << nnz_found << "\n";
               std::cout << "\n" << std::flush;
           }
           MPI_Barrier(comm);
       } // end of the loop over all processors, one after another
       */

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

       SparseMatrix diag1_copy(diag1);

       SparseMatrix diag2;
       PMP_L2->GetDiag(diag2);

       diag1_copy.Add(-1.0, diag2);

       for (int i = 0; i < num_procs; ++i)
       {
           if (myid == i)
           {
               if (diag1_copy.MaxNorm() > ZEROTOL)
               {
                   std::cout << "I am " << myid << "\n";
                   std::cout << "For L2 diagonal blocks are not equal, max norm = " << diag1_copy.MaxNorm() << "! \n";
                   std::cout << "\n" << std::flush;
               }

           }
           MPI_Barrier(comm);
       } // end fo loop over all processors, one after another

       SparseMatrix offd1;
       int * cmap1;
       Mass_L2c->GetOffd(offd1, cmap1);

       SparseMatrix offd1_copy(offd1);

       SparseMatrix offd2;
       int * cmap2;
       PMP_L2->GetOffd(offd2, cmap2);

       offd1_copy.Add(-1.0, offd2);

       for (int i = 0; i < num_procs; ++i)
       {
           if (myid == i)
           {
               if (offd1_copy.MaxNorm() > ZEROTOL)
               {
                   //std::cout << "I am " << myid << "\n";
                   //std::cout << "For L2 off-diagonal blocks are not equal, max norm = " << offd1_copy.MaxNorm() << "! \n";
#ifdef VERBOSE_OFFD
                   Compare_Offd_detailed(offd1, cmap1, offd2, cmap2);
#endif
                   //std::cout << "\n" << std::flush;
               }

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

       SparseMatrix diag1_copy(diag1);

       SparseMatrix diag2;
       PMP_Hcurl->GetDiag(diag2);

       diag1_copy.Add(-1.0, diag2);

       for (int i = 0; i < num_procs; ++i)
       {
           if (myid == i)
           {
               if (diag1_copy.MaxNorm() > ZEROTOL)
               {
                   std::cout << "I am " << myid << "\n";
                   std::cout << "For Hcurl diagonal blocks are not equal, max norm = " << diag1_copy.MaxNorm() << "! \n";
                   std::cout << std::flush;
               }
           }
           MPI_Barrier(comm);
       } // end fo loop over all processors, one after another

       Array<int> all_bdr(pmesh_lvls[1]->bdr_attributes.Max());
       all_bdr = 1;
       Array<int> bdrtdofs;
       Hcurl_space_lvls[1]->GetEssentialTrueDofs(all_bdr, bdrtdofs);

       int ngroups = pmesh_lvls[1]->GetNGroups();

       int tdof_offset = Hcurl_space_lvls[1]->GetMyTDofOffset();

       /*
       std::vector<int> tdofs_to_edges_fine(Hcurl_space_lvls[0]->TrueVSize());
       std::vector<int> tdofs_to_edges_coarse(Hcurl_space_lvls[1]->TrueVSize());

       for (int i = 0; i < num_procs; ++i)
       {
           if (myid == i)
           {
               std::cout << "I am " << myid << "\n";
               std::vector<int> edges_to_tdofs_coarse(pmesh_lvls[1]->GetNEdges());
               Array<int> dofs;
               for (int i = 0; i < pmesh_lvls[1]->GetNEdges(); ++i)
               {
                   Hcurl_space_lvls[1]->GetEdgeDofs(i, dofs);
                   if (dofs.Size() != 1)
                       std::cout << "error: dofs size must be 1 but equals " << dofs.Size() << "\n";
                   //std::cout << "edge:" << i << " its dofs: " << dofs[0] << "\n";
                   edges_to_tdofs_coarse[i] = Hcurl_space_lvls[1]->GetLocalTDofNumber(dofs[0]);
               }
               for (int i = 0; i < pmesh_lvls[1]->GetNEdges(); ++i)
                   if (edges_to_tdofs_coarse[i] > -1)
                       tdofs_to_edges_coarse[edges_to_tdofs_coarse[i]] = i;

               //std::cout << "Look at my tdofs_to-edges relation for the coarse mesh: \n";
               //for (int i = 0; i < tdofs_to_edges_coarse.size(); ++i)
                   //std::cout << "tdof: " << i << " edge: " << tdofs_to_edges_coarse[i] << "\n";


               std::vector<int> edges_to_tdofs_fine(pmesh_lvls[0]->GetNEdges());
               //Array<int> dofs;
               for (int i = 0; i < pmesh_lvls[0]->GetNEdges(); ++i)
               {
                   Hcurl_space_lvls[0]->GetEdgeDofs(i, dofs);
                   if (dofs.Size() != 1)
                       std::cout << "error: dofs size must be 1 but equals " << dofs.Size() << "\n";
                   //std::cout << "edge:" << i << " its dofs: " << dofs[0] << "\n";
                   edges_to_tdofs_fine[i] = Hcurl_space_lvls[0]->GetLocalTDofNumber(dofs[0]);
               }
               for (int i = 0; i < pmesh_lvls[0]->GetNEdges(); ++i)
                   if (edges_to_tdofs_fine[i] > -1)
                       tdofs_to_edges_fine[edges_to_tdofs_fine[i]] = i;

               //std::cout << "Look at my tdofs_to-edges relation for the fine mesh: \n";
               //for (int i = 0; i < tdofs_to_edges_fine.size(); ++i)
                   //std::cout << "tdof: " << i << " edge: " << tdofs_to_edges_fine[i] << "\n";

               std::cout << "\n" << std::flush;
           }
           MPI_Barrier(comm);
       } // end fo loop over all processors, one after another
       */

       // testing on a linear (or constant) function

       /*

       SparseMatrix diag_P;
       TrueP_Hcurl[0]->GetDiag(diag_P);

       SparseMatrix offd_P;
       int * cmap_P;
       TrueP_Hcurl[0]->GetOffd(offd_P, cmap_P);

       VectorFunctionCoefficient vecfun_coeff(dim, testVectorFun);
       ParGridFunction * testgrfun_coarse = new ParGridFunction(Hcurl_space_lvls[1]);
       testgrfun_coarse->ProjectCoefficient(vecfun_coeff);
       Vector testv_coarse(Hcurl_space_lvls[1]->TrueVSize());
       testgrfun_coarse->ParallelProject(testv_coarse);

       ParGridFunction * testgrfun_fine = new ParGridFunction(Hcurl_space_lvls[0]);
       testgrfun_fine->ProjectCoefficient(vecfun_coeff);
       Vector testv_fine(Hcurl_space_lvls[0]->TrueVSize());
       testgrfun_fine->ParallelProject(testv_fine);

       Vector testv_proj(Hcurl_space_lvls[0]->TrueVSize());
       TrueP_Hcurl[0]->Mult(testv_coarse, testv_proj);

       Vector testv_diff(Hcurl_space_lvls[0]->TrueVSize());
       testv_diff = testv_proj;
       testv_diff -= testv_fine;

       std::set<int> bad_rows_P;
       bool first_bad_row_P = true;
       std::set<int> bad_row_cols_P;

       /*
        * failed to find the bad fine edge from parallel case.
        * the meshes are geometrically different for np = 1 and np = 2? weird
       if (num_procs == 1 && dim == 4)
       {
           int frow_special = -1;
           for (int frow = 0; frow < diag_P.Height(); ++frow)
           {
               int edgeind = tdofs_to_edges_fine[frow];

               std::cout << "edgeind = " << edgeind << "\n";

               Array<int> edgeverts;
               pmesh_lvls[0]->GetEdgeVertices(edgeind, edgeverts);

               bool find_vertex1;
               bool find_vertex2;
               int found = 0;
               for (int i = 0; i < edgeverts.Size(); ++i)
               {
                   double * vertcoos = pmesh_lvls[0]->GetVertex(edgeverts[i]);

                   find_vertex1 = ( fabs(vertcoos[0] - 0.5) < 1.0e-10 && fabs(vertcoos[1] - 0.875) < 1.0e-10
                           && fabs(vertcoos[2] - 0.0) < 1.0e-10 && fabs(vertcoos[3] - 0.5) < 1.0e-10);
                   find_vertex2 = ( fabs(vertcoos[0] - 0.625) < 1.0e-10 && fabs(vertcoos[1] - 0.75) < 1.0e-10
                           && fabs(vertcoos[2] - 0.25) < 1.0e-10 && fabs(vertcoos[3] - 0.5) < 1.0e-10);

                   if (find_vertex1 || find_vertex2)
                   {
                       std::cout << "Find something: vertex1 ? " << find_vertex1 << ", vertex2 ? " << find_vertex2 << "\n";
                       found++;
                   }
               }

               if (found == 2)
               {
                   frow_special = frow;
                   std::cout << "found the desired edge! fine row = " << frow_special << "\n";
               }
           }

           std::cout << "Looking at the `bad in parallel' row: " << frow_special << " in P \n";
           std::set<int> special_row_cols_P;
           {
               int rowsize = diag_P.RowSize(frow_special);
               int * cols = diag_P.GetRowColumns(frow_special);
               double * entries = diag_P.GetRowEntries(frow_special);
               for (int j = 0; j < rowsize; ++j)
               {
                   std::cout << "(" << cols[j] << ", " << entries[j] << ") for " << testv_coarse[cols[j]] << " ";
                   special_row_cols_P.insert(cols[j]);
               }
           }
           std::cout << "\n";

           std::cout << "edges for `bad in parallel' row cols of P: \n";
           std::set<int>::iterator it;
           for ( it = special_row_cols_P.begin(); it != special_row_cols_P.end(); it++ )
           {
               std::cout << "special row col: " << *it << "\n";
               int edgeind = tdofs_to_edges_coarse[*it];

               std::cout << "its edge coords: ";
               Array<int> edgeverts;
               pmesh_lvls[1]->GetEdgeVertices(edgeind, edgeverts);
               for (int i = 0; i < edgeverts.Size(); ++i)
               {
                   double * vertcoos = pmesh_lvls[1]->GetVertex(edgeverts[i]);
                   std::cout << "(";
                   for (int cooind = 0; cooind < pmesh_lvls[1]->Dimension(); ++cooind)
                   {
                       if (cooind > 0)
                            std::cout << ", " << vertcoos[cooind];
                       else
                           std::cout << vertcoos[cooind];
                   }
                   std::cout << ") ";
               }
               std::cout << "\n";

           }
           std::cout << "\n";
       }

       MPI_Finalize();
       return 0;
       */

       /*
       for (int i = 0; i < num_procs; ++i)
       {
           if (myid == i)
           {
               std::cout << "I am " << myid << "\n";
               std::cout << "testv_diff norm = " << testv_diff.Norml2() << "\n";
               //testv_diff.Print();

               //std::cout << "testv_fine \n";
               //testv_fine.Print();

               //std::cout << "testv_proj \n";
               //testv_proj.Print();

               if (testv_diff.Norml2() > 1.0e-15)
               {
                   for (int i = 0; i < testv_diff.Size(); ++i)
                   {
                       if (fabs(testv_diff[i]) > 1.0e-15)
                       {
                           bad_rows_P.insert(i);
                           std::cout << "row: " << i << " of P has wrong entries! \n";
                           std::cout << "correct from fine projection = " << testv_fine[i] << ", interpolated = " << testv_proj[i] << "\n";
                           //break;
                       }
                   }

                   std::cout << "bad rows of P: ";
                   std::set<int>::iterator it;
                   for ( it = bad_rows_P.begin(); it != bad_rows_P.end(); it++ )
                   {
                       std::cout << *it << " ";
                   }
                   std::cout << "\n";

                   int count = 0;
                   for ( it = bad_rows_P.begin(); it != bad_rows_P.end(); it++ )
                   {
                       int brow = *it;
                       //int brow = 100;

                       std::cout << "Looking at the bad row: " << brow << " in P \n";
                       std::cout << "its fine grid edge: \n";
                       {
                           int edgeind = tdofs_to_edges_fine[brow];

                           std::cout << "its edge coords: ";
                           Array<int> edgeverts;
                           pmesh_lvls[0]->GetEdgeVertices(edgeind, edgeverts);
                           for (int i = 0; i < edgeverts.Size(); ++i)
                           {
                               double * vertcoos = pmesh_lvls[0]->GetVertex(edgeverts[i]);
                               std::cout << "(";
                               for (int cooind = 0; cooind < pmesh_lvls[0]->Dimension(); ++cooind)
                               {
                                   if (cooind > 0)
                                        std::cout << ", " << vertcoos[cooind];
                                   else
                                       std::cout << vertcoos[cooind];
                               }
                               std::cout << ") ";
                           }
                           std::cout << "\n";
                       }

                       // for diag part of P
                       {
                           int rowsize = diag_P.RowSize(brow);
                           int * cols = diag_P.GetRowColumns(brow);
                           double * entries = diag_P.GetRowEntries(brow);
                           for (int j = 0; j < rowsize; ++j)
                           {
                               std::cout << "(" << cols[j] << ", " << entries[j] << ") for " << testv_coarse[cols[j]] << " ";
                               if (first_bad_row_P)
                                  bad_row_cols_P.insert(cols[j]);
                           }
                       }
                       std::cout << "\n";
                       // for offd part of P
                       {
                           int rowsize = offd_P.RowSize(brow);
                           int * cols = offd_P.GetRowColumns(brow);
                           double * entries = offd_P.GetRowEntries(brow);
                           for (int j = 0; j < rowsize; ++j)
                           {
                               std::cout << "(" << cols[j] << "=(true)" << cmap_P[cols[j]] << ", " << entries[j] << ") for "
                                         << testv_coarse[cmap_P[cols[j]]] << " ";
                               if (first_bad_row_P)
                                  bad_row_cols_P.insert(cmap_P[cols[j]]);
                           }
                       }
                       std::cout << "\n";

                       if (first_bad_row_P)
                          first_bad_row_P = false;


                       std::cout << "bad row cols of P: ";
                       std::set<int>::iterator it;
                       for ( it = bad_row_cols_P.begin(); it != bad_row_cols_P.end(); it++ )
                       {
                           std::cout << *it << " ";
                       }
                       std::cout << "\n";


                       std::cout << "edges for bad row cols of P: \n";
                       for ( it = bad_row_cols_P.begin(); it != bad_row_cols_P.end(); it++ )
                       {
                           std::cout << "bad row col: " << *it << "\n";
                           int edgeind = tdofs_to_edges_coarse[*it];

                           std::cout << "its edge coords: ";
                           Array<int> edgeverts;
                           pmesh_lvls[1]->GetEdgeVertices(edgeind, edgeverts);
                           for (int i = 0; i < edgeverts.Size(); ++i)
                           {
                               double * vertcoos = pmesh_lvls[1]->GetVertex(edgeverts[i]);
                               std::cout << "(";
                               for (int cooind = 0; cooind < pmesh_lvls[1]->Dimension(); ++cooind)
                               {
                                   if (cooind > 0)
                                        std::cout << ", " << vertcoos[cooind];
                                   else
                                       std::cout << vertcoos[cooind];
                               }
                               std::cout << ") ";
                           }
                           std::cout << "\n";

                       }
                       std::cout << "\n";

                       ++count;

                       if (count == 1)
                           break;

                   }


               }
               //std::cout << "Look at my testv_diff \n";
               //testv_diff.Print();
           }
           std::cout << std::flush;
           MPI_Barrier(comm);
       }

       int bad_row = 0;
       bool first_bad_row = true;
       int bad_col = 0;
       bool first_bad_col = true;

       for (int i = 0; i < num_procs; ++i)
       {
           if (myid == i)
           {
               std::cout << "I am " << myid << "\n";
               //std::cout << "bdrtdofs (with tdof_offset) \n";
               //for (int i = 0; i < bdrtdofs.Size(); ++i )
                   //std::cout << bdrtdofs[i] + tdof_offset << " ";

               std::set<int> shared_edgedofs;

               for (int grind = 0; grind < ngroups; ++grind)
               {
                   int ngroupedges = pmesh_lvls[1]->GroupNEdges(grind);
                   std::cout << "ngroupedges = " << ngroupedges << "\n";
                   Array<int> dofs;
                   Array<int> dofs2;
                   Array<int> edge_verts;
                   for (int edgeind = 0; edgeind < ngroupedges; ++edgeind)
                   {
                       //std::cout << "edgeind = " << edgeind << "\n";

                       int l_edge, ori;
                       pmesh_lvls[1]->GroupEdge(grind, edgeind, l_edge, ori);

                       Hcurl_space_lvls[1]->GetEdgeDofs(l_edge, dofs2);
                       //dofs2.Print();

                       Hcurl_space_lvls[1]->GetSharedEdgeDofs(grind, edgeind, dofs);
                       //dofs.Print();
                       for (int dofind = 0; dofind < dofs.Size(); ++dofind)
                       {
                           //std::cout << "dofs[dofind] = " << dofs[dofind] << "\n";
                           //shared_edgedofs.insert(Hcurl_space_lvls[1]->GetGlobalTDofNumber(dofs[dofind]));
                           shared_edgedofs.insert(Hcurl_space_lvls[1]->GetGlobalTDofNumber(dofs2[dofind]));
                       }

                   }
               }


               std::cout << "shared edge tdofs \n";
               std::set<int>::iterator it;
               for ( it = shared_edgedofs.begin(); it != shared_edgedofs.end(); it++ )
               {
                   std::cout << *it << " ";
               }

               std::cout << "my tdof offset = " << tdof_offset << "\n";

               std::cout << "\n" << std::flush;

               if (diag1_copy.MaxNorm() > ZEROTOL)
               {
                   std::cout << "I am " << myid << "\n";
                   std::cout << "For Hcurl diagonal blocks are not equal, max norm = " << diag1_copy.MaxNorm() << "! \n";

                   std::set<int> bad_cols;
                   std::set<int> bad_rows;

                   //int row_count = 0;
                   for (int row = 0; row < diag1_copy.Height(); ++row)
                   {
                       if (diag1_copy.GetRowNorml1(row) > 1.0e-15)
                       {
                           bad_rows.insert(row);

                           if (first_bad_row)
                               std::cout << "row: " << row << " has nonzero values! \n";
#if 0
                           std::cout << "row of diag1 \n";
                           int * cols1 = diag1.GetRowColumns(row);
                           double * entries1 = diag1.GetRowEntries(row);
                           int rowsize1 = diag1.RowSize(row);
                           for (int j = 0; j < rowsize1; ++j)
                               if (fabs(entries1[j]) > 1.0e-15)
                                    std::cout << "(" << cols1[j] << ", " << entries1[j] << ") ";
                           std::cout << "\n\n";

                           std::cout << "row of diag2 \n";
                           int * cols2 = diag2.GetRowColumns(row);
                           double * entries2 = diag2.GetRowEntries(row);
                           int rowsize2 = diag2.RowSize(row);
                           for (int j = 0; j < rowsize2; ++j)
                               if (fabs(entries2[j]) > 1.0e-15)
                                    std::cout << "(" << cols2[j] << ", " << entries2[j] << ") ";
                           std::cout << "\n\n";
#endif

                           if (first_bad_row)
                               std::cout << "row of diag1 - diag2 \n";
                           int * cols = diag1_copy.GetRowColumns(row);
                           double * entries = diag1_copy.GetRowEntries(row);
                           int rowsize = diag1_copy.RowSize(row);
                           for (int j = 0; j < rowsize; ++j)
                               if (fabs(entries[j]) > 1.0e-15)
                               {
                                   bad_cols.insert(cols[j]);

                                   if (first_bad_col == true)
                                   {
                                       bad_col = cols[j];
                                       first_bad_col = false;
                                   }
                                   if (first_bad_row)
                                       std::cout << "(" << cols[j] << ", " << entries[j] << ") ";
                               }
                           if (first_bad_row)
                               std::cout << "\n";


                           //++row_count;

                           if (first_bad_row)
                           {
                               bad_row = row;
                               if (row == 210)
                                first_bad_row = false;
                           }

                           //if (row_count == 1)
                               //bad_row = row;
                           //if (row_count == 10)
                               //break;

                       }
                   }
                   //diag1_copy.Print();

                   std::set<int>::iterator it;
                   std::cout << "bad rows: \n";
                   for ( it = bad_rows.begin(); it != bad_rows.end(); it++ )
                       std::cout << *it << " ";
                   std::cout << "\n";

                   std::cout << "bad cols: \n";
                   for ( it = bad_cols.begin(); it != bad_cols.end(); it++ )
                       std::cout << *it << " ";
                   std::cout << "\n";

                   std::set<int> bdr_cols;
                   for (int i = 0; i < bdrtdofs.Size(); ++i )
                       bdr_cols.insert(bdrtdofs[i]);

                   std::cout << "bdr cols: \n";
                   for ( it = bdr_cols.begin(); it != bdr_cols.end(); it++ )
                       std::cout << *it << " ";

                   std::cout << "\n" << std::flush;
               }

           }
           MPI_Barrier(comm);
       } // end fo loop over all processors, one after another

       // testing on a particular vector
       for (int i = 0; i < num_procs; ++i)
       {
           if (myid == i)
           {
               std::cout << "I am " << myid << "\n";
               if (bad_row >= 0)
               {
                   std::cout << "bad row = " << bad_row << ", bad col = " << bad_col << "\n";

               }
               std::cout << "\n" << std::flush;
           }
           MPI_Barrier(comm);
       } // end fo loop over all processors, one after another

#ifndef USE_TSL
       bad_row = 7;
#endif

       bad_row = 210;

       Vector testvec_c(PMP_Hcurl->Width());
       testvec_c = 0.0;
       if (myid == 0)
           testvec_c[bad_row] = 1.0;

       Vector testvec_f(TrueP_Hcurl[0]->Height());
       TrueP_Hcurl[0]->Mult(testvec_c, testvec_f);

       for (int i = 0; i < num_procs; ++i)
       {
           if (myid == i)
           {
               std::cout << "I am " << myid << "\n";
               std::cout << "Look at my testvec_c \n";
               for (int i = 0; i < testvec_c.Size(); ++i)
                   if (fabs(testvec_c[i]) > 1.0e-14)
                   {
                       std::cout << "nonzero: (" << i << ", " << testvec_c[i] << ")\n";
#if 0
                       std::cout << "its edge coords: ";
                       Array<int> edgeverts;
                       pmesh_lvls[1]->GetEdgeVertices(tdofs_to_edges_coarse[i], edgeverts);
                       for (int i = 0; i < edgeverts.Size(); ++i)
                       {
                           double * vertcoos = pmesh_lvls[1]->GetVertex(edgeverts[i]);
                           std::cout << "(";
                           for (int cooind = 0; cooind < pmesh_lvls[1]->Dimension(); ++cooind)
                           {
                               std::cout << vertcoos[cooind] << ", ";
                           }
                           std::cout << ") ";
                       }
                       std::cout << "\n";
#endif
                   }
               std::cout << "Look at my testvec_f \n";
               for (int i = 0; i < testvec_f.Size(); ++i)
                   if (fabs(testvec_f[i]) > 1.0e-14)
                   {
                       std::cout << "nonzero: (" << i << ", " << testvec_f[i] << ")\n";
#if 0
                       std::cout << "its edge coords: ";
                       Array<int> edgeverts;
                       pmesh_lvls[0]->GetEdgeVertices(tdofs_to_edges_fine[i], edgeverts);
                       for (int i = 0; i < edgeverts.Size(); ++i)
                       {
                           double * vertcoos = pmesh_lvls[0]->GetVertex(edgeverts[i]);
                           std::cout << "(";
                           for (int cooind = 0; cooind < pmesh_lvls[0]->Dimension(); ++cooind)
                           {
                               std::cout << vertcoos[cooind] << ", ";
                           }
                           std::cout << ") ";
                       }
                       std::cout << "\n";
#endif
                   }
           }
           MPI_Barrier(comm);
       } // end fo loop over all processors, one after another


       MPI_Finalize();
       return 0;
       */


       SparseMatrix offd1;
       int * cmap1;
       Mass_Hcurlc->GetOffd(offd1, cmap1);

       SparseMatrix offd1_copy(offd1);

       SparseMatrix offd2;
       int * cmap2;
       PMP_Hcurl->GetOffd(offd2, cmap2);

       offd1_copy.Add(-1.0, offd2);

       for (int i = 0; i < num_procs; ++i)
       {
           if (myid == i)
           {
               if (offd1_copy.MaxNorm() > ZEROTOL)
               {
                   std::cout << "I am " << myid << "\n";
                   //std::cout << "For Hcurl off-diagonal blocks are not equal, max norm = " << offd1_copy.MaxNorm() << "! \n";
#ifdef VERBOSE_OFFD
                   Compare_Offd_detailed(offd1, cmap1, offd2, cmap2);
#endif
                   std::cout << "If you see complains between this line and line starting with I am,"
                                " then something is wrong for the off-diagonal blocks of Hcurl as well \n";
                   std::cout << "\n" << std::flush;
               }

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

       SparseMatrix diag1_copy(diag1);

       SparseMatrix diag2;
       PMP_Hdivskew->GetDiag(diag2);

       diag1_copy.Add(-1.0, diag2);

       for (int i = 0; i < num_procs; ++i)
       {
           if (myid == i)
           {
               if (diag1_copy.MaxNorm() > ZEROTOL)
               {
                   std::cout << "I am " << myid << "\n";
                   std::cout << "For Hdivskew diagonal blocks are not equal, max norm = " << diag1_copy.MaxNorm() << "! \n";
                   std::cout << "\n" << std::flush;
               }
           }
           MPI_Barrier(comm);
       }

       /*
       int tdof_offset = Hdivskew_space_lvls[1]->GetMyTDofOffset();

       int ngroups = pmesh_lvls[1]->GetNGroups();

       for (int i = 0; i < num_procs; ++i)
       {
           if (myid == i)
           {
               std::cout << "I am " << myid << "\n";
               std::set<int> shared_planartdofs;

               for (int grind = 0; grind < ngroups; ++grind)
               {
                   int ngroupplanars = pmesh_lvls[1]->GroupNPlanars(grind);
                   std::cout << "ngroupplanars = " << ngroupplanars << "\n";
                   Array<int> dofs;
                   for (int planarind = 0; planarind < ngroupplanars; ++planarind)
                   {
                       Hdivskew_space_lvls[1]->GetSharedPlanarDofs(grind, planarind, dofs);
                       for (int dofind = 0; dofind < dofs.Size(); ++dofind)
                       {
                           shared_planartdofs.insert(Hdivskew_space_lvls[1]->GetGlobalTDofNumber(dofs[dofind]));
                       }
                   }
               }

               std::cout << "shared planar tdofs \n";
               std::set<int>::iterator it;
               for ( it = shared_planartdofs.begin(); it != shared_planartdofs.end(); it++ )
               {
                   std::cout << *it << " ";
               }

               std::cout << "my tdof offset = " << tdof_offset << "\n";
               if (diag1_copy.MaxNorm() > ZEROTOL)
               {
                   std::cout << "I am " << myid << "\n";
                   std::cout << "For Hdiv diagonal blocks are not equal, max norm = " << diag1_copy.MaxNorm() << "! \n";
                   std::cout << "\n" << std::flush;
               }

               std::cout << "\n" << std::flush;
           }
           MPI_Barrier(comm);
       } // end fo loop over all processors, one after another
       */


       SparseMatrix offd1;
       int * cmap1;
       Mass_Hdivskewc->GetOffd(offd1, cmap1);

       SparseMatrix offd1_copy(offd1);

       SparseMatrix offd2;
       int * cmap2;
       PMP_Hdivskew->GetOffd(offd2, cmap2);

       offd1_copy.Add(-1.0, offd2);

       for (int i = 0; i < num_procs; ++i)
       {
           if (myid == i)
           {
               if (offd1_copy.MaxNorm() > ZEROTOL)
               {
                   //std::cout << "I am " << myid << "\n";
                   //std::cout << "For Hdivskew off-diagonal blocks are not equal, max norm = " << offd1_copy.MaxNorm() << "! \n";
#ifdef VERBOSE_OFFD
                   Compare_Offd_detailed(offd1, cmap1, offd2, cmap2);
#endif
                   //std::cout << "\n" << std::flush;
               }

               // checking on a specific vector
               //Vector testort(offd1.Width());

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

    std::multiset<int> bad_columns;

    for ( int row = 0; row < offd1.Height(); ++row)
    {
        int nnz_rowshift1 = offd1.GetI()[row];
        std::map<int,double> row_entries1;
        for (int colind = 0; colind < offd1.RowSize(row); ++colind)
        {
            int col1 = offd1.GetJ()[nnz_rowshift1 + colind];
            int truecol1 = cmap1[col1];
            double val1 = offd1.GetData()[nnz_rowshift1 + colind];
            if (fabs(val1) > ZEROTOL)
                 row_entries1.insert(std::make_pair(truecol1, val1));
        }

        int nnz_rowshift2 = offd2.GetI()[row];
        std::map<int,double> row_entries2;
        for (int colind = 0; colind < offd2.RowSize(row); ++colind)
        {
            int col2 = offd2.GetJ()[nnz_rowshift2 + colind];
            int truecol2 = cmap2[col2];
            double val2 = offd2.GetData()[nnz_rowshift2 + colind];
            if (fabs(val2) > ZEROTOL)
                 row_entries2.insert(std::make_pair(truecol2, val2));
        }

        /*
        if (row == 48 || row == 11513 || row == 11513 - 11511)
        {
            std::cout << "very special print: row = " << row << "\n";

            std::map<int, double>::iterator it;
            for ( it = row_entries2.begin(); it != row_entries2.end(); it++ )
            {
                std::cout << "(" << it->first << ", " << it->second << ") ";
            }
            std::cout << "\n";
        }
        */

        if (row_entries1.size() != row_entries2.size())
        {
            std::cout << "row = " << row << ": ";
            std::cout << "row_entries1.size() = " << row_entries1.size() << " != " << row_entries2.size() << " = row_entries2.size() \n";
        }

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
                if ( fabs(value2 - value1) / fabs(value1) > ZEROTOL &&  (fabs(value1) > ZEROTOL || fabs(value2) > ZEROTOL ) )
                {
                    std::cout << "row = " << row << ": ";
                    std::cout << "For truecol = " << truecol1 << " values are different: " << value1 << " != " << value2 << "\n";
                    bad_columns.insert(it2->first);
                }
                row_entries2.erase(it2);
            }
            else
            {
                std::cout << "row = " << row << ": ";
                std::cout << "Cannot find pair (" << truecol1 << ", " << value1 << ") from row_entries1 in row_entries2 \n";
            }
        }

        if (row_entries2.size() != 0)
        {
            std::cout << "row = " << row << ": ";
            for ( it2 = row_entries2.begin(); it2 != row_entries2.end(); it2++ )
            {
                int truecol2 = it2->first;
                double value2 = it2->second;
                std::cout << "additional item in row_entry2: (" << truecol2 << ", " << value2 << ") ";
                bad_columns.insert(it2->first);
            }
            std::cout << "\n";
        }
        //int nnz_rowshift = offd1.GetI()[row];
        //if (offd1.GetI()[row] != offd2.GetI()[row])
            //std::cout << "offd1.GetI()[row] = " << offd1.GetI()[row] << " != " << offd2.GetI()[row] << " = offd2.GetI()[row], row = " << row << "\n";

        /*
        if (offd1.RowSize(row) != offd2.RowSize(row))
        {
            std::cout << "row = " << row << ": ";
            std::cout << "offd1.RowSize(row) = " << offd1.RowSize(row) << " != " << offd2.RowSize(row) << " = offd2.RowSize(row) \n";
        }

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
                std::cout << "row = " << row << ": ";
                std::cout << "colind = " << colind << ": " << "truecol1 = " << truecol1 << "!= " << truecol2 << " = truecol2 \n";
                std::cout << "colind = " << colind << ": " << "value1 = " << val1 << "!= " << val2 << " = value2 \n";
            }
            else
            {
                if ( fabs(val1 - val2) / fabs(val1) > 1.0e-14)
                {
                    std::cout << "row = " << row << ": ";
                    std::cout << "colind = " << colind << ": " << "value1 = " << val1 << "!= " << val2 << " = value2 \n";
                }
            }
        }

        //int nnz_rowshift2 = offd2.GetI()[row];
        if (offd1.RowSize(row) != offd2.RowSize(row))
        {
            std::cout << "row = " << row << ": ";
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
        */

    } // end of loop over all rows

    if (bad_columns.size() > 0)
    {
        std::cout << "bad columns \n";
        std::multiset<int>::iterator it3;
        for ( it3 = bad_columns.begin(); it3 != bad_columns.end(); it3++ )
        {
            std::cout << *it3 << " ";
        }
        std::cout << "\n" << std::flush;
    }

    //std::cout << "full offd1 \n";
    //offd1.Print();
    //std::cout << "\n" << std::flush;

    //std::cout << "offd1 max norm = " << offd1.MaxNorm() << "\n";
    //std::cout << "offd2 max norm = " << offd2.MaxNorm() << "\n";
    //std::cout << "\n" << std::flush;
    //std::cout << "offd1 \n";
    //offd1.Print();
    //std::cout << "offd2 \n";
    //offd2.Print();

}

void testVectorFun(const Vector& xt, Vector& res)
{
    /*
    double x = xt(0);
    double y = xt(1);
    double z;
    if (xt.Size() > 3)
        z = xt(2);
    double t = xt(xt.Size() - 1);
    */

    res.SetSize(xt.Size());
    res = 1.0;
}

// eltype must be "linearH1" or "RT0", for any other finite element the code doesn't work
// the fespace must correspond to the eltype provided
// bot_to_top_bels is the link between boundary elements (at the bottom and at the top)
// which can be taken out of ParMeshTSL

std::set<std::pair<int,int> >* CreateBotToTopDofsLink(const char * eltype, FiniteElementSpace& fespace,
                                                         std::vector<std::pair<int,int> > & bot_to_top_bels, bool verbose)
{
    if (strcmp(eltype, "linearH1") != 0 && strcmp(eltype, "RT0") != 0)
        MFEM_ABORT ("Provided eltype is not supported in CreateBotToTopDofsLink: must be linearH1 or RT0 strictly! \n");

    int nbelpairs = bot_to_top_bels.size();
    // estimating the maximal memory size required
    Array<int> dofs;
    fespace.GetBdrElementDofs(0, dofs);
    int ndofpairs_max = nbelpairs * dofs.Size();

    if (verbose)
        std::cout << "nbelpairs = " << nbelpairs << ", estimated ndofpairs_max = " << ndofpairs_max << "\n";

    std::set<std::pair<int,int> > * res = new std::set<std::pair<int,int> >;
    //res->reserve(ndofpairs_max);

    Mesh * mesh = fespace.GetMesh();

    for (int i = 0; i < nbelpairs; ++i)
    {
        if (verbose)
            std::cout << "pair " << i << ": \n";

        if (strcmp(eltype, "RT0") == 0)
        {
            int belind_first = bot_to_top_bels[i].first;
            Array<int> bel_dofs_first;
            fespace.GetBdrElementDofs(belind_first, bel_dofs_first);

            int belind_second = bot_to_top_bels[i].second;
            Array<int> bel_dofs_second;
            fespace.GetBdrElementDofs(belind_second, bel_dofs_second);

            if (verbose)
            {
                std::cout << "bel_dofs_first \n";
                bel_dofs_first.Print();
                std::cout << "bel_dofs_second \n";
                bel_dofs_second.Print();
            }


            if (bel_dofs_first.Size() != 1 || bel_dofs_second.Size() != 1)
            {
                MFEM_ABORT("For RT0 exactly one dof must correspond to each boundary element \n");
            }

            res->insert(std::pair<int,int>(bel_dofs_first[0], bel_dofs_second[0]));

        }

        if (strcmp(eltype, "linearH1") == 0)
        {
            int belind_first = bot_to_top_bels[i].first;
            Array<int> bel_dofs_first;
            fespace.GetBdrElementDofs(belind_first, bel_dofs_first);

            Array<int> belverts_first;
            mesh->GetBdrElementVertices(belind_first, belverts_first);

            int nverts = mesh->GetBdrElement(belind_first)->GetNVertices();

            int belind_second = bot_to_top_bels[i].second;
            Array<int> bel_dofs_second;
            fespace.GetBdrElementDofs(belind_second, bel_dofs_second);

            if (verbose)
            {
                std::cout << "bel_dofs first: \n";
                bel_dofs_first.Print();

                std::cout << "bel_dofs second: \n";
                bel_dofs_second.Print();
            }

            Array<int> belverts_second;
            mesh->GetBdrElementVertices(belind_second, belverts_second);


            if (bel_dofs_first.Size() != nverts || bel_dofs_second.Size() != nverts)
            {
                MFEM_ABORT("For linearH1 exactly #bel.vertices of dofs must correspond to each boundary element \n");
            }

            /*
            Array<int> P, Po;
            fespace.GetMesh()->GetBdrElementPlanars(i, P, Po);

            std::cout << "P: \n";
            P.Print();
            std::cout << "Po: \n";
            Po.Print();

            Array<int> belverts_first;
            mesh->GetBdrElementVertices(belind_first, belverts_first);
            */

            std::vector<std::vector<double> > vertscoos_first(nverts);
            if (verbose)
                std::cout << "verts of first bdr el \n";
            for (int vert = 0; vert < nverts; ++vert)
            {
                vertscoos_first[vert].resize(mesh->Dimension());
                double * vertcoos = mesh->GetVertex(belverts_first[vert]);
                if (verbose)
                    std::cout << "vert = " << vert << ": ";
                for (int j = 0; j < mesh->Dimension(); ++j)
                {
                    vertscoos_first[vert][j] = vertcoos[j];
                    if (verbose)
                        std::cout << vertcoos[j] << " ";
                }
                if (verbose)
                    std::cout << "\n";
            }

            int * verts_permutation_first = new int[nverts];
            sortingPermutationNew(vertscoos_first, verts_permutation_first);

            if (verbose)
            {
                std::cout << "permutation first: ";
                for (int i = 0; i < mesh->Dimension(); ++i)
                    std::cout << verts_permutation_first[i] << " ";
                std::cout << "\n";
            }

            std::vector<std::vector<double> > vertscoos_second(nverts);
            if (verbose)
                std::cout << "verts of second bdr el \n";
            for (int vert = 0; vert < nverts; ++vert)
            {
                vertscoos_second[vert].resize(mesh->Dimension());
                double * vertcoos = mesh->GetVertex(belverts_second[vert]);
                if (verbose)
                    std::cout << "vert = " << vert << ": ";
                for (int j = 0; j < mesh->Dimension(); ++j)
                {
                    vertscoos_second[vert][j] = vertcoos[j];
                    if (verbose)
                        std::cout << vertcoos[j] << " ";
                }
                if (verbose)
                    std::cout << "\n";
            }

            int * verts_permutation_second = new int[nverts];
            sortingPermutationNew(vertscoos_second, verts_permutation_second);

            if (verbose)
            {
                std::cout << "permutation second: ";
                for (int i = 0; i < mesh->Dimension(); ++i)
                    std::cout << verts_permutation_second[i] << " ";
                std::cout << "\n";
            }

            /*
            int * verts_perm_second_inverse = new int[nverts];
            invert_permutation(verts_permutation_second, nverts, verts_perm_second_inverse);

            if (verbose)
            {
                std::cout << "inverted permutation second: ";
                for (int i = 0; i < mesh->Dimension(); ++i)
                    std::cout << verts_perm_second_inverse[i] << " ";
                std::cout << "\n";
            }
            */

            int * verts_perm_first_inverse = new int[nverts];
            invert_permutation(verts_permutation_first, nverts, verts_perm_first_inverse);

            if (verbose)
            {
                std::cout << "inverted permutation first: ";
                for (int i = 0; i < mesh->Dimension(); ++i)
                    std::cout << verts_perm_first_inverse[i] << " ";
                std::cout << "\n";
            }


            for (int dofno = 0; dofno < bel_dofs_first.Size(); ++dofno)
            {
                //int dofno_second = verts_perm_second_inverse[verts_permutation_first[dofno]];
                int dofno_second = verts_permutation_second[verts_perm_first_inverse[dofno]];
                res->insert(std::pair<int,int>(bel_dofs_first[dofno],
                                                  bel_dofs_second[dofno_second]));

                if (verbose)
                    std::cout << "matching dofs pair: <" << bel_dofs_first[dofno] << ","
                          << bel_dofs_second[dofno_second] << "> \n";
            }

            if (verbose)
               std::cout << "\n";
        }

    } // end of loop over all pairs of boundary elements

    if (verbose)
    {
        if (strcmp(eltype,"RT0") == 0)
            std::cout << "dof pairs for Hdiv: \n";
        if (strcmp(eltype,"linearH1") == 0)
            std::cout << "dof pairs for H1: \n";
        std::set<std::pair<int,int> >::iterator it;
        for ( it = res->begin(); it != res->end(); it++ )
        {
            std::cout << "<" << it->first << ", " << it->second << "> \n";
        }
    }


    return res;
}

double testH1fun(Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size() - 1);

    if (xt.Size() == 3)
        return (x*x + y*y + 1.0);
    if (xt.Size() == 4)
        return (x*x + y*y + z*z + 1.0);
    return 0.0;
}


void testHdivfun(const Vector& xt, Vector &res)
{
    res.SetSize(xt.Size());

    double x = xt(0);
    double y = xt(1);
    double z;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size() - 1);

    res = 0.0;

    if (xt.Size() == 3)
    {
        res(2) = (x*x + y*y + 1.0);
    }
    if (xt.Size() == 4)
    {
        res(3) = (x*x + y*y + z*z + 1.0);
    }
}
