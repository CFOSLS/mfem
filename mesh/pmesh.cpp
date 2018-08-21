// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "mesh_headers.hpp"
#include "../fem/fem.hpp"
#include "../general/sets.hpp"
#include "../general/sort_pairs.hpp"
#include "../general/text.hpp"

#include <iostream>
using namespace std;

namespace mfem
{

ParMesh::ParMesh(const ParMesh &pmesh, bool copy_nodes)
   : Mesh(pmesh, false),
     group_svert(pmesh.group_svert),
     group_sedge(pmesh.group_sedge),
     group_splan(pmesh.group_splan),
     group_sface(pmesh.group_sface),
     gtopo(pmesh.gtopo)
{
   MyComm = pmesh.MyComm;
   NRanks = pmesh.NRanks;
   MyRank = pmesh.MyRank;

   // Duplicate the shared_edges
   shared_edges.SetSize(pmesh.shared_edges.Size());
   for (int i = 0; i < shared_edges.Size(); i++)
   {
      shared_edges[i] = pmesh.shared_edges[i]->Duplicate(this);
   }

   // Duplicate the shared_planars
   shared_planars.SetSize(pmesh.shared_planars.Size());
   for (int i = 0; i < shared_planars.Size(); i++)
   {
      shared_planars[i] = pmesh.shared_planars[i]->Duplicate(this);
   }

   // Duplicate the shared_faces
   shared_faces.SetSize(pmesh.shared_faces.Size());
   for (int i = 0; i < shared_faces.Size(); i++)
   {
      shared_faces[i] = pmesh.shared_faces[i]->Duplicate(this);
   }

   // Copy the shared-to-local index Arrays
   pmesh.svert_lvert.Copy(svert_lvert);
   pmesh.sedge_ledge.Copy(sedge_ledge);
   pmesh.splan_lplan.Copy(splan_lplan);
   pmesh.sface_lface.Copy(sface_lface);

   // Do not copy face-neighbor data (can be generated if needed)
   have_face_nbr_data = false;

   MFEM_VERIFY(pmesh.pncmesh == NULL,
               "copying non-conforming meshes is not implemented");
   pncmesh = NULL;

   // Copy the Nodes as a ParGridFunction, including the FiniteElementCollection
   // and the FiniteElementSpace (as a ParFiniteElementSpace)
   if (pmesh.Nodes && copy_nodes)
   {
      FiniteElementSpace *fes = pmesh.Nodes->FESpace();
      const FiniteElementCollection *fec = fes->FEColl();
      FiniteElementCollection *fec_copy =
         FiniteElementCollection::New(fec->Name());
      ParFiniteElementSpace *pfes_copy =
         new ParFiniteElementSpace(this, fec_copy, fes->GetVDim(),
                                   fes->GetOrdering());
      Nodes = new ParGridFunction(pfes_copy);
      Nodes->MakeOwner(fec_copy);
      *Nodes = *pmesh.Nodes;
      own_nodes = 1;
   }
}

ParMesh::ParMesh(MPI_Comm comm, Mesh &mesh, int *partitioning_,
                 int part_method)
   : gtopo(comm)
{
   int i, j;
   int *partitioning;
   Array<bool> activeBdrElem;

   MyComm = comm;
   MPI_Comm_size(MyComm, &NRanks);
   MPI_Comm_rank(MyComm, &MyRank);

   if (mesh.Nonconforming())
   {
      pncmesh = new ParNCMesh(comm, *mesh.ncmesh);

      // save the element partitioning before Prune()
      int* partition = new int[mesh.GetNE()];
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         partition[i] = pncmesh->InitialPartition(i);
      }

      pncmesh->Prune();

      Mesh::InitFromNCMesh(*pncmesh);
      pncmesh->OnMeshUpdated(this);

      ncmesh = pncmesh;
      meshgen = mesh.MeshGenerator();

      mesh.attributes.Copy(attributes);
      mesh.bdr_attributes.Copy(bdr_attributes);

      GenerateNCFaceInfo();

      if (mesh.GetNodes())
      {
         Nodes = new ParGridFunction(this, mesh.GetNodes(), partition);
         own_nodes = 1;
      }
      delete [] partition;

      have_face_nbr_data = false;
      return;
   }

   Dim = mesh.Dim;
   spaceDim = mesh.spaceDim;

   BaseGeom = mesh.BaseGeom;
   BaseBdrGeom = mesh.BaseBdrGeom;

   ncmesh = pncmesh = NULL;

   if (partitioning_)
   {
      partitioning = partitioning_;
   }
   else
   {
      partitioning = mesh.GeneratePartitioning(NRanks, part_method);
   }

   // re-enumerate the partitions to better map to actual processor
   // interconnect topology !?

   Array<int> vert;
   Array<int> vert_global_local(mesh.GetNV());
   int vert_counter, element_counter, bdrelem_counter;

   // build vert_global_local
   vert_global_local = -1;

   element_counter = 0;
   vert_counter = 0;
   for (i = 0; i < mesh.GetNE(); i++)
      if (partitioning[i] == MyRank)
      {
         mesh.GetElementVertices(i, vert);
         element_counter++;
         for (j = 0; j < vert.Size(); j++)
            if (vert_global_local[vert[j]] < 0)
            {
               vert_global_local[vert[j]] = vert_counter++;
            }
      }

   NumOfVertices = vert_counter;
   NumOfElements = element_counter;
   vertices.SetSize(NumOfVertices);

   // re-enumerate the local vertices to preserve the global ordering
   for (i = vert_counter = 0; i < vert_global_local.Size(); i++)
      if (vert_global_local[i] >= 0)
      {
         vert_global_local[i] = vert_counter++;
      }

   // determine vertices
   for (i = 0; i < vert_global_local.Size(); i++)
      if (vert_global_local[i] >= 0)
      {
         vertices[vert_global_local[i]].SetCoords(mesh.SpaceDimension(),
                                                  mesh.GetVertex(i));
      }

   // determine elements
   element_counter = 0;
   elements.SetSize(NumOfElements);
   swappedElements.SetSize(NumOfElements);
   for (i = 0; i < mesh.GetNE(); i++)
      if (partitioning[i] == MyRank)
      {
         elements[element_counter] = mesh.GetElement(i)->Duplicate(this);
         if (Dim==4) { swappedElements[element_counter] = mesh.getSwappedElementInfo(i); }
         int *v = elements[element_counter]->GetVertices();
         int nv = elements[element_counter]->GetNVertices();
         for (j = 0; j < nv; j++)
         {
            v[j] = vert_global_local[v[j]];
         }
         element_counter++;
      }

   Table *edge_element = NULL;
   if (mesh.NURBSext)
   {
      activeBdrElem.SetSize(mesh.GetNBE());
      activeBdrElem = false;
   }
   // build boundary elements
   if (Dim >= 3)
   {
      NumOfBdrElements = 0;
      for (i = 0; i < mesh.GetNBE(); i++)
      {
         int face, o, el1, el2;
         mesh.GetBdrElementFace(i, &face, &o);
         mesh.GetFaceElements(face, &el1, &el2);
         if (partitioning[(o % 2 == 0 || el2 < 0) ? el1 : el2] == MyRank)
         {
            NumOfBdrElements++;
            if (mesh.NURBSext)
            {
               activeBdrElem[i] = true;
            }
         }
      }

      bdrelem_counter = 0;
      boundary.SetSize(NumOfBdrElements);
      for (i = 0; i < mesh.GetNBE(); i++)
      {
         int face, o, el1, el2;
         mesh.GetBdrElementFace(i, &face, &o);
         mesh.GetFaceElements(face, &el1, &el2);
         if (partitioning[(o % 2 == 0 || el2 < 0) ? el1 : el2] == MyRank)
         {
            boundary[bdrelem_counter] = mesh.GetBdrElement(i)->Duplicate(this);
            int *v = boundary[bdrelem_counter]->GetVertices();
            int nv = boundary[bdrelem_counter]->GetNVertices();
            for (j = 0; j < nv; j++)
            {
               v[j] = vert_global_local[v[j]];
            }
            bdrelem_counter++;
         }
      }
   }
   else if (Dim == 2)
   {
      edge_element = new Table;
      Transpose(mesh.ElementToEdgeTable(), *edge_element, mesh.GetNEdges());

      NumOfBdrElements = 0;
      for (i = 0; i < mesh.GetNBE(); i++)
      {
         int edge = mesh.GetBdrElementEdgeIndex(i);
         int el1 = edge_element->GetRow(edge)[0];
         if (partitioning[el1] == MyRank)
         {
            NumOfBdrElements++;
            if (mesh.NURBSext)
            {
               activeBdrElem[i] = true;
            }
         }
      }

      bdrelem_counter = 0;
      boundary.SetSize(NumOfBdrElements);
      for (i = 0; i < mesh.GetNBE(); i++)
      {
         int edge = mesh.GetBdrElementEdgeIndex(i);
         int el1 = edge_element->GetRow(edge)[0];
         if (partitioning[el1] == MyRank)
         {
            boundary[bdrelem_counter] = mesh.GetBdrElement(i)->Duplicate(this);
            int *v = boundary[bdrelem_counter]->GetVertices();
            int nv = boundary[bdrelem_counter]->GetNVertices();
            for (j = 0; j < nv; j++)
            {
               v[j] = vert_global_local[v[j]];
            }
            bdrelem_counter++;
         }
      }
   }
   else if (Dim == 1)
   {
      NumOfBdrElements = 0;
      for (i = 0; i < mesh.GetNBE(); i++)
      {
         int vert = mesh.boundary[i]->GetVertices()[0];
         int el1, el2;
         mesh.GetFaceElements(vert, &el1, &el2);
         if (partitioning[el1] == MyRank)
         {
            NumOfBdrElements++;
         }
      }

      bdrelem_counter = 0;
      boundary.SetSize(NumOfBdrElements);
      for (i = 0; i < mesh.GetNBE(); i++)
      {
         int vert = mesh.boundary[i]->GetVertices()[0];
         int el1, el2;
         mesh.GetFaceElements(vert, &el1, &el2);
         if (partitioning[el1] == MyRank)
         {
            boundary[bdrelem_counter] = mesh.GetBdrElement(i)->Duplicate(this);
            int *v = boundary[bdrelem_counter]->GetVertices();
            v[0] = vert_global_local[v[0]];
            bdrelem_counter++;
         }
      }
   }

   meshgen = mesh.MeshGenerator();

   mesh.attributes.Copy(attributes);
   mesh.bdr_attributes.Copy(bdr_attributes);

   // this is called by the default Mesh constructor
   // InitTables();

   if (Dim > 1)
   {
      el_to_edge = new Table;
      NumOfEdges = Mesh::GetElementToEdgeTable(*el_to_edge, be_to_edge);
   }
   else
   {
      NumOfEdges = 0;
   }

   STable3D *faces_tbl = NULL;
   STable4D *faces_tbl_4d = NULL;
   if (Dim == 3)
   {
      faces_tbl = GetElementToFaceTable(1);
   }
   else if (Dim == 4)
   {
      faces_tbl_4d = GetElementToFaceTable4D(1);
   }
   else { NumOfFaces = 0; }
   GenerateFaces();

   NumOfPlanars = 0;
   el_to_planar = NULL;
   STable3D *planar_tbl = NULL;
   if (Dim==4)
   {
      planar_tbl = GetElementToPlanarTable(1);
      GeneratePlanars();
   }

   ListOfIntegerSets  groups;
   IntegerSet         group;

   // the first group is the local one
   group.Recreate(1, &MyRank);
   groups.Insert(group);

#ifdef MFEM_DEBUG
   if (Dim < 3 && mesh.GetNFaces() != 0)
   {
      cerr << "ParMesh::ParMesh (proc " << MyRank << ") : "
           "(Dim < 3 && mesh.GetNFaces() != 0) is true!" << endl;
      mfem_error();
   }
#endif
   // determine shared faces
   int sface_counter = 0;
   Array<int> face_group(mesh.GetNFaces());
   for (i = 0; i < face_group.Size(); i++)
   {
      int el[2];
      face_group[i] = -1;
      mesh.GetFaceElements(i, &el[0], &el[1]);
      if (el[1] >= 0)
      {
         el[0] = partitioning[el[0]];
         el[1] = partitioning[el[1]];
         if ((el[0] == MyRank && el[1] != MyRank) ||
             (el[0] != MyRank && el[1] == MyRank))
         {
            group.Recreate(2, el);
            face_group[i] = groups.Insert(group) - 1;
            sface_counter++;
         }
      }
   }

   /*
   for (int i = 0; i < NRanks; ++i)
   {
       if (MyRank == i)
       {
           std::cout << "I am " << MyRank << "\n";
           std::cout << "face_group.Size() = " << face_group.Size() << "\n";
           for (int j = 0; j < face_group.Size(); j++)
           {
              int el[2];
              mesh.GetFaceElements(j, &el[0], &el[1]);

              if (el[1] >= 0)
              {
                 el[0] = partitioning[el[0]];
                 el[1] = partitioning[el[1]];
                 if ((el[0] == MyRank && el[1] != MyRank) ||
                     (el[0] != MyRank && el[1] == MyRank))
                 {
                     std::cout << "a shared face found, j = " << j << "\n";
                     const Element * face = mesh.GetFace(j);
                     const int * vertices = face->GetVertices();
                     int nv = face->GetNVertices();
                     std::cout << "its vertices: \n";
                     for (int vind = 0; vind < nv; ++vind)
                     {
                         double * vcoords = mesh.GetVertex(vertices[vind]);
                         for (int i = 0; i < mesh.Dimension(); ++i)
                             std::cout << vcoords[i] << " ";
                         std::cout << "\n";
                     }
                     std::cout << "\n";
                 }
              }
           }

           //face_group.Print();
           std::cout << "the end \n" << std::flush;
       }
       MPI_Barrier(comm);
   } // end fo loop over all processors, one after another
   std::cout << "Continuing \n" << std::flush;
   */

   // determine shared planars
   Table *plan_element = NULL;
   int splan_counter = 0;
   if (Dim==4)
   {
      plan_element = new Table;
      Transpose(mesh.ElementToPlanTable(), *plan_element, mesh.GetNPlanars());

      for (i = 0; i < plan_element->Size(); i++)
      {
         int me = 0, others = 0;
         for (j = plan_element->GetI()[i]; j < plan_element->GetI()[i+1]; j++)
         {
            plan_element->GetJ()[j] = partitioning[plan_element->GetJ()[j]];
            if (plan_element->GetJ()[j] == MyRank)
            {
               me = 1;
            }
            else
            {
               others = 1;
            }
         }

         if (me && others)
         {
            splan_counter++;
            group.Recreate(plan_element->RowSize(i), plan_element->GetRow(i));
            plan_element->GetRow(i)[0] = groups.Insert(group) - 1;
         }
         else
         {
            plan_element->GetRow(i)[0] = -1;
         }
      }
   }
   //   cout << "shared planars: " << splan_counter << endl;

   // determine shared edges
   int sedge_counter = 0;
   if (!edge_element)
   {
      edge_element = new Table;
      if (Dim == 1)
      {
         edge_element->SetDims(0,0);
      }
      else
      {
         Transpose(mesh.ElementToEdgeTable(), *edge_element, mesh.GetNEdges());
      }
   }
   for (i = 0; i < edge_element->Size(); i++)
   {
      int me = 0, others = 0;
      for (j = edge_element->GetI()[i]; j < edge_element->GetI()[i+1]; j++)
      {
         edge_element->GetJ()[j] = partitioning[edge_element->GetJ()[j]];
         if (edge_element->GetJ()[j] == MyRank)
         {
            me = 1;
         }
         else
         {
            others = 1;
         }
      }

      if (me && others)
      {
         sedge_counter++;
         group.Recreate(edge_element->RowSize(i), edge_element->GetRow(i));
         edge_element->GetRow(i)[0] = groups.Insert(group) - 1;
      }
      else
      {
         edge_element->GetRow(i)[0] = -1;
      }
   }

   // determine shared vertices
   int svert_counter = 0;
   Table *vert_element = mesh.GetVertexToElementTable(); // we must delete this

   for (i = 0; i < vert_element->Size(); i++)
   {
      int me = 0, others = 0;
      for (j = vert_element->GetI()[i]; j < vert_element->GetI()[i+1]; j++)
      {
         vert_element->GetJ()[j] = partitioning[vert_element->GetJ()[j]];
         if (vert_element->GetJ()[j] == MyRank)
         {
            me = 1;
         }
         else
         {
            others = 1;
         }
      }

      if (me && others)
      {
         svert_counter++;
         group.Recreate(vert_element->RowSize(i), vert_element->GetRow(i));
         vert_element->GetI()[i] = groups.Insert(group) - 1;
      }
      else
      {
         vert_element->GetI()[i] = -1;
      }
   }

   // build group_sface
   group_sface.MakeI(groups.Size()-1);

   for (i = 0; i < face_group.Size(); i++)
   {
      if (face_group[i] >= 0)
      {
         group_sface.AddAColumnInRow(face_group[i]);
      }
   }

   group_sface.MakeJ();

   sface_counter = 0;
   for (i = 0; i < face_group.Size(); i++)
   {
      if (face_group[i] >= 0)
      {
         group_sface.AddConnection(face_group[i], sface_counter++);
      }
   }

   group_sface.ShiftUpI();

   //build group_splan
   if (Dim==4)
   {
      group_splan.MakeI(groups.Size()-1);

      for (i = 0; i < plan_element->Size(); i++)
         if (plan_element->GetRow(i)[0] >= 0)
         {
            group_splan.AddAColumnInRow(plan_element->GetRow(i)[0]);
         }

      group_splan.MakeJ();

      splan_counter = 0;
      for (i = 0; i < plan_element->Size(); i++)
         if (plan_element->GetRow(i)[0] >= 0)
            group_splan.AddConnection(plan_element->GetRow(i)[0],
                                      splan_counter++);

      group_splan.ShiftUpI();
   }

   // build group_sedge
   group_sedge.MakeI(groups.Size()-1);

   for (i = 0; i < edge_element->Size(); i++)
   {
      if (edge_element->GetRow(i)[0] >= 0)
      {
         group_sedge.AddAColumnInRow(edge_element->GetRow(i)[0]);
      }
   }

   group_sedge.MakeJ();

   sedge_counter = 0;
   for (i = 0; i < edge_element->Size(); i++)
   {
      if (edge_element->GetRow(i)[0] >= 0)
      {
         group_sedge.AddConnection(edge_element->GetRow(i)[0], sedge_counter++);
      }
   }

   group_sedge.ShiftUpI();

   // build group_svert
   group_svert.MakeI(groups.Size()-1);

   for (i = 0; i < vert_element->Size(); i++)
   {
      if (vert_element->GetI()[i] >= 0)
      {
         group_svert.AddAColumnInRow(vert_element->GetI()[i]);
      }
   }

   group_svert.MakeJ();

   svert_counter = 0;
   for (i = 0; i < vert_element->Size(); i++)
   {
      if (vert_element->GetI()[i] >= 0)
      {
         group_svert.AddConnection(vert_element->GetI()[i], svert_counter++);
      }
   }

   group_svert.ShiftUpI();

   // build shared_faces and sface_lface
   shared_faces.SetSize(sface_counter);
   sface_lface. SetSize(sface_counter);

   if ( Dim == 4)
   {
      sface_counter = 0;
      for (i = 0; i < face_group.Size(); i++)
         if (face_group[i] >= 0)
         {
            shared_faces[sface_counter] = mesh.GetFace(i)->Duplicate(this);
            int *v = shared_faces[sface_counter]->GetVertices();
            int nv = shared_faces[sface_counter]->GetNVertices();
            for (j = 0; j < nv; j++) { v[j] = vert_global_local[v[j]]; }

            switch (shared_faces[sface_counter]->GetType())
            {
               case Element::TETRAHEDRON:
               {
                  sface_lface[sface_counter] = (*faces_tbl_4d)(v[0], v[1], v[2], v[3]);

                  // flip the shared face info in the processor that owns the
                  // second element (in 'mesh')
                  {
                     int gl_el1, gl_el2;
                     mesh.GetFaceElements(i, &gl_el1, &gl_el2);

                     if (mesh.getSwappedFaceElementInfo(i)) { Swap(v); }

                     if (MyRank == partitioning[gl_el2])
                     {
                        //                   faces_info[sface_lface[sface_counter]].Elem1Inf += 1;
                        //                   Swap(v);
                     }
                  }
               }
            }

            sface_counter++;
         }

      delete faces_tbl_4d;
   }

   if (Dim == 3)
   {
      sface_counter = 0;
      for (i = 0; i < face_group.Size(); i++)
      {
         if (face_group[i] >= 0)
         {
            shared_faces[sface_counter] = mesh.GetFace(i)->Duplicate(this);
            int *v = shared_faces[sface_counter]->GetVertices();
            int nv = shared_faces[sface_counter]->GetNVertices();
            for (j = 0; j < nv; j++)
            {
               v[j] = vert_global_local[v[j]];
            }
            switch (shared_faces[sface_counter]->GetType())
            {
               case Element::TRIANGLE:
                  sface_lface[sface_counter] = (*faces_tbl)(v[0], v[1], v[2]);
                  // mark the shared face for refinement by reorienting
                  // it according to the refinement flag in the tetrahedron
                  // to which this shared face belongs to.
                  {
                     int lface = sface_lface[sface_counter];
                     Tetrahedron *tet =
                        (Tetrahedron *)(elements[faces_info[lface].Elem1No]);
                     tet->GetMarkedFace(faces_info[lface].Elem1Inf/64, v);
                     // flip the shared face in the processor that owns the
                     // second element (in 'mesh')
                     {
                        int gl_el1, gl_el2;
                        mesh.GetFaceElements(i, &gl_el1, &gl_el2);
                        if (MyRank == partitioning[gl_el2])
                        {
                           std::swap(v[0], v[1]);
                        }
                     }
                  }
                  break;
               case Element::QUADRILATERAL:
                  sface_lface[sface_counter] =
                     (*faces_tbl)(v[0], v[1], v[2], v[3]);
                  break;
            }
            sface_counter++;
         }
      }

      delete faces_tbl;
   }

   // build shared_edges and sedge_ledge
   shared_edges.SetSize(sedge_counter);
   sedge_ledge. SetSize(sedge_counter);

   shared_planars.SetSize(splan_counter);
   splan_lplan.SetSize(splan_counter);

   {
      DSTable v_to_v(NumOfVertices);
      GetVertexToVertexTable(v_to_v);

      sedge_counter = 0;
      for (i = 0; i < edge_element->Size(); i++)
      {
         if (edge_element->GetRow(i)[0] >= 0)
         {
            mesh.GetEdgeVertices(i, vert);

            shared_edges[sedge_counter] =
               new Segment(vert_global_local[vert[0]],
                           vert_global_local[vert[1]], 1);

            if ((sedge_ledge[sedge_counter] =
                    v_to_v(vert_global_local[vert[0]],
                           vert_global_local[vert[1]])) < 0)
            {
               cerr << "\n\n\n" << MyRank << ": ParMesh::ParMesh: "
                    << "ERROR in v_to_v\n\n" << endl;
               mfem_error();
            }

            sedge_counter++;
         }
      }

      if (Dim==4)
      {
         splan_counter = 0;
         for (i = 0; i < plan_element->Size(); i++)
            if (plan_element->GetRow(i)[0] >= 0)
            {
               mesh.GetPlanVertices(i, vert);

               shared_planars[splan_counter] = new Triangle(vert_global_local[vert[0]],
                                                            vert_global_local[vert[1]],vert_global_local[vert[2]], 1);
               splan_lplan[splan_counter] = (*planar_tbl)(vert_global_local[vert[0]],
                                                          vert_global_local[vert[1]],vert_global_local[vert[2]]);

               splan_counter++;
            }

         delete planar_tbl;
      }
   }

   delete edge_element;
   if (Dim==4)
   {
      delete plan_element;
   }

   // build svert_lvert
   svert_lvert.SetSize(svert_counter);

   svert_counter = 0;
   for (i = 0; i < vert_element->Size(); i++)
   {
      if (vert_element->GetI()[i] >= 0)
      {
         svert_lvert[svert_counter++] = vert_global_local[i];
      }
   }

   delete vert_element;

   // build the group communication topology
   gtopo.Create(groups, 822);

   if (mesh.NURBSext)
   {
      NURBSext = new ParNURBSExtension(comm, mesh.NURBSext, partitioning,
                                       activeBdrElem);
   }

   if (mesh.GetNodes()) // curved mesh
   {
      Nodes = new ParGridFunction(this, mesh.GetNodes());
      own_nodes = 1;

      Array<int> gvdofs, lvdofs;
      Vector lnodes;
      element_counter = 0;
      for (i = 0; i < mesh.GetNE(); i++)
         if (partitioning[i] == MyRank)
         {
            Nodes->FESpace()->GetElementVDofs(element_counter, lvdofs);
            mesh.GetNodes()->FESpace()->GetElementVDofs(i, gvdofs);
            mesh.GetNodes()->GetSubVector(gvdofs, lnodes);
            Nodes->SetSubVector(lvdofs, lnodes);
            element_counter++;
         }
   }

   if (partitioning_ == NULL)
   {
      delete [] partitioning;
   }

   have_face_nbr_data = false;
}

// protected method
ParMesh::ParMesh(const ParNCMesh &pncmesh)
   : MyComm(pncmesh.MyComm)
   , NRanks(pncmesh.NRanks)
   , MyRank(pncmesh.MyRank)
   , gtopo(MyComm)
   , pncmesh(NULL)
{
   Mesh::InitFromNCMesh(pncmesh);
   have_face_nbr_data = false;
}

ParMesh::ParMesh(MPI_Comm comm, istream &input)
   : gtopo(comm)
{
   MyComm = comm;
   MPI_Comm_size(MyComm, &NRanks);
   MPI_Comm_rank(MyComm, &MyRank);

   have_face_nbr_data = false;
   pncmesh = NULL;

   string ident;

   // read the serial part of the mesh
   int gen_edges = 1;

   // Tell Loader() to read up to 'mfem_serial_mesh_end' instead of
   // 'mfem_mesh_end', as we have additional parallel mesh data to load in from
   // the stream.
   Loader(input, gen_edges, "mfem_serial_mesh_end");

   skip_comment_lines(input, '#');

   // read the group topology
   input >> ident;
   MFEM_VERIFY(ident == "communication_groups",
               "input stream is not a parallel MFEM mesh");
   gtopo.Load(input);

   skip_comment_lines(input, '#');

   DSTable *v_to_v = NULL;
   STable3D *faces_tbl = NULL;
   // read and set the sizes of svert_lvert, group_svert
   {
      int num_sverts;
      input >> ident >> num_sverts; // total_shared_vertices
      svert_lvert.SetSize(num_sverts);
      group_svert.SetDims(GetNGroups()-1, num_sverts);
   }
   // read and set the sizes of sedge_ledge, group_sedge
   if (Dim >= 2)
   {
      skip_comment_lines(input, '#');
      int num_sedges;
      input >> ident >> num_sedges; // total_shared_edges
      sedge_ledge.SetSize(num_sedges);
      shared_edges.SetSize(num_sedges);
      group_sedge.SetDims(GetNGroups()-1, num_sedges);
      v_to_v = new DSTable(NumOfVertices);
      GetVertexToVertexTable(*v_to_v);
   }
   else
   {
      group_sedge.SetSize(GetNGroups()-1, 0);   // create empty group_sedge
   }
   // read and set the sizes of sface_lface, group_sface
   if (Dim >= 3)
   {
      skip_comment_lines(input, '#');
      int num_sfaces;
      input >> ident >> num_sfaces; // total_shared_faces
      sface_lface.SetSize(num_sfaces);
      shared_faces.SetSize(num_sfaces);
      group_sface.SetDims(GetNGroups()-1, num_sfaces);
      faces_tbl = GetFacesTable();
   }
   else
   {
      group_sface.SetSize(GetNGroups()-1, 0);   // create empty group_sface
   }

   // read, group by group, the contents of group_svert, svert_lvert,
   // group_sedge, shared_edges, group_sface, shared_faces
   //
   // derive the contents of sedge_ledge, sface_lface
   int svert_counter = 0, sedge_counter = 0, sface_counter = 0;
   for (int gr = 1; gr < GetNGroups(); gr++)
   {
      int g;
      input >> ident >> g; // group
      if (g != gr)
      {
         cerr << "ParMesh::ParMesh : expecting group " << gr
              << ", read group " << g << endl;
         mfem_error();
      }

      {
         int nv;
         input >> ident >> nv; // shared_vertices (in this group)
         nv += svert_counter;
         MFEM_VERIFY(nv <= group_svert.Size_of_connections(),
                     "incorrect number of total_shared_vertices");
         group_svert.GetI()[gr] = nv;
         for ( ; svert_counter < nv; svert_counter++)
         {
            group_svert.GetJ()[svert_counter] = svert_counter;
            input >> svert_lvert[svert_counter];
         }
      }
      if (Dim >= 2)
      {
         int ne, v[2];
         input >> ident >> ne; // shared_edges (in this group)
         ne += sedge_counter;
         MFEM_VERIFY(ne <= group_sedge.Size_of_connections(),
                     "incorrect number of total_shared_edges");
         group_sedge.GetI()[gr] = ne;
         for ( ; sedge_counter < ne; sedge_counter++)
         {
            group_sedge.GetJ()[sedge_counter] = sedge_counter;
            input >> v[0] >> v[1];
            shared_edges[sedge_counter] = new Segment(v[0], v[1], 1);
            sedge_ledge[sedge_counter] = (*v_to_v)(v[0], v[1]);
         }
      }
      if (Dim >= 3)
      {
         int nf;
         input >> ident >> nf; // shared_faces (in this group)
         nf += sface_counter;
         MFEM_VERIFY(nf <= group_sface.Size_of_connections(),
                     "incorrect number of total_shared_faces");
         group_sface.GetI()[gr] = nf;
         for ( ; sface_counter < nf; sface_counter++)
         {
            group_sface.GetJ()[sface_counter] = sface_counter;
            Element *sface = ReadElementWithoutAttr(input);
            shared_faces[sface_counter] = sface;
            const int *v = sface->GetVertices();
            switch (sface->GetType())
            {
               case Element::TRIANGLE:
                  sface_lface[sface_counter] = (*faces_tbl)(v[0], v[1], v[2]);
                  break;
               case Element::QUADRILATERAL:
                  sface_lface[sface_counter] =
                     (*faces_tbl)(v[0], v[1], v[2], v[3]);
                  break;
            }
         }
      }
   }
   delete faces_tbl;
   delete v_to_v;

   const bool refine = true;
   const bool fix_orientation = false;
   Finalize(refine, fix_orientation);

   // If the mesh has Nodes, convert them from GridFunction to ParGridFunction?

   // note: attributes and bdr_attributes are local lists

   // TODO: AMR meshes, NURBS meshes?
}

ParMesh::ParMesh(ParMesh *orig_mesh, int ref_factor, int ref_type)
   : Mesh(orig_mesh, ref_factor, ref_type),
     MyComm(orig_mesh->GetComm()),
     NRanks(orig_mesh->GetNRanks()),
     MyRank(orig_mesh->GetMyRank()),
     gtopo(orig_mesh->gtopo),
     have_face_nbr_data(false),
     pncmesh(NULL)
{
   // Need to initialize:
   // - shared_edges, shared_faces
   // - group_svert, group_sedge, group_sface
   // - svert_lvert, sedge_ledge, sface_lface

   H1_FECollection rfec(ref_factor, Dim, ref_type);
   ParFiniteElementSpace rfes(orig_mesh, &rfec);

   // count the number of entries in each row of group_s{vert,edge,face}
   group_svert.MakeI(GetNGroups()-1); // exclude the local group 0
   group_sedge.MakeI(GetNGroups()-1);
   group_sface.MakeI(GetNGroups()-1);
   for (int gr = 1; gr < GetNGroups(); gr++)
   {
      // orig vertex -> vertex
      group_svert.AddColumnsInRow(gr-1, orig_mesh->GroupNVertices(gr));
      // orig edge -> (ref_factor-1) vertices and (ref_factor) edges
      const int orig_ne = orig_mesh->GroupNEdges(gr);
      group_svert.AddColumnsInRow(gr-1, (ref_factor-1)*orig_ne);
      group_sedge.AddColumnsInRow(gr-1, ref_factor*orig_ne);
      // orig face -> (?) vertices, (?) edges, and (?) faces
      const int  orig_nf = orig_mesh->GroupNFaces(gr);
      const int *orig_sf = orig_mesh->group_sface.GetRow(gr-1);
      for (int fi = 0; fi < orig_nf; fi++)
      {
         const int orig_l_face = orig_mesh->sface_lface[orig_sf[fi]];
         const int geom = orig_mesh->GetFaceBaseGeometry(orig_l_face);
         const int nvert = Geometry::NumVerts[geom];
         RefinedGeometry &RG =
            *GlobGeometryRefiner.Refine(geom, ref_factor, ref_factor);

         // count internal vertices
         group_svert.AddColumnsInRow(gr-1, rfec.DofForGeometry(geom));
         // count internal edges
         group_sedge.AddColumnsInRow(gr-1, RG.RefEdges.Size()/2-RG.NumBdrEdges);
         // count refined faces
         group_sface.AddColumnsInRow(gr-1, RG.RefGeoms.Size()/nvert);
      }
   }

   group_svert.MakeJ();
   svert_lvert.Reserve(group_svert.Size_of_connections());

   group_sedge.MakeJ();
   shared_edges.Reserve(group_sedge.Size_of_connections());
   sedge_ledge.SetSize(group_sedge.Size_of_connections());

   group_sface.MakeJ();
   shared_faces.Reserve(group_sface.Size_of_connections());
   sface_lface.SetSize(group_sface.Size_of_connections());

   Array<int> rdofs;
   for (int gr = 1; gr < GetNGroups(); gr++)
   {
      // add shared vertices from original shared vertices
      const int orig_n_verts = orig_mesh->GroupNVertices(gr);
      for (int j = 0; j < orig_n_verts; j++)
      {
         rfes.GetVertexDofs(orig_mesh->GroupVertex(gr, j), rdofs);
         group_svert.AddConnection(gr-1, svert_lvert.Append(rdofs[0])-1);
      }

      // add refined shared edges; add shared vertices from refined shared edges
      const int orig_n_edges = orig_mesh->GroupNEdges(gr);
      const int geom = Geometry::SEGMENT;
      const int nvert = Geometry::NumVerts[geom];
      RefinedGeometry &RG = *GlobGeometryRefiner.Refine(geom, ref_factor);
      for (int e = 0; e < orig_n_edges; e++)
      {
         rfes.GetSharedEdgeDofs(gr, e, rdofs);
         MFEM_ASSERT(rdofs.Size() == RG.RefPts.Size(), "");
         // add the internal edge 'rdofs' as shared vertices
         for (int j = 2; j < rdofs.Size(); j++)
         {
            group_svert.AddConnection(gr-1, svert_lvert.Append(rdofs[j])-1);
         }
         const int *c2h_map = rfec.GetDofMap(geom);
         for (int j = 0; j < RG.RefGeoms.Size(); j += nvert)
         {
            Element *elem = NewElement(geom);
            int *v = elem->GetVertices();
            for (int k = 0; k < nvert; k++)
            {
               int cid = RG.RefGeoms[j+k]; // local Cartesian index
               v[k] = rdofs[c2h_map[cid]];
            }
            group_sedge.AddConnection(gr-1, shared_edges.Append(elem)-1);
         }
      }
      // add refined shared faces; add shared edges and shared vertices from
      // refined shared faces
      const int  orig_nf = orig_mesh->group_sface.RowSize(gr-1);
      const int *orig_sf = orig_mesh->group_sface.GetRow(gr-1);
      for (int f = 0; f < orig_nf; f++)
      {
         const int orig_l_face = orig_mesh->sface_lface[orig_sf[f]];
         const int geom = orig_mesh->GetFaceBaseGeometry(orig_l_face);
         const int nvert = Geometry::NumVerts[geom];
         RefinedGeometry &RG =
            *GlobGeometryRefiner.Refine(geom, ref_factor, ref_factor);

         rfes.GetSharedFaceDofs(gr, f, rdofs);
         MFEM_ASSERT(rdofs.Size() == RG.RefPts.Size(), "");
         // add the internal face 'rdofs' as shared vertices
         const int num_int_verts = rfec.DofForGeometry(geom);
         for (int j = rdofs.Size()-num_int_verts; j < rdofs.Size(); j++)
         {
            group_svert.AddConnection(gr-1, svert_lvert.Append(rdofs[j])-1);
         }
         const int *c2h_map = rfec.GetDofMap(geom);
         // add the internal (for the shared face) edges as shared edges
         for (int j = 2*RG.NumBdrEdges; j < RG.RefEdges.Size(); j += 2)
         {
            Element *elem = NewElement(Geometry::SEGMENT);
            int *v = elem->GetVertices();
            for (int k = 0; k < 2; k++)
            {
               v[k] = rdofs[c2h_map[RG.RefEdges[j+k]]];
            }
            group_sedge.AddConnection(gr-1, shared_edges.Append(elem)-1);
         }
         // add refined shared faces
         for (int j = 0; j < RG.RefGeoms.Size(); j += nvert)
         {
            Element *elem = NewElement(geom);
            int *v = elem->GetVertices();
            for (int k = 0; k < nvert; k++)
            {
               int cid = RG.RefGeoms[j+k]; // local Cartesian index
               v[k] = rdofs[c2h_map[cid]];
            }
            group_sface.AddConnection(gr-1, shared_faces.Append(elem)-1);
         }
      }
   }
   group_svert.ShiftUpI();
   group_sedge.ShiftUpI();
   group_sface.ShiftUpI();

   // determine sedge_ledge
   if (shared_edges.Size() > 0)
   {
      DSTable v_to_v(NumOfVertices);
      GetVertexToVertexTable(v_to_v);
      for (int se = 0; se < shared_edges.Size(); se++)
      {
         const int *v = shared_edges[se]->GetVertices();
         const int l_edge = v_to_v(v[0], v[1]);
         MFEM_ASSERT(l_edge >= 0, "invalid shared edge");
         sedge_ledge[se] = l_edge;
      }
   }

   // determine sface_lface
   if (shared_faces.Size() > 0)
   {
      STable3D *faces_tbl = GetFacesTable();
      for (int sf = 0; sf < shared_faces.Size(); sf++)
      {
         int l_face;
         const int *v = shared_faces[sf]->GetVertices();
         switch (shared_faces[sf]->GetGeometryType())
         {
            case Geometry::TRIANGLE:
               l_face = (*faces_tbl)(v[0], v[1], v[2]);
               break;
            case Geometry::SQUARE:
               l_face = (*faces_tbl)(v[0], v[1], v[2], v[3]);
               break;
            default:
               MFEM_ABORT("invalid face geometry");
               l_face = -1;
         }
         MFEM_ASSERT(l_face >= 0, "invalid shared face");
         sface_lface[sf] = l_face;
      }
      delete faces_tbl;
   }
}

void ParMesh::GroupEdge(int group, int i, int &edge, int &o)
{
   int sedge = group_sedge.GetRow(group-1)[i];
   edge = sedge_ledge[sedge];
   int *v = shared_edges[sedge]->GetVertices();
   o = (v[0] < v[1]) ? (+1) : (-1);
}

void ParMesh::GroupPlanar(int group, int i, int &planar, int &o)
{
   int splan = group_splan.GetJ()[group_splan.GetI()[group-1]+i];
   planar = splan_lplan[splan];
   // face gives the base orientation
   if (planars[planar]->GetType() == Element::TRIANGLE)
      o = GetTriOrientation(planars[planar]->GetVertices(),
                            shared_planars[splan]->GetVertices());
   if (planars[planar]->GetType() == Element::QUADRILATERAL)
      o = GetQuadOrientation(planars[planar]->GetVertices(),
                             shared_planars[splan]->GetVertices());
}

void ParMesh::GroupFace(int group, int i, int &face, int &o)
{
   int sface = group_sface.GetRow(group-1)[i];
   face = sface_lface[sface];
   // face gives the base orientation
   if (faces[face]->GetType() == Element::TRIANGLE)
   {
      o = GetTriOrientation(faces[face]->GetVertices(),
                            shared_faces[sface]->GetVertices());
   }
   else if (faces[face]->GetType() == Element::QUADRILATERAL)
   {
      o = GetQuadOrientation(faces[face]->GetVertices(),
                             shared_faces[sface]->GetVertices());
   }
   else if (faces[face]->GetType() == Element::TETRAHEDRON)
      o = GetTetOrientation(faces[face]->GetVertices(),
                            shared_faces[sface]->GetVertices());
}

void ParMesh::MarkTetMeshForRefinement(DSTable &v_to_v)
{
   Array<int> order;
   GetEdgeOrdering(v_to_v, order); // local edge ordering

   // create a GroupCommunicator on the shared edges
   GroupCommunicator sedge_comm(gtopo);
   {
      // initialize sedge_comm
      Table &gr_sedge = sedge_comm.GroupLDofTable(); // differs from group_sedge
      gr_sedge.SetDims(GetNGroups(), shared_edges.Size());
      gr_sedge.GetI()[0] = 0;
      for (int gr = 1; gr <= GetNGroups(); gr++)
      {
         gr_sedge.GetI()[gr] = group_sedge.GetI()[gr-1];
      }
      for (int k = 0; k < shared_edges.Size(); k++)
      {
         gr_sedge.GetJ()[k] = group_sedge.GetJ()[k];
      }
      sedge_comm.Finalize();
   }

   Array<int> sedge_ord(shared_edges.Size());
   Array<Pair<int,int> > sedge_ord_map(shared_edges.Size());
   for (int k = 0; k < shared_edges.Size(); k++)
   {
      sedge_ord[k] = order[sedge_ledge[group_sedge.GetJ()[k]]];
   }

   sedge_comm.Bcast<int>(sedge_ord, 1);

   for (int k = 0, gr = 1; gr < GetNGroups(); gr++)
   {
      const int n = group_sedge.RowSize(gr-1);
      if (n == 0) { continue; }
      sedge_ord_map.SetSize(n);
      for (int j = 0; j < n; j++)
      {
         sedge_ord_map[j].one = sedge_ord[k+j];
         sedge_ord_map[j].two = j;
      }
      SortPairs<int, int>(sedge_ord_map, n);
      for (int j = 0; j < n; j++)
      {
         int sedge_from = group_sedge.GetJ()[k+j];
         sedge_ord[k+j] = order[sedge_ledge[sedge_from]];
      }
      std::sort(&sedge_ord[k], &sedge_ord[k] + n);
      for (int j = 0; j < n; j++)
      {
         int sedge_to = group_sedge.GetJ()[k+sedge_ord_map[j].two];
         order[sedge_ledge[sedge_to]] = sedge_ord[k+j];
      }
      k += n;
   }

#ifdef MFEM_DEBUG
   {
      Array<Pair<int, double> > ilen_len(order.Size());

      for (int i = 0; i < NumOfVertices; i++)
      {
         for (DSTable::RowIterator it(v_to_v, i); !it; ++it)
         {
            int j = it.Index();
            ilen_len[j].one = order[j];
            ilen_len[j].two = GetLength(i, it.Column());
         }
      }

      SortPairs<int, double>(ilen_len, order.Size());

      double d_max = 0.;
      for (int i = 1; i < order.Size(); i++)
      {
         d_max = std::max(d_max, ilen_len[i-1].two-ilen_len[i].two);
      }

#if 0
      // Debug message from every MPI rank.
      cout << "proc. " << MyRank << '/' << NRanks << ": d_max = " << d_max
           << endl;
#else
      // Debug message just from rank 0.
      double glob_d_max;
      MPI_Reduce(&d_max, &glob_d_max, 1, MPI_DOUBLE, MPI_MAX, 0, MyComm);
      if (MyRank == 0)
      {
         cout << "glob_d_max = " << glob_d_max << endl;
      }
#endif
   }
#endif

   // use 'order' to mark the tets, the boundary triangles, and the shared
   // triangle faces
   for (int i = 0; i < NumOfElements; i++)
   {
      if (elements[i]->GetType() == Element::TETRAHEDRON)
      {
         elements[i]->MarkEdge(v_to_v, order);
      }
   }

   for (int i = 0; i < NumOfBdrElements; i++)
   {
      if (boundary[i]->GetType() == Element::TRIANGLE)
      {
         boundary[i]->MarkEdge(v_to_v, order);
      }
   }

   for (int i = 0; i < shared_faces.Size(); i++)
   {
      if (shared_faces[i]->GetType() == Element::TRIANGLE)
      {
         shared_faces[i]->MarkEdge(v_to_v, order);
      }
   }
}

// For a line segment with vertices v[0] and v[1], return a number with
// the following meaning:
// 0 - the edge was not refined
// 1 - the edge e was refined once by splitting v[0],v[1]
int ParMesh::GetEdgeSplittings(Element *edge, const DSTable &v_to_v,
                               int *middle)
{
   int m, *v = edge->GetVertices();

   if ((m = v_to_v(v[0], v[1])) != -1 && middle[m] != -1)
   {
      return 1;
   }
   else
   {
      return 0;
   }
}

// For a triangular face with (correctly ordered) vertices v[0], v[1], v[2]
// return a number with the following meaning:
// 0 - the face was not refined
// 1 - the face was refined once by splitting v[0],v[1]
// 2 - the face was refined twice by splitting v[0],v[1] and then v[1],v[2]
// 3 - the face was refined twice by splitting v[0],v[1] and then v[0],v[2]
// 4 - the face was refined three times (as in 2+3)
int ParMesh::GetFaceSplittings(Element *face, const DSTable &v_to_v,
                               int *middle)
{
   int m, right = 0;
   int number_of_splittings = 0;
   int *v = face->GetVertices();

   if ((m = v_to_v(v[0], v[1])) != -1 && middle[m] != -1)
   {
      number_of_splittings++;
      if ((m = v_to_v(v[1], v[2])) != -1 && middle[m] != -1)
      {
         right = 1;
         number_of_splittings++;
      }
      if ((m = v_to_v(v[2], v[0])) != -1 && middle[m] != -1)
      {
         number_of_splittings++;
      }

      switch (number_of_splittings)
      {
         case 2:
            if (right == 0)
            {
               number_of_splittings++;
            }
            break;
         case 3:
            number_of_splittings++;
            break;
      }
   }

   return number_of_splittings;
}

void ParMesh::GenerateOffsets(int N, HYPRE_Int loc_sizes[],
                              Array<HYPRE_Int> *offsets[]) const
{
   if (HYPRE_AssumedPartitionCheck())
   {
      Array<HYPRE_Int> temp(N);
      MPI_Scan(loc_sizes, temp.GetData(), N, HYPRE_MPI_INT, MPI_SUM, MyComm);
      for (int i = 0; i < N; i++)
      {
         offsets[i]->SetSize(3);
         (*offsets[i])[0] = temp[i] - loc_sizes[i];
         (*offsets[i])[1] = temp[i];
      }
      MPI_Bcast(temp.GetData(), N, HYPRE_MPI_INT, NRanks-1, MyComm);
      for (int i = 0; i < N; i++)
      {
         (*offsets[i])[2] = temp[i];
         // check for overflow
         MFEM_VERIFY((*offsets[i])[0] >= 0 && (*offsets[i])[1] >= 0,
                     "overflow in offsets");
      }
   }
   else
   {
      Array<HYPRE_Int> temp(N*NRanks);
      MPI_Allgather(loc_sizes, N, HYPRE_MPI_INT, temp.GetData(), N,
                    HYPRE_MPI_INT, MyComm);
      for (int i = 0; i < N; i++)
      {
         Array<HYPRE_Int> &offs = *offsets[i];
         offs.SetSize(NRanks+1);
         offs[0] = 0;
         for (int j = 0; j < NRanks; j++)
         {
            offs[j+1] = offs[j] + temp[i+N*j];
         }
         // Check for overflow
         MFEM_VERIFY(offs[MyRank] >= 0 && offs[MyRank+1] >= 0,
                     "overflow in offsets");
      }
   }
}

void ParMesh::GetFaceNbrElementTransformation(
   int i, IsoparametricTransformation *ElTr)
{
   DenseMatrix &pointmat = ElTr->GetPointMat();
   Element *elem = face_nbr_elements[i];

   ElTr->Attribute = elem->GetAttribute();
   ElTr->ElementNo = NumOfElements + i;

   if (Nodes == NULL)
   {
      const int nv = elem->GetNVertices();
      const int *v = elem->GetVertices();

      pointmat.SetSize(spaceDim, nv);
      for (int k = 0; k < spaceDim; k++)
      {
         for (int j = 0; j < nv; j++)
         {
            pointmat(k, j) = face_nbr_vertices[v[j]](k);
         }
      }

      ElTr->SetFE(GetTransformationFEforElementType(elem->GetType()));
   }
   else
   {
      Array<int> vdofs;
      ParGridFunction *pNodes = dynamic_cast<ParGridFunction *>(Nodes);
      if (pNodes)
      {
         pNodes->ParFESpace()->GetFaceNbrElementVDofs(i, vdofs);
         int n = vdofs.Size()/spaceDim;
         pointmat.SetSize(spaceDim, n);
         for (int k = 0; k < spaceDim; k++)
         {
            for (int j = 0; j < n; j++)
            {
               pointmat(k,j) = (pNodes->FaceNbrData())(vdofs[n*k+j]);
            }
         }

         ElTr->SetFE(pNodes->ParFESpace()->GetFaceNbrFE(i));
      }
      else
      {
         MFEM_ABORT("Nodes are not ParGridFunction!");
      }
   }
}

void ParMesh::DeleteFaceNbrData()
{
   if (!have_face_nbr_data)
   {
      return;
   }

   have_face_nbr_data = false;
   face_nbr_group.DeleteAll();
   face_nbr_elements_offset.DeleteAll();
   face_nbr_vertices_offset.DeleteAll();
   for (int i = 0; i < face_nbr_elements.Size(); i++)
   {
      FreeElement(face_nbr_elements[i]);
   }
   face_nbr_elements.DeleteAll();
   face_nbr_vertices.DeleteAll();
   send_face_nbr_elements.Clear();
   send_face_nbr_vertices.Clear();
}

void ParMesh::ExchangeFaceNbrData()
{
   if (have_face_nbr_data)
   {
      return;
   }

   if (Nonconforming())
   {
      // with ParNCMesh we can set up face neighbors without communication
      pncmesh->GetFaceNeighbors(*this);
      have_face_nbr_data = true;

      ExchangeFaceNbrNodes();
      return;
   }

   Table *gr_sface;
   int   *s2l_face;
   if (Dim == 1)
   {
      gr_sface = &group_svert;
      s2l_face = svert_lvert;
   }
   else if (Dim == 2)
   {
      gr_sface = &group_sedge;
      s2l_face = sedge_ledge;
   }
   else
   {
      gr_sface = &group_sface;
      s2l_face = sface_lface;
   }

   int num_face_nbrs = 0;
   for (int g = 1; g < GetNGroups(); g++)
      if (gr_sface->RowSize(g-1) > 0)
      {
         num_face_nbrs++;
      }

   face_nbr_group.SetSize(num_face_nbrs);

   if (num_face_nbrs == 0)
   {
      have_face_nbr_data = true;
      return;
   }

   {
      // sort face-neighbors by processor rank
      Array<Pair<int, int> > rank_group(num_face_nbrs);

      for (int g = 1, counter = 0; g < GetNGroups(); g++)
         if (gr_sface->RowSize(g-1) > 0)
         {
#ifdef MFEM_DEBUG
            if (gtopo.GetGroupSize(g) != 2)
               mfem_error("ParMesh::ExchangeFaceNbrData() : "
                          "group size is not 2!");
#endif
            const int *nbs = gtopo.GetGroup(g);
            int lproc = (nbs[0]) ? nbs[0] : nbs[1];
            rank_group[counter].one = gtopo.GetNeighborRank(lproc);
            rank_group[counter].two = g;
            counter++;
         }

      SortPairs<int, int>(rank_group, rank_group.Size());

      for (int fn = 0; fn < num_face_nbrs; fn++)
      {
         face_nbr_group[fn] = rank_group[fn].two;
      }
   }

   MPI_Request *requests = new MPI_Request[2*num_face_nbrs];
   MPI_Request *send_requests = requests;
   MPI_Request *recv_requests = requests + num_face_nbrs;
   MPI_Status  *statuses = new MPI_Status[num_face_nbrs];

   int *nbr_data = new int[6*num_face_nbrs];
   int *nbr_send_data = nbr_data;
   int *nbr_recv_data = nbr_data + 3*num_face_nbrs;

   Array<int> el_marker(GetNE());
   Array<int> vertex_marker(GetNV());
   el_marker = -1;
   vertex_marker = -1;

   Table send_face_nbr_elemdata, send_face_nbr_facedata;

   send_face_nbr_elements.MakeI(num_face_nbrs);
   send_face_nbr_vertices.MakeI(num_face_nbrs);
   send_face_nbr_elemdata.MakeI(num_face_nbrs);
   send_face_nbr_facedata.MakeI(num_face_nbrs);
   for (int fn = 0; fn < num_face_nbrs; fn++)
   {
      int nbr_group = face_nbr_group[fn];
      int  num_sfaces = gr_sface->RowSize(nbr_group-1);
      int *sface = gr_sface->GetRow(nbr_group-1);
      for (int i = 0; i < num_sfaces; i++)
      {
         int lface = s2l_face[sface[i]];
         int el = faces_info[lface].Elem1No;
         if (el_marker[el] != fn)
         {
            el_marker[el] = fn;
            send_face_nbr_elements.AddAColumnInRow(fn);

            const int nv = elements[el]->GetNVertices();
            const int *v = elements[el]->GetVertices();
            for (int j = 0; j < nv; j++)
               if (vertex_marker[v[j]] != fn)
               {
                  vertex_marker[v[j]] = fn;
                  send_face_nbr_vertices.AddAColumnInRow(fn);
               }

            send_face_nbr_elemdata.AddColumnsInRow(fn, nv + 2);
         }
      }
      send_face_nbr_facedata.AddColumnsInRow(fn, 2*num_sfaces);

      nbr_send_data[3*fn  ] = send_face_nbr_elements.GetI()[fn];
      nbr_send_data[3*fn+1] = send_face_nbr_vertices.GetI()[fn];
      nbr_send_data[3*fn+2] = send_face_nbr_elemdata.GetI()[fn];

      int nbr_rank = GetFaceNbrRank(fn);
      int tag = 0;

      MPI_Isend(&nbr_send_data[3*fn], 3, MPI_INT, nbr_rank, tag, MyComm,
                &send_requests[fn]);
      MPI_Irecv(&nbr_recv_data[3*fn], 3, MPI_INT, nbr_rank, tag, MyComm,
                &recv_requests[fn]);
   }
   send_face_nbr_elements.MakeJ();
   send_face_nbr_vertices.MakeJ();
   send_face_nbr_elemdata.MakeJ();
   send_face_nbr_facedata.MakeJ();
   el_marker = -1;
   vertex_marker = -1;
   for (int fn = 0; fn < num_face_nbrs; fn++)
   {
      int nbr_group = face_nbr_group[fn];
      int  num_sfaces = gr_sface->RowSize(nbr_group-1);
      int *sface = gr_sface->GetRow(nbr_group-1);
      for (int i = 0; i < num_sfaces; i++)
      {
         int lface = s2l_face[sface[i]];
         int el = faces_info[lface].Elem1No;
         if (el_marker[el] != fn)
         {
            el_marker[el] = fn;
            send_face_nbr_elements.AddConnection(fn, el);

            const int nv = elements[el]->GetNVertices();
            const int *v = elements[el]->GetVertices();
            for (int j = 0; j < nv; j++)
               if (vertex_marker[v[j]] != fn)
               {
                  vertex_marker[v[j]] = fn;
                  send_face_nbr_vertices.AddConnection(fn, v[j]);
               }

            send_face_nbr_elemdata.AddConnection(fn, GetAttribute(el));
            send_face_nbr_elemdata.AddConnection(
               fn, GetElementBaseGeometry(el));
            send_face_nbr_elemdata.AddConnections(fn, v, nv);
         }
         send_face_nbr_facedata.AddConnection(fn, el);
         int info = faces_info[lface].Elem1Inf;
         // change the orientation in info to be relative to the shared face
         //   in 1D and 2D keep the orientation equal to 0
         if (Dim == 3)
         {
            Element *lf = faces[lface];
            const int *sf_v = shared_faces[sface[i]]->GetVertices();

            if  (lf->GetGeometryType() == Geometry::TRIANGLE)
            {
               info += GetTriOrientation(sf_v, lf->GetVertices());
            }
            else
            {
               info += GetQuadOrientation(sf_v, lf->GetVertices());
            }
         }
         /*
         if (Dim == 4)
         {
            Element *lf = faces[lface];
            const int *sf_v = shared_faces[sface[i]]->GetVertices();

            std::cout << "Got into Dim == 4 case in pmesh ExchangeFaceNbrData \n";
            MFEM_ABORT("Got into Dim == 4 case in pmesh ExchangeFaceNbrData");
            if  (lf->GetGeometryType() == Geometry::TETRAHEDRON)
            {
               info += GetTetOrientation(sf_v, lf->GetVertices());
            }
         }
         */
         send_face_nbr_facedata.AddConnection(fn, info);
      }
   }
   send_face_nbr_elements.ShiftUpI();
   send_face_nbr_vertices.ShiftUpI();
   send_face_nbr_elemdata.ShiftUpI();
   send_face_nbr_facedata.ShiftUpI();

   // convert the vertex indices in send_face_nbr_elemdata
   // convert the element indices in send_face_nbr_facedata
   for (int fn = 0; fn < num_face_nbrs; fn++)
   {
      int  num_elems  = send_face_nbr_elements.RowSize(fn);
      int *elems      = send_face_nbr_elements.GetRow(fn);
      int  num_verts  = send_face_nbr_vertices.RowSize(fn);
      int *verts      = send_face_nbr_vertices.GetRow(fn);
      int *elemdata   = send_face_nbr_elemdata.GetRow(fn);
      int  num_sfaces = send_face_nbr_facedata.RowSize(fn)/2;
      int *facedata   = send_face_nbr_facedata.GetRow(fn);

      for (int i = 0; i < num_verts; i++)
      {
         vertex_marker[verts[i]] = i;
      }

      for (int el = 0; el < num_elems; el++)
      {
         const int nv = elements[el]->GetNVertices();
         elemdata += 2; // skip the attribute and the geometry type
         for (int j = 0; j < nv; j++)
         {
            elemdata[j] = vertex_marker[elemdata[j]];
         }
         elemdata += nv;

         el_marker[elems[el]] = el;
      }

      for (int i = 0; i < num_sfaces; i++)
      {
         facedata[2*i] = el_marker[facedata[2*i]];
      }
   }

   MPI_Waitall(num_face_nbrs, recv_requests, statuses);

   Array<int> recv_face_nbr_facedata;
   Table recv_face_nbr_elemdata;

   // fill-in face_nbr_elements_offset, face_nbr_vertices_offset
   face_nbr_elements_offset.SetSize(num_face_nbrs + 1);
   face_nbr_vertices_offset.SetSize(num_face_nbrs + 1);
   recv_face_nbr_elemdata.MakeI(num_face_nbrs);
   face_nbr_elements_offset[0] = 0;
   face_nbr_vertices_offset[0] = 0;
   for (int fn = 0; fn < num_face_nbrs; fn++)
   {
      face_nbr_elements_offset[fn+1] =
         face_nbr_elements_offset[fn] + nbr_recv_data[3*fn];
      face_nbr_vertices_offset[fn+1] =
         face_nbr_vertices_offset[fn] + nbr_recv_data[3*fn+1];
      recv_face_nbr_elemdata.AddColumnsInRow(fn, nbr_recv_data[3*fn+2]);
   }
   recv_face_nbr_elemdata.MakeJ();

   MPI_Waitall(num_face_nbrs, send_requests, statuses);

   // send and receive the element data
   for (int fn = 0; fn < num_face_nbrs; fn++)
   {
      int nbr_rank = GetFaceNbrRank(fn);
      int tag = 0;

      MPI_Isend(send_face_nbr_elemdata.GetRow(fn),
                send_face_nbr_elemdata.RowSize(fn),
                MPI_INT, nbr_rank, tag, MyComm, &send_requests[fn]);

      MPI_Irecv(recv_face_nbr_elemdata.GetRow(fn),
                recv_face_nbr_elemdata.RowSize(fn),
                MPI_INT, nbr_rank, tag, MyComm, &recv_requests[fn]);
   }

   // convert the element data into face_nbr_elements
   face_nbr_elements.SetSize(face_nbr_elements_offset[num_face_nbrs]);
   while (true)
   {
      int fn;
      MPI_Waitany(num_face_nbrs, recv_requests, &fn, statuses);

      if (fn == MPI_UNDEFINED)
      {
         break;
      }

      int  vert_off      = face_nbr_vertices_offset[fn];
      int  elem_off      = face_nbr_elements_offset[fn];
      int  num_elems     = face_nbr_elements_offset[fn+1] - elem_off;
      int *recv_elemdata = recv_face_nbr_elemdata.GetRow(fn);

      for (int i = 0; i < num_elems; i++)
      {
         Element *el = NewElement(recv_elemdata[1]);
         el->SetAttribute(recv_elemdata[0]);
         recv_elemdata += 2;
         int nv = el->GetNVertices();
         for (int j = 0; j < nv; j++)
         {
            recv_elemdata[j] += vert_off;
         }
         el->SetVertices(recv_elemdata);
         recv_elemdata += nv;
         face_nbr_elements[elem_off++] = el;
      }
   }

   MPI_Waitall(num_face_nbrs, send_requests, statuses);

   // send and receive the face data
   recv_face_nbr_facedata.SetSize(
      send_face_nbr_facedata.Size_of_connections());
   for (int fn = 0; fn < num_face_nbrs; fn++)
   {
      int nbr_rank = GetFaceNbrRank(fn);
      int tag = 0;

      MPI_Isend(send_face_nbr_facedata.GetRow(fn),
                send_face_nbr_facedata.RowSize(fn),
                MPI_INT, nbr_rank, tag, MyComm, &send_requests[fn]);

      // the size of the send and receive face data is the same
      MPI_Irecv(&recv_face_nbr_facedata[send_face_nbr_facedata.GetI()[fn]],
                send_face_nbr_facedata.RowSize(fn),
                MPI_INT, nbr_rank, tag, MyComm, &recv_requests[fn]);
   }

   // transfer the received face data into faces_info
   while (true)
   {
      int fn;
      MPI_Waitany(num_face_nbrs, recv_requests, &fn, statuses);

      if (fn == MPI_UNDEFINED)
      {
         break;
      }

      int  elem_off   = face_nbr_elements_offset[fn];
      int  nbr_group  = face_nbr_group[fn];
      int  num_sfaces = gr_sface->RowSize(nbr_group-1);
      int *sface      = gr_sface->GetRow(nbr_group-1);
      int *facedata =
         &recv_face_nbr_facedata[send_face_nbr_facedata.GetI()[fn]];

      for (int i = 0; i < num_sfaces; i++)
      {
         int lface = s2l_face[sface[i]];
         FaceInfo &face_info = faces_info[lface];
         face_info.Elem2No = -1 - (facedata[2*i] + elem_off);
         int info = facedata[2*i+1];
         // change the orientation in info to be relative to the local face
         if (Dim < 3)
         {
            info++; // orientation 0 --> orientation 1
         }
         else
         {
            int nbr_ori = info%64, nbr_v[4];
            Element *lf = faces[lface];
            const int *sf_v = shared_faces[sface[i]]->GetVertices();

            if  (lf->GetGeometryType() == Geometry::TRIANGLE)
            {
               // apply the nbr_ori to sf_v to get nbr_v
               const int *perm = tri_t::Orient[nbr_ori];
               for (int j = 0; j < 3; j++)
               {
                  nbr_v[perm[j]] = sf_v[j];
               }
               // get the orientation of nbr_v w.r.t. the local face
               nbr_ori = GetTriOrientation(lf->GetVertices(), nbr_v);
            }
            else
            {
               // apply the nbr_ori to sf_v to get nbr_v
               const int *perm = quad_t::Orient[nbr_ori];
               for (int j = 0; j < 4; j++)
               {
                  nbr_v[perm[j]] = sf_v[j];
               }
               // get the orientation of nbr_v w.r.t. the local face
               nbr_ori = GetQuadOrientation(lf->GetVertices(), nbr_v);
            }

            info = 64*(info/64) + nbr_ori;
         }
         face_info.Elem2Inf = info;
      }
   }

   MPI_Waitall(num_face_nbrs, send_requests, statuses);

   // allocate the face_nbr_vertices
   face_nbr_vertices.SetSize(face_nbr_vertices_offset[num_face_nbrs]);

   delete [] nbr_data;

   delete [] statuses;
   delete [] requests;

   have_face_nbr_data = true;

   ExchangeFaceNbrNodes();
}

void ParMesh::ExchangeFaceNbrNodes()
{
   if (!have_face_nbr_data)
   {
      ExchangeFaceNbrData(); // calls this method at the end
   }
   else if (Nodes == NULL)
   {
      if (Nonconforming())
      {
         // with ParNCMesh we already have the vertices
         return;
      }

      int num_face_nbrs = GetNFaceNeighbors();

      MPI_Request *requests = new MPI_Request[2*num_face_nbrs];
      MPI_Request *send_requests = requests;
      MPI_Request *recv_requests = requests + num_face_nbrs;
      MPI_Status  *statuses = new MPI_Status[num_face_nbrs];

      // allocate buffer and copy the vertices to be sent
      Array<Vertex> send_vertices(send_face_nbr_vertices.Size_of_connections());
      for (int i = 0; i < send_vertices.Size(); i++)
      {
         send_vertices[i] = vertices[send_face_nbr_vertices.GetJ()[i]];
      }

      // send and receive the vertices
      for (int fn = 0; fn < num_face_nbrs; fn++)
      {
         int nbr_rank = GetFaceNbrRank(fn);
         int tag = 0;

         MPI_Isend(send_vertices[send_face_nbr_vertices.GetI()[fn]](),
                   3*send_face_nbr_vertices.RowSize(fn),
                   MPI_DOUBLE, nbr_rank, tag, MyComm, &send_requests[fn]);

         MPI_Irecv(face_nbr_vertices[face_nbr_vertices_offset[fn]](),
                   3*(face_nbr_vertices_offset[fn+1] -
                      face_nbr_vertices_offset[fn]),
                   MPI_DOUBLE, nbr_rank, tag, MyComm, &recv_requests[fn]);
      }

      MPI_Waitall(num_face_nbrs, recv_requests, statuses);
      MPI_Waitall(num_face_nbrs, send_requests, statuses);

      delete [] statuses;
      delete [] requests;
   }
   else
   {
      ParGridFunction *pNodes = dynamic_cast<ParGridFunction *>(Nodes);
      MFEM_VERIFY(pNodes != NULL, "Nodes are not ParGridFunction!");
      pNodes->ExchangeFaceNbrData();
   }
}

int ParMesh::GetFaceNbrRank(int fn) const
{
   if (Conforming())
   {
      int nbr_group = face_nbr_group[fn];
      const int *nbs = gtopo.GetGroup(nbr_group);
      int nbr_lproc = (nbs[0]) ? nbs[0] : nbs[1];
      int nbr_rank = gtopo.GetNeighborRank(nbr_lproc);
      return nbr_rank;
   }
   else
   {
      // NC: simplified handling of face neighbor ranks
      return face_nbr_group[fn];
   }
}

Table *ParMesh::GetFaceToAllElementTable() const
{
   const Array<int> *s2l_face;
   if (Dim == 1)
   {
      s2l_face = &svert_lvert;
   }
   else if (Dim == 2)
   {
      s2l_face = &sedge_ledge;
   }
   else
   {
      s2l_face = &sface_lface;
   }

   Table *face_elem = new Table;

   face_elem->MakeI(faces_info.Size());

   for (int i = 0; i < faces_info.Size(); i++)
   {
      if (faces_info[i].Elem2No >= 0)
      {
         face_elem->AddColumnsInRow(i, 2);
      }
      else
      {
         face_elem->AddAColumnInRow(i);
      }
   }
   for (int i = 0; i < s2l_face->Size(); i++)
   {
      face_elem->AddAColumnInRow((*s2l_face)[i]);
   }

   face_elem->MakeJ();

   for (int i = 0; i < faces_info.Size(); i++)
   {
      face_elem->AddConnection(i, faces_info[i].Elem1No);
      if (faces_info[i].Elem2No >= 0)
      {
         face_elem->AddConnection(i, faces_info[i].Elem2No);
      }
   }
   for (int i = 0; i < s2l_face->Size(); i++)
   {
      int lface = (*s2l_face)[i];
      int nbr_elem_idx = -1 - faces_info[lface].Elem2No;
      face_elem->AddConnection(lface, NumOfElements + nbr_elem_idx);
   }

   face_elem->ShiftUpI();

   return face_elem;
}

ElementTransformation* ParMesh::GetGhostFaceTransformation(
   FaceElementTransformations* FETr, int face_type, int face_geom)
{
   // calculate composition of FETr->Loc1 and FETr->Elem1
   DenseMatrix &face_pm = FaceTransformation.GetPointMat();
   if (Nodes == NULL)
   {
      FETr->Elem1->Transform(FETr->Loc1.Transf.GetPointMat(), face_pm);
      FaceTransformation.SetFE(GetTransformationFEforElementType(face_type));
   }
   else
   {
      const FiniteElement* face_el =
         Nodes->FESpace()->GetTraceElement(FETr->Elem1No, face_geom);

#if 0 // TODO: handle the case of non-interpolatory Nodes
      DenseMatrix I;
      face_el->Project(Transformation.GetFE(), FETr->Loc1.Transf, I);
      MultABt(Transformation.GetPointMat(), I, pm_face);
#else
      IntegrationRule eir(face_el->GetDof());
      FETr->Loc1.Transform(face_el->GetNodes(), eir);
      Nodes->GetVectorValues(*FETr->Elem1, eir, face_pm);
#endif
      FaceTransformation.SetFE(face_el);
   }
   return &FaceTransformation;
}

FaceElementTransformations *ParMesh::
GetSharedFaceTransformations(int sf, bool fill2)
{
   int FaceNo = GetSharedFace(sf);

   FaceInfo &face_info = faces_info[FaceNo];

   bool is_slave = Nonconforming() && IsSlaveFace(face_info);
   bool is_ghost = Nonconforming() && FaceNo >= GetNumFaces();

   NCFaceInfo* nc_info = NULL;
   if (is_slave) { nc_info = &nc_faces_info[face_info.NCFace]; }

   int local_face = is_ghost ? nc_info->MasterFace : FaceNo;
   int face_type = GetFaceElementType(local_face);
   int face_geom = GetFaceGeometryType(local_face);

   // setup the transformation for the first element
   FaceElemTr.Elem1No = face_info.Elem1No;
   GetElementTransformation(FaceElemTr.Elem1No, &Transformation);
   FaceElemTr.Elem1 = &Transformation;

   // setup the transformation for the second (neighbor) element
   if (fill2)
   {
      FaceElemTr.Elem2No = -1 - face_info.Elem2No;
      GetFaceNbrElementTransformation(FaceElemTr.Elem2No, &Transformation2);
      FaceElemTr.Elem2 = &Transformation2;
   }
   else
   {
      FaceElemTr.Elem2No = -1;
   }

   // setup the face transformation if the face is not a ghost
   FaceElemTr.FaceGeom = face_geom;
   if (!is_ghost)
   {
      FaceElemTr.Face = GetFaceTransformation(FaceNo);
      // NOTE: The above call overwrites FaceElemTr.Loc1
   }

   // setup Loc1 & Loc2
   int elem_type = GetElementType(face_info.Elem1No);
   GetLocalFaceTransformation(face_type, elem_type, FaceElemTr.Loc1.Transf,
                              face_info.Elem1Inf);

   if (fill2)
   {
      elem_type = face_nbr_elements[FaceElemTr.Elem2No]->GetType();
      GetLocalFaceTransformation(face_type, elem_type, FaceElemTr.Loc2.Transf,
                                 face_info.Elem2Inf);
   }

   // adjust Loc1 or Loc2 of the master face if this is a slave face
   if (is_slave)
   {
      // is a ghost slave? -> master not a ghost -> choose Elem1 local transf
      // not a ghost slave? -> master is a ghost -> choose Elem2 local transf
      IsoparametricTransformation &loctr =
         is_ghost ? FaceElemTr.Loc1.Transf : FaceElemTr.Loc2.Transf;

      if (is_ghost || fill2)
      {
         ApplyLocalSlaveTransformation(loctr, face_info);
      }

      if (face_type == Element::SEGMENT && fill2)
      {
         // fix slave orientation in 2D: flip Loc2 to match Loc1 and Face
         DenseMatrix &pm = FaceElemTr.Loc2.Transf.GetPointMat();
         std::swap(pm(0,0), pm(0,1));
         std::swap(pm(1,0), pm(1,1));
      }
   }

   // for ghost faces we need a special version of GetFaceTransformation
   if (is_ghost)
   {
      FaceElemTr.Face =
         GetGhostFaceTransformation(&FaceElemTr, face_type, face_geom);
   }

   return &FaceElemTr;
}

int ParMesh::GetNSharedFaces() const
{
   if (Conforming())
   {
      switch (Dim)
      {
         case 1:  return svert_lvert.Size();
         case 2:  return sedge_ledge.Size();
         default: return sface_lface.Size();
      }
   }
   else
   {
      MFEM_ASSERT(Dim > 1, "");
      const NCMesh::NCList &shared = pncmesh->GetSharedList(Dim-1);
      return shared.conforming.size() + shared.slaves.size();
   }
}

int ParMesh::GetSharedFace(int sface) const
{
   if (Conforming())
   {
      switch (Dim)
      {
         case 1:  return svert_lvert[sface];
         case 2:  return sedge_ledge[sface];
         default: return sface_lface[sface];
      }
   }
   else
   {
      MFEM_ASSERT(Dim > 1, "");
      const NCMesh::NCList &shared = pncmesh->GetSharedList(Dim-1);
      int csize = (int) shared.conforming.size();
      return sface < csize
             ? shared.conforming[sface].index
             : shared.slaves[sface - csize].index;
   }
}

void ParMesh::ReorientTetMesh()
{
   if (Dim != 3 || !(meshgen & 1))
   {
      return;
   }

   Mesh::ReorientTetMesh();

   int *v;

   // The local edge and face numbering is changed therefore we need to
   // update sedge_ledge and sface_lface.
   {
      DSTable v_to_v(NumOfVertices);
      GetVertexToVertexTable(v_to_v);
      for (int i = 0; i < shared_edges.Size(); i++)
      {
         v = shared_edges[i]->GetVertices();
         sedge_ledge[i] = v_to_v(v[0], v[1]);
      }
   }

   // Rotate shared faces and update sface_lface.
   // Note that no communication is needed to ensure that the shared
   // faces are rotated in the same way in both processors. This is
   // automatic due to various things, e.g. the global to local vertex
   // mapping preserves the global order; also the way new vertices
   // are introduced during refinement is essential.
   {
      STable3D *faces_tbl = GetFacesTable();
      for (int i = 0; i < shared_faces.Size(); i++)
         if (shared_faces[i]->GetType() == Element::TRIANGLE)
         {
            v = shared_faces[i]->GetVertices();

            Rotate3(v[0], v[1], v[2]);

            sface_lface[i] = (*faces_tbl)(v[0], v[1], v[2]);
         }
      delete faces_tbl;
   }
}

void ParMesh::LocalRefinement(const Array<int> &marked_el, int type)
{
   int i, j;

   if (pncmesh)
   {
      MFEM_ABORT("Local and nonconforming refinements cannot be mixed.");
   }

   DeleteFaceNbrData();

   InitRefinementTransforms();

   if (Dim == 4)
   {
      // 1. Get table of vertex to vertex connections.
      DSTable v_to_v(NumOfVertices);
      GetVertexToVertexTable(v_to_v);

      // 2. Get edge to element connections in arrays edge1 and edge2
      Array<int> middle(v_to_v.NumberOfEntries());
      middle = -1;

      // 3. Do the red refinement.
      for (int i = 0; i < marked_el.Size(); i++)
      {
         RedRefinementPentatope(marked_el[i], v_to_v, middle);
      }

      // 5. Update the boundary elements.
      for (int i = 0; i < NumOfBdrElements; i++)
         if (boundary[i]->NeedRefinement(v_to_v, middle))
         {
            RedRefinementBoundaryTet(i, v_to_v, middle);
         }
      NumOfBdrElements = boundary.Size();



      // 5a. Update the groups after refinement.
      if (el_to_face != NULL)
      {
         RefineGroups(v_to_v, middle);
         //         GetElementToFaceTable4D(); // Called by RefineGroups
         GenerateFaces();

         //         Update4DFaceFlipInfo();

         //         GetElementToPlanarTable(); // Called by RefineGroups
         GeneratePlanars();

      }

      // 7. Free the allocated memory.
      middle.DeleteAll();

      if (el_to_edge != NULL)
      {
         NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
      }



   }
   else if (Dim == 3)
   {
      int uniform_refinement = 0;
      if (type < 0)
      {
         type = -type;
         uniform_refinement = 1;
      }

      // 1. Get table of vertex to vertex connections.
      DSTable v_to_v(NumOfVertices);
      GetVertexToVertexTable(v_to_v);

      // 2. Create a marker array for all edges (vertex to vertex connections).
      Array<int> middle(v_to_v.NumberOfEntries());
      middle = -1;

      // 3. Do the red refinement.
      switch (type)
      {
         case 1:
            for (i = 0; i < marked_el.Size(); i++)
            {
               Bisection(marked_el[i], v_to_v, NULL, NULL, middle);
            }
            break;
         case 2:
            for (i = 0; i < marked_el.Size(); i++)
            {
               Bisection(marked_el[i], v_to_v, NULL, NULL, middle);

               Bisection(NumOfElements - 1, v_to_v, NULL, NULL, middle);
               Bisection(marked_el[i], v_to_v, NULL, NULL, middle);
            }
            break;
         case 3:
            for (i = 0; i < marked_el.Size(); i++)
            {
               Bisection(marked_el[i], v_to_v, NULL, NULL, middle);

               j = NumOfElements - 1;
               Bisection(j, v_to_v, NULL, NULL, middle);
               Bisection(NumOfElements - 1, v_to_v, NULL, NULL, middle);
               Bisection(j, v_to_v, NULL, NULL, middle);

               Bisection(marked_el[i], v_to_v, NULL, NULL, middle);
               Bisection(NumOfElements-1, v_to_v, NULL, NULL, middle);
               Bisection(marked_el[i], v_to_v, NULL, NULL, middle);
            }
            break;
      }

      // 4. Do the green refinement (to get conforming mesh).
      int need_refinement;
      int refined_edge[5][3] =
      {
         {0, 0, 0},
         {1, 0, 0},
         {1, 1, 0},
         {1, 0, 1},
         {1, 1, 1}
      };
      int faces_in_group, max_faces_in_group = 0;
      // face_splittings identify how the shared faces have been split
      int **face_splittings = new int*[GetNGroups()-1];
      for (i = 0; i < GetNGroups()-1; i++)
      {
         faces_in_group = GroupNFaces(i+1);
         face_splittings[i] = new int[faces_in_group];
         if (faces_in_group > max_faces_in_group)
         {
            max_faces_in_group = faces_in_group;
         }
      }
      int neighbor, *iBuf = new int[max_faces_in_group];

      Array<int> group_faces;

      MPI_Request request;
      MPI_Status  status;

#ifdef MFEM_DEBUG_PARMESH_LOCALREF
      int ref_loops_all = 0, ref_loops_par = 0;
#endif
      do
      {
         need_refinement = 0;
         for (i = 0; i < NumOfElements; i++)
         {
            if (elements[i]->NeedRefinement(v_to_v, middle))
            {
               need_refinement = 1;
               Bisection(i, v_to_v, NULL, NULL, middle);
            }
         }
#ifdef MFEM_DEBUG_PARMESH_LOCALREF
         ref_loops_all++;
#endif

         if (uniform_refinement)
         {
            continue;
         }

         // if the mesh is locally conforming start making it globally
         // conforming
         if (need_refinement == 0)
         {
#ifdef MFEM_DEBUG_PARMESH_LOCALREF
            ref_loops_par++;
#endif
            // MPI_Barrier(MyComm);
            const int tag = 293;

            // (a) send the type of interface splitting
            for (i = 0; i < GetNGroups()-1; i++)
            {
               group_sface.GetRow(i, group_faces);
               faces_in_group = group_faces.Size();
               // it is enough to communicate through the faces
               if (faces_in_group == 0) { continue; }

               for (j = 0; j < faces_in_group; j++)
               {
                  face_splittings[i][j] =
                     GetFaceSplittings(shared_faces[group_faces[j]], v_to_v,
                                       middle);
               }
               const int *nbs = gtopo.GetGroup(i+1);
               neighbor = gtopo.GetNeighborRank(nbs[0] ? nbs[0] : nbs[1]);
               MPI_Isend(face_splittings[i], faces_in_group, MPI_INT,
                         neighbor, tag, MyComm, &request);
            }

            // (b) receive the type of interface splitting
            for (i = 0; i < GetNGroups()-1; i++)
            {
               group_sface.GetRow(i, group_faces);
               faces_in_group = group_faces.Size();
               if (faces_in_group == 0) { continue; }

               const int *nbs = gtopo.GetGroup(i+1);
               neighbor = gtopo.GetNeighborRank(nbs[0] ? nbs[0] : nbs[1]);
               MPI_Recv(iBuf, faces_in_group, MPI_INT, neighbor,
                        tag, MyComm, &status);

               for (j = 0; j < faces_in_group; j++)
               {
                  if (iBuf[j] == face_splittings[i][j]) { continue; }

                  int *v = shared_faces[group_faces[j]]->GetVertices();
                  for (int k = 0; k < 3; k++)
                  {
                     if (refined_edge[iBuf[j]][k] != 1 ||
                         refined_edge[face_splittings[i][j]][k] != 0)
                     { continue; }

                     int ind[2] = { v[k], v[(k+1)%3] };
                     int ii = v_to_v(ind[0], ind[1]);
                     if (middle[ii] == -1)
                     {
                        need_refinement = 1;
                        middle[ii] = NumOfVertices++;
                        vertices.Append(Vertex());
                        AverageVertices(ind, 2, vertices.Size()-1);
                     }
                  }
               }
            }

            i = need_refinement;
            MPI_Allreduce(&i, &need_refinement, 1, MPI_INT, MPI_LOR, MyComm);
         }
      }
      while (need_refinement == 1);

#ifdef MFEM_DEBUG_PARMESH_LOCALREF
      i = ref_loops_all;
      MPI_Reduce(&i, &ref_loops_all, 1, MPI_INT, MPI_MAX, 0, MyComm);
      if (MyRank == 0)
      {
         cout << "\n\nParMesh::LocalRefinement : max. ref_loops_all = "
              << ref_loops_all << ", ref_loops_par = " << ref_loops_par
              << '\n' << endl;
      }
#endif

      delete [] iBuf;
      for (i = 0; i < GetNGroups()-1; i++)
      {
         delete [] face_splittings[i];
      }
      delete [] face_splittings;


      // 5. Update the boundary elements.
      do
      {
         need_refinement = 0;
         for (i = 0; i < NumOfBdrElements; i++)
         {
            if (boundary[i]->NeedRefinement(v_to_v, middle))
            {
               need_refinement = 1;
               Bisection(i, v_to_v, middle);
            }
         }
      }
      while (need_refinement == 1);

      if (NumOfBdrElements != boundary.Size())
         mfem_error("ParMesh::LocalRefinement :"
                    " (NumOfBdrElements != boundary.Size())");

      // 5a. Update the groups after refinement.
      if (el_to_face != NULL)
      {
         RefineGroups(v_to_v, middle);
         // GetElementToFaceTable(); // Called by RefineGroups
         GenerateFaces();
      }

      // 6. Un-mark the Pf elements.
      int refinement_edges[2], type, flag;
      for (i = 0; i < NumOfElements; i++)
      {
         Tetrahedron* el = (Tetrahedron*) elements[i];
         el->ParseRefinementFlag(refinement_edges, type, flag);

         if (type == Tetrahedron::TYPE_PF)
         {
            el->CreateRefinementFlag(refinement_edges, Tetrahedron::TYPE_PU,
                                     flag);
         }
      }

      // 7. Free the allocated memory.
      middle.DeleteAll();

      delete el_to_el;
      delete face_edge;
      delete edge_vertex;
      edge_vertex = NULL;

      if (el_to_edge != NULL)
      {
         NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
      }
   } //  'if (Dim == 3)'


   if (Dim == 2)
   {
      int uniform_refinement = 0;
      if (type < 0)
      {
         type = -type;
         uniform_refinement = 1;
      }

      // 1. Get table of vertex to vertex connections.
      DSTable v_to_v(NumOfVertices);
      GetVertexToVertexTable(v_to_v);

      // 2. Get edge to element connections in arrays edge1 and edge2
      int nedges  = v_to_v.NumberOfEntries();
      int *edge1  = new int[nedges];
      int *edge2  = new int[nedges];
      int *middle = new int[nedges];

      for (i = 0; i < nedges; i++)
      {
         edge1[i] = edge2[i] = middle[i] = -1;
      }

      for (i = 0; i < NumOfElements; i++)
      {
         int *v = elements[i]->GetVertices();
         for (j = 0; j < 3; j++)
         {
            int ind = v_to_v(v[j], v[(j+1)%3]);
            (edge1[ind] == -1) ? (edge1[ind] = i) : (edge2[ind] = i);
         }
      }

      // 3. Do the red refinement.
      for (i = 0; i < marked_el.Size(); i++)
      {
         RedRefinement(marked_el[i], v_to_v, edge1, edge2, middle);
      }

      // 4. Do the green refinement (to get conforming mesh).
      int need_refinement;
      int edges_in_group, max_edges_in_group = 0;
      // edge_splittings identify how the shared edges have been split
      int **edge_splittings = new int*[GetNGroups()-1];
      for (i = 0; i < GetNGroups()-1; i++)
      {
         edges_in_group = GroupNEdges(i+1);
         edge_splittings[i] = new int[edges_in_group];
         if (edges_in_group > max_edges_in_group)
         {
            max_edges_in_group = edges_in_group;
         }
      }
      int neighbor, *iBuf = new int[max_edges_in_group];

      Array<int> group_edges;

      MPI_Request request;
      MPI_Status  status;
      Vertex V;
      V(2) = 0.0;

#ifdef MFEM_DEBUG_PARMESH_LOCALREF
      int ref_loops_all = 0, ref_loops_par = 0;
#endif
      do
      {
         need_refinement = 0;
         for (i = 0; i < nedges; i++)
            if (middle[i] != -1 && edge1[i] != -1)
            {
               need_refinement = 1;
               GreenRefinement(edge1[i], v_to_v, edge1, edge2, middle);
            }
#ifdef MFEM_DEBUG_PARMESH_LOCALREF
         ref_loops_all++;
#endif

         if (uniform_refinement)
         {
            continue;
         }

         // if the mesh is locally conforming start making it globally
         // conforming
         if (need_refinement == 0)
         {
#ifdef MFEM_DEBUG_PARMESH_LOCALREF
            ref_loops_par++;
#endif
            // MPI_Barrier(MyComm);

            // (a) send the type of interface splitting
            for (i = 0; i < GetNGroups()-1; i++)
            {
               group_sedge.GetRow(i, group_edges);
               edges_in_group = group_edges.Size();
               // it is enough to communicate through the edges
               if (edges_in_group != 0)
               {
                  for (j = 0; j < edges_in_group; j++)
                     edge_splittings[i][j] =
                        GetEdgeSplittings(shared_edges[group_edges[j]], v_to_v,
                                          middle);
                  const int *nbs = gtopo.GetGroup(i+1);
                  if (nbs[0] == 0)
                  {
                     neighbor = gtopo.GetNeighborRank(nbs[1]);
                  }
                  else
                  {
                     neighbor = gtopo.GetNeighborRank(nbs[0]);
                  }
                  MPI_Isend(edge_splittings[i], edges_in_group, MPI_INT,
                            neighbor, 0, MyComm, &request);
               }
            }

            // (b) receive the type of interface splitting
            for (i = 0; i < GetNGroups()-1; i++)
            {
               group_sedge.GetRow(i, group_edges);
               edges_in_group = group_edges.Size();
               if (edges_in_group != 0)
               {
                  const int *nbs = gtopo.GetGroup(i+1);
                  if (nbs[0] == 0)
                  {
                     neighbor = gtopo.GetNeighborRank(nbs[1]);
                  }
                  else
                  {
                     neighbor = gtopo.GetNeighborRank(nbs[0]);
                  }
                  MPI_Recv(iBuf, edges_in_group, MPI_INT, neighbor,
                           MPI_ANY_TAG, MyComm, &status);

                  for (j = 0; j < edges_in_group; j++)
                     if (iBuf[j] == 1 && edge_splittings[i][j] == 0)
                     {
                        int *v = shared_edges[group_edges[j]]->GetVertices();
                        int ii = v_to_v(v[0], v[1]);
#ifdef MFEM_DEBUG_PARMESH_LOCALREF
                        if (middle[ii] != -1)
                           mfem_error("ParMesh::LocalRefinement (triangles) : "
                                      "Oops!");
#endif
                        need_refinement = 1;
                        middle[ii] = NumOfVertices++;
                        for (int c = 0; c < 2; c++)
                        {
                           V(c) = 0.5 * (vertices[v[0]](c) + vertices[v[1]](c));
                        }
                        vertices.Append(V);
                     }
               }
            }

            i = need_refinement;
            MPI_Allreduce(&i, &need_refinement, 1, MPI_INT, MPI_LOR, MyComm);
         }
      }
      while (need_refinement == 1);

#ifdef MFEM_DEBUG_PARMESH_LOCALREF
      i = ref_loops_all;
      MPI_Reduce(&i, &ref_loops_all, 1, MPI_INT, MPI_MAX, 0, MyComm);
      if (MyRank == 0)
      {
         cout << "\n\nParMesh::LocalRefinement : max. ref_loops_all = "
              << ref_loops_all << ", ref_loops_par = " << ref_loops_par
              << '\n' << endl;
      }
#endif

      for (i = 0; i < GetNGroups()-1; i++)
      {
         delete [] edge_splittings[i];
      }
      delete [] edge_splittings;

      delete [] iBuf;

      // 5. Update the boundary elements.
      int v1[2], v2[2], bisect, temp;
      temp = NumOfBdrElements;
      for (i = 0; i < temp; i++)
      {
         int *v = boundary[i]->GetVertices();
         bisect = v_to_v(v[0], v[1]);
         if (middle[bisect] != -1)
         {
            // the element was refined (needs updating)
            if (boundary[i]->GetType() == Element::SEGMENT)
            {
               v1[0] =           v[0]; v1[1] = middle[bisect];
               v2[0] = middle[bisect]; v2[1] =           v[1];

               boundary[i]->SetVertices(v1);
               boundary.Append(new Segment(v2, boundary[i]->GetAttribute()));
            }
            else
               mfem_error("Only bisection of segment is implemented for bdr"
                          " elem.");
         }
      }
      NumOfBdrElements = boundary.Size();

      // 5a. Update the groups after refinement.
      RefineGroups(v_to_v, middle);

      // 6. Free the allocated memory.
      delete [] edge1;
      delete [] edge2;
      delete [] middle;

      if (el_to_edge != NULL)
      {
         NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
         GenerateFaces();
      }
   } //  'if (Dim == 2)'

   if (Dim == 1) // --------------------------------------------------------
   {
      int cne = NumOfElements, cnv = NumOfVertices;
      NumOfVertices += marked_el.Size();
      NumOfElements += marked_el.Size();
      vertices.SetSize(NumOfVertices);
      elements.SetSize(NumOfElements);
      CoarseFineTr.embeddings.SetSize(NumOfElements);

      for (j = 0; j < marked_el.Size(); j++)
      {
         i = marked_el[j];
         Segment *c_seg = (Segment *)elements[i];
         int *vert = c_seg->GetVertices(), attr = c_seg->GetAttribute();
         int new_v = cnv + j, new_e = cne + j;
         AverageVertices(vert, 2, new_v);
         elements[new_e] = new Segment(new_v, vert[1], attr);
         vert[1] = new_v;

         CoarseFineTr.embeddings[i] = Embedding(i, 1);
         CoarseFineTr.embeddings[new_e] = Embedding(i, 2);
      }

      static double seg_children[3*2] = { 0.0,1.0, 0.0,0.5, 0.5,1.0 };
      CoarseFineTr.point_matrices.UseExternalData(seg_children, 1, 2, 3);

      GenerateFaces();
   } // end of 'if (Dim == 1)'

   last_operation = Mesh::REFINE;
   sequence++;

   UpdateNodes();

#ifdef MFEM_DEBUG
   CheckElementOrientation(false);
   CheckBdrElementOrientation(false);
#endif
}

void ParMesh::NonconformingRefinement(const Array<Refinement> &refinements,
                                      int nc_limit)
{
   if (NURBSext)
   {
      MFEM_ABORT("ParMesh::NonconformingRefinement: NURBS meshes are not "
                 "supported. Project the NURBS to Nodes first.");
   }

   if (!pncmesh)
   {
      MFEM_ABORT("Can't convert conforming ParMesh to nonconforming ParMesh "
                 "(you need to initialize the ParMesh from a nonconforming "
                 "serial Mesh)");
   }

   // NOTE: no check of !refinements.Size(), in parallel we would have to reduce

   // do the refinements
   pncmesh->MarkCoarseLevel();
   pncmesh->Refine(refinements);

   if (nc_limit > 0)
   {
      pncmesh->LimitNCLevel(nc_limit);
   }

   // create a second mesh containing the finest elements from 'pncmesh'
   ParMesh* pmesh2 = new ParMesh(*pncmesh);
   pncmesh->OnMeshUpdated(pmesh2);

   attributes.Copy(pmesh2->attributes);
   bdr_attributes.Copy(pmesh2->bdr_attributes);

   // now swap the meshes, the second mesh will become the old coarse mesh
   // and this mesh will be the new fine mesh
   Swap(*pmesh2, false);

   delete pmesh2; // NOTE: old face neighbors destroyed here

   GenerateNCFaceInfo();

   last_operation = Mesh::REFINE;
   sequence++;

   if (Nodes) // update/interpolate curved mesh
   {
      Nodes->FESpace()->Update();
      Nodes->Update();
   }
}

bool ParMesh::NonconformingDerefinement(Array<double> &elem_error,
                                        double threshold, int nc_limit, int op)
{
   const Table &dt = pncmesh->GetDerefinementTable();

   pncmesh->SynchronizeDerefinementData(elem_error, dt);

   Array<int> level_ok;
   if (nc_limit > 0)
   {
      pncmesh->CheckDerefinementNCLevel(dt, level_ok, nc_limit);
   }

   Array<int> derefs;
   for (int i = 0; i < dt.Size(); i++)
   {
      if (nc_limit > 0 && !level_ok[i]) { continue; }

      const int* fine = dt.GetRow(i);
      int size = dt.RowSize(i);

      double error = 0.0;
      for (int j = 0; j < size; j++)
      {
         MFEM_VERIFY(fine[j] < elem_error.Size(), "");

         double err_fine = elem_error[fine[j]];
         switch (op)
         {
            case 0: error = std::min(error, err_fine); break;
            case 1: error += err_fine; break;
            case 2: error = std::max(error, err_fine); break;
         }
      }

      if (error < threshold) { derefs.Append(i); }
   }

   long glob_size = ReduceInt(derefs.Size());
   if (glob_size)
   {
      DerefineMesh(derefs);
      return true;
   }

   return false;
}

void ParMesh::Rebalance()
{
   if (Conforming())
   {
      MFEM_ABORT("Load balancing is currently not supported for conforming"
                 " meshes.");
   }

   DeleteFaceNbrData();

   pncmesh->Rebalance();

   ParMesh* pmesh2 = new ParMesh(*pncmesh);
   pncmesh->OnMeshUpdated(pmesh2);

   attributes.Copy(pmesh2->attributes);
   bdr_attributes.Copy(pmesh2->bdr_attributes);

   Swap(*pmesh2, false);
   delete pmesh2;

   GenerateNCFaceInfo();

   last_operation = Mesh::REBALANCE;
   sequence++;

   if (Nodes) // redistribute curved mesh
   {
      Nodes->FESpace()->Update();
      Nodes->Update();
   }
}

void ParMesh::RefineGroups(const DSTable &v_to_v, int *middle)
{
   int i, attr, newv[3], ind, f_ind, *v;

   int group;
   Array<int> group_verts, group_edges, group_planars, group_faces;

   // To update the groups after a refinement, we observe that:
   // - every (new and old) vertex, edge and face belongs to exactly one group
   // - the refinement does not create new groups
   // - a new vertex appears only as the middle of a refined edge
   // - a face can be refined 2, 3 or 4 times producing new edges and faces

   int *I_group_svert, *J_group_svert;
   int *I_group_sedge, *J_group_sedge;
   int *I_group_splan, *J_group_splan;
   int *I_group_sface, *J_group_sface;

   I_group_svert = new int[GetNGroups()+1];
   I_group_sedge = new int[GetNGroups()+1];
   if (Dim == 3 || Dim == 4)
   {
      I_group_sface = new int[GetNGroups()+1];
   }
   else
   {
      I_group_sface = NULL;
   }
   if (Dim==4) { I_group_splan = new int[GetNGroups()+1]; }

   I_group_svert[0] = I_group_svert[1] = 0;
   I_group_sedge[0] = I_group_sedge[1] = 0;
   if (Dim==4) { I_group_splan[0] = I_group_splan[1] = 0; }
   if (Dim == 3 || Dim == 4)
   {
      I_group_sface[0] = I_group_sface[1] = 0;
   }

   // overestimate the size of the J arrays
   if (Dim == 4)
   {
      J_group_svert = new int[group_svert.Size_of_connections()
                              + group_sedge.Size_of_connections()];
      J_group_sedge = new int[2*group_sedge.Size_of_connections()
                              + 3*group_splan.Size_of_connections()
                              + group_sface.Size_of_connections()];
      J_group_splan = new int[4*group_splan.Size_of_connections()
                              + 8*group_sface.Size_of_connections()];
      J_group_sface = new int[8*group_sface.Size_of_connections()];
   }
   else if (Dim == 3)
   {
      J_group_svert = new int[group_svert.Size_of_connections()
                              + group_sedge.Size_of_connections()];
      J_group_sedge = new int[2*group_sedge.Size_of_connections()
                              + 3*group_sface.Size_of_connections()];
      J_group_sface = new int[4*group_sface.Size_of_connections()];
   }
   else if (Dim == 2)
   {
      J_group_svert = new int[group_svert.Size_of_connections()
                              + group_sedge.Size_of_connections()];
      J_group_sedge = new int[2*group_sedge.Size_of_connections()];
      J_group_sface = NULL;
   }
   else
   {
      J_group_svert = J_group_sedge = J_group_sface = NULL;
   }

   for (group = 0; group < GetNGroups()-1; group++)
   {
      // Get the group shared objects
      group_svert.GetRow(group, group_verts);
      group_sedge.GetRow(group, group_edges);
      if (Dim==4) { group_splan.GetRow(group, group_planars); }
      group_sface.GetRow(group, group_faces);

      // Check which edges have been refined
      for (i = 0; i < group_sedge.RowSize(group); i++)
      {
         v = shared_edges[group_edges[i]]->GetVertices();
         ind = middle[v_to_v(v[0], v[1])];
         if (ind != -1)
         {
            // add a vertex
            group_verts.Append(svert_lvert.Append(ind)-1);
            // update the edges
            attr = shared_edges[group_edges[i]]->GetAttribute();
            shared_edges.Append(new Segment(v[1], ind, attr));
            group_edges.Append(sedge_ledge.Append(-1)-1);
            v[1] = ind;
         }
      }

      // Check which planars have been refined
      if (Dim==4)
      {
         for (i = 0; i < group_splan.RowSize(group); i++)
         {
            v = shared_planars[group_planars[i]]->GetVertices();
            ind = middle[v_to_v(v[0], v[1])];
            if (ind != -1)
            {
               if (shared_planars[group_planars[i]]->GetGeometryType() == Element::TRIANGLE)
               {
                  attr = shared_planars[group_planars[i]]->GetAttribute();

                  //we have to add 3 new shared edges
                  int midEdges[3];
                  midEdges[0] = middle[v_to_v(v[0],v[1])];
                  midEdges[1] = middle[v_to_v(v[0],v[2])];
                  midEdges[2] = middle[v_to_v(v[1],v[2])];

                  shared_edges.Append(new Segment(midEdges[0], midEdges[1], attr));
                  group_edges.Append(sedge_ledge.Append(-1)-1);
                  shared_edges.Append(new Segment(midEdges[0], midEdges[2], attr));
                  group_edges.Append(sedge_ledge.Append(-1)-1);
                  shared_edges.Append(new Segment(midEdges[1], midEdges[2], attr));
                  group_edges.Append(sedge_ledge.Append(-1)-1);

                  shared_planars.Append(new Triangle(v[0], midEdges[0], midEdges[1], attr));
                  group_planars.Append(splan_lplan.Append(-1)-1);
                  shared_planars.Append(new Triangle(midEdges[0], v[1], midEdges[2], attr));
                  group_planars.Append(splan_lplan.Append(-1)-1);
                  shared_planars.Append(new Triangle(midEdges[1], midEdges[2], v[2], attr));
                  group_planars.Append(splan_lplan.Append(-1)-1);
                  int w[3]; w[0] = midEdges[0]; w[1] = midEdges[1]; w[2] = midEdges[2];
                  shared_planars[group_planars[i]]->SetVertices(w);
               }
            }
         }
      }

      // Check which faces have been refined
      for (i = 0; i < group_sface.RowSize(group); i++)
      {
         v = shared_faces[group_faces[i]]->GetVertices();
         ind = middle[v_to_v(v[0], v[1])];
         if (ind != -1)
         {
            if (shared_faces[group_faces[i]]->GetGeometryType() == Element::TRIANGLE)
            {
               attr = shared_faces[group_faces[i]]->GetAttribute();
               // add the refinement edge
               shared_edges.Append(new Segment(v[2], ind, attr));
               group_edges.Append(sedge_ledge.Append(-1)-1);
               // add a face
               f_ind = group_faces.Size();
               shared_faces.Append(new Triangle(v[1], v[2], ind, attr));
               group_faces.Append(sface_lface.Append(-1)-1);
               newv[0] = v[2]; newv[1] = v[0]; newv[2] = ind;
               shared_faces[group_faces[i]]->SetVertices(newv);

               // check if the left face has also been refined
               // v = shared_faces[group_faces[i]]->GetVertices();
               ind = middle[v_to_v(v[0], v[1])];
               if (ind != -1)
               {
                  // add the refinement edge
                  shared_edges.Append(new Segment(v[2], ind, attr));
                  group_edges.Append(sedge_ledge.Append(-1)-1);
                  // add a face
                  shared_faces.Append(new Triangle(v[1], v[2], ind, attr));
                  group_faces.Append(sface_lface.Append(-1)-1);
                  newv[0] = v[2]; newv[1] = v[0]; newv[2] = ind;
                  shared_faces[group_faces[i]]->SetVertices(newv);
               }

               // check if the right face has also been refined
               v = shared_faces[group_faces[f_ind]]->GetVertices();
               ind = middle[v_to_v(v[0], v[1])];
               if (ind != -1)
               {
                  // add the refinement edge
                  shared_edges.Append(new Segment(v[2], ind, attr));
                  group_edges.Append(sedge_ledge.Append(-1)-1);
                  // add a face
                  shared_faces.Append(new Triangle(v[1], v[2], ind, attr));
                  group_faces.Append(sface_lface.Append(-1)-1);
                  newv[0] = v[2]; newv[1] = v[0]; newv[2] = ind;
                  shared_faces[group_faces[f_ind]]->SetVertices(newv);
               }
            }
            else if (shared_faces[group_faces[i]]->GetGeometryType() ==
                     Element::TETRAHEDRON)
            {
               attr = shared_faces[group_faces[i]]->GetAttribute();

               int faceIndex = sface_lface[group_faces[i]];

               bool swapped = swappedFaces[faceIndex];
               swapped = false;
               if (swapped) { Swap(v); }

               //we have to add 13 new shared edges
               const int* ei;
               int midEdges[6];
               for (int j=0; j<6; j++)
               {
                  ei = shared_faces[group_faces[i]]->GetEdgeVertices(j);
                  midEdges[j] = middle[v_to_v(v[ei[0]],v[ei[1]])];
               }


               shared_edges.Append(new Segment(midEdges[1], midEdges[4], attr));
               group_edges.Append(sedge_ledge.Append(-1)-1);

               shared_planars.Append(new Triangle(midEdges[0], midEdges[1], midEdges[2],
                                                  attr)); group_planars.Append(splan_lplan.Append(-1)-1);
               //           shared_trigs.Append(new Triangle(midEdges[0], midEdges[1], midEdges[3], attr)); group_trigs.Append(strig_ltrig.Append(-1)-1);
               shared_planars.Append(new Triangle(midEdges[0], midEdges[1], midEdges[4],
                                                  attr)); group_planars.Append(splan_lplan.Append(-1)-1);
               //           shared_trigs.Append(new Triangle(midEdges[0], midEdges[2], midEdges[4], attr)); group_trigs.Append(strig_ltrig.Append(-1)-1);
               shared_planars.Append(new Triangle(midEdges[0], midEdges[3], midEdges[4],
                                                  attr)); group_planars.Append(splan_lplan.Append(-1)-1);
               shared_planars.Append(new Triangle(midEdges[1], midEdges[2], midEdges[4],
                                                  attr)); group_planars.Append(splan_lplan.Append(-1)-1);
               //           shared_trigs.Append(new Triangle(midEdges[1], midEdges[2], midEdges[5], attr)); group_trigs.Append(strig_ltrig.Append(-1)-1);
               shared_planars.Append(new Triangle(midEdges[1], midEdges[3], midEdges[4],
                                                  attr)); group_planars.Append(splan_lplan.Append(-1)-1);
               shared_planars.Append(new Triangle(midEdges[1], midEdges[3], midEdges[5],
                                                  attr)); group_planars.Append(splan_lplan.Append(-1)-1);
               shared_planars.Append(new Triangle(midEdges[1], midEdges[4], midEdges[5],
                                                  attr)); group_planars.Append(splan_lplan.Append(-1)-1);
               shared_planars.Append(new Triangle(midEdges[2], midEdges[4], midEdges[5],
                                                  attr)); group_planars.Append(splan_lplan.Append(-1)-1);
               //           shared_trigs.Append(new Triangle(midEdges[3], midEdges[4], midEdges[5], attr)); group_trigs.Append(strig_ltrig.Append(-1)-1);

               int w[4];
               bool mySwaped;
               w[0] = v[0];     w[1] = midEdges[0]; w[2] = midEdges[1]; w[3] = midEdges[2];
               mySwaped = swapped;
               if (mySwaped) { Swap(w); } shared_faces.Append(new Tetrahedron(w, attr));
               group_faces.Append(sface_lface.Append(-1)-1);
               w[0] = midEdges[0]; w[1] = v[1];     w[2] = midEdges[3]; w[3] = midEdges[4];
               mySwaped = swapped;
               if (mySwaped) { Swap(w); } shared_faces.Append(new Tetrahedron(w, attr));
               group_faces.Append(sface_lface.Append(-1)-1);
               w[0] = midEdges[1]; w[1] = midEdges[3]; w[2] = v[2];     w[3] = midEdges[5];
               mySwaped = swapped;
               if (mySwaped) { Swap(w); } shared_faces.Append(new Tetrahedron(w, attr));
               group_faces.Append(sface_lface.Append(-1)-1);
               w[0] = midEdges[2]; w[1] = midEdges[4]; w[2] = midEdges[5]; w[3] = v[3];
               mySwaped = swapped;
               if (mySwaped) { Swap(w); } shared_faces.Append(new Tetrahedron(w, attr));
               group_faces.Append(sface_lface.Append(-1)-1);

               w[0] = midEdges[0]; w[1] = midEdges[1]; w[2] = midEdges[3]; w[3] = midEdges[4];
               mySwaped = !swapped; mySwaped = false;
               if (mySwaped) { Swap(w); } shared_faces.Append(new Tetrahedron(w, attr));
               group_faces.Append(sface_lface.Append(-1)-1);
               w[0] = midEdges[0]; w[1] = midEdges[1]; w[2] = midEdges[2]; w[3] = midEdges[4];
               mySwaped = swapped;
               if (mySwaped) { Swap(w); } shared_faces.Append(new Tetrahedron(w, attr));
               group_faces.Append(sface_lface.Append(-1)-1);
               w[0] = midEdges[1]; w[1] = midEdges[3]; w[2] = midEdges[4]; w[3] = midEdges[5];
               mySwaped = !swapped; mySwaped = false;
               if (mySwaped) { Swap(w); } shared_faces.Append(new Tetrahedron(w, attr));
               group_faces.Append(sface_lface.Append(-1)-1);
               w[0] = midEdges[1]; w[1] = midEdges[2]; w[2] = midEdges[4]; w[3] = midEdges[5];
               mySwaped = swapped;
               if (mySwaped) { Swap(w); } shared_faces[group_faces[i]]->SetVertices(
                  w); //sface_lface[group_faces[i]] =  -1;
            }
         }
      }

      I_group_svert[group+1] = I_group_svert[group] + group_verts.Size();
      I_group_sedge[group+1] = I_group_sedge[group] + group_edges.Size();
      if (Dim==4) { I_group_splan[group+1] = I_group_splan[group] + group_planars.Size(); }
      if (Dim == 3 || Dim == 4)
      {
         I_group_sface[group+1] = I_group_sface[group] + group_faces.Size();
      }

      int *J;
      J = J_group_svert+I_group_svert[group];
      for (i = 0; i < group_verts.Size(); i++)
      {
         J[i] = group_verts[i];
      }
      J = J_group_sedge+I_group_sedge[group];
      for (i = 0; i < group_edges.Size(); i++)
      {
         J[i] = group_edges[i];
      }
      if (Dim==4)
      {
         J = J_group_splan+I_group_splan[group];
         for (i = 0; i < group_planars.Size(); i++)
         {
            J[i] = group_planars[i];
         }
      }
      if (Dim == 3 || Dim == 4)
      {
         J = J_group_sface+I_group_sface[group];
         for (i = 0; i < group_faces.Size(); i++)
         {
            J[i] = group_faces[i];
         }
      }
   }

   // Fix the local numbers of shared edges and faces
   {
      DSTable new_v_to_v(NumOfVertices);
      GetVertexToVertexTable(new_v_to_v);
      for (i = 0; i < shared_edges.Size(); i++)
      {
         v = shared_edges[i]->GetVertices();
         sedge_ledge[i] = new_v_to_v(v[0], v[1]);
      }
   }
   if (Dim == 3)
   {
      STable3D *faces_tbl = GetElementToFaceTable(1);
      for (i = 0; i < shared_faces.Size(); i++)
      {
         v = shared_faces[i]->GetVertices();
         sface_lface[i] = (*faces_tbl)(v[0], v[1], v[2]);
      }
      delete faces_tbl;
   }
   else if (Dim ==4)
   {
      STable4D *faces_tbl = GetElementToFaceTable4D(1);
      for (i = 0; i < shared_faces.Size(); i++)
      {
         v = shared_faces[i]->GetVertices();
         sface_lface[i] = (*faces_tbl)(v[0], v[1], v[2], v[3]);
      }
      delete faces_tbl;

      STable3D *plan_tbl = GetElementToPlanarTable(1);
      for (i = 0; i < shared_planars.Size(); i++)
      {
         v = shared_planars[i]->GetVertices();
         splan_lplan[i] = (*plan_tbl)(v[0], v[1], v[2]);
      }
      delete plan_tbl;
   }

   group_svert.SetIJ(I_group_svert, J_group_svert);
   group_sedge.SetIJ(I_group_sedge, J_group_sedge);
   if (Dim==4) { group_splan.SetIJ(I_group_splan, J_group_splan); }
   if (Dim == 3 || Dim == 4)
   {
      group_sface.SetIJ(I_group_sface, J_group_sface);
   }
}

void ParMesh::QuadUniformRefinement()
{
   DeleteFaceNbrData();

   int oedge = NumOfVertices;

   // call Mesh::QuadUniformRefinement so that it won't update the nodes
   {
      GridFunction *nodes = Nodes;
      Nodes = NULL;
      Mesh::QuadUniformRefinement();
      Nodes = nodes;
   }

   // update the groups
   {
      int i, attr, ind, *v;

      int group;
      Array<int> sverts, sedges;

      int *I_group_svert, *J_group_svert;
      int *I_group_sedge, *J_group_sedge;

      I_group_svert = new int[GetNGroups()+1];
      I_group_sedge = new int[GetNGroups()+1];

      I_group_svert[0] = I_group_svert[1] = 0;
      I_group_sedge[0] = I_group_sedge[1] = 0;

      // compute the size of the J arrays
      J_group_svert = new int[group_svert.Size_of_connections()
                              + group_sedge.Size_of_connections()];
      J_group_sedge = new int[2*group_sedge.Size_of_connections()];

      for (group = 0; group < GetNGroups()-1; group++)
      {
         // Get the group shared objects
         group_svert.GetRow(group, sverts);
         group_sedge.GetRow(group, sedges);

         // Process all the edges
         for (i = 0; i < group_sedge.RowSize(group); i++)
         {
            v = shared_edges[sedges[i]]->GetVertices();
            ind = oedge + sedge_ledge[sedges[i]];
            // add a vertex
            sverts.Append(svert_lvert.Append(ind)-1);
            // update the edges
            attr = shared_edges[sedges[i]]->GetAttribute();
            shared_edges.Append(new Segment(v[1], ind, attr));
            sedges.Append(sedge_ledge.Append(-1)-1);
            v[1] = ind;
         }

         I_group_svert[group+1] = I_group_svert[group] + sverts.Size();
         I_group_sedge[group+1] = I_group_sedge[group] + sedges.Size();

         int *J;
         J = J_group_svert+I_group_svert[group];
         for (i = 0; i < sverts.Size(); i++)
         {
            J[i] = sverts[i];
         }
         J = J_group_sedge+I_group_sedge[group];
         for (i = 0; i < sedges.Size(); i++)
         {
            J[i] = sedges[i];
         }
      }

      // Fix the local numbers of shared edges
      DSTable v_to_v(NumOfVertices);
      GetVertexToVertexTable(v_to_v);
      for (i = 0; i < shared_edges.Size(); i++)
      {
         v = shared_edges[i]->GetVertices();
         sedge_ledge[i] = v_to_v(v[0], v[1]);
      }

      group_svert.SetIJ(I_group_svert, J_group_svert);
      group_sedge.SetIJ(I_group_sedge, J_group_sedge);
   }

   UpdateNodes();
}

void ParMesh::HexUniformRefinement()
{
   DeleteFaceNbrData();

   int oedge = NumOfVertices;
   int oface = oedge + NumOfEdges;

   DSTable v_to_v(NumOfVertices);
   GetVertexToVertexTable(v_to_v);
   STable3D *faces_tbl = GetFacesTable();

   // call Mesh::HexUniformRefinement so that it won't update the nodes
   {
      GridFunction *nodes = Nodes;
      Nodes = NULL;
      Mesh::HexUniformRefinement();
      Nodes = nodes;
   }

   // update the groups
   {
      int i, attr, newv[4], ind, m[5];
      Array<int> v;

      int group;
      Array<int> group_verts, group_edges, group_faces;

      int *I_group_svert, *J_group_svert;
      int *I_group_sedge, *J_group_sedge;
      int *I_group_sface, *J_group_sface;

#if 0
      I_group_svert = new int[GetNGroups()+1];
      I_group_sedge = new int[GetNGroups()+1];
      I_group_sface = new int[GetNGroups()+1];

      I_group_svert[0] = I_group_svert[1] = 0;
      I_group_sedge[0] = I_group_sedge[1] = 0;
      I_group_sface[0] = I_group_sface[1] = 0;
#else
      I_group_svert = new int[GetNGroups()];
      I_group_sedge = new int[GetNGroups()];
      I_group_sface = new int[GetNGroups()];

      I_group_svert[0] = 0;
      I_group_sedge[0] = 0;
      I_group_sface[0] = 0;
#endif

      // compute the size of the J arrays
      J_group_svert = new int[group_svert.Size_of_connections()
                              + group_sedge.Size_of_connections()
                              + group_sface.Size_of_connections()];
      J_group_sedge = new int[2*group_sedge.Size_of_connections()
                              + 4*group_sface.Size_of_connections()];
      J_group_sface = new int[4*group_sface.Size_of_connections()];

      for (group = 0; group < GetNGroups()-1; group++)
      {
         // Get the group shared objects
         group_svert.GetRow(group, group_verts);
         group_sedge.GetRow(group, group_edges);
         group_sface.GetRow(group, group_faces);

         // Process the edges that have been refined
         for (i = 0; i < group_sedge.RowSize(group); i++)
         {
            shared_edges[group_edges[i]]->GetVertices(v);
            ind = oedge + v_to_v(v[0], v[1]);
            // add a vertex
            group_verts.Append(svert_lvert.Append(ind)-1);
            // update the edges
            attr = shared_edges[group_edges[i]]->GetAttribute();
            shared_edges.Append(new Segment(v[1], ind, attr));
            group_edges.Append(sedge_ledge.Append(-1)-1);
            newv[0] = v[0]; newv[1] = ind;
            shared_edges[group_edges[i]]->SetVertices(newv);
         }

         // Process the faces that have been refined
         for (i = 0; i < group_sface.RowSize(group); i++)
         {
            shared_faces[group_faces[i]]->GetVertices(v);
            m[0] = oface+(*faces_tbl)(v[0], v[1], v[2], v[3]);
            // add a vertex
            group_verts.Append(svert_lvert.Append(m[0])-1);
            // add the refinement edges
            attr = shared_faces[group_faces[i]]->GetAttribute();
            m[1] = oedge + v_to_v(v[0], v[1]);
            m[2] = oedge + v_to_v(v[1], v[2]);
            m[3] = oedge + v_to_v(v[2], v[3]);
            m[4] = oedge + v_to_v(v[3], v[0]);
            shared_edges.Append(new Segment(m[1], m[0], attr));
            group_edges.Append(sedge_ledge.Append(-1)-1);
            shared_edges.Append(new Segment(m[2], m[0], attr));
            group_edges.Append(sedge_ledge.Append(-1)-1);
            shared_edges.Append(new Segment(m[3], m[0], attr));
            group_edges.Append(sedge_ledge.Append(-1)-1);
            shared_edges.Append(new Segment(m[4], m[0], attr));
            group_edges.Append(sedge_ledge.Append(-1)-1);
            // update faces
            newv[0] = v[0]; newv[1] = m[1]; newv[2] = m[0]; newv[3] = m[4];
            shared_faces[group_faces[i]]->SetVertices(newv);
            shared_faces.Append(new Quadrilateral(m[1],v[1],m[2],m[0],attr));
            group_faces.Append(sface_lface.Append(-1)-1);
            shared_faces.Append(new Quadrilateral(m[0],m[2],v[2],m[3],attr));
            group_faces.Append(sface_lface.Append(-1)-1);
            shared_faces.Append(new Quadrilateral(m[4],m[0],m[3],v[3],attr));
            group_faces.Append(sface_lface.Append(-1)-1);
         }

         I_group_svert[group+1] = I_group_svert[group] + group_verts.Size();
         I_group_sedge[group+1] = I_group_sedge[group] + group_edges.Size();
         I_group_sface[group+1] = I_group_sface[group] + group_faces.Size();

         int *J;
         J = J_group_svert+I_group_svert[group];
         for (i = 0; i < group_verts.Size(); i++)
         {
            J[i] = group_verts[i];
         }
         J = J_group_sedge+I_group_sedge[group];
         for (i = 0; i < group_edges.Size(); i++)
         {
            J[i] = group_edges[i];
         }
         J = J_group_sface+I_group_sface[group];
         for (i = 0; i < group_faces.Size(); i++)
         {
            J[i] = group_faces[i];
         }
      }

      // Fix the local numbers of shared edges and faces
      DSTable new_v_to_v(NumOfVertices);
      GetVertexToVertexTable(new_v_to_v);
      for (i = 0; i < shared_edges.Size(); i++)
      {
         shared_edges[i]->GetVertices(v);
         sedge_ledge[i] = new_v_to_v(v[0], v[1]);
      }

      delete faces_tbl;
      faces_tbl = GetFacesTable();
      for (i = 0; i < shared_faces.Size(); i++)
      {
         shared_faces[i]->GetVertices(v);
         sface_lface[i] = (*faces_tbl)(v[0], v[1], v[2], v[3]);
      }
      delete faces_tbl;

      group_svert.SetIJ(I_group_svert, J_group_svert);
      group_sedge.SetIJ(I_group_sedge, J_group_sedge);
      group_sface.SetIJ(I_group_sface, J_group_sface);
   }

   UpdateNodes();
}

void ParMesh::NURBSUniformRefinement()
{
   if (MyRank == 0)
   {
      cout << "\nParMesh::NURBSUniformRefinement : Not supported yet!\n";
   }
}

void ParMesh::PrintXG(std::ostream &out) const
{
   MFEM_ASSERT(Dim == spaceDim, "2D manifolds not supported");
   if (Dim == 3 && meshgen == 1)
   {
      int i, j, nv;
      const int *ind;

      out << "NETGEN_Neutral_Format\n";
      // print the vertices
      out << NumOfVertices << '\n';
      for (i = 0; i < NumOfVertices; i++)
      {
         for (j = 0; j < Dim; j++)
         {
            out << " " << vertices[i](j);
         }
         out << '\n';
      }

      // print the elements
      out << NumOfElements << '\n';
      for (i = 0; i < NumOfElements; i++)
      {
         nv = elements[i]->GetNVertices();
         ind = elements[i]->GetVertices();
         out << elements[i]->GetAttribute();
         for (j = 0; j < nv; j++)
         {
            out << " " << ind[j]+1;
         }
         out << '\n';
      }

      // print the boundary + shared faces information
      out << NumOfBdrElements + shared_faces.Size() << '\n';
      // boundary
      for (i = 0; i < NumOfBdrElements; i++)
      {
         nv = boundary[i]->GetNVertices();
         ind = boundary[i]->GetVertices();
         out << boundary[i]->GetAttribute();
         for (j = 0; j < nv; j++)
         {
            out << " " << ind[j]+1;
         }
         out << '\n';
      }
      // shared faces
      for (i = 0; i < shared_faces.Size(); i++)
      {
         nv = shared_faces[i]->GetNVertices();
         ind = shared_faces[i]->GetVertices();
         out << shared_faces[i]->GetAttribute();
         for (j = 0; j < nv; j++)
         {
            out << " " << ind[j]+1;
         }
         out << '\n';
      }
   }

   if (Dim == 3 && meshgen == 2)
   {
      int i, j, nv;
      const int *ind;

      out << "TrueGrid\n"
          << "1 " << NumOfVertices << " " << NumOfElements << " 0 0 0 0 0 0 0\n"
          << "0 0 0 1 0 0 0 0 0 0 0\n"
          << "0 0 " << NumOfBdrElements+shared_faces.Size()
          << " 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
          << "0.0 0.0 0.0 0 0 0.0 0.0 0 0.0\n"
          << "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n";

      // print the vertices
      for (i = 0; i < NumOfVertices; i++)
         out << i+1 << " 0.0 " << vertices[i](0) << " " << vertices[i](1)
             << " " << vertices[i](2) << " 0.0\n";

      // print the elements
      for (i = 0; i < NumOfElements; i++)
      {
         nv = elements[i]->GetNVertices();
         ind = elements[i]->GetVertices();
         out << i+1 << " " << elements[i]->GetAttribute();
         for (j = 0; j < nv; j++)
         {
            out << " " << ind[j]+1;
         }
         out << '\n';
      }

      // print the boundary information
      for (i = 0; i < NumOfBdrElements; i++)
      {
         nv = boundary[i]->GetNVertices();
         ind = boundary[i]->GetVertices();
         out << boundary[i]->GetAttribute();
         for (j = 0; j < nv; j++)
         {
            out << " " << ind[j]+1;
         }
         out << " 1.0 1.0 1.0 1.0\n";
      }

      // print the shared faces information
      for (i = 0; i < shared_faces.Size(); i++)
      {
         nv = shared_faces[i]->GetNVertices();
         ind = shared_faces[i]->GetVertices();
         out << shared_faces[i]->GetAttribute();
         for (j = 0; j < nv; j++)
         {
            out << " " << ind[j]+1;
         }
         out << " 1.0 1.0 1.0 1.0\n";
      }
   }

   if (Dim == 2)
   {
      int i, j, attr;
      Array<int> v;

      out << "areamesh2\n\n";

      // print the boundary + shared edges information
      out << NumOfBdrElements + shared_edges.Size() << '\n';
      // boundary
      for (i = 0; i < NumOfBdrElements; i++)
      {
         attr = boundary[i]->GetAttribute();
         boundary[i]->GetVertices(v);
         out << attr << "     ";
         for (j = 0; j < v.Size(); j++)
         {
            out << v[j] + 1 << "   ";
         }
         out << '\n';
      }
      // shared edges
      for (i = 0; i < shared_edges.Size(); i++)
      {
         attr = shared_edges[i]->GetAttribute();
         shared_edges[i]->GetVertices(v);
         out << attr << "     ";
         for (j = 0; j < v.Size(); j++)
         {
            out << v[j] + 1 << "   ";
         }
         out << '\n';
      }

      // print the elements
      out << NumOfElements << '\n';
      for (i = 0; i < NumOfElements; i++)
      {
         attr = elements[i]->GetAttribute();
         elements[i]->GetVertices(v);

         out << attr << "   ";
         if ((j = GetElementType(i)) == Element::TRIANGLE)
         {
            out << 3 << "   ";
         }
         else if (j == Element::QUADRILATERAL)
         {
            out << 4 << "   ";
         }
         else if (j == Element::SEGMENT)
         {
            out << 2 << "   ";
         }
         for (j = 0; j < v.Size(); j++)
         {
            out << v[j] + 1 << "  ";
         }
         out << '\n';
      }

      // print the vertices
      out << NumOfVertices << '\n';
      for (i = 0; i < NumOfVertices; i++)
      {
         for (j = 0; j < Dim; j++)
         {
            out << vertices[i](j) << " ";
         }
         out << '\n';
      }
   }
   out.flush();
}

bool ParMesh::WantSkipSharedMaster(const NCMesh::Master &master) const
{
   // In 2D, this is a workaround for a CPU boundary rendering artifact. We need
   // to skip a shared master edge if one of its slaves has the same rank.

   const NCMesh::NCList &list = pncmesh->GetEdgeList();
   for (int i = master.slaves_begin; i < master.slaves_end; i++)
   {
      if (!pncmesh->IsGhost(1, list.slaves[i].index)) { return true; }
   }
   return false;
}

void ParMesh::Print(std::ostream &out) const
{
   bool print_shared = true;
   int i, j, shared_bdr_attr;
   Array<int> nc_shared_faces;

   if (NURBSext)
   {
      Printer(out); // does not print shared boundary
      return;
   }

   const Array<int>* s2l_face;
   if (!pncmesh)
   {
      s2l_face = ((Dim == 1) ? &svert_lvert :
                  ((Dim == 2) ? &sedge_ledge : &sface_lface));
   }
   else
   {
      s2l_face = &nc_shared_faces;
      if (Dim >= 2)
      {
         // get a list of all shared non-ghost faces
         const NCMesh::NCList& sfaces =
            (Dim == 3) ? pncmesh->GetSharedFaces() : pncmesh->GetSharedEdges();
         const int nfaces = GetNumFaces();
         for (unsigned i = 0; i < sfaces.conforming.size(); i++)
         {
            int index = sfaces.conforming[i].index;
            if (index < nfaces) { nc_shared_faces.Append(index); }
         }
         for (unsigned i = 0; i < sfaces.masters.size(); i++)
         {
            if (Dim == 2 && WantSkipSharedMaster(sfaces.masters[i])) { continue; }
            int index = sfaces.masters[i].index;
            if (index < nfaces) { nc_shared_faces.Append(index); }
         }
         for (unsigned i = 0; i < sfaces.slaves.size(); i++)
         {
            int index = sfaces.slaves[i].index;
            if (index < nfaces) { nc_shared_faces.Append(index); }
         }
      }
   }

   out << "MFEM mesh v1.0\n";

   // optional
   out <<
       "\n#\n# MFEM Geometry Types (see mesh/geom.hpp):\n#\n"
       "# POINT       = 0\n"
       "# SEGMENT     = 1\n"
       "# TRIANGLE    = 2\n"
       "# SQUARE      = 3\n"
       "# TETRAHEDRON = 4\n"
       "# CUBE        = 5\n"
       "#\n";

   out << "\ndimension\n" << Dim
       << "\n\nelements\n" << NumOfElements << '\n';
   for (i = 0; i < NumOfElements; i++)
   {
      PrintElement(elements[i], out);
   }

   int num_bdr_elems = NumOfBdrElements;
   if (print_shared && Dim > 1)
   {
      num_bdr_elems += s2l_face->Size();
   }
   out << "\nboundary\n" << num_bdr_elems << '\n';
   for (i = 0; i < NumOfBdrElements; i++)
   {
      PrintElement(boundary[i], out);
   }

   if (print_shared && Dim > 1)
   {
      if (bdr_attributes.Size())
      {
         shared_bdr_attr = bdr_attributes.Max() + MyRank + 1;
      }
      else
      {
         shared_bdr_attr = MyRank + 1;
      }
      for (i = 0; i < s2l_face->Size(); i++)
      {
         // Modify the attributes of the faces (not used otherwise?)
         faces[(*s2l_face)[i]]->SetAttribute(shared_bdr_attr);
         PrintElement(faces[(*s2l_face)[i]], out);
      }
   }
   out << "\nvertices\n" << NumOfVertices << '\n';
   if (Nodes == NULL)
   {
      out << spaceDim << '\n';
      for (i = 0; i < NumOfVertices; i++)
      {
         out << vertices[i](0);
         for (j = 1; j < spaceDim; j++)
         {
            out << ' ' << vertices[i](j);
         }
         out << '\n';
      }
      out.flush();
   }
   else
   {
      out << "\nnodes\n";
      Nodes->Save(out);
   }
}

static void dump_element(const Element* elem, Array<int> &data)
{
   data.Append(elem->GetGeometryType());

   int nv = elem->GetNVertices();
   const int *v = elem->GetVertices();
   for (int i = 0; i < nv; i++)
   {
      data.Append(v[i]);
   }
}

void ParMesh::PrintAsOne(std::ostream &out)
{
   int i, j, k, p, nv_ne[2], &nv = nv_ne[0], &ne = nv_ne[1], vc;
   const int *v;
   MPI_Status status;
   Array<double> vert;
   Array<int> ints;

   if (MyRank == 0)
   {
      out << "MFEM mesh v1.0\n";

      // optional
      out <<
          "\n#\n# MFEM Geometry Types (see mesh/geom.hpp):\n#\n"
          "# POINT       = 0\n"
          "# SEGMENT     = 1\n"
          "# TRIANGLE    = 2\n"
          "# SQUARE      = 3\n"
          "# TETRAHEDRON = 4\n"
          "# CUBE        = 5\n"
          "#\n";

      out << "\ndimension\n" << Dim;
   }

   nv = NumOfElements;
   MPI_Reduce(&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
   if (MyRank == 0)
   {
      out << "\n\nelements\n" << ne << '\n';
      for (i = 0; i < NumOfElements; i++)
      {
         // processor number + 1 as attribute and geometry type
         out << 1 << ' ' << elements[i]->GetGeometryType();
         // vertices
         nv = elements[i]->GetNVertices();
         v  = elements[i]->GetVertices();
         for (j = 0; j < nv; j++)
         {
            out << ' ' << v[j];
         }
         out << '\n';
      }
      vc = NumOfVertices;
      for (p = 1; p < NRanks; p++)
      {
         MPI_Recv(nv_ne, 2, MPI_INT, p, 444, MyComm, &status);
         ints.SetSize(ne);
         if (ne)
         {
            MPI_Recv(&ints[0], ne, MPI_INT, p, 445, MyComm, &status);
         }
         for (i = 0; i < ne; )
         {
            // processor number + 1 as attribute and geometry type
            out << p+1 << ' ' << ints[i];
            // vertices
            k = Geometries.GetVertices(ints[i++])->GetNPoints();
            for (j = 0; j < k; j++)
            {
               out << ' ' << vc + ints[i++];
            }
            out << '\n';
         }
         vc += nv;
      }
   }
   else
   {
      // for each element send its geometry type and its vertices
      ne = 0;
      for (i = 0; i < NumOfElements; i++)
      {
         ne += 1 + elements[i]->GetNVertices();
      }
      nv = NumOfVertices;
      MPI_Send(nv_ne, 2, MPI_INT, 0, 444, MyComm);

      ints.Reserve(ne);
      ints.SetSize(0);
      for (i = 0; i < NumOfElements; i++)
      {
         dump_element(elements[i], ints);
      }
      MFEM_ASSERT(ints.Size() == ne, "");
      if (ne)
      {
         MPI_Send(&ints[0], ne, MPI_INT, 0, 445, MyComm);
      }
   }

   // boundary + shared boundary
   ne = NumOfBdrElements;
   if (!pncmesh)
   {
      ne += ((Dim == 2) ? shared_edges : shared_faces).Size();
   }
   else if (Dim > 1)
   {
      const NCMesh::NCList &list = pncmesh->GetSharedList(Dim - 1);
      ne += list.conforming.size() + list.masters.size() + list.slaves.size();
   }
   ints.Reserve(ne * (1 + 2*(Dim-1))); // just an upper bound
   ints.SetSize(0);

   // for each boundary and shared boundary element send its geometry type
   // and its vertices
   ne = 0;
   for (i = j = 0; i < NumOfBdrElements; i++)
   {
      dump_element(boundary[i], ints); ne++;
   }
   if (!pncmesh)
   {
      Array<Element*> &shared = (Dim == 2) ? shared_edges : shared_faces;
      for (i = 0; i < shared.Size(); i++)
      {
         dump_element(shared[i], ints); ne++;
      }
   }
   else if (Dim > 1)
   {
      const NCMesh::NCList &list = pncmesh->GetSharedList(Dim - 1);
      const int nfaces = GetNumFaces();
      for (i = 0; i < (int) list.conforming.size(); i++)
      {
         int index = list.conforming[i].index;
         if (index < nfaces) { dump_element(faces[index], ints); ne++; }
      }
      for (i = 0; i < (int) list.masters.size(); i++)
      {
         int index = list.masters[i].index;
         if (index < nfaces) { dump_element(faces[index], ints); ne++; }
      }
      for (i = 0; i < (int) list.slaves.size(); i++)
      {
         int index = list.slaves[i].index;
         if (index < nfaces) { dump_element(faces[index], ints); ne++; }
      }
   }

   MPI_Reduce(&ne, &k, 1, MPI_INT, MPI_SUM, 0, MyComm);
   if (MyRank == 0)
   {
      out << "\nboundary\n" << k << '\n';
      vc = 0;
      for (p = 0; p < NRanks; p++)
      {
         if (p)
         {
            MPI_Recv(nv_ne, 2, MPI_INT, p, 446, MyComm, &status);
            ints.SetSize(ne);
            if (ne)
            {
               MPI_Recv(ints.GetData(), ne, MPI_INT, p, 447, MyComm, &status);
            }
         }
         else
         {
            ne = ints.Size();
            nv = NumOfVertices;
         }
         for (i = 0; i < ne; )
         {
            // processor number + 1 as bdr. attr. and bdr. geometry type
            out << p+1 << ' ' << ints[i];
            k = Geometries.GetVertices(ints[i++])->GetNPoints();
            // vertices
            for (j = 0; j < k; j++)
            {
               out << ' ' << vc + ints[i++];
            }
            out << '\n';
         }
         vc += nv;
      }
   }
   else
   {
      nv = NumOfVertices;
      ne = ints.Size();
      MPI_Send(nv_ne, 2, MPI_INT, 0, 446, MyComm);
      if (ne)
      {
         MPI_Send(ints.GetData(), ne, MPI_INT, 0, 447, MyComm);
      }
   }

   // vertices / nodes
   MPI_Reduce(&NumOfVertices, &nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
   if (MyRank == 0)
   {
      out << "\nvertices\n" << nv << '\n';
   }
   if (Nodes == NULL)
   {
      if (MyRank == 0)
      {
         out << spaceDim << '\n';
         for (i = 0; i < NumOfVertices; i++)
         {
            out << vertices[i](0);
            for (j = 1; j < spaceDim; j++)
            {
               out << ' ' << vertices[i](j);
            }
            out << '\n';
         }
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv(&nv, 1, MPI_INT, p, 448, MyComm, &status);
            vert.SetSize(nv*spaceDim);
            if (nv)
            {
               MPI_Recv(&vert[0], nv*spaceDim, MPI_DOUBLE, p, 449, MyComm, &status);
            }
            for (i = 0; i < nv; i++)
            {
               out << vert[i*spaceDim];
               for (j = 1; j < spaceDim; j++)
               {
                  out << ' ' << vert[i*spaceDim+j];
               }
               out << '\n';
            }
         }
         out.flush();
      }
      else
      {
         MPI_Send(&NumOfVertices, 1, MPI_INT, 0, 448, MyComm);
         vert.SetSize(NumOfVertices*spaceDim);
         for (i = 0; i < NumOfVertices; i++)
         {
            for (j = 0; j < spaceDim; j++)
            {
               vert[i*spaceDim+j] = vertices[i](j);
            }
         }
         if (NumOfVertices)
         {
            MPI_Send(&vert[0], NumOfVertices*spaceDim, MPI_DOUBLE, 0, 449, MyComm);
         }
      }
   }
   else
   {
      if (MyRank == 0)
      {
         out << "\nnodes\n";
      }
      ParGridFunction *pnodes = dynamic_cast<ParGridFunction *>(Nodes);
      if (pnodes)
      {
         pnodes->SaveAsOne(out);
      }
      else
      {
         ParFiniteElementSpace *pfes =
            dynamic_cast<ParFiniteElementSpace *>(Nodes->FESpace());
         if (pfes)
         {
            // create a wrapper ParGridFunction
            ParGridFunction ParNodes(pfes, Nodes);
            ParNodes.SaveAsOne(out);
         }
         else
         {
            mfem_error("ParMesh::PrintAsOne : Nodes have no parallel info!");
         }
      }
   }
}

void ParMesh::PrintAsOneXG(std::ostream &out)
{
   MFEM_ASSERT(Dim == spaceDim, "2D Manifolds not supported.");
   if (Dim == 3 && meshgen == 1)
   {
      int i, j, k, nv, ne, p;
      const int *ind, *v;
      MPI_Status status;
      Array<double> vert;
      Array<int> ints;

      if (MyRank == 0)
      {
         out << "NETGEN_Neutral_Format\n";
         // print the vertices
         ne = NumOfVertices;
         MPI_Reduce(&ne, &nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         out << nv << '\n';
         for (i = 0; i < NumOfVertices; i++)
         {
            for (j = 0; j < Dim; j++)
            {
               out << " " << vertices[i](j);
            }
            out << '\n';
         }
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv(&nv, 1, MPI_INT, p, 444, MyComm, &status);
            vert.SetSize(Dim*nv);
            MPI_Recv(&vert[0], Dim*nv, MPI_DOUBLE, p, 445, MyComm, &status);
            for (i = 0; i < nv; i++)
            {
               for (j = 0; j < Dim; j++)
               {
                  out << " " << vert[Dim*i+j];
               }
               out << '\n';
            }
         }

         // print the elements
         nv = NumOfElements;
         MPI_Reduce(&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         out << ne << '\n';
         for (i = 0; i < NumOfElements; i++)
         {
            nv = elements[i]->GetNVertices();
            ind = elements[i]->GetVertices();
            out << 1;
            for (j = 0; j < nv; j++)
            {
               out << " " << ind[j]+1;
            }
            out << '\n';
         }
         k = NumOfVertices;
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv(&nv, 1, MPI_INT, p, 444, MyComm, &status);
            MPI_Recv(&ne, 1, MPI_INT, p, 446, MyComm, &status);
            ints.SetSize(4*ne);
            MPI_Recv(&ints[0], 4*ne, MPI_INT, p, 447, MyComm, &status);
            for (i = 0; i < ne; i++)
            {
               out << p+1;
               for (j = 0; j < 4; j++)
               {
                  out << " " << k+ints[i*4+j]+1;
               }
               out << '\n';
            }
            k += nv;
         }
         // print the boundary + shared faces information
         nv = NumOfBdrElements + shared_faces.Size();
         MPI_Reduce(&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         out << ne << '\n';
         // boundary
         for (i = 0; i < NumOfBdrElements; i++)
         {
            nv = boundary[i]->GetNVertices();
            ind = boundary[i]->GetVertices();
            out << 1;
            for (j = 0; j < nv; j++)
            {
               out << " " << ind[j]+1;
            }
            out << '\n';
         }
         // shared faces
         for (i = 0; i < shared_faces.Size(); i++)
         {
            nv = shared_faces[i]->GetNVertices();
            ind = shared_faces[i]->GetVertices();
            out << 1;
            for (j = 0; j < nv; j++)
            {
               out << " " << ind[j]+1;
            }
            out << '\n';
         }
         k = NumOfVertices;
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv(&nv, 1, MPI_INT, p, 444, MyComm, &status);
            MPI_Recv(&ne, 1, MPI_INT, p, 446, MyComm, &status);
            ints.SetSize(3*ne);
            MPI_Recv(&ints[0], 3*ne, MPI_INT, p, 447, MyComm, &status);
            for (i = 0; i < ne; i++)
            {
               out << p+1;
               for (j = 0; j < 3; j++)
               {
                  out << " " << k+ints[i*3+j]+1;
               }
               out << '\n';
            }
            k += nv;
         }
      }
      else
      {
         ne = NumOfVertices;
         MPI_Reduce(&ne, &nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Send(&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         vert.SetSize(Dim*NumOfVertices);
         for (i = 0; i < NumOfVertices; i++)
            for (j = 0; j < Dim; j++)
            {
               vert[Dim*i+j] = vertices[i](j);
            }
         MPI_Send(&vert[0], Dim*NumOfVertices, MPI_DOUBLE,
                  0, 445, MyComm);
         // elements
         ne = NumOfElements;
         MPI_Reduce(&ne, &nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Send(&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         MPI_Send(&NumOfElements, 1, MPI_INT, 0, 446, MyComm);
         ints.SetSize(NumOfElements*4);
         for (i = 0; i < NumOfElements; i++)
         {
            v = elements[i]->GetVertices();
            for (j = 0; j < 4; j++)
            {
               ints[4*i+j] = v[j];
            }
         }
         MPI_Send(&ints[0], 4*NumOfElements, MPI_INT, 0, 447, MyComm);
         // boundary + shared faces
         nv = NumOfBdrElements + shared_faces.Size();
         MPI_Reduce(&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Send(&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         ne = NumOfBdrElements + shared_faces.Size();
         MPI_Send(&ne, 1, MPI_INT, 0, 446, MyComm);
         ints.SetSize(3*ne);
         for (i = 0; i < NumOfBdrElements; i++)
         {
            v = boundary[i]->GetVertices();
            for (j = 0; j < 3; j++)
            {
               ints[3*i+j] = v[j];
            }
         }
         for ( ; i < ne; i++)
         {
            v = shared_faces[i-NumOfBdrElements]->GetVertices();
            for (j = 0; j < 3; j++)
            {
               ints[3*i+j] = v[j];
            }
         }
         MPI_Send(&ints[0], 3*ne, MPI_INT, 0, 447, MyComm);
      }
   }

   if (Dim == 3 && meshgen == 2)
   {
      int i, j, k, nv, ne, p;
      const int *ind, *v;
      MPI_Status status;
      Array<double> vert;
      Array<int> ints;

      int TG_nv, TG_ne, TG_nbe;

      if (MyRank == 0)
      {
         MPI_Reduce(&NumOfVertices, &TG_nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Reduce(&NumOfElements, &TG_ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         nv = NumOfBdrElements + shared_faces.Size();
         MPI_Reduce(&nv, &TG_nbe, 1, MPI_INT, MPI_SUM, 0, MyComm);

         out << "TrueGrid\n"
             << "1 " << TG_nv << " " << TG_ne << " 0 0 0 0 0 0 0\n"
             << "0 0 0 1 0 0 0 0 0 0 0\n"
             << "0 0 " << TG_nbe << " 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
             << "0.0 0.0 0.0 0 0 0.0 0.0 0 0.0\n"
             << "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n";

         // print the vertices
         nv = TG_nv;
         for (i = 0; i < NumOfVertices; i++)
            out << i+1 << " 0.0 " << vertices[i](0) << " " << vertices[i](1)
                << " " << vertices[i](2) << " 0.0\n";
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv(&nv, 1, MPI_INT, p, 444, MyComm, &status);
            vert.SetSize(Dim*nv);
            MPI_Recv(&vert[0], Dim*nv, MPI_DOUBLE, p, 445, MyComm, &status);
            for (i = 0; i < nv; i++)
               out << i+1 << " 0.0 " << vert[Dim*i] << " " << vert[Dim*i+1]
                   << " " << vert[Dim*i+2] << " 0.0\n";
         }

         // print the elements
         ne = TG_ne;
         for (i = 0; i < NumOfElements; i++)
         {
            nv = elements[i]->GetNVertices();
            ind = elements[i]->GetVertices();
            out << i+1 << " " << 1;
            for (j = 0; j < nv; j++)
            {
               out << " " << ind[j]+1;
            }
            out << '\n';
         }
         k = NumOfVertices;
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv(&nv, 1, MPI_INT, p, 444, MyComm, &status);
            MPI_Recv(&ne, 1, MPI_INT, p, 446, MyComm, &status);
            ints.SetSize(8*ne);
            MPI_Recv(&ints[0], 8*ne, MPI_INT, p, 447, MyComm, &status);
            for (i = 0; i < ne; i++)
            {
               out << i+1 << " " << p+1;
               for (j = 0; j < 8; j++)
               {
                  out << " " << k+ints[i*8+j]+1;
               }
               out << '\n';
            }
            k += nv;
         }

         // print the boundary + shared faces information
         ne = TG_nbe;
         // boundary
         for (i = 0; i < NumOfBdrElements; i++)
         {
            nv = boundary[i]->GetNVertices();
            ind = boundary[i]->GetVertices();
            out << 1;
            for (j = 0; j < nv; j++)
            {
               out << " " << ind[j]+1;
            }
            out << " 1.0 1.0 1.0 1.0\n";
         }
         // shared faces
         for (i = 0; i < shared_faces.Size(); i++)
         {
            nv = shared_faces[i]->GetNVertices();
            ind = shared_faces[i]->GetVertices();
            out << 1;
            for (j = 0; j < nv; j++)
            {
               out << " " << ind[j]+1;
            }
            out << " 1.0 1.0 1.0 1.0\n";
         }
         k = NumOfVertices;
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv(&nv, 1, MPI_INT, p, 444, MyComm, &status);
            MPI_Recv(&ne, 1, MPI_INT, p, 446, MyComm, &status);
            ints.SetSize(4*ne);
            MPI_Recv(&ints[0], 4*ne, MPI_INT, p, 447, MyComm, &status);
            for (i = 0; i < ne; i++)
            {
               out << p+1;
               for (j = 0; j < 4; j++)
               {
                  out << " " << k+ints[i*4+j]+1;
               }
               out << " 1.0 1.0 1.0 1.0\n";
            }
            k += nv;
         }
      }
      else
      {
         MPI_Reduce(&NumOfVertices, &TG_nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Reduce(&NumOfElements, &TG_ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         nv = NumOfBdrElements + shared_faces.Size();
         MPI_Reduce(&nv, &TG_nbe, 1, MPI_INT, MPI_SUM, 0, MyComm);

         MPI_Send(&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         vert.SetSize(Dim*NumOfVertices);
         for (i = 0; i < NumOfVertices; i++)
            for (j = 0; j < Dim; j++)
            {
               vert[Dim*i+j] = vertices[i](j);
            }
         MPI_Send(&vert[0], Dim*NumOfVertices, MPI_DOUBLE, 0, 445, MyComm);
         // elements
         MPI_Send(&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         MPI_Send(&NumOfElements, 1, MPI_INT, 0, 446, MyComm);
         ints.SetSize(NumOfElements*8);
         for (i = 0; i < NumOfElements; i++)
         {
            v = elements[i]->GetVertices();
            for (j = 0; j < 8; j++)
            {
               ints[8*i+j] = v[j];
            }
         }
         MPI_Send(&ints[0], 8*NumOfElements, MPI_INT, 0, 447, MyComm);
         // boundary + shared faces
         MPI_Send(&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         ne = NumOfBdrElements + shared_faces.Size();
         MPI_Send(&ne, 1, MPI_INT, 0, 446, MyComm);
         ints.SetSize(4*ne);
         for (i = 0; i < NumOfBdrElements; i++)
         {
            v = boundary[i]->GetVertices();
            for (j = 0; j < 4; j++)
            {
               ints[4*i+j] = v[j];
            }
         }
         for ( ; i < ne; i++)
         {
            v = shared_faces[i-NumOfBdrElements]->GetVertices();
            for (j = 0; j < 4; j++)
            {
               ints[4*i+j] = v[j];
            }
         }
         MPI_Send(&ints[0], 4*ne, MPI_INT, 0, 447, MyComm);
      }
   }

   if (Dim == 2)
   {
      int i, j, k, attr, nv, ne, p;
      Array<int> v;
      MPI_Status status;
      Array<double> vert;
      Array<int> ints;


      if (MyRank == 0)
      {
         out << "areamesh2\n\n";

         // print the boundary + shared edges information
         nv = NumOfBdrElements + shared_edges.Size();
         MPI_Reduce(&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         out << ne << '\n';
         // boundary
         for (i = 0; i < NumOfBdrElements; i++)
         {
            attr = boundary[i]->GetAttribute();
            boundary[i]->GetVertices(v);
            out << attr << "     ";
            for (j = 0; j < v.Size(); j++)
            {
               out << v[j] + 1 << "   ";
            }
            out << '\n';
         }
         // shared edges
         for (i = 0; i < shared_edges.Size(); i++)
         {
            attr = shared_edges[i]->GetAttribute();
            shared_edges[i]->GetVertices(v);
            out << attr << "     ";
            for (j = 0; j < v.Size(); j++)
            {
               out << v[j] + 1 << "   ";
            }
            out << '\n';
         }
         k = NumOfVertices;
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv(&nv, 1, MPI_INT, p, 444, MyComm, &status);
            MPI_Recv(&ne, 1, MPI_INT, p, 446, MyComm, &status);
            ints.SetSize(2*ne);
            MPI_Recv(&ints[0], 2*ne, MPI_INT, p, 447, MyComm, &status);
            for (i = 0; i < ne; i++)
            {
               out << p+1;
               for (j = 0; j < 2; j++)
               {
                  out << " " << k+ints[i*2+j]+1;
               }
               out << '\n';
            }
            k += nv;
         }

         // print the elements
         nv = NumOfElements;
         MPI_Reduce(&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         out << ne << '\n';
         for (i = 0; i < NumOfElements; i++)
         {
            attr = elements[i]->GetAttribute();
            elements[i]->GetVertices(v);
            out << 1 << "   " << 3 << "   ";
            for (j = 0; j < v.Size(); j++)
            {
               out << v[j] + 1 << "  ";
            }
            out << '\n';
         }
         k = NumOfVertices;
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv(&nv, 1, MPI_INT, p, 444, MyComm, &status);
            MPI_Recv(&ne, 1, MPI_INT, p, 446, MyComm, &status);
            ints.SetSize(3*ne);
            MPI_Recv(&ints[0], 3*ne, MPI_INT, p, 447, MyComm, &status);
            for (i = 0; i < ne; i++)
            {
               out << p+1 << " " << 3;
               for (j = 0; j < 3; j++)
               {
                  out << " " << k+ints[i*3+j]+1;
               }
               out << '\n';
            }
            k += nv;
         }

         // print the vertices
         ne = NumOfVertices;
         MPI_Reduce(&ne, &nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         out << nv << '\n';
         for (i = 0; i < NumOfVertices; i++)
         {
            for (j = 0; j < Dim; j++)
            {
               out << vertices[i](j) << " ";
            }
            out << '\n';
         }
         for (p = 1; p < NRanks; p++)
         {
            MPI_Recv(&nv, 1, MPI_INT, p, 444, MyComm, &status);
            vert.SetSize(Dim*nv);
            MPI_Recv(&vert[0], Dim*nv, MPI_DOUBLE, p, 445, MyComm, &status);
            for (i = 0; i < nv; i++)
            {
               for (j = 0; j < Dim; j++)
               {
                  out << " " << vert[Dim*i+j];
               }
               out << '\n';
            }
         }
      }
      else
      {
         // boundary + shared faces
         nv = NumOfBdrElements + shared_edges.Size();
         MPI_Reduce(&nv, &ne, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Send(&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         ne = NumOfBdrElements + shared_edges.Size();
         MPI_Send(&ne, 1, MPI_INT, 0, 446, MyComm);
         ints.SetSize(2*ne);
         for (i = 0; i < NumOfBdrElements; i++)
         {
            boundary[i]->GetVertices(v);
            for (j = 0; j < 2; j++)
            {
               ints[2*i+j] = v[j];
            }
         }
         for ( ; i < ne; i++)
         {
            shared_edges[i-NumOfBdrElements]->GetVertices(v);
            for (j = 0; j < 2; j++)
            {
               ints[2*i+j] = v[j];
            }
         }
         MPI_Send(&ints[0], 2*ne, MPI_INT, 0, 447, MyComm);
         // elements
         ne = NumOfElements;
         MPI_Reduce(&ne, &nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Send(&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         MPI_Send(&NumOfElements, 1, MPI_INT, 0, 446, MyComm);
         ints.SetSize(NumOfElements*3);
         for (i = 0; i < NumOfElements; i++)
         {
            elements[i]->GetVertices(v);
            for (j = 0; j < 3; j++)
            {
               ints[3*i+j] = v[j];
            }
         }
         MPI_Send(&ints[0], 3*NumOfElements, MPI_INT, 0, 447, MyComm);
         // vertices
         ne = NumOfVertices;
         MPI_Reduce(&ne, &nv, 1, MPI_INT, MPI_SUM, 0, MyComm);
         MPI_Send(&NumOfVertices, 1, MPI_INT, 0, 444, MyComm);
         vert.SetSize(Dim*NumOfVertices);
         for (i = 0; i < NumOfVertices; i++)
            for (j = 0; j < Dim; j++)
            {
               vert[Dim*i+j] = vertices[i](j);
            }
         MPI_Send(&vert[0], Dim*NumOfVertices, MPI_DOUBLE,
                  0, 445, MyComm);
      }
   }
}

void ParMesh::GetBoundingBox(Vector &gp_min, Vector &gp_max, int ref)
{
   int sdim;
   Vector p_min, p_max;

   this->Mesh::GetBoundingBox(p_min, p_max, ref);

   sdim = SpaceDimension();

   gp_min.SetSize(sdim);
   gp_max.SetSize(sdim);

   MPI_Allreduce(p_min.GetData(), gp_min, sdim, MPI_DOUBLE, MPI_MIN, MyComm);
   MPI_Allreduce(p_max.GetData(), gp_max, sdim, MPI_DOUBLE, MPI_MAX, MyComm);
}

void ParMesh::GetCharacteristics(double &gh_min, double &gh_max,
                                 double &gk_min, double &gk_max)
{
   double h_min, h_max, kappa_min, kappa_max;

   this->Mesh::GetCharacteristics(h_min, h_max, kappa_min, kappa_max);

   MPI_Allreduce(&h_min, &gh_min, 1, MPI_DOUBLE, MPI_MIN, MyComm);
   MPI_Allreduce(&h_max, &gh_max, 1, MPI_DOUBLE, MPI_MAX, MyComm);
   MPI_Allreduce(&kappa_min, &gk_min, 1, MPI_DOUBLE, MPI_MIN, MyComm);
   MPI_Allreduce(&kappa_max, &gk_max, 1, MPI_DOUBLE, MPI_MAX, MyComm);
}

void ParMesh::PrintInfo(std::ostream &out)
{
   int i;
   DenseMatrix J(Dim);
   double h_min, h_max, kappa_min, kappa_max, h, kappa;

   if (MyRank == 0)
   {
      out << "Parallel Mesh Stats:" << '\n';
   }

   for (i = 0; i < NumOfElements; i++)
   {
      GetElementJacobian(i, J);
      h = pow(fabs(J.Det()), 1.0/double(Dim));
      kappa = J.CalcSingularvalue(0) / J.CalcSingularvalue(Dim-1);
      if (i == 0)
      {
         h_min = h_max = h;
         kappa_min = kappa_max = kappa;
      }
      else
      {
         if (h < h_min) { h_min = h; }
         if (h > h_max) { h_max = h; }
         if (kappa < kappa_min) { kappa_min = kappa; }
         if (kappa > kappa_max) { kappa_max = kappa; }
      }
   }

   double gh_min, gh_max, gk_min, gk_max;
   MPI_Reduce(&h_min, &gh_min, 1, MPI_DOUBLE, MPI_MIN, 0, MyComm);
   MPI_Reduce(&h_max, &gh_max, 1, MPI_DOUBLE, MPI_MAX, 0, MyComm);
   MPI_Reduce(&kappa_min, &gk_min, 1, MPI_DOUBLE, MPI_MIN, 0, MyComm);
   MPI_Reduce(&kappa_max, &gk_max, 1, MPI_DOUBLE, MPI_MAX, 0, MyComm);

   long long ldata[6]; // vert, edge, planar, face, elem, neighbors;
   long long mindata[6], maxdata[6], sumdata[6];

   // count locally owned vertices, edges, and faces
   ldata[0] = GetNV();
   ldata[1] = GetNEdges();
   ldata[2] = GetNPlanars();
   ldata[3] = GetNFaces();
   ldata[4] = GetNE();
   ldata[5] = gtopo.GetNumNeighbors()-1;
   for (int gr = 1; gr < GetNGroups(); gr++)
      if (!gtopo.IAmMaster(gr)) // we are not the master
      {
         ldata[0] -= group_svert.RowSize(gr-1);
         ldata[1] -= group_sedge.RowSize(gr-1);
         if (Dim == 4) { ldata[2] -= group_splan.RowSize(gr-1); }
         ldata[3] -= group_sface.RowSize(gr-1);
      }

   MPI_Reduce(ldata, mindata, 6, MPI_LONG_LONG, MPI_MIN, 0, MyComm);
   MPI_Reduce(ldata, sumdata, 6, MPI_LONG_LONG, MPI_SUM, 0, MyComm);
   MPI_Reduce(ldata, maxdata, 6, MPI_LONG_LONG, MPI_MAX, 0, MyComm);

   if (MyRank == 0)
   {
      out << '\n'
          << "           "
          << setw(12) << "minimum"
          << setw(12) << "average"
          << setw(12) << "maximum"
          << setw(12) << "total" << '\n';
      out << " vertices  "
          << setw(12) << mindata[0]
          << setw(12) << sumdata[0]/NRanks
          << setw(12) << maxdata[0]
          << setw(12) << sumdata[0] << '\n';
      out << " edges     "
          << setw(12) << mindata[1]
          << setw(12) << sumdata[1]/NRanks
          << setw(12) << maxdata[1]
          << setw(12) << sumdata[1] << '\n';
      if (Dim == 4)
         out << " planars   "
             << setw(12) << mindata[2]
             << setw(12) << sumdata[2]/NRanks
             << setw(12) << maxdata[2]
             << setw(12) << sumdata[2] << '\n';
      if (Dim == 3 || Dim == 4)
         out << " faces     "
             << setw(12) << mindata[3]
             << setw(12) << sumdata[3]/NRanks
             << setw(12) << maxdata[3]
             << setw(12) << sumdata[3] << '\n';
      out << " elements  "
          << setw(12) << mindata[4]
          << setw(12) << sumdata[4]/NRanks
          << setw(12) << maxdata[4]
          << setw(12) << sumdata[4] << '\n';
      out << " neighbors "
          << setw(12) << mindata[5]
          << setw(12) << sumdata[5]/NRanks
          << setw(12) << maxdata[5] << '\n';
      out << '\n'
          << "       "
          << setw(12) << "minimum"
          << setw(12) << "maximum" << '\n';
      out << " h     "
          << setw(12) << gh_min
          << setw(12) << gh_max << '\n';
      out << " kappa "
          << setw(12) << gk_min
          << setw(12) << gk_max << '\n';
      if (Dim==2)
         out << '\n'
             << " Euler number  "
             << setw(12) << sumdata[0]-sumdata[3]+sumdata[4]  << '\n';
      else if (Dim==3)
         out << '\n'
             << " Euler number  "
             << setw(12) << sumdata[0]-sumdata[1]+sumdata[3]-sumdata[4]  << '\n';
      else if (Dim==4)
         out << '\n'
             << " Euler number  "
             << setw(12) << sumdata[0]-sumdata[1]+sumdata[2]-sumdata[3]+sumdata[4]  << '\n';

      out << std::flush;
   }
}

long ParMesh::ReduceInt(int value) const
{
   long local = value, global;
   MPI_Allreduce(&local, &global, 1, MPI_LONG, MPI_SUM, MyComm);
   return global;
}

void ParMesh::ParPrint(ostream &out) const
{
   if (NURBSext || pncmesh)
   {
      // TODO: AMR meshes, NURBS meshes.
      Print(out);
      return;
   }

   // Write out serial mesh.  Tell serial mesh to deliniate the end of it's
   // output with 'mfem_serial_mesh_end' instead of 'mfem_mesh_end', as we will
   // be adding additional parallel mesh information.
   Printer(out, "mfem_serial_mesh_end");

   // write out group topology info.
   gtopo.Save(out);

   out << "\ntotal_shared_vertices " << svert_lvert.Size() << '\n';
   if (Dim >= 2)
   {
      out << "total_shared_edges " << shared_edges.Size() << '\n';
   }
   if (Dim >= 3)
   {
      out << "total_shared_faces " << shared_faces.Size() << '\n';
   }
   for (int gr = 1; gr < GetNGroups(); gr++)
   {
      {
         const int  nv = group_svert.RowSize(gr-1);
         const int *sv = group_svert.GetRow(gr-1);
         out << "\n#group " << gr << "\nshared_vertices " << nv << '\n';
         for (int i = 0; i < nv; i++)
         {
            out << svert_lvert[sv[i]] << '\n';
         }
      }
      if (Dim >= 2)
      {
         const int  ne = group_sedge.RowSize(gr-1);
         const int *se = group_sedge.GetRow(gr-1);
         out << "\nshared_edges " << ne << '\n';
         for (int i = 0; i < ne; i++)
         {
            const int *v = shared_edges[se[i]]->GetVertices();
            out << v[0] << ' ' << v[1] << '\n';
         }
      }
      if (Dim >= 3)
      {
         const int  nf = group_sface.RowSize(gr-1);
         const int *sf = group_sface.GetRow(gr-1);
         out << "\nshared_faces " << nf << '\n';
         for (int i = 0; i < nf; i++)
         {
            PrintElementWithoutAttr(shared_faces[sf[i]], out);
         }
      }
   }

   // Write out section end tag for mesh.
   out << "\nmfem_mesh_end" << endl;
}

ParMesh::~ParMesh()
{
   delete pncmesh;
   ncmesh = pncmesh = NULL;

   DeleteFaceNbrData();

   for (int i = 0; i < shared_faces.Size(); i++)
   {
      FreeElement(shared_faces[i]);
   }
   for (int i = 0; i < shared_planars.Size(); i++)
   {
      FreeElement(shared_planars[i]);
   }
   for (int i = 0; i < shared_edges.Size(); i++)
   {
      FreeElement(shared_edges[i]);
   }

   // The Mesh destructor is called automatically
}

// a copy constructor
ParMeshCyl::ParMeshCyl(ParMeshCyl& pmeshcyl)
    : ParMesh(pmeshcyl), meshbase(pmeshcyl.meshbase), have_slabs_structure(false)
{
    bot_to_top_bels = pmeshcyl.bot_to_top_bels;

    if (pmeshcyl.have_slabs_structure)
    {
        slabs_struct = new Slabs_Structure(*pmeshcyl.slabs_struct);
        have_slabs_structure = true;
    }
}


// parallel version 2
// from a given base mesh (tetrahedral or triangular) produces a space-time mesh for a cylinder
// with thegiven base and Nsteps * tau height in time
// enumeration of space-time vertices: time slab after time slab
// boundary attributes: 1 for t=0, 2 for lateral boundaries, 3 for t = tau*Nsteps
//void ParMesh3DtoParMesh4D (MPI_Comm comm, ParMesh& mesh3d,
//                     ParMesh& mesh4d, double tau, int Nsteps, int bnd_method, int local_method)
ParMeshCyl::ParMeshCyl(MPI_Comm comm, ParMesh& Meshbase, double Tinit, double Tau, int Nsteps,
                       int bnd_method, int local_method, int Nslabs, Array<int>* Slabs_widths)
    : meshbase(Meshbase), bot_to_top_bels(Meshbase.GetNE()), slabs_struct(NULL), have_slabs_structure(false)
{
    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    int dim = meshbase.Dimension() + 1;

    if (meshbase.Dimension() != 3 && meshbase.Dimension() != 2 && myid == 0)
    {
        cout << "Case meshbase dim = " << meshbase.Dimension() << " is not supported "
                                             "in parmesh constructor" << endl << flush;
        return;
    }

    if ( bnd_method != 0 && bnd_method != 1 && myid == 0)
    {
        cout << "Illegal value of bnd_method = " << bnd_method << " (must be 0 or 1)"
             << endl << flush;
        return;
    }
    if ( (local_method < 0 || local_method > 2) && myid == 0)
    {
        cout << "Illegal value of local_method = " << local_method << " (must be 0,1 "
                                                              "or 2)" << endl << flush;
        return;
    }

    if (Slabs_widths)
        have_slabs_structure = true;

    // ****************************************************************************
    // step 1 of 4: creating local space-time part of the mesh from local part of base mesh
    // ****************************************************************************

    if (Nslabs > 1)
    {
        slabs_struct = new Slabs_Structure(Nslabs, Nsteps, Slabs_widths);
        have_slabs_structure = true;
    }

    InitTables();

    // creating local parts of space-time mesh
    MeshSpaceTimeCylinder_onlyArrays(Tinit, Tau, Nsteps, bnd_method, local_method);

    MPI_Barrier(comm);

    // ****************************************************************************
    // step 2 of 4: set additional fields (except the main ones which are
    // shared entities) required for parmesh
    // In particular, set refinement flags in 2D->3D case
    // ****************************************************************************

    MyComm = comm;
    MPI_Comm_size(MyComm, &NRanks);
    MPI_Comm_rank(MyComm, &MyRank);

    gtopo.SetComm(comm);

    int i;

    if (dim == 4)
    {
        BaseGeom = Geometry::PENTATOPE;         // PENTATOPE case only
        BaseBdrGeom = Geometry::TETRAHEDRON;    // PENTATOPE case only
    }
    else //dim == 3
    {
        BaseGeom = Geometry::TETRAHEDRON;       // TETRAHEDRON case only
        BaseBdrGeom = Geometry::TRIANGLE;       // TETRAHEDRON case only
    }

    ncmesh = pncmesh = NULL;

    if( dim == 4)
    {
        swappedElements.SetSize(GetNE());
        DenseMatrix J(4,4);

        for ( i = 0; i < GetNE(); ++i )
        {
            if (elements[i]->GetType() == Element::PENTATOPE)
            {
                int *v = elements[i]->GetVertices();
                Sort5(v[0], v[1], v[2], v[3], v[4]);

                GetElementJacobian(i, J);

                if(J.Det() < 0.0)
                {
                    swappedElements[i] = true;
                    Swap(v);
                }else
                {
                    swappedElements[i] = false;
                }
            }
        }
    }

    meshgen = meshbase.MeshGenerator(); // FIX IT: Not sure at all what it is

    //attributes.Copy(meshbase.attributes);
    //bdr_attributes.Copy(meshbase.bdr_attributes);

    CheckElementOrientation(true);
    if ( dim == 3)
    {
        // FIX IT:
        // version of MarkForRefinement from ParMesh cannot be used here, no gtopo is created so far
        // version of Mesh::MarkForRefinement cannot be used here because it has virtual MarkTetMeshForRefinement
        // which will be overwritten with ParMesh's implementation which cannot be called here for the same reason
        if (meshgen & 1)
        {
           if (Dim == 2)
           {
              MarkTriMeshForRefinement();
           }
           else if (Dim == 3)
           {
              DSTable v_to_v(NumOfVertices);
              GetVertexToVertexTable(v_to_v);
              Mesh::MarkTetMeshForRefinement(v_to_v);
           }
        }
        //MarkForRefinement(); -- was working in mfem 3.2
    }

    NumOfEdges = 0;

    STable3D *faces_tbl_3d = NULL;
    if ( dim == 3 )
        faces_tbl_3d = GetElementToFaceTable(1);

    /*
    STable4D *faces_tbl_4d = NULL;
    if ( dim == 4 )
    {
        faces_tbl_4d = GetElementToFaceTable4D(1);
    }
    */

    //GenerateFaces();

    NumOfPlanars = 0;
    el_to_planar = NULL;

    /*
    STable3D *planar_tbl = NULL;
    if( dim == 4 )
    {
       planar_tbl = GetElementToPlanarTable(1);
       GeneratePlanars();
    }
    */

    /*
    if (NumOfBdrElements == 0 && Dim > 2)
    {
       // in 3D, generate boundary elements before we 'MarkForRefinement'
       if(Dim==3) GetElementToFaceTable();
       else if(Dim==4)
       {
           GetElementToFaceTable4D();
       }
       GenerateFaces();
       GenerateBoundaryElements();
    }
    */

    int curved = 0;
    int generate_edges = 1;

    CheckElementOrientation(true);

    // generate the faces
    if (Dim > 2)
    {
           if(Dim==3) GetElementToFaceTable();
           else if(Dim==4)
           {
               GetElementToFaceTable4D();
           }

           GenerateFaces();

           if(Dim==4)
           {
              ReplaceBoundaryFromFaces();

              GetElementToPlanarTable();
              GeneratePlanars();

 //			 GetElementToQuadTable4D();
 //			 GenerateQuads4D();
           }

       // check and fix boundary element orientation
       if ( !(curved && (meshgen & 1)) )
       {
          CheckBdrElementOrientation();
       }
    }
    else
    {
       NumOfFaces = 0;
    }

    // generate edges if requested
    if (Dim > 1 && generate_edges == 1)
    {
       // el_to_edge may already be allocated (P2 VTK meshes)
       if (!el_to_edge)
       {
           el_to_edge = new Table;
       }
       NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
       if (Dim == 2)
       {
          GenerateFaces(); // 'Faces' in 2D refers to the edges
          if (NumOfBdrElements == 0)
          {
             GenerateBoundaryElements();
          }
          // check and fix boundary element orientation
          if ( !(curved && (meshgen & 1)) )
          {
             CheckBdrElementOrientation();
          }
       }
    }
    else
    {
       NumOfEdges = 0;
    }

    have_face_nbr_data = false;

    // ****************************************************************************
    // step 3 of 4: set parmesh fields for shared entities for mesh4d
    // ****************************************************************************

    ParMeshSpaceTime_createShared( comm, Nsteps );

    // some clean up for unneeded tables

    /*
    if (dim == 4)
    {
        delete faces_tbl_4d;
        //delete planar_tbl;
    }
    */
    if (dim == 3)
        delete faces_tbl_3d;

    // ****************************************************************************
    // step 4 of 4: set internal mesh structure (present in both mesh and
    // parmesh classes
    // ****************************************************************************

    int refine = 1;
    CreateInternalMeshStructure(refine);

    return;
}

void ParMesh::PrintSharedStructParMesh ( int* permutation )
{
    int num_procs, myid;
    MPI_Comm comm = GetComm();
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    if (BaseGeom != Geometry::PENTATOPE && BaseGeom != Geometry::TETRAHEDRON && BaseGeom != Geometry::TRIANGLE)
    {
        if (myid == 0)
            cout << "PrintSharedStructParMesh() is implemented only for pentatops, "
                    "tetrahedrons and triangles" << endl << flush;
        return;
    }

    cout << flush;
    MPI_Barrier(comm);
    if (myid == 0)
        cout << "PrintSharedStructParMesh:" << endl;
    cout << flush;
    MPI_Barrier(comm);


    for ( int proc = 0; proc < num_procs; ++proc )
    {
        if ( proc == myid )
        {
            cout << "I am " << proc << ", parmesh myrank = " << MyRank << endl;
            cout << "myid = " << myid << ", num_procs = " << num_procs << endl;


            if ( Dimension() >= 3 )
            {
                cout << "group_sface" << endl;
                group_sface.Print(cout,10);
            }
            if (Dimension() == 4)
            {
                cout << "group_splan" << endl;
                group_splan.Print(cout,20);
            }
            cout << "group_svert" << endl;
            group_svert.Print();

            for ( int row = 0; row < group_svert.Size(); ++row)
            {
                int rowsize = group_svert.RowSize(row);
                int * rowcols = group_svert.GetRow(row);

                cout << "Row = " << row << endl;
                for ( int col = 0; col < rowsize; ++col)
                {
                    cout << "Vert No." << col << endl;

                    cout << "(";
                    double * vcoords = GetVertex(svert_lvert[rowcols[col]]);
                    for ( int coord = 0; coord < Dimension(); ++coord)
                    {
                        cout << vcoords[coord] << " ";
                    }
                    cout << ")  " << endl;
                    //cout << "rowcols[col] = " << rowcols[col];
                }

            }



            if (Dimension() >= 3)
            {
                cout << "shared_faces" << endl;
                for ( int i = 0; i < shared_faces.Size(); ++i)
                {
                    Element * el = shared_faces[i];
                    int *v = el->GetVertices();
                    if ( !permutation)
                    {
                        for ( int vert = 0; vert < Dimension(); ++vert)
                            cout << v[vert] << " ";
                        cout << endl;
                        //cout << v[0] << " " << v[1] << " " << v[2] << " " << v[3] << endl;
                    }
                    else
                    {
                        for ( int vert = 0; vert < Dimension(); ++vert)
                            cout << permutation[v[vert]] << " ";
                        cout << endl;
                    }
                }



                for ( int row = 0; row < group_sface.Size(); ++row)
                {
                    int rowsize = group_sface.RowSize(row);
                    int * rowcols = group_sface.GetRow(row);

                    cout << "Row = " << row << endl;
                    for ( int col = 0; col < rowsize; ++col)
                    {
                        cout << "Face No." << col << endl;

                        cout << "rowcols[col] = " << rowcols[col] << endl;

                        Element * el = shared_faces[rowcols[col]];
                        int *v = el->GetVertices();

                        for ( int vertno = 0; vertno < el->GetNVertices(); ++vertno)
                        {
                            //simple
                            //cout << v[vertno] << " ";
                            // with coords
                            double * vcoords = GetVertex(v[vertno]);
                            cout << vertno << ": (";
                            for ( int coord = 0; coord < Dimension(); ++coord)
                            {
                                cout << vcoords[coord] << " ";
                            }
                            cout << ")  " << endl;
                        }
                        cout << endl;
                    }

                }

            } // end of priting shared faces

            if (Dimension() == 4)
            {
                cout << "shared_planars" << endl;
                for ( int i = 0; i < shared_planars.Size(); ++i)
                {
                    Element * el = shared_planars[i];
                    int *v = el->GetVertices();
                    if ( !permutation)
                        cout << v[0] << " " << v[1] << " " << v[2] << endl;
                    else
                        cout << permutation[v[0]] << " " <<
                                permutation[v[1]] << " " << permutation[v[2]] << endl;
                }

                for ( int row = 0; row < group_splan.Size(); ++row)
                {
                    int rowsize = group_splan.RowSize(row);
                    int * rowcols = group_splan.GetRow(row);

                    cout << "Row = " << row << endl;
                    for ( int col = 0; col < rowsize; ++col)
                    {
                        cout << "Planar No." << col << endl;

                        cout << "rowcols[col] = " << rowcols[col] << endl;

                        Element * el = shared_planars[rowcols[col]];
                        int *v = el->GetVertices();

                        for ( int vertno = 0; vertno < el->GetNVertices(); ++vertno)
                        {
                            //simple
                            //cout << v[vertno] << " ";
                            // with coords
                            double * vcoords = GetVertex(v[vertno]);
                            cout << vertno << ": (";
                            for ( int coord = 0; coord < Dimension(); ++coord)
                            {
                                cout << vcoords[coord] << " ";
                            }
                            cout << ")  " << endl;
                        }
                        cout << endl;
                    }

                }
            }


            cout << "shared_edges" << endl;
            for ( int i = 0; i < shared_edges.Size(); ++i)
            {
                Element * el = shared_edges[i];
                int *v = el->GetVertices();
                if ( !permutation)
                    cout << v[0] << " " << v[1] << endl;
                else
                    cout << permutation[v[0]] << " " << permutation[v[1]] << endl;
            }
            cout << "sedge_ledge" << endl;
            sedge_ledge.Print();
            cout << "group_sedge" << endl;
            group_sedge.Print(cout, 10);


            //GetEdgeVertexTable(); //this call crashes everything because it changes the edges
            // if you don't delete edge_vertex and set it to NULL afterwards
            //delete edge_vertex;
            //edge_vertex = NULL;

            /*
            if (edge_vertex)
            {
                cout << "I already have edge_vertex" << endl;
                edge_vertex->Print();
            }
            else
                cout << "I don't have here edge_vertex" << endl;


            DSTable v_to_v(NumOfVertices);
            GetVertexToVertexTable(v_to_v);

            int nedges = v_to_v.NumberOfEntries();

            if (!edge_vertex)
            {
                cout << "Creating edge_vertex" << endl;

                edge_vertex = new Table(nedges, 2);


                for (int i = 0; i < NumOfVertices; i++)
                {
                   for (DSTable::RowIterator it(v_to_v, i); !it; ++it)
                   {
                      int j = it.Index();
                      edge_vertex->Push(j, i);
                      edge_vertex->Push(j, it.Column());
                   }
                }

                edge_vertex->Finalize();
                delete edge_vertex;
                edge_vertex = NULL;
            }

            */


            for ( int row = 0; row < group_sedge.Size(); ++row)
            {
                int rowsize = group_sedge.RowSize(row);
                int * rowcols = group_sedge.GetRow(row);

                cout << "Row = " << row << endl;
                for ( int col = 0; col < rowsize; ++col)
                {
                    cout << "Edge No." << col << endl;

                    Array<int> v;
                    GetEdgeVertices(sedge_ledge[rowcols[col]], v);

                    for ( int vertno = 0; vertno < 2; ++vertno)
                    {
                        //simple
                        //cout << v[vertno] << " ";
                        // with coords
                        double * vcoords = GetVertex(v[vertno]);
                        cout << vertno << ": (";
                        for ( int coord = 0; coord < Dim; ++coord)
                        {
                            cout << vcoords[coord] << " ";
                        }
                        cout << ")  " << endl;
                    }
                    cout << endl;
                }

            }
            //if not delete here, get segfault for more than two parallel refinements
            delete edge_vertex;
            edge_vertex = NULL;




            cout << "sface_lface" << endl;
            sface_lface.Print();
            if (Dimension() == 4)
            {
                cout << "splan_lplan" << endl;
                splan_lplan.Print();
            }
            cout << "sedge_ledge" << endl;
            sedge_ledge.Print();
            cout << "svert_lvert" << endl;
            svert_lvert.Print();


            cout << flush;
        }
        MPI_Barrier(comm);
    }

    return;
}

// Creates ParMeshCyl internal structure (including shared entities)
// after the main arrays (elements, vertices and boundary) are already defined for the
// future space-time mesh. Used only inside the ParMeshCyl constructor.
void ParMeshCyl::ParMeshSpaceTime_createShared( MPI_Comm comm, int Nsteps )
{
    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    int DimBase = meshbase.Dimension();
    int Dim = DimBase + 1;
    int vert_per_baseface = meshbase.GetFace(0)->GetNVertices();
    int nv_base = meshbase.GetNV();

    //cout << "vert_per_face =  " << vert_per_face << endl;
    //cout << "vert_per_elembase = " << vert_per_elembase << endl;

    if (DimBase != 2 && DimBase != 3 && myid == 0)
    {
        cout << "Case dimbase = " << DimBase << " is not supported in createShared()"
             << endl << flush;
        return;
    }

    if (BaseGeom != Geometry::PENTATOPE && BaseGeom != Geometry::TETRAHEDRON)
    {
        if (myid == 0)
            cout << "ParMeshSpaceTime_createShared() is implemented only for "
                    "pentatops and tetrahedrons" << endl << flush;
        return;
    }


    ListOfIntegerSets  groups; // this group list will play the same role as "groups"
    //in the ParMesh constructor from MFEM.
    IntegerSet         group;

    // ****************************************************************************
    // step 0 of 4: looking at local part of the base mesh
    // ****************************************************************************

    // ****************************************************************************
    // step 1 of 4: creating temporary arrays needed for shared entities.
    // The arrays created are related to the space-time mesh structure
    // ****************************************************************************

    // creating sets for each kind of shared entities
    // for each shared entity these array store the number of the corresponding group
    // of processors in LOCAL processors numeration
    Array<int> sface_groupbase;
    Array<int> sedge_groupbase;
    Array<int> svert_groupbase;

    // for each shared entity these array store the index of the entity in the
    // corresponding group of processors ~ position in group
    Array<int> sface_posingroupbase;
    Array<int> sedge_posingroupbase;
    Array<int> svert_posingroupbase;

    // temporary shortcuts
    int meshbase_shared_faces_size = meshbase.shared_faces.Size();
    int meshbase_shared_edges_size = meshbase.shared_edges.Size();
    int meshbase_svert_lvert_size = meshbase.svert_lvert.Size();
    int meshbase_group_sface_size = meshbase.group_sface.Size();
    int meshbase_group_sedge_size = meshbase.group_sedge.Size();
    int meshbase_group_svert_size = meshbase.group_svert.Size();

    // maybe an ugly way to get sface_group from group_sface;
    // actually, just manually transposing.
    if (Dim == 4)
    {
        sface_groupbase.SetSize(meshbase_shared_faces_size);
        sface_posingroupbase.SetSize(meshbase_shared_faces_size);
        for ( int row = 0; row < meshbase_group_sface_size; ++row )
        {
            int * v = meshbase.group_sface.GetRow(row);
            for (int colno = 0; colno < meshbase.group_sface.RowSize(row); ++colno)
            {
                sface_groupbase[v[colno]] = row;
                sface_posingroupbase[v[colno]] = colno;
            }
        }
    }
    else //Dim == 3
    {
        sface_groupbase.SetSize(meshbase_shared_edges_size);
        sface_posingroupbase.SetSize(meshbase_shared_edges_size);
        for ( int row = 0; row < meshbase_group_sedge_size; ++row )
        {
            int * v = meshbase.group_sedge.GetRow(row);
            for (int colno = 0; colno < meshbase.group_sedge.RowSize(row); ++colno)
            {
                sface_groupbase[v[colno]] = row;
                sface_posingroupbase[v[colno]] = colno;
            }
        }
    }

    sedge_groupbase.SetSize(meshbase_shared_edges_size);
    sedge_posingroupbase.SetSize(meshbase_shared_edges_size);
    for ( int row = 0; row < meshbase_group_sedge_size; ++row )
    {
        int * v = meshbase.group_sedge.GetRow(row);
        for (int colno = 0; colno < meshbase.group_sedge.RowSize(row); ++colno)
        {
            sedge_groupbase[v[colno]] = row;
            sedge_posingroupbase[v[colno]] = colno;
        }
    }

    svert_groupbase.SetSize(meshbase_svert_lvert_size);
    svert_posingroupbase.SetSize(meshbase_svert_lvert_size);
    for ( int row = 0; row < meshbase_group_svert_size; ++row )
    {
        int * v = meshbase.group_svert.GetRow(row);
        for (int colno = 0; colno < meshbase.group_svert.RowSize(row); ++colno)
        {
            svert_groupbase[v[colno]] = row;
            svert_posingroupbase[v[colno]] = colno;
        }
    }

    // creating maps for each kind of base mesh shared entities

    // map structure from shared entities (faces, edges and vertices)
    // to pairs (group number, position inside the group)
    std::map<set<int>, vector<int> > ShfacesBase;
    std::map<set<int>, vector<int> > ShedgesBase;
    // could be a map<int,int>, but somehow the code gets ugly at some place,
    // around "findproj" stuff.
    std::map<set<int>, vector<int> > ShvertsBase;

    for ( int shvertind = 0; shvertind < meshbase_svert_lvert_size; ++shvertind)
    {
        set<int> buff (meshbase.svert_lvert + shvertind, meshbase.svert_lvert + shvertind + 1 );
        ShvertsBase[buff] = vector<int>{svert_groupbase[shvertind],
                                            svert_posingroupbase[shvertind]};
    }

    for ( int shedgeind = 0; shedgeind < meshbase_shared_edges_size; ++shedgeind)
    {
        Element * shedge = meshbase.shared_edges[shedgeind];

        int * verts = shedge->GetVertices();
        set<int> buff(verts, verts+2);      //edges always have two vertices

        ShedgesBase[buff] = vector<int>{sedge_groupbase[shedgeind],
                                            sedge_posingroupbase[shedgeind]};
    }

    if (Dim == 4)
    {
        for ( int shfaceind = 0; shfaceind < meshbase_shared_faces_size; ++shfaceind)
        {
            Element * shface = meshbase.shared_faces[shfaceind];

            int * verts = shface->GetVertices();
            set<int> buff(verts, verts+vert_per_baseface);

            ShfacesBase[buff] = vector<int>{sface_groupbase[shfaceind],
                    sface_posingroupbase[shfaceind]};
        }
    }
    else // Dim == 3
        ShfacesBase = ShedgesBase; //just a convention, that faces in 2D are the same as edges in these temporary structures


    // actually here we need group_proc, which can be obtained from gtopo.GetGroup combined
    // with converting lproc indices to proc indices using lproc_proc from gtopo.

    Array<int> lproc_proc(meshbase.gtopo.GetNumNeighbors());
    for ( int i = 0; i < lproc_proc.Size(); ++i )
        lproc_proc[i] = meshbase.gtopo.GetNeighborRank(i);

    Table group_proc;
    group_proc.MakeI(meshbase.gtopo.NGroups());
    for ( int row = 0; row < meshbase.gtopo.NGroups(); ++row )
    {
        group_proc.AddColumnsInRow(row, meshbase.gtopo.GetGroupSize(row));
    }
    group_proc.MakeJ();

    int rowsize;
    for ( int row = 0; row < meshbase.gtopo.NGroups(); ++row )
    {
        rowsize = meshbase.gtopo.GetGroupSize(row);
        const int * group = meshbase.gtopo.GetGroup(row);
        int group_with_proc[rowsize];

        for ( int col = 0; col < rowsize; ++col )
        {
            group_with_proc[col] = lproc_proc[group[col]];
        }

        group_proc.AddConnections(row, group_with_proc, rowsize);
    }
    group_proc.ShiftUpI();

    /*
    for ( int proc = 0; proc < num_procs; ++proc )
    {
        if ( proc == myid )
        //if ( proc == 1 && proc == myid )
        {
            cout << "I am " << proc << ", parmesh myrank = " << mesh3d.MyRank << endl;
            //cout << "group_lproc3d" << endl;
            //group_lproc.Print(cout,10);

            cout << "lproc_proc3d" << endl;
            lproc_proc.Print();

            //cout << "proc_lproc3d" << endl;
            //proc_lproc.Print();

            cout << " groups_proc3d " << endl;
            group_proc.Print(cout,10);

            cout << flush;
        }
        MPI_Barrier(comm);
    }
    */

    // ****************************************************************************
    // step 2 of 4: creating groups which will be for the space-time mesh exactly the same as for
    // the base mesh (because we are just extending the existing base parts in time)
    // But one should be careful with processors numeration (global vs local)
    // ****************************************************************************

    for ( int row = 0; row < group_proc.Size(); ++row )
    {
        group.Recreate(group_proc.RowSize(row), group_proc.GetRow(row));
        groups.Insert(group);
    }

    // ****************************************************************************
    // step 3 of 4: creating main parmesh structures for shared entities
    // The main idea is:
    // 1. to loop over the local entities,
    // 2. to project them onto the base (2D or 3D)
    // 3. determine whether the projection is inside the shared 3d entities list
    // 4. change correspondignly the shared 4d entities structure
    // Say, a shared 4d planar (a triangle, basically) will be projected
    // either to a shared 3d face or to a shared 3d edge.
    // ****************************************************************************

    int groupind;

    // 3.1 shared faces 3d -> shared faces 4d (or shared edges 2d -> shared faces 3d)
    // 4d case
    // Nsteps * 3 shared tetrahedrons produced by each shared triangle (shared face 3d)
    // which gives for each time slab a 3d-in-4d prism (which is decomposed into 3 tetrahedrons)
    // as lateral face of a 4d space-time prism cell.
    // 3d case
    // Nsteps * 2 shared triangles produced by each shared edge (~shared face 2D) which gives
    // a space-time rectangle (2D in 3D) which is decomposed into 2 triangles

    int face2Dto3D_coeff = DimBase; // 3 for 4d and 2 for 3d
    if ( Dim == 4 )
        shared_faces.SetSize( Nsteps * face2Dto3D_coeff * meshbase_shared_faces_size);
    else // Dim = 3
        shared_faces.SetSize( Nsteps * face2Dto3D_coeff * meshbase_shared_edges_size);

    sface_lface.SetSize( shared_faces.Size());

    // alternative way to construct group_sface - from I and J arrays manually
    int * group_sface_I, * group_sface_J;
    group_sface_I = new int[group_proc.Size() + 1];

    group_sface_I[0] = 0;
    if (Dim == 4)
        for ( int row = 0; row < group_proc.Size(); ++row )
        {
            group_sface_I[row + 1] = group_sface_I[row] + Nsteps * face2Dto3D_coeff *
                    meshbase.group_sface.RowSize(row);
        }
    else //Dim == 3
        for ( int row = 0; row < group_proc.Size(); ++row )
        {
            // without this if, valgrind reports "Invalid read"
            // because meshbase.group_sedge has size 0 (for serial case)
            if (meshbase.group_sedge.Size() == 0)
                group_sface_I[row + 1] = group_sface_I[row];
            else
                group_sface_I[row + 1] = group_sface_I[row] + Nsteps * face2Dto3D_coeff *
                        meshbase.group_sedge.RowSize(row);
        }

    group_sface_J = new int[group_sface_I[group_proc.Size() - 1]];

    cout << flush;
    MPI_Barrier(comm);


    int cnt_shfaces = 0;

    for ( int faceind = 0; faceind < GetNumFaces(); ++faceind)
    {
        Element * face = faces[faceind];
        int * v = face->GetVertices();

        set<int> faceproj;
        for ( int vert = 0; vert < face->GetNVertices() ; ++vert )
        {
            // assuming  all time slabs have the same number of nodes and
            // no additional vertices are added to the 4d prisms
            faceproj.insert( v[vert] % nv_base);
        }


        auto findproj = ShfacesBase.find(faceproj);
        if (findproj != ShfacesBase.end() )
        {

            sface_lface[cnt_shfaces] = faceind;

            groupind = findproj->second[0] + 1;

            group.Recreate(group_proc.RowSize(groupind), group_proc.GetRow(groupind));

            int temp = groups.Insert(group);


            if (Dim == 4)
            {
                if(getSwappedFaceElementInfo(faceind))
                    Swap(v); // FIX IT. 100% UNSURE about whether it is correct
                shared_faces[cnt_shfaces] = new Tetrahedron(v);
            }
            else // Dim == 3
            {
                // this is from MFEM 3.3. Have not tested with this release at all
                //Tetrahedron *tet = (Tetrahedron *)(elements[faces_info[faceind].Elem1No]);
                //tet->GetMarkedFace(faces_info[faceind].Elem1Inf/64, v);


                Tetrahedron *tet =
                   (Tetrahedron *)(elements[faces_info[faceind].Elem1No]);
                int re[2], type, flag, *tv;
                tet->ParseRefinementFlag(re, type, flag);
                tv = tet->GetVertices();

                switch (faces_info[faceind].Elem1Inf/64)
                {
                   case 0:
                      switch (re[1])
                      {
                         case 1: v[0] = tv[1]; v[1] = tv[2]; v[2] = tv[3];
                            break;
                         case 4: v[0] = tv[3]; v[1] = tv[1]; v[2] = tv[2];
                            break;
                         case 5: v[0] = tv[2]; v[1] = tv[3]; v[2] = tv[1];
                            break;
                      }
                      break;
                   case 1:
                      switch (re[0])
                      {
                         case 2: v[0] = tv[2]; v[1] = tv[0]; v[2] = tv[3];
                            break;
                         case 3: v[0] = tv[0]; v[1] = tv[3]; v[2] = tv[2];
                            break;
                         case 5: v[0] = tv[3]; v[1] = tv[2]; v[2] = tv[0];
                            break;
                      }
                      break;
                   case 2:
                      v[0] = tv[0]; v[1] = tv[1]; v[2] = tv[3];
                      break;
                   case 3:
                      v[0] = tv[1]; v[1] = tv[0]; v[2] = tv[2];
                      break;
                }


                // Here a flip is made for one of the two processes who share the face
                // To fix the things, swap is been made on the process whose rank is larger.
                //cout << "group size = " << group.Size() << endl;
                const Array<int>& groupme = group;
                //groupme.Print();
                if (myid > min(groupme[0], groupme[1]))
                {
                    //cout << "Swap is made, my id = " << myid << endl;
                    Swap(v);
                }


                /*
                 * old way of face vertices reordering which turned out to be inconsistent
                 * with refinement used for tetrahedron case

                for ( int i = 0; i < 3; ++i)
                    vcoords[i] = GetVertex(v[i]);

                sortingPermutation(3, vcoords, ordering);

                if ( permutation_sign(ordering, 3) == -1 ) //possible values are -1 or 1
                    Swap(v);

                */

                shared_faces[cnt_shfaces] = new Triangle(v);
            }

            // computing the local index of one of the tetrahedrons(4d) or triangles(3d)
            // which is projected onto the same face(edge) in 3d(2d):

            int pos;
            int tslab, tslab_localind;

            // time slab which the tetrahedron (triangle) belongs to.
            // It is initialized with the maximum possible value and then defined as a
            // minimum tslab number over all tetrahedronv vertices
            tslab = Nsteps - 1;
            for ( int vert = 0; vert < face->GetNVertices() ; ++vert )
                if (v[vert] / nv_base  < tslab)
                    tslab = v[vert]/nv_base;

            // The order within a time slab is as follows: All tetrahedra(triangles)
            // are one above the other, so 0 goes for the lowest in the timeslab,
            // 1 for the next one, etc...
            int nv_lower = 0; // number of vertices on the lower base of the time slab prism
            for ( int vert = 0; vert < face->GetNVertices() ; ++vert )
            {
                if (v[vert] / nv_base == tslab)
                    nv_lower++;
                else if (v[vert] / nv_base != tslab + 1)
                {
                    cout << "Strange: a vertex is neither on the top nor on the bottom" << endl;
                    cout << "tslab = " << tslab << " ";
                    cout << "v[vert] = " << v[vert] << " ";
                    cout << "nv_base = " << nv_base << endl;
                    cout << flush;
                }

            }

            if (nv_lower < 1 || nv_lower > DimBase)
                cout << "Strange: nv_lower = " << nv_lower << " either too many or"
                              " too few vertices on the lower base" << endl << flush;
            else
            {
                tslab_localind = DimBase - nv_lower;
            }

            pos = findproj->second[1] * Nsteps * face2Dto3D_coeff +
                    tslab * face2Dto3D_coeff + tslab_localind;


            group_sface_J[group_sface_I[temp - 1] + pos] = cnt_shfaces;

            cnt_shfaces++;

        }
    }

    if (cnt_shfaces != shared_faces.Size())
        cout << "Error: smth wrong with the number of shared faces" << endl << flush;

    group_sface.SetIJ(group_sface_I, group_sface_J, group_proc.Size() - 1);


    // 3.2 shared_edges 3d & shared_faces 3d -> shared_planars 4d
    // ...

    int cnt_inface = 0, cnt_inedge = 0;
    if (Dim == 4)
    {
        // Nsteps + 1 triangles as bases for lateral 3d-in-4d prisms for a one 4d space-time prism
        // Nsteps * 2 triangles inside decomposition of each lateral 3d-in-4d prism into
        // tetrahedrons with shared triangle3d (shared face 3d) as the base
        // Nsteps * 2 triangles on the vertical lateral sides of each 3d-in-4d lateral prism
        // for each 4d prism with shared segment3d (shared edge 3d) as the base
        shared_planars.SetSize( (Nsteps * 2 + (Nsteps + 1))*meshbase_shared_faces_size
                                       + Nsteps * 2 * meshbase_shared_edges_size);
        splan_lplan.SetSize( shared_planars.Size());

        //alternative way to construct group_splan - from I and J arrays manually
        int * group_splan_I, * group_splan_J;
        group_splan_I = new int[group_proc.Size() + 1];
        group_splan_I[0] = 0;
        for ( int row = 0; row < group_proc.Size(); ++row )
        {
            group_splan_I[row + 1] = group_splan_I[row] +
                    (Nsteps * 2 + (Nsteps + 1))*meshbase.group_sface.RowSize(row) +
                    Nsteps * 2 * meshbase.group_sedge.RowSize(row);
        }
        group_splan_J = new int[group_splan_I[group_proc.Size() - 1]];

        vector<double *> vcoords(3);
        vector<vector<double> > vcoordsNew(3);
        int ordering[3];
        //int orderingNew[3];

        for ( int planind4d = 0; planind4d < GetNPlanars(); ++planind4d)
        {
            //if (myid == 4)
                //cout << "planind4d =  " << planind4d << " / " << GetNPlanars() << endl;

            Element * plan4d = planars[planind4d];
            int * v = plan4d->GetVertices();

            set<int> planproj;
            for ( int vert = 0; vert < plan4d->GetNVertices() ; ++vert )
            {
                // assuming  all time slabs have the same number of nodes and
                // no additional vertices are added to the 4d prisms
                planproj.insert( v[vert] % nv_base);
            }

            auto findproj_inface = ShfacesBase.find(planproj);
            auto findproj_inedge = ShedgesBase.find(planproj);

            // = 0 for planars projected onto the 3d face and smth for planars
            // projected onto the 3d edge
            int shift;

            if ( findproj_inface != ShfacesBase.end())
            {
                //if ( myid == 4 )
                    //cout << "appending a 4d planar because of 3d face " << endl << flush;

                groupind = findproj_inface->second[0] + 1;
                group.Recreate(group_proc.RowSize(groupind), group_proc.GetRow(groupind));

                int temp = groups.Insert(group);

                // computing the local index of one of the planars which produce
                // the same projection onto 3d which happens to be a shared face 3d:
                // For all time prisms over the shared face 3d (which is a triangle) there are
                // Nsteps + 1 bases of the prisms and 2 *  Nsteps triangles which are 2d in 4d and
                // are between the bases.
                // 0 for the lowest 3d-like base
                // 1,2 for planars above it in the same time slab with 2 and 1 points on the
                // lowest base
                // 3 ~ 0 but in the next time slab
                // etc...
                // Trying to understand, think of all 3d tetrahedrons which decompose a long prism
                // and their faces = planars(trianles) with Nsteps + 1 plane sections.
                // We consider all triangles which are projected onto the base and create
                // a numeration over them.

                shift = 0;

                // time slab which the triangle belongs to. (minimal time slab over all vertices)
                // It is initialized with the maximum possible value and then defined as a minimum
                // tslab number over all tetrahedronv vertices
                int tslab = Nsteps; //the uppermost base will formally be in this time slab
                for ( int vert = 0; vert < plan4d->GetNVertices() ; ++vert )
                    if (v[vert] / nv_base  < tslab)
                        tslab = v[vert]/nv_base;

                // index within a timeslab: 0,1, or 2 based on the following order:
                // 0 for the lower base, 1 ...
                // and 2 for the triangle which is higher than the others in the timeslab
                // because planars are actually one above of the other.
                int tslabprism_lind;
                int nv_lower = 0; // number of vertices on the lwer 3d base of the time slab prism
                for ( int vert = 0; vert < plan4d->GetNVertices() ; ++vert )
                {
                    if (v[vert] / nv_base == tslab)
                        nv_lower++;
                    else if (v[vert] / nv_base != tslab + 1)
                        cout << "Strange face-type planar: a vertex is neither on the"
                                " top nor on the bottom" << endl << flush;
                }

                if (nv_lower > 2)
                    tslabprism_lind = 0;
                else if (nv_lower < 2)
                    tslabprism_lind = 2;
                else if (nv_lower == 2)
                    tslabprism_lind = 1;
                else
                    cout << "Strange face-type planar: nv_lower is not 1,2,3" << endl;

                int pos = shift + findproj_inface->second[1] * (Nsteps * 2 + (Nsteps + 1))
                        + tslab * 3 + tslabprism_lind;

                group_splan_J[group_splan_I[temp - 1] + pos] = cnt_inface + cnt_inedge;

                cnt_inface++;
            }

            if ( findproj_inedge != ShedgesBase.end())
            {
                //if ( myid == 4 )
                    //cout << "appending a 4d planar because of 3d edge " << endl << flush;

                groupind = findproj_inedge->second[0] + 1;
                group.Recreate(group_proc.RowSize(groupind), group_proc.GetRow(groupind));

                int temp = groups.Insert(group);

                // computing the local index of one of the planars which produce
                // the same projection onto 3d which happens to be a shared edge 3d:
                // For all time prisms over the shared face 3d (which is a triangle) there are
                // Nsteps + 1 bases of the prisms and 2 *  Nsteps triangles which are 2d in 4d and
                // are between the bases.
                // 0 for the lowest 3d-like base
                // 1,2 for planars above it in the same time slab with 2 and 1 points
                // on the lowest base
                // 3 ~ 0 but in the next time slab
                // etc...
                // Trying to understand, think of all planars (triangles) which are projected
                // onto a given shared edge = long rectangle (1d + time) with Nsteps + 1 plane
                // sections.
                // We consider all triangles which are projected onto the base-edge and create
                // a numeration over them.

                // first we need to jump over places reserved for planars which are projected
                // onto the shared faces 3d for a given group of processors
                shift = (Nsteps * 2 + (Nsteps + 1))*meshbase.group_sface.RowSize(temp - 1);

                // time slab which the triangle belongs to. (minimal time slab over all vertices)
                // It is initialized with the maximum possible value and then defined as a minimum
                // tslab number over all vertices
                int tslab = Nsteps - 1;
                for ( int vert = 0; vert < plan4d->GetNVertices() ; ++vert )
                    if (v[vert] / nv_base  < tslab)
                        tslab = v[vert]/nv_base;

                // index within a timeslab: 0 or 1 based on the following order:
                // Consider a rectangle in space-time whith shared edge as the base.
                // There is one diagonal which splits it into two triangles. We set:
                // 0 for the lower one (with 2 vertices on the base) and 1 for the other.
                int tslabprism_lind;
                int nv_lower = 0; // number of vertices on the lower 3d-like base of the
                // space-time rectangle
                for ( int vert = 0; vert < plan4d->GetNVertices() ; ++vert )
                {
                    if (v[vert] / nv_base == tslab)
                        nv_lower++;
                    else if (v[vert] / nv_base != tslab + 1)
                        cout << "Strange edge-type planar: a vertex is neither on the "
                                "top nor on the bottom" << endl << flush;
                }

                if (nv_lower == 2)
                    tslabprism_lind = 0;
                else if (nv_lower == 1)
                    tslabprism_lind = 1;
                else
                    cout << "Strange edge-type planar: nv_lower is not 1 or 2" << endl << flush;

                int pos = shift + findproj_inedge->second[1] * Nsteps * 2
                        + tslab * 2 + tslabprism_lind;

                group_splan_J[group_splan_I[temp - 1] + pos] = cnt_inface + cnt_inedge;

                cnt_inedge++;
            }


            if (findproj_inface != ShfacesBase.end() || findproj_inedge != ShedgesBase.end())
            {
                // Here we swap the planars so that their orientation is consistent across
                // processes. For that we use the order based on the geometric ordering of
                // vertices:
                // Vertex A > Vertex B <-> x(A) > x(B) or ( x(A) = x(B) and y(A) > y(B) or (...))

                for ( int i = 0; i < 3; ++i)
                    vcoords[i] = GetVertex(v[i]);

                // old
                //sortingPermutation(4, vcoords, ordering);

                for (int vert = 0; vert < 3; ++vert)
                    vcoordsNew[vert].assign(vcoords[vert],
                                            vcoords[vert] + Dim);

                sortingPermutationNew(vcoordsNew, ordering);

                /*
                sortingPermutationNew(vcoordsNew, orderingNew);

                cout << " Comparing sorting permutation functions" << endl;
                for ( int i = 0; i < 3; ++i)
                    if (ordering[i] != orderingNew[i])
                        cout << "ERRRRRRRRRRRORRRRRRRRRR";
                */


                if ( permutation_sign(ordering, 3) == -1 ) //possible values are -1 or 1
                    Swap(v);

                shared_planars[cnt_inface + cnt_inedge - 1] = new Triangle(v);

                splan_lplan[cnt_inface + cnt_inedge - 1] = planind4d;

            }



        }

        group_splan.SetIJ(group_splan_I, group_splan_J, group_proc.Size() - 1);

        int cnt_shplanars = cnt_inface + cnt_inedge;

        if (cnt_shplanars != shared_planars.Size())
            cout << "Error: smth wrong with the number of shared planars" << endl;

    } // end of if Dim == 4 case for creating planars


    // 3.3 shared vertices 3d & shared edges 3d -> shared edges 4d
    // (or shared vertices 2d & shared edges 2d -> shared edges 3d)

    // 4d case (3d case):
    // Nsteps + 1 segments which are parallel to a shared edge 3d(2d)
    // Nsteps segments which are diagonals in 2D-in-4d(3d) space-time rectangles
    // with a shared edge 3d(2d) as the base
    // Nsteps segments which are vertical sides in 2D-in-4d space-time rectangles
    // with a shared edge 3d(2d) as the base, which are actually one vertical segment for
    // each shared vertex.

    shared_edges.SetSize( ((Nsteps + 1) + Nsteps)*meshbase_shared_edges_size
                                   + Nsteps*meshbase_svert_lvert_size);
    sedge_ledge.SetSize( shared_edges.Size());

    // alternative way to construct group_sedge - from I and J arrays manually
    int * group_sedge_I, * group_sedge_J;
    group_sedge_I = new int[group_proc.Size() + 1];
    group_sedge_I[0] = 0;
    for ( int row = 0; row < group_proc.Size(); ++row )
    {
        // without this if, valgrind reports "Invalid read"
        // because meshbase.group_svert and meshbase.group_sedg
        // has size 0 (for serial case)
        if (meshbase.group_svert.Size() == 0 || meshbase.group_sedge.Size() == 0)
            group_sedge_I[row + 1] = group_sedge_I[row];
        else
            group_sedge_I[row + 1] = group_sedge_I[row] +
                ((Nsteps + 1) + Nsteps)*meshbase.group_sedge.RowSize(row) +
                Nsteps*meshbase.group_svert.RowSize(row);
    }
    group_sedge_J = new int[group_sedge_I[group_proc.Size() - 1]];

    cnt_inedge = 0; // was already used for planars in 4d case
    int cnt_invert = 0;
    Array<int> verts;
    for ( int edgeind = 0; edgeind < GetNEdges(); ++edgeind)
    {
        GetEdgeVertices(edgeind, verts);

        set<int> edgeproj;
        for ( int vert = 0; vert < verts.Size() ; ++vert )
        {
            //assuming  all time slabs have the same number of nodes and
            // no additional vertices are added to the 4d prisms
            edgeproj.insert( verts[vert] % nv_base);

        }

        // = 0 for edges projected onto the 3d edge and smth for edges
        // projected onto the 3d vertex
        int shift;

        auto findproj_inedge = ShedgesBase.find(edgeproj);
        auto findproj_invert = ShvertsBase.find(edgeproj);

        if (findproj_inedge != ShedgesBase.end() || findproj_invert != ShvertsBase.end())
        {
            shared_edges[cnt_inedge + cnt_invert] = new Segment(verts,1);
            sedge_ledge[cnt_inedge + cnt_invert] = edgeind;
        }

        if ( findproj_inedge != ShedgesBase.end())
        {
            groupind = findproj_inedge->second[0] + 1;
            group.Recreate(group_proc.RowSize(groupind), group_proc.GetRow(groupind));

            int temp = groups.Insert(group);

            // computing the local index of one of the edges which produce
            // the same projection which happens to be a shared edge 3d (2d):
            // To understand, think of all 4d(3d) edges which project to the same edge.
            // They form a long space-time rectangle with Nsteps + 1 plane sections.
            // Within each time slab there is also one diagonal inside the corresponding
            // rectangle. We consider all not-vertical edges there and create a numeration
            // for them.
            // These are (omitting strictly time vertical edges):
            // Nsteps + 1 bases-edges of the rectangle and Nsteps diagonals.
            // 0 for the lowest edge parallel to the base (3d or 2d)
            // 1 for the diagonal in the same time slab
            // 2 ~ 0 but in the next time slab
            // etc...

            shift = 0;

            // time slab which the edge belongs to. (minimal time slab over all vertices)
            // It is initialized with the maximum possible value and then defined as a minimum
            // tslab number over all tetrahedronv vertices
            int tslab = Nsteps; //the uppermost base will formally be in this time slab
            for ( int vert = 0; vert < verts.Size() ; ++vert )
                if (verts[vert] / nv_base  < tslab)
                    tslab = verts[vert]/nv_base;

            // index within a timeslab: 0 or 1 based on the following order:
            // 0 for the 3d-like base-edge
            // 1 for the diagonal within the same timeslab
            int tslab_localind;
            // number of vertices on the lower 3d-like base of the space-time rectangle
            int nv_lower = 0;
            for ( int vert = 0; vert < verts.Size() ; ++vert )
            {
                if (verts[vert] / nv_base == tslab)
                    nv_lower++;
                else if (verts[vert] / nv_base != tslab + 1)
                    cout << "Strange edge-type edge: a vertex is neither on the top"
                            " nor on the bottom" << endl << flush;
            }

            if (nv_lower < 1)
                cout << "Strange edge-type edge: nv_lower is not 1 or 2" << endl << flush;
            else
            {
                tslab_localind = 2 - nv_lower;
            }

            /*
            if (nv_lower == 2)
                tslabprism_lind = 0;
            else if (nv_lower = 1)
                tslabprism_lind = 1;
            else
                cout << "Strange edge-type edge: nv_lower is not 1 or 2" << endl;
            */

            int pos = shift + findproj_inedge->second[1] * ((Nsteps + 1) + Nsteps)
                    + tslab * 2 + tslab_localind;

            group_sedge_J[group_sedge_I[temp - 1] + pos] = cnt_inedge + cnt_invert;

            cnt_inedge++;
        }

        if ( findproj_invert != ShvertsBase.end())
        {
            groupind = findproj_invert->second[0] + 1;
            group.Recreate(group_proc.RowSize(groupind), group_proc.GetRow(groupind));

            int temp = groups.Insert(group);

            // computing the local index of one of the edges which produce
            // the same projection onto 3d which happens to be a shared vertex 3d:
            // Trying to understand, think of all 4d edges which project to the same 3d vertex.
            // They form a long time vertical line with Nsteps + 1 points on it.
            // We consider all vertical edges there and create a numeration for them.
            // There are Nsteps of them, one per time slab:

            // first we need to jump over places reserved for edges which are projected
            // onto the shared edges 3d for a given group of processors
            shift = ((Nsteps + 1) + Nsteps)*meshbase.group_sedge.RowSize(temp - 1);

            // time slab which the edge belongs to. (minimal time slab over all vertices)
            // It is initialized with the maximum possible value and then defined as a minimum
            // tslab number over all tetrahedron vertices
            int tslab = Nsteps; //the uppermost base will formally be in this time slab
            for ( int vert = 0; vert < verts.Size() ; ++vert )
                if (verts[vert] / nv_base  < tslab)
                    tslab = verts[vert]/nv_base;

            int pos = shift + findproj_invert->second[1] * Nsteps + tslab;

            group_sedge_J[group_sedge_I[temp - 1] + pos] = cnt_inedge + cnt_invert;

            cnt_invert++;
        }

    }
    group_sedge.SetIJ(group_sedge_I, group_sedge_J, group_proc.Size() - 1);


    int cnt_shedges = cnt_inedge + cnt_invert;

    if (cnt_shedges != shared_edges.Size())
        cout << "Error: smth wrong with the number of shared faces" << endl << flush;

    /*

    for ( int proc = 0; proc < num_procs; ++proc )
    {
        if ( proc == myid )
        //if ( proc == 0 )
        //if ( proc == 1 && proc == myid )
        {
            cout << "I am " << proc << ", parmesh myrank = " << MyRank << endl;
            cout << "sedge_ledge 4d(3d)" << endl;
            sedge_ledge.Print(cout, 10);
            cout << "group_sedge 4d(3d)" << endl;
            group_sedge.Print(cout, 10);
            for ( int row = 0; row < group_sedge.Size(); ++row)
            {
                int rowsize = group_sedge.RowSize(row);
                int * rowcols = group_sedge.GetRow(row);

                cout << "Row = " << row << endl;
                for ( int col = 0; col < rowsize; ++col)
                {
                    cout << "Edge No." << col << endl;

                    Array<int> v;
                    GetEdgeVertices(sedge_ledge[rowcols[col]], v);

                    for ( int vertno = 0; vertno < 2; ++vertno)
                    {
                        //simple
                        //cout << v[vertno] << " ";
                        // with coords
                        double * vcoords = GetVertex(v[vertno]);
                        cout << vertno << ": (";
                        for ( int coord = 0; coord < Dim; ++coord)
                        {
                            cout << vcoords[coord] << " ";
                        }
                        cout << ")  " << endl;
                    }
                    cout << endl;
                }

            }

            cout << flush;
        }
        MPI_Barrier(comm);
    }

    cout << flush;
    MPI_Barrier(comm);
    */

    // 3.4 shared vertices in th base (3d or 2d) -> shared vertices (4d or 3d)
    // ...
    // Nsteps + 1 time slabs = Nsteps + 1 copies for each shared vertex in the base
    svert_lvert.SetSize( (Nsteps + 1)*meshbase_svert_lvert_size);

    //alternative way to construct group_sedge - from I and J arrays manually
    int * group_svert_I, * group_svert_J;
    group_svert_I = new int[group_proc.Size() + 1];
    group_svert_I[0] = 0;
    for ( int row = 0; row < group_proc.Size(); ++row )
    {
        // without this if, valgrind reports "Invalid read"
        // because meshbase.group_svert has size 0 (for serial case)
        if (meshbase.group_svert.Size() == 0)
            group_svert_I[row + 1] = group_svert_I[row];
        else
            group_svert_I[row + 1] = group_svert_I[row] +
                (Nsteps + 1)*meshbase.group_svert.RowSize(row);
    }
    group_svert_J = new int[group_svert_I[group_proc.Size() - 1]];


    int cnt_shverts = 0;
    for ( int vertind = 0; vertind < GetNV(); ++vertind)
    {
        set<int> vertproj;
        vertproj.insert ( vertind % nv_base );

        auto findproj_inverts = ShvertsBase.find(vertproj);

        if ( findproj_inverts != ShvertsBase.end())
        {
            svert_lvert[cnt_shverts] = vertind;

            groupind = findproj_inverts->second[0] + 1;
            group.Recreate(group_proc.RowSize(groupind), group_proc.GetRow(groupind));

            int temp = groups.Insert(group);

            // computing the local index of one of the vertices which produce
            // the same projection onto 3d which is  a shared vertex 3d:
            // All such vertices are in a time-like line with shared vertex 3d at the bottom
            // There are Nsteps + 1 of them, one per each time moment (including 0):

            // time moment for the vertex
            int timemoment = vertind / nv_base;

            int pos = findproj_inverts->second[1] * (Nsteps + 1) + timemoment;

            group_svert_J[group_svert_I[temp - 1] + pos] = cnt_shverts;

            cnt_shverts++;
        }

    }

    group_svert.SetIJ(group_svert_I, group_svert_J, group_proc.Size() - 1);


    cout << flush;
    MPI_Barrier(comm);

    /*
    for ( int proc = 0; proc < num_procs; ++proc )
    {
        if ( proc == myid )
        {
            cout << "I am " << proc << ", parmesh myrank = " << MyRank << endl;
            cout << "svert_lvert 4d(3d)" << endl;
            svert_lvert.Print(cout, 10);
            cout << "group_svert 4d(3d)" << endl;
            group_svert.Print(cout, 10);
            for ( int row = 0; row < group_svert.Size(); ++row)
            {
                int rowsize = group_svert.RowSize(row);
                int * rowcols = group_svert.GetRow(row);

                cout << "Row = " << row << endl;
                for ( int col = 0; col < rowsize; ++col)
                {
                    cout << "Vert No." << col << endl;

                    double * vcoords = GetVertex(svert_lvert[rowcols[col]]);

                    cout << "(";
                    for ( int coord = 0; coord < Dim; ++coord)
                    {
                        cout << vcoords[coord] << " ";
                    }
                    cout << ")  " << endl;

                }

            }

            cout << flush;
        }
        MPI_Barrier(comm);
    }

    cout << flush;
    MPI_Barrier(comm);
    */


    // ****************************************************************************
    // step 4 of x: creating main communication structure for parmesh 4d = gtopo
    // ****************************************************************************

    gtopo.Create(groups, 822);

    MPI_Barrier(comm);
    return;
}


// Takes the 4d mesh with elements, vertices and boundary already created
// and creates all the internal structure.
// Used inside the Mesh constructor.
// "refine" argument is added for handling 2D case, when refinement marker routines
// should be called before creating structures for shared entities which goes
// before the call to CreateInternal...()
// Probably for parallel mesh generator some tables are generated twice // FIX IT
void ParMeshCyl::CreateInternalMeshStructure (int refine)
{
    int j, curved = 0;
    //int refine = 1;
    bool fix_orientation = true;
    int generate_edges = 1;

    Nodes = NULL;
    own_nodes = 1;
    NURBSext = NULL;
    ncmesh = NULL;
    last_operation = Mesh::NONE;
    sequence = 0;

    // FIXME: DestroyTables() was added here to get rid of memory leaks
    // might not be optimal since some of the tables will be already initialized before the call to
    // CreateInternalMeshStructure, but here just the safest way is chosen
    DestroyTables();
    InitTables();

    // for a 4d mesh sort the element and boundary element indices by the node numbers
    if (spaceDim == 4)
    {
        swappedElements.SetSize(NumOfElements);
        DenseMatrix J(4,4);
        for (j = 0; j < NumOfElements; j++)
        {
            if (elements[j]->GetType() == Element::PENTATOPE)
            {
                int *v = elements[j]->GetVertices();
                Sort5(v[0], v[1], v[2], v[3], v[4]);

                GetElementJacobian(j, J);
                if(J.Det() < 0.0)
                {
                    swappedElements[j] = true;
                    Swap(v);
                }else
                {
                    swappedElements[j] = false;
                }
            }

        }
        for (j = 0; j < NumOfBdrElements; j++)
        {
            if (boundary[j]->GetType() == Element::TETRAHEDRON)
            {
                int *v = boundary[j]->GetVertices();
                Sort4(v[0], v[1], v[2], v[3]);
            }
        }
    }

    // at this point the following should be defined:
    //  1) Dim
    //  2) NumOfElements, elements
    //  3) NumOfBdrElements, boundary
    //  4) NumOfVertices, with allocated space in vertices
    //  5) curved
    //  5a) if curved == 0, vertices must be defined
    //  5b) if curved != 0 and read_gf != 0,
    //         'input' must point to a GridFunction
    //  5c) if curved != 0 and read_gf == 0,
    //         vertices and Nodes must be defined

    if (spaceDim == 0)
    {
       spaceDim = Dim;
    }

    InitBaseGeom();

    // set the mesh type ('meshgen')
    SetMeshGen();


    if (NumOfBdrElements == 0 && Dim > 2)
    {
       // in 3D, generate boundary elements before we 'MarkForRefinement'
       if(Dim==3) GetElementToFaceTable();
       else if(Dim==4)
       {
           GetElementToFaceTable4D();
       }
       GenerateFaces();
       GenerateBoundaryElements();
    }


    if (!curved)
    {
       // check and fix element orientation
       CheckElementOrientation(fix_orientation);

       if (refine)
       {
          MarkForRefinement();
       }
    }

    if (Dim == 1)
    {
       GenerateFaces();
    }

    // generate the faces
    if (Dim > 2)
    {
           if(Dim==3) GetElementToFaceTable();
           else if(Dim==4)
           {
               GetElementToFaceTable4D();
           }

           GenerateFaces();

           if(Dim==4)
           {
              ReplaceBoundaryFromFaces();

              GetElementToPlanarTable();
              GeneratePlanars();

 //			 GetElementToQuadTable4D();
 //			 GenerateQuads4D();
           }

       // check and fix boundary element orientation
       if ( !(curved && (meshgen & 1)) )
       {
          CheckBdrElementOrientation();
       }
    }
    else
    {
       NumOfFaces = 0;
    }

    // generate edges if requested
    if (Dim > 1 && generate_edges == 1)
    {
       // el_to_edge may already be allocated (P2 VTK meshes)
       if (!el_to_edge)
       {
           el_to_edge = new Table;
       }
       NumOfEdges = GetElementToEdgeTable(*el_to_edge, be_to_edge);
       if (Dim == 2)
       {
          GenerateFaces(); // 'Faces' in 2D refers to the edges
          if (NumOfBdrElements == 0)
          {
             GenerateBoundaryElements();
          }
          // check and fix boundary element orientation
          if ( !(curved && (meshgen & 1)) )
          {
             CheckBdrElementOrientation();
          }
       }
    }
    else
    {
       NumOfEdges = 0;
    }

    //// generate the arrays 'attributes' and ' bdr_attributes'
    SetAttributes();

    return;
}

void ParMeshCyl::PrintSlabsStruct()
{
    if (!have_slabs_structure)
        std::cout << "No slabs structure available \n";
    else
    {
        std::cout << "nslabs: " << slabs_struct->nslabs << "\n";
        std::cout << "slabs_offsets: \n";
        slabs_struct->slabs_offsets.Print();
        std::cout << "element markers: \n";
        slabs_struct->el_slabs_markers.Print();
        std::cout << "boundary element markers: \n";
        slabs_struct->bdrel_slabs_markers.Print();
        std::cout << "\n";
    }
}


// from a given base mesh (3d tetrahedrons or 2D triangles) produces a space-time mesh
// for a space-time cylinder with the given base, Nsteps * tau height in time
// enumeration of space-time vertices: time slab after time slab
// boundary attributes: 1 for t=0, 2 for lateral boundaries, 3 for t = tau*Nsteps
void ParMeshCyl::MeshSpaceTimeCylinder_onlyArrays ( double tinit, double tau, int Nsteps,
                                              int bnd_method, int local_method)
{
    int DimBase = meshbase.Dimension(), NumOfBaseElements = meshbase.GetNE(),
            NumOfBaseBdrElements = meshbase.GetNBE(),
            NumOfBaseVertices = meshbase.GetNV();
    int NumOfSTElements, NumOfSTBdrElements, NumOfSTVertices;

    if ( DimBase != 3 && DimBase != 2 )
    {
        cerr << "Wrong dimension in MeshSpaceTimeCylinder(): " << DimBase << endl << flush;
        return;
    }

    if ( DimBase == 2 )
    {
        if ( local_method == 1 )
        {
            cerr << "This local method = " << local_method << " is not supported by case "
                                                     "dim = " << DimBase << endl << flush;
            return;
        }
    }

    int Dim = DimBase + 1;

    // for each base element and each time slab a space-time prism with base mesh element as a base
    // is decomposed into (Dim) simplices (tetrahedrons in 3d and pentatops in 4d);
    NumOfSTElements = NumOfBaseElements * Dim * Nsteps;
    NumOfSTVertices = NumOfBaseVertices * (Nsteps + 1); // no additional vertices inbetween time slabs so far
    // lateral 4d bdr faces (one for each 3d bdr face) + lower + upper bases
    // of the space-time cylinder
    NumOfSTBdrElements = NumOfBaseBdrElements * DimBase * Nsteps + 2 * NumOfBaseElements;

    if (slabs_struct)
    {
        slabs_struct->el_slabs_markers.SetSize(NumOfSTElements);
        slabs_struct->bdrel_slabs_markers.SetSize(NumOfSTBdrElements);
    }

    // assuming that the 3D mesh contains elements of the same type = tetrahedrons
    int vert_per_base = meshbase.GetElement(0)->GetNVertices();
    int vert_per_prism = 2 * vert_per_base;
    int vert_per_latface = DimBase * 2;

    InitMesh(Dim,Dim,NumOfSTVertices,NumOfSTElements,NumOfSTBdrElements);

    Element * el;

    int * simplexes;
    if (local_method == 1 || local_method == 2)
    {
        simplexes = new int[Dim * (Dim + 1)]; // array for storing vertex indices for constructed simplices
    }
    else // local_method = 0 (deprecated)
    {
        // why 5? how many slivers can b created by qhull?
        // maybe 0 if we don't joggle inside qhull but perturb the coordinates before?
        int nsliver = 5;
        // array for storing vertex indices for constructed simplices + probably sliver pentatopes
        simplexes = new int[(Dim + nsliver) * (Dim + 1)];
    }

    // stores indices of space-time element face vertices produced by qhull for all lateral faces
    // Used in local_method = 1 only.
    int * facesimplicesAll;
    if (local_method == 1 )
        facesimplicesAll = new int[DimBase * (DimBase + 1) * Dim ];

    Array<int> elverts_base;
    Array<int> elverts_prism;

    // temporary array for vertex indices of a pentatope face (used in local_method = 0 and 2)
    int * tempface = new int[Dim];
    int * temp = new int[Dim + 1]; //temp array for simplex vertices in local_method = 1;

    // three arrays below are used only in local_method = 1
    Array2D<int> vert_to_vert_prism; // for a 4D prism
    // row ~ lateral face of the 4d prism
    // first 6 columns - indices of vertices belonging to the lateral face,
    // last 2 columns - indices of the rest 2 vertices of the prism
    Array2D<int> latfacets_struct;
    // coordinates of vertices of a lateral face of 4D prism
    double * vert_latface;
    // coordinates of vertices of a 3D base (triangle) of a lateral face of 4D prism
    double * vert_3Dlatface;
    if (local_method == 1)
    {
        vert_latface =  new double[Dim * vert_per_latface];
        vert_3Dlatface = new double[DimBase * vert_per_latface];
        latfacets_struct.SetSize(Dim, vert_per_prism);
        vert_to_vert_prism.SetSize(vert_per_prism, vert_per_prism);
    }

    // coordinates of vertices of the space-time prism
    double * elvert_coordprism = new double[Dim * vert_per_prism];

    char * qhull_flags;
    if (local_method == 0 || local_method == 1)
    {
        qhull_flags = new char[250];
        sprintf(qhull_flags, "qhull d Qbb");
    }

    int simplex_count = 0;
    Element * NewEl;
    Element * NewBdrEl;

    double * tempvert = new double[Dim];

    if (local_method < 0 && local_method > 2)
    {
        cout << "Local method = " << local_method << " is not supported" << endl << flush;
        return;
    }

    if ( bnd_method != 0 && bnd_method != 1)
    {
        cout << "Illegal value of bnd_method = " << bnd_method << " (must be 0 or 1)"
             << endl << flush;
        return;
    }

    Vector vert_coord3d(DimBase * meshbase.GetNV());
    meshbase.GetVertices(vert_coord3d);
    //printDouble2D(vert_coord3d, 10, Dim3D);

    // adding all space-time vertices to the mesh
    for ( int tslab = 0; tslab <= Nsteps; ++tslab)
    {
        // adding the vertices from the slab to the output space-time mesh
        for ( int vert = 0; vert < NumOfBaseVertices; ++vert)
        {
            for ( int j = 0; j < DimBase; ++j)
            {
                tempvert[j] = vert_coord3d[vert + j * NumOfBaseVertices];
                tempvert[Dim-1] = tinit + tau * tslab;
            }
            AddVertex(tempvert);
        }
    }

    delete [] tempvert;

    int * almostjogglers = new int[Dim];
    //int permutation[Dim];
    //vector<double*> lcoords(Dim);
    vector<vector<double> > lcoordsNew(Dim);

    // for each (of Dim) base mesh element faces stores 1 if it is at the boundary and 0 else
    int * facebdrmarker = new int[Dim];
    // std::set of the base mesh boundary elements. Using set allows one to perform a search
    // with O(log N_elem) operations
    std::set< std::vector<int> > BdrTriSet;
    Element * bdrel;

    Array<int> face_bndflags;
    if (bnd_method == 1)
    {
        if (Dim == 4)
            face_bndflags.SetSize(meshbase.GetNFaces());
        if (Dim == 3)
            face_bndflags.SetSize(meshbase.GetNEdges());
    }

    Table * localel_to_face;
    Array<int> localbe_to_face;

    // if = 0, a search algorithm is used for defining whether faces of a given base mesh element
    // are at the boundary.
    // if = 1, instead an array face_bndflags is used, which stores 0 and 1 depending on
    // whether the face is at the boundary, + el_to_face table which is usually already
    // generated for the base mesh
    //int bnd_method = 1;

    if (bnd_method == 0)
    {
        // putting base mesh boundary elements from base mesh structure to the set BdrTriSet
        for ( int boundelem = 0; boundelem < NumOfBaseBdrElements; ++boundelem)
        {
            //cout << "boundelem No. " << boundelem << endl;
            bdrel = meshbase.GetBdrElement(boundelem);
            int * bdrverts = bdrel->GetVertices();

            std::vector<int> buff (bdrverts, bdrverts+DimBase);
            std::sort (buff.begin(), buff.begin()+DimBase);

            BdrTriSet.insert(buff);
        }
        /*
        for (vector<int> temp : BdrTriSet)
        {
            cout << temp[0] << " " <<  temp[1] << " " << temp[2] << endl;
        }
        cout<<endl;
        */
    }
    else // bnd_method = 1
    {
        if (Dim == 4)
        {
            if (meshbase.el_to_face == NULL)
            {
                cout << "Have to built el_to_face" << endl;
                meshbase.GetElementToFaceTable(0);
            }
            localel_to_face = meshbase.el_to_face;
            localbe_to_face.MakeRef(meshbase.be_to_face);
        }
        if (Dim == 3)
        {
            if (meshbase.el_to_edge == NULL)
            {
                cout << "Have to built el_to_edge" << endl;
                meshbase.GetElementToEdgeTable(*(meshbase.el_to_edge), meshbase.be_to_edge);
            }
            localel_to_face = meshbase.el_to_edge;
            localbe_to_face.MakeRef(meshbase.be_to_edge);
        }

        //cout << "Special print" << endl;
        //cout << mesh3d.el_to_face(elind, facelind);
        //cout << "be_to_face" << endl;
        //mesh3d.be_to_face.Print();
        //localbe_to_face.Print();


        //cout << "nfaces = " << meshbase.GetNFaces();
        //cout << "nbe = " << meshbase.GetNBE() << endl;
        //cout << "boundary.size = " << mesh3d.boundary.Size() << endl;

        face_bndflags = -1;
        for ( int i = 0; i < meshbase.GetNBE(); ++i )
            //face_bndflags[meshbase.be_to_face[i]] = 1;
            face_bndflags[localbe_to_face[i]] = 1;

        //cout << "face_bndflags" << endl;
        //face_bndflags.Print();
    }

    int * ordering = new int [vert_per_base];
    //int antireordering[vert_per_base]; // used if bnd_method = 0 and local_method = 2
    Array<int> tempelverts(vert_per_base);

    // main loop creates space-time elements over all time slabs over all base mesh elements
    // loop over base mesh elements
    for ( int elind = 0; elind < NumOfBaseElements; elind++ )
    //for ( int elind = 0; elind < 1; ++elind )
    {
        //cout << "element " << elind << endl;

        el = meshbase.GetElement(elind);

        // 1. getting indices of base mesh element vertices and their coordinates in the prism
        el->GetVertices(elverts_base);

        //for ( int k = 0; k < elverts_base.Size(); ++k )
          //  cout << "elverts[" << k << "] = " << elverts_base[k] << endl;

        // for local_method 2 we need to reorder the local vertices of the prism to preserve
        // the the order in some global sense  = lexicographical order of the vertex coordinates
        if (local_method == 2)
        {
            // using elvert_coordprism as a temporary buffer for changing elverts_base
            for ( int vert = 0; vert < vert_per_base; ++vert)
            {
                for ( int j = 0; j < DimBase; ++j)
                {
                    elvert_coordprism[Dim * vert + j] =
                            vert_coord3d[elverts_base[vert] + j * NumOfBaseVertices];
                }
            }

            /*
             * old one
            for (int vert = 0; vert < Dim; ++vert)
                lcoords[vert] = elvert_coordprism + Dim * vert;

            sortingPermutation(DimBase, lcoords, ordering);

            cout << "ordering 1:" << endl;
            for ( int i = 0; i < vert_per_base; ++i)
                cout << ordering[i] << " ";
            cout << endl;
            */

            for (int vert = 0; vert < Dim; ++vert)
                lcoordsNew[vert].assign(elvert_coordprism + Dim * vert,
                                        elvert_coordprism + Dim * vert + DimBase);

            sortingPermutationNew(lcoordsNew, ordering);

            //cout << "ordering 2:" << endl;
            //for ( int i = 0; i < vert_per_base; ++i)
                //cout << ordering[i] << " ";
            //cout << endl;

            // UGLY: Fix it
            for ( int i = 0; i < vert_per_base; ++i)
                tempelverts[i] = elverts_base[ordering[i]];

            for ( int i = 0; i < vert_per_base; ++i)
                elverts_base[i] = tempelverts[i];
        }

        // 2. understanding which of the base mesh element faces (triangles) are at the boundary
        int local_nbdrfaces = 0;
        set<set<int> > LocalBdrs;
        if (bnd_method == 0) // in this case one looks in the set of base mesh boundary elements
        {
            vector<int> face(DimBase);
            for (int i = 0; i < Dim; ++i )
            {
                // should be consistent with lateral faces ordering in latfacet structure
                // if used with local_method = 1

                for ( int j = 0; j < DimBase; ++j)
                    face[j] = elverts_base[(i+j)%Dim];

                sort(face.begin(), face.begin()+DimBase);
                //cout << face[0] << " " <<  face[1] << " " << face[2] << endl;

                if (BdrTriSet.find(face) != BdrTriSet.end() )
                {
                    local_nbdrfaces++;
                    facebdrmarker[i] = 1;
                    set<int> face_as_set;

                    for ( int j = 0; j < DimBase; ++j)
                        face_as_set.insert((i+j)%Dim);

                    LocalBdrs.insert(face_as_set);
                }
                else
                    facebdrmarker[i] = 0;
            }

        } //end of if bnd_method == 0
        else // in this case one uses el_to_face and face_bndflags to check whether mesh base
             //face is at the boundary
        {
            int * faceinds = localel_to_face->GetRow(elind);
            Array<int> temp(DimBase);
            for ( int facelind = 0; facelind < Dim; ++facelind)
            {
                int faceind = faceinds[facelind];
                if (face_bndflags[faceind] == 1)
                {
                    meshbase.GetFaceVertices(faceind, temp);

                    set<int> face_as_set;
                    for ( int vert = 0; vert < DimBase; ++vert )
                        face_as_set.insert(temp[vert]);

                    LocalBdrs.insert(face_as_set);

                    local_nbdrfaces++;
                }

            } // end of loop over element faces

        }

        //cout << "Welcome the facebdrmarker" << endl;
        //printInt2D(facebdrmarker, 1, Dim);

        /*
        cout << "Welcome the LocalBdrs" << endl;
        for ( set<int> tempset: LocalBdrs )
        {
            cout << "element of LocalBdrs for el = " << elind << endl;
            for (int ind: tempset)
                cout << ind << " ";
            cout << endl;
        }
        */

        // 3. loop over all space-time slabs above a given mesh base element
        int current_timeslab_index = 0;
        for ( int tslab = 0; tslab < Nsteps; ++tslab)
        {
            if (slabs_struct)
            {
                if (tslab == slabs_struct->slabs_offsets[current_timeslab_index + 1])
                    ++current_timeslab_index;
            }
            //cout << "tslab " << tslab << endl;

            //3.1 getting vertex indices for the space-time prism
            elverts_prism.SetSize(vert_per_prism);

            for ( int i = 0; i < vert_per_base; ++i)
            {
                elverts_prism[i] = elverts_base[i] + tslab * NumOfBaseVertices;
                elverts_prism[i + vert_per_base] = elverts_base[i] +
                        (tslab + 1) * NumOfBaseVertices;
            }
            //cout << "New elverts_prism" << endl;
            //elverts_prism.Print(cout, 10);
            //return;


            // 3.2 for the first time slab we add the base mesh elements in the lower base
            // to the space-time bdr elements
            if ( tslab == 0 )
            {
                //cout << "zero slab: adding boundary element:" << endl;
                if (Dim == 3)
                    NewBdrEl = new Triangle(elverts_prism);
                if (Dim == 4)
                    NewBdrEl = new Tetrahedron(elverts_prism);
                NewBdrEl->SetAttribute(1);
                AddBdrElement(NewBdrEl);
                if (slabs_struct)
                    slabs_struct->bdrel_slabs_markers[NumOfBdrElements - 1] = current_timeslab_index;
                bot_to_top_bels[elind].first = NumOfBdrElements - 1;
            }
            // 3.3 for the last time slab we add the base mesh elements in the upper base
            // to the space-time bdr elements
            if ( tslab == Nsteps - 1 )
            {
                //cout << "last slab: adding boundary element:" << endl;
                if (Dim == 3)
                    NewBdrEl = new Triangle(elverts_prism + vert_per_base);
                if (Dim == 4)
                    NewBdrEl = new Tetrahedron(elverts_prism + vert_per_base);
                NewBdrEl->SetAttribute(3);
                AddBdrElement(NewBdrEl);
                if (slabs_struct)
                    slabs_struct->bdrel_slabs_markers[NumOfBdrElements - 1] = current_timeslab_index;
                bot_to_top_bels[elind].second = NumOfBdrElements - 1;
            }

            if (local_method == 0 || local_method == 1)
            {
                // 3.4 setting vertex coordinates for space-time prism, lower base
                for ( int vert = 0; vert < vert_per_base; ++vert)
                {
                    for ( int j = 0; j < DimBase; ++j)
                        elvert_coordprism[Dim * vert + j] =
                                vert_coord3d[elverts_base[vert] + j * NumOfBaseVertices];
                    elvert_coordprism[Dim * vert + Dim-1] = tslab * tau;
                }

                //cout << "Welcome the vertex coordinates for the 4d prism base " << endl;
                //printDouble2D(elvert_coordprism, vert_per_base, Dim);

                /*
                 * old
                for (int vert = 0; vert < Dim; ++vert)
                    lcoords[vert] = elvert_coordprism + Dim * vert;


                //cout << "vector double * lcoords:" << endl;
                //for ( int i = 0; i < Dim; ++i)
                    //cout << "lcoords[" << i << "]: " << lcoords[i][0] << " " << lcoords[i][1] << " " << lcoords[i][2] << endl;

                sortingPermutation(DimBase, lcoords, permutation);
                */

                // here we compute the permutation "ordering" which preserves the geometric order of vertices
                // which is based on their coordinates comparison and compute jogglers for qhull
                // from the "ordering"

                for (int vert = 0; vert < Dim; ++vert)
                    lcoordsNew[vert].assign(elvert_coordprism + Dim * vert,
                                            elvert_coordprism + Dim * vert + DimBase);

                sortingPermutationNew(lcoordsNew, ordering);


                //cout << "Welcome the permutation:" << endl;
                //cout << ordering[0] << " " << ordering[1] << " " <<ordering[2] << " " << ordering[3] << endl;

                int joggle_coeff = 0;
                for ( int i = 0; i < Dim; ++i)
                    almostjogglers[ordering[i]] = joggle_coeff++;


                // 3.5 setting vertex coordinates for space-time prism, upper layer
                // Joggling is required for getting unique Delaunay tesselation and should be
                // the same for vertices shared between different elements or at least produce
                // the same Delaunay triangulation in the shared faces.
                // So here it is not exactly the same, but if joggle(vertex A) > joggle(vertex B)
                // on one element, then the same inequality will hold in another element which also has
                // vertices A and B.
                double joggle;
                for ( int vert = 0; vert < vert_per_base; ++vert)
                {
                    for ( int j = 0; j < DimBase; ++j)
                        elvert_coordprism[Dim * (vert_per_base + vert) + j] =
                                elvert_coordprism[Dim * vert + j];
                    joggle = 1.0e-2 * (almostjogglers[vert]);
                    //joggle = 1.0e-2 * elverts_prism[i + vert_per_base] * 1.0 / NumOf4DVertices;
                    //double joggle = 1.0e-2 * i;
                    elvert_coordprism[Dim * (vert_per_base + vert) + Dim-1] =
                            (tslab + 1) * tau * ( 1.0 + joggle );
                }

                //cout << "Welcome the vertex coordinates for the 4d prism" << endl;
                //printDouble2D(elvert_coordprism, 2 * vert_per_base, Dim);

                // 3.6 - 3.10: constructing new space-time simplices and space-time boundary elements
                if (local_method == 0)
                {
#ifdef WITH_QHULL
                    qhT qh_qh;                /* Qhull's data structure.  First argument of most calls */
                    qhT *qh= &qh_qh;
                    int curlong, totlong;     /* memory remaining after qh_memfreeshort */

                    double volumetol = 1.0e-8;
                    qhull_wrapper(simplexes, qh, elvert_coordprism, Dim, volumetol, qhull_flags);

                    qh_freeqhull(qh, !qh_ALL);
                    qh_memfreeshort(qh, &curlong, &totlong);
                    if (curlong || totlong)  /* could also check previous runs */
                    {
                      fprintf(stderr, "qhull internal warning (user_eg, #3): did not free %d bytes"
                                      " of long memory (%d pieces)\n", totlong, curlong);
                    }
#else
                        cout << "Wrong local method, WITH_QHULL flag was not set" << endl;
#endif
                } // end of if local_method = 0

                if (local_method == 1) // works only in 4D case. Just historically the first implementation
                {
                    setzero(&vert_to_vert_prism);

                    // 3.6 creating vert_to_vert for the prism before Delaunay
                    // (adding 4d prism edges)
                    for ( int i = 0; i < el->GetNEdges(); i++)
                    {
                        const int * edge = el->GetEdgeVertices(i);
                        //cout << "edge: " << edge[0] << " " << edge[1] << std::endl;
                        vert_to_vert_prism(edge[0], edge[1]) = 1;
                        vert_to_vert_prism(edge[1], edge[0]) = 1;
                        vert_to_vert_prism(edge[0] + vert_per_base, edge[1] + vert_per_base) = 1;
                        vert_to_vert_prism(edge[1] + vert_per_base, edge[0] + vert_per_base) = 1;
                    }

                    for ( int i = 0; i < vert_per_base; i++)
                    {
                        vert_to_vert_prism(i, i) = 1;
                        vert_to_vert_prism(i + vert_per_base, i + vert_per_base) = 1;
                        vert_to_vert_prism(i, i + vert_per_base) = 1;
                        vert_to_vert_prism(i + vert_per_base, i) = 1;
                    }

                    //cout << "vert_to_vert before delaunay" << endl;
                    //printArr2DInt (&vert_to_vert_prism);
                    //cout << endl;

                    // 3.7 creating latfacet structure (brute force), for 4D tetrahedron case
                    // indices are local w.r.t to the 4d prism!!!
                    latfacets_struct(0,0) = 0;
                    latfacets_struct(0,1) = 1;
                    latfacets_struct(0,2) = 2;
                    latfacets_struct(0,6) = 3;

                    latfacets_struct(1,0) = 1;
                    latfacets_struct(1,1) = 2;
                    latfacets_struct(1,2) = 3;
                    latfacets_struct(1,6) = 0;

                    latfacets_struct(2,0) = 2;
                    latfacets_struct(2,1) = 3;
                    latfacets_struct(2,2) = 0;
                    latfacets_struct(2,6) = 1;

                    latfacets_struct(3,0) = 3;
                    latfacets_struct(3,1) = 0;
                    latfacets_struct(3,2) = 1;
                    latfacets_struct(3,6) = 2;

                    for ( int i = 0; i < Dim; ++i)
                    {
                        latfacets_struct(i,3) = latfacets_struct(i,0) + vert_per_base;
                        latfacets_struct(i,4) = latfacets_struct(i,1) + vert_per_base;
                        latfacets_struct(i,5) = latfacets_struct(i,2) + vert_per_base;
                        latfacets_struct(i,7) = latfacets_struct(i,6) + vert_per_base;
                    }

                    //cout << "latfacets_struct (vertex indices)" << endl;
                    //printArr2DInt (&latfacets_struct);

                    //(*)const int * base_face = el->GetFaceVertices(i); // not implemented in MFEM for Tetrahedron ?!

                    int * tetrahedrons;
                    int shift = 0;


                    // 3.8 loop over lateral facets, creating Delaunay triangulations
                    for ( int latfacind = 0; latfacind < Dim; ++latfacind)
                    {
                        //cout << "latface = " << latfacind << endl;
                        for ( int vert = 0; vert < vert_per_latface ; ++vert )
                        {
                            //cout << "vert index = " << latfacets_struct(latfacind,vert) << endl;
                            for ( int coord = 0; coord < Dim; ++coord)
                            {
                                vert_latface[vert*Dim + coord] =
                                  elvert_coordprism[latfacets_struct(latfacind,vert) * Dim + coord];
                            }

                        }

                        //cout << "Welcome the vertices of a lateral face" << endl;
                        //printDouble2D(vert_latface, vert_per_latface, Dim);

                        // creating from 3Dprism in 4D a true 3D prism in 3D by change of
                        // coordinates = computing input argument vert_3Dlatface for qhull wrapper
                        // we know that the first three coordinated of a lateral face is actually
                        // a triangle, so we set the first vertex to be the origin,
                        // the first-to-second edge to be one of the axis
                        if ( Dim == 4 )
                        {
                            double x1, x2, x3, y1, y2, y3;
                            double dist12, dist13, dist23;
                            double area, h, p;

                            dist12 = dist(vert_latface, vert_latface+Dim , Dim);
                            dist13 = dist(vert_latface, vert_latface+2*Dim , Dim);
                            dist23 = dist(vert_latface+Dim, vert_latface+2*Dim , Dim);

                            p = 0.5 * (dist12 + dist13 + dist23);
                            area = sqrt (p * (p - dist12) * (p - dist13) * (p - dist23));
                            h = 2.0 * area / dist12;

                            x1 = 0.0;
                            y1 = 0.0;
                            x2 = dist12;
                            y2 = 0.0;
                            if ( dist13 - h < 0.0 )
                                if ( fabs(dist13 - h) > 1.0e-10)
                                {
                                    std::cout << "strange: dist13 = " << dist13 << " h = "
                                              << h << std::endl;
                                    return;
                                }
                                else
                                    x3 = 0.0;
                            else
                                x3 = sqrt(dist13*dist13 - h*h);
                            y3 = h;


                            // the time coordinate remains the same
                            for ( int vert = 0; vert < vert_per_latface ; ++vert )
                                vert_3Dlatface[vert*DimBase + 2] = vert_latface[vert*Dim + 3];

                            // first & fourth vertex
                            vert_3Dlatface[0*DimBase + 0] = x1;
                            vert_3Dlatface[0*DimBase + 1] = y1;
                            vert_3Dlatface[3*DimBase + 0] = x1;
                            vert_3Dlatface[3*DimBase + 1] = y1;

                            // second & fifth vertex
                            vert_3Dlatface[1*DimBase + 0] = x2;
                            vert_3Dlatface[1*DimBase + 1] = y2;
                            vert_3Dlatface[4*DimBase + 0] = x2;
                            vert_3Dlatface[4*DimBase + 1] = y2;

                            // third & sixth vertex
                            vert_3Dlatface[2*DimBase + 0] = x3;
                            vert_3Dlatface[2*DimBase + 1] = y3;
                            vert_3Dlatface[5*DimBase + 0] = x3;
                            vert_3Dlatface[5*DimBase + 1] = y3;
                        } //end of creating a true 3d prism

                        //cout << "Welcome the vertices of a lateral face in 3D" << endl;
                        //printDouble2D(vert_3Dlatface, vert_per_latface, Dim3D);

                        tetrahedrons = facesimplicesAll + shift;

#ifdef WITH_QHULL
                        qhT qh_qh;                /* Qhull's data structure.  First argument of most calls */
                        qhT *qh= &qh_qh;
                        int curlong, totlong;     /* memory remaining after qh_memfreeshort */

                        double volumetol = MYZEROTOL;
                        qhull_wrapper(tetrahedrons, qh, vert_3Dlatface, DimBase, volumetol, qhull_flags);

                        qh_freeqhull(qh, !qh_ALL);
                        qh_memfreeshort(qh, &curlong, &totlong);
                        if (curlong || totlong)  /* could also check previous runs */
                          cerr<< "qhull internal warning (user_eg, #3): did not free " << totlong
                          << "bytes of long memory (" << curlong << " pieces)" << endl;
#else
                        cout << "Wrong local method, WITH_QHULL flag was not set" << endl;
#endif
                        // convert local 3D prism (lateral face) vertex indices back to the
                        // 4D prism indices and adding boundary elements from tetrahedrins
                        // for lateral faces of the 4d prism ...
                        for ( int tetraind = 0; tetraind < DimBase; ++tetraind)
                        {
                            //cout << "tetraind = " << tetraind << endl;

                            for ( int vert = 0; vert < Dim; ++vert)
                            {
                                int temp = tetrahedrons[tetraind*Dim + vert];
                                tetrahedrons[tetraind*Dim + vert] = latfacets_struct(latfacind, temp);
                            }

                            if ( bnd_method == 0 )
                            {
                                if ( facebdrmarker[latfacind] == 1 )
                                {
                                    //cout << "lateral facet " << latfacind << " is at the boundary: adding bnd element" << endl;

                                    tempface[0] = elverts_prism[tetrahedrons[tetraind*Dim + 0]];
                                    tempface[1] = elverts_prism[tetrahedrons[tetraind*Dim + 1]];
                                    tempface[2] = elverts_prism[tetrahedrons[tetraind*Dim + 2]];
                                    tempface[3] = elverts_prism[tetrahedrons[tetraind*Dim + 3]];

                                    // wrong because indices in tetrahedrons are local to 4d prism
                                    //NewBdrTri = new Tetrahedron(tetrahedrons + tetraind*Dim);

                                    NewBdrEl = new Tetrahedron(tempface);
                                    NewBdrEl->SetAttribute(2);
                                    AddBdrElement(NewBdrEl);

                                }
                            }
                            else // bnd_method = 1
                            {
                                set<int> latface3d_set;
                                for ( int i = 0; i < DimBase; ++i)
                                    latface3d_set.insert(elverts_prism[latfacets_struct(latfacind,i)] % NumOfBaseVertices);

                                // checking whether a face is at the boundary of 3d mesh
                                if ( LocalBdrs.find(latface3d_set) != LocalBdrs.end())
                                {
                                    // converting local indices to global indices and
                                    // adding the new boundary element
                                    tempface[0] = elverts_prism[tetrahedrons[tetraind*Dim + 0]];
                                    tempface[1] = elverts_prism[tetrahedrons[tetraind*Dim + 1]];
                                    tempface[2] = elverts_prism[tetrahedrons[tetraind*Dim + 2]];
                                    tempface[3] = elverts_prism[tetrahedrons[tetraind*Dim + 3]];

                                    NewBdrEl = new Tetrahedron(tempface);
                                    NewBdrEl->SetAttribute(2);
                                    AddBdrElement(NewBdrEl);
                                }
                            }



                         } //end of loop over tetrahedrons for a given lateral face

                        shift += DimBase * (DimBase + 1);

                        //return;
                    } // end of loop over lateral faces

                    // 3.9 adding the new edges from created tetrahedrons into the vert_to_vert
                    for ( int k = 0; k < Dim; ++k )
                        for (int i = 0; i < DimBase; ++i )
                        {
                            int vert0 = facesimplicesAll[k*DimBase*(DimBase+1) +
                                    i*(DimBase + 1) + 0];
                            int vert1 = facesimplicesAll[k*DimBase*(DimBase+1) +
                                    i*(DimBase + 1) + 1];
                            int vert2 = facesimplicesAll[k*DimBase*(DimBase+1) +
                                    i*(DimBase + 1) + 2];
                            int vert3 = facesimplicesAll[k*DimBase*(DimBase+1) +
                                    i*(DimBase + 1) + 3];

                            vert_to_vert_prism(vert0, vert1) = 1;
                            vert_to_vert_prism(vert1, vert0) = 1;

                            vert_to_vert_prism(vert0, vert2) = 1;
                            vert_to_vert_prism(vert2, vert0) = 1;

                            vert_to_vert_prism(vert0, vert3) = 1;
                            vert_to_vert_prism(vert3, vert0) = 1;

                            vert_to_vert_prism(vert1, vert2) = 1;
                            vert_to_vert_prism(vert2, vert1) = 1;

                            vert_to_vert_prism(vert1, vert3) = 1;
                            vert_to_vert_prism(vert3, vert1) = 1;

                            vert_to_vert_prism(vert2, vert3) = 1;
                            vert_to_vert_prism(vert3, vert2) = 1;
                        }

                    //cout << "vert_to_vert after delaunay" << endl;
                    //printArr2DInt (&vert_to_vert_prism);

                    int count_penta = 0;

                    // 3.10 creating finally 4d pentatopes:
                    // take a tetrahedron related to a lateral face, find out which of the rest
                    // 2 vertices of the 4d prism (one is not) is connected to all vertices of
                    // tetrahedron, and get a pentatope from tetrahedron + this vertex
                    // If pentatope is new, add it to the final structure
                    // To make checking for new pentatopes easy, reoder the pentatope indices
                    // in the default std order

                    for ( int tetraind = 0; tetraind < DimBase * Dim; ++tetraind)
                    {
                        // creating a pentatop temp
                        int latface_ind = tetraind / DimBase;
                        for ( int vert = 0; vert < Dim; vert++ )
                            temp[vert] = facesimplicesAll[tetraind * Dim + vert];

                        //cout << "tetrahedron" << endl;
                        //printInt2D(temp,1,4); // tetrahedron

                        bool isconnected = true;
                        for ( int vert = 0; vert < 4; ++vert)
                            if (vert_to_vert_prism(temp[vert],
                                                   latfacets_struct(latface_ind,6)) == 0)
                                isconnected = false;

                        if ( isconnected == true)
                            temp[4] = latfacets_struct(latface_ind,6);
                        else
                        {
                            bool isconnectedCheck = true;
                            for ( int vert = 0; vert < 4; ++vert)
                                if (vert_to_vert_prism(temp[vert],
                                                       latfacets_struct(latface_ind,7)) == 0)
                                    isconnectedCheck = false;
                            if (isconnectedCheck == 0)
                            {
                                cout << "Error: Both vertices are disconnected" << endl;
                                cout << "tetraind = " << tetraind << ", checking for " <<
                                             latfacets_struct(latface_ind,6) << " and " <<
                                             latfacets_struct(latface_ind,7) << endl;
                                return;
                            }
                            else
                                temp[4] = latfacets_struct(latface_ind,7);
                        }

                        //printInt2D(temp,1,5);

                        // replacing local vertex indices w.r.t to 4d prism to global!
                        temp[0] = elverts_prism[temp[0]];
                        temp[1] = elverts_prism[temp[1]];
                        temp[2] = elverts_prism[temp[2]];
                        temp[3] = elverts_prism[temp[3]];
                        temp[4] = elverts_prism[temp[4]];

                        // sorting the vertex indices
                        std::vector<int> buff (temp, temp+5);
                        std::sort (buff.begin(), buff.begin()+5);

                        // looking whether the current pentatop is new
                        bool isnew = true;
                        for ( int i = 0; i < count_penta; ++i )
                        {
                            std::vector<int> pentatop (simplexes+i*(Dim+1), simplexes+(i+1)*(Dim+1));

                            if ( pentatop == buff )
                                isnew = false;
                        }

                        if ( isnew == true )
                        {
                            for ( int i = 0; i < Dim + 1; ++i )
                                simplexes[count_penta*(Dim+1) + i] = buff[i];
                            //cout << "found a new pentatop from tetraind = " << tetraind << endl;
                            //cout << "now we have " << count_penta << " pentatops" << endl;
                            //printInt2D(pentatops + count_penta*(Dim+1), 1, Dim + 1);

                            ++count_penta;
                        }
                        //cout << "element " << elind << endl;
                        //printInt2D(pentatops, count_penta, Dim + 1);
                    }

                    //cout<< count_penta << " pentatops created" << endl;
                    if ( count_penta != Dim )
                        cout << "Error: Wrong number of simplexes constructed: got " <<
                                count_penta << ", needed " << Dim << endl << flush;
                    //printInt2D(pentatops, count_penta, Dim + 1);

                }

            } //end of if local_method = 0 or 1
            else // local_method == 2
            {
                // The simplest way to generate space-time simplices.
                // But requires to reorder the vertices at first, as done before.
                for ( int count_simplices = 0; count_simplices < Dim; ++count_simplices)
                {
                    for ( int i = 0; i < Dim + 1; ++i )
                    {
                        simplexes[count_simplices*(Dim+1) + i] = count_simplices + i;
                    }

                }
                //cout << "Welcome created pentatops" << endl;
                //printInt2D(pentatops, Dim, Dim + 1);
            }


            // adding boundary elements in local method =  0 or 2
            if (local_method == 0 || local_method == 2)
            {
                //if (local_method == 2)
                    //for ( int i = 0; i < vert_per_base; ++i)
                        //antireordering[ordering[i]] = i;

                if (local_nbdrfaces > 0) //if there is at least one base mesh element face at
                                         // the boundary for a given base element
                {
                    for ( int simplexind = 0; simplexind < Dim; ++simplexind)
                    {
                        //cout << "simplexind = " << simplexind << endl;
                        //printInt2D(pentatops + pentaind*(Dim+1), 1, 5);

                        for ( int faceind = 0; faceind < Dim + 1; ++faceind)
                        {
                            //cout << "faceind = " << faceind << endl;
                            set<int> faceproj;

                            // creating local vertex indices for a simplex face
                            // and projecting the face onto the 3d base
                            if (bnd_method == 0)
                            {
                                int cnt = 0;
                                for ( int j = 0; j < Dim + 1; ++j)
                                {
                                    if ( j != faceind )
                                    {
                                        tempface[cnt] = simplexes[simplexind*(Dim + 1) + j];
                                        if (tempface[cnt] > vert_per_base - 1)
                                            faceproj.insert(tempface[cnt] - vert_per_base);
                                        else
                                            faceproj.insert(tempface[cnt]);
                                        cnt++;
                                    }
                                }

                                //cout << "tempface in local indices" << endl;
                                //printInt2D(tempface,1,4);
                            }
                            else // for bnd_method = 1 we create tempface and projection
                                 // in global indices
                            {
                                int cnt = 0;
                                for ( int j = 0; j < Dim + 1; ++j)
                                {
                                    if ( j != faceind )
                                    {
                                        tempface[cnt] =
                                                elverts_prism[simplexes[simplexind*(Dim + 1) + j]];
                                        faceproj.insert(tempface[cnt] % NumOfBaseVertices );
                                        cnt++;
                                    }
                                }

                                //cout << "tempface in global indices" << endl;
                                //printInt2D(tempface,1,4);
                            }

                            /*
                            cout << "faceproj:" << endl;
                            for ( int temp : faceproj)
                                cout << temp << " ";
                            cout << endl;
                            */

                            // checking whether the projection is at the boundary of base mesh
                            // using the local-to-element LocalBdrs set which has at most Dim elements
                            if ( LocalBdrs.find(faceproj) != LocalBdrs.end())
                            {
                                //cout << "Found a new boundary element" << endl;
                                //cout << "With local indices: " << endl;
                                //printInt2D(tempface, 1, Dim);

                                // converting local indices to global indices and
                                // adding the new boundary element
                                if (bnd_method == 0)
                                {
                                    for ( int facevert = 0; facevert < Dim; ++facevert )
                                        tempface[facevert] = elverts_prism[tempface[facevert]];
                                }

                                //cout << "With global indices: " << endl;
                                //printInt2D(tempface, 1, Dim);

                                if (Dim == 3)
                                    NewBdrEl = new Triangle(tempface);
                                if (Dim == 4)
                                    NewBdrEl = new Tetrahedron(tempface);
                                NewBdrEl->SetAttribute(2);
                                AddBdrElement(NewBdrEl);
                                if (slabs_struct)
                                    slabs_struct->bdrel_slabs_markers[NumOfBdrElements - 1] = current_timeslab_index;
                            }


                        } // end of loop over space-time simplex faces
                    } // end of loop over space-time simplices
                } // end of if local_nbdrfaces > 0

                // By this point, for the given base mesh element:
                // space-time elements are constructed, but stored in local array
                // boundary elements are constructed which correspond to the elements in the space-time prism
                // converting local-to-prism indices in simplices to the global indices
                for ( int simplexind = 0; simplexind < Dim; ++simplexind)
                {
                    for ( int j = 0; j < Dim + 1; j++)
                    {
                        simplexes[simplexind*(Dim + 1) + j] =
                                elverts_prism[simplexes[simplexind*(Dim + 1) + j]];
                    }
                }

            } //end of if local_method = 0 or 2

            // printInt2D(pentatops, Dim, Dim + 1);


            // 3.11 adding the constructed space-time simplices to the output mesh
            for ( int simplex_ind = 0; simplex_ind < Dim; ++simplex_ind)
            {
                //if (Dim == 3)
                    //NewEl = new Tetrahedron(simplexes + simplex_ind*(Dim+1));
                if (Dim == 4)
                {
                    NewEl = new Pentatope(simplexes + simplex_ind*(Dim+1));
                    NewEl->SetAttribute(1);
                }
                if (Dim == 3)
                    AddTet(simplexes + simplex_ind*(Dim+1), 1);
                if (Dim == 4)
                    AddElement(NewEl);

                if (slabs_struct)
                {
                    slabs_struct->el_slabs_markers[NumOfElements - 1] = current_timeslab_index;
                }

                /*
                 * unneeded, because CheckElementOrientation is still called afterwards
                 * in the ParMeshCyl constructor
                // TODO: Probably there is a pattern of which space-time elements got the wrong orientation
                // fixing element orientation
                int j, k, *vi = 0;
                double *v[Dim + 1];

                vi = elements[NumOfElements - 1]->GetVertices();
                DenseMatrix J(Dim, Dim);
                for (j = 0; j < Dim + 1; j++)
                {
                   v[j] = vertices[vi[j]]();
                }
                for (j = 0; j < Dim; j++)
                   for (k = 0; k < Dim; k++)
                   {
                      J(j, k) = v[j+1][k] - v[0][k];
                   }
                if (J.Det() < 0.0)
                {
                    std::cout << "elind = " << elind << ", tslab = " << tslab << ", simplex_ind = " << simplex_ind << "\n";
                    std::cout << "element with bad orientation: ";
                    mfem::Swap(vi[0], vi[1]);
                    std::cout << "was fixed! \n";
                }
                */

                ++simplex_count;
            }

            //printArr2DInt (&vert_to_vert_prism);

        } // end of loop over time slabs
    } // end of loop over base elements

    // checking the correspondence bot_to_top_bels created in the loop above
    /*
    MPI_Comm_size(meshbase.GetComm(), &num_procs);
    MPI_Comm_rank(meshbase.GetComm(), &myid);

    for (int i = 0; i < num_procs; ++i)
    {
        if (myid == i)
        {
            std::cout << "I am " << myid << "\n";
            PrintBotToTopBels();

            // checking bot_to_top_bels
            for (int i = 0; i < meshbase.GetNBE(); ++i)
            {
                 std::cout << "pair " << i << ": \n";
                 int belind_first = bot_to_top_bels[i].first;
                 Element * bel_first = GetBdrElement(belind_first);
                 Array<int> verts_first;
                 bel_first->GetVertices(verts_first);

                 std::cout << "first boundary element: \n";
                 for (int vno = 0; vno < verts_first.Size(); ++vno)
                 {
                     double * vcoos = GetVertex(verts_first[vno]);
                     std::cout << "(";
                     for (int coind = 0; coind < Dimension(); ++coind)
                     {
                         if (coind < Dimension() - 1)
                             std::cout << vcoos[coind] << ", ";
                         else
                             std::cout << vcoos[coind] << ")";
                     }
                 }
                 std::cout << "\n";

                 int belind_second = bot_to_top_bels[i].second;
                 Element * bel_second = GetBdrElement(belind_second);
                 Array<int> verts_second;
                 bel_second->GetVertices(verts_second);

                 std::cout << "second boundary element: \n";
                 for (int vno = 0; vno < verts_second.Size(); ++vno)
                 {
                     double * vcoos = GetVertex(verts_second[vno]);
                     std::cout << "(";
                     for (int coind = 0; coind < Dimension(); ++coind)
                     {
                         if (coind < Dimension() - 1)
                             std::cout << vcoos[coind] << ", ";
                         else
                             std::cout << vcoos[coind] << ")";
                     }
                 }
                 std::cout << "\n";
                 std::cout << "\n";
            }
            std::cout << std::flush;
        }
        MPI_Barrier(meshbase.GetComm());
    }
    */


    if ( NumOfSTElements != GetNE() )
        std::cout << "Error: Wrong number of elements generated: " << GetNE() << " instead of " <<
                        NumOfSTElements << std::endl;
    if ( NumOfSTVertices != GetNV() )
        std::cout << "Error: Wrong number of vertices generated: " << GetNV() << " instead of " <<
                        NumOfSTVertices << std::endl;
    if ( NumOfSTBdrElements!= GetNBE() )
        std::cout << "Error: Wrong number of bdr elements generated: " << GetNBE() << " instead of " <<
                        NumOfSTBdrElements << std::endl;

    delete [] facebdrmarker;
    delete [] ordering;
    delete [] almostjogglers;
    delete [] temp;
    delete [] tempface;
    delete [] simplexes;
    delete [] elvert_coordprism;

    if (local_method == 1)
    {
        delete [] vert_latface;
        delete [] vert_3Dlatface;
        delete [] facesimplicesAll;
    }
    if (local_method == 0 || local_method == 1)
        delete [] qhull_flags;

    return;
}

/*
// FIXME: probably redundant
ParMesh * ParMeshCyl::ExtractTimeSlab(int slab_index)
{
    if (!have_slabs_structure)
    {
        MFEM_ABORT("For current implementation of ExtractTimeSlab, ParMeshCyl must have a slab structure \n");
    }

    MFEM_ASSERT( slab_index >= 0 && slab_index < slabs_struct->nslabs, "Invalid slab_index.");

    return new ParMeshCyl(*this, slab_index);
}

// Extraction constructor
ParMeshCyl::ParMeshCyl(ParMeshCyl& gmesh, int slab_index)
{
    if (gmesh.have_slabs_structure == false)
    {
        MFEM_ABORT("For extraction constructor, input ParMeshCyl must have a slab structure \n");
    }

    comm = gmesh.GetComm();

    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    int dim = gmesh.Dimension() + 1;

    have_slabs_structure = false;

    // ****************************************************************************
    // step 1 of 4: creating local space-time part of the mesh from the global mesh
    // ****************************************************************************

    // creating local parts of space-time mesh
    ExtractTimeSlab_onlyArrays(slab_index);

    MPI_Barrier(comm);
}
*/

void ParMeshCyl::TimeShift(double shift)
{
    int nv = vertices.Size();
    for (int i = 0; i < nv; i++)
        vertices[i](spaceDim - 1) += shift;
}


void ParMeshCyl::PrintBotToTopBels() const
{
    for (unsigned int i = 0; i < bot_to_top_bels.size(); ++i)
    {
        std::cout << "i = " << i  << ": (" << bot_to_top_bels[i].first << ", " << bot_to_top_bels[i].second << ") \n";
    }
}

void ParMeshCyl::Find_be_ordering(SparseMatrix& BE_AE_be, int BE_index, std::vector<int> *be_indices,
                                  std::vector<int> *ordering, bool verbose)
{
    // for the bottom BE and its be's
    std::vector<std::vector<double> > all_verts_coordinates;
    std::set<int> all_vert_indices;
    std::vector<int> verts_pushed;
    std::vector<std::pair<int,int> > verts_newverts_link;
    int * cols = BE_AE_be.GetRowColumns(BE_index);
    double * entries = BE_AE_be.GetRowEntries(BE_index);

    // to each be in BE we assign a sorted array of vertex indices
    // where vertices are ordered by a geometrical coordinate ordering

    // first we collect the vertex coordinates for all be vertices
    int be_count = 0;
    for (int j = 0; j < BE_AE_be.RowSize(BE_index); ++j)
    {
        if (fabs(entries[j]) > 1.0e-10) // in general should be 0 or 1
        {
            int be = cols[j];
            int nverts = GetBdrElement(be)->GetNVertices();
            int * be_verts = GetBdrElement(be)->GetVertices();

            be_indices->push_back(be);

            be_count++;

            for ( int i = 0; i < nverts; ++i)
            {
                if (all_vert_indices.find(be_verts[i]) == all_vert_indices.end())
                {
                    all_vert_indices.insert(be_verts[i]);
                    double * vcoords = GetVertex(be_verts[i]);
                    all_verts_coordinates.push_back(std::vector<double>(vcoords, vcoords + Dimension() - 1));
                    verts_pushed.push_back(be_verts[i]);
                }
            }
        }
    }

    if (verbose)
    {
        std::cout << "all_verts_coordinates: \n";
        for (unsigned int i = 0; i < all_verts_coordinates.size(); ++i)
        {
            std::cout << "vertex: " << i << "\n";
            for (int unsigned j = 0; j < all_verts_coordinates[i].size(); ++j)
                std::cout << all_verts_coordinates[i][j] << " ";
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    if (verbose)
    {
        std::cout << "verts_pushed: \n";
        for (unsigned int i = 0; i < verts_pushed.size(); ++i)
            std::cout << verts_pushed[i] << " ";
        std::cout << "\n";
    }

    // now permutation defines the geometrical order of vertices
    int * permutation = new int[all_verts_coordinates.size()];
    sortingPermutationNew(all_verts_coordinates, permutation);

    if (verbose)
    {
        std::cout << "permutation: \n";
        for (unsigned int i = 0; i < verts_pushed.size(); ++i)
            std::cout << permutation[i] << " ";
        std::cout << "\n";
    }

    int * inv_permutation = new int[all_verts_coordinates.size()];
    invert_permutation(permutation, all_verts_coordinates.size(), inv_permutation);

    if (verbose)
    {
        std::cout << "inverse permutation: \n";
        for (unsigned int i = 0; i < verts_pushed.size(); ++i)
            std::cout << inv_permutation[i] << " ";
        std::cout << "\n";
    }

    for (unsigned int i = 0; i < all_verts_coordinates.size(); ++i)
    {
        verts_newverts_link.push_back(std::pair<int,int>(verts_pushed[i],inv_permutation[i]));
    }

    delete [] permutation;
    delete [] inv_permutation;

    if (verbose)
    {
        std::cout << "verts_newverts_link: \n";
        for (unsigned int i = 0; i < verts_newverts_link.size(); ++i)
            std::cout << "<" << verts_newverts_link[i].first << "," << verts_newverts_link[i].second << "> ";
        std::cout << "\n";
    }

    // to each be we assign now a vector of the new vertex indices
    // corresponding to vertices geometrical ordering
    std::vector<std::set<int> > bels_newverts(be_count);
    int count = 0;
    for (int j = 0; j < BE_AE_be.RowSize(BE_index); ++j)
    {
        if (fabs(entries[j]) > 1.0e-10) // in general should be 0 or 1
        {
            int be = cols[j];
            int nverts = GetBdrElement(be)->GetNVertices();
            int * be_verts = GetBdrElement(be)->GetVertices();

            for ( int i = 0; i < nverts; ++i)
            {
                int rel_vert_index = -1;
                for (unsigned int k = 0; k < verts_pushed.size(); ++k)
                    if (verts_pushed[k] == be_verts[i])
                        rel_vert_index = k;

                MFEM_ASSERT(rel_vert_index != -1, "Error: be vertex was not found among "
                                                  "verts_pushed entries \n");
                bels_newverts[count].insert(verts_newverts_link[rel_vert_index].second);
            }

            count++;
        }
    }

    if (verbose)
    {
        count = 0;
        for (int j = 0; j < BE_AE_be.RowSize(BE_index); ++j)
        {
            if (fabs(entries[j]) > 1.0e-10) // in general should be 0 or 1
            {
                int be = cols[j];
                int nverts = GetBdrElement(be)->GetNVertices();

                int * be_verts = GetBdrElement(be)->GetVertices();

                std::cout << "be: " << be << " has following vertices \n";
                for ( int i = 0; i < nverts; ++i)
                {
                    double * vcoords = GetVertex(be_verts[i]);
                    for (int k = 0; k < Dimension(); ++k)
                        std::cout << vcoords[k] << " ";
                    std::cout << "\n";
                }


                std::cout << "be: " << be << " has new relative verts indices \n";
                std::set<int>::iterator it;
                for (it = bels_newverts[count].begin(); it != bels_newverts[count].end(); ++it)
                    std::cout << *it << " ";
                std::cout << "\n";

                ++count;
            }
        }
    }

    // then we sort the arrays for be's
    // the sorting gives an ordering for be's which will
    // be the same for the top boundary

    std::vector<int> index(bels_newverts.size(), 0);
    for (unsigned int i = 0 ; i != index.size() ; i++)
        index[i] = i;

    std::sort(index.begin(), index.end(),
        [&](const int& a, const int& b) {
            return (bels_newverts[a] < bels_newverts[b]);
        }
    );

    if (verbose)
    {
        std::cout << "ordering of the bels: \n";
        for (unsigned int i = 0 ; i != index.size() ; i++) {
            std::cout << index[i] << endl;
        }
    }

    for (unsigned int i = 0 ; i != index.size() ; i++)
        ordering->push_back(index[i]);
}

void ParMeshCyl::UpdateBotToTopLink(SparseMatrix& BE_AE_be, bool verbose)
{
    //MFEM_ABORT("UpdateBotToTopLink() was not implemented \n");
    //std::vector<std::pair<int,int> > new_bot_to_top_link;
    int old_size = bot_to_top_bels.size();
    bot_to_top_bels = std::vector<std::pair<int,int> >();
    //new_bot_to_top_link.reserve(ipow(2, Dimension() - 1) * old_size);
    bot_to_top_bels.reserve(ipow(2, Dimension() - 1) * old_size);
    for (int BE_bot = 0; BE_bot < BE_AE_be.Height() / 2; ++BE_bot)
    {
        if (verbose)
            std::cout << "be at the bottom No. " << BE_bot << "\n";
        int BE_top = BE_bot + BE_AE_be.Height() / 2;

        std::vector<int> be_indices_bot;
        std::vector<int> ordering_bot;
        Find_be_ordering(BE_AE_be, BE_bot, &be_indices_bot, &ordering_bot, verbose);
        /*
        std::cout << "ordering_bot \n";
        for (unsigned int k = 0; k < ordering_bot.size(); ++k)
            std::cout << ordering_bot[k] << " ";
        std::cout << "\n";
        std::cout << "be indices bot \n";
        for (unsigned int k = 0; k < be_indices_bot.size(); ++k)
            std::cout << be_indices_bot[k] << " ";
        std::cout << "\n";
        */

        // do the same at the top boundary
        std::vector<int> be_indices_top;
        std::vector<int> ordering_top;
        Find_be_ordering(BE_AE_be, BE_top, &be_indices_top, &ordering_top, verbose);
        /*
        std::cout << "ordering_top \n";
        for (unsigned int k = 0; k < ordering_top.size(); ++k)
            std::cout << ordering_top[k] << " ";
        std::cout << "\n";
        std::cout << "be indices top \n";
        for (unsigned int k = 0; k < be_indices_top.size(); ++k)
            std::cout << be_indices_top[k] << " ";
        std::cout << "\n";
        */

        // now by matching the ordered be indices we can create the desired new pairs
        for (unsigned int i = 0; i < be_indices_bot.size(); ++i)
        {
            int be1 = be_indices_bot[ordering_bot[i]];
            int be2 = be_indices_top[ordering_top[i]];
            //std::cout << "<" << be1 << "," << be2 << "> \n";
            //new_bot_to_top_link.push_back(std::pair<int,int>(be1, be2));
            bot_to_top_bels.push_back(std::pair<int,int>(be1, be2));

            // checking the coordinates of the matched elements
            if (verbose)
            {
                int nverts = GetBdrElement(be1)->GetNVertices();
                int * be_verts1 = GetBdrElement(be1)->GetVertices();
                std::cout << "be1: " << be1 << " has following vertices \n";
                for ( int i = 0; i < nverts; ++i)
                {
                    double * vcoords = GetVertex(be_verts1[i]);
                    for (int k = 0; k < Dimension(); ++k)
                        std::cout << vcoords[k] << " ";
                    std::cout << "\n";
                }
                int * be_verts2 = GetBdrElement(be2)->GetVertices();
                std::cout << "be2: " << be2 << " has following vertices \n";
                for ( int i = 0; i < nverts; ++i)
                {
                    double * vcoords = GetVertex(be_verts2[i]);
                    for (int k = 0; k < Dimension(); ++k)
                        std::cout << vcoords[k] << " ";
                    std::cout << "\n";
                }
            }
        } // end of creating a matching between bels inside given BE

    } // end of loop over bottom BEs
}

// Creates be_to_e relation between marked(!) boundary elements
// (those which are used in bot_to_top relation) and elements
SparseMatrix * ParMeshCyl::Create_be_to_e( const char * full_or_marked)
{
    if (strcmp(full_or_marked,"marked") != 0 && strcmp(full_or_marked, "full") != 0)
    {
        MFEM_ABORT("Input argument in Create_be_to_e must be 'marked' or 'full' \n");
    }

    int m;
    int npairs;
    if (strcmp(full_or_marked,"marked") == 0)
    {
        npairs = bot_to_top_bels.size();
        m = 2 * npairs;
    }
    else // "full" case
        m = GetNBE();
    int n = GetNE();

    // each boundary element belongs to one and only one element
    int * ia = new int[m + 1];
    ia[0] = 0;
    for (int i = 0; i < m; ++i)
        ia[i + 1] = ia[i] + 1;

    int * ja = new int [ia[m]];
    double * data = new double [ia[m]];

    int count = 0;
    int f, o, el1, el2;

    // going over marked or all boundary elements
    for (int i = 0; i < m; ++i)
    {
        int bdrel;
        if (strcmp(full_or_marked,"marked") == 0)
        {
            if (i < npairs) // first going over marked bdr elements at the bottom
                bdrel = bot_to_top_bels[i].first;
            else // then at the top
                bdrel = bot_to_top_bels[i - npairs].second;
        }
        else // "full" case
            bdrel = i;

        GetBdrElementFace(bdrel, &f, &o); // f is the bdrel index as a face

        GetFaceElements(f, &el1, &el2);

        //std::cout << "el1 = " << el1 << ", el2 = " << el2 << "\n";
        MFEM_ASSERT(el2 == -1, "Boundary element should have el2 = -1 "
                               "(and belong to the element indexed by el1 \n");

        ja[count] = el1;
        data[count] = 1.0;

        ++count;
    }

    return new SparseMatrix(ia, ja, data, m, n);
}

// refines the space-time mesh while updating bot_to_top relation
void ParMeshCyl::Refine(int par_ref_levels)
{
    /*
    if (par_ref_levels != 0)
    {
        MFEM_ABORT("ParMeshCyl::Refine() implementation was not finished \n");
    }
    else
        return;
    */

    FiniteElementCollection * l2_coll_tmp = new L2_FECollection(0, Dimension());
    ParFiniteElementSpace * L2_space_tmp = new ParFiniteElementSpace(this, l2_coll_tmp);

    SparseMatrix * BE_E;
    for (int l = 0; l < par_ref_levels; ++l)
    {
        // create BE_E relation for the mesh before the refinement
        // only for the marked boundary elements
        BE_E = Create_be_to_e("marked");

        // refine the mesh
        UniformRefinement();

        // get the E_e as interpolation matrix in L2_h
        L2_space_tmp->Update();
        SparseMatrix * P_W_l = (SparseMatrix *)L2_space_tmp->GetUpdateOperator();
        SparseMatrix * E_e = Transpose(*P_W_l);

        // create e_be relation for the refined mesh
        SparseMatrix * be_e = Create_be_to_e("full");

        // compute BE_be relation
        // BE_be = BE_E * E_e * e _be
        SparseMatrix * tmp1 = Transpose(*BE_E);
        SparseMatrix * tmp2 = Transpose(*be_e);

        // now BE_be is relation for BE and be which belong to the same AE
        // thus, row of BE contains not only be indices which are exactly
        // children of BE after refinement, but also some neighbors
        // The additional be's are taken care of below.
        SparseMatrix * BE_AE_be = RAP(*tmp1, *E_e, *tmp2);

        /*
        std::cout << "bdr attributes for be at the fine mesh \n";
        for (int be = 0; be < GetNBE(); ++be)
            std::cout << GetBdrAttribute(be) << " ";
        std::cout << "\n";
        */

        for (int row = 0; row < BE_AE_be->Height(); ++row)
        {
            //std::cout << "row: " << row << "\n";
            int BE_bdrattr;
            if (row < BE_AE_be->Height() / 2)
                BE_bdrattr = 1;
            else
                BE_bdrattr = 3;
            //std::cout << "BE_bdrattr: " << BE_bdrattr << "\n";

            int ncols = BE_AE_be->RowSize(row);
            //std::cout << "ncols: " << ncols << "\n";

            int * cols = BE_AE_be->GetRowColumns(row);
            double * entries = BE_AE_be->GetRowEntries(row);
            for (int j = 0; j < ncols; ++j)
            {
                int col = cols[j];
                //std::cout << "col = " << col << "\n";
                // if be doesn't belong to the bottom or bot boundary
                //std::cout << "its bdr attr = " << GetBdrAttribute(col) << "\n";

                if (GetBdrAttribute(col) != BE_bdrattr)
                    entries[j] = 0.0;
            }
        }

        //E_e->Print();

        //delete P_W_l;
        delete E_e;
        delete tmp1;
        delete tmp2;

        delete BE_E;
        delete be_e;

        //BE_AE_be->Print();

        // checking row sums after we get rid of the be's from the wrong boundary parts
        //Vector row_sums(BE_AE_be->Height());
        //BE_AE_be->GetRowSums(row_sums);
        //row_sums.Print();

        MPI_Comm comm = GetComm();
        int num_procs, myid;
        MPI_Comm_size(comm, &num_procs);
        MPI_Comm_rank(comm, &myid);

        for (int i = 0; i < num_procs; ++i)
        {
            if (myid == i)
            {
                //std::cout << "I am " << myid << "\n";
                UpdateBotToTopLink(*BE_AE_be);

                //std::cout << "\n" << std::flush;
            }
            MPI_Barrier(comm);
        } // end fo loop over all processors, one after another

        /*

        // update the bot_to_top relation using BE_be
        // ...
        UpdateBotToTopLink(*BE_AE_be);
        */

        delete BE_AE_be;
    }

    delete L2_space_tmp;
    delete l2_coll_tmp;
}


// simple algorithm which computes sign of a given permutatation
// for now, this function is applied to permutations of size 3
// so there is no sense in implementing anything more complicated
// the sign is defined so that it is 1 for the loop of length = size
int permutation_sign( int * permutation, int size)
{
    int res = 0;
    int * temp = new int[size]; //visited or not
    for ( int i = 0; i < size; ++i)
        temp[i] = -1;

    int pos = 0;
    while ( pos < size )
    {
        if (temp[pos] == -1) // if element is unvisited
        {
            int cycle_len = 1;

            //computing cycle length which starts with unvisited element
            int k = pos;
            while (permutation[k] != pos )
            {
                temp[permutation[k]] = 1;
                k = permutation[k];
                cycle_len++;
            }
            //cout << "pos = " << pos << endl;
            //cout << "cycle of len " << cycle_len << " was found there" << endl;

            res += (cycle_len-1)%2;

            temp[pos] = 1;
        }

        pos++;
    }

    delete [] temp;

    if (res % 2 == 0)
        return 1;
    else
        return -1;
}

// zero-based indexing for perm_in and perm_out
void invert_permutation(int *perm_in, int size, int * perm_out)
{
  // Inserting position at their
  // respective element in second array
  for (int i = 0; i < size; i++)
    perm_out[perm_in[i]] = i;
}

// zero-based indexing for perm_in and perm_out
void invert_permutation(std::vector<int> perm_in, std::vector<int>& perm_out)
{
  int size = perm_in.size();
  perm_out.resize(size);
  // Inserting position at their
  // respective element in second array
  for (int i = 0; i < size; i++)
    perm_out[perm_in[i]] = i;
}

//used for comparing the d-dimensional points by their coordinates
typedef std::pair<std::vector<double>, int> PairPoint;
struct CmpPairPoint
{
    bool operator()(const PairPoint& a, const PairPoint& b)
    {
        unsigned int size = a.first.size();
        if ( size != b.first.size() )
        {
            std::cerr << "Error: Points have different dimensions" << std::endl << std::flush;
            return false;
        }
        else
        {
            for ( unsigned int i = 0; i < size; ++i)
                if ( fabs(a.first[i] - b.first[i]) > 1.0e-15 )
                    return a.first[i] < b.first[i];
            std::cerr << "Error, points are the same!" << std::endl << std::flush;
            std::cerr << "Point 1:" << std::endl;
            for ( unsigned int i = 0; i < size; ++i)
                std::cerr << a.first[i] << " ";
            std::cerr << std::endl;
            std::cerr << "Point 2:" << std::endl;
            for ( unsigned int i = 0; i < size; ++i)
                std::cerr << b.first[i] << " ";
            std::cerr << std::endl << std::flush;
            return false;
        }

    }
};

// takes coordinates of points and returns a permutation which makes the given vertices
// preserve the geometrical order (based on their coordinates comparison)
void sortingPermutationNew( const std::vector<std::vector<double> >& values, int * permutation)
{
    vector<PairPoint> pairs;
    pairs.reserve(values.size());
    for (unsigned int i = 0; i < values.size(); i++)
    {
        //cout << "i = " << i << endl;
        //for (int j = 0; j < values[i].size(); ++j)
            //cout << values[i][j] << " ";
        //cout << endl;
        pairs.push_back(PairPoint(values[i], i));
    }

    sort(pairs.begin(), pairs.end(), CmpPairPoint());

    typedef std::vector<PairPoint>::const_iterator I;
    int count = 0;
    for (I p = pairs.begin(); p != pairs.end(); ++p)
        permutation[count++] = p->second;

    //cout << "inside sorting permutation is" << endl;
    //for ( int i = 0; i < values.size(); ++i)
        //cout << permutation[i] << " ";
    //cout << endl;
}

// M and N are two d-dimensional points 9double * arrays with their coordinates
inline double dist( double * M, double * N , int d)
{
    double res = 0.0;
    for ( int i = 0; i < d; ++i )
        res += (M[i] - N[i])*(M[i] - N[i]);
    return sqrt(res);
}


int setzero(Array2D<int>* arrayint)
{
    for ( int i = 0; i < arrayint->NumRows(); ++i )
        for ( int j = 0; j < arrayint->NumCols(); ++j)
            (*arrayint)(i,j) = 0;
    return 0;
}

int ipow(int base, int exp)
{
    int result = 1;
    while (exp)
    {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        base *= base;
    }

    return result;
}



} // end of namespace mfem

#endif
