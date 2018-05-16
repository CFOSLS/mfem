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

#ifndef MFEM_PMESH
#define MFEM_PMESH

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "../general/communication.hpp"
#include "mesh.hpp"
#include "pncmesh.hpp"
#include <iostream>

namespace mfem
{

/// Class for parallel meshes
class ParMesh : public Mesh
{
protected:
   ParMesh() : MyComm(0), NRanks(0), MyRank(-1),
      have_face_nbr_data(false), pncmesh(NULL) {}

   MPI_Comm MyComm;
   int NRanks, MyRank;

   Array<Element *> shared_edges;
   Array<Element *> shared_planars;
   Array<Element *> shared_faces;

   /// Shared objects in each group.
   Table group_svert;
   Table group_sedge;
   Table group_splan;
   Table group_sface;

   /// Shared to local index mapping.
   Array<int> svert_lvert;
   Array<int> sedge_ledge;
   Array<int> splan_lplan;
   Array<int> sface_lface;

   /// Create from a nonconforming mesh.
   ParMesh(const ParNCMesh &pncmesh);

   // Mark all tets to ensure consistency across MPI tasks; also mark the
   // shared and boundary triangle faces using the consistently marked tets.
   virtual void MarkTetMeshForRefinement(DSTable &v_to_v);

   /// Return a number(0-1) identifying how the given edge has been split
   int GetEdgeSplittings(Element *edge, const DSTable &v_to_v, int *middle);
   /// Return a number(0-4) identifying how the given face has been split
   int GetFaceSplittings(Element *face, const DSTable &v_to_v, int *middle);

   void GetFaceNbrElementTransformation(
      int i, IsoparametricTransformation *ElTr);

   ElementTransformation* GetGhostFaceTransformation(
      FaceElementTransformations* FETr, int face_type, int face_geom);

   /// Refine quadrilateral mesh.
   virtual void QuadUniformRefinement();

   /// Refine a hexahedral mesh.
   virtual void HexUniformRefinement();

   virtual void NURBSUniformRefinement();

   /// This function is not public anymore. Use GeneralRefinement instead.
   virtual void LocalRefinement(const Array<int> &marked_el, int type = 3);

   /// This function is not public anymore. Use GeneralRefinement instead.
   virtual void NonconformingRefinement(const Array<Refinement> &refinements,
                                        int nc_limit = 0);

   virtual bool NonconformingDerefinement(Array<double> &elem_error,
                                          double threshold, int nc_limit = 0,
                                          int op = 1);
   void DeleteFaceNbrData();

   bool WantSkipSharedMaster(const NCMesh::Master &master) const;

public:
   /** Copy constructor. Performs a deep copy of (almost) all data, so that the
       source mesh can be modified (e.g. deleted, refined) without affecting the
       new mesh. If 'copy_nodes' is false, use a shallow (pointer) copy for the
       nodes, if present. */
   explicit ParMesh(const ParMesh &pmesh, bool copy_nodes = true);

   ParMesh(MPI_Comm comm, Mesh &mesh, int *partitioning_ = NULL,
           int part_method = 1);

   /// Read a parallel mesh, each MPI rank from its own file/stream.
   ParMesh(MPI_Comm comm, std::istream &input);

   /// Create a uniformly refined (by any factor) version of @a orig_mesh.
   /** @param[in] orig_mesh  The starting coarse mesh.
       @param[in] ref_factor The refinement factor, an integer > 1.
       @param[in] ref_type   Specify the positions of the new vertices. The
                             options are BasisType::ClosedUniform or
                             BasisType::GaussLobatto.

       The refinement data which can be accessed with GetRefinementTransforms()
       is set to reflect the performed refinements.

       @note The constructed ParMesh is linear, i.e. it does not have nodes. */
   ParMesh(ParMesh *orig_mesh, int ref_factor, int ref_type);

   MPI_Comm GetComm() const { return MyComm; }
   int GetNRanks() const { return NRanks; }
   int GetMyRank() const { return MyRank; }

   GroupTopology gtopo;

   // Face-neighbor elements and vertices
   bool             have_face_nbr_data;
   Array<int>       face_nbr_group;
   Array<int>       face_nbr_elements_offset;
   Array<int>       face_nbr_vertices_offset;
   Array<Element *> face_nbr_elements;
   Array<Vertex>    face_nbr_vertices;
   // Local face-neighbor elements and vertices ordered by face-neighbor
   Table            send_face_nbr_elements;
   Table            send_face_nbr_vertices;

   ParNCMesh* pncmesh;

   int GetNGroups() const { return gtopo.NGroups(); }

   ///@{ @name These methods require group > 0
   int GroupNVertices(int group) { return group_svert.RowSize(group-1); }
   int GroupNEdges(int group)    { return group_sedge.RowSize(group-1); }
   int GroupNPlanars(int group)  { return group_splan.RowSize(group-1); }
   int GroupNFaces(int group)    { return group_sface.RowSize(group-1); }

   int GroupVertex(int group, int i)
   { return svert_lvert[group_svert.GetRow(group-1)[i]]; }
   void GroupEdge(int group, int i, int &edge, int &o);
   void GroupPlanar(int group, int i, int &planar, int &o);
   void GroupFace(int group, int i, int &face, int &o);
   ///@}

   void GenerateOffsets(int N, HYPRE_Int loc_sizes[],
                        Array<HYPRE_Int> *offsets[]) const;

   void ExchangeFaceNbrData();
   void ExchangeFaceNbrNodes();

   int GetNFaceNeighbors() const { return face_nbr_group.Size(); }
   int GetFaceNbrGroup(int fn) const { return face_nbr_group[fn]; }
   int GetFaceNbrRank(int fn) const;

   /** Similar to Mesh::GetFaceToElementTable with added face-neighbor elements
       with indices offset by the local number of elements. */
   Table *GetFaceToAllElementTable() const;

   /** Get the FaceElementTransformations for the given shared face (edge 2D).
       In the returned object, 1 and 2 refer to the local and the neighbor
       elements, respectively. */
   FaceElementTransformations *
   GetSharedFaceTransformations(int sf, bool fill2 = true);

   /// Return the number of shared faces (3D), edges (2D), vertices (1D)
   int GetNSharedFaces() const;

   /// Return the local face index for the given shared face.
   int GetSharedFace(int sface) const;

   /// See the remarks for the serial version in mesh.hpp
   virtual void ReorientTetMesh();

   /// Utility function: sum integers from all processors (Allreduce).
   virtual long ReduceInt(int value) const;

   /// Update the groups after tet refinement
   void RefineGroups(const DSTable &v_to_v, int *middle);

   /// Load balance the mesh. NC meshes only.
   void Rebalance();

   /** Print the part of the mesh in the calling processor adding the interface
       as boundary (for visualization purposes) using the mfem v1.0 format. */
   virtual void Print(std::ostream &out = std::cout) const;

   /** Print the part of the mesh in the calling processor adding the interface
       as boundary (for visualization purposes) using Netgen/Truegrid format .*/
   virtual void PrintXG(std::ostream &out = std::cout) const;

   /** Write the mesh to the stream 'out' on Process 0 in a form suitable for
       visualization: the mesh is written as a disjoint mesh and the shared
       boundary is added to the actual boundary; both the element and boundary
       attributes are set to the processor number.  */
   void PrintAsOne(std::ostream &out = std::cout);

   /// Old mesh format (Netgen/Truegrid) version of 'PrintAsOne'
   void PrintAsOneXG(std::ostream &out = std::cout);

   /// Returns the minimum and maximum corners of the mesh bounding box. For
   /// high-order meshes, the geometry is refined first "ref" times.
   void GetBoundingBox(Vector &p_min, Vector &p_max, int ref = 2);

   void GetCharacteristics(double &h_min, double &h_max,
                           double &kappa_min, double &kappa_max);

   /// Print various parallel mesh stats
   virtual void PrintInfo(std::ostream &out = std::cout);

   /// Save the mesh in a parallel mesh format.
   void ParPrint(std::ostream &out) const;

   virtual ~ParMesh();

   // Outputs information about shared entites, applying vertex indices permutation if provided
   void PrintSharedStructParMesh ( int * permutation = NULL );

   // TODO: Remove this if clean code is needed.
   // It's a temporary crutch because meshgen functions use protected members of ParMesh
   friend class ParMeshCyl;
};

/*
/// Class for a single time slab mesh
class ParMeshTSL : public ParMesh
{
protected:
    std::vector<int> bottom_brdel_indices;
    std::vector<int> top_brdel_indices;
    bool was_extracted;
    // link between element indices in the time slab and element indices in the parent global mesh
    std::vector<std::pair<int,int> > el_link;
    // links between bottom and top boundary element in the time slab and faces in the parent global mesh
    std::vector<std::pair<int,int> > bot_brdel_link;
    std::vector<std::pair<int,int> > top_brdel_link;
public:
    ParMeshTSL();

    friend class ParMeshParareal;
};
*/

/// Class for parallel meshes in time-slabbing framework (old choice, before considering parareal)
class ParMeshCyl : public ParMesh
{
public:
    ParMesh & meshbase;
    std::vector<std::pair<int,int> > bot_to_top_bels;

    // TODO: Get rid of this, it doesn't belong to this class
    // TODO: But only after the new interface is completed and tested
    // additional structures created in the constructor which is used for time slab extraction if needed
    struct Slabs_Structure
    {
        int nslabs;
        Array<int> el_slabs_markers;
        Array<int> bdrel_slabs_markers;
        Array<int> slabs_offsets; // im time steps

        Slabs_Structure(int Nslabs, int Nlayers, Array<int>* Slabs_widths) : nslabs(Nslabs)
        {
            slabs_offsets.SetSize(nslabs + 1);
            slabs_offsets[0] = 0;
            MFEM_ASSERT(nslabs > 0, "Number of time slabs must be positive!");
            if (nslabs > 1)
            {
                MFEM_ASSERT(Slabs_widths, "For nslabs > 1, slabs widths must be provided \n");
                MFEM_ASSERT(Slabs_widths->Size() == nslabs, "Slabs widths (Array) size mismatch number of time slabs \n");
                for (int i = 0; i < nslabs; ++i)
                    slabs_offsets[i + 1] = (*Slabs_widths)[i];
                slabs_offsets.PartialSum();
                MFEM_ASSERT(slabs_offsets[nslabs] == Nlayers, "Total number of time steps mismatch"
                                                        " provided time slabs widths \n");
            }
            else
                slabs_offsets[1] = Nlayers;
        }

        Slabs_Structure(Slabs_Structure& slabs_structure)
        {
            nslabs = slabs_structure.nslabs;
            slabs_offsets.SetSize(nslabs + 1);
            for (int i = 0; i < slabs_offsets.Size(); ++i)
                slabs_offsets[i] = slabs_structure.slabs_offsets[i];
        }
    } * slabs_struct;

    bool have_slabs_structure;

public:
    // Warning: only PENTATOPE case for 4D mesh and TETRAHEDRON case for 3D are considered

   // Actual parallel 3D->4D/2D->3D mesh generator, version 2 (main).
   // bnd_method: way to create boundary elements
   // bnd_method = 0: el_to_face is not used, face_bndflags not created, but log searches are used
   // for creating boundary elements
   // bnd_method = 1: a little bit more memory but no log searches for creating boundary elements
   // local_method: way to create pentatopes for space-time prisms
   // local_method = 0: ~ SHORTWAY, qhull is used for space-time prisms
   // local_method = 1: ~ LONGWAY, qhull is used for lateral faces of space-time prisms (then combined)
   // local_method = 2: qhull is not used, a simple procedure for simplices is used.

   ParMeshCyl(MPI_Comm comm, ParMesh& Meshbase, double Tinit, double Tau, int Nsteps, int bnd_method, int local_method,
              int Nslabs, Array<int>* Slabs_widths);
   ParMeshCyl(MPI_Comm comm, ParMesh& Meshbase, double Tinit, double Tau, int Nsteps, int Nslabs, Array<int>* Slabs_widths)
       : ParMeshCyl(comm, Meshbase, Tinit, Tau, Nsteps, 1, 2, Nslabs, Slabs_widths) {}
   ParMeshCyl(MPI_Comm comm, ParMesh& Meshbase, double Tinit, double Tau, int Nsteps, int bnd_method, int local_method)
       : ParMeshCyl(comm, Meshbase, Tinit, Tau, Nsteps, bnd_method, local_method, 1, NULL) {}
   ParMeshCyl(MPI_Comm comm, ParMesh& Meshbase, double Tinit, double Tau, int Nsteps)
       : ParMeshCyl(comm, Meshbase, Tinit, Tau, Nsteps, 1, 2) {}

   ParMeshCyl(ParMeshCyl& pmeshcyl);

   void PrintSlabsStruct();

protected:
   // Creates ParMesh internal structure (including shared entities)
   // after the main arrays (elements, vertices and boundary) are already defined for the
   // future space-time mesh. Used only inside the ParMesh constructor.
   void ParMeshSpaceTime_createShared(MPI_Comm comm, int Nsteps );
   void CreateInternalMeshStructure (int refine);

   SparseMatrix *Create_be_to_e(const char *full_or_marked);
public:
   void Refine(int par_ref_levels); // remove this
   //ParMesh *ExtractTimeSlab(int slab_index);
private:
   void Find_be_ordering(SparseMatrix& BE_AE_be, int BE_index, std::vector<int> *be_indices,
                         std::vector<int> *ordering, bool verbose = false);

public:

   // A simple structure which is used to store temporarily the 4d mesh main arrays in
   // parallel mesh generator, version 1.
   struct IntermediateMesh{
       int dim;
       int ne;
       int nv;
       int nbe;

       int * elements;
       int * bdrelements;
       double * vertices;
       int * elattrs;
       int * bdrattrs;

       int withgindicesflag;
       int * vert_gindices; // allocated and used only for 4d parmesh construction
   };
private:
   // Creates an IntermediateMesh whihc stores main arrays of the Mesh.
   IntermediateMesh * ExtractMeshToInterMesh ();
   // Allocation of IntermediateMesh structure.
   void IntermeshInit( IntermediateMesh * intermesh, int dim, int nv, int ne, int nbdr, int with_gindices_flag);
   void IntermeshDelete( IntermediateMesh * intermesh_pt);
   void InterMeshPrint (IntermediateMesh * local_intermesh, int suffix, const char * filename);
   // This function only creates elements, vertices and boundary elements and
   // stores them in the output IntermediateMesh structure. It is used in ParMesh-to-Mesh version
   // of space-time mesh constructor.
   // Description of bnd_method, local_method - see in the constructor which calls this function.
   IntermediateMesh * MeshSpaceTimeCylinder_toInterMesh (double tau, int Nsteps, int bnd_method, int local_method);

   // Calls InitMesh() and creates elements, vertices and boundary elements
   // Used only as part of Mesh constructor thus private.
   // Description of bnd_method, local_method - see in the constructor which calls this function.
   void MeshSpaceTimeCylinder_onlyArrays (double tinit, double tau, int Nsteps,
                                          int bnd_method, int local_method);

   // Reads the elements, vertices and boundary from the input IntermediatMesh.
   // It is like Load() in MFEM but for arrays instead of an input stream.
   // No internal mesh structures are initialized inside.
   void LoadMeshfromArrays( int nv, double * vertices,
                                     int ne, int * elements, int * elattris,
                                     int nbe, int * bdrelements, int * bdrattrs, int dim );

public:
   // Computes domain and boundary volumes, and checks,
   // that faces and boundary elements list is consistent with the actual element faces
   int MeshCheck (bool verbose);

   void PrintBotToTopBels() const;
   ParMesh* GetBaseParMesh() {return &meshbase;}
   void TimeShift(double shift);
protected:
   // takes the BE_be relation between marked BE's and be's which belong to the same AE
   // and creates a new bot_to_top link between boundary elements.
   // Used in Refine().
   void UpdateBotToTopLink(SparseMatrix& BE_AE_be, bool verbose = false);
};

/*
/// Class for a global domain mesh in parareal setup
class ParMeshParareal : public ParMeshCyl
{
protected:
    std::vector<ParMeshTSL*> extracted_tslabs;
    std::vector<std::vector<std::pair<int,int> > > extracted_el_links;
    std::vector<std::vector<std::pair<int,int> > > extracted_bot_brdel_links;
    std::vector<std::vector<std::pair<int,int> > > extracted_top_brdel_links;
public:
    ParMeshParareal();
    ParMeshTSL * ExtractTimeSlab(int tslab);
};
*/

inline double dist( double * M, double * N , int d);
int setzero(Array2D<int>* arrayint);
void sortingPermutationNew( const std::vector<std::vector<double> >& values, int * permutation);
int permutation_sign( int * permutation, int size);
void invert_permutation(int *perm_in, int size, int * perm_out);
void invert_permutation(std::vector<int> perm_in, std::vector<int> &perm_out);
int ipow(int base, int exp);

} // end of namespace mfem

#endif // MFEM_USE_MPI

#endif
