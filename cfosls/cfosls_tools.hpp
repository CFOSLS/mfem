#include <iostream>
#include <deque>
#include "testhead.hpp"

#ifndef MFEM_CFOSLS_TOOLS
#define MFEM_CFOSLS_TOOLS

// some constants used for vtk visualization
#define VTKTETRAHEDRON 10
#define VTKWEDGE 13
#define VTKTRIANGLE 5
#define VTKQUADRIL 9

using namespace std;
using namespace mfem;

/// TODO: Replace references by pointers for return arguments to the
/// TODO: dynamically allocated objects (e.g., bdr attributes)

namespace mfem
{

class FOSLSEstimator;
class LocalProblemSolver;
class HcurlGSSSmoother;
class CoarsestProblemSolver;
class CoarsestProblemHcurlSolver;
class FOSLSProblem;
class FOSLSFEFormulation;
class MonolithicGSBlockSmoother;

SparseMatrix * RemoveZeroEntries(const SparseMatrix& in);

HypreParMatrix * CreateRestriction(const char * top_or_bot, ParFiniteElementSpace& pfespace,
                                   std::vector<std::pair<int,int> >& bot_to_top_tdofs_link);
std::vector<std::pair<int,int> >* CreateBotToTopDofsLink(const char * eltype, FiniteElementSpace& fespace,
                                                         std::vector<std::pair<int,int> > & bot_to_top_bels, bool verbose = false);

void Eliminate_ib_block(HypreParMatrix& Op_hpmat, const Array<int>& EssBdrTrueDofs_dom, const Array<int>& EssBdrTrueDofs_range );
void Eliminate_bb_block(HypreParMatrix& Op_hpmat, const Array<int>& EssBdrTrueDofs );

/// Conjugate gradient method which checks for boundary conditions (used for debugging)
class CGSolver_mod : public CGSolver
{
protected:
    Array<int>& check_indices;

    bool IndicesAreCorrect(const Vector& vec) const;

public:
   CGSolver_mod(Array<int>& Check_Indices) : CGSolver(), check_indices(Check_Indices) {}

#ifdef MFEM_USE_MPI
   CGSolver_mod(MPI_Comm _comm, Array<int>& Check_Indices) : CGSolver(_comm), check_indices(Check_Indices) { }
#endif

   virtual void Mult(const Vector &b, Vector &x) const;

};

/// Conjugate gradient method which stops based on the values of true residual in
/// weighted L2 norm unlike MFEM's implementation which uses preconditioned residual
class CGSolver_mod2 : public CGSolver
{
protected:
public:
   CGSolver_mod2() : CGSolver() {}

#ifdef MFEM_USE_MPI
   CGSolver_mod2(MPI_Comm _comm) : CGSolver(_comm) { }
#endif

   virtual void Mult(const Vector &b, Vector &x) const;
};



// a class for square block operators where each block is given as a HypreParMatrix
// used as an interface to handle coarsened operators for multigrid
// TODO: Who should delete the matrices?
class BlkHypreOperator : public Operator
{
protected:
    int numblocks;
    Array2D<HypreParMatrix*> hpmats;
    Array<int> block_offsets;
public:
    BlkHypreOperator(Array2D<HypreParMatrix*> & Hpmats)
        : numblocks(Hpmats.NumRows())
    {
        hpmats.SetSize(numblocks, numblocks);
        for (int i = 0; i < numblocks; ++i )
            for (int j = 0; j < numblocks; ++j )
                if (Hpmats(i,j))
                    hpmats(i,j) = Hpmats(i,j);
                else
                    hpmats(i,j) = NULL;


        block_offsets.SetSize(numblocks + 1);
        block_offsets[0] = 0;
        for (int i = 0; i < numblocks; ++i )
            block_offsets[i + 1] = hpmats(i,i)->Height();
        block_offsets.PartialSum();
    }

    virtual void Mult(const Vector &x, Vector &y) const;
    virtual void MultTranspose(const Vector &x, Vector &y) const;
};

struct BdrConditions
{
protected:
    int numblocks;
    int nattribs;
    bool initialized;
protected:
    std::vector<Array<int>* > bdr_attribs;

public:
    BdrConditions(ParMesh& pmesh_, int nblocks)
    : numblocks(nblocks), nattribs(pmesh_.bdr_attributes.Max()), initialized(false)
    {
        bdr_attribs.resize(numblocks);
        for (unsigned int i = 0; i < bdr_attribs.size(); ++i)
        {
            bdr_attribs[i] = new Array<int>(nattribs);
            for (int j = 0; j < bdr_attribs[i]->Size(); ++j)
                (*bdr_attribs[i])[j] = -1;
        }

    }

    BdrConditions(const std::vector<Array<int>* >& bdr_attribs_)
        : numblocks(bdr_attribs_.size())
    {
        MFEM_ASSERT(bdr_attribs_[0], "NULL check failed");
        nattribs = bdr_attribs_[0]->Size();

        bdr_attribs.resize(numblocks);
        for (unsigned int i = 0; i < bdr_attribs.size(); ++i)
        {
            MFEM_VERIFY(bdr_attribs_[i]->Size() == nattribs,"");
            bdr_attribs[i] = new Array<int>(nattribs);
            for (int j = 0; j < bdr_attribs[i]->Size(); ++j)
                (*bdr_attribs[i])[j] = (*bdr_attribs_[i])[j];
        }

        initialized = true;
    }

    BdrConditions(const std::vector<const Array<int>* >& bdr_attribs_)
        : numblocks(bdr_attribs_.size())
    {
        MFEM_ASSERT(bdr_attribs_[0], "NULL check failed");
        nattribs = bdr_attribs_[0]->Size();

        bdr_attribs.resize(numblocks);
        for (unsigned int i = 0; i < bdr_attribs.size(); ++i)
        {
            MFEM_VERIFY(bdr_attribs_[i]->Size() == nattribs,"");
            bdr_attribs[i] = new Array<int>(nattribs);
            for (int j = 0; j < bdr_attribs[i]->Size(); ++j)
                (*bdr_attribs[i])[j] = (*bdr_attribs_[i])[j];
        }

        initialized = true;
    }

    std::vector< Array<int>* >& GetAllBdrAttribs()
    {
        MFEM_ASSERT(initialized, "Boundary conditions were not initialized \n");
        return bdr_attribs;
    }

    const Array<int>& GetBdrAttribs(int blk)
    {
        MFEM_ASSERT(initialized, "Boundary conditions were not initialized \n");
        if (!(blk >= 0 && blk < numblocks))
            std::cout << "Breakpoint \n";
        MFEM_ASSERT(blk >= 0 && blk < numblocks,
                    "Invalid block number in BdrConditions::GetBdrAttribs()");

        return *bdr_attribs[blk];
    }

    bool Initialized() const {return initialized;}

    // copies the provided bdr_attribs inside
    void Set(const std::vector<Array<int>* >& bdr_attribs_);

    void Reset()
    {
        MFEM_ASSERT(initialized, "Cannot reset bdr conditions which were not initialized");
        for (unsigned int i = 0; i < bdr_attribs.size(); ++i)
            delete bdr_attribs[i];
        initialized = false;
    }
};

struct BdrConditions_CFOSLS_HdivL2_Hyper : public BdrConditions
{
public:
    BdrConditions_CFOSLS_HdivL2_Hyper(ParMesh& pmesh_)
        : BdrConditions(pmesh_, 2)
    {
        for (int j = 0; j < bdr_attribs[0]->Size(); ++j)
            (*bdr_attribs[0])[j] = 0;
        (*bdr_attribs[0])[0] = 1;

        for (int j = 0; j < bdr_attribs[1]->Size(); ++j)
            (*bdr_attribs[1])[j] = 0;

        initialized = true;
    }
};

struct BdrConditions_CFOSLS_HdivL2L2_Hyper : public BdrConditions
{
public:
    BdrConditions_CFOSLS_HdivL2L2_Hyper(ParMesh& pmesh_)
        : BdrConditions(pmesh_, 3)
    {
        for (int j = 0; j < bdr_attribs[0]->Size(); ++j)
            (*bdr_attribs[0])[j] = 0;
        (*bdr_attribs[0])[0] = 1;

        for (int j = 0; j < bdr_attribs[1]->Size(); ++j)
            (*bdr_attribs[1])[j] = 0;

        for (int j = 0; j < bdr_attribs[2]->Size(); ++j)
            (*bdr_attribs[2])[j] = 0;

        initialized = true;
    }

};

struct BdrConditions_CFOSLS_HdivH1_Hyper : public BdrConditions
{
public:
    BdrConditions_CFOSLS_HdivH1_Hyper(ParMesh& pmesh_)
        : BdrConditions(pmesh_, 3)
    {
        for (int j = 0; j < bdr_attribs[0]->Size(); ++j)
            (*bdr_attribs[0])[j] = 0;

        for (int j = 0; j < bdr_attribs[1]->Size(); ++j)
            (*bdr_attribs[1])[j] = 0;
        (*bdr_attribs[1])[0] = 1;

        for (int j = 0; j < bdr_attribs[2]->Size(); ++j)
            (*bdr_attribs[2])[j] = 0;

        initialized = true;
    }

};

struct BdrConditions_CFOSLS_HdivH1_Parab : public BdrConditions
{
public:
    BdrConditions_CFOSLS_HdivH1_Parab(ParMesh& pmesh_)
        : BdrConditions(pmesh_, 3)
    {
        for (int j = 0; j < bdr_attribs[0]->Size(); ++j)
            (*bdr_attribs[0])[j] = 0;

        for (int j = 0; j < bdr_attribs[1]->Size(); ++j)
            (*bdr_attribs[1])[j] = 1;
        (*bdr_attribs[1])[bdr_attribs[1]->Size() - 1] = 0;

        for (int j = 0; j < bdr_attribs[2]->Size(); ++j)
            (*bdr_attribs[2])[j] = 0;

        initialized = true;
    }
};

struct BdrConditions_CFOSLS_HdivH1_Wave : public BdrConditions
{
public:
    BdrConditions_CFOSLS_HdivH1_Wave(ParMesh& pmesh_)
        : BdrConditions(pmesh_, 3)
    {
        for (int j = 0; j < bdr_attribs[0]->Size(); ++j)
            (*bdr_attribs[0])[j] = 0;
        (*bdr_attribs[0])[0] = 1;

        for (int j = 0; j < bdr_attribs[1]->Size(); ++j)
            (*bdr_attribs[1])[j] = 1;
        (*bdr_attribs[1])[bdr_attribs[1]->Size() - 1] = 0;

        for (int j = 0; j < bdr_attribs[2]->Size(); ++j)
            (*bdr_attribs[2])[j] = 0;

        initialized = true;
    }

};

struct BdrConditions_CFOSLS_HdivH1Laplace : public BdrConditions
{
public:
    BdrConditions_CFOSLS_HdivH1Laplace(ParMesh& pmesh_)
        : BdrConditions(pmesh_, 3)
    {
        for (int j = 0; j < bdr_attribs[0]->Size(); ++j)
            (*bdr_attribs[0])[j] = 0;

        for (int j = 0; j < bdr_attribs[1]->Size(); ++j)
            (*bdr_attribs[1])[j] = 1;

        for (int j = 0; j < bdr_attribs[2]->Size(); ++j)
            (*bdr_attribs[2])[j] = 0;

        initialized = true;
    }
};

// one of the choices which is not from a singular problem
struct BdrConditions_MixedLaplace : public BdrConditions
{
public:
    BdrConditions_MixedLaplace(ParMesh& pmesh_)
        : BdrConditions(pmesh_, 2)
    {
        for (int j = 0; j < bdr_attribs[0]->Size(); ++j)
            (*bdr_attribs[0])[j] = 0;
        (*bdr_attribs[0])[0] = 1;

        for (int j = 0; j < bdr_attribs[1]->Size(); ++j)
            (*bdr_attribs[1])[j] = 0;

        initialized = true;
    }

};

// one of the choices which is not from a singular problem
struct BdrConditions_Laplace : public BdrConditions
{
public:
    BdrConditions_Laplace(ParMesh& pmesh_)
        : BdrConditions(pmesh_, 1)
    {
        *bdr_attribs[0] = 1;
        initialized = true;
    }

};

enum SpaceName {HDIV = 0, H1 = 1, L2 = 2, HCURL = 3, HDIVSKEW = 4};

class FOSLSFormulation;

// a class for hierarchy of spaces of finite element spaces based on a nested sequence of meshes
class GeneralHierarchy
{
protected:
    int num_lvls;

    // the finest mesh (a copy of it is also stored in pmesh_lvls)
    // used for updating the hierarchy when the hierarchy's finest mesh is refined
    ParMesh& pmesh;
    // pfespaces and fecolls which live on the pmesh
    // used for updating the hierarchy
    ParFiniteElementSpace * Hdiv_space;
    ParFiniteElementSpace * Hcurl_space;
    ParFiniteElementSpace * H1_space;
    ParFiniteElementSpace * L2_space;
    ParFiniteElementSpace * Hdivskew_space;

    FiniteElementCollection *hdiv_coll;
    FiniteElementCollection *hcurl_coll;
    FiniteElementCollection *h1_coll;
    FiniteElementCollection *l2_coll;
    FiniteElementCollection *hdivskew_coll;

    /// stores meshes at all levels
    /// when the hierarchy gets more levels, new meshes are prepended
    /// but not changed!
    /// (*) for dynamic update of an aboject build on finest level of the hierarchy
    /// one should use pmesh and corresponding spaces
    Array<ParMesh*> pmesh_lvls;
    Array<ParFiniteElementSpace* > Hdiv_space_lvls;
    Array<ParFiniteElementSpace* > Hcurl_space_lvls;
    Array<ParFiniteElementSpace* > H1_space_lvls;
    Array<ParFiniteElementSpace* > L2_space_lvls;
    Array<ParFiniteElementSpace* > Hdivskew_space_lvls;

    Array<SparseMatrix*> P_H1_lvls;
    Array<SparseMatrix*> P_Hdiv_lvls;
    Array<SparseMatrix*> P_L2_lvls;
    Array<SparseMatrix*> P_Hcurl_lvls;
    Array<SparseMatrix*> P_Hdivskew_lvls;

    Array<HypreParMatrix*> TrueP_H1_lvls;
    Array<HypreParMatrix*> TrueP_Hdiv_lvls;
    Array<HypreParMatrix*> TrueP_L2_lvls;
    Array<HypreParMatrix*> TrueP_Hcurl_lvls;
    Array<HypreParMatrix*> TrueP_Hdivskew_lvls;

    Array<const HypreParMatrix*> DivfreeDops_lvls;

    Array< HypreParMatrix* > DofTrueDof_L2_lvls;
    Array< HypreParMatrix* > DofTrueDof_H1_lvls;
    Array< HypreParMatrix* > DofTrueDof_Hdiv_lvls;
    Array< HypreParMatrix* > DofTrueDof_Hcurl_lvls;
    Array< HypreParMatrix* > DofTrueDof_Hdivskew_lvls;

    bool with_hcurl;

    bool divfreedops_constructed;
    bool doftruedofs_constructed;

    int pmesh_ne;

    int update_counter;

    Array<FOSLSProblem*> problems;

public:
    // by default we construct div-free space (Hcurl) in 3D
    GeneralHierarchy(int num_levels, ParMesh& pmesh_, int feorder, bool verbose)
        : GeneralHierarchy(num_levels, pmesh_, feorder, verbose, true) {}
    // but we might want not to do so, due to the limitations of higher-order Nedelec spaces
    // w.r.t to further mesh refinement
    GeneralHierarchy(int num_levels, ParMesh& pmesh_, int feorder, bool verbose, bool with_hcurl_);

    ParMesh* GetFinestParMesh() {return &pmesh;}

    // should be called if the finest mesh was refined and
    // one wants to extend the hierarchy
    virtual void Update();

    void ConstructDivfreeDops();
    void ConstructDofTrueDofs();

    void RefineAndCopy(int lvl, ParMesh* pmesh);

    ParMesh * GetPmesh(int l) {return pmesh_lvls[l];}

    // probably should be all replaced by GetSpace()
    ParFiniteElementSpace * GetHdiv_space(int l) {return Hdiv_space_lvls[l];}
    ParFiniteElementSpace * GetH1_space(int l) {return H1_space_lvls[l];}
    ParFiniteElementSpace * GetL2_space(int l) {return L2_space_lvls[l];}
    ParFiniteElementSpace * GetHcurl_space(int l) {return Hcurl_space_lvls[l];}

    SparseMatrix * GetP_Hdiv(int l) {return P_Hdiv_lvls[l];}
    SparseMatrix * GetP_H1(int l) {return P_H1_lvls[l];}
    SparseMatrix * GetP_L2(int l) {return P_L2_lvls[l];}
    SparseMatrix * GetP_Hcurl(int l) {return P_Hcurl_lvls[l];}

    HypreParMatrix * GetTrueP_Hdiv(int l) {return TrueP_Hdiv_lvls[l];}
    HypreParMatrix * GetTrueP_H1(int l) {return TrueP_H1_lvls[l];}
    HypreParMatrix * GetTrueP_L2(int l) {return TrueP_L2_lvls[l];}
    HypreParMatrix * GetTrueP_Hcurl(int l) {return TrueP_Hcurl_lvls[l];}

    ParFiniteElementSpace *GetSpace(SpaceName space, int level);

    ParFiniteElementSpace *GetFinestSpace(SpaceName space);

    HypreParMatrix * GetTruePspace(SpaceName space, int level);

    SparseMatrix * GetPspace(SpaceName space, int level);

    const HypreParMatrix * GetDivfreeDop(int level)
    {
        MFEM_ASSERT(divfreedops_constructed, "Divfree discrete operators were not constructed!");
        return DivfreeDops_lvls[level];
    }

    int Nlevels() const {return num_lvls;}

    const Array<int>* ConstructTrueOffsetsforFormul(int level, const Array<SpaceName>& space_names);
    BlockOperator* ConstructTruePforFormul(int level, const Array<SpaceName>& space_names,
                                           const Array<int>& row_offsets, const Array<int> &col_offsets);
    BlockOperator* ConstructTruePforFormul(int level, const FOSLSFormulation& formul,
                                           const Array<int>& row_offsets, const Array<int> &col_offsets);

    const Array<int>* ConstructOffsetsforFormul(int level, const Array<SpaceName>& space_names);
    BlockMatrix* ConstructPforFormul(int level, const Array<SpaceName>& space_names,
                                                             const Array<int>& row_offsets, const Array<int>& col_offsets);


    Array<int>& GetEssBdrTdofsOrDofs(const char * tdof_or_dof, SpaceName space_name,
                                           const Array<int>& essbdr_attribs, int level) const;

    std::vector<Array<int>* >& GetEssBdrTdofsOrDofs(const char * tdof_or_dof, const Array<SpaceName>& space_names,
                                                    std::vector<Array<int>*>& essbdr_attribs, int level) const;

    /*
    std::vector<Array<int>* >& GetEssBdrTdofsOrDofs(const char * tdof_or_dof, const Array<SpaceName>& space_names,
                                                    std::vector<const Array<int>*>& essbdr_attribs, int level) const;
    */

    SparseMatrix* GetElementToDofs(SpaceName space_name, int level) const;

    BlockMatrix* GetElementToDofs(const Array<SpaceName>& space_names, int level,
                                  Array<int>& row_offsets, Array<int>& col_offsets) const;

    BlockMatrix* GetElementToDofs(const Array<SpaceName>& space_names, int level,
                                  Array<int>* row_offsets, Array<int>* col_offsets) const;

    HypreParMatrix *GetDofTrueDof(SpaceName space_name, int level) const;
    std::vector<HypreParMatrix*> & GetDofTrueDof(const Array<SpaceName>& space_names, int level) const;
    BlockOperator* GetDofTrueDof(const Array<SpaceName> &space_names, int level,
                                 Array<int>& row_offsets, Array<int>& col_offsets) const;

    int GetUpdateCounter() const {return update_counter;}

    // constructs a FOSLSProblem of given (by template parameter) subtype
    // using the spaces at level l (defined on pmesh_lvls[l])
    template <class Problem> Problem* BuildStaticProblem(int l, BdrConditions& bdr_conditions,
                                                         FOSLSFEFormulation& fe_formulation,
                                                         int prec_option, bool verbose);

    // constructs a FOSLSProblem of given (by template parameter) subtype
    // using the spaces defines at pmesh (dynamically updated instance of the finest pmesh)
    template <class Problem> Problem* BuildDynamicProblem(BdrConditions& bdr_conditions,
                                                          FOSLSFEFormulation& fe_formulation,
                                                          int prec_option, bool verbose);

    // attaches a given problem living at the finest level to the problems (defined on pmesh)
    // which is useful for updating the problem
    void AttachProblem(FOSLSProblem* problem);

    FOSLSProblem* GetProblem(int i) {return problems[i];}
    int Nproblems() const {return problems.Size();}
};

template <class Problem> Problem*
GeneralHierarchy::BuildStaticProblem(int l, BdrConditions& bdr_conditions, FOSLSFEFormulation& fe_formulation,
                                     int prec_option, bool verbose)
{
    return new Problem(*this, l, bdr_conditions, fe_formulation, prec_option, verbose);
}

template <class Problem> Problem*
GeneralHierarchy::BuildDynamicProblem(BdrConditions& bdr_conditions, FOSLSFEFormulation& fe_formulation,
                                      int prec_option, bool verbose)
{
    return new Problem(*this, bdr_conditions, fe_formulation, prec_option, verbose);
}


class GeneralCylHierarchy : public GeneralHierarchy
{
protected:
    Array<ParMeshCyl*> pmeshcyl_lvls;

    std::vector<std::vector<std::pair<int,int> > > tdofs_link_H1_lvls;
    std::vector<std::vector<std::pair<int,int> > > tdofs_link_Hdiv_lvls;

    Array<HypreParMatrix*> TrueP_bndbot_H1_lvls;
    Array<HypreParMatrix*> TrueP_bndbot_Hdiv_lvls;
    Array<HypreParMatrix*> TrueP_bndtop_H1_lvls;
    Array<HypreParMatrix*> TrueP_bndtop_Hdiv_lvls;
    Array<HypreParMatrix*> Restrict_bot_H1_lvls;
    Array<HypreParMatrix*> Restrict_bot_Hdiv_lvls;
    Array<HypreParMatrix*> Restrict_top_H1_lvls;
    Array<HypreParMatrix*> Restrict_top_Hdiv_lvls;
protected:
    void ConstructRestrictions();
    void ConstructInterpolations();
    void ConstructTdofsLinks();

public:
    GeneralCylHierarchy(int num_levels, ParMeshCyl& pmesh, int feorder, bool verbose)
        : GeneralHierarchy(num_levels, pmesh, feorder, verbose)
    {
        pmeshcyl_lvls.SetSize(num_lvls);
        for (int l = 0; l < num_lvls; ++l)
        {
            ParMeshCyl * temp = dynamic_cast<ParMeshCyl*>(pmesh_lvls[l]);
            if (temp)
                pmeshcyl_lvls[l] = temp;
            else
            {
                MFEM_ABORT ("Unsuccessful cast \n");
            }
        }

        // don't change the order of these calls
        ConstructTdofsLinks();
        ConstructRestrictions();
        ConstructInterpolations();
    }

    // should be called if the finest mesh was refined and
    // one wants to extend the hierarchy
    virtual void Update() override { MFEM_ABORT("Update() is not implemented for GeneralCylHierarchy!");}

    ParMeshCyl * GetPmeshcyl(int l) {return pmeshcyl_lvls[l];}

    std::vector<std::pair<int,int> > * GetTdofs_Hdiv_link(int l) {return &(tdofs_link_Hdiv_lvls[l]);}
    std::vector<std::pair<int,int> > * GetTdofs_H1_link(int l) {return &(tdofs_link_H1_lvls[l]);}

    HypreParMatrix * GetTrueP_bndbot_Hdiv (int l) {return TrueP_bndbot_Hdiv_lvls[l];}
    HypreParMatrix * GetTrueP_bndtop_Hdiv (int l) {return TrueP_bndtop_Hdiv_lvls[l];}
    HypreParMatrix * GetTrueP_bndbot_H1 (int l) {return TrueP_bndtop_H1_lvls[l];}
    HypreParMatrix * GetTrueP_bndtop_H1 (int l) {return TrueP_bndtop_H1_lvls[l];}

    HypreParMatrix * GetRestrict_bot_Hdiv (int l) {return Restrict_bot_Hdiv_lvls[l];}
    HypreParMatrix * GetRestrict_top_Hdiv (int l) {return Restrict_top_Hdiv_lvls[l];}
    HypreParMatrix * GetRestrict_bot_H1 (int l) {return Restrict_bot_H1_lvls[l];}
    HypreParMatrix * GetRestrict_top_H1 (int l) {return Restrict_top_H1_lvls[l];}

    int GetLinksize_Hdiv(int l) const {return tdofs_link_Hdiv_lvls[l].size();}
    int GetLinksize_H1(int l) const {return tdofs_link_H1_lvls[l].size();}

    std::vector<std::pair<int,int> > * GetTdofsLink(int level, SpaceName space_name)
    {
        switch(space_name)
        {
        case HDIV:
            return &(tdofs_link_Hdiv_lvls[level]);
        case H1:
            return &(tdofs_link_H1_lvls[level]);
        default:
            {
                MFEM_ABORT("Unknown or unsupported space name in GetTdofsLink() \n");
                break;
            }
        }

        return NULL;
    }

    int GetTdofsLinkSize(int level, SpaceName space_name)
    {
        switch(space_name)
        {
        case HDIV:
            return tdofs_link_Hdiv_lvls[level].size();
        case H1:
            return tdofs_link_H1_lvls[level].size();
        default:
            {
                MFEM_ABORT("Unknown or unsupported space name in GetTdofsLinkSize() \n");
                break;
            }
        }
        return -1;
    }

    HypreParMatrix * GetRestrict_bnd(const char * top_or_bot, int level, SpaceName space_name)
    {
        MFEM_ASSERT(strcmp(top_or_bot,"top") == 0 || strcmp(top_or_bot,"bot") == 0,
                    "top_or_bot must be either 'top' or 'bot'!");

        switch(space_name)
        {
        case HDIV:
            if (strcmp(top_or_bot,"top") == 0)
                return Restrict_top_Hdiv_lvls[level];
            else
                return Restrict_bot_Hdiv_lvls[level];
        case H1:
            if (strcmp(top_or_bot,"top") == 0)
                return Restrict_top_H1_lvls[level];
            else
                return Restrict_bot_H1_lvls[level];
        default:
            {
                MFEM_ABORT("Unknown or unsupported space name in GetRestrict_bnd() \n");
                break;
            }
        }
        return NULL;
    }

    HypreParMatrix * GetTrueP_bnd(const char * top_or_bot, int level, SpaceName space_name)
    {
        MFEM_ASSERT(strcmp(top_or_bot,"top") == 0 || strcmp(top_or_bot,"bot") == 0,
                    "top_or_bot must be either 'top' or 'bot'!");

        switch(space_name)
        {
        case HDIV:
            if (strcmp(top_or_bot,"top") == 0)
                return TrueP_bndtop_Hdiv_lvls[level];
            else
                return TrueP_bndbot_Hdiv_lvls[level];
        case H1:
            if (strcmp(top_or_bot,"top") == 0)
                return TrueP_bndtop_H1_lvls[level];
            else
                return TrueP_bndbot_H1_lvls[level];
        default:
            {
                MFEM_ABORT("Unknown or unsupported space name in GetTrueP_bnd() \n");
                break;
            }
        }

        return NULL;
    }
};

/// abstract structure for a (C)FOSLS formulation
/// CFOSLS is considered to be a FOSLS formulation with constraint
///
/// blk_structure is a vector of size = nblocks which contains:
/// pairs <'a','b'> where, 'a' = 0,1,2 describes the type of the variable
/// and 'b' is the index in the test coefficient array (corresponding to 'a').
/// Values for 'a': 0 (scalar) or 1 (vector).

/// For example, <1,2> at place 3 means that the third equation corresponds
/// to a vector unknown ('a' = 1), for which we have in the FOSLS_test a
/// VectorFunctionCoefficient stored at test.vec_coeffs[2]('b' = 2).
/// If a variable is not present in the FOSLS test (e.g., it's a Lagrange multiplier)
/// then one must set 'a' = -1, 'b' = -1.

/// It is implicitly assumed that first (unknowns_number) of equations
/// are related to the FOSLS functional and the rest (up to the total number
/// equal numblocks) are constrains.
/// Thus, in FOSLSEstimator the first block of (unknowns_number) x (unknowns_number)
/// of integrators is used as functional integrators(forms)
struct FOSLSFormulation
{
protected:
    const int dim;
    const int numblocks;
    const int unknowns_number;
    const bool have_constraint;
    Array2D<BilinearFormIntegrator*> blfis;
    Array<LinearFormIntegrator*> lfis;
    std::vector<std::pair<int,int> > blk_structure;
    mutable Array<SpaceName>* space_names;
    mutable Array<SpaceName>* space_names_funct;
protected:
    virtual void InitBlkStructure() = 0;
    virtual void ConstructSpacesDescriptor() const = 0;
    virtual void ConstructFunctSpacesDescriptor() const = 0;
public:
    FOSLSFormulation(int dimension, int num_blocks, int num_unknowns, bool do_have_constraint);

    const Array<SpaceName>* GetSpacesDescriptor() const
    {
        if (!space_names)
            ConstructSpacesDescriptor();

        return space_names;
    }

    const Array<SpaceName>* GetFunctSpacesDescriptor() const
    {
        if (!space_names_funct)
            ConstructFunctSpacesDescriptor();

        return space_names_funct;
    }

    SpaceName GetSpaceName(int i) const
    {
        if (!space_names)
            ConstructSpacesDescriptor();

        return (*space_names)[i];
    }

    virtual int GetUnknownWithInitCnd() const
    {
        MFEM_ABORT("FOSLSFormulation::GetUnknownWithInitCnd() is not overriden! \n");
        return -1;
    }

    std::pair<int,int>& GetPair(int pair) {return blk_structure[pair];}
    virtual FOSLS_test * GetTest() = 0;

    int Dim() const {return dim;}
    int Nblocks() const {return numblocks;}
    int Nunknowns() const {return unknowns_number;}

    BilinearFormIntegrator* GetBlfi(int i, int j)
    {
        MFEM_ASSERT(i >=0 && i < blfis.NumRows()
                    && j >=0 && j < blfis.NumCols(), "Index pair for blfis out of bounds \n");
        return blfis(i,j);
    }

    LinearFormIntegrator* GetLfi(int i)
    {
        MFEM_ASSERT(i >=0 && i < lfis.Size(), "Index for lfis out of bounds \n");
        return lfis[i];
    }

    virtual int NumSol() const
    { MFEM_ABORT("NumSol() must not be called from the base class FOSLSFormulation! \n"); return -1;}

};

// specific FOSLSFormulations

struct CFOSLSFormulation_HdivL2Hyper : public FOSLSFormulation
{
protected:
    int numsol;
    Hyper_test test;
protected:
    void InitBlkStructure() override;
    void ConstructSpacesDescriptor() const override;
    void ConstructFunctSpacesDescriptor() const override;
public:
    CFOSLSFormulation_HdivL2Hyper(int dimension, int num_solution, bool verbose);

    FOSLS_test * GetTest() override {return &test;}

    int GetUnknownWithInitCnd() const override {return 0;}

    int NumSol() const override {return numsol;}
};

struct CFOSLSFormulation_HdivL2L2Hyper : public FOSLSFormulation
{
protected:
    int numsol;
    Hyper_test test;
protected:
    void InitBlkStructure() override;
    void ConstructSpacesDescriptor() const override;
    void ConstructFunctSpacesDescriptor() const override;
public:
    CFOSLSFormulation_HdivL2L2Hyper(int dimension, int num_solution, bool verbose);

    FOSLS_test * GetTest() override {return &test;}

    int GetUnknownWithInitCnd() const override {return 0;}

    int NumSol() const override {return numsol;}
};

struct CFOSLSFormulation_HdivH1Hyper : public FOSLSFormulation
{
protected:
    int numsol;
    Hyper_test test;
protected:
    void InitBlkStructure() override;
    void ConstructSpacesDescriptor() const override;
    void ConstructFunctSpacesDescriptor() const override;
public:
    CFOSLSFormulation_HdivH1Hyper(int dimension, int num_solution, bool verbose);

    FOSLS_test * GetTest() override {return &test;}

    int GetUnknownWithInitCnd() const override {return 1;}

    int NumSol() const override {return numsol;}
};

struct CFOSLSFormulation_HdivH1DivfreeHyp : public FOSLSFormulation
{
protected:
    int numsol;
    Hyper_test test;
protected:
    void InitBlkStructure() override;
    void ConstructSpacesDescriptor() const override;
    void ConstructFunctSpacesDescriptor() const override;
public:
    CFOSLSFormulation_HdivH1DivfreeHyp(int dimension, int num_solution, bool verbose);

    CFOSLSFormulation_HdivH1DivfreeHyp(CFOSLSFormulation_HdivH1Hyper& hdivh1_formul, bool verbose)
        : CFOSLSFormulation_HdivH1DivfreeHyp(hdivh1_formul.Dim(), hdivh1_formul.NumSol(), verbose) {}

    FOSLS_test * GetTest() override {return &test;}

    int GetUnknownWithInitCnd() const override {return 1;}

};

struct CFOSLSFormulation_HdivDivfreeHyp : public FOSLSFormulation
{
protected:
    int numsol;
    Hyper_test test;
protected:
    void InitBlkStructure() override;
    void ConstructSpacesDescriptor() const override;
    void ConstructFunctSpacesDescriptor() const override;
public:
    CFOSLSFormulation_HdivDivfreeHyp(int dimension, int num_solution, bool verbose);

    CFOSLSFormulation_HdivDivfreeHyp(CFOSLSFormulation_HdivL2Hyper& hdivl2_formul, bool verbose)
        : CFOSLSFormulation_HdivDivfreeHyp(hdivl2_formul.Dim(), hdivl2_formul.NumSol(), verbose) {}

    FOSLS_test * GetTest() override {return &test;}

    int GetUnknownWithInitCnd() const override {return 1;}

};

struct CFOSLSFormulation_HdivH1Parab : public FOSLSFormulation
{
protected:
    int numsol;
    Parab_test test;
protected:
    void InitBlkStructure() override;
    void ConstructSpacesDescriptor() const override;
    void ConstructFunctSpacesDescriptor() const override;
public:
    CFOSLSFormulation_HdivH1Parab(int dimension, int num_solution, bool verbose);

    FOSLS_test * GetTest() override {return &test;}

    int GetUnknownWithInitCnd() const override {return 1;}

    int NumSol() const override {return numsol;}
};

struct CFOSLSFormulation_HdivH1Wave : public FOSLSFormulation
{
protected:
    int numsol;
    Wave_test test;
protected:
    void InitBlkStructure() override;
    void ConstructSpacesDescriptor() const override;
    void ConstructFunctSpacesDescriptor() const override;
public:
    CFOSLSFormulation_HdivH1Wave(int dimension, int num_solution, bool verbose);

    FOSLS_test * GetTest() override {return &test;}

    int GetUnknownWithInitCnd() const override {return 1;}

    int NumSol() const override {return numsol;}
};

// Hdiv-H1 formulation
struct CFOSLSFormulation_Laplace : public FOSLSFormulation
{
protected:
    int numsol;
    Laplace_test test;
protected:
    void InitBlkStructure() override;
    void ConstructSpacesDescriptor() const override;
    void ConstructFunctSpacesDescriptor() const override;
public:
    CFOSLSFormulation_Laplace(int dimension, int num_solution, bool verbose);

    FOSLS_test * GetTest() override {return &test;}

    int GetUnknownWithInitCnd() const override {return 1;}

    int NumSol() const override {return numsol;}
};

struct CFOSLSFormulation_MixedLaplace : public FOSLSFormulation
{
protected:
    int numsol;
    Laplace_test test;
protected:
    void InitBlkStructure() override;
    void ConstructSpacesDescriptor() const override;
    void ConstructFunctSpacesDescriptor() const override;
public:
    CFOSLSFormulation_MixedLaplace(int dimension, int num_solution, bool verbose);

    int GetUnknownWithInitCnd() const override {return 0;}

    FOSLS_test * GetTest() override {return &test;}

    int NumSol() const override {return numsol;}
};

struct FOSLSFormulation_Laplace : public FOSLSFormulation
{
protected:
    int numsol;
    Laplace_test test;
protected:
    void InitBlkStructure() override;
    void ConstructSpacesDescriptor() const override;
    void ConstructFunctSpacesDescriptor() const override;
public:
    FOSLSFormulation_Laplace(int dimension, int num_solution, bool verbose);

    int GetUnknownWithInitCnd() const override {return 0;}

    FOSLS_test * GetTest() override {return &test;}

    int NumSol() const override {return numsol;}
};

/// general class for FOSLS finite element formulations
/// constructed on top of the FOSLS formulation
struct FOSLSFEFormulation
{
protected:
    FOSLSFormulation& formul;
    Array<FiniteElementCollection*> fecolls;
    int feorder;
public:
    FOSLSFEFormulation(FOSLSFormulation& formulation) : FOSLSFEFormulation(formulation, 0) {}
    FOSLSFEFormulation(FOSLSFormulation& formulation, int fe_order) : formul(formulation), feorder(fe_order)
    {
        fecolls.SetSize(formul.Nblocks());
        for (int i = 0; i < formul.Nblocks(); ++i)
            fecolls[i] = NULL;
    }

    FOSLSFormulation * GetFormulation() {return &formul;}

    BilinearFormIntegrator* GetBlfi(int i, int j) {return formul.GetBlfi(i,j);}
    LinearFormIntegrator* GetLfi(int i) {return formul.GetLfi(i);}
    FiniteElementCollection* GetFeColl(int i)
    {
        MFEM_ASSERT( i >= 0 && i < fecolls.Size(), "i < 0 or i > size fo fecolls \n");
        return fecolls[i];
    }

    int Nblocks() const {return formul.Nblocks();}
    int Nunknowns() const {return formul.Nunknowns();}
    int Feorder() const {return feorder;}

};

// specific FOSLSFEFormulation

struct CFOSLSFEFormulation_HdivL2Hyper : FOSLSFEFormulation
{
public:
    CFOSLSFEFormulation_HdivL2Hyper(CFOSLSFormulation_HdivL2Hyper& formulation, int fe_order)
        : FOSLSFEFormulation(formulation, fe_order)
    {
        int dim = formul.Dim();
        if (dim == 4)
            fecolls[0] = new RT0_4DFECollection;
        else
            fecolls[0] = new RT_FECollection(feorder, dim);

        fecolls[1] = new L2_FECollection(feorder, dim);
    }
};

struct CFOSLSFEFormulation_HdivL2L2Hyper : FOSLSFEFormulation
{
public:
    CFOSLSFEFormulation_HdivL2L2Hyper(CFOSLSFormulation_HdivL2L2Hyper& formulation, int fe_order)
        : FOSLSFEFormulation(formulation, fe_order)
    {
        int dim = formul.Dim();
        if (dim == 4)
            fecolls[0] = new RT0_4DFECollection;
        else
            fecolls[0] = new RT_FECollection(feorder, dim);

        fecolls[1] = new L2_FECollection(feorder, dim);
        fecolls[2] = new L2_FECollection(feorder, dim);
    }
};

struct CFOSLSFEFormulation_HdivH1Hyper : FOSLSFEFormulation
{
public:
    CFOSLSFEFormulation_HdivH1Hyper(CFOSLSFormulation_HdivH1Hyper& formulation, int fe_order)
        : FOSLSFEFormulation(formulation, fe_order)
    {
        int dim = formul.Dim();
        if (dim == 4)
            fecolls[0] = new RT0_4DFECollection;
        else
            fecolls[0] = new RT_FECollection(feorder, dim);

        if (dim == 4)
            fecolls[1] = new LinearFECollection;
        else
            fecolls[1] = new H1_FECollection(feorder + 1, dim);

        fecolls[2] = new L2_FECollection(feorder, dim);
    }
};

struct CFOSLSFEFormulation_HdivH1DivfreeHyper : FOSLSFEFormulation
{
public:
    CFOSLSFEFormulation_HdivH1DivfreeHyper(CFOSLSFormulation_HdivH1DivfreeHyp& formulation, int fe_order)
        : FOSLSFEFormulation(formulation, fe_order)
    {
        int dim = formul.Dim();

        if (dim == 4)
            fecolls[0] = new DivSkew1_4DFECollection;
        else
            fecolls[0] = new ND_FECollection(feorder + 1, dim);

        if (dim == 4)
            fecolls[1] = new LinearFECollection;
        else
            fecolls[1] = new H1_FECollection(feorder + 1, dim);

    }
};

struct CFOSLSFEFormulation_HdivDivfreeHyp : FOSLSFEFormulation
{
public:
    CFOSLSFEFormulation_HdivDivfreeHyp(CFOSLSFormulation_HdivDivfreeHyp& formulation, int fe_order)
        : FOSLSFEFormulation(formulation, fe_order)
    {
        int dim = formul.Dim();

        if (dim == 4)
            fecolls[0] = new DivSkew1_4DFECollection;
        else
            fecolls[0] = new ND_FECollection(feorder + 1, dim);
    }
};

struct CFOSLSFEFormulation_HdivH1Parab : FOSLSFEFormulation
{
public:
    CFOSLSFEFormulation_HdivH1Parab(CFOSLSFormulation_HdivH1Parab& formulation, int fe_order)
        : FOSLSFEFormulation(formulation, fe_order)
    {
        int dim = formul.Dim();
        if (dim == 4)
            fecolls[0] = new RT0_4DFECollection;
        else
            fecolls[0] = new RT_FECollection(feorder, dim);

        if (dim == 4)
            fecolls[1] = new LinearFECollection;
        else
            fecolls[1] = new H1_FECollection(feorder + 1, dim);

        fecolls[2] = new L2_FECollection(feorder, dim);
    }
};

struct CFOSLSFEFormulation_HdivH1Wave : FOSLSFEFormulation
{
public:
    CFOSLSFEFormulation_HdivH1Wave(CFOSLSFormulation_HdivH1Wave& formulation, int fe_order)
        : FOSLSFEFormulation(formulation, fe_order)
    {
        int dim = formul.Dim();
        if (dim == 4)
            fecolls[0] = new RT0_4DFECollection;
        else
            fecolls[0] = new RT_FECollection(feorder, dim);

        if (dim == 4)
            fecolls[1] = new LinearFECollection;
        else
            fecolls[1] = new H1_FECollection(feorder + 1, dim);

        fecolls[2] = new L2_FECollection(feorder, dim);
    }
};

struct CFOSLSFEFormulation_HdivH1L2_Laplace : FOSLSFEFormulation
{
public:
    CFOSLSFEFormulation_HdivH1L2_Laplace(CFOSLSFormulation_Laplace& formulation, int fe_order)
        : FOSLSFEFormulation(formulation, fe_order)
    {
        int dim = formul.Dim();
        if (dim == 4)
            fecolls[0] = new RT0_4DFECollection;
        else
            fecolls[0] = new RT_FECollection(feorder, dim);

        if (dim == 4)
            fecolls[1] = new LinearFECollection;
        else
            fecolls[1] = new H1_FECollection(feorder + 1, dim);

        fecolls[2] = new L2_FECollection(feorder, dim);
    }
};

struct CFOSLSFEFormulation_MixedLaplace : FOSLSFEFormulation
{
public:
    CFOSLSFEFormulation_MixedLaplace(CFOSLSFormulation_MixedLaplace& formulation, int fe_order)
        : FOSLSFEFormulation(formulation, fe_order)
    {
        int dim = formul.Dim();
        if (dim == 4)
            fecolls[0] = new RT0_4DFECollection;
        else
            fecolls[0] = new RT_FECollection(feorder, dim);

        fecolls[1] = new L2_FECollection(feorder, dim);
    }
};

struct FOSLSFEFormulation_Laplace : FOSLSFEFormulation
{
public:
    FOSLSFEFormulation_Laplace(FOSLSFormulation_Laplace& formulation, int fe_order)
        : FOSLSFEFormulation(formulation, fe_order)
    {
        int dim = formul.Dim();
        fecolls[0] = new H1_FECollection(feorder, dim);
    }
};


class BlockProblemForms
{
protected:
    const int numblocks;
    Array<ParBilinearForm*> diag_forms;
    Array2D<ParMixedBilinearForm*> offd_forms;
    bool initialized_forms;
public:
    BlockProblemForms(int num_blocks) : numblocks(num_blocks), initialized_forms(false)
    {
        diag_forms.SetSize(num_blocks);
        for (int i = 0; i < num_blocks; ++i)
            diag_forms[i] = NULL;
        offd_forms.SetSize(numblocks, num_blocks);
        for (int i = 0; i < num_blocks; ++i)
            for (int j = 0; j < num_blocks; ++j)
                offd_forms(i,j) = NULL;
    }

    void InitForms(FOSLSFEFormulation& fe_formul, Array<ParFiniteElementSpace *> &pfes);

    ParBilinearForm* & diag(int i)
    {
        MFEM_ASSERT(initialized_forms, "Calling diag() when forms were not initialized is forbidden");
        return diag_forms[i];
    }
    ParMixedBilinearForm* & offd(int i, int j)
    {
        MFEM_ASSERT(initialized_forms, "Calling offd() when forms were not initialized is forbidden");
        return offd_forms(i,j);
    }

    void Update();
};

// class for general CFOSLS problem
class FOSLSProblem
{
protected:
    ParMesh& pmesh;

    FOSLSFEFormulation& fe_formul;

    BdrConditions& bdr_conds;

    /// (optional)
    /// initialized only when the problem is related to a hierarchy
    GeneralHierarchy * hierarchy;

    /// (optional)
    /// index of this Problem inside the hierarchy.problems (if attached)
    /// or -1, otherwise
    /// changed only in GeneralHierarchy::AttachProblem()
    int attached_index;

    /// true if Update() is possible when the underlying pmesh is updated
    /// false, otherwise (if it was created as a "static" problem from the hierarchy)
    bool is_dynamic;

    Array<FOSLSEstimator*> estimators; // (optional)

    bool spaces_initialized;
    bool forms_initialized;
    bool system_assembled;
    bool solver_initialized;
    bool hierarchy_initialized;
    bool hpmats_initialized;

    // all par grid functions which are relevant to the formulation
    // e.g., solution components and right hand sides (2 * numblocks)
    // with that, righthand sides are essentially vector representations
    // of the linear forms in the rhs of the variational formulation
    Array<ParGridFunction*> grfuns;

    Array<ParFiniteElementSpace*> pfes;
    BlockProblemForms pbforms;
    Array<ParLinearForm*> plforms;

    Array<int> blkoffsets_true;
    Array<int> blkoffsets_func_true;
    Array<int> blkoffsets;
    Array2D<HypreParMatrix*> hpmats;
    BlockOperator *CFOSLSop;
    Array2D<HypreParMatrix*> hpmats_nobnd;
    BlockOperator *CFOSLSop_nobnd;

    BlockVector * trueRhs;
    BlockVector * trueX;
    BlockVector * trueBnd;
    BlockVector * x; // inital condition (~bnd conditions)
    int prec_option;
    Solver *prec;
    IterativeSolver * solver;

    mutable StopWatch chrono;

    bool verbose;

protected:
    void InitSpacesFromHierarchy(GeneralHierarchy& hierarchy, int level, const Array<SpaceName> &spaces_descriptor);
    /// the dynamically updated finest level f.e. spaces are used
    /// (unlike the version with explicit specification of the level
    void InitSpacesFromHierarchy(GeneralHierarchy& hierarchy, const Array<SpaceName> &spaces_descriptor);
    void InitSpaces(ParMesh& pmesh);
    void InitForms();
    void AssembleSystem(bool verbose);
    virtual void CreatePrec(BlockOperator & op, int prec_option, bool verbose) {}
    void UpdateSolverMat(Operator& op) { solver->SetOperator(op); }
    void SetPrecOption(int option) { prec_option = option; }

    void InitGrFuns();
    void DistributeSolution() const;

public:
    void InitSolver(bool verbose);

    void UpdateSolverPrec() { solver->SetPreconditioner(*prec); }

    void SetPrec(Solver & Prec)
    {
        MFEM_ASSERT(solver_initialized, "Cannot set a preconditioner before the solver is initialized \n");
        prec = &Prec;
        solver->SetPreconditioner(*prec);
    }

    void DistributeToGrfuns(const Vector& vec) const;

    BlockVector * GetInitialCondition();
    BlockVector * GetTrueInitialCondition();
    BlockVector * GetTrueInitialConditionFunc();
    BlockVector * GetExactSolProj();

    /// builds a Problem on a given mesh, creating f.e. spaces and all the necessary data inside
    /// (unlike constructors wghich take the hierarchy as an input argument)
    FOSLSProblem(ParMesh& pmesh_, BdrConditions& bdr_conditions, FOSLSFEFormulation& fe_formulation,
                 bool verbose_, bool assemble_system);

    /// builds a "static" problem on the level l of the hierarchy
    /// "static" means that the corresponding mesh and spaces are not updated
    /// when the hierarchy gets more levels
    FOSLSProblem(GeneralHierarchy& Hierarchy, int level, BdrConditions& bdr_conditions,
                 FOSLSFEFormulation& fe_formulation, bool verbose_, bool assemble_system);

    /// builds a "dynamic" problem on the finesh level of the hierarchy
    /// "dynamic" means that the corresponding mesh and spaces are updated
    /// when the hierarchy gets more levels
    FOSLSProblem(GeneralHierarchy& Hierarchy, BdrConditions& bdr_conditions,
                 FOSLSFEFormulation& fe_formulation, bool verbose_, bool assemble_system);

    // shorter constructor versions
    FOSLSProblem(ParMesh& pmesh_, BdrConditions& bdr_conditions, FOSLSFEFormulation& fe_formulation, bool verbose_)
        : FOSLSProblem(pmesh_, bdr_conditions, fe_formulation, verbose_, true) {}

    FOSLSProblem(GeneralHierarchy& Hierarchy, int level, BdrConditions& bdr_conditions,
                 FOSLSFEFormulation& fe_formulation, bool verbose_)
        : FOSLSProblem(Hierarchy, level, bdr_conditions, fe_formulation, verbose_, true) {}

    FOSLSProblem(GeneralHierarchy& Hierarchy, BdrConditions& bdr_conditions,
                 FOSLSFEFormulation& fe_formulation, bool verbose_)
        : FOSLSProblem(Hierarchy, bdr_conditions, fe_formulation, verbose_, true) {}

    void Solve(bool verbose, bool compute_error) const
    { SolveProblem(*trueRhs, verbose, compute_error); }
    void SolveProblem(const Vector& rhs, bool verbose, bool compute_error) const;
    void SolveProblem(const Vector& rhs, Vector& sol, bool verbose, bool compute_error) const;

    void BuildSystem(bool verbose);
    virtual void Update();

    ParMesh * GetParMesh(int level = 0)
    {
        if (level == 0)
            return &pmesh;
        else
        {
            MFEM_ASSERT(hierarchy, "Hierarchy doesn't exist (hence, also coarser meshes) !");
            return hierarchy->GetPmesh(level);
        }
    }

    FOSLSFEFormulation& GetFEformulation() { return fe_formul; }

    int GlobalTrueProblemSize() const {return CFOSLSop->Height();}

    MPI_Comm GetComm() {return pmesh.GetComm();}

    int GetNEstimators() const {return estimators.Size();}

    int AddEstimator(FOSLSEstimator& estimator)
    {
        estimators.Append(&estimator);
        return estimators.Size();
    }

    FOSLSEstimator* GetEstimator(int i)
    {
        MFEM_ASSERT(i >=0 && i < estimators.Size(), "Index for estimators out of bounds");
        return estimators[i];
    }

    /*
    virtual void CreateEstimator(int option)
    {
        // TODO: Implement a default error estimator here, FOSLS block + constraints
        MFEM_ABORT("CreateEstimator is not implemented in the base class FOSLSProblem");
    }
    */

    Array<ParGridFunction*> & GetGrFuns() {return grfuns;}

    ParGridFunction* GetGrFun(int i) {return grfuns[i];}

    ParFiniteElementSpace * GetPfes(int i) {return pfes[i];}

    Array<int>& GetTrueOffsets() { return blkoffsets_true;}

    Array<int>& GetTrueOffsetsFunc() { return blkoffsets_func_true;}

    BlockOperator* GetOp() { return CFOSLSop; }

    // doesn't own its blocks (taken from CFOSLSop)
    BlockOperator* GetFunctOp(const Array<int>& offsets);
    BlockOperator* GetFunctOp_nobnd(const Array<int>& offsets);

    BlockOperator* GetOp_nobnd() { return CFOSLSop_nobnd; }

    void ComputeAnalyticalRhs() const {ComputeAnalyticalRhs(*trueRhs);}
    void ComputeAnalyticalRhs(Vector& rhs) const;

    void ComputeRhsBlock(Vector& rhs, int blk) const;

    BlockVector& GetSol() {return *trueX;}

    BlockVector& GetRhs() {return *trueRhs;}

    BdrConditions& GetBdrConditions() {return bdr_conds;}

    //void ResetSolverOp(Operator& op) {solver->SetOperator(op);}

    void ResetOp(BlockOperator& op)
    {
        MFEM_ASSERT(op.Height() == blkoffsets_true[blkoffsets_true.Size() - 1]
                    && op.Width() == op.Height(), "Replacing operator sizes mismatch"
                                                  " the existing's");
        CFOSLSop = &op;
    }
    void ResetOp_nobnd(BlockOperator& op_nobnd)
    {
        MFEM_ASSERT(op_nobnd.Height() == blkoffsets_true[blkoffsets_true.Size() - 1]
                    && op_nobnd.Width() == op_nobnd.Height(), "Replacing operator sizes"
                                                              " mismatch the existing's");
        CFOSLSop_nobnd = &op_nobnd;
    }

    void ZeroBndValues(Vector& vec) const;

    void SetExactBndValues(Vector& vec) const;

    void ComputeError(const Vector& vec, bool verbose, bool checkbnd) const;
    void ComputeError(const Vector& vec, bool verbose, bool checkbnd, int blk) const;

    virtual void ComputeExtraError(const Vector& vec) const {}
    virtual void ComputeFuncError(const Vector& vec) const {}


    void ComputeBndError(const Vector& vec) const;
    void ComputeBndError(const Vector& vec, int blk) const;

    void ComputeError(bool verbose, bool checkbnd) const
    { ComputeError(*trueX, verbose, checkbnd);}

    void ComputeExtraError() const
    { ComputeExtraError(*trueX); }

    // constructs the BlockMatrix and the offsets, if given NULL
    virtual BlockMatrix* ConstructFunctBlkMat(const Array<int> &offsets);

    void CreateOffsetsRhsSol();

    void SetRelTol(double rtol) {solver->SetRelTol(rtol);}
    void SetAbsTol(double atol) {solver->SetAbsTol(atol);}

    void ResetPrec (int new_prec_option)
    {
        if (new_prec_option != prec_option)
        {
            if (prec)
                delete prec;
            CreatePrec(*CFOSLSop, new_prec_option, verbose);
            UpdateSolverPrec();
        }
        else
            if (!prec)
            {
                CreatePrec(*CFOSLSop, new_prec_option, verbose);
                UpdateSolverPrec();
            }
    }

    Solver * GetPrec() {return prec;}

    bool IsDynamic() const {return is_dynamic;}

    friend void GeneralHierarchy::AttachProblem(FOSLSProblem* problem);

    int GetAttachedIndex() const {return attached_index;}
};

/// FIXME: Looks like this shouldn't have happened
/// that I create a specific HdivL2L2 problem but take
/// a general FOSLSFEFormulation as an input.
/// However, if one changes the parameter type to a specific FOSLSFEFormulation (a child)
/// Then problems occur in the constructor of FOSLSProbHierarchy
/// which relies on the fact that all FOSLSProblem's children have exactly the same
/// signature for their constructors.
class FOSLSProblem_HdivL2hyp : virtual public FOSLSProblem
{
protected:
    virtual void CreatePrec(BlockOperator &op, int prec_option, bool verbose) override;
public:
    FOSLSProblem_HdivL2hyp(ParMesh& Pmesh, BdrConditions& bdr_conditions,
                    FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Pmesh, bdr_conditions, fe_formulation, verbose_)
    {
        SetPrecOption(precond_option);
        CreatePrec(*CFOSLSop, prec_option, verbose);
        UpdateSolverPrec();
    }

    FOSLSProblem_HdivL2hyp(GeneralHierarchy& Hierarchy, int level, BdrConditions& bdr_conditions,
                   FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Hierarchy, level, bdr_conditions, fe_formulation, verbose_)
    {
        SetPrecOption(precond_option);
        CreatePrec(*CFOSLSop, prec_option, verbose);
        UpdateSolverPrec();
    }

    FOSLSProblem_HdivL2hyp(GeneralHierarchy& Hierarchy, BdrConditions& bdr_conditions,
                   FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Hierarchy, bdr_conditions, fe_formulation, verbose_)
    {
        SetPrecOption(precond_option);
        CreatePrec(*CFOSLSop, prec_option, verbose);
        UpdateSolverPrec();
    }

    void ComputeExtraError(const Vector& vec) const override;

    void ComputeFuncError(const Vector& vec) const override;

    ParGridFunction * RecoverS() const
    { return RecoverS(trueX->GetBlock(0));}

    ParGridFunction * RecoverS(const Vector& sigma) const;
};

/// See the previous FIXME message
class FOSLSProblem_HdivL2L2hyp : virtual public FOSLSProblem
{
protected:
    virtual void CreatePrec(BlockOperator &op, int prec_option, bool verbose) override;
public:
    FOSLSProblem_HdivL2L2hyp(ParMesh& Pmesh, BdrConditions& bdr_conditions,
                    FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Pmesh, bdr_conditions, fe_formulation, verbose_)
    {
        SetPrecOption(precond_option);
        CreatePrec(*CFOSLSop, prec_option, verbose);
        UpdateSolverPrec();
    }

    FOSLSProblem_HdivL2L2hyp(GeneralHierarchy& Hierarchy, int level, BdrConditions& bdr_conditions,
                   FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Hierarchy, level, bdr_conditions, fe_formulation, verbose_)
    {
        SetPrecOption(precond_option);
        CreatePrec(*CFOSLSop, prec_option, verbose);
        UpdateSolverPrec();
    }

    FOSLSProblem_HdivL2L2hyp(GeneralHierarchy& Hierarchy, BdrConditions& bdr_conditions,
                   FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Hierarchy, bdr_conditions, fe_formulation, verbose_)
    {
        SetPrecOption(precond_option);
        CreatePrec(*CFOSLSop, prec_option, verbose);
        UpdateSolverPrec();
    }

    void ComputeExtraError(const Vector& vec) const override;

    void ComputeFuncError(const Vector& vec) const override;
};

/// See the previous FIXME message
class FOSLSProblem_HdivH1L2hyp : virtual public FOSLSProblem
{
protected:
    virtual void CreatePrec(BlockOperator &op, int prec_option, bool verbose) override;
public:
    FOSLSProblem_HdivH1L2hyp(ParMesh& Pmesh, BdrConditions& bdr_conditions,
                    FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Pmesh, bdr_conditions, fe_formulation, verbose_)
    {
        SetPrecOption(precond_option);
        CreatePrec(*CFOSLSop, prec_option, verbose);
        UpdateSolverPrec();
    }

    FOSLSProblem_HdivH1L2hyp(GeneralHierarchy& Hierarchy, int level, BdrConditions& bdr_conditions,
                   FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Hierarchy, level, bdr_conditions, fe_formulation, verbose_)
    {
        SetPrecOption(precond_option);
        CreatePrec(*CFOSLSop, prec_option, verbose);
        UpdateSolverPrec();
    }

    FOSLSProblem_HdivH1L2hyp(GeneralHierarchy& Hierarchy, BdrConditions& bdr_conditions,
                   FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Hierarchy, bdr_conditions, fe_formulation, verbose_)
    {
        SetPrecOption(precond_option);
        CreatePrec(*CFOSLSop, prec_option, verbose);
        UpdateSolverPrec();
    }
    void ComputeExtraError(const Vector& vec) const override;

    void ComputeFuncError(const Vector& vec) const override;
};

/// See the previous FIXME message
class FOSLSProblem_HdivH1parab : virtual public FOSLSProblem
{
protected:
    virtual void CreatePrec(BlockOperator &op, int prec_option, bool verbose) override;
public:
    FOSLSProblem_HdivH1parab(ParMesh& Pmesh, BdrConditions& bdr_conditions,
                    FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Pmesh, bdr_conditions, fe_formulation, verbose_)
    {
        SetPrecOption(precond_option);
        CreatePrec(*CFOSLSop, prec_option, verbose);
        UpdateSolverPrec();
    }

    FOSLSProblem_HdivH1parab(GeneralHierarchy& Hierarchy, int level, BdrConditions& bdr_conditions,
                   FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Hierarchy, level, bdr_conditions, fe_formulation, verbose_)
    {
        SetPrecOption(precond_option);
        CreatePrec(*CFOSLSop, prec_option, verbose);
        UpdateSolverPrec();
    }

    FOSLSProblem_HdivH1parab(GeneralHierarchy& Hierarchy, BdrConditions& bdr_conditions,
                   FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Hierarchy, bdr_conditions, fe_formulation, verbose_)
    {
        SetPrecOption(precond_option);
        CreatePrec(*CFOSLSop, prec_option, verbose);
        UpdateSolverPrec();
    }

    void ComputeExtraError(const Vector& vec) const override;

    void ComputeFuncError(const Vector& vec) const override;
};

/// See the previous FIXME message
class FOSLSProblem_HdivH1wave : virtual public FOSLSProblem
{
protected:
    virtual void CreatePrec(BlockOperator &op, int prec_option, bool verbose) override;
public:
    FOSLSProblem_HdivH1wave(ParMesh& Pmesh, BdrConditions& bdr_conditions,
                    FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Pmesh, bdr_conditions, fe_formulation, verbose_)
    {
        SetPrecOption(precond_option);
        CreatePrec(*CFOSLSop, prec_option, verbose);
        UpdateSolverPrec();
    }

    FOSLSProblem_HdivH1wave(GeneralHierarchy& Hierarchy, int level, BdrConditions& bdr_conditions,
                   FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Hierarchy, level, bdr_conditions, fe_formulation, verbose_)
    {
        SetPrecOption(precond_option);
        CreatePrec(*CFOSLSop, prec_option, verbose);
        UpdateSolverPrec();
    }

    FOSLSProblem_HdivH1wave(GeneralHierarchy& Hierarchy, BdrConditions& bdr_conditions,
                   FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Hierarchy, bdr_conditions, fe_formulation, verbose_)
    {
        SetPrecOption(precond_option);
        CreatePrec(*CFOSLSop, prec_option, verbose);
        UpdateSolverPrec();
    }

    void ComputeExtraError(const Vector& vec) const override;

    void ComputeFuncError(const Vector& vec) const override;
};

/// FIXME: See the previous FIXME messages
class FOSLSProblem_HdivH1lapl : virtual public FOSLSProblem
{
protected:
    virtual void CreatePrec(BlockOperator &op, int prec_option, bool verbose) override;
public:
    FOSLSProblem_HdivH1lapl(ParMesh& Pmesh, BdrConditions& bdr_conditions,
                    FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Pmesh, bdr_conditions, fe_formulation, verbose_)
    {
        SetPrecOption(precond_option);
        CreatePrec(*CFOSLSop, prec_option, verbose);
        UpdateSolverPrec();
    }

    FOSLSProblem_HdivH1lapl(GeneralHierarchy& Hierarchy, int level, BdrConditions& bdr_conditions,
                   FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Hierarchy, level, bdr_conditions, fe_formulation, verbose_)
    {
        SetPrecOption(precond_option);
        CreatePrec(*CFOSLSop, prec_option, verbose);
        UpdateSolverPrec();
    }

    FOSLSProblem_HdivH1lapl(GeneralHierarchy& Hierarchy, BdrConditions& bdr_conditions,
                   FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Hierarchy, bdr_conditions, fe_formulation, verbose_)
    {
        SetPrecOption(precond_option);
        CreatePrec(*CFOSLSop, prec_option, verbose);
        UpdateSolverPrec();
    }

    void ComputeExtraError(const Vector& vec) const override;

    void ComputeFuncError(const Vector& vec) const override;
};

/// FIXME: See the previous FIXME messages
class FOSLSProblem_MixedLaplace : virtual public FOSLSProblem
{
protected:
    virtual void CreatePrec(BlockOperator &op, int prec_option, bool verbose) override;
public:
    FOSLSProblem_MixedLaplace(ParMesh& Pmesh, BdrConditions& bdr_conditions,
                    FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Pmesh, bdr_conditions, fe_formulation, verbose_)
    {
        SetPrecOption(precond_option);
        CreatePrec(*CFOSLSop, prec_option, verbose);
        UpdateSolverPrec();
    }

    FOSLSProblem_MixedLaplace(GeneralHierarchy& Hierarchy, int level, BdrConditions& bdr_conditions,
                   FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Hierarchy, level, bdr_conditions, fe_formulation, verbose_)
    {
        SetPrecOption(precond_option);
        CreatePrec(*CFOSLSop, prec_option, verbose);
        UpdateSolverPrec();
    }

    FOSLSProblem_MixedLaplace(GeneralHierarchy& Hierarchy, BdrConditions& bdr_conditions,
                   FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Hierarchy, bdr_conditions, fe_formulation, verbose_)
    {
        SetPrecOption(precond_option);
        CreatePrec(*CFOSLSop, prec_option, verbose);
        UpdateSolverPrec();
    }

    void ComputeExtraError(const Vector& vec) const override;

    void ComputeFuncError(const Vector& vec) const override;
};

/// FIXME: See the previous FIXME messages
class FOSLSProblem_Laplace : virtual public FOSLSProblem
{
protected:
    virtual void CreatePrec(BlockOperator &op, int prec_option, bool verbose) override;
public:
    FOSLSProblem_Laplace(ParMesh& Pmesh, BdrConditions& bdr_conditions,
                    FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Pmesh, bdr_conditions, fe_formulation, verbose_)
    {
        SetPrecOption(precond_option);
        CreatePrec(*CFOSLSop, prec_option, verbose);
        UpdateSolverPrec();
    }

    FOSLSProblem_Laplace(GeneralHierarchy& Hierarchy, int level, BdrConditions& bdr_conditions,
                   FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Hierarchy, level, bdr_conditions, fe_formulation, verbose_)
    {
        SetPrecOption(precond_option);
        CreatePrec(*CFOSLSop, prec_option, verbose);
        UpdateSolverPrec();
    }

    FOSLSProblem_Laplace(GeneralHierarchy& Hierarchy, BdrConditions& bdr_conditions,
                   FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Hierarchy, bdr_conditions, fe_formulation, verbose_)
    {
        SetPrecOption(precond_option);
        CreatePrec(*CFOSLSop, prec_option, verbose);
        UpdateSolverPrec();
    }

    void ComputeExtraError(const Vector& vec) const override;

    void ComputeFuncError(const Vector& vec) const override;

    void ChangeSolver(double new_rtol, double new_atol);
};


// a regular FOSLS problem with additional routines for the divfree space
class FOSLSDivfreeProblem : virtual public FOSLSProblem
{
protected:
    FiniteElementCollection *hdiv_fecoll;
    ParFiniteElementSpace * hdiv_pfespace;
    HypreParMatrix * divfree_hpmat;
    HypreParMatrix * divfree_hpmat_nobnd;
public:
    FOSLSDivfreeProblem(ParMesh& Pmesh, BdrConditions& bdr_conditions,
                    FOSLSFEFormulation& fe_formulation, bool verbose_);

    FOSLSDivfreeProblem(ParMesh& Pmesh, BdrConditions& bdr_conditions,
                        FOSLSFEFormulation& fe_formulation, FiniteElementCollection& Hdiv_coll,
                        ParFiniteElementSpace& Hdiv_space, bool verbose_);

    FOSLSDivfreeProblem(GeneralHierarchy& Hierarchy, int level, BdrConditions& bdr_conditions,
                 FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose);

    FOSLSDivfreeProblem(GeneralHierarchy& Hierarchy, BdrConditions& bdr_conditions,
                 FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose);

    void ConstructDivfreeHpMats();

    const HypreParMatrix& GetDivfreeHpMat()  const {return *divfree_hpmat;}
    const HypreParMatrix& GetDivfreeHpMat_nobnd()  const {return *divfree_hpmat_nobnd;}

    //ParFiniteElementSpace * GetDivfreeFESpace() {return divfree_pfespace;}

    virtual void Update();

    void ChangeSolver();
    virtual void CreatePrec(BlockOperator & op, int prec_option, bool verbose) override;
};

/*
class FOSLSProblem_HcurlH1hyp : virtual public FOSLSProblem
{
protected:
public:
    FOSLSProblem_HcurlH1hyp(ParMesh& Pmesh, BdrConditions& bdr_conditions,
                    FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Pmesh, bdr_conditions, fe_formulation, verbose_)
    {
        SetPrecOption(precond_option);
        CreatePrec(*CFOSLSop, prec_option, verbose);
        UpdateSolverPrec();
    }

    FOSLSProblem_HcurlH1hyp(GeneralHierarchy& Hierarchy, int level, BdrConditions& bdr_conditions,
                   FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Hierarchy, level, bdr_conditions, fe_formulation, verbose_)
    {
        SetPrecOption(precond_option);
        CreatePrec(*CFOSLSop, prec_option, verbose);
        UpdateSolverPrec();
    }

};
*/


template <class Problem, class Hierarchy>
class FOSLSProblHierarchy
{
protected:
    FOSLSFEFormulation& fe_formulation;
    BdrConditions& bdr_conditions;
    int nlevels;
    Hierarchy& hierarchy;
    Array<Problem*> problems_lvls;
    Array<BlockOperator*> TrueP_lvls;
    Array<BlockOperator*> CoarsenedOps_lvls;
    Array<BlockOperator*> CoarsenedOps_nobnd_lvls;
    int prec_option;
    bool verbose;
public:
    FOSLSProblHierarchy(Hierarchy& hierarchy_, int nlevels_, BdrConditions& bdr_conditions_,
                          FOSLSFEFormulation& fe_formulation_, int precond_option, bool verbose_);

    virtual void Update(bool recoarsen);

    Problem* GetProblem(int l)
    {
        MFEM_ASSERT(l >=0 && l < nlevels, "Index in GetProblem() is out of bounds");
        return problems_lvls[l];
    }

    // from coarser to finer
    void Interpolate(int coarse_lvl, int fine_lvl, const Vector& vec_in, Vector& vec_out);

    // from finer to coarser
    void Restrict(int fine_lvl, int coarse_lvl, const Vector& vec_in, Vector& vec_out);

    int Nlevels() const {return hierarchy.Nlevels();}
    Hierarchy& GetHierarchy() const {return hierarchy;}

    BlockOperator * GetCoarsenedOp (int level) { return CoarsenedOps_lvls[level];}
    BlockOperator * GetCoarsenedOp_nobnd (int level) { return CoarsenedOps_nobnd_lvls[level];}
    BlockOperator * GetTrueP(int level) { return TrueP_lvls[level];}

    Array<int>* ConstructBndIndices(int level);

    void ConstructCoarsenedOps();
    void ConstructCoarsenedOps_nobnd();

    int GetUpdateCounter() const {return hierarchy.GetUpdateCounter();}

protected:
    HypreParMatrix& CoarsenFineBlockWithBND(int level, int i, int j, HypreParMatrix& input);
};

template <class Problem, class Hierarchy>
FOSLSProblHierarchy<Problem, Hierarchy>::FOSLSProblHierarchy(Hierarchy& hierarchy_,
                                                             int nlevels_, BdrConditions& bdr_conditions_,
                                                             FOSLSFEFormulation& fe_formulation_,
                                                             int precond_option, bool verbose_)
    : fe_formulation(fe_formulation_), bdr_conditions(bdr_conditions_),
      nlevels(nlevels_),
      hierarchy(hierarchy_), prec_option(precond_option),
      verbose(verbose_)
{
    problems_lvls.SetSize(nlevels);
    TrueP_lvls.SetSize(nlevels - 1);
    for (int l = 0; l < nlevels; ++l )
    {
        //std::cout << "I am here, verbose = " << verbose << "\n";
        problems_lvls[l] = new Problem(hierarchy, l, bdr_conditions, fe_formulation, prec_option, verbose);
        //std::cout << "I created a problem, l = " << l << "\n";
        if (l > 0)
        {
            Array<int>& blkoffsets_true_row = problems_lvls[l - 1]->GetTrueOffsets();
            Array<int>& blkoffsets_true_col = problems_lvls[l]->GetTrueOffsets();

            const Array<SpaceName>* space_names = fe_formulation.GetFormulation()->GetSpacesDescriptor();

            TrueP_lvls[l - 1] = new BlockOperator(blkoffsets_true_row, blkoffsets_true_col);

            int numblocks = fe_formulation.Nblocks(); // must be equal to the length of space_names

            for (int blk = 0; blk < numblocks; ++blk)
            {
                HypreParMatrix * TrueP_blk = hierarchy.GetTruePspace((*space_names)[blk], l - 1);
                TrueP_lvls[l - 1]->SetBlock(blk, blk, TrueP_blk);
            }
        }
    }

    CoarsenedOps_lvls.SetSize(nlevels);
    for (int l = 0; l < nlevels; ++l )
        CoarsenedOps_lvls[l] = NULL;

    CoarsenedOps_nobnd_lvls.SetSize(nlevels);
    for (int l = 0; l < nlevels; ++l )
        CoarsenedOps_nobnd_lvls[l] = NULL;

    ConstructCoarsenedOps();
    ConstructCoarsenedOps_nobnd();
}

template <class Problem, class Hierarchy>
void FOSLSProblHierarchy<Problem, Hierarchy>::Update(bool recoarsen)
{
    // update hierarchy
    hierarchy.Update();

    // create the new finest-level problem
    Problem * problem_new = new Problem(hierarchy, 0, bdr_conditions, fe_formulation, prec_option, verbose);
    problems_lvls.Prepend(problem_new);

    // create new interpolation block operator
    Array<int>& blkoffsets_true_row = problems_lvls[0]->GetTrueOffsets();
    Array<int>& blkoffsets_true_col = problems_lvls[1]->GetTrueOffsets();
    const Array<SpaceName>* space_names = fe_formulation.GetFormulation()->GetSpacesDescriptor();

    BlockOperator * TrueP_new = new BlockOperator(blkoffsets_true_row, blkoffsets_true_col);

    int numblocks = fe_formulation.Nblocks();
    for (int blk = 0; blk < numblocks; ++blk)
    {
        HypreParMatrix * TrueP_blk = hierarchy.GetTruePspace((*space_names)[blk], 0);
        TrueP_new->SetBlock(blk, blk, TrueP_blk);
    }

    TrueP_lvls.Prepend(TrueP_new);

    // update number of levels
    nlevels = hierarchy.Nlevels();

    // delete the old coarsened ops
    // if l == 0, these are the operators of the previous finest problem
    // so we don't delete them
    for (int l = 1; l < CoarsenedOps_lvls.Size(); ++l )
    {
        delete CoarsenedOps_lvls[l];
        delete CoarsenedOps_nobnd_lvls[l];
    }

    CoarsenedOps_lvls.SetSize(nlevels);
    for (int l = 1; l < nlevels; ++l)
        CoarsenedOps_lvls[l] = NULL;

    CoarsenedOps_nobnd_lvls.SetSize(nlevels);
    for (int l = 1; l < nlevels; ++l)
        CoarsenedOps_nobnd_lvls[l] = NULL;

    // reconstruct coarsened operators if required
    if (recoarsen)
    {
        ConstructCoarsenedOps();
        ConstructCoarsenedOps_nobnd();
    }

}


template <class Problem, class Hierarchy>
void FOSLSProblHierarchy<Problem, Hierarchy>::Interpolate(int coarse_lvl, int fine_lvl, const Vector& vec_in, Vector& vec_out)
{
    MFEM_ASSERT(coarse_lvl == fine_lvl + 1, "Interpolate works only between the neighboring levels");
    Array<int>& blkoffsets_true_fine = problems_lvls[fine_lvl]->GetTrueOffsets();
    Array<int>& blkoffsets_true_coarse = problems_lvls[coarse_lvl]->GetTrueOffsets();

    BlockVector viewer_in(vec_in.GetData(),  blkoffsets_true_coarse);
    BlockVector viewer_out(vec_out.GetData(),  blkoffsets_true_fine);
    TrueP_lvls[fine_lvl]->Mult(viewer_in, viewer_out);
}

template <class Problem, class Hierarchy>
Array<int>* FOSLSProblHierarchy<Problem, Hierarchy>::ConstructBndIndices(int level)
{
    int numblocks = problems_lvls[level]->GetFEformulation().Nblocks();
    BdrConditions& bdr_conds = problems_lvls[level]->GetBdrConditions();

    int nbnd_indices = 0;
    for (int blk= 0; blk < numblocks; ++blk)
    {
        const Array<int> &essbdr_attrs = bdr_conds.GetBdrAttribs(blk);

        Array<int> ess_bnd_tdofs;
        problems_lvls[level]->GetPfes(blk)->GetEssentialTrueDofs(essbdr_attrs, ess_bnd_tdofs);

        nbnd_indices += ess_bnd_tdofs.Size();
    }

    Array<int> * res = new Array<int>(nbnd_indices);

    int blk_shift = 0;
    int count = 0;
    for (int blk = 0; blk < numblocks; ++blk)
    {
        const Array<int> &essbdr_attrs = bdr_conds.GetBdrAttribs(blk);

        //essbdr_attrs.Print();

        Array<int> ess_bnd_tdofs;
        problems_lvls[level]->GetPfes(blk)->GetEssentialTrueDofs(essbdr_attrs, ess_bnd_tdofs);

        //ess_bnd_tdofs.Print();

        for (int j = 0; j < ess_bnd_tdofs.Size(); ++j)
        {
            (*res)[count] = ess_bnd_tdofs[j] + blk_shift;
            ++count;
        }

        blk_shift += problems_lvls[level]->GetTrueOffsets()[blk + 1];
    }

    //res->Print();


    MFEM_ASSERT(count == nbnd_indices, "An error in counting bnd indices occured!");


    return res;
}


template <class Problem, class Hierarchy>
void FOSLSProblHierarchy<Problem, Hierarchy>::Restrict(int fine_lvl, int coarse_lvl, const Vector& vec_in, Vector& vec_out)
{
    MFEM_ASSERT(coarse_lvl == fine_lvl + 1, "Interpolate works only between the neighboring levels");
    Array<int>& blkoffsets_true_fine = problems_lvls[fine_lvl]->GetTrueOffsets();
    Array<int>& blkoffsets_true_coarse = problems_lvls[coarse_lvl]->GetTrueOffsets();

    BlockVector viewer_in(vec_in.GetData(), blkoffsets_true_fine);
    BlockVector viewer_out(vec_out.GetData(), blkoffsets_true_coarse);
    TrueP_lvls[fine_lvl]->MultTranspose(viewer_in, viewer_out);

    // FIXME: Do we need to clear the boundary conditions on the coarse level after that?
    // I guess, no.
}


// coarsens and restores boundary conditions
// level l is the level where interpolation matrix should be taken
// e.g., for coarsening from 0th level to the 1st level,
// one should use interpolation matrix from level 0
template <class Problem, class Hierarchy>
HypreParMatrix& FOSLSProblHierarchy<Problem, Hierarchy>::CoarsenFineBlockWithBND
(int l, int i, int j, HypreParMatrix& input)
{
    HypreParMatrix * res;

    HypreParMatrix * TrueP_i = &((HypreParMatrix&)(TrueP_lvls[l]->GetBlock(i,i)));

    const Array<int> &essbdr_attrs = bdr_conditions.GetBdrAttribs(i);
    Array<int> temp_i;
    problems_lvls[l + 1]->GetPfes(i)->GetEssentialTrueDofs(essbdr_attrs, temp_i);

    if (i == j) // we can use RAP for diagonal blocks
    {
        res = RAP(TrueP_i, &input, TrueP_i);
        res->CopyRowStarts();
        res->CopyRowStarts();

        Eliminate_ib_block(*res, temp_i, temp_i );
        HypreParMatrix * temphpmat = res->Transpose();
        Eliminate_ib_block(*temphpmat, temp_i, temp_i );
        res = temphpmat->Transpose();
        Eliminate_bb_block(*res, temp_i);
        SparseMatrix diag;
        res->GetDiag(diag);
        diag.MoveDiagonalFirst();

        res->CopyRowStarts();
        res->CopyColStarts();

        delete temphpmat;
    }
    else
    {
        HypreParMatrix * TrueP_i_T = TrueP_i->Transpose();
        HypreParMatrix * TrueP_j = &((HypreParMatrix&)(TrueP_lvls[l]->GetBlock(j,j)));

        const Array<int> &essbdr_attrs = bdr_conditions.GetBdrAttribs(j);

        Array<int> temp_j;
        problems_lvls[l + 1]->GetPfes(j)->GetEssentialTrueDofs(essbdr_attrs, temp_j);
        //const Array<int> *temp_j = EssBdrTrueDofs_Funct_lvls[l][0];

        HypreParMatrix * temp_prod = ParMult(&input, TrueP_j);
        res = ParMult(TrueP_i_T, temp_prod);

        Eliminate_ib_block(*res, temp_j, temp_i );
        HypreParMatrix * temphpmat = res->Transpose();
        Eliminate_ib_block(*temphpmat, temp_i, temp_j );
        res = temphpmat->Transpose();
        res->CopyRowStarts();
        res->CopyColStarts();
        delete temphpmat;

        delete TrueP_i_T;
        delete temp_prod;
    }
    return *res;
}

template <class Problem, class Hierarchy>
void FOSLSProblHierarchy<Problem, Hierarchy>::ConstructCoarsenedOps()
{
    CoarsenedOps_lvls[0] = problems_lvls[0]->GetOp();

    int numblocks = problems_lvls[0]->GetFEformulation().Nblocks();
    for (int l = 1; l < nlevels; ++l )
    {
        Array2D<HypreParMatrix*> coarseop_lvl(numblocks, numblocks);
        for (int i = 0; i < numblocks; ++i)
            for (int j = i; j < numblocks; ++j)
            {
                coarseop_lvl(i,j) = NULL;

                if (!CoarsenedOps_lvls[l - 1]->IsZeroBlock(i,j))
                {
                    HypreParMatrix& Fine_blk_ij = (HypreParMatrix&)(CoarsenedOps_lvls[l - 1]->GetBlock(i,j));

                    if (i == j)
                        coarseop_lvl(i,j) = &CoarsenFineBlockWithBND(l - 1, i, j, Fine_blk_ij );
                    else
                    {
                        coarseop_lvl(i,j) = &CoarsenFineBlockWithBND(l - 1, i, j, Fine_blk_ij );

                        coarseop_lvl(j,i) = coarseop_lvl(i,j)->Transpose();
                        coarseop_lvl(j,i)->CopyRowStarts();
                        coarseop_lvl(j,i)->CopyColStarts();
                    }
                }

            } // end of an iteration for fixed (i,j)
        CoarsenedOps_lvls[l] = new BlockOperator(problems_lvls[l]->GetTrueOffsets());

        for (int i = 0; i < numblocks; ++i)
            for (int j = i; j < numblocks; ++j)
                if (coarseop_lvl(i,j))
                    CoarsenedOps_lvls[l]->SetBlock(i,j, coarseop_lvl(i,j));

    } // end of loop over levels
}

template <class Problem, class Hierarchy>
void FOSLSProblHierarchy<Problem, Hierarchy>::ConstructCoarsenedOps_nobnd()
{
    CoarsenedOps_nobnd_lvls[0] = problems_lvls[0]->GetOp_nobnd();

    int numblocks = problems_lvls[0]->GetFEformulation().Nblocks();
    for (int l = 1; l < nlevels; ++l )
    {
        Array2D<HypreParMatrix*> coarseop_lvl(numblocks, numblocks);
        for (int i = 0; i < numblocks; ++i)
        {
            HypreParMatrix * TrueP_i = &((HypreParMatrix&)(TrueP_lvls[l - 1]->GetBlock(i,i)));

            for (int j = i; j < numblocks; ++j)
            {
                coarseop_lvl(i,j) = NULL;

                if (!CoarsenedOps_nobnd_lvls[l - 1]->IsZeroBlock(i,j))
                {
                    HypreParMatrix& Fine_blk_ij = (HypreParMatrix&)(CoarsenedOps_nobnd_lvls[l - 1]->GetBlock(i,j));

                    if (i == j)
                    {
                        coarseop_lvl(i,j) = RAP(TrueP_i, &Fine_blk_ij, TrueP_i);
                        coarseop_lvl(i,j)->CopyRowStarts();
                        coarseop_lvl(i,j)->CopyRowStarts();
                    }
                    else
                    {
                        HypreParMatrix * TrueP_j = &((HypreParMatrix&)(TrueP_lvls[l - 1]->GetBlock(j,j)));

                        coarseop_lvl(i,j) = RAP(TrueP_i, &Fine_blk_ij, TrueP_j);
                        coarseop_lvl(i,j)->CopyRowStarts();
                        coarseop_lvl(i,j)->CopyRowStarts();

                        coarseop_lvl(j,i) = coarseop_lvl(i,j)->Transpose();
                        coarseop_lvl(j,i)->CopyRowStarts();
                        coarseop_lvl(j,i)->CopyColStarts();
                    }
                }
            }
        } // end of an iteration for fixed (i,j)
        CoarsenedOps_nobnd_lvls[l] = new BlockOperator(problems_lvls[l]->GetTrueOffsets());

        for (int i = 0; i < numblocks; ++i)
            for (int j = i; j < numblocks; ++j)
                if (coarseop_lvl(i,j))
                    CoarsenedOps_nobnd_lvls[l]->SetBlock(i,j, coarseop_lvl(i,j));

    } // end of loop over levels
}

template <class Problem, class Hierarchy> class FOSLSCylProblHierarchy : public FOSLSProblHierarchy<Problem, Hierarchy>
{
    // additional routines and data members related to the cylinder structure go here
public:
    FOSLSCylProblHierarchy(Hierarchy& hierarchy_, int nlevels_, BdrConditions& bdr_conditions_,
                          FOSLSFEFormulation& fe_formulation_, int precond_option, bool verbose_);

public:
    void InterpolateAtBase(const char * top_or_bot, int lvl, const Vector& vec_in, Vector& vec_out);
    void RestrictAtBase(const char * top_or_bot, int lvl, const Vector& vec_in, Vector& vec_out)
    { MFEM_ABORT("RestrictAtBase has not been implemented properly \n"); }

    // probably, there is no need of this
    Vector *GetExactBase(const char * top_or_bot, int level)
    { return FOSLSProblHierarchy<Problem, Hierarchy>::problems_lvls[level]->GetExactBase(top_or_bot); }
};

template <class Problem, class Hierarchy>
FOSLSCylProblHierarchy<Problem,Hierarchy>::FOSLSCylProblHierarchy
                (Hierarchy& hierarchy_, int nlevels_, BdrConditions& bdr_conditions_,
                      FOSLSFEFormulation& fe_formulation_, int precond_option, bool verbose_)
    : FOSLSProblHierarchy<Problem,Hierarchy>(hierarchy_, nlevels_, bdr_conditions_, fe_formulation_, precond_option, verbose_)
{

}

template <class Problem, class Hierarchy>
void FOSLSCylProblHierarchy<Problem,Hierarchy>::InterpolateAtBase(const char * top_or_bot,
                                                          int lvl, const Vector& vec_in, Vector& vec_out)
{
    Problem * problem_lvl = FOSLSProblHierarchy<Problem, Hierarchy>::problems_lvls[lvl];

    FOSLSFEFormulation * fe_formul = problem_lvl->GetFEformulation();

    // index of the unknown with boundary condition
    int index = fe_formul->GetFormulation()->GetUnknownWithInitCnd();

    SpaceName space_name = fe_formul->GetFormulation()->GetSpaceName(index);

    FOSLSProblHierarchy<Problem, Hierarchy>::hierarchy.GetTrueP_bnd(top_or_bot, lvl, space_name)->Mult(vec_in, vec_out);
}

class GeneralMultigrid : public Solver
{
protected:
    int nlevels;
    const Array<Operator*> &P_lvls;
    const Array<Operator*> &Op_lvls;
    const Operator& CoarseOp;
    const Array<Operator*> &PreSmoothers_lvls;
    const Array<Operator*> &PostSmoothers_lvls;

    bool symmetric;

    mutable Array<Vector*> correction;
    mutable Array<Vector*> residual;

    mutable Vector res_aux;
    mutable Vector cor_cor;
    mutable Vector cor_aux;

    mutable int current_level;

public:
    GeneralMultigrid(int Nlevels, const Array<Operator*> &P_lvls_, const Array<Operator*> &Op_lvls_,
                     const Operator& CoarseOp_, const Array<Operator*> &Smoothers_lvls_)
        : GeneralMultigrid(Nlevels, P_lvls_, Op_lvls_, CoarseOp_, Smoothers_lvls_, Smoothers_lvls_)
    { symmetric = true; }

    GeneralMultigrid(int Nlevels, const Array<Operator*> &P_lvls_,
                     const Array<Operator*> &Op_lvls_, const Operator& CoarseOp_,
                     const Array<Operator*> &PreSmoothers_lvls_, const Array<Operator*> &PostSmoothers_lvls_);

    void MG_Cycle() const;

    virtual void Mult(const Vector & x, Vector & y) const override;

    virtual void SetOperator(const Operator &op) override
    { MFEM_ABORT("SetOperator() not implemented in the GeneralMultigrid class"); }

};

/// The operator x -> R*A*P*x.
class RAPBlockHypreOperator : public BlockOperator
{
protected:
   int nblocks;
   const Array<int>& offsets;
public:
   /// Construct the RAP operator given R^T, A and P as a block operators
   /// with each block being a HypreParMatrix
   RAPBlockHypreOperator(BlockOperator &Rt_, BlockOperator &A_, BlockOperator &P_, const Array<int>& Offsets);
};

class OperatorProduct : public Operator
{
protected:
    const Operator& op_first;
    const Operator& op_second;
    // additional memory for storing intermediate results
    mutable Vector * tmp;
    mutable Vector * tmp_tr;
public:
    OperatorProduct(const Operator & op1, const Operator& op2) : op_first(op1), op_second(op2)
    { tmp = new Vector(op1.Height()); tmp_tr = new Vector(op2.Height());}

    ~OperatorProduct() {delete tmp; delete tmp_tr;}

    void Mult(const Vector & x, Vector & y) const override
    { op_first.Mult(x, *tmp); op_second.Mult(*tmp, y); }

    void MultTranspose(const Vector & x, Vector & y) const override
    { op_second.Mult(x, *tmp_tr); op_first.Mult(*tmp_tr, y); }

};

// SmootherSum   * x = (Smoo1 + Smoo2 - Smoo2 * A * Smoo1) * x
// SmootherSum^T * x = (Smoo2^T + Smoo1^T - Smoo1^T * A * Smoo2^T) * x
class SmootherSum : public Operator
{
protected:
    const Operator& smoo_fst;
    const Operator& smoo_snd;
    const Operator& op;
    // additional memory for storing intermediate results
    mutable Vector * tmp1;
    mutable Vector * tmp2;

public:
    SmootherSum(const Operator & smoo1, const Operator& smoo2, const Operator& Aop) : smoo_fst(smoo1), smoo_snd(smoo2), op(Aop)
    {
        tmp1 = new Vector(smoo_fst.Height()); tmp2 = new Vector(smoo_snd.Height());
        //MPI_Barrier(MPI_COMM_WORLD);
        //std::cout << "sizes are: \n";
        //std::cout << "smoo1: " << smoo_fst.Height() << " x " << smoo_fst.Width() << "\n";
        //std::cout << "smoo2: " << smoo_snd.Height() << " x " << smoo_snd.Width() << "\n";
        //std::cout << "op: " << op.Height() << " x " << op.Width() << "\n";
        //std::cout << std::flush;
        //MPI_Barrier(MPI_COMM_WORLD);
    }

    ~SmootherSum() {delete tmp1; delete tmp2;}

    void Mult(const Vector & x, Vector & y) const override
    {
        //std::cout << "input to SmootherSum, x, norm = " << x.Norml2() / sqrt(x.Size()) << "\n";
        smoo_snd.Mult(x, y);

        //std::cout << "Smoo2 * x, norm = " << y.Norml2() / sqrt(y.Size()) << "\n";

        //Vector temp(y.Size());
        //temp = y;

        //MPI_Barrier(MPI_COMM_WORLD);
        //std::cout << std::flush;
        //MPI_Barrier(MPI_COMM_WORLD);

        smoo_fst.Mult(x, *tmp1);

        //std::cout << "Smoo1 * x, norm = " << tmp1->Norml2() / sqrt(tmp1->Size()) << "\n";
        //MPI_Barrier(MPI_COMM_WORLD);
        //std::cout << std::flush;
        //MPI_Barrier(MPI_COMM_WORLD);

        //MPI_Barrier(MPI_COMM_WORLD);
        //std::cout << "I am here \n";
        //std::cout << "y size = " << y.Size() << "\n";
        //std::cout << "tmp1 size = " << tmp1->Size() << "\n";
        //std::cout << "tmp1 norm = " << tmp1->Norml2() / sqrt (tmp1->Size()) << "\n";
        //std::cout << std::flush;
        //MPI_Barrier(MPI_COMM_WORLD);

        y += *tmp1;
        //y.Add(1.0, *tmp1);

        //std::cout << "Smoo1 * x + Smoo2 * x, norm = " << y.Norml2() / sqrt(y.Size()) << "\n";

        op.Mult(*tmp1, *tmp2);
        smoo_snd.Mult(*tmp2, *tmp1);

        //std::cout << "Smoo2 * A * Smoo1 * x, norm = " << tmp1->Norml2() / sqrt(tmp1->Size()) << "\n";

        //temp -= *tmp1;
        //std::cout << "Smoo2 * x - Smoo2 * A * Smoo1 * x, norm = " << temp.Norml2() / sqrt(temp.Size()) << "\n";

        y -= *tmp1;

        //std::cout << "output to SmootherSum, y, norm = " << y.Norml2() / sqrt(y.Size()) << "\n";

        //MPI_Barrier(MPI_COMM_WORLD);
        //std::cout << std::flush;
        //MPI_Barrier(MPI_COMM_WORLD);
    }

    void MultTranspose(const Vector & x, Vector & y) const override
    {
        //std::cout << "input to SmootherSum, x, norm = " << x.Norml2() / sqrt(x.Size()) << "\n";

        smoo_fst.MultTranspose(x, y);

        //Vector temp(y.Size());
        //temp = y;

        //std::cout << "Smoo1 * x, norm = " << y.Norml2() / sqrt(y.Size()) << "\n";

        smoo_snd.MultTranspose(x, *tmp1);
        y += *tmp1;

        //std::cout << "Smoo2 * x, norm = " << tmp1->Norml2() / sqrt(tmp1->Size()) << "\n";

        op.Mult(*tmp1, *tmp2);
        smoo_fst.MultTranspose(*tmp2, *tmp1);

        //temp -= *tmp1;
        //std::cout << "Smoo1 * x - Smoo1 * A * Smoo2 * x, norm = " << temp.Norml2() / sqrt(temp.Size()) << "\n";

        y -= *tmp1;

        //std::cout << "output to SmootherSum, y, norm = " << y.Norml2() / sqrt(y.Size()) << "\n";
    }

};

class BlkInterpolationWithBNDforTranspose : public BlockOperator
{
protected:
    int nblocks;
    BlockOperator& P;
    const Array<int>& row_offsets;
    const Array<int>& col_offsets;
    Array<int> * bnd_indices;
public:
    BlkInterpolationWithBNDforTranspose(BlockOperator& P_, Array<int>& BndIndices_,
                                        const Array<int>& Row_offsets, const Array<int>& Col_offsets)
        : BlockOperator(Row_offsets, Col_offsets),
          nblocks(P_.NumRowBlocks()), P(P_),
          row_offsets(Row_offsets), col_offsets(Col_offsets),
          bnd_indices(&BndIndices_)
    {
        for (int i = 0; i < nblocks; ++i)
            SetBlock(i,i, &(P.GetBlock(i,i)));
    }

    BlkInterpolationWithBNDforTranspose(BlockOperator& P_, Array<int>* BndIndices_,
                                        const Array<int>& Row_offsets, const Array<int>& Col_offsets)
        : BlockOperator(Row_offsets, Col_offsets),
          nblocks(P_.NumRowBlocks()), P(P_),
          row_offsets(Row_offsets), col_offsets(Col_offsets)
    {
        for (int i = 0; i < nblocks; ++i)
            SetBlock(i,i, &(P.GetBlock(i,i)));

        // FIXME: Can't we just copy the pointer instead? Not sure what is the best.
        MFEM_ASSERT(BndIndices_, "Bnd indices must not be NULL as an input argument");
        bnd_indices = new Array<int>(BndIndices_->Size());
        for (int i = 0; i < bnd_indices->Size(); ++i)
            (*bnd_indices)[i] = (*BndIndices_)[i];
    }

    void Mult(const Vector &x, Vector &y) const override {P.Mult(x,y);}

    void MultTranspose(const Vector &x, Vector &y) const override;
};


class InterpolationWithBNDforTranspose : public Operator
{
protected:
    Operator& P;
    Array<int> * bnd_indices;
public:
    InterpolationWithBNDforTranspose(Operator& P_, Array<int>& BndIndices_)
        : Operator(P_.Height(), P_.Width()), P(P_), bnd_indices(&BndIndices_) {}

    InterpolationWithBNDforTranspose(Operator& P_, Array<int>* BndIndices_)
        : Operator(P_.Height(), P_.Width()), P(P_)
    {
        MFEM_ASSERT(BndIndices_, "Bnd indices must not be NULL as an input argument");
        bnd_indices = new Array<int>(BndIndices_->Size());
        for (int i = 0; i < bnd_indices->Size(); ++i)
            (*bnd_indices)[i] = (*BndIndices_)[i];
    }

    void Mult(const Vector &x, Vector &y) const override {P.Mult(x,y);}

    void MultTranspose(const Vector &x, Vector &y) const override;
};

struct ComponentsDescriptor
{
    bool with_Schwarz;
    bool optimized_Schwarz;
    bool with_Hcurl;
    bool with_coarsest_partfinder;
    bool with_coarsest_hcurl;
    bool with_monolithic_GS;
public:
    ComponentsDescriptor() : ComponentsDescriptor(false, false, false, false, false, false) {}

    ComponentsDescriptor(bool with_Schwarz_, bool optimized_Schwarz_, bool with_Hcurl_,
                         bool with_coarsest_partfinder_, bool with_coarsest_hcurl_,
                         bool with_monolithic_GS_)
        : with_Schwarz(with_Schwarz_), optimized_Schwarz(optimized_Schwarz_),
          with_Hcurl(with_Hcurl_),
          with_coarsest_partfinder(with_coarsest_partfinder_),
          with_coarsest_hcurl(with_coarsest_hcurl_),
          with_monolithic_GS(with_monolithic_GS_)
    {}
};

class MultigridToolsHierarchy
{
protected:
    GeneralHierarchy& hierarchy;
    int nlevels;
    FOSLSProblem* problem;
    ComponentsDescriptor descr;
protected:
    Array<SparseMatrix*> AE_e_lvls;
    Array<BlockOperator*> BlockP_nobnd_lvls;
    Array<Operator*> P_bnd_lvls;
    Array<BlockOperator*> FunctOps_lvls;
    Array<Operator*> Ops_lvls;
    Array<BlockOperator*> FunctOps_nobnd_lvls;
    Array<LocalProblemSolver*> SchwarzSmoothers_lvls;
    Array<HcurlGSSSmoother*> HcurlSmoothers_lvls;
    Array<Operator*> CombinedSmoothers_lvls;
    Array<MonolithicGSBlockSmoother*> MonolithicGSSmoothers_lvls;
    CoarsestProblemSolver* CoarsestSolver_partfinder;
    CoarsestProblemHcurlSolver* CoarsestSolver_hcurl;

    std::deque<Array<int>* > coarsebnd_indces_funct_lvls;
    std::deque<const Array<int>* > offsets_funct;
    std::deque<const Array<int>* > offsets_sp_funct;
    Array<SparseMatrix*> Mass_mat_lvls;
    Array<BlockMatrix*> Funct_mat_lvls;
    Array<SparseMatrix*> Constraint_mat_lvls;

    std::deque<Array<int>* > el2dofs_row_offsets;
    std::deque<Array<int>* > el2dofs_col_offsets;
    /*
    Array<int> row_offsets_coarse, col_offsets_coarse;
    std::vector<Array<int>* > essbdr_tdofs_funct_coarse;
    std::vector<Array<int>* > essbdr_dofs_funct_coarse;
    std::vector<Array<int>* > fullbdr_attribs;
    */

protected:
    int update_counter;

public:
    MultigridToolsHierarchy(GeneralHierarchy& hierarchy_, int problem_index,
                            ComponentsDescriptor& descriptor_)
        : MultigridToolsHierarchy(hierarchy_, *hierarchy_.GetProblem(problem_index), descriptor_) {}

    void Update(bool recoarsen);

protected:
    // troubles with such a constructor is the implementation of Update(), in particular,
    // for the problem when it is not attached to the hierarchy's finest level
    MultigridToolsHierarchy(GeneralHierarchy& hierarchy_, FOSLSProblem& problem_,
                            ComponentsDescriptor& descriptor_);
public:
    /// Getters
    Array<Operator*>& GetCombinedSmoothers() {return CombinedSmoothers_lvls;}
    Array<BlockOperator*>& GetBlockOps() {return FunctOps_lvls;}
    Array<Operator*>& GetOps() {return Ops_lvls;}
    Array<BlockOperator*>& GetBlockOps_nobnd() {return FunctOps_nobnd_lvls;}
    Array<Operator*>& GetPs_bnd() {return P_bnd_lvls;}
    Array<BlockOperator*>& GetBlockPs_nobnd() {return BlockP_nobnd_lvls;}
    CoarsestProblemSolver* GetCoarsestSolver_Partfinder() {return CoarsestSolver_partfinder;}
    CoarsestProblemHcurlSolver* GetCoarsestSolver_Hcurl() {return CoarsestSolver_hcurl;}
    FOSLSProblem* GetProblem() {return problem;}
    GeneralHierarchy* GetHierarchy() {return &hierarchy;}
    Array<LocalProblemSolver*>& GetSchwarzSmoothers() {return SchwarzSmoothers_lvls;}
    Array<HcurlGSSSmoother*>& GetHcurlSmoothers() {return HcurlSmoothers_lvls;}
    Array<SparseMatrix*>& GetMassSpmats() {return Mass_mat_lvls;}
    Array<BlockMatrix*>& GetFunctBlockMats() {return Funct_mat_lvls;}
    Array<SparseMatrix*>& GetConstraintSpmats() {return Constraint_mat_lvls;}
    std::deque<const Array<int>* > & GetOffsetsFunct() {return offsets_funct;}
    std::deque<const Array<int>* > & GetSpOffsetsFunct() {return offsets_sp_funct;}
    std::deque<Array<int>* >& GetOffsets_El2dofs_row() {return el2dofs_row_offsets;}
    std::deque<Array<int>* >& GetOffsets_El2dofs_col() {return el2dofs_col_offsets;}
    Array<MonolithicGSBlockSmoother*> & GetMonolitGSSmoothers() {return MonolithicGSSmoothers_lvls;}

    /// Getters for component description options
    bool With_Hcurl() {return descr.with_Hcurl;}
    bool With_Coarsest_partfinder() {return descr.with_coarsest_partfinder;}
    bool With_Schwarz() {return descr.with_Schwarz;}
    bool With_Coarsest_hcurl() {return descr.with_coarsest_hcurl;}
};

HypreParMatrix * CopyHypreParMatrix(const HypreParMatrix& hpmat);

void EliminateBoundaryBlocks(BlockOperator& BlockOp, const std::vector<Array<int>* > esstdofs_blks);

SparseMatrix *ElementToDofs(const FiniteElementSpace &fes);

BlockMatrix *RAP(const BlockMatrix &Rt, const BlockMatrix &A, const BlockMatrix &P);

// time moments: t0 + i * deltat, i = 0, ... Nmoments - 1
void ComputeSlices(const Mesh& mesh, double t0, int Nmoments, double deltat, int myid);
void Compute_elpartition (const Mesh& mesh, double t0, int Nmoments, double deltat, std::vector<std::vector<int> > & elpartition);
void computeSliceCell (const Mesh& mesh, int elind, std::vector<std::vector<double> > & pvec,
                       std::vector<std::vector<double> > & ipoints, std::vector<int>& edgemarkers,
                       std::vector<std::vector<double> >& cellpnts, std::vector<int>& elvertslocal, int & nip, int & vertex_count);
void outputSliceMeshVTK (const Mesh& mesh, std::stringstream& fname, std::vector<std::vector<double> > & ipoints,
std::list<int> &celltypes, int cellstructusize, std::list<std::vector<int> > &elvrtindices);

void reorder_cellvertices ( int dim, int nip, std::vector<std::vector<double> > & cellpnts, std::vector<int> & elvertexes);
bool sortWedge3d(std::vector<std::vector<double> > & Points, int * permutation);
bool sortQuadril2d(std::vector<std::vector<double> > & Points, int * permutation);
double l2Norm(std::vector<double> vec);
double sprod(std::vector<double> vec1, std::vector<double> vec2);
// compares pairs<int,double> with respect to the second (double) elements
bool intdComparison(const std::pair<int,double> &a,const std::pair<int,double> &b);

ParGridFunction * FindParticularSolution(ParFiniteElementSpace *Hdiv_space, const HypreParMatrix & B, const Vector& rhs, bool verbose);

void ReplaceBlockByIdentityHpmat(BlockOperator& block_op, int i);

BlockOperator * ConstructDivfreeProblemOp(FOSLSDivfreeProblem& problem_divfree, FOSLSProblem& problem);

} // for namespace mfem


#endif
