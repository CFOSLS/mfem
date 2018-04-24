#include <iostream>
#include "testhead.hpp"

#ifndef MFEM_CFOSLS_TOOLS
#define MFEM_CFOSLS_TOOLS

using namespace std;
using namespace mfem;

namespace mfem
{

//HypreParMatrix * CopyRAPHypreParMatrix (HypreParMatrix& inputmat)
//HypreParMatrix * CopyHypreParMatrix (HypreParMatrix& inputmat)

template<typename T> void ConvertSTDvecToArray(std::vector<T>& stdvector, Array<int>& array_);


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

// a class for square block operators where each block is given as a HypreParMatrix
// used as an interface to handle coarsened operators for multigrid
// FIXME: Who should delete the matrices?
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
    ParMesh& pmesh;
    bool initialized;
protected:
    std::vector<std::vector<int> > bdr_attribs;

public:
    BdrConditions(ParMesh& pmesh_, int nblocks)
    : numblocks(nblocks), pmesh(pmesh_), initialized(false)
    {
        bdr_attribs.resize(numblocks);
        for (unsigned int i = 0; i < bdr_attribs.size(); ++i)
        {
            bdr_attribs[i].resize(pmesh.bdr_attributes.Max());
            for (unsigned int j = 0; j < bdr_attribs[i].size(); ++j)
                bdr_attribs[i][j] = -1;
        }
    }

    std::vector<std::vector<int> >* GetAllBdrAttribs()
    {
        if (initialized)
            return &bdr_attribs;
        else
            return NULL;
    }

    std::vector<int> * GetBdrAttribs(int blk)
    {
        MFEM_ASSERT(blk >= 0 && blk < numblocks, "Invalid block number in BdrConditions::GetBdrAttribs()");
        if (initialized)
            return &(bdr_attribs[blk]);
        else
            return NULL;
    }

    bool Initialized() const {return initialized;}
};

struct BdrConditions_CFOSLS_HdivL2_Hyper : public BdrConditions
{
public:
    BdrConditions_CFOSLS_HdivL2_Hyper(ParMesh& pmesh_)
        : BdrConditions(pmesh_, 2)
    {
        for (unsigned int j = 0; j < bdr_attribs[0].size(); ++j)
            bdr_attribs[0][j] = 0;
        bdr_attribs[0][0] = 1;

        for (unsigned int j = 0; j < bdr_attribs[1].size(); ++j)
            bdr_attribs[1][j] = 0;

        initialized = true;
    }

};

struct BdrConditions_CFOSLS_HdivH1_Hyper : public BdrConditions
{
public:
    BdrConditions_CFOSLS_HdivH1_Hyper(ParMesh& pmesh_)
        : BdrConditions(pmesh_, 3)
    {
        for (unsigned int j = 0; j < bdr_attribs[0].size(); ++j)
            bdr_attribs[0][j] = 0;

        for (unsigned int j = 0; j < bdr_attribs[1].size(); ++j)
            bdr_attribs[1][j] = 0;
        bdr_attribs[1][0] = 1;

        for (unsigned int j = 0; j < bdr_attribs[2].size(); ++j)
            bdr_attribs[2][j] = 0;

        initialized = true;
    }

};

enum SpaceName {HDIV = 0, H1 = 1, L2 = 2, HCURL = 3, HDIVSKEW = 4};

// a class for hierarchy of spaces of finite element spaces based on a nested sequence of meshes
class GeneralHierarchy
{
protected:
    int num_lvls;
    std::vector<ParMesh*> pmesh_lvls;
    std::vector<ParFiniteElementSpace* > Hdiv_space_lvls;
    std::vector<ParFiniteElementSpace* > Hcurl_space_lvls;
    std::vector<ParFiniteElementSpace* > H1_space_lvls;
    std::vector<ParFiniteElementSpace* > L2_space_lvls;
    std::vector<ParFiniteElementSpace* > Hdivskew_space_lvls;

    std::vector<SparseMatrix*> P_H1_lvls;
    std::vector<SparseMatrix*> P_Hdiv_lvls;
    std::vector<SparseMatrix*> P_L2_lvls;
    std::vector<SparseMatrix*> P_Hcurl_lvls;
    std::vector<SparseMatrix*> P_Hdivskew_lvls;

    std::vector<HypreParMatrix*> TrueP_H1_lvls;
    std::vector<HypreParMatrix*> TrueP_Hdiv_lvls;
    std::vector<HypreParMatrix*> TrueP_L2_lvls;
    std::vector<HypreParMatrix*> TrueP_Hcurl_lvls;
    std::vector<HypreParMatrix*> TrueP_Hdivskew_lvls;

public:

    GeneralHierarchy(int num_levels, ParMesh& pmesh, int feorder, bool verbose);

    void RefineAndCopy(int lvl, ParMesh* pmesh)
    {
        //if (!dynamic_cast<ParMeshCyl*> (pmesh))
            //std::cout << "Unsuccessful cast \n";
        ParMeshCyl * pmeshcyl_view = dynamic_cast<ParMeshCyl*> (pmesh);

        if (lvl == num_lvls - 1)
            if (pmeshcyl_view)
            {
                //ParMesh * temp = new ParMeshCyl(*pmeshcyl_view);
                //pmesh_lvls[lvl] = dynamic_cast<ParMesh*>(temp);
                pmesh_lvls[lvl] = new ParMeshCyl(*pmeshcyl_view);
            }
            else
                pmesh_lvls[lvl] = new ParMesh(*pmesh);
        else
        {
            if (pmeshcyl_view)
            {
                pmeshcyl_view->Refine(1);
                pmesh_lvls[lvl] = new ParMeshCyl(*pmeshcyl_view);
            }
            else
            {
                pmesh->UniformRefinement();
                pmesh_lvls[lvl] = new ParMesh(*pmesh);
            }
            //pmesh->UniformRefinement();
        }
    }

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

    ParFiniteElementSpace * GetSpace(SpaceName space, int level)
    {
        switch(space)
        {
        case HDIV:
            return Hdiv_space_lvls[level];
        case H1:
            return H1_space_lvls[level];
        case L2:
            return L2_space_lvls[level];
        case HCURL:
            return Hcurl_space_lvls[level];
        case HDIVSKEW:
            return Hdivskew_space_lvls[level];
        default:
            {
                MFEM_ABORT("Unknown or unsupported space name \n");
                break;
            }
        }

        return NULL;
    }

    HypreParMatrix * GetTruePspace(SpaceName space, int level)
    {
        switch(space)
        {
        case HDIV:
            return TrueP_Hdiv_lvls[level];
        case H1:
            return TrueP_H1_lvls[level];
        case L2:
            return TrueP_L2_lvls[level];
        case HCURL:
            return TrueP_Hcurl_lvls[level];
        case HDIVSKEW:
            return TrueP_Hdivskew_lvls[level];
        default:
            {
                MFEM_ABORT("Unknown or unsupported space name \n");
                break;
            }
        }

        return NULL;
    }

    SparseMatrix * GetPspace(SpaceName space, int level)
    {
        switch(space)
        {
        case HDIV:
            return P_Hdiv_lvls[level];
        case H1:
            return P_H1_lvls[level];
        case L2:
            return P_L2_lvls[level];
        case HCURL:
            return P_Hcurl_lvls[level];
        case HDIVSKEW:
            return P_Hdivskew_lvls[level];
        default:
            {
                MFEM_ABORT("Unknown or unsupported space name \n");
                break;
            }
        }

        return NULL;
    }

    int Nlevels() const {return num_lvls;}
};

class GeneralCylHierarchy : public GeneralHierarchy
{
protected:
    std::vector<ParMeshCyl*> pmeshcyl_lvls;

    std::vector<std::vector<std::pair<int,int> > > tdofs_link_H1_lvls;
    std::vector<std::vector<std::pair<int,int> > > tdofs_link_Hdiv_lvls;

    std::vector<HypreParMatrix*> TrueP_bndbot_H1_lvls;
    std::vector<HypreParMatrix*> TrueP_bndbot_Hdiv_lvls;
    std::vector<HypreParMatrix*> TrueP_bndtop_H1_lvls;
    std::vector<HypreParMatrix*> TrueP_bndtop_Hdiv_lvls;
    std::vector<HypreParMatrix*> Restrict_bot_H1_lvls;
    std::vector<HypreParMatrix*> Restrict_bot_Hdiv_lvls;
    std::vector<HypreParMatrix*> Restrict_top_H1_lvls;
    std::vector<HypreParMatrix*> Restrict_top_Hdiv_lvls;
protected:
    void ConstructRestrictions();
    void ConstructInterpolations();
    void ConstructTdofsLinks();

public:
    GeneralCylHierarchy(int num_levels, ParMeshCyl& pmesh, int feorder, bool verbose)
        : GeneralHierarchy(num_levels, pmesh, feorder, verbose)
    {
        pmeshcyl_lvls.resize(num_lvls);
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

// abstract structure for a (C)FOSLS formulation
// CFOSLS is considered to be a FOSLS formulation with constraint
// blk_structure is a vector of size = nblocks which contains:
// pairs <'a','b'> where, 'a' = 0,1,2 describes the type of the variable
// and 'b' is the index in the test coefficient array (corresponding to 'a')
// For example, <1,2> at place 3 means that the third equation corresponds
// to a vector unknown ('a' = 1), for which we have in the FOSLS_test a
// VectorFunctionCoefficient stored at test.vec_coeffs[2]('b' = 2).
// If a variable is not present in the FOSLS test (e.g., it's a Lagrange multiplier)
// then one must set 'a' = -1, 'b' = -1.

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
protected:
    virtual void InitBlkStructure() = 0;

public:
    FOSLSFormulation(int dimension, int num_blocks, int num_unknowns, bool do_have_constraint);

    virtual Array<SpaceName>& GetSpacesDescriptor() = 0;

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
};

struct CFOSLSFormulation_HdivL2Hyper : public FOSLSFormulation
{
protected:
    int numsol;
    Hyper_test test;
public:
    CFOSLSFormulation_HdivL2Hyper(int dimension, int num_solution, bool verbose);

    virtual FOSLS_test * GetTest() override {return &test;}
    virtual void InitBlkStructure() override;
    virtual Array<SpaceName>& GetSpacesDescriptor();

    int GetUnknownWithInitCnd() const override {return 0;}
};

struct CFOSLSFormulation_HdivH1Hyper : public FOSLSFormulation
{
protected:
    int numsol;
    Hyper_test test;
public:
    CFOSLSFormulation_HdivH1Hyper(int dimension, int num_solution, bool verbose);

    virtual FOSLS_test * GetTest() override {return &test;}
    virtual void InitBlkStructure() override;
    virtual Array<SpaceName>& GetSpacesDescriptor();

    int GetUnknownWithInitCnd() const override {return 1;}
};

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

struct CFOSLSFEFormulation_HdivL2Hyper : FOSLSFEFormulation
{
public:
    CFOSLSFEFormulation_HdivL2Hyper(FOSLSFormulation& formulation, int fe_order);
};

struct CFOSLSFEFormulation_HdivH1Hyper : FOSLSFEFormulation
{
public:
    CFOSLSFEFormulation_HdivH1Hyper(FOSLSFormulation& formulation, int fe_order);
};


class BlockProblemForms
{
    friend class CFOSLSHyperbolicProblem;
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

};


// class for general CFOSLS problem
class FOSLSProblem
{
protected:
    ParMesh& pmesh;

    FOSLSFEFormulation& fe_formul;

    BdrConditions& bdr_conds;

    GeneralHierarchy * hierarchy; // (optional)
    int level_in_hierarchy;       // (optional)

    bool spaces_initialized;
    bool forms_initialized;
    bool solver_initialized;
    bool hierarchy_initialized;

    // all par grid functions which are relevant to the formulation
    // e.g., solution components and right hand sides
    Array<ParGridFunction*> grfuns;

    Array<ParFiniteElementSpace*> pfes;
    BlockProblemForms pbforms;
    Array<ParLinearForm*> plforms;

    Array<int> blkoffsets_true;
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
    void InitSpacesFromHierarchy(GeneralHierarchy& hierarchy, int level, Array<SpaceName> &spaces_descriptor);
    void InitSpaces(ParMesh& pmesh);
    void InitForms();
    void AssembleSystem(bool verbose);
    void InitSolver(bool verbose);
    void SetPrec(Solver & Prec)
    {
        MFEM_ASSERT(solver_initialized, "Cannot set a preconditioner before the solver is initialized \n");
        prec = &Prec;
        solver->SetPreconditioner(*prec);
    }
    virtual void CreatePrec(BlockOperator & op, int prec_option, bool verbose) {}
    void UpdatePrec() { solver->SetPreconditioner(*prec); }
    void SetPrecOption(int option) { prec_option = option; }

    BlockVector * SetInitialCondition();
    BlockVector * SetTrueInitialCondition();
    void InitGrFuns();
    void DistributeSolution() const;
    void ComputeError(bool verbose, bool checkbnd) const;
    virtual void ComputeExtraError() const {}
public:
    FOSLSProblem(ParMesh& pmesh_, BdrConditions& bdr_conditions, FOSLSFEFormulation& fe_formulation, bool verbose_);
    FOSLSProblem(GeneralHierarchy& Hierarchy, int level, BdrConditions& bdr_conditions, FOSLSFEFormulation& fe_formulation, bool verbose_);
    void Solve(bool verbose) const;
    void Update(); // not implemented

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

    int GlobalTrueProblemSize() const {return CFOSLSop->Height();}
};

struct CFOSLSHyperbolicFormulation
{
    friend class CFOSLSHyperbolicProblem;

protected:
    const int dim;
    const int numsol;
    const char * space_for_S;
    const char * space_for_sigma;
    bool have_constraint;
    const int bdrattrnum;
    int numblocks;
    int unknowns_number;
    const char * formulation;
    //bool keep_divdiv; unsupported because then we need additional integrators (sum of smth)
    Array2D<BilinearFormIntegrator*> blfis;
    Array<LinearFormIntegrator*> lfis;
    Array<Array<int>* > essbdr_attrs;

public:
    CFOSLSHyperbolicFormulation(int dimension, int solution_number,
                            const char * S_space, const char * sigma_space,
                            bool with_constraint, int number_of_bdrattribs, bool verbose)
        : dim(dimension), numsol(solution_number),
          space_for_S(S_space), space_for_sigma(sigma_space),
          have_constraint(with_constraint), bdrattrnum(number_of_bdrattribs)
          //, keep_divdiv(with_divdiv)
    {
        if (with_constraint)
            formulation = "cfosls";
        else
            formulation = "fosls";
        MFEM_ASSERT(strcmp(formulation,"cfosls") == 0 || strcmp(formulation,"fosls") == 0,
                    "Formulation must be cfosls or fosls!\n");
        MFEM_ASSERT(strcmp(space_for_S,"H1") == 0 || strcmp(space_for_S,"L2") == 0,
                    "Space for S must be H1 or L2!\n");
        MFEM_ASSERT(strcmp(space_for_sigma,"Hdiv") == 0 || strcmp(space_for_sigma,"H1") == 0,
                    "Space for sigma must be Hdiv or H1!\n");
        MFEM_ASSERT(!strcmp(space_for_sigma,"H1") == 0 || (strcmp(space_for_sigma,"H1") == 0
                                                           && strcmp(space_for_S,"H1") == 0),
                    "Sigma from H1vec must be coupled with S from H1!\n");

        Transport_test Mytest(dim,numsol);

        numblocks = 1;

        if (strcmp(space_for_S,"H1") == 0)
            numblocks++;

        unknowns_number = numblocks;

        if (strcmp(formulation,"cfosls") == 0)
            numblocks++;

        if (verbose)
            std::cout << "Number of blocks in the formulation: " << numblocks << "\n";

        //if (strcmp(formulation,"cfosls") == 0)
            //essbdr_attrs.SetSize(numblocks - 1);
        //else // fosls
            //essbdr_attrs.SetSize(numblocks);
        essbdr_attrs.SetSize(numblocks);

        for (int i = 0; i < essbdr_attrs.Size(); ++i)
        {
            essbdr_attrs[i] = new Array<int>(bdrattrnum);
            (*essbdr_attrs[i]) = 0;
        }

        // S is from H1, so we impose bdr condition for S at t = 0
        if (strcmp(space_for_S,"H1") == 0)
            (*essbdr_attrs[1])[0] = 1; // t = 0;

        // S is from L2, so we impose bdr condition for sigma at t = 0
        if (strcmp(space_for_S,"L2") == 0)
            (*essbdr_attrs[0])[0] = 1; // t = 0;

        if (verbose)
        {
            std::cout << "Boundary conditions: \n";
            std::cout << "ess bdr for sigma: \n";
            essbdr_attrs[0]->Print(std::cout, bdrattrnum);
            if (strcmp(space_for_S,"H1") == 0)
            {
                std::cout << "ess bdr for S: \n";
                essbdr_attrs[1]->Print(std::cout, bdrattrnum);
            }
        }

        // bilinear forms
        blfis.SetSize(numblocks, numblocks);
        for (int i = 0; i < numblocks; ++i)
            for (int j = 0; j < numblocks; ++j)
                blfis(i,j) = NULL;

        int blkcount = 0;
        if (strcmp(space_for_S,"H1") == 0) // S is from H1
        {
            if (strcmp(space_for_sigma,"Hdiv") == 0) // sigma is from Hdiv
                blfis(0,0) = new VectorFEMassIntegrator;
            else // sigma is from H1vec
                blfis(0,0) = new ImproperVectorMassIntegrator;
        }
        else // "L2"
            blfis(0,0) = new VectorFEMassIntegrator(*Mytest.Ktilda);
        ++blkcount;

        if (strcmp(space_for_S,"H1") == 0)
        {
            if (strcmp(space_for_sigma,"Hdiv") == 0)
                blfis(1,1) = new H1NormIntegrator(*Mytest.bbT, *Mytest.bTb);
            else
                blfis(1,1) = new MassIntegrator(*Mytest.bTb);
            ++blkcount;
        }

        if (strcmp(space_for_S,"H1") == 0) // S is present
        {
            if (strcmp(space_for_sigma,"Hdiv") == 0) // sigma is from Hdiv
            {
                //Bblock->AddDomainIntegrator(new VectorFEMassIntegrator(*Mytest.b));
                blfis(1,0) = new VectorFEMassIntegrator(*Mytest.minb);
            }
            else // sigma is from H1
                blfis(1,0) = new MixedVectorScalarIntegrator(*Mytest.minb);
        }

        if (strcmp(formulation,"cfosls") == 0)
        {
           if (strcmp(space_for_sigma,"Hdiv") == 0) // sigma is from Hdiv
             blfis(blkcount,0) = new VectorFEDivergenceIntegrator;
           else // sigma is from H1vec
             blfis(blkcount,0) = new VectorDivergenceIntegrator;
        }

        // linear forms
        lfis.SetSize(numblocks);
        for (int i = 0; i < numblocks; ++i)
            lfis[i] = NULL;

        blkcount = 1;
        if (strcmp(space_for_S,"H1") == 0)
        {
            lfis[1] = new GradDomainLFIntegrator(*Mytest.bf);
            ++blkcount;
        }

        if (strcmp(formulation,"cfosls") == 0)
            lfis[blkcount] = new DomainLFIntegrator(*Mytest.scalardivsigma);
    }

};

class CFOSLSHyperbolicProblem
{
protected:
    int feorder;
    CFOSLSHyperbolicFormulation& struct_formul;
    bool spaces_initialized;
    bool forms_initialized;
    bool solver_initialized;

    FiniteElementCollection *hdiv_coll;
    FiniteElementCollection *h1_coll;
    FiniteElementCollection *l2_coll;
    ParFiniteElementSpace * Hdiv_space;
    ParFiniteElementSpace * H1_space;
    ParFiniteElementSpace * H1vec_space;
    ParFiniteElementSpace * L2_space;

    // FIXME: to be removed in the abstract base class
    ParFiniteElementSpace * Sigma_space;
    ParFiniteElementSpace * S_space;

    // all par grid functions which are relevant to the formulation
    // e.g., solution components and right hand sides
    Array<ParGridFunction*> grfuns;

    Array<ParFiniteElementSpace*> pfes;
    BlockProblemForms pbforms;
    Array<ParLinearForm*> plforms;


    Array<int> blkoffsets_true;
    Array<int> blkoffsets;
    Array2D<HypreParMatrix*> hpmats;
    BlockOperator *CFOSLSop;
    Array2D<HypreParMatrix*> hpmats_nobnd;
    BlockOperator *CFOSLSop_nobnd;
    BlockVector * trueRhs;
    BlockVector * trueX;
    BlockVector * trueBnd;
    BlockVector * x; // inital condition (~bnd conditions)
    BlockDiagonalPreconditioner *prec;
    IterativeSolver * solver;

    StopWatch chrono;

protected:
    void InitFEColls(bool verbose);
    void InitSpaces(ParMesh& pmesh);
    void InitForms();
    void AssembleSystem(bool verbose);
    void InitSolver(bool verbose);
    void InitPrec(int prec_option, bool verbose);
    BlockVector *  SetInitialCondition();
    BlockVector * SetTrueInitialCondition();
    void InitGrFuns();
    void DistributeSolution();
    void ComputeError(bool verbose, bool checkbnd);
public:
    CFOSLSHyperbolicProblem(CFOSLSHyperbolicFormulation& struct_formulation,
                            int fe_order, bool verbose);
    CFOSLSHyperbolicProblem(ParMesh& pmesh, CFOSLSHyperbolicFormulation& struct_formulation,
                            int fe_order, int prec_option, bool verbose);
    void BuildCFOSLSSystem(ParMesh& pmesh, bool verbose);
    void Solve(bool verbose);
    void Update();
    // deletes everything which was related to a specific mesh
    void Reset() {MFEM_ABORT("Not implemented \n");}
};


} // for namespace mfem


#endif
