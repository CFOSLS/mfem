#include <iostream>
#include "testhead.hpp"

#ifndef MFEM_CFOSLS_TIMESTEPPING
#define MFEM_CFOSLS_TIMESTEPPING

using namespace std;
using namespace mfem;

namespace mfem
{

// base class for a FOSLS problem in a time cylinder
class FOSLSCylProblem : public FOSLSProblem
{
public:
    using FOSLSProblem::Solve;
protected:
    ParMeshCyl &pmeshcyl;
    GeneralCylHierarchy * cyl_hierarchy;

    SpaceName init_cond_space;
    int init_cond_block;

    std::vector<std::pair<int,int> > tdofs_link;
    HypreParMatrix* Restrict_bot;
    HypreParMatrix* Restrict_top;

protected:
    void ConstructTdofLink();
    void ConstructRestrictions();

public:
    FOSLSCylProblem (ParMeshCyl& Pmeshcyl, BdrConditions& bdr_conditions,
                    FOSLSFEFormulation& fe_formulation, bool verbose_)
        : FOSLSProblem(Pmeshcyl, bdr_conditions, fe_formulation, verbose_),
          pmeshcyl(Pmeshcyl), cyl_hierarchy(NULL),
          init_cond_block(fe_formul.GetFormulation()->GetUnknownWithInitCnd())
    {
        Array<SpaceName>& spacenames = fe_formul.GetFormulation()->GetSpacesDescriptor();
        init_cond_space = spacenames[init_cond_block];
        ConstructTdofLink();
    }

    FOSLSCylProblem(GeneralCylHierarchy& Hierarchy, int level, BdrConditions& bdr_conditions,
                   FOSLSFEFormulation& fe_formulation, bool verbose_)
        : FOSLSProblem(Hierarchy, level, bdr_conditions, fe_formulation, verbose_),
          pmeshcyl(*Hierarchy.GetPmeshcyl(level)), cyl_hierarchy(&Hierarchy),
          init_cond_block(fe_formul.GetFormulation()->GetUnknownWithInitCnd())
    {
        Array<SpaceName>& spacenames = fe_formul.GetFormulation()->GetSpacesDescriptor();
        init_cond_space = spacenames[init_cond_block];
        tdofs_link = *cyl_hierarchy->GetTdofsLink(level, init_cond_space);
    }

    // (delete this?)
    // interpretation of the input and output vectors depend on the implementation of Solve
    // but they are related to the boundary conditions at the input and at he output
    //virtual void Solve(const Vector& vec_in, Vector& vec_out) const = 0;

    ParMeshCyl * GetParMeshCyl() {return &pmeshcyl;}

    void Solve(const Vector& rhs, Vector& sol, const Vector &bnd_tdofs_bot, Vector &bnd_tdofs_top) const;
    void Solve(const Vector &bnd_tdofs_bot, Vector &bnd_tdofs_top) const;

protected:
    void ExtractTopTdofs(Vector& bnd_tdofs_top) const;
    void ExtractBotTdofs(Vector& bnd_tdofs_bot) const;
public:
    void CorrectRhsFromInitCnd(const Vector& bnd_tdofs_bot) const
    { CorrectRhsFromInitCnd(*CFOSLSop_nobnd, bnd_tdofs_bot);}

    void CorrectRhsFromInitCnd(const Operator& op, const Vector& bnd_tdofs_bot) const;

};

class FOSLSCylProblem_CFOSLS_HdivL2_Hyper : public FOSLSCylProblem
{
protected:
    virtual void CreatePrec(BlockOperator &op, int prec_option, bool verbose) override;
public:
    FOSLSCylProblem_CFOSLS_HdivL2_Hyper(ParMeshCyl& Pmeshcyl, BdrConditions& bdr_conditions,
                    FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSCylProblem(Pmeshcyl, bdr_conditions, fe_formulation, verbose_)
    {
        SetPrecOption(precond_option);
        CreatePrec(*CFOSLSop, prec_option, verbose);
        UpdatePrec();
    }

    FOSLSCylProblem_CFOSLS_HdivL2_Hyper(GeneralCylHierarchy& Hierarchy, int level, BdrConditions& bdr_conditions,
                   FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSCylProblem(Hierarchy, level, bdr_conditions, fe_formulation, verbose_)
    {
        SetPrecOption(precond_option);
        CreatePrec(*CFOSLSop, prec_option, verbose);
        UpdatePrec();
    }

    void ComputeExtraError() const;
};


// abstract base class for a problem in a time cylinder
class TimeCyl
{
protected:
    ParMeshCyl * pmeshtsl;
    double t_init;
    double tau;
    int nt;
    bool own_pmeshtsl;
public:
    virtual ~TimeCyl();
    TimeCyl (ParMesh& Pmeshbase, double T_init, double Tau, int Nt);
    TimeCyl (ParMeshCyl& Pmeshtsl) : pmeshtsl(&Pmeshtsl), own_pmeshtsl(false) {}

    // interpretation of the input and output vectors depend on the implementation of Solve
    // but they are related to the boundary conditions at the input and at he output
    virtual void Solve(const Vector& vec_in, Vector& vec_out) const = 0;
};

// specific class for time-slabbing in hyperbolic problems
class TimeCylHyper : public TimeCyl
{
private:
    MPI_Comm comm;

protected:
    int ref_lvls;
    const char *formulation;
    const char *space_for_S;
    const char *space_for_sigma;
    int feorder;
    int dim;
    int numsol;

    std::vector<int> ess_bdrat_S;
    std::vector<int> ess_bdrat_sigma;

    GeneralCylHierarchy * hierarchy;

    std::vector<ParFiniteElementSpace* > Sigma_space_lvls; // shortcut (may be useful if consider vector H1 for sigma at some moment
    std::vector<ParFiniteElementSpace* > S_space_lvls;     // shortcut

    std::vector<Array<int>*> block_trueOffsets_lvls;
    std::vector<BlockOperator*> CFOSLSop_lvls;
    std::vector<BlkHypreOperator*> CFOSLSop_coarsened_lvls;
    std::vector<BlockOperator*> CFOSLSop_nobnd_lvls;
    std::vector<BlockDiagonalPreconditioner*> prec_lvls;
    std::vector<MINRESSolver*> solver_lvls;
    std::vector<BlockVector*> trueRhs_nobnd_lvls;
    std::vector<BlockVector*> trueX_lvls;

    std::vector<BlockOperator*> TrueP_lvls;

    bool visualization;
public:
    bool verbose;

protected:
    void InitProblem(int numsol);

public:
    ~TimeCylHyper();
    TimeCylHyper (ParMesh& Pmeshbase, double T_init, double Tau, int Nt, int Ref_lvls,
                   const char *Formulation, const char *Space_for_S, const char *Space_for_sigma, int Numsol);
    TimeCylHyper (ParMeshCyl& Pmeshtsl, int Ref_Lvls,
                   const char *Formulation, const char *Space_for_S, const char *Space_for_sigma, int Numsol);

    virtual void Solve(const Vector &bnd_tdofs_bot, Vector &bnd_tdofs_top) const override
    { Solve(0, bnd_tdofs_bot, bnd_tdofs_top); }

    void Solve(int lvl, const Vector &bnd_tdofs_bot, Vector &bnd_tdofs_top) const;
    void Solve(int lvl, const Vector& rhs, Vector& sol, const Vector &bnd_tdofs_bot, Vector &bnd_tdofs_top) const;
    void Solve(const char * mode, int lvl, const Vector& rhs, Vector& sol, const Vector &bnd_tdofs_bot, Vector &bnd_tdofs_top) const;

    void ComputeAnalyticalRhs(int lvl);

    GeneralCylHierarchy * GetHierarchy() {return hierarchy;}

    int GetInitCondSize(int lvl) const
    {
        if (strcmp(space_for_S,"H1") == 0)
            return hierarchy->GetTdofs_H1_link(lvl)->size();
        else
            return hierarchy->GetTdofs_Hdiv_link(lvl)->size();
    }

    int GetNLevels() {return ref_lvls;}

    std::vector<std::pair<int,int> > * GetTdofsLink(int lvl)
    {
        if (strcmp(space_for_S,"H1") == 0)
            return hierarchy->GetTdofs_H1_link(lvl);
        else
            return hierarchy->GetTdofs_Hdiv_link(lvl);
    }

    ParFiniteElementSpace * Get_S_space(int lvl = 0) {return S_space_lvls[lvl];}
    ParFiniteElementSpace * Get_Sigma_space(int lvl = 0) {return Sigma_space_lvls[lvl];}

    /*
    HypreParMatrix * Get_TrueP_H1(int lvl)
    {
        if (lvl >= 0 && lvl < ref_lvls)
            if (TrueP_H1_lvls[lvl])
                return TrueP_H1_lvls[lvl];
        return NULL;
    }
    HypreParMatrix * Get_TrueP_Hdiv(int lvl)
    {
        if (lvl >= 0 && lvl < ref_lvls)
            if (TrueP_Hdiv_lvls[lvl])
                return TrueP_Hdiv_lvls[lvl];
        return NULL;
    }
    */

    ParMeshCyl * GetParMeshCyl(int lvl) { return hierarchy->GetPmeshcyl(lvl);}

    Vector* GetSol(int lvl)
    { return trueX_lvls[lvl]; }

    Vector *GetExactBase(const char * top_or_bot, int level);

    Array<int>* GetBlockTrueOffsets(int lvl)
    { return block_trueOffsets_lvls[lvl]; }

    int ProblemSize(int lvl)
    {return CFOSLSop_lvls[lvl]->Height();}

    bool NeedSignSwitch() {return (strcmp(space_for_S, "L2") == 0);}

    void InterpolateAtBase(const char * top_or_bot, int lvl, const Vector& vec_in, Vector& vec_out);

    void Interpolate(int lvl, const Vector& vec_in, Vector& vec_out);

    // FIXME: Does one need to scale the restriction?
    // Yes, in general it should be a canonical interpolator transpose, not just transpose of the standard
    void RestrictAtBase(const char * top_or_bot, int lvl, const Vector& vec_in, Vector& vec_out);

    void Restrict(int lvl, const Vector& vec_in, Vector& vec_out);

    // Takes a vector of values corresponding to the bdr condition
    // and computes the corresponding change to the rhs side
    void ConvertBdrCndIntoRhs(int lvl, const Vector& vec_in, Vector& vec_out);

    // vec_in is considered as a vector of strictly values at the bottom boundary,
    // vec_out is a full vector which coincides with vec_in at initial boundary and
    // has 0's for all the rest entries
    void ConvertInitCndToFullVector(int lvl, const Vector& vec_in, Vector& vec_out);

    void ComputeResidual(int lvl, const Vector& initcond_in, const Vector& sol, Vector& residual);

    void ComputeError(int lvl, Vector& sol) const;

};

class TimeSteppingScheme
{
protected:
    std::vector<TimeCylHyper*> timeslab_problems;
    int nslabs;
    int nlevels;
    bool verbose;
    std::vector<std::vector<Vector*> > vec_ins_lvls;
    std::vector<std::vector<Vector*> > vec_outs_lvls;

    std::vector<std::vector<Vector*> > residuals_lvls;
    std::vector<std::vector<Vector*> > sols_lvls;

public:
    TimeSteppingScheme (std::vector<TimeCylHyper*> & timeslab_problems);

    void Solve(char const * mode, const char *level_mode, std::vector<Vector *> rhss, int level, bool compute_accuracy);
    void Solve(char const * mode, const char *level_mode, int level, bool compute_accuracy);

    void ComputeResiduals(int level);

    void RestrictToCoarser(int level, std::vector<Vector*> vec_ins, std::vector<Vector*> vec_outs);
    void InterpolateToFiner(int level, std::vector<Vector*> vec_ins, std::vector<Vector*> vec_outs);

    void ComputeAnalyticalRhs(int level);

    std::vector<Vector*> * Get_vec_ins(int level){return &vec_ins_lvls[level];}
    std::vector<Vector*> * Get_vec_outs(int level){return &vec_outs_lvls[level];}
    std::vector<Vector*> * Get_sols(int level){return &sols_lvls[level];}
    std::vector<Vector*> * Get_residuals(int level){return &residuals_lvls[level];}

    TimeCylHyper * GetTimeSlab(int tslab) {return timeslab_problems[tslab];}

    void SetInitialCondition(const Vector& x_init, int level)
    { *vec_ins_lvls[level][0] = x_init;}

    void SetInitialConditions(std::vector<Vector*> x_inits, int level)
    {
        MFEM_ASSERT( (int) (x_inits.size()) >= nslabs, "Number of initial vectors is less than number of time slabs! \n");
        for (int tslab = 0; tslab < nslabs; ++tslab)
            *vec_ins_lvls[level][tslab] = *x_inits[tslab];
    }

    int GetNSlabs() {return nslabs;}
    int GetNLevels() {return nlevels;}
};


class SpaceTimeTwoGrid
{
protected:
    TimeSteppingScheme& timestepping;
    int nslabs;
    int max_iter;
    double tol;

    const int num_lvls = 2;

    std::vector<std::vector<Vector*> > res_lvls;
    std::vector<std::vector<Vector*> > corr_lvls;

public:
    SpaceTimeTwoGrid(TimeSteppingScheme& TimeStepping, int Max_Iter, double Tol )
        : timestepping(TimeStepping),
          nslabs(TimeStepping.GetNSlabs()),
          max_iter(Max_Iter),
          tol(Tol)
    {
        MFEM_ASSERT(timestepping.GetNLevels() > 1, "For a two-grid method at least two levels must exist! \n");

        res_lvls.resize(num_lvls);
        for (unsigned int l = 0; l < res_lvls.size(); ++l)
        {
            res_lvls[l].resize(nslabs);
            for (int slab = 0; slab < nslabs; ++slab)
                res_lvls[l][slab] = new Vector(timestepping.GetTimeSlab(slab)->ProblemSize(l));
        }

        corr_lvls.resize(num_lvls);
        for (unsigned int l = 0; l < corr_lvls.size(); ++l)
        {
            corr_lvls[l].resize(nslabs);
            for (int slab = 0; slab < nslabs; ++slab)
                corr_lvls[l][slab] = new Vector(timestepping.GetTimeSlab(slab)->ProblemSize(l));
        }

    }
    void Solve(std::vector<Vector *> rhss, std::vector<Vector *> sols);

protected:
    void Iterate(std::vector<Vector*> ress, std::vector<Vector*> corrs);
    void ComputeResidual(std::vector<Vector*> rhss, std::vector<Vector*> sols);
    void UpdateResidual(std::vector<Vector*> corrs);
    void UpdateSolution(std::vector<Vector*> sols, std::vector<Vector*> corrs);

};

} // for namespace mfem


#endif
