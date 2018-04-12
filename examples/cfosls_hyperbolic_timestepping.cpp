//                       CFOSLS formultation for transport equation in 3D/4D with time-slabbing technique

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>
#include <unistd.h>

#include "cfosls_testsuite.hpp"
#include "divfree_solver_tools.hpp"

#define ZEROTOL (1.0e-13)

#define NONHOMO_TEST

// must be active
#define USE_TSL

using namespace std;
using namespace mfem;

// TODO: Instead of specifying tdofs_link_H1 and _Hdiv and manually choosing by if-clauses,
// which to use for the Solve() int TimeCyl, it would be better to implement it as a block case
// with arbitrary number of blocks. Then input and output would be BlockVectors and there will be
// less switche calls

std::vector<std::pair<int,int> >* CreateBotToTopDofsLink(const char * eltype, FiniteElementSpace& fespace,
                                                         std::vector<std::pair<int,int> > & bot_to_top_bels, bool verbose = false);
HypreParMatrix * CreateRestriction(const char * top_or_bot, ParFiniteElementSpace& pfespace,
                                   std::vector<std::pair<int,int> >& bot_to_top_tdofs_link);

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

TimeCyl::~TimeCyl()
{
    if (own_pmeshtsl)
        delete pmeshtsl;
}

TimeCyl::TimeCyl(ParMesh& Pmeshbase, double T_init, double Tau, int Nt)
    : t_init(T_init), tau(Tau), nt(Nt), own_pmeshtsl(true)
{
    pmeshtsl = new ParMeshCyl(Pmeshbase.GetComm(), Pmeshbase, t_init, tau, nt);
}

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

    std::vector<ParMeshCyl*> pmeshtsl_lvls;
    std::vector<ParFiniteElementSpace* > Hdiv_space_lvls;
    std::vector<ParFiniteElementSpace* > H1_space_lvls;
    std::vector<ParFiniteElementSpace* > L2_space_lvls;
    std::vector<ParFiniteElementSpace* > Sigma_space_lvls; // shortcut (may be useful if consider vector H1 for sigma at some moment
    std::vector<ParFiniteElementSpace* > S_space_lvls;     // shortcut

    std::vector<Array<int>*> block_trueOffsets_lvls;
    std::vector<BlockOperator*> CFOSLSop_lvls;
    std::vector<Operator*> CFOSLSop_coarsened_lvls;
    std::vector<BlockOperator*> CFOSLSop_nobnd_lvls;
    std::vector<BlockDiagonalPreconditioner*> prec_lvls;
    std::vector<MINRESSolver*> solver_lvls;
    std::vector<BlockVector*> trueRhs_nobnd_lvls;
    std::vector<BlockVector*> trueX_lvls;

    std::vector<int> init_cond_size_lvls;
    std::vector<std::vector<std::pair<int,int> > > tdofs_link_H1_lvls;
    std::vector<std::vector<std::pair<int,int> > > tdofs_link_Hdiv_lvls;

    std::vector<SparseMatrix*> P_H1_lvls;
    std::vector<SparseMatrix*> P_Hdiv_lvls;
    std::vector<SparseMatrix*> P_L2_lvls;
    std::vector<HypreParMatrix*> TrueP_H1_lvls;
    std::vector<HypreParMatrix*> TrueP_Hdiv_lvls;
    std::vector<HypreParMatrix*> TrueP_L2_lvls;
    std::vector<BlockOperator*> TrueP_lvls;

    std::vector<HypreParMatrix*> TrueP_bndbot_H1_lvls;
    std::vector<HypreParMatrix*> TrueP_bndbot_Hdiv_lvls;
    std::vector<HypreParMatrix*> TrueP_bndtop_H1_lvls;
    std::vector<HypreParMatrix*> TrueP_bndtop_Hdiv_lvls;
    std::vector<HypreParMatrix*> Restrict_bot_H1_lvls;
    std::vector<HypreParMatrix*> Restrict_bot_Hdiv_lvls;
    std::vector<HypreParMatrix*> Restrict_top_H1_lvls;
    std::vector<HypreParMatrix*> Restrict_top_Hdiv_lvls;

    bool verbose;
    bool visualization;

protected:
    void InitProblem();

public:
    ~TimeCylHyper();
    TimeCylHyper (ParMesh& Pmeshbase, double T_init, double Tau, int Nt, int Ref_lvls,
                   const char *Formulation, const char *Space_for_S, const char *Space_for_sigma);
    TimeCylHyper (ParMeshCyl& Pmeshtsl, int Ref_Lvls,
                   const char *Formulation, const char *Space_for_S, const char *Space_for_sigma);

    virtual void Solve(const Vector &bnd_tdofs_bot, Vector &bnd_tdofs_top) const override
    { Solve(0, bnd_tdofs_bot, bnd_tdofs_top); }

    void Solve(int lvl, const Vector &bnd_tdofs_bot, Vector &bnd_tdofs_top) const;
    void Solve(int lvl, const Vector& rhs, Vector& sol, const Vector &bnd_tdofs_bot, Vector &bnd_tdofs_top) const;
    void Solve(const char * mode, int lvl, const Vector& rhs, Vector& sol, const Vector &bnd_tdofs_bot, Vector &bnd_tdofs_top) const;
    int GetInitCondSize(int lvl) {return init_cond_size_lvls[lvl];}
    std::vector<std::pair<int,int> > * GetTdofsLink(int lvl)
    {
        if (strcmp(space_for_S,"H1") == 0)
            return &(tdofs_link_H1_lvls[lvl]);
        else
            return &(tdofs_link_Hdiv_lvls[lvl]);
    }

    ParFiniteElementSpace * Get_S_space(int lvl = 0) {return S_space_lvls[lvl];}
    ParFiniteElementSpace * Get_Sigma_space(int lvl = 0) {return Sigma_space_lvls[lvl];}
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

    ParMeshCyl * Get_ParMeshCyl(int lvl)
    {
        if (lvl >= 0 && lvl <= ref_lvls)
            if (pmeshtsl_lvls[lvl])
                return pmeshtsl_lvls[lvl];
        return NULL;
    }

    Vector* GetSol(int lvl)
    { return trueX_lvls[lvl]; }

    Array<int>* GetBlockTrueOffsets(int lvl)
    { return block_trueOffsets_lvls[lvl]; }

    int ProblemSize(int lvl)
    {return CFOSLSop_lvls[lvl]->Height();}

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

TimeCylHyper::~TimeCylHyper()
{
    for (unsigned int i = 0; i < Sigma_space_lvls.size(); ++i)
        delete Sigma_space_lvls[i];
    for (unsigned int i = 0; i < S_space_lvls.size(); ++i)
        delete S_space_lvls[i];
    if (strcmp(space_for_S,"H1") == 0)
        for (unsigned int i = 0; i < L2_space_lvls.size(); ++i)
            delete L2_space_lvls[i];
    for (unsigned int i = 0; i < CFOSLSop_lvls.size(); ++i)
        delete CFOSLSop_lvls[i];
    for (unsigned int i = 1; i < CFOSLSop_coarsened_lvls.size(); ++i)
        delete CFOSLSop_coarsened_lvls[i];
    for (unsigned int i = 0; i < CFOSLSop_nobnd_lvls.size(); ++i)
        delete CFOSLSop_nobnd_lvls[i];
    for (unsigned int i = 0; i < trueRhs_nobnd_lvls.size(); ++i)
        delete trueRhs_nobnd_lvls[i];
    for (unsigned int i = 0; i < trueX_lvls.size(); ++i)
        delete trueX_lvls[i];
    for (unsigned int i = 0; i < prec_lvls.size(); ++i)
        delete prec_lvls[i];
    for (unsigned int i = 0; i < solver_lvls.size(); ++i)
        delete solver_lvls[i];
    for (unsigned int i = 0; i < P_H1_lvls.size(); ++i)
        delete P_H1_lvls[i];
    for (unsigned int i = 0; i < P_Hdiv_lvls.size(); ++i)
        delete P_Hdiv_lvls[i];
    for (unsigned int i = 0; i < TrueP_H1_lvls.size(); ++i)
        delete TrueP_H1_lvls[i];
    for (unsigned int i = 0; i < TrueP_Hdiv_lvls.size(); ++i)
        delete TrueP_Hdiv_lvls[i];
    for (unsigned int i = 0; i < P_L2_lvls.size(); ++i)
        delete P_L2_lvls[i];
    for (unsigned int i = 0; i < TrueP_L2_lvls.size(); ++i)
        delete TrueP_L2_lvls[i];

    for (unsigned int i = 0; i < TrueP_bndbot_H1_lvls.size(); ++i)
        delete TrueP_bndbot_H1_lvls[i];
    for (unsigned int i = 0; i < TrueP_bndbot_Hdiv_lvls.size(); ++i)
        delete TrueP_bndbot_Hdiv_lvls[i];
    for (unsigned int i = 0; i < TrueP_bndtop_H1_lvls.size(); ++i)
        delete TrueP_bndtop_H1_lvls[i];
    for (unsigned int i = 0; i < TrueP_bndtop_Hdiv_lvls.size(); ++i)
        delete TrueP_bndtop_Hdiv_lvls[i];

    for (unsigned int i = 0; i < pmeshtsl_lvls.size(); ++i)
        delete pmeshtsl_lvls[i];
    for (unsigned int i = 0; i < block_trueOffsets_lvls.size(); ++i)
        delete block_trueOffsets_lvls[i];
    for (unsigned int i = 0; i < Restrict_bot_H1_lvls.size(); ++i)
        delete Restrict_bot_H1_lvls[i];
    for (unsigned int i = 0; i < Restrict_bot_Hdiv_lvls.size(); ++i)
        delete Restrict_bot_Hdiv_lvls[i];
    for (unsigned int i = 0; i < Restrict_top_H1_lvls.size(); ++i)
        delete Restrict_top_H1_lvls[i];
    for (unsigned int i = 0; i < Restrict_top_Hdiv_lvls.size(); ++i)
        delete Restrict_top_Hdiv_lvls[i];
}

void TimeCylHyper::Interpolate(int lvl, const Vector& vec_in, Vector& vec_out)
{
    BlockVector viewer_in(vec_in.GetData(),  *block_trueOffsets_lvls[lvl + 1]);
    BlockVector viewer_out(vec_out.GetData(),  *block_trueOffsets_lvls[lvl]);
    TrueP_lvls[lvl]->Mult(viewer_in, viewer_out);
}

void TimeCylHyper::InterpolateAtBase(const char * top_or_bot, int lvl, const Vector& vec_in, Vector& vec_out)
{
    //MFEM_ABORT("Interpolate not implemented \n");
    if (strcmp(space_for_S, "H1") == 0)
    {
        if (strcmp(top_or_bot, "top") == 0)
            TrueP_bndtop_H1_lvls[lvl]->Mult(vec_in, vec_out);
        else if (strcmp(top_or_bot, "bot") == 0)
            TrueP_bndbot_H1_lvls[lvl]->Mult(vec_in, vec_out);
        else
        {
            MFEM_ABORT("In TimeCylHyper::InterpolateAtBase() top_or_bot must be 'top' or 'bot'!");
        }
    }
    else
    {
        if (strcmp(top_or_bot, "top") == 0)
            TrueP_bndtop_Hdiv_lvls[lvl]->Mult(vec_in, vec_out);
        else if (strcmp(top_or_bot, "bot") == 0)
            TrueP_bndbot_Hdiv_lvls[lvl]->Mult(vec_in, vec_out);
        else
        {
            MFEM_ABORT("In TimeCylHyper::InterpolateAtBase() top_or_bot must be 'top' or 'bot'!");
        }

    }
}

void TimeCylHyper::Restrict(int lvl, const Vector& vec_in, Vector& vec_out)
{
    BlockVector viewer_in(vec_in.GetData(),  *block_trueOffsets_lvls[lvl]);
    BlockVector viewer_out(vec_out.GetData(),  *block_trueOffsets_lvls[lvl + 1]);
    TrueP_lvls[lvl]->MultTranspose(viewer_in, viewer_out);
    // FIXME: Do we need to clear the boundary conditions on the coarse level after that?
    // I guess, no.
}


void TimeCylHyper::RestrictAtBase(const char * top_or_bot, int lvl, const Vector& vec_in, Vector& vec_out)
{
    if (strcmp(space_for_S, "H1") == 0)
    {
        if (strcmp(top_or_bot, "top") == 0)
            TrueP_bndtop_H1_lvls[lvl - 1]->MultTranspose(vec_in, vec_out);
        else if (strcmp(top_or_bot, "bot") == 0)
            TrueP_bndbot_H1_lvls[lvl - 1]->MultTranspose(vec_in, vec_out);
        else
        {
            MFEM_ABORT("In TimeCylHyper::RestrictAtBase() top_or_bot must be 'top' or 'bot'!");
        }
    }
    else
    {
        if (strcmp(top_or_bot, "top") == 0)
            TrueP_bndtop_Hdiv_lvls[lvl - 1]->MultTranspose(vec_in, vec_out);
        else if (strcmp(top_or_bot, "bot") == 0)
            TrueP_bndbot_Hdiv_lvls[lvl - 1]->MultTranspose(vec_in, vec_out);
        else
        {
            MFEM_ABORT("In TimeCylHyper::RestrictAtBase() top_or_bot must be 'top' or 'bot'!");
        }

    }
}

void TimeCylHyper::ConvertInitCndToFullVector(int lvl, const Vector& vec_in, Vector& vec_out)
{
    BlockVector viewer(vec_out.GetData(),  *block_trueOffsets_lvls[lvl]);
    vec_out = 0.0;

    if (strcmp(space_for_S, "H1") == 0)
    {
        if (vec_in.Size() != tdofs_link_H1_lvls[lvl].size())
        {
            MFEM_ABORT("Size of vec_in in ConvertInitCndToFullVector differs from the size of tdofs_link_H1_lvls \n");
        }

        for (int i = 0; i < vec_in.Size(); ++i)
        {
            int tdof = tdofs_link_H1_lvls[lvl][i].first;
            viewer.GetBlock(1)[tdof] = vec_in[i];
        }
    }
    else
    {
        if (vec_in.Size() != tdofs_link_Hdiv_lvls[lvl].size())
        {
            MFEM_ABORT("Size of vec_in in ConvertInitCndToFullVector differs from the size of tdofs_link_Hdiv_lvls \n");
        }

        for (int i = 0; i < vec_in.Size(); ++i)
        {
            int tdof = tdofs_link_H1_lvls[lvl][i].first;
            viewer.GetBlock(0)[tdof] = vec_in[i];
        }
    }

}

// it is assumed that CFOSLSop_nobnd was already created
void TimeCylHyper::ConvertBdrCndIntoRhs(int lvl, const Vector& vec_in, Vector& vec_out)
{
    vec_out = 0.0;
    CFOSLSop_nobnd_lvls[lvl]->Mult(vec_in, vec_out);

    BlockVector viewer(vec_out.GetData(), *block_trueOffsets_lvls[lvl]);

    if (strcmp(space_for_S, "H1") == 0)
    {
        Array<int> essbdr_tdofs;

        Array<int> ess_bdr_S(ess_bdrat_S.size());
        for (unsigned int i = 0; i < ess_bdrat_S.size(); ++i)
            ess_bdr_S[i] = ess_bdrat_S[i];

        S_space_lvls[lvl]->GetEssentialTrueDofs(ess_bdr_S, essbdr_tdofs);

        for (int i = 0; i < essbdr_tdofs.Size(); ++i)
        {
            int tdof = essbdr_tdofs[i];
            viewer.GetBlock(1)[tdof] = vec_in[tdof];
        }
    }
    else
    {
        Array<int> essbdr_tdofs;

        Array<int> ess_bdr_sigma(ess_bdrat_sigma.size());
        for (unsigned int i = 0; i < ess_bdrat_sigma.size(); ++i)
            ess_bdr_sigma[i] = ess_bdrat_sigma[i];

        Sigma_space_lvls[lvl]->GetEssentialTrueDofs(ess_bdr_sigma, essbdr_tdofs);

        for (int i = 0; i < essbdr_tdofs.Size(); ++i)
        {
            int tdof = essbdr_tdofs[i];
            viewer.GetBlock(0)[tdof] = vec_in[tdof];
        }
    }
}

void TimeCylHyper::ComputeResidual(int lvl, const Vector& initcond_in, const Vector& sol, Vector& residual)
{
    // transform vector with initial condition values into a full vector with nonzero bdr values
    BlockVector full_initcond_in(*block_trueOffsets_lvls[lvl]);
    ConvertInitCndToFullVector(lvl, initcond_in, full_initcond_in);

    // compute the correction to the rhs which is implied by bdr vector
    BlockVector bdr_corr(*block_trueOffsets_lvls[lvl]);
    ConvertBdrCndIntoRhs(lvl, full_initcond_in, bdr_corr);

    BlockVector Asol(*block_trueOffsets_lvls[lvl]);
    CFOSLSop_lvls[lvl]->Mult(sol, Asol);

    residual = *trueRhs_nobnd_lvls[lvl];
    residual -= Asol;
    residual -= bdr_corr;
}

void TimeCylHyper::ComputeError(int lvl, Vector& sol) const
{
    BlockVector sol_viewer(sol.GetData(), *block_trueOffsets_lvls[lvl]);

    ParMeshCyl * pmeshtsl = pmeshtsl_lvls[lvl];

    ParFiniteElementSpace * S_space = S_space_lvls[lvl];
    ParFiniteElementSpace * Sigma_space = Sigma_space_lvls[lvl];
    ParFiniteElementSpace * L2_space = L2_space_lvls[lvl];

    Transport_test Mytest(dim, numsol);

    ParGridFunction *S_exact = new ParGridFunction(S_space);
    S_exact->ProjectCoefficient(*(Mytest.scalarS));

    ParGridFunction * sigma_exact = new ParGridFunction(Sigma_space);
    sigma_exact->ProjectCoefficient(*(Mytest.sigma));

    ParGridFunction * sigma = new ParGridFunction(Sigma_space);
    sigma->Distribute(&(sol_viewer.GetBlock(0)));

    ParGridFunction * S = new ParGridFunction(S_space);
    if (strcmp(space_for_S,"H1") == 0) // S is present
        S->Distribute(&(sol_viewer.GetBlock(1)));
    else // no S in the formulation
    {
        ParBilinearForm *Cblock(new ParBilinearForm(S_space));
        Cblock->AddDomainIntegrator(new MassIntegrator(*(Mytest.bTb)));
        Cblock->Assemble();
        Cblock->Finalize();
        HypreParMatrix * C = Cblock->ParallelAssemble();

        ParMixedBilinearForm *Bblock(new ParMixedBilinearForm(Sigma_space, S_space));
        Bblock->AddDomainIntegrator(new VectorFEMassIntegrator(*(Mytest.b)));
        Bblock->Assemble();
        Bblock->Finalize();
        HypreParMatrix * B = Bblock->ParallelAssemble();
        Vector bTsigma(C->Height());
        B->Mult(sol_viewer.GetBlock(0),bTsigma);

        Vector trueS(C->Height());

        CG(*C, bTsigma, trueS, 0, 5000, 1e-9, 1e-12);
        S->Distribute(trueS);
    }

    // 13. Extract the parallel grid function corresponding to the finite element
    //     approximation X. This is the local solution on each processor. Compute
    //     L2 error norms.

    int order_quad = max(2, 2*feorder+1);
    const IntegrationRule *irs[Geometry::NumGeom];
    for (int i=0; i < Geometry::NumGeom; ++i)
    {
       irs[i] = &(IntRules.Get(i, order_quad));
    }


    double err_sigma = sigma->ComputeL2Error(*(Mytest.sigma), irs);
    double norm_sigma = ComputeGlobalLpNorm(2, *(Mytest.sigma), *pmeshtsl, irs);
    if (verbose)
        cout << "|| sigma - sigma_ex || / || sigma_ex || = " << err_sigma / norm_sigma << endl;

    DiscreteLinearOperator Div(Sigma_space, L2_space);
    Div.AddDomainInterpolator(new DivergenceInterpolator());
    ParGridFunction DivSigma(L2_space);
    Div.Assemble();
    Div.Mult(*sigma, DivSigma);

    double err_div = DivSigma.ComputeL2Error(*(Mytest.scalardivsigma),irs);
    double norm_div = ComputeGlobalLpNorm(2, *(Mytest.scalardivsigma), *pmeshtsl, irs);

    if (verbose)
    {
        cout << "|| div (sigma_h - sigma_ex) || / ||div (sigma_ex)|| = "
                  << err_div/norm_div  << "\n";
    }

    if (verbose)
    {
        cout << "Actually it will be ~ continuous L2 + discrete L2 for divergence" << endl;
        cout << "|| sigma_h - sigma_ex ||_Hdiv / || sigma_ex ||_Hdiv = "
                  << sqrt(err_sigma*err_sigma + err_div * err_div)/sqrt(norm_sigma*norm_sigma + norm_div * norm_div)  << "\n";
    }

    // Computing error for S

    double err_S = S->ComputeL2Error((*Mytest.scalarS), irs);
    double norm_S = ComputeGlobalLpNorm(2, (*Mytest.scalarS), *pmeshtsl, irs);
    if (verbose)
    {
        std::cout << "|| S_h - S_ex || / || S_ex || = " <<
                     err_S / norm_S << "\n";
    }

    if (strcmp(space_for_S,"H1") == 0) // S is from H1
    {
        FiniteElementCollection * hcurl_coll;
        if(dim==4)
            hcurl_coll = new ND1_4DFECollection;
        else
            hcurl_coll = new ND_FECollection(feorder+1, dim);
        ParFiniteElementSpace* N_space = new ParFiniteElementSpace(pmeshtsl, hcurl_coll);

        DiscreteLinearOperator Grad(S_space, N_space);
        Grad.AddDomainInterpolator(new GradientInterpolator());
        ParGridFunction GradS(N_space);
        Grad.Assemble();
        Grad.Mult(*S, GradS);

        if (numsol != -34 && verbose)
            std::cout << "For this norm we are grad S for S from numsol = -34 \n";
        VectorFunctionCoefficient GradS_coeff(dim, uFunTest_ex_gradxt);
        double err_GradS = GradS.ComputeL2Error(GradS_coeff, irs);
        double norm_GradS = ComputeGlobalLpNorm(2, GradS_coeff, *pmeshtsl, irs);
        if (verbose)
        {
            std::cout << "|| Grad_h (S_h - S_ex) || / || Grad S_ex || = " <<
                         err_GradS / norm_GradS << "\n";
            std::cout << "|| S_h - S_ex ||_H^1 / || S_ex ||_H^1 = " <<
                         sqrt(err_S*err_S + err_GradS*err_GradS) / sqrt(norm_S*norm_S + norm_GradS*norm_GradS) << "\n";
        }

        delete hcurl_coll;
        delete N_space;
    }

    if (verbose)
        cout << "Computing projection errors \n";

    double projection_error_sigma = sigma_exact->ComputeL2Error(*(Mytest.sigma), irs);

    if(verbose)
    {
        cout << "|| sigma_ex - Pi_h sigma_ex || / || sigma_ex || = "
                        << projection_error_sigma / norm_sigma << endl;
    }

    double projection_error_S = S_exact->ComputeL2Error(*(Mytest.scalarS), irs);

    if(verbose)
        cout << "|| S_ex - Pi_h S_ex || / || S_ex || = "
                        << projection_error_S / norm_S << endl;

    if (visualization && dim < 4)
    {
        int num_procs, myid;
        MPI_Comm_size(comm, &num_procs);
        MPI_Comm_rank(comm, &myid);


       char vishost[] = "localhost";
       int  visport   = 19916;
       socketstream u_sock(vishost, visport);
       u_sock << "parallel " << num_procs << " " << myid << "\n";
       u_sock.precision(8);
       u_sock << "solution\n" << *pmeshtsl << *sigma_exact << "window_title 'sigma_exact'"
              << endl;
       // Make sure all ranks have sent their 'u' solution before initiating
       // another set of GLVis connections (one from each rank):


       socketstream uu_sock(vishost, visport);
       uu_sock << "parallel " << num_procs << " " << myid << "\n";
       uu_sock.precision(8);
       uu_sock << "solution\n" << *pmeshtsl << *sigma << "window_title 'sigma'"
              << endl;

       *sigma_exact -= *sigma;

       socketstream uuu_sock(vishost, visport);
       uuu_sock << "parallel " << num_procs << " " << myid << "\n";
       uuu_sock.precision(8);
       uuu_sock << "solution\n" << *pmeshtsl << *sigma_exact << "window_title 'difference for sigma'"
              << endl;

       socketstream s_sock(vishost, visport);
       s_sock << "parallel " << num_procs << " " << myid << "\n";
       s_sock.precision(8);
       MPI_Barrier(comm);
       s_sock << "solution\n" << *pmeshtsl << *S_exact << "window_title 'S_exact'"
               << endl;

       socketstream ss_sock(vishost, visport);
       ss_sock << "parallel " << num_procs << " " << myid << "\n";
       ss_sock.precision(8);
       MPI_Barrier(comm);
       ss_sock << "solution\n" << *pmeshtsl << *S << "window_title 'S'"
               << endl;

       *S_exact -= *S;
       socketstream sss_sock(vishost, visport);
       sss_sock << "parallel " << num_procs << " " << myid << "\n";
       sss_sock.precision(8);
       MPI_Barrier(comm);
       sss_sock << "solution\n" << *pmeshtsl << *S_exact
                << "window_title 'difference for S'" << endl;

       MPI_Barrier(comm);
    }

    if (verbose)
        std::cout << "\n";
}


TimeCylHyper::TimeCylHyper (ParMesh& Pmeshbase, double T_init, double Tau, int Nt, int Ref_lvls,
                              const char *Formulation, const char *Space_for_S, const char *Space_for_sigma)
    : TimeCyl(Pmeshbase, T_init, Tau, Nt), ref_lvls(Ref_lvls),
      formulation(Formulation), space_for_S(Space_for_S), space_for_sigma(Space_for_sigma)
{
    InitProblem();
}

TimeCylHyper::TimeCylHyper (ParMeshCyl& Pmeshtsl, int Ref_Lvls,
                              const char *Formulation, const char *Space_for_S, const char *Space_for_sigma)
    : TimeCyl(Pmeshtsl), ref_lvls(Ref_Lvls),
      formulation(Formulation), space_for_S(Space_for_S), space_for_sigma(Space_for_sigma)
{
    InitProblem();
}

void TimeCylHyper::Solve(int lvl, const Vector& rhs, Vector& sol,
                         const Vector& bnd_tdofs_bot, Vector& bnd_tdofs_top) const
{
    return Solve("regular", lvl, rhs, sol, bnd_tdofs_bot, bnd_tdofs_top);
}

// mode options:
// a) "regular" to solve with a matrix assembled from bilinear forms at corr. level
// b) "coarsened" to solve with RAP-based matrix
void TimeCylHyper::Solve(const char * mode, int lvl, const Vector& rhs, Vector& sol,
                         const Vector& bnd_tdofs_bot, Vector& bnd_tdofs_top) const
{
    if (!(lvl >= 0 && lvl <= ref_lvls))
    {
        MFEM_ABORT("Incorrect lvl argument for TimeCylHyper::Solve() \n");
    }

    if (strcmp(mode,"regular") != 0 && strcmp(mode,"coarsened") != 0 )
    {
        MFEM_ABORT("Incorrect mode for TimeCylHyper::Solve() \n");
    }

    if (strcmp(mode,"coarsened") == 0)
    {
        MFEM_ABORT("Mode coarsened was not implemented yet \n");
    }

    int init_cond_size = init_cond_size_lvls[lvl];

    if (bnd_tdofs_bot.Size() != init_cond_size || bnd_tdofs_top.Size() != init_cond_size)
    {
        std::cerr << "Error: sizes mismatch, input vector's size = " <<  bnd_tdofs_bot.Size()
                  << ", output's size = " << bnd_tdofs_top.Size() << ", expected: " << init_cond_size << "\n";
        MFEM_ABORT("Wrong size of the input and output vectors");
    }

    BlockOperator* CFOSLSop = CFOSLSop_lvls[lvl];
    BlockOperator* CFOSLSop_nobnd = CFOSLSop_nobnd_lvls[lvl];
    ParFiniteElementSpace * S_space = S_space_lvls[lvl];
    ParFiniteElementSpace * Sigma_space = Sigma_space_lvls[lvl];
    ParFiniteElementSpace * L2_space = L2_space_lvls[lvl];
    MINRESSolver * solver = solver_lvls[lvl];
    ParMeshCyl * pmeshtsl = pmeshtsl_lvls[lvl];
    Array<int> block_trueOffsets(block_trueOffsets_lvls[lvl]->Size());
    for (int i = 0; i < block_trueOffsets.Size(); ++i)
        block_trueOffsets[i] = (*block_trueOffsets_lvls[lvl])[i];

    std::vector<std::pair<int,int> > tdofs_link_H1;
    std::vector<std::pair<int,int> > tdofs_link_Hdiv;
    if (strcmp(space_for_S, "H1") == 0)
        tdofs_link_H1 = tdofs_link_H1_lvls[lvl];
    else
        tdofs_link_Hdiv = tdofs_link_Hdiv_lvls[lvl];

    Array<int> ess_bdrS(pmeshtsl->bdr_attributes.Max());
    for (unsigned int i = 0; i < ess_bdrat_S.size(); ++i)
        ess_bdrS[i] = ess_bdrat_S[i];

    Array<int> ess_bdrSigma(pmeshtsl->bdr_attributes.Max());
    for (unsigned int i = 0; i < ess_bdrat_sigma.size(); ++i)
        ess_bdrSigma[i] = ess_bdrat_sigma[i];

    int numblocks = CFOSLSop->NumRowBlocks();
    BlockVector trueX(block_trueOffsets);
    trueX = 0.0;

    // using an alternative way of imposing boundary conditions on the right hand side
    BlockVector trueBnd(block_trueOffsets);
    trueBnd = 0.0;

    if (strcmp(space_for_S, "H1") == 0)
    {
        for (unsigned int i = 0; i < tdofs_link_H1.size(); ++i)
        {
            int tdof_bot = tdofs_link_H1[i].first;
            trueBnd.GetBlock(1)[tdof_bot] = bnd_tdofs_bot[i];
        }
    }
    else // S is from l2
    {
        for (unsigned int i = 0; i < tdofs_link_Hdiv.size(); ++i)
        {
            int tdof_bot = tdofs_link_Hdiv[i].first;
            trueBnd.GetBlock(0)[tdof_bot] = bnd_tdofs_bot[i];
        }
    }

    BlockVector trueBndCor(block_trueOffsets);
    trueBndCor = 0.0;

    //trueBnd.Print();

    CFOSLSop_nobnd->Mult(trueBnd, trueBndCor); // more general that lines below

    BlockVector trueRhs_nobnd(block_trueOffsets);
    BlockVector rhs_viewer(rhs.GetData(), block_trueOffsets);
    trueRhs_nobnd = rhs_viewer;

    *trueRhs_nobnd_lvls[lvl] = trueRhs_nobnd;

    BlockVector trueRhs2(block_trueOffsets);
    trueRhs2 = trueRhs_nobnd;

    //trueRhs2.Print();

    trueRhs2 -= trueBndCor;

    // TODO: this is just for faster integration.
    // TODO: After checks this can be everywhere replaced by trueRhs2
    //trueRhs = trueRhs2;

    {
        Array<int> EssBnd_tdofs_sigma;
        Sigma_space->GetEssentialTrueDofs(ess_bdrSigma, EssBnd_tdofs_sigma);

        for (int i = 0; i < EssBnd_tdofs_sigma.Size(); ++i)
        {
            int tdof = EssBnd_tdofs_sigma[i];
            trueRhs2.GetBlock(0)[tdof] = trueBnd.GetBlock(0)[tdof];
        }

        if (strcmp(space_for_S,"H1") == 0) // S is present
        {
            Array<int> EssBnd_tdofs_S;
            S_space->GetEssentialTrueDofs(ess_bdrS, EssBnd_tdofs_S);

            for (int i = 0; i < EssBnd_tdofs_S.Size(); ++i)
            {
                int tdof = EssBnd_tdofs_S[i];
                trueRhs2.GetBlock(1)[tdof] = trueBnd.GetBlock(1)[tdof];
            }
        }
    }


    trueX = 0.0;

    StopWatch chrono;
    chrono.Clear();
    chrono.Start();

    solver->Mult(trueRhs2, trueX);
    chrono.Stop();

    if (strcmp(space_for_S, "H1") == 0)
    {
        for (unsigned int i = 0; i < tdofs_link_H1.size(); ++i)
        {
            int tdof_top = tdofs_link_H1[i].second;
            bnd_tdofs_top[i] = trueX.GetBlock(1)[tdof_top];
        }
    }
    else // S is from l2
    {
        for (unsigned int i = 0; i < tdofs_link_Hdiv.size(); ++i)
        {
            int tdof_top = tdofs_link_Hdiv[i].second;
            bnd_tdofs_top[i] = trueX.GetBlock(0)[tdof_top];
        }
    }

    BlockVector viewer_out(sol.GetData(), block_trueOffsets);
    viewer_out = *trueX;
}

void TimeCylHyper::Solve(int lvl, const Vector& bnd_tdofs_bot, Vector& bnd_tdofs_top) const
{
    if (!(lvl >= 0 && lvl <= ref_lvls))
    {
        MFEM_ABORT("Incorrect lvl argument for TimeCylHyper::Solve() \n");
    }

    int init_cond_size = init_cond_size_lvls[lvl];

    if (bnd_tdofs_bot.Size() != init_cond_size || bnd_tdofs_top.Size() != init_cond_size)
    {
        std::cerr << "Error: sizes mismatch, input vector's size = " <<  bnd_tdofs_bot.Size()
                  << ", output's size = " << bnd_tdofs_top.Size() << ", expected: " << init_cond_size << "\n";
        MFEM_ABORT("Wrong size of the input and output vectors");
    }

    BlockOperator* CFOSLSop = CFOSLSop_lvls[lvl];
    BlockOperator* CFOSLSop_nobnd = CFOSLSop_nobnd_lvls[lvl];
    ParFiniteElementSpace * S_space = S_space_lvls[lvl];
    ParFiniteElementSpace * Sigma_space = Sigma_space_lvls[lvl];
    ParFiniteElementSpace * L2_space = L2_space_lvls[lvl];
    MINRESSolver * solver = solver_lvls[lvl];
    ParMeshCyl * pmeshtsl = pmeshtsl_lvls[lvl];
    Array<int> block_trueOffsets(block_trueOffsets_lvls[lvl]->Size());
    for (int i = 0; i < block_trueOffsets.Size(); ++i)
        block_trueOffsets[i] = (*block_trueOffsets_lvls[lvl])[i];

    std::vector<std::pair<int,int> > tdofs_link_H1;
    std::vector<std::pair<int,int> > tdofs_link_Hdiv;
    if (strcmp(space_for_S, "H1") == 0)
        tdofs_link_H1 = tdofs_link_H1_lvls[lvl];
    else
        tdofs_link_Hdiv = tdofs_link_Hdiv_lvls[lvl];

    Array<int> ess_bdrS(pmeshtsl->bdr_attributes.Max());
    for (unsigned int i = 0; i < ess_bdrat_S.size(); ++i)
        ess_bdrS[i] = ess_bdrat_S[i];

    Array<int> ess_bdrSigma(pmeshtsl->bdr_attributes.Max());
    for (unsigned int i = 0; i < ess_bdrat_sigma.size(); ++i)
        ess_bdrSigma[i] = ess_bdrat_sigma[i];


    int numblocks = CFOSLSop->NumRowBlocks();
    //Array<int> block_trueOffsets(numblocks + 1);
    //for (int i = 0; i < block_trueOffsets.Size(); ++i)
        //block_trueOffsets[i] = block_trueoffsets[i];
    //block_trueOffsets.Print();
    //BlockVector trueX(block_trueOffsets);
    //BlockVector trueRhs(block_trueOffsets);
    //trueX = 0.0;
    *trueX_lvls[lvl] = 0.0;

    Transport_test Mytest(dim, numsol);

    ParGridFunction *S_exact = new ParGridFunction(S_space);
    S_exact->ProjectCoefficient(*(Mytest.scalarS));

    ParGridFunction * sigma_exact = new ParGridFunction(Sigma_space);
    sigma_exact->ProjectCoefficient(*(Mytest.sigma));

    // using an alternative way of imposing boundary conditions on the right hand side
    BlockVector trueBnd(block_trueOffsets);
    trueBnd = 0.0;

    if (strcmp(space_for_S, "H1") == 0)
    {
        for (unsigned int i = 0; i < tdofs_link_H1.size(); ++i)
        {
            int tdof_bot = tdofs_link_H1[i].first;
            trueBnd.GetBlock(1)[tdof_bot] = bnd_tdofs_bot[i];
        }
    }
    else // S is from l2
    {
        for (unsigned int i = 0; i < tdofs_link_Hdiv.size(); ++i)
        {
            int tdof_bot = tdofs_link_Hdiv[i].first;
            trueBnd.GetBlock(0)[tdof_bot] = bnd_tdofs_bot[i];
        }
    }


    /*
    BlockVector viewer(bnd_tdofs_bot.GetData(), block_trueOffsets);
    BlockVector trueBnd(block_trueOffsets);
    trueBnd = 0.0;
    {
        Array<int> EssBnd_tdofs_sigma;
        Sigma_space->GetEssentialTrueDofs(ess_bdrSigma, EssBnd_tdofs_sigma);

        for (int i = 0; i < EssBnd_tdofs_sigma.Size(); ++i)
        {
            int tdof = EssBnd_tdofs_sigma[i];
            trueBnd.GetBlock(0)[tdof] = viewer.GetBlock(0)[tdof];
        }

        if (strcmp(space_for_S,"H1") == 0) // S is present
        {
            Array<int> EssBnd_tdofs_S;
            S_space->GetEssentialTrueDofs(ess_bdrS, EssBnd_tdofs_S);

            for (int i = 0; i < EssBnd_tdofs_S.Size(); ++i)
            {
                int tdof = EssBnd_tdofs_S[i];
                trueBnd.GetBlock(1)[tdof] = viewer.GetBlock(1)[tdof];
            }
        }
    }
    */

    BlockVector trueBndCor(block_trueOffsets);
    trueBndCor = 0.0;

    //trueBnd.Print();

    CFOSLSop_nobnd->Mult(trueBnd, trueBndCor); // more general that lines below

    ParLinearForm *fform_nobnd = new ParLinearForm(Sigma_space);
    fform_nobnd->Assemble();

    ParLinearForm *qform_nobnd;
    if (strcmp(space_for_S,"H1") == 0)
    {
        qform_nobnd = new ParLinearForm(S_space);
    }

    if (strcmp(space_for_S,"H1") == 0)
    {
        //if (strcmp(space_for_sigma,"Hdiv") == 0 )
            qform_nobnd->AddDomainIntegrator(new GradDomainLFIntegrator(*Mytest.bf));
        qform_nobnd->Assemble();//qform->Print();
    }

    ParLinearForm *gform_nobnd;
    if (strcmp(formulation,"cfosls") == 0)
    {
        gform_nobnd = new ParLinearForm(L2_space);
        gform_nobnd->AddDomainIntegrator(new DomainLFIntegrator(*Mytest.scalardivsigma));
        gform_nobnd->Assemble();
    }

    BlockVector trueRhs_nobnd(block_trueOffsets);
    trueRhs_nobnd = 0.0;

    if (strcmp(space_for_S,"H1") == 0)
        qform_nobnd->ParallelAssemble(trueRhs_nobnd.GetBlock(1));
    gform_nobnd->ParallelAssemble(trueRhs_nobnd.GetBlock(numblocks - 1));

    *trueRhs_nobnd_lvls[lvl] = trueRhs_nobnd;

    BlockVector trueRhs2(block_trueOffsets);
    trueRhs2 = trueRhs_nobnd;

    //trueRhs2.Print();

    trueRhs2 -= trueBndCor;

    // TODO: this is just for faster integration.
    // TODO: After checks this can be everywhere replaced by trueRhs2
    //trueRhs = trueRhs2;

    {
        Array<int> EssBnd_tdofs_sigma;
        Sigma_space->GetEssentialTrueDofs(ess_bdrSigma, EssBnd_tdofs_sigma);

        for (int i = 0; i < EssBnd_tdofs_sigma.Size(); ++i)
        {
            int tdof = EssBnd_tdofs_sigma[i];
            trueRhs2.GetBlock(0)[tdof] = trueBnd.GetBlock(0)[tdof];
        }

        if (strcmp(space_for_S,"H1") == 0) // S is present
        {
            Array<int> EssBnd_tdofs_S;
            S_space->GetEssentialTrueDofs(ess_bdrS, EssBnd_tdofs_S);

            for (int i = 0; i < EssBnd_tdofs_S.Size(); ++i)
            {
                int tdof = EssBnd_tdofs_S[i];
                trueRhs2.GetBlock(1)[tdof] = trueBnd.GetBlock(1)[tdof];
            }
        }
    }

    *trueX_lvls[lvl] = 0.0;

    StopWatch chrono;
    chrono.Clear();
    chrono.Start();

    //trueRhs2.Print();
    //SparseMatrix diag;
    //((HypreParMatrix&)(CFOSLSop->GetBlock(0,0))).GetDiag(diag);
    //diag.Print();

    solver->Mult(trueRhs2, *trueX_lvls[lvl]);
    chrono.Stop();

    if (strcmp(space_for_S, "H1") == 0)
    {
        for (unsigned int i = 0; i < tdofs_link_H1.size(); ++i)
        {
            int tdof_top = tdofs_link_H1[i].second;
            bnd_tdofs_top[i] = trueX_lvls[lvl]->GetBlock(1)[tdof_top];
        }
    }
    else // S is from l2
    {
        for (unsigned int i = 0; i < tdofs_link_Hdiv.size(); ++i)
        {
            int tdof_top = tdofs_link_Hdiv[i].second;
            bnd_tdofs_top[i] = trueX_lvls[lvl]->GetBlock(0)[tdof_top];
        }
    }

    /*
    BlockVector trueOut(bnd_tdofs_top.GetData(), block_trueOffsets);
    trueOut = 0.0;
    {
        Array<int> EssBnd_tdofs_sigma;
        Sigma_space->GetEssentialTrueDofs(ess_bdrSigma, EssBnd_tdofs_sigma);

        for (int i = 0; i < EssBnd_tdofs_sigma.Size(); ++i)
        {
            int tdof = EssBnd_tdofs_sigma[i];
            trueOut.GetBlock(0)[tdof] = trueX.GetBlock(0)[tdof];
        }

        if (strcmp(space_for_S,"H1") == 0) // S is present
        {
            Array<int> EssBnd_tdofs_S;
            S_space->GetEssentialTrueDofs(ess_bdrS, EssBnd_tdofs_S);

            for (int i = 0; i < EssBnd_tdofs_S.Size(); ++i)
            {
                int tdof = EssBnd_tdofs_S[i];
                trueOut.GetBlock(1)[tdof] = trueX.GetBlock(1)[tdof];
            }
        }
    }
    */

    /*
    {
        for ( unsigned int i = 0; i < tdofs_link_Hdiv.size(); ++i)
        {
            int tdof_bot = tdofs_link_Hdiv[i].first;
            int tdof_top = tdofs_link_Hdiv[i].second;
            trueOut.GetBlock(0)[tdof_bot] = trueX.GetBlock(0)[tdof_top];
        }

        if (strcmp(space_for_S,"H1") == 0) // S is present
        {
            for ( unsigned int i = 0; i < tdofs_link_H1.size(); ++i)
            {
                int tdof_bot = tdofs_link_H1[i].first;
                int tdof_top = tdofs_link_H1[i].second;
                trueOut.GetBlock(1)[tdof_bot] = trueX.GetBlock(1)[tdof_top];
            }
        }

    }
    */

    if (verbose)
    {
       if (solver->GetConverged())
          std::cout << "MINRES converged in " << solver->GetNumIterations()
                    << " iterations with a residual norm of " << solver->GetFinalNorm() << ".\n";
       else
          std::cout << "MINRES did not converge in " << solver->GetNumIterations()
                    << " iterations. Residual norm is " << solver->GetFinalNorm() << ".\n";
       std::cout << "MINRES solver took " << chrono.RealTime() << "s. \n";
    }


    // checking boundary conditions
    /*
    Vector sigma_exact_truedofs(Sigma_space->TrueVSize());
    sigma_exact->ParallelAssemble(sigma_exact_truedofs);

    Array<int> EssBnd_tdofs_sigma;
    Sigma_space->GetEssentialTrueDofs(ess_bdrSigma, EssBnd_tdofs_sigma);

    for (int i = 0; i < EssBnd_tdofs_sigma.Size(); ++i)
    {
        int tdof = EssBnd_tdofs_sigma[i];
        double value_ex = sigma_exact_truedofs[tdof];
        double value_com = trueX.GetBlock(0)[tdof];

        if (fabs(value_ex - value_com) > ZEROTOL)
        {
            std::cout << "bnd condition is violated for sigma, tdof = " << tdof << " exact value = "
                      << value_ex << ", value_com = " << value_com << ", diff = " << value_ex - value_com << "\n";
            std::cout << "rhs side at this tdof = " << trueRhs2.GetBlock(0)[tdof] << "\n";
        }
    }

    if (strcmp(space_for_S,"H1") == 0) // S is present
    {
        Vector S_exact_truedofs(S_space->TrueVSize());
        S_exact->ParallelAssemble(S_exact_truedofs);

        Array<int> EssBnd_tdofs_S;
        S_space->GetEssentialTrueDofs(ess_bdrS, EssBnd_tdofs_S);

        for (int i = 0; i < EssBnd_tdofs_S.Size(); ++i)
        {
            int tdof = EssBnd_tdofs_S[i];
            double value_ex = S_exact_truedofs[tdof];
            double value_com = trueX.GetBlock(1)[tdof];

            if (fabs(value_ex - value_com) > ZEROTOL)
            {
                std::cout << "bnd condition is violated for S, tdof = " << tdof << " exact value = "
                          << value_ex << ", value_com = " << value_com << ", diff = " << value_ex - value_com << "\n";
                std::cout << "rhs side at this tdof = " << trueRhs2.GetBlock(1)[tdof] << "\n";
            }
        }
    }
    */

    //ComputeError(lvl, *trueX_lvls[lvl]);

#if 0
    ParGridFunction * sigma = new ParGridFunction(Sigma_space);
    sigma->Distribute(&(trueX.GetBlock(0)));

    ParGridFunction * S = new ParGridFunction(S_space);
    if (strcmp(space_for_S,"H1") == 0) // S is present
        S->Distribute(&(trueX.GetBlock(1)));
    else // no S in the formulation
    {
        ParBilinearForm *Cblock(new ParBilinearForm(S_space));
        Cblock->AddDomainIntegrator(new MassIntegrator(*(Mytest.bTb)));
        Cblock->Assemble();
        Cblock->Finalize();
        HypreParMatrix * C = Cblock->ParallelAssemble();

        ParMixedBilinearForm *Bblock(new ParMixedBilinearForm(Sigma_space, S_space));
        Bblock->AddDomainIntegrator(new VectorFEMassIntegrator(*(Mytest.b)));
        Bblock->Assemble();
        Bblock->Finalize();
        HypreParMatrix * B = Bblock->ParallelAssemble();
        Vector bTsigma(C->Height());
        B->Mult(trueX.GetBlock(0),bTsigma);

        Vector trueS(C->Height());

        CG(*C, bTsigma, trueS, 0, 5000, 1e-9, 1e-12);
        S->Distribute(trueS);
    }

    // 13. Extract the parallel grid function corresponding to the finite element
    //     approximation X. This is the local solution on each processor. Compute
    //     L2 error norms.

    int order_quad = max(2, 2*feorder+1);
    const IntegrationRule *irs[Geometry::NumGeom];
    for (int i=0; i < Geometry::NumGeom; ++i)
    {
       irs[i] = &(IntRules.Get(i, order_quad));
    }


    double err_sigma = sigma->ComputeL2Error(*(Mytest.sigma), irs);
    double norm_sigma = ComputeGlobalLpNorm(2, *(Mytest.sigma), *pmeshtsl, irs);
    if (verbose)
        cout << "|| sigma - sigma_ex || / || sigma_ex || = " << err_sigma / norm_sigma << endl;

    DiscreteLinearOperator Div(Sigma_space, L2_space);
    Div.AddDomainInterpolator(new DivergenceInterpolator());
    ParGridFunction DivSigma(L2_space);
    Div.Assemble();
    Div.Mult(*sigma, DivSigma);

    double err_div = DivSigma.ComputeL2Error(*(Mytest.scalardivsigma),irs);
    double norm_div = ComputeGlobalLpNorm(2, *(Mytest.scalardivsigma), *pmeshtsl, irs);

    if (verbose)
    {
        cout << "|| div (sigma_h - sigma_ex) || / ||div (sigma_ex)|| = "
                  << err_div/norm_div  << "\n";
    }

    if (verbose)
    {
        cout << "Actually it will be ~ continuous L2 + discrete L2 for divergence" << endl;
        cout << "|| sigma_h - sigma_ex ||_Hdiv / || sigma_ex ||_Hdiv = "
                  << sqrt(err_sigma*err_sigma + err_div * err_div)/sqrt(norm_sigma*norm_sigma + norm_div * norm_div)  << "\n";
    }

    // Computing error for S

    double err_S = S->ComputeL2Error((*Mytest.scalarS), irs);
    double norm_S = ComputeGlobalLpNorm(2, (*Mytest.scalarS), *pmeshtsl, irs);
    if (verbose)
    {
        std::cout << "|| S_h - S_ex || / || S_ex || = " <<
                     err_S / norm_S << "\n";
    }

    if (strcmp(space_for_S,"H1") == 0) // S is from H1
    {
        FiniteElementCollection * hcurl_coll;
        if(dim==4)
            hcurl_coll = new ND1_4DFECollection;
        else
            hcurl_coll = new ND_FECollection(feorder+1, dim);
        ParFiniteElementSpace* N_space = new ParFiniteElementSpace(pmeshtsl, hcurl_coll);

        DiscreteLinearOperator Grad(S_space, N_space);
        Grad.AddDomainInterpolator(new GradientInterpolator());
        ParGridFunction GradS(N_space);
        Grad.Assemble();
        Grad.Mult(*S, GradS);

        if (numsol != -34 && verbose)
            std::cout << "For this norm we are grad S for S from numsol = -34 \n";
        VectorFunctionCoefficient GradS_coeff(dim, uFunTest_ex_gradxt);
        double err_GradS = GradS.ComputeL2Error(GradS_coeff, irs);
        double norm_GradS = ComputeGlobalLpNorm(2, GradS_coeff, *pmeshtsl, irs);
        if (verbose)
        {
            std::cout << "|| Grad_h (S_h - S_ex) || / || Grad S_ex || = " <<
                         err_GradS / norm_GradS << "\n";
            std::cout << "|| S_h - S_ex ||_H^1 / || S_ex ||_H^1 = " <<
                         sqrt(err_S*err_S + err_GradS*err_GradS) / sqrt(norm_S*norm_S + norm_GradS*norm_GradS) << "\n";
        }

        delete hcurl_coll;
        delete N_space;
    }

#if 0
    // Check value of functional and mass conservation
    if (strcmp(formulation,"cfosls") == 0) // if CFOSLS, otherwise code requires some changes
    {
        Array<int> EssBnd_tdofs_sigma;
        Sigma_space->GetEssentialTrueDofs(ess_bdrSigma, EssBnd_tdofs_sigma);
        Array<int> EssBnd_tdofs_S;
        if (strcmp(space_for_S,"H1") == 0) // S is present
            S_space->GetEssentialTrueDofs(ess_bdrS, EssBnd_tdofs_S);

        BlockVector trueX_nobnd(block_trueOffsets);
        trueX_nobnd = trueX;
        BlockVector trueRhs_nobnd(block_trueOffsets);
        trueRhs_nobnd = trueRhs2;

        for (int i = 0; i < EssBnd_tdofs_sigma.Size(); ++i)
        {
            int tdof = EssBnd_tdofs_sigma[i];

            trueX_nobnd.GetBlock(0)[tdof] = 0.0;
            trueRhs_nobnd.GetBlock(0)[tdof] = 0.0;
        }

        if (strcmp(space_for_S,"H1") == 0) // S is present
        {
            for (int i = 0; i < EssBnd_tdofs_S.Size(); ++i)
            {
                int tdof = EssBnd_tdofs_S[i];
                trueX_nobnd.GetBlock(1)[tdof] = 0.0;
                trueRhs_nobnd.GetBlock(1)[tdof] = 0.0;
            }
        }

        double localFunctional = 0.0;//-2.0*(trueX.GetBlock(0)*trueRhs.GetBlock(0));
        if (strcmp(space_for_S,"H1") == 0) // S is present
        {
             localFunctional += -2.0*(trueX_nobnd.GetBlock(1)*trueRhs_nobnd.GetBlock(1));
             double f_norm = sqrt(trueRhs_nobnd.GetBlock(numblocks - 1)*trueRhs_nobnd.GetBlock(numblocks - 1));
             localFunctional += f_norm * f_norm;;
        }

        //////////////////////////////////////
        /*
        ParBilinearForm *Cblock1, *Cblock2;
        HypreParMatrix *C1, *C2;
        if (strcmp(space_for_S,"H1") == 0)
        {
            Cblock1 = new ParBilinearForm(S_space);
            Cblock1->AddDomainIntegrator(new MassIntegrator(*Mytest.bTb));
            Cblock1->Assemble();
            Cblock1->EliminateEssentialBC(ess_bdrS, x.GetBlock(1), *qform);
            Cblock1->Finalize();
            C1 = Cblock1->ParallelAssemble();

            Cblock2 = new ParBilinearForm(S_space);
            if (strcmp(space_for_sigma,"Hdiv") == 0)
                 Cblock2->AddDomainIntegrator(new DiffusionIntegrator(*Mytest.bbT));
            Cblock2->Assemble();
            Cblock2->EliminateEssentialBC(ess_bdrS, x.GetBlock(1), *qform);
            Cblock2->Finalize();
            C2 = Cblock2->ParallelAssemble();
        }

        Vector C2S(S_space->TrueVSize());
        C2->Mult(trueX.GetBlock(1), C2S);

        double local_piece2 = C2S * trueX.GetBlock(1);
        */
        //////////////////////////////////////


        trueX.GetBlock(numblocks - 1) = 0.0;
        trueRhs_nobnd = 0.0;;
        CFOSLSop->Mult(trueX_nobnd, trueRhs_nobnd);
        localFunctional += trueX_nobnd*(trueRhs_nobnd);

        double globalFunctional;
        MPI_Reduce(&localFunctional, &globalFunctional, 1,
                   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (verbose)
        {
            if (strcmp(space_for_S,"H1") == 0) // S is present
            {
                cout << "|| sigma_h - L(S_h) ||^2 + || div_h (bS_h) - f ||^2 = " << globalFunctional+err_div*err_div << "\n";
                cout << "|| f ||^2 = " << norm_div*norm_div  << "\n";
                cout << "Smth is wrong with the functional computation for H1 case \n";
                cout << "Relative Energy Error = " << sqrt(globalFunctional+err_div*err_div)/norm_div << "\n";
            }
            else // if S is from L2
            {
                cout << "|| sigma_h - L(S_h) ||^2 + || div_h (sigma_h) - f ||^2 = " << globalFunctional+err_div*err_div << "\n";
                cout << "Energy Error = " << sqrt(globalFunctional+err_div*err_div) << "\n";
            }
        }

        ParLinearForm massform(L2_space);
        massform.AddDomainIntegrator(new DomainLFIntegrator(*(Mytest.scalardivsigma)));
        massform.Assemble();

        double mass_loc = massform.Norml1();
        double mass;
        MPI_Reduce(&mass_loc, &mass, 1,
                   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (verbose)
            cout << "Sum of local mass = " << mass<< "\n";

        trueRhs2.GetBlock(numblocks - 1) -= massform;
        double mass_loss_loc = trueRhs2.GetBlock(numblocks - 1).Norml1();
        double mass_loss;
        MPI_Reduce(&mass_loss_loc, &mass_loss, 1,
                   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (verbose)
            cout << "Sum of local mass loss = " << mass_loss << "\n";
    }
#endif

    if (verbose)
        cout << "Computing projection errors \n";

    double projection_error_sigma = sigma_exact->ComputeL2Error(*(Mytest.sigma), irs);

    if(verbose)
    {
        cout << "|| sigma_ex - Pi_h sigma_ex || / || sigma_ex || = "
                        << projection_error_sigma / norm_sigma << endl;
    }

    double projection_error_S = S_exact->ComputeL2Error(*(Mytest.scalarS), irs);

    if(verbose)
        cout << "|| S_ex - Pi_h S_ex || / || S_ex || = "
                        << projection_error_S / norm_S << endl;

    if (visualization && dim < 4)
    {
        int num_procs, myid;
        MPI_Comm_size(comm, &num_procs);
        MPI_Comm_rank(comm, &myid);


       char vishost[] = "localhost";
       int  visport   = 19916;
       socketstream u_sock(vishost, visport);
       u_sock << "parallel " << num_procs << " " << myid << "\n";
       u_sock.precision(8);
       u_sock << "solution\n" << *pmeshtsl << *sigma_exact << "window_title 'sigma_exact'"
              << endl;
       // Make sure all ranks have sent their 'u' solution before initiating
       // another set of GLVis connections (one from each rank):


       socketstream uu_sock(vishost, visport);
       uu_sock << "parallel " << num_procs << " " << myid << "\n";
       uu_sock.precision(8);
       uu_sock << "solution\n" << *pmeshtsl << *sigma << "window_title 'sigma'"
              << endl;

       *sigma_exact -= *sigma;

       socketstream uuu_sock(vishost, visport);
       uuu_sock << "parallel " << num_procs << " " << myid << "\n";
       uuu_sock.precision(8);
       uuu_sock << "solution\n" << *pmeshtsl << *sigma_exact << "window_title 'difference for sigma'"
              << endl;

       socketstream s_sock(vishost, visport);
       s_sock << "parallel " << num_procs << " " << myid << "\n";
       s_sock.precision(8);
       MPI_Barrier(comm);
       s_sock << "solution\n" << *pmeshtsl << *S_exact << "window_title 'S_exact'"
               << endl;

       socketstream ss_sock(vishost, visport);
       ss_sock << "parallel " << num_procs << " " << myid << "\n";
       ss_sock.precision(8);
       MPI_Barrier(comm);
       ss_sock << "solution\n" << *pmeshtsl << *S << "window_title 'S'"
               << endl;

       *S_exact -= *S;
       socketstream sss_sock(vishost, visport);
       sss_sock << "parallel " << num_procs << " " << myid << "\n";
       sss_sock.precision(8);
       MPI_Barrier(comm);
       sss_sock << "solution\n" << *pmeshtsl << *S_exact
                << "window_title 'difference for S'" << endl;

       MPI_Barrier(comm);
    }
#endif

}

void TimeCylHyper::InitProblem()
{
    dim = pmeshtsl->Dimension();
    comm = pmeshtsl->GetComm();

    // 1. Initialize MPI.
    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    feorder = 0;

#ifdef NONHOMO_TEST
   if (dim == 3)
       numsol = -33;
   else // 4D case
       numsol = -44;
#else
   if (dim == 3)
       numsol = -3;
   else // 4D case
       numsol = -4;
#endif

    int prec_option = 1; //defines whether to use preconditioner or not, and which one
    bool use_ADS;

    switch (prec_option)
    {
    case 1: // smth simple like AMS
        use_ADS = false;
        break;
    case 2: // MG
        use_ADS = true;
        break;
    default: // no preconditioner
        break;
    }

    int max_iter = 150000;
    double rtol = 1e-12;//1e-7;//1e-9;
    double atol = 1e-14;//1e-9;//1e-12;

    verbose = (myid == 0);

    visualization = 0;

    FiniteElementCollection *hdiv_coll;
    FiniteElementCollection *l2_coll;

    if (dim == 4)
        hdiv_coll = new RT0_4DFECollection;
    else
        hdiv_coll = new RT_FECollection(feorder, dim);

    l2_coll = new L2_FECollection(feorder, dim);

    FiniteElementCollection *h1_coll;
    if (dim == 3)
        h1_coll = new H1_FECollection(feorder + 1, dim);
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

    ParFiniteElementSpace *Hdiv_space;
    Hdiv_space = new ParFiniteElementSpace(pmeshtsl, hdiv_coll);

    ParFiniteElementSpace *L2_space;
    L2_space = new ParFiniteElementSpace(pmeshtsl, l2_coll);

    ParFiniteElementSpace *H1_space;
    H1_space = new ParFiniteElementSpace(pmeshtsl, h1_coll);

    /////////////////////////////////////////////////////////////////
    int num_lvls = ref_lvls + 1;

    pmeshtsl_lvls.resize(num_lvls);
    Hdiv_space_lvls.resize(num_lvls);
    H1_space_lvls.resize(num_lvls);
    L2_space_lvls.resize(num_lvls);
    Sigma_space_lvls.resize(num_lvls);
    S_space_lvls.resize(num_lvls);

    block_trueOffsets_lvls.resize(num_lvls);
    CFOSLSop_lvls.resize(num_lvls);
    CFOSLSop_coarsened_lvls.resize(num_lvls);
    trueRhs_nobnd_lvls.resize(num_lvls);
    trueX_lvls.resize(num_lvls);
    CFOSLSop_nobnd_lvls.resize(num_lvls);
    prec_lvls.resize(num_lvls);
    solver_lvls.resize(num_lvls);

    TrueP_lvls.resize(num_lvls - 1);
    TrueP_L2_lvls.resize(num_lvls - 1);
    TrueP_H1_lvls.resize(num_lvls - 1);
    TrueP_Hdiv_lvls.resize(num_lvls - 1);
    P_H1_lvls.resize(num_lvls - 1);
    P_Hdiv_lvls.resize(num_lvls - 1);
    P_L2_lvls.resize(num_lvls - 1);
    TrueP_bndbot_H1_lvls.resize(num_lvls - 1);
    TrueP_bndbot_Hdiv_lvls.resize(num_lvls - 1);
    TrueP_bndtop_H1_lvls.resize(num_lvls - 1);
    TrueP_bndtop_Hdiv_lvls.resize(num_lvls - 1);
    Restrict_bot_H1_lvls.resize(num_lvls);
    Restrict_bot_Hdiv_lvls.resize(num_lvls);
    Restrict_top_H1_lvls.resize(num_lvls);
    Restrict_top_Hdiv_lvls.resize(num_lvls);

    init_cond_size_lvls.resize(num_lvls);
    tdofs_link_H1_lvls.resize(num_lvls);
    tdofs_link_Hdiv_lvls.resize(num_lvls);

    const SparseMatrix* P_Hdiv_local;
    const SparseMatrix* P_H1_local;
    const SparseMatrix* P_L2_local;

    // 0 will correspond to the finest level for all items in the hierarchy

    for (int l = num_lvls - 1; l >= 0; --l)
    {
        // creating pmesh for level l
        if (l == num_lvls - 1)
        {
            pmeshtsl_lvls[l] = new ParMeshCyl(*pmeshtsl);
        }
        else
        {
            pmeshtsl->Refine(1);
            pmeshtsl_lvls[l] = new ParMeshCyl(*pmeshtsl);

            // be careful about the update of bot_to_top so that it doesn't get lost
        }

        // creating pfespaces for level l
        Hdiv_space_lvls[l] = new ParFiniteElementSpace(pmeshtsl_lvls[l], hdiv_coll);
        L2_space_lvls[l] = new ParFiniteElementSpace(pmeshtsl_lvls[l], l2_coll);
        H1_space_lvls[l] = new ParFiniteElementSpace(pmeshtsl_lvls[l], h1_coll);

        for (int i = 0; i < num_procs; ++i)
        {
            if (myid == i)
            {
                //std::cout << "I am " << myid << ", creating my tdof link \n";

                std::vector<std::pair<int,int> > * dofs_link_H1 =
                        CreateBotToTopDofsLink("linearH1",*H1_space_lvls[l], pmeshtsl_lvls[l]->bot_to_top_bels);
                std::cout << std::flush;

                tdofs_link_H1_lvls[l].reserve(dofs_link_H1->size());

                int count = 0;
                for ( unsigned int i = 0; i < dofs_link_H1->size(); ++i )
                {
                    //std::cout << "<" << it->first << ", " << it->second << "> \n";
                    int dof1 = (*dofs_link_H1)[i].first;
                    int dof2 = (*dofs_link_H1)[i].second;
                    int tdof1 = H1_space_lvls[l]->GetLocalTDofNumber(dof1);
                    int tdof2 = H1_space_lvls[l]->GetLocalTDofNumber(dof2);
                    //std::cout << "corr. dof pair: <" << dof1 << "," << dof2 << ">\n";
                    //std::cout << "corr. tdof pair: <" << tdof1 << "," << tdof2 << ">\n";
                    if (tdof1 * tdof2 < 0)
                        MFEM_ABORT( "unsupported case: tdof1 and tdof2 belong to different processors! \n");

                    if (tdof1 > -1)
                    {
                        tdofs_link_H1_lvls[l].push_back(std::pair<int,int>(tdof1, tdof2));
                        ++count;
                    }
                    else
                    {
                        //std::cout << "Ignored dofs pair which are not own tdofs \n";
                    }
                }
            }
            MPI_Barrier(comm);
        } // end fo loop over all processors, one after another

        for (int i = 0; i < num_procs; ++i)
        {
            if (myid == i)
            {
                std::vector<std::pair<int,int> > * dofs_link_RT0 =
                           CreateBotToTopDofsLink("RT0",*Hdiv_space_lvls[l], pmeshtsl_lvls[l]->bot_to_top_bels);
                std::cout << std::flush;

                tdofs_link_Hdiv_lvls[l].reserve(dofs_link_RT0->size());

                int count = 0;
                //std::cout << "dof pairs for Hdiv: \n";
                for ( unsigned int i = 0; i < dofs_link_RT0->size(); ++i)
                {
                    int dof1 = (*dofs_link_RT0)[i].first;
                    int dof2 = (*dofs_link_RT0)[i].second;
                    //std::cout << "<" << it->first << ", " << it->second << "> \n";
                    int tdof1 = Hdiv_space_lvls[l]->GetLocalTDofNumber(dof1);
                    int tdof2 = Hdiv_space_lvls[l]->GetLocalTDofNumber(dof2);
                    //std::cout << "corr. tdof pair: <" << tdof1 << "," << tdof2 << ">\n";
                    if ((tdof1 > 0 && tdof2 < 0) || (tdof1 < 0 && tdof2 > 0))
                    {
                        //std::cout << "Caught you! tdof1 = " << tdof1 << ", tdof2 = " << tdof2 << "\n";
                        MFEM_ABORT( "unsupported case: tdof1 and tdof2 belong to different processors! \n");
                    }

                    if (tdof1 > -1)
                    {
                        tdofs_link_Hdiv_lvls[l].push_back(std::pair<int,int>(tdof1, tdof2));
                        ++count;
                    }
                    else
                    {
                        //std::cout << "Ignored a dofs pair which are not own tdofs \n";
                    }
                }
            }
            MPI_Barrier(comm);
        } // end fo loop over all processors, one after another

        // creating restriction matrices from all tdofs to bot tdofs
        Restrict_bot_H1_lvls[l] = CreateRestriction("bot", *H1_space_lvls[l], tdofs_link_H1_lvls[l]);
        Restrict_bot_Hdiv_lvls[l] = CreateRestriction("bot", *Hdiv_space_lvls[l], tdofs_link_Hdiv_lvls[l]);
        Restrict_top_H1_lvls[l] = CreateRestriction("top", *H1_space_lvls[l], tdofs_link_H1_lvls[l]);
        Restrict_top_Hdiv_lvls[l] = CreateRestriction("top", *Hdiv_space_lvls[l], tdofs_link_Hdiv_lvls[l]);

        // for all but one levels we create projection matrices between levels
        // and projectors assembled on true dofs if MG preconditioner is used
        if (l < num_lvls - 1)
        {
            Hdiv_space->Update();
            H1_space->Update();
            L2_space->Update();

            // TODO: Rewrite these computations

            P_Hdiv_local = (SparseMatrix *)Hdiv_space->GetUpdateOperator();
            P_Hdiv_lvls[l] = RemoveZeroEntries(*P_Hdiv_local);

            auto d_td_coarse_Hdiv = Hdiv_space_lvls[l + 1]->Dof_TrueDof_Matrix();
            SparseMatrix * RP_Hdiv_local = Mult(*Hdiv_space_lvls[l]->GetRestrictionMatrix(), *P_Hdiv_lvls[l]);
            TrueP_Hdiv_lvls[l] = d_td_coarse_Hdiv->LeftDiagMult(
                        *RP_Hdiv_local, Hdiv_space_lvls[l]->GetTrueDofOffsets());
            TrueP_Hdiv_lvls[l]->CopyColStarts();
            TrueP_Hdiv_lvls[l]->CopyRowStarts();

            delete RP_Hdiv_local;


            P_H1_local = (SparseMatrix *)H1_space->GetUpdateOperator();
            P_H1_lvls[l] = RemoveZeroEntries(*P_H1_local);

            auto d_td_coarse_H1 = H1_space_lvls[l + 1]->Dof_TrueDof_Matrix();
            SparseMatrix * RP_H1_local = Mult(*H1_space_lvls[l]->GetRestrictionMatrix(), *P_H1_lvls[l]);
            TrueP_H1_lvls[l] = d_td_coarse_H1->LeftDiagMult(
                        *RP_H1_local, H1_space_lvls[l]->GetTrueDofOffsets());
            TrueP_H1_lvls[l]->CopyColStarts();
            TrueP_H1_lvls[l]->CopyRowStarts();

            delete RP_H1_local;

            P_L2_local = (SparseMatrix *)L2_space->GetUpdateOperator();
            P_L2_lvls[l] = RemoveZeroEntries(*P_L2_local);

            auto d_td_coarse_L2 = L2_space_lvls[l + 1]->Dof_TrueDof_Matrix();
            SparseMatrix * RP_L2_local = Mult(*L2_space_lvls[l]->GetRestrictionMatrix(), *P_L2_lvls[l]);
            TrueP_L2_lvls[l] = d_td_coarse_L2->LeftDiagMult(
                        *RP_L2_local, L2_space_lvls[l]->GetTrueDofOffsets());
            TrueP_L2_lvls[l]->CopyColStarts();
            TrueP_L2_lvls[l]->CopyRowStarts();

            delete RP_L2_local;

            TrueP_bndbot_H1_lvls[l] = RAP(Restrict_bot_H1_lvls[l], TrueP_H1_lvls[l], Restrict_bot_H1_lvls[l + 1]);
            TrueP_bndbot_H1_lvls[l]->CopyColStarts();
            TrueP_bndbot_H1_lvls[l]->CopyRowStarts();

            TrueP_bndtop_H1_lvls[l] = RAP(Restrict_top_H1_lvls[l], TrueP_H1_lvls[l], Restrict_top_H1_lvls[l + 1]);
            TrueP_bndtop_H1_lvls[l]->CopyColStarts();
            TrueP_bndtop_H1_lvls[l]->CopyRowStarts();

            TrueP_bndbot_Hdiv_lvls[l] = RAP(Restrict_bot_Hdiv_lvls[l], TrueP_Hdiv_lvls[l], Restrict_bot_Hdiv_lvls[l + 1]);
            TrueP_bndbot_Hdiv_lvls[l]->CopyColStarts();
            TrueP_bndbot_Hdiv_lvls[l]->CopyRowStarts();

            TrueP_bndtop_Hdiv_lvls[l] = RAP(Restrict_top_Hdiv_lvls[l], TrueP_Hdiv_lvls[l], Restrict_top_Hdiv_lvls[l + 1]);
            TrueP_bndtop_Hdiv_lvls[l]->CopyColStarts();
            TrueP_bndtop_Hdiv_lvls[l]->CopyRowStarts();
        }

        /*
        if (verbose)
             std::cout << "Drawing in H1 case \n";

        ParGridFunction * testfullH1 = new ParGridFunction(H1_space);
        FunctionCoefficient testH1_coeff(testH1fun);
        testfullH1->ProjectCoefficient(testH1_coeff);
        Vector testfullH1_tdofs(H1_space->TrueVSize());
        testfullH1->ParallelAssemble(testfullH1_tdofs);

        Vector testH1_bot_tdofs(H1_space->TrueVSize());
        testH1_bot_tdofs = 0.0;

        for ( unsigned int i = 0; i < tdofs_link_H1.size(); ++i )
        {
            int tdof_bot = tdofs_link_H1[i].first;
            testH1_bot_tdofs[tdof_bot] = testfullH1_tdofs[tdof_bot];
        }

        ParGridFunction * testH1_bot = new ParGridFunction(H1_space);
        testH1_bot->Distribute(&testH1_bot_tdofs);

        Vector testH1_top_tdofs(H1_space->TrueVSize());
        testH1_top_tdofs = 0.0;

        for ( unsigned int i = 0; i < tdofs_link_H1.size(); ++i )
        {
            int tdof_top = tdofs_link_H1[i].second;
            testH1_top_tdofs[tdof_top] = testfullH1_tdofs[tdof_top];
        }

        ParGridFunction * testH1_top = new ParGridFunction(H1_space);
        testH1_top->Distribute(&testH1_top_tdofs);

        if (visualization && dim < 4)
        {
            if (verbose)
                 std::cout << "Sending to GLVis in H1 case \n";

            char vishost[] = "localhost";
            int  visport   = 19916;
            socketstream u_sock(vishost, visport);
            u_sock << "parallel " << num_procs << " " << myid << "\n";
            u_sock.precision(8);
            u_sock << "solution\n" << *pmeshtsl << *testfullH1 << "window_title 'testfullH1'"
                   << endl;

            socketstream ubot_sock(vishost, visport);
            ubot_sock << "parallel " << num_procs << " " << myid << "\n";
            ubot_sock.precision(8);
            ubot_sock << "solution\n" << *pmeshtsl << *testH1_bot << "window_title 'testH1bot'"
                   << endl;

            socketstream utop_sock(vishost, visport);
            utop_sock << "parallel " << num_procs << " " << myid << "\n";
            utop_sock.precision(8);
            utop_sock << "solution\n" << *pmeshtsl << *testH1_top << "window_title 'testH1top'"
                   << endl;
        }
        */

        /*
        if (verbose)
             std::cout << "Drawing in Hdiv case \n";

        ParGridFunction * testfullHdiv = new ParGridFunction(Hdiv_space);
        VectorFunctionCoefficient testHdiv_coeff(dim, testHdivfun);
        testfullHdiv->ProjectCoefficient(testHdiv_coeff);
        Vector testfullHdiv_tdofs(Hdiv_space->TrueVSize());
        testfullHdiv->ParallelAssemble(testfullHdiv_tdofs);

        Vector testHdiv_bot_tdofs(Hdiv_space->TrueVSize());
        testHdiv_bot_tdofs = 0.0;

        for ( unsigned int i = 0; i < tdofs_link_Hdiv.size(); ++i)
        {
            int tdof_bot = tdofs_link_Hdiv[i].first;
            testHdiv_bot_tdofs[tdof_bot] = testfullHdiv_tdofs[tdof_bot];
        }

        ParGridFunction * testHdiv_bot = new ParGridFunction(Hdiv_space);
        testHdiv_bot->Distribute(&testHdiv_bot_tdofs);

        Vector testHdiv_top_tdofs(Hdiv_space->TrueVSize());
        testHdiv_top_tdofs = 0.0;

        for ( unsigned int i = 0; i < tdofs_link_Hdiv.size(); ++i)
        {
            int tdof_top = tdofs_link_Hdiv[i].second;
            testHdiv_top_tdofs[tdof_top] = testfullHdiv_tdofs[tdof_top];
        }

        ParGridFunction * testHdiv_top = new ParGridFunction(Hdiv_space);
        testHdiv_top->Distribute(&testHdiv_top_tdofs);

        if (visualization && dim < 4)
        {
            if (verbose)
                 std::cout << "Sending to GLVis in Hdiv case \n";

            char vishost[] = "localhost";
            int  visport   = 19916;
            socketstream u_sock(vishost, visport);
            u_sock << "parallel " << num_procs << " " << myid << "\n";
            u_sock.precision(8);
            u_sock << "solution\n" << *pmeshtsl << *testfullHdiv << "window_title 'testfullHdiv'"
                   << endl;

            socketstream ubot_sock(vishost, visport);
            ubot_sock << "parallel " << num_procs << " " << myid << "\n";
            ubot_sock.precision(8);
            ubot_sock << "solution\n" << *pmeshtsl << *testHdiv_bot << "window_title 'testHdivbot'"
                   << endl;

            socketstream utop_sock(vishost, visport);
            utop_sock << "parallel " << num_procs << " " << myid << "\n";
            utop_sock.precision(8);
            utop_sock << "solution\n" << *pmeshtsl << *testHdiv_top << "window_title 'testHdivtop'"
                   << endl;
        }
        */

        // critical for the considered problem
        if (strcmp(space_for_sigma,"H1") == 0)
            MFEM_ABORT ("Not supported case sigma from vector H1, think of the boundary conditions there");

        if (strcmp(space_for_S, "H1") == 0)
            init_cond_size_lvls[l] = tdofs_link_H1_lvls[l].size();
        else // L2
            init_cond_size_lvls[l] = tdofs_link_Hdiv_lvls[l].size();

        //ParFiniteElementSpace *H1vec_space;
        //if (strcmp(space_for_sigma,"H1") == 0)
            //H1vec_space = new ParFiniteElementSpace(pmeshtsl, h1_coll, dim, Ordering::byVDIM);
        //if (strcmp(space_for_sigma,"Hdiv") == 0)
            //Sigma_space_lvls[l] = Hdiv_space_lvls[l];
        //else
            //Sigma_space_lvls[l] = H1vec_space_lvls[l];
        Sigma_space_lvls[l] = Hdiv_space_lvls[l];

        if (strcmp(space_for_S,"H1") == 0)
            S_space_lvls[l] = H1_space_lvls[l];
        else // "L2"
            S_space_lvls[l] = L2_space_lvls[l];

        HYPRE_Int dimR = Hdiv_space_lvls[l]->GlobalTrueVSize();
        HYPRE_Int dimH = H1_space_lvls[l]->GlobalTrueVSize();
        HYPRE_Int dimHvec;
        //if (strcmp(space_for_sigma,"H1") == 0)
            //dimHvec = H1vec_space_lvls[l]->GlobalTrueVSize();
        HYPRE_Int dimW = L2_space_lvls[l]->GlobalTrueVSize();

        if (verbose)
        {
           std::cout << "***********************************************************\n";
           std::cout << "dim H(div)_h = " << dimR << ", ";
           //if (strcmp(space_for_sigma,"H1") == 0)
               //std::cout << "dim H1vec_h = " << dimHvec << ", ";
           std::cout << "dim H1_h = " << dimH << ", ";
           std::cout << "dim L2_h = " << dimW << "\n";
           std::cout << "Spaces we use: \n";
           if (strcmp(space_for_sigma,"Hdiv") == 0)
               std::cout << "H(div)";
           else
               std::cout << "H1vec";
           if (strcmp(space_for_S,"H1") == 0)
               std::cout << " x H1";
           if (strcmp(formulation,"cfosls") == 0)
               std::cout << " x L2 \n";
           std::cout << "***********************************************************\n";
        }

        // 7. Define the two BlockStructure of the problem.  block_offsets is used
        //    for Vector based on dof (like ParGridFunction or ParLinearForm),
        //    block_trueOffstes is used for Vector based on trueDof (HypreParVector
        //    for the rhs and solution of the linear system).  The offsets computed
        //    here are local to the processor.
        int numblocks = 1;

        if (strcmp(space_for_S,"H1") == 0)
            numblocks++;
        if (strcmp(formulation,"cfosls") == 0)
            numblocks++;

        if (verbose)
            std::cout << "Number of blocks in the formulation: " << numblocks << "\n";

        Array<int> block_offsets(numblocks + 1); // number of variables + 1
        int tempblknum = 0;
        block_offsets[0] = 0;
        tempblknum++;
        block_offsets[tempblknum] = Sigma_space_lvls[l]->GetVSize();
        tempblknum++;

        if (strcmp(space_for_S,"H1") == 0)
        {
            block_offsets[tempblknum] = H1_space_lvls[l]->GetVSize();
            tempblknum++;
        }
        if (strcmp(formulation,"cfosls") == 0)
        {
            block_offsets[tempblknum] = L2_space_lvls[l]->GetVSize();
            tempblknum++;
        }
        block_offsets.PartialSum();

        //Array<int> block_trueOffsets(numblocks + 1); // number of variables + 1
        block_trueOffsets_lvls[l] = new Array<int>(numblocks + 1);
        tempblknum = 0;
        (*block_trueOffsets_lvls[l])[0] = 0;
        tempblknum++;
        (*block_trueOffsets_lvls[l])[tempblknum] = Sigma_space_lvls[l]->TrueVSize();
        tempblknum++;

        if (strcmp(space_for_S,"H1") == 0)
        {
            (*block_trueOffsets_lvls[l])[tempblknum] = H1_space_lvls[l]->TrueVSize();
            tempblknum++;
        }
        if (strcmp(formulation,"cfosls") == 0)
        {
            (*block_trueOffsets_lvls[l])[tempblknum] = L2_space_lvls[l]->TrueVSize();
            tempblknum++;
        }
        block_trueOffsets_lvls[l]->PartialSum();

        //block_trueoffsets.resize(numblocks + 1);
        //for (int i = 0; i < block_trueOffsets.Size(); ++i)
            //block_trueoffsets[i] = block_trueOffsets[i];

       Transport_test Mytest(dim, numsol);

       // 8. Define the coefficients, analytical solution, and rhs of the PDE.

       //----------------------------------------------------------
       // Setting boundary conditions.
       //----------------------------------------------------------

       ess_bdrat_S.resize(pmeshtsl_lvls[l]->bdr_attributes.Max());
       for (unsigned int i = 0; i < ess_bdrat_S.size(); ++i)
           ess_bdrat_S[i] = 0;
       if (strcmp(space_for_S,"H1") == 0)
           ess_bdrat_S[0] = 1; // t = 0

       ess_bdrat_sigma.resize(pmeshtsl_lvls[l]->bdr_attributes.Max());
       for (unsigned int i = 0; i < ess_bdrat_sigma.size(); ++i)
           ess_bdrat_sigma[i] = 0;
       if (strcmp(space_for_S,"L2") == 0) // if S is from L2 we impose bdr condition for sigma at t = 0
           ess_bdrat_sigma[0] = 1;

       Array<int> ess_bdrS(pmeshtsl_lvls[l]->bdr_attributes.Max());
       for (unsigned int i = 0; i < ess_bdrat_S.size(); ++i)
           ess_bdrS[i] = ess_bdrat_S[i];

       /*
       ess_bdrS = 0;
       if (strcmp(space_for_S,"H1") == 0)
           ess_bdrS[0] = 1; // t = 0
       */
       Array<int> ess_bdrSigma(pmeshtsl->bdr_attributes.Max());
       for (unsigned int i = 0; i < ess_bdrat_sigma.size(); ++i)
           ess_bdrSigma[i] = ess_bdrat_sigma[i];
       /*
       ess_bdrSigma = 0;
       if (strcmp(space_for_S,"L2") == 0) // if S is from L2 we impose bdr condition for sigma at t = 0
       {
           ess_bdrSigma[0] = 1;
       }
       */

       if (verbose)
       {
           std::cout << "Boundary conditions: \n";
           std::cout << "ess bdr Sigma: \n";
           ess_bdrSigma.Print(std::cout, pmeshtsl_lvls[l]->bdr_attributes.Max());
           std::cout << "ess bdr S: \n";
           ess_bdrS.Print(std::cout, pmeshtsl_lvls[l]->bdr_attributes.Max());
       }
       //-----------------------

       // 9. Define the parallel grid function and parallel linear forms, solution
       //    vector and rhs.

       // 10. Assemble the finite element matrices for the CFOSLS operator  A
       //     where:

       ParBilinearForm *Ablock(new ParBilinearForm(Sigma_space_lvls[l]));
       HypreParMatrix *A;
       if (strcmp(space_for_S,"H1") == 0) // S is from H1
       {
           if (strcmp(space_for_sigma,"Hdiv") == 0) // sigma is from Hdiv
               Ablock->AddDomainIntegrator(new VectorFEMassIntegrator);
           else // sigma is from H1vec
               Ablock->AddDomainIntegrator(new ImproperVectorMassIntegrator);
       }
       else // "L2"
       {
           Ablock->AddDomainIntegrator(new VectorFEMassIntegrator(*Mytest.Ktilda));
     #ifdef REGULARIZE_A
           if (verbose)
               std::cout << "regularization is ON \n";
           double h_min, h_max, kappa_min, kappa_max;
           pmesh->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);
           if (verbose)
               std::cout << "coarse mesh steps: min " << h_min << " max " << h_max << "\n";

           double reg_param;
           reg_param = 0.1 * h_min * h_min;
           if (verbose)
               std::cout << "regularization parameter: " << reg_param << "\n";
           ConstantCoefficient reg_coeff(reg_param);
           Ablock->AddDomainIntegrator(new VectorFEMassIntegrator(reg_coeff)); // reduces the convergence rate but helps with iteration count
           //Ablock->AddDomainIntegrator(new DivDivIntegrator(reg_coeff)); // doesn't change much in the iteration count
     #endif
       }
       Ablock->Assemble();
       Ablock->EliminateEssentialBC(ess_bdrSigma);
       Ablock->Finalize();
       A = Ablock->ParallelAssemble();

       ParBilinearForm *Ablock_nobnd(new ParBilinearForm(Sigma_space_lvls[l]));
       HypreParMatrix *A_nobnd;
       if (strcmp(space_for_S,"H1") == 0) // S is from H1
       {
           if (strcmp(space_for_sigma,"Hdiv") == 0) // sigma is from Hdiv
               Ablock_nobnd->AddDomainIntegrator(new VectorFEMassIntegrator);
           else // sigma is from H1vec
               Ablock_nobnd->AddDomainIntegrator(new ImproperVectorMassIntegrator);
       }
       else // "L2"
       {
           Ablock_nobnd->AddDomainIntegrator(new VectorFEMassIntegrator(*Mytest.Ktilda));
     #ifdef REGULARIZE_A
           if (verbose)
               std::cout << "regularization is ON \n";
           double h_min, h_max, kappa_min, kappa_max;
           pmesh->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);
           if (verbose)
               std::cout << "coarse mesh steps: min " << h_min << " max " << h_max << "\n";

           double reg_param;
           reg_param = 0.1 * h_min * h_min;
           if (verbose)
               std::cout << "regularization parameter: " << reg_param << "\n";
           ConstantCoefficient reg_coeff(reg_param);
           Ablock_nobnd->AddDomainIntegrator(new VectorFEMassIntegrator(reg_coeff)); // reduces the convergence rate but helps with iteration count
     #endif
       }
       Ablock_nobnd->Assemble();
       Ablock_nobnd->Finalize();
       A_nobnd = Ablock_nobnd->ParallelAssemble();


       /*
       if (verbose)
           std::cout << "Checking the A matrix \n";

       MPI_Finalize();
       return 0;
       */

       //---------------
       //  C Block:
       //---------------

       ParBilinearForm *Cblock;
       HypreParMatrix *C;
       if (strcmp(space_for_S,"H1") == 0) // S is present
       {
           Cblock = new ParBilinearForm(S_space_lvls[l]);
           if (strcmp(space_for_S,"H1") == 0)
           {
               Cblock->AddDomainIntegrator(new MassIntegrator(*Mytest.bTb));
               if (strcmp(space_for_sigma,"Hdiv") == 0)
                    Cblock->AddDomainIntegrator(new DiffusionIntegrator(*Mytest.bbT));
           }
           else // "L2" & !eliminateS
           {
               Cblock->AddDomainIntegrator(new MassIntegrator(*(Mytest.bTb)));
           }
           Cblock->Assemble();
           Cblock->EliminateEssentialBC(ess_bdrS);
           Cblock->Finalize();
           C = Cblock->ParallelAssemble();

           SparseMatrix C_diag;
           C->GetDiag(C_diag);
           Array<int> EssBnd_tdofs_S;
           S_space_lvls[l]->GetEssentialTrueDofs(ess_bdrS, EssBnd_tdofs_S);
           for (int i = 0; i < EssBnd_tdofs_S.Size(); ++i)
           {
               int tdof = EssBnd_tdofs_S[i];
               C_diag.EliminateRow(tdof,1.0);
           }
       }

       ParBilinearForm *Cblock_nobnd;
       HypreParMatrix *C_nobnd;
       if (strcmp(space_for_S,"H1") == 0) // S is present
       {
           Cblock_nobnd = new ParBilinearForm(S_space_lvls[l]);
           if (strcmp(space_for_S,"H1") == 0)
           {
               Cblock_nobnd->AddDomainIntegrator(new MassIntegrator(*Mytest.bTb));
               if (strcmp(space_for_sigma,"Hdiv") == 0)
                    Cblock_nobnd->AddDomainIntegrator(new DiffusionIntegrator(*Mytest.bbT));
           }
           else // "L2" & !eliminateS
           {
               Cblock_nobnd->AddDomainIntegrator(new MassIntegrator(*(Mytest.bTb)));
           }
           Cblock_nobnd->Assemble();
           Cblock_nobnd->Finalize();
           C_nobnd = Cblock_nobnd->ParallelAssemble();
       }

       //---------------
       //  B Block:
       //---------------

       ParMixedBilinearForm *Bblock;
       HypreParMatrix *B;
       HypreParMatrix *BT;
       if (strcmp(space_for_S,"H1") == 0) // S is present
       {
           Bblock = new ParMixedBilinearForm(Sigma_space_lvls[l], S_space_lvls[l]);
           if (strcmp(space_for_sigma,"Hdiv") == 0) // sigma is from Hdiv
           {
               //Bblock->AddDomainIntegrator(new VectorFEMassIntegrator(*Mytest.b));
               Bblock->AddDomainIntegrator(new VectorFEMassIntegrator(*Mytest.minb));
           }
           else // sigma is from H1
               Bblock->AddDomainIntegrator(new MixedVectorScalarIntegrator(*Mytest.minb));
           Bblock->Assemble();
           {
               Vector testx(Bblock->Width());
               testx = 0.0;
               Vector testrhs(Bblock->Height());
               testrhs = 0.0;
               Bblock->EliminateTrialDofs(ess_bdrSigma, testx, testrhs);
           }
           Bblock->EliminateTestDofs(ess_bdrS);
           Bblock->Finalize();

           B = Bblock->ParallelAssemble();
           //*B *= -1.;
           BT = B->Transpose();
       }

       ParMixedBilinearForm *Bblock_nobnd;
       HypreParMatrix *B_nobnd;
       HypreParMatrix *BT_nobnd;
       if (strcmp(space_for_S,"H1") == 0) // S is present
       {
           Bblock_nobnd = new ParMixedBilinearForm(Sigma_space_lvls[l], S_space_lvls[l]);
           if (strcmp(space_for_sigma,"Hdiv") == 0) // sigma is from Hdiv
           {
               //Bblock->AddDomainIntegrator(new VectorFEMassIntegrator(*Mytest.b));
               Bblock_nobnd->AddDomainIntegrator(new VectorFEMassIntegrator(*Mytest.minb));
           }
           else // sigma is from H1
               Bblock_nobnd->AddDomainIntegrator(new MixedVectorScalarIntegrator(*Mytest.minb));
           Bblock_nobnd->Assemble();
           Bblock_nobnd->Finalize();

           B_nobnd = Bblock_nobnd->ParallelAssemble();
           //*B *= -1.;
           BT_nobnd = B_nobnd->Transpose();
       }

       //----------------
       //  D Block:
       //-----------------

       HypreParMatrix *D;
       HypreParMatrix *DT;

       if (strcmp(formulation,"cfosls") == 0)
       {
          ParMixedBilinearForm *Dblock(new ParMixedBilinearForm(Sigma_space_lvls[l], L2_space_lvls[l]));
          if (strcmp(space_for_sigma,"Hdiv") == 0) // sigma is from Hdiv
            Dblock->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
          else // sigma is from H1vec
            Dblock->AddDomainIntegrator(new VectorDivergenceIntegrator);
          Dblock->Assemble();
          {
              Vector testx(Dblock->Width());
              testx = 0.0;
              Vector testrhs(Dblock->Height());
              testrhs = 0.0;
              Dblock->EliminateTrialDofs(ess_bdrSigma, testx, testrhs);
          }

          Dblock->Finalize();
          D = Dblock->ParallelAssemble();
          DT = D->Transpose();
       }

       HypreParMatrix *D_nobnd;
       HypreParMatrix *DT_nobnd;

       if (strcmp(formulation,"cfosls") == 0)
       {
          ParMixedBilinearForm *Dblock_nobnd(new ParMixedBilinearForm(Sigma_space_lvls[l], L2_space_lvls[l]));
          if (strcmp(space_for_sigma,"Hdiv") == 0) // sigma is from Hdiv
            Dblock_nobnd->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
          else // sigma is from H1vec
            Dblock_nobnd->AddDomainIntegrator(new VectorDivergenceIntegrator);
          Dblock_nobnd->Assemble();
          Dblock_nobnd->Finalize();
          D_nobnd = Dblock_nobnd->ParallelAssemble();
          DT_nobnd = D_nobnd->Transpose();
       }

       //=======================================================
       // Setting up the block system Matrix
       //-------------------------------------------------------

       if (l < num_lvls - 1)
       {
           TrueP_lvls[l] = new BlockOperator(*block_trueOffsets_lvls[l], *block_trueOffsets_lvls[l + 1]);
           TrueP_lvls[l]->SetBlock(0,0, TrueP_Hdiv_lvls[l]);
           if (strcmp(space_for_S,"H1") == 0) // S is present
           {
               TrueP_lvls[l]->SetBlock(1,1, TrueP_H1_lvls[l]);
               TrueP_lvls[l]->SetBlock(2,2, TrueP_L2_lvls[l]);
           }
           else
               TrueP_lvls[l]->SetBlock(1,1, TrueP_L2_lvls[l]);
      }

      CFOSLSop_lvls[l] = new BlockOperator(*block_trueOffsets_lvls[l]);

      trueRhs_nobnd_lvls[l] = new BlockVector(*block_trueOffsets_lvls[l]);
      trueX_lvls[l] = new BlockVector(*block_trueOffsets_lvls[l]);

      //block_trueOffsets.Print();

      CFOSLSop_lvls[l]->SetBlock(0,0, A);
      if (strcmp(space_for_S,"H1") == 0) // S is present
      {
          CFOSLSop_lvls[l]->SetBlock(0,1, BT);
          CFOSLSop_lvls[l]->SetBlock(1,0, B);
          CFOSLSop_lvls[l]->SetBlock(1,1, C);
          if (strcmp(formulation,"cfosls") == 0)
          {
            CFOSLSop_lvls[l]->SetBlock(0,2, DT);
            CFOSLSop_lvls[l]->SetBlock(2,0, D);
          }
      }
      else // no S
          if (strcmp(formulation,"cfosls") == 0)
          {
            CFOSLSop_lvls[l]->SetBlock(0,1, DT);
            CFOSLSop_lvls[l]->SetBlock(1,0, D);
          }

      CFOSLSop_nobnd_lvls[l] = new BlockOperator(*block_trueOffsets_lvls[l]);
      CFOSLSop_nobnd_lvls[l]->SetBlock(0,0, A_nobnd);
      if (strcmp(space_for_S,"H1") == 0) // S is present
      {
          CFOSLSop_nobnd_lvls[l]->SetBlock(0,1, BT_nobnd);
          CFOSLSop_nobnd_lvls[l]->SetBlock(1,0, B_nobnd);
          CFOSLSop_nobnd_lvls[l]->SetBlock(1,1, C_nobnd);
          if (strcmp(formulation,"cfosls") == 0)
          {
            CFOSLSop_nobnd_lvls[l]->SetBlock(0,2, DT_nobnd);
            CFOSLSop_nobnd_lvls[l]->SetBlock(2,0, D_nobnd);
          }
      }
      else // no S
          if (strcmp(formulation,"cfosls") == 0)
          {
            CFOSLSop_nobnd_lvls[l]->SetBlock(0,1, DT_nobnd);
            CFOSLSop_nobnd_lvls[l]->SetBlock(1,0, D_nobnd);
          }
       if (verbose)
           cout << "Final saddle point matrix assembled \n";
       MPI_Barrier(MPI_COMM_WORLD);

       //=======================================================
       // Setting up the preconditioner
       //-------------------------------------------------------

       // Construct the operators for preconditioner
       if (verbose)
       {
           std::cout << "Block diagonal preconditioner: \n";
           if (use_ADS)
               std::cout << "ADS(A) for H(div) \n";
           else
                std::cout << "Diag(A) for H(div) or H1vec \n";
           if (strcmp(space_for_S,"H1") == 0) // S is from H1
               std::cout << "BoomerAMG(C) for H1 \n";
           if (strcmp(formulation,"cfosls") == 0 )
           {
               std::cout << "BoomerAMG(D Diag^(-1)(A) D^t) for L2 lagrange multiplier \n";
           }
           std::cout << "\n";
       }

       HypreParMatrix *Schur;
       if (strcmp(formulation,"cfosls") == 0 )
       {
          HypreParMatrix *AinvDt = D->Transpose();
          HypreParVector *Ad = new HypreParVector(MPI_COMM_WORLD, A->GetGlobalNumRows(),
                                               A->GetRowStarts());
          A->GetDiag(*Ad);
          AinvDt->InvScaleRows(*Ad);
          Schur = ParMult(D, AinvDt);
       }

       Solver * invA;
       if (use_ADS)
           invA = new HypreADS(*A, Sigma_space_lvls[l]);
       else // using Diag(A);
            invA = new HypreDiagScale(*A);

       invA->iterative_mode = false;

       Solver * invC;
       if (strcmp(space_for_S,"H1") == 0) // S is from H1
       {
           invC = new HypreBoomerAMG(*C);
           ((HypreBoomerAMG*)invC)->SetPrintLevel(0);
           ((HypreBoomerAMG*)invC)->iterative_mode = false;
       }

       Solver * invS;
       if (strcmp(formulation,"cfosls") == 0 )
       {
            invS = new HypreBoomerAMG(*Schur);
            ((HypreBoomerAMG *)invS)->SetPrintLevel(0);
            ((HypreBoomerAMG *)invS)->iterative_mode = false;
       }

       prec_lvls[l] = new BlockDiagonalPreconditioner(*block_trueOffsets_lvls[l]);
       if (prec_option > 0)
       {
           tempblknum = 0;
           prec_lvls[l]->SetDiagonalBlock(tempblknum, invA);
           tempblknum++;
           if (strcmp(space_for_S,"H1") == 0) // S is present
           {
               prec_lvls[l]->SetDiagonalBlock(tempblknum, invC);
               tempblknum++;
           }
           if (strcmp(formulation,"cfosls") == 0)
                prec_lvls[l]->SetDiagonalBlock(tempblknum, invS);
       }
       else
           if (verbose)
               cout << "No preconditioner is used. \n";

       // 12. Solve the linear system with MINRES.
       //     Check the norm of the unpreconditioned residual.

       solver_lvls[l] = new MINRESSolver(MPI_COMM_WORLD);
       solver_lvls[l]->SetAbsTol(atol);
       solver_lvls[l]->SetRelTol(rtol);
       solver_lvls[l]->SetMaxIter(max_iter);
       solver_lvls[l]->SetOperator(*CFOSLSop_lvls[l]);
       if (prec_option > 0)
            solver_lvls[l]->SetPreconditioner(*prec_lvls[l]);
       solver_lvls[l]->SetPrintLevel(0);

    }

    for (int l = 0; l < num_lvls; ++l)
    {
        if (l == 0)
        {
            CFOSLSop_coarsened_lvls[l] = CFOSLSop_lvls[0];
        }
        else // coarsening
        {
            CFOSLSop_coarsened_lvls[l] = new RAPOperator(*TrueP_lvls[l - 1], *CFOSLSop_coarsened_lvls[l - 1], *TrueP_lvls[l - 1]);
        }
    }

    /////////////////////////////////////////////////////////////////

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
   int numsol          = 0;

   int ser_ref_levels  = 2;
   int par_ref_levels  = 0;

   // 2. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
#ifdef USE_TSL
   const char *meshbase_file = "../data/star.mesh";
   int Nt = 4;
   double tau = 0.25;
#endif

   const char *formulation = "cfosls"; // "cfosls" or "fosls"
   const char *space_for_S = "L2";     // "H1" or "L2"
   const char *space_for_sigma = "Hdiv"; // "Hdiv" or "H1"
   bool eliminateS = true;            // in case space_for_S = "L2" defines whether we eliminate S from the system

   // solver options
   int prec_option = 1; //defines whether to use preconditioner or not, and which one
   bool use_ADS;

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
   args.AddOption(&prec_option, "-precopt", "--prec-option",
                  "Preconditioner choice (0, 1 or 2 for now).");
   args.AddOption(&formulation, "-form", "--formul",
                  "Formulation to use (cfosls or fosls).");
   args.AddOption(&space_for_S, "-spaceS", "--spaceS",
                  "Space for S (H1 or L2).");
   args.AddOption(&space_for_sigma, "-spacesigma", "--spacesigma",
                  "Space for sigma (Hdiv or H1).");
   args.AddOption(&eliminateS, "-elims", "--eliminateS", "-no-elims",
                  "--no-eliminateS",
                  "Turn on/off elimination of S in L2 formulation.");

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

   if (verbose)
   {
       if (strcmp(formulation,"cfosls") == 0)
           std::cout << "formulation: CFOSLS \n";
       else
           std::cout << "formulation: FOSLS \n";

       if (strcmp(space_for_sigma,"Hdiv") == 0)
           std::cout << "Space for sigma: Hdiv \n";
       else
           std::cout << "Space for sigma: H1 \n";

       if (strcmp(space_for_S,"H1") == 0)
           std::cout << "Space for S: H1 \n";
       else
           std::cout << "Space for S: L2 \n";

       if (strcmp(space_for_S,"L2") == 0)
       {
           std::cout << "S: is ";
           if (!eliminateS)
               std::cout << "not ";
           std::cout << "eliminated from the system \n";
       }
   }

   switch (prec_option)
   {
   case 1: // smth simple like AMS
       use_ADS = false;
       break;
   case 2: // MG
       use_ADS = true;
       break;
   default: // no preconditioner
       break;
   }

   if (verbose)
   {
       std::cout << "use_ADS = " << use_ADS << "\n";
   }

   MFEM_ASSERT(strcmp(formulation,"cfosls") == 0 || strcmp(formulation,"fosls") == 0, "Formulation must be cfosls or fosls!\n");
   MFEM_ASSERT(strcmp(space_for_S,"H1") == 0 || strcmp(space_for_S,"L2") == 0, "Space for S must be H1 or L2!\n");
   MFEM_ASSERT(strcmp(space_for_sigma,"Hdiv") == 0 || strcmp(space_for_sigma,"H1") == 0, "Space for sigma must be Hdiv or H1!\n");

   MFEM_ASSERT(!strcmp(space_for_sigma,"H1") == 0 || (strcmp(space_for_sigma,"H1") == 0 && strcmp(space_for_S,"H1") == 0), "Sigma from H1vec must be coupled with S from H1!\n");
   MFEM_ASSERT(!strcmp(space_for_sigma,"H1") == 0 || (strcmp(space_for_sigma,"H1") == 0 && use_ADS == false), "ADS cannot be used when sigma is from H1vec!\n");

   StopWatch chrono;

   //DEFAULTED LINEAR SOLVER OPTIONS
   //int max_iter = 150000;
   //double rtol = 1e-12;//1e-7;//1e-9;
   //double atol = 1e-14;//1e-9;//1e-12;

#ifdef NONHOMO_TEST
   if (nDimensions == 3)
       numsol = -33;
   else // 4D case
       numsol = -44;
#else
   if (nDimensions == 3)
       numsol = -3;
   else // 4D case
       numsol = -4;
#endif

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
       if (verbose)
            std::cout << "meshbase_file: " << meshbase_file << "\n";
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

   meshbase->CheckElementOrientation(true);

   for (int l = 0; l < ser_ref_levels; l++)
       meshbase->UniformRefinement();

   meshbase->CheckElementOrientation(true);

   ParMesh * pmeshbase = new ParMesh(comm, *meshbase);
   for (int l = 0; l < par_ref_levels; l++)
       pmeshbase->UniformRefinement();

   pmeshbase->CheckElementOrientation(true);

   //if (verbose)
       //std::cout << "pmeshbase shared structure \n";
   //pmeshbase->PrintSharedStructParMesh();

   delete meshbase;

   /*
   int nslabs = 3;
   Array<int> slabs_widths(nslabs);
   slabs_widths[0] = Nt / 2;
   slabs_widths[1] = Nt / 2 - 1;
   slabs_widths[2] = 1;
   ParMeshCyl * pmesh = new ParMeshCyl(comm, *pmeshbase, 0.0, tau, Nt, nslabs, &slabs_widths);
   */

   ParMeshCyl * pmesh = new ParMeshCyl(comm, *pmeshbase, 0.0, tau, Nt);

   //pmesh->PrintSlabsStruct();

   //delete pmeshbase;
   //delete pmesh;
   //MPI_Finalize();
   //return 0;

   /*
   if (num_procs == 1)
   {
       std::stringstream fname;
       fname << "pmesh_check.mesh";
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
   {
       std::cout << "mesh_file: " << mesh_file << "\n";
       std::cout << "Number of mpi processes: " << num_procs << "\n" << std::flush;
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
       for (int l = 0; l < par_ref_levels; l++)
           pmesh->UniformRefinement();
       delete mesh;
   }

#endif

   int dim = nDimensions;
   Transport_test Mytest(dim, numsol);

   MPI_Barrier(comm);
   std::cout << std::flush;
   MPI_Barrier(comm);

   if (verbose)
      std::cout << "Checking a single solve from a one TimeCylHyper instance "
                    "created for the entire domain \n";

  {
      int pref_lvls_tslab = 0;
      int solve_at_lvl = 0;
      TimeCylHyper * timeslab_test = new TimeCylHyper (*pmeshbase, 0.0, tau, Nt, pref_lvls_tslab,
                                                         formulation, space_for_S, space_for_sigma);

      int init_cond_size = timeslab_test->GetInitCondSize(solve_at_lvl);
      std::vector<std::pair<int,int> > * tdofs_link = timeslab_test->GetTdofsLink(solve_at_lvl);
      Vector Xinit(init_cond_size);
      ParFiniteElementSpace * testfespace;
      ParGridFunction * sol_exact;
      if (strcmp(space_for_S,"H1") == 0) // S is present
      {
          testfespace = timeslab_test->Get_S_space(solve_at_lvl);
          sol_exact = new ParGridFunction(testfespace);
          sol_exact->ProjectCoefficient(*Mytest.scalarS);
      }
      else
      {
          testfespace = timeslab_test->Get_Sigma_space(solve_at_lvl);
          sol_exact = new ParGridFunction(testfespace);
          sol_exact->ProjectCoefficient(*Mytest.sigma);
      }

      Vector sol_exact_truedofs(testfespace->TrueVSize());
      sol_exact->ParallelProject(sol_exact_truedofs);

      for (int i = 0; i < init_cond_size; ++i)
      {
          int tdof_bot = (*tdofs_link)[i].first;
          Xinit[i] = sol_exact_truedofs[tdof_bot];
      }

      //Xinit.Print();

      Vector Xout(init_cond_size);

      timeslab_test->Solve(solve_at_lvl,Xinit, Xout);

      // checking the error at the top boundary
      Vector Xout_exact(init_cond_size);
      for (int i = 0; i < init_cond_size; ++i)
      {
          int tdof_top = (*tdofs_link)[i].second;
          Xout_exact[i] = sol_exact_truedofs[tdof_top];
      }

      Vector Xout_error(init_cond_size);
      Xout_error = Xout;
      Xout_error -= Xout_exact;
      if (verbose)
      {
          std::cout << "|| Xout  - Xout_exact || = " << Xout_error.Norml2() / sqrt (Xout_error.Size()) << "\n";
          std::cout << "|| Xout  - Xout_exact || / || Xout_exact || = " << (Xout_error.Norml2() / sqrt (Xout_error.Size())) /
                       (Xout_exact.Norml2() / sqrt (Xout_exact.Size()))<< "\n";
      }

      // testing InterpolateAtBase()
      if (solve_at_lvl == 1)
      {
          Vector Xout_fine(timeslab_test->GetInitCondSize(0));
          timeslab_test->InterpolateAtBase("top", 0, Xout, Xout_fine);

          Vector Xout_truedofs(timeslab_test->Get_S_space(solve_at_lvl)->TrueVSize());
          Xout_truedofs = 0.0;
          for (unsigned int i = 0; i < tdofs_link->size(); ++i)
          {
              Xout_truedofs[(*tdofs_link)[i].second] = Xout[i];
          }
          ParGridFunction * Xout_dofs = new ParGridFunction(timeslab_test->Get_S_space(solve_at_lvl));
          Xout_dofs->Distribute(&Xout_truedofs);

          std::vector<std::pair<int,int> > * tdofs_fine_link = timeslab_test->GetTdofsLink(0);
          Vector Xout_fine_truedofs(timeslab_test->Get_S_space(0)->TrueVSize());
          Xout_fine_truedofs = 0.0;
          for (unsigned int i = 0; i < tdofs_fine_link->size(); ++i)
          {
              Xout_fine_truedofs[(*tdofs_fine_link)[i].second] = Xout_fine[i];
          }
          ParGridFunction * Xout_fine_dofs = new ParGridFunction(timeslab_test->Get_S_space(0));
          Xout_fine_dofs->Distribute(&Xout_fine_truedofs);

          ParMeshCyl * pmeshcyl_coarse = timeslab_test->Get_ParMeshCyl(solve_at_lvl);

          //std::cout << "pmeshcyl_coarse ne = " << pmeshcyl_coarse->GetNE() << "\n";
          ParMeshCyl * pmeshcyl_fine = timeslab_test->Get_ParMeshCyl(0);

          char vishost[] = "localhost";
          int  visport   = 19916;

          socketstream uuu_sock(vishost, visport);
          uuu_sock << "parallel " << num_procs << " " << myid << "\n";
          uuu_sock.precision(8);
          uuu_sock << "solution\n" << *pmeshcyl_coarse <<
                      *Xout_dofs << "window_title 'Xout coarse'"
                 << endl;

          socketstream s_sock(vishost, visport);
          s_sock << "parallel " << num_procs << " " << myid << "\n";
          s_sock.precision(8);
          MPI_Barrier(comm);
          s_sock << "solution\n" << *pmeshcyl_fine <<
                    *Xout_fine_dofs << "window_title 'Xout fine'"
                  << endl;

      }


      delete timeslab_test;
  }

  //MPI_Finalize();
  //return 0;

  if (verbose)
    std::cout << "Checking a sequential solve within several TimeCylHyper instances \n";

  {
      int pref_lvls_tslab = 1;
      int solve_at_lvl = 0;

      int nslabs = 2;
      std::vector<TimeCylHyper*> timeslabs(nslabs);
      double slab_tau = 0.125;
      int slab_width = 4; // in time steps (as time intervals) withing a single time slab
      double tinit_tslab = 0.0;

      if (verbose)
      {
          std::cout << "Creating a sequence of time slabs: \n";
          std::cout << "# of slabs: " << nslabs << "\n";
          std::cout << "# of time intervals per slab: " << slab_width << "\n";
          std::cout << "time step within a time slab: " << slab_tau << "\n";
          std::cout << "# of refinements: " << pref_lvls_tslab << "\n";
          if (solve_at_lvl == 0)
              std::cout << "solution level: " << solve_at_lvl << "\n";
      }

      for (int tslab = 0; tslab < nslabs; ++tslab )
      {
          timeslabs[tslab] = new TimeCylHyper (*pmeshbase, tinit_tslab, slab_tau, slab_width, pref_lvls_tslab,
                                                formulation, space_for_S, space_for_sigma);
          tinit_tslab += slab_tau * slab_width;
      }

      MFEM_ASSERT(fabs(tinit_tslab - 1.0) < 1.0e-14, "The slabs should cover the time interval "
                                                    "[0,1] but the upper bound doesn't match \n");

      Vector Xinit;

      int init_cond_size = timeslabs[0]->GetInitCondSize(solve_at_lvl);
      std::vector<std::pair<int,int> > * tdofs_link = timeslabs[0]->GetTdofsLink(solve_at_lvl);
      Xinit.SetSize(init_cond_size);

      ParFiniteElementSpace * testfespace;
      ParGridFunction * sol_exact;

      if (strcmp(space_for_S,"H1") == 0) // S is present
      {
          testfespace = timeslabs[0]->Get_S_space(solve_at_lvl);
          sol_exact = new ParGridFunction(testfespace);
          sol_exact->ProjectCoefficient(*Mytest.scalarS);
      }
      else
      {
          testfespace = timeslabs[0]->Get_Sigma_space(solve_at_lvl);
          sol_exact = new ParGridFunction(testfespace);
          sol_exact->ProjectCoefficient(*Mytest.sigma);
      }

      Vector sol_exact_truedofs(testfespace->TrueVSize());
      sol_exact->ParallelProject(sol_exact_truedofs);

      for (int i = 0; i < init_cond_size; ++i)
      {
          int tdof_bot = (*tdofs_link)[i].first;
          Xinit[i] = sol_exact_truedofs[tdof_bot];
      }

      // initializing the input boundary condition for the first vector

      Vector Xout(init_cond_size);

      for (int tslab = 0; tslab < nslabs; ++tslab )
      {
          /*
           * only for debugging
          if (tslab > 0)
          {
              if (strcmp(space_for_S,"H1") == 0) // S is present
              {
                  ParGridFunction * S_exact = new ParGridFunction(timeslabs[tslab]->Get_S_space());
                  S_exact->ProjectCoefficient(*Mytest.scalarS);
                  Vector S_exact_truedofs(timeslabs[tslab]->Get_S_space()->TrueVSize());
                  S_exact->ParallelAssemble(S_exact_truedofs);

                  for (int i = 0; i < init_cond_size; ++i)
                  {
                      int tdof_bot = (*tdofs_link)[i].first;
                      Xinit[i] = S_exact_truedofs[tdof_bot];
                  }
              }
              else
              {
                  ParGridFunction * sigma_exact = new ParGridFunction(timeslabs[tslab]->Get_Sigma_space());
                  sigma_exact->ProjectCoefficient(*Mytest.sigma);
                  Vector sigma_exact_truedofs(timeslabs[tslab]->Get_Sigma_space()->TrueVSize());
                  sigma_exact->ParallelAssemble(sigma_exact_truedofs);

                  for (int i = 0; i < init_cond_size; ++i)
                  {
                      int tdof_bot = (*tdofs_link)[i].first;
                      Xinit[i] = sigma_exact_truedofs[tdof_bot];
                  }
              }
          }
          */
          //Xinit.Print();
          timeslabs[tslab]->Solve(solve_at_lvl, Xinit, Xout);
          Xinit = Xout;
          if (strcmp(space_for_S,"L2") == 0)
              Xinit *= -1.0;

          // checking the error at the top boundary
          Vector Xout_exact(init_cond_size);
          for (int i = 0; i < init_cond_size; ++i)
          {
              int tdof_top = (*tdofs_link)[i].second;
              Xout_exact[i] = sol_exact_truedofs[tdof_top];
          }

          Vector Xout_error(init_cond_size);
          Xout_error = Xout;
          Xout_error -= Xout_exact;
          if (verbose)
          {
              std::cout << "|| Xout  - Xout_exact || = " << Xout_error.Norml2() / sqrt (Xout_error.Size()) << "\n";
              std::cout << "|| Xout  - Xout_exact || / || Xout_exact || = " << (Xout_error.Norml2() / sqrt (Xout_error.Size())) /
                           (Xout_exact.Norml2() / sqrt (Xout_exact.Size()))<< "\n";
          }

      }
  }

  //MPI_Finalize();
  //return 0;

  if (verbose)
    std::cout << "Checking a two-grid scheme with independent fine and sequential coarse solvers \n";
  {
      int fine_lvl = 0;
      int coarse_lvl = 1;

      int pref_lvls_tslab = 1;

      int nslabs = 2;
      std::vector<TimeCylHyper*> timeslabs(nslabs);
      double slab_tau = 0.125;
      int slab_width = 4; // in time steps (as time intervals) withing a single time slab

      if (verbose)
      {
          std::cout << "Creating a sequence of time slabs: \n";
          std::cout << "# of slabs: " << nslabs << "\n";
          std::cout << "# of time intervals per slab: " << slab_width << "\n";
          std::cout << "time step within a time slab: " << slab_tau << "\n";
          std::cout << "# of refinements: " << pref_lvls_tslab << "\n";
      }

      double tinit_tslab = 0.0;
      for (int tslab = 0; tslab < nslabs; ++tslab )
      {
          timeslabs[tslab] = new TimeCylHyper (*pmeshbase, tinit_tslab, slab_tau, slab_width, pref_lvls_tslab,
                                                formulation, space_for_S, space_for_sigma);
          tinit_tslab += slab_tau * slab_width;
      }

      MFEM_ASSERT(fabs(tinit_tslab - 1.0) < 1.0e-14, "The slabs should cover the time interval "
                                                    "[0,1] but the upper bound doesn't match \n");

      // getting some approximations for the first iteration from the coarse solver
      // sequential coarse solve
      if (verbose)
          std::cout << "Sequential coarse solve: \n";

      std::vector<Vector*> Xouts_coarse(nslabs + 1);
      int solve_at_lvl = 1;

      Vector Xinit;
      // initializing the input boundary condition for the first vector
      int init_cond_size = timeslabs[0]->GetInitCondSize(solve_at_lvl);
      std::vector<std::pair<int,int> > * tdofs_link = timeslabs[0]->GetTdofsLink(solve_at_lvl);
      Xinit.SetSize(init_cond_size);

      ParFiniteElementSpace * testfespace;
      ParGridFunction * sol_exact;

      if (strcmp(space_for_S,"H1") == 0) // S is present
      {
          testfespace = timeslabs[0]->Get_S_space(solve_at_lvl);
          sol_exact = new ParGridFunction(testfespace);
          sol_exact->ProjectCoefficient(*Mytest.scalarS);
      }
      else
      {
          testfespace = timeslabs[0]->Get_Sigma_space(solve_at_lvl);
          sol_exact = new ParGridFunction(testfespace);
          sol_exact->ProjectCoefficient(*Mytest.sigma);
      }

      Vector sol_exact_truedofs(testfespace->TrueVSize());
      sol_exact->ParallelProject(sol_exact_truedofs);

      for (int i = 0; i < init_cond_size; ++i)
      {
          int tdof_bot = (*tdofs_link)[i].first;
          Xinit[i] = sol_exact_truedofs[tdof_bot];
      }

      Vector Xout(init_cond_size);

      Xouts_coarse[0] = new Vector(init_cond_size);
      (*Xouts_coarse[0]) = Xinit;

      for (int tslab = 0; tslab < nslabs; ++tslab )
      {
          timeslabs[tslab]->Solve(solve_at_lvl, Xinit, Xout);
          Xinit = Xout;
          if (strcmp(space_for_S,"L2") == 0)
              Xinit *= -1.0;

          Xouts_coarse[tslab + 1] = new Vector(init_cond_size);
          (*Xouts_coarse[tslab]) = Xinit;


          Vector Xout_exact(init_cond_size);

          // checking the error at the top boundary
          ParFiniteElementSpace * testfespace;
          ParGridFunction * sol_exact;

          if (strcmp(space_for_S,"H1") == 0) // S is present
          {
              testfespace = timeslabs[0]->Get_S_space(solve_at_lvl);
              sol_exact = new ParGridFunction(testfespace);
              sol_exact->ProjectCoefficient(*Mytest.scalarS);
          }
          else
          {
              testfespace = timeslabs[0]->Get_Sigma_space(solve_at_lvl);
              sol_exact = new ParGridFunction(testfespace);
              sol_exact->ProjectCoefficient(*Mytest.sigma);
          }

          Vector sol_exact_truedofs(testfespace->TrueVSize());
          sol_exact->ParallelProject(sol_exact_truedofs);

          for (int i = 0; i < init_cond_size; ++i)
          {
              int tdof_top = (*tdofs_link)[i].second;
              Xout_exact[i] = sol_exact_truedofs[tdof_top];
          }

          Vector Xout_error(init_cond_size);
          Xout_error = Xout;
          Xout_error -= Xout_exact;
          if (verbose)
          {
              std::cout << "|| Xout  - Xout_exact || = " << Xout_error.Norml2() / sqrt (Xout_error.Size()) << "\n";
              std::cout << "|| Xout  - Xout_exact || / || Xout_exact || = " << (Xout_error.Norml2() / sqrt (Xout_error.Size())) /
                           (Xout_exact.Norml2() / sqrt (Xout_exact.Size()))<< "\n";
          }

      } // end of loop over all time slabs, performing a coarse solve

      if (verbose)
          std::cout << "Creating initial data for the two-grid method \n";
      solve_at_lvl = 0;

      std::vector<Vector*> Xinits_fine(nslabs + 1);
      std::vector<Vector*> Xouts_fine(nslabs + 1);

      int init_cond_size_fine = timeslabs[0]->GetInitCondSize(solve_at_lvl);

      Xinits_fine[0] = new Vector(init_cond_size_fine);
      timeslabs[0]->InterpolateAtBase("bot", solve_at_lvl, *Xouts_coarse[0], *Xinits_fine[0]);
      Xouts_fine[0] = new Vector(init_cond_size_fine);
      *Xouts_fine[0] = *Xinits_fine[0];

      for (int tslab = 0; tslab < nslabs; ++tslab )
      {
          Xinits_fine[tslab + 1] = new Vector(init_cond_size_fine);

          // interpolate Xouts_coarse on the finer mesh into Xinits_fine
          timeslabs[tslab]->InterpolateAtBase("top", solve_at_lvl, *Xouts_coarse[tslab + 1], *Xinits_fine[tslab + 1]);

          Xouts_fine[tslab + 1] = new Vector(init_cond_size_fine);
      }

      // now we have Xinits_fine as initial conditions for the fine grid solves

      if (verbose)
          std::cout << "Starting two-grid iterations ... \n";

      int numlvls = pref_lvls_tslab + 1;
      MFEM_ASSERT(numlvls == 2, "Current implementation allows only a two-grid scheme \n");
      std::vector<Array<Vector*>*> residuals_lvls(numlvls);
      std::vector<Array<Vector*>*> corr_lvls(numlvls);
      std::vector<Array<Vector*>*> sol_lvls(numlvls);
      for (unsigned int i = 0; i < residuals_lvls.size(); ++i)
      {
          residuals_lvls[i] = new Array<Vector*>(nslabs);
          corr_lvls[i] = new Array<Vector*>(nslabs);
          sol_lvls[i] = new Array<Vector*>(nslabs);
          for (int tslab = 0; tslab < nslabs; ++tslab )
          {
              (*residuals_lvls[i])[tslab] = new Vector(timeslabs[tslab]->ProblemSize(i));
              (*corr_lvls[i])[tslab] = new Vector(timeslabs[tslab]->ProblemSize(i));
              (*sol_lvls[i])[tslab] = new Vector(timeslabs[tslab]->ProblemSize(i));
              *(*sol_lvls[i])[tslab] = 0.0;
          }
      }

      for (int it = 0; it < 2; ++it)
      {
          // 1. parallel-in-time smoothing
          for (int tslab = 0; tslab < nslabs; ++tslab )
          {
              timeslabs[tslab]->Solve(fine_lvl, *Xinits_fine[tslab], *Xouts_fine[tslab]);
              *(*sol_lvls[fine_lvl])[tslab] = *(timeslabs[tslab]->GetSol(fine_lvl));
              if (tslab > 0)
                  timeslabs[tslab]->ComputeResidual(fine_lvl, *Xouts_fine[tslab - 1], *(*sol_lvls[fine_lvl])[tslab],
                          *(*residuals_lvls[fine_lvl])[tslab]);
              else
                  timeslabs[tslab]->ComputeResidual(fine_lvl, *Xinits_fine[0], *(*sol_lvls[fine_lvl])[0],
                          *(*residuals_lvls[fine_lvl])[0]);

              //(*residuals_lvls[fine_lvl])[tslab]->Print();
              //;
          }


          // 2. projecting onto coarse space
          for (int tslab = 0; tslab < nslabs; ++tslab )
          {
              timeslabs[tslab]->Restrict(fine_lvl, *(*residuals_lvls[fine_lvl])[tslab], *(*residuals_lvls[coarse_lvl])[tslab]);
          }

          // 3. coarse problem solve
          Xinit = 0.0;
          for (int tslab = 0; tslab < nslabs; ++tslab )
          {
              //(*residuals_lvls[coarse_lvl])[tslab]->Print();

              timeslabs[tslab]->Solve("coarsened", coarse_lvl, *(*residuals_lvls[coarse_lvl])[tslab],
                                      *(*corr_lvls[coarse_lvl])[tslab], Xinit, Xout);
              Xinit = Xout;
              if (strcmp(space_for_S,"L2") == 0)
                  Xinit *= -1.0;

              (*Xouts_coarse[tslab]) = Xinit;

          } // end of loop over all time slabs, performing a coarse solve

          // 4. interpolating back and updating the solution
          for (int tslab = 0; tslab < nslabs; ++tslab )
          {
              //for (int i = 0; i < (*corr_lvls[coarse_lvl])[tslab]->Size(); ++i)
                  //std::cout << "corr coarse = " << (*(*corr_lvls[coarse_lvl])[tslab])[i] << "\n";

              //(*corr_lvls[coarse_lvl])[tslab]->Print();

              timeslabs[tslab]->Interpolate(fine_lvl, *(*corr_lvls[coarse_lvl])[tslab], *(*corr_lvls[fine_lvl])[tslab]);
          }

          // computing error in each time slab before update
          if (verbose)
              std::cout << "Errors before adding the coarse grid corrections \n";
          for (int tslab = 0; tslab < nslabs; ++tslab )
          {
              timeslabs[tslab]->ComputeError(fine_lvl, *(*sol_lvls[fine_lvl])[tslab]);
          }

          for (int tslab = 0; tslab < nslabs; ++tslab )
          {
              //for (int i = 0; i < (*sol_lvls[fine_lvl])[tslab]->Size(); ++i)
                  //std::cout << "sol before = " << (*(*sol_lvls[fine_lvl])[tslab])[i] <<
                               //", corr = " << (*(*corr_lvls[fine_lvl])[tslab])[i] << "\n";
              *(*sol_lvls[fine_lvl])[tslab] += *(*corr_lvls[fine_lvl])[tslab];
          }

          // 4.5 computing error in each time slab
          if (verbose)
              std::cout << "Errors after adding the coarse grid corrections \n";
          for (int tslab = 0; tslab < nslabs; ++tslab )
          {
              timeslabs[tslab]->ComputeError(fine_lvl, *(*sol_lvls[fine_lvl])[tslab]);
          }


          // 5. update initial conditions to start the next iteration
          int bdrcond_block = -1;
          if (strcmp(space_for_S,"H1") == 0) // S is present
              bdrcond_block = 1;
          else
              bdrcond_block = 0;

          for (int tslab = 0; tslab < nslabs; ++tslab )
          {
              std::vector<std::pair<int,int> > * tdofs_link = timeslabs[tslab]->GetTdofsLink(fine_lvl);
              *Xinits_fine[tslab] = 0.0;

              BlockVector sol_viewer((*sol_lvls[fine_lvl])[tslab]->GetData(),
                                     *timeslabs[tslab]->GetBlockTrueOffsets(fine_lvl));

              // FIXME: We have actually two values at all interfaces from two time slabs
              // Here I simply chose the time slab which is above the interface
              for (int i = 0; i < init_cond_size_fine; ++i)
              {
                  int tdof_bot = (*tdofs_link)[i].first;
                  (*Xinits_fine[tslab])[i] = sol_viewer.GetBlock(bdrcond_block)[tdof_bot];
              }

          }

      }

  }



#if 0
  if (verbose)
    std::cout << "Checking a sequential coarse solve with following ~parallel fine solves \n";

  {
      int pref_lvls_tslab = 1;

      int nslabs = 2;
      std::vector<TimeCylHyper*> timeslabs(nslabs);
      double slab_tau = 0.125;
      int slab_width = 4; // in time steps (as time intervals) withing a single time slab

      if (verbose)
      {
          std::cout << "Creating a sequence of time slabs: \n";
          std::cout << "# of slabs: " << nslabs << "\n";
          std::cout << "# of time intervals per slab: " << slab_width << "\n";
          std::cout << "time step within a time slab: " << slab_tau << "\n";
          std::cout << "# of refinements: " << pref_lvls_tslab << "\n";
      }

      double tinit_tslab = 0.0;
      for (int tslab = 0; tslab < nslabs; ++tslab )
      {
          timeslabs[tslab] = new TimeCylHyper (*pmeshbase, tinit_tslab, slab_tau, slab_width, pref_lvls_tslab,
                                                formulation, space_for_S, space_for_sigma);
          tinit_tslab += slab_tau * slab_width;
      }

      MFEM_ASSERT(fabs(tinit_tslab - 1.0) < 1.0e-14, "The slabs should cover the time interval "
                                                    "[0,1] but the upper bound doesn't match \n");


      // sequential coarse solve
      if (verbose)
          std::cout << "Sequential coarse solve: \n";

      std::vector<Vector*> Xouts_coarse(nslabs + 1);
      int solve_at_lvl = 1;

      Vector Xinit;
      // initializing the input boundary condition for the first vector
      int init_cond_size = timeslabs[0]->GetInitCondSize(solve_at_lvl);
      std::vector<std::pair<int,int> > * tdofs_link = timeslabs[0]->GetTdofsLink(solve_at_lvl);
      Xinit.SetSize(init_cond_size);

      ParFiniteElementSpace * testfespace;
      ParGridFunction * sol_exact;

      if (strcmp(space_for_S,"H1") == 0) // S is present
      {
          testfespace = timeslabs[0]->Get_S_space(solve_at_lvl);
          sol_exact = new ParGridFunction(testfespace);
          sol_exact->ProjectCoefficient(*Mytest.scalarS);
      }
      else
      {
          testfespace = timeslabs[0]->Get_Sigma_space(solve_at_lvl);
          sol_exact = new ParGridFunction(testfespace);
          sol_exact->ProjectCoefficient(*Mytest.sigma);
      }

      Vector sol_exact_truedofs(testfespace->TrueVSize());
      sol_exact->ParallelProject(sol_exact_truedofs);

      for (int i = 0; i < init_cond_size; ++i)
      {
          int tdof_bot = (*tdofs_link)[i].first;
          Xinit[i] = sol_exact_truedofs[tdof_bot];
      }

      Vector Xout(init_cond_size);

      Xouts_coarse[0] = new Vector(init_cond_size);
      (*Xouts_coarse[0]) = Xinit;

      for (int tslab = 0; tslab < nslabs; ++tslab )
      {
          timeslabs[tslab]->Solve(solve_at_lvl, Xinit, Xout);
          Xinit = Xout;
          if (strcmp(space_for_S,"L2") == 0)
              Xinit *= -1.0;

          Xouts_coarse[tslab + 1] = new Vector(init_cond_size);
          (*Xouts_coarse[tslab]) = Xinit;


          Vector Xout_exact(init_cond_size);

          // checking the error at the top boundary
          ParFiniteElementSpace * testfespace;
          ParGridFunction * sol_exact;

          if (strcmp(space_for_S,"H1") == 0) // S is present
          {
              testfespace = timeslabs[0]->Get_S_space(solve_at_lvl);
              sol_exact = new ParGridFunction(testfespace);
              sol_exact->ProjectCoefficient(*Mytest.scalarS);
          }
          else
          {
              testfespace = timeslabs[0]->Get_Sigma_space(solve_at_lvl);
              sol_exact = new ParGridFunction(testfespace);
              sol_exact->ProjectCoefficient(*Mytest.sigma);
          }

          Vector sol_exact_truedofs(testfespace->TrueVSize());
          sol_exact->ParallelProject(sol_exact_truedofs);

          for (int i = 0; i < init_cond_size; ++i)
          {
              int tdof_top = (*tdofs_link)[i].second;
              Xout_exact[i] = sol_exact_truedofs[tdof_top];
          }

          Vector Xout_error(init_cond_size);
          Xout_error = Xout;
          Xout_error -= Xout_exact;
          if (verbose)
          {
              std::cout << "|| Xout  - Xout_exact || = " << Xout_error.Norml2() / sqrt (Xout_error.Size()) << "\n";
              std::cout << "|| Xout  - Xout_exact || / || Xout_exact || = " << (Xout_error.Norml2() / sqrt (Xout_error.Size())) /
                           (Xout_exact.Norml2() / sqrt (Xout_exact.Size()))<< "\n";
          }

      } // end of loop over all time slabs, performing a coarse solve

      if (verbose)
          std::cout << "Creating initial data for fine grid solves \n";
      solve_at_lvl = 0;

      std::vector<Vector*> Xinits_fine(nslabs + 1);
      std::vector<Vector*> Xouts_fine(nslabs + 1);

      int init_cond_size_fine = timeslabs[0]->GetInitCondSize(solve_at_lvl);

      Xinits_fine[0] = new Vector(init_cond_size_fine);
      timeslabs[0]->InterpolateAtBase("bot", solve_at_lvl, *Xouts_coarse[0], *Xinits_fine[0]);
      Xouts_fine[0] = new Vector(init_cond_size_fine);
      *Xouts_fine[0] = *Xinits_fine[0];

      for (int tslab = 0; tslab < nslabs; ++tslab )
      {
          Xinits_fine[tslab + 1] = new Vector(init_cond_size_fine);

          // interpolate Xouts_coarse on the finer mesh into Xinits_fine
          timeslabs[tslab]->InterpolateAtBase("top", solve_at_lvl, *Xouts_coarse[tslab + 1], *Xinits_fine[tslab + 1]);

          Xouts_fine[tslab + 1] = new Vector(init_cond_size_fine);
      }


      if (verbose)
          std::cout << "Solving fine grid problems \n";

      // can be done in parallel, instead of for loop since the fine grid problems are independent
      solve_at_lvl = 0;
      for (int tslab = 0; tslab < nslabs; ++tslab )
      {
          timeslabs[tslab]->Solve(solve_at_lvl, *Xinits_fine[tslab], *Xouts_fine[tslab]);
          /*
          Xinit = Xout;
          if (strcmp(space_for_S,"L2") == 0)
              Xinit *= -1.0;

          Vector Xout_exact(init_cond_size);

          // checking the error at the top boundary
          ParFiniteElementSpace * testfespace;
          ParGridFunction * sol_exact;

          if (strcmp(space_for_S,"H1") == 0) // S is present
          {
              testfespace = timeslabs[0]->Get_S_space(solve_at_lvl);
              sol_exact = new ParGridFunction(testfespace);
              sol_exact->ProjectCoefficient(*Mytest.scalarS);
          }
          else
          {
              testfespace = timeslabs[0]->Get_Sigma_space(solve_at_lvl);
              sol_exact = new ParGridFunction(testfespace);
              sol_exact->ProjectCoefficient(*Mytest.sigma);
          }

          Vector sol_exact_truedofs(testfespace->TrueVSize());
          sol_exact->ParallelProject(sol_exact_truedofs);

          for (int i = 0; i < init_cond_size; ++i)
          {
              int tdof_top = (*tdofs_link)[i].second;
              Xout_exact[i] = sol_exact_truedofs[tdof_top];
          }

          Vector Xout_error(init_cond_size);
          Xout_error = Xout;
          Xout_error -= Xout_exact;
          if (verbose)
          {
              std::cout << "|| Xout  - Xout_exact || = " << Xout_error.Norml2() / sqrt (Xout_error.Size()) << "\n";
              std::cout << "|| Xout  - Xout_exact || / || Xout_exact || = " << (Xout_error.Norml2() / sqrt (Xout_error.Size())) /
                           (Xout_exact.Norml2() / sqrt (Xout_exact.Size()))<< "\n";
          }
          */
      } // end of loop over all time slabs, performing fine solves

      // Computing the corrections as the coarse level initial conditions
      if (verbose)
          std::cout << "Computing corrections \n";

      std::vector<Vector*> Xcorrs_coarse(nslabs + 1);

      solve_at_lvl = 1;

      int init_cond_size_coarse = timeslabs[0]->GetInitCondSize(solve_at_lvl);

      Xcorrs_coarse[0] = new Vector(init_cond_size_coarse);
      timeslabs[0]->RestrictAtBase("bot", solve_at_lvl, *Xouts_fine[0], *Xcorrs_coarse[0]);

      for (int tslab = 0; tslab < nslabs; ++tslab )
      {
          Xcorrs_coarse[tslab + 1] = new Vector(init_cond_size_coarse);

          // restricts Xouts_fine to the coarser mesh into Xcorrs_coarse
          timeslabs[tslab]->RestrictAtBase("top", solve_at_lvl, *Xouts_fine[tslab], *Xcorrs_coarse[tslab + 1]);
      }

      for (int t = 0; t <= nslabs; ++t )
      {
          *Xcorrs_coarse[t] += *Xouts_coarse[t];
      }

      // prolongating correstiions in time
      if (verbose)
          std::cout << "Prolongating corrections \n";



  }// end of block of testing a parallel-in-time solver
#endif

   // 17. Free the used memory.
   MPI_Barrier(comm);

   MPI_Finalize();
   return 0;
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

HypreParMatrix * CreateRestriction(const char * top_or_bot, ParFiniteElementSpace& pfespace, std::vector<std::pair<int,int> >& bot_to_top_tdofs_link)
{
    if (strcmp(top_or_bot, "top") != 0 && strcmp(top_or_bot, "bot") != 0)
    {
        MFEM_ABORT ("In num_lvls() top_or_bot must be 'top' or 'bot'!\n");
    }

    MPI_Comm comm = pfespace.GetComm();

    int m = bot_to_top_tdofs_link.size();
    int n = pfespace.TrueVSize();
    int * ia = new int[m + 1];
    ia[0] = 0;
    for (int i = 0; i < m; ++i)
        ia[i + 1] = ia[i] + 1;
    int * ja = new int [ia[m]];
    double * data = new double [ia[m]];
    int count = 0;
    for (int row = 0; row < m; ++row)
    {
        if (strcmp(top_or_bot, "bot") == 0)
            ja[count] = bot_to_top_tdofs_link[row].first;
        else
            ja[count] = bot_to_top_tdofs_link[row].second;
        data[count] = 1.0;
        count++;
    }
    SparseMatrix * diag = new SparseMatrix(ia, ja, data, m, n);

    int local_size = bot_to_top_tdofs_link.size();
    int global_marked_tdofs = 0;
    MPI_Allreduce(&local_size, &global_marked_tdofs, 1, MPI_INT, MPI_SUM, comm);

    //std::cout << "Got after Allreduce \n";

    int global_num_rows = global_marked_tdofs;
    int global_num_cols = pfespace.GlobalTrueVSize();

    int num_procs;
    MPI_Comm_size(comm, &num_procs);

    int myid;
    MPI_Comm_rank(comm, &myid);

    int * local_row_offsets = new int[num_procs + 1];
    local_row_offsets[0] = 0;
    MPI_Allgather(&m, 1, MPI_INT, local_row_offsets + 1, 1, MPI_INT, comm);

    int * local_col_offsets = new int[num_procs + 1];
    local_col_offsets[0] = 0;
    MPI_Allgather(&n, 1, MPI_INT, local_col_offsets + 1, 1, MPI_INT, comm);

    for (int j = 1; j < num_procs + 1; ++j)
        local_row_offsets[j] += local_row_offsets[j - 1];

    for (int j = 1; j < num_procs + 1; ++j)
        local_col_offsets[j] += local_col_offsets[j - 1];

    int * row_starts = new int[3];
    row_starts[0] = local_row_offsets[myid];
    row_starts[1] = local_row_offsets[myid + 1];
    row_starts[2] = local_row_offsets[num_procs];
    int * col_starts = new int[3];
    col_starts[0] = local_col_offsets[myid];
    col_starts[1] = local_col_offsets[myid + 1];
    col_starts[2] = local_col_offsets[num_procs];

    /*
    for (int i = 0; i < num_procs; ++i)
    {
        if (myid == i)
        {
            std::cout << "I am " << myid << "\n";
            std::cout << "my local_row_offsets not summed: \n";
            for (int j = 0; j < num_procs + 1; ++j)
                std::cout << local_row_offsets[j] << " ";
            std::cout << "\n";

            std::cout << "my local_col_offsets not summed: \n";
            for (int j = 0; j < num_procs + 1; ++j)
                std::cout << local_col_offsets[j] << " ";
            std::cout << "\n";
            std::cout << "\n";

            for (int j = 1; j < num_procs + 1; ++j)
                local_row_offsets[j] += local_row_offsets[j - 1];

            for (int j = 1; j < num_procs + 1; ++j)
                local_col_offsets[j] += local_col_offsets[j - 1];

            std::cout << "my local_row_offsets: \n";
            for (int j = 0; j < num_procs + 1; ++j)
                std::cout << local_row_offsets[j] << " ";
            std::cout << "\n";

            std::cout << "my local_col_offsets: \n";
            for (int j = 0; j < num_procs + 1; ++j)
                std::cout << local_row_offsets[j] << " ";
            std::cout << "\n";
            std::cout << "\n";

            int * row_starts = new int[3];
            row_starts[0] = local_row_offsets[myid];
            row_starts[1] = local_row_offsets[myid + 1];
            row_starts[2] = local_row_offsets[num_procs];
            int * col_starts = new int[3];
            col_starts[0] = local_col_offsets[myid];
            col_starts[1] = local_col_offsets[myid + 1];
            col_starts[2] = local_col_offsets[num_procs];

            std::cout << "my computed row starts: \n";
            std::cout << row_starts[0] << " " <<  row_starts[1] << " " << row_starts[2];
            std::cout << "\n";

            std::cout << "my computed col starts: \n";
            std::cout << col_starts[0] << " " <<  col_starts[1] << " " << col_starts[2];
            std::cout << "\n";

            std::cout << std::flush;
        }

        MPI_Barrier(comm);
    } // end fo loop over all processors, one after another
    */


    // FIXME:
    // MFEM_ABORT("Don't know how to create row_starts and col_starts \n");

    //std::cout << "Creating resT \n";

    HypreParMatrix * resT = new HypreParMatrix(comm, global_num_rows, global_num_cols, row_starts, col_starts, diag);

    //std::cout << "resT created \n";


    HypreParMatrix * res = resT->Transpose();
    res->CopyRowStarts();
    res->CopyColStarts();

    //std::cout << "Got after resT creation \n";

    return res;
}

// eltype must be "linearH1" or "RT0", for any other finite element the code doesn't work
// the fespace must correspond to the eltype provided
// bot_to_top_bels is the link between boundary elements (at the bottom and at the top)
// which can be taken out of ParMeshCyl

std::vector<std::pair<int,int> >* CreateBotToTopDofsLink(const char * eltype, FiniteElementSpace& fespace,
                                                         std::vector<std::pair<int,int> > & bot_to_top_bels, bool verbose)
{
    if (strcmp(eltype, "linearH1") != 0 && strcmp(eltype, "RT0") != 0)
    {
        MFEM_ABORT ("Provided eltype is not supported in CreateBotToTopDofsLink: must be linearH1 or RT0 strictly! \n");
    }

    int nbelpairs = bot_to_top_bels.size();
    // estimating the maximal memory size required
    Array<int> dofs;
    fespace.GetBdrElementDofs(0, dofs);
    int ndofpairs_max = nbelpairs * dofs.Size();

    if (verbose)
        std::cout << "nbelpairs = " << nbelpairs << ", estimated ndofpairs_max = " << ndofpairs_max << "\n";

    std::vector<std::pair<int,int> > * res = new std::vector<std::pair<int,int> >;
    res->reserve(ndofpairs_max);

    std::set<std::pair<int,int> > res_set;

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
                std::cout << "belind1: " << belind_first << ", bel_dofs_first: \n";
                bel_dofs_first.Print();
                std::cout << "belind2: " << belind_second << ", bel_dofs_second: \n";
                bel_dofs_second.Print();
            }


            if (bel_dofs_first.Size() != 1 || bel_dofs_second.Size() != 1)
            {
                MFEM_ABORT("For RT0 exactly one dof must correspond to each boundary element \n");
            }

            if (res_set.find(std::pair<int,int>(bel_dofs_first[0], bel_dofs_second[0])) == res_set.end())
            {
                res_set.insert(std::pair<int,int>(bel_dofs_first[0], bel_dofs_second[0]));
                res->push_back(std::pair<int,int>(bel_dofs_first[0], bel_dofs_second[0]));
            }

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
                std::cout << "belind1: " << belind_first << ", bel_dofs_first: \n";
                bel_dofs_first.Print();
                std::cout << "belind2: " << belind_second << ", bel_dofs_second: \n";
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

                if (res_set.find(std::pair<int,int>(bel_dofs_first[dofno], bel_dofs_second[dofno_second])) == res_set.end())
                {
                    res_set.insert(std::pair<int,int>(bel_dofs_first[dofno], bel_dofs_second[dofno_second]));
                    res->push_back(std::pair<int,int>(bel_dofs_first[dofno], bel_dofs_second[dofno_second]));
                }
                //res_set.insert(std::pair<int,int>(bel_dofs_first[dofno],
                                                  //bel_dofs_second[dofno_second]));

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
        for ( unsigned int i = 0; i < res->size(); ++i )
        {
            std::cout << "<" << (*res)[i].first << ", " << (*res)[i].second << "> \n";
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
        res(2) = 1.0;//(x*x + y*y + 1.0);
    }
    if (xt.Size() == 4)
    {
        res(3) = 1.0;//(x*x + y*y + z*z + 1.0);
    }
}


