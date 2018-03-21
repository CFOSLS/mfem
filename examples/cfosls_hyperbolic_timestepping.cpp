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

#define USE_TSL

using namespace std;
using namespace mfem;

// FIXME: In parallel, for shared essential tdofs matrices would have (for H1) not 1.0 at the diagonal but #procs which shared that vertex (=dof).
// Thus, boundary conditions are imposed incorrectly in this case. Fix that.

// TODO: Instead of specifying tdofs_link_H1 and _Hdiv and manually choosing by if-clauses,
// which to use for the Solve() int TimeSlab, it would be better to implement it as a block case
// with arbitrary number of blocks. Then input and output would be BlockVectors and there will be
// less switches

// abstract base class for time-slabbing
// TODO: Rename this and its children. This is not a time slab but a time cylinder
class TimeSlab
{
protected:
    ParMeshCyl * pmeshtsl;
    double t_init;
    double tau;
    int nt;
    bool own_pmeshtsl;
public:
    virtual ~TimeSlab();
    TimeSlab (ParMesh& Pmeshbase, double T_init, double Tau, int Nt);
    TimeSlab (ParMeshCyl& Pmeshtsl) : pmeshtsl(&Pmeshtsl), own_pmeshtsl(false) {}

    // interpretation of the input and output vectors depend on the implementation of Solve
    // but they are related to the boundary conditions at the input and at he output
    virtual void Solve(const Vector& vec_in, Vector& vec_out) const = 0;
};

TimeSlab::~TimeSlab()
{
    if (own_pmeshtsl)
        delete pmeshtsl;
}

TimeSlab::TimeSlab(ParMesh& Pmeshbase, double T_init, double Tau, int Nt)
    : t_init(T_init), tau(Tau), nt(Nt), own_pmeshtsl(true)
{
    pmeshtsl = new ParMeshCyl(Pmeshbase.GetComm(), Pmeshbase, t_init, tau, nt);
}

// specific class for time-slabbing in hyperbolic problems
class TimeSlabHyper : public TimeSlab
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
    std::vector<BlockOperator*> CFOSLSop_nobnd_lvls;
    std::vector<BlockDiagonalPreconditioner*> prec_lvls;
    std::vector<MINRESSolver*> solver_lvls;

    std::vector<int> init_cond_size_lvls;
    std::vector<std::vector<std::pair<int,int> > > tdofs_link_H1_lvls;
    std::vector<std::vector<std::pair<int,int> > > tdofs_link_Hdiv_lvls;

    std::vector<SparseMatrix*> P_H1_lvls;
    std::vector<SparseMatrix*> P_Hdiv_lvls;
    std::vector<HypreParMatrix*> TrueP_H1_lvls;
    std::vector<HypreParMatrix*> TrueP_Hdiv_lvls;

    bool verbose;
    bool visualization;

protected:
    void InitProblem();

public:
    ~TimeSlabHyper();
    TimeSlabHyper (ParMesh& Pmeshbase, double T_init, double Tau, int Nt, int Ref_lvls,
                   const char *Formulation, const char *Space_for_S, const char *Space_for_sigma);
    TimeSlabHyper (ParMeshCyl& Pmeshtsl, int Ref_Lvls,
                   const char *Formulation, const char *Space_for_S, const char *Space_for_sigma);

    virtual void Solve(const Vector &bnd_tdofs_bot, Vector &bnd_tdofs_top) const override
    { Solve(0, bnd_tdofs_bot, bnd_tdofs_top); }

    void Solve(int lvl, const Vector &bnd_tdofs_bot, Vector &bnd_tdofs_top) const;
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
    }
    HypreParMatrix * Get_TrueP_Hdiv(int lvl)
    {
        if (lvl >= 0 && lvl < ref_lvls)
            if (TrueP_Hdiv_lvls[lvl])
                return TrueP_Hdiv_lvls[lvl];
    }
};

TimeSlabHyper::~TimeSlabHyper()
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
    for (unsigned int i = 0; i < CFOSLSop_nobnd_lvls.size(); ++i)
        delete CFOSLSop_nobnd_lvls[i];
    for (unsigned int i = 0; i < prec_lvls.size(); ++i)
        delete prec_lvls[i];
    for (unsigned int i = 0; i < solver_lvls.size(); ++i)
        delete solver_lvls[i];
    for (unsigned int i = 0; i < P_H1_lvls.size(); ++i)
        delete P_H1_lvls[i];
    for (unsigned int i = 0; i < P_Hdiv_lvls.size(); ++i)
        delete P_Hdiv_lvls[i];
}

TimeSlabHyper::TimeSlabHyper (ParMesh& Pmeshbase, double T_init, double Tau, int Nt, int Ref_lvls,
                              const char *Formulation, const char *Space_for_S, const char *Space_for_sigma)
    : TimeSlab(Pmeshbase, T_init, Tau, Nt), ref_lvls(Ref_lvls),
      formulation(Formulation), space_for_S(Space_for_S), space_for_sigma(Space_for_sigma)
{
    InitProblem();
}

TimeSlabHyper::TimeSlabHyper (ParMeshCyl& Pmeshtsl, int Ref_Lvls,
                              const char *Formulation, const char *Space_for_S, const char *Space_for_sigma)
    : TimeSlab(Pmeshtsl), ref_lvls(Ref_Lvls),
      formulation(Formulation), space_for_S(Space_for_S), space_for_sigma(Space_for_sigma)
{
    InitProblem();
}

void TimeSlabHyper::Solve(int lvl, const Vector& bnd_tdofs_bot, Vector& bnd_tdofs_top) const
{
    if (!(lvl >= 0 && lvl <= ref_lvls))
    {
        MFEM_ABORT("Incorrect lvl argument for TimeSlabHyper::Solve() \n");
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
    BlockVector trueX(block_trueOffsets);
    //BlockVector trueRhs(block_trueOffsets);
    trueX = 0.0;

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

    BlockVector trueRhs2(block_trueOffsets);
    trueRhs2 = trueRhs_nobnd;
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

}

void TimeSlabHyper::InitProblem()
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

    visualization = 1;

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
    pmeshtsl_lvls.resize(ref_lvls);
    Hdiv_space_lvls.resize(ref_lvls);
    H1_space_lvls.resize(ref_lvls);
    L2_space_lvls.resize(ref_lvls);
    Sigma_space_lvls.resize(ref_lvls);
    S_space_lvls.resize(ref_lvls);

    block_trueOffsets_lvls.resize(ref_lvls);
    CFOSLSop_lvls.resize(ref_lvls);
    CFOSLSop_nobnd_lvls.resize(ref_lvls);
    prec_lvls.resize(ref_lvls);
    solver_lvls.resize(ref_lvls);

    TrueP_H1_lvls.resize(ref_lvls - 1);
    TrueP_Hdiv_lvls.resize(ref_lvls - 1);
    P_H1_lvls.resize(ref_lvls - 1);
    P_Hdiv_lvls.resize(ref_lvls - 1);

    init_cond_size_lvls.resize(ref_lvls);
    tdofs_link_H1_lvls.resize(ref_lvls);
    tdofs_link_Hdiv_lvls.resize(ref_lvls);

    const SparseMatrix* P_Hdiv_local;
    const SparseMatrix* P_H1_local;

    for (int l = ref_lvls - 1; l >= 0; --l)
    {
        // creating pmesh for level l
        if (l == ref_lvls - 1)
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

        // for all but one levels we create projection matrices between levels
        // and projectors assembled on true dofs if MG preconditioner is used
        if (l < ref_lvls - 1)
        {
            Hdiv_space->Update();
            H1_space->Update();

            // TODO: Rewrite these computations

            P_Hdiv_local = (SparseMatrix *)Hdiv_space->GetUpdateOperator();
            P_Hdiv_lvls[l] = RemoveZeroEntries(*P_Hdiv_local);

            auto d_td_coarse_Hdiv = Hdiv_space_lvls[l + 1]->Dof_TrueDof_Matrix();
            SparseMatrix * RP_Hdiv_local = Mult(*Hdiv_space_lvls[l]->GetRestrictionMatrix(), *P_Hdiv_lvls[l]);
            TrueP_Hdiv_lvls[ref_lvls - 2 - l] = d_td_coarse_Hdiv->LeftDiagMult(
                        *RP_Hdiv_local, Hdiv_space_lvls[l]->GetTrueDofOffsets());
            TrueP_Hdiv_lvls[ref_lvls - 2 - l]->CopyColStarts();
            TrueP_Hdiv_lvls[ref_lvls - 2 - l]->CopyRowStarts();

            delete RP_Hdiv_local;


            P_H1_local = (SparseMatrix *)H1_space->GetUpdateOperator();
            P_H1_lvls[l] = RemoveZeroEntries(*P_H1_local);

            auto d_td_coarse_H1 = H1_space_lvls[l + 1]->Dof_TrueDof_Matrix();
            SparseMatrix * RP_H1_local = Mult(*H1_space_lvls[l]->GetRestrictionMatrix(), *P_H1_lvls[l]);
            TrueP_H1_lvls[ref_lvls - 2 - l] = d_td_coarse_H1->LeftDiagMult(
                        *RP_H1_local, H1_space_lvls[l]->GetTrueDofOffsets());
            TrueP_H1_lvls[ref_lvls - 2 - l]->CopyColStarts();
            TrueP_H1_lvls[ref_lvls - 2 - l]->CopyRowStarts();

            delete RP_H1_local;

        }

        for (int i = 0; i < num_procs; ++i)
        {
            if (myid == i)
            {
                std::cout << "I am " << myid << "\n";

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
                    std::cout << "corr. dof pair: <" << dof1 << "," << dof2 << ">\n";
                    std::cout << "corr. tdof pair: <" << tdof1 << "," << tdof2 << ">\n";
                    if (tdof1 * tdof2 < 0)
                        MFEM_ABORT( "unsupported case: tdof1 and tdof2 belong to different processors! \n");

                    if (tdof1 > -1)
                    {
                        tdofs_link_H1_lvls[l].push_back(std::pair<int,int>(tdof1, tdof2));
                        ++count;
                    }
                    else
                        std::cout << "Ignored dofs pair which are not own tdofs \n";
                }
            }
            MPI_Barrier(comm);
        } // end fo loop over all processors, one after another

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
                        std::cout << "Ignored a dofs pair which are not own tdofs \n";
                }
            }
            MPI_Barrier(comm);
        } // end fo loop over all processors, one after another

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

      CFOSLSop_lvls[l] = new BlockOperator(*block_trueOffsets_lvls[l]);

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
   const char *space_for_S = "H1";     // "H1" or "L2"
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
   int max_iter = 150000;
   double rtol = 1e-12;//1e-7;//1e-9;
   double atol = 1e-14;//1e-9;//1e-12;

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

#ifdef USE_TSL
   std::vector<std::pair<int,int> > * tdofs_link_H1 = new std::vector<std::pair<int,int> >;
   for (int i = 0; i < num_procs; ++i)
   {
       if (myid == i)
       {
           std::vector<std::pair<int,int> > * dofs_link_H1 =
                   CreateBotToTopDofsLink("linearH1",*H1_space, pmesh->bot_to_top_bels);
           std::cout << std::flush;

           tdofs_link_H1->reserve(dofs_link_H1->size());

           //std::cout << "dof pairs for H1: \n";
           for ( unsigned int i = 0; i < dofs_link_H1->size(); ++i )
           {
               int dof1 = (*dofs_link_H1)[i].first;
               int dof2 = (*dofs_link_H1)[i].second;
               //std::cout << "<" << dof1 << ", " << dof2 << "> \n";
               int tdof1 = H1_space->GetLocalTDofNumber(dof1);
               int tdof2 = H1_space->GetLocalTDofNumber(dof2);
               //std::cout << "corr. tdof pair: <" << tdof1 << "," << tdof2 << ">\n";
               if (tdof1 * tdof2 < 0)
                   MFEM_ABORT( "unsupported case: tdof1 and tdof2 belong to different processors! \n");

               if (tdof1 > -1)
                   tdofs_link_H1->push_back(std::pair<int,int>(tdof1, tdof2));
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
   testfullH1->ParallelAssemble(testfullH1_tdofs);

   Vector testH1_bot_tdofs(H1_space->TrueVSize());
   testH1_bot_tdofs = 0.0;

   for ( unsigned int i = 0; i < tdofs_link_H1->size(); ++i )
   {
       int tdof_bot = (*tdofs_link_H1)[i].first;
       testH1_bot_tdofs[tdof_bot] = testfullH1_tdofs[tdof_bot];
   }

   ParGridFunction * testH1_bot = new ParGridFunction(H1_space);
   testH1_bot->Distribute(&testH1_bot_tdofs);

   Vector testH1_top_tdofs(H1_space->TrueVSize());
   testH1_top_tdofs = 0.0;

   //std::set<std::pair<int,int> >::iterator it;
   for ( unsigned int i = 0; i < tdofs_link_H1->size(); ++i )
   {
       int tdof_top = (*tdofs_link_H1)[i].second;
       testH1_top_tdofs[tdof_top] = testfullH1_tdofs[tdof_top];
   }

   ParGridFunction * testH1_top = new ParGridFunction(H1_space);
   testH1_top->Distribute(&testH1_top_tdofs);

   if (visualization && nDimensions < 4)
   {
       if (verbose)
            std::cout << "Sending to GLVis in H1 case \n";

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


   std::vector<std::pair<int,int> > * tdofs_link_Hdiv = new std::vector<std::pair<int,int> >;
   for (int i = 0; i < num_procs; ++i)
   {
       if (myid == i)
       {
           std::vector<std::pair<int,int> > * dofs_link_RT0 =
                      CreateBotToTopDofsLink("RT0",*Hdiv_space, pmesh->bot_to_top_bels);
           std::cout << std::flush;
           tdofs_link_Hdiv->reserve(dofs_link_RT0->size());

           //std::cout << "dof pairs for Hdiv: \n";
           std::set<std::pair<int,int> >::iterator it;
           for ( unsigned int i = 0; i < dofs_link_RT0->size(); ++i)
           {
               int dof1 = (*dofs_link_RT0)[i].first;
               int dof2 = (*dofs_link_RT0)[i].second;
               //std::cout << "<" << dof1 << ", " << dof2 << "> \n";
               int tdof1 = Hdiv_space->GetLocalTDofNumber(dof1);
               int tdof2 = Hdiv_space->GetLocalTDofNumber(dof2);
               //std::cout << "corr. tdof pair: <" << tdof1 << "," << tdof2 << ">\n";
               if ((tdof1 > 0 && tdof2 < 0) || (tdof1 < 0 && tdof2 > 0))
               {
                   //std::cout << "Caught you! tdof1 = " << tdof1 << ", tdof2 = " << tdof2 << "\n";
                   MFEM_ABORT( "unsupported case: tdof1 and tdof2 belong to different processors! \n");
               }

               if (tdof1 > -1)
                   tdofs_link_Hdiv->push_back(std::pair<int,int>(tdof1, tdof2));
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
   testfullHdiv->ParallelAssemble(testfullHdiv_tdofs);

   Vector testHdiv_bot_tdofs(Hdiv_space->TrueVSize());
   testHdiv_bot_tdofs = 0.0;

   for ( unsigned int i = 0; i < tdofs_link_Hdiv->size(); ++i)
   {
       int tdof_bot = (*tdofs_link_Hdiv)[i].first;
       testHdiv_bot_tdofs[tdof_bot] = testfullHdiv_tdofs[tdof_bot];
   }

   ParGridFunction * testHdiv_bot = new ParGridFunction(Hdiv_space);
   testHdiv_bot->Distribute(&testHdiv_bot_tdofs);

   Vector testHdiv_top_tdofs(Hdiv_space->TrueVSize());
   testHdiv_top_tdofs = 0.0;

   for ( unsigned int i = 0; i < tdofs_link_Hdiv->size(); ++i)
   {
       int tdof_top = (*tdofs_link_Hdiv)[i].second;
       testHdiv_top_tdofs[tdof_top] = testfullHdiv_tdofs[tdof_top];
   }

   ParGridFunction * testHdiv_top = new ParGridFunction(Hdiv_space);
   testHdiv_top->Distribute(&testHdiv_top_tdofs);

   if (visualization && nDimensions < 4)
   {
       if (verbose)
            std::cout << "Sending to GLVis in Hdiv case \n";

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
#endif

   ParFiniteElementSpace *H1vec_space;
   if (strcmp(space_for_sigma,"H1") == 0)
       H1vec_space = new ParFiniteElementSpace(pmesh, h1_coll, dim, Ordering::byVDIM);

   ParFiniteElementSpace * Sigma_space;
   if (strcmp(space_for_sigma,"Hdiv") == 0)
       Sigma_space = Hdiv_space;
   else
       Sigma_space = H1vec_space;

   ParFiniteElementSpace * S_space;
   if (strcmp(space_for_S,"H1") == 0)
       S_space = H1_space;
   else // "L2"
       S_space = L2_space;

   HYPRE_Int dimR = Hdiv_space->GlobalTrueVSize();
   HYPRE_Int dimH = H1_space->GlobalTrueVSize();
   HYPRE_Int dimHvec;
   if (strcmp(space_for_sigma,"H1") == 0)
       dimHvec = H1vec_space->GlobalTrueVSize();
   HYPRE_Int dimW = L2_space->GlobalTrueVSize();

   if (verbose)
   {
      std::cout << "***********************************************************\n";
      std::cout << "dim H(div)_h = " << dimR << ", ";
      if (strcmp(space_for_sigma,"H1") == 0)
          std::cout << "dim H1vec_h = " << dimHvec << ", ";
      std::cout << "dim H1_h = " << dimH << ", ";
      std::cout << "dim L2_h = " << dimW << "\n";
      std::cout << "Spaces we use: \n";
      if (strcmp(space_for_sigma,"Hdiv") == 0)
          std::cout << "H(div)";
      else
          std::cout << "H1vec";
      if (strcmp(space_for_S,"H1") == 0)
          std::cout << " x H1";
      else // "L2"
          if (!eliminateS)
              std::cout << " x L2";
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
   else // "L2"
       if (!eliminateS)
           numblocks++;
   if (strcmp(formulation,"cfosls") == 0)
       numblocks++;

   if (verbose)
       std::cout << "Number of blocks in the formulation: " << numblocks << "\n";

   Array<int> block_offsets(numblocks + 1); // number of variables + 1
   int tempblknum = 0;
   block_offsets[0] = 0;
   tempblknum++;
   block_offsets[tempblknum] = Sigma_space->GetVSize();
   tempblknum++;

   if (strcmp(space_for_S,"H1") == 0)
   {
       block_offsets[tempblknum] = H1_space->GetVSize();
       tempblknum++;
   }
   else // "L2"
       if (!eliminateS)
       {
           block_offsets[tempblknum] = L2_space->GetVSize();
           tempblknum++;
       }
   if (strcmp(formulation,"cfosls") == 0)
   {
       block_offsets[tempblknum] = L2_space->GetVSize();
       tempblknum++;
   }
   block_offsets.PartialSum();

   Array<int> block_trueOffsets(numblocks + 1); // number of variables + 1
   tempblknum = 0;
   block_trueOffsets[0] = 0;
   tempblknum++;
   block_trueOffsets[tempblknum] = Sigma_space->TrueVSize();
   tempblknum++;

   if (strcmp(space_for_S,"H1") == 0)
   {
       block_trueOffsets[tempblknum] = H1_space->TrueVSize();
       tempblknum++;
   }
   else // "L2"
       if (!eliminateS)
       {
           block_trueOffsets[tempblknum] = L2_space->TrueVSize();
           tempblknum++;
       }
   if (strcmp(formulation,"cfosls") == 0)
   {
       block_trueOffsets[tempblknum] = L2_space->TrueVSize();
       tempblknum++;
   }
   block_trueOffsets.PartialSum();

   BlockVector x(block_offsets), rhs(block_offsets);
   BlockVector trueX(block_trueOffsets);
   BlockVector trueRhs(block_trueOffsets);
   x = 0.0;
   rhs = 0.0;
   trueX = 0.0;
   trueRhs = 0.0;

   Transport_test Mytest(nDimensions, numsol);

   ParGridFunction *S_exact = new ParGridFunction(S_space);
   S_exact->ProjectCoefficient(*(Mytest.scalarS));
   //ConstantCoefficient one(1.0);
   //S_exact->ProjectCoefficient(one);

   ParGridFunction * sigma_exact = new ParGridFunction(Sigma_space);
   sigma_exact->ProjectCoefficient(*(Mytest.sigma));

   x.GetBlock(0) = *sigma_exact;
   x.GetBlock(1) = *S_exact;

  // 8. Define the coefficients, analytical solution, and rhs of the PDE.
  ConstantCoefficient zero(.0);

  //----------------------------------------------------------
  // Setting boundary conditions.
  //----------------------------------------------------------

  Array<int> ess_bdrS(pmesh->bdr_attributes.Max());
  ess_bdrS = 0;
  if (strcmp(space_for_S,"H1") == 0)
      ess_bdrS[0] = 1; // t = 0
  Array<int> ess_bdrSigma(pmesh->bdr_attributes.Max());
  ess_bdrSigma = 0;
  if (strcmp(space_for_S,"L2") == 0) // if S is from L2 we impose bdr condition for sigma at t = 0
  {
      ess_bdrSigma[0] = 1;
  }

  if (verbose)
  {
      std::cout << "Boundary conditions: \n";
      std::cout << "ess bdr Sigma: \n";
      ess_bdrSigma.Print(std::cout, pmesh->bdr_attributes.Max());
      std::cout << "ess bdr S: \n";
      ess_bdrS.Print(std::cout, pmesh->bdr_attributes.Max());
  }
  //-----------------------

  // 9. Define the parallel grid function and parallel linear forms, solution
  //    vector and rhs.

  ParLinearForm *fform = new ParLinearForm(Sigma_space);

  fform->Assemble();

  ParLinearForm *qform;
  if (strcmp(space_for_S,"H1") == 0 || !eliminateS)
  {
      qform = new ParLinearForm(S_space);
      qform->Update(S_space, rhs.GetBlock(1), 0);
  }

  if (strcmp(space_for_S,"H1") == 0)
  {
      //if (strcmp(space_for_sigma,"Hdiv") == 0 )
          qform->AddDomainIntegrator(new GradDomainLFIntegrator(*Mytest.bf));
      qform->Assemble();//qform->Print();
  }
  else // "L2"
  {
      if (!eliminateS)
      {
          qform->AddDomainIntegrator(new DomainLFIntegrator(zero));
          qform->Assemble();
      }
  }

  ParLinearForm *gform;
  if (strcmp(formulation,"cfosls") == 0)
  {
      gform = new ParLinearForm(L2_space);
      gform->AddDomainIntegrator(new DomainLFIntegrator(*Mytest.scalardivsigma));
      gform->Assemble();
  }

  // 10. Assemble the finite element matrices for the CFOSLS operator  A
  //     where:

  ParBilinearForm *Ablock(new ParBilinearForm(Sigma_space));
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
      if (eliminateS) // S is eliminated
          Ablock->AddDomainIntegrator(new VectorFEMassIntegrator(*Mytest.Ktilda));
      else // S is present
          Ablock->AddDomainIntegrator(new VectorFEMassIntegrator);
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
  Ablock->EliminateEssentialBC(ess_bdrSigma, x.GetBlock(0), *fform);
  Ablock->Finalize();
  A = Ablock->ParallelAssemble();

  ParBilinearForm *Ablock_nobnd(new ParBilinearForm(Sigma_space));
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
      if (eliminateS) // S is eliminated
          Ablock_nobnd->AddDomainIntegrator(new VectorFEMassIntegrator(*Mytest.Ktilda));
      else // S is present
          Ablock_nobnd->AddDomainIntegrator(new VectorFEMassIntegrator);
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
  if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
  {
      Cblock = new ParBilinearForm(S_space);
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
      Cblock->EliminateEssentialBC(ess_bdrS, x.GetBlock(1), *qform);
      Cblock->Finalize();
      C = Cblock->ParallelAssemble();

      SparseMatrix C_diag;
      C->GetDiag(C_diag);
      Array<int> EssBnd_tdofs_S;
      S_space->GetEssentialTrueDofs(ess_bdrS, EssBnd_tdofs_S);
      for (int i = 0; i < EssBnd_tdofs_S.Size(); ++i)
      {
          int tdof = EssBnd_tdofs_S[i];
          C_diag.EliminateRow(tdof,1.0);
      }

  }

  ParBilinearForm *Cblock_nobnd;
  HypreParMatrix *C_nobnd;
  if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
  {
      Cblock_nobnd = new ParBilinearForm(S_space);
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
  if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
  {
      Bblock = new ParMixedBilinearForm(Sigma_space, S_space);
      if (strcmp(space_for_sigma,"Hdiv") == 0) // sigma is from Hdiv
      {
          //Bblock->AddDomainIntegrator(new VectorFEMassIntegrator(*Mytest.b));
          Bblock->AddDomainIntegrator(new VectorFEMassIntegrator(*Mytest.minb));
      }
      else // sigma is from H1
          Bblock->AddDomainIntegrator(new MixedVectorScalarIntegrator(*Mytest.minb));
      Bblock->Assemble();
      Bblock->EliminateTrialDofs(ess_bdrSigma, x.GetBlock(0), *qform);
      Bblock->EliminateTestDofs(ess_bdrS);
      Bblock->Finalize();

      B = Bblock->ParallelAssemble();
      //*B *= -1.;
      BT = B->Transpose();
  }

  ParMixedBilinearForm *Bblock_nobnd;
  HypreParMatrix *B_nobnd;
  HypreParMatrix *BT_nobnd;
  if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
  {
      Bblock_nobnd = new ParMixedBilinearForm(Sigma_space, S_space);
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

#ifdef TESTING
   Array<int> block_truetestOffsets(3); // number of variables + 1
   block_truetestOffsets[0] = 0;
   //block_truetestOffsets[1] = C_space->TrueVSize();
   block_truetestOffsets[1] = Sigma_space->TrueVSize();
   if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
       block_truetestOffsets[2] = S_space->TrueVSize();
   block_truetestOffsets.PartialSum();

   BlockOperator *TestOp = new BlockOperator(block_truetestOffsets);

   TestOp->SetBlock(0,0, A);
   TestOp->SetBlock(0,1, BT);
   TestOp->SetBlock(1,0, B);
   TestOp->SetBlock(1,1, C);

   IterativeSolver * testsolver;
   testsolver = new CGSolver(comm);
   if (verbose)
       cout << "Linear test solver: CG \n";

   testsolver->SetAbsTol(atol);
   testsolver->SetRelTol(rtol);
   testsolver->SetMaxIter(max_iter);
   testsolver->SetOperator(*TestOp);

   testsolver->SetPrintLevel(0);

   BlockVector truetestX(block_truetestOffsets), truetestRhs(block_truetestOffsets);
   truetestX = 0.0;
   truetestRhs = 1.0;

   fform->ParallelAssemble(truetestRhs.GetBlock(0));
   if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
       qform->ParallelAssemble(truetestRhs.GetBlock(1));

   truetestX = 0.0;
   testsolver->Mult(truetestRhs, truetestX);

   chrono.Stop();

   if (verbose)
   {
       if (testsolver->GetConverged())
           std::cout << "Linear solver converged in " << testsolver->GetNumIterations()
                     << " iterations with a residual norm of " << testsolver->GetFinalNorm() << ".\n";
       else
           std::cout << "Linear solver did not converge in " << testsolver->GetNumIterations()
                     << " iterations. Residual norm is " << testsolver->GetFinalNorm() << ".\n";
       std::cout << "Linear solver took " << chrono.RealTime() << "s. \n";
   }

   MPI_Finalize();
   return 0;
#endif


  //----------------
  //  D Block:
  //-----------------

  HypreParMatrix *D;
  HypreParMatrix *DT;

  if (strcmp(formulation,"cfosls") == 0)
  {
     ParMixedBilinearForm *Dblock(new ParMixedBilinearForm(Sigma_space, L2_space));
     if (strcmp(space_for_sigma,"Hdiv") == 0) // sigma is from Hdiv
       Dblock->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
     else // sigma is from H1vec
       Dblock->AddDomainIntegrator(new VectorDivergenceIntegrator);
     Dblock->Assemble();
     Dblock->EliminateTrialDofs(ess_bdrSigma, x.GetBlock(0), *gform);
     Dblock->Finalize();
     D = Dblock->ParallelAssemble();
     DT = D->Transpose();
  }

  HypreParMatrix *D_nobnd;
  HypreParMatrix *DT_nobnd;

  if (strcmp(formulation,"cfosls") == 0)
  {
     ParMixedBilinearForm *Dblock_nobnd(new ParMixedBilinearForm(Sigma_space, L2_space));
     if (strcmp(space_for_sigma,"Hdiv") == 0) // sigma is from Hdiv
       Dblock_nobnd->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
     else // sigma is from H1vec
       Dblock_nobnd->AddDomainIntegrator(new VectorDivergenceIntegrator);
     Dblock_nobnd->Assemble();
     Dblock_nobnd->Finalize();
     D_nobnd = Dblock_nobnd->ParallelAssemble();
     DT_nobnd = D_nobnd->Transpose();
  }
#ifdef TESTING2
  {
      MFEM_ASSERT(strcmp(formulation,"cfosls") == 0, "For TESTING2 we need gform thus cfosls formulation \n");

      Array<int> block_truetestOffsets(3); // number of variables + 1
      block_truetestOffsets[0] = 0;
      //block_truetestOffsets[1] = C_space->TrueVSize();
      block_truetestOffsets[1] = Sigma_space->TrueVSize();
      if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
          block_truetestOffsets[2] = W_space->TrueVSize();
      block_truetestOffsets.PartialSum();

      BlockOperator *TestOp = new BlockOperator(block_truetestOffsets);

      TestOp->SetBlock(0,0, A);
      TestOp->SetBlock(0,1, DT);
      TestOp->SetBlock(1,0, D);

      IterativeSolver * testsolver;
      testsolver = new MINRESSolver(comm);
      if (verbose)
          cout << "Linear test solver: MINRES \n";

      testsolver->SetAbsTol(atol);
      testsolver->SetRelTol(rtol);
      testsolver->SetMaxIter(max_iter);
      testsolver->SetOperator(*TestOp);

      testsolver->SetPrintLevel(1);

      BlockVector truetestX(block_truetestOffsets), truetestRhs(block_truetestOffsets);
      truetestX = 0.0;
      truetestRhs = 0.0;

      gform->ParallelAssemble(truetestRhs.GetBlock(1));

      truetestX = 0.0;
      testsolver->Mult(truetestRhs, truetestX);

      chrono.Stop();

      if (verbose)
      {
          if (testsolver->GetConverged())
              std::cout << "Linear solver converged in " << testsolver->GetNumIterations()
                        << " iterations with a residual norm of " << testsolver->GetFinalNorm() << ".\n";
          else
              std::cout << "Linear solver did not converge in " << testsolver->GetNumIterations()
                        << " iterations. Residual norm is " << testsolver->GetFinalNorm() << ".\n";
          std::cout << "Linear solver took " << chrono.RealTime() << "s. \n";
      }
  }


  MPI_Finalize();
  return 0;
#endif


  //=======================================================
  // Setting up the block system Matrix
  //-------------------------------------------------------

 tempblknum = 0;
 fform->ParallelAssemble(trueRhs.GetBlock(tempblknum));
 tempblknum++;
 if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
 {
   qform->ParallelAssemble(trueRhs.GetBlock(tempblknum));
   tempblknum++;
 }
 if (strcmp(formulation,"cfosls") == 0)
    gform->ParallelAssemble(trueRhs.GetBlock(tempblknum));

 BlockOperator *CFOSLSop = new BlockOperator(block_trueOffsets);
 CFOSLSop->SetBlock(0,0, A);
 if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
 {
     CFOSLSop->SetBlock(0,1, BT);
     CFOSLSop->SetBlock(1,0, B);
     CFOSLSop->SetBlock(1,1, C);
     if (strcmp(formulation,"cfosls") == 0)
     {
       CFOSLSop->SetBlock(0,2, DT);
       CFOSLSop->SetBlock(2,0, D);
     }
 }
 else // no S
     if (strcmp(formulation,"cfosls") == 0)
     {
       CFOSLSop->SetBlock(0,1, DT);
       CFOSLSop->SetBlock(1,0, D);
     }

 BlockOperator *CFOSLSop_nobnd = new BlockOperator(block_trueOffsets);
 CFOSLSop_nobnd->SetBlock(0,0, A_nobnd);
 if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
 {
     CFOSLSop_nobnd->SetBlock(0,1, BT_nobnd);
     CFOSLSop_nobnd->SetBlock(1,0, B_nobnd);
     CFOSLSop_nobnd->SetBlock(1,1, C_nobnd);
     if (strcmp(formulation,"cfosls") == 0)
     {
       CFOSLSop_nobnd->SetBlock(0,2, DT_nobnd);
       CFOSLSop_nobnd->SetBlock(2,0, D_nobnd);
     }
 }
 else // no S
     if (strcmp(formulation,"cfosls") == 0)
     {
       CFOSLSop_nobnd->SetBlock(0,1, DT_nobnd);
       CFOSLSop_nobnd->SetBlock(1,0, D_nobnd);
     }
  if (verbose)
      cout << "Final saddle point matrix assembled \n";
  MPI_Barrier(MPI_COMM_WORLD);


  // checking an alternative way of imposing boundary conditions on the right hand side
  BlockVector trueBnd(block_trueOffsets);
  trueBnd = 0.0;
  {
      Vector sigma_exact_truedofs(Sigma_space->TrueVSize());
      sigma_exact->ParallelProject(sigma_exact_truedofs);

      Array<int> EssBnd_tdofs_sigma;
      Sigma_space->GetEssentialTrueDofs(ess_bdrSigma, EssBnd_tdofs_sigma);

      for (int i = 0; i < EssBnd_tdofs_sigma.Size(); ++i)
      {
          int tdof = EssBnd_tdofs_sigma[i];
          trueBnd.GetBlock(0)[tdof] = sigma_exact_truedofs[tdof];
          //std::cout << "tdof = " << tdof << "truebnd.block0 = " << trueBnd.GetBlock(0)[tdof]
                       //<< ", truerhs.block0 = " << trueRhs.GetBlock(0)[tdof] << "\n";
      }

      if (strcmp(space_for_S,"H1") == 0) // S is present
      {
          Array<int> EssBnd_tdofs_S;
          Vector S_exact_truedofs(S_space->TrueVSize());
          S_exact->ParallelProject(S_exact_truedofs);
          S_space->GetEssentialTrueDofs(ess_bdrS, EssBnd_tdofs_S);

          for (int i = 0; i < EssBnd_tdofs_S.Size(); ++i)
          {
              int tdof = EssBnd_tdofs_S[i];
              trueBnd.GetBlock(1)[tdof] = S_exact_truedofs[tdof];
              //std::cout << "tdof = " << tdof << "truebnd.block1 = " << trueBnd.GetBlock(1)[tdof]
                           //<< ", truerhs.block1 = " << trueRhs.GetBlock(1)[tdof] << "\n";
          }
      }
  }

  //MPI_Finalize();
  //return 0;

  BlockVector trueBndCor(block_trueOffsets);
  trueBndCor = 0.0;
  CFOSLSop_nobnd->Mult(trueBnd, trueBndCor); // more general than lines below
  /* works only for H1
  if (strcmp(space_for_S,"H1") == 0) // S is present
  {
      BT_nobnd->Mult(trueBnd.GetBlock(1), trueBndCor.GetBlock(0));
      C_nobnd->Mult(trueBnd.GetBlock(1), trueBndCor.GetBlock(1));
  }
  */

  ParLinearForm *fform_nobnd = new ParLinearForm(Sigma_space);
  fform_nobnd->Assemble();

  ParLinearForm *qform_nobnd;
  if (strcmp(space_for_S,"H1") == 0 || !eliminateS)
  {
      qform_nobnd = new ParLinearForm(S_space);
  }

  if (strcmp(space_for_S,"H1") == 0)
  {
      //if (strcmp(space_for_sigma,"Hdiv") == 0 )
          qform_nobnd->AddDomainIntegrator(new GradDomainLFIntegrator(*Mytest.bf));
      qform_nobnd->Assemble();//qform->Print();
  }
  else // "L2"
  {
      if (!eliminateS)
      {
          qform_nobnd->AddDomainIntegrator(new DomainLFIntegrator(zero));
          qform_nobnd->Assemble();
      }
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

  if (strcmp(space_for_S,"H1") == 0 || !eliminateS)
      qform_nobnd->ParallelAssemble(trueRhs_nobnd.GetBlock(1));
  gform_nobnd->ParallelAssemble(trueRhs_nobnd.GetBlock(numblocks - 1));

  BlockVector trueRhs2(block_trueOffsets);

  trueRhs2 = trueRhs_nobnd;
  trueRhs2 -= trueBndCor;

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

  /*
  // incorrect, for debugging
  //trueBnd = 1.0;
  trueRhs2 = trueBndCor;

  for (int i = 0; i < num_procs; ++i)
  {
      if (myid == i)
      {
          std::cout << "I am " << myid << "\n";
          std::cout << "my true bnd \n";
          trueBnd.Print();
          std::cout << "\n" << std::flush;
      }
      MPI_Barrier(comm);
  } // end fo loop over all processors, one after another
  */

  BlockVector trueRhs_diff(block_trueOffsets);
  trueRhs_diff = trueRhs;
  trueRhs_diff -= trueRhs2;

  //std::cout << "trueRhs block 0 \n";
  //trueRhs_diff.GetBlock(0).Print();
  std::cout << "|| trueRhs - trueRhs2 || block 0 = " << trueRhs_diff.GetBlock(0).Norml2() /
               sqrt (trueRhs_diff.GetBlock(0).Size()) << "\n";
  if (strcmp(space_for_S,"H1") == 0) // S is present
  {
      //std::cout << "trueRhs block 1 \n";
      //trueRhs_diff.GetBlock(1).Print();
      std::cout << "|| trueRhs - trueRhs2 || block 1 = " << trueRhs_diff.GetBlock(1).Norml2() /
                   sqrt (trueRhs_diff.GetBlock(1).Size()) << "\n";
  }
  //std::cout << "trueRhs block 2 \n";
  //trueRhs_diff.GetBlock(numblocks - 1).Print();
  std::cout << "|| trueRhs - trueRhs2 || block " << numblocks - 1 << " = " <<
               trueRhs_diff.GetBlock(numblocks - 1).Norml2() /
               sqrt (trueRhs_diff.GetBlock(numblocks - 1).Size()) << "\n";

  std::cout << "|| trueRhs - trueRhs2 || full = " << trueRhs_diff.Norml2() / sqrt (trueRhs_diff.Size()) << "\n";

  //MPI_Finalize();
  //return 0;


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
      else
      {
          if (!eliminateS) // S is from L2 and not eliminated
               std::cout << "Diag(C) for L2 \n";
      }
      if (strcmp(formulation,"cfosls") == 0 )
      {
          std::cout << "BoomerAMG(D Diag^(-1)(A) D^t) for L2 lagrange multiplier \n";
      }
      std::cout << "\n";
  }
  chrono.Clear();
  chrono.Start();

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
      invA = new HypreADS(*A, Sigma_space);
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
  else // S from L2
  {
      if (!eliminateS) // S is from L2 and not eliminated
      {
          invC = new HypreDiagScale(*C);
          ((HypreDiagScale*)invC)->iterative_mode = false;
      }
  }

  Solver * invS;
  if (strcmp(formulation,"cfosls") == 0 )
  {
       invS = new HypreBoomerAMG(*Schur);
       ((HypreBoomerAMG *)invS)->SetPrintLevel(0);
       ((HypreBoomerAMG *)invS)->iterative_mode = false;
  }

  BlockDiagonalPreconditioner prec(block_trueOffsets);
  if (prec_option > 0)
  {
      tempblknum = 0;
      prec.SetDiagonalBlock(tempblknum, invA);
      tempblknum++;
      if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
      {
          prec.SetDiagonalBlock(tempblknum, invC);
          tempblknum++;
      }
      if (strcmp(formulation,"cfosls") == 0)
           prec.SetDiagonalBlock(tempblknum, invS);

      if (verbose)
          std::cout << "Preconditioner built in " << chrono.RealTime() << "s. \n";
  }
  else
      if (verbose)
          cout << "No preconditioner is used. \n";

  // 12. Solve the linear system with MINRES.
  //     Check the norm of the unpreconditioned residual.

  chrono.Clear();
  chrono.Start();
  MINRESSolver solver(MPI_COMM_WORLD);
  solver.SetAbsTol(atol);
  solver.SetRelTol(rtol);
  solver.SetMaxIter(max_iter);
  solver.SetOperator(*CFOSLSop);
  if (prec_option > 0)
       solver.SetPreconditioner(prec);
  solver.SetPrintLevel(0);
  trueX = 0.0;

  chrono.Clear();
  chrono.Start();
#ifdef NONHOMO_TEST
  solver.Mult(trueRhs2, trueX);
#else
  solver.Mult(trueRhs, trueX);
#endif
  //solver.Mult(trueRhs, trueX);
  chrono.Stop();

  if (verbose)
  {
     if (solver.GetConverged())
        std::cout << "MINRES converged in " << solver.GetNumIterations()
                  << " iterations with a residual norm of " << solver.GetFinalNorm() << ".\n";
     else
        std::cout << "MINRES did not converge in " << solver.GetNumIterations()
                  << " iterations. Residual norm is " << solver.GetFinalNorm() << ".\n";
     std::cout << "MINRES solver took " << chrono.RealTime() << "s. \n";
  }

  // checking boundary conditions

  /*
  int ngroups = pmesh->GetNGroups();

  for (int i = 0; i < num_procs; ++i)
  {
      if (myid == i)
      {
          std::cout << "I am " << myid << "\n";
          std::set<int> shared_vertdofs;

          for (int grind = 0; grind < ngroups; ++grind)
          {
              int ngroupverts = pmesh->GroupNVertices(grind);
              std::cout << "ngroupverts = " << ngroupverts << "\n";
              Array<int> dofs;
              for (int faceind = 0; faceind < ngroupverts; ++faceind)
              {
                  H1_space->GetSharedFaceDofs(grind, faceind, dofs);
                  for (int dofind = 0; dofind < dofs.Size(); ++dofind)
                  {
                      shared_vertdofs.insert(H1_space->GetGlobalTDofNumber(dofs[dofind]));
                  }
              }
          }

          std::cout << "shared vertices tdofs \n";
          std::set<int>::iterator it;
          for ( it = shared_vertdofs.begin(); it != shared_vertdofs.end(); it++ )
          {
              std::cout << *it << " ";
          }

          std::cout << "\n" << std::flush;
      }
      MPI_Barrier(comm);
  } // end fo loop over all processors, one after another
  */

  Vector sigma_exact_truedofs(Sigma_space->TrueVSize());
  sigma_exact->ParallelProject(sigma_exact_truedofs);

  Array<int> EssBnd_tdofs_sigma;
  Sigma_space->GetEssentialTrueDofs(ess_bdrSigma, EssBnd_tdofs_sigma);

  for (int i = 0; i < EssBnd_tdofs_sigma.Size(); ++i)
  {
      SparseMatrix A_diag;
      A->GetDiag(A_diag);

      SparseMatrix DT_diag;
      DT->GetDiag(DT_diag);

      int tdof = EssBnd_tdofs_sigma[i];
      double value_ex = sigma_exact_truedofs[tdof];
      double value_com = trueX.GetBlock(0)[tdof];

      if (fabs(value_ex - value_com) > ZEROTOL)
      {
          std::cout << "bnd condition is violated for sigma, tdof = " << tdof << " exact value = "
                    << value_ex << ", value_com = " << value_com << ", diff = " << value_ex - value_com << "\n";
          std::cout << "rhs side at this tdof = " << trueRhs.GetBlock(0)[tdof] << "\n";
          std::cout << "rhs side2 at this tdof = " << trueRhs2.GetBlock(0)[tdof] << "\n";
          std::cout << "bnd at this tdof = " << trueBnd.GetBlock(0)[tdof] << "\n";
          std::cout << "row entries of A matrix: \n";
          int * A_rowcols = A_diag.GetRowColumns(tdof);
          double * A_rowentries = A_diag.GetRowEntries(tdof);
          for (int j = 0; j < A_diag.RowSize(tdof); ++j)
              std::cout << "(" << A_rowcols[j] << "," << A_rowentries[j] << ") ";
          std::cout << "\n";

          std::cout << "row entries of DT matrix: \n";
          int * DT_rowcols = DT_diag.GetRowColumns(tdof);
          double * DT_rowentries = DT_diag.GetRowEntries(tdof);
          for (int j = 0; j < DT_diag.RowSize(tdof); ++j)
              std::cout << "(" << DT_rowcols[j] << "," << DT_rowentries[j] << ") ";
          std::cout << "\n";
      }
  }

  if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
  {
      SparseMatrix C_diag;
      C->GetDiag(C_diag);

      SparseMatrix B_diag;
      B->GetDiag(B_diag);

      Vector S_exact_truedofs(S_space->TrueVSize());
      S_exact->ParallelProject(S_exact_truedofs);

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
              std::cout << "rhs side at this tdof = " << trueRhs.GetBlock(1)[tdof] << "\n";
              std::cout << "rhs side2 at this tdof = " << trueRhs2.GetBlock(1)[tdof] << "\n";
              std::cout << "bnd at this tdof = " << trueBnd.GetBlock(1)[tdof] << "\n";
              std::cout << "row entries of C matrix: \n";
              int * C_rowcols = C_diag.GetRowColumns(tdof);
              double * C_rowentries = C_diag.GetRowEntries(tdof);
              for (int j = 0; j < C_diag.RowSize(tdof); ++j)
                  std::cout << "(" << C_rowcols[j] << "," << C_rowentries[j] << ") ";
              std::cout << "\n";
              std::cout << "row entries of B matrix: \n";
              int * B_rowcols = B_diag.GetRowColumns(tdof);
              double * B_rowentries = B_diag.GetRowEntries(tdof);
              for (int j = 0; j < B_diag.RowSize(tdof); ++j)
                  std::cout << "(" << B_rowcols[j] << "," << B_rowentries[j] << ") ";
              std::cout << "\n";

          }
      }
  }


  ParGridFunction * sigma = new ParGridFunction(Sigma_space);
  sigma->Distribute(&(trueX.GetBlock(0)));

  ParGridFunction * S = new ParGridFunction(S_space);
  if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
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
  double norm_sigma = ComputeGlobalLpNorm(2, *(Mytest.sigma), *pmesh, irs);
  if (verbose)
      cout << "|| sigma - sigma_ex || / || sigma_ex || = " << err_sigma / norm_sigma << endl;

  DiscreteLinearOperator Div(Sigma_space, L2_space);
  Div.AddDomainInterpolator(new DivergenceInterpolator());
  ParGridFunction DivSigma(L2_space);
  Div.Assemble();
  Div.Mult(*sigma, DivSigma);

  double err_div = DivSigma.ComputeL2Error(*(Mytest.scalardivsigma),irs);
  double norm_div = ComputeGlobalLpNorm(2, *(Mytest.scalardivsigma), *pmesh, irs);

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
  double norm_S = ComputeGlobalLpNorm(2, (*Mytest.scalarS), *pmesh, irs);
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
      auto *N_space = new ParFiniteElementSpace(pmesh, hcurl_coll);

      DiscreteLinearOperator Grad(S_space, N_space);
      Grad.AddDomainInterpolator(new GradientInterpolator());
      ParGridFunction GradS(N_space);
      Grad.Assemble();
      Grad.Mult(*S, GradS);

      if (numsol != -34 && verbose)
          std::cout << "For this norm we are grad S for S from numsol = -34 \n";
      VectorFunctionCoefficient GradS_coeff(dim, uFunTest_ex_gradxt);
      double err_GradS = GradS.ComputeL2Error(GradS_coeff, irs);
      double norm_GradS = ComputeGlobalLpNorm(2, GradS_coeff, *pmesh, irs);
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
      trueRhs_nobnd = trueRhs;

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

      trueRhs.GetBlock(numblocks - 1) -= massform;
      double mass_loss_loc = trueRhs.GetBlock(numblocks - 1).Norml1();
      double mass_loss;
      MPI_Reduce(&mass_loss_loc, &mass_loss, 1,
                 MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      if (verbose)
          cout << "Sum of local mass loss = " << mass_loss << "\n";
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


  //MPI_Finalize();
  //return 0;

  if (verbose)
    std::cout << "Checking a single solve from a one TimeSlabHyper instance "
                 "created for the entire domain \n";

  {
      //TimeSlabHyper * timeslab_test = new TimeSlabHyper (*pmesh, formulation, space_for_S, space_for_sigma);
      int pref_lvls_tslab = 1; // doesn't work with other values
      TimeSlabHyper * timeslab_test = new TimeSlabHyper (*pmeshbase, 0.0, tau, Nt, pref_lvls_tslab,
                                                         formulation, space_for_S, space_for_sigma);

      int init_cond_size = timeslab_test->GetInitCondSize(0);
      std::vector<std::pair<int,int> > * tdofs_link = timeslab_test->GetTdofsLink(0);
      Vector Xinit(init_cond_size);
      if (strcmp(space_for_S,"H1") == 0) // S is present
      {
          ParGridFunction * S_exact = new ParGridFunction(timeslab_test->Get_S_space());
          S_exact->ProjectCoefficient(*Mytest.scalarS);
          Vector S_exact_truedofs(timeslab_test->Get_S_space()->TrueVSize());
          S_exact->ParallelProject(S_exact_truedofs);

          for (int i = 0; i < init_cond_size; ++i)
          {
              int tdof_bot = (*tdofs_link)[i].first;
              Xinit[i] = S_exact_truedofs[tdof_bot];
          }
      }
      else
      {
          ParGridFunction * sigma_exact = new ParGridFunction(timeslab_test->Get_Sigma_space());
          sigma_exact->ProjectCoefficient(*Mytest.sigma);
          Vector sigma_exact_truedofs(timeslab_test->Get_Sigma_space()->TrueVSize());
          sigma_exact->ParallelProject(sigma_exact_truedofs);

          for (int i = 0; i < init_cond_size; ++i)
          {
              int tdof_bot = (*tdofs_link)[i].first;
              Xinit[i] = sigma_exact_truedofs[tdof_bot];
          }
      }
      //Xinit.Print();

      Vector Xout(init_cond_size);

      timeslab_test->Solve(Xinit, Xout);

      Vector Xout_exact(init_cond_size);

      // checking the error at the top boundary
      if (strcmp(space_for_S,"H1") == 0) // S is present
      {
          ParGridFunction * S_exact = new ParGridFunction(timeslab_test->Get_S_space());
          S_exact->ProjectCoefficient(*Mytest.scalarS);
          Vector S_exact_truedofs(timeslab_test->Get_S_space()->TrueVSize());
          S_exact->ParallelProject(S_exact_truedofs);

          for (int i = 0; i < init_cond_size; ++i)
          {
              int tdof_top = (*tdofs_link)[i].second;
              Xout_exact[i] = S_exact_truedofs[tdof_top];
          }
      }
      else
      {
          ParGridFunction * sigma_exact = new ParGridFunction(timeslab_test->Get_Sigma_space());
          sigma_exact->ProjectCoefficient(*Mytest.sigma);
          Vector sigma_exact_truedofs(timeslab_test->Get_Sigma_space()->TrueVSize());
          sigma_exact->ParallelProject(sigma_exact_truedofs);

          for (int i = 0; i < init_cond_size; ++i)
          {
              int tdof_top = (*tdofs_link)[i].second;
              Xout_exact[i] = sigma_exact_truedofs[tdof_top];
          }
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

      delete timeslab_test;
  }

  MPI_Finalize();
  return 0;

  if (verbose)
    std::cout << "Checking a sequential solve within several TimeSlabHyper instances \n";

  {
      int nslabs = 2;
      std::vector<TimeSlabHyper*> timeslabs(nslabs);
      //pmeshbase->UniformRefinement();
      double slab_tau = 0.125;
      int slab_width = 4; // in time steps (as time intervals) withing a single time slab
      double tinit_tslab = 0.0;
      for (int tslab = 0; tslab < nslabs; ++tslab )
      {
          timeslabs[tslab] = new TimeSlabHyper (*pmeshbase, tinit_tslab, slab_tau, slab_width, 0,
                                                formulation, space_for_S, space_for_sigma);
          tinit_tslab += slab_tau * slab_width;
      }

      MFEM_ASSERT(fabs(tinit_tslab - 1.0) < 1.0e-14, "The slabs should cover the time interval "
                                                    "[0,1] but the upper bound doesn't match \n");

      Vector Xinit;
      // initializing the input boundary condition for the first vector
      int init_cond_size = timeslabs[0]->GetInitCondSize(0);
      std::vector<std::pair<int,int> > * tdofs_link = timeslabs[0]->GetTdofsLink(0);
      Xinit.SetSize(init_cond_size);

      if (strcmp(space_for_S,"H1") == 0) // S is present
      {
          ParGridFunction * S_exact = new ParGridFunction(timeslabs[0]->Get_S_space());
          S_exact->ProjectCoefficient(*Mytest.scalarS);
          Vector S_exact_truedofs(timeslabs[0]->Get_S_space()->TrueVSize());
          S_exact->ParallelProject(S_exact_truedofs);

          for (int i = 0; i < init_cond_size; ++i)
          {
              int tdof_bot = (*tdofs_link)[i].first;
              Xinit[i] = S_exact_truedofs[tdof_bot];
          }
      }
      else
      {
          ParGridFunction * sigma_exact = new ParGridFunction(timeslabs[0]->Get_Sigma_space());
          sigma_exact->ProjectCoefficient(*Mytest.sigma);
          Vector sigma_exact_truedofs(timeslabs[0]->Get_Sigma_space()->TrueVSize());
          sigma_exact->ParallelProject(sigma_exact_truedofs);

          for (int i = 0; i < init_cond_size; ++i)
          {
              int tdof_bot = (*tdofs_link)[i].first;
              Xinit[i] = sigma_exact_truedofs[tdof_bot];
          }
      }

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
          timeslabs[tslab]->Solve(Xinit, Xout);
          Xinit = Xout;
          if (strcmp(space_for_S,"L2") == 0)
              Xinit *= -1.0;

          Vector Xout_exact(init_cond_size);

          // checking the error at the top boundary
          if (strcmp(space_for_S,"H1") == 0) // S is present
          {
              ParGridFunction * S_exact = new ParGridFunction(timeslabs[tslab]->Get_S_space());
              S_exact->ProjectCoefficient(*Mytest.scalarS);
              Vector S_exact_truedofs(timeslabs[tslab]->Get_S_space()->TrueVSize());
              S_exact->ParallelProject(S_exact_truedofs);

              for (int i = 0; i < init_cond_size; ++i)
              {
                  int tdof_top = (*tdofs_link)[i].second;
                  Xout_exact[i] = S_exact_truedofs[tdof_top];
              }
          }
          else
          {
              ParGridFunction * sigma_exact = new ParGridFunction(timeslabs[tslab]->Get_Sigma_space());
              sigma_exact->ProjectCoefficient(*Mytest.sigma);
              Vector sigma_exact_truedofs(timeslabs[tslab]->Get_Sigma_space()->TrueVSize());
              sigma_exact->ParallelProject(sigma_exact_truedofs);

              for (int i = 0; i < init_cond_size; ++i)
              {
                  int tdof_top = (*tdofs_link)[i].second;
                  Xout_exact[i] = sigma_exact_truedofs[tdof_top];
              }
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

  if (visualization && nDimensions < 4)
  {
     char vishost[] = "localhost";
     int  visport   = 19916;
     socketstream u_sock(vishost, visport);
     u_sock << "parallel " << num_procs << " " << myid << "\n";
     u_sock.precision(8);
     u_sock << "solution\n" << *pmesh << *sigma_exact << "window_title 'sigma_exact'"
            << endl;
     // Make sure all ranks have sent their 'u' solution before initiating
     // another set of GLVis connections (one from each rank):


     socketstream uu_sock(vishost, visport);
     uu_sock << "parallel " << num_procs << " " << myid << "\n";
     uu_sock.precision(8);
     uu_sock << "solution\n" << *pmesh << *sigma << "window_title 'sigma'"
            << endl;

     *sigma_exact -= *sigma;

     socketstream uuu_sock(vishost, visport);
     uuu_sock << "parallel " << num_procs << " " << myid << "\n";
     uuu_sock.precision(8);
     uuu_sock << "solution\n" << *pmesh << *sigma_exact << "window_title 'difference for sigma'"
            << endl;

     socketstream s_sock(vishost, visport);
     s_sock << "parallel " << num_procs << " " << myid << "\n";
     s_sock.precision(8);
     MPI_Barrier(pmesh->GetComm());
     s_sock << "solution\n" << *pmesh << *S_exact << "window_title 'S_exact'"
             << endl;

     socketstream ss_sock(vishost, visport);
     ss_sock << "parallel " << num_procs << " " << myid << "\n";
     ss_sock.precision(8);
     MPI_Barrier(pmesh->GetComm());
     ss_sock << "solution\n" << *pmesh << *S << "window_title 'S'"
             << endl;

     *S_exact -= *S;
     socketstream sss_sock(vishost, visport);
     sss_sock << "parallel " << num_procs << " " << myid << "\n";
     sss_sock.precision(8);
     MPI_Barrier(pmesh->GetComm());
     sss_sock << "solution\n" << *pmesh << *S_exact
              << "window_title 'difference for S'" << endl;

     MPI_Barrier(pmesh->GetComm());
  }

  // 17. Free the used memory.
  //delete fform;
  //delete CFOSLSop;
  //delete A;

  //delete Ablock;
  if (strcmp(space_for_S,"H1") == 0) // S was from H1
       delete H1_space;
  delete L2_space;
  delete Hdiv_space;
  if (strcmp(space_for_sigma,"H1") == 0) // S was from H1
       delete H1vec_space;
  delete l2_coll;
  delete h1_coll;
  delete hdiv_coll;

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

// eltype must be "linearH1" or "RT0", for any other finite element the code doesn't work
// the fespace must correspond to the eltype provided
// bot_to_top_bels is the link between boundary elements (at the bottom and at the top)
// which can be taken out of ParMeshCyl

std::vector<std::pair<int,int> >* CreateBotToTopDofsLink(const char * eltype, FiniteElementSpace& fespace,
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


