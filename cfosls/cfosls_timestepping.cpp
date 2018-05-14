#include <iostream>
#include "testhead.hpp"

using namespace std;

namespace mfem
{

void FOSLSCylProblem::ConstructRestrictions()
{
    Restrict_bot = CreateRestriction("bot", *pfes[init_cond_block], tdofs_link);
    Restrict_top = CreateRestriction("top", *pfes[init_cond_block], tdofs_link);
}

void FOSLSCylProblem::ConstructTdofLink()
{
    std::vector<std::pair<int,int> > * dofs_link;

    switch (init_cond_space)
    {
    case HDIV:
    {
        dofs_link = CreateBotToTopDofsLink("RT0", *pfes[init_cond_block], pmeshcyl.bot_to_top_bels);
        break;
    }
    case H1:
    {
        dofs_link = CreateBotToTopDofsLink("linearH1", *pfes[init_cond_block], pmeshcyl.bot_to_top_bels);
        break;
    }
    default:
        MFEM_ABORT("Unsupported space name for initial condition variable");
        return;
    }

    tdofs_link.reserve(dofs_link->size());

    int count = 0;
    for ( unsigned int i = 0; i < dofs_link->size(); ++i )
    {
        //std::cout << "<" << it->first << ", " << it->second << "> \n";
        int dof1 = (*dofs_link)[i].first;
        int dof2 = (*dofs_link)[i].second;
        int tdof1 = pfes[init_cond_block]->GetLocalTDofNumber(dof1);
        int tdof2 = pfes[init_cond_block]->GetLocalTDofNumber(dof2);
        //std::cout << "corr. dof pair: <" << dof1 << "," << dof2 << ">\n";
        //std::cout << "corr. tdof pair: <" << tdof1 << "," << tdof2 << ">\n";
        if (tdof1 * tdof2 < 0)
            MFEM_ABORT( "unsupported case: tdof1 and tdof2 belong to different processors! \n");

        if (tdof1 > -1)
        {
            tdofs_link.push_back(std::pair<int,int>(tdof1, tdof2));
            ++count;
        }
        else
        { /*std::cout << "Ignored dofs pair which are not own tdofs \n";*/ }
    }

    delete dofs_link;
}

void FOSLSCylProblem::ExtractAtBase(const char * top_or_bot, const Vector &x, Vector& base_tdofs) const
{
    MFEM_ASSERT(strcmp(top_or_bot,"bot") == 0 || strcmp(top_or_bot,"top") == 0,
                "In GetExactBase() top_or_bot must equal 'top' or 'bot'! \n");

    BlockVector x_viewer(x.GetData(), blkoffsets_true);
    for (unsigned int i = 0; i < tdofs_link.size(); ++i)
    {
        int tdof = ( strcmp(top_or_bot,"bot") == 0 ? tdofs_link[i].first : tdofs_link[i].second);
        base_tdofs[i] = x_viewer.GetBlock(init_cond_block)[tdof];
    }
}

void FOSLSCylProblem::SetAtBase(const char * top_or_bot, const Vector &base_tdofs, Vector& vec) const
{
    MFEM_ASSERT(strcmp(top_or_bot,"bot") == 0 || strcmp(top_or_bot,"top") == 0,
                "In SetAtBase() top_or_bot must equal 'top' or 'bot'! \n");

    MFEM_ASSERT(base_tdofs.Size() == GetInitCondSize(), "Incorrect input size in SetAtBase()");

    BlockVector vec_viewer(vec.GetData(), blkoffsets_true);
    for (unsigned int i = 0; i < tdofs_link.size(); ++i)
    {
        int tdof = ( strcmp(top_or_bot,"bot") == 0 ? tdofs_link[i].first : tdofs_link[i].second);
        vec_viewer.GetBlock(init_cond_block)[tdof] = base_tdofs[i];
    }

}


Vector& FOSLSCylProblem::ExtractAtBase(const char * top_or_bot, const Vector &x) const
{
    Vector * res = new Vector(tdofs_link.size());

    ExtractAtBase(top_or_bot, x, *res);

    return *res;
}


void FOSLSCylProblem::ExtractTopTdofs(const Vector &x, Vector& bnd_tdofs_top) const
{
    BlockVector x_viewer(x.GetData(), blkoffsets_true);
    for (unsigned int i = 0; i < tdofs_link.size(); ++i)
    {
        int tdof_top = tdofs_link[i].second;
        bnd_tdofs_top[i] = x_viewer.GetBlock(init_cond_block)[tdof_top];
    }
}

void FOSLSCylProblem::ExtractBotTdofs(const Vector& x, Vector& bnd_tdofs_bot) const
{
    BlockVector x_viewer(x.GetData(), blkoffsets_true);

    for (unsigned int i = 0; i < tdofs_link.size(); ++i)
    {
        int tdof_top = tdofs_link[i].first;
        bnd_tdofs_bot[i] = x_viewer.GetBlock(init_cond_block)[tdof_top];
    }
}

void FOSLSCylProblem::CorrectFromInitCnd(const Operator& op,
                                            const Vector& bnd_tdofs_bot, Vector& vec) const
{
    int init_cond_size = tdofs_link.size();// init_cond_size_lvls[lvl];

    if (bnd_tdofs_bot.Size() != init_cond_size)
    {
        std::cerr << "Error: sizes mismatch, input vector's size = " <<  bnd_tdofs_bot.Size()
                  << ", expected: " << init_cond_size << "\n";
        MFEM_ABORT("Wrong size of the input vector");
    }

    // using an alternative way of imposing boundary conditions on the right hand side
    BlockVector trueBnd(blkoffsets_true);
    trueBnd = 0.0;

    for (unsigned int i = 0; i < tdofs_link.size(); ++i)
    {
        int tdof_bot = tdofs_link[i].first;
        trueBnd.GetBlock(init_cond_block)[tdof_bot] = bnd_tdofs_bot[i];
    }

    BlockVector trueBndCor(blkoffsets_true);
    trueBndCor = 0.0;

    //trueBnd.Print();

    op.Mult(trueBnd, trueBndCor);

    // vec := vec - op * initial_condition
    vec -= trueBndCor;

    // correcting vec entries for bdr tdofs
    BlockVector vec_viewer(vec.GetData(), blkoffsets_true);

    for (int blk = 0; blk < fe_formul.Nblocks(); ++blk)
    {
        Array<int>& essbdr_attrs = bdr_conds.GetBdrAttribs(blk);

        Array<int> ess_bnd_tdofs;
        pfes[blk]->GetEssentialTrueDofs(essbdr_attrs, ess_bnd_tdofs);

        for (int i = 0; i < ess_bnd_tdofs.Size(); ++i)
        {
            int tdof = ess_bnd_tdofs[i];
            vec_viewer.GetBlock(blk)[tdof] = trueBnd.GetBlock(blk)[tdof];
        }
    }

}

void FOSLSCylProblem::Solve(const Vector& rhs, const Vector& bnd_tdofs_bot,
                            Vector& bnd_tdofs_top, bool compute_error) const
{
    // copying righthand side
    BlockVector rhs_viewer(rhs.GetData(), blkoffsets_true);

    *trueRhs = rhs_viewer;

    //std::cout << "before \n";
    //trueRhs->Print();

    //std::cout << "bnd_tdofs_bot \n";
    //bnd_tdofs_bot.Print();

    // correcting rhs with the given initial condition
    CorrectFromInitCnd(bnd_tdofs_bot, *trueRhs);

    //std::cout << "after \n";
    //trueRhs->Print();

    // solving the system
    FOSLSProblem::Solve(verbose, compute_error);

    // computing the outputs: full solution vector and tdofs
    // (for the possible next cylinder's initial condition at the top interface)

    ExtractTopTdofs(*trueX, bnd_tdofs_top);
}

void FOSLSCylProblem::Solve(const Vector& bnd_tdofs_bot, Vector &bnd_tdofs_top, bool compute_error) const
{
    // 1. compute trueRhs as assembled linear forms of the rhs
    ComputeAnalyticalRhs();

    Solve(*trueRhs, bnd_tdofs_bot, bnd_tdofs_top, compute_error);
}

void FOSLSCylProblem::Solve(const Vector& rhs, const Vector& bnd_tdofs_bot,
                            Vector& sol, Vector& bnd_tdofs_top, bool compute_error) const
{
    Solve(rhs, bnd_tdofs_bot, bnd_tdofs_top, compute_error);

    BlockVector viewer_out(sol.GetData(), blkoffsets_true);
    viewer_out = *trueX;
}


void FOSLSCylProblem::ConvertInitCndToFullVector(const Vector& vec_in, Vector& vec_out)
{
    MFEM_ASSERT(vec_in.Size() == (int)(tdofs_link.size()), "Input vector size mismatch the link size \n");

    BlockVector viewer(vec_out.GetData(), blkoffsets_true);
    vec_out = 0.0;

    int index = fe_formul.GetFormulation()->GetUnknownWithInitCnd();

    for (int i = 0; i < vec_in.Size(); ++i)
    {
        int tdof = tdofs_link[i].first;
        viewer.GetBlock(index)[tdof] = vec_in[i];
    }
}

// it is assumed that CFOSLSop_nobnd was already created
// Takes the vector which stores inhomogeneous bdr values and zeros
// and computes the new vector which has the same boundary values
// but also a contribution from inhomog. bdr conditions (input) to the other rhs values
void FOSLSCylProblem::ConvertBdrCndIntoRhs(const Vector& vec_in, Vector& vec_out)
{
    vec_out = 0.0;
    CFOSLSop_nobnd->Mult(vec_in, vec_out);

    BlockVector viewer(vec_out.GetData(), blkoffsets_true);

    for (int blk = 0; blk < fe_formul.Nblocks(); ++blk)
    {
        Array<int>& essbdr_attrs = bdr_conds.GetBdrAttribs(blk);

        Array<int> ess_tdofs;
        pfes[blk]->GetEssentialTrueDofs(essbdr_attrs, ess_tdofs);

        for (int i = 0; i < ess_tdofs.Size(); ++i)
        {
            int tdof = ess_tdofs[i];
            viewer.GetBlock(blk)[tdof] = vec_in[tdof];
        }

    }
}


void FOSLSCylProblem::CorrectFromInitCond(const Vector& init_cond, Vector& vec_out, double coeff)
{
    //Vector * initcond_fullvec = new Vector(GlobalTrueProblemSize());
    //ConvertInitCndToFullVector(init_cond, *initcond_fullvec);


    //Vector * rhs_correction = new Vector(GlobalTrueProblemSize());
    //ConvertBdrCndIntoRhs(*initcond_fullvec, *rhs_correction);

    //*rhs_correction *= coeff;
    //vec_out += *rhs_correction;

    ConvertInitCndToFullVector(init_cond, *temp_vec1);

    ConvertBdrCndIntoRhs(*temp_vec1, *temp_vec2);

    *temp_vec2 *= coeff;
    vec_out += *temp_vec2;

    //delete initcond_fullvec;
    //delete rhs_correction;
}


Vector* FOSLSCylProblem::GetExactBase(const char * top_or_bot)
{
    MFEM_ASSERT(strcmp(top_or_bot,"bot") == 0 || strcmp(top_or_bot,"top") == 0,
                "In GetExactBase() top_or_bot must equal 'top' or 'bot'! \n");

    // index of the unknown with boundary condition
    int index = fe_formul.GetFormulation()->GetUnknownWithInitCnd();
    FOSLS_test * test = fe_formul.GetFormulation()->GetTest();

    ParFiniteElementSpace * pfespace = pfes[index];
    ParGridFunction * exsol_pgfun = new ParGridFunction(pfespace);

    int coeff_index = fe_formul.GetFormulation()->GetPair(index).second;
    MFEM_ASSERT(coeff_index >= 0, "Value of coeff_index must be nonnegative at least \n");
    switch (fe_formul.GetFormulation()->GetPair(index).first)
    {
    case 0: // function coefficient
        exsol_pgfun->ProjectCoefficient(*test->GetFuncCoeff(coeff_index));
        break;
    case 1: // vector function coefficient
        exsol_pgfun->ProjectCoefficient(*test->GetVecCoeff(coeff_index));
        break;
    default:
        {
            MFEM_ABORT("Unsupported type of coefficient for the call to ProjectCoefficient");
        }
        break;
    }

    Vector exsol_tdofs(pfespace->TrueVSize());
    exsol_pgfun->ParallelProject(exsol_tdofs);

    int init_cond_size = tdofs_link.size();

    Vector * Xout_exact = new Vector(init_cond_size);

    for (int i = 0; i < init_cond_size; ++i)
    {
        int tdof = ( strcmp(top_or_bot,"bot") == 0 ? tdofs_link[i].first : tdofs_link[i].second);
        (*Xout_exact)[i] = exsol_tdofs[tdof];
    }

    delete exsol_pgfun;

    return Xout_exact;
}

void FOSLSCylProblem::ComputeErrorAtBase(const char * top_or_bot, const Vector& base_vec)
{
    MFEM_ASSERT(strcmp(top_or_bot,"bot") == 0 || strcmp(top_or_bot,"top") == 0,
                "In GetExactBase() top_or_bot must equal 'top' or 'bot'! \n");

    int init_cond_size = tdofs_link.size();
    MFEM_ASSERT(base_vec.Size() == init_cond_size, "Input vector size mismatch the length of tdofs link");

    Vector * Xout_exact = GetExactBase(top_or_bot);

    Vector Xout_error(init_cond_size);
    Xout_error = base_vec;
    Xout_error -= *Xout_exact;

    std::cout << "|| Xout  - Xout_exact || = " << Xout_error.Norml2() / sqrt (Xout_error.Size()) << "\n";
    std::cout << "|| Xout  - Xout_exact || / || Xout_exact || = " << (Xout_error.Norml2() / sqrt (Xout_error.Size())) /
                 (Xout_exact->Norml2() / sqrt (Xout_exact->Size()))<< "\n";

    delete Xout_exact;
}

//#######################################################################################################################

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

void TimeCylHyper::ComputeAnalyticalRhs(int lvl)
{
    MFEM_ABORT ("TimeCylHyper::ComputeAnalyticalRhs hasn't been implemented yet");
}


Vector* TimeCylHyper::GetExactBase(const char * top_or_bot, int level)
{
    if (strcmp(top_or_bot,"bot") != 0 && strcmp(top_or_bot,"top") != 0 )
    {
        MFEM_ABORT("In TimeCylHyper::GetExactBase() top_or_bot must equal 'top' or 'bot'! \n");
    }

    // checking the error at the top boundary
    ParFiniteElementSpace * testfespace;
    ParGridFunction * sol_exact;

    Transport_test Mytest(dim, numsol);

    if (strcmp(space_for_S,"H1") == 0) // S is present
    {
        testfespace = Get_S_space(level);
        sol_exact = new ParGridFunction(testfespace);
        sol_exact->ProjectCoefficient(*Mytest.scalarS);
    }
    else
    {
        testfespace = Get_Sigma_space(level);
        sol_exact = new ParGridFunction(testfespace);
        sol_exact->ProjectCoefficient(*Mytest.sigma);
    }

    Vector sol_exact_truedofs(testfespace->TrueVSize());
    sol_exact->ParallelProject(sol_exact_truedofs);

    Vector * Xout_exact = new Vector(GetInitCondSize(level));

    std::vector<std::pair<int,int> > * tdofs_link = GetTdofsLink(level);

    for (int i = 0; i < ProblemSize(level); ++i)
    {
        int tdof_top = ( strcmp(top_or_bot,"bot") == 0 ? (*tdofs_link)[i].first : (*tdofs_link)[i].second);
        (*Xout_exact)[i] = sol_exact_truedofs[tdof_top];
    }

    delete sol_exact;

    return Xout_exact;
}



TimeCylHyper::~TimeCylHyper()
{
    for (unsigned int i = 0; i < Sigma_space_lvls.size(); ++i)
        delete Sigma_space_lvls[i];
    for (unsigned int i = 0; i < S_space_lvls.size(); ++i)
        delete S_space_lvls[i];
    //if (strcmp(space_for_S,"H1") == 0)
        //for (unsigned int i = 0; i < L2_space_lvls.size(); ++i)
            //delete L2_space_lvls[i];
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
    /*
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
    */


    // TODO: Add a proper destructor for the hierarchy class
    delete hierarchy;
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
            hierarchy->GetTrueP_bndtop_H1(lvl)->Mult(vec_in, vec_out);
            //TrueP_bndtop_H1_lvls[lvl]->Mult(vec_in, vec_out);
        else if (strcmp(top_or_bot, "bot") == 0)
            hierarchy->GetTrueP_bndbot_H1(lvl)->Mult(vec_in, vec_out);
            //TrueP_bndbot_H1_lvls[lvl]->Mult(vec_in, vec_out);
        else
        {
            MFEM_ABORT("In TimeCylHyper::InterpolateAtBase() top_or_bot must be 'top' or 'bot'!");
        }
    }
    else
    {
        if (strcmp(top_or_bot, "top") == 0)
            hierarchy->GetTrueP_bndtop_Hdiv(lvl)->Mult(vec_in, vec_out);
            //TrueP_bndtop_Hdiv_lvls[lvl]->Mult(vec_in, vec_out);
        else if (strcmp(top_or_bot, "bot") == 0)
            hierarchy->GetTrueP_bndbot_Hdiv(lvl)->Mult(vec_in, vec_out);
            //TrueP_bndbot_Hdiv_lvls[lvl]->Mult(vec_in, vec_out);
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
            hierarchy->GetTrueP_bndtop_H1(lvl - 1)->MultTranspose(vec_in, vec_out);
            //TrueP_bndtop_H1_lvls[lvl - 1]->MultTranspose(vec_in, vec_out);
        else if (strcmp(top_or_bot, "bot") == 0)
            hierarchy->GetTrueP_bndbot_H1(lvl - 1)->MultTranspose(vec_in, vec_out);
            //TrueP_bndbot_H1_lvls[lvl - 1]->MultTranspose(vec_in, vec_out);
        else
        {
            MFEM_ABORT("In TimeCylHyper::RestrictAtBase() top_or_bot must be 'top' or 'bot'!");
        }
    }
    else
    {
        if (strcmp(top_or_bot, "top") == 0)
            hierarchy->GetTrueP_bndtop_Hdiv(lvl - 1)->MultTranspose(vec_in, vec_out);
            //TrueP_bndtop_Hdiv_lvls[lvl - 1]->MultTranspose(vec_in, vec_out);
        else if (strcmp(top_or_bot, "bot") == 0)
            hierarchy->GetTrueP_bndbot_Hdiv(lvl - 1)->MultTranspose(vec_in, vec_out);
            //TrueP_bndbot_Hdiv_lvls[lvl - 1]->MultTranspose(vec_in, vec_out);
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
        if (vec_in.Size() != hierarchy->GetLinksize_H1(lvl))
        {
            MFEM_ABORT("Size of vec_in in ConvertInitCndToFullVector differs from the size of tdofs_link_H1_lvls \n");
        }

        for (int i = 0; i < vec_in.Size(); ++i)
        {
            int tdof = (*hierarchy->GetTdofs_H1_link(lvl))[i].first;// tdofs_link_H1_lvls[lvl][i].first;
            viewer.GetBlock(1)[tdof] = vec_in[i];
        }
    }
    else
    {
        if (vec_in.Size() != hierarchy->GetLinksize_Hdiv(lvl))
        {
            MFEM_ABORT("Size of vec_in in ConvertInitCndToFullVector differs from the size of tdofs_link_Hdiv_lvls \n");
        }

        for (int i = 0; i < vec_in.Size(); ++i)
        {
            int tdof = (*hierarchy->GetTdofs_Hdiv_link(lvl))[i].first; //tdofs_link_Hdiv_lvls[lvl][i].first;
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

    ParMeshCyl * pmeshtsl = hierarchy->GetPmeshcyl(lvl);//pmeshtsl_lvls[lvl];

    ParFiniteElementSpace * S_space = S_space_lvls[lvl];
    ParFiniteElementSpace * Sigma_space = Sigma_space_lvls[lvl];
    ParFiniteElementSpace * L2_space = hierarchy->GetL2_space(lvl);//L2_space_lvls[lvl];

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
                              const char *Formulation, const char *Space_for_S,
                            const char *Space_for_sigma, int Numsol)
    : TimeCyl(Pmeshbase, T_init, Tau, Nt), ref_lvls(Ref_lvls),
      formulation(Formulation), space_for_S(Space_for_S), space_for_sigma(Space_for_sigma),
      numsol(Numsol)
{
    InitProblem(numsol);
}

TimeCylHyper::TimeCylHyper (ParMeshCyl& Pmeshtsl, int Ref_Lvls,
                              const char *Formulation, const char *Space_for_S,
                            const char *Space_for_sigma, int Numsol)
    : TimeCyl(Pmeshtsl), ref_lvls(Ref_Lvls),
      formulation(Formulation), space_for_S(Space_for_S), space_for_sigma(Space_for_sigma),
      numsol(Numsol)
{
    InitProblem(Numsol);
}

void TimeCylHyper::Solve(int lvl, const Vector& rhs, Vector& sol,
                         const Vector& bnd_tdofs_bot, Vector& bnd_tdofs_top) const
{
    return Solve("regular", lvl, rhs, sol, bnd_tdofs_bot, bnd_tdofs_top);
}

// mode options:
// a) "regular" to solve with a matrix assembled from bilinear forms at corr. level
// b) "coarsened" to solve with RAP-coarsened matrix
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

    int init_cond_size = GetInitCondSize(lvl);// init_cond_size_lvls[lvl];

    if (bnd_tdofs_bot.Size() != init_cond_size || bnd_tdofs_top.Size() != init_cond_size)
    {
        std::cerr << "Error: sizes mismatch, input vector's size = " <<  bnd_tdofs_bot.Size()
                  << ", output's size = " << bnd_tdofs_top.Size() << ", expected: " << init_cond_size << "\n";
        MFEM_ABORT("Wrong size of the input and output vectors");
    }

    //BlockOperator* CFOSLSop = CFOSLSop_lvls[lvl];
    BlockOperator* CFOSLSop_nobnd = CFOSLSop_nobnd_lvls[lvl];
    ParFiniteElementSpace * S_space = S_space_lvls[lvl];
    ParFiniteElementSpace * Sigma_space = Sigma_space_lvls[lvl];
    //ParFiniteElementSpace * L2_space = hierarchy->GetL2_space(lvl);// L2_space_lvls[lvl];
    MINRESSolver * solver = solver_lvls[lvl];
    ParMeshCyl * pmeshtsl = hierarchy->GetPmeshcyl(lvl);// pmeshtsl_lvls[lvl];
    Array<int> block_trueOffsets(block_trueOffsets_lvls[lvl]->Size());
    for (int i = 0; i < block_trueOffsets.Size(); ++i)
        block_trueOffsets[i] = (*block_trueOffsets_lvls[lvl])[i];

    std::vector<std::pair<int,int> > tdofs_link_H1;
    std::vector<std::pair<int,int> > tdofs_link_Hdiv;
    if (strcmp(space_for_S, "H1") == 0)
        tdofs_link_H1 = *hierarchy->GetTdofs_H1_link(lvl); //tdofs_link_H1_lvls[lvl];
    else
        tdofs_link_Hdiv = *hierarchy->GetTdofs_Hdiv_link(lvl); //tdofs_link_Hdiv_lvls[lvl];

    Array<int> ess_bdrS(pmeshtsl->bdr_attributes.Max());
    for (unsigned int i = 0; i < ess_bdrat_S.size(); ++i)
        ess_bdrS[i] = ess_bdrat_S[i];

    Array<int> ess_bdrSigma(pmeshtsl->bdr_attributes.Max());
    for (unsigned int i = 0; i < ess_bdrat_sigma.size(); ++i)
        ess_bdrSigma[i] = ess_bdrat_sigma[i];

    //int numblocks = CFOSLSop->NumRowBlocks();
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

    int init_cond_size = GetInitCondSize(lvl);// init_cond_size_lvls[lvl];

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
    ParFiniteElementSpace * L2_space = hierarchy->GetL2_space(lvl);// L2_space_lvls[lvl];
    MINRESSolver * solver = solver_lvls[lvl];
    ParMeshCyl * pmeshtsl = hierarchy->GetPmeshcyl(lvl);// pmeshtsl_lvls[lvl];
    Array<int> block_trueOffsets(block_trueOffsets_lvls[lvl]->Size());
    for (int i = 0; i < block_trueOffsets.Size(); ++i)
        block_trueOffsets[i] = (*block_trueOffsets_lvls[lvl])[i];

    std::vector<std::pair<int,int> > tdofs_link_H1;
    std::vector<std::pair<int,int> > tdofs_link_Hdiv;
    if (strcmp(space_for_S, "H1") == 0)
        tdofs_link_H1 = *hierarchy->GetTdofs_H1_link(lvl); //tdofs_link_H1_lvls[lvl];
    else
        tdofs_link_Hdiv = *hierarchy->GetTdofs_Hdiv_link(lvl); //tdofs_link_Hdiv_lvls[lvl];

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

    //gform_nobnd->Print();

    BlockVector trueRhs_nobnd(block_trueOffsets);
    trueRhs_nobnd = 0.0;

    if (strcmp(space_for_S,"H1") == 0)
        qform_nobnd->ParallelAssemble(trueRhs_nobnd.GetBlock(1));
    gform_nobnd->ParallelAssemble(trueRhs_nobnd.GetBlock(numblocks - 1));

    //trueRhs_nobnd.Print();

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

    //trueRhs2.Print();

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

void TimeCylHyper::InitProblem(int numsol)
{
    dim = pmeshtsl->Dimension();
    comm = pmeshtsl->GetComm();

    // 1. Initialize MPI.
    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    feorder = 0;

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

    /*
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
    */

    /////////////////////////////////////////////////////////////////
    int num_lvls = ref_lvls + 1;

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

    std::vector<Array2D<HypreParMatrix*> *> Funct_hpmat_lvls(num_lvls);

    int numblocks = 1;

    if (strcmp(space_for_S,"H1") == 0)
        numblocks++;
    int numvars = numblocks;

    if (strcmp(formulation,"cfosls") == 0)
        numblocks++;

    if (verbose)
        std::cout << "Number of blocks in the formulation: " << numblocks << "\n";

    hierarchy = new GeneralCylHierarchy(num_lvls, *pmeshtsl, feorder, verbose);

    for (int l = num_lvls - 1; l >= 0; --l)
    {
        // critical for the considered problem
        if (strcmp(space_for_sigma,"H1") == 0)
            MFEM_ABORT ("Not supported case sigma from vector H1, think of the boundary conditions there");

        // aliases
        ParFiniteElementSpace * Hdiv_space_lvl = hierarchy->GetHdiv_space(l);
        ParFiniteElementSpace * H1_space_lvl = hierarchy->GetH1_space(l);
        ParFiniteElementSpace * L2_space_lvl = hierarchy->GetL2_space(l);

        ParMeshCyl * pmeshtsl_lvl = hierarchy->GetPmeshcyl(l);

        //ParFiniteElementSpace *H1vec_space;
        //if (strcmp(space_for_sigma,"H1") == 0)
            //H1vec_space = new ParFiniteElementSpace(pmeshtsl, h1_coll, dim, Ordering::byVDIM);
        //if (strcmp(space_for_sigma,"Hdiv") == 0)
            //Sigma_space_lvls[l] = Hdiv_space_lvls[l];
        //else
            //Sigma_space_lvls[l] = H1vec_space_lvls[l];
        Sigma_space_lvls[l] = Hdiv_space_lvl;

        if (strcmp(space_for_S,"H1") == 0)
            S_space_lvls[l] = H1_space_lvl;
        else // "L2"
            S_space_lvls[l] = L2_space_lvl;

        HYPRE_Int dimR = Hdiv_space_lvl->GlobalTrueVSize();
        HYPRE_Int dimH = H1_space_lvl->GlobalTrueVSize();
        //HYPRE_Int dimHvec;
        //if (strcmp(space_for_sigma,"H1") == 0)
            //dimHvec = H1vec_space_lvls[l]->GlobalTrueVSize();
        HYPRE_Int dimW = L2_space_lvl->GlobalTrueVSize();

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

        Array<int> block_offsets(numblocks + 1); // number of variables + 1
        int tempblknum = 0;
        block_offsets[0] = 0;
        tempblknum++;
        block_offsets[tempblknum] = Sigma_space_lvls[l]->GetVSize();
        tempblknum++;

        if (strcmp(space_for_S,"H1") == 0)
        {
            block_offsets[tempblknum] = H1_space_lvl->GetVSize();
            tempblknum++;
        }
        if (strcmp(formulation,"cfosls") == 0)
        {
            block_offsets[tempblknum] = L2_space_lvl->GetVSize();
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
            (*block_trueOffsets_lvls[l])[tempblknum] = H1_space_lvl->TrueVSize();
            tempblknum++;
        }
        if (strcmp(formulation,"cfosls") == 0)
        {
            (*block_trueOffsets_lvls[l])[tempblknum] = L2_space_lvl->TrueVSize();
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

       ess_bdrat_S.resize(pmeshtsl_lvl->bdr_attributes.Max());
       for (unsigned int i = 0; i < ess_bdrat_S.size(); ++i)
           ess_bdrat_S[i] = 0;
       if (strcmp(space_for_S,"H1") == 0)
           ess_bdrat_S[0] = 1; // t = 0

       ess_bdrat_sigma.resize(pmeshtsl_lvl->bdr_attributes.Max());
       for (unsigned int i = 0; i < ess_bdrat_sigma.size(); ++i)
           ess_bdrat_sigma[i] = 0;
       if (strcmp(space_for_S,"L2") == 0) // if S is from L2 we impose bdr condition for sigma at t = 0
           ess_bdrat_sigma[0] = 1;

       Array<int> ess_bdrS(pmeshtsl_lvl->bdr_attributes.Max());
       for (unsigned int i = 0; i < ess_bdrat_S.size(); ++i)
           ess_bdrS[i] = ess_bdrat_S[i];

       Array<int> ess_bdrSigma(pmeshtsl_lvl->bdr_attributes.Max());
       for (unsigned int i = 0; i < ess_bdrat_sigma.size(); ++i)
           ess_bdrSigma[i] = ess_bdrat_sigma[i];

       if (verbose)
       {
           std::cout << "Boundary conditions: \n";
           std::cout << "ess bdr Sigma: \n";
           ess_bdrSigma.Print(std::cout, pmeshtsl_lvl->bdr_attributes.Max());
           std::cout << "ess bdr S: \n";
           ess_bdrS.Print(std::cout, pmeshtsl_lvl->bdr_attributes.Max());
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
          ParMixedBilinearForm *Dblock(new ParMixedBilinearForm(Sigma_space_lvls[l], L2_space_lvl));
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
          ParMixedBilinearForm *Dblock_nobnd(new ParMixedBilinearForm(Sigma_space_lvls[l], L2_space_lvl));
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

    for (int l = num_lvls - 1 - 1; l >= 0; --l)
    {
        HypreParMatrix * TrueP_Hdiv_lvl = hierarchy->GetTrueP_Hdiv(l);
        HypreParMatrix * TrueP_H1_lvl = hierarchy->GetTrueP_H1(l);
        HypreParMatrix * TrueP_L2_lvl = hierarchy->GetTrueP_L2(l);

        TrueP_lvls[l] = new BlockOperator(*block_trueOffsets_lvls[l], *block_trueOffsets_lvls[l + 1]);
        TrueP_lvls[l]->SetBlock(0,0, TrueP_Hdiv_lvl);
        if (strcmp(space_for_S,"H1") == 0) // S is present
        {
            TrueP_lvls[l]->SetBlock(1,1, TrueP_H1_lvl);
            TrueP_lvls[l]->SetBlock(2,2, TrueP_L2_lvl);
        }
        else
            TrueP_lvls[l]->SetBlock(1,1, TrueP_L2_lvl);
    }

    for (int l = 0; l < num_lvls; ++l)
    {
        Funct_hpmat_lvls[l] = new Array2D<HypreParMatrix*>(numvars, numvars);

        if (l == 0)
        {
            for (int i = 0; i < numvars; ++i)
                for (int j = 0; j < numvars; ++j)
                {
                    HypreParMatrix& Block = (HypreParMatrix&)CFOSLSop_lvls[0]->GetBlock(0,0);

                    (*Funct_hpmat_lvls[0])(i,j) = &Block;
                }
        }
        else // doing RAP for the Functional matrix as an Array2D<HypreParMatrix*>
        {
            // aliases
            HypreParMatrix * TrueP_Hdiv_lvl = hierarchy->GetTrueP_Hdiv(l-1);
            HypreParMatrix * TrueP_H1_lvl = hierarchy->GetTrueP_H1(l-1);
            //HypreParMatrix * TrueP_L2_lvl = hierarchy->GetTrueP_L2(l-1);

            // TODO: Rewrite this in a general form
            (*Funct_hpmat_lvls[l])(0,0) = RAP(TrueP_Hdiv_lvl, (*Funct_hpmat_lvls[l-1])(0,0), TrueP_Hdiv_lvl);
            (*Funct_hpmat_lvls[l])(0,0)->CopyRowStarts();
            (*Funct_hpmat_lvls[l])(0,0)->CopyRowStarts();

            {
                Array<int> temp_dom;

                Array<int> ess_bdr_sigma(ess_bdrat_sigma.size());
                for (unsigned int i = 0; i < ess_bdrat_sigma.size(); ++i)
                    ess_bdr_sigma[i] = ess_bdrat_sigma[i];

                Sigma_space_lvls[l]->GetEssentialTrueDofs(ess_bdr_sigma, temp_dom);

                //const Array<int> *temp_dom = EssBdrTrueDofs_Funct_lvls[l][0];

                Eliminate_ib_block(*(*Funct_hpmat_lvls[l])(0,0), temp_dom, temp_dom );
                HypreParMatrix * temphpmat = (*Funct_hpmat_lvls[l])(0,0)->Transpose();
                Eliminate_ib_block(*temphpmat, temp_dom, temp_dom );
                (*Funct_hpmat_lvls[l])(0,0) = temphpmat->Transpose();
                Eliminate_bb_block(*(*Funct_hpmat_lvls[l])(0,0), temp_dom);
                SparseMatrix diag;
                (*Funct_hpmat_lvls[l])(0,0)->GetDiag(diag);
                diag.MoveDiagonalFirst();

                (*Funct_hpmat_lvls[l])(0,0)->CopyRowStarts();
                (*Funct_hpmat_lvls[l])(0,0)->CopyColStarts();
                delete temphpmat;
            }


            if (strcmp(space_for_S,"H1") == 0)
            {
                (*Funct_hpmat_lvls[l])(1,1) = RAP(TrueP_H1_lvl, (*Funct_hpmat_lvls[l-1])(1,1), TrueP_H1_lvl);
                //(*Funct_hpmat_lvls[l])(1,1)->CopyRowStarts();
                //(*Funct_hpmat_lvls[l])(1,1)->CopyRowStarts();

                {
                    Array<int> temp_dom;

                    Array<int> ess_bdr_S(ess_bdrat_S.size());
                    for (unsigned int i = 0; i < ess_bdrat_S.size(); ++i)
                        ess_bdr_S[i] = ess_bdrat_S[i];

                    S_space_lvls[l]->GetEssentialTrueDofs(ess_bdr_S, temp_dom);

                    //const Array<int> *temp_dom = EssBdrTrueDofs_Funct_lvls[l][1];

                    Eliminate_ib_block(*(*Funct_hpmat_lvls[l])(1,1), temp_dom, temp_dom );
                    HypreParMatrix * temphpmat = (*Funct_hpmat_lvls[l])(1,1)->Transpose();
                    Eliminate_ib_block(*temphpmat, temp_dom, temp_dom );
                    (*Funct_hpmat_lvls[l])(1,1) = temphpmat->Transpose();
                    Eliminate_bb_block(*(*Funct_hpmat_lvls[l])(1,1), temp_dom);
                    SparseMatrix diag;
                    (*Funct_hpmat_lvls[l])(1,1)->GetDiag(diag);
                    diag.MoveDiagonalFirst();

                    (*Funct_hpmat_lvls[l])(1,1)->CopyRowStarts();
                    (*Funct_hpmat_lvls[l])(1,1)->CopyColStarts();
                    delete temphpmat;
                }

                HypreParMatrix * TrueP_Hdiv_T = TrueP_Hdiv_lvl->Transpose();
                HypreParMatrix * temp1 = ParMult((*Funct_hpmat_lvls[l-1])(0,1), TrueP_H1_lvl);
                (*Funct_hpmat_lvls[l])(0,1) = ParMult(TrueP_Hdiv_T, temp1);
                //(*Funct_hpmat_lvls[l])(0,1)->CopyRowStarts();
                //(*Funct_hpmat_lvls[l])(0,1)->CopyRowStarts();

                {
                    Array<int> temp_dom;

                    Array<int> ess_bdr_S(ess_bdrat_S.size());
                    for (unsigned int i = 0; i < ess_bdrat_S.size(); ++i)
                        ess_bdr_S[i] = ess_bdrat_S[i];

                    S_space_lvls[l]->GetEssentialTrueDofs(ess_bdr_S, temp_dom);

                    Array<int> temp_range;

                    Array<int> ess_bdr_sigma(ess_bdrat_sigma.size());
                    for (unsigned int i = 0; i < ess_bdrat_sigma.size(); ++i)
                        ess_bdr_sigma[i] = ess_bdrat_sigma[i];

                    Sigma_space_lvls[l]->GetEssentialTrueDofs(ess_bdr_sigma, temp_range);


                    //const Array<int> *temp_range = EssBdrTrueDofs_Funct_lvls[l][0];
                    //const Array<int> *temp_dom = EssBdrTrueDofs_Funct_lvls[l][1];

                    Eliminate_ib_block(*(*Funct_hpmat_lvls[l])(0,1), temp_dom, temp_range );
                    HypreParMatrix * temphpmat = (*Funct_hpmat_lvls[l])(0,1)->Transpose();
                    Eliminate_ib_block(*temphpmat, temp_range, temp_dom );
                    (*Funct_hpmat_lvls[l])(0,1) = temphpmat->Transpose();
                    (*Funct_hpmat_lvls[l])(0,1)->CopyRowStarts();
                    (*Funct_hpmat_lvls[l])(0,1)->CopyColStarts();
                    delete temphpmat;
                }



                (*Funct_hpmat_lvls[l])(1,0) = (*Funct_hpmat_lvls[l])(0,1)->Transpose();
                (*Funct_hpmat_lvls[l])(1,0)->CopyRowStarts();
                (*Funct_hpmat_lvls[l])(1,0)->CopyRowStarts();

                delete TrueP_Hdiv_T;
                delete temp1;
            }

        } // end of else for if (l == 0)

    }

    for (int l = 0; l < num_lvls; ++l)
    {
        CFOSLSop_coarsened_lvls[l] = new BlkHypreOperator(*Funct_hpmat_lvls[l]);
    }

    /////////////////////////////////////////////////////////////////

}

void TimeSteppingScheme::ComputeAnalyticalRhs(int level)
{
    for (int tslab = 0; tslab < nslabs; ++tslab )
        timeslab_problems[tslab]->ComputeAnalyticalRhs(level);
}

void TimeSteppingScheme::RestrictToCoarser(int level, std::vector<Vector*> vec_ins, std::vector<Vector*> vec_outs)
{
    for (int tslab = 0; tslab < nslabs; ++tslab )
    {
        timeslab_problems[tslab]->Restrict(level, *vec_ins[tslab], *vec_outs[tslab]);
    }
}

void TimeSteppingScheme::InterpolateToFiner(int level, std::vector<Vector*> vec_ins, std::vector<Vector*> vec_outs)
{
    for (int tslab = 0; tslab < nslabs; ++tslab )
    {
        timeslab_problems[tslab]->Interpolate(level, *vec_ins[tslab], *vec_outs[tslab]);
    }
}


void TimeSteppingScheme::ComputeResiduals(int level)
{
    timeslab_problems[0]->ComputeResidual(level, *vec_ins_lvls[level][0], *sols_lvls[level][0],
            *residuals_lvls[level][0]);

    for (int tslab = 1; tslab < nslabs; ++tslab )
            timeslab_problems[tslab]->ComputeResidual(level, *vec_outs_lvls[level][tslab - 1], *sols_lvls[level][tslab],
                    *residuals_lvls[level][tslab]);
}

void TimeSteppingScheme::Solve(char const * mode, char const * level_mode, std::vector<Vector*> rhss, int level, bool compute_accuracy)
{
    if (strcmp(mode,"parallel") !=0 && strcmp(mode, "sequential") != 0)
    {
        MFEM_ABORT("In TimeSteppingScheme::Solve mode must be 'sequential' or 'parallel'");
    }

    if (strcmp(mode,"sequential") == 0) // sequential time-stepping, time slab after time slab
    {
        for (int tslab = 0; tslab < nslabs; ++tslab )
        {
            timeslab_problems[tslab]->Solve(level_mode, level, *rhss[tslab], *sols_lvls[level][tslab],
                                            *vec_ins_lvls[level][tslab], *vec_outs_lvls[level][tslab]);

            *vec_ins_lvls[level][tslab + 1] = *vec_outs_lvls[level][tslab];

            if (timeslab_problems[tslab]->NeedSignSwitch())
                *vec_ins_lvls[level][tslab + 1] *= -1;

            //Xinit = Xout;
            //if (strcmp(space_for_S,"L2") == 0)
                //Xinit *= -1.0;

            if (compute_accuracy)
            {
                Vector * Xout_exact = timeslab_problems[tslab]->GetExactBase("top", level);

                Vector Xout_error(timeslab_problems[tslab]->GetInitCondSize(level));
                Xout_error = *vec_outs_lvls[level][tslab];
                Xout_error -= *Xout_exact;
                if (verbose)
                {
                    std::cout << "|| Xout  - Xout_exact || = " << Xout_error.Norml2() / sqrt (Xout_error.Size()) << "\n";
                    std::cout << "|| Xout  - Xout_exact || / || Xout_exact || = " << (Xout_error.Norml2() / sqrt (Xout_error.Size())) /
                                 (Xout_exact->Norml2() / sqrt (Xout_exact->Size()))<< "\n";
                }

                delete Xout_exact;
            }


        } // end of loop over all time slabs, performing a coarse solve

    }
    else // 'parallel'
    {
        for (int tslab = 0; tslab < nslabs; ++tslab )
        {
            timeslab_problems[tslab]->Solve(level_mode, level, *rhss[tslab], *sols_lvls[level][tslab],
                                            *vec_ins_lvls[level][tslab], *vec_outs_lvls[level][tslab]);
        }
    }

    MFEM_ABORT("Not implemented \n");
}

TimeSteppingScheme::TimeSteppingScheme (std::vector<TimeCylHyper*> & timeslab_problems)
    : timeslab_problems(timeslab_problems),
      nslabs(timeslab_problems.size()),
      nlevels(timeslab_problems[0]->GetNLevels()),
      verbose(timeslab_problems[0]->verbose)
{
    vec_ins_lvls.resize(nlevels);
    for (unsigned int l = 0; l < vec_ins_lvls.size(); ++l)
    {
        vec_ins_lvls[l].resize(nslabs + 1);
        for (int slab = 0; slab < nslabs; ++slab)
            vec_ins_lvls[l][slab] = new Vector(timeslab_problems[slab]->GetInitCondSize(l));
        vec_ins_lvls[l][nslabs] = new Vector(timeslab_problems[nslabs - 1]->GetInitCondSize(l));
    }

    vec_outs_lvls.resize(nlevels);
    for (unsigned int l = 0; l < vec_outs_lvls.size(); ++l)
    {
        vec_outs_lvls[l].resize(nslabs + 1);
        for (int slab = 0; slab < nslabs; ++slab)
            vec_outs_lvls[l][slab] = new Vector(timeslab_problems[slab]->GetInitCondSize(l));
        vec_outs_lvls[l][nslabs] = new Vector(timeslab_problems[nslabs - 1]->GetInitCondSize(l));
    }

    sols_lvls.resize(nlevels);
    for (unsigned int l = 0; l < sols_lvls.size(); ++l)
    {
        sols_lvls[l].resize(nslabs);
        for (int slab = 0; slab < nslabs; ++slab)
            sols_lvls[l][slab] = new Vector(timeslab_problems[slab]->ProblemSize(l));
    }

    residuals_lvls.resize(nlevels);
    for (unsigned int l = 0; l < residuals_lvls.size(); ++l)
    {
        residuals_lvls[l].resize(nslabs);
        for (int slab = 0; slab < nslabs; ++slab)
            residuals_lvls[l][slab] = new Vector(timeslab_problems[slab]->ProblemSize(l));
    }

}

void SpaceTimeTwoGrid::ComputeResidual(std::vector<Vector*> rhss, std::vector<Vector*> sols)
{
    MFEM_ABORT("Not implemented \n");
}

void SpaceTimeTwoGrid::UpdateResidual(std::vector<Vector*> corrs)
{
    MFEM_ABORT("Not implemented \n");
}

void SpaceTimeTwoGrid::UpdateSolution(std::vector<Vector*> sols, std::vector<Vector*> corrs)
{
    for (int tslab = 0; tslab < nslabs; ++tslab )
        *sols[tslab] += * corrs[tslab];
}


void SpaceTimeTwoGrid::Solve(std::vector<Vector*> rhss, std::vector<Vector*> sols)
{
    bool converged = false;

    ComputeResidual(rhss, sols);

    for (int it = 0; it < max_iter; ++it)
    {
        Iterate(res_lvls[0], corr_lvls[0]);

        UpdateResidual(corr_lvls[0]);

        UpdateSolution(sols, corr_lvls[0]);

        if (converged)
            break;
    }
}

void SpaceTimeTwoGrid::Iterate(std::vector<Vector*> ress, std::vector<Vector *> corrs)
{
    MFEM_ABORT("Not implemented yet \n");
    /*
    const int fine_level = 0;
    const int coarse_level = 1;
    // 1. parallel-in-time smoothing
    timestepping.Solve("parallel", fine_level, false);

    // 2. compute residuals
    timestepping.ComputeResiduals(fine_level);

    // 3. restrict residuals
    timestepping.RestrictToCoarser(fine_level, *timestepping.Get_residuals(0), *timestepping.Get_residuals(1));

    // 4. solve at the coarse level with zero initial guess
    timestepping.SetZeroInitialCondition(coarse_level);
    timestepping.Solve("sequential", coarse_level, false);

    // 5.
    timestepping

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
    */
}

} // for namespace mfem
