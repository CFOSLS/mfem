#include <iostream>
#include "testhead.hpp"

using namespace std;

namespace mfem
{

HypreParMatrix * FOSLSCylProblem::ConstructRestriction(const char * top_or_bot) const
{
    MFEM_ASSERT(strcmp(top_or_bot,"bot") == 0 || strcmp(top_or_bot,"top") == 0,
                "In ConstructRestriction() top_or_bot must equal 'top' or 'bot'! \n");

    return CreateRestriction(top_or_bot, *pfes[init_cond_block], tdofs_link);
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

// FIXME: Make this return a pointer to avoid the memory leak
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
        const Array<int>& essbdr_attrs = bdr_conds.GetBdrAttribs(blk);

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
        const Array<int>& essbdr_attrs = bdr_conds.GetBdrAttribs(blk);

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
    ConvertInitCndToFullVector(init_cond, temp_vec1);

    ConvertBdrCndIntoRhs(temp_vec1, temp_vec2);

    temp_vec2 *= coeff;
    vec_out += temp_vec2;
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

} // for namespace mfem
