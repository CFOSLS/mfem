// TODO: split this into hpp and cpp, but the first attempt failed
#include <iostream>

using namespace std;
using namespace mfem;

HypreParMatrix * CreateRestriction(const char * top_or_bot, ParFiniteElementSpace& pfespace,
                                   std::vector<std::pair<int,int> >& bot_to_top_tdofs_link);
std::vector<std::pair<int,int> >* CreateBotToTopDofsLink(const char * eltype, FiniteElementSpace& fespace,
                                                         std::vector<std::pair<int,int> > & bot_to_top_bels, bool verbose = false);

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

class BlockProblemForms
{
    friend class CFOSLSHyperbolicProblem;
protected:
    const int numblocks;
    Array<ParBilinearForm*> diag_forms;
    Array2D<ParMixedBilinearForm*> offd_forms;
public:
    BlockProblemForms(int num_blocks) : numblocks(num_blocks)
    {
        diag_forms.SetSize(num_blocks);
        for (int i = 0; i < num_blocks; ++i)
            diag_forms[i] = NULL;
        offd_forms.SetSize(numblocks, num_blocks);
        for (int i = 0; i < num_blocks; ++i)
            for (int j = 0; j < num_blocks; ++j)
                offd_forms(i,j) = NULL;
    }
    ParBilinearForm* & diag(int i) {return diag_forms[i];}
    ParMixedBilinearForm* & offd(int i, int j) {return offd_forms(i,j);}
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

CFOSLSHyperbolicProblem::CFOSLSHyperbolicProblem(CFOSLSHyperbolicFormulation &struct_formulation,
                                                 int fe_order, bool verbose)
    : feorder (fe_order), struct_formul(struct_formulation),
      spaces_initialized(false), forms_initialized(false), solver_initialized(false),
      pbforms(struct_formul.numblocks)
{
    InitFEColls(verbose);
}

CFOSLSHyperbolicProblem::CFOSLSHyperbolicProblem(ParMesh& pmesh, CFOSLSHyperbolicFormulation &struct_formulation,
                                                 int fe_order, int prec_option, bool verbose)
    : feorder (fe_order), struct_formul(struct_formulation), pbforms(struct_formul.numblocks)
{
    InitFEColls(verbose);
    InitSpaces(pmesh);
    spaces_initialized = true;
    InitForms();
    forms_initialized = true;
    AssembleSystem(verbose);
    InitPrec(prec_option, verbose);
    InitSolver(verbose);
    solver_initialized = true;
    InitGrFuns();
}

void CFOSLSHyperbolicProblem::InitFEColls(bool verbose)
{
    if ( struct_formul.dim == 4 )
    {
        hdiv_coll = new RT0_4DFECollection;
        if(verbose)
            cout << "RT: order 0 for 4D" << endl;
    }
    else
    {
        hdiv_coll = new RT_FECollection(feorder, struct_formul.dim);
        if(verbose)
            cout << "RT: order " << feorder << " for 3D" << endl;
    }

    if (struct_formul.dim == 4)
        MFEM_ASSERT(feorder == 0, "Only lowest order elements are support in 4D!");

    if (struct_formul.dim == 4)
    {
        h1_coll = new LinearFECollection;
        if (verbose)
            cout << "H1 in 4D: linear elements are used" << endl;
    }
    else
    {
        h1_coll = new H1_FECollection(feorder+1, struct_formul.dim);
        if(verbose)
            cout << "H1: order " << feorder + 1 << " for 3D" << endl;
    }
    l2_coll = new L2_FECollection(feorder, struct_formul.dim);
    if (verbose)
        cout << "L2: order " << feorder << endl;
}

void CFOSLSHyperbolicProblem::InitSpaces(ParMesh &pmesh)
{
    Hdiv_space = new ParFiniteElementSpace(&pmesh, hdiv_coll);
    H1_space = new ParFiniteElementSpace(&pmesh, h1_coll);
    L2_space = new ParFiniteElementSpace(&pmesh, l2_coll);
    H1vec_space = new ParFiniteElementSpace(&pmesh, h1_coll, struct_formul.dim, Ordering::byVDIM);

    pfes.SetSize(struct_formul.numblocks);

    int blkcount = 0;
    if (strcmp(struct_formul.space_for_sigma,"Hdiv") == 0)
        pfes[0] = Hdiv_space;
    else
        pfes[0] = H1vec_space;
    Sigma_space = pfes[0];
    ++blkcount;

    if (strcmp(struct_formul.space_for_S,"H1") == 0)
    {
        pfes[blkcount] = H1_space;
        S_space = pfes[blkcount];
        ++blkcount;
    }
    else // "L2"
    {
        S_space = L2_space;
    }

    if (struct_formul.have_constraint)
        pfes[blkcount] = L2_space;

}

void CFOSLSHyperbolicProblem::InitForms()
{
    MFEM_ASSERT(spaces_initialized, "Spaces must have been initialized by this moment!\n");

    plforms.SetSize(struct_formul.numblocks);
    for (int i = 0; i < struct_formul.numblocks; ++i)
    {
        plforms[i] = new ParLinearForm(pfes[i]);
        if (struct_formul.lfis[i])
            plforms[i]->AddDomainIntegrator(struct_formul.lfis[i]);
    }

    for (int i = 0; i < struct_formul.numblocks; ++i)
        for (int j = 0; j < struct_formul.numblocks; ++j)
        {
            if (i == j)
                pbforms.diag(i) = new ParBilinearForm(pfes[i]);
            else
                pbforms.offd(i,j) = new ParMixedBilinearForm(pfes[j], pfes[i]);

            if (struct_formul.blfis(i,j))
            {
                if (i == j)
                    pbforms.diag(i)->AddDomainIntegrator(struct_formul.blfis(i,j));
                else
                    pbforms.offd(i,j)->AddDomainIntegrator(struct_formul.blfis(i,j));
            }
        }

}

BlockVector * CFOSLSHyperbolicProblem::SetTrueInitialCondition()
{
    BlockVector * truebnd = new BlockVector(blkoffsets_true);
    *truebnd = 0.0;

    Transport_test Mytest(struct_formul.dim,struct_formul.numsol);

    ParGridFunction * sigma_exact = new ParGridFunction(Sigma_space);
    sigma_exact->ProjectCoefficient(*(Mytest.sigma));
    Vector sigma_exact_truedofs(Sigma_space->TrueVSize());
    sigma_exact->ParallelProject(sigma_exact_truedofs);

    Array<int> ess_tdofs_sigma;
    Sigma_space->GetEssentialTrueDofs(*struct_formul.essbdr_attrs[0], ess_tdofs_sigma);

    for (int j = 0; j < ess_tdofs_sigma.Size(); ++j)
    {
        int tdof = ess_tdofs_sigma[j];
        truebnd->GetBlock(0)[tdof] = sigma_exact_truedofs[tdof];
    }

    if (strcmp(struct_formul.space_for_S,"H1") == 0)
    {
        ParGridFunction *S_exact = new ParGridFunction(S_space);
        S_exact->ProjectCoefficient(*(Mytest.scalarS));
        Vector S_exact_truedofs(S_space->TrueVSize());
        S_exact->ParallelProject(S_exact_truedofs);

        Array<int> ess_tdofs_S;
        S_space->GetEssentialTrueDofs(*struct_formul.essbdr_attrs[1], ess_tdofs_S);

        for (int j = 0; j < ess_tdofs_S.Size(); ++j)
        {
            int tdof = ess_tdofs_S[j];
            truebnd->GetBlock(1)[tdof] = S_exact_truedofs[tdof];
        }

    }

    return truebnd;
}

BlockVector * CFOSLSHyperbolicProblem::SetInitialCondition()
{
    BlockVector * init_cond = new BlockVector(blkoffsets);
    *init_cond = 0.0;

    Transport_test Mytest(struct_formul.dim,struct_formul.numsol);

    ParGridFunction * sigma_exact = new ParGridFunction(Sigma_space);
    sigma_exact->ProjectCoefficient(*(Mytest.sigma));

    init_cond->GetBlock(0) = *sigma_exact;
    if (strcmp(struct_formul.space_for_S,"H1") == 0)
    {
        ParGridFunction *S_exact = new ParGridFunction(S_space);
        S_exact->ProjectCoefficient(*(Mytest.scalarS));
        init_cond->GetBlock(1) = *S_exact;
    }

    return init_cond;
}

void CFOSLSHyperbolicProblem::InitGrFuns()
{
    // + 1 for the f stored as a grid function from L2
    grfuns.SetSize(struct_formul.unknowns_number + 1);
    for (int i = 0; i < struct_formul.unknowns_number; ++i)
        grfuns[i] = new ParGridFunction(pfes[i]);
    grfuns[struct_formul.unknowns_number] = new ParGridFunction(L2_space);

    Transport_test Mytest(struct_formul.dim,struct_formul.numsol);
    grfuns[struct_formul.unknowns_number]->ProjectCoefficient(*Mytest.scalardivsigma);
}

void CFOSLSHyperbolicProblem::BuildCFOSLSSystem(ParMesh &pmesh, bool verbose)
{
    if (!spaces_initialized)
    {
        Hdiv_space = new ParFiniteElementSpace(&pmesh, hdiv_coll);
        H1_space = new ParFiniteElementSpace(&pmesh, h1_coll);
        L2_space = new ParFiniteElementSpace(&pmesh, l2_coll);

        if (strcmp(struct_formul.space_for_sigma,"H1") == 0)
            H1vec_space = new ParFiniteElementSpace(&pmesh, h1_coll, struct_formul.dim, Ordering::byVDIM);

        if (strcmp(struct_formul.space_for_sigma,"Hdiv") == 0)
            Sigma_space = Hdiv_space;
        else
            Sigma_space = H1vec_space;

        if (strcmp(struct_formul.space_for_S,"H1") == 0)
            S_space = H1_space;
        else // "L2"
            S_space = L2_space;

        MFEM_ASSERT(!forms_initialized, "Forms cannot have been already initialized by this moment!");

        InitForms();
    }

    AssembleSystem(verbose);
}

void CFOSLSHyperbolicProblem::Solve(bool verbose)
{
    *trueX = 0;

    chrono.Clear();
    chrono.Start();

    //trueRhs->Print();
    //SparseMatrix diag;
    //((HypreParMatrix&)(CFOSLSop->GetBlock(0,0))).GetDiag(diag);
    //diag.Print();

    solver->Mult(*trueRhs, *trueX);

    chrono.Stop();

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

    DistributeSolution();

    ComputeError(verbose, true);
}

void CFOSLSHyperbolicProblem::DistributeSolution()
{
    for (int i = 0; i < struct_formul.unknowns_number; ++i)
        grfuns[i]->Distribute(&(trueX->GetBlock(i)));
}

void CFOSLSHyperbolicProblem::ComputeError(bool verbose, bool checkbnd)
{
    Transport_test Mytest(struct_formul.dim,struct_formul.numsol);

    ParMesh * pmesh = pfes[0]->GetParMesh();

    ParGridFunction * sigma = grfuns[0];

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

    ParGridFunction * S;
    if (strcmp(struct_formul.space_for_S,"H1") == 0)
    {
        //std::cout << "I am here \n";
        S = grfuns[1];
    }
    else
    {
        //std::cout << "I am there \n";
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
        B->Mult(trueX->GetBlock(0),bTsigma);

        Vector trueS(C->Height());

        CG(*C, bTsigma, trueS, 0, 5000, 1e-9, 1e-12);

        S = new ParGridFunction(S_space);
        S->Distribute(trueS);

        delete Cblock;
        delete Bblock;
        delete B;
        delete C;
    }

    //std::cout << "I compute S_h one way or another \n";

    double err_S = S->ComputeL2Error((*Mytest.scalarS), irs);
    double norm_S = ComputeGlobalLpNorm(2, (*Mytest.scalarS), *pmesh, irs);
    if (verbose)
    {
        std::cout << "|| S_h - S_ex || / || S_ex || = " <<
                     err_S / norm_S << "\n";
    }

    if (checkbnd)
    {
        ParGridFunction * sigma_exact = new ParGridFunction(Sigma_space);
        sigma_exact->ProjectCoefficient(*Mytest.sigma);
        Vector sigma_exact_truedofs(Sigma_space->TrueVSize());
        sigma_exact->ParallelProject(sigma_exact_truedofs);

        Array<int> EssBnd_tdofs_sigma;
        Sigma_space->GetEssentialTrueDofs(*struct_formul.essbdr_attrs[0], EssBnd_tdofs_sigma);

        for (int i = 0; i < EssBnd_tdofs_sigma.Size(); ++i)
        {
            int tdof = EssBnd_tdofs_sigma[i];
            double value_ex = sigma_exact_truedofs[tdof];
            double value_com = trueX->GetBlock(0)[tdof];

            if (fabs(value_ex - value_com) > MYZEROTOL)
            {
                std::cout << "bnd condition is violated for sigma, tdof = " << tdof << " exact value = "
                          << value_ex << ", value_com = " << value_com << ", diff = " << value_ex - value_com << "\n";
                std::cout << "rhs side at this tdof = " << trueRhs->GetBlock(0)[tdof] << "\n";
            }
        }

        if (strcmp(struct_formul.space_for_S,"H1") == 0) // S is present
        {
            ParGridFunction * S_exact = new ParGridFunction(S_space);
            S_exact->ProjectCoefficient(*Mytest.scalarS);

            Vector S_exact_truedofs(S_space->TrueVSize());
            S_exact->ParallelProject(S_exact_truedofs);

            Array<int> EssBnd_tdofs_S;
            S_space->GetEssentialTrueDofs(*struct_formul.essbdr_attrs[1], EssBnd_tdofs_S);

            for (int i = 0; i < EssBnd_tdofs_S.Size(); ++i)
            {
                int tdof = EssBnd_tdofs_S[i];
                double value_ex = S_exact_truedofs[tdof];
                double value_com = trueX->GetBlock(1)[tdof];

                if (fabs(value_ex - value_com) > MYZEROTOL)
                {
                    std::cout << "bnd condition is violated for S, tdof = " << tdof << " exact value = "
                              << value_ex << ", value_com = " << value_com << ", diff = " << value_ex - value_com << "\n";
                    std::cout << "rhs side at this tdof = " << trueRhs->GetBlock(1)[tdof] << "\n";
                }
            }
        }
    }
}


// works correctly only for problems with homogeneous initial conditions?
// see the times-stepping branch, think of how boundary conditions for off-diagonal blocks are imposed
// system is assumed to be symmetric
void CFOSLSHyperbolicProblem::AssembleSystem(bool verbose)
{
    int numblocks = struct_formul.numblocks;

    blkoffsets_true.SetSize(numblocks + 1);
    blkoffsets_true[0] = 0;
    for (int i = 0; i < numblocks; ++i)
        blkoffsets_true[i + 1] = pfes[i]->TrueVSize();
    blkoffsets_true.PartialSum();

    blkoffsets.SetSize(numblocks + 1);
    blkoffsets[0] = 0;
    for (int i = 0; i < numblocks; ++i)
        blkoffsets[i + 1] = pfes[i]->GetVSize();
    blkoffsets.PartialSum();

    x = SetInitialCondition();

    trueRhs = new BlockVector(blkoffsets_true);
    trueX = new BlockVector(blkoffsets_true);

    for (int i = 0; i < numblocks; ++i)
        plforms[i]->Assemble();

    hpmats_nobnd.SetSize(numblocks, numblocks);
    for (int i = 0; i < numblocks; ++i)
        for (int j = 0; j < numblocks; ++j)
            hpmats_nobnd(i,j) = NULL;
    for (int i = 0; i < numblocks; ++i)
        for (int j = 0; j < numblocks; ++j)
        {
            if (i == j)
            {
                if (pbforms.diag(i))
                {
                    pbforms.diag(i)->Assemble();
                    pbforms.diag(i)->Finalize();
                    hpmats_nobnd(i,j) = pbforms.diag(i)->ParallelAssemble();
                }
            }
            else // off-diagonal
            {
                if (pbforms.offd(i,j) || pbforms.offd(j,i))
                {
                    int exist_row, exist_col;
                    if (pbforms.offd(i,j))
                    {
                        exist_row = i;
                        exist_col = j;
                    }
                    else
                    {
                        exist_row = j;
                        exist_col = i;
                    }

                    pbforms.offd(exist_row,exist_col)->Assemble();

                    pbforms.offd(exist_row,exist_col)->Finalize();
                    hpmats_nobnd(exist_row,exist_col) = pbforms.offd(exist_row,exist_col)->ParallelAssemble();
                    hpmats_nobnd(exist_col, exist_row) = hpmats_nobnd(exist_row,exist_col)->Transpose();
                }
            }
        }

    for (int i = 0; i < numblocks; ++i)
        for (int j = 0; j < numblocks; ++j)
            if (i == j)
                pbforms.diag(i)->LoseMat();
            else
                if (pbforms.offd(i,j))
                    pbforms.offd(i,j)->LoseMat();

    hpmats.SetSize(numblocks, numblocks);
    for (int i = 0; i < numblocks; ++i)
        for (int j = 0; j < numblocks; ++j)
            hpmats(i,j) = NULL;

    for (int i = 0; i < numblocks; ++i)
        for (int j = 0; j < numblocks; ++j)
        {
            if (i == j)
            {
                if (pbforms.diag(i))
                {
                    pbforms.diag(i)->Assemble();

                    //pbforms.diag(i)->EliminateEssentialBC(*struct_formul.essbdr_attrs[i],
                            //x->GetBlock(i), *plforms[i]);
                    Vector dummy(pbforms.diag(i)->Height());
                    dummy = 0.0;
                    pbforms.diag(i)->EliminateEssentialBC(*struct_formul.essbdr_attrs[i],
                            x->GetBlock(i), dummy);
                    pbforms.diag(i)->Finalize();
                    hpmats(i,j) = pbforms.diag(i)->ParallelAssemble();

                    SparseMatrix diag;
                    hpmats(i,j)->GetDiag(diag);
                    Array<int> essbnd_tdofs;
                    pfes[i]->GetEssentialTrueDofs(*struct_formul.essbdr_attrs[i], essbnd_tdofs);
                    for (int i = 0; i < essbnd_tdofs.Size(); ++i)
                    {
                        int tdof = essbnd_tdofs[i];
                        diag.EliminateRow(tdof,1.0);
                    }

                }
            }
            else // off-diagonal
            {
                if (pbforms.offd(i,j) || pbforms.offd(j,i))
                {
                    int exist_row, exist_col;
                    if (pbforms.offd(i,j))
                    {
                        exist_row = i;
                        exist_col = j;
                    }
                    else
                    {
                        exist_row = j;
                        exist_col = i;
                    }

                    pbforms.offd(exist_row,exist_col)->Assemble();

                    //pbforms.offd(exist_row,exist_col)->EliminateTrialDofs(*struct_formul.essbdr_attrs[exist_col],
                                                                          //x->GetBlock(exist_col), *plforms[exist_row]);
                    //pbforms.offd(exist_row,exist_col)->EliminateTestDofs(*struct_formul.essbdr_attrs[exist_row]);

                    Vector dummy(pbforms.offd(exist_row,exist_col)->Height());
                    dummy = 0.0;
                    pbforms.offd(exist_row,exist_col)->EliminateTrialDofs(*struct_formul.essbdr_attrs[exist_col],
                                                                          x->GetBlock(exist_col), dummy);
                    pbforms.offd(exist_row,exist_col)->EliminateTestDofs(*struct_formul.essbdr_attrs[exist_row]);


                    pbforms.offd(exist_row,exist_col)->Finalize();
                    hpmats(exist_row,exist_col) = pbforms.offd(exist_row,exist_col)->ParallelAssemble();
                    hpmats(exist_col, exist_row) = hpmats(exist_row,exist_col)->Transpose();
                }
            }
        }

   CFOSLSop = new BlockOperator(blkoffsets_true);
   for (int i = 0; i < numblocks; ++i)
       for (int j = 0; j < numblocks; ++j)
           CFOSLSop->SetBlock(i,j, hpmats(i,j));

   CFOSLSop_nobnd = new BlockOperator(blkoffsets_true);
   for (int i = 0; i < numblocks; ++i)
       for (int j = 0; j < numblocks; ++j)
           CFOSLSop_nobnd->SetBlock(i,j, hpmats_nobnd(i,j));

   // assembling rhs forms without boundary conditions
   for (int i = 0; i < numblocks; ++i)
   {
       plforms[i]->ParallelAssemble(trueRhs->GetBlock(i));
   }

   //trueRhs->Print();

   trueBnd = SetTrueInitialCondition();

   // moving the contribution from inhomogenous bnd conditions
   // from the rhs
   BlockVector trueBndCor(blkoffsets_true);
   trueBndCor = 0.0;

   //trueBnd->Print();

   CFOSLSop_nobnd->Mult(*trueBnd, trueBndCor);

   //trueBndCor.Print();

   *trueRhs -= trueBndCor;

   // restoring correct boundary values for boundary tdofs
   for (int i = 0; i < numblocks; ++i)
   {
       Array<int> ess_bnd_tdofs;
       pfes[i]->GetEssentialTrueDofs(*struct_formul.essbdr_attrs[i], ess_bnd_tdofs);

       for (int j = 0; j < ess_bnd_tdofs.Size(); ++j)
       {
           int tdof = ess_bnd_tdofs[j];
           trueRhs->GetBlock(i)[tdof] = trueBnd->GetBlock(i)[tdof];
       }
   }

   if (verbose)
        cout << "Final saddle point matrix assembled \n";
    MPI_Comm comm = pfes[0]->GetComm();
    MPI_Barrier(comm);
}

void CFOSLSHyperbolicProblem::InitSolver(bool verbose)
{
    MPI_Comm comm = pfes[0]->GetComm();

    int max_iter = 100000;
    double rtol = 1e-12;//1e-7;//1e-9;
    double atol = 1e-14;//1e-9;//1e-12;

    solver = new MINRESSolver(comm);
    solver->SetAbsTol(atol);
    solver->SetRelTol(rtol);
    solver->SetMaxIter(max_iter);
    solver->SetOperator(*CFOSLSop);
    if (prec)
         solver->SetPreconditioner(*prec);
    solver->SetPrintLevel(0);

    if (verbose)
        std::cout << "Here you should print out parameters of the linear solver \n";
}

// this works only for hyperbolic case
// and should be a virtual function in the abstract base
void CFOSLSHyperbolicProblem::InitPrec(int prec_option, bool verbose)
{
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

    HypreParMatrix & A = (HypreParMatrix&)CFOSLSop->GetBlock(0,0);
    HypreParMatrix * C;
    int blkcount = 1;
    if (strcmp(struct_formul.space_for_S,"H1") == 0) // S is from H1
    {
        C = &((HypreParMatrix&)CFOSLSop->GetBlock(1,1));
        ++blkcount;
    }
    HypreParMatrix & D = (HypreParMatrix&)CFOSLSop->GetBlock(blkcount,0);

    HypreParMatrix *Schur;
    if (struct_formul.have_constraint)
    {
       HypreParMatrix *AinvDt = D.Transpose();
       //FIXME: Do we actually need a hypreparvector here? Can't we just use a vector?
       //HypreParVector *Ad = new HypreParVector(comm, A.GetGlobalNumRows(),
                                            //A.GetRowStarts());
       //A.GetDiag(*Ad);
       //AinvDt->InvScaleRows(*Ad);
       Vector Ad;
       A.GetDiag(Ad);
       AinvDt->InvScaleRows(Ad);
       Schur = ParMult(&D, AinvDt);
    }

    Solver * invA;
    if (use_ADS)
        invA = new HypreADS(A, Sigma_space);
    else // using Diag(A);
         invA = new HypreDiagScale(A);

    invA->iterative_mode = false;

    Solver * invC;
    if (strcmp(struct_formul.space_for_S,"H1") == 0) // S is from H1
    {
        invC = new HypreBoomerAMG(*C);
        ((HypreBoomerAMG*)invC)->SetPrintLevel(0);
        ((HypreBoomerAMG*)invC)->iterative_mode = false;
    }

    Solver * invS;
    if (struct_formul.have_constraint)
    {
         invS = new HypreBoomerAMG(*Schur);
         ((HypreBoomerAMG *)invS)->SetPrintLevel(0);
         ((HypreBoomerAMG *)invS)->iterative_mode = false;
    }

    prec = new BlockDiagonalPreconditioner(blkoffsets_true);
    if (prec_option > 0)
    {
        int tempblknum = 0;
        prec->SetDiagonalBlock(tempblknum, invA);
        tempblknum++;
        if (strcmp(struct_formul.space_for_S,"H1") == 0) // S is present
        {
            prec->SetDiagonalBlock(tempblknum, invC);
            tempblknum++;
        }
        if (struct_formul.have_constraint)
             prec->SetDiagonalBlock(tempblknum, invS);

        if (verbose)
            std::cout << "Preconditioner built in " << chrono.RealTime() << "s. \n";
    }
    else
        if (verbose)
            cout << "No preconditioner is used. \n";
}


void CFOSLSHyperbolicProblem::Update()
{
    // update spaces
    Hdiv_space->Update();
    H1vec_space->Update();
    H1_space->Update();
    L2_space->Update();
    // this is not enough, better update all pfes as above
    //for (int i = 0; i < numblocks; ++i)
        //pfes[i]->Update();

    // update grid functions
    for (int i = 0; i < grfuns.Size(); ++i)
        grfuns[i]->Update();
}

SparseMatrix * RemoveZeroEntries(const SparseMatrix& in);

// a class for hierarchy of spaces of finite element spaces based on a nested sequence of meshes
class GeneralHierarchy
{
protected:
    int num_lvls;
    std::vector<ParMesh*> pmesh_lvls;
    std::vector<ParFiniteElementSpace* > Hdiv_space_lvls;
    std::vector<ParFiniteElementSpace* > H1_space_lvls;
    std::vector<ParFiniteElementSpace* > L2_space_lvls;

    std::vector<SparseMatrix*> P_H1_lvls;
    std::vector<SparseMatrix*> P_Hdiv_lvls;
    std::vector<SparseMatrix*> P_L2_lvls;
    std::vector<HypreParMatrix*> TrueP_H1_lvls;
    std::vector<HypreParMatrix*> TrueP_Hdiv_lvls;
    std::vector<HypreParMatrix*> TrueP_L2_lvls;

public:
    GeneralHierarchy(int num_levels, ParMesh& pmesh, int feorder, bool verbose);

    virtual void RefineAndCopy(int lvl, ParMesh* pmesh)
    {
        if (lvl == num_lvls - 1)
            pmesh_lvls[lvl] = new ParMesh(*pmesh);
        else
        {
            pmesh->UniformRefinement();
            pmesh_lvls[lvl] = new ParMesh(*pmesh);
        }
    }
};

GeneralHierarchy::GeneralHierarchy(int num_levels, ParMesh& pmesh, int feorder, bool verbose)
    : num_lvls (num_levels)
{
    int dim = pmesh.Dimension();

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
    Hdiv_space = new ParFiniteElementSpace(&pmesh, hdiv_coll);

    ParFiniteElementSpace *L2_space;
    L2_space = new ParFiniteElementSpace(&pmesh, l2_coll);

    ParFiniteElementSpace *H1_space;
    H1_space = new ParFiniteElementSpace(&pmesh, h1_coll);

    const SparseMatrix* P_Hdiv_local;
    const SparseMatrix* P_H1_local;
    const SparseMatrix* P_L2_local;

    for (int l = num_lvls - 1; l >= 0; --l)
    {
        RefineAndCopy(l, &pmesh);

        // creating pfespaces for level l
        Hdiv_space_lvls[l] = new ParFiniteElementSpace(pmesh_lvls[l], hdiv_coll);
        L2_space_lvls[l] = new ParFiniteElementSpace(pmesh_lvls[l], l2_coll);
        H1_space_lvls[l] = new ParFiniteElementSpace(pmesh_lvls[l], h1_coll);

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
        }

    } // end of loop over levels

}


class GeneralCylHierarchy : public GeneralHierarchy
{
protected:
    std::vector<ParMeshCyl*> pmeshcyl_lvls;

    std::vector<int> init_cond_size_lvls;
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
        // don't change the order of these calls
        ConstructTdofsLinks();
        ConstructRestrictions();
        ConstructInterpolations();
    }

    virtual void RefineAndCopy(int lvl, ParMeshCyl* pmesh);
};

void GeneralCylHierarchy::RefineAndCopy(int lvl, ParMeshCyl* pmeshcyl)
{
    if (lvl == num_lvls - 1)
    {
        pmeshcyl_lvls[lvl] = new ParMeshCyl(*pmeshcyl);
        pmesh_lvls[lvl] = pmeshcyl_lvls[lvl];
    }
    else
    {
        pmeshcyl->Refine(1);
        pmeshcyl_lvls[lvl] = new ParMeshCyl(*pmeshcyl);
        pmesh_lvls[lvl] = pmeshcyl_lvls[lvl];
        // be careful about the update of bot_to_top so that it doesn't get lost
    }
}

void GeneralCylHierarchy::ConstructRestrictions()
{
    Restrict_bot_H1_lvls.resize(num_lvls);
    Restrict_bot_Hdiv_lvls.resize(num_lvls);
    Restrict_top_H1_lvls.resize(num_lvls);
    Restrict_top_Hdiv_lvls.resize(num_lvls);

    for (int l = num_lvls - 1; l >= 0; --l)
    {
        Restrict_bot_H1_lvls[l] = CreateRestriction("bot", *H1_space_lvls[l], tdofs_link_H1_lvls[l]);
        Restrict_bot_Hdiv_lvls[l] = CreateRestriction("bot", *Hdiv_space_lvls[l], tdofs_link_Hdiv_lvls[l]);
        Restrict_top_H1_lvls[l] = CreateRestriction("top", *H1_space_lvls[l], tdofs_link_H1_lvls[l]);
        Restrict_top_Hdiv_lvls[l] = CreateRestriction("top", *Hdiv_space_lvls[l], tdofs_link_Hdiv_lvls[l]);
    }
}

void GeneralCylHierarchy::ConstructInterpolations()
{
    TrueP_bndbot_H1_lvls.resize(num_lvls - 1);
    TrueP_bndbot_Hdiv_lvls.resize(num_lvls - 1);
    TrueP_bndtop_H1_lvls.resize(num_lvls - 1);
    TrueP_bndtop_Hdiv_lvls.resize(num_lvls - 1);

    for (int l = num_lvls - 1; l >= 0; --l)
    {
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
}

void GeneralCylHierarchy::ConstructTdofsLinks()
{
    init_cond_size_lvls.resize(num_lvls);
    tdofs_link_H1_lvls.resize(num_lvls);
    tdofs_link_Hdiv_lvls.resize(num_lvls);

    for (int l = num_lvls - 1; l >= 0; --l)
    {
        std::vector<std::pair<int,int> > * dofs_link_H1 =
                CreateBotToTopDofsLink("linearH1",*H1_space_lvls[l], pmeshcyl_lvls[l]->bot_to_top_bels);
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

        std::vector<std::pair<int,int> > * dofs_link_RT0 =
                   CreateBotToTopDofsLink("RT0",*Hdiv_space_lvls[l], pmeshcyl_lvls[l]->bot_to_top_bels);
        std::cout << std::flush;

        tdofs_link_Hdiv_lvls[l].reserve(dofs_link_RT0->size());

        count = 0;
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
}
