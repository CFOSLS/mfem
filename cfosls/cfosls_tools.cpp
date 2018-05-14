#include <iostream>
#include "testhead.hpp"

using namespace std;

namespace mfem
{

bool CGSolver_mod::IndicesAreCorrect(const Vector& vec) const
{
    bool res = true;

    for (int i = 0; i < check_indices.Size(); ++i)
        if (fabs(vec[check_indices[i]]) > 1.0e-14)
        {
            std::cout << "index " << i << "has a nonzero value: " << vec[check_indices[i]] << "\n";
            res = false;
        }

    return res;
}

void CGSolver_mod::Mult(const Vector &b, Vector &x) const
{
   int i;
   double r0, den, nom, nom0, betanom, alpha, beta;

   //std::cout << "look at b at the entrance \n";
   //b.Print();
   std::cout << "check for b: " << IndicesAreCorrect(b) << "\n";
   MFEM_ASSERT(IndicesAreCorrect(b), "Indices check fails for b \n");

   if (iterative_mode)
   {
      oper->Mult(x, r);
      subtract(b, r, r); // r = b - A x
   }
   else
   {
      r = b;
      x = 0.0;
   }

   MFEM_ASSERT(IndicesAreCorrect(r), "Indices check fails for r \n");
   std::cout << "check for initial r: " << IndicesAreCorrect(r) << "\n";
   check_indices.Print();
   for ( int i = 0; i < check_indices.Size(); ++i)
       std::cout << r[check_indices[i]] << " ";
   std::cout << "\n";
   //std::cout << "look at the initial residual\n";
   //r.Print();

   if (prec)
   {
      prec->Mult(r, z); // z = B r
      //std::cout << "look at preconditioned residual at the entrance \n";
      //z.Print();
      d = z;
   }
   else
   {
      d = r;
   }

   std::cout << "check for initial d: " << IndicesAreCorrect(d) << "\n";
   MFEM_ASSERT(IndicesAreCorrect(b), "Indices check fails for d \n");

   //std::cout << "look at residual at the entrance \n";
   //r.Print();

   nom0 = nom = Dot(d, r);
   MFEM_ASSERT(IsFinite(nom), "nom = " << nom);

   std::cout << "nom = " << nom << "\n";

   if (print_level == 1 || print_level == 3)
   {
      cout << "   Iteration : " << setw(3) << 0 << "  (B r, r) = "
           << nom << (print_level == 3 ? " ...\n" : "\n");
   }

   r0 = std::max(nom*rel_tol*rel_tol, abs_tol*abs_tol);
   if (nom <= r0)
   {
      converged = 1;
      final_iter = 0;
      final_norm = sqrt(nom);
      return;
   }

   oper->Mult(d, z);  // z = A d
   den = Dot(z, d);
   MFEM_ASSERT(IsFinite(den), "den = " << den);

   if (print_level >= 0 && den < 0.0)
   {
      cout << "Negative denominator in step 0 of PCG: " << den << '\n';
   }

   if (den == 0.0)
   {
      converged = 0;
      final_iter = 0;
      final_norm = sqrt(nom);
      return;
   }

   // start iteration
   converged = 0;
   final_iter = max_iter;
   for (i = 1; true; )
   {
      alpha = nom/den;
      add(x,  alpha, d, x);     //  x = x + alpha d
      add(r, -alpha, z, r);     //  r = r - alpha A d

      std::cout << "check for new r: " << IndicesAreCorrect(r) << ", i = " << i << " \n";

      if (prec)
      {
         prec->Mult(r, z);      //  z = B r
         std::cout << "check for new z: " << IndicesAreCorrect(z) << ", i = " << i << " \n";
         betanom = Dot(r, z);
      }
      else
      {
         betanom = Dot(r, r);
      }

      MFEM_ASSERT(IsFinite(betanom), "betanom = " << betanom);

      if (print_level == 1)
      {
         cout << "   Iteration : " << setw(3) << i << "  (B r, r) = "
              << betanom << '\n';
      }

      if (betanom < r0)
      {
         if (print_level == 2)
         {
            cout << "Number of PCG iterations: " << i << '\n';
         }
         else if (print_level == 3)
         {
            cout << "   Iteration : " << setw(3) << i << "  (B r, r) = "
                 << betanom << '\n';
         }
         converged = 1;
         final_iter = i;
         break;
      }

      if (++i > max_iter)
      {
         break;
      }

      beta = betanom/nom;
      if (prec)
      {
         add(z, beta, d, d);   //  d = z + beta d
         std::cout << "check for new d: " << IndicesAreCorrect(d) << ", i = " << i << " \n";
      }
      else
      {
         add(r, beta, d, d);
      }
      oper->Mult(d, z);       //  z = A d
      den = Dot(d, z);
      MFEM_ASSERT(IsFinite(den), "den = " << den);
      if (den <= 0.0)
      {
         if (print_level >= 0 && Dot(d, d) > 0.0)
            cout << "PCG: The operator is not positive definite. (Ad, d) = "
                 << den << '\n';
      }
      nom = betanom;
   }
   if (print_level >= 0 && !converged)
   {
      if (print_level != 1)
      {
         if (print_level != 3)
         {
            cout << "   Iteration : " << setw(3) << 0 << "  (B r, r) = "
                 << nom0 << " ...\n";
         }
         cout << "   Iteration : " << setw(3) << final_iter << "  (B r, r) = "
              << betanom << '\n';
      }
      cout << "PCG: No convergence!" << '\n';
   }
   if (print_level >= 1 || (print_level >= 0 && !converged))
   {
      cout << "Average reduction factor = "
           << pow (betanom/nom0, 0.5/final_iter) << '\n';
   }
   final_norm = sqrt(betanom);
}

void BlkHypreOperator::Mult(const Vector &x, Vector &y) const
{
    BlockVector x_viewer(x.GetData(), block_offsets);
    BlockVector y_viewer(y.GetData(), block_offsets);

    for (int i = 0; i < numblocks; ++i)
    {
        for (int j = 0; j < numblocks; ++j)
            if (hpmats(i,j))
                hpmats(i,j)->Mult(x_viewer.GetBlock(j), y_viewer.GetBlock(i));
    }
}

void BlkHypreOperator::MultTranspose(const Vector &x, Vector &y) const
{
    BlockVector x_viewer(x.GetData(), block_offsets);
    BlockVector y_viewer(y.GetData(), block_offsets);

    for (int i = 0; i < numblocks; ++i)
    {
        for (int j = 0; j < numblocks; ++j)
            if (hpmats(i,j))
                hpmats(i,j)->MultTranspose(x_viewer.GetBlock(j), y_viewer.GetBlock(i));
    }
}

FOSLSFormulation::FOSLSFormulation(int dimension, int num_blocks, int num_unknowns, bool do_have_constraint)
    : dim(dimension),
      numblocks(num_blocks), unknowns_number(num_unknowns),
      have_constraint(do_have_constraint)
{
    blfis.SetSize(numblocks, numblocks);
    for (int i = 0; i < numblocks; ++i)
        for (int j = 0; j < numblocks; ++j)
            blfis(i,j) = NULL;
    lfis.SetSize(numblocks);
    for (int i = 0; i < numblocks; ++i)
        lfis[i] = NULL;

    blk_structure.resize(numblocks);
}

CFOSLSFormulation_HdivL2Hyper::CFOSLSFormulation_HdivL2Hyper (int dimension, int num_solution, bool verbose)
    : FOSLSFormulation(dimension, 2, 1, true), numsol(num_solution), test(dim, numsol)
{
    blfis(0,0) = new VectorFEMassIntegrator(*test.GetKtilda());
    blfis(1,0) = new VectorFEDivergenceIntegrator;

    lfis[1] = new DomainLFIntegrator(*test.GetRhs());

    InitBlkStructure();
}

void CFOSLSFormulation_HdivL2Hyper::InitBlkStructure()
{
    blk_structure[0] = std::make_pair<int,int>(1,0);
    blk_structure[1] = std::make_pair<int,int>(-1,-1);
}

const Array<SpaceName> &CFOSLSFormulation_HdivL2Hyper::GetSpacesDescriptor() const
{
    Array<SpaceName> * res = new Array<SpaceName>(numblocks);

    (*res)[0] = SpaceName::HDIV;
    (*res)[1] = SpaceName::L2;

    return *res;
}

CFOSLSFormulation_HdivH1Hyper::CFOSLSFormulation_HdivH1Hyper (int dimension, int num_solution, bool verbose)
    : FOSLSFormulation(dimension, 3, 2, true), numsol(num_solution), test(dim, numsol)
{
    blfis(0,0) = new VectorFEMassIntegrator();
    blfis(1,1) = new H1NormIntegrator(*test.GetBBt(), *test.GetBtB());
    blfis(1,0) = new VectorFEMassIntegrator(*test.GetMinB());
    blfis(2,0) = new VectorFEDivergenceIntegrator;

    lfis[1] = new GradDomainLFIntegrator(*test.GetBf());
    lfis[2] = new DomainLFIntegrator(*test.GetRhs());

    InitBlkStructure();
}

void CFOSLSFormulation_HdivH1Hyper::InitBlkStructure()
{
    blk_structure[0] = std::make_pair<int,int>(1,0);
    blk_structure[1] = std::make_pair<int,int>(0,0);
    blk_structure[2] = std::make_pair<int,int>(-1,-1);
}

const Array<SpaceName> &CFOSLSFormulation_HdivH1Hyper::GetSpacesDescriptor() const
{
    Array<SpaceName> * res = new Array<SpaceName>(numblocks);

    (*res)[0] = SpaceName::HDIV;
    (*res)[1] = SpaceName::H1;
    (*res)[2] = SpaceName::L2;

    return *res;
}

CFOSLSFEFormulation_HdivL2Hyper::CFOSLSFEFormulation_HdivL2Hyper(FOSLSFormulation& formulation, int fe_order)
    : FOSLSFEFormulation(formulation, fe_order)
{
    int dim = formul.Dim();
    if (dim == 4)
        fecolls[0] = new RT0_4DFECollection;
    else
        fecolls[0] = new RT_FECollection(feorder, dim);

    fecolls[1] = new L2_FECollection(feorder, dim);
}

CFOSLSFEFormulation_HdivH1Hyper::CFOSLSFEFormulation_HdivH1Hyper(FOSLSFormulation& formulation, int fe_order)
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

void BlockProblemForms::Update()
{
    MFEM_ASSERT(initialized_forms, "Cannot update forms which were not initialized \n");

    for (int i = 0; i < numblocks; ++i)
        for (int j = 0; j < numblocks; ++j)
        {
            if (i == j)
                diag_forms[i]->Update();
            else
                offd_forms(i,j)->Update();
        }
}

void BlockProblemForms::InitForms(FOSLSFEFormulation& fe_formul, Array<ParFiniteElementSpace*>& pfes)
{
    MFEM_ASSERT(numblocks == fe_formul.Nblocks(), "numblocks mismatch in BlockProblemForms::InitForms!");
    MFEM_ASSERT(pfes.Size() == numblocks, "size of pfes is different from numblocks in BlockProblemForms::InitForms!");

    for (int i = 0; i < numblocks; ++i)
        for (int j = 0; j < numblocks; ++j)
        {
            if (i == j)
                diag_forms[i] = new ParBilinearForm(pfes[i]);
            else
                offd_forms(i,j) = new ParMixedBilinearForm(pfes[j], pfes[i]);

            if (fe_formul.GetBlfi(i,j))
            {
                if (i == j)
                    diag_forms[i]->AddDomainIntegrator(fe_formul.GetBlfi(i,j));
                else
                    offd_forms(i,j)->AddDomainIntegrator(fe_formul.GetBlfi(i,j));
            }
        }

    initialized_forms = true;
}

FOSLSProblem::FOSLSProblem(GeneralHierarchy& Hierarchy, int level, BdrConditions& bdr_conditions,
             FOSLSFEFormulation& fe_formulation, bool verbose_)
    : pmesh(*Hierarchy.GetPmesh(level)), fe_formul(fe_formulation), bdr_conds(bdr_conditions),
      hierarchy(&Hierarchy), level_in_hierarchy(level),
      spaces_initialized(false), forms_initialized(false), system_assembled(false), solver_initialized(false),
      hierarchy_initialized(true),
      pbforms(fe_formul.Nblocks()), prec_option(0), verbose(verbose_)
{
    estimators.SetSize(0);

    InitSpacesFromHierarchy(*hierarchy, level, fe_formulation.GetFormulation()->GetSpacesDescriptor());
    spaces_initialized = true;
    InitForms();
    forms_initialized = true;
    InitGrFuns();

    AssembleSystem(verbose);
    system_assembled = true;

    //InitPrec(prec_option, verbose);
    InitSolver(verbose);
    solver_initialized = true;
}


FOSLSProblem::FOSLSProblem(ParMesh& pmesh_, BdrConditions &bdr_conditions,
                           FOSLSFEFormulation& fe_formulation, bool verbose_)
    : pmesh(pmesh_), fe_formul(fe_formulation), bdr_conds(bdr_conditions),
      hierarchy(NULL), level_in_hierarchy(-1),
      spaces_initialized(false), forms_initialized(false), system_assembled(false), solver_initialized(false),
      hierarchy_initialized(true),
      pbforms(fe_formul.Nblocks()), prec_option(0), verbose(verbose_)
{
    estimators.SetSize(0);

    InitSpaces(pmesh);
    spaces_initialized = true;
    InitForms();
    forms_initialized = true;
    InitGrFuns();

    AssembleSystem(verbose);
    system_assembled = true;

    //InitPrec(prec_option, verbose);
    InitSolver(verbose);
    solver_initialized = true;
}

void FOSLSProblem::Update()
{
    for (int i = 0; i < pfes.Size(); ++i)
    {
        pfes[i]->Update();
        pfes[i]->Dof_TrueDof_Matrix();
    }

    for (int i = 0; i < grfuns.Size(); ++i)
        grfuns[i]->Update();

    pbforms.Update();

    for (int i = 0; i < plforms.Size(); ++i)
        plforms[i]->Update();

    for (int i = 0; i < estimators.Size(); ++i)
        estimators[i]->Update();

    delete trueRhs;
    delete trueX;
    delete trueBnd;
    delete x;

    delete solver;

    if (prec)
        delete prec;

    for (int i = 0; i < hpmats.NumRows(); ++i)
        for (int j = 0; j < hpmats.NumCols(); ++j)
            if (hpmats(i,j))
                delete hpmats(i,j);

    for (int i = 0; i < hpmats_nobnd.NumRows(); ++i)
        for (int j = 0; j < hpmats_nobnd.NumCols(); ++j)
            if (hpmats_nobnd(i,j))
                delete hpmats_nobnd(i,j);

    delete CFOSLSop;
    delete CFOSLSop_nobnd;

    system_assembled = false;
    solver_initialized = false;
}

void FOSLSProblem::InitForms()
{
    // for bilinear forms
    pbforms.InitForms(fe_formul, pfes);

    // for linear forms

    plforms.SetSize(fe_formul.Nblocks());
    for (int i = 0; i < plforms.Size(); ++i)
    {
        plforms[i] = new ParLinearForm(pfes[i]);

        if (fe_formul.GetLfi(i))
        {
            plforms[i]->AddDomainIntegrator(fe_formul.GetLfi(i));
        }
    }
}

void FOSLSProblem::InitSpacesFromHierarchy(GeneralHierarchy& hierarchy, int level, const Array<SpaceName> &spaces_descriptor)
{
    pfes.SetSize(fe_formul.Nblocks());

    for (int i = 0; i < fe_formul.Nblocks(); ++i)
    {
        pfes[i] = hierarchy.GetSpace(spaces_descriptor[i], level);
    }
}


void FOSLSProblem::InitSpaces(ParMesh &pmesh)
{
    pfes.SetSize(fe_formul.Nblocks());

    for (int i = 0; i < fe_formul.Nblocks(); ++i)
        pfes[i] = new ParFiniteElementSpace(&pmesh, fe_formul.GetFeColl(i));
}

void FOSLSProblem::InitGrFuns()
{
    int numblocks = fe_formul.Nblocks();
    grfuns.SetSize(2 * numblocks);
    for (int i = 0; i < numblocks; ++i)
    {
        // grfun for a solution component
        grfuns[i] = new ParGridFunction(pfes[i]);
        // grfun for a rhs component
        grfuns[i + numblocks] = new ParGridFunction(pfes[i]);
    }
}

void FOSLSProblem::InitSolver(bool verbose)
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
    if (prec_option)
         solver->SetPreconditioner(*prec);
    solver->SetPrintLevel(0);

    if (verbose)
        std::cout << "Here you should print out parameters of the linear solver \n";
}

BlockVector * FOSLSProblem::GetTrueInitialCondition()
{
    // alias
    FOSLS_test * test = fe_formul.GetFormulation()->GetTest();

    //blkoffsets_true.Print();

    BlockVector * truebnd = new BlockVector(blkoffsets_true);
    *truebnd = 0.0;

    for (int blk = 0; blk < fe_formul.Nblocks(); ++blk)
    {
        if (fe_formul.GetFormulation()->GetPair(blk).first != -1)
        {
            ParGridFunction * exsol_pgfun = new ParGridFunction(pfes[blk]);

            int coeff_index = fe_formul.GetFormulation()->GetPair(blk).second;
            MFEM_ASSERT(coeff_index >= 0, "Value of coeff_index must be nonnegative at least \n");
            switch (fe_formul.GetFormulation()->GetPair(blk).first)
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

            Vector exsol_tdofs(pfes[blk]->TrueVSize());
            exsol_pgfun->ParallelProject(exsol_tdofs);

            Array<int>& essbdr_attrs = bdr_conds.GetBdrAttribs(blk);

            Array<int> ess_tdofs;
            pfes[blk]->GetEssentialTrueDofs(essbdr_attrs, ess_tdofs);

            for (int j = 0; j < ess_tdofs.Size(); ++j)
            {
                int tdof = ess_tdofs[j];
                truebnd->GetBlock(blk)[tdof] = exsol_tdofs[tdof];
            }

            delete exsol_pgfun;
        }
    }

    return truebnd;
}

BlockVector * FOSLSProblem::GetInitialCondition()
{
    // alias
    FOSLS_test * test = fe_formul.GetFormulation()->GetTest();

    BlockVector * init_cond = new BlockVector(blkoffsets);
    *init_cond = 0.0;

    for (int blk = 0; blk < fe_formul.Nblocks(); ++blk)
    {
        if (fe_formul.GetFormulation()->GetPair(blk).first != -1)
        {
            ParGridFunction * exsol_pgfun = new ParGridFunction(pfes[blk]);

            int coeff_index = fe_formul.GetFormulation()->GetPair(blk).second;
            MFEM_ASSERT(coeff_index >= 0, "Value of coeff_index must be nonnegative at least \n");
            switch (fe_formul.GetFormulation()->GetPair(blk).first)
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

            init_cond->GetBlock(blk) = *exsol_pgfun;

            delete exsol_pgfun;
        }
    }
    return init_cond;
}

void FOSLSProblem::BuildSystem(bool verbose)
{
    MFEM_ASSERT(spaces_initialized && forms_initialized, "Cannot build system if spaces or forms were not initialized");

    AssembleSystem(verbose);
    system_assembled = true;

    InitSolver(verbose);
    solver_initialized = true;

    CreatePrec(*CFOSLSop, prec_option, verbose);
    UpdateSolverPrec();
}

/*
FOSLSEstimator& FOSLSProblem::ExtractEstimator(bool verbose)
{
    FOSLSEstimator * res = new FOSLSEstimator(MPI_Comm& Comm, Array<ParGridFunction*>& solutions, Array2D<BilinearFormIntegrator*>& integrators, bool Verbose = false);;
    MFEM_ABORT("Not implemented");

    return *res;
}
*/

// works correctly only for problems with homogeneous initial conditions?
// see the times-stepping branch, think of how boundary conditions for off-diagonal blocks are imposed
// system is assumed to be symmetric
void FOSLSProblem::AssembleSystem(bool verbose)
{
    int numblocks = fe_formul.Nblocks();

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

    x = GetInitialCondition();

    trueRhs = new BlockVector(blkoffsets_true);
    trueX = new BlockVector(blkoffsets_true);

    for (int i = 0; i < numblocks; ++i)
        plforms[i]->Assemble();

    for (int i = 0; i < numblocks; ++i)
        *grfuns[i + numblocks] = *plforms[i];

    //plforms[1]->Print();

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

                    Array<int>& essbdr_attrs = bdr_conds.GetBdrAttribs(i);

                    Vector dummy(pbforms.diag(i)->Height());
                    dummy = 0.0;

                    //pbforms.diag(i)->EliminateEssentialBC(*struct_formul.essbdr_attrs[i],
                    pbforms.diag(i)->EliminateEssentialBC(essbdr_attrs,
                            x->GetBlock(i), dummy);
                    pbforms.diag(i)->Finalize();
                    hpmats(i,j) = pbforms.diag(i)->ParallelAssemble();

                    SparseMatrix diag;
                    hpmats(i,j)->GetDiag(diag);
                    Array<int> essbnd_tdofs;
                    //pfes[i]->GetEssentialTrueDofs(*struct_formul.essbdr_attrs[i], essbnd_tdofs);
                    pfes[i]->GetEssentialTrueDofs(essbdr_attrs, essbnd_tdofs);
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

                    Array<int>& essbdr_attrs = bdr_conds.GetBdrAttribs(exist_col);

                    Vector dummy(pbforms.offd(exist_row,exist_col)->Height());
                    dummy = 0.0;
                    //pbforms.offd(exist_row,exist_col)->EliminateTrialDofs(*struct_formul.essbdr_attrs[exist_col],

                    pbforms.offd(exist_row,exist_col)->EliminateTrialDofs(essbdr_attrs, x->GetBlock(exist_col), dummy);

                    Array<int>& essbdr_attrs2 = bdr_conds.GetBdrAttribs(exist_row);

                    //pbforms.offd(exist_row,exist_col)->EliminateTestDofs(*struct_formul.essbdr_attrs[exist_row]);
                    pbforms.offd(exist_row,exist_col)->EliminateTestDofs(essbdr_attrs2);


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

   trueBnd = GetTrueInitialCondition();

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
       Array<int>& essbdr_attrs = bdr_conds.GetBdrAttribs(i);

       Array<int> ess_bnd_tdofs;
       pfes[i]->GetEssentialTrueDofs(essbdr_attrs, ess_bnd_tdofs);

       for (int j = 0; j < ess_bnd_tdofs.Size(); ++j)
       {
           int tdof = ess_bnd_tdofs[j];
           trueRhs->GetBlock(i)[tdof] = trueBnd->GetBlock(i)[tdof];
       }
   }

   //if (verbose)
       //cout << "Final saddle point matrix and rhs assembled \n";
   MPI_Comm comm = pfes[0]->GetComm();
   MPI_Barrier(comm);
}

void FOSLSProblem::DistributeSolution() const
{
    for (int i = 0; i < fe_formul.Nblocks(); ++i)
        grfuns[i]->Distribute(&(trueX->GetBlock(i)));
}

void FOSLSProblem::ComputeBndError(const Vector& vec) const
{
    const BlockVector vec_viewer(vec.GetData(), blkoffsets_true);

    // alias
    FOSLS_test * test = fe_formul.GetFormulation()->GetTest();

    for (int blk = 0; blk < fe_formul.Nunknowns(); ++blk)
    {
        ParGridFunction * exsol_pgfun = new ParGridFunction(pfes[blk]);

        int coeff_index = fe_formul.GetFormulation()->GetPair(blk).second;

        MFEM_ASSERT(coeff_index >= 0, "Value of coeff_index must be nonnegative at least \n");
        switch (fe_formul.GetFormulation()->GetPair(blk).first)
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

        Vector exsol_tdofs(pfes[blk]->TrueVSize());
        exsol_pgfun->ParallelProject(exsol_tdofs);

        Array<int>& essbdr_attrs = bdr_conds.GetBdrAttribs(blk);

        Array<int> essbnd_tdofs;
        pfes[blk]->GetEssentialTrueDofs(essbdr_attrs, essbnd_tdofs);
        for (int i = 0; i < essbnd_tdofs.Size(); ++i)
        {
            int tdof = essbnd_tdofs[i];

            double value_ex = exsol_tdofs[tdof];
            double value_com = vec_viewer.GetBlock(blk)[tdof];

            if (fabs(value_ex - value_com) > MYZEROTOL)
            {
                std::cout << "bnd condition is violated for sigma, tdof = " << tdof << " exact value = "
                          << value_ex << ", value_com = " << value_com << ", diff = " << value_ex - value_com << "\n";
                std::cout << "rhs side at this tdof = " << trueRhs->GetBlock(blk)[tdof] << "\n";
            }
        }

        delete exsol_pgfun;
    }
}


void FOSLSProblem::ComputeError(const Vector& vec, bool verbose, bool checkbnd) const
{
    // alias
    FOSLS_test * test = fe_formul.GetFormulation()->GetTest();

    const BlockVector vec_viewer(vec.GetData(), blkoffsets_true);

    for (int i = 0; i < fe_formul.Nblocks(); ++i)
        grfuns[i]->Distribute(&(vec_viewer.GetBlock(i)));

    for (int blk = 0; blk < fe_formul.Nunknowns(); ++blk)
    {
        int order_quad = max(2, 2*fe_formul.Feorder() + 1);
        const IntegrationRule *irs[Geometry::NumGeom];
        for (int i = 0; i < Geometry::NumGeom; ++i)
        {
           irs[i] = &(IntRules.Get(i, order_quad));
        }

        double err =  0.0;
        double norm_exsol = 0.0;

        int coeff_index = fe_formul.GetFormulation()->GetPair(blk).second;

        switch (fe_formul.GetFormulation()->GetPair(blk).first)
        {
        case 0: // function coefficient
            err = grfuns[blk]->ComputeL2Error(*test->GetFuncCoeff(coeff_index), irs);
            norm_exsol = ComputeGlobalLpNorm(2, *test->GetFuncCoeff(coeff_index), pmesh, irs);
            break;
        case 1: // vector function coefficient
            err = grfuns[blk]->ComputeL2Error(*test->GetVecCoeff(coeff_index), irs);
            norm_exsol = ComputeGlobalLpNorm(2, *test->GetVecCoeff(coeff_index), pmesh, irs);
            break;
        default:
            {
                MFEM_ABORT("Unsupported type of coefficient for the call to ProjectCoefficient");
            }
            break;
        }


        //double err = grfuns[blk]->ComputeL2Error(*(Mytest.sigma), irs);
        //double norm_exsol = ComputeGlobalLpNorm(2, *(Mytest.sigma), *pmesh, irs);
        if (verbose)
            cout << "component No. " << blk << ": || error || / || exact_sol || = " << err / norm_exsol << endl;

        double projection_error = -1.0;

        ParGridFunction * exsol_pgfun = new ParGridFunction(pfes[blk]);

        MFEM_ASSERT(coeff_index >= 0, "Value of coeff_index must be nonnegative at least \n");
        switch (fe_formul.GetFormulation()->GetPair(blk).first)
        {
        case 0: // function coefficient
            exsol_pgfun->ProjectCoefficient(*test->GetFuncCoeff(coeff_index));
            projection_error = exsol_pgfun->ComputeL2Error(*test->GetFuncCoeff(coeff_index), irs);
            break;
        case 1: // vector function coefficient
            exsol_pgfun->ProjectCoefficient(*test->GetVecCoeff(coeff_index));
            projection_error = exsol_pgfun->ComputeL2Error(*test->GetVecCoeff(coeff_index), irs);
            break;
        default:
            {
                MFEM_ABORT("Unsupported type of coefficient for the call to ProjectCoefficient");
            }
            break;
        }

        if (checkbnd)
            ComputeBndError(vec);

        if (verbose)
            std::cout << "component No. " << blk << ": || exact - proj || / || exact || = "
                            << projection_error / norm_exsol << "\n";

        delete exsol_pgfun;

    }

    ComputeExtraError(vec);
}

void FOSLSProblem::ZeroBndValues(Vector& vec) const
{
    BlockVector vec_viewer(vec.GetData(), blkoffsets_true);

    int numblocks = fe_formul.Nblocks();
    for (int i = 0; i < numblocks; ++i)
    {
        Array<int>& essbdr_attrs = bdr_conds.GetBdrAttribs(i);

        Array<int> ess_bnd_tdofs;
        pfes[i]->GetEssentialTrueDofs(essbdr_attrs, ess_bnd_tdofs);

        for (int j = 0; j < ess_bnd_tdofs.Size(); ++j)
        {
            int tdof = ess_bnd_tdofs[j];
            vec_viewer.GetBlock(i)[tdof] = 0.0;
        }
    }

}


void FOSLSProblem::ComputeAnalyticalRhs(Vector& rhs) const
{
    int numblocks = fe_formul.Nblocks();

    for (int i = 0; i < numblocks; ++i)
        plforms[i]->Assemble();

    for (int i = 0; i < numblocks; ++i)
        *grfuns[i + numblocks] = *plforms[i];

    // assembling rhs forms without boundary conditions
    BlockVector rhs_viewer(rhs.GetData(), blkoffsets_true);
    for (int i = 0; i < numblocks; ++i)
    {
        plforms[i]->ParallelAssemble(rhs_viewer.GetBlock(i));
    }
}

void FOSLSProblem::Solve(bool verbose, bool compute_error) const
{
    MFEM_ASSERT(solver_initialized && system_assembled, "Either solver is not initialized or system is not assembled \n");

    *trueX = 0;

    chrono.Clear();
    chrono.Start();

    //trueRhs->Print();
    //SparseMatrix diag;
    //((HypreParMatrix&)(CFOSLSop->GetBlock(0,0))).GetDiag(diag);
    //diag.Print();

    //trueRhs->Print();

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

    bool checkbnd = false;
    if (compute_error)
        ComputeError(verbose, checkbnd);
}

void FOSLSProblem_HdivL2L2hyp::ComputeExtraError(const Vector& vec) const
{
    Hyper_test * test = dynamic_cast<Hyper_test*>(fe_formul.GetFormulation()->GetTest());

    MFEM_ASSERT(test, "Unsuccessful cast into Hyper_test*");

    // aliases
    ParFiniteElementSpace * Hdiv_space = pfes[0];
    ParFiniteElementSpace * L2_space = pfes[1];
    ParGridFunction * sigma = grfuns[0];

    int order_quad = max(2, 2*fe_formul.Feorder() + 1);
    const IntegrationRule *irs[Geometry::NumGeom];
    for (int i = 0; i < Geometry::NumGeom; ++i)
    {
       irs[i] = &(IntRules.Get(i, order_quad));
    }

    DiscreteLinearOperator Div(Hdiv_space, L2_space);
    Div.AddDomainInterpolator(new DivergenceInterpolator());
    ParGridFunction DivSigma(L2_space);
    Div.Assemble();
    Div.Mult(*sigma, DivSigma);

    double err_div = DivSigma.ComputeL2Error(*test->GetRhs(),irs);
    double norm_div = ComputeGlobalLpNorm(2, *test->GetRhs(), pmesh, irs);

    if (verbose)
    {
        cout << "|| div (sigma_h - sigma_ex) || / ||div (sigma_ex)|| = "
                  << err_div/norm_div  << "\n";
    }

    ParBilinearForm *Cblock = new ParBilinearForm(L2_space);
    Cblock->AddDomainIntegrator(new MassIntegrator(*test->GetBtB()));
    Cblock->Assemble();
    Cblock->Finalize();
    HypreParMatrix * C = Cblock->ParallelAssemble();

    ParMixedBilinearForm *Bblock = new ParMixedBilinearForm(Hdiv_space, L2_space);
    Bblock->AddDomainIntegrator(new VectorFEMassIntegrator(*test->GetB()));
    Bblock->Assemble();
    Bblock->Finalize();
    HypreParMatrix * B = Bblock->ParallelAssemble();

    Vector bTsigma(C->Height());
    B->Mult(trueX->GetBlock(0),bTsigma);

    Vector trueS(C->Height());

    CG(*C, bTsigma, trueS, 0, 5000, 1e-9, 1e-12);

    ParGridFunction * S = new ParGridFunction(L2_space);
    S->Distribute(trueS);

    delete Cblock;
    delete Bblock;
    delete B;
    delete C;

    double err_S = S->ComputeL2Error(*test->GetU(), irs);
    double norm_S = ComputeGlobalLpNorm(2, *test->GetU(), pmesh, irs);
    if (verbose)
    {
        std::cout << "|| S_h - S_ex || / || S_ex || = " <<
                     err_S / norm_S << "\n";
    }

    ParGridFunction * S_exact = new ParGridFunction(L2_space);
    S_exact->ProjectCoefficient(*test->GetU());

    double projection_error_S = S_exact->ComputeL2Error(*test->GetU(), irs);

    if (verbose)
        std::cout << "|| S_ex - Pi_h S_ex || / || S_ex || = "
                        << projection_error_S / norm_S << "\n";

    delete S;
}

// prec_option:
// 0 for no preconditioner
// 1 for diag(A) + BoomerAMG (Bt diag(A)^-1 B)
// 2 for ADS(A) + BommerAMG (Bt diag(A)^-1 B)
void FOSLSProblem_HdivL2L2hyp::CreatePrec(BlockOperator& op, int prec_option, bool verbose)
{
    MFEM_ASSERT(prec_option >= 0, "Invalid prec option was provided");

    if (verbose)
    {
        std::cout << "Block diagonal preconditioner: \n";
        if (prec_option == 2)
            std::cout << "ADS(A) for H(div) \n";
        else
             std::cout << "Diag(A) for H(div) or H1vec \n";

        std::cout << "BoomerAMG(D Diag^(-1)(A) D^t) for L2 lagrange multiplier \n";
    }

    HypreParMatrix & A = ((HypreParMatrix&)(CFOSLSop->GetBlock(0,0)));
    HypreParMatrix & D = ((HypreParMatrix&)(CFOSLSop->GetBlock(1,0)));


    HypreParMatrix *Schur;

    HypreParMatrix *AinvDt = D.Transpose();
    HypreParVector *Ad = new HypreParVector(MPI_COMM_WORLD, A.GetGlobalNumRows(),
                                         A.GetRowStarts());
    A.GetDiag(*Ad);
    AinvDt->InvScaleRows(*Ad);
    Schur = ParMult(&D, AinvDt);

    Solver * invA;
    if (prec_option == 2)
        invA = new HypreADS(A, pfes[0]);
    else // using Diag(A);
        invA = new HypreDiagScale(A);

    invA->iterative_mode = false;

    Solver * invS = new HypreBoomerAMG(*Schur);
    ((HypreBoomerAMG *)invS)->SetPrintLevel(0);
    ((HypreBoomerAMG *)invS)->iterative_mode = false;

    prec = new BlockDiagonalPreconditioner(blkoffsets_true);
    if (prec_option > 0)
    {
        ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(0, invA);
        ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(1, invS);
    }
    else
        if (verbose)
            cout << "No preconditioner is used. \n";

}


// prec_option:
// 0 for no preconditioner
// 1 for diag(A) + BoomerAMG (Bt diag(A)^-1 B)
// 2 for ADS(A) + BommerAMG (Bt diag(A)^-1 B)
void FOSLSProblem_HdivH1L2hyp::CreatePrec(BlockOperator& op, int prec_option, bool verbose)
{
    MFEM_ASSERT(prec_option >= 0, "Invalid prec option was provided");

    if (verbose)
    {
        std::cout << "Block diagonal preconditioner: \n";
        std::cout << "Diag(A) for H(div) \n";
        std::cout << "BoomerAMG(C) for H1 \n";
        std::cout << "BoomerAMG(D Diag^(-1)(A) D^t) for the Lagrange multiplier \n";
    }

    HypreParMatrix & A = ((HypreParMatrix&)(CFOSLSop->GetBlock(0,0)));
    HypreParMatrix & C = ((HypreParMatrix&)(CFOSLSop->GetBlock(1,1)));
    HypreParMatrix & D = ((HypreParMatrix&)(CFOSLSop->GetBlock(2,0)));

    HypreParMatrix *Schur;

    HypreParMatrix *AinvDt = D.Transpose();
    HypreParVector *Ad = new HypreParVector(MPI_COMM_WORLD, A.GetGlobalNumRows(),
                                         A.GetRowStarts());
    A.GetDiag(*Ad);
    AinvDt->InvScaleRows(*Ad);
    Schur = ParMult(&D, AinvDt);

    Solver * invA;
    invA = new HypreDiagScale(A);
    invA->iterative_mode = false;

    Solver * invC = new HypreBoomerAMG(C);
    ((HypreBoomerAMG*)invC)->SetPrintLevel(0);
    ((HypreBoomerAMG*)invC)->iterative_mode = false;

    Solver * invS = new HypreBoomerAMG(*Schur);
    ((HypreBoomerAMG *)invS)->SetPrintLevel(0);
    ((HypreBoomerAMG *)invS)->iterative_mode = false;

    prec = new BlockDiagonalPreconditioner(blkoffsets_true);
    if (prec_option > 0)
    {
        ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(0, invA);
        ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(1, invC);
        ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(2, invS);
    }
    else
        if (verbose)
            cout << "No preconditioner is used. \n";

}

GeneralMultigrid::GeneralMultigrid(int Nlevels, const Array<Operator*> &P_lvls_, const Array<Operator*> &Op_lvls_,
                                   const Operator& CoarseOp_,
                 const Array<Operator*> &PreSmoothers_lvls_, const Array<Operator*> &PostSmoothers_lvls_)
    : Solver(Op_lvls_[0]->Height()), nlevels(Nlevels), P_lvls(P_lvls_), Op_lvls(Op_lvls_), CoarseOp(CoarseOp_),
      PreSmoothers_lvls(PreSmoothers_lvls_), PostSmoothers_lvls(PostSmoothers_lvls_),
      symmetric(false), current_level(0)
{
    MFEM_ASSERT(nlevels <= P_lvls.Size() + 1, "Number of interpolation matrices cannot be less"
                                              " than number of levels - 1");
    MFEM_ASSERT(nlevels <= PreSmoothers_lvls.Size() + 1, "Number of pre-smoothers cannot be less "
                                                         "than number of levels - 1");
    MFEM_ASSERT(nlevels <= PostSmoothers_lvls.Size() + 1, "Number of post-smoothers cannot be less "
                                                          "than number of levels - 1");

    residual.SetSize(nlevels);
    correction.SetSize(nlevels);
    for (int l = 0; l < nlevels; l++)
    {
        if (l < nlevels - 1)
        {
            residual[l] = new Vector(Op_lvls[l]->Height());
            if (l > 0)
                correction[l] = new Vector(Op_lvls[l]->Width());
            else
                correction[l] = new Vector();
        }
        else // exist because of SetDataAndSize call to correction.Last() in GeneralMultigrid::Mult
             // which drops the  data (if allocated here, i.e. if no if-clause)
        {
            residual[l] = new Vector(CoarseOp.Height());
            correction[l] = new Vector(CoarseOp.Height());
        }
    }

}

void GeneralMultigrid::Mult(const Vector & x, Vector & y) const
{
    *residual[0] = x;

    correction[0]->SetDataAndSize(y.GetData(), y.Size());
    MG_Cycle();
}

void GeneralMultigrid::MG_Cycle() const
{
    Operator * Operator_l, *PreSmoother_l, *PostSmoother_l;
    if (current_level < nlevels - 1)
    {
        Operator_l = Op_lvls[current_level];
        PreSmoother_l = PreSmoothers_lvls[current_level];
        PostSmoother_l = PostSmoothers_lvls[current_level];
    }

    Vector& residual_l = *residual[current_level];
    Vector& correction_l = *correction[current_level];

    Vector help(residual_l.Size());
    help = 0.0;

    // PreSmoothing
    if (current_level < nlevels - 1 && PreSmoother_l)
    {
        //std::cout << "residual before presmoothing, new MG \n";
        //residual_l.Print();

        //std::cout << "residual before smoothing, new MG, "
                     //"norm = " << residual_l.Norml2() / sqrt (residual_l.Size()) << "\n";

        PreSmoother_l->Mult(residual_l, correction_l);

        //std::cout << "correction after smoothing, new MG, "
                     //"norm = " << correction_l.Norml2() / sqrt (correction_l.Size()) << "\n";

        //std::cout << "correction after presmoothing, new MG \n";
        //correction_l.Print();

        Operator_l->Mult(correction_l, help);
        residual_l -= help;

        //std::cout << "help, new MG, "
                     //"norm = " << help.Norml2() / sqrt (help.Size()) << "\n";

        //std::cout << "help, new MG \n";
        //help.Print();


        //std::cout << "new residual after presmoothing, new MG, "
                     //"norm = " << residual_l.Norml2() / sqrt (residual_l.Size()) << "\n";
        //residual_l.Print();
    }


    // Coarse grid correction
    if (current_level < nlevels - 1)
    {
        const Operator& P_l = *P_lvls[current_level];

        P_l.MultTranspose(residual_l, *residual[current_level + 1]);

        //std::cout << "residual after coarsening, new MG, "
                     //"norm = " << residual[current_level + 1]->Norml2() / sqrt (residual[current_level + 1]->Size()) << "\n";

        //std::cout << "residual after projecting onto coarser level, new MG \n";
        //residual[current_level + 1]->Print();

        current_level++;
        MG_Cycle();
        current_level--;

        cor_cor.SetSize(residual_l.Size());
        P_l.Mult(*correction[current_level + 1], cor_cor);

        //cor_cor.Print();

        correction_l += cor_cor;
        Operator_l->Mult(cor_cor, help);
        residual_l -= help;
    }
    else
    {
        //std::cout << "residual at the coarsest level, new MG, "
                     //"norm = " << residual_l.Norml2() / sqrt (residual_l.Size()) << "\n";

        CoarseOp.Mult(residual_l, correction_l);

        //std::cout << "correction at the coarsest level, new MG, "
                     //"norm = " << correction_l.Norml2() / sqrt (correction_l.Size()) << "\n";
    }

    // PostSmoothing
    if (current_level < nlevels - 1)
    {
        if (symmetric)
        {
            if (PreSmoother_l)
                PreSmoother_l->MultTranspose(residual_l, cor_cor);
        }
        else // nonsymmetric
            if (PostSmoother_l)
                PostSmoother_l->Mult(residual_l, cor_cor);
        correction_l += cor_cor;
    }

}

RAPBlockHypreOperator::RAPBlockHypreOperator(BlockOperator &Rt_, BlockOperator &A_, BlockOperator &P_,
                                             const Array<int>& Offsets)
   : BlockOperator(Offsets),
     nblocks(A_.NumRowBlocks()),
     offsets(Offsets)
{
    for (int i = 0; i < nblocks; ++i)
        for (int j = 0; j < nblocks; ++j)
        {
            //Operator& Rt_blk_i = Rt.GetBlock(i,i);
            //Operator& P_blk_j = P.GetBlock(j,j);
            //Operator& A_blk_ij = A.GetBlock(i,j);

            HypreParMatrix* Rt_blk_i = dynamic_cast<HypreParMatrix*>(&(Rt_.GetBlock(i,i)));
            HypreParMatrix* P_blk_j = dynamic_cast<HypreParMatrix*>(&(P_.GetBlock(j,j)));
            HypreParMatrix* A_blk_ij = dynamic_cast<HypreParMatrix*>(&(A_.GetBlock(i,j)));
            HypreParMatrix * op_block = RAP(Rt_blk_i, A_blk_ij, P_blk_j);

            SetBlock(i,j, op_block);
        }

}

void BlkInterpolationWithBNDforTranspose::MultTranspose(const Vector &x, Vector &y) const
{
    P.MultTranspose(x, y);

    //bnd_indices->Print();

    for (int i = 0; i < bnd_indices->Size(); ++i)
    {
        int index = (*bnd_indices)[i];
        y[index] = 0.0;
    }
}

void InterpolationWithBNDforTranspose::MultTranspose(const Vector &x, Vector &y) const
{
    P.MultTranspose(x, y);

    //bnd_indices->Print();

    for (int i = 0; i < bnd_indices->Size(); ++i)
    {
        int index = (*bnd_indices)[i];
        y[index] = 0.0;
    }
}

//##############################################################################################

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

GeneralHierarchy::GeneralHierarchy(int num_levels, ParMesh& pmesh, int feorder, bool verbose)
    : num_lvls(num_levels), divfreedops_constructed (false)
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

    FiniteElementCollection *hcurl_coll;
    if (dim == 4)
        hcurl_coll = new ND1_4DFECollection;
    else
        hcurl_coll = new ND_FECollection(feorder + 1, dim);

    FiniteElementCollection *hdivskew_coll;
    if (dim == 4)
        hdivskew_coll = new DivSkew1_4DFECollection;
    else
        hdivskew_coll = NULL;


    ParFiniteElementSpace *Hdiv_space;
    Hdiv_space = new ParFiniteElementSpace(&pmesh, hdiv_coll);

    ParFiniteElementSpace *L2_space;
    L2_space = new ParFiniteElementSpace(&pmesh, l2_coll);

    ParFiniteElementSpace *H1_space;
    H1_space = new ParFiniteElementSpace(&pmesh, h1_coll);

    ParFiniteElementSpace *Hcurl_space;
    Hcurl_space = new ParFiniteElementSpace(&pmesh, hcurl_coll);

    ParFiniteElementSpace *Hdivskew_space;
    if (dim == 4)
        Hdivskew_space = new ParFiniteElementSpace(&pmesh, hdivskew_coll);

    const SparseMatrix* P_Hdiv_local;
    const SparseMatrix* P_H1_local;
    const SparseMatrix* P_L2_local;
    const SparseMatrix* P_Hcurl_local;
    const SparseMatrix* P_Hdivskew_local;

    pmesh_lvls.resize(num_lvls);
    Hdiv_space_lvls.resize(num_lvls);
    H1_space_lvls.resize(num_lvls);
    L2_space_lvls.resize(num_lvls);
    Hcurl_space_lvls.resize(num_lvls);
    if (dim == 4)
        Hdivskew_space_lvls.resize(num_lvls);
    P_Hdiv_lvls.resize(num_lvls - 1);
    P_H1_lvls.resize(num_lvls - 1);
    P_L2_lvls.resize(num_lvls - 1);
    P_Hcurl_lvls.resize(num_lvls - 1);
    if (dim == 4)
        P_Hdivskew_lvls.resize(num_lvls - 1);
    TrueP_Hdiv_lvls.resize(num_lvls - 1);
    TrueP_H1_lvls.resize(num_lvls - 1);
    TrueP_L2_lvls.resize(num_lvls - 1);
    TrueP_Hcurl_lvls.resize(num_lvls - 1);
    if (dim == 4)
        TrueP_Hdivskew_lvls.resize(num_lvls - 1);

    //std::cout << "Checking test for dynamic cast \n";
    //if (dynamic_cast<testB*> (testA))
        //std::cout << "Unsuccessful cast \n";

    for (int l = num_lvls - 1; l >= 0; --l)
    {
        RefineAndCopy(l, &pmesh);

        // creating pfespaces for level l
        Hdiv_space_lvls[l] = new ParFiniteElementSpace(pmesh_lvls[l], hdiv_coll);
        L2_space_lvls[l] = new ParFiniteElementSpace(pmesh_lvls[l], l2_coll);
        H1_space_lvls[l] = new ParFiniteElementSpace(pmesh_lvls[l], h1_coll);
        Hcurl_space_lvls[l] = new ParFiniteElementSpace(pmesh_lvls[l], hcurl_coll);
        if (dim == 4)
            Hdivskew_space_lvls[l] = new ParFiniteElementSpace(pmesh_lvls[l], hdivskew_coll);

        // for all but one levels we create projection matrices between levels
        // and projectors assembled on true dofs if MG preconditioner is used
        if (l < num_lvls - 1)
        {
            Hdiv_space->Update();
            H1_space->Update();
            L2_space->Update();
            Hcurl_space->Update();
            if (dim == 4)
                Hdivskew_space->Update();

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

            P_Hcurl_local = (SparseMatrix *)Hcurl_space->GetUpdateOperator();
            P_Hcurl_lvls[l] = RemoveZeroEntries(*P_Hcurl_local);

            auto d_td_coarse_Hcurl = Hcurl_space_lvls[l + 1]->Dof_TrueDof_Matrix();
            SparseMatrix * RP_Hcurl_local = Mult(*Hcurl_space_lvls[l]->GetRestrictionMatrix(), *P_Hcurl_lvls[l]);
            TrueP_Hcurl_lvls[l] = d_td_coarse_Hcurl->LeftDiagMult(
                        *RP_Hcurl_local, Hcurl_space_lvls[l]->GetTrueDofOffsets());
            TrueP_Hcurl_lvls[l]->CopyColStarts();
            TrueP_Hcurl_lvls[l]->CopyRowStarts();

            delete RP_Hcurl_local;

            if (dim == 4)
            {
                P_Hdivskew_local = (SparseMatrix *)Hdivskew_space->GetUpdateOperator();
                P_Hdivskew_lvls[l] = RemoveZeroEntries(*P_Hdivskew_local);

                auto d_td_coarse_Hdivskew = Hdivskew_space_lvls[l + 1]->Dof_TrueDof_Matrix();
                SparseMatrix * RP_Hdivskew_local = Mult(*Hdivskew_space_lvls[l]->GetRestrictionMatrix(), *P_Hdivskew_lvls[l]);
                TrueP_Hdivskew_lvls[l] = d_td_coarse_Hdivskew->LeftDiagMult(
                            *RP_Hdivskew_local, Hdivskew_space_lvls[l]->GetTrueDofOffsets());
                TrueP_Hdivskew_lvls[l]->CopyColStarts();
                TrueP_Hdivskew_lvls[l]->CopyRowStarts();

            }

        }

    } // end of loop over levels

}

void GeneralHierarchy::ConstructDivfreeDops()
{
    int dim = pmesh_lvls[0]->Dimension();

    DivfreeDops_lvls.resize(num_lvls);

    for (int l = 0; l < num_lvls; ++l)
    {
        ParDiscreteLinearOperator * Divfree_op;
        if (dim == 3)
        {
            Divfree_op = new ParDiscreteLinearOperator(Hcurl_space_lvls[l], Hdiv_space_lvls[l]);
            Divfree_op->AddDomainInterpolator(new CurlInterpolator);
        }
        else
        {
            Divfree_op = new ParDiscreteLinearOperator(Hdivskew_space_lvls[l], Hdiv_space_lvls[l]);
            Divfree_op->AddDomainInterpolator(new DivSkewInterpolator);
        }

        Divfree_op->Assemble();
        Divfree_op->Finalize();
        DivfreeDops_lvls[l] = Divfree_op->ParallelAssemble();
    }

    divfreedops_constructed = true;
}


const Array<int>& GeneralHierarchy::ConstructOffsetsforFormul(int level, const Array<SpaceName>& space_names)
{
    Array<int> * res = new Array<int>(space_names.Size() + 1);

    (*res)[0] = 0;
    for (int i = 0; i < space_names.Size(); ++i)
        (*res)[i + 1] = GetSpace(space_names[i], level)->TrueVSize();
    res->PartialSum();

    return *res;
}


BlockOperator* GeneralHierarchy::ConstructTruePforFormul(int level, const Array<SpaceName>& space_names,
                                                         const Array<int>& row_offsets, const Array<int>& col_offsets)
{
    BlockOperator * res = new BlockOperator(row_offsets, col_offsets);

    for (int i = 0; i < space_names.Size(); ++i)
        res->SetDiagonalBlock(i, GetTruePspace(space_names[i], level), 1.0);

    return res;
}



BlockOperator* GeneralHierarchy::ConstructTruePforFormul(int level, const FOSLSFormulation& formul,
                                                         const Array<int>& row_offsets, const Array<int>& col_offsets)
{
    const Array<SpaceName> & space_names  = formul.GetSpacesDescriptor();
    return ConstructTruePforFormul(level, space_names, row_offsets, col_offsets);
}

const Array<int>& GeneralHierarchy::GetEssBdrTdofsOrDofs(const char * tdof_or_dof,
                                                         SpaceName space_name, const Array<int>& essbdr_attribs,
                                                         int level) const
{
    MFEM_ASSERT(strcmp(tdof_or_dof,"dof") == 0 || strcmp(tdof_or_dof,"tdof") == 0,
                "First argument must be 'dof' or 'tdof' \n");
    ParFiniteElementSpace * pfes;
    switch(space_name)
    {
    case HDIV:
        pfes = Hdiv_space_lvls[level];
        break;
    case H1:
        pfes = H1_space_lvls[level];
        break;
    case L2:
        pfes = L2_space_lvls[level];
        break;
    case HCURL:
        pfes = Hcurl_space_lvls[level];
        break;
    case HDIVSKEW:
        pfes = Hdivskew_space_lvls[level];
        break;
    default:
        {
            MFEM_ABORT("Unknown or unsupported space name \n");
            break;
        }
    }

    Array<int>* res = new Array<int>;

    if (strcmp(tdof_or_dof, "tdof") == 0)
        pfes->GetEssentialTrueDofs(essbdr_attribs, *res);
    else // dof
        pfes->GetEssentialVDofs(essbdr_attribs, *res);

    return *res;
}

std::vector<Array<int>* >& GeneralHierarchy::GetEssBdrTdofsOrDofs(const char * tdof_or_dof,
                                                                  const Array<SpaceName>& space_names,
                                                                  std::vector<const Array<int>*>& essbdr_attribs,
                                                                  int level) const
{
    MFEM_ASSERT(strcmp(tdof_or_dof,"dof") == 0 || strcmp(tdof_or_dof,"tdof") == 0,
                "First argument must be 'dof' or 'tdof' \n");

    ParFiniteElementSpace* pfes;
    std::vector<Array<int>* > * res = new std::vector<Array<int>* >();
    res->resize(space_names.Size());

    for (int i = 0; i < space_names.Size(); ++i)
    {
        switch(space_names[i])
        {
        case HDIV:
            pfes = Hdiv_space_lvls[level];
            break;
        case H1:
            pfes = H1_space_lvls[level];
            break;
        case L2:
            pfes = L2_space_lvls[level];
            break;
        case HCURL:
            pfes = Hcurl_space_lvls[level];
            break;
        case HDIVSKEW:
            pfes = Hdivskew_space_lvls[level];
            break;
        default:
            {
                MFEM_ABORT("Unknown or unsupported space name \n");
                break;
            }
        }

        (*res)[i] = new Array<int>();

        if (strcmp(tdof_or_dof, "tdof") == 0)
            pfes->GetEssentialTrueDofs(*essbdr_attribs[i], *(*res)[i]);
        else
            pfes->GetEssentialVDofs(*essbdr_attribs[i], *(*res)[i]);
    }

    return *res;
}

std::vector<Array<int>* >& GeneralHierarchy::GetEssBdrTdofsOrDofs(const char * tdof_or_dof,
                                                            const Array<SpaceName> &space_names,
                                                            std::vector<Array<int>*>& essbdr_attribs,
                                                            int level) const
{
    MFEM_ASSERT(strcmp(tdof_or_dof,"dof") == 0 || strcmp(tdof_or_dof,"tdof") == 0,
                "First argument must be 'dof' or 'tdof' \n");

    ParFiniteElementSpace* pfes;
    std::vector<Array<int>* > * res = new std::vector<Array<int>* >();
    res->resize(space_names.Size());

    for (int i = 0; i < space_names.Size(); ++i)
    {
        switch(space_names[i])
        {
        case HDIV:
            pfes = Hdiv_space_lvls[level];
            break;
        case H1:
            pfes = H1_space_lvls[level];
            break;
        case L2:
            pfes = L2_space_lvls[level];
            break;
        case HCURL:
            pfes = Hcurl_space_lvls[level];
            break;
        case HDIVSKEW:
            pfes = Hdivskew_space_lvls[level];
            break;
        default:
            {
                MFEM_ABORT("Unknown or unsupported space name \n");
                break;
            }
        }

        (*res)[i] = new Array<int>();

        if (strcmp(tdof_or_dof, "tdof") == 0)
            pfes->GetEssentialTrueDofs(*essbdr_attribs[i], *(*res)[i]);
        else
            pfes->GetEssentialVDofs(*essbdr_attribs[i], *(*res)[i]);
    }

    return *res;
}

ParFiniteElementSpace * GeneralHierarchy::GetSpace(SpaceName space, int level)
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

HypreParMatrix * GeneralHierarchy::GetTruePspace(SpaceName space, int level)
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

SparseMatrix * GeneralHierarchy::GetPspace(SpaceName space, int level)
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

void GeneralHierarchy::RefineAndCopy(int lvl, ParMesh* pmesh)
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

SparseMatrix& GeneralHierarchy::GetElementToDofs(SpaceName space_name, int level) const
{
    ParFiniteElementSpace * pfes;
    switch(space_name)
    {
    case HDIV:
        pfes = Hdiv_space_lvls[level];
        break;
    case H1:
        pfes = H1_space_lvls[level];
        break;
    case L2:
        pfes = L2_space_lvls[level];
        break;
    case HCURL:
        pfes = Hcurl_space_lvls[level];
        break;
    case HDIVSKEW:
        pfes = Hdivskew_space_lvls[level];
        break;
    default:
        {
            MFEM_ABORT("Unknown or unsupported space name \n");
            break;
        }
    }

    return ElementToDofs(*pfes);
}

BlockMatrix& GeneralHierarchy::GetElementToDofs(const Array<SpaceName>& space_names, int level,
                                                Array<int>& row_offsets, Array<int>& col_offsets) const
{
    Array<ParFiniteElementSpace*> pfess(space_names.Size());

    row_offsets.SetSize(space_names.Size() + 1);
    row_offsets[0] = 0;
    col_offsets.SetSize(space_names.Size() + 1);
    col_offsets[0] = 0;

    for (int i = 0; i < pfess.Size(); ++i)
    {
        switch(space_names[i])
        {
        case HDIV:
            pfess[i] = Hdiv_space_lvls[level];
            break;
        case H1:
            pfess[i] = H1_space_lvls[level];
            break;
        case L2:
            pfess[i] = L2_space_lvls[level];
            break;
        case HCURL:
            pfess[i] = Hcurl_space_lvls[level];
            break;
        case HDIVSKEW:
            pfess[i] = Hdivskew_space_lvls[level];
            break;
        default:
            {
                MFEM_ABORT("Unknown or unsupported space name \n");
                break;
            }
        }

        row_offsets[i + 1] = pfess[i]->GetNE();
        col_offsets[i + 1] = pfess[i]->GetVSize();
    }

    row_offsets.PartialSum();
    col_offsets.PartialSum();

    BlockMatrix * res = new BlockMatrix(row_offsets, col_offsets);

    for (int i = 0; i < res->NumRowBlocks(); ++i)
    {
        SparseMatrix * el2dofs_blk = &ElementToDofs(*pfess[i]);
        res->SetBlock(i,i, el2dofs_blk);
    }

    res->owns_blocks = true;

    return * res;
}

HypreParMatrix& GeneralHierarchy::GetDofTrueDof(SpaceName space_name, int level) const
{
    ParFiniteElementSpace * pfes;
    switch(space_name)
    {
    case HDIV:
        pfes = Hdiv_space_lvls[level];
        break;
    case H1:
        pfes = H1_space_lvls[level];
        break;
    case L2:
        pfes = L2_space_lvls[level];
        break;
    case HCURL:
        pfes = Hcurl_space_lvls[level];
        break;
    case HDIVSKEW:
        pfes = Hdivskew_space_lvls[level];
        break;
    default:
        {
            MFEM_ABORT("Unknown or unsupported space name \n");
            break;
        }
    }

    HypreParMatrix * temp = pfes->Dof_TrueDof_Matrix();

    return *CopyHypreParMatrix(*temp);
}

std::vector<HypreParMatrix*> & GeneralHierarchy::GetDofTrueDof(const Array<SpaceName> &space_names, int level) const
{
    std::vector<HypreParMatrix*> * res = new std::vector<HypreParMatrix*>;
    res->resize(space_names.Size());
    for (int i = 0; i < space_names.Size(); ++i)
    {
        (*res)[i] = &GetDofTrueDof(space_names[i], level);
    }

    return *res;
}

BlockOperator* GeneralHierarchy::GetDofTrueDof(const Array<SpaceName>& space_names, int level,
                                               Array<int>& row_offsets, Array<int>& col_offsets) const
{
    std::vector<HypreParMatrix*> & temp = GetDofTrueDof(space_names, level);

    row_offsets.SetSize(space_names.Size() + 1);
    row_offsets[0] = 0;

    col_offsets.SetSize(row_offsets.Size());
    col_offsets[0] = 0;

    for (int i = 0; i < space_names.Size(); ++i)
    {
        row_offsets[i + 1] = temp[i]->Height();
        col_offsets[i + 1] = temp[i]->Width();
    }

    row_offsets.PartialSum();
    col_offsets.PartialSum();

    BlockOperator * res = new BlockOperator(row_offsets, col_offsets);
    for (int i = 0; i < space_names.Size(); ++i)
        res->SetDiagonalBlock(i, temp[i]);

    return res;
}



/*
void GeneralCylHierarchy::RefineAndCopy(int lvl, ParMesh* pmesh)
{
    if (lvl == num_lvls - 1)
        pmesh_lvls[lvl] = new ParMesh(*pmesh);
    else
    {
        ParMeshCyl * pmeshcyl_view = dynamic_cast<ParMeshCyl*> (pmesh);
        if (!pmeshcyl_view)
        {
            MFEM_ABORT("Dynamic cast into ParMeshCyl returned NULL \n");
        }
        pmeshcyl_view->Refine(1);
        pmesh_lvls[lvl] = new ParMesh(*pmesh);
    }
}
*/

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

    for (int l = num_lvls - 2; l >= 0; --l)
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
    //init_cond_size_lvls.resize(num_lvls);
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

SparseMatrix * RemoveZeroEntries(const SparseMatrix& in)
{
    int * I = in.GetI();
    int * J = in.GetJ();
    double * Data = in.GetData();
    double * End = Data+in.NumNonZeroElems();

    int nnz = 0;
    for (double * data_ptr = Data; data_ptr != End; data_ptr++)
    {
        if (*data_ptr != 0)
            nnz++;
    }

    int * outI = new int[in.Height()+1];
    int * outJ = new int[nnz];
    double * outData = new double[nnz];
    nnz = 0;
    for (int i = 0; i < in.Height(); i++)
    {
        outI[i] = nnz;
        for (int j = I[i]; j < I[i+1]; j++)
        {
            if (Data[j] !=0)
            {
                outJ[nnz] = J[j];
                outData[nnz++] = Data[j];
            }
        }
    }
    outI[in.Height()] = nnz;

    return new SparseMatrix(outI, outJ, outData, in.Height(), in.Width());
}

// Eliminates all entries in the Operator acting in a pair of spaces,
// assembled as a HypreParMatrix, which connect internal dofs to boundary dofs
// Used to modife the Curl and Divskew operator for the new multigrid solver
void Eliminate_ib_block(HypreParMatrix& Op_hpmat, const Array<int>& EssBdrTrueDofs_dom, const Array<int>& EssBdrTrueDofs_range )
{
    MPI_Comm comm = Op_hpmat.GetComm();

    int ntdofs_dom = Op_hpmat.Width();
    Array<int> btd_flags(ntdofs_dom);
    btd_flags = 0;
    //if (verbose)
        //std::cout << "EssBdrTrueDofs_dom \n";
    //EssBdrTrueDofs_dom.Print();

    for ( int i = 0; i < EssBdrTrueDofs_dom.Size(); ++i )
    {
        int tdof = EssBdrTrueDofs_dom[i];
        btd_flags[tdof] = 1;
    }

    int * td_btd_i = new int[ ntdofs_dom + 1];
    td_btd_i[0] = 0;
    for (int i = 0; i < ntdofs_dom; ++i)
        td_btd_i[i + 1] = td_btd_i[i] + 1;

    int * td_btd_j = new int [td_btd_i[ntdofs_dom]];
    double * td_btd_data = new double [td_btd_i[ntdofs_dom]];
    for (int i = 0; i < ntdofs_dom; ++i)
    {
        td_btd_j[i] = i;
        if (btd_flags[i] != 0)
            td_btd_data[i] = 1.0;
        else
            td_btd_data[i] = 0.0;
    }

    SparseMatrix * td_btd_diag = new SparseMatrix(td_btd_i, td_btd_j, td_btd_data, ntdofs_dom, ntdofs_dom);

    HYPRE_Int * row_starts = Op_hpmat.GetColStarts();

    HypreParMatrix * td_btd_hpmat = new HypreParMatrix(comm, Op_hpmat.N(),
            row_starts, td_btd_diag);
    td_btd_hpmat->CopyColStarts();
    td_btd_hpmat->CopyRowStarts();

    HypreParMatrix * C_td_btd = ParMult(&Op_hpmat, td_btd_hpmat);

    // processing local-to-process block of the Divfree matrix
    SparseMatrix C_td_btd_diag;
    C_td_btd->GetDiag(C_td_btd_diag);

    //C_td_btd_diag.Print();

    SparseMatrix C_diag;
    Op_hpmat.GetDiag(C_diag);

    //C_diag.Print();

    int ntdofs_range = Op_hpmat.Height();

    //std::cout << "Op_hpmat = " << Op_hpmat.Height() << " x " << Op_hpmat.Width() << "\n";
    Array<int> btd_flags_range(ntdofs_range);
    btd_flags_range = 0;
    for ( int i = 0; i < EssBdrTrueDofs_range.Size(); ++i )
    {
        int tdof = EssBdrTrueDofs_range[i];
        btd_flags_range[tdof] = 1;
    }

    //if (verbose)
        //std::cout << "EssBdrTrueDofs_range \n";
    //EssBdrTrueDofs_range.Print();

    for (int row = 0; row < C_td_btd_diag.Height(); ++row)
    {
        if (btd_flags_range[row] == 0)
        {
            for (int colind = 0; colind < C_td_btd_diag.RowSize(row); ++colind)
            {
                int nnz_ind = C_td_btd_diag.GetI()[row] + colind;
                int col = C_td_btd_diag.GetJ()[nnz_ind];
                double fabs_entry = fabs(C_td_btd_diag.GetData()[nnz_ind]);

                if (fabs_entry > 1.0e-14)
                {
                    for (int j = 0; j < C_diag.RowSize(row); ++j)
                    {
                        int colorig = C_diag.GetJ()[C_diag.GetI()[row] + j];
                        if (colorig == col)
                        {
                            //std::cout << "Changes made in row = " << row << ", col = " << colorig << "\n";
                            C_diag.GetData()[C_diag.GetI()[row] + j] = 0.0;

                        }
                    }
                } // else of if fabs_entry is large enough

            }
        } // end of if row corresponds to the non-boundary range dof
    }

    //C_diag.Print();

    // processing the off-diagonal block of the Divfree matrix
    SparseMatrix C_td_btd_offd;
    HYPRE_Int * C_td_btd_cmap;
    C_td_btd->GetOffd(C_td_btd_offd, C_td_btd_cmap);

    SparseMatrix C_offd;
    HYPRE_Int * C_cmap;
    Op_hpmat.GetOffd(C_offd, C_cmap);

    //int * row_starts = Op_hpmat.GetRowStarts();

    for (int row = 0; row < C_td_btd_offd.Height(); ++row)
    {
        if (btd_flags_range[row] == 0)
        {
            for (int colind = 0; colind < C_td_btd_offd.RowSize(row); ++colind)
            {
                int nnz_ind = C_td_btd_offd.GetI()[row] + colind;
                int truecol = C_td_btd_cmap[C_td_btd_offd.GetJ()[nnz_ind]];
                double fabs_entry = fabs(C_td_btd_offd.GetData()[nnz_ind]);

                if (fabs_entry > 1.0e-14)
                {
                    for (int j = 0; j < C_offd.RowSize(row); ++j)
                    {
                        int col = C_offd.GetJ()[C_offd.GetI()[row] + j];
                        int truecolorig = C_cmap[col];
                        /*
                        int tdof_for_truecolorig;
                        if (truecolorig < row_starts[0])
                            tdof_for_truecolorig = truecolorig;
                        else
                            tdof_for_truecolorig = truecolorig - row_starts[1];
                        */
                        if (truecolorig == truecol)
                        {
                            //std::cout << "Changes made in off-d: row = " << row << ", col = " << col << ", truecol = " << truecolorig << "\n";
                            C_offd.GetData()[C_offd.GetI()[row] + j] = 0.0;

                        }
                    }
                } // else of if fabs_entry is large enough

            }
        } // end of if row corresponds to the non-boundary range dof

    }
}


// Replaces "bb" block in the Operator acting in the same space,
// assembled as a HypreParMatrix, which connects boundary dofs to boundary dofs by identity
void Eliminate_bb_block(HypreParMatrix& Op_hpmat, const Array<int>& EssBdrTrueDofs )
{
    MFEM_ASSERT(Op_hpmat.Width() == Op_hpmat.Height(), "The matrix must be square in Eliminate_bb_block()! \n");

    int ntdofs = Op_hpmat.Width();

    Array<int> btd_flags(ntdofs);
    btd_flags = 0;
    //if (verbose)
        //std::cout << "EssBdrTrueDofs \n";
    //EssBdrTrueDofs.Print();

    for ( int i = 0; i < EssBdrTrueDofs.Size(); ++i )
    {
        int tdof = EssBdrTrueDofs[i];
        btd_flags[tdof] = 1;
    }

    SparseMatrix C_diag;
    Op_hpmat.GetDiag(C_diag);

    // processing local-to-process block of the matrix
    for (int row = 0; row < C_diag.Height(); ++row)
    {
        if (btd_flags[row] != 0) // if the row tdof is at the boundary
        {
            for (int j = 0; j < C_diag.RowSize(row); ++j)
            {
                int col = C_diag.GetJ()[C_diag.GetI()[row] + j];
                if  (col == row)
                    C_diag.GetData()[C_diag.GetI()[row] + j] = 1.0;
                else
                    C_diag.GetData()[C_diag.GetI()[row] + j] = 0.0;
            }
        } // end of if row corresponds to the boundary tdof
    }

    //C_diag.Print();

    SparseMatrix C_offd;
    HYPRE_Int * C_cmap;
    Op_hpmat.GetOffd(C_offd, C_cmap);

    // processing the off-diagonal block of the matrix
    for (int row = 0; row < C_offd.Height(); ++row)
    {
        if (btd_flags[row] != 0) // if the row tdof is at the boundary
        {
            for (int j = 0; j < C_offd.RowSize(row); ++j)
            {
                C_offd.GetData()[C_offd.GetI()[row] + j] = 0.0;
            }

        } // end of if row corresponds to the boundary tdof

    }
}

template<typename T> void ConvertSTDvecToArray(std::vector<T>& stdvector, Array<int>& array_)
{
    array_.SetSize((int) (stdvector.size()));
    for (int i = 0; i < array_.Size(); ++i)
        array_[i] = stdvector[i];
}


/*
// self-written copy routine for HypreParMatrices
// faces the issues with LeftDiagMult and ParMult combination
// My guess is that offd.num_rownnz != 0 is the bug
// but no proof for now
HypreParMatrix * CopyHypreParMatrix (HypreParMatrix& inputmat)
{
    MPI_Comm comm = inputmat.GetComm();
    int num_procs;
    MPI_Comm_size(comm, &num_procs);

    HYPRE_Int global_num_rows = inputmat.M();
    HYPRE_Int global_num_cols = inputmat.N();

    int size_starts = num_procs;
    if (num_procs > 1) // in thi case offd exists
    {
        //int myid;
        //MPI_Comm_rank(comm,&myid);

        HYPRE_Int * row_starts_in = inputmat.GetRowStarts();
        HYPRE_Int * col_starts_in = inputmat.GetColStarts();

        HYPRE_Int * row_starts = new HYPRE_Int[num_procs];
        memcpy(row_starts, row_starts_in, size_starts * sizeof(HYPRE_Int));
        HYPRE_Int * col_starts = new HYPRE_Int[num_procs];
        memcpy(col_starts, col_starts_in, size_starts * sizeof(HYPRE_Int));

        //std::cout << "memcpy calls finished \n";

        SparseMatrix diag_in;
        inputmat.GetDiag(diag_in);
        SparseMatrix * diag_out = new SparseMatrix(diag_in);

        //std::cout << "diag copied \n";

        SparseMatrix offdiag_in;
        HYPRE_Int * offdiag_cmap_in;
        inputmat.GetOffd(offdiag_in, offdiag_cmap_in);

        int size_offdiag_cmap = offdiag_in.Width();

        SparseMatrix * offdiag_out = new SparseMatrix(offdiag_in);
        HYPRE_Int * offdiag_cmap_out = new HYPRE_Int[size_offdiag_cmap];

        memcpy(offdiag_cmap_out, offdiag_cmap_in, size_offdiag_cmap * sizeof(int));


        return new HypreParMatrix(comm, global_num_rows, global_num_cols,
                                  row_starts, col_starts,
                                  diag_out, offdiag_out, offdiag_cmap_out);

        //std::cout << "constructor called \n";
    }
    else // in this case offd doesn't exist and we have to use a different constructor
    {
        HYPRE_Int * row_starts = new HYPRE_Int[2];
        row_starts[0] = 0;
        row_starts[1] = global_num_rows;
        HYPRE_Int * col_starts = new HYPRE_Int[2];
        col_starts[0] = 0;
        col_starts[1] = global_num_cols;

        SparseMatrix diag_in;
        inputmat.GetDiag(diag_in);
        SparseMatrix * diag_out = new SparseMatrix(diag_in);

        return new HypreParMatrix(comm, global_num_rows, global_num_cols,
                                  row_starts, col_starts, diag_out);
    }

}

// faces the same issues as CopyHypreParMatrix
HypreParMatrix * CopyRAPHypreParMatrix (HypreParMatrix& inputmat)
{
    MPI_Comm comm = inputmat.GetComm();
    int num_procs;
    MPI_Comm_size(comm, &num_procs);

    HYPRE_Int global_num_rows = inputmat.M();
    HYPRE_Int global_num_cols = inputmat.N();

    int size_starts = 2;

    HYPRE_Int * row_starts_in = inputmat.GetRowStarts();
    HYPRE_Int * col_starts_in = inputmat.GetColStarts();

    HYPRE_Int * row_starts = new HYPRE_Int[num_procs];
    memcpy(row_starts, row_starts_in, size_starts * sizeof(HYPRE_Int));
    HYPRE_Int * col_starts = new HYPRE_Int[num_procs];
    memcpy(col_starts, col_starts_in, size_starts * sizeof(HYPRE_Int));

    int num_local_rows = row_starts[1] - row_starts[0];
    int num_local_cols = col_starts[1] - col_starts[0];
    int * ia_id = new int[num_local_rows + 1];
    ia_id[0] = 0;
    for ( int i = 0; i < num_local_rows; ++i)
        ia_id[i + 1] = ia_id[i] + 1;

    int id_nnz = num_local_rows;
    int * ja_id = new int[id_nnz];
    double * a_id = new double[id_nnz];
    for ( int i = 0; i < id_nnz; ++i)
    {
        ja_id[i] = i;
        a_id[i] = 1.0;
    }

    SparseMatrix * id_diag = new SparseMatrix(ia_id, ja_id, a_id, num_local_rows, num_local_cols);

    HypreParMatrix * id = new HypreParMatrix(comm, global_num_rows, global_num_cols,
                                             row_starts, col_starts, id_diag);

    return RAP(&inputmat,id);
}
*/

/// simple copy by using Transpose (and temporarily allocating
/// additional memory of size = size of the inpt matrix)
HypreParMatrix * CopyHypreParMatrix(const HypreParMatrix& divfree_dop)
{
    HypreParMatrix * temp = divfree_dop.Transpose();
    temp->CopyColStarts();
    temp->CopyRowStarts();

    HypreParMatrix * res = temp->Transpose();
    res->CopyColStarts();
    res->CopyRowStarts();

    delete temp;

    return res;
}

void EliminateBoundaryBlocks(BlockOperator& BlockOp, const std::vector<Array<int>* > esstdofs_blks)
{
    int nblocks = BlockOp.NumRowBlocks();

    for (int i = 0; i < nblocks; ++i)
        for (int j = 0; j < nblocks; ++j)
        {
            const Array<int> *temp_range = esstdofs_blks[i];
            const Array<int> *temp_dom = esstdofs_blks[j];

            HypreParMatrix * op_blk = dynamic_cast<HypreParMatrix*>(&(BlockOp.GetBlock(i,j)));

#if 0
            /*
            if (i == j && i == 1)
            {
                std::cout << "op_blk size = " << op_blk->Height() << "\n";

                SparseMatrix diag;
                op_blk->GetDiag(diag);
                std::cout << "diag of 11 block in EliminateBoundaryBlocks = " << diag.MaxNorm() << "\n";

                //diag.Print();

                temp_dom->Print();
            }
            */

            if (i == 1 && i == j)
            {
                Eliminate_ib_block(*op_blk, *temp_dom, *temp_range );

                {
                    std::cout << "op_blk size = " << op_blk->Height() << "\n";

                    temp_dom->Print();

                    //SparseMatrix diag;
                    //op_blk->GetDiag(diag);

                    //std::cout << "diag in op_blk, elim bnd blocks \n";
                    //diag.Print();
                }

                HypreParMatrix * temphpmat = op_blk->Transpose();
                Eliminate_ib_block(*temphpmat, *temp_range, *temp_dom );

                /*
                //if (l == 2)
                {
                    SparseMatrix diag;
                    temphpmat->GetDiag(diag);

                    std::cout << "diag in temphpma, elim bnd blocks \n";
                    diag.Print();
                }
                */

                op_blk = temphpmat->Transpose();

                /*
                {
                    SparseMatrix diag;
                    op_blk->GetDiag(diag);

                    std::cout << "diag in op_blk afterwards, elim bnd blocks \n";
                    diag.Print();

                }
                */

                /*
                if (i == j)
                {
                    Eliminate_bb_block(*op_blk, *temp_dom);
                    SparseMatrix diag;
                    op_blk->GetDiag(diag);
                    diag.MoveDiagonalFirst();
                }

                op_blk->CopyColStarts();
                op_blk->CopyRowStarts();
                delete temphpmat;
                */
            }
#endif

            Eliminate_ib_block(*op_blk, *temp_dom, *temp_range );

            HypreParMatrix * temphpmat = op_blk->Transpose();
            Eliminate_ib_block(*temphpmat, *temp_range, *temp_dom );
            op_blk = temphpmat->Transpose();

            if (i == j)
            {
                //Eliminate_bb_block(*op_blk, *temp_dom);
                SparseMatrix diag;
                op_blk->GetDiag(diag);
                diag.MoveDiagonalFirst();
            }

            op_blk->CopyColStarts();
            op_blk->CopyRowStarts();
            delete temphpmat;

            BlockOp.SetBlock(i,j, op_blk);

        }
}

SparseMatrix& ElementToDofs(const FiniteElementSpace &fes)
{
    // Returns a SparseMatrix with the relation Element to Dofs
    int * I = new int[fes.GetNE() + 1];
    Array<int> vdofs_R;

    I[0] = 0;
    for (int i = 0; i < fes.GetNE(); ++i)
    {
        fes.GetElementVDofs(i, vdofs_R);
        I[i + 1] = I[i] + vdofs_R.Size();
    }
    int * J = new int[I[fes.GetNE()]];
    double * data = new double[I[fes.GetNE()]];

    for (int i = 0; i < fes.GetNE(); ++i)
    {
        // Returns indexes of dofs in array for ith' elements'
        fes.GetElementVDofs(i,vdofs_R);
        fes.AdjustVDofs(vdofs_R);
        for (int j = I[i]; j < I[i + 1]; ++j)
        {
            J[j] = vdofs_R[j - I[i]];
            data[j] =1;
        }

    }
    SparseMatrix * res = new SparseMatrix(I, J, data, fes.GetNE(), fes.GetVSize());
    return *res;
}


} // for namespace mfem
