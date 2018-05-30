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
      have_constraint(do_have_constraint),
      space_names(NULL), space_names_funct(NULL)
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

void CFOSLSFormulation_HdivL2Hyper::ConstructSpacesDescriptor() const
{
    space_names = new Array<SpaceName>(numblocks);

    (*space_names)[0] = SpaceName::HDIV;
    (*space_names)[1] = SpaceName::L2;
}

void CFOSLSFormulation_HdivL2Hyper::ConstructFunctSpacesDescriptor() const
{
    space_names_funct = new Array<SpaceName>(1);
    (*space_names_funct)[0] = SpaceName::HDIV;
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

void CFOSLSFormulation_HdivH1Hyper::ConstructSpacesDescriptor() const
{
    space_names = new Array<SpaceName>(numblocks);

    (*space_names)[0] = SpaceName::HDIV;
    (*space_names)[1] = SpaceName::H1;
    (*space_names)[2] = SpaceName::L2;
}

void CFOSLSFormulation_HdivH1Hyper::ConstructFunctSpacesDescriptor() const
{
    space_names_funct = new Array<SpaceName>(2);

    (*space_names_funct)[0] = SpaceName::HDIV;
    (*space_names_funct)[1] = SpaceName::H1;
}

CFOSLSFormulation_HdivH1DivfreeHyp::CFOSLSFormulation_HdivH1DivfreeHyp (int dimension, int num_solution, bool verbose)
    : FOSLSFormulation(dimension, 2, 2, false), numsol(num_solution), test(dim, numsol)
{
    blfis(0,0) = new CurlCurlIntegrator();
    blfis(1,1) = new H1NormIntegrator(*test.GetBBt(), *test.GetBtB());

    //MFEM_ABORT("Set the correct linear forms, or decide whether it is needed at all \n");
    //lfis[1] = new GradDomainLFIntegrator(*test.GetBf());

    InitBlkStructure();
}

void CFOSLSFormulation_HdivH1DivfreeHyp::InitBlkStructure()
{
    blk_structure[0] = std::make_pair<int,int>(1,0);
    blk_structure[1] = std::make_pair<int,int>(1,-1);
}

void CFOSLSFormulation_HdivH1DivfreeHyp::ConstructSpacesDescriptor() const
{
    space_names = new Array<SpaceName>(numblocks);

    (*space_names)[0] = SpaceName::HCURL;
    (*space_names)[1] = SpaceName::H1;
}


void CFOSLSFormulation_HdivH1DivfreeHyp::ConstructFunctSpacesDescriptor() const
{
    space_names_funct = new Array<SpaceName>(1);
    (*space_names_funct)[0] = SpaceName::HCURL;
}

// FE formulations

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

CFOSLSFEFormulation_HdivH1DivfreeHyper::CFOSLSFEFormulation_HdivH1DivfreeHyper(FOSLSFormulation& formulation, int fe_order)
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
    MFEM_ASSERT(numblocks == fe_formul.Nblocks(),
                "numblocks mismatch in BlockProblemForms::InitForms!");
    MFEM_ASSERT(pfes.Size() == numblocks,
                "size of pfes is different from numblocks in BlockProblemForms::InitForms!");

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
             FOSLSFEFormulation& fe_formulation, bool verbose_, bool assemble_system)
    : pmesh(*Hierarchy.GetPmesh(level)), fe_formul(fe_formulation), bdr_conds(bdr_conditions),
      hierarchy(&Hierarchy), level_in_hierarchy(level),
      spaces_initialized(false), forms_initialized(false), system_assembled(false),
      solver_initialized(false), hierarchy_initialized(true), hpmats_initialized(false),
      pbforms(fe_formul.Nblocks()), prec_option(0), verbose(verbose_)
{
    estimators.SetSize(0);

    InitSpacesFromHierarchy(*hierarchy, level, *fe_formulation.GetFormulation()->GetSpacesDescriptor());
    InitForms();
    InitGrFuns();

    x = NULL;
    trueX = NULL;
    trueRhs = NULL;
    trueBnd = NULL;
    CreateOffsetsRhsSol();

    CFOSLSop = NULL;
    CFOSLSop_nobnd = NULL;

    if (assemble_system)
    {
        AssembleSystem(verbose);
        InitSolver(verbose);
    }
}

FOSLSProblem::FOSLSProblem(ParMesh& pmesh_, BdrConditions& bdr_conditions,
                           FOSLSFEFormulation& fe_formulation, bool verbose_, bool assemble_system)
    : pmesh(pmesh_), fe_formul(fe_formulation), bdr_conds(bdr_conditions),
      hierarchy(NULL), level_in_hierarchy(-1),
      spaces_initialized(false), forms_initialized(false), system_assembled(false),
      solver_initialized(false), hierarchy_initialized(true), hpmats_initialized(false),
      pbforms(fe_formul.Nblocks()), prec_option(0), verbose(verbose_)
{
    estimators.SetSize(0);

    InitSpaces(pmesh);
    InitForms();
    InitGrFuns();

    x = NULL;
    trueX = NULL;
    trueRhs = NULL;
    trueBnd = NULL;
    CreateOffsetsRhsSol();

    CFOSLSop = NULL;
    CFOSLSop_nobnd = NULL;

    if (assemble_system)
    {
        AssembleSystem(verbose);
        InitSolver(verbose);
    }
}

void FOSLSProblem::Update()
{
    for (int i = 0; i < pfes.Size(); ++i)
    {
        pfes[i]->Update();
        // FIXME: Is it necessary?
        pfes[i]->Dof_TrueDof_Matrix();
    }

    for (int i = 0; i < grfuns.Size(); ++i)
        grfuns[i]->Update();

    pbforms.Update();

    for (int i = 0; i < plforms.Size(); ++i)
        plforms[i]->Update();

    // If it's a basic FOSLSEstimator constructed without any
    // extra grfuns, this call does nothing since all the grfuns
    // are already updated several lines above
    for (int i = 0; i < estimators.Size(); ++i)
        estimators[i]->Update();

    if (trueRhs)
        delete trueRhs;
    trueRhs = NULL;
    if (trueX)
        delete trueX;
    trueX = NULL;
    if (trueBnd)
        delete trueBnd;
    trueBnd = NULL;

    if (x)
        delete x;
    x = NULL;

    delete solver;

    if (prec)
        delete prec;

    if (hpmats_initialized)
        for (int i = 0; i < hpmats.NumRows(); ++i)
            for (int j = 0; j < hpmats.NumCols(); ++j)
                if (hpmats(i,j))
                    delete hpmats(i,j);

    if (hpmats_initialized)
        for (int i = 0; i < hpmats_nobnd.NumRows(); ++i)
            for (int j = 0; j < hpmats_nobnd.NumCols(); ++j)
                if (hpmats_nobnd(i,j))
                    delete hpmats_nobnd(i,j);

    if  (CFOSLSop)
        delete CFOSLSop;
    CFOSLSop = NULL;

    if (CFOSLSop_nobnd)
        delete CFOSLSop_nobnd;
    CFOSLSop_nobnd = NULL;

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

    forms_initialized = true;
}

void FOSLSProblem::InitSpacesFromHierarchy(GeneralHierarchy& hierarchy, int level, const Array<SpaceName> &spaces_descriptor)
{
    pfes.SetSize(fe_formul.Nblocks());

    for (int i = 0; i < fe_formul.Nblocks(); ++i)
    {
        pfes[i] = hierarchy.GetSpace(spaces_descriptor[i], level);
    }

    spaces_initialized = true;
}


void FOSLSProblem::InitSpaces(ParMesh &pmesh)
{
    pfes.SetSize(fe_formul.Nblocks());

    for (int i = 0; i < fe_formul.Nblocks(); ++i)
        pfes[i] = new ParFiniteElementSpace(&pmesh, fe_formul.GetFeColl(i));

    spaces_initialized = true;
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

    solver_initialized = true;
}

BlockVector * FOSLSProblem::GetTrueInitialCondition()
{
    // alias
    FOSLS_test * test = fe_formul.GetFormulation()->GetTest();

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

            const Array<int>& essbdr_attrs = bdr_conds.GetBdrAttribs(blk);

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

BlockVector * FOSLSProblem::GetTrueInitialConditionFunc()
{
    // alias
    FOSLS_test * test = fe_formul.GetFormulation()->GetTest();

    BlockVector * truebnd = new BlockVector(blkoffsets_func_true);
    *truebnd = 0.0;

    for (int blk = 0; blk < fe_formul.Nunknowns(); ++blk)
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

            const Array<int>& essbdr_attrs = bdr_conds.GetBdrAttribs(blk);

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
    MFEM_ASSERT(spaces_initialized && forms_initialized,
                "Cannot build system if spaces or forms were not initialized");

    CreateOffsetsRhsSol();

    AssembleSystem(verbose);

    InitSolver(verbose);

    CreatePrec(*CFOSLSop, prec_option, verbose);
    UpdateSolverPrec();
}

// works correctly only for problems with homogeneous initial conditions?
// see the times-stepping branch, think of how boundary conditions for off-diagonal blocks are imposed
// system is assumed to be symmetric
void FOSLSProblem::AssembleSystem(bool verbose)
{
    int numblocks = fe_formul.Nblocks();

    if (x)
        delete x;
    x = GetInitialCondition();

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

                    const Array<int>& essbdr_attrs = bdr_conds.GetBdrAttribs(i);

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

                    const Array<int>& essbdr_attrs = bdr_conds.GetBdrAttribs(exist_col);

                    Vector dummy(pbforms.offd(exist_row,exist_col)->Height());
                    dummy = 0.0;
                    //pbforms.offd(exist_row,exist_col)->EliminateTrialDofs(*struct_formul.essbdr_attrs[exist_col],

                    pbforms.offd(exist_row,exist_col)->EliminateTrialDofs(essbdr_attrs, x->GetBlock(exist_col), dummy);

                    const Array<int>& essbdr_attrs2 = bdr_conds.GetBdrAttribs(exist_row);

                    //pbforms.offd(exist_row,exist_col)->EliminateTestDofs(*struct_formul.essbdr_attrs[exist_row]);
                    pbforms.offd(exist_row,exist_col)->EliminateTestDofs(essbdr_attrs2);


                    pbforms.offd(exist_row,exist_col)->Finalize();
                    hpmats(exist_row,exist_col) = pbforms.offd(exist_row,exist_col)->ParallelAssemble();
                    hpmats(exist_col, exist_row) = hpmats(exist_row,exist_col)->Transpose();
                }
            }
        }

   hpmats_initialized = true;

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

   if (trueBnd)
       delete trueBnd;

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
       const Array<int>& essbdr_attrs = bdr_conds.GetBdrAttribs(i);

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
   //MPI_Comm comm = pfes[0]->GetComm();
   //MPI_Barrier(comm);

   system_assembled = true;
}

void FOSLSProblem::DistributeSolution() const
{
    for (int i = 0; i < fe_formul.Nblocks(); ++i)
        grfuns[i]->Distribute(&(trueX->GetBlock(i)));
}

void FOSLSProblem::DistributeToGrfuns(const Vector& vec) const
{
    const BlockVector vec_viewer(vec.GetData(), blkoffsets_true);
    for (int i = 0; i < fe_formul.Nblocks(); ++i)
        grfuns[i]->Distribute(&(vec_viewer.GetBlock(i)));
}

void FOSLSProblem::ComputeBndError(const Vector& vec, int blk) const
{
    const BlockVector vec_viewer(vec.GetData(), blkoffsets_true);

    // alias
    FOSLS_test * test = fe_formul.GetFormulation()->GetTest();

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

    const Array<int>& essbdr_attrs = bdr_conds.GetBdrAttribs(blk);

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

        const Array<int>& essbdr_attrs = bdr_conds.GetBdrAttribs(blk);

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

void FOSLSProblem::ComputeError(const Vector& vec, bool verbose, bool checkbnd, int blk) const
{
    // alias
    FOSLS_test * test = fe_formul.GetFormulation()->GetTest();

    const BlockVector vec_viewer(vec.GetData(), blkoffsets_true);

    grfuns[blk]->Distribute(&(vec_viewer.GetBlock(blk)));

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
        ComputeBndError(vec, blk);

    if (verbose)
        std::cout << "component No. " << blk << ": || exact - proj || / || exact || = "
                        << projection_error / norm_exsol << "\n";

    delete exsol_pgfun;
}

void FOSLSProblem::CreateOffsetsRhsSol()
{
    int numblocks = fe_formul.Nblocks();
    int numunknowns = fe_formul.Nunknowns();

    blkoffsets_true.SetSize(numblocks + 1);
    blkoffsets_true[0] = 0;
    for (int i = 0; i < numblocks; ++i)
        blkoffsets_true[i + 1] = pfes[i]->TrueVSize();
    blkoffsets_true.PartialSum();

    blkoffsets_func_true.SetSize(numunknowns + 1);
    blkoffsets_func_true[0] = 0;
    for (int i = 0; i < numunknowns; ++i)
        blkoffsets_func_true[i + 1] = pfes[i]->TrueVSize();
    blkoffsets_func_true.PartialSum();

    blkoffsets.SetSize(numblocks + 1);
    blkoffsets[0] = 0;
    for (int i = 0; i < numblocks; ++i)
        blkoffsets[i + 1] = pfes[i]->GetVSize();
    blkoffsets.PartialSum();

    if (trueRhs)
        delete trueRhs;

    if (trueX)
        delete trueX;

    trueRhs = new BlockVector(blkoffsets_true);
    trueX = new BlockVector(blkoffsets_true);
}


void FOSLSProblem::ZeroBndValues(Vector& vec) const
{
    BlockVector vec_viewer(vec.GetData(), blkoffsets_true);

    int numblocks = fe_formul.Nblocks();
    for (int i = 0; i < numblocks; ++i)
    {
        const Array<int>& essbdr_attrs = bdr_conds.GetBdrAttribs(i);

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

BlockMatrix* FOSLSProblem::ConstructFunctBlkMat(const Array<int>& offsets)
{
    int num_unknowns = fe_formul.GetFormulation()->Nunknowns();

    Array2D<SparseMatrix*> funct_blocks(num_unknowns, num_unknowns);

    for (int i = 0; i < num_unknowns; ++i)
        for (int j = 0; j < num_unknowns; ++j)
        {
            funct_blocks(i,j) = NULL;
            if (i == j)
            {
                if (pbforms.diag(i))
                {
                    //pbforms.diag(i)->Update();
                    //pbforms.diag(i)->Assemble();
                    //pbforms.diag(i)->Finalize();

                    funct_blocks(i,j) = &pbforms.diag(i)->SpMat();
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

                    //pbforms.offd(exist_row,exist_col)->Update();
                    //pbforms.offd(exist_row,exist_col)->Assemble();
                    //pbforms.offd(exist_row,exist_col)->Finalize();

                    //funct_blocks(exist_row, exist_col) = pbforms.offd(exist_row,exist_col)->LoseMat();
                    funct_blocks(exist_row, exist_col) = &pbforms.offd(exist_row,exist_col)->SpMat();
                    funct_blocks(exist_col, exist_row) = Transpose(*funct_blocks(exist_row, exist_col));
                }
            }
        }

    /*
    if (!offsets)
    {
        offsets = new Array<int>*[1];
        *offsets = new Array<int>();
        (*offsets)->SetSize(num_unknowns + 1);

        (*(*offsets))[0] = 0;
        for (int i = 0; i < num_unknowns; ++i)
            (*(*offsets))[i + 1] = funct_blocks(i,i)->Height();
        (*offsets)->PartialSum();
    }
    */

    BlockMatrix * res = new BlockMatrix(offsets);

    for (int i = 0; i < num_unknowns; ++i)
        for (int j = 0; j < num_unknowns; ++j)
            res->SetBlock(i,j, funct_blocks(i,j));

    res->owns_blocks = true;

    return res;
}

BlockOperator* FOSLSProblem::GetFunctOp(const Array<int> &offsets)
{
    BlockOperator * funct_op = new BlockOperator(offsets);

    int numblocks = offsets.Size() - 1;

    for (int i = 0; i < numblocks; ++i)
        for (int j = 0; j < numblocks; ++j)
            if (!CFOSLSop->IsZeroBlock(i,j))
                funct_op->SetBlock(i,j, (HypreParMatrix*)(&CFOSLSop->GetBlock(i,j)));

    funct_op->owns_blocks = false;
    return funct_op;
}


void FOSLSProblem::SolveProblem(const Vector& rhs, Vector& sol, bool verbose, bool compute_error) const
{
    MFEM_ASSERT(solver_initialized, "Solver is not initialized \n");

    chrono.Clear();
    chrono.Start();

    solver->Mult(rhs, sol);

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
}

void FOSLSProblem::SolveProblem(const Vector& rhs, bool verbose, bool compute_error) const
{
    *trueX = 0.0;
    SolveProblem(rhs, *trueX, verbose, compute_error);

    DistributeSolution();

    bool checkbnd = false;
    if (compute_error)
        ComputeError(verbose, checkbnd);
}

void FOSLSProblem_HdivL2L2hyp::ComputeExtraError(const Vector& vec) const
{
    BlockVector vec_viewer(vec.GetData(), blkoffsets_true);

    Hyper_test * test = dynamic_cast<Hyper_test*>(fe_formul.GetFormulation()->GetTest());
    MFEM_ASSERT(test, "Unsuccessful cast into Hyper_test*");

    // aliases
    ParFiniteElementSpace * Hdiv_space = pfes[0];
    ParFiniteElementSpace * L2_space = pfes[1];
    ParGridFunction sigma(Hdiv_space);
    sigma.Distribute(&vec_viewer.GetBlock(0));

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
    Div.Mult(sigma, DivSigma);

    double err_div = DivSigma.ComputeL2Error(*test->GetRhs(),irs);
    double norm_div = ComputeGlobalLpNorm(2, *test->GetRhs(), pmesh, irs);

    if (verbose)
    {
        std::cout << "|| div (sigma_h - sigma_ex) || / ||div (sigma_ex)|| = "
                  << err_div / norm_div  << "\n";
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
    B->Mult(vec_viewer.GetBlock(0),bTsigma);

    Vector trueS(C->Height());

    CG(*C, bTsigma, trueS, 0, 5000, 1e-9, 1e-12);

    ParGridFunction S(L2_space);
    S.Distribute(trueS);

    delete Cblock;
    delete Bblock;
    delete B;
    delete C;

    double err_S = S.ComputeL2Error(*test->GetU(), irs);
    double norm_S = ComputeGlobalLpNorm(2, *test->GetU(), pmesh, irs);
    if (verbose)
    {
        std::cout << "|| S_h - S_ex || / || S_ex || = " <<
                     err_S / norm_S << "\n";
    }

    ParGridFunction S_exact(L2_space);
    S_exact.ProjectCoefficient(*test->GetU());

    double projection_error_S = S_exact.ComputeL2Error(*test->GetU(), irs);

    if (verbose)
        std::cout << "|| S_ex - Pi_h S_ex || / || S_ex || = "
                        << projection_error_S / norm_S << "\n";

    ComputeFuncError(vec);
}

ParGridFunction * FOSLSProblem_HdivL2L2hyp::RecoverS()
{
    Hyper_test * test = dynamic_cast<Hyper_test*>(fe_formul.GetFormulation()->GetTest());
    MFEM_ASSERT(test, "Unsuccessful cast into Hyper_test*");

    // aliases
    ParFiniteElementSpace * Hdiv_space = pfes[0];
    ParFiniteElementSpace * L2_space = pfes[1];

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

    return S;
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
        if (prec_option == 100)
            std::cout << "Using cheaper Gauss-Seidel smoothers for all blocks! \n";

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

    Solver * invA, *invS;
    if (prec_option == 100)
    {
        invA = new HypreSmoother(A, HypreSmoother::Type::l1GS, 1);
        invS = new HypreSmoother(*Schur, HypreSmoother::Type::l1GS, 1);
    }
    else // standard case
    {
        if (prec_option == 2)
            invA = new HypreADS(A, pfes[0]);
        else // using Diag(A);
            invA = new HypreDiagScale(A);

        invA->iterative_mode = false;

        invS = new HypreBoomerAMG(*Schur);
        ((HypreBoomerAMG *)invS)->SetPrintLevel(0);
        ((HypreBoomerAMG *)invS)->iterative_mode = false;
    }

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

// computes || sigma - L(S) ||
void FOSLSProblem_HdivL2L2hyp::ComputeFuncError(const Vector& vec) const
{
    Hyper_test * test = dynamic_cast<Hyper_test*>(fe_formul.GetFormulation()->GetTest());

    BlockVector vec_viewer(vec.GetData(), blkoffsets_true);

    ParFiniteElementSpace * Hdiv_space = pfes[0];
    ParFiniteElementSpace * L2_space = pfes[1];

    ParGridFunction sigma(Hdiv_space);
    sigma.Distribute(&vec_viewer.GetBlock(0));

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
    Div.Mult(sigma, DivSigma);

    double err_div = DivSigma.ComputeL2Error(*test->GetRhs(),irs);
    double norm_div = ComputeGlobalLpNorm(2, *test->GetRhs(), pmesh, irs);

    Vector trueSigma(Hdiv_space->TrueVSize());
    trueSigma = vec_viewer.GetBlock(0);

    Vector MtrueSigma(Hdiv_space->TrueVSize());
    MtrueSigma = 0.0;

    HypreParMatrix * M = (HypreParMatrix*)(&CFOSLSop->GetBlock(0,0));
    M->Mult(trueSigma, MtrueSigma);
    double localFunctional = trueSigma * MtrueSigma;

    double globalFunctional;
    MPI_Reduce(&localFunctional, &globalFunctional, 1,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (verbose)
    {
        std::cout << "|| sigma_h - L(S_h) ||^2 = " << globalFunctional << "\n";
        std::cout << "Energy Error = " << sqrt(globalFunctional + err_div * err_div) << "\n";
        std::cout << "Relative Energy Error = " << sqrt(globalFunctional + err_div * err_div)
                     / norm_div << "\n";
    }

    ParLinearForm gform(L2_space);
    gform.AddDomainIntegrator(new DomainLFIntegrator(*fe_formul.
                                                     GetFormulation()->GetTest()->GetRhs()));
    gform.Assemble();

    Vector Rhs(L2_space->TrueVSize());
    Rhs = *gform.ParallelAssemble();

    double mass_loc = Rhs.Norml1();
    double mass;
    MPI_Reduce(&mass_loc, &mass, 1,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (verbose)
        cout << "Sum of local mass = " << mass << "\n";

    Vector DtrueSigma(L2_space->TrueVSize());
    DtrueSigma = 0.0;
    HypreParMatrix * Bdiv = (HypreParMatrix*)(&CFOSLSop->GetBlock(1,0));
    Bdiv->Mult(trueSigma, DtrueSigma);
    DtrueSigma -= Rhs;
    double mass_loss_loc = DtrueSigma.Norml1();
    double mass_loss;
    MPI_Reduce(&mass_loss_loc, &mass_loss, 1,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (verbose)
        std::cout << "Sum of local mass loss = " << mass_loss << "\n";

}

// computes || sigma - L(S) || only, no term with || div bS - f
void FOSLSProblem_HdivH1L2hyp::ComputeFuncError(const Vector& vec) const
{
    BlockVector vec_viewer(vec.GetData(), blkoffsets_true);

    ParFiniteElementSpace * Hdiv_space = pfes[0];
    ParFiniteElementSpace * H1_space = pfes[1];
    ParFiniteElementSpace * L2_space = pfes[2];

    Vector trueSigma(Hdiv_space->TrueVSize());
    trueSigma = vec_viewer.GetBlock(0);

    Vector MtrueSigma(Hdiv_space->TrueVSize());
    MtrueSigma = 0.0;

    HypreParMatrix * M = (HypreParMatrix*)(&CFOSLSop->GetBlock(0,0));
    M->Mult(trueSigma, MtrueSigma);
    double localFunctional = trueSigma * MtrueSigma;

    Vector GtrueSigma(H1_space->TrueVSize());
    GtrueSigma = 0.0;

    HypreParMatrix * BT = (HypreParMatrix*)(&CFOSLSop->GetBlock(1,0));
    BT->Mult(trueSigma, GtrueSigma);
    localFunctional += 2.0 * (vec_viewer.GetBlock(1)*GtrueSigma);

    Vector XtrueS(H1_space->TrueVSize());
    XtrueS = 0.0;
    HypreParMatrix * C = (HypreParMatrix*)(&CFOSLSop->GetBlock(1,1));
    C->Mult(vec_viewer.GetBlock(1), XtrueS);
    localFunctional += vec_viewer.GetBlock(1)*XtrueS;

    double globalFunctional;
    MPI_Reduce(&localFunctional, &globalFunctional, 1,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (verbose)
    {
        std::cout << "|| sigma_h - L(S_h) ||^2 = " << globalFunctional << "\n";
    }

    ParLinearForm gform(L2_space);
    gform.AddDomainIntegrator(new DomainLFIntegrator(*fe_formul.
                                                     GetFormulation()->GetTest()->GetRhs()));
    gform.Assemble();

    Vector Rhs(L2_space->TrueVSize());
    Rhs = *gform.ParallelAssemble();

    double mass_loc = Rhs.Norml1();
    double mass;
    MPI_Reduce(&mass_loc, &mass, 1,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (verbose)
        cout << "Sum of local mass = " << mass << "\n";

    Vector DtrueSigma(L2_space->TrueVSize());
    DtrueSigma = 0.0;
    HypreParMatrix * Bdiv = (HypreParMatrix*)(&CFOSLSop->GetBlock(2,0));
    Bdiv->Mult(trueSigma, DtrueSigma);
    DtrueSigma -= Rhs;
    double mass_loss_loc = DtrueSigma.Norml1();
    double mass_loss;
    MPI_Reduce(&mass_loss_loc, &mass_loss, 1,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (verbose)
        std::cout << "Sum of local mass loss = " << mass_loss << "\n";

}

void FOSLSProblem_HdivH1L2hyp::ComputeExtraError(const Vector& vec) const
{
    BlockVector vec_viewer(vec.GetData(), blkoffsets_true);

    Hyper_test * test = dynamic_cast<Hyper_test*>(fe_formul.GetFormulation()->GetTest());
    MFEM_ASSERT(test, "Unsuccessful cast into Hyper_test*");

    // aliases
    ParFiniteElementSpace * Hdiv_space = pfes[0];
    ParFiniteElementSpace * L2_space = pfes[2];
    ParGridFunction sigma(Hdiv_space);
    sigma.Distribute(&(vec_viewer.GetBlock(0)));

    //std::cout << "vec norm = " << vec.Norml2() / sqrt (vec.Size()) << "\n";
    //sigma->Print();

    //std::cout << "sigma norm = " << sigma.Norml2() / sqrt (sigma.Size()) << "\n";

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
    Div.Mult(sigma, DivSigma);

    //std::cout << "DivSigma norm = " << DivSigma.Norml2() / sqrt (DivSigma.Size()) << "\n";

    double err_div = DivSigma.ComputeL2Error(*test->GetRhs(),irs);
    double norm_div = ComputeGlobalLpNorm(2, *test->GetRhs(), pmesh, irs);

    if (verbose)
    {
        //std::cout << "err_div = " << err_div << ", norm_div = " << norm_div << "\n";
        cout << "|| div (sigma_h - sigma_ex) || / ||div (sigma_ex)|| = "
                  << err_div / norm_div  << "\n";
    }

    ComputeFuncError(vec);
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
        if (prec_option == 100)
            std::cout << "Using cheaper Gauss-Seidel smoothers for all blocks! \n";
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

    Solver *invA, *invC, *invS;
    if (prec_option == 100)
    {
        invA = new HypreSmoother(A, HypreSmoother::Type::l1GS, 1);
        invC = new HypreSmoother(C, HypreSmoother::Type::l1GS, 1);
        invS = new HypreSmoother(*Schur, HypreSmoother::Type::l1GS, 1);
    }
    else // standard case
    {
        invA = new HypreDiagScale(A);
        invA->iterative_mode = false;

        invC = new HypreBoomerAMG(C);
        ((HypreBoomerAMG*)invC)->SetPrintLevel(0);
        ((HypreBoomerAMG*)invC)->iterative_mode = false;

        invS = new HypreBoomerAMG(*Schur);
        ((HypreBoomerAMG *)invS)->SetPrintLevel(0);
        ((HypreBoomerAMG *)invS)->iterative_mode = false;
    }


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

FOSLSDivfreeProblem::FOSLSDivfreeProblem(GeneralHierarchy& Hierarchy, int level, BdrConditions& bdr_conditions,
             FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose)
    : FOSLSProblem(Hierarchy, level, bdr_conditions, fe_formulation, verbose, false)
{
    int dim = pmesh.Dimension();
    MFEM_ASSERT(dim == 3 || dim == 4, "Divfree problem is implemented only for 3D and 4D");

    hdiv_pfespace = Hierarchy.GetSpace(SpaceName::HDIV, level);
    hdiv_fecoll = NULL;
}

FOSLSDivfreeProblem::FOSLSDivfreeProblem(ParMesh& Pmesh, BdrConditions& bdr_conditions,
                FOSLSFEFormulation& fe_formulation, bool verbose_)
    : FOSLSProblem(Pmesh, bdr_conditions, fe_formulation, verbose_, false)
{
    int dim = Pmesh.Dimension();
    int feorder = fe_formulation.Feorder();
    MFEM_ASSERT(dim == 3 || dim == 4, "Divfree problem is implemented only for 3D and 4D");

    if (dim == 4)
        hdiv_fecoll = new RT0_4DFECollection;
    else
        hdiv_fecoll = new RT_FECollection(feorder, dim);

    hdiv_pfespace = new ParFiniteElementSpace(&pmesh, hdiv_fecoll);
}

FOSLSDivfreeProblem::FOSLSDivfreeProblem(ParMesh& Pmesh, BdrConditions& bdr_conditions,
                FOSLSFEFormulation& fe_formulation, FiniteElementCollection& Hdiv_coll, ParFiniteElementSpace &Hdiv_space, bool verbose_)
    : FOSLSProblem(Pmesh, bdr_conditions, fe_formulation, verbose_, false),
      hdiv_fecoll(&Hdiv_coll), hdiv_pfespace(&Hdiv_space)
{
    int dim = Pmesh.Dimension();
    MFEM_ASSERT(dim == 3 || dim == 4, "Divfree problem is implemented only for 3D and 4D");
}

void FOSLSDivfreeProblem::ConstructDivfreeHpMats()
{
    ParDiscreteLinearOperator * divfree_op = new ParDiscreteLinearOperator(pfes[0], hdiv_pfespace);

    int dim = pmesh.Dimension();

    if (dim == 3)
        divfree_op->AddDomainInterpolator(new CurlInterpolator);
    else
        divfree_op->AddDomainInterpolator(new DivSkewInterpolator);

    divfree_op->Assemble();
    divfree_op->Finalize();
    divfree_hpmat_nobnd = divfree_op->ParallelAssemble();

    HypreParMatrix * temp = divfree_hpmat_nobnd->Transpose();
    divfree_hpmat = temp->Transpose();
    divfree_hpmat->CopyColStarts();
    divfree_hpmat->CopyRowStarts();
    delete temp;

    const Array<int> & essbdr_attribs = bdr_conds.GetBdrAttribs(0);

    Array<int> essbdr_tdofs_Hcurl;
    pfes[0]->GetEssentialTrueDofs(essbdr_attribs, essbdr_tdofs_Hcurl);
    Array<int> essbdr_tdofs_Hdiv;
    hdiv_pfespace->GetEssentialTrueDofs(essbdr_attribs, essbdr_tdofs_Hdiv);

    Eliminate_ib_block(*divfree_hpmat, essbdr_tdofs_Hcurl, essbdr_tdofs_Hdiv);

    delete divfree_op;
}

void FOSLSDivfreeProblem::Update()
{
    FOSLSProblem::Update();

    hdiv_pfespace->Update();

    if (divfree_hpmat)
        delete divfree_hpmat;

    if (divfree_hpmat_nobnd)
        delete divfree_hpmat_nobnd;
}

void FOSLSDivfreeProblem::CreatePrec(BlockOperator & op, int prec_option, bool verbose)
{
    MFEM_ASSERT(prec_option >= 0, "Invalid prec option was provided");

    if (verbose)
    {
        std::cout << "Block diagonal preconditioner: \n";
        std::cout << "BoomerAMG for H(curl) \n";
        if (op.NumRowBlocks() > 1) // case when S is present
            std::cout << "BoomerAMG(C) for H1 (if necessary) \n";
        if (prec_option == 100)
            std::cout << "Using cheaper Gauss-Seidel smoothers for all blocks! \n";
    }

    HypreParMatrix & A = ((HypreParMatrix&)(CFOSLSop->GetBlock(0,0)));
    SparseMatrix A_diag;
    A.GetDiag(A_diag);
    A_diag.MoveDiagonalFirst();

    HypreParMatrix * C;
    if (op.NumRowBlocks() > 1) // case when S is present
    {
        C = (HypreParMatrix*)(&CFOSLSop->GetBlock(1,1));
        SparseMatrix C_diag;
        C->GetDiag(C_diag);
        C_diag.MoveDiagonalFirst();
    }

    Solver * invA, *invC;
    if (prec_option == 100)
    {
        invA = new HypreSmoother(A, HypreSmoother::Type::l1GS, 1);
        if (op.NumRowBlocks() > 1) // case when S is present
            invC = new HypreSmoother(*C, HypreSmoother::Type::l1GS, 1);
    }
    else // standard case
    {
        invA = new HypreBoomerAMG(A);
        ((HypreBoomerAMG*)invA)->SetPrintLevel(0);
        ((HypreBoomerAMG*)invA)->iterative_mode = false;

        if (op.NumRowBlocks() > 1) // case when S is present
        {
            invC = new HypreBoomerAMG(*C);
            ((HypreBoomerAMG*)invC)->SetPrintLevel(0);
            ((HypreBoomerAMG*)invC)->iterative_mode = false;
        }
    }



    prec = new BlockDiagonalPreconditioner(blkoffsets_true);

    ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(0, invA);
    if (op.NumRowBlocks() > 1) // case when S is present
        ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(1, invC);

    /*
    Vector testvec1(invC->Width());
    for (int i = 0; i < testvec1.Size(); ++i)
        testvec1[i] = cos(i * 100);

    std::cout << "testvec1 norm = " << testvec1.Norml2() / sqrt(testvec1.Size())
              << "\n";

    Vector testvec2(invC->Width());

    C->Mult(testvec1, testvec2);
    std::cout << "C * testvec1 norm = " << testvec2.Norml2() / sqrt(testvec2.Size())
              << "\n";

    invC->Mult(testvec1, testvec2);
    std::cout << "invC * testvec1 norm = " << testvec2.Norml2() / sqrt(testvec2.Size())
              << "\n";

    std::cout << "Check \n";

    MFEM_ABORT("Looks like C is singular. "
               "But it should be a weighted diffusion term "
               "The issue appeared because BoomerAMG requires the diagonal element"
               " to be the first in the row. So it disappeared when I added MoveDiagonalFIrst() "
               "for the diagonal of C. A clean-up commit to come. \n");
   */
}



//////////////////////////////////////

GeneralMultigrid::GeneralMultigrid(int Nlevels, const Array<Operator*> &P_lvls_,
                                   const Array<Operator*> &Op_lvls_,
                                   const Operator& CoarseOp_,\
                                   const Array<Operator*> &PreSmoothers_lvls_,
                                   const Array<Operator*> &PostSmoothers_lvls_)
    : Solver(Op_lvls_[0]->Height()), nlevels(Nlevels), P_lvls(P_lvls_),
      Op_lvls(Op_lvls_), CoarseOp(CoarseOp_),
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

        std::cout << "residual before smoothing, new MG, "
                     "norm = " << residual_l.Norml2() / sqrt (residual_l.Size()) << "\n";

        PreSmoother_l->Mult(residual_l, correction_l);

        std::cout << "correction after smoothing, new MG, "
                     "norm = " << correction_l.Norml2() / sqrt (correction_l.Size()) << "\n";

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
    InitForms();
    AssembleSystem(verbose);
    InitPrec(prec_option, verbose);
    InitSolver(verbose);
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

    spaces_initialized = true;
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

   CFOSLSop->owns_blocks = false;

   CFOSLSop_nobnd = new BlockOperator(blkoffsets_true);
   for (int i = 0; i < numblocks; ++i)
       for (int j = 0; j < numblocks; ++j)
           CFOSLSop_nobnd->SetBlock(i,j, hpmats_nobnd(i,j));

   CFOSLSop_nobnd->owns_blocks = false;

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

GeneralHierarchy::GeneralHierarchy(int num_levels, ParMesh& pmesh_, int feorder, bool verbose)
    : num_lvls(num_levels), pmesh(pmesh_),
      divfreedops_constructed (false), doftruedofs_constructed (true),
      pmesh_ne(0), update_counter(0)
{
    pmesh_ne = pmesh.GetNE();
    int dim = pmesh.Dimension();

    if (dim == 4)
        hdiv_coll = new RT0_4DFECollection;
    else
        hdiv_coll = new RT_FECollection(feorder, dim);

    l2_coll = new L2_FECollection(feorder, dim);

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

    if (dim == 4)
        hcurl_coll = new ND1_4DFECollection;
    else
        hcurl_coll = new ND_FECollection(feorder + 1, dim);

    if (dim == 4)
        hdivskew_coll = new DivSkew1_4DFECollection;
    else
        hdivskew_coll = NULL;

    Hdiv_space = new ParFiniteElementSpace(&pmesh, hdiv_coll);

    L2_space = new ParFiniteElementSpace(&pmesh, l2_coll);

    H1_space = new ParFiniteElementSpace(&pmesh, h1_coll);

    Hcurl_space = new ParFiniteElementSpace(&pmesh, hcurl_coll);

    if (dim == 4)
        Hdivskew_space = new ParFiniteElementSpace(&pmesh, hdivskew_coll);
    else
        Hdivskew_space = NULL;

    const SparseMatrix* P_Hdiv_local;
    const SparseMatrix* P_H1_local;
    const SparseMatrix* P_L2_local;
    const SparseMatrix* P_Hcurl_local;
    const SparseMatrix* P_Hdivskew_local;

    pmesh_lvls.SetSize(num_lvls);
    Hdiv_space_lvls.SetSize(num_lvls);
    H1_space_lvls.SetSize(num_lvls);
    L2_space_lvls.SetSize(num_lvls);
    Hcurl_space_lvls.SetSize(num_lvls);
    if (dim == 4)
        Hdivskew_space_lvls.SetSize(num_lvls);
    P_Hdiv_lvls.SetSize(num_lvls - 1);
    P_H1_lvls.SetSize(num_lvls - 1);
    P_L2_lvls.SetSize(num_lvls - 1);
    P_Hcurl_lvls.SetSize(num_lvls - 1);
    if (dim == 4)
        P_Hdivskew_lvls.SetSize(num_lvls - 1);
    TrueP_Hdiv_lvls.SetSize(num_lvls - 1);
    TrueP_H1_lvls.SetSize(num_lvls - 1);
    TrueP_L2_lvls.SetSize(num_lvls - 1);
    TrueP_Hcurl_lvls.SetSize(num_lvls - 1);
    if (dim == 4)
        TrueP_Hdivskew_lvls.SetSize(num_lvls - 1);

    //std::cout << "Checking test for dynamic cast \n";
    //if (dynamic_cast<testB*> (testA))
        //std::cout << "Unsuccessful cast \n";

    for (int l = num_lvls - 1; l >= 0; --l)
    {
        RefineAndCopy(l, &pmesh);
        pmesh_ne = pmesh.GetNE();

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

void GeneralHierarchy::Update()
{
    // TODO: Instead of checking the mesh number of elements to define
    // TODO: whether the hierarchy is to be updated one can use the same
    // TODO: trick as in FOSLSEstimator method MeshIsModified
    bool update_required = (pmesh.GetNE() != pmesh_ne);
    if (update_required)
    {
        MFEM_ASSERT(pmesh.GetLastOperation() == Mesh::Operation::REFINE, "It is assumed that a refinement was done \n");

        // updating mesh
        ParMesh * pmesh_new = new ParMesh(pmesh);
        pmesh_lvls.Prepend(pmesh_new);

        int dim = pmesh.Dimension();

        ParFiniteElementSpace * Hdiv_space_new = new ParFiniteElementSpace(pmesh_lvls[0], hdiv_coll);
        ParFiniteElementSpace * L2_space_new = new ParFiniteElementSpace(pmesh_lvls[0], l2_coll);
        ParFiniteElementSpace * H1_space_new = new ParFiniteElementSpace(pmesh_lvls[0], h1_coll);
        ParFiniteElementSpace * Hcurl_space_new = new ParFiniteElementSpace(pmesh_lvls[0], hcurl_coll);
        ParFiniteElementSpace * Hdivskew_space_new;
        if (dim == 4)
            Hdivskew_space_new = new ParFiniteElementSpace(pmesh_lvls[0], hdivskew_coll);
        else
            Hdivskew_space_new = NULL;

        Hdiv_space_lvls.Prepend(Hdiv_space_new);
        L2_space_lvls.Prepend(L2_space_new);
        H1_space_lvls.Prepend(H1_space_new);
        Hcurl_space_lvls.Prepend(Hcurl_space_new);
        Hdivskew_space_lvls.Prepend(Hdivskew_space_new);

        Hdiv_space->Update();
        H1_space->Update();
        L2_space->Update();
        Hcurl_space->Update();
        if (dim == 4)
            Hdivskew_space->Update();

        const SparseMatrix* P_Hdiv_local;
        const SparseMatrix* P_H1_local;
        const SparseMatrix* P_L2_local;
        const SparseMatrix* P_Hcurl_local;
        const SparseMatrix* P_Hdivskew_local;

        // Hdiv
        P_Hdiv_local = (SparseMatrix *)Hdiv_space->GetUpdateOperator();

        SparseMatrix * P_Hdiv_new = RemoveZeroEntries(*P_Hdiv_local);
        P_Hdiv_lvls.Prepend(P_Hdiv_new);

        auto d_td_coarse_Hdiv = Hdiv_space_lvls[1]->Dof_TrueDof_Matrix();
        SparseMatrix * RP_Hdiv_local = Mult(*Hdiv_space_lvls[0]->GetRestrictionMatrix(), *P_Hdiv_lvls[0]);

        HypreParMatrix * TrueP_Hdiv_new = d_td_coarse_Hdiv->LeftDiagMult(
                    *RP_Hdiv_local, Hdiv_space_lvls[0]->GetTrueDofOffsets());
        TrueP_Hdiv_new->CopyColStarts();
        TrueP_Hdiv_new->CopyRowStarts();

        TrueP_Hdiv_lvls.Prepend(TrueP_Hdiv_new);
        delete RP_Hdiv_local;

        // H1
        P_H1_local = (SparseMatrix *)H1_space->GetUpdateOperator();

        SparseMatrix * P_H1_new = RemoveZeroEntries(*P_H1_local);
        P_H1_lvls.Prepend(P_H1_new);

        auto d_td_coarse_H1 = H1_space_lvls[1]->Dof_TrueDof_Matrix();
        SparseMatrix * RP_H1_local = Mult(*H1_space_lvls[0]->GetRestrictionMatrix(), *P_H1_lvls[0]);

        HypreParMatrix * TrueP_H1_new = d_td_coarse_H1->LeftDiagMult(
                    *RP_H1_local, H1_space_lvls[0]->GetTrueDofOffsets());
        TrueP_H1_new->CopyColStarts();
        TrueP_H1_new->CopyRowStarts();

        TrueP_H1_lvls.Prepend(TrueP_H1_new);
        delete RP_H1_local;

        // L2
        P_L2_local = (SparseMatrix *)L2_space->GetUpdateOperator();

        SparseMatrix * P_L2_new = RemoveZeroEntries(*P_L2_local);
        P_L2_lvls.Prepend(P_L2_new);

        auto d_td_coarse_L2 = L2_space_lvls[1]->Dof_TrueDof_Matrix();
        SparseMatrix * RP_L2_local = Mult(*L2_space_lvls[0]->GetRestrictionMatrix(), *P_L2_lvls[0]);

        HypreParMatrix * TrueP_L2_new = d_td_coarse_L2->LeftDiagMult(
                    *RP_L2_local, L2_space_lvls[0]->GetTrueDofOffsets());
        TrueP_L2_new->CopyColStarts();
        TrueP_L2_new->CopyRowStarts();

        TrueP_L2_lvls.Prepend(TrueP_L2_new);
        delete RP_L2_local;

        // Hcurl
        P_Hcurl_local = (SparseMatrix *)Hcurl_space->GetUpdateOperator();

        SparseMatrix * P_Hcurl_new = RemoveZeroEntries(*P_Hcurl_local);
        P_Hcurl_lvls.Prepend(P_Hcurl_new);

        auto d_td_coarse_Hcurl = Hcurl_space_lvls[1]->Dof_TrueDof_Matrix();
        SparseMatrix * RP_Hcurl_local = Mult(*Hcurl_space_lvls[0]->GetRestrictionMatrix(), *P_Hcurl_lvls[0]);

        HypreParMatrix * TrueP_Hcurl_new = d_td_coarse_Hcurl->LeftDiagMult(
                    *RP_Hcurl_local, Hcurl_space_lvls[0]->GetTrueDofOffsets());
        TrueP_Hcurl_new->CopyColStarts();
        TrueP_Hcurl_new->CopyRowStarts();

        TrueP_Hcurl_lvls.Prepend(TrueP_Hcurl_new);
        delete RP_Hcurl_local;

        // Hdivskew
        if (dim == 4)
        {
            P_Hdivskew_local = (SparseMatrix *)Hdivskew_space->GetUpdateOperator();

            SparseMatrix * P_Hdivskew_new = RemoveZeroEntries(*P_Hdivskew_local);
            P_Hdivskew_lvls.Prepend(P_Hdivskew_new);

            auto d_td_coarse_Hdivskew = Hdivskew_space_lvls[1]->Dof_TrueDof_Matrix();
            SparseMatrix * RP_Hdivskew_local = Mult(*Hdivskew_space_lvls[0]->GetRestrictionMatrix(), *P_Hdivskew_lvls[0]);

            HypreParMatrix * TrueP_Hdivskew_new = d_td_coarse_Hdivskew->LeftDiagMult(
                        *RP_Hdivskew_local, Hdivskew_space_lvls[0]->GetTrueDofOffsets());
            TrueP_Hdivskew_new->CopyColStarts();
            TrueP_Hdivskew_new->CopyRowStarts();

            TrueP_Hdivskew_lvls.Prepend(TrueP_Hdivskew_new);
            delete RP_Hdivskew_local;
        }

        if (divfreedops_constructed)
        {
            int dim = pmesh_lvls[0]->Dimension();

            ParDiscreteLinearOperator * Divfree_op;
            if (dim == 3)
            {
                Divfree_op = new ParDiscreteLinearOperator(Hcurl_space_lvls[0], Hdiv_space_lvls[0]);
                Divfree_op->AddDomainInterpolator(new CurlInterpolator);
            }
            else
            {
                Divfree_op = new ParDiscreteLinearOperator(Hdivskew_space_lvls[0], Hdiv_space_lvls[0]);
                Divfree_op->AddDomainInterpolator(new DivSkewInterpolator);
            }

            Divfree_op->Assemble();
            Divfree_op->Finalize();
            HypreParMatrix * DivfreeDops_new = Divfree_op->ParallelAssemble();

            delete Divfree_op;

            DivfreeDops_lvls.Prepend(DivfreeDops_new);
        }

        if (doftruedofs_constructed)
        {
            int dim = pmesh_lvls[0]->Dimension();

            HypreParMatrix * DofTrueDof_L2_new = L2_space_lvls[0]->Dof_TrueDof_Matrix();
            HypreParMatrix * DofTrueDof_H1_new = H1_space_lvls[0]->Dof_TrueDof_Matrix();
            HypreParMatrix * DofTrueDof_Hdiv_new = Hdiv_space_lvls[0]->Dof_TrueDof_Matrix();
            HypreParMatrix * DofTrueDof_Hcurl_new = Hcurl_space_lvls[0]->Dof_TrueDof_Matrix();
            HypreParMatrix * DofTrueDof_Hdivskew_new;
            if (dim == 4)
                DofTrueDof_Hdivskew_new = Hdivskew_space_lvls[0]->Dof_TrueDof_Matrix();

            DofTrueDof_L2_lvls.Prepend(DofTrueDof_L2_new);
            DofTrueDof_H1_lvls.Prepend(DofTrueDof_H1_new);
            DofTrueDof_Hdiv_lvls.Prepend(DofTrueDof_Hdiv_new);
            DofTrueDof_Hcurl_lvls.Prepend(DofTrueDof_Hcurl_new);
            if (dim == 4)
                DofTrueDof_Hdivskew_lvls.Prepend(DofTrueDof_Hdivskew_new);
        }

        pmesh_ne = pmesh.GetNE();

        ++num_lvls;

        ++update_counter;
    } // end of if update is required
}


void GeneralHierarchy::ConstructDivfreeDops()
{
    int dim = pmesh_lvls[0]->Dimension();

    DivfreeDops_lvls.SetSize(num_lvls);

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

        delete Divfree_op;
    }

    divfreedops_constructed = true;
}

void GeneralHierarchy::ConstructDofTrueDofs()
{
    int dim = pmesh_lvls[0]->Dimension();

    DofTrueDof_L2_lvls.SetSize(num_lvls);
    DofTrueDof_H1_lvls.SetSize(num_lvls);
    DofTrueDof_Hdiv_lvls.SetSize(num_lvls);
    DofTrueDof_Hcurl_lvls.SetSize(num_lvls);
    if (dim == 4)
        DofTrueDof_Hdivskew_lvls.SetSize(num_lvls);

    for (int l = 0; l < num_lvls; ++l)
    {
        DofTrueDof_L2_lvls[l] = L2_space_lvls[l]->Dof_TrueDof_Matrix();
        DofTrueDof_H1_lvls[l] = H1_space_lvls[l]->Dof_TrueDof_Matrix();
        DofTrueDof_Hdiv_lvls[l] = Hdiv_space_lvls[l]->Dof_TrueDof_Matrix();
        DofTrueDof_Hcurl_lvls[l] = Hcurl_space_lvls[l]->Dof_TrueDof_Matrix();
        if (dim == 4)
            DofTrueDof_Hdivskew_lvls[l] = Hdivskew_space_lvls[l]->Dof_TrueDof_Matrix();
    }
    doftruedofs_constructed = true;
}


const Array<int>* GeneralHierarchy::ConstructTrueOffsetsforFormul(int level, const Array<SpaceName>& space_names)
{
    Array<int> * res = new Array<int>(space_names.Size() + 1);

    (*res)[0] = 0;
    for (int i = 0; i < space_names.Size(); ++i)
        (*res)[i + 1] = GetSpace(space_names[i], level)->TrueVSize();
    res->PartialSum();

    return res;
}

const Array<int>* GeneralHierarchy::ConstructOffsetsforFormul(int level, const Array<SpaceName>& space_names)
{
    Array<int> * res = new Array<int>(space_names.Size() + 1);

    (*res)[0] = 0;
    for (int i = 0; i < space_names.Size(); ++i)
        (*res)[i + 1] = GetSpace(space_names[i], level)->GetVSize();
    res->PartialSum();

    return res;
}


BlockOperator* GeneralHierarchy::ConstructTruePforFormul(int level, const Array<SpaceName>& space_names,
                                                         const Array<int>& row_offsets, const Array<int>& col_offsets)
{
    BlockOperator * res = new BlockOperator(row_offsets, col_offsets);

    for (int i = 0; i < space_names.Size(); ++i)
        res->SetDiagonalBlock(i, GetTruePspace(space_names[i], level), 1.0);

    return res;
}

BlockMatrix* GeneralHierarchy::ConstructPforFormul(int level, const Array<SpaceName>& space_names,
                                                   const Array<int>& row_offsets,
                                                   const Array<int>& col_offsets)
{
    BlockMatrix * res = new BlockMatrix(row_offsets, col_offsets);

    for (int i = 0; i < space_names.Size(); ++i)
        res->SetBlock(i, i, GetPspace(space_names[i], level));

    return res;
}



BlockOperator* GeneralHierarchy::ConstructTruePforFormul(int level, const FOSLSFormulation& formul,
                                                         const Array<int>& row_offsets, const Array<int>& col_offsets)
{
    const Array<SpaceName> * space_names  = formul.GetSpacesDescriptor();
    return ConstructTruePforFormul(level, *space_names, row_offsets, col_offsets);
}

Array<int>& GeneralHierarchy::GetEssBdrTdofsOrDofs(const char * tdof_or_dof,
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

/*
std::vector<Array<int>* >& GeneralHierarchy::GetEssBdrTdofsOrDofs(const char * tdof_or_dof,
                                                            const Array<SpaceName> &space_names,
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
*/

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

SparseMatrix* GeneralHierarchy::GetElementToDofs(SpaceName space_name, int level) const
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

BlockMatrix* GeneralHierarchy::GetElementToDofs(const Array<SpaceName>& space_names, int level,
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
        SparseMatrix * el2dofs_blk = ElementToDofs(*pfess[i]);
        res->SetBlock(i,i, el2dofs_blk);
    }

    res->owns_blocks = true;

    return res;
}

HypreParMatrix* GeneralHierarchy::GetDofTrueDof(SpaceName space_name, int level) const
{
    switch(space_name)
    {
    case HDIV:
        return DofTrueDof_Hdiv_lvls[level];
    case H1:
        return DofTrueDof_H1_lvls[level];
    case L2:
        return DofTrueDof_L2_lvls[level];
    case HCURL:
        return DofTrueDof_Hcurl_lvls[level];
    case HDIVSKEW:
        return DofTrueDof_Hdivskew_lvls[level];
    default:
        {
            MFEM_ABORT("Unknown or unsupported space name \n");
            break;
        }
    }

    return NULL;
}

std::vector<HypreParMatrix*> & GeneralHierarchy::GetDofTrueDof(const Array<SpaceName> &space_names, int level) const
{
    std::vector<HypreParMatrix*> * res = new std::vector<HypreParMatrix*>;
    res->resize(space_names.Size());
    for (int i = 0; i < space_names.Size(); ++i)
    {
        (*res)[i] = GetDofTrueDof(space_names[i], level);
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

void GeneralCylHierarchy::ConstructRestrictions()
{
    Restrict_bot_H1_lvls.SetSize(num_lvls);
    Restrict_bot_Hdiv_lvls.SetSize(num_lvls);
    Restrict_top_H1_lvls.SetSize(num_lvls);
    Restrict_top_Hdiv_lvls.SetSize(num_lvls);

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
    TrueP_bndbot_H1_lvls.SetSize(num_lvls - 1);
    TrueP_bndbot_Hdiv_lvls.SetSize(num_lvls - 1);
    TrueP_bndtop_H1_lvls.SetSize(num_lvls - 1);
    TrueP_bndtop_Hdiv_lvls.SetSize(num_lvls - 1);

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
HypreParMatrix * CopyHypreParMatrix(const HypreParMatrix& hpmat)
{
    HypreParMatrix * temp = hpmat.Transpose();
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

SparseMatrix* ElementToDofs(const FiniteElementSpace &fes)
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
    return res;
}

BlockMatrix * RAP(const BlockMatrix &Rt, const BlockMatrix &A, const BlockMatrix &P)
{
   BlockMatrix * R = Transpose(Rt);
   BlockMatrix * RA = Mult(*R,A);
   delete R;
   BlockMatrix * out = Mult(*RA, P);
   delete RA;
   return out;
}


// computes elpartition array which is used for computing slice meshes over different time moments
// elpartition is the output
// elpartition stores for each time moment a vector of integer indices of the mesh elements which intersect
// with the corresponding time plane
void Compute_elpartition (const Mesh& mesh, double t0, int Nmoments, double deltat, vector<vector<int> > & elpartition)
{
    bool verbose = false;
    int dim = mesh.Dimension();

    const Element * el;
    const int * vind;
    const double * vcoords;
    double eltmin, eltmax;

    for ( int elind = 0; elind < mesh.GetNE(); ++elind)
    {
        if (verbose)
            cout << "elind = " << elind << endl;
        el = mesh.GetElement(elind);
        vind = el->GetVertices();

        // computing eltmin and eltmax for an element = minimal and maximal time moments for each element
        eltmin = t0 + Nmoments * deltat;
        eltmax = 0.0;
        for (int vno = 0; vno < el->GetNVertices(); ++vno )
        {
            vcoords = mesh.GetVertex(vind[vno]);
            if ( vcoords[dim - 1] > eltmax )
                eltmax = vcoords[dim - 1];
            if ( vcoords[dim - 1] < eltmin )
                eltmin = vcoords[dim - 1];
        }


        if (verbose)
        {
            cout << "Special print: elind = " << elind << endl;
            for (int vno = 0; vno < el->GetNVertices(); ++vno )
            {
                cout << "vertex: ";
                vcoords = mesh.GetVertex(vind[vno]);
                for ( int coo = 0; coo < dim; ++coo )
                    cout << vcoords[coo] << " ";
                cout << endl;
            }

            cout << "eltmin = " << eltmin << " eltmax = " << eltmax << endl;
        }




        // deciding which time moments intersect the element if any
        //if ( (eltmin > t0 && eltmin < t0 + (Nmoments-1) * deltat) ||  (eltmax > t0 && eltmax < t0 + (Nmoments-1) * deltat))
        if ( (eltmax > t0 && eltmin < t0 + (Nmoments-1) * deltat))
        {
            if (verbose)
            {
                cout << "the element is intersected by some time moments" << endl;
                cout << "t0 = " << t0 << " deltat = " << deltat << endl;
                cout << fixed << setprecision(6);
                cout << "low bound = " << ceil( (max(eltmin,t0) - t0) / deltat  ) << endl;
                cout << "top bound = " << floor ((min(eltmax,t0+(Nmoments-1)*deltat) - t0) / deltat) << endl;
                cout << "4isl for low = " << max(eltmin,t0) - t0 << endl;
                cout << "magic number for low = " << (max(eltmin,t0) - t0) / deltat << endl;
                cout << "magic number for top = " << (min(eltmax,t0+(Nmoments-1)*deltat) - t0) / deltat << endl;
            }
            for ( int k = ceil( (max(eltmin,t0) - t0) / deltat  ); k <= floor ((min(eltmax,t0+(Nmoments-1)*deltat) - t0) / deltat) ; ++k)
            {
                //if (myid == 0 )
                if (verbose)
                {
                    cout << "k = " << k << endl;
                }
                elpartition[k].push_back(elind);
            }
        }
        else
        {
            if (verbose)
                cout << "the element is not intersected by any time moments" << endl;
        }
    }

    // intermediate output
    /*
    for ( int i = 0; i < Nmoments; ++i)
    {
        cout << "moment " << i << ": time = " << t0 + i * deltat << endl;
        cout << "size for this partition = " << elpartition[i].size() << endl;
        for ( int j = 0; j < elpartition[i].size(); ++j)
            cout << "el: " << elpartition[i][j] << endl;
    }
    */
    return;
}


// computes number of slice cell vertexes, slice cell vertex indices and coordinates
// for a given element with index = elind.
// updates the edgemarkers and vertex_count correspondingly
// pvec defines the slice plane
void computeSliceCell (const Mesh& mesh, int elind, vector<vector<double> > & pvec, vector<vector<double> > & ipoints, vector<int>& edgemarkers,
                             vector<vector<double> >& cellpnts, vector<int>& elvertslocal, int & nip, int & vertex_count )
{
    bool verbose = false; // probably should be a function argument
    int dim = mesh.Dimension();

    const int * edgeindices;
    int edgenolen, edgeind;
    Array<int> edgev(2);
    const double * v1, * v2;

    vector<vector<double> > edgeends(dim);
    edgeends[0].reserve(dim);
    edgeends[1].reserve(dim);

    DenseMatrix M(dim, dim);
    Vector sol(dim), rh(dim);

    vector<double> ip(dim);

    const Table& el_to_edge = mesh.ElementToEdgeTable();

    edgeindices = el_to_edge.GetRow(elind);
    edgenolen = el_to_edge.RowSize(elind);

    nip = 0;

    if (verbose)
        std::cout << "\nStarting the main over edges in computeSliceCell \n";

    for ( int edgeno = 0; edgeno < edgenolen; ++edgeno)
    {
        // true mesh edge index
        edgeind = edgeindices[edgeno];

        if (verbose)
            cout << "edgeind " << edgeind << endl;
        if (edgemarkers[edgeind] == -2) // if this edge was not considered
        {
            mesh.GetEdgeVertices(edgeind, edgev);

            // vertex coordinates
            v1 = mesh.GetVertex(edgev[0]);
            v2 = mesh.GetVertex(edgev[1]);

            // vertex coordinates as vectors of doubles, edgeends 0 is lower in time coordinate than edgeends[1]
            if (v1[dim-1] < v2[dim-1])
            {
                for ( int coo = 0; coo < dim; ++coo)
                {
                    edgeends[0][coo] = v1[coo];
                    edgeends[1][coo] = v2[coo];
                }
            }
            else
            {
                for ( int coo = 0; coo < dim; ++coo)
                {
                    edgeends[0][coo] = v2[coo];
                    edgeends[1][coo] = v1[coo];
                }
            }


            if (verbose)
            {
                cout << "edge vertices:" << endl;
                for (int i = 0; i < 2; ++i)
                {
                    cout << "vert ";
                    for ( int coo = 0; coo < dim; ++coo)
                        cout << edgeends[i][coo] << " ";
                    cout << "   ";
                }
                cout << endl;
            }


            // creating the matrix for computing the intersection point
            for ( int i = 0; i < dim; ++i)
                for ( int j = 0; j < dim - 1; ++j)
                    M(i,j) = pvec[j + 1][i];
            for ( int i = 0; i < dim; ++i)
                M(i,dim - 1) = edgeends[0][i] - edgeends[1][i];

            /*
            cout << "M" << endl;
            M.Print();
            cout << "M.Det = " << M.Det() << endl;
            */

            if ( fabs(M.Det()) > MYZEROTOL )
            {
                M.Invert();

                // setting righthand side
                for ( int i = 0; i < dim; ++i)
                    rh[i] = edgeends[0][i] - pvec[0][i];

                // solving the system
                M.Mult(rh, sol);

                if ( sol[dim-1] > 0.0 - MYZEROTOL && sol[dim-1] <= 1.0 + MYZEROTOL)
                {
                    for ( int i = 0; i < dim; ++i)
                        ip[i] = edgeends[0][i] + sol[dim-1] * (edgeends[1][i] - edgeends[0][i]);

                    if (verbose)
                    {
                        cout << "intersection point for this edge: " << endl;
                        for ( int i = 0; i < dim; ++i)
                            cout << ip[i] << " ";
                        cout << endl;
                    }

                    ipoints.push_back(ip);
                    //vrtindices[momentind].push_back(vertex_count);
                    elvertslocal.push_back(vertex_count);
                    edgemarkers[edgeind] = vertex_count;
                    cellpnts.push_back(ip);
                    nip++;
                    vertex_count++;
                }
                else
                {
                    if (verbose)
                        cout << "Line but not edge intersects" << endl;
                    edgemarkers[edgeind] = -1;
                }

            }
            else
                if (verbose)
                    cout << "Edge is parallel" << endl;
        }
        else // the edge was already considered -> edgemarkers store the vertex index
        {
            if (edgemarkers[edgeind] >= 0)
            {
                elvertslocal.push_back(edgemarkers[edgeind]);
                cellpnts.push_back(ipoints[edgemarkers[edgeind]]);
                nip++;
            }
        }

        //cout << "tempvec.size = " << tempvec.size() << endl;

    } // end of loop over element edges

    if (verbose)
        std::cout << "nip = " << nip << "\n";

    return;
}

// outputs the slice mesh information in VTK format
void outputSliceMeshVTK (const Mesh& mesh, std::stringstream& fname, std::vector<std::vector<double> > & ipoints,
                                std::list<int> &celltypes, int cellstructsize, std::list<std::vector<int> > &elvrtindices)
{
    int dim = mesh.Dimension();
    // output in the vtk format for paraview
    std::ofstream ofid(fname.str().c_str());
    ofid.precision(8);

    ofid << "# vtk DataFile Version 3.0" << endl;
    ofid << "Generated by MFEM" << endl;
    ofid << "ASCII" << endl;
    ofid << "DATASET UNSTRUCTURED_GRID" << endl;

    ofid << "POINTS " << ipoints.size() << " double" << endl;
    for (unsigned int vno = 0; vno < ipoints.size(); ++vno)
    {
        for ( int c = 0; c < dim - 1; ++c )
        {
            ofid << ipoints[vno][c] << " ";
        }
        if (dim == 3)
            ofid << ipoints[vno][dim - 1] << " ";
        ofid << endl;
    }

    ofid << "CELLS " << celltypes.size() << " " << cellstructsize << endl;
    std::list<int>::const_iterator iter;
    std::list<vector<int> >::const_iterator iter2;
    for (iter = celltypes.begin(), iter2 = elvrtindices.begin();
         iter != celltypes.end() && iter2 != elvrtindices.end()
         ; ++iter, ++iter2)
    {
        //cout << *it;
        int npoints;
        if (*iter == VTKTETRAHEDRON)
            npoints = 4;
        else if (*iter == VTKWEDGE)
            npoints = 6;
        else if (*iter == VTKQUADRIL)
            npoints = 4;
        else //(*iter == VTKTRIANGLE)
            npoints = 3;
        ofid << npoints << " ";

        for ( int i = 0; i < npoints; ++i)
            ofid << (*iter2)[i] << " ";
        ofid << endl;
    }

    ofid << "CELL_TYPES " << celltypes.size() << endl;
    for (iter = celltypes.begin(); iter != celltypes.end(); ++iter)
    {
        ofid << *iter << endl;
    }

    // test lines for cell data
    ofid << "CELL_DATA " << celltypes.size() << endl;
    ofid << "SCALARS cekk_scalars double 1" << endl;
    ofid << "LOOKUP_TABLE default" << endl;
    int cnt = 0;
    for (iter = celltypes.begin(); iter != celltypes.end(); ++iter)
    {
        ofid << cnt * 1.0 << endl;
        cnt++;
    }
    return;
}

// reorders the cell vertices so as to have the cell vertex ordering compatible with VTK format
// the output is the sorted elvertexes (which is also the input)
void reorder_cellvertices ( int dim, int nip, std::vector<std::vector<double> > & cellpnts, std::vector<int> & elvertexes)
{
    bool verbose = false;
    // used only for checking the orientation of tetrahedrons
    DenseMatrix Mtemp(3, 3);

    // special reordering of vertices is required for the vtk wedge, so that
    // vertices are added one base after another and not as a mix

    if (nip == 6)
    {

        /*
        cout << "Sorting the future wedge" << endl;
        cout << "Before sorting: " << endl;
        for (int i = 0; i < 6; ++i)
        {
            cout << "vert " << i << ":";
            for ( int j = 0; j < dim - 1; ++j)
                cout << cellpnts[i][j] << " ";
            cout << endl;
        }
        */


        // FIX IT: NOT TESTED AT ALL
        int permutation[6];
        if ( sortWedge3d (cellpnts, permutation) == false )
        {
            cout << "sortWedge returns false, possible bad behavior" << endl;
            return;
        }

        /*
        cout << "After sorting: " << endl;
        for (int i = 0; i < 6; ++i)
        {
            cout << "vert " << i << ":";
            for ( int j = 0; j < dim - 1; ++j)
                cout << cellpnts[permutation[i]][j] << " ";
            cout << endl;
        }
        */

        int temp[6];
        for ( int i = 0; i < 6; ++i)
            temp[i] = elvertexes[permutation[i]];
        for ( int i = 0; i < 6; ++i)
            elvertexes[i] = temp[i];


        double det = 0.0;

        for ( int i = 0; i < dim - 1; ++i)
            Mtemp(i,0) = (1.0/3.0)*(cellpnts[permutation[3]][i] + cellpnts[permutation[4]][i] + cellpnts[permutation[5]][i])
                    - cellpnts[permutation[0]][i];

        for ( int i = 0; i < dim - 1; ++i)
            Mtemp(i,1) = cellpnts[permutation[2]][i] - cellpnts[permutation[0]][i];

        for ( int i = 0; i < dim - 1; ++i)
            Mtemp(i,2) = cellpnts[permutation[1]][i] - cellpnts[permutation[0]][i];

        det = Mtemp.Det();

        if (verbose)
        {
            if (det < 0)
                cout << "orientation for wedge = negative" << endl;
            else if (det == 0.0)
                cout << "error for wedge: bad volume" << endl;
            else
                cout << "orientation for wedge = positive" << endl;
        }

        if (det < 0)
        {
            if (verbose)
                cout << "Have to swap the vertices to change the orientation of wedge" << endl;
            int tmp;
            tmp = elvertexes[1];
            elvertexes[1] = elvertexes[0];
            elvertexes[1] = tmp;
            //Swap(*(elvrtindices[momentind].end()));
            tmp = elvertexes[4];
            elvertexes[4] = elvertexes[3];
            elvertexes[4] = tmp;
        }

    }


    // positive orientation is required for vtk tetrahedron
    // normal to the plane with first three vertexes should poit towards the 4th vertex

    if (nip == 4 && dim == 4)
    {
        /*
        cout << "tetrahedra points" << endl;
        for (int i = 0; i < 4; ++i)
        {
            cout << "vert " << i << ":";
            for ( int j = 0; j < dim - 1; ++j)
                cout << cellpnts[i][j] << " ";
            cout << endl;
        }
        */

        double det = 0.0;

        for ( int i = 0; i < dim - 1; ++i)
            Mtemp(i,0) = cellpnts[3][i] - cellpnts[0][i];

        for ( int i = 0; i < dim - 1; ++i)
            Mtemp(i,1) = cellpnts[2][i] - cellpnts[0][i];

        for ( int i = 0; i < dim - 1; ++i)
            Mtemp(i,2) = cellpnts[1][i] - cellpnts[0][i];

        //Mtemp.Print();

        det = Mtemp.Det();

        if (verbose)
        {
            if (det < 0)
                cout << "orientation for tetra = negative" << endl;
            else if (det == 0.0)
                cout << "error for tetra: bad volume" << endl;
            else
                cout << "orientation for tetra = positive" << endl;
        }

        //return;

        if (det < 0)
        {
            if (verbose)
                cout << "Have to swap the vertices to change the orientation of tetrahedron" << endl;
            int tmp = elvertexes[1];
            elvertexes[1] = elvertexes[0];
            elvertexes[1] = tmp;
            //Swap(*(elvrtindices[momentind].end()));
        }

    }


    // in 2D case the vertices of a quadrilateral should be umbered in a counter-clock wise fashion
    if (nip == 4 && dim == 3)
    {
        /*
        cout << "Sorting the future quadrilateral" << endl;
        cout << "Before sorting: " << endl;
        for (int i = 0; i < nip; ++i)
        {
            cout << "vert " << elvertexes[i] << ":";
            for ( int j = 0; j < dim - 1; ++j)
                cout << cellpnts[i][j] << " ";
            cout << endl;
        }
        */

        int permutation[4];
        sortQuadril2d(cellpnts, permutation);

        int temp[4];
        for ( int i = 0; i < 4; ++i)
            temp[i] = elvertexes[permutation[i]];
        for ( int i = 0; i < 4; ++i)
            elvertexes[i] = temp[i];

        /*
        cout << "After sorting: " << endl;
        for (int i = 0; i < nip; ++i)
        {
            cout << "vert " << elvertexes[i] << ":";
            for ( int j = 0; j < dim - 1; ++j)
                cout << cellpnts[permutation[i]][j] << " ";
            cout << endl;
        }
        */

    }

    return;
}

// scalar product of two vectors (outputs 0 if vectors have different length)
double sprod(std::vector<double> vec1, std::vector<double> vec2)
{
    if (vec1.size() != vec2.size())
        return 0.0;
    double res = 0.0;
    for ( unsigned int c = 0; c < vec1.size(); ++c)
        res += vec1[c] * vec2[c];
    return res;
}

double l2Norm(std::vector<double> vec)
{
    return sqrt(sprod(vec,vec));
}

bool intdComparison(const std::pair<int,double> &a,const std::pair<int,double> &b)
{
    return a.second>b.second;
}


// only first 2 coordinates of each element of Points is used (although now the
// input is 4 3-dimensional points but the last coordinate is time so it is not used
// because the slice is with t = const planes
// sorts in a counter-clock fashion required by VTK format for quadrilateral
// the main output is the permutation of the input points array
bool sortQuadril2d(std::vector<std::vector<double> > & Points, int * permutation)
{
    bool verbose = false;

    if (Points.size() != 4)
    {
        cout << "Error: sortQuadril2d should be called only for a vector storing 4 points" << endl;
        return false;
    }
    /*
    for ( int p = 0; p < Points.size(); ++p)
        if (Points[p].size() != 2)
        {
            cout << "Error: sortQuadril2d should be called only for a vector storing 4 2d-points" << endl;
            return false;
        }
    */

    /*
    cout << "Points inside sortQuadril2d() \n";
    for (int i = 0; i < 4; ++i)
    {
        cout << "vert " << i << ":";
        for ( int j = 0; j < 2; ++j)
            cout << Points[i][j] << " ";
        cout << endl;
    }
    */


    int argbottom = 0; // index of the the vertex with the lowest y-coordinate
    for (int p = 1; p < 4; ++p)
        if (Points[p][1] < Points[argbottom][1])
            argbottom = p;

    if (verbose)
        cout << "argbottom = " << argbottom << endl;

    // cosinuses of angles between radius vectors from vertex argbottom to the others and positive x-direction
    vector<pair<int, double> > cos(3);
    vector<vector<double> > radiuses(3);
    vector<double> xort(2);
    xort[0] = 1.0;
    xort[1] = 0.0;
    int cnt = 0;
    for (int p = 0; p < 4; ++p)
    {
        if (p != argbottom)
        {
            cos[cnt].first = p;
            for ( int c = 0; c < 2; ++c)
                radiuses[cnt].push_back(Points[p][c] - Points[argbottom][c]);
            cos[cnt].second = sprod(radiuses[cnt], xort) / l2Norm(radiuses[cnt]);
            cnt ++;
        }
    }

    //int permutation[4];
    permutation[0] = argbottom;

    std::sort(cos.begin(), cos.end(), intdComparison);

    for ( int i = 0; i < 3; ++i)
        permutation[1 + i] = cos[i].first;

    if (verbose)
    {
        cout << "permutation:" << endl;
        for (int i = 0; i < 4; ++i)
            cout << permutation[i] << " ";
        cout << endl;
    }

    // not needed actually. onlt for debugging. actually the output is the correct permutation
    /*
    vector<vector<double>> temp(4);
    for ( int p = 0; p < 4; ++p)
        for ( int i = 0; i < 3; ++i)
            temp[p].push_back(Points[permutation[p]][i]);

    for ( int p = 0; p < 4; ++p)
        for ( int i = 0; i < 3; ++i)
            Points[p][i] = temp[p][i];
    */
    return true;
}

// sorts the vertices in order for the points to form a proper vtk wedge
// first three vertices should be the base, with normal to (0,1,2)
// looking opposite to the direction of where the second base is.
// This ordering is required by VTK format for wedges, look
// in vtk wedge class definitio for explanations
// the main output is the permutation of the input vertexes array
bool sortWedge3d(std::vector<std::vector<double> > & Points, int * permutation)
{
    /*
    cout << "wedge points:" << endl;
    for ( int i = 0; i < Points.size(); ++i)
    {
        for ( int j = 0; j < Points[i].size(); ++j)
            cout << Points[i][j] << " ";
        cout << endl;
    }
    */

    vector<double> p1 = Points[0];
    int pn2 = -1;
    vector<int> pnum2;

    //bestimme die 2 quadrate
    for(unsigned int i=1; i<Points.size(); i++)
    {
        vector<double> dets;
        for(unsigned int k=1; k<Points.size()-1; k++)
        {
            for(unsigned int l=k+1; l<Points.size(); l++)
            {
                if(k!=i && l!=i)
                {
                    vector<double> Q1(3);
                    vector<double> Q2(3);
                    vector<double> Q3(3);

                    for ( int c = 0; c < 3; c++)
                        Q1[c] = p1[c] - Points[i][c];
                    for ( int c = 0; c < 3; c++)
                        Q2[c] = p1[c] - Points[k][c];
                    for ( int c = 0; c < 3; c++)
                        Q3[c] = p1[c] - Points[l][c];

                    //vector<double> Q1 = p1 - Points[i];
                    //vector<double> Q2 = p1 - Points[k];
                    //vector<double> Q3 = p1 - Points[l];

                    DenseMatrix MM(3,3);
                    MM(0,0) = Q1[0]; MM(0,1) = Q2[0]; MM(0,2) = Q3[0];
                    MM(1,0) = Q1[1]; MM(1,1) = Q2[1]; MM(1,2) = Q3[1];
                    MM(2,0) = Q1[2]; MM(2,1) = Q2[2]; MM(2,2) = Q3[2];
                    double determ = MM.Det();

                    dets.push_back(determ);
                }
            }
        }

        double max_ = 0; double min_ = fabs(dets[0]);
        for(unsigned int m=0; m<dets.size(); m++)
        {
            if(max_<fabs(dets[m])) max_ = fabs(dets[m]);
            if(min_>fabs(dets[m])) min_ = fabs(dets[m]);
        }

        //for ( int in = 0; in < dets.size(); ++in)
            //cout << "det = " << dets[in] << endl;

        if(max_!=0) for(unsigned int m=0; m<dets.size(); m++) dets[m] /= max_;

        //cout << "max_ = " << max_ << endl;

        int count = 0;
        vector<bool> el;
        for(unsigned int m=0; m<dets.size(); m++) { if(fabs(dets[m]) < 1e-8) { count++; el.push_back(true); } else el.push_back(false); }

        if(count==2)
        {
            for(unsigned int k=1, m=0; k<Points.size()-1; k++)
                for(unsigned int l=k+1; l<Points.size(); l++)
                {
                    if(k!=i && l!=i)
                    {
                        if(el[m]) { pnum2.push_back(k); pnum2.push_back(l); }
                        m++;
                    }

                }

            pn2 = i;
            break;
        }

        if(count == 0 || count > 2)
        {
            //cout << "count == 0 || count > 2" << endl;
            //cout << "count = " << count << endl;
            return false;
        }
    }

    if(pn2<0)
    {
        //cout << "pn2 < 0" << endl;
        return false;
    }


    vector<int> oben(3); oben[0] = pn2;
    vector<int> unten(3); unten[0] = 0;

    //winkel berechnen
    vector<double> pp1(3);
    vector<double> pp2(3);
    for ( int c = 0; c < 3; c++)
        pp1[c] = Points[0][c] - Points[pn2][c];
    for ( int c = 0; c < 3; c++)
        pp2[c] = Points[pnum2[0]][c] - Points[pn2][c];
    //vector<double> pp1 = Points[0] - Points[pn2];
    //vector<double> pp2 = Points[pnum2[0]] - Points[pn2];
    double w1 = sprod(pp1, pp2)/(l2Norm(pp1)*l2Norm(pp2));
    for ( int c = 0; c < 3; c++)
        pp2[c] = Points[pnum2[1]][c] - Points[pn2][c];
    //pp2 = Points[pnum2[1]]- Points[pn2];
    double w2 = sprod(pp1, pp2)/(l2Norm(pp1)*l2Norm(pp2));

    if(w1 < w2)  { oben[1] = pnum2[0]; unten[1] = pnum2[1]; }
    else{ oben[1] = pnum2[1]; unten[1] = pnum2[0]; }

    for ( int c = 0; c < 3; c++)
        pp2[c] = Points[pnum2[2]][c] - Points[pn2][c];
    //pp2 = Points[pnum2[2]] - Points[pn2];
    w1 = sprod(pp1, pp2)/(l2Norm(pp1)*l2Norm(pp2));
    for ( int c = 0; c < 3; c++)
        pp2[c] = Points[pnum2[3]][c] - Points[pn2][c];
    //pp2 = Points[pnum2[3]]- Points[pn2];
    w2 = sprod(pp1, pp2)/(l2Norm(pp1)*l2Norm(pp2));

    if(w1 < w2)  { oben[2] = pnum2[2]; unten[2] = pnum2[3]; }
    else{ oben[2] = pnum2[3]; unten[2] = pnum2[2]; }

    for(unsigned int i=0; i<unten.size(); i++) permutation[i] = unten[i];
    for(unsigned int i=0; i<oben.size(); i++)  permutation[i + unten.size()] = oben[i];

    //not needed since we actually need the permutation only
    /*
    vector<vector<double>> pointssort;
    for(unsigned int i=0; i<unten.size(); i++) pointssort.push_back(Points[unten[i]]);
    for(unsigned int i=0; i<oben.size(); i++) pointssort.push_back(Points[oben[i]]);

    for(unsigned int i=0; i<pointssort.size(); i++) Points[i] = pointssort[i];
    */

    return true;
}


// Computes and outputs in VTK format slice meshes of a given 3D or 4D mesh
// by time-like planes t = t0 + k * deltat, k = 0, ..., Nmoments - 1
// myid is used for creating different output files by different processes
// if the mesh is parallel
// usually it is reasonable to refer myid to the process id in the communicator
// so as to produce a correct output for parallel ParaView visualization
void ComputeSlices(const Mesh& mesh, double t0, int Nmoments, double deltat, int myid)
{
    bool verbose = false;

    if ( mesh.GetElementBaseGeometry() != Geometry::PENTATOPE &&
         mesh.GetElementBaseGeometry() != Geometry::TETRAHEDRON )
    {
        std::cout << "ComputeSlices() is implemented only for pentatops "
                    "and tetrahedrons \n" << std::flush;
        return;
    }

    int dim = mesh.Dimension();

    //const Table& el_to_edge = mesh.ElementToEdgeTable();

    /*
    if (!el_to_edge)
    {
        el_to_edge = new Table;
        NumOfEdges = mesh.GetElementToEdgeTable(*el_to_edge, be_to_edge);
    }
    */

    // = -2 if not considered, -1 if considered, but does not intersected, index of this vertex in the new 3d mesh otherwise
    // refilled for each time moment
    vector<int> edgemarkers(mesh.GetNEdges());

    // stores indices of elements which are intersected by planes related to the time moments
    vector<vector<int> > elpartition(Nmoments);
    // can make it faster, if any estimates are known for how many elements are intersected by a single time plane
    //for ( int i = 0; i < Nmoments; ++i)
        //elpartition[i].reserve(100);

    // *************************************************************************
    // step 1 of x: loop over all elements and compute elpartition for all time
    // moments.
    // *************************************************************************

    Compute_elpartition (mesh, t0, Nmoments, deltat, elpartition);


    // *************************************************************************
    // step 2 of x: looping over time momemnts and slicing elements for each
    // given time moment, and outputs the resulting slice mesh in VTK format
    // *************************************************************************

    // slicing the elements, time moment over time moment
    int elind;

    vector<vector<double> > pvec(dim);
    for ( int i = 0; i < dim; ++i)
        pvec[i].reserve(dim);

    // used only for checking the orientation of tetrahedrons and quadrilateral vertexes reordering
    //DenseMatrix Mtemp(3, 3);

    // output data structures for vtk format
    // for each time moment holds a list with cell type for each cell
    vector<std::list<int> > celltypes(Nmoments);
    // for each time moment holds a list with vertex indices
    //vector<std::list<int>> vrtindices(Nmoments);
    // for each time moment holds a list with cell type for each cell
    vector<std::list<vector<int> > > elvrtindices(Nmoments);

    // number of integers in cell structure - for each cell 1 integer (number of vertices) +
    // + x integers (vertex indices)
    int cellstructsize;
    int vertex_count; // number of vertices in the slice mesh for a single time moment

    // loop over time moments
    for ( int momentind = 0; momentind < Nmoments; ++momentind )
    {
        if (verbose)
            cout << "Time moment " << momentind << ": time = " << t0 + momentind * deltat << endl;

        // refilling edgemarkers, resetting vertex_count and cellstructsize
        for ( int i = 0; i < mesh.GetNEdges(); ++i)
            edgemarkers[i] = -2;

        vertex_count = 0;
        cellstructsize = 0;

        vector<vector<double> > ipoints; // one of main arrays: all intersection points for a given time moment

        // vectors, defining the plane of the slice p0, p1, p2 (and p3 in 4D)
        // p0 is the time aligned vector for the given time moment
        // p1, p2 (and p3) - basis orts for the plane
        // pvec is {p0,p1,p2,p3} vector
        for ( int i = 0; i < dim; ++i)
            for ( int j = 0; j < dim; ++j)
                pvec[i][dim - 1 - j] = ( i == j ? 1.0 : 0.0);
        pvec[0][dim - 1] = t0 + momentind * deltat;

        // loop over elements intersected by the plane realted to a given time moment
        // here, elno = index in elpartition[momentind]
        for ( unsigned int elno = 0; elno < elpartition[momentind].size(); ++elno)
        //for ( int elno = 0; elno < 3; ++elno)
        {
            vector<int> tempvec;             // vertex indices for the cell of the slice mesh
            tempvec.reserve(6);
            vector<vector<double> > cellpnts; //points of the cell of the slice mesh
            cellpnts.reserve(6);

            // true mesh element index
            elind = elpartition[momentind][elno];
            //Element * el = GetElement(elind);

            if (verbose)
                cout << "Element: " << elind << endl;

            // computing number of intersection points, indices and coordinates for
            // local slice cell vertexes (cellpnts and tempvec)  and adding new intersection
            // points and changing edges markers for a given element elind
            // and plane defined by pvec
            int nip;
            computeSliceCell (mesh, elind, pvec, ipoints, edgemarkers, cellpnts, tempvec, nip, vertex_count);

            if ( (dim == 4 && (nip != 4 && nip != 6)) || (dim == 3 && (nip != 3 && nip != 4)) )
                cout << "Strange nip =  " << nip << " for elind = " << elind << ", time = " << t0 + momentind * deltat << endl;
            else
            {
                if (nip == 4) // tetrahedron in 3d or quadrilateral in 2d
                    if (dim == 4)
                        celltypes[momentind].push_back(VTKTETRAHEDRON);
                    else // dim == 3
                        celltypes[momentind].push_back(VTKQUADRIL);
                else if (nip == 6) // prism
                    celltypes[momentind].push_back(VTKWEDGE);
                else // nip == 3 = triangle
                    celltypes[momentind].push_back(VTKTRIANGLE);

                cellstructsize += nip + 1;

                elvrtindices[momentind].push_back(tempvec);

                // special reordering of cell vertices, required for the wedge,
                // tetrahedron and quadrilateral cells
                reorder_cellvertices (dim, nip, cellpnts, elvrtindices[momentind].back());

                if (verbose)
                    cout << "nip for the element = " << nip << endl;
            }

        } // end of loop over elements for a given time moment

        // intermediate output
        std::stringstream fname;
        fname << "slicemesh_"<< dim - 1 << "d_myid_" << myid << "_moment_" << momentind << ".vtk";
        outputSliceMeshVTK (mesh, fname, ipoints, celltypes[momentind], cellstructsize, elvrtindices[momentind]);


    } //end of loop over time moments

    /*
    // if not deleted here, gets segfault for more than two parallel refinements afterwards
    delete edge_vertex;
    edge_vertex = NULL;
    */

    return;
}

// finds a prticular solution to a divergence constraint
// by solving a Poisson equation
ParGridFunction * FindParticularSolution(ParFiniteElementSpace * Hdiv_space,
                                         const HypreParMatrix & B, const Vector& rhs, bool verbose)
{
    MPI_Comm comm = Hdiv_space->GetComm();

    if (verbose)
        std::cout << "Solving Poisson problem for finding a particular solution \n";

    MFEM_ASSERT(Hdiv_space->TrueVSize() == B.Width(),
                "Dimension of Hdiv_space and divergence matrix B mismatch!");

    ParGridFunction * sigma_hat = new ParGridFunction(Hdiv_space);

    HypreParMatrix *BT = B.Transpose();
    HypreParMatrix *BBT = ParMult(&B, BT);

    HypreBoomerAMG * invBBT = new HypreBoomerAMG(*BBT);
    invBBT->SetPrintLevel(0);

    mfem::CGSolver solver(comm);
    solver.SetPrintLevel(0);
    solver.SetMaxIter(70000);
    solver.SetRelTol(1.0e-12);
    solver.SetAbsTol(1.0e-14);
    solver.SetPreconditioner(*invBBT);
    solver.SetOperator(*BBT);

    Vector lapl_sol(B.Height());
    solver.Mult(rhs, lapl_sol);

    Vector truesigma_hat(Hdiv_space->TrueVSize());
    B.MultTranspose(lapl_sol, truesigma_hat);

    sigma_hat->Distribute(truesigma_hat);

    delete invBBT;
    delete BBT;
    delete BT;

    if (verbose)
        std::cout << "Particular solution has been computed \n";

    return sigma_hat;
}



} // for namespace mfem
