///                           MFEM(with 4D elements) CFOSLS for 3D/4D transport equation
///                                solved by an anisotropic multigrid preconditioner
///                                              in div-free setting.
///
/// The problem considered in this example is
///                             du/dt + b * u = f (either 3D or 4D in space-time)
/// casted in the CFOSLS formulation
/// 1) either in Hdiv-L2 case:
///                             (K sigma, sigma) -> min
/// where sigma is from H(div), u is recovered (as an element of L^2) from sigma = b * u,
/// and K = (I - bbT / || b ||);
/// 2) or in Hdiv-H1-L2 case
///                             || sigma - b * u || ^2 -> min
/// where sigma is from H(div) and u is from H^1;
/// minimizing in all cases under the constraint
///                             div sigma = f.
///
/// The problem is discretized using RT, linear Lagrange and discontinuous constants in 3D/4D.
/// The current 3D tests are either in cube.
///
/// The problem is then solved in the div-free formulation, where
/// we first find a particular solution to the divergence constraint, and then search for the
/// div-free correction, casting the system's first component into H(curl) (in 3D).
///
/// Then, to find the div-free correction, CG is used, preconditioned by a geometric multigrid.

/// This example demonstrates usage of such classes from mfem/cfosls/ as
/// FOSLSProblem, FOSLSDivfreeProblem, GeneralAnisoHierarchy, GeneralMultirid,
/// MultigridToolsHierarchy and DivConstraintSolver.
///
/// (*) Due to the lack of copy constructor for non-conforming ParMesh in MFEM,
/// the actual anisotropic multigrid has not been tested (and the corresponding
/// setup in 4. is commented). But hopefully, it would work.
///
/// (**) This code was tested in serial with isotropic multigrid
/// (***) This code has not been tested for memory leaks.
///
/// Typical run of this example: ./cfosls_hyperbolic_anisoMG --whichD 3 --spaceS H1 -no-vis
/// If you want to use the Hdiv-H1-L2 formulation, you will need not only change --spaceS option but also
/// change the source code, around 3.
///
/// Other examples with geometric multigrid are cfosls_hyperbolic_multigrid.cpp
/// and cfosls_hyperbolic_newsolver.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>

using namespace std;
using namespace mfem;
using std::shared_ptr;
using std::make_shared;

int main(int argc, char *argv[])
{
    // 1. Initialize MPI
    int num_procs, myid;

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    bool verbose = (myid == 0);
    bool visualization = 0;

    int numsol          = 0;

    const char *space_for_S = "H1";    // "H1" or "L2"

    // solver options
    int prec_option = 1;
    int max_num_iter = 2000;
    double rtol = 1e-12;//1e-7;//1e-9;
    double atol = 1e-14;//1e-9;//1e-12;

    int feorder = 0;

    if (verbose)
        cout << "Solving (С)FOSLS Transport equation with anisotropic multigrid \n";

    // 2. Parse command-line options.
    OptionsParser args(argc, argv);
    args.AddOption(&feorder, "-o", "--feorder",
                   "Finite element order (polynomial degree).");
    args.AddOption(&space_for_S, "-spaceS", "--spaceS",
                   "Space for S: L2 or H1.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&prec_option, "-precopt", "--prec-option",
                   "Preconditioner choice: 2: block diagonal MG   3: monolithic MG.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");

    args.Parse();
    if (!args.Good())
    {
       if (verbose)
       {
          args.PrintUsage(cout);
       }
       MPI_Finalize();
       return 1;
    }
    if (verbose)
    {
       args.PrintOptions(cout);
    }

    numsol = -3;
    if (verbose)
        std::cout << "For the records: numsol = " << numsol << "\n";

    StopWatch chrono;

    // 3. Setting aliases for the type of the problem to be considered
    // Hdiv-H1 case
    MFEM_ASSERT(strcmp(space_for_S,"H1") == 0, "Space for S must be H1 in this case!\n");
    using FormulType = CFOSLSFormulation_HdivH1Hyper;
    using FEFormulType = CFOSLSFEFormulation_HdivH1Hyper;
    using BdrCondsType = BdrConditions_CFOSLS_HdivH1_Hyper;
    using ProblemType = FOSLSProblem_HdivH1L2hyp;
    using DivfreeFormulType = CFOSLSFormulation_HdivH1DivfreeHyp;
    using DivfreeFEFormulType = CFOSLSFEFormulation_HdivH1DivfreeHyper;

    /*
    // Hdiv-L2 case
    MFEM_ASSERT(strcmp(space_for_S,"L2") == 0, "Space for S must be H1 in this case!\n");
    using FormulType = CFOSLSFormulation_HdivL2Hyper;
    using FEFormulType = CFOSLSFEFormulation_HdivL2Hyper;
    using BdrCondsType = BdrConditions_CFOSLS_HdivL2_Hyper;
    using ProblemType = FOSLSProblem_HdivL2hyp;
    using DivfreeFormulType = CFOSLSFormulation_HdivDivfreeHyp;
    using DivfreeFEFormulType = CFOSLSFEFormulation_HdivDivfreeHyp;
    */

    // 4. Creating the hexahedral mesh and the hierarchy of meshes on top of it
    auto mesh = make_shared<Mesh>(2, 2, 2, Element::HEXAHEDRON, 1);
    // doing one refinement to turn the mesh into non-conforming
    // since "ParMesh must be constructed from non-conforming serial mesh" happens
    // later otherwise
    Array<Refinement> refs(mesh->GetNE());
    for (int i = 0; i < mesh->GetNE(); i++)
        refs[i] = Refinement(i, 7);
    mesh->GeneralRefinement(refs, -1, -1);

    auto pmesh = make_shared<ParMesh>(comm, *mesh);

    // creating ref. flags for the hierarchy
    // there will be applied from end to start
    // (ref_flags[0] will be applied last)
    int par_ref_lvls = 1;
    Array<int> ref_flags(par_ref_lvls);
    ref_flags[0] = 3;

    /*
    int par_ref_lvls = 5;
    Array<int> ref_flags(par_ref_lvls);
    ref_flags[0] = 3;
    ref_flags[1] = 3;
    ref_flags[2] = 4;
    ref_flags[3] = 4;
    ref_flags[4] = 7;
    */

    /*
    int par_ref_lvls = 2;
    Array<int> ref_flags(par_ref_lvls);
    ref_flags[0] = -1;
    ref_flags[1] = -1;
    */

    const int dim = 3;

    // constructing the general hierarchy on top of the coarse mesh,
    // refining it, keeping pmesh as the finest mesh
    GeneralAnisoHierarchy * hierarchy = new GeneralAnisoHierarchy(ref_flags, *pmesh, 0, verbose, true);
    hierarchy->ConstructDivfreeDops();
    hierarchy->ConstructDofTrueDofs();
    hierarchy->ConstructEl2Dofs();

    int nlevels = hierarchy->Nlevels();

    // 5. Creating instances of original problem and a problem reformulation in the
    // div-free space for h(div) component

    FormulType * formulat = new FormulType (dim, numsol, verbose);
    FEFormulType* fe_formulat = new FEFormulType(*formulat, feorder);
    BdrConditions * bdr_conds = new BdrCondsType(*pmesh);

    // creating the original problem
    FOSLSProblem* problem = hierarchy->BuildDynamicProblem<ProblemType>
            (*bdr_conds, *fe_formulat, prec_option, verbose);
    hierarchy->AttachProblem(problem);

    // creating div-free problem
    DivfreeFormulType * formulat_divfree = new DivfreeFormulType (dim, numsol, verbose);
    DivfreeFEFormulType * fe_formulat_divfree = new DivfreeFEFormulType(*formulat_divfree, feorder);

    FOSLSDivfreeProblem* divfree_problem = hierarchy->BuildDynamicProblem<FOSLSDivfreeProblem>
            (*bdr_conds, *fe_formulat_divfree, prec_option, verbose);

    divfree_problem->ConstructDivfreeHpMats();
    divfree_problem->CreateOffsetsRhsSol();
    BlockOperator * divfree_problem_op = ConstructDivfreeProblemOp(*divfree_problem, *problem);
    divfree_problem->ResetOp(*divfree_problem_op, true);

    // first, creating a preconditioner for the divfree problem (later to be replaced by multigrid)
    divfree_problem->InitSolver(verbose);
    divfree_problem->CreatePrec(*divfree_problem->GetOp(), prec_option, verbose);
    divfree_problem->ChangeSolver();
    divfree_problem->UpdateSolverPrec();

    hierarchy->AttachProblem(divfree_problem);

    // 6. Creating necessary ingredients for particular solution finder
    ComponentsDescriptor * descriptor;
    {
        bool with_Schwarz = true;
        bool optimized_Schwarz = true;
        bool with_Hcurl = true;
        bool with_coarsest_partfinder = true;
        bool with_coarsest_hcurl = true;
        bool with_monolithic_GS = false;
        bool with_nobnd_op = true;
        descriptor = new ComponentsDescriptor(with_Schwarz, optimized_Schwarz,
                                              with_Hcurl, with_coarsest_partfinder,
                                              with_coarsest_hcurl, with_monolithic_GS,
                                              with_nobnd_op);
    }
    MultigridToolsHierarchy * mgtools_hierarchy =
            new MultigridToolsHierarchy(*hierarchy, 0, *descriptor);

    bool opt_localsolvers = true;
    bool with_hcurl_smoothers = true;
    DivConstraintSolver * partsol_finder = new DivConstraintSolver(*mgtools_hierarchy, opt_localsolvers,
                                      with_hcurl_smoothers, verbose);

    // 7. Creating necessary ingredients for the multigrid and the multigrid preconditioner itself

    ComponentsDescriptor * divfree_descriptor;
    {
        bool with_Schwarz = false;
        bool optimized_Schwarz = false;
        bool with_Hcurl = false;
        bool with_coarsest_partfinder = false;
        bool with_coarsest_hcurl = false;
        bool with_monolithic_GS = true;
        bool with_nobnd_op = false;
        divfree_descriptor = new ComponentsDescriptor(with_Schwarz, optimized_Schwarz,
                                                      with_Hcurl, with_coarsest_partfinder,
                                                      with_coarsest_hcurl, with_monolithic_GS,
                                                      with_nobnd_op);
    }

    MultigridToolsHierarchy * mgtools_divfree_hierarchy =
            new MultigridToolsHierarchy(*hierarchy, divfree_problem->GetAttachedIndex(),
                                        *divfree_descriptor);

    // setting the coarsest level problem solver for the multigrid in divergence-free formulation
    int coarsest_level = nlevels - 1;
    CGSolver *CoarseSolver_mg = new CGSolver(comm);
    CoarseSolver_mg->SetAbsTol(sqrt(1e-32));
    CoarseSolver_mg->SetRelTol(sqrt(1e-12));
    CoarseSolver_mg->SetMaxIter(100);
    CoarseSolver_mg->SetPrintLevel(0);
    CoarseSolver_mg->SetOperator(*mgtools_divfree_hierarchy->GetOps()[coarsest_level]);
    CoarseSolver_mg->iterative_mode = false;

    BlockOperator * coarsest_op = mgtools_divfree_hierarchy->GetBlockOps()[coarsest_level];

    BlockDiagonalPreconditioner * CoarsePrec_mg =
            new BlockDiagonalPreconditioner(coarsest_op->ColOffsets());

    HypreParMatrix &blk00 = (HypreParMatrix&)coarsest_op->GetBlock(0,0);
    HypreSmoother * precU = new HypreSmoother(blk00, HypreSmoother::Type::l1GS, 1);
    CoarsePrec_mg->SetDiagonalBlock(0, precU);

    if (strcmp(space_for_S,"H1") == 0)
    {
        HypreParMatrix &blk11 = (HypreParMatrix&)coarsest_op->GetBlock(1,1);
        HypreSmoother * precS = new HypreSmoother(blk11, HypreSmoother::Type::l1GS, 1);

        CoarsePrec_mg->SetDiagonalBlock(1, precS);
    }
    CoarsePrec_mg->owns_blocks = true;
    CoarseSolver_mg->SetPreconditioner(*CoarsePrec_mg);

    Array<Operator*> casted_monolitGSSmoothers(nlevels - 1);
    for (int l = 0; l < nlevels - 1; ++l)
        casted_monolitGSSmoothers[l] = mgtools_divfree_hierarchy->GetMonolitGSSmoothers()[l];

    GeneralMultigrid * GeneralMGprec =
            new GeneralMultigrid(nlevels,
                                 mgtools_divfree_hierarchy->GetPs_bnd(),
                                 mgtools_divfree_hierarchy->GetOps(),
                                 *CoarseSolver_mg,
                                 casted_monolitGSSmoothers);

    chrono.Stop();
    if (verbose)
        std::cout << "Discrete divfree problem is ready \n";

    // 8. Finding the particular solution to the constraint
    FunctionCoefficient * rhs_coeff = problem->GetFEformulation().GetFormulation()->GetTest()->GetRhs();
    ParLinearForm * constrfform_new = new ParLinearForm(hierarchy->GetSpace(SpaceName::L2, 0));
    constrfform_new->AddDomainIntegrator(new DomainLFIntegrator(*rhs_coeff));
    constrfform_new->Assemble();
    Vector ConstrRhs(hierarchy->GetSpace(SpaceName::L2, 0)->TrueVSize());
    constrfform_new->ParallelAssemble(ConstrRhs);
    delete constrfform_new;

    BlockVector * Xinit_truedofs = problem->GetTrueInitialConditionFunc();

    BlockVector truePartSolFunc(problem->GetTrueOffsetsFunc());
    truePartSolFunc = 0.0;

    // finding the particular solution
    partsol_finder->FindParticularSolution(*Xinit_truedofs, truePartSolFunc, ConstrRhs, verbose);

    delete Xinit_truedofs;

    BlockVector truePartSol(problem->GetTrueOffsets());
    truePartSol = 0.0;
    for (int blk = 0; blk < problem->GetTrueOffsetsFunc().Size() - 1; ++blk)
        truePartSol.GetBlock(blk) = truePartSolFunc.GetBlock(blk);

    // 9. Creating the future solution and the rhs for the div-free problem
    // taking the particular solution into account
    BlockVector truetemp1(problem->GetTrueOffsets());
    problem->GetOp()->Mult(truePartSol, truetemp1);
    truetemp1 -= problem->GetRhs();
    truetemp1 *= -1;

    BlockVector trueRhs_divfree(divfree_problem->GetTrueOffsets());
    trueRhs_divfree = 0.0;
    divfree_problem->GetDivfreeHpMat().MultTranspose(truetemp1.GetBlock(0), trueRhs_divfree.GetBlock(0));
    if (strcmp(space_for_S,"H1") == 0)
        trueRhs_divfree.GetBlock(1) = truetemp1.GetBlock(1);

    BlockVector trueX_divfree(divfree_problem->GetTrueOffsets());
    trueX_divfree = 0.0;

    // 10. Solving system after creating and setting the solver

    CGSolver solver(comm);
    if (verbose)
        std::cout << "Linear solver: CG \n";

    solver.SetAbsTol(sqrt(atol));
    solver.SetRelTol(sqrt(rtol));
    solver.SetMaxIter(max_num_iter);
    solver.SetOperator(*divfree_problem->GetOp());
    solver.SetPreconditioner(*GeneralMGprec);
    solver.SetPrintLevel(1);

    // Solving the system
    chrono.Clear();
    chrono.Start();
    solver.Mult(trueRhs_divfree, trueX_divfree);
    chrono.Stop();

    if (verbose)
    {
        if (solver.GetConverged())
            std::cout << "Linear solver converged in " << solver.GetNumIterations()
                      << " iterations with a residual norm of " << solver.GetFinalNorm() << ".\n";
        else
            std::cout << "Linear solver did not converge in " << solver.GetNumIterations()
                      << " iterations. Residual norm is " << solver.GetFinalNorm() << ".\n";
        std::cout << "Linear solver took " << chrono.RealTime() << "s. \n";
    }

    // converting first component of the solution back into Hcurl
    BlockVector finalSol(problem->GetTrueOffsets());
    finalSol = 0.0;
    divfree_problem->GetDivfreeHpMat().Mult(trueX_divfree.GetBlock(0), finalSol.GetBlock(0));
    if (strcmp(space_for_S,"H1") == 0)
        finalSol.GetBlock(1) = trueX_divfree.GetBlock(1);
    // and adding the particular solution
    finalSol += truePartSol;

    // checking the error for the final solution
    bool checkbnd = true;
    problem->ComputeError(finalSol, verbose, checkbnd);

    if (verbose)
        std::cout << "Errors in the MG code were computed via FOSLSProblem routine \n";

    // 11. Deallocating the used memory.
    delete GeneralMGprec;

    delete CoarsePrec_mg;
    delete CoarseSolver_mg;

    delete partsol_finder;

    delete mgtools_divfree_hierarchy;
    delete divfree_descriptor;

    delete descriptor;
    delete mgtools_hierarchy;

    delete problem;
    delete divfree_problem;
    delete hierarchy;

    delete bdr_conds;
    delete formulat;
    delete fe_formulat;

    delete formulat_divfree;
    delete fe_formulat_divfree;

    MPI_Finalize();
    return 0;
}
