///                           MFEM(with 4D elements) CFOSLS for 3D/4D transport equation
///                       solved by geometric multigrid preconditioner in div-free setting
///                                   and also by a minimization solver.
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
/// The problem is then solved by two different multigrid setups:
/// 1) In the div-free formulation, where we first find a particular solution to the
/// divergence constraint, and then search for the div-free correction, casting the system's
/// first component into H(curl) (in 3D).
/// Then, to find the div-free correction, CG is used, preconditioned by a geometric multigrid.
/// 2) With a minimization solver, where we first find a particular solution to the
/// divergence constraint and then minimize the functional over the correction subspace.
/// Unlike 1), we don't cast the problem explicitly into H(curl), but iterations of the
/// minimization solver keep the vectors in the corresponding subspace where first block component
/// is from H(div) and satisfies the prescribed divergence constraint.

/// This example demonstrates usage of such classes from mfem/cfosls/ as
/// FOSLSProblem, FOSLSDivfreeProblem, GeneralAnisoHierarchy, GeneralMultirid,
/// MultigridToolsHierarchy and DivConstraintSolver.
///
/// (*) This code was tested in serial and in parallel.
/// (**) The example was tested for memory leaks with valgrind, in 3D.
///
/// Typical run of this example: ./cfosls_hyperbolic_multigrid --whichD 3 --spaceS L2 -no-vis
/// If you want to use the Hdiv-H1-L2 formulation, you will need not only change --spaceS option but also
/// change the source code.
///
/// Other examples with geometric multigrid are cfosls_hyperbolic_anisoMG.cpp (look there
/// for a cleaner example with geometric MG only, using the newest interfaces)
/// and cfosls_hyperbolic_newsolver (much more messy).

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>
#include <unistd.h>

// If active, constructs and uses MultigridToolsHierarchy, which is
// the latest and shortest way to create multigrid preconditioner
// Otherwise, multigrid components are constructe in a more straightforward way
#define USE_MULTIGRID_TOOLS

// if active, constructs a particular solution for the minimization solver
// via DivConstraintSolver instead of simply re-using the particular solution
// created for the geometric MG for the div-free formulation using DivPart
// (older class, not recommended for usage)
#define USE_DIVCONSTRAINT_SOLVER

// if active, constructs a particular solution for the multigrid in H(curl)
// (and for the minimization solver if USE_DIVCONSTRAINT_SOLVER is not #defined)
// via either older multilevel code or by simple solving the Poisson equation
// (the option's switch is with_multilevel = true/false flag in the code)
#define USE_OLD_PARTSOL

// activates using the local solvers (Schwarz smoothers) for the minimization solver
#define SOLVE_WITH_LOCALSOLVERS

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
    bool visualization = 1;

    int nDimensions     = 3;
    int numsol          = 4;

    int ser_ref_levels  = 1;
    int par_ref_levels  = 2;

    const char *space_for_S = "H1";    // "H1" or "L2"

    // for the older (legacy) part of the code
    // if this is true, DivPart is used for computing the particular solution.
    // else, a Poisson problem is solved for that purpose.
    bool with_multilevel = true;

    // defines for the old multilevel algorithm (with_multilevel = true)
    // whether to use mass matrix for the H(div) block or rather assume
    // that it's identity. In the latter case, the code works faster.
    // In any case, the resulting particular solution satisfies the constraint.
    bool useM_in_divpart = true;

    // solver options
    int prec_option = 1;        // defines whether to use preconditioner(0) or not(!0)
    int max_num_iter = 2000;
    double rtol = 1e-12;//1e-7;//1e-9;
    double atol = 1e-14;//1e-9;//1e-12;

    //const char *mesh_file = "../data/cube_3d_fine.mesh";
    const char *mesh_file = "../data/cube_3d_moderate.mesh";
    //const char *mesh_file = "../data/square_2d_moderate.mesh";

    //const char *mesh_file = "../data/cube4d_low.MFEM";
    //const char *mesh_file = "../data/cube4d_96.MFEM";
    //const char *mesh_file = "../data/orthotope3D_moderate.mesh";
    //const char * mesh_file = "../data/orthotope3D_fine.mesh";

    int feorder         = 0;

    if (verbose)
        cout << "Solving CFOSLS Transport equation, multigrid for the div-free approach, minimization solver \n";

    // 2. Parse command-line options.
    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&feorder, "-o", "--feorder",
                   "Finite element order (polynomial degree).");
    args.AddOption(&ser_ref_levels, "-sref", "--sref",
                   "Number of serial refinements 4d mesh.");
    args.AddOption(&par_ref_levels, "-pref", "--pref",
                   "Number of parallel refinements 4d mesh.");
    args.AddOption(&nDimensions, "-dim", "--whichD",
                   "Dimension of the space-time problem.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&prec_option, "-precopt", "--prec-option",
                   "Preconditioner choice.");
    args.AddOption(&with_multilevel, "-ml", "--multilvl", "-no-ml",
                   "--no-multilvl",
                   "Enable or disable multilevel algorithm for finding a particular solution.");
    args.AddOption(&useM_in_divpart, "-useM", "--useM", "-no-useM", "--no-useM",
                   "Whether to use M to compute a particular solution");
    args.AddOption(&space_for_S, "-spaceS", "--spaceS",
                   "Space for S: L2 or H1.");
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

#ifdef SOLVE_WITH_LOCALSOLVERS
    if (verbose)
        std::cout << "SOLVE_WITH_LOCALSOLVERS active \n";
#else
    if (verbose)
        std::cout << "SOLVE_WITH_LOCALSOLVERS passive \n";
#endif

#if !defined(USE_OLD_PARTSOL) && !defined(USE_DIVCONSTRAINT_SOLVER)
    { MFEM_ABORT("At least one of the options for finding the particular solution must be #defined");}
#endif

#if defined(USE_OLD_PARTSOL) && defined(USE_DIVCONSTRAINT_SOLVER)
    if (verbose)
        std::cout << "Warning: USE_DIVCONSTRAINT_SOLVER overrides USE_OLD_PARTSOL \n";
#endif

    std::cout << std::flush;
    MPI_Barrier(MPI_COMM_WORLD);

    MFEM_ASSERT(strcmp(space_for_S,"H1") == 0 || strcmp(space_for_S,"L2") == 0,
                "Space for S must be H1 or L2!\n");

    if (verbose)
    {
        if (strcmp(space_for_S,"H1") == 0)
            std::cout << "Space for S: H1 \n";
        else
            std::cout << "Space for S: L2 \n";

        if (strcmp(space_for_S,"L2") == 0)
            std::cout << "S is eliminated from the system \n";
    }

    if (nDimensions == 3)
    {
        numsol = -3;
        mesh_file = "../data/cube_3d_moderate.mesh";
    }
    else // 4D case
    {
        numsol = -4;
        mesh_file = "../data/cube4d_96.MFEM";
    }

    if (verbose)
        std::cout << "For the records: numsol = " << numsol
                  << ", mesh_file = " << mesh_file << "\n";

    if (verbose)
        cout << "Number of mpi processes: " << num_procs << endl << flush;

    bool with_prec = (prec_option != 0);

    StopWatch chrono;
    StopWatch chrono_total;

    chrono_total.Clear();
    chrono_total.Start();

    // 3. Reading the mesh and performing a prescribed number of serial and parallel
    // refinements which produce the initial coarse mesh for the multigrid
    Mesh *mesh = NULL;

    shared_ptr<ParMesh> pmesh;

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
        pmesh = make_shared<ParMesh>(comm, *mesh);
        delete mesh;
    }

    // 4. Define the problem to be solved (CFOSLS Hdiv-H1-L2 formulation or Hdiv-L2, e.g.)

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

    // 5. Constructing hierarchy of meshes and f.e. spaces, in two ways.
    // First, using GeneralHierarchy. Second, using the explicit way

    int dim = nDimensions;

    int ref_levels = par_ref_levels;

    int num_levels = ref_levels + 1;

    chrono.Clear();
    chrono.Start();

    int numblocks_funct = 1;
    if (strcmp(space_for_S,"H1") == 0) // S is present
        numblocks_funct++;

    // 5.1 Constructing the general hierarchy on top of the coarse mesh,
    // refine it, pmesh will be the finest mesh after that
    int nlevels = ref_levels + 1;
    GeneralHierarchy * hierarchy = new GeneralHierarchy(nlevels, *pmesh, 0, verbose);
    hierarchy->ConstructDivfreeDops();
    hierarchy->ConstructDofTrueDofs();
    hierarchy->ConstructEl2Dofs();

    pmesh->PrintInfo(std::cout); if(verbose) cout << "\n";

    // 5.2 Creating the FOSLS problem and it's div-free counterpart on top of the hierarchy

    // creating an instance of the problem at the finest mesh of the hierarchy
    FormulType * formulat = new FormulType (dim, numsol, verbose);
    FEFormulType* fe_formulat = new FEFormulType(*formulat, feorder);
    BdrConditions * bdr_conds = new BdrCondsType(*pmesh);

    FOSLSProblem* problem = hierarchy->BuildDynamicProblem<ProblemType>
            (*bdr_conds, *fe_formulat, prec_option, verbose);
    hierarchy->AttachProblem(problem);

    // creating an instance of the div-free problem at the finest mesh of the hierarchy
    DivfreeFormulType * formulat_divfree = new DivfreeFormulType (dim, numsol, verbose);
    DivfreeFEFormulType * fe_formulat_divfree = new DivfreeFEFormulType(*formulat_divfree, feorder);

    FOSLSDivfreeProblem* divfree_problem = hierarchy->BuildDynamicProblem<FOSLSDivfreeProblem>
            (*bdr_conds, *fe_formulat_divfree, prec_option, verbose);
    divfree_problem->ConstructDivfreeHpMats();
    divfree_problem->CreateOffsetsRhsSol();
    BlockOperator * divfree_problem_op = ConstructDivfreeProblemOp(*divfree_problem, *problem);
    divfree_problem->ResetOp(*divfree_problem_op, true);

    divfree_problem->InitSolver(verbose);
    // creating a preconditioner for the divfree problem
    divfree_problem->CreatePrec(*divfree_problem->GetOp(), prec_option, verbose);
    divfree_problem->ChangeSolver();
    divfree_problem->UpdateSolverPrec();

    hierarchy->AttachProblem(divfree_problem);

    // 5.3 creating components which will be used for multigrid and divconstraint solver
    // built on the original problem, if USE_MULTIGRID_TOOLS was #defined
#ifdef USE_MULTIGRID_TOOLS
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
#endif

    std::vector<Array<int>*>& essbdr_attribs = problem->GetBdrConditions().GetAllBdrAttribs();

    if (verbose)
    {
        std::cout << "Boundary conditions: \n";
        for (unsigned int i = 0; i < essbdr_attribs.size(); ++i)
        {
            std::cout << "component " << i << ": \n";
            essbdr_attribs[i]->Print(std::cout, pmesh->bdr_attributes.Max());
        }
    }

    // 6. Finding a particular solution to the divergence constraint
    chrono.Clear();
    chrono.Start();

    Vector Sigmahat_truedofs(problem->GetPfes(0)->TrueVSize());
    /// finding a particular solution via the old code: either multilevel or simply solving a Poisson problem
#ifdef USE_OLD_PARTSOL

    if (with_multilevel)
    {
        if (verbose)
            std::cout << "Using an old implementation of the multilevel algorithm to find"
                         " a particular solution \n";

        ConstantCoefficient k(1.0);

        SparseMatrix *M_local;
        if (useM_in_divpart)
        {
            ParBilinearForm Massform(hierarchy->GetSpace(SpaceName::HDIV, 0));
            Massform.AddDomainIntegrator(new VectorFEMassIntegrator(k));
            Massform.Assemble();
            Massform.Finalize();
            M_local = Massform.LoseMat();
        }
        else
            M_local = NULL;

        ParMixedBilinearForm DivForm(hierarchy->GetSpace(SpaceName::HDIV, 0),
                                     hierarchy->GetSpace(SpaceName::L2, 0));
        DivForm.AddDomainIntegrator(new VectorFEDivergenceIntegrator);
        DivForm.Assemble();
        DivForm.Finalize();
        SparseMatrix *B_local = DivForm.LoseMat();

        //Right hand size
        ParLinearForm gform(hierarchy->GetSpace(SpaceName::L2, 0));
        gform.AddDomainIntegrator(new DomainLFIntegrator(*problem->GetFEformulation().
                                                          GetFormulation()->GetTest()->GetRhs()));
        gform.Assemble();

        Vector G_fine(hierarchy->GetSpace(SpaceName::HDIV, 0)->GetVSize());

        G_fine = .0;

        Array< SparseMatrix*> el2dofs_R(ref_levels);
        Array< SparseMatrix*> el2dofs_W(ref_levels);
        Array< SparseMatrix*> P_Hdiv_lvls(ref_levels);
        Array< SparseMatrix*> P_L2_lvls(ref_levels);
        Array< SparseMatrix*> e_AE_lvls(ref_levels);

        for (int l = 0; l < ref_levels; ++l)
        {
            el2dofs_R[l] = hierarchy->GetElementToDofs(SpaceName::HDIV, l);
            el2dofs_W[l] = hierarchy->GetElementToDofs(SpaceName::L2, l);

            P_Hdiv_lvls[l] = hierarchy->GetPspace(SpaceName::HDIV, l);
            P_L2_lvls[l] = hierarchy->GetPspace(SpaceName::L2, l);
            e_AE_lvls[l] = P_L2_lvls[l];
        }

        const Array<int>* coarse_essbdr_dofs_Hdiv = hierarchy->GetEssBdrTdofsOrDofs
                ("dof", SpaceName::HDIV, *essbdr_attribs[0], num_levels - 1);

        DivPart divp;

        ParGridFunction sigmahat(problem->GetPfes(0));

        divp.div_part(ref_levels,
                      M_local, B_local,
                      G_fine,
                      gform,
                      P_L2_lvls, P_Hdiv_lvls, e_AE_lvls,
                      el2dofs_R,
                      el2dofs_W,
                      hierarchy->GetDofTrueDof(SpaceName::HDIV, num_levels - 1),
                      hierarchy->GetDofTrueDof(SpaceName::L2, num_levels - 1),
                      hierarchy->GetSpace(SpaceName::HDIV, num_levels - 1)->GetDofOffsets(),
                      hierarchy->GetSpace(SpaceName::L2, num_levels - 1)->GetDofOffsets(),
                      sigmahat,
                      *coarse_essbdr_dofs_Hdiv);

        sigmahat.ParallelProject(Sigmahat_truedofs);

        delete coarse_essbdr_dofs_Hdiv;

        delete M_local;
        delete B_local;
    }
    else
    {
        if (verbose)
            std::cout << "Solving Poisson problem for finding a particular solution \n";
        ParGridFunction *sigma_exact;
        HypreParMatrix *Bdiv;
        HypreParMatrix *BdivT;
        HypreParMatrix *BBT;

        sigma_exact = new ParGridFunction(hierarchy->GetSpace(SpaceName::HDIV, 0));
        sigma_exact->ProjectCoefficient(*problem->GetFEformulation().
                                        GetFormulation()->GetTest()->GetSigma());

        ParLinearForm gform(hierarchy->GetSpace(SpaceName::L2, 0));
        gform.AddDomainIntegrator(new DomainLFIntegrator(*problem->GetFEformulation().
                                                          GetFormulation()->GetTest()->GetRhs()));
        gform.Assemble();

        ParMixedBilinearForm Bblock(hierarchy->GetSpace(SpaceName::HDIV, 0),
                                    hierarchy->GetSpace(SpaceName::L2, 0));
        Bblock.AddDomainIntegrator(new VectorFEDivergenceIntegrator);
        Bblock.Assemble();
        Bblock.EliminateTrialDofs(*essbdr_attribs[0], *sigma_exact, gform);
        Bblock.Finalize();
        Bdiv = Bblock.ParallelAssemble();

        BdivT = Bdiv->Transpose();
        BBT = ParMult(Bdiv, BdivT);

        Vector * Rhs = gform.ParallelAssemble();

        HypreBoomerAMG * invBBT = new HypreBoomerAMG(*BBT);
        invBBT->SetPrintLevel(0);

        mfem::CGSolver solver(comm);
        solver.SetPrintLevel(0);
        solver.SetMaxIter(70000);
        solver.SetRelTol(1.0e-12);
        solver.SetAbsTol(1.0e-14);
        solver.SetPreconditioner(*invBBT);
        solver.SetOperator(*BBT);

        Vector tempsol(hierarchy->GetSpace(SpaceName::L2, 0)->TrueVSize());
        solver.Mult(*Rhs, tempsol);

        BdivT->Mult(tempsol, Sigmahat_truedofs);

        delete sigma_exact;
        delete invBBT;
        delete BBT;
        delete Rhs;
        delete Bdiv;
        delete BdivT;
    }
    // in either way now Sigmahat_truedofs correspond to a function from H(div) s.t. div Sigmahat = div sigma = f

    chrono.Stop();
    if (verbose)
        cout << "Particular solution found in " << chrono.RealTime() << " seconds.\n";

    if (verbose)
        std::cout << "Checking that particular solution in parallel version "
                     "satisfies the divergence constraint \n";

    if (!CheckConstrRes(Sigmahat_truedofs, problem->GetOp_nobnd()->GetBlock(numblocks_funct, 0),
                        &problem->GetRhs().GetBlock(numblocks_funct),
                        "in the old code for the particular solution"))
        std::cout << "Failure! \n";
    else
        if (verbose)
            std::cout << "Success \n";
#endif // for #ifdef USE_OLD_PARTSOL

    Array<int>& offsets_problem = problem->GetTrueOffsets();
    Array<int>& offsets_func = problem->GetTrueOffsetsFunc();

    BlockVector ParticSol(offsets_func);
    ParticSol = 0.0;

#ifdef USE_DIVCONSTRAINT_SOLVER
    if (verbose)
        std::cout << "Constructing a DivConstraintSolver to find the particular solution \n";
    chrono.Clear();
    chrono.Start();

    bool opt_localsolvers = true;
    bool with_hcurl_smoothers = true;

#ifndef USE_MULTIGRID_TOOLS
    // old, worked
    // in this way PartSolFinder would build and own its components
    DivConstraintSolver * PartsolFinder = new DivConstraintSolver(*problem, *hierarchy, opt_localsolvers,
                                      with_hcurl_smoothers, verbose);
#else
    // newer way to construct DivConstraintSolver, main components are borrowed from mgtools_hierarchy
    DivConstraintSolver * PartsolFinder = new DivConstraintSolver(*mgtools_hierarchy, opt_localsolvers,
                                      with_hcurl_smoothers, verbose);
#endif

    // Constructing the constraint rhs
    FunctionCoefficient * rhs_coeff = problem->GetFEformulation().GetFormulation()->GetTest()->GetRhs();
    ParLinearForm * constrfform_new = new ParLinearForm(hierarchy->GetSpace(SpaceName::L2, 0));
    constrfform_new->AddDomainIntegrator(new DomainLFIntegrator(*rhs_coeff));
    constrfform_new->Assemble();
    Vector ConstrRhs(hierarchy->GetSpace(SpaceName::L2, 0)->TrueVSize());
    constrfform_new->ParallelAssemble(ConstrRhs);
    delete constrfform_new;

    chrono.Stop();
    if (verbose)
        std::cout << "DivConstraintSolver was created in "<< chrono.RealTime() <<" seconds.\n";
    chrono.Clear();
    chrono.Start();

    BlockVector * Xinit_truedofs = problem->GetTrueInitialConditionFunc();
    PartsolFinder->FindParticularSolution(*Xinit_truedofs, ParticSol, ConstrRhs, verbose);

    chrono.Stop();

    if (verbose)
        std::cout << "Particular solution was found in " << chrono.RealTime() <<" seconds.\n";
    chrono.Clear();
    chrono.Start();

    // checking that the computed particular solution satisfies essential boundary conditions
    if (verbose)
        std::cout << "Checking that particular solution satisfied the boundary conditions \n";
    for (int blk = 0; blk < numblocks_funct; ++blk)
        problem->ComputeBndError(ParticSol, blk);

    if (verbose)
        std::cout << "Checking that particular solution in parallel version satisfies the divergence constraint \n";
    MFEM_ASSERT(CheckConstrRes(ParticSol.GetBlock(0), problem->GetOp_nobnd()->GetBlock(numblocks_funct, 0),
                               &problem->GetRhs().GetBlock(numblocks_funct),
                               "in the main code for the particular solution"), "");

    Sigmahat_truedofs = ParticSol.GetBlock(0);
#else
    ParticSol.GetBlock(0) = Sigmahat_truedofs;
#endif // for #ifdef USE_DIVCONSTRAINT_SOLVER

    if (verbose)
        std::cout << "End of setting up the problem in the divergence-free formulation \n";

    chrono.Clear();
    chrono.Start();

    // 7. Creating the necessary multigrid components and multigrid preconditioner for div-free formulation
#ifndef USE_MULTIGRID_TOOLS
    const Array<SpaceName>* space_names_divfree = divfree_problem->GetFEformulation().GetFormulation()
            ->GetFunctSpacesDescriptor();

    // constructing coarsest level essential boundary dofs indices in terms of
    // the global vector (for all components at once)
    std::vector< Array<int>* > coarsebnd_indces_divfree_lvls(num_levels);
    for (int l = 0; l < num_levels - 1; ++l)
    {
        std::vector<Array<int>* > essbdr_tdofs_hcurlfunct =
                hierarchy->GetEssBdrTdofsOrDofs("tdof", *space_names_divfree, essbdr_attribs, l + 1);

        int ncoarse_bndtdofs = 0;
        for (int blk = 0; blk < numblocks_funct; ++blk)
        {
            ncoarse_bndtdofs += essbdr_tdofs_hcurlfunct[blk]->Size();
        }

        coarsebnd_indces_divfree_lvls[l] = new Array<int>(ncoarse_bndtdofs);

        int shift_bnd_indices = 0;
        int shift_tdofs_indices = 0;
        for (int blk = 0; blk < numblocks_funct; ++blk)
        {
            for (int j = 0; j < essbdr_tdofs_hcurlfunct[blk]->Size(); ++j)
                (*coarsebnd_indces_divfree_lvls[l])[j + shift_bnd_indices] =
                    (*essbdr_tdofs_hcurlfunct[blk])[j] + shift_tdofs_indices;

            shift_bnd_indices += essbdr_tdofs_hcurlfunct[blk]->Size();
            shift_tdofs_indices += hierarchy->GetSpace((*space_names_divfree)[blk], l + 1)->TrueVSize();
        }

        for (unsigned int i = 0; i < essbdr_tdofs_hcurlfunct.size(); ++i)
            delete essbdr_tdofs_hcurlfunct[i];
    }

    Array<BlockOperator*> BlockP_mg_nobnd(nlevels - 1);
    Array<Operator*> P_mg(nlevels - 1);
    Array<BlockOperator*> BlockOps_mg(nlevels);
    Array<Operator*> Ops_mg(nlevels);
    Array<Operator*> Smoo_mg(nlevels - 1);
    Operator* CoarseSolver_mg;

    std::vector<const Array<int> *> divfree_offsets(nlevels);
    divfree_offsets[0] = hierarchy->ConstructTrueOffsetsforFormul(0, *space_names_divfree);

    for (int l = 0; l < num_levels; ++l)
    {
        if (l < num_levels - 1)
        {
            divfree_offsets[l + 1] = hierarchy->ConstructTrueOffsetsforFormul(l + 1, *space_names_divfree);

            BlockP_mg_nobnd[l] = hierarchy->ConstructTruePforFormul(l, *space_names_divfree,
                                                                    *divfree_offsets[l],
                                                                    *divfree_offsets[l + 1]);
            P_mg[l] = new BlkInterpolationWithBNDforTranspose(
                        *BlockP_mg_nobnd[l],
                        *coarsebnd_indces_divfree_lvls[l],
                        *divfree_offsets[l], *divfree_offsets[l + 1]);
        }

        if (l == 0)
            BlockOps_mg[l] = divfree_problem->GetOp();
        else
        {
            BlockOps_mg[l] = new RAPBlockHypreOperator(*BlockP_mg_nobnd[l - 1],
                    *BlockOps_mg[l - 1], *BlockP_mg_nobnd[l - 1], *divfree_offsets[l]);

            std::vector<Array<int>* > essbdr_tdofs_hcurlfunct =
                    hierarchy->GetEssBdrTdofsOrDofs("tdof", *space_names_divfree, essbdr_attribs, l);
            EliminateBoundaryBlocks(*BlockOps_mg[l], essbdr_tdofs_hcurlfunct);

            for (unsigned int i = 0; i < essbdr_tdofs_hcurlfunct.size(); ++i)
                delete essbdr_tdofs_hcurlfunct[i];
        }

        Ops_mg[l] = BlockOps_mg[l];

        if (l < num_levels - 1)
            Smoo_mg[l] = new MonolithicGSBlockSmoother( *BlockOps_mg[l], *divfree_offsets[l],
                                                        false, HypreSmoother::Type::l1GS, 1);
    }

    // setting the coarsest level problem solver for the multigrid in divergence-free formulation
    int coarsest_level = num_levels - 1;
    CoarseSolver_mg = new CGSolver(comm);
    ((CGSolver*)CoarseSolver_mg)->SetAbsTol(sqrt(1e-32));
    ((CGSolver*)CoarseSolver_mg)->SetRelTol(sqrt(1e-12));
    ((CGSolver*)CoarseSolver_mg)->SetMaxIter(100);
    ((CGSolver*)CoarseSolver_mg)->SetPrintLevel(0);
    ((CGSolver*)CoarseSolver_mg)->SetOperator(*Ops_mg[coarsest_level]);
    ((CGSolver*)CoarseSolver_mg)->iterative_mode = false;

    BlockDiagonalPreconditioner * CoarsePrec_mg =
            new BlockDiagonalPreconditioner(BlockOps_mg[coarsest_level]->ColOffsets());

    HypreParMatrix &blk00 = (HypreParMatrix&)BlockOps_mg[coarsest_level]->GetBlock(0,0);
    HypreSmoother * precU = new HypreSmoother(blk00, HypreSmoother::Type::l1GS, 1);
    ((BlockDiagonalPreconditioner*)CoarsePrec_mg)->SetDiagonalBlock(0, precU);

    if (strcmp(space_for_S,"H1") == 0)
    {
        HypreParMatrix &blk11 = (HypreParMatrix&)BlockOps_mg[coarsest_level]->GetBlock(1,1);

        HypreSmoother * precS = new HypreSmoother(blk11, HypreSmoother::Type::l1GS, 1);

        ((BlockDiagonalPreconditioner*)CoarsePrec_mg)->SetDiagonalBlock(1, precS);
    }
    CoarsePrec_mg->owns_blocks = true;

    ((CGSolver*)CoarseSolver_mg)->SetPreconditioner(*CoarsePrec_mg);

    // old interface
    GeneralMultigrid * GeneralMGprec =
            new GeneralMultigrid(nlevels, P_mg, Ops_mg, *CoarseSolver_mg, Smoo_mg);

#else // i.e., if USE_MULTIGRID_TOOLD was #defined
    // newer interface, using MultigridToolsHierarchy
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
            new MultigridToolsHierarchy(*hierarchy, 1, *divfree_descriptor);

    Array<Operator*> casted_monolitGSSmoothers(nlevels - 1);
    for (int l = 0; l < nlevels - 1; ++l)
        casted_monolitGSSmoothers[l] = mgtools_divfree_hierarchy->GetMonolitGSSmoothers()[l];

    int coarsest_level = num_levels - 1;
    CGSolver * CoarseSolver_mg = new CGSolver(comm);
    CoarseSolver_mg->SetAbsTol(sqrt(1e-32));
    CoarseSolver_mg->SetRelTol(sqrt(1e-12));
    CoarseSolver_mg->SetMaxIter(100);
    CoarseSolver_mg->SetPrintLevel(0);
    //CoarseSolver_mg->SetOperator(*Ops_mg[coarsest_level]);
    CoarseSolver_mg->SetOperator(*mgtools_divfree_hierarchy->GetOps()[coarsest_level]);
    CoarseSolver_mg->iterative_mode = false;

    BlockDiagonalPreconditioner * CoarsePrec_mg =
            new BlockDiagonalPreconditioner(mgtools_divfree_hierarchy->GetBlockOps()[coarsest_level]->ColOffsets());

    HypreParMatrix &blk00 = (HypreParMatrix&)mgtools_divfree_hierarchy->GetBlockOps()[coarsest_level]->GetBlock(0,0);
    HypreSmoother * precU = new HypreSmoother(blk00, HypreSmoother::Type::l1GS, 1);
    CoarsePrec_mg->SetDiagonalBlock(0, precU);

    if (strcmp(space_for_S,"H1") == 0)
    {
        HypreParMatrix &blk11 = (HypreParMatrix&)mgtools_divfree_hierarchy->GetBlockOps()[coarsest_level]->GetBlock(1,1);
        HypreSmoother * precS = new HypreSmoother(blk11, HypreSmoother::Type::l1GS, 1);
        CoarsePrec_mg->SetDiagonalBlock(1, precS);
    }
    CoarsePrec_mg->owns_blocks = true;

    CoarseSolver_mg->SetPreconditioner(*CoarsePrec_mg);

    GeneralMultigrid * GeneralMGprec =
            new GeneralMultigrid(nlevels,
                                 mgtools_divfree_hierarchy->GetPs_bnd(),
                                 mgtools_divfree_hierarchy->GetOps(),
                                 *CoarseSolver_mg,
                                 casted_monolitGSSmoothers);
#endif

    chrono.Stop();
    if (verbose)
        std::cout << "A multigrid preconditioner was created in " << chrono.RealTime() << " seconds.\n";

    chrono.Clear();
    chrono.Start();

    // 8. Preparing rhs for the div-free formulation (accounting for the particular solution)
    // and solving the div-free problem with the multigrid preconditioner

    BlockVector trueXhat(problem->GetTrueOffsets());
    trueXhat = 0.0;
    trueXhat.GetBlock(0) = Sigmahat_truedofs;

    BlockVector truetemp1(problem->GetTrueOffsets());
    problem->GetOp()->Mult(trueXhat, truetemp1);
    truetemp1 -= problem->GetRhs();
    truetemp1 *= -1;

    BlockVector trueRhs_divfree(divfree_problem->GetTrueOffsetsFunc());
    trueRhs_divfree = 0.0;
    divfree_problem->GetDivfreeHpMat().MultTranspose(truetemp1.GetBlock(0), trueRhs_divfree.GetBlock(0));

    if (strcmp(space_for_S,"H1") == 0)
        trueRhs_divfree.GetBlock(1) = truetemp1.GetBlock(1);

    chrono.Stop();
    if (verbose)
        std::cout << "Discrete divfree problem is ready \n";

    CGSolver solver(comm);
    if (verbose)
        std::cout << "Linear solver: CG \n";

    solver.SetAbsTol(sqrt(atol));
    solver.SetRelTol(sqrt(rtol));
    solver.SetMaxIter(max_num_iter);
    solver.SetOperator(*divfree_problem->GetOp());
    if (with_prec)
        solver.SetPreconditioner(*GeneralMGprec);
    solver.SetPrintLevel(1);

    BlockVector trueX(divfree_problem->GetTrueOffsets());
    trueX = 0.0;

    chrono.Clear();
    chrono.Start();
    solver.Mult(trueRhs_divfree, trueX);
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

    // casting the solution back to the original formulation and checking the error
    BlockVector finalSol(problem->GetTrueOffsets());
    finalSol = 0.0;
    divfree_problem->GetDivfreeHpMat().Mult(trueX.GetBlock(0), finalSol.GetBlock(0));
    if (strcmp(space_for_S,"H1") == 0)
        finalSol.GetBlock(1) = trueX.GetBlock(1);
    finalSol += trueXhat;

    bool checkbnd = true;
    problem->ComputeError(finalSol, verbose, checkbnd);

    if (verbose)
        std::cout << "Errors for the div-free formulation have been computed \n";

    chrono.Clear();
    chrono.Start();

    // 9. Creating necessary components and constructing a minimization solver (in the form of a special multigrid)
#ifndef USE_MULTIGRID_TOOLS
    // defining space names for the original problem, for the functional and the related divfree problem
    // just useful aliases
    const Array<SpaceName>* space_names_funct = problem->GetFEformulation().GetFormulation()
            ->GetFunctSpacesDescriptor();

    // extracting some boundary attributes
    const Array<int> &essbdr_attribs_Hcurl = problem->GetBdrConditions().GetBdrAttribs(0);

    std::vector<const Array<int> *> offsets_hdivh1(nlevels);
    offsets_hdivh1[0] = hierarchy->ConstructTrueOffsetsforFormul(0, *space_names_funct);

    std::vector<const Array<int> *> offsets_sp_hdivh1(nlevels);
    offsets_sp_hdivh1[0] = hierarchy->ConstructOffsetsforFormul(0, *space_names_funct);

    // setting multigrid components from the older parts of the code
    Array<BlockOperator*> BlockP_mg_nobnd_plus(nlevels - 1);
    Array<Operator*> P_mg_plus(nlevels - 1);
    Array<BlockOperator*> BlockOps_mg_plus(nlevels);
    Array<Operator*> Ops_mg_plus(nlevels);
    Array<Operator*> HcurlSmoothers_lvls(nlevels - 1);
    Array<Operator*> SchwarzSmoothers_lvls(nlevels - 1);
    Array<Operator*> Smoo_mg_plus(nlevels - 1);
    Operator* CoarseSolver_mg_plus;

    std::vector<Operator*> Ops_mg_special(nlevels - 1);

    std::vector< Array<int>* > coarsebnd_indces_funct_lvls(num_levels - 1); // num_lvls or num_lvls - 1 ?

    for (int l = 0; l < num_levels - 1; ++l)
    {
        std::vector<Array<int>* > essbdr_tdofs_funct =
                hierarchy->GetEssBdrTdofsOrDofs("tdof", *space_names_funct, essbdr_attribs, l + 1);

        int ncoarse_bndtdofs = 0;
        for (int blk = 0; blk < numblocks_funct; ++blk)
        {

            ncoarse_bndtdofs += essbdr_tdofs_funct[blk]->Size();
        }

        coarsebnd_indces_funct_lvls[l] = new Array<int>(ncoarse_bndtdofs);

        int shift_bnd_indices = 0;
        int shift_tdofs_indices = 0;

        for (int blk = 0; blk < numblocks_funct; ++blk)
        {
            for (int j = 0; j < essbdr_tdofs_funct[blk]->Size(); ++j)
                (*coarsebnd_indces_funct_lvls[l])[j + shift_bnd_indices] =
                    (*essbdr_tdofs_funct[blk])[j] + shift_tdofs_indices;

            shift_bnd_indices += essbdr_tdofs_funct[blk]->Size();
            shift_tdofs_indices += hierarchy->GetSpace((*space_names_funct)[blk], l + 1)->TrueVSize();
        }

        for (unsigned int i = 0; i < essbdr_tdofs_funct.size(); ++i)
            delete essbdr_tdofs_funct[i];
    }

    std::vector<Array<int>* > dtd_row_offsets(num_levels);
    std::vector<Array<int>* > dtd_col_offsets(num_levels);

    std::vector<Array<int>* > el2dofs_row_offsets(num_levels - 1);
    std::vector<Array<int>* > el2dofs_col_offsets(num_levels - 1);

    Array<SparseMatrix*> Constraint_mat_lvls_mg(num_levels);
    Array<BlockMatrix*> Funct_mat_lvls_mg(num_levels);
    Array<SparseMatrix*> AE_e_lvls(num_levels - 1);

    Array<BlockMatrix*> el2dofs_funct_lvls(num_levels - 1);
    std::deque<std::vector<HypreParMatrix*> > d_td_Funct_lvls(num_levels - 1);

    std::vector<Array<int>*> fullbdr_attribs(numblocks_funct);
    for (unsigned int i = 0; i < fullbdr_attribs.size(); ++i)
    {
        fullbdr_attribs[i] = new Array<int>(pmesh->bdr_attributes.Max());
        (*fullbdr_attribs[i]) = 1;
    }

    for (int l = 0; l < num_levels; ++l)
    {
        dtd_row_offsets[l] = new Array<int>();
        dtd_col_offsets[l] = new Array<int>();

        if (l < num_levels - 1)
        {
            offsets_hdivh1[l + 1] = hierarchy->ConstructTrueOffsetsforFormul(l + 1, *space_names_funct);
            BlockP_mg_nobnd_plus[l] = hierarchy->ConstructTruePforFormul
                    (l, *space_names_funct, *offsets_hdivh1[l], *offsets_hdivh1[l + 1]);
            P_mg_plus[l] = new BlkInterpolationWithBNDforTranspose
                    (*BlockP_mg_nobnd_plus[l], *coarsebnd_indces_funct_lvls[l], *offsets_hdivh1[l],
                     *offsets_hdivh1[l + 1]);
        }

        if (l == 0)
            //BlockOps_mg_plus[l] = funct_op;
            BlockOps_mg_plus[l] = problem->GetFunctOp(problem->GetTrueOffsetsFunc());
        else
        {
            BlockOps_mg_plus[l] = new RAPBlockHypreOperator(*BlockP_mg_nobnd_plus[l - 1],
                    *BlockOps_mg_plus[l - 1], *BlockP_mg_nobnd_plus[l - 1], *offsets_hdivh1[l]);

            std::vector<Array<int>* > essbdr_tdofs_funct =
                    hierarchy->GetEssBdrTdofsOrDofs("tdof", *space_names_funct, essbdr_attribs, l);
            EliminateBoundaryBlocks(*BlockOps_mg_plus[l], essbdr_tdofs_funct);

            for (unsigned int i = 0; i < essbdr_tdofs_funct.size(); ++i)
                delete essbdr_tdofs_funct[i];
        }

        Ops_mg_plus[l] = BlockOps_mg_plus[l];

        if (l == 0)
        {
            ParMixedBilinearForm *Divblock = new ParMixedBilinearForm(hierarchy->GetSpace(SpaceName::HDIV, 0),
                                                                    hierarchy->GetSpace(SpaceName::L2, 0));
            Divblock->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
            Divblock->Assemble();
            Divblock->Finalize();
            Constraint_mat_lvls_mg[0] = Divblock->LoseMat();
            delete Divblock;

            Funct_mat_lvls_mg[0] = problem->ConstructFunctBlkMat(*offsets_sp_hdivh1[0]);
        }
        else
        {
            offsets_sp_hdivh1[l] = hierarchy->ConstructOffsetsforFormul(l, *space_names_funct);

            Constraint_mat_lvls_mg[l] = RAP(*hierarchy->GetPspace(SpaceName::L2, l - 1),
                                            *Constraint_mat_lvls_mg[l - 1], *hierarchy->GetPspace(SpaceName::HDIV, l - 1));

            BlockMatrix * P_Funct = hierarchy->ConstructPforFormul(l - 1, *space_names_funct,
                                                                       *offsets_sp_hdivh1[l - 1], *offsets_sp_hdivh1[l]);
            Funct_mat_lvls_mg[l] = RAP(*P_Funct, *Funct_mat_lvls_mg[l - 1], *P_Funct);

            delete P_Funct;
        }

        if (l < num_levels - 1)
        {
            Array<int> SweepsNum(numblocks_funct);

            SweepsNum = ipow(1, l);
            if (verbose)
            {
                std::cout << "Sweeps num: \n";
                SweepsNum.Print();
            }

            std::vector<Array<int>*> essbdr_tdofs_funct = hierarchy->GetEssBdrTdofsOrDofs
                    ("tdof", *space_names_funct, essbdr_attribs, l);

            std::vector<Array<int>*> essbdr_dofs_funct = hierarchy->GetEssBdrTdofsOrDofs
                    ("dof", *space_names_funct, essbdr_attribs, l);

            std::vector<Array<int>*> fullbdr_dofs_funct = hierarchy->GetEssBdrTdofsOrDofs
                    ("dof", *space_names_funct, fullbdr_attribs, l);

            Array<int> * essbdr_hcurl = hierarchy->GetEssBdrTdofsOrDofs("tdof", SpaceName::HCURL,
                                                                       essbdr_attribs_Hcurl, l);

            HcurlSmoothers_lvls[l] = new HcurlGSSSmoother(*BlockOps_mg_plus[l],
                                                     *hierarchy->GetDivfreeDop(l),
                                                     *essbdr_hcurl,
                                                     essbdr_tdofs_funct,
                                                     &SweepsNum, *offsets_hdivh1[l]);

            int size = BlockOps_mg_plus[l]->Height();

            bool optimized_localsolve = true;

            el2dofs_row_offsets[l] = new Array<int>();
            el2dofs_col_offsets[l] = new Array<int>();

            el2dofs_funct_lvls[l] = hierarchy->GetElementToDofs(*space_names_funct, l, el2dofs_row_offsets[l],
                                                                el2dofs_col_offsets[l]);

            d_td_Funct_lvls[l] = hierarchy->GetDofTrueDof(*space_names_funct, l);

            AE_e_lvls[l] = Transpose(*hierarchy->GetPspace(SpaceName::L2, l));
            if (strcmp(space_for_S,"H1") == 0) // S is present
            {
                SchwarzSmoothers_lvls[l] = new LocalProblemSolverWithS
                        (size, *Funct_mat_lvls_mg[l], *Constraint_mat_lvls_mg[l],
                         d_td_Funct_lvls[l],
                         *AE_e_lvls[l],
                         *el2dofs_funct_lvls[l],
                         *hierarchy->GetElementToDofs(SpaceName::L2, l),
                         fullbdr_dofs_funct,
                         essbdr_dofs_funct,
                         optimized_localsolve);
            }
            else // no S
            {
                SchwarzSmoothers_lvls[l] = new LocalProblemSolver
                        (size, *Funct_mat_lvls_mg[l], *Constraint_mat_lvls_mg[l],
                         d_td_Funct_lvls[l],
                         *AE_e_lvls[l],
                         *el2dofs_funct_lvls[l],
                         *hierarchy->GetElementToDofs(SpaceName::L2, l),
                         fullbdr_dofs_funct,
                         essbdr_dofs_funct,
                         optimized_localsolve);
            }

            for (unsigned int i = 0; i < essbdr_tdofs_funct.size(); ++i)
                delete essbdr_tdofs_funct[i];

            for (unsigned int i = 0; i < fullbdr_dofs_funct.size(); ++i)
                delete fullbdr_dofs_funct[i];

            for (unsigned int i = 0; i < essbdr_dofs_funct.size(); ++i)
                delete essbdr_dofs_funct[i];

            delete essbdr_hcurl;

#ifdef SOLVE_WITH_LOCALSOLVERS
            Smoo_mg_plus[l] = new SmootherSum(*SchwarzSmoothers_lvls[l], *HcurlSmoothers_lvls[l], *Ops_mg_plus[l]);
#else
            Smoo_mg_plus[l] = HcurlSmoothers_lvls[l];
#endif
        } // end of if l < num_levels - 1
    }

    for (unsigned int i = 0; i < fullbdr_attribs.size(); ++i)
        delete fullbdr_attribs[i];

    for (int l = 0; l < nlevels - 1; ++l)
        Ops_mg_special[l] = Ops_mg_plus[l];

    if (verbose)
        std::cout << "Creating the new coarsest solver which works in the div-free subspace \n" << std::flush;

    std::vector<Array<int> * > essbdr_tdofs_funct_coarse1 =
            hierarchy->GetEssBdrTdofsOrDofs("tdof", *space_names_funct, essbdr_attribs, num_levels - 1);

    Array<int> * essbdr_hcurl_coarse = hierarchy->GetEssBdrTdofsOrDofs("tdof", SpaceName::HCURL,
                                                                       essbdr_attribs_Hcurl, num_levels - 1);

    CoarseSolver_mg_plus = new CoarsestProblemHcurlSolver(Ops_mg_plus[num_levels - 1]->Height(),
                                                     *BlockOps_mg_plus[num_levels - 1],
                                                     *hierarchy->GetDivfreeDop(num_levels - 1),
                                                     essbdr_tdofs_funct_coarse1,
                                                     *essbdr_hcurl_coarse);

    for (unsigned int i = 0; i < essbdr_tdofs_funct_coarse1.size(); ++i)
        delete essbdr_tdofs_funct_coarse1[i];
    delete essbdr_hcurl_coarse;

    ((CoarsestProblemHcurlSolver*)CoarseSolver_mg_plus)->SetMaxIter(100);
    ((CoarsestProblemHcurlSolver*)CoarseSolver_mg_plus)->SetAbsTol(sqrt(1.0e-32));
    ((CoarsestProblemHcurlSolver*)CoarseSolver_mg_plus)->SetRelTol(sqrt(1.0e-12));
    ((CoarsestProblemHcurlSolver*)CoarseSolver_mg_plus)->ResetSolverParams();

    // old, working interface
    GeneralMultigrid * GeneralMGprec_plus =
            new GeneralMultigrid(nlevels, P_mg_plus, Ops_mg_plus, *CoarseSolver_mg_plus, Smoo_mg_plus);

#else // i.e., if USE_MULTIGRID_TOOLS was #defined

    Array<Operator*> Smoo_ops(nlevels - 1);
    for (int i = 0; i < Smoo_ops.Size(); ++i)
    {
#ifdef SOLVE_WITH_LOCALSOLVERS
        Smoo_ops[i] = mgtools_hierarchy->GetCombinedSmoothers()[i];
#else
        Smoo_ops[i] = mgtools_hierarchy->GetHcurlSmoothers()[i];
#endif
    }

    // newer interface using MultigridTools
    GeneralMultigrid * GeneralMGprec_plus =
            new GeneralMultigrid(nlevels, mgtools_hierarchy->GetPs_bnd(), mgtools_hierarchy->GetOps(),
                                 *mgtools_hierarchy->GetCoarsestSolver_Hcurl(),
                                 Smoo_ops);
#endif

    // 10. Preparing rhs (accounting for the particular solution) and solving the original problem with
    // a minimization solver (which works in the subspace of functions which satisfy the divergence constraint)

    BlockVector trueXtest(offsets_func);
    trueXtest = 0.0;

    BlockVector trueRhstest(offsets_func);
    for (int blk = 0; blk < numblocks_funct; ++blk)
        trueRhstest.GetBlock(blk) = problem->GetRhs().GetBlock(blk);

    BlockVector trueRhstest_funct(offsets_func);
    trueRhstest_funct = trueRhstest;

    // trueRhstest = F - Funct * particular solution (= residual), on true dofs
    BlockVector truevec(offsets_problem);
    truevec = 0.0;
    for (int blk = 0; blk < numblocks_funct; ++blk)
        truevec.GetBlock(blk) = ParticSol.GetBlock(blk);

    BlockVector truetemp(offsets_problem);
    problem->GetOp()->Mult(truevec, truetemp);
    truetemp.GetBlock(numblocks_funct) = 0.0;

    for (int blk = 0; blk < numblocks_funct; ++blk)
        trueRhstest.GetBlock(blk) -= truetemp.GetBlock(blk);

    // Creating the CG solver
    int TestmaxIter(400);

    CGSolver Testsolver(MPI_COMM_WORLD);
    Testsolver.SetAbsTol(sqrt(atol));
    Testsolver.SetRelTol(sqrt(rtol));
    Testsolver.SetMaxIter(TestmaxIter);

#ifndef USE_MULTIGRID_TOOLS
    Testsolver.SetOperator(*BlockOps_mg_plus[0]);
#else
    Testsolver.SetOperator(*mgtools_hierarchy->GetOps()[0]);
#endif

    Testsolver.SetPreconditioner(*GeneralMGprec_plus);

    Testsolver.SetPrintLevel(1);

    chrono.Stop();
    if (verbose)
        std::cout << "Solving the system \n";
    chrono.Clear();
    chrono.Start();

    Testsolver.Mult(trueRhstest, trueXtest);

    chrono.Stop();

    if (verbose)
    {
        if (Testsolver.GetConverged())
            std::cout << "Linear solver converged in " << Testsolver.GetNumIterations()
                      << " iterations with a residual norm of " << Testsolver.GetFinalNorm() << ".\n";
        else
            std::cout << "Linear solver did not converge in " << Testsolver.GetNumIterations()
                      << " iterations. Residual norm is " << Testsolver.GetFinalNorm() << ".\n";
        std::cout << "Linear solver (CG + new solver) took " << chrono.RealTime() << "s. \n";
        std::cout << "Global system size = " << "... not computed \n";
    }

    chrono.Clear();
    chrono.Start();

    // Adding back the particular solution
    trueXtest += ParticSol;

    BlockVector tempvec(offsets_problem);
    tempvec = 0.0;
    for (int blk = 0; blk < numblocks_funct; ++blk)
        tempvec.GetBlock(blk) = trueXtest.GetBlock(blk);

    problem->ComputeError(tempvec, verbose, checkbnd);

    chrono.Stop();
    if (verbose)
        std::cout << "Errors for the minimization solver were computed in " << chrono.RealTime() <<" seconds.\n";

    chrono_total.Stop();
    if (verbose)
        std::cout << "Total time consumed was " << chrono_total.RealTime() <<" seconds.\n";

    // 11. Deallocating used memory.

#ifdef USE_DIVCONSTRAINT_SOLVER
    delete Xinit_truedofs;
#endif

    delete hierarchy;
    delete problem;
    delete divfree_problem;

    delete PartsolFinder;

#ifndef USE_MULTIGRID_TOOLS
    for (unsigned int i = 0; i < coarsebnd_indces_divfree_lvls.size(); ++i)
        delete coarsebnd_indces_divfree_lvls[i];

    for (int i = 0; i < P_mg.Size(); ++i)
        delete P_mg[i];

    for (int i = 0; i < BlockP_mg_nobnd.Size(); ++i)
        delete BlockP_mg_nobnd[i];

    // BlockOps_mg[0] is the operator of divfree_problem, already deleted
    for (int i = 1; i < BlockOps_mg.Size(); ++i)
        delete BlockOps_mg[i];

    for (int i = 0; i < Smoo_mg.Size(); ++i)
        delete Smoo_mg[i];

    for (unsigned int i = 0; i < divfree_offsets.size(); ++i)
        delete divfree_offsets[i];

    for (unsigned int i = 0; i < coarsebnd_indces_funct_lvls.size(); ++i)
        delete coarsebnd_indces_funct_lvls[i];

    for (int i = 0; i < Smoo_mg_plus.Size(); ++i)
        delete Smoo_mg_plus[i];

    for (int i = 0; i < P_mg_plus.Size(); ++i)
        delete P_mg_plus[i];

    for (int i = 0; i < BlockP_mg_nobnd_plus.Size(); ++i)
        delete BlockP_mg_nobnd_plus[i];

    for (int i = 0; i < Funct_mat_lvls_mg.Size(); ++i)
        delete Funct_mat_lvls_mg[i];

    for (int i = 0; i < Constraint_mat_lvls_mg.Size(); ++i)
        delete Constraint_mat_lvls_mg[i];

    for (int i = 0; i < HcurlSmoothers_lvls.Size(); ++i)
        delete HcurlSmoothers_lvls[i];

    for (int i = 0; i < SchwarzSmoothers_lvls.Size(); ++i)
        delete SchwarzSmoothers_lvls[i];

    for (int i = 0; i < AE_e_lvls.Size(); ++i)
        delete AE_e_lvls[i];

    for (int i = 0; i < el2dofs_funct_lvls.Size(); ++i)
        delete el2dofs_funct_lvls[i];

    delete CoarseSolver_mg_plus;

    for (unsigned int i = 0; i < dtd_row_offsets.size(); ++i)
        delete dtd_row_offsets[i];

    for (unsigned int i = 0; i < dtd_col_offsets.size(); ++i)
        delete dtd_col_offsets[i];

    for (unsigned int i = 0; i < el2dofs_row_offsets.size(); ++i)
        delete el2dofs_row_offsets[i];

    for (unsigned int i = 0; i < el2dofs_col_offsets.size(); ++i)
        delete el2dofs_col_offsets[i];

    for (unsigned int i = 0; i < offsets_hdivh1.size(); ++i)
        delete offsets_hdivh1[i];

    for (unsigned int i = 0; i < offsets_sp_hdivh1.size(); ++i)
        delete offsets_sp_hdivh1[i];

    for (int i = 0; i < BlockOps_mg_plus.Size(); ++i)
        delete BlockOps_mg_plus[i];
#else
    delete descriptor;
    delete mgtools_hierarchy;

    delete divfree_descriptor;
    delete mgtools_divfree_hierarchy;
#endif

    delete CoarseSolver_mg;
    delete CoarsePrec_mg;

    delete GeneralMGprec;
    delete GeneralMGprec_plus;

    delete bdr_conds;
    delete formulat;
    delete fe_formulat;

    delete formulat_divfree;
    delete fe_formulat_divfree;

    MPI_Finalize();
    return 0;
}

