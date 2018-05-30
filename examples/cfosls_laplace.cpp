//
//                        MFEM CFOSLS Poisson equation
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>
#include <unistd.h>

#define MYZEROTOL (1.0e-13)

using namespace std;
using namespace mfem;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

int main(int argc, char *argv[])
{
    int num_procs, myid;
    bool visualization = 0;

    // 1. Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    bool verbose = (myid == 0);

    int nDimensions     = 3;
    int numsol          = 4;

    int ser_ref_levels  = 1;
    int par_ref_levels  = 2;

    const char *space_for_S = "H1";    // "H1" or "L2"

    bool aniso_refine = false;
    bool refine_t_first = false;

    // solver options
    int prec_option = 1;        // defines whether to use preconditioner or not, and which one

    //const char *mesh_file = "../data/cube_3d_fine.mesh";
    const char *mesh_file = "../data/cube_3d_moderate.mesh";
    //const char *mesh_file = "../data/square_2d_moderate.mesh";

    //const char *mesh_file = "../data/cube4d_low.MFEM";
    //const char *mesh_file = "../data/cube4d_96.MFEM";
    //const char *mesh_file = "dsadsad";
    //const char *mesh_file = "../data/orthotope3D_moderate.mesh";
    //const char * mesh_file = "../data/orthotope3D_fine.mesh";

    int feorder         = 0;

    if (verbose)
        cout << "Solving CFOSLS Poisson equation with MFEM & hypre \n";

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
    args.AddOption(&aniso_refine, "-aniso", "--aniso-refine", "-iso",
                   "--iso-refine",
                   "Using anisotropic or isotropic refinement.");
    args.AddOption(&refine_t_first, "-refine-t-first", "--refine-time-first",
                   "-refine-x-first", "--refine-space-first",
                   "Refine time or space first in anisotropic refinement.");
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

    MFEM_ASSERT(strcmp(space_for_S,"H1") == 0, "Space for S must be H1 for the laplace equation!\n");

    if (verbose)
        std::cout << "Space for S: H1 \n";

    if (verbose)
        std::cout << "Running tests for the paper: \n";

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

    Laplace_test Mytest(nDimensions, numsol);

    ConstantCoefficient zerocoeff(0.0);
    Vector zerovec(nDimensions);
    zerovec = 0.0;
    VectorConstantCoefficient zerocoeff_vec(zerovec);

    if (verbose)
        cout << "Number of mpi processes: " << num_procs << endl << flush;

    bool with_prec;

    switch (prec_option)
    {
    case 1: // smth simple like AMS
        with_prec = true;
        break;
    default: // no preconditioner (default)
        with_prec = false;
        break;
    }

    if (verbose)
    {
        cout << "with_prec = " << with_prec << endl;
        cout << flush;
    }

    StopWatch chrono;
    StopWatch chrono_total;

    chrono_total.Clear();
    chrono_total.Start();

    //DEFAULTED LINEAR SOLVER OPTIONS
    int max_num_iter = 150000;
    double rtol = 1e-12;//1e-7;//1e-9;
    double atol = 1e-14;//1e-9;//1e-12;

    Mesh *mesh = NULL;

    shared_ptr<ParMesh> pmesh;

    if (nDimensions == 3 || nDimensions == 4)
    {
        if (aniso_refine)
        {
            if (verbose)
                std::cout << "Anisotropic refinement is ON \n";
            if (nDimensions == 3)
            {
                if (verbose)
                    std::cout << "Using hexahedral mesh in 3D for anisotr. refinement code \n";
                mesh = new Mesh(2, 2, 2, Element::HEXAHEDRON, 1);
            }
            else // dim == 4
            {
                if (verbose)
                    cerr << "Anisotr. refinement is not implemented in 4D case with tesseracts \n" << std::flush;
                MPI_Finalize();
                return -1;
            }
        }
        else // no anisotropic refinement
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
        if (aniso_refine)
        {
            // for anisotropic refinement, the serial mesh needs at least one
            // serial refine to turn the mesh into a nonconforming mesh
            MFEM_ASSERT(ser_ref_levels > 0, "need ser_ref_levels > 0 for aniso_refine");

            for (int l = 0; l < ser_ref_levels-1; l++)
                mesh->UniformRefinement();

            Array<Refinement> refs(mesh->GetNE());
            for (int i = 0; i < mesh->GetNE(); i++)
            {
                refs[i] = Refinement(i, 7);
            }
            mesh->GeneralRefinement(refs, -1, -1);

            par_ref_levels *= 2;
        }
        else
        {
            for (int l = 0; l < ser_ref_levels; l++)
                mesh->UniformRefinement();
        }

        if (verbose)
            cout << "Creating parmesh(" << nDimensions <<
                    "d) from the serial mesh (" << nDimensions << "d)" << endl << flush;
        pmesh = make_shared<ParMesh>(comm, *mesh);
        delete mesh;
    }

    for (int l = 0; l < par_ref_levels; l++)
    {
       pmesh->UniformRefinement();
    }

    pmesh->PrintInfo(std::cout); if(verbose) cout << endl;

    int dim = nDimensions;

    using FormulType = CFOSLSFormulation_Laplace;
    using FEFormulType = CFOSLSFEFormulation_Laplace;
    using BdrCondsType = BdrConditions_CFOSLS_Laplace;
    using ProblemType = FOSLSProblem_lapl;

    FOSLSFormulation * formulat = new FormulType (dim, numsol, verbose);
    FOSLSFEFormulation * fe_formulat = new FEFormulType(*formulat, feorder);
    BdrConditions * bdr_conds = new BdrCondsType(*pmesh);

    ProblemType * problem = new ProblemType
            (*pmesh, *bdr_conds, *fe_formulat, prec_option, verbose);

    bool checkbnd = true;
    if (verbose)
        std::cout << "Solving the problem using the new interfaces \n";
    problem->Solve(verbose, checkbnd);

    if (verbose)
        std::cout << "Now proceeding with the older way which involves more explicit problem construction\n";

    Array<int> ess_bdrSigma(pmesh->bdr_attributes.Max());
    ess_bdrSigma = 0;

    Array<int> ess_bdrS(pmesh->bdr_attributes.Max());
    ess_bdrS = 1;

    Array<int> all_bdrSigma(pmesh->bdr_attributes.Max());
    all_bdrSigma = 1;

    Array<int> all_bdrS(pmesh->bdr_attributes.Max());
    all_bdrS = 1;

    FiniteElementCollection *hdiv_coll;
    ParFiniteElementSpace *R_space;
    FiniteElementCollection *l2_coll;
    ParFiniteElementSpace *W_space;

    if (dim == 4)
        hdiv_coll = new RT0_4DFECollection;
    else
        hdiv_coll = new RT_FECollection(feorder, dim);

    R_space = new ParFiniteElementSpace(pmesh.get(), hdiv_coll);

    l2_coll = new L2_FECollection(feorder, nDimensions);
    W_space = new ParFiniteElementSpace(pmesh.get(), l2_coll);

    FiniteElementCollection *hdivfree_coll;
    ParFiniteElementSpace *C_space;

    if (dim == 3)
        hdivfree_coll = new ND_FECollection(feorder + 1, nDimensions);
    else // dim == 4
        hdivfree_coll = new DivSkew1_4DFECollection;

    C_space = new ParFiniteElementSpace(pmesh.get(), hdivfree_coll);

    FiniteElementCollection *h1_coll;
    ParFiniteElementSpace *H_space;
    if (dim == 3)
        h1_coll = new H1_FECollection(feorder+1, nDimensions);
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
    H_space = new ParFiniteElementSpace(pmesh.get(), h1_coll);

    ParFiniteElementSpace * S_space;
    if (strcmp(space_for_S,"H1") == 0)
        S_space = H_space;
    else // "L2"
        S_space = W_space;

    ParGridFunction * sigma_exact_finest;
    sigma_exact_finest = new ParGridFunction(R_space);
    sigma_exact_finest->ProjectCoefficient(*Mytest.GetSigma());
    Vector sigma_exact_truedofs(R_space->GetTrueVSize());
    sigma_exact_finest->ParallelProject(sigma_exact_truedofs);

    ParGridFunction * S_exact_finest;
    Vector S_exact_truedofs;
    S_exact_finest = new ParGridFunction(S_space);
    S_exact_finest->ProjectCoefficient(*Mytest.GetU());
    S_exact_truedofs.SetSize(S_space->GetTrueVSize());
    S_exact_finest->ParallelProject(S_exact_truedofs);


    chrono.Clear();
    chrono.Start();

    int numblocks = 2;

    Array<int> block_offsets(numblocks + 2); // number of variables + 1
    block_offsets[0] = 0;
    block_offsets[1] = R_space->GetVSize();
    block_offsets[2] = S_space->GetVSize();
    block_offsets[3] = W_space->GetVSize();
    block_offsets.PartialSum();

    Array<int> block_trueOffsets(numblocks + 2); // number of variables + 1
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = R_space->TrueVSize();
    block_trueOffsets[2] = S_space->TrueVSize();
    block_trueOffsets[3] = W_space->TrueVSize();
    block_trueOffsets.PartialSum();

    HYPRE_Int dimR = R_space->GlobalTrueVSize();
    HYPRE_Int dimS = S_space->GlobalTrueVSize();
    HYPRE_Int dimW = W_space->GlobalTrueVSize();
    if (verbose)
    {
        std::cout << "***********************************************************\n";
        std::cout << "dim(R) = " << dimR << ", ";
        std::cout << "dim(S) = " << dimS << ", ";
        std::cout << "dim(W) = " << dimW << "\n";
        std::cout << "neqns in the funct = " << dimR + dimS << "\n";
        std::cout << "neqns in the constr = " << dimW << "\n";
        std::cout << "***********************************************************\n";
    }

    BlockVector xblks(block_offsets), rhsblks(block_offsets);
    BlockVector trueX(block_trueOffsets), trueRhs(block_trueOffsets);
    xblks = 0.0;
    rhsblks = 0.0;
    trueX = 0.0;
    trueRhs = 0.0;

    //----------------------------------------------------------
    // Setting boundary conditions.
    //----------------------------------------------------------

    if (verbose)
    {
        std::cout << "Boundary conditions: \n";
        std::cout << "all bdr Sigma: \n";
        all_bdrSigma.Print(std::cout, pmesh->bdr_attributes.Max());
        std::cout << "ess bdr Sigma: \n";
        ess_bdrSigma.Print(std::cout, pmesh->bdr_attributes.Max());
        std::cout << "ess bdr S: \n";
        ess_bdrS.Print(std::cout, pmesh->bdr_attributes.Max());
    }

    chrono.Stop();
    if (verbose)
        std::cout << "Small things were done in "<< chrono.RealTime() <<" seconds.\n";
    chrono.Clear();
    chrono.Start();

    ParGridFunction *S_exact;
    S_exact = new ParGridFunction(S_space);
    S_exact->ProjectCoefficient(*Mytest.GetU());

    ParGridFunction * sigma_exact = new ParGridFunction(R_space);
    sigma_exact->ProjectCoefficient(*Mytest.GetSigma());

    xblks.GetBlock(0) = *sigma_exact;
    xblks.GetBlock(1) = *S_exact;

    ParLinearForm * Constrrhsform = new ParLinearForm(W_space);
    Constrrhsform->AddDomainIntegrator(new DomainLFIntegrator(*Mytest.GetRhs()));
    Constrrhsform->Assemble();

    ParLinearForm * Sigmarhsform = new ParLinearForm(R_space);
    Sigmarhsform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(zerocoeff_vec));
    Sigmarhsform->Assemble();

    ParLinearForm * Srhsform = new ParLinearForm(S_space);
    Srhsform->AddDomainIntegrator(new DomainLFIntegrator(zerocoeff));
    Srhsform->Assemble();

    // mass matrix for H(div)
    ParBilinearForm *Mblock(new ParBilinearForm(R_space));
    Mblock->AddDomainIntegrator(new VectorFEMassIntegrator);
    Mblock->Assemble();
    Mblock->EliminateEssentialBC(ess_bdrSigma, *sigma_exact, *Sigmarhsform);
    Mblock->Finalize();
    HypreParMatrix *M = Mblock->ParallelAssemble();

    HypreParMatrix *C, *B, *BT;
    // diagonal block for H^1
    ParBilinearForm *Cblock = new ParBilinearForm(S_space);
    Cblock->AddDomainIntegrator(new DiffusionIntegrator);
    Cblock->Assemble();
    Cblock->EliminateEssentialBC(ess_bdrS, xblks.GetBlock(1),*Srhsform);
    Cblock->Finalize();
    C = Cblock->ParallelAssemble();

    // off-diagonal block for (H(div), Space_for_S) block
    ParMixedBilinearForm *Bblock(new ParMixedBilinearForm(S_space, R_space));
    Bblock->AddDomainIntegrator(new MixedVectorGradientIntegrator);
    Bblock->Assemble();
    Bblock->EliminateTrialDofs(ess_bdrS, *S_exact, *Sigmarhsform);
    Bblock->EliminateTestDofs(ess_bdrSigma);
    Bblock->Finalize();
    B = Bblock->ParallelAssemble();
    BT = B->Transpose();

    HypreParMatrix * Constr, * ConstrT;
    {
        ParMixedBilinearForm *Dblock = new ParMixedBilinearForm(R_space, W_space);
        Dblock->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
        Dblock->Assemble();
        Dblock->EliminateTrialDofs(ess_bdrSigma, xblks.GetBlock(0), *Sigmarhsform); // new
        Dblock->Finalize();
        Constr = Dblock->ParallelAssemble();
        ConstrT = Constr->Transpose();
    }

    Sigmarhsform->ParallelAssemble(trueRhs.GetBlock(0));
    Srhsform->ParallelAssemble(trueRhs.GetBlock(1));
    Constrrhsform->ParallelAssemble(trueRhs.GetBlock(2));

    BlockOperator *MainOp = new BlockOperator(block_trueOffsets);

    // setting block operator of the system
    MainOp->SetBlock(0,0, M);
    if (strcmp(space_for_S,"H1") == 0) // S is present
    {
        MainOp->SetBlock(0,1, B);
        MainOp->SetBlock(1,0, BT);
        MainOp->SetBlock(1,1, C);
    }
    MainOp->SetBlock(0,2, ConstrT);
    MainOp->SetBlock(2,0, Constr);

    // testing
    Array<int> blockfunct_trueOffsets(numblocks + 1);
    blockfunct_trueOffsets[0] = 0;
    blockfunct_trueOffsets[1] = M->Height();
    blockfunct_trueOffsets[2] = C->Height();
    blockfunct_trueOffsets.PartialSum();

    BlockOperator *MainOpFunct = new BlockOperator(blockfunct_trueOffsets);

    // setting block operator of the system
    MainOpFunct->SetBlock(0,0, M);
    MainOpFunct->SetBlock(0,1, B);
    MainOpFunct->SetBlock(1,0, BT);
    MainOpFunct->SetBlock(1,1, C);

    BlockVector truesol(blockfunct_trueOffsets);
    truesol.GetBlock(0) = sigma_exact_truedofs;
    truesol.GetBlock(1) = S_exact_truedofs;

    BlockVector rhsfunct(blockfunct_trueOffsets);
    rhsfunct.GetBlock(0) = trueRhs.GetBlock(0);
    rhsfunct.GetBlock(1) = trueRhs.GetBlock(1);

    BlockVector resfunct(blockfunct_trueOffsets);
    MainOpFunct->Mult(truesol, resfunct);
    resfunct -= rhsfunct;

    double res_funct_norm = resfunct.Norml2() / sqrt (resfunct.Size());

    if (verbose)
        std::cout << "res_funct_norm = " << res_funct_norm << "\n";


    // checking the residual for the projections of exact solution for Hdiv equation
    Vector Msigma(M->Height());
    M->Mult(sigma_exact_truedofs, Msigma);
    Vector BS(B->Height());
    B->Mult(S_exact_truedofs, BS);

    Vector res_Hdiv(M->Height());
    res_Hdiv = Msigma;
    res_Hdiv += BS;

    double res_Hdiv_norm = res_Hdiv.Norml2() / sqrt (res_Hdiv.Size());

    if (verbose)
        std::cout << "res_Hdiv_norm = " << res_Hdiv_norm << "\n";

    // checking the residual for the projections of exact solution for H1 equation
    Vector CS(C->Height());
    C->Mult(S_exact_truedofs, CS);
    Vector BTsigma(BT->Height());
    BT->Mult(sigma_exact_truedofs, BTsigma);

    Vector res_H1(C->Height());
    res_H1 = CS;
    res_H1 += BTsigma;

    double res_H1_norm = res_H1.Norml2() / sqrt (res_H1.Size());

    if (verbose)
        std::cout << "res_H1_norm = " << res_H1_norm << "\n";


    Vector res_Constr(Constr->Height());
    Constr->Mult(sigma_exact_truedofs, res_Constr);
    res_Constr -= trueRhs.GetBlock(2);

    double res_Constr_norm = res_Constr.Norml2() / sqrt (res_Constr.Size());

    if (verbose)
        std::cout << "res_Constr_norm = " << res_Constr_norm << "\n";

    chrono.Stop();
    if (verbose)
        cout<<"Discretized problem is assembled in "<< chrono.RealTime() <<" seconds.\n";
    chrono.Clear();
    chrono.Start();

    Solver *prec;
    prec = new BlockDiagonalPreconditioner(block_trueOffsets);

    HypreParMatrix *Schur;
    HypreParMatrix *MinvDt = Constr->Transpose();
    HypreParVector *Md = new HypreParVector(MPI_COMM_WORLD, M->GetGlobalNumRows(),
                                         M->GetRowStarts());
    M->GetDiag(*Md);
    MinvDt->InvScaleRows(*Md);
    Schur = ParMult(Constr, MinvDt);

    HypreBoomerAMG * precSchur = new HypreBoomerAMG(*Schur);
    precSchur->SetPrintLevel(0);
    precSchur->iterative_mode = false;

    HypreDiagScale * precSigma = new HypreDiagScale(*M);
    precSigma->iterative_mode = false;

    HypreBoomerAMG * precS = new HypreBoomerAMG(*C);
    precS->SetPrintLevel(0);
    precS->iterative_mode = false;


    ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(0, precSigma);
    ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(1, precS);
    ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(2, precSchur);

    chrono.Stop();
    if (verbose)
        std::cout << "Preconditioner was created in "<< chrono.RealTime() <<" seconds.\n";

    MINRESSolver solver(comm);
    if (verbose)
        cout << "Linear solver: MINRES \n";

    solver.SetAbsTol(atol);
    solver.SetRelTol(rtol);
    solver.SetMaxIter(max_num_iter);
    solver.SetOperator(*MainOp);

    if (with_prec)
        solver.SetPreconditioner(*prec);
    solver.SetPrintLevel(0);
    trueX = 0.0;

    chrono.Clear();
    chrono.Start();
    solver.Mult(trueRhs, trueX);
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

    chrono.Clear();
    chrono.Start();

    {
        BlockVector sol(blockfunct_trueOffsets);
        sol.GetBlock(0) = trueX.GetBlock(0);
        sol.GetBlock(1) = trueX.GetBlock(1);

        BlockVector rhsfunct(blockfunct_trueOffsets);
        rhsfunct.GetBlock(0) = trueRhs.GetBlock(0);
        rhsfunct.GetBlock(1) = trueRhs.GetBlock(1);

        BlockVector resfunct(blockfunct_trueOffsets);
        MainOpFunct->Mult(sol, resfunct);
        resfunct -= rhsfunct;

        double res_funct_norm = resfunct.Norml2() / sqrt (resfunct.Size());

        if (verbose)
            std::cout << "res_funct_norm for sol = " << res_funct_norm << "\n";

        Vector Msigma(M->Height());
        M->Mult(trueX.GetBlock(0), Msigma);
        Vector BS(B->Height());
        B->Mult(trueX.GetBlock(1), BS);

        Vector res_Hdiv(M->Height());
        res_Hdiv = Msigma;
        res_Hdiv += BS;

        double res_Hdiv_norm = res_Hdiv.Norml2() / sqrt (res_Hdiv.Size());

        if (verbose)
            std::cout << "res_Hdiv_norm for sol = " << res_Hdiv_norm << "\n";

        // checking the residual for the projections of exact solution for H1 equation
        Vector CS(C->Height());
        C->Mult(trueX.GetBlock(1), CS);
        Vector BTsigma(BT->Height());
        BT->Mult(trueX.GetBlock(0), BTsigma);

        Vector res_H1(C->Height());
        res_H1 = CS;
        res_H1 += BTsigma;

        double res_H1_norm = res_H1.Norml2() / sqrt (res_H1.Size());

        if (verbose)
            std::cout << "res_H1_norm for sol = " << res_H1_norm << "\n";
    }

    Vector trueerr_sigma(trueX.GetBlock(0).Size());
    trueerr_sigma = sigma_exact_truedofs;
    trueerr_sigma -= trueX.GetBlock(0);

    double trueerr_sigma_norm = trueerr_sigma.Norml2() / sqrt (trueerr_sigma.Size());
    double truesigma_norm = sigma_exact_truedofs.Norml2() / sqrt (sigma_exact_truedofs.Size());

    double trueerr_sigma_relnorm = trueerr_sigma_norm /  truesigma_norm;

    if (verbose)
    {
        std::cout << "true err sigma norm = " << trueerr_sigma_norm << "\n";
        std::cout << "true err sigma rel norm = " << trueerr_sigma_relnorm << "\n";
    }

    ParGridFunction * S = new ParGridFunction(H_space);
    S->Distribute(&(trueX.GetBlock(1)));

    ParGridFunction * sigma = new ParGridFunction(R_space);
    sigma->Distribute(&(trueX.GetBlock(0)));

    // 13. Extract the parallel grid function corresponding to the finite element
    //     approximation X. This is the local solution on each processor. Compute
    //     L2 error norms.

    int order_quad = max(2, 2*feorder+1);
    const IntegrationRule *irs[Geometry::NumGeom];
    for (int i=0; i < Geometry::NumGeom; ++i)
    {
        irs[i] = &(IntRules.Get(i, order_quad));
    }

    double err_sigma = sigma->ComputeL2Error(*Mytest.GetSigma(), irs);
    double norm_sigma = ComputeGlobalLpNorm(2, *Mytest.GetSigma(), *pmesh, irs);

    if (verbose)
        cout << "sigma_h = sigma_hat + div-free part, div-free part = curl u_h \n";

    if (verbose)
    {
        if ( norm_sigma > MYZEROTOL )
            cout << "|| sigma_h - sigma_ex || / || sigma_ex || = " << err_sigma / norm_sigma << endl;
        else
            cout << "|| sigma || = " << err_sigma << " (sigma_ex = 0)" << endl;
    }

    DiscreteLinearOperator Div(R_space, W_space);
    Div.AddDomainInterpolator(new DivergenceInterpolator());
    ParGridFunction DivSigma(W_space);
    Div.Assemble();
    Div.EliminateTestDofs(ess_bdrSigma);
    Div.Mult(*sigma, DivSigma);

    double err_div = DivSigma.ComputeL2Error(*Mytest.GetRhs(),irs);
    double norm_div = ComputeGlobalLpNorm(2, *Mytest.GetRhs(), *pmesh, irs);

    if (verbose)
    {
        cout << "|| div (sigma_h - sigma_ex) || / ||div (sigma_ex)|| = "
                  << err_div/norm_div  << "\n";
    }

    if (verbose)
    {
        //cout << "Actually it will be ~ continuous L2 + discrete L2 for divergence" << endl;
        cout << "|| sigma_h - sigma_ex ||_Hdiv / || sigma_ex ||_Hdiv = "
                  << sqrt(err_sigma*err_sigma + err_div * err_div)/sqrt(norm_sigma*norm_sigma + norm_div * norm_div)  << "\n";
    }

    double norm_S = 0.0;
    //if (withS)
    {
        ParGridFunction * S_exact = new ParGridFunction(S_space);
        S_exact->ProjectCoefficient(*Mytest.GetU());

        double err_S = S->ComputeL2Error(*Mytest.GetU(), irs);
        norm_S = ComputeGlobalLpNorm(2, *Mytest.GetU(), *pmesh, irs);
        if (verbose)
        {
            if ( norm_S > MYZEROTOL )
                std::cout << "|| S_h - S_ex || / || S_ex || = " <<
                             err_S / norm_S << "\n";
            else
                std::cout << "|| S_h || = " << err_S << " (S_ex = 0) \n";
        }

        if (strcmp(space_for_S,"H1") == 0)
        {
            ParFiniteElementSpace * GradSpace;
            FiniteElementCollection *hcurl_coll;
            if (dim == 3)
                GradSpace = C_space;
            else // dim == 4
            {
                hcurl_coll = new ND1_4DFECollection;
                GradSpace = new ParFiniteElementSpace(pmesh.get(), hcurl_coll);
            }
            DiscreteLinearOperator Grad(S_space, GradSpace);
            Grad.AddDomainInterpolator(new GradientInterpolator());
            ParGridFunction GradS(GradSpace);
            Grad.Assemble();
            Grad.Mult(*S, GradS);

            if (numsol != -34 && verbose)
                std::cout << "For this norm we are grad S for S from _Lap solution \n";
            VectorFunctionCoefficient GradS_coeff(dim, uFunTestLap_grad);
            double err_GradS = GradS.ComputeL2Error(GradS_coeff, irs);
            double norm_GradS = ComputeGlobalLpNorm(2, GradS_coeff, *pmesh, irs);
            if (verbose)
            {
                std::cout << "|| Grad_h (S_h - S_ex) || / || Grad S_ex || = " <<
                             err_GradS / norm_GradS << "\n";
                std::cout << "|| S_h - S_ex ||_H^1 / || S_ex ||_H^1 = " <<
                             sqrt(err_S*err_S + err_GradS*err_GradS) / sqrt(norm_S*norm_S + norm_GradS*norm_GradS) << "\n";
            }

            if (dim != 3)
            {
                delete GradSpace;
                delete hcurl_coll;
            }

        }

        delete S_exact;
    }

    if (verbose)
        cout << "Computing projection errors \n";

    double projection_error_sigma = sigma_exact->ComputeL2Error(*Mytest.GetSigma(), irs);

    if(verbose)
    {
        if ( norm_sigma > MYZEROTOL )
        {
            cout << "|| sigma_ex - Pi_h sigma_ex || / || sigma_ex || = " << projection_error_sigma / norm_sigma << endl;
        }
        else
            cout << "|| Pi_h sigma_ex || = " << projection_error_sigma << " (sigma_ex = 0) \n ";
    }

    //if (withS)
    {
        double projection_error_S = S_exact->ComputeL2Error(*Mytest.GetU(), irs);

        if(verbose)
        {
            if ( norm_S > MYZEROTOL )
                cout << "|| S_ex - Pi_h S_ex || / || S_ex || = " << projection_error_S / norm_S << endl;
            else
                cout << "|| Pi_h S_ex ||  = " << projection_error_S << " (S_ex = 0) \n";
        }
    }

    chrono.Stop();
    if (verbose)
        std::cout << "Errors were computed in "<< chrono.RealTime() <<" seconds.\n";

    if (visualization && nDimensions < 4)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;

        //if (withS)
        {
            socketstream S_ex_sock(vishost, visport);
            S_ex_sock << "parallel " << num_procs << " " << myid << "\n";
            S_ex_sock.precision(8);
            MPI_Barrier(pmesh->GetComm());
            S_ex_sock << "solution\n" << *pmesh << *S_exact << "window_title 'S_exact'"
                   << endl;

            socketstream S_h_sock(vishost, visport);
            S_h_sock << "parallel " << num_procs << " " << myid << "\n";
            S_h_sock.precision(8);
            MPI_Barrier(pmesh->GetComm());
            S_h_sock << "solution\n" << *pmesh << *S << "window_title 'S_h'"
                   << endl;

            *S -= *S_exact;
            socketstream S_diff_sock(vishost, visport);
            S_diff_sock << "parallel " << num_procs << " " << myid << "\n";
            S_diff_sock.precision(8);
            MPI_Barrier(pmesh->GetComm());
            S_diff_sock << "solution\n" << *pmesh << *S << "window_title 'S_h - S_exact'"
                   << endl;
        }

        socketstream sigma_sock(vishost, visport);
        sigma_sock << "parallel " << num_procs << " " << myid << "\n";
        sigma_sock.precision(8);
        MPI_Barrier(pmesh->GetComm());
        sigma_sock << "solution\n" << *pmesh << *sigma_exact
               << "window_title 'sigma_exact'" << endl;
        // Make sure all ranks have sent their 'u' solution before initiating
        // another set of GLVis connections (one from each rank):

        socketstream sigmah_sock(vishost, visport);
        sigmah_sock << "parallel " << num_procs << " " << myid << "\n";
        sigmah_sock.precision(8);
        MPI_Barrier(pmesh->GetComm());
        sigmah_sock << "solution\n" << *pmesh << *sigma << "window_title 'sigma'"
                << endl;

        *sigma_exact -= *sigma;
        socketstream sigmadiff_sock(vishost, visport);
        sigmadiff_sock << "parallel " << num_procs << " " << myid << "\n";
        sigmadiff_sock.precision(8);
        MPI_Barrier(pmesh->GetComm());
        sigmadiff_sock << "solution\n" << *pmesh << *sigma_exact
                 << "window_title 'sigma_ex - sigma_h'" << endl;

        MPI_Barrier(pmesh->GetComm());
    }

    MPI_Finalize();
    return 0;

    chrono.Clear();
    chrono.Start();

    if (visualization && nDimensions < 4)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;

        //if (withS)
        {
            socketstream S_ex_sock(vishost, visport);
            S_ex_sock << "parallel " << num_procs << " " << myid << "\n";
            S_ex_sock.precision(8);
            MPI_Barrier(pmesh->GetComm());
            S_ex_sock << "solution\n" << *pmesh << *S_exact << "window_title 'S_exact'"
                   << endl;

            socketstream S_h_sock(vishost, visport);
            S_h_sock << "parallel " << num_procs << " " << myid << "\n";
            S_h_sock.precision(8);
            MPI_Barrier(pmesh->GetComm());
            S_h_sock << "solution\n" << *pmesh << *S << "window_title 'S_h'"
                   << endl;

            *S -= *S_exact;
            socketstream S_diff_sock(vishost, visport);
            S_diff_sock << "parallel " << num_procs << " " << myid << "\n";
            S_diff_sock.precision(8);
            MPI_Barrier(pmesh->GetComm());
            S_diff_sock << "solution\n" << *pmesh << *S << "window_title 'S_h - S_exact'"
                   << endl;
        }

        socketstream sigma_sock(vishost, visport);
        sigma_sock << "parallel " << num_procs << " " << myid << "\n";
        sigma_sock.precision(8);
        MPI_Barrier(pmesh->GetComm());
        sigma_sock << "solution\n" << *pmesh << *sigma_exact
               << "window_title 'sigma_exact'" << endl;
        // Make sure all ranks have sent their 'u' solution before initiating
        // another set of GLVis connections (one from each rank):

        socketstream sigmah_sock(vishost, visport);
        sigmah_sock << "parallel " << num_procs << " " << myid << "\n";
        sigmah_sock.precision(8);
        MPI_Barrier(pmesh->GetComm());
        sigmah_sock << "solution\n" << *pmesh << *sigma << "window_title 'sigma'"
                << endl;

        *sigma_exact -= *sigma;
        socketstream sigmadiff_sock(vishost, visport);
        sigmadiff_sock << "parallel " << num_procs << " " << myid << "\n";
        sigmadiff_sock.precision(8);
        MPI_Barrier(pmesh->GetComm());
        sigmadiff_sock << "solution\n" << *pmesh << *sigma_exact
                 << "window_title 'sigma_ex - sigma_h'" << endl;

        MPI_Barrier(pmesh->GetComm());
    }

    chrono_total.Stop();
    if (verbose)
        std::cout << "Total time consumed was " << chrono_total.RealTime() <<" seconds.\n";

    MPI_Finalize();
    return 0;
}

