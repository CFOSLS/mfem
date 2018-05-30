//
//                                MFEM CFOSLS Wave equation (+ mesh generator) solved by hypre
//
// Compile with: make
//
// Description:
//               This example code solves a simple 4D Wave equation over [0,1]^d, d=3,4
//               written in first-order formulation
//                                  sigma_1 + grad u   = 0
//                                  sigma_2 - u_t      = 0
//                                  div_(x,t) sigma    = f
//                       with boundary conditions:
//                                   u(x,t)            = 0      on the spatial boundary
//                                   u(x,0)            = u0     (initial condition on u)
//                                   du/dt(x,0)        = ut0    (initial condition on du/dt)
//               Here, we use a given exact solution
//                                  u(xt) = uFun_ex(xt)
//               and compute the corresponding r.h.s.
//               We discretize with Raviart-Thomas finite elements (sigma), continuous H1 elements (u) and
//		 discontinuous polynomials (mu) for the lagrange multiplier.
//               Solver: ~ hypre with a block-diagonal preconditioner with BoomerAMG
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>

#define MYZEROTOL (1.0e-13)

using namespace std;
using namespace mfem;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

int main(int argc, char *argv[])
{
    StopWatch chrono;


    // 1. Initialize MPI.
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    bool verbose = (myid == 0);
    bool visualization = 0;

    int nDimensions     = 3;
    int numsol          = 3;

    int ser_ref_levels  = 1;
    int par_ref_levels  = 1;

    const char *formulation = "cfosls";      // "cfosls" or "fosls"
    bool with_divdiv = false;                // should be true for fosls and can be false for cfosls
    bool use_ADS = false;                    // works only in 3D and for with_divdiv = true

    const char *mesh_file = "../data/cube_3d_fine.mesh";
    //const char *mesh_file = "../data/square_2d_moderate.mesh";

    //const char *mesh_file = "../data/cube4d_low.MFEM";
    //const char *mesh_file = "../data/cube4d.MFEM";
    //const char *mesh_file = "../data/pmesh_cube_for_test.mesh";
    //const char *mesh_file = "../data/mesh4_saved";
    //const char *mesh_file = "dsadsad";
    //const char *mesh_file = "../build3/mesh_par1_id0_np_1.mesh";
    //const char *mesh_file = "../build3/mesh_par1_id0_np_2.mesh";
    //const char *mesh_file = "../data/tempmesh_frompmesh.mesh";
    //const char *mesh_file = "../data/orthotope3D_moderate.mesh";
    //const char *mesh_file = "../data/sphere3D_0.1to0.2.mesh";
    //const char * mesh_file = "../data/orthotope3D_fine.mesh";

    //const char * meshbase_file = "../data/sphere3D_0.1to0.2.mesh";
    //const char * meshbase_file = "../data/sphere3D_0.05to0.1.mesh";
    //const char * meshbase_file = "../data/sphere3D_veryfine.mesh";
    //const char * meshbase_file = "../data/beam-tet.mesh";
    //const char * meshbase_file = "../data/escher-p3.mesh";
    //const char * meshbase_file = "../data/orthotope3D_moderate.mesh";
    //const char * meshbase_file = "../data/orthotope3D_fine.mesh";
    //const char * meshbase_file = "../data/square_2d_moderate.mesh";
    //const char * meshbase_file = "../data/square_2d_fine.mesh";
    //const char * meshbase_file = "../data/square-disc.mesh";
    //const char *meshbase_file = "dsadsad";
    //const char * meshbase_file = "../data/circle_fine_0.1.mfem";
    //const char * meshbase_file = "../data/circle_moderate_0.2.mfem";

    int feorder         = 0;

    if (verbose)
        cout << "Solving (C)FOSLS Wave equation with MFEM & hypre" << endl;

    OptionsParser args(argc, argv);
    //args.AddOption(&mesh_file, "-m", "--mesh",
    //               "Mesh file to use.");
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
    args.AddOption(&formulation, "-form", "--formul",
                   "Formulation to use.");
    args.AddOption(&with_divdiv, "-divdiv", "--with-divdiv", "-no-divdiv",
                   "--no-divdiv",
                   "Decide whether div-div term is present.");
    args.AddOption(&use_ADS, "-ADS", "--with-ADS", "-no-ADS",
                   "--no-ADS",
                   "Decide whether to use ADS.");
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

    if (verbose)
        std::cout << "Running tests for the paper: \n";

    if (nDimensions == 3)
    {
        numsol = -34;
        mesh_file = "../data/cube_3d_moderate.mesh";
    }
    else if (nDimensions == 4)// 4D case
    {
        numsol = -34;
        mesh_file = "../data/cube4d_96.MFEM";
    }
    else
    {
        numsol = -34;
        mesh_file = "../data/square_2d_moderate.mesh";
    }

    if (verbose)
        std::cout << "For the records: numsol = " << numsol
                  << ", mesh_file = " << mesh_file << "\n";


    if (verbose)
        cout << "Number of mpi processes: " << num_procs << endl << flush;

    if ( ((strcmp(formulation,"cfosls") == 0 && (!with_divdiv)) || nDimensions != 3) && use_ADS)
    {
        if (verbose)
            cout << "ADS cannot be used if dim != 3 or if div-div term is absent" << endl;
        MPI_Finalize();
        return 0;
    }

    //DEFAULTED LINEAR SOLVER OPTIONS
    int max_num_iter = 150000;
    double rtol = 1e-12;
    double atol = 1e-14;

    Mesh *mesh = NULL;

    shared_ptr<ParMesh> pmesh;

    // not generating from a lower dimensional mesh
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

    if (mesh) // if only serial mesh was generated previously, parallel mesh is initialized here
    {
        for (int l = 0; l < ser_ref_levels; l++)
            mesh->UniformRefinement();

        if ( verbose )
            cout << "Creating parmesh(" << nDimensions <<
                    "d) from the serial mesh (" << nDimensions << "d)" << endl << flush;
        pmesh = make_shared<ParMesh>(comm, *mesh);
        delete mesh;
    }

    for (int l = 0; l < par_ref_levels; l++)
    {
       pmesh->UniformRefinement();
    }

    int dim = pmesh->Dimension();

    //if(dim==3) pmesh->ReorientTetMesh();

    pmesh->PrintInfo(std::cout); if(verbose) cout << endl;

    using FormulType = CFOSLSFormulation_HdivH1Wave;
    using FEFormulType = CFOSLSFEFormulation_HdivH1Wave;
    using BdrCondsType = BdrConditions_CFOSLS_HdivH1_Wave;
    using ProblemType = FOSLSProblem_HdivH1wave;

    FOSLSFormulation * formulat = new FormulType (dim, numsol, verbose);
    FOSLSFEFormulation * fe_formulat = new FEFormulType(*formulat, feorder);
    BdrConditions * bdr_conds = new BdrCondsType(*pmesh);

    int prec_option = 1;
    ProblemType * problem = new ProblemType
            (*pmesh, *bdr_conds, *fe_formulat, prec_option, verbose);

    bool checkbnd = true;
    if (verbose)
        std::cout << "Solving the problem using the new interfaces \n";

    // There is slight difference for the new code and the old code, final residual is different
    // though the iteration count and errors are all the same. For unknown reason.
    problem->Solve(verbose, checkbnd);

    if (verbose)
        std::cout << "Now proceeding with the older way which involves more explicit problem construction\n";


    // 6. Define a parallel finite element space on the parallel mesh. Here we
    //    use the Raviart-Thomas finite elements of the specified order.

    FiniteElementCollection *hdiv_coll, *h1_coll, *l2_coll;
    if (dim == 4)
    {
        hdiv_coll = new RT0_4DFECollection;
        if (verbose)cout << "RT: order 0 for 4D" << endl;
        if(feorder <= 1)
        {
            h1_coll = new LinearFECollection;
            if (verbose)cout << "H1: order 1 for 4D" << endl;
        }
        else
        {
            h1_coll = new QuadraticFECollection;
            if (verbose)cout << "H1: order 2 for 4D" << endl;
        }

        l2_coll = new L2_FECollection(0, dim);
        if (verbose)cout << "L2: order 0 for 4D" << endl;
    }
    else
    {
        hdiv_coll = new RT_FECollection(feorder, dim);
        if (verbose)cout << "RT: order " << feorder << " for 3D" << endl;
        h1_coll = new H1_FECollection(feorder+1, dim);
        if (verbose)cout << "H1: order " << feorder + 1 << " for 3D" << endl;
        // even in cfosls needed to estimate divergence
        l2_coll = new L2_FECollection(feorder, dim);
        if (verbose)cout << "L2: order " << feorder << " for 3D" << endl;
    }

    ParFiniteElementSpace *R_space = new ParFiniteElementSpace(pmesh.get(), hdiv_coll);
    ParFiniteElementSpace *H_space = new ParFiniteElementSpace(pmesh.get(), h1_coll);
    ParFiniteElementSpace *W_space = new ParFiniteElementSpace(pmesh.get(), l2_coll);

    HYPRE_Int dimR = R_space->GlobalTrueVSize();
    HYPRE_Int dimH = H_space->GlobalTrueVSize();
    HYPRE_Int dimW;
    if (strcmp(formulation,"cfosls") == 0)
        dimW = W_space->GlobalTrueVSize();

    if (verbose)
    {
        std::cout << "***********************************************************\n";
        std::cout << "dim(R) = " << dimR << "\n";
        std::cout << "dim(H) = " << dimH << "\n";
        if (strcmp(formulation,"cfosls") == 0)
        {
            std::cout << "dim(W) = " << dimW << "\n";
            std::cout << "dim(R+H+W) = " << dimR + dimH + dimW << "\n";
        }
        else // fosls
            std::cout << "dim(R+H) = " << dimR + dimH << "\n";
        std::cout << "***********************************************************\n";
    }

    // 7. Define the two BlockStructure of the problem.  block_offsets is used
    //    for Vector based on dof (like ParGridFunction or ParLinearForm),
    //    block_trueOffstes is used for Vector based on trueDof (HypreParVector
    //    for the rhs and solution of the linear system).  The offsets computed
    //    here are local to the processor.

    int numblocks = 2;
    if (strcmp(formulation,"cfosls") == 0)
        numblocks = 3;


    Array<int> block_offsets(numblocks + 1); // number of variables + 1
    block_offsets[0] = 0;
    block_offsets[1] = R_space->GetVSize();
    block_offsets[2] = H_space->GetVSize();
    if (strcmp(formulation,"cfosls") == 0)
        block_offsets[3] = W_space->GetVSize();
    block_offsets.PartialSum();

    Array<int> block_trueOffsets(numblocks + 1); // number of variables + 1
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = R_space->TrueVSize();
    block_trueOffsets[2] = H_space->TrueVSize();
    if (strcmp(formulation,"cfosls") == 0)
        block_trueOffsets[3] = W_space->TrueVSize();
    block_trueOffsets.PartialSum();


    // 8. Define the coefficients, analytical solution, and rhs of the PDE.
    Wave_test Mytest(nDimensions,numsol);

    ConstantCoefficient k(1.0);
    ConstantCoefficient zero(.0);
    Vector vzero(dim); vzero = 0.;
    VectorConstantCoefficient vzero_coeff(vzero);

    //----------------------------------------------------------
    // Setting boundary conditions.
    //----------------------------------------------------------

    Array<int> ess_bdrS(pmesh->bdr_attributes.Max());       // applied to H^1 variable
    ess_bdrS = 1;
    ess_bdrS[pmesh->bdr_attributes.Max()-1] = 0;
    Array<int> ess_bdrSigma(pmesh->bdr_attributes.Max());   // applied to Hdiv variable
    ess_bdrSigma = 0;
    ess_bdrSigma[0] = 1; // t = 0 = essential boundary for sigma from Hdiv = for dS/dt|t=0

    if (verbose)
    {
        std::cout << "Boundary conditions: \n";
        std::cout << "ess bdr Sigma: \n";
        ess_bdrSigma.Print(std::cout, pmesh->bdr_attributes.Max());
        std::cout << "ess bdr S: \n";
        ess_bdrS.Print(std::cout, pmesh->bdr_attributes.Max());
    }

    // 8.5 some additional parelag stuff which is used for coarse lagrange
    // multiplier implementation at the matrix level

    // 9. Define the parallel grid function and parallel linear forms, solution
    //    vector and rhs.

    BlockVector * x, * rhs;
    x = new BlockVector(block_offsets);
    *x = 0.0;
    rhs = new BlockVector(block_offsets);
    *rhs = 0.0;

    ParGridFunction *S_exact = new ParGridFunction(H_space);
    S_exact->ProjectCoefficient(*(Mytest.GetU()));

    ParGridFunction * sigma_exact = new ParGridFunction(R_space);
    sigma_exact->ProjectCoefficient(*(Mytest.GetSigma()));

    x->GetBlock(0) = *sigma_exact;
    x->GetBlock(1) = *S_exact;

    ParLinearForm *fform(new ParLinearForm);
    fform->Update(R_space, rhs->GetBlock(0), 0);
    if (strcmp(formulation,"cfosls") == 0) // cfosls case
        if (with_divdiv)
        {
            if (verbose)
                cout << "Adding div-driven rhside term to the formulation" << endl;
            fform->AddDomainIntegrator(new VectordivDomainLFIntegrator(*(Mytest.GetRhs())));
        }
        else
        {
            if (verbose)
                cout << "No div-driven rhside term in the formulation" << endl;
            fform->AddDomainIntegrator(new VectordivDomainLFIntegrator(zero));
        }
    else // if fosls, then we need righthand side term here
    {
        if (verbose)
            cout << "Adding div-driven rhside term to the formulation" << endl;
        fform->AddDomainIntegrator(new VectordivDomainLFIntegrator(*(Mytest.GetRhs())));
    }
    fform->Assemble();

    ParLinearForm *qform(new ParLinearForm);
    qform->Update(H_space, rhs->GetBlock(1), 0);
    qform->AddDomainIntegrator(new VectorDomainLFIntegrator(vzero_coeff));
    qform->Assemble();

    ParLinearForm *gform(new ParLinearForm);
    if (strcmp(formulation,"cfosls") == 0)
    {
        gform->Update(W_space, rhs->GetBlock(2), 0);
        gform->AddDomainIntegrator(new DomainLFIntegrator(*(Mytest.GetRhs())));
        gform->Assemble();
    }


    // 10. Assemble the finite element matrices for the operator
    //
    //                       CFOSLS = [  A   B  D^T ]
    //                                [ B^T  C   0  ]
    //                                [  D   0   0  ]
    //     where:
    //
    //     A = ( sigma, tau)_{H(div)}
    //     B = (sigma, [ dx(S), -dtS] )
    //     C = ( [dx(S), -dtS], [dx(V),-dtV] )
    //     D = ( div(sigma), mu )

    chrono.Clear();
    chrono.Start();

    //---------------
    //  A Block:
    //---------------

    ParBilinearForm *Ablock(new ParBilinearForm(R_space));
    HypreParMatrix *A;

    Ablock->AddDomainIntegrator(new VectorFEMassIntegrator);
    if (strcmp(formulation,"cfosls") != 0) // fosls, then we need div-div term
    {
        if (verbose)
            cout << "Adding div-div term to the formulation" << endl;
        Ablock->AddDomainIntegrator(new DivDivIntegrator());
    }
    else // cfosls case
        if (with_divdiv)
        {
            if (verbose)
                cout << "Adding div-div term to the formulation" << endl;
            Ablock->AddDomainIntegrator(new DivDivIntegrator());
        }
        else
        {
            if (verbose)
                cout << "No div-div term in the formulation" << endl;
        }
    Ablock->Assemble();
    Ablock->EliminateEssentialBC(ess_bdrSigma, x->GetBlock(0), *fform);
    Ablock->Finalize();
    A = Ablock->ParallelAssemble();

    //---------------
    //  C Block:
    //---------------

    ParBilinearForm *Cblock(new ParBilinearForm(H_space));
    HypreParMatrix *C;
    Cblock->AddDomainIntegrator(new CFOSLS_WaveIntegrator);
    Cblock->Assemble();
    Cblock->EliminateEssentialBC(ess_bdrS, x->GetBlock(1), *qform);
    Cblock->Finalize();
    C = Cblock->ParallelAssemble();

    //---------------
    //  B Block:
    //---------------

    ParMixedBilinearForm *Bblock(new ParMixedBilinearForm(H_space, R_space));
    HypreParMatrix *B;
    Bblock->AddDomainIntegrator(new CFOSLS_MixedWaveIntegrator);
    Bblock->Assemble();
    Bblock->EliminateTestDofs(ess_bdrSigma);
    Bblock->EliminateTrialDofs(ess_bdrS, x->GetBlock(1), *fform);
    Bblock->Finalize();
    B = Bblock->ParallelAssemble();
    HypreParMatrix *BT = B->Transpose();

    //----------------
    //  D Block:
    //-----------------

    HypreParMatrix *D;
    HypreParMatrix *DT;

    if (strcmp(formulation,"cfosls") == 0)
    {
        ParMixedBilinearForm *Dblock(new ParMixedBilinearForm(R_space, W_space));
        Dblock->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
        Dblock->Assemble();
        //Dblock->EliminateTestDofs(ess_bdrSigma); // incorrect!
        Dblock->EliminateTrialDofs(ess_bdrSigma, x->GetBlock(0), *gform); // new
        Dblock->Finalize();
        D = Dblock->ParallelAssemble();
        DT = D->Transpose();
    }

    //=======================================================
    // Assembling the Matrix
    //-------------------------------------------------------

    BlockOperator *CFOSLSop;

    CFOSLSop = new BlockOperator(block_trueOffsets);

    CFOSLSop->SetBlock(0,0, A);
    CFOSLSop->SetBlock(0,1, B);
    CFOSLSop->SetBlock(1,0, BT);
    CFOSLSop->SetBlock(1,1, C);
    if (strcmp(formulation,"cfosls") == 0)
    {
        CFOSLSop->SetBlock(0,2, DT);
        CFOSLSop->SetBlock(2,0, D);
    }


    if (verbose)
        std::cout << "System built in " << chrono.RealTime() << "s. \n";

    // 11. Construct the operators for preconditioner
    //
    //                 P = [ diag(M)         0         ]
    //                     [  0       B diag(M)^-1 B^T ]
    //
    //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
    //     pressure Schur Complement.
    if (verbose)
    {
        if (use_ADS == true)
            cout << "Using ADS (+ I) preconditioner for sigma (and lagrange multiplier)" << endl;
        else
            cout << "Using Diag(A) (and D Diag^(-1)(A) Dt) preconditioner for sigma (and lagrange multiplier)" << endl;
    }

    chrono.Clear();
    chrono.Start();
    Solver * invA;
    HypreParMatrix *DAinvDt;
    if (use_ADS == false)
    {
        if (strcmp(formulation,"cfosls") == 0)
        {
            HypreParMatrix *AinvDt = D->Transpose();
            HypreParVector *Ad = new HypreParVector(MPI_COMM_WORLD, A->GetGlobalNumRows(),
                                                    A->GetRowStarts());
            A->GetDiag(*Ad);

            AinvDt->InvScaleRows(*Ad);
            DAinvDt= ParMult(D, AinvDt);
        }
        invA = new HypreDiagScale(*A);
    }
    else // use_ADS
    {
        invA = new HypreADS(*A, R_space);
    }
    invA->iterative_mode = false;

    Operator * invL;
    if (strcmp(formulation,"cfosls") == 0)
    {
        if (use_ADS == false)
        {
            invL= new HypreBoomerAMG(*DAinvDt);
            ((HypreBoomerAMG *)invL)->SetPrintLevel(0);
            ((HypreBoomerAMG *)invL)->iterative_mode = false;
        }
        else // use_ADS
        {
            invL = new IdentityOperator(D->Height());
        }
    }


    if (verbose)
        cout << "Using boomerAMG for scalar unknown S" << endl;
    HypreBoomerAMG * invC = new HypreBoomerAMG(*C);
    invC->SetPrintLevel(0);

    invC->iterative_mode = false;

    BlockDiagonalPreconditioner * prec;

    prec = new BlockDiagonalPreconditioner(block_trueOffsets);
    //BlockDiagonalPreconditioner prec(block_trueOffsets);
    prec->SetDiagonalBlock(0, invA);
    prec->SetDiagonalBlock(1, invC);
    if (strcmp(formulation,"cfosls") == 0)
        prec->SetDiagonalBlock(2, invL);

    if (verbose)
        std::cout << "Preconditioner built in " << chrono.RealTime() << "s. \n";

    // 12. Solve the linear system with MINRES.
    //     Check the norm of the unpreconditioned residual.

    MINRESSolver solver(MPI_COMM_WORLD);
    solver.SetAbsTol(atol);
    solver.SetRelTol(rtol);
    solver.SetMaxIter(max_num_iter);
    solver.SetOperator(*CFOSLSop);
    solver.SetPreconditioner(*prec);
    solver.SetPrintLevel(0);
    BlockVector * trueX;

    trueX = new BlockVector(block_trueOffsets);
    *trueX = 0.0;

    BlockVector *trueRhs;

    trueRhs = new BlockVector(block_trueOffsets);
    *trueRhs=.0;

    fform->ParallelAssemble(trueRhs->GetBlock(0));
    qform->ParallelAssemble(trueRhs->GetBlock(1));
    if (strcmp(formulation,"cfosls") == 0)
    {
        gform->ParallelAssemble(trueRhs->GetBlock(2));
    }

    chrono.Clear();
    chrono.Start();
    solver.Mult(*trueRhs, *trueX);
    chrono.Stop();

    if (verbose)
    {
       if (solver.GetConverged())
          cout << "MINRES converged in " << solver.GetNumIterations()
                    << " iterations with a residual norm of " << solver.GetFinalNorm() << ".\n";
       else
          cout << "MINRES did not converge in " << solver.GetNumIterations()
                    << " iterations. Residual norm is " << solver.GetFinalNorm() << ".\n";
       cout << "MINRES solver took " << chrono.RealTime() << "s. \n";
    }

    // Checking the residual in the divergence constraint
    Vector vec1 = trueX->GetBlock(0);
    Vector Dvec1(trueRhs->GetBlock(2).Size());
    D->Mult(vec1, Dvec1);
    Dvec1 -= trueRhs->GetBlock(2);
    double local_res_norm = Dvec1.Norml2();
    double global_res_norm = 0.0;
    MPI_Reduce(&local_res_norm, &global_res_norm, 1,
               MPI_DOUBLE, MPI_SUM, 0, comm);
    double local_rhs_norm = trueRhs->GetBlock(2).Norml2();
    double global_rhs_norm = 0.0;
    MPI_Reduce(&local_rhs_norm, &global_rhs_norm, 1,
               MPI_DOUBLE, MPI_SUM, 0, comm);
    if (verbose)
    {
        cout << "rel res_norm for the conservation law = " << global_res_norm / global_rhs_norm << endl;
    }


    ParGridFunction *S(new ParGridFunction);
    S->MakeRef(H_space, x->GetBlock(1), 0);
    S->Distribute(&(trueX->GetBlock(1)));

    ParGridFunction *sigma(new ParGridFunction);
    sigma->MakeRef(R_space, x->GetBlock(0), 0);
    sigma->Distribute(&(trueX->GetBlock(0)));

    // 13. Extract the parallel grid function corresponding to the finite element
    //     approximation X. This is the local solution on each processor. Compute
    //     L2 error norms.

    int order_quad = max(2, 2*feorder+1);
    const IntegrationRule *irs[Geometry::NumGeom];
    for (int i=0; i < Geometry::NumGeom; ++i)
        irs[i] = &(IntRules.Get(i, order_quad));


    double err_sigma = sigma->ComputeL2Error(*(Mytest.GetSigma()), irs);
    double norm_sigma = ComputeGlobalLpNorm(2, *(Mytest.GetSigma()), *pmesh, irs);

    if (verbose)
    {
        //cout << "|| sigma_h - sigma_ex ||  = "
        //          << err_sigma  << "\n";
        cout << "|| sigma_h - sigma_ex || / || sigma_ex || = "
                  << err_sigma/norm_sigma  << "\n";
    }

    ParDiscreteLinearOperator Div(R_space, W_space);
    Div.AddDomainInterpolator(new DivergenceInterpolator());
    Div.Assemble();
    Div.Finalize();
    Div.ParallelAssemble();

    ParGridFunction DivSigma(W_space);
    Div.Mult(*sigma, DivSigma);

    ParGridFunction DivSigma_exact(W_space);
    DivSigma_exact.ProjectCoefficient(*(Mytest.GetRhs()));

    double err_div = DivSigma.ComputeL2Error(*(Mytest.GetRhs()),irs);
    double norm_div = ComputeGlobalLpNorm(2, *(Mytest.GetRhs()), *pmesh, irs);

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
    double err_S  = S->ComputeL2Error(*(Mytest.GetU()), irs);
    double norm_S = ComputeGlobalLpNorm(2, *(Mytest.GetU()), *pmesh, irs);

    if (verbose)
    {
        //cout << "|| S_h - S_ex || = "
        //          << err_S << "\n";
        cout << "|| S_h - S_ex || / || S_ex || = "
                  << err_S/norm_S  << "\n";
    }

    ParFiniteElementSpace * GradSpace;
    FiniteElementCollection *hcurl_coll;
    if (dim == 4)
    {
        hcurl_coll = new ND1_4DFECollection;
        GradSpace = new ParFiniteElementSpace(pmesh.get(), hcurl_coll);
    }
    else
    {
        hcurl_coll = new ND_FECollection(feorder+1, dim);
        GradSpace = new ParFiniteElementSpace(pmesh.get(), hcurl_coll);
    }

    DiscreteLinearOperator Grad(H_space, GradSpace);
    Grad.AddDomainInterpolator(new GradientInterpolator());
    ParGridFunction GradS(GradSpace);
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

    // Check value of functional and mass conservation
    {
        trueX->GetBlock(2) = 0.0;
        *trueRhs = 0.0;;
        CFOSLSop->Mult(*trueX, *trueRhs);
        double localFunctional = (*trueX)*(*trueRhs);
        double globalFunctional;
        MPI_Reduce(&localFunctional, &globalFunctional, 1,
                   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (verbose)
        {
            cout << "|| sigma_h - L(S_h) ||^2 = " << globalFunctional<< "\n";
            cout << "|| div_h sigma_h - f ||^2 = " << err_div*err_div  << "\n";
            cout << "|| f ||^2 = " << norm_div*norm_div  << "\n";
            cout << "Relative Energy Error = " << sqrt(globalFunctional+err_div*err_div)/norm_div<< "\n";
        }

        ParLinearForm massform(W_space);
        massform.AddDomainIntegrator(new DomainLFIntegrator(*(Mytest.GetRhs())));
        massform.Assemble();

        double mass_loc = massform.Norml1();
        double mass;
        MPI_Reduce(&mass_loc, &mass, 1,
                   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (verbose)
            cout << "Sum of local mass = " << mass<< "\n";

        trueRhs->GetBlock(2) -= massform;
        double mass_loss_loc = trueRhs->GetBlock(2).Norml1();
        double mass_loss;
        MPI_Reduce(&mass_loss_loc, &mass_loss, 1,
                   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (verbose)
            cout << "Sum of local mass loss = " << mass_loss<< "\n";
    }

    if (verbose)
        cout << "Computing projection errors \n";

    double projection_error_sigma = sigma_exact->ComputeL2Error(*(Mytest.GetSigma()), irs);

    if(verbose)
    {
        if ( norm_sigma > MYZEROTOL )
        {
            cout << "|| sigma_ex - Pi_h sigma_ex || / || sigma_ex || = " << projection_error_sigma / norm_sigma << endl;
        }
        else
            cout << "|| Pi_h sigma_ex || = " << projection_error_sigma << " (sigma_ex = 0) \n ";
    }
    double projection_error_S = S_exact->ComputeL2Error(*(Mytest.GetU()), irs);

    if(verbose)
    {
       if ( norm_S > MYZEROTOL )
           cout << "|| S_ex - Pi_h S_ex || / || S_ex || = " << projection_error_S / norm_S << endl;
       else
           cout << "|| Pi_h S_ex ||  = " << projection_error_S << " (S_ex = 0) \n";
    }

    if (visualization)
    {
        // Make sure all ranks have sent their 'u' solution before initiating
        // another set of GLVis connections (one from each rank):
        char vishost[] = "localhost";
        int  visport   = 19916;
        socketstream uu_sock(vishost, visport);
        uu_sock << "parallel " << num_procs << " " << myid << "\n";
        uu_sock.precision(8);
        MPI_Barrier(pmesh->GetComm());
        uu_sock << "solution\n" << *pmesh << *sigma << "window_title 'sigma'"
                << endl;

        socketstream u_sock(vishost, visport);
        u_sock << "parallel " << num_procs << " " << myid << "\n";
        u_sock.precision(8);
        MPI_Barrier(pmesh->GetComm());
        u_sock << "solution\n" << *pmesh << *sigma_exact
               << "window_title 'sigma_exact'" << endl;

        *sigma_exact -= *sigma;
        socketstream uuu_sock(vishost, visport);
        uuu_sock << "parallel " << num_procs << " " << myid << "\n";
        uuu_sock.precision(8);
        MPI_Barrier(pmesh->GetComm());
        uuu_sock << "solution\n" << *pmesh << *sigma_exact
                 << "window_title 'difference for sigma'" << endl;

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


        socketstream ds_sock(vishost, visport);
        ds_sock << "parallel " << num_procs << " " << myid << "\n";
        ds_sock.precision(8);
        MPI_Barrier(pmesh->GetComm());
        ds_sock << "solution\n" << *pmesh << DivSigma << "window_title 'divsigma'"
                << endl;

        socketstream dse_sock(vishost, visport);
        dse_sock << "parallel " << num_procs << " " << myid << "\n";
        dse_sock.precision(8);
        MPI_Barrier(pmesh->GetComm());
        dse_sock << "solution\n" << *pmesh << DivSigma_exact << "window_title 'divsigma exact'"
                << endl;

        DivSigma -= DivSigma_exact;
        socketstream dsd_sock(vishost, visport);
        dsd_sock << "parallel " << num_procs << " " << myid << "\n";
        dsd_sock.precision(8);
        MPI_Barrier(pmesh->GetComm());
        dsd_sock << "solution\n" << *pmesh << DivSigma << "window_title 'divsigma error'"
                << endl;

    }

    // 17. Free the used memory.
    delete fform;
    delete gform;
    delete CFOSLSop;
    if (strcmp(formulation,"cfosls") == 0)
    {
        delete DT;
        delete D;
    }
    delete C;
    delete BT;
    delete B;
    delete A;

    delete Ablock;
    delete Bblock;
    delete Cblock;
    delete H_space;
    delete R_space;
    delete W_space;
    delete hdiv_coll;
    delete h1_coll;
    delete l2_coll;
    delete hcurl_coll;
    delete GradSpace;

    MPI_Finalize();
    return 0;
}


