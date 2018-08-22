//                                MFEM(with 4D elements) Implicit time-stepping for 3D/4D heat equation
//

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

double tdf_rhs_3D(const Vector & vec, double t);
double tdf_rhs_4D(const Vector & vec, double t);
double tdf_u_3D(const Vector & vec, double t);
double tdf_u_4D(const Vector & vec, double t);

int main(int argc, char *argv[])
{
    // 1. Initialize MPI.
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    bool verbose = (myid == 0);
    bool compute_error = false;
    bool visualization = 0;

    int nDimensions     = 4; // dimension of the space + time

    int ser_ref_levels  = 0;
    int par_ref_levels  = 1;

    int ntsteps = 26;

    const char *mesh_file = "../data/cube_3d_moderate.mesh";
    //const char *mesh_file = "../data/square_2d_moderate.mesh";

    //const char *mesh_file = "../data/cube4d_low.MFEM";
    //const char *mesh_file = "../data/cube4d.MFEM";
    //const char *mesh_file = "../data/orthotope3D_moderate.mesh";
    //const char *mesh_file = "../data/sphere3D_0.1to0.2.mesh";
    //const char * mesh_file = "../data/orthotope3D_fine.mesh";

    int feorder         = 0;

    if (verbose)
        cout << "Solving heat equation with time-stepping \n";

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use (must define a space only mesh).");
    args.AddOption(&feorder, "-o", "--feorder",
                   "Finite element order (polynomial degree).");
    args.AddOption(&ser_ref_levels, "-sref", "--sref",
                   "Number of serial refinements 4d mesh.");
    args.AddOption(&par_ref_levels, "-pref", "--pref",
                   "Number of parallel refinements 4d mesh.");
    args.AddOption(&nDimensions, "-dim", "--whichD",
                   "Dimension of the space-time problem.");
    args.AddOption(&ntsteps, "-nst", "--nsteps",
                   "Number of time steps.");
    //args.AddOption(&deltat, "-dt", "--deltat",
                   //"Time step.");
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

    double deltat = 1.0/(1.0 * ntsteps);

    if (verbose)
        std::cout << "Running tests for the report: \n";

    if (nDimensions == 3)
        mesh_file = "../data/square_2d_moderate.mesh";
    else if (nDimensions == 4)
        mesh_file = "../data/cube_3d_0.07to0.09.netgen";
    else
    {
        MFEM_ABORT("This example was desgined to work in either 3D or 4D (space-time).")
    }

    if (verbose)
        std::cout << "For the records: mesh_file = " << mesh_file << "\n";

    if (verbose)
        cout << "Number of mpi processes: " << num_procs << endl << flush;

    int max_num_iter = 10000;
    double rtol = 1e-12;
    double atol = 1e-14;

    Mesh *mesh = NULL;

    shared_ptr<ParMesh> pmesh;

    StopWatch chrono;

    if (verbose)
        cout << "Reading a " << nDimensions - 1 << "d space mesh from the file " << mesh_file << endl;
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
            cout << "Creating parmesh(" << nDimensions - 1 <<
                    "d) from the serial mesh (" << nDimensions - 1 << "d)" << endl << flush;
        pmesh = make_shared<ParMesh>(comm, *mesh);
        delete mesh;
    }

    for (int l = 0; l < par_ref_levels; l++)
       pmesh->UniformRefinement();

    int spacedim = pmesh->Dimension();

    //if(spacedim==3) pmesh->ReorientTetMesh();

    pmesh->PrintInfo(std::cout); if(verbose) cout << endl;
    if (verbose)
        std::cout << "# of timesteps: " << ntsteps << "\n"
                  << "deltat: " << deltat << "\n";

    /*
    {
        int local_nverts = pmesh->GetNV();
        int global_nverts = 0;
        MPI_Reduce(&local_nverts, &global_nverts, 1, MPI_INT, MPI_SUM, 0, comm);

        if (verbose)
            std::cout << "Global number of vertices (in the space mesh, "
                         "with multiple counts for shared vertices) = " << global_nverts << "\n";
        if (verbose)
            std::cout << "Global number of vertices (with time steps, with multiple "
                         "counts for shared vertices) = " << global_nverts * (ntsteps + 1) << "\n";

        chrono.Clear();
        chrono.Start();

        MPI_Finalize();
        return 0;
    }
    */

    // 6. Define a parallel finite element space on the parallel mesh. Here we
    //    use the Raviart-Thomas finite elements of the specified order.

    FiniteElementCollection *h1_coll;
    h1_coll = new H1_FECollection(feorder+1, spacedim);

    ParFiniteElementSpace *H_space = new ParFiniteElementSpace(pmesh.get(), h1_coll);

    HYPRE_Int dimH = H_space->GlobalTrueVSize();

    if (verbose)
    {
        std::cout << "***********************************************************\n";
        std::cout << "dim(H) = " << dimH << "\n";
        std::cout << "***********************************************************\n";
    }

    // 7. Define the block structure of the problem.
    //    block_offsets is used for Vector based on dof (like ParGridFunction or ParLinearForm),
    //    block_trueOffstes is used for Vector based on trueDof (HypreParVector
    //    for the rhs and solution of the linear system).  The offsets computed
    //    here are local to the processor.

    int numblocks = 1;

    Array<int> block_offsets(numblocks + 1); // number of variables + 1
    block_offsets[0] = 0;
    block_offsets[1] = H_space->GetVSize();
    block_offsets.PartialSum();

    Array<int> block_trueOffsets(numblocks + 1); // number of variables + 1
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = H_space->TrueVSize();
    block_trueOffsets.PartialSum();

    BlockVector x(block_offsets);
    x = 0.0;
    BlockVector trueX(block_trueOffsets), trueRhs(block_trueOffsets);
    trueX = 0.0;
    trueRhs = 0.0;

    // 8. Define the coefficients, analytical solution, and rhs of the PDE.

    FunctionCoefficient * rhs_func_coeff;
    if (spacedim == 2)
        rhs_func_coeff = new FunctionCoefficient(tdf_rhs_3D);
    else
        rhs_func_coeff = new FunctionCoefficient(tdf_rhs_4D);

    FunctionCoefficient * u_func_coeff;
    if (spacedim == 2)
        u_func_coeff = new FunctionCoefficient(tdf_u_3D);
    else
        u_func_coeff = new FunctionCoefficient(tdf_u_4D);

    ConstantCoefficient invtau_coeff(1.0/deltat);

    //----------------------------------------------------------
    // Setting boundary conditions.
    //----------------------------------------------------------

    Array<int> ess_bdrS(pmesh->bdr_attributes.Max());
    ess_bdrS = 1;

    if (verbose)
    {
        std::cout << "Boundary conditions: \n";
        std::cout << "ess bdr S: \n";
        ess_bdrS.Print(std::cout, pmesh->bdr_attributes.Max());
    }

    // 10. Assemble the finite element matrices for the implicit in time approximation:
    //
    //                       M * (u^{n+1} - u^n) / tau  + L * u^{n+1} = M * f^{n+1}
    //
    //                                                or
    //
    //                 A * u^{n+1} = (1/tau * M + L) u^{n+1} = M * f^{n+1} + 1/tau * M * u^n
    //     where:
    //     M = (u,v), mass matrix for H^1
    //     L = (-laplace(u),v) = (grad u, grad v)

    chrono.Clear();
    chrono.Start();

    ParBilinearForm *Ablock;
    Ablock = new ParBilinearForm(H_space);
    Ablock->AddDomainIntegrator(new DiffusionIntegrator);
    Ablock->AddDomainIntegrator(new MassIntegrator(invtau_coeff));
    Ablock->Assemble();
    Ablock->EliminateEssentialBC(ess_bdrS);
    Ablock->Finalize();
    HypreParMatrix *A = Ablock->ParallelAssemble();
    delete Ablock;

    Ablock = new ParBilinearForm(H_space);
    Ablock->AddDomainIntegrator(new DiffusionIntegrator);
    Ablock->AddDomainIntegrator(new MassIntegrator(invtau_coeff));
    Ablock->Assemble();
    Ablock->Finalize();
    HypreParMatrix *A_nobnd = Ablock->ParallelAssemble();
    delete Ablock;

    ParBilinearForm *Mblock = new ParBilinearForm(H_space);
    Mblock->AddDomainIntegrator(new MassIntegrator(invtau_coeff));
    Mblock->Assemble();
    Mblock->EliminateEssentialBC(ess_bdrS);
    Mblock->Finalize();
    HypreParMatrix * M = Mblock->ParallelAssemble();
    delete Mblock;

    BlockOperator *CFOSLSop = new BlockOperator(block_trueOffsets);
    CFOSLSop->SetBlock(0,0, A);
    CFOSLSop->owns_blocks = true;

    if (verbose)
        std::cout << "System built in " << chrono.RealTime() << "s. \n";

    chrono.Clear();
    chrono.Start();

    if (verbose)
        cout << "Using boomerAMG for scalar unknown S \n";
    HypreBoomerAMG * invA = new HypreBoomerAMG(*A);
    invA->SetPrintLevel(0);
    invA->iterative_mode = false;

    BlockDiagonalPreconditioner * prec = new BlockDiagonalPreconditioner(block_trueOffsets);
    prec->SetDiagonalBlock(0, invA);
    prec->owns_blocks = true;

    if (verbose)
        std::cout << "Preconditioner built in " << chrono.RealTime() << "s. \n";

    CGSolver solver(MPI_COMM_WORLD);
    solver.SetAbsTol(sqrt(atol));
    solver.SetRelTol(sqrt(rtol));
    solver.SetMaxIter(max_num_iter);
    solver.SetOperator(*CFOSLSop);
    solver.SetPreconditioner(*prec);
    solver.SetPrintLevel(0);

    Array<int> essbdr_tdofs;
    H_space->GetEssentialTrueDofs(ess_bdrS, essbdr_tdofs);

    // Main loop over time iterations.
    double time = 0.0;
    int n_step_viz = 5;

    u_func_coeff->SetTime(time);
    ParGridFunction *S_exact = new ParGridFunction(H_space);
    S_exact->ProjectCoefficient(*u_func_coeff);

    ParGridFunction *S = new ParGridFunction(H_space);

    Vector tempvec(H_space->GetTrueVSize());

    Vector prev_sol(H_space->GetTrueVSize());
    S_exact->ParallelProject(prev_sol);

    Vector next_sol(H_space->GetTrueVSize());

    int local_nverts = pmesh->GetNV();
    int global_nverts = 0;
    MPI_Reduce(&local_nverts, &global_nverts, 1, MPI_INT, MPI_SUM, 0, comm);

    int global_system_size = A->GetGlobalNumCols();
    if (verbose)
        std::cout << "Global system size to be solved at each time step = " << global_system_size << "\n";
    if (verbose)
        std::cout << "Global number of vertices (in the space mesh, "
                     "with multiple counts for shared vertices) = " << global_nverts << "\n";
    if (verbose)
        std::cout << "Global number of vertices (with time steps, with multiple "
                     "counts for shared vertices) = " << global_nverts * (ntsteps + 1) << "\n";

#ifdef MFEM_DEBUG
    StopWatch chrono_2;
#endif

    chrono.Clear();
    chrono.Start();

    for (int n = 0; n < ntsteps; ++n)
    {
        ParLinearForm *fform = new ParLinearForm(H_space);
        rhs_func_coeff->SetTime(time + deltat);
        fform->AddDomainIntegrator(new DomainLFIntegrator(*rhs_func_coeff));
        fform->Assemble();
        fform->ParallelAssemble(trueRhs.GetBlock(0));

        // it is assumed that boundary conditions are homogeneous!
        for (int i = 0; i < essbdr_tdofs.Size(); ++i)
        {
            int tdof = essbdr_tdofs[i];
            trueRhs.GetBlock(0)[tdof] = 0.0;
        }

        M->Mult(prev_sol, tempvec);

        trueRhs.GetBlock(0) += tempvec;

        next_sol = prev_sol;
        //next_sol = 0.0;

#ifdef MFEM_DEBUG
        chrono_2.Clear();
        chrono_2.Start();
#endif

        solver.Mult(trueRhs, next_sol);

#ifdef MFEM_DEBUG
        if (verbose)
        {
           if (solver.GetConverged())
              std::cout << "CG converged in " << solver.GetNumIterations()
                        << " iterations with a residual norm of " << solver.GetFinalNorm() << ".\n";
           else
              std::cout << "CG did not converge in " << solver.GetNumIterations()
                        << " iterations. Residual norm is " << solver.GetFinalNorm() << ".\n";
           std::cout << "CG solver took " << chrono_2.RealTime() << "s. \n";
        }
#endif
        // Computing error for S
        if (compute_error)
        {
            S->Distribute(&next_sol);

            int order_quad = max(2, 2*feorder+1);
            const IntegrationRule *irs[Geometry::NumGeom];
            for (int i=0; i < Geometry::NumGeom; ++i)
                irs[i] = &(IntRules.Get(i, order_quad));

            u_func_coeff->SetTime(time + deltat);

            double err_S = S->ComputeL2Error(*u_func_coeff, irs);
            double norm_S = ComputeGlobalLpNorm(2, *u_func_coeff, *pmesh, irs);

            if (verbose)
            {
                if (norm_S > 1.0e-13)
                    std::cout << "|| S_h - S_ex || / || S_ex || = " <<
                             err_S / norm_S << "\n";
                else
                    std::cout << "|| S_h || = " << err_S << " (S_ex = 0) \n";
            }
        }

        if (visualization && (n % n_step_viz == 0 || n == ntsteps - 1))
        {
            char vishost[] = "localhost";
            int  visport   = 19916;

            delete S_exact;
            u_func_coeff->SetTime(time + deltat);
            S_exact = new ParGridFunction(H_space);
            S_exact->ProjectCoefficient(*u_func_coeff);

            socketstream s_sock(vishost, visport);
            s_sock << "parallel " << num_procs << " " << myid << "\n";
            s_sock.precision(8);
            MPI_Barrier(pmesh->GetComm());
            s_sock << "solution\n" << *pmesh << *S_exact << "window_title 'S_exact, step No." << n << "'"
                    << endl;

            S->Distribute(&next_sol);

            socketstream ss_sock(vishost, visport);
            ss_sock << "parallel " << num_procs << " " << myid << "\n";
            ss_sock.precision(8);
            MPI_Barrier(pmesh->GetComm());
            ss_sock << "solution\n" << *pmesh << *S << "window_title 'S, step No." << n << "'"
                    << endl;

            *S_exact -= *S;
            socketstream sss_sock(vishost, visport);
            sss_sock << "parallel " << num_procs << " " << myid << "\n";
            sss_sock.precision(8);
            MPI_Barrier(pmesh->GetComm());
            sss_sock << "solution\n" << *pmesh << *S_exact
                     << "window_title 'difference for S, step No." << n << "'" << endl;
        }

        prev_sol = next_sol;
        time += deltat;

        delete fform;
    }

    if (verbose)
        std::cout << "Time-stepping loop was finished in " << chrono.RealTime() << "s. \n";

    MFEM_ASSERT(fabs(time - 1.0) < 1.0e-13, "The time interval must be [0,1]");

    delete u_func_coeff;
    delete rhs_func_coeff;

    delete CFOSLSop;
    delete A_nobnd;
    delete prec;

    delete S;
    delete S_exact;

    delete H_space;
    delete h1_coll;

    MPI_Finalize();
    return 0;
}

double tdf_u_3D(const Vector & vec, double t)
{
    double x = vec(0);
    double y = vec(1);

    double res = t*t*exp(t) * sin (3.0 * M_PI * x);
    res *= sin (2.0 * M_PI * y);

    return res;
}

double tdf_rhs_3D(const Vector & vec, double t)
{
    double x = vec(0);
    double y = vec(1);

    // res = - (u_xx + u_yy) first ...
    double res = (3.0 * M_PI * 3.0 * M_PI + 2.0 * M_PI * 2.0 * M_PI) * tdf_u_3D(vec, t);
    // adding u_t!
    res += sin (3.0 * M_PI * x) * sin (2.0 * M_PI * y) * (t*t + 2.0 * t)*exp(t) ;

    return res;

}

double tdf_u_4D(const Vector & vec, double t)
{
    double x = vec(0);
    double y = vec(1);
    double z = vec(2);

    double res = t*t*exp(t) * sin (3.0 * M_PI * x);
    res *= sin (2.0 * M_PI * y);
    res *= sin (M_PI * z);

    return res;
}

double tdf_rhs_4D(const Vector & vec, double t)
{
    double x = vec(0);
    double y = vec(1);
    double z = vec(2);

    // res = - (u_xx + u_yy + u_zz first) ...
    double res = (3.0 * M_PI * 3.0 * M_PI + 2.0 * M_PI * 2.0 * M_PI + M_PI * M_PI) * tdf_u_4D(vec, t);
    // adding u_t!
    res += sin (3.0 * M_PI * x) * sin (2.0 * M_PI * y) * sin (M_PI * z) * (t*t + 2.0 * t)*exp(t) ;

    return res;
}



