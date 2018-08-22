//                                MFEM(with 4D elements) CFOSLS for 3D/4D hyperbolic equation
//                                  with mesh generator and visualization
//
// Compile with: make
//
// Description:  This example code solves a simple 3D/4D hyperbolic problem over [0,1]^3(4)
//               corresponding to the saddle point system
//                                  sigma_1 = u * b
//							 		sigma_2 - u        = 0
//                                  div_(x,t) sigma    = f
//                       with b = vector function (~velocity),
//						 NO boundary conditions (which work only in case when b * n = 0 pointwise at the domain space boundary)
//						 and initial condition:
//                                  u(x,0)            = 0
//               Here, we use a given exact solution
//                                  u(xt) = uFun_ex(xt)
//               and compute the corresponding r.h.s.
//               We discretize with Raviart-Thomas finite elements (sigma), continuous H1 elements (u) and
//					  discontinuous polynomials (mu) for the lagrange multiplier.
//
//				 If you want to run your own solution, be sure to change uFun_ex, as well as fFun_ex and check
//				 that the bFun_ex satisfies the condition b * n = 0 (see above).
// Solver: MINRES preconditioned by boomerAMG or ADS

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

void f_Func(const Vector &p, Vector &f);
void u_Func(const Vector &p, Vector &u);
void Hdivtest_fun(const Vector& xt, Vector& out );

int main(int argc, char *argv[])
{
    int num_procs, myid;

    // 1. Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    bool verbose = (myid == 0);
    bool solve_problem = 1; // if true, solves a model problem
    bool visualization = 1; // if true, created VTK output for paraview
    bool convert_to_mesh = 0; // if true, converts the pmesh to a serial mesh and prints it out

    if (verbose)
        std::cout << "Started example for the parallel mesh generator" << std::endl;

    int nDimensions     = 4;

    int ser_ref_levels  = 0;
    int par_ref_levels  = 1;
    int par_ref_cyl_levels = 1; // number of additional refinements to be done for the generated space-time cylinder
    int Nt              = 2;   // number of time slabs (e.g. Nsteps = 2 corresponds to 3 levels: t = 0, t = tau, t = 2 * tau
    double tau          = 0.5; // time step for a slab

    //const char * meshbase_file = "./data/orthotope3D_moderate.mesh";
    //const char * meshbase_file = "./data/orthotope3D_fine.mesh";
    const char * meshbase_file = "../data/cube_3d_moderate.mesh";

    int feorder         = 0; // in 4D cannot use feorder > 0

    if (verbose)
        std::cout << "Parsing input options" << std::endl;

    OptionsParser args(argc, argv);
    args.AddOption(&meshbase_file, "-mbase", "--meshbase",
                   "Mesh base file to use.");
    args.AddOption(&feorder, "-o", "--feorder",
                   "Finite element order (polynomial degree).");
    args.AddOption(&ser_ref_levels, "-sref", "--sref",
                   "Number of serial refinements for the base mesh.");
    args.AddOption(&par_ref_levels, "-pref", "--pref",
                   "Number of parallel refinements for the base mesh.");
    args.AddOption(&par_ref_cyl_levels, "-pcref", "--pref-cyl",
                   "Number of parallel refinements for the generated space-time mesh.");
    args.AddOption(&nDimensions, "-dim", "--whichD",
                   "Dimension of the space-time problem.");
    args.AddOption(&Nt, "-nstps", "--nsteps",
                   "Number of time steps.");
    args.AddOption(&tau, "-tau", "--tau",
                   "Time step.");
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
       if (verbose)
           std::cerr << "Bad input arguments" << std:: endl;
       MPI_Finalize();
       return 1;
    }
    if (verbose)
    {
       args.PrintOptions(cout);
    }

    if (verbose)
        cout << "Number of mpi processes: " << num_procs << endl << flush;

    StopWatch chrono;

    Mesh *meshbase = NULL;
    ifstream imesh(meshbase_file);
    if (!imesh)
    {
        std::cerr << "\nCan not open mesh base file: " << meshbase_file << '\n' << std::endl;
        MPI_Finalize();
        return -2;
    }
    else
    {
        if (verbose)
             std::cout << "meshbase_file: " << meshbase_file << "\n";
        meshbase = new Mesh(imesh, 1, 1);
        imesh.close();
    }

    meshbase->CheckElementOrientation(true);

    for (int l = 0; l < ser_ref_levels; l++)
        meshbase->UniformRefinement();

    ParMesh * pmeshbase = new ParMesh(comm, *meshbase);
    for (int l = 0; l < par_ref_levels; l++)
        pmeshbase->UniformRefinement();

    delete meshbase;

    ParMesh * pmesh = new ParMeshCyl(comm, *pmeshbase, 0.0, tau, Nt);

    delete pmeshbase;

    (dynamic_cast<ParMeshCyl*>(pmesh))->Refine(par_ref_cyl_levels);

#ifdef MFEM_MEM_ALLOC
    if (nDimensions == 4)
        std::cout << "Memory leak was reported by valgrind in 4D around faces and bdr elements"
                     "when MFEM_MEM_ALLOC = YES is set in defaults.mk prior to building MFEM \n";
#endif

    // if true, converts a pmesh to a mesh (so a global mesh will be produced on each process)
    // which can be printed in a file (as a whole)
    // can be useful for testing purposes
    if (convert_to_mesh)
    {
        int * partitioning = new int [pmesh->GetNE()];
        Mesh * convertedpmesh = new Mesh (*pmesh, &partitioning);
        if (verbose)
        {
            std::stringstream fname;
            fname << "converted_pmesh.mesh";
            std::ofstream ofid(fname.str().c_str());
            ofid.precision(8);
            convertedpmesh->Print(ofid);
        }
        delete partitioning;
        delete convertedpmesh;
    }

    //if(dim==3) pmesh->ReorientTetMesh();

    pmesh->PrintInfo(cout); if(verbose) cout << endl;

    if (verbose)
        cout << "Mesh generator was called successfully" << endl;

    // solving a model problem in Hdiv if solve_problem = true
    if (solve_problem)
    {
        int dim = nDimensions;
        int order = feorder;
        // taken from ex4D_RT

        FiniteElementCollection *fec;
        if(dim==4) fec = new RT0_4DFECollection;
        else fec = new RT_FECollection(order,dim);
        ParFiniteElementSpace fespace(pmesh, fec);

        int dofs = fespace.GlobalTrueVSize();
        if(verbose) cout << "dofs: " << dofs << endl;

        chrono.Clear(); chrono.Start();

        VectorFunctionCoefficient f(dim, f_Func);
        ParLinearForm b(&fespace);
        b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
        b.Assemble();

        ParGridFunction x(&fespace);
        VectorFunctionCoefficient u_exact(dim, u_Func);
        x = 0.0;


        ParBilinearForm a(&fespace);
        a.AddDomainIntegrator(new DivDivIntegrator);
        a.AddDomainIntegrator(new VectorFEMassIntegrator);
        a.Assemble();
        if(pmesh->bdr_attributes.Size())
        {
           Array<int> ess_bdr(pmesh->bdr_attributes.Max()); ess_bdr = 1;
           x.ProjectCoefficient(u_exact);
           a.EliminateEssentialBC(ess_bdr, x, b);
        }
        a.Finalize();

        chrono.Stop();
        if(verbose) cout << "Assembling took " << chrono.UserTime() << "s." << endl;

        HypreParMatrix *A = a.ParallelAssemble();
        HypreParVector *B = b.ParallelAssemble();
        HypreParVector *X = x.ParallelAverage();
        *X = 0.0;


        chrono.Clear(); chrono.Start();

        HypreSolver *prec = NULL;
        if (dim == 2) prec = new HypreAMS(*A, &fespace);
        else if(dim==3)  prec = new HypreADS(*A, &fespace);


        //  HypreGMRES *pcg = new HypreGMRES(*A);
        HyprePCG *pcg = new HyprePCG(*A);
        pcg->SetTol(1e-10);
        pcg->SetMaxIter(50000);
        //pcg->SetPrintLevel(2);
        pcg->SetPrintLevel(0);
        if(prec!=NULL) pcg->SetPreconditioner(*prec);
        pcg->Mult(*B, *X);

        chrono.Stop();
        if(verbose) cout << "Solving took " << chrono.UserTime() << "s." << endl;

        x = *X;

        chrono.Clear(); chrono.Start();
        // FIXME: There is a memory leak related to the IntegrationRules in 4D, haven't fixed that
        int intOrder = 2;//8;
        const IntegrationRule *irs[Geometry::NumGeom];
        for (int i=0; i < Geometry::NumGeom; ++i)
            irs[i] = &(IntRules.Get(i, intOrder));
        double norm = x.ComputeL2Error(u_exact, irs);
        if(verbose) cout << "L2 norm: " << norm << endl;
        if(verbose) cout << "Computing error took " << chrono.UserTime() << "s." << endl;

        x = 0.0; x.ProjectCoefficient(u_exact);
        double projection_error = x.ComputeL2Error(u_exact, irs);
        if(verbose) cout << "L2 norm of projection error: " << projection_error << endl;

        // 15. Free the used memory.
        delete pcg;
        if(prec != NULL) delete prec;
        delete X;
        delete B;
        delete A;

        delete fec;
    }

    if (verbose)
        cout << "Test problem was solved successfully" << endl;

    if (visualization && nDimensions > 2)
    {
        int dim = nDimensions;

        FiniteElementCollection *hdiv_coll;
        if ( dim == 4 )
        {
            hdiv_coll = new RT0_4DFECollection;
            cout << "RT: order 0 for 4D" << endl;
        }
        else
        {
            hdiv_coll = new RT_FECollection(feorder, dim);
            cout << "RT: order " << feorder << " for 3D" << endl;
        }

        ParFiniteElementSpace *R_space = new ParFiniteElementSpace(pmesh, hdiv_coll);

        // creating Hdiv grid-function slices (and printing them in VTK format in a file for paraview)
        ParGridFunction *pgridfuntest = new ParGridFunction(R_space);
        VectorFunctionCoefficient Hdivtest_fun_coeff(nDimensions, Hdivtest_fun);
        pgridfuntest->ProjectCoefficient(Hdivtest_fun_coeff);
        ComputeSlices (*pgridfuntest, 0.1, 2, 0.3, myid, false);

        // creating mesh slices (and printing them in VTK format in a file for paraview)
        ComputeSlices (*pmesh, 0.1, 2, 0.3, myid);

        if (verbose)
            cout << "Test Hdiv function was sliced successfully" << endl;

        delete hdiv_coll;
        delete R_space;
    }

    delete pmesh;

    MPI_Finalize();
    return 0;
}

void videofun(const Vector& xt, Vector& vecvalue )
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    vecvalue.SetSize(xt.Size());
    vecvalue(0) = 3 * x * ( 1 + 0.4 * sin (M_PI * (t + 1.0))) + 2.0 * (y * (y - 0.5) - z) * exp(-0.5*t) + exp(-100.0*(x*x + y * y + (z-0.5)*(z-0.5)));
    vecvalue(1) = 0.0;
    vecvalue(2) = 0.0;
    vecvalue(3) = 0.0;
    //return 3 * x * ( 1 + 0.2 * sin (M_PI * 0.5 * t/(t + 1))) + 2.0 * (y * (y - 0.5) - z) * exp(-0.5*t);
}

void f_Func(const Vector &p, Vector &f)
{
   int dim = p.Size();

   f(0) = sin(M_PI*p(0));
   f(1) = sin(M_PI*p(1));
   if (dim >= 3) f(2) = sin(M_PI*p(2));
   if (dim == 4) f(3) = sin(M_PI*p(3));

   f *= (1.0+M_PI*M_PI);
}

void u_Func(const Vector &p, Vector &u)
{
   int dim = p.Size();

   u(0) = sin(M_PI*p(0));
   u(1) = sin(M_PI*p(1));
   if (dim >= 3) u(2) = sin(M_PI*p(2));
   if (dim == 4) u(3) = sin(M_PI*p(3));
}

void Hdivtest_fun(const Vector& xt, Vector& out )
{
    out.SetSize(xt.Size());

    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    out(0) = x;
    out(1) = 0.0;
    out(2) = 0.0;
    out(xt.Size()-1) = 0.;

}
