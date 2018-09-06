#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

#include "ls_temp.cpp"

// if active, no serial AMR in 4D is done
//#define ONLY_PAR_UR

// if defined, activates the Fichera corner test in 4D. The domain then will be constructed
// as tensor-extension of 3D Fichera corner mesh into 4D.
#define FICHERA_CORNER_SOLUTION

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
    using FormulType = CFOSLSFormulation_Laplace;
    using FEFormulType = CFOSLSFEFormulation_HdivH1L2_Laplace;
    using BdrCondsType = BdrConditions_CFOSLS_HdivH1Laplace;
    using ProblemType = FOSLSProblem_HdivH1lapl;

    // 1. Initialize MPI
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);
    bool verbose = (myid == 0);

    MPI_Comm comm_myid;
    MPI_Comm_split(comm, myid, 0, &comm_myid );

    StopWatch chrono;
    chrono.Clear();
    chrono.Start();
    
    // 2. Parse command-line options.

    //const char *mesh_file = "../data/cube4d_96.MFEM";
    const char *mesh_file = "../data/cube4d_24.MFEM";
    int order = 0;
    bool visualization = 0;
    int numofrefinement = 1;
#ifndef ONLY_PAR_UR
    //int maxdofs = 900000;
    double error_frac = .80;
    double betavalue = 0.1;
    int strat = 1;
#endif
    int numsol = 111;
    int prec_option = 1;

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree).");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
#ifndef ONLY_PAR_UR
    //args.AddOption(&maxdofs, "-r", "-refine","-r");
    args.AddOption(&strat, "-rs", "--refinementstrategy", "Which refinement strategy to implement for the LS Refiner");
    args.AddOption(&error_frac, "-ef","--errorfraction", "Weight in Dorfler Marking Strategy");
    args.AddOption(&betavalue, "-b","--beta", "Beta in the Difference Term of Estimator");
#endif

    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(cout);
        return 1;
    }
    args.PrintOptions(cout);

    // 2. Read the mesh from the given mesh file (or do some more steps in case of the Fichera corner test).
    Mesh * mesh;

#ifdef FICHERA_CORNER_SOLUTION
    if (verbose)
        std::cout << "Fichera corner solution \n";
    /*
    // for the Fichera corner, we create a mesh by mesh generator in 4D
    numsol = 1111;
    mesh_file = "../data/fichera_3d_coarse_-11.mesh";

    if (verbose)
        std::cout << "Extending 3D Fichera corner mesh into 4D \n";

    std::stringstream fname_fichera_temp;
    fname_fichera_temp << "mesh_file_fichera_temp.mesh";

    if (myid == 0)
    {
        Mesh *meshbase = new Mesh(mesh_file, 1, 1);
        ParMesh * pmeshbase = new ParMesh(comm, *meshbase);
        delete meshbase;

        int Nt              = 2;   // number of time slabs (e.g. Nsteps = 2 corresponds to 3 levels: t = 0, t = tau, t = 2 * tau
        double tau          = 0.5; // time step for a slab
        ParMesh * pmesh = new ParMeshCyl(comm_myid, *pmeshbase, 0.0, tau, Nt);
        delete pmeshbase;

        std::ofstream ofid(fname_fichera_temp.str().c_str());
        ofid.precision(8);
        pmesh->Print(ofid);
        delete pmesh;
    }

    std::ifstream ifid_fichera(fname_fichera_temp.str().c_str());
    mesh = new Mesh(ifid_fichera, 1, 1);
    */
    numsol = 1111;
    mesh_file = "../data/fichera_4d_cylinder.mesh";
    mesh = new Mesh(mesh_file, 1, 1);
#else
    mesh = new Mesh(mesh_file, 1, 1);
#endif
    
    int dim = mesh->Dimension();

    for (int l = 0; l < numofrefinement; l++)
        mesh->UniformRefinement();

    // 3. Define weak f.e. formulation for the problem at hand
    // and create FOSLSProblem on top of them
    FormulType * formulat = new FormulType (dim, numsol, verbose);
    FEFormulType * fe_formulat = new FEFormulType(*formulat, order);

#ifndef ONLY_PAR_UR
    // Perform adaptive refinement in serial

    if (myid == 0)
    {
        ParMesh * pmesh = new ParMesh(comm_myid, *mesh);
        pmesh->PrintInfo(std::cout); if(verbose) cout << endl;

        BdrConditions * bdr_conds = new BdrCondsType(*pmesh);

        ProblemType * problem = new ProblemType (*pmesh, *bdr_conds, *fe_formulat,
                                                 prec_option, verbose);

        // 4. Creating the estimator
        int numfoslsfuns = -1;

        int fosls_func_version = 1;
        if (verbose)
         std::cout << "fosls_func_version = " << fosls_func_version << "\n";

        if (fosls_func_version == 1)
        {
            numfoslsfuns = 1;
            ++numfoslsfuns;
        }

        int numblocks_funct = 1;
        ++numblocks_funct;

        std::vector<std::pair<int,int> > grfuns_descriptor(numfoslsfuns);

        Array2D<BilinearFormIntegrator *> integs(numfoslsfuns, numfoslsfuns);
        for (int i = 0; i < integs.NumRows(); ++i)
            for (int j = 0; j < integs.NumCols(); ++j)
                integs(i,j) = NULL;

        // version 1, only || sigma + grad S ||^2, or || sigma ||^2
        if (fosls_func_version == 1)
        {
            // this works
            grfuns_descriptor[0] = std::make_pair<int,int>(1, 0);
            integs(0,0) = new VectorFEMassIntegrator;

            grfuns_descriptor[1] = std::make_pair<int,int>(1, 1);
            integs(1,1) = new DiffusionIntegrator;
            integs(0,1) = new MixedVectorGradientIntegrator;
        }
        else
        {
            MFEM_ABORT("Unsupported version of fosls functional \n");
        }

        FOSLSEstimator * estimator;
        estimator = new FOSLSEstimator(*problem, grfuns_descriptor, NULL, integs, verbose);
        problem->AddEstimator(*estimator);

        // cannot work due to the absence of LcoalRefinement in 4D for MFEM meshes
        //ThresholdRefiner refiner(*estimator);
        //refiner.SetTotalErrorFraction(0.5);

        NDLSRefiner * refiner = new NDLSRefiner(*estimator);
        refiner->SetTotalErrorFraction(error_frac);
        refiner->SetTotalErrorNormP(2.0);
        refiner->SetRefinementStrategy(strat);
        refiner->SetBetaCalc(0);
        refiner->SetBetaConstants(betavalue);
        refiner->version_difference = false;

        std::cout << "#dofs for the first solve: " << problem->GlobalTrueProblemSize() << "\n";
        problem->Solve(verbose, true);

        int global_dofs;
        int max_dofs = 300000;
        int max_amr_iter = 10;

        for (int it = 0; it < max_amr_iter; ++it)
        {
            if (verbose)
            {
               cout << "\nAMR iteration " << it << "\n";
               cout << "Refining the mesh ... \n";
            }

            refiner->Apply(*mesh);
            if(refiner->Stop())
            {
                cout<< "Maximum number of dofs has been reached \n";
                break;
            }

            // cannot use Update() because CoarseToFine transformations are not generated
            // through MARS wrapper right now
            //problem->Update();
            //problem->BuildSystem(verbose);

            delete estimator;
            delete problem;

            mesh->AllocateSwappedElements();
            Mesh * mesh_copy = new Mesh(*mesh);
            delete mesh;
            mesh = mesh_copy;

            ParMesh * pmesh_copy = new ParMesh(comm_myid, *mesh);
            delete pmesh;
            pmesh = pmesh_copy;

            pmesh->PrintInfo(std::cout); if(verbose) cout << endl;

            problem = new ProblemType (*pmesh, *bdr_conds, *fe_formulat,
                                                     prec_option, verbose);

            estimator = new FOSLSEstimator(*problem, grfuns_descriptor, NULL, integs, verbose);
            problem->AddEstimator(*estimator);

            delete refiner;
            refiner = new NDLSRefiner(*estimator);
            refiner->SetTotalErrorFraction(error_frac);
            refiner->SetTotalErrorNormP(2.0);
            refiner->SetRefinementStrategy(strat);
            refiner->SetBetaCalc(0);
            refiner->SetBetaConstants(betavalue);
            refiner->version_difference = false;

            global_dofs = problem->GlobalTrueProblemSize();
            if (global_dofs > max_dofs)
            {
               if (verbose)
                  cout << "Reached the maximum number of dofs, current problem "
                          "#dofs = " << global_dofs << ". Stop. \n";
               break;
            }

            if (verbose)
            {
               cout << "\nAMR iteration " << it << "\n";
               cout << "Number of unknowns: " << global_dofs << "\n";
            }
            problem->Solve(verbose, true);

        }

        delete problem;
        delete refiner;
        delete estimator;

        delete bdr_conds;
        delete pmesh;
    }

    // Have to print and (later) read the mesh from file, because it's broekn as Mesh after 4D serial AMR through MARS
    std::stringstream fname;
    fname << "mesh_file_temp.MFEM";

    if (myid == 0)
    {
        std::ofstream ofid(fname.str().c_str());
        ofid.precision(8);
        mesh->Print(ofid);
        delete mesh;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //mesh = new Mesh("../data/cube4d_96.MFEM", 1, 1);
    std::ifstream ifid(fname.str().c_str());
    mesh = new Mesh(ifid, 1, 1);
#endif

    // Perform uniform refinement in parallel
    if (verbose)
        std::cout << "Parallel uniform refinement stage \n";
    int nprefs = 2;

    //std::cout << "Got here 0 \n" << std::flush;
    //MPI_Barrier(MPI_COMM_WORLD);
    ParMesh * pmesh = new ParMesh(comm, *mesh);
    //std::cout << "Got here 1 \n" << std::flush;
    //MPI_Barrier(MPI_COMM_WORLD);
    delete mesh;
    BdrConditions * bdr_conds = new BdrCondsType(*pmesh);
    pmesh->PrintInfo(std::cout); if(verbose) cout << endl;

    ProblemType * problem = new ProblemType (*pmesh, *bdr_conds, *fe_formulat,
                                             prec_option, verbose);
    int global_dofs;
    int max_dofs_prefs = 1600000;

    for (int pref = 0; pref < nprefs; ++pref)
    {
        global_dofs = problem->GlobalTrueProblemSize();
        if (verbose)
        {
           cout << "\nUR iteration " << pref << "\n";
           cout << "Number of unknowns: " << global_dofs << "\n";
        }
        if (global_dofs > max_dofs_prefs)
        {
           if (verbose)
              cout << "Global #dofs: " << global_dofs << ". Reached the maximum number of dofs. Stop. \n";
           break;
        }

        problem->Solve(verbose, true);

        if (pref < nprefs - 1)
        {
            pmesh->UniformRefinement();
            problem->Update();
            problem->BuildSystem(verbose);
        }

    }

    delete fe_formulat;
    delete formulat;

    delete problem;
    delete bdr_conds;
    delete pmesh;

    MPI_Finalize();
    return 0;
}



