//
//                        MFEM CFOSLS Poisson equation with multigrid (debugging & testing of a new multilevel solver)
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>
#include <unistd.h>

//#define VISUALIZATION

#define TIMING

#define MYZEROTOL (1.0e-13)

using namespace std;
using namespace mfem;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

void ExternalUpdateResImitation(Operator& oper, double coeff, const Vector* rhs_l, const Vector& x_l, Vector &out_l)
{
    oper.Mult(x_l, out_l);
    out_l *= coeff;

    if (rhs_l)
        out_l += *rhs_l;
}

int main(int argc, char *argv[])
{
    int num_procs, myid;
    bool visualization = 1;

    // 1. Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    bool verbose = (myid == 0);

    int nDimensions     = 3;

    int ser_ref_levels  = 1;
    int par_ref_levels  = 2;

    const char *space_for_S = "H1";    // "H1" or "L2"

    //const char *mesh_file = "../data/cube_3d_fine.mesh";
    const char *mesh_file = "../data/cube_3d_moderate.mesh";
    //const char *mesh_file = "../data/square_2d_moderate.mesh";

    int feorder         = 0;

    if (verbose)
        cout << "Scalability issue 1 reproducer \n";

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&feorder, "-o", "--feorder",
                   "Finite element order (polynomial degree).");
    args.AddOption(&ser_ref_levels, "-sref", "--sref",
                   "Number of serial refinements 4d mesh.");
    args.AddOption(&par_ref_levels, "-pref", "--pref",
                   "Number of parallel refinements 4d mesh.");
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

    std::cout << std::flush;
    MPI_Barrier(MPI_COMM_WORLD);

    MFEM_ASSERT(strcmp(space_for_S,"H1") == 0, "Space for S must be H1 for the laplace equation!\n");

    if (verbose)
        std::cout << "Space for S: H1 \n";

    if (verbose)
        std::cout << "Running tests for the paper: \n";

    if (verbose)
        cout << "Number of mpi processes: " << num_procs << endl << flush;

    StopWatch chrono;

    Mesh *mesh = NULL;

    ParMesh * pmesh;

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

        if (verbose)
            cout << "Creating parmesh(" << nDimensions <<
                    "d) from the serial mesh (" << nDimensions << "d)" << endl << flush;
        pmesh = new ParMesh(comm, *mesh);
        delete mesh;
    }

    int dim = nDimensions;

    Array<int> ess_bdrSigma(pmesh->bdr_attributes.Max());
    ess_bdrSigma = 0;

    Array<int> ess_bdrS(pmesh->bdr_attributes.Max());
    ess_bdrS = 1;

    int num_levels = par_ref_levels + 1;

    chrono.Clear();
    chrono.Start();

    FiniteElementCollection *hdiv_coll = new RT_FECollection(feorder, dim);
    FiniteElementCollection *h1_coll = new H1_FECollection(feorder+1, nDimensions);

    BlockOperator* Funct_global;
    std::vector<Operator*> Funct_global_lvls(num_levels);

    if (verbose)
        std::cout << "Creating a hierarchy of meshes by successive refinements "
                     "(with multilevel and multigrid prerequisites) \n";

    for (int l = num_levels - 1; l > 0; --l)
        pmesh->UniformRefinement();

    pmesh->PrintInfo(std::cout);
    if (verbose)
        std::cout << "\n";

    int l = 0;

    ParFiniteElementSpace * R_space = new ParFiniteElementSpace(pmesh, hdiv_coll);
    ParFiniteElementSpace * H_space = new ParFiniteElementSpace(pmesh, h1_coll);

    ParBilinearForm *Ablock(new ParBilinearForm(R_space));
    Ablock->AddDomainIntegrator(new VectorFEMassIntegrator);
    Ablock->Assemble();
    Ablock->EliminateEssentialBC(ess_bdrSigma);
    Ablock->Finalize();

    ParBilinearForm *Cblock;
    ParMixedBilinearForm *Bblock;
    Cblock = new ParBilinearForm(H_space);
    Bblock = new ParMixedBilinearForm(H_space, R_space);


    // Creating global functional matrix
    Array<int> offsets_global(3);
    offsets_global[0] = 0;
    offsets_global[1] = R_space->GetTrueVSize();
    offsets_global[2] = H_space->GetTrueVSize();
    offsets_global.PartialSum();

    if (verbose)
        offsets_global.Print();

    Funct_global = new BlockOperator(offsets_global);

    Ablock->Assemble();
    Ablock->EliminateEssentialBC(ess_bdrSigma);
    Ablock->Finalize();
    HypreParMatrix * A = Ablock->ParallelAssemble();
    Funct_global->SetBlock(0,0, A);

    Cblock->Assemble();
    {
        Vector temp1(Cblock->Width());
        temp1 = 0.0;
        Vector temp2(Cblock->Height());
        temp2 = 0.0;
        Cblock->EliminateEssentialBC(ess_bdrS, temp1, temp2);
    }
    Cblock->Finalize();
    HypreParMatrix * C = Cblock->ParallelAssemble();
    Funct_global->SetBlock(1,1, C);
    Bblock->Assemble();
    {
        Vector temp1(Bblock->Width());
        temp1 = 0.0;
        Vector temp2(Bblock->Height());
        temp2 = 0.0;
        Bblock->EliminateTrialDofs(ess_bdrS, temp1, temp2);
        Bblock->EliminateTestDofs(ess_bdrSigma);
    }
    Bblock->Finalize();
    HypreParMatrix * B = Bblock->ParallelAssemble();
    HypreParMatrix * BT = B->Transpose();

    Funct_global->SetBlock(0,1, B);
    Funct_global->SetBlock(1,0, B->Transpose());

    Funct_global_lvls[0] = Funct_global;

    // Looking at global and local nnz in the funct
    MPI_Barrier(comm);
    if (verbose)
    {
        int global_nnz = A->NNZ() + C->NNZ() + B->NNZ() + BT->NNZ();
        std::cout << "Global nnz in Funct = " << global_nnz << "\n" << std::flush;
    }
    MPI_Barrier(comm);

    for (int i = 0; i < num_procs; ++i)
    {
        if (myid == i)
        {
            std::cout << "I am " << myid << "\n";
            std::cout << "Look at my local NNZ in Funct: \n";
            int local_nnz = 0;

            SparseMatrix diag;
            SparseMatrix offd;
            HYPRE_Int * cmap;

            A->GetDiag(diag);
            local_nnz += diag.NumNonZeroElems();
            A->GetOffd(offd, cmap);
            local_nnz += offd.NumNonZeroElems();

            C->GetDiag(diag);
            local_nnz += diag.NumNonZeroElems();
            C->GetOffd(offd, cmap);
            local_nnz += offd.NumNonZeroElems();

            B->GetDiag(diag);
            local_nnz += diag.NumNonZeroElems();
            B->GetOffd(offd, cmap);
            local_nnz += offd.NumNonZeroElems();

            BT->GetDiag(diag);
            local_nnz += diag.NumNonZeroElems();
            BT->GetOffd(offd, cmap);
            local_nnz += offd.NumNonZeroElems();

            std::cout << "local nnz in Funct = " << local_nnz << "\n";
            std::cout << "\n" << std::flush;
        }
        MPI_Barrier(comm);
    }
    MPI_Barrier(comm);


#ifdef TIMING
    // testing Functional action as operator timing with an external imitating routine
    for (int l = 0; l < num_levels - 1; ++l)
    {
        if (l == 0)
        {
            Vector testRhs(Funct_global_lvls[l]->Height());
            testRhs = 1.0;
            Vector testX(Funct_global_lvls[l]->Width());
            testX = 0.0;

            Vector testsuppl(Funct_global_lvls[l]->Height());
            testsuppl.Randomize(2000);

            StopWatch chrono_debug;

            MPI_Barrier(comm);
            chrono_debug.Clear();
            chrono_debug.Start();
            for (int it = 0; it < 100; ++it)
            {
                ExternalUpdateResImitation(*Funct_global_lvls[l], -1.0, &testsuppl, testRhs, testX);
                testRhs += testX;
            }

            MPI_Barrier(comm);
            chrono_debug.Stop();

            if (verbose)
               std::cout << "UpdateRes imitating routine at level " << l << "  has finished in " << chrono_debug.RealTime() << " \n" << std::flush;

            MPI_Barrier(comm);

        }

    }

    // testing Functional action as operator timing

    for (int l = 0; l < num_levels - 1; ++l)
    {
        if (l == 0)
        {
            StopWatch chrono_debug;

            Vector testRhs(Funct_global_lvls[l]->Height());
            testRhs = 1.0;
            Vector testX(Funct_global_lvls[l]->Width());
            testX = 0.0;

            MPI_Barrier(comm);
            chrono_debug.Clear();
            chrono_debug.Start();
            for (int it = 0; it < 100; ++it)
            {
                Funct_global_lvls[l]->Mult(testRhs, testX);
                testRhs += testX;
            }

            MPI_Barrier(comm);
            chrono_debug.Stop();

            if (verbose)
               std::cout << "Funct action at level " << l << "  has finished in " << chrono_debug.RealTime() << " \n" << std::flush;

            MPI_Barrier(comm);

        }
    }
#endif

    MPI_Finalize();
    return 0;
}

