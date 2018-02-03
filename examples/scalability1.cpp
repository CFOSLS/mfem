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
    int numsol          = 4;

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
    args.AddOption(&nDimensions, "-dim", "--whichD",
                   "Dimension of the space-time problem.");
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

    StopWatch chrono;

    Mesh *mesh = NULL;

    ParMesh * pmesh;

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
        pmesh = new ParMesh(comm, *mesh);
        delete mesh;
    }

    int dim = nDimensions;

    Array<int> ess_bdrSigma(pmesh->bdr_attributes.Max());
    ess_bdrSigma = 0;

    Array<int> ess_bdrS(pmesh->bdr_attributes.Max());
    ess_bdrS = 1;

    Array<int> all_bdrSigma(pmesh->bdr_attributes.Max());
    all_bdrSigma = 1;

    Array<int> all_bdrS(pmesh->bdr_attributes.Max());
    all_bdrS = 1;

    int ref_levels = par_ref_levels;

    int num_levels = ref_levels + 1;

    chrono.Clear();
    chrono.Start();

    Array<ParFiniteElementSpace*> R_space_lvls(num_levels);
    Array<ParFiniteElementSpace*> H_space_lvls(num_levels);

    FiniteElementCollection *hdiv_coll;
    ParFiniteElementSpace *R_space;

    if (dim == 4)
        hdiv_coll = new RT0_4DFECollection;
    else
        hdiv_coll = new RT_FECollection(feorder, dim);

    R_space = new ParFiniteElementSpace(pmesh, hdiv_coll);

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
    H_space = new ParFiniteElementSpace(pmesh, h1_coll);

    int numblocks_funct = 1;
    numblocks_funct++;

    //std::cout << "num_levels - 1 = " << num_levels << "\n";

    std::vector<std::vector<HypreParMatrix*> > Dof_TrueDof_Func_lvls(num_levels);

    BlockOperator* Funct_global;
    std::vector<Operator*> Funct_global_lvls(num_levels);
    Array<int> offsets_global(numblocks_funct + 1);

   for (int l = 0; l < num_levels; ++l)
   {
       Dof_TrueDof_Func_lvls[l].resize(numblocks_funct);
   }

    if (verbose)
        std::cout << "Creating a hierarchy of meshes by successive refinements "
                     "(with multilevel and multigrid prerequisites) \n";

    for (int l = num_levels - 1; l >= 0; --l)
        pmesh->UniformRefinement();

    pmesh->PrintInfo(std::cout);
    if (verbose)
        std::cout << "\n";

    int l = 0;
    // creating pfespaces for level l
    R_space_lvls[l] = new ParFiniteElementSpace(pmesh, hdiv_coll);
    H_space_lvls[l] = new ParFiniteElementSpace(pmesh, h1_coll);

    ParBilinearForm *Ablock(new ParBilinearForm(R_space_lvls[l]));
    Ablock->AddDomainIntegrator(new VectorFEMassIntegrator);
    Ablock->Assemble();
    Ablock->EliminateEssentialBC(ess_bdrSigma);//, *sigma_exact_finest, *fform); // makes res for sigma_special happier
    Ablock->Finalize();

    // getting pointers to dof_truedof matrices

    Dof_TrueDof_Func_lvls[l][0] = R_space_lvls[l]->Dof_TrueDof_Matrix();
    Dof_TrueDof_Func_lvls[l][1] = H_space_lvls[l]->Dof_TrueDof_Matrix();

    ParBilinearForm *Cblock;
    ParMixedBilinearForm *Bblock;
    Cblock = new ParBilinearForm(H_space_lvls[l]);
    Bblock = new ParMixedBilinearForm(H_space_lvls[l], R_space_lvls[l]);


    // Creating global functional matrix
    offsets_global[0] = 0;
    for ( int blk = 0; blk < numblocks_funct; ++blk)
        offsets_global[blk + 1] = Dof_TrueDof_Func_lvls[l][blk]->Width();
    offsets_global.PartialSum();

    Funct_global = new BlockOperator(offsets_global);

    Ablock->Assemble();
    Ablock->EliminateEssentialBC(ess_bdrSigma);//, *sigma_exact_finest, *fform); // makes res for sigma_special happier
    Ablock->Finalize();
    Funct_global->SetBlock(0,0, Ablock->ParallelAssemble());

    Cblock->Assemble();
    {
        Vector temp1(Cblock->Width());
        temp1 = 0.0;
        Vector temp2(Cblock->Height());
        temp2 = 0.0;
        Cblock->EliminateEssentialBC(ess_bdrS, temp1, temp2);
    }
    Cblock->Finalize();
    Funct_global->SetBlock(1,1, Cblock->ParallelAssemble());
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
    Funct_global->SetBlock(0,1, B);
    Funct_global->SetBlock(1,0, B->Transpose());

    Funct_global_lvls[0] = Funct_global;

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

            StopWatch chrono_debug;

            MPI_Barrier(comm);
            chrono_debug.Clear();
            chrono_debug.Start();
            for (int it = 0; it < 20; ++it)
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
            for (int it = 0; it < 20; ++it)
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

