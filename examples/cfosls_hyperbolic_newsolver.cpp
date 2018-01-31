//
//                        MFEM CFOSLS Transport equation with multigrid (debugging & testing of a new multilevel solver)
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>
#include <unistd.h>

#define SERIALMESH


using namespace std;
using namespace mfem;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

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
    int par_ref_levels  = 1;

    const char *mesh_file = "../data/cube_3d_moderate.mesh";

    int feorder         = 0;

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

    Mesh *mesh = NULL;

    ParMesh* pmesh;

    if (nDimensions == 3 || nDimensions == 4)
    {
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

  
#ifdef SERIALMESH
    Mesh * serialmesh = NULL;
    ParMesh * serialpmesh;

        ifstream imesh2(mesh_file);
        if (!imesh2)
        {
            std::cerr << "\nCan not open mesh file: " << mesh_file << '\n' << std::endl;
            MPI_Finalize();
            return -2;
        }
        else
        {
            serialmesh = new Mesh(imesh2, 1, 1);
            imesh2.close();
        }
        for (int l = 0; l < ser_ref_levels; l++)
            serialmesh->UniformRefinement();
        for (int l = 0; l < par_ref_levels; l++)
            serialmesh->UniformRefinement();
        serialpmesh = new ParMesh(comm, *serialmesh);
        if (verbose)
            std::cout << "serialpmesh info \n" << std::flush;
        //serialpmesh->PrintInfo(std::cout);
        delete serialmesh;

#endif

    //MPI_Barrier(comm);
    //if (verbose)
       //std::cout << "Created serialpmesh \n" << std::flush;
    //MPI_Barrier(comm);

    
//#if 0
    if (mesh) // if only serial mesh was generated previously, parallel mesh is initialized here
    {
        {
            for (int l = 0; l < ser_ref_levels; l++)
                mesh->UniformRefinement();
        }
        
        pmesh = new ParMesh(comm, *mesh);
        delete mesh;
    }
//#endif

    MPI_Barrier(comm);
    if (verbose)
       std::cout << "Got here \n" << std::flush;
    MPI_Barrier(comm);

     MPI_Finalize();
     return 0;

}
