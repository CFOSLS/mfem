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

#define VISUALIZATION

// (de)activates solving of the discrete global problem
#define OLD_CODE

#define WITH_DIVCONSTRAINT_SOLVER

// switches on/off usage of smoother in the new minimization solver
// in parallel GS smoother works a little bit different from serial
#define WITH_SMOOTHERS

// activates a check for the symmetry of the new smoother setup
//#define CHECK_SPDSMOOTHER

// activates using the new interface to local problem solvers
// via a separated class called LocalProblemSolver
#define SOLVE_WITH_LOCALSOLVERS

// activates a test where new solver is used as a preconditioner
#define USE_AS_A_PREC

#define HCURL_COARSESOLVER

// activates a check for the symmetry of the new solver
//#define CHECK_SPDSOLVER

// activates constraint residual check after each iteration of the minimization solver
#define CHECK_CONSTR

#define CHECK_BNDCND

//#define HCURL_MG_TEST

#define BND_FOR_MULTIGRID
//#define BLKDIAG_SMOOTHER

#define MINSOLVER_TESTING

//#define TIMING

#ifdef TIMING
#undef CHECK_LOCALSOLVE
#undef CHECK_CONSTR
#undef CHECK_BNDCND
#endif

#define MYZEROTOL (1.0e-13)

// must be always active
#define USE_CURLMATRIX


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
    int numsol          = 11;

    int ser_ref_levels  = 1;
    int par_ref_levels  = 1;

    const char *space_for_S = "H1";    // "H1" or "L2"
    bool eliminateS = true;            // in case space_for_S = "L2" defines whether we eliminate S from the system

    bool aniso_refine = false;
    bool refine_t_first = false;

    bool with_multilevel = true;
    bool monolithicMG = false;

    bool useM_in_divpart = true;

    // solver options
    int prec_option = 3;        // defines whether to use preconditioner or not, and which one
    bool prec_is_MG;

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
        cout << "Solving CFOSLS Poisson equation with MFEM & hypre, div-free approach, minimization solver \n";

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
    args.AddOption(&eliminateS, "-elims", "--eliminateS", "-no-elims",
                   "--no-eliminateS",
                   "Turn on/off elimination of S in L2 formulation.");
    args.AddOption(&prec_option, "-precopt", "--prec-option",
                   "Preconditioner choice.");
    args.AddOption(&with_multilevel, "-ml", "--multilvl", "-no-ml",
                   "--no-multilvl",
                   "Enable or disable multilevel algorithm for finding a particular solution.");
    args.AddOption(&useM_in_divpart, "-useM", "--useM", "-no-useM", "--no-useM",
                   "Whether to use M to compute a partilar solution");
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

#ifdef WITH_SMOOTHERS
    if (verbose)
        std::cout << "WITH_SMOOTHERS active \n";
#else
    if (verbose)
        std::cout << "WITH_SMOOTHERS passive \n";
#endif

#ifdef SOLVE_WITH_LOCALSOLVERS
    if (verbose)
        std::cout << "SOLVE_WITH_LOCALSOLVERS active \n";
#else
    if (verbose)
        std::cout << "SOLVE_WITH_LOCALSOLVERS passive \n";
#endif

#ifdef HCURL_COARSESOLVER
    if (verbose)
        std::cout << "HCURL_COARSESOLVER active \n";
#else
    if (verbose)
        std::cout << "HCURL_COARSESOLVER passive \n";
#endif

#ifdef USE_AS_A_PREC
    if (verbose)
        std::cout << "USE_AS_A_PREC active \n";
#else
    if (verbose)
        std::cout << "USE_AS_A_PREC passive \n";
#endif

#ifdef OLD_CODE
    if (verbose)
        std::cout << "OLD_CODE active \n";
#else
    if (verbose)
        std::cout << "OLD_CODE passive \n";
#endif
#ifdef TIMING
    if (verbose)
        std::cout << "TIMING active \n";
#else
    if (verbose)
        std::cout << "TIMING passive \n";
#endif
#ifdef CHECK_BNDCND
    if (verbose)
        std::cout << "CHECK_BNDCND active \n";
#else
    if (verbose)
        std::cout << "CHECK_BNDCND passive \n";
#endif
#ifdef CHECK_CONSTR
    if (verbose)
        std::cout << "CHECK_CONSTR active \n";
#else
    if (verbose)
        std::cout << "CHECK_CONSTR passive \n";
#endif

#ifdef BND_FOR_MULTIGRID
    if (verbose)
        std::cout << "BND_FOR_MULTIGRID active \n";
#else
    if (verbose)
        std::cout << "BND_FOR_MULTIGRID passive \n";
#endif

#ifdef BLKDIAG_SMOOTHER
    if (verbose)
        std::cout << "BLKDIAG_SMOOTHER active \n";
#else
    if (verbose)
        std::cout << "BLKDIAG_SMOOTHER passive \n";
#endif

    std::cout << std::flush;
    MPI_Barrier(MPI_COMM_WORLD);

    MFEM_ASSERT(strcmp(space_for_S,"H1") == 0, "Space for S must be H1 for the laplace equation!\n");

    if (verbose)
        std::cout << "Space for S: H1 \n";

    if (verbose)
        std::cout << "Running tests for the paper: \n";

    if (numsol == -3 || numsol == 3)
    {
        mesh_file = "../data/cube_3d_moderate.mesh";
    }

    if (numsol == -4 || numsol == 4)
    {
        mesh_file = "../data/cube4d_96.MFEM";
    }

    if (numsol == 11)
    {
        mesh_file = "../data/netgen_lshape3D_onemoretry.netgen";
    }

    if (verbose)
        std::cout << "For the records: numsol = " << numsol
                  << ", mesh_file = " << mesh_file << "\n";

    Laplace_test Mytest(nDimensions, numsol);

    ConstantCoefficient zerocoeff(0.0);

    if (verbose)
        cout << "Number of mpi processes: " << num_procs << endl << flush;

    bool with_prec;

    switch (prec_option)
    {
    case 1: // smth simple like AMS
        with_prec = true;
        prec_is_MG = false;
        break;
    case 2: // MG
        with_prec = true;
        prec_is_MG = true;
        monolithicMG = false;
        break;
    case 3: // block MG
        with_prec = true;
        prec_is_MG = true;
        monolithicMG = true;
        break;
    default: // no preconditioner (default)
        with_prec = false;
        prec_is_MG = false;
        monolithicMG = false;
        break;
    }

    if (verbose)
    {
        cout << "with_prec = " << with_prec << endl;
        cout << "prec_is_MG = " << prec_is_MG << endl;
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

    /*
    std::stringstream fname;
    fname << "netgen_lshape3D.mesh";
    std::ofstream ofid(fname.str().c_str());
    ofid.precision(8);
    pmesh->Print(ofid);
    */

    MFEM_ASSERT(!(aniso_refine && (with_multilevel || nDimensions == 4)),"Anisotropic refinement works only in 3D and without multilevel algorithm \n");

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

    //sleep(10);

    Array<ParMesh*> pmesh_lvls(num_levels);
    Array<ParFiniteElementSpace*> R_space_lvls(num_levels);
    Array<ParFiniteElementSpace*> W_space_lvls(num_levels);
    Array<ParFiniteElementSpace*> C_space_lvls(num_levels);
    Array<ParFiniteElementSpace*> H_space_lvls(num_levels);

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

#ifdef OLD_CODE
    ParFiniteElementSpace * S_space;
    if (strcmp(space_for_S,"H1") == 0)
        S_space = H_space;
    else // "L2"
        S_space = W_space;
#endif

    // For geometric multigrid
    Array<HypreParMatrix*> TrueP_C(par_ref_levels);
    Array<HypreParMatrix*> TrueP_H(par_ref_levels);

    Array<HypreParMatrix*> TrueP_R(par_ref_levels);

    Array< SparseMatrix*> P_W(ref_levels);
    Array< SparseMatrix*> P_R(ref_levels);
    //Array< SparseMatrix*> P_H(ref_levels);
    Array< SparseMatrix*> Element_dofs_R(ref_levels);
    Array< SparseMatrix*> Element_dofs_H(ref_levels);
    Array< SparseMatrix*> Element_dofs_W(ref_levels);

    Array<SparseMatrix*> Mass_mat_lvls(num_levels);

    const SparseMatrix* P_W_local;
    const SparseMatrix* P_R_local;
    const SparseMatrix* P_H_local;

    DivPart divp;

    int numblocks_funct = 1;
    if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
        numblocks_funct++;
    std::vector<std::vector<Array<int>* > > BdrDofs_Funct_lvls(num_levels, std::vector<Array<int>* >(numblocks_funct));
    std::vector<std::vector<Array<int>* > > EssBdrDofs_Funct_lvls(num_levels, std::vector<Array<int>* >(numblocks_funct));
    std::vector<std::vector<Array<int>* > > EssBdrTrueDofs_Funct_lvls(num_levels, std::vector<Array<int>* >(numblocks_funct));

#ifdef OLD_CODE
    std::vector<std::vector<Array<int>* > > EssBdrTrueDofs_HcurlFunct_lvls(num_levels, std::vector<Array<int>* >(numblocks_funct));
#endif

    Array< SparseMatrix* > P_C_lvls(num_levels - 1);
#ifdef HCURL_COARSESOLVER
    Array<HypreParMatrix* > Dof_TrueDof_Hcurl_lvls(num_levels);
    std::vector<Array<int>* > EssBdrDofs_Hcurl(num_levels);
    std::vector<Array<int>* > EssBdrTrueDofs_Hcurl(num_levels);
    std::vector<Array<int>* > EssBdrTrueDofs_H1(num_levels);
#else
    Array<HypreParMatrix* > Dof_TrueDof_Hcurl_lvls(num_levels - 1);
    std::vector<Array<int>* > EssBdrDofs_Hcurl(num_levels - 1); // FIXME: Proably, minus 1 for all Hcurl entries?
    std::vector<Array<int>* > EssBdrTrueDofs_Hcurl(num_levels - 1);
    std::vector<Array<int>* > EssBdrTrueDofs_H1(num_levels - 1);
#endif

    std::vector<Array<int>* > EssBdrDofs_H1(num_levels);
    Array< SparseMatrix* > P_H_lvls(num_levels - 1);
    Array<HypreParMatrix* > Dof_TrueDof_H1_lvls(num_levels);
    Array<HypreParMatrix* > Dof_TrueDof_Hdiv_lvls(num_levels);

    std::vector<std::vector<HypreParMatrix*> > Dof_TrueDof_Func_lvls(num_levels);
    std::vector<HypreParMatrix*> Dof_TrueDof_L2_lvls(num_levels);

    Array<SparseMatrix*> Divfree_mat_lvls(num_levels);
    std::vector<Array<int>*> Funct_mat_offsets_lvls(num_levels);
    Array<BlockMatrix*> Funct_mat_lvls(num_levels);
    Array<SparseMatrix*> Constraint_mat_lvls(num_levels);

    Array<HypreParMatrix*> Divfree_hpmat_mod_lvls(num_levels);
    std::vector<Array2D<HypreParMatrix*> *> Funct_hpmat_lvls(num_levels);

    BlockOperator* Funct_global;
    std::vector<Operator*> Funct_global_lvls(num_levels);
    BlockVector* Functrhs_global;
    Array<int> offsets_global(numblocks_funct + 1);

   for (int l = 0; l < num_levels; ++l)
   {
       Dof_TrueDof_Func_lvls[l].resize(numblocks_funct);
       BdrDofs_Funct_lvls[l][0] = new Array<int>;
       EssBdrDofs_Funct_lvls[l][0] = new Array<int>;
       EssBdrTrueDofs_Funct_lvls[l][0] = new Array<int>;
       EssBdrTrueDofs_HcurlFunct_lvls[l][0] = new Array<int>;
#ifndef HCURL_COARSESOLVER
       if (l < num_levels - 1)
       {
           EssBdrDofs_Hcurl[l] = new Array<int>;
           EssBdrTrueDofs_Hcurl[l] = new Array<int>;
           EssBdrTrueDofs_H1[l] = new Array<int>;
       }
#else
       EssBdrDofs_Hcurl[l] = new Array<int>;
       EssBdrTrueDofs_Hcurl[l] = new Array<int>;
       EssBdrTrueDofs_H1[l] = new Array<int>;
#endif
       Funct_mat_offsets_lvls[l] = new Array<int>;
       if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
       {
           BdrDofs_Funct_lvls[l][1] = new Array<int>;
           EssBdrDofs_Funct_lvls[l][1] = new Array<int>;
           EssBdrTrueDofs_Funct_lvls[l][1] = new Array<int>;
           EssBdrTrueDofs_HcurlFunct_lvls[l][1] = new Array<int>;
           EssBdrDofs_H1[l] = new Array<int>;
       }

       Funct_hpmat_lvls[l] = new Array2D<HypreParMatrix*>(numblocks_funct, numblocks_funct);
   }

   const SparseMatrix* P_C_local;

   //Actually this and LocalSolver_partfinder_lvls handle the same objects
   Array<Operator*>* LocalSolver_lvls;
   LocalSolver_lvls = new Array<Operator*>(num_levels - 1);

   Array<LocalProblemSolver*>* LocalSolver_partfinder_lvls;
   LocalSolver_partfinder_lvls = new Array<LocalProblemSolver*>(num_levels - 1);

   Array<Operator*> Smoothers_lvls(num_levels - 1);


   Operator* CoarsestSolver;
   CoarsestProblemSolver* CoarsestSolver_partfinder;

   Array<BlockMatrix*> Element_dofs_Func(num_levels - 1);
   std::vector<Array<int>*> row_offsets_El_dofs(num_levels - 1);
   std::vector<Array<int>*> col_offsets_El_dofs(num_levels - 1);

   Array<BlockMatrix*> P_Func(ref_levels);
   std::vector<Array<int>*> row_offsets_P_Func(num_levels - 1);
   std::vector<Array<int>*> col_offsets_P_Func(num_levels - 1);

   Array<BlockOperator*> TrueP_Func(ref_levels);
   std::vector<Array<int>*> row_offsets_TrueP_Func(num_levels - 1);
   std::vector<Array<int>*> col_offsets_TrueP_Func(num_levels - 1);

   for (int l = 0; l < num_levels; ++l)
       if (l < num_levels - 1)
       {
           row_offsets_El_dofs[l] = new Array<int>(numblocks_funct + 1);
           col_offsets_El_dofs[l] = new Array<int>(numblocks_funct + 1);
           row_offsets_P_Func[l] = new Array<int>(numblocks_funct + 1);
           col_offsets_P_Func[l] = new Array<int>(numblocks_funct + 1);
           row_offsets_TrueP_Func[l] = new Array<int>(numblocks_funct + 1);
           col_offsets_TrueP_Func[l] = new Array<int>(numblocks_funct + 1);
       }

   Array<SparseMatrix*> P_WT(num_levels - 1); //AE_e matrices

    chrono.Clear();
    chrono.Start();

    if (verbose)
        std::cout << "Creating a hierarchy of meshes by successive refinements "
                     "(with multilevel and multigrid prerequisites) \n";

    for (int l = num_levels - 1; l >= 0; --l)
    {
        // creating pmesh for level l
        if (l == num_levels - 1)
        {
            pmesh_lvls[l] = new ParMesh(*pmesh);
            //pmesh_lvls[l] = pmesh.get();
        }
        else
        {
            if (aniso_refine && refine_t_first)
            {
                Array<Refinement> refs(pmesh->GetNE());
                if (l < par_ref_levels/2+1)
                {
                    for (int i = 0; i < pmesh->GetNE(); i++)
                        refs[i] = Refinement(i, 4);
                }
                else
                {
                    for (int i = 0; i < pmesh->GetNE(); i++)
                        refs[i] = Refinement(i, 3);
                }
                pmesh->GeneralRefinement(refs, -1, -1);
            }
            else if (aniso_refine && !refine_t_first)
            {
                Array<Refinement> refs(pmesh->GetNE());
                if (l < par_ref_levels/2+1)
                {
                    for (int i = 0; i < pmesh->GetNE(); i++)
                        refs[i] = Refinement(i, 3);
                }
                else
                {
                    for (int i = 0; i < pmesh->GetNE(); i++)
                        refs[i] = Refinement(i, 4);
                }
                pmesh->GeneralRefinement(refs, -1, -1);
            }
            else
            {
                pmesh->UniformRefinement();
            }

            pmesh_lvls[l] = new ParMesh(*pmesh);
        }

        // creating pfespaces for level l
        R_space_lvls[l] = new ParFiniteElementSpace(pmesh_lvls[l], hdiv_coll);
        W_space_lvls[l] = new ParFiniteElementSpace(pmesh_lvls[l], l2_coll);
        C_space_lvls[l] = new ParFiniteElementSpace(pmesh_lvls[l], hdivfree_coll);
        if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
            H_space_lvls[l] = new ParFiniteElementSpace(pmesh_lvls[l], h1_coll);

        // getting boundary and essential boundary dofs
        R_space_lvls[l]->GetEssentialVDofs(all_bdrSigma, *BdrDofs_Funct_lvls[l][0]);
        R_space_lvls[l]->GetEssentialVDofs(ess_bdrSigma, *EssBdrDofs_Funct_lvls[l][0]);
        R_space_lvls[l]->GetEssentialTrueDofs(ess_bdrSigma, *EssBdrTrueDofs_Funct_lvls[l][0]);
#ifndef HCURL_COARSESOLVER
        if (l < num_levels - 1)
        {
            C_space_lvls[l]->GetEssentialVDofs(ess_bdrSigma, *EssBdrDofs_Hcurl[l]);
            C_space_lvls[l]->GetEssentialTrueDofs(ess_bdrSigma, *EssBdrTrueDofs_Hcurl[l]);
            H_space_lvls[l]->GetEssentialTrueDofs(ess_bdrS, *EssBdrTrueDofs_H1[l]);
        }
#else
        C_space_lvls[l]->GetEssentialVDofs(ess_bdrSigma, *EssBdrDofs_Hcurl[l]);
        C_space_lvls[l]->GetEssentialTrueDofs(ess_bdrSigma, *EssBdrTrueDofs_Hcurl[l]);
        C_space_lvls[l]->GetEssentialTrueDofs(ess_bdrSigma, *EssBdrTrueDofs_HcurlFunct_lvls[l][0]);
        H_space_lvls[l]->GetEssentialTrueDofs(ess_bdrS, *EssBdrTrueDofs_H1[l]);
#endif
        if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
        {
            H_space_lvls[l]->GetEssentialVDofs(all_bdrS, *BdrDofs_Funct_lvls[l][1]);
            H_space_lvls[l]->GetEssentialVDofs(ess_bdrS, *EssBdrDofs_Funct_lvls[l][1]);
            H_space_lvls[l]->GetEssentialTrueDofs(ess_bdrS, *EssBdrTrueDofs_Funct_lvls[l][1]);
            H_space_lvls[l]->GetEssentialTrueDofs(ess_bdrS, *EssBdrTrueDofs_HcurlFunct_lvls[l][1]);
            H_space_lvls[l]->GetEssentialVDofs(ess_bdrS, *EssBdrDofs_H1[l]);
        }

        ParBilinearForm mass_form(W_space_lvls[l]);
        mass_form.AddDomainIntegrator(new MassIntegrator);
        mass_form.Assemble();
        mass_form.Finalize();
        Mass_mat_lvls[l] = mass_form.LoseMat();

        // getting operators at level l
        // curl or divskew operator from C_space into R_space
        ParDiscreteLinearOperator Divfree_op(C_space_lvls[l], R_space_lvls[l]); // from Hcurl or HDivSkew(C_space) to Hdiv(R_space)
        if (dim == 3)
            Divfree_op.AddDomainInterpolator(new CurlInterpolator);
        else // dim == 4
            Divfree_op.AddDomainInterpolator(new DivSkewInterpolator);
        Divfree_op.Assemble();
        Divfree_op.Finalize();
        Divfree_mat_lvls[l] = Divfree_op.LoseMat();

        ParBilinearForm *Ablock(new ParBilinearForm(R_space_lvls[l]));
        //Ablock->AddDomainIntegrator(new VectorFEMassIntegrator);
        Ablock->AddDomainIntegrator(new VectorFEMassIntegrator);
        Ablock->Assemble();
        Ablock->EliminateEssentialBC(ess_bdrSigma);//, *sigma_exact_finest, *fform); // makes res for sigma_special happier
        Ablock->Finalize();

        // getting pointers to dof_truedof matrices
#ifndef HCURL_COARSESOLVER
        if (l < num_levels - 1)
#endif
          Dof_TrueDof_Hcurl_lvls[l] = C_space_lvls[l]->Dof_TrueDof_Matrix();
        Dof_TrueDof_Func_lvls[l][0] = R_space_lvls[l]->Dof_TrueDof_Matrix();
        Dof_TrueDof_Hdiv_lvls[l] = Dof_TrueDof_Func_lvls[l][0];
        Dof_TrueDof_L2_lvls[l] = W_space_lvls[l]->Dof_TrueDof_Matrix();
        if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
        {
            Dof_TrueDof_H1_lvls[l] = H_space_lvls[l]->Dof_TrueDof_Matrix();
            Dof_TrueDof_Func_lvls[l][1] = Dof_TrueDof_H1_lvls[l];
        }

        if (l == 0)
        {
            ParBilinearForm *Cblock;
            ParMixedBilinearForm *Bblock;
            if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
            {
                MFEM_ASSERT(strcmp(space_for_S,"H1") == 0, "Case when S is from L2 but is not"
                                                           " eliminated is not supported currently! \n");

                // diagonal block for H^1
                Cblock = new ParBilinearForm(H_space_lvls[l]);
                Cblock->AddDomainIntegrator(new DiffusionIntegrator);
                Cblock->Assemble();
                // FIXME: What about boundary conditons here?
                //Cblock->EliminateEssentialBC(ess_bdrS, xblks.GetBlock(1),*qform);
                Cblock->Finalize();

                // off-diagonal block for (H(div), Space_for_S) block
                // you need to create a new integrator here to swap the spaces
                Bblock = new ParMixedBilinearForm(H_space_lvls[l], R_space_lvls[l]);
                Bblock->AddDomainIntegrator(new MixedVectorGradientIntegrator);
                Bblock->Assemble();
                // FIXME: What about boundary conditons here?
                //BTblock->EliminateTrialDofs(ess_bdrSigma, *sigma_exact, *qform);
                //BTblock->EliminateTestDofs(ess_bdrS);
                Bblock->Finalize();
            }

            Funct_mat_offsets_lvls[l]->SetSize(numblocks_funct + 1);
            (*Funct_mat_offsets_lvls[l])[0] = 0;
            (*Funct_mat_offsets_lvls[l])[1] = Ablock->Height();
            if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
                (*Funct_mat_offsets_lvls[l])[2] = Cblock->Height();
            Funct_mat_offsets_lvls[l]->PartialSum();

            Funct_mat_lvls[l] = new BlockMatrix(*Funct_mat_offsets_lvls[l]);
            Funct_mat_lvls[l]->SetBlock(0,0,Ablock->LoseMat());
            if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
            {
                Funct_mat_lvls[l]->SetBlock(1,1,Cblock->LoseMat());
                Funct_mat_lvls[l]->SetBlock(0,1,Bblock->LoseMat());
                Funct_mat_lvls[l]->SetBlock(1,0,Transpose(Funct_mat_lvls[l]->GetBlock(0,1)));
            }

            ParMixedBilinearForm *Dblock = new ParMixedBilinearForm(R_space_lvls[l], W_space_lvls[l]);
            Dblock->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
            Dblock->Assemble();
            Dblock->Finalize();
            Constraint_mat_lvls[l] = Dblock->LoseMat();

            // Creating global functional matrix
            offsets_global[0] = 0;
            for ( int blk = 0; blk < numblocks_funct; ++blk)
                offsets_global[blk + 1] = Dof_TrueDof_Func_lvls[l][blk]->Width();
            offsets_global.PartialSum();

            Funct_global = new BlockOperator(offsets_global);

            Functrhs_global = new BlockVector(offsets_global);

            Functrhs_global->GetBlock(0) = 0.0;

            ParLinearForm *secondeqn_rhs;
            if (strcmp(space_for_S,"H1") == 0 || !eliminateS)
            {
                secondeqn_rhs = new ParLinearForm(H_space_lvls[l]);
                secondeqn_rhs->AddDomainIntegrator(new DomainLFIntegrator(zerocoeff));
                secondeqn_rhs->Assemble();

                secondeqn_rhs->ParallelAssemble(Functrhs_global->GetBlock(1));
                for (int tdofind = 0; tdofind < EssBdrDofs_Funct_lvls[0][1]->Size(); ++tdofind)
                {
                    int tdof = (*EssBdrDofs_Funct_lvls[0][1])[tdofind];
                    Functrhs_global->GetBlock(1)[tdof] = 0.0;
                }
            }

            Ablock->Assemble();
            Ablock->EliminateEssentialBC(ess_bdrSigma);//, *sigma_exact_finest, *fform); // makes res for sigma_special happier
            Ablock->Finalize();
            Funct_global->SetBlock(0,0, Ablock->ParallelAssemble());

            if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
            {
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
            }

            if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
            {
                delete Cblock;
                delete Bblock;
                delete secondeqn_rhs;
            }
            delete Dblock;
        }

        // for all but one levels we create projection matrices between levels
        // and projectors assembled on true dofs if MG preconditioner is used
        if (l < num_levels - 1)
        {
            C_space->Update();
            P_C_local = (SparseMatrix *)C_space->GetUpdateOperator();
            P_C_lvls[l] = RemoveZeroEntries(*P_C_local);

            W_space->Update();
            R_space->Update();

            if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
            {
                H_space->Update();
                P_H_local = (SparseMatrix *)H_space->GetUpdateOperator();
                SparseMatrix* H_Element_to_dofs1 = new SparseMatrix();
                P_H_lvls[l] = RemoveZeroEntries(*P_H_local);
                divp.Elem2Dofs(*H_space, *H_Element_to_dofs1);
                Element_dofs_H[l] = H_Element_to_dofs1;
            }

            P_W_local = (SparseMatrix *)W_space->GetUpdateOperator();
            P_R_local = (SparseMatrix *)R_space->GetUpdateOperator();

            SparseMatrix* R_Element_to_dofs1 = new SparseMatrix();
            SparseMatrix* W_Element_to_dofs1 = new SparseMatrix();

            divp.Elem2Dofs(*R_space, *R_Element_to_dofs1);
            divp.Elem2Dofs(*W_space, *W_Element_to_dofs1);

            P_W[l] = RemoveZeroEntries(*P_W_local);
            P_R[l] = RemoveZeroEntries(*P_R_local);

            Element_dofs_R[l] = R_Element_to_dofs1;
            Element_dofs_W[l] = W_Element_to_dofs1;

            // computing projectors assembled on true dofs

            // TODO: Rewrite these computations
            auto d_td_coarse_R = R_space_lvls[l + 1]->Dof_TrueDof_Matrix();
            SparseMatrix * RP_R_local = Mult(*R_space_lvls[l]->GetRestrictionMatrix(), *P_R[l]);
            TrueP_R[l] = d_td_coarse_R->LeftDiagMult(
                        *RP_R_local, R_space_lvls[l]->GetTrueDofOffsets());
            TrueP_R[l]->CopyColStarts();
            TrueP_R[l]->CopyRowStarts();

            delete RP_R_local;

            if (prec_is_MG)
            {
                auto d_td_coarse_C = C_space_lvls[l + 1]->Dof_TrueDof_Matrix();
                SparseMatrix * RP_C_local = Mult(*C_space_lvls[l]->GetRestrictionMatrix(), *P_C_lvls[l]);
                TrueP_C[num_levels - 2 - l] = d_td_coarse_C->LeftDiagMult(
                            *RP_C_local, C_space_lvls[l]->GetTrueDofOffsets());
                TrueP_C[num_levels - 2 - l]->CopyColStarts();
                TrueP_C[num_levels - 2 - l]->CopyRowStarts();

                delete RP_C_local;
                //delete d_td_coarse_C;
            }

            if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
            {
                auto d_td_coarse_H = H_space_lvls[l + 1]->Dof_TrueDof_Matrix();
                SparseMatrix * RP_H_local = Mult(*H_space_lvls[l]->GetRestrictionMatrix(), *P_H_lvls[l]);
                TrueP_H[num_levels - 2 - l] = d_td_coarse_H->LeftDiagMult(
                            *RP_H_local, H_space_lvls[l]->GetTrueDofOffsets());
                TrueP_H[num_levels - 2 - l]->CopyColStarts();
                TrueP_H[num_levels - 2 - l]->CopyRowStarts();

                delete RP_H_local;
                //delete d_td_coarse_H;
            }

        }

        // FIXME: TrueP_C and TrueP_H has different level ordering compared to TrueP_R

        // creating additional structures required for local problem solvers
        if (l < num_levels - 1)
        {
            (*row_offsets_El_dofs[l])[0] = 0;
            (*row_offsets_El_dofs[l])[1] = Element_dofs_R[l]->Height();
            if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
                (*row_offsets_El_dofs[l])[2] = Element_dofs_H[l]->Height();
            row_offsets_El_dofs[l]->PartialSum();

            (*col_offsets_El_dofs[l])[0] = 0;
            (*col_offsets_El_dofs[l])[1] = Element_dofs_R[l]->Width();
            if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
                (*col_offsets_El_dofs[l])[2] = Element_dofs_H[l]->Width();
            col_offsets_El_dofs[l]->PartialSum();

            Element_dofs_Func[l] = new BlockMatrix(*row_offsets_El_dofs[l], *col_offsets_El_dofs[l]);
            Element_dofs_Func[l]->SetBlock(0,0, Element_dofs_R[l]);
            if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
                Element_dofs_Func[l]->SetBlock(1,1, Element_dofs_H[l]);

            (*row_offsets_P_Func[l])[0] = 0;
            (*row_offsets_P_Func[l])[1] = P_R[l]->Height();
            if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
                (*row_offsets_P_Func[l])[2] = P_H_lvls[l]->Height();
            row_offsets_P_Func[l]->PartialSum();

            (*col_offsets_P_Func[l])[0] = 0;
            (*col_offsets_P_Func[l])[1] = P_R[l]->Width();
            if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
                (*col_offsets_P_Func[l])[2] = P_H_lvls[l]->Width();
            col_offsets_P_Func[l]->PartialSum();

            P_Func[l] = new BlockMatrix(*row_offsets_P_Func[l], *col_offsets_P_Func[l]);
            P_Func[l]->SetBlock(0,0, P_R[l]);
            if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
                P_Func[l]->SetBlock(1,1, P_H_lvls[l]);

            (*row_offsets_TrueP_Func[l])[0] = 0;
            (*row_offsets_TrueP_Func[l])[1] = TrueP_R[l]->Height();
            if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
                (*row_offsets_TrueP_Func[l])[2] = TrueP_H[num_levels - 2 - l]->Height();
            row_offsets_TrueP_Func[l]->PartialSum();

            (*col_offsets_TrueP_Func[l])[0] = 0;
            (*col_offsets_TrueP_Func[l])[1] = TrueP_R[l]->Width();
            if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
                (*col_offsets_TrueP_Func[l])[2] = TrueP_H[num_levels - 2 - l]->Width();
            col_offsets_TrueP_Func[l]->PartialSum();

            TrueP_Func[l] = new BlockOperator(*row_offsets_TrueP_Func[l], *col_offsets_TrueP_Func[l]);
            TrueP_Func[l]->SetBlock(0,0, TrueP_R[l]);
            if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
                TrueP_Func[l]->SetBlock(1,1, TrueP_H[num_levels - 2 - l]);

            P_WT[l] = Transpose(*P_W[l]);
        }

        delete Ablock;
    } // end of loop over all levels

    for ( int l = 0; l < num_levels - 1; ++l)
    {
        BlockMatrix * temp = mfem::Mult(*Funct_mat_lvls[l],*P_Func[l]);
        BlockMatrix * PT_temp = Transpose(*P_Func[l]);
        Funct_mat_lvls[l + 1] = mfem::Mult(*PT_temp, *temp);
        delete temp;
        delete PT_temp;

        SparseMatrix * temp_sp = mfem::Mult(*Constraint_mat_lvls[l], P_Func[l]->GetBlock(0,0));
        Constraint_mat_lvls[l + 1] = mfem::Mult(*P_WT[l], *temp_sp);
        delete temp_sp;
    }

    HypreParMatrix * Constraint_global;

    for (int l = 0; l < num_levels; ++l)
    {
        if (l == 0)
            Funct_global_lvls[l] = Funct_global;
        else
            Funct_global_lvls[l] = new RAPOperator(*TrueP_Func[l - 1], *Funct_global_lvls[l - 1], *TrueP_Func[l - 1]);
    }

    for (int l = 0; l < num_levels; ++l)
    {
        ParDiscreteLinearOperator Divfree_op2(C_space_lvls[l], R_space_lvls[l]); // from Hcurl or HDivSkew(C_space) to Hdiv(R_space)
        if (dim == 3)
            Divfree_op2.AddDomainInterpolator(new CurlInterpolator);
        else // dim == 4
            Divfree_op2.AddDomainInterpolator(new DivSkewInterpolator);
        Divfree_op2.Assemble();
        Divfree_op2.Finalize();
        Divfree_hpmat_mod_lvls[l] = Divfree_op2.ParallelAssemble();

        // modifying the divfree operator so that the block which connects internal dofs to boundary dofs is zero
        Eliminate_ib_block(*Divfree_hpmat_mod_lvls[l], *EssBdrTrueDofs_Hcurl[l], *EssBdrTrueDofs_Funct_lvls[l][0]);
    }

    for (int l = 0; l < num_levels; ++l)
    {
        if (l == 0)
        {
            ParBilinearForm *Ablock(new ParBilinearForm(R_space_lvls[l]));
            Ablock->AddDomainIntegrator(new VectorFEMassIntegrator);
            Ablock->Assemble();
            Ablock->EliminateEssentialBC(ess_bdrSigma);//, *sigma_exact_finest, *fform); // makes res for sigma_special happier
            Ablock->Finalize();

            (*Funct_hpmat_lvls[l])(0,0) = Ablock->ParallelAssemble();

            delete Ablock;

            ParBilinearForm *Cblock;
            ParMixedBilinearForm *Bblock;
            if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
            {
                MFEM_ASSERT(strcmp(space_for_S,"H1") == 0, "Case when S is from L2 but is not"
                                                           " eliminated is not supported currently! \n");

                // diagonal block for H^1
                Cblock = new ParBilinearForm(H_space_lvls[l]);
                Cblock->AddDomainIntegrator(new DiffusionIntegrator);
                Cblock->Assemble();
                {
                    Vector temp1(Cblock->Width());
                    temp1 = 0.0;
                    Vector temp2(Cblock->Height());
                    temp2 = 0.0;
                    Cblock->EliminateEssentialBC(ess_bdrS, temp1, temp2);
                }
                Cblock->Finalize();

                // off-diagonal block for (H(div), Space_for_S) block
                // you need to create a new integrator here to swap the spaces
                Bblock = new ParMixedBilinearForm(H_space_lvls[l], R_space_lvls[l]);
                Bblock->AddDomainIntegrator(new MixedVectorGradientIntegrator);
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

                (*Funct_hpmat_lvls[l])(1,1) = Cblock->ParallelAssemble();
                HypreParMatrix * B = Bblock->ParallelAssemble();
                (*Funct_hpmat_lvls[l])(0,1) = B;
                (*Funct_hpmat_lvls[l])(1,0) = B->Transpose();

                delete Cblock;
                delete Bblock;
            }
        }
        else // doing RAP for the Functional matrix as an Array2D<HypreParMatrix*>
        {
             // TODO: Rewrite this in a general form
            (*Funct_hpmat_lvls[l])(0,0) = RAP(TrueP_R[l-1], (*Funct_hpmat_lvls[l-1])(0,0), TrueP_R[l-1]);
            (*Funct_hpmat_lvls[l])(0,0)->CopyRowStarts();
            (*Funct_hpmat_lvls[l])(0,0)->CopyRowStarts();

            {
                const Array<int> *temp_dom = EssBdrTrueDofs_Funct_lvls[l][0];

                Eliminate_ib_block(*(*Funct_hpmat_lvls[l])(0,0), *temp_dom, *temp_dom );
                HypreParMatrix * temphpmat = (*Funct_hpmat_lvls[l])(0,0)->Transpose();
                Eliminate_ib_block(*temphpmat, *temp_dom, *temp_dom );
                (*Funct_hpmat_lvls[l])(0,0) = temphpmat->Transpose();
                Eliminate_bb_block(*(*Funct_hpmat_lvls[l])(0,0), *temp_dom);
                SparseMatrix diag;
                (*Funct_hpmat_lvls[l])(0,0)->GetDiag(diag);
                diag.MoveDiagonalFirst();

                (*Funct_hpmat_lvls[l])(0,0)->CopyRowStarts();
                (*Funct_hpmat_lvls[l])(0,0)->CopyColStarts();
                delete temphpmat;
            }


            if (strcmp(space_for_S,"H1") == 0)
            {
                (*Funct_hpmat_lvls[l])(1,1) = RAP(TrueP_H[num_levels - 2 - (l-1)], (*Funct_hpmat_lvls[l-1])(1,1), TrueP_H[num_levels - 2 - (l-1)]);
                //(*Funct_hpmat_lvls[l])(1,1)->CopyRowStarts();
                //(*Funct_hpmat_lvls[l])(1,1)->CopyRowStarts();

                {
                    const Array<int> *temp_dom = EssBdrTrueDofs_Funct_lvls[l][1];

                    Eliminate_ib_block(*(*Funct_hpmat_lvls[l])(1,1), *temp_dom, *temp_dom );
                    HypreParMatrix * temphpmat = (*Funct_hpmat_lvls[l])(1,1)->Transpose();
                    Eliminate_ib_block(*temphpmat, *temp_dom, *temp_dom );
                    (*Funct_hpmat_lvls[l])(1,1) = temphpmat->Transpose();
                    Eliminate_bb_block(*(*Funct_hpmat_lvls[l])(1,1), *temp_dom);
                    SparseMatrix diag;
                    (*Funct_hpmat_lvls[l])(1,1)->GetDiag(diag);
                    diag.MoveDiagonalFirst();

                    (*Funct_hpmat_lvls[l])(1,1)->CopyRowStarts();
                    (*Funct_hpmat_lvls[l])(1,1)->CopyColStarts();
                    delete temphpmat;
                }

                HypreParMatrix * P_R_T = TrueP_R[l-1]->Transpose();
                HypreParMatrix * temp1 = ParMult((*Funct_hpmat_lvls[l-1])(0,1), TrueP_H[num_levels - 2 - (l-1)]);
                (*Funct_hpmat_lvls[l])(0,1) = ParMult(P_R_T, temp1);
                //(*Funct_hpmat_lvls[l])(0,1)->CopyRowStarts();
                //(*Funct_hpmat_lvls[l])(0,1)->CopyRowStarts();

                {
                    const Array<int> *temp_range = EssBdrTrueDofs_Funct_lvls[l][0];
                    const Array<int> *temp_dom = EssBdrTrueDofs_Funct_lvls[l][1];

                    Eliminate_ib_block(*(*Funct_hpmat_lvls[l])(0,1), *temp_dom, *temp_range );
                    HypreParMatrix * temphpmat = (*Funct_hpmat_lvls[l])(0,1)->Transpose();
                    Eliminate_ib_block(*temphpmat, *temp_range, *temp_dom );
                    (*Funct_hpmat_lvls[l])(0,1) = temphpmat->Transpose();
                    (*Funct_hpmat_lvls[l])(0,1)->CopyRowStarts();
                    (*Funct_hpmat_lvls[l])(0,1)->CopyColStarts();
                    delete temphpmat;
                }



                (*Funct_hpmat_lvls[l])(1,0) = (*Funct_hpmat_lvls[l])(0,1)->Transpose();
                (*Funct_hpmat_lvls[l])(1,0)->CopyRowStarts();
                (*Funct_hpmat_lvls[l])(1,0)->CopyRowStarts();

                delete P_R_T;
                delete temp1;
            }

        } // end of else for if (l == 0)

    } // end of loop over levels which create Funct matrices at each level

    for (int l = 0; l < num_levels; ++l)
    {
        if (l == 0)
        {
            ParMixedBilinearForm *Bblock = new ParMixedBilinearForm(R_space_lvls[l], W_space_lvls[l]);
            Bblock->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
            Bblock->Assemble();
            Vector tempsol(Bblock->Width());
            tempsol = 0.0;
            Vector temprhs(Bblock->Height());
            temprhs = 0.0;
            //Bblock->EliminateTrialDofs(ess_bdrSigma, tempsol, temprhs);
            //Bblock->EliminateTestDofs(ess_bdrSigma);
            Bblock->Finalize();
            Constraint_global = Bblock->ParallelAssemble();

            delete Bblock;
        }
    }

    //MPI_Finalize();
    //return 0;

    for (int l = num_levels - 1; l >=0; --l)
    {
        if (l < num_levels - 1)
        {
#ifdef WITH_SMOOTHERS
            Array<int> SweepsNum(numblocks_funct);
            Array<int> offsets_global(numblocks_funct + 1);
            offsets_global[0] = 0;
            for ( int blk = 0; blk < numblocks_funct; ++blk)
                offsets_global[blk + 1] = Dof_TrueDof_Func_lvls[l][blk]->Width();
            offsets_global.PartialSum();
            SweepsNum = ipow(1, l);
            if (verbose)
            {
                std::cout << "Sweeps num: \n";
                SweepsNum.Print();
            }
            /*
            if (l == 0)
            {
                if (verbose)
                {
                    std::cout << "Sweeps num: \n";
                    SweepsNum.Print();
                }
            }
            */
            Smoothers_lvls[l] = new HcurlGSSSmoother(*Funct_hpmat_lvls[l], *Divfree_hpmat_mod_lvls[l],
                                                     *EssBdrTrueDofs_Hcurl[l],
                                                     EssBdrTrueDofs_Funct_lvls[l],
                                                     &SweepsNum, offsets_global);
#else // for #ifdef WITH_SMOOTHERS
            Smoothers_lvls[l] = NULL;
#endif

#ifdef CHECK_SPDSMOOTHER
            {
                if (num_procs == 1)
                {
                    Vector Vec1(Smoothers_lvls[l]->Height());
                    Vec1.Randomize(2000);
                    Vector Vec2(Smoothers_lvls[l]->Height());
                    Vec2.Randomize(-39);

                    Vector Tempy(Smoothers_lvls[l]->Height());

                    /*
                    for ( int i = 0; i < Vec1.Size(); ++i )
                    {
                        if ((*EssBdrDofs_R[0][0])[i] != 0 )
                        {
                            Vec1[i] = 0.0;
                            Vec2[i] = 0.0;
                        }
                    }
                    */

                    Vector VecDiff(Vec1.Size());
                    VecDiff = Vec1;

                    std::cout << "Norm of Vec1 = " << VecDiff.Norml2() / sqrt(VecDiff.Size())  << "\n";

                    VecDiff -= Vec2;

                    MFEM_ASSERT(VecDiff.Norml2() / sqrt(VecDiff.Size()) > 1.0e-10, "Vec1 equals Vec2 but they must be different");
                    //VecDiff.Print();
                    std::cout << "Norm of (Vec1 - Vec2) = " << VecDiff.Norml2() / sqrt(VecDiff.Size())  << "\n";

                    Smoothers_lvls[l]->Mult(Vec1, Tempy);
                    double scal1 = Tempy * Vec2;
                    double scal3 = Tempy * Vec1;
                    //std::cout << "A Vec1 norm = " << Tempy.Norml2() / sqrt (Tempy.Size()) << "\n";

                    Smoothers_lvls[l]->Mult(Vec2, Tempy);
                    double scal2 = Tempy * Vec1;
                    double scal4 = Tempy * Vec2;
                    //std::cout << "A Vec2 norm = " << Tempy.Norml2() / sqrt (Tempy.Size()) << "\n";

                    std::cout << "scal1 = " << scal1 << "\n";
                    std::cout << "scal2 = " << scal2 << "\n";

                    if ( fabs(scal1 - scal2) / fabs(scal1) > 1.0e-12)
                    {
                        std::cout << "Smoother is not symmetric on two random vectors: \n";
                        std::cout << "vec2 * (A * vec1) = " << scal1 << " != " << scal2 << " = vec1 * (A * vec2)" << "\n";
                        std::cout << "difference = " << scal1 - scal2 << "\n";
                    }
                    else
                    {
                        std::cout << "Smoother was symmetric on the given vectors: dot product = " << scal1 << "\n";
                    }

                    std::cout << "scal3 = " << scal3 << "\n";
                    std::cout << "scal4 = " << scal4 << "\n";

                    if (scal3 < 0 || scal4 < 0)
                    {
                        std::cout << "The operator (new smoother) is not s.p.d. \n";
                    }
                    else
                    {
                        std::cout << "The smoother is s.p.d. on the two random vectors: (Av,v) > 0 \n";
                    }
                }
                else
                    if (verbose)
                        std::cout << "Symmetry check for the smoother works correctly only in the serial case \n";

            }
#endif
        }

        // creating local problem solver hierarchy
        if (l < num_levels - 1)
        {
            int size = 0;
            for (int blk = 0; blk < numblocks_funct; ++blk)
                size += Dof_TrueDof_Func_lvls[l][blk]->GetNumCols();

            bool optimized_localsolve = true;
            if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
            {
                (*LocalSolver_partfinder_lvls)[l] = new LocalProblemSolverWithS(size, *Funct_mat_lvls[l],
                                                         *Constraint_mat_lvls[l],
                                                         Dof_TrueDof_Func_lvls[l],
                                                         *P_WT[l],
                                                         *Element_dofs_Func[l],
                                                         *Element_dofs_W[l],
                                                         BdrDofs_Funct_lvls[l],
                                                         EssBdrDofs_Funct_lvls[l],
                                                         optimized_localsolve);
            }
            else // no S
            {
                (*LocalSolver_partfinder_lvls)[l] = new LocalProblemSolver(size, *Funct_mat_lvls[l],
                                                         *Constraint_mat_lvls[l],
                                                         Dof_TrueDof_Func_lvls[l],
                                                         *P_WT[l],
                                                         *Element_dofs_Func[l],
                                                         *Element_dofs_W[l],
                                                         BdrDofs_Funct_lvls[l],
                                                         EssBdrDofs_Funct_lvls[l],
                                                         optimized_localsolve);
            }

            (*LocalSolver_lvls)[l] = (*LocalSolver_partfinder_lvls)[l];

        }
    }

    //MPI_Finalize();
    //return 0;

    // Creating the coarsest problem solver
    int size = 0;
    for (int blk = 0; blk < numblocks_funct; ++blk)
        size += Dof_TrueDof_Func_lvls[num_levels - 1][blk]->GetNumCols();
    size += Dof_TrueDof_L2_lvls[num_levels - 1]->GetNumCols();

    CoarsestSolver_partfinder = new CoarsestProblemSolver(size, *Funct_mat_lvls[num_levels - 1],
                                                     *Constraint_mat_lvls[num_levels - 1],
                                                     Dof_TrueDof_Func_lvls[num_levels - 1],
                                                     *Dof_TrueDof_L2_lvls[num_levels - 1],
                                                     EssBdrDofs_Funct_lvls[num_levels - 1],
                                                     EssBdrTrueDofs_Funct_lvls[num_levels - 1]);
#ifdef HCURL_COARSESOLVER
    if (verbose)
        std::cout << "Creating the new coarsest solver which works in the div-free subspace \n" << std::flush;

    int size_sp = 0;
    for (int blk = 0; blk < numblocks_funct; ++blk)
        size_sp += Dof_TrueDof_Func_lvls[num_levels - 1][blk]->GetNumCols();
    CoarsestSolver = new CoarsestProblemHcurlSolver(size_sp,
                                                     *Funct_hpmat_lvls[num_levels - 1],
                                                     *Divfree_hpmat_mod_lvls[num_levels - 1],
                                                     EssBdrDofs_Funct_lvls[num_levels - 1],
                                                     EssBdrTrueDofs_Funct_lvls[num_levels - 1],
                                                     *EssBdrDofs_Hcurl[num_levels - 1],
                                                     *EssBdrTrueDofs_Hcurl[num_levels - 1]);

    ((CoarsestProblemHcurlSolver*)CoarsestSolver)->SetMaxIter(100);
    ((CoarsestProblemHcurlSolver*)CoarsestSolver)->SetAbsTol(1.0e-7);
    ((CoarsestProblemHcurlSolver*)CoarsestSolver)->SetRelTol(1.0e-7);
    ((CoarsestProblemHcurlSolver*)CoarsestSolver)->ResetSolverParams();
#else
    CoarsestSolver = CoarsestSolver_partfinder;
    CoarsestSolver_partfinder->SetMaxIter(1000);
    CoarsestSolver_partfinder->SetAbsTol(1.0e-12);
    CoarsestSolver_partfinder->SetRelTol(1.0e-12);
    CoarsestSolver_partfinder->ResetSolverParams();
#endif

    if (verbose)
    {
#ifdef HCURL_COARSESOLVER
        std::cout << "CoarseSolver size = " << Divfree_hpmat_mod_lvls[num_levels - 1]->M()
                    + (*Funct_hpmat_lvls[num_levels - 1])(1,1)->M() << "\n";
#else
        std::cout << "CoarseSolver size = " << Dof_TrueDof_Func_lvls[num_levels - 1][0]->N()
                + Dof_TrueDof_Func_lvls[num_levels - 1][1]->N() + Dof_TrueDof_L2_lvls[num_levels - 1]->N() << "\n";
#endif
    }

    if (verbose)
        std::cout << "End of the creating a hierarchy of meshes AND pfespaces \n";

    ParGridFunction * sigma_exact_finest;
    sigma_exact_finest = new ParGridFunction(R_space_lvls[0]);
    sigma_exact_finest->ProjectCoefficient(*Mytest.GetSigma());
    Vector sigma_exact_truedofs(R_space_lvls[0]->GetTrueVSize());
    sigma_exact_finest->ParallelProject(sigma_exact_truedofs);

    ParGridFunction * S_exact_finest;
    Vector S_exact_truedofs;
    if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
    {
        S_exact_finest = new ParGridFunction(H_space_lvls[0]);
        S_exact_finest->ProjectCoefficient(*Mytest.GetU());
        S_exact_truedofs.SetSize(H_space_lvls[0]->GetTrueVSize());
        S_exact_finest->ParallelProject(S_exact_truedofs);
    }

    chrono.Stop();
    if (verbose)
        std::cout << "Hierarchy of f.e. spaces and stuff was constructed in "<< chrono.RealTime() <<" seconds.\n";

    pmesh->PrintInfo(std::cout); if(verbose) cout << "\n";

    //////////////////////////////////////////////////

#if !defined (WITH_DIVCONSTRAINT_SOLVER) || defined (OLD_CODE)
    chrono.Clear();
    chrono.Start();
    ParGridFunction * Sigmahat = new ParGridFunction(R_space);
    ParLinearForm *gform;
    HypreParMatrix *Bdiv;

    Vector F_fine(P_W[0]->Height());
    Vector G_fine(P_R[0]->Height());
    Vector sigmahat_pau;

    if (with_multilevel)
    {
        if (verbose)
            std::cout << "Using multilevel algorithm for finding a particular solution \n";

        ConstantCoefficient k(1.0);

        SparseMatrix *M_local;
        ParBilinearForm *mVarf;
        if (useM_in_divpart)
        {
            mVarf = new ParBilinearForm(R_space);
            mVarf->AddDomainIntegrator(new VectorFEMassIntegrator(k));
            mVarf->Assemble();
            mVarf->Finalize();
            SparseMatrix &M_fine(mVarf->SpMat());
            M_local = &M_fine;
        }
        else
        {
            M_local = NULL;
        }

        ParMixedBilinearForm *bVarf(new ParMixedBilinearForm(R_space, W_space));
        bVarf->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
        bVarf->Assemble();
        bVarf->Finalize();
        Bdiv = bVarf->ParallelAssemble();
        SparseMatrix &B_fine = bVarf->SpMat();
        SparseMatrix *B_local = &B_fine;

        //Right hand size

        gform = new ParLinearForm(W_space);
        gform->AddDomainIntegrator(new DomainLFIntegrator(*Mytest.GetRhs()));
        gform->Assemble();

        F_fine = *gform;
        G_fine = .0;

        divp.div_part(ref_levels,
                      M_local, B_local,
                      G_fine,
                      F_fine,
                      P_W, P_R, P_W,
                      Element_dofs_R,
                      Element_dofs_W,
                      Dof_TrueDof_Func_lvls[num_levels - 1][0],
                      Dof_TrueDof_L2_lvls[num_levels - 1],
                      R_space_lvls[num_levels - 1]->GetDofOffsets(),
                      W_space_lvls[num_levels - 1]->GetDofOffsets(),
                      sigmahat_pau,
                      *EssBdrDofs_Funct_lvls[num_levels - 1][0]);

#ifdef MFEM_DEBUG
        Vector sth(F_fine.Size());
        B_fine.Mult(sigmahat_pau, sth);
        sth -= F_fine;
        std::cout << "sth.Norml2() = " << sth.Norml2() << "\n";
        MFEM_ASSERT(sth.Norml2()<1e-8, "The particular solution does not satisfy the divergence constraint");
#endif

        //delete M_local;
        //delete B_local;
        delete bVarf;
        delete mVarf;

        *Sigmahat = sigmahat_pau;
    }
    else
    {
        if (verbose)
            std::cout << "Solving Poisson problem for finding a particular solution \n";
        ParGridFunction *sigma_exact;
        ParMixedBilinearForm *Bblock;
        HypreParMatrix *BdivT;
        HypreParMatrix *BBT;
        HypreParVector *Rhs;

        sigma_exact = new ParGridFunction(R_space);
        sigma_exact->ProjectCoefficient(*Mytest.GetSigma());

        gform = new ParLinearForm(W_space);
        gform->AddDomainIntegrator(new DomainLFIntegrator(*Mytest.GetRhs()));
        gform->Assemble();

        Bblock = new ParMixedBilinearForm(R_space, W_space);
        Bblock->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
        Bblock->Assemble();
        Bblock->EliminateTrialDofs(ess_bdrSigma, *sigma_exact, *gform);

        Bblock->Finalize();
        Bdiv = Bblock->ParallelAssemble();
        BdivT = Bdiv->Transpose();
        BBT = ParMult(Bdiv, BdivT);
        Rhs = gform->ParallelAssemble();

        HypreBoomerAMG * invBBT = new HypreBoomerAMG(*BBT);
        invBBT->SetPrintLevel(0);

        mfem::CGSolver solver(comm);
        solver.SetPrintLevel(0);
        solver.SetMaxIter(70000);
        solver.SetRelTol(1.0e-12);
        solver.SetAbsTol(1.0e-14);
        solver.SetPreconditioner(*invBBT);
        solver.SetOperator(*BBT);

        Vector * Temphat = new Vector(W_space->TrueVSize());
        *Temphat = 0.0;
        solver.Mult(*Rhs, *Temphat);

        Vector * Temp = new Vector(R_space->TrueVSize());
        BdivT->Mult(*Temphat, *Temp);

        Sigmahat->Distribute(*Temp);
        //Sigmahat->SetFromTrueDofs(*Temp);

        delete sigma_exact;
        delete invBBT;
        delete BBT;
        delete Bblock;
        delete Rhs;
        delete Temphat;
        delete Temp;
    }

    // in either way now Sigmahat is a function from H(div) s.t. div Sigmahat = div sigma = f

    chrono.Stop();
    if (verbose)
        cout << "Particular solution found in "<< chrono.RealTime() <<" seconds.\n";

    if (verbose)
        std::cout << "Checking that particular solution in parallel version satisfies the divergence constraint \n";

    {
        ParLinearForm * constrfform = new ParLinearForm(W_space);
        constrfform->AddDomainIntegrator(new DomainLFIntegrator(*Mytest.GetRhs()));
        constrfform->Assemble();

        Vector Floc(P_W[0]->Height());
        Floc = *constrfform;

        Vector Sigmahat_truedofs(R_space->TrueVSize());
        Sigmahat->ParallelProject(Sigmahat_truedofs);

        if (!CheckConstrRes(Sigmahat_truedofs, *Constraint_global, &Floc, "in the old code for the particular solution"))
        {
            std::cout << "Failure! \n";
        }
        else
            if (verbose)
                std::cout << "Success \n";

    }
#endif


#ifdef OLD_CODE
    chrono.Clear();
    chrono.Start();

    // 6. Define a parallel finite element space on the parallel mesh. Here we
    //    use the Raviart-Thomas finite elements of the specified order.

    int numblocks = 1;
    if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
        numblocks++;

    Array<int> block_offsets(numblocks + 1); // number of variables + 1
    block_offsets[0] = 0;
    block_offsets[1] = C_space->GetVSize();
    if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
        block_offsets[2] = S_space->GetVSize();
    block_offsets.PartialSum();

    Array<int> block_trueOffsets(numblocks + 1); // number of variables + 1
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = C_space->TrueVSize();
    if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
        block_trueOffsets[2] = S_space->TrueVSize();
    block_trueOffsets.PartialSum();

    HYPRE_Int dimC = C_space->GlobalTrueVSize();
    HYPRE_Int dimR = R_space->GlobalTrueVSize();
    HYPRE_Int dimS;
    if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
        dimS = S_space->GlobalTrueVSize();
    if (verbose)
    {
        std::cout << "***********************************************************\n";
        std::cout << "dim(C) = " << dimC << "\n";
        if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
        {
            std::cout << "dim(S) = " << dimS << ", ";
            std::cout << "dim(C+S) = " << dimC + dimS << "\n";
            std::cout << "dim(R+S) = " << dimR + dimS << "\n";
        }
        std::cout << "dim(R) = " << dimR << "\n";
        std::cout << "***********************************************************\n";
    }

    BlockVector xblks(block_offsets), rhsblks(block_offsets);
    BlockVector trueX(block_trueOffsets), trueRhs(block_trueOffsets);
    xblks = 0.0;
    rhsblks = 0.0;
    trueX = 0.0;
    trueRhs = 0.0;

    //VectorFunctionCoefficient f(dim, f_exact);
    //VectorFunctionCoefficient vone(dim, vone_exact);
    //VectorFunctionCoefficient vminusone(dim, vminusone_exact);
    //VectorFunctionCoefficient E(dim, E_exact);
    //VectorFunctionCoefficient curlE(dim, curlE_exact);

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
        std::cout << "Small things in OLD_CODE were done in "<< chrono.RealTime() <<" seconds.\n";

    chrono.Clear();
    chrono.Start();

    // the div-free part
    ParGridFunction *S_exact = new ParGridFunction(S_space);
    S_exact->ProjectCoefficient(*Mytest.GetU());

    ParGridFunction * sigma_exact = new ParGridFunction(R_space);
    sigma_exact->ProjectCoefficient(*Mytest.GetSigma());

    {
        Vector Sigmahat_truedofs(R_space->TrueVSize());
        Sigmahat->ParallelProject(Sigmahat_truedofs);

        Vector sigma_exact_truedofs((R_space->TrueVSize()));
        sigma_exact->ParallelProject(sigma_exact_truedofs);

        MFEM_ASSERT(CheckBdrError(Sigmahat_truedofs, &sigma_exact_truedofs, *EssBdrTrueDofs_Funct_lvls[0][0], true),
                                  "for the particular solution Sigmahat in the old code");
    }

    // FIXME: remove this
    {
        const Array<int> *temp = EssBdrDofs_Funct_lvls[0][0];

        for ( int tdof = 0; tdof < temp->Size(); ++tdof)
        {
            if ( (*temp)[tdof] != 0 && fabs( (*Sigmahat)[tdof]) > 1.0e-14 )
                std::cout << "bnd cnd is violated for Sigmahat! value = "
                          << (*Sigmahat)[tdof]
                          << "exact val = " << (*sigma_exact)[tdof] << ", index = " << tdof << "\n";
        }
    }

    xblks.GetBlock(0) = 0.0;

    if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S from H1 or (S from L2 and no elimination)
        xblks.GetBlock(1) = *S_exact;

#ifdef USE_CURLMATRIX
    if (verbose)
        std::cout << "Creating div-free system using the explicit discrete div-free operator \n";

    ParGridFunction* rhside_Hdiv = new ParGridFunction(R_space);  // rhside for the first equation in the original cfosls system
    *rhside_Hdiv = 0.0;

    ParLinearForm *qform;
    if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
    {
        qform = new ParLinearForm(S_space);
        qform->AddDomainIntegrator(new DomainLFIntegrator(zerocoeff));
        qform->Assemble();
    }

    BlockOperator *MainOp = new BlockOperator(block_trueOffsets);

    // curl or divskew operator from C_space into R_space
    /*
    ParDiscreteLinearOperator Divfree_op(C_space, R_space); // from Hcurl or HDivSkew(C_space) to Hdiv(R_space)
    if (dim == 3)
        Divfree_op.AddDomainInterpolator(new CurlInterpolator());
    else // dim == 4
        Divfree_op.AddDomainInterpolator(new DivSkewInterpolator());
    Divfree_op.Assemble();
    Divfree_op.Finalize();
    HypreParMatrix * Divfree_dop = Divfree_op.ParallelAssemble(); // from Hcurl or HDivSkew(C_space) to Hdiv(R_space)
    HypreParMatrix * DivfreeT_dop = Divfree_dop->Transpose();
    */

    HypreParMatrix * Divfree_dop = Divfree_hpmat_mod_lvls[0];
    HypreParMatrix * DivfreeT_dop = Divfree_dop->Transpose();

    // mass matrix for H(div)
    ParBilinearForm *Mblock(new ParBilinearForm(R_space));
    Mblock->AddDomainIntegrator(new VectorFEMassIntegrator);
    Mblock->Assemble();
    Mblock->EliminateEssentialBC(ess_bdrSigma, *sigma_exact, *rhside_Hdiv);
    Mblock->Finalize();

    HypreParMatrix *M = Mblock->ParallelAssemble();

    // div-free operator matrix (curl in 3D, divskew in 4D)
    // either as DivfreeT_dop * M * Divfree_dop
    auto A = RAP(Divfree_dop, M, Divfree_dop);
    A->CopyRowStarts();
    A->CopyColStarts();

    Eliminate_ib_block(*A, *EssBdrTrueDofs_Hcurl[0], *EssBdrTrueDofs_Hcurl[0] );
    HypreParMatrix * temphpmat = A->Transpose();
    Eliminate_ib_block(*temphpmat, *EssBdrTrueDofs_Hcurl[0], *EssBdrTrueDofs_Hcurl[0] );
    A = temphpmat->Transpose();
    A->CopyColStarts();
    A->CopyRowStarts();
    SparseMatrix diag;
    A->GetDiag(diag);
    diag.MoveDiagonalFirst();
    delete temphpmat;
    Eliminate_bb_block(*A, *EssBdrTrueDofs_Hcurl[0]);

    HypreParMatrix *C, *CH, *CHT, *B, *BT;
    if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
    {
        // diagonal block for H^1
        ParBilinearForm *Cblock = new ParBilinearForm(S_space);
        Cblock->AddDomainIntegrator(new DiffusionIntegrator);
        Cblock->Assemble();
        Cblock->EliminateEssentialBC(ess_bdrS, xblks.GetBlock(1),*qform);
        Cblock->Finalize();
        C = Cblock->ParallelAssemble();

        // off-diagonal block for (H(div), Space_for_S) block
        // you need to create a new integrator here to swap the spaces
        ParMixedBilinearForm *Bblock(new ParMixedBilinearForm(S_space, R_space));
        Bblock->AddDomainIntegrator(new MixedVectorGradientIntegrator);
        Bblock->Assemble();
        Bblock->EliminateTrialDofs(ess_bdrS, *S_exact, *rhside_Hdiv);
        Bblock->EliminateTestDofs(ess_bdrSigma);
        Bblock->Finalize();
        B = Bblock->ParallelAssemble();
        BT = B->Transpose();

        CHT = ParMult(DivfreeT_dop, B);
        CHT->CopyColStarts();
        CHT->CopyRowStarts();
        CH = CHT->Transpose();

        delete Cblock;
        delete Bblock;
    }

    // additional temporary vectors on true dofs required for various matvec
    Vector tempHdiv_true(R_space->TrueVSize());
    Vector temp2Hdiv_true(R_space->TrueVSize());

    // assembling local rhs vectors from inhomog. boundary conditions
    if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
        qform->ParallelAssemble(trueRhs.GetBlock(1));
    rhside_Hdiv->ParallelAssemble(tempHdiv_true);
    DivfreeT_dop->Mult(tempHdiv_true, trueRhs.GetBlock(0));

    //trueRhs.GetBlock(0).Print();

    // subtracting from rhs for sigma a part from Sigmahat
    Sigmahat->ParallelProject(tempHdiv_true);
    M->Mult(tempHdiv_true, temp2Hdiv_true);
    //DivfreeT_dop->Mult(temp2Hdiv_true, tempHcurl_true);
    //trueRhs.GetBlock(0) -= tempHcurl_true;
    DivfreeT_dop->Mult(-1.0, temp2Hdiv_true, 1.0, trueRhs.GetBlock(0));

    // subtracting from rhs for S a part from Sigmahat
    //BT->Mult(tempHdiv_true, tempH1_true);
    //trueRhs.GetBlock(1) -= tempH1_true;
    if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
        BT->Mult(-1.0, tempHdiv_true, 1.0, trueRhs.GetBlock(1));

    // setting block operator of the system
    MainOp->SetBlock(0,0, A);
    if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
    {
        MainOp->SetBlock(0,1, CHT);
        MainOp->SetBlock(1,0, CH);
        MainOp->SetBlock(1,1, C);
    }
#else // if using the integrators for creating the div-free system
    if (verbose)
        std::cout << "This case is not supported any more \n";
    MPI_Finalize();
    return -1;
#endif

    /*
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

    // checking the residual for the projections of exact solution for Hdiv equation after extracting sigmahat
    Vector Msigmahat(M->Height());

    //Vector sigma_hat_truedofs(M->Width());
    //Sigmahat->ParallelProject(sigma_hat_truedofs);
    //M->Mult(sigma_hat_truedofs, Msigmahat);

    //M->Mult(tempHdiv_true, Msigmahat);

    Msigmahat = temp2Hdiv_true;

    Vector res_Hdiv2(M->Height());
    res_Hdiv2 = res_Hdiv;
    res_Hdiv2 -= Msigmahat;
    res_Hdiv2 += temp2Hdiv_true;

    double res_Hdiv2_norm = res_Hdiv2.Norml2() / sqrt (res_Hdiv2.Size());

    if (verbose)
        std::cout << "res_Hdiv_withsigmahat_norm = " << res_Hdiv2_norm << "\n";

    // checking the residual for the projections of exact solution for Hcurl equation after extracting sigmahat
    Vector res_Hcurl(DivfreeT_dop->Height());
    DivfreeT_dop->Mult(res_Hdiv2, res_Hcurl);

    double res_Hcurl_norm = res_Hcurl.Norml2() / sqrt (res_Hcurl.Size());

    if (verbose)
        std::cout << "res_Hcurl_norm = " << res_Hcurl_norm << "\n";

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

    ParMixedBilinearForm *Bblock = new ParMixedBilinearForm(R_space, W_space);
    Bblock->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
    Bblock->Assemble();
    Bblock->Finalize();
    HypreParMatrix * Constr_forcheck = Bblock->ParallelAssemble();

    Vector res_Constr(Constr_forcheck->Height());
    Constr_forcheck->Mult(sigma_exact_truedofs, res_Constr);
    res_Constr -= F_fine;

    double res_Constr_norm = res_Constr.Norml2() / sqrt (res_Constr.Size());

    if (verbose)
        std::cout << "res_Constr_norm = " << res_Constr_norm << "\n";

    //MPI_Barrier(comm);
    //MPI_Finalize();
    //return 0;
    */

    //delete Divfree_dop;
    //delete DivfreeT_dop;
    delete rhside_Hdiv;

    chrono.Stop();
    if (verbose)
        std::cout << "Discretized problem is assembled" << endl << flush;

    chrono.Clear();
    chrono.Start();

    Solver *prec;
    Array<BlockOperator*> P;
    std::vector<Array<int> *> offsets_f;
    std::vector<Array<int> *> offsets_c;

    if (with_prec)
    {
        if(dim<=4)
        {
            if (prec_is_MG)
            {
                if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
                {
                    if (monolithicMG)
                    {
                        P.SetSize(TrueP_C.Size());

                        offsets_f.resize(num_levels);
                        offsets_c.resize(num_levels);

                        for (int l = 0; l < P.Size(); l++)
                        {
                            offsets_f[l] = new Array<int>(3);
                            offsets_c[l] = new Array<int>(3);

                            (*offsets_f[l])[0] = (*offsets_c[l])[0] = 0;
                            (*offsets_f[l])[1] = TrueP_C[l]->Height();
                            (*offsets_c[l])[1] = TrueP_C[l]->Width();
                            (*offsets_f[l])[2] = (*offsets_f[l])[1] + TrueP_H[l]->Height();
                            (*offsets_c[l])[2] = (*offsets_c[l])[1] + TrueP_H[l]->Width();

                            P[l] = new BlockOperator(*offsets_f[l], *offsets_c[l]);
                            P[l]->SetBlock(0, 0, TrueP_C[l]);
                            P[l]->SetBlock(1, 1, TrueP_H[l]);
                        }

#ifdef BND_FOR_MULTIGRID
                        prec = new MonolithicMultigrid(*MainOp, P, EssBdrTrueDofs_HcurlFunct_lvls);
#else
                        prec = new MonolithicMultigrid(*MainOp, P);
#endif
                    }
                    else
                    {
                        prec = new BlockDiagonalPreconditioner(block_trueOffsets);
#ifdef BND_FOR_MULTIGRID
                        Operator * precU = new Multigrid(*A, TrueP_C, EssBdrTrueDofs_Hcurl);
                        Operator * precS = new Multigrid(*C, TrueP_H, EssBdrTrueDofs_H1);
#else
                        Operator * precU = new Multigrid(*A, TrueP_C);
                        Operator * precS = new Multigrid(*C, TrueP_H);
#endif
                        ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(0, precU);
                        ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(1, precS);
                    }
                }
                else // only equation in div-free subspace
                {
                    if (monolithicMG && verbose)
                        std::cout << "There is only one variable in the system because there is no S, \n"
                                     "So monolithicMG is the same as block-diagonal MG \n";
                    if (prec_is_MG)
                    {
                        prec = new BlockDiagonalPreconditioner(block_trueOffsets);
                        Operator * precU;
#ifdef MARTIN_PREC
                        if (dim == 4)
                        {
                            if (verbose)
                                std::cout << "Creating an instance of DivSkew4dPrec \n";

                            Coefficient *alpha = new ConstantCoefficient(1.0);
                            Coefficient *beta = new ConstantCoefficient(1.0);
                            int order = feorder + 1;
                            bool exactH1Solver = false;
                            precU = new DivSkew4dPrec(A, C_space_lvls[0], alpha, beta, ess_bdrSigma, order, exactH1Solver);
                        }
                        else
                            precU = new Multigrid(*A, TrueP_C);
#else
#ifdef BND_FOR_MULTIGRID

#ifdef COARSEPREC_AMS
                        precU = new Multigrid(*A, TrueP_C, EssBdrTrueDofs_Hcurl, C_space_lvls[num_levels - 1]);
#else
                        precU = new Multigrid(*A, TrueP_C, EssBdrTrueDofs_Hcurl);
#endif
#else
                        precU = new Multigrid(*A, TrueP_C);
#endif
                        //precU = new IdentityOperator(A->Height());
#endif

                        ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(0, precU);
                    }

                    //mfem_error("MG is not implemented when there is no S in the system");
                }
            }
            else // prec is AMS-like for the div-free part (block-diagonal for the system with boomerAMG for S)
            {
                if (dim == 3)
                {
                    if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
                    {
                        prec = new BlockDiagonalPreconditioner(block_trueOffsets);
                        /*
                        Operator * precU = new HypreAMS(*A, C_space);
                        ((HypreAMS*)precU)->SetSingularProblem();
                        */

                        // Why in this case, when S is even in H1 as in the paper,
                        // CG is saying that the operator is not pos.def.
                        // And I checked that this is precU block that causes the trouble
                        // For, example, the following works:
                        Operator * precU = new IdentityOperator(A->Height());

                        Operator * precS;
                        /*
                        if (strcmp(space_for_S,"H1") == 0) // S is from H1
                        {
                            precS = new HypreBoomerAMG(*C);
                            ((HypreBoomerAMG*)precS)->SetPrintLevel(0);

                            //FIXME: do we need to set iterative mode = false here and around this place?
                        }
                        else // S is from L2
                        {
                            precS = new HypreDiagScale(*C);
                            //precS->SetPrintLevel(0);
                        }
                        */

                        precS = new IdentityOperator(C->Height());

                        //auto precSmatrix = ((HypreDiagScale*)precS)->GetData();
                        //SparseMatrix precSdiag;
                        //precSmatrix->GetDiag(precSdiag);
                        //precSdiag.Print();

                        ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(0, precU);
                        ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(1, precS);
                    }
                    else // no S, i.e. only an equation in div-free subspace
                    {
                        prec = new BlockDiagonalPreconditioner(block_trueOffsets);
                        /*
                        Operator * precU = new HypreAMS(*A, C_space);
                        ((HypreAMS*)precU)->SetSingularProblem();
                        */

                        // See the remark below, for the case when S is present
                        // CG is saying that the operator is not pos.def.
                        // And I checked that this is precU block that causes the trouble
                        // For, example, the following works:
                        Operator * precU = new IdentityOperator(A->Height());

                        ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(0, precU);
                    }

                }
                else // dim == 4
                {
                    if (verbose)
                        std::cout << "Aux. space prec is not implemented in 4D \n";
                    MPI_Finalize();
                    return 0;
                }
            }
        }

        if (verbose)
            cout << "Preconditioner is ready" << endl << flush;
    }
    else
        if (verbose)
            cout << "Using no preconditioner \n";

    chrono.Stop();
    if (verbose)
        std::cout << "Preconditioner was created in "<< chrono.RealTime() <<" seconds.\n";


    CGSolver solver(comm);
    if (verbose)
        cout << "Linear solver: CG \n";
    //solver = new MINRESSolver(comm);
    //if (verbose)
        //cout << "Linear solver: MINRES \n";
#ifdef HCURL_MG_TEST
    Solver *testprec;
    testprec = new Multigrid(*A, TrueP_C, NULL);

    solver.SetAbsTol(sqrt(atol));
    solver.SetRelTol(sqrt(rtol));
    solver.SetMaxIter(max_num_iter);
    solver.SetOperator(*A);

    Vector testVec(DivfreeT_dop->Width());
    testVec = 1.0;
    Vector testRhs(A->Height());
    DivfreeT_dop->Mult(testVec, testRhs);
    Vector testX(A->Width());

    solver.SetPrintLevel(0);
    solver.SetPreconditioner(*testprec);

    chrono.Clear();
    chrono.Start();
    solver.Mult(testRhs, testX);
    chrono.Stop();

    delete Divfree_dop;
    delete DivfreeT_dop;
    delete rhside_Hdiv;

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
    MPI_Finalize();
    return 0;
#endif

    solver.SetAbsTol(sqrt(atol));
    solver.SetRelTol(sqrt(rtol));
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

    ParGridFunction * u = new ParGridFunction(C_space);
    ParGridFunction * S;

    u->Distribute(&(trueX.GetBlock(0)));

    // 13. Extract the parallel grid function corresponding to the finite element
    //     approximation X. This is the local solution on each processor. Compute
    //     L2 error norms.

    int order_quad = max(2, 2*feorder+1);
    const IntegrationRule *irs[Geometry::NumGeom];
    for (int i=0; i < Geometry::NumGeom; ++i)
    {
        irs[i] = &(IntRules.Get(i, order_quad));
    }

    ParGridFunction * opdivfreepart = new ParGridFunction(R_space);
    Vector u_truedofs(Divfree_hpmat_mod_lvls[0]->Width());
    u->ParallelProject(u_truedofs);

    Vector opdivfree_truedofs(Divfree_hpmat_mod_lvls[0]->Height());
    Divfree_hpmat_mod_lvls[0]->Mult(u_truedofs, opdivfree_truedofs);
    opdivfreepart->Distribute(opdivfree_truedofs);

    // FIXME: remove this
    {
        const Array<int> *temp = EssBdrDofs_Funct_lvls[0][0];

        for ( int tdof = 0; tdof < temp->Size(); ++tdof)
        {
            if ( (*temp)[tdof] != 0 && fabs( (*opdivfreepart)[tdof]) > 1.0e-14 )
            {
                std::cout << "bnd cnd is violated for opdivfreepart! value = "
                          << (*opdivfreepart)[tdof]
                          << ", index = " << tdof << "\n";
            }
        }
    }

    ParGridFunction * sigma = new ParGridFunction(R_space);
    *sigma = *Sigmahat;         // particular solution
    *sigma += *opdivfreepart;   // plus div-free guy

    // FIXME: remove this
    {
        const Array<int> *temp = EssBdrDofs_Funct_lvls[0][0];

        for ( int tdof = 0; tdof < temp->Size(); ++tdof)
        {
            if ( (*temp)[tdof] != 0 && fabs( (*sigma)[tdof]) > 1.0e-14 )
                std::cout << "bnd cnd is violated for sigma! value = "
                          << (*sigma)[tdof]
                          << "exact val = " << (*sigma_exact)[tdof] << ", index = " << tdof << "\n";
        }
    }

    if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
    {
        S = new ParGridFunction(S_space);
        S->Distribute(&(trueX.GetBlock(1)));
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

    /*
    double err_sigmahat = Sigmahat->ComputeL2Error(*Mytest.GetSigma(), irs);
    if (verbose && !withDiv)
        if ( norm_sigma > MYZEROTOL )
            cout << "|| sigma_hat - sigma_ex || / || sigma_ex || = " << err_sigmahat / norm_sigma << endl;
        else
            cout << "|| sigma_hat || = " << err_sigmahat << " (sigma_ex = 0)" << endl;
    */

    DiscreteLinearOperator Div(R_space, W_space);
    Div.AddDomainInterpolator(new DivergenceInterpolator());
    ParGridFunction DivSigma(W_space);
    Div.Assemble();
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

#ifdef USE_CURLMATRIX
        // Check value of functional and mass conservation
        if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
        {
            Vector trueSigma(R_space->TrueVSize());
            trueSigma = 0.0;
            sigma->ParallelProject(trueSigma);

            Vector MtrueSigma(R_space->TrueVSize());
            MtrueSigma = 0.0;
            M->Mult(trueSigma, MtrueSigma);
            double localFunctional = trueSigma*MtrueSigma;

            Vector GtrueSigma(S_space->TrueVSize());
            GtrueSigma = 0.0;

            BT->Mult(trueSigma, GtrueSigma);
            localFunctional += 2.0*(trueX.GetBlock(1)*GtrueSigma);

            Vector XtrueS(S_space->TrueVSize());
            XtrueS = 0.0;
            C->Mult(trueX.GetBlock(1), XtrueS);
            localFunctional += trueX.GetBlock(1)*XtrueS;

            double globalFunctional;
            MPI_Reduce(&localFunctional, &globalFunctional, 1,
                       MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (verbose)
            {
                cout << "|| sigma_h - L(S_h) ||^2 + || div_h sigma_h - f ||^2 = "
                     << globalFunctional+err_div*err_div<< "\n";
                cout << "|| f ||^2 = " << norm_div*norm_div  << "\n";
                cout << "Relative Energy Error = "
                     << sqrt(globalFunctional+err_div*err_div)/norm_div<< "\n";
            }

            auto trueRhs_part = gform->ParallelAssemble();
            double mass_loc = trueRhs_part->Norml1();
            double mass;
            MPI_Reduce(&mass_loc, &mass, 1,
                       MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (verbose)
                cout << "Sum of local mass = " << mass<< "\n";

            Vector DtrueSigma(W_space->TrueVSize());
            DtrueSigma = 0.0;
            Bdiv->Mult(trueSigma, DtrueSigma);
            DtrueSigma -= *trueRhs_part;
            double mass_loss_loc = DtrueSigma.Norml1();
            double mass_loss;
            MPI_Reduce(&mass_loss_loc, &mass_loss, 1,
                       MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (verbose)
                cout << "Sum of local mass loss = " << mass_loss<< "\n";

            delete trueRhs_part;
        }
#endif
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
        std::cout << "Errors in the MG code were computed in "<< chrono.RealTime() <<" seconds.\n";

    //MPI_Finalize();
    //return 0;

#endif // for #ifdef OLD_CODE

#ifdef VISUALIZATION
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
#endif

    chrono.Clear();
    chrono.Start();

    if (verbose)
        std::cout << "\nCreating an instance of the minimization solver \n";

    //ParLinearForm *fform = new ParLinearForm(R_space);

    ParLinearForm * constrfform = new ParLinearForm(W_space_lvls[0]);
    constrfform->AddDomainIntegrator(new DomainLFIntegrator(*Mytest.GetRhs()));
    constrfform->Assemble();

    /*
    ParMixedBilinearForm *Bblock(new ParMixedBilinearForm(R_space, W_space));
    Bblock->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
    Bblock->Assemble();
    //Bblock->EliminateTrialDofs(ess_bdrSigma, *sigma_exact_finest, *constrfform); // // makes res for sigma_special happier
    Bblock->Finalize();
    */

    Vector Floc(P_W[0]->Height());
    Floc = *constrfform;

    delete constrfform;

    BlockVector Xinit(Funct_mat_lvls[0]->ColOffsets());
    Xinit.GetBlock(0) = 0.0;
    MFEM_ASSERT(Xinit.GetBlock(0).Size() == sigma_exact_finest->Size(),
                "Xinit and sigma_exact_finest have different sizes! \n");

    for (int i = 0; i < sigma_exact_finest->Size(); ++i )
    {
        // just setting Xinit to store correct boundary values at essential boundary
        if ( (*EssBdrDofs_Funct_lvls[0][0])[i] != 0)
            Xinit.GetBlock(0)[i] = (*sigma_exact_finest)[i];
    }

    Array<int> new_trueoffsets(numblocks_funct + 1);
    new_trueoffsets[0] = 0;
    for ( int blk = 0; blk < numblocks_funct; ++blk)
        new_trueoffsets[blk + 1] = Dof_TrueDof_Func_lvls[0][blk]->Width();
    new_trueoffsets.PartialSum();
    BlockVector Xinit_truedofs(new_trueoffsets);
    Xinit_truedofs = 0.0;

    for (int i = 0; i < EssBdrTrueDofs_Funct_lvls[0][0]->Size(); ++i )
    {
        int tdof = (*EssBdrTrueDofs_Funct_lvls[0][0])[i];
        Xinit_truedofs.GetBlock(0)[tdof] = sigma_exact_truedofs[tdof];
    }

    if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
    {
        for (int i = 0; i < S_exact_finest->Size(); ++i )
        {
            // just setting Xinit to store correct boundary values at essential boundary
            if ( (*EssBdrDofs_Funct_lvls[0][1])[i] != 0)
                Xinit.GetBlock(1)[i] = (*S_exact_finest)[i];
        }

        for (int i = 0; i < EssBdrTrueDofs_Funct_lvls[0][1]->Size(); ++i )
        {
            int tdof = (*EssBdrTrueDofs_Funct_lvls[0][1])[i];
            Xinit_truedofs.GetBlock(1)[tdof] = S_exact_truedofs[tdof];
        }
    }

    //MPI_Finalize();
    //return 0;

    // testing some matrix properties:
    // realizing that we miss canonical projections to have
    // coarsened curl orthogonal to coarsened divergence
    /*
    // 1. looking at (P_RT)^T * P_RT - it is not diagonal!
    SparseMatrix * temppp = Transpose(P_Func[0]->GetBlock(0,0));
    SparseMatrix * testtt = Mult(*temppp, P_Func[0]->GetBlock(0,0));

    //testtt->Print();

    // 2. looking at B_0 * C_0
    SparseMatrix * testtt2 = Mult(Bloc,Divfree_op_sp);
    //testtt2->Print();

    // 3. looking at B_0 * (P_RT)^T * P_RT * C_0
    SparseMatrix * temppp2 = Mult(Bloc,P_Func[0]->GetBlock(0,0));
    SparseMatrix * temppp3 = Mult(*temppp,Divfree_op_sp);

    SparseMatrix * testtt3 = Mult(*temppp2,*temppp3);
    //testtt3->Print();
    */


    /*
    Vector ones_v(pmesh->Dimension());
    ones_v = 1.0;
    VectorConstantCoefficient ones_vcoeff(ones_v);

    Vector Truevec1(C_space_lvls[0]->GetTrueVSize());
    ParGridFunction * hcurl_guy = new ParGridFunction(C_space_lvls[0]);
    hcurl_guy->ProjectCoefficient(ones_vcoeff);
    hcurl_guy->ParallelProject(Truevec1);

    if (myid == 0)
    {
        ofstream ofs("hcurl_guy_0.txt");
        ofs << Truevec1.Size() << "\n";
        Truevec1.Print(ofs,1);
    }
    if (myid == 1)
    {
        ofstream ofs("hcurl_guy_1.txt");
        ofs << Truevec1.Size() << "\n";
        Truevec1.Print(ofs,1);
    }
    if (myid == 2)
    {
        ofstream ofs("hcurl_guy_2.txt");
        ofs << Truevec1.Size() << "\n";
        Truevec1.Print(ofs,1);
    }
    if (myid == 3)
    {
        ofstream ofs("hcurl_guy_3.txt");
        ofs << Truevec1.Size() << "\n";
        Truevec1.Print(ofs,1);
    }

    Vector Truevec2(R_space_lvls[0]->GetTrueVSize());
    ParGridFunction * hdiv_guy = new ParGridFunction(R_space_lvls[0]);
    hdiv_guy->ProjectCoefficient(ones_vcoeff);
    hdiv_guy->ParallelProject(Truevec2);

    if (myid == 0)
    {
        ofstream ofs("hdiv_guy_0.txt");
        std::cout << "Truevec2 size = " << Truevec2.Size() << "\n";
        ofs << Truevec2.Size() << "\n";
        Truevec2.Print(ofs,1);
    }
    if (myid == 1)
    {
        ofstream ofs("hdiv_guy_1.txt");
        std::cout << "Truevec2 size = " << Truevec2.Size() << "\n";
        ofs << Truevec2.Size() << "\n";
        Truevec2.Print(ofs,1);
    }
    if (myid == 2)
    {
        ofstream ofs("hdiv_guy_2.txt");
        std::cout << "Truevec2 size = " << Truevec2.Size() << "\n";
        ofs << Truevec2.Size() << "\n";
        Truevec2.Print(ofs,1);
    }
    if (myid == 3)
    {
        ofstream ofs("hdiv_guy_3.txt");
        std::cout << "Truevec2 size = " << Truevec2.Size() << "\n";
        ofs << Truevec2.Size() << "\n";
        Truevec2.Print(ofs,1);
    }

    Vector error1(Truevec2.Size());
    error1 = Truevec2;

    int local_size1 = error1.Size();
    int global_size1 = 0;
    MPI_Allreduce(&local_size1, &global_size1, 1, MPI_INT, MPI_SUM, comm);

    double local_normsq1 = error1 * error1;
    double global_norm1 = 0.0;
    MPI_Allreduce(&local_normsq1, &global_norm1, 1, MPI_DOUBLE, MPI_SUM, comm);
    global_norm1 = sqrt (global_norm1 / global_size1);

    if (verbose)
        std::cout << "error1 norm special = " << global_norm1 << "\n";
    */

    chrono.Stop();
    if (verbose)
        std::cout << "Intermediate allocations for the new solver were done in "<< chrono.RealTime() <<" seconds.\n";
    chrono.Clear();
    chrono.Start();

    if (verbose)
        std::cout << "Calling constructor of the new solver \n";

    int stopcriteria_type = 1;

#ifdef TIMING
    std::list<double>* Times_mult = new std::list<double>;
    std::list<double>* Times_solve = new std::list<double>;
    std::list<double>* Times_localsolve = new std::list<double>;
    std::list<double>* Times_localsolve_lvls = new std::list<double>[num_levels - 1];
    std::list<double>* Times_smoother = new std::list<double>;
    std::list<double>* Times_smoother_lvls = new std::list<double>[num_levels - 1];
    std::list<double>* Times_coarsestproblem = new std::list<double>;
    std::list<double>* Times_resupdate = new std::list<double>;
    std::list<double>* Times_fw = new std::list<double>;
    std::list<double>* Times_up = new std::list<double>;
#endif

#ifdef WITH_DIVCONSTRAINT_SOLVER
    DivConstraintSolver PartsolFinder(comm, num_levels, P_WT,
                                      TrueP_Func, P_W,
                                      Mass_mat_lvls,
                                      EssBdrTrueDofs_Funct_lvls,
                                      Funct_global_lvls,
                                      *Constraint_global,
                                      Floc,
                                      Smoothers_lvls,
                                      LocalSolver_partfinder_lvls,
                                      CoarsestSolver_partfinder, verbose);
    CoarsestSolver_partfinder->SetMaxIter(70000);
    CoarsestSolver_partfinder->SetAbsTol(1.0e-18);
    CoarsestSolver_partfinder->SetRelTol(1.0e-18);
    CoarsestSolver_partfinder->ResetSolverParams();
#endif

    GeneralMinConstrSolver NewSolver(comm, num_levels,
                     TrueP_Func, EssBdrTrueDofs_Funct_lvls,
                     *Functrhs_global, Smoothers_lvls,
                     Funct_global_lvls,
#ifdef CHECK_CONSTR
                     *Constraint_global, Floc,
#endif
#ifdef TIMING
                     Times_mult, Times_solve, Times_localsolve, Times_localsolve_lvls, Times_smoother, Times_smoother_lvls, Times_coarsestproblem, Times_resupdate, Times_fw, Times_up,
#endif
#ifdef SOLVE_WITH_LOCALSOLVERS
                     LocalSolver_lvls,
#else
                     NULL,
#endif
                     CoarsestSolver, stopcriteria_type);

    double newsolver_reltol = 1.0e-6;

    if (verbose)
        std::cout << "newsolver_reltol = " << newsolver_reltol << "\n";

    NewSolver.SetRelTol(newsolver_reltol);
    NewSolver.SetMaxIter(40);
    NewSolver.SetPrintLevel(0);
    NewSolver.SetStopCriteriaType(0);
    //NewSolver.SetLocalSolvers(LocalSolver_lvls);

    BlockVector ParticSol(new_trueoffsets);
    ParticSol = 0.0;

    chrono.Stop();
    if (verbose)
        std::cout << "New solver and PartSolFinder were created in "<< chrono.RealTime() <<" seconds.\n";
    chrono.Clear();
    chrono.Start();

#ifdef WITH_DIVCONSTRAINT_SOLVER
    if (verbose)
    {
        std::cout << "CoarsestSolver parameters for the PartSolFinder: \n" << std::flush;
        CoarsestSolver_partfinder->PrintSolverParams();
    }

    PartsolFinder.FindParticularSolution(Xinit_truedofs, ParticSol, Floc, verbose);
#else
    Sigmahat->ParallelProject(ParticSol.GetBlock(0));
#endif

    chrono.Stop();

#ifdef TIMING
#ifdef WITH_SMOOTHERS
    for (int l = 0; l < num_levels - 1; ++l)
        ((HcurlGSSSmoother*)Smoothers_lvls[l])->ResetInternalTimings();
#endif
#endif

#ifndef HCURL_COARSESOLVER
    CoarsestSolver_partfinder->SetMaxIter(200);
    CoarsestSolver_partfinder->SetAbsTol(1.0e-9); // -9
    CoarsestSolver_partfinder->SetRelTol(1.0e-9); // -9 for USE_AS_A_PREC
    CoarsestSolver_partfinder->ResetSolverParams();
#else
    ((CoarsestProblemHcurlSolver*)CoarsestSolver)->SetMaxIter(100);
    ((CoarsestProblemHcurlSolver*)CoarsestSolver)->SetAbsTol(sqrt(1.0e-32));
    ((CoarsestProblemHcurlSolver*)CoarsestSolver)->SetRelTol(sqrt(1.0e-12));
    ((CoarsestProblemHcurlSolver*)CoarsestSolver)->ResetSolverParams();
#endif
    if (verbose)
    {
        std::cout << "CoarsestSolver parameters for the new solver: \n" << std::flush;
#ifndef HCURL_COARSESOLVER
        CoarsestSolver_partfinder->PrintSolverParams();
#else
        ((CoarsestProblemHcurlSolver*)CoarsestSolver)->PrintSolverParams();
#endif
    }
    if (verbose)
        std::cout << "Particular solution was found in " << chrono.RealTime() <<" seconds.\n";
    chrono.Clear();
    chrono.Start();

    // checking that the particular solution satisfies the divergence constraint
    BlockVector temp_dofs(Funct_mat_lvls[0]->RowOffsets());
    for ( int blk = 0; blk < numblocks_funct; ++blk)
    {
        Dof_TrueDof_Func_lvls[0][blk]->Mult(ParticSol.GetBlock(blk), temp_dofs.GetBlock(blk));
    }

    Vector temp_constr(Constraint_mat_lvls[0]->Height());
    Constraint_mat_lvls[0]->Mult(temp_dofs.GetBlock(0), temp_constr);
    temp_constr -= Floc;

    // 3.1 if not, abort
    if ( ComputeMPIVecNorm(comm, temp_constr,"", verbose) > 1.0e-13 )
    {
        std::cout << "Initial vector does not satisfies divergence constraint. \n";
        double temp = ComputeMPIVecNorm(comm, temp_constr,"", verbose);
        //temp_constr.Print();
        if (verbose)
            std::cout << "Constraint residual norm: " << temp << "\n";
        MFEM_ABORT("");
    }

    for (int blk = 0; blk < numblocks_funct; ++blk)
    {
        MFEM_ASSERT(CheckBdrError(ParticSol.GetBlock(blk), &(Xinit_truedofs.GetBlock(blk)),
                                  *EssBdrTrueDofs_Funct_lvls[0][blk], true),
                                  "for the particular solution");
    }

    Vector error3(ParticSol.Size());
    error3 = ParticSol;

    int local_size3 = error3.Size();
    int global_size3 = 0;
    MPI_Allreduce(&local_size3, &global_size3, 1, MPI_INT, MPI_SUM, comm);

    double local_normsq3 = error3 * error3;
    double global_norm3 = 0.0;
    MPI_Allreduce(&local_normsq3, &global_norm3, 1, MPI_DOUBLE, MPI_SUM, comm);
    global_norm3 = sqrt (global_norm3 / global_size3);

    if (verbose)
        std::cout << "error3 norm special = " << global_norm3 << "\n";

    if (verbose)
        std::cout << "Checking that particular solution in parallel version satisfies the divergence constraint \n";

    //MFEM_ASSERT(CheckConstrRes(*PartSolDofs, *Constraint_mat_lvls[0], &Floc, "in the main code for the particular solution"), "Failure");
    //if (!CheckConstrRes(*PartSolDofs, *Constraint_mat_lvls[0], &Floc, "in the main code for the particular solution"))
    if (!CheckConstrRes(ParticSol.GetBlock(0), *Constraint_global, &Floc, "in the main code for the particular solution"))
    {
        std::cout << "Failure! \n";
    }
    else
        if (verbose)
            std::cout << "Success \n";
    //MPI_Finalize();
    //return 0;

    /*
    Vector tempp(sigma_exact_finest->Size());
    tempp = *sigma_exact_finest;
    tempp -= Xinit;

    std::cout << "norm of sigma_exact = " << sigma_exact_finest->Norml2() / sqrt (sigma_exact_finest->Size()) << "\n";
    std::cout << "norm of sigma_exact - Xinit = " << tempp.Norml2() / sqrt (tempp.Size()) << "\n";

    Vector res(Funct_mat_lvls[0]->GetBlock(0,0).Height());
    Funct_mat_lvls[0]->GetBlock(0,0).Mult(*sigma_exact_finest, res);
    double func_norm = res.Norml2() / sqrt (res.Size());
    std::cout << "Functional norm for sigma_exact projection:  = " << func_norm << " ... \n";

#ifdef OLD_CODE
    res = 0.0;
    Funct_mat_lvls[0]->GetBlock(0,0).Mult(*sigma, res);
    func_norm = res.Norml2() / sqrt (res.Size());
    std::cout << "Functional norm for exact sigma_h:  = " << func_norm << " ... \n";
#endif
    */

    chrono.Stop();
    if (verbose)
        std::cout << "Intermediate things were done in " << chrono.RealTime() <<" seconds.\n";
    chrono.Clear();
    chrono.Start();

    ParGridFunction * NewSigmahat = new ParGridFunction(R_space_lvls[0]);

    ParGridFunction * NewS;
    if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
        NewS = new ParGridFunction(H_space_lvls[0]);

    //Vector Tempx(sigma_exact_finest->Size());
    //Tempx = 0.0;
    //Vector Tempy(Tempx.Size());
    Vector Tempy(ParticSol.Size());
    Tempy = 0.0;

#ifdef CHECK_SPDSOLVER

    // checking that for unsymmetric version the symmetry check does
    // provide the negative answer
    //NewSolver.SetUnSymmetric();

    Vector Vec1(NewSolver.Height());
    Vec1.Randomize(2000);
    Vector Vec2(NewSolver.Height());
    Vec2.Randomize(-39);

    for ( int i = 0; i < EssBdrTrueDofs_Funct_lvls[0][0]->Size(); ++i )
    {
        int tdof = (*EssBdrTrueDofs_Funct_lvls[0][0])[i];
        Vec1[tdof] = 0.0;
        Vec2[tdof] = 0.0;
    }

    Vector VecDiff(Vec1.Size());
    VecDiff = Vec1;

    std::cout << "Norm of Vec1 = " << VecDiff.Norml2() / sqrt(VecDiff.Size())  << "\n";

    VecDiff -= Vec2;

    MFEM_ASSERT(VecDiff.Norml2() / sqrt(VecDiff.Size()) > 1.0e-10, "Vec1 equals Vec2 but they must be different");
    //VecDiff.Print();
    std::cout << "Norm of (Vec1 - Vec2) = " << VecDiff.Norml2() / sqrt(VecDiff.Size())  << "\n";

    NewSolver.SetAsPreconditioner(true);
    NewSolver.SetMaxIter(1);

    NewSolver.Mult(Vec1, Tempy);
    double scal1 = Tempy * Vec2;
    double scal3 = Tempy * Vec1;
    //std::cout << "A Vec1 norm = " << Tempy.Norml2() / sqrt (Tempy.Size()) << "\n";

    NewSolver.Mult(Vec2, Tempy);
    double scal2 = Tempy * Vec1;
    double scal4 = Tempy * Vec2;
    //std::cout << "A Vec2 norm = " << Tempy.Norml2() / sqrt (Tempy.Size()) << "\n";

    std::cout << "scal1 = " << scal1 << "\n";
    std::cout << "scal2 = " << scal2 << "\n";

    if ( fabs(scal1 - scal2) / fabs(scal1) > 1.0e-12)
    {
        std::cout << "Solver is not symmetric on two random vectors: \n";
        std::cout << "vec2 * (A * vec1) = " << scal1 << " != " << scal2 << " = vec1 * (A * vec2)" << "\n";
        std::cout << "difference = " << scal1 - scal2 << "\n";
        std::cout << "relative difference = " << fabs(scal1 - scal2) / fabs(scal1) << "\n";
    }
    else
    {
        std::cout << "Solver was symmetric on the given vectors: dot product = " << scal1 << "\n";
    }

    std::cout << "scal3 = " << scal3 << "\n";
    std::cout << "scal4 = " << scal4 << "\n";

    if (scal3 < 0 || scal4 < 0)
    {
        std::cout << "The operator (new solver) is not s.p.d. \n";
    }
    else
    {
        std::cout << "The solver is s.p.d. on the two random vectors: (Av,v) > 0 \n";
    }

    MPI_Finalize();
    return 0;

#endif


#ifdef USE_AS_A_PREC
    if (verbose)
        std::cout << "Using the new solver as a preconditioner for CG for the correction \n";

    chrono.Clear();
    chrono.Start();

    ParLinearForm *fformtest = new ParLinearForm(R_space_lvls[0]);
    ConstantCoefficient zerotest(.0);
    fformtest->AddDomainIntegrator(new VectordivDomainLFIntegrator(zerotest));
    fformtest->Assemble();

    ParLinearForm *qformtest;
    if (strcmp(space_for_S,"H1") == 0 || !eliminateS)
    {
        qformtest = new ParLinearForm(H_space_lvls[0]);
        qformtest->AddDomainIntegrator(new DomainLFIntegrator(zerocoeff));
        qformtest->Assemble();
    }

    ParBilinearForm *Ablocktest(new ParBilinearForm(R_space_lvls[0]));
    HypreParMatrix *Atest;
    Ablocktest->AddDomainIntegrator(new VectorFEMassIntegrator);
    Ablocktest->Assemble();
    Ablocktest->EliminateEssentialBC(ess_bdrSigma, *sigma_exact_finest, *fformtest);
    Ablocktest->Finalize();
    Atest = Ablocktest->ParallelAssemble();

    delete Ablocktest;

    HypreParMatrix *Ctest;
    if (strcmp(space_for_S,"H1") == 0)
    {
        ParBilinearForm * Cblocktest = new ParBilinearForm(H_space_lvls[0]);
        Cblocktest->AddDomainIntegrator(new DiffusionIntegrator);
        Cblocktest->Assemble();
        Cblocktest->EliminateEssentialBC(ess_bdrS, *S_exact_finest, *qformtest);
        Cblocktest->Finalize();

        Ctest = Cblocktest->ParallelAssemble();

        delete Cblocktest;
    }

    HypreParMatrix *Btest;
    HypreParMatrix *BTtest;
    if (strcmp(space_for_S,"H1") == 0)
    {
        ParMixedBilinearForm *Bblocktest = new ParMixedBilinearForm(H_space_lvls[0], R_space_lvls[0]);
        Bblocktest->AddDomainIntegrator(new MixedVectorGradientIntegrator);
        Bblocktest->Assemble();
        Bblocktest->EliminateTrialDofs(ess_bdrS, *S_exact_finest, *fformtest);
        Bblocktest->EliminateTestDofs(ess_bdrSigma);
        Bblocktest->Finalize();

        Btest = Bblocktest->ParallelAssemble();
        BTtest = Btest->Transpose();

        delete Bblocktest;
    }

    Array<int> blocktest_offsets(numblocks_funct + 1);
    blocktest_offsets[0] = 0;
    blocktest_offsets[1] = Atest->Height();
    if (strcmp(space_for_S,"H1") == 0)
        blocktest_offsets[2] = Ctest->Height();
    blocktest_offsets.PartialSum();

    BlockVector trueXtest(blocktest_offsets);
    BlockVector trueRhstest(blocktest_offsets);
    trueRhstest = 0.0;

    fformtest->ParallelAssemble(trueRhstest.GetBlock(0));
    if (strcmp(space_for_S,"H1") == 0)
        qformtest->ParallelAssemble(trueRhstest.GetBlock(1));

    delete fformtest;
    if (strcmp(space_for_S,"H1") == 0)
        delete qformtest;

    BlockOperator *BlockMattest = new BlockOperator(blocktest_offsets);
    BlockMattest->SetBlock(0,0, Atest);
    if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
    {
        BlockMattest->SetBlock(1,0, BTtest);
        BlockMattest->SetBlock(0,1, Btest);
        BlockMattest->SetBlock(1,1, Ctest);
    }

    NewSolver.SetAsPreconditioner(true);
    NewSolver.SetPrintLevel(0);
    if (verbose)
        NewSolver.PrintAllOptions();

    int TestmaxIter(400);

    CGSolver Testsolver(MPI_COMM_WORLD);
    Testsolver.SetAbsTol(sqrt(atol));
    Testsolver.SetRelTol(sqrt(rtol));
    Testsolver.SetMaxIter(TestmaxIter);
    Testsolver.SetOperator(*BlockMattest);
    Testsolver.SetPrintLevel(0);
    Testsolver.SetPreconditioner(NewSolver);

    trueXtest = 0.0;

    // trueRhstest = F - Funct * particular solution (= residual), on true dofs
    BlockVector truetemp(blocktest_offsets);
    BlockMattest->Mult(ParticSol, truetemp);
    trueRhstest -= truetemp;

    chrono.Stop();
    if (verbose)
        std::cout << "Global system for the CG was built in " << chrono.RealTime() <<" seconds.\n";
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
        std::cout << "System size: " << Atest->M() + Ctest->M() << "\n" << std::flush;
    }

    chrono.Clear();

#ifdef TIMING
    double temp_sum;

    temp_sum = 0.0;
    for (list<double>::iterator i = Times_mult->begin(); i != Times_mult->end(); ++i)
        temp_sum += *i;
    if (verbose)
        std::cout << "time_mult = " << temp_sum << "\n";
    delete Times_mult;
    temp_sum = 0.0;
    for (list<double>::iterator i = Times_solve->begin(); i != Times_solve->end(); ++i)
        temp_sum += *i;
    if (verbose)
        std::cout << "time_solve = " << temp_sum << "\n";
    delete Times_solve;
    temp_sum = 0.0;
    for (list<double>::iterator i = Times_localsolve->begin(); i != Times_localsolve->end(); ++i)
        temp_sum += *i;
    if (verbose)
        std::cout << "time_localsolve = " << temp_sum << "\n";
    delete Times_localsolve;

    for (int l = 0; l < num_levels - 1; ++l)
    {
        temp_sum = 0.0;
        for (list<double>::iterator i = Times_localsolve_lvls[l].begin(); i != Times_localsolve_lvls[l].end(); ++i)
            temp_sum += *i;
        if (verbose)
            std::cout << "time_localsolve lvl " << l << " = " << temp_sum << "\n";
    }
    //delete Times_localsolve_lvls;

    temp_sum = 0.0;
    for (list<double>::iterator i = Times_smoother->begin(); i != Times_smoother->end(); ++i)
        temp_sum += *i;
    if (verbose)
        std::cout << "time_smoother = " << temp_sum << "\n";
    delete Times_smoother;

    for (int l = 0; l < num_levels - 1; ++l)
    {
        temp_sum = 0.0;
        for (list<double>::iterator i = Times_smoother_lvls[l].begin(); i != Times_smoother_lvls[l].end(); ++i)
            temp_sum += *i;
        if (verbose)
            std::cout << "time_smoother lvl " << l << " = " << temp_sum << "\n";
    }
    //delete Times_smoother_lvls;
#ifdef WITH_SMOOTHERS
    for (int l = 0; l < num_levels - 1; ++l)
    {
        if (verbose)
        {
           std::cout << "Internal timing of the smoother at level " << l << ": \n";
           std::cout << "global mult time: " << ((HcurlGSSSmoother*)Smoothers_lvls[l])->GetGlobalMultTime() << " \n" << std::flush;
           std::cout << "internal mult time: " << ((HcurlGSSSmoother*)Smoothers_lvls[l])->GetInternalMultTime() << " \n" << std::flush;
           std::cout << "before internal mult time: " << ((HcurlGSSSmoother*)Smoothers_lvls[l])->GetBeforeIntMultTime() << " \n" << std::flush;
           std::cout << "after internal mult time: " << ((HcurlGSSSmoother*)Smoothers_lvls[l])->GetAfterIntMultTime() << " \n" << std::flush;
        }
    }
#endif
    temp_sum = 0.0;
    for (list<double>::iterator i = Times_coarsestproblem->begin(); i != Times_coarsestproblem->end(); ++i)
        temp_sum += *i;
    if (verbose)
        std::cout << "time_coarsestproblem = " << temp_sum << "\n";
    delete Times_coarsestproblem;

    MPI_Barrier(comm);
    temp_sum = 0.0;
    for (list<double>::iterator i = Times_resupdate->begin(); i != Times_resupdate->end(); ++i)
        temp_sum += *i;
    if (verbose)
        std::cout << "time_resupdate = " << temp_sum << "\n";
    delete Times_resupdate;

    temp_sum = 0.0;
    for (list<double>::iterator i = Times_fw->begin(); i != Times_fw->end(); ++i)
        temp_sum += *i;
    if (verbose)
        std::cout << "time_fw = " << temp_sum << "\n";
    delete Times_fw;
    temp_sum = 0.0;
    for (list<double>::iterator i = Times_up->begin(); i != Times_up->end(); ++i)
        temp_sum += *i;
    if (verbose)
        std::cout << "time_up = " << temp_sum << "\n";
    delete Times_up;
#endif

    chrono.Start();

    trueXtest += ParticSol;
    NewSigmahat->Distribute(trueXtest.GetBlock(0));
    if (strcmp(space_for_S,"H1") == 0)
        NewS->Distribute(trueXtest.GetBlock(1));

    {
        int order_quad = max(2, 2*feorder+1);
        const IntegrationRule *irs[Geometry::NumGeom];
        for (int i = 0; i < Geometry::NumGeom; ++i)
        {
            irs[i] = &(IntRules.Get(i, order_quad));
        }

        double norm_sigma = ComputeGlobalLpNorm(2, *Mytest.GetSigma(), *pmesh, irs);
        double err_newsigmahat = NewSigmahat->ComputeL2Error(*Mytest.GetSigma(), irs);
        if (verbose)
        {
            if ( norm_sigma > MYZEROTOL )
                cout << "|| new sigma_h - sigma_ex || / || sigma_ex || = " << err_newsigmahat / norm_sigma << endl;
            else
                cout << "|| new sigma_h || = " << err_newsigmahat << " (sigma_ex = 0)" << endl;
        }

        DiscreteLinearOperator Div(R_space, W_space);
        Div.AddDomainInterpolator(new DivergenceInterpolator());
        ParGridFunction DivSigma(W_space);
        Div.Assemble();
        Div.Mult(*NewSigmahat, DivSigma);

        double err_div = DivSigma.ComputeL2Error(*Mytest.GetRhs(),irs);
        double norm_div = ComputeGlobalLpNorm(2, *Mytest.GetRhs(), *pmesh, irs);

        if (verbose)
        {
            cout << "|| div (new sigma_h - sigma_ex) || / ||div (sigma_ex)|| = "
                      << err_div/norm_div  << "\n";
        }

        //////////////////////////////////////////////////////
        if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
        {
            double max_bdr_error = 0;
            for ( int dof = 0; dof < Xinit.GetBlock(1).Size(); ++dof)
            {
                if ( (*EssBdrDofs_Funct_lvls[0][1])[dof] != 0.0)
                {
                    //std::cout << "ess dof index: " << dof << "\n";
                    double bdr_error_dof = fabs(Xinit.GetBlock(1)[dof] - (*NewS)[dof]);
                    if ( bdr_error_dof > max_bdr_error )
                        max_bdr_error = bdr_error_dof;

                    if (bdr_error_dof > 1.0e-11)
                        std::cout << "dof " << dof << ": ex_val = " << Xinit.GetBlock(1)[dof]
                                     <<  ", val = " << (*NewS)[dof] << ", s_exact_val = " << (*S_exact)[dof] << "\n";
                }
            }

            if (max_bdr_error > 1.0e-14)
                std::cout << "Error, boundary values for the solution (S) are wrong:"
                             " max_bdr_error = " << max_bdr_error << "\n";

            // 13. Extract the parallel grid function corresponding to the finite element
            //     approximation X. This is the local solution on each processor. Compute
            //     L2 error norms.

            int order_quad = max(2, 2*feorder+1);
            const IntegrationRule *irs[Geometry::NumGeom];
            for (int i=0; i < Geometry::NumGeom; ++i)
            {
               irs[i] = &(IntRules.Get(i, order_quad));
            }

            // Computing error for S

            double err_S = NewS->ComputeL2Error((*Mytest.GetU()), irs);
            double norm_S = ComputeGlobalLpNorm(2, (*Mytest.GetU()), *pmesh, irs);
            if (verbose)
            {
                std::cout << "|| S_h - S_ex || / || S_ex || = " <<
                             err_S / norm_S << "\n";
            }
        }
        /////////////////////////////////////////////////////////
    }

    chrono.Stop();
    if (verbose)
        std::cout << "Errors in USE_AS_A_PREC were computed in " << chrono.RealTime() <<" seconds.\n";
    chrono.Clear();
    chrono.Start();

#else // for USE_AS_A_PREC

    if (verbose)
        std::cout << "\nCalling the new multilevel solver \n";

    chrono.Clear();
    chrono.Start();

    BlockVector NewRhs(new_trueoffsets);
    NewRhs = 0.0;

    if (numblocks_funct > 1)
    {
        if (verbose)
            std::cout << "This place works only for homogeneous boundary conditions \n";
        ParLinearForm *secondeqn_rhs;
        if (strcmp(space_for_S,"H1") == 0 || !eliminateS)
        {
            secondeqn_rhs = new ParLinearForm(H_space_lvls[0]);
            secondeqn_rhs->AddDomainIntegrator(new GradDomainLFIntegrator(*Mytest.bf));
            secondeqn_rhs->Assemble();
            secondeqn_rhs->ParallelAssemble(NewRhs.GetBlock(1));

            delete secondeqn_rhs;

            for ( int i = 0; i < EssBdrTrueDofs_Funct_lvls[0][1]->Size(); ++i)
            {
                int bdrtdof = (*EssBdrTrueDofs_Funct_lvls[0][1])[i];
                NewRhs.GetBlock(1)[bdrtdof] = 0.0;
            }

        }
    }

    BlockVector NewX(new_trueoffsets);
    NewX = 0.0;

    MFEM_ASSERT(CheckConstrRes(ParticSol.GetBlock(0), *Constraint_global, &Floc, "in the main code for the ParticSol"), "blablabla");

    NewSolver.SetInitialGuess(ParticSol);
    //NewSolver.SetUnSymmetric(); // FIXME: temporarily, for debugging purposes!

    if (verbose)
        NewSolver.PrintAllOptions();

    chrono.Stop();
    if (verbose)
        std::cout << "NewSolver was prepared for solving in " << chrono.RealTime() <<" seconds.\n";
    chrono.Clear();
    chrono.Start();

    NewSolver.Mult(NewRhs, NewX);

    chrono.Stop();

    if (verbose)
    {
        std::cout << "Linear solver (new solver only) took " << chrono.RealTime() << "s. \n";
    }



#ifdef TIMING
    double temp_sum;
    /*
    for (int i = 0; i < num_procs; ++i)
    {
        if (myid == i && myid % 10 == 0)
        {
            std::cout << "I am " << myid << "\n";
            std::cout << "Look at my list for mult timings: \n";

            for (list<double>::iterator i = Times_mult->begin(); i != Times_mult->end(); ++i)
                std::cout << *i << " ";
            std::cout << "\n" << std::flush;
        }
        MPI_Barrier(comm);
    }
    */
    temp_sum = 0.0;
    for (list<double>::iterator i = Times_mult->begin(); i != Times_mult->end(); ++i)
        temp_sum += *i;
    if (verbose)
        std::cout << "time_mult = " << temp_sum << "\n";
    delete Times_mult;
    temp_sum = 0.0;
    for (list<double>::iterator i = Times_solve->begin(); i != Times_solve->end(); ++i)
        temp_sum += *i;
    if (verbose)
        std::cout << "time_solve = " << temp_sum << "\n";
    delete Times_solve;
    temp_sum = 0.0;
    for (list<double>::iterator i = Times_localsolve->begin(); i != Times_localsolve->end(); ++i)
        temp_sum += *i;
    if (verbose)
        std::cout << "time_localsolve = " << temp_sum << "\n";
    delete Times_localsolve;
    for (int l = 0; l < num_levels - 1; ++l)
    {
        temp_sum = 0.0;
        for (list<double>::iterator i = Times_localsolve_lvls[l].begin(); i != Times_localsolve_lvls[l].end(); ++i)
            temp_sum += *i;
        if (verbose)
            std::cout << "time_localsolve lvl " << l << " = " << temp_sum << "\n";
    }
    //delete Times_localsolve_lvls;

    temp_sum = 0.0;
    for (list<double>::iterator i = Times_smoother->begin(); i != Times_smoother->end(); ++i)
        temp_sum += *i;
    if (verbose)
        std::cout << "time_smoother = " << temp_sum << "\n";
    delete Times_smoother;

    for (int l = 0; l < num_levels - 1; ++l)
    {
        temp_sum = 0.0;
        for (list<double>::iterator i = Times_smoother_lvls[l].begin(); i != Times_smoother_lvls[l].end(); ++i)
            temp_sum += *i;
        if (verbose)
            std::cout << "time_smoother lvl " << l << " = " << temp_sum << "\n";
    }
    if (verbose)
        std::cout << "\n";
    //delete Times_smoother_lvls;
#ifdef WITH_SMOOTHERS
    for (int l = 0; l < num_levels - 1; ++l)
    {
        if (verbose)
        {
           std::cout << "Internal timing of the smoother at level " << l << ": \n";
           std::cout << "global mult time: " << ((HcurlGSSSmoother*)Smoothers_lvls[l])->GetGlobalMultTime() << " \n" << std::flush;
           std::cout << "internal mult time: " << ((HcurlGSSSmoother*)Smoothers_lvls[l])->GetInternalMultTime() << " \n" << std::flush;
           std::cout << "before internal mult time: " << ((HcurlGSSSmoother*)Smoothers_lvls[l])->GetBeforeIntMultTime() << " \n" << std::flush;
           std::cout << "after internal mult time: " << ((HcurlGSSSmoother*)Smoothers_lvls[l])->GetAfterIntMultTime() << " \n" << std::flush;
        }
    }
#endif
    temp_sum = 0.0;
    for (list<double>::iterator i = Times_coarsestproblem->begin(); i != Times_coarsestproblem->end(); ++i)
        temp_sum += *i;
    if (verbose)
        std::cout << "time_coarsestproblem = " << temp_sum << "\n";
    delete Times_coarsestproblem;

    MPI_Barrier(comm);
    temp_sum = 0.0;
    for (list<double>::iterator i = Times_resupdate->begin(); i != Times_resupdate->end(); ++i)
        temp_sum += *i;
    if (verbose)
        std::cout << "time_resupdate = " << temp_sum << "\n";
    delete Times_resupdate;

    temp_sum = 0.0;
    for (list<double>::iterator i = Times_fw->begin(); i != Times_fw->end(); ++i)
        temp_sum += *i;
    if (verbose)
        std::cout << "time_fw = " << temp_sum << "\n";
    delete Times_fw;
    temp_sum = 0.0;
    for (list<double>::iterator i = Times_up->begin(); i != Times_up->end(); ++i)
        temp_sum += *i;
    if (verbose)
        std::cout << "time_up = " << temp_sum << "\n";
    delete Times_up;
#endif

    NewSigmahat->Distribute(&(NewX.GetBlock(0)));

    // FIXME: remove this
    {
        const Array<int> *temp = EssBdrDofs_Funct_lvls[0][0];

        for ( int tdof = 0; tdof < temp->Size(); ++tdof)
        {
            if ( (*temp)[tdof] != 0 && fabs( (*NewSigmahat)[tdof]) > 1.0e-14 )
                std::cout << "bnd cnd is violated for NewSigmahat! value = "
                          << (*NewSigmahat)[tdof]
                          << "exact val = " << (*sigma_exact_finest)[tdof] << ", index = " << tdof << "\n";
        }
    }

    if (verbose)
        std::cout << "Solution computed via the new solver \n";

    double max_bdr_error = 0;
    for ( int dof = 0; dof < Xinit.GetBlock(0).Size(); ++dof)
    {
        if ( (*EssBdrDofs_Funct_lvls[0][0])[dof] != 0.0)
        {
            //std::cout << "ess dof index: " << dof << "\n";
            double bdr_error_dof = fabs(Xinit.GetBlock(0)[dof] - (*NewSigmahat)[dof]);
            if ( bdr_error_dof > max_bdr_error )
                max_bdr_error = bdr_error_dof;
        }
    }

    if (max_bdr_error > 1.0e-14)
        std::cout << "Error, boundary values for the solution (sigma) are wrong:"
                     " max_bdr_error = " << max_bdr_error << "\n";
    {
        int order_quad = max(2, 2*feorder+1);
        const IntegrationRule *irs[Geometry::NumGeom];
        for (int i = 0; i < Geometry::NumGeom; ++i)
        {
            irs[i] = &(IntRules.Get(i, order_quad));
        }

        double norm_sigma = ComputeGlobalLpNorm(2, *Mytest.GetSigma(), *pmesh, irs);
        double err_newsigmahat = NewSigmahat->ComputeL2Error(*Mytest.GetSigma(), irs);
        if (verbose)
        {
            if ( norm_sigma > MYZEROTOL )
                cout << "|| new sigma_h - sigma_ex || / || sigma_ex || = " << err_newsigmahat / norm_sigma << endl;
            else
                cout << "|| new sigma_h || = " << err_newsigmahat << " (sigma_ex = 0)" << endl;
        }

        DiscreteLinearOperator Div(R_space, W_space);
        Div.AddDomainInterpolator(new DivergenceInterpolator());
        ParGridFunction DivSigma(W_space);
        Div.Assemble();
        Div.Mult(*NewSigmahat, DivSigma);

        double err_div = DivSigma.ComputeL2Error(*Mytest.GetRhs(),irs);
        double norm_div = ComputeGlobalLpNorm(2, *Mytest.GetRhs(), *pmesh, irs);

        if (verbose)
        {
            cout << "|| div (new sigma_h - sigma_ex) || / ||div (sigma_ex)|| = "
                      << err_div/norm_div  << "\n";
        }
    }

    //////////////////////////////////////////////////////
    if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
    {
        NewS->Distribute(&(NewX.GetBlock(1)));

        double max_bdr_error = 0;
        for ( int dof = 0; dof < Xinit.GetBlock(1).Size(); ++dof)
        {
            if ( (*EssBdrDofs_Funct_lvls[0][1])[dof] != 0.0)
            {
                //std::cout << "ess dof index: " << dof << "\n";
                double bdr_error_dof = fabs(Xinit.GetBlock(1)[dof] - (*NewS)[dof]);
                if ( bdr_error_dof > max_bdr_error )
                    max_bdr_error = bdr_error_dof;
            }
        }

        if (max_bdr_error > 1.0e-14)
            std::cout << "Error, boundary values for the solution (S) are wrong:"
                         " max_bdr_error = " << max_bdr_error << "\n";

        // 13. Extract the parallel grid function corresponding to the finite element
        //     approximation X. This is the local solution on each processor. Compute
        //     L2 error norms.

        int order_quad = max(2, 2*feorder+1);
        const IntegrationRule *irs[Geometry::NumGeom];
        for (int i=0; i < Geometry::NumGeom; ++i)
        {
           irs[i] = &(IntRules.Get(i, order_quad));
        }

        // Computing error for S

        double err_S = NewS->ComputeL2Error((*Mytest.GetU()), irs);
        double norm_S = ComputeGlobalLpNorm(2, (*Mytest.GetU()), *pmesh, irs);
        if (verbose)
        {
            std::cout << "|| S_h - S_ex || / || S_ex || = " <<
                         err_S / norm_S << "\n";
        }
    }
    /////////////////////////////////////////////////////////

    chrono.Stop();


    if (verbose)
        std::cout << "\n";
#endif // for else for USE_AS_A_PREC

#ifdef VISUALIZATION
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
            S_h_sock << "solution\n" << *pmesh << *NewS << "window_title 'S_h'"
                   << endl;

            *NewS -= *S_exact;
            socketstream S_diff_sock(vishost, visport);
            S_diff_sock << "parallel " << num_procs << " " << myid << "\n";
            S_diff_sock.precision(8);
            MPI_Barrier(pmesh->GetComm());
            S_diff_sock << "solution\n" << *pmesh << *NewS << "window_title 'S_h - S_exact'"
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
#endif

    MPI_Finalize();
    return 0;

#ifdef MINSOLVER_TESTING
    if (verbose)
        std::cout << "Testing behavior of the minimization solver for this nice problem \n";

    using FormulType = CFOSLSFormulation_Laplace;
    using FEFormulType = CFOSLSFEFormulation_HdivH1L2_Laplace;
    using BdrCondsType = BdrConditions_CFOSLS_HdivH1Laplace;
    using ProblemType = FOSLSProblem_HdivH1lapl;

    FOSLSFormulation * formulat = new FormulType (dim, numsol, verbose);
    FOSLSFEFormulation * fe_formulat = new FEFormulType(*formulat, feorder);
    BdrConditions * bdr_conds = new BdrCondsType(*pmesh);

    int nlevels = ref_levels + 1;
    GeneralHierarchy * hierarchy = new GeneralHierarchy(nlevels, *pmesh_lvls[num_levels - 1], 0, verbose);
    hierarchy->ConstructDivfreeDops();
    hierarchy->ConstructDofTrueDofs();

    FOSLSProblem* problem_mgtools = hierarchy->BuildDynamicProblem<ProblemType>
            (*bdr_conds, *fe_formulat, prec_option, verbose);
    hierarchy->AttachProblem(problem_mgtools);

    ComponentsDescriptor * descriptor;
    {
        bool with_Schwarz = true;
        bool optimized_Schwarz = true;
        bool with_Hcurl = true;
        bool with_coarsest_partfinder = true;
        bool with_coarsest_hcurl = true;
        bool with_monolithic_GS = false;
        descriptor = new ComponentsDescriptor(with_Schwarz, optimized_Schwarz,
                                                      with_Hcurl, with_coarsest_partfinder,
                                                      with_coarsest_hcurl, with_monolithic_GS);
    }
    MultigridToolsHierarchy * mgtools_hierarchy =
            new MultigridToolsHierarchy(*hierarchy, problem_mgtools->GetAttachedIndex(), *descriptor);

    GeneralMinConstrSolver * MinSolver;
    {
        bool with_local_smoothers = true;
        bool optimized_localsolvers = true;
        bool with_hcurl_smoothers = true;

        int stopcriteria_type = 1;

        int numblocks_funct = numblocks; // only for this example

        int size_funct = problem_mgtools->GetTrueOffsetsFunc()[numblocks_funct];
        MinSolver = new GeneralMinConstrSolver(size_funct, *mgtools_hierarchy, with_local_smoothers,
                                         optimized_localsolvers, with_hcurl_smoothers, stopcriteria_type, verbose);
    }

    MinSolver->SetRelTol(1.0e-6);
    MinSolver->SetMaxIter(200);
    MinSolver->SetPrintLevel(2);
    MinSolver->SetStopCriteriaType(0);

    MinSolver->SetInitialGuess(ParticSol);
    MinSolver->SetConstrRhs(problem_mgtools->GetRhs().GetBlock(numblocks_funct));
    //MinSolver->SetUnSymmetric();

    if (verbose)
        MinSolver->PrintAllOptions();

    Vector NewRhs(MinSolver->Size());
    NewRhs = 0.0;

    BlockVector divfree_part(problem_mgtools->GetTrueOffsetsFunc());
    MinSolver->Mult(NewRhs, divfree_part);

    BlockVector& problem_sol = problem_mgtools->GetSol();
    for (int blk = 0; blk < numblocks_funct; ++blk)
    {
        //problem_sol.GetBlock(blk) = ParticSol.GetBlock(blk);
        //problem_sol.GetBlock(blk) += divfree_part.GetBlock(blk);

        problem_sol.GetBlock(blk) = divfree_part.GetBlock(blk);
    }

    bool checkbnd = true;
    problem_mgtools->ComputeError(problem_sol, verbose, checkbnd);

    BlockVector tmp1(problem_mgtools->GetTrueOffsetsFunc());
    for (int blk = 0; blk < numblocks_funct; ++blk)
        tmp1.GetBlock(blk) = problem_mgtools->GetExactSolProj()->GetBlock(blk);

    CheckFunctValue(comm,*MinSolver->GetFunctOp(0), NULL, tmp1,
                    "for the projection of the exact solution ", verbose);

#endif

    chrono.Stop();
    if (verbose)
        std::cout << "Deallocating memory \n";
    chrono.Clear();
    chrono.Start();

    for (int l = 0; l < num_levels; ++l)
    {
        delete BdrDofs_Funct_lvls[l][0];
        delete EssBdrDofs_Funct_lvls[l][0];
        delete EssBdrTrueDofs_Funct_lvls[l][0];
#ifndef HCURL_COARSESOLVER
        if (l < num_levels - 1)
        {
            delete EssBdrDofs_Hcurl[l];
            delete EssBdrTrueDofs_Hcurl[l];
        }
#else
        delete EssBdrDofs_Hcurl[l];
        delete EssBdrTrueDofs_Hcurl[l];
#endif
        if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
        {
            delete BdrDofs_Funct_lvls[l][1];
            delete EssBdrDofs_Funct_lvls[l][1];
            delete EssBdrTrueDofs_Funct_lvls[l][1];
            delete EssBdrDofs_H1[l];
        }

        if (l < num_levels - 1)
        {
            if (LocalSolver_partfinder_lvls)
                if ((*LocalSolver_partfinder_lvls)[l])
                    delete (*LocalSolver_partfinder_lvls)[l];
        }

#ifdef WITH_SMOOTHERS
        if (l < num_levels - 1)
            if (Smoothers_lvls[l])
                delete Smoothers_lvls[l];
#endif

        if (l < num_levels - 1)
            delete Divfree_hpmat_mod_lvls[l];
        for (int blk1 = 0; blk1 < Funct_hpmat_lvls[l]->NumRows(); ++blk1)
            for (int blk2 = 0; blk2 < Funct_hpmat_lvls[l]->NumCols(); ++blk2)
                if ((*Funct_hpmat_lvls[l])(blk1,blk2))
                    delete (*Funct_hpmat_lvls[l])(blk1,blk2);
        //delete Funct_hpmat_lvls[l];

        if (l < num_levels - 1)
        {
            delete Element_dofs_Func[l];
            delete P_Func[l];
            delete TrueP_Func[l];
        }

        if (l == 0)
            // this happens because for l = 0 object is created in a different way,
            // thus it doesn't own the blocks and cannot delete it from destructor
            for (int blk1 = 0; blk1 < Funct_mat_lvls[l]->NumRowBlocks(); ++blk1)
                for (int blk2 = 0; blk2 < Funct_mat_lvls[l]->NumColBlocks(); ++blk2)
                    delete &(Funct_mat_lvls[l]->GetBlock(blk1,blk2));
        delete Funct_mat_lvls[l];
        delete Funct_mat_offsets_lvls[l];

        delete Constraint_mat_lvls[l];

        delete Divfree_mat_lvls[l];

        delete R_space_lvls[l];
        delete W_space_lvls[l];
        delete C_space_lvls[l];
        if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
            delete H_space_lvls[l];
        delete pmesh_lvls[l];

        if (l < num_levels - 1)
        {
            delete P_W[l];
            delete P_WT[l];
            delete P_R[l];
            delete P_C_lvls[l];
            if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
                delete P_H_lvls[l];
            delete TrueP_R[l];
            if (prec_is_MG)
                delete TrueP_C[l];
            if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
                delete TrueP_H[l];

            delete Element_dofs_R[l];
            delete Element_dofs_W[l];
            if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
                delete Element_dofs_H[l];
        }

        if (l < num_levels - 1)
        {
            delete row_offsets_El_dofs[l];
            delete col_offsets_El_dofs[l];
            delete row_offsets_P_Func[l];
            delete col_offsets_P_Func[l];
            delete row_offsets_TrueP_Func[l];
            delete col_offsets_TrueP_Func[l];
        }

    }

    delete LocalSolver_partfinder_lvls;
    delete LocalSolver_lvls;

    for (int blk1 = 0; blk1 < Funct_global->NumRowBlocks(); ++blk1)
        for (int blk2 = 0; blk2 < Funct_global->NumColBlocks(); ++blk2)
            if (Funct_global->IsZeroBlock(blk1, blk2) == false)
                delete &(Funct_global->GetBlock(blk1,blk2));
    delete Funct_global;

    delete Functrhs_global;

    delete hdiv_coll;
    delete R_space;
    delete l2_coll;
    delete W_space;
    delete hdivfree_coll;
    delete C_space;

    delete h1_coll;
    delete H_space;

    delete CoarsestSolver_partfinder;
#ifdef HCURL_COARSESOLVER
    delete CoarsestSolver;
#endif

    delete sigma_exact_finest;
    if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
        delete S_exact_finest;

    delete NewSigmahat;
    if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
        delete NewS;

#ifdef USE_AS_A_PREC
    delete Atest;
    if (strcmp(space_for_S,"H1") == 0)
    {
        delete Ctest;
        delete Btest;
        delete BTtest;
    }
    delete BlockMattest;
#endif

#ifdef OLD_CODE
    delete gform;
    delete Bdiv;

    delete S_exact;
    delete sigma_exact;
    delete opdivfreepart;
    delete sigma;
    delete S;

    delete Sigmahat;
    delete u;

#ifdef   USE_CURLMATRIX
    if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
        delete qform;
    delete MainOp;
    delete Mblock;
    delete M;
    delete A;
    if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
    {
        delete C;
        delete CHT;
        delete CH;
        delete B;
        delete BT;
    }
#endif
    if(dim<=4)
    {
        if (prec_is_MG)
        {
            if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
            {
                if (monolithicMG)
                {
                    for (int l = 0; l < num_levels; ++l)
                    {
                        delete offsets_f[l];
                        delete offsets_c[l];
                    }
                }
                else
                {
                    for ( int blk = 0; blk < ((BlockDiagonalPreconditioner*)prec)->NumBlocks(); ++blk)
                        delete ((Multigrid*)(&(((BlockDiagonalPreconditioner*)prec)->GetDiagonalBlock(blk))));
                            //if (&(((BlockDiagonalPreconditioner*)prec)->GetDiagonalBlock(blk)))
                                //delete &(((BlockDiagonalPreconditioner*)prec)->GetDiagonalBlock(blk));
                }
            }
            else
                for ( int blk = 0; blk < ((BlockDiagonalPreconditioner*)prec)->NumBlocks(); ++blk)
                    delete ((Multigrid*)(&(((BlockDiagonalPreconditioner*)prec)->GetDiagonalBlock(blk))));
                        //if (&(((BlockDiagonalPreconditioner*)prec)->GetDiagonalBlock(blk)))
                            //delete &(((BlockDiagonalPreconditioner*)prec)->GetDiagonalBlock(blk));
        }
        else
        {
            for ( int blk = 0; blk < ((BlockDiagonalPreconditioner*)prec)->NumBlocks(); ++blk)
                    if (&(((BlockDiagonalPreconditioner*)prec)->GetDiagonalBlock(blk)))
                        delete &(((BlockDiagonalPreconditioner*)prec)->GetDiagonalBlock(blk));
        }
    }

    delete prec;
    for (int i = 0; i < P.Size(); ++i)
        delete P[i];

#endif // end of #ifdef OLD_CODE in the memory deallocating

    chrono.Stop();
    if (verbose)
        std::cout << "Deallocation of memory was done in " << chrono.RealTime() <<" seconds.\n";
    chrono.Clear();
    chrono.Start();

    chrono_total.Stop();
    if (verbose)
        std::cout << "Total time consumed was " << chrono_total.RealTime() <<" seconds.\n";

    MPI_Finalize();
    return 0;
}
