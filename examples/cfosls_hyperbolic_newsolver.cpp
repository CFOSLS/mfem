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

#define NEW_INTERFACE
#define NEW_INTERFACE2

// (de)activates solving of the discrete global problem
#define OLD_CODE

#define WITH_DIVCONSTRAINT_SOLVER

// switches on/off usage of smoother in the new minimization solver
// in parallel GS smoother works a little bit different from serial
#define WITH_SMOOTHERS

// activates using the new interface to local problem solvers
// via a separated class called LocalProblemSolver
#define SOLVE_WITH_LOCALSOLVERS

// activates a test where new solver is used as a preconditioner
#define USE_AS_A_PREC

#define HCURL_COARSESOLVER

// activates constraint residual check after each iteration of the minimization solver
#define CHECK_CONSTR

#define CHECK_BNDCND

#define BND_FOR_MULTIGRID

//#define COARSEPREC_AMS

// activates more detailed timing of the new multigrid code
//#define TIMING

#ifdef TIMING
#undef CHECK_LOCALSOLVE
#undef CHECK_CONSTR
#undef CHECK_BNDCND
#endif

#define MYZEROTOL (1.0e-13)

//#define WITH_PENALTY

//#define ONLY_DIVFREEPART

//#define K_IDENTITY

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
    int numcurl         = 0;

    int ser_ref_levels  = 1;
    int par_ref_levels  = 2;

    const char *space_for_S = "H1";    // "H1" or "L2"

    // Hdiv-H1 case
    using FormulType = CFOSLSFormulation_HdivH1Hyper;
    using FEFormulType = CFOSLSFEFormulation_HdivH1Hyper;
    using BdrCondsType = BdrConditions_CFOSLS_HdivH1_Hyper;
    using ProblemType = FOSLSProblem_HdivH1L2hyp;

    /*
    // Hdiv-L2 case
    using FormulType = CFOSLSFormulation_HdivL2Hyper;
    using FEFormulType = CFOSLSFEFormulation_HdivL2Hyper;
    using BdrCondsType = BdrConditions_CFOSLS_HdivL2_Hyper;
    using ProblemType = FOSLSProblem_HdivL2L2hyp;
    */

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
        cout << "Solving CFOSLS Transport equation with MFEM & hypre, div-free approach, minimization solver \n";

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

#ifdef COARSEPREC_AMS
    if (verbose)
        std::cout << "COARSEPREC_AMS active \n";
#else
    if (verbose)
        std::cout << "COARSEPREC_AMS passive \n";
#endif

#ifdef COMPARE_MG
    if (verbose)
        std::cout << "COMPARE_MG active \n";
#else
    if (verbose)
        std::cout << "COMPARE_MG passive \n";
#endif

#ifdef WITH_PENALTY
    if (verbose)
        std::cout << "WITH_PENALTY active \n";
#else
    if (verbose)
        std::cout << "WITH_PENALTY passive \n";
#endif

#ifdef K_IDENTITY
    if (verbose)
        std::cout << "K_IDENTITY active \n";
#else
    if (verbose)
        std::cout << "K_IDENTITY passive \n";
#endif

    std::cout << std::flush;
    MPI_Barrier(MPI_COMM_WORLD);

    MFEM_ASSERT(strcmp(space_for_S,"H1") == 0 || strcmp(space_for_S,"L2") == 0, "Space for S must be H1 or L2!\n");
    MFEM_ASSERT(!(strcmp(space_for_S,"L2") == 0 && !eliminateS), "Case: L2 space for S and S is not eliminated is working incorrectly, non pos.def. matrix. \n");

    if (verbose)
    {
        if (strcmp(space_for_S,"H1") == 0)
            std::cout << "Space for S: H1 \n";
        else
            std::cout << "Space for S: L2 \n";

        if (strcmp(space_for_S,"L2") == 0)
        {
            std::cout << "S is ";
            if (!eliminateS)
                std::cout << "not ";
            std::cout << "eliminated from the system \n";
        }
    }

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

    Transport_test_divfree Mytest(nDimensions, numsol, numcurl);

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
    int max_num_iter = 2000;
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

#ifdef WITH_PENALTY
    if (verbose)
        std::cout << "regularization is ON \n";
    double h_min, h_max, kappa_min, kappa_max;
    pmesh->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);
    if (verbose)
        std::cout << "coarse mesh steps: min " << h_min << " max " << h_max << "\n";

    double reg_param;
    reg_param = 1.0 * h_min * h_min;
    reg_param *= 1.0 / (pow(2.0, par_ref_levels) * pow(2.0, par_ref_levels));
    if (verbose)
        std::cout << "regularization parameter: " << reg_param << "\n";
    ConstantCoefficient reg_coeff(reg_param);
#endif


    MFEM_ASSERT(!(aniso_refine && (with_multilevel || nDimensions == 4)),"Anisotropic refinement works only in 3D and without multilevel algorithm \n");

    ////////////////////////////////// new

    int dim = nDimensions;

    Array<int> ess_bdrSigma(pmesh->bdr_attributes.Max());
    ess_bdrSigma = 0;
    if (strcmp(space_for_S,"L2") == 0) // S is from L2, so we impose bdr condition for sigma at t = 0
    {
        ess_bdrSigma[0] = 1;
        //ess_bdrSigma = 1;
        //ess_bdrSigma[pmesh->bdr_attributes.Max()-1] = 0;
    }

    Array<int> ess_bdrS(pmesh->bdr_attributes.Max());
    ess_bdrS = 0;
    if (strcmp(space_for_S,"H1") == 0) // S is from H1
    {
        ess_bdrS[0] = 1; // t = 0
        //ess_bdrS = 1;
    }

    Array<int> all_bdrSigma(pmesh->bdr_attributes.Max());
    all_bdrSigma = 1;

    Array<int> all_bdrS(pmesh->bdr_attributes.Max());
    all_bdrS = 1;

    int ref_levels = par_ref_levels;

    int num_levels = ref_levels + 1;

    chrono.Clear();
    chrono.Start();

    int numblocks_funct = 1;
    if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
        numblocks_funct++;

    FOSLSFormulation * formulat = new FormulType (dim, numsol, verbose);
    FOSLSFEFormulation * fe_formulat = new FEFormulType(*formulat, feorder);
    BdrConditions * bdr_conds = new BdrCondsType(*pmesh);

    int nlevels = ref_levels + 1;
    GeneralHierarchy * hierarchy = new GeneralHierarchy(nlevels, *pmesh, 0, verbose);
    hierarchy->ConstructDivfreeDops();
    hierarchy->ConstructDofTrueDofs();

    FOSLSProblem * problem = new ProblemType(*pmesh, *bdr_conds,
                                             *fe_formulat, prec_option, verbose);

    Array<int> &essbdr_attribs_Hcurl = problem->GetBdrConditions().GetBdrAttribs(0);

    std::vector<Array<int>*>& essbdr_attribs = problem->GetBdrConditions().GetAllBdrAttribs();

    std::vector<Array<int>*>& fullbdr_attribs = problem->GetBdrConditions().GetFullBdrAttribs();

    const Array<SpaceName>& space_names_problem = problem->GetFEformulation().GetFormulation()->GetSpacesDescriptor();

    Array<SpaceName> space_names_funct(numblocks_funct);
    space_names_funct[0] = SpaceName::HDIV;
    if (strcmp(space_for_S,"H1") == 0)
        space_names_funct[1] = SpaceName::H1;

    Array<SpaceName> space_names_divfree(numblocks_funct);
    space_names_divfree[0] = SpaceName::HCURL;
    if (strcmp(space_for_S,"H1") == 0)
        space_names_divfree[1] = SpaceName::H1;

    std::vector< Array<int>* > coarsebnd_indces_divfree_lvls(num_levels);
    for (int l = 0; l < num_levels - 1; ++l)
    {
        std::vector<Array<int>* > &essbdr_tdofs_hcurlfunct =
                hierarchy->GetEssBdrTdofsOrDofs("tdof", space_names_divfree, essbdr_attribs, l + 1);

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
            shift_tdofs_indices += hierarchy->GetSpace(space_names_divfree[blk], l + 1)->TrueVSize();
        }

    }

    Array<BlockOperator*> BlockP_mg_nobnd(nlevels - 1);
    Array<Operator*> P_mg(nlevels - 1);
    Array<BlockOperator*> BlockOps_mg(nlevels);
    Array<Operator*> Ops_mg(nlevels);
    Array<Operator*> Smoo_mg(nlevels - 1);
    Operator* CoarseSolver_mg;

    std::vector<const Array<int> *> offsets(nlevels);
    offsets[0] = &hierarchy->ConstructTrueOffsetsforFormul(0, space_names_divfree);

    BlockOperator * orig_op = problem->GetOp();
    const HypreParMatrix * divfree_dop = hierarchy->GetDivfreeDop(0);

    HypreParMatrix * divfree_dop_mod = CopyHypreParMatrix(*divfree_dop);

    Eliminate_ib_block(*divfree_dop_mod,
                       hierarchy->GetEssBdrTdofsOrDofs("tdof", SpaceName::HCURL, essbdr_attribs_Hcurl, 0),
                       hierarchy->GetEssBdrTdofsOrDofs("tdof", SpaceName::HDIV, *essbdr_attribs[0], 0));

    // transferring the first block of the functional oiperator from hdiv into hcurl
    BlockOperator * divfree_funct_op = new BlockOperator(*offsets[0]);

    HypreParMatrix * op_00 = dynamic_cast<HypreParMatrix*>(&(orig_op->GetBlock(0,0)));
    HypreParMatrix * A00 = RAP(divfree_dop_mod, op_00, divfree_dop_mod);
    A00->CopyRowStarts();
    A00->CopyColStarts();
    divfree_funct_op->SetBlock(0,0, A00);

    HypreParMatrix * A10, * A01, *op_11;
    if (strcmp(space_for_S,"H1") == 0)
    {
        op_11 = dynamic_cast<HypreParMatrix*>(&(orig_op->GetBlock(1,1)));

        HypreParMatrix * op_10 = dynamic_cast<HypreParMatrix*>(&(orig_op->GetBlock(1,0)));
        A10 = ParMult(op_10, divfree_dop_mod);
        A10->CopyRowStarts();
        A10->CopyColStarts();

        A01 = A10->Transpose();
        A01->CopyRowStarts();
        A01->CopyColStarts();

        divfree_funct_op->SetBlock(1,0, A10);
        divfree_funct_op->SetBlock(0,1, A01);
        divfree_funct_op->SetBlock(1,1, op_11);
    }

    // setting multigrid components from the older parts of the code
    for (int l = 0; l < num_levels; ++l)
    {
        if (l < num_levels - 1)
        {
            offsets[l + 1] = &hierarchy->ConstructTrueOffsetsforFormul(l + 1, space_names_divfree);
            P_mg[l] = new BlkInterpolationWithBNDforTranspose(
                        *hierarchy->ConstructTruePforFormul(l, space_names_divfree,
                                                            *offsets[l], *offsets[l + 1]),
                        *coarsebnd_indces_divfree_lvls[l],
                        *offsets[l], *offsets[l + 1]);
            BlockP_mg_nobnd[l] = hierarchy->ConstructTruePforFormul(l, space_names_divfree,
                                                                    *offsets[l], *offsets[l + 1]);
        }

        if (l == 0)
            BlockOps_mg[l] = divfree_funct_op;
        else
        {
            BlockOps_mg[l] = new RAPBlockHypreOperator(*BlockP_mg_nobnd[l - 1],
                    *BlockOps_mg[l - 1], *BlockP_mg_nobnd[l - 1], *offsets[l]);

            std::vector<Array<int>* > &essbdr_tdofs_hcurlfunct =
                    hierarchy->GetEssBdrTdofsOrDofs("tdof", space_names_divfree, essbdr_attribs, l);
            EliminateBoundaryBlocks(*BlockOps_mg[l], essbdr_tdofs_hcurlfunct);
        }

        Ops_mg[l] = BlockOps_mg[l];

        if (l < num_levels - 1)
            Smoo_mg[l] = new MonolithicGSBlockSmoother( *BlockOps_mg[l],
                                                        *offsets[l], false, HypreSmoother::Type::l1GS, 1);


        //P_mg[l] = ((MonolithicMultigrid*)prec)->GetInterpolation(l);
        //P_mg[l] = new InterpolationWithBNDforTranspose(
                    //*((MonolithicMultigrid*)prec)->GetInterpolation
                    //(num_levels - 1 - 1 - l), *coarsebnd_indces_divfree_lvls[l]);
        //Ops_mg[l] = ((MonolithicMultigrid*)prec)->GetOp(num_levels - 1 - l);
        //Smoo_mg[l] = ((MonolithicMultigrid*)prec)->GetSmoother(num_levels - 1 - l);

    }
    //CoarseSolver_mg = ((MonolithicMultigrid*)prec)->GetCoarsestSolver();

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

    ((CGSolver*)CoarseSolver_mg)->SetPreconditioner(*CoarsePrec_mg);

    GeneralMultigrid * GeneralMGprec =
            new GeneralMultigrid(nlevels, P_mg, Ops_mg, *CoarseSolver_mg, Smoo_mg);

    if (verbose)
        std::cout << "End of the creating a hierarchy of meshes AND pfespaces \n";

    /////////////////////////  beginning temporary crutches while cleaning up

    Array<ParFiniteElementSpace *> R_space_lvls(num_levels);
    Array<ParFiniteElementSpace *> W_space_lvls(num_levels);
    Array<ParFiniteElementSpace *> H_space_lvls(num_levels);
    Array<ParFiniteElementSpace *> C_space_lvls(num_levels);
    for (int l = 0; l < num_levels; ++l)
    {
        R_space_lvls[l] = hierarchy->GetSpace(SpaceName::HDIV, l);
        W_space_lvls[l] = hierarchy->GetSpace(SpaceName::L2, l);
        H_space_lvls[l] = hierarchy->GetSpace(SpaceName::H1, l);
        if (dim == 3)
            C_space_lvls[l] = hierarchy->GetSpace(SpaceName::HCURL, l);
        else
            C_space_lvls[l] = hierarchy->GetSpace(SpaceName::HDIVSKEW, l);
    }

    ParFiniteElementSpace * S_space;
    if (strcmp(space_for_S,"H1") == 0)
        S_space = H_space_lvls[0];
    else // "L2"
        S_space = W_space_lvls[0];

    ParFiniteElementSpace * R_space = R_space_lvls[0];
    ParFiniteElementSpace * W_space = W_space_lvls[0];
    ParFiniteElementSpace * C_space = C_space_lvls[0];

    //std::vector<std::vector<Array<int>* > > BdrDofs_Funct_lvls(num_levels, std::vector<Array<int>* >(numblocks_funct));
    std::vector<std::vector<Array<int>* > > EssBdrDofs_Funct_lvls(num_levels, std::vector<Array<int>* >(numblocks_funct));
    std::vector<std::vector<Array<int>* > > EssBdrTrueDofs_Funct_lvls(num_levels, std::vector<Array<int>* >(numblocks_funct));
    std::vector<std::vector<Array<int>* > > EssBdrTrueDofs_HcurlFunct_lvls(num_levels, std::vector<Array<int>* >(numblocks_funct));

    std::vector<const Array<int>* > EssBdrTrueDofs_Hcurl(num_levels);
    std::vector<const Array<int>* > EssBdrTrueDofs_H1(num_levels);

    for (int l = 0; l < num_levels; ++l)
    {
        EssBdrTrueDofs_Funct_lvls[l] = hierarchy->GetEssBdrTdofsOrDofs("tdof", space_names_funct, essbdr_attribs, l);
        EssBdrTrueDofs_HcurlFunct_lvls[l] = hierarchy->GetEssBdrTdofsOrDofs("tdof", space_names_divfree, essbdr_attribs, l);
        EssBdrDofs_Funct_lvls[l] = hierarchy->GetEssBdrTdofsOrDofs("dof", space_names_funct, essbdr_attribs, l);
        EssBdrTrueDofs_H1[l] = &hierarchy->GetEssBdrTdofsOrDofs("tdof", SpaceName::H1, *essbdr_attribs[0], l);
        if (dim == 3)
            EssBdrTrueDofs_Hcurl[l] = &hierarchy->GetEssBdrTdofsOrDofs("tdof", SpaceName::HCURL, *essbdr_attribs[0], l);
        else
            EssBdrTrueDofs_Hcurl[l] = &hierarchy->GetEssBdrTdofsOrDofs("tdof", SpaceName::HDIVSKEW, *essbdr_attribs[0], l);
    }

    HypreParMatrix * Constraint_global = (HypreParMatrix*)(&problem->GetOp_nobnd()->GetBlock(numblocks_funct, 0));

    Array<SparseMatrix*> P_W(num_levels - 1);
    for (int l = 0; l < num_levels - 1; ++l)
        P_W[l] = hierarchy->GetPspace(SpaceName::L2, l);

    Array<HypreParMatrix*> TrueP_H(par_ref_levels);
    Array<HypreParMatrix*> TrueP_C(par_ref_levels);
    for (int l = 0; l < num_levels - 1; ++l)
    {
        TrueP_H[num_levels - 2 - l] = hierarchy->GetTruePspace(SpaceName::H1, l);
        if (dim == 3)
            TrueP_C[num_levels - 2 - l] = hierarchy->GetTruePspace(SpaceName::HCURL, l);
        else
            TrueP_C[num_levels - 2 - l] = hierarchy->GetTruePspace(SpaceName::HDIVSKEW, l);
    }

    Array<const HypreParMatrix*> Divfree_hpmat_mod_lvls(num_levels);
    for (int l = 0; l < num_levels; ++l)
        Divfree_hpmat_mod_lvls[l] = hierarchy->GetDivfreeDop(l);

    // Creating global functional matrix
    Array<int> offsets_global(numblocks_funct + 1);
    offsets_global[0] = 0;
    for ( int blk = 0; blk < numblocks_funct; ++blk)
        offsets_global[blk + 1] = hierarchy->GetDofTrueDof(space_names_funct[blk], 0)->Width();
    offsets_global.PartialSum();

    BlockVector * Functrhs_global = new BlockVector(offsets_global);

    Functrhs_global->GetBlock(0) = 0.0;

    ParLinearForm *secondeqn_rhs;
    if (strcmp(space_for_S,"H1") == 0 || !eliminateS)
    {
        secondeqn_rhs = new ParLinearForm(H_space_lvls[0]);
        secondeqn_rhs->AddDomainIntegrator(new GradDomainLFIntegrator(*Mytest.bf));
        secondeqn_rhs->Assemble();

        secondeqn_rhs->ParallelAssemble(Functrhs_global->GetBlock(1));
        for (int tdofind = 0; tdofind < EssBdrDofs_Funct_lvls[0][1]->Size(); ++tdofind)
        {
            int tdof = (*EssBdrDofs_Funct_lvls[0][1])[tdofind];
            Functrhs_global->GetBlock(1)[tdof] = 0.0;
        }
    }

    /////////////////////////  end of temporary crutches

    ParGridFunction * sigma_exact_finest;
    sigma_exact_finest = new ParGridFunction(R_space_lvls[0]);
    sigma_exact_finest->ProjectCoefficient(*Mytest.sigma);
    Vector sigma_exact_truedofs(R_space_lvls[0]->GetTrueVSize());
    sigma_exact_finest->ParallelProject(sigma_exact_truedofs);

    ParGridFunction * S_exact_finest;
    Vector S_exact_truedofs;
    if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
    {
        S_exact_finest = new ParGridFunction(H_space_lvls[0]);
        S_exact_finest->ProjectCoefficient(*Mytest.scalarS);
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
    HypreParMatrix *Bdiv;

    ParLinearForm *gform;
    Vector sigmahat_pau;

    if (with_multilevel)
    {
        if (verbose)
            std::cout << "Using multilevel algorithm for finding a particular solution \n";

        ConstantCoefficient k(1.0);

        SparseMatrix *M_local;
        if (useM_in_divpart)
        {
            ParBilinearForm *Massform = new ParBilinearForm(hierarchy->GetSpace(SpaceName::HDIV, 0));
            Massform->AddDomainIntegrator(new VectorFEMassIntegrator(k));
            Massform->Assemble();
            Massform->Finalize();
            M_local = Massform->LoseMat();
            delete Massform;
        }
        else
            M_local = NULL;

        ParMixedBilinearForm *DivForm(new ParMixedBilinearForm
                                      (hierarchy->GetSpace(SpaceName::HDIV, 0),
                                       hierarchy->GetSpace(SpaceName::L2, 0)));
        DivForm->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
        DivForm->Assemble();
        DivForm->Finalize();
        Bdiv = DivForm->ParallelAssemble();
        SparseMatrix *B_local = DivForm->LoseMat();

        //Right hand size
        gform = new ParLinearForm(hierarchy->GetSpace(SpaceName::L2, 0));
        gform->AddDomainIntegrator(new DomainLFIntegrator(*problem->GetFEformulation().
                                                          GetFormulation()->GetTest()->GetRhs()));
        gform->Assemble();

        Vector F_fine(hierarchy->GetSpace(SpaceName::L2, 0)->GetVSize());
        Vector G_fine(hierarchy->GetSpace(SpaceName::HDIV, 0)->GetVSize());

        F_fine = *gform;
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

        const Array<int>& coarse_essbdr_dofs_Hdiv = hierarchy->GetEssBdrTdofsOrDofs
                ("dof", SpaceName::HDIV, *essbdr_attribs[0], num_levels - 1);

        DivPart divp;

        divp.div_part(ref_levels,
                      M_local, B_local,
                      G_fine,
                      F_fine,
                      P_L2_lvls, P_Hdiv_lvls, e_AE_lvls,
                      el2dofs_R,
                      el2dofs_W,
                      hierarchy->GetDofTrueDof(SpaceName::HDIV, num_levels - 1),
                      hierarchy->GetDofTrueDof(SpaceName::L2, num_levels - 1),
                      hierarchy->GetSpace(SpaceName::HDIV, num_levels - 1)->GetDofOffsets(),
                      hierarchy->GetSpace(SpaceName::L2, num_levels - 1)->GetDofOffsets(),
                      //R_space_lvls[num_levels - 1]->GetDofOffsets(),
                      //W_space_lvls[num_levels - 1]->GetDofOffsets(),
                      sigmahat_pau,
                      coarse_essbdr_dofs_Hdiv);

        delete DivForm;

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

        sigma_exact = new ParGridFunction(hierarchy->GetSpace(SpaceName::HDIV, 0));
        sigma_exact->ProjectCoefficient(*problem->GetFEformulation().
                                        GetFormulation()->GetTest()->GetSigma());
        //sigma_exact->ProjectCoefficient(*Mytest.sigma);

        gform = new ParLinearForm(hierarchy->GetSpace(SpaceName::L2, 0));
        gform->AddDomainIntegrator(new DomainLFIntegrator(*problem->GetFEformulation().
                                                          GetFormulation()->GetTest()->GetRhs()));
        //gform->AddDomainIntegrator(new DomainLFIntegrator(*Mytest.scalardivsigma));
        gform->Assemble();

        Bblock = new ParMixedBilinearForm(hierarchy->GetSpace(SpaceName::HDIV, 0),
                                          hierarchy->GetSpace(SpaceName::L2, 0));
        Bblock->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
        Bblock->Assemble();
        Bblock->EliminateTrialDofs(*essbdr_attribs[0], *sigma_exact, *gform);

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

        Vector * Temphat = new Vector(hierarchy->GetSpace(SpaceName::L2, 0)->TrueVSize());
        *Temphat = 0.0;
        solver.Mult(*Rhs, *Temphat);

        Vector * Temp = new Vector(hierarchy->GetSpace(SpaceName::HDIV, 0)->TrueVSize());
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
        constrfform->AddDomainIntegrator(new DomainLFIntegrator(*Mytest.scalardivsigma));
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

    ParGridFunction *S_exact = new ParGridFunction(S_space);
    S_exact->ProjectCoefficient(*Mytest.scalarS);

    ParGridFunction * sigma_exact = new ParGridFunction(R_space);
    sigma_exact->ProjectCoefficient(*Mytest.sigma);

    {
        Vector Sigmahat_truedofs(R_space->TrueVSize());
        Sigmahat->ParallelProject(Sigmahat_truedofs);

        Vector sigma_exact_truedofs((R_space->TrueVSize()));
        sigma_exact->ParallelProject(sigma_exact_truedofs);

        MFEM_ASSERT(CheckBdrError(Sigmahat_truedofs, &sigma_exact_truedofs, *EssBdrTrueDofs_Funct_lvls[0][0], true),
                                  "for the particular solution Sigmahat in the old code");
    }

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

    //////////////////////// beginning of some more crutches

    HypreParMatrix * A = (HypreParMatrix*)(&divfree_funct_op->GetBlock(0,0));
    HypreParMatrix * C;
    if (strcmp(space_for_S,"H1") == 0)
        C = (HypreParMatrix*)(&divfree_funct_op->GetBlock(1,1));

    HypreParMatrix * M = (HypreParMatrix*)(&orig_op->GetBlock(0,0));

    Vector Sigmahat_truedofs(R_space->TrueVSize());
    Sigmahat->ParallelProject(Sigmahat_truedofs);

    BlockVector trueXhat(orig_op->ColOffsets());
    trueXhat = 0.0;
    trueXhat.GetBlock(0) = Sigmahat_truedofs;

    BlockVector truetemp1(orig_op->ColOffsets());
    orig_op->Mult(trueXhat, truetemp1);
    truetemp1 -= problem->GetRhs();

    truetemp1 *= -1;

    const HypreParMatrix * Divfree_dop = hierarchy->GetDivfreeDop(0);// Divfree_hpmat_mod_lvls[0];
    //HypreParMatrix * Divfree_dop = Divfree_hpmat_mod_lvls[0];
    HypreParMatrix * DivfreeT_dop = Divfree_dop->Transpose();

    BlockVector trueRhs_divfree(divfree_funct_op->ColOffsets());
    trueRhs_divfree = 0.0;
    DivfreeT_dop->Mult(truetemp1.GetBlock(0), trueRhs_divfree.GetBlock(0));
    if (strcmp(space_for_S,"H1") == 0)
        trueRhs_divfree.GetBlock(1) = truetemp1.GetBlock(1);

    //////////////////////// end of some more crutches

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
                        prec = new MonolithicMultigrid(*divfree_funct_op,/* *MainOp,*/ P, EssBdrTrueDofs_HcurlFunct_lvls);
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

                        ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(0, precU);
                    }

                }
            }
            else // prec is AMS-like for the div-free part (block-diagonal for the system with boomerAMG for S)
            {
                if (dim == 3)
                {
                    if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
                    {
                        prec = new BlockDiagonalPreconditioner(block_trueOffsets);
                        Operator * precU = new IdentityOperator(A->Height());

                        Operator * precS;
                        precS = new IdentityOperator(C->Height());

                        ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(0, precU);
                        ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(1, precS);
                    }
                    else // no S, i.e. only an equation in div-free subspace
                    {
                        //prec = new BlockDiagonalPreconditioner(block_trueOffsets);
                        //Operator * precU = new IdentityOperator(A->Height());

                        //Operator * precU = new HypreAMS(*A, C_space);
                        //((HypreAMS*)precU)->SetSingularProblem();

                        //((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(0, precU);

                        prec = new HypreAMS(*A, C_space);
                        ((HypreAMS*)prec)->SetSingularProblem();
                        //((HypreAMS*)prec)->SetPrintLevel(2);

                        //prec = new BlockDiagonalPreconditioner(block_trueOffsets);
                        //Operator * precU = new IdentityOperator(A->Height());
                        //((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(0, precU);

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
        cout << "Linear solver: CG" << endl << flush;

    solver.SetAbsTol(sqrt(atol));
    solver.SetRelTol(sqrt(rtol));
    solver.SetMaxIter(max_num_iter);

#ifdef NEW_INTERFACE
    solver.SetOperator(*divfree_funct_op);
    if (with_prec)
        solver.SetPreconditioner(*GeneralMGprec);
#else
    solver.SetOperator(*MainOp);
    if (with_prec)
        solver.SetPreconditioner(*prec);
#endif

    solver.SetPrintLevel(1);
    trueX = 0.0;

    chrono.Clear();
    chrono.Start();
    solver.Mult(trueRhs_divfree/*trueRhs*/, trueX);
    chrono.Stop();

    MFEM_ASSERT(CheckBdrError(trueX.GetBlock(0), NULL, *EssBdrTrueDofs_Hcurl[0], true),
                              "for u_truedofs in the old code");

    for (int blk = 0; blk < numblocks; ++blk)
    {
        const Array<int> *temp;
        if (blk == 0)
            temp = EssBdrTrueDofs_Hcurl[0];
        else
            temp = EssBdrTrueDofs_Funct_lvls[0][blk];

        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            int tdof = (*temp)[tdofind];
            trueX.GetBlock(blk)[tdof] = 0.0;
        }
    }

    //MFEM_ASSERT(CheckBdrError(trueX.GetBlock(0), NULL, *EssBdrTrueDofs_Hcurl[0], true),
                              //"for u_truedofs in the old code");
    //MFEM_ASSERT(CheckBdrError(trueX.GetBlock(1), NULL, *EssBdrTrueDofs_Funct_lvls[0][1], true),
                              //"for S_truedofs from trueX in the old code");

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
    else // no S, then we compute S from sigma
    {
        // temporary for checking the computation of S below
        //sigma->ProjectCoefficient(*Mytest.sigma);

        S = new ParGridFunction(S_space);

        ParBilinearForm *Cblock(new ParBilinearForm(S_space));
        Cblock->AddDomainIntegrator(new MassIntegrator(*Mytest.bTb));
        Cblock->Assemble();
        Cblock->Finalize();
        HypreParMatrix * C = Cblock->ParallelAssemble();

        ParMixedBilinearForm *Bblock(new ParMixedBilinearForm(R_space, S_space));
        Bblock->AddDomainIntegrator(new VectorFEMassIntegrator(*Mytest.b));
        Bblock->Assemble();
        Bblock->Finalize();
        HypreParMatrix * B = Bblock->ParallelAssemble();
        Vector bTsigma(C->Height());
        Vector trueSigma(R_space->TrueVSize());
        sigma->ParallelProject(trueSigma);

        B->Mult(trueSigma,bTsigma);

        Vector trueS(C->Height());
        trueS = 0.0;

        CGSolver cg(comm);
        cg.SetPrintLevel(0);
        cg.SetMaxIter(5000);
        cg.SetRelTol(sqrt(1.0e-9));
        cg.SetAbsTol(sqrt(1.0e-12));
        cg.SetOperator(*C);
        cg.Mult(bTsigma, trueS);

        //CG(*C, bTsigma, trueS, 0, 5000, 1e-9, 1e-12);
        S->Distribute(trueS);

        delete B;
        delete C;
        delete Bblock;
        delete Cblock;
    }

    double err_sigma = sigma->ComputeL2Error(*Mytest.sigma, irs);
    double norm_sigma = ComputeGlobalLpNorm(2, *Mytest.sigma, *pmesh, irs);

    if (verbose)
        cout << "sigma_h = sigma_hat + div-free part, div-free part = curl u_h \n";

    if (verbose)
    {
        std::cout << "err_sigma = " << err_sigma << ", norm_sigma = " << norm_sigma << "\n";
        if ( norm_sigma > MYZEROTOL )
            cout << "|| sigma_h - sigma_ex || / || sigma_ex || = " << err_sigma / norm_sigma << endl;
        else
            cout << "|| sigma || = " << err_sigma << " (sigma_ex = 0)" << endl;
    }

    /*
    double err_sigmahat = Sigmahat->ComputeL2Error(*Mytest.sigma, irs);
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

    double err_div = DivSigma.ComputeL2Error(*Mytest.scalardivsigma,irs);
    double norm_div = ComputeGlobalLpNorm(2, *Mytest.scalardivsigma, *pmesh, irs);

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
        S_exact->ProjectCoefficient(*Mytest.scalarS);

        double err_S = S->ComputeL2Error(*Mytest.scalarS, irs);
        norm_S = ComputeGlobalLpNorm(2, *Mytest.scalarS, *pmesh, irs);
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

            if (dim != 3)
            {
                delete GradSpace;
                delete hcurl_coll;
            }

        }

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

            HypreParMatrix * BT = (HypreParMatrix*)(&orig_op->GetBlock(1,0));
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

        delete S_exact;
    }

    if (verbose)
        cout << "Computing projection errors \n";

    double projection_error_sigma = sigma_exact->ComputeL2Error(*Mytest.sigma, irs);

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
        double projection_error_S = S_exact->ComputeL2Error(*Mytest.scalarS, irs);

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

    BlockVector finalSol(orig_op->ColOffsets());
    finalSol = 0.0;
    sigma->ParallelProject(finalSol.GetBlock(0));
    if (strcmp(space_for_S,"H1") == 0)
        S->ParallelProject(finalSol.GetBlock(1));

    bool checkbnd = true;
    problem->ComputeError(finalSol, verbose, checkbnd);

    if (verbose)
        std::cout << "Errors in the MG code were computed via FOSLSProblem routine \n";

    //MPI_Finalize();
    //return 0;
#endif // for #ifdef OLD_CODE

    chrono.Clear();
    chrono.Start();

//#ifdef NEW_INTERFACE2
    std::vector<const Array<int> *> offsets_hdivh1(nlevels);
    offsets_hdivh1[0] = &hierarchy->ConstructTrueOffsetsforFormul(0, space_names_funct);

    std::vector<const Array<int> *> offsets_sp_hdivh1(nlevels);
    offsets_sp_hdivh1[0] = &hierarchy->ConstructOffsetsforFormul(0, space_names_funct);

    Array<int> offsets_funct_hdivh1;

    // manually truncating the original problem's operator into hdiv-h1 operator
    BlockOperator * hdivh1_op = new BlockOperator(*offsets_hdivh1[0]);

    HypreParMatrix * hdivh1_op_00 = dynamic_cast<HypreParMatrix*>(&(orig_op->GetBlock(0,0)));
    hdivh1_op->SetBlock(0,0, hdivh1_op_00);

    HypreParMatrix * hdivh1_op_01, *hdivh1_op_10, *hdivh1_op_11;
    if (strcmp(space_for_S,"H1") == 0)
    {
        hdivh1_op_01 = dynamic_cast<HypreParMatrix*>(&(orig_op->GetBlock(0,1)));
        hdivh1_op_10 = dynamic_cast<HypreParMatrix*>(&(orig_op->GetBlock(1,0)));
        hdivh1_op_11 = dynamic_cast<HypreParMatrix*>(&(orig_op->GetBlock(1,1)));

        hdivh1_op->SetBlock(0,1, hdivh1_op_01);
        hdivh1_op->SetBlock(1,0, hdivh1_op_10);
        hdivh1_op->SetBlock(1,1, hdivh1_op_11);
    }


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

    std::vector< Array<int>* > coarsebnd_indces_funct_lvls(num_levels);

    for (int l = 0; l < num_levels - 1; ++l)
    {
        std::vector<Array<int>* > &essbdr_tdofs_funct =
                hierarchy->GetEssBdrTdofsOrDofs("tdof", space_names_funct, essbdr_attribs, l + 1);

        int ncoarse_bndtdofs = 0;
        for (int blk = 0; blk < numblocks; ++blk)
        {

            ncoarse_bndtdofs += essbdr_tdofs_funct[blk]->Size();
        }

        coarsebnd_indces_funct_lvls[l] = new Array<int>(ncoarse_bndtdofs);

        int shift_bnd_indices = 0;
        int shift_tdofs_indices = 0;

        for (int blk = 0; blk < numblocks; ++blk)
        {
            for (int j = 0; j < essbdr_tdofs_funct[blk]->Size(); ++j)
                (*coarsebnd_indces_funct_lvls[l])[j + shift_bnd_indices] =
                    (*essbdr_tdofs_funct[blk])[j] + shift_tdofs_indices;

            shift_bnd_indices += essbdr_tdofs_funct[blk]->Size();
            shift_tdofs_indices += hierarchy->GetSpace(space_names_funct[blk], l + 1)->TrueVSize();
        }

    }

    std::vector<Array<int>* > dtd_row_offsets(num_levels);
    std::vector<Array<int>* > dtd_col_offsets(num_levels);

    std::vector<Array<int>* > el2dofs_row_offsets(num_levels);
    std::vector<Array<int>* > el2dofs_col_offsets(num_levels);

    Array<SparseMatrix*> Constraint_mat_lvls_mg(num_levels);
    Array<BlockMatrix*> Funct_mat_lvls_mg(num_levels);

    for (int l = 0; l < num_levels; ++l)
    {
        dtd_row_offsets[l] = new Array<int>();
        dtd_col_offsets[l] = new Array<int>();

        el2dofs_row_offsets[l] = new Array<int>();
        el2dofs_col_offsets[l] = new Array<int>();

        if (l < num_levels - 1)
        {
            offsets_hdivh1[l + 1] = &hierarchy->ConstructTrueOffsetsforFormul(l + 1, space_names_funct);
            BlockP_mg_nobnd_plus[l] = hierarchy->ConstructTruePforFormul(l, space_names_funct,
                                                                         *offsets_hdivh1[l], *offsets_hdivh1[l + 1]);
            P_mg_plus[l] = new BlkInterpolationWithBNDforTranspose(*BlockP_mg_nobnd_plus[l],
                                                              *coarsebnd_indces_funct_lvls[l],
                                                              *offsets_hdivh1[l], *offsets_hdivh1[l + 1]);
        }

        if (l == 0)
            BlockOps_mg_plus[l] = hdivh1_op;
        else
        {
            BlockOps_mg_plus[l] = new RAPBlockHypreOperator(*BlockP_mg_nobnd_plus[l - 1],
                    *BlockOps_mg_plus[l - 1], *BlockP_mg_nobnd_plus[l - 1], *offsets_hdivh1[l]);

            std::vector<Array<int>* > &essbdr_tdofs_funct = hierarchy->GetEssBdrTdofsOrDofs("tdof", space_names_funct, essbdr_attribs, l);
            EliminateBoundaryBlocks(*BlockOps_mg_plus[l], essbdr_tdofs_funct);
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

            //offsets_sp_hdivh1[l + 1] = &hierarchy->ConstructOffsetsforFormul(l + 1, space_names_funct);

            Funct_mat_lvls_mg[0] = problem->ConstructFunctBlkMat(offsets_funct_hdivh1);
        }
        else
        {
            offsets_sp_hdivh1[l] = &hierarchy->ConstructOffsetsforFormul(l, space_names_funct);

            Constraint_mat_lvls_mg[l] = RAP(*hierarchy->GetPspace(SpaceName::L2, l - 1),
                                            *Constraint_mat_lvls_mg[l - 1], *hierarchy->GetPspace(SpaceName::HDIV, l - 1));

            BlockMatrix * P_Funct = hierarchy->ConstructPforFormul(l - 1, space_names_funct,
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
            // getting smoothers from the older mg setup
            //HcurlSmoothers_lvls[l] = Smoothers_lvls[l];
            //SchwarzSmoothers_lvls[l] = (*LocalSolver_lvls)[l];

            HcurlSmoothers_lvls[l] = new HcurlGSSSmoother(*BlockOps_mg_plus[l],
                                                     *hierarchy->GetDivfreeDop(l),
                                                     hierarchy->GetEssBdrTdofsOrDofs("tdof", SpaceName::HCURL,
                                                                               essbdr_attribs_Hcurl, l),
                                                     hierarchy->GetEssBdrTdofsOrDofs("tdof",
                                                                               space_names_funct,
                                                                               essbdr_attribs, l),
                                                     &SweepsNum, *offsets_hdivh1[l]);

            int size = BlockOps_mg_plus[l]->Height();

            /// TODO:
            /// Next steps are:
            /// 5) If have time, also look into the simple parabolic example (especially on adding
            /// the parabolic test to FOSLStest setup

            bool optimized_localsolve = true;

            SparseMatrix * P_L2_T = Transpose(*hierarchy->GetPspace(SpaceName::L2, l));
            if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
            {
                SchwarzSmoothers_lvls[l] = new LocalProblemSolverWithS(size, *Funct_mat_lvls_mg[l],
                                                         *Constraint_mat_lvls_mg[l],
                                                         hierarchy->GetDofTrueDof(space_names_funct, l),
                                                         *P_L2_T,
                                                         *hierarchy->GetElementToDofs(space_names_funct, l,
                                                                                     *el2dofs_row_offsets[l],
                                                                                     *el2dofs_col_offsets[l]),
                                                         *hierarchy->GetElementToDofs(SpaceName::L2, l),
                                                         hierarchy->GetEssBdrTdofsOrDofs("dof", space_names_funct,
                                                                                  fullbdr_attribs, l),
                                                         hierarchy->GetEssBdrTdofsOrDofs("dof", space_names_funct,
                                                                                  essbdr_attribs, l),
                                                         optimized_localsolve);
            }
            else // no S
            {
                SchwarzSmoothers_lvls[l] = new LocalProblemSolver(size, *Funct_mat_lvls_mg[l],
                                                                  *Constraint_mat_lvls_mg[l],
                                                                  hierarchy->GetDofTrueDof(space_names_funct, l),
                                                                  *P_L2_T,
                                                                  *hierarchy->GetElementToDofs(space_names_funct, l,
                                                                                              *el2dofs_row_offsets[l],
                                                                                              *el2dofs_col_offsets[l]),
                                                                  *hierarchy->GetElementToDofs(SpaceName::L2, l),
                                                                  hierarchy->GetEssBdrTdofsOrDofs("dof", space_names_funct,
                                                                                           fullbdr_attribs, l),
                                                                  hierarchy->GetEssBdrTdofsOrDofs("dof", space_names_funct,
                                                                                           essbdr_attribs, l),
                                                                  optimized_localsolve);
            }

            delete P_L2_T;

#ifdef SOLVE_WITH_LOCALSOLVERS
            Smoo_mg_plus[l] = new SmootherSum(*SchwarzSmoothers_lvls[l], *HcurlSmoothers_lvls[l], *Ops_mg_plus[l]);
#else
            Smoo_mg_plus[l] = HcurlSmoothers_lvls[l];
#endif
        }
    }

    for (int l = 0; l < nlevels - 1; ++l)
        Ops_mg_special[l] = Ops_mg_plus[l];

    if (verbose)
        std::cout << "Creating the new coarsest solver which works in the div-free subspace \n" << std::flush;

    //CoarseSolver_mg_plus = CoarsestSolver;

    CoarseSolver_mg_plus = new CoarsestProblemHcurlSolver(Ops_mg_plus[num_levels - 1]->Height(),
                                                     *BlockOps_mg_plus[num_levels - 1],
                                                     *hierarchy->GetDivfreeDop(num_levels - 1),
                                                     hierarchy->GetEssBdrTdofsOrDofs("dof", space_names_funct,
                                                                                     fullbdr_attribs, num_levels - 1),
                                                     hierarchy->GetEssBdrTdofsOrDofs("tdof",
                                                                                     space_names_funct,
                                                                                     essbdr_attribs, num_levels - 1),
                                                     hierarchy->GetEssBdrTdofsOrDofs("dof", SpaceName::HCURL,
                                                                                     essbdr_attribs_Hcurl, num_levels - 1),
                                                     hierarchy->GetEssBdrTdofsOrDofs("tdof", SpaceName::HCURL,
                                                                                     essbdr_attribs_Hcurl, num_levels - 1));

    if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
    {
        ((CoarsestProblemHcurlSolver*)CoarseSolver_mg_plus)->SetMaxIter(100);
        ((CoarsestProblemHcurlSolver*)CoarseSolver_mg_plus)->SetAbsTol(sqrt(1.0e-32));
        ((CoarsestProblemHcurlSolver*)CoarseSolver_mg_plus)->SetRelTol(sqrt(1.0e-12));
        ((CoarsestProblemHcurlSolver*)CoarseSolver_mg_plus)->ResetSolverParams();
    }
    else // L2 case requires more iterations
    {
        ((CoarsestProblemHcurlSolver*)CoarseSolver_mg_plus)->SetMaxIter(100);
        ((CoarsestProblemHcurlSolver*)CoarseSolver_mg_plus)->SetAbsTol(sqrt(1.0e-32));
        ((CoarsestProblemHcurlSolver*)CoarseSolver_mg_plus)->SetRelTol(sqrt(1.0e-12));
        ((CoarsestProblemHcurlSolver*)CoarseSolver_mg_plus)->ResetSolverParams();
    }

    GeneralMultigrid * GeneralMGprec_plus =
            new GeneralMultigrid(nlevels, P_mg_plus, Ops_mg_plus, *CoarseSolver_mg_plus, Smoo_mg_plus);
//#endif

    if (verbose)
        std::cout << "\nCreating an instance of the new Hcurl smoother and the minimization solver \n";

    //ParLinearForm *fform = new ParLinearForm(R_space);

    ParLinearForm * constrfform = new ParLinearForm(W_space_lvls[0]);
    constrfform->AddDomainIntegrator(new DomainLFIntegrator(*Mytest.scalardivsigma));
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

    BlockVector Xinit(Funct_mat_lvls_mg[0]->ColOffsets());
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
        new_trueoffsets[blk + 1] = hierarchy->GetDofTrueDof(space_names_problem[blk], 0)/*Dof_TrueDof_Func_lvls[0][blk]*/->Width();
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

    Array<LocalProblemSolver*> LocalSolver_partfinder_lvls_new(num_levels - 1);
    for (int l = 0; l < num_levels - 1; ++l)
    {
        if (strcmp(space_for_S,"H1") == 0)
            LocalSolver_partfinder_lvls_new[l] = dynamic_cast<LocalProblemSolverWithS*>(SchwarzSmoothers_lvls[l]);
        else
            LocalSolver_partfinder_lvls_new[l] = dynamic_cast<LocalProblemSolver*>(SchwarzSmoothers_lvls[l]);
        MFEM_ASSERT(LocalSolver_partfinder_lvls_new[l], "*Unsuccessful cast of the Schwars smoother \n");
    }

    // Creating the coarsest problem solver
    int coarse_size = 0;
    for (int i = 0; i < space_names_problem.Size(); ++i)
        coarse_size += hierarchy->GetSpace(space_names_problem[i], num_levels - 1)->TrueVSize();

    Array<int> row_offsets_coarse, col_offsets_coarse;

    std::vector<Array<int>* > &essbdr_tdofs_funct_coarse =
            hierarchy->GetEssBdrTdofsOrDofs("tdof", space_names_funct, essbdr_attribs, num_levels - 1);

    std::vector<Array<int>* > &essbdr_dofs_funct_coarse =
            hierarchy->GetEssBdrTdofsOrDofs("dof", space_names_funct, essbdr_attribs, num_levels - 1);


    CoarsestProblemSolver* CoarsestSolver_partfinder_new =
            new CoarsestProblemSolver(coarse_size,
                                      *Funct_mat_lvls_mg[num_levels - 1],
            *Constraint_mat_lvls_mg[num_levels - 1],
            hierarchy->GetDofTrueDof(space_names_funct, num_levels - 1, row_offsets_coarse, col_offsets_coarse),
            *hierarchy->GetDofTrueDof(SpaceName::L2, num_levels - 1),
            essbdr_dofs_funct_coarse,
            essbdr_tdofs_funct_coarse);

    CoarsestSolver_partfinder_new->SetMaxIter(70000);
    CoarsestSolver_partfinder_new->SetAbsTol(1.0e-18);
    CoarsestSolver_partfinder_new->SetRelTol(1.0e-18);
    CoarsestSolver_partfinder_new->ResetSolverParams();

#ifdef NEW_INTERFACE
    //std::vector<Array<int>* > &essbdr_tdofs_funct =
            //hierarchy->GetEssBdrTdofsOrDofs("tdof", space_names, essbdr_attribs, l + 1);

    //Array< SparseMatrix*> el2dofs_R(ref_levels);
    //Array< SparseMatrix*> el2dofs_W(ref_levels);
    //Array< SparseMatrix*> P_Hdiv_lvls(ref_levels);
    Array< SparseMatrix*> P_L2_lvls(ref_levels);
    Array< SparseMatrix*> AE_e_lvls(ref_levels);

    for (int l = 0; l < ref_levels; ++l)
    {
        //el2dofs_R[l] = hierarchy->GetElementToDofs(SpaceName::HDIV, l);
        //el2dofs_W[l] = hierarchy->GetElementToDofs(SpaceName::L2, l);
        //P_Hdiv_lvls[l] = hierarchy->GetPspace(SpaceName::HDIV, l);
        P_L2_lvls[l] = hierarchy->GetPspace(SpaceName::L2, l);
        AE_e_lvls[l] = Transpose(*P_L2_lvls[l]);
    }

    std::vector< std::vector<Array<int>* > > essbdr_tdofs_funct_lvls(num_levels);
    for (int l = 0; l < num_levels; ++l)
    {
        essbdr_tdofs_funct_lvls[l] = hierarchy->GetEssBdrTdofsOrDofs
                ("tdof", space_names_funct, essbdr_attribs, l);
    }

    BlockVector * xinit_new = problem->GetTrueInitialConditionFunc();

    FunctionCoefficient * rhs_coeff = problem->GetFEformulation().GetFormulation()->GetTest()->GetRhs();
    ParLinearForm * constrfform_new = new ParLinearForm(hierarchy->GetSpace(SpaceName::L2, 0));
    constrfform_new->AddDomainIntegrator(new DomainLFIntegrator(*rhs_coeff));
    constrfform_new->Assemble();

    DivConstraintSolver PartsolFinder(comm, num_levels,
                                      AE_e_lvls,
                                      BlockP_mg_nobnd_plus,
                                      P_L2_lvls,
                                      essbdr_tdofs_funct_lvls,
                                      Ops_mg_special,
                                      (HypreParMatrix&)(problem->GetOp_nobnd()->GetBlock(numblocks_funct,0)),
                                      *constrfform_new, //Floc,
                                      HcurlSmoothers_lvls,
                                      *xinit_new,
#ifdef CHECK_CONSTR
                                      *constrfform_new,
#endif
                                      &LocalSolver_partfinder_lvls_new,
                                      CoarsestSolver_partfinder_new);

#else
    DivConstraintSolver PartsolFinder(comm, num_levels, P_WT,
                                      TrueP_Func, P_W,
                                      EssBdrTrueDofs_Funct_lvls,
                                      Ops_mg_special,
                                      //Funct_global_lvls,
                                      *Constraint_global,
                                      Floc,
                                      HcurlSmoothers_lvls,
                                      //Smoothers_lvls,
                                      Xinit_truedofs,
#ifdef CHECK_CONSTR
                                      Floc,
#endif
                                      &LocalSolver_partfinder_lvls_new,
                                      //LocalSolver_partfinder_lvls,
                                      CoarsestSolver_partfinder_new);
                                      //CoarsestSolver_partfinder);
    CoarsestSolver_partfinder->SetMaxIter(70000);
    CoarsestSolver_partfinder->SetAbsTol(1.0e-18);
    CoarsestSolver_partfinder->SetRelTol(1.0e-18);
    CoarsestSolver_partfinder->ResetSolverParams();
#endif
#endif

    GeneralMinConstrSolver NewSolver( comm, num_levels,
                                      BlockP_mg_nobnd_plus,
                                      //TrueP_Func,
                                      EssBdrTrueDofs_Funct_lvls,
                                      *Functrhs_global,
                                      HcurlSmoothers_lvls, //Smoothers_lvls,
                                      //Xinit_truedofs, Funct_global_lvls,
                                      Xinit_truedofs, Ops_mg_special,
#ifdef CHECK_CONSTR
                                     *Constraint_global, Floc,
#endif
#ifdef TIMING
                                     Times_mult, Times_solve, Times_localsolve, Times_localsolve_lvls, Times_smoother, Times_smoother_lvls, Times_coarsestproblem, Times_resupdate, Times_fw, Times_up,
#endif
#ifdef SOLVE_WITH_LOCALSOLVERS
                                      &SchwarzSmoothers_lvls, //LocalSolver_lvls,
#else
                     NULL,
#endif
                                      CoarseSolver_mg_plus, //CoarsestSolver,
                                      stopcriteria_type);

    double newsolver_reltol = 1.0e-6;

    if (verbose)
        std::cout << "newsolver_reltol = " << newsolver_reltol << "\n";

    NewSolver.SetRelTol(newsolver_reltol);
    NewSolver.SetMaxIter(200);
    NewSolver.SetPrintLevel(0);
    NewSolver.SetStopCriteriaType(0);

    BlockVector ParticSol(new_trueoffsets);
    ParticSol = 0.0;

    chrono.Stop();
    if (verbose)
        std::cout << "New solver and PartSolFinder were created in "<< chrono.RealTime() <<" seconds.\n";
    chrono.Clear();
    chrono.Start();

#ifdef WITH_DIVCONSTRAINT_SOLVER
#ifndef NEW_INTERFACE
    if (verbose)
    {
        std::cout << "CoarsestSolver parameters for the PartSolFinder: \n" << std::flush;
        CoarsestSolver_partfinder->PrintSolverParams();
    }
#endif
    PartsolFinder.Mult(Xinit_truedofs, ParticSol);

    //std::cout << "partic sol norm = " << ParticSol.Norml2() / sqrt (ParticSol.Size()) << "\n";
    //MPI_Finalize();
    //return 0;
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
    if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
    {
        CoarsestSolver_partfinder->SetMaxIter(200);
        CoarsestSolver_partfinder->SetAbsTol(1.0e-9); // -9
        CoarsestSolver_partfinder->SetRelTol(1.0e-9); // -9 for USE_AS_A_PREC
        CoarsestSolver_partfinder->ResetSolverParams();
    }
    else
    {
        CoarsestSolver_partfinder->SetMaxIter(400);
        CoarsestSolver_partfinder->SetAbsTol(1.0e-15); // -9
        CoarsestSolver_partfinder->SetRelTol(1.0e-15); // -9 for USE_AS_A_PREC
        CoarsestSolver_partfinder->ResetSolverParams();
    }
#else
#endif
    if (verbose)
    {
        std::cout << "CoarsestSolver parameters for the new solver: \n" << std::flush;
#ifndef HCURL_COARSESOLVER
        CoarsestSolver_partfinder->PrintSolverParams();
#else
#endif
    }
    if (verbose)
        std::cout << "Particular solution was found in " << chrono.RealTime() <<" seconds.\n";
    chrono.Clear();
    chrono.Start();

    // checking that the computed particular solution satisfies essential boundary conditions
    for ( int blk = 0; blk < numblocks_funct; ++blk)
    {
        MFEM_ASSERT(CheckBdrError(ParticSol.GetBlock(blk), &(Xinit_truedofs.GetBlock(blk)), *EssBdrTrueDofs_Funct_lvls[0][blk], true),
                                  "for the particular solution");
    }

    // checking that the boundary conditions are not violated for the initial guess
    for ( int blk = 0; blk < numblocks_funct; ++blk)
    {
        for (int i = 0; i < EssBdrTrueDofs_Funct_lvls[0][blk]->Size(); ++i)
        {
            int tdofind = (*EssBdrTrueDofs_Funct_lvls[0][blk])[i];
            if ( fabs(ParticSol.GetBlock(blk)[tdofind]) > 1.0e-14 )
            {
                std::cout << "blk = " << blk << ": bnd cnd is violated for the ParticSol! \n";
                std::cout << "tdofind = " << tdofind << ", value = " << ParticSol.GetBlock(blk)[tdofind] << "\n";
            }
        }
    }

    // checking that the particular solution satisfies the divergence constraint
    BlockVector temp_dofs(Funct_mat_lvls_mg[0]->RowOffsets());
    for ( int blk = 0; blk < numblocks_funct; ++blk)
    {
        hierarchy->GetDofTrueDof(space_names_problem[blk], 0)/*Dof_TrueDof_Func_lvls[0][blk]*/->Mult(ParticSol.GetBlock(blk), temp_dofs.GetBlock(blk));
    }

    Vector temp_constr(Constraint_mat_lvls_mg[0]->Height());
    Constraint_mat_lvls_mg[0]->Mult(temp_dofs.GetBlock(0), temp_constr);
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
        MFEM_ASSERT(CheckBdrError(ParticSol.GetBlock(blk), &(Xinit_truedofs.GetBlock(blk)), *EssBdrTrueDofs_Funct_lvls[0][blk], true),
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
        qformtest->AddDomainIntegrator(new GradDomainLFIntegrator(*Mytest.bf));
        qformtest->Assemble();
    }

    ParBilinearForm *Ablocktest(new ParBilinearForm(R_space_lvls[0]));
    HypreParMatrix *Atest;
    if (strcmp(space_for_S,"H1") == 0)
        Ablocktest->AddDomainIntegrator(new VectorFEMassIntegrator);
    else
        Ablocktest->AddDomainIntegrator(new VectorFEMassIntegrator(*Mytest.Ktilda));
#ifdef WITH_PENALTY
    Ablocktest->AddDomainIntegrator(new VectorFEMassIntegrator(reg_coeff));
#endif
    Ablocktest->Assemble();
    Ablocktest->EliminateEssentialBC(ess_bdrSigma, *sigma_exact_finest, *fformtest);
    Ablocktest->Finalize();
    Atest = Ablocktest->ParallelAssemble();

    delete Ablocktest;

    HypreParMatrix *Ctest;
    if (strcmp(space_for_S,"H1") == 0)
    {
        ParBilinearForm * Cblocktest = new ParBilinearForm(H_space_lvls[0]);
        if (strcmp(space_for_S,"H1") == 0)
        {
            Cblocktest->AddDomainIntegrator(new MassIntegrator(*Mytest.bTb));
            Cblocktest->AddDomainIntegrator(new DiffusionIntegrator(*Mytest.bbT));
        }
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
        ParMixedBilinearForm *Bblocktest = new ParMixedBilinearForm(R_space_lvls[0], H_space_lvls[0]);
        Bblocktest->AddDomainIntegrator(new VectorFEMassIntegrator(*Mytest.minb));
        Bblocktest->Assemble();
        Bblocktest->EliminateTrialDofs(ess_bdrSigma, *sigma_exact_finest, *qformtest);
        Bblocktest->EliminateTestDofs(ess_bdrS);
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
        BlockMattest->SetBlock(0,1, BTtest);
        BlockMattest->SetBlock(1,0, Btest);
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

    Testsolver.SetOperator(*hdivh1_op);
#ifdef NEW_INTERFACE2
    Testsolver.SetPreconditioner(*GeneralMGprec_plus);
#else
    Testsolver.SetPreconditioner(NewSolver);
#endif // for ifdef NEW_INTERFACE2

    Testsolver.SetPrintLevel(1);

    trueXtest = 0.0;

    BlockVector trueRhstest_funct(blocktest_offsets);
    trueRhstest_funct = trueRhstest;

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

        if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
            std::cout << "System size: " << Atest->M() + Ctest->M() << "\n" << std::flush;
        else
            std::cout << "System size: " << Atest->M() << "\n" << std::flush;
    }

    chrono.Clear();

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

        double norm_sigma = ComputeGlobalLpNorm(2, *Mytest.sigma, *pmesh, irs);
        double err_newsigmahat = NewSigmahat->ComputeL2Error(*Mytest.sigma, irs);
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

        double err_div = DivSigma.ComputeL2Error(*Mytest.scalardivsigma,irs);
        double norm_div = ComputeGlobalLpNorm(2, *Mytest.scalardivsigma, *pmesh, irs);

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

            double err_S = NewS->ComputeL2Error((*Mytest.scalarS), irs);
            double norm_S = ComputeGlobalLpNorm(2, (*Mytest.scalarS), *pmesh, irs);
            if (verbose)
            {
                std::cout << "|| S_h - S_ex || / || S_ex || = " <<
                             err_S / norm_S << "\n";
            }
        }
        /////////////////////////////////////////////////////////

        double localFunctional = -2.0 * (trueXtest * trueRhstest_funct); //0.0;//-2.0*(trueX.GetBlock(0)*trueRhs.GetBlock(0));
        BlockMattest->Mult(trueXtest, trueRhstest_funct);
        localFunctional += trueXtest * trueRhstest_funct;

        double globalFunctional;
        MPI_Reduce(&localFunctional, &globalFunctional, 1,
                   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (verbose)
        {
            //cout << "|| sigma_h - L(S_h) ||^2 + || div_h (bS_h) - f ||^2 = " << globalFunctional+err_div*err_div << "\n";
            //cout << "|| f ||^2 = " << norm_div*norm_div  << "\n";
            //cout << "Relative Energy Error = " << sqrt(globalFunctional+err_div*err_div)/norm_div << "\n";

            if (strcmp(space_for_S,"H1") == 0) // S is present
            {
                cout << "|| sigma_h - L(S_h) ||^2 + || div_h (bS_h) - f ||^2 = " << globalFunctional+err_div*err_div << "\n";
                cout << "|| f ||^2 = " << norm_div*norm_div  << "\n";
                cout << "Relative Energy Error = " << sqrt(globalFunctional+err_div*err_div)/norm_div << "\n";
            }
            else // if S is from L2
            {
                cout << "|| sigma_h - L(S_h) ||^2 + || div_h (sigma_h) - f ||^2 = " << globalFunctional+err_div*err_div << "\n";
                cout << "Energy Error = " << sqrt(globalFunctional+err_div*err_div) << "\n";
            }
        }
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

        double norm_sigma = ComputeGlobalLpNorm(2, *Mytest.sigma, *pmesh, irs);
        double err_newsigmahat = NewSigmahat->ComputeL2Error(*Mytest.sigma, irs);
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

        double err_div = DivSigma.ComputeL2Error(*Mytest.scalardivsigma,irs);
        double norm_div = ComputeGlobalLpNorm(2, *Mytest.scalardivsigma, *pmesh, irs);

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

        double err_S = NewS->ComputeL2Error((*Mytest.scalarS), irs);
        double norm_S = ComputeGlobalLpNorm(2, (*Mytest.scalarS), *pmesh, irs);
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

    chrono_total.Stop();
    if (verbose)
        std::cout << "Total time consumed was " << chrono_total.RealTime() <<" seconds.\n";
    MPI_Finalize();
    return 0;
}

