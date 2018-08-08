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

#define WITH_DIVCONSTRAINT_SOLVER

// switches on/off usage of smoother in the new minimization solver
// in parallel GS smoother works a little bit different from serial
#define WITH_SMOOTHERS

// activates using the new interface to local problem solvers
// via a separated class called LocalProblemSolver
#define SOLVE_WITH_LOCALSOLVERS

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
    int par_ref_levels  = 2;

    const char *space_for_S = "L2";    // "H1" or "L2"

    /*
    // Hdiv-H1 case
    using FormulType = CFOSLSFormulation_HdivH1Hyper;
    using FEFormulType = CFOSLSFEFormulation_HdivH1Hyper;
    using BdrCondsType = BdrConditions_CFOSLS_HdivH1_Hyper;
    using ProblemType = FOSLSProblem_HdivH1L2hyp;
    using DivfreeFormulType = CFOSLSFormulation_HdivH1DivfreeHyp;
    using DivfreeFEFormulType = CFOSLSFEFormulation_HdivH1DivfreeHyper;
    */

    // Hdiv-L2 case
    using FormulType = CFOSLSFormulation_HdivL2Hyper;
    using FEFormulType = CFOSLSFEFormulation_HdivL2Hyper;
    using BdrCondsType = BdrConditions_CFOSLS_HdivL2_Hyper;
    using ProblemType = FOSLSProblem_HdivL2hyp;
    using DivfreeFormulType = CFOSLSFormulation_HdivDivfreeHyp;
    using DivfreeFEFormulType = CFOSLSFEFormulation_HdivDivfreeHyp;

    bool aniso_refine = false;
    bool refine_t_first = false;

    bool with_multilevel = true;

    bool useM_in_divpart = true;

    // solver options
    int prec_option = 1;        // defines whether to use preconditioner(0) or not(!0)

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

    std::cout << std::flush;
    MPI_Barrier(MPI_COMM_WORLD);

    MFEM_ASSERT(strcmp(space_for_S,"H1") == 0 || strcmp(space_for_S,"L2") == 0, "Space for S must be H1 or L2!\n");

    if (verbose)
    {
        if (strcmp(space_for_S,"H1") == 0)
            std::cout << "Space for S: H1 \n";
        else
            std::cout << "Space for S: L2 \n";

        if (strcmp(space_for_S,"L2") == 0)
            std::cout << "S is eliminated from the system \n";
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

    if (verbose)
        cout << "Number of mpi processes: " << num_procs << endl << flush;

    bool with_prec = (prec_option != 0);

    StopWatch chrono;
    StopWatch chrono_total;

    chrono_total.Clear();
    chrono_total.Start();

    //DEFAULTED LINEAR SOLVER OPTIONS
    int max_num_iter = 2000;
    double rtol = 1e-12;//1e-7;//1e-9;
    double atol = 1e-14;//1e-9;//1e-12;

    // constructing the coarse mesh(pmesh) for the problem
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

    //////////////////////////////////

    int dim = nDimensions;

    int ref_levels = par_ref_levels;

    int num_levels = ref_levels + 1;

    chrono.Clear();
    chrono.Start();

    int numblocks_funct = 1;
    if (strcmp(space_for_S,"H1") == 0) // S is present
        numblocks_funct++;

    FormulType * formulat = new FormulType (dim, numsol, verbose);
    FEFormulType* fe_formulat = new FEFormulType(*formulat, feorder);
    BdrConditions * bdr_conds = new BdrCondsType(*pmesh);

    // constructing the general hierarchy on top of the coarse mesh,
    // refining it so that in the en pmesh is the finest mesh
    int nlevels = ref_levels + 1;
    GeneralHierarchy * hierarchy = new GeneralHierarchy(nlevels, *pmesh, 0, verbose);
    hierarchy->ConstructDivfreeDops();
    hierarchy->ConstructDofTrueDofs();

    FOSLSProblem* problem = hierarchy->BuildDynamicProblem<ProblemType>
            (*bdr_conds, *fe_formulat, prec_option, verbose);
    hierarchy->AttachProblem(problem);

    ComponentsDescriptor * descriptor;
    {
        bool with_Schwarz = true;
        bool optimized_Schwarz = true;
        bool with_Hcurl = true;
        bool with_coarsest_partfinder = true;
        bool with_coarsest_hcurl = true;
        bool with_monolithic_GS = false;
        bool with_nobnd_op = true;
        descriptor = new ComponentsDescriptor(with_Schwarz, optimized_Schwarz,
                                              with_Hcurl, with_coarsest_partfinder,
                                              with_coarsest_hcurl, with_monolithic_GS,
                                              with_nobnd_op);
    }
    MultigridToolsHierarchy * mgtools_hierarchy =
            new MultigridToolsHierarchy(*hierarchy, 0, *descriptor);


    // defining the original problem on the finest mesh
    //FOSLSProblem * problem = new ProblemType(*pmesh, *bdr_conds,
                                             //*fe_formulat, prec_option, verbose);

    BlockVector * Xinit_truedofs = problem->GetTrueInitialConditionFunc();

    // defining space names for the original problem, for the functional and the related divfree problem
    const Array<SpaceName>* space_names_problem = problem->GetFEformulation().
            GetFormulation()->GetSpacesDescriptor();

    //const Array<SpaceName>* space_names_funct = problem->GetFEformulation().
                //GetFormulation()->GetFunctSpacesDescriptor();
    Array<SpaceName> space_names_funct(numblocks_funct);
    space_names_funct[0] = SpaceName::HDIV;
    if (strcmp(space_for_S,"H1") == 0)
        space_names_funct[1] = SpaceName::H1;

    Array<SpaceName> space_names_divfree(numblocks_funct);
    space_names_divfree[0] = SpaceName::HCURL;
    if (strcmp(space_for_S,"H1") == 0)
        space_names_divfree[1] = SpaceName::H1;

    // extracting the boundary attributes
    const Array<int> &essbdr_attribs_Hcurl = problem->GetBdrConditions().GetBdrAttribs(0);
    std::vector<Array<int>*>& essbdr_attribs = problem->GetBdrConditions().GetAllBdrAttribs();

    std::vector<Array<int>*> fullbdr_attribs(numblocks_funct);
    for (unsigned int i = 0; i < fullbdr_attribs.size(); ++i)
    {
        fullbdr_attribs[i] = new Array<int>(pmesh->bdr_attributes.Max());
        (*fullbdr_attribs[i]) = 1;
    }

    if (verbose)
    {
        std::cout << "Boundary conditions: \n";
        for (unsigned int i = 0; i < essbdr_attribs.size(); ++i)
        {
            std::cout << "component " << i << ": \n";
            essbdr_attribs[i]->Print(std::cout, pmesh->bdr_attributes.Max());
        }
    }

    // creating different block offsets (all on truedofs)
    Array<int>& offsets_problem = problem->GetTrueOffsets();
    Array<int>& offsets_func = problem->GetTrueOffsetsFunc();
    std::vector<const Array<int> *> divfree_offsets(nlevels);
    divfree_offsets[0] = hierarchy->ConstructTrueOffsetsforFormul(0, space_names_divfree);

    HypreParMatrix * Constraint_global = (HypreParMatrix*)
            (&problem->GetOp_nobnd()->GetBlock(numblocks_funct, 0));

    pmesh->PrintInfo(std::cout); if(verbose) cout << "\n";

    /// finding a particular solution via the old code: either multilevel or simple solving a Poisson problem

    chrono.Clear();
    chrono.Start();

    Vector Sigmahat_truedofs(problem->GetPfes(0)->TrueVSize());

    if (with_multilevel)
    {
        if (verbose)
            std::cout << "Using an old implementation of the multilevel algorithm to find"
                         " a particular solution \n";

        ConstantCoefficient k(1.0);

        SparseMatrix *M_local;
        if (useM_in_divpart)
        {
            ParBilinearForm Massform(hierarchy->GetSpace(SpaceName::HDIV, 0));
            Massform.AddDomainIntegrator(new VectorFEMassIntegrator(k));
            Massform.Assemble();
            Massform.Finalize();
            M_local = Massform.LoseMat();
        }
        else
            M_local = NULL;

        ParMixedBilinearForm DivForm(hierarchy->GetSpace(SpaceName::HDIV, 0),
                                     hierarchy->GetSpace(SpaceName::L2, 0));
        DivForm.AddDomainIntegrator(new VectorFEDivergenceIntegrator);
        DivForm.Assemble();
        DivForm.Finalize();
        SparseMatrix *B_local = DivForm.LoseMat();

        //Right hand size
        ParLinearForm gform(hierarchy->GetSpace(SpaceName::L2, 0));
        gform.AddDomainIntegrator(new DomainLFIntegrator(*problem->GetFEformulation().
                                                          GetFormulation()->GetTest()->GetRhs()));
        gform.Assemble();

        Vector G_fine(hierarchy->GetSpace(SpaceName::HDIV, 0)->GetVSize());

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

        const Array<int>* coarse_essbdr_dofs_Hdiv = hierarchy->GetEssBdrTdofsOrDofs
                ("dof", SpaceName::HDIV, *essbdr_attribs[0], num_levels - 1);

        DivPart divp;

        ParGridFunction sigmahat(problem->GetPfes(0));

        divp.div_part(ref_levels,
                      M_local, B_local,
                      G_fine,
                      gform,
                      P_L2_lvls, P_Hdiv_lvls, e_AE_lvls,
                      el2dofs_R,
                      el2dofs_W,
                      hierarchy->GetDofTrueDof(SpaceName::HDIV, num_levels - 1),
                      hierarchy->GetDofTrueDof(SpaceName::L2, num_levels - 1),
                      hierarchy->GetSpace(SpaceName::HDIV, num_levels - 1)->GetDofOffsets(),
                      hierarchy->GetSpace(SpaceName::L2, num_levels - 1)->GetDofOffsets(),
                      sigmahat,
                      *coarse_essbdr_dofs_Hdiv);

        sigmahat.ParallelProject(Sigmahat_truedofs);

        for (int l = 0; l < ref_levels; ++l)
        {
            delete el2dofs_R[l];
            delete el2dofs_W[l];
        }

        delete coarse_essbdr_dofs_Hdiv;

        delete B_local;
    }
    else
    {
        if (verbose)
            std::cout << "Solving Poisson problem for finding a particular solution \n";
        ParGridFunction *sigma_exact;
        HypreParMatrix *Bdiv;
        HypreParMatrix *BdivT;
        HypreParMatrix *BBT;

        sigma_exact = new ParGridFunction(hierarchy->GetSpace(SpaceName::HDIV, 0));
        sigma_exact->ProjectCoefficient(*problem->GetFEformulation().
                                        GetFormulation()->GetTest()->GetSigma());

        ParLinearForm gform(hierarchy->GetSpace(SpaceName::L2, 0));
        gform.AddDomainIntegrator(new DomainLFIntegrator(*problem->GetFEformulation().
                                                          GetFormulation()->GetTest()->GetRhs()));
        gform.Assemble();

        ParMixedBilinearForm Bblock(hierarchy->GetSpace(SpaceName::HDIV, 0),
                                    hierarchy->GetSpace(SpaceName::L2, 0));
        Bblock.AddDomainIntegrator(new VectorFEDivergenceIntegrator);
        Bblock.Assemble();
        Bblock.EliminateTrialDofs(*essbdr_attribs[0], *sigma_exact, gform);
        Bblock.Finalize();
        Bdiv = Bblock.ParallelAssemble();

        BdivT = Bdiv->Transpose();
        BBT = ParMult(Bdiv, BdivT);

        Vector * Rhs = gform.ParallelAssemble();

        HypreBoomerAMG * invBBT = new HypreBoomerAMG(*BBT);
        invBBT->SetPrintLevel(0);

        mfem::CGSolver solver(comm);
        solver.SetPrintLevel(0);
        solver.SetMaxIter(70000);
        solver.SetRelTol(1.0e-12);
        solver.SetAbsTol(1.0e-14);
        solver.SetPreconditioner(*invBBT);
        solver.SetOperator(*BBT);

        Vector tempsol(hierarchy->GetSpace(SpaceName::L2, 0)->TrueVSize());
        solver.Mult(*Rhs, tempsol);

        BdivT->Mult(tempsol, Sigmahat_truedofs);

        delete sigma_exact;
        delete invBBT;
        delete BBT;
        delete Rhs;
        delete Bdiv;
        delete BdivT;
    }
    // in either way now Sigmahat_truedofs correspond to a function from H(div) s.t. div Sigmahat = div sigma = f

    chrono.Stop();
    if (verbose)
        cout << "Particular solution found in " << chrono.RealTime() << " seconds.\n";

    if (verbose)
        std::cout << "Checking that particular solution in parallel version "
                     "satisfies the divergence constraint \n";

    if (!CheckConstrRes(Sigmahat_truedofs, *Constraint_global,
                        &problem->GetRhs().GetBlock(numblocks_funct),
                        "in the old code for the particular solution"))
        std::cout << "Failure! \n";
    else
        if (verbose)
            std::cout << "Success \n";

    // creating the block operator for the divergence-free problem

    BlockOperator * orig_op = problem->GetOp();
    const HypreParMatrix * divfree_dop = hierarchy->GetDivfreeDop(0);

    HypreParMatrix * divfree_dop_mod = CopyHypreParMatrix(*divfree_dop);

    const Array<int> * hcurl_essbdr_tdofs = hierarchy->GetEssBdrTdofsOrDofs
                                        ("tdof", SpaceName::HCURL, essbdr_attribs_Hcurl, 0);

    const Array<int> * hdiv_essbdr_tdofs = hierarchy->GetEssBdrTdofsOrDofs
                                        ("tdof", SpaceName::HDIV, *essbdr_attribs[0], 0);

    Eliminate_ib_block(*divfree_dop_mod,
                       *hcurl_essbdr_tdofs,
                       *hdiv_essbdr_tdofs);

    delete hcurl_essbdr_tdofs;
    delete hdiv_essbdr_tdofs;

    // transferring the first block of the functional oiperator from hdiv into hcurl
    BlockOperator * divfree_funct_op = new BlockOperator(*divfree_offsets[0]);

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

    if (verbose)
        std::cout << "End of setting up the problem in the divergence-free formulation \n";

    chrono.Clear();
    chrono.Start();

    // setting multigrid components

    // constructing coarsest level essential boundary dofs indices in terms of
    // the global vector (for all components at once)
    std::vector< Array<int>* > coarsebnd_indces_divfree_lvls(num_levels);
    for (int l = 0; l < num_levels - 1; ++l)
    {
        std::vector<Array<int>* > essbdr_tdofs_hcurlfunct =
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
            shift_tdofs_indices += hierarchy->GetSpace
                    (space_names_divfree[blk], l + 1)->TrueVSize();
        }

        for (unsigned int i = 0; i < essbdr_tdofs_hcurlfunct.size(); ++i)
            delete essbdr_tdofs_hcurlfunct[i];
    }

    Array<BlockOperator*> BlockP_mg_nobnd(nlevels - 1);
    Array<Operator*> P_mg(nlevels - 1);
    Array<BlockOperator*> BlockOps_mg(nlevels);
    Array<Operator*> Ops_mg(nlevels);
    Array<Operator*> Smoo_mg(nlevels - 1);
    Operator* CoarseSolver_mg;

    for (int l = 0; l < num_levels; ++l)
    {
        if (l < num_levels - 1)
        {
            divfree_offsets[l + 1] = hierarchy->ConstructTrueOffsetsforFormul(l + 1, space_names_divfree);
            // FIXME: Memory leak here because of ConstructTruePforFormul
            P_mg[l] = new BlkInterpolationWithBNDforTranspose(
                        *hierarchy->ConstructTruePforFormul(l, space_names_divfree,
                                                            *divfree_offsets[l], *divfree_offsets[l + 1]),
                        *coarsebnd_indces_divfree_lvls[l],
                        *divfree_offsets[l], *divfree_offsets[l + 1]);
            // FIXME: Memory leak here because of ConstructTruePforFormul
            BlockP_mg_nobnd[l] = hierarchy->ConstructTruePforFormul(l, space_names_divfree,
                                                                    *divfree_offsets[l],
                                                                    *divfree_offsets[l + 1]);
        }

        if (l == 0)
            BlockOps_mg[l] = divfree_funct_op;
        else
        {
            BlockOps_mg[l] = new RAPBlockHypreOperator(*BlockP_mg_nobnd[l - 1],
                    *BlockOps_mg[l - 1], *BlockP_mg_nobnd[l - 1], *divfree_offsets[l]);

            std::vector<Array<int>* > essbdr_tdofs_hcurlfunct =
                    hierarchy->GetEssBdrTdofsOrDofs("tdof", space_names_divfree, essbdr_attribs, l);
            EliminateBoundaryBlocks(*BlockOps_mg[l], essbdr_tdofs_hcurlfunct);

            for (unsigned int i = 0; i < essbdr_tdofs_hcurlfunct.size(); ++i)
                delete essbdr_tdofs_hcurlfunct[i];
        }

        Ops_mg[l] = BlockOps_mg[l];

        if (l < num_levels - 1)
            Smoo_mg[l] = new MonolithicGSBlockSmoother( *BlockOps_mg[l], *divfree_offsets[l],
                                                        false, HypreSmoother::Type::l1GS, 1);
    }

    // setting the coarsest level problem solver for the multigrid in divergence-free formulation
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
    CoarsePrec_mg->owns_blocks = true;

    ((CGSolver*)CoarseSolver_mg)->SetPreconditioner(*CoarsePrec_mg);

    // newer interface, using MultigridToolsHierarchy
    DivfreeFormulType * formulat_divfree = new DivfreeFormulType (dim, numsol, verbose);
    DivfreeFEFormulType * fe_formulat_divfree = new DivfreeFEFormulType(*formulat_divfree, feorder);

    FOSLSDivfreeProblem* divfree_problem = hierarchy->BuildDynamicProblem<FOSLSDivfreeProblem>
            (*bdr_conds, *fe_formulat_divfree, prec_option, verbose);
    divfree_problem->ConstructDivfreeHpMats();
    divfree_problem->CreateOffsetsRhsSol();
    BlockOperator * divfree_problem_op = ConstructDivfreeProblemOp(*divfree_problem, *problem);
    divfree_problem->ResetOp(*divfree_problem_op);

    divfree_problem->InitSolver(verbose);
    // creating a preconditioner for the divfree problem
    divfree_problem->CreatePrec(*divfree_problem->GetOp(), prec_option, verbose);
    divfree_problem->ChangeSolver();
    divfree_problem->UpdateSolverPrec();

    hierarchy->AttachProblem(divfree_problem);
    ComponentsDescriptor * divfree_descriptor;
    {
        bool with_Schwarz = false;
        bool optimized_Schwarz = false;
        bool with_Hcurl = false;
        bool with_coarsest_partfinder = false;
        bool with_coarsest_hcurl = false;
        bool with_monolithic_GS = true;
        bool with_nobnd_op = false;
        divfree_descriptor = new ComponentsDescriptor(with_Schwarz, optimized_Schwarz,
                                                      with_Hcurl, with_coarsest_partfinder,
                                                      with_coarsest_hcurl, with_monolithic_GS,
                                                      with_nobnd_op);
    }

    MultigridToolsHierarchy * mgtools_divfree_hierarchy =
            new MultigridToolsHierarchy(*hierarchy, 1, *divfree_descriptor);

    Array<Operator*> casted_monolitGSSmoothers(nlevels - 1);
    for (int l = 0; l < nlevels - 1; ++l)
        casted_monolitGSSmoothers[l] = mgtools_divfree_hierarchy->GetMonolitGSSmoothers()[l];

    GeneralMultigrid * GeneralMGprec =
            new GeneralMultigrid(nlevels,
                                 //P_mg,
                                 mgtools_divfree_hierarchy->GetPs_bnd(),
                                 //Ops_mg,
                                 mgtools_divfree_hierarchy->GetOps(),
                                 *CoarseSolver_mg,
                                 //*mgtools_divfree_hierarchy->GetCoarsestSolver_Hcurl(),
                                 //Smoo_mg);
                                 casted_monolitGSSmoothers);

    // old interface
    //GeneralMultigrid * GeneralMGprec =
            //new GeneralMultigrid(nlevels, P_mg, Ops_mg, *CoarseSolver_mg, Smoo_mg);

    chrono.Stop();
    if (verbose)
        std::cout << "A multigrid preconditioner was created in " << chrono.RealTime() << " seconds.\n";

    chrono.Clear();
    chrono.Start();

    //////////////////////// computing right hand side for the problem in divergence-free setting

    BlockVector trueXhat(orig_op->ColOffsets());
    trueXhat = 0.0;
    trueXhat.GetBlock(0) = Sigmahat_truedofs;

    BlockVector truetemp1(orig_op->ColOffsets());
    orig_op->Mult(trueXhat, truetemp1);
    truetemp1 -= problem->GetRhs();

    truetemp1 *= -1;

    HypreParMatrix * DivfreeT_dop = divfree_dop_mod->Transpose();

    BlockVector trueRhs_divfree(divfree_funct_op->ColOffsets());
    trueRhs_divfree = 0.0;
    DivfreeT_dop->Mult(truetemp1.GetBlock(0), trueRhs_divfree.GetBlock(0));

    delete DivfreeT_dop;

    if (strcmp(space_for_S,"H1") == 0)
        trueRhs_divfree.GetBlock(1) = truetemp1.GetBlock(1);

    ////////////////////////

    chrono.Stop();
    if (verbose)
        std::cout << "Discrete divfree problem is ready \n";

    CGSolver solver(comm);
    if (verbose)
        std::cout << "Linear solver: CG \n";

    solver.SetAbsTol(sqrt(atol));
    solver.SetRelTol(sqrt(rtol));
    solver.SetMaxIter(max_num_iter);
    solver.SetOperator(*divfree_funct_op);
    if (with_prec)
        solver.SetPreconditioner(*GeneralMGprec);
    solver.SetPrintLevel(1);

    BlockVector trueX(*divfree_offsets[0]);
    trueX = 0.0;

    chrono.Clear();
    chrono.Start();
    solver.Mult(trueRhs_divfree, trueX);
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

    BlockVector finalSol(orig_op->ColOffsets());
    finalSol = 0.0;
    divfree_dop_mod->Mult(trueX.GetBlock(0), finalSol.GetBlock(0));
    if (strcmp(space_for_S,"H1") == 0)
        finalSol.GetBlock(1) = trueX.GetBlock(1);
    finalSol += trueXhat;

    bool checkbnd = true;
    problem->ComputeError(finalSol, verbose, checkbnd);

    if (verbose)
        std::cout << "Errors in the MG code were computed via FOSLSProblem routine \n";

    chrono.Clear();
    chrono.Start();

    std::vector<const Array<int> *> offsets_hdivh1(nlevels);
    offsets_hdivh1[0] = hierarchy->ConstructTrueOffsetsforFormul(0, space_names_funct);

    std::vector<const Array<int> *> offsets_sp_hdivh1(nlevels);
    offsets_sp_hdivh1[0] = hierarchy->ConstructOffsetsforFormul(0, space_names_funct);

    //Array<int> offsets_funct_hdivh1;

    // manually truncating the original problem's operator into the operator
    // which correspond to the functional (dropping the constraint)
    BlockOperator * funct_op = new BlockOperator(*offsets_hdivh1[0]);

    HypreParMatrix * funct_op_00 = dynamic_cast<HypreParMatrix*>(&(orig_op->GetBlock(0,0)));
    funct_op->SetBlock(0,0, funct_op_00);

    HypreParMatrix * funct_op_01, *funct_op_10, *funct_op_11;
    if (strcmp(space_for_S,"H1") == 0)
    {
        funct_op_01 = dynamic_cast<HypreParMatrix*>(&(orig_op->GetBlock(0,1)));
        funct_op_10 = dynamic_cast<HypreParMatrix*>(&(orig_op->GetBlock(1,0)));
        funct_op_11 = dynamic_cast<HypreParMatrix*>(&(orig_op->GetBlock(1,1)));

        funct_op->SetBlock(0,1, funct_op_01);
        funct_op->SetBlock(1,0, funct_op_10);
        funct_op->SetBlock(1,1, funct_op_11);
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

    std::vector< Array<int>* > coarsebnd_indces_funct_lvls(num_levels - 1); // num_lvls or num_lvls - 1 ?

    for (int l = 0; l < num_levels - 1; ++l)
    {
        std::vector<Array<int>* > essbdr_tdofs_funct =
                hierarchy->GetEssBdrTdofsOrDofs("tdof", space_names_funct, essbdr_attribs, l + 1);

        int ncoarse_bndtdofs = 0;
        for (int blk = 0; blk < numblocks_funct; ++blk)
        {

            ncoarse_bndtdofs += essbdr_tdofs_funct[blk]->Size();
        }

        coarsebnd_indces_funct_lvls[l] = new Array<int>(ncoarse_bndtdofs);

        int shift_bnd_indices = 0;
        int shift_tdofs_indices = 0;

        for (int blk = 0; blk < numblocks_funct; ++blk)
        {
            for (int j = 0; j < essbdr_tdofs_funct[blk]->Size(); ++j)
                (*coarsebnd_indces_funct_lvls[l])[j + shift_bnd_indices] =
                    (*essbdr_tdofs_funct[blk])[j] + shift_tdofs_indices;

            shift_bnd_indices += essbdr_tdofs_funct[blk]->Size();
            shift_tdofs_indices += hierarchy->GetSpace(space_names_funct[blk], l + 1)->TrueVSize();
        }

        for (unsigned int i = 0; i < essbdr_tdofs_funct.size(); ++i)
            delete essbdr_tdofs_funct[i];
    }

    std::vector<Array<int>* > dtd_row_offsets(num_levels);
    std::vector<Array<int>* > dtd_col_offsets(num_levels);

    std::vector<Array<int>* > el2dofs_row_offsets(num_levels - 1);
    std::vector<Array<int>* > el2dofs_col_offsets(num_levels - 1);

    Array<SparseMatrix*> Constraint_mat_lvls_mg(num_levels);
    Array<BlockMatrix*> Funct_mat_lvls_mg(num_levels);
    Array<SparseMatrix*> AE_e_lvls(num_levels - 1);

    /*
    // Deallocating memory
    delete hierarchy;
    delete problem;

    delete descriptor;
    delete mgtools_hierarchy;

    delete mgtools_divfree_hierarchy;

    for (unsigned int i = 0; i < fullbdr_attribs.size(); ++i)
        delete fullbdr_attribs[i];

    for (unsigned int i = 0; i < coarsebnd_indces_funct_lvls.size(); ++i)
        delete coarsebnd_indces_funct_lvls[i];

    for (int i = 0; i < P_mg.Size(); ++i)
        delete P_mg[i];

    for (int i = 0; i < BlockOps_mg.Size(); ++i)
        delete BlockOps_mg[i];

    for (int i = 0; i < Smoo_mg.Size(); ++i)
        delete Smoo_mg[i];

    delete CoarsePrec_mg;

#ifdef WITH_DIVCONSTRAINT_SOLVER
    delete CoarsestSolver_partfinder_new;
#endif

    for (unsigned int i = 0; i < offsets_hdivh1.size(); ++i)
        delete offsets_hdivh1[i];

    for (unsigned int i = 0; i < offsets_sp_hdivh1.size(); ++i)
        delete offsets_sp_hdivh1[i];

    for (unsigned int i = 0; i < divfree_offsets.size(); ++i)
        delete divfree_offsets[i];

    delete bdr_conds;
    delete formulat;
    delete fe_formulat;

    delete formulat_divfree;
    delete fe_formulat_divfree;

    MPI_Finalize();
    return 0;
    */

    for (int l = 0; l < num_levels; ++l)
    {
        dtd_row_offsets[l] = new Array<int>();
        dtd_col_offsets[l] = new Array<int>();

        if (l < num_levels - 1)
        {
            offsets_hdivh1[l + 1] = hierarchy->ConstructTrueOffsetsforFormul(l + 1, space_names_funct);
            //offsets_hdivh1[l]->Print();
            //offsets_hdivh1[l + 1]->Print();
            // FIXME: Memory leak for BlockP_mg_nobnd_plus
            BlockP_mg_nobnd_plus[l] = hierarchy->ConstructTruePforFormul(l, space_names_funct,
                                                                         *offsets_hdivh1[l], *offsets_hdivh1[l + 1]);
            P_mg_plus[l] = new BlkInterpolationWithBNDforTranspose(*BlockP_mg_nobnd_plus[l],
                                                              *coarsebnd_indces_funct_lvls[l],
                                                              *offsets_hdivh1[l], *offsets_hdivh1[l + 1]);
        }

        if (l == 0)
            BlockOps_mg_plus[l] = funct_op;
        else
        {
            BlockOps_mg_plus[l] = new RAPBlockHypreOperator(*BlockP_mg_nobnd_plus[l - 1],
                    *BlockOps_mg_plus[l - 1], *BlockP_mg_nobnd_plus[l - 1], *offsets_hdivh1[l]);

            std::vector<Array<int>* > essbdr_tdofs_funct = hierarchy->GetEssBdrTdofsOrDofs
                    ("tdof", space_names_funct, essbdr_attribs, l);
            EliminateBoundaryBlocks(*BlockOps_mg_plus[l], essbdr_tdofs_funct);

            for (unsigned int i = 0; i < essbdr_tdofs_funct.size(); ++i)
                delete essbdr_tdofs_funct[i];
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

            Funct_mat_lvls_mg[0] = problem->ConstructFunctBlkMat(*offsets_sp_hdivh1[0]/*offsets_funct_hdivh1*/);
        }
        else
        {
            offsets_sp_hdivh1[l] = hierarchy->ConstructOffsetsforFormul(l, space_names_funct);

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

            std::vector<Array<int>*> essbdr_tdofs_funct = hierarchy->GetEssBdrTdofsOrDofs
                    ("tdof", space_names_funct, essbdr_attribs, l);

            std::vector<Array<int>*> essbdr_dofs_funct = hierarchy->GetEssBdrTdofsOrDofs
                    ("dof", space_names_funct, essbdr_attribs, l);

            std::vector<Array<int>*> fullbdr_dofs_funct = hierarchy->GetEssBdrTdofsOrDofs
                    ("dof", space_names_funct, fullbdr_attribs, l);

            HcurlSmoothers_lvls[l] = new HcurlGSSSmoother(*BlockOps_mg_plus[l],
                                                     *hierarchy->GetDivfreeDop(l),
                                                     *hierarchy->GetEssBdrTdofsOrDofs("tdof", SpaceName::HCURL,
                                                                               essbdr_attribs_Hcurl, l),
                                                     //hierarchy->GetEssBdrTdofsOrDofs("tdof",
                                                                               //space_names_funct,
                                                                               //essbdr_attribs, l),
                                                     essbdr_tdofs_funct,
                                                     &SweepsNum, *offsets_hdivh1[l]);

            int size = BlockOps_mg_plus[l]->Height();

            bool optimized_localsolve = true;

            el2dofs_row_offsets[l] = new Array<int>();
            el2dofs_col_offsets[l] = new Array<int>();

            AE_e_lvls[l] = Transpose(*hierarchy->GetPspace(SpaceName::L2, l));
            if (strcmp(space_for_S,"H1") == 0) // S is present
            {
                SchwarzSmoothers_lvls[l] = new LocalProblemSolverWithS
                        (size, *Funct_mat_lvls_mg[l], *Constraint_mat_lvls_mg[l],
                         hierarchy->GetDofTrueDof(space_names_funct, l), *AE_e_lvls[l],
                         *hierarchy->GetElementToDofs(space_names_funct, l, el2dofs_row_offsets[l],
                                                      el2dofs_col_offsets[l]),
                         *hierarchy->GetElementToDofs(SpaceName::L2, l),
                         //hierarchy->GetEssBdrTdofsOrDofs("dof", space_names_funct, fullbdr_attribs, l),
                         fullbdr_dofs_funct,
                         //hierarchy->GetEssBdrTdofsOrDofs("dof", space_names_funct, essbdr_attribs, l),
                         essbdr_dofs_funct,
                         optimized_localsolve);
            }
            else // no S
            {
                SchwarzSmoothers_lvls[l] = new LocalProblemSolver
                        (size, *Funct_mat_lvls_mg[l], *Constraint_mat_lvls_mg[l],
                         hierarchy->GetDofTrueDof(space_names_funct, l), *AE_e_lvls[l],
                         *hierarchy->GetElementToDofs(space_names_funct, l, el2dofs_row_offsets[l],
                                                      el2dofs_col_offsets[l]),
                         *hierarchy->GetElementToDofs(SpaceName::L2, l),
                         //hierarchy->GetEssBdrTdofsOrDofs("dof", space_names_funct, fullbdr_attribs, l),
                         fullbdr_dofs_funct,
                         //hierarchy->GetEssBdrTdofsOrDofs("dof", space_names_funct, essbdr_attribs, l),
                         essbdr_dofs_funct,
                         optimized_localsolve);
            }

            for (unsigned int i = 0; i < essbdr_tdofs_funct.size(); ++i)
                delete essbdr_tdofs_funct[i];

            for (unsigned int i = 0; i < fullbdr_dofs_funct.size(); ++i)
                delete fullbdr_dofs_funct[i];

            for (unsigned int i = 0; i < essbdr_dofs_funct.size(); ++i)
                delete essbdr_dofs_funct[i];

#ifdef SOLVE_WITH_LOCALSOLVERS
            Smoo_mg_plus[l] = new SmootherSum(*SchwarzSmoothers_lvls[l], *HcurlSmoothers_lvls[l], *Ops_mg_plus[l]);
#else
            Smoo_mg_plus[l] = HcurlSmoothers_lvls[l];
#endif
        } // end of if l < num_levels - 1
    }

    for (int l = 0; l < nlevels - 1; ++l)
        Ops_mg_special[l] = Ops_mg_plus[l];

    if (verbose)
        std::cout << "Creating the new coarsest solver which works in the div-free subspace \n" << std::flush;

    std::vector<Array<int> * > fullbdr_tdofs_funct_coarse =
            hierarchy->GetEssBdrTdofsOrDofs("tdof", space_names_funct, essbdr_attribs, num_levels - 1);

    Array<int> * essbdr_hcurl_coarse = hierarchy->GetEssBdrTdofsOrDofs("tdof", SpaceName::HCURL,
                                                                       essbdr_attribs_Hcurl, num_levels - 1);

    CoarseSolver_mg_plus = new CoarsestProblemHcurlSolver(Ops_mg_plus[num_levels - 1]->Height(),
                                                     *BlockOps_mg_plus[num_levels - 1],
                                                     *hierarchy->GetDivfreeDop(num_levels - 1),
                                                     //hierarchy->GetEssBdrTdofsOrDofs("tdof",
                                                                                     //space_names_funct,
                                                                                     //essbdr_attribs, num_levels - 1),
                                                     fullbdr_tdofs_funct_coarse,
                                                     *essbdr_hcurl_coarse);

    for (unsigned int i = 0; i < fullbdr_tdofs_funct_coarse.size(); ++i)
        delete fullbdr_tdofs_funct_coarse[i];
    delete essbdr_hcurl_coarse;

    ((CoarsestProblemHcurlSolver*)CoarseSolver_mg_plus)->SetMaxIter(100);
    ((CoarsestProblemHcurlSolver*)CoarseSolver_mg_plus)->SetAbsTol(sqrt(1.0e-32));
    ((CoarsestProblemHcurlSolver*)CoarseSolver_mg_plus)->SetRelTol(sqrt(1.0e-12));
    ((CoarsestProblemHcurlSolver*)CoarseSolver_mg_plus)->ResetSolverParams();

    // old, working interface
    //GeneralMultigrid * GeneralMGprec_plus =
            //new GeneralMultigrid(nlevels, P_mg_plus, Ops_mg_plus, *CoarseSolver_mg_plus, Smoo_mg_plus);

    Array<Operator*> Smoo_ops(nlevels - 1);
#ifdef SOLVE_WITH_LOCALSOLVERS
    for (int i = 0; i < Smoo_ops.Size(); ++i)
        Smoo_ops[i] = mgtools_hierarchy->GetCombinedSmoothers()[i];
#else
    for (int i = 0; i < Smoo_ops.Size(); ++i)
        Smoo_ops[i] = mgtools_hierarchy->GetHcurlSmoothers()[i];
#endif

    // newer interface using MultigridTools
    GeneralMultigrid * GeneralMGprec_plus =
            new GeneralMultigrid(nlevels, mgtools_hierarchy->GetPs_bnd(), mgtools_hierarchy->GetOps(),
                                 *mgtools_hierarchy->GetCoarsestSolver_Hcurl(),
                                 Smoo_ops);

#ifdef WITH_DIVCONSTRAINT_SOLVER
    if (verbose)
        std::cout << "Constructing additionally a particular solution finder through a minimization approach \n";
    chrono.Clear();
    chrono.Start();

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
    for (int i = 0; i < space_names_problem->Size(); ++i)
        coarse_size += hierarchy->GetSpace((*space_names_problem)[i], num_levels - 1)->TrueVSize();

    Array<int> row_offsets_coarse, col_offsets_coarse;

    std::vector<Array<int>* > essbdr_tdofs_funct_coarse =
            hierarchy->GetEssBdrTdofsOrDofs("tdof", space_names_funct, essbdr_attribs, num_levels - 1);

    std::vector<Array<int>* > essbdr_dofs_funct_coarse =
            hierarchy->GetEssBdrTdofsOrDofs("dof", space_names_funct, essbdr_attribs, num_levels - 1);


    CoarsestProblemSolver* CoarsestSolver_partfinder_new =
            new CoarsestProblemSolver(coarse_size,
                                      *Funct_mat_lvls_mg[num_levels - 1],
            *Constraint_mat_lvls_mg[num_levels - 1],
            hierarchy->GetDofTrueDof(space_names_funct, num_levels - 1, row_offsets_coarse, col_offsets_coarse),
            *hierarchy->GetDofTrueDof(SpaceName::L2, num_levels - 1),
            essbdr_dofs_funct_coarse,
            essbdr_tdofs_funct_coarse, true);

    CoarsestSolver_partfinder_new->SetMaxIter(70000);
    CoarsestSolver_partfinder_new->SetAbsTol(1.0e-18);
    CoarsestSolver_partfinder_new->SetRelTol(1.0e-18);
    CoarsestSolver_partfinder_new->ResetSolverParams();

    for (unsigned int i = 0; i < essbdr_dofs_funct_coarse.size(); ++i)
        delete essbdr_dofs_funct_coarse[i];

    for (unsigned int i = 0; i < essbdr_tdofs_funct_coarse.size(); ++i)
        delete essbdr_tdofs_funct_coarse[i];

    // newer constructor
    bool opt_localsolvers = true;
    bool with_hcurl_smoothers = true;
    // old, works
    //DivConstraintSolver PartsolFinder(*problem, *hierarchy, opt_localsolvers,
                                      //with_hcurl_smoothers, verbose);

    // newer
    DivConstraintSolver PartsolFinder(*mgtools_hierarchy, opt_localsolvers,
                                      with_hcurl_smoothers, verbose);

    FunctionCoefficient * rhs_coeff = problem->GetFEformulation().GetFormulation()->GetTest()->GetRhs();
    ParLinearForm * constrfform_new = new ParLinearForm(hierarchy->GetSpace(SpaceName::L2, 0));
    constrfform_new->AddDomainIntegrator(new DomainLFIntegrator(*rhs_coeff));
    constrfform_new->Assemble();
    Vector ConstrRhs(hierarchy->GetSpace(SpaceName::L2, 0)->TrueVSize());
    constrfform_new->ParallelAssemble(ConstrRhs);
    delete constrfform_new;

    // older alternative constructor
    /*

    Array< SparseMatrix*> P_L2_lvls(ref_levels);
    //Array< SparseMatrix*> AE_e_lvls(ref_levels); // declared and defined above

    for (int l = 0; l < ref_levels; ++l)
    {
        P_L2_lvls[l] = hierarchy->GetPspace(SpaceName::L2, l);
        //AE_e_lvls[l] = Transpose(*P_L2_lvls[l]);
    }

    std::vector< std::vector<Array<int>* > > essbdr_tdofs_funct_lvls(num_levels);
    for (int l = 0; l < num_levels; ++l)
    {
        essbdr_tdofs_funct_lvls[l] = hierarchy->GetEssBdrTdofsOrDofs
                ("tdof", space_names_funct, essbdr_attribs, l);
    }

    DivConstraintSolver PartsolFinder(comm, num_levels,
                                      AE_e_lvls,
                                      BlockP_mg_nobnd_plus,
                                      P_L2_lvls,
                                      essbdr_tdofs_funct_lvls,
                                      Ops_mg_special,
                                      (HypreParMatrix&)(problem->GetOp_nobnd()->GetBlock(numblocks_funct,0)),
                                      ConstrRhs, // *constrfform_new,
                                      HcurlSmoothers_lvls,
                                      &LocalSolver_partfinder_lvls_new,
                                      CoarsestSolver_partfinder_new, verbose);
    */

#endif
    chrono.Stop();
    if (verbose)
        std::cout << "PartSolFinder was created in "<< chrono.RealTime() <<" seconds.\n";
    chrono.Clear();
    chrono.Start();

    BlockVector ParticSol(offsets_func);
    ParticSol = 0.0;

#ifdef WITH_DIVCONSTRAINT_SOLVER
    //PartsolFinder.Mult(*Xinit_truedofs, ParticSol);
    PartsolFinder.FindParticularSolution(*Xinit_truedofs, ParticSol, ConstrRhs, verbose);

    //std::cout << "ParticSol norm = " << ParticSol.Norml2() / sqrt (ParticSol.Size()) << "\n";
    //MPI_Finalize();
    //return 0;
#else
    ParticSol.GetBlock(0) = Sigmahat_truedofs;
    //Sigmahat->ParallelProject(ParticSol.GetBlock(0));
#endif

    chrono.Stop();

    if (verbose)
        std::cout << "Particular solution was found in " << chrono.RealTime() <<" seconds.\n";
    chrono.Clear();
    chrono.Start();

    // checking that the computed particular solution satisfies essential boundary conditions
    if (verbose)
        std::cout << "Checking that particular solution satisfied the boundary conditions \n";
    for (int blk = 0; blk < numblocks_funct; ++blk)
        problem->ComputeBndError(ParticSol, blk);

    if (verbose)
        std::cout << "Checking that particular solution in parallel version satisfies the divergence constraint \n";
    MFEM_ASSERT(CheckConstrRes(ParticSol.GetBlock(0), *Constraint_global, &problem->GetRhs().GetBlock(numblocks_funct),
                               "in the main code for the particular solution"), "");

    int TestmaxIter(400);

    CGSolver Testsolver(MPI_COMM_WORLD);
    Testsolver.SetAbsTol(sqrt(atol));
    Testsolver.SetRelTol(sqrt(rtol));
    Testsolver.SetMaxIter(TestmaxIter);

    Testsolver.SetOperator(*funct_op);
    Testsolver.SetPreconditioner(*GeneralMGprec_plus);

    Testsolver.SetPrintLevel(1);

    BlockVector trueXtest(offsets_func);
    trueXtest = 0.0;

    BlockVector trueRhstest(offsets_func);
    for (int blk = 0; blk < numblocks_funct; ++blk)
        trueRhstest.GetBlock(blk) = problem->GetRhs().GetBlock(blk);

    BlockVector trueRhstest_funct(offsets_func);
    trueRhstest_funct = trueRhstest;

    // trueRhstest = F - Funct * particular solution (= residual), on true dofs
    BlockVector truevec(offsets_problem);
    truevec = 0.0;
    for (int blk = 0; blk < numblocks_funct; ++blk)
        truevec.GetBlock(blk) = ParticSol.GetBlock(blk);

    BlockVector truetemp(offsets_problem);
    problem->GetOp()->Mult(truevec, truetemp);
    truetemp.GetBlock(numblocks_funct) = 0.0;

    for (int blk = 0; blk < numblocks_funct; ++blk)
        trueRhstest.GetBlock(blk) -= truetemp.GetBlock(blk);

    chrono.Stop();
    if (verbose)
        std::cout << "Solving the system \n";
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
        std::cout << "Global system size = " << "... not computed \n";

        /*
        if (strcmp(space_for_S,"H1") == 0) // S is present
            std::cout << "System size: " << Atest->M() + Ctest->M() << "\n" << std::flush;
        else
            std::cout << "System size: " << Atest->M() << "\n" << std::flush;
        */
    }

    chrono.Clear();
    chrono.Start();

    // Adding back the particular solution
    trueXtest += ParticSol;

    BlockVector tempvec(offsets_problem);
    tempvec = 0.0;
    for (int blk = 0; blk < numblocks_funct; ++blk)
        tempvec.GetBlock(blk) = trueXtest.GetBlock(blk);

    problem->ComputeError(tempvec, verbose, checkbnd);

    //for (int blk = 0; blk < numblocks_funct; ++blk)
        //problem->ComputeError(trueXtest, verbose, checkbnd, blk);

    chrono.Stop();
    if (verbose)
        std::cout << "Errors were computed in " << chrono.RealTime() <<" seconds.\n";

    chrono_total.Stop();
    if (verbose)
        std::cout << "Total time consumed was " << chrono_total.RealTime() <<" seconds.\n";

    // Deallocating memory

    delete Xinit_truedofs;
    delete divfree_dop_mod;

    delete hierarchy;
    delete problem;

    delete descriptor;
    delete mgtools_hierarchy;

    delete divfree_descriptor;
    delete mgtools_divfree_hierarchy;

    for (unsigned int i = 0; i < fullbdr_attribs.size(); ++i)
        delete fullbdr_attribs[i];

    for (unsigned int i = 0; i < coarsebnd_indces_funct_lvls.size(); ++i)
        delete coarsebnd_indces_funct_lvls[i];

    for (unsigned int i = 0; i < coarsebnd_indces_divfree_lvls.size(); ++i)
        delete coarsebnd_indces_divfree_lvls[i];

    for (int i = 0; i < P_mg.Size(); ++i)
        delete P_mg[i];

    for (int i = 0; i < BlockOps_mg.Size(); ++i)
        delete BlockOps_mg[i];

    for (int i = 0; i < Smoo_mg.Size(); ++i)
        delete Smoo_mg[i];

    for (int i = 0; i < P_mg_plus.Size(); ++i)
        delete P_mg_plus[i];

    for (int i = 0; i < BlockOps_mg_plus.Size(); ++i)
        delete BlockOps_mg_plus[i];

    for (int i = 0; i < Funct_mat_lvls_mg.Size(); ++i)
        delete Funct_mat_lvls_mg[i];

    for (int i = 0; i < Constraint_mat_lvls_mg.Size(); ++i)
        delete Constraint_mat_lvls_mg[i];

    for (int i = 0; i < HcurlSmoothers_lvls.Size(); ++i)
        delete HcurlSmoothers_lvls[i];

    for (int i = 0; i < SchwarzSmoothers_lvls.Size(); ++i)
        delete SchwarzSmoothers_lvls[i];

    for (int i = 0; i < AE_e_lvls.Size(); ++i)
        delete AE_e_lvls[i];

    delete CoarsePrec_mg;
    delete CoarseSolver_mg_plus;

#ifdef WITH_DIVCONSTRAINT_SOLVER
    delete CoarsestSolver_partfinder_new;
#endif

    delete GeneralMGprec_plus;

    for (unsigned int i = 0; i < dtd_row_offsets.size(); ++i)
        delete dtd_row_offsets[i];

    for (unsigned int i = 0; i < dtd_col_offsets.size(); ++i)
        delete dtd_col_offsets[i];

    for (unsigned int i = 0; i < el2dofs_row_offsets.size(); ++i)
        delete el2dofs_row_offsets[i];

    for (unsigned int i = 0; i < el2dofs_col_offsets.size(); ++i)
        delete el2dofs_col_offsets[i];

    for (unsigned int i = 0; i < offsets_hdivh1.size(); ++i)
        delete offsets_hdivh1[i];

    for (unsigned int i = 0; i < offsets_sp_hdivh1.size(); ++i)
        delete offsets_sp_hdivh1[i];

    for (unsigned int i = 0; i < divfree_offsets.size(); ++i)
        delete divfree_offsets[i];

    delete bdr_conds;
    delete formulat;
    delete fe_formulat;

    delete formulat_divfree;
    delete fe_formulat_divfree;

    MPI_Finalize();
    return 0;
}

