///                           MFEM(with 4D elements) CFOSLS for 3D/4D transport equation
///                                     solved by a preconditioner MINRES.
///
/// ARCHIVED EXAMPLE: runnable, but uncleaned for memory leaks and not very well commented.
///
/// Essentially, it's a modification of cfosls_hyperbolic.cpp example with older setup (no
/// stuff from cfosls/) but allows one to use a coarser Lagrange multiplier for the constraint
/// and to use vector H1 for sigma.
/// The level gap between fine level (used to discretize sigma (and S)) and coarse level (where
/// Lagrange multiplier lives) is controlled by parameter 'level_gap'.
///
/// (*) The example was tested in serial and in parallel and produced reasonable results.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>

// if active, adds ||div bS - f||^2 to the minimization functional
#define DIV_BS

using namespace std;
using namespace mfem;
using std::shared_ptr;
using std::make_shared;

int main(int argc, char *argv[])
{
    // 1. Initialize MPI
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    bool verbose = (myid == 0);
    bool visualization = 0;

    int nDimensions     = 3;
    int numsol          = 0;

    int ser_ref_levels  = 1;
    int par_ref_levels  = 1;

    const char *formulation = "cfosls"; // "cfosls" or "fosls"
    const char *space_for_S = "H1";     // "H1" or "L2"
    const char *space_for_sigma = "H1"; // "Hdiv" or "H1"
    bool eliminateS = false;            // in case space_for_S = "L2" defines whether we eliminate S from the system
    bool keep_divdiv = false;           // in case space_for_S = "L2" defines whether we keep div-div term in the system

    // solver options
    int prec_option = 1; //defines whether to use preconditioner or not, and which one
    bool use_ADS;

    // level gap between coarse an fine grid (T_H and T_h)
    int level_gap = 0;

    const char *mesh_file = "../data/cube_3d_moderate.mesh";
    //const char *mesh_file = "../data/square_2d_moderate.mesh";
    //const char *mesh_file = "../data/cube4d_low.MFEM";
    //const char *mesh_file = "../data/cube4d.MFEM";
    //const char *mesh_file = "../data/orthotope3D_moderate.mesh";
    //const char *mesh_file = "../data/sphere3D_0.1to0.2.mesh";
    //const char * mesh_file = "../data/orthotope3D_fine.mesh";

    int feorder         = 0;

    if (verbose)
        cout << "Solving (С)FOSLS Transport equation with coarsened space for Lagrange multiplier \n";

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
    args.AddOption(&prec_option, "-precopt", "--prec-option",
                   "Preconditioner choice (0, 1 or 2 for now).");
    args.AddOption(&formulation, "-form", "--formul",
                   "Formulation to use (cfosls or fosls).");
    args.AddOption(&space_for_S, "-spaceS", "--spaceS",
                   "Space for S (H1 or L2).");
    args.AddOption(&space_for_sigma, "-spacesigma", "--spacesigma",
                   "Space for sigma (Hdiv or H1).");
    args.AddOption(&eliminateS, "-elims", "--eliminateS", "-no-elims",
                   "--no-eliminateS",
                   "Turn on/off elimination of S in L2 formulation.");
    args.AddOption(&keep_divdiv, "-divdiv", "--divdiv", "-no-divdiv",
                   "--no-divdiv",
                   "Defines if div-div term is/ is not kept in the system.");
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

    if (verbose)
    {
        if (strcmp(formulation,"cfosls") == 0)
            std::cout << "formulation: CFOSLS \n";
        else
            std::cout << "formulation: FOSLS \n";

        if (strcmp(space_for_sigma,"Hdiv") == 0)
            std::cout << "Space for sigma: Hdiv \n";
        else
            std::cout << "Space for sigma: H1 \n";

        if (strcmp(space_for_S,"H1") == 0)
            std::cout << "Space for S: H1 \n";
        else
            std::cout << "Space for S: L2 \n";

        if (strcmp(space_for_S,"L2") == 0)
        {
            std::cout << "S: is ";
            if (!eliminateS)
                std::cout << "not ";
            std::cout << "eliminated from the system \n";
        }

        std::cout << "div-div term: is ";
        if (keep_divdiv)
            std::cout << "not ";
        std::cout << "eliminated \n";
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


    switch (prec_option)
    {
    case 1: // smth simple like AMS
        use_ADS = false;
        break;
    case 2: // MG
        use_ADS = true;
        break;
    default: // no preconditioner
        break;
    }

    if (verbose)
    {
        std::cout << "use_ADS = " << use_ADS << "\n";
    }

    MFEM_ASSERT(strcmp(formulation,"cfosls") == 0 || strcmp(formulation,"fosls") == 0,
                "Formulation must be cfosls or fosls!\n");
    MFEM_ASSERT(strcmp(space_for_S,"H1") == 0 || strcmp(space_for_S,"L2") == 0,
                "Space for S must be H1 or L2!\n");
    MFEM_ASSERT(strcmp(space_for_sigma,"Hdiv") == 0 || strcmp(space_for_sigma,"H1") == 0,
                "Space for sigma must be Hdiv or H1!\n");

    MFEM_ASSERT(!strcmp(space_for_sigma,"H1") == 0 || (strcmp(space_for_sigma,"H1") == 0
                                                       && strcmp(space_for_S,"H1") == 0),
                "Sigma from H1vec must be coupled with S from H1!\n");
    MFEM_ASSERT(!strcmp(space_for_sigma,"H1") == 0 || (strcmp(space_for_sigma,"H1") == 0 && use_ADS == false),
                "ADS cannot be used when sigma is from H1vec!\n");
    MFEM_ASSERT(!(strcmp(formulation,"fosls") == 0 && strcmp(space_for_S,"L2") == 0 && !keep_divdiv),
                "For FOSLS formulation with S from L2 div-div term must be present!\n");
    MFEM_ASSERT(!(strcmp(formulation,"cfosls") == 0 && strcmp(space_for_S,"H1") == 0 && keep_divdiv),
                "For CFOSLS formulation with S from H1 div-div term must not be present for sigma!\n");

    if (verbose)
        std::cout << "Number of mpi processes: " << num_procs << "\n";

    StopWatch chrono;

    int max_iter = 150000;
    double rtol = 1e-12;//1e-7;//1e-9;
    double atol = 1e-14;//1e-9;//1e-12;

    Mesh *mesh = NULL;

    shared_ptr<ParMesh> pmesh;

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
    else //if nDimensions is no 3 or 4
    {
        if (verbose)
            cerr << "Case nDimensions = " << nDimensions << " is not supported \n"
                 << flush;
        MPI_Finalize();
        return -1;

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

    MFEM_ASSERT(level_gap>=0 && level_gap<=par_ref_levels, "invalid level_gap!");
    for (int l = 0; l < par_ref_levels-level_gap; l++)
       pmesh->UniformRefinement();

    double h_min, h_max, kappa_min, kappa_max;
    pmesh->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);
    if (verbose)
        std::cout << "coarse mesh steps: min " << h_min << " max " << h_max << "\n";

    int dim = nDimensions;

    FiniteElementCollection *hdiv_coll;
    if ( dim == 4 )
    {
        hdiv_coll = new RT0_4DFECollection;
        if(verbose)
            cout << "RT: order 0 for 4D" << endl;
    }
    else
    {
        hdiv_coll = new RT_FECollection(feorder, dim);
        if(verbose)
            cout << "RT: order " << feorder << " for 3D" << endl;
    }

    if (dim == 4)
        MFEM_ASSERT(feorder==0, "Only lowest order elements are support in 4D!");
    FiniteElementCollection *h1_coll;
    if (dim == 4)
    {
        h1_coll = new LinearFECollection;
        if (verbose)
            cout << "H1 in 4D: linear elements are used" << endl;
    }
    else
    {
        h1_coll = new H1_FECollection(feorder+1, dim);
        if(verbose)
            cout << "H1: order " << feorder + 1 << " for 3D" << endl;
    }
    FiniteElementCollection *l2_coll = new L2_FECollection(feorder, dim);
    if(verbose)
        cout << "L2: order " << feorder << endl;

    HypreParMatrix * P_W;
    Array<HypreParMatrix*> P_Ws(level_gap);
    ParFiniteElementSpace *coarseW_space_help = new ParFiniteElementSpace(pmesh.get(), l2_coll); // space for mu

    ParFiniteElementSpace *coarseW_space = new ParFiniteElementSpace(pmesh.get(), l2_coll);
    ParFiniteElementSpace *W_space = new ParFiniteElementSpace(pmesh.get(), l2_coll);

    for (int l = 0; l < level_gap; l++)
    {
        coarseW_space_help->Update();

        pmesh->UniformRefinement();

        W_space->Update();

        {
            auto d_td_coarse_W = coarseW_space_help->Dof_TrueDof_Matrix();
            auto P_W_loc_tmp = (SparseMatrix *)W_space->GetUpdateOperator();
            auto P_W_local = RemoveZeroEntries(*P_W_loc_tmp);
            unique_ptr<SparseMatrix>RP_W_local(
                        Mult(*W_space->GetRestrictionMatrix(), *P_W_local));

            if (level_gap==1)
            {
                P_W = d_td_coarse_W->LeftDiagMult(
                            *RP_W_local, W_space->GetTrueDofOffsets());
                P_W->CopyColStarts();
                P_W->CopyRowStarts();
            }
            else
            {
                P_Ws[l] = d_td_coarse_W->LeftDiagMult(
                            *RP_W_local, W_space->GetTrueDofOffsets());
                P_Ws[l]->CopyColStarts();
                P_Ws[l]->CopyRowStarts();
            }
            delete P_W_local;
        }
    } // end of loop over mesh levels

    // Combine the interpolation matrices from different levels
    Array<HypreParMatrix*> help_Ws(level_gap-2);
    if (level_gap > 2)
    {
        help_Ws[0] = ParMult(P_Ws[1],P_Ws[0]);
    }
    else if (level_gap == 2)
    {
        P_W = ParMult(P_Ws[1],P_Ws[0]);
    }

    for (int l = 0; l < level_gap-3; l++)
    {
        help_Ws[l+1] = ParMult(P_Ws[l+2],help_Ws[l]);
    }
    if (level_gap > 2)
    {
        P_W = ParMult(P_Ws[level_gap-1],help_Ws[level_gap-3]);
    }


    //if(dim==3) pmesh->ReorientTetMesh();

    pmesh->PrintInfo(std::cout); if(verbose) cout << endl;

    // 6. Define a parallel finite element space on the parallel mesh. Here we
    //    use the Raviart-Thomas finite elements of the specified order.
    ParFiniteElementSpace *R_space = new ParFiniteElementSpace(pmesh.get(), hdiv_coll);
    ParFiniteElementSpace *H_space = new ParFiniteElementSpace(pmesh.get(), h1_coll);

    ParFiniteElementSpace *H1vec_space;
    if (strcmp(space_for_sigma,"H1") == 0)
        H1vec_space = new ParFiniteElementSpace(pmesh.get(), h1_coll, dim, Ordering::byVDIM);

    ParFiniteElementSpace * Sigma_space;
    if (strcmp(space_for_sigma,"Hdiv") == 0)
        Sigma_space = R_space;
    else
        Sigma_space = H1vec_space;

    ParFiniteElementSpace * S_space;
    if (strcmp(space_for_S,"H1") == 0)
        S_space = H_space;
    else // "L2"
        S_space = W_space;

    HYPRE_Int dimR = R_space->GlobalTrueVSize();
    HYPRE_Int dimH = H_space->GlobalTrueVSize();
    HYPRE_Int dimHvec;
    if (strcmp(space_for_sigma,"H1") == 0)
        dimHvec = H1vec_space->GlobalTrueVSize();
    HYPRE_Int dimW = W_space->GlobalTrueVSize();
    HYPRE_Int dimWcoarse = coarseW_space->GlobalTrueVSize();

    if (verbose)
    {
       std::cout << "***********************************************************\n";
       std::cout << "dim H(div)_h = " << dimR << ", ";
       if (strcmp(space_for_sigma,"H1") == 0)
           std::cout << "dim H1vec_h = " << dimHvec << ", ";
       std::cout << "dim H1_h = " << dimH << ", ";
       std::cout << "dim L2_h = " << dimW << "\n";
       if (level_gap > 0)
        std::cout << "dim L2_H = " << P_W->Width() << " \n";
       if (strcmp(space_for_sigma,"H1") == 0)
           std::cout << "Number of primary unknowns (sigma and S): " << dimHvec + dimH << "\n";
       else
           std::cout << "Number of primary unknowns (sigma and S): " << dimR + dimH << "\n";
       if (level_gap > 0)
           std::cout << "Number of equations in constraint: " << dimWcoarse << "\n";
       else
           std::cout << "Number of equations in constraint: " << dimW << "\n";
       std::cout << "Spaces we use: \n";
       if (strcmp(space_for_sigma,"Hdiv") == 0)
           std::cout << "H(div)";
       else
           std::cout << "H1vec";
       if (strcmp(space_for_S,"H1") == 0)
           std::cout << " x H1";
       else // "L2"
           if (!eliminateS)
               std::cout << " x L2";
       if (strcmp(formulation,"cfosls") == 0)
           std::cout << " x L2 \n";
       std::cout << "***********************************************************\n";
    }

    // 7. Define the two BlockStructure of the problem.  block_offsets is used
    //    for Vector based on dof (like ParGridFunction or ParLinearForm),
    //    block_trueOffstes is used for Vector based on trueDof (HypreParVector
    //    for the rhs and solution of the linear system).  The offsets computed
    //    here are local to the processor.
    int numblocks = 1;

    if (strcmp(space_for_S,"H1") == 0)
        numblocks++;
    else // "L2"
        if (!eliminateS)
            numblocks++;
    if (strcmp(formulation,"cfosls") == 0)
        numblocks++;

    if (verbose)
        std::cout << "Number of blocks in the formulation: " << numblocks << "\n";

    Array<int> block_offsets(numblocks + 1); // number of variables + 1
    int tempblknum = 0;
    block_offsets[0] = 0;
    tempblknum++;
    block_offsets[tempblknum] = Sigma_space->GetVSize();
    tempblknum++;

    if (strcmp(space_for_S,"H1") == 0)
    {
        block_offsets[tempblknum] = H_space->GetVSize();
        tempblknum++;
    }
    else // "L2"
        if (!eliminateS)
        {
            block_offsets[tempblknum] = W_space->GetVSize();
            tempblknum++;
        }
    if (strcmp(formulation,"cfosls") == 0)
    {
        block_offsets[tempblknum] = W_space->GetVSize();
        tempblknum++;
    }
    block_offsets.PartialSum();

    Array<int> block_trueOffsets(numblocks + 1); // number of variables + 1
    tempblknum = 0;
    block_trueOffsets[0] = 0;
    tempblknum++;
    block_trueOffsets[tempblknum] = Sigma_space->TrueVSize();
    tempblknum++;

    if (strcmp(space_for_S,"H1") == 0)
    {
        block_trueOffsets[tempblknum] = H_space->TrueVSize();
        tempblknum++;
    }
    else // "L2"
        if (!eliminateS)
        {
            block_trueOffsets[tempblknum] = W_space->TrueVSize();
            tempblknum++;
        }
    if (strcmp(formulation,"cfosls") == 0)
    {
        block_trueOffsets[tempblknum] = W_space->TrueVSize();
        tempblknum++;
    }
    block_trueOffsets.PartialSum();

    BlockVector x(block_offsets), rhs(block_offsets);
    BlockVector trueX(block_trueOffsets);
    BlockVector trueRhs(block_trueOffsets);
    x = 0.0;
    rhs = 0.0;
    trueX = 0.0;
    trueRhs = 0.0;

    Array<int> block_finalOffsets(numblocks + 1); // number of variables + 1
    tempblknum = 0;
    block_finalOffsets[0] = 0;
    tempblknum++;
    block_finalOffsets[tempblknum] = Sigma_space->TrueVSize();
    tempblknum++;

    if (strcmp(space_for_S,"H1") == 0)
    {
        block_finalOffsets[tempblknum] = H_space->TrueVSize();
        tempblknum++;
    }
    else // "L2"
        if (!eliminateS)
        {
            block_finalOffsets[tempblknum] = coarseW_space->TrueVSize();
            tempblknum++;
        }
    if (strcmp(formulation,"cfosls") == 0)
    {
        block_finalOffsets[tempblknum] = coarseW_space->TrueVSize();
        tempblknum++;
    }
    block_finalOffsets.PartialSum();
    BlockVector finalRhs(block_finalOffsets);
    finalRhs = 0.0;

    Hyper_test Mytest(nDimensions,numsol);

    ParGridFunction *S_exact = new ParGridFunction(S_space);
    S_exact->ProjectCoefficient(*(Mytest.GetU()));

    ParGridFunction * sigma_exact = new ParGridFunction(Sigma_space);
    sigma_exact->ProjectCoefficient(*(Mytest.GetSigma()));

    x.GetBlock(0) = *sigma_exact;
    if (strcmp(space_for_S,"H1") == 0)
        x.GetBlock(1) = *S_exact;

   // 8. Define the coefficients, analytical solution, and rhs of the PDE.
   ConstantCoefficient zero(.0);

   //----------------------------------------------------------
   // Setting boundary conditions.
   //----------------------------------------------------------

   Array<int> ess_bdrS(pmesh->bdr_attributes.Max());
   ess_bdrS = 0;
   if (strcmp(space_for_S,"H1") == 0)
       ess_bdrS[0] = 1; // t = 0
   Array<int> ess_bdrSigma(pmesh->bdr_attributes.Max());
   ess_bdrSigma = 0;
   if (strcmp(space_for_S,"L2") == 0) // S is from L2, so we impose bdr condition for sigma at t = 0
   {
       ess_bdrSigma[0] = 1;
   }

   if (verbose)
   {
       std::cout << "Boundary conditions: \n";
       std::cout << "ess bdr Sigma: \n";
       ess_bdrSigma.Print(std::cout, pmesh->bdr_attributes.Max());
       std::cout << "ess bdr S: \n";
       ess_bdrS.Print(std::cout, pmesh->bdr_attributes.Max());
   }
   //-----------------------

   // 9. Define the parallel grid function and parallel linear forms, solution
   //    vector and rhs.

   ParLinearForm *fform = new ParLinearForm(Sigma_space);
   if (strcmp(space_for_S,"L2") == 0 && keep_divdiv) // if L2 for S and we keep div-div term
       fform->AddDomainIntegrator(new VectordivDomainLFIntegrator(*Mytest.GetRhs()));

   fform->Assemble();

   ParLinearForm *qform;
   if (strcmp(space_for_S,"H1") == 0 || !eliminateS)
   {
       qform = new ParLinearForm(S_space);
       qform->Update(S_space, rhs.GetBlock(1), 0);
   }

   if (strcmp(space_for_S,"H1") == 0)
   {
#ifdef DIV_BS
       qform->AddDomainIntegrator(new GradDomainLFIntegrator(*Mytest.GetBf()));
#endif
       qform->Assemble();//qform->Print();
   }
   else // "L2"
   {
       if (!eliminateS)
       {
           qform->AddDomainIntegrator(new DomainLFIntegrator(zero));
           qform->Assemble();
       }
   }

   ParLinearForm *gform;
   if (strcmp(formulation,"cfosls") == 0)
   {
       gform = new ParLinearForm(W_space);
       gform->AddDomainIntegrator(new DomainLFIntegrator(*Mytest.GetRhs()));
       gform->Assemble();
   }

   // 10. Assemble the finite element matrices for the CFOSLS operator  A
   //     where:

   ParBilinearForm *Ablock(new ParBilinearForm(Sigma_space));
   HypreParMatrix *A;
   if (strcmp(space_for_S,"H1") == 0) // S is from H1
   {
       if (strcmp(space_for_sigma,"Hdiv") == 0) // sigma is from Hdiv
           Ablock->AddDomainIntegrator(new VectorFEMassIntegrator);
       else // sigma is from H1vec
           Ablock->AddDomainIntegrator(new ImproperVectorMassIntegrator);
   }
   else // "L2"
   {
       if (eliminateS) // S is eliminated
           Ablock->AddDomainIntegrator(new VectorFEMassIntegrator(*Mytest.GetKtilda()));
       else // S is present
           Ablock->AddDomainIntegrator(new VectorFEMassIntegrator);
       if (keep_divdiv)
           Ablock->AddDomainIntegrator(new DivDivIntegrator);
   }
   Ablock->Assemble();
   Ablock->EliminateEssentialBC(ess_bdrSigma, x.GetBlock(0), *fform);
   Ablock->Finalize();
   A = Ablock->ParallelAssemble();

   //---------------
   //  C Block:
   //---------------

   ParBilinearForm *Cblock;
   HypreParMatrix *C;
   if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
   {
       Cblock = new ParBilinearForm(S_space);
       if (strcmp(space_for_S,"H1") == 0)
       {
           Cblock->AddDomainIntegrator(new MassIntegrator(*Mytest.GetBtB()));
#ifdef DIV_BS
           Cblock->AddDomainIntegrator(new DiffusionIntegrator(*Mytest.GetBBt()));
#endif
       }
       else // "L2" & !eliminateS
       {
           Cblock->AddDomainIntegrator(new MassIntegrator(*(Mytest.GetBtB())));
       }
       Cblock->Assemble();
       Cblock->EliminateEssentialBC(ess_bdrS, x.GetBlock(1), *qform);
       Cblock->Finalize();
       C = Cblock->ParallelAssemble();
   }

   //---------------
   //  B Block:
   //---------------

   ParMixedBilinearForm *Bblock;
   HypreParMatrix *B;
   HypreParMatrix *BT;
   if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
   {
       Bblock = new ParMixedBilinearForm(Sigma_space, S_space);
       if (strcmp(space_for_sigma,"Hdiv") == 0) // sigma is from Hdiv
       {
           //Bblock->AddDomainIntegrator(new VectorFEMassIntegrator(*Mytest.GetB()));
           Bblock->AddDomainIntegrator(new VectorFEMassIntegrator(*Mytest.GetMinB()));
       }
       else // sigma is from H1
           Bblock->AddDomainIntegrator(new MixedVectorScalarIntegrator(*Mytest.GetMinB()));
       Bblock->Assemble();
       Bblock->EliminateTrialDofs(ess_bdrSigma, x.GetBlock(0), *qform);
       Bblock->EliminateTestDofs(ess_bdrS);
       Bblock->Finalize();

       B = Bblock->ParallelAssemble();
       //*B *= -1.;
       BT = B->Transpose();
   }

   //----------------
   //  D Block:
   //-----------------

   HypreParMatrix *D;
   HypreParMatrix *DT;

   if (strcmp(formulation,"cfosls") == 0)
   {
      ParMixedBilinearForm *Dblock(new ParMixedBilinearForm(Sigma_space, W_space));
      if (strcmp(space_for_sigma,"Hdiv") == 0) // sigma is from Hdiv
        Dblock->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
      else // sigma is from H1vec
        Dblock->AddDomainIntegrator(new VectorDivergenceIntegrator);
      Dblock->Assemble();
      Dblock->EliminateTrialDofs(ess_bdrSigma, x.GetBlock(0), *gform);
      Dblock->Finalize();
      D = Dblock->ParallelAssemble();
      DT = D->Transpose();
   }

   if (level_gap > 0)
   {
       HypreParMatrix *DT_coarse = ParMult(DT, P_W);
       HypreParMatrix *D_coarse = DT_coarse->Transpose();

       D = D_coarse;
       DT = DT_coarse;
   }

   //=======================================================
   // Setting up the block system Matrix
   //-------------------------------------------------------

  tempblknum = 0;
  fform->ParallelAssemble(trueRhs.GetBlock(tempblknum));
  finalRhs.GetBlock(tempblknum) = trueRhs.GetBlock(tempblknum);
  tempblknum++;

  if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
  {
    qform->ParallelAssemble(trueRhs.GetBlock(tempblknum));
    finalRhs.GetBlock(tempblknum) = trueRhs.GetBlock(tempblknum);
    tempblknum++;
  }
  if (strcmp(formulation,"cfosls") == 0)
  {
     gform->ParallelAssemble(trueRhs.GetBlock(tempblknum));
     if (level_gap > 0)
        P_W->MultTranspose(trueRhs.GetBlock(tempblknum), finalRhs.GetBlock(tempblknum));
     else
         finalRhs.GetBlock(tempblknum) = trueRhs.GetBlock(tempblknum);
  }

  BlockOperator *CFOSLSop = new BlockOperator(block_finalOffsets);
  CFOSLSop->SetBlock(0,0, A);
  if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
  {
      CFOSLSop->SetBlock(0,1, BT);
      CFOSLSop->SetBlock(1,0, B);
      CFOSLSop->SetBlock(1,1, C);
      if (strcmp(formulation,"cfosls") == 0)
      {
        CFOSLSop->SetBlock(0,2, DT);
        CFOSLSop->SetBlock(2,0, D);
      }
  }
  else // no S
      if (strcmp(formulation,"cfosls") == 0)
      {
        CFOSLSop->SetBlock(0,1, DT);
        CFOSLSop->SetBlock(1,0, D);
      }

   if (verbose)
       cout << "Final saddle point matrix assembled \n";
   MPI_Barrier(MPI_COMM_WORLD);

   //========================================================
   // Checking residual in the constraint on the exact solution
   //--------------------------------------------------------

   Vector TrueSig(Sigma_space->TrueVSize());
   sigma_exact->ParallelProject(TrueSig);

   Vector resW(coarseW_space->TrueVSize());
   D->Mult(TrueSig, resW);
   resW -= finalRhs.GetBlock(numblocks - 1);

   double norm_resW = resW.Norml2() / sqrt (resW.Size());
   double norm_rhsW = finalRhs.GetBlock(numblocks - 1).Norml2() / sqrt (finalRhs.GetBlock(numblocks - 1).Size());

   std::cout << "|| div * sigma_exact - f || = " << norm_resW << "\n";
   std::cout << "|| div * sigma_exact - f || / || f || = " << norm_resW / norm_rhsW << "\n";

   if (verbose)
       cout << "Residuals checked \n";
   MPI_Barrier(MPI_COMM_WORLD);

   //=======================================================
   // Setting up the preconditioner
   //-------------------------------------------------------

   // Construct the operators for preconditioner
   if (verbose)
   {
       std::cout << "Block diagonal preconditioner: \n";
       if (use_ADS)
           std::cout << "ADS(A) for H(div) \n";
       else
            std::cout << "Diag(A) for H(div) or H1vec \n";
       if (strcmp(space_for_S,"H1") == 0) // S is from H1
           std::cout << "BoomerAMG(C) for H1 \n";
       else
       {
           if (!eliminateS) // S is from L2 and not eliminated
                std::cout << "Diag(C) for L2 \n";
       }
       if (strcmp(formulation,"cfosls") == 0 )
       {
           std::cout << "BoomerAMG(D Diag^(-1)(A) D^t) for L2 lagrange multiplier \n";
       }
       std::cout << "\n";
   }
   chrono.Clear();
   chrono.Start();

   HypreParMatrix *Schur;
   if (strcmp(formulation,"cfosls") == 0 )
   {
      HypreParMatrix *AinvDt = D->Transpose();
      HypreParVector *Ad = new HypreParVector(MPI_COMM_WORLD, A->GetGlobalNumRows(),
                                           A->GetRowStarts());
      A->GetDiag(*Ad);
      AinvDt->InvScaleRows(*Ad);
      Schur = ParMult(D, AinvDt);
   }

   Solver * invA;
   if (use_ADS)
       invA = new HypreADS(*A, Sigma_space);
   else // using Diag(A);
        invA = new HypreDiagScale(*A);

   invA->iterative_mode = false;

   Solver * invC;
   if (strcmp(space_for_S,"H1") == 0) // S is from H1
   {
#ifdef DIV_BS
       invC = new HypreBoomerAMG(*C);
       ((HypreBoomerAMG*)invC)->SetPrintLevel(0);
       ((HypreBoomerAMG*)invC)->iterative_mode = false;
#else
       invC = new HypreDiagScale(*C);
       ((HypreDiagScale*)invC)->iterative_mode = false;
#endif
   }
   else // S from L2
   {
       if (!eliminateS) // S is from L2 and not eliminated
       {
           invC = new HypreDiagScale(*C);
           ((HypreDiagScale*)invC)->iterative_mode = false;
       }
   }

   Solver * invS;
   if (strcmp(formulation,"cfosls") == 0 )
   {
        invS = new HypreBoomerAMG(*Schur);
        ((HypreBoomerAMG *)invS)->SetPrintLevel(0);
        ((HypreBoomerAMG *)invS)->iterative_mode = false;
   }

   BlockDiagonalPreconditioner prec(block_finalOffsets);
   if (prec_option > 0)
   {
       tempblknum = 0;
       prec.SetDiagonalBlock(tempblknum, invA);
       tempblknum++;
       if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
       {
           prec.SetDiagonalBlock(tempblknum, invC);
           tempblknum++;
       }
       if (strcmp(formulation,"cfosls") == 0)
            prec.SetDiagonalBlock(tempblknum, invS);

       if (verbose)
           std::cout << "Preconditioner built in " << chrono.RealTime() << "s. \n";
   }
   else
       if (verbose)
           cout << "No preconditioner is used. \n";

   // 12. Solve the linear system with MINRES.
   //     Check the norm of the unpreconditioned residual.

   chrono.Clear();
   chrono.Start();
   MINRESSolver solver(MPI_COMM_WORLD);
   solver.SetAbsTol(atol);
   solver.SetRelTol(rtol);
   solver.SetMaxIter(max_iter);
   solver.SetOperator(*CFOSLSop);
   if (prec_option > 0)
        solver.SetPreconditioner(prec);
   solver.SetPrintLevel(0);

   BlockVector finalX(block_finalOffsets);
   finalX = 0.0;

   chrono.Clear();
   chrono.Start();
   solver.Mult(finalRhs, finalX);
   chrono.Stop();

   if (verbose)
   {
      if (solver.GetConverged())
         std::cout << "MINRES converged in " << solver.GetNumIterations()
                   << " iterations with a residual norm of " << solver.GetFinalNorm() << ".\n";
      else
         std::cout << "MINRES did not converge in " << solver.GetNumIterations()
                   << " iterations. Residual norm is " << solver.GetFinalNorm() << ".\n";
      std::cout << "MINRES solver took " << chrono.RealTime() << "s. \n";
   }

   ParGridFunction * sigma = new ParGridFunction(Sigma_space);
   sigma->Distribute(&(finalX.GetBlock(0)));

   ParGridFunction * S = new ParGridFunction(S_space);
   if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
       S->Distribute(&(finalX.GetBlock(1)));
   else // no S in the formulation
   {
       ParBilinearForm *Cblock(new ParBilinearForm(S_space));
       Cblock->AddDomainIntegrator(new MassIntegrator(*(Mytest.GetBtB())));
       Cblock->Assemble();
       Cblock->Finalize();
       HypreParMatrix * C = Cblock->ParallelAssemble();

       ParMixedBilinearForm *Bblock(new ParMixedBilinearForm(Sigma_space, S_space));
       Bblock->AddDomainIntegrator(new VectorFEMassIntegrator(*(Mytest.GetB())));
       Bblock->Assemble();
       Bblock->Finalize();
       HypreParMatrix * B = Bblock->ParallelAssemble();
       Vector bTsigma(C->Height());
       B->Mult(finalX.GetBlock(0),bTsigma);

       Vector trueS(C->Height());

       CG(*C, bTsigma, trueS, 0, 5000, 1e-9, 1e-12);
       S->Distribute(trueS);
   }

   // 13. Extract the parallel grid function corresponding to the finite element
   //     approximation X. This is the local solution on each processor. Compute
   //     L2 error norms.

   int order_quad = max(2, 2*feorder+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }


   double err_sigma = sigma->ComputeL2Error(*(Mytest.GetSigma()), irs);
   double norm_sigma = ComputeGlobalLpNorm(2, *(Mytest.GetSigma()), *pmesh, irs);
   if (verbose)
       cout << "|| sigma - sigma_ex || / || sigma_ex || = " << err_sigma / norm_sigma << endl;

   double err_div;
   double norm_div;

   ParGridFunction * DivSigma;
   if (strcmp(space_for_sigma,"Hdiv") == 0) // sigma is from Hdiv
   {
       DivSigma = new ParGridFunction(W_space);
       DiscreteLinearOperator Div(Sigma_space, W_space);
       Div.AddDomainInterpolator(new DivergenceInterpolator());
       Div.Assemble();
       Div.Mult(*sigma, *DivSigma);

       err_div = DivSigma->ComputeL2Error(*(Mytest.GetRhs()),irs);
       norm_div = ComputeGlobalLpNorm(2, *(Mytest.GetRhs()), *pmesh, irs);

   }
   else // sigma is from H1 vec
   {
       DivSigma = new ParGridFunction(coarseW_space);

       if (verbose)
           std::cout << "Don't know how to compute error for div sigma in H1vec case \n";

       /*
        * this doesn't work without creating copies of par mesh
        * for now number of elements for coarseW_sapce is wrong
        * because the pmesh was refined
       ParBilinearForm *Massblock = new ParBilinearForm(coarseW_space);
       Massblock->AddDomainIntegrator(new MassIntegrator);
       Massblock->Assemble();
       Massblock->Finalize();
       HypreParMatrix *MassL2 = Massblock->ParallelAssemble();

       CGSolver solver(comm);

       solver.SetAbsTol(atol);
       solver.SetRelTol(rtol);
       solver.SetMaxIter(max_iter);
       solver.SetOperator(*MassL2);
       solver.SetPrintLevel(0);

       Vector trueRhs(coarseW_space->GetTrueVSize());
       D->Mult(finalX.GetBlock(0), trueRhs);

       Vector trueX(coarseW_space->GetTrueVSize());
       trueX = 0.0;

       chrono.Clear();
       chrono.Start();
       solver.Mult(trueRhs, trueX);
       chrono.Stop();

       DivSigma->Distribute(&trueX);
       */
   }

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

   double err_S = S->ComputeL2Error((*Mytest.GetU()), irs);
   double norm_S = ComputeGlobalLpNorm(2, (*Mytest.GetU()), *pmesh, irs);
   if (verbose)
   {
       std::cout << "|| S_h - S_ex || / || S_ex || = " <<
                    err_S / norm_S << "\n";
   }

   if (strcmp(space_for_S,"H1") == 0) // S is from H1
   {
       FiniteElementCollection * hcurl_coll;
       if(dim==4)
           hcurl_coll = new ND1_4DFECollection;
       else
           hcurl_coll = new ND_FECollection(feorder+1, dim);
       auto *N_space = new ParFiniteElementSpace(pmesh.get(), hcurl_coll);

       DiscreteLinearOperator Grad(S_space, N_space);
       Grad.AddDomainInterpolator(new GradientInterpolator());
       ParGridFunction GradS(N_space);
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

       delete hcurl_coll;
       delete N_space;
   }

   // Check value of functional and mass conservation (unchecked)
   if (strcmp(formulation,"cfosls") == 0) // if CFOSLS, otherwise code requires some changes
   {
       /*
#ifdef DIV_BS
       double localFunctional = 0.0;//-2.0*(trueX.GetBlock(0)*trueRhs.GetBlock(0));
       if (strcmp(space_for_S,"H1") == 0) // S is present
       {
          localFunctional += -2.0*(finalX.GetBlock(1)*finalRhs.GetBlock(1));
          double f_norm = ComputeGlobalLpNorm(2, *Mytest.GetRhs(), *pmesh, irs);
          localFunctional += f_norm * f_norm;
       }

       finalX.GetBlock(numblocks - 1) = 0.0;
       finalRhs = 0.0;;
       CFOSLSop->Mult(finalX, finalRhs);
       localFunctional += finalX*(finalRhs);

       double globalFunctional;
       MPI_Reduce(&localFunctional, &globalFunctional, 1,
                  MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
       if (verbose)
       {
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
#else
       */

       double localFunctional_error = 0.0;

       finalX.GetBlock(numblocks - 1) = 0.0;

       Vector sigma_exact_truedofs(Sigma_space->GetTrueVSize());
       sigma_exact->ParallelProject(sigma_exact_truedofs);
       finalX.GetBlock(0) -= sigma_exact_truedofs;

       if (strcmp(space_for_S,"H1") == 0) // S is present
       {
           Vector S_exact_truedofs(H_space->GetTrueVSize());
           S_exact->ParallelProject(S_exact_truedofs);
               finalX.GetBlock(1) -= S_exact_truedofs;
       }

       finalRhs = 0.0;;
       CFOSLSop->Mult(finalX, finalRhs);
       localFunctional_error += finalX*(finalRhs);

       double globalFunctional_error;
       MPI_Reduce(&localFunctional_error, &globalFunctional_error, 1,
                  MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

       double localFunctional_norm = 0.0;

       BlockVector solvec(block_finalOffsets);
       solvec.GetBlock(numblocks - 1) = 0.0;
       solvec.GetBlock(0) = sigma_exact_truedofs;
       if (strcmp(space_for_S,"H1") == 0) // S is present
       {
           Vector S_exact_truedofs(H_space->GetTrueVSize());
           S_exact->ParallelProject(S_exact_truedofs);
           solvec.GetBlock(1) = S_exact_truedofs;
       }

       BlockVector Asolvec(block_finalOffsets);
       Asolvec = 0.0;

       CFOSLSop->Mult(solvec, Asolvec);
       localFunctional_norm += solvec*(Asolvec);

       double globalFunctional_norm;
       MPI_Reduce(&localFunctional_norm, &globalFunctional_norm, 1,
                  MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

       if (verbose)
       {
           cout << "|| err_sigma_h - L(err_S_h) ||^2 / || sigma_ex_h - L(S_ex_h) ||^2 = " <<
                   globalFunctional_error / globalFunctional_norm << "\n";
       }

//#endif

       ParLinearForm massform(W_space);
       massform.AddDomainIntegrator(new DomainLFIntegrator(*(Mytest.GetRhs())));
       massform.Assemble();

       double mass_loc = massform.Norml1();
       double mass;
       MPI_Reduce(&mass_loc, &mass, 1,
                  MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
       if (verbose)
           cout << "Sum of local mass = " << mass<< "\n";

       trueRhs.GetBlock(numblocks - 1) -= massform;
       double mass_loss_loc = finalRhs.GetBlock(numblocks - 1).Norml1();
       double mass_loss;
       MPI_Reduce(&mass_loss_loc, &mass_loss, 1,
                  MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
       if (verbose)
           cout << "Sum of local mass loss = " << mass_loss << "\n";
   }

   if (verbose)
       cout << "Computing projection errors \n";

   double projection_error_sigma = sigma_exact->ComputeL2Error(*(Mytest.GetSigma()), irs);

   if(verbose)
   {
       cout << "|| sigma_ex - Pi_h sigma_ex || / || sigma_ex || = "
                       << projection_error_sigma / norm_sigma << endl;
   }

   double projection_error_S = S_exact->ComputeL2Error(*(Mytest.GetU()), irs);

   if(verbose)
       cout << "|| S_ex - Pi_h S_ex || / || S_ex || = "
                       << projection_error_S / norm_S << endl;


   if (visualization && nDimensions < 4)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream u_sock(vishost, visport);
      u_sock << "parallel " << num_procs << " " << myid << "\n";
      u_sock.precision(8);
      u_sock << "solution\n" << *pmesh << *sigma_exact << "window_title 'sigma_exact'"
             << endl;
      // Make sure all ranks have sent their 'u' solution before initiating
      // another set of GLVis connections (one from each rank):


      socketstream uu_sock(vishost, visport);
      uu_sock << "parallel " << num_procs << " " << myid << "\n";
      uu_sock.precision(8);
      uu_sock << "solution\n" << *pmesh << *sigma << "window_title 'sigma'"
             << endl;

      *sigma_exact -= *sigma;

      socketstream uuu_sock(vishost, visport);
      uuu_sock << "parallel " << num_procs << " " << myid << "\n";
      uuu_sock.precision(8);
      uuu_sock << "solution\n" << *pmesh << *sigma_exact << "window_title 'difference for sigma'"
             << endl;

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

      MPI_Barrier(pmesh->GetComm());
   }

   // 17. Free the used memory.
   //delete fform;
   //delete CFOSLSop;
   //delete A;

   //delete Ablock;
   if (strcmp(space_for_S,"H1") == 0) // S was from H1
        delete H_space;
   delete W_space;
   delete R_space;
   if (strcmp(space_for_sigma,"H1") == 0) // S was from H1
        delete H1vec_space;
   delete l2_coll;
   delete h1_coll;
   delete hdiv_coll;

   //delete pmesh;

   MPI_Finalize();

   return 0;
}

