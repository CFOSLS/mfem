//                                MFEM(with 4D elements) CFOSLS with S from H1 for 3D/4D hyperbolic equation
//                                  with adaptive refinement
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
// Solver: MINRES preconditioned by boomerAMG or ADS

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>

#define MYZEROTOL (1.0e-13)

//#include "cfosls_testsuite.hpp"
//#include "cfosls_integrators.hpp"
//#include "cfosls_tools.hpp"


#define NEW_SETUP
//#define REGULARIZE_A

#define NEW_INTERFACE


using namespace std;
using namespace mfem;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

double uFun_ex(const Vector& x); // Exact Solution
double uFun_ex_dt(const Vector& xt);
void uFun_ex_gradx(const Vector& xt, Vector& grad);

void bFun_ex (const Vector& xt, Vector& b);
double  bFundiv_ex(const Vector& xt);

double uFun3_ex(const Vector& x); // Exact Solution
double uFun3_ex_dt(const Vector& xt);
void uFun3_ex_gradx(const Vector& xt, Vector& grad);

double uFun4_ex(const Vector& x); // Exact Solution
double uFun4_ex_dt(const Vector& xt);
void uFun4_ex_gradx(const Vector& xt, Vector& grad);

//void bFun4_ex (const Vector& xt, Vector& b);

//void bFun6_ex (const Vector& xt, Vector& b);

double uFun5_ex(const Vector& x); // Exact Solution
double uFun5_ex_dt(const Vector& xt);
void uFun5_ex_gradx(const Vector& xt, Vector& grad);

double uFun6_ex(const Vector& x); // Exact Solution
double uFun6_ex_dt(const Vector& xt);
void uFun6_ex_gradx(const Vector& xt, Vector& grad);

double uFunCylinder_ex(const Vector& x); // Exact Solution
double uFunCylinder_ex_dt(const Vector& xt);
void uFunCylinder_ex_gradx(const Vector& xt, Vector& grad);

double uFun66_ex(const Vector& x); // Exact Solution
double uFun66_ex_dt(const Vector& xt);
void uFun66_ex_gradx(const Vector& xt, Vector& grad);


double uFun2_ex(const Vector& x); // Exact Solution
double uFun2_ex_dt(const Vector& xt);
void uFun2_ex_gradx(const Vector& xt, Vector& grad);

void Hdivtest_fun(const Vector& xt, Vector& out );
double  L2test_fun(const Vector& xt);

double uFun33_ex(const Vector& x); // Exact Solution
double uFun33_ex_dt(const Vector& xt);
void uFun33_ex_gradx(const Vector& xt, Vector& grad);

double uFun10_ex(const Vector& x); // Exact Solution
double uFun10_ex_dt(const Vector& xt);
void uFun10_ex_gradx(const Vector& xt, Vector& grad);

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
    int numsol          = -33;

    int ser_ref_levels  = 0;
    int par_ref_levels  = 0;

    const char *formulation = "cfosls"; // "cfosls" or "fosls"
    const char *space_for_S = "L2";     // "H1" or "L2"
    const char *space_for_sigma = "Hdiv"; // "Hdiv" or "H1"
    bool eliminateS = true;            // in case space_for_S = "L2" defines whether we eliminate S from the system
    bool keep_divdiv = false;           // in case space_for_S = "L2" defines whether we keep div-div term in the system

    // solver options
    int prec_option = 1; //defines whether to use preconditioner or not, and which one
    bool use_ADS;

    const char *mesh_file = "../data/cube_3d_moderate.mesh";
    //const char *mesh_file = "../data/square_2d_moderate.mesh";

    //const char *mesh_file = "../data/cube4d_low.MFEM";

    //const char *mesh_file = "../data/cube4d.MFEM";
    //const char *mesh_file = "dsadsad";
    //const char *mesh_file = "../data/orthotope3D_moderate.mesh";
    //const char *mesh_file = "../data/sphere3D_0.1to0.2.mesh";
    //const char * mesh_file = "../data/orthotope3D_fine.mesh";

    //const char * meshbase_file = "../data/sphere3D_0.1to0.2.mesh";
    //const char * meshbase_file = "../data/sphere3D_0.05to0.1.mesh";
    //const char * meshbase_file = "../data/sphere3D_veryfine.mesh";
    //const char * meshbase_file = "../data/beam-tet.mesh";
    //const char * meshbase_file = "../data/escher-p3.mesh";
    //const char * meshbase_file = "../data/orthotope3D_moderate.mesh";
    //const char * meshbase_file = "../data/orthotope3D_fine.mesh";
    //const char * meshbase_file = "../data/square_2d_moderate.mesh";
    //const char * meshbase_file = "../data/square_2d_fine.mesh";
    //const char * meshbase_file = "../data/square-disc.mesh";
    //const char *meshbase_file = "dsadsad";
    //const char * meshbase_file = "../data/circle_fine_0.1.mfem";
    //const char * meshbase_file = "../data/circle_moderate_0.2.mfem";

    int feorder         = 0;

    if (verbose)
        cout << "Solving (С)FOSLS Transport equation with MFEM & hypre \n";

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


    //mesh_file = "../data/netgen_cylinder_mesh_0.1to0.2.mesh";
    //mesh_file = "../data/pmesh_cylinder_moderate_0.2.mesh";
    //mesh_file = "../data/pmesh_cylinder_fine_0.1.mesh";

    mesh_file = "../data/pmesh_check.mesh";
    //mesh_file = "../data/cube_3d_moderate.mesh";


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

    //MFEM_ASSERT(numsol == 8 && nDimensions == 3, "Adaptive refinement is tested currently only for the older reports' problem in the cylinder! \n");

    MFEM_ASSERT(strcmp(formulation,"cfosls") == 0 || strcmp(formulation,"fosls") == 0, "Formulation must be cfosls or fosls!\n");
    MFEM_ASSERT(strcmp(space_for_S,"H1") == 0 || strcmp(space_for_S,"L2") == 0, "Space for S must be H1 or L2!\n");
    MFEM_ASSERT(strcmp(space_for_sigma,"Hdiv") == 0 || strcmp(space_for_sigma,"H1") == 0, "Space for sigma must be Hdiv or H1!\n");

    MFEM_ASSERT(!strcmp(space_for_sigma,"H1") == 0 || (strcmp(space_for_sigma,"H1") == 0 && strcmp(space_for_S,"H1") == 0), "Sigma from H1vec must be coupled with S from H1!\n");
    MFEM_ASSERT(!strcmp(space_for_sigma,"H1") == 0 || (strcmp(space_for_sigma,"H1") == 0 && use_ADS == false), "ADS cannot be used when sigma is from H1vec!\n");
    MFEM_ASSERT(!(strcmp(formulation,"fosls") == 0 && strcmp(space_for_S,"L2") == 0 && !keep_divdiv), "For FOSLS formulation with S from L2 div-div term must be present!\n");
    MFEM_ASSERT(!(strcmp(formulation,"cfosls") == 0 && strcmp(space_for_S,"H1") == 0 && keep_divdiv), "For CFOSLS formulation with S from H1 div-div term must not be present for sigma!\n");

    if (verbose)
        std::cout << "Number of mpi processes: " << num_procs << "\n";

    StopWatch chrono;

    //DEFAULTED LINEAR SOLVER OPTIONS
    int max_iter = 100000;
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
    //mesh = new Mesh(2, 2, 2, Element::HEXAHEDRON, 1);

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

    for (int l = 0; l < par_ref_levels; l++)
    {
       pmesh->UniformRefinement();
    }

    //if(dim==3) pmesh->ReorientTetMesh();

    pmesh->PrintInfo(std::cout); if(verbose) cout << endl;

    // 6. Define a parallel finite element space on the parallel mesh. Here we
    //    use the Raviart-Thomas finite elements of the specified order.
    int dim = nDimensions;

    /*
#ifdef NEW_SETUP
    CFOSLSHyperbolicFormulation problem_structure(dim, numsol, space_for_S, space_for_sigma, true, pmesh->bdr_attributes.Max(), verbose);
    CFOSLSHyperbolicProblem problem(*pmesh, problem_structure, feorder, prec_option, verbose);

    problem.Solve(verbose);

#else
#endif

    MPI_Finalize();
    return 0;
    */

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

    ParFiniteElementSpace *R_space = new ParFiniteElementSpace(pmesh.get(), hdiv_coll);
    ParFiniteElementSpace *H_space = new ParFiniteElementSpace(pmesh.get(), h1_coll);
    ParFiniteElementSpace *W_space = new ParFiniteElementSpace(pmesh.get(), l2_coll);

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

    if (verbose)
    {
       std::cout << "***********************************************************\n";
       std::cout << "dim H(div)_h = " << dimR << ", ";
       if (strcmp(space_for_sigma,"H1") == 0)
           std::cout << "dim H1vec_h = " << dimHvec << ", ";
       std::cout << "dim H1_h = " << dimH << ", ";
       std::cout << "dim L2_h = " << dimW << "\n";
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

    BlockVector x(block_offsets)/*, rhs(block_offsets)*/;
    BlockVector trueX(block_trueOffsets);
    BlockVector trueRhs(block_trueOffsets);
    x = 0.0;
    //rhs = 0.0;
    trueX = 0.0;
    trueRhs = 0.0;

    Transport_test Mytest(nDimensions,numsol);

    ParGridFunction *S_exact = new ParGridFunction(S_space);
    S_exact->ProjectCoefficient(*(Mytest.scalarS));

    ParGridFunction * sigma_exact = new ParGridFunction(Sigma_space);
    sigma_exact->ProjectCoefficient(*(Mytest.sigma));

    x.GetBlock(0) = *sigma_exact;
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
       fform->AddDomainIntegrator(new VectordivDomainLFIntegrator(*Mytest.scalardivsigma));
   fform->Assemble();

   ParLinearForm *qform;
   if (strcmp(space_for_S,"H1") == 0 || !eliminateS)
   {
       qform = new ParLinearForm(S_space);
       //qform->Update(S_space, rhs.GetBlock(1), 0);
   }

   if (strcmp(space_for_S,"H1") == 0)
   {
       //if (strcmp(space_for_sigma,"Hdiv") == 0 )
           qform->AddDomainIntegrator(new GradDomainLFIntegrator(*Mytest.bf));
       qform->Assemble();
       //qform->Print();
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
       gform->AddDomainIntegrator(new DomainLFIntegrator(*Mytest.scalardivsigma));
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
           Ablock->AddDomainIntegrator(new VectorFEMassIntegrator(*Mytest.Ktilda));
       else // S is present
           Ablock->AddDomainIntegrator(new VectorFEMassIntegrator);
       if (keep_divdiv)
           Ablock->AddDomainIntegrator(new DivDivIntegrator);
#ifdef REGULARIZE_A
       if (verbose)
           std::cout << "regularization is ON \n";
       double h_min, h_max, kappa_min, kappa_max;
       pmesh->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);
       if (verbose)
           std::cout << "coarse mesh steps: min " << h_min << " max " << h_max << "\n";

       double reg_param;
       reg_param = 0.1 * h_min * h_min;
       if (verbose)
           std::cout << "regularization parameter: " << reg_param << "\n";
       ConstantCoefficient reg_coeff(reg_param);
       Ablock->AddDomainIntegrator(new VectorFEMassIntegrator(reg_coeff)); // reduces the convergence rate but helps with iteration count
       //Ablock->AddDomainIntegrator(new DivDivIntegrator(reg_coeff)); // doesn't change much in the iteration count
#endif
   }
   Ablock->Assemble();
   Ablock->EliminateEssentialBC(ess_bdrSigma, x.GetBlock(0), *fform);
   Ablock->Finalize();
   A = Ablock->ParallelAssemble();

   /*
   if (verbose)
       std::cout << "Checking the A matrix \n";

   MPI_Finalize();
   return 0;
   */

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
           if (strcmp(space_for_sigma,"Hdiv") == 0)
               Cblock->AddDomainIntegrator(new H1NormIntegrator(*Mytest.bbT, *Mytest.bTb));
           else
               Cblock->AddDomainIntegrator(new MassIntegrator(*Mytest.bTb));

           /*
            * old code, w/o H1NormIntegrator, gives the same result
           Cblock->AddDomainIntegrator(new MassIntegrator(*Mytest.bTb));
           if (strcmp(space_for_sigma,"Hdiv") == 0)
                Cblock->AddDomainIntegrator(new DiffusionIntegrator(*Mytest.bbT));
           */
       }
       else // "L2" & !eliminateS
       {
           Cblock->AddDomainIntegrator(new MassIntegrator(*(Mytest.bTb)));
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
           //Bblock->AddDomainIntegrator(new VectorFEMassIntegrator(*Mytest.b));
           Bblock->AddDomainIntegrator(new VectorFEMassIntegrator(*Mytest.minb));
       }
       else // sigma is from H1
           Bblock->AddDomainIntegrator(new MixedVectorScalarIntegrator(*Mytest.minb));
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

   ParMixedBilinearForm *Dblock;
   HypreParMatrix *D;
   HypreParMatrix *DT;

   if (strcmp(formulation,"cfosls") == 0)
   {
      Dblock = new ParMixedBilinearForm(Sigma_space, W_space);
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

   //=======================================================
   // Setting up the block system Matrix
   //-------------------------------------------------------

  tempblknum = 0;
  fform->ParallelAssemble(trueRhs.GetBlock(tempblknum));
  tempblknum++;
  if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
  {
    qform->ParallelAssemble(trueRhs.GetBlock(tempblknum));
    tempblknum++;
  }
  if (strcmp(formulation,"cfosls") == 0)
     gform->ParallelAssemble(trueRhs.GetBlock(tempblknum));

  //SparseMatrix C_diag;
  //C->GetDiag(C_diag);
  //C_diag.Print();

  //SparseMatrix B_diag;
  //B->GetDiag(B_diag);
  //B_diag.Print();


  BlockOperator *CFOSLSop = new BlockOperator(block_trueOffsets);
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
       invC = new HypreBoomerAMG(*C);
       ((HypreBoomerAMG*)invC)->SetPrintLevel(0);
       ((HypreBoomerAMG*)invC)->iterative_mode = false;
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

   BlockDiagonalPreconditioner prec(block_trueOffsets);
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
   trueX = 0.0;

   //trueRhs.Print();

   chrono.Clear();
   chrono.Start();
   solver.Mult(trueRhs, trueX);
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

   Vector sigma_exact_truedofs(Sigma_space->TrueVSize());
   sigma_exact->ParallelProject(sigma_exact_truedofs);

   Array<int> EssBnd_tdofs_sigma;
   Sigma_space->GetEssentialTrueDofs(ess_bdrSigma, EssBnd_tdofs_sigma);

   for (int i = 0; i < EssBnd_tdofs_sigma.Size(); ++i)
   {
       SparseMatrix A_diag;
       A->GetDiag(A_diag);

       SparseMatrix DT_diag;
       DT->GetDiag(DT_diag);

       int tdof = EssBnd_tdofs_sigma[i];
       double value_ex = sigma_exact_truedofs[tdof];
       double value_com = trueX.GetBlock(0)[tdof];

       if (fabs(value_ex - value_com) > MYZEROTOL)
       {
           std::cout << "bnd condition is violated for sigma, tdof = " << tdof << " exact value = "
                     << value_ex << ", value_com = " << value_com << ", diff = " << value_ex - value_com << "\n";
           std::cout << "rhs side at this tdof = " << trueRhs.GetBlock(0)[tdof] << "\n";
           //std::cout << "rhs side2 at this tdof = " << trueRhs2.GetBlock(0)[tdof] << "\n";
           //std::cout << "bnd at this tdof = " << trueBnd.GetBlock(0)[tdof] << "\n";
           std::cout << "row entries of A matrix: \n";
           int * A_rowcols = A_diag.GetRowColumns(tdof);
           double * A_rowentries = A_diag.GetRowEntries(tdof);
           for (int j = 0; j < A_diag.RowSize(tdof); ++j)
               std::cout << "(" << A_rowcols[j] << "," << A_rowentries[j] << ") ";
           std::cout << "\n";

           std::cout << "row entries of DT matrix: \n";
           int * DT_rowcols = DT_diag.GetRowColumns(tdof);
           double * DT_rowentries = DT_diag.GetRowEntries(tdof);
           for (int j = 0; j < DT_diag.RowSize(tdof); ++j)
               std::cout << "(" << DT_rowcols[j] << "," << DT_rowentries[j] << ") ";
           std::cout << "\n";
       }
   }

   Vector checkvec1(S_space->TrueVSize());
   checkvec1 = 0.0;
   ParGridFunction * checkgrfun1 = new ParGridFunction(S_space);

   Vector checkvec2(S_space->TrueVSize());
   checkvec2 = 0.0;
   ParGridFunction * checkgrfun2 = new ParGridFunction(S_space);


   if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
   {
       SparseMatrix C_diag;
       C->GetDiag(C_diag);

       SparseMatrix B_diag;
       B->GetDiag(B_diag);

       Vector S_exact_truedofs(S_space->TrueVSize());
       S_exact->ParallelProject(S_exact_truedofs);

       Array<int> EssBnd_tdofs_S;
       S_space->GetEssentialTrueDofs(ess_bdrS, EssBnd_tdofs_S);

       std::set<int> bnd_tdofs_S;

       for (int i = 0; i < EssBnd_tdofs_S.Size(); ++i)
       {
           int tdof = EssBnd_tdofs_S[i];
           bnd_tdofs_S.insert(tdof);
           double value_ex = S_exact_truedofs[tdof];
           double value_com = trueX.GetBlock(1)[tdof];

           checkvec1[tdof] = S_exact_truedofs[tdof];
           checkvec2[tdof] = trueX.GetBlock(1)[tdof];

           //std::cout << "diff = " << value_ex - value_com << "\n";
           if (fabs(value_ex - value_com) > MYZEROTOL)
           {
               std::cout << "bnd condition is violated for S, tdof = " << tdof << " exact value = "
                         << value_ex << ", value_com = " << value_com << ", diff = " << value_ex - value_com << "\n";
               std::cout << "rhs side at this tdof = " << trueRhs.GetBlock(1)[tdof] << "\n";
               //std::cout << "rhs side2 at this tdof = " << trueRhs2.GetBlock(1)[tdof] << "\n";
               //std::cout << "bnd at this tdof = " << trueBnd.GetBlock(1)[tdof] << "\n";
               std::cout << "row entries of C matrix: \n";
               int * C_rowcols = C_diag.GetRowColumns(tdof);
               double * C_rowentries = C_diag.GetRowEntries(tdof);
               for (int j = 0; j < C_diag.RowSize(tdof); ++j)
                   std::cout << "(" << C_rowcols[j] << "," << C_rowentries[j] << ") ";
               std::cout << "\n";
               std::cout << "row entries of B matrix: \n";
               int * B_rowcols = B_diag.GetRowColumns(tdof);
               double * B_rowentries = B_diag.GetRowEntries(tdof);
               for (int j = 0; j < B_diag.RowSize(tdof); ++j)
                   std::cout << "(" << B_rowcols[j] << "," << B_rowentries[j] << ") ";
               std::cout << "\n";

           }
       }

       /*
       for (int i = 0; i < S_exact_truedofs.Size(); ++i)
       {
           if (bnd_tdofs_S.find(i) == bnd_tdofs_S.end())
               trueX.GetBlock(1)[i] = S_exact_truedofs[i];
       }
       */

   }

   //checkvec1.Print();

   checkgrfun1->Distribute(&checkvec1);
   checkgrfun2->Distribute(&checkvec2);

   ParGridFunction * sigma = new ParGridFunction(Sigma_space);
   sigma->Distribute(&(trueX.GetBlock(0)));

   //std::cout << "sigma linf norm = " << sigma->Normlinf() << "\n";
   //sigma->Print();

   ParGridFunction * S = new ParGridFunction(S_space);
   if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
       S->Distribute(&(trueX.GetBlock(1)));
   else // no S in the formulation
   {
       ParBilinearForm *Cblock(new ParBilinearForm(S_space));
       Cblock->AddDomainIntegrator(new MassIntegrator(*(Mytest.bTb)));
       Cblock->Assemble();
       Cblock->Finalize();
       HypreParMatrix * C = Cblock->ParallelAssemble();

       ParMixedBilinearForm *Bblock(new ParMixedBilinearForm(Sigma_space, S_space));
       Bblock->AddDomainIntegrator(new VectorFEMassIntegrator(*(Mytest.b)));
       Bblock->Assemble();
       Bblock->Finalize();
       HypreParMatrix * B = Bblock->ParallelAssemble();
       Vector bTsigma(C->Height());
       B->Mult(trueX.GetBlock(0),bTsigma);

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


   double err_sigma = sigma->ComputeL2Error(*(Mytest.sigma), irs);
   double norm_sigma = ComputeGlobalLpNorm(2, *(Mytest.sigma), *pmesh, irs);
   if (verbose)
       cout << "|| sigma - sigma_ex || / || sigma_ex || = " << err_sigma / norm_sigma << endl;

   DiscreteLinearOperator Div(Sigma_space, W_space);
   Div.AddDomainInterpolator(new DivergenceInterpolator());
   ParGridFunction DivSigma(W_space);
   Div.Assemble();
   Div.Mult(*sigma, DivSigma);

   double err_div = DivSigma.ComputeL2Error(*(Mytest.scalardivsigma),irs);
   double norm_div = ComputeGlobalLpNorm(2, *(Mytest.scalardivsigma), *pmesh, irs);

   if (verbose)
   {
       if (fabs(norm_div) > 1.0e-13)
            cout << "|| div (sigma_h - sigma_ex) || / ||div (sigma_ex)|| = "
                 << err_div/norm_div  << "\n";
       else
           cout << "|| div (sigma_h) || = "
                << err_div  << " (norm_div = 0) \n";
   }

   /*
   if (verbose)
   {
       cout << "Actually it will be ~ continuous L2 + discrete L2 for divergence" << endl;
       cout << "|| sigma_h - sigma_ex ||_Hdiv / || sigma_ex ||_Hdiv = "
                 << sqrt(err_sigma*err_sigma + err_div * err_div)/sqrt(norm_sigma*norm_sigma + norm_div * norm_div)  << "\n";
   }
   */

   // Computing error for S

   double err_S = S->ComputeL2Error((*Mytest.scalarS), irs);
   double norm_S = ComputeGlobalLpNorm(2, (*Mytest.scalarS), *pmesh, irs);
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

   /*
   // Check value of functional and mass conservation
   if (strcmp(formulation,"cfosls") == 0) // if CFOSLS, otherwise code requires some changes
   {
       double localFunctional = 0.0;//-2.0*(trueX.GetBlock(0)*trueRhs.GetBlock(0));
       if (strcmp(space_for_S,"H1") == 0) // S is present
            localFunctional += -2.0*(trueX.GetBlock(1)*trueRhs.GetBlock(1));

       trueX.GetBlock(numblocks - 1) = 0.0;
       trueRhs = 0.0;;
       CFOSLSop->Mult(trueX, trueRhs);
       localFunctional += trueX*(trueRhs);

       double globalFunctional;
       MPI_Reduce(&localFunctional, &globalFunctional, 1,
                  MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
       if (verbose)
       {
           if (strcmp(space_for_S,"H1") == 0) // S is present
           {
               cout << "|| sigma_h - L(S_h) ||^2 + || div_h (bS_h) - f ||^2 = " << globalFunctional+err_div*err_div << "\n";
               cout << "|| f ||^2 = " << norm_div*norm_div  << "\n";
               cout << "Smth is wrong with the functional computation for H1 case \n";
               cout << "Relative Energy Error = " << sqrt(globalFunctional+norm_div*norm_div)/norm_div << "\n";
           }
           else // if S is from L2
           {
               cout << "|| sigma_h - L(S_h) ||^2 + || div_h (sigma_h) - f ||^2 = " << globalFunctional+err_div*err_div << "\n";
               cout << "Energy Error = " << sqrt(globalFunctional+err_div*err_div) << "\n";
           }
       }

       ParLinearForm massform(W_space);
       massform.AddDomainIntegrator(new DomainLFIntegrator(*(Mytest.scalardivsigma)));
       massform.Assemble();

       double mass_loc = massform.Norml1();
       double mass;
       MPI_Reduce(&mass_loc, &mass, 1,
                  MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
       if (verbose)
           cout << "Sum of local mass = " << mass<< "\n";

       trueRhs.GetBlock(numblocks - 1) -= massform;
       double mass_loss_loc = trueRhs.GetBlock(numblocks - 1).Norml1();
       double mass_loss;
       MPI_Reduce(&mass_loss_loc, &mass_loss, 1,
                  MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
       if (verbose)
           cout << "Sum of local mass loss = " << mass_loss << "\n";
   }
   */

   if (verbose)
       cout << "Computing projection errors \n";

   double projection_error_sigma = sigma_exact->ComputeL2Error(*(Mytest.sigma), irs);

   if(verbose)
   {
       cout << "|| sigma_ex - Pi_h sigma_ex || / || sigma_ex || = "
                       << projection_error_sigma / norm_sigma << endl;
   }

   double projection_error_S = S_exact->ComputeL2Error(*(Mytest.scalarS), irs);

   if(verbose)
       cout << "|| S_ex - Pi_h S_ex || / || S_ex || = "
                       << projection_error_S / norm_S << endl;

   if (visualization && nDimensions < 4)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;

      /*
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
      */

      /*
      socketstream check1_sock(vishost, visport);
      check1_sock << "parallel " << num_procs << " " << myid << "\n";
      check1_sock.precision(8);
      MPI_Barrier(pmesh->GetComm());
      check1_sock << "solution\n" << *pmesh << *checkgrfun1 << "window_title 'checkgrfun1 (exact)'"
              << endl;

      socketstream check2_sock(vishost, visport);
      check2_sock << "parallel " << num_procs << " " << myid << "\n";
      check2_sock.precision(8);
      MPI_Barrier(pmesh->GetComm());
      check2_sock << "solution\n" << *pmesh << *checkgrfun2 << "window_title 'checkgrfun2 (computed)'"
              << endl;
      */

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

   //MPI_Finalize();
   //return 0;

   if (verbose)
       std::cout << "Running AMR ... \n";

#ifdef NEW_INTERFACE
   //mfem::D * testtttt = new mfem::D(1,2,3);

   // Hdiv-L2 formulation
   FOSLSFormulation * formulat = new CFOSLSFormulation_HdivL2Hyper (dim, numsol, verbose);
   FOSLSFEFormulation * fe_formulat = new CFOSLSFEFormulation_HdivL2Hyper(*formulat, feorder);
   BdrConditions * bdr_conds = new BdrConditions_CFOSLS_HdivL2_Hyper(*pmesh);
   FOSLSProblem_CFOSLS_HdivL2_Hyper * problem = new FOSLSProblem_CFOSLS_HdivL2_Hyper
           (*pmesh, *bdr_conds, *fe_formulat, prec_option, verbose);

   int numfoslsfuns = 1;

   std::vector<std::pair<int,int> > grfuns_descriptor(numfoslsfuns);
   // this works
   grfuns_descriptor[0] = std::make_pair<int,int>(1, 0);

   /*
   // and this is a test for providing extra grfuns for the estimator
   int n_extragrfuns = 1;
   grfuns_descriptor[0] = std::make_pair<int,int>(-1,0);
   Array<ParGridFunction*> extra_grfuns(n_extragrfuns);
   extra_grfuns[0] = new ParGridFunction(problem->GetPfes(1));
   extra_grfuns[0]->ProjectCoefficient(*problem->GetFEformulation().GetFormulation()->GetTest()->GetRhs());
   */

   Array2D<BilinearFormIntegrator *> integs(numfoslsfuns, numfoslsfuns);
   for (int i = 0; i < integs.NumRows(); ++i)
       for (int j = 0; j < integs.NumCols(); ++j)
           integs(i,j) = NULL;

   integs(0,0) = new VectorFEMassIntegrator(*Mytest.Ktilda);

   // this works
   FOSLSEstimator estimator(*problem, grfuns_descriptor, NULL, integs, verbose);
   // and this is for testing the extra grfuns setup
   //FOSLSEstimator estimator(*problem, grfuns_descriptor, &extra_grfuns, integs, verbose);

   // Hdiv-H1 formulation
   /*
   FOSLSFormulation * formulat = new CFOSLSFormulation_HdivH1Hyper (dim, numsol, verbose);
   FOSLSFEFormulation * fe_formulat = new CFOSLSFEFormulation_HdivH1Hyper(*formulat, feorder);
   BdrConditions * bdr_conds = new BdrConditions_CFOSLS_HdivH1_Hyper(*pmesh);

   FOSLSProblem_CFOSLS_HdivH1_Hyper * problem = new FOSLSProblem_CFOSLS_HdivH1_Hyper
           (*pmesh, *bdr_conds, *fe_formulat, prec_option, verbose);
   */

   //int estimator_option = 1;
   //problem->CreateEstimator(estimator_option, verbose);
   //FOSLSEstimator& estimator = problem->ExtractEstimator(0);
   //FOSLSEstimator estimator(comm, grfuns, integs, verbose);

   problem->AddEstimator(estimator);

   ThresholdRefiner refiner(estimator);
   refiner.SetTotalErrorFraction(0.5);

   // 12. The main AMR loop. In each iteration we solve the problem on the
   //     current mesh, visualize the solution, and refine the mesh.
   const int max_dofs = 1600000;
   for (int it = 0; ; it++)
   {
       HYPRE_Int global_dofs = problem->GlobalTrueProblemSize();

       if (myid == 0)
       {
          cout << "\nAMR iteration " << it << endl;
          cout << "Number of unknowns: " << global_dofs << endl;
       }

       problem->Solve(verbose);

       // 17. Send the solution by socket to a GLVis server.
       if (visualization)
       {
           char vishost[] = "localhost";
           int  visport   = 19916;

           socketstream sigma_sock(vishost, visport);
           sigma_sock << "parallel " << num_procs << " " << myid << "\n";
           sigma_sock << "solution\n" << *pmesh << *sigma << "window_title 'sigma, AMR iter No."
                  << it <<"'" << flush;

           socketstream s_sock(vishost, visport);
           s_sock << "parallel " << num_procs << " " << myid << "\n";
           s_sock << "solution\n" << *pmesh << *S << "window_title 'S, AMR iter No."
                  << it <<"'" << flush;
       }

       // 18. Call the refiner to modify the mesh. The refiner calls the error
       //     estimator to obtain element errors, then it selects elements to be
       //     refined and finally it modifies the mesh. The Stop() method can be
       //     used to determine if a stopping criterion was met.
       refiner.Apply(*problem->GetParMesh());
       if (refiner.Stop())
       {
          if (myid == 0)
          {
             cout << "Stopping criterion satisfied. Stop." << endl;
          }
          break;
       }

       if (global_dofs > max_dofs)
       {
          if (myid == 0)
          {
             cout << "Reached the maximum number of dofs. Stop." << endl;
          }
          break;
       }

       problem->Update();

       problem->BuildSystem(verbose);
   }
   //MPI_Finalize();
   //return 0;
#else

   ParGridFunction * f = new ParGridFunction(W_space);
   f->ProjectCoefficient(*Mytest.scalardivsigma);

   int fosls_func_version = 1;

   int numfoslsfuns = -1;
   if (fosls_func_version == 1)
       numfoslsfuns = numblocks - 1;
   else if (fosls_func_version == 2)
       numfoslsfuns = 3;

   Array<ParGridFunction*> grfuns(numfoslsfuns);

   Array2D<BilinearFormIntegrator *> integs(numfoslsfuns, numfoslsfuns);
   for (int i = 0; i < integs.NumRows(); ++i)
       for (int j = 0; j < integs.NumCols(); ++j)
           integs(i,j) = NULL;

   // version 1, only || sigma - b S ||^2, or || K sigma ||^2
   if (fosls_func_version == 1)
   {
       grfuns[0] = sigma;
       if (strcmp(space_for_S,"H1") == 0) // S is present
           grfuns[1] = S;

       for (int i = 0; i < integs.NumRows(); ++i)
           for (int j = 0; j < integs.NumCols(); ++j)
               integs(i,j) = NULL;

       if (strcmp(space_for_S,"H1") == 0) // S is from H1
       {
           if (strcmp(space_for_sigma,"Hdiv") == 0) // sigma is from Hdiv
               integs(0,0) = new VectorFEMassIntegrator;
           else // sigma is from H1vec
               integs(0,0) = new ImproperVectorMassIntegrator;
       }
       else // "L2"
           integs(0,0) = new VectorFEMassIntegrator(*Mytest.Ktilda);

       if (strcmp(space_for_S,"H1") == 0) // S is present
       {
            /*
             * if using this function for fosls, one also includes (b grad S, b grad S)
             * but then one needs additional terms to make it || div bS - f ||^2
             * which are currently not implemented
            if (strcmp(space_for_sigma,"Hdiv") == 0)
                integs(1,1) = new H1NormIntegrator(*Mytest.bbT, *Mytest.bTb);
            else
                integs(1,1) = new MassIntegrator(*Mytest.bTb);
            */
            integs(1,1) = new MassIntegrator(*Mytest.bTb);

            if (strcmp(space_for_sigma,"Hdiv") == 0) // sigma is from Hdiv
                integs(1,0) = new VectorFEMassIntegrator(*Mytest.minb);
            else // sigma is from H1
                integs(1,0) = new MixedVectorScalarIntegrator(*Mytest.minb);
       }
   }
   else if (fosls_func_version == 2)
   {
       // version 2, only || sigma - b S ||^2 + || div bS - f ||^2
       MFEM_ASSERT(strcmp(space_for_S,"H1") == 0, "Version 2 works only if S is from H1 \n");

       grfuns[0] = sigma;
       grfuns[1] = S;
       grfuns[2] = f;

       if (strcmp(space_for_sigma,"Hdiv") == 0) // sigma is from Hdiv
           integs(0,0) = new VectorFEMassIntegrator;
       else // sigma is from H1vec
           integs(0,0) = new ImproperVectorMassIntegrator;

       integs(1,1) = new H1NormIntegrator(*Mytest.bbT, *Mytest.bTb);
       //integs(1,1) = new MassIntegrator(*Mytest.bTb);

       integs(1,0) = new VectorFEMassIntegrator(*Mytest.minb);

       // integrators related to f (rhs side)
       integs(2,2) = new MassIntegrator;
       integs(1,2) = new MixedDirectionalDerivativeIntegrator(*Mytest.minb);
   }
   else
   {
       MFEM_ABORT("Unsupported version of fosls functional \n");
   }

   // old interface which doesn't work for the blocked case
   //BilinearFormIntegrator *integ = new VectorFEMassIntegrator(*Mytest.Ktilda);
   //FOSLSEstimator estimator(*sigma, *integ);

   FOSLSEstimator estimator(comm, grfuns, integs, verbose);

   //double after_initial_solveestimator.GetEstimate();

   ThresholdRefiner refiner(estimator);
   refiner.SetTotalErrorFraction(0.5);

   delete Ablock;
   Ablock = new ParBilinearForm(Sigma_space);
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
           Ablock->AddDomainIntegrator(new VectorFEMassIntegrator(*Mytest.Ktilda));
       else // S is present
           Ablock->AddDomainIntegrator(new VectorFEMassIntegrator);
       if (keep_divdiv)
           Ablock->AddDomainIntegrator(new DivDivIntegrator);
#ifdef REGULARIZE_A
       if (verbose)
           std::cout << "regularization is ON \n";
       double h_min, h_max, kappa_min, kappa_max;
       pmesh->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);
       if (verbose)
           std::cout << "coarse mesh steps: min " << h_min << " max " << h_max << "\n";

       double reg_param;
       reg_param = 0.1 * h_min * h_min;
       if (verbose)
           std::cout << "regularization parameter: " << reg_param << "\n";
       ConstantCoefficient reg_coeff(reg_param);
       Ablock->AddDomainIntegrator(new VectorFEMassIntegrator(reg_coeff)); // reduces the convergence rate but helps with iteration count
       //Ablock->AddDomainIntegrator(new DivDivIntegrator(reg_coeff)); // doesn't change much in the iteration count
#endif
   }

   delete Dblock;
   Dblock = new ParMixedBilinearForm(Sigma_space, W_space);
   Dblock->AddDomainIntegrator(new VectorFEDivergenceIntegrator);

   if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
   {
       delete Cblock;
       Cblock = new ParBilinearForm(S_space);
       if (strcmp(space_for_sigma,"Hdiv") == 0)
           Cblock->AddDomainIntegrator(new H1NormIntegrator(*Mytest.bbT, *Mytest.bTb));
       else
           Cblock->AddDomainIntegrator(new MassIntegrator(*Mytest.bTb));
   }

   if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
   {
       delete Bblock;
       Bblock = new ParMixedBilinearForm(Sigma_space, S_space);
       if (strcmp(space_for_sigma,"Hdiv") == 0) // sigma is from Hdiv
       {
           //Bblock->AddDomainIntegrator(new VectorFEMassIntegrator(*Mytest.b));
           Bblock->AddDomainIntegrator(new VectorFEMassIntegrator(*Mytest.minb));
       }
       else // sigma is from H1
           Bblock->AddDomainIntegrator(new MixedVectorScalarIntegrator(*Mytest.minb));
   }

   // 12. The main AMR loop. In each iteration we solve the problem on the
   //     current mesh, visualize the solution, and refine the mesh.
   const int max_dofs = 1600000;
   for (int it = 0; ; it++)
   {
      HYPRE_Int global_dofs = Sigma_space->GlobalTrueVSize();
      if (strcmp(space_for_S,"H1") == 0)
          global_dofs += S_space->GlobalTrueVSize();
      if (strcmp(formulation,"cfosls") == 0)
          global_dofs += W_space->GlobalTrueVSize();

      if (myid == 0)
      {
         cout << "\nAMR iteration " << it << endl;
         cout << "Number of unknowns: " << global_dofs << endl;
      }

      //Array<int> block_trueOffsets(numblocks + 1);
      //BlockOperator * CFOSLS
      //BuildSystem("hyperbolic", space_for_S, space_for_sigma)
      // to be implemented... then all PDE's examples could be simplified

      HypreParMatrix *A, *D, *DT, *C, *B, *BT;

      // 13. Assemble the stiffness matrix and the right-hand side. Note that
      //     MFEM doesn't care at this point that the mesh is nonconforming
      //     and parallel. The FE space is considered 'cut' along hanging
      //     edges/faces, and also across processor boundaries.

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

      BlockVector x(block_offsets);
      x = 0.0;

      ParGridFunction *S_exact = new ParGridFunction(S_space);
      S_exact->ProjectCoefficient(*(Mytest.scalarS));

      ParGridFunction * sigma_exact = new ParGridFunction(Sigma_space);
      sigma_exact->ProjectCoefficient(*(Mytest.sigma));

      x.GetBlock(0) = *sigma_exact;
      if (strcmp(space_for_S,"H1") == 0)
          x.GetBlock(1) = *S_exact;

      fform->Assemble();
      if (strcmp(space_for_S,"H1") == 0)
          qform->Assemble();
      gform->Assemble();

      Ablock->Assemble();
      Ablock->EliminateEssentialBC(ess_bdrSigma, x.GetBlock(0), *fform);
      Ablock->Finalize();
      A = Ablock->ParallelAssemble();

      if (strcmp(space_for_S,"H1") == 0)
      {
          Cblock->Assemble();
          Cblock->EliminateEssentialBC(ess_bdrS, x.GetBlock(1), *qform);
          Cblock->Finalize();
          C = Cblock->ParallelAssemble();

          Bblock->Assemble();
          Bblock->EliminateTrialDofs(ess_bdrSigma, x.GetBlock(0), *qform);
          Bblock->EliminateTestDofs(ess_bdrS);
          Bblock->Finalize();
          B = Bblock->ParallelAssemble();
          BT = B->Transpose();
      }

      Dblock->Assemble();
      Dblock->EliminateTrialDofs(ess_bdrSigma, x.GetBlock(0), *gform);
      Dblock->Finalize();
      D = Dblock->ParallelAssemble();
      DT = D->Transpose();

      // 14. Create the parallel linear system: eliminate boundary conditions,
      //     constrain hanging nodes and nodes across processor boundaries.
      //     The system will be solved for true (unconstrained/unique) DOFs only.

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

      BlockVector trueX(block_trueOffsets);
      BlockVector trueRhs(block_trueOffsets);
      trueX = 0.0;
      trueRhs = 0.0;

      tempblknum = 0;
      fform->ParallelAssemble(trueRhs.GetBlock(tempblknum));
      tempblknum++;
      if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
      {
        qform->ParallelAssemble(trueRhs.GetBlock(tempblknum));
        tempblknum++;
      }
      if (strcmp(formulation,"cfosls") == 0)
         gform->ParallelAssemble(trueRhs.GetBlock(tempblknum));

      //SparseMatrix C_diag;
      //C->GetDiag(C_diag);
      //C_diag.Print();

      //SparseMatrix B_diag;
      //B->GetDiag(B_diag);
      //B_diag.Print();

      BlockOperator *CFOSLSop = new BlockOperator(block_trueOffsets);
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


      // 15. Define and apply a parallel PCG solver for AX=B with the BoomerAMG
      //     preconditioner from hypre.

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
          invC = new HypreBoomerAMG(*C);
          ((HypreBoomerAMG*)invC)->SetPrintLevel(0);
          ((HypreBoomerAMG*)invC)->iterative_mode = false;
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

      BlockDiagonalPreconditioner prec(block_trueOffsets);
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


      MINRESSolver solver(MPI_COMM_WORLD);
      solver.SetAbsTol(atol);
      solver.SetRelTol(rtol);
      solver.SetMaxIter(max_iter);
      solver.SetOperator(*CFOSLSop);
      if (prec_option > 0)
           solver.SetPreconditioner(prec);
      solver.SetPrintLevel(0);
      trueX = 0.0;

      //trueRhs.Print();

      chrono.Clear();
      chrono.Start();
      solver.Mult(trueRhs, trueX);

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

      sigma->Distribute(&(trueX.GetBlock(0)));

      if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
          S->Distribute(&(trueX.GetBlock(1)));
      else // no S in the formulation
      {
          ParBilinearForm *Cblock(new ParBilinearForm(S_space));
          Cblock->AddDomainIntegrator(new MassIntegrator(*(Mytest.bTb)));
          Cblock->Assemble();
          Cblock->Finalize();
          HypreParMatrix * C = Cblock->ParallelAssemble();

          ParMixedBilinearForm *Bblock(new ParMixedBilinearForm(Sigma_space, S_space));
          Bblock->AddDomainIntegrator(new VectorFEMassIntegrator(*(Mytest.b)));
          Bblock->Assemble();
          Bblock->Finalize();
          HypreParMatrix * B = Bblock->ParallelAssemble();
          Vector bTsigma(C->Height());
          B->Mult(trueX.GetBlock(0),bTsigma);

          Vector trueS(C->Height());

          CG(*C, bTsigma, trueS, 0, 5000, 1e-9, 1e-12);
          S->Distribute(trueS);

          delete Cblock;
          delete Bblock;
          delete B;
          delete C;
      }

      int order_quad = max(2, 2*feorder+1);
      const IntegrationRule *irs[Geometry::NumGeom];
      for (int i=0; i < Geometry::NumGeom; ++i)
      {
         irs[i] = &(IntRules.Get(i, order_quad));
      }

      double err_sigma = sigma->ComputeL2Error(*(Mytest.sigma), irs);
      double norm_sigma = ComputeGlobalLpNorm(2, *(Mytest.sigma), *pmesh, irs);
      if (verbose)
          cout << "|| sigma - sigma_ex || / || sigma_ex || = " << err_sigma / norm_sigma << endl;

      double err_S = S->ComputeL2Error((*Mytest.scalarS), irs);
      double norm_S = ComputeGlobalLpNorm(2, (*Mytest.scalarS), *pmesh, irs);
      if (verbose)
      {
          std::cout << "|| S_h - S_ex || / || S_ex || = " <<
                       err_S / norm_S << "\n";
      }

      // 17. Send the solution by socket to a GLVis server.
      if (visualization)
      {
          char vishost[] = "localhost";
          int  visport   = 19916;

          socketstream sigma_sock(vishost, visport);
          sigma_sock << "parallel " << num_procs << " " << myid << "\n";
          sigma_sock << "solution\n" << *pmesh << *sigma << "window_title 'sigma, AMR iter No."
                 << it <<"'" << flush;

          socketstream s_sock(vishost, visport);
          s_sock << "parallel " << num_procs << " " << myid << "\n";
          s_sock << "solution\n" << *pmesh << *S << "window_title 'S, AMR iter No."
                 << it <<"'" << flush;
      }

      if (global_dofs > max_dofs)
      {
         if (myid == 0)
         {
            cout << "Reached the maximum number of dofs. Stop." << endl;
         }
         break;
      }

      // 18. Call the refiner to modify the mesh. The refiner calls the error
      //     estimator to obtain element errors, then it selects elements to be
      //     refined and finally it modifies the mesh. The Stop() method can be
      //     used to determine if a stopping criterion was met.
      refiner.Apply(*pmesh);
      if (refiner.Stop())
      {
         if (myid == 0)
         {
            cout << "Stopping criterion satisfied. Stop." << endl;
         }
         break;
      }

      // 19. Update the finite element space (recalculate the number of DOFs,
      //     etc.) and create a grid function update matrix. Apply the matrix
      //     to any GridFunctions over the space. In this case, the update
      //     matrix is an interpolation matrix so the updated GridFunction will
      //     still represent the same function as before refinement.
      Sigma_space->Update();
      if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
          S_space->Update();
      W_space->Update();
      sigma->Update();
      S->Update();

      f->Update();
      f->ProjectCoefficient(*Mytest.scalardivsigma);

      // 21. Inform also the bilinear and linear forms that the space has
      //     changed.
      fform->Update();
      if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
          qform->Update();
      gform->Update();
      Ablock->Update();
      Dblock->Update();
      if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
      {
          Bblock->Update();
          Cblock->Update();
      }

      delete sigma_exact;
      if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
          delete S_exact;
      delete A;
      delete D;
      delete DT;
      if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
      {
          delete C;
          delete B;
          delete BT;
      }
      delete CFOSLSop;
   }
#endif

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

double uFun_ex(const Vector& xt)
{
    double t = xt(xt.Size()-1);
    //return t;
    ////setback
    return sin(t)*exp(t);
}

double uFun_ex_dt(const Vector& xt)
{
    double t = xt(xt.Size()-1);
    return (cos(t) + sin(t)) * exp(t);
}

void uFun_ex_gradx(const Vector& xt, Vector& gradx )
{
    gradx.SetSize(xt.Size() - 1);
    gradx = 0.0;
}


/*

double fFun(const Vector& xt)
{
    double t = xt(xt.Size()-1);
    //double tmp = (xt.Size()==4) ? 1.0 - 2.0 * xt(2) : 0;
    double tmp = (xt.Size()==4) ? 2*M_PI * sin(2*xt(2)*M_PI) : 0;
    //double tmp = (xt.Size()==4) ? M_PI * cos(xt(2)*M_PI) : 0;
    //double tmp = (xt.Size()==4) ? M_PI * sin(xt(2)*M_PI) : 0;
    return cos(t)*exp(t)+sin(t)*exp(t)+(M_PI*cos(xt(1)*M_PI)*cos(xt(0)*M_PI)+
                   2*M_PI*cos(xt(0)*2*M_PI)*cos(xt(1)*M_PI)+tmp) *uFun_ex(xt);
    //return cos(t)*exp(t)+sin(t)*exp(t)+(1.0 - 2.0 * xt(0) + 1.0 - 2.0 * xt(1) +tmp) *uFun_ex(xt);
}
*/

void bFun_ex(const Vector& xt, Vector& b )
{
    b.SetSize(xt.Size());

    //for (int i = 0; i < xt.Size()-1; i++)
        //b(i) = xt(i) * (1 - xt(i));

    //if (xt.Size() == 4)
        //b(2) = 1-cos(2*xt(2)*M_PI);
        //b(2) = sin(xt(2)*M_PI);
        //b(2) = 1-cos(xt(2)*M_PI);

    b(0) = sin(xt(0)*2*M_PI)*cos(xt(1)*M_PI);
    b(1) = sin(xt(1)*M_PI)*cos(xt(0)*M_PI);
    b(2) = 1-cos(2*xt(2)*M_PI);

    b(xt.Size()-1) = 1.;

}

double bFundiv_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
//    double t = xt(xt.Size()-1);
    if (xt.Size() == 4)
        return 2*M_PI * cos(x*2*M_PI)*cos(y*M_PI) + M_PI * cos(y*M_PI)*cos(x*M_PI) + 2*M_PI * sin(2*z*M_PI);
    if (xt.Size() == 3)
        return 2*M_PI * cos(x*2*M_PI)*cos(y*M_PI) + M_PI * cos(y*M_PI)*cos(x*M_PI);
    return 0.0;
}

double uFun2_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    return t * exp(t) * sin (((x - 0.5)*(x - 0.5) + y*y));
}

double uFun2_ex_dt(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    return (1.0 + t) * exp(t) * sin (((x - 0.5)*(x - 0.5) + y*y));
}

void uFun2_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = t * exp(t) * 2.0 * (x - 0.5) * cos (((x - 0.5)*(x - 0.5) + y*y));
    gradx(1) = t * exp(t) * 2.0 * y         * cos (((x - 0.5)*(x - 0.5) + y*y));
}

/*
double fFun2(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    Vector b(3);
    bFunCircle2D_ex(xt,b);
    return (t + 1) * exp(t) * sin (((x - 0.5)*(x - 0.5) + y*y)) +
             t * exp(t) * 2.0 * (x - 0.5) * cos (((x - 0.5)*(x - 0.5) + y*y)) * b(0) +
             t * exp(t) * 2.0 * y         * cos (((x - 0.5)*(x - 0.5) + y*y)) * b(1);
}
*/

double uFun3_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);
    return sin(t)*exp(t) * sin ( M_PI * (x + y + z));
}

double uFun3_ex_dt(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);
    return (sin(t) + cos(t)) * exp(t) * sin ( M_PI * (x + y + z));
}

void uFun3_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = sin(t) * exp(t) * M_PI * cos ( M_PI * (x + y + z));
    gradx(1) = sin(t) * exp(t) * M_PI * cos ( M_PI * (x + y + z));
    gradx(2) = sin(t) * exp(t) * M_PI * cos ( M_PI * (x + y + z));
}


/*
double fFun3(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);
    Vector b(4);
    bFun_ex(xt,b);

    return (cos(t)*exp(t)+sin(t)*exp(t)) * sin ( M_PI * (x + y + z)) +
            sin(t)*exp(t) * M_PI * cos ( M_PI * (x + y + z)) * b(0) +
            sin(t)*exp(t) * M_PI * cos ( M_PI * (x + y + z)) * b(1) +
            sin(t)*exp(t) * M_PI * cos ( M_PI * (x + y + z)) * b(2) +
            (2*M_PI*cos(x*2*M_PI)*cos(y*M_PI) +
             M_PI*cos(y*M_PI)*cos(x*M_PI)+
             + 2*M_PI*sin(z*2*M_PI)) * uFun3_ex(xt);
}
*/

double uFun4_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    return exp(t) * sin (((x - 0.5)*(x - 0.5) + y*y));
}

double uFun4_ex_dt(const Vector& xt)
{
    return uFun4_ex(xt);
}

void uFun4_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = exp(t) * 2.0 * (x - 0.5) * cos (((x - 0.5)*(x - 0.5) + y*y));
    gradx(1) = exp(t) * 2.0 * y         * cos (((x - 0.5)*(x - 0.5) + y*y));
}

double uFun33_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    return exp(t) * sin (((x - 0.5)*(x - 0.5) + y*y + (z -0.25)*(z-0.25) ));
}

double uFun33_ex_dt(const Vector& xt)
{
    return uFun33_ex(xt);
}

void uFun33_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = exp(t) * 2.0 * (x - 0.5) * cos (((x - 0.5)*(x - 0.5) + y*y + (z -0.25)*(z-0.25)));
    gradx(1) = exp(t) * 2.0 * y         * cos (((x - 0.5)*(x - 0.5) + y*y + (z -0.25)*(z-0.25)));
    gradx(2) = exp(t) * 2.0 * (z -0.25) * cos (((x - 0.5)*(x - 0.5) + y*y + (z -0.25)*(z-0.25)));
}
/*
double fFun4(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    Vector b(3);
    bFunCircle2D_ex(xt,b);
    return exp(t) * sin (((x - 0.5)*(x - 0.5) + y*y)) +
             exp(t) * 2.0 * (x - 0.5) * cos (((x - 0.5)*(x - 0.5) + y*y)) * b(0) +
             exp(t) * 2.0 * y         * cos (((x - 0.5)*(x - 0.5) + y*y)) * b(1);
}


double f_natural(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    if ( t > MYZEROTOL)
        return 0.0;
    else
        return (-uFun5_ex(xt));
}
*/

double uFun5_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    if ( t < MYZEROTOL)
        return exp(-100.0 * ((x - 0.5) * (x - 0.5) + y * y));
    else
        return 0.0;
}

double uFun5_ex_dt(const Vector& xt)
{
//    double x = xt(0);
//    double y = xt(1);
//    double t = xt(xt.Size()-1);
    return 0.0;
}

void uFun5_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
//    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = -100.0 * 2.0 * (x - 0.5) * uFun5_ex(xt);
    gradx(1) = -100.0 * 2.0 * y * uFun5_ex(xt);
}


double uFun6_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    return exp(-100.0 * ((x - 0.5) * (x - 0.5) + y * y)) * exp(-10.0*t);
}

double uFun6_ex_dt(const Vector& xt)
{
//    double x = xt(0);
//    double y = xt(1);
//    double t = xt(xt.Size()-1);
    return -10.0 * uFun6_ex(xt);
}

void uFun6_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
//    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = -100.0 * 2.0 * (x - 0.5) * uFun6_ex(xt);
    gradx(1) = -100.0 * 2.0 * y * uFun6_ex(xt);
}

double uFun66_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);
    return exp(-100.0 * ((x - 0.5) * (x - 0.5) + y * y + (z - 0.25)*(z - 0.25))) * exp(-10.0*t);
}

double uFun66_ex_dt(const Vector& xt)
{
//    double x = xt(0);
//    double y = xt(1);
//    double z = xt(2);
//    double t = xt(xt.Size()-1);
    return -10.0 * uFun6_ex(xt);
}

void uFun66_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
//    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = -100.0 * 2.0 * (x - 0.5)  * uFun6_ex(xt);
    gradx(1) = -100.0 * 2.0 * y          * uFun6_ex(xt);
    gradx(2) = -100.0 * 2.0 * (z - 0.25) * uFun6_ex(xt);
}

void Hdivtest_fun(const Vector& xt, Vector& out )
{
    out.SetSize(xt.Size());

    double x = xt(0);
//    double y = xt(1);
//    double z = xt(2);
//    double t = xt(xt.Size()-1);

    out(0) = x;
    out(1) = 0.0;
    out(2) = 0.0;
    out(xt.Size()-1) = 0.;

}

double L2test_fun(const Vector& xt)
{
    double x = xt(0);
//    double y = xt(1);
//    double z = xt(2);
//    double t = xt(xt.Size()-1);

    return x;
}


double uFun10_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    return sin(t)*exp(t)*x*y;
}

double uFun10_ex_dt(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
//    double z = xt(2);
    double t = xt(xt.Size()-1);
    return (cos(t)*exp(t) + sin(t)*exp(t)) * x * y;
}

void uFun10_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
//    double z = xt(2);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = sin(t)*exp(t)*y;
    gradx(1) = sin(t)*exp(t)*x;
    gradx(2) = 0.0;
}
