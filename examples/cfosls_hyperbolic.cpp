///                           MFEM(with 4D elements) CFOSLS for 3D/4D transport equation
///                                     solved by a preconditioner MINRES.
///
/// The problem considered in this example is
///                             du/dt + b * u = f (either 3D or 4D in space-time)
/// casted in the CFOSLS formulation
/// 1) either in Hdiv-L2 case:
///                             (K sigma, sigma) -> min
/// where sigma is from H(div), u is recovered (as an element of L^2) from sigma = b * u,
/// and K = (I - bbT / || b ||);
/// 2) or in Hdiv-H1-L2 case
///                             || sigma - b * u || ^2 -> min
/// where sigma is from H(div) and u is from H^1;
/// minimizing in all cases under the constraint
///                             div sigma = f.
///
/// The problem is discretized using RT, linear Lagrange and discontinuous constants in 3D/4D.
/// The current 3D tests are either in cube (preferred) or in a cylinder, with a rotational velocity field b.
///
/// This example demonstrates usage of several classes from mfem/cfosls/, such as
/// FOSLSProblem and show its equivalence to the standard MFEM's way of assembling and solving the problem
///
/// (*) There are a lot of options for the formulation in this example, many of them were not tested in a while.
/// Hence they might work incorrectly.
/// (**) This code was tested in serial and in parallel.
/// (***) The example was tested for memory leaks with valgrind, in 3D.
///
/// Typical run of this example: ./cfosls_hyperbolic --whichD 3 --spaceS L2 -no-vis
/// If you want to use the Hdiv-H1-L2 formulation, you will need not only change --spaceS option but also
/// change the source code, around 4.
///
/// Another examples of the same kind are cfosls_parabolic.cpp, cfosls_wave.cpp and cfosls_laplace.cpp.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>

// Adds regularization to the weighted mass matrix in Hdiv-L2 formulation
// The report on regularization though showed that in this particular case regularization
// doesn't help
//#define REGULARIZE_A

// Activates a special code block where eigenvalues of the weighted mass matrix from
// Hdiv-L2 formulation are computed
//#define EIGENVALUE_STUDY

using namespace std;
using namespace mfem;
using std::shared_ptr;
using std::make_shared;


// Some Operator inheriting classes used for analyzing the preconditioner

// class for Op_new = beta * Identity  + gamma * Op
class MyAXPYOperator : public Operator
{
private:
    Operator & op;
    double beta;
    double gamma;
public:
    virtual ~MyAXPYOperator() {}
    MyAXPYOperator(Operator& Op, double Beta = 0.0, double Gamma = 1.0)
        : Operator(Op.Height(),Op.Width()), op(Op), beta(Beta), gamma(Gamma) {}

    // Operator application
    void Mult(const Vector& x, Vector& y) const;
};

// Computes y = beta * x + gamma * Op  * x
void MyAXPYOperator::Mult(const Vector& x, Vector& y) const
{
    op.Mult(x, y);
    y *= gamma; // y = gamma * Op * x

    Vector tmp(x.Size());
    tmp = x;
    tmp *= beta; // tmp = beta * x

    y += tmp;    // y +=  beta * x, finally
}

// class for Op_new = scale * Op
class MyScaledOperator : public Operator
{
private:
    Operator& op;
    double scale;
public:
    virtual ~MyScaledOperator() {}
    MyScaledOperator(Operator& Op, double Scale = 1.0)
        : Operator(Op.Height(),Op.Width()), op(Op), scale(Scale) {}

    // Operator application
    void Mult(const Vector& x, Vector& y) const;

};

// Computes y = scale * Op  * x
void MyScaledOperator::Mult(const Vector& x, Vector& y) const
{
    op.Mult(x, y);
    y *= scale;
}

class MyOperator : public Operator
{
private:
    HypreParMatrix & leftmat;
    HypreParMatrix & rightmat;
    Operator & middleop;
    //int inner_niter;
public:
    virtual ~MyOperator() {}
    // Constructor
    MyOperator(HypreParMatrix& LeftMatrix, Operator& MiddleOp, HypreParMatrix& RightMatrix/*, int Inner_NIter = 1*/)
        : Operator(LeftMatrix.Height(),RightMatrix.Width()),
          leftmat(LeftMatrix), rightmat(RightMatrix), middleop(MiddleOp)//,
          //inner_niter(Inner_NIter)
    {}

    // Operator application
    void Mult(const Vector& x, Vector& y) const;
};

// Computes y = leftmat * middleop * rightmat * x
void MyOperator::Mult(const Vector& x, Vector& y) const
{
    Vector tmp1(rightmat.Height());
    rightmat.Mult(x, tmp1);
    Vector tmp2(leftmat.Width());

    middleop.Mult(tmp1, tmp2);
    leftmat.Mult(tmp2, y);
}

int main(int argc, char *argv[])
{
    // 1. Initialize MPI
    int num_procs, myid;

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    bool verbose = (myid == 0);

    bool visualization = 1;

    int nDimensions     = 3;
    int numsol          = -3;

    int ser_ref_levels  = 1;
    int par_ref_levels  = 0;

    const char *formulation = "cfosls"; // "cfosls" or "fosls" (switch on/off constraint)
    const char *space_for_S = "L2";     // "H1" or "L2"
    const char *space_for_sigma = "Hdiv"; // "Hdiv" or "H1"
    // in case space_for_S = "L2" defines whether we eliminate S from the system
    bool eliminateS = false;
    // in case space_for_S = "L2" defines whether we keep div-div term in the system
    bool keep_divdiv = false;

    // solver options
    int prec_option = 1; //defines whether to use preconditioner or not, and which one
    bool use_ADS = false;
    double rtol = 1e-12;//1e-7;//1e-9;
    double atol = 1e-14;//1e-9;//1e-12;

    const char *mesh_file = "../data/cube_3d_moderate.mesh";
    //const char *mesh_file = "../data/square_2d_moderate.mesh";

    //const char *mesh_file = "../data/cube4d_low.MFEM";
    //const char *mesh_file = "../data/cube4d.MFEM";
    //const char *mesh_file = "../data/orthotope3D_moderate.mesh";
    //const char *mesh_file = "../data/sphere3D_0.1to0.2.mesh";
    //const char *mesh_file = "../data/orthotope3D_fine.mesh";

    int feorder         = 0;

    if (verbose)
        cout << "Solving (С)FOSLS Transport equation \n";

    // 2. Parse command-line options.

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

    mesh_file = "../data/cube_3d_moderate.mesh";

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
        std::cout << "use_ADS = " << use_ADS << "\n";

    MFEM_ASSERT(strcmp(formulation,"cfosls") == 0 || strcmp(formulation,"fosls") == 0,
                "Formulation must be cfosls or fosls!\n");
    MFEM_ASSERT(strcmp(space_for_S,"H1") == 0 || strcmp(space_for_S,"L2") == 0,
                "Space for S must be H1 or L2!\n");
    MFEM_ASSERT(strcmp(space_for_sigma,"Hdiv") == 0 || strcmp(space_for_sigma,"H1") == 0,
                "Space for sigma must be Hdiv or H1!\n");

    MFEM_ASSERT(!strcmp(space_for_sigma,"H1") == 0 || (strcmp(space_for_sigma,"H1") == 0
                                                       && strcmp(space_for_S,"H1") == 0),
                "Sigma from H1vec must be coupled with S from H1!\n");
    MFEM_ASSERT(!strcmp(space_for_sigma,"H1") == 0 || (strcmp(space_for_sigma,"H1") == 0
                                                       && use_ADS == false),
                "ADS cannot be used when sigma is from H1vec!\n");
    MFEM_ASSERT(!(strcmp(formulation,"fosls") == 0 && strcmp(space_for_S,"L2") == 0 && !keep_divdiv),
                "For FOSLS formulation with S from L2 div-div term must be present!\n");
    MFEM_ASSERT(!(strcmp(formulation,"cfosls") == 0 && strcmp(space_for_S,"H1") == 0 && keep_divdiv),
                "For CFOSLS formulation with S from H1 div-div term must not be present for sigma!\n");

    if (verbose)
        std::cout << "Number of mpi processes: " << num_procs << "\n";

    StopWatch chrono;

    // 4. Reading the mesh and performing a prescribed number of serial and parallel refinements
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

    for (int l = 0; l < par_ref_levels; l++)
       pmesh->UniformRefinement();

    // for the cylinder test in the cube, we dilate the mesh so that it covers [-1,1]^d x [0,1]
    if (numsol == 8)
    {
        Vector vert_coos;
        pmesh->GetVertices(vert_coos);
        int nv = pmesh->GetNV();
        for (int vind = 0; vind < nv; ++vind)
        {
            for (int j = 0; j < nDimensions; ++j)
            {
                if (j < nDimensions - 1) // shift only in space
                {
                    // translation by -0.5 in space variables
                    vert_coos(j*nv + vind) -= 0.5;
                    // dilation so that the resulting mesh covers [-1,1] ^d in space
                    vert_coos(j*nv + vind) *= 2.0;
                }
                // dilation in time so that final time interval is [0,2]
                if (j == nDimensions - 1)
                    vert_coos(j*nv + vind) *= 2.0;
            }
        }
        pmesh->SetVertices(vert_coos);
    }

    //if(dim==3) pmesh->ReorientTetMesh();

    pmesh->PrintInfo(std::cout); if(verbose) cout << endl;

    // 5. Define a parallel finite element space on the parallel mesh. Here we
    //    use the Raviart-Thomas finite elements of the specified order.
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

    // 6. Define the two BlockStructure of the problem.  block_offsets is used
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

    Hyper_test Mytest(nDimensions,numsol);

    ParGridFunction *S_exact = new ParGridFunction(S_space);
    S_exact->ProjectCoefficient(*Mytest.GetU());

    ParGridFunction * sigma_exact = new ParGridFunction(Sigma_space);
    sigma_exact->ProjectCoefficient(*Mytest.GetSigma());

    x.GetBlock(0) = *sigma_exact;
    x.GetBlock(1) = *S_exact;

   // 7. Setting boundary conditions (attirbutes)

   Array<int> ess_bdrS(pmesh->bdr_attributes.Max());
   Array<int> ess_bdrSigma(pmesh->bdr_attributes.Max());


   ess_bdrS = 0;
   ess_bdrSigma = 0;

   // setting specially boundary attributes for cylinder test in the cube
   // for now, we make the problem overconstrained, see more comments around
   // OVERCONSTRAINED macro in the cfosls_hyperbolic_adref_Hcurl_new.cpp
   if (numsol == 8)
   {
       MFEM_ASSERT(pmesh->bdr_attributes.Max() == 6, "");
       if (strcmp(space_for_S,"H1") == 0)
       {
           ess_bdrS = 1;
           ess_bdrS[5] = 0;
       }
       else // L2 case
       {
           ess_bdrSigma = 1;
           ess_bdrSigma[5] = 0;
       }

   }
   else
   {
       if (strcmp(space_for_S,"H1") == 0)
           ess_bdrS[0] = 1; // t = 0
       if (strcmp(space_for_S,"L2") == 0) // S is from L2, so we impose bdr condition for sigma at t = 0
       {
           ess_bdrSigma[0] = 1;
       }
   }


   if (verbose)
   {
       std::cout << "Boundary conditions: \n";
       std::cout << "ess bdr Sigma: \n";
       ess_bdrSigma.Print(std::cout, pmesh->bdr_attributes.Max());
       std::cout << "ess bdr S: \n";
       ess_bdrS.Print(std::cout, pmesh->bdr_attributes.Max());
   }

   // 8. Define the parallel grid function and parallel linear forms, solution
   //    vector and rhs.

   ConstantCoefficient zero(.0);

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
       //if (strcmp(space_for_sigma,"Hdiv") == 0 )
           qform->AddDomainIntegrator(new GradDomainLFIntegrator(*Mytest.GetBf()));
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

   // 9. Assemble the finite element matrices for the CFOSLS operator

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

   delete Ablock;

#ifdef EIGENVALUE_STUDY
   if (verbose)
       std::cout << "Studying eigenvalues of (weighted) mass matrix which is A block \n";

   {
       Array<double> eigenvalues;
       int nev = 20;
       int seed = 75;
       int maxiter = 600;
       double eigtol = 1.0e-8;
       {

           HypreLOBPCG * lobpcg = new HypreLOBPCG(MPI_COMM_WORLD);

           lobpcg->SetNumModes(nev);
           lobpcg->SetRandomSeed(seed);
           lobpcg->SetMaxIter(maxiter);
           lobpcg->SetTol(eigtol);
           lobpcg->SetPrintLevel(1);
           // checking for A
           lobpcg->SetOperator(*A);

           // 4. Compute the eigenmodes and extract the array of eigenvalues. Define a
           //    parallel grid function to represent each of the eigenmodes returned by
           //    the solver.
           lobpcg->Solve();
           lobpcg->GetEigenvalues(eigenvalues);

           std::cout << "The computed minimal eigenvalues for M are: \n";
           if (verbose)
               eigenvalues.Print();
       }

       double beta = eigenvalues[0] * 10000.0; // should be enough
       Operator * revA_op = new MyAXPYOperator(*A, beta, -1.0);
       {
           HypreLOBPCG * lobpcg2 = new HypreLOBPCG(MPI_COMM_WORLD);

           lobpcg2->SetNumModes(nev);
           lobpcg2->SetRandomSeed(seed);
           lobpcg2->SetMaxIter(maxiter);
           lobpcg2->SetTol(eigtol*1.0e-4);
           lobpcg2->SetPrintLevel(1);
           // checking for beta * Id - A
           lobpcg2->SetOperator(*revA_op);

           // 4. Compute the eigenmodes and extract the array of eigenvalues. Define a
           //    parallel grid function to represent each of the eigenmodes returned by
           //    the solver.
           lobpcg2->Solve();
           lobpcg2->GetEigenvalues(eigenvalues);

           std::cout << "The computed maximal eigenvalues for M are: \n";
           for ( int i = 0; i < nev; ++i)
               eigenvalues[i] = beta - eigenvalues[i];
           if (verbose)
                eigenvalues.Print();
       }

    }
    delete revA_op;
    delete lobpcg;

    MPI_Finalize();
    return 0;

#endif

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
           if (strcmp(space_for_sigma,"Hdiv") == 0)
                Cblock->AddDomainIntegrator(new DiffusionIntegrator(*Mytest.GetBBt()));
       }
       else // "L2" & !eliminateS
       {
           Cblock->AddDomainIntegrator(new MassIntegrator(*Mytest.GetBtB()));
       }
       Cblock->Assemble();
       Cblock->EliminateEssentialBC(ess_bdrS, x.GetBlock(1), *qform);
       Cblock->Finalize();
       C = Cblock->ParallelAssemble();

       delete Cblock;
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

       delete Bblock;
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

      delete Dblock;
   }

   // 10. Setting up the linear system to be solved and a preconditioner for it's matrix
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

   CFOSLSop->owns_blocks = true;

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
   if (strcmp(formulation,"cfosls") == 0 && prec_option > 0 )
   {
      HypreParMatrix *AinvDt = D->Transpose();
      HypreParVector *Ad = new HypreParVector(MPI_COMM_WORLD, A->GetGlobalNumRows(),
                                           A->GetRowStarts());
      A->GetDiag(*Ad);
      AinvDt->InvScaleRows(*Ad);
      Schur = ParMult(D, AinvDt);
      Schur->CopyColStarts();
      Schur->CopyRowStarts();

      delete AinvDt;
      delete Ad;
   }

   Solver * invA;
   if (prec_option > 0)
   {
       if (use_ADS)
           invA = new HypreADS(*A, Sigma_space);
       else // using Diag(A);
           invA = new HypreDiagScale(*A);
   }

   invA->iterative_mode = false;

   Solver * invC;
   if (strcmp(space_for_S,"H1") == 0 && prec_option > 0) // S is from H1
   {
       invC = new HypreBoomerAMG(*C);
       ((HypreBoomerAMG*)invC)->SetPrintLevel(0);
       ((HypreBoomerAMG*)invC)->iterative_mode = false;
   }
   else // S from L2
   {
       if (!eliminateS && prec_option > 0) // S is from L2 and not eliminated
       {
           invC = new HypreDiagScale(*C);
           ((HypreDiagScale*)invC)->iterative_mode = false;
       }
   }

   Solver * invS;
   if (strcmp(formulation,"cfosls") == 0 && prec_option > 0 )
   {
        invS = new HypreBoomerAMG(*Schur);
        ((HypreBoomerAMG *)invS)->SetPrintLevel(0);
        ((HypreBoomerAMG *)invS)->iterative_mode = false;
   }

   BlockDiagonalPreconditioner * prec = new BlockDiagonalPreconditioner(block_trueOffsets);
   if (prec_option > 0)
   {
       tempblknum = 0;
       prec->SetDiagonalBlock(tempblknum, invA);
       tempblknum++;
       if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
       {
           prec->SetDiagonalBlock(tempblknum, invC);
           tempblknum++;
       }
       if (strcmp(formulation,"cfosls") == 0)
            prec->SetDiagonalBlock(tempblknum, invS);

       if (verbose)
           std::cout << "Preconditioner built in " << chrono.RealTime() << "s. \n";

       prec->owns_blocks = true;
   }
   else
       if (verbose)
           cout << "No preconditioner is used. \n";

   // 11. Solve the linear system with MINRES.
   //     Check the norm of the unpreconditioned residual.

   if (verbose)
       std::cout << "Dofs for main unknowns + Lagrange multiplier = " << CFOSLSop->ColOffsets()[numblocks] << "\n";

   chrono.Clear();
   chrono.Start();
   MINRESSolver solver(MPI_COMM_WORLD);
   solver.SetAbsTol(atol);
   solver.SetRelTol(rtol);
   solver.SetMaxIter(4000);
   solver.SetOperator(*CFOSLSop);
   solver.iterative_mode = false;
   if (prec_option > 0)
        solver.SetPreconditioner(*prec);
   solver.SetPrintLevel(1);
   trueX = 0.0;

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

   ParGridFunction * sigma = new ParGridFunction(Sigma_space);
   sigma->Distribute(&(trueX.GetBlock(0)));

   ParGridFunction * S = new ParGridFunction(S_space);
   if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
       S->Distribute(&(trueX.GetBlock(1)));
   else // no S in the formulation
   {
       ParBilinearForm *Cblock(new ParBilinearForm(S_space));
       Cblock->AddDomainIntegrator(new MassIntegrator(*Mytest.GetBtB()));
       Cblock->Assemble();
       Cblock->Finalize();
       HypreParMatrix * C = Cblock->ParallelAssemble();

       ParMixedBilinearForm *Bblock(new ParMixedBilinearForm(Sigma_space, S_space));
       Bblock->AddDomainIntegrator(new VectorFEMassIntegrator(*Mytest.GetB()));
       Bblock->Assemble();
       Bblock->Finalize();
       HypreParMatrix * B = Bblock->ParallelAssemble();
       Vector bTsigma(C->Height());
       B->Mult(trueX.GetBlock(0),bTsigma);

       Vector trueS(C->Height());

       //(*C, bTsigma, trueS, 0, 5000, 1e-9, 1e-12);

       CGSolver cg;
       cg.SetPrintLevel(0);
       cg.SetMaxIter(5000);
       cg.SetRelTol(sqrt(1.0e-12));
       cg.SetAbsTol(sqrt(1.0e-15));
       cg.SetOperator(*C);
       cg.iterative_mode = false;

       cg.Mult(bTsigma, trueS);

       S->Distribute(trueS);

       delete Cblock;
       delete Bblock;
       delete B;
       delete C;
   }

   // 12. Extract the parallel grid function corresponding to the finite element
   //     approximation X. This is the local solution on each processor. Compute
   //     L2 error norms.

   int order_quad = max(2, 2*feorder+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }


   double err_sigma = sigma->ComputeL2Error(*Mytest.GetSigma(), irs);
   double norm_sigma = ComputeGlobalLpNorm(2, *Mytest.GetSigma(), *pmesh, irs);
   if (verbose)
       cout << "|| sigma - sigma_ex || / || sigma_ex || = " << err_sigma / norm_sigma << endl;

   DiscreteLinearOperator Div(Sigma_space, W_space);
   Div.AddDomainInterpolator(new DivergenceInterpolator());
   ParGridFunction DivSigma(W_space);
   Div.Assemble();
   Div.Mult(*sigma, DivSigma);

   double err_div = DivSigma.ComputeL2Error(*Mytest.GetRhs(),irs);
   double norm_div = ComputeGlobalLpNorm(2, *Mytest.GetRhs(), *pmesh, irs);

   if (verbose)
   {
       if (fabs(norm_div) > 1.0e-13)
           cout << "|| div (sigma_h - sigma_ex) || / ||div (sigma_ex)|| = "
                 << err_div/norm_div  << "\n";
       else
           cout << "|| div sigma_h || ( div (sigma_ex) == 0) = "
                 << err_div << "\n";
   }

   if (verbose)
   {
       cout << "Actually it will be ~ continuous L2 + discrete L2 for divergence" << endl;
       cout << "|| sigma_h - sigma_ex ||_Hdiv / || sigma_ex ||_Hdiv = "
                 << sqrt(err_sigma*err_sigma + err_div * err_div)/sqrt(norm_sigma*norm_sigma + norm_div * norm_div)  << "\n";
   }

   // Computing error for S

   double err_S = S->ComputeL2Error(*Mytest.GetU(), irs);
   double norm_S = ComputeGlobalLpNorm(2, *Mytest.GetU(), *pmesh, irs);
   if (verbose)
   {
       std::cout << "|| S_h - S_ex || / || S_ex || = " <<
                    err_S / norm_S << "\n";
   }

   if (strcmp(space_for_S,"H1") == 0) // S is from H1
   {
       FiniteElementCollection * hcurl_coll;
       if (dim == 4)
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

   // Check value of functional and mass conservation
   // This might be not fully correct, old code
   // The newer version of functional computation can be found in corresponding FOSLSProblem children
   // in cfosls/cfosls_tools.cpp, e.g., for Hdiv-L2 formulation look at ComputeFuncError() in
   // FOSLSProblem_HdivL2hyp class.
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
       massform.AddDomainIntegrator(new DomainLFIntegrator(*(Mytest.GetRhs())));
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

   if (verbose)
       cout << "Computing projection errors \n";

   double projection_error_sigma = sigma_exact->ComputeL2Error(*Mytest.GetSigma(), irs);

   if(verbose)
   {
       cout << "|| sigma_ex - Pi_h sigma_ex || / || sigma_ex || = "
                       << projection_error_sigma / norm_sigma << endl;
   }

   double projection_error_S = S_exact->ComputeL2Error(*Mytest.GetU(), irs);

   if(verbose)
       cout << "|| S_ex - Pi_h S_ex || / || S_ex || = "
                       << projection_error_S / norm_S << endl;


   // 13. Visualization (optional)
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

   // 14. Free the used memory.
   delete S_exact;
   delete sigma_exact;

   delete S;
   delete sigma;

   delete fform;
   if (strcmp(space_for_S,"H1") == 0 || !eliminateS)
       delete qform;
   delete gform;

   delete H_space;
   delete W_space;
   delete R_space;
   if (strcmp(space_for_sigma,"H1") == 0) // S was from H1
        delete H1vec_space;
   delete l2_coll;
   delete h1_coll;
   delete hdiv_coll;

   delete CFOSLSop;
   if (prec_option > 0)
       delete Schur;
   delete prec;

   MPI_Finalize();
   return 0;
}


