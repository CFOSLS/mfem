//
//                        MFEM CFOSLS Poisson equation
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>
#include <unistd.h>

#include "cfosls_testsuite.hpp"

//#define VISUALIZATION

// (de)activates solving of the discrete global problem
#define OLD_CODE

// activates constraint residual check after each iteration of the minimization solver
#define CHECK_CONSTR

#define CHECK_BNDCND

#define TIMING

#ifdef TIMING
#undef CHECK_LOCALSOLVE
#undef CHECK_CONSTR
#undef CHECK_BNDCND
#endif

#include "divfree_solver_tools.hpp"

#define MYZEROTOL (1.0e-13)

using namespace std;
using namespace mfem;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

class VectorcurlDomainLFIntegrator : public LinearFormIntegrator
{
    DenseMatrix curlshape;
    DenseMatrix curlshape_dFadj;
    DenseMatrix curlshape_dFT;
    DenseMatrix dF_curlshape;
    VectorCoefficient &VQ;
    int oa, ob;
public:
    /// Constructs a domain integrator with a given Coefficient
    VectorcurlDomainLFIntegrator(VectorCoefficient &VQF, int a = 2, int b = 0)
        : VQ(VQF), oa(a), ob(b) { }

    /// Constructs a domain integrator with a given Coefficient
    VectorcurlDomainLFIntegrator(VectorCoefficient &VQF, const IntegrationRule *ir)
        : LinearFormIntegrator(ir), VQ(VQF), oa(1), ob(1) { }

    /** Given a particular Finite Element and a transformation (Tr)
       computes the element right hand side element vector, elvect. */
    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect);

    using LinearFormIntegrator::AssembleRHSElementVect;
};

void VectorcurlDomainLFIntegrator::AssembleRHSElementVect(
        const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
    int dof = el.GetDof();

    int dim = el.GetDim();
    MFEM_ASSERT(dim == 3, "VectorcurlDomainLFIntegrator is working only in 3D currently \n");

    curlshape.SetSize(dof,3);           // matrix of size dof x 3, works only in 3D
    curlshape_dFadj.SetSize(dof,3);     // matrix of size dof x 3, works only in 3D
    curlshape_dFT.SetSize(dof,3);       // matrix of size dof x 3, works only in 3D
    dF_curlshape.SetSize(3,dof);        // matrix of size dof x 3, works only in 3D
    Vector vecval(3);
    //Vector vecval_new(3);
    //DenseMatrix invdfdx(3,3);

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        // ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob + Tr.OrderW());
        ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob);
        // int order = 2 * el.GetOrder() ; // <--- OK for RTk
        // ir = &IntRules.Get(el.GetGeomType(), order);
    }

    elvect.SetSize(dof);
    elvect = 0.0;

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        el.CalcCurlShape(ip, curlshape);

        Tr.SetIntPoint (&ip);

        VQ.Eval(vecval,Tr,ip);                  // plain evaluation

        MultABt(curlshape, Tr.Jacobian(), curlshape_dFT);

        curlshape_dFT.AddMult_a(ip.weight, vecval, elvect);
    }

}

class VectordivDomainLFIntegrator : public LinearFormIntegrator
{
    Vector divshape;
    Coefficient &Q;
    int oa, ob;
public:
    /// Constructs a domain integrator with a given Coefficient
    VectordivDomainLFIntegrator(Coefficient &QF, int a = 2, int b = 0)
    // the old default was a = 1, b = 1
    // for simple elliptic problems a = 2, b = -2 is ok
        : Q(QF), oa(a), ob(b) { }

    /// Constructs a domain integrator with a given Coefficient
    VectordivDomainLFIntegrator(Coefficient &QF, const IntegrationRule *ir)
        : LinearFormIntegrator(ir), Q(QF), oa(1), ob(1) { }

    /** Given a particular Finite Element and a transformation (Tr)
       computes the element right hand side element vector, elvect. */
    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect);

    using LinearFormIntegrator::AssembleRHSElementVect;
};
//---------

//------------------
void VectordivDomainLFIntegrator::AssembleRHSElementVect(
        const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)//don't need the matrix but the vector
{
    int dof = el.GetDof();

    divshape.SetSize(dof);       // vector of size dof
    elvect.SetSize(dof);
    elvect = 0.0;

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        // ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob + Tr.OrderW());
        ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob);
        // int order = 2 * el.GetOrder() ; // <--- OK for RTk
        // ir = &IntRules.Get(el.GetGeomType(), order);
    }

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        el.CalcDivShape(ip, divshape);

        Tr.SetIntPoint (&ip);
        //double val = Tr.Weight() * Q.Eval(Tr, ip);
        // Chak: Looking at how MFEM assembles in VectorFEDivergenceIntegrator, I think you dont need Tr.Weight() here
        // I think this is because the RT (or other vector FE) basis is scaled by the geometry of the mesh
        double val = Q.Eval(Tr, ip);

        add(elvect, ip.weight * val, divshape, elvect);
        //cout << "elvect = " << elvect << endl;
    }
}

class GradDomainLFIntegrator : public LinearFormIntegrator
{
    DenseMatrix dshape;
    DenseMatrix invdfdx;
    DenseMatrix dshapedxt;
    Vector bf;
    Vector bfdshapedxt;
    VectorCoefficient &Q;
    int oa, ob;
public:
    /// Constructs a domain integrator with a given Coefficient
    GradDomainLFIntegrator(VectorCoefficient &QF, int a = 2, int b = 0)
    // the old default was a = 1, b = 1
    // for simple elliptic problems a = 2, b = -2 is ok
        : Q(QF), oa(a), ob(b) { }

    /// Constructs a domain integrator with a given Coefficient
    GradDomainLFIntegrator(VectorCoefficient &QF, const IntegrationRule *ir)
        : LinearFormIntegrator(ir), Q(QF), oa(1), ob(1) { }

    /** Given a particular Finite Element and a transformation (Tr)
        computes the element right hand side element vector, elvect. */
    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect);

    using LinearFormIntegrator::AssembleRHSElementVect;
};

void GradDomainLFIntegrator::AssembleRHSElementVect(
        const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
    int dof = el.GetDof();
    int dim  = el.GetDim();

    dshape.SetSize(dof,dim);       // vector of size dof
    elvect.SetSize(dof);
    elvect = 0.0;

    invdfdx.SetSize(dim,dim);
    dshapedxt.SetSize(dof,dim);
    bf.SetSize(dim);
    bfdshapedxt.SetSize(dof);
    double w;

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        //       ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob
        //                          + Tr.OrderW());
        //      ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob);
        // int order = 2 * el.GetOrder() ; // <--- OK for RTk
        int order = (Tr.OrderW() + el.GetOrder() + el.GetOrder());
        ir = &IntRules.Get(el.GetGeomType(), order);
    }

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        el.CalcDShape(ip, dshape);

        //double val = Tr.Weight() * Q.Eval(Tr, ip);

        Tr.SetIntPoint (&ip);
        w = ip.weight;// * Tr.Weight();
        CalcAdjugate(Tr.Jacobian(), invdfdx);
        Mult(dshape, invdfdx, dshapedxt);

        Q.Eval(bf, Tr, ip);

        dshapedxt.Mult(bf, bfdshapedxt);

        add(elvect, w, bfdshapedxt, elvect);
    }
}

void curlhcurlFun3D_ex(const Vector& xt, Vector& vecvalue);
void hcurlFun3D_ex(const Vector& xt, Vector& vecvalue);

void hcurlFun3D_2_ex(const Vector& xt, Vector& vecvalue);
void curlhcurlFun3D_2_ex(const Vector& xt, Vector& vecvalue);

void zerovec_ex(const Vector& xt, Vector& vecvalue);
void zerovecMat4D_ex(const Vector& xt, Vector& vecvalue);

double freq = 1.0, kappa;

template<void (*Sfullgrad)(const Vector&, Vector& )>
void sigmaTemplate(const Vector& xt, Vector& sigma);

template<void (*Sfullgrad)(const Vector&, Vector& ),
         void (*opdivfreevec)(const Vector&, Vector& )>
void sigmahatTemplate(const Vector& xt, Vector& sigmahatv);

template<void (*Sfullgrad)(const Vector&, Vector& ),
         void (*opdivfreevec)(const Vector&, Vector& )>
void minsigmahatTemplate(const Vector& xt, Vector& sigmahatv);

template<double (*Slaplace)(const Vector & xt)> \
double divsigmaTemplate(const Vector& xt);

template<double (*S)(const Vector & xt) > double SnonhomoTemplate(const Vector& xt);

class Laplace_test_divfree
{
protected:
    int dim;
    int numsol;
    int numcurl;

public:
    FunctionCoefficient * scalarS;
    FunctionCoefficient * scalardivsigma;         // = - laplace(S) = div sigma = f
    VectorFunctionCoefficient * sigma;
    VectorFunctionCoefficient * sigmahat;         // sigma_hat = sigma_exact - op divfreepart (curl hcurlpart in 3D)
    VectorFunctionCoefficient * divfreepart;      // additional part added for testing div-free solver
    VectorFunctionCoefficient * opdivfreepart;    // curl of the additional part which is added to sigma_exact for testing div-free solver
    VectorFunctionCoefficient * minsigmahat;      // -sigma_hat
public:
    Laplace_test_divfree (int Dim, int NumSol, int NumCurl);

    int GetDim() {return dim;}
    int GetNumSol() {return numsol;}
    int GetNumCurl() {return numcurl;}
    void SetDim(int Dim) { dim = Dim;}
    void SetNumSol(int NumSol) { numsol = NumSol;}
    void SetNumCurl(int NumCurl) { numcurl = NumCurl;}
    bool CheckTestConfig();

    ~Laplace_test_divfree();
private:
    template<double (*S)(const Vector & xt), void(*Sfullgrad)(const Vector & xt, Vector & gradx), double (*Slaplace)(const Vector & xt),
             void(*divfreevec)(const Vector & x, Vector & vec), void(*opdivfreevec)(const Vector & x, Vector & vec)> \
    void SetTestCoeffs ( );

    void SetScalarSFun( double (*S)(const Vector & xt))
    { scalarS = new FunctionCoefficient(S);}

    template<void(*Sfullgrad)(const Vector & xt, Vector & vec)> \
    void SetSigmaVec()
    { sigma = new VectorFunctionCoefficient(dim, sigmaTemplate<Sfullgrad>); }

    template<double (*Slaplace)(const Vector & xt)> \
    void SetdivSigma()
    { scalardivsigma = new FunctionCoefficient(divsigmaTemplate<Slaplace>);}

    void SetDivfreePart( void(*divfreevec)(const Vector & x, Vector & vec))
    { divfreepart = new VectorFunctionCoefficient(dim, divfreevec);}

    void SetOpDivfreePart( void(*opdivfreevec)(const Vector & x, Vector & vec))
    { opdivfreepart = new VectorFunctionCoefficient(dim, opdivfreevec);}

    template<void (*Sfullgrad)(const Vector&xt, Vector& vec),
             void(*opdivfreevec)(const Vector & xt, Vector & vec)> \
    void Setsigmahat()
    { sigmahat = new VectorFunctionCoefficient(dim, sigmahatTemplate<Sfullgrad, opdivfreevec>);}

    template<void (*Sfullgrad)(const Vector& xt, Vector& vec ),
             void(*opdivfreevec)(const Vector & xt, Vector & vec)> \
    void Setminsigmahat()
    { minsigmahat = new VectorFunctionCoefficient(dim, minsigmahatTemplate<Sfullgrad, opdivfreevec>);}

};

template<double (*S)(const Vector & xt), void(*Sfullgrad)(const Vector & xt, Vector & gradx), double (*Slaplace)(const Vector & xt),
         void(*divfreevec)(const Vector & x, Vector & vec), void(*opdivfreevec)(const Vector & x, Vector & vec)> \
void Laplace_test_divfree::SetTestCoeffs ()
{
    SetScalarSFun(S);
    SetSigmaVec<Sfullgrad>();
    SetdivSigma<Slaplace>();
    SetDivfreePart(divfreevec);
    SetOpDivfreePart(opdivfreevec);
    Setsigmahat<Sfullgrad, opdivfreevec>();
    Setminsigmahat<Sfullgrad, opdivfreevec>();
    return;
}


bool Laplace_test_divfree::CheckTestConfig()
{
    if (dim == 4 || dim == 3)
    {
        if ( numsol == -3 && dim == 3 )
            return true;
        if ( numsol == -4 && dim == 4 )
            return true;
        return false;
    }
    else
        return false;

}

Laplace_test_divfree::~Laplace_test_divfree()
{
    delete scalarS;
    delete scalardivsigma;
    delete sigma;
    delete sigmahat;
    delete divfreepart;
    delete opdivfreepart;
    delete minsigmahat;
}

Laplace_test_divfree::Laplace_test_divfree (int Dim, int NumSol, int NumCurl)
{
    dim = Dim;
    numsol = NumSol;
    numcurl = NumCurl;

    if ( CheckTestConfig() == false )
        std::cout << "Inconsistent dim = " << dim << " and numsol = " << numsol <<  std::endl << std::flush;
    else
    {
        if (numsol == -3)
        {
            if (numcurl == 1)
                SetTestCoeffs<&uFunTestLap_ex, &uFunTestLap_grad, &uFunTestLap_lap, &hcurlFun3D_ex, &curlhcurlFun3D_ex>();
            else if (numcurl == 2)
                SetTestCoeffs<&uFunTestLap_ex, &uFunTestLap_grad, &uFunTestLap_lap, &hcurlFun3D_2_ex, &curlhcurlFun3D_2_ex>();
            else
                SetTestCoeffs<&uFunTestLap_ex, &uFunTestLap_grad, &uFunTestLap_lap, &zerovec_ex, &zerovec_ex>();
        }
        if (numsol == -4)
        {
            if (numcurl == 1 || numcurl == 2)
            {
                std::cout << "Critical error: Explicit analytic div-free guy is not implemented in 4D \n";
            }
            else
                SetTestCoeffs<&uFunTestLap_ex, &uFunTestLap_grad, &uFunTestLap_lap, &zerovecMat4D_ex, &zerovec_ex>();
        }
    } // end of setting test coefficients in correct case
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
    int numcurl         = 0;

    int ser_ref_levels  = 1;
    int par_ref_levels  = 3;

    const char *space_for_S = "H1";    // "H1" or "L2"
    bool eliminateS = true;            // in case space_for_S = "L2" defines whether we eliminate S from the system

    bool aniso_refine = false;
    bool refine_t_first = false;

    // solver options
    int prec_option = 1;        // defines whether to use preconditioner or not, and which one

    //const char *mesh_file = "../data/cube_3d_fine.mesh";
    const char *mesh_file = "../data/cube_3d_moderate.mesh";
    //const char *mesh_file = "../data/square_2d_moderate.mesh";

    //const char *mesh_file = "../data/cube4d_low.MFEM";
    //const char *mesh_file = "../data/cube4d_96.MFEM";
    //const char *mesh_file = "dsadsad";
    //const char *mesh_file = "../data/orthotope3D_moderate.mesh";
    //const char * mesh_file = "../data/orthotope3D_fine.mesh";

    int feorder         = 0;

    kappa = freq * M_PI;

    if (verbose)
        cout << "Solving CFOSLS Poisson equation with MFEM & hypre \n";

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

    Laplace_test_divfree Mytest(nDimensions, numsol, numcurl);

    ConstantCoefficient zerocoeff(0.0);
    Vector zerovec(nDimensions);
    zerovec = 0.0;
    VectorConstantCoefficient zerocoeff_vec(zerovec);

    if (verbose)
        cout << "Number of mpi processes: " << num_procs << endl << flush;

    bool with_prec;

    switch (prec_option)
    {
    case 1: // smth simple like AMS
        with_prec = true;
        break;
    default: // no preconditioner (default)
        with_prec = false;
        break;
    }

    if (verbose)
    {
        cout << "with_prec = " << with_prec << endl;
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

    for (int l = 0; l < par_ref_levels; l++)
    {
       pmesh->UniformRefinement();
    }

    pmesh->PrintInfo(std::cout); if(verbose) cout << endl;

    int dim = nDimensions;

    Array<int> ess_bdrSigma(pmesh->bdr_attributes.Max());
    ess_bdrSigma = 0;

    Array<int> ess_bdrS(pmesh->bdr_attributes.Max());
    ess_bdrS = 1;

    Array<int> all_bdrSigma(pmesh->bdr_attributes.Max());
    all_bdrSigma = 1;

    Array<int> all_bdrS(pmesh->bdr_attributes.Max());
    all_bdrS = 1;

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

    ParGridFunction * sigma_exact_finest;
    sigma_exact_finest = new ParGridFunction(R_space);
    sigma_exact_finest->ProjectCoefficient(*Mytest.sigma);
    Vector sigma_exact_truedofs(R_space->GetTrueVSize());
    sigma_exact_finest->ParallelProject(sigma_exact_truedofs);

    ParGridFunction * S_exact_finest;
    Vector S_exact_truedofs;
    S_exact_finest = new ParGridFunction(S_space);
    S_exact_finest->ProjectCoefficient(*Mytest.scalarS);
    S_exact_truedofs.SetSize(S_space->GetTrueVSize());
    S_exact_finest->ParallelProject(S_exact_truedofs);


    chrono.Clear();
    chrono.Start();

    int numblocks = 2;

    Array<int> block_offsets(numblocks + 2); // number of variables + 1
    block_offsets[0] = 0;
    block_offsets[1] = R_space->GetVSize();
    block_offsets[2] = S_space->GetVSize();
    block_offsets[3] = W_space->GetVSize();
    block_offsets.PartialSum();

    Array<int> block_trueOffsets(numblocks + 2); // number of variables + 1
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = R_space->TrueVSize();
    block_trueOffsets[2] = S_space->TrueVSize();
    block_trueOffsets[3] = W_space->TrueVSize();
    block_trueOffsets.PartialSum();

    HYPRE_Int dimR = R_space->GlobalTrueVSize();
    HYPRE_Int dimS = S_space->GlobalTrueVSize();
    HYPRE_Int dimW = W_space->GlobalTrueVSize();
    if (verbose)
    {
        std::cout << "***********************************************************\n";
        std::cout << "dim(R) = " << dimR << ", ";
        std::cout << "dim(S) = " << dimS << ", ";
        std::cout << "dim(W) = " << dimW << "\n";
        std::cout << "neqns in the funct = " << dimR + dimS << "\n";
        std::cout << "neqns in the constr = " << dimW << "\n";
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

    ParGridFunction *S_exact;
    S_exact = new ParGridFunction(S_space);
    S_exact->ProjectCoefficient(*Mytest.scalarS);

    ParGridFunction * sigma_exact = new ParGridFunction(R_space);
    sigma_exact->ProjectCoefficient(*Mytest.sigma);

    xblks.GetBlock(0) = *sigma_exact;
    xblks.GetBlock(1) = *S_exact;

    ParLinearForm * Constrrhsform = new ParLinearForm(W_space);
    Constrrhsform->AddDomainIntegrator(new DomainLFIntegrator(*Mytest.scalardivsigma));
    Constrrhsform->Assemble();

    ParLinearForm * Sigmarhsform = new ParLinearForm(R_space);
    Sigmarhsform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(zerocoeff_vec));
    Sigmarhsform->Assemble();

    ParLinearForm * Srhsform = new ParLinearForm(S_space);
    Srhsform->AddDomainIntegrator(new DomainLFIntegrator(zerocoeff));
    Srhsform->Assemble();

    // mass matrix for H(div)
    ParBilinearForm *Mblock(new ParBilinearForm(R_space));
    Mblock->AddDomainIntegrator(new VectorFEMassIntegrator);
    Mblock->Assemble();
    Mblock->EliminateEssentialBC(ess_bdrSigma, *sigma_exact, *Sigmarhsform);
    Mblock->Finalize();
    HypreParMatrix *M = Mblock->ParallelAssemble();

    HypreParMatrix *C, *B, *BT;
    // diagonal block for H^1
    ParBilinearForm *Cblock = new ParBilinearForm(S_space);
    Cblock->AddDomainIntegrator(new DiffusionIntegrator);
    Cblock->Assemble();
    Cblock->EliminateEssentialBC(ess_bdrS, xblks.GetBlock(1),*Srhsform);
    Cblock->Finalize();
    C = Cblock->ParallelAssemble();

    // off-diagonal block for (H(div), Space_for_S) block
    ParMixedBilinearForm *Bblock(new ParMixedBilinearForm(S_space, R_space));
    Bblock->AddDomainIntegrator(new MixedVectorGradientIntegrator);
    Bblock->Assemble();
    Bblock->EliminateTrialDofs(ess_bdrS, *S_exact, *Sigmarhsform);
    Bblock->EliminateTestDofs(ess_bdrSigma);
    Bblock->Finalize();
    B = Bblock->ParallelAssemble();
    BT = B->Transpose();

    HypreParMatrix * Constr, * ConstrT;
    {
        ParMixedBilinearForm *Dblock = new ParMixedBilinearForm(R_space, W_space);
        Dblock->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
        Dblock->Assemble();
        Dblock->Finalize();
        Constr = Dblock->ParallelAssemble();
        ConstrT = Constr->Transpose();
    }

    Sigmarhsform->ParallelAssemble(trueRhs.GetBlock(0));
    Srhsform->ParallelAssemble(trueRhs.GetBlock(1));
    Constrrhsform->ParallelAssemble(trueRhs.GetBlock(2));

    BlockOperator *MainOp = new BlockOperator(block_trueOffsets);

    // setting block operator of the system
    MainOp->SetBlock(0,0, M);
    if (strcmp(space_for_S,"H1") == 0) // S is present
    {
        MainOp->SetBlock(0,1, B);
        MainOp->SetBlock(1,0, BT);
        MainOp->SetBlock(1,1, C);
    }
    MainOp->SetBlock(0,2, ConstrT);
    MainOp->SetBlock(2,0, Constr);

    // testing
    Array<int> blockfunct_trueOffsets(numblocks + 1);
    blockfunct_trueOffsets[0] = 0;
    blockfunct_trueOffsets[1] = M->Height();
    blockfunct_trueOffsets[2] = C->Height();
    blockfunct_trueOffsets.PartialSum();

    BlockOperator *MainOpFunct = new BlockOperator(blockfunct_trueOffsets);

    // setting block operator of the system
    MainOpFunct->SetBlock(0,0, M);
    MainOpFunct->SetBlock(0,1, B);
    MainOpFunct->SetBlock(1,0, BT);
    MainOpFunct->SetBlock(1,1, C);

    BlockVector truesol(blockfunct_trueOffsets);
    truesol.GetBlock(0) = sigma_exact_truedofs;
    truesol.GetBlock(1) = S_exact_truedofs;

    BlockVector rhsfunct(blockfunct_trueOffsets);
    rhsfunct.GetBlock(0) = trueRhs.GetBlock(0);
    rhsfunct.GetBlock(1) = trueRhs.GetBlock(1);

    BlockVector resfunct(blockfunct_trueOffsets);
    MainOpFunct->Mult(truesol, resfunct);
    resfunct -= rhsfunct;

    double res_funct_norm = resfunct.Norml2() / sqrt (resfunct.Size());

    if (verbose)
        std::cout << "res_funct_norm = " << res_funct_norm << "\n";


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


    Vector res_Constr(Constr->Height());
    Constr->Mult(sigma_exact_truedofs, res_Constr);
    res_Constr -= trueRhs.GetBlock(2);

    double res_Constr_norm = res_Constr.Norml2() / sqrt (res_Constr.Size());

    if (verbose)
        std::cout << "res_Constr_norm = " << res_Constr_norm << "\n";

    chrono.Stop();
    if (verbose)
        cout<<"Discretized problem is assembled in "<< chrono.RealTime() <<" seconds.\n";
    chrono.Clear();
    chrono.Start();

    Solver *prec;
    prec = new BlockDiagonalPreconditioner(block_trueOffsets);

    HypreParMatrix *Schur;
    HypreParMatrix *MinvDt = Constr->Transpose();
    HypreParVector *Md = new HypreParVector(MPI_COMM_WORLD, M->GetGlobalNumRows(),
                                         M->GetRowStarts());
    M->GetDiag(*Md);
    MinvDt->InvScaleRows(*Md);
    Schur = ParMult(Constr, MinvDt);

    HypreBoomerAMG * precSchur = new HypreBoomerAMG(*Schur);
    precSchur->SetPrintLevel(0);
    precSchur->iterative_mode = false;

    HypreDiagScale * precSigma = new HypreDiagScale(*M);
    precSigma->iterative_mode = false;

    HypreBoomerAMG * precS = new HypreBoomerAMG(*C);
    precS->SetPrintLevel(0);
    precS->iterative_mode = false;


    ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(0, precSigma);
    ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(1, precS);
    ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(2, precSchur);

    chrono.Stop();
    if (verbose)
        std::cout << "Preconditioner was created in "<< chrono.RealTime() <<" seconds.\n";

    MINRESSolver solver(comm);
    if (verbose)
        cout << "Linear solver: MINRES \n";

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

    {
        BlockVector sol(blockfunct_trueOffsets);
        sol.GetBlock(0) = trueX.GetBlock(0);
        sol.GetBlock(1) = trueX.GetBlock(1);

        BlockVector rhsfunct(blockfunct_trueOffsets);
        rhsfunct.GetBlock(0) = trueRhs.GetBlock(0);
        rhsfunct.GetBlock(1) = trueRhs.GetBlock(1);

        BlockVector resfunct(blockfunct_trueOffsets);
        MainOpFunct->Mult(sol, resfunct);
        resfunct -= rhsfunct;

        double res_funct_norm = resfunct.Norml2() / sqrt (resfunct.Size());

        if (verbose)
            std::cout << "res_funct_norm for sol = " << res_funct_norm << "\n";

        Vector Msigma(M->Height());
        M->Mult(trueX.GetBlock(0), Msigma);
        Vector BS(B->Height());
        B->Mult(trueX.GetBlock(1), BS);

        Vector res_Hdiv(M->Height());
        res_Hdiv = Msigma;
        res_Hdiv += BS;

        double res_Hdiv_norm = res_Hdiv.Norml2() / sqrt (res_Hdiv.Size());

        if (verbose)
            std::cout << "res_Hdiv_norm for sol = " << res_Hdiv_norm << "\n";

        // checking the residual for the projections of exact solution for H1 equation
        Vector CS(C->Height());
        C->Mult(trueX.GetBlock(1), CS);
        Vector BTsigma(BT->Height());
        BT->Mult(trueX.GetBlock(0), BTsigma);

        Vector res_H1(C->Height());
        res_H1 = CS;
        res_H1 += BTsigma;

        double res_H1_norm = res_H1.Norml2() / sqrt (res_H1.Size());

        if (verbose)
            std::cout << "res_H1_norm for sol = " << res_H1_norm << "\n";
    }

    Vector trueerr_sigma(trueX.GetBlock(0).Size());
    trueerr_sigma = sigma_exact_truedofs;
    trueerr_sigma -= trueX.GetBlock(0);

    double trueerr_sigma_norm = trueerr_sigma.Norml2() / sqrt (trueerr_sigma.Size());
    double truesigma_norm = sigma_exact_truedofs.Norml2() / sqrt (sigma_exact_truedofs.Size());

    double trueerr_sigma_relnorm = trueerr_sigma_norm /  truesigma_norm;

    if (verbose)
    {
        std::cout << "true err sigma norm = " << trueerr_sigma_norm << "\n";
        std::cout << "true err sigma rel norm = " << trueerr_sigma_relnorm << "\n";
    }

    ParGridFunction * S = new ParGridFunction(H_space);
    S->Distribute(&(trueX.GetBlock(1)));

    ParGridFunction * sigma = new ParGridFunction(R_space);
    sigma->Distribute(&(trueX.GetBlock(0)));

    // 13. Extract the parallel grid function corresponding to the finite element
    //     approximation X. This is the local solution on each processor. Compute
    //     L2 error norms.

    int order_quad = max(2, 2*feorder+1);
    const IntegrationRule *irs[Geometry::NumGeom];
    for (int i=0; i < Geometry::NumGeom; ++i)
    {
        irs[i] = &(IntRules.Get(i, order_quad));
    }

    double err_sigma = sigma->ComputeL2Error(*Mytest.sigma, irs);
    double norm_sigma = ComputeGlobalLpNorm(2, *Mytest.sigma, *pmesh, irs);

    if (verbose)
        cout << "sigma_h = sigma_hat + div-free part, div-free part = curl u_h \n";

    if (verbose)
    {
        if ( norm_sigma > MYZEROTOL )
            cout << "|| sigma_h - sigma_ex || / || sigma_ex || = " << err_sigma / norm_sigma << endl;
        else
            cout << "|| sigma || = " << err_sigma << " (sigma_ex = 0)" << endl;
    }

    DiscreteLinearOperator Div(R_space, W_space);
    Div.AddDomainInterpolator(new DivergenceInterpolator());
    ParGridFunction DivSigma(W_space);
    Div.Assemble();
    Div.EliminateTestDofs(ess_bdrSigma);
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
        std::cout << "Errors were computed in "<< chrono.RealTime() <<" seconds.\n";

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

    MPI_Finalize();
    return 0;

    chrono.Clear();
    chrono.Start();

#ifdef VISUALIZATION
    if (visualization && nDimensions < 4)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;

        if (!withDiv)
        {
            socketstream uex_sock(vishost, visport);
            uex_sock << "parallel " << num_procs << " " << myid << "\n";
            uex_sock.precision(8);
            MPI_Barrier(pmesh->GetComm());
            uex_sock << "solution\n" << *pmesh << *u_exact << "window_title 'u_exact'"
                   << endl;

            socketstream uh_sock(vishost, visport);
            uh_sock << "parallel " << num_procs << " " << myid << "\n";
            uh_sock.precision(8);
            MPI_Barrier(pmesh->GetComm());
            uh_sock << "solution\n" << *pmesh << *u << "window_title 'u_h'"
                   << endl;

            *u -= *u_exact;
            socketstream udiff_sock(vishost, visport);
            udiff_sock << "parallel " << num_procs << " " << myid << "\n";
            udiff_sock.precision(8);
            MPI_Barrier(pmesh->GetComm());
            udiff_sock << "solution\n" << *pmesh << *u << "window_title 'u_h - u_exact'"
                   << endl;

            socketstream opdivfreepartex_sock(vishost, visport);
            opdivfreepartex_sock << "parallel " << num_procs << " " << myid << "\n";
            opdivfreepartex_sock.precision(8);
            MPI_Barrier(pmesh->GetComm());
            opdivfreepartex_sock << "solution\n" << *pmesh << *opdivfreepart_exact << "window_title 'curl u_exact'"
                   << endl;

            socketstream opdivfreepart_sock(vishost, visport);
            opdivfreepart_sock << "parallel " << num_procs << " " << myid << "\n";
            opdivfreepart_sock.precision(8);
            MPI_Barrier(pmesh->GetComm());
            opdivfreepart_sock << "solution\n" << *pmesh << *opdivfreepart << "window_title 'curl u_h'"
                   << endl;

            *opdivfreepart -= *opdivfreepart_exact;
            socketstream opdivfreepartdiff_sock(vishost, visport);
            opdivfreepartdiff_sock << "parallel " << num_procs << " " << myid << "\n";
            opdivfreepartdiff_sock.precision(8);
            MPI_Barrier(pmesh->GetComm());
            opdivfreepartdiff_sock << "solution\n" << *pmesh << *opdivfreepart << "window_title 'curl u_h - curl u_exact'"
                   << endl;
        }

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


template <void (*Sfullgrad)(const Vector&, Vector& )> \
void sigmaTemplate(const Vector& xt, Vector& sigma)
{
    sigma.SetSize(xt.Size());
    Sfullgrad(xt, sigma);
    sigma *= -1;
    return;
}


template<double (*Slaplace)(const Vector & xt)> \
double divsigmaTemplate(const Vector& xt)
{
    return (-1) * Slaplace(xt);
}

template <void (*Sfullgrad)(const Vector&, Vector& ),
          void (*opdivfreevec)(const Vector&, Vector& )> \
void sigmahatTemplate(const Vector& xt, Vector& sigmahatv)
{
    Vector fullgrad;
    Sfullgrad(xt,fullgrad);

    Vector opdivfree;
    opdivfreevec(xt, opdivfree);

    sigmahatv = 0.0;
    sigmahatv -= opdivfree;
#ifndef ONLY_DIVFREEPART
    sigmahatv += fullgrad;
#endif
    return;
}

template <void (*Sfullgrad)(const Vector&, Vector& ),
          void (*opdivfreevec)(const Vector&, Vector& )> \
void minsigmahatTemplate(const Vector& xt, Vector& minsigmahatv)
{
    minsigmahatv.SetSize(xt.Size());
    sigmahatTemplate<Sfullgrad, opdivfreevec>(xt, minsigmahatv);
    minsigmahatv *= -1;

    return;
}

////////////////
void hcurlFun3D_ex(const Vector& xt, Vector& vecvalue)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    vecvalue.SetSize(xt.Size());

    //vecvalue(0) = -y * (1 - t);
    //vecvalue(1) = x * (1 - t);
    //vecvalue(2) = 0;
    //vecvalue(0) = x * (1 - x);
    //vecvalue(1) = y * (1 - y);
    //vecvalue(2) = t * (1 - t);

    // Martin's function
    vecvalue(0) = sin(kappa * y);
    vecvalue(1) = sin(kappa * t);
    vecvalue(2) = sin(kappa * x);

    return;
}

void curlhcurlFun3D_ex(const Vector& xt, Vector& vecvalue)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    vecvalue.SetSize(xt.Size());

    //vecvalue(0) = 0.0;
    //vecvalue(1) = 0.0;
    //vecvalue(2) = -2.0 * (1 - t);

    // Martin's function's curl
    vecvalue(0) = - kappa * cos(kappa * t);
    vecvalue(1) = - kappa * cos(kappa * x);
    vecvalue(2) = - kappa * cos(kappa * y);

    return;
}

////////////////
void hcurlFun3D_2_ex(const Vector& xt, Vector& vecvalue)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    vecvalue.SetSize(xt.Size());

    //
    vecvalue(0) = 100.0 * x * x * (1-x) * (1-x) * y * y * (1-y) * (1-y) * t * t * (1-t) * (1-t);
    vecvalue(1) = 0.0;
    vecvalue(2) = 0.0;

    return;
}

void curlhcurlFun3D_2_ex(const Vector& xt, Vector& vecvalue)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    vecvalue.SetSize(xt.Size());

    //
    vecvalue(0) = 0.0;
    vecvalue(1) = 100.0 * ( 2.0) * t * (1-t) * (1.-2.*t) * x * x * (1-x) * (1-x) * y * y * (1-y) * (1-y);
    vecvalue(2) = 100.0 * (-2.0) * y * (1-y) * (1.-2.*y) * x * x * (1-x) * (1-x) * t * t * (1-t) * (1-t);

    return;
}

void zerovec_ex(const Vector& xt, Vector& vecvalue)
{
    vecvalue.SetSize(xt.Size());
    vecvalue = 0.0;
    return;
}

void zerovecMat4D_ex(const Vector& xt, Vector& vecvalue)
{
    vecvalue.SetSize(6);
    vecvalue = 0.0;
    return;
}

