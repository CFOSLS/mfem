//
//                                MFEM CFOSLS Wave equation (+ mesh generator) solved by hypre
//
// Compile with: make
//
// Description:
//               This example code solves a simple 4D Wave equation over [0,1]^d, d=3,4
//               written in first-order formulation
//                                  sigma_1 + grad u   = 0
//                                  sigma_2 - u_t      = 0
//                                  div_(x,t) sigma    = f
//                       with boundary conditions:
//                                   u(x,t)            = 0      on the spatial boundary
//                                   u(x,0)            = u0     (initial condition on u)
//                                   du/dt(x,0)        = ut0    (initial condition on du/dt)
//               Here, we use a given exact solution
//                                  u(xt) = uFun_ex(xt)
//               and compute the corresponding r.h.s.
//               We discretize with Raviart-Thomas finite elements (sigma), continuous H1 elements (u) and
//		 discontinuous polynomials (mu) for the lagrange multiplier.
//               Solver: ~ hypre with a block-diagonal preconditioner with BoomerAMG
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>

#include"cfosls_testsuite.hpp"

#define MYZEROTOL (1.0e-13)

using namespace std;
using namespace mfem;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

//********* NEW STUFF FOR 4D Wave CFOSLS
//-----------------------

class WaveVectorFEIntegratorB: public BilinearFormIntegrator
{
private:
    Coefficient *Q;
    VectorCoefficient *VQ;
    MatrixCoefficient *MQ;
    void Init(Coefficient *q, VectorCoefficient *vq, MatrixCoefficient *mq)
    { Q = q; VQ = vq; MQ = mq; }

#ifndef MFEM_THREAD_SAFE
    Vector shape;
    Vector D;
    Vector trial_shape;
    Vector test_shape;//<<<<<<<
    DenseMatrix K;
    DenseMatrix test_vshape;
    DenseMatrix trial_vshape;
    DenseMatrix trial_dshape;//<<<<<<<<<<<<<<
    DenseMatrix test_dshape;//<<<<<<<<<<<<<<

#endif

public:
    WaveVectorFEIntegratorB() { Init(NULL, NULL, NULL); }
    WaveVectorFEIntegratorB(Coefficient *_q) { Init(_q, NULL, NULL); }
    WaveVectorFEIntegratorB(Coefficient &q) { Init(&q, NULL, NULL); }
    WaveVectorFEIntegratorB(VectorCoefficient *_vq) { Init(NULL, _vq, NULL); }
    WaveVectorFEIntegratorB(VectorCoefficient &vq) { Init(NULL, &vq, NULL); }
    WaveVectorFEIntegratorB(MatrixCoefficient *_mq) { Init(NULL, NULL, _mq); }
    WaveVectorFEIntegratorB(MatrixCoefficient &mq) { Init(NULL, NULL, &mq); }

    virtual void AssembleElementMatrix(const FiniteElement &el,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
    virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                        const FiniteElement &test_fe,
                                        ElementTransformation &Trans,
                                        DenseMatrix &elmat);
};

//=-=-=-=--=-=-=-=-=-=-=-=-=
/// Integrator for (Q u, v) for VectorFiniteElements
class WaveVectorFEIntegrator: public BilinearFormIntegrator
{
private:
    Coefficient *Q;
    VectorCoefficient *VQ;
    MatrixCoefficient *MQ;
    void Init(Coefficient *q, VectorCoefficient *vq, MatrixCoefficient *mq)
    { Q = q; VQ = vq; MQ = mq; }

#ifndef MFEM_THREAD_SAFE
    Vector shape;
    Vector D;
    Vector trial_shape;
    Vector test_shape;//<<<<<<<
    DenseMatrix K;
    DenseMatrix test_vshape;
    DenseMatrix trial_vshape;
    DenseMatrix trial_dshape;//<<<<<<<<<<<<<<
    DenseMatrix test_dshape;//<<<<<<<<<<<<<<
    DenseMatrix dshape;
    DenseMatrix dshapedxt;
    DenseMatrix invdfdx;

#endif

public:
    WaveVectorFEIntegrator() { Init(NULL, NULL, NULL); }
    WaveVectorFEIntegrator(Coefficient *_q) { Init(_q, NULL, NULL); }
    WaveVectorFEIntegrator(Coefficient &q) { Init(&q, NULL, NULL); }
    WaveVectorFEIntegrator(VectorCoefficient *_vq) { Init(NULL, _vq, NULL); }
    WaveVectorFEIntegrator(VectorCoefficient &vq) { Init(NULL, &vq, NULL); }
    WaveVectorFEIntegrator(MatrixCoefficient *_mq) { Init(NULL, NULL, _mq); }
    WaveVectorFEIntegrator(MatrixCoefficient &mq) { Init(NULL, NULL, &mq); }

    virtual void AssembleElementMatrix(const FiniteElement &el,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
    virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                        const FiniteElement &test_fe,
                                        ElementTransformation &Trans,
                                        DenseMatrix &elmat);
};

//=-=-=-=-=-=-=-=-=-=-=-=-=-
void WaveVectorFEIntegratorB::AssembleElementMatrix(
        const FiniteElement &el,
        ElementTransformation &Trans,
        DenseMatrix &elmat)
{}

void WaveVectorFEIntegratorB::AssembleElementMatrix2(
        const FiniteElement &trial_fe, const FiniteElement &test_fe,
        ElementTransformation &Trans, DenseMatrix &elmat)
{
    // assume both test_fe and trial_fe are vector FE
    int dim  = test_fe.GetDim();
    int trial_dof = trial_fe.GetDof();
    int test_dof = test_fe.GetDof();
    double w;

    if (VQ || MQ) // || = or
        mfem_error("WaveVectorFEIntegratorB::AssembleElementMatrix2(...)\n"
                   "   is not implemented for vector/tensor permeability");

    DenseMatrix trial_dshapedxt(trial_dof,dim);
    DenseMatrix invdfdx(dim,dim);

#ifdef MFEM_THREAD_SAFE
    // DenseMatrix trial_vshape(trial_dof, dim);
    Vector trial_shape(trial_dof); //PAULI
    DenseMatrix trial_dshape(trial_dof,dim);
    DenseMatrix test_vshape(test_dof,dim);
#else
    //trial_vshape.SetSize(trial_dof, dim);
    trial_shape.SetSize(trial_dof); //PAULI
    trial_dshape.SetSize(trial_dof,dim); //Pauli
    test_vshape.SetSize(test_dof,dim);
#endif
    //elmat.SetSize (test_dof, trial_dof);
    elmat.SetSize (test_dof, trial_dof);

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        int order = (Trans.OrderW() + test_fe.GetOrder() + trial_fe.GetOrder());
        ir = &IntRules.Get(test_fe.GetGeomType(), order);
    }

    elmat = 0.0;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);

        trial_fe.CalcShape(ip, trial_shape);
        trial_fe.CalcDShape(ip, trial_dshape);

        Trans.SetIntPoint (&ip);
        test_fe.CalcVShape(Trans, test_vshape);

        w = ip.weight * Trans.Weight();
        CalcInverse(Trans.Jacobian(), invdfdx);
        Mult(trial_dshape, invdfdx, trial_dshapedxt);
        if (Q)
        {
            w *= Q -> Eval (Trans, ip);
        }

        for (int j = 0; j < test_dof; j++)
        {
            for (int k = 0; k < trial_dof; k++)
            {
                for (int d = 0; d < dim - 1; d++ )
                    elmat(j, k) += 1.0 * w * test_vshape(j, d) * trial_dshapedxt(k, d);
                elmat(j, k) -= w * test_vshape(j, dim - 1) * trial_dshapedxt(k, dim - 1);
            }
        }
    }
}

void WaveVectorFEIntegrator::AssembleElementMatrix(
        const FiniteElement &el,
        ElementTransformation &Trans,
        DenseMatrix &elmat)
{
    int dof = el.GetDof();
    int dim  = el.GetDim();
    double w;

    if (VQ || MQ) // || = or
        mfem_error("WaveVectorFEIntegrator::AssembleElementMatrix2(...)\n"
                   "   is not implemented for vector/tensor permeability");

#ifdef MFEM_THREAD_SAFE
    Vector shape(dof);
    DenseMatrix dshape(dof,dim);
    DenseMatrix dshapedxt(dof,dim);
    DenseMatrix invdfdx(dim,dim);
#else
    shape.SetSize(dof);
    dshape.SetSize(dof,dim);
    dshapedxt.SetSize(dof,dim);
    invdfdx.SetSize(dim,dim);
#endif
    //elmat.SetSize (test_dof, trial_dof);
    elmat.SetSize (dof, dof);
    elmat = 0.0;

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        int order = (Trans.OrderW() + el.GetOrder() + el.GetOrder());
        ir = &IntRules.Get(el.GetGeomType(), order);
    }

    elmat = 0.0;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);

        //chak Trans.SetIntPoint (&ip);

        el.CalcShape(ip, shape);
        el.CalcDShape(ip, dshape);

        Trans.SetIntPoint (&ip);
        CalcInverse(Trans.Jacobian(), invdfdx);
        w = ip.weight * Trans.Weight();
        Mult(dshape, invdfdx, dshapedxt);

        if (Q)
        {
            w *= Q -> Eval (Trans, ip);
        }

        for (int j = 0; j < dof; j++)
            for (int k = 0; k < dof; k++)
            {
                for (int d = 0; d < dim; d++ )
                    elmat(j, k) +=  w * dshapedxt(j, d) * dshapedxt(k, d);
            }

    }
}

void WaveVectorFEIntegrator::AssembleElementMatrix2(
        const FiniteElement &trial_fe, const FiniteElement &test_fe,
        ElementTransformation &Trans, DenseMatrix &elmat)
{}

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

void VectordivDomainLFIntegrator::AssembleRHSElementVect(
        const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
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
        // Chak: Looking at how MFEM assembles in VectorFEDivergenceIntegrator,
        // I think you dont need Tr.Weight() here I think this is because the RT
        // (or other vector FE) basis is scaled by the geometry of the mesh
        double val = Q.Eval(Tr, ip);

        add(elvect, ip.weight * val, divshape, elvect);
    }

}



// Define the analytical solution and forcing terms / boundary conditions
//double u0_function(const Vector &x);
double uFun_ex(const Vector & x); // Exact Solution
double uFun_ex_dt(const Vector & xt);
double uFun_ex_dt2(const Vector & xt);
double uFun_ex_laplace(const Vector & xt);
double uFun_ex_dtlaplace(const Vector & xt);
void uFun_ex_gradx(const Vector& xt, Vector& gradx );
void uFun_ex_dtgradx(const Vector& xt, Vector& gradx );

double uFun1_ex(const Vector & x); // Exact Solution
double uFun1_ex_dt(const Vector & xt);
double uFun1_ex_dt2(const Vector & xt);
double uFun1_ex_laplace(const Vector & xt);
double uFun1_ex_dtlaplace(const Vector & xt);
void uFun1_ex_gradx(const Vector& xt, Vector& gradx );
void uFun1_ex_dtgradx(const Vector& xt, Vector& gradx );

double uFun2_ex(const Vector & x); // Exact Solution
double uFun2_ex_dt(const Vector & xt);
double uFun2_ex_dt2(const Vector & xt);
double uFun2_ex_laplace(const Vector & xt);
double uFun2_ex_dtlaplace(const Vector & xt);
void uFun2_ex_gradx(const Vector& xt, Vector& gradx );
void uFun2_ex_dtgradx(const Vector& xt, Vector& gradx );

double uFun3_ex(const Vector & x); // Exact Solution
double uFun3_ex_dt(const Vector & xt);
double uFun3_ex_dt2(const Vector & xt);
double uFun3_ex_laplace(const Vector & xt);
double uFun3_ex_dtlaplace(const Vector & xt);
void uFun3_ex_gradx(const Vector& xt, Vector& gradx );
void uFun3_ex_dtgradx(const Vector& xt, Vector& gradx );

double uFun4_ex(const Vector & x); // Exact Solution
double uFun4_ex_dt(const Vector & xt);
double uFun4_ex_dt2(const Vector & xt);
double uFun4_ex_laplace(const Vector & xt);
double uFun4_ex_dtlaplace(const Vector & xt);
void uFun4_ex_gradx(const Vector& xt, Vector& gradx );
void uFun4_ex_dtgradx(const Vector& xt, Vector& gradx );

double uFun5_ex(const Vector & x); // Exact Solution
double uFun5_ex_dt(const Vector & xt);
double uFun5_ex_dt2(const Vector & xt);
double uFun5_ex_laplace(const Vector & xt);
double uFun5_ex_dtlaplace(const Vector & xt);
void uFun5_ex_gradx(const Vector& xt, Vector& gradx );
void uFun5_ex_dtgradx(const Vector& xt, Vector& gradx );

template<double (*S)(const Vector & xt), double (*d2Sdt2)(const Vector & xt), \
         double (*Slaplace)(const Vector & xt), double (*dSdtlaplace)(const Vector & xt)> \
    double rhsideTemplate(const Vector& xt);

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt)> \
    double SnonhomoTemplate(const Vector& xt);

template<double (*S)(const Vector & xt), double (*d2Sdt2)(const Vector & xt), double (*Slaplace)(const Vector & xt)> \
    double divsigmaTemplate(const Vector& xt);


template <double (*dSdt)(const Vector&), void(*Sgradxvec)(const Vector & x, Vector & gradx) >
    void sigmaTemplate(const Vector& xt, Vector& sigma);

template <double (*dSdt)(const Vector&), void(*Sgradxvec)(const Vector & x, Vector & gradx), void (*dSdtgradxvec)(const Vector&, Vector& )> \
    void sigmaNonHomoTemplate(const Vector& xt, Vector& sigma);


class Wave_test
{
protected:
    int dim;
    int numsol;

public:
    FunctionCoefficient * scalarS;             // S
    FunctionCoefficient * scalardivsigma;      // = d2 S/dt2 - laplace S                  - what is used for computing error
    VectorFunctionCoefficient * sigma;
public:
    Wave_test (int Dim, int NumSol);

    int GetDim() {return dim;}
    int GetNumSol() {return numsol;}
    void SetDim(int Dim) { dim = Dim;}
    void SetNumSol(int NumSol) { numsol = NumSol;}
    bool CheckTestConfig();

    ~Wave_test () {}
private:
    void SetScalarSFun( double (*f)(const Vector & xt))
    { scalarS = new FunctionCoefficient(f);}

    template<double (*S)(const Vector & xt), double (*d2Sdt2)(const Vector & xt), double (*Slaplace)(const Vector & xt)> \
    void SetDivSigma()
    { scalardivsigma = new FunctionCoefficient(divsigmaTemplate<S, d2Sdt2, Slaplace>);}

    template<double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & vec)> \
    void SetSigmaVec()
    {
        sigma = new VectorFunctionCoefficient(dim, sigmaTemplate<dSdt,Sgradxvec>);
    }

    template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), double (*d2Sdt2)(const Vector & xt),\
             double (*Slaplace)(const Vector & xt), double (*dSdtlaplace)(const Vector & xt), \
             void(*Sgradxvec)(const Vector & x, Vector & gradx), void (*dSdtgradxvec)(const Vector&, Vector& ) > \
    void SetTestCoeffs ( );
};


template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), double (*d2Sdt2)(const Vector & xt), \
         double (*Slaplace)(const Vector & xt), double (*dSdtlaplace)(const Vector & xt), \
         void(*Sgradxvec)(const Vector & x, Vector & gradx), void (*dSdtgradxvec)(const Vector&, Vector& ) > \
void Wave_test::SetTestCoeffs ()
{
    SetScalarSFun(S);
    SetSigmaVec<dSdt,Sgradxvec>();
    SetDivSigma<S, d2Sdt2, Slaplace>();
    return;
}


bool Wave_test::CheckTestConfig()
{
    if (dim == 4 || dim == 3)
    {
        if (numsol == 0 || numsol == 1)
            return true;
        if (numsol == 2 && dim == 4)
            return true;
        if (numsol == 3 && dim == 3)
            return true;
        if (numsol == 4 && dim == 3)
            return true;
        if (numsol == 5 && dim == 3)
            return true;
        if (numsol == -34 && (dim == 3 || dim == 4))
            return true;
        return false;
    }
    else if (dim == 2)
    {
        if (numsol == -34)
            return true;
        return false;
    }
    else
        return false;

}

Wave_test::Wave_test (int Dim, int NumSol)
{
    dim = Dim;
    numsol = NumSol;

    if ( CheckTestConfig() == false )
        std::cout << "Inconsistent dim and numsol \n" << std::flush;
    else
    {
        if (numsol == -34)
        {
            SetTestCoeffs<&uFunTest_ex, &uFunTest_ex_dt, &uFunTest_ex_dt2, &uFunTest_ex_laplace, &uFunTest_ex_dtlaplace, &uFunTest_ex_gradx, &uFunTest_ex_dtgradx>();
        }
        if (numsol == 0)
        {
            SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_dt2, &uFun_ex_laplace, &uFun_ex_dtlaplace, &uFun_ex_gradx, &uFun_ex_dtgradx>();
        }
        if (numsol == 1)
        {
            SetTestCoeffs<&uFun1_ex, &uFun1_ex_dt, &uFun1_ex_dt2, &uFun1_ex_laplace, &uFun1_ex_dtlaplace, &uFun1_ex_gradx, &uFun1_ex_dtgradx>();
        }
        if (numsol == 2)
        {
            SetTestCoeffs<&uFun2_ex, &uFun2_ex_dt, &uFun2_ex_dt2, &uFun2_ex_laplace, &uFun2_ex_dtlaplace, &uFun2_ex_gradx, &uFun2_ex_dtgradx>();
        }
        if (numsol == 3)
        {
            SetTestCoeffs<&uFun3_ex, &uFun3_ex_dt, &uFun3_ex_dt2, &uFun3_ex_laplace, &uFun3_ex_dtlaplace, &uFun3_ex_gradx, &uFun3_ex_dtgradx>();
        }
        if (numsol == 4)
        {
            SetTestCoeffs<&uFun4_ex, &uFun4_ex_dt, &uFun4_ex_dt2, &uFun4_ex_laplace, &uFun4_ex_dtlaplace, &uFun4_ex_gradx, &uFun4_ex_dtgradx>();
        }
        if (numsol == 5)
        {
            SetTestCoeffs<&uFun5_ex, &uFun5_ex_dt, &uFun5_ex_dt2, &uFun5_ex_laplace, &uFun5_ex_dtlaplace, &uFun5_ex_gradx, &uFun5_ex_dtgradx>();
        }
    }
}


int main(int argc, char *argv[])
{
    StopWatch chrono;


    // 1. Initialize MPI.
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    bool verbose = (myid == 0);
    bool visualization = 0;

    int nDimensions     = 3;
    int numsol          = 3;

    int ser_ref_levels  = 1;
    int par_ref_levels  = 1;

    const char *formulation = "cfosls";      // "cfosls" or "fosls"
    bool with_divdiv = false;                // should be true for fosls and can be false for cfosls
    bool use_ADS = false;                    // works only in 3D and for with_divdiv = true

    const char *mesh_file = "../data/cube_3d_fine.mesh";
    //const char *mesh_file = "../data/square_2d_moderate.mesh";

    //const char *mesh_file = "../data/cube4d_low.MFEM";
    //const char *mesh_file = "../data/cube4d.MFEM";
    //const char *mesh_file = "../data/pmesh_cube_for_test.mesh";
    //const char *mesh_file = "../data/mesh4_saved";
    //const char *mesh_file = "dsadsad";
    //const char *mesh_file = "../build3/mesh_par1_id0_np_1.mesh";
    //const char *mesh_file = "../build3/mesh_par1_id0_np_2.mesh";
    //const char *mesh_file = "../data/tempmesh_frompmesh.mesh";
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
        cout << "Solving (C)FOSLS Wave equation with MFEM & hypre" << endl;

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
    args.AddOption(&formulation, "-form", "--formul",
                   "Formulation to use.");
    args.AddOption(&with_divdiv, "-divdiv", "--with-divdiv", "-no-divdiv",
                   "--no-divdiv",
                   "Decide whether div-div term is present.");
    args.AddOption(&use_ADS, "-ADS", "--with-ADS", "-no-ADS",
                   "--no-ADS",
                   "Decide whether to use ADS.");
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
        std::cout << "Running tests for the paper: \n";

    if (nDimensions == 3)
    {
        numsol = -34;
        mesh_file = "../data/cube_3d_moderate.mesh";
    }
    else if (nDimensions == 4)// 4D case
    {
        numsol = -34;
        mesh_file = "../data/cube4d_96.MFEM";
    }
    else
    {
        numsol = -34;
        mesh_file = "../data/square_2d_moderate.mesh";
    }

    if (verbose)
        std::cout << "For the records: numsol = " << numsol
                  << ", mesh_file = " << mesh_file << "\n";


    if (verbose)
        cout << "Number of mpi processes: " << num_procs << endl << flush;

    if ( ((strcmp(formulation,"cfosls") == 0 && (!with_divdiv)) || nDimensions != 3) && use_ADS)
    {
        if (verbose)
            cout << "ADS cannot be used if dim != 3 or if div-div term is absent" << endl;
        MPI_Finalize();
        return 0;
    }

    //DEFAULTED LINEAR SOLVER OPTIONS
    int max_num_iter = 150000;
    double rtol = 1e-12;
    double atol = 1e-14;

    Mesh *mesh = NULL;

    shared_ptr<ParMesh> pmesh;

    // not generating from a lower dimensional mesh
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

    int dim = pmesh->Dimension();

    //if(dim==3) pmesh->ReorientTetMesh();

    pmesh->PrintInfo(std::cout); if(verbose) cout << endl;

    // 6. Define a parallel finite element space on the parallel mesh. Here we
    //    use the Raviart-Thomas finite elements of the specified order.

    FiniteElementCollection *hdiv_coll, *h1_coll, *l2_coll;
    if (dim == 4)
    {
        hdiv_coll = new RT0_4DFECollection;
        if (verbose)cout << "RT: order 0 for 4D" << endl;
        if(feorder <= 1)
        {
            h1_coll = new LinearFECollection;
            if (verbose)cout << "H1: order 1 for 4D" << endl;
        }
        else
        {
            h1_coll = new QuadraticFECollection;
            if (verbose)cout << "H1: order 2 for 4D" << endl;
        }

        l2_coll = new L2_FECollection(0, dim);
        if (verbose)cout << "L2: order 0 for 4D" << endl;
    }
    else
    {
        hdiv_coll = new RT_FECollection(feorder, dim);
        if (verbose)cout << "RT: order " << feorder << " for 3D" << endl;
        h1_coll = new H1_FECollection(feorder+1, dim);
        if (verbose)cout << "H1: order " << feorder + 1 << " for 3D" << endl;
        // even in cfosls needed to estimate divergence
        l2_coll = new L2_FECollection(feorder, dim);
        if (verbose)cout << "L2: order " << feorder << " for 3D" << endl;
    }

    ParFiniteElementSpace *R_space = new ParFiniteElementSpace(pmesh.get(), hdiv_coll);
    ParFiniteElementSpace *H_space = new ParFiniteElementSpace(pmesh.get(), h1_coll);
    ParFiniteElementSpace *W_space = new ParFiniteElementSpace(pmesh.get(), l2_coll);

    HYPRE_Int dimR = R_space->GlobalTrueVSize();
    HYPRE_Int dimH = H_space->GlobalTrueVSize();
    HYPRE_Int dimW;
    if (strcmp(formulation,"cfosls") == 0)
        dimW = W_space->GlobalTrueVSize();

    if (verbose)
    {
        std::cout << "***********************************************************\n";
        std::cout << "dim(R) = " << dimR << "\n";
        std::cout << "dim(H) = " << dimH << "\n";
        if (strcmp(formulation,"cfosls") == 0)
        {
            std::cout << "dim(W) = " << dimW << "\n";
            std::cout << "dim(R+H+W) = " << dimR + dimH + dimW << "\n";
        }
        else // fosls
            std::cout << "dim(R+H) = " << dimR + dimH << "\n";
        std::cout << "***********************************************************\n";
    }

    // 7. Define the two BlockStructure of the problem.  block_offsets is used
    //    for Vector based on dof (like ParGridFunction or ParLinearForm),
    //    block_trueOffstes is used for Vector based on trueDof (HypreParVector
    //    for the rhs and solution of the linear system).  The offsets computed
    //    here are local to the processor.

    int numblocks = 2;
    if (strcmp(formulation,"cfosls") == 0)
        numblocks = 3;


    Array<int> block_offsets(numblocks + 1); // number of variables + 1
    block_offsets[0] = 0;
    block_offsets[1] = R_space->GetVSize();
    block_offsets[2] = H_space->GetVSize();
    if (strcmp(formulation,"cfosls") == 0)
        block_offsets[3] = W_space->GetVSize();
    block_offsets.PartialSum();

    Array<int> block_trueOffsets(numblocks + 1); // number of variables + 1
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = R_space->TrueVSize();
    block_trueOffsets[2] = H_space->TrueVSize();
    if (strcmp(formulation,"cfosls") == 0)
        block_trueOffsets[3] = W_space->TrueVSize();
    block_trueOffsets.PartialSum();


    // 8. Define the coefficients, analytical solution, and rhs of the PDE.

    Wave_test Mytest(nDimensions,numsol);

    ConstantCoefficient k(1.0);
    ConstantCoefficient zero(.0);
    Vector vzero(dim); vzero = 0.;
    VectorConstantCoefficient vzero_coeff(vzero);

    //----------------------------------------------------------
    // Setting boundary conditions.
    //----------------------------------------------------------

    Array<int> ess_bdrS(pmesh->bdr_attributes.Max());       // applied to H^1 variable
    ess_bdrS = 1;
    ess_bdrS[pmesh->bdr_attributes.Max()-1] = 0;
    Array<int> ess_bdrSigma(pmesh->bdr_attributes.Max());   // applied to Hdiv variable
    ess_bdrSigma = 0;
    ess_bdrSigma[0] = 1; // t = 0 = essential boundary for sigma from Hdiv = for dS/dt|t=0

    if (verbose)
    {
        std::cout << "Boundary conditions: \n";
        std::cout << "ess bdr Sigma: \n";
        ess_bdrSigma.Print(std::cout, pmesh->bdr_attributes.Max());
        std::cout << "ess bdr S: \n";
        ess_bdrS.Print(std::cout, pmesh->bdr_attributes.Max());
    }

    // 8.5 some additional parelag stuff which is used for coarse lagrange
    // multiplier implementation at the matrix level

    // 9. Define the parallel grid function and parallel linear forms, solution
    //    vector and rhs.

    BlockVector * x, * rhs;
    x = new BlockVector(block_offsets);
    *x = 0.0;
    rhs = new BlockVector(block_offsets);
    *rhs = 0.0;

    ParGridFunction *S_exact = new ParGridFunction(H_space);
    S_exact->ProjectCoefficient(*(Mytest.scalarS));

    ParGridFunction * sigma_exact = new ParGridFunction(R_space);
    sigma_exact->ProjectCoefficient(*(Mytest.sigma));

    x->GetBlock(0) = *sigma_exact;
    x->GetBlock(1) = *S_exact;

    ParLinearForm *fform(new ParLinearForm);
    fform->Update(R_space, rhs->GetBlock(0), 0);
    if (strcmp(formulation,"cfosls") == 0) // cfosls case
        if (with_divdiv)
        {
            if (verbose)
                cout << "Adding div-driven rhside term to the formulation" << endl;
            fform->AddDomainIntegrator(new VectordivDomainLFIntegrator(*(Mytest.scalardivsigma)));
        }
        else
        {
            if (verbose)
                cout << "No div-driven rhside term in the formulation" << endl;
            fform->AddDomainIntegrator(new VectordivDomainLFIntegrator(zero));
        }
    else // if fosls, then we need righthand side term here
    {
        if (verbose)
            cout << "Adding div-driven rhside term to the formulation" << endl;
        fform->AddDomainIntegrator(new VectordivDomainLFIntegrator(*(Mytest.scalardivsigma)));
    }
    fform->Assemble();

    ParLinearForm *qform(new ParLinearForm);
    qform->Update(H_space, rhs->GetBlock(1), 0);
    qform->AddDomainIntegrator(new VectorDomainLFIntegrator(vzero_coeff));
    qform->Assemble();

    ParLinearForm *gform(new ParLinearForm);
    if (strcmp(formulation,"cfosls") == 0)
    {
        gform->Update(W_space, rhs->GetBlock(2), 0);
        gform->AddDomainIntegrator(new DomainLFIntegrator(*(Mytest.scalardivsigma)));
        gform->Assemble();
    }


    // 10. Assemble the finite element matrices for the operator
    //
    //                       CFOSLS = [  A   B  D^T ]
    //                                [ B^T  C   0  ]
    //                                [  D   0   0  ]
    //     where:
    //
    //     A = ( sigma, tau)_{H(div)}
    //     B = (sigma, [ dx(S), -dtS] )
    //     C = ( [dx(S), -dtS], [dx(V),-dtV] )
    //     D = ( div(sigma), mu )

    chrono.Clear();
    chrono.Start();

    //---------------
    //  A Block:
    //---------------

    ParBilinearForm *Ablock(new ParBilinearForm(R_space));
    HypreParMatrix *A;

    Ablock->AddDomainIntegrator(new VectorFEMassIntegrator);
    if (strcmp(formulation,"cfosls") != 0) // fosls, then we need div-div term
    {
        if (verbose)
            cout << "Adding div-div term to the formulation" << endl;
        Ablock->AddDomainIntegrator(new DivDivIntegrator());
    }
    else // cfosls case
        if (with_divdiv)
        {
            if (verbose)
                cout << "Adding div-div term to the formulation" << endl;
            Ablock->AddDomainIntegrator(new DivDivIntegrator());
        }
        else
        {
            if (verbose)
                cout << "No div-div term in the formulation" << endl;
        }
    Ablock->Assemble();
    Ablock->EliminateEssentialBC(ess_bdrSigma, x->GetBlock(0), rhs->GetBlock(0)); // new
    Ablock->Finalize();
    A = Ablock->ParallelAssemble();

    //---------------
    //  C Block:
    //---------------

    ParBilinearForm *Cblock(new ParBilinearForm(H_space));
    HypreParMatrix *C;
    Cblock->AddDomainIntegrator(new WaveVectorFEIntegrator);
    Cblock->Assemble();
    Cblock->EliminateEssentialBC(ess_bdrS, x->GetBlock(1), rhs->GetBlock(1));
    Cblock->Finalize();
    C = Cblock->ParallelAssemble();

    //---------------
    //  B Block:
    //---------------

    ParMixedBilinearForm *Bblock(new ParMixedBilinearForm(H_space, R_space));
    HypreParMatrix *B;
    Bblock->AddDomainIntegrator(new WaveVectorFEIntegratorB);
    Bblock->Assemble();
    Bblock->EliminateTestDofs(ess_bdrSigma); // new
    Bblock->EliminateTrialDofs(ess_bdrS, x->GetBlock(1), rhs->GetBlock(0));
    Bblock->Finalize();
    B = Bblock->ParallelAssemble();
    HypreParMatrix *BT = B->Transpose();

    //----------------
    //  D Block:
    //-----------------

    HypreParMatrix *D;
    HypreParMatrix *DT;

    if (strcmp(formulation,"cfosls") == 0)
    {
        ParMixedBilinearForm *Dblock(new ParMixedBilinearForm(R_space, W_space));
        Dblock->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
        Dblock->Assemble();
        Dblock->EliminateTestDofs(ess_bdrSigma); // new
        Dblock->Finalize();
        D = Dblock->ParallelAssemble();
        DT = D->Transpose();
    }

    //=======================================================
    // Assembling the Matrix
    //-------------------------------------------------------

    BlockOperator *CFOSLSop;

    CFOSLSop = new BlockOperator(block_trueOffsets);

    CFOSLSop->SetBlock(0,0, A);
    CFOSLSop->SetBlock(0,1, B);
    CFOSLSop->SetBlock(1,0, BT);
    CFOSLSop->SetBlock(1,1, C);
    if (strcmp(formulation,"cfosls") == 0)
    {
        CFOSLSop->SetBlock(0,2, DT);
        CFOSLSop->SetBlock(2,0, D);
    }


    if (verbose)
        std::cout << "System built in " << chrono.RealTime() << "s. \n";

    // 11. Construct the operators for preconditioner
    //
    //                 P = [ diag(M)         0         ]
    //                     [  0       B diag(M)^-1 B^T ]
    //
    //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
    //     pressure Schur Complement.
    if (verbose)
    {
        if (use_ADS == true)
            cout << "Using ADS (+ I) preconditioner for sigma (and lagrange multiplier)" << endl;
        else
            cout << "Using Diag(A) (and D Diag^(-1)(A) Dt) preconditioner for sigma (and lagrange multiplier)" << endl;
    }

    chrono.Clear();
    chrono.Start();
    Solver * invA;
    HypreParMatrix *DAinvDt;
    if (use_ADS == false)
    {
        if (strcmp(formulation,"cfosls") == 0)
        {
            HypreParMatrix *AinvDt = D->Transpose();
            HypreParVector *Ad = new HypreParVector(MPI_COMM_WORLD, A->GetGlobalNumRows(),
                                                    A->GetRowStarts());
            A->GetDiag(*Ad);

            AinvDt->InvScaleRows(*Ad);
            DAinvDt= ParMult(D, AinvDt);
        }
        invA = new HypreDiagScale(*A);
    }
    else // use_ADS
    {
        invA = new HypreADS(*A, R_space);
    }
    invA->iterative_mode = false;

    Operator * invL;
    if (strcmp(formulation,"cfosls") == 0)
    {
        if (use_ADS == false)
        {
            invL= new HypreBoomerAMG(*DAinvDt);
            ((HypreBoomerAMG *)invL)->SetPrintLevel(0);
            ((HypreBoomerAMG *)invL)->iterative_mode = false;
        }
        else // use_ADS
        {
            invL = new IdentityOperator(D->Height());
        }
    }


    if (verbose)
        cout << "Using boomerAMG for scalar unknown S" << endl;
    HypreBoomerAMG * invC = new HypreBoomerAMG(*C);
    invC->SetPrintLevel(0);

    invC->iterative_mode = false;

    BlockDiagonalPreconditioner * prec;

    prec = new BlockDiagonalPreconditioner(block_trueOffsets);
    //BlockDiagonalPreconditioner prec(block_trueOffsets);
    prec->SetDiagonalBlock(0, invA);
    prec->SetDiagonalBlock(1, invC);
    if (strcmp(formulation,"cfosls") == 0)
        prec->SetDiagonalBlock(2, invL);

    if (verbose)
        std::cout << "Preconditioner built in " << chrono.RealTime() << "s. \n";

    // 12. Solve the linear system with MINRES.
    //     Check the norm of the unpreconditioned residual.


    chrono.Clear();
    chrono.Start();
    MINRESSolver solver(MPI_COMM_WORLD);
    solver.SetAbsTol(atol);
    solver.SetRelTol(rtol);
    solver.SetMaxIter(max_num_iter);
    solver.SetOperator(*CFOSLSop);
    solver.SetPreconditioner(*prec);
    solver.SetPrintLevel(0);
    BlockVector * trueX;

    trueX = new BlockVector(block_trueOffsets);
    *trueX = 0.0;

    BlockVector *trueRhs;

    trueRhs = new BlockVector(block_trueOffsets);
    *trueRhs=.0;

    fform->ParallelAssemble(trueRhs->GetBlock(0));
    qform->ParallelAssemble(trueRhs->GetBlock(1));
    if (strcmp(formulation,"cfosls") == 0)
    {
        gform->ParallelAssemble(trueRhs->GetBlock(2));
    }

    solver.Mult(*trueRhs, *trueX);
    chrono.Stop();

    if (verbose)
    {
       if (solver.GetConverged())
          cout << "MINRES converged in " << solver.GetNumIterations()
                    << " iterations with a residual norm of " << solver.GetFinalNorm() << ".\n";
       else
          cout << "MINRES did not converge in " << solver.GetNumIterations()
                    << " iterations. Residual norm is " << solver.GetFinalNorm() << ".\n";
       cout << "MINRES solver took " << chrono.RealTime() << "s. \n";
    }

    // Checking the residual in the divergence constraint
    Vector vec1 = trueX->GetBlock(0);
    Vector Dvec1(trueRhs->GetBlock(2).Size());
    D->Mult(vec1, Dvec1);
    Dvec1 -= trueRhs->GetBlock(2);
    double local_res_norm = Dvec1.Norml2();
    double global_res_norm = 0.0;
    MPI_Reduce(&local_res_norm, &global_res_norm, 1,
               MPI_DOUBLE, MPI_SUM, 0, comm);
    double local_rhs_norm = trueRhs->GetBlock(2).Norml2();
    double global_rhs_norm = 0.0;
    MPI_Reduce(&local_rhs_norm, &global_rhs_norm, 1,
               MPI_DOUBLE, MPI_SUM, 0, comm);
    if (verbose)
    {
        cout << "rel res_norm for the conservation law = " << global_res_norm / global_rhs_norm << endl;
    }


    ParGridFunction *S(new ParGridFunction);
    S->MakeRef(H_space, x->GetBlock(1), 0);
    S->Distribute(&(trueX->GetBlock(1)));

    ParGridFunction *sigma(new ParGridFunction);
    sigma->MakeRef(R_space, x->GetBlock(0), 0);
    sigma->Distribute(&(trueX->GetBlock(0)));

    // 13. Extract the parallel grid function corresponding to the finite element
    //     approximation X. This is the local solution on each processor. Compute
    //     L2 error norms.

    int order_quad = max(2, 2*feorder+1);
    const IntegrationRule *irs[Geometry::NumGeom];
    for (int i=0; i < Geometry::NumGeom; ++i)
        irs[i] = &(IntRules.Get(i, order_quad));


    double err_sigma = sigma->ComputeL2Error(*(Mytest.sigma), irs);
    double norm_sigma = ComputeGlobalLpNorm(2, *(Mytest.sigma), *pmesh, irs);

    if (verbose)
    {
        //cout << "|| sigma_h - sigma_ex ||  = "
        //          << err_sigma  << "\n";
        cout << "|| sigma_h - sigma_ex || / || sigma_ex || = "
                  << err_sigma/norm_sigma  << "\n";
    }

    ParDiscreteLinearOperator Div(R_space, W_space);
    Div.AddDomainInterpolator(new DivergenceInterpolator());
    Div.Assemble();
    Div.Finalize();
    Div.ParallelAssemble();

    ParGridFunction DivSigma(W_space);
    Div.Mult(*sigma, DivSigma);

    ParGridFunction DivSigma_exact(W_space);
    DivSigma_exact.ProjectCoefficient(*(Mytest.scalardivsigma));

    double err_div = DivSigma.ComputeL2Error(*(Mytest.scalardivsigma),irs);
    double norm_div = ComputeGlobalLpNorm(2, *(Mytest.scalardivsigma), *pmesh, irs);

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
    double err_S  = S->ComputeL2Error(*(Mytest.scalarS), irs);
    double norm_S = ComputeGlobalLpNorm(2, *(Mytest.scalarS), *pmesh, irs);

    if (verbose)
    {
        //cout << "|| S_h - S_ex || = "
        //          << err_S << "\n";
        cout << "|| S_h - S_ex || / || S_ex || = "
                  << err_S/norm_S  << "\n";
    }

    ParFiniteElementSpace * GradSpace;
    FiniteElementCollection *hcurl_coll;
    if (dim == 4)
    {
        hcurl_coll = new ND1_4DFECollection;
        GradSpace = new ParFiniteElementSpace(pmesh.get(), hcurl_coll);
    }
    else
    {
        hcurl_coll = new ND_FECollection(feorder+1, dim);
        GradSpace = new ParFiniteElementSpace(pmesh.get(), hcurl_coll);
    }

    DiscreteLinearOperator Grad(H_space, GradSpace);
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

    // Check value of functional and mass conservation
    {
        trueX->GetBlock(2) = 0.0;
        *trueRhs = 0.0;;
        CFOSLSop->Mult(*trueX, *trueRhs);
        double localFunctional = (*trueX)*(*trueRhs);
        double globalFunctional;
        MPI_Reduce(&localFunctional, &globalFunctional, 1,
                   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (verbose)
        {
            cout << "|| sigma_h - L(S_h) ||^2 = " << globalFunctional<< "\n";
            cout << "|| div_h sigma_h - f ||^2 = " << err_div*err_div  << "\n";
            cout << "|| f ||^2 = " << norm_div*norm_div  << "\n";
            cout << "Relative Energy Error = " << sqrt(globalFunctional+err_div*err_div)/norm_div<< "\n";
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

        trueRhs->GetBlock(2) -= massform;
        double mass_loss_loc = trueRhs->GetBlock(2).Norml1();
        double mass_loss;
        MPI_Reduce(&mass_loss_loc, &mass_loss, 1,
                   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (verbose)
            cout << "Sum of local mass loss = " << mass_loss<< "\n";
    }

    if (verbose)
        cout << "Computing projection errors \n";

    double projection_error_sigma = sigma_exact->ComputeL2Error(*(Mytest.sigma), irs);

    if(verbose)
    {
        if ( norm_sigma > MYZEROTOL )
        {
            cout << "|| sigma_ex - Pi_h sigma_ex || / || sigma_ex || = " << projection_error_sigma / norm_sigma << endl;
        }
        else
            cout << "|| Pi_h sigma_ex || = " << projection_error_sigma << " (sigma_ex = 0) \n ";
    }
    double projection_error_S = S_exact->ComputeL2Error(*(Mytest.scalarS), irs);

    if(verbose)
    {
       if ( norm_S > MYZEROTOL )
           cout << "|| S_ex - Pi_h S_ex || / || S_ex || = " << projection_error_S / norm_S << endl;
       else
           cout << "|| Pi_h S_ex ||  = " << projection_error_S << " (S_ex = 0) \n";
    }

    if (visualization)
    {
        // Make sure all ranks have sent their 'u' solution before initiating
        // another set of GLVis connections (one from each rank):
        char vishost[] = "localhost";
        int  visport   = 19916;
        socketstream uu_sock(vishost, visport);
        uu_sock << "parallel " << num_procs << " " << myid << "\n";
        uu_sock.precision(8);
        MPI_Barrier(pmesh->GetComm());
        uu_sock << "solution\n" << *pmesh << *sigma << "window_title 'sigma'"
                << endl;

        socketstream u_sock(vishost, visport);
        u_sock << "parallel " << num_procs << " " << myid << "\n";
        u_sock.precision(8);
        MPI_Barrier(pmesh->GetComm());
        u_sock << "solution\n" << *pmesh << *sigma_exact
               << "window_title 'sigma_exact'" << endl;

        *sigma_exact -= *sigma;
        socketstream uuu_sock(vishost, visport);
        uuu_sock << "parallel " << num_procs << " " << myid << "\n";
        uuu_sock.precision(8);
        MPI_Barrier(pmesh->GetComm());
        uuu_sock << "solution\n" << *pmesh << *sigma_exact
                 << "window_title 'difference for sigma'" << endl;

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


        socketstream ds_sock(vishost, visport);
        ds_sock << "parallel " << num_procs << " " << myid << "\n";
        ds_sock.precision(8);
        MPI_Barrier(pmesh->GetComm());
        ds_sock << "solution\n" << *pmesh << DivSigma << "window_title 'divsigma'"
                << endl;

        socketstream dse_sock(vishost, visport);
        dse_sock << "parallel " << num_procs << " " << myid << "\n";
        dse_sock.precision(8);
        MPI_Barrier(pmesh->GetComm());
        dse_sock << "solution\n" << *pmesh << DivSigma_exact << "window_title 'divsigma exact'"
                << endl;

        DivSigma -= DivSigma_exact;
        socketstream dsd_sock(vishost, visport);
        dsd_sock << "parallel " << num_procs << " " << myid << "\n";
        dsd_sock.precision(8);
        MPI_Barrier(pmesh->GetComm());
        dsd_sock << "solution\n" << *pmesh << DivSigma << "window_title 'divsigma error'"
                << endl;

    }

    // 17. Free the used memory.
    delete fform;
    delete gform;
    delete CFOSLSop;
    if (strcmp(formulation,"cfosls") == 0)
    {
        delete DT;
        delete D;
    }
    delete C;
    delete BT;
    delete B;
    delete A;

    delete Ablock;
    delete Bblock;
    delete Cblock;
    delete H_space;
    delete R_space;
    delete W_space;
    delete hdiv_coll;
    delete h1_coll;
    delete l2_coll;
    delete hcurl_coll;
    delete GradSpace;

    MPI_Finalize();

    return 0;
}

template <double (*dSdt)(const Vector&), void (*Sgradxvec)(const Vector&, Vector& )> \
void sigmaTemplate(const Vector& xt, Vector& sigma)
{
    sigma.SetSize(xt.Size());

    Vector gradS;
    Sgradxvec(xt,gradS);

    sigma(xt.Size()-1) = dSdt(xt);
    for (int i = 0; i < xt.Size()-1; i++)
        sigma(i) = - gradS(i);

    return;
}

template <double (*dSdt)(const Vector&), void (*Sgradxvec)(const Vector&, Vector& ), void (*dSdtgradxvec)(const Vector&, Vector& )> \
void sigmaNonHomoTemplate(const Vector& xt, Vector& sigma) // sigmaNonHomo = ( - grad u, u_t) for u = S(t=0)
{
    sigma.SetSize(xt.Size());

    double t = xt(xt.Size() - 1);

    Vector xteq0(xt.Size()); // xt with t = 0
    xteq0 = xt;
    xteq0(xteq0.Size()-1) = 0.0;

    Vector gradS;
    Sgradxvec(xteq0,gradS);
    Vector graddSdt;
    dSdtgradxvec(xteq0,graddSdt);
    //Sgradxvec(xt,gradS);

    sigma(xt.Size()-1) = dSdt(xteq0); /////////////// because dSdt|t=0 = - sigma * n |t = 0
    for (int i = 0; i < xt.Size()-1; i++)
        sigma(i) = - gradS(i) - t * graddSdt(i);

    return;
}


template<double (*S)(const Vector & xt), double (*d2Sdt2)(const Vector & xt), double (*Slaplace)(const Vector & xt), double (*dSdtlaplace)(const Vector & xt) > \
double rhsideTemplate(const Vector& xt)
{
    Vector xt0(xt.Size());
    xt0 = xt;
    xt0(xt0.Size() - 1) = 0;

    double t = xt(xt.Size() - 1);

    return d2Sdt2(xt) - Slaplace(xt) + Slaplace(xt0) + t * dSdtlaplace(xt0);
}

template<double (*S)(const Vector & xt), double (*d2Sdt2)(const Vector & xt), double (*Slaplace)(const Vector & xt) > \
double divsigmaTemplate(const Vector& xt)
{
    Vector xt0(xt.Size());
    xt0 = xt;
    xt0 (xt0.Size() - 1) = 0;

    return d2Sdt2(xt) - Slaplace(xt);
}

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt)> \
    double SnonhomoTemplate(const Vector& xt)
{
    Vector xt0(xt.Size());
    xt0 = xt;
    xt0 (xt0.Size() - 1) = 0;

    double t = xt(xt.Size()-1);

    return S(xt0) + t * dSdt(xt0);
}


double uFun_ex(const Vector & xt)
{
    const double PI = 3.141592653589793;
    double xi(xt(0));
    double yi(xt(1));
    double zi(0.0);
    double vi(0.0);

    if (xt.Size() == 3)
    {
        double t = xt(2);
        return sin(PI*xi)*sin(PI*yi) * t * t;
        //return sin(PI*xi)*sin(PI*yi);
        //return sin(PI*xi)*sin(PI*yi) * t;
    }
    if (xt.Size() == 4)
    {
        zi = xt(2);
        vi = xt(3);
        //cout << "sol for 4D" << endl;
        return sin(PI*xi)*sin(PI*yi)*sin(PI*zi)*vi;
    }

    return 0.0;
}

double uFun_ex_dt(const Vector & xt)
{
    const double PI = 3.141592653589793;
    double xi(xt(0));
    double yi(xt(1));
    double zi(0.0);


    if (xt.Size() == 3)
    {
        double t(xt(2));
        return sin(PI*xi)*sin(PI*yi)*2*t;
        //return 1.0;
        //return 0.0;
        //return sin(PI*xi)*sin(PI*yi);
    }
    if (xt.Size() == 4)
    {
        zi = xt(2);
        return sin(PI*xi)*sin(PI*yi)*sin(PI*zi);
    }


    return 0.0;
}

double uFun_ex_dt2(const Vector & xt)
{
    double xi(xt(0));
    double yi(xt(1));
//    double zi(0.0);

    if (xt.Size() == 3)
    {
        return sin(M_PI*xi)*sin(M_PI*yi)*2.0;
        //return 0.0;
    }

    return 0.0;

}

double uFun_ex_laplace(const Vector & xt)
{
    return (-(xt.Size()-1) * M_PI * M_PI) *uFun_ex(xt);
    //return 0.0;
}

double uFun_ex_dtlaplace(const Vector & xt)
{
    double xi(xt(0));
    double yi(xt(1));
//    double zi(0.0);
    double t(xt(xt.Size() - 1));
    //return (-(xt.Size()-1) * PI * PI) *uFun_ex(xt);
    //return (-(xt.Size()-1) * M_PI * M_PI) *sin(M_PI*xi)*sin(M_PI*yi);         // for t * sin x * sin y
    return (-(xt.Size()-1) * M_PI * M_PI) *sin(M_PI*xi)*sin(M_PI*yi) * 2.0 * t; // for t^2 * sin x * sin y
    return 0.0;
}

void uFun_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
//    double z(0.0);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);


    if (xt.Size() == 3)
    {
        gradx(0) = t * t * M_PI * cos (M_PI * x) * sin (M_PI * y);
        gradx(1) = t * t * M_PI * sin (M_PI * x) * cos (M_PI * y);
    }

    /*
    if (xt.Size() == 4)
    {
        z = xt(2);
        gradx(0) = t * PI * cos (PI * x) * sin (PI * y) * sin (PI * z);
        gradx(1) = t * PI * sin (PI * x) * cos (PI * y) * sin (PI * z);
        gradx(2) = t * PI * sin (PI * x) * sin (PI * y) * cos (PI * z);
    }
    */


    /*
    if (xt.Size() == 3)
    {
        gradx(0) = M_PI * cos (M_PI * x) * sin (M_PI * y);
        gradx(1) = M_PI * sin (M_PI * x) * cos (M_PI * y);
    }
    */


    /*
    if (xt.Size() == 3)
    {
        gradx(0) = t * M_PI * cos (M_PI * x) * sin (M_PI * y);
        gradx(1) = t * M_PI * sin (M_PI * x) * cos (M_PI * y);
    }
    */


}

void uFun_ex_dtgradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
//    double z(0.0);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    // for t * sin x * sin y
    /*
    if (xt.Size() == 3)
    {
        gradx(0) = M_PI * cos (M_PI * x) * sin (M_PI * y);
        gradx(1) = M_PI * sin (M_PI * x) * cos (M_PI * y);
    }
    */

    // for t^2 * sin x * sin y
    if (xt.Size() == 3)
    {
        gradx(0) = M_PI * cos (M_PI * x) * sin (M_PI * y) * 2.0 * t;
        gradx(1) = M_PI * sin (M_PI * x) * cos (M_PI * y) * 2.0 * t;
    }

}

double fFun(const Vector & x)
{
    double xi(x(0));
    double yi(x(1));
    double zi(0.0);
    double vi(0.0);
    if (x.Size() == 3)
    {
     zi = x(2);
       return 2*M_PI*M_PI*sin(M_PI*xi)*sin(M_PI*yi)*zi+sin(M_PI*xi)*sin(M_PI*yi);
    }

    if (x.Size() == 4)
    {
     zi = x(2);
         vi = x(3);
         //cout << "rhand for 4D" << endl;
       return 3*M_PI*M_PI*sin(M_PI*xi)*sin(M_PI*yi)*sin(M_PI*zi)*vi + sin(M_PI*xi)*sin(M_PI*yi)*sin(M_PI*zi);
    }

    return 0.0;
}

void sigmaFun_ex(const Vector & x, Vector & u)
{
    const double PI = 3.141592653589793;
    double xi(x(0));
    double yi(x(1));
    double zi(0.0);
    double vi(0.0);
    if (x.Size() == 3)
    {
        zi = x(2);
        u(0) = - PI * cos (PI * xi) * sin (PI * yi) * zi;
        u(1) = - PI * cos (PI * yi) * sin (PI * xi) * zi;
        u(2) = uFun_ex(x);
        return;
    }

    if (x.Size() == 4)
    {
        zi = x(2);
        vi = x(3);
        u(0) = - PI * cos (PI * xi) * sin (PI * yi) * sin(PI * zi) * vi;
        u(1) = - sin (PI * xi) * PI * cos (PI * yi) * sin(PI * zi) * vi;
        u(2) = - sin (PI * xi) * sin(PI * yi) * PI * cos (PI * zi) * vi;
        u(3) = uFun_ex(x);
        return;
    }

    if (x.Size() == 2)
    {
        u(0) =  exp(-PI*PI*yi)*PI*cos(PI*xi);
        u(1) = -sin(PI*xi)*exp(-1*PI*PI*yi);
        return;
    }

    return;
}



double uFun1_ex(const Vector & xt)
{
    double tmp = (xt.Size() == 4) ? sin(M_PI*xt(2)) : 1.0;
    return exp(-xt(xt.Size()-1))*sin(M_PI*xt(0))*sin(M_PI*xt(1))*tmp;
}

double uFun1_ex_dt(const Vector & xt)
{
    return - uFun1_ex(xt);
}

double uFun1_ex_dt2(const Vector & xt)
{
    return uFun1_ex(xt);
}

double uFun1_ex_laplace(const Vector & xt)
{
    return (- (xt.Size() - 1) * M_PI * M_PI ) * uFun1_ex(xt);
}

double uFun1_ex_dtlaplace(const Vector & xt)
{
    return -uFun1_ex_laplace(xt);
}

void uFun1_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double z(0.0);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    if (xt.Size() == 3)
    {
        gradx(0) = exp(-t) * M_PI * cos (M_PI * x) * sin (M_PI * y);
        gradx(1) = exp(-t) * M_PI * sin (M_PI * x) * cos (M_PI * y);
    }
    if (xt.Size() == 4)
    {
        z = xt(2);
        gradx(0) = exp(-t) * M_PI * cos (M_PI * x) * sin (M_PI * y) * sin (M_PI * z);
        gradx(1) = exp(-t) * M_PI * sin (M_PI * x) * cos (M_PI * y) * sin (M_PI * z);
        gradx(2) = exp(-t) * M_PI * sin (M_PI * x) * sin (M_PI * y) * cos (M_PI * z);
    }

}

void uFun1_ex_dtgradx(const Vector& xt, Vector& gradx )
{
    gradx.SetSize(xt.Size() - 1);

    Vector gradS;
    uFun1_ex_gradx(xt,gradS);

    for ( int d = 0; d < xt.Size() - 1; ++d)
        gradx(d) = - gradS(d);
}

double fFun1(const Vector & x)
{
    return ( (x.Size()-1)*M_PI*M_PI - 1. ) * uFun1_ex(x);
}

void sigmaFun1_ex(const Vector & x, Vector & sigma)
{
    sigma.SetSize(x.Size());
    sigma(0) = -M_PI*exp(-x(x.Size()-1))*cos(M_PI*x(0))*sin(M_PI*x(1));
    sigma(1) = -M_PI*exp(-x(x.Size()-1))*sin(M_PI*x(0))*cos(M_PI*x(1));
    if (x.Size() == 4)
    {
        sigma(0) *= sin(M_PI*x(2));
        sigma(1) *= sin(M_PI*x(2));
        sigma(2) = -M_PI*exp(-x(x.Size()-1))*sin(M_PI*x(0))
                *sin(M_PI*x(1))*cos(M_PI*x(2));
    }
    sigma(x.Size()-1) = uFun1_ex(x);

    return;
}

double uFun2_ex(const Vector & xt)
{
    if (xt.Size() != 4)
        cout << "Error, this is only 4-d = 3-d + time solution" << endl;
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(3);

    return exp(-t) * x * sin (M_PI * x) * (1 + y) * sin (M_PI * y) * (2 - z) * sin (M_PI * z);
}

double uFun2_ex_dt(const Vector & xt)
{
    return - uFun2_ex(xt);
}

double uFun2_ex_dt2(const Vector & xt)
{
    return uFun2_ex(xt);
}

double uFun2_ex_laplace(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(3);

    double res = 0.0;
    res += exp(-t) * (2.0 * M_PI * cos(M_PI * x) - x * M_PI * M_PI * sin (M_PI * x)) * (1 + y) * sin (M_PI * y) * (2 - z) * sin (M_PI * z);
    res += exp(-t) * x * sin (M_PI * x) * (2.0 * M_PI * cos(M_PI * y) - (1 + y) * M_PI * M_PI * sin(M_PI * y)) * (2 - z) * sin (M_PI * z);
    res += exp(-t) * x * sin (M_PI * x) * (1 + y) * sin (M_PI * y) * (2.0 * (-1) * cos(M_PI * z) - (2 - z) * M_PI * M_PI * sin(M_PI * z));
    return res;
}

double uFun2_ex_dtlaplace(const Vector & xt)
{
    return -uFun2_ex_laplace(xt);
}

void uFun2_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(3);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = exp(-t) * (sin (M_PI * x) + x * M_PI * cos(M_PI * x)) * (1 + y) * sin (M_PI * y) * (2 - z) * sin (M_PI * z);
    gradx(1) = exp(-t) * x * sin (M_PI * x) * (sin (M_PI * y) + (1 + y) * M_PI * cos(M_PI * y)) * (2 - z) * sin (M_PI * z);
    gradx(2) = exp(-t) * x * sin (M_PI * x) * (1 + y) * sin (M_PI * y) * (- sin (M_PI * z) + (2 - z) * M_PI * cos(M_PI * z));
}

void uFun2_ex_dtgradx(const Vector& xt, Vector& gradx )
{
    gradx.SetSize(xt.Size() - 1);

    Vector gradS;
    uFun2_ex_gradx(xt,gradS);

    for ( int d = 0; d < xt.Size() - 1; ++d)
        gradx(d) = - gradS(d);
}

double uFun4_ex(const Vector & xt)
{
    if (xt.Size() != 3)
        cout << "Error, this is only 3-d solution" << endl;
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    return 16.0 * x * (x - 1) * y * (y - 1) * t * t;
}

double uFun4_ex_dt(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    return 16.0 * x * (x - 1) * y * (y - 1) * 2.0 * t;
}

double uFun4_ex_dt2(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
//    double t = xt(2);

    return 16.0 * x * (x - 1) * y * (y - 1) * 2.0;
}

double uFun4_ex_laplace(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    return 16.0 * (2.0 * y * (y - 1) + 2.0 * x * (x - 1)) * t * t;
}

double uFun4_ex_dtlaplace(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    return 16.0 * (2.0 * y * (y - 1) + 2.0 * x * (x - 1)) * 2.0 * t;
}

void uFun4_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = 16.0 * (2.0 * x - 1) * y * (y - 1) * t * t;
    gradx(1) = 16.0 * x * (x - 1) * (2.0 * y - 1) * t * t;

}

void uFun4_ex_dtgradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = 16.0 * (2.0 * x - 1) * y * (y - 1) * 2.0 * t;
    gradx(1) = 16.0 * x * (x - 1) * (2.0 * y - 1) * 2.0 * t;
}

double uFun3_ex(const Vector & xt)
{
    if (xt.Size() != 3)
        cout << "Error, this is only 3-d = 2d + time solution" << endl;
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    return exp(-t) * x * sin (M_PI * x) * (1 + y) * sin (M_PI * y);
}

double uFun3_ex_dt(const Vector & xt)
{
    return - uFun3_ex(xt);
}

double uFun3_ex_dt2(const Vector & xt)
{
    return uFun3_ex(xt);
}

double uFun3_ex_laplace(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    double res = 0.0;
    res += exp(-t) * (2.0 * M_PI * cos(M_PI * x) - x * M_PI * M_PI * sin (M_PI * x)) * (1 + y) * sin (M_PI * y);
    res += exp(-t) * x * sin (M_PI * x) * (2.0 * M_PI * cos(M_PI * y) - (1 + y) * M_PI * M_PI * sin(M_PI * y));
    return res;
}

double uFun3_ex_dtlaplace(const Vector & xt)
{
    return -uFun3_ex_laplace(xt);
}

void uFun3_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = exp(-t) * (sin (M_PI * x) + x * M_PI * cos(M_PI * x)) * (1 + y) * sin (M_PI * y);
    gradx(1) = exp(-t) * x * sin (M_PI * x) * (sin (M_PI * y) + (1 + y) * M_PI * cos(M_PI * y));
}

void uFun3_ex_dtgradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = - exp(-t) * (sin (M_PI * x) + x * M_PI * cos(M_PI * x)) * (1 + y) * sin (M_PI * y);
    gradx(1) = - exp(-t) * x * sin (M_PI * x) * (sin (M_PI * y) + (1 + y) * M_PI * cos(M_PI * y));

    /*
    gradx.SetSize(xt.Size() - 1);

    Vector gradS;
    uFun3_ex_gradx(xt,gradS);

    for ( int d = 0; d < xt.Size() - 1; ++d)
        gradx(d) = - gradS(d);
        */
}


double uFun5_ex(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    return 16.0 * x * x * (x - 1) * (x - 1) * y * y * (y - 1) * (y - 1) * t * t;
}

double uFun5_ex_dt(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    return 16.0 * x * x * (x - 1) * (x - 1) * y * y * (y - 1) * (y - 1) * 2.0 * t;
}

double uFun5_ex_dt2(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
//    double t = xt(2);

    return 16.0 * x * x * (x - 1) * (x - 1) * y * y * (y - 1) * (y - 1) * 2.0;
}

double uFun5_ex_laplace(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    return 16.0 * (2.0 * ((x-1)*(2*x-1) + x*(2*x-1) + 2*x*(x-1)) * y * (y - 1) * y * (y - 1)\
                   + 2.0 * ((y-1)*(2*y-1) + y*(2*y-1) + 2*y*(y-1)) * x * (x - 1) * x * (x - 1)) * t * t;
}

double uFun5_ex_dtlaplace(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    return 16.0 * (2.0 * ((x-1)*(2*x-1) + x*(2*x-1) + 2*x*(x-1)) * y * (y - 1) * y * (y - 1)\
                   + 2.0 * ((y-1)*(2*y-1) + y*(2*y-1) + 2*y*(y-1)) * x * (x - 1) * x * (x - 1)) * 2.0 * t;
}

void uFun5_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = 16.0 * 2.0 * x * (x - 1) * (2.0 * x - 1) * y * (y - 1) * y * (y - 1) * t * t;
    gradx(1) = 16.0 * x * (x - 1) * x * (x - 1) * 2.0 * y * (y - 1) * (2.0 * y - 1) * t * t;

}

void uFun5_ex_dtgradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = 16.0 * 2.0 * x * (x - 1) * (2.0 * x - 1) * y * (y - 1) * y * (y - 1) * 2.0 * t;
    gradx(1) = 16.0 * x * (x - 1) * x * (x - 1) * 2.0 * y * (y - 1) * (2.0 * y - 1) * 2.0 * t;
}

