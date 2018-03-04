//
//                        MFEM CFOSLS Heat equation with multilevel algorithm and multigrid (div-free part)
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>
#include <unistd.h>

#include "cfosls_testsuite.hpp"

// (de)activates solving of the discrete global problem
#define OLD_CODE

//#define WITH_DIVCONSTRAINT_SOLVER

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

//#define CHECK_SPDCOARSESTSOLVER

//#define DEBUG_SMOOTHER

// activates a check for the symmetry of the new solver
//#define CHECK_SPDSOLVER

// activates constraint residual check after each iteration of the minimization solver
#define CHECK_CONSTR

#define CHECK_BNDCND

//#define MARTIN_PREC

#define BND_FOR_MULTIGRID
//#define BLKDIAG_SMOOTHER

//#define COMPARE_MG

//#define COARSEPREC_AMS

#ifdef COMPARE_MG // options for multigrid, specific for detailed comparison of mg

#define NCOARSEITER 4

//#define NO_COARSESOLVE
//#define NO_POSTSMOOTH
//#define NO_PRESMOOTH

//#define COMPARE_COARSE_SOLVERS
//#define COMPARE_SMOOTHERS
#endif // for ifdef COMPARE_MG

//#define TIMING

#ifdef TIMING
#undef CHECK_LOCALSOLVE
#undef CHECK_CONSTR
#undef CHECK_BNDCND
#endif

#include "divfree_solver_tools.hpp"

#define MYZEROTOL (1.0e-13)

// must be always active
#define USE_CURLMATRIX

using namespace std;
using namespace mfem;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

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
};

//=-=-=-=-=-=-=-=-=-=-=-=-=-
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

int ipow(int base, int exp);

// Define the analytical solution and forcing terms / boundary conditions
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

void hcurlFun3D_ex(const Vector& xt, Vector& vecvalue);
void curlhcurlFun3D_ex(const Vector& xt, Vector& vecvalue);
void hcurlFun3D_2_ex(const Vector& xt, Vector& vecvalue);
void curlhcurlFun3D_2_ex(const Vector& xt, Vector& vecvalue);

double zero_ex(const Vector& xt);
void zerovectx_ex(const Vector& xt, Vector& vecvalue);
void zerovecx_ex(const Vector& xt, Vector& zerovecx );
void zerovecMat4D_ex(const Vector& xt, Vector& vecvalue);

void vminusone_exact(const Vector &x, Vector &vminusone);
void vone_exact(const Vector &x, Vector &vone);

// Exact solution, E, and r.h.s., f. See below for implementation.
void E_exact(const Vector &, Vector &);
void curlE_exact(const Vector &x, Vector &curlE);
void f_exact(const Vector &, Vector &);
double freq = 1.0, kappa;

template<double (*S)(const Vector & xt), double (*d2Sdt2)(const Vector & xt), double (*Slaplace)(const Vector & xt)> \
    double divsigmaTemplate(const Vector& xt);


template <double (*dSdt)(const Vector&), void(*Sgradxvec)(const Vector & x, Vector & gradx) >
    void sigmaTemplate(const Vector& xt, Vector& sigma);

////////////////////////


template<double (*S)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx),
        void (*opdivfreevec)(const Vector&, Vector& )>
        void sigmahatTemplate(const Vector& xt, Vector& sigmahatv);
template<double (*S)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx),
        void (*opdivfreevec)(const Vector&, Vector& )>
        void minsigmahatTemplate(const Vector& xt, Vector& sigmahatv);


class Wave_test_divfree
{
protected:
    int dim;
    int numsol;
    int numcurl;
    bool testisgood;

public:
    FunctionCoefficient * scalarS;                // S
    FunctionCoefficient * scalardivsigma;         // = d^2S/dt^2 - laplace S                  - what is used for computing error
    VectorFunctionCoefficient * sigma;
    VectorFunctionCoefficient * sigmahat;         // sigma_hat = sigma_exact - op divfreepart (curl hcurlpart in 3D)
    VectorFunctionCoefficient * divfreepart;      // additional part added for testing div-free solver
    VectorFunctionCoefficient * opdivfreepart;    // curl of the additional part which is added to sigma_exact for testing div-free solver
    VectorFunctionCoefficient * minsigmahat;      // -sigma_hat
public:
    Wave_test_divfree (int Dim, int NumSol, int NumCurl);

    int GetDim() {return dim;}
    int GetNumSol() {return numsol;}
    int GetNumCurl() {return numcurl;}
    int CheckIfTestIsGood() {return testisgood;}
    void SetDim(int Dim) { dim = Dim;}
    void SetNumSol(int NumSol) { numsol = NumSol;}
    void SetNumCurl(int NumCurl) { numcurl = NumCurl;}
    bool CheckTestConfig();

    ~Wave_test_divfree () {}
private:
    void SetScalarFun( double (*f)(const Vector & xt))
    { scalarS = new FunctionCoefficient(f);}

    template<double (*S)(const Vector & xt), double (*d2Sdt2)(const Vector & xt), double (*Slaplace)(const Vector & xt)> \
    void SetDivSigma()
    { scalardivsigma = new FunctionCoefficient(divsigmaTemplate<S, d2Sdt2, Slaplace>);}


    template<double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & vec)> \
    void SetSigmaVec()
    {
        sigma = new VectorFunctionCoefficient(dim, sigmaTemplate<dSdt,Sgradxvec>);
    }

    void SetDivfreePart( void(*divfreevec)(const Vector & x, Vector & vec))
    { divfreepart = new VectorFunctionCoefficient(dim, divfreevec);}

    void SetOpDivfreePart( void(*opdivfreevec)(const Vector & x, Vector & vec))
    { opdivfreepart = new VectorFunctionCoefficient(dim, opdivfreevec);}

    template<double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx),
             void(*opdivfreevec)(const Vector & x, Vector & vec)> \
    void Setsigmahat()
    { sigmahat = new VectorFunctionCoefficient(dim, sigmahatTemplate<dSdt, Sgradxvec, opdivfreevec>);}

    template<double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx),
             void(*opdivfreevec)(const Vector & x, Vector & vec)> \
    void Setminsigmahat()
    { minsigmahat = new VectorFunctionCoefficient(dim, minsigmahatTemplate<dSdt, Sgradxvec, opdivfreevec>);}

    template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), double (*d2Sdt2)(const Vector & xt),\
             double (*Slaplace)(const Vector & xt), double (*dSdtlaplace)(const Vector & xt), \
             void(*Sgradxvec)(const Vector & x, Vector & gradx), void (*dSdtgradxvec)(const Vector&, Vector& ), \
             void(*divfreevec)(const Vector & x, Vector & vec), void(*opdivfreevec)(const Vector & x, Vector & vec)> \
    void SetTestCoeffs ( );

};


template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), double (*d2Sdt2)(const Vector & xt),\
         double (*Slaplace)(const Vector & xt), double (*dSdtlaplace)(const Vector & xt), \
         void(*Sgradxvec)(const Vector & x, Vector & gradx), void (*dSdtgradxvec)(const Vector&, Vector& ), \
         void(*divfreevec)(const Vector & x, Vector & vec), void(*opdivfreevec)(const Vector & x, Vector & vec)> \
void Wave_test_divfree::SetTestCoeffs ()
{
    SetScalarFun(S);
    SetSigmaVec<dSdt,Sgradxvec>();
    SetDivSigma<S, d2Sdt2, Slaplace>();
    SetDivfreePart(divfreevec);
    SetOpDivfreePart(opdivfreevec);
    Setsigmahat<dSdt, Sgradxvec, opdivfreevec>();
    Setminsigmahat<dSdt, Sgradxvec, opdivfreevec>();
    return;
}


bool Wave_test_divfree::CheckTestConfig()
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
    else
        return false;

}

Wave_test_divfree::Wave_test_divfree (int Dim, int NumSol, int NumCurl)
{
    dim = Dim;
    numsol = NumSol;
    numcurl = NumCurl;

    if ( CheckTestConfig() == false )
    {
        std::cerr << "Inconsistent dim and numsol \n" << std::flush;
        testisgood = false;
    }
    else
    {
        if (numsol == -34)
        {
            if (dim == 3)
                SetTestCoeffs<&uFunTest_ex, &uFunTest_ex_dt, &uFunTest_ex_dt2, &uFunTest_ex_laplace, &uFunTest_ex_dtlaplace,
                        &uFunTest_ex_gradx, &uFunTest_ex_dtgradx, &zerovectx_ex, &zerovectx_ex>();
            else // dim == 4
                SetTestCoeffs<&uFunTest_ex, &uFunTest_ex_dt, &uFunTest_ex_dt2, &uFunTest_ex_laplace, &uFunTest_ex_dtlaplace,
                        &uFunTest_ex_gradx, &uFunTest_ex_dtgradx, &zerovecMat4D_ex, &zerovectx_ex>();
        }

        if (numsol == 0)
        {
            if (numcurl == 1)
                SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_dt2, &uFun_ex_laplace, &uFun_ex_dtlaplace, &uFun_ex_gradx, &uFun_ex_dtgradx, &hcurlFun3D_ex, &curlhcurlFun3D_ex>();
            else if (numcurl == 2)
                SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_dt2, &uFun_ex_laplace, &uFun_ex_dtlaplace, &uFun_ex_gradx, &uFun_ex_dtgradx, &hcurlFun3D_2_ex, &curlhcurlFun3D_2_ex>();
            else
                SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_dt2, &uFun_ex_laplace, &uFun_ex_dtlaplace, &uFun_ex_gradx, &uFun_ex_dtgradx, &zerovectx_ex, &zerovectx_ex>();
        }
        if (numsol == 1)
        {
            if (numcurl == 1)
                SetTestCoeffs<&uFun1_ex, &uFun1_ex_dt, &uFun1_ex_dt2, &uFun1_ex_laplace, &uFun1_ex_dtlaplace, &uFun1_ex_gradx, &uFun1_ex_dtgradx, &hcurlFun3D_ex, &curlhcurlFun3D_ex>();
            else if (numcurl == 2)
                SetTestCoeffs<&uFun1_ex, &uFun1_ex_dt, &uFun1_ex_dt2, &uFun1_ex_laplace, &uFun1_ex_dtlaplace, &uFun1_ex_gradx, &uFun1_ex_dtgradx, &hcurlFun3D_2_ex, &curlhcurlFun3D_2_ex>();
            else
                SetTestCoeffs<&uFun1_ex, &uFun1_ex_dt, &uFun1_ex_dt2, &uFun1_ex_laplace, &uFun1_ex_dtlaplace, &uFun1_ex_gradx, &uFun1_ex_dtgradx, &zerovectx_ex, &zerovectx_ex>();
        }
        if (numsol == 2)
        {
            //if (numcurl == 1)
                //SetTestCoeffs<&uFun2_ex, &uFun2_ex_dt, &uFun2_ex_dt2, &uFun2_ex_laplace, &uFun2_ex_dtlaplace, &uFun2_ex_gradx, &uFun2_ex_dtgradx, &hcurlFun3D_ex, &curlhcurlFun3D_ex>();
            //else if (numcurl == 2)
                //SetTestCoeffs<&uFun2_ex, &uFun2_ex_dt, &uFun2_ex_dt2, &uFun2_ex_laplace, &uFun2_ex_dtlaplace, &uFun2_ex_gradx, &uFun2_ex_dtgradx, &hcurlFun3D_2_ex, &curlhcurlFun3D_2_ex>();
            if (numcurl == 1 || numcurl == 2)
            {
                std::cout << "Critical error: Explicit analytic div-free guy is not implemented in 4D \n";
            }
            else
                SetTestCoeffs<&uFun2_ex, &uFun2_ex_dt, &uFun2_ex_dt2, &uFun2_ex_laplace, &uFun2_ex_dtlaplace, &uFun2_ex_gradx, &uFun2_ex_dtgradx, &zerovecMat4D_ex, &zerovectx_ex>();
        }
        if (numsol == 3)
        {
            if (numcurl == 1)
                SetTestCoeffs<&uFun3_ex, &uFun3_ex_dt, &uFun3_ex_dt2, &uFun3_ex_laplace, &uFun3_ex_dtlaplace, &uFun3_ex_gradx, &uFun3_ex_dtgradx, &hcurlFun3D_ex, &curlhcurlFun3D_ex>();
            else if (numcurl == 2)
                SetTestCoeffs<&uFun3_ex, &uFun3_ex_dt, &uFun3_ex_dt2, &uFun3_ex_laplace, &uFun3_ex_dtlaplace, &uFun3_ex_gradx, &uFun3_ex_dtgradx, &hcurlFun3D_2_ex, &curlhcurlFun3D_2_ex>();
            else
                SetTestCoeffs<&uFun3_ex, &uFun3_ex_dt, &uFun3_ex_dt2, &uFun3_ex_laplace, &uFun3_ex_dtlaplace, &uFun3_ex_gradx, &uFun3_ex_dtgradx, &zerovectx_ex, &zerovectx_ex>();
        }
        if (numsol == 4)
        {
            if (numcurl == 1)
                SetTestCoeffs<&uFun4_ex, &uFun4_ex_dt, &uFun4_ex_dt2, &uFun4_ex_laplace, &uFun4_ex_dtlaplace, &uFun4_ex_gradx, &uFun4_ex_dtgradx, &hcurlFun3D_ex, &curlhcurlFun3D_ex>();
            else if (numcurl == 2)
                SetTestCoeffs<&uFun4_ex, &uFun4_ex_dt, &uFun4_ex_dt2, &uFun4_ex_laplace, &uFun4_ex_dtlaplace, &uFun4_ex_gradx, &uFun4_ex_dtgradx, &hcurlFun3D_2_ex, &curlhcurlFun3D_2_ex>();
            else
                SetTestCoeffs<&uFun4_ex, &uFun4_ex_dt, &uFun4_ex_dt2, &uFun4_ex_laplace, &uFun4_ex_dtlaplace, &uFun4_ex_gradx, &uFun4_ex_dtgradx, &zerovectx_ex, &zerovectx_ex>();
        }
        if (numsol == 5)
        {
            if (numcurl == 1)
                SetTestCoeffs<&uFun5_ex, &uFun5_ex_dt, &uFun5_ex_dt2, &uFun5_ex_laplace, &uFun5_ex_dtlaplace, &uFun5_ex_gradx, &uFun5_ex_dtgradx, &hcurlFun3D_ex, &curlhcurlFun3D_ex>();
            else if (numcurl == 2)
                SetTestCoeffs<&uFun5_ex, &uFun5_ex_dt, &uFun5_ex_dt2, &uFun5_ex_laplace, &uFun5_ex_dtlaplace, &uFun5_ex_gradx, &uFun5_ex_dtgradx, &hcurlFun3D_2_ex, &curlhcurlFun3D_2_ex>();
            else
                SetTestCoeffs<&uFun5_ex, &uFun5_ex_dt, &uFun5_ex_dt2, &uFun5_ex_laplace, &uFun5_ex_dtlaplace, &uFun5_ex_gradx, &uFun5_ex_dtgradx, &zerovectx_ex, &zerovectx_ex>();
        }

        testisgood = true;
    }
}

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
    int numsol          = -34;
    int numcurl         = 0;

    int ser_ref_levels  = 1;
    int par_ref_levels  = 1;

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

    kappa = freq * M_PI;

    if (verbose)
        cout << "Solving CFOSLS Wave equation with MFEM & hypre, div-free approach, minimization solver \n";

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

    std::cout << std::flush;
    MPI_Barrier(MPI_COMM_WORLD);

    if (verbose)
        std::cout << "Running tests for the paper: \n";

    if (nDimensions == 3)
    {
        numsol = -34;
        mesh_file = "../data/cube_3d_moderate.mesh";
    }
    else // 4D case
    {
        numsol = -34;
        mesh_file = "../data/cube4d_96.MFEM";
    }

    if (verbose)
        std::cout << "For the records: numsol = " << numsol
                  << ", mesh_file = " << mesh_file << "\n";

    Wave_test_divfree Mytest(nDimensions, numsol, numcurl);

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

    MFEM_ASSERT(!(aniso_refine && (with_multilevel || nDimensions == 4)),"Anisotropic refinement works only in 3D and without multilevel algorithm \n");

    ////////////////////////////////// new

    int dim = nDimensions;

    Array<int> ess_bdrSigma(pmesh->bdr_attributes.Max());
    ess_bdrSigma = 0;
    ess_bdrSigma[0] = 1;

    Array<int> ess_bdrS(pmesh->bdr_attributes.Max());
    ess_bdrS = 0;
    ess_bdrS = 1;
    ess_bdrS[pmesh->bdr_attributes.Max() - 1] = 0;

    Array<int> all_bdrSigma(pmesh->bdr_attributes.Max());
    all_bdrSigma = 1;

    Array<int> all_bdrS(pmesh->bdr_attributes.Max());
    all_bdrS = 1;

    int ref_levels = par_ref_levels;

    int num_levels = ref_levels + 1;

    chrono.Clear();
    chrono.Start();

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

    const SparseMatrix* P_W_local;
    const SparseMatrix* P_R_local;
    const SparseMatrix* P_H_local;

    DivPart divp;

    int numblocks_funct = 2;

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
       BdrDofs_Funct_lvls[l][1] = new Array<int>;
       EssBdrDofs_Funct_lvls[l][1] = new Array<int>;
       EssBdrTrueDofs_Funct_lvls[l][1] = new Array<int>;
       EssBdrTrueDofs_HcurlFunct_lvls[l][1] = new Array<int>;
       EssBdrDofs_H1[l] = new Array<int>;

       Funct_mat_offsets_lvls[l] = new Array<int>;

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
        H_space_lvls[l]->GetEssentialVDofs(all_bdrS, *BdrDofs_Funct_lvls[l][1]);
        H_space_lvls[l]->GetEssentialVDofs(ess_bdrS, *EssBdrDofs_Funct_lvls[l][1]);
        H_space_lvls[l]->GetEssentialTrueDofs(ess_bdrS, *EssBdrTrueDofs_Funct_lvls[l][1]);
        H_space_lvls[l]->GetEssentialTrueDofs(ess_bdrS, *EssBdrTrueDofs_HcurlFunct_lvls[l][1]);
        H_space_lvls[l]->GetEssentialVDofs(ess_bdrS, *EssBdrDofs_H1[l]);

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
        Ablock->AddDomainIntegrator(new VectorFEMassIntegrator);
        Ablock->Assemble();
        //Ablock->EliminateEssentialBC(ess_bdrSigma);//, *sigma_exact_finest, *fform); // makes res for sigma_special happier
        Ablock->Finalize();

        // getting pointers to dof_truedof matrices
#ifndef HCURL_COARSESOLVER
        if (l < num_levels - 1)
#endif
          Dof_TrueDof_Hcurl_lvls[l] = C_space_lvls[l]->Dof_TrueDof_Matrix();
        Dof_TrueDof_Func_lvls[l][0] = R_space_lvls[l]->Dof_TrueDof_Matrix();
        Dof_TrueDof_Hdiv_lvls[l] = Dof_TrueDof_Func_lvls[l][0];
        Dof_TrueDof_L2_lvls[l] = W_space_lvls[l]->Dof_TrueDof_Matrix();
        Dof_TrueDof_H1_lvls[l] = H_space_lvls[l]->Dof_TrueDof_Matrix();
        Dof_TrueDof_Func_lvls[l][1] = Dof_TrueDof_H1_lvls[l];

        if (l == 0)
        {
            ParBilinearForm *Cblock;
            // diagonal block for H^1
            Cblock = new ParBilinearForm(H_space_lvls[l]);
            Cblock->AddDomainIntegrator(new WaveVectorFEIntegrator);
            Cblock->Assemble();
            // FIXME: What about boundary conditons here?
            //Cblock->EliminateEssentialBC(ess_bdrS, xblks.GetBlock(1),*qform);
            Cblock->Finalize();

            // off-diagonal block for (H(div), H1) block
            ParMixedBilinearForm *Dblock(new ParMixedBilinearForm(H_space_lvls[l], R_space_lvls[l]));
            Dblock->AddDomainIntegrator(new WaveVectorFEIntegratorB);
            Dblock->Assemble();
            Dblock->Finalize();

            Funct_mat_offsets_lvls[l]->SetSize(numblocks_funct + 1);
            (*Funct_mat_offsets_lvls[l])[0] = 0;
            (*Funct_mat_offsets_lvls[l])[1] = Ablock->Height();
            (*Funct_mat_offsets_lvls[l])[2] = Cblock->Height();
            Funct_mat_offsets_lvls[l]->PartialSum();

            Funct_mat_lvls[l] = new BlockMatrix(*Funct_mat_offsets_lvls[l]);
            Funct_mat_lvls[l]->SetBlock(0,0,Ablock->LoseMat());
            Funct_mat_lvls[l]->SetBlock(1,1,Cblock->LoseMat());
            Funct_mat_lvls[l]->SetBlock(0,1,Dblock->LoseMat());
            Funct_mat_lvls[l]->SetBlock(1,0,Transpose(Funct_mat_lvls[l]->GetBlock(0,1)));

            ParMixedBilinearForm *Bblock = new ParMixedBilinearForm(R_space_lvls[l], W_space_lvls[l]);
            Bblock->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
            Bblock->Assemble();
            Bblock->Finalize();
            Constraint_mat_lvls[l] = Bblock->LoseMat();

            // Creating global functional matrix
            offsets_global[0] = 0;
            for ( int blk = 0; blk < numblocks_funct; ++blk)
                offsets_global[blk + 1] = Dof_TrueDof_Func_lvls[l][blk]->Width();
            offsets_global.PartialSum();

            Funct_global = new BlockOperator(offsets_global);

            Functrhs_global = new BlockVector(offsets_global);

            Functrhs_global->GetBlock(0) = 0.0;
            Functrhs_global->GetBlock(1) = 0.0;

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
            Dblock->Assemble();
            {
                Vector temp1(Dblock->Width());
                temp1 = 0.0;
                Vector temp2(Dblock->Height());
                temp2 = 0.0;
                Dblock->EliminateTrialDofs(ess_bdrS, temp1, temp2);
                Dblock->EliminateTestDofs(ess_bdrSigma);
            }
            Dblock->Finalize();
            HypreParMatrix * D = Dblock->ParallelAssemble();
            Funct_global->SetBlock(0,1, D);
            Funct_global->SetBlock(1,0, D->Transpose());

            delete Cblock;
            delete Dblock;
            delete Bblock;
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

            H_space->Update();
            P_H_local = (SparseMatrix *)H_space->GetUpdateOperator();
            SparseMatrix* H_Element_to_dofs1 = new SparseMatrix();
            P_H_lvls[l] = RemoveZeroEntries(*P_H_local);
            divp.Elem2Dofs(*H_space, *H_Element_to_dofs1);
            Element_dofs_H[l] = H_Element_to_dofs1;

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

            auto d_td_coarse_H = H_space_lvls[l + 1]->Dof_TrueDof_Matrix();
            SparseMatrix * RP_H_local = Mult(*H_space_lvls[l]->GetRestrictionMatrix(), *P_H_lvls[l]);
            TrueP_H[num_levels - 2 - l] = d_td_coarse_H->LeftDiagMult(
                        *RP_H_local, H_space_lvls[l]->GetTrueDofOffsets());
            TrueP_H[num_levels - 2 - l]->CopyColStarts();
            TrueP_H[num_levels - 2 - l]->CopyRowStarts();

            delete RP_H_local;

        }

        // FIXME: TrueP_C and TrueP_H has different level ordering compared to TrueP_R

        // creating additional structures required for local problem solvers
        if (l < num_levels - 1)
        {
            (*row_offsets_El_dofs[l])[0] = 0;
            (*row_offsets_El_dofs[l])[1] = Element_dofs_R[l]->Height();
            (*row_offsets_El_dofs[l])[2] = Element_dofs_H[l]->Height();
            row_offsets_El_dofs[l]->PartialSum();

            (*col_offsets_El_dofs[l])[0] = 0;
            (*col_offsets_El_dofs[l])[1] = Element_dofs_R[l]->Width();
            (*col_offsets_El_dofs[l])[2] = Element_dofs_H[l]->Width();
            col_offsets_El_dofs[l]->PartialSum();

            Element_dofs_Func[l] = new BlockMatrix(*row_offsets_El_dofs[l], *col_offsets_El_dofs[l]);
            Element_dofs_Func[l]->SetBlock(0,0, Element_dofs_R[l]);
            Element_dofs_Func[l]->SetBlock(1,1, Element_dofs_H[l]);

            (*row_offsets_P_Func[l])[0] = 0;
            (*row_offsets_P_Func[l])[1] = P_R[l]->Height();
            (*row_offsets_P_Func[l])[2] = P_H_lvls[l]->Height();
            row_offsets_P_Func[l]->PartialSum();

            (*col_offsets_P_Func[l])[0] = 0;
            (*col_offsets_P_Func[l])[1] = P_R[l]->Width();
            (*col_offsets_P_Func[l])[2] = P_H_lvls[l]->Width();
            col_offsets_P_Func[l]->PartialSum();

            P_Func[l] = new BlockMatrix(*row_offsets_P_Func[l], *col_offsets_P_Func[l]);
            P_Func[l]->SetBlock(0,0, P_R[l]);
            P_Func[l]->SetBlock(1,1, P_H_lvls[l]);

            (*row_offsets_TrueP_Func[l])[0] = 0;
            (*row_offsets_TrueP_Func[l])[1] = TrueP_R[l]->Height();
            (*row_offsets_TrueP_Func[l])[2] = TrueP_H[num_levels - 2 - l]->Height();
            row_offsets_TrueP_Func[l]->PartialSum();

            (*col_offsets_TrueP_Func[l])[0] = 0;
            (*col_offsets_TrueP_Func[l])[1] = TrueP_R[l]->Width();
            (*col_offsets_TrueP_Func[l])[2] = TrueP_H[num_levels - 2 - l]->Width();
            col_offsets_TrueP_Func[l]->PartialSum();

            TrueP_Func[l] = new BlockOperator(*row_offsets_TrueP_Func[l], *col_offsets_TrueP_Func[l]);
            TrueP_Func[l]->SetBlock(0,0, TrueP_R[l]);
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
            ParMixedBilinearForm *Dblock;

            Cblock = new ParBilinearForm(H_space_lvls[l]);
            Cblock->AddDomainIntegrator(new WaveVectorFEIntegrator);

            Cblock->Assemble();
            {
                Vector temp1(Cblock->Width());
                temp1 = 0.0;
                Vector temp2(Cblock->Height());
                temp2 = 0.0;
                Cblock->EliminateEssentialBC(ess_bdrS, temp1, temp2);
            }
            Cblock->Finalize();

            // off-diagonal block for (H(div), H1) block
            Dblock = new ParMixedBilinearForm(H_space_lvls[l], R_space_lvls[l]);
            Dblock->AddDomainIntegrator(new WaveVectorFEIntegratorB);
            Dblock->Assemble();
            {
                Vector temp1(Dblock->Width());
                temp1 = 0.0;
                Vector temp2(Dblock->Height());
                temp2 = 0.0;
                Dblock->EliminateTrialDofs(ess_bdrS, temp1, temp2);
                Dblock->EliminateTestDofs(ess_bdrSigma);
            }
            Dblock->Finalize();
            HypreParMatrix * D = Dblock->ParallelAssemble();

            (*Funct_hpmat_lvls[l])(1,1) = Cblock->ParallelAssemble();
            (*Funct_hpmat_lvls[l])(0,1) = D;
            (*Funct_hpmat_lvls[l])(1,0) = D->Transpose();

            delete Cblock;
            delete Dblock;
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
            bool optimized_localsolve = true;
            (*LocalSolver_partfinder_lvls)[l] = new LocalProblemSolverWithS(*Funct_mat_lvls[l],
                                                     *Constraint_mat_lvls[l],
                                                     Dof_TrueDof_Func_lvls[l],
                                                     *P_WT[l],
                                                     *Element_dofs_Func[l],
                                                     *Element_dofs_W[l],
                                                     BdrDofs_Funct_lvls[l],
                                                     EssBdrDofs_Funct_lvls[l],
                                                     optimized_localsolve);

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
        if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
            std::cout << "CoarseSolver size = " << Dof_TrueDof_Func_lvls[num_levels - 1][0]->N()
                    + Dof_TrueDof_Func_lvls[num_levels - 1][1]->N() + Dof_TrueDof_L2_lvls[num_levels - 1]->N() << "\n";
        else
            std::cout << "CoarseSolver size = " << Dof_TrueDof_Func_lvls[num_levels - 1][0]->N() + Dof_TrueDof_L2_lvls[num_levels - 1]->N() << "\n";
#endif
    }

#ifdef CHECK_SPDCOARSESTSOLVER

#ifdef HCURL_COARSESOLVER
    ((CoarsestProblemHcurlSolver*)CoarsestSolver)->SetMaxIter(200);
    ((CoarsestProblemHcurlSolver*)CoarsestSolver)->SetAbsTol(sqrt(1.0e-14));
    ((CoarsestProblemHcurlSolver*)CoarsestSolver)->SetRelTol(sqrt(1.0e-14));
    ((CoarsestProblemHcurlSolver*)CoarsestSolver)->ResetSolverParams();
#else
    CoarsestSolver = CoarsestSolver_partfinder;
    CoarsestSolver_partfinder->SetMaxIter(1000);
    CoarsestSolver_partfinder->SetAbsTol(1.0e-12);
    CoarsestSolver_partfinder->SetRelTol(1.0e-12);
    CoarsestSolver_partfinder->ResetSolverParams();
#endif


    HypreParMatrix * temp = Divfree_hpmat_mod_lvls[num_levels - 1];
    HypreParMatrix * tempT = temp->Transpose();
    HypreParMatrix * CurlCurlT = ParMult(temp, tempT);

    SparseMatrix diag;
    CurlCurlT->GetDiag(diag);

    if (verbose)
        std::cout << "diag of CurlCurlT unsymmetry measure = " << diag.IsSymmetric() << "\n";

    {
        Vector Tempy(CoarsestSolver->Height());

        Vector Vec1(CoarsestSolver->Height());
        Vec1.Randomize(2000);
        Vector Vec2(CoarsestSolver->Height());
        Vec2.Randomize(-39);

        for ( int i = 0; i < EssBdrTrueDofs_Funct_lvls[num_levels - 1][0]->Size(); ++i )
        {
            int tdof = (*EssBdrTrueDofs_Funct_lvls[num_levels - 1][0])[i];
            //std::cout << "index = " << tdof << "\n";
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

        CoarsestSolver->Mult(Vec1, Tempy);
        //CurlCurlT->Mult(Vec1, Tempy);
        double scal1 = Tempy * Vec2;
        double scal3 = Tempy * Vec1;
        //std::cout << "A Vec1 norm = " << Tempy.Norml2() / sqrt (Tempy.Size()) << "\n";

        CoarsestSolver->Mult(Vec2, Tempy);
        //CurlCurlT->Mult(Vec2, Tempy);
        double scal2 = Tempy * Vec1;
        double scal4 = Tempy * Vec2;
        //std::cout << "A Vec2 norm = " << Tempy.Norml2() / sqrt (Tempy.Size()) << "\n";

        std::cout << "scal1 = " << scal1 << "\n";
        std::cout << "scal2 = " << scal2 << "\n";

        if ( fabs(scal1 - scal2) / fabs(scal1) > 1.0e-12)
        {
            std::cout << "CoarsestSolver is not symmetric on two random vectors: \n";
            std::cout << "vec2 * (S * vec1) = " << scal1 << " != " << scal2 << " = vec1 * (S * vec2)" << "\n";
            std::cout << "difference = " << scal1 - scal2 << "\n";
            std::cout << "relative difference = " << fabs(scal1 - scal2) / fabs(scal1) << "\n";
        }
        else
        {
            std::cout << "CoarsestSolver was symmetric on the given vectors: dot product = " << scal1 << "\n";
        }

        std::cout << "scal3 = " << scal3 << "\n";
        std::cout << "scal4 = " << scal4 << "\n";

        if (scal3 < 0 || scal4 < 0)
        {
            std::cout << "The operator (CoarsestSolver) is not s.p.d. \n";
        }
        else
        {
            std::cout << "The CoarsestSolver is s.p.d. on the two random vectors: (Sv,v) > 0 \n";
        }

        //MPI_Finalize();
        //return 0;
    }
#endif


    /*
    StopWatch chrono_debug;

    Vector testRhs(CoarsestSolver->Height());
    testRhs = 1.0;
    Vector testX(CoarsestSolver->Width());
    testX = 0.0;

    MPI_Barrier(comm);
    chrono_debug.Clear();
    chrono_debug.Start();
    for (int it = 0; it < 20; ++it)
    {
        CoarsestSolver->Mult(testRhs, testX);
        testRhs = testX;
    }

    MPI_Barrier(comm);
    chrono_debug.Stop();

    if (verbose)
       std::cout << "CoarsestSolver test run is finished in " << chrono_debug.RealTime() << " \n" << std::flush;

    //delete CoarsestSolver;
    //MPI_Finalize();
    //return 0;
    */

    /*
    // comparing Divfreehpmat with smth from the Divfree_spmat at level 0
    SparseMatrix d_td_Hdiv_diag;
    Dof_TrueDof_Func_lvls[0][0]->GetDiag(d_td_Hdiv_diag);

    SparseMatrix * d_td_Hdiv_diag_T = Transpose(d_td_Hdiv_diag);


    SparseMatrix * tempRA = mfem::Mult(*d_td_Hdiv_diag_T, *Divfree_mat_lvls[0]);
    HypreParMatrix * tempRAP = Dof_TrueDof_Hcurl_lvls[0]->LeftDiagMult(*tempRA, R_space_lvls[0]->GetTrueDofOffsets() );

    ParGridFunction * temppgrfunc = new ParGridFunction(C_space_lvls[0]);
    temppgrfunc->ProjectCoefficient(*Mytest.divfreepart);

    Vector testvec1(tempRAP->Width());
    temppgrfunc->ParallelAssemble(testvec1);
    Vector testvec2(tempRAP->Height());
    tempRAP->Mult(testvec1, testvec2);

    temppgrfunc->ParallelAssemble(testvec1);
    Vector testvec3(tempRAP->Height());
    Divfree_hpmat_mod_lvls[0]->Mult(testvec1, testvec3);

    Vector diffvec(tempRAP->Height());
    double diffnorm = diffvec.Norml2() / sqrt (diffvec.Size());
    MPI_Barrier(comm);
    std::cout << "diffnorm = " << diffnorm << "\n" << std::flush;
    MPI_Barrier(comm);
    */

#ifdef TIMING
    //testing the smoother performance

#ifdef WITH_SMOOTHERS
    for (int l = 0; l < num_levels - 1; ++l)
    {
        StopWatch chrono_debug;

        Vector testRhs(Smoothers_lvls[l]->Height());
        testRhs = 1.0;
        Vector testX(Smoothers_lvls[l]->Width());
        testX = 0.0;

        MPI_Barrier(comm);
        chrono_debug.Clear();
        chrono_debug.Start();
        for (int it = 0; it < 1; ++it)
        {
            Smoothers_lvls[l]->Mult(testRhs, testX);
            testRhs += testX;
        }

        MPI_Barrier(comm);
        chrono_debug.Stop();

        if (verbose)
           std::cout << "Smoother at level " << l << "  has finished in " << chrono_debug.RealTime() << " \n" << std::flush;

        if (verbose)
        {
           std::cout << "Internal timing of the smoother at level " << l << ": \n";
           std::cout << "global mult time: " << ((HcurlGSSSmoother*)Smoothers_lvls[l])->GetGlobalMultTime() << " \n" << std::flush;
           std::cout << "internal mult time: " << ((HcurlGSSSmoother*)Smoothers_lvls[l])->GetInternalMultTime() << " \n" << std::flush;
           std::cout << "before internal mult time: " << ((HcurlGSSSmoother*)Smoothers_lvls[l])->GetBeforeIntMultTime() << " \n" << std::flush;
           std::cout << "after internal mult time: " << ((HcurlGSSSmoother*)Smoothers_lvls[l])->GetAfterIntMultTime() << " \n" << std::flush;
        }
        MPI_Barrier(comm);

    }
    for (int l = 0; l < num_levels - 1; ++l)
        ((HcurlGSSSmoother*)Smoothers_lvls[l])->ResetInternalTimings();
#endif
#endif

    if (verbose)
        std::cout << "End of the creating a hierarchy of meshes AND pfespaces \n";

    ParGridFunction * sigma_exact_finest;
    sigma_exact_finest = new ParGridFunction(R_space_lvls[0]);
    sigma_exact_finest->ProjectCoefficient(*Mytest.sigma);
    Vector sigma_exact_truedofs(R_space_lvls[0]->GetTrueVSize());
    sigma_exact_finest->ParallelProject(sigma_exact_truedofs);

    ParGridFunction * S_exact_finest;
    Vector S_exact_truedofs;
    S_exact_finest = new ParGridFunction(H_space_lvls[0]);
    S_exact_finest->ProjectCoefficient(*Mytest.scalarS);
    S_exact_truedofs.SetSize(H_space_lvls[0]->GetTrueVSize());
    S_exact_finest->ParallelProject(S_exact_truedofs);

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
        gform->AddDomainIntegrator(new DomainLFIntegrator(*Mytest.scalardivsigma));
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
        sigma_exact->ProjectCoefficient(*Mytest.sigma);

        gform = new ParLinearForm(W_space);
        gform->AddDomainIntegrator(new DomainLFIntegrator(*Mytest.scalardivsigma));
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

#ifndef USE_CURLMATRIX
    shared_ptr<mfem::HypreParMatrix> A;
    ParBilinearForm *Ablock;
    ParLinearForm *ffform;
#endif
    int numblocks = 2;

    Array<int> block_offsets(numblocks + 1); // number of variables + 1
    block_offsets[0] = 0;
    block_offsets[1] = C_space->GetVSize();
    block_offsets[2] = H_space->GetVSize();
    block_offsets.PartialSum();

    Array<int> block_trueOffsets(numblocks + 1); // number of variables + 1
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = C_space->TrueVSize();
    block_trueOffsets[2] = H_space->TrueVSize();
    block_trueOffsets.PartialSum();

    HYPRE_Int dimC = C_space->GlobalTrueVSize();
    HYPRE_Int dimR = R_space->GlobalTrueVSize();
    HYPRE_Int dimH = H_space->GlobalTrueVSize();
    if (verbose)
    {
       std::cout << "***********************************************************\n";
       std::cout << "dim(C) = " << dimC << "\n";
       std::cout << "dim(H) = " << dimH << ", ";
       std::cout << "dim(C+H) = " << dimC + dimH << "\n";
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
    //ConstantCoefficient minusone(-1.0);
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
    ParGridFunction *u_exact = new ParGridFunction(C_space);
    u_exact->ProjectCoefficient(*Mytest.divfreepart);

    ParGridFunction *S_exact = new ParGridFunction(H_space);
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
    xblks.GetBlock(1) = *S_exact;

#ifdef USE_CURLMATRIX
    if (verbose)
        std::cout << "Creating div-free system using the explicit discrete div-free operator \n";

    ParGridFunction* rhside_Hdiv = new ParGridFunction(R_space);  // rhside for the first equation in the original cfosls system
    *rhside_Hdiv = 0.0;
    ParGridFunction* rhside_H1 = new ParGridFunction(H_space);    // rhside for the second eqn in div-free system
    *rhside_H1 = 0.0;

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

    /*
    std::cout << "Comparing M in the old code and (0,0) block of the Functional \n";
    {
        SparseMatrix diag1;
        M->GetDiag(diag1);

        SparseMatrix diag2;
        (*Funct_hpmat_lvls[0])(0,0)->GetDiag(diag2);

        SparseMatrix diag2_copy(diag2);
        diag2_copy.Add(-1.0, diag1);

        //diag2_copy.Print();
        std::cout << "diag(M) - Funct_00 norm = " << diag2_copy.MaxNorm() << "\n";
    }
    */

    // div-free operator matrix (curl in 3D, divskew in 4D)
    // either as DivfreeT_dop * M * Divfree_dop
    auto A = RAP(Divfree_dop, M, Divfree_dop);
    //A->CopyRowStarts();
    //A->CopyColStarts();

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

    /*
    {
        SparseMatrix diag1;
        A->GetDiag(diag1);

        HypreParMatrix * A_Funct = RAP(Divfree_hpmat_mod_lvls[0], (*Funct_hpmat_lvls[0])(0,0), Divfree_hpmat_mod_lvls[0] );

        SparseMatrix diag2;
        A_Funct->GetDiag(diag2);

        SparseMatrix diag2_copy(diag2);
        diag2_copy.Add(-1.0, diag1);

        //diag2_copy.Print();
        std::cout << "diag(A) - diag(CT Funct_0 C) norm = " << diag2_copy.MaxNorm() << "\n";
    }
    MPI_Finalize();
    return 0;
    */

    /*
    ParBilinearForm *Checkblock(new ParBilinearForm(C_space_lvls[0]));
    //Checkblock->AddDomainIntegrator(new CurlCurlIntegrator);
    Checkblock->AddDomainIntegrator(new CurlCurlIntegrator(*Mytest.Ktilda));
#ifdef WITH_PENALTY
    Checkblock->AddDomainIntegrator(new VectorFEMassIntegrator(reg_coeff));
#endif
    Checkblock->Assemble();
    {
        Vector temp1(Checkblock->Width());
        temp1 = 0.0;
        Vector temp2(Checkblock->Width());
        Checkblock->EliminateEssentialBC(ess_bdrSigma, temp1, temp2);
    }
    Checkblock->Finalize();
    auto A = Checkblock->ParallelAssemble();
    */

    // diagonal block for H^1
    ParBilinearForm * Cblock;
    Cblock = new ParBilinearForm(H_space);
    // integrates ((-q, grad_x q)^T, (-p, grad_x p)^T)
    Cblock->AddDomainIntegrator(new WaveVectorFEIntegrator);
    Cblock->Assemble();
    Cblock->EliminateEssentialBC(ess_bdrS, xblks.GetBlock(1), *rhside_H1);
    Cblock->Finalize();
    HypreParMatrix * C = Cblock->ParallelAssemble();

    // off-diagonal block for (H(div), H1) block
    ParMixedBilinearForm *Bblock(new ParMixedBilinearForm(H_space, R_space));
    Bblock->AddDomainIntegrator(new WaveVectorFEIntegratorB);
    Bblock->Assemble();
    Bblock->EliminateTrialDofs(ess_bdrS, xblks.GetBlock(1), *rhside_Hdiv);
    Bblock->EliminateTestDofs(ess_bdrSigma);
    Bblock->Finalize();
    auto B = Bblock->ParallelAssemble();
    auto BT = B->Transpose();

    auto CHT = ParMult(DivfreeT_dop, B);
    CHT->CopyColStarts();
    CHT->CopyRowStarts();
    auto CH = CHT->Transpose();

    delete Cblock;
    delete Bblock;

    // additional temporary vectors on true dofs required for various matvec
    Vector tempHdiv_true(R_space->TrueVSize());
    Vector temp2Hdiv_true(R_space->TrueVSize());

    // assembling local rhs vectors from inhomog. boundary conditions
    rhside_H1->ParallelAssemble(trueRhs.GetBlock(1));
    rhside_Hdiv->ParallelAssemble(tempHdiv_true);
    DivfreeT_dop->Mult(tempHdiv_true, trueRhs.GetBlock(0));

    // subtracting from rhs a part from Sigmahat
    Sigmahat->ParallelProject(tempHdiv_true);
    M->Mult(tempHdiv_true, temp2Hdiv_true);
    //DivfreeT_dop->Mult(temp2Hdiv_true, tempHcurl_true);
    //trueRhs.GetBlock(0) -= tempHcurl_true;
    DivfreeT_dop->Mult(-1.0, temp2Hdiv_true, 1.0, trueRhs.GetBlock(0));

    // subtracting from rhs for S a part from Sigmahat
    //BT->Mult(tempHdiv_true, tempH1_true);
    //trueRhs.GetBlock(1) -= tempH1_true;
    BT->Mult(-1.0, tempHdiv_true, 1.0, trueRhs.GetBlock(1));

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
            trueRhs.GetBlock(blk)[tdof] = 0.0;
        }
    }

    // setting block operator of the system
    MainOp->SetBlock(0,0, A);
    MainOp->SetBlock(0,1, CHT);
    MainOp->SetBlock(1,0, CH);
    MainOp->SetBlock(1,1, C);

#else // if using the integrators for creating the div-free system
    if (verbose)
        std::cout << "This case is not supported any more \n";
    MPI_Finalize();
    return -1;
#endif

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
            else // prec is AMS-like for the div-free part (block-diagonal for the system with boomerAMG for S)
            {
                if (dim == 3)
                {
                    prec = new BlockDiagonalPreconditioner(block_trueOffsets);
                    Operator * precU = new HypreAMS(*A, C_space);
                    ((HypreAMS*)precU)->SetSingularProblem();
                    Operator * precS = new HypreBoomerAMG(*C);
                    ((HypreBoomerAMG*)precS)->SetPrintLevel(0);

                    ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(0, precU);
                    ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(1, precS);
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

#ifndef COMPARE_MG

    CGSolver solver(comm);
    if (verbose)
        cout << "Linear solver: CG" << endl << flush;

    solver.SetAbsTol(sqrt(atol));
    solver.SetRelTol(sqrt(rtol));
    solver.SetMaxIter(max_num_iter);
    solver.SetOperator(*MainOp);

    if (with_prec)
        solver.SetPreconditioner(*prec);
    solver.SetPrintLevel(1);
    trueX = 0.0;

    chrono.Clear();
    chrono.Start();
    solver.Mult(trueRhs, trueX);
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
    S = new ParGridFunction(H_space);
    S->Distribute(&(trueX.GetBlock(1)));

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

    double norm_S;
    S_exact = new ParGridFunction(H_space);
    S_exact->ProjectCoefficient(*Mytest.scalarS);

    double err_S = S->ComputeL2Error(*(Mytest.scalarS), irs);
    norm_S = ComputeGlobalLpNorm(2, *(Mytest.scalarS), *pmesh, irs);
    if (verbose)
    {
        if ( norm_S > MYZEROTOL )
            std::cout << "|| S_h - S_ex || / || S_ex || = " <<
                     err_S / norm_S << "\n";
        else
            std::cout << "|| S_h || = " << err_S << " (S_ex = 0) \n";
    }


    ParFiniteElementSpace * GradSpace;
    if (dim == 3)
        GradSpace = C_space;
    else // dim == 4
    {
        FiniteElementCollection *hcurl_coll;
        hcurl_coll = new ND1_4DFECollection;
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

#ifdef USE_CURLMATRIX
    // Check value of functional and mass conservation
    {
        Vector trueSigma(R_space->TrueVSize());
        trueSigma = 0.0;
        sigma->ParallelProject(trueSigma);

        Vector MtrueSigma(R_space->TrueVSize());
        MtrueSigma = 0.0;
        M->Mult(trueSigma, MtrueSigma);
        double localFunctional = trueSigma*MtrueSigma;

        Vector GtrueSigma(H_space->TrueVSize());
        GtrueSigma = 0.0;
        BT->Mult(trueSigma, GtrueSigma);
        localFunctional += 2.0*(trueX.GetBlock(1)*GtrueSigma);

        Vector XtrueS(H_space->TrueVSize());
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
    }
#endif


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
    double projection_error_S = S_exact->ComputeL2Error(*Mytest.scalarS, irs);

    if(verbose)
    {
       if ( norm_S > MYZEROTOL )
           cout << "|| S_ex - Pi_h S_ex || / || S_ex || = " << projection_error_S / norm_S << endl;
       else
           cout << "|| Pi_h S_ex ||  = " << projection_error_S << " (S_ex = 0) \n";
    }

    chrono.Stop();
    if (verbose)
        std::cout << "Errors in the MG code were computed in "<< chrono.RealTime() <<" seconds.\n";

    //MPI_Finalize();
    //return 0;
#endif // for #ifndef COMPARE_MG

#endif // for #ifdef OLD_CODE

    chrono.Clear();
    chrono.Start();

    if (verbose)
        std::cout << "\nCreating an instance of the new Hcurl smoother and the minimization solver \n";

    //ParLinearForm *fform = new ParLinearForm(R_space);

    ParLinearForm * constrfform = new ParLinearForm(W_space_lvls[0]);
    constrfform->AddDomainIntegrator(new DomainLFIntegrator(*Mytest.scalardivsigma));
    constrfform->Assemble();

    ParMixedBilinearForm *Bblock2(new ParMixedBilinearForm(R_space, W_space));
    Bblock2->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
    Bblock2->Assemble();
    //Bblock2->EliminateTrialDofs(ess_bdrSigma, *sigma_exact_finest, *constrfform); // // makes res for sigma_special happier
    Bblock2->Finalize();

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
                                      EssBdrTrueDofs_Funct_lvls,
                                      Funct_global_lvls,
                                      *Constraint_global,
                                      Floc,
                                      Smoothers_lvls,
                                      Xinit_truedofs,
#ifdef CHECK_CONSTR
                                      Floc,
#endif
                                      LocalSolver_partfinder_lvls,
                                      CoarsestSolver_partfinder);
    CoarsestSolver_partfinder->SetMaxIter(70000);
    CoarsestSolver_partfinder->SetAbsTol(1.0e-18);
    CoarsestSolver_partfinder->SetRelTol(1.0e-18);
    CoarsestSolver_partfinder->ResetSolverParams();
#endif

    GeneralMinConstrSolver NewSolver( comm, num_levels,
                     TrueP_Func, EssBdrTrueDofs_Funct_lvls,
                     *Functrhs_global, Smoothers_lvls,
                     Xinit_truedofs, Funct_global_lvls,
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
    chrono.Clear();
    chrono.Start();

    PartsolFinder.Mult(Xinit_truedofs, ParticSol);
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
#ifdef COMPARE_MG
    ((CoarsestProblemHcurlSolver*)CoarsestSolver)->SetMaxIter(NCOARSEITER);
#else
    ((CoarsestProblemHcurlSolver*)CoarsestSolver)->SetMaxIter(100);
#endif
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
    ConstantCoefficient zero_coeff(0.0);
    fformtest->Assemble();

    ParLinearForm *qformtest;
    qformtest = new ParLinearForm(H_space_lvls[0]);
    qformtest->Assemble();
    //*qformtest = 0.0;

    ParBilinearForm *Ablocktest = new ParBilinearForm(R_space_lvls[0]);
    HypreParMatrix *Atest;
    Ablocktest->AddDomainIntegrator(new VectorFEMassIntegrator);
    Ablocktest->Assemble();
    Ablocktest->EliminateEssentialBC(ess_bdrSigma, *sigma_exact_finest, *fformtest);
    Ablocktest->Finalize();
    Atest = Ablocktest->ParallelAssemble();

    delete Ablocktest;

    HypreParMatrix *Ctest;
    ParBilinearForm * Cblocktest = new ParBilinearForm(H_space_lvls[0]);
    Cblocktest->AddDomainIntegrator(new WaveVectorFEIntegrator);
    Cblocktest->Assemble();
    Cblocktest->EliminateEssentialBC(ess_bdrS, *S_exact_finest, *qformtest);
    Cblocktest->Finalize();

    Ctest = Cblocktest->ParallelAssemble();

    HypreParMatrix *Dtest;
    HypreParMatrix *DTtest;

    ParMixedBilinearForm *DTblocktest = new ParMixedBilinearForm(H_space_lvls[0], R_space_lvls[0]);
    DTblocktest->AddDomainIntegrator(new WaveVectorFEIntegratorB);
    DTblocktest->Assemble();
    DTblocktest->EliminateTrialDofs(ess_bdrS, *S_exact_finest, *fformtest);
    DTblocktest->EliminateTestDofs(ess_bdrSigma);
    DTblocktest->Finalize();

    DTtest = DTblocktest->ParallelAssemble();
    Dtest = DTtest->Transpose();

    Array<int> blocktest_offsets(numblocks_funct + 1);
    blocktest_offsets[0] = 0;
    blocktest_offsets[1] = Atest->Height();
    blocktest_offsets[2] = Ctest->Height();
    blocktest_offsets.PartialSum();

    BlockVector trueXtest(blocktest_offsets);
    BlockVector trueRhstest(blocktest_offsets);
    trueRhstest = 0.0;

    //fformtest->ParallelAssemble(trueRhstest.GetBlock(0));
    //qformtest->ParallelAssemble(trueRhstest.GetBlock(1));

    //trueRhstest.Print();

    BlockOperator *BlockMattest = new BlockOperator(blocktest_offsets);
    BlockMattest->SetBlock(0,0, Atest);
    BlockMattest->SetBlock(0,1, DTtest);
    BlockMattest->SetBlock(1,0, Dtest);
    BlockMattest->SetBlock(1,1, Ctest);

    NewSolver.SetAsPreconditioner(true);
    NewSolver.SetPrintLevel(0);
    if (verbose)
        NewSolver.PrintAllOptions();

#ifdef  COMPARE_MG
    if (verbose)
        std::cout << "\nComparing geometric MG with modified new MG (w/o Schwarz smoother) \n";

    //MFEM_ASSERT(strcmp(space_for_S,"L2") == 0, "Right now the check works only for S in L2 case!\n");
    //MFEM_ASSERT(num_procs == 1, "Right now the check operates only in serial case \n");
    //MFEM_ASSERT(num_levels == 2, "Check works only for 2-level case \n");

    Array<int> offsets_new(numblocks_funct + 1);
    offsets_new = 0;
    for (int blk = 0; blk < numblocks_funct; ++blk)
        offsets_new[blk + 1] = (*Funct_hpmat_lvls[0])(blk,blk)->Height();
    offsets_new.PartialSum();

    BlockVector inFunctvec(offsets_new);
    inFunctvec.GetBlock(0) = sigma_exact_truedofs;
    inFunctvec.GetBlock(1) = S_exact_truedofs;

    for (int blk = 0; blk < numblocks_funct; ++blk)
    {
        const Array<int> *temp = EssBdrTrueDofs_Funct_lvls[0][blk];
        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            if ( fabs(inFunctvec.GetBlock(blk)[(*temp)[tdofind]]) > 1.0e-14 )
                std::cout << "bnd cnd is violated for inFunctvec, blk = " << blk << ",  value = "
                          << inFunctvec.GetBlock(blk)[(*temp)[tdofind]]
                          << ", index = " << (*temp)[tdofind] << "\n";
        }
    }
    /*
    Vector inHdivvec(NewSolver.Width());
    inHdivvec = sigma_exact_truedofs;

#ifdef CHECK_BNDCND
    {
        const Array<int> *temp = EssBdrTrueDofs_Funct_lvls[0][0];
        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            if ( fabs(inHdivvec[(*temp)[tdofind]]) > 1.0e-14 )
                std::cout << "bnd cnd is violated for inHdivvec, value = "
                          << inHdivvec[(*temp)[tdofind]]
                          << ", index = " << (*temp)[tdofind] << "\n";
        }
    }
#endif
    */

    Array<int> offsets_hcurlfunct_new(numblocks_funct + 1);
    offsets_hcurlfunct_new = 0;
    for (int blk = 0; blk < numblocks_funct; ++blk)
        if (blk == 0)
            offsets_hcurlfunct_new[blk + 1] = Divfree_hpmat_mod_lvls[0]->Width();
        else
            offsets_hcurlfunct_new[blk + 1] = (*Funct_hpmat_lvls[0])(blk,blk)->Height();
    offsets_hcurlfunct_new.PartialSum();

    BlockVector inFunctHcurlvec(offsets_hcurlfunct_new);
    for (int blk = 0; blk < numblocks_funct; ++blk)
        if (blk == 0)
            Divfree_hpmat_mod_lvls[0]->MultTranspose(inFunctvec.GetBlock(0), inFunctHcurlvec.GetBlock(0));
        else
            inFunctHcurlvec.GetBlock(blk) = inFunctvec.GetBlock(blk);

    for (int blk = 0; blk < numblocks_funct; ++blk)
    {
        const Array<int> *temp;
        if (blk == 0)
            temp = EssBdrTrueDofs_Hcurl[0];
        else
            temp = EssBdrTrueDofs_Funct_lvls[0][blk];
        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            if ( fabs(inFunctHcurlvec.GetBlock(blk)[(*temp)[tdofind]]) > 1.0e-14 )
                std::cout << "bnd cnd is violated for inFunctHcurlvec, blk = " << blk << ",  value = "
                          << inFunctHcurlvec.GetBlock(blk)[(*temp)[tdofind]]
                          << ", index = " << (*temp)[tdofind] << "\n";
        }
    }
    /*
     * checking the Divfree and Divfree_T operators
#ifdef CHECK_BNDCND
    MFEM_ASSERT(strcmp(space_for_S,"L2") == 0, "Right now the check works only for S in L2 case!\n");
    auto Divfree_T = Divfree_hpmat_mod_lvls[0]->Transpose();
    MPI_Barrier(comm);
    for (int i = 0; i < num_procs; ++i)
    {
        if (myid == i)
        {
            std::cout << "I am " << myid << "\n";

            const Array<int> *temp2 = EssBdrTrueDofs_Funct_lvls[0][0];

            Array<int> bndtdofs_Hdiv(R_space_lvls[0]->TrueVSize());
            bndtdofs_Hdiv = 0;
            //std::cout << "bnd tdofs Hdiv \n";
            for ( int tdofind = 0; tdofind < temp2->Size(); ++tdofind)
            {
                //std::cout << (*temp2)[tdofind] << " ";
                bndtdofs_Hdiv[(*temp2)[tdofind]] = 1;
            }
            //std::cout << "\n";


            const Array<int> *temp = EssBdrTrueDofs_Hcurl[0];

            Array<int> bndtdofs_Hcurl(C_space_lvls[0]->TrueVSize());
            bndtdofs_Hcurl = 0;
            //std::cout << "bnd tdofs Hcurl \n";
            for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
            {
                //std::cout << (*temp)[tdofind] << " ";
                bndtdofs_Hcurl[(*temp)[tdofind]] = 1;
            }
            //std::cout << "\n";

            int special_row;
            bool found = false;

            for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
            {
                if ( fabs(inHcurlvec[(*temp)[tdofind]]) > 1.0e-14 )
                {
                    std::cout << "bnd cnd is violated for inHcurlvec, value = "
                              << inHcurlvec[(*temp)[tdofind]]
                              << ", index = " << (*temp)[tdofind] << "\n";
                    std::cout << " ... was corrected \n";
                    if (found == false)
                    {
                        special_row = (*temp)[tdofind];
                        found = true;
                    }
                }
                inHcurlvec[(*temp)[tdofind]] = 0.0;
            }

            if (found)
            {
                int special_col2;
                int special_col3;
                bool found2 = false;
                bool found3 = false;

                SparseMatrix spmat;
                Divfree_T->GetDiag(spmat);

                {
                    std::cout << "Looking for incorrect values in the diag part of Divfree_T \n";
                    int row = special_row;
                    int row_shift = spmat.GetI()[row];
                    std::cout << "row = " << row << "\n";
                    for (int j = 0; j < spmat.RowSize(row); ++j)
                    {
                        int col = spmat.GetJ()[row_shift + j];
                        double value = spmat.GetData()[row_shift + j];
                        if (fabs(value) > 1.0e-14)
                        {
                            std::cout << "(" << col << ", " << value << ") ";
                            std::cout << "for hdivvec value = " << inHdivvec[col] << " ";
                            if (bndtdofs_Hdiv[col] != 0)
                                std::cout << " at the boundary! ";
                            else
                            {
                                std::cout << "not at the boundary! ";
                                found2 = true;
                                special_col2 = col;
                            }

                        }
                    }
                    std::cout << "\n";
                }

                SparseMatrix spmat_offd;
                int * cmap_offd;
                Divfree_T->GetOffd(spmat_offd, cmap_offd);

                {
                    std::cout << "Looking for incorrect values in the offd part of Divfree_T \n";
                    int row = special_row;
                    int row_shift = spmat_offd.GetI()[row];
                    std::cout << "row = " << row << "\n";
                    for (int j = 0; j < spmat_offd.RowSize(row); ++j)
                    {
                        int col = spmat_offd.GetJ()[row_shift + j];
                        int truecol = cmap_offd[col];
                        double value = spmat_offd.GetData()[row_shift + j];
                        if (fabs(value) > 1.0e-14)
                        {
                            std::cout << "col = " << col << ": (" << truecol << ", " << value << ") ";
                            //std::cout << "for hdivvec value = " << inHdivvec[col] << " ";
                            if (bndtdofs_Hdiv[col] != 0)
                                std::cout << " at the boundary! ";
                            else
                            {
                                std::cout << "not at the boundary! ";
                                found3 = true;
                                special_col3 = truecol;
                            }

                        }
                    }
                    std::cout << "\n";
                }

            }

        }
        MPI_Barrier(comm);
    }
    delete Divfree_T;

#endif // for CHECK_BND
    */

    //MPI_Finalize();
    //return 0;

    // studying why inHcurlvec has nonzeros at the boundary
    /*
    {
        std::cout << "bnd indices for Hdiv \n";
        const Array<int> *temp = EssBdrTrueDofs_Funct_lvls[0][0];
        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            std::cout << (*temp)[tdofind] << " ";
        }
        std::cout << "\n";
    }

    {
        std::cout << "bnd indices for Hcurl \n";
        const Array<int> *temp = EssBdrTrueDofs_Hcurl[0];
        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            std::cout << (*temp)[tdofind] << " ";
        }
        std::cout << "\n";
    }


    SparseMatrix spmat;
    Divfree_hpmat_mod_lvls[0]->Transpose()->GetDiag(spmat);

    {
        int row = 2;
        int row_shift = spmat.GetI()[row];
        std::cout << "row = " << row << "\n";
        for (int j = 0; j < spmat.RowSize(row); ++j)
        {
            std::cout << "(" << spmat.GetJ()[row_shift + j] << ", " << spmat.GetData()[row_shift + j] << ") ";
            std::cout << "for hdivvec value = " << inHdivvec[spmat.GetJ()[row_shift + j]] << " ";
        }
        std::cout << "\n";
    }

    SparseMatrix spmat2;
    Divfree_hpmat_mod_lvls[0]->GetDiag(spmat2);
    {
        int row = 2;
        int row_shift = spmat2.GetI()[row];
        std::cout << "row = " << row << "\n";
        for (int j = 0; j < spmat2.RowSize(row); ++j)
        {
            std::cout << "(" << spmat2.GetJ()[row_shift + j] << ", " << spmat2.GetData()[row_shift + j] << ") ";
        }
        std::cout << "\n";
    }


    MPI_Finalize();
    return 0;
    */

    //HypreParMatrix * A_coarse = ((Multigrid*) (&(((BlockDiagonalPreconditioner*)prec)->GetDiagonalBlock(0))))->GetCoarseOp();
    //Array2D<HypreParMatrix*> CoarseOperator(numblocks_funct, numblocks_funct);
    //CoarseOperator(0,0) = A_coarse;
    //((CoarsestProblemHcurlSolver*)CoarsestSolver)->SetCoarseOperator(CoarseOperator);

    BlockVector outFunctvec(offsets_new);

    if (verbose)
        std::cout << "Computing action for the new MG ... \n";
    NewSolver.Mult(inFunctvec, outFunctvec);

    for (int blk = 0; blk < numblocks_funct; ++blk)
    {
        const Array<int> *temp;
        temp = EssBdrTrueDofs_Funct_lvls[0][blk];
        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            if ( fabs(outFunctvec.GetBlock(blk)[(*temp)[tdofind]]) > 1.0e-14 )
                std::cout << "bnd cnd is violated for outFunctvec, blk = " << blk << ",  value = "
                          << outFunctvec.GetBlock(blk)[(*temp)[tdofind]]
                          << ", index = " << (*temp)[tdofind] << "\n";
        }
    }

    BlockVector outFunctHcurlvec(offsets_hcurlfunct_new);
    outFunctHcurlvec = 0.0;
    if (verbose)
        std::cout << "Computing action for the geometric MG ... \n";
    prec->Mult(inFunctHcurlvec, outFunctHcurlvec);

    for (int blk = 0; blk < numblocks_funct; ++blk)
    {
        const Array<int> *temp;
        if (blk == 0)
            temp = EssBdrTrueDofs_Hcurl[0];
        else
            temp = EssBdrTrueDofs_Funct_lvls[0][blk];
        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            if ( fabs(outFunctHcurlvec.GetBlock(blk)[(*temp)[tdofind]]) > 1.0e-14 )
                std::cout << "bnd cnd is violated for outFunctHcurlvec, blk = " << blk << ",  value = "
                          << outFunctHcurlvec.GetBlock(blk)[(*temp)[tdofind]]
                          << ", index = " << (*temp)[tdofind] << "\n";
        }
    }

    BlockVector out2Functvec(offsets_new);
    for (int blk = 0; blk < numblocks_funct; ++blk)
        if (blk == 0)
            Divfree_hpmat_mod_lvls[0]->Mult(outFunctHcurlvec.GetBlock(0), out2Functvec.GetBlock(0));
        else
            out2Functvec.GetBlock(blk) = outFunctHcurlvec.GetBlock(blk);

    BlockVector diff(offsets_new);
    diff = outFunctvec;
    diff -= out2Functvec;

    /*
    std::cout << "blk 0 \n";
    diff.GetBlock(0).Print();
    if (numblocks_funct > 1)
    {
        std::cout << "blk 1 \n";
        diff.GetBlock(1).Print();
    }
    */

    double diff_norm = diff.Norml2() / sqrt (diff.Size());
    double geommg_norm = out2Functvec.Norml2() / sqrt(out2Functvec.Size());
    if (verbose)
    {
        std::cout << "|| NewMG * vec - C MG * C^T vec || = " << diff_norm << "\n";
        std::cout << "|| NewMG * vec - C MG * C^T vec || / || C MG * C^T vec || = " << diff_norm / geommg_norm << "\n";
    }

    for (int blk = 0; blk < numblocks_funct; ++blk)
    {
        double diffblk_norm = diff.GetBlock(blk).Norml2() / sqrt (diff.GetBlock(blk).Size());
        if (verbose)
        {
            std::cout << "|| NewMG * vec - C MG * C^T vec ||, block " << blk << " = " << diffblk_norm << "\n";
        }
    }

    // checking that A is exactly CT Funct_0 C in serial
    SparseMatrix diag1;
    A->GetDiag(diag1);

    HypreParMatrix * A_Funct = RAP(Divfree_hpmat_mod_lvls[0], (*Funct_hpmat_lvls[0])(0,0), Divfree_hpmat_mod_lvls[0] );

    SparseMatrix diag2;
    A_Funct->GetDiag(diag2);

    SparseMatrix diag2_copy(diag2);
    diag2_copy.Add(-1.0, diag1);

    //diag2_copy.Print();
    std::cout << "diag(A) - diag(CT Funct_0 C) norm = " << diag2_copy.MaxNorm() << "\n";

    // checking that A has 1's on the diagonal and 0's for other columns for boundary entries
    /*
    MPI_Barrier(comm);
    for (int i = 0; i < num_procs; ++i)
    {
        if (myid == i)
        {
            std::cout << "I am " << myid << "\n";

            const Array<int> *temp = EssBdrTrueDofs_Hcurl[0];

            Array<int> bndtdofs(C_space_lvls[0]->TrueVSize());
            bndtdofs = 0;
            for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
            {
                //std::cout << (*temp)[tdofind] << " ";
                bndtdofs[(*temp)[tdofind]] = 1;
            }
            //std::cout << "\n";

            //if (verbose)
                //bndtdofs.Print();

            int tdof_offset = C_space_lvls[0]->GetMyTDofOffset();

            SparseMatrix diag;
            A->GetDiag(diag);

            std::cout << "Checking diagonal part of A in geom mg \n";
            for (int row = 0; row < diag.Height(); ++row)
            {
                if ( bndtdofs[row + tdof_offset] != 0)
                {
                    int nnz_shift = diag.GetI()[row];
                    for (int j = 0; j < diag.RowSize(row); ++j)
                    {
                        int col = diag.GetJ()[nnz_shift + j];
                        if ( col != row && fabs(diag.GetData()[nnz_shift + j]) > 1.0e-14 )
                        {
                            if (bndtdofs[col + tdof_offset] == 0)
                            {
                                std::cout << "Found nonzero for the boundary row = " << row << "(" << col << ", " << diag.GetData()[nnz_shift + j] << ") \n";
                                std::cout << "which lives not on the boundary! \n";
                            }
                        }
                    }
                } // end of if row is for the boundary tdof
            }// end of loop over rows
        }
        MPI_Barrier(comm);
    }
    MPI_Finalize();
    return 0;
    */

#ifdef COMPARE_SMOOTHERS

    if (verbose)
        std::cout << " \nComparing separately smoothers \n";

    for (int l = 0; l < num_levels - 1; ++l)
    {
        if (verbose)
            std::cout << "level: " << l << "\n";

        BlockVector outSmooHdivvec(offsets_new);
        Smoothers_lvls[0]->Mult(inFunctvec, outSmooHdivvec);

        //std::cout << "inFunctvec \n";
        //inFunctvec.Print();

        //std::cout << "outSmooHdivvec\n";
        //outSmooHdivvec.Print();

        /*
        inFunctvec = outSmooHdivvec; // iter no 2

        //std::cout << "outSmooHdivvec after the 1st iteration \n";
        //outSmooHdivvec.Print();

        Smoothers_lvls[0]->Mult(inFunctvec, outSmooHdivvec);
        */

        //Vector outSmooHdivvec(Smoothers_lvls[0]->Height());
        //Smoothers_lvls[0]->Mult(inHdivvec, outSmooHdivvec);

        HypreSmoother * Smoothers_fromMG_0 = new HypreSmoother(*A, HypreSmoother::Type::l1GS, 1);
        HypreSmoother * Smoothers_fromMG_1;
        if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
            Smoothers_fromMG_1 = new HypreSmoother(*C, HypreSmoother::Type::l1GS, 1);

        BlockVector outSmooHcurlvec(offsets_hcurlfunct_new);
        Smoothers_fromMG_0->Mult(inFunctHcurlvec.GetBlock(0), outSmooHcurlvec.GetBlock(0));
        if (strcmp(space_for_S,"H1") == 0 || !eliminateS) // S is present
            Smoothers_fromMG_1->Mult(inFunctHcurlvec.GetBlock(1), outSmooHcurlvec.GetBlock(1));

        /*
        //std::cout << "outSmooHdivvec after the 1st iteration \n";
        //outSmooHdivvec.Print();

        inFunctHcurlvec = outSmooHcurlvec; // iter no 2
        Smoothers_fromMG_0->Mult(inFunctHcurlvec.GetBlock(0), outSmooHcurlvec.GetBlock(0));
        if (strcmp(space_for_S,"H1") == 0 || !eliminateS)
            Smoothers_fromMG_1->Mult(inFunctHcurlvec.GetBlock(1), outSmooHcurlvec.GetBlock(1));
        */

        //Vector outSmooHcurlvec(Smoothers_fromMG_0->Height());
        //Smoothers_fromMG_0->Mult(inHcurlvec, outSmooHcurlvec);

#ifdef CHECK_BNDCND
        for (int blk = 0; blk < numblocks_funct; ++blk)
        {
            const Array<int> *temp;
            if (blk == 0)
                temp = EssBdrTrueDofs_Hcurl[0];
            else
                temp = EssBdrTrueDofs_Funct_lvls[0][blk];

            for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
            {
                //std::cout << "index = " << (*temp)[tdofind] << "\n";
                if ( fabs(outSmooHcurlvec.GetBlock(blk)[(*temp)[tdofind]]) > 1.0e-14 )
                {
                    std::cout << "bnd cnd is violated for outSmooHcurlvec, blk = " << blk << ", value = "
                              << outSmooHcurlvec.GetBlock(blk)[(*temp)[tdofind]]
                              << ", index = " << (*temp)[tdofind] << "\n";
                    //std::cout << "... was corrected \n";
                    //outSmooHcurlvec.GetBlock(blk)[(*temp)[tdofind]] = 0.0;
                }
            }
        }
#endif

        BlockVector out2SmooHdivvec(offsets_new);
        for (int blk = 0; blk < numblocks_funct; ++blk)
        {
            if (blk == 0)
                Divfree_hpmat_mod_lvls[0]->Mult(outSmooHcurlvec.GetBlock(0), out2SmooHdivvec.GetBlock(0));
            else
                out2SmooHdivvec.GetBlock(blk) = outSmooHcurlvec.GetBlock(blk);
        }

        //Vector out2SmooHdivvec(Divfree_hpmat_mod_lvls[0]->Height());
        //Divfree_hpmat_mod_lvls[0]->Mult(outSmooHcurlvec, out2SmooHdivvec);

        BlockVector diffsmoo(offsets_new);
        //Vector diffsmoo(R_space_lvls[0]->TrueVSize());
        diffsmoo = outSmooHdivvec;
        diffsmoo -= out2SmooHdivvec;

        MPI_Barrier(comm);
        for (int i = 0; i < num_procs; ++i)
        {
            if (myid == i)
            {
                std::cout << "I am " << myid << "\n";

                double diffsmoo_norm = diffsmoo.Norml2() / sqrt (diffsmoo.Size());
                double geommgsmoo_norm = out2SmooHdivvec.Norml2() / sqrt(out2SmooHdivvec.Size());
                std::cout << "|| diff of smoothers action || = " << diffsmoo_norm << "\n";
                std::cout << "|| diff of smoothers action || / || geommg smoother action || = " << diffsmoo_norm / geommgsmoo_norm << "\n";
            }
            MPI_Barrier(comm);
        }
    }

    //MPI_Finalize();
    //return 0;

#endif

#if 0
    HypreParMatrix * prod1 = ParMult(Divfree_hpmat_mod_lvls[0], TrueP_C[0]);
    SparseMatrix diag_prod1;
    prod1->GetDiag(diag_prod1);

    HypreParMatrix * prod2 = ParMult(TrueP_R[0], Divfree_hpmat_mod_lvls[1]);
    SparseMatrix diag_prod2;
    prod2->GetDiag(diag_prod2);

    SparseMatrix diag_prod2_copy(diag_prod2);
    diag_prod2_copy.Add(-1.0, diag_prod1);

    //diag_prod2.Print();
    MPI_Barrier(comm);
    for (int i = 0; i < num_procs; ++i)
    {
        if (myid == i)
        {
            const Array<int> *temp2 = EssBdrTrueDofs_Funct_lvls[0][0];

            Array<int> bndtdofs_Hdiv(R_space_lvls[0]->TrueVSize());
            bndtdofs_Hdiv = 0;
            //std::cout << "bnd tdofs Hdiv \n";
            for ( int tdofind = 0; tdofind < temp2->Size(); ++tdofind)
            {
                //std::cout << (*temp2)[tdofind] << " ";
                bndtdofs_Hdiv[(*temp2)[tdofind]] = 1;
            }
            //std::cout << "\n";


            const Array<int> *temp = EssBdrTrueDofs_Hcurl[1];

            Array<int> bndtdofs_Hcurl(C_space_lvls[1]->TrueVSize());
            bndtdofs_Hcurl = 0;
            //std::cout << "bnd tdofs Hcurl \n";
            for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
            {
                bndtdofs_Hcurl[(*temp)[tdofind]] = 1;
            }

            std::cout << "I am " << myid << "\n";

            for (int i = 0; i < diag_prod2_copy.Height(); ++i)
            {
                for (int j = 0; j < diag_prod2_copy.RowSize(i); ++j)
                {
                    int col = diag_prod2_copy.GetJ()[diag_prod2_copy.GetI()[i] + j];
                    if (fabs(diag_prod2_copy.GetData()[diag_prod2_copy.GetI()[i] + j]) > 1.0e-13)
                    {
                        if (!(bndtdofs_Hdiv[i] != 0 && bndtdofs_Hcurl[col] != 0) )
                        {
                            std::cout << "nonzero entry of type ";
                            if (bndtdofs_Hdiv[i] != 0)
                                std::cout << "b-";
                            else
                                std::cout << "i-";
                            if (bndtdofs_Hcurl[col] != 0)
                                std::cout << "-b ";
                            else
                                std::cout << "-i ";
                            std::cout << ": (" << i << ", " << col << ", " << diag_prod1.GetData()[diag_prod1.GetI()[i] + j] << ") vs ";
                            std::cout << " (" << i << ", " << col << ", " << diag_prod2.GetData()[diag_prod2.GetI()[i] + j] << ") \n";
                        }
                        else
                        {
                            std::cout << "for bb nonzero entry, (" << i << ", " << col << ", " << diag_prod1.GetData()[diag_prod1.GetI()[i] + j] << ") vs ";
                            std::cout << " (" << i << ", " << col << ", " << diag_prod2.GetData()[diag_prod2.GetI()[i] + j] << ") \n";
                        }
                    }
                }
            }
        }
        MPI_Barrier(comm);
    }

    if (verbose)
        std::cout << "diag(P_R C1) - diag(C_0 P_C) norm = " << diag_prod2_copy.MaxNorm() << "\n";
#endif // for #if 0

#ifdef COMPARE_COARSE_SOLVERS
    if (verbose)
        std::cout << " \nComparing separately coarse level solvers \n";

    /*
    if (verbose)
        std::cout << " \nComparing coarsest level matrices \n";
    {
        SparseMatrix diag1;
        //A->GetDiag(diag1);
        HypreParMatrix * A_coarse = ((Multigrid*) (&(((BlockDiagonalPreconditioner*)prec)->GetDiagonalBlock(0))))->GetCoarseOp();
        A_coarse->GetDiag(diag1);

        SparseMatrix diag2;
        //std::cout << "size of Divfree_hpmat_mod_lvls[1] = " << Divfree_hpmat_mod_lvls[0]->Height() << " x " << Divfree_hpmat_mod_lvls[0]->Width() << "\n";
        //std::cout << "size of (*Funct_hpmat_lvls[1])(0,0)] = " << (*Funct_hpmat_lvls[1])(0,0)->Height() << " x " << (*Funct_hpmat_lvls[1])(0,0)->Width() << "\n";
        HypreParMatrix * HcurlOp = RAP(Divfree_hpmat_mod_lvls[1], (*Funct_hpmat_lvls[1])(0,0), Divfree_hpmat_mod_lvls[1]);
        HcurlOp->GetDiag(diag2);

        diag2.Add(-1.0, diag1);

        if (verbose)
            std::cout << "diag2 - diag1 norm = " << diag2.MaxNorm() << "\n";

        HypreParMatrix * tempm = RAP(TrueP_R[0], (*Funct_hpmat_lvls[0])(0,0), TrueP_R[0] );

        HypreParMatrix * HcurlOp_2 = RAP(Divfree_hpmat_mod_lvls[1], tempm, Divfree_hpmat_mod_lvls[1]);

        SparseMatrix diag3;
        HcurlOp_2->GetDiag(diag3);

        diag3.Add(-1.0, diag1);

        if (verbose)
            std::cout << "diag3 - diag1 norm = " << diag3.MaxNorm() << "\n";

        HypreParMatrix * tempmm = RAP(Divfree_hpmat_mod_lvls[0], (*Funct_hpmat_lvls[0])(0,0), Divfree_hpmat_mod_lvls[0]);
        HypreParMatrix * HcurlOp_3 = RAP(TrueP_C[0], tempmm, TrueP_C[0] );

        SparseMatrix diag4;
        HcurlOp_3->GetDiag(diag4);

        diag4.Add(-1.0, diag1);

        if (verbose)
            std::cout << "diag4 - diag1 norm = " << diag4.MaxNorm() << "\n";

        MPI_Finalize();
        return 0;
    }
    */

    // comparison at the coarsest level
    /*
    Vector inCoarseHdivvec(CoarsestSolver->Width());
    TrueP_R[0]->MultTranspose(inHdivvec, inCoarseHdivvec); // project

    {
        const Array<int> *temp = EssBdrTrueDofs_Funct_lvls[1][0];
        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            inCoarseHdivvec[(*temp)[tdofind]] = 0.0;
        }

    }

#ifdef CHECK_BNDCND
    {
        const Array<int> *temp = EssBdrTrueDofs_Funct_lvls[1][0];
        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            if ( fabs(inCoarseHdivvec[(*temp)[tdofind]]) > 1.0e-14 )
            {
                std::cout << "bnd cnd is violated for inCoarseHdivvec, value = "
                          << inCoarseHdivvec[(*temp)[tdofind]]
                          << ", index = " << (*temp)[tdofind] << "\n";
                std::cout << " ... was corrected \n";
            }
            inCoarseHdivvec[(*temp)[tdofind]] = 0.0;
        }

    }
#endif

    Vector outCoarseHdivvec(CoarsestSolver->Height());
    CoarsestSolver->Mult(inCoarseHdivvec, outCoarseHdivvec); // coarse solve

    Vector inCoarseHcurlvec( Divfree_hpmat_mod_lvls[1]->Width());
    Divfree_hpmat_mod_lvls[1]->MultTranspose(inCoarseHdivvec, inCoarseHcurlvec); // move to coarse Hcurl

#ifdef CHECK_BNDCND
    {
        const Array<int> *temp = EssBdrTrueDofs_Hcurl[1];
        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            if ( fabs(inCoarseHcurlvec[(*temp)[tdofind]]) > 1.0e-14 )
            {
                std::cout << "bnd cnd is violated for inCoarseHcurlvec, value = "
                          << inCoarseHcurlvec[(*temp)[tdofind]]
                          << ", index = " << (*temp)[tdofind] << "\n";
                std::cout << " ... was corrected \n";
            }
            inCoarseHcurlvec[(*temp)[tdofind]] = 0.0;
        }
    }
#endif
    CGSolver * Geommg_Coarsesolver = ((Multigrid*) (&(((BlockDiagonalPreconditioner*)prec)->GetDiagonalBlock(0))))->GetCoarseSolver();
    Vector outCoarseHcurlvec(Geommg_Coarsesolver->Height());
    Geommg_Coarsesolver->Mult(inCoarseHcurlvec, outCoarseHcurlvec); // solve in coarse Hcurl

    Vector out2CoarseHdivvec(Divfree_hpmat_mod_lvls[1]->Height());
    Divfree_hpmat_mod_lvls[1]->Mult(outCoarseHcurlvec, out2CoarseHdivvec); // move to coarse Hdiv back

    Vector diffcoarse(R_space_lvls[1]->TrueVSize());
    diffcoarse = outCoarseHdivvec;
    diffcoarse -= out2CoarseHdivvec;

    for (int i = 0; i < diffcoarse.Size(); ++i)
        if (fabs(diffcoarse[i] > 1.0e-13))
            std::cout << "nonzero entry: (" << i << ", " << diffcoarse[i] << ") \n";

    double diffcoarse_norm = diffcoarse.Norml2() / sqrt (diffcoarse.Size());
    double geommgcoarse_norm = out2CoarseHdivvec.Norml2() / sqrt(out2CoarseHdivvec.Size());
    if (verbose)
    {
        std::cout << "|| diff of coarse solvers action || = " << diffcoarse_norm << "\n";
        std::cout << "|| diff of coarse solvers action || / || geommg coarse solver action || = " << diffcoarse_norm / geommgcoarse_norm << "\n";
    }
    */

    // comparison with transfers from and to the finest level

#ifdef CHECK_BNDCND
    {
        const Array<int> *temp = EssBdrTrueDofs_Hcurl[0];
        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            if ( fabs(inHcurlvec[(*temp)[tdofind]]) > 1.0e-14 )
            {
                std::cout << "bnd cnd is violated for inHcurlvec, value = "
                          << inHcurlvec[(*temp)[tdofind]]
                          << ", index = " << (*temp)[tdofind] << "\n";
                //std::cout << " ... was corrected \n";
            }
            //inHcurlvec[(*temp)[tdofind]] = 0.0;
        }

    }
#endif

    Vector inCoarseHdivvec(CoarsestSolver->Width());
    TrueP_R[0]->MultTranspose(inHdivvec, inCoarseHdivvec); // project

    Vector outCoarseHdivvec(CoarsestSolver->Height());
    CoarsestSolver->Mult(inCoarseHdivvec, outCoarseHdivvec); // coarse solve

#ifdef CHECK_BNDCND
    {
        const Array<int> *temp = EssBdrTrueDofs_Funct_lvls[1][0];
        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            if ( fabs(outCoarseHdivvec[(*temp)[tdofind]]) > 1.0e-14 )
            {
                std::cout << "bnd cnd is violated for outCoarseHdivvec, value = "
                          << outCoarseHdivvec[(*temp)[tdofind]]
                          << ", index = " << (*temp)[tdofind] << "\n";
                //std::cout << " ... was corrected \n";
            }
            //outCoarseHdivvec[(*temp)[tdofind]] = 0.0;
        }

    }
#endif

    Vector outFineCoarseHdivvec(TrueP_R[0]->Height());
    TrueP_R[0]->Mult(outCoarseHdivvec, outFineCoarseHdivvec); // interpolate back

#ifdef CHECK_BNDCND
    {
        const Array<int> *temp = EssBdrTrueDofs_Funct_lvls[0][0];
        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            if ( fabs(outFineCoarseHdivvec[(*temp)[tdofind]]) > 1.0e-14 )
            {
                std::cout << "bnd cnd is violated for outFineCoarseHdivvec, value = "
                          << outFineCoarseHdivvec[(*temp)[tdofind]]
                          << ", index = " << (*temp)[tdofind] << "\n";
                //std::cout << " ... was corrected \n";
            }
            //outFoneCoarseHdivvec[(*temp)[tdofind]] = 0.0;
        }

    }
#endif
    Vector inCoarseHcurlvec(TrueP_C[0]->Width());
    TrueP_C[0]->MultTranspose(inHcurlvec, inCoarseHcurlvec); // project after moving from Hcurl

#ifdef CHECK_BNDCND
    {
        const Array<int> *temp = EssBdrTrueDofs_Hcurl[1];
        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            if ( fabs(inCoarseHcurlvec[(*temp)[tdofind]]) > 1.0e-14 )
            {
                std::cout << "bnd cnd is violated for inCoarseHcurlvec, value = "
                          << inCoarseHcurlvec[(*temp)[tdofind]]
                          << ", index = " << (*temp)[tdofind] << "\n";
                std::cout << " ... was corrected \n";
            }
            inCoarseHcurlvec[(*temp)[tdofind]] = 0.0;
        }

    }
#endif

    // checking that at coarse level Curl^T * Hdivvec = Hcurlvec
    Vector check(Divfree_hpmat_mod_lvls[1]->Width());
    Divfree_hpmat_mod_lvls[1]->MultTranspose(inCoarseHdivvec, check);
    check -= inCoarseHcurlvec;
    std::cout << "check_norm = " << check.Norml2() / sqrt (check.Size()) << "\n";

    MPI_Barrier(comm);
    std::cout << std::flush;
    MPI_Barrier(comm);

    CGSolver * Geommg_Coarsesolver = ((Multigrid*) (&(((BlockDiagonalPreconditioner*)prec)->GetDiagonalBlock(0))))->GetCoarseSolver();
    Vector outCoarseHcurlvec(Geommg_Coarsesolver->Height());
    Geommg_Coarsesolver->Mult(inCoarseHcurlvec, outCoarseHcurlvec); // solve

#ifdef CHECK_BNDCND
    {
        const Array<int> *temp = EssBdrTrueDofs_Hcurl[1];
        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            if ( fabs(outCoarseHcurlvec[(*temp)[tdofind]]) > 1.0e-14 )
            {
                std::cout << "bnd cnd is violated for outCoarseHcurlvec, value = "
                          << outCoarseHcurlvec[(*temp)[tdofind]]
                          << ", index = " << (*temp)[tdofind] << "\n";
                std::cout << " ... was corrected \n";
            }
            outCoarseHcurlvec[(*temp)[tdofind]] = 0.0;
        }

    }
#endif

    Vector outFineCoarseHcurlvec(TrueP_C[0]->Height());
    TrueP_C[0]->Mult(outCoarseHcurlvec, outFineCoarseHcurlvec);   // interpolate back

#ifdef CHECK_BNDCND
    {
        const Array<int> *temp = EssBdrTrueDofs_Hcurl[0];
        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            if ( fabs(outFineCoarseHcurlvec[(*temp)[tdofind]]) > 1.0e-14 )
            {
                std::cout << "bnd cnd is violated for outFineCoarseHcurlvec, value = "
                          << outCoarseHcurlvec[(*temp)[tdofind]]
                          << ", index = " << (*temp)[tdofind] << "\n";
                //std::cout << " ... was corrected \n";
            }
            //outFineCoarseHcurlvec[(*temp)[tdofind]] = 0.0;
        }

    }
#endif
    Vector out2FineCoarseHdivvec(Divfree_hpmat_mod_lvls[0]->Height());
    Divfree_hpmat_mod_lvls[0]->Mult(outFineCoarseHcurlvec, out2FineCoarseHdivvec); // move to Hdiv back

#ifdef CHECK_BNDCND
    {
        const Array<int> *temp = EssBdrTrueDofs_Funct_lvls[0][0];
        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            if ( fabs(out2FineCoarseHdivvec[(*temp)[tdofind]]) > 1.0e-14 )
            {
                std::cout << "bnd cnd is violated for out2FineCoarseHdivvec, value = "
                          << out2FineCoarseHdivvec[(*temp)[tdofind]]
                          << ", index = " << (*temp)[tdofind] << "\n";
                //std::cout << " ... was corrected \n";
            }
            //out2FineCoarseHdivvec[(*temp)[tdofind]] = 0.0;
        }

    }
#endif

    Vector diffcoarse(R_space_lvls[0]->TrueVSize());
    diffcoarse = outFineCoarseHdivvec;
    diffcoarse -= out2FineCoarseHdivvec;

    //diffcoarse.Print();

    MPI_Barrier(comm);
    for (int i = 0; i < num_procs; ++i)
    {
        if (myid == i)
        {
            std::cout << "I am " << myid << "\n";

            double diffcoarse_norm = diffcoarse.Norml2() / sqrt (diffcoarse.Size());
            double geommgcoarse_norm = out2FineCoarseHdivvec.Norml2() / sqrt(out2FineCoarseHdivvec.Size());
            std::cout << "|| diff of coarse solvers action || = " << diffcoarse_norm << "\n";
            std::cout << "|| diff of coarse solvers action || / || geommg coarse solver action || = " << diffcoarse_norm / geommgcoarse_norm << "\n";
            std::cout << "\n" << std::flush;
        }
        MPI_Barrier(comm);
    }

    if (verbose)
        std::cout << " \nChecking the coarsest level matrix in geometric MG \n";

    {
        HypreParMatrix * A_coarse = ((Multigrid*) (&(((BlockDiagonalPreconditioner*)prec)->GetDiagonalBlock(0))))->GetCoarseOp();

        Vector testinCoarseHcurlvec(A_coarse->Width());
        testinCoarseHcurlvec = 1.0;
        {
            const Array<int> *temp = EssBdrTrueDofs_Hcurl[1];
            for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
            {
                std::cout << (*temp)[tdofind] << " ";
                testinCoarseHcurlvec[(*temp)[tdofind]] = 0.0;
            }
            std::cout << "\n";

        }

        Vector testCoarseHcurlvec(A_coarse->Height());
        A_coarse->Mult(inCoarseHcurlvec, testCoarseHcurlvec);
        {
            const Array<int> *temp = EssBdrTrueDofs_Hcurl[1];
            for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
            {
                if ( fabs(testCoarseHcurlvec[(*temp)[tdofind]]) > 1.0e-14 )
                {
                    std::cout << "bnd cnd is violated for testCoarseHcurlvec, value = "
                              << testCoarseHcurlvec[(*temp)[tdofind]]
                              << ", index = " << (*temp)[tdofind] << "\n";
                }
            }

        }
    }

#endif // for #ifdef COMPARE_COARSE_SOLVERS

    MPI_Finalize();
    return 0;
#else
    int TestmaxIter(400);

    CGSolver Testsolver(MPI_COMM_WORLD);
    Testsolver.SetAbsTol(sqrt(atol));
    Testsolver.SetRelTol(sqrt(rtol));
    Testsolver.SetMaxIter(TestmaxIter);
    Testsolver.SetOperator(*BlockMattest);
    Testsolver.SetPrintLevel(1);

    NewSolver.SetAsPreconditioner(true);
    NewSolver.SetPrintLevel(0);
    if (verbose)
        NewSolver.PrintAllOptions();

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

        // Computing error for S

        double err_S = NewS->ComputeL2Error(*Mytest.scalarS, irs);
        double norm_S = ComputeGlobalLpNorm(2, *Mytest.scalarS, *pmesh, irs);
        if (verbose)
        {
            std::cout << "|| S_h - S_ex || / || S_ex || = " <<
                         err_S / norm_S << "\n";
        }
        /////////////////////////////////////////////////////////
    }

    chrono.Stop();
    if (verbose)
        std::cout << "Errors in USE_AS_A_PREC were computed in " << chrono.RealTime() <<" seconds.\n";
    chrono.Clear();
    chrono.Start();
#endif

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
        secondeqn_rhs = new ParLinearForm(H_space_lvls[0]);
        secondeqn_rhs->Assemble();
        secondeqn_rhs->ParallelAssemble(NewRhs.GetBlock(1));

        for ( int i = 0; i < EssBdrTrueDofs_Funct_lvls[0][1]->Size(); ++i)
        {
            int bdrtdof = (*EssBdrTrueDofs_Funct_lvls[0][1])[i];
            NewRhs.GetBlock(1)[bdrtdof] = 0.0;
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

    //MPI_Finalize();
    //return 0;

#ifndef COMPARE_MG

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
        delete BdrDofs_Funct_lvls[l][1];
        delete EssBdrDofs_Funct_lvls[l][1];
        delete EssBdrTrueDofs_Funct_lvls[l][1];
        delete EssBdrDofs_H1[l];

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
        delete H_space_lvls[l];
        delete pmesh_lvls[l];

        if (l < num_levels - 1)
        {
            delete P_W[l];
            delete P_WT[l];
            delete P_R[l];
            delete P_C_lvls[l];
            delete P_H_lvls[l];
            delete TrueP_R[l];
            if (prec_is_MG)
                delete TrueP_C[l];
            delete TrueP_H[l];

            delete Element_dofs_R[l];
            delete Element_dofs_W[l];
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
    delete S_exact_finest;

    delete NewSigmahat;
    delete NewS;

#ifdef USE_AS_A_PREC
    delete Atest;
    delete Ctest;
    delete Dtest;
    delete DTtest;
    delete BlockMattest;
#endif

#ifdef OLD_CODE
    delete gform;
    delete Bdiv;

    delete u_exact;
    delete S_exact;
    delete sigma_exact;
    delete opdivfreepart;
    delete sigma;
    delete S;

    delete Sigmahat;
    delete u;

#ifdef   USE_CURLMATRIX
    delete MainOp;
    delete Mblock;
    delete M;
    delete A;
    delete C;

    delete CHT;
    delete CH;
    delete B;
    delete BT;
#endif
    if(dim<=4)
    {
        if (prec_is_MG)
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
#endif
    MPI_Finalize();
    return 0;
}

void zerovecx_ex(const Vector& xt, Vector& zerovecx )
{
    zerovecx.SetSize(xt.Size() - 1);
    zerovecx = 0.0;
}

void zerovectx_ex(const Vector& xt, Vector& vecvalue)
{
    vecvalue.SetSize(xt.Size());
    vecvalue = 0.0;
    return;
}

double zero_ex(const Vector& xt)
{
    return 0.0;
}

void zerovecMat4D_ex(const Vector& xt, Vector& vecvalue)
{
    vecvalue.SetSize(6);
    vecvalue = 0.0;
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

template<double (*S)(const Vector & xt), double (*d2Sdt2)(const Vector & xt), double (*Slaplace)(const Vector & xt) > \
double divsigmaTemplate(const Vector& xt)
{
    Vector xt0(xt.Size());
    xt0 = xt;
    xt0 (xt0.Size() - 1) = 0;

    return d2Sdt2(xt) - Slaplace(xt);
}

template <double (*dSdt)(const Vector & xt), void (*Sgradxvec)(const Vector&, Vector& ),
          void (*opdivfreevec)(const Vector&, Vector& )> \
void sigmahatTemplate(const Vector& xt, Vector& sigmahatv)
{
    sigmahatv.SetSize(xt.Size());

    Vector sigma(xt.Size());
    sigmaTemplate<dSdt, Sgradxvec>(xt, sigma);

    Vector opdivfree;
    opdivfreevec(xt, opdivfree);

    sigmahatv = 0.0;
    sigmahatv -= opdivfree;
    sigmahatv += sigma;
    return;
}

template <double (*dSdt)(const Vector & xt), void (*Sgradxvec)(const Vector&, Vector& ),
          void (*opdivfreevec)(const Vector&, Vector& )> \
void minsigmahatTemplate(const Vector& xt, Vector& minsigmahatv)
{
    minsigmahatv.SetSize(xt.Size());
    sigmahatTemplate<dSdt, Sgradxvec, opdivfreevec>(xt, minsigmahatv);
    minsigmahatv *= -1;

    return;
}


void vminusone_exact(const Vector &x, Vector &vminusone)
{
    vminusone.SetSize(x.Size());
    vminusone = -1.0;
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

int ipow(int base, int exp)
{
    int result = 1;
    while (exp)
    {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        base *= base;
    }

    return result;
}
