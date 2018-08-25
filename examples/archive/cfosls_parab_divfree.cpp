///
///                        MFEM CFOSLS Heat equation with multilevel algorithm and multigrid (div-free part)
///
///
/// DEEP ARCHIVE:
/// The code similar to this but for transport equation, cfosls_hyperb_divfree.cpp first evolved
/// into cfosls_hyperbolic_newsolver.cpp and then into cfosls_hyperbolic_multigrid.cpp. The latter is much
/// cleaner and uses better newer interfaces. This code was saved for some hypothetical unknown reason of
/// digging up older codes.
/// If you really want to use multigrid for heat equation, consider changing the problem (which reduces to
/// replacing about 50 lines) in cfosls_hyperbolic_multigrid.cpp.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>

// this one was deleted. either look for it in the history of the repository,
// or replace the Heat_test_divfree by Parab_test from cfosls/cfosls_testsuite.hpp
// an remove numcurl from the example code.
// E.g., exactly this was done in cfosls_parabolic_newsolver.cpp which is in general
// a newer (but not the newest!) version of the code here.
#include "cfosls_testsuite.hpp"

// this can be found in the examples/archive/ folder
#include "divfree_solver_tools.hpp"

#define MYZEROTOL (1.0e-13)

// if undefined, a code with new integrators is used
// for now, the "integrators'" code is giving wrong results
// TODO: once it will be good to have the integrators working
#define USE_CURLMATRIX

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

//////////////////// new
class HeatVectorFECurlSigmaIntegrator: public BilinearFormIntegrator
{
private:
#ifndef MFEM_THREAD_SAFE
    Vector shape;
    DenseMatrix dshape;
    DenseMatrix curlshape;
    DenseMatrix curlshape_dFT;
    DenseMatrix dshapedxt;
    DenseMatrix invdfdx;
    //old
    DenseMatrix curlshapeTrial;
    DenseMatrix vshapeTest;
    DenseMatrix curlshapeTrial_dFT;
#endif
public:
    HeatVectorFECurlSigmaIntegrator() {}
    virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat) { }
    virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

void HeatVectorFECurlSigmaIntegrator::AssembleElementMatrix2(
const FiniteElement &trial_fe, const FiniteElement &test_fe,
ElementTransformation &Trans, DenseMatrix &elmat)
{
    int trial_nd = trial_fe.GetDof(), test_nd = test_fe.GetDof(), i;

    int dim;
    int vector_dof, scalar_dof;

    MFEM_ASSERT(trial_fe.GetMapType() == mfem::FiniteElement::H_CURL ||
               test_fe.GetMapType() == mfem::FiniteElement::H_CURL,
               "At least one of the finite elements must be in H(Curl)");

    //int curl_nd;
    //int vec_nd;
    if ( trial_fe.GetMapType() == mfem::FiniteElement::H_CURL )
    {
      //curl_nd = trial_nd;
      vector_dof = trial_fe.GetDof();
      //vec_nd  = test_nd;
      scalar_dof = test_fe.GetDof();
      dim = trial_fe.GetDim();
    }
    else
    {
      //curl_nd = test_nd;
      vector_dof = test_fe.GetDof();
      //vec_nd  = trial_nd;
      scalar_dof = trial_fe.GetDof();
      dim = test_fe.GetDim();
    }

    MFEM_ASSERT(dim == 3, "HeatVectorFECurlSigmaIntegrator is working only in 3D currently \n");

    #ifdef MFEM_THREAD_SAFE
    DenseMatrix curlshapeTrial(curl_nd, dimc);
    DenseMatrix curlshapeTrial_dFT(curl_nd, dimc);
    DenseMatrix vshapeTest(vec_nd, dimc);
    DenseMatrix dshape(scalar_dof,dim);
    DenseMatrix dshapedxt(scalar_dof,dim);
    DenseMatrix invdfdx(dim,dim);
    #else
    //curlshapeTrial.SetSize(curl_nd, dimc);
    //curlshapeTrial_dFT.SetSize(curl_nd, dimc);
    //vshapeTest.SetSize(vec_nd, dimc);
    #endif
    //Vector shapeTest(vshapeTest.GetData(), vec_nd);

    curlshape.SetSize(vector_dof, dim);
    curlshape_dFT.SetSize(vector_dof, dim);
    shape.SetSize(scalar_dof);
    dshape.SetSize(scalar_dof,dim);
    dshapedxt.SetSize(scalar_dof,dim);
    invdfdx.SetSize(dim,dim);

    elmat.SetSize(test_nd, trial_nd);

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
      int order = trial_fe.GetOrder() + test_fe.GetOrder() - 1; // <--
      ir = &IntRules.Get(trial_fe.GetGeomType(), order);
    }

    elmat = 0.0;
    for (i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);

        Trans.SetIntPoint(&ip);

        double w = ip.weight * Trans.Weight();

        if (dim == 3)
        {
            if ( trial_fe.GetMapType() == mfem::FiniteElement::H_CURL )
            {
               trial_fe.CalcCurlShape(ip, curlshape);
               test_fe.CalcShape(ip, shape);
               test_fe.CalcDShape(ip, dshape);
            }
            else
            {
               test_fe.CalcCurlShape(ip, curlshape);
               trial_fe.CalcShape(ip, shape);
               trial_fe.CalcDShape(ip, dshape);
            }
            CalcInverse(Trans.Jacobian(), invdfdx);
            Mult(dshape, invdfdx, dshapedxt);

            MultABt(curlshape, Trans.Jacobian(), curlshape_dFT);

            ///////////////////////////
            for (int d = 0; d < dim; d++)
            {
               for (int j = 0; j < scalar_dof; j++)
               {
                   for (int k = 0; k < vector_dof; k++)
                   {
                       // last component: instead of d scalar_phi_i / dt
                       // we use - scalar_phi
                       if (d == dim - 1)
                           elmat(j, k) += w * (-shape(j)) * curlshape_dFT(k, d);
                       else
                           elmat(j, k) += w * dshapedxt(j,d) * curlshape_dFT(k, d);
                   }
               }
            }
            ///////////////////////////
         } // end of if dim = 3
    } // end of loop over integration points
}
class HeatSigmaLFIntegrator: public LinearFormIntegrator
{
    Vector el_shape;
    DenseMatrix el_dshape;
    DenseMatrix invdfdx;
    DenseMatrix el_dshapedxt;
    Vector coeffvec;
    Vector local_vec;
    VectorCoefficient &Q;
    int oa, ob;
 public:
    /// Constructs a domain integrator with a given Coefficient
    HeatSigmaLFIntegrator(VectorCoefficient &QF, int a = 2, int b = 0)
    // the old default was a = 1, b = 1
    // for simple elliptic problems a = 2, b = -2 is ok
       : Q(QF), oa(a), ob(b) { }

    /// Constructs a domain integrator with a given Coefficient
    HeatSigmaLFIntegrator(VectorCoefficient &QF, const IntegrationRule *ir)
       : LinearFormIntegrator(ir), Q(QF), oa(1), ob(1) { }

    /** Given a particular Finite Element and a transformation (Tr)
        computes the element right hand side element vector, elvect. */
    virtual void AssembleRHSElementVect(const FiniteElement &el,
                                        ElementTransformation &Tr,
                                        Vector &elvect);

    using LinearFormIntegrator::AssembleRHSElementVect;
};

void HeatSigmaLFIntegrator::AssembleRHSElementVect(
    const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
    int dof = el.GetDof();
    int dim  = el.GetDim();

    el_shape.SetSize(dof);
    el_dshape.SetSize(dof,dim);
    invdfdx.SetSize(dim,dim);
    el_dshapedxt.SetSize(dof,dim);
    local_vec.SetSize(dof);

    elvect.SetSize(dof);
    elvect = 0.0;

    double w;

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
       int order = (Tr.OrderW() + el.GetOrder() + el.GetOrder());
       ir = &IntRules.Get(el.GetGeomType(), order);
    }

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
       const IntegrationPoint &ip = ir->IntPoint(i);
       Tr.SetIntPoint (&ip);

       el.CalcShape(ip, el_shape);
       el.CalcDShape(ip, el_dshape);

       w = ip.weight * Tr.Weight();
       CalcInverse(Tr.Jacobian(), invdfdx);
       // dshapedxt = matrix d phi_i / dx^j ~ (dof,dim)
       Mult(el_dshape, invdfdx, el_dshapedxt);

       // replacing last column (related to t) of dshapedxt by (-phi_i)
       for (int dofind = 0; dofind < dof; ++dofind)
           el_dshapedxt(dofind, dim - 1) = - el_shape(dofind);

       // coeffvec = values of vector coefficient at dofs
       Q.Eval(coeffvec, Tr, ip);

       el_dshapedxt.Mult(coeffvec, local_vec);
       add(elvect, w, local_vec, elvect);
    }
}
//////////////////// new




//********* NEW STUFF FOR 4D HEAT CFOSLS
//-----------------------
/// Integrator for (Q u, v) for VectorFiniteElements

class PAUVectorFEMassIntegrator: public BilinearFormIntegrator
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
    PAUVectorFEMassIntegrator() { Init(NULL, NULL, NULL); }
    PAUVectorFEMassIntegrator(Coefficient *_q) { Init(_q, NULL, NULL); }
    PAUVectorFEMassIntegrator(Coefficient &q) { Init(&q, NULL, NULL); }
    PAUVectorFEMassIntegrator(VectorCoefficient *_vq) { Init(NULL, _vq, NULL); }
    PAUVectorFEMassIntegrator(VectorCoefficient &vq) { Init(NULL, &vq, NULL); }
    PAUVectorFEMassIntegrator(MatrixCoefficient *_mq) { Init(NULL, NULL, _mq); }
    PAUVectorFEMassIntegrator(MatrixCoefficient &mq) { Init(NULL, NULL, &mq); }

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
class PAUVectorFEMassIntegrator2: public BilinearFormIntegrator
{
private:
    Coefficient *Q;
    void Init(Coefficient *q)
    { Q = q; }

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
    PAUVectorFEMassIntegrator2() { Init(NULL); }
    PAUVectorFEMassIntegrator2(Coefficient *_q) { Init(_q); }
    PAUVectorFEMassIntegrator2(Coefficient &q) { Init(&q); }

    virtual void AssembleElementMatrix(const FiniteElement &el,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

//=-=-=-=-=-=-=-=-=-=-=-=-=-
void PAUVectorFEMassIntegrator::AssembleElementMatrix(
        const FiniteElement &el,
        ElementTransformation &Trans,
        DenseMatrix &elmat)
{}

void PAUVectorFEMassIntegrator::AssembleElementMatrix2(
        const FiniteElement &trial_fe, const FiniteElement &test_fe,
        ElementTransformation &Trans, DenseMatrix &elmat)
{
    // assume both test_fe and trial_fe are vector FE
    int dim  = test_fe.GetDim();
    int trial_dof = trial_fe.GetDof();
    int test_dof = test_fe.GetDof();
    double w;

    if (VQ || MQ) // || = or
        mfem_error("VectorFEMassIntegrator::AssembleElementMatrix2(...)\n"
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
                elmat(j, k) -= w * test_vshape(j, dim - 1) * trial_shape(k);
            }
        }
    }
}

void PAUVectorFEMassIntegrator2::AssembleElementMatrix(
        const FiniteElement &el,
        ElementTransformation &Trans,
        DenseMatrix &elmat)
{
    int dof = el.GetDof();
    int dim  = el.GetDim();
    double w;

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
                for (int d = 0; d < dim - 1; d++ )
                    elmat(j, k) +=  w * dshapedxt(j, d) * dshapedxt(k, d);
                elmat(j, k) +=  w * shape(j) * shape(k);
            }

    }
}

// Define the analytical solution and forcing terms / boundary conditions
//double u0_function(const Vector &x);
double uFun_ex(const Vector & x); // Exact Solution
double uFun_ex_dt(const Vector & xt);
double uFun_ex_laplace(const Vector & xt);
void uFun_ex_gradx(const Vector& xt, Vector& gradx );

double uFun1_ex(const Vector & x); // Exact Solution
double uFun1_ex_dt(const Vector & xt);
double uFun1_ex_laplace(const Vector & xt);
void uFun1_ex_gradx(const Vector& xt, Vector& gradx );

double uFun2_ex(const Vector & x); // Exact Solution
double uFun2_ex_dt(const Vector & xt);
double uFun2_ex_laplace(const Vector & xt);
void uFun2_ex_gradx(const Vector& xt, Vector& gradx );

double uFun3_ex(const Vector & x); // Exact Solution
double uFun3_ex_dt(const Vector & xt);
double uFun3_ex_laplace(const Vector & xt);
void uFun3_ex_gradx(const Vector& xt, Vector& gradx );

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

// 4d test from Martin's example
void E_exactMat_vec(const Vector &x, Vector &E);
void E_exactMat(const Vector &, DenseMatrix &);
void f_exactMat(const Vector &, DenseMatrix &);

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), double (*Slaplace)(const Vector & xt)> \
    double divsigmaTemplate(const Vector& xt);


template <double (*ufunc)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )>
    void sigmaTemplate(const Vector& xt, Vector& sigma);

template<double (*S)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx),
        void (*opdivfreevec)(const Vector&, Vector& )>
        void sigmahatTemplate(const Vector& xt, Vector& sigmahatv);
template<double (*S)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx),
        void (*opdivfreevec)(const Vector&, Vector& )>
        void minsigmahatTemplate(const Vector& xt, Vector& sigmahatv);


class Heat_test_divfree
{
protected:
    int dim;
    int numsol;
    int numcurl;
    bool testisgood;

public:
    FunctionCoefficient * scalarS;                // S
    FunctionCoefficient * scalardivsigma;         // = dS/dt - laplace S                  - what is used for computing error
    VectorFunctionCoefficient * sigma;
    VectorFunctionCoefficient * sigmahat;         // sigma_hat = sigma_exact - op divfreepart (curl hcurlpart in 3D)
    VectorFunctionCoefficient * divfreepart;      // additional part added for testing div-free solver
    VectorFunctionCoefficient * opdivfreepart;    // curl of the additional part which is added to sigma_exact for testing div-free solver
    VectorFunctionCoefficient * minsigmahat;      // -sigma_hat
public:
    Heat_test_divfree (int Dim, int NumSol, int NumCurl);

    int GetDim() {return dim;}
    int GetNumSol() {return numsol;}
    int GetNumCurl() {return numcurl;}
    int CheckIfTestIsGood() {return testisgood;}
    void SetDim(int Dim) { dim = Dim;}
    void SetNumSol(int NumSol) { numsol = NumSol;}
    void SetNumCurl(int NumCurl) { numcurl = NumCurl;}
    bool CheckTestConfig();

    ~Heat_test_divfree () {}
private:
    void SetScalarSFun( double (*S)(const Vector & xt))
    { scalarS = new FunctionCoefficient(S);}

    template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), double (*Slaplace)(const Vector & xt)> \
    void SetDivSigma()
    { scalardivsigma = new FunctionCoefficient(divsigmaTemplate<S, dSdt, Slaplace>);}

    template<double (*f1)(const Vector & xt), void(*f2)(const Vector & x, Vector & vec)> \
    void SetSigmaVec()
    {
        sigma = new VectorFunctionCoefficient(dim, sigmaTemplate<f1,f2>);
    }

    void SetDivfreePart( void(*divfreevec)(const Vector & x, Vector & vec))
    { divfreepart = new VectorFunctionCoefficient(dim, divfreevec);}

    void SetOpDivfreePart( void(*opdivfreevec)(const Vector & x, Vector & vec))
    { opdivfreepart = new VectorFunctionCoefficient(dim, opdivfreevec);}

    template<double (*S)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx),
             void(*opdivfreevec)(const Vector & x, Vector & vec)> \
    void Setsigmahat()
    { sigmahat = new VectorFunctionCoefficient(dim, sigmahatTemplate<S, Sgradxvec, opdivfreevec>);}

    template<double (*S)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx),
             void(*opdivfreevec)(const Vector & x, Vector & vec)> \
    void Setminsigmahat()
    { minsigmahat = new VectorFunctionCoefficient(dim, minsigmahatTemplate<S, Sgradxvec, opdivfreevec>);}


    template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt),
             double (*Slaplace)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
             void(*divfreevec)(const Vector & x, Vector & vec), void(*opdivfreevec)(const Vector & x, Vector & vec)> \
    void SetTestCoeffs ( );
};


template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt),
         double (*Slaplace)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
         void(*divfreevec)(const Vector & x, Vector & vec), void(*opdivfreevec)(const Vector & x, Vector & vec)> \
void Heat_test_divfree::SetTestCoeffs ()
{
    SetScalarSFun(S);
    SetSigmaVec<S,Sgradxvec>();
    SetDivSigma<S, dSdt, Slaplace>();
    SetDivfreePart(divfreevec);
    SetOpDivfreePart(opdivfreevec);
    Setsigmahat<S, Sgradxvec, opdivfreevec>();
    Setminsigmahat<S, Sgradxvec, opdivfreevec>();
    return;
}


bool Heat_test_divfree::CheckTestConfig()
{
    if (dim == 4 || dim == 3)
    {
        if (numsol == 0 || numsol == 1)
            return true;
        if (numsol == 2 && dim == 4)
            return true;
        if (numsol == 3 && dim == 3)
            return true;
        if (numsol == -34 && (dim == 3 || dim == 4))
            return true;
        return false;
    }
    else
        return false;

}

Heat_test_divfree::Heat_test_divfree (int Dim, int NumSol, int NumCurl)
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
                SetTestCoeffs<&uFunTest_ex, &uFunTest_ex_dt, &uFunTest_ex_laplace, &uFunTest_ex_gradx, &zerovectx_ex, &zerovectx_ex>();
            else // dim == 4
                SetTestCoeffs<&uFunTest_ex, &uFunTest_ex_dt, &uFunTest_ex_laplace, &uFunTest_ex_gradx, &zerovecMat4D_ex, &zerovectx_ex>();
        }
        if (numsol == 0)
        {
            //std::cout << "The domain should be either a unit rectangle or cube" << std::endl << std::flush;
            if (numcurl == 1)
                SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_laplace, &uFun_ex_gradx, &hcurlFun3D_ex, &curlhcurlFun3D_ex>();
            else if (numcurl == 2)
                SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_laplace, &uFun_ex_gradx, &hcurlFun3D_2_ex, &curlhcurlFun3D_2_ex>();
            else
                SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_laplace, &uFun_ex_gradx, &zerovectx_ex, &zerovectx_ex>();
        }
        if (numsol == 1)
        {
            //std::cout << "The domain should be either a unit rectangle or cube" << std::endl << std::flush;
            if (numcurl == 1)
                SetTestCoeffs<&uFun1_ex, &uFun1_ex_dt, &uFun1_ex_laplace, &uFun1_ex_gradx, &hcurlFun3D_ex, &curlhcurlFun3D_ex>();
            else if (numcurl == 2)
                SetTestCoeffs<&uFun1_ex, &uFun1_ex_dt, &uFun1_ex_laplace, &uFun1_ex_gradx, &hcurlFun3D_2_ex, &curlhcurlFun3D_2_ex>();
            else
                SetTestCoeffs<&uFun1_ex, &uFun1_ex_dt, &uFun1_ex_laplace, &uFun1_ex_gradx, &zerovectx_ex, &zerovectx_ex>();
        }
        if (numsol == 2)
        {
            //if (numcurl == 1)
                //SetTestCoeffs<&uFun2_ex, &uFun2_ex_dt, &uFun2_ex_laplace, &uFun2_ex_gradx, &hcurlFun3D_ex, &curlhcurlFun3D_ex>();
            //else if (numcurl == 2)
                //SetTestCoeffs<&uFun2_ex, &uFun2_ex_dt, &uFun2_ex_laplace, &uFun2_ex_gradx, &hcurlFun3D_2_ex, &curlhcurlFun3D_2_ex>();
            if (numcurl == 1 || numcurl == 2)
            {
                std::cout << "Critical error: Explicit analytic div-free guy is not implemented in 4D \n";
            }
            else
                SetTestCoeffs<&uFun2_ex, &uFun2_ex_dt, &uFun2_ex_laplace, &uFun2_ex_gradx, &zerovecMat4D_ex, &zerovectx_ex>();
        }
        if (numsol == 3)
        {
            if (numcurl == 1)
                SetTestCoeffs<&uFun3_ex, &uFun3_ex_dt, &uFun3_ex_laplace, &uFun3_ex_gradx, &hcurlFun3D_ex, &curlhcurlFun3D_ex>();
            else if (numcurl == 2)
                SetTestCoeffs<&uFun3_ex, &uFun3_ex_dt, &uFun3_ex_laplace, &uFun3_ex_gradx, &hcurlFun3D_2_ex, &curlhcurlFun3D_2_ex>();
            else
                SetTestCoeffs<&uFun3_ex, &uFun3_ex_dt, &uFun3_ex_laplace, &uFun3_ex_gradx, &zerovectx_ex, &zerovectx_ex>();
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
    bool refine_t_first = true;

    bool withDiv = true;
    bool with_multilevel = true;
    //bool withS = true;
    //bool blockedversion = true;

    // solver options
    int prec_option = 2;        // defines whether to use preconditioner or not, and which one
    bool prec_is_MG;
    bool monolithicMG = false;

    bool useM_in_divpart = false;

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
        cout << "Solving CFOSLS Heat equation with MFEM & hypre, div-free approach \n";

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
        break;
    case 3: // block MG
        with_prec = true;
        prec_is_MG = true;
        monolithicMG = true;
        break;
    default: // no preconditioner (default)
        with_prec = false;
        prec_is_MG = false;
        break;
    }

    if (verbose)
    {
        cout << "with_prec = " << with_prec << endl;
        cout << "prec_is_MG = " << prec_is_MG << endl;
        cout << flush;
    }

    StopWatch chrono;

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

    MFEM_ASSERT(!(aniso_refine && (with_multilevel || nDimensions == 4)),"Anisotropic refinement works only in 3D and without multilevel algorithm \n");

    int dim = nDimensions;

    Array<int> ess_bdrSigma(pmesh->bdr_attributes.Max());
    ess_bdrSigma = 0;

    FiniteElementCollection *hdiv_coll;
    ParFiniteElementSpace *R_space;
    FiniteElementCollection *l2_coll;
    ParFiniteElementSpace *W_space;

    if (dim == 4)
        hdiv_coll = new RT0_4DFECollection;
    else
        hdiv_coll = new RT_FECollection(feorder, dim);

    R_space = new ParFiniteElementSpace(pmesh.get(), hdiv_coll);

    if (withDiv)
    {
        l2_coll = new L2_FECollection(feorder, nDimensions);
        W_space = new ParFiniteElementSpace(pmesh.get(), l2_coll);
    }

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
    Array<HypreParMatrix*> P_C(par_ref_levels);
    ParFiniteElementSpace *coarseC_space;
    if (prec_is_MG)
        coarseC_space = new ParFiniteElementSpace(pmesh.get(), hdivfree_coll);

    Array<HypreParMatrix*> P_H(par_ref_levels);
    ParFiniteElementSpace *coarseH_space;
    if (prec_is_MG)
        coarseH_space = new ParFiniteElementSpace(pmesh.get(), h1_coll);

//#ifdef PAULINA_CODE
    Vector sigmahat_pau;

    ParFiniteElementSpace *coarseR_space;
    ParFiniteElementSpace *coarseW_space;

    HypreParMatrix * d_td_coarse_R;
    HypreParMatrix * d_td_coarse_W;
    // Input to the algorithm::

    int ref_levels = par_ref_levels;

    Array< SparseMatrix*> P_W(ref_levels);
    Array< SparseMatrix*> P_R(ref_levels);
    Array< SparseMatrix*> Element_dofs_R(ref_levels);
    Array< SparseMatrix*> Element_dofs_W(ref_levels);
    // Array< int * > Sol_sig_level(ref_levels);

    const SparseMatrix* P_W_local;
    const SparseMatrix* P_R_local;

    Array<int> ess_dof_coarsestlvl_list;
    DivPart divp;

    chrono.Clear();
    chrono.Start();
    if (with_multilevel)
    {
        if (verbose)
            std::cout << "Creating a hierarchy of meshes by successive refinements "
                         "(with multilevel and multigrid prerequisites) \n";

        if (!withDiv && verbose)
            std::cout << "Multilevel code cannot be used without withDiv flag \n";

        coarseR_space = new ParFiniteElementSpace(pmesh.get(), hdiv_coll);
        coarseW_space = new ParFiniteElementSpace(pmesh.get(), l2_coll);

        // Dofs_TrueDofs at each space:

        d_td_coarse_R = coarseR_space->Dof_TrueDof_Matrix();
        d_td_coarse_W = coarseW_space->Dof_TrueDof_Matrix();

        for (int l = 0; l < ref_levels+1; l++)
        {
            if (l > 0){

                if (l == 1)
                {
                    R_space->GetEssentialVDofs(ess_bdrSigma, ess_dof_coarsestlvl_list);
                    //ess_dof_list.Print();
                }

                if (prec_is_MG)
                    coarseC_space->Update();

                if (prec_is_MG)
                    coarseH_space->Update();

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

                C_space->Update();
                if (prec_is_MG)
                {
                    auto d_td_coarse_C = coarseC_space->Dof_TrueDof_Matrix();
                    auto P_C_loc_tmp = (SparseMatrix *)C_space->GetUpdateOperator();
                    auto P_C_local = RemoveZeroEntries(*P_C_loc_tmp);
                    unique_ptr<SparseMatrix>RP_C_local(
                                Mult(*C_space->GetRestrictionMatrix(), *P_C_local));
                    P_C[l-1] = d_td_coarse_C->LeftDiagMult(
                                *RP_C_local, C_space->GetTrueDofOffsets());
                    P_C[l-1]->CopyColStarts();
                    P_C[l-1]->CopyRowStarts();
                    delete P_C_local;
                }

                H_space->Update();
                if (prec_is_MG)
                {
                    auto d_td_coarse_H = coarseH_space->Dof_TrueDof_Matrix();
                    auto P_H_loc_tmp = (SparseMatrix *)H_space->GetUpdateOperator();
                    auto P_H_local = RemoveZeroEntries(*P_H_loc_tmp);
                    unique_ptr<SparseMatrix>RP_H_local(
                                Mult(*H_space->GetRestrictionMatrix(), *P_H_local));
                    P_H[l-1] = d_td_coarse_H->LeftDiagMult(
                                *RP_H_local, H_space->GetTrueDofOffsets());
                    P_H[l-1]->CopyColStarts();
                    P_H[l-1]->CopyRowStarts();
                    delete P_H_local;
                }

                P_W_local = (SparseMatrix *)W_space->GetUpdateOperator();
                P_R_local = (SparseMatrix *)R_space->GetUpdateOperator();

                SparseMatrix* R_Element_to_dofs1 = new SparseMatrix();
                SparseMatrix* W_Element_to_dofs1 = new SparseMatrix();

                divp.Elem2Dofs(*R_space, *R_Element_to_dofs1);
                divp.Elem2Dofs(*W_space, *W_Element_to_dofs1);

                P_W[ref_levels -l] = RemoveZeroEntries(*P_W_local);
                P_R[ref_levels -l] = RemoveZeroEntries(*P_R_local);

                Element_dofs_R[ref_levels - l] = R_Element_to_dofs1;
                Element_dofs_W[ref_levels - l] = W_Element_to_dofs1;

            }
        } // end of loop over levels
    }
    else // not a multilevel algo
    {
        if (verbose)
            std::cout << "Creating a hierarchy of meshes by successive refinements (and multigrid prerequisites) \n";

        for (int l = 0; l < par_ref_levels; l++)
        {
            if (prec_is_MG)
                coarseC_space->Update();

            if (prec_is_MG)
                coarseH_space->Update();

            if (aniso_refine && refine_t_first)
            {
                Array<Refinement> refs(pmesh->GetNE());
                if (l < par_ref_levels/2)
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
                if (l < par_ref_levels/2)
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

            if (withDiv)
                W_space->Update();
            R_space->Update();
            C_space->Update();
            H_space->Update();

            if (prec_is_MG)
            {
                auto d_td_coarse_C = coarseC_space->Dof_TrueDof_Matrix();
                auto P_C_loc_tmp = (SparseMatrix *)C_space->GetUpdateOperator();
                auto P_C_local = RemoveZeroEntries(*P_C_loc_tmp);
                unique_ptr<SparseMatrix>RP_C_local(
                            Mult(*C_space->GetRestrictionMatrix(), *P_C_local));
                P_C[l] = d_td_coarse_C->LeftDiagMult(
                            *RP_C_local, C_space->GetTrueDofOffsets());
                P_C[l]->CopyColStarts();
                P_C[l]->CopyRowStarts();
                delete P_C_local;
            }

            if (prec_is_MG)
            {
                auto d_td_coarse_H = coarseH_space->Dof_TrueDof_Matrix();
                auto P_H_loc_tmp = (SparseMatrix *)H_space->GetUpdateOperator();
                auto P_H_local = RemoveZeroEntries(*P_H_loc_tmp);
                unique_ptr<SparseMatrix>RP_H_local(
                            Mult(*H_space->GetRestrictionMatrix(), *P_H_local));
                P_H[l] = d_td_coarse_H->LeftDiagMult(
                            *RP_H_local, H_space->GetTrueDofOffsets());
                P_H[l]->CopyColStarts();
                P_H[l]->CopyRowStarts();
                delete P_H_local;
            }
        } // end of loop over mesh levels
    } // end of else (not a multilevel algo)
    if (verbose)
        cout<<"MG hierarchy constructed in "<< chrono.RealTime() <<" seconds.\n";

    //if(dim==3) pmesh->ReorientTetMesh();

    pmesh->PrintInfo(std::cout); if(verbose) cout << "\n";

    Heat_test_divfree Mytest(nDimensions, numsol, numcurl);

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
       if (withDiv)
           std::cout << "dim(R) = " << dimR << "\n";
       std::cout << "***********************************************************\n";
    }

    BlockVector xblks(block_offsets);//, rhsblks(block_offsets);
    BlockVector trueX(block_trueOffsets), trueRhs(block_trueOffsets);
    xblks = 0.0;
    //rhsblks = 0.0;
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

    Array<int> ess_bdrU(pmesh->bdr_attributes.Max());
    ess_bdrU = 0;

    Array<int> ess_bdrS(pmesh->bdr_attributes.Max());
    ess_bdrS = 0;
    ess_bdrS = 1;
    ess_bdrS[pmesh->bdr_attributes.Max() - 1] = 0;
    //ess_bdrS[0] = 1; // t = 0

    if (verbose)
    {
        std::cout << "Boundary conditions: \n";
        std::cout << "ess bdr Sigma: \n";
        ess_bdrSigma.Print(std::cout, pmesh->bdr_attributes.Max());
#ifndef USE_CURLMATRIX
        std::cout << "ess bdr U: \n";
#else
        std::cout << "ess bdr U: not used in USE_CURLMATRIX mode \n";
#endif
        ess_bdrU.Print(std::cout, pmesh->bdr_attributes.Max());
        std::cout << "ess bdr S: \n";
        ess_bdrS.Print(std::cout, pmesh->bdr_attributes.Max());
    }

    chrono.Clear();
    chrono.Start();
    ParGridFunction * Sigmahat = new ParGridFunction(R_space);
    ParLinearForm *gform;
    HypreParMatrix *Bdiv;
    if (withDiv)
    {
        if (with_multilevel)
        {
            if (verbose)
                std::cout << "Using multilevel algorithm for finding a particular solution \n";

            ConstantCoefficient k(1.0);

            SparseMatrix *M_local;
            if (useM_in_divpart)
            {
                ParBilinearForm *mVarf(new ParBilinearForm(R_space));
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
            Vector F_fine(P_W[0]->Height());
            Vector G_fine(P_R[0]->Height());

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
                          d_td_coarse_R,
                          d_td_coarse_W,
                          sigmahat_pau,
                          ess_dof_coarsestlvl_list);

    #ifdef MFEM_DEBUG
            Vector sth(F_fine.Size());
            B_fine.Mult(sigmahat_pau, sth);
            sth -= F_fine;
            std::cout << "sth.Norml2() = " << sth.Norml2() << "\n";
            MFEM_ASSERT(sth.Norml2()<1e-8, "The particular solution does not satisfy the divergence constraint");
    #endif

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
            sigma_exact->ProjectCoefficient(*(Mytest.sigma));

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
        }

    }
    else // solving a div-free system with some analytical solution for the div-free part
    {
        if (verbose)
            std::cout << "Using exact sigma minus curl of a given function from H(curl,0) as a particular solution \n";
        Sigmahat->ProjectCoefficient(*(Mytest.sigmahat));
    }
    if (verbose)
        cout<<"Particular solution found in "<< chrono.RealTime() <<" seconds.\n";
    // in either way now Sigmahat is a function from H(div) s.t. div Sigmahat = div sigma = f

    //MFEM_ASSERT(dim == 3, "For now only 3D case is considered \n");

    // the div-free part
    ParGridFunction *u_exact = new ParGridFunction(C_space);
    u_exact->ProjectCoefficient(*(Mytest.divfreepart));

    ParGridFunction * curlu_exact = new ParGridFunction(R_space);
    curlu_exact->ProjectCoefficient(*(Mytest.opdivfreepart));

    ParGridFunction *S_exact = new ParGridFunction(H_space);
    S_exact->ProjectCoefficient(*(Mytest.scalarS));

    ParGridFunction * sigma_exact = new ParGridFunction(R_space);
    sigma_exact->ProjectCoefficient(*(Mytest.sigma));

    if (withDiv)
        xblks.GetBlock(0) = 0.0;
    else
        xblks.GetBlock(0) = *u_exact;
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
    ParDiscreteLinearOperator Divfree_op(C_space, R_space); // from Hcurl or HDivSkew(C_space) to Hdiv(R_space)
    if (dim == 3)
        Divfree_op.AddDomainInterpolator(new CurlInterpolator());
    else // dim == 4
        Divfree_op.AddDomainInterpolator(new DivSkewInterpolator());
    Divfree_op.Assemble();
    Divfree_op.Finalize();
    HypreParMatrix * Divfree_dop = Divfree_op.ParallelAssemble(); // from Hcurl or HDivSkew(C_space) to Hdiv(R_space)
    HypreParMatrix * DivfreeT_dop = Divfree_dop->Transpose();

    // mass matrix for H(div)
    ParBilinearForm *Mblock(new ParBilinearForm(R_space));
    Mblock->AddDomainIntegrator(new VectorFEMassIntegrator);
    Mblock->EliminateEssentialBC(ess_bdrSigma, *sigma_exact, *rhside_Hdiv);
    Mblock->Assemble();
    Mblock->Finalize();

    HypreParMatrix *M = Mblock->ParallelAssemble();

    // curl-curl matrix for H(curl)
    // either as DivfreeT_dop * M * Divfree_dop
    auto temp = ParMult(DivfreeT_dop,M);
    auto A = ParMult(temp, Divfree_dop);
    // or as curl-curl integrator, results are the same
    /*
    ParBilinearForm *Ablock = new ParBilinearForm(C_space);
    Coefficient *one = new ConstantCoefficient(1.0);
    // integrates (curl phi, curl psi)
    Ablock->AddDomainIntegrator(new CurlCurlIntegrator(*one));
    Ablock->Assemble();
    Ablock->EliminateEssentialBC(ess_bdrU,xblks.GetBlock(0),*rhside_Hcurl);
    Ablock->Finalize();
    HypreParMatrix *A = Ablock->ParallelAssemble();
    */


    // diagonal block for H^1
    ParBilinearForm * Cblock;
    Cblock = new ParBilinearForm(H_space);
    // integrates ((-q, grad_x q)^T, (-p, grad_x p)^T)
    Cblock->AddDomainIntegrator(new PAUVectorFEMassIntegrator2);
    Cblock->Assemble();
    Cblock->EliminateEssentialBC(ess_bdrS, xblks.GetBlock(1), *rhside_H1);
    Cblock->Finalize();
    HypreParMatrix * C = Cblock->ParallelAssemble();

    // off-diagonal block for (H(div), H1) block
    ParMixedBilinearForm *Bblock(new ParMixedBilinearForm(H_space, R_space));
    Bblock->AddDomainIntegrator(new PAUVectorFEMassIntegrator);
    Bblock->Assemble();
    Bblock->EliminateTrialDofs(ess_bdrS, xblks.GetBlock(1), *rhside_Hdiv);
    Bblock->EliminateTestDofs(ess_bdrSigma);
    Bblock->Finalize();
    auto B = Bblock->ParallelAssemble();
    auto BT = B->Transpose();

    auto CHT = ParMult(DivfreeT_dop, B);
    auto CH = CHT->Transpose();

    // additional temporary vectors on true dofs required for various matvec
    Vector tempHdiv_true(R_space->TrueVSize());
    Vector temp2Hdiv_true(R_space->TrueVSize());

    // assembling local rhs vectors from inhomog. boundary conditions
    rhside_H1->ParallelAssemble(trueRhs.GetBlock(1));
    rhside_Hdiv->ParallelAssemble(tempHdiv_true);
    DivfreeT_dop->Mult(tempHdiv_true, trueRhs.GetBlock(0));

    // subtracting from Hcurl rhs a part from Sigmahat
    Sigmahat->ParallelProject(tempHdiv_true);
    M->Mult(tempHdiv_true, temp2Hdiv_true);
    //DivfreeT_dop->Mult(temp2Hdiv_true, tempHcurl_true);
    //trueRhs.GetBlock(0) -= tempHcurl_true;
    DivfreeT_dop->Mult(-1.0, temp2Hdiv_true, 1.0, trueRhs.GetBlock(0));

    // subtracting from H1 rhs a part from Sigmahat
    //BT->Mult(tempHdiv_true, tempH1_true);
    //trueRhs.GetBlock(1) -= tempH1_true;
    BT->Mult(-1.0, tempHdiv_true, 1.0, trueRhs.GetBlock(1));

    // setting block operator of the system
    MainOp->SetBlock(0,0, A);
    MainOp->SetBlock(0,1, CHT);
    MainOp->SetBlock(1,0, CH);
    MainOp->SetBlock(1,1, C);

    // checking the residual for the exact solution
#if 0
    {
        if (verbose)
            std::cout << "Calculating some residuals \n";

        ParGridFunction * temp1R = new ParGridFunction(R_space);
        ParGridFunction * temp2R = new ParGridFunction(R_space);
        ParGridFunction * temp1C = new ParGridFunction(C_space);

        ParGridFunction *res_Hcurl = new ParGridFunction(C_space);
        if (!withDiv)
        {
            ParGridFunction * curl_divfree_exact = new ParGridFunction(R_space);
            DivfreeT_dop->MultTranspose(*u_exact, *curl_divfree_exact);
            *curl_divfree_exact -= *curlu_exact;
            std::cout << "diff between curl * u_exact and curlu_exact = " << curl_divfree_exact->Norml2() / sqrt( curl_divfree_exact->Size()) << "\n";

            //CurlT->MultTranspose(*u_exact, *temp1R);
            DivfreeT_dop->MultTranspose(*u_exact, *temp1R);
            *temp1R += *Sigmahat;
            *temp1R -= *sigma_exact;
            std::cout << "diff between curl * u_exact and (sigma_exact - sigmahat) = " << temp1R->Norml2() / sqrt( temp1R->Size()) << "\n";

            *res_Hcurl = *rhside_Hcurl;
            std::cout << "rhside_Hcurl norm = " << rhside_Hcurl->Norml2() / sqrt (rhside_Hcurl->Size()) << "\n";
            ParGridFunction *temp1_Hcurl = new ParGridFunction(C_space);
            A->Mult(*u_exact, *temp1_Hcurl);
            //std::cout << "A * u_exact norm = " << temp1_Hcurl->Norml2() / sqrt (temp1_Hcurl->Size()) << "\n";
            DivfreeT_dop->MultTranspose(*u_exact, *temp1R);
            M->Mult(*temp1R, *temp2R);
            DivfreeT_dop->Mult(*temp2R, *temp1C);
            //std::cout << "CurlT M C curl u_exact norm = " << temp1C->Norml2() / sqrt (temp1C->Size()) << "\n";
            *res_Hcurl -= *temp1_Hcurl;
            CHT->Mult(*S_exact, *temp1_Hcurl);
            //std::cout << "CHT * S_exact norm = " << temp1_Hcurl->Norml2() / sqrt (temp1_Hcurl->Size()) << "\n";
            *res_Hcurl -= *temp1_Hcurl;

            std::cout << "abs. residual for H(curl) eqn = " << res_Hcurl->Norml2() / sqrt(res_Hcurl->Size()) << "\n";
            std::cout << "rel. residual for H(curl) eqn = " << res_Hcurl->Norml2() / rhside_Hcurl->Norml2() << "\n";

            ParGridFunction *res_H1 = new ParGridFunction(H_space);
            *res_H1 = *rhside_H1;
            ParGridFunction *temp1_H1 = new ParGridFunction(H_space);
            CH->Mult(*u_exact, *temp1_H1);
            *res_H1 -= *temp1_H1;
            C->Mult(*S_exact, *temp1_H1);
            *res_H1 -= *temp1_H1;

            std::cout << "abs. residual for H1 eqn = " << res_H1->Norml2() / sqrt(res_H1->Size()) << "\n";
            std::cout << "rel. residual for H1 eqn = " << res_H1->Norml2() / rhside_H1->Norml2() << "\n";
        }

        ParGridFunction *res_Hdiv = new ParGridFunction(R_space);
        *res_Hdiv = *rhside_Hdiv;
        std::cout << "rhside_Hdiv norm = " << rhside_Hdiv->Norml2() / sqrt (rhside_Hdiv->Size()) << "\n";
        //CurlT->Mult(*rhside_Hdiv, *temp1C);
        DivfreeT_dop->Mult(*rhside_Hdiv, *temp1C);
        //std::cout << "CurlT * rhside_Hdiv norm = " << temp1C->Norml2() / sqrt (temp1C->Size()) << "\n";
        ParGridFunction *Msigmatilda = new ParGridFunction(R_space);
        ParGridFunction *temp2_Hdiv = new ParGridFunction(R_space);
        *temp2_Hdiv = *sigma_exact;
        *temp2_Hdiv -= *Sigmahat;
        M->Mult(*temp2_Hdiv, *Msigmatilda);
        ParGridFunction *CurlTMsigmatilda = new ParGridFunction(C_space);
        DivfreeT_dop->Mult(*Msigmatilda, *CurlTMsigmatilda);
        //std::cout << "CurlT M sigmatilda norm = " << CurlTMsigmatilda->Norml2() / sqrt (CurlTMsigmatilda->Size()) << "\n";
        *res_Hdiv -= *Msigmatilda;
        ParGridFunction *BS_exact = new ParGridFunction(R_space);
        B->Mult(*S_exact, *BS_exact);
        //std::cout << "B * S_exact norm = " << BS_exact->Norml2() / sqrt (BS_exact->Size()) << "\n";
        DivfreeT_dop->Mult(*BS_exact, *temp1C);
        //std::cout << "CurlT * B * S_exact norm = " << temp1C->Norml2() / sqrt (temp1C->Size()) << "\n";
        *res_Hdiv -= *BS_exact;

        if (rhside_Hdiv->Norml2() > MYZEROTOL)
            std::cout << "rel. residual for H(div) eqn with (sigma - sigmahat) and no lambda = " << res_Hdiv->Norml2() / rhside_Hdiv->Norml2() << "\n";
        else
            std::cout << "abs. residual for H(div) eqn with (sigma - sigmahat) and no lambda = " << res_Hdiv->Norml2() / sqrt(res_Hdiv->Size()) << "\n";

        if (!withDiv)
        {
            ParGridFunction * checkk = new ParGridFunction(C_space);
            DivfreeT_dop->Mult(*res_Hdiv, *checkk);
            *checkk -= *res_Hcurl;

            std::cout << "diff between CurlT * residual for Hdiv and residual for Hcurl = " << checkk->Norml2() / sqrt( checkk->Size()) << "\n";
        }

        ParGridFunction * curl_res_Hdiv = new ParGridFunction(C_space);
        DivfreeT_dop->Mult(*res_Hdiv, *curl_res_Hdiv);

        std::cout << "Curl of residual for H(div) eqn with (sigma - sigmahat) and no lambda = "
                  << curl_res_Hdiv->Norml2() / curl_res_Hdiv->Size() << "\n";
    }
#endif // end of #if 0 for calculating residuals
#else // if using the integrators for creating the div-free system

    if (verbose)
        std::cout << "Creating div-free system using new integrators \n";

    if (verbose)
        std::cout << "Gives wrong results now! \n";

    ffform = new ParLinearForm(C_space);


    if (!withDiv)
    {
        ffform->Update(C_space, rhsblks.GetBlock(0), 0);
        ffform->AddDomainIntegrator(new VectorcurlDomainLFIntegrator(*(Mytest.minsigmahat)));
        ffform->Assemble();
    }
    else // if withDiv = true
    {
        ParMixedBilinearForm * Tblock = new ParMixedBilinearForm(R_space, C_space);
        // integrates (- phi, curl psi)
        Tblock->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(minusone));
        Tblock->Assemble();
        Tblock->EliminateTestDofs(ess_bdrU); // ess_bdrU or ess_bdrSigma?
        Tblock->Finalize();
        Tblock->Mult(*Sigmahat, *ffform);
    }

    ParLinearForm *qform(new ParLinearForm);
    /*
    if (!withDiv)
    {
        qform->Update(H_space, rhsblks.GetBlock(1), 0);
        qform->AddDomainIntegrator(new HeatSigmaLFIntegrator(*Mytest.sigmahat));
        qform->Assemble();
    }
    else
    */
    {
        qform->Update(H_space, rhsblks.GetBlock(1), 0);

        ParMixedBilinearForm * LTblock = new ParMixedBilinearForm(H_space, R_space);
        // integrates (phi, (-q, grad_x q)^T)
        LTblock->AddDomainIntegrator(new PAUVectorFEMassIntegrator);
        LTblock->Assemble();
        LTblock->EliminateTestDofs(ess_bdrS);
        LTblock->Finalize();
        // we need to change the sign for the rh side of our system
        *Sigmahat *= -1.0;
        LTblock->MultTranspose(*Sigmahat, *qform);
        *Sigmahat *= -1.0;
    }

    Ablock = new ParBilinearForm(C_space);
    Coefficient *one = new ConstantCoefficient(1.0);
    // integrates (curl phi, curl psi)
    Ablock->AddDomainIntegrator(new CurlCurlIntegrator(*one));
    Ablock->Assemble();
    Ablock->EliminateEssentialBC(ess_bdrU,xblks.GetBlock(0),*ffform);
    Ablock->Finalize();
    HypreParMatrix * tempA = Ablock->ParallelAssemble();
    A = make_shared<HypreParMatrix>(*tempA);

    ParBilinearForm *Cblock;
    HypreParMatrix *C;

    Cblock = new ParBilinearForm(H_space);
    // integrates ((-q, grad_x q)^T, (-p, grad_x p)^T)
    Cblock->AddDomainIntegrator(new PAUVectorFEMassIntegrator2);
    //Cblock->AddDomainIntegrator(new MassIntegrator(*Mytest.bTb));
    //Cblock->AddDomainIntegrator(new DiffusionIntegrator(*Mytest.bbT));
    Cblock->Assemble();
    Cblock->EliminateEssentialBC(ess_bdrS, xblks.GetBlock(1),*qform);
    Cblock->Finalize();
    C = Cblock->ParallelAssemble();

    ParMixedBilinearForm *CHblock;
    HypreParMatrix *CH, *CHT;

    CHblock = new ParMixedBilinearForm(C_space, H_space);
    // integrates (curl phi, (-p, grad_x p)^T)
    CHblock->AddDomainIntegrator(new HeatVectorFECurlSigmaIntegrator);
    //CHblock->AddDomainIntegrator(new VectorFECurlVQIntegrator(*Mytest.minb));
    CHblock->Assemble();
    CHblock->EliminateTestDofs(ess_bdrS);
    CHblock->EliminateTrialDofs(ess_bdrU, xblks.GetBlock(0), *qform);

    CHblock->Finalize();

    CH = CHblock->ParallelAssemble();
    //*CH *= 0.0;
    CHT = CH->Transpose();

    //*ffform *= 0.0;


    ffform->ParallelAssemble(trueRhs.GetBlock(0));
    qform->ParallelAssemble(trueRhs.GetBlock(1));

    BlockOperator *MainOp = new BlockOperator(block_trueOffsets);

    MainOp->SetBlock(0,0, A.get());
    MainOp->SetBlock(0,1, CHT);
    MainOp->SetBlock(1,0, CH);
    MainOp->SetBlock(1,1, C);
#endif // end of #ifdef-else for creating the div-free system

    if (verbose)
        std::cout << "Discretized problem is assembled" << endl << flush;

    chrono.Clear();
    chrono.Start();

    Solver *prec;
    Array<BlockOperator*> P;
    if (with_prec)
    {
        if(dim<=4)
        {
            if (prec_is_MG)
            {
                if (monolithicMG)
                {
                    P.SetSize(P_C.Size());

                    for (int l = 0; l < P.Size(); l++)
                    {
                        auto offsets_f  = new Array<int>(3);
                        auto offsets_c  = new Array<int>(3);
                        (*offsets_f)[0] = (*offsets_c)[0] = 0;
                        (*offsets_f)[1] = P_C[l]->Height();
                        (*offsets_c)[1] = P_C[l]->Width();
                        (*offsets_f)[2] = (*offsets_f)[1] + P_H[l]->Height();
                        (*offsets_c)[2] = (*offsets_c)[1] + P_H[l]->Width();

                        P[l] = new BlockOperator(*offsets_f, *offsets_c);
                        P[l]->SetBlock(0, 0, P_C[l]);
                        P[l]->SetBlock(1, 1, P_H[l]);
                    }
                    prec = new MonolithicMultigrid(*MainOp, P);
                }
                else
                {
                    prec = new BlockDiagonalPreconditioner(block_trueOffsets);
                    Operator * precU = new Multigrid(*A, P_C);
                    Operator * precS = new Multigrid(*C, P_H);
                    ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(0, precU);
                    ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(1, precS);
                }            }
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
            cout << "Using no preconditioner" << endl << flush;
    if (verbose)
        std::cout << "Preconditioner built in " << chrono.RealTime() << "s. \n";

    IterativeSolver * solver;
    solver = new CGSolver(comm);
    if (verbose)
        cout << "Linear solver: CG" << endl << flush;

    solver->SetAbsTol(atol);
    solver->SetRelTol(rtol);
    solver->SetMaxIter(max_num_iter);
    solver->SetOperator(*MainOp);

    if (with_prec)
        solver->SetPreconditioner(*prec);
    solver->SetPrintLevel(0);
    trueX = 0.0;
    solver->Mult(trueRhs, trueX);
    chrono.Stop();

    if (verbose)
    {
       if (solver->GetConverged())
          std::cout << "Linear solver converged in " << solver->GetNumIterations()
                    << " iterations with a residual norm of " << solver->GetFinalNorm() << ".\n";
       else
          std::cout << "Linear solver did not converge in " << solver->GetNumIterations()
                    << " iterations. Residual norm is " << solver->GetFinalNorm() << ".\n";
       std::cout << "Linear solver took " << chrono.RealTime() << "s. \n";
    }

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

    double err_u, norm_u;

    if (!withDiv)
    {
        err_u = u->ComputeL2Error(*(Mytest.divfreepart), irs);
        norm_u = ComputeGlobalLpNorm(2, *(Mytest.divfreepart), *pmesh, irs);

        if (verbose)
        {
            if ( norm_u > MYZEROTOL )
            {
                //std::cout << "norm_u = " << norm_u << "\n";
                cout << "|| u - u_ex || / || u_ex || = " << err_u / norm_u << endl;
            }
            else
                cout << "|| u || = " << err_u << " (u_ex = 0)" << endl;
        }
    }

    ParGridFunction * opdivfreepart = new ParGridFunction(R_space);
    DiscreteLinearOperator Divfree_h(C_space, R_space);
    if (dim == 3)
        Divfree_h.AddDomainInterpolator(new CurlInterpolator());
    else // dim == 4
        Divfree_h.AddDomainInterpolator(new DivSkewInterpolator());
    Divfree_h.Assemble();
    Divfree_h.Mult(*u, *opdivfreepart);

    ParGridFunction * opdivfreepart_exact;
    double err_opdivfreepart, norm_opdivfreepart;

    if (!withDiv)
    {
        opdivfreepart_exact = new ParGridFunction(R_space);
        opdivfreepart_exact->ProjectCoefficient(*(Mytest.opdivfreepart));

        err_opdivfreepart = opdivfreepart->ComputeL2Error(*(Mytest.opdivfreepart), irs);
        norm_opdivfreepart = ComputeGlobalLpNorm(2, *(Mytest.opdivfreepart), *pmesh, irs);

        if (verbose)
        {
            if (norm_opdivfreepart > MYZEROTOL )
            {
                //cout << "|| opdivfreepart_ex || = " << norm_opdivfreepart << endl;
                cout << "|| Divfree_h u_h - opdivfreepart_ex || / || opdivfreepart_ex || = " << err_opdivfreepart / norm_opdivfreepart << endl;
            }
            else
                cout << "|| Divfree_h u_h || = " << err_opdivfreepart << " (divfreepart_ex = 0)" << endl;
        }
    }

    ParGridFunction * sigma = new ParGridFunction(R_space);
    *sigma = *Sigmahat;         // particular solution
    *sigma += *opdivfreepart;   // plus div-free guy

    double err_sigma = sigma->ComputeL2Error(*(Mytest.sigma), irs);
    double norm_sigma = ComputeGlobalLpNorm(2, *(Mytest.sigma), *pmesh, irs);

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
    double err_sigmahat = Sigmahat->ComputeL2Error(*(Mytest.sigma), irs);
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

    double err_div = DivSigma.ComputeL2Error(*(Mytest.scalardivsigma),irs);
    double norm_div = ComputeGlobalLpNorm(2, *(Mytest.scalardivsigma), *pmesh, irs);

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
    S_exact->ProjectCoefficient(*(Mytest.scalarS));

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

    if (!withDiv)
    {
        l2_coll = new L2_FECollection(feorder, nDimensions);
        W_space = new ParFiniteElementSpace(pmesh.get(), l2_coll);
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

    if (verbose)
        cout << "Computing projection errors \n";

    if(!withDiv)
    {
        double projection_error_u = u_exact->ComputeL2Error(*(Mytest.divfreepart), irs);
        if (verbose)
        {
            if ( norm_u > MYZEROTOL )
            {
                //std::cout << "Debug: || u_ex || = " << norm_u << "\n";
                //std::cout << "Debug: proj error = " << projection_error_u << "\n";
                cout << "|| u_ex - Pi_h u_ex || / || u_ex || = " << projection_error_u / norm_u << endl;
            }
            else
                cout << "|| Pi_h u_ex || = " << projection_error_u << " (u_ex = 0) \n ";
        }
    }

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

    if (visualization && nDimensions < 4)
    {
       char vishost[] = "localhost";
       int  visport   = 19916;

       if (!withDiv)
       {
           socketstream uex_sock(vishost, visport);
           uex_sock << "parallel " << num_procs << " " << myid << "\n";
           uex_sock.precision(8);
           uex_sock << "solution\n" << *pmesh << *u_exact << "window_title 'u_exact'"
                  << endl;
           socketstream uh_sock(vishost, visport);
           uh_sock << "parallel " << num_procs << " " << myid << "\n";
           uh_sock.precision(8);
           uh_sock << "solution\n" << *pmesh << *u << "window_title 'u_h'"
                  << endl;

           *u -= *u_exact;
           socketstream udiff_sock(vishost, visport);
           udiff_sock << "parallel " << num_procs << " " << myid << "\n";
           udiff_sock.precision(8);
           udiff_sock << "solution\n" << *pmesh << *u << "window_title 'u_h - u_exact'"
                  << endl;


           socketstream opdivfreepartex_sock(vishost, visport);
           opdivfreepartex_sock << "parallel " << num_procs << " " << myid << "\n";
           opdivfreepartex_sock.precision(8);
           opdivfreepartex_sock << "solution\n" << *pmesh << *opdivfreepart_exact << "window_title 'curl u_exact'"
                  << endl;

           socketstream opdivfreepart_sock(vishost, visport);
           opdivfreepart_sock << "parallel " << num_procs << " " << myid << "\n";
           opdivfreepart_sock.precision(8);
           opdivfreepart_sock << "solution\n" << *pmesh << *opdivfreepart << "window_title 'curl u_h'"
                  << endl;

           *opdivfreepart -= *opdivfreepart_exact;
           socketstream opdivfreepartdiff_sock(vishost, visport);
           opdivfreepartdiff_sock << "parallel " << num_procs << " " << myid << "\n";
           opdivfreepartdiff_sock.precision(8);
           opdivfreepartdiff_sock << "solution\n" << *pmesh << *opdivfreepart << "window_title 'curl u_h - curl u_exact'"
                  << endl;
       }

       socketstream S_ex_sock(vishost, visport);
       S_ex_sock << "parallel " << num_procs << " " << myid << "\n";
       S_ex_sock.precision(8);
       S_ex_sock << "solution\n" << *pmesh << *S_exact << "window_title 'S_exact'"
              << endl;

       socketstream S_h_sock(vishost, visport);
       S_h_sock << "parallel " << num_procs << " " << myid << "\n";
       S_h_sock.precision(8);
       S_h_sock << "solution\n" << *pmesh << *S << "window_title 'S_h'"
              << endl;

       *S -= *S_exact;
       socketstream S_diff_sock(vishost, visport);
       S_diff_sock << "parallel " << num_procs << " " << myid << "\n";
       S_diff_sock.precision(8);
       S_diff_sock << "solution\n" << *pmesh << *S << "window_title 'S_h - S_exact'"
              << endl;


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

    // 17. Free the used memory.
#ifndef USE_CURLMATRIX
    delete ffform;
    delete qform;

    delete Ablock;
    delete Cblock;
    delete CHblock;

    delete C_space;
    delete hdivfree_coll;
    delete R_space;
    delete hdiv_coll;
    delete H_space;
    delete h1_coll;
    if (dim == 4)
        delete GradSpace;
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

void zerovecMat4D_ex(const Vector& xt, Vector& vecvalue)
{
    vecvalue.SetSize(6);
    vecvalue = 0.0;
    return;
}

void testfun_ex(const Vector& xt, Vector& vecvalue)
{
    vecvalue.SetSize(xt.Size());
    vecvalue = xt;
    return;
}


double zero_ex(const Vector& xt)
{
    return 0.0;
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

template <double (*S)(const Vector&), void (*Sgradxvec)(const Vector&, Vector& )> \
void sigmaTemplate(const Vector& xt, Vector& sigma)
{
    sigma.SetSize(xt.Size());

    Vector gradS;
    Sgradxvec(xt,gradS);

    sigma(xt.Size()-1) = S(xt);
    for (int i = 0; i < xt.Size()-1; i++)
        sigma(i) = - gradS(i);

    return;
}

template <double (*S)(const Vector & xt), void (*Sgradxvec)(const Vector&, Vector& ),
          void (*opdivfreevec)(const Vector&, Vector& )> \
void sigmahatTemplate(const Vector& xt, Vector& sigmahatv)
{
    sigmahatv.SetSize(xt.Size());

    Vector sigma(xt.Size());
    sigmaTemplate<S, Sgradxvec>(xt, sigma);

    Vector opdivfree;
    opdivfreevec(xt, opdivfree);

    sigmahatv = 0.0;
    sigmahatv -= opdivfree;
    sigmahatv += sigma;
    return;
}

template <double (*S)(const Vector & xt), void (*Sgradxvec)(const Vector&, Vector& ),
          void (*opdivfreevec)(const Vector&, Vector& )> \
void minsigmahatTemplate(const Vector& xt, Vector& minsigmahatv)
{
    minsigmahatv.SetSize(xt.Size());
    sigmahatTemplate<S, Sgradxvec, opdivfreevec>(xt, minsigmahatv);
    minsigmahatv *= -1;

    return;
}

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), double (*Slaplace)(const Vector & xt) > \
double divsigmaTemplate(const Vector& xt)
{
    Vector xt0(xt.Size());
    xt0 = xt;
    xt0 (xt0.Size() - 1) = 0;

    return dSdt(xt) - Slaplace(xt);
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
        zi = xt(2);
        return sin(PI*xi)*sin(PI*yi)*zi;
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
        return sin(PI*xi)*sin(PI*yi);
    if (xt.Size() == 4)
    {
        zi = xt(2);
        return sin(PI*xi)*sin(PI*yi)*sin(PI*zi);
    }

    return 0.0;
}

double uFun_ex_laplace(const Vector & xt)
{
    const double PI = 3.141592653589793;
    return (-(xt.Size()-1) * PI * PI) *uFun_ex(xt);
}

void uFun_ex_gradx(const Vector& xt, Vector& gradx )
{
    const double PI = 3.141592653589793;

    double x = xt(0);
    double y = xt(1);
    double z(0.0);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    if (xt.Size() == 3)
    {
        gradx(0) = t * PI * cos (PI * x) * sin (PI * y);
        gradx(1) = t * PI * sin (PI * x) * cos (PI * y);
    }
    if (xt.Size() == 4)
    {
        z = xt(2);
        gradx(0) = t * PI * cos (PI * x) * sin (PI * y) * sin (PI * z);
        gradx(1) = t * PI * sin (PI * x) * cos (PI * y) * sin (PI * z);
        gradx(2) = t * PI * sin (PI * x) * sin (PI * y) * cos (PI * z);
    }

}


double fFun(const Vector & x)
{
    const double PI = 3.141592653589793;
    double xi(x(0));
    double yi(x(1));
    double zi(0.0);
    double vi(0.0);
    if (x.Size() == 3)
    {
     zi = x(2);
       return 2*PI*PI*sin(PI*xi)*sin(PI*yi)*zi+sin(PI*xi)*sin(PI*yi);
    }

    if (x.Size() == 4)
    {
     zi = x(2);
         vi = x(3);
         //cout << "rhand for 4D" << endl;
       return 3*PI*PI*sin(PI*xi)*sin(PI*yi)*sin(PI*zi)*vi + sin(PI*xi)*sin(PI*yi)*sin(PI*zi);
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

double uFun1_ex_laplace(const Vector & xt)
{
    return (- (xt.Size() - 1) * M_PI * M_PI ) * uFun1_ex(xt);
}

void uFun1_ex_gradx(const Vector& xt, Vector& gradx )
{
    const double PI = 3.141592653589793;

    double x = xt(0);
    double y = xt(1);
    double z(0.0);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    if (xt.Size() == 3)
    {
        gradx(0) = exp(-t) * PI * cos (PI * x) * sin (PI * y);
        gradx(1) = exp(-t) * PI * sin (PI * x) * cos (PI * y);
    }
    if (xt.Size() == 4)
    {
        z = xt(2);
        gradx(0) = exp(-t) * PI * cos (PI * x) * sin (PI * y) * sin (PI * z);
        gradx(1) = exp(-t) * PI * sin (PI * x) * cos (PI * y) * sin (PI * z);
        gradx(2) = exp(-t) * PI * sin (PI * x) * sin (PI * y) * cos (PI * z);
    }

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
        cout << "Error, this is only 4-d solution" << endl;
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

double uFun2_ex_laplace(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(3);

    double res = 0.0;
    res += exp(-t) * (2.0 * M_PI * cos(M_PI * x) - x * M_PI * M_PI * sin (M_PI * x)) * (1 + y) * sin (M_PI * y) * (2 - z) * sin (M_PI * z);
    res += exp(-t) * x * sin (M_PI * x) * (2.0 * M_PI * cos(M_PI * y) - (1 + y) * M_PI * M_PI * sin(M_PI * y)) * (2 - z) * sin (M_PI * z);
    res += exp(-t) * x * sin (M_PI * x) * (1 + y) * sin (M_PI * y) * (2.0 * (-1) * M_PI * cos(M_PI * z) - (2 - z) * M_PI * M_PI * sin(M_PI * z));
    return res;
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

double uFun3_ex(const Vector & xt)
{
    if (xt.Size() != 3)
        cout << "Error, this is only 3-d = 2-d + time solution" << endl;
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    return exp(-t) * x * sin (M_PI * x) * (1 + y) * sin (M_PI * y);
}

double uFun3_ex_dt(const Vector & xt)
{
    return - uFun3_ex(xt);
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

void uFun3_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = exp(-t) * (sin (M_PI * x) + x * M_PI * cos(M_PI * x)) * (1 + y) * sin (M_PI * y);
    gradx(1) = exp(-t) * x * sin (M_PI * x) * (sin (M_PI * y) + (1 + y) * M_PI * cos(M_PI * y));
}

void vminusone_exact(const Vector &x, Vector &vminusone)
{
    vminusone.SetSize(x.Size());
    vminusone = -1.0;
}



void E_exactMat_vec(const Vector &x, Vector &E)
{
    int dim = x.Size();

    if (dim==4)
    {
        E.SetSize(6);

        double s0 = sin(M_PI*x(0)), s1 = sin(M_PI*x(1)), s2 = sin(M_PI*x(2)),
                s3 = sin(M_PI*x(3));
        double c0 = cos(M_PI*x(0)), c1 = cos(M_PI*x(1)), c2 = cos(M_PI*x(2)),
                c3 = cos(M_PI*x(3));

        E(0) =  c0*c1*s2*s3;
        E(1) = -c0*s1*c2*s3;
        E(2) =  c0*s1*s2*c3;
        E(3) =  s0*c1*c2*s3;
        E(4) = -s0*c1*s2*c3;
        E(5) =  s0*s1*c2*c3;
    }
}

void E_exactMat(const Vector &x, DenseMatrix &E)
{
    int dim = x.Size();

    E.SetSize(dim*dim);

    if (dim==4)
    {
        Vector vecE;
        E_exactMat_vec(x, vecE);

        E = 0.0;

        E(0,1) = vecE(0);
        E(0,2) = vecE(1);
        E(0,3) = vecE(2);
        E(1,2) = vecE(3);
        E(1,3) = vecE(4);
        E(2,3) = vecE(5);

        E(1,0) =  -E(0,1);
        E(2,0) =  -E(0,2);
        E(3,0) =  -E(0,3);
        E(2,1) =  -E(1,2);
        E(3,1) =  -E(1,3);
        E(3,2) =  -E(2,3);
    }
}



//f_exact = E + 0.5 * P( curl DivSkew E ), where P is the 4d permutation operator
void f_exactMat(const Vector &x, DenseMatrix &f)
{
    int dim = x.Size();

    f.SetSize(dim,dim);

    if (dim==4)
    {
        f = 0.0;

        double s0 = sin(M_PI*x(0)), s1 = sin(M_PI*x(1)), s2 = sin(M_PI*x(2)),
                s3 = sin(M_PI*x(3));
        double c0 = cos(M_PI*x(0)), c1 = cos(M_PI*x(1)), c2 = cos(M_PI*x(2)),
                c3 = cos(M_PI*x(3));

        f(0,1) =  (1.0 + 1.0  * M_PI*M_PI)*c0*c1*s2*s3;
        f(0,2) = -(1.0 + 0.0  * M_PI*M_PI)*c0*s1*c2*s3;
        f(0,3) =  (1.0 + 1.0  * M_PI*M_PI)*c0*s1*s2*c3;
        f(1,2) =  (1.0 - 1.0  * M_PI*M_PI)*s0*c1*c2*s3;
        f(1,3) = -(1.0 + 0.0  * M_PI*M_PI)*s0*c1*s2*c3;
        f(2,3) =  (1.0 + 1.0  * M_PI*M_PI)*s0*s1*c2*c3;

        f(1,0) =  -f(0,1);
        f(2,0) =  -f(0,2);
        f(3,0) =  -f(0,3);
        f(2,1) =  -f(1,2);
        f(3,1) =  -f(1,3);
        f(3,2) =  -f(2,3);
    }
}
