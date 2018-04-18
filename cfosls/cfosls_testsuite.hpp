#include "testhead.hpp"

#ifndef MFEM_CFOSLS_TESTSUITE
#define MFEM_CFOSLS_TESTSUITE

namespace mfem
{

double uFunTest_ex(const Vector& x); // Exact Solution
double uFunTest_ex_dt(const Vector& xt);
double uFunTest_ex_dt2(const Vector & xt);
double uFunTest_ex_laplace(const Vector & xt);
double uFunTest_ex_dtlaplace(const Vector & xt);
void uFunTest_ex_gradx(const Vector& xt, Vector& grad);
void uFunTest_ex_gradxt(const Vector& xt, Vector& gradxt);
void uFunTest_ex_dtgradx(const Vector& xt, Vector& gradx );

double uFunTestNh_ex(const Vector& x); // Exact Solution
double uFunTestNh_ex_dt(const Vector& xt);
double uFunTestNh_ex_dt2(const Vector & xt);
double uFunTestNh_ex_laplace(const Vector & xt);
double uFunTestNh_ex_dtlaplace(const Vector & xt);
void uFunTestNh_ex_gradx(const Vector& xt, Vector& grad);
void uFunTestNh_ex_gradxt(const Vector& xt, Vector& gradxt);
void uFunTestNh_ex_dtgradx(const Vector& xt, Vector& gradx );

double uFunCylinder_ex(const Vector& xt);
double uFunCylinder_ex_dt(const Vector& xt);
void uFunCylinder_ex_gradx(const Vector& xt, Vector& grad);

void bFunRect2D_ex(const Vector& xt, Vector& b );
double  bFunRect2Ddiv_ex(const Vector& xt);

void bFunCube3D_ex(const Vector& xt, Vector& b );
double  bFunCube3Ddiv_ex(const Vector& xt);

void bFunSphere3D_ex(const Vector& xt, Vector& b );
double  bFunSphere3Ddiv_ex(const Vector& xt);

void bFunCircle2D_ex (const Vector& xt, Vector& b);
double  bFunCircle2Ddiv_ex(const Vector& xt);

void testVectorFun(const Vector& xt, Vector& res);

double testH1fun(Vector& xt);
void testHdivfun(const Vector& xt, Vector& res);

template <double (*Sfunc)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )>
    void sigmaTemplate(const Vector& xt, Vector& sigma);
template <void (*bvecfunc)(const Vector&, Vector& )>
    void KtildaTemplate(const Vector& xt, DenseMatrix& Ktilda);
template <void (*bvecfunc)(const Vector&, Vector& )>
        void bbTTemplate(const Vector& xt, DenseMatrix& bbT);
template <void (*bvecfunc)(const Vector&, Vector& )>
    double bTbTemplate(const Vector& xt);
template<void(*bvec)(const Vector & x, Vector & vec)>
    void minbTemplate(const Vector& xt, Vector& minb);

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
         void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
    double rhsideTemplate(const Vector& xt);

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
        void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
        void bfTemplate(const Vector& xt, Vector& bf);

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
        void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
        double divsigmaTemplate(const Vector& xt);

template<double (*S)(const Vector & xt) > double uNonhomoTemplate(const Vector& xt);


class Transport_test
{
protected:
    int dim;
    int numsol;

public:
    FunctionCoefficient * scalarS;
    FunctionCoefficient * scalardivsigma;         // = dS/dt + div (bS) = div sigma
    FunctionCoefficient * bTb;
    VectorFunctionCoefficient * sigma;
    VectorFunctionCoefficient * b;
    VectorFunctionCoefficient * minb;
    VectorFunctionCoefficient * bf;
    MatrixFunctionCoefficient * Ktilda;
    MatrixFunctionCoefficient * bbT;
public:
    Transport_test (int Dim, int NumSol);

    int GetDim() {return dim;}
    int GetNumSol() {return numsol;}
    void SetDim(int Dim) { dim = Dim;}
    void SetNumSol(int NumSol) { numsol = NumSol;}
    bool CheckTestConfig();

    ~Transport_test () {}
private:
    template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
             void(*bvec)(const Vector & x, Vector & vec), double (*divbvec)(const Vector & xt)> \
    void SetTestCoeffs ( );

    void SetScalarSFun( double (*S)(const Vector & xt))
    { scalarS = new FunctionCoefficient(S);}

    template< void(*bvec)(const Vector & x, Vector & vec)>  \
    void SetScalarBtB()
    {
        bTb = new FunctionCoefficient(bTbTemplate<bvec>);
    }

    template<double (*S)(const Vector & xt), void(*bvec)(const Vector & x, Vector & vec)> \
    void SetSigmaVec()
    {
        sigma = new VectorFunctionCoefficient(dim, sigmaTemplate<S,bvec>);
    }

    void SetbVec( void(*bvec)(const Vector & x, Vector & vec))
    { b = new VectorFunctionCoefficient(dim, bvec);}

    template<void(*bvec)(const Vector & x, Vector & vec)> \
    void SetminbVec()
    { minb = new VectorFunctionCoefficient(dim, minbTemplate<bvec>);}

    template< void(*f2)(const Vector & x, Vector & vec)>  \
    void SetKtildaMat()
    {
        Ktilda = new MatrixFunctionCoefficient(dim, KtildaTemplate<f2>);
    }

    template< void(*bvec)(const Vector & x, Vector & vec)>  \
    void SetBBtMat()
    {
        bbT = new MatrixFunctionCoefficient(dim, bbTTemplate<bvec>);
    }

    template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
             void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
    void SetbfVec()
    { bf = new VectorFunctionCoefficient(dim, bfTemplate<S,dSdt,Sgradxvec,bvec,divbfunc>);}

    template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
             void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
    void SetDivSigma()
    { scalardivsigma = new FunctionCoefficient(divsigmaTemplate<S, dSdt, Sgradxvec, bvec, divbfunc>);}

};

} // for namespace mfem

#endif
