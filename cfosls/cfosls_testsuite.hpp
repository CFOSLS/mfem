#include "testhead.hpp"

#ifndef MFEM_CFOSLS_TESTSUITE
#define MFEM_CFOSLS_TESTSUITE

namespace mfem
{

//// from parabolic example
double uFun_ex_parab(const Vector & x); // Exact Solution
double uFun_ex_parab_dt(const Vector & xt);
double uFun_ex_parab_laplace(const Vector & xt);
void uFun_ex_parab_gradx(const Vector& xt, Vector& gradx );

double uFun1_ex_parab(const Vector & x); // Exact Solution
double uFun1_ex_parab_dt(const Vector & xt);
double uFun1_ex_parab_laplace(const Vector & xt);
void uFun1_ex_parab_gradx(const Vector& xt, Vector& gradx );

double uFun2_ex_parab(const Vector & x); // EWxact Solution
double uFun2_ex_parab_dt(const Vector & xt);
double uFun2_ex_parab_laplace(const Vector & xt);
void uFun2_ex_parab_gradx(const Vector& xt, Vector& gradx );

double uFun3_ex_parab(const Vector & x); // Exact Solution
double uFun3_ex_parab_dt(const Vector & xt);
double uFun3_ex_parab_laplace(const Vector & xt);
void uFun3_ex_parab_gradx(const Vector& xt, Vector& gradx );
////////////////////////////////////////

//// from wave
double uFun_ex_wave(const Vector & x); // Exact Solution
double uFun_ex_wave_dt(const Vector & xt);
double uFun_ex_wave_dt2(const Vector & xt);
double uFun_ex_wave_laplace(const Vector & xt);
double uFun_ex_wave_dtlaplace(const Vector & xt);
void uFun_ex_wave_gradx(const Vector& xt, Vector& gradx );
void uFun_ex_wave_dtgradx(const Vector& xt, Vector& gradx );

double uFun1_ex_wave(const Vector & x); // Exact Solution
double uFun1_ex_wave_dt(const Vector & xt);
double uFun1_ex_wave_dt2(const Vector & xt);
double uFun1_ex_wave_laplace(const Vector & xt);
double uFun1_ex_wave_dtlaplace(const Vector & xt);
void uFun1_ex_wave_gradx(const Vector& xt, Vector& gradx );
void uFun1_ex_wave_dtgradx(const Vector& xt, Vector& gradx );

double uFun2_ex_wave(const Vector & x); // Exact Solution
double uFun2_ex_wave_dt(const Vector & xt);
double uFun2_ex_wave_dt2(const Vector & xt);
double uFun2_ex_wave_laplace(const Vector & xt);
double uFun2_ex_wave_dtlaplace(const Vector & xt);
void uFun2_ex_wave_gradx(const Vector& xt, Vector& gradx );
void uFun2_ex_wave_dtgradx(const Vector& xt, Vector& gradx );

double uFun3_ex_wave(const Vector & x); // Exact Solution
double uFun3_ex_wave_dt(const Vector & xt);
double uFun3_ex_wave_dt2(const Vector & xt);
double uFun3_ex_wave_laplace(const Vector & xt);
double uFun3_ex_wave_dtlaplace(const Vector & xt);
void uFun3_ex_wave_gradx(const Vector& xt, Vector& gradx );
void uFun3_ex_wave_dtgradx(const Vector& xt, Vector& gradx );

double uFun4_ex_wave(const Vector & x); // Exact Solution
double uFun4_ex_wave_dt(const Vector & xt);
double uFun4_ex_wave_dt2(const Vector & xt);
double uFun4_ex_wave_laplace(const Vector & xt);
double uFun4_ex_wave_dtlaplace(const Vector & xt);
void uFun4_ex_wave_gradx(const Vector& xt, Vector& gradx );
void uFun4_ex_wave_dtgradx(const Vector& xt, Vector& gradx );

double uFun5_ex_wave(const Vector & x); // Exact Solution
double uFun5_ex_wave_dt(const Vector & xt);
double uFun5_ex_wave_dt2(const Vector & xt);
double uFun5_ex_wave_laplace(const Vector & xt);
double uFun5_ex_wave_dtlaplace(const Vector & xt);
void uFun5_ex_wave_gradx(const Vector& xt, Vector& gradx );
void uFun5_ex_wave_dtgradx(const Vector& xt, Vector& gradx );
///////////////////////////////////////////////////////////

///// for laplace equation
double uFunTestLap_ex(const Vector& xt);
double uFunTestLap_lap(const Vector& xt);
void uFunTestLap_grad(const Vector& xt, Vector& grad );
/////////////////////////////////////

double delta_center_ex(const Vector& xt);


///// tests which can be used for any problem
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

////// additional functions (mainly for the hyperbolic problems)
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

void zerovecx_ex(const Vector& xt, Vector& zerovecx );
void zerovec_ex(const Vector& xt, Vector& vecvalue);
void zerovecMat4D_ex(const Vector& xt, Vector& vecvalue);
double zero_ex(const Vector& xt);

void hcurlFun3D_ex(const Vector& xt, Vector& vecvalue);
void curlhcurlFun3D_ex(const Vector& xt, Vector& vecvalue);

void hcurlFun3D_2_ex(const Vector& xt, Vector& vecvalue);
void curlhcurlFun3D_2_ex(const Vector& xt, Vector& vecvalue);

void DivmatFun4D_ex(const Vector& xt, Vector& vecvalue);
void DivmatDivmatFun4D_ex(const Vector& xt, Vector& vecvalue);

// templates for all problems
template <double (*Sfunc)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )>
    void sigmaTemplate_hyper(const Vector& xt, Vector& sigma);
template <void (*bvecfunc)(const Vector&, Vector& )>
    void KtildaTemplate(const Vector& xt, DenseMatrix& Ktilda);
template <void (*bvecfunc)(const Vector&, Vector& )>
        void bbTTemplate(const Vector& xt, DenseMatrix& bbT);
template <void (*bvecfunc)(const Vector&, Vector& )>
    double bTbTemplate(const Vector& xt);
template<void(*bvec)(const Vector & x, Vector & vec)>
    void minbTemplate(const Vector& xt, Vector& minb);

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
        void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
        void bfTemplate(const Vector& xt, Vector& bf);

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
        void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
        double divsigmaTemplate_hyper(const Vector& xt);

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), double (*Slaplace)(const Vector & xt)> \
       double divsigmaTemplate_parab(const Vector& xt);

template <double (*ufunc)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )>
       void sigmaTemplate_parab(const Vector& xt, Vector& sigma);

template<double (*S)(const Vector & xt), double (*d2Sdt2)(const Vector & xt), double (*Slaplace)(const Vector & xt)> \
       double divsigmaTemplate_wave(const Vector& xt);

template <double (*dSdt)(const Vector&), void(*Sgradxvec)(const Vector & x, Vector & gradx) >
       void sigmaTemplate_wave(const Vector& xt, Vector& sigma);

template<void (*Sfullgrad)(const Vector&, Vector& )>
       void sigmaTemplate_lapl(const Vector& xt, Vector& sigma);

template<double (*Slaplace)(const Vector & xt)> double divsigmaTemplate_lapl(const Vector& xt);


template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
    void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
void bdivsigmaTemplate(const Vector& xt, Vector& bf);

template <double (*S)(const Vector&), void (*bvecfunc)(const Vector&, Vector& ), void (*opdivfreevec)(const Vector&, Vector& )>
void minKsigmahatTemplate(const Vector& xt, Vector& minKsigmahatv);

template<double (*S)(const Vector & xt), void(*bvec)(const Vector & x, Vector & vec), void (*opdivfreevec)(const Vector&, Vector& )>
double bsigmahatTemplate(const Vector& xt);

template<double (*S)(const Vector & xt), void (*bvecfunc)(const Vector&, Vector& ),
         void (*opdivfreevec)(const Vector&, Vector& )>
void sigmahatTemplate(const Vector& xt, Vector& sigmahatv);
template<double (*S)(const Vector & xt), void (*bvecfunc)(const Vector&, Vector& ),
         void (*opdivfreevec)(const Vector&, Vector& )>
void minsigmahatTemplate(const Vector& xt, Vector& sigmahatv);

// main base class for FOSLS tests (which has wave, hyperbolic and parabolic tests as childs)
class FOSLS_test
{
protected:
    int dim;

    int nfunc_coeffs;
    int nvec_coeffs;
    int nmat_coeffs;

    Array<FunctionCoefficient*> func_coeffs;
    Array<VectorFunctionCoefficient*> vec_coeffs;
    Array<MatrixFunctionCoefficient*> mat_coeffs;

public:
    FOSLS_test(int dimension, int nfunc_coefficients, int nvec_coefficients, int nmat_coefficients);

    int Dim() const {return dim;}

    virtual FunctionCoefficient* GetU() = 0;
    virtual VectorFunctionCoefficient* GetSigma() = 0;
    virtual FunctionCoefficient* GetRhs() = 0;

    FunctionCoefficient * GetFuncCoeff(int ind) {return func_coeffs[ind];}
    VectorFunctionCoefficient * GetVecCoeff(int ind) {return vec_coeffs[ind];}
    MatrixFunctionCoefficient * GetMatCoeff(int ind) {return mat_coeffs[ind];}
};

// TODO: Rename it to Transport_test and remove the older version of that from everywhere including examples
// func_coeffs:
// [0] = u, scalar unknown
// [1] = f
// [2] = bTb
// vec_coeffs:
// [0] = sigma, vector unknown
// [1] = b
// [2] = -b
// [3] = b * f
// mat_coeffs:
// [0] = Ktilda
// [1] = bbT

class Hyper_test : public FOSLS_test
{
protected:
    int numsol;
protected:
    void Init();
    bool CheckTestConfig();

    template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
             void(*bvec)(const Vector & x, Vector & vec), double (*divbvec)(const Vector & xt)> \
    void SetTestCoeffs ( );

    void SetScalarSFun( double (*S)(const Vector & xt))
    { func_coeffs[0] = new FunctionCoefficient(S);}

    template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
             void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
    void SetDivSigma()
    { func_coeffs[1] = new FunctionCoefficient(divsigmaTemplate_hyper<S, dSdt, Sgradxvec, bvec, divbfunc>);}

    template< void(*bvec)(const Vector & x, Vector & vec)>  \
    void SetScalarBtB()
    {
        func_coeffs[2] = new FunctionCoefficient(bTbTemplate<bvec>);
    }

    template<double (*S)(const Vector & xt), void(*bvec)(const Vector & x, Vector & vec)> \
    void SetSigmaVec()
    {
        vec_coeffs[0] = new VectorFunctionCoefficient(dim, sigmaTemplate_hyper<S,bvec>);
    }

    void SetbVec( void(*bvec)(const Vector & x, Vector & vec))
    { vec_coeffs[1] = new VectorFunctionCoefficient(dim, bvec);}

    template<void(*bvec)(const Vector & x, Vector & vec)> \
    void SetminbVec()
    { vec_coeffs[2] = new VectorFunctionCoefficient(dim, minbTemplate<bvec>);}

    template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
             void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
    void SetbfVec()
    { vec_coeffs[3] = new VectorFunctionCoefficient(dim, bfTemplate<S,dSdt,Sgradxvec,bvec,divbfunc>);}

    template< void(*f2)(const Vector & x, Vector & vec)>  \
    void SetKtildaMat()
    {
        mat_coeffs[0] = new MatrixFunctionCoefficient(dim, KtildaTemplate<f2>);
    }

    template< void(*bvec)(const Vector & x, Vector & vec)>  \
    void SetBBtMat()
    {
        mat_coeffs[1] = new MatrixFunctionCoefficient(dim, bbTTemplate<bvec>);
    }

public:
    Hyper_test(int dimension, int num_solution);

    FunctionCoefficient* GetU() override {return func_coeffs[0];}
    VectorFunctionCoefficient* GetSigma() override {return vec_coeffs[0];}
    FunctionCoefficient* GetRhs() override {return func_coeffs[1];}

    FunctionCoefficient* GetBtB() {return func_coeffs[2];}
    VectorFunctionCoefficient * GetB() {return vec_coeffs[1];}
    VectorFunctionCoefficient * GetMinB() {return vec_coeffs[2];}
    VectorFunctionCoefficient * GetBf() {return vec_coeffs[3];}

    MatrixFunctionCoefficient * GetKtilda() {return mat_coeffs[0];}
    MatrixFunctionCoefficient * GetBBt() {return mat_coeffs[1];}

    int Numsol() const {return numsol;}
};

// func_coeffs:
// [0] = u, scalar unknown
// [1] = f
// vec_coeffs:
// [0] = sigma, vector unknown
// mat_coeffs:
// (empty)

class Parab_test : public FOSLS_test
{
protected:
    int numsol;
protected:
    void Init();
    bool CheckTestConfig();

    template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt),
             double (*Slaplace)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx)> \
    void SetTestCoeffs ( );

    void SetScalarSFun( double (*S)(const Vector & xt))
    { func_coeffs[0] = new FunctionCoefficient(S);}

    template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), double (*Slaplace)(const Vector & xt)> \
    void SetDivSigma()
    { func_coeffs[1] = new FunctionCoefficient(divsigmaTemplate_parab<S, dSdt, Slaplace>);}

    template<double (*f1)(const Vector & xt), void(*f2)(const Vector & x, Vector & vec)> \
    void SetSigmaVec()
    { vec_coeffs[0] = new VectorFunctionCoefficient(dim, sigmaTemplate_parab<f1,f2>); }

public:
    Parab_test(int dimension, int num_solution);

    FunctionCoefficient* GetU() override {return func_coeffs[0];}
    VectorFunctionCoefficient* GetSigma() override {return vec_coeffs[0];}
    FunctionCoefficient* GetRhs() override {return func_coeffs[1];}

    int Numsol() const {return numsol;}
};

// func_coeffs:
// [0] = u, scalar unknown
// [1] = f
// vec_coeffs:
// [0] = sigma, vector unknown
// mat_coeffs:
// (empty)

class Wave_test : public FOSLS_test
{
protected:
    int numsol;
protected:
    void Init();
    bool CheckTestConfig();

    template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), double (*d2Sdt2)(const Vector & xt),\
             double (*Slaplace)(const Vector & xt), double (*dSdtlaplace)(const Vector & xt), \
             void(*Sgradxvec)(const Vector & x, Vector & gradx), void (*dSdtgradxvec)(const Vector&, Vector& ) > \
    void SetTestCoeffs ( );

    void SetScalarSFun( double (*S)(const Vector & xt))
    { func_coeffs[0] = new FunctionCoefficient(S);}

    template<double (*S)(const Vector & xt), double (*d2Sdt2)(const Vector & xt), double (*Slaplace)(const Vector & xt)> \
    void SetDivSigma()
    { func_coeffs[1] = new FunctionCoefficient(divsigmaTemplate_wave<S, d2Sdt2, Slaplace>);}

    template<double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & vec)> \
    void SetSigmaVec()
    { vec_coeffs[0] = new VectorFunctionCoefficient(dim, sigmaTemplate_wave<dSdt,Sgradxvec>); }

public:
    Wave_test(int dimension, int num_solution);

    FunctionCoefficient* GetU() override {return func_coeffs[0];}
    VectorFunctionCoefficient* GetSigma() override {return vec_coeffs[0];}
    FunctionCoefficient* GetRhs() override {return func_coeffs[1];}

    int Numsol() const {return numsol;}
};

// func_coeffs:
// [0] = u, scalar unknown
// [1] = f
// vec_coeffs:
// [0] = sigma, vector unknown
// mat_coeffs:
// (empty)

class Laplace_test : public FOSLS_test
{
protected:
    int numsol;
protected:
    void Init();
    bool CheckTestConfig();

    template<double (*S)(const Vector & xt), void(*Sfullgrad)(const Vector & xt, Vector & gradx),
             double (*Slaplace)(const Vector & xt)> \
    void SetTestCoeffs ( );

    void SetScalarSFun( double (*S)(const Vector & xt))
    { func_coeffs[0] = new FunctionCoefficient(S);}

    template<double (*Slaplace)(const Vector & xt)> \
    void SetdivSigma()
    { func_coeffs[1] = new FunctionCoefficient(divsigmaTemplate_lapl<Slaplace>);}

    template<void(*Sfullgrad)(const Vector & xt, Vector & vec)> \
    void SetSigmaVec()
    { vec_coeffs[0] = new VectorFunctionCoefficient(dim, sigmaTemplate_lapl<Sfullgrad>); }

public:
    Laplace_test(int dimension, int num_solution);

    FunctionCoefficient* GetU() override {return func_coeffs[0];}
    VectorFunctionCoefficient* GetSigma() override {return vec_coeffs[0];}
    FunctionCoefficient* GetRhs() override {return func_coeffs[1];}

    int Numsol() const {return numsol;}
};

// the old class which should be removed whenever possible by Hyper_test (inheriting the FOSLS_test)
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
        sigma = new VectorFunctionCoefficient(dim, sigmaTemplate_hyper<S,bvec>);
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
    { scalardivsigma = new FunctionCoefficient(divsigmaTemplate_hyper<S, dSdt, Sgradxvec, bvec, divbfunc>);}

};

class Transport_test_divfree
{
protected:
    int dim;
    int numsol;
    int numcurl;

public:
    FunctionCoefficient * scalarS;
    FunctionCoefficient * scalardivsigma;         // = dS/dt + div (bS) = div sigma
    FunctionCoefficient * bTb;                    // b^T * b
    FunctionCoefficient * bsigmahat;              // b * sigma_hat
    VectorFunctionCoefficient * sigma;
    VectorFunctionCoefficient * sigmahat;         // sigma_hat = sigma_exact - op divfreepart (curl hcurlpart in 3D)
    VectorFunctionCoefficient * b;
    VectorFunctionCoefficient * minb;
    VectorFunctionCoefficient * bf;
    VectorFunctionCoefficient * bdivsigma;        // b * div sigma = b * initial f (before modifying it due to inhomogenuity)
    MatrixFunctionCoefficient * Ktilda;
    MatrixFunctionCoefficient * bbT;
    VectorFunctionCoefficient * divfreepart;        // additional part added for testing div-free solver
    VectorFunctionCoefficient * opdivfreepart;    // curl of the additional part which is added to sigma_exact for testing div-free solver
    VectorFunctionCoefficient * minKsigmahat;     // - K * sigma_hat
    VectorFunctionCoefficient * minsigmahat;      // -sigma_hat
public:
    Transport_test_divfree (int Dim, int NumSol, int NumCurl);

    int GetDim() {return dim;}
    int GetNumSol() {return numsol;}
    int GetNumCurl() {return numcurl;}
    void SetDim(int Dim) { dim = Dim;}
    void SetNumSol(int NumSol) { numsol = NumSol;}
    void SetNumCurl(int NumCurl) { numcurl = NumCurl;}
    bool CheckTestConfig();

    ~Transport_test_divfree();
private:
    template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
             void(*bvec)(const Vector & x, Vector & vec), double (*divbvec)(const Vector & xt), \
             void(*hcurlvec)(const Vector & x, Vector & vec), void(*opdivfreevec)(const Vector & x, Vector & vec)> \
    void SetTestCoeffs ( );

    void SetScalarSFun( double (*S)(const Vector & xt))
    { scalarS = new FunctionCoefficient(S);}

    template<void(*bvec)(const Vector & x, Vector & vec)>  \
    void SetScalarBtB()
    {
        bTb = new FunctionCoefficient(bTbTemplate<bvec>);
    }

    template<double (*S)(const Vector & xt), void(*bvec)(const Vector & x, Vector & vec)> \
    void SetSigmaVec()
    {
        sigma = new VectorFunctionCoefficient(dim, sigmaTemplate_hyper<S,bvec>);
    }

    template<void(*bvec)(const Vector & x, Vector & vec)> \
    void SetminbVec()
    { minb = new VectorFunctionCoefficient(dim, minbTemplate<bvec>);}

    void SetbVec( void(*bvec)(const Vector & x, Vector & vec) )
    { b = new VectorFunctionCoefficient(dim, bvec);}

    template< void(*bvec)(const Vector & x, Vector & vec)>  \
    void SetKtildaMat()
    {
        Ktilda = new MatrixFunctionCoefficient(dim, KtildaTemplate<bvec>);
    }

    template< void(*bvec)(const Vector & x, Vector & vec)>  \
    void SetBBtMat()
    {
        bbT = new MatrixFunctionCoefficient(dim, bbTTemplate<bvec>);
    }

    template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
             void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
    void SetdivSigma()
    { scalardivsigma = new FunctionCoefficient(divsigmaTemplate_hyper<S, dSdt, Sgradxvec, bvec, divbfunc>);}

    template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
             void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
    void SetbfVec()
    { bf = new VectorFunctionCoefficient(dim, bfTemplate<S,dSdt,Sgradxvec,bvec,divbfunc>);}

    template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
             void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
    void SetbdivsigmaVec()
    { bdivsigma = new VectorFunctionCoefficient(dim, bdivsigmaTemplate<S,dSdt,Sgradxvec,bvec,divbfunc>);}

    void SetDivfreePart( void(*divfreevec)(const Vector & x, Vector & vec))
    { divfreepart = new VectorFunctionCoefficient(dim, divfreevec);}

    void SetOpDivfreePart( void(*opdivfreevec)(const Vector & x, Vector & vec))
    { opdivfreepart = new VectorFunctionCoefficient(dim, opdivfreevec);}

    template<double (*S)(const Vector & xt), void(*bvec)(const Vector & x, Vector & vec), void(*opdivfreevec)(const Vector & x, Vector & vec)> \
    void SetminKsigmahat()
    { minKsigmahat = new VectorFunctionCoefficient(dim, minKsigmahatTemplate<S, bvec, opdivfreevec>);}

    template<double (*S)(const Vector & xt), void(*bvec)(const Vector & x, Vector & vec), void(*opdivfreevec)(const Vector & x, Vector & vec)> \
    void Setbsigmahat()
    { bsigmahat = new FunctionCoefficient(bsigmahatTemplate<S, bvec, opdivfreevec>);}

    template<double (*S)(const Vector & xt), void (*bvec)(const Vector&, Vector& ),
             void(*opdivfreevec)(const Vector & x, Vector & vec)> \
    void Setsigmahat()
    { sigmahat = new VectorFunctionCoefficient(dim, sigmahatTemplate<S, bvec, opdivfreevec>);}

    template<double (*S)(const Vector & xt), void (*bvec)(const Vector&, Vector& ),
             void(*opdivfreevec)(const Vector & x, Vector & vec)> \
    void Setminsigmahat()
    { minsigmahat = new VectorFunctionCoefficient(dim, minsigmahatTemplate<S, bvec, opdivfreevec>);}

};

} // for namespace mfem

#endif
