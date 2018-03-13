//                                MFEM (with 4D elements) CFOSLS combined with Petrov-Galerkin formulation
//                                  using S from H1 for 3D/4D hyperbolic equation, no sigma
//
// Compile with: make
//
// Description:  This example code solves a simple 3D/4D hyperbolic problem over [0,1]^{d+1}, d = 3 or 4
//               corresponding to the linear transport equation
//                                  div_(x,t) [b, 1]^T * S      = f
//                       where b is a given vector function (aka velocity),
//						 NO boundary conditions (which work only in case when b * n = 0 pointwise at the domain space boundary)
//						 and initial condition:
//                                  u(x,0)            = 0
//                       The system is solved via minimization principle:
//                                  J(S) = || S ||_H1^2 ----> min
//                       over H1 with a Petrov-Galerkin constraint
//                                  (div_(x,t) [b, 1]^T * S - f, teta)   = 0, teta from L2
//                       Lambda is the corresponding Lagrange multiplier.
//               For tests we use a manufactured exact solution and compute the corresponding r.h.s.
//               We discretize with continuous H1 elements (S) and discontinuous polynomials (lambda).
//
// Solver: MINRES with a block-diagonal preconditioner (using BoomerAMG)

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>

//#define TSHIFTING
//#define TSHIFT (0.9)

#include"cfosls_testsuite.hpp"
#include "divfree_solver_tools.hpp"

//#define BBT_instead_H1norm
//#define EPSILON_SHIFT (0.2)

//#define BoomerAMG_BBT_check
//#define BAinvBT_check
//#define BBT_check

//#define BoomerAMG_check
//#define M_cond
//#define BhAinvBhT_spectral

//#define TIME_STEPPING // not ready yet, problems with interpolation

#define MYZEROTOL (1.0e-13)

using namespace std;
using namespace mfem;
using std::unique_ptr;
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
    /*
    if (inner_niter > 1)
    {
        std::cout << "Implementation is wrong \n";
        for ( int iter = 0; iter < inner_niter; ++iter)
        {
            middleop.Mult(tmp1, tmp2);
            if (iter < inner_niter - 1)
                tmp1 = tmp2;
        }
    }
    else
        middleop.Mult(tmp1, tmp2);
    */
    middleop.Mult(tmp1, tmp2);
    leftmat.Mult(tmp2, y);
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

   dshape.SetSize(dof,dim);
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
      int order = (Tr.OrderW() + el.GetOrder() + el.GetOrder());
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);

      Tr.SetIntPoint (&ip);
      w = ip.weight;// * Tr.Weight();
      CalcAdjugate(Tr.Jacobian(), invdfdx);
      Mult(dshape, invdfdx, dshapedxt);

      Q.Eval(bf, Tr, ip);

      dshapedxt.Mult(bf, bfdshapedxt);

      add(elvect, w, bfdshapedxt, elvect);
   }
}

// Some functions used for analytical tests previously

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

// Templates for Transport_test class

template <double (*S)(const Vector&), void (*bvec)(const Vector&, Vector& )>
    void sigmaTemplate(const Vector& xt, Vector& sigma);

template <void (*bvec)(const Vector&, Vector& )>
    void KtildaTemplate(const Vector& xt, DenseMatrix& Ktilda);

template <void (*bvec)(const Vector&, Vector& )>
        void bbTTemplate(const Vector& xt, DenseMatrix& bbT);

template <void (*bvec)(const Vector&, Vector& )>
    double bTbTemplate(const Vector& xt);

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), \
         void(*Sgradxvec)(const Vector & x, Vector & gradx), \
        void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
        void bfTemplate(const Vector& xt, Vector& bf);

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), \
         void(*Sgradxvec)(const Vector & x, Vector & gradx), \
        void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
        double divsigmaTemplate(const Vector& xt);

#ifdef BBT_instead_H1norm
template <void (*bvec)(const Vector&, Vector& )>
       void bbTepsTemplate(const Vector& xt, DenseMatrix& bbT);
#endif
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
        template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), \
                 void(*Sgradxvec)(const Vector & x, Vector & gradx), \
                 void(*bvec)(const Vector & x, Vector & vec), double (*divbvec)(const Vector & xt)> \
        void SetTestCoeffs ( );

        void SetScalarSFun( double (*S)(const Vector & xt))
        { scalarS = new FunctionCoefficient(S);}

        template< void(*bvec)(const Vector & x, Vector & vec)>  \
        void SetScalarBtBFun()
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

        template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), \
                 void(*Sgradxvec)(const Vector & x, Vector & gradx), \
                 void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
        void SetbfVec()
        { bf = new VectorFunctionCoefficient(dim, bfTemplate<S,dSdt,Sgradxvec,bvec,divbfunc>);}

        template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), \
                 void(*Sgradxvec)(const Vector & x, Vector & gradx), \
                 void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
        void SetDivSigmaFun()
        { scalardivsigma = new FunctionCoefficient(divsigmaTemplate<S, dSdt, Sgradxvec, bvec, divbfunc>);}

};

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), \
         void(*Sgradxvec)(const Vector & x, Vector & gradx),  \
         void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
void Transport_test::SetTestCoeffs ()
{
    SetScalarSFun(S);
    SetbVec(bvec);
    SetbfVec<S, dSdt, Sgradxvec, bvec, divbfunc>();
    SetSigmaVec<S,bvec>();
    SetKtildaMat<bvec>();
    SetScalarBtBFun<bvec>();
    SetDivSigmaFun<S, dSdt, Sgradxvec, bvec, divbfunc>();
    SetBBtMat<bvec>();
    return;
}


bool Transport_test::CheckTestConfig()
{
    if (dim == 4 || dim == 3)
    {
        if (numsol == 0)
            return true;
        if ( numsol == 1 && dim == 3 )
            return true;
        if ( numsol == 2 && dim == 4 )
            return true;
        if ( numsol == 3 && dim == 3 )
            return true;
        if ( numsol == 33 && dim == 4 )
            return true;
        if ( numsol == 4 && dim == 3 )
            return true;
        if ( numsol == 44 && dim == 3 )
            return true;
        if ( numsol == 100 && dim == 3 )
            return true;
        if ( numsol == 200 && dim == 3 )
            return true;
        if ( numsol == 5 && dim == 3 )
            return true;
        if ( numsol == 55 && dim == 4 )
            return true;
        if ( numsol == 444 && dim == 4 )
            return true;
        if ( numsol == 1000 && dim == 3 )
            return true;
        if ( numsol == 8 && dim == 3 )
            return true;
        if (numsol == 10 && dim == 4)
            return true;
        if (numsol == -3 && dim == 3)
            return true;
        if (numsol == -4 && dim == 4)
            return true;
        return false;
    }
    else
        return false;
}

Transport_test::Transport_test (int Dim, int NumSol)
{
    dim = Dim;
    numsol = NumSol;

    if ( CheckTestConfig() == false )
        std::cout << "Inconsistent dim and numsol \n" << std::flush;
    else
    {
        if (numsol == -3) // 3D test for the paper
        {
            SetTestCoeffs<&uFunTest_ex, &uFunTest_ex_dt, &uFunTest_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex>();
        }
        if (numsol == -4) // 4D test for the paper
        {
            SetTestCoeffs<&uFunTest_ex, &uFunTest_ex_dt, &uFunTest_ex_gradx, &bFunCube3D_ex, &bFunCube3Ddiv_ex>();
        }
        if (numsol == 0)
        {
            //std::cout << "The domain is rectangular or cubic, velocity does not"
                         //" satisfy divergence condition" << std::endl << std::flush;
            SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex>();
            //SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_gradx, &bFunCube3D_ex, &bFunCube3Ddiv_ex>();
        }
        if (numsol == 1)
        {
            //std::cout << "The domain must be a cylinder over a circle" << std::endl << std::flush;
            SetTestCoeffs<&uFun2_ex, &uFun2_ex_dt, &uFun2_ex_gradx, &bFunCircle2D_ex, &bFunCircle2Ddiv_ex>();
        }
        if (numsol == 100)
        {
            //std::cout << "The domain must be a cylinder over a unit square" << std::endl << std::flush;
            SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex>();
        }
        if (numsol == 200)
        {
            //std::cout << "The domain must be a cylinder over a unit circle" << std::endl << std::flush;
            SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_gradx, &bFunCircle2D_ex, &bFunCircle2Ddiv_ex>();
        }
        if (numsol == 2)
        {
            //std::cout << "The domain must be a cylinder over a 3D cube, velocity does not"
                         //" satisfy divergence condition" << std::endl << std::flush;
            SetTestCoeffs<&uFun3_ex, &uFun3_ex_dt, &uFun3_ex_gradx, &bFun_ex, &bFundiv_ex>();
        }
        if (numsol == 3)
        {
            //std::cout << "The domain must be a cylinder over a circle" << std::endl << std::flush;
            SetTestCoeffs<&uFun4_ex, &uFun4_ex_dt, &uFun4_ex_gradx, &bFunCircle2D_ex, &bFunCircle2Ddiv_ex>();
        }
        if (numsol == 4) // no exact solution in fact, ~ unsuccessfully trying to get a picture from the report
        {
            //std::cout << "The domain must be a cylinder over a circle" << std::endl << std::flush;
            //std::cout << "Using new interface \n";
            SetTestCoeffs<&uFun5_ex, &uFun5_ex_dt, &uFun5_ex_gradx, &bFunCircle2D_ex, &bFunCircle2Ddiv_ex>();
        }
        if (numsol == 44) // no exact solution in fact, ~ unsuccessfully trying to get a picture from the report
        {
            //std::cout << "The domain must be a cylinder over a circle" << std::endl << std::flush;
            //std::cout << "Using new interface \n";
            SetTestCoeffs<&uFun6_ex, &uFun6_ex_dt, &uFun6_ex_gradx, &bFunCircle2D_ex, &bFunCircle2Ddiv_ex>();
        }
        if (numsol == 8)
        {
            //std::cout << "The domain must be a cylinder over a circle" << std::endl << std::flush;
            SetTestCoeffs<&uFunCylinder_ex, &uFunCylinder_ex_dt, &uFunCylinder_ex_gradx, &bFunCircle2D_ex, &bFunCircle2Ddiv_ex>();
        }
        if (numsol == 5)
        {
            //std::cout << "The domain must be a cylinder over a square" << std::endl << std::flush;
            SetTestCoeffs<&uFun4_ex, &uFun4_ex_dt, &uFun4_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex>();
        }
        if (numsol == 1000)
        {
            //std::cout << "The domain must be a cylinder over a square" << std::endl << std::flush;
            SetTestCoeffs<&uFun_ex, &uFun_ex_dt, &uFun_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex>();
        }
        if (numsol == 33)
        {
            //std::cout << "The domain must be a cylinder over a sphere" << std::endl << std::flush;
            SetTestCoeffs<&uFun33_ex, &uFun33_ex_dt, &uFun33_ex_gradx, &bFunSphere3D_ex, &bFunSphere3Ddiv_ex>();
        }
        if (numsol == 444) // no exact solution in fact, ~ unsuccessfully trying to get something beauitiful
        {
            //std::cout << "The domain must be a cylinder over a sphere" << std::endl << std::flush;
            SetTestCoeffs<&uFun66_ex, &uFun66_ex_dt, &uFun66_ex_gradx, &bFunSphere3D_ex, &bFunSphere3Ddiv_ex>();
        }
        if (numsol == 55)
        {
            //std::cout << "The domain must be a cylinder over a cube" << std::endl << std::flush;
            SetTestCoeffs<&uFun33_ex, &uFun33_ex_dt, &uFun33_ex_gradx, &bFunCube3D_ex, &bFunCube3Ddiv_ex>();
        }
        if (numsol == 10)
        {
            SetTestCoeffs<&uFun10_ex, &uFun10_ex_dt, &uFun10_ex_gradx, &bFunCube3D_ex, &bFunCube3Ddiv_ex>();
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
    int numsol          = 0;
    double sx = 1.0;
    double sy = 1.0;
    double sz = 0.1;
    int Nx = 10;
    int Ny = 10;
    int Nz = 1;
    //double st = 0.0;

    int ser_ref_levels  = 0;
    int par_ref_levels  = 3;

    const char *formulation = "cfosls";
    bool regularization = false;     // turned out to be a bad idea, since BBT turned out to be non-singular

    bool with_epsilon = true;

    bool aniso_refine = false;
    bool refine_t_first = false;
    bool refine_only_t = false;
    bool refine_only_x = true;

    // solver options
    int prec_option = 1; //defines whether to use preconditioner or not, and which one. (*) 4 and 5 produce almost the same results
    bool direct_solver = false;

    // variables(options) derived from prec_option
    bool with_prec;           // to be defined from prec_option value
    bool identity_Schur ;     // to be defined from prec_option value
    double identity_scale;

    // level gap between coarse an fine grid (T_H and T_h)
    int level_gap = 2;

    const char *mesh_file = "../data/cube_3d_moderate.mesh";

    //const char * mesh_file = "../data/cube4d_low.MFEM";
    //const char * mesh_file = "../data/cube4d.MFEM";
    //const char * mesh_file = "dsadsad";
    //const char * mesh_file = "../data/orthotope3D_moderate.mesh";
    //const char * mesh_file = "../data/orthotope3D_fine.mesh";
    //const char * mesh_file = "../data/square_2d_moderate.mesh";

    //const char * meshbase_file = "../data/sphere3D_0.1to0.2.mesh";
    //const char * meshbase_file = "../data/sphere3D_0.05to0.1.mesh";
    //const char * meshbase_file = "../data/sphere3D_veryfine.mesh";
    //const char * meshbase_file = "../data/beam-tet.mesh";
    //const char * meshbase_file = "../data/orthotope3D_moderate.mesh";
    //const char * meshbase_file = "../data/orthotope3D_fine.mesh";
    //const char * meshbase_file = "../data/square_2d_moderate.mesh";
    //const char * meshbase_file = "../data/square_2d_fine.mesh";
    //const char * meshbase_file = "dsadsad";
    //const char * meshbase_file = "../data/circle_fine_0.1.mfem";
    //const char * meshbase_file = "../data/circle_moderate_0.2.mfem";

    int feorder         = 0;

    if (verbose)
        cout << "Solving within СFOSLS & Petrov-Galerkin formulation of the transport equation with MFEM \n";

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&feorder, "-o", "--feorder",
                   "Finite element order (polynomial degree).");
    args.AddOption(&ser_ref_levels, "-sref", "--sref",
                   "Number of serial refinements.");
    args.AddOption(&par_ref_levels, "-pref", "--pref",
                   "Number of parallel refinements.");
    args.AddOption(&nDimensions, "-dim", "--whichD",
                   "Dimension of the space-time problem.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&prec_option, "-precopt", "--prec-option",
                   "Preconditioner choice.");
    args.AddOption(&formulation, "-form", "--formul",
                   "Formulation to use.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&level_gap, "-lg", "--level-gap",
                   "level gap between coarse an fine grid (T_H and T_h).");

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
    case 1:
        with_prec = true;
        identity_Schur = false;
        break;
    case 2:
        with_prec = true;
        identity_Schur = true;
        break;
    default: // no preconditioner (default)
        with_prec = false;
        identity_Schur = false;
        break;
    }

    if (verbose)
        std::cout << "Number of mpi processes: " << num_procs << "\n";

    StopWatch chrono;

    //DEFAULT LINEAR SOLVER OPTIONS
    int max_iter = 150000;
    double rtol = 1e-12;//1e-7;//1e-9;
    double atol = 1e-14;//1e-9;//1e-12;

    Mesh *mesh = NULL;

    shared_ptr<ParMesh> pmesh;

    if (nDimensions == 3 || nDimensions == 4)
    {
        if (verbose)
            cout << "Reading a " << nDimensions << "d mesh from the file " << mesh_file << "\n";
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

    //mesh = new Mesh(Nx, Ny, Nz, Element::HEXAHEDRON, 1, sx, sy, sz);
    Array<Refinement> refs(mesh->GetNE());

    for (int l = 0; l < ser_ref_levels; l++)
    {
        if (aniso_refine)
        {
            if (refine_only_t)
                for (int i = 0; i < mesh->GetNE(); i++)
                    refs[i] = Refinement(i, 4);
            else if (refine_only_x)
                for (int i = 0; i < mesh->GetNE(); i++)
                    refs[i] = Refinement(i, 3);
            else
            {
                if (verbose)
                    std::cout << "Don't know what to do \n";
                MPI_Finalize();
                return 0;
            }
            mesh->GeneralRefinement(refs, -1, -1);
        }
        else
            mesh->UniformRefinement();
    }

    pmesh = make_shared<ParMesh>(comm, *mesh);
    delete mesh;

    /*
    if (mesh) // if only serial mesh was generated previously, parallel mesh is initialized here
    {
        for (int l = 0; l < ser_ref_levels; l++)
            mesh->UniformRefinement();

        if ( verbose )
            cout << "Creating parmesh(" << nDimensions <<
                    "d) from the serial mesh (" << nDimensions << "d) \n" << flush;
        pmesh = make_shared<ParMesh>(comm, *mesh);
        delete mesh;
    }
    */

    MFEM_ASSERT(level_gap>0 && level_gap<=par_ref_levels, "invalid level_gap!");
    for (int l = 0; l < par_ref_levels-level_gap; l++)
    {
        if (aniso_refine)
        {
            Array<Refinement> refs(pmesh->GetNE());
            if (refine_only_t)
                for (int i = 0; i < pmesh->GetNE(); i++)
                    refs[i] = Refinement(i, 4);
            else if (refine_only_x)
                for (int i = 0; i < pmesh->GetNE(); i++)
                    refs[i] = Refinement(i, 3);
            else
            {
                if (verbose)
                    std::cout << "Don't know what to do \n";
                MPI_Finalize();
                return 0;
            }
            pmesh->GeneralRefinement(refs, -1, -1);
        }
        else
            pmesh->UniformRefinement();
    }

    double h_min, h_max, kappa_min, kappa_max;
    pmesh->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);
    if (verbose)
        std::cout << "coarse mesh steps: min " << h_min << " max " << h_max << "\n";

    double regparam;
    if (regularization)
    {
        regparam = - h_min * h_min;
        regparam *= 1.0;
        if (verbose)
        {
            std::cout << "regularization is ON \n";
            std::cout << "regularization parameter: " << regparam << "\n";
        }
    }
    else
        if (verbose)
            std::cout << "regularization is OFF \n";


    int dim = nDimensions;
    FiniteElementCollection *l2_coll = new L2_FECollection(feorder, dim);
    if(verbose)
        cout << "L2: order " << feorder << "\n";

    ParFiniteElementSpace *W_space = new ParFiniteElementSpace(pmesh.get(), l2_coll); // space for mu
    ParFiniteElementSpace *coarseW_space = new ParFiniteElementSpace(pmesh.get(), l2_coll); // space for mu

    ParFiniteElementSpace *coarseW_space_help = new ParFiniteElementSpace(pmesh.get(), l2_coll); // space for mu

    HypreParMatrix * P_W;
    Array<HypreParMatrix*> P_Ws(level_gap);
    for (int l = 0; l < level_gap; l++)
    {
        coarseW_space_help->Update();

        if (aniso_refine)
        {
            Array<Refinement> refs(pmesh->GetNE());
            if (refine_only_t)
                for (int i = 0; i < pmesh->GetNE(); i++)
                    refs[i] = Refinement(i, 4);
            else if (refine_only_x)
                for (int i = 0; i < pmesh->GetNE(); i++)
                    refs[i] = Refinement(i, 3);
            else
            {
                if (verbose)
                    std::cout << "Don't know what to do \n";
                MPI_Finalize();
                return 0;
            }
            pmesh->GeneralRefinement(refs, -1, -1);
        }
        else
            pmesh->UniformRefinement();
        //pmesh->UniformRefinement();

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

    // 6. Define a parallel finite element space on the parallel mesh.

    if (dim == 4)
        MFEM_ASSERT(feorder==0, "Only lowest order elements are supported in 4D!");
    FiniteElementCollection *h1_coll;
    if (dim == 4)
    {
        h1_coll = new LinearFECollection;
        if (verbose)
            cout << "H1 in 4D: linear elements are used \n";
    }
    else
    {
        h1_coll = new H1_FECollection(feorder+1, dim);
        if(verbose)
            cout << "H1: order " << feorder + 1 << " for 3D \n";
    }

    ParFiniteElementSpace *H_space = new ParFiniteElementSpace(pmesh.get(), h1_coll);

    HYPRE_Int dimH = H_space->GlobalTrueVSize();
    HYPRE_Int dimW = W_space->GlobalTrueVSize();

    if (verbose)
    {
       std::cout << "***********************************************************\n";
       std::cout << "dim(H) = " << dimH << ", ";
       std::cout << "dim(W_fine) = " << dimW << ", ";
       std::cout << "dim(W_coarse) = " << P_W->Width() << ", ";
       std::cout << "dim(H+W_coarse) = " << dimH + P_W->Width() << "\n";
       std::cout << "Number of primary unknowns (S): " << dimH << "\n";
       std::cout << "Number of equations in the constraint: " << P_W->Width() << "\n";
       std::cout << "***********************************************************\n";
    }

    MFEM_ASSERT(dimH > P_W->Width(), "Overconstrained system!");

    // 7. Define the two BlockStructure of the problem.  block_offsets is used
    //    for Vector based on dof (like ParGridFunction or ParLinearForm),
    //    block_trueOffstes is used for Vector based on trueDof (HypreParVector
    //    for the rhs and solution of the linear system).  The offsets computed
    //    here are local to the processor.
    int numblocks = 2;

    Array<int> block_offsets(numblocks + 1); // number of variables + 1
    block_offsets[0] = 0;
    block_offsets[1] = H_space->GetVSize();
    block_offsets[2] = W_space->GetVSize();
    block_offsets.PartialSum();

    Array<int> block_trueOffsets(numblocks + 1); // number of variables + 1
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = H_space->TrueVSize();
    block_trueOffsets[2] = W_space->TrueVSize();
    block_trueOffsets.PartialSum();

    Array<int> block_finalOffsets(numblocks + 1); // number of variables + 1
    block_finalOffsets[0] = 0;
    block_finalOffsets[1] = H_space->TrueVSize();
    block_finalOffsets[2] = coarseW_space->TrueVSize();
    block_finalOffsets.PartialSum();

    BlockVector x(block_offsets), rhs(block_offsets);
    BlockVector trueX(block_finalOffsets), trueRhs(block_trueOffsets);
    x = 0.0;
    rhs = 0.0;
    trueX = 0.0;
    trueRhs = 0.0;

    Transport_test Mytest(nDimensions,numsol);

    ParGridFunction *S_exact = new ParGridFunction(H_space);
    S_exact->ProjectCoefficient(*(Mytest.scalarS));

    x.GetBlock(0) = *S_exact;

   // 8. Define the constant/function coefficients.
   ConstantCoefficient zero(.0);
   Vector zerovec(dim);
   zerovec = 0.0;
   VectorConstantCoefficient zerov(zerovec);

   //----------------------------------------------------------
   // Setting boundary conditions.
   //----------------------------------------------------------

   Array<int> ess_bdrS(pmesh->bdr_attributes.Max());
   ess_bdrS = 0;
   ess_bdrS[0] = 1; // t = 0
   Array<int> ess_bdrSigma(pmesh->bdr_attributes.Max());
   ess_bdrSigma = 0;
   //-----------------------

   // 9. Define the parallel grid function and parallel linear forms, solution
   //    vector and rhs.

   ParLinearForm *rhsS_form(new ParLinearForm);
   rhsS_form->Update(H_space, rhs.GetBlock(0), 0);
   rhsS_form->AddDomainIntegrator(new DomainLFIntegrator(zero));
   rhsS_form->Assemble();
   //rhsS_form->Print(std::cout);

   ParLinearForm *rhslam_form(new ParLinearForm);
   rhslam_form->Update(W_space, rhs.GetBlock(1), 0);
   rhslam_form->AddDomainIntegrator(new DomainLFIntegrator(*Mytest.scalardivsigma));
   rhslam_form->Assemble();

   // 10. Assemble the finite element matrices for the CFOSLS operator

   //---------------
   //  A Block: block-diagonal with stiffness matrix for H1
   //---------------

   //----------------
   //  A Block: (S, p) + (grad S, grad p) with S and p from H1
   //-----------------

   ParBilinearForm *A_block(new ParBilinearForm(H_space));
   HypreParMatrix *A;
   A_block->AddDomainIntegrator(new MassIntegrator);
#ifdef BBT_instead_H1norm
   MatrixFunctionCoefficient bbTeps_matcoeff(dim, bbTepsTemplate<bFunRect2D_ex>);
   if (verbose)
       std::cout << "Using bbT as a matrix coefficient for diffusion integrator \n";
   if (with_epsilon)
       A_block->AddDomainIntegrator(new DiffusionIntegrator(bbTeps_matcoeff));
   else
       A_block->AddDomainIntegrator(new DiffusionIntegrator(*Mytest.bbT));
#else
   A_block->AddDomainIntegrator(new DiffusionIntegrator);
#endif
   A_block->Assemble();
   A_block->EliminateEssentialBC(ess_bdrS, x.GetBlock(0), *rhsS_form);
   A_block->Finalize();
   A = A_block->ParallelAssemble();

   //---------------
   //  B Block: block with Lagrange multiplier's stuff
   //---------------

   //----------------
   //  B Block: divergence constraint
   //-----------------

   HypreParMatrix *B, *BT;

   ParMixedBilinearForm *B_block(new ParMixedBilinearForm(H_space, W_space));
   // assuming that div b = 0, then div ([b,1]^T S) =  [b,1]^T * grad S
   B_block->AddDomainIntegrator(new MixedDirectionalDerivativeIntegrator(*Mytest.b));
   B_block->Assemble();
   B_block->EliminateTrialDofs(ess_bdrS, x.GetBlock(0), *rhslam_form);
   B_block->Finalize();
   auto B_tmp = B_block->ParallelAssemble();
   B = ParMult(P_W->Transpose(), B_tmp);

   BT = B->Transpose();

   HypreParMatrix *W11;
   if (regularization)
   {
       ConstantCoefficient h2coeff(regparam*regparam);

       ParBilinearForm *W11_block(new ParBilinearForm(W_space));
       W11_block->AddDomainIntegrator(new MassIntegrator(h2coeff));
       W11_block->Assemble();
       W11_block->Finalize();
       auto W11tmp = W11_block->ParallelAssemble();
       auto W11tmp2 = ParMult(P_W->Transpose(), W11tmp);
       W11 = ParMult(W11tmp2,P_W);
   }

   //=======================================================
   // Assembling the righthand side
   //-------------------------------------------------------

  rhsS_form->ParallelAssemble(trueRhs.GetBlock(0));
  rhslam_form->ParallelAssemble(trueRhs.GetBlock(1));

  //========================================================
  // Checking residuals in the constraints on exact solutions (serial version)
  //--------------------------------------------------------

  Vector TrueS(H_space->TrueVSize());
  S_exact->ParallelProject(TrueS);

  Vector resW(coarseW_space->TrueVSize());
  Vector tempW(coarseW_space->TrueVSize());
  B->Mult(TrueS, resW);
  P_W->MultTranspose(trueRhs.GetBlock(1), tempW);
  resW -= tempW;

  double norm_resW = resW.Norml2() / sqrt (resW.Size());
  double norm_rhsW = trueRhs.GetBlock(1).Norml2() / sqrt (trueRhs.GetBlock(1).Size());

  std::cout << "Residuals in constraints for exact solution: \n";
  std::cout << "norm_resW = " << norm_resW << "\n";
  std::cout << "rel. norm_resW = " << norm_resW / norm_rhsW << "\n";

  //if (verbose)
      //std::cout << "Residual computation is not implemented for this problem \n";

  //========================================================
  // Checking functional on exact solutions
  //--------------------------------------------------------

  Vector energyS(H_space->TrueVSize());
  A->Mult(TrueS, energyS);

  double local_energyS = energyS * TrueS;

  double global_energyS;
  MPI_Reduce(&local_energyS, &global_energyS, 1,
             MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  global_energyS = sqrt(global_energyS);
  if (verbose)
  {
      std::cout << "Maybe without dividing by the vector size: \n";
      std::cout << "TrueS H1 norm = " << global_energyS << "\n";
      std::cout << "Quadratic functional on exact solution = " << global_energyS * global_energyS << "\n";
  }

  //MPI_Finalize();
  //return 0;

  //=======================================================
  // Assembling the Matrix
  //-------------------------------------------------------


  BlockOperator *CFOSLSop = new BlockOperator(block_finalOffsets);
  CFOSLSop->SetBlock(0,0, A);
  CFOSLSop->SetBlock(0,1, BT);
  CFOSLSop->SetBlock(1,0, B);
  if (regularization)
  {
      CFOSLSop->SetBlock(1,1, W11);
  }


   if (verbose)
       cout << "Final saddle point matrix assembled \n" << flush;
   MPI_Barrier(MPI_COMM_WORLD);


#ifdef BoomerAMG_BBT_check
   {
       auto BBT = ParMult(B,BT);

       if (verbose)
           std::cout << "Checking iteration count for BoomerAMG for matrix BBT \n";

       MINRESSolver solver(MPI_COMM_WORLD);
       solver.SetAbsTol(1.0e-10);
       solver.SetRelTol(1.0e-10);
       solver.SetMaxIter(100000);
       solver.SetOperator(*BBT);
       Solver * Ainv;
       // using BoomerAMG
       Ainv = new HypreBoomerAMG(*BBT);
       ((HypreBoomerAMG*)Ainv)->SetPrintLevel(0);
       ((HypreBoomerAMG*)Ainv)->iterative_mode = true;
       solver.SetPreconditioner(*Ainv);

       Vector testX(BBT->Height());
       testX = 0.0;
       Vector testRhs(BBT->Height());
       testRhs = 1.0;
       for ( int i = 0; i < BBT->Height(); ++i)
           testRhs(i) = i * 1.0 / (i + 3.0) * (3.0 * i - 4.0);

       solver.SetPrintLevel(0);
       solver.Mult(testRhs, testX);

       chrono.Stop();


       if (verbose) // iterative solver reports about its convergence
       {
          if (solver.GetConverged())
             std::cout << "MINRES for BBT with BoomerAMG preconditioner converged in " << solver.GetNumIterations()
                       << " iterations with a residual norm of " << solver.GetFinalNorm() << ".\n";
          else
             std::cout << "MINRES did not converge in " << solver.GetNumIterations()
                       << " iterations. Residual norm is " << solver.GetFinalNorm() << ".\n";
          std::cout << "MINRES solver took " << chrono.RealTime() << "s. \n";
       }

       //MPI_Finalize();
       //return 0;
   }
#endif


#ifdef BoomerAMG_check
   {
       if (verbose)
           std::cout << "Checking iteration count for BoomerAMG for matrix A \n";

       MINRESSolver solver(MPI_COMM_WORLD);
       solver.SetAbsTol(atol);
       solver.SetRelTol(rtol);
       solver.SetMaxIter(max_iter);
       solver.SetOperator(*A);
       Solver * Ainv;
       // using BoomerAMG
       Ainv = new HypreBoomerAMG(*A);
       ((HypreBoomerAMG*)Ainv)->SetPrintLevel(0);
       ((HypreBoomerAMG*)Ainv)->iterative_mode = false;
       solver.SetPreconditioner(*Ainv);

       Vector testX(A->Height());
       testX = 0.0;
       Vector testRhs(A->Height());
       testRhs = 1.0;

       solver.SetPrintLevel(0);
       solver.Mult(testRhs, testX);

       chrono.Stop();


       if (verbose) // iterative solver reports about its convergence
       {
          if (solver.GetConverged())
             std::cout << "MINRES converged in " << solver.GetNumIterations()
                       << " iterations with a residual norm of " << solver.GetFinalNorm() << ".\n";
          else
             std::cout << "MINRES did not converge in " << solver.GetNumIterations()
                       << " iterations. Residual norm is " << solver.GetFinalNorm() << ".\n";
          std::cout << "MINRES solver took " << chrono.RealTime() << "s. \n";
       }

       MPI_Finalize();
       return 0;
   }
#endif

#ifdef M_cond
   {
       ParBilinearForm *M_block(new ParBilinearForm(W_space));
       M_block->AddDomainIntegrator(new MassIntegrator);
       M_block->Assemble();
       M_block->Finalize();
       auto Mtmp = M_block->ParallelAssemble();
       auto Mtmp2 = ParMult(P_W->Transpose(), Mtmp);
       HypreParMatrix * M = ParMult(Mtmp2,P_W);

       Array<double> eigenvalues;
       int nev = 20;
       int seed = 75;
       {

           HypreLOBPCG * lobpcg = new HypreLOBPCG(MPI_COMM_WORLD);

           lobpcg->SetNumModes(nev);
           lobpcg->SetRandomSeed(seed);
           lobpcg->SetMaxIter(600);
           lobpcg->SetTol(1e-8);
           lobpcg->SetPrintLevel(1);
           // checking for M
           lobpcg->SetOperator(*M);

           // 4. Compute the eigenmodes and extract the array of eigenvalues. Define a
           //    parallel grid function to represent each of the eigenmodes returned by
           //    the solver.
           lobpcg->Solve();
           lobpcg->GetEigenvalues(eigenvalues);

           std::cout << "The computed minimal eigenvalues for M are: \n";
           eigenvalues.Print();
       }

       double beta = eigenvalues[0] * 1000.0; // should be enough
       Operator * revM_op = new MyAXPYOperator(*M, beta, -1.0);
       {
           HypreLOBPCG * lobpcg2 = new HypreLOBPCG(MPI_COMM_WORLD);

           lobpcg2->SetNumModes(nev);
           lobpcg2->SetRandomSeed(seed);
           lobpcg2->SetMaxIter(600);
           lobpcg2->SetTol(1e-10);
           lobpcg2->SetPrintLevel(1);
           // checking for beta * Id - B * Ainv * BT
           lobpcg2->SetOperator(*revM_op);

           // 4. Compute the eigenmodes and extract the array of eigenvalues. Define a
           //    parallel grid function to represent each of the eigenmodes returned by
           //    the solver.
           lobpcg2->Solve();
           lobpcg2->GetEigenvalues(eigenvalues);

           std::cout << "The computed maximal eigenvalues for M are: \n";
           for ( int i = 0; i < nev; ++i)
               eigenvalues[i] = beta - eigenvalues[i];
           eigenvalues.Print();
       }

   }

#endif

#ifdef BAinvBT_check
   {
       // 1. form BAinvBT
#ifndef BBT_check
       Solver * Ainv;
       // using BoomerAMG
       //Ainv = new HypreBoomerAMG(*A);
       //((HypreBoomerAMG*)Ainv)->SetPrintLevel(0);
       //((HypreBoomerAMG*)Ainv)->iterative_mode = false;
       // using HyprePCG
       Ainv = new HyprePCG(*A);
       ((HyprePCG*)Ainv)->SetTol(1.0e-8);
#endif
       //int inner_niter = 10;
#ifdef BhAinvBhT_spectral
       HypreParMatrix *Bsp, *BspT;

       ParMixedBilinearForm *Bsp_block(new ParMixedBilinearForm(H_space, W_space));
       Bsp_block->AddDomainIntegrator(new MixedDirectionalDerivativeIntegrator(*Mytest.b));
       Bsp_block->Assemble();
       Bsp_block->EliminateTrialDofs(ess_bdrS, x.GetBlock(0), *rhslam_form);
       Bsp_block->Finalize();
       Bsp = Bsp_block->ParallelAssemble();
       BspT = Bsp->Transpose();
       MyOperator * BAinvBT_op = new MyOperator(*Bsp, *Ainv, *BspT);
#else
       //MyOperator * BAinvBT_op = new MyOperator(*B, *Ainv, *BT, inner_niter);
#ifdef BBT_check
       IdentityOperator * Id_op = new IdentityOperator(B->Height());
       MyOperator * BAinvBT_op = new MyOperator(*B, *Id_op, *BT);
#else
       MyOperator * BAinvBT_op = new MyOperator(*B, *Ainv, *BT);
#endif

#endif

       /*
       Vector onesvec(dim);
       onesvec = 1.0;
       VectorConstantCoefficient onev(onesvec);

       HypreParMatrix *Btest, *BtestT;
       ParMixedBilinearForm *Btest_block(new ParMixedBilinearForm(H_space, W_space));
       Btest_block->AddDomainIntegrator(new MixedDirectionalDerivativeIntegrator(onev));
       Btest_block->Assemble();
       Btest_block->EliminateTrialDofs(ess_bdrS, x.GetBlock(0), *rhslam_form);
       Btest_block->Finalize();
       auto Btest_tmp = Btest_block->ParallelAssemble();
       Btest = ParMult(P_W->Transpose(), Btest_tmp);
       BtestT = Btest->Transpose();

       MyOperator * BAinvBT_op = new MyOperator(*Btest, *Ainv, *BtestT);
       */

       // 3. call eigensolver to compute minimal eigenvalues of BAinvBT
       Array<double> eigenvalues;
       int nev = 20;
       int seed = 75;
       {
           HypreLOBPCG * lobpcg = new HypreLOBPCG(MPI_COMM_WORLD);

           lobpcg->SetNumModes(nev);
           lobpcg->SetRandomSeed(seed);
           lobpcg->SetMaxIter(600);
           lobpcg->SetTol(1e-12);
           lobpcg->SetPrintLevel(1);
           // checking for B * Ainv * BT
           lobpcg->SetOperator(*BAinvBT_op);

           /*
           ParBilinearForm *M_block(new ParBilinearForm(W_space));
           M_block->AddDomainIntegrator(new MassIntegrator);
           M_block->Assemble();
           M_block->Finalize();
           auto Mtmp = M_block->ParallelAssemble();
           auto Mtmp2 = ParMult(P_W->Transpose(), Mtmp);
           HypreParMatrix * M = ParMult(Mtmp2,P_W);

           lobpcg->SetMassMatrix(*M);
           */


           // 4. Compute the eigenmodes and extract the array of eigenvalues. Define a
           //    parallel grid function to represent each of the eigenmodes returned by
           //    the solver.
           lobpcg->Solve();
           lobpcg->GetEigenvalues(eigenvalues);

           std::cout << "The computed minimal eigenvalues for BAinvBT are: \n";
           eigenvalues.Print();
       }

       // 5. form beta * Id - BAinvBT
       double beta = eigenvalues[0] * 1000.0; // should be enough
       Operator * revBAinvBT_op = new MyAXPYOperator(*BAinvBT_op, beta, -1.0);
       {
           HypreLOBPCG * lobpcg2 = new HypreLOBPCG(MPI_COMM_WORLD);

           lobpcg2->SetNumModes(nev);
           lobpcg2->SetRandomSeed(seed);
           lobpcg2->SetMaxIter(600);
           lobpcg2->SetTol(1e-12);
           lobpcg2->SetPrintLevel(1);
           // checking for beta * Id - B * Ainv * BT
           lobpcg2->SetOperator(*revBAinvBT_op);

           /*
           ParBilinearForm *M_block(new ParBilinearForm(W_space));
           M_block->AddDomainIntegrator(new MassIntegrator);
           M_block->Assemble();
           M_block->Finalize();
           auto Mtmp = M_block->ParallelAssemble();
           auto Mtmp2 = ParMult(P_W->Transpose(), Mtmp);
           HypreParMatrix * M = ParMult(Mtmp2,P_W);
           lobpcg2->SetMassMatrix(*M);
           */

           // 4. Compute the eigenmodes and extract the array of eigenvalues. Define a
           //    parallel grid function to represent each of the eigenmodes returned by
           //    the solver.
           lobpcg2->Solve();
           lobpcg2->GetEigenvalues(eigenvalues);

           std::cout << "The computed maximal eigenvalues for BAinvBT are: \n";
           for ( int i = 0; i < nev; ++i)
               eigenvalues[i] = beta - eigenvalues[i];
           eigenvalues.Print();
       }



       MPI_Finalize();
       return 0;
    }
#endif

   // 12. Solve the linear system with MINRES.
   //     Check the norm of the unpreconditioned residual.

   BlockDiagonalPreconditioner prec(block_finalOffsets);
   if (with_prec > 0 && !direct_solver)
   {
       // Construct the operators for preconditioner
       if (verbose)
           cout << "Using a block diagonal preconditioner \n";
       chrono.Clear();
       chrono.Start();

       Solver * invA;
       invA = new HypreBoomerAMG(*A);
       ((HypreBoomerAMG*)invA)->SetPrintLevel(0);
       ((HypreBoomerAMG*)invA)->iterative_mode = false;

       Operator * invLam;
       Operator * Identity_op;
       HypreParMatrix * Schur;

       if (!identity_Schur)
       {
           HypreParVector *Ad = new HypreParVector(MPI_COMM_WORLD, A->GetGlobalNumRows(), A->GetRowStarts());
           A->GetDiag(*Ad);
           HypreParMatrix * Temp = B->Transpose();
           Temp->InvScaleRows(*Ad);
           Schur = ParMult(B, Temp);
           if (regularization)
               *Schur += *W11;

           invLam = new HypreBoomerAMG(*Schur);
           ((HypreBoomerAMG *)invLam)->SetPrintLevel(0);
           ((HypreBoomerAMG *)invLam)->iterative_mode = false;
       }
       else
       {
           ParBilinearForm *M_block(new ParBilinearForm(W_space));
           M_block->AddDomainIntegrator(new MassIntegrator);
           M_block->Assemble();
           M_block->Finalize();
           auto Mtmp = M_block->ParallelAssemble();
           auto Mtmp2 = ParMult(P_W->Transpose(), Mtmp);
           HypreParMatrix * M = ParMult(Mtmp2,P_W);

           SparseMatrix Mdiag;
           M->GetDiag(Mdiag);
           Vector diagMdiag;
           Mdiag.GetDiag(diagMdiag);

           //diagMdiag.Print();

           double local_appr_Mscale = diagMdiag.Max();
           double global_appr_Mscale;

           // so, we set identity_scale to be coarse_h^2 * || M ||_inf
           //double h_min, h_max, kappa_min, kappa_max;
           //pmesh->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);
           //identity_scale = h_min * h_min;
           MPI_Reduce(&local_appr_Mscale, &global_appr_Mscale, 1,
                      MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
           MPI_Bcast( &global_appr_Mscale, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

           if (verbose)
               std::cout << "global_appr_Mscale = " << global_appr_Mscale << "\n";

           //identity_scale = (1.0/ (h_min * h_min * global_appr_Mscale));
           //identity_scale = (1.0/ global_appr_Mscale);

           //identity_scale = 1.0;
           identity_scale = 1.0 / (0.05 * global_appr_Mscale);
           //identity_scale = 1.0 / 0.000005; // for pref = 1
           //identity_scale = 1.0 / 0.00001; // for pref = 2

           if (identity_Schur && verbose)
               std::cout << "identity_scale = " << identity_scale << "\n";

           Identity_op = new IdentityOperator(B->Height());
           invLam = new MyScaledOperator(*Identity_op, identity_scale);
           if (regularization)
           {
               std::cout << "Identity operator is not coupled with regularization case \n";
               MPI_Finalize();
               return 0;
           }
       }

       // only for debugging
       //invLam = new MyOperator(*B, *invA, *BT);

       prec.SetDiagonalBlock(0, invA);
       prec.SetDiagonalBlock(1, invLam);

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

   //GMRESSolver solver(MPI_COMM_WORLD);    // too slow
   MINRESSolver solver(MPI_COMM_WORLD);
#ifdef MFEM_USE_SUITESPARSE
   Solver * DirectSolver;
   BlockMatrix * SystemBlockMat;
   SparseMatrix * SystemMat;
   SparseMatrix *A_diag, *B_diag, *BT_diag;
#endif
   if (direct_solver)
   {
#ifdef MFEM_USE_SUITESPARSE
       if (verbose)
           std::cout << "Using serial UMFPack direct solver \n";
       SystemBlockMat = new BlockMatrix(block_finalOffsets);
       A_diag = new SparseMatrix();
       B_diag = new SparseMatrix();
       BT_diag = new SparseMatrix();
       A->GetDiag(*A_diag);
       B->GetDiag(*B_diag);
       BT->GetDiag(*BT_diag);

       std::cout << "infinitness measure, A = " << A_diag->CheckFinite() << "\n";
       std::cout << "infinitness measure, B = " << B_diag->CheckFinite() << "\n";
       std::cout << "infinitness measure, BT = " << BT_diag->CheckFinite() << "\n";
       std::cout << "norm, A = " << A_diag->MaxNorm() << "\n";
       std::cout << "norm, B = " << B_diag->MaxNorm() << "\n";

       SystemBlockMat->SetBlock(0,0,A_diag);
       SystemBlockMat->SetBlock(0,1,BT_diag);
       SystemBlockMat->SetBlock(1,0,B_diag);
       SystemMat = SystemBlockMat->CreateMonolithic();

       DirectSolver = new UMFPackSolver(*SystemMat);
       DirectSolver->iterative_mode = false;
       ((UMFPackSolver*)DirectSolver)->SetPrintLevel(10);

       std::cout << "unsymmetry measure = " << SystemMat->IsSymmetric() << "\n";
       std::cout << "infinitness measure = " << SystemMat->CheckFinite() << "\n";
       //SystemMat->Print();
       solver.SetOperator(*SystemMat);
//       solver.SetPreconditioner(*DirectSolver);
#else
       if (verbose)
            std::cout << "Error: no suitesparse and no superlu, direct solver cannot be used \n";
       MPI_Finalize();
       return 0;
#endif
   } // end of case when direct solver is used
   else // iterative solver
   {
       solver.SetAbsTol(atol);
       solver.SetRelTol(rtol);
       solver.SetMaxIter(max_iter);
       solver.SetOperator(*CFOSLSop); // overwritten in case of UMFPackSolver
       if (with_prec > 0 && !direct_solver )
            solver.SetPreconditioner(prec);
   }
   trueX = 0.0;
   BlockVector finalRhs(block_finalOffsets);
   finalRhs = 0.0;
   finalRhs.GetBlock(0) = trueRhs.GetBlock(0);
   P_W->MultTranspose(trueRhs.GetBlock(1), finalRhs.GetBlock(1));
   if (direct_solver)
       DirectSolver->Mult(finalRhs, trueX);
   else
   {
       solver.SetPrintLevel(0);
       solver.Mult(finalRhs, trueX);
   }

   chrono.Stop();


   if (verbose && !direct_solver) // iterative solver reports about its convergence
   {
      if (solver.GetConverged())
         std::cout << "MINRES converged in " << solver.GetNumIterations()
                   << " iterations with a residual norm of " << solver.GetFinalNorm() << ".\n";
      else
         std::cout << "MINRES did not converge in " << solver.GetNumIterations()
                   << " iterations. Residual norm is " << solver.GetFinalNorm() << ".\n";
      std::cout << "MINRES solver took " << chrono.RealTime() << "s. \n";
   }

   ParGridFunction * S = new ParGridFunction(H_space);
   S->Distribute(&(trueX.GetBlock(0)));

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

   double err_S = S->ComputeL2Error((*Mytest.scalarS), irs);
   double norm_S = ComputeGlobalLpNorm(2, (*Mytest.scalarS), *pmesh, irs);
   if (verbose)
   {
       std::cout << "|| S_h - S_ex || / || S_ex || = " <<
                    err_S / norm_S << "\n";
   }

   {
       FiniteElementCollection * hcurl_coll;
       if(dim==4)
           hcurl_coll = new ND1_4DFECollection;
       else
           hcurl_coll = new ND_FECollection(feorder+1, dim);
       auto *N_space = new ParFiniteElementSpace(pmesh.get(), hcurl_coll);

       DiscreteLinearOperator Grad(H_space, N_space);
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

   if (verbose)
       cout << "Computing projection errors" << endl;

   double projection_error_S = S_exact->ComputeL2Error(*(Mytest.scalarS), irs);

   if(verbose)
       cout << "|| S_ex - Pi_h S_ex || / || S_ex || = "
                       << projection_error_S / norm_S << endl;


   if (visualization && nDimensions < 4)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
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

      double S_local_max_norm = S_exact->Normlinf();
      double S_global_max_norm;
      MPI_Reduce(&S_local_max_norm, &S_global_max_norm, 1,
                 MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      if (verbose)
          std::cout << "|| S ||_inf = " << S_global_max_norm << "\n";

      *S_exact -= *S;
      *S_exact /= S_global_max_norm;
      socketstream sss_sock(vishost, visport);
      sss_sock << "parallel " << num_procs << " " << myid << "\n";
      sss_sock.precision(8);
      MPI_Barrier(pmesh->GetComm());
      sss_sock << "solution\n" << *pmesh << *S_exact
               << "window_title 'difference for S scaled by ||S||_inf'" << endl;

      MPI_Barrier(pmesh->GetComm());
   }

   // 17. Free the used memory.
   delete H_space;
   delete W_space;
   delete l2_coll;
   delete h1_coll;

   MPI_Finalize();
   return 0;
}

template <void (*bvecfunc)(const Vector&, Vector& )> \
void KtildaTemplate(const Vector& xt, DenseMatrix& Ktilda)
{
    int nDimensions = xt.Size();
    Vector b;
    bvecfunc(xt,b);
    double bTbInv = (-1./(b*b));
    Ktilda.Diag(1.0,nDimensions);
    AddMult_a_VVt(bTbInv,b,Ktilda);
}

template <void (*bvecfunc)(const Vector&, Vector& )> \
void bbTTemplate(const Vector& xt, DenseMatrix& bbT)
{
//    int nDimensions = xt.Size();
    Vector b;
    bvecfunc(xt,b);
    MultVVt(b, bbT);
}

#ifdef BBT_instead_H1norm
template <void (*bvecfunc)(const Vector&, Vector& )> \
void bbTepsTemplate(const Vector& xt, DenseMatrix& bbT)
{
//    int nDimensions = xt.Size();
    Vector b;
    bvecfunc(xt,b);
    MultVVt(b, bbT);

    DenseMatrix Epsterm(xt.Size());
    Epsterm.Diag(1.0, xt.Size());
    Epsterm *= EPSILON_SHIFT;
    Epsterm(xt.Size()-1, xt.Size()-1) = 0;

    //std::cout << "bbT before adding Epsterm \n";
    //bbT.Print();
    //std::cout << "Epsterm \n";
    //Epsterm.Print();

    bbT += Epsterm;
}
#endif


template <double (*ufunc)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )> \
void sigmaTemplate(const Vector& xt, Vector& sigma)
{
    Vector b;
    bvecfunc(xt, b);
    sigma.SetSize(xt.Size());
    sigma(xt.Size()-1) = ufunc(xt);
    for (int i = 0; i < xt.Size()-1; i++)
        sigma(i) = b(i) * sigma(xt.Size()-1);

    return;
}

template <void (*bvecfunc)(const Vector&, Vector& )> \
double bTbTemplate(const Vector& xt)
{
    Vector b;
    bvecfunc(xt,b);
    return b*b;
}


template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
         void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
double divsigmaTemplate(const Vector& xt)
{
    Vector b;
    bvec(xt,b);

    Vector gradS;
    Sgradxvec(xt,gradS);

    double res = 0.0;

    res += dSdt(xt);
    for ( int i= 0; i < xt.Size() - 1; ++i )
        res += b(i) * gradS(i);
    res += divbfunc(xt) * S(xt);

    return res;
}

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
         void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
void bfTemplate(const Vector& xt, Vector& bf)
{
    Vector b;
    bvec(xt,b);

    Vector gradS;
    Sgradxvec(xt,gradS);

    Vector xt0(xt.Size());
    xt0 = xt;
    xt0 (xt0.Size() - 1) = 0;

    Vector gradS0;
    Sgradxvec(xt0,gradS0);

    double res = 0.0;

    res += dSdt(xt);
    for ( int i= 0; i < xt.Size() - 1; ++i )
        res += b(i) * (gradS(i) - gradS0(i));
    res += divbfunc(xt) * (S(xt) - S(xt0));

    bf.SetSize(xt.Size());

    for (int i = 0; i < bf.Size(); ++i)
        bf(i) = res * b(i);
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


double GaussianHill(const Vector&xvec)
{
    double x = xvec(0);
    double y = xvec(1);
    return exp(-100.0 * ((x - 0.5) * (x - 0.5) + y * y));
}

double uFunCylinder_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double r = sqrt(x*x + y*y);
    double teta = atan(y/x);
    /*
    if (fabs(x) < MYZEROTOL && y > 0)
        teta = M_PI / 2.0;
    else if (fabs(x) < MYZEROTOL && y < 0)
        teta = - M_PI / 2.0;
    else
        teta = atan(y,x);
    */
    double t = xt(xt.Size()-1);
    Vector xvec(2);
    xvec(0) = r * cos (teta - t);
    xvec(1) = r * sin (teta - t);
    return GaussianHill(xvec);
}

double uFunCylinder_ex_dt(const Vector& xt)
{
    return 0.0;
}

void uFunCylinder_ex_gradx(const Vector& xt, Vector& gradx )
{
//    double x = xt(0);
//    double y = xt(1);
//    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = 0.0;
    gradx(1) = 0.0;
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

