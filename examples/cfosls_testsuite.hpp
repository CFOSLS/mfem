// TODO: split this into hpp and cpp, but the first attempt failed
//#include "../mfem.hpp"
//extern class mfem::Vector;
using namespace mfem;
//#include "../linalg/vector.hpp"

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


void bFunRect2D_ex(const Vector& xt, Vector& b );
double  bFunRect2Ddiv_ex(const Vector& xt);

void bFunCube3D_ex(const Vector& xt, Vector& b );
double  bFunCube3Ddiv_ex(const Vector& xt);

void bFunSphere3D_ex(const Vector& xt, Vector& b );
double  bFunSphere3Ddiv_ex(const Vector& xt);

void bFunCircle2D_ex (const Vector& xt, Vector& b);
double  bFunCircle2Ddiv_ex(const Vector& xt);

double uFunTest_ex(const Vector& xt)
{
    double x = xt(0);
    double y;
    if (xt.Size() >= 3)
        y = xt(1);
    double z;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size()-1);

    double res = t*t*exp(t) * sin (3.0 * M_PI * x);
    if (xt.Size() >= 3)
        res *= sin (2.0 * M_PI * y);
    if (xt.Size() == 4)
        res *= sin (M_PI * z);

    return res;
}

double uFunTest_ex_dt(const Vector& xt)
{
    double x = xt(0);
    double y;
    if (xt.Size() >= 3)
        y = xt(1);
    double z;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size()-1);

    double res = (t*t + 2.0 * t)*exp(t) * sin (3.0 * M_PI * x);
    if (xt.Size() >= 3)
        res *= sin (2.0 * M_PI * y);
    if (xt.Size() == 4)
        res *= sin (M_PI * z);

    return res;
}

double uFunTest_ex_dt2(const Vector& xt)
{
    double x = xt(0);
    double y;
    if (xt.Size() >= 3)
        y = xt(1);
    double z;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size()-1);

    double res = ((t*t + 2.0 * t) + (2.0 + 2.0 * t))*exp(t) * sin (3.0 * M_PI * x);
    if (xt.Size() >= 3)
        res *= sin (2.0 * M_PI * y);
    if (xt.Size() == 4)
        res *= sin (M_PI * z);

    return res;
}

double uFunTest_ex_laplace(const Vector& xt)
{
    double x = xt(0);
    double y;
    if (xt.Size() >= 3)
        y = xt(1);
    double z;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size()-1);

    double res = sin (3.0 * M_PI * x) * M_PI * M_PI;
    if (xt.Size() == 2)
        res *= (3.0 * 3.0);
    else
    {
        if (xt.Size() == 3)
        {
            res *= sin (2.0 * M_PI * y);
            res *= (2.0 * 2.0 + 3.0 * 3.0);
        }
        else // 4D
        {
            res *= sin (2.0 * M_PI * y) * sin (M_PI * z);
            res *= (2.0 * 2.0 + 3.0 * 3.0 + 1.0  * 1.0);
        }
    }
    res *= (-1) * t*t*exp(t);

    return res;
}

double uFunTest_ex_dtlaplace(const Vector& xt)
{
    double x = xt(0);
    double y;
    if (xt.Size() >= 3)
        y = xt(1);
    double z;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size()-1);

    double res = sin (3.0 * M_PI * x) * M_PI * M_PI;
    if (xt.Size() == 2)
        res *= (3.0 * 3.0);
    else
    {
        if (xt.Size() == 3)
        {
            res *= sin (2.0 * M_PI * y);
            res *= (2.0 * 2.0 + 3.0 * 3.0);
        }
        else // 4D
        {
            res *= sin (2.0 * M_PI * y) * sin (M_PI * z);
            res *= (2.0 * 2.0 + 3.0 * 3.0 + 1.0  * 1.0);
        }
    }
    res *= (-1) * (t*t + 2.0 * t)*exp(t);

    return res;
}

void uFunTest_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y;
    if (xt.Size() >= 3)
        y = xt(1);
    double z;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = t*t*exp(t) * 3.0 * M_PI * cos (3.0 * M_PI * x);
    if (xt.Size() >= 3)
    {
        gradx(0) *= sin (2.0 * M_PI * y);
        gradx(1) = t*t*exp(t) * sin (3.0 * M_PI * x) * 2.0 * M_PI * cos ( 2.0 * M_PI * y);
    }
    if (xt.Size() == 4)
    {
        gradx(0) *= sin (M_PI * z);
        gradx(1) *= sin (M_PI * z);
        gradx(2) = t*t*exp(t) * sin (3.0 * M_PI * x) * sin ( 2.0 * M_PI * y) * M_PI * cos (M_PI * z);
    }
}

void uFunTest_ex_gradxt(const Vector& xt, Vector& gradxt)
{
    double x = xt(0);
    double y;
    if (xt.Size() >= 3)
        y = xt(1);
    double z;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size()-1);

    gradxt.SetSize(xt.Size());

    gradxt(0) = t*t*exp(t) * 3.0 * M_PI * cos (3.0 * M_PI * x);
    if (xt.Size() >= 3)
    {
        gradxt(0) *= sin (2.0 * M_PI * y);
        gradxt(1) = t*t*exp(t) * sin (3.0 * M_PI * x) * 2.0 * M_PI * cos ( 2.0 * M_PI * y);
    }
    if (xt.Size() == 4)
    {
        gradxt(0) *= sin (M_PI * z);
        gradxt(1) *= sin (M_PI * z);
        gradxt(2) = t*t*exp(t) * sin (3.0 * M_PI * x) * sin ( 2.0 * M_PI * y) * M_PI * cos (M_PI * z);
    }

    gradxt(xt.Size()-1) = (t*t + 2.0 * t)*exp(t) * sin (3.0 * M_PI * x);
    if (xt.Size() >= 3)
        gradxt(xt.Size()-1) *= sin (2.0 * M_PI * y);
    if (xt.Size() == 4)
        gradxt(xt.Size()-1) *= sin (M_PI * z);
}

void uFunTest_ex_dtgradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y;
    if (xt.Size() >= 3)
        y = xt(1);
    double z;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = (t*t + 2.0 * t)*exp(t) * 3.0 * M_PI * cos (3.0 * M_PI * x);
    if (xt.Size() >= 3)
    {
        gradx(0) *= sin (2.0 * M_PI * y);
        gradx(1) = (t*t + 2.0 * t)*exp(t) * sin (3.0 * M_PI * x) * 2.0 * M_PI * cos ( 2.0 * M_PI * y);
    }
    if (xt.Size() == 4)
    {
        gradx(0) *= sin (M_PI * z);
        gradx(1) *= sin (M_PI * z);
        gradx(2) = (t*t + 2.0 * t)*exp(t) * sin (3.0 * M_PI * x) * sin ( 2.0 * M_PI * y) * M_PI * cos (M_PI * z);
    }

}

// velocity for hyperbolic problems
void bFunRect2D_ex(const Vector& xt, Vector& b )
{
    double x = xt(0);
    double y = xt(1);

    b.SetSize(xt.Size());

    b(0) = sin(x*M_PI)*cos(y*M_PI);
    b(1) = - sin(y*M_PI)*cos(x*M_PI);

    b(xt.Size()-1) = 1.;

}

double bFunRect2Ddiv_ex(const Vector& xt)
{
    return 0.0;
}

void bFunCube3D_ex(const Vector& xt, Vector& b )
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);

    b.SetSize(xt.Size());

    b(0) = sin(x*M_PI)*cos(y*M_PI)*cos(z*M_PI);
    b(1) = - 0.5 * sin(y*M_PI)*cos(x*M_PI) * cos(z*M_PI);
    b(2) = - 0.5 * sin(z*M_PI)*cos(x*M_PI) * cos(y*M_PI);

    b(xt.Size()-1) = 1.;

}

double bFunCube3Ddiv_ex(const Vector& xt)
{
    return 0.0;
}

void bFunSphere3D_ex(const Vector& xt, Vector& b )
{
    double x = xt(0);
    double y = xt(1);
//    double t = xt(xt.Size()-1);

    b.SetSize(xt.Size());

    b(0) = -y;  // -x2
    b(1) = x;   // x1
    b(2) = 0.0;

    b(xt.Size()-1) = 1.;
    return;
}

double bFunSphere3Ddiv_ex(const Vector& xt)
{
    return 0.0;
}
void bFunCircle2D_ex(const Vector& xt, Vector& b )
{
    double x = xt(0);
    double y = xt(1);
//    double t = xt(xt.Size()-1);

    b.SetSize(xt.Size());

    b(0) = -y;  // -x2
    b(1) = x;   // x1

    b(xt.Size()-1) = 1.;
    return;
}

double bFunCircle2Ddiv_ex(const Vector& xt)
{
//    double x = xt(0);
//    double y = xt(1);
//    double t = xt(xt.Size()-1);
    return 0.0;
}

double uFunTestLap_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size()-1);

    //return sin (M_PI * x) * sin (M_PI * y) * sin (M_PI * t);

    double tpart = 16.0 * t * t * (1 - t) * (1 - t) * exp(t);
    double res = tpart * sin (3.0 * M_PI * x) * sin (2.0 * M_PI * y);
    if (xt.Size() == 4)
        res *= sin (M_PI * z);

    return res;
}

double uFunTestLap_lap(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size()-1);

    double tpart = 16.0 * t * t * (1 - t) * (1 - t) * exp(t);

    // d2/dx2 + d2/dy2
    double res1 = - tpart * (3.0 * M_PI * 3.0 * M_PI + 2.0 * M_PI * 2.0 * M_PI) * sin (3.0 * M_PI * x) * sin (2.0 * M_PI * y);
    if (xt.Size() == 4)
        res1 *= sin (M_PI * z);

    // d2/dt2
    double d2tpart = 16.0 * exp(t) * (t * t * (1 - t) * (1 - t) + 2.0 * ( 2.0 * t * (t - 1) * (2.0 * t - 1) ) + (12.0 * t * t - 12.0 * t + 2) );
    double res2 = d2tpart * sin (3.0 * M_PI * x) * sin (2.0 * M_PI * y);
    if (xt.Size() == 4)
        res2 *= sin (M_PI * z);

    //d2/dz2
    double res3 = 0.0;
    if (xt.Size() == 4)
        res3 = (-1) * M_PI * M_PI * tpart * sin (3.0 * M_PI * x) * sin (2.0 * M_PI * y) * sin (M_PI * z);

    return res1 + res2 + res3;

    //return (-1) * 3.0 * M_PI * M_PI * uFunTestLap_ex(xt);
}

void uFunTestLap_grad(const Vector& xt, Vector& grad )
{
    double x = xt(0);
    double y = xt(1);
    double z;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size()-1);

    double tpart = 16.0 * t * t * (1 - t) * (1 - t) * exp(t);
    double dttpart = 16.0 * exp(t) * (t * t * (1 - t) * (1 - t) + 2.0 * t * (t - 1) * (2.0 * t - 1) );

    grad.SetSize(xt.Size());

    grad(0) = tpart * 3.0 * M_PI * cos (3.0 * M_PI * x) * sin (2.0 * M_PI * y);
    grad(1) = tpart * 2.0 * M_PI * cos (2.0 * M_PI * y) * sin (3.0 * M_PI * x);
    grad(xt.Size() - 1) =  dttpart * sin (3.0 * M_PI * x) * sin (2.0 * M_PI * y);

    if (xt.Size() == 4)
    {
        grad(2) = tpart * sin (3.0 * M_PI * x) * sin (2.0 * M_PI * y) * M_PI * cos (M_PI * z);
        grad(0) *= sin (M_PI * z);
        grad(1) *= sin (M_PI * z);
        grad(xt.Size() - 1) *= sin (M_PI * z);
    }

    //grad(0) = M_PI * cos (M_PI * x) * sin (M_PI * y) * sin (M_PI * t);
    //grad(1) = M_PI * sin (M_PI * x) * cos (M_PI * y) * sin (M_PI * t);
    //grad(2) = M_PI * sin (M_PI * x) * sin (M_PI * y) * cos (M_PI * t);

    return;
}

/// Integrator for (Q u, v)
/// where Q is a vector coefficient, u is from vector FE space created
/// from scalar FE collection and v is from scalar FE space
class MixedVectorScalarIntegrator : public BilinearFormIntegrator
{
private:
   VectorCoefficient *VQ;
   void Init(VectorCoefficient *vq)
   { VQ = vq; }

#ifndef MFEM_THREAD_SAFE
   Vector shape;
   Vector D;
   Vector test_shape;
   Vector b;
   Vector trial_shape;
   DenseMatrix trial_vshape; // components are test shapes
#endif

public:
   MixedVectorScalarIntegrator() { Init(NULL); }
   MixedVectorScalarIntegrator(VectorCoefficient *_vq) { Init(_vq); }
   MixedVectorScalarIntegrator(VectorCoefficient &vq) { Init(&vq); }

   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

void MixedVectorScalarIntegrator::AssembleElementMatrix2(
        const FiniteElement &trial_fe, const FiniteElement &test_fe,
        ElementTransformation &Trans, DenseMatrix &elmat)
{
    // assume trial_fe is vector FE but created from scalar f.e. collection,
    // and test_fe is scalar FE

    MFEM_ASSERT(test_fe.GetRangeType() == FiniteElement::SCALAR && trial_fe.GetRangeType() == FiniteElement::SCALAR,
                "The improper vector FE should have a scalar type in the current implementation \n");

    int dim  = test_fe.GetDim();
    //int vdim = dim;
    int trial_dof = trial_fe.GetDof();
    int test_dof = test_fe.GetDof();
    double w;

    if (VQ == NULL)
        mfem_error("MixedVectorScalarIntegrator::AssembleElementMatrix2(...)\n"
                "   is not implemented for non-vector coefficients");

#ifdef MFEM_THREAD_SAFE
    Vector trial_shape(trial_dof);
    DenseMatrix test_vshape(test_dof,dim);
#else
    trial_vshape.SetSize(trial_dof*dim,dim);
    test_shape.SetSize(test_dof);
#endif
    elmat.SetSize (test_dof, trial_dof * dim);

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
        test_fe.CalcShape(ip, test_shape);

        Trans.SetIntPoint (&ip);
        trial_fe.CalcShape(ip, trial_shape);

        //std::cout << "trial_shape \n";
        //trial_shape.Print();

        for (int d = 0; d < dim; ++d )
            for (int l = 0; l < dim; ++l)
                for (int k = 0; k < trial_dof; ++k)
                {
                    if (l == d)
                        trial_vshape(l*trial_dof+k,d) = trial_shape(k);
                    else
                        trial_vshape(l*trial_dof+k,d) = 0.0;
                }
        // now trial_vshape is of size trial_dof(scalar)*dim x dim

        //trial_fe.CalcVShape(Trans, trial_vshape); would be nice if it worked but no

        w = ip.weight * Trans.Weight();
        VQ->Eval (b, Trans, ip);

        for (int l = 0; l < dim; ++l)
            for (int j = 0; j < trial_dof; j++)
                for (int k = 0; k < test_dof; k++)
                    for (int d = 0; d < dim; d++ )
                    {
                        elmat(k, l*trial_dof + j) += w*trial_vshape(l*trial_dof + j,d)*b(d)*test_shape(k);
                    }
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

      Tr.SetIntPoint (&ip);
      w = ip.weight;// * Tr.Weight();
      CalcAdjugate(Tr.Jacobian(), invdfdx);
      Mult(dshape, invdfdx, dshapedxt);

      Q.Eval(bf, Tr, ip);

      dshapedxt.Mult(bf, bfdshapedxt);

      add(elvect, w, bfdshapedxt, elvect);
   }
}

/// Integrator for (q * u, v)
/// where q is a scalar coefficient, u and v are from vector FE space
/// created from scalar FE collection (called improper vector FE)
class ImproperVectorMassIntegrator : public BilinearFormIntegrator
{
private:
   Coefficient *Q;
   void Init(Coefficient *q)
   { Q = q; }

#ifndef MFEM_THREAD_SAFE
   Vector scalar_shape;
   DenseMatrix vector_vshape; // components are test shapes
#endif

public:
   ImproperVectorMassIntegrator() { Init(NULL); }
   ImproperVectorMassIntegrator(Coefficient *_q) { Init(_q); }
   ImproperVectorMassIntegrator(Coefficient &q) { Init(&q); }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

void ImproperVectorMassIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
    MFEM_ASSERT(el.GetRangeType() == FiniteElement::SCALAR,
                "The improper vector FE should have a scalar type in the current implementation \n");

    int dim  = el.GetDim();
    int nd = el.GetDof();
    int improper_nd = nd * dim;

    double w;

#ifdef MFEM_THREAD_SAFE
    Vector scalar_shape.SetSize(nd);
    DenseMatrix vector_vshape(improper_nd, dim);
#else
    scalar_shape.SetSize(nd);
    vector_vshape.SetSize(improper_nd, dim);
#endif
    elmat.SetSize (improper_nd, improper_nd);

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

       Trans.SetIntPoint (&ip);

       el.CalcShape(ip, scalar_shape);
       for (int d = 0; d < dim; ++d )
           for (int l = 0; l < dim; ++l)
               for (int k = 0; k < nd; ++k)
               {
                   if (l == d)
                       vector_vshape(l*nd+k,d) = scalar_shape(k);
                   else
                       vector_vshape(l*nd+k,d) = 0.0;
               }
       // now vector_vshape is of size improper_nd x dim
       //el.CalcVShape(Trans, vector_vshape); // would be easy but this doesn't work for improper vector L2

       w = ip.weight * Trans.Weight();
       if (Q)
          w *= Q->Eval(Trans, ip);

       AddMult_a_AAt (w, vector_vshape, elmat);

    }
}


void testVectorFun(const Vector& xt, Vector& res);

std::vector<std::pair<int,int> >* CreateBotToTopDofsLink(const char * eltype, FiniteElementSpace& fespace,
                                                         std::vector<std::pair<int,int> > & bot_to_top_bels, bool verbose = false);

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

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx),  \
         void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
void Transport_test::SetTestCoeffs ()
{
    SetScalarSFun(S);
    SetbVec(bvec);
    SetminbVec<bvec>();
    SetbfVec<S, dSdt, Sgradxvec, bvec, divbfunc>();
    SetSigmaVec<S,bvec>();
    SetKtildaMat<bvec>();
    SetScalarBtB<bvec>();
    SetDivSigma<S, dSdt, Sgradxvec, bvec, divbfunc>();
    SetBBtMat<bvec>();
    return;
}


bool Transport_test::CheckTestConfig()
{
    if (dim == 4 || dim == 3)
    {
        if (numsol == -3 && dim == 3)
            return true;
        if (numsol == -4 && dim == 4)
            return true;
        if (numsol == -33 && dim == 3)
            return true;
        if (numsol == -44 && dim == 4)
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
        if (numsol == -33) // 3D test for the paper
        {
            SetTestCoeffs<&uFunTestNh_ex, &uFunTestNh_ex_dt, &uFunTestNh_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex>();
        }
        if (numsol == -4) // 4D test for the paper
        {
            SetTestCoeffs<&uFunTest_ex, &uFunTest_ex_dt, &uFunTest_ex_gradx, &bFunCube3D_ex, &bFunCube3Ddiv_ex>();
        }
        if (numsol == -44) // 4D test for the paper
        {
            SetTestCoeffs<&uFunTestNh_ex, &uFunTestNh_ex_dt, &uFunTestNh_ex_gradx, &bFunCube3D_ex, &bFunCube3Ddiv_ex>();
        }
    } // end of setting test coefficients in correct case
}

template <void (*bvecfunc)(const Vector&, Vector& )> \
void KtildaTemplate(const Vector& xt, DenseMatrix& Ktilda)
{
    int nDimensions = xt.Size();
    Ktilda.SetSize(nDimensions);
    Vector b;
    bvecfunc(xt,b);
    double bTbInv = (-1./(b*b));
    Ktilda.Diag(1.0,nDimensions);
#ifndef K_IDENTITY
    AddMult_a_VVt(bTbInv,b,Ktilda);
#endif
}

template <void (*bvecfunc)(const Vector&, Vector& )> \
void bbTTemplate(const Vector& xt, DenseMatrix& bbT)
{
    Vector b;
    bvecfunc(xt,b);
    MultVVt(b, bbT);
}


template <double (*S)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )> \
void sigmaTemplate(const Vector& xt, Vector& sigma)
{
    Vector b;
    bvecfunc(xt, b);
    sigma.SetSize(xt.Size());
    sigma(xt.Size()-1) = S(xt);
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

template <double (*S)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )> \
double minbTbSnonhomoTemplate(const Vector& xt)
{
    Vector xt0(xt.Size());
    xt0 = xt;
    xt0 (xt0.Size() - 1) = 0;

    return - bTbTemplate<bvecfunc>(xt) * S(xt0);
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
    bf.SetSize(xt.Size());

    Vector b;
    bvec(xt,b);

    double f = divsigmaTemplate<S, dSdt, Sgradxvec, bvec, divbfunc>(xt);

    for (int i = 0; i < bf.Size(); ++i)
        bf(i) = f * b(i);
}

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
         void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
void bdivsigmaTemplate(const Vector& xt, Vector& bdivsigma)
{
    bdivsigma.SetSize(xt.Size());

    Vector b;
    bvec(xt,b);

    double divsigma = divsigmaTemplate<S, dSdt, Sgradxvec, bvec, divbfunc>(xt);

    for (int i = 0; i < bdivsigma.Size(); ++i)
        bdivsigma(i) = divsigma * b(i);
}

template<void(*bvec)(const Vector & x, Vector & vec)>
void minbTemplate(const Vector& xt, Vector& minb)
{
    minb.SetSize(xt.Size());

    bvec(xt,minb);

    minb *= -1;
}

template<double (*S)(const Vector & xt) > double SnonhomoTemplate(const Vector& xt)
{
    Vector xt0(xt.Size());
    xt0 = xt;
    xt0 (xt0.Size() - 1) = 0;

    return S(xt0);
}


double uFunTestNh_ex(const Vector& xt)
{
    double x = xt(0);
    double y;
    if (xt.Size() >= 3)
        y = xt(1);
    double z;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size()-1);

    double res = exp(t) * sin (3.0 * M_PI * x);
    if (xt.Size() >= 3)
        res *= sin (2.0 * M_PI * y);
    if (xt.Size() == 4)
        res *= sin (M_PI * z);

    return res;
}

double uFunTestNh_ex_dt(const Vector& xt)
{
    return uFunTestNh_ex(xt);
}

double uFunTestNh_ex_dt2(const Vector& xt)
{
    return uFunTestNh_ex(xt);
}

double uFunTestNh_ex_laplace(const Vector& xt)
{
    double x = xt(0);
    double y;
    if (xt.Size() >= 3)
        y = xt(1);
    double z;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size()-1);

    double res = sin (3.0 * M_PI * x) * M_PI * M_PI;
    if (xt.Size() == 2)
        res *= (3.0 * 3.0);
    else
    {
        if (xt.Size() == 3)
        {
            res *= sin (2.0 * M_PI * y);
            res *= (2.0 * 2.0 + 3.0 * 3.0);
        }
        else // 4D
        {
            res *= sin (2.0 * M_PI * y) * sin (M_PI * z);
            res *= (2.0 * 2.0 + 3.0 * 3.0 + 1.0  * 1.0);
        }
    }
    res *= (-1) * exp(t);

    return res;
}

double uFunTestNh_ex_dtlaplace(const Vector& xt)
{
    return uFunTestNh_ex_laplace(xt);
}

void uFunTestNh_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y;
    if (xt.Size() >= 3)
        y = xt(1);
    double z;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = exp(t) * 3.0 * M_PI * cos (3.0 * M_PI * x);
    if (xt.Size() >= 3)
    {
        gradx(0) *= sin (2.0 * M_PI * y);
        gradx(1) = exp(t) * sin (3.0 * M_PI * x) * 2.0 * M_PI * cos ( 2.0 * M_PI * y);
    }
    if (xt.Size() == 4)
    {
        gradx(0) *= sin (M_PI * z);
        gradx(1) *= sin (M_PI * z);
        gradx(2) = exp(t) * sin (3.0 * M_PI * x) * sin ( 2.0 * M_PI * y) * M_PI * cos (M_PI * z);
    }
}

void uFunTestNh_ex_gradxt(const Vector& xt, Vector& gradxt)
{
    double x = xt(0);
    double y;
    if (xt.Size() >= 3)
        y = xt(1);
    double z;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size()-1);

    gradxt.SetSize(xt.Size());

    gradxt(0) = exp(t) * 3.0 * M_PI * cos (3.0 * M_PI * x);
    if (xt.Size() >= 3)
    {
        gradxt(0) *= sin (2.0 * M_PI * y);
        gradxt(1) = exp(t) * sin (3.0 * M_PI * x) * 2.0 * M_PI * cos ( 2.0 * M_PI * y);
    }
    if (xt.Size() == 4)
    {
        gradxt(0) *= sin (M_PI * z);
        gradxt(1) *= sin (M_PI * z);
        gradxt(2) = exp(t) * sin (3.0 * M_PI * x) * sin ( 2.0 * M_PI * y) * M_PI * cos (M_PI * z);
    }

    gradxt(xt.Size()-1) = exp(t) * sin (3.0 * M_PI * x);
    if (xt.Size() >= 3)
        gradxt(xt.Size()-1) *= sin (2.0 * M_PI * y);
    if (xt.Size() == 4)
        gradxt(xt.Size()-1) *= sin (M_PI * z);
}

void uFunTestNh_ex_dtgradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y;
    if (xt.Size() >= 3)
        y = xt(1);
    double z;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = exp(t) * 3.0 * M_PI * cos (3.0 * M_PI * x);
    if (xt.Size() >= 3)
    {
        gradx(0) *= sin (2.0 * M_PI * y);
        gradx(1) = exp(t) * sin (3.0 * M_PI * x) * 2.0 * M_PI * cos ( 2.0 * M_PI * y);
    }
    if (xt.Size() == 4)
    {
        gradx(0) *= sin (M_PI * z);
        gradx(1) *= sin (M_PI * z);
        gradx(2) = exp(t) * sin (3.0 * M_PI * x) * sin ( 2.0 * M_PI * y) * M_PI * cos (M_PI * z);
    }

}
