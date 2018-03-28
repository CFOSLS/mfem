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

#include "cfosls_testsuite.hpp"

//#define REGULARIZE_A

#define MYZEROTOL (1.0e-13)

using namespace std;
using namespace mfem;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

double FOSLSErrorEstimator(BilinearFormIntegrator &blfi, GridFunction &sigma, Vector &error_estimates)
{
    FiniteElementSpace * fes = sigma.FESpace();
    int ne = fes->GetNE();
    error_estimates.SetSize(ne);

    double total_error = 0.0;
    for (int i = 0; i < ne; ++i)
    {
        const FiniteElement * fe = fes->GetFE(i);
        ElementTransformation * eltrans = fes->GetElementTransformation(i);
        DenseMatrix elmat;
        blfi.AssembleElementMatrix(*fe, *eltrans, elmat);

        Array<int> eldofs;
        fes->GetElementDofs(i, eldofs);
        Vector localv;
        sigma.GetSubVector(eldofs, localv);

        Vector localAv(localv.Size());
        elmat.Mult(localv, localAv);

        //std::cout << "sigma linf norm = " << sigma.Normlinf() << "\n";
        //sigma.Print();
        //eldofs.Print();
        //localv.Print();
        //localAv.Print();

        double err = localAv * localv;


        error_estimates(i) = std::sqrt(err);
        total_error += err;
    }

    std::cout << "error estimates linf norm = " << error_estimates.Normlinf() << "\n";

    return std::sqrt(total_error);
}

double FOSLSErrorEstimator(Array2D<BilinearFormIntegrator*> &blfis, Array<ParGridFunction*> & sols, Vector &error_estimates)
{
    if (sols.Size() == 1)
    {
        return FOSLSErrorEstimator(*blfis(0,0), *sols[0], error_estimates);
    }
    else
    {
        Array<FiniteElementSpace*> fess(sols.Size());
        for (int i = 0; i < sols.Size(); ++i)
            fess = sols[i]->FESpace();

        int ne = fess[0]->GetNE();
        error_estimates.SetSize(ne);

        double total_error = 0.0;
        for (int i = 0; i < ne; ++i)
        {
            double err = 0.0;
            for (int rowblk = 0; rowblk < blfis.NumRows(); ++rowblk)
                for (int colblk = 0; colblk < blfis.NumCols(); ++colblk)
                    if (blfis(rowblk,colblk))
                    {
                        FiniteElementSpace * fes1 = fess[rowblk];
                        FiniteElementSpace * fes2 = fess[colblk];
                        const FiniteElement * fe1 = fes1->GetFE(i);
                        const FiniteElement * fe2 = fes2->GetFE(i);
                        ElementTransformation * eltrans = fes2->GetElementTransformation(i);
                        DenseMatrix elmat;
                        blfis(rowblk,colblk)->AssembleElementMatrix2(*fe1, *fe2, *eltrans, elmat);

                        Vector localv1;
                        Array<int> eldofs1;
                        fes1->GetElementDofs(i, eldofs1);
                        sols[rowblk]->GetSubVector(eldofs1, localv1);

                        Vector localv2;
                        Array<int> eldofs2;
                        fes2->GetElementDofs(i, eldofs2);
                        sols[colblk]->GetSubVector(eldofs2, localv2);

                        Vector localAv2(localv2.Size());
                        elmat.Mult(localv2, localAv2);

                        //std::cout << "sigma linf norm = " << sigma.Normlinf() << "\n";
                        //sigma.Print();
                        //eldofs.Print();
                        //localv.Print();
                        //localAv.Print();

                        err += localAv2 * localv1;
                    }

            error_estimates(i) = std::sqrt(err);
            total_error += err;
        }

        return std::sqrt(total_error);
    }
}

class FOSLSEstimator : public ErrorEstimator
{
protected:
    const int numblocks;
    long current_sequence;
    Array<ParGridFunction*> sols;
    Array2D<BilinearFormIntegrator*> integs;
    ParGridFunction *sol;           // is not owned
    BilinearFormIntegrator *integ;  // is not owned
    Vector error_estimates;
    double total_error;

    /// Check if the mesh of the solution was modified (copied from L2ZienkkiewiczZhuEstimator).
    bool MeshIsModified();

    /// Compute the element error estimates (copied from L2ZienkkiewiczZhuEstimator).
    void ComputeEstimates();
public:
    ~FOSLSEstimator() {}
    FOSLSEstimator(ParGridFunction &solution, BilinearFormIntegrator &integrator);
    FOSLSEstimator(Array<ParGridFunction*>& solutions, Array2D<BilinearFormIntegrator*>& integrators);
    virtual const Vector & GetLocalErrors () override;
    virtual void Reset () override;
};

FOSLSEstimator::FOSLSEstimator(ParGridFunction &solution, BilinearFormIntegrator &integrator)
    : numblocks(1), current_sequence(-1), sol(&solution), integ(&integrator), total_error(0.0)
{
    sols.SetSize(numblocks);
    sols[0] = &solution;
    integs.SetSize(numblocks, numblocks);
    integs(0,0) = &integrator;
}

FOSLSEstimator::FOSLSEstimator(Array<ParGridFunction *> &solutions, Array2D<BilinearFormIntegrator *> &integrators)
    : numblocks(solutions.Size()), current_sequence(-1), total_error(0.0)
{
    sols.SetSize(numblocks);
    for (int i = 0; i < numblocks; ++i)
        sols[i] = solutions[i];
    integs.SetSize(numblocks, numblocks);
    for (int i = 0; i < numblocks; ++i)
        for (int j = 0; j < numblocks; ++j)
            integs(i,j) = integrators(i,j);
}

bool FOSLSEstimator::MeshIsModified()
{
   long mesh_sequence = sol->FESpace()->GetMesh()->GetSequence();
   MFEM_ASSERT(mesh_sequence >= current_sequence, "");
   return (mesh_sequence > current_sequence);
}

const Vector & FOSLSEstimator::GetLocalErrors()
{
    if (MeshIsModified()) { ComputeEstimates(); }
    return error_estimates;
}

void FOSLSEstimator::ComputeEstimates()
{
    total_error = FOSLSErrorEstimator(integs, sols, error_estimates);

    //error_estimates.Print();

    current_sequence = sol->FESpace()->GetMesh()->GetSequence();
}

void FOSLSEstimator::Reset()
{
    current_sequence = -1;
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


/// Integrator for (q * u, v)
/// where q is a scalar coefficient, u is from vector FE space created
/// from scalar FE collection (called improper vector FE) and v is from
/// proper vector FE space (like RT or ND)
class MixedVectorVectorFEMassIntegrator : public BilinearFormIntegrator
{
private:
   Coefficient *Q;
   void Init(Coefficient *q)
   { Q = q; }

#ifndef MFEM_THREAD_SAFE
   DenseMatrix test_vshape;
   Vector scalar_shape;
   DenseMatrix trial_vshape; // components are test shapes
#endif

public:
   MixedVectorVectorFEMassIntegrator() { Init(NULL); }
   MixedVectorVectorFEMassIntegrator(Coefficient *_q) { Init(_q); }
   MixedVectorVectorFEMassIntegrator(Coefficient &q) { Init(&q); }

   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

void MixedVectorVectorFEMassIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
    // here we assume for a moment that proper vector FE is the trial one
    MFEM_ASSERT(test_fe.GetRangeType() == FiniteElement::SCALAR,
                "The improper vector FE should have a scalar type in the current implementation \n");

    int dim  = test_fe.GetDim();
    int trial_dof = trial_fe.GetDof();
    int test_dof = test_fe.GetDof();
    int improper_testdof = dim * test_dof;

    double w;

#ifdef MFEM_THREAD_SAFE
    DenseMatrix trial_vshape(trial_dof, dim);
    DenseMatrix test_vshape(improper_testdof,dim);
    Vector scalar_shape(test_dof);
#else
    trial_vshape.SetSize(trial_dof, dim);
    test_vshape.SetSize(improper_testdof,dim);
    scalar_shape.SetSize(test_dof);
#endif

    elmat.SetSize (improper_testdof, trial_dof);

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

       Trans.SetIntPoint (&ip);

       trial_fe.CalcVShape(Trans, trial_vshape);

       test_fe.CalcShape(ip, scalar_shape);
       for (int d = 0; d < dim; ++d )
           for (int l = 0; l < dim; ++l)
               for (int k = 0; k < test_dof; ++k)
               {
                   if (l == d)
                       test_vshape(l*test_dof+k,d) = scalar_shape(k);
                   else
                       test_vshape(l*test_dof+k,d) = 0.0;
               }
       // now test_vshape is of size trial_dof(scalar)*dim x dim
       //test_fe.CalcVShape(Trans, test_vshape); // would be easy but this doesn't work for improper vector L2

       //std::cout << "trial_vshape \n";
       //trial_vshape.Print();

       //std::cout << "scalar_shape \n";
       //scalar_shape.Print();
       //std::cout << "improper test_vshape \n";
       //test_vshape.Print();

       w = ip.weight * Trans.Weight();
       if (Q)
          w *= Q->Eval(Trans, ip);

       for (int l = 0; l < dim; ++l)
       {
          for (int j = 0; j < test_dof; j++)
          {
             for (int k = 0; k < trial_dof; k++)
             {
                 for (int d = 0; d < dim; d++)
                 {
                    elmat(l*test_dof+j, k) += w * test_vshape(l*test_dof+j, d) * trial_vshape(k, d);
                 }
             }
          }
       }

       //std::cout << "elmat \n";
       //elmat.Print();
       //int p = 2;
       //p++;

    }

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

//********* NEW STUFF FOR 4D CFOSLS
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
   Vector test_shape;
   Vector b;
   DenseMatrix trial_vshape;
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
   PAUVectorFEMassIntegrator2() { Init(NULL, NULL, NULL); }
   PAUVectorFEMassIntegrator2(Coefficient *_q) { Init(_q, NULL, NULL); }
   PAUVectorFEMassIntegrator2(Coefficient &q) { Init(&q, NULL, NULL); }
   PAUVectorFEMassIntegrator2(VectorCoefficient *_vq) { Init(NULL, _vq, NULL); }
   PAUVectorFEMassIntegrator2(VectorCoefficient &vq) { Init(NULL, &vq, NULL); }
   PAUVectorFEMassIntegrator2(MatrixCoefficient *_mq) { Init(NULL, NULL, _mq); }
   PAUVectorFEMassIntegrator2(MatrixCoefficient &mq) { Init(NULL, NULL, &mq); }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
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
    // assume both test_fe is vector FE, trial_fe is scalar FE
    int dim  = test_fe.GetDim();
    int trial_dof = trial_fe.GetDof();
    int test_dof = test_fe.GetDof();
    double w;

    if (VQ == NULL) // || = or
        mfem_error("VectorFEMassIntegrator::AssembleElementMatrix2(...)\n"
                "   is not implemented for non-vector coefficients");

#ifdef MFEM_THREAD_SAFE
    Vector trial_shape(trial_dof);
    DenseMatrix test_vshape(test_dof,dim);
#else
    trial_vshape.SetSize(trial_dof,dim);
    test_shape.SetSize(test_dof);
#endif
    elmat.SetSize (test_dof, trial_dof);

    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        int order = (Trans.OrderW() + test_fe.GetOrder() + trial_fe.GetOrder());
        ir = &IntRules.Get(test_fe.GetGeomType(), order);
    }

    elmat = 0.0;
//    b.SetSize(dim);
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        test_fe.CalcShape(ip, test_shape);

        Trans.SetIntPoint (&ip);
        trial_fe.CalcVShape(Trans, trial_vshape);

        w = ip.weight * Trans.Weight();
        VQ->Eval (b, Trans, ip);

        for (int j = 0; j < trial_dof; j++)
            for (int k = 0; k < test_dof; k++)
                for (int d = 0; d < dim; d++ )
                    elmat(k, j) += w*trial_vshape(j,d)*b(d)*test_shape(k);
    }
}
///////////////////////////

void PAUVectorFEMassIntegrator2::AssembleElementMatrix(
        const FiniteElement &el,
        ElementTransformation &Trans,
        DenseMatrix &elmat)
{
    int dof = el.GetDof();
    int dim  = el.GetDim();
    double w;

    if (VQ || MQ) // || = or
        mfem_error("VectorFEMassIntegrator::AssembleElementMatrix2(...)\n"
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
        {
            for (int k = 0; k < dof; k++)
            {
                for (int d = 0; d < dim - 1; d++ )
                    elmat(j, k) +=  w * dshapedxt(j, d) * dshapedxt(k, d);
                elmat(j, k) +=  w * shape(j) * shape(k);
            }
        }
    }
}

void PAUVectorFEMassIntegrator2::AssembleElementMatrix2(
        const FiniteElement &trial_fe, const FiniteElement &test_fe,
        ElementTransformation &Trans, DenseMatrix &elmat)
{}

//********* END OF NEW STUFF FOR CFOSLS 4D

//********* NEW STUFF FOR 4D CFOSLS
//---------
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
//      double val = Tr.Weight() * Q.Eval(Tr, ip);
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

//------------------
//********* END OF NEW STUFF FOR CFOSLS 4D


//------------------
//********* END OF NEW BilinearForm and LinearForm integrators FOR CFOSLS 4D (used only for heat equation, so can be deleted)

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
    int numsol          = 8;

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
    mesh_file = "../data/pmesh_cylinder_moderate_0.2.mesh";

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
       //gform->AddDomainIntegrator(new DomainLFIntegrator(*Mytest.scalardivsigma));
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
           Cblock->AddDomainIntegrator(new MassIntegrator(*Mytest.bTb));
           if (strcmp(space_for_sigma,"Hdiv") == 0)
                Cblock->AddDomainIntegrator(new DiffusionIntegrator(*Mytest.bbT));
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
   solver.SetPrintLevel(1);
   trueX = 0.0;

   chrono.Clear();
   chrono.Start();
   solver.Mult(trueRhs, trueX);
   chrono.Stop();

   //trueRhs.Print();


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


   if (verbose)
       std::cout << "Running AMR if there was no S in the formulation... \n";

   if (numblocks != 2)
   {
       MFEM_ABORT("Currently adaptive mesh refinement is implemented when only sigma is present, "
                  "numblocks must equal 2 for CFOSLS formulation! \n");
   }


   BilinearFormIntegrator *integ = new VectorFEMassIntegrator(*Mytest.Ktilda);

   FOSLSEstimator estimator(*sigma, *integ);
   ThresholdRefiner refiner(estimator);
   refiner.SetTotalErrorFraction(0.5);

   delete Ablock;
   Ablock = new ParBilinearForm(Sigma_space);
   Ablock->AddDomainIntegrator(new VectorFEMassIntegrator(*Mytest.Ktilda));

   delete Dblock;
   Dblock = new ParMixedBilinearForm(Sigma_space, W_space);
   Dblock->AddDomainIntegrator(new VectorFEDivergenceIntegrator);

   // 12. The main AMR loop. In each iteration we solve the problem on the
   //     current mesh, visualize the solution, and refine the mesh.
   const int max_dofs = 200000;
   for (int it = 0; ; it++)
   {
      HYPRE_Int global_dofs = Sigma_space->GlobalTrueVSize();
      if (myid == 0)
      {
         cout << "\nAMR iteration " << it << endl;
         cout << "Number of unknowns: " << global_dofs << endl;
      }

      HypreParMatrix *A, *D, *DT;

      // 13. Assemble the stiffness matrix and the right-hand side. Note that
      //     MFEM doesn't care at this point that the mesh is nonconforming
      //     and parallel. The FE space is considered 'cut' along hanging
      //     edges/faces, and also across processor boundaries.

      Array<int> block_offsets(numblocks + 1);
      block_offsets[0] = 0;
      block_offsets[1] = Sigma_space->GetVSize();
      block_offsets[2] = W_space->GetVSize();
      block_offsets.PartialSum();
      BlockVector x(block_offsets);
      x = 0.0;
      ParGridFunction * sigma_exact = new ParGridFunction(Sigma_space);
      sigma_exact->ProjectCoefficient(*(Mytest.sigma));
      x.GetBlock(0) = *sigma_exact;

      fform->Assemble();
      gform->Assemble();

      Ablock->Assemble();
      Ablock->EliminateEssentialBC(ess_bdrSigma, x.GetBlock(0), *fform);
      Ablock->Finalize();
      A = Ablock->ParallelAssemble();

      Dblock->Assemble();
      Dblock->EliminateTrialDofs(ess_bdrSigma, x.GetBlock(0), *gform);
      Dblock->Finalize();
      D = Dblock->ParallelAssemble();
      DT = D->Transpose();

      // 14. Create the parallel linear system: eliminate boundary conditions,
      //     constrain hanging nodes and nodes across processor boundaries.
      //     The system will be solved for true (unconstrained/unique) DOFs only.

      Array<int> block_trueOffsets(numblocks + 1);
      block_trueOffsets[0] = 0;
      block_trueOffsets[1] = Sigma_space->GetTrueVSize();
      block_trueOffsets[2] = W_space->TrueVSize();
      block_trueOffsets.PartialSum();

      BlockVector trueX(block_trueOffsets);
      BlockVector trueRhs(block_trueOffsets);
      trueX = 0.0;
      trueRhs = 0.0;

      fform->ParallelAssemble(trueRhs.GetBlock(0));
      gform->ParallelAssemble(trueRhs.GetBlock(1));

      BlockOperator *CFOSLSop = new BlockOperator(block_trueOffsets);
      CFOSLSop->SetBlock(0,0, A);
      CFOSLSop->SetBlock(0,1, DT);
      CFOSLSop->SetBlock(1,0, D);

      // 15. Define and apply a parallel PCG solver for AX=B with the BoomerAMG
      //     preconditioner from hypre.

      HypreParMatrix *Schur;
      HypreParMatrix *AinvDt = D->Transpose();
      HypreParVector *Ad = new HypreParVector(MPI_COMM_WORLD, A->GetGlobalNumRows(), A->GetRowStarts());
      A->GetDiag(*Ad);
      AinvDt->InvScaleRows(*Ad);
      Schur = ParMult(D, AinvDt);

      Solver * invA;
      if (use_ADS)
          invA = new HypreADS(*A, Sigma_space);
      else // using Diag(A);
           invA = new HypreDiagScale(*A);

      invA->iterative_mode = false;

      Solver * invS;
      invS = new HypreBoomerAMG(*Schur);
      ((HypreBoomerAMG *)invS)->SetPrintLevel(0);
      ((HypreBoomerAMG *)invS)->iterative_mode = false;

      BlockDiagonalPreconditioner prec(block_trueOffsets);
      if (prec_option > 0)
      {
          prec.SetDiagonalBlock(0, invA);
          prec.SetDiagonalBlock(1, invS);
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
      //solver.SetPrintLevel(1);
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

      sigma->Distribute(&(trueX.GetBlock(0)));

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

      // 17. Send the solution by socket to a GLVis server.
      if (visualization)
      {
          char vishost[] = "localhost";
          int  visport   = 19916;

          socketstream s_sock(vishost, visport);

          s_sock << "parallel " << num_procs << " " << myid << "\n";
          s_sock << "solution\n" << *pmesh << *sigma << "window_title 'sigma, AMR iter No."
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
      W_space->Update();
      sigma->Update();

      // 21. Inform also the bilinear and linear forms that the space has
      //     changed.
      fform->Update();
      gform->Update();
      Ablock->Update();
      Dblock->Update();

      delete sigma_exact;
      delete A;
      delete D;
      delete DT;
      delete CFOSLSop;
   }


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

template<void(*bvec)(const Vector & x, Vector & vec)>
void minbTemplate(const Vector& xt, Vector& minb)
{
    minb.SetSize(xt.Size());

    bvec(xt,minb);

    minb *= -1;
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

template<double (*S)(const Vector & xt) > double uNonhomoTemplate(const Vector& xt)
{
    Vector xt0(xt.Size());
    xt0 = xt;
    xt0 (xt0.Size() - 1) = 0;

    return S(xt0);
}


template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
         void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
double rhsideTemplate(const Vector& xt)
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

    // only for debugging casuality weight usage
    //double t = xt[xt.Size()-1];
    //res *= exp (-t / 0.01);

    return res;

    /*
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    Vector b(3);
    bFunCircle2D_ex(xt,b);
    return 0.0 - (
           -100.0 * 2.0 * (x-0.5) * exp(-100.0 * ((x - 0.5) * (x - 0.5) + y * y)) * b(0) +
           -100.0 * 2.0 *    y    * exp(-100.0 * ((x - 0.5) * (x - 0.5) + y * y)) * b(1) );
    */
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
        res += b(i) * gradS(i);
    res += divbfunc(xt) * S(xt) ;

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
    //return exp(-100.0 * ((x - 0.5) * (x - 0.5) + y * y));
    double r = sqrt(x*x + y*y);
    double teta = atan2(y,x);
    //if (x > 0)
        //teta = atan(y/x);
    //else
        //teta = M_PIatan(y/x);
    return exp (-100.0 * (r * r - r * cos(teta) + 0.25));
}

double GaussianHill_dteta(const Vector&xvec)
{
    double x = xvec(0);
    double y = xvec(1);
    double r = sqrt(x*x + y*y);
    double teta = atan2(y,x);

    return -100.0 * r * sin (teta) * GaussianHill(xvec);
}

double GaussianHill_dr(const Vector&xvec)
{
    double x = xvec(0);
    double y = xvec(1);
    double r = sqrt(x*x + y*y);
    double teta = atan2(y,x);

    return -100.0 * (2.0 * r - cos (teta)) * GaussianHill(xvec);
}

double uFunCylinder_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double r = sqrt(x*x + y*y);
    double teta = atan2(y,x);

    double t = xt(xt.Size()-1);
    Vector xvec(2);
    xvec(0) = r * cos (teta - t);
    xvec(1) = r * sin (teta - t);
    return GaussianHill(xvec);
}

double uFunCylinder_ex_dt(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);
    double r = sqrt(x*x + y*y);
    double teta = atan2(y,x);
    return 100.0 * r * sin (teta - t) * uFunCylinder_ex(xt);
}

void uFunCylinder_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    // old, from parelag example slot 1
    // provides the same result as the different formula below

    // (x0,y0) = Q^tr * (x,y) = initial particle location
    double x0 = x * cos(t) + y * sin(t);
    double y0 = x * (-sin(t)) + y * cos(t);

    Vector r0vec(3);

    r0vec(0) = x0;
    r0vec(1) = y0;
    r0vec(2) = 0;

    // tempvec = grad u(x0,y0) at t = 0
    Vector tempvec(2);
    tempvec(0) = -100.0 * 2.0 * (x0 - 0.5) * uFunCylinder_ex(r0vec);
    tempvec(1) = -100.0 * 2.0 * y0 * uFunCylinder_ex(r0vec);


    //gradx = Q * tempvec
    gradx.SetSize(xt.Size() - 1);

    gradx(0) = tempvec(0) * cos(t) + tempvec(1) * (-sin(t));
    gradx(1) = tempvec(0) * sin(t) + tempvec(1) * cos(t);

    /*
     * new formula, gives the same result as the formula above
    double r = sqrt(x*x + y*y);
    double teta = atan2(y,x);

    double dSdr = uFunCylinder_ex(xt) * (-100.0) * (2.0 * r - cos (teta - t));
    double dSdteta = uFunCylinder_ex(xt) * (-100.0) * (r * sin (teta - t));

    double dtetadx = - (1.0/r) * sin(teta);
    double dtetady = + (1.0/r) * cos(teta);

    gradx.SetSize(xt.Size() - 1);
    gradx(0) = dSdr * cos(teta) + dSdteta * dtetadx;
    gradx(1) = dSdr * sin(teta) + dSdteta * dtetady;
    */

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

