#include "testhead.hpp"

#ifndef MFEM_CFOSLS_INTEGRATORS
#define MFEM_CFOSLS_INTEGRATORS

namespace mfem
{

/// Integrator for (qmass u, v) + (qdiff grad_u, grad_v)
/// where u and v are from the same scalar f.e. space
/// and qmass is a scalar and qdiff is a scalar/tensor coefficients
class H1NormIntegrator : public BilinearFormIntegrator
{
private:
   Vector vec, pointflux, shape;
#ifndef MFEM_THREAD_SAFE
   DenseMatrix dshape, dshapedxt, invdfdx;
#endif
   Coefficient *Qdiff;
   Coefficient *Qmass;
   MatrixCoefficient *MQ;

public:
   /// Construct an H1 norm integrator (mass + diffusion) with unit coefficients
   H1NormIntegrator() { Qdiff = NULL; Qmass = NULL; MQ = NULL; }

   /// Construct an H1 norm integrator with a scalar coefficients qdiff and qmass
   H1NormIntegrator (Coefficient &qdiff, Coefficient &qmass) : Qdiff(&qdiff), Qmass(&qmass) { MQ = NULL; }

   /// Construct an H1 norm integrator with a matrix coefficient q
   H1NormIntegrator (MatrixCoefficient &q, Coefficient &qmass) : Qmass(&qmass), MQ(&q) { Qdiff = NULL; }

   /** Given a particular Finite Element
       computes the element stiffness matrix elmat. */
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
};

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

/// FIXME: Isn't it one of the standard integrators? Is it used anywhere?
/// Integrator for (sigma, Q * v)
/// where sigma is from a vector f.e. (like H(div)), u is
/// from a scalar f.e. space and Q is a vector coefficient
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

   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

/// Integrator for (q * (- grad_x u, grad_t u)^T, (- grad_x v, grad_t v)^T)
/// where u and v are from the same scalar f.e. and q is a scalar coefficient
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
};

/// Bilinear integrator for (curl u, Q v) for Nedelec and scalar finite element for v.
/// If the trial and test spaces are switched, assembles the form (u, curl v).
class VectorFECurlVQIntegrator: public BilinearFormIntegrator
{
private:
    VectorCoefficient *VQ;
#ifndef MFEM_THREAD_SAFE
    Vector shape;
    DenseMatrix curlshape;
    DenseMatrix curlshape_dFT;
    Vector D;
#endif
    void Init(VectorCoefficient *vq)
    { VQ = vq; }
public:
    VectorFECurlVQIntegrator() { Init(NULL); }
    VectorFECurlVQIntegrator(VectorCoefficient &vq) { Init(&vq); }
    VectorFECurlVQIntegrator(VectorCoefficient *vq) { Init(vq); }
    virtual void AssembleElementMatrix(const FiniteElement &el,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat) { }
    virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                        const FiniteElement &test_fe,
                                        ElementTransformation &Trans,
                                        DenseMatrix &elmat);
};

using MixedVectorFECurlVQScalarIntegrator = VectorFECurlVQIntegrator;

/// Linear form integrator for (f, Q * curl psi)
/// where psi is from H(curl) and Q is a vector coefficient
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

/// Linear form integrator for (f, Q * div sigma)
/// where sigma is from a vector f.e. space (like H(div))
/// and Q is a scalar coefficient
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

/// Linear form integrator for (f, Q * grad_xt u)
/// where u is from scalar f.e. space and Q is a vector coefficient
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

/// Integrator for (q * u, v)
/// where u and v are from vector FE space
/// created from scalar FE collection (called improper vector FE)
/// and q is a scalar coefficient
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

/// Integrators used for the CFOSLS formnulation of the heat equation

/// Integrator for (sigma, Q * (-grad_x u, u)^T)
/// where sigma from Hdiv (VectorFE) and u from H^1 (scalar type)
/// and Q is a scalar coefficient
class CFOSLS_MixedHeatIntegrator: public BilinearFormIntegrator
{
private:
    Coefficient *Q;
    VectorCoefficient *VQ;
    MatrixCoefficient *MQ;
    void Init(Coefficient *q, VectorCoefficient *vq, MatrixCoefficient *mq)
    { Q = q; VQ = vq; MQ = mq; }

#ifndef MFEM_THREAD_SAFE
    Vector trial_shape;
    DenseMatrix test_vshape;
    DenseMatrix trial_dshape;//<<<<<<<<<<<<<<
#endif

public:
    CFOSLS_MixedHeatIntegrator() { Init(NULL, NULL, NULL); }
    CFOSLS_MixedHeatIntegrator(Coefficient *_q) { Init(_q, NULL, NULL); }
    CFOSLS_MixedHeatIntegrator(Coefficient &q) { Init(&q, NULL, NULL); }
    CFOSLS_MixedHeatIntegrator(VectorCoefficient *_vq) { Init(NULL, _vq, NULL); }
    CFOSLS_MixedHeatIntegrator(VectorCoefficient &vq) { Init(NULL, &vq, NULL); }
    CFOSLS_MixedHeatIntegrator(MatrixCoefficient *_mq) { Init(NULL, NULL, _mq); }
    CFOSLS_MixedHeatIntegrator(MatrixCoefficient &mq) { Init(NULL, NULL, &mq); }

    virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                        const FiniteElement &test_fe,
                                        ElementTransformation &Trans,
                                        DenseMatrix &elmat);
};

/// Integrator for (Q * (-grad_x u, u)^T, (-grad_x v, v)^T)
/// where u and v are from the same scalar f.e. space
/// and Q is a scalar coefficient
class CFOSLS_HeatIntegrator: public BilinearFormIntegrator
{
private:
    Coefficient *Q;
    VectorCoefficient *VQ;
    MatrixCoefficient *MQ;
    void Init(Coefficient *q, VectorCoefficient *vq, MatrixCoefficient *mq)
    { Q = q; VQ = vq; MQ = mq; }

#ifndef MFEM_THREAD_SAFE
    Vector shape;
    DenseMatrix dshape;
    DenseMatrix dshapedxt;
    DenseMatrix invdfdx;
#endif

public:
    CFOSLS_HeatIntegrator() { Init(NULL, NULL, NULL); }
    CFOSLS_HeatIntegrator(Coefficient *_q) { Init(_q, NULL, NULL); }
    CFOSLS_HeatIntegrator(Coefficient &q) { Init(&q, NULL, NULL); }
    CFOSLS_HeatIntegrator(VectorCoefficient *_vq) { Init(NULL, _vq, NULL); }
    CFOSLS_HeatIntegrator(VectorCoefficient &vq) { Init(NULL, &vq, NULL); }
    CFOSLS_HeatIntegrator(MatrixCoefficient *_mq) { Init(NULL, NULL, _mq); }
    CFOSLS_HeatIntegrator(MatrixCoefficient &mq) { Init(NULL, NULL, &mq); }

    virtual void AssembleElementMatrix(const FiniteElement &el,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

/// Integrator for (sigma, (- grad_x u, grad_t u)^T)
/// where sigma is from a vector f.e. (like H(div)), u
/// is from a scalar f.e., and q is a scalar coefficient
class CFOSLS_MixedWaveIntegrator: public BilinearFormIntegrator
{
private:
    Coefficient *Q;
    VectorCoefficient *VQ;
    MatrixCoefficient *MQ;
    void Init(Coefficient *q, VectorCoefficient *vq, MatrixCoefficient *mq)
    { Q = q; VQ = vq; MQ = mq; }

#ifndef MFEM_THREAD_SAFE
    Vector trial_shape;
    DenseMatrix test_vshape;
    DenseMatrix trial_dshape;//<<<<<<<<<<<<<<
#endif

public:
    CFOSLS_MixedWaveIntegrator() { Init(NULL, NULL, NULL); }
    CFOSLS_MixedWaveIntegrator(Coefficient *_q) { Init(_q, NULL, NULL); }
    CFOSLS_MixedWaveIntegrator(Coefficient &q) { Init(&q, NULL, NULL); }
    CFOSLS_MixedWaveIntegrator(VectorCoefficient *_vq) { Init(NULL, _vq, NULL); }
    CFOSLS_MixedWaveIntegrator(VectorCoefficient &vq) { Init(NULL, &vq, NULL); }
    CFOSLS_MixedWaveIntegrator(MatrixCoefficient *_mq) { Init(NULL, NULL, _mq); }
    CFOSLS_MixedWaveIntegrator(MatrixCoefficient &mq) { Init(NULL, NULL, &mq); }

    virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                        const FiniteElement &test_fe,
                                        ElementTransformation &Trans,
                                        DenseMatrix &elmat);
};

/// Integrator for (q * (- grad_x u, grad_t u)^T, (- grad_x v, grad_t v)^T)
/// where u and v are from the same scalar f.e. space,
/// and q is a scalar coefficient
class CFOSLS_WaveIntegrator: public BilinearFormIntegrator
{
private:
    Coefficient *Q;
    VectorCoefficient *VQ;
    MatrixCoefficient *MQ;
    void Init(Coefficient *q, VectorCoefficient *vq, MatrixCoefficient *mq)
    { Q = q; VQ = vq; MQ = mq; }

#ifndef MFEM_THREAD_SAFE
    Vector shape;
    DenseMatrix dshape;
    DenseMatrix dshapedxt;
    DenseMatrix invdfdx;
#endif

public:
    CFOSLS_WaveIntegrator() { Init(NULL, NULL, NULL); }
    CFOSLS_WaveIntegrator(Coefficient *_q) { Init(_q, NULL, NULL); }
    CFOSLS_WaveIntegrator(Coefficient &q) { Init(&q, NULL, NULL); }
    CFOSLS_WaveIntegrator(VectorCoefficient *_vq) { Init(NULL, _vq, NULL); }
    CFOSLS_WaveIntegrator(VectorCoefficient &vq) { Init(NULL, &vq, NULL); }
    CFOSLS_WaveIntegrator(MatrixCoefficient *_mq) { Init(NULL, NULL, _mq); }
    CFOSLS_WaveIntegrator(MatrixCoefficient &mq) { Init(NULL, NULL, &mq); }

    virtual void AssembleElementMatrix(const FiniteElement &el,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

} // for namespace mfem

#endif
