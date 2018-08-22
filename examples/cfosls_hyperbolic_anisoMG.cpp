//                                MFEM(with 4D elements) CFOSLS with S from H1 for 3D/4D hyperbolic equation
//                                  with mesh generator and visualization
//
// ARCHIVED EXAMPLE: Used a looong time ago for experiments with multigrid on anisotropic mesh hierarchy
//
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

#include"cfosls_testsuite.hpp"
#include"divfree_solver_tools.hpp"

#define MYZEROTOL (1.0e-13)

using namespace std;
using namespace mfem;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

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


template <double (*ufunc)(const Vector&), void (*bvecfunc)(const Vector&, Vector& )>
    void sigmaTemplate(const Vector& xt, Vector& sigma);
template <void (*bvecfunc)(const Vector&, Vector& )>
        void bbTTemplate(const Vector& xt, DenseMatrix& bbT);
template <void (*bvecfunc)(const Vector&, Vector& )>
    double bTbTemplate(const Vector& xt);
template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
        void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
        void bfTemplate(const Vector& xt, Vector& bf);
template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
        void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
        double divsigmaTemplate(const Vector& xt);

class Transport_test
    {
    protected:
        int dim;
        int numsol;

    public:
        FunctionCoefficient * scalaru;
        FunctionCoefficient * divsigma;         // = dS/dt + div (bS) = div sigma
        FunctionCoefficient * bTb;
        VectorFunctionCoefficient * sigma;
        VectorFunctionCoefficient * conv;
        VectorFunctionCoefficient * bf;
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

        void SetScalarFun( double (*S)(const Vector & xt))
        { scalaru = new FunctionCoefficient(S);}

        template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
                 void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
        void SetScalarfFun()
        { divsigma = new FunctionCoefficient(
                        divsigmaTemplate<S, dSdt, Sgradxvec, bvec, divbfunc>);}

        template< void(*f2)(const Vector & x, Vector & vec)>  \
        void SetScalarBtB()
        {
            bTb = new FunctionCoefficient(bTbTemplate<f2>);
        }

        template<double (*f1)(const Vector & xt), void(*f2)(const Vector & x, Vector & vec)> \
        void SetHdivVec()
        {
            sigma = new VectorFunctionCoefficient(dim, sigmaTemplate<f1,f2>);
        }

        void SetConvVec( void(*f)(const Vector & x, Vector & vec))
        { conv = new VectorFunctionCoefficient(dim, f);}

        template< void(*bvec)(const Vector & x, Vector & vec)>  \
        void SetBBtMat()
        {
            bbT = new MatrixFunctionCoefficient(dim, bbTTemplate<bvec>);
        }

        template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
                 void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt) > \
        void SetConvfFunVec()
        { bf = new VectorFunctionCoefficient(dim, bfTemplate<S,dSdt,Sgradxvec,bvec,divbfunc>);}

        template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
                 void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
        void SetDivSigma()
        { divsigma = new FunctionCoefficient(divsigmaTemplate<S, dSdt, Sgradxvec, bvec, divbfunc>);}

    };

    template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx),  \
             void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
    void Transport_test::SetTestCoeffs ()
    {
        SetScalarFun(S);
        SetScalarfFun<S, dSdt, Sgradxvec, bvec, divbfunc>();
        SetConvVec(bvec);
        SetConvfFunVec<S, dSdt, Sgradxvec, bvec, divbfunc>();
        SetHdivVec<S,bvec>();
        SetScalarBtB<bvec>();
        SetDivSigma<S, dSdt, Sgradxvec, bvec, divbfunc>();
        SetBBtMat<bvec>();
        return;
    }

    Transport_test::Transport_test (int Dim, int NumSol)
    {
        dim = Dim;
        numsol = NumSol;

        {
            if (numsol == -3) // 3D test for the paper
            {
                SetTestCoeffs<&uFunTest_ex, &uFunTest_ex_dt, &uFunTest_ex_gradx, &bFunRect2D_ex, &bFunRect2Ddiv_ex>();
            }
            if (numsol == -4) // 4D test for the paper
            {
                SetTestCoeffs<&uFunTest_ex, &uFunTest_ex_dt, &uFunTest_ex_gradx, &bFunCube3D_ex, &bFunCube3Ddiv_ex>();
            }
        } // end of setting test coefficients in correct case
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
    int numsol          = 0;

    int par_ref_levels  = 3;

    // solver options
    int prec_option = 2; // 2: block diagonal MG   3: monolithic MG

    bool aniso_refine = true;
    bool refine_t_first = false;
    bool with_multilevel = true;

    int feorder = 0;

    if (verbose)
        cout << "Solving (С)FOSLS Transport equation with MFEM & hypre" << endl;

    OptionsParser args(argc, argv);
    args.AddOption(&feorder, "-o", "--feorder",
                   "Finite element order (polynomial degree).");
    args.AddOption(&par_ref_levels, "-pref", "--pref",
                   "Number of parallel refinements 4d mesh.");
    args.AddOption(&nDimensions, "-dim", "--whichD",
                   "Dimension of the space-time problem.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&prec_option, "-precopt", "--prec-option",
                   "Preconditioner choice: 2: block diagonal MG   3: monolithic MG.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&aniso_refine, "-aniso", "--aniso-refine", "-iso",
                   "--iso-refine",
                   "Using anisotropic or isotropic refinement.");
    args.AddOption(&refine_t_first, "-refine-t-first", "--refine-time-first",
                   "-refine-x-first", "--refine-space-first",
                   "Refine time or space first in anisotropic refinement.");
    args.AddOption(&with_multilevel, "-ml", "--multilvl", "-no-ml",
                   "--no-multilvl",
                   "Enable or disable multilevel algorithm for finding a particular solution.");

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

    numsol = -3;
    if (verbose)
        std::cout << "For the records: numsol = " << numsol << "\n";

    StopWatch chrono;

    //DEFAULTED LINEAR SOLVER OPTIONS
    int max_iter = 50000;
    double rtol = 1e-12;
    double atol = 1e-14;

    auto mesh = make_shared<Mesh>(2, 2, 2, Element::HEXAHEDRON, 1);

    // Do a general refine and turn the mesh into nonconforming mesh
    Array<Refinement> refs(mesh->GetNE());
    for (int i = 0; i < mesh->GetNE(); i++)
    {
        refs[i] = Refinement(i, 7);
    }
    mesh->GeneralRefinement(refs, -1, -1);
    auto pmesh = make_shared<ParMesh>(comm, *mesh);

    // 6. Define a parallel finite element space on the parallel mesh. Here we
    //    use the Raviart-Thomas finite elements of the specified order.
    int dim = nDimensions;

    auto hdiv_coll = new RT_FECollection(feorder, dim);
    auto h1_coll = new H1_FECollection(feorder+1, dim);
    auto l2_coll = new L2_FECollection(feorder, dim);
    auto hcurl_coll = new ND_FECollection(feorder + 1,  dim);

    auto C_space = new ParFiniteElementSpace(pmesh.get(), hcurl_coll);
    auto R_space = new ParFiniteElementSpace(pmesh.get(), hdiv_coll);
    auto H_space = new ParFiniteElementSpace(pmesh.get(), h1_coll);
    auto W_space = new ParFiniteElementSpace(pmesh.get(), l2_coll);

    if (aniso_refine)
        par_ref_levels *= 2;

    Array<HypreParMatrix*> P_C(par_ref_levels), P_H(par_ref_levels);
    auto coarseC_space = new ParFiniteElementSpace(pmesh.get(), hcurl_coll);
    auto coarseH_space = new ParFiniteElementSpace(pmesh.get(), h1_coll);

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

        coarseR_space = new ParFiniteElementSpace(pmesh.get(), hdiv_coll);
        coarseW_space = new ParFiniteElementSpace(pmesh.get(), l2_coll);

        // Dofs_TrueDofs at each space:

        d_td_coarse_R = coarseR_space->Dof_TrueDof_Matrix();
        d_td_coarse_W = coarseW_space->Dof_TrueDof_Matrix();

        Array<int> ess_bdrSigma(pmesh->bdr_attributes.Max());
        ess_bdrSigma = 0;
        for (int l = 0; l < ref_levels+1; l++)
        {
            if (l > 0){

                if (l == 1)
                {
                    R_space->GetEssentialVDofs(ess_bdrSigma, ess_dof_coarsestlvl_list);
                }

                coarseC_space->Update();

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

                H_space->Update();
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


                P_W_local = ((const SparseMatrix *)W_space->GetUpdateOperator());
                P_R_local = ((const SparseMatrix *)R_space->GetUpdateOperator());

                SparseMatrix* R_Element_to_dofs1 = new SparseMatrix();
                SparseMatrix* W_Element_to_dofs1 = new SparseMatrix();

                divp.Elem2Dofs(*R_space, *R_Element_to_dofs1);
                divp.Elem2Dofs(*W_space, *W_Element_to_dofs1);

                P_W[ref_levels -l] = RemoveZeroEntries(*P_W_local);
                P_R[ref_levels -l] = RemoveZeroEntries(*P_R_local);

                Element_dofs_R[ref_levels - l] = R_Element_to_dofs1;
                Element_dofs_W[ref_levels - l] = W_Element_to_dofs1;

            }
        }
    }
    else // not a multilevel algo
    {
        if (verbose)
            std::cout << "Creating a hierarchy of meshes by successive refinements (and multigrid prerequisites) \n";

        for (int l = 0; l < par_ref_levels; l++)
        {
            coarseC_space->Update();

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

            W_space->Update();
            R_space->Update();
            C_space->Update();
            H_space->Update();

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
        } // end of loop over mesh levels
    } // end of else (not a multilevel algo)
    if (verbose)
        cout<<"MG hierarchy constructed in "<< chrono.RealTime() <<" seconds.\n";

    pmesh->PrintInfo(std::cout); if(verbose) cout << "\n";

    HYPRE_Int dimR = R_space->GlobalTrueVSize();
    HYPRE_Int dimH = H_space->GlobalTrueVSize();
    HYPRE_Int dimW = W_space->GlobalTrueVSize();

    if (verbose)
    {
       std::cout << "***********************************************************\n";
       std::cout << "dim(R) = " << dimR << ", ";
       std::cout << "dim(H) = " << dimH << ", ";
       std::cout << "dim(W) = " << dimW << ", ";
       std::cout << "dim(R+H+W) = " << dimR + dimH + dimW << "\n";
       std::cout << "***********************************************************\n";
    }

    // 7. Define the two BlockStructure of the problem.

    Array<int> block_offsets(3); // number of variables + 1
    block_offsets[0] = 0;
    block_offsets[1] = C_space->GetVSize();
    block_offsets[2] = H_space->GetVSize();
    block_offsets.PartialSum();

    Array<int> block_trueOffsets(3); // number of variables + 1
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = C_space->TrueVSize();
    block_trueOffsets[2] = H_space->TrueVSize();
    block_trueOffsets.PartialSum();

    Vector x(H_space->GetVSize());
    BlockVector trueX(block_trueOffsets), trueRhs(block_trueOffsets);
    x = 0.0;
    trueX = 0.0;
    trueRhs = 0.0;

   // 8. Define the coefficients, analytical solution, and rhs of the PDE.
   Transport_test Mytest(nDimensions,numsol);

   // for S boundary conditions are essential only for t = 0
   Array<int> ess_bdrS(pmesh->bdr_attributes.Max());
   ess_bdrS = 0;
   ess_bdrS[0] = 1; // t = 0
   //-----------------------

   // 9. Define the parallel grid function and parallel linear forms, solution
   //    vector and rhs.

   ParLinearForm *qform(new ParLinearForm(H_space));
   qform->AddDomainIntegrator(new GradDomainLFIntegrator(*Mytest.bf));
   qform->Assemble();

   ParLinearForm *gform(new ParLinearForm(W_space));
   gform->AddDomainIntegrator(new DomainLFIntegrator(*Mytest.divsigma));
   gform->Assemble();

   // 10. Assemble the finite element matrices for the CFOSLS operator  A
   //     where:

   ParBilinearForm *Mblock(new ParBilinearForm(R_space));
   Mblock->AddDomainIntegrator(new VectorFEMassIntegrator);
   Mblock->Assemble();
   Mblock->Finalize();
   HypreParMatrix *M = Mblock->ParallelAssemble();

//---------------
//  C Block:
//---------------

   ParBilinearForm *Xblock(new ParBilinearForm(H_space));
   Xblock->AddDomainIntegrator(new MassIntegrator(*Mytest.bTb));
   Xblock->AddDomainIntegrator(new DiffusionIntegrator(*Mytest.bbT));
   Xblock->Assemble();
   Xblock->EliminateEssentialBC(ess_bdrS, x, *qform);
   Xblock->Finalize();
   HypreParMatrix *X = Xblock->ParallelAssemble();

//---------------
//  B Block:
//---------------

   ParMixedBilinearForm *Gblock(new ParMixedBilinearForm(R_space, H_space));
   Gblock->AddDomainIntegrator(new VectorFEMassIntegrator(*Mytest.conv));
   Gblock->Assemble();
   Gblock->EliminateTestDofs(ess_bdrS);
   Gblock->Finalize();
   HypreParMatrix *G = Gblock->ParallelAssemble();
   *G *= -1.;

//----------------
//  D Block:
//-----------------

   ParMixedBilinearForm *Dblock(new ParMixedBilinearForm(R_space, W_space));
   Dblock->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   Dblock->Assemble();
   Dblock->Finalize();
   HypreParMatrix *D = Dblock->ParallelAssemble();
   HypreParMatrix *DT = D->Transpose();

//----------------
//  Discrete curl:
//-----------------

   ParDiscreteLinearOperator DiscreteCurlForm(C_space, R_space);
   DiscreteCurlForm.AddDomainInterpolator(new CurlInterpolator());
   DiscreteCurlForm.Assemble();
   DiscreteCurlForm.Finalize();
   HypreParMatrix * C = DiscreteCurlForm.ParallelAssemble();
   HypreParMatrix * CT = C->Transpose();


//=======================================================
// Assembling the Matrix
//-------------------------------------------------------

  auto CM = ParMult(CT, M);
  auto CMC = ParMult(CM, C);

  auto GC = ParMult(G, C);
  auto GCT = GC->Transpose();

  BlockOperator *CFOSLSop = new BlockOperator(block_trueOffsets);
  CFOSLSop->SetBlock(0,0, CMC);
  CFOSLSop->SetBlock(0,1, GCT);
  CFOSLSop->SetBlock(1,0, GC);
  CFOSLSop->SetBlock(1,1, X);

   if (verbose)
       cout<< "Final block system assembled"<<endl << flush;
   MPI_Barrier(MPI_COMM_WORLD);

   mfem::CGSolver solver(comm);
   solver.SetPrintLevel(0);
   solver.SetMaxIter(max_iter);
   solver.SetRelTol(rtol);
   solver.SetAbsTol(atol);

   // Find a particular solution
   chrono.Clear();
   chrono.Start();
   ParGridFunction * Sigmahat = new ParGridFunction(R_space);\
   HypreParMatrix *Bdiv;
   if (with_multilevel)
   {
       if (verbose)
           std::cout << "Using multilevel algorithm for finding a particular solution \n";

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

       F_fine = *gform;
       G_fine = .0;

       divp.div_part(ref_levels,
                     NULL, B_local,
                     G_fine,
                     F_fine,
                     P_W, P_R, P_W,
                     Element_dofs_R,
                     Element_dofs_W,
                     d_td_coarse_R,
                     d_td_coarse_W,
                     *Sigmahat,
                     ess_dof_coarsestlvl_list);
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

       Bblock = new ParMixedBilinearForm(R_space, W_space);
       Bblock->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
       Bblock->Assemble();
       Bblock->Finalize();
       Bdiv = Bblock->ParallelAssemble();
       BdivT = Bdiv->Transpose();
       BBT = ParMult(Bdiv, BdivT);
       Rhs = gform->ParallelAssemble();

       HypreBoomerAMG * invBBT = new HypreBoomerAMG(*BBT);
       invBBT->SetPrintLevel(0);

       solver.SetPreconditioner(*invBBT);
       solver.SetOperator(*BBT);

       Vector * Temphat = new Vector(W_space->TrueVSize());
       *Temphat = 0.0;
       solver.Mult(*Rhs, *Temphat);

       Vector * Temp = new Vector(R_space->TrueVSize());
       BdivT->Mult(*Temphat, *Temp);

       Sigmahat->Distribute(*Temp);
   }
   if (verbose)
       cout << "A particular solution found in "
            << chrono.RealTime() << " seconds.\n";

   Vector sigma_part(R_space->TrueVSize());
   sigma_part = 0.0;
   Sigmahat->ParallelProject(sigma_part);

   // Computing right hand side
   CM->Mult(-1.0, sigma_part, 1.0, trueRhs.GetBlock(0));
   qform->ParallelAssemble(trueRhs.GetBlock(1));
   G->Mult(-1.0, sigma_part, 1.0, trueRhs.GetBlock(1));

   // 12. Solve the linear system with CG.
   chrono.Clear();
   chrono.Start();

   Solver *prec;
   Array<BlockOperator*> P;
   if (prec_option==3)
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
       prec = new MonolithicMultigrid(*CFOSLSop, P);
   }
   else
   {
       MFEM_ASSERT(prec_option == 2, "prec_option can either be 2 or 3");
       Multigrid * invCMC = new Multigrid(*CMC, P_C);
       Multigrid * invX = new Multigrid(*X, P_H);

       prec = new BlockDiagonalPreconditioner(block_trueOffsets);
       ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(0, invCMC);
       ((BlockDiagonalPreconditioner*)prec)->SetDiagonalBlock(1, invX);
   }

   if (verbose)
       std::cout << "Preconditioner built in " << chrono.RealTime() << "s. \n";

   // 12. Solve the linear system with MINRES.
   //     Check the norm of the unpreconditioned residual.
   chrono.Clear();
   chrono.Start();
   solver.SetPreconditioner(*prec);
   solver.SetOperator(*CFOSLSop);
   trueX = 0.0;
   solver.Mult(trueRhs, trueX);
   chrono.Stop();

   if (verbose)
   {
      if (solver.GetConverged())
         std::cout << "CG converged in " << solver.GetNumIterations()
                   << " iterations with a residual norm of " << solver.GetFinalNorm() << ".\n";
      else
         std::cout << "CG did not converge in " << solver.GetNumIterations()
                   << " iterations. Residual norm is " << solver.GetFinalNorm() << ".\n";
      std::cout << "CG solver took " << chrono.RealTime() << "s. \n";
   }

   Vector trueSigma(R_space->TrueVSize());
   trueSigma = 0.0;
   C->Mult(trueX.GetBlock(0), trueSigma);
   trueSigma += sigma_part;

   ParGridFunction * sigma = new ParGridFunction(R_space);
   sigma->Distribute(trueSigma);

   ParGridFunction * S = new ParGridFunction(H_space);
   S->Distribute(&(trueX.GetBlock(1)));

   ParGridFunction *sigma_exact = new ParGridFunction(R_space);
   sigma_exact->ProjectCoefficient(*(Mytest.sigma));

   ParGridFunction *S_exact = new ParGridFunction(H_space);
   S_exact->ProjectCoefficient(*(Mytest.scalaru));

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
       cout << "|| sigma - sigma_ex || / || sigma_ex || = "
            << err_sigma / norm_sigma << endl;

   DiscreteLinearOperator Div(R_space, W_space);
   Div.AddDomainInterpolator(new DivergenceInterpolator());
   ParGridFunction DivSigma(W_space);
   Div.Assemble();
   Div.Mult(*sigma, DivSigma);

   double err_div = DivSigma.ComputeL2Error(*(Mytest.divsigma),irs);
   double norm_div = ComputeGlobalLpNorm(2, *(Mytest.divsigma), *pmesh, irs);

   if (verbose)
   {
       cout << "|| div_h (sigma_h - sigma_ex) || / ||div (sigma_ex)|| = "
                 << err_div/norm_div  << "\n";
   }

   if (verbose)
   {
      cout << "|| sigma_h - sigma_ex ||_Hdiv / || sigma_ex ||_Hdiv = "
                 << sqrt(err_sigma*err_sigma + err_div * err_div)/
                    sqrt(norm_sigma*norm_sigma + norm_div * norm_div)  << "\n";
   }

   // Computing error for S

   double err_S = S->ComputeL2Error((*Mytest.scalaru), irs);
   double norm_S = ComputeGlobalLpNorm(2, (*Mytest.scalaru), *pmesh, irs);
   if (verbose)
   {
       std::cout << "|| S_h - S_ex || / || S_ex || = " <<
                    err_S / norm_S << "\n";
   }

   {
       DiscreteLinearOperator Grad(H_space, C_space);
       Grad.AddDomainInterpolator(new GradientInterpolator());
       ParGridFunction GradS(C_space);
       Grad.Assemble();
       Grad.Mult(*S, GradS);

       VectorFunctionCoefficient GradS_coeff(dim, uFunTest_ex_gradxt);
       double err_GradS = GradS.ComputeL2Error(GradS_coeff, irs);
       double norm_GradS = ComputeGlobalLpNorm(2, GradS_coeff, *pmesh, irs);
       if (verbose)
       {
           std::cout << "|| Grad_h (S_h - S_ex) || / || Grad S_ex || = " <<
                        err_GradS / norm_GradS << "\n";
           std::cout << "|| S_h - S_ex ||_H^1 / || S_ex ||_H^1 = " <<
                        sqrt(err_S*err_S + err_GradS*err_GradS) /
                        sqrt(norm_S*norm_S + norm_GradS*norm_GradS) << "\n";
       }
   }

   // Check value of functional and mass conservation
   {
       trueRhs.GetBlock(1) = 0.0;
       qform->ParallelAssemble(trueRhs.GetBlock(1));
       double localFunctional = -2.0*(trueX.GetBlock(1)*trueRhs.GetBlock(1));

       Vector MtrueSigma(R_space->TrueVSize());
       MtrueSigma = 0.0;
       M->Mult(trueSigma, MtrueSigma);
       localFunctional += trueSigma*MtrueSigma;

       Vector GtrueSigma(H_space->TrueVSize());
       GtrueSigma = 0.0;
       G->Mult(trueSigma, GtrueSigma);
       localFunctional += 2.0*(trueX.GetBlock(1)*GtrueSigma);

       Vector XtrueS(H_space->TrueVSize());
       XtrueS = 0.0;
       X->Mult(trueX.GetBlock(1), XtrueS);
       localFunctional += trueX.GetBlock(1)*XtrueS;

       double globalFunctional;
       MPI_Reduce(&localFunctional, &globalFunctional, 1,
                  MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
       if (verbose)
       {
           cout << "|| sigma_h - L(S_h) ||^2 + || div_h (bS_h) - f ||^2 = "
                << globalFunctional+norm_div*norm_div<< "\n";
           cout << "|| f ||^2 = " << norm_div*norm_div  << "\n";
           cout << "Relative Energy Error = "
                << sqrt(globalFunctional+norm_div*norm_div)/norm_div<< "\n";
       }

       Vector trueRhs_part(W_space->TrueVSize());
       trueRhs_part = 0.0;
       gform->ParallelAssemble(trueRhs_part);
       double mass_loc = trueRhs_part.Norml1();
       double mass;
       MPI_Reduce(&mass_loc, &mass, 1,
                  MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
       if (verbose)
           cout << "Sum of local mass = " << mass<< "\n";

       Vector DtrueSigma(W_space->TrueVSize());
       DtrueSigma = 0.0;
       D->Mult(trueSigma, DtrueSigma);
       DtrueSigma -= trueRhs_part;
       double mass_loss_loc = DtrueSigma.Norml1();
       double mass_loss;
       MPI_Reduce(&mass_loss_loc, &mass_loss, 1,
                  MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
       if (verbose)
           cout << "Sum of local mass loss = " << mass_loss<< "\n";
   }

   if (verbose)
       cout << "Computing projection errors" << endl;

   double proj_error_sigma = sigma_exact->ComputeL2Error(*(Mytest.sigma), irs);

   if(verbose)
   {
       cout << "|| sigma_ex - Pi_h sigma_ex || / || sigma_ex || = "
                       << proj_error_sigma / norm_sigma << endl;
   }

   double proj_error_S = S_exact->ComputeL2Error(*(Mytest.scalaru), irs);

   if(verbose)
       cout << "|| S_ex - Pi_h S_ex || / || S_ex || = "
                       << proj_error_S / norm_S << endl;

   // 17. Free the used memory.
   delete M;
   delete G;
   delete X;
   delete D;
   delete DT;
   delete C;
   delete CT;

   delete C_space;
   delete W_space;
   delete H_space;
   delete R_space;
   delete hcurl_coll;
   delete l2_coll;
   delete h1_coll;
   delete hdiv_coll;

   MPI_Finalize();

   return 0;
}

template <void (*bvecfunc)(const Vector&, Vector& )> \
void bbTTemplate(const Vector& xt, DenseMatrix& bbT)
{
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

    double res = 0.0;

    res += dSdt(xt);
    for ( int i= 0; i < xt.Size() - 1; ++i )
        res += b(i) * gradS(i);
    res += divbfunc(xt) * S(xt);

    bf.SetSize(xt.Size());

    for (int i = 0; i < bf.Size(); ++i)
        bf(i) = res * b(i);
}

