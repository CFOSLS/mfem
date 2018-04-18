#include "testhead.hpp"


namespace mfem
{

void H1NormIntegrator::AssembleElementMatrix
( const FiniteElement &el, ElementTransformation &Trans,
  DenseMatrix &elmat )
{
   int nd = el.GetDof();
   int dim = el.GetDim();
   int spaceDim = Trans.GetSpaceDim();
   bool square = (dim == spaceDim);
   double w;


#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape(nd,dim), dshapedxt(nd,spaceDim), invdfdx(dim,spaceDim);
   Vector shape;
#else
   dshape.SetSize(nd,dim);
   dshapedxt.SetSize(nd,spaceDim);
   invdfdx.SetSize(dim,spaceDim);
#endif
   elmat.SetSize(nd);
   shape.SetSize(nd);

   const IntegrationRule *ir_diff = IntRule;
   if (ir_diff == NULL)
   {
      int order;
      if (el.Space() == FunctionSpace::Pk)
      {
         order = 2*el.GetOrder() - 2;
      }
      else
         // order = 2*el.GetOrder() - 2;  // <-- this seems to work fine too
      {
         order = 2*el.GetOrder() + dim - 1;
      }

      if (el.Space() == FunctionSpace::rQk)
      {
         ir_diff = &RefinedIntRules.Get(el.GetGeomType(), order);
      }
      else
      {
         ir_diff = &IntRules.Get(el.GetGeomType(), order);
      }
   }

   elmat = 0.0;
   for (int i = 0; i < ir_diff->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir_diff->IntPoint(i);
      el.CalcDShape(ip, dshape);

      Trans.SetIntPoint(&ip);
      w = Trans.Weight();
      w = ip.weight / (square ? w : w*w*w);
      // AdjugateJacobian = / adj(J),         if J is square
      //                    \ adj(J^t.J).J^t, otherwise
      Mult(dshape, Trans.AdjugateJacobian(), dshapedxt);
      if (!MQ)
      {
         if (Qdiff)
         {
            w *= Qdiff->Eval(Trans, ip);
         }
         AddMult_a_AAt(w, dshapedxt, elmat);
      }
      else
      {
         MQ->Eval(invdfdx, Trans, ip);
         invdfdx *= w;
         Mult(dshapedxt, invdfdx, dshape);
         AddMultABt(dshape, dshapedxt, elmat);
      }
   }

   const IntegrationRule *ir_mass = IntRule;
   if (ir_mass == NULL)
   {
      // int order = 2 * el.GetOrder();
      int order = 2 * el.GetOrder() + Trans.OrderW();

      if (el.Space() == FunctionSpace::rQk)
      {
         ir_mass = &RefinedIntRules.Get(el.GetGeomType(), order);
      }
      else
      {
         ir_mass = &IntRules.Get(el.GetGeomType(), order);
      }
   }

   for (int i = 0; i < ir_mass->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir_mass->IntPoint(i);
      el.CalcShape(ip, shape);

      Trans.SetIntPoint (&ip);
      w = Trans.Weight() * ip.weight;
      if (Qmass)
      {
         w *= Qmass -> Eval(Trans, ip);
      }

      AddMult_a_VVt(w, shape, elmat);
   }
}

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

} // for namespace mfem

