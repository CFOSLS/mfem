//
//  GridFunctionOperators.cpp
//  LeastSquaresFEM
//


#include "GridFunctionOperators.hpp"

namespace mfem
{
Vector VectorMinusGrad(GridFunction &vec, GridFunction &scalar) {
    FiniteElementSpace *fes = vec.FESpace();
    int NE = fes->GetNE();
    Vector output(NE), vec_val, grad;
    output = 0.0;
    
    int order;
    
    for (int elnum = 0; elnum <NE; ++elnum){
        const FiniteElement *FE = fes->GetFE(elnum);
        ElementTransformation *tr = fes->GetElementTransformation(elnum);
        order = 2*FE->GetOrder()+1;
        const IntegrationRule &ir = IntRules.Get(FE->GetGeomType(),order);
        
        for (int j = 0; j<ir.GetNPoints(); ++j){
            const IntegrationPoint &ip  = ir.IntPoint(j);
            
            tr->SetIntPoint(&ip);
            
            //both getvectorvalue and getgradient calculate in the physical space, as desired.
            vec.GetVectorValue(elnum, ip, vec_val);
            scalar.GetGradient(*tr, grad);
            grad-=vec_val;
           
            output(elnum) +=   (grad*grad)*ip.weight*tr->Weight();
        }
    }
    return output;
}
    
   
void VectorMinusGrad(GridFunction &vec, GridFunction &scalar, Vector &output) {FiniteElementSpace *fes = vec.FESpace();
    int NE = fes->GetNE();
    Vector vec_val, grad;
    
#ifdef MFEM_DEBUG
    if (output.Size() != NE){
    mfem_error("VectorMinusGrad(), input vector is not of appropriate length");
    }
#endif
    
    int order;
    
    for (int elnum = 0; elnum <NE; ++elnum){
        const FiniteElement *FE = fes->GetFE(elnum);
        ElementTransformation *tr = fes->GetElementTransformation(elnum);
        order = 2*FE->GetOrder()+1;
        const IntegrationRule &ir = IntRules.Get(FE->GetGeomType(),order);
        
        for (int j = 0; j<ir.GetNPoints(); ++j){
            const IntegrationPoint &ip  = ir.IntPoint(j);
            
            tr->SetIntPoint(&ip);
            
            //both getvectorvalue and getgradient calculate in the physical space, as desired.
            vec.GetVectorValue(elnum, ip, vec_val);
            scalar.GetGradient(*tr, grad);
            grad-=vec_val;
            
            output(elnum) +=   (grad*grad)*ip.weight*tr->Weight();
        }
    }
}
    
Vector DivMinusFun(GridFunction &vec, FunctionCoefficient &fun){
    //Assuming RT-type space
    FiniteElementSpace *fes = vec.FESpace();
    int NE = fes->GetNE();
    Vector output(NE);
    output = 0.0;
    
    int order;
    
    for (int elnum = 0; elnum <NE; ++elnum){
        const FiniteElement *FE = fes->GetFE(elnum);
        ElementTransformation *tr = fes->GetElementTransformation(elnum);
        
        order = 2*FE->GetOrder()+1;
        const IntegrationRule &ir = IntRules.Get(FE->GetGeomType(),order);
        
        for (int j = 0; j<ir.GetNPoints(); ++j){
            const IntegrationPoint &ip  = ir.IntPoint(j);
            
            tr->SetIntPoint(&ip);
            output(elnum) += pow(vec.GetDivergence(*tr) - fun.Eval(*tr, ip),2)*ip.weight*tr->Weight();
        }
    }
    return output;
}
    

void DivMinusFun(GridFunction &vec, FunctionCoefficient &fun,Vector &output){
    //Assuming RT-type space
    FiniteElementSpace *fes = vec.FESpace();
    int NE = fes->GetNE();
 
#ifdef MFEM_DEBUG
    if (output.Size() != NE){
        mfem_error("VectorMinusGrad(), input vector is not of appropriate length");
    }
#endif
    
    int order;
    
    for (int elnum = 0; elnum <NE; ++elnum){
        const FiniteElement *FE = fes->GetFE(elnum);
        ElementTransformation *tr = fes->GetElementTransformation(elnum);
        
        order = 2*FE->GetOrder()+1;
        const IntegrationRule &ir = IntRules.Get(FE->GetGeomType(),order);
        
        for (int j = 0; j<ir.GetNPoints(); ++j){
            const IntegrationPoint &ip  = ir.IntPoint(j);
            
            tr->SetIntPoint(&ip);
            output(elnum) += pow(vec.GetDivergence(*tr) - fun.Eval(*tr, ip),2)*ip.weight*tr->Weight();
        }
    }
}

    
       Vector VectorMinusVector(GridFunction &vec, GridFunction &scalar, VectorFunctionCoefficient &fun){
        FiniteElementSpace *fes = vec.FESpace();
        int NE = fes->GetNE();
        Vector output(NE);
        output = 0.0;
           
           Vector vec_val, fun_val;
        
        int order;
        
        for (int elnum = 0; elnum <NE; ++elnum){
            const FiniteElement *FE = fes->GetFE(elnum);
            ElementTransformation *tr = fes->GetElementTransformation(elnum);
            
            order = 2*FE->GetOrder()+1;
            const IntegrationRule &ir = IntRules.Get(FE->GetGeomType(),order);
            
            for (int j = 0; j<ir.GetNPoints(); ++j){
                const IntegrationPoint &ip  = ir.IntPoint(j);
                 tr->SetIntPoint(&ip);
                vec.GetVectorValue(elnum, ip, vec_val);
                fun.Eval(fun_val,*tr,ip);
                fun_val*=scalar.GetValue(elnum,ip);
                vec_val-=fun_val;
                output(elnum) += (vec_val*vec_val)*ip.weight*tr->Weight();
            }
        }
        return output;
    }
}
