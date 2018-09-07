//
//  GridFunctionOperators.hpp
//  LeastSquaresFEM
//
//  Created by Hamilton, Aidan Porter on 5/1/18.
//  Copyright © 2018 Hamilton, Aidan Porter. All rights reserved.
//

#ifndef GridFunctionOperators_hpp
#define GridFunctionOperators_hpp

#include <stdio.h>
#include "../linalg/vector.hpp"
#include "gridfunc.hpp"
#include "coefficient.hpp"

namespace mfem
{
    
Vector VectorMinusGrad(GridFunction &vec, GridFunction &scalar);

//overload that return output+= VectorMinusGrad computation. Output must be of length of the number of elements of the mesh. It is also assumed that output is initialized to something.
void VectorMinusGrad(GridFunction &vec, GridFunction &scalar, Vector &output);
    
Vector DivMinusFun(GridFunction &vec, FunctionCoefficient &fun);

//overload that return output+= DivMinusFun computation. Output must be of length of the number of elements of the mesh. It is also assumed that output is initialized to something.
void DivMinusFun(GridFunction &vec, FunctionCoefficient &fun,Vector &output);
    
// computes the integral (vec-fun*scalar)^2 for each element.
    Vector VectorMinusVector( GridFunction &vec, GridFunction &scalar,  VectorFunctionCoefficient &fun);
    
   // void VectorMinusVector(const GridFunction &vec,const GridFunction &scalar,const VectorFunctionCoefficient &fun,Vector &output)
}
#endif /* GridFunctionOperators_hpp */
