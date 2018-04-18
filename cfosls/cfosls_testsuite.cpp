using namespace mfem;

namespace mfem
{

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

    //double r = x * x + y * y;
    //b(0) = -y / r;  // -x2
    //b(1) = x / r;   // x1

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
        if ( numsol == 8 && dim == 3 )
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
        if (numsol == 8)
        {
            //std::cout << "The domain must be a cylinder over a circle" << std::endl << std::flush;
            SetTestCoeffs<&uFunCylinder_ex, &uFunCylinder_ex_dt, &uFunCylinder_ex_gradx, &bFunCircle2D_ex, &bFunCircle2Ddiv_ex>();
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

} // for namespace mfem

