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
    if (xt.Size() == 3)
    {
        gradx(0) *= sin (2.0 * M_PI * y);
        gradx(1) = t*t*exp(t) * sin (3.0 * M_PI * x) * 2.0 * M_PI * cos ( 2.0 * M_PI * y);
    }
    if (xt.Size() == 4)
    {
        gradx(0) *= sin (2.0 * M_PI * y) * sin (M_PI * z);
        gradx(1) *= sin (2.0 * M_PI * y) * sin (M_PI * z);
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
    if (xt.Size() == 3)
    {
        gradxt(0) *= sin (2.0 * M_PI * y);
        gradxt(1) = t*t*exp(t) * sin (3.0 * M_PI * x) * 2.0 * M_PI * cos ( 2.0 * M_PI * y);
    }
    if (xt.Size() == 4)
    {
        gradxt(0) *= sin (2.0 * M_PI * y) * sin (M_PI * z);
        gradxt(1) *= sin (2.0 * M_PI * y) * sin (M_PI * z);
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
    if (xt.Size() == 3)
    {
        gradx(0) *= sin (2.0 * M_PI * y);
        gradx(1) = (t*t + 2.0 * t)*exp(t) * sin (3.0 * M_PI * x) * 2.0 * M_PI * cos ( 2.0 * M_PI * y);
    }
    if (xt.Size() == 4)
    {
        gradx(0) *= sin (2.0 * M_PI * y) * sin (M_PI * z);
        gradx(1) *= sin (2.0 * M_PI * y) * sin (M_PI * z);
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
