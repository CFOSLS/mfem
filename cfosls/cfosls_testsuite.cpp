#include "testhead.hpp"


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

void bFunCircleT3D_ex(const Vector& xt, Vector& b )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    b.SetSize(xt.Size());

    b(0) = -y * (t + 1);  // -x2
    b(1) = x * (t + 1);   // x1
    b(2) = 0.0;

    b(xt.Size()-1) = 1.;
    return;
}

double bFunCircleT3Ddiv_ex(const Vector& xt)
{
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

double delta_center_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = 0.0;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size()-1);

    double xcenter = 0.5;
    double ycenter = 0.5;
    double zcenter = 0.0;
    if (xt.Size() == 4)
        zcenter = 0.5;
    double tcenter = 0.0;
    if (xt.Size() >= 3)
        tcenter = 0.5;

    double side_len = 0.1;

    if (fabs(x - xcenter) < side_len && fabs (y - ycenter) < side_len &&
            fabs (z - zcenter) < side_len && fabs(t - tcenter) < side_len)
        return 1.0 / pow(side_len, xt.Size());
    else
        return 0.0;

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

// http://hpfem.org/hermes-doc/hermes-examples/html/src/hermes2d/benchmarks-general/lshape.html
// r^(2/3) sin (2 phi/3 + pi/3), phi = atan(x/y), r(x,y) = sqrt(x^2 + y^2)
double uFunTestLapLshape_ex(const Vector& xt)
{
    MFEM_ASSERT(xt.Size() == 3,"TThis example works only in 3D");
    double x = xt(0);
    double y = xt(1);
    //double t = xt(2);

    double r = sqrt(x * x + y * y);
    double phi = atan2(x,y);

    return pow(r, 2.0/3.0) * sin ( 2.0/3.0 * phi + M_PI / 3.0 );
}

double uFunTestLapLshape_lap(const Vector& xt)
{
    MFEM_ASSERT(xt.Size() == 3,"TThis example works only in 3D");
    return 0.0;
}

void uFunTestLapLshape_grad(const Vector& xt, Vector& grad )
{
    MFEM_ASSERT(xt.Size() == 3,"TThis example works only in 3D");
    double x = xt(0);
    double y = xt(1);
    //double t = xt(2);

    double r = sqrt(x * x + y * y);
    double r_x = x / r;
    double r_y = y / r;

    double phi = atan2(x,y);
    double phi_x = y / (r * r);
    double phi_y = - x / (r * r);

    double u_r = 2.0/3.0 * uFunTestLapLshape_ex(xt) / r;
    double u_phi = 2.0/3.0 * pow(r, 2.0/3.0) * cos ( 2.0/3.0 * phi + M_PI / 3.0 );

    grad.SetSize(xt.Size());

    grad(0) = u_r * r_x + u_phi * phi_x;
    grad(1) = u_r * r_y + u_phi * phi_y;
    grad(2) =  0.0;

    return;
}

// Fichera Corner with Vertex Singularity, check out at
// https://math.nist.gov/cgi-bin/amr-display-problem.cgi
// If input vector is (d+1) dimensional, Sobolev regularity is
// (1.5 + q), when d = 3
// (1.0 + q), when d = 2
double uFunTestFichera_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = 0.0;
    if (xt.Size() == 4)
        z = xt(2);
    //double t = xt(xt.Size()-1);

    double r = sqrt (x * x + y * y + z * z);

    return pow(r,FICHERA_Q);
}

double uFunTestFichera_lap(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = 0.0;
    if (xt.Size() == 4)
        z = xt(2);
    //double t = xt(xt.Size()-1);

    double r = sqrt (x * x + y * y + z * z);

    return (FICHERA_Q) * (FICHERA_Q + 1) * pow(r, FICHERA_Q - 2.0);
}

void uFunTestFichera_grad(const Vector& xt, Vector& grad )
{
    double x = xt(0);
    double y = xt(1);
    double z = 0.0;
    if (xt.Size() == 4)
        z = xt(2);
    //double t = xt(xt.Size()-1);

    double r = sqrt (x * x + y * y + z * z);

    grad.SetSize(xt.Size());

    grad(0) = (FICHERA_Q) * x * pow(r, FICHERA_Q - 2.0);
    grad(1) = (FICHERA_Q) * y * pow(r, FICHERA_Q - 2.0);
    if (xt.Size() == 4)
        grad(2) = (FICHERA_Q) * z * pow(r, FICHERA_Q - 2.0);

    grad(xt.Size() - 1) = 0.0;

    return;
}

// much alike uFunTestFicheraT, but smoothly depending on t
// If input vector is (d+1) dimensional, Sobolev regularity is
// (1.5 + q), when d = 3
// (1.0 + q), when d = 2
// not tested yet
double uFunTestFicheraT_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = 0.0;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size()-1);

    double r = sqrt (x * x + y * y + z * z);

    return pow(r,FICHERA_Q) * t * (1.0 - t);
}

double uFunTestFicheraT_lap(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = 0.0;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size()-1);

    double r = sqrt (x * x + y * y + z * z);

    return (FICHERA_Q) * (FICHERA_Q + 1) * pow(r, FICHERA_Q - 2.0) * t * (1.0 - t) +
            (-2.0) * pow(r,FICHERA_Q);
}

void uFunTestFicheraT_grad(const Vector& xt, Vector& grad )
{
    double x = xt(0);
    double y = xt(1);
    double z = 0.0;
    if (xt.Size() == 4)
        z = xt(2);
    double t = xt(xt.Size()-1);

    double r = sqrt (x * x + y * y + z * z);

    grad.SetSize(xt.Size());

    grad(0) = (FICHERA_Q) * x * pow(r, FICHERA_Q - 2.0) * t * (1.0 - t);
    grad(1) = (FICHERA_Q) * y * pow(r, FICHERA_Q - 2.0) * t * (1.0 - t);
    if (xt.Size() == 4)
        grad(2) = (FICHERA_Q) * z * pow(r, FICHERA_Q - 2.0) * t * (1.0 - t);

    grad(xt.Size() - 1) = pow(r,FICHERA_Q) * (1.0 - 2.0 * t);

    return;
}


FOSLS_test::FOSLS_test(int dimension, int nfunc_coefficients, int nvec_coefficients, int nmat_coefficients)
    : dim(dimension), nfunc_coeffs(nfunc_coefficients), nvec_coeffs(nvec_coefficients), nmat_coeffs(nmat_coefficients)
{
    func_coeffs.SetSize(nfunc_coeffs);
    for (int i = 0; i < func_coeffs.Size(); ++i)
        func_coeffs[i] = NULL;

    vec_coeffs.SetSize(nvec_coeffs);
    for (int i = 0; i < vec_coeffs.Size(); ++i)
        vec_coeffs[i] = NULL;

    mat_coeffs.SetSize(nmat_coeffs);
    for (int i = 0; i < mat_coeffs.Size(); ++i)
        mat_coeffs[i] = NULL;
}

FOSLS_test::~FOSLS_test()
{
    for (int i = 0; i < func_coeffs.Size(); ++i)
        if (func_coeffs[i])
            delete func_coeffs[i];

    vec_coeffs.SetSize(nvec_coeffs);
    for (int i = 0; i < vec_coeffs.Size(); ++i)
        if (vec_coeffs[i])
            delete vec_coeffs[i];

    mat_coeffs.SetSize(nmat_coeffs);
    for (int i = 0; i < mat_coeffs.Size(); ++i)
        if (mat_coeffs[i])
            delete mat_coeffs[i];

}


Hyper_test::Hyper_test(int dimension, int num_solution)
    : FOSLS_test(dimension, 3, 4, 2), numsol(num_solution)
{
    Init();
}

void Hyper_test::Init()
{
    if ( CheckTestConfig() == false )
    {
        MFEM_ABORT("Inconsistent dim and numsol \n");
    }
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
        if (numsol == 8) // a Gaussian hill rotating in (x,y)-plane around the origin with a constant velocity
        {
            SetTestCoeffs<&uFunCylinder_ex, &uFunCylinder_ex_dt, &uFunCylinder_ex_gradx, &bFunCircle2D_ex, &bFunCircle2Ddiv_ex>();
        }
        if (numsol == 88) // a Gaussian hill rotating in (x,y)-plane around the origin with
                          // a time-increasing velocity
        {
            SetTestCoeffs<&uFunCylinder4D_ex, &uFunCylinder4D_ex_dt, &uFunCylinder4D_ex_gradx,
                    &bFunCircleT3D_ex, &bFunCircleT3Ddiv_ex>();
        }
    } // end of setting test coefficients in correct case
}

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx),  \
         void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
void Hyper_test::SetTestCoeffs ()
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

bool Hyper_test::CheckTestConfig()
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
        if ( numsol == 88 && dim == 4 )
            return true;
        return false;
    }
    else
        return false;
}

Parab_test::Parab_test(int dimension, int num_solution)
    : FOSLS_test(dimension, 2, 1, 0), numsol(num_solution)
{
    Init();
}

void Parab_test::Init()
{
    if ( CheckTestConfig() == false )
    {
        MFEM_ABORT("Inconsistent dim and numsol \n");
    }
    else
    {
        if (numsol == -3) // 3D test for the paper
        {
            SetTestCoeffs<&uFunTest_ex, &uFunTest_ex_dt, &uFunTest_ex_laplace, &uFunTest_ex_gradx>();
        }
        if (numsol == -33) // 3D test for the paper
        {
            SetTestCoeffs<&uFunTestNh_ex, &uFunTestNh_ex_dt, &uFunTestNh_ex_laplace, &uFunTestNh_ex_gradx>();
        }
        if (numsol == -4) // 4D test for the paper
        {
            SetTestCoeffs<&uFunTest_ex, &uFunTest_ex_dt, &uFunTest_ex_laplace, &uFunTest_ex_gradx>();
        }
        if (numsol == -44) // 4D test for the paper
        {
            SetTestCoeffs<&uFunTestNh_ex, &uFunTestNh_ex_dt, &uFunTestNh_ex_laplace, &uFunTestNh_ex_gradx>();
        }

        if (numsol == 0)
        {
            //std::cout << "The domain should be either a unit rectangle or cube" << std::endl << std::flush;
            SetTestCoeffs<&uFun_ex_parab, &uFun_ex_parab_dt, &uFun_ex_parab_laplace, &uFun_ex_parab_gradx>();
        }
        if (numsol == 1)
        {
            //std::cout << "The domain should be either a unit rectangle or cube" << std::endl << std::flush;
            SetTestCoeffs<&uFun1_ex_parab, &uFun1_ex_parab_dt, &uFun1_ex_parab_laplace, &uFun1_ex_parab_gradx>();
        }
        if (numsol == 2)
        {
            SetTestCoeffs<&uFun2_ex_parab, &uFun2_ex_parab_dt, &uFun2_ex_parab_laplace, &uFun2_ex_parab_gradx>();
        }
        if (numsol == 3)
        {
            SetTestCoeffs<&uFun3_ex_parab, &uFun3_ex_parab_dt, &uFun3_ex_parab_laplace, &uFun3_ex_parab_gradx>();
        }
        if (numsol == -34)
        {
            SetTestCoeffs<&uFunTest_ex, &uFunTest_ex_dt, &uFunTest_ex_laplace, &uFunTest_ex_gradx>();
        }

    } // end of setting test coefficients in correct case
}

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt),
         double (*Slaplace)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx)> \
void Parab_test::SetTestCoeffs ( )
{
    SetScalarSFun(S);
    SetSigmaVec<S,Sgradxvec>();
    SetDivSigma<S, dSdt, Slaplace>();
    return;
}

bool Parab_test::CheckTestConfig()
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
        if (numsol == 0 || numsol == 1)
            return true;
        if (numsol == 2 && dim == 4)
            return true;
        if (numsol == 3 && dim == 3)
            return true;
        if (numsol == -34 && (dim == 3 || dim == 4))
            return true;
        return false;
    }
    else
        return false;
}

Wave_test::Wave_test(int dimension, int num_solution)
    : FOSLS_test(dimension, 2, 1, 0), numsol(num_solution)
{
    Init();
}

void Wave_test::Init()
{
    if ( CheckTestConfig() == false )
    {
        MFEM_ABORT("Inconsistent dim and numsol \n");
    }
    else
    {
        if (numsol == -3) // 3D test for the paper
        {
            SetTestCoeffs<&uFunTest_ex, &uFunTest_ex_dt, &uFunTest_ex_dt2, &uFunTest_ex_laplace, &uFunTest_ex_dtlaplace,
                    &uFunTest_ex_gradx, &uFunTest_ex_dtgradx>();
        }
        if (numsol == -33) // 3D test for the paper
        {
            SetTestCoeffs<&uFunTestNh_ex, &uFunTestNh_ex_dt, &uFunTestNh_ex_dt2, &uFunTestNh_ex_laplace, &uFunTestNh_ex_dtlaplace,
                    &uFunTestNh_ex_gradx, &uFunTestNh_ex_dtgradx>();
        }
        if (numsol == -4) // 4D test for the paper
        {
            SetTestCoeffs<&uFunTest_ex, &uFunTest_ex_dt, &uFunTest_ex_dt2, &uFunTest_ex_laplace, &uFunTest_ex_dtlaplace,
                    &uFunTest_ex_gradx, &uFunTest_ex_dtgradx>();
        }
        if (numsol == -44) // 4D test for the paper
        {
            SetTestCoeffs<&uFunTestNh_ex, &uFunTestNh_ex_dt, &uFunTestNh_ex_dt2, &uFunTestNh_ex_laplace, &uFunTestNh_ex_dtlaplace,
                    &uFunTestNh_ex_gradx, &uFunTestNh_ex_dtgradx>();
        }

        if (numsol == -34)
        {
            SetTestCoeffs<&uFunTest_ex, &uFunTest_ex_dt, &uFunTest_ex_dt2, &uFunTest_ex_laplace, &uFunTest_ex_dtlaplace,
                    &uFunTest_ex_gradx, &uFunTest_ex_dtgradx>();
        }
        if (numsol == 0)
        {
            SetTestCoeffs<&uFun_ex_wave, &uFun_ex_wave_dt, &uFun_ex_wave_dt2, &uFun_ex_wave_laplace,
                    &uFun_ex_wave_dtlaplace, &uFun_ex_wave_gradx, &uFun_ex_wave_dtgradx>();
        }
        if (numsol == 1)
        {
            SetTestCoeffs<&uFun1_ex_wave, &uFun1_ex_wave_dt, &uFun1_ex_wave_dt2, &uFun1_ex_wave_laplace, &uFun1_ex_wave_dtlaplace,
                    &uFun1_ex_wave_gradx, &uFun1_ex_wave_dtgradx>();
        }
        if (numsol == 2)
        {
            SetTestCoeffs<&uFun2_ex_wave, &uFun2_ex_wave_dt, &uFun2_ex_wave_dt2, &uFun2_ex_wave_laplace, &uFun2_ex_wave_dtlaplace,
                    &uFun2_ex_wave_gradx, &uFun2_ex_wave_dtgradx>();
        }
        if (numsol == 3)
        {
            SetTestCoeffs<&uFun3_ex_wave, &uFun3_ex_wave_dt, &uFun3_ex_wave_dt2, &uFun3_ex_wave_laplace,
                    &uFun3_ex_wave_dtlaplace, &uFun3_ex_wave_gradx, &uFun3_ex_wave_dtgradx>();
        }
        if (numsol == 4)
        {
            SetTestCoeffs<&uFun4_ex_wave, &uFun4_ex_wave_dt, &uFun4_ex_wave_dt2, &uFun4_ex_wave_laplace,
                    &uFun4_ex_wave_dtlaplace, &uFun4_ex_wave_gradx, &uFun4_ex_wave_dtgradx>();
        }
        if (numsol == 5)
        {
            SetTestCoeffs<&uFun5_ex_wave, &uFun5_ex_wave_dt, &uFun5_ex_wave_dt2, &uFun5_ex_wave_laplace,
                    &uFun5_ex_wave_dtlaplace, &uFun5_ex_wave_gradx, &uFun5_ex_wave_dtgradx>();
        }
    } // end of setting test coefficients in correct case
}

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), double (*d2Sdt2)(const Vector & xt),\
         double (*Slaplace)(const Vector & xt), double (*dSdtlaplace)(const Vector & xt), \
         void(*Sgradxvec)(const Vector & x, Vector & gradx), void (*dSdtgradxvec)(const Vector&, Vector& ) > \
void Wave_test::SetTestCoeffs()
{
    SetScalarSFun(S);
    SetSigmaVec<dSdt,Sgradxvec>();
    SetDivSigma<S, d2Sdt2, Slaplace>();
    return;
}

bool Wave_test::CheckTestConfig()
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

        if (numsol == 0 || numsol == 1)
            return true;
        if (numsol == 2 && dim == 4)
            return true;
        if (numsol == 3 && dim == 3)
            return true;
        if (numsol == 4 && dim == 3)
            return true;
        if (numsol == 5 && dim == 3)
            return true;
        if (numsol == -34 && (dim == 3 || dim == 4))
            return true;
        return false;
    }
    else
        return false;
}

Laplace_test::Laplace_test(int dimension, int num_solution)
    : FOSLS_test(dimension, 2, 1, 0), numsol(num_solution)
{
    Init();
}

void Laplace_test::Init()
{
    if ( CheckTestConfig() == false )
    {
        MFEM_ABORT("Inconsistent dim and numsol \n");
    }
    else
    {
        if (numsol == -3 || numsol == -4 || numsol == -34)
            SetTestCoeffs<&uFunTestLap_ex, &uFunTestLap_grad, &uFunTestLap_lap>();
        if (numsol == 11 && dim == 3)
            SetTestCoeffs<&uFunTestLapLshape_ex, &uFunTestLapLshape_grad, &uFunTestLapLshape_lap>();
        if (numsol == 111 && dim == 4)
            SetTestCoeffs<&uFunTestLap_ex, &uFunTestLap_grad, &uFunTestLap_lap>();
        if (numsol == 1111) // corner singulairt r^q, r in space only, not tested yet
            SetTestCoeffs<&uFunTestFichera_ex, &uFunTestFichera_grad, &uFunTestFichera_lap>();
        if (numsol == 1112) // corner singulairt r^q * t(1-t), r in space only, not tested yet
            SetTestCoeffs<&uFunTestFicheraT_ex, &uFunTestFicheraT_grad, &uFunTestFicheraT_lap>();
        if (numsol == -9)
            SetTestCoeffs<&zero_ex, &zerovec_ex, &delta_center_ex>();
    } // end of setting test coefficients in correct case
}

template<double (*S)(const Vector & xt), void(*Sfullgrad)(const Vector & xt, Vector & gradx),
         double (*Slaplace)(const Vector & xt)> \
void Laplace_test::SetTestCoeffs ()
{
    SetScalarSFun(S);
    SetSigmaVec<Sfullgrad>();
    SetdivSigma<Slaplace>();
    return;
}


bool Laplace_test::CheckTestConfig()
{
    if (dim == 4 || dim == 3)
    {
        if (numsol == -3 && dim == 3)
            return true;
        if (numsol == -4 && dim == 4)
            return true;
        if (numsol == -34 && (dim == 3 || dim == 4))
            return true;
        if (numsol == -9 && dim == 3)
            return true;
        if (numsol == 11 && dim == 3)
            return true;
        if (numsol == 111 && dim == 4)
            return true;
        if (numsol == 1111 && (dim == 3 || dim == 4))
            return true;
        if (numsol == 1112 && (dim == 3 || dim == 4))
            return true;
        return false;
    }
    else
        return false;
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
void sigmaTemplate_hyper(const Vector& xt, Vector& sigma)
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
double divsigmaTemplate_hyper(const Vector& xt)
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

    if (fabs(res) > 1.0e-10)
        std::cout << "error if solution is the cylindric test w/o dissipation \n";

    return res;
}

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), void(*Sgradxvec)(const Vector & x, Vector & gradx), \
         void(*bvec)(const Vector & x, Vector & vec), double (*divbfunc)(const Vector & xt)> \
void bfTemplate(const Vector& xt, Vector& bf)
{
    bf.SetSize(xt.Size());

    Vector b;
    bvec(xt,b);

    double f = divsigmaTemplate_hyper<S, dSdt, Sgradxvec, bvec, divbfunc>(xt);

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

    double divsigma = divsigmaTemplate_hyper<S, dSdt, Sgradxvec, bvec, divbfunc>(xt);

    for (int i = 0; i < bdivsigma.Size(); ++i)
        bdivsigma(i) = divsigma * b(i);
}

template <double (*S)(const Vector & xt), void (*bvecfunc)(const Vector&, Vector& ),
          void (*opdivfreevec)(const Vector&, Vector& )> \
void minKsigmahatTemplate(const Vector& xt, Vector& minKsigmahatv)
{
    minKsigmahatv.SetSize(xt.Size());

    Vector b;
    bvecfunc(xt, b);

    Vector sigmahatv;
    sigmahatTemplate<S, bvecfunc, opdivfreevec>(xt, sigmahatv);

    DenseMatrix Ktilda;
    KtildaTemplate<bvecfunc>(xt, Ktilda);

    Ktilda.Mult(sigmahatv, minKsigmahatv);

    minKsigmahatv *= -1.0;
    return;
}

template <double (*S)(const Vector & xt), void (*bvecfunc)(const Vector&, Vector& ),
          void (*opdivfreevec)(const Vector&, Vector& )> \
double bsigmahatTemplate(const Vector& xt)
{
    Vector b;
    bvecfunc(xt, b);

    Vector sigmahatv;
    sigmahatTemplate<S, bvecfunc, opdivfreevec>(xt, sigmahatv);

    return b * sigmahatv;
}

template <double (*S)(const Vector & xt), void (*bvecfunc)(const Vector&, Vector& ),
          void (*opdivfreevec)(const Vector&, Vector& )> \
void sigmahatTemplate(const Vector& xt, Vector& sigmahatv)
{
    sigmahatv.SetSize(xt.Size());

    Vector b;
    bvecfunc(xt, b);

    Vector sigma(xt.Size());
    sigma(xt.Size()-1) = S(xt);
    for (int i = 0; i < xt.Size()-1; i++)
        sigma(i) = b(i) * sigma(xt.Size()-1);

    Vector opdivfree;
    opdivfreevec(xt, opdivfree);

    sigmahatv = 0.0;
    sigmahatv -= opdivfree;
#ifndef ONLY_DIVFREEPART
    sigmahatv += sigma;
#endif
    return;
}

template <double (*S)(const Vector & xt), void (*bvecfunc)(const Vector&, Vector& ),
          void (*opdivfreevec)(const Vector&, Vector& )> \
void minsigmahatTemplate(const Vector& xt, Vector& minsigmahatv)
{
    minsigmahatv.SetSize(xt.Size());
    sigmahatTemplate<S, bvecfunc, opdivfreevec>(xt, minsigmahatv);
    minsigmahatv *= -1;

    return;
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


template <double (*S)(const Vector&), void (*Sgradxvec)(const Vector&, Vector& )> \
void sigmaTemplate_parab(const Vector& xt, Vector& sigma)
{
    sigma.SetSize(xt.Size());

    Vector gradS;
    Sgradxvec(xt,gradS);

    sigma(xt.Size()-1) = S(xt);
    for (int i = 0; i < xt.Size()-1; i++)
        sigma(i) = - gradS(i);

    return;
}

template<double (*S)(const Vector & xt), double (*dSdt)(const Vector & xt), double (*Slaplace)(const Vector & xt) > \
double divsigmaTemplate_parab(const Vector& xt)
{
    Vector xt0(xt.Size());
    xt0 = xt;
    xt0 (xt0.Size() - 1) = 0;

    return dSdt(xt) - Slaplace(xt);
}

template <double (*dSdt)(const Vector&), void (*Sgradxvec)(const Vector&, Vector& )> \
void sigmaTemplate_wave(const Vector& xt, Vector& sigma)
{
    sigma.SetSize(xt.Size());

    Vector gradS;
    Sgradxvec(xt,gradS);

    sigma(xt.Size()-1) = dSdt(xt);
    for (int i = 0; i < xt.Size()-1; i++)
        sigma(i) = - gradS(i);

    return;
}

template<double (*S)(const Vector & xt), double (*d2Sdt2)(const Vector & xt), double (*Slaplace)(const Vector & xt) > \
double divsigmaTemplate_wave(const Vector& xt)
{
    Vector xt0(xt.Size());
    xt0 = xt;
    xt0 (xt0.Size() - 1) = 0;

    return d2Sdt2(xt) - Slaplace(xt);
}

template <void (*Sfullgrad)(const Vector&, Vector& )> \
void sigmaTemplate_lapl(const Vector& xt, Vector& sigma)
{
    sigma.SetSize(xt.Size());
    Sfullgrad(xt, sigma);
    sigma *= -1;
    return;
}


template<double (*Slaplace)(const Vector & xt)> \
double divsigmaTemplate_lapl(const Vector& xt)
{
    return (-1) * Slaplace(xt);
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

double GaussianHill3D(const Vector&xvec)
{
    double x = xvec(0);
    double y = xvec(1);
    double z = xvec(2);

    return exp(-100.0 * ((x - 0.5) * (x - 0.5) + y * y + (z - 0.25) * (z - 0.25)));
}

double GaussianHill3D_dphi(const Vector&xvec)
{
    double x = xvec(0);
    double y = xvec(1);
    double z = xvec(2);

    double r = sqrt(x*x + y*y + z*z);
    double phi = atan2(y,x);
    double teta = acos(z/r);

    // d( (r sin(teta) cos(phi) - 0.5)^2 ) / dphi
    double term1 = 2.0 * (r * sin(teta) * cos(phi) - 0.5) * r * sin(teta) * (-sin(phi));

    // d( (r sin(teta) sin(phi))^2 ) / dphi
    double term2 = 2.0 * r * sin(teta) * sin(phi) * r * sin(teta) * cos(phi);

    return -100.0 * (term1  + term2) * GaussianHill3D(xvec);
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

    double alt = exp( -100.0 * ( ( x*cos(t) + y*sin(t) -0.5)*( x*cos(t) + y*sin(t) -0.5)
                                 + (y * cos(t) - x * sin(t) )*(y * cos(t) - x * sin(t) ) ));

    if (fabs(GaussianHill(xvec) - alt) > 1.0e-13 )
        std::cout << "Error \n";

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

double uFunCylinder4D_ex(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double r = sqrt(x*x + y*y + z*z);
    double phi = atan2(y,x);
    double teta = acos(z/r);

    double t = xt(xt.Size()-1);
    Vector xvec(3);

    // now each point was rotating with the speed (t+1) around the circle,
    // i.e. we need to go back by \int_0^t {(t+1)dt}
    xvec(0) = r * cos (phi - t - 0.5 * t * t) * sin(teta);
    xvec(1) = r * sin (phi - t - 0.5 * t * t) * sin(teta);
    xvec(2) = z;
    //xvec(0) = r * cos (teta - t);
    //xvec(1) = r * sin (teta - t);

    return GaussianHill3D(xvec);
}

double uFunCylinder4D_ex_dt(const Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double r = sqrt(x*x + y*y + z*z);
    double phi = atan2(y,x);
    double teta = acos(z/r);

    double t = xt(xt.Size()-1);
    Vector xvec(3);
    // now each point was rotating with the speed (t+1) around the circle,
    // i.e. we need to go back by \int_0^t {(t+1)dt}
    xvec(0) = r * cos (phi - t - 0.5 * t * t) * sin(teta);
    xvec(1) = r * sin (phi - t - 0.5 * t * t) * sin(teta);
    xvec(2) = z;

    return GaussianHill3D_dphi(xvec) * (- 1.0 - t);
}

void uFunCylinder4D_ex_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double r = sqrt(x*x + y*y + z*z);
    //double phi = atan2(y,x);
    //double teta = acos(z/r);
    double t = xt(xt.Size()-1);

    // We compute gradient at point (x,y,z,t) with the following steps:

    // 1. Computing (x0,y0) = Q^tr * (x,y) = initial particle location
    // consider the fact the we know the rotation speed to be (t+1) for any t
    // the arc_length covered by the particle is then \int_0^t {(t+1)dt}
    double arc_length = t + 0.5 * t * t;
    double x0 = x * cos(arc_length) + y * sin(arc_length);
    double y0 = x * (-sin(arc_length)) + y * cos(arc_length);
    double z0 = z;

    // 2. Computing gradient of the solution for t = 0 at the initial particle location
    Vector r0vec(4);
    r0vec(0) = x0;
    r0vec(1) = y0;
    r0vec(2) = z0;
    r0vec(3) = 0;

    // tempvec = grad u(x0,y0,z0) at t = 0
    Vector tempvec(3);
    tempvec(0) = -100.0 * 2.0 * (x0 - 0.5) * uFunCylinder4D_ex(r0vec);
    tempvec(1) = -100.0 * 2.0 * y0 * uFunCylinder4D_ex(r0vec);
    tempvec(2) = -100.0 * 2.0 * (z0 - 0.25) * uFunCylinder4D_ex(r0vec);

    // 3. Applying the inverse rotation transform to the gradient,
    // which gives the gradient at the current point (x,y,z,t)
    //gradx = Q * tempvec
    gradx.SetSize(xt.Size() - 1);

    gradx(0) = tempvec(0) * cos(arc_length) + tempvec(1) * (-sin(arc_length));
    gradx(1) = tempvec(0) * sin(arc_length) + tempvec(1) * cos(arc_length);
    gradx(2) = tempvec(2);
}

void zerovecx_ex(const Vector& xt, Vector& zerovecx )
{
    zerovecx.SetSize(xt.Size() - 1);
    zerovecx = 0.0;
}

void zerovec_ex(const Vector& xt, Vector& vecvalue)
{
    vecvalue.SetSize(xt.Size());
    vecvalue = 0.0;
    return;
}

void zerovecMat4D_ex(const Vector& xt, Vector& vecvalue)
{
    vecvalue.SetSize(6);
    vecvalue = 0.0;
    return;
}

double zero_ex(const Vector& xt)
{
    return 0.0;
}

////////////////
void hcurlFun3D_ex(const Vector& xt, Vector& vecvalue)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    double freq = 1.0;
    double kappa = freq * M_PI;

    vecvalue.SetSize(xt.Size());

    //vecvalue(0) = -y * (1 - t);
    //vecvalue(1) = x * (1 - t);
    //vecvalue(2) = 0;
    //vecvalue(0) = x * (1 - x);
    //vecvalue(1) = y * (1 - y);
    //vecvalue(2) = t * (1 - t);

    // Martin's function
    vecvalue(0) = sin(kappa * y);
    vecvalue(1) = sin(kappa * t);
    vecvalue(2) = sin(kappa * x);

    return;
}

void curlhcurlFun3D_ex(const Vector& xt, Vector& vecvalue)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    double freq = 1.0;
    double kappa = freq * M_PI;

    vecvalue.SetSize(xt.Size());

    //vecvalue(0) = 0.0;
    //vecvalue(1) = 0.0;
    //vecvalue(2) = -2.0 * (1 - t);

    // Martin's function's curl
    vecvalue(0) = - kappa * cos(kappa * t);
    vecvalue(1) = - kappa * cos(kappa * x);
    vecvalue(2) = - kappa * cos(kappa * y);

    return;
}

////////////////
void DivmatFun4D_ex(const Vector& xt, Vector& vecvalue)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    double freq = 1.0;
    double kappa = freq * M_PI;

    vecvalue.SetSize(xt.Size());

    // 4D counterpart of the Martin's 3D function
    //std::cout << "Error: DivmatFun4D_ex is incorrect \n";
    vecvalue(0) = sin(kappa * y);
    vecvalue(1) = sin(kappa * z);
    vecvalue(2) = sin(kappa * t);
    vecvalue(3) = sin(kappa * x);

    return;
}

void DivmatDivmatFun4D_ex(const Vector& xt, Vector& vecvalue)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(xt.Size()-1);

    double freq = 1.0;
    double kappa = freq * M_PI;

    vecvalue.SetSize(xt.Size());

    //vecvalue(0) = 0.0;
    //vecvalue(1) = 0.0;
    //vecvalue(2) = -2.0 * (1 - t);

    // Divmat of the 4D counterpart of the Martin's 3D function
    std::cout << "Error: DivmatDivmatFun4D_ex is incorrect \n";
    vecvalue(0) = - kappa * cos(kappa * t);
    vecvalue(1) = - kappa * cos(kappa * x);
    vecvalue(2) = - kappa * cos(kappa * y);
    vecvalue(3) = z;

    return;
}

void hcurlFun3D_2_ex(const Vector& xt, Vector& vecvalue)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    vecvalue.SetSize(xt.Size());

    //
    vecvalue(0) = 100.0 * x * x * (1-x) * (1-x) * y * y * (1-y) * (1-y) * t * t * (1-t) * (1-t);
    vecvalue(1) = 0.0;
    vecvalue(2) = 0.0;

    return;
}

void curlhcurlFun3D_2_ex(const Vector& xt, Vector& vecvalue)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(xt.Size()-1);

    vecvalue.SetSize(xt.Size());

    //
    vecvalue(0) = 0.0;
    vecvalue(1) = 100.0 * ( 2.0) * t * (1-t) * (1.-2.*t) * x * x * (1-x) * (1-x) * y * y * (1-y) * (1-y);
    vecvalue(2) = 100.0 * (-2.0) * y * (1-y) * (1.-2.*y) * x * x * (1-x) * (1-x) * t * t * (1-t) * (1-t);

    return;
}

/////// from parabolic example
double uFun_ex_parab(const Vector & xt)
{
    const double PI = 3.141592653589793;
    double xi(xt(0));
    double yi(xt(1));
    double zi(0.0);
    double vi(0.0);

    if (xt.Size() == 3)
    {
        zi = xt(2);
        return sin(PI*xi)*sin(PI*yi)*zi;
    }
    if (xt.Size() == 4)
    {
        zi = xt(2);
        vi = xt(3);
        //cout << "sol for 4D" << endl;
        return sin(PI*xi)*sin(PI*yi)*sin(PI*zi)*vi;
    }

    return 0.0;
}


double uFun_ex_parab_dt(const Vector & xt)
{
    const double PI = 3.141592653589793;
    double xi(xt(0));
    double yi(xt(1));
    double zi(0.0);

    if (xt.Size() == 3)
        return sin(PI*xi)*sin(PI*yi);
    if (xt.Size() == 4)
    {
        zi = xt(2);
        return sin(PI*xi)*sin(PI*yi)*sin(PI*zi);
    }

    return 0.0;
}

double uFun_ex_parab_laplace(const Vector & xt)
{
    const double PI = 3.141592653589793;
    return (-(xt.Size()-1) * PI * PI) *uFun_ex_parab(xt);
}

void uFun_ex_parab_gradx(const Vector& xt, Vector& gradx )
{
    const double PI = 3.141592653589793;

    double x = xt(0);
    double y = xt(1);
    double z(0.0);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    if (xt.Size() == 3)
    {
        gradx(0) = t * PI * cos (PI * x) * sin (PI * y);
        gradx(1) = t * PI * sin (PI * x) * cos (PI * y);
    }
    if (xt.Size() == 4)
    {
        z = xt(2);
        gradx(0) = t * PI * cos (PI * x) * sin (PI * y) * sin (PI * z);
        gradx(1) = t * PI * sin (PI * x) * cos (PI * y) * sin (PI * z);
        gradx(2) = t * PI * sin (PI * x) * sin (PI * y) * cos (PI * z);
    }

}


double fFun(const Vector & x)
{
    const double PI = 3.141592653589793;
    double xi(x(0));
    double yi(x(1));
    double zi(0.0);
    double vi(0.0);
    if (x.Size() == 3)
    {
     zi = x(2);
       return 2*PI*PI*sin(PI*xi)*sin(PI*yi)*zi+sin(PI*xi)*sin(PI*yi);
    }

    if (x.Size() == 4)
    {
     zi = x(2);
         vi = x(3);
         //cout << "rhand for 4D" << endl;
       return 3*PI*PI*sin(PI*xi)*sin(PI*yi)*sin(PI*zi)*vi + sin(PI*xi)*sin(PI*yi)*sin(PI*zi);
    }

    return 0.0;
}

void sigmaFun_ex(const Vector & x, Vector & u)
{
    const double PI = 3.141592653589793;
    double xi(x(0));
    double yi(x(1));
    double zi(0.0);
    double vi(0.0);
    if (x.Size() == 3)
    {
        zi = x(2);
        u(0) = - PI * cos (PI * xi) * sin (PI * yi) * zi;
        u(1) = - PI * cos (PI * yi) * sin (PI * xi) * zi;
        u(2) = uFun_ex_parab(x);
        return;
    }

    if (x.Size() == 4)
    {
        zi = x(2);
        vi = x(3);
        u(0) = - PI * cos (PI * xi) * sin (PI * yi) * sin(PI * zi) * vi;
        u(1) = - sin (PI * xi) * PI * cos (PI * yi) * sin(PI * zi) * vi;
        u(2) = - sin (PI * xi) * sin(PI * yi) * PI * cos (PI * zi) * vi;
        u(3) = uFun_ex_parab(x);
        return;
    }

    if (x.Size() == 2)
    {
        u(0) =  exp(-PI*PI*yi)*PI*cos(PI*xi);
        u(1) = -sin(PI*xi)*exp(-1*PI*PI*yi);
        return;
    }

    return;
}



double uFun1_ex_parab(const Vector & xt)
{
    double tmp = (xt.Size() == 4) ? sin(M_PI*xt(2)) : 1.0;
    return exp(-xt(xt.Size()-1))*sin(M_PI*xt(0))*sin(M_PI*xt(1))*tmp;
}

double uFun1_ex_parab_dt(const Vector & xt)
{
    return - uFun1_ex_parab(xt);
}

double uFun1_ex_parab_laplace(const Vector & xt)
{
    return (- (xt.Size() - 1) * M_PI * M_PI ) * uFun1_ex_parab(xt);
}

void uFun1_ex_parab_gradx(const Vector& xt, Vector& gradx )
{
    const double PI = 3.141592653589793;

    double x = xt(0);
    double y = xt(1);
    double z(0.0);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    if (xt.Size() == 3)
    {
        gradx(0) = exp(-t) * PI * cos (PI * x) * sin (PI * y);
        gradx(1) = exp(-t) * PI * sin (PI * x) * cos (PI * y);
    }
    if (xt.Size() == 4)
    {
        z = xt(2);
        gradx(0) = exp(-t) * PI * cos (PI * x) * sin (PI * y) * sin (PI * z);
        gradx(1) = exp(-t) * PI * sin (PI * x) * cos (PI * y) * sin (PI * z);
        gradx(2) = exp(-t) * PI * sin (PI * x) * sin (PI * y) * cos (PI * z);
    }

}

double fFun1(const Vector & x)
{
    return ( (x.Size()-1)*M_PI*M_PI - 1. ) * uFun1_ex_parab(x);
}

void sigmaFun1_ex(const Vector & x, Vector & sigma)
{
    sigma.SetSize(x.Size());
    sigma(0) = -M_PI*exp(-x(x.Size()-1))*cos(M_PI*x(0))*sin(M_PI*x(1));
    sigma(1) = -M_PI*exp(-x(x.Size()-1))*sin(M_PI*x(0))*cos(M_PI*x(1));
    if (x.Size() == 4)
    {
        sigma(0) *= sin(M_PI*x(2));
        sigma(1) *= sin(M_PI*x(2));
        sigma(2) = -M_PI*exp(-x(x.Size()-1))*sin(M_PI*x(0))
                *sin(M_PI*x(1))*cos(M_PI*x(2));
    }
    sigma(x.Size()-1) = uFun1_ex_parab(x);

    return;
}

double uFun2_ex_parab(const Vector & xt)
{
    if (xt.Size() != 4)
        cout << "Error, this is only 4-d solution" << endl;
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(3);

    return exp(-t) * x * sin (M_PI * x) * (1 + y) * sin (M_PI * y) * (2 - z) * sin (M_PI * z);
}

double uFun2_ex_parab_dt(const Vector & xt)
{
    return - uFun2_ex_parab(xt);
}

double uFun2_ex_parab_laplace(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(3);

    double res = 0.0;
    res += exp(-t) * (2.0 * M_PI * cos(M_PI * x) - x * M_PI * M_PI * sin (M_PI * x)) * (1 + y) * sin (M_PI * y) * (2 - z) * sin (M_PI * z);
    res += exp(-t) * x * sin (M_PI * x) * (2.0 * M_PI * cos(M_PI * y) - (1 + y) * M_PI * M_PI * sin(M_PI * y)) * (2 - z) * sin (M_PI * z);
    res += exp(-t) * x * sin (M_PI * x) * (1 + y) * sin (M_PI * y) * (2.0 * (-1) * M_PI * cos(M_PI * z) - (2 - z) * M_PI * M_PI * sin(M_PI * z));
    return res;
}

void uFun2_ex_parab_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(3);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = exp(-t) * (sin (M_PI * x) + x * M_PI * cos(M_PI * x)) * (1 + y) * sin (M_PI * y) * (2 - z) * sin (M_PI * z);
    gradx(1) = exp(-t) * x * sin (M_PI * x) * (sin (M_PI * y) + (1 + y) * M_PI * cos(M_PI * y)) * (2 - z) * sin (M_PI * z);
    gradx(2) = exp(-t) * x * sin (M_PI * x) * (1 + y) * sin (M_PI * y) * (- sin (M_PI * z) + (2 - z) * M_PI * cos(M_PI * z));
}

double uFun3_ex_parab(const Vector & xt)
{
    if (xt.Size() != 3)
        cout << "Error, this is only 3-d = 2-d + time solution" << endl;
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    return exp(-t) * x * sin (M_PI * x) * (1 + y) * sin (M_PI * y);
}

double uFun3_ex_parab_dt(const Vector & xt)
{
    return - uFun3_ex_parab(xt);
}

double uFun3_ex_parab_laplace(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    double res = 0.0;
    res += exp(-t) * (2.0 * M_PI * cos(M_PI * x) - x * M_PI * M_PI * sin (M_PI * x)) * (1 + y) * sin (M_PI * y);
    res += exp(-t) * x * sin (M_PI * x) * (2.0 * M_PI * cos(M_PI * y) - (1 + y) * M_PI * M_PI * sin(M_PI * y));
    return res;
}

void uFun3_ex_parab_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = exp(-t) * (sin (M_PI * x) + x * M_PI * cos(M_PI * x)) * (1 + y) * sin (M_PI * y);
    gradx(1) = exp(-t) * x * sin (M_PI * x) * (sin (M_PI * y) + (1 + y) * M_PI * cos(M_PI * y));
}
////////////////////////////

/////////// from the wave example

double uFun_ex_wave(const Vector & xt)
{
    const double PI = 3.141592653589793;
    double xi(xt(0));
    double yi(xt(1));
    double zi(0.0);
    double vi(0.0);

    if (xt.Size() == 3)
    {
        double t = xt(2);
        return sin(PI*xi)*sin(PI*yi) * t * t;
        //return sin(PI*xi)*sin(PI*yi);
        //return sin(PI*xi)*sin(PI*yi) * t;
    }
    if (xt.Size() == 4)
    {
        zi = xt(2);
        vi = xt(3);
        //cout << "sol for 4D" << endl;
        return sin(PI*xi)*sin(PI*yi)*sin(PI*zi)*vi;
    }

    return 0.0;
}

double uFun_ex_wave_dt(const Vector & xt)
{
    const double PI = 3.141592653589793;
    double xi(xt(0));
    double yi(xt(1));
    double zi(0.0);


    if (xt.Size() == 3)
    {
        double t(xt(2));
        return sin(PI*xi)*sin(PI*yi)*2*t;
        //return 1.0;
        //return 0.0;
        //return sin(PI*xi)*sin(PI*yi);
    }
    if (xt.Size() == 4)
    {
        zi = xt(2);
        return sin(PI*xi)*sin(PI*yi)*sin(PI*zi);
    }


    return 0.0;
}

double uFun_ex_wave_dt2(const Vector & xt)
{
    double xi(xt(0));
    double yi(xt(1));
//    double zi(0.0);

    if (xt.Size() == 3)
    {
        return sin(M_PI*xi)*sin(M_PI*yi)*2.0;
        //return 0.0;
    }

    return 0.0;

}

double uFun_ex_wave_laplace(const Vector & xt)
{
    return (-(xt.Size()-1) * M_PI * M_PI) *uFun_ex_wave(xt);
    //return 0.0;
}

double uFun_ex_wave_dtlaplace(const Vector & xt)
{
    double xi(xt(0));
    double yi(xt(1));
//    double zi(0.0);
    double t(xt(xt.Size() - 1));
    //return (-(xt.Size()-1) * PI * PI) *uFun_ex_wave(xt);
    //return (-(xt.Size()-1) * M_PI * M_PI) *sin(M_PI*xi)*sin(M_PI*yi);         // for t * sin x * sin y
    return (-(xt.Size()-1) * M_PI * M_PI) *sin(M_PI*xi)*sin(M_PI*yi) * 2.0 * t; // for t^2 * sin x * sin y
    return 0.0;
}

void uFun_ex_wave_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
//    double z(0.0);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);


    if (xt.Size() == 3)
    {
        gradx(0) = t * t * M_PI * cos (M_PI * x) * sin (M_PI * y);
        gradx(1) = t * t * M_PI * sin (M_PI * x) * cos (M_PI * y);
    }

    /*
    if (xt.Size() == 4)
    {
        z = xt(2);
        gradx(0) = t * PI * cos (PI * x) * sin (PI * y) * sin (PI * z);
        gradx(1) = t * PI * sin (PI * x) * cos (PI * y) * sin (PI * z);
        gradx(2) = t * PI * sin (PI * x) * sin (PI * y) * cos (PI * z);
    }
    */


    /*
    if (xt.Size() == 3)
    {
        gradx(0) = M_PI * cos (M_PI * x) * sin (M_PI * y);
        gradx(1) = M_PI * sin (M_PI * x) * cos (M_PI * y);
    }
    */


    /*
    if (xt.Size() == 3)
    {
        gradx(0) = t * M_PI * cos (M_PI * x) * sin (M_PI * y);
        gradx(1) = t * M_PI * sin (M_PI * x) * cos (M_PI * y);
    }
    */


}

void uFun_ex_wave_dtgradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
//    double z(0.0);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    // for t * sin x * sin y
    /*
    if (xt.Size() == 3)
    {
        gradx(0) = M_PI * cos (M_PI * x) * sin (M_PI * y);
        gradx(1) = M_PI * sin (M_PI * x) * cos (M_PI * y);
    }
    */

    // for t^2 * sin x * sin y
    if (xt.Size() == 3)
    {
        gradx(0) = M_PI * cos (M_PI * x) * sin (M_PI * y) * 2.0 * t;
        gradx(1) = M_PI * sin (M_PI * x) * cos (M_PI * y) * 2.0 * t;
    }

}

double uFun1_ex_wave(const Vector & xt)
{
    double tmp = (xt.Size() == 4) ? sin(M_PI*xt(2)) : 1.0;
    return exp(-xt(xt.Size()-1))*sin(M_PI*xt(0))*sin(M_PI*xt(1))*tmp;
}

double uFun1_ex_wave_dt(const Vector & xt)
{
    return - uFun1_ex_wave(xt);
}

double uFun1_ex_wave_dt2(const Vector & xt)
{
    return uFun1_ex_wave(xt);
}

double uFun1_ex_wave_laplace(const Vector & xt)
{
    return (- (xt.Size() - 1) * M_PI * M_PI ) * uFun1_ex_wave(xt);
}

double uFun1_ex_wave_dtlaplace(const Vector & xt)
{
    return -uFun1_ex_wave_laplace(xt);
}

void uFun1_ex_wave_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double z(0.0);
    double t = xt(xt.Size()-1);

    gradx.SetSize(xt.Size() - 1);

    if (xt.Size() == 3)
    {
        gradx(0) = exp(-t) * M_PI * cos (M_PI * x) * sin (M_PI * y);
        gradx(1) = exp(-t) * M_PI * sin (M_PI * x) * cos (M_PI * y);
    }
    if (xt.Size() == 4)
    {
        z = xt(2);
        gradx(0) = exp(-t) * M_PI * cos (M_PI * x) * sin (M_PI * y) * sin (M_PI * z);
        gradx(1) = exp(-t) * M_PI * sin (M_PI * x) * cos (M_PI * y) * sin (M_PI * z);
        gradx(2) = exp(-t) * M_PI * sin (M_PI * x) * sin (M_PI * y) * cos (M_PI * z);
    }

}

void uFun1_ex_wave_dtgradx(const Vector& xt, Vector& gradx )
{
    gradx.SetSize(xt.Size() - 1);

    Vector gradS;
    uFun1_ex_wave_gradx(xt,gradS);

    for ( int d = 0; d < xt.Size() - 1; ++d)
        gradx(d) = - gradS(d);
}

double uFun2_ex_wave(const Vector & xt)
{
    if (xt.Size() != 4)
        cout << "Error, this is only 4-d = 3-d + time solution" << endl;
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(3);

    return exp(-t) * x * sin (M_PI * x) * (1 + y) * sin (M_PI * y) * (2 - z) * sin (M_PI * z);
}

double uFun2_ex_wave_dt(const Vector & xt)
{
    return - uFun2_ex_wave(xt);
}

double uFun2_ex_wave_dt2(const Vector & xt)
{
    return uFun2_ex_wave(xt);
}

double uFun2_ex_wave_laplace(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(3);

    double res = 0.0;
    res += exp(-t) * (2.0 * M_PI * cos(M_PI * x) - x * M_PI * M_PI * sin (M_PI * x)) * (1 + y) * sin (M_PI * y) * (2 - z) * sin (M_PI * z);
    res += exp(-t) * x * sin (M_PI * x) * (2.0 * M_PI * cos(M_PI * y) - (1 + y) * M_PI * M_PI * sin(M_PI * y)) * (2 - z) * sin (M_PI * z);
    res += exp(-t) * x * sin (M_PI * x) * (1 + y) * sin (M_PI * y) * (2.0 * (-1) * cos(M_PI * z) - (2 - z) * M_PI * M_PI * sin(M_PI * z));
    return res;
}

double uFun2_ex_wave_dtlaplace(const Vector & xt)
{
    return -uFun2_ex_wave_laplace(xt);
}

void uFun2_ex_wave_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double z = xt(2);
    double t = xt(3);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = exp(-t) * (sin (M_PI * x) + x * M_PI * cos(M_PI * x)) * (1 + y) * sin (M_PI * y) * (2 - z) * sin (M_PI * z);
    gradx(1) = exp(-t) * x * sin (M_PI * x) * (sin (M_PI * y) + (1 + y) * M_PI * cos(M_PI * y)) * (2 - z) * sin (M_PI * z);
    gradx(2) = exp(-t) * x * sin (M_PI * x) * (1 + y) * sin (M_PI * y) * (- sin (M_PI * z) + (2 - z) * M_PI * cos(M_PI * z));
}

void uFun2_ex_wave_dtgradx(const Vector& xt, Vector& gradx )
{
    gradx.SetSize(xt.Size() - 1);

    Vector gradS;
    uFun2_ex_wave_gradx(xt,gradS);

    for ( int d = 0; d < xt.Size() - 1; ++d)
        gradx(d) = - gradS(d);
}

double uFun4_ex_wave(const Vector & xt)
{
    if (xt.Size() != 3)
        cout << "Error, this is only 3-d solution" << endl;
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    return 16.0 * x * (x - 1) * y * (y - 1) * t * t;
}

double uFun4_ex_wave_dt(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    return 16.0 * x * (x - 1) * y * (y - 1) * 2.0 * t;
}

double uFun4_ex_wave_dt2(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
//    double t = xt(2);

    return 16.0 * x * (x - 1) * y * (y - 1) * 2.0;
}

double uFun4_ex_wave_laplace(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    return 16.0 * (2.0 * y * (y - 1) + 2.0 * x * (x - 1)) * t * t;
}

double uFun4_ex_wave_dtlaplace(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    return 16.0 * (2.0 * y * (y - 1) + 2.0 * x * (x - 1)) * 2.0 * t;
}

void uFun4_ex_wave_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = 16.0 * (2.0 * x - 1) * y * (y - 1) * t * t;
    gradx(1) = 16.0 * x * (x - 1) * (2.0 * y - 1) * t * t;

}

void uFun4_ex_wave_dtgradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = 16.0 * (2.0 * x - 1) * y * (y - 1) * 2.0 * t;
    gradx(1) = 16.0 * x * (x - 1) * (2.0 * y - 1) * 2.0 * t;
}

double uFun3_ex_wave(const Vector & xt)
{
    if (xt.Size() != 3)
        cout << "Error, this is only 3-d = 2d + time solution" << endl;
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    return exp(-t) * x * sin (M_PI * x) * (1 + y) * sin (M_PI * y);
}

double uFun3_ex_wave_dt(const Vector & xt)
{
    return - uFun3_ex_wave(xt);
}

double uFun3_ex_wave_dt2(const Vector & xt)
{
    return uFun3_ex_wave(xt);
}

double uFun3_ex_wave_laplace(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    double res = 0.0;
    res += exp(-t) * (2.0 * M_PI * cos(M_PI * x) - x * M_PI * M_PI * sin (M_PI * x)) * (1 + y) * sin (M_PI * y);
    res += exp(-t) * x * sin (M_PI * x) * (2.0 * M_PI * cos(M_PI * y) - (1 + y) * M_PI * M_PI * sin(M_PI * y));
    return res;
}

double uFun3_ex_wave_dtlaplace(const Vector & xt)
{
    return -uFun3_ex_wave_laplace(xt);
}

void uFun3_ex_wave_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = exp(-t) * (sin (M_PI * x) + x * M_PI * cos(M_PI * x)) * (1 + y) * sin (M_PI * y);
    gradx(1) = exp(-t) * x * sin (M_PI * x) * (sin (M_PI * y) + (1 + y) * M_PI * cos(M_PI * y));
}

void uFun3_ex_wave_dtgradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = - exp(-t) * (sin (M_PI * x) + x * M_PI * cos(M_PI * x)) * (1 + y) * sin (M_PI * y);
    gradx(1) = - exp(-t) * x * sin (M_PI * x) * (sin (M_PI * y) + (1 + y) * M_PI * cos(M_PI * y));
}


double uFun5_ex_wave(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    return 16.0 * x * x * (x - 1) * (x - 1) * y * y * (y - 1) * (y - 1) * t * t;
}

double uFun5_ex_wave_dt(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    return 16.0 * x * x * (x - 1) * (x - 1) * y * y * (y - 1) * (y - 1) * 2.0 * t;
}

double uFun5_ex_wave_dt2(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
//    double t = xt(2);

    return 16.0 * x * x * (x - 1) * (x - 1) * y * y * (y - 1) * (y - 1) * 2.0;
}

double uFun5_ex_wave_laplace(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    return 16.0 * (2.0 * ((x-1)*(2*x-1) + x*(2*x-1) + 2*x*(x-1)) * y * (y - 1) * y * (y - 1)\
                   + 2.0 * ((y-1)*(2*y-1) + y*(2*y-1) + 2*y*(y-1)) * x * (x - 1) * x * (x - 1)) * t * t;
}

double uFun5_ex_wave_dtlaplace(const Vector & xt)
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    return 16.0 * (2.0 * ((x-1)*(2*x-1) + x*(2*x-1) + 2*x*(x-1)) * y * (y - 1) * y * (y - 1)\
                   + 2.0 * ((y-1)*(2*y-1) + y*(2*y-1) + 2*y*(y-1)) * x * (x - 1) * x * (x - 1)) * 2.0 * t;
}

void uFun5_ex_wave_gradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = 16.0 * 2.0 * x * (x - 1) * (2.0 * x - 1) * y * (y - 1) * y * (y - 1) * t * t;
    gradx(1) = 16.0 * x * (x - 1) * x * (x - 1) * 2.0 * y * (y - 1) * (2.0 * y - 1) * t * t;

}

void uFun5_ex_wave_dtgradx(const Vector& xt, Vector& gradx )
{
    double x = xt(0);
    double y = xt(1);
    double t = xt(2);

    gradx.SetSize(xt.Size() - 1);

    gradx(0) = 16.0 * 2.0 * x * (x - 1) * (2.0 * x - 1) * y * (y - 1) * y * (y - 1) * 2.0 * t;
    gradx(1) = 16.0 * x * (x - 1) * x * (x - 1) * 2.0 * y * (y - 1) * (2.0 * y - 1) * 2.0 * t;
}


////////////////////////////


void testVectorFun(const Vector& xt, Vector& res)
{
    res.SetSize(xt.Size());
    res = 1.0;
}

double testH1fun(Vector& xt)
{
    double x = xt(0);
    double y = xt(1);
    double z;
    if (xt.Size() == 4)
        z = xt(2);
    //double t = xt(xt.Size() - 1);

    if (xt.Size() == 3)
        return (x*x + y*y + 1.0);
    if (xt.Size() == 4)
        return (x*x + y*y + z*z + 1.0);
    return 0.0;
}

void testHdivfun(const Vector& xt, Vector &res)
{
    res.SetSize(xt.Size());

    double x = xt(0);
    double y = xt(1);
    double z;
    if (xt.Size() == 4)
        z = xt(2);
    //double t = xt(xt.Size() - 1);

    res = 0.0;

    if (xt.Size() == 3)
    {
        res(2) = (x*x + y*y + 1.0);
    }
    if (xt.Size() == 4)
    {
        res(3) = (x*x + y*y + z*z + 1.0);
    }
}

} // for namespace mfem

