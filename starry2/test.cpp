#include <stdlib.h>
#include <iostream>
#include <cmath>
#include "utils.h"
#include "rotation.h"
#include "basis.h"
using namespace utils;
using namespace basis;
using namespace rotation;

/*
void spectral() {

    int y_deg = 1;
    int u_deg = 1;
    int lmax = 2;
    int nwav = 2;
    double theta;
    Matrix<double> p_uy;
    VectorT<Matrix<double>> grad_p1;
    VectorT<Matrix<double>> grad_p2;
    Vector<double> I1(2), I2(2), dIdtheta(2);
    VectorT<double> rtmp, rtmp2, rtmp3;
    double eps = 1.e-8;

    // Mock sph harm and limb darkening vectors
    Matrix<double> y(9, 2);
    Matrix<double> p_u(9, 2);
    y <<   2, 4,
           3, 6,
           5, 10,
           7, 14,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    p_u << 0, 0,
           0, 0,
           0.5, 0.33,
           0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    // Mock polynomial basis vector
    VectorT<double> pT(9);
    pT << 1, 2, 3, 4, 5, 6, 7, 8, 9;

    // Mock rotation solution vector
    VectorT<double> rT(9);
    rT << 5, 7, 4, 3, -3, 6, 10, 2, 1;

    // Mock rotation matrix
    Matrix<double> R(9, 9);
    R.setZero();
    R(0, 0) = 1;
    Matrix<double> dRdtheta(9, 9);
    dRdtheta.setZero();

    // Mock change of basis matrix
    Matrix<double> A1;
    A1 = Matrix<double>::Identity(9, 9);

    // Analytic derivative
    // dI/dtheta = p^T . d LD / d(A1 . R . y) . A1 . d(R . y) / dtheta
    theta = M_PI / 3;
    R(1, 1) = sin(theta);
    R(2, 2) = tan(theta);
    R(3, 3) = cos(theta);
    polymul(y_deg, Matrix<double>(A1 * R * y), u_deg, p_u, lmax, p_uy, grad_p1, grad_p2);
    rtmp = dot(rT, Matrix<double>(A1 * R * y));
    rtmp2 = dot(rT, p_u);
    rtmp3 = cwiseQuotient(rtmp, rtmp2);

    // Chain rule
    for (int n = 0; n < nwav; ++n) {
        auto foo = getColumn(p_uy, n) * rT;
        grad_p1(n) = (grad_p1(n) * getColumn(rtmp3, n)) + foo / getColumn(rtmp2, n);
        grad_p2(n) = (grad_p2(n) * getColumn(rtmp3, n)) - foo / getColumn(rtmp, n) / (getColumn(rtmp2, n) * getColumn(rtmp2, n));
    }

    colwiseMult(p_uy, rtmp3);

    dRdtheta(1, 1) = cos(theta);
    dRdtheta(2, 2) = 1 / (cos(theta) * cos(theta));
    dRdtheta(3, 3) = -sin(theta);

    for (int n = 0; n < nwav; ++n) {
        dIdtheta(n) = pT * grad_p1(n) * A1 * getColumn(Matrix<double>(dRdtheta * y), n);
    }
    std::cout << "Analytic: " << dIdtheta.transpose() << std::endl;

    // Numerical derivative
    theta = M_PI / 3 - eps;
    R(1, 1) = sin(theta);
    R(2, 2) = tan(theta);
    R(3, 3) = cos(theta);
    polymul(y_deg, Matrix<double>(A1 * R * y), u_deg, p_u, lmax, p_uy);
    rtmp = dot(rT, Matrix<double>(A1 * R * y));
    rtmp2 = dot(rT, p_u);
    rtmp3 = cwiseQuotient(rtmp, rtmp2);
    colwiseMult(p_uy, rtmp3);
    I1 = pT * p_uy;
    theta = M_PI / 3 + eps;
    R(1, 1) = sin(theta);
    R(2, 2) = tan(theta);
    R(3, 3) = cos(theta);
    polymul(y_deg, Matrix<double>(A1 * R * y), u_deg, p_u, lmax, p_uy);
    rtmp = dot(rT, Matrix<double>(A1 * R * y));
    rtmp2 = dot(rT, p_u);
    rtmp3 = cwiseQuotient(rtmp, rtmp2);
    colwiseMult(p_uy, rtmp3);
    I2 = pT * p_uy;
    dIdtheta = (I2 - I1) / (2 * eps);
    std::cout << "Numerical: " << dIdtheta.transpose() << std::endl;

}
*/

void scalar() {

    int y_deg = 1;
    int u_deg = 1;
    int lmax = 2;
    int nwav = 1;
    double theta;
    Vector<double> p_uy;
    VectorT<Matrix<double>> grad_p1;
    VectorT<Matrix<double>> grad_p2;
    double I1, I2, dIdtheta;
    double rtmp, rtmp2, rtmp3;
    double eps = 1.e-8;

    // Mock sph harm and limb darkening vectors
    Vector<double> y(9);
    Vector<double> p_u(9);
    y <<   10,
           3,
           1,
           5,
           0, 0, 0, 0, 0;
    p_u << 0,
           0,
           0.5,
           0,
           0, 0, 0, 0, 0;

    // Mock polynomial basis vector
    VectorT<double> pT(9);
    pT << 1, 2, 3, 4, 5, 6, 7, 8, 9;

    // Mock rotation solution vector
    VectorT<double> rT(9);
    rT << 5, 7, 4, 3, -3, 6, 10, 2, 1;

    // Mock rotation matrix
    Matrix<double> R(9, 9);
    R.setZero();
    R(0, 0) = 1;
    Matrix<double> dRdtheta(9, 9);
    dRdtheta.setZero();

    // Mock change of basis matrix
    Matrix<double> A1;
    A1 = Matrix<double>::Identity(9, 9);

    // Analytic derivative
    // dI/dtheta = p^T . d LD / d(A1 . R . y) . A1 . d(R . y) / dtheta
    theta = M_PI / 3;
    R(1, 1) = sin(theta);
    R(2, 2) = tan(theta);
    R(3, 3) = cos(theta);
    polymul(y_deg, Vector<double>(A1 * R * y), u_deg, p_u, lmax, p_uy, grad_p1, grad_p2);
    rtmp = dot(rT, Vector<double>(A1 * R * y));
    rtmp2 = dot(rT, p_uy);
    rtmp3 = cwiseQuotient(rtmp, rtmp2);

    // Chain rule
    Matrix<double> mattmp;
    Matrix<double> foo;
    for (int n = 0; n < nwav; ++n) {

        mattmp = getColumn(p_uy, n) * rT;
        mattmp /= getColumn(rtmp2, n);
        grad_p1(n) *= getColumn(rtmp3, n);
        foo = grad_p1(n) + mattmp - mattmp * grad_p1(n);
        grad_p1(n) = foo;

        //grad_p1(n) = grad_p1(n) + mattmp - mattmp * grad_p1(n);




    }

    // Apply the normalization
    colwiseMult(p_uy, rtmp3);

    dRdtheta(1, 1) = cos(theta);
    dRdtheta(2, 2) = 1 / (cos(theta) * cos(theta));
    dRdtheta(3, 3) = -sin(theta);

    dIdtheta = pT * grad_p1(0) * A1 * getColumn(Vector<double>(dRdtheta * y), 0);
    std::cout << "Analytic: " << dIdtheta << std::endl;

    // Numerical derivative
    theta = M_PI / 3 - eps;
    R(1, 1) = sin(theta);
    R(2, 2) = tan(theta);
    R(3, 3) = cos(theta);
    polymul(y_deg, Vector<double>(A1 * R * y), u_deg, p_u, lmax, p_uy);
    rtmp = dot(rT, Vector<double>(A1 * R * y));
    rtmp2 = dot(rT, p_uy);
    rtmp3 = cwiseQuotient(rtmp, rtmp2);
    colwiseMult(p_uy, rtmp3);
    I1 = pT * p_uy;
    theta = M_PI / 3 + eps;
    R(1, 1) = sin(theta);
    R(2, 2) = tan(theta);
    R(3, 3) = cos(theta);
    polymul(y_deg, Vector<double>(A1 * R * y), u_deg, p_u, lmax, p_uy);
    rtmp = dot(rT, Vector<double>(A1 * R * y));
    rtmp2 = dot(rT, p_uy);
    rtmp3 = cwiseQuotient(rtmp, rtmp2);
    colwiseMult(p_uy, rtmp3);
    I2 = pT * p_uy;
    dIdtheta = (I2 - I1) / (2 * eps);
    std::cout << "Numerical: " << dIdtheta << std::endl;

}

int main() {
    scalar();
}
