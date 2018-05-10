#include <iostream>
#include <iomanip>
#include <Eigen/Core>
#include <cmath>
#include "solver.h"
#include "rotation.h"
#include "maps.h"
#include <unsupported/Eigen/AutoDiff>
template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T>
using UnitVector = Eigen::Matrix<T, 3, 1>;
using namespace std;
typedef Eigen::AutoDiffScalar<Eigen::Vector2d> Grad2;
typedef Eigen::AutoDiffScalar<Eigen::Vector4d> Grad4;
typedef Eigen::AutoDiffScalar<Eigen::Matrix<double, 7, 1>> Grad7;

void test_sT() {

    // User inputs
    int lmax = 2;
    double b0 = 0.3;
    double r0 = 0.1;
    double eps = 1e-7;

    // Let's compute the derivatives numerically
    double b1 = b0 + eps;
    double r1 = r0 + eps;
    Vector<double> y0 = Vector<double>::Zero((lmax + 1) * (lmax + 1));
    for (int n = 0; n < (lmax + 1) * (lmax + 1); n++) y0(n) = 1;
    solver::Greens<double> G0(lmax);
    solver::computesT(G0, b0, r0, y0);
    solver::Greens<double> G01(lmax);
    solver::computesT(G01, b1, r0, y0);
    solver::Greens<double> G02(lmax);
    solver::computesT(G02, b0, r1, y0);

    // Let's autodifferentiate!
    Grad2 b = Grad2(b0, 2, 0);
    Grad2 r = Grad2(r0, 2, 1);
    Vector<Grad2> y = Vector<Grad2>::Zero((lmax + 1) * (lmax + 1));
    for (int n = 0; n < (lmax + 1) * (lmax + 1); n++) y(n) = 1;
    solver::Greens<Grad2> G(lmax);
    solver::computesT(G, b, r, y);

    // Plot the values and the derivatives w/ respect to b and r
    for (int n = 0; n < (lmax + 1) * (lmax + 1); n++) {
        cout << n << ": " << G.sT(n).value() << endl;
        cout << "   " << G.sT(n).derivatives()(0) << " (" << (G01.sT(n) - G0.sT(n)) / eps << ")  " << endl;
        cout << "   " << G.sT(n).derivatives()(1) << " (" << (G02.sT(n) - G0.sT(n)) / eps << ")  " << endl;
    }

}

void test_rotation() {

    int lmax = 1;

    Grad4 ux = Grad4(1, 4, 0);
    Grad4 uy = Grad4(0, 4, 1);
    Grad4 uz = Grad4(0, 4, 2);
    Grad4 theta = Grad4(M_PI / 3., 4, 3);

    UnitVector<Grad4> axis({ux, uy, uz});
    Grad4 costheta = cos(theta);
    Grad4 sintheta = sin(theta);

    rotation::Wigner<Grad4> R(lmax);
    rotation::computeR(lmax, axis, costheta, sintheta, R.Complex, R.Real);

    cout << R.Real[1](2, 0).derivatives() << endl;

}

int main() {
    // Compute auto gradients, **all the way through!**

    // User
    int lmax = 2;
    double ux = 1;
    double uy = 0;
    double uz = 0;
    double theta = 0.5;
    double xo = 0.3;
    double yo = 0.5;
    double ro = 0.7;

    // Derivatives with respect to...
    Grad7 ux_g = Grad7(ux, 7, 0);
    Grad7 uy_g = Grad7(uy, 7, 1);
    Grad7 uz_g = Grad7(uz, 7, 2);
    Grad7 theta_g = Grad7(theta, 7, 3);
    Grad7 xo_g = Grad7(xo, 7, 4);
    Grad7 yo_g = Grad7(yo, 7, 5);
    Grad7 ro_g = Grad7(ro, 7, 6);
    UnitVector<Grad7> axis_g({ux_g, uy_g, uz_g});

    // Set a map coefficient
    maps::Map<Grad7> map(lmax);
    map.set_coeff(1, 0, 1);

    // Compute the flux
    Grad7 flux = map.flux(axis_g, theta_g, xo_g, yo_g, ro_g);

    // Print the flux and all the derivatives
    cout << flux.value() << endl;
    cout << flux.derivatives() << endl;

    return 0;
}
