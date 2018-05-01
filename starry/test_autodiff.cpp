#include <iostream>
#include <iomanip>
#include <Eigen/Core>
#include <cmath>
#include "solver.h"
#include <unsupported/Eigen/AutoDiff>
template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
using namespace std;
typedef Eigen::AutoDiffScalar<Eigen::Vector2d> Grad;

int main() {

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
    Grad b = Grad(b0, 2, 0);
    Grad r = Grad(r0, 2, 1);
    Vector<Grad> y = Vector<Grad>::Zero((lmax + 1) * (lmax + 1));
    for (int n = 0; n < (lmax + 1) * (lmax + 1); n++) y(n) = 1;
    solver::Greens<Grad> G(lmax);
    solver::computesT(G, b, r, y);

    // Plot the values and the derivatives w/ respect to b and r
    for (int n = 0; n < (lmax + 1) * (lmax + 1); n++) {
        cout << n << ": " << G.sT(n).value() << endl;
        cout << "   " << G.sT(n).derivatives()(0) << " (" << (G01.sT(n) - G0.sT(n)) / eps << ")  " << endl;
        cout << "   " << G.sT(n).derivatives()(1) << " (" << (G02.sT(n) - G0.sT(n)) / eps << ")  " << endl;
    }
    return 0;

}
