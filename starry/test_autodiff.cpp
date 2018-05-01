#include <iostream>
#include <iomanip>
#include <Eigen/Core>
#include <cmath>
#include "solver.h"
#include <unsupported/Eigen/AutoDiff>
template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

using namespace std;

int main() {

    typedef Eigen::AutoDiffScalar<Eigen::Vector2d> Grad;

    // User inputs
    int lmax = 2;
    Grad b = Grad(0.3, 2, 0);
    Grad r = Grad(0.1, 2, 1);
    Vector<Grad> y = Vector<Grad>::Zero((lmax + 1) * (lmax + 1));
    for (int n = 0; n < (lmax + 1) * (lmax + 1); n++) y(n) = 1;
    Grad sn;
    solver::Greens<Grad> G(lmax);

    solver::computesT(G, b, r, y);
    for (int n = 0; n < (lmax + 1) * (lmax + 1); n++)
        cout << G.sT(n).value() << G.sT(n).derivatives().transpose() << endl;

    return 0;

}
