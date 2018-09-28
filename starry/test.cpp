#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <vector>
#include "utils.h"
#include "limbdark.h"
#include "solver.h"
#include "maps.h"

int main() {

    using namespace utils;
    using T = double;
    using A = ADScalar<T, 2>;
    using limbdark::GreensLimbDark;
    using limbdark::computeC;
    using limbdark::normC;

    int lmax = 5;
    T b = 0.5;
    T r = 0.1;
    Vector<T> u(lmax + 1);
    u(0) = NAN;
    u(1) = 0.2;
    u(2) = 0.3;
    u(3) = 0.4;
    u(4) = 0.5;
    u(5) = 0.6;

    // *** Agol ***

    VectorT<T> dFdc(lmax + 1);
    Matrix<T> dcdu;
    VectorT<T> dFdu;
    GreensLimbDark<A> L_grad(lmax);
    A b_grad, r_grad, f_grad;

    // Compute c(u), norm(u), and dc / du
    Vector<T> c = computeC(u, dcdu);
    T norm = normC(c);

    // Set up AutoDiff
    b_grad.value() = b;
    b_grad.derivatives() = Vector<T>::Unit(2, 0);
    r_grad.value() = r;
    r_grad.derivatives() = Vector<T>::Unit(2, 1);

    // Compute F, dF / db, and dF / dr
    L_grad.compute(b_grad, r_grad);
    f_grad = L_grad.S.dot(c) / norm;

    // Compute dF / dc
    for (int l = 0; l <= lmax; ++l)
        dFdc(l) = L_grad.S(l).value();
    dFdc /= norm;
    dFdc(0) -= f_grad.value() / norm * pi<T>();
    dFdc(1) -= 2.0 * pi<T>() / 3.0 * f_grad.value() / norm;

    // Chain rule to get dF / du
    dFdu = dFdc * dcdu.block(0, 1, lmax + 1, lmax);



    // Print the derivs
    std::cout << f_grad.derivatives() << std::endl;
    std::cout << dFdu.transpose() << std::endl;
    std::cout << std::endl;


    // *** Luger ***
    maps::Map<Vector<T>> map(lmax);
    map.setU(u.segment(1, lmax));
    map.flux(0, b, 0, r, true);
    std::cout << map.getGradient() << std::endl;

}
