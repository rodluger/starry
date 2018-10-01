#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <vector>
#include "utils.h"
#include "limbdark.h"
#include "solver.h"
#include "maps.h"

using namespace utils;

int main() {


    using T = Matrix<double>;
    int nwav = 3;

    //using T = Vector<double>;
    //int nwav = 1;

    // Class-level
    using A = ADScalar<Scalar<T>, 2>;
    using limbdark::GreensLimbDark;
    using limbdark::computeC;
    using limbdark::normC;
    int lmax = 5;
    Scalar<T> b = 0.5;
    Scalar<T> r = 0.1;
    T u;
    resize(u, lmax + 1, nwav);
    setRow(u, 0, NAN);
    setRow(u, 1, 0.1);
    setRow(u, 2, 0.2);
    setRow(u, 3, 0.3);
    setRow(u, 4, 0.4);
    setRow(u, 5, 0.5);
    A b_grad, r_grad;
    Vector<Matrix<Scalar<T>>> dagol_cdu(nwav);
    GreensLimbDark<Scalar<T>> L(lmax);
    GreensLimbDark<A> L_grad(lmax);
    Row<T> agol_norm;
    resize(agol_norm, 0, nwav);
    T agol_c(lmax + 1, nwav);


    // * -- * -- *


    // Temporaries
    Row<T> flux;
    resize(flux, 0, nwav);
    T dFdc(lmax + 1, nwav);
    T dFdu(lmax, nwav);
    VectorT<Scalar<T>> dSdb(lmax + 1);
    VectorT<Scalar<T>> dSdr(lmax + 1);
    Row<T> dFdb, dFdr;
    resize(dFdb, 0, nwav);
    resize(dFdr, 0, nwav);

    // Pre-compute this
    for (int n = 0; n < nwav; ++n) {
        agol_c.col(n) = computeC(getColumn(u, n), dagol_cdu(n));
        setIndex(agol_norm, n, normC(getColumn(agol_c, n)));
    }

    // Set up AutoDiff
    b_grad.value() = b;
    b_grad.derivatives() = Vector<Scalar<T>>::Unit(2, 0);
    r_grad.value() = r;
    r_grad.derivatives() = Vector<Scalar<T>>::Unit(2, 1);

    // Compute S, dS / db, and dS / dr
    L_grad.compute(b_grad, r_grad);

    // Store the value of the S vector and its derivatives
    for (int i = 0; i <= lmax; ++i) {
        L.S(i) = L_grad.S(i).value();
        dSdb(i) = L_grad.S(i).derivatives()(0);
        dSdr(i) = L_grad.S(i).derivatives()(1);
    }

    // Compute the value of the flux and its derivatives
    for (int n = 0; n < nwav; ++n) {

        // F, dF / db and dF / dr
        setIndex(flux, n, L.S.dot(getColumn(agol_c, n)) * getIndex(agol_norm, n));
        setIndex(dFdb, n, dSdb.dot(getColumn(agol_c, n)) * getIndex(agol_norm, n));
        setIndex(dFdr, n, dSdr.dot(getColumn(agol_c, n)) * getIndex(agol_norm, n));

        // Compute dF / dc
        dFdc.block(0, n, lmax + 1, 1) = L.S.transpose() * getIndex(agol_norm, n);
        dFdc(0, n) -= getIndex(flux, n) * getIndex(agol_norm, n) * pi<Scalar<T>>();
        dFdc(1, n) -= 2.0 * pi<Scalar<T>>() / 3.0 * getIndex(flux, n) * getIndex(agol_norm, n);

        // Chain rule to get dF / du
        dFdu.block(0, n, lmax, 1).transpose() = dFdc.block(0, n, lmax + 1, 1).transpose() * dagol_cdu(n).block(0, 1, lmax + 1, lmax);

    }

    // Print the derivs
    std::cout << dFdb << std::endl;
    std::cout << dFdr << std::endl;
    std::cout << dFdu << std::endl;
    std::cout << std::endl;



    // *** Luger ***
    maps::Map<T> map(lmax, nwav);
    map.setU(u.block(1, 0, lmax, nwav));
    std::cout << map.flux(0, b, 0, r, true) << std::endl;
    std::cout << map.flux(0, b, 0, r, false) << std::endl;
    std::cout << map.getGradient() << std::endl;

    // *** Luger ***



}
