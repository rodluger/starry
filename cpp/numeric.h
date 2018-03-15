/**
Numerical integration by adaptive mesh refinement.

*/

#ifndef _STARRY_NUMERIC_H_
#define _STARRY_NUMERIC_H_

#include <iostream>
#include <cmath>
#include <Eigen/Core>

// Shorthand
template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T>
using VectorT = Eigen::Matrix<T, 1, Eigen::Dynamic>;
template <typename T>
using UnitVector = Eigen::Matrix<T, 3, 1>;

namespace numeric {

    // Evaluate a map `p` at a given (x, y) coordinate during an occultation
    template <typename T>
    T evaluate(T x, T y, T xo, T yo, T ro, int lmax, Vector<T>& p) {

        // Basis
        Vector<T> basis;
        basis.resize((lmax + 1) * (lmax + 1));

        // Check if outside the sphere
        if (x * x + y * y >= 1) return 0;

        // Check if inside the occultor
        if ((x - xo) * (x - xo) + (y - yo) * (y - yo) <= ro * ro) return 0;

        int l, m, mu, nu, n = 0;
        T z = sqrt(1.0 - x * x - y * y);

        // Compute the polynomial basis where it is needed
        for (l=0; l<lmax+1; l++) {
            for (m=-l; m<l+1; m++) {
                if (std::abs(p(n)) < 1e-14) {
                    basis(n) = 0;
                } else {
                    mu = l - m;
                    nu = l + m;
                    if ((nu % 2) == 0)
                        basis(n) = pow(x, mu / 2) * pow(y, nu / 2);
                    else
                        basis(n) = pow(x, (mu - 1) / 2) *
                                   pow(y, (nu - 1) / 2) * z;
                }
                n++;
            }
        }

        // Dot the coefficients in to our polynomial map
        return p.dot(basis);

    }

    // Return the flux in a cell
    template <typename T>
    T fcell(T r1, T r2, T t1, T t2, T xo, T yo, T ro, int lmax, Vector<T>& p) {
        return 0.125 * (r2 * r2 - r1 * r1) * (t2 - t1) *
               (evaluate<T>(r1 * cos(t1), r1 * sin(t1), xo, yo, ro, lmax, p) +
                evaluate<T>(r1 * cos(t2), r1 * sin(t2), xo, yo, ro, lmax, p) +
                evaluate<T>(r2 * cos(t1), r2 * sin(t1), xo, yo, ro, lmax, p) +
                evaluate<T>(r2 * cos(t2), r2 * sin(t2), xo, yo, ro, lmax, p));
    }

    // Return the numerically computed flux
    template <typename T>
    void fnum(T r1, T r2, T t1, T t2, T xo, T yo, T ro, double tol, int lmax, Vector<T>& p, T* f) {
        // Coarse estimate
        T fcoarse = fcell<T>(r1, r2, t1, t2, xo, yo, ro, lmax, p);

        // Fine estimate (bisection)
        T r = 0.5 * (r1 + r2);
        T t = 0.5 * (t1 + t2);
        T ffine = (fcell<T>(r1, r, t1, t, xo, yo, ro, lmax, p) +
                   fcell<T>(r1, r, t, t2, xo, yo, ro, lmax, p) +
                   fcell<T>(r, r2, t1, t, xo, yo, ro, lmax, p) +
                   fcell<T>(r, r2, t, t2, xo, yo, ro, lmax, p));

        // Compare
        if (std::abs(fcoarse - ffine) > tol) {
            // Recurse
            fnum<T>(r1, r, t1, t, xo, yo, ro, tol, lmax, p, f);
            fnum<T>(r1, r, t, t2, xo, yo, ro, tol, lmax, p, f);
            fnum<T>(r, r2, t1, t, xo, yo, ro, tol, lmax, p, f);
            fnum<T>(r, r2, t, t2, xo, yo, ro, tol, lmax, p, f);
        } else {
            // We're done at this level
            (*f) += ffine;
            return;
        }
    }

    // Compute the total flux during or outside of an occultation
    template <typename T>
    T flux(T xo, T yo, T ro, int lmax, Vector<T>& p, double tol) {
        T f = 0;
        T b;
        T theta;
        if (isinf(xo) || isinf(yo)) {
            b = INFINITY;
            theta = 0;
        } else {
            b = sqrt(xo * xo + yo * yo);
            theta = atan2(yo, xo);
        }
        tol /= M_PI;
        if (b > 1 + ro) {
            fnum<T>(0, 1, theta, theta + 2 * M_PI, xo, yo, ro, tol, lmax, p, &f);
        } else if (b > 1) {
            fnum<T>(0, (1 + b - ro) / 2., theta, theta + 2 * M_PI, xo, yo, ro, tol, lmax, p, &f);
            fnum<T>((1 + b - ro) / 2., 1, theta, theta + 2 * M_PI, xo, yo, ro, tol, lmax, p, &f);
        } else {
            fnum<T>(0, b, theta, theta + 2 * M_PI, xo, yo, ro, tol, lmax, p, &f);
            fnum<T>(b, 1, theta, theta + 2 * M_PI, xo, yo, ro, tol, lmax, p, &f);
        }
        return f;
    }

}; // namespace numeric

#endif
