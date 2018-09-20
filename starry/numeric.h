/**
Numerical integration by adaptive mesh refinement.

NOTE: This is a lazily-coded, unoptimized module used
primarily for debugging. Use at your own risk!

*/

#ifndef _STARRY_NUMERIC_H_
#define _STARRY_NUMERIC_H_

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>
#include "utils.h"

namespace numeric {

    using std::abs;
    using std::fmod;
    using namespace utils;

    // Evaluate a map `p` at a given (x, y) coordinate during an occultation
    template <typename T>
    T evaluate(T x, T y, T xo, T yo, T ro, int lmax, const Vector<T>& p) {

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
                if (abs(p(n)) < 1e-14) {
                    basis(n) = 0;
                } else {
                    mu = l - m;
                    nu = l + m;
                   if ((nu % 2) == 0) {
                       if ((mu > 0) && (nu > 0))
                           basis(n) = pow(x, mu / 2) * pow(y, nu / 2);
                       else if (mu > 0)
                           basis(n) = pow(x, mu / 2);
                       else if (nu > 0)
                           basis(n) = pow(y, nu / 2);
                       else
                           basis(n) = 1;
                   } else {
                       if ((mu > 1) && (nu > 1))
                           basis(n) = pow(x, (mu - 1) / 2) *
                                      pow(y, (nu - 1) / 2) * z;
                       else if (mu > 1)
                           basis(n) = pow(x, (mu - 1) / 2) * z;
                       else if (nu > 1)
                           basis(n) = pow(y, (nu - 1) / 2) * z;
                       else
                           basis(n) = z;
                   }
                }
                n++;
            }
        }

        // Dot the coefficients in to our polynomial map
        return p.dot(basis);

    }

    // Return the flux in a cell
    template <typename T>
    T fcell(T r1, T r2, T t1, T t2, T xo, T yo, T ro, int lmax,
            const Vector<T>& p) {
        T numer = t1 + M_PI - t2;
        T denom = 2 * M_PI;
        T modulo = fmod(numer, denom);
        if (t1 + M_PI - t2 < 0) modulo += 2 * M_PI;
        T deltheta = abs(modulo - M_PI);
        return 0.125 * abs(r2 * r2 - r1 * r1) * deltheta *
               (evaluate<T>(r1 * cos(t1), r1 * sin(t1), xo, yo, ro, lmax, p) +
                evaluate<T>(r1 * cos(t2), r1 * sin(t2), xo, yo, ro, lmax, p) +
                evaluate<T>(r2 * cos(t1), r2 * sin(t1), xo, yo, ro, lmax, p) +
                evaluate<T>(r2 * cos(t2), r2 * sin(t2), xo, yo, ro, lmax, p));
    }

    // Return the numerically computed flux
    template <typename T>
    void fnum(T r1, T r2, T t1, T t2, T xo, T yo, T ro, T tol, int lmax,
              const Vector<T>& p, T* f) {
        // Coarse estimate
        T fcoarse = fcell<T>(r1, r2, t1, t2, xo, yo, ro, lmax, p);

        // Fine estimate (bisection)
        T r = 0.5 * (r1 + r2);
        T t = atan2(sin(t1) + sin(t2), cos(t1) + cos(t2));
        T ffine = (fcell<T>(r1, r, t1, t, xo, yo, ro, lmax, p) +
                   fcell<T>(r1, r, t, t2, xo, yo, ro, lmax, p) +
                   fcell<T>(r, r2, t1, t, xo, yo, ro, lmax, p) +
                   fcell<T>(r, r2, t, t2, xo, yo, ro, lmax, p));

        // Compare
        if (abs(fcoarse - ffine) > tol) {
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
    T flux(T xo, T yo, T ro, int lmax, const Vector<T>& p, T tol) {
        tol /= M_PI;
        T f = 0;
        T b = sqrt(xo * xo + yo * yo);
        T theta = atan2(yo, xo);
        T theta0 = theta - M_PI;
        if (theta0 < 0) theta0 += 2 * M_PI;
        else if (theta0 > 2 * M_PI) theta0 -= 2 * M_PI;
        T rmid, deltheta, theta1, theta2;
        if (b <= ro) {
            rmid = 0.95;
            deltheta = 0.25;
        } else if (b > 1 + ro) {
            rmid = 0.5;
            deltheta = 0.5;
        } else if (b > 1) {
            rmid = 0.5 * (1 + b - ro);
            deltheta = 0.95 * abs(acos((b * b - ro * ro + rmid * rmid) /
                       (2 * b * rmid)));
        } else {
            rmid = b;
            deltheta = 0.95 * abs(acos(1 - 0.5 * (ro * ro) / (b * b)));
        }
        theta1 = theta - deltheta;
        if (theta1 < 0) theta1 += 2 * M_PI;
        else if (theta1 > 0) theta1 -= 2 * M_PI;
        theta2 = theta + deltheta;
        if (theta2 < 0) theta2 += 2 * M_PI;
        else if (theta2 > 2 * M_PI) theta2 -= 2 * M_PI;


        // Compute the six segments
        // A
        fnum<T>(0, rmid, theta1, theta2, xo, yo, ro, tol, lmax, p, &f);
        // B
        fnum<T>(rmid, 1, theta1, theta2, xo, yo, ro, tol, lmax, p, &f);
        // C
        fnum<T>(0, rmid, theta0, theta1, xo, yo, ro, tol, lmax, p, &f);
        // D
        fnum<T>(rmid, 1, theta0, theta1, xo, yo, ro, tol, lmax, p, &f);
        // E
        fnum<T>(0, rmid, theta0, theta2, xo, yo, ro, tol, lmax, p, &f);
        // F
        fnum<T>(rmid, 1, theta0, theta2, xo, yo, ro, tol, lmax, p, &f);

        // We're done
        return f;
    }

} // namespace numeric

#endif
