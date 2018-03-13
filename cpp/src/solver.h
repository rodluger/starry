/**
Spherical harmonic integration utilities.

*/

#ifndef _STARRY_INTEGRATE_H_
#define _STARRY_INTEGRATE_H_

#include <iostream>
#include <cmath>
#include <Eigen/Core>

template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T>
using VectorT = Eigen::Matrix<T, 1, Eigen::Dynamic>;

namespace solver {

    // Heaviside step function
    template <typename T>
    T step(T x) {
        if (x <= 0)
            return 0;
        else
            return 1;
    }

    // Check if number is even (or doubly, triply, quadruply... even)
    bool is_even(int n, int ntimes=1) {
        for (int i = 0; i < ntimes; i++) {
            if ((n % 2) == 1) return false;
            n /= 2;
        }
        return true;
    }

    // Return the n^th term of the *r* phase curve solution vector.
    double rn(int mu, int nu) {
            double a, b, c;
            if (is_even(mu, 2) && is_even(nu, 2)) {
                a = fact::gamma_sup(mu / 4);
                b = fact::gamma_sup(nu / 4);
                c = fact::gamma((mu + nu) / 4 + 2);
                return a * b / c;
            } else if (is_even(mu - 1, 2) && is_even(nu - 1, 2)) {
                a = fact::gamma_sup((mu - 1) / 4);
                b = fact::gamma_sup((nu - 1) / 4);
                c = fact::gamma_sup((mu + nu - 2) / 4 + 2) * M_2_SQRTPI;
                return a * b / c;
            } else {
                return 0;
            }
    }

    // Compute the *r^T* phase curve solution vector
    void computerT(int lmax, VectorT<double>& rT) {
        rT.resize((lmax + 1) * (lmax + 1));
        int l, m, mu, nu;
        int n = 0;
        for (l=0; l<lmax+1; l++) {
            for (m=-l; m<l+1; m++) {
                mu = l - m;
                nu = l + m;
                rT(n) = rn(mu, nu);
                n++;
            }
        }
        return;
    }

}; // namespace solver

#endif
