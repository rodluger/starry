/**
Spherical harmonic, polynomial, and Green's basis utilities.

*/

#ifndef _STARRY_BASIS_H_
#define _STARRY_BASIS_H_

#include <cmath>
#include <Eigen/Core>
#include <pybind11/eigen.h>

namespace basis {

using std::abs;

    // A surface map vector class
    template <class T>
    class Map {
        int lmax, N;
        Eigen::Matrix<T, Eigen::Dynamic, 1> map;
        Eigen::Matrix<T, Eigen::Dynamic, 1> basis;
    public:
        // Constructor: initialize map to zeros
        Map(int lmax) : lmax(lmax) {
            N = (lmax + 1) * (lmax + 1);
            map = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(N);
            basis.resize(N, 1);
        }
        // Constructor: initialize map to array
        Map(Eigen::Matrix<T, Eigen::Dynamic, 1>& map) : map(map) {
            lmax = floor(sqrt((double)map.size()) - 1);
            N = (lmax + 1) * (lmax + 1);
            basis.resize(N, 1);
        }
        T evaluate(const T& x, const T&y);
    };

    // Evaluate a polynomial vector at a given (x, y) coordinate
    template <class T>
    T Map<T>::evaluate (const T& x, const T&y) {

        // Check if outside the sphere
        if (x * x + y * y > 1.0) return NAN;

        int l, m, mu, nu, n = 0;
        T z = sqrt(1.0 - x * x - y * y);

        // Compute the basis
        for (l=0; l<lmax+1; l++) {
            for (m=-l; m<l+1; m++) {
                mu = l - m;
                nu = l + m;
                if ((nu % 2) == 0)
                    basis(n) = pow(x, mu / 2) * pow(y, nu / 2);
                else
                    basis(n) = pow(x, (mu - 1) / 2) * pow(y, (nu - 1) / 2) * z;
                n++;
            }
        }

        // Dot the coefficients in
        return map.dot(basis);

    }

    // Factorial function (TODO!!!)
    template <typename T>
    double factorial(T n) {
        return 0;
    }

    // Contraction coefficient for the Ylms (Equation 10)
    double C(int p, int q, int k) {
        if ((p > k) && ((p - k) % 2 == 0))
            return 0;
        else if ((q > p) && ((q - p) % 2 == 0))
            return 0;
        else
            return factorial(0.5 * k) / (factorial(0.5 * q) *
                                         factorial(0.5 * (k - p)) *
                                         factorial(0.5 * (p - q)));
    }


}; // namespace basis

#endif
