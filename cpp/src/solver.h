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

    // Forward declarations
    template <class T>
    class Primitive;
    template <class T>
    class Greens;

    // Compute the primitive integral helper matrix H
    template <typename T>
    T computeH(Greens<T>& G, int i, int j) {
        return 0; // TODO
    }

    // Compute the primitive integral helper matrix I
    template <typename T>
    T computeI(Greens<T>& G, int i, int j) {
        return 0; // TODO
    }

    // Compute the primitive integral helper matrix J
    template <typename T>
    T computeJ(Greens<T>& G, int i, int j) {
        return 0; // TODO
    }

    // Compute the primitive integral helper matrix M
    template <typename T>
    T computeM(Greens<T>& G, int i, int j) {
        return 0; // TODO
    }

    // Primitive integral helper matrices
    template <class T>
    class Primitive {

        int lmax;
        int N;
        Matrix<bool> set;
        Matrix<T> matrix;
        T (*setter)(Greens<T>&, int, int);
        Greens<T>& G;

    public:

        // Constructor
        Primitive(Greens<T>& G, int lmax, T (*setter)(Greens<T>&, int, int)) : lmax(lmax), setter(setter), G(G) {
            N = 2 * lmax + 1;
            set = Matrix<bool>::Zero(N, N);
            matrix.resize(N, N);
        }

        // Getter function. G is a pointer to the current Greens struct, and setter
        // is a pointer to the function that computes the (i, j) element
        // of this primitive matrix
        T value(int i, int j) {
            if (!set(i, j)) {
                matrix(i, j) = (*setter)(G, i, j);
                set(i, j) = true;
            }
            return matrix(i, j);
        }

    };

    // Greens integration housekeeping data
    template <class T>
    class Greens {

        int lmax;
        int N;

    public:

        int l;
        int m;
        int mu;
        int nu;

        T b;
        T b2;
        T br;
        T br32;
        T ksq;
        T k;
        T E;
        T K;
        T PI;
        T E1;
        T E2;

        Vector<T> r;
        Vector<T> b_r;
        Vector<T> cosphi;
        Vector<T> sinphi;
        Vector<T> coslam;
        Vector<T> sinlam;

        Primitive<T> H;
        Primitive<T> I;
        Primitive<T> J;
        Primitive<T> M;

        VectorT<T> sT;

        // Constructor
        Greens(int lmax) : lmax(lmax),
                           H(*this, lmax, computeH),
                           I(*this, lmax, computeI),
                           J(*this, lmax, computeJ),
                           M(*this, lmax, computeM) {

            // Initialize some stuff
            N = 2 * lmax + 1;
            if (N < 2) N = 2;
            r.resize(N);
            b_r.resize(N);
            cosphi.resize(N);
            sinphi.resize(N);
            coslam.resize(N);
            sinlam.resize(N);
            sT.resize((lmax + 1) * (lmax + 1));

        }

    };

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

    // Compute the *s^T* occultation solution vector
    template <typename T>
    void computesT(Greens<T>& G) {
        // TODO
    }

}; // namespace solver

#endif
