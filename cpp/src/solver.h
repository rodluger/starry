/**
Spherical harmonic integration utilities.

*/

#ifndef _STARRY_INTEGRATE_H_
#define _STARRY_INTEGRATE_H_

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include "ellip.h"

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

    // Compute the n=2 term of the *s^T* occultation solution vector.
    // This is the Mandel & Agol solution for linear limb darkening,
    // reparametrized for speed
    template <typename T>
    T s2(Greens<T>& G) {
        T Lambda;
        T xi = 2 * G.br * (4 - 7 * G.r(2) - G.b2);
        T rpb = G.r(1) + G.b;
        T rmb = G.r(1) - G.b;
        if (G.b == 0) {
            Lambda = -2. / 3. * pow(1. - G.r(2), 1.5);
        } else if (G.b == G.r(1)) {
            if (G.r(1) == 0.5)
                Lambda = (1. / 3.) - 4. / (9. * M_PI);
            else if (G.r(1) < 0.5)
                Lambda = (1. / 3.) +
                         2. / (9. * M_PI) * (4. * (2. * G.r(2) - 1.) * ellip::E(4 * G.r(2)) +
                         (1 - 4 * G.r(2)) * ellip::K(4 * G.r(2)));
            else
                Lambda = (1. / 3.) +
                         16. * G.r(1) / (9. * M_PI) * (2. * G.r(2) - 1.) * ellip::E(1. / (4 * G.r(2))) -
                         (1 - 4 * G.r(2)) * (3 - 8 * G.r(2)) / (9 * M_PI * G.r(1)) * ellip::K(1. / (4 * G.r(2)));
        } else {
            if (G.ksq < 1)
                Lambda = ((rpb * rpb - 1.) / rpb * (-2 * G.r(1) * (2 * rpb * rpb + rpb * rmb - 3.) * G.K -
                         3. * rmb * G.PI) - 2. * xi * G.E) / (9. * M_PI * sqrt(G.br));
            else if (G.ksq > 1)
                Lambda = 2. * ((1 - rpb * rpb) * (sqrt(1. - rmb * rmb) * G.K - 3 * rmb / rpb * G.PI) -
                         sqrt(1. - rmb * rmb) * (4 - 7 * G.r(2) - G.b2) * G.E) / (9. * M_PI);
            else
                Lambda = 2. / (3. * M_PI) * acos(1. - 2 * G.r(1)) -
                         4 / (9 * M_PI) * (3 + 2 * G.r(1) - 8 * G.r(2)) * sqrt(G.br) -
                         2. / 3. * step(G.r(1) - 0.5);
        }
        return (2. * M_PI / 3.) * (1 - 1.5 * Lambda - step(G.r(1) - G.b));
    }

    // Compute the primitive integral helper matrix H
    template <typename T>
    T computeH(Greens<T>& G, int u, int v) {
        if (!is_even(u)) {
            return 0;
        } else if ((u == 0) && (v == 0)) {
            return 2 * asin(G.sinlam(1)) + M_PI;
        } else if ((u == 0) && (v == 1)) {
            return -2 * G.coslam(1);
        } else if (u >= 2) {
            return (2 * G.coslam(u - 1) * G.sinlam(v + 1) + (u - 1) * G.H.value(u - 2, v)) / (u + v);
        } else {
            return (-2 * G.coslam(u + 1) * G.sinlam(v - 1) + (v - 1) * G.H.value(u, v - 2)) / (u + v);
        }
    }

    // Compute the primitive integral helper matrix I
    template <typename T>
    T computeI(Greens<T>& G, int u, int v) {
        if (!is_even(u)) {
            return 0;
        } else if ((u == 0) && (v == 0)) {
            return 2 * asin(G.sinphi(1)) + M_PI;
        } else if ((u == 0) && (v == 1)) {
            return -2 * G.cosphi(1);
        } else if (u >= 2) {
            return (2 * G.cosphi(u - 1) * G.sinphi(v + 1) + (u - 1) * G.I.value(u - 2, v)) / (u + v);
        } else {
            return (-2 * G.cosphi(u + 1) * G.sinphi(v - 1) + (v - 1) * G.I.value(u, v - 2)) / (u + v);
        }
    }

    // Compute the primitive integral helper matrix J
    template <typename T>
    T computeJ(Greens<T>& G, int u, int v) {
        T res = 0;
        if (G.b == 0) {
            return pow(1 - G.r(2), 1.5) * G.I.value(u, v);
        } else {
            for (int i = 0; i < v + 1; i++)
                if (is_even(i - v - u))
                    res += fact::choose(v, i) * G.M.value(u + 2 * i, u + 2 * v - 2 * i);
                else
                    res -= fact::choose(v, i) * G.M.value(u + 2 * i, u + 2 * v - 2 * i);
            res *= pow(2, u + 3) * (G.br32);
        }
        return res;
    }

    // Compute the primitive integral helper matrix M
    template <typename T>
    T computeM(Greens<T>& G, int p, int q) {
        if (!is_even(p) || !is_even(q)) {
            return 0;
        } else if ((p == 0) && (q == 0)) {
            return ((8 - 12 * G.ksq) * G.E1 + (-8 + 16 * G.ksq) * G.E2) / 3.;
        } else if ((p == 0) && (q == 2)) {
            return ((8 - 24 * G.ksq) * G.E1 + (-8 + 28 * G.ksq + 12 * G.ksq * G.ksq) * G.E2) / 15.;
        } else if ((p == 2) && (q == 0)) {
            return ((32 - 36 * G.ksq) * G.E1 + (-32 + 52 * G.ksq - 12 * G.ksq * G.ksq) * G.E2) / 15.;
        } else if ((p == 2) && (q == 2)) {
            return ((32 - 60 * G.ksq + 12 * G.ksq * G.ksq) * G.E1 + (-32 + 76 * G.ksq - 36 * G.ksq * G.ksq + 24 * G.ksq * G.ksq * G.ksq) * G.E2) / 105.;
        } else if (q >= 4) {
            double d1, d2;
            d1 = q + 2 + (p + q - 2) * (1 - G.ksq);
            d2 = (3 - q) * (1 - G.ksq);
            return (d1 * G.M.value(p, q - 2) + d2 * G.M.value(p, q - 4)) / (p + q + 3);
        } else if (p >= 4) {
            double d3, d4;
            d3 = 2 * p + q - (p + q - 2) * (1 - G.ksq);
            d4 = (3 - p) + (p - 3) * (1 - G.ksq);
            return (d3 * G.M.value(p - 2, q) + d4 * G.M.value(p - 4, q)) / (p + q + 3);
        } else {
            std::cout << "ERROR: Domain error in function M()." << std::endl;
            exit(1);
        }
    }

    // The helper primitive integral K_{u,v}
    template <typename T>
    T K(Greens<T>& G, int u, int v) {
        T res = 0;
        for (int i = 0; i < v + 1; i++)
            res += fact::choose(v, i) * G.b_r(v - i) * G.I.value(u, i);
        return res;
    }

    // The helper primitive integral L_{u,v}
    template <typename T>
    T L(Greens<T>& G, int u, int v) {
        T res = 0;
        for (int i = 0; i < v + 1; i++)
            res += fact::choose(v, i) * G.b_r(v - i) * G.J.value(u, i);
        return res;
    }

    // The primitive integral P(G_n)
    template <typename T>
    T P(Greens<T>& G){
        if (is_even(G.nu))
            return G.r(G.l + 2) * K(G, (G.mu + 4) / 2, G.nu / 2);
        else if ((G.mu == 1) && is_even(G.l))
            return G.b * G.r(G.l - 2) * L(G, G.l - 2, 0) - G.r(G.l - 1) * L(G, G.l - 2, 1);
        else if ((G.mu == 1) && !is_even(G.l))
            return G.b * G.r(G.l - 2) * L(G, G.l - 3, 1) - G.r(G.l - 1) * L(G, G.l - 3, 2);
        else
            return G.r(G.l - 1) * L(G, (G.mu - 1) / 2, (G.nu - 1) / 2);
    }

    // The primitive integral Q(G_n)
    template <typename T>
    T Q(Greens<T>& G){
        if (is_even(G.nu))
            return G.H.value((G.mu + 4) / 2, G.nu / 2);
        else
            return 0;
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

        // Resetter
        void reset() {
            set.setZero(N, N);
        }

    };

    // Greens integration housekeeping data
    template <class T>
    class Greens {

    public:

        int lmax;
        int N;
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
            r(0) = 1;
            b_r.resize(N);
            b_r(0) = 1;
            cosphi.resize(N);
            cosphi(0) = 1;
            sinphi.resize(N);
            sinphi(0) = 1;
            coslam.resize(N);
            coslam(0) = 1;
            sinlam.resize(N);
            sinlam(0) = 1;
            sT.resize((lmax + 1) * (lmax + 1));

        }

    };

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
    void computesT(Greens<T>& G, T b, T r) {

        // Initialize the housekeeping variables
        int l, m;
        int n = 0;
        G.H.reset();
        G.I.reset();
        G.J.reset();
        G.M.reset();
        T sinphi, cosphi, sinlam, coslam;
        T b_r = b / r;
        G.b = b;
        G.b2 = b * b;
        G.br = b * r;
        G.br32 = pow(G.br, 1.5);
        if ((std::abs(1 - r) < b) && (b < 1 + r)) {
            // sin(arcsin(x)) = x
            // cos(arcsin(x)) = sqrt(1 - x * x)
            sinphi = (1 - r * r - G.b2) / (2 * G.br);
            cosphi = sqrt(1 - sinphi * sinphi);
            sinlam = (1 - r * r + G.b2) / (2 * G.b);
            coslam = sqrt(1 - sinlam * sinlam);
        } else {
            sinphi = 1;
            cosphi = 0;
            sinlam = 1;
            coslam = 0;
        }
        for (l = 1; l < G.N; l++) {
            G.r(l) = r * G.r(l - 1);
            G.b_r(l) = b_r * G.b_r(l - 1);
            G.cosphi(l) = cosphi * G.cosphi(l - 1);
            G.sinphi(l) = sinphi * G.sinphi(l - 1);
            G.coslam(l) = coslam * G.coslam(l - 1);
            G.sinlam(l) = sinlam * G.sinlam(l - 1);
        }

        // Compute the elliptic integrals
        G.ksq = (1 - G.r(2) - G.b * G.b + 2 * G.br) / (4 * G.br);
        G.k = sqrt(G.ksq);
        if (G.b == 0) {
            // We don't need elliptic integrals in this case!
            G.K = 0;
            G.E = 0;
            G.PI = 0;
            G.E1 = 0;
            G.E2 = 0;
        } else if (G.ksq == 1) {
            // Special case
            G.K = 0;
            G.E = 1;
            G.PI = 0;
            G.E1 = 0;
            G.E2 = 1;
        } else {
            if (G.ksq < 1) {
                G.K = ellip::K(G.ksq);
                G.E = ellip::E(G.ksq);
                G.PI = ellip::PI(1 - 1. / ((G.b - G.r(1)) * (G.b - G.r(1))), G.ksq);
                G.E1 = (1 - G.ksq) * G.K;
                G.E2 = G.E;
            } else {
                G.K = ellip::K(1. / G.ksq);
                G.E = ellip::E(1. / G.ksq);
                G.PI = ellip::PI(1. / (G.ksq - 1. / (4 * G.br)), 1. / G.ksq);
                G.E1 = (1 - G.ksq) / G.k * G.K;
                G.E2 = G.k * G.E + (1 - G.ksq) / G.k * G.K;
            }
        }

        // Populate the vector
        for (l = 0; l < G.lmax + 1; l++) {
            G.l = l;
            for (m = -l; m < l + 1; m++) {
                G.m = m;
                G.mu = l - m;
                G.nu = l + m;
                if ((l == 1) && (m == 0))
                    G.sT(n) = s2(G);
                else
                    G.sT(n) = Q(G) - P(G);
                n++;
            }
        }

    }

}; // namespace solver

#endif
