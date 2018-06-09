/**
Spherical harmonic integration utilities.

*/

#ifndef _STARRY_SOLVER_H_
#define _STARRY_SOLVER_H_

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <vector>
#include "constants.h"
#include "ellip.h"
#include "errors.h"
#include "lld.h"
#include "utils.h"
#include "tables.h"

namespace solver {

    using std::abs;
    using std::max;
    using std::vector;

    // Forward declarations
    template <class T>
    class Primitive;
    template <class T>
    class Greens;

    // Return the n^th term of the *r* phase curve solution vector
    template <typename T>
    T rn(int mu, int nu, const T& twosqrtpi) {
            T a, b, c;
            if (is_even(mu, 2) && is_even(nu, 2)) {
                a = tables::gamma_sup<T>(mu / 4);
                b = tables::gamma_sup<T>(nu / 4);
                c = tables::gamma<T>((mu + nu) / 4 + 2);
                return a * b / c;
            } else if (is_even(mu - 1, 2) && is_even(nu - 1, 2)) {
                a = tables::gamma_sup<T>((mu - 1) / 4);
                b = tables::gamma_sup<T>((nu - 1) / 4);
                c = tables::gamma_sup<T>((mu + nu - 2) / 4 + 2) * twosqrtpi;
                return a * b / c;
            } else {
                return 0;
            }
    }

    // Compute the *r^T* phase curve solution vector
    template <typename T>
    void computerT(int lmax, VectorT<T>& rT) {
        rT.resize((lmax + 1) * (lmax + 1));
        T twosqrtpi = 2.0 / sqrt(T(BIGPI));
        int l, m, mu, nu;
        int n = 0;
        for (l=0; l<lmax+1; l++) {
            for (m=-l; m<l+1; m++) {
                mu = l - m;
                nu = l + m;
                rT(n) = rn<T>(mu, nu, twosqrtpi);
                n++;
            }
        }
        return;
    }

    // Fast powers of a variable
    template <class T>
    class Power {

            vector<T> vec;

        public:

            // Constructor
            Power(T val) {
                vec.push_back(1.0 + (val * 0));
                vec.push_back(val);
            }

            // Getter function
            inline T value(int n) {
                if (n < 0) {
                    throw errors::BadIndex();
                }
                while (n >= vec.size()) {
                    vec.push_back(vec[1] * vec[vec.size() - 1]);
                }
                return vec[n];
            }

            // Overload () to get the function value without calling value()
            inline T operator() () { return value(1); }
            inline T operator() (int n) { return value(n); }

            // Resetter
            void reset(T val) {
                vec.clear();
                vec.push_back(1.0 + (val * 0));
                vec.push_back(val);
            }

    };

    // Compute the n=2 term of the *s^T* occultation solution vector.
    // This is the Mandel & Agol solution for linear limb darkening,
    // reparametrized for speed
    template <typename T>
    inline T s2(Greens<T>& G) {
        T b = G.b();
        T r = G.r();
        T ksq = G.ksq();
        T K = G.ELL.K();
        T E = G.ELL.E();
        return lld::s2(b, r, ksq, K, E, G.pi);
    }

    // Compute the flux for a transit of a quadratically limb-darkened star
    // This code has been stripped of a lot of the overhead for speed, so
    // it may be a bit opaque. Basically, for a quadratically limb-darkened star,
    // the only terms that matter in the Greens polynomial basis are those at
    // indices n = 0, 2, 4, and 8. We therefore only compute those indices of the
    // solution vector -- we do it directly, without any recurrence relations.
    // Note, importantly, that the term g(4) is *always* 1/3 * g(8), so we fold
    // that into `s8` below.
    template <typename T>
    inline T QuadLimbDark(Greens<T>& G, const T& b, const T& r, T& g0, T& g2, T& g8) {

        // Initialize only the necessary variables
        T s0, s8;
        G.br = b * r;
        G.b.reset(b);
        G.r.reset(r);
        G.delta.reset((b - r) / (2 * r));
        G.ksq.reset((1 - G.r(2) - G.b(2) + 2 * G.br) / (4 * G.br));
        G.k = sqrt(G.ksq());
        G.ELL.reset();

        if ((abs(1 - r) < b) && (b < 1 + r)) {
            G.sinphi.reset((1 - G.r(2) - G.b(2)) / (2 * G.br));
            G.cosphi.reset(sqrt(1 - G.sinphi() * G.sinphi()));
            G.sinlam.reset((1 - G.r(2) + G.b(2)) / (2 * G.b()));
            G.coslam.reset(sqrt(1 - G.sinlam() * G.sinlam()));
            G.phi = asin(G.sinphi());
            G.lam = asin(G.sinlam());
            s0 = G.lam + G.pi_over_2 + G.sinlam() * G.coslam() -
                 G.r(2) * (G.phi + G.pi_over_2 + G.sinphi() * G.cosphi());
            s8 = 0.5 * (G.pi_over_2 + G.lam) + (1. / 3.) * G.coslam() * G.sinlam() -
                 (1. / 6.) * G.coslam(3) * G.sinlam() + (1. / 6.) * G.coslam() * G.sinlam(3) -
                 (G.r(2) * G.b(2) * (G.pi_over_2 + G.phi + G.cosphi() * G.sinphi()) -
                  G.r(3) * G.b() * G.cosphi() * (1. + (1. / 3.) * G.cosphi(2) - G.sinphi(2)) +
                  G.r(4) * (0.5 * (G.pi_over_2 + G.phi) + (1. / 3.) * G.cosphi() * G.sinphi() -
                            (1. / 6.) * G.cosphi(3) * G.sinphi() + (1. / 6.) * G.cosphi() * G.sinphi(3)));
        } else {
            G.sinphi.reset(1);
            G.cosphi.reset(0);
            G.sinlam.reset(1);
            G.coslam.reset(0);
            G.phi = 0.5 * G.pi;
            G.lam = 0.5 * G.pi;
            s0 = G.pi * (1 - G.r(2));
            s8 = G.pi_over_2 - G.pi * G.r(2) * (0.5 * G.r(2) + G.b(2));
        }

        return s0 * g0 + s2(G) * g2 + s8 * g8;

    }

    // Compute the primitive integral helper matrix H
    template <typename T>
    inline T computeH(Greens<T>& G, int u, int v) {
        if (!is_even(u)) {
            return 0;
        } else if (G.off_limb && !is_even(v)) {
            // By symmetry, from the integral definition of H
            return 0;
        } else if ((u == 0) && (v == 0)) {
            return 2 * G.lam + G.pi;
        } else if ((u == 0) && (v == 1)) {
            return -2 * G.coslam(1);
        } else if (u >= 2) {
            return (2 * G.coslam(u - 1) * G.sinlam(v + 1) + (u - 1) * G.H(u - 2, v)) / (u + v);
        } else {
            return (-2 * G.coslam(u + 1) * G.sinlam(v - 1) + (v - 1) * G.H(u, v - 2)) / (u + v);
        }
    }

    // The helper primitive integral K_{u,v}
    template <typename T>
    inline T K(Greens<T>& G, int u, int v) {
        T res = 0;
        for (int i = 0; i < v + 1; i++)
            res += tables::choose<T>(v, i) * G.b_r(v - i) * G.I(u, i);
        return res;
    }

    // The helper primitive integral L_{u,v}
    template <typename T>
    inline T L(Greens<T>& G, int u, int v) {
        T res = 0;
        for (int i = 0; i < v + 1; i++)
            res += tables::choose<T>(v, i) * G.b_r(v - i) * G.J(u, i);
        return res;
    }

    // Compute the double-binomial coefficient A_{i,u,v}
    template <typename T>
    inline T computeAiuv(Greens<T>& G, int i, int u, int v) {
        T res = 0;
        int j1 = u - i;
        if (j1 < 0) j1 = 0;
        int j2 = u + v - i;
        if (j2 > u) j2 = u;
        int c;
        for (int j = j1; j <= j2; j++) {
            c = u + v - i - j;
            if (c < 0)
                break;
            if (is_even(u + j))
                res += tables::choose<T>(u, j) * tables::choose<T>(v, c) * G.delta(c);
            else
                res -= tables::choose<T>(u, j) * tables::choose<T>(v, c) * G.delta(c);
        }
        return res;
    }

    // The Taylor expansion of J near b = 0 (k^2 = \infty)
    template <typename T>
    inline T taylorJ(Greens<T>& G, int v){
        T res = 0;
        int n = 0;
        int sign = 1;
        // TODO: convergence criterion
        while (n < 30) {
            res += sign / (tables::factorial<T>(n) * tables::half_factorial<T>(3 - 2 * n))
                        * (tables::gamma_sup<T>(n + v) / tables::gamma<T>(n + v + 1))
                        / G.ksq(n);
            sign *= -1;
            n++;
        }
        // Note that sqrt(pi) * (3/2)! = 3 * pi / 4
        res *= 0.75 * G.pi;
        return res;
    }

    // The Taylor expansion of P near b = 0 (k^2 = \infty)
    template <typename T>
    inline T taylorP(Greens<T>& G){
        T res = 0;
        int a, b;
        if ((G.mu == 1) && is_even(G.l)) {
            a = (G.l - 2) / 2;
            b = 0;
            for (int i = 0; i <= a + b; i++)
                res += 2 * computeAiuv(G, i, a, b) * taylorJ(G, a + i + 1)
                         - computeAiuv(G, i, a, b) * taylorJ(G, a + i);
            res *= -pow(2 * G.r(), G.l - 1) * pow(1 - (G.b() - G.r()) * (G.b() - G.r()), 1.5);
        } else if ((G.mu == 1) && !is_even(G.l)) {
            a = (G.l - 3) / 2;
            b = 1;
            for (int i = 0; i <= a + b; i++)
                res += 2 * computeAiuv(G, i, a, b) * taylorJ(G, a + i + 1)
                         - computeAiuv(G, i, a, b) * taylorJ(G, a + i);
            res *= -pow(2 * G.r(), G.l - 1) * pow(1 - (G.b() - G.r()) * (G.b() - G.r()), 1.5);
        } else {
            a = (G.mu - 1) / 4;
            b = (G.nu - 1) / 2;
            for (int i = 0; i <= a + b; i++)
                res += computeAiuv(G, i, a, b) * taylorJ(G, a + i);
            res *= 2 * pow(2 * G.r(), G.l - 1) * pow(1 - (G.b() - G.r()) * (G.b() - G.r()), 1.5);
        }
        return res;
    }

    // The primitive integral P(G_n)
    template <typename T>
    inline T P(Greens<T>& G){
        if (is_even(G.mu, 2)) {
            return G.r(G.l + 2) * K(G, (G.mu + 4) / 2, G.nu / 2);
        } else if ((G.mu == 1) && is_even(G.l)) {
            if (J_unstable(G))
                return taylorP(G);
            else
                return -G.r(G.l - 1) * G.J(G.l - 2, 1);
        } else if ((G.mu == 1) && !is_even(G.l)) {
            if (J_unstable(G))
                return taylorP(G);
            else
                return -G.r(G.l - 2) * (G.b() * G.J(G.l - 3, 1) + G.r() * G.J(G.l - 3, 2));
        } else if (is_even(G.mu - 1, 2)) {
            if (J_unstable(G))
                return taylorP(G);
            else
                return G.r(G.l - 1) * L(G, (G.mu - 1) / 2, (G.nu - 1) / 2);
        } else {
            return 0;
        }
    }

    // The primitive integral Q(G_n)
    template <typename T>
    inline T Q(Greens<T>& G){
        // From the integral definition of Q, the result is zero
        // unless both mu/2 and nu/2 are even when the occultor
        // is not touching the limb of the planet.
        if (G.off_limb && (!is_even(G.mu, 2) || !is_even(G.nu, 2)))
            return 0;
        else if (is_even(G.mu, 2))
            return G.H((G.mu + 4) / 2, G.nu / 2);
        else
            return 0;
    }

    // Elliptic integral storage class
    template <class T>
    class Elliptic {

            T vK;
            T vE;
            T vE1;
            T vE2;
            bool bK;
            bool bE;
            bool bE1;
            bool bE2;
            Greens<T>& G;

        public:

            // Constructor
            Elliptic(Greens<T>& G) : G(G) {
                reset();
            }

            // Elliptic integral of the first kind
            inline T K() {
                if (!bK) {
                    if ((G.b() == 0) || (G.ksq() == 1))
                        vK = 0;
                    else if (G.ksq() < 1)
                        vK = ellip::K(G.ksq());
                    else
                        vK = ellip::K(T(1. / G.ksq()));
                    bK = true;
                }
                return vK;
            }

            // Elliptic integral of the second kind
            inline T E() {
                if (!bE) {
                    if (G.b() == 0)
                        vE = 0;
                    else if (G.ksq() == 1)
                        vE = 1;
                    else if (G.ksq() < 1)
                        vE = ellip::E(G.ksq());
                    else
                        vE = ellip::E(T(1. / G.ksq()));
                    bE = true;
                }
                return vE;
            }

            // First elliptic function
            inline T E1() {
                if (!bE1) {
                    if ((G.b() == 0) || (G.ksq() == 1))
                        vE1 = 0;
                    else if (G.ksq() < 1)
                        vE1 = (1 - G.ksq()) * K();
                    else
                        vE1 = (1 - G.ksq()) / G.k * K();
                    bE1 = true;
                }
                return vE1;
            }

            // Second elliptic function
            inline T E2() {
                if (!bE2) {
                    if (G.b() == 0)
                        vE2 = 0;
                    else if (G.ksq() == 1)
                        vE2 = 1;
                    else if (G.ksq() < 1)
                        vE2 = E();
                    else
                        vE2 = G.k * E() + (1 - G.ksq()) / G.k * K();
                    bE2 = true;
                }
                return vE2;
            }

            // Resetter
            void reset() {
                bK = false;
                bE = false;
                bE1 = false;
                bE2 = false;
            }

    };

    // Primitive integral storage class
    template <class T>
    class Primitive {

            Matrix<bool> set;
            Matrix<T> matrix;
            int rows;
            int cols;
            T (*setter)(Greens<T>&, int, int);
            Greens<T>& G;

        public:

            // Constructor
            Primitive(Greens<T>& G, T (*setter)(Greens<T>&, int, int), int rows, int cols=1) : setter(setter), rows(rows), cols(cols), G(G) {
                set = Matrix<bool>::Zero(rows, cols);
                matrix.resize(rows, cols);
            }

            // Getter function. G is a pointer to the current Greens struct,
            // and setter is a pointer to the function that computes the
            // (i, j) element of this primitive matrix
            inline T value(int i, int j=0) {
                if ((i < 0) || (j < 0) || (i > rows - 1) || (j > cols - 1)) {
                    throw errors::BadIndex();
                }
                if (!set(i, j)) {
                    matrix(i, j) = (*setter)(G, i, j);
                    set(i, j) = true;
                }
                return matrix(i, j);
            }

            // Overload () to get the function value without calling value()
            inline T operator() (int i, int j=0) { return value(i, j); }

            // Resetter
            void reset() {
                set.setZero(rows, cols);
            }

    };



    // Greens integration housekeeping data
    template <class T>
    class Greens {

        public:

            // Indices
            int lmax;
            int l;
            int m;
            int mu;
            int nu;

            // Occultor off the limb of the body (simpler formulae in this case)?
            bool off_limb;

            // Some basic variables
            T br;
            T br32;
            T k;
            T phi;
            T lam;

            // Powers of basic variables
            Power<T> ksq;
            Power<T> b;
            Power<T> r;
            Power<T> b_r;
            Power<T> cosphi;
            Power<T> sinphi;
            Power<T> coslam;
            Power<T> sinlam;
            Power<T> delta;
            Power<T> F;

            // Elliptic integrals
            Elliptic<T> ELL;

            // Primitive matrices/vectors
            Primitive<T> H;
            Primitive<T> I;
            Primitive<T> J;

            // The solution vector
            VectorT<T> sT;

            // The value of pi, computed at
            // the user-requested precision
            T pi;
            T pi_over_2;

            // Constructor
            Greens(int lmax) :
                   lmax(lmax),
                   off_limb(false),
                   ksq(0),
                   b(0),
                   r(0),
                   b_r(0),
                   cosphi(0),
                   sinphi(0),
                   coslam(0),
                   sinlam(0),
                   delta(0),
                   F(0),
                   ELL(*this),
                   H(*this, computeH, lmax + 3, lmax + 1),
                   I(*this, computeI, 2 * lmax + 3),
                   J(*this, computeJ, 2 * lmax) {

                // Initialize the solution vector
                sT = VectorT<T>::Zero((lmax + 1) * (lmax + 1));

                // Compute pi at the actual precision of the T type
                pi = T(BIGPI);
                pi_over_2 = T(0.5 * pi);

            }

    };

    // Compute the *s^T* occultation solution vector
    template <typename T>
    void computesT(Greens<T>& G, const T& b, const T& r, const Vector<T>& y) {

        // Initialize the basic variables
        int l, m;
        int n = 0;
        T ksq, k;
        G.br = b * r;
        G.br32 = pow(G.br, 1.5);
        G.b.reset(b);
        G.r.reset(r);
        G.delta.reset((b - r) / (2 * r));
        G.b_r.reset(b / r);
        if (r <= 1)
            ksq = (1 - G.r(2) - G.b(2) + 2 * G.br) / (4 * G.br);
        else
            ksq = (1 - (b - r)) * (1 + (b - r)) / (4 * G.br);
        k = sqrt(ksq);
        // Override the NaNs
        if ((b == 0) || (r == 0)) {
            set_derivs_to_zero(ksq);
            set_derivs_to_zero(k);
        }
        G.ksq.reset(ksq);
        G.k = k;
        if ((abs(1 - r) < b) && (b < 1 + r)) {
            G.off_limb = false;
            if (r <= 1) {
                G.sinphi.reset((1 - G.r(2) - G.b(2)) / (2 * G.br));
                G.cosphi.reset(sqrt(1 - G.sinphi() * G.sinphi()));
                G.sinlam.reset((1 - G.r(2) + G.b(2)) / (2 * G.b()));
                G.coslam.reset(sqrt(1 - G.sinlam() * G.sinlam()));
                G.phi = asin(G.sinphi());
                G.lam = asin(G.sinlam());
            } else {
                G.sinphi.reset(2 * (G.ksq() - 0.5));
                G.cosphi.reset(2 * G.k * sqrt(1 - G.ksq()));
                G.sinlam.reset(0.5 * ((1. / b) + (b - r) * (1. + r / b)));
                G.coslam.reset(sqrt(1 - G.sinlam() * G.sinlam()));
                G.phi = asin(G.sinphi());
                G.lam = asin(G.sinlam());
            }
        } else {
            G.off_limb = true;
            G.sinphi.reset(1);
            G.cosphi.reset(0);
            G.sinlam.reset(1);
            G.coslam.reset(0);
            G.phi = 0.5 * G.pi;
            G.lam = 0.5 * G.pi;
        }

        // Initialize our storage classes
        G.H.reset();
        G.I.reset();
        G.J.reset();
        G.M.reset();
        G.ELL.reset();

        // Populate the solution vector
        for (l = 0; l < G.lmax + 1; l++) {
            G.l = l;
            for (m = -l; m < l + 1; m++) {
                G.m = m;
                G.mu = l - m;
                G.nu = l + m;
                if (abs(y(n)) > 10 * mach_eps<T>()) {
                    if ((l == 1) && (m == 0))
                        G.sT(n) = s2(G);
                    // These terms are zero because they are proportional to
                    // odd powers of x, so we don't need to compute them!
                    else if ((is_even(G.mu - 1)) && (!is_even((G.mu - 1) / 2)))
                        G.sT(n) = 0;
                    // These terms are also zero for the same reason
                    else if ((is_even(G.mu)) && (!is_even(G.mu / 2)))
                        G.sT(n) = 0;
                    else
                        G.sT(n) = Q(G) - P(G);
                } else {
                    G.sT(n) = 0;
                }
                n++;
            }
        }
    }

}; // namespace solver

#endif
