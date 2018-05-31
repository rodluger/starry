/**
Spherical harmonic integration utilities.

*/

#ifndef _STARRY_INTEGRATE_H_
#define _STARRY_INTEGRATE_H_

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include "constants.h"
#include "ellip.h"
#include "fact.h"
#include "errors.h"
#include "lld.h"
#include "taylor.h"
#include "utils.h"

namespace solver {

    using std::abs;
    using std::max;
    using std::vector;

    // Forward declarations
    template <class T>
    class Primitive;
    template <class T>
    class Greens;

    // Helper function to figure out if we're using multiprecision
    template <typename T>
    inline bool is_bigdouble(T x) {
        return false;
    }

    // Helper function to figure out if we're using multiprecision
    template <>
    inline bool is_bigdouble(bigdouble x) {
        return true;
    }

    // Check if number is even (or doubly, triply, quadruply... even)
    inline bool is_even(int n, int ntimes=1) {
        for (int i = 0; i < ntimes; i++) {
            if ((n % 2) != 0) return false;
            n /= 2;
        }
        return true;
    }

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
        return lld::s2(b, r, ksq, K, E, G.pi, G.taylor);
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

    // Compute the primitive integral helper matrix I
    template <typename T>
    inline T computeI(Greens<T>& G, int u, int v) {
        if (!is_even(u)) {
            return 0;
        } else if ((u == 0) && (v == 0)) {
            return 2 * G.phi + G.pi;
        } else if ((u == 0) && (v == 1)) {
            return -2 * G.cosphi(1);
        } else if (u >= 2) {
            return (2 * G.cosphi(u - 1) * G.sinphi(v + 1) + (u - 1) * G.I(u - 2, v)) / (u + v);
        } else {
            return (-2 * G.cosphi(u + 1) * G.sinphi(v - 1) + (v - 1) * G.I(u, v - 2)) / (u + v);
        }
    }

    // Compute the primitive integral helper matrix J
    template <typename T>
    inline T computeJ(Greens<T>& G, int u, int v) {
        T res = 0;
        if (G.b() == 0) {
            // Special case
            return pow(1 - G.r(2), 1.5) * G.I(u, v);
        } else if ((G.taylor || is_bigdouble(G.b())) && (G.b() < STARRY_B_THRESH_J<T>(G.r()))) {
            // Normally we don't do any approximations/re-parametrizations when using multiprecision,
            // but it turns out that `J` is **extremely** unstable at very small impact parameter.
            // Instabilities in `J` take hold for `b` as high as 1e-6, even for 128 **digits** of precision
            // (that's ~ **hexadecuple** precision), so we Taylor expand when type `T` is `bigdouble`.
            return taylor::computeJ(G, u, v);
        } else {
            for (int i = 0; i < v + 1; i++) {
                if (is_even(i - v - u))
                    res += fact::choose(v, i) * G.M(u + 2 * i, u + 2 * v - 2 * i);
                else
                    res -= fact::choose(v, i) * G.M(u + 2 * i, u + 2 * v - 2 * i);
            }
            // Note that we multiply by the factor of (br)^1.5 inside computeM()
            // for small occultors and inside P() for large occultors.
            res *= pow(2, u + 3);
        }
        return res;
    }

    // Compute the primitive integral helper matrix M
    template <typename T>
    inline T computeM(Greens<T>& G, int p, int q) {
        if (!is_even(p) || !is_even(q)) {
            return 0;
        } else if ((G.taylor) && (G.r() > STARRY_RADIUS_THRESH_M)) {
            // Taylor expansion for large occultor
            return taylor::computeM(G, p, q);
        } else if ((p == 0) && (q == 0)) {
            return G.br32 * ((8 - 12 * G.ksq()) * G.ELL.E1() + (-8 + 16 * G.ksq()) * G.ELL.E2()) / 3.;
        } else if ((p == 0) && (q == 2)) {
            return G.br32 * ((8 - 24 * G.ksq()) * G.ELL.E1() + (-8 + 28 * G.ksq() + 12 * G.ksq(2)) * G.ELL.E2()) / 15.;
        } else if ((p == 2) && (q == 0)) {
            return G.br32 * ((32 - 36 * G.ksq()) * G.ELL.E1() + (-32 + 52 * G.ksq() - 12 * G.ksq(2)) * G.ELL.E2()) / 15.;
        } else if ((p == 2) && (q == 2)) {
            return G.br32 * ((32 - 60 * G.ksq() + 12 * G.ksq(2)) * G.ELL.E1() + (-32 + 76 * G.ksq() - 36 * G.ksq(2) + 24 * G.ksq(3)) * G.ELL.E2()) / 105.;
        } else if (q >= 4) {
            T d1, d2;
            T res1, res2;
            // Terms independent of ksq
            d1 = q + 2 + (p + q - 2);
            d2 = (3 - q);
            res1 = (d1 * G.M(p, q - 2) + d2 * G.M(p, q - 4)) / (p + q + 3);
            // Terms proportional to ksq
            d1 = (p + q - 2);
            d2 = (3 - q);
            res2 = (d1 * G.M(p, q - 2) + d2 * G.M(p, q - 4)) / (p + q + 3);
            res2 *= -G.ksq();
            // Add them
            return res1 + res2;
        } else if (p >= 4) {
            T d3, d4;
            T res1, res2;
            // Terms independent of ksq
            d3 = 2 * p + q - (p + q - 2);
            d4 = 0;
            res1 = (d3 * G.M(p - 2, q)) / (p + q + 3);
            // Terms proportional to ksq
            d3 = -(p + q - 2);
            d4 = (p - 3);
            res2 = (d3 * G.M(p - 2, q) + d4 * G.M(p - 4, q)) / (p + q + 3);
            res2 *= -G.ksq();
            // Add them
            return res1 + res2;
        } else throw errors::Domain();
    }

    // The helper primitive integral K_{u,v}
    template <typename T>
    inline T K(Greens<T>& G, int u, int v) {
        T res = 0;
        for (int i = 0; i < v + 1; i++)
            res += fact::choose(v, i) * G.b_r(v - i) * G.I(u, i);
        return res;
    }

    // The helper primitive integral L_{u,v}
    template <typename T>
    inline T L(Greens<T>& G, int u, int v) {
        T res = 0;
        for (int i = 0; i < v + 1; i++)
            res += fact::choose(v, i) * G.b_r(v - i) * G.J(u, i);
        return res;
    }

    // The primitive integral P(G_n)
    // Note that for large occultors, we multiply all the
    // terms here by (br)^1.5 instead of in the J matrix.
    template <typename T>
    inline T P(Greens<T>& G){
        T factor;
        if ((G.taylor) && (G.r() > STARRY_RADIUS_THRESH_M))
            factor = G.br32;
        else
            factor = 1;
        if (is_even(G.mu, 2)) {
            if ((G.taylor) && (G.r() > 1) && (STARRY_QUARTIC_APPROX(G.l, G.b(), G.r()))) {
                return taylor::P(G);
            } else {
                return G.r(G.l + 2) * K(G, (G.mu + 4) / 2, G.nu / 2);
            }
        } else if ((G.mu == 1) && is_even(G.l))
            return factor * -G.r(G.l - 1) * G.J(G.l - 2, 1);
        else if ((G.mu == 1) && !is_even(G.l))
            return factor * -G.r(G.l - 2) * (G.b() * G.J(G.l - 3, 1) + G.r() * G.J(G.l - 3, 2));
        else if (is_even(G.mu - 1, 2))
            return factor * G.r(G.l - 1) * L(G, (G.mu - 1) / 2, (G.nu - 1) / 2);
        else
            return 0;
    }

    // The primitive integral Q(G_n)
    template <typename T>
    inline T Q(Greens<T>& G){
        if (is_even(G.mu, 2))
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
            T (*setter)(Greens<T>&, int, int);
            Greens<T>& G;

        public:

            // Constructor
            Primitive(Greens<T>& G, T (*setter)(Greens<T>&, int, int)) : setter(setter), G(G) {
                set = Matrix<bool>::Zero(G.N, G.N);
                matrix.resize(G.N, G.N);
            }

            // Getter function. G is a pointer to the current Greens struct,
            // and setter is a pointer to the function that computes the
            // (i, j) element of this primitive matrix
            inline T value(int i, int j) {
                if ((i < 0) || (j < 0) || (i > G.N - 1) || (j > G.N - 1)) {
                    throw errors::BadIndex();
                }
                if (!set(i, j)) {
                    matrix(i, j) = (*setter)(G, i, j);
                    set(i, j) = true;
                }
                return matrix(i, j);
            }

            // Overload () to get the function value without calling value()
            inline T operator() (int i, int j) { return value(i, j); }

            // Resetter
            void reset() {
                set.setZero(G.N, G.N);
            }

    };

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

    // Greens integration housekeeping data
    template <class T>
    class Greens {

        public:

            // Indices
            int lmax;
            int N;
            int l;
            int m;
            int mu;
            int nu;

            // Taylor expand stuff?
            bool taylor;

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

            // Elliptic integrals
            Elliptic<T> ELL;

            // Primitive matrices
            Primitive<T> H;
            Primitive<T> I;
            Primitive<T> J;
            Primitive<T> M;

            // The solution vector
            VectorT<T> sT;

            // The value of pi, computed at
            // the user-requested precision
            T pi;
            T pi_over_2;

            // Constructor
            Greens(int lmax, bool taylor=true) :
                   lmax(lmax),
                   N(max(lmax + 5, 2 * lmax + 1) + 12), // DEBUG: +12 for taylor::computeJ. Find smallest value of N that will work.
                   taylor(taylor),
                   ksq(0),
                   b(0),
                   r(0),
                   b_r(0),
                   cosphi(0),
                   sinphi(0),
                   coslam(0),
                   sinlam(0),
                   ELL(*this),
                   H(*this, computeH),
                   I(*this, computeI),
                   J(*this, computeJ),
                   M(*this, computeM) {

                // Initialize the solution vector
                sT = VectorT<T>::Zero((lmax + 1) * (lmax + 1));

                // Compute pi at the actual precision of the T type
                pi = T(BIGPI);
                pi_over_2 = T(0.5 * pi);

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
    void computesT(Greens<T>& G, const T& b, const T& r, const Vector<T>& y) {

        // Check for likely instability
        if ((G.taylor) && (r >= 1) && (G.lmax > STARRY_LMAX_LARGE_OCC))
            throw errors::LargeOccultorsUnstable();

        // Initialize the basic variables
        int l, m;
        int n = 0;
        T ksq, k;
        G.br = b * r;
        G.br32 = pow(G.br, 1.5);
        G.b.reset(b);
        G.r.reset(r);
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
                if (abs(y(n)) > STARRY_MAP_TOLERANCE) {
                    if ((l == 1) && (m == 0))
                        G.sT(n) = s2(G);
                    // These terms are zero because they are proportional to
                    // odd powers of x, so we don't need to compute them!
                    else if ((G.taylor) && (is_even(G.mu - 1)) && (!is_even((G.mu - 1) / 2)))
                        G.sT(n) = 0;
                    // These terms are also zero for the same reason
                    else if ((G.taylor) && (is_even(G.mu)) && (!is_even(G.mu / 2)))
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
