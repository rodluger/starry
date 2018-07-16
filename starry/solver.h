/**
Spherical harmonic integration utilities.

*/

#ifndef _STARRY_SOLVER_H_
#define _STARRY_SOLVER_H_

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <vector>
#include <boost/math/special_functions/gamma.hpp>
#include "constants.h"
#include "ellip.h"
#include "errors.h"
#include "lld.h"
#include "utils.h"
#include "tables.h"

namespace solver {

    using std::abs;
    using std::max;
    using std::swap;
    using std::vector;
    using std::min;

    // Forward declaration
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
            Power(T val, int reserve=100) {
                if (reserve) vec.reserve(reserve);
                vec.push_back(1.0 + (val * 0));
                vec.push_back(val);
            }

            // Getter function
            inline T value(int n) {
                if (n < 0) {
                    throw errors::BadIndex();
                }
                while (n >= (int)vec.size()) {
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

    // Elliptic integral storage class
    template <class T>
    class Elliptic {

            T vK;
            T vE;
            bool bK;
            bool bE;
            Power<T>& ksq;
            T& invksq;

        public:

            // Constructor
            Elliptic(Power<T>& ksq, T& invksq) : ksq(ksq), invksq(invksq) {
                reset();
            }

            // Resetter
            void reset() {
                bK = false;
                bE = false;
            }

            // Elliptic integral of the first kind
            inline T K() {
                if (!bK) {
                    if ((isinf(get_value(ksq()))) || (ksq() == 1))
                        vK = 0;
                    else if (ksq() < 1)
                        vK = ellip::K(ksq());
                    else
                        vK = ellip::K(invksq);
                    bK = true;
                }
                return vK;
            }

            // Elliptic integral of the second kind
            inline T E() {
                if (!bE) {
                    if (isinf(get_value(ksq())))
                        vE = 0;
                    else if (ksq() == 1)
                        vE = 1;
                    else if (ksq() < 1)
                        vE = ellip::E(ksq());
                    else
                        vE = ellip::E(invksq);
                    bE = true;
                }
                return vE;
            }

    };

    // The factor multiplying `L`
    template <typename T>
    inline T LFac(Greens<T>& G){
        return G.twor(G.l - 1) * G.lfac;
    }

    // Compute the n=2 term of the *s^T* occultation solution vector.
    // This is the Mandel & Agol solution for linear limb darkening,
    // reparametrized for speed
    template <typename T>
    inline T s2(Greens<T>& G) {
        T K = G.ELL.K();
        T E = G.ELL.E();
        T ksq = G.ksq();
        return lld::s2(G.b, G.r, ksq, K, E, G.pi);
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
    inline void QuadLimbDark(Greens<T>& G, const T& b, const T& r, const T& g0, const T& g2, const T& g8, T& s0_val, T& s2_val, T& s8_val) {
        T b2 = b * b;
        T r2 = r * r;
        T pi_over_2 = 0.5 * G.pi;
        if ((abs(1 - r) < b) && (b < 1 + r)) {
            T r3 = r * r2;
            T r4 = r2 * r2;
            T sp = (1 - r2 - b2) / (2 * b * r);
            T cp = sqrt(1 - sp * sp);
            T sl = (1 - r2 + b2) / (2 * b);
            T cl = sqrt(1 - sl * sl);
            T l2 = asin(sl) + pi_over_2;
            T p2  = asin(sp) + pi_over_2;
            T cp2 = cp * cp;
            T sp2 = sp * sp;
            T cpsp = cp * sp;
            T cp3 = cp * cp2;
            T sp3 = sp * sp2;
            T clsl = cl * sl;
            T cl3 = cl * cl * cl;
            T sl3 = sl * sl * sl;
            s0_val = l2 + clsl - r2 * (p2 + cpsp);
            s8_val = 0.5 * l2 + (1. / 3.) * clsl - (1. / 6.) * cl3 * sl + (1. / 6.) * cl * sl3 -
                     (r2 * b2 * (p2 + cpsp) - r3 * b * cp * (1. + (1. / 3.) * cp2 - sp2) +
                      r4 * (0.5 * p2 + (1. / 3.) * cpsp - (1. / 6.) * cp3 * sp + (1. / 6.) * cp * sp3));
        } else {
            s0_val = G.pi * (1 - r2);
            s8_val = pi_over_2 - G.pi * r2 * (0.5 * r2 + b2);
        }
        G.b = b;
        G.r = r;
        if ((b == 0) || (r == 0)) {
            G.ksq.reset(T(INFINITY));
            G.invksq = 0;
        } else {
            G.ksq.reset((1 - (b - r)) * (1 + (b - r)) / (4 * b * r));
            G.invksq = (4 * b * r) / ((1 - (b - r)) * (1 + (b - r)));
        }
        G.ELL.reset();
        s2_val = s2(G);
    }

    // Vieta's theorem coefficient A_{i,u,v}
    template <class T>
    class A {

            Vector<bool>** set;
            Vector<T>** vec;
            int umax;
            int vmax;
            int j, j1, j2, c;
            T res;
            Power<T>& delta;

        public:

            // Constructor
            A(int lmax, Power<T>& delta) :
                    umax(is_even(lmax) ? (lmax + 2) / 2 : (lmax + 3) / 2),
                    vmax(lmax > 0 ? lmax : 1), delta(delta) {
                vec = new Vector<T>*[umax + 1];
                set = new Vector<bool>*[umax + 1];
                for (int u = 0; u < umax + 1; u++) {
                    vec[u] = new Vector<T>[vmax + 1];
                    set[u] = new Vector<bool>[vmax + 1];
                    for (int v = 0; v < vmax + 1; v++) {
                        vec[u][v].resize(u + v + 1);
                        set[u][v].setZero(u + v + 1);
                    }
                }

            }

            // Destructor
            ~A() {
                for (int u = 0; u < umax + 1; u++) {
                    delete [] vec[u];
                    delete [] set[u];
                }
                delete [] vec;
                delete [] set;
            }

            // Compute the double-binomial coefficient A_{i,u,v}
            inline T compute(int i, int u, int v) {
                res = 0;
                j1 = u - i;
                if (j1 < 0) j1 = 0;
                j2 = u + v - i;
                if (j2 > u) j2 = u;
                for (j = j1; j <= j2; j++) {
                    c = u + v - i - j;
                    if (c < 0)
                        break;
                    if (is_even(u + j))
                        res += tables::choose<T>(u, j) * tables::choose<T>(v, c) * delta(c);
                    else
                        res -= tables::choose<T>(u, j) * tables::choose<T>(v, c) * delta(c);
                }
                return res;
            }

            // Getter function
            inline T get_value(int i, int u, int v) {
                if ((i < 0) || (u < 0) || (v < 0) || (u > umax) || (v > vmax) || (i > u + v)) {
                    throw errors::BadIndex();
                }
                if (!set[u][v](i)) {
                    vec[u][v](i) = compute(i, u, v);
                    set[u][v](i) = true;
                }
                return vec[u][v](i);
            }

            // Overload () to get the function value without calling value()
            inline T operator() (int i, int u, int v) { return get_value(i, u, v); }

            // Resetter
            void reset() {
                for (int u = 0; u < umax + 1; u++) {
                    for (int v = 0; v < vmax + 1; v++) {
                        set[u][v].setZero(u + v + 1);
                    }
                }
            }

    };

    // The helper primitive integral H_{u,v}
    template <class T>
    class H {

            Matrix<bool> set;
            Matrix<T> value;
            int umax;
            int vmax;
            T pi;
            Power<T>& sinlam;
            Power<T>& coslam;

        public:

            // Constructor
            H(int lmax, Power<T>& sinlam, Power<T>& coslam) : umax(lmax + 2), vmax(max(1, lmax)),
                    sinlam(sinlam), coslam(coslam) {
                set = Matrix<bool>::Zero(umax + 1, vmax + 1);
                value.resize(umax + 1, vmax + 1);
                pi = T(BIGPI);
            }

            // Reset flags and compute H_00 and H_01
            inline void reset(int downward=false) {
                if (downward)
                    throw errors::NotImplemented();
                set.setZero(umax + 1, vmax + 1);
                if (coslam() == 0) {
                    // When sinlam = 1, asin(sinlam) = pi
                    // but the derivative is undefined, so
                    // we sidestep the computation here to
                    // prevent NaNs in the autodiff calculation.
                    value(0, 0) = 2 * pi;
                    value(0, 1) = 0;
                } else {
                    if (sinlam() < 0.5)
                        value(0, 0) = 2 * asin(sinlam()) + pi;
                    else
                        value(0, 0) = 2 * acos(coslam()) + pi;
                    value(0, 1) = -2 * coslam(1);
                }
                set(0, 0) = true;
                set(0, 1) = true;
            }

            // Getter function
            inline T get_value(int u, int v) {
                if ((u < 0) || (v < 0) || (u > umax) || (v > vmax)) {
                    throw errors::BadIndex();
                } else if (!is_even(u) || ((coslam() == 0) && !is_even(v))) {
                    return 0;
                } else if (!set(u, v)) {
                    if (u >= 2)
                        value(u, v) = (2 * coslam(u - 1) * sinlam(v + 1) + (u - 1) * get_value(u - 2, v)) / (u + v);
                    else
                        value(u, v) = (-2 * coslam(u + 1) * sinlam(v - 1) + (v - 1) * get_value(u, v - 2)) / (u + v);
                    set(u, v) = true;
                }
                return value(u, v);
            }

            // Overload () to get the function value without calling value()
            inline T operator() (int u, int v) { return get_value(u, v); }

    };

    // The helper primitive integral I_v
    template <class T>
    class I {

            Vector<bool> set;
            Vector<T> value;
            int vmax;
            Power<T>& ksq;
            T& k;
            T& kc;
            T& kkc;
            T& kap0;
            T sqrtpi;
            Vector<T> ivgamma;

        public:

            I(int lmax, Power<T>& ksq, T& k, T& kc, T& kkc, T& kap0) : vmax(2 * lmax + 2), ksq(ksq), k(k), kc(kc), kkc(kkc), kap0(kap0) {
                set = Vector<bool>::Zero(vmax + 1);
                value.resize(vmax + 1);

                // Pre-tabulate I_v for ksq >= 1
                sqrtpi = sqrt(T(BIGPI));
                ivgamma.resize(vmax + 1);
                for (int v = 0; v <= vmax; v++)
                    ivgamma(v) = sqrtpi * T(boost::math::tgamma_delta_ratio(Multi(v + 0.5), Multi(0.5)));

            }

            // Reset flags and compute either I_0 or I_vmax
            inline void reset(int downward=true) {

                // No need to reset anything, as in the ksq >= 1
                // case we can just use the pre-tabulated values of I_v!
                if (ksq() >= 1) return;

                // Reset flags
                set.setZero(vmax + 1);

                if (downward) {

                    // Downward recursion: compute I_vmax
                    T tol = mach_eps<T>() * ksq();
                    T error = T(INFINITY);

                    // Computing leading coefficient (n=0):
                    T coeff = 2.0 / (2 * vmax + 1);

                    // Add leading term to I_vmax:
                    T res = coeff;

                    // Now, compute higher order terms until desired precision is reached
                    int n = 1;
                    while ((n < STARRY_IJ_MAX_ITER) && (abs(error) > tol)) {
                        coeff *= (2.0 * n - 1.0) * 0.5 * (2 * n + 2 * vmax - 1) / (n * (2 * n + 2 * vmax + 1)) * ksq();
                        error = coeff;
                        res += coeff;
                        n++;
                    }

                    // If we didn't converge, fall back to upward recursion
                    if (n == STARRY_IJ_MAX_ITER) throw errors::Primitive("I");

                    value(vmax) = ksq(vmax) * k * res;
                    set(vmax) = true;

                } else {

                    // Upward recursion: compute I_0
                    value(0) = kap0;
                    set(0) = true;

                }

            }

            // Getter function
            inline T get_value(int v) {
                if ((v < 0) || (v > vmax)) throw errors::BadIndex();
                if (ksq() >= 1) {
                    // Easy: these are constant & tabulated!
                    return ivgamma(v);
                } else if (!set(v)) {
                    if (set(vmax))
                        // Downward recursion (preferred)
                        value(v) = 2.0 / (2 * v + 1) * ((v + 1) * get_value(v + 1) + ksq(v) * kkc);
                    else if (set(0))
                        // Upward recursion
                        value(v) = ((2 * v - 1) / 2.0 * get_value(v - 1) - ksq(v - 1) * kkc) / v;
                    else
                        throw errors::Recursion();
                    set(v) = true;
                }
                return value(v);
            }

            // Overload () to get the function value without calling get_value()
            inline T operator() (int v) { return get_value(v); }

    };

    // The helper primitive integral J_v
    template <class T>
    class J {

            vector<int> vvec;
            Vector<bool> set;
            Vector<T> value;
            int vmax;
            Elliptic<T>& ELL;
            Power<T>& ksq;
            Power<T>& two;
            T& k;
            T& kc;
            T& invksq;
            T pi;

        public:

            J(int lmax, Elliptic<T>& ELL, Power<T>& ksq, Power<T>& two, T& k, T& kc, T& invksq) : vmax(max(1, 2 * lmax - 1)), ELL(ELL), ksq(ksq), two(two), k(k), kc(kc), invksq(invksq) {
                set = Vector<bool>::Zero(vmax + 1);
                value.resize(vmax + 1);
                pi = T(BIGPI);
                // These are the values of v we pre-compute on downward recursion
                vvec.push_back(vmax);
                vvec.push_back(vmax - 1);
                // If lmax is large (>~ 15), we can lose significant precision
                // by the time we get down to l = 0. So let's pre-compute
                // the two J_v's at the midpoint for higher accuracy at low l
                if (vmax >= 30) {
                    vvec.push_back(vmax / 2);
                    vvec.push_back(vmax / 2 - 1);
                }
            }

            // Reset flags and compute J_vmax and J_{vmax - 1} with a series expansion
            inline void reset(bool downward=true) {

                // Reset flags
                set.setZero(vmax + 1);

                if (downward) {

                    // Downward recursion: compute J_vmax and J_{vmax - 1}
                    T tol;
                    if (ksq() >= 1)
                        tol = mach_eps<T>() * invksq;
                    else
                        tol = mach_eps<T>() * ksq();
                    for (int v : vvec) {

                        // Computing leading coefficient (n=0):
                        T coeff;
                        if (ksq() >= 1) {
                            coeff = pi;
                            for (int i = 1; i <= v; i++) coeff *= (1 - 0.5 / i);
                        } else {
                            coeff = 3 * pi / (two(2 + v) * tables::factorial<T>(v + 2));
                            for (int i = 1; i <= v; i++) coeff *= (2.0 * i - 1);
                        }

                        // Add leading term to J_vmax:
                        T res = coeff;

                        // Now, compute higher order terms until desired precision is reached
                        int n = 1;
                        T error = T(INFINITY);
                        while ((n < STARRY_IJ_MAX_ITER) && (abs(error) > tol)) {
                            if (ksq() >= 1)
                                coeff *= (1.0 - 2.5 / n) * (1.0 - 0.5 / (n + v)) * invksq;
                            else
                                coeff *= (2.0 * n - 1.0) * (2.0 * (n + v) - 1.0) * 0.25 / (n * (n + v + 2)) * ksq();
                            error = coeff;
                            res += coeff;
                            n++;
                        }
                        if (n == STARRY_IJ_MAX_ITER) throw errors::Primitive("J");

                        // Store the result
                        if (ksq() >= 1)
                            value(v) = res;
                        else
                            value(v) = ksq(v) * k * res;
                        set(v) = true;
                    }

                } else {

                    if (ksq() >= 1) {

                        // Upward recursion: compute J_0 and J_1
                        value(0) = (2.0 / 3.0) * (2 * (2 - invksq) * ELL.E() - (1 - invksq) * ELL.K());
                        value(1) = (2.0 / 15.0) * ((-3 * ksq() + 13 - 8 * invksq) * ELL.E() + (1 - invksq) * (3 * ksq() - 4) * ELL.K());

                        // NOTE: These expressions are in principle more stable, but I have not seen
                        // this in practice. The advantage of the expressions above is that the elliptic
                        // integrals are often used in the computation of `s2`, so they are available for free!
                        /*
                        T k2inv = 1 / ksq();
                        T fe = 2 * (2 - k2inv);
                        T fk = -1 + k2inv;
                        value(0) = (2.0 / 3.0) * ellip::CEL(k2inv, kc, T(1), T(fk + fe), T(fk + fe * (1 - k2inv)), pi);
                        fe = -6 * ksq() + 26 - 16 * k2inv;
                        fk = 2 * (1 - k2inv) * (3 * ksq() - 4);
                        value(1) = ellip::CEL(k2inv, kc, T(1), T(fk + fe), T(fk + fe * (1 - k2inv)), pi) / 15.;
                        */

                    } else {

                        // Upward recursion: compute J_0 and J_1
                        value(0) = 2.0 / (3.0 * ksq() * k) * (2 * (2 * ksq() - 1) * ELL.E() + (1 - ksq()) * (2 - 3 * ksq()) * ELL.K());
                        value(1) = 2.0 / (15.0 * ksq() * k) * ((-3 * ksq(2) + 13 * ksq() - 8) * ELL.E() + (1 - ksq()) * (8 - 9 * ksq()) * ELL.K());

                        // See NOTE above regarding these expressions
                        /*
                        T fe = 2 * (2 * ksq() - 1);
                        T fk = (1 - ksq()) * (2 - 3 * ksq());
                        value(0) = 2 / (3 * ksq() * k) * ellip::CEL(ksq(), kc, T(1), T(fk + fe), T(fk + fe * (1 - ksq())), pi);
                        fe = -3 * ksq(2) + 13 * ksq() - 8;
                        fk = (1 - ksq()) * (8 - 9 * ksq());
                        value(1) = 2 / (15 * ksq() * k) * ellip::CEL(ksq(), kc, T(1), T(fk + fe), T(fk + fe * (1 - ksq())), pi);
                        */

                    }

                    set(0) = true;
                    set(1) = true;

                }

            }

            // Getter function
            inline T get_value(int v) {
                if ((v < 0) || (v > vmax)) throw errors::BadIndex();
                if (!set(v)) {
                    if (set(vmax)) {
                        // Downward recursion (preferred)
                        T f1, f2, f3;
                        if (ksq() < 1) {
                            f2 = ksq() * (2 * v + 1);
                            f1 = 2 * (3 + v + ksq() * (1 + v)) / f2;
                            f3 = (2 * v + 7) / f2;
                        } else {
                            f3 = (2. * v + 7) / (2. * v + 1) * invksq;
                            f1 = (2. / (2. * v + 1)) * ((3 + v) * invksq + 1 + v);
                        }
                        value(v) = f1 * get_value(v + 1) - f3 * get_value(v + 2);
                    } else if (set(0)) {
                        // Upward recursion
                        T f1 = 2 * (v + (v - 1) * ksq() + 1);
                        T f2 = ksq() * (2 * v - 3);
                        value(v) = (f1 * get_value(v - 1) - f2 * get_value(v - 2)) / (2 * v + 3);
                    } else {
                        throw errors::Recursion();
                    }
                    set(v) = true;
                }
                return value(v);
            }

            // Overload () to get the function value without calling get_value()
            inline T operator() (int v) { return get_value(v); }

    };

    // The helper primitive integral K_{u,v}
    template <typename T>
    inline T K(Greens<T>& G, int u, int v) {
        T res = 0;
        for (int i = 0; i < u + v + 1; i++)
            res += G.A_P(i, u, v) * G.I_P(i + u);
        return res;
    }

    // The helper primitive integral L_{u,v}^(t)
    template <typename T>
    inline T L(Greens<T>& G, int u, int v, int t) {
        T res = 0;
        for (int i = 0; i < u + v + 1; i++) {
            if (G.b == 0) {
                // Special case, J = I (evident from integral definition)
                res += G.A_P(i, u, v) * G.I_P(i + u + t);
            } else {
                res += G.A_P(i, u, v) * G.J_P(i + u + t);
            }
        }
        return res;
    }

    // The primitive integral P(G_n)
    template <typename T>
    inline T P(Greens<T>& G){
        if (is_even(G.mu, 2)) {
            return 2 * G.twor(G.l + 2) * K(G, (G.mu + 4) / 4, G.nu / 2);
        } else if ((G.mu == 1) && is_even(G.l)) {
            return LFac(G) * (L(G, (G.l - 2) / 2, 0, 0) - 2 * L(G, (G.l - 2) / 2, 0, 1));
        } else if ((G.mu == 1) && !is_even(G.l)) {
            return LFac(G) * (L(G, (G.l - 3) / 2, 1, 0) - 2 * L(G, (G.l - 3) / 2, 1, 1));
        } else if (is_even(G.mu - 1, 2)) {
            return 2 * LFac(G) * L(G, (G.mu - 1) / 4, (G.nu - 1) / 2, 0);
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
        if ((G.coslam() == 0) && (!is_even(G.mu, 2) || !is_even(G.nu, 2)))
            return 0;
        else if (!is_even(G.mu, 2))
            return 0;
        else {
            return G.H_Q((G.mu + 4) / 2, G.nu / 2);
        }
    }

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

            // Basic variables
            T b;
            T r;
            T k;
            T kc;
            T kkc;
            T kap0;
            T fourbr32;
            T lfac;
            T invksq;

            // Powers of basic variables
            Power<T> ksq;
            Power<T> twor;
            Power<T> delta;
            Power<T> sinlam;
            Power<T> coslam;

            // Static stuff
            Power<T> two;
            T miny;

            // Primitive matrices/vectors
            Elliptic<T> ELL;
            H<T> H_Q;
            I<T> I_P;
            J<T> J_P;
            A<T> A_P;

            // The solution vector
            VectorT<T> sT;

            // The value of pi, computed at
            // the user-requested precision
            T pi;

            // Constructor
            Greens(int lmax) :
                   lmax(lmax),
                   ksq(0),
                   twor(0),
                   delta(0),
                   sinlam(0),
                   coslam(0),
                   two(0),
                   ELL((*this).ksq, (*this).invksq),
                   H_Q(lmax, (*this).sinlam, (*this).coslam),
                   I_P(lmax, (*this).ksq, (*this).k, (*this).kc, (*this).kkc, (*this).kap0),
                   J_P(lmax, (*this).ELL, (*this).ksq, (*this).two, (*this).k, (*this).kc, (*this).invksq),
                   A_P(lmax, (*this).delta) {

                // Initialize the solution vector
                sT = VectorT<T>::Zero((lmax + 1) * (lmax + 1));

                // Compute pi at the actual precision of the T type
                pi = T(BIGPI);

                // Initialize static stuff
                two.reset(2);

                // Smallest coefficient for which we'll actually
                // bother to compute the integrals
                // NOTE: When doing autodiff, we always want to
                // compute the solution vector for all indices!!!
                if (is_Grad(pi))
                    miny = 0;
                else
                    miny = 10 * mach_eps<T>();

            }

    };

    // Compute the *s^T* occultation solution vector
    template <typename T>
    void computesT(Greens<T>& G, const T& b, const T& r, const Vector<T>& y) {

        // Initialize the basic variables
        int l, m;
        int n = 0;
        G.b = b;
        G.r = r;
        T ksq;
        if ((b == 0) || (r == 0)) {
            ksq = T(INFINITY);
            G.k = T(INFINITY);
            G.kc = 1;
            G.kkc = T(INFINITY);
            G.invksq = 0;
            G.kap0 = 0;
        } else {
            ksq = (1 - (b - r)) * (1 + (b - r)) / (4 * b * r);
            G.invksq = (4 * b * r) / ((1 - (b - r)) * (1 + (b - r)));
            G.k = sqrt(ksq);
            if (ksq > 1) {
                G.kc = sqrt(abs(((b + r) * (b + r) - 1) / ((b - r) * (b - r) - 1)));
                G.kkc = G.k * G.kc;
                G.kap0 = 0; // Not used!
            } else {
                G.kc = sqrt(abs(((b + r) * (b + r) - 1) / (4 * b * r)));
                // Eric Agol's "kite" method to compute a stable
                // version of k * kc and I_0 = kap0
                // Used to be
                //   G.kkc = G.k * G.kc;
                //   G.kap0 = 2 * acos(G.kc);
                T p0 = 1, p1 = b, p2 = r;
                if (p0 < p1) swap(p0, p1);
                if (p1 < p2) swap(p1, p2);
                if (p0 < p1) swap(p0, p1);
                T kite_area2 = sqrt((p0 + (p1 + p2)) * (p2 - (p0 - p1)) * (p2 + (p0 - p1)) * (p0 + (p1 - p2)));
                G.kkc = kite_area2 / (4 * b * r);
                G.kap0 = atan2(kite_area2, (r - 1) * (r + 1) + b * b);
            }
        }
        G.ksq.reset(ksq);
        G.fourbr32 = pow(4 * b * r, 1.5);
        G.lfac = pow(1 - (b - r) * (b - r), 1.5);

        // Powers of basic variables
        G.twor.reset(2 * r);
        G.delta.reset((b - r) / (2 * r));
        if ((abs(1 - r) < b) && (b < 1 + r)) {
            G.sinlam.reset(0.5 * ((1. / b) + (b - r) * (1. + r / b)));
            G.coslam.reset(sqrt(1 - G.sinlam() * G.sinlam()));
            // Stability override
            if (G.sinlam() > 0.5) {
                T delta = 1 - (b + r);
                T eps = ((r / b) * delta + (delta * delta) / (2 * b));
                G.sinlam.reset(1 + eps);
                G.coslam.reset(sqrt(-eps * (2 + eps)));
            }
        } else {
            G.sinlam.reset(1);
            G.coslam.reset(0);
        }

        // Initialize our storage classes
        // For 0.5 < ksq < 2 we do upward recursion,
        // otherwise we do downward recursion.
        G.ELL.reset();
        G.H_Q.reset();
        G.I_P.reset(ksq < 0.5);
        G.J_P.reset((ksq < 0.5) || (ksq > 2));
        G.A_P.reset();

        // Populate the solution vector
        for (l = 0; l < G.lmax + 1; l++) {
            G.l = l;
            for (m = -l; m < l + 1; m++) {
                G.m = m;
                G.mu = l - m;
                G.nu = l + m;
                if (abs(y(n)) >= G.miny) {
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
