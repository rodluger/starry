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
                    if ((G.b == 0) || (G.ksq == 1))
                        vK = 0;
                    else if (G.ksq < 1)
                        vK = ellip::K(G.ksq);
                    else
                        vK = ellip::K(T(1. / G.ksq));
                    bK = true;
                }
                return vK;
            }

            // Elliptic integral of the second kind
            inline T E() {
                if (!bE) {
                    if (G.b == 0)
                        vE = 0;
                    else if (G.ksq == 1)
                        vE = 1;
                    else if (G.ksq < 1)
                        vE = ellip::E(G.ksq);
                    else
                        vE = ellip::E(T(1. / G.ksq));
                    bE = true;
                }
                return vE;
            }

            // First elliptic function
            inline T E1() {
                if (!bE1) {
                    if ((G.b == 0) || (G.ksq == 1))
                        vE1 = 0;
                    else if (G.ksq < 1)
                        vE1 = (1 - G.ksq) * K();
                    else
                        vE1 = (1 - G.ksq) / G.k() * K();
                    bE1 = true;
                }
                return vE1;
            }

            // Second elliptic function
            inline T E2() {
                if (!bE2) {
                    if (G.b == 0)
                        vE2 = 0;
                    else if (G.ksq == 1)
                        vE2 = 1;
                    else if (G.ksq < 1)
                        vE2 = E();
                    else
                        vE2 = G.k() * E() + (1 - G.ksq) / G.k() * K();
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

    // The constant factor F_l
    template <typename T>
    inline T F(Greens<T>& G){
        return G.twor(G.l - 1) * G.fourbr32;
    }

    // Compute the n=2 term of the *s^T* occultation solution vector.
    // This is the Mandel & Agol solution for linear limb darkening,
    // reparametrized for speed
    template <typename T>
    inline T s2(Greens<T>& G) {
        T K = G.ELL.K();
        T E = G.ELL.E();
        return lld::s2(G.b, G.r, G.ksq, K, E, G.pi);
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
            Greens<T>& G;

        public:

            // Constructor
            A(Greens<T>& G) : G(G) {
                umax = is_even(G.lmax) ? (G.lmax + 2) / 2 : (G.lmax + 3) / 2;
                vmax = G.lmax > 0 ? G.lmax : 1;
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
                        res += tables::choose<T>(u, j) * tables::choose<T>(v, c) * G.delta(c);
                    else
                        res -= tables::choose<T>(u, j) * tables::choose<T>(v, c) * G.delta(c);
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
            Greens<T>& G;

        public:

            // Constructor
            H(Greens<T>& G) : G(G) {
                umax = G.lmax + 2;
                vmax = G.lmax;
                set = Matrix<bool>::Zero(umax + 1, vmax + 1);
                value.resize(umax + 1, vmax + 1);
            }

            // Compute H_{u,v}
            inline T compute(int u, int v) {
                if (!is_even(u)) {
                    return 0;
                } else if (G.off_limb && !is_even(v)) {
                    return 0;
                } else if ((u == 0) && (v == 0)) {
                    return 2 * asin(G.sinlam()) + G.pi;
                } else if ((u == 0) && (v == 1)) {
                    return -2 * G.coslam(1);
                } else if (u >= 2) {
                    return (2 * G.coslam(u - 1) * G.sinlam(v + 1) + (u - 1) * G.H(u - 2, v)) / (u + v);
                } else {
                    return (-2 * G.coslam(u + 1) * G.sinlam(v - 1) + (v - 1) * G.H(u, v - 2)) / (u + v);
                }
            }

            // Getter function
            inline T get_value(int u, int v) {
                if ((u < 0) || (v < 0) || (u > umax) || (v > vmax)) {
                    throw errors::BadIndex();
                }
                if (!set(u, v)) {
                    value(u, v) = compute(u, v);
                    set(u, v) = true;
                }
                return value(u, v);
            }

            // Overload () to get the function value without calling value()
            inline T operator() (int u, int v) { return get_value(u, v); }

            // Resetter
            void reset() {
                set.setZero(umax + 1, vmax + 1);
            }

    };

    // The helper primitive integral I_v
    template <class T>
    class I {

            Vector<bool> set;
            Vector<T> value;
            int vmax;
            Greens<T>& G;

        public:

            I(Greens<T>& G) : G(G) {
                vmax = 2 * G.lmax + 2;
                set = Vector<bool>::Zero(vmax + 1);
                value.resize(vmax + 1);
            }

            // Reset flags and compute I_vmax with a series expansion
            inline void reset() {

                // Reset flags
                set.setZero(vmax + 1);

                // Convergence
                T tol = mach_eps<T>() * G.ksq;
                T error = T(INFINITY);

                // Computing leading coefficient (n=0):
                T coeff = 2.0 / (2 * vmax + 1);

                // Add leading term to I_vmax:
                T res = coeff;

                // Now, compute higher order terms until desired precision is reached
                int n = 1;
                while ((n < STARRY_IJ_MAX_ITER) && (error > tol)) {
                    coeff *= (2.0 * n - 1.0) * 0.5 * (2 * n + 2 * vmax - 1) / (n * (2 * n + 2 * vmax + 1)) * G.ksq;
                    error = coeff;
                    res += coeff;
                    n++;
                }

                // Store the result
                value(vmax) = G.k(1 + 2 * vmax) * res;
                set(vmax) = true;
            }

            // Getter function
            inline T get_value(int v) {
                if ((v < 0) || (v > vmax)) throw errors::BadIndex();
                if (!set(v)) {
                    value(v) = 2.0 / (2 * v + 1) * ((v + 1) * get_value(v + 1) + G.k(2 * v + 1) * G.kc);
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

            Vector<bool> set;
            Vector<T> value;
            int vmax;
            Greens<T>& G;

        public:

            J(Greens<T>& G) : G(G) {
                vmax = 2 * G.lmax - 1;
                set = Vector<bool>::Zero(vmax + 1);
                value.resize(vmax + 1);
            }

            // Reset flags and compute J_vmax and J_{vmax - 1} with a series expansion
            inline void reset() {

                // Reset flags
                set.setZero(vmax + 1);

                // Convergence
                T tol = mach_eps<T>() * G.ksq;

                // Compute the highest two terms in J
                for (int v = vmax; v >= vmax - 1; v--) {

                    // Computing leading coefficient (n=0):
                    T coeff = 3 * G.pi / (pow(2, 2 + v) * tables::factorial<T>(v + 2));
                    for (int i = 1; i <= v; i++) coeff *= (2.0 * i - 1);

                    // Add leading term to J_vmax:
                    T res = coeff;

                    // Now, compute higher order terms until desired precision is reached
                    int n = 1;
                    T error = T(INFINITY);
                    while ((n < STARRY_IJ_MAX_ITER) && (error > tol)) {
                        coeff *= (2.0 * n - 1.0) * (2.0 * (n + v) - 1.0) * 0.25 / (n * (n + v + 2)) * G.ksq;
                        error = coeff;
                        res += coeff;
                        n++;
                    }

                    // Store the result
                    value(v) = G.k(1 + 2 * v) * res;
                    set(v) = true;
                }

            }

            // Getter function
            inline T get_value(int v) {
                if ((v < 0) || (v > vmax)) throw errors::BadIndex();
                if (!set(v)) {
                    T f2 = G.ksq * (2 * v + 1);
                    T f1 = 2 * (3 + v + G.ksq * (1 + v)) / f2;
                    T f3 = (2 * v + 7) / f2;
                    value(v) = f1 * get_value(v + 1) - f3 * get_value(v + 2);
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
            res += G.A(i, u, v) * G.I(i + u);
        return res;
    }

    // The helper primitive integral L_{u,v}^(t)
    template <typename T>
    inline T L(Greens<T>& G, int u, int v, int t) {
        T res = 0;
        for (int i = 0; i < u + v + 1; i++)
            res += G.A(i, u, v) * G.J(i + u + t);
        return G.k(3) * res;
    }

    // The primitive integral P(G_n)
    template <typename T>
    inline T P(Greens<T>& G){
        if (is_even(G.mu, 2)) {
            return 2 * G.twor(G.l + 2) * K(G, (G.mu + 4) / 4, G.nu / 2);
        } else if ((G.mu == 1) && is_even(G.l)) {
            return F(G) * (L(G, (G.l - 2) / 2, 0, 0) - 2 * L(G, (G.l - 2) / 2, 0, 1));
        } else if ((G.mu == 1) && !is_even(G.l)) {
            return F(G) * (L(G, (G.l - 3) / 2, 1, 0) - 2 * L(G, (G.l - 3) / 2, 1, 1));
        } else if (is_even(G.mu - 1, 2)) {
            return 2 * F(G) * L(G, (G.mu - 1) / 4, (G.nu - 1) / 2, 0);
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

            // Occultor off the limb of the body?
            bool off_limb;

            // Basic variables
            T b;
            T r;
            T ksq;
            T kc;
            T fourbr32;

            // Powers of basic variables
            Power<T> k;
            Power<T> twor;
            Power<T> delta;
            Power<T> coslam;
            Power<T> sinlam;

            // Elliptic integrals
            Elliptic<T> ELL;

            // Primitive matrices/vectors
            H<T> H;
            I<T> I;
            J<T> J;
            A<T> A;

            // The solution vector
            VectorT<T> sT;

            // The value of pi, computed at
            // the user-requested precision
            T pi;

            // Constructor
            Greens(int lmax) :
                   lmax(lmax),
                   off_limb(false),
                   k(0),
                   twor(0),
                   delta(0),
                   coslam(0),
                   sinlam(0),
                   ELL(*this),
                   H(*this),
                   I(*this),
                   J(*this),
                   A(*this) {

                // Initialize the solution vector
                sT = VectorT<T>::Zero((lmax + 1) * (lmax + 1));

                // Compute pi at the actual precision of the T type
                pi = T(BIGPI);

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
        G.ksq = (1 - (b - r)) * (1 + (b - r)) / (4 * b * r);
        G.kc = sqrt(((b + r) * (b + r) - 1) / (4 * b * r));
        G.fourbr32 = pow(4 * b * r, 1.5);

        // Powers of basic variables
        G.twor.reset(2 * r);
        G.delta.reset((b - r) / (2 * r));
        if ((abs(1 - r) < b) && (b < 1 + r)) {
            G.off_limb = false;
            G.sinlam.reset(0.5 * ((1. / b) + (b - r) * (1. + r / b)));
            G.coslam.reset(sqrt(1 - G.sinlam() * G.sinlam()));
        } else {
            G.off_limb = true;
        }

        // Override the NaNs in the derivs when b = 0 or r = 0
        T k = sqrt(G.ksq);
        if ((b == 0) || (r == 0)) {
            set_derivs_to_zero(G.ksq);
            set_derivs_to_zero(G.kc);
            set_derivs_to_zero(k);
        }
        G.k.reset(k);

        // Initialize our storage classes
        G.ELL.reset();
        G.H.reset();
        G.I.reset();
        G.J.reset();
        G.A.reset();

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

/*
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
                    / G.k(2 * n);
        sign *= -1;
        n++;
    }
    // Note that sqrt(pi) * (3/2)! = 3 * pi / 4
    res *= 0.75 * G.pi;
    return res;
}
*/

#endif
