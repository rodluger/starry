/**
Spherical harmonic integration utilities.

*/

#ifndef _STARRY_SOLVER_H_
#define _STARRY_SOLVER_H_

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>
#include <vector>
#include <boost/math/special_functions/gamma.hpp>
#include "ellip.h"
#include "errors.h"
#include "utils.h"
#include "tables.h"
#include "lld.h"

namespace solver {

    using std::abs;
    using std::max;
    using std::swap;
    using std::vector;
    using std::min;
    using namespace utils;

    // Forward declaration
    template <class T> class Greens;

    /**
    Fast powers of a variable

    */
    template <class T>
    class Power {

            vector<T> vec;

        public:

            //! Default constructor
            Power() {
                vec.push_back(T(1.0));
                vec.push_back(T(0.0));
            }

            //! Constructor
            explicit Power(T val, int reserve=100) {
                if (reserve) vec.reserve(reserve);
                vec.push_back(1.0 + (val * 0));
                vec.push_back(val);
            }

            //! Getter function
            inline T value(int n) {
                if (n < 0) {
                    throw errors::IndexError("Invalid index in the "
                                             "`Power` class.");
                }
                while (n >= (int)vec.size()) {
                    vec.push_back(vec[1] * vec[vec.size() - 1]);
                }
                return vec[n];
            }

            //! Overload () to get the function value without calling value()
            inline T operator() () { return vec[1]; }
            inline T operator() (int n) { return value(n); }

            //! Resetter
            void reset(T val) {
                vec.clear();
                vec.push_back(1.0 + (val * 0));
                vec.push_back(val);
            }

    };

    /**
    Elliptic integral storage class

    */
    template <class T>
    class Elliptic {

            T vK;
            T vE;
            bool bK;
            bool bE;
            Power<T>& ksq;
            T& invksq;

        public:

            //! Constructor
            Elliptic(Power<T>& ksq, T& invksq) : ksq(ksq), invksq(invksq) {
                reset();
            }

            //! Resetter
            void reset() {
                bK = false;
                bE = false;
            }

            //! Elliptic integral of the first kind
            inline T K() {
                if (!bK) {
                    if ((invksq == 0) || (ksq() == 1))
                        vK = 0;
                    else if (ksq() < 1)
                        vK = ellip::K(ksq());
                    else
                        vK = ellip::K(invksq);
                    bK = true;
                }
                return vK;
            }

            //! Elliptic integral of the second kind
            inline T E() {
                if (!bE) {
                    if (invksq == 0)
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

    /**
    The factor multiplying `L`

    */
    template <typename T>
    inline T LFac(Greens<T>& G){
        return G.twor(G.l - 1) * G.lfac;
    }

    /**
    Compute the n=2 term of the *s^T* occultation solution vector.
    This is the Mandel & Agol solution for linear limb darkening,
    reparametrized for speed

    */
    template <typename T>
    inline T s2(Greens<T>& G) {
        T K = G.ELL.K();
        T E = G.ELL.E();
        T ksq = G.ksq();
        return lld::s2(G.b, G.r, ksq, K, E);
    }

    /**
    Vieta's theorem coefficient A_{i,u,v}

    */
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

            //! Constructor
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

            //! Destructor
            ~A() {
                for (int u = 0; u < umax + 1; u++) {
                    delete [] vec[u];
                    delete [] set[u];
                }
                delete [] vec;
                delete [] set;
            }

            //! Compute the double-binomial coefficient A_{i,u,v}
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
                        res += tables::choose<T>(u, j) *
                               tables::choose<T>(v, c) * delta(c);
                    else
                        res -= tables::choose<T>(u, j) *
                               tables::choose<T>(v, c) * delta(c);
                }
                return res;
            }

            //! Getter function
            inline T get_value(int i, int u, int v) {
                if ((i < 0) || (u < 0) || (v < 0) ||
                    (u > umax) || (v > vmax) || (i > u + v)) {
                    throw errors::IndexError("Invalid index in Vieta's "
                                             "theorem evaluation.");
                }
                if (!set[u][v](i)) {
                    vec[u][v](i) = compute(i, u, v);
                    set[u][v](i) = true;
                }
                return vec[u][v](i);
            }

            //! Overload () to get the function value without calling value()
            inline T operator() (int i, int u, int v) {
                return get_value(i, u, v);
            }

            //! Resetter
            void reset() {
                for (int u = 0; u < umax + 1; u++) {
                    for (int v = 0; v < vmax + 1; v++) {
                        set[u][v].setZero(u + v + 1);
                    }
                }
            }

    };

    /**
    The helper primitive integral H_{u,v}

    */
    template <class T>
    class H {

            Matrix<bool> set;
            Matrix<T> value;
            int umax;
            int vmax;
            Power<T>& sinlam;
            Power<T>& coslam;

        public:

            //! Constructor
            H(int lmax, Power<T>& sinlam, Power<T>& coslam) :
                    umax(lmax + 2), vmax(max(1, lmax)),
                    sinlam(sinlam), coslam(coslam) {
                set = Matrix<bool>::Zero(umax + 1, vmax + 1);
                value.resize(umax + 1, vmax + 1);
            }

            //! Reset flags and compute `H_00` and `H_01`
            inline void reset(int downward=false) {
                if (downward)
                    throw errors::NotImplementedError("Downward recursion is "
                                                      "not implemented for the "
                                                      "`H` integral.");
                set.setZero(umax + 1, vmax + 1);
                if (coslam() == 0) {
                    // When sinlam = 1, asin(sinlam) = pi
                    // but the derivative is undefined, so
                    // we sidestep the computation here to
                    // prevent NaNs in the autodiff calculation.
                    value(0, 0) = 2 * pi<T>();
                    value(0, 1) = 0;
                } else {
                    if (sinlam() < 0.5)
                        value(0, 0) = 2 * asin(sinlam()) + pi<T>();
                    else
                        value(0, 0) = 2 * acos(coslam()) + pi<T>();
                    value(0, 1) = -2 * coslam(1);
                }
                set(0, 0) = true;
                set(0, 1) = true;
            }

            //! Getter function
            inline T get_value(int u, int v) {
                if ((u < 0) || (v < 0) || (u > umax) || (v > vmax)) {
                    throw errors::IndexError("Invalid index in `H` "
                                             "integral evaluation.");
                } else if (!is_even(u) || ((coslam() == 0) && !is_even(v))) {
                    return T(0.0);
                } else if (!set(u, v)) {
                    if (u >= 2)
                        value(u, v) = (2 * coslam(u - 1) * sinlam(v + 1) +
                                      (u - 1) * get_value(u - 2, v)) / (u + v);
                    else
                        value(u, v) = (-2 * coslam(u + 1) * sinlam(v - 1) +
                                      (v - 1) * get_value(u, v - 2)) / (u + v);
                    set(u, v) = true;
                }
                return value(u, v);
            }

            //! Overload () to get the function value without calling `value()`
            inline T operator() (int u, int v) { return get_value(u, v); }

    };

    /**
    The helper primitive integral I_v

    */
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
            Vector<T> ivgamma;

        public:

            //! Constructor
            I(int lmax, Power<T>& ksq, T& k, T& kc, T& kkc, T& kap0) :
                    vmax(2 * lmax + 2), ksq(ksq), k(k), kc(kc),
                    kkc(kkc), kap0(kap0) {

                set = Vector<bool>::Zero(vmax + 1);
                value.resize(vmax + 1);

                // Pre-tabulate I_v for ksq >= 1
                ivgamma.resize(vmax + 1);
                for (int v = 0; v <= vmax; v++)
                    ivgamma(v) = root_pi<T>() *
                                 T(boost::math::tgamma_delta_ratio(
                                                Multi(v + 0.5), Multi(0.5)));

            }

            //! Reset flags and compute either `I_0` or `I_vmax`
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
                    T coeff = T(2.0 / (2 * vmax + 1));

                    // Add leading term to I_vmax:
                    T res = coeff;

                    // Now, compute higher order terms until
                    // desired precision is reached
                    int n = 1;
                    while ((n < STARRY_IJ_MAX_ITER) && (abs(error) > tol)) {
                        coeff *= (2.0 * n - 1.0) * 0.5 *
                                 (2 * n + 2 * vmax - 1) /
                                 (n * (2 * n + 2 * vmax + 1)) * ksq();
                        error = coeff;
                        res += coeff;
                        n++;
                    }

                    // Check for convergence
                    if (n == STARRY_IJ_MAX_ITER)
                        throw errors::ConvergenceError("Primitive integral "
                                                       "`I` did not converge.");

                    value(vmax) = ksq(vmax) * k * res;
                    set(vmax) = true;

                } else {

                    // Upward recursion: compute I_0
                    value(0) = kap0;
                    set(0) = true;

                }

            }

            //! Getter function
            inline T get_value(int v) {
                if ((v < 0) || (v > vmax))
                    throw errors::IndexError("Invalid index in the "
                                             "evaluation of the `I` primitive "
                                             "integral");
                if (ksq() >= 1) {
                    // Easy: these are constant & tabulated!
                    return ivgamma(v);
                } else if (!set(v)) {
                    if (set(vmax))
                        // Downward recursion (preferred)
                        value(v) = 2.0 / (2 * v + 1) * ((v + 1) *
                                   get_value(v + 1) + ksq(v) * kkc);
                    else if (set(0))
                        // Upward recursion
                        value(v) = ((2 * v - 1) / 2.0 *
                                    get_value(v - 1) - ksq(v - 1) * kkc) / v;
                    else
                        throw errors::ConvergenceError("Primitive integral "
                                                       "`I` did not converge.");
                    set(v) = true;
                }
                return value(v);
            }

            //! Overload () to get the function value w/o calling `get_value()`
            inline T operator() (int v) { return get_value(v); }

    };

    /**
    The helper primitive integral J_v

    */
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

        public:

            //! Constructor
            J(int lmax, Elliptic<T>& ELL, Power<T>& ksq,
              Power<T>& two, T& k, T& kc, T& invksq) :
                    vmax(max(1, 2 * lmax - 1)), ELL(ELL), ksq(ksq),
                    two(two), k(k), kc(kc), invksq(invksq) {

                set = Vector<bool>::Zero(vmax + 1);
                value.resize(vmax + 1);

                // These are the values of `v` we pre-compute
                // on downward recursion
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

            /**
            Reset flags and compute J_vmax and
            J_{vmax - 1} with a series expansion

            */
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
                            coeff = pi<T>();
                            for (int i = 1; i <= v; i++) coeff *= (1 - 0.5 / i);
                        } else {
                            coeff = 3 * pi<T>() / (two(2 + v) *
                                    tables::factorial<T>(v + 2));
                            for (int i = 1; i <= v; i++) coeff *= (2.0 * i - 1);
                        }

                        // Add leading term to J_vmax:
                        T res = coeff;

                        // Now, compute higher order terms until
                        // desired precision is reached
                        int n = 1;
                        T error = T(INFINITY);
                        while ((n < STARRY_IJ_MAX_ITER) && (abs(error) > tol)) {
                            if (ksq() >= 1)
                                coeff *= (1.0 - 2.5 / n) *
                                         (1.0 - 0.5 / (n + v)) * invksq;
                            else
                                coeff *= (2.0 * n - 1.0) *
                                         (2.0 * (n + v) - 1.0) * 0.25 /
                                         (n * (n + v + 2)) * ksq();
                            error = coeff;
                            res += coeff;
                            n++;
                        }

                        // Check convergence
                        if (n == STARRY_IJ_MAX_ITER)
                            throw errors::ConvergenceError("Primitive integral `J` did not converge.");

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
                        value(0) = (2.0 / 3.0) *
                                   (2 * (2 - invksq) * ELL.E() -
                                    (1 - invksq) * ELL.K());
                        value(1) = (2.0 / 15.0) *
                                   ((-3 * ksq() + 13 - 8 * invksq) * ELL.E() +
                                    (1 - invksq) * (3 * ksq() - 4) * ELL.K());

                    } else {

                        // Upward recursion: compute J_0 and J_1
                        value(0) = 2.0 / (3.0 * ksq() * k) *
                                   (2 * (2 * ksq() - 1) * ELL.E() +
                                    (1 - ksq()) * (2 - 3 * ksq()) * ELL.K());
                        value(1) = 2.0 / (15.0 * ksq() * k) *
                                   ((-3 * ksq(2) + 13 * ksq() - 8) * ELL.E() +
                                    (1 - ksq()) * (8 - 9 * ksq()) * ELL.K());

                    }

                    set(0) = true;
                    set(1) = true;

                }

            }

            //! Getter function
            inline T get_value(int v) {
                if ((v < 0) || (v > vmax))
                    throw errors::IndexError("Invalid index in the "
                                             "evaluation of primitive "
                                             "integral `J`.");
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
                        value(v) = (f1 * get_value(v - 1) - f2 *
                                    get_value(v - 2)) / (2 * v + 3);
                    } else {
                        throw errors::ConvergenceError("Primitive integral "
                                                       "`J` did not converge.");
                    }
                    set(v) = true;
                }
                return value(v);
            }

            //! Overload () to get the function value w/o calling `get_value()`
            inline T operator() (int v) { return get_value(v); }

    };

    /**
    The helper primitive integral K_{u,v}

    */
    template <typename T>
    inline T K(Greens<T>& G, int u, int v) {
        T res(0.0);
        for (int i = 0; i < u + v + 1; i++)
            res += G.A_P(i, u, v) * G.I_P(i + u);
        return res;
    }

    /**
    The helper primitive integral L_{u,v}^(t)

    */
    template <typename T>
    inline T L(Greens<T>& G, int u, int v, int t) {
        T res(0.0);
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

    /**
    The primitive integral P(G_n)

    */
    template <typename T>
    inline T P(Greens<T>& G){
        if (is_even(G.mu, 2)) {
            return 2 * G.twor(G.l + 2) * K(G, (G.mu + 4) / 4, G.nu / 2);
        } else if ((G.mu == 1) && is_even(G.l)) {
            return LFac(G) * (L(G, (G.l - 2) / 2, 0, 0) - 2 *
                   L(G, (G.l - 2) / 2, 0, 1));
        } else if ((G.mu == 1) && !is_even(G.l)) {
            return LFac(G) * (L(G, (G.l - 3) / 2, 1, 0) - 2 *
                   L(G, (G.l - 3) / 2, 1, 1));
        } else if (is_even(G.mu - 1, 2)) {
            return 2 * LFac(G) * L(G, (G.mu - 1) / 4, (G.nu - 1) / 2, 0);
        } else {
            return T(0.0);
        }
    }

    /**
    The primitive integral Q(G_n)

    */
    template <typename T>
    inline T Q(Greens<T>& G){
        // From the integral definition of Q, the result is zero
        // unless both mu/2 and nu/2 are even when the occultor
        // is not touching the limb of the planet.
        if ((G.coslam() == 0) && (!is_even(G.mu, 2) || !is_even(G.nu, 2)))
            return T(0.0);
        else if (!is_even(G.mu, 2))
            return T(0.0);
        else {
            return G.H_Q((G.mu + 4) / 2, G.nu / 2);
        }
    }

    /**
    Greens integration housekeeping data

    */
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

            // Off switch for certain elements of s^T
            Vector<bool> skip;

            // Constructor
            explicit Greens(int lmax) :
                   lmax(lmax),
                   ksq(T(0.0)),
                   twor(T(0.0)),
                   delta(T(0.0)),
                   sinlam(T(0.0)),
                   coslam(T(0.0)),
                   two(T(2.0)),
                   ELL((*this).ksq, (*this).invksq),
                   H_Q(lmax, (*this).sinlam, (*this).coslam),
                   I_P(lmax, (*this).ksq, (*this).k, (*this).kc, (*this).kkc,
                       (*this).kap0),
                   J_P(lmax, (*this).ELL, (*this).ksq, (*this).two, (*this).k,
                       (*this).kc, (*this).invksq),
                   A_P(lmax, (*this).delta),
                   sT(VectorT<T>::Zero((lmax + 1) * (lmax + 1))),
                   skip(Vector<bool>::Zero((lmax + 1) * (lmax + 1))) { }

        // Compute the solution vector
        inline void compute(const T& b_, const T& r_);
        inline void quad(const T& b_, const T& r_);

    };

    /**
    Compute the solution vector for a transit of a
    quadratically limb-darkened star.

    This code has been stripped of a lot of the overhead for speed, so
    it may be a bit opaque. Basically, for a quadratically limb-darkened star,
    the only terms that matter in the Greens polynomial basis are those at
    indices n = 0, 2, 4, and 8. We therefore only compute those indices of the
    solution vector -- we do it directly, without any recurrence relations.
    NOTE: The term g(4) is *always* 1/3 * g(8), so we fold
    that into `s8` below. The solution vector `sT` is therefore NOT
    TECHNICALLY CORRECT in this case, so users should be careful when calling
    `map.getS()`, as the result will be misleading.

    */
    template <class T>
    inline void Greens<T>::quad(const T& b_, const T& r_) {
        b = b_;
        r = r_;
        T b2 = b * b;
        T r2 = r * r;
        if ((abs(1 - r) < b) && (b < 1 + r)) {
            T sp = (1 - r2 - b2) / (2 * b * r);
            T cp = sqrt(1 - sp * sp);
            T sl = (1 - r2 + b2) / (2 * b);
            T cl = sqrt(1 - sl * sl);
            T l2 = asin(sl) + 0.5 * pi<T>();
            T p2  = asin(sp) + 0.5 * pi<T>();
            T cpsp = cp * sp;
            T clsl = cl * sl;
            sT(0) = l2 + clsl - r2 * (p2 + cpsp);
            if (lmax > 1) {
                T r3 = r * r2;
                T r4 = r2 * r2;
                T sp2 = sp * sp;
                T cp2 = cp * cp;
                T cp3 = cp * cp2;
                T sp3 = sp * sp2;
                T cl3 = cl * cl * cl;
                T sl3 = sl * sl * sl;
                sT(8) = 0.5 * l2 + (1. / 3.) * clsl - (1. / 6.) * cl3 *
                        sl + (1. / 6.) * cl * sl3 -
                        (r2 * b2 * (p2 + cpsp) - r3 * b * cp *
                        (1. + (1. / 3.) * cp2 - sp2) +
                         r4 * (0.5 * p2 + (1. / 3.) * cpsp -
                         (1. / 6.) * cp3 * sp + (1. / 6.) * cp * sp3));
            }
        } else {
            sT(0) = pi<T>() * (1 - r2);
            if (lmax > 1)
                sT(8) = 0.5 * pi<T>() - pi<T>() * r2 * (0.5 * r2 + b2);
        }
        if (lmax > 0) {
            if ((b == 0) || (r == 0)) {
                ksq.reset(T(INFINITY));
                invksq = 0;
            } else {
                ksq.reset((1 - (b - r)) * (1 + (b - r)) / (4 * b * r));
                invksq = (4 * b * r) / ((1 - (b - r)) * (1 + (b - r)));
            }
            ELL.reset();
            sT(2) = s2(*this);
        }
    }

    /**
    Compute the `s^T` occultation solution vector

    */
    template <class T>
    inline void Greens<T>::compute(const T& b_, const T& r_) {

        // Initialize the basic variables
        int n = 0;
        b = b_;
        r = r_;
        T ksq_;
        if ((b == 0) || (r == 0)) {
            ksq_ = T(INFINITY);
            k = T(INFINITY);
            kc = 1;
            kkc = T(INFINITY);
            invksq = 0;
            kap0 = 0;
        } else {
            ksq_ = (1 - (b - r)) * (1 + (b - r)) / (4 * b * r);
            invksq = (4 * b * r) / ((1 - (b - r)) * (1 + (b - r)));
            k = sqrt(ksq_);
            if (ksq_ > 1) {
                kc = sqrt(1 - invksq);
                kkc = k * kc;
                kap0 = 0; // Not used!
            } else {
                kc = sqrt(abs(((b + r) * (b + r) - 1) / (4 * b * r)));

                // Eric Agol's "kite" method to compute a stable
                // version of k * kc and I_0 = kap0
                // Used to be
                //   G.kkc = G.k * G.kc;
                //   G.kap0 = 2 * acos(G.kc);
                T p0 = T(1.0), p1 = b, p2 = r;
                if (p0 < p1) swap(p0, p1);
                if (p1 < p2) swap(p1, p2);
                if (p0 < p1) swap(p0, p1);
                T kite_area2 = sqrt((p0 + (p1 + p2)) * (p2 - (p0 - p1)) *
                                    (p2 + (p0 - p1)) * (p0 + (p1 - p2)));
                kkc = kite_area2 / (4 * b * r);
                kap0 = atan2(kite_area2, (r - 1) * (r + 1) + b * b);
            }
        }
        ksq.reset(ksq_);
        fourbr32 = pow(4 * b * r, 1.5);
        lfac = pow(1 - (b - r) * (b - r), 1.5);

        // Powers of basic variables
        twor.reset(2 * r);
        delta.reset((b - r) / (2 * r));
        if ((abs(1 - r) < b) && (b < 1 + r)) {

            sinlam.reset(0.5 * ((1. / b) + (b - r) * (1. + r / b)));
            coslam.reset(sqrt(1 - sinlam() * sinlam()));

            // Stability override
            if (sinlam() > 0.5) {
                T delta = 1 - (b + r);
                T eps = ((r / b) * delta + (delta * delta) / (2 * b));
                sinlam.reset(1 + eps);
                coslam.reset(sqrt(-eps * (2 + eps)));
            }

        } else {
            sinlam.reset(T(1.0));
            coslam.reset(T(0.0));
        }

        // Initialize our storage classes.
        // For 0.5 < ksq < 2 we do upward recursion,
        // otherwise we do downward recursion.
        ELL.reset();
        H_Q.reset();
        I_P.reset(ksq_ < 0.5);
        J_P.reset((ksq_ < 0.5) || (ksq_ > 2));
        A_P.reset();

        // Populate the solution vector
        for (l = 0; l < lmax + 1; l++) {
            for (m = -l; m < l + 1; m++) {
                mu = l - m;
                nu = l + m;
                if (!skip(n)) {

                    // Special case
                    if ((l == 1) && (m == 0))
                        sT(n) = s2(*this);

                    // These terms are zero because they are proportional to
                    // odd powers of x, so we don't need to compute them!
                    else if ((is_even(mu - 1)) && (!is_even((mu - 1) / 2)))
                        sT(n) = 0;

                    // These terms are also zero for the same reason
                    else if ((is_even(mu)) && (!is_even(mu / 2)))
                        sT(n) = 0;

                    // Business as usual
                    else
                        sT(n) = Q(*this) - P(*this);

                } else {

                    // The map coefficient is zero, so we don't need this term
                    sT(n) = 0;

                }
                n++;
            }
        }
    }

} // namespace solver

#endif
