/**
Limb darkening utilities from Agol & Luger (2018).

*/

#ifndef _STARRY_LIMBDARK_H_
#define _STARRY_LIMBDARK_H_

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include "ellip.h"
#include "errors.h"
#include "lld.h"
#include "tables.h"

namespace limbdark {

    using std::abs;
    using std::max;
    using std::swap;
    using namespace utils;

    /**
    The `c_n` basis normalization constant.

    */
    template <class T>
    inline T normC(const Vector<T>& c) {
        return 1.0 / (pi<T>() * (c(0) + 2.0 * c(1) / 3.0));
    }

    /**
    Transform the `u_n` coefficients to `c_n`, which are coefficients
    of the basis in which the `P(G_n)` functions are computed.

    */
    template <class T>
    inline Vector<T> computeC(const Vector<T>& u) {
        T bcoeff;
        size_t N = u.size();
        Vector<T> c(N);
        Vector<T> a(N);

        // Compute the a_n coefficients
        a.setZero();
        a(0) = 1.0;
        for (size_t i = 1; i < N; ++i) {
            bcoeff = 1.0;
            int sgn = 1;
            for (size_t j = 0; j <= i; ++j) {
                a(j) -= u(i) * bcoeff * sgn;
                sgn *= -1;
                bcoeff *= (T(i - j) / (j + 1));
            }
        }

        // Now, compute the c_n coefficients
        for (size_t j = N - 1; j >= 2; --j) {
            if (j >= N - 2)
                c(j) = a(j) / (j + 2);
            else
                c(j) = a(j) / (j + 2) + c(j + 2);
        }
        if (N >= 4)
            c(1) = a(1) + 3 * c(3);
        else
            c(1) = a(1);
        if (N >= 3)
            c(0) = a(0) + 2 * c(2);
        else
            c(0) = a(0);

        // The total flux is given by `(S . c) * normC`
        return c;

    }

    /**
    Transform the u_n coefficients to `c_n`, which are coefficients
    of the basis in which the `P(G_n)` functions are computed.
    Also compute the derivative matrix `dc / du`.

    */
    template <class T>
    inline Vector<T> computeC(const Vector<T>& u, Matrix<T>& dcdu) {
        T bcoeff;
        size_t N = u.size();
        Vector<T> c(N);
        Vector<T> a(N);
        Matrix<T> dadu;

        // Compute the a_n coefficients
        a.setZero();
        a(0) = 1.0;
        dadu.setZero(N, N);
        for (size_t i = 1; i < N; ++i) {
            bcoeff = 1.0;
            int sgn = 1;
            for (size_t j = 0; j <= i; ++j) {
                a(j) -= u(i) * bcoeff * sgn;
                dadu(j, i) -= bcoeff * sgn;
                sgn *= -1;
                bcoeff *= (T(i - j) / (j + 1));
            }
        }

        // Now, compute the c_n coefficients
        dcdu.setZero(N, N);
        for (size_t j = N - 1; j >= 2; --j) {
            if (j >= N - 2) {
                c(j) = a(j) / (j + 2);
                dcdu.block(j, 0, 1, N) = dadu.block(j, 0, 1, N) / (j + 2);
            } else {
                c(j) = a(j) / (j + 2) + c(j + 2);
                dcdu.block(j, 0, 1, N) = dadu.block(j, 0, 1, N) / (j + 2) +
                                         dcdu.block(j + 2, 0, 1, N);
            }
        }

        if (N >= 4) {
            c(1) = a(1) + 3 * c(3);
            dcdu.block(1, 0, 1, N) = dadu.block(1, 0, 1, N) +
                                     3 * dcdu.block(3, 0, 1, N);
        } else {
            c(1) = a(1);
            dcdu.block(1, 0, 1, N) = dadu.block(1, 0, 1, N);
        }

        if (N >= 3) {
            c(0) = a(0) + 2 * c(2);
            dcdu.block(0, 0, 1, N) = dadu.block(0, 0, 1, N) +
                                     2 * dcdu.block(2, 0, 1, N);
        } else {
            c(0) = a(0);
            dcdu.block(0, 0, 1, N) = dadu.block(0, 0, 1, N);
        }

        // The total flux is given by `(S . c) * normC`
        return c;

    }

    /**
    Greens integration housekeeping data

    */
    template <class T>
    class GreensLimbDark {

        public:

            // Indices
            int lmax;

            // Basic variables
            T b;
            T r;
            T k;
            T ksq;
            T kc;
            T kkc;
            T kap0;
            T invksq;
            T fourbr;
            T invfourbr;
            T E;
            T K;
            T rmb;
            T twob;
            T Sn;

            // Primitive matrices/vectors
            std::vector<T> ivgamma;
            std::vector<T> I;
            std::vector<T> J;
            int ivmax;
            int jvmax;

            // The solution vector
            VectorT<T> S;

            // Constructor
            explicit GreensLimbDark(int lmax) :
                lmax(lmax),
                S(VectorT<T>::Zero(lmax + 1)) {

                   // Figure out I and J dims
                   if (is_even(lmax + 1))
                       ivmax = (lmax + 1) / 2 + 2;
                   else
                       ivmax = lmax / 2 + 2;
                   jvmax = ivmax;
                   I.resize(ivmax + 1);
                   J.resize(jvmax + 1);

                   // Pre-tabulate I for ksq >= 1
                   ivgamma.resize(ivmax + 1);
                   for (int v = 0; v <= ivmax; v++)
                       ivgamma[v] = root_pi<T>() *
                                    T(boost::math::tgamma_delta_ratio(
                                      Multi(v + 0.5), Multi(0.5)));

            }

            inline void compute(const T& b_, const T& r_);
            inline void computeI();
            inline void computeJ();
            inline void computeEK();

    };

    /**

    */
    template <class T>
    inline void GreensLimbDark<T>::computeI() {

        if (ksq >= 1) {

            I = ivgamma;

        } else {

            // Downward recursion
            if (ksq < 0.5) {

                T tol = mach_eps<T>() * ksq;
                T error = T(INFINITY);

                // Computing leading coefficient (n=0):
                T coeff = T(2.0 / (2 * ivmax + 1));

                // Add leading term to I_ivmax:
                T res = coeff;

                // Now, compute higher order terms until
                // desired precision is reached
                int n = 1;
                while ((n < STARRY_IJ_MAX_ITER) && (abs(error) > tol)) {
                    coeff *= (2.0 * n - 1.0) * 0.5 *
                             (2 * n + 2 * ivmax - 1) /
                             (n * (2 * n + 2 * ivmax + 1)) * ksq;
                    error = coeff;
                    res += coeff;
                    n++;
                }

                // Check for convergence
                if (n == STARRY_IJ_MAX_ITER)
                    throw errors::ConvergenceError("Primitive integral "
                                                   "`I` did not converge.");

                I[ivmax] = pow(ksq, ivmax) * k * res;

                // Remaining terms
                for (int v = ivmax - 1; v >= 0; --v)
                    I[v] = 2.0 / (2 * v + 1) * ((v + 1) * I[v + 1] + pow(ksq, v) * kkc);

            // Upward recursion
            } else {

                I[0] = kap0;
                for (int v = 1; v <= ivmax; ++v)
                    I[v] = ((2 * v - 1) / 2.0 * I[v - 1] - pow(ksq, v - 1) * kkc) / v;

            }

        }

    }

    /**

    */
    template <class T>
    inline void GreensLimbDark<T>::computeJ() {

        // Downward recursion
        if ((ksq < 0.5) || (ksq > 2)) {

            T tol, coeff, res, error, f1, f2, f3;
            int i, n, v;
            if (ksq >= 1)
                tol = mach_eps<T>() * invksq;
            else
                tol = mach_eps<T>() * ksq;

            // Compute the highest two values
            for (v = jvmax; v >= jvmax - 1; --v) {

                // Computing leading coefficient (n=0):
                if (ksq >= 1) {
                    coeff = pi<T>();
                    for (i = 1; i <= v; ++i)
                        coeff *= (1 - 0.5 / i);
                } else {
                    coeff = 3 * pi<T>() / (pow(2, 2 + v) * tables::factorial<T>(v + 2));
                    for (i = 1; i <= v; ++i)
                        coeff *= (2.0 * i - 1);
                }

                // Add leading term to J_jvmax:
                res = coeff;

                // Now, compute higher order terms until
                // desired precision is reached
                n = 1;
                error = T(INFINITY);
                while ((abs(error) > tol) && (n < STARRY_IJ_MAX_ITER)) {
                    if (ksq >= 1)
                        coeff *= (1.0 - 2.5 / n) *
                                 (1.0 - 0.5 / (n + v)) * invksq;
                    else
                        coeff *= (2.0 * n - 1.0) *
                                 (2.0 * (n + v) - 1.0) * 0.25 /
                                 (n * (n + v + 2)) * ksq;
                    error = coeff;
                    res += coeff;
                    n++;
                }

                // Check convergence
                if (n == STARRY_IJ_MAX_ITER)
                    throw errors::ConvergenceError("Primitive integral `J` did not converge.");

                // Store the result
                if (ksq >= 1)
                    J[v] = res;
                else
                    J[v] = pow(ksq, v) * k * res;

            }

            // Compute all other values
            for (v = jvmax - 2; v >= 0; --v) {
                if (ksq < 1) {
                    f2 = ksq * (2 * v + 1);
                    f1 = 2 * (3 + v + ksq * (1 + v)) / f2;
                    f3 = (2 * v + 7) / f2;
                } else {
                    f3 = (2. * v + 7) / (2. * v + 1) * invksq;
                    f1 = (2. / (2. * v + 1)) * ((3 + v) * invksq + 1 + v);
                }
                J[v] = f1 * J[v + 1] - f3 * J[v + 2];
            }

        // Upward recursion
        } else {

            T f1, f2;
            int v;

            // First two values
            if (ksq >= 1) {
                J[0] = (2.0 / 3.0) * (2 * (2 - invksq) * E -
                                      (1 - invksq) * K);
                J[1] = (2.0 / 15.0) * ((-3 * ksq + 13 - 8 * invksq) * E +
                                       (1 - invksq) * (3 * ksq - 4) * K);
            } else {
                J[0] = 2.0 / (3.0 * ksq * k) * (2 * (2 * ksq - 1) * E +
                                               (1 - ksq) * (2 - 3 * ksq) * K);
                J[1] = 2.0 / (15.0 * ksq * k) * ((-3 * ksq * ksq + 13 * ksq - 8) * E +
                                                (1 - ksq) * (8 - 9 * ksq) * K);
            }

            // Higher order values
            for (v = 2; v <= jvmax; ++v) {
                f1 = 2 * (v + (v - 1) * ksq + 1);
                f2 = ksq * (2 * v - 3);
                J[v] = (f1 * J[v - 1] - f2 * J[v - 2]) / (2 * v + 3);
            }

        }

    }

    /**

    */
    template <class T>
    inline void GreensLimbDark<T>::computeEK() {
        if (unlikely((invksq == 0) || (ksq == 1)))
            K = 0;
        else if (ksq < 1)
            K = ellip::K(ksq);
        else
            K = ellip::K(invksq);
        if (unlikely(invksq == 0))
            E = 0;
        else if (unlikely(ksq == 1))
            E = 1;
        else if (ksq < 1)
            E = ellip::E(ksq);
        else
            E = ellip::E(invksq);
    }

    /**
    Compute the `s^T` occultation solution vector

    */
    template <class T>
    inline void GreensLimbDark<T>::compute(const T& b, const T& r) {

        // Initialize the basic variables
        T fac, amp;
        int n0;
        int sgn;
        int i, n;
        rmb = r - b;
        twob = 2 * b;
        fourbr = 4 * b * r;
        invfourbr = 1.0 / fourbr;
        if (unlikely((b == 0) || (r == 0))) {
            ksq = T(INFINITY);
            k = T(INFINITY);
            kc = 1;
            kkc = T(INFINITY);
            invksq = 0;
            kap0 = 0;
            S(0) = pi<T>() * (1 - r * r);
        } else {
            ksq = (1 - (b - r)) * (1 + (b - r)) * invfourbr;
            invksq = fourbr / ((1 - (b - r)) * (1 + (b - r)));
            k = sqrt(ksq);
            if (ksq > 1) {
                kc = sqrt(1 - invksq);
                kkc = k * kc;
                kap0 = 0; // Not used!
                S(0) = pi<T>() * (1 - r * r);
            } else {
                kc = sqrt(abs(((b + r) * (b + r) - 1) * invfourbr));
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
                kkc = kite_area2 * invfourbr;
                kap0 = atan2(kite_area2, (r - 1) * (r + 1) + b * b);
                T kap1 = atan2(kite_area2, (1 - r) * (1 + r) + b * b);
                T Alens = kap1 + r * r * kap0 - kite_area2 * 0.5;
                S(0) = pi<T>() - Alens;
            }
        }

        // Special case
        if (lmax == 0) return;

        // Compute the elliptic integrals
        computeEK();

        // Compute the linear limb darkening term
        S(1) = lld::s2(b, r, ksq, K, E);

        // Special case
        if (lmax == 1) return;

        // Special case
        if (unlikely(b == 0)) {
            T term = 1 - r * r;
            T fac = sqrt(term);
            for (int n = 2; n < lmax + 1; ++n) {
                S(n) = -term * r * r * 2 * pi<T>();
                term *= fac;
            }
            return;
        }

        // Even higher order terms
        computeI();
        fac = -2 * r * fourbr;
        n0 = 1;
        sgn = -1;
        for (n = 2; n < lmax + 1; n += 2) {
            Sn = 0;
            amp = sgn;
            for (i = 0; i <= n0; ++i) {
                Sn += amp * (rmb * I[n0 - i] + twob * I[n0 - i + 1]);
                amp *= -ksq * (n0 - i) / (i + 1.0);
            }
            S(n) = Sn * fac;
            fac *= fourbr;
            ++n0;
            sgn *= -1;
        }

        // Special case
        if (lmax == 2) return;

        // Odd higher order terms
        computeJ();
        fac = -2 * r * pow(1 - (b - r) * (b - r), 1.5);
        n0 = 0;
        sgn = 1;
        for (n = 3; n < lmax + 1; n += 2) {
            Sn = 0;
            amp = sgn;
            for (i = 0; i <= n0; ++i) {
                Sn += amp * (rmb * J[n0 - i] + twob * J[n0 - i + 1]);
                amp *= -ksq * (n0 - i) / (i + 1.0);
            }
            S(n) = Sn * fac;
            fac *= fourbr;
            ++n0;
            sgn *= -1;
        }

    }

} // namespace limbdark

#endif
