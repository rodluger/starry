/**
Limb darkening utilities from Agol & Luger (2018).

TODO: Loop downward in v until J[v] !=0
TODO: Test all special cases

*/

#ifndef _STARRY_LIMBDARK_H_
#define _STARRY_LIMBDARK_H_

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include "ellip.h"
#include "errors.h"
#include "tables.h"

namespace starry {
namespace limbdark {

    using std::abs;
    using std::max;
    using std::swap;
    using namespace utils;

    /**
    The linear limb darkening flux term.

    */
    template <typename T>
    inline T s2(const T& b, const T& r, 
                const T& third, const T& b2, const T& r2,
                const T& ksq, const T& kcsq, const T& kc, const T& invksq,
                const T& onembmr2, const T& onembmr2inv, const T& sqonembmr2,
                T& Eofk, T& Em1mKdm, T& ds2db, T& ds2dr, bool gradient=false) {
        T Lambda1 = 0;
        if ((b >= 1.0 + r) ||  (r == 0.0)) {
            // No occultation (Case 1)
            Lambda1 = 0;
        } else if (b <= r - 1.0) {
            // Full occultation (Case 11)
            Lambda1 = 0;
        } else {
            if (unlikely(b == 0)) {
                // Case 10
                T sqrt1mr2 = sqrt(1.0 - r2);
                Lambda1 = -2.0 * pi<T>() * sqrt1mr2 * sqrt1mr2 * sqrt1mr2; 
                Eofk = 0.5 * pi<T>();
                Em1mKdm = 0.25 * pi<T>();
                if (gradient) {
                    ds2db = 0;
                    ds2dr = -2.0 * pi<T>() * sqrt1mr2;
                }
            } else if (unlikely(b == r)) {
                if (unlikely(r == 0.5)) {
                    // Case 6
                    Lambda1 = pi<T>() - 4.0 * third;
                    Eofk = 1.0;
                    Em1mKdm = 1.0;
                    if (gradient) {
                        ds2db = 2.0 * third;
                        ds2dr = -2.0;
                    }
                } else if (r < 0.5) {
                    // Case 5
                    T m = 4 * r2;
                    Eofk = ellip::CEL(m, T(1.0), T(1.0), T(1.0 - m));
                    Em1mKdm = ellip::CEL(m, T(1.0), T(1.0), T(0.0));
                    Lambda1 = pi<T>() + 2.0 * third * ((2 * m - 3) * Eofk - m * Em1mKdm);
                    if (gradient) {
                        ds2db = -4.0 * r * third * (Eofk - 2 * Em1mKdm);
                        ds2dr = -4.0 * r * Eofk;
                    }
                } else {
                    // Case 7
                    T m = 4 * r2; 
                    T minv = 1.0 / m; 
                    Eofk = ellip::CEL(minv, T(1.0), T(1.0), T(1.0 - minv));
                    Em1mKdm = ellip::CEL(minv, T(1.0), T(1.0), T(0.0));
                    Lambda1 = pi<T>() + third / r * (-m * Eofk + (2 * m - 3) * Em1mKdm);
                    if (gradient) {
                        ds2db = 2 * third * (2 * Eofk - Em1mKdm);
                        ds2dr = -2 * Em1mKdm;
                    }
                }
            } else {
                T onembpr2 = (1 - r - b) * (1 + r + b);  
                T fourbr = 4 * b * r; 
                if (ksq < 1) {
                    // Case 2, Case 8
                    T sqbr = sqrt(b * r); 
                    T sqbrinv = 1.0 / sqbr;
                    T Piofk;
                    ellip::CEL(ksq, kc, T((b - r) * (b - r) * kcsq), T(0.0), T(1.0), T(1.0), T(3 * kcsq * (b - r) * (b + r)), kcsq, T(0.0), Piofk, Eofk, Em1mKdm);
                    Lambda1 = onembmr2 * (Piofk + (-3 + 6 * r2 + 2 * b * r) * Em1mKdm - fourbr * Eofk) * sqbrinv * third;
                    if (gradient) {
                        ds2db = 2 * r * onembmr2 * (-Em1mKdm + 2 * Eofk) * sqbrinv * third;
                        ds2dr = -2 * r * onembmr2 * Em1mKdm * sqbrinv;
                    }
                } else if (ksq > 1) {
                    // Case 3, Case 9
                    T bmrdbpr = (b - r) / (b + r); 
                    T mu = 3 * bmrdbpr * onembmr2inv;
                    T p = bmrdbpr * bmrdbpr * onembpr2 * onembmr2inv;
                    T Piofk;
                    ellip::CEL(invksq, kc, p, T(1 + mu), T(1.0), T(1.0), T(p + mu), kcsq, T(0.0), Piofk, Eofk, Em1mKdm);
                    Lambda1 = 2 * sqonembmr2 * (onembpr2 * Piofk - (4 - 7 * r2 - b2) * Eofk) * third;
                    if (gradient) {
                        ds2db = -4 * r * third * sqonembmr2 * (Eofk - 2 * Em1mKdm);
                        ds2dr = -4 * r * sqonembmr2 * Eofk;
                    }
                } else {
                    // Case 4
                    Lambda1 = 2 * acos(1.0 - 2.0 * r) - 4 * third * (3 + 2 * r - 8 * r2) * sqrt(r * (1 - r)) - 2 * pi<T>() * int(r > 0.5);
                    Eofk = 1.0;
                    Em1mKdm = 1.0;
                    if (gradient) {
                        ds2dr = -8 * r * sqrt(r * (1 - r));
                        ds2db = -ds2dr * third;
                    }
                }
            }
        }
        return ((1.0 - int(r > b)) * 2 * pi<T>() - Lambda1) * third;
    }

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
        dcdu.setZero(N - 1, N);
        for (size_t j = N - 1; j >= 2; --j) {
            if (j >= N - 2) {
                c(j) = a(j) / (j + 2);
                dcdu.transpose().block(j, 0, 1, N - 1) = dadu.block(j, 1, 1, N - 1) / (j + 2);
            } else {
                c(j) = a(j) / (j + 2) + c(j + 2);
                dcdu.transpose().block(j, 0, 1, N - 1) = dadu.block(j, 1, 1, N - 1) / (j + 2) +
                                             dcdu.transpose().block(j + 2, 0, 1, N - 1);
            }
        }

        if (N >= 4) {
            c(1) = a(1) + 3 * c(3);
            dcdu.transpose().block(1, 0, 1, N - 1) = dadu.block(1, 1, 1, N - 1) +
                                         3 * dcdu.transpose().block(3, 0, 1, N - 1);
        } else {
            c(1) = a(1);
            dcdu.transpose().block(1, 0, 1, N - 1) = dadu.block(1, 1, 1, N - 1);
        }

        if (N >= 3) {
            c(0) = a(0) + 2 * c(2);
            dcdu.transpose().block(0, 0, 1, N - 1) = dadu.block(0, 1, 1, N - 1) +
                                         2 * dcdu.transpose().block(2, 0, 1, N - 1);
        } else {
            c(0) = a(0);
            dcdu.transpose().block(0, 0, 1, N - 1) = dadu.block(0, 1, 1, N - 1);
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
            T kcsq;
            T kkc;
            T kap0;
            T kap1;
            T invksq;
            T fourbr;
            T invfourbr;
            T b2;
            T r2;
            T invr;
            T invb;
            T bmr;
            T bpr;
            T onembmr2;
            T onembmr2inv;
            T sqonembmr2;
            T kite_area2;
            T Sn;
            T third;
            T ds2db;
            T ds2dr;
            T Eofk;
            T Em1mKdm;
            T Piofk;

            // Powers of ksq
            std::vector<T> pow_ksq;

            // Primitive matrices/vectors
            std::vector<T> ivgamma;
            std::vector<T> I;
            std::vector<T> dIdk;
            std::vector<T> J;
            std::vector<T> dJdk;
            int ivmax;
            int jvmax;

            // Coefficients
            Vector<T> Icoeff;
            std::vector<Vector<T>> Jcoeff_largek;
            std::vector<Vector<T>> Jcoeff_smallk;
            std::vector<Vector<T>> dJdkcoeff_largek;
            std::vector<Vector<T>> dJdkcoeff_smallk;

            // The solution vector
            VectorT<T> S;
            VectorT<T> dSdb;
            VectorT<T> dSdr;

            // Constructor
            explicit GreensLimbDark(int lmax) :
                lmax(lmax),
                Jcoeff_largek(2),
                Jcoeff_smallk(2),
                dJdkcoeff_largek(2),
                dJdkcoeff_smallk(2),
                S(VectorT<T>::Zero(lmax + 1)),
                dSdb(VectorT<T>::Zero(lmax + 1)),
                dSdr(VectorT<T>::Zero(lmax + 1)) {

                    // Figure out I and J dims
                    if (is_even(lmax + 1))
                        ivmax = (lmax + 1) / 2 + 2;
                    else
                        ivmax = lmax / 2 + 2;
                    jvmax = ivmax;
                    I.resize(ivmax + 1);
                    J.resize(jvmax + 1);
                    dIdk.resize(ivmax + 1);
                    dJdk.resize(jvmax + 1);

                    // Powers of ksq
                    pow_ksq.resize(jvmax + 1);
                    pow_ksq[0] = 1;

                    // Pre-tabulate I and J coeffs
                    computeIcoeffs();
                    computeJcoeffs();

                    // Pre-tabulate I for ksq >= 1
                    ivgamma.resize(ivmax + 1);
                    for (int v = 0; v <= ivmax; v++)
                        ivgamma[v] = root_pi<T>() *
                                        T(boost::math::tgamma_delta_ratio(
                                        Multi(v + 0.5), Multi(0.5)));

                    // Constants
                    third = T(1.0) / T(3.0);

            }

            inline void compute(const T& b_, const T& r_, bool gradient=false);
            inline void computeI(bool gradient=false);
            inline void computeJ(bool gradient=false);
            inline void computeIcoeffs();
            inline void computeJcoeffs();

    };

    /**
    Pre-compute the coefficients in the series expansion of
    the I integral for v = vmax. This is
    done a single time when the class is instantiated.

    */
    template <class T>
    inline void GreensLimbDark<T>::computeIcoeffs() {
        Icoeff.resize(STARRY_IJ_MAX_ITER + 1);
        Icoeff[0] = 2.0 / (2 * ivmax + 1);
        for (int n = 1; n <= STARRY_IJ_MAX_ITER; ++n) {
            Icoeff[n] = (2.0 * n - 1.0) * 0.5 * (2 * n + 2 * ivmax - 1) / (n * (2 * n + 2 * ivmax + 1));
        }
    }

    /**

    */
    template <class T>
    inline void GreensLimbDark<T>::computeI(bool gradient) {

        if (ksq >= 1) {

            I = ivgamma;

        } else {

            // Pre-compute powers of ksq
            for (int v = 1; v <= jvmax; ++v)
                pow_ksq[v] = pow_ksq[v - 1] * ksq;

            // Downward recursion
            if (ksq < 0.5) {

                T tol = mach_eps<T>() * ksq;
                T error = T(INFINITY);

                // Leading coefficient (n=0):
                T coeff = Icoeff[0];
                T res = coeff;

                // Now, compute higher order terms until
                // desired precision is reached
                int n = 1;
                while ((n < STARRY_IJ_MAX_ITER) && (abs(error) > tol)) {
                    coeff *= Icoeff[n] * ksq;
                    error = coeff;
                    res += coeff;
                    n++;
                }

                // Check for convergence
                if (n == STARRY_IJ_MAX_ITER)
                    throw errors::ConvergenceError("Primitive integral "
                                                   "`I` did not converge.");

                I[ivmax] = pow_ksq[ivmax] * k * res;

                // Remaining terms
                for (int v = ivmax - 1; v >= 0; --v)
                    I[v] = 2.0 / (2 * v + 1) * ((v + 1) * I[v + 1] + pow_ksq[v] * kkc);

            // Upward recursion
            } else {

                I[0] = kap0;
                for (int v = 1; v <= ivmax; ++v)
                    I[v] = (0.5 * (2 * v - 1) * I[v - 1] - pow_ksq[v - 1] * kkc) / v;

            }

        }

    }

    /**
    Pre-compute the coefficients in the series expansion of
    the J integral for v = vmax and v = vmax - 1. This is
    done a single time when the class is instantiated.

    */
    template <class T>
    inline void GreensLimbDark<T>::computeJcoeffs() {

        T coeff;
        int n, v;

        // Pre-compute for vmax and vmax - 1
        for (int j = 0; j < 2; ++j) {

            //
            v = jvmax - j;

            // Allocate mem
            Jcoeff_largek[j].resize(STARRY_IJ_MAX_ITER + 1); 
            Jcoeff_smallk[j].resize(STARRY_IJ_MAX_ITER + 1); 
            dJdkcoeff_largek[j].resize(STARRY_IJ_MAX_ITER + 1); 
            dJdkcoeff_smallk[j].resize(STARRY_IJ_MAX_ITER + 1); 

            // ksq < 1
            coeff = 0.75 * pi<T>() / tables::factorial<T>(v + 2);
            for (int i = 2; i <= 2 * v; i += 2)
                coeff *= 0.5 * (i - 1);
            Jcoeff_smallk[j](0) = coeff;
            dJdkcoeff_smallk[j](0) = coeff * (2 * v + 1);
            for (int i = 1; i <= STARRY_IJ_MAX_ITER; ++i) {
                n = 2 * i;
                coeff *= (n - 1) * (n + 2 * v - 1);
                coeff /= n * (n + 2 * v + 4);
                Jcoeff_smallk[j](i) = coeff;
                dJdkcoeff_smallk[j](i) = coeff * (n + 2 * v + 1);
            }

            // ksq >= 1
            coeff = pi<T>();
            for (int i = 2; i <= 2 * v; i += 2)
                coeff *= (i - 1.0) / i;
            Jcoeff_largek[j](0) = coeff;
            dJdkcoeff_largek[j](0) = 0;
            for (int i = 1; i <= STARRY_IJ_MAX_ITER; ++i) {
                n = 2 * i;
                coeff *= (n - 5) * (n + 2 * v - 1);
                coeff /= n * (n + 2 * v);
                Jcoeff_largek[j](i) = coeff;
                dJdkcoeff_largek[j](i) = -n * coeff;
            }

        }

    }

    /**

    */
    template <class T>
    inline void GreensLimbDark<T>::computeJ(bool gradient) {

        // Downward recursion
        if ((ksq < 0.5) || (ksq > 2)) {
            
            T tol;
            T Jv, dJvdk;
            T k2n, term, dtermdk;
            T f1, f2, f3;
            int n, v;

            // Compute the highest two values
            for (int j = 0; j < 2; ++j) {
                v = jvmax - j;
                if (ksq < 1) {
                    tol = mach_eps<T>() * ksq;
                    // Constant term
                    Jv = Jcoeff_smallk[j](0);
                    dJvdk = dJdkcoeff_smallk[j](0);
                    k2n = 1.0;
                    term = 0; 
                    dtermdk = 0;
                    // Higher order terms
                    for (n = 1; n <= STARRY_IJ_MAX_ITER; ++n) {
                        k2n *= ksq;
                        term = k2n * Jcoeff_smallk[j](n);
                        Jv += term;
                        dtermdk = k2n * dJdkcoeff_smallk[j](n);
                        dJvdk += dtermdk;
                        if (abs(term) < tol) break;
                    }
                    term = pow(ksq, v);
                    dJvdk *= term;
                    Jv *= term * k;
                } else {
                    tol = mach_eps<T>() * invksq;
                    Jv = Jcoeff_largek[j](0);
                    dJvdk = dJdkcoeff_largek[j](0);
                    k2n = 1; 
                    term = 0;
                    for (n = 1; n <= STARRY_IJ_MAX_ITER; ++n) {
                        k2n *= invksq;
                        term = k2n * Jcoeff_largek[j](n);
                        Jv += term;
                        dtermdk = k2n * dJdkcoeff_largek[j](n);
                        dJvdk += dtermdk;
                        if (abs(term) < tol) break;
                    }
                    dJvdk /= k;
                }
                if (n == STARRY_IJ_MAX_ITER)
                    throw errors::ConvergenceError("Primitive integral `J` did not converge.");
                J[v] = Jv;
                dJdk[v] = dJvdk;
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
                if (gradient)
                    dJdk[v] = (dJdk[v + 1] + (3.0 * k * invksq) * J[v + 1]) * invksq;
            }

        // Upward recursion
        } else {

            T f1, f2;
            int v;

            // First two values
            if (ksq < 1) {
                J[0] = 2.0 / (3.0 * ksq * k) * (ksq * (3.0 * ksq - 2.0) * Em1mKdm + ksq * Eofk);
                J[1] = 2.0 / (15.0 * ksq * k) * (ksq * (4.0 - 3.0 * ksq) * Eofk + ksq * (9.0 * ksq - 8) * Em1mKdm);
                if (gradient) {
                    dJdk[0] = 2.0 / ksq * (-Eofk + 2 * Em1mKdm);
                    dJdk[1] = -3.0 * k * invksq * J[1] + ksq * dJdk[0];
                }
            } else {
                J[0] = (2.0 / 3.0) * ((3.0 - 2.0 * invksq) * Eofk + invksq * Em1mKdm);
                J[1] = 0.4 * (1.0 / 3.0) * ((-3.0 + 4.0 * invksq) * Em1mKdm + (9.0 - 8.0 * invksq) * Eofk);
                if (gradient) {
                    dJdk[0] = 2.0 / (ksq * k) * (2.0 * Eofk - Em1mKdm);
                    dJdk[1] = -3.0 * k * invksq * J[1] + ksq * dJdk[0];
                }
            }

            // Higher order values
            for (v = 2; v <= jvmax; ++v) {
                f1 = 2 * (v + (v - 1) * ksq + 1);
                f2 = ksq * (2 * v - 3);
                J[v] = (f1 * J[v - 1] - f2 * J[v - 2]) / (2 * v + 3);
                if (gradient)
                    dJdk[v] = -3.0 * k * invksq * J[v] + ksq * dJdk[v - 1];
            }

        }

    }

    /**
    Compute the `s^T` occultation solution vector

    */
    template <class T>
    inline void GreensLimbDark<T>::compute(const T& b, const T& r, bool gradient) {

        // Initialize the basic variables
        b2 = b * b;
        r2 = r * r;
        invr = 1.0 / r;
        invb = 1.0 / b;
        bmr = b - r;
        bpr = b + r;
        fourbr = 4 * b * r;
        invfourbr = 0.25 * invr * invb;
        onembmr2 = (1 + bmr) * (1 - bmr);
        onembmr2inv = 1.0 / onembmr2; 
        sqonembmr2 = sqrt(onembmr2);

        // Compute the kite area and the k^2 variables
        if (unlikely((b == 0) || (r == 0))) {
            ksq = T(INFINITY);
            k = T(INFINITY);
            kc = 1;
            kcsq = 1;
            kkc = T(INFINITY);
            invksq = 0;
            kap0 = 0; // Not used!
            kap1 = 0; // Not used!
            kite_area2 = 0; // Not used!
            S(0) = pi<T>() * (1 - r * r);
            if (gradient) {
                dSdb(0) = 0;
                dSdr(0) = -2 * pi<T>() * r;
            }
        } else {
            ksq = (1 - bmr) * (1 + bmr) * invfourbr;
            invksq = fourbr / ((1 - bmr) * (1 + bmr));
            k = sqrt(ksq);
            if (ksq > 1) {
                kcsq = 1 - invksq;
                kc = sqrt(kcsq);
                kkc = k * kc;
                kap0 = 0; // Not used!
                kap1 = 0; // Not used!
                kite_area2 = 0; // Not used!
                S(0) = pi<T>() * (1 - r2);
                if (gradient) {
                    dSdb(0) = 0;
                    dSdr(0) = -2 * pi<T>() * r;
                }
            } else {
                kcsq = abs((bpr * bpr - 1) * invfourbr);
                kc = sqrt(kcsq);
                // Eric Agol's "kite" method to compute a stable
                // version of k * kc and I_0 = kap0
                // Used to be
                //   G.kkc = G.k * G.kc;
                //   G.kap0 = 2 * acos(G.kc);
                T p0 = T(1.0), p1 = b, p2 = r;
                if (p0 < p1) swap(p0, p1);
                if (p1 < p2) swap(p1, p2);
                if (p0 < p1) swap(p0, p1);
                kite_area2 = sqrt((p0 + (p1 + p2)) * (p2 - (p0 - p1)) *
                                  (p2 + (p0 - p1)) * (p0 + (p1 - p2)));
                kkc = kite_area2 * invfourbr;
                kap0 = atan2(kite_area2, (r - 1) * (r + 1) + b2);
                kap1 = atan2(kite_area2, (1 - r) * (1 + r) + b2);
                T Alens = kap1 + r2 * kap0 - kite_area2 * 0.5;
                S(0) = pi<T>() - Alens;
                if (gradient) {
                    dSdb(0) = kite_area2 * invb;
                    dSdr(0) = -2.0 * r * kap0;
                }
            }
        }

        // Special case
        if (lmax == 0) return;

        // Compute the linear limb darkening term
        // and the elliptic integrals
        S(1) = s2(b, r, third, b2, r2, ksq, kcsq, kc, invksq, onembmr2, 
                  onembmr2inv, sqonembmr2, Eofk, Em1mKdm, 
                  dSdb(1), dSdr(1), gradient);

        // Special case
        if (lmax == 1) return;

        // Special case
        if (unlikely(b == 0)) {
            T term = 1 - r2;
            T dtermdr = -2 * r;
            T fac = sqrt(term);
            T dfacdr = -r / fac;
            for (int n = 2; n < lmax + 1; ++n) {
                S(n) = -term * r2 * 2 * pi<T>();
                if (gradient) {
                    dSdb(n) = 0;
                    dSdr(n) = -2 * pi<T>() * r * (2 * term + r * dtermdr);
                    dtermdr = dfacdr * term + fac * dtermdr;
                }
                term *= fac;
            }
            return;
        }

        // Special case
        if (lmax == 2) {
            T r2pb2 = (r2 + b2);
            T eta2 = 0.5 * r2 * (r2pb2 + b2);
            T four_pi_eta;
            T detadb, detadr;
            if (ksq > 1) {
                four_pi_eta = 4 * pi<T>() * (eta2 - 0.5);
                if (gradient) {
                    T deta2dr =  2 * r * r2pb2;
                    T deta2db = 2 * b * r2;
                    detadr = 4 * pi<T>() * deta2dr;
                    detadb = 4 * pi<T>() * deta2db;
                }
            } else {
                four_pi_eta = 2 * (-(pi<T>() - kap1) + 2 * eta2 * kap0 - 0.25 * kite_area2 * (1.0 + 5 * r2 + b2));
                if (gradient) {
                    detadr = 8 * r * (r2pb2 * kap0 - kite_area2);
                    detadb = 2.0 * invb * (4 * b2 * r2 * kap0 - (1 + r2pb2) * kite_area2);
                }
            }
            S(2) = 2 * S(0) + four_pi_eta;
            if (gradient) {
                dSdb(2) = 2 * dSdb(0) + detadb;
                dSdr(2) = 2 * dSdr(0) + detadr;
            }
            return;
        }

        // Derivatives
        T dkdb, dkdr;
        if (gradient) {
            dkdb = 0.5 * (r2 - b2 - 1) * invb * invfourbr * invksq * k;
            dkdr = 0.5 * (b2 - r2 - 1) * invr * invfourbr * invksq * k;
        }

        // Compute the I and J integrals
        computeI(gradient);
        computeJ(gradient);
        
        // Compute the higher order S terms
        int n0, nmi;
        T norm, fac1, k2n, term, coeff;
        T Iv1, Iv2, Jv1, Jv2;
        T pofgn, dpdr, dpdb, dpdk;
        T rmb_on_onembmr2 = -bmr * onembmr2inv;
        T mfbri = -fourbr; 
        T mfbrj = 1; 
        for (int n = 2; n < lmax + 1; ++n) {
            pofgn = 0;
            dpdr = 0;
            dpdb = 0;
            dpdk = 0;
            if (is_even(n)) {
                // For even values of n, sum over I_v:
                n0 = n / 2;
                coeff = mfbri;
                mfbri *= -fourbr;
                // Compute i=0 term
                Iv1 = I[n0]; 
                Iv2 = I[n0 + 1];
                pofgn = coeff * (-bmr * Iv1 + 2 * b * Iv2);
                if (gradient) {
                    dpdr = coeff * Iv1;
                    dpdb = coeff * (-Iv1 + 2 * Iv2);
                    dpdr += (n0 + 1) * pofgn * invr;
                    dpdb += n0 * pofgn * invb;
                }
                k2n = coeff;
                // For even n, compute coefficients for the sum over I_v:
                for (int i = 1; i < n0 + 1; ++i) {
                    nmi = n0 - i;
                    Iv2 = Iv1; 
                    Iv1 = I[nmi];
                    k2n *= -ksq;
                    coeff = tables::choose<T>(n0, i) * k2n;
                    term = coeff * (-bmr * Iv1 + 2 * b * Iv2);
                    pofgn += term;
                    if (gradient) {
                        dpdr += coeff * Iv1;
                        dpdb += coeff * (-Iv1 + 2.0 * Iv2);
                        fac1 = (2 * i) * rmb_on_onembmr2;
                        dpdr += term * (-fac1 + (nmi + 1) * invr);
                        dpdb += term * (fac1 + nmi * invb);
                    }
                }
                pofgn *= 2 * r;
                if (gradient) {
                    dpdr *= 2 * r;
                    dpdb *= 2 * r;
                }
            } else {
                // Now do the same for odd n in sum over J_v:
                n0 = (n - 3) / 2;
                coeff = mfbrj;
                mfbrj *= -fourbr;
                // Compute i=0 term
                Jv1 = J[n0]; 
                Jv2 = J[n0 + 1];
                pofgn = coeff * (-bmr * Jv1 + 2 * b * Jv2);
                if (gradient) {
                    dpdr = coeff * Jv1;
                    dpdb = coeff * (-Jv1 + 2 * Jv2);
                    dpdr += pofgn * (-3 * rmb_on_onembmr2 + (n0 + 1) * invr);
                    dpdb += pofgn * (3 * rmb_on_onembmr2 + n0 * invb);
                    dpdk = coeff * (-bmr * dJdk[n0] + 2 * b * dJdk[n0 + 1]);
                }
                // For odd n, compute coefficients for the sum over J_v:
                k2n = coeff;
                for (int i = 1; i < n0 + 1; ++i) {
                    nmi = n0 - i;
                    k2n *= -ksq;
                    coeff = tables::choose<T>(n0, i) * k2n;
                    Jv2 = Jv1; 
                    Jv1 = J[nmi];
                    term = coeff * (-bmr * Jv1 + 2 * b * Jv2);
                    pofgn += term;
                    if (gradient) {
                        dpdr += coeff * Jv1;
                        dpdb += coeff * (-Jv1 + 2.0 * Jv2);
                        fac1 = (2 * i + 3) * rmb_on_onembmr2;
                        dpdr += term * (-fac1 + (nmi + 1) * invr);
                        dpdb += term * (fac1 + nmi * invb);
                        dpdk += coeff * (-bmr * dJdk[nmi] + 2 * b * dJdk[nmi + 1]);
                    }
                }
                norm = 2 * r * onembmr2 * sqonembmr2;
                pofgn *= norm; 
                if (gradient) {
                    dpdr *= norm;
                    dpdb *= norm;
                    dpdk *= norm;
                }
            }

            // Q(G_n) is zero in this case since on limb of star 
            // z^n = 0 at the stellar boundary for n > 0.
            S(n) = -pofgn;
            if (gradient) {
                dSdb(n) = -(dpdb + dpdk * dkdb);
                dSdr(n) = -(dpdr + dpdk * dkdr);
            }
        }

    }

} // namespace limbdark
} // namespace starry

#endif
