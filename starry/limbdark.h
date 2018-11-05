/**
Limb darkening utilities from Agol & Luger (2018).

TODO: Loop downward in v until J[v] !=0
TODO: Use reparameterized elliptic integrals everywhere!
TODO: Replace all pows
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
    inline T s2(const T& b, const T& r, T& Eofk, T& Em1mKdm, T& ds2db, T& ds2dr, bool gradient=false) {
        T third = 1.0 / 3.0;
        T Lambda1 = 0;
        T b2 = b * b;
        T r2 = r * r; 
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
                T onembmr2 = (r + 1 - b) * (1 - r + b); 
                T fourbr = 4 * b * r; 
                T fourbrinv = 1.0 / fourbr;
                T k2 = onembpr2 * fourbrinv + 1;
                if (k2 < 1) {
                    // Case 2, Case 8
                    T kc2 = -onembpr2 * fourbrinv; 
                    T kc = sqrt(kc2); 
                    T sqbr = sqrt(b * r); 
                    T sqbrinv = 1.0 / sqbr;
                    T Piofk;
                    ellip::CEL(k2, kc, T((b - r) * (b - r) * kc2), T(0.0), T(1.0), T(1.0), T(3 * kc2 * (b - r) * (b + r)), kc2, T(0.0), Piofk, Eofk, Em1mKdm);
                    Lambda1 = onembmr2 * (Piofk + (-3 + 6 * r2 + 2 * b * r) * Em1mKdm - fourbr * Eofk) * sqbrinv * third;
                    if (gradient) {
                        ds2db = 2 * r * onembmr2 * (-Em1mKdm + 2 * Eofk) * sqbrinv * third;
                        ds2dr = -2 * r * onembmr2 * Em1mKdm * sqbrinv;
                    }
                } else if (k2 > 1) {
                    // Case 3, Case 9
                    T onembmr2inv = 1.0 / onembmr2; 
                    T k2inv = 1.0 / k2; 
                    T kc2 = onembpr2 * onembmr2inv; 
                    T kc = sqrt(kc2);
                    T bmrdbpr = (b - r) / (b + r); 
                    T mu = 3 * bmrdbpr * onembmr2inv;
                    T p = bmrdbpr * bmrdbpr * onembpr2 * onembmr2inv;
                    T Piofk;
                    ellip::CEL(k2inv, kc, p, T(1 + mu), T(1.0), T(1.0), T(p + mu), kc2, T(0.0), Piofk, Eofk, Em1mKdm);
                    T sqonembmr2 = sqrt(onembmr2);
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
            std::vector<T> dIdk;
            std::vector<T> J;
            std::vector<T> dJdk;
            int ivmax;
            int jvmax;

            // Coefficients
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

                   // Pre-tabulate J coeffs
                   computeJcoeffs();

                   // Pre-tabulate I for ksq >= 1
                   ivgamma.resize(ivmax + 1);
                   for (int v = 0; v <= ivmax; v++)
                       ivgamma[v] = root_pi<T>() *
                                    T(boost::math::tgamma_delta_ratio(
                                      Multi(v + 0.5), Multi(0.5)));

            }

            inline void compute(const T& b_, const T& r_, bool gradient=false);
            inline void computeI(bool gradient=false);
            inline void computeJ(bool gradient=false);
            inline void computeJcoeffs();
            inline void computeEK();

    };

    /**

    */
    template <class T>
    inline void GreensLimbDark<T>::computeI(bool gradient) {

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

            // k2 < 1
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

            // k2 >= 1
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
                    dJdk[v] = (dJdk[v + 1] + (3.0 / k) * J[v + 1]) * invksq;
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
                if (gradient) {
                    dJdk[0] = (2 * (2 - ksq) * E + 2 * (ksq - 1) * K) / (ksq * k);
                    dJdk[1] = -3.0 / k * J[1] + ksq * dJdk[0];
                }
            } else {
                J[0] = 2.0 / (3.0 * ksq * k) * (2 * (2 * ksq - 1) * E +
                                               (1 - ksq) * (2 - 3 * ksq) * K);
                J[1] = 2.0 / (15.0 * ksq * k) * ((-3 * ksq * ksq + 13 * ksq - 8) * E +
                                                (1 - ksq) * (8 - 9 * ksq) * K);
                if (gradient) {
                    dJdk[0] = 2.0 * invksq * invksq * ((2 - ksq) * E + 2 * (ksq - 1) * K);
                    dJdk[1] = -3.0 / k * J[1] + ksq * dJdk[0];
                }
            }

            // Higher order values
            for (v = 2; v <= jvmax; ++v) {
                f1 = 2 * (v + (v - 1) * ksq + 1);
                f2 = ksq * (2 * v - 3);
                J[v] = (f1 * J[v - 1] - f2 * J[v - 2]) / (2 * v + 3);
                if (gradient)
                    dJdk[v] = -3.0 / k * J[v] + ksq * dJdk[v - 1];
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
    inline void GreensLimbDark<T>::compute(const T& b, const T& r, bool gradient) {

        // Initialize the basic variables
        T dkdb, dkdr;
        T rinv = 1.0 / r;
        T binv = 1.0 / b;
        rmb = r - b;
        twob = 2 * b;
        fourbr = 4 * b * r;
        invfourbr = 0.25 * rinv * binv;
        
        if (unlikely((b == 0) || (r == 0))) {
            ksq = T(INFINITY);
            k = T(INFINITY);
            kc = 1;
            kkc = T(INFINITY);
            invksq = 0;
            kap0 = 0;
            S(0) = pi<T>() * (1 - r * r);
            if (gradient) {
                dSdb(0) = 0;
                dSdr(0) = -2 * pi<T>() * r;
            }
        } else {
            ksq = (1 - (b - r)) * (1 + (b - r)) * invfourbr;
            invksq = fourbr / ((1 - (b - r)) * (1 + (b - r)));
            k = sqrt(ksq);
            if (ksq > 1) {
                kc = sqrt(1 - invksq);
                kkc = k * kc;
                kap0 = 0; // Not used!
                S(0) = pi<T>() * (1 - r * r);
                if (gradient) {
                    dSdb(0) = 0;
                    dSdr(0) = -2 * pi<T>() * r;
                }
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
                if (gradient) {
                    dSdb(0) = kite_area2 * binv;
                    dSdr(0) = -2.0 * r * kap0;
                }
            }
        }

        // Special case
        if (lmax == 0) return;

        // Compute the elliptic integrals
        computeEK();

        // Compute the linear limb darkening term
        T Eofk, Em1mKdm;
        S(1) = s2(b, r, Eofk, Em1mKdm, dSdb(1), dSdr(1), gradient);

        // Special case
        if (lmax == 1) return;

        // Special case
        if (unlikely(b == 0)) {
            T term = 1 - r * r;
            T dtermdr = -2 * r;
            T fac = sqrt(term);
            T dfacdr = -r / fac;
            for (int n = 2; n < lmax + 1; ++n) {
                S(n) = -term * r * r * 2 * pi<T>();
                if (gradient) {
                    dSdb(n) = 0;
                    dSdr(n) = -2 * pi<T>() * r * (2 * term + r * dtermdr);
                    dtermdr = dfacdr * term + fac * dtermdr;
                }
                term *= fac;
            }
            return;
        }

        // Derivatives
        if (gradient) {
            dkdb = (r * r - b * b - 1) / (8 * k * b * b * r);
            dkdr = (b * b - r * r - 1) / (8 * k * b * r * r);
        }

        // Compute I and J integrals for the higher order terms
        computeI(gradient);
        computeJ(gradient);
        
        // 
        int n0, nmi;
        T norm, fac1, k2n, term, coeff;
        T Iv1, Iv2, Jv1, Jv2;
        T pofgn, dpdr, dpdb, dpdk;
        T onembmr2 = (r + 1 - b) * (1 - r + b);
        T onembmr2inv = 1.0 / onembmr2; 
        T rmb_on_onembmr2 = (r - b) * onembmr2inv;
        T sqonembmr2 = sqrt(onembmr2);
        
        // Compute the higher order S terms
        for (int n = 2; n < lmax + 1; ++n) {
            pofgn = 0;
            dpdr = 0;
            dpdb = 0;
            dpdk = 0;
            if (is_even(n)) {
                // For even values of n, sum over I_v:
                n0 = n / 2;
                coeff = pow(-fourbr, n0); // TODO: Speed up
                // Compute i=0 term
                Iv1 = I[n0]; 
                Iv2 = I[n0 + 1];
                pofgn = coeff * ((r - b) * Iv1 + 2 * b * Iv2);
                if (gradient) {
                    dpdr = coeff * Iv1;
                    dpdb = coeff * (-Iv1 + 2 * Iv2);
                    dpdr += (n0 + 1) * pofgn * rinv;
                    dpdb += n0 * pofgn * binv;
                }
                k2n = coeff;
                // For even n, compute coefficients for the sum over I_v:
                for (int i = 1; i < n0 + 1; ++i) {
                    nmi = n0 - i;
                    Iv2 = Iv1; 
                    Iv1 = I[nmi];
                    k2n *= -ksq;
                    coeff = tables::choose<T>(n0, i) * k2n;
                    term = coeff * ((r - b) * Iv1 + 2 * b * Iv2);
                    pofgn += term;
                    if (gradient) {
                        dpdr += coeff * Iv1;
                        dpdb += coeff * (-Iv1 + 2.0 * Iv2);
                        fac1 = i * 2.0 * rmb_on_onembmr2;
                        dpdr += term * (-fac1 + (nmi + 1) * rinv);
                        dpdb += term * (fac1 + nmi * binv);
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
                coeff = pow(-fourbr, n0); // TODO: Speed up
                // Compute i=0 term
                Jv1 = J[n0]; 
                Jv2 = J[n0 + 1];
                pofgn = coeff * ((r - b) * Jv1 + 2 * b * Jv2);
                if (gradient) {
                    dpdr = coeff * Jv1;
                    dpdb = coeff * (-Jv1 + 2 * Jv2);
                    dpdr += pofgn * (-3 * rmb_on_onembmr2 + (n0 + 1) * rinv);
                    dpdb += pofgn * (3 * rmb_on_onembmr2 + n0 * binv);
                    dpdk = coeff * ((r - b) * dJdk[n0] + 2 * b * dJdk[n0 + 1]);
                }
                // For odd n, compute coefficients for the sum over J_v:
                k2n = coeff;
                for (int i = 1; i < n0 + 1; ++i) {
                    nmi = n0 - i;
                    k2n *= -ksq;
                    coeff = tables::choose<T>(n0, i) * k2n;
                    Jv2 = Jv1; 
                    Jv1 = J[nmi];
                    term = coeff * ((r - b) * Jv1 + 2 * b * Jv2);
                    pofgn += term;
                    if (gradient) {
                        dpdr += coeff * Jv1;
                        dpdb += coeff * (-Jv1 + 2.0 * Jv2);
                        fac1 = (i * 2.0 + 3.0) * rmb_on_onembmr2;
                        dpdr += term * (-fac1 + (nmi + 1) * rinv);
                        dpdb += term * (fac1 + nmi * binv);
                        dpdk += coeff * ((r - b) * dJdk[nmi] + 2 * b * dJdk[nmi + 1]);
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
