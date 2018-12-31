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
#include "utils.h"
#include "ellip.h"
#include "errors.h"


namespace starry2 {
namespace limbdark {

    using std::abs;
    using std::max;
    using std::swap;
    using namespace utils;

    /**
    Return (something like the) Wallis ratio, 

        Gamma(1 + n / 2) / Gamma(3 / 2 + n / 2)

    using Boost if enabled. Otherwise computes it recursively. Using 
    double precision, the error is below 1e-14 well past n = 100.

    */
    template <typename T>
    inline T wallis(int n) {

#ifdef STARRY_ENABLE_BOOST

        return boost::math::tgamma_delta_ratio<T>(1 + 0.5 * n, 0.5);

#else

        int z, dz;
        if (is_even(n)) {
            z = 1 + n / 2;
            dz = -1;
        } else {
            z = 1 + (n - 1) / 2;
            dz = 0;
        }
        T A = 1.0;
        T B = root_pi<T>();
        for (int i = 1; i < z + dz; ++i) {
            A *= i + 1;
            B *= i - 0.5;
        }
        for (int i = max(1, z + dz); i < z + 1; ++i)
            B *= i - 0.5;
        if (is_even(n))
            return A / B;
        else
            return B / A;

#endif

    }

    /**
    Transform the u_n coefficients to `c_n`, which are coefficients
    of the basis in which the `P(G_n)` functions are computed.
    Also compute the derivative matrix `dc / du`.
    
    This is the single-wavelength case.

    */
    template <typename Derived1, typename Derived2, typename Derived3>
    inline void computeC (
        const MatrixBase<Derived1>& u, 
        MatrixBase<Derived2>& c,
        MatrixBase<Derived3>& dcdu,
        const typename Derived1::Scalar& y00
    ) {
        using T = typename Derived1::Scalar;
        T bcoeff;
        size_t N = u.rows();
        Vector<T> a(N);
        Matrix<T> dadu;
        a.setZero();
        dadu.setZero(N, N);
        dcdu.setZero();

        // Compute the a_n coefficients
        a(0) = 1.0;
        for (size_t i = 1; i < N; ++i) {
            bcoeff = 1.0;
            int sgn = 1;
            for (size_t j = 0; j <= i; ++j) {
                a(j) -= u(i) * bcoeff * sgn;
                dadu(j, i) -= bcoeff * sgn;
                sgn *= -1;
                bcoeff *= ((T)(i - j) / (j + 1));
            }
        }

        // Now, compute the c_n coefficients
        for (size_t j = N - 1; j >= 2; --j) {
            if (j >= N - 2) {
                c(j) = a(j) / (j + 2);
                dcdu.transpose().block(j, 0, 1, N - 1) = 
                    dadu.block(j, 1, 1, N - 1) / (j + 2);
            } else {
                c(j) = a(j) / (j + 2) + c(j + 2);
                dcdu.transpose().block(j, 0, 1, N - 1) = 
                    dadu.block(j, 1, 1, N - 1) / (j + 2) +
                    dcdu.transpose().block(j + 2, 0, 1, N - 1);
            }
        }

        if (N >= 4) {
            c(1) = a(1) + 3 * c(3);
            dcdu.transpose().block(1, 0, 1, N - 1) = 
                dadu.block(1, 1, 1, N - 1) +
                3 * dcdu.transpose().block(3, 0, 1, N - 1);
        } else {
            c(1) = a(1);
            dcdu.transpose().block(1, 0, 1, N - 1) = 
                dadu.block(1, 1, 1, N - 1);
        }

        if (N >= 3) {
            c(0) = a(0) + 2 * c(2);
            dcdu.transpose().block(0, 0, 1, N - 1) = 
                dadu.block(0, 1, 1, N - 1) +
                2 * dcdu.transpose().block(2, 0, 1, N - 1);
        } else {
            c(0) = a(0);
            dcdu.transpose().block(0, 0, 1, N - 1) = 
                dadu.block(0, 1, 1, N - 1);
        }

        // Normalize `c` and `dcdu`
        // The total flux is given by `(S . c)`
        T norm = y00 / (pi<T>() * (c(0) + 2.0 * c(1) / 3.0));
        c *= norm;
        dcdu *= norm;

    }

    /**
    Transform the u_n coefficients to `c_n`, which are coefficients
    of the basis in which the `P(G_n)` functions are computed.
    Also compute the derivative matrix `dc / du`.

    This is the spectral case.

    */
    template <class T, typename Derived>
    inline void computeC (
        const Matrix<T>& u, 
        Matrix<T>& c,
        Matrix<T>& dcdu,
        const MatrixBase<Derived>& y00
    ) {
        int lmax = u.rows() - 1;
        int ncol = u.cols();

        // NOTE: It is *surprisingly* difficult to avoid this
        // copy operation with Eigen, because Matrix blocks
        // cannot be passed as lvalues to functions.
        // See http://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
        Matrix<T> dcdu_n(lmax, lmax + 1);
        for (int n = 0; n < ncol; ++n) {
            Eigen::Map<const Vector<T>> u_n(u.col(n).data(), u.rows());
            Eigen::Map<Vector<T>> c_n(c.col(n).data(), c.rows());
            computeC(u_n, c_n, dcdu_n, y00(n));
            dcdu.block(n * lmax, 0, lmax, lmax + 1) = dcdu_n;
        }
    }

    /**
    Greens integration housekeeping data

    */
    template <class T>
    class GreensLimbDark 
    {

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
        T onembpr2; 
        T b2mr22;
        T onemr2mb2;
        T sqarea;
        T sqbr;
        T kite_area2;
        T third;
        T Eofk;
        T Em1mKdm;

        // Helper intergrals
        RowVector<T> M;
        RowVector<T> N;
        Matrix<T> M_coeff;
        Matrix<T> N_coeff;

        // The solution vector
        RowVector<T> s;
        RowVector<T> dsdb;
        RowVector<T> dsdr;

        // Constructor
        explicit GreensLimbDark(
            int lmax
        ) :
            lmax(lmax),
            M(lmax + 1),
            N(lmax + 1),
            M_coeff(4, STARRY_MN_MAX_ITER),
            N_coeff(2, STARRY_MN_MAX_ITER),
            s(RowVector<T>::Zero(lmax + 1)),
            dsdb(RowVector<T>::Zero(lmax + 1)),
            dsdr(RowVector<T>::Zero(lmax + 1)) 
        {
            // Constants
            computeMCoeff();
            third = T(1.0) / T(3.0);
        }

        inline void computeS1 (
            bool gradient=false
        );

        inline void computeMCoeff ();

        inline void computeM0123 ();

        inline void upwardM ();

        inline void downwardM ();

        inline void computeNCoeff ();

        inline void computeN01 ();

        inline void upwardN ();

        inline void downwardN ();

        inline void compute (
            const T& b_, 
            const T& r_, 
            bool gradient=false
        );
    };

    /**
    The linear limb darkening flux term.

    */
    template <class T>
    inline void GreensLimbDark<T>::computeS1 (
        bool gradient
    ) {
        T Lambda1 = 0;
        if ((b >= 1.0 + r) ||  (r == 0.0)) {
            // No occultation (Case 1)
            Lambda1 = 0;
            dsdb(1) = 0;
            dsdr(1) = 0;
            Eofk = 0; // Check
            Em1mKdm = 0; // Check
        } else if (b <= r - 1.0) {
            // Full occultation (Case 11)
            Lambda1 = 0;
            dsdb(1) = 0;
            dsdr(1) = 0;
            Eofk = 0; // Check
            Em1mKdm = 0; // Check
        } else {
            if (unlikely(b == 0)) {
                // Case 10
                T sqrt1mr2 = sqrt(1.0 - r2);
                Lambda1 = -2.0 * pi<T>() * sqrt1mr2 * sqrt1mr2 * sqrt1mr2; 
                Eofk = 0.5 * pi<T>();
                Em1mKdm = 0.25 * pi<T>();
                if (gradient) {
                    dsdb(1) = 0;
                    dsdr(1) = -2.0 * pi<T>() * r * sqrt1mr2;
                }
            } else if (unlikely(b == r)) {
                if (unlikely(r == 0.5)) {
                    // Case 6
                    Lambda1 = pi<T>() - 4.0 * third;
                    Eofk = 1.0;
                    Em1mKdm = 1.0;
                    if (gradient) {
                        dsdb(1) = 2.0 * third;
                        dsdr(1) = -2.0;
                    }
                } else if (r < 0.5) {
                    // Case 5
                    T m = 4 * r2;
                    Eofk = ellip::CEL(m, T(1.0), T(1.0), T(1.0 - m));
                    Em1mKdm = ellip::CEL(m, T(1.0), T(1.0), T(0.0));
                    Lambda1 = pi<T>() + 2.0 * third * 
                              ((2 * m - 3) * Eofk - m * Em1mKdm);
                    if (gradient) {
                        dsdb(1) = -4.0 * r * third * (Eofk - 2 * Em1mKdm);
                        dsdr(1) = -4.0 * r * Eofk;
                    }
                } else {
                    // Case 7
                    T m = 4 * r2; 
                    T minv = 1.0 / m; 
                    Eofk = ellip::CEL(minv, T(1.0), T(1.0), T(1.0 - minv));
                    Em1mKdm = ellip::CEL(minv, T(1.0), T(1.0), T(0.0));
                    Lambda1 = pi<T>() + third * invr * 
                              (-m * Eofk + (2 * m - 3) * Em1mKdm);
                    if (gradient) {
                        dsdb(1) = 2 * third * (2 * Eofk - Em1mKdm);
                        dsdr(1) = -2 * Em1mKdm;
                    }
                }
            } else { 
                if (ksq < 1) {
                    // Case 2, Case 8
                    T sqbrinv = 1.0 / sqbr;
                    T Piofk;
                    ellip::CEL(ksq, kc, T((b - r) * (b - r) * kcsq), T(0.0), 
                               T(1.0), T(1.0), T(3 * kcsq * (b - r) * (b + r)), 
                               kcsq, T(0.0), Piofk, Eofk, Em1mKdm);
                    Lambda1 = onembmr2 * (Piofk + (-3 + 6 * r2 + 2 * b * r) * 
                              Em1mKdm - fourbr * Eofk) * sqbrinv * third;
                    if (gradient) {
                        dsdb(1) = 2 * r * onembmr2 * (-Em1mKdm + 2 * Eofk) * 
                                sqbrinv * third;
                        dsdr(1) = -2 * r * onembmr2 * Em1mKdm * sqbrinv;
                    }
                } else if (ksq > 1) {
                    // Case 3, Case 9
                    T bmrdbpr = (b - r) / (b + r); 
                    T mu = 3 * bmrdbpr * onembmr2inv;
                    T p = bmrdbpr * bmrdbpr * onembpr2 * onembmr2inv;
                    T Piofk;
                    ellip::CEL(invksq, kc, p, T(1 + mu), T(1.0), T(1.0), 
                               T(p + mu), kcsq, T(0.0), Piofk, Eofk, Em1mKdm);
                    Lambda1 = 2 * sqonembmr2 * 
                              (onembpr2 * Piofk - (4 - 7 * r2 - b2) * Eofk) * 
                              third;
                    if (gradient) {
                        dsdb(1) = -4 * r * third * sqonembmr2 * 
                                (Eofk - 2 * Em1mKdm);
                        dsdr(1) = -4 * r * sqonembmr2 * Eofk;
                    }
                } else {
                    // Case 4
                    T rootr1mr = sqrt(r * (1 - r));
                    Lambda1 = 2 * acos(1.0 - 2.0 * r) - 4 * third * 
                              (3 + 2 * r - 8 * r2) * 
                              rootr1mr - 2 * pi<T>() * int(r > 0.5);
                    Eofk = 1.0;
                    Em1mKdm = 1.0;
                    if (gradient) {
                        dsdr(1) = -8 * r * rootr1mr;
                        dsdb(1) = -dsdr(1) * third;
                    }
                }
            }
        }
        s(1) = ((1.0 - int(r > b)) * 2 * pi<T>() - Lambda1) * third;
    }

    /** 
    Compute the coefficients of the series expansion
    for the highest four terms of the `M` integral.

    */
    template <class T>
    inline void GreensLimbDark<T>::computeMCoeff () {
    
        T coeff;
        int n;

        // ksq < 1
        for (int j = 0; j < 4; ++j) {
            n = lmax - 3 + j;
            coeff = root_pi<T>() * wallis<T>(n);

            // Add leading term to M
            M_coeff(j, 0) = coeff;
            // Now, compute higher order terms until 
            // desired precision is reached
            for (int i = 1; i < STARRY_MN_MAX_ITER; ++i) {
                coeff *= T((2 * i - 1) * (2 * i - 1)) / 
                         T(2 * i * (1 + n + 2 * i));
                M_coeff(j, i) = coeff;
            }
        }
        
    }

    /** 
    Compute the first four terms of the M integral.

    */
    template <class T>
    inline void GreensLimbDark<T>::computeM0123 () {
        if (ksq < 1.0) {
            M(0) = kap0;
            M(1) = 2 * sqbr * 2 * ksq * Em1mKdm;
            M(2) = kap0 * onemr2mb2 + kite_area2;
            M(3) = (2.0 * sqbr) * (2.0 * sqbr) * (2.0 * sqbr) * 2.0 * 
                   third * ksq * (Eofk + (3.0 * ksq - 2.0) * Em1mKdm);
        } else {
            M(0) = pi<T>();
            M(1) = 2.0 * sqonembmr2 * Eofk;
            M(2) = pi<T>() * onemr2mb2;
            M(3) = sqonembmr2 * sqonembmr2 * sqonembmr2 * 2.0 * third * 
                   ((3.0 - 2.0 * invksq) * Eofk + invksq * Em1mKdm);
        }
    }

    /** 
    Compute the terms in the M integral by upward recursion.

    */
    template <class T>
    inline void GreensLimbDark<T>::upwardM () {
        // Compute lowest four exactly
        computeM0123();

        // Recurse upward
        for (int n = 4; n < lmax + 1; ++n)
            M(n) = (2.0 * (n - 1) * onemr2mb2 * M(n - 2) + 
                    (n - 2) * sqarea * M(n - 4)) / n;
    }

    /** 
    Compute the terms in the M integral by downward recursion.

    */
    template <class T>
    inline void GreensLimbDark<T>::downwardM () {
        T val, k2n, tol, fac, term;
        T invsqarea = T(1.0) / sqarea;

        // Compute highest four using a series solution
        if (ksq < 1) {

            // Compute leading coefficient (n=0)
            tol = mach_eps<T>() * ksq;
            term = 0.0;
            fac = 1.0;
            for (int n = 0; n < lmax - 3; ++n)
                fac *= sqonembmr2;
            fac *= k;

            // Now, compute higher order terms until 
            // desired precision is reached
            for (int j = 0; j < 4; ++j) {
                // Add leading term to M
                val = M_coeff(j, 0);
                k2n = 1.0;

                // Compute higher order terms
                for (int n = 1; n < STARRY_MN_MAX_ITER; ++n) {
                    k2n *= ksq;
                    term = k2n * M_coeff(j, n);
                    val += term;
                    if (abs(term) < tol)
                        break;
                }
                M(lmax - 3 + j) = val * fac;
                fac *= sqonembmr2;
            }

        } else {

            throw errors::NotImplementedError(
                "Downward recursion in `M` not implemented for `k^2` >= 1.");

        }

        // Recurse downward
        for (int n = lmax - 4; n > 3; --n)
            M(n) = ((n + 4) * M(n + 4) - 2.0 * (n + 3) * onemr2mb2 * M(n + 2)) 
                    * invsqarea / (n + 2);

        // Compute lowest four exactly
        computeM0123();
    }

    /** 
    Compute the coefficients of the series expansion
    for the highest two terms of the `N` integral.

    */
    template <class T>
    inline void GreensLimbDark<T>::computeNCoeff () {
        T coeff = 0.0;
        int n;

        // ksq < 1
        for (int j = 0; j < 2; ++j) {
            n = lmax - 1 + j;

            // Add leading term to N
            coeff = root_pi<T>() * wallis<T>(n) / (n + 3.0);
            N_coeff(j, 0) = coeff;

            // Now, compute higher order terms until
            // desired precision is reached
            for (int i = 1; i < STARRY_MN_MAX_ITER; ++i) {
                coeff *= T(4 * i * i - 1) / T(2 * i * (3 + n + 2 * i));
                N_coeff(j, i) = coeff;
            }
        }
    }

    /** 
    Compute the first two terms of the N integral.

    */
    template <class T>
    inline void GreensLimbDark<T>::computeN01 () {
        if (ksq <= 1.0) {
            N(0) = 0.5 * kap0 - k * kc; 
            N(1) = 4.0 * third * sqbr * ksq * (-Eofk + 2.0 * Em1mKdm);
        } else {
            N(0) = 0.5 * pi<T>();
            N(1) = 4.0 * third * sqbr * k * (2.0 * Eofk - Em1mKdm);
        }
    }

    /** 
    Compute the terms in the N integral by upward recursion.

    */
    template <class T>
    inline void GreensLimbDark<T>::upwardN () {
        // Compute lowest two exactly
        computeN01();

        // Recurse upward
        for (int n = 2; n < lmax + 1; ++n)
            N(n) = (M(n) + n * onembpr2 * N(n - 2)) / (n + 2);
    }

    /** 
    Compute the terms in the N integral by downward recursion.

    */
    template <class T>
    inline void GreensLimbDark<T>::downwardN () {
        // Compute highest two using a series solution
        if (ksq < 1) {
            // Compute leading coefficient (n=0)
            T val, k2n;
            T tol = mach_eps<T>() * ksq; 
            T term = 0.0;
            T fac = 1.0;
            for (int n = 0; n < lmax - 1; ++n)
                fac *= sqonembmr2;
            fac *= k * ksq;

            // Now, compute higher order terms until 
            // desired precision is reached
            for (int j = 0; j < 2; ++j) {
                val = N_coeff(j, 0);
                k2n = 1.0;
                for (int n = 1; n < STARRY_MN_MAX_ITER; ++n) {
                    k2n *= ksq;
                    term = k2n * N_coeff(j, n);
                    val += term;
                    if (abs(term) < tol)
                        break;
                }
                N(lmax - 1 + j) = val * fac;
                fac *= sqonembmr2;
            }

        } else {
            throw errors::NotImplementedError(
                "Downward recursion in `N` not implemented for `k^2` >= 1.");
        }

        // Recurse downward
        T onembpr2inv = T(1.0) / onembpr2;
        for (int n = lmax - 2; n > 1; --n)
            N(n) = ((n + 4) * N(n + 2) - M(n + 2)) * onembpr2inv / (n + 2);

        // Compute lowest two exactly
        computeN01();
    }

    /**
    Compute the `s^T` occultation solution vector

    */
    template <class T>
    inline void GreensLimbDark<T>::compute (
        const T& b_, 
        const T& r_, 
        bool gradient
    ) {
        // Initialize the basic variables
        b = b_;
        r = r_;
        b2 = b * b;
        r2 = r * r;
        invr = 1.0 / r;
        invb = 1.0 / b;
        bmr = b - r;
        bpr = b + r;
        fourbr = 4 * b * r;
        invfourbr = 0.25 * invr * invb;
        onembmr2 = (1.0 + bmr) * (1.0 - bmr);
        onembmr2inv = 1.0 / onembmr2; 
        onembpr2 = (1 - r - b) * (1 + r + b); 
        sqonembmr2 = sqrt(onembmr2);
        b2mr22 = (b2 - r2) * (b2 - r2);
        onemr2mb2 = (1.0 - r) * (1.0 + r) - b2;
        sqbr = sqrt(b * r); 

        // Compute the kite area and the k^2 variables
        T p0 = T(1.0), p1 = b, p2 = r;
        if (p0 < p1) swap(p0, p1);
        if (p1 < p2) swap(p1, p2);
        if (p0 < p1) swap(p0, p1);
        sqarea = (p0 + (p1 + p2)) * (p2 - (p0 - p1)) *
                 (p2 + (p0 - p1)) * (p0 + (p1 - p2));
        kite_area2 = sqrt(max(T(0.0), sqarea));

        if (unlikely((b == 0) || (r == 0))) {
            ksq = T(INFINITY);
            k = T(INFINITY);
            kc = 1;
            kcsq = 1;
            kkc = T(INFINITY);
            invksq = 0;
            kap0 = 0; // Not used!
            kap1 = 0; // Not used!
            s(0) = pi<T>() * (1 - r * r);
            if (gradient) {
                dsdb(0) = 0;
                dsdr(0) = -2 * pi<T>() * r;
            }
        } else {
            ksq = onembpr2 * invfourbr + 1.0;
            invksq = T(1.0) / ksq;
            k = sqrt(ksq);
            if (ksq > 1) {
                kcsq = onembpr2 * onembmr2inv;
                kc = sqrt(kcsq);
                kkc = k * kc;
                kap0 = 0; // Not used!
                kap1 = 0; // Not used!
                s(0) = pi<T>() * (1 - r2);
                if (gradient) {
                    dsdb(0) = 0;
                    dsdr(0) = -2 * pi<T>() * r;
                }
            } else {
                kcsq = -onembpr2 * invfourbr;
                kc = sqrt(kcsq);
                kkc = kite_area2 * invfourbr;
                kap0 = atan2(kite_area2, (r - 1) * (r + 1) + b2);
                kap1 = atan2(kite_area2, (1 - r) * (1 + r) + b2);
                T Alens = kap1 + r2 * kap0 - kite_area2 * 0.5;
                s(0) = pi<T>() - Alens;
                if (gradient) {
                    dsdb(0) = kite_area2 * invb;
                    dsdr(0) = -2.0 * r * kap0;
                }
            }
        }

        // Special case
        if (lmax == 0) return;

        // Compute the linear limb darkening term
        // and the elliptic integrals
        computeS1(gradient);

        // Special case
        if (lmax == 1) return;

        // Special case
        if (unlikely(b == 0)) {
            T term = 1 - r2;
            T dtermdr = -2 * r;
            T fac = sqrt(term);
            T dfacdr = -r / fac;
            for (int n = 2; n < lmax + 1; ++n) {
                s(n) = -term * r2 * 2 * pi<T>();
                if (gradient) {
                    dsdb(n) = 0;
                    dsdr(n) = -2 * pi<T>() * r * (2 * term + r * dtermdr);
                    dtermdr = dfacdr * term + fac * dtermdr;
                }
                term *= fac;
            }
            return;
        }
        
        // TODO: Special cases for r == 0 and for no/complete occultation

        // Compute the quadratic term
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
            four_pi_eta = 2 * (-(pi<T>() - kap1) + 2 * eta2 * 
                            kap0 - 0.25 * kite_area2 * (1.0 + 5 * r2 + b2));
            if (gradient) {
                detadr = 8 * r * (r2pb2 * kap0 - kite_area2);
                detadb = 2.0 * invb * (4 * b2 * r2 * kap0 - 
                            (1 + r2pb2) * kite_area2);
            }
        }
        s(2) = 2 * s(0) + four_pi_eta;
        if (gradient) {
            dsdb(2) = 2 * dsdb(0) + detadb;
            dsdr(2) = 2 * dsdr(0) + detadr;
        }

        if (lmax == 2) return;

        // Now onto the higher order terms...
        if ((ksq < 0.5) && (lmax > 3))
            downwardM();
        else
            upwardM();

        // TODO: This can be vectorized so nicely!
        for (int n = 3; n < lmax + 1; ++n) {
            s(n) = -(2.0 * r2 * M(n) - n / (n + 2.0) * 
                     (onemr2mb2 * M(n) + sqarea * M(n - 2)));
        }

        // Compute ds/dr
        if (gradient) {
            // TODO: This can be vectorized so nicely!
            for (int n = 3; n < lmax + 1; ++n) {
                dsdr(n) = -2 * r * ((n + 2) * M(n) - n * M(n - 2));
            }
        
            if (b > STARRY_BCUT) {
                // TODO: This can be vectorized so nicely!
                for (int n = 3; n < lmax + 1; ++n) {
                    dsdb(n) = -(n * invb * ((M(n) - M(n - 2)) * 
                                (r2 + b2) + b2mr22 * M(n - 2)));
                }
            } else {
                // Small b reparametrization
                T r3 = r2 * r;
                T b3 = b2 * b;
                if ((ksq < 0.5) && (lmax > 3))
                    downwardN();
                else
                    upwardN();
                // TODO: This can be vectorized so nicely!
                for (int n = 3; n < lmax + 1; ++n) {
                    dsdb(n) = -(n * (M(n - 2) * (2 * r3 + b3 - b - 3 * r2 * b) 
                                + b * M(n) - 4 * r3 * N(n - 2)));
                }
            }
        
        }

    }

} // namespace limbdark
} // namespace starry2

#endif
