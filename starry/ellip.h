/**
Elliptic integrals computed following:

            Bulirsch 1965, Numerische Mathematik, 7, 78
            Bulirsch 1965, Numerische Mathematik, 7, 353

and the implementation by E. Agol (private communication).
Adapted from DFM's AstroFlow: https://github.com/dfm/AstroFlow/
*/

#ifndef _STARRY_ELLIP_H_
#define _STARRY_ELLIP_H_

#include <cmath>
#include <unsupported/Eigen/AutoDiff>
#include "utils.h"
#include "errors.h"

namespace starry {

//! Maximum number of iterations in elliptic integral evaluations
#define STARRY_ELLIP_MAX_ITER 200

namespace ellip {

    using std::abs;
    using utils::pi;
    using utils::Multi;
    using utils::mach_eps;

    // EA: Elliptic integral convergence tolerance should be sqrt of machine precision
    static const double tol_double = sqrt(std::numeric_limits<double>::epsilon());
    static const Multi tol_Multi = sqrt(std::numeric_limits<Multi>::epsilon());

    //! Elliptic integral convergence tolerance
    template <typename T>
    inline T tol(){ return T(tol_double); }

    //! Elliptic integral convergence tolerance (multi-precision)
    template <>
    inline Multi tol(){ return tol_Multi; }

    /**
    Complete elliptic integral of the first kind

    */
    template <typename T>
    T K (const T& ksq) {
        T kc = sqrt(1.0 - ksq), m = T(1.0), h;
        for (int i = 0; i < STARRY_ELLIP_MAX_ITER; ++i) {
            h = m;
            m += kc;
            if (abs(h - kc) / h <= tol<T>()) return pi<T>() / m;
            kc = sqrt(h * kc);
            m *= 0.5;
        }
        throw errors::ConvergenceError("Elliptic integral K did not converge.");
    }

    /**
    Complete elliptic integral of the second kind

    */
    template <typename T>
    T E (const T& ksq) {
        T b = 1.0 - ksq, kc = sqrt(b), m = T(1.0), c = T(1.0), a = b + 1.0, m0;
        for (int i = 0; i < STARRY_ELLIP_MAX_ITER; ++i) {
            b = 2.0 * (c * kc + b);
            c = a;
            m0 = m;
            m += kc;
            a += b / m;
            if (abs(m0 - kc) / m0 <= tol<T>()) return 0.25 * pi<T>() * a / m;
            kc = 2.0 * sqrt(kc * m0);
        }
        throw errors::ConvergenceError("Elliptic integral E did not converge.");
    }

    /**
    Complete elliptic integral of the third kind

    */
    template <typename T>
    T Pi (const T& n, const T& ksq) {
        T kc = sqrt(1.0 - ksq),
          p = sqrt(1.0 - n),
          m0 = 1.0,
          c = 1.0,
          d = 1.0 / p,
          e = kc,
          f,
          g;
        for (int i = 0; i < STARRY_ELLIP_MAX_ITER; ++i) {
            f = c;
            c += d / p;
            g = e / p;
            d = 2.0 * (f * g + d);
            p = g + p;
            g = m0;
            m0 = kc + m0;
            if (abs(1.0 - kc / g) <= tol<T>())
                return 0.5 * pi<T>() * (c * m0 + d) / (m0 * (m0 + p));
            kc = 2.0 * sqrt(e);
            e = kc * m0;
        }
        throw errors::ConvergenceError("Elliptic integral Pi did not converge.");
    }

    /**
    Computes the function `cel(kc, p, a, b)` from Bulirsch (1969)

    */
    template <typename T>
    T CEL (const T& ksq0, const T& kc0, const T& p0, const T& a0, const T& b0) {

        // Local copies of const inputs
        T p = p0;
        T a = a0;
        T b = b0;
        T ksq, kc;

        // In some rare cases, k^2 is so close to zero that it can actually
        // go slightly negative. Let's explicitly force it to zero.
        if (ksq0 >= 0) ksq = ksq0;
        else ksq = 0;
        if (kc0 >= 0) kc = kc0;
        else kc = 0;

        // If k^2 is very small, we get better precision
        // evaluating `kc` like this
        if (ksq < 1e-5) kc = sqrt(1 - ksq);

        // We actually need kc to be nonzero, so let's
        // set it to a very small number
        if ((ksq == 1) || (kc == 0)) kc = mach_eps<T>() * ksq;

        // I haven't encountered cases where k^2 > 1 due to
        // roundoff error, but they could happen. If so, change the
        // line below to avoid an exception
        if (ksq > 1) throw errors::ValueError("Elliptic integral `CEL` "
                                              "was called with `ksq` > 1.");
        T ca = sqrt(mach_eps<T>() * ksq);

        if (ca <= 0) ca = std::numeric_limits<T>::min();
        T m = 1.0;
        T q, g, f, ee;
        ee = kc;

        if (p > 0) {
            p = sqrt(p);
            b /= p;
        } else {
            q = ksq;
            g = 1.0 - p;
            f = g - ksq;
            q *= (b - a * p);
            p = sqrt(f / g);
            a = (a - b) / g;
            b = -q / (g * g * p) + a * p;
        }

        f = a;
        a += b / p;
        g = ee / p;
        b += f * g;
        b += b;
        p += g;
        g = m;
        m += kc;

        for (int i = 0; i < STARRY_ELLIP_MAX_ITER; ++i) {
            kc = sqrt(ee);
            kc += kc;
            ee = kc * m;
            f = a;
            a += b / p;
            g = ee / p;
            b += f * g;
            b += b;
            p += g;
            g = m;
            m += kc;
            if (abs(g - kc) < g * ca)
                return 0.5 * pi<T>() * (a * m + b) / (m * (m + p));
        }
        throw errors::ConvergenceError("Elliptic integral CEL did not converge.");
    }

    /**
    Computes the function `cel(kc, p, a, b)` from Bulirsch (1969)

    */
    template <typename T>
    T CEL (const T& ksq0, const T& p0, const T& a0, const T& b0) {
        T kc;
        // Avoid undefined k2=1 case:
        if (ksq0 != 1.0)
            kc = sqrt(1.0 - ksq0);
        else
            kc = mach_eps<T>() * ksq0;
        return CEL(ksq0, kc, p0, a0, b0);
    }


    /**
    Computes the function `cel(kc, p, a, b)` from Bulirsch (1969).
    Vectorized version to improve speed when computing multiple
    elliptic integrals with the same value of `kc`.
    This assumes first value of a and b uses p; the rest have p = 1.

    */
    template <typename T>
    inline void CEL (T k2, T kc, T p, T a1, T a2, T a3, T b1, T b2, T b3, T& Piofk, T& Eofk, T& Em1mKdm) {

        // Bounds checks
        if (k2 > 1) 
            throw errors::ValueError("Invalid value of `k2` passed to `ellip::CEL`.");
        else if ((k2 == 1.0) || (kc == 0.0)) 
            kc = mach_eps<T>() * k2;
        
        // Tolerance
        T ca = sqrt(mach_eps<T>() * k2);

        // Temporary vars
        T p1, pinv, pinv1, q, g, g1, ginv, f, f1, f2, f3;

        // Initialize values:
        T ee = kc; 
        T m = 1.0;
        if (p > 0.0) {
            p = sqrt(p); 
            pinv = 1.0 / p; 
            b1 *= pinv;
        } else {
            q = k2; 
            g = 1.0 - p; 
            f = g - k2;
            q *= (b1 - a1 * p); 
            ginv = 1.0 / g; 
            p = sqrt(f * ginv); 
            a1 = (a1 - b1) * ginv;
            pinv = 1.0 / p;
            b1 = -q * ginv * ginv * pinv + a1 * p;
        }
        // Compute recursion:
        f1 = a1;
        // First compute the first integral with p:
        a1 += b1 * pinv; 
        g = ee * pinv; 
        b1 += f1 * g; 
        b1 += b1; 
        p += g; 
        g = m;
        // Next, compute the remainder with p = 1:
        p1 = 1.0; 
        g1 = ee;
        f2 = a2; 
        f3 = a3;
        a2 += b2; 
        b2 += f2 * g1; 
        b2 += b2;
        a3 += b3; 
        b3 += f3 * g1; 
        b3 += b3;
        p1 += g1;
        g1 = m;
        m += kc;
        size_t iter = 0; 
        while (((abs(g - kc) > g * ca) || (abs(g1 - kc) > g1 * ca)) && (iter < STARRY_ELLIP_MAX_ITER)) {
            kc = sqrt(ee);
            kc += kc;
            ee = kc * m;
            f1 = a1; 
            f2 = a2; 
            f3 = a3;
            pinv = 1.0 / p;
            pinv1 = 1.0 / p1;
            a1 += b1 * pinv;
            a2 += b2 * pinv1;
            a3 += b3 * pinv1;
            g = ee * pinv;
            g1 = ee * pinv1;
            b1 += f1 * g;
            b2 += f2 * g1;
            b3 += f3 * g1;
            b1 += b1;
            b2 += b2;
            b3 += b3;
            p += g;
            p1 += g1;
            g = m;
            m += kc;
            ++iter;
        }
        if (iter == STARRY_ELLIP_MAX_ITER)
            throw errors::ConvergenceError("Elliptic integral CEL did not converge.");
        Piofk = 0.5 * pi<T>() * (a1 * m + b1) / (m * (m + p));
        Eofk = 0.5 * pi<T>() * (a2 * m + b2) / (m * (m + p1));
        Em1mKdm = 0.5 * pi<T>() * (a3 * m + b3) / (m * (m + p1));
    }

    /**
    Gradient of K

    */
    template <typename T>
    Eigen::AutoDiffScalar<T> K (const Eigen::AutoDiffScalar<T>& z)
    {
        typename T::Scalar ksq = z.value(), Kz = K(ksq), Ez = E(ksq);
        return Eigen::AutoDiffScalar<T>(
            Kz,
            z.derivatives() * (Ez / (1.0 - ksq) - Kz) / (2 * ksq)
        );
    }

    /**
    Gradient of E

    */
    template <typename T>
    Eigen::AutoDiffScalar<T> E (const Eigen::AutoDiffScalar<T>& z)
    {
        typename T::Scalar ksq = z.value(), Kz = K(ksq), Ez = E(ksq);
        return Eigen::AutoDiffScalar<T>(
            Ez,
            z.derivatives() * (Ez - Kz) / (2 * ksq)
        );
    }

} // namespace ellip
} // namespace starry

#endif
