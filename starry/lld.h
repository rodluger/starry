/**
Linear limb darkening (s2) integration utilities.

*/

#ifndef _STARRY_S2_H_
#define _STARRY_S2_H_

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include "ellip.h"
#include "errors.h"

using std::abs;

namespace lld {

// Re-parametrize EllipticPi when b + r is within this distance of 1
#define STARRY_BPLUSR_THRESH_S2                 1.e-5

// Re-parametrize s2() when |b-r| < this value
#define STARRY_BMINUSR_THRESH_S2                1.e-2

// Taylor expansion of the s2 term above this radius
#define STARRY_RADIUS_THRESH_S2                 2.0

// Taylor expansion of E() - K() when r >= 1
#define STARRY_EMINUSK_ORDER                          13
static const double STARRY_EMINUSK_COEFF[STARRY_EMINUSK_ORDER] =
{0, 0.25, 0.09375, 0.05859375, 0.042724609375, 0.0336456298828125,
 0.027757644653320312, 0.023627042770385742, 0.020568184554576874,
 0.018211413407698274, 0.016339684807462618, 0.01481712326858542,
 0.013554300262740071};

    // Heaviside step function
    template <typename T>
    inline T step(T x) {
        if (x <= 0)
            return 0;
        else
            return 1;
    }

    // Term containing the elliptic integral of the third kind
    template <typename T>
    inline T PITerm(T& b, T& r, T& ksq, T& pi, bool taylor) {
        T res;
        if ((b == 0) || (ksq == 1))
            res = 0;
        else if (ksq < 1)
            // TODO: We need to find a reparametrization of PI here, since it diverges
            // when b = r. My expression is valid to *zeroth* order near b = r.
            if ((taylor) && (abs(b - r) < 1e-8) && (r < STARRY_RADIUS_THRESH_S2)) {
                if (b > r)
                    res = -(6. * pi * r * r) / (1 - 4. * r * r);
                else
                    res = (6. * pi * r * r) / (1 - 4. * r * r);
            } else {
                res = 3 * (b - r) * ellip::PI(T(ksq * (b + r) * (b + r)), ksq);
            }
        else {
            T EPI;
            if ((taylor) && (abs(b - r) < STARRY_BMINUSR_THRESH_S2)) {
                // This is a reparameterization of the complete elliptic integral
                // of the third kind, necessary to suppress numerical instabilities when b ~ r.
                // It relies on expressing PI in terms of the incomplete elliptic integrals
                // of the first and second kind. I haven't done speed tests, but I suspect
                // it has to be slower, so we only do this when b is really close to r.
                // Use transformation of 17.7.14 in Abramowitz & Stegun:
                T one_minus_n = (b - r) * (b - r) *
                                (1. - (b + r) * (b + r)) /
                                (1. - (b - r) * (b - r)) /
                                ((b + r) * (b + r));
                T EK = ellip::K(T(1. / ksq));
                T EE = ellip::E(T(1. / ksq));
                T psi = asin(sqrt(one_minus_n / (1. - 1. / ksq)));
                T mc = 1. - 1. / ksq;
                // Compute Heuman's Lambda Function via A&S 17.4.40:
                T EEI = ellip::E(mc, psi);
                T EFI = ellip::F(mc, psi);
                T HLam = 2. / pi * (EK * EEI - (EK - EE) * EFI);
                T d2 = sqrt((1. / one_minus_n - 1.) / (1. - one_minus_n - 1. / ksq));
                // Equation 17.7.14 in A&S:
                EPI = EK + 0.5 * pi * d2 * (1. - HLam);
            } else {
                // Compute the elliptic integral directly
                EPI = ellip::PI(T(1. / (ksq * (b + r) * (b + r))), T(1. / ksq));
            }
            // TODO: There are small numerical issue here. As b - r --> 1,
            // the denominator diverges. Should re-parametrize.
            if (abs(b - r) != 1.0)
                res = 3 * (b - r) / (b + r) * EPI /
                       sqrt(1 - (b - r) * (b - r));
            else
                res = 0;
            }

        return res;
    }

    // Taylor expand the difference between the elliptic integrals
    // for the s2 term when r >= 1; much more numerically stable!
    template <typename T>
    inline T s2_taylor(T& b, T& r, T& ksq, T& K, T& E, T& pi) {
        T x = r / b;
        T eps = x - 1;
        T EP, goodterm;
        if (abs(b - r) > 1e-8) {
            EP = ellip::PI(T(1 - 1. / ((b - r) * (b - r))), ksq);
            goodterm = (3 * (b + r) / (b - r) * EP) / sqrt(b * r);
        } else {
            // Numerically stable first order expansion when b = r
            goodterm = 6 * pi * r * (0.5 - step(T(r - b))) / sqrt(b * r);
        }
        T EminusK = 0;
        T ksqi = 1;
        for (int i = 0; i < STARRY_EMINUSK_ORDER; i++) {
            EminusK += STARRY_EMINUSK_COEFF[i] * ksqi;
            ksqi *= ksq;
        }
        EminusK *= -pi;
        T taylor = 2 * b * b * b * sqrt(x) * (EminusK * (16 + 28 * eps + 14 * eps * eps) - eps * eps * (2 + 3 * eps) * K);
        T badterm = taylor + sqrt(b * r) * ((8 - 3 / (b * r) + 12 / (b / r)) * K - 16 * E);
        T Lambda = (badterm + goodterm) / (9 * pi);
        return (2. * pi / 3.) * (1 - 1.5 * Lambda - step(T(r - b)));
    }

    /* Eric Agol's reparametrized solution for Lambda when b + r is very close to 1.
       In this limit, the elliptic integral Pi diverges, so we need to reparameterize it.

       Specifically, this transforms complete elliptic integral of the third kind using
       Byrd & Friedman equation 117.06 (first equation).
       Note that B&F 117.06 has a sign error - right hand side should be negative.

       The expression Piofnk3 is equal to Pi(n,ksq)*((r+b)^2-1)/(b+r)
    */
    template <typename T>
    inline T LambdaBPlusROnePlusEpsilon(T& b, T& r, T& ksq, T& K, T& E, T& pi) {
        T mc = 1.0 - ksq;
        T beta = asin(sqrt(b * r) * 2 / (b + r));
        T xi = 2 * b * r * (4 - 7 * r * r - b * b);
        T Kprime = ellip::K(mc);
        T Piprime = ellip::PI(T(-(1 / ((b + r) * (b + r)) - 1)), mc);
        T Fprime = ellip::F(mc, beta);
        T Piofnk3 = (K * (-Kprime + Piprime) / (b + r) + pi * sqrt(b * r) / abs(b - r) * Fprime) / Kprime;
        return (((r + b) * (r + b) - 1) / (r + b) * (-2 * r * (2 * (r + b) * (r + b) + (r + b) * (r - b) - 3) * K) + 3 * (b - r) * Piofnk3 - 2 * xi * E) / (9 * pi * sqrt(b * r));
    }

    /* Eric Agol's reparametrized solution for Lambda when b + r is very close to 1.
       In this limit, the elliptic integral Pi diverges, so we need to reparameterize it.

       See notes in `LambdaBPlusROnePlusEpsilon` above.

       The expression Piofnk3 is equal to Pi(n,m)/(b+r)*(1-(r+b)^2)/sqrt(1-(b-r)^2)
    */
    template <typename T>
    inline T LambdaBPlusROneMinusEpsilon(T& b, T& r, T& ksq, T& K, T& E, T& pi) {
        T mc = 1.0 - 1.0 / ksq;
        T beta = asin(sqrt(1.0 - (b - r) * (b - r)));
        T Kprime = ellip::K(mc);
        T Piprime = ellip::PI(T(-((b + r) * (b + r) - 1)), mc);
        T Fprime = ellip::F(mc, beta);
        T Piofnk3 = -(b + r) * K * (1.0 - Piprime / Kprime) / sqrt(1.0 - (b - r) * (b - r)) + pi / 2 / abs(b - r) * Fprime / Kprime;
        return 2 * ((1 - (r + b) * (r + b)) * sqrt(1 - (b - r) * (b - r)) * K + 3 * (b - r) * Piofnk3 - sqrt(1 - (b - r) * (b - r)) * (4 - 7 * r * r - b * b) * E) / (9 * pi);
    }

    // Compute the n=2 term of the *s^T* occultation solution vector.
    // This is the Mandel & Agol solution for linear limb darkening,
    // reparametrized for speed and stability
    template <typename T>
    inline T s2(T& b, T& r, T& ksq, T& K, T& E, T& pi, bool taylor) {

        // Taylor expand for r > 2?
        if ((taylor) && (r >= STARRY_RADIUS_THRESH_S2))
            return s2_taylor(b, r, ksq, K, E, pi);

        T Lambda;
        T r2 = r * r;
        T b2 = b * b;
        T xi = 2 * b * r * (4 - 7 * r2 - b2);
        T bpr = b + r;
        T bpr2 = bpr * bpr;
        T bmr = b - r;
        if (b == 0) {
            Lambda = -2. / 3. * pow(1. - r2, 1.5);
        } else if (b == r) {
            if (r == 0.5)
                Lambda = (1. / 3.) - 4. / (9. * pi);
            else if (r < 0.5)
                Lambda = (1. / 3.) +
                         2. / (9. * pi) * (4. * (2. * r2 - 1.) * ellip::E(T(4 * r2)) +
                         (1 - 4 * r2) * ellip::K(T(4 * r2)));
            else
                Lambda = (1. / 3.) +
                         16. * r / (9. * pi) * (2. * r2 - 1.) * ellip::E(T(1. / (4 * r2))) -
                         (1 - 4 * r2) * (3 - 8 * r2) / (9 * pi * r) * ellip::K(T(1. / (4 * r2)));
        } else {
            if (ksq < 1) {
                if ((!taylor) || (b + r > 1 + STARRY_BPLUSR_THRESH_S2))
                    Lambda = ((bpr2 - 1) / bpr * (-2 * r * (2 * bpr2 - bpr * bmr - 3) * K + PITerm(b, r, ksq, pi, taylor)) - 2 * xi * E) / (9 * pi * sqrt(b * r));
                else
                    Lambda = LambdaBPlusROnePlusEpsilon(b, r, ksq, K, E, pi);
            } else if (ksq > 1) {
                if ((!taylor) || (b + r < 1 - STARRY_BPLUSR_THRESH_S2)) {
                    T bmr2 = bmr * bmr;
                    Lambda = 2 * ((1 - bpr2) * (sqrt(1 - bmr2) * K + PITerm(b, r, ksq, pi, taylor)) - sqrt(1 - bmr2) * (4 - 7 * r2 - b2) * E) / (9 * pi);
                } else {
                    Lambda = LambdaBPlusROneMinusEpsilon(b, r, ksq, K, E, pi);
                }
            } else {
                Lambda = 2. / (3. * pi) * acos(1. - 2 * r) -
                         4 / (9 * pi) * (3 + 2 * r - 8 * r2) * sqrt(b * r) -
                         2. / 3. * step(r - 0.5);
            }
        }
        return (2. * pi / 3.) * (1 - 1.5 * Lambda - step(T(-bmr)));
    }

}; // namespace lld

#endif
