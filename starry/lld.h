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

namespace lld {

    using std::abs;

    // Return true if our regular expression for `s2` is likely to be unstable
    template <typename T>
    inline bool s2_unstable(const T& b, const T& r) {
        if (r > 1)
            return true;
        else if (abs(b - r) < STARRY_EPS_BMR_ZERO)
            return true;
        else if ((abs(b - r) > 1 - STARRY_EPS_BMR_ONE) && (abs(b - r) < 1 + STARRY_EPS_BMR_ONE))
            return true;
        else if ((abs(b + r) > 1 - STARRY_EPS_BPR_ONE) && (abs(b + r) < 1 + STARRY_EPS_BPR_ONE))
            return true;
        else
            return false;
    }

    // Term containing the elliptic integral of the third kind
    template <typename T>
    inline T PITerm(const T& b, const T& r, const T& ksq, const T& bmr, const T& bmr2, const T& bpr2, const T& bmrdbpr) {
        if ((b == 0) || (ksq == 1))
            return 0;
        else if ((bmr == 1) || (bmr == -1))
            return 0;
        else if (ksq < 1)
            return 3 * bmr * ellip::PI(T(ksq * bpr2), ksq);
        else
            return 3 * bmrdbpr * ellip::PI(T(1. / (ksq * bpr2)), T(1. / ksq)) / sqrt(1 - bmr2);
    }

    // Compute the n=2 term of the *s^T* occultation solution vector.
    // This is the Mandel & Agol solution for linear limb darkening,
    // reparametrized for speed and stability
    template <typename T>
    inline T s2(const T& b, const T& r, const T& ksq, const T& K, const T& E, const T& pi) {

        T Lambda1, k2, k2c, kc, Eofk, k2inv, mu, p;
        T m, minv;
        T r2 = r * r;
        T b2 = b * b;
        T br = b * r;
        T bpr = b + r;
        T bmr = b - r;
        T bpr2 = bpr * bpr;
        T bmr2 = bmr * bmr;
        T xi = 2 * br * (4 - 7 * r2 - b2);
        T onembpr2 = 1 - bpr * bpr;
        T onembmr2 = 1 - bmr * bmr;
        T fourbr = 4 * br;
        T bmrdbpr = bmr / bpr;

        if ((b >= 1.0 + r) ||  (r == 0.0)) {
            // No occultation
            Lambda1 = 0;
        } else if  (b <= r - 1.0) {
            // Full occultation
            Lambda1 = 0;
        } else {
            if (b == 0) {
                Lambda1 = -(2. / 3.) * sqrt((1.0 - r2) * (1.0 - r2) * (1.0 - r2));
            } else if (b == r) {
                if (r == 0.5) {
                    Lambda1 = (1. / 3.) - 4. / (9. * pi);
                } else if (r < 0.5) {
                    m = 4 * r2;
                    Lambda1 = (1. / 3.) + 2. / (9. * pi) * ellip::CEL(m, T(sqrt(1.0 - m)), T(1), T(m - 3), T((1 - m) * (2 * m - 3)), pi);
                } else {
                    m = 4 * r2;
                    minv = 1.0 / m;
                    Lambda1 = (1. / 3.) + 1. / (9. * pi * r) * ellip::CEL(minv, T(sqrt(1.0 - minv)), T(1), T(m - 3), T(1 - m), pi);
                }
            } else {
                k2 = onembpr2 / fourbr + 1;
                if ((b + r) > 1.0) {
                    // k^2 < 1
                    if (s2_unstable(b, r)) {
                        k2c = -onembpr2 / fourbr;
                        kc = sqrt(k2c);
                        Lambda1 = onembmr2 * (k2c * bmr * bpr * ellip::CEL(k2, kc, T(bmr2 * k2c), T(0), T(3), pi)
                                  + ellip::CEL(k2, kc, T(1), T(-3 + 6 * r2 - 2 * br), onembpr2, pi)) / (9. * pi * sqrt(br));

                    } else {
                        Lambda1 = ((bpr2 - 1) / bpr * (-2 * r * (2 * bpr2 - bpr * bmr - 3) * K + PITerm(b, r, ksq, bmr, bmr2, bpr2, bmrdbpr)) - 2 * xi * E) / (9 * pi * sqrt(br));
                    }
                } else if ((b + r) < 1.0) {
                    // k^2 > 1
                    if (s2_unstable(b, r)) {
                        k2inv = 1.0 / k2;
                        k2c = onembpr2 / onembmr2;
                        kc = sqrt(k2c);
                        Eofk = ellip::CEL(k2inv, kc, T(1), T(1), k2c, pi);
                        mu = 3 * bmrdbpr / onembmr2;
                        p = bmrdbpr * bmrdbpr * onembpr2 / onembmr2;
                        Lambda1 = 2 * sqrt(onembmr2) * (onembpr2 * ellip::CEL(k2inv, kc, p, T(1 + mu), T(p + mu), pi) - (4 - 7 * r2 - b2) * Eofk) / (9. * pi);
                    } else {
                        Lambda1 = 2 * ((1 - bpr2) * (sqrt(1 - bmr2) * K + PITerm(b, r, ksq, bmr, bmr2, bpr2, bmrdbpr)) - sqrt(1 - bmr2) * (4 - 7 * r2 - b2) * E) / (9 * pi);
                    }
                } else {
                    // b + r  = 1 or k^2 = 1
                    Lambda1 = 2. / (3. * pi) * acos(1. - 2. * r) - 4 / (9. * pi) * (3 + 2 * r - 8 * r2) * sqrt(br) - (2. / 3.) * T(r > 0.5);
                }
            }
        }

        return (2. * pi / 3.) * (1.0 - 1.5 * Lambda1 - T(r > b));

    }

}; // namespace lld

#endif
