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

//! Re-parameterize solution vector when abs(b - r) < STARRY_EPS_BMR_ZERO
#define STARRY_EPS_BMR_ZERO 1e-2

//! Re-parameterize solution vector when
//! 1 - STARRY_EPS_BMR_ONE < abs(b - r) < 1 + STARRY_EPS_BMR_ONE
#define STARRY_EPS_BMR_ONE 1e-5

//! Re-parameterize solution vector when
//! 1 - STARRY_EPS_BMR_ONE < abs(b + r) < 1 + STARRY_EPS_BPR_ONE
#define STARRY_EPS_BPR_ONE 1e-5

//! Re-parameterize solution vector when abs(b) < STARRY_EPS_B_ZERO
#define STARRY_EPS_B_ZERO 1e-1

namespace lld {

    using std::abs;
    using namespace utils;

    /**
    Return true if our regular expression for `s2` is likely to be unstable

    */
    template <typename T>
    inline bool s2_unstable(const T& b, const T& r) {
        if (r > 1)
            return true;
        else if (abs(b - r) < STARRY_EPS_BMR_ZERO)
            return true;
        else if ((abs(b - r) > 1 - STARRY_EPS_BMR_ONE) &&
                 (abs(b - r) < 1 + STARRY_EPS_BMR_ONE))
            return true;
        else if ((abs(b + r) > 1 - STARRY_EPS_BPR_ONE) &&
                 (abs(b + r) < 1 + STARRY_EPS_BPR_ONE))
            return true;
        else
            return false;
    }

    /**
    Term containing the elliptic integral of the third kind

    */
    template <typename T>
    inline T PiTerm(const T& b, const T& ksq, const T& bmr,
                    const T& bmr2, const T& bpr2, const T& bmrdbpr) {
        if ((b == 0) || (ksq == 1))
            return 0;
        else if ((bmr == 1) || (bmr == -1))
            return 0;
        else if (ksq < 1)
            return 3 * bmr * ellip::Pi(T(ksq * bpr2), ksq);
        else
            return 3 * bmrdbpr * ellip::Pi(T(1. / (ksq * bpr2)),
                                           T(1. / ksq)) / sqrt(1 - bmr2);
    }

    /**
    This is the Mandel & Agol solution for linear limb darkening,
    reparametrized for speed and stability

    */
    template <typename T>
    inline T Lambda(const T& b, const T& r, const T& ksq, const T& K, const T& E) {

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
                    Lambda1 = (1. / 3.) - 4. / (9. * pi<T>());
                } else if (r < 0.5) {
                    m = 4 * r2;
                    Lambda1 = (1. / 3.) + 2. / (9. * pi<T>()) *
                              ellip::CEL(m, T(sqrt(1.0 - m)), T(1), T(m - 3),
                                         T((1 - m) * (2 * m - 3)));
                } else {
                    m = 4 * r2;
                    minv = 1.0 / m;
                    Lambda1 = (1. / 3.) + 1. / (9. * pi<T>() * r) *
                              ellip::CEL(minv, T(sqrt(1.0 - minv)),
                                         T(1), T(m - 3), T(1 - m));
                }
            } else {
                k2 = onembpr2 / fourbr + 1;
                if ((b + r) > 1.0) {
                    // k^2 < 1
                    if (s2_unstable(b, r)) {
                        k2c = -onembpr2 / fourbr;
                        kc = sqrt(k2c);
                        Lambda1 = onembmr2 * (k2c * bmr * bpr *
                                              ellip::CEL(k2, kc, T(bmr2 * k2c), T(0), T(3))
                                  + ellip::CEL(k2, kc, T(1), T(-3 + 6 * r2 - 2 * br),
                                               onembpr2)) / (9. * pi<T>() * sqrt(br));

                    } else {
                        Lambda1 = ((bpr2 - 1) / bpr * (-2 * r * (2 * bpr2 - bpr * bmr - 3) * K +
                                   PiTerm(b, ksq, bmr, bmr2, bpr2, bmrdbpr)) - 2 * xi * E) /
                                  (9 * pi<T>() * sqrt(br));
                    }
                } else if ((b + r) < 1.0) {
                    // k^2 > 1
                    if (s2_unstable(b, r)) {
                        k2inv = 1.0 / k2;
                        k2c = onembpr2 / onembmr2;
                        kc = sqrt(k2c);
                        Eofk = ellip::CEL(k2inv, kc, T(1), T(1), k2c);
                        mu = 3 * bmrdbpr / onembmr2;
                        p = bmrdbpr * bmrdbpr * onembpr2 / onembmr2;
                        Lambda1 = 2 * sqrt(onembmr2) * (onembpr2 *
                                                        ellip::CEL(k2inv, kc, p, T(1 + mu), T(p + mu))
                                                        - (4 - 7 * r2 - b2) * Eofk) / (9. * pi<T>());
                    } else {
                        Lambda1 = 2 * ((1 - bpr2) * (sqrt(1 - bmr2) * K +
                                                     PiTerm(b, ksq, bmr, bmr2, bpr2, bmrdbpr)) -
                                       sqrt(1 - bmr2) * (4 - 7 * r2 - b2) * E) / (9 * pi<T>());
                    }
                } else {
                    // b + r  = 1 or k^2 = 1
                    Lambda1 = 2. / (3. * pi<T>()) * acos(1. - 2. * r) - 4 / (9. * pi<T>()) *
                              (3 + 2 * r - 8 * r2) * sqrt(br) - (2. / 3.) * T(r > 0.5);
                }
            }
        }
        return Lambda1;
    }


    /**
    Gradient of Lambda. We manually specify it to circumvent instabilities
    in certain limits via reparametrization in terms of CEL.
    See https://github.com/rodluger/starry/issues/113

    */
    template <typename T>
    Eigen::AutoDiffScalar<T> Lambda (const Eigen::AutoDiffScalar<T>& b,
                                     const Eigen::AutoDiffScalar<T>& r,
                                     const Eigen::AutoDiffScalar<T>& ksq,
                                     const Eigen::AutoDiffScalar<T>& K,
                                     const Eigen::AutoDiffScalar<T>& E)
    {
      typename T::Scalar b_value = b.value(),
                         r_value = r.value(),
                         ksq_value = ksq.value(),
                         K_value = K.value(),
                         E_value = E.value(),
                         Lambda_value = Lambda(b_value, r_value, ksq_value, K_value, E_value),
                         onembmr2 = 1 - (b_value - r_value) * (b_value - r_value),
                         onembpr2 = 1 - (b_value + r_value) * (b_value + r_value),
                         dLdb, dLdr, k2c, kc;

      if (ksq_value < 1) {
          typename T::Scalar sqrtbr = sqrt(b_value * r_value);
          k2c = -onembpr2 / (4 * b_value * r_value);
          kc = sqrt(k2c);
          dLdr = 1.0 / (pi<typename T::Scalar>() * sqrtbr) * ellip::CEL(ksq_value, kc, typename T::Scalar(1),
                                                             typename T::Scalar(2 * r_value * onembmr2),
                                                             typename T::Scalar(0));
          dLdb = onembmr2 / (3 * pi<typename T::Scalar>() * sqrtbr) * ellip::CEL(ksq_value, kc, typename T::Scalar(1),
                                                                      typename T::Scalar(-2 * r_value),
                                                                      typename T::Scalar(onembpr2 / b_value));
      } else {
          typename T::Scalar k2inv = 1.0 / ksq_value,
                             fac = 4.0 * r_value / pi<typename T::Scalar>() * sqrt(onembmr2);
          k2c = onembpr2 / onembmr2;
          kc = sqrt(k2c);
          dLdr = fac * ellip::CEL(k2inv, kc, typename T::Scalar(1), typename T::Scalar(1), k2c);
          dLdb = fac / 3.0 * ellip::CEL(k2inv, kc, typename T::Scalar(1), typename T::Scalar(-1), k2c);
      }

      return Eigen::AutoDiffScalar<T>(Lambda_value, b.derivatives() * dLdb + r.derivatives() * dLdr);
    }

    /**
    Compute the n=2 term of the *s^T* occultation solution vector.

    */
    template <typename T>
    inline T s2(const T& b, const T& r, const T& ksq, const T& K, const T& E) {
        return (2. * pi<T>() / 3.) * (1.0 - 1.5 * Lambda(b, r, ksq, K, E) - T(r > b));
    }

} // namespace lld

#endif
