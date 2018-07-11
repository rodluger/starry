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
#include "constants.h"
#include "errors.h"
#include <limits>
#include <unsupported/Eigen/AutoDiff>
#include <boost/math/special_functions/ellint_1.hpp>
#include <boost/math/special_functions/ellint_2.hpp>

namespace ellip {

  using boost::math::ellint_1;
  using boost::math::ellint_2;
  using std::abs;

  // EA: Elliptic integral convergence tolerance should be sqrt of machine precision
  static const double tol_double = sqrt(std::numeric_limits<double>::epsilon());
  static const Multi tol_Multi = sqrt(std::numeric_limits<Multi>::epsilon());

  template <typename T>
  inline T tol(){ return T(tol_double); }

  template <>
  inline Multi tol(){ return tol_Multi; }

  // Incomplete elliptic integral of the first kind
  // Currently using boost's implementation
  template <typename T>
  T F (const T& ksq, const T& phi) {
      return ellint_1(sqrt(ksq), phi);
  }

  // Incomplete elliptic integral of the second kind
  // Currently using boost's implementation
  template <typename T>
  T E (const T& ksq, const T& phi) {
      return ellint_2(sqrt(ksq), phi);
  }

  // Complete elliptic integral of the first kind
  template <typename T>
  T K (const T& ksq) {
    T kc = sqrt(1.0 - ksq), m = T(1.0), h;
    for (int i = 0; i < STARRY_ELLIP_MAX_ITER; ++i) {
      h = m;
      m += kc;
      if (abs(h - kc) / h <= tol<T>()) return M_PI / m;
      kc = sqrt(h * kc);
      m *= 0.5;
    }
    throw errors::Elliptic("K");
  }

  // Complete elliptic integral of the second kind
  template <typename T>
  T E (const T& ksq) {
    T b = 1.0 - ksq, kc = sqrt(b), m = T(1.0), c = T(1.0), a = b + 1.0, m0;
    for (int i = 0; i < STARRY_ELLIP_MAX_ITER; ++i) {
      b = 2.0 * (c * kc + b);
      c = a;
      m0 = m;
      m += kc;
      a += b / m;
      if (abs(m0 - kc) / m0 <= tol<T>()) return M_PI_4 * a / m;
      kc = 2.0 * sqrt(kc * m0);
    }
    throw errors::Elliptic("E");
  }

  // Complete elliptic integral of the third kind
  template <typename T>
  T PI (const T& n, const T& ksq) {
    T kc = sqrt(1.0 - ksq), p = sqrt(1.0 - n), m0 = 1.0, c = 1.0, d = 1.0 / p, e = kc, f, g;
    for (int i = 0; i < STARRY_ELLIP_MAX_ITER; ++i) {
      f = c;
      c += d / p;
      g = e / p;
      d = 2.0 * (f * g + d);
      p = g + p;
      g = m0;
      m0 = kc + m0;
      if (abs(1.0 - kc / g) <= tol<T>()) return M_PI_2 * (c * m0 + d) / (m0 * (m0 + p));
      kc = 2.0 * sqrt(e);
      e = kc * m0;
    }
    throw errors::Elliptic("PI");
  }

  // Computes the function cel(kc, p, a, b) from Bulirsch (1969)
  template <typename T>
  T CEL (const T& ksq0, const T& kc0, const T& p0, const T& a0, const T& b0, const T& pi) {
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
      // If k^2 is very small, we get better precision evaluating `kc` like this
      if (ksq < 1e-5) kc = sqrt(1 - ksq);
      // We actually need kc to be nonzero, so let's set it to a very small number
      if ((ksq == 1) || (kc == 0)) kc = mach_eps<T>() * ksq;
      // I haven't encountered cases where k^2 > 1 due to roundoff error,
      // but they could happen. If so, change the line below to avoid an exception
      if (ksq > 1) throw errors::Elliptic("CEL (ksq > 1)");
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
            return pi / 2 * (a * m + b) / (m * (m + p));
      }
      throw errors::Elliptic("CEL");
  }

  // Gradient of F
  template <typename T>
  Eigen::AutoDiffScalar<T> F (const Eigen::AutoDiffScalar<T>& ksq,
                              const Eigen::AutoDiffScalar<T>& phi)
  {
    typename T::Scalar ksq_value = ksq.value(),
                       phi_value = phi.value(),
                       F_value = F(ksq_value, phi_value),
                       E_value = E(ksq_value, phi_value),
                       sin_phi = sin(phi_value),
                       cos_phi = cos(phi_value),
                       fac = 1. / sqrt(1 - ksq_value * sin_phi * sin_phi);
    return Eigen::AutoDiffScalar<T>(
      F_value,
      phi.derivatives() * fac +
      ksq.derivatives() * (0.5 * cos_phi * sin_phi * fac / (ksq_value - 1) -
                           E_value / (2 * ksq_value * (ksq_value - 1)) -
                           F_value / (2 * ksq_value))
    );
  }

  // Gradient of E (incomplete)
  template <typename T>
  Eigen::AutoDiffScalar<T> E (const Eigen::AutoDiffScalar<T>& ksq,
                              const Eigen::AutoDiffScalar<T>& phi)
  {
    typename T::Scalar ksq_value = ksq.value(),
                       phi_value = phi.value(),
                       F_value = F(ksq_value, phi_value),
                       E_value = E(ksq_value, phi_value),
                       sin_phi = sin(phi_value),
                       fac = sqrt(1 - ksq_value * sin_phi * sin_phi);
    return Eigen::AutoDiffScalar<T>(
      F_value,
      phi.derivatives() * fac +
      ksq.derivatives() * (E_value - F_value) / (2 * ksq_value)
    );
  }

  // Gradient of K
  template <typename T>
  Eigen::AutoDiffScalar<T> K (const Eigen::AutoDiffScalar<T>& z)
  {
    typename T::Scalar ksq = z.value(),
                       Kz = K(ksq),
                       Ez = E(ksq);
    return Eigen::AutoDiffScalar<T>(
      Kz,
      z.derivatives() * (Ez / (1.0 - ksq) - Kz) / (2 * ksq)
    );
  }

  // Gradient of E (complete)
  template <typename T>
  Eigen::AutoDiffScalar<T> E (const Eigen::AutoDiffScalar<T>& z)
  {
    typename T::Scalar ksq = z.value(),
                       Kz = K(ksq),
                       Ez = E(ksq);
    return Eigen::AutoDiffScalar<T>(
      Ez,
      z.derivatives() * (Ez - Kz) / (2 * ksq)
    );
  }

  // Gradient of PI
  // TODO: We should use the cached version of K and E to speed this up!
  template <typename T>
  Eigen::AutoDiffScalar<T> PI (const Eigen::AutoDiffScalar<T>& n,
                               const Eigen::AutoDiffScalar<T>& ksq)
  {
    typename T::Scalar ksq_value = ksq.value(),
                       n_value = n.value(),
                       Kk = K(ksq_value),
                       Ek = E(ksq_value),
                       Pnk = PI(n_value, ksq_value),
                       nsq = n_value * n_value;
    return Eigen::AutoDiffScalar<T>(
      Pnk,
      (n.derivatives() * 0.5 * (Ek + (Kk * (ksq_value - n_value) +
      Pnk * (nsq - ksq_value)) / n_value) / (n_value - 1.0) -
      ksq.derivatives() * 0.5 * (Ek / (ksq_value - 1.0) + Pnk)) /
      (ksq_value - n_value)
    );
  }

  // Gradient of CEL
  // NOTE: I don't think this is currently used in the code.
  template <typename T>
  Eigen::AutoDiffScalar<T> CEL (const Eigen::AutoDiffScalar<T>& ksq,
                                const Eigen::AutoDiffScalar<T>& kc,
                                const Eigen::AutoDiffScalar<T>& p,
                                const Eigen::AutoDiffScalar<T>& a,
                                const Eigen::AutoDiffScalar<T>& b,
                                const Eigen::AutoDiffScalar<T>& pi)
  {
    typename T::Scalar ksq_value = ksq.value(),
                       kc_value = kc.value(),
                       kc2 = kc_value * kc_value,
                       p_value = p.value(),
                       a_value = a.value(),
                       b_value = b.value(),
                       pi_value = pi.value(),
                       oneminusp = 1 - p_value,
                       kc2minusp = kc2 - p_value,
                       CEL_value = CEL(ksq_value, kc_value, p_value, a_value, b_value, pi_value),
                       dCdkc, dCda, dCdb, lambda, dCdp;

      dCdkc = (-kc_value / (p_value - kc2)) * (CEL(ksq_value, kc_value, kc2, a_value, b_value, pi_value) - CEL_value);
      if (a.derivatives().any())
          dCda = CEL(ksq_value, kc_value, p_value, typename T::Scalar(1), typename T::Scalar(0), pi_value);
      else
          dCda = 0;
      if (b.derivatives().any())
          dCdb = CEL(ksq_value, kc_value, p_value, typename T::Scalar(0), typename T::Scalar(1), pi_value);
      else
          dCdb = 0;
      if (p.derivatives().any()) {
          lambda = kc2 * (b_value + a_value * p_value - 2 * b_value * p_value) + p_value * (3 * b_value * p_value - a_value * p_value * p_value - 2 * b_value);
          dCdp = (CEL(ksq_value, kc_value, p_value, typename T::Scalar(0), lambda, pi_value) + (b_value - a_value * p_value) * CEL(ksq_value, kc_value, typename T::Scalar(1), oneminusp, kc2minusp, pi_value)) / (-2 * p_value * oneminusp * kc2minusp);
          // Equivalent but probably less stable version:
          // dCdp = (2 * p_value * b_value - p_value * a_value - b_value) / (2 * p_value * (1 - p_value)) * CEL(ksq_value, kc_value, p_value, typename T::Scalar(0), typename T::Scalar(1), pi_value) +
          // (b_value - a_value * p_value) / (2 * p_value * (1 - p_value) * (p_value - kc2)) * CEL(ksq_value, kc_value, typename T::Scalar(1), typename T::Scalar(1), kc2, pi_value) -
          // (b_value - a_value * p_value) / (2 * (1 - p_value) * (p_value - kc2)) * CEL(ksq_value, kc_value, p_value, typename T::Scalar(1), typename T::Scalar(1), pi_value);
      } else {
          dCdp = 0;
      }

    return Eigen::AutoDiffScalar<T>(
      CEL_value,
      // TODO: Why do I not need this line ?!
      // Tests show the gradient is correect without it
      //ksq.derivatives() * dCdkc / (-2 * kc_value) +
      kc.derivatives() * dCdkc +
      p.derivatives() * dCdp +
      a.derivatives() * dCda +
      b.derivatives() * dCdb
    );
  }

}; // namespace ellip

#endif
