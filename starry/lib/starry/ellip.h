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

#ifndef STARRY_NO_AUTODIFF
#include "AutoDiffScalar.h"
#endif

namespace ellip {

using std::abs;



  // Complete elliptic integral of the first kind
  template <typename T>
  T K (const T& ksq) {
    T kc = sqrt(1.0 - ksq), m = T(1.0), h;
    for (int i = 0; i < STARRY_ELLIP_MAX_ITER; ++i) {
      h = m;
      m += kc;
      if (abs(h - kc) / h <= STARRY_ELLIP_CONV_TOL) break;
      kc = sqrt(h * kc);
      m *= 0.5;
    }
    return M_PI / m;
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
      if (abs(m0 - kc) / m0 <= STARRY_ELLIP_CONV_TOL) break;
      kc = 2.0 * sqrt(kc * m0);
    }
    return M_PI_4 * a / m;
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
      if (abs(1.0 - kc / g) <= STARRY_ELLIP_CONV_TOL) break;
      kc = 2.0 * sqrt(e);
      e = kc * m0;
    }
    return M_PI_2 * (c * m0 + d) / (m0 * (m0 + p));
  }


#ifndef STARRY_NO_AUTODIFF
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

  // Gradient of E
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
      (n.derivatives() * 0.5 * (Ek + (Kk * (ksq - n_value) +
      Pnk * (nsq - ksq)) / n_value) / (n_value - 1.0) -
      ksq.derivatives() * 0.5 * (Ek / (ksq - 1.0) + Pnk)) /
      (ksq - n_value)
    );
  }

#endif

}; // namespace ellip

#endif
