/**
\file special.h
\brief Numerical integration.

*/

#ifndef _STARRY_OBLATE_NUMERICAL_H_
#define _STARRY_OBLATE_NUMERICAL_H_

#include "../quad.h"
#include "../utils.h"

namespace starry {
namespace oblate {
namespace numerical {

using std::abs;
using namespace utils;
using namespace quad;

template <typename Scalar>
inline Vector<Scalar>
pTodd_integrand(const int &deg, const Scalar &bo, const Scalar &ro,
                const Scalar &f, const Scalar &theta, const Scalar &costheta,
                const Scalar &sintheta, const Scalar &phi) {

  // Pre-compute common terms
  int ncoeff = (deg + 1) * (deg + 1);
  Scalar cosphi = cos(phi);
  Scalar sinphi = sin(phi);
  Scalar cosvphi = costheta * cosphi + sintheta * sinphi;
  Scalar sinvphi = -sintheta * cosphi + costheta * sinphi;
  Scalar x = ro * cosvphi + bo * sintheta;
  Scalar y = (ro * sinvphi + bo * costheta) / (1.0 - f);
  Scalar z2 = 1 - x * x - y * y;
  Scalar z3 = z2 > 0 ? z2 * sqrt(z2) : 0.0;
  Scalar z3x = -ro * (1 - f) * sinvphi * z3;
  Scalar z3y = ro * cosvphi * z3;

  // Compute all the integrands
  Vector<Scalar> integrand(ncoeff);

  // Case 2
  integrand(2) = ro * (ro + bo * sinphi) * (1.0 - z3) / (3.0 - 3.0 * z2);

  // Cases 3-5
  Scalar xi, yj;
  int n;
  xi = 1.0;
  for (int i = 0; i < deg - 1; ++i) {
    if (is_even(i)) {
      // Case 3
      n = i * i + 6 * i + 7;
      if (n < ncoeff) {
        integrand(n) = xi * z3x;
      }
      // Case 4
      n += 2 * i + 7;
      if (n < ncoeff) {
        integrand(n) = xi * y * z3x;
      }
    }
    // Case 5
    yj = 1.0;
    for (int j = 0; j < deg - 1 - i; ++j) {
      n = (i + j) * (i + j) + 4 * i + 6 * j + 5;
      integrand(n) = xi * yj * z3y;
      yj *= y;
    }
    xi *= x;
  }

  return integrand;
}

/**
  Numerical evaluation of the `pT` integral for all odd `mu`.

*/
template <typename Scalar, int N>
inline Vector<ADScalar<Scalar, N>>
pTodd(const int &deg, const ADScalar<Scalar, N> &bo,
      const ADScalar<Scalar, N> &ro, const ADScalar<Scalar, N> &f,
      const ADScalar<Scalar, N> &theta, const ADScalar<Scalar, N> &costheta,
      const ADScalar<Scalar, N> &sintheta, const ADScalar<Scalar, N> &phi1,
      const ADScalar<Scalar, N> &phi2, Quad<Scalar> &QUAD) {

  // Compute the function value
  int ncoeff = (deg + 1) * (deg + 1);
  Vector<ADScalar<Scalar, N>> pT;
  pT.setZero(ncoeff);

  if (N > 0) {

    // TODO! Compute the derivatives
    throw std::runtime_error("Derivs not yet implemented!");

  } else {

    // Compute the function
    std::function<Vector<Scalar>(Scalar)> func = //
        [deg, bo, ro, f, theta, costheta, sintheta](Scalar phi) {
          return pTodd_integrand(deg, bo.value(), ro.value(), f.value(),
                                 theta.value(), costheta.value(),
                                 sintheta.value(), phi);
        };
    Vector<Scalar> res =
        QUAD.integrate(phi1.value(), phi2.value(), func, ncoeff);

    // Copy into ADScalar
    for (int n = 0; n < ncoeff; ++n) {
      pT(n).value() = res(n);
    }
  }
  return pT;
}

} // namespace numerical
} // namespace oblate
} // namespace starry

#endif
