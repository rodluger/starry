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
inline Matrix<Scalar>
pTodd_integrand(const int &deg, const Scalar &bo, const Scalar &ro,
                const Scalar &f, const Scalar &theta, const Scalar &costheta,
                const Scalar &sintheta, const Vector<Scalar> &phi) {

  // Pre-compute common terms
  int sz = phi.size();
  int ncoeff = (deg + 1) * (deg + 1);
  auto one = Vector<Scalar>::Ones(sz);
  auto zero = Vector<Scalar>::Zero(sz);
  Vector<Scalar> cosphi(sz), sinphi(sz);
  cosphi.array() = cos(phi.array());
  sinphi.array() = sin(phi.array());
  Vector<Scalar> cosvphi = costheta * cosphi + sintheta * sinphi;
  Vector<Scalar> sinvphi = -sintheta * cosphi + costheta * sinphi;
  Vector<Scalar> x = ro * cosvphi + bo * sintheta * one;
  Vector<Scalar> y = (ro * sinvphi + bo * costheta * one) / (1.0 - f);
  Vector<Scalar> z2 = one - x.cwiseProduct(x) - y.cwiseProduct(y);
  Vector<Scalar> z3(sz);
  z3.array() =
      (z2.array() < 0).select(zero.array(), z2.array() * sqrt(z2.array()));
  Vector<Scalar> z3x = -ro * (1 - f) * sinvphi.cwiseProduct(z3);
  Vector<Scalar> z3y = ro * cosvphi.cwiseProduct(z3);

  // Compute all the integrands at all points
  Matrix<Scalar> integrand(sz, ncoeff);

  // Case 2
  integrand.col(2) = ro * (ro * one + bo * sinphi)
                              .cwiseProduct(one - z3)
                              .cwiseQuotient(3.0 * (one - z2));

  // Cases 3-5
  Vector<Scalar> xi(sz), yj(sz);
  int n;
  xi.setOnes();
  for (int i = 0; i < deg - 1; ++i) {
    if (is_even(i)) {
      // Case 3
      n = i * i + 6 * i + 7;
      if (n < ncoeff) {
        integrand.col(n) = xi.cwiseProduct(z3x);
      }
      // Case 4
      n += 2 * i + 7;
      if (n < ncoeff) {
        integrand.col(n) = xi.cwiseProduct(y).cwiseProduct(z3x);
      }
    }
    // Case 5
    yj.setOnes();
    for (int j = 0; j < deg - 1 - i; ++j) {
      n = (i + j) * (i + j) + 4 * i + 6 * j + 5;
      integrand.col(n) = xi.cwiseProduct(yj).cwiseProduct(z3y);
      yj = yj.cwiseProduct(y);
    }
    xi = xi.cwiseProduct(x);
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
    std::function<Matrix<Scalar>(Vector<Scalar>)> func = //
        [deg, bo, ro, f, theta, costheta, sintheta](Vector<Scalar> phi) {
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
