/**
\file special.h
\brief Special functions.

*/

#ifndef _STARRY_OBLATE_SPECIAL_H_
#define _STARRY_OBLATE_SPECIAL_H_

#include "../quad.h"
#include "../utils.h"

namespace starry {
namespace oblate {
namespace special {

using std::abs;
using namespace utils;
using namespace quad;

/**
  The Gauss hypergeometric function 2F1.

*/
template <typename Scalar>
inline Scalar hyp2f1(const Scalar &a_, const Scalar &b_, const Scalar &c_,
                     const Scalar &z) {

  // Compute the value
  Scalar a = a_;
  Scalar b = b_;
  Scalar c = c_;
  Scalar term = a * b * z / c;
  Scalar value = 1.0 + term;
  int n = 1;
  while ((abs(term) > STARRY_2F1_MAXTOL) && (n < STARRY_2F1_MAXITER)) {
    a += 1;
    b += 1;
    c += 1;
    n += 1;
    term *= a * b * z / c / n;
    value += term;
  }
  if ((n == STARRY_2F1_MAXITER) && (abs(term) > STARRY_2F1_MINTOL)) {
    std::stringstream args;
    args << "a_ = " << a_ << ", "
         << "b_ = " << b_ << ", "
         << "c_ = " << c_ << ", "
         << "z = " << z;
    throw StarryException("Series for 2F1 did not converge.",
                          "oblate/special.h", "hyp2f1", args.str());
  }
  return value;
}

/**
  The Gauss hypergeometric function 2F1 and its z derivative.

*/
template <typename Scalar, int N>
inline ADScalar<Scalar, N> hyp2f1(const Scalar &a, const Scalar &b,
                                  const Scalar &c,
                                  const ADScalar<Scalar, N> &z) {
  ADScalar<Scalar, N> F;
  F.value() = hyp2f1(a, b, c, z.value());
  if (N > 0)
    F.derivatives() =
        z.derivatives() * a * b / c * hyp2f1(a + 1, b + 1, c + 1, z.value());
  return F;
}

/**
 * Special case of 2F1, defined as
 *
 *    2F1(-1/2, n + 1, n + 2, z)
 *
 */
template <typename Scalar, int N>
inline ADScalar<Scalar, N> hypspecial1(const int &n,
                                       const ADScalar<Scalar, N> &z) {
  using A = ADScalar<Scalar, N>;
  if ((0 <= z) && (z <= 0.85)) {
    return hyp2f1(-0.5, n + 1.0, n + 2.0, z);
  } else if ((0.85 < z) && (z < 1)) {
    A c1 = -2.0 * (n + 1.0) / 3.0 * pow(A(1.0 - z), 1.5);
    A c2 = 2.0 / 3.0;
    for (int m = 1; m < n + 1; ++m)
      c2 *= 2.0 * (m + 1.0) / (3.0 + 2.0 * m);
    return c1 * hyp2f1(n + 2.5, 1.0, 2.5, A(1.0 - z)) +
           c2 * hyp2f1(-0.5, n + 1.0, -0.5, A(1.0 - z));
  } else if (z < 0) {
    return sqrt(1.0 - z) * hyp2f1(-0.5, 1.0, n + 2.0, A(z / (z - 1.0)));
  } else if (z == 1) {
    return A(INFINITY);
  } else {
    std::stringstream args;
    args << "n_ = " << n << ", "
         << "z = " << z;
    throw StarryException("Complex result in `hyp2f1`.", "oblate/special.h",
                          "hypspecial1", args.str());
  }
}

/**
 * Special case of 2F1, defined as
 *
 *    2F1(1, n + 5/2, 5/2, z)
 *
 */
template <typename Scalar, int N>
inline ADScalar<Scalar, N> hypspecial2(const int &n,
                                       const ADScalar<Scalar, N> &z) {
  using A = ADScalar<Scalar, N>;
  return 3.0 / ((2.0 * n + 3.0) * (1.0 - z)) *
         hyp2f1(1.0, -1.0 * n, -1.0 * n - 0.5, A(1.0 / (1.0 - z)));
}

/**
  Integrand of the `pT_2` term, for numerical integration.

*/
template <typename Scalar>
inline Scalar p2_integrand(const Scalar &bo, const Scalar &ro, const Scalar &f,
                           const Scalar &theta, const Scalar &bc,
                           const Scalar &bs, const Scalar &phi) {
  Scalar cpt = cos(phi - theta);
  Scalar spt = sin(phi - theta);
  Scalar x = ro * cpt + bs;
  Scalar y = (ro * spt + bc) / (1 - f);
  Scalar z = sqrt(1 - x * x - y * y);
  if (z < 1e-12)
    z = 1e-12;
  if (z > 1 - 1e-12)
    z = 1 - 1e-12;
  return ro * (ro + bo * sin(phi)) * (1 - z * z * z) / (3 * (1 - z * z));
}

template <typename Scalar>
inline Scalar dp2dbo_integrand(const Scalar &bo, const Scalar &ro,
                               const Scalar &f, const Scalar &theta,
                               const Scalar &bc, const Scalar &bs,
                               const Scalar &phi) {
  Scalar cpt = cos(phi - theta);
  Scalar spt = sin(phi - theta);
  Scalar costheta = cos(theta);
  Scalar sintheta = sin(theta);
  Scalar sinphi = sin(phi);
  Scalar x = ro * cpt + bs;
  Scalar y = (ro * spt + bc) / (1 - f);
  Scalar z = sqrt(1 - x * x - y * y);
  if (z < 1e-12)
    z = 1e-12;
  if (z > 1 - 1e-12)
    z = 1 - 1e-12;
  Scalar s = 1 + z + z * z;
  Scalar q =
      ro * ((y * costheta) / (1 - f) + x * sintheta) * (ro + bo * sinphi);
  return (ro * s * sinphi * (1 + z) - q * (2 + z)) / (3 * (1 + z) * (1 + z));
}

template <typename Scalar>
inline Scalar dp2dro_integrand(const Scalar &bo, const Scalar &ro,
                               const Scalar &f, const Scalar &theta,
                               const Scalar &bc, const Scalar &bs,
                               const Scalar &phi) {
  Scalar cpt = cos(phi - theta);
  Scalar spt = sin(phi - theta);
  Scalar sinphi = sin(phi);
  Scalar x = ro * cpt + bs;
  Scalar y = (ro * spt + bc) / (1 - f);
  Scalar z = sqrt(1 - x * x - y * y);
  if (z < 1e-12)
    z = 1e-12;
  if (z > 1 - 1e-12)
    z = 1 - 1e-12;
  Scalar s = 1 + z + z * z;
  Scalar q = ro * (x * cpt + (y * spt) / (1 - f)) * (ro + bo * sinphi);
  return (2 * ro * s * (1 + z) + bo * s * sinphi * (1 + z) - q * (2 + z)) /
         (3 * (1 + z) * (1 + z));
}

template <typename Scalar>
inline Scalar dp2df_integrand(const Scalar &bo, const Scalar &ro,
                              const Scalar &f, const Scalar &theta,
                              const Scalar &bc, const Scalar &bs,
                              const Scalar &phi) {
  Scalar cpt = cos(phi - theta);
  Scalar spt = sin(phi - theta);
  Scalar sinphi = sin(phi);
  Scalar x = ro * cpt + bs;
  Scalar y = (ro * spt + bc) / (1 - f);
  Scalar z = sqrt(1 - x * x - y * y);
  if (z < 1e-12)
    z = 1e-12;
  if (z > 1 - 1e-12)
    z = 1 - 1e-12;
  Scalar q = ro * y * y * (ro + bo * sinphi);
  return -q / (1 - f) * (2 + z) / (3 * (1 + z) * (1 + z));
}

template <typename Scalar>
inline Scalar dp2dtheta_integrand(const Scalar &bo, const Scalar &ro,
                                  const Scalar &f, const Scalar &theta,
                                  const Scalar &bc, const Scalar &bs,
                                  const Scalar &phi) {
  Scalar cpt = cos(phi - theta);
  Scalar spt = sin(phi - theta);
  Scalar sinphi = sin(phi);
  Scalar x = ro * cpt + bs;
  Scalar y = (ro * spt + bc) / (1 - f);
  Scalar z = sqrt(1 - x * x - y * y);
  if (z < 1e-12)
    z = 1e-12;
  if (z > 1 - 1e-12)
    z = 1 - 1e-12;
  Scalar q = ro * x * y * (ro + bo * sinphi) * (2 - f) * f / (1 - f);
  return q * (2 + z) / (3 * (1 + z) * (1 + z));
}

/**
  Numerical evaluation of the `pT_2` integral.

*/
template <typename Scalar, int N>
inline ADScalar<Scalar, N>
p2_numerical(const ADScalar<Scalar, N> &bo, const ADScalar<Scalar, N> &ro,
             const ADScalar<Scalar, N> &f, const ADScalar<Scalar, N> &theta,
             const ADScalar<Scalar, N> &phi1, const ADScalar<Scalar, N> &phi2,
             Quad<Scalar> &QUAD) {

  // Constants
  ADScalar<Scalar, N> bc = bo * cos(theta);
  ADScalar<Scalar, N> bs = bo * sin(theta);

  // The functions for the integral and its derivatives
  std::function<Scalar(Scalar)> func = [bo, ro, f, theta, bc, bs](Scalar phi) {
    return p2_integrand(bo.value(), ro.value(), f.value(), theta.value(),
                        bc.value(), bs.value(), phi);
  };
  std::function<Scalar(Scalar)> dfuncdbo = [bo, ro, f, theta, bc,
                                            bs](Scalar phi) {
    return dp2dbo_integrand(bo.value(), ro.value(), f.value(), theta.value(),
                            bc.value(), bs.value(), phi);
  };
  std::function<Scalar(Scalar)> dfuncdro = [bo, ro, f, theta, bc,
                                            bs](Scalar phi) {
    return dp2dro_integrand(bo.value(), ro.value(), f.value(), theta.value(),
                            bc.value(), bs.value(), phi);
  };
  std::function<Scalar(Scalar)> dfuncdf = [bo, ro, f, theta, bc,
                                           bs](Scalar phi) {
    return dp2df_integrand(bo.value(), ro.value(), f.value(), theta.value(),
                           bc.value(), bs.value(), phi);
  };
  std::function<Scalar(Scalar)> dfuncdtheta = [bo, ro, f, theta, bc,
                                               bs](Scalar phi) {
    return dp2dtheta_integrand(bo.value(), ro.value(), f.value(), theta.value(),
                               bc.value(), bs.value(), phi);
  };

  // Compute the function value
  ADScalar<Scalar, N> res = 0.0;
  res.value() = QUAD.integrate(phi1.value(), phi2.value(), func);

  // Compute the derivatives.
  // Deriv wrt phi is easy; need to integrate for the other ones
  if (N > 0) {
    res.derivatives() =
        (func(phi2.value()) * phi2.derivatives() +
         func(phi1.value()) * phi1.derivatives() +
         QUAD.integrate(phi1.value(), phi2.value(), dfuncdbo) *
             bo.derivatives() +
         QUAD.integrate(phi1.value(), phi2.value(), dfuncdro) *
             ro.derivatives() +
         QUAD.integrate(phi1.value(), phi2.value(), dfuncdf) * f.derivatives() +
         QUAD.integrate(phi1.value(), phi2.value(), dfuncdtheta) *
             theta.derivatives());
  }
  return res;
}

} // namespace special
} // namespace oblate
} // namespace starry

#endif
