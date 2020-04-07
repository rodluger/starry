/**
\file special.h
\brief Special functions.

*/

#ifndef _STARRY_REFLECTED_SPECIAL_H_
#define _STARRY_REFLECTED_SPECIAL_H_

#include "../quad.h"
#include "../utils.h"
#include "constants.h"

namespace starry {
namespace reflected {
namespace special {

using std::abs;
using namespace utils;
using namespace quad;

/**

  Return the sum over pairwise differences of an array.

  This is used to evaluate a (series of) definite integral(s) given
  the antiderivatives at each of the integration limits.

*/
template <typename T> inline T pairdiff(const Vector<T> &array) {
  size_t K = array.size();
  if (K > 1) {
    if (K % 2 == 0) {
      int sgn = -1;
      T result = 0.0;
      for (size_t i = 0; i < K; ++i) {
        result += sgn * array(i);
        sgn *= -1;
      }
      return result;
    } else {
      throw std::runtime_error(
          "Array length must be even in call to `pairdiff`.");
    }
  } else if (K == 1) {
    throw std::runtime_error(
        "Array length must be even in call to `pairdiff`.");
  } else {
    // Empty array
    return 0.0;
  }
}

/**
  Integrand of the P_2 term, for numerical integration.
*/
template <typename T>
inline T P2_integrand(const T &bo, const T &ro, const T &phi) {
  T c = cos(phi);
  T z = 1 - ro * ro - bo * bo - 2 * bo * ro * c;
  if (z < 1e-12)
    z = 1e-12;
  if (z > 1 - 1e-12)
    z = 1 - 1e-12;
  return (1.0 - z * sqrt(z)) / (1.0 - z) * (ro + bo * c) * ro / 3.0;
}

/**
  Derivative of the integrand of the P_2 term, for numerical integration.
*/
template <typename T>
inline T dP2dbo_integrand(const T &bo, const T &ro, const T &phi) {
  T c = cos(phi);
  T z = 1 - ro * ro - bo * bo - 2 * bo * ro * c;
  if (z < 1e-12)
    z = 1e-12;
  if (z > 1 - 1e-12)
    z = 1 - 1e-12;
  T P = (1.0 - z * sqrt(z)) / (1.0 - z) * (ro + bo * c) * ro / 3.0;
  T q = 3.0 * sqrt(z) / (1.0 - z * sqrt(z)) - 2.0 / (1.0 - z);
  return P * ((bo + ro * c) * q + 1.0 / (bo + ro / c));
}

/**
  Derivative of the integrand of the P_2 term, for numerical integration.
*/
template <typename T>
inline T dP2dro_integrand(const T &bo, const T &ro, const T &phi) {
  T c = cos(phi);
  T z = 1 - ro * ro - bo * bo - 2 * bo * ro * c;
  if (z < 1e-12)
    z = 1e-12;
  if (z > 1 - 1e-12)
    z = 1 - 1e-12;
  T P = (1.0 - z * sqrt(z)) / (1.0 - z) * (ro + bo * c) * ro / 3.0;
  T q = 3.0 * sqrt(z) / (1.0 - z * sqrt(z)) - 2.0 / (1.0 - z);
  return P * ((ro + bo * c) * q + 1.0 / ro + 1.0 / (ro + bo * c));
}

/**
  Numerical version of the Pal integrals (P2), used in cases where
  the analytic expression is numerically unstable.

*/
template <typename T>
inline T P2_numerical(const T &bo, const T &ro, const Vector<T> &kappa,
                      Quad<typename T::Scalar> &QUAD) {

  using Scalar = typename T::Scalar;
  size_t K = kappa.size();

  std::function<Scalar(Scalar)> f = [bo, ro](Scalar phi) {
    return P2_integrand(bo.value(), ro.value(), phi);
  };
  std::function<Scalar(Scalar)> dfdbo = [bo, ro](Scalar phi) {
    return dP2dbo_integrand(bo.value(), ro.value(), phi);
  };
  std::function<Scalar(Scalar)> dfdro = [bo, ro](Scalar phi) {
    return dP2dro_integrand(bo.value(), ro.value(), phi);
  };

  // Compute the function value
  T res = 0.0;
  for (size_t i = 0; i < K; i += 2)
    res.value() += QUAD.integrate(kappa(i).value() - pi<Scalar>(),
                                  kappa(i + 1).value() - pi<Scalar>(), f);

  // Compute the derivatives.
  // Deriv wrt kappa is easy; need to integrate for the other two
  for (size_t i = 0; i < K; i += 2) {
    res.derivatives() +=
        f(kappa(i + 1).value() - pi<Scalar>()) * kappa(i + 1).derivatives();
    res.derivatives() -=
        f(kappa(i).value() - pi<Scalar>()) * kappa(i).derivatives();
    res.derivatives() +=
        bo.derivatives() *
        QUAD.integrate(kappa(i).value() - pi<Scalar>(),
                       kappa(i + 1).value() - pi<Scalar>(), dfdbo);
    res.derivatives() +=
        ro.derivatives() *
        QUAD.integrate(kappa(i).value() - pi<Scalar>(),
                       kappa(i + 1).value() - pi<Scalar>(), dfdro);
  }

  return res;
}

/**
  The P2 term from Pal (2012). This is the definite primitive integral of the
  linear limb darkening term.

*/
template <typename T>
inline T P2(const T &bo, const T &ro, const T &k2, const Vector<T> &kappa,
            const Vector<T> &s1, const Vector<T> &s2, const Vector<T> &c1,
            const T &F, const T &E, const T &PIp,
            Quad<typename T::Scalar> &QUAD) {

#if (!STARRY_USE_INCOMPLETE_INTEGRALS)

  // Compute this integral numerically
  return P2_numerical(bo, ro, kappa, QUAD);

#else

  // Use the formula from Pal (2012). Not significantly
  // faster and not numerically stable in general.

  // Useful variables
  size_t K = kappa.size();
  T r2 = ro * ro;
  T b2 = bo * bo;
  T br = bo * ro;
  T bpr = bo + ro;
  T bmr = bo - ro;
  T d2 = r2 + b2 - 2 * br;
  T term = 0.5 / sqrt(br * k2);
  T p0 = 4.0 - 7.0 * r2 - b2;
  Vector<T> q2(K);
  q2.array() = r2 + b2 - 2 * br * (1 - 2 * s2.array());

  // Special cases
  if (bo == 0.0) {

    // Analytic limit
    if (ro < 1.0)
      return (1 - (1 - r2) * sqrt(1 - r2)) * pairdiff(kappa) / 3.0;
    else
      return pairdiff(kappa) / 3.0;

  } else if (abs(bo - ro) < STARRY_PAL_BO_EQUALS_RO_TOL) {

    // Solve numerically
    return P2_numerical(bo, ro, kappa, QUAD);

  } else if (abs(bo - (ro - 1)) < STARRY_PAL_BO_EQUALS_RO_MINUS_ONE_TOL) {

    // Solve numerically
    return P2_numerical(bo, ro, kappa, QUAD);

  } else if (abs(bo - (1 - ro)) < STARRY_PAL_BO_EQUALS_ONE_MINUS_RO_TOL) {

    // Solve numerically
    return P2_numerical(bo, ro, kappa, QUAD);
  }

  // Constant term
  T A = 0.0;
  int sgn = -1;
  for (size_t i = 0; i < K; ++i) {
    A -= sgn * atan2(-bmr * c1(i), bpr * s1(i));
    if (kappa(i) > 3 * pi<T>()) {
      if (bmr > 0)
        A -= sgn * 2 * pi<T>();
      else if (bmr < 0)
        A += sgn * 2 * pi<T>();
    }
    A += 0.5 * sgn * kappa(i);
    if (q2(i) < 1.0)
      A -= 2 * (2.0 / 3.0) * br * sgn * (s1(i) * c1(i) * sqrt(1 - q2(i)));
    sgn *= -1;
  }

  // Carlson RD term
  T C = -2 * (2.0 / 3.0) * br * p0 * term * k2;

  // Carlson RF term
  T fac = -bpr / bmr;
  T B = -((((1.0 + 2.0 * r2 * r2 - 4.0 * r2) +
            (2.0 / 3.0) * br * (p0 + 5 * br) + fac) *
           term) +
          C);

  // Carlson PIprime term
  T D = -(2.0 / 3.0) * fac / d2 * term * br;

  return (A + B * F + C * E + D * PIp) / 3.0;

#endif
}

/**
  Integrand of the J_N term, for numerical integration.

*/
template <typename T>
inline T J_integrand(const int N, const T &k2, const T &phi) {
  T s2 = sin(phi);
  s2 *= s2;
  T term = 1 - s2 / k2;
  if (term < 0)
    term = 0;
  return pow(s2, N) * term * sqrt(term);
}

/**
  Derivative of the integrand of the J_N term, for numerical integration.
*/
template <typename T>
inline T dJdk2_integrand(const int N, const T &k2, const T &phi) {
  T s2 = sin(phi);
  s2 *= s2;
  T term = 1 - s2 / k2;
  if (term < 0)
    term = 0;
  return (1.5 / (k2 * k2)) * pow(s2, N + 1) * sqrt(term);
}

/**
  J is analytic from recursion relations, but gets unstable for high `n`. We
  evaluate J for `n = 0` (analytically) `n = nmax` (numerically) and solve the
  problem with a forward & backward pass to improve numerical stability.

*/
template <typename T>
inline T J_numerical(const int N, const T &k2, const Vector<T> &kappa,
                     Quad<typename T::Scalar> &QUAD) {

  using Scalar = typename T::Scalar;
  size_t K = kappa.size();

  std::function<Scalar(Scalar)> f = [N, k2](Scalar phi) {
    return J_integrand(N, k2.value(), phi);
  };
  std::function<Scalar(Scalar)> dfdk2 = [N, k2](Scalar phi) {
    return dJdk2_integrand(N, k2.value(), phi);
  };

  // Compute the function value
  T res = 0.0;
  for (size_t i = 0; i < K; i += 2)
    res.value() +=
        QUAD.integrate(0.5 * kappa(i).value(), 0.5 * kappa(i + 1).value(), f);

  // Compute the derivatives.
  // Deriv wrt kappa is easy; need to integrate for k2
  for (size_t i = 0; i < K; i += 2) {
    res.derivatives() +=
        0.5 * f(0.5 * kappa(i + 1).value()) * kappa(i + 1).derivatives();
    res.derivatives() -=
        0.5 * f(0.5 * kappa(i).value()) * kappa(i).derivatives();
    res.derivatives() += k2.derivatives() *
                         QUAD.integrate(0.5 * kappa(i).value(),
                                        0.5 * kappa(i + 1).value(), dfdk2);
  }

  return res;
}

/**
  The Gauss hypergeometric function 2F1.

*/
template <typename T>
inline T hyp2f1(const T &a_, const T &b_, const T &c_, const T &z) {

  // Compute the value
  T a = a_;
  T b = b_;
  T c = c_;
  T term = a * b * z / c;
  T value = 1.0 + term;
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
                          "reflected/special.h", "hyp2f1", args.str());
  }
  return value;
}

/**
  The Gauss hypergeometric function 2F1 and its z derivative.

*/
template <typename T, int N>
inline ADScalar<T, N> hyp2f1(const T &a, const T &b, const T &c,
                             const ADScalar<T, N> &z) {
  ADScalar<T, N> F;
  F.value() = hyp2f1(a, b, c, z.value());
  F.derivatives() =
      z.derivatives() * a * b / c * hyp2f1(a + 1, b + 1, c + 1, z.value());
  return F;
}

/**
  Integrand of the I_N term, for numerical integration.

*/
template <typename T> inline T I_integrand(const int N, const T &phi) {
  T s2 = sin(phi);
  s2 *= s2;
  return pow(s2, N);
}

/**
  The I helper integral evaluated numerically. The expression for the terms in
  I is analytic from recursion relations, but gets unstable for high `n`. We
  evaluate I for `n = 0` (analytically) `n = nmax` (numerically) and solve the
  problem with a forward & backward pass to improve numerical stability.

*/
template <typename T>
inline T I_numerical(const int N, const Vector<T> &kappa,
                     Quad<typename T::Scalar> &QUAD) {

  using Scalar = typename T::Scalar;
  size_t K = kappa.size();

  std::function<Scalar(Scalar)> f = [N](Scalar phi) {
    return I_integrand(N, phi);
  };

  // Compute the function value
  T res = 0.0;
  for (size_t i = 0; i < K; i += 2)
    res.value() +=
        QUAD.integrate(0.5 * kappa(i).value(), 0.5 * kappa(i + 1).value(), f);

  // Compute the derivatives (easy)
  for (size_t i = 0; i < K; i += 2) {
    res.derivatives() +=
        0.5 * f(0.5 * kappa(i + 1).value()) * kappa(i + 1).derivatives();
    res.derivatives() -=
        0.5 * f(0.5 * kappa(i).value()) * kappa(i).derivatives();
  }

  return res;
}

} // namespace special
} // namespace reflected
} // namespace starry

#endif
