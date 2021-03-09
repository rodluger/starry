/**
\file ellip.h
\brief Incomplete elliptic integral computation.

Elliptic integrals computed following

    Bulirsch 1965, Numerische Mathematik, 7, 78
    Bulirsch 1965, Numerische Mathematik, 7, 353


*/

#ifndef _STARRY_OBLATE_ELLIP_H_
#define _STARRY_OBLATE_ELLIP_H_

#include "../ellip.h"
#include "../utils.h"
#include "constants.h"
#include <iostream>

namespace starry {

using ellip::CEL;

namespace oblate {
namespace ellip {

using std::abs;
using namespace utils;

/**
  Autodiff-safe arc tangent.
*/
template <typename T> inline T arctan(const T &x) { return atan(x); }

/**
  Autodiff-safe arc tangent.
*/
template <typename T, int N>
inline ADScalar<T, N> arctan(const ADScalar<T, N> &x) {
  ADScalar<T, N> result;
  result.value() = atan(x.value());
  result.derivatives() = x.derivatives() / (x.value() * x.value() + 1);
  return result;
}

/**
  Autodiff-safe hyperbolic arc cosine.
*/
template <typename T> inline T arccosh(const T &x) { return acosh(x); }

/**
  Autodiff-safe hyperbolic arc cosine.
*/
template <typename T, int N>
inline ADScalar<T, N> arccosh(const ADScalar<T, N> &x) {
  ADScalar<T, N> result;
  result.value() = acosh(x.value());
  result.derivatives() = x.derivatives() / sqrt(x.value() * x.value() - 1);
  return result;
}

/**
  Vectorized implementation of the `el2` function from
  Bulirsch (1965). In this case, `x` is a *vector* of integration
  limits. The halting condition does not depend on the value of `x`,
  so it's much faster to evaluate all values of `x` at once!

*/
template <typename T>
inline Pair<T> el2(const Pair<T> &x_, const T &kc_, const T &a_, const T &b_) {

  // Make copies
  T kc = kc_;
  T a = a_;
  T b = b_;

  if (kc == 0) {
    std::stringstream args;
    args << "x_ = " << x_ << ", "
         << "kc_ = " << kc_ << ", "
         << "a_ = " << a_ << ", "
         << "b_ = " << b_;
    throw StarryException(
        "Elliptic integral el2 did not converge because k = 1.",
        "oblate/ellip.h", "el2", args.str());
  }

  // We declare these params as vectors,
  // but operate on them as arrays (because Eigen...)
  Pair<T> c_, d_, p_, y_, f_, l_, g_, q_;
  f_ = x_ * 0;
  l_ = x_ * 0;
  auto x = x_.array();
  auto c = c_.array();
  auto d = d_.array();
  auto p = p_.array();
  auto y = y_.array();
  auto f = f_.array();
  auto l = l_.array();
  auto g = g_.array();
  auto q = q_.array();

  // Scalars
  T z, i, m, e, gp;
  int n;

  // Initial conditions
  c = x * x;
  d = c + 1.0;
  p = sqrt((1.0 + kc * kc * c) / d);
  d = x / d;
  c = d / (2 * p);
  z = a - b;
  i = a;
  a = (b + a) / 2;
  y = abs(1.0 / x);
  m = 1.0;

  // Iterate until convergence
  for (n = 0; i < STARRY_EL2_MAX_ITER; ++n) {

    b = i * kc + b;
    e = m * kc;
    g = e / p;
    d = f * g + d;
    f = c;
    i = a;
    p = g + p;
    c = (d / p + c) / 2;
    gp = m;
    m = kc + m;
    a = (b / m + a) / 2;
    y = -e / y + y;
    y = (y == 0).select(sqrt(e) * c * b, y);

    if (abs(gp - kc) > STARRY_EL2_CA * gp) {

      kc = sqrt(e) * 2;
      l = l * 2;
      l = (y < 0).select(1.0 + l, l);

    } else {

      break;
    }
  }

  // Check for convergence
  if (n == STARRY_EL2_MAX_ITER - 1) {
    std::stringstream args;
    args << "x_ = " << x_ << ", "
         << "kc_ = " << kc_ << ", "
         << "a_ = " << a_ << ", "
         << "b_ = " << b_;
    throw StarryException("Elliptic integral el2 did not converge.",
                          "oblate/ellip.h", "el2", args.str());
  }

  l = (y < 0).select(1.0 + l, l);
  q = (atan(m / y) + pi<T>() * l) * a / m;
  q = (x < 0).select(-q, q);
  return (q + c * z).matrix();
}

template <class T> class IncompleteEllipticIntegrals {

  // Autodiff wrt {b, theta, bo, ro}
  using A = ADScalar<T, 4>;

protected:
  // Inputs
  A k2;
  Pair<A> kappa;

  // Helper vars
  A k2inv;
  A k;
  A kinv;
  A kc;
  A kc2inv;

  // Complete elliptic integrals
  A F0;
  A E0;

  // Vectorized output
  Pair<A> Fv;
  Pair<A> Ev;

  /**

  */
  inline void compute_el2(const Pair<A> &tanphi_, const A &m_) {

    // Get the values
    Pair<T> tanphi;
    tanphi(0) = tanphi_(0).value();
    tanphi(1) = tanphi_(1).value();
    T m = m_.value();
    T mc = 1 - m;

    // Compute the elliptic integrals
    Fv = el2(tanphi, sqrt(1 - m), 1.0, 1.0);
    Ev = el2(tanphi, sqrt(1 - m), 1.0, 1 - m);

    // Compute their derivatives
    T p2, q2, t2, ddtanphi, ddm;
    for (size_t i = 0; i < 2; ++i) {
      t2 = tanphi(i) * tanphi(i);
      p2 = 1.0 / (1.0 + t2);
      q2 = p2 * t2;
      ddtanphi = p2 / sqrt(1.0 - m * q2);
      ddm = 0.5 * (Ev(i).value() / (m * mc) - Fv(i).value() / m -
                   tanphi(i) * ddtanphi / mc);
      Fv(i).derivatives() =
          ddtanphi * tanphi_(i).derivatives() + ddm * m_.derivatives();
      ddtanphi = p2 * sqrt(1.0 - m * q2);
      ddm = 0.5 * (Ev(i).value() - Fv(i).value()) / m;
      Ev(i).derivatives() =
          ddtanphi * tanphi_(i).derivatives() + ddm * m_.derivatives();
    }
  }

  /**
    Compute the incomplete elliptic integrals of the first and second kinds.

  */
  inline void compute_E() {

    E = 0.0;

    if ((k2 < 1) && (k2 >= 0)) {

      // Analytic continuation from (17.4.15-16) in Abramowitz & Stegun
      // A better format is here: https://dlmf.nist.gov/19.7#ii

      // Helper variables
      Pair<A> arg, arg2, tanphi;
      arg.array() = kinv * sin(0.5 * kappa.array());
      arg2.array() = 1.0 - arg.array() * arg.array();
      tanphi.array() =
          (arg.array() >= 1.0)
              .select(STARRY_HUGE_TAN,
                      (arg.array() <= -1.0)
                          .select(-STARRY_HUGE_TAN,
                                  arg.array() * pow(arg2.array(), -0.5)));

      // Compute the incomplete elliptic integrals
      compute_el2(tanphi, k2);
      Fv.array() *= k;
      Ev.array() = kinv * (Ev.array() - (1 - k2) * kinv * Fv.array());

      // Compute the *definite* integrals
      // Add offsets to account for the limited domain of `el2`
      int sgn = -1;
      for (size_t i = 0; i < 2; ++i) {
        if (kappa(i) > -pi<T>()) {
          E += sgn * Ev(i);
        } else {
          E += sgn * (-2 * E0 - Ev(i));
        }
        sgn *= -1;
      }

    } else {

      // Helper variables
      Pair<A> tanphi;
      tanphi.array() = tan(0.5 * kappa.array());

      // Compute the incomplete elliptic integrals
      compute_el2(tanphi, k2inv);

      // Compute the *definite* integrals
      // Add offsets to account for the limited domain of `el2`
      int sgn = -1;
      for (size_t i = 0; i < 2; ++i) {
        if (kappa(i) > -pi<T>()) {
          E += sgn * Ev(i);
        } else {
          E += sgn * (-2 * E0 + Ev(i));
        }
        sgn *= -1;
      }
    }
  }

public:
  // Outputs
  A E;

  //! Constructor
  explicit IncompleteEllipticIntegrals(const A &ksq, const Pair<A> &phi) {

    // Domain checks. This class is only verified for -3pi/2 <= phi <= pi/2.
    if ((phi(0) < -1.5 * pi<T>()) || (phi(0) > 0.5 * pi<T>()) ||
        (phi(1) < -1.5 * pi<T>()) || (phi(1) > 0.5 * pi<T>())) {
      std::stringstream args;
      args << "phi0 = " << phi(0).value() << ", "
           << "phi1 = " << phi(1).value();
      throw StarryException(
          "Angle out of bounds for incomplete elliptic integral E.",
          "oblate/ellip.h", "IncompleteEllipticIntegrals", args.str());
    }

    // Note that internally we compute things using
    // the algorithm for the reciprocal for legacy reasons...
    k2 = 1.0 / ksq;
    k2inv = ksq;
    kappa = 2.0 * phi;
    k = sqrt(k2);
    kinv = 1.0 / k;
    kc = sqrt(1 - k2);

    // Nudge k2 away from 1 for stability
    if (abs(1 - k2.value()) < STARRY_K2_ONE_TOL) {
      if (k2 == 1.0) {
        k2 = 1 + STARRY_K2_ONE_TOL;
        k2inv = 1 - STARRY_K2_ONE_TOL;
      } else if (k2 < 1.0) {
        k2 = 1 - STARRY_K2_ONE_TOL;
        k2inv = 1 + STARRY_K2_ONE_TOL;
      } else {
        k2 = 1 + STARRY_K2_ONE_TOL;
        k2inv = 1 - STARRY_K2_ONE_TOL;
      }
    }

    // Complete elliptic integrals
    if ((k2.value() < 1) && (k2.value() >= 0)) {

      // Values
      F0.value() = k.value() * CEL(k2.value(), 1.0, 1.0, 1.0);
      E0.value() =
          kinv.value() * (CEL(k2.value(), 1.0, 1.0, 1.0 - k2.value()) -
                          (1.0 - k2.value()) * kinv.value() * F0.value());

      // Derivatives
      E0.derivatives() =
          0.5 / k2.value() * (E0.value() - F0.value()) * k2.derivatives();

    } else if (k2.value() >= 0) {

      // Values
      F0.value() = CEL(k2inv.value(), 1.0, 1.0, 1.0);
      E0.value() = CEL(k2inv.value(), 1.0, 1.0, 1.0 - k2inv.value());

      // Derivatives
      F0.derivatives() = 0.5 / k2inv.value() *
                         (E0.value() / (1 - k2inv.value()) - F0.value()) *
                         k2inv.derivatives();
      E0.derivatives() =
          0.5 / k2inv.value() * (E0.value() - F0.value()) * k2inv.derivatives();
    } else {

      // k2 is negative, so we should use the analytic continuation
      // trick from https://dlmf.nist.gov/19.7#E2 to compute E0 using
      // CEL and a positive argument. For now, it's easier to just
      // compute E0 using the incomplete elliptic integral algorithm,
      // which actually works fine for negative k2.
      // NOTE that F0 is not computed since we don't need it above.
      Pair<A> tanphi;
      tanphi(0) = 0;
      tanphi(1) = STARRY_HUGE_TAN;
      compute_el2(tanphi, k2inv);
      E0 = Ev(1) - Ev(0);
    }

    // Compute the incomplete elliptic integral of the second kind.
    // The expression
    //
    //      E(k2, phi)
    //
    // for vector `phi` is equivalent to the definite integral
    //
    //      ellipe(phi[1], k2) - ellipe(phi[0], k2)
    //
    // computed using (say) `mpmath.ellipe` in Python.
    //
    // Note that this function is specially coded to allow for
    // negative values of `k2` using analytic continuation;
    // these are required in the evaluation of the oblate
    // integrals.
    compute_E();
  }
};

} // namespace ellip
} // namespace oblate
} // namespace starry

#endif
