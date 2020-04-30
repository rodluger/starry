/**
\file ellip.h
\brief Incomplete elliptic integral computation.

Elliptic integrals computed following

    Bulirsch 1965, Numerische Mathematik, 7, 78
    Bulirsch 1965, Numerische Mathematik, 7, 353


*/

#ifndef _STARRY_REFLECTED_ELLIP_H_
#define _STARRY_REFLECTED_ELLIP_H_

#include "../ellip.h"
#include "../utils.h"
#include "constants.h"

namespace starry {

using ellip::CEL;

namespace reflected {
namespace ellip {

using std::abs;
using namespace utils;

template <class T> inline Vector<T> F(const Vector<T> &tanphi, const T &k2);
template <class T> inline Vector<T> E(const Vector<T> &tanphi, const T &k2);

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
inline Vector<T> el2(const Vector<T> &x_, const T &kc_, const T &a_,
                     const T &b_) {

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
        "reflected/ellip.h", "el2", args.str());
  }

  // We declare these params as vectors,
  // but operate on them as arrays (because Eigen...)
  Vector<T> c_, d_, p_, y_, f_, l_, g_, q_;
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
                          "reflected/ellip.h", "el2", args.str());
  }

  l = (y < 0).select(1.0 + l, l);
  q = (atan(m / y) + pi<T>() * l) * a / m;
  q = (x < 0).select(-q, q);
  return (q + c * z).matrix();
}

/**
  Scalar implementation of the Carlson elliptic integral RJ.

  Based on

      Bille Carlson,
      Computing Elliptic Integrals by Duplication,
      Numerische Mathematik,
      Volume 33, 1979, pages 1-16.

      Bille Carlson, Elaine Notis,
      Algorithm 577, Algorithms for Incomplete Elliptic Integrals,
      ACM Transactions on Mathematical Software,
      Volume 7, Number 3, pages 398-403, September 1981

      https://people.sc.fsu.edu/~jburkardt/f77_src/toms577/toms577.f

  NOTE: This function has pretty poor numerical stability. We should code
        up Bulirsch's `el3` instead.

*/
template <typename T>
inline T rj(const T &x_, const T &y_, const T &z_, const T &p_) {

  // Constants
  const T C1 = 3.0 / 14.0;
  const T C2 = 1.0 / 3.0;
  const T C3 = 3.0 / 22.0;
  const T C4 = 3.0 / 26.0;

  // Make copies
  T x = x_;
  T y = y_;
  T z = z_;
  T p = p_;

  // Limit checks
  if (x < STARRY_CRJ_LO_LIM)
    x = STARRY_CRJ_LO_LIM;
  else if (x > STARRY_CRJ_HI_LIM)
    x = STARRY_CRJ_HI_LIM;

  if (y < STARRY_CRJ_LO_LIM)
    y = STARRY_CRJ_LO_LIM;
  else if (y > STARRY_CRJ_HI_LIM)
    y = STARRY_CRJ_HI_LIM;

  if (z < STARRY_CRJ_LO_LIM)
    z = STARRY_CRJ_LO_LIM;
  else if (z > STARRY_CRJ_HI_LIM)
    z = STARRY_CRJ_HI_LIM;

  if (p < STARRY_CRJ_LO_LIM)
    p = STARRY_CRJ_LO_LIM;
  else if (p > STARRY_CRJ_HI_LIM)
    p = STARRY_CRJ_HI_LIM;

  T xn = x;
  T yn = y;
  T zn = z;
  T pn = p;
  T sigma = 0.0;
  T power4 = 1.0;

  T mu, invmu;
  T xndev, yndev, zndev, pndev;
  T eps;
  T ea, eb, ec, e2, e3, s1, s2, s3, value;
  T xnroot, ynroot, znroot;
  T lam, alpha, beta;

  for (int k = 0; k < STARRY_CRJ_MAX_ITER; ++k) {

    mu = 0.2 * (xn + yn + zn + pn + pn);
    invmu = 1.0 / mu;
    xndev = (mu - xn) * invmu;
    yndev = (mu - yn) * invmu;
    zndev = (mu - zn) * invmu;
    pndev = (mu - pn) * invmu;

    // Poor man's `max`
    eps = abs(xndev);
    if (abs(yndev) > eps)
      eps = abs(yndev);
    if (abs(zndev) > eps)
      eps = abs(zndev);
    if (abs(pndev) > eps)
      eps = abs(pndev);

    if (eps < STARRY_CRJ_TOL) {

      ea = xndev * (yndev + zndev) + yndev * zndev;
      eb = xndev * yndev * zndev;
      ec = pndev * pndev;
      e2 = ea - 3.0 * ec;
      e3 = eb + 2.0 * pndev * (ea - ec);
      s1 = 1.0 + e2 * (-C1 + 0.75 * C3 * e2 - 1.5 * C4 * e3);
      s2 = eb * (0.5 * C2 + pndev * (-C3 - C3 + pndev * C4));
      s3 = pndev * ea * (C2 - pndev * C3) - C2 * pndev * ec;
      value = 3.0 * sigma + power4 * (s1 + s2 + s3) / (mu * sqrt(mu));
      return value;
    }

    xnroot = sqrt(xn);
    ynroot = sqrt(yn);
    znroot = sqrt(zn);
    lam = xnroot * (ynroot + znroot) + ynroot * znroot;
    alpha = pn * (xnroot + ynroot + znroot) + xnroot * ynroot * znroot;
    alpha = alpha * alpha;
    beta = pn * (pn + lam) * (pn + lam);
    if (alpha < beta)
      sigma += power4 * acos(sqrt(alpha / beta)) / sqrt(beta - alpha);
    else if (alpha > beta)
      sigma += power4 * arccosh(T(sqrt(alpha / beta))) / sqrt(alpha - beta);
    else
      sigma = sigma + power4 / sqrt(beta);

    power4 *= 0.25;
    xn = 0.25 * (xn + lam);
    yn = 0.25 * (yn + lam);
    zn = 0.25 * (zn + lam);
    pn = 0.25 * (pn + lam);
  }

  // Bad...
  std::stringstream args;
  args << "x_ = " << x_ << ", "
       << "y_ = " << y_ << ", "
       << "z_ = " << z_ << ", "
       << "p_ = " << p_;
  throw StarryException("Elliptic integral rj did not converge.",
                        "reflected/ellip.h", "rj", args.str());
}

template <class T> class IncompleteEllipticIntegrals {

  // Autodiff wrt {b, theta, bo, ro}
  using A = ADScalar<T, 4>;

protected:
  // Inputs
  A bo;
  A ro;
  Vector<A> kappa;
  size_t K;

  // Helper vars
  A k2;
  A k2inv;
  A k;
  A kinv;
  A kc2;
  A kc;
  A kc2inv;
  A kcinv;
  A p0;
  Vector<A> p;

  Vector<A> phi;
  Vector<A> coskap;
  Vector<A> cosphi;
  Vector<A> sinphi;
  Vector<A> w;

  // Complete elliptic integrals
  A F0;
  A E0;
  A PIp0;

  // Vectorized output
  Vector<A> Fv;
  Vector<A> Ev;

  /**

  */
  inline void compute_el2(const Vector<A> &tanphi_, const A &m_) {

    // Get the values
    Vector<T> tanphi(K);
    for (size_t i = 0; i < K; ++i)
      tanphi(i) = tanphi_(i).value();
    T m = m_.value();
    T mc = 1 - m;

    // Compute the elliptic integrals
    Fv = el2(tanphi, sqrt(1 - m), 1.0, 1.0);
    Ev = el2(tanphi, sqrt(1 - m), 1.0, 1 - m);

    // Compute their derivatives
    T p2, q2, t2, ddtanphi, ddm;
    for (size_t i = 0; i < K; ++i) {
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
  inline void compute_FE() {

    F = 0.0;
    E = 0.0;

    if (k2 < 1) {

      // Analytic continuation from (17.4.15-16) in Abramowitz & Stegun
      // A better format is here: https://dlmf.nist.gov/19.7#ii

      // Helper variables
      Vector<A> arg(K), arg2(K), tanphi(K);
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
      for (size_t i = 0; i < K; ++i) {
        if (kappa(i) > 3 * pi<T>()) {
          F += sgn * (4 * F0 + Fv(i));
          E += sgn * (4 * E0 + Ev(i));
        } else if (kappa(i) > pi<T>()) {
          F += sgn * (2 * F0 - Fv(i));
          E += sgn * (2 * E0 - Ev(i));
        } else {
          F += sgn * Fv(i);
          E += sgn * Ev(i);
        }
        sgn *= -1;
      }

    } else {

      // Helper variables
      Vector<A> tanphi(K);
      tanphi.array() = tan(0.5 * kappa.array());

      // Compute the incomplete elliptic integrals
      compute_el2(tanphi, k2inv);

      // Compute the *definite* integrals
      // Add offsets to account for the limited domain of `el2`
      int sgn = -1;
      for (size_t i = 0; i < K; ++i) {
        if (kappa(i) > 3 * pi<T>()) {
          F += sgn * (4 * F0 + Fv(i));
          E += sgn * (4 * E0 + Ev(i));
        } else if (kappa(i) > pi<T>()) {
          F += sgn * (2 * F0 + Fv(i));
          E += sgn * (2 * E0 + Ev(i));
        } else {
          F += sgn * Fv(i);
          E += sgn * Ev(i);
        }
        sgn *= -1;
      }
    }
  }

  /**

    Modified incomplete elliptic integral of the third kind.

    This integral is proportional to the Carlson elliptic integral RJ:

        PI' = -2 sin^3(phi) * RJ(cos^2 phi, 1 - k^2 sin^2 phi, 1, 1 - n sin^2
    phi)

    where

        phi = kappa / 2
        n = -4 b r / (r - b)^2

    It can also be written in terms of the Legendre forms:

        PI' = 6 / n * (F(phi, 1/k^2) - PI(phi, n, 1/k^2))

    This integral is only used in the expression for computing the linear limb
    darkening
    term (2) in the primitive integral P, based on the expressions in Pal
    (2012).

    NOTE: We are currently autodiffing the Carlson RJ expression to get the
          derivative of this integral. The RJ function is already numerically
          unstable, so autodiffing it is a **terrible** idea.

  */
  inline void compute_PIp() {

    // Stability hack
    if (fabs(bo.value() - ro.value()) < STARRY_PAL_BO_EQUALS_RO_TOL) {
      PIp = 0.0;
      return;
    }

    // Helper variables
    A val;

    // Compute the integrals
    int sgn = -1;
    PIp = 0.0;
    for (size_t i = 0; i < K; ++i) {

      // Compute the integral, valid for -pi < kappa < pi
      if (w(i) < 0)
        w(i) = 0.0;
      val = (1.0 - coskap(i)) * cosphi(i) *
            rj(w(i), A(sinphi(i) * sinphi(i)), A(1.0), p(i));

      // Add offsets to account for the limited domain of `rj`
      if (kappa(i) > 3 * pi<T>()) {
        PIp += sgn * (2 * PIp0 + val);
      } else if (kappa(i) > pi<T>()) {
        PIp += sgn * (PIp0 + val);
      } else {
        PIp += sgn * val;
      }

      sgn *= -1;
    }
  }

public:
  // Outputs
  A F;
  A E;
  A PIp;

  //! Constructor
  explicit IncompleteEllipticIntegrals(const A &bo, const A &ro,
                                       const Vector<A> &kappa)
      : bo(bo), ro(ro), kappa(kappa), K(kappa.size()), p(K), phi(K), coskap(K),
        cosphi(K), sinphi(K), w(K) {

    // Helper vars
    phi.array() = 0.5 * (kappa.array() - pi<T>());
    for (size_t i = 0; i < K; ++i) {
      phi(i) = angle(phi(i), A(pi<T>()));
    }
    coskap.array() = cos(kappa.array());
    cosphi.array() = cos(phi.array());
    sinphi.array() = sin(phi.array());
    k2 = (1 - ro * ro - bo * bo + 2 * bo * ro) / (4 * bo * ro);
    k2inv = 1.0 / k2;
    k = sqrt(k2);
    kinv = 1.0 / k;
    kc2 = 1 - k2;
    kc = sqrt(kc2);
    kc2inv = 1 - k2inv;
    kcinv = sqrt(kc2inv);
    p0 = (ro * ro + bo * bo + 2 * ro * bo) / (ro * ro + bo * bo - 2 * ro * bo);
    p.array() = (ro * ro + bo * bo - 2 * ro * bo * coskap.array()) /
                (ro * ro + bo * bo - 2 * ro * bo);
    w.array() = 1.0 - cosphi.array() * cosphi.array() / k2;

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
    if (k2.value() < 1) {

      // Values
      F0.value() = k.value() * CEL(k2.value(), 1.0, 1.0, 1.0);
      E0.value() =
          kinv.value() * (CEL(k2.value(), 1.0, 1.0, 1.0 - k2.value()) -
                          (1.0 - k2.value()) * kinv.value() * F0.value());

      // Derivatives
      F0.derivatives() = 0.5 / k2.value() *
                         (E0.value() / (1 - k2.value()) - F0.value()) *
                         k2.derivatives();
      E0.derivatives() =
          0.5 / k2.value() * (E0.value() - F0.value()) * k2.derivatives();

      // Third kind
      // NOTE: I don't think this offset term is needed when k2 < 1.
      // PIp0 only comes into play if successive terms in kappa span either
      // side of the discontinuities at kappa = pi and kappa = 3 pi
      // (otherwise, the PIp0 terms cancel).
      // In any event, our expression for P2 isn't even valid in these cases, as
      // it disagrees with numerical integration. We should investigate this,
      // but I suspect we can just set PIp0 to zero here and skip all the
      // overhead with no adverse effects.
      // if (bo != ro)
      //    PIp0 = -4 * k2 * k * rj(0.0, 1 - k2, 1.0, 1.0 / ((ro - bo) * (ro -
      //    bo)));
      // else
      //    PIp0 = 0.0;
      PIp0 = 0.0;

    } else {

      // Values
      F0.value() = CEL(k2inv.value(), 1.0, 1.0, 1.0);
      E0.value() = CEL(k2inv.value(), 1.0, 1.0, 1.0 - k2inv.value());

      // Derivatives
      F0.derivatives() = 0.5 / k2inv.value() *
                         (E0.value() / (1 - k2inv.value()) - F0.value()) *
                         k2inv.derivatives();
      E0.derivatives() =
          0.5 / k2inv.value() * (E0.value() - F0.value()) * k2inv.derivatives();

      // Third kind
      if ((bo != 0) && (bo != ro)) {
        A n0 = 1.0 - p0; // = -4 * bo * ro / ((bo - ro) * (bo - ro));
        A PI0;
        PI0.value() = CEL(k2inv.value(), p0.value(), 1.0, 1.0);
        T dPI0dkinv = kinv.value() / (n0.value() - k2inv.value()) *
                      (E0.value() / (k2inv.value() - 1.0) + PI0.value());
        T dPI0dn0 = 1.0 /
                    (2.0 * (k2inv.value() - n0.value()) * (n0.value() - 1.0)) *
                    (E0.value() +
                     (k2inv.value() - n0.value()) / n0.value() * F0.value() +
                     (n0.value() * n0.value() - k2inv.value()) / n0.value() *
                         PI0.value());
        PI0.derivatives() =
            dPI0dkinv * kinv.derivatives() + dPI0dn0 * n0.derivatives();
        PIp0 = 2 * 6.0 / n0 * (F0 - PI0);
      } else {
        PIp0 = 0.0;
      }
    }

    // First and second kinds
    compute_FE();

    // Third kind
    compute_PIp();
  }
};

} // namespace ellip
} // namespace reflected
} // namespace starry

#endif
