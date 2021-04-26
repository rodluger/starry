/**
\file special.h
\brief Numerical integration.

*/

#ifndef _STARRY_OBLATE_NUMERICAL_H_
#define _STARRY_OBLATE_NUMERICAL_H_

#include "../utils.h"

namespace starry {
namespace oblate {
namespace numerical {

using std::abs;
using namespace utils;

template <typename Scalar>
using IntegralVector = Eigen::Matrix<Scalar, STARRY_QUAD_POINTS, 1>;

template <typename Scalar>
using IntegralRowVector = Eigen::Matrix<Scalar, 1, STARRY_QUAD_POINTS>;

template <typename Scalar>
using IntegralArray = Eigen::Array<Scalar, STARRY_QUAD_POINTS, 1>;

template <typename Scalar>
using IntegralMatrix =
    Eigen::Matrix<Scalar, STARRY_QUAD_POINTS, Eigen::Dynamic>;

template <typename Scalar, int N>
using IntegralDerivMatrix = Eigen::Matrix<IntegralMatrix<Scalar>, N, 1>;

/*! Implementation of Gauss-Legendre quadrature based on
 *  http://en.wikipedia.org/wiki/Gaussian_quadrature
 *  http://rosettacode.org/wiki/Numerical_integration/Gauss-Legendre_Quadrature
 */
template <typename Scalar, int N> class Quad {

  using A = ADScalar<Scalar, N>;

private:
  Scalar p;
  IntegralVector<Scalar> q;
  IntegralRowVector<Scalar> vT;
  Vector<A> res;

public:
  template <typename Function>
  inline Vector<A> integrate(const A &a, const A &b, Function f,
                             const int &nint, const Vector<A> &args) {
    const LegendrePolynomial &L = s_LegendrePolynomial;

    // Initialize
    IntegralMatrix<Scalar> integrand;
    IntegralDerivMatrix<Scalar, N> derivs;
    integrand.setZero(STARRY_QUAD_POINTS, nint);
    for (int i = 0; i < N; ++i) {
      derivs(i).setZero(STARRY_QUAD_POINTS, nint);
    }
    res.resize(nint);

    // Compute the function & its derivs on the grid
    p = 0.5 * (b.value() - a.value());
    vT = p * L.wT;
    q.setConstant(0.5 * (b.value() + a.value()));
    f(p * L.r + q, integrand, derivs);

    if (N == 0) {

      // Evaluate the integral directly into the ADScalar
      for (int n = 0; n < nint; ++n) {
        res(n).value() = vT.dot(integrand.col(n));
      }

    } else {

      // Evaluate the integral directly into the ADScalar
      for (int n = 0; n < nint; ++n) {
        res(n).value() = vT.dot(integrand.col(n));
        res(n).derivatives() =
            integrand(STARRY_QUAD_POINTS - 1, n) * b.derivatives() -
            integrand(0, n) * a.derivatives();
        for (int i = 0; i < N; ++i) {
          res(n).derivatives() +=
              vT.dot(derivs(i).col(n)) * args(i).derivatives();
        }
      }
    }

    return res;
  }

private:
  class LegendrePolynomial {
  public:
    IntegralVector<Scalar> r;
    IntegralRowVector<Scalar> wT;

    LegendrePolynomial() {
      // Note that we pad the endpoints so that we always
      // evaluate the function exactly at the limits of
      // integration (needed when propagating derivs)
      for (int i = 0; i < STARRY_QUAD_POINTS - 2; ++i) {
        Scalar dr = 1;
        Evaluation eval(cos(M_PI * (i + 0.75) / (STARRY_QUAD_POINTS - 1.5)));
        do {
          dr = eval.v() / eval.d();
          eval.evaluate(eval.x() - dr);
        } while (abs(dr) > 2e-16);
        this->r(i + 1) = eval.x();
        this->wT(i + 1) = 2 / ((1 - eval.x() * eval.x()) * eval.d() * eval.d());
      }
      this->r(0) = -1.0;
      this->r(STARRY_QUAD_POINTS - 1) = 1.0;
      this->wT(0) = 0.0;
      this->wT(STARRY_QUAD_POINTS - 1) = 0.0;
    }

    Scalar root(int i) const { return this->r(i); }
    Scalar weight(int i) const { return this->wT(i); }

  private:
    class Evaluation {
    public:
      explicit Evaluation(Scalar x) : _x(x), _v(1), _d(0) { this->evaluate(x); }

      void evaluate(Scalar x) {
        this->_x = x;
        Scalar vsub1 = x;
        Scalar vsub2 = 1;
        Scalar f = 1 / (x * x - 1);
        for (int i = 2; i <= STARRY_QUAD_POINTS - 2; ++i) {
          this->_v = ((2 * i - 1) * x * vsub1 - (i - 1) * vsub2) / i;
          this->_d = i * f * (x * this->_v - vsub1);
          vsub2 = vsub1;
          vsub1 = this->_v;
        }
      }

      Scalar v() const { return this->_v; }
      Scalar d() const { return this->_d; }
      Scalar x() const { return this->_x; }

    private:
      Scalar _x;
      Scalar _v;
      Scalar _d;
    };
  };

  static LegendrePolynomial s_LegendrePolynomial;
};

template <typename Scalar, int N>
typename Quad<Scalar, N>::LegendrePolynomial
    Quad<Scalar, N>::s_LegendrePolynomial;

template <typename Scalar, int N>
inline void pTodd_integrand(const int &deg, const Scalar &bo, const Scalar &ro,
                            const Scalar &f, const Scalar &theta,
                            const Scalar &costheta, const Scalar &sintheta,
                            const IntegralVector<Scalar> &phi,
                            IntegralMatrix<Scalar> &integrand,
                            IntegralDerivMatrix<Scalar, N> &derivs) {

  using Array = IntegralArray<Scalar>;

  // Pre-compute common terms
  int ncoeff = (deg + 1) * (deg + 1);
  Scalar invb = 1.0 / (1.0 - f);
  Array cosphi = cos(phi.array());
  Array sinphi = sin(phi.array());
  Array cosvphi = costheta * cosphi + sintheta * sinphi;
  Array sinvphi = -sintheta * cosphi + costheta * sinphi;
  Array x = ro * cosvphi + bo * sintheta;
  Array y = (ro * sinvphi + bo * costheta) * invb;
  Array z2 = 1.0 - x * x - y * y;
  Array z = (z2 < 0).select(Array::Zero(), sqrt(z2));
  Array z3 = z2 * z;
  Array z3x = -ro * (1.0 - f) * sinvphi * z3;
  Array z3y = ro * cosvphi * z3;

  Array p, q, r, s, t, u, v;

  // Case 2
  p = ro + bo * sinphi;
  integrand.col(2) = ro * p * (1.0 - z3) / (3.0 * (1.0 - z2));
  if (N > 0) {

    q = ro * ((y * costheta) * invb + x * sintheta) * p;
    r = ro * (x * cosvphi + (y * sinvphi) * invb) * p;
    s = (1 + z + z2) * (1 + z);
    t = ro * y * y * p;
    u = ro * x * y * p * (2 - f) * f * invb;
    v = 1.0 / (3 * (1 + z) * (1 + z));

    derivs(0).col(2) = (ro * s * sinphi - q * (2 + z)) * v;
    derivs(1).col(2) = (2 * ro * s + bo * s * sinphi - r * (2 + z)) * v;
    derivs(2).col(2) = -t * invb * (2 + z) * v;
    derivs(3).col(2) = u * (2 + z) * v;
  }

  // Cases 3-5
  Array xi, yj;
  int n;
  xi.setOnes();
  for (int i = 0; i < deg - 1; ++i) {
    if (is_even(i)) {
      // Case 3
      n = i * i + 6 * i + 7;
      if (n < ncoeff) {
        integrand.col(n) = xi * z3x;
      }
      // Case 4
      n += 2 * i + 7;
      if (n < ncoeff) {
        integrand.col(n) = xi * y * z3x;
      }
    }
    // Case 5
    yj.setOnes();
    for (int j = 0; j < deg - 1 - i; ++j) {
      n = (i + j) * (i + j) + 4 * i + 6 * j + 5;
      integrand.col(n) = xi * yj * z3y;
      yj = yj * y;
    }
    xi = xi * x;
  }
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
      const ADScalar<Scalar, N> &phi2, Quad<Scalar, N> &QUAD) {
  std::function<void( //
      const IntegralVector<Scalar> &,
      IntegralMatrix<Scalar> &,        //
      IntegralDerivMatrix<Scalar, N> & //
      )>
      func =                                      //
      [deg, bo, ro, f, theta, costheta, sintheta] //
      (                                           //
          const IntegralVector<Scalar> &phi,      //
          IntegralMatrix<Scalar> &integrand,      //
          IntegralDerivMatrix<Scalar, N> &derivs  //
      ) {
        pTodd_integrand(deg, bo.value(), ro.value(), f.value(), theta.value(),
                        costheta.value(), sintheta.value(), phi, integrand,
                        derivs);
      };
  int ncoeff = (deg + 1) * (deg + 1);
  Vector<ADScalar<Scalar, N>> args(4);
  args << bo, ro, f, theta;
  return QUAD.integrate(phi1, phi2, func, ncoeff, args);
}

} // namespace numerical
} // namespace oblate
} // namespace starry

#endif
