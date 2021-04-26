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

template <typename T>
using IntegralVector = Eigen::Matrix<T, STARRY_QUAD_POINTS, 1>;

template <typename T>
using IntegralRowVector = Eigen::Matrix<T, 1, STARRY_QUAD_POINTS>;

template <typename T>
using IntegralArray = Eigen::Array<T, STARRY_QUAD_POINTS, 1>;

template <typename T>
using IntegralMatrix = Eigen::Matrix<T, STARRY_QUAD_POINTS, Eigen::Dynamic>;

/*! Implementation of Gauss-Legendre quadrature based on
 *  http://en.wikipedia.org/wiki/Gaussian_quadrature
 *  http://rosettacode.org/wiki/Numerical_integration/Gauss-Legendre_Quadrature
 */
template <typename Scalar, int N> class Quad {

  using A = ADScalar<Scalar, N>;

private:
  A p;
  IntegralVector<A> q;
  IntegralRowVector<A> vT;

public:
  template <typename Function>
  inline Vector<A> integrate(const A &a, const A &b, Function f,
                             const int &nint) {
    const LegendrePolynomial &L = s_LegendrePolynomial;

    // Initialize
    IntegralMatrix<A> integrand;
    integrand.setZero(STARRY_QUAD_POINTS, nint);

    // Compute the function on the grid
    p = 0.5 * (b - a);
    vT = p * L.wT;
    q.setConstant(0.5 * (b + a));
    f(p * L.r + q, integrand);
    return vT * integrand;
  }

private:
  class LegendrePolynomial {
  public:
    IntegralVector<Scalar> r;
    IntegralRowVector<Scalar> wT;

    LegendrePolynomial() {
      for (int i = 0; i < STARRY_QUAD_POINTS; ++i) {
        Scalar dr = 1;
        Evaluation eval(cos(M_PI * (i + 0.75) / (STARRY_QUAD_POINTS + 0.5)));
        do {
          dr = eval.v() / eval.d();
          eval.evaluate(eval.x() - dr);
        } while (abs(dr) > 2e-16);
        this->r(i) = eval.x();
        this->wT(i) = 2 / ((1 - eval.x() * eval.x()) * eval.d() * eval.d());
      }
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
        for (int i = 2; i <= STARRY_QUAD_POINTS; ++i) {
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
inline void pTodd_integrand(const int &deg, const ADScalar<Scalar, N> &bo,
                            const ADScalar<Scalar, N> &ro,
                            const ADScalar<Scalar, N> &f,
                            const ADScalar<Scalar, N> &theta,
                            const ADScalar<Scalar, N> &costheta,
                            const ADScalar<Scalar, N> &sintheta,
                            const IntegralVector<ADScalar<Scalar, N>> &phi,
                            IntegralMatrix<ADScalar<Scalar, N>> &integrand) {

  using Array = IntegralArray<ADScalar<Scalar, N>>;

  // Pre-compute common terms
  int ncoeff = (deg + 1) * (deg + 1);
  Array cosphi = cos(phi.array());
  Array sinphi = sin(phi.array());
  Array cosvphi = costheta * cosphi + sintheta * sinphi;
  Array sinvphi = -sintheta * cosphi + costheta * sinphi;
  Array x = ro * cosvphi + bo * sintheta;
  Array y = (ro * sinvphi + bo * costheta) / (1.0 - f);
  Array z2 = 1.0 - x * x - y * y;
  Array z = (z2 <= 0).select(Array::Zero(), sqrt(z2));
  Array z3 = z2 * z;
  Array z3x = -ro * (1.0 - f) * sinvphi * z3;
  Array z3y = ro * cosvphi * z3;

  // Case 2
  integrand.col(2) = ro * (ro + bo * sinphi) * (1.0 - z3) / (3.0 * (1.0 - z2));

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
      const IntegralVector<ADScalar<Scalar, N>> &,
      IntegralMatrix<ADScalar<Scalar, N>> & //
      )>
      func =                                              //
      [deg, bo, ro, f, theta, costheta, sintheta]         //
      (                                                   //
          const IntegralVector<ADScalar<Scalar, N>> &phi, //
          IntegralMatrix<ADScalar<Scalar, N>> &integrand  //
      ) {
        pTodd_integrand(deg, bo, ro, f, theta, costheta, sintheta, phi,
                        integrand);
      };
  int ncoeff = (deg + 1) * (deg + 1);
  return QUAD.integrate(phi1, phi2, func, ncoeff);
}

} // namespace numerical
} // namespace oblate
} // namespace starry

#endif
