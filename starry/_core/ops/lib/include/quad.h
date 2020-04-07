/**
\file quad.h
\brief Gauss-Legendre quadrature.

Adapted from
https://rosettacode.org/wiki/Numerical_integration/Gauss-Legendre_Quadrature

*/

#ifndef _STARRY_QUAD_H_
#define _STARRY_QUAD_H_

namespace starry {
namespace quad {

using namespace utils;

/*! Implementation of Gauss-Legendre quadrature
*  http://en.wikipedia.org/wiki/Gaussian_quadrature
*  http://rosettacode.org/wiki/Numerical_integration/Gauss-Legendre_Quadrature
*
*/
template <typename T> class Quad {
public:
  enum { eDEGREE = STARRY_QUAD_POINTS };

  /*! Compute the integral of a functor
  *
  *   @param a    lower limit of integration
  *   @param b    upper limit of integration
  *   @param f    the function to integrate
  *   @param err  callback in case of problems
  */
  template <typename Function> inline T integrate(T a, T b, Function f) {
    T p = (b - a) / 2;
    T q = (b + a) / 2;
    const LegendrePolynomial &legpoly = s_LegendrePolynomial;

    T sum = 0;
    for (int i = 1; i <= eDEGREE; ++i) {
      sum += legpoly.weight(i) * f(p * legpoly.root(i) + q);
    }

    return p * sum;
  }

  /*! Print out roots and weights for information
  */
  void print_roots_and_weights(std::ostream &out) const {
    const LegendrePolynomial &legpoly = s_LegendrePolynomial;
    out << "Roots:  ";
    for (int i = 0; i <= eDEGREE; ++i) {
      out << ' ' << legpoly.root(i);
    }
    out << '\n';
    out << "Weights:";
    for (int i = 0; i <= eDEGREE; ++i) {
      out << ' ' << legpoly.weight(i);
    }
    out << '\n';
  }

private:
  /*! Implementation of the Legendre polynomials that form
  *   the basis of this quadrature
  */
  class LegendrePolynomial {
  public:
    LegendrePolynomial() {
      // Solve roots and weights
      for (int i = 0; i <= eDEGREE; ++i) {
        T dr = 1;

        // Find zero
        Evaluation eval(cos(M_PI * (i - 0.25) / (eDEGREE + 0.5)));
        do {
          dr = eval.v() / eval.d();
          eval.evaluate(eval.x() - dr);
        } while (abs(dr) > 2e-16);

        this->_r[i] = eval.x();
        this->_w[i] = 2 / ((1 - eval.x() * eval.x()) * eval.d() * eval.d());
      }
    }

    T root(int i) const { return this->_r[i]; }
    T weight(int i) const { return this->_w[i]; }

  private:
    T _r[eDEGREE + 1];
    T _w[eDEGREE + 1];

    /*! Evaluate the value *and* derivative of the
    *   Legendre polynomial
    */
    class Evaluation {
    public:
      explicit Evaluation(T x) : _x(x), _v(1), _d(0) { this->evaluate(x); }

      void evaluate(T x) {
        this->_x = x;

        T vsub1 = x;
        T vsub2 = 1;
        T f = 1 / (x * x - 1);

        for (int i = 2; i <= eDEGREE; ++i) {
          this->_v = ((2 * i - 1) * x * vsub1 - (i - 1) * vsub2) / i;
          this->_d = i * f * (x * this->_v - vsub1);

          vsub2 = vsub1;
          vsub1 = this->_v;
        }
      }

      T v() const { return this->_v; }
      T d() const { return this->_d; }
      T x() const { return this->_x; }

    private:
      T _x;
      T _v;
      T _d;
    };
  };

  /*! Pre-compute the weights and abscissae of the Legendre polynomials
  */
  static LegendrePolynomial s_LegendrePolynomial;
};

template <typename T>
typename Quad<T>::LegendrePolynomial Quad<T>::s_LegendrePolynomial;

} // namespace quad
} // namespace starry

#endif
