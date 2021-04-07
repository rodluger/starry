/**
\file geometry.h
\brief Circle-ellipse intersection stuff.

*/

#ifndef _STARRY_OBLATE_GEOMETRY_H_
#define _STARRY_OBLATE_GEOMETRY_H_

#include "../utils.h"
#include <Eigen/Eigenvalues>

namespace starry {
namespace oblate {
namespace geometry {

using namespace utils;

/**
    Polynomial root finder using an eigensolver.
    `coeffs` is a vector of coefficients, highest power first.

    Adapted from http://www.sgh1.net/posts/cpp-root-finder.md
*/
template <class Scalar>
inline std::vector<std::complex<Scalar>>
eigen_roots(const std::vector<Scalar> &coeffs, bool &success) {

  int matsz = coeffs.size() - 1;
  std::vector<std::complex<Scalar>> vret;
  Matrix<Scalar> companion_mat(matsz, matsz);
  companion_mat.setZero();

  for (int n = 0; n < matsz; ++n) {
    for (int m = 0; m < matsz; ++m) {

      if (n == m + 1)
        companion_mat(n, m) = 1.0;

      if (m == matsz - 1)
        companion_mat(n, m) = -coeffs[matsz - n] / coeffs[0];
    }
  }

  Eigen::EigenSolver<Matrix<Scalar>> solver(companion_mat);
  if (solver.info() == Eigen::Success) {
    success = true;
  } else {
    success = false;
  }

  Matrix<std::complex<Scalar>> eig = solver.eigenvalues();
  for (int i = 0; i < matsz; ++i)
    vret.push_back(eig(i));

  return vret;
}

/**
    Compute the points of intersection between a circle and an ellipse
    in the frame where the ellipse is centered at the origin,
    the semi-major axis of the ellipse is aligned with the x axis,
    and the circle is centered at `(xo, yo)`.

*/
template <class Scalar, int N>
inline Vector<ADScalar<Scalar, N>>
get_roots(const ADScalar<Scalar, N> &b_, const ADScalar<Scalar, N> &theta_,
          const ADScalar<Scalar, N> &costheta_,
          const ADScalar<Scalar, N> &sintheta_, const ADScalar<Scalar, N> &bo_,
          const ADScalar<Scalar, N> &ro_) {

  // Get the *values*
  using A = ADScalar<Scalar, N>;
  using Complex = std::complex<Scalar>;
  Scalar b = b_.value();
  Scalar costheta = costheta_.value();
  Scalar sintheta = sintheta_.value();
  Scalar bo = bo_.value();
  Scalar ro = ro_.value();

  // Roots and derivs
  int nroots = 0;
  Vector<Scalar> x(4), dxdb(4), dxdtheta(4), dxdbo(4), dxdro(4);

  // We'll solve for circle-ellipse intersections
  // in the frame where the ellipse is centered at the origin,
  // the semi-major axis of the ellipse is aligned with the x axis,
  // and the circle is centered at `(xo, yo)`.
  Scalar xo = bo * sintheta;
  Scalar yo = bo * costheta;

  // Useful quantities
  Scalar b2 = b * b;
  Scalar b4 = b2 * b2;
  Scalar ro2 = ro * ro;
  Scalar xo2 = xo * xo;
  Scalar yo2 = yo * yo;

  // Special case: b = 0
  if (abs(b) < STARRY_B_ZERO_TOL) {

    // Roots
    Scalar term = sqrt(ro2 - yo2);
    if (abs(xo + term) < 1)
      x(nroots++) = xo + term;
    if ((abs(xo - term) < 1) && term != 0)
      x(nroots++) = xo - term;

    // Derivatives
    int s = yo < 0 ? 1 : -1;
    for (int n = 0; n < nroots; ++n) {
      dxdb(n) = s *
                sqrt((1 - x(n) * x(n)) * (ro2 - (x(n) - xo) * (x(n) - xo))) /
                (x(n) - xo);
      dxdtheta(n) = bo * (costheta - s * sqrt(ro2 - (x(n) - xo) * (x(n) - xo)) /
                                         (x(n) - xo) * sintheta);
      dxdbo(n) = sintheta + s * sqrt(ro2 - (x(n) - xo) * (x(n) - xo)) /
                                (x(n) - xo) * costheta;
      dxdro(n) = ro / (x(n) - xo);
    }

    // Need to solve a quartic
  } else {

    // Get the roots (eigenvalue problem)
    std::vector<Scalar> coeffs;
    coeffs.push_back((1 - b2) * (1 - b2));
    coeffs.push_back(-4 * xo * (1 - b2));
    coeffs.push_back(-2 *
                     (b4 + ro2 - 3 * xo2 - yo2 - b2 * (1 + ro2 - xo2 + yo2)));
    coeffs.push_back(-4 * xo * (b2 - ro2 + xo2 + yo2));
    coeffs.push_back(b4 - 2 * b2 * (ro2 - xo2 + yo2) +
                     (ro2 - xo2 - yo2) * (ro2 - xo2 - yo2));
    bool success = false;
    std::vector<std::complex<Scalar>> roots = eigen_roots(coeffs, success);
    if (!success) {
      std::stringstream args;
      args << "b_ = " << b_ << ", "
           << "theta_ = " << theta_ << ", "
           << "costheta_ = " << costheta_ << ", "
           << "sintheta_ = " << sintheta_ << ", "
           << "bo_ = " << bo_ << ", "
           << "ro_ = " << ro_;
      throw StarryException("Root eigensolver did not converge.",
                            "oblate/geometry.h", "get_roots", args.str());
    }

    // Polish the roots using Newton's method on the *original*
    // function, which is more stable than the quartic expression.
    Complex fA, fB, f, df, minx;
    Scalar absf, minf, minx_re;
    Eigen::Matrix<Scalar, 2, 2> diff;
    typename Eigen::Matrix<Scalar, 2, 2>::Index minRow, minCol;
    Scalar p, q, v, w, t;
    Scalar s0, s1;
    for (int n = 0; n < 4; ++n) {

      /*
      We're looking for the intersection of the function

           y1 = +/- b * sqrt(1 - x^2)

      and the function

           y2 = yo +/- sqrt(ro^2 - (x - xo^2))

      Let's figure out which of the four cases (+/-, +/-) this
      root is a solution to. We're then going to polish
      the root by minimizing the function

           f = y1 - y2
      */
      fA = sqrt(1.0 - roots[n] * roots[n]);
      fB = sqrt(ro2 - (roots[n] - xo) * (roots[n] - xo));
      diff <<                       //
          abs(b * fA - (yo + fB)),  //
          abs(b * fA - (yo - fB)),  //
          abs(-b * fA - (yo + fB)), //
          abs(-b * fA - (yo - fB));
      absf = diff.minCoeff(&minRow, &minCol);
      s0 = minRow == 0 ? 1 : -1;
      s1 = minCol == 0 ? 1 : -1;

      // Apply Newton's method to polish the root
      minf = INFINITY;
      for (int k = 0; k < STARRY_ROOT_MAX_ITER; ++k) {
        fA = sqrt(1.0 - roots[n] * roots[n]);
        fB = sqrt(ro2 - (roots[n] - xo) * (roots[n] - xo));
        f = s0 * b * fA - (yo + s1 * fB);
        absf = abs(f);
        if (absf < minf) {
          minf = absf;
          minx = roots[n];
          if (minf <= STARRY_ROOT_TOL_HIGH)
            break;
        }
        df = -s0 * b * roots[n] / fA + s1 * (roots[n] - xo) / fB;
        roots[n] -= f / df;
      }

      // Only keep the root if the solver actually converged
      if (minf < STARRY_ROOT_TOL_MED) {

        // Only keep the root if it's real
        if ((abs(minx.imag()) < STARRY_ROOT_TOL_HIGH) &&
            (abs(minx.real()) <= 1)) {

          // Discard the (tiny) imaginary part
          minx_re = minx.real();

          // Check that we haven't included this root already
          bool good = true;
          for (int n = 0; n < nroots; ++n) {
            if (abs(x(n) - minx) < STARRY_ROOT_TOL_DUP) {
              good = false;
              break;
            }
          }
          if (good) {

            // Store the root
            x(nroots) = minx_re;

            // Now compute its derivatives
            q = sqrt(ro2 - (minx_re - xo) * (minx_re - xo));
            p = sqrt(1 - minx_re * minx_re);
            v = (minx_re - xo) / q;
            w = b / p;
            t = 1.0 / (w * minx_re - (s1 * s0) * v);
            dxdb(nroots) = t * p;
            dxdtheta(nroots) = (s1 * sintheta * v - costheta) * (bo * t * s0);
            dxdbo(nroots) = -(sintheta + s1 * costheta * v) * (t * s0);
            dxdro(nroots) = -ro * t / q * s1 * s0;

            // Move on
            ++nroots;
          }
        }
      }
    }
  }

  // We're done!
  Vector<A> result(nroots);
  for (int n = 0; n < nroots; ++n) {
    result(n).value() = x(n);
    result(n).derivatives() =
        dxdb(n) * b_.derivatives() + dxdtheta(n) * theta_.derivatives() +
        dxdbo(n) * bo_.derivatives() + dxdro(n) * ro_.derivatives();
  }

  return result;
}

/**
    Compute the angles at which the circle intersects the ellipse
    in the frame where the ellipse is centered at the origin,
    the semi-major axis of the ellipse is at an angle `theta` with
    respect to the x axis, and the circle is centered at `(0, bo)`.

*/
template <class Scalar, int N>
inline void
get_angles(const ADScalar<Scalar, N> &bo_, const ADScalar<Scalar, N> &ro_,
           const ADScalar<Scalar, N> &f_, const ADScalar<Scalar, N> &theta_,
           ADScalar<Scalar, N> &phi1, ADScalar<Scalar, N> &phi2,
           ADScalar<Scalar, N> &xi1, ADScalar<Scalar, N> &xi2) {

  using A = ADScalar<Scalar, N>;

  // We may need to adjust these, so make a copy
  A bo = bo_;
  A ro = ro_;
  A f = f_;
  A b = 1 - f_;
  A theta = theta_;
  A costheta = cos(theta);
  A sintheta = sin(theta);

  // Enforce bo >= 0
  if (bo < 0) {
    bo = -bo;
    theta -= pi<Scalar>();
  }

  // Trivial cases
  if (bo <= ro - 1 + STARRY_COMPLETE_OCC_TOL) {

    // Complete occultation
    phi1 = phi2 = xi1 = xi2 = 0.0;

  } else if (bo + ro + f <= 1 + STARRY_GRAZING_TOL) {

    // Regular occultation, but occultor doesn't touch the limb
    phi2 = xi1 = 0.0;
    phi1 = xi2 = 2 * pi<Scalar>();

  } else if (bo >= 1 + ro - STARRY_NO_OCC_TOL) {

    // No occultation
    phi1 = phi2 = xi1 = 0.0;
    xi2 = 2 * pi<Scalar>();
  }

  // HACK: This grazing configuration leads to instabilities
  // in the root solver. Let's avoid it.
  if ((1 - ro - STARRY_GRAZING_TOL <= bo) &&
      (bo <= 1 - ro + STARRY_GRAZING_TOL))
    bo = 1 - ro + STARRY_GRAZING_TOL;

  // HACK: The eigensolver doesn't converge when ro = 1 and theta = pi / 2.
  if ((abs(1 - ro) < STARRY_THETA_UNIT_RADIUS_TOL) &&
      (abs(costheta) < STARRY_THETA_UNIT_RADIUS_TOL)) {
    costheta += (costheta > 0 ? STARRY_THETA_UNIT_RADIUS_TOL
                              : -STARRY_THETA_UNIT_RADIUS_TOL);
  }

  // Get the points of intersection between the circle & ellipse
  // These are the roots to a quartic equation.
  A xo = bo * sintheta;
  A yo = bo * costheta;
  Vector<A> x = get_roots(b, theta, costheta, sintheta, bo, ro);
  int nroots = x.size();

  if (nroots < 2) {

    // There are no intersections between the circle
    // and the ellipse, or there is a single intersection
    // (grazing occultation). There are 3 possibilies.

    // Is the center of the circle outside the ellipse?
    if (abs(yo) > b * sqrt(1 - xo * xo)) {

      // Is the center of the ellipse outside the circle?
      if (bo > ro) {

        // No occultation
        phi1 = phi2 = xi1 = 0.0;
        xi2 = 2 * pi<Scalar>();

      } else {

        // Complete occultation
        phi1 = phi2 = xi1 = xi2 = 0.0;
      }

    } else {

      // Regular occultation, but occultor doesn't touch the limb
      phi2 = xi1 = 0.0;
      phi1 = xi2 = 2 * pi<Scalar>();
    }

  } else if (nroots == 2) {

    // Regular occultation
    A y, rhs, xm, ym, mid;

    // First root
    y = b * sqrt(1 - x(0) * x(0));
    rhs = (ro * ro - (x(0) - xo) * (x(0) - xo));
    if (abs((y - yo) * (y - yo) - rhs) < abs((y - yo) * (y - yo) - rhs)) {
      phi1 = theta + atan2(y - yo, x(0) - xo);
      xi1 = atan2(sqrt(1 - x(0) * x(0)), x(0));
    } else {
      phi1 = theta + atan2(y + yo, x(0) - xo);
      xi1 = atan2(-sqrt(1 - x(0) * x(0)), x(0));
    }

    // Second root
    y = b * sqrt(1 - x(1) * x(1));
    rhs = (ro * ro - (x(1) - xo) * (x(1) - xo));
    if (abs((y - yo) * (y - yo) - rhs) < abs((y - yo) * (y - yo) - rhs)) {
      phi2 = theta + atan2(y - yo, x(1) - xo);
      xi2 = atan2(sqrt(1 - x(0) * x(0)), x(0));
    } else {
      phi2 = theta + atan2(y + yo, x(1) - xo);
      xi2 = atan2(-sqrt(1 - x(0) * x(0)), x(0));
    }

    // Wrap and sort the angles
    phi1 = angle(phi1);
    phi2 = angle(phi2);
    xi1 = angle(xi1);
    xi2 = angle(xi2);
    if (xi1 > xi2) {
      std::swap(xi1, xi2);
      std::swap(phi1, phi2);
    }

    // Ensure the T integral does not take us through the inside of the occultor
    mid = 0.5 * (xi1 + xi2);
    xm = cos(mid);
    ym = b * sin(mid);
    if ((xm - xo) * (xm - xo) + (ym - yo) * (ym - yo) < ro * ro) {
      std::swap(xi1, xi2);
      xi2 += 2 * pi<Scalar>();
    }

    // Ensure the P integral takes us through the inside of the star
    mid = 0.5 * (phi1 + phi2);
    xm = xo + ro * cos(theta - mid);
    ym = yo - ro * sin(theta - mid);
    if (ym * ym > b * b * (1 - xm * xm)) {
      phi1 += 2 * pi<Scalar>();
    }

  } else {

    // Pathological case?
    std::stringstream args;
    args << "bo_ = " << bo_ << ", "
         << "ro_ = " << ro_ << ", "
         << "f_ = " << f_ << ", "
         << "theta_ = " << theta_;
    throw StarryException("Unexpected branch.", "oblate/geometry.h",
                          "get_angles", args.str());
  }
}

} // namespace geometry
} // namespace oblate
} // namespace starry

#endif
