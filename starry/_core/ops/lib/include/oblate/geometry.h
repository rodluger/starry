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
inline void
get_roots(const ADScalar<Scalar, N> &b_, const ADScalar<Scalar, N> &theta_,
          const ADScalar<Scalar, N> &costheta_,
          const ADScalar<Scalar, N> &sintheta_, const ADScalar<Scalar, N> &bo_,
          const ADScalar<Scalar, N> &ro_, Vector<ADScalar<Scalar, N>> &x,
          Vector<ADScalar<Scalar, N>> &y) {

  // Get the *values*
  using Complex = std::complex<Scalar>;
  Scalar b = b_.value();
  Scalar costheta = costheta_.value();
  Scalar sintheta = sintheta_.value();
  Scalar bo = bo_.value();
  Scalar ro = ro_.value();

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
  Complex fA, fB, f, df, minxc, minyc;
  Scalar absf, minerr;
  Scalar p, q, v, w, t;
  Vector<Scalar> error(18);
  Vector<Complex> xc(18), yc(18);
  Vector<Scalar> sgn0(18), sgn1(18);
  Vector<int> keep(18);
  error.setZero();
  xc.setZero();
  yc.setZero();
  sgn0.setZero();
  sgn1.setZero();
  keep.setZero();
  int m = 0;
  for (int n = 0; n < 4; ++n) {

    /*
    We're looking for the intersection of the function

         y1 = +/- b * sqrt(1 - x^2)

    and the function

         y2 = yo +/- sqrt(ro^2 - (x - xo^2))

    Let's figure out which of the four cases (+/-, +/-) this
    root could be a solution to. We're then going to polish
    the root(s) by minimizing the function

         f = y1 - y2
    */
    fA = sqrt(1.0 - roots[n] * roots[n]);
    fB = sqrt(ro2 - (roots[n] - xo) * (roots[n] - xo));
    for (Scalar s0 = -1; s0 < 2; s0 += 2) {
      for (Scalar s1 = -1; s1 < 2; s1 += 2) {

        // If the error is moderately small, keep & polish this root
        minerr = abs(s0 * b * fA - (yo + s1 * fB));
        if (minerr < STARRY_ROOT_TOL_LOW) {

          // Apply Newton's method to polish the root
          for (int k = 0; k < STARRY_ROOT_MAX_ITER; ++k) {
            fA = sqrt(1.0 - roots[n] * roots[n]);
            fB = sqrt(ro2 - (roots[n] - xo) * (roots[n] - xo));
            f = s0 * b * fA - (yo + s1 * fB);
            absf = abs(f);
            if (absf <= minerr) {
              minerr = absf;
              minxc = roots[n];
              minyc = s0 * b * fA;
              // Break if the error is close to mach eps
              if (minerr <= STARRY_ROOT_TOL_HIGH)
                break;
            }
            if ((fA == 0.0) || (fB == 0.0))
              break;
            df = -s0 * b * roots[n] / fA + s1 * (roots[n] - xo) / fB;
            if (df == 0.0)
              break;
            roots[n] -= f / df;
          }

          // Store this root (regardless of convergence)
          // We'll prune the results below.
          keep(m) = 1;
          xc(m) = minxc;
          yc(m) = minyc;
          error(m) = minerr;
          sgn0(m) = s0;
          sgn1(m) = s1;

        } else {

          // Discard this root now.
          keep(m) = 0;
        }

        ++m;
      }
    }
  }

  // The Newton solver will not converge if the root is
  // at x = +/- 1 since the derivative is infinite. Here we
  // explicitly add these two roots. If they are not solutions
  // to the problem, they get discarded in the next step.
  Scalar s1 = 1.0;
  for (m = 16; m < 18; ++m) {
    keep(m) = 1;
    xc(m) = s1;
    yc(m) = 0.0;
    sgn0(m) = 1;
    if (yo < 0) {
      error(m) = abs(yo + sqrt(ro2 - (xc(m) - xo) * (xc(m) - xo)));
      sgn1(m) = 1;
    } else {
      error(m) = abs(yo - sqrt(ro2 - (xc(m) - xo) * (xc(m) - xo)));
      sgn1(m) = -1;
    }
    s1 *= -1;
  }

  // Discard the roots with the highest error until we have 4.
  typename Eigen::Matrix<int, 18, 1>::Index index;
  while (keep.sum() < 4) {
    error.cwiseProduct(keep.template cast<Scalar>()).maxCoeff(&index);
    keep(index) = 0;
  }

  // Eliminate complex roots and roots with large error
  for (int i = 0; i < 18; ++i) {
    if ((abs(xc(i).imag()) > STARRY_ROOT_TOL_FINAL) ||
        (abs(yc(i).imag()) > STARRY_ROOT_TOL_FINAL) ||
        (error(i) > STARRY_ROOT_TOL_FINAL))
      keep(i) = 0;
  }

  // Eliminate duplicate roots. Note that we will eliminate
  // *both* roots, since a duplicate real root corresponds to
  // a grazing configuration, which we can ignore.
  Complex dx, dy;
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < i; ++j) {
      if (keep(i) && keep(j)) {
        dx = (xc(i) - xc(j));
        dy = (yc(i) - yc(j));
        if (abs(dx * dx + dy * dy) < STARRY_ROOT_TOL_DUP) {
          keep(i) = 0;
          keep(j) = 0;
        }
      }
    }
  }

  // Collect the valid roots and return
  int nroots = keep.sum();
  x.resize(nroots);
  y.resize(nroots);
  int n = 0;
  for (m = 0; m < 18; ++m) {

    // DEBUG
    /*
    std::cout << std::setprecision(12) << "(" << keep(m) << ") " << xc(m).real()
              << " + " << xc(m).imag() << "j,  " << yc(m).real() << " + "
              << yc(m).imag() << "j" << std::endl;
    std::cout << error(m) << std::endl << std::endl;
    */

    if (keep(m)) {

      // Get the x value of the root
      x(n).value() = xc(m).real();

      // Compute its derivatives
      if (N > 0) {

        Scalar dxdb, dxdtheta, dxdbo, dxdro;
        q = sqrt(ro2 - (xc(m).real() - xo) * (xc(m).real() - xo));
        p = sqrt(1 - xc(m).real() * xc(m).real());
        v = (xc(m).real() - xo) / q;
        w = b / p;
        t = 1.0 / (w * xc(m).real() - (sgn1(m) * sgn0(m)) * v);
        dxdb = t * p;
        dxdtheta = (sgn1(m) * sintheta * v - costheta) * (bo * t * sgn0(m));
        dxdbo = -(sintheta + sgn1(m) * costheta * v) * (t * sgn0(m));
        dxdro = -ro * t / q * sgn1(m) * sgn0(m);
        x(n).derivatives() =
            dxdb * b_.derivatives() + dxdtheta * theta_.derivatives() +
            dxdbo * bo_.derivatives() + dxdro * ro_.derivatives();
        y(n) = sgn0(m) * b_ * sqrt(1.0 - x(n) * x(n));

      } else {

        // The y value of the root
        y(n).value() = yc(m).real();
      }

      ++n;
    }
  }
}

/**
    Compute the angles at which the circle intersects the ellipse.

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

  // Enforce bo >= 0
  if (bo < 0) {
    bo = -bo;
    theta -= pi<Scalar>();
  }

  // Avoid f = 0 issues
  if (f < STARRY_MIN_F) {
    f = STARRY_MIN_F;
    b = 1 - f;
  }

  A costheta = cos(theta);
  A sintheta = sin(theta);

  // Trivial cases
  if (bo <= ro - 1 + STARRY_COMPLETE_OCC_TOL) {

    // Complete occultation
    phi1 = phi2 = xi1 = xi2 = 0.0;
    return;

  } else if (bo + ro + f <= 1 + STARRY_GRAZING_TOL) {

    // Regular occultation, but occultor doesn't touch the limb
    phi2 = xi1 = 0.0;
    phi1 = xi2 = 2 * pi<Scalar>();
    return;

  } else if (bo >= 1 + ro - STARRY_NO_OCC_TOL) {

    // No occultation
    phi1 = phi2 = xi1 = 0.0;
    xi2 = 2 * pi<Scalar>();
    return;
  }

  // This grazing configuration leads to instabilities
  // in the root solver. Let's avoid it.
  if ((1 - ro - STARRY_GRAZING_TOL <= bo) &&
      (bo <= 1 - ro + STARRY_GRAZING_TOL))
    bo = 1 - ro + STARRY_GRAZING_TOL;

  // The eigensolver doesn't converge when ro = 1 and theta = pi / 2.
  if ((abs(1 - ro) < STARRY_THETA_UNIT_RADIUS_TOL) &&
      (abs(costheta) < STARRY_THETA_UNIT_RADIUS_TOL)) {
    costheta += (costheta > 0 ? STARRY_THETA_UNIT_RADIUS_TOL
                              : -STARRY_THETA_UNIT_RADIUS_TOL);
  }

  // Get the points of intersection between the circle & ellipse
  // These are the roots to a quartic equation.
  A xo = bo * sintheta;
  A yo = bo * costheta;
  Vector<A> x, y;
  get_roots(b, theta, costheta, sintheta, bo, ro, x, y);
  int nroots = x.size();

  if (nroots == 0) {

    // There are no intersections between the circle
    // and the ellipse. There are 3 possibilies.

    // Is the center of the circle outside the ellipse?
    if ((abs(xo) > 1) || (abs(yo) > b * sqrt(1 - xo * xo))) {

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
    A xm, ym, mid;

    // Get the angles
    phi1 = theta + atan2(y(0) - yo, x(0) - xo);
    phi2 = theta + atan2(y(1) - yo, x(1) - xo);
    xi1 = atan2(sqrt(1 - x(0) * x(0)), x(0));
    if (y(0) < 0)
      xi1 *= -1;
    xi2 = atan2(sqrt(1 - x(1) * x(1)), x(1));
    if (y(1) < 0)
      xi2 *= -1;

    // Wrap and sort the angles
    phi1 = angle(phi1);
    phi2 = angle(phi2);
    xi1 = angle(xi1);
    xi2 = angle(xi2);

    // xi is always counter-clockwise
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
      if (phi1 < phi2) {
        phi1 += 2 * pi<Scalar>();
      } else {
        phi2 += 2 * pi<Scalar>();
      }
    }

    // phi is always clockwise
    if (phi2 > phi1) {
      std::swap(phi1, phi2);
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
