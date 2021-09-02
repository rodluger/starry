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

  // Initialize
  x.resize(4);
  y.resize(4);

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
  std::vector<Complex> roots = eigen_roots(coeffs, success);
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

  // Apply Newton's method to polish the roots
  int nroots = 0;
  for (int n = 0; n < 4; ++n) {

    Complex root = roots[n];
    Complex minxc = root;
    Complex root2, root3, root4;
    Complex f, df;
    Scalar absf, minerr = INFINITY;

    // Polish
    for (int k = 0; k < STARRY_ROOT_MAX_ITER; ++k) {

      // Compute the error
      root2 = root * root;
      root3 = root2 * root;
      root4 = root3 * root;
      f = coeffs[0] * root4 + coeffs[1] * root3 + coeffs[2] * root2 +
          coeffs[3] * root + coeffs[4];
      absf = abs(f);
      if (absf <= minerr) {
        minerr = absf;
        minxc = root;
        if (minerr <= STARRY_ROOT_TOL_HIGH)
          break;
      }

      // Take a step
      df = 4.0 * coeffs[0] * root3 + 3.0 * coeffs[1] * root2 +
           2.0 * coeffs[2] * root + coeffs[3];
      if (df == 0.0)
        break;
      root -= f / df;
    }
    root = minxc;

    // Keep the root if it's real
    if (abs(root.imag()) < STARRY_ROOT_TOL_HIGH) {

      // Nudge the root away from the endpoints
      if (root.real() > 1)
        root = 1.0 - STARRY_ROOT_TOL_HIGH;
      else if (root.real() < -1)
        root = -1.0 + STARRY_ROOT_TOL_HIGH;
      else if (root.real() < xo - ro)
        root = xo - ro + STARRY_ROOT_TOL_HIGH;
      else if (root.real() > xo + ro)
        root = xo + ro - STARRY_ROOT_TOL_HIGH;

      // Determine the y value of the point on the ellipse
      // corresponding to each root and the signs of the
      // functions describing the intersecting circle &
      // ellipse segments.
      Scalar s0, s1;
      Complex fA, fB;
      fA = b * sqrt(Scalar(1.0) - root * root);
      fB = sqrt(ro2 - (root - xo) * (root - xo));
      Vector<Scalar> diff(4);
      diff <<                   //
          abs(fA - (yo + fB)),  //
          abs(fA - (yo - fB)),  //
          abs(-fA - (yo + fB)), //
          abs(-fA - (yo - fB)); //
      typename Vector<Scalar>::Index idx(4);
      diff.minCoeff(&idx);
      if (idx < 2) {
        s0 = 1.0;
      } else {
        s0 = -1.0;
      }
      if (is_even(idx)) {
        s1 = 1.0;
      } else {
        s1 = -1.0;
      }

      // Save the root
      x(nroots).value() = root.real();
      y(nroots).value() = s0 * fA.real();

      // Compute its derivatives
      if (N > 0) {
        Scalar dxdb, dxdtheta, dxdbo, dxdro;
        Scalar p, q, v, w, t;
        q = sqrt(ro2 - (root.real() - xo) * (root.real() - xo));
        p = sqrt(1 - root.real() * root.real());
        v = (root.real() - xo) / q;
        w = b / p;
        t = 1.0 / (w * root.real() - (s1 * s0) * v);
        dxdb = t * p;
        dxdtheta =
            -(s1 * costheta * v - sintheta) * (bo * t * s0); // TODO: Check me
        dxdbo = -(costheta + s1 * sintheta * v) * (t * s0);
        dxdro = -ro * t / q * s1 * s0;
        x(nroots).derivatives() =
            dxdb * b_.derivatives() + dxdtheta * theta_.derivatives() +
            dxdbo * bo_.derivatives() + dxdro * ro_.derivatives();
        y(nroots) = s0 * b_ * sqrt(1.0 - x(nroots) * x(nroots));
      }

      ++nroots;
    }
  }

  // We're done!
  x.conservativeResize(nroots);
  y.conservativeResize(nroots);
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

  A costheta = cos(theta);
  A sintheta = sin(theta);

  // Trivial cases
  if (bo <= ro - 1 + STARRY_COMPLETE_OCC_TOL) {

    // Complete occultation
    phi1 = phi2 = xi1 = 0.0;
    xi2 = 2 * pi<Scalar>();
    return;

  } else if (bo + ro + f <= 1 + STARRY_GRAZING_TOL) {

    // Regular occultation, but occultor doesn't touch the limb
    phi1 = xi1 = xi2 = 0.0;
    phi2 = 2 * pi<Scalar>();
    return;

  } else if (bo >= 1 + ro - STARRY_NO_OCC_TOL) {

    // No occultation
    phi1 = phi2 = xi1 = xi2 = 0.0;
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
        phi1 = phi2 = xi1 = xi2 = 0.0;

      } else {

        // Complete occultation
        phi1 = phi2 = xi1 = 0.0;
        xi2 = 2 * pi<Scalar>();
      }

    } else {

      // Regular occultation, but occultor doesn't touch the limb
      phi1 = xi1 = xi2 = 0.0;
      phi2 = 2 * pi<Scalar>();
    }

  } else if (nroots == 1) {

    // Grazing configuration?
    // TODO: Perturb and repeat.
    std::stringstream args;
    args << "bo_ = " << bo_ << ", "
         << "ro_ = " << ro_ << ", "
         << "f_ = " << f_ << ", "
         << "theta_ = " << theta_;
    throw StarryException("Unexpected branch.", "oblate/geometry.h",
                          "get_angles", args.str());

  } else if (nroots == 2) {

    // Regular occultation
    A xm, ym, mid, term;

    // Get the angles
    phi1 = theta + atan2(y(0) - yo, x(0) - xo);
    phi2 = theta + atan2(y(1) - yo, x(1) - xo);
    term = 1 - x(0) * x(0);
    xi1 = atan2(sqrt(term < 0 ? 0 : term), x(0));
    if (y(0) < 0)
      xi1 *= -1;
    term = 1 - x(1) * x(1);
    xi2 = atan2(sqrt(term < 0 ? 0 : term), x(1));
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

    // Ensure the T integral takes us through the inside of the occultor
    mid = 0.5 * (xi1 + xi2);
    xm = cos(mid);
    ym = b * sin(mid);
    if ((xm - xo) * (xm - xo) + (ym - yo) * (ym - yo) >= ro * ro) {
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

    // phi is always counter-clockwise
    if (phi1 > phi2) {
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
