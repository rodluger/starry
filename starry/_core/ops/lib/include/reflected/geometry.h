/**
\file geometry.h
\brief Circle-ellipse intersection stuff.

*/

#ifndef _STARRY_GEOMETRY_H_
#define _STARRY_GEOMETRY_H_

#include "../utils.h"
#include "constants.h"
#include <Eigen/Eigenvalues>

namespace starry {
namespace reflected {
namespace geometry {

using namespace utils;

/**
    Return True if a point is on the dayside.
*/
template <typename T>
inline bool on_dayside(const T &b, const T &theta, const T &costheta,
                       const T &sintheta, const T &x, const T &y) {
  T xr = x * costheta + y * sintheta;
  T yr = -x * sintheta + y * costheta;
  T term = 1 - xr * xr;
  T yt = b * sqrt(term);
  return bool(yr >= yt);
}

/**
    Sort a pair of `phi` angles.

    The basic rule here: the direction phi1 -> phi2 must
    always span the dayside.
*/
template <typename T>
inline Vector<T> sort_phi(const T &b, const T &theta, const T &costheta,
                          const T &sintheta, const T &bo, const T &ro,
                          const Vector<T> &phi_) {

  // First ensure the range is correct
  T phi1 = angle(phi_(0));
  T phi2 = angle(phi_(1));
  Vector<T> phi(2);
  phi << phi1, phi2;
  if (phi(1) < phi(0))
    phi(1) += 2 * pi<T>();

  // Now take the midpoint and check that it's on-planet and on the
  // dayside. If not, we swap the integration limits.
  T phim = phi.mean();
  T x = ro * cos(phim);
  T y = bo + ro * sin(phim);
  if ((x * x + y * y > 1) || !on_dayside(b, theta, costheta, sintheta, x, y))
    phi << angle(phi2), angle(phi1);
  if (phi(1) < phi(0))
    phi(1) += 2 * pi<T>();
  return phi;
}

/**
    Sort a pair of `xi` angles.

    The basic rule here: the direction xi2 --> xi1 must
    always span the inside of the occultor. (Note that
    the limits of the `T` integral are xi2 --> xi1, since
    we integrate *clockwise* along the arc.) Since xi
    is limited to the range [0, pi], enforcing this is
    actually trivial: we just need to make sure they are
    arranged in decreasing order.
*/
template <typename T>
inline Vector<T> sort_xi(const T &b, const T &theta, const T &costheta,
                         const T &sintheta, const T &bo, const T &ro,
                         const Vector<T> &xi_) {

  T xi1 = angle(xi_(0));
  T xi2 = angle(xi_(1));
  Vector<T> xi(2);
  if (xi1 > xi2)
    xi << xi1, xi2;
  else
    xi << xi2, xi1;
  return xi;
}

/**
    Sort a pair of `lam` angles.

    The basic rule here: the direction lam1 --> lam2
    must always span the inside of the occultor and the
    dayside.
*/
template <typename T>
inline Vector<T> sort_lam(const T &b, const T &theta, const T &costheta,
                          const T &sintheta, const T &bo, const T &ro,
                          const Vector<T> &lam_) {

  // First ensure the range is correct
  T lam1 = angle(lam_(0));
  T lam2 = angle(lam_(1));
  Vector<T> lam(2);
  lam << lam1, lam2;
  if (lam(1) < lam(0))
    lam(1) += 2 * pi<T>();

  // Now take the midpoint and ensure it is inside
  // the occultor *and* on the dayside. If not, swap
  // the integration limits.
  T lamm = lam.mean();
  T x = cos(lamm);
  T y = sin(lamm);
  if ((x * x + (y - bo) * (y - bo) > ro * ro) ||
      !on_dayside(b, theta, costheta, sintheta, T((1 - STARRY_ANGLE_TOL) * x),
                  T((1 - STARRY_ANGLE_TOL) * y)))
    lam << angle(lam2), angle(lam1);
  if (lam(1) < lam(0))
    lam(1) += 2 * pi<T>();
  return lam;
}

/**
    Polynomial root finder using an eigensolver.
    `coeffs` is a vector of coefficients, highest power first.

    Adapted from http://www.sgh1.net/posts/cpp-root-finder.md
*/
template <typename T>
inline std::vector<std::complex<T>> eigen_roots(const std::vector<T> &coeffs,
                                                bool &success) {
  int N = coeffs.size();
  int matsz = N - 1;
  std::vector<std::complex<T>> vret;

  Matrix<T> companion_mat(matsz, matsz);
  companion_mat.setZero();

  for (int n = 0; n < matsz; ++n) {
    for (int m = 0; m < matsz; ++m) {

      if (n == m + 1)
        companion_mat(n, m) = 1.0;

      if (m == matsz - 1)
        companion_mat(n, m) = -coeffs[matsz - n] / coeffs[0];
    }
  }

  Eigen::EigenSolver<Matrix<T>> solver(companion_mat);
  if (solver.info() == Eigen::Success) {
    success = true;
  } else {
    success = false;
  }

  Matrix<std::complex<T>> eig = solver.eigenvalues();
  for (int i = 0; i < matsz; ++i)
    vret.push_back(eig(i));

  return vret;
}

/**
    Compute the points of intersection between the occultor and the terminator.

*/
template <typename T>
inline Vector<T> get_roots(const T &b_, const T &theta_, const T &costheta_,
                           const T &sintheta_, const T &bo_, const T &ro_) {

  // Get the *values*
  using Scalar = typename T::Scalar;
  using Complex = std::complex<Scalar>;
  Scalar b = b_.value();
  Scalar costheta = costheta_.value();
  Scalar sintheta = sintheta_.value();
  Scalar bo = bo_.value();
  Scalar ro = ro_.value();

  // Roots and derivs
  int nroots = 0;
  Vector<Scalar> x(4), dxdb(4), dxdtheta(4), dxdbo(4), dxdro(4);

  // We'll solve for occultor-terminator intersections
  // in the frame where the semi-major axis of the
  // terminator ellipse is aligned with the x axis
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
      dxdtheta(n) = bo * (costheta -
                          s * sqrt(ro2 - (x(n) - xo) * (x(n) - xo)) /
                              (x(n) - xo) * sintheta);
      dxdbo(n) =
          sintheta +
          s * sqrt(ro2 - (x(n) - xo) * (x(n) - xo)) / (x(n) - xo) * costheta;
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
                            "reflected/geometry.h", "get_roots", args.str());
    }

    // Polish the roots using Newton's method on the *original*
    // function, which is more stable than the quartic expression.
    Complex A, B, f, df, minx;
    Scalar absfp, absfm, absf, minf, minx_re;
    Scalar p, q, v, w, t;
    Scalar s;
    for (int n = 0; n < 4; ++n) {

      /*
      We're looking for the intersection of the function

           y1 = b * sqrt(1 - x^2)

      and the function

           y2 = yo +/- sqrt(ro^2 - (x - xo^2))

      Let's figure out which of the two cases (+/-) this
      root is a solution to. We're then going to polish
      the root by minimizing the function

           f = y1 - y2
      */

      A = sqrt(1.0 - roots[n] * roots[n]);
      B = sqrt(ro2 - (roots[n] - xo) * (roots[n] - xo));
      absfp = abs(b * A - yo + B);
      absfm = abs(b * A - yo - B);

      if (absfp < absfm) {
        s = 1;
        absf = absfp;
      } else {
        s = -1;
        absf = absfm;
      }

      /*
      Some roots may instead correspond to

           y = -b * sqrt(1 - x^2)

      which is the wrong half of the terminator ellipse.
      Let's only move forward if |f| is decently small.
      */

      if (absf < STARRY_ROOT_TOL_LOW) {

        // Apply Newton's method to polish the root
        minf = INFINITY;
        for (int k = 0; k < STARRY_ROOT_MAX_ITER; ++k) {
          A = sqrt(1.0 - roots[n] * roots[n]);
          B = sqrt(ro2 - (roots[n] - xo) * (roots[n] - xo));
          f = b * A + s * B - yo;
          absf = abs(f);
          if (absf < minf) {
            minf = absf;
            minx = roots[n];
            if (minf <= STARRY_ROOT_TOL_HIGH)
              break;
          }
          df = -(b * roots[n] / A + s * (roots[n] - xo) / B);
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
              t = 1.0 / (-w * minx_re - s * v);
              dxdb(nroots) = -t * p;
              dxdtheta(nroots) = -t * bo * (sintheta + s * v * costheta);
              dxdbo(nroots) = t * (costheta - s * v * sintheta);
              dxdro(nroots) = -t * s * ro / q;

              // Move on
              ++nroots;
            }
          }
        }
      }
    }
  }

  // Check if the extrema of the terminator ellipse are occulted
  bool e1 = costheta * costheta + (sintheta - bo) * (sintheta - bo) <
            ro2 + STARRY_ROOT_TOL_HIGH;
  bool e2 = costheta * costheta + (sintheta + bo) * (sintheta + bo) <
            ro2 + STARRY_ROOT_TOL_HIGH;

  // One is occulted, the other is not.
  // Usually we should have a single root, but
  // pathological cases with 3 roots (and maybe 4?)
  // are also possible.
  if ((e1 && (!e2)) || (e2 && (!e1))) {
    if ((nroots == 0) || (nroots == 2)) {
      std::stringstream args;
      args << "b_ = " << b_ << ", "
           << "theta_ = " << theta_ << ", "
           << "costheta_ = " << costheta_ << ", "
           << "sintheta_ = " << sintheta_ << ", "
           << "bo_ = " << bo_ << ", "
           << "ro_ = " << ro_;
      throw StarryException("Solver did not find the correct number of roots.",
                            "reflected/geometry.h", "get_roots", args.str());
    }
  }

  // There is one root but none of the extrema are occulted.
  // This likely corresponds to a grazing occultation of the
  // dayside or nightside.
  if (nroots == 1) {
    if ((!e1) && (!e2)) {
      // Delete the root!
      nroots = 0;
    }
  }

  // We're done!
  Vector<T> result(nroots);
  for (int n = 0; n < nroots; ++n) {
    result(n).value() = x(n);
    result(n).derivatives() =
        dxdb(n) * b_.derivatives() + dxdtheta(n) * theta_.derivatives() +
        dxdbo(n) * bo_.derivatives() + dxdro(n) * ro_.derivatives();
  }

  return result;
}

template <typename T>
inline int get_angles(const T &b, const T &theta_, const T &costheta_,
                      const T &sintheta_, const T &bo_, const T &ro,
                      Vector<T> &kappa, Vector<T> &lam, Vector<T> &xi) {

  // We may need to adjust these, so make a copy
  T bo = bo_;
  T theta = theta_;
  T costheta = costheta_;
  T sintheta = sintheta_;

  // Helper angle
  Vector<T> phi;

  // Trivial cases
  if (bo <= ro - 1 + STARRY_COMPLETE_OCC_TOL) {

    // Complete occultation
    kappa.setZero(0);
    lam.setZero(0);
    xi.setZero(0);
    return FLUX_ZERO;

  } else if (bo >= 1 + ro - STARRY_NO_OCC_TOL) {

    // No occultation
    kappa.setZero(0);
    lam.setZero(0);
    xi.setZero(0);
    return FLUX_SIMPLE_REFL;

  } else if (b >= 1.0 - STARRY_B_ONE_TOL) {

    // Only the night side is visible,
    // so the total flux is zero
    kappa.setZero(0);
    lam.setZero(0);
    xi.setZero(0);
    return FLUX_ZERO;

  } else if (b <= -1.0 + STARRY_B_ONE_TOL) {

    // The substellar point is the center of
    // the disk, so this is analytically equivalent
    // to the linear limb darkening solution
    kappa.setZero(0);
    lam.setZero(0);
    xi.setZero(0);
    return FLUX_NOON;
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

  // Get the points of intersection between the occultor & terminator
  // These are the roots to a quartic equation.
  T xo = bo * sintheta;
  T yo = bo * costheta;
  Vector<T> x = get_roots(b, theta, costheta, sintheta, bo, ro);
  int nroots = x.size();

  // P-Q
  if (nroots == 0) {

    // Trivial: use the standard starry algorithm
    kappa.setZero(0);
    lam.setZero(0);
    xi.setZero(0);

    if ((abs(1 - ro) <= bo) && (bo <= 1 + ro)) {

      // The occultor intersects the limb at this point
      T q = (1 - ro * ro + bo * bo) / (2 * bo);
      T xp = sqrt(1 - q * q);
      T yp = q;

      if (on_dayside(b, theta, costheta, sintheta, xp, yp)) {

        // This point is guaranteed to be on the night side
        // We're going to check if it's under the occultor or not
        xp = (1 - STARRY_ANGLE_TOL) * cos(theta + 3 * pi<T>() / 2);
        yp = (1 - STARRY_ANGLE_TOL) * sin(theta + 3 * pi<T>() / 2);

        if (xp * xp + (yp - bo) * (yp - bo) <= ro * ro) {

          // The occultor is blocking some daylight
          // and all of the night side
          return FLUX_SIMPLE_OCC;

        } else {

          // The occultor is only blocking daylight
          return FLUX_SIMPLE_OCC_REFL;
        }

      } else {

        // This point is guaranteed to be on the day side
        // We're going to check if it's under the occultor or not
        xp = (1 - STARRY_ANGLE_TOL) * cos(theta + pi<T>() / 2);
        yp = (1 - STARRY_ANGLE_TOL) * sin(theta + pi<T>() / 2);

        if (xp * xp + (yp - bo) * (yp - bo) <= ro * ro) {

          // The occultor is blocking some night side
          // and all of the day side
          return FLUX_ZERO;

        } else {

          // The occultor is only blocking the night side
          return FLUX_SIMPLE_REFL;
        }
      }

    } else {

      // The occultor does not intersect the limb or the terminator
      if (on_dayside(b, theta, costheta, sintheta, T(0.0), bo)) {

        // The occultor is only blocking daylight
        return FLUX_SIMPLE_OCC_REFL;

      } else {

        // The occultor is only blocking the night side
        return FLUX_SIMPLE_REFL;
      }
    }

    // P-Q-T
  } else if (nroots == 1) {

    // PHI
    // ---

    // Angle of intersection with limb
    T phi_l = asin((1 - ro * ro - bo * bo) / (2 * bo * ro));
    // There are always two points; always pick the one
    // that's on the dayside for definiteness
    if (!on_dayside(b, theta, costheta, sintheta, T(ro * cos(phi_l)),
                    T((bo + ro * sin(phi_l)))))
      phi_l = pi<T>() - phi_l;

    // Angle of intersection with the terminator
    T phi_t = theta + atan2(b * sqrt(1 - x(0) * x(0)) - yo, x(0) - xo);

    // Now ensure phi *only* spans the dayside.
    phi.resize(2);
    phi << phi_l, phi_t;
    phi = sort_phi(b, theta, costheta, sintheta, bo, ro, phi);
    kappa.resize(phi.size());
    kappa.array() = phi.array() + pi<T>() / 2;

    // LAMBDA
    // ------

    // Angle of intersection with occultor
    T lam_o = asin((1 - ro * ro + bo * bo) / (2 * bo));
    // There are always two points; always pick the one
    // that's on the dayside for definiteness
    if (!on_dayside(b, theta, costheta, sintheta, T(cos(lam_o)), T(sin(lam_o))))
      lam_o = pi<T>() - lam_o;

    // Angle of intersection with the terminator
    T lam_t = theta;
    // There are always two points; always pick the one
    // that's inside the occultor
    if (cos(lam_t) * cos(lam_t) + (sin(lam_t) - bo) * (sin(lam_t) - bo) >
        ro * ro)
      lam_t = pi<T>() + theta;

    // Now ensure lam *only* spans the inside of the occultor.
    lam.resize(2);
    lam << lam_o, lam_t;
    lam = sort_lam(b, theta, costheta, sintheta, bo, ro, lam);

    // XI
    // --

    // Angle of intersection with occultor
    T xi_o = atan2(sqrt(1 - x(0) * x(0)), x(0));

    // Angle of intersection with the limb
    T xi_l = ((1 - xo) * (1 - xo) + yo * yo < ro * ro) ? 0 : pi<T>();

    // Now ensure xi *only* spans the inside of the occultor.
    xi.resize(2);
    xi << xi_l, xi_o;
    xi = sort_xi(b, theta, costheta, sintheta, bo, ro, xi);

    // In all cases, we're computing the dayside occulted flux
    return FLUX_DAY_OCC;

    // P-T
  } else if (nroots == 2) {

    // Angles are easy
    lam.setZero(0);
    phi.resize(2);
    T phi0 = angle(T(theta + atan2(b * sqrt(1 - x(0) * x(0)) - yo, x(0) - xo)));
    T phi1 = angle(T(theta + atan2(b * sqrt(1 - x(1) * x(1)) - yo, x(1) - xo)));
    if (phi0 > phi1)
      phi << phi1, phi0;
    else
      phi << phi0, phi1;
    xi.resize(2);
    T xi0 = angle(T(atan2(sqrt(1 - x(0) * x(0)), x(0))));
    T xi1 = angle(T(atan2(sqrt(1 - x(1) * x(1)), x(1))));
    if (xi0 > xi1)
      xi << xi1, xi0;
    else
      xi << xi0, xi1;

    // Cases
    if (bo <= 1 - ro) {

      // No intersections with the limb (easy)
      phi = sort_phi(b, theta, costheta, sintheta, bo, ro, phi);
      kappa.resize(phi.size());
      kappa.array() = phi.array() + pi<T>() / 2;
      xi = sort_xi(b, theta, costheta, sintheta, bo, ro, xi);
      return FLUX_DAY_OCC;

    } else {

      // The occultor intersects the limb, so we need to
      // integrate along the simplest path.

      // 1. Rotate the points of intersection into a frame where the
      // semi-major axis of the terminator ellipse lies along the x axis
      // We're going to choose xi(0) to be the rightmost point in
      // this frame, so that the integration is counter-clockwise along
      // the terminator to xi(1).
      Vector<T> x(2), y(2), xr(2);
      x.array() = costheta * cos(xi.array()) - b * sintheta * sin(xi.array());
      y.array() = sintheta * cos(xi.array()) + b * costheta * sin(xi.array());
      xr.array() = x.array() * costheta + y.array() * sintheta;
      if (xr(1) > xr(0)) {
        Vector<T> tmp(2);
        tmp << xi(1), xi(0);
        xi = tmp;
      }

      // 2. Now we need the point corresponding to xi(1) to be the same as the
      // point corresponding to phi(0) in order for the path to be continuous
      T x_xi1 = costheta * cos(xi(1)) - b * sintheta * sin(xi(1));
      T y_xi1 = sintheta * cos(xi(1)) + b * costheta * sin(xi(1));
      T x_phi0 = ro * cos(phi(0));
      T y_phi0 = bo + ro * sin(phi(0));
      T x_phi1 = ro * cos(phi(1));
      T y_phi1 = bo + ro * sin(phi(1));
      T d0 = (x_xi1 - x_phi0) * (x_xi1 - x_phi0) +
             (y_xi1 - y_phi0) * (y_xi1 - y_phi0);
      T d1 = (x_xi1 - x_phi1) * (x_xi1 - x_phi1) +
             (y_xi1 - y_phi1) * (y_xi1 - y_phi1);
      if (d1 < d0) {
        Vector<T> tmp(2);
        tmp << phi(1), phi(0);
        phi = tmp;
      }

      // 3. Compare the *curvature* of the two sides of the
      // integration area. The curvatures are similar (i.e., same sign)
      // when cos(theta) < 0, in which case we must integrate *clockwise* along
      // P.
      if (costheta < 0) {
        // Integrate *clockwise* along P
        if (phi(0) < phi(1))
          phi(0) += 2 * pi<T>();
      } else {
        // Integrate *counter-clockwise* along P
        if (phi(1) < phi(0))
          phi(1) += 2 * pi<T>();
      }

      // 4. Determine the integration code. Let's identify the midpoint
      // along each integration path and average their (x, y)
      // coordinates to determine what kind of region we are
      // bounding.
      T xim = xi.mean();
      T x_xi = costheta * cos(xim) - b * sintheta * sin(xim);
      T y_xi = sintheta * cos(xim) + b * costheta * sin(xim);
      T phim = phi.mean();
      T x_phi = ro * cos(phim);
      T y_phi = bo + ro * sin(phim);
      T xp = 0.5 * (x_xi + x_phi);
      T yp = 0.5 * (y_xi + y_phi);
      if (on_dayside(b, theta, costheta, sintheta, xp, yp)) {
        if (xp * xp + (yp - bo) * (yp - bo) < ro * ro) {
          // Dayside under occultor.
          // We need to reverse the integration path, since
          // the terminator is *under* the arc along the limb
          // and we should instead start at the *leftmost* xi
          // value.
          Vector<T> tmp(2);
          tmp << phi(1), phi(0);
          phi = tmp;
          kappa.resize(phi.size());
          kappa.array() = phi.array() + pi<T>() / 2;
          tmp << xi(1), xi(0);
          xi = tmp;
          return FLUX_DAY_OCC;
        } else {
          // Dayside visible
          if (b < 0) {
            Vector<T> tmp(2);
            tmp << phi(1), phi(0);
            phi = tmp;
            tmp << xi(1), xi(0);
            xi = tmp;
          }
          kappa.resize(phi.size());
          kappa.array() = phi.array() + pi<T>() / 2;
          return FLUX_DAY_VIS;
        }
      } else {

        if (xp * xp + (yp - bo) * (yp - bo) < ro * ro) {
          // Nightside under occultor
          kappa.resize(phi.size());
          kappa.array() = phi.array() + pi<T>() / 2;
          return FLUX_NIGHT_OCC;
        } else {
          // Nightside visible
          kappa.resize(phi.size());
          kappa.array() = phi.array() + pi<T>() / 2;
          return FLUX_NIGHT_VIS;
        }
      }
    }

    // There's a pathological case with 3 roots
  } else if (nroots == 3) {

    // Pre-compute some angles
    std::sort(x.data(), x.data() + x.size());
    T phi_l = asin((1 - ro * ro - bo * bo) / (2 * bo * ro));
    T lam_o = asin((1 - ro * ro + bo * bo) / (2 * bo));
    phi.resize(4);
    xi.resize(4);
    lam.resize(2);

    // We need to do this case-by-case
    // TODO: This section is really messy / cumbersome. Clean it up.
    if (b > 0) {

      if ((-1 - xo) * (-1 - xo) + yo * yo < ro * ro) {

        Vector<T> tmp(3);
        tmp << x(2), x(1), x(0);
        x = tmp;

        phi(0) =
            angle(T(theta + atan2(b * sqrt(1 - x(0) * x(0)) - yo, x(0) - xo)));
        phi(1) =
            angle(T(theta + atan2(b * sqrt(1 - x(1) * x(1)) - yo, x(1) - xo)));
        phi(2) =
            angle(T(theta + atan2(b * sqrt(1 - x(2) * x(2)) - yo, x(2) - xo)));
        phi(3) = angle(phi_l);
        while (phi(1) < phi(0))
          phi(1) += 2 * pi<T>();
        while (phi(2) < phi(1))
          phi(2) += 2 * pi<T>();
        while (phi(3) < phi(2))
          phi(3) += 2 * pi<T>();

        xi << angle(atan2(sqrt(1 - x(1) * x(1)), x(1))),
            angle(atan2(sqrt(1 - x(0) * x(0)), x(0))), pi<T>(),
            angle(atan2(sqrt(1 - x(2) * x(2)), x(2)));

        lam << angle(lam_o), angle(pi<T>() + theta);
        if (lam(1) < lam(0))
          lam(1) += 2 * pi<T>();

      } else {

        Vector<T> tmp(3);
        tmp << x(1), x(0), x(2);
        x = tmp;

        phi(0) =
            angle(T(theta + atan2(b * sqrt(1 - x(0) * x(0)) - yo, x(0) - xo)));
        phi(1) =
            angle(T(theta + atan2(b * sqrt(1 - x(1) * x(1)) - yo, x(1) - xo)));
        phi(2) = angle(pi<T>() - phi_l);
        phi(3) =
            angle(T(theta + atan2(b * sqrt(1 - x(2) * x(2)) - yo, x(2) - xo)));
        while (phi(1) < phi(0))
          phi(1) += 2 * pi<T>();
        while (phi(2) < phi(1))
          phi(2) += 2 * pi<T>();
        while (phi(3) < phi(2))
          phi(3) += 2 * pi<T>();

        xi << angle(atan2(sqrt(1 - x(1) * x(1)), x(1))),
            angle(atan2(sqrt(1 - x(0) * x(0)), x(0))),
            angle(atan2(sqrt(1 - x(2) * x(2)), x(2))), 0.0;

        lam << angle(theta), angle(pi<T>() - lam_o);
        if (lam(1) < lam(0))
          lam(1) += 2 * pi<T>();
      }

      kappa.resize(phi.size());
      kappa.array() = phi.array() + pi<T>() / 2;
      return FLUX_TRIP_DAY_OCC;

    } else {

      if ((-1 - xo) * (-1 - xo) + yo * yo < ro * ro) {

        Vector<T> tmp(3);
        tmp << x(1), x(2), x(0);
        x = tmp;

        phi(0) =
            angle(T(theta + atan2(b * sqrt(1 - x(0) * x(0)) - yo, x(0) - xo)));
        phi(1) =
            angle(T(theta + atan2(b * sqrt(1 - x(1) * x(1)) - yo, x(1) - xo)));
        phi(2) = angle(pi<T>() - phi_l);
        phi(3) =
            angle(T(theta + atan2(b * sqrt(1 - x(2) * x(2)) - yo, x(2) - xo)));
        while (phi(1) < phi(0))
          phi(1) += 2 * pi<T>();
        while (phi(2) < phi(1))
          phi(2) += 2 * pi<T>();
        while (phi(3) < phi(2))
          phi(3) += 2 * pi<T>();

        xi << angle(atan2(sqrt(1 - x(1) * x(1)), x(1))),
            angle(atan2(sqrt(1 - x(0) * x(0)), x(0))),
            angle(atan2(sqrt(1 - x(2) * x(2)), x(2))), pi<T>();

        lam << angle(pi<T>() + theta), angle(pi<T>() - lam_o);
        if (lam(1) < lam(0))
          lam(1) += 2 * pi<T>();

      } else {

        phi(0) =
            angle(T(theta + atan2(b * sqrt(1 - x(0) * x(0)) - yo, x(0) - xo)));
        phi(1) =
            angle(T(theta + atan2(b * sqrt(1 - x(1) * x(1)) - yo, x(1) - xo)));
        phi(2) =
            angle(T(theta + atan2(b * sqrt(1 - x(2) * x(2)) - yo, x(2) - xo)));
        phi(3) = angle(phi_l);
        while (phi(1) < phi(0))
          phi(1) += 2 * pi<T>();
        while (phi(2) < phi(1))
          phi(2) += 2 * pi<T>();
        while (phi(3) < phi(2))
          phi(3) += 2 * pi<T>();

        xi << angle(atan2(sqrt(1 - x(1) * x(1)), x(1))),
            angle(atan2(sqrt(1 - x(0) * x(0)), x(0))), 0.0,
            angle(atan2(sqrt(1 - x(2) * x(2)), x(2)));

        lam << angle(lam_o), angle(theta);
        if (lam(1) < lam(0))
          lam(1) += 2 * pi<T>();
      }

      kappa.resize(phi.size());
      kappa.array() = phi.array() + pi<T>() / 2;
      return FLUX_TRIP_NIGHT_OCC;
    }

    // And a pathological case with 4 roots
  } else if (nroots == 4) {

    lam.setZero(0);
    phi.resize(4);
    xi.resize(4);

    phi << angle(theta + atan2(b * sqrt(1 - x(0) * x(0)) - yo, x(0) - xo)),
        angle(theta + atan2(b * sqrt(1 - x(1) * x(1)) - yo, x(1) - xo)),
        angle(theta + atan2(b * sqrt(1 - x(2) * x(2)) - yo, x(2) - xo)),
        angle(theta + atan2(b * sqrt(1 - x(3) * x(3)) - yo, x(3) - xo));
    std::sort(phi.data(), phi.data() + phi.size());

    Vector<T> tmp(4);
    tmp << phi(1), phi(0), phi(3), phi(2);
    phi = tmp;
    kappa.resize(phi.size());
    kappa.array() = phi.array() + pi<T>() / 2;

    xi << angle(atan2(sqrt(1 - x(0) * x(0)), x(0))),
        angle(atan2(sqrt(1 - x(1) * x(1)), x(1))),
        angle(atan2(sqrt(1 - x(2) * x(2)), x(2))),
        angle(atan2(sqrt(1 - x(3) * x(3)), x(3)));
    std::sort(xi.data(), xi.data() + xi.size());

    if (b > 0) {
      return FLUX_QUAD_NIGHT_VIS;
    } else {
      Vector<T> tmp(4);
      tmp << xi(1), xi(0), xi(3), xi(2);
      xi = tmp;
      return FLUX_QUAD_DAY_VIS;
    }

  } else {
    std::stringstream args;
    args << "b = " << b << ", "
         << "theta_ = " << theta_ << ", "
         << "costheta_ = " << costheta_ << ", "
         << "sintheta_ = " << sintheta_ << ", "
         << "bo_ = " << bo_ << ", "
         << "ro = " << ro << ", "
         << "kappa = " << kappa << ", "
         << "lam = " << lam << ", "
         << "xi = " << xi;
    throw StarryException("Unexpected branch.", "reflected/geometry.h",
                          "get_angles", args.str());
  }
}

} // namespace geometry
} // namespace reflected
} // namespace starry

#endif
