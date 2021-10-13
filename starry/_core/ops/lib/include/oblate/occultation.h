
/**
\file occultation.h
\brief Solver for occultations of oblate bodies.

*/

#ifndef _STARRY_OBLATE_OCCULTATION_H_
#define _STARRY_OBLATE_OCCULTATION_H_

#include "../quad.h"
#include "../utils.h"
#include "geometry.h"
#include "numerical.h"

namespace starry {
namespace oblate {
namespace occultation {

using namespace utils;
using namespace numerical;
using namespace geometry;
using std::min;
using std::max;

template <class Scalar, int N> class Occultation {

  using A = ADScalar<Scalar, N>;

protected:
  // Misc
  int deg;
  int ncoeff;

  // Inputs
  A bo;
  A ro;
  A f;
  A theta;
  A costheta;
  A sintheta;

  // Integrals
  Matrix<A> M;
  Matrix<A> Lp;
  Matrix<A> Lt;
  Vector<A> K;

  // Helper matrices
  Matrix<A> S;
  Matrix<A> C;

  // Numerical integration
  Quad<Scalar, N> QUAD;

  /**
   *
   * Compute the matrix of `L` integrals.
   *
   * The term `L_{i,j}` is the solution to the integral of
   *
   *    cos(phi)^i * sin(phi)^j
   *
   * from phi = phip1 to phi = phip2, where
   *
   *    phip = phi - theta
   *
   * (pT integral) or
   *
   *    phip = xi
   *
   * (tT integral).
   *
   */
  inline void compute_L(const A &phip1, const A &phip2, Matrix<A> &L) {

    //
    int nmax = deg + 3;
    A cp1 = cos(phip1);
    A cp2 = cos(phip2);
    A sp1 = sin(phip1);
    A sp2 = sin(phip2);
    L.resize(nmax, nmax);

    // Lower boundary
    L(0, 0) = phip2 - phip1;
    L(1, 0) = sp2 - sp1;
    L(0, 1) = cp1 - cp2;
    L(1, 1) = 0.5 * (cp1 * cp1 - cp2 * cp2);

    // Recursion coeffs
    A fac, A1, B1, C1, D1;
    A A0 = cp1 * sp1;
    A B0 = cp2 * sp2;
    A C0 = cp2 * sp2;
    A D0 = cp1 * sp1;

    // Recurse
    for (int u = 0; u < nmax; ++u) {
      A1 = A0;
      B1 = B0;
      C1 = C0;
      D1 = D0;
      for (int v = 2; v < nmax; ++v) {
        fac = 1.0 / (u + v);
        L(u, v) = fac * (A1 - B1 + (v - 1) * L(u, v - 2));
        L(v, u) = fac * (C1 - D1 + (v - 1) * L(v - 2, u));
        A1 *= sp1;
        B1 *= sp2;
        C1 *= cp2;
        D1 *= cp1;
      }
      A0 *= cp1;
      B0 *= cp2;
      C0 *= sp2;
      D0 *= sp1;
    }
  }

  /**
   *
   * Compute the matrix of `L` integrals for the special case
   * phi1 = 0, phi2 = 2 * pi (no occultation).
   *
   */
  inline void compute_L0(Matrix<A> &L) {

    //
    int nmax = deg + 3;
    L.setZero(nmax, nmax);

    // Lower boundary
    L(0, 0) = 2 * pi<Scalar>();

    // Recurse
    A fac;
    for (int u = 0; u < nmax; u += 2) {
      for (int v = 2; v < nmax; v += 2) {
        fac = (v - 1.0) / (u + v);
        L(u, v) = fac * L(u, v - 2);
        L(v, u) = fac * L(v - 2, u);
      }
    }
  }

  /**
   * Nudge the inputs away from singular points
   */
  inline void nudge_inputs() {
    if (abs(bo - ro) < STARRY_BO_EQUALS_RO_TOL)
      bo = ro + (bo > ro ? STARRY_BO_EQUALS_RO_TOL : -STARRY_BO_EQUALS_RO_TOL);
    if ((abs(bo - ro) < STARRY_BO_EQUALS_RO_EQUALS_HALF_TOL) &&
        (abs(ro - 0.5) < STARRY_BO_EQUALS_RO_EQUALS_HALF_TOL))
      bo = ro + STARRY_BO_EQUALS_RO_EQUALS_HALF_TOL;
    if (abs(bo) < STARRY_BO_EQUALS_ZERO_TOL)
      bo = STARRY_BO_EQUALS_ZERO_TOL;
    if ((ro > 0) && (ro < STARRY_RO_EQUALS_ZERO_TOL))
      ro = STARRY_RO_EQUALS_ZERO_TOL;
    if (abs(1 - bo - ro) < STARRY_BO_EQUALS_ONE_MINUS_RO_TOL)
      bo = 1 - ro + STARRY_BO_EQUALS_ONE_MINUS_RO_TOL;
    if (abs(theta - 0.5 * pi<Scalar>()) < STARRY_ROOT_TOL_THETA_PI_TWO) {
      theta += (theta > 0.5 * pi<Scalar>() ? 1.0 : -1.0) *
               STARRY_ROOT_TOL_THETA_PI_TWO;
      costheta = cos(theta);
      sintheta = sin(theta);
    } else if (abs(theta + 0.5 * pi<Scalar>()) < STARRY_ROOT_TOL_THETA_PI_TWO) {
      theta += (theta > -0.5 * pi<Scalar>() ? 1.0 : -1.0) *
               STARRY_ROOT_TOL_THETA_PI_TWO;
      costheta = cos(theta);
      sintheta = sin(theta);
    } else if (abs(sintheta) < STARRY_T_TOL) {
      sintheta = sintheta > 0 ? STARRY_T_TOL : -STARRY_T_TOL;
      theta = costheta > 0 ? 0 : pi<Scalar>();
    }
    if (f < STARRY_MIN_F)
      f = STARRY_MIN_F;
  }

public:
  RowVector<A> sT;
  RowVector<A> sTbar;
  RowVector<A> sT0;
  A phi1, phi2, xi1, xi2;

  explicit Occultation(int deg) : deg(deg), ncoeff((deg + 1) * (deg + 1)) {
    compute_phase();
  }

  /**
      Compute the solution vector in the case
      of no occultation. The phase curve solution
      vector for any value of `f` is just

        sT = (1 - f) * sT0

  */
  inline void compute_phase() {
    compute_L0(Lt);
    sT0.setZero(ncoeff);
    int mu, nu, n = 0;
    for (int l = 0; l < deg + 1; ++l) {
      for (int m = -l; m < l + 1; ++m) {
        mu = l - m;
        nu = l + m;
        if (is_even(nu)) {
          // Case 1
          sT0(n) = Lt(mu / 2 + 2, nu / 2);
        } else if ((l == 1) && (m == 0)) {
          // Case 2
          sT0(n) = 2 * pi<Scalar>() / 3;
        }
        ++n;
      }
    }
  }

  /**
      Compute the full solution vector s^T.


  */
  inline void compute(const A &bo, const A &ro, const A &f, const A &theta) {
    compute_complement(bo, ro, f, theta);
    sT = (1 - f) * sT0 - sTbar;
  }

  /**
      Compute the complement of the solution vector.
      This is the integral *under* the occultor.

  */
  inline void compute_complement(const A &bo_, const A &ro_, const A &f_,
                                 const A &theta_) {
    // Make local copies of the inputs
    // Nudge them away from singular points
    bo = bo_;
    ro = ro_;
    f = f_;
    theta = theta_;
    costheta = cos(theta);
    sintheta = sin(theta);
    nudge_inputs();

    // Compute the angles of intersection
    if (ro == 0) {

      // No occultation
      sTbar.setZero(ncoeff);
      return;

    } else {

      get_angles(bo, ro, f, theta, phi1, phi2, xi1, xi2);
    }

    // Special cases
    if ((phi1 == 0.0) && (phi2 == 0.0) && (xi1 == 0.0) &&
        (xi2 == 2 * pi<Scalar>())) {

      // Complete occultation
      sTbar = (1 - f) * sT0;
      return;

    } else if ((phi1 == 0.0) && (phi2 == 0.0) && (xi1 == 0.0) && (xi2 == 0.0)) {

      // No occultation
      sTbar.setZero(ncoeff);
      return;
    }

    // Compute the M integral analytically (even `mu`)
    S.setZero(deg + 3, deg + 3);
    C.setZero(deg + 3, deg + 3);
    A fac, fac0, facc, facs;
    fac0 = 1.0;
    for (int i = 0; i < deg + 3; ++i) {
      facs = ro * fac0;
      facc = fac0;
      for (int j = 0; j < deg + 3; ++j) {
        if (j < i + 1) {
          S(i, deg + 2 + j - i) = facs;
          C(deg + 2 + j - i, i) = facc;
          fac = bo * (i - j) / (ro * (j + 1.0));
          facs *= fac * sintheta;
          facc *= fac * costheta;
        }
      }
      fac0 *= ro;
    }
    compute_L(phi1 - theta, phi2 - theta, Lp);
    M = S.topRightCorner(deg + 2, deg + 2) * Lp.reverse().topRows(deg + 2) *
        C.leftCols(deg + 2);

    // Compute the K integral numerically (odd `mu`)
    K = pTodd(deg, bo, ro, f, theta, costheta, sintheta, phi1, phi2, QUAD);

    // Compute L (t integral)
    compute_L(xi1, xi2, Lt);

    // Go through the cases
    A pT, tT;
    sTbar.resize(ncoeff);
    int mu, nu, n = 0;
    for (int l = 0; l < deg + 1; ++l) {
      for (int m = -l; m < l + 1; ++m) {
        mu = l - m;
        nu = l + m;

        // Compute the pT and tT integrals
        if (is_even(nu)) {
          // Case 1
          pT = pow(1 - f, -(nu / 2)) * M((mu + 2) / 2, nu / 2);
          tT = (1 - f) * Lt(mu / 2 + 2, nu / 2);
        } else if ((l == 1) && (m == 0)) {
          // Case 2
          pT = K(n);
          tT = (1 - f) * (xi2 - xi1) / 3;
        } else if (is_even(l) && (mu == 1)) {
          // Case 3
          pT = K(n);
          tT = 0.0;
        } else if (!is_even(l) && (mu == 1)) {
          // Case 4
          pT = K(n);
          tT = 0.0;
        } else {
          // Case 5
          pT = K(n);
          tT = 0.0;
        }

        // The surface integral is just their sum
        sTbar(n) = pT + tT;
        ++n;
      }
    }
  }
};

} // namespace occultation
} // namespace oblate
} // namespace starry

#endif