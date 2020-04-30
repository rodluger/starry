
/**
\file occultation.h
\brief Solver for occultations of bodies with a night side (i.e., in reflected
light).

*/

#ifndef _STARRY_REFLECTED_OCCULTATION_H_
#define _STARRY_REFLECTED_OCCULTATION_H_

#include "../basis.h"
#include "../quad.h"
#include "../solver.h"
#include "../utils.h"
#include "constants.h"
#include "geometry.h"
#include "phasecurve.h"
#include "primitive.h"
#include "scatter.h"

namespace starry {
namespace reflected {
namespace occultation {

using namespace utils;
using namespace geometry;
using namespace primitive;

template <class T> class Occultation {

  using Scalar = typename T::Scalar;

protected:
  // Misc
  int deg;
  int deg_lamb;
  int deg_on94;
  int N;
  int N_lamb;
  int N_on94;
  Eigen::SparseMatrix<T> I;
  Vector<T> kappa;
  Vector<T> lam;
  Vector<T> xi;
  Vector<T> PIntegral;
  Vector<T> QIntegral;
  Vector<T> TIntegral;
  RowVector<T> PQT;
  RowVector<T> total_em;

  // Angles
  T costheta;
  T sintheta;
  Vector<T> cosnt;
  Vector<T> sinnt;
  Vector<T> cosmt;
  Vector<T> sinmt;

  // Helper solvers
  basis::Basis<Scalar> B;
  phasecurve::PhaseCurve<T> R;
  solver::Solver<T, true> G_Small; // Lambertian case
  solver::Solver<T, true> G_Big;   // Oren-Nayar case

  // Numerical integration
  quad::Quad<Scalar> QUAD;

  /**
      Weight the solution vector by the illumination profile.
      This profile contains both the cosine illumination *and*
      the scattering law (constant, i.e. isotropic, if sigr == 0).
      Note that we need I to transform Greens --> Greens.

  */
  inline RowVector<T> illuminate(const T &b, const T &theta,
                                 const RowVector<T> &sT, const T &sigr) {
    scatter::computeI(deg, I, b, theta, sigr, B);
    RowVector<T> sTw;
    sTw = sT * B.A2_Reflected.block(0, 0, sT.cols(), sT.cols());
    sTw = sTw * I;
    sTw = sTw * B.A2Inv_Reflected.block(0, 0, sTw.cols(), sTw.cols());
    return sTw;
  }

  /**
      AutoDiff-enabled standard starry occultation solution.

  */
  inline RowVector<T> sTe(const T &bo, const T &ro, const T &sigr) {
    if (sigr > 0) {
      G_Big.compute(bo, ro);
      return G_Big.sT;
    } else {
      G_Small.compute(bo, ro);
      return G_Small.sT;
    }
  }

  /**
   *
  */
  inline RowVector<T> sTr(const T &b, const T &theta, const T &sigr) {

    // Compute the reflection solution in the terminator frame
    R.compute(b, sigr);

    // Transform to ylms and rotate into the occultor frame
    RowVector<T> rTA1 =
        R.rT * B.A1_Reflected.block(0, 0, R.rT.cols(), R.rT.cols());
    RowVector<T> rTA1R(N);
    cosnt(1) = cos(theta);
    sinnt(1) = sin(-theta);
    for (int n = 2; n < deg + 1; ++n) {
      cosnt(n) = 2.0 * cosnt(n - 1) * cosnt(1) - cosnt(n - 2);
      sinnt(n) = 2.0 * sinnt(n - 1) * cosnt(1) - sinnt(n - 2);
    }
    int n = 0;
    for (int l = 0; l < deg + 1; ++l) {
      for (int m = -l; m < 0; ++m) {
        cosmt(n) = cosnt(-m);
        sinmt(n) = -sinnt(-m);
        ++n;
      }
      for (int m = 0; m < l + 1; ++m) {
        cosmt(n) = cosnt(m);
        sinmt(n) = sinnt(m);
        ++n;
      }
      for (int j = 0; j < 2 * l + 1; ++j) {
        rTA1R(l * l + j) = rTA1(l * l + j) * cosmt(l * l + j) +
                           rTA1(l * l + 2 * l - j) * sinmt(l * l + j);
      }
    }

    // Transform back to Green's polynomials
    return rTA1R * B.AInv_Reflected.block(0, 0, rTA1R.cols(), rTA1R.cols());
  }

  /**
    The complement of sTr.
  */
  inline RowVector<T> sTr_hat(const T &b, const T &theta, const T &sigr) {

    // Compute the complement of the reflection
    // solution in the terminator frame.
    int deg_eff;
    if (sigr > 0)
      deg_eff = deg_on94;
    else
      deg_eff = deg_lamb;
    R.compute_unweighted(b, deg_eff);
    scatter::computeI(deg, I, b, T(0.0), sigr, B);
    RowVector<T> rT = -(total_em.segment(0, R.rT0.cols()) - R.rT0) * I;

    // Transform to ylms and rotate into the occultor frame
    RowVector<T> rTA1 = rT * B.A1_Reflected.block(0, 0, rT.cols(), rT.cols());
    RowVector<T> rTA1R(N);
    cosnt(1) = cos(theta);
    sinnt(1) = sin(-theta);
    for (int n = 2; n < deg + 1; ++n) {
      cosnt(n) = 2.0 * cosnt(n - 1) * cosnt(1) - cosnt(n - 2);
      sinnt(n) = 2.0 * sinnt(n - 1) * cosnt(1) - sinnt(n - 2);
    }
    int n = 0;
    for (int l = 0; l < deg + 1; ++l) {
      for (int m = -l; m < 0; ++m) {
        cosmt(n) = cosnt(-m);
        sinmt(n) = -sinnt(-m);
        ++n;
      }
      for (int m = 0; m < l + 1; ++m) {
        cosmt(n) = cosnt(m);
        sinmt(n) = sinnt(m);
        ++n;
      }
      for (int j = 0; j < 2 * l + 1; ++j) {
        rTA1R(l * l + j) = rTA1(l * l + j) * cosmt(l * l + j) +
                           rTA1(l * l + 2 * l - j) * sinmt(l * l + j);
      }
    }

    // Transform back to Green's polynomials
    return rTA1R * B.AInv_Reflected.block(0, 0, rTA1R.cols(), rTA1R.cols());
  }

public:
  int code;
  RowVector<T> sT;

  explicit Occultation(int deg, const basis::Basis<Scalar> &B)
      : deg(deg), deg_lamb(deg + 1), deg_on94(deg + STARRY_OREN_NAYAR_DEG),
        N((deg + 1) * (deg + 1)), N_lamb((deg_lamb + 1) * (deg_lamb + 1)),
        N_on94((deg_on94 + 1) * (deg_on94 + 1)), B(B), R(deg, B),
        G_Small(deg_lamb), G_Big(deg_on94), sT(N) {

    // Rotation vectors
    cosnt.resize(max(2, deg + 1));
    cosnt(0) = 1.0;
    sinnt.resize(max(2, deg + 1));
    sinnt(0) = 0.0;
    cosmt.resize(N);
    sinmt.resize(N);

    // Total flux from the classical starry solution
    basis::computerT(deg_on94, total_em);
  }

  /**
      Compute the full solution vector s^T.

  */
  inline void compute(const T &b, const T &theta, const T &bo, const T &ro,
                      const T &sigr) {

    int deg_eff;
    if (sigr > 0)
      deg_eff = deg_on94;
    else
      deg_eff = deg_lamb;

    // Get the angles of intersection
    costheta = cos(theta);
    sintheta = sin(theta);
    code = get_angles(b, theta, costheta, sintheta, bo, ro, kappa, lam, xi);

    // The full solution vector is a combination of the
    // current vector, the standard starry vector, and the
    // reflected light phase curve solution vector. The contributions
    // of each depend on the integration code.

    if (code == FLUX_ZERO) {

      // Complete occultation!
      sT.setZero(N);

    } else if (code == FLUX_SIMPLE_OCC) {

      // The occultor is blocking all of the nightside
      // and some dayside flux
      sT = illuminate(b, theta, sTe(bo, ro, sigr), sigr);

    } else if (code == FLUX_SIMPLE_REFL) {

      // The total flux is the full dayside flux
      sT = sTr(b, theta, sigr);

    } else if (code == FLUX_SIMPLE_OCC_REFL) {

      // The occultor is only blocking dayside flux
      sT = illuminate(b, theta, sTe(bo, ro, sigr), sigr) +
           sTr_hat(b, theta, sigr);

    } else if (code == FLUX_NOON) {

      // The substellar point is the center of the disk, so this is
      // analytically equivalent to the linear limb darkening solution
      sT = illuminate(b, theta, sTe(bo, ro, sigr), sigr);

    } else {

      // These cases require us to solve incomplete
      // elliptic integrals.

      // Compute the primitive integrals
      computeP(deg_eff, bo, ro, kappa, PIntegral, QUAD);
      computeQ(deg_eff, lam, QIntegral);
      computeT(deg_eff, b, theta, xi, TIntegral);
      PQT = (PIntegral + QIntegral + TIntegral).transpose();

      if ((code == FLUX_DAY_OCC) || (code == FLUX_TRIP_DAY_OCC)) {

        //
        sT = sTr(b, theta, sigr) - illuminate(b, theta, PQT, sigr);

      } else if ((code == FLUX_NIGHT_OCC) || (code == FLUX_TRIP_NIGHT_OCC)) {

        //
        sT = illuminate(b, theta, sTe(bo, ro, sigr) + PQT, sigr) +
             sTr_hat(b, theta, sigr);

      } else if ((code == FLUX_DAY_VIS) || (code == FLUX_QUAD_DAY_VIS)) {

        // The solution vector is *just* the reflected light solution vector.
        sT = illuminate(b, theta, PQT, sigr);

      } else if ((code == FLUX_NIGHT_VIS) || (code == FLUX_QUAD_NIGHT_VIS)) {

        //
        sT = illuminate(b, theta, sTe(bo, ro, sigr) - PQT, sigr);

      } else {

        // ?!
        std::stringstream args;
        args << "b = " << b << ", "
             << "theta = " << theta << ", "
             << "bo = " << bo << ", "
             << "ro = " << ro << ", "
             << "sigr = " << sigr;
        throw StarryException("Unexpected branch.", "reflected/occultation.h",
                              "Occultation.compute", args.str());
      }
    }
  }
};

} // namespace occultation
} // namespace reflected
} // namespace starry

#endif