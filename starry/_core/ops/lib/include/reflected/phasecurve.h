/**
\file solver.h
\brief Computes surface integrals over reflected-light maps.

*/

#ifndef _STARRY_REFLECTED_PHASECURVE_H_
#define _STARRY_REFLECTED_PHASECURVE_H_

#include "../basis.h"
#include "../ellip.h"
#include "../utils.h"
#include "scatter.h"

namespace starry {
namespace reflected {
namespace phasecurve {

using namespace starry::utils;

/**
  Greens integral solver wrapper class. Reflected
  light phase curves specialization.

*/
template <class T> class PhaseCurve {

protected:
  const int deg;
  const int deg_lamb;
  const int deg_on94;
  const int N;
  const int N_lamb;
  const int N_on94;
  Vector<T> hj;
  Vector<T> kj;
  Matrix<T> Lij;
  Matrix<T> Mij;
  Eigen::SparseMatrix<T> ILLUM;
  basis::Basis<typename T::Scalar> B;
  T tol;

  /**
    Computes the matrices

        Lij = (G(0.5 * (i + 1)) * G(0.5 * (j + 1))) / G(0.5 * (i + j + 4))

    and

        Mij = (G(0.5 * (i + 1)) * G(0.5 * (j + 4))) / G(0.5 * (i + j + 5))

    where `G` is the Gamma function.

  */
  inline void computeJK(const int deg_eff) {
    Lij.setZero(deg_eff + 1, deg_eff + 1);
    Mij.setZero(deg_eff + 1, deg_eff + 1);
    Lij(0, 0) = pi<T>();
    Lij(0, 1) = T(4.0) / T(3.0);
    Mij(0, 0) = Lij(0, 1);
    Mij(0, 1) = 0.375 * Lij(0, 0);
    for (int j = 0; j < deg_eff - 1; ++j) {
      Lij(0, j + 2) = Lij(0, j) * T(j + 1.0) / T(j + 4.0);
      Mij(0, j + 2) = Mij(0, j) * T(j + 4.0) / T(j + 5.0);
      for (int i = 0; i < deg_eff - 1; i += 2) {
        Lij(i + 2, j) = Lij(i, j) * T(i + 1.0) / T(i + j + 4.0);
        Mij(i + 2, j) = Mij(i, j) * T(i + 1.0) / T(i + j + 5.0);
      }
    }
    for (int j = deg_eff - 1; j < deg_eff + 1; ++j) {
      for (int i = 0; i < deg_eff - 1; i += 2) {
        Lij(i + 2, j) = Lij(i, j) * T(i + 1.0) / T(i + j + 4.0);
        Mij(i + 2, j) = Mij(i, j) * T(i + 1.0) / T(i + j + 5.0);
      }
    }
  }

  /*
    Computes the arrays

        hj = 0.5 * (1 - b ** (j + 1))

    and

        kj = int_b^1 a^j (1 - a^2)^(1/2) da

  */
  inline void computeHI(const T &bterm, const int deg_eff) {
    hj.setZero(deg_eff + 1);
    kj.setZero(deg_eff + 1);
    T fac0 = sqrt(T(1.0) - bterm * bterm);
    T fac1 = (T(1.0) - bterm * bterm) * fac0;
    kj(0) = 0.5 * (acos(bterm) - bterm * fac0);
    kj(1) = fac1 / T(3.0);
    T fac2 = bterm;
    hj(0) = 0.5 * (T(1.0) - fac2);
    fac2 *= bterm;
    hj(1) = 0.5 * (T(1.0) - fac2);
    fac1 *= bterm;
    fac2 *= bterm;
    for (int j = 0; j < deg_eff - 1; ++j) {
      kj(j + 2) = T(1.0) / T(j + 4.0) * (fac1 + (j + 1) * kj(j));
      hj(j + 2) = 0.5 * (T(1.0) - fac2);
      fac1 *= bterm;
      fac2 *= bterm;
    }
  }

public:
  RowVector<T> rT0;
  RowVector<T> rT;

  /**
    Computes the unweighted reflectance integrals.

  */
  inline void compute_unweighted(const T &bterm, const int deg_eff) {
    computeHI(bterm, deg_eff);
    rT0.setZero((deg_eff + 1) * (deg_eff + 1));
    int n = 0;
    int i, j;
    int mu, nu;
    for (int l = 0; l < deg_eff + 1; ++l) {
      for (int m = -l; m < l + 1; ++m) {
        mu = l - m;
        nu = l + m;
        if (is_even(nu)) {
          i = mu / 2;
          j = nu / 2;
          rT0(n) = hj(j) * Lij(i, j);
        } else {
          i = (mu - 1) / 2;
          j = (nu - 1) / 2;
          rT0(n) = kj(j) * Mij(i, j);
        }
        ++n;
      }
    }
  }

  /**
    Computes the reflectance integrals weighted
    by the illumination function.

  */
  inline void compute(const T &bterm, const T &sigr) {
    if (sigr > 0.0)
      compute_unweighted(bterm, deg_on94);
    else
      compute_unweighted(bterm, deg_lamb);
    scatter::computeI(deg, ILLUM, bterm, T(0.0), sigr, B);
    rT = rT0 * ILLUM;
  }

  explicit PhaseCurve(int deg, const basis::Basis<typename T::Scalar> &B)
      : deg(deg), deg_lamb(deg + 1), deg_on94(deg + STARRY_OREN_NAYAR_DEG),
        N((deg + 1) * (deg + 1)), N_lamb((deg_lamb + 1) * (deg_lamb + 1)),
        N_on94((deg_on94 + 1) * (deg_on94 + 1)), B(B), tol(sqrt(mach_eps<T>())),
        rT(N) {
    // Pre-compute the Lij and Mij matrices for the largest case
    computeJK(deg_on94);
  }
};

} // namespace phasecurve
} // namespace reflected
} // namespace starry

#endif