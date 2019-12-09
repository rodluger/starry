/**
\file solver_reflected.h
\brief Computes surface integrals over reflected-light maps.

*/

#ifndef _STARRY_SOLVER_REFL_H_
#define _STARRY_SOLVER_REFL_H_

#include "ellip.h"
#include "utils.h"

namespace starry {
namespace solver {

using namespace starry::utils;

/**
Greens integral solver wrapper class. Reflected
light specialization.

*/
template <class Scalar> class GreensReflected {

protected:
  const int lmax;
  const int N;
  Vector<Scalar> H;
  Vector<Scalar> I;
  Vector<Scalar> DHDb;
  Vector<Scalar> DIDb;
  Matrix<Scalar> J;
  Matrix<Scalar> K;
  Scalar tol;

  /**
  Computes the matrices

      J = (G(0.5 * (i + 1)) * G(0.5 * (j + 1))) / G(0.5 * (i + j + 4))

  and

      K = (G(0.5 * (i + 1)) * G(0.5 * (j + 4))) / G(0.5 * (i + j + 5))

  where `G` is the Gamma function.

  */
  inline void computeJK() {
    J.setZero();
    K.setZero();
    J(0, 0) = pi<Scalar>();
    J(0, 1) = Scalar(4.0) / Scalar(3.0);
    K(0, 0) = J(0, 1);
    K(0, 1) = 0.375 * J(0, 0);
    for (int j = 0; j < lmax + 1; ++j) {
      J(0, j + 2) = J(0, j) * Scalar(j + 1.0) / Scalar(j + 4.0);
      K(0, j + 2) = K(0, j) * Scalar(j + 4.0) / Scalar(j + 5.0);
      for (int i = 0; i < lmax + 1; i += 2) {
        J(i + 2, j) = J(i, j) * Scalar(i + 1.0) / Scalar(i + j + 4.0);
        K(i + 2, j) = K(i, j) * Scalar(i + 1.0) / Scalar(i + j + 5.0);
      }
    }
    for (int j = lmax + 1; j < lmax + 3; ++j) {
      for (int i = 0; i < lmax + 1; i += 2) {
        J(i + 2, j) = J(i, j) * Scalar(i + 1.0) / Scalar(i + j + 4.0);
        K(i + 2, j) = K(i, j) * Scalar(i + 1.0) / Scalar(i + j + 5.0);
      }
    }
  }

  /*
  Computes the arrays

      H = 0.5 * (1 - b ** (j + 1))

  and

      I = int_b^1 a^j (1 - a^2)^(1/2) da

  */
  inline void computeHI(const Scalar &bterm) {
    H.setZero();
    I.setZero();
    Scalar fac0 = sqrt(Scalar(1.0) - bterm * bterm);
    Scalar fac1 = (Scalar(1.0) - bterm * bterm) * fac0;
    I(0) = 0.5 * (acos(bterm) - bterm * fac0);
    I(1) = fac1 / Scalar(3.0);
    Scalar fac2 = bterm;
    H(0) = 0.5 * (Scalar(1.0) - fac2);
    fac2 *= bterm;
    H(1) = 0.5 * (Scalar(1.0) - fac2);
    fac1 *= bterm;
    fac2 *= bterm;
    for (int j = 0; j < lmax + 1; ++j) {
      I(j + 2) = Scalar(1.0) / Scalar(j + 4.0) * (fac1 + (j + 1) * I(j));
      H(j + 2) = 0.5 * (Scalar(1.0) - fac2);
      fac1 *= bterm;
      fac2 *= bterm;
    }
  }

  inline void computeHI_with_grad(const Scalar &bterm) {
    H.setZero();
    I.setZero();
    DHDb.setZero();
    DIDb.setZero();
    Scalar fac0 = sqrt(Scalar(1.0) - bterm * bterm);
    Scalar fac1 = (Scalar(1.0) - bterm * bterm) * fac0;
    Scalar Dfac1Db = -3.0 * bterm * fac0;
    I(0) = 0.5 * (acos(bterm) - bterm * fac0);
    DIDb(0) = -fac0;
    I(1) = fac1 / Scalar(3.0);
    DIDb(1) = Dfac1Db / Scalar(3.0);
    Scalar fac2 = bterm;
    Scalar Dfac2Db = Scalar(1.0);
    H(0) = 0.5 * (Scalar(1.0) - fac2);
    DHDb(0) = -0.5 * Dfac2Db;
    Dfac2Db = fac2 + Dfac2Db * bterm;
    fac2 *= bterm;
    H(1) = 0.5 * (Scalar(1.0) - fac2);
    DHDb(1) = -0.5 * Dfac2Db;
    Dfac1Db = fac1 + Dfac1Db * bterm;
    fac1 *= bterm;
    Dfac2Db = fac2 + Dfac2Db * bterm;
    fac2 *= bterm;
    for (int j = 0; j < lmax + 1; ++j) {
      I(j + 2) = Scalar(1.0) / Scalar(j + 4.0) * (fac1 + (j + 1) * I(j));
      DIDb(j + 2) =
          Scalar(1.0) / Scalar(j + 4.0) * (Dfac1Db + (j + 1) * DIDb(j));
      H(j + 2) = 0.5 * (Scalar(1.0) - fac2);
      DHDb(j + 2) = -0.5 * Dfac2Db;
      Dfac1Db = fac1 + Dfac1Db * bterm;
      fac1 *= bterm;
      Dfac2Db = fac2 + Dfac2Db * bterm;
      fac2 *= bterm;
    }
  }

public:
  RowVector<Scalar> rT;

  /**
  Computes the complete reflectance integrals.

  */
  inline void compute(const Scalar &bterm) {
    computeHI(bterm);
    rT.setZero();
    Scalar fac = sqrt(Scalar(1.0) - bterm * bterm);
    int n = 0;
    int i, j;
    int mu, nu;
    for (int l = 0; l < lmax + 1; ++l) {
      for (int m = -l; m < l + 1; ++m) {
        mu = l - m;
        nu = l + m;
        if (is_even(nu)) {
          i = mu / 2;
          j = nu / 2;
          rT(n) = fac * H(j + 1) * J(i, j + 1) - bterm * I(j) * K(i, j);
        } else {
          i = (mu - 1) / 2;
          j = (nu - 1) / 2;
          rT(n) = fac * I(j + 1) * K(i, j + 1) -
                  bterm * (H(j) * J(i, j) - H(j) * J(i + 2, j) -
                           H(j + 2) * J(i, j + 2));
        }
        ++n;
      }
    }
  }

  /**
  Computes the (backprop) gradient of the complete reflectance integrals.

  */
  inline Scalar compute(const Scalar &bterm, const RowVector<Scalar> &brT) {
    Scalar bb = 0.0;
    computeHI_with_grad(bterm);
    // TODO: The gradient is infinite when bterm = +/- 1
    // Not sure how best to handle this.
    Scalar fac = sqrt(max(Scalar(1.0) - bterm * bterm, tol));
    Scalar DfacDb = -bterm / fac;
    int n = 0;
    int i, j;
    int mu, nu;

    for (int l = 0; l < lmax + 1; ++l) {
      for (int m = -l; m < l + 1; ++m) {
        mu = l - m;
        nu = l + m;
        if (is_even(nu)) {
          i = mu / 2;
          j = nu / 2;
          bb += ((DfacDb * H(j + 1) + fac * DHDb(j + 1)) * J(i, j + 1) -
                 (I(j) + bterm * DIDb(j)) * K(i, j)) *
                brT(n);
        } else {
          i = (mu - 1) / 2;
          j = (nu - 1) / 2;
          bb +=
              ((DfacDb * I(j + 1) + fac * DIDb(j + 1)) * K(i, j + 1) -
               ((H(j) * J(i, j) - H(j) * J(i + 2, j) - H(j + 2) * J(i, j + 2)) +
                bterm * (DHDb(j) * J(i, j) - DHDb(j) * J(i + 2, j) -
                         DHDb(j + 2) * J(i, j + 2)))) *
              brT(n);
        }
        ++n;
      }
    }
    return bb;
  }

  explicit GreensReflected(int lmax)
      : lmax(lmax), N((lmax + 1) * (lmax + 1)), H(lmax + 3), I(lmax + 3),
        DHDb(lmax + 3), DIDb(lmax + 3), J(lmax + 3, lmax + 3),
        K(lmax + 3, lmax + 3), tol(sqrt(mach_eps<Scalar>())), rT(N) {
    // Pre-compute the J and K matrices
    computeJK();
  }
};

} // namespace solver
} // namespace starry

#endif