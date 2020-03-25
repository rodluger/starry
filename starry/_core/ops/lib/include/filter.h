/**
\file filter.h
\brief Map filter operations.

*/

#ifndef _STARRY_FILTER_H_
#define _STARRY_FILTER_H_

#include "basis.h"
#include "utils.h"

namespace starry {
namespace filter {

using namespace utils;

/**
Filter operations on a spherical harmonic map.

*/
template <typename Scalar> class Filter {
protected:
  basis::Basis<Scalar> &B;
  const int ydeg; /**< */
  const int Ny;   /**< Number of spherical harmonic `(l, m)` coefficients */
  const int udeg; /**< */
  const int Nu;   /**< Number of limb darkening coefficients */
  const int fdeg; /**< */
  const int Nf;   /**< Number of filter `(l, m)` coefficients */
  const int deg;  /**< */
  const int N;    /**< */
  const int Nuf;  /**< */
  Vector<Eigen::SparseMatrix<Scalar>>
      DFDp; /**< Deriv of the filter operator w/ respect to the complete filter
               polynomial */

public:
  Matrix<Scalar>
      F; /**< The filter operator in the polynomial basis. TODO: Make sparse? */
  Vector<Scalar> bu;
  Vector<Scalar> bf;

  // Constructor: compute the matrices
  explicit Filter(basis::Basis<Scalar> &B)
      : B(B), ydeg(B.ydeg), Ny((ydeg + 1) * (ydeg + 1)), udeg(B.udeg),
        Nu(udeg + 1), fdeg(B.fdeg), Nf((fdeg + 1) * (fdeg + 1)), deg(B.deg),
        N((deg + 1) * (deg + 1)), Nuf((udeg + fdeg + 1) * (udeg + fdeg + 1)),
        DFDp((udeg + fdeg + 1) * (udeg + fdeg + 1)) {
    // Pre-compute dF / dp
    computePolynomialProductMatrixGradient();
  }

  /**
  Compute the polynomial product matrix.

  */
  inline void computePolynomialProductMatrix(const int plmax,
                                             const Vector<Scalar> &p,
                                             Matrix<Scalar> &M) {
    bool odd1;
    int l, n;
    int n1 = 0, n2 = 0;
    M.setZero((plmax + ydeg + 1) * (plmax + ydeg + 1), Ny);
    for (int l1 = 0; l1 < ydeg + 1; ++l1) {
      for (int m1 = -l1; m1 < l1 + 1; ++m1) {
        odd1 = (l1 + m1) % 2 == 0 ? false : true;
        n2 = 0;
        for (int l2 = 0; l2 < plmax + 1; ++l2) {
          for (int m2 = -l2; m2 < l2 + 1; ++m2) {
            l = l1 + l2;
            n = l * l + l + m1 + m2;
            if (odd1 && ((l2 + m2) % 2 != 0)) {
              M(n - 4 * l + 2, n1) += p(n2);
              M(n - 2, n1) -= p(n2);
              M(n + 2, n1) -= p(n2);
            } else {
              M(n, n1) += p(n2);
            }
            ++n2;
          }
        }
        ++n1;
      }
    }
  }

  /**
  Compute a polynomial product.

  */
  inline void computePolynomialProduct(const int lmax1,
                                       const Vector<Scalar> &p1,
                                       const int lmax2,
                                       const Vector<Scalar> &p2,
                                       Vector<Scalar> &p1p2) {
    int n1, n2, l1, m1, l2, m2, l, n;
    bool odd1;
    p1p2.setZero((lmax1 + lmax2 + 1) * (lmax1 + lmax2 + 1));
    Scalar mult;
    n1 = 0;
    for (l1 = 0; l1 < lmax1 + 1; ++l1) {
      for (m1 = -l1; m1 < l1 + 1; ++m1) {
        odd1 = (l1 + m1) % 2 == 0 ? false : true;
        n2 = 0;
        for (l2 = 0; l2 < lmax2 + 1; ++l2) {
          for (m2 = -l2; m2 < l2 + 1; ++m2) {
            l = l1 + l2;
            n = l * l + l + m1 + m2;
            mult = p1(n1) * p2(n2);
            if (odd1 && ((l2 + m2) % 2 != 0)) {
              p1p2(n - 4 * l + 2) += mult;
              p1p2(n - 2) -= mult;
              p1p2(n + 2) -= mult;
            } else {
              p1p2(n) += mult;
            }
            ++n2;
          }
        }
        ++n1;
      }
    }
  }

  /**
  Compute the gradient of the polynomial product matrix.
  This is independent of the filter polynomials, so we can
  just pre-compute it!

  */
  inline void computePolynomialProductMatrixGradient() {
    bool odd1;
    int l, n;
    int n1 = 0, n2 = 0;
    Vector<Matrix<Scalar>> DFDp_dense(Nuf);
    for (n = 0; n < Nuf; ++n)
      DFDp_dense(n).setZero(N, Ny);
    for (int l1 = 0; l1 < ydeg + 1; ++l1) {
      for (int m1 = -l1; m1 < l1 + 1; ++m1) {
        odd1 = (l1 + m1) % 2 == 0 ? false : true;
        n2 = 0;
        for (int l2 = 0; l2 < udeg + fdeg + 1; ++l2) {
          for (int m2 = -l2; m2 < l2 + 1; ++m2) {
            l = l1 + l2;
            n = l * l + l + m1 + m2;
            if (odd1 && ((l2 + m2) % 2 != 0)) {
              DFDp_dense[n2](n - 4 * l + 2, n1) += 1;
              DFDp_dense[n2](n - 2, n1) -= 1;
              DFDp_dense[n2](n + 2, n1) -= 1;
            } else {
              DFDp_dense[n2](n, n1) += 1;
            }
            ++n2;
          }
        }
        ++n1;
      }
    }
    for (n = 0; n < Nuf; ++n)
      DFDp(n) = DFDp_dense(n).sparseView();
  }

  /**
  Compute the gradient of the polynomial product.

  */
  inline void
  computePolynomialProduct(const int lmax1, const Vector<Scalar> &p1,
                           const int lmax2, const Vector<Scalar> &p2,
                           Matrix<Scalar> &grad_p1, Matrix<Scalar> &grad_p2) {
    int n1, n2, l1, m1, l2, m2, l, n;
    bool odd1;
    int N1 = (lmax1 + 1) * (lmax1 + 1);
    int N2 = (lmax2 + 1) * (lmax2 + 1);
    int N12 = (lmax1 + lmax2 + 1) * (lmax1 + lmax2 + 1);
    n1 = 0;
    grad_p1.setZero(N12, N1);
    grad_p2.setZero(N12, N2);
    for (l1 = 0; l1 < lmax1 + 1; ++l1) {
      for (m1 = -l1; m1 < l1 + 1; ++m1) {
        odd1 = (l1 + m1) % 2 == 0 ? false : true;
        n2 = 0;
        for (l2 = 0; l2 < lmax2 + 1; ++l2) {
          for (m2 = -l2; m2 < l2 + 1; ++m2) {
            l = l1 + l2;
            n = l * l + l + m1 + m2;
            if (odd1 && ((l2 + m2) % 2 != 0)) {
              grad_p1(n - 4 * l + 2, n1) += p2(n2);
              grad_p2(n - 4 * l + 2, n2) += p1(n1);
              grad_p1(n - 2, n1) -= p2(n2);
              grad_p2(n - 2, n2) -= p1(n1);
              grad_p1(n + 2, n1) -= p2(n2);
              grad_p2(n + 2, n2) -= p1(n1);
            } else {
              grad_p1(n, n1) += p2(n2);
              grad_p2(n, n2) += p1(n1);
            }
            ++n2;
          }
        }
        ++n1;
      }
    }
  }

  /**
  Compute the polynomial filter operator.

  */
  void computeF(const Vector<Scalar> &u, const Vector<Scalar> &f) {
    // Compute the two polynomials
    Vector<Scalar> tmp = B.U1 * u;
    Scalar norm =
        Scalar(1.0) / B.rT.segment(0, (udeg + 1) * (udeg + 1)).dot(tmp);
    Vector<Scalar> pu = tmp * norm * pi<Scalar>();
    Vector<Scalar> pf;
    pf = B.A1_f * f;

    // Multiply them
    Vector<Scalar> p;
    if (udeg > fdeg) {
      computePolynomialProduct(udeg, pu, fdeg, pf, p);
    } else {
      computePolynomialProduct(fdeg, pf, udeg, pu, p);
    }

    // Compute the polynomial filter operator
    computePolynomialProductMatrix(udeg + fdeg, p, F);
  }

  /**
  Compute the gradient of the polynomial filter operator.

  */
  void computeF(const Vector<Scalar> &u, const Vector<Scalar> &f,
                const Matrix<Scalar> &bF) {
    Matrix<Scalar> DpDpu;
    Matrix<Scalar> DpDpf;

    // Compute the two polynomials
    Vector<Scalar> tmp = B.U1 * u;
    Scalar norm =
        Scalar(1.0) / B.rT.segment(0, (udeg + 1) * (udeg + 1)).dot(tmp);
    Vector<Scalar> pu = tmp * norm * pi<Scalar>();
    Vector<Scalar> pf;
    pf = B.A1_f * f;

    // Multiply them
    // TODO: DpDpf and DpDpu are sparse, and we should probably exploit that
    Vector<Scalar> p;
    if (udeg > fdeg) {
      computePolynomialProduct(udeg, pu, fdeg, pf, DpDpu, DpDpf);
    } else {
      computePolynomialProduct(fdeg, pf, udeg, pu, DpDpf, DpDpu);
    }

    // Backprop p
    RowVector<Scalar> bp(Nuf);
    for (int j = 0; j < Nuf; ++j)
      bp(j) = DFDp(j).cwiseProduct(bF).sum();

    // Compute the limb darkening derivatives
    Matrix<Scalar> DpuDu =
        pi<Scalar>() * norm * B.U1 -
        pu * B.rT.segment(0, (udeg + 1) * (udeg + 1)) * B.U1 * norm;
    bu = bp * DpDpu * DpuDu;

    // Compute the Ylm filter derivatives
    bf = bp * DpDpf * B.A1_f;
  }
};

} // namespace filter
} // namespace starry
#endif