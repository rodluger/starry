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
template <typename Scalar>
class Filter {
 protected:
  basis::Basis<Scalar> &B;
  const int ydeg;    /**< */
  const int Ny;      /**< Number of spherical harmonic `(l, m)` coefficients */
  const int udeg;    /**< */
  const int Nu;      /**< Number of limb darkening coefficients */
  const int fdeg;    /**< */
  const int Nf;      /**< Number of filter `(l, m)` coefficients */
  const int deg;     /**< */
  const int N;       /**< */
  const int Nuf;     /**< */
  const int drorder; /**< Order of the diff rot operator */
  Vector<Eigen::SparseMatrix<Scalar>>
      DFDp; /**< Deriv of the filter operator w/ respect to the complete filter
               polynomial */

 public:
  Matrix<Scalar>
      F; /**< The filter operator in the polynomial basis. TODO: Make sparse? */
  Vector<Scalar> bu;
  Vector<Scalar> bf;

  std::vector<Matrix<Scalar>>
      D; /**< The differential rotation operator. TODO: Make sparse */

  // Constructor: compute the matrices
  explicit Filter(basis::Basis<Scalar> &B, const int &drorder) :
      B(B), ydeg(B.ydeg), Ny((ydeg + 1) * (ydeg + 1)), udeg(B.udeg),
      Nu(udeg + 1), fdeg(B.fdeg), Nf((fdeg + 1) * (fdeg + 1)), deg(B.deg),
      N((deg + 1) * (deg + 1)), Nuf((udeg + fdeg + 1) * (udeg + fdeg + 1)),
      drorder(drorder), DFDp((udeg + fdeg + 1) * (udeg + fdeg + 1)) {
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
    for (n = 0; n < Nuf; ++n) DFDp_dense(n).setZero(N, Ny);
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
    for (n = 0; n < Nuf; ++n) DFDp(n) = DFDp_dense(n).sparseView();
  }

  /**
  Compute the gradient of the polynomial product.

  */
  inline void computePolynomialProduct(const int lmax1,
                                       const Vector<Scalar> &p1,
                                       const int lmax2,
                                       const Vector<Scalar> &p2,
                                       Matrix<Scalar> &grad_p1,
                                       Matrix<Scalar> &grad_p2) {
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
    for (int j = 0; j < Nuf; ++j) bp(j) = DFDp(j).cwiseProduct(bF).sum();

    // Compute the limb darkening derivatives
    Matrix<Scalar> DpuDu =
        pi<Scalar>() * norm * B.U1 -
        pu * B.rT.segment(0, (udeg + 1) * (udeg + 1)) * B.U1 * norm;
    bu = bp * DpDpu * DpuDu;

    // Compute the Ylm filter derivatives
    bf = bp * DpDpf * B.A1_f;
  }

  /**
  Compute a sparse polynomial product using Eigen Triplets.

  */
  inline void computeSparsePolynomialProduct(
      const std::vector<Eigen::Triplet<Scalar>> &p1,
      const std::vector<Eigen::Triplet<Scalar>> &p2,
      std::vector<Eigen::Triplet<Scalar>> &p1p2) {
    int l1, m1, l2, m2;
    bool odd1;
    Scalar prod;
    p1p2.clear();
    for (Eigen::Triplet<Scalar> t1 : p1) {
      l1 = t1.row();
      m1 = t1.col();
      odd1 = (l1 + m1) % 2 == 0 ? false : true;
      for (Eigen::Triplet<Scalar> t2 : p2) {
        l2 = t2.row();
        m2 = t2.col();
        prod = t1.value() * t2.value();
        if (odd1 && ((l2 + m2) % 2 != 0)) {
          p1p2.push_back(Eigen::Triplet<Scalar>(l1 + l2 - 2, m1 + m2, prod));
          p1p2.push_back(Eigen::Triplet<Scalar>(l1 + l2, m1 + m2 - 2, -prod));
          p1p2.push_back(Eigen::Triplet<Scalar>(l1 + l2, m1 + m2 + 2, -prod));
        } else {
          p1p2.push_back(Eigen::Triplet<Scalar>(l1 + l2, m1 + m2, prod));
        }
      }
    }
  }

  /**
  Compute the differential rotation operator. (Slow, dense, deprecated)

  */
  void computeD_Working_But_Very_Slow(const Vector<Scalar> &wta) {
    // TODO: Declare up top
    int ddeg = 4 * drorder;
    int Ddeg = (ddeg + 1) * ydeg;
    int ND = (Ddeg + 1) * (Ddeg + 1);

    // Sine and cosine expansions
    Vector<Scalar> c((ddeg + 1) * (ddeg + 1));
    c.setZero();
    Vector<Scalar> s((ddeg + 1) * (ddeg + 1));
    s.setZero();

    // x and z polynomials
    Vector<Scalar> x(4), z(4), y(4), xc((ddeg + 2) * (ddeg + 2)),
        zc((ddeg + 2) * (ddeg + 2)), xs((ddeg + 2) * (ddeg + 2)),
        zs((ddeg + 2) * (ddeg + 2)), xD((ddeg + 2) * (ddeg + 2)),
        zD((ddeg + 2) * (ddeg + 2));
    Vector<Scalar> tmp;
    x << 0, 1, 0, 0;
    z << 0, 0, 1, 0;
    y << 0, 0, 0, 1;

    // Loop over all times
    D.resize(wta.size());
    for (int i = 0; i < wta.size(); ++i) {
      // (omega * t * alpha)^2
      Scalar wtai_2 = wta(i) * wta(i);

      // Cosine expansions
      Scalar fac = 1.0;
      for (int l = 0; l < ddeg + 1; l += 4) {
        c((l + 1) * (l + 1) - 1) = fac;
        fac *= -(4.0 * wtai_2) / ((l + 4.0) * (l + 2.0));
      }
      computePolynomialProduct(ddeg, c, 1, x, xc);
      computePolynomialProduct(ddeg, c, 1, z, zc);

      // Sine expansions
      fac = wta(i);
      for (int l = 2; l < ddeg + 1; l += 4) {
        s((l + 1) * (l + 1) - 1) = fac;
        fac *= -(4.0 * wtai_2) / ((l + 4.0) * (l + 2.0));
      }
      computePolynomialProduct(ddeg, s, 1, x, xs);
      computePolynomialProduct(ddeg, s, 1, z, zs);

      // Differentially-rotated x and z terms
      xD = xc - zs;
      zD = xs + zc;

      // Construct the matrix
      D[i].setZero(ND, Ny);

      // l = 0 and l = 1
      // TODO: Assert l > 0
      D[i](0, 0) = 1.0;
      D[i].col(1).segment(0, (ddeg + 2) * (ddeg + 2)) = xD;
      D[i].col(2).segment(0, (ddeg + 2) * (ddeg + 2)) = zD;
      D[i](3, 3) = 1.0;

      // Loop over the remaining degrees
      int np, nc, n;
      for (int l = 2; l < ydeg + 1; ++l) {
        // First index of previous, current, and next degree
        np = (l - 1) * (l - 1);
        nc = l * l;

        // Multiply every term of the previous degree by xD
        n = nc;
        for (int j = np; j < nc; ++j) {
          computePolynomialProduct((ddeg + 1) * (l - 1), D[i].col(j), ddeg + 1,
                                   xD, tmp);
          D[i].col(n).segment(0, tmp.size()) = tmp;
          ++n;
        }

        // The last two terms of this degree
        computePolynomialProduct((ddeg + 1) * (l - 1), D[i].col(nc - 1), 1, y,
                                 tmp);
        D[i].col(n + 1).segment(0, tmp.size()) = tmp;
        computePolynomialProduct((ddeg + 1) * (l - 1), D[i].col(nc - 2), 1, y,
                                 tmp);
        D[i].col(n).segment(0, tmp.size()) = tmp;
      }
    }
  }

  /**
  Compute the differential rotation operator.

  TODO: Assert l > 0

  */
  void computeD(const Vector<Scalar> &wta) {
    // TODO: Declare all of this up top
    int ddeg = 4 * drorder;
    int Ddeg = (ddeg + 1) * ydeg;
    int ND = (Ddeg + 1) * (Ddeg + 1);
    Scalar fac;

    std::vector<Eigen::Triplet<Scalar>> t_1, t_x, t_z, t_neg_z, t_y;
    std::vector<Eigen::Triplet<Scalar>> t_c, t_s;
    std::vector<Eigen::Triplet<Scalar>> t_xc, t_zc, t_xs, t_zs, t_neg_zs;
    std::vector<Eigen::Triplet<Scalar>> t_xD, t_zD;

    t_1.push_back(Eigen::Triplet<Scalar>(0, 0, 1));
    t_x.push_back(Eigen::Triplet<Scalar>(1, -1, 1));
    t_z.push_back(Eigen::Triplet<Scalar>(1, 0, 1));
    t_neg_z.push_back(Eigen::Triplet<Scalar>(1, 0, -1));
    t_y.push_back(Eigen::Triplet<Scalar>(1, 1, 1));

    Eigen::SparseMatrix<Scalar> A1, A1Inv;
    basis::computeA1(Ddeg, A1, B.norm);
    basis::computeA1Inv(Ddeg, A1, A1Inv);
    A1Inv = A1Inv.topRows(Ny);

    // Loop over all times
    D.resize(wta.size());

    for (int i = 0; i < wta.size(); ++i) {
      Scalar wtai_2 = wta(i) * wta(i);
      t_c.clear();
      t_s.clear();
      t_xD.clear();
      t_zD.clear();

      // Cosine expansion
      fac = 1.0;
      for (int l = 0; l < ddeg + 1; l += 4) {
        t_c.push_back(Eigen::Triplet<Scalar>(l, l, fac));
        fac *= -(4.0 * wtai_2) / ((l + 4.0) * (l + 2.0));
      }
      computeSparsePolynomialProduct(t_x, t_c, t_xc);
      computeSparsePolynomialProduct(t_z, t_c, t_zc);

      // Sine expansion
      fac = wta(i);
      for (int l = 2; l < ddeg + 1; l += 4) {
        t_s.push_back(Eigen::Triplet<Scalar>(l, l, fac));
        fac *= -(4.0 * wtai_2) / ((l + 4.0) * (l + 2.0));
      }
      computeSparsePolynomialProduct(t_x, t_s, t_xs);
      computeSparsePolynomialProduct(t_z, t_s, t_zs);
      computeSparsePolynomialProduct(t_neg_z, t_s, t_neg_zs);

      // Differentially-rotated x and z terms
      for (Eigen::Triplet<Scalar> term : t_xc) t_xD.push_back(term);
      for (Eigen::Triplet<Scalar> term : t_neg_zs) t_xD.push_back(term);
      for (Eigen::Triplet<Scalar> term : t_xs) t_zD.push_back(term);
      for (Eigen::Triplet<Scalar> term : t_zc) t_zD.push_back(term);

      // Construct the matrix
      std::vector<std::vector<Eigen::Triplet<Scalar>>> t_D;
      t_D.resize(Ny);

      // l = 0
      t_D[0] = t_1;

      // l = 1
      t_D[1] = t_xD;
      t_D[2] = t_zD;
      t_D[3] = t_y;

      // Loop over the remaining degrees
      int np, nc, n;
      for (int l = 2; l < ydeg + 1; ++l) {
        // First index of previous & current degree
        np = (l - 1) * (l - 1);
        nc = l * l;

        // Multiply every term of the previous degree by xD
        n = nc;
        for (int j = np; j < nc; ++j) {
          computeSparsePolynomialProduct(t_D[j], t_xD, t_D[n]);
          ++n;
        }

        // The last two terms of this degree
        computeSparsePolynomialProduct(t_D[nc - 1], t_y, t_D[n + 1]);
        computeSparsePolynomialProduct(t_D[nc - 2], t_y, t_D[n]);
      }

      // Construct the sparse D operator from the triplets
      std::vector<Eigen::Triplet<Scalar>> coeffs;
      for (int col = 0; col < Ny; ++col) {
        for (Eigen::Triplet<Scalar> term : t_D[col]) {
          int l = term.row();
          int m = term.col();
          int row = l * l + l + m;
          coeffs.push_back(Eigen::Triplet<Scalar>(row, col, term.value()));
        }
      }
      Eigen::SparseMatrix<Scalar> spD(ND, Ny);
      spD.setFromTriplets(coeffs.begin(), coeffs.end());

      // Rotate into Ylm space
      spD = A1Inv * spD * B.A1;

      D[i] = Matrix<Scalar>(spD);
    }
  }
};

}  // namespace filter
}  // namespace starry
#endif