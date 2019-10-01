/**
\file diffrot.h
\brief Differential rotation operator.

*/

#ifndef _STARRY_DIFFROT_H_
#define _STARRY_DIFFROT_H_

#include "basis.h"
#include "utils.h"

namespace starry {
namespace diffrot {

using namespace utils;

/**
Differential rotation operator class.

*/
template <typename Scalar>
class DiffRot {
 protected:
  using Triplet = Eigen::Triplet<Scalar>;
  using Triplets = std::vector<Triplet>;

  basis::Basis<Scalar> &B;
  const int ydeg;    /**< */
  const int Ny;      /**< Number of spherical harmonic `(l, m)` coefficients */
  const int drorder; /**< Order of the diff rot operator */
  const int ddeg;
  const int Ddeg;
  const int ND;

  Triplets t_1, t_x, t_z, t_neg_z, t_y;
  Triplets t_c, t_s;
  Triplets t_xc, t_zc, t_xs, t_zs, t_neg_zs;
  Triplets t_xD, t_zD;
  Triplets coeffs;
  std::vector<Triplets> t_D;

  Eigen::SparseMatrix<Scalar> A1, A1Inv;
  Eigen::SparseMatrix<Scalar> D;

 public:
  Matrix<Scalar> tensordotD_result;

  // Constructor: compute the matrices
  explicit DiffRot(basis::Basis<Scalar> &B, const int &drorder) :
      B(B), ydeg(B.ydeg), Ny((ydeg + 1) * (ydeg + 1)), drorder(drorder),
      ddeg(4 * drorder), Ddeg((ddeg + 1) * ydeg), ND((Ddeg + 1) * (Ddeg + 1)) {
    // Initialize the constant triplets
    t_1.push_back(Triplet(0, 0, 1));
    t_x.push_back(Triplet(1, -1, 1));
    t_z.push_back(Triplet(1, 0, 1));
    t_neg_z.push_back(Triplet(1, 0, -1));
    t_y.push_back(Triplet(1, 1, 1));

    // Pre-compute the change-of-basis matrices
    // A1 is (Ny x Ny) as usual, but we need
    // A1Inv to be (Ny x ND)
    basis::computeA1(Ddeg, A1, B.norm);
    basis::computeA1Inv(Ddeg, A1, A1Inv);
    A1Inv = A1Inv.topRows(Ny);
    A1 = B.A1;

    // Initialize the sparse D matrix
    D.resize(ND, Ny);
  }

  /**
  Compute a sparse polynomial product using Eigen Triplets.

  */
  inline void computeSparsePolynomialProduct(const Triplets &p1,
                                             const Triplets &p2,
                                             Triplets &p1p2) {
    int l1, m1, l2, m2;
    bool odd1;
    Scalar prod;
    p1p2.clear();
    for (Triplet t1 : p1) {
      l1 = t1.row();
      m1 = t1.col();
      odd1 = (l1 + m1) % 2 == 0 ? false : true;
      for (Triplet t2 : p2) {
        l2 = t2.row();
        m2 = t2.col();
        prod = t1.value() * t2.value();
        if (odd1 && ((l2 + m2) % 2 != 0)) {
          p1p2.push_back(Triplet(l1 + l2 - 2, m1 + m2, prod));
          p1p2.push_back(Triplet(l1 + l2, m1 + m2 - 2, -prod));
          p1p2.push_back(Triplet(l1 + l2, m1 + m2 + 2, -prod));
        } else {
          p1p2.push_back(Triplet(l1 + l2, m1 + m2, prod));
        }
      }
    }
  }

  /**
  Apply the differential rotation operator to a matrix `M` on the right.

  */
  template <typename T1>
  void tensordotD(const MatrixBase<T1> &M, const Vector<Scalar> &wta) {
    // Pathological case
    if (ydeg == 0) {
      tensordotD_result = M;
      return;
    }

    // Size checks
    size_t npts = wta.size();
    if (((size_t)M.rows() != npts) || ((int)M.cols() != Ny))
      throw std::runtime_error("Incompatible shapes in `tensordotD`.");

    // Rotate the matrix into polynomial space
    Matrix<Scalar> MA1Inv = M * A1Inv;
    Matrix<Scalar> MA1InvD(npts, Ny);

    // Loop over all times
    for (int i = 0; i < wta.size(); ++i) {
      Scalar wtai_2 = wta(i) * wta(i);
      t_c.clear();
      t_s.clear();
      t_xD.clear();
      t_zD.clear();

      // Cosine expansion
      Scalar fac = 1.0;
      for (int l = 0; l < ddeg + 1; l += 4) {
        t_c.push_back(Triplet(l, l, fac));
        fac *= -(4.0 * wtai_2) / ((l + 4.0) * (l + 2.0));
      }
      computeSparsePolynomialProduct(t_x, t_c, t_xc);
      computeSparsePolynomialProduct(t_z, t_c, t_zc);

      // Sine expansion
      fac = wta(i);
      for (int l = 2; l < ddeg + 1; l += 4) {
        t_s.push_back(Triplet(l, l, fac));
        fac *= -(4.0 * wtai_2) / ((l + 4.0) * (l + 2.0));
      }
      computeSparsePolynomialProduct(t_x, t_s, t_xs);
      computeSparsePolynomialProduct(t_z, t_s, t_zs);
      computeSparsePolynomialProduct(t_neg_z, t_s, t_neg_zs);

      // Differentially-rotated x and z terms
      for (Triplet term : t_xc) t_xD.push_back(term);
      for (Triplet term : t_neg_zs) t_xD.push_back(term);
      for (Triplet term : t_xs) t_zD.push_back(term);
      for (Triplet term : t_zc) t_zD.push_back(term);

      // Construct the matrix
      t_D.clear();
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
      coeffs.clear();
      for (int col = 0; col < Ny; ++col) {
        for (Triplet term : t_D[col]) {
          int l = term.row();
          int m = term.col();
          int row = l * l + l + m;
          coeffs.push_back(Triplet(row, col, term.value()));
        }
      }
      D.setFromTriplets(coeffs.begin(), coeffs.end());

      // Dot it into the current row
      MA1InvD.row(i) = MA1Inv.row(i) * D;
    }

    // Rotate fully to Ylm space
    tensordotD_result = MA1InvD * B.A1;
  }
};

}  // namespace diffrot
}  // namespace starry
#endif