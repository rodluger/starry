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

  Triplets t_0, t_1, t_x, t_z, t_neg_z, t_y;
  Triplets t_c, t_s, t_dc, t_ds;
  Triplets t_xc, t_zc, t_xs, t_zs, t_neg_zs;
  Triplets t_dxc, t_dzc, t_dxs, t_dzs, t_neg_dzs;
  Triplets t_xD, t_zD, t_dxD, t_dzD;
  Triplets coeffs, dcoeffs;
  std::vector<Triplets> t_D, t_dD;

  Eigen::SparseMatrix<Scalar> A1, A1Inv;
  Eigen::SparseMatrix<Scalar> D, dD;

 public:
  Matrix<Scalar> tensordotD_result;
  Vector<Scalar> tensordotD_bwta;
  Matrix<Scalar> tensordotD_bM;

  // Constructor: compute the matrices
  explicit DiffRot(basis::Basis<Scalar> &B, const int &drorder) :
      B(B), ydeg(B.ydeg), Ny((ydeg + 1) * (ydeg + 1)), drorder(drorder),
      ddeg(4 * drorder), Ddeg((ddeg + 1) * ydeg), ND((Ddeg + 1) * (Ddeg + 1)) {
    // Trivial cases
    if ((ydeg == 0) || (drorder == 0)) {
      return;
    }

    // Initialize the constant triplets
    t_0.push_back(Triplet(0, 0, 0));
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
    dD.resize(ND, Ny);
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
    // Trivial cases
    if ((ydeg == 0) || (drorder == 0)) {
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
    tensordotD_result = MA1InvD * A1;
  }

  /**
  Compute the gradient of the differential rotation operation.

  */
  template <typename T1>
  inline void tensordotD(const MatrixBase<T1> &M, const Vector<Scalar> &wta,
                         const Matrix<Scalar> &bf) {
    // Size checks
    size_t npts = wta.size();
    if (((size_t)M.rows() != npts) || ((int)M.cols() != Ny))
      throw std::runtime_error("Incompatible shapes in `tensordotD`.");

    tensordotD_bwta.setZero(npts);
    tensordotD_bM.setZero(npts, Ny);

    // Trivial case
    if ((ydeg == 0) || (drorder == 0)) {
      tensordotD_bM = bf;
      return;
    }

    // Temporary matrices for computing bM and bwta
    Matrix<Scalar> A1bfT = A1 * bf.transpose();
    Matrix<Scalar> DA1bfT(ND, npts);
    Matrix<Scalar> MA1Inv = M * A1Inv;

    // Loop over all times
    for (int i = 0; i < wta.size(); ++i) {
      t_c.clear();
      t_dc.clear();
      t_s.clear();
      t_ds.clear();
      t_xD.clear();
      t_dxD.clear();
      t_zD.clear();
      t_dzD.clear();

      // Cosine expansion
      Scalar fac = 1.0;
      Scalar dfac = 0.0;
      Scalar tmp;
      for (int l = 0; l < ddeg + 1; l += 4) {
        t_c.push_back(Triplet(l, l, fac));
        t_dc.push_back(Triplet(l, l, dfac));
        tmp = -(4.0 * wta(i)) / ((l + 4.0) * (l + 2.0));
        dfac = tmp * (wta(i) * dfac + 2 * fac);
        fac = tmp * wta(i) * fac;
      }
      computeSparsePolynomialProduct(t_x, t_c, t_xc);
      computeSparsePolynomialProduct(t_x, t_dc, t_dxc);
      computeSparsePolynomialProduct(t_z, t_c, t_zc);
      computeSparsePolynomialProduct(t_z, t_dc, t_dzc);

      // Sine expansion
      fac = wta(i);
      dfac = 1.0;
      for (int l = 2; l < ddeg + 1; l += 4) {
        t_s.push_back(Triplet(l, l, fac));
        t_ds.push_back(Triplet(l, l, dfac));
        tmp = -(4.0 * wta(i)) / ((l + 4.0) * (l + 2.0));
        dfac = tmp * (wta(i) * dfac + 2 * fac);
        fac = tmp * wta(i) * fac;
      }
      computeSparsePolynomialProduct(t_x, t_s, t_xs);
      computeSparsePolynomialProduct(t_x, t_ds, t_dxs);
      computeSparsePolynomialProduct(t_z, t_s, t_zs);
      computeSparsePolynomialProduct(t_z, t_ds, t_dzs);
      computeSparsePolynomialProduct(t_neg_z, t_s, t_neg_zs);
      computeSparsePolynomialProduct(t_neg_z, t_ds, t_neg_dzs);

      // Differentially-rotated x and z terms
      for (Triplet term : t_xc) t_xD.push_back(term);
      for (Triplet term : t_dxc) t_dxD.push_back(term);
      for (Triplet term : t_neg_zs) t_xD.push_back(term);
      for (Triplet term : t_neg_dzs) t_dxD.push_back(term);
      for (Triplet term : t_xs) t_zD.push_back(term);
      for (Triplet term : t_dxs) t_dzD.push_back(term);
      for (Triplet term : t_zc) t_zD.push_back(term);
      for (Triplet term : t_dzc) t_dzD.push_back(term);

      // Construct the matrix
      t_D.clear();
      t_dD.clear();
      t_D.resize(Ny);
      t_dD.resize(Ny);

      // l = 0
      t_D[0] = t_1;
      t_dD[0] = t_0;

      // l = 1
      t_D[1] = t_xD;
      t_dD[1] = t_dxD;
      t_D[2] = t_zD;
      t_dD[2] = t_dzD;
      t_D[3] = t_y;
      t_dD[3] = t_0;

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
          // Chain rule
          Triplets tmp;
          computeSparsePolynomialProduct(t_dD[j], t_xD, t_dD[n]);
          computeSparsePolynomialProduct(t_D[j], t_dxD, tmp);
          for (Triplet term : tmp) {
            t_dD[n].push_back(term);
          }
          ++n;
        }

        // The last two terms of this degree
        computeSparsePolynomialProduct(t_D[nc - 1], t_y, t_D[n + 1]);
        computeSparsePolynomialProduct(t_D[nc - 2], t_y, t_D[n]);
        // Chain rule
        computeSparsePolynomialProduct(t_dD[nc - 1], t_y, t_dD[n + 1]);
        computeSparsePolynomialProduct(t_dD[nc - 2], t_y, t_dD[n]);

      }

      // Construct the sparse D operator from the triplets
      coeffs.clear();
      dcoeffs.clear();
      for (int col = 0; col < Ny; ++col) {
        for (Triplet term : t_D[col]) {
          int l = term.row();
          int m = term.col();
          int row = l * l + l + m;
          coeffs.push_back(Triplet(row, col, term.value()));
        }
        for (Triplet term : t_dD[col]) {
          int l = term.row();
          int m = term.col();
          int row = l * l + l + m;
          dcoeffs.push_back(Triplet(row, col, term.value()));
        }
      }
      D.setFromTriplets(coeffs.begin(), coeffs.end());
      dD.setFromTriplets(dcoeffs.begin(), dcoeffs.end());

      // Used to compute bM below
      DA1bfT.col(i) = D * A1bfT.col(i);

      // bwta
      tensordotD_bwta(i) = MA1Inv.row(i) * dD * A1bfT.col(i);
    }

    // Finish computing bM
    tensordotD_bM = (A1Inv * DA1bfT).transpose();

  }
};

}  // namespace diffrot
}  // namespace starry
#endif