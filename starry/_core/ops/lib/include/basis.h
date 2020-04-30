/**
\file basis.h
\brief Miscellaneous utilities related to basis transformations.

*/

#ifndef _STARRY_BASIS_H_
#define _STARRY_BASIS_H_

#include "reflected/oren_nayar.h"
#include "utils.h"

namespace starry {
namespace basis {

using namespace utils;

/**
Multiply a polynomial vector/matrix by `z`.

*/
template <typename T1, typename T2>
inline void polymulz(int lmax, const MatrixBase<T1> &p, MatrixBase<T2> &pz) {
  int n = 0;
  int lz, nz;
  bool odd1;
  pz.setZero();
  for (int l = 0; l < lmax + 1; ++l) {
    for (int m = -l; m < l + 1; ++m) {
      odd1 = (l + m) % 2 == 0 ? false : true;
      lz = l + 1;
      nz = lz * lz + lz + m;
      if (odd1) {
        pz.row(nz - 4 * lz + 2) += p.row(n);
        pz.row(nz - 2) -= p.row(n);
        pz.row(nz + 2) -= p.row(n);
      } else {
        pz.row(nz) += p.row(n);
      }
      ++n;
    }
  }
}

/**
Compute the `P(z)` part of the Ylm vectors.

TODO: This can be sped up with sparse algebra.

*/
template <typename Scalar>
inline void legendre(int lmax,
                     std::vector<std::vector<Eigen::Triplet<Scalar>>> &M) {
  // Compute densely
  int N = (lmax + 1) * (lmax + 1);
  int ip, im;
  Scalar term = 1.0, fac = 1.0;
  Vector<Scalar> colvec(N);
  Matrix<Scalar> dnsM(N, N);
  dnsM.setZero();
  for (int m = 0; m < lmax + 1; ++m) {
    // 1
    ip = m * m + 2 * m;
    im = m * m;
    dnsM(0, ip) = fac;
    dnsM(0, im) = fac;
    if (m < lmax) {
      // z
      ip = m * m + 4 * m + 2;
      im = m * m + 2 * m + 2;
      dnsM(2, ip) = (2 * m + 1) * dnsM(m * m + 2 * m, 0);
      dnsM(2, im) = dnsM(2, ip);
    }
    for (int l = m + 1; l < lmax + 1; ++l) {
      // Recurse
      ip = l * l + l + m;
      im = l * l + l - m;
      polymulz(lmax - 1, dnsM.col((l - 1) * (l - 1) + l - 1 + m), colvec);
      dnsM.col(ip) = (2 * l - 1) * colvec / (l - m);
      if (l > m + 1)
        dnsM.col(ip) -=
            (l + m - 1) * dnsM.col((l - 2) * (l - 2) + l - 2 + m) / (l - m);
      dnsM.col(im) = dnsM.col(ip);
    }
    fac *= -term;
    term += 2;
  }

  // Store as triplets
  M.resize(N);
  for (int col = 0; col < N; ++col) {
    int n2 = 0;
    for (int l = 0; l < lmax + 1; ++l) {
      for (int m = -l; m < l + 1; ++m) {
        if (dnsM(n2, col) != 0)
          M[col].push_back(Eigen::Triplet<Scalar>(l, m, dnsM(n2, col)));
        ++n2;
      }
    }
  }
}

/**
Compute the `theta(x, y)` term of the Ylm vectors.

*/
template <typename Scalar>
inline void theta(int lmax,
                  std::vector<std::vector<Eigen::Triplet<Scalar>>> &M) {
  int N = (lmax + 1) * (lmax + 1);
  M.resize(N);
  Scalar term1, term2;
  int n1, n2;
  for (int m = 0; m < lmax + 1; ++m) {
    term1 = 1.0;
    term2 = m;
    for (int j = 0; j < m + 1; j += 2) {
      if (j > 0) {
        term1 *= -(m - j + 1.0) * (m - j + 2.0) / (j * (j - 1.0));
        term2 *= -(m - j) * (m - j + 1.0) / (j * (j + 1.0));
      }
      for (int l = m; l < lmax + 1; ++l) {
        n1 = l * l + l + m;
        n2 = l * l + l - m;
        M[n1].push_back(Eigen::Triplet<Scalar>(m, 2 * j - m, term1));
        if (j < m) {
          M[n2].push_back(Eigen::Triplet<Scalar>(m, 2 * (j + 1) - m, term2));
        }
      }
    }
  }
}

/**
Compute the amplitudes of the Ylm vectors.

*/
template <typename Derived> inline void amp(int lmax, MatrixBase<Derived> &M) {
  M.setZero();
  typename Derived::Scalar inv_root_two = sqrt(0.5);
  for (int l = 0; l < lmax + 1; ++l) {
    M.col(l * l + l).setConstant(sqrt(2 * (2 * l + 1)));
    for (int m = 1; m < l + 1; ++m) {
      M.col(l * l + l + m) =
          -M.col(l * l + l + m - 1) / sqrt((l + m) * (l - m + 1));
      M.col(l * l + l - m) = M.col(l * l + l + m);
    }
    M.col(l * l + l) *= inv_root_two;
  }
  M /= (2 * root_pi<typename Derived::Scalar>());
}

/**
Compute a sparse polynomial product using Eigen Triplets.

*/
template <typename Scalar>
inline void
computeSparsePolynomialProduct(const std::vector<Eigen::Triplet<Scalar>> &p1,
                               const std::vector<Eigen::Triplet<Scalar>> &p2,
                               std::vector<Eigen::Triplet<Scalar>> &p1p2) {
  using Triplet = Eigen::Triplet<Scalar>;
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
Compute the *sparse* change of basis matrix `A1`.

*/
template <typename Scalar>
inline void computeA1(int lmax, Eigen::SparseMatrix<Scalar> &A1,
                      const Scalar &norm) {
  using Triplet = Eigen::Triplet<Scalar>;
  using Triplets = std::vector<Triplet>;

  int N = (lmax + 1) * (lmax + 1);

  // Amplitude
  Matrix<Scalar> C(N, N);
  amp(lmax, C);

  // Z terms
  std::vector<Triplets> t_Z(N);
  legendre(lmax, t_Z);

  // XY terms
  std::vector<Triplets> t_XY(N);
  theta(lmax, t_XY);

  // Construct the change of basis matrix
  Triplets t_M, coeffs;
  for (int col = 0; col < N; ++col) {
    // Multiply Z and XY
    computeSparsePolynomialProduct(t_Z[col], t_XY[col], t_M);

    // Parse the terms and store in `coeffs`
    for (Triplet term : t_M) {
      int l = term.row();
      int m = term.col();
      int row = l * l + l + m;
      Scalar value = term.value() * norm * C(row, col);
      coeffs.push_back(Triplet(row, col, value));
    }
  }
  A1.resize(N, N);
  A1.setFromTriplets(coeffs.begin(), coeffs.end());
}

/**
Compute the inverse of the change of basis matrix `A1`.

*/
template <typename T>
void computeA1Inv(int lmax, const Eigen::SparseMatrix<T> &A1,
                  Eigen::SparseMatrix<T> &A1Inv) {
  int N = (lmax + 1) * (lmax + 1);
  Eigen::SparseLU<Eigen::SparseMatrix<T>> solver;
  solver.compute(A1);
  if (solver.info() != Eigen::Success)
    throw std::runtime_error(
        "Error computing the change of basis matrix `A1Inv`.");
  Eigen::SparseMatrix<T> I = Matrix<T>::Identity(N, N).sparseView();
  A1Inv = solver.solve(I);

  // NOTE: Stability hack. Slow but worth it.
  if (lmax <= 30) {
    Matrix<T> A1Inv_ = A1Inv;
    Matrix<T> Zeros = Matrix<T>::Zero(N, N);
    A1Inv_ = (A1Inv_.cwiseAbs().array() > mach_eps<T>()).select(A1Inv_, Zeros);
    A1Inv = A1Inv_.sparseView();
  }
}

/**
Compute the full change of basis matrix, `A`.

*/
template <typename T>
void computeA(int lmax, const Eigen::SparseMatrix<T> &A1,
              Eigen::SparseMatrix<T> &A2, Eigen::SparseMatrix<T> &A) {
  int i, n, l, m, mu, nu;
  int N = (lmax + 1) * (lmax + 1);

  // Let's compute the inverse of A2, since it's easier
  Matrix<T> A2InvDense = Matrix<T>::Zero(N, N);
  n = 0;
  for (l = 0; l < lmax + 1; ++l) {
    for (m = -l; m < l + 1; ++m) {
      mu = l - m;
      nu = l + m;
      if (nu % 2 == 0) {
        // x^(mu/2) y^(nu/2)
        A2InvDense(n, n) = (mu + 2) / 2;
      } else if ((l == 1) && (m == 0)) {
        // z
        A2InvDense(n, n) = 1;
      } else if ((mu == 1) && (l % 2 == 0)) {
        // x^(l-2) y z
        i = l * l + 3;
        A2InvDense(i, n) = 3;
      } else if ((mu == 1) && (l % 2 == 1)) {
        // x^(l-3) z
        i = 1 + (l - 2) * (l - 2);
        A2InvDense(i, n) = -1;
        // x^(l-1) z
        i = l * l + 1;
        A2InvDense(i, n) = 1;
        // x^(l-3) y^2 z
        i = l * l + 5;
        A2InvDense(i, n) = 4;
      } else {
        if (mu != 3) {
          // x^((mu - 5)/2) y^((nu - 1)/2)
          i = nu + ((mu - 4 + nu) * (mu - 4 + nu)) / 4;
          A2InvDense(i, n) = (mu - 3) / 2;
          // x^((mu - 5)/2) y^((nu + 3)/2)
          i = nu + 4 + ((mu + nu) * (mu + nu)) / 4;
          A2InvDense(i, n) = -(mu - 3) / 2;
        }
        // x^((mu - 1)/2) y^((nu - 1)/2)
        i = nu + (mu + nu) * (mu + nu) / 4;
        A2InvDense(i, n) = -(mu + 3) / 2;
      }
      ++n;
    }
  }

  // Sparse dot A2 into A1
  Eigen::SparseMatrix<T> A2Inv = A2InvDense.sparseView();
  Eigen::SparseLU<Eigen::SparseMatrix<T>> solver;
  solver.compute(A2Inv);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error(
        "Error computing the change of basis matrix `A2`.");
  }
  Eigen::SparseMatrix<T> I = Matrix<T>::Identity(N, N).sparseView();
  A2 = solver.solve(I);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error(
        "Error computing the change of basis matrix `A2`.");
  }
  A = solver.solve(A1);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error(
        "Error computing the change of basis matrix `A1`.");
  }
}

/**

  Compute the change of basis matrix `A2` and its inverse.

*/
template <typename Scalar>
void computeA2(int lmax, Eigen::SparseMatrix<Scalar> &A2,
               Eigen::SparseMatrix<Scalar> &A2Inv) {

  int i, n, l, m, mu, nu;
  int N = (lmax + 1) * (lmax + 1);
  Matrix<Scalar> A2InvDense = Matrix<Scalar>::Zero(N, N);
  n = 0;
  for (l = 0; l < lmax + 1; ++l) {
    for (m = -l; m < l + 1; ++m) {
      mu = l - m;
      nu = l + m;
      if (nu % 2 == 0) {
        // x^(mu/2) y^(nu/2)
        A2InvDense(n, n) = (mu + 2) / 2;
      } else if ((l == 1) && (m == 0)) {
        // z
        A2InvDense(n, n) = 1;
      } else if ((mu == 1) && (l % 2 == 0)) {
        // x^(l-2) y z
        i = l * l + 3;
        A2InvDense(i, n) = 3;
      } else if ((mu == 1) && (l % 2 == 1)) {
        // x^(l-3) z
        i = 1 + (l - 2) * (l - 2);
        A2InvDense(i, n) = -1;
        // x^(l-1) z
        i = l * l + 1;
        A2InvDense(i, n) = 1;
        // x^(l-3) y^2 z
        i = l * l + 5;
        A2InvDense(i, n) = 4;
      } else {
        if (mu != 3) {
          // x^((mu - 5)/2) y^((nu - 1)/2)
          i = nu + ((mu - 4 + nu) * (mu - 4 + nu)) / 4;
          A2InvDense(i, n) = (mu - 3) / 2;
          // x^((mu - 5)/2) y^((nu + 3)/2)
          i = nu + 4 + ((mu + nu) * (mu + nu)) / 4;
          A2InvDense(i, n) = -(mu - 3) / 2;
        }
        // x^((mu - 1)/2) y^((nu - 1)/2)
        i = nu + (mu + nu) * (mu + nu) / 4;
        A2InvDense(i, n) = -(mu + 3) / 2;
      }
      ++n;
    }
  }

  // Get the inverse
  A2Inv = A2InvDense.sparseView();
  Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> solver;
  solver.compute(A2Inv);
  if (solver.info() != Eigen::Success) {
    std::stringstream args;
    args << "N = " << N;
    throw StarryException("Error computing the change of basis matrix `A2Inv`.",
                          "basis.h", "computeA2", args.str());
  }
  Eigen::SparseMatrix<Scalar> Id = Matrix<Scalar>::Identity(N, N).sparseView();
  A2 = solver.solve(Id);
  if (solver.info() != Eigen::Success) {
    std::stringstream args;
    args << "N = " << N;
    throw StarryException("Error computing the change of basis matrix `A2`.",
                          "basis.h", "computeA2", args.str());
  }

  A2Inv = A2InvDense.sparseView();
}

/**
Compute the `r^T` phase curve solution vector.

*/
template <typename T> void computerT(int lmax, RowVector<T> &rT) {
  T amp0, amp, lfac1, lfac2;
  int mu, nu;
  rT.resize((lmax + 1) * (lmax + 1));
  rT.setZero();
  amp0 = pi<T>();
  lfac1 = 1.0;
  lfac2 = 2.0 / 3.0;
  for (int l = 0; l < lmax + 1; l += 4) {
    amp = amp0;
    for (int m = 0; m < l + 1; m += 4) {
      mu = l - m;
      nu = l + m;
      rT(l * l + l + m) = amp * lfac1;
      rT(l * l + l - m) = amp * lfac1;
      if (l < lmax) {
        rT((l + 1) * (l + 1) + l + m + 1) = amp * lfac2;
        rT((l + 1) * (l + 1) + l - m + 1) = amp * lfac2;
      }
      amp *= (nu + 2.0) / (mu - 2.0);
    }
    lfac1 /= (l / 2 + 2) * (l / 2 + 3);
    lfac2 /= (l / 2 + 2.5) * (l / 2 + 3.5);
    amp0 *= 0.0625 * (l + 2) * (l + 2);
  }
  amp0 = 0.5 * pi<T>();
  lfac1 = 0.5;
  lfac2 = 4.0 / 15.0;
  for (int l = 2; l < lmax + 1; l += 4) {
    amp = amp0;
    for (int m = 2; m < l + 1; m += 4) {
      mu = l - m;
      nu = l + m;
      rT(l * l + l + m) = amp * lfac1;
      rT(l * l + l - m) = amp * lfac1;
      if (l < lmax) {
        rT((l + 1) * (l + 1) + l + m + 1) = amp * lfac2;
        rT((l + 1) * (l + 1) + l - m + 1) = amp * lfac2;
      }
      amp *= (nu + 2.0) / (mu - 2.0);
    }
    lfac1 /= (l / 2 + 2) * (l / 2 + 3);
    lfac2 /= (l / 2 + 2.5) * (l / 2 + 3.5);
    amp0 *= 0.0625 * l * (l + 4);
  }
}

/**
Compute the change of basis matrices from limb darkening coefficients
to polynomial and Green's polynomial coefficients.

*/
template <typename T>
void computeU(int lmax, const Eigen::SparseMatrix<T> &A1,
              const Eigen::SparseMatrix<T> &A, Eigen::SparseMatrix<T> &U1,
              T norm) {
  T twol, amp, lfac, lchoosek, fac0, fac;
  int N = (lmax + 1) * (lmax + 1);
  Matrix<T> U0;
  Matrix<T> LT, YT;
  LT.setZero(lmax + 1, lmax + 1);
  YT.setZero(lmax + 1, lmax + 1);

  // Compute L^T
  for (int l = 0; l < lmax + 1; ++l) {
    lchoosek = 1;
    for (int k = 0; k < l + 1; ++k) {
      if ((k + 1) % 2 == 0)
        LT(k, l) = lchoosek;
      else
        LT(k, l) = -lchoosek;
      lchoosek *= (l - k) / (k + 1.0);
    }
  }

  // Compute Y^T
  // Even terms
  twol = 1.0;
  lfac = 1.0;
  fac0 = 1.0;
  for (int l = 0; l < lmax + 1; l += 2) {
    amp = twol * sqrt((2 * l + 1) / (4 * pi<T>())) / lfac;
    lchoosek = 1;
    fac = fac0;
    for (int k = 0; k < l + 1; k += 2) {
      YT(k, l) = amp * lchoosek * fac;
      fac *= (k + l + 1.0) / (k - l + 1.0);
      lchoosek *= (l - k) * (l - k - 1) / ((k + 1.0) * (k + 2.0));
    }
    fac0 *= -0.25 * (l + 1) * (l + 1);
    lfac *= (l + 1.0) * (l + 2.0);
    twol *= 4.0;
  }
  // Odd terms
  twol = 2.0;
  lfac = 1.0;
  fac0 = 0.5;
  for (int l = 1; l < lmax + 1; l += 2) {
    amp = twol * sqrt((2 * l + 1) / (4 * pi<T>())) / lfac;
    lchoosek = l;
    fac = fac0;
    for (int k = 1; k < l + 1; k += 2) {
      YT(k, l) = amp * lchoosek * fac;
      fac *= (k + l + 1.0) / (k - l + 1.0);
      lchoosek *= (l - k) * (l - k - 1) / ((k + 1.0) * (k + 2.0));
    }
    fac0 *= -0.25 * (l + 2) * l;
    lfac *= (l + 1.0) * (l + 2.0);
    twol *= 4.0;
  }

  // Compute U0
  Eigen::HouseholderQR<Matrix<T>> solver(lmax + 1, lmax + 1);
  solver.compute(YT);
  U0 = solver.solve(LT);

  // Normalize it. Since we compute `U0` from the *inverse*
  // of `A1`, we must *divide* by the normalization constant
  U0 /= norm;

  // Compute U1
  Matrix<T> X(N, lmax + 1);
  X.setZero();
  for (int l = 0; l < lmax + 1; ++l)
    X(l * (l + 1), l) = 1;
  Eigen::SparseMatrix<T> XU0 = (X * U0).sparseView();
  U1 = A1 * XU0;
}

// --

/**
Basis transform matrices and operations.

*/
template <typename T> class Basis {
public:
  const int ydeg; /**< The highest degree of the spherical harmonic map */
  const int udeg; /**< The highest degree of the limb darkening map */
  const int fdeg; /**< The highest degree of the filter map */
  const int deg;
  const double norm;         /**< Map normalization constant */
  Eigen::SparseMatrix<T> A1; /**< The polynomial change of basis matrix */
  Eigen::SparseMatrix<T>
      A1_big; /**< The augmented polynomial change of basis matrix */
  Eigen::SparseMatrix<T> A1_f; /**< The polynomial change of basis matrix for
                                  the filter operator */
  Eigen::SparseMatrix<T>
      A1Inv; /**< The inverse of the polynomial change of basis matrix */
  Eigen::SparseMatrix<T> A2; /**< The Green's change of basis matrix */
  Eigen::SparseMatrix<T> A;  /**< The full change of basis matrix */
  RowVector<T> rT;           /**< The rotation solution vector */
  RowVector<T> rTA1;         /**< The rotation vector in Ylm space */
  Eigen::SparseMatrix<T>
      U1; /**< The limb darkening to polynomial change of basis matrix */

  // Special sizes for reflected light stuff
  Eigen::SparseMatrix<T> A1_Reflected;
  Eigen::SparseMatrix<T> A1Inv_Reflected;
  Eigen::SparseMatrix<T> A2_Reflected;
  Eigen::SparseMatrix<T> A2Inv_Reflected;
  Eigen::SparseMatrix<T> AInv_Reflected;

  // Poly basis
  RowVector<T> x_cache, y_cache, z_cache;
  int deg_cache;
  Matrix<T, RowMajor> pT;

  // Constructor: compute the matrices
  explicit Basis(int ydeg, int udeg, int fdeg, T norm = 2.0 / root_pi<T>())
      : ydeg(ydeg), udeg(udeg), fdeg(fdeg), deg(ydeg + udeg + fdeg), norm(norm),
        x_cache(0), y_cache(0), z_cache(0), deg_cache(-1) {

    // TODO: This class needs to be re-written. We're computing the same
    // things over and over again just to get different shapes...

    // Compute the augmented matrices
    Eigen::SparseMatrix<T> A1Inv_, A2_, A_, U1_;
    RowVector<T> rT_, rTA1_;
    computeA1(deg, A1_big, norm);
    computeA1Inv(deg, A1_big, A1Inv_);
    computeA(deg, A1_big, A2_, A_);
    computerT(deg, rT_);
    rTA1_ = rT_ * A1_big;
    computeU(deg, A1_big, A_, U1_, norm);

    // Resize to the shapes actually used in the code
    int Ny = (ydeg + 1) * (ydeg + 1);
    int Nf = (fdeg + 1) * (fdeg + 1);
    A1 = A1_big.block(0, 0, Ny, Ny);
    A1_f = A1_big.block(0, 0, Nf, Nf);
    A1Inv = A1Inv_;
    A2 = A2_.block(0, 0, Ny, Ny);
    A = A_;
    rT = rT_;
    rTA1 = rTA1_.segment(0, Ny);
    U1 = U1_.block(0, 0, (udeg + 1) * (udeg + 1), udeg + 1);

    // Special augmented matrices for reflected light maps
    computeA2(deg + STARRY_OREN_NAYAR_DEG, A2_Reflected, A2Inv_Reflected);
    computeA1(deg + STARRY_OREN_NAYAR_DEG, A1_Reflected, norm);
    computeA1Inv(deg + STARRY_OREN_NAYAR_DEG, A1_Reflected, A1Inv_Reflected);
    AInv_Reflected = A1Inv_Reflected * A2Inv_Reflected;
  };

  /**
    Compute the polynomial basis at a vector of points.

  */
  inline void computePolyBasis(const int deg, const RowVector<T> &x,
                               const RowVector<T> &y, const RowVector<T> &z) {
    // Dimensions
    size_t npts = x.cols();
    int N = (deg + 1) * (deg + 1);
    pT.resize(npts, N);

    // Check the cache
    if ((npts == size_t(x_cache.size())) && (x == x_cache) && (y == y_cache) &&
        (z == z_cache) && (deg == deg_cache)) {
      return;
    } else if (npts == 0) {
      return;
    }
    x_cache = x;
    y_cache = y;
    z_cache = z;
    deg_cache = deg;

    // Optimized polynomial basis computation
    // A little opaque, sorry...
    Matrix<T> xarr(npts, N), yarr(npts, N);
    RowVector<T> xterm(npts), yterm(npts);
    xterm.setOnes();
    yterm.setOnes();
    xterm += 0.0 * z; // Ensures we get `nan`s off the disk
    yterm += 0.0 * z; // Ensures we get `nan`s off the disk
    int i0 = 0, di0 = 3, j0 = 0, dj0 = 2;
    int i, j, di, dj, n;
    for (n = 0; n < deg + 1; ++n) {
      i = i0;
      di = di0;
      xarr.col(i) = xterm;
      j = j0;
      dj = dj0;
      yarr.col(j) = yterm;
      i = i0 + di - 1;
      j = j0 + dj - 1;
      while (i + 1 < N) {
        xarr.col(i) = xterm;
        xarr.col(i + 1) = xterm;
        di += 2;
        i += di;
        yarr.col(j) = yterm;
        yarr.col(j + 1) = yterm;
        dj += 2;
        j += dj - 1;
      }
      xterm = xterm.cwiseProduct(x);
      i0 += 2 * n + 1;
      di0 += 2;
      yterm = yterm.cwiseProduct(y);
      j0 += 2 * (n + 1) + 1;
      dj0 += 2;
    }
    n = 0;
    for (int l = 0; l < deg + 1; ++l) {
      for (int m = -l; m < l + 1; ++m) {
        pT.col(n) = xarr.col(n).cwiseProduct(yarr.col(n));
        if ((l + m) % 2 != 0)
          pT.col(n) = pT.col(n).cwiseProduct(z.transpose());
        ++n;
      }
    }
  }
};

} // namespace basis
} // namespace starry
#endif