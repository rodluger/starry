/**
\file basis.h
\brief Miscellaneous utilities related to basis transformations.

*/

#ifndef _STARRY_BASIS_H_
#define _STARRY_BASIS_H_

#include "utils.h"
#include "errors.h"

namespace starry { 
namespace basis {

using namespace utils;

/**
Multiply a polynomial vector/matrix by `z`.

*/
template <typename T1, typename T2> 
inline void polymulz (
    int lmax, 
    const MatrixBase<T1>& p,
    MatrixBase<T2>& pz
) {
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
Multiply two polynomial vectors.

*/
template <typename Scalar>
inline void polymul (
    int lmax1, 
    const Vector<Scalar>& p1,
    int lmax2,
    const Vector<Scalar>& p2,
    int lmax12, 
    Vector<Scalar>& p1p2
) {
    bool odd1;
    int l, n;
    int n2, n1 = 0;
    Scalar fac1;
    p1p2.setZero();
    for (int l1 = 0; l1 < lmax1 + 1; ++l1) {
        for (int m1 = -l1; m1 < l1 + 1; ++m1) {
            if (p1(n1)) {
                odd1 = (l1 + m1) % 2 == 0 ? false : true;
                n2 = 0;
                for (int l2 = 0; l2 < lmax2 + 1; ++l2) {
                    if (l1 + l2 > lmax12) break;
                    for (int m2 = -l2; m2 < l2 + 1; ++m2) {
                        if (p2(n2)) {
                            l = l1 + l2;
                            n = l * l + l + m1 + m2;
                            fac1 = p1(n1) * p2(n2);
                            if (odd1 && ((l2 + m2) % 2 != 0)) {
                                p1p2(n - 4 * l + 2) += fac1;
                                p1p2(n - 2) -= fac1;
                                p1p2(n + 2) -= fac1;
                            } else {
                                p1p2(n) += fac1;
                            }
                        }
                        ++n2;
                    }
                }
            }
            ++n1;
        }
    }  
}

/**
Multiply two polynomial vectors and propagate the gradient.

*/
template <typename Scalar>
inline void polymul (
    int lmax1, 
    const Vector<Scalar>& p1,
    int lmax2,
    const Vector<Scalar>& p2,
    int lmax12, 
    Vector<Scalar>& p1p2,
    const RowVector<Scalar>& vT,
    RowVector<Scalar>& vTDp1p2Dp1,
    RowVector<Scalar>& vTDp1p2Dp2
) {
    bool odd1;
    int l, n;
    int n2, n1 = 0;
    Scalar fac1, fac2;
    p1p2.setZero();
    vTDp1p2Dp1.setZero();
    vTDp1p2Dp2.setZero();
    for (int l1 = 0; l1 < lmax1 + 1; ++l1) {
        for (int m1 = -l1; m1 < l1 + 1; ++m1) {
            if (p1(n1)) {
                odd1 = (l1 + m1) % 2 == 0 ? false : true;
                n2 = 0;
                for (int l2 = 0; l2 < lmax2 + 1; ++l2) {
                    if (l1 + l2 > lmax12) break;
                    for (int m2 = -l2; m2 < l2 + 1; ++m2) {
                        if (p2(n2)) {
                            l = l1 + l2;
                            n = l * l + l + m1 + m2;
                            fac1 = p1(n1) * p2(n2);
                            if (odd1 && ((l2 + m2) % 2 != 0)) {
                                p1p2(n - 4 * l + 2) += fac1;
                                p1p2(n - 2) -= fac1;
                                p1p2(n + 2) -= fac1;
                                fac2 = vT(n - 4 * l + 2) - vT(n - 2) - vT(n + 2);
                                vTDp1p2Dp1(n1) += fac2 * p2(n2);
                                vTDp1p2Dp2(n2) += fac2 * p1(n1);
                            } else {
                                p1p2(n) += fac1;
                                vTDp1p2Dp1(n1) += vT(n) * p2(n2);
                                vTDp1p2Dp2(n2) += vT(n) * p1(n1);
                            }
                        }
                        ++n2;
                    }
                }
            }
            ++n1;
        }
    }  
}

/**
Multiply two polynomial matrices.

*/
template <typename Derived>
inline void polymul (
    int lmax1, 
    const MatrixBase<Derived>& p1,
    int lmax2,
    const MatrixBase<Derived>& p2,
    int lmax12, 
    MatrixBase<Derived>& p1p2
) {
    bool odd1;
    int l, n;
    int n2, n1 = 0;
    RowVector<typename Derived::Scalar> fac1;
    p1p2.setZero();
    for (int l1 = 0; l1 < lmax1 + 1; ++l1) {
        for (int m1 = -l1; m1 < l1 + 1; ++m1) {
            if (p1.row(n1).any()) {
                odd1 = (l1 + m1) % 2 == 0 ? false : true;
                n2 = 0;
                for (int l2 = 0; l2 < lmax2 + 1; ++l2) {
                    if (l1 + l2 > lmax12) break;
                    for (int m2 = -l2; m2 < l2 + 1; ++m2) {
                        if (p2.row(n2).any()) {
                            l = l1 + l2;
                            n = l * l + l + m1 + m2;
                            fac1 = p1.row(n1).cwiseProduct(p2.row(n2));
                            if (odd1 && ((l2 + m2) % 2 != 0)) {
                                p1p2.row(n - 4 * l + 2) += fac1;
                                p1p2.row(n - 2) -= fac1;
                                p1p2.row(n + 2) -= fac1;
                            } else {
                                p1p2.row(n) += fac1;
                            }
                        }
                        ++n2;
                    }
                }
            }
            ++n1;
        }
    }  
}

/**
Multiply two polynomial matrices and propagate the gradient.

*/
template <typename Scalar>
inline void polymul (
    int lmax1, 
    const Matrix<Scalar>& p1,
    int lmax2,
    const Matrix<Scalar>& p2,
    int lmax12, 
    Matrix<Scalar>& p1p2,
    const RowVector<Scalar>& vT,
    Matrix<Scalar>& vTDp1p2Dp1,
    Matrix<Scalar>& vTDp1p2Dp2
) {
    bool odd1;
    int l, n;
    int n2, n1 = 0;
    RowVector<Scalar> fac1;
    Scalar fac2;
    p1p2.setZero();
    vTDp1p2Dp1.setZero();
    vTDp1p2Dp2.setZero();
    for (int l1 = 0; l1 < lmax1 + 1; ++l1) {
        for (int m1 = -l1; m1 < l1 + 1; ++m1) {
            if (p1.row(n1).any()) {
                odd1 = (l1 + m1) % 2 == 0 ? false : true;
                n2 = 0;
                for (int l2 = 0; l2 < lmax2 + 1; ++l2) {
                    if (l1 + l2 > lmax12) break;
                    for (int m2 = -l2; m2 < l2 + 1; ++m2) {
                        if (p2.row(n2).any()) {
                            l = l1 + l2;
                            n = l * l + l + m1 + m2;
                            fac1 = p1.row(n1).cwiseProduct(p2.row(n2));
                            if (odd1 && ((l2 + m2) % 2 != 0)) {
                                p1p2.row(n - 4 * l + 2) += fac1;
                                p1p2.row(n - 2) -= fac1;
                                p1p2.row(n + 2) -= fac1;
                                fac2 = vT(n - 4 * l + 2) - vT(n - 2) - vT(n + 2);
                                vTDp1p2Dp1.col(n1) += fac2 * p2.row(n2);
                                vTDp1p2Dp2.col(n2) += fac2 * p1.row(n1);

                            } else {
                                p1p2.row(n) += fac1;
                                vTDp1p2Dp1.col(n1) += vT(n) * p2.row(n2);
                                vTDp1p2Dp2.col(n2) += vT(n) * p1.row(n1);
                            }
                        }
                        ++n2;
                    }
                }
            }
            ++n1;
        }
    }  
}

/**
Compute the `P(z)` part of the Ylm vectors.

*/
template <typename Derived>
inline void legendre (
    int lmax, 
    MatrixBase<Derived>& M
) {
    M.setZero();
    int ip, im;
    typename Derived::Scalar term = 1, fac = 1;
    Vector<typename Derived::Scalar> col((lmax + 1) * (lmax + 1));
    for (int m = 0; m < lmax + 1; ++m) {
        // 1
        ip = m * m + 2 * m;
        im = m * m;
        M(0, ip) = fac;
        M(0, im) = fac;
        if (m < lmax) {
            // z
            ip = m * m + 4 * m + 2;
            im = m * m + 2 * m + 2;
            M(2, ip) = (2 * m + 1) * M(m * m + 2 * m, 0);
            M(2, im) = M(2, ip);
        }
        for (int l = m + 1; l < lmax + 1; ++l) {
            // Recurse
            ip = l * l + l + m;
            im = l * l + l - m;
            polymulz(lmax - 1, M.col((l - 1) * (l - 1) + l - 1 + m), col);
            M.col(ip) = (2 * l - 1) * col / (l - m);
            if (l > m + 1)
                M.col(ip) -= (l + m - 1) * 
                             M.col((l - 2) * (l - 2) + l - 2 + m) / (l - m);
            M.col(im) = M.col(ip);
        }
        fac *= -term;
        term += 2;
    }
}

/**
Compute the `theta(x, y)` term of the Ylm vectors.

*/
template <typename Derived>
inline void theta (
    int lmax, 
    MatrixBase<Derived>& M
) {
    int N = (lmax + 1) * (lmax + 1);
    typename Derived::Scalar term1, term2;
    int n1, n2, np1, np2;
    M.setZero();
    for (int m = 0; m < lmax + 1; ++m) {
        term1 = 1.0;
        term2 = m;
        for (int j = 0; j < m + 1; j += 2) {
            if (j > 0) {
                term1 *= -(m - j + 1.0) * (m - j + 2.0) / (j * (j - 1.0));
                term2 *= -(m - j) * (m - j + 1.0) / (j * (j + 1.0));
            }
            np1 = m * m + 2 * j;
            np2 = m * m + 2 * (j + 1);
            for (int l = m; l < lmax + 1; ++l) {
                n1 = l * l + l + m;
                n2 = l * l + l - m;
                M(np1, n1) = term1;
                if (np2 < N) 
                    M(np2, n2) = term2;
            }
        }
    }
}

/**
Compute the amplitudes of the Ylm vectors.

*/
template <typename Derived>
inline void amp (
    int lmax, 
    MatrixBase<Derived>& M
) {
    M.setZero();
    typename Derived::Scalar inv_root_two = sqrt(0.5);
    for (int l = 0; l < lmax + 1; ++l) {
        M.col(l * l + l).setConstant(sqrt(2 * (2 * l + 1)));
        for (int m = 1; m < l + 1; ++m) {
            M.col(l * l + l + m) = -M.col(l * l + l + m - 1) 
                                    / sqrt((l + m) * (l - m + 1));
            M.col(l * l + l - m) = M.col(l * l + l + m);
        }
        M.col(l * l + l) *= inv_root_two;
    }
    M /= (2 * root_pi<typename Derived::Scalar>());
}


/**
Compute the *dense* change of basis matrix `A1`.

*/
template <typename Derived>
inline void computeA1 (
    int lmax, 
    MatrixBase<Derived>& M,
    const typename Derived::Scalar& norm
) {
    typename Derived::PlainObject C, Z, XY;
    int N = (lmax + 1) * (lmax + 1);
    C.resize(N, N);
    Z.resize(N, N);
    XY.resize(N, N);
    amp(lmax, C);
    legendre(lmax, Z);
    theta(lmax, XY);
    polymul(lmax, Z, lmax, XY, lmax, M); 
    M = M.cwiseProduct(C);
    M *= norm;
}

/**
Compute the *sparse* change of basis matrix `A1`.

*/
template <typename T>
inline void computeA1 (
    int lmax, 
    Eigen::SparseMatrix<T>& A1,
    const T& norm
) {
    int N = (lmax + 1) * (lmax + 1);
    Matrix<T> A1Dense(N, N);
    computeA1(lmax, A1Dense, norm);
    A1 = A1Dense.sparseView();
}

/**
Compute the inverse of the change of basis matrix `A1`.

*/
template <typename T>
void computeA1Inv (
    int lmax, 
    const Eigen::SparseMatrix<T>& A1,
    Eigen::SparseMatrix<T>& A1Inv
) {
    int N = (lmax + 1) * (lmax + 1);
    Eigen::SparseLU<Eigen::SparseMatrix<T>> solver;
    solver.compute(A1);
    if (solver.info() != Eigen::Success)
        throw errors::LinearAlgebraError(
            "Error computing the change of basis matrix `A1Inv`.");
    Eigen::SparseMatrix<T> I = Matrix<T>::Identity(N, N).sparseView();
    A1Inv = solver.solve(I);
}

/**
Compute the full change of basis matrix, `A`.

*/
template <typename T>
void computeA(
    int lmax, 
    const Eigen::SparseMatrix<T>& A1,
    Eigen::SparseMatrix<T>& A2, 
    Eigen::SparseMatrix<T>& A
) {
    int i, n, l, m, mu, nu;
    int N = (lmax + 1) * (lmax + 1);

    // Let's compute the inverse of A2, since it's easier
    Matrix<T> A2InvDense = Matrix<T>::Zero(N, N);
    n = 0;
    for (l = 0; l < lmax + 1; ++l) {
        for (m = -l; m < l + 1; ++m){
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
        throw errors::LinearAlgebraError("Error computing the change "
                                         "of basis matrix `A2`.");
    }
    Eigen::SparseMatrix<T> I = Matrix<T>::Identity(N, N).sparseView();
    A2 = solver.solve(I);
    if (solver.info() != Eigen::Success) {
        throw errors::LinearAlgebraError("Error computing the change "
                                         "of basis matrix `A2`.");
    }
    A = solver.solve(A1);
    if (solver.info() != Eigen::Success) {
        throw errors::LinearAlgebraError("Error computing the change "
                                         "of basis matrix `A1`.");
    }
}

/**
Compute the change of basis matrices from limb darkening coefficients
to polynomial and Green's polynomial coefficients.

*/
template <typename T>
void computeU(
    int lmax, 
    const Eigen::SparseMatrix<T>& A1,
    const Eigen::SparseMatrix<T>& A, 
    Eigen::SparseMatrix<T>& U1,
    Eigen::SparseMatrix<T>& U, 
    T norm
) {
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

    // Compute U1 and U
    Matrix<T> X(N, lmax + 1);
    X.setZero();
    for (int l = 0; l < lmax + 1; ++l)
        X(l * (l + 1), l) = 1;
    Eigen::SparseMatrix<T> XU0 = (X * U0).sparseView();

    U1 = A1 * XU0;
    U = A * XU0;
}

/**
Compute the `r^T` phase curve solution vector.

*/
template <typename T>
void computerT (
    int lmax, 
    RowVector<T>& rT
) {
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


// --


/**
Basis transform matrices and operations.

*/
template <typename T>
class Basis {

public:

    const int lmax;                                                            /**< The highest degree of the map */
    const double norm;                                                         /**< Map normalization constant */
    Eigen::SparseMatrix<T> A1;                                                 /**< The polynomial change of basis matrix */
    Eigen::SparseMatrix<T> A1Inv;                                              /**< The inverse of the polynomial change of basis matrix */
    Eigen::SparseMatrix<T> A2;                                                 /**< The Green's change of basis matrix */
    Eigen::SparseMatrix<T> A;                                                  /**< The full change of basis matrix */
    RowVector<T> rT;                                                           /**< The rotation solution vector */
    RowVector<T> rTA1;                                                         /**< The rotation vector in Ylm space */
    RowVector<T> rTU1;                                                         /**< The rotation vector times the LD change of basis matrix */
    Eigen::SparseMatrix<T> U1;                                                 /**< The limb darkening to polynomial change of basis matrix */
    Eigen::SparseMatrix<T> U;                                                  /**< The full limb darkening change of basis matrix */

    // Constructor: compute the matrices
    explicit Basis(
        int lmax, 
        T norm=2.0 / root_pi<T>() 
    ) :
        lmax(lmax), 
        norm(norm)
    {
        computeA1(lmax, A1, norm);
        computeA(lmax, A1, A2, A);
        computeA1Inv(lmax, A1, A1Inv);
        computerT(lmax, rT);
        rTA1 = rT * A1;
        computeU(lmax, A1, A, U1, U, norm);
        rTU1 = rT * U1;
    }

    // Public methods

    inline void computePolyBasis ( 
        const T& x, 
        const T& y, 
        RowVector<T>& basis
    );

    template <typename T1, typename T2> 
    inline void computePolyBasis ( 
        const MatrixBase<T1>& x,
        const MatrixBase<T2>& y,
        Matrix<T>& basis
    );

    inline void computePolyMatrix (
        size_t res,
        Matrix<T>& P
    );

    inline void computeIlluminationMatrix (
        size_t res,
        const UnitVector<T>& source,
        Vector<T>& I
    );

};

/**
Compute the polynomial basis at a point.

*/
template <typename T> 
inline void Basis<T>::computePolyBasis ( 
    const T& x, 
    const T& y, 
    RowVector<T>& basis
) {
    T r2 = x * x + y * y;
    if (r2 > 1) {
        basis.setConstant(T(NAN));
        return;
    }
    int N = (lmax + 1) * (lmax + 1);
    RowVector<T> xarr(N), yarr(N);
    T xterm = 1, yterm = 1;
    T z = sqrt(1 - r2);
    int i0 = 0,
        di0 = 3,
        j0 = 0,
        dj0 = 2;
    int i, j, di, dj, n;
    for (n = 0; n < lmax + 1; ++n) {
        i = i0;
        di = di0;
        xarr(i) = xterm;
        j = j0;
        dj = dj0;
        yarr(j) = yterm;
        i = i0 + di - 1;
        j = j0 + dj - 1;
        while (i + 1 < N) {
            xarr(i) = xterm;
            xarr(i + 1) = xterm;
            di += 2;
            i += di;
            yarr(j) = yterm;
            yarr(j + 1) = yterm;
            dj += 2;
            j += dj - 1;
        }
        xterm *= x;
        i0 += 2 * n + 1;
        di0 += 2;
        yterm *= y;
        j0 += 2 * (n + 1) + 1;
        dj0 += 2;
    }
    n = 0;
    for (int l = 0; l < lmax + 1; ++l) {
        for (int m = -l; m < l + 1; ++m) {
            if ((l + m) % 2 == 0)
                basis(n) = xarr(n) * yarr(n);
            else
                basis(n) = xarr(n) * yarr(n) * z;
            ++n;
        }
    }
}

/**
Compute the polynomial basis at a vector of points.

*/
template <typename T> 
template <typename T1, typename T2> 
inline void Basis<T>::computePolyBasis ( 
    const MatrixBase<T1>& x,
    const MatrixBase<T2>& y,
    Matrix<T>& basis
) {
    int N = (lmax + 1) * (lmax + 1);
    int npts = x.cols();
    RowVector<T> x2 = x.cwiseProduct(x);
    RowVector<T> y2 = y.cwiseProduct(y);
    RowVector<T> z2 = RowVector<T>::Ones(npts) - x2 - y2;
    RowVector<T> z = z2.cwiseSqrt();
    Matrix<T> xarr(npts, N), yarr(npts, N);
    RowVector<T> xterm(npts), yterm(npts);
    xterm.setOnes();
    yterm.setOnes();
    int i0 = 0,
        di0 = 3,
        j0 = 0,
        dj0 = 2;
    int i, j, di, dj, n;
    for (n = 0; n < lmax + 1; ++n) {
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
    for (int l = 0; l < lmax + 1; ++l) {
        for (int m = -l; m < l + 1; ++m) {
            basis.col(n) = xarr.col(n).cwiseProduct(yarr.col(n));
            if ((l + m) % 2 != 0)
                basis.col(n) = basis.col(n).cwiseProduct(z.transpose());
            ++n;
        }
    }
}

/**
Compute the polynomial basis over the visible disk at a given
resolution. This is effectively a change of basis matrix from spherical
harmonic coefficients to pixel values on a grid.

*/
template <typename T>
inline void Basis<T>::computePolyMatrix (
    size_t res,
    Matrix<T>& P
) {
    RowVector<T> pts(RowVector<T>::LinSpaced(res, -1.0, 1.0));
    RowVector<T> y = pts.replicate(1, res);
    Matrix<T> tmp = y.replicate(res, 1);
    Eigen::Map<RowVector<T>> x(tmp.data(), res * res);
    P.resize(res * res, (lmax + 1) * (lmax + 1));
    computePolyBasis(y, x, P);
}

/**
Compute the illumination matrix (actually a vector) at a given resolution.

*/
template <typename T>
inline void Basis<T>::computeIlluminationMatrix (
    size_t res,
    const UnitVector<T>& source,
    Vector<T>& I
) {
    // Define our grid
    RowVector<T> pts(RowVector<T>::LinSpaced(res, -1.0, 1.0));
    RowVector<T> x = pts.replicate(1, res);
    Matrix<T> tmp = x.replicate(res, 1);
    Eigen::Map<RowVector<T>> y(tmp.data(), res * res);
    I.resize(res * res);

    // Get the source vector components
    T sx = source(0);
    T sy = source(1);
    T sz = source(2);

    // Compute the illumination matrix
    T b = -sz;
    RowVector<T> yrot(res * res);
    RowVector<T> Ones = RowVector<T>::Ones(res * res);
    if (unlikely((sx == 0) && (sy == 0))) {
        if (sz < 0) {
            I.setZero();
            return;
        }
        yrot = y;
    } else {
        T invsr = T(1.0) / sqrt(sx * sx + sy * sy);
        T cosw = sy * invsr;
        T sinw = -sx * invsr;
        yrot = -x * sinw + y * cosw;
    }
    RowVector<T> z = (Ones - x.cwiseProduct(x) - y.cwiseProduct(y)).cwiseSqrt();
    I = sqrt(T(1.0) - T(b * b)) * yrot - b * z;

    // Mask the night side
    I = (I.array() < 0).select(0, I);
}

} // namespace basis
} // namespace starry
#endif