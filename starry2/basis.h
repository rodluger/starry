/**
Spherical harmonic, polynomial, and Green's basis utilities.

*/

#ifndef _STARRY_BASIS_H_
#define _STARRY_BASIS_H_

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseLU>
#include <limits>
#include "errors.h"
#include "utils.h"
#include "tables.h"

namespace basis {

    using namespace utils;
    using std::abs;

    /**
    Contraction coefficient for the Ylms

    */
    template <typename T>
    T C(int p, int q, int k) {
        if ((p > k) && ((p - k) % 2 == 0)) {
            return T(0);
        } else if ((q > p) && ((q - p) % 2 == 0)) {
            return T(0);
        } else {
            return tables::half_factorial<T>(k) /
                        (tables::half_factorial<T>(q) *
                         tables::half_factorial<T>(k - p) *
                         tables::half_factorial<T>(p - q));
        }
    }

    /**
    Return the normalization constant A for a Ylm

    */
    template <typename T>
    T Norm(int l, int m) {
        return sqrt((1. / (4 * pi<T>())) *
                    (2 - (int)(m == 0)) *
                    (2 * l + 1) *
                    tables::factorial<T>(l - abs(m)) /
                    tables::factorial<T>(l + abs(m)));
    }

    /**
    Return the B coefficient for a Ylm

    */
    template <typename T>
    T B(int l, int m, int j, int k) {

        // Is it zero?
        int i1 = l + m + k - 1;
        int i2 = -l + m + k - 1;
        if ((i1 < 0) && (i1 % 2 == 0)) return T(INFINITY);
        else if ((i2 < 0) && (i2 % 2 == 0)) return T(0);
        else if (m - j < 0) return T(0);
        else if (l - m - k < 0) return T(0);

        // Ok, let's calculate it
        T two_l = 1;
        for (int i=0; i < l; i++)
            two_l *= 2;
        T a = tables::factorial<T>(m);
        T b = tables::half_factorial<T>(i1);
        T c = tables::factorial<T>(j);
        T d = tables::factorial<T>(k);
        T e = tables::factorial<T>(m - j);
        T f = tables::factorial<T>(l - m - k);
        T g = tables::half_factorial<T>(i2);
        return two_l * a * b / (c * d * e * f * g);
    }

    /**
    Return the ijk tensor element of the spherical harmonic Ylm

    */
    template <typename T>
    T Lijk(int l, int m, int i, int j, int k) {
        if ((i == abs(m) + k) && (j <= abs(m))) {
            if ((m >= 0) && (j % 2 == 0)) {
                if ((j / 2) % 2 == 0)
                    return Norm<T>(l, m) * B<T>(l, m, j, k);
                else
                    return -Norm<T>(l, m) * B<T>(l, m, j, k);
            } else if ((m < 0) && (j % 2 == 1)) {
                if (((j - 1) / 2) % 2 == 0)
                    return Norm<T>(l, -m) * B<T>(l, -m, j, k);
                else
                    return -Norm<T>(l, -m) * B<T>(l, -m, j, k);
            } else {
                return T(0);
            }
        } else {
            return T(0);
        }
    }

    /**
    Compute the first change of basis matrix, `A_1`.

    NOTE: This routine is **not optimized**. We could compute the
    elements of the sparse matrix `A1` directly, but instead we compute
    the elements of the tensors `Ylm`, contract these tensors to column vectors
    in the dense version of `A1`, then convert it to sparse form.
    Fortunately, this routine is only run **once** when a `Map` class is
    instantiated.
    */
    template <typename T>
    void computeA1(int lmax, Eigen::SparseMatrix<T>& A1, T tol=10 * std::numeric_limits<T>::epsilon()) {
        int l, m;
        int n = 0;
        int i, j, k, p, q, v;
        int N = (lmax + 1) * (lmax + 1);
        T coeff;
        Matrix<T> Ylm0(lmax + 1, lmax + 1);
        Matrix<T> Ylm1(lmax + 1, lmax + 1);
        Matrix<T> A1Dense = Matrix<T>::Zero(N, N);

        // Iterate over the spherical harmonic orders and degrees
        for (l=0; l<lmax+1; l++) {
            for (m=-l; m<l+1; m++) {

                // Compute the contracted polynomial tensor
                for (i=0;i<l+1;i++) {
                    for (j=0;j<l+1;j++){
                        Ylm0(i, j) = 0.;
                        Ylm1(i, j) = 0.;
                    }
                }
                for (k=0; k<l+1; k++) {
                    for (i=k; i<l+1; i++) {
                        for (j=0; j<i-k+1; j++) {
                            coeff = Lijk<T>(l, m, i, j, k);
                            if (coeff != 0) {
                                if (k == 0) {
                                    Ylm0(i, j) += coeff;
                                } else if (k == 1) {
                                    Ylm1(i, j) += coeff;
                                } else if ((k % 2) == 0) {
                                    // Even power of z
                                    for (p=0; p<k+1; p+=2) {
                                        for (q=0; q<p+1; q+=2) {
                                            if ((p / 2) % 2 == 0)
                                                Ylm0(i - k + p, j + q) += C<T>(p, q, k) * coeff;
                                            else
                                                Ylm0(i - k + p, j + q) -= C<T>(p, q, k) * coeff;
                                        }
                                    }
                                } else {
                                    // Odd power of z
                                    for (p=0; p<k+1; p+=2) {
                                        for (q=0; q<p+1; q+=2) {
                                            if ((p / 2) % 2 == 0)
                                                Ylm1(i - k + p + 1, j + q) += C<T>(p, q, k - 1) * coeff;
                                            else
                                                Ylm1(i - k + p + 1, j + q) -= C<T>(p, q, k - 1) * coeff;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Now contract the tensor down to a column vector in A1Dense
                v = 0;
                for (i=0; i<l+1; i++) {
                    for (j=0; j<i+1; j++) {
                        if (abs(Ylm0(i, j)) > tol)
                            A1Dense(v, n) = Ylm0(i, j);
                        v++;
                        if (j < i) {
                            if (abs(Ylm1(i, j)) > tol)
                                A1Dense(v, n) = Ylm1(i, j);
                            v++;
                        }
                    }
                }

                // Next term
                n++;
            }
        }

        // Make sparse
        A1 = A1Dense.sparseView();

        return;
    }

    /**
    Compute the full change of basis matrix, `A`

    */
    template <typename T>
    void computeA(int lmax, Eigen::SparseMatrix<T>& A1, Eigen::SparseMatrix<T>& A, T tol=10 * std::numeric_limits<T>::epsilon()) {
        int i, n, l, m, mu, nu;
        int N = (lmax + 1) * (lmax + 1);

        // Let's compute the inverse of A2, since it's easier
        Matrix<T> A2InvDense = Matrix<T>::Zero(N, N);
        n = 0;
        for (l=0; l<lmax+1; l++) {
            for (m=-l; m<l+1; m++){
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
                n++;
            }
        }

        // Sparse dot A2 into A1
        Eigen::SparseMatrix<T> A2Inv = A2InvDense.sparseView();
        Eigen::SparseLU<Eigen::SparseMatrix<T>> solver;
        solver.compute(A2Inv);
        if (solver.info() != Eigen::Success) {
            throw errors::LinearAlgebraError("Error computing the change of basis matrix `A2`.");
        }
        A = solver.solve(A1);
        if (solver.info() != Eigen::Success) {
            throw errors::LinearAlgebraError("Error computing the change of basis matrix `A1`.");
        }

        return;
    }

    /**
    Compute the change of basis from limb darkening coefficients
    to spherical harmonic coefficients

    */
    template <typename T>
    void computeU(int lmax, Matrix<T>& U) {
        T amp;
        Matrix<T> LT, YT;
        LT.setZero(lmax + 1, lmax + 1);
        YT.setZero(lmax + 1, lmax + 1);

        // Compute L^T and Y^T
        for (int l = 0; l < lmax + 1; l++) {
            amp = pow(2, l) * sqrt((2 * l + 1) / (4 * pi<T>())) / tables::factorial<T>(l);
            for (int k = 0; k < l + 1; k++) {
                if ((k + 1) % 2 == 0)
                    LT(k, l) = tables::choose<T>(l, k);
                else
                    LT(k, l) = -tables::choose<T>(l, k);
                YT(k, l) = amp * tables::choose<T>(l, k) * tables::half_factorial<T>(k + l - 1) / tables::half_factorial<T>(k - l - 1);
            }
        }

        // Compute U
        Eigen::HouseholderQR<Matrix<T>> solver(lmax + 1, lmax + 1);
        solver.compute(YT);
        U = solver.solve(LT);

    }

    /**
    Return the n^th term of the `r` phase curve solution vector

    */
    template <typename T>
    T rn(int mu, int nu) {
        T a, b, c;
        if (is_even(mu, 2) && is_even(nu, 2)) {
            a = tables::gamma_sup<T>(mu / 4);
            b = tables::gamma_sup<T>(nu / 4);
            c = tables::gamma<T>((mu + nu) / 4 + 2);
            return a * b / c;
        } else if (is_even(mu - 1, 2) && is_even(nu - 1, 2)) {
            a = tables::gamma_sup<T>((mu - 1) / 4);
            b = tables::gamma_sup<T>((nu - 1) / 4);
            c = tables::gamma_sup<T>((mu + nu - 2) / 4 + 2) * (2.0 / root_pi<T>());
            return a * b / c;
        } else {
            return 0;
        }
    }

    /**
    Compute the `r^T` phase curve solution vector

    */
    template <typename T>
    void computerT(int lmax, VectorT<T>& rT) {
        rT.resize((lmax + 1) * (lmax + 1));
        int l, m, mu, nu;
        int n = 0;
        for (l=0; l<lmax+1; l++) {
            for (m=-l; m<l+1; m++) {
                mu = l - m;
                nu = l + m;
                rT(n) = rn<T>(mu, nu);
                n++;
            }
        }
        return;
    }

    /**
    Basis transform matrices

    */
    template <class T>
    class Basis {

        public:

            const int lmax;                                                     /**< The highest degree of the map */
            Eigen::SparseMatrix<T> A1;                                          /**< The polynomial change of basis matrix */
            Eigen::SparseMatrix<T> A;                                           /**< The Green's change of basis matrix */
            Matrix<T> U;                                                        /**< The limb darkening change of basis matrix */
            VectorT<T> rT;                                                      /**< The rotation solution vector */
            VectorT<T> rTA1;                                                    /**< The rotation vector times the `Ylm` change of basis matrix */

            // Constructor: compute the matrices
            Basis(int lmax) : lmax(lmax) {
                computeA1(lmax, A1);
                computeA(lmax, A1, A);
                computeU(lmax, U);
                computerT(lmax, rT);
                rTA1 = rT * A1;
            }

    };

}; // namespace basis

#endif
