/**
\file filter.h
\brief Map filter operations.

*/

#ifndef _STARRY_FILTER_H_
#define _STARRY_FILTER_H_

#include "utils.h"
#include "basis.h"

namespace starry { 
namespace filter {

using namespace utils;


/**
Filter operations on a spherical harmonic map.

*/
template <typename Scalar>
class Filter {

protected:

    basis::Basis<Scalar>& B;
    const int ydeg;
    const int Ny;                                                              /**< Number of spherical harmonic `(l, m)` coefficients */
    const int udeg;
    const int Nu;                                                              /**< Number of limb darkening coefficients */
    const int fdeg;
    const int Nf;                                                              /**< Number of filter `(l, m)` coefficients */
    const int deg;
    const int N;

public:

    Matrix<Scalar> F;                                                          /**< The filter operator in the polynomial basis. TODO: Make sparse? */
    Vector<Scalar> bu;
    Vector<Scalar> bf;

    // Constructor: compute the matrices
    explicit Filter(
        basis::Basis<Scalar>& B
    ) :
        B(B),
        ydeg(B.ydeg), 
        Ny((ydeg + 1) * (ydeg + 1)),
        udeg(B.udeg),
        Nu(udeg + 1),
        fdeg(B.fdeg),
        Nf((fdeg + 1) * (fdeg + 1)),
        deg(B.deg),
        N((deg + 1) * (deg + 1))
    {
    
        //

    }

    /**
    
    */
    template <bool GRADIENT=false>
    inline void computePolynomialProductMatrix (
        const int plmax, 
        const Vector<Scalar>& p,
        Matrix<Scalar>& M,
        Vector<Matrix<Scalar>>& dMdp
    ) {
        bool odd1;
        int l, n;
        int n1 = 0, n2 = 0;
        M.setZero((plmax + ydeg + 1) * (plmax + ydeg + 1), Ny);
        if (GRADIENT) {
            dMdp.resize((plmax + 1) * (plmax + 1));
            for (n = 0; n < (plmax + 1) * (plmax + 1); ++n)
                dMdp(n).setZero((plmax + ydeg + 1) * (plmax + ydeg + 1), Ny);
        }
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
                            if (GRADIENT) {
                                dMdp[n2](n - 4 * l + 2, n1) += 1;
                                dMdp[n2](n - 2, n1) -= 1;
                                dMdp[n2](n + 2, n1) -= 1;
                            }
                        } else {
                            M(n, n1) += p(n2);
                            if (GRADIENT) {
                                dMdp[n2](n, n1) += 1;
                            }
                        }
                        ++n2;
                    }
                }
                ++n1;
            }
        }  
    }

    /**

    */
    template <bool GRADIENT=false>
    inline void computePolynomialProduct(
        const int lmax1, 
        const Vector<Scalar>& p1, 
        const int lmax2,
        const Vector<Scalar>& p2, 
        Vector<Scalar>& p1p2,
        Matrix<Scalar>& grad_p1,
        Matrix<Scalar>& grad_p2
    ) {
        int n1, n2, l1, m1, l2, m2, l, n;
        bool odd1;
        int N1 = (lmax1 + 1) * (lmax1 + 1);
        int N2 = (lmax2 + 1) * (lmax2 + 1);
        int N12 = (lmax1 + lmax2 + 1) * (lmax1 + lmax2 + 1);
        p1p2.setZero(N);
        Scalar mult;
        n1 = 0;
        if (GRADIENT) {
            grad_p1.setZero(N12, N1);
            grad_p2.setZero(N12, N2);
        }
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
                            if (GRADIENT) {
                                grad_p1(n - 4 * l + 2, n1) += p2(n2);
                                grad_p2(n - 4 * l + 2, n2) += p1(n1);
                                grad_p1(n - 2, n1) -= p2(n2);
                                grad_p2(n - 2, n2) -= p1(n1);
                                grad_p1(n + 2, n1) -= p2(n2);
                                grad_p2(n + 2, n2) -= p1(n1);
                            }  
                        } else {
                            p1p2(n) += mult;
                            if (GRADIENT) {
                                grad_p1(n, n1) += p2(n2);
                                grad_p2(n, n2) += p1(n1);
                            }
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
    void compute (
        const Vector<Scalar>& u,
        const Vector<Scalar>& f
    ) {

        Vector<Matrix<Scalar>> DFDp; // not used
        Matrix<Scalar> DpDpu; // not used
        Matrix<Scalar> DpDpf; // not used

        // Compute the two polynomials
        Vector<Scalar> tmp = B.U1 * u;
        Scalar norm = Scalar(1.0) / B.rT.segment(0, (udeg + 1) * (udeg + 1)).dot(tmp);
        Vector<Scalar> pu = tmp * norm * pi<Scalar>();
        Vector<Scalar> pf;
        pf = B.A1_f * f;

        // Multiply them
        Vector<Scalar> p;
        if (udeg > fdeg) {
            computePolynomialProduct<false>(udeg, pu, fdeg, pf, p, DpDpu, DpDpf);
        } else {
            computePolynomialProduct<false>(fdeg, pf, udeg, pu, p, DpDpf, DpDpu);
        }

        // Compute the polynomial filter operator
        computePolynomialProductMatrix<false>(udeg + fdeg, p, F, DFDp);

    }

    /**
    Compute the gradient of the polynomial filter operator.

    */
    void compute (
        const Vector<Scalar>& u,
        const Vector<Scalar>& f,
        const Matrix<Scalar>& bF
    ) {

        Vector<Matrix<Scalar>> DFDp;
        Matrix<Scalar> DpDpu;
        Matrix<Scalar> DpDpf;

        // Compute the two polynomials
        Vector<Scalar> tmp = B.U1 * u;
        Scalar norm = Scalar(1.0) / B.rT.segment(0, (udeg + 1) * (udeg + 1)).dot(tmp);
        Vector<Scalar> pu = tmp * norm * pi<Scalar>();
        Vector<Scalar> pf;
        pf = B.A1_f * f;

        // Multiply them
        Vector<Scalar> p;
        if (udeg > fdeg) {
            computePolynomialProduct<true>(udeg, pu, fdeg, pf, p, DpDpu, DpDpf);
        } else {
            computePolynomialProduct<true>(fdeg, pf, udeg, pu, p, DpDpf, DpDpu);
        }

        // Compute the polynomial filter operator
        computePolynomialProductMatrix<true>(udeg + fdeg, p, F, DFDp);

        // Compute the limb darkening derivatives
        Vector<Matrix<Scalar>> DFDu(Nu);
        Matrix<Scalar> DpuDu = pi<Scalar>() * norm * B.U1 - 
            pu * B.rT.segment(0, (udeg + 1) * (udeg + 1)) * B.U1 * norm;
        for (int l = 0; l < udeg + 1; ++l) {
            DFDu(l).setZero(N, Ny);
        }
        Matrix<Scalar> DpDu = DpDpu * DpuDu;
        for (int j = 0; j < (udeg + fdeg + 1) * (udeg + fdeg + 1); ++j) {
            for (int l = 0; l < udeg + 1; ++l) {
                DFDu(l) += DFDp(j) * DpDu(j, l);
            }
        }

        // Compute the filter derivatives
        Vector<Matrix<Scalar>> DFDf(Nf);
        Matrix<Scalar> DpfDf = B.A1_f;
        for (int l = 0; l < Nf; ++l) {
            DFDf(l).setZero(N, Ny);
        }
        Matrix<Scalar> DpDf = DpDpf * DpfDf;
        for (int j = 0; j < (udeg + fdeg + 1) * (udeg + fdeg + 1); ++j) {
            for (int l = 0; l < Nf; ++l) {
                DFDf(l) += DFDp(j) * DpDf(j, l);
            }
        }

        // Backprop (TODO: compute these directly)
        bu.resize(Nu);
        if (udeg > 0) {
            for (int l = 0; l < Nu; ++l) {
                bu(l) = DFDu(l).cwiseProduct(bF).sum();
            }
        }
        bf.resize(Nf);
        if (fdeg > 0) {
            for (int l = 0; l < Nf; ++l) {
                bf(l) = DFDf(l).cwiseProduct(bF).sum();
            }
        } else {
            bf.resize(0);
        }

    }

};


} // namespace filter
} // namespace starry
#endif