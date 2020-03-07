
/**
\file occultation.h
\brief Solver for occultations of bodies with a night side (i.e., in reflected light).

*/

#ifndef _STARRY_REFLECTED_OCCULTATION_H_
#define _STARRY_REFLECTED_OCCULTATION_H_

#include "../utils.h"
#include "../solver.h"
#include "constants.h"
#include "geometry.h"
#include "primitive.h"
#include "phasecurve.h"

namespace starry {
namespace reflected {
namespace occultation {

using namespace utils;
using namespace geometry;
using namespace primitive;

template <class T> class Occultation {

    using Scalar = typename T::Scalar;

protected:

    int ydeg;
    int N2;
    int N1;
    Eigen::SparseMatrix<Scalar> A2;
    Eigen::SparseMatrix<Scalar> A2Inv;
    Matrix<T> I;
    Vector<T> kappa;
    Vector<T> lam;
    Vector<T> xi;
    Vector<T> PIntegral;
    Vector<T> QIntegral;
    Vector<T> TIntegral;

    T costheta;
    T sintheta;

    // Helper solvers
    phasecurve::PhaseCurve<Scalar> RO;
    solver::Solver<T, true> G;

    /**
    
        Compute the change of basis matrix `A2` and its inverse.

    */
    void computeA2() {

        int i, n, l, m, mu, nu;
        Matrix<Scalar> A2InvDense = Matrix<Scalar>::Zero(N2, N2);
        n = 0;
        for (l = 0; l < ydeg + 2; ++l) {
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
            throw std::runtime_error(
                "Error computing the change of basis matrix `A2`.");
        }
        Eigen::SparseMatrix<Scalar> Id = Matrix<Scalar>::Identity(N2, N2).sparseView();
        A2 = solver.solve(Id);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error(
                "Error computing the change of basis matrix `A2`.");
        }

        // Reshape. A2 should be (N2 x N2) and A2^-1 should be (N1 x N1).
        A2Inv = A2InvDense.block(0, 0, N1, N1).sparseView();

    }

    /**

        Illumination matrix.

        TODO: We can backprop through this pretty easily.
        TODO: Make me sparse!

    */
    inline void computeI(const T& b, const T& theta) {
        
        // Reset
        I.setZero();

        // NOTE: 3 / 2 is the starry normalization for reflected light maps
        T y0 = sqrt(1 - b * b);
        T x = -y0 * sin(theta);
        T y = y0 * cos(theta);
        T z = -b;
        Vector<T> p(4);
        p << 0, 1.5 * x, 1.5 * z, 1.5 * y;
        
        // Populate the matrix
        int n1 = 0;
        int n2 = 0;
        int l, n;
        bool odd1;
        for (int l1 = 0; l1 < ydeg + 1; ++l1) {
            for (int m1 = -l1; m1 < l1 + 1; ++m1) {
                if (is_even(l1 + m1)) odd1 = false;
                else odd1 = true;
                n2 = 0;
                for (int l2 = 0; l2 < 2; ++l2) {
                    for (int m2 = -l2; m2 < l2 + 1; ++m2) {
                        l = l1 + l2;
                        n = l * l + l + m1 + m2;
                        if (odd1 && (!is_even(l2 + m2))) {
                            I(n - 4 * l + 2, n1) += p(n2);
                            I(n - 2, n1) -= p(n2);
                            I(n + 2, n1) -= p(n2);
                        } else {
                            I(n, n1) += p(n2);
                        }
                        n2 += 1;
                    }
                }
                n1 += 1;
            }
        }

    }

    /**
        Weight the solution vector by a cosine-like illumination profile.
        Note that we need I to transform Greens --> Greens.

    */
    inline RowVector<T> illuminate(const T& b, const T& theta, const RowVector<T>& sT) {
        computeI(b, theta);
        RowVector<T> sTw = sT * A2;
        sTw = sTw * I;
        sTw = sTw * A2Inv;
        return sTw;
    }

    /**
        AutoDiff-enabled standard starry occultation solution.

    */
    inline RowVector<T> sTe(const T& bo, const T& ro) {
        G.compute(bo, ro);
        return G.sT;
    }

    /**
        TODO: Figure out the geometry. Propagate the derivs.
    */
    inline RowVector<T> rTr(const T& x, const T& y, const T& z) {
        RowVector<T> rT(N1);
        rT.setZero();
        return rT;
    }

public:

    int code;
    RowVector<T> sT;

    explicit Occultation(int ydeg, phasecurve::PhaseCurve<Scalar> RO) : 
        ydeg(ydeg),
        N2((ydeg + 2) * (ydeg + 2)),
        N1((ydeg + 1) * (ydeg + 1)),
        I(N2, N1),
        PIntegral(N2),
        QIntegral(N2),
        TIntegral(N2),
        RO(RO),
        G(ydeg + 1),
        sT(N2)
    {

        // Compute the change of basis matrix (constant)
        computeA2();

    }

    /**
        Compute the full solution vector s^T.

    */
    inline void compute(const T& b, const T& theta, const T& bo, const T& ro) {
        
        // Get the angles of intersection
        costheta = cos(theta);
        sintheta = sin(theta);
        code = get_angles(b, theta, costheta, sintheta, bo, ro, kappa, lam, xi);

        // The full solution vector is a combination of the
        // current vector, the standard starry vector, and the
        // reflected light phase curve solution vector. The contributions
        // of each depend on the integration code.

        if (code == FLUX_ZERO) {

            // Complete occultation!
            sT.setZero(N2);

        } else if (code == FLUX_SIMPLE_OCC) {

            sT = illuminate(b, theta, sTe(bo, ro));

        } else if (code == FLUX_SIMPLE_REFL) {

            T y0 = sqrt(1 - b * b);
            T xs = -y0 * sintheta;
            T ys = y0 * costheta;
            T zs = -b;
            sT = rTr(xs, ys, zs);

        } else if (code == FLUX_SIMPLE_OCC_REFL) {

            T y0 = sqrt(1 - b * b);
            T xs = -y0 * sintheta;
            T ys = y0 * costheta;
            T zs = -b;
            sT = illuminate(b, theta, sTe(bo, ro)) + rTr(-xs, -ys, -zs);

        } else {

            // Compute the primitive integrals
            computeP(ydeg + 1, bo, ro, kappa, PIntegral);
            computeQ(ydeg + 1, lam, QIntegral);
            computeT(ydeg + 1, b, theta, xi, TIntegral);
            sT = (PIntegral + QIntegral + TIntegral).transpose();

            // The full solution vector is a combination of the
            // current vector, the standard starry vector, and the
            // reflected light phase curve solution vector.
            if ((code == FLUX_DAY_OCC) || (code == FLUX_TRIP_DAY_OCC)) {

                T y0 = sqrt(1 - b * b);
                T xs = -y0 * sintheta;
                T ys = y0 * costheta;
                T zs = -b;
                sT = rTr(xs, ys, zs) - illuminate(b, theta, sT);

            } else if ((code == FLUX_NIGHT_OCC) || (code == FLUX_TRIP_NIGHT_OCC)) {

                T y0 = sqrt(1 - b * b);
                T xs = -y0 * sintheta;
                T ys = y0 * costheta;
                T zs = -b;
                sT = illuminate(b, theta, sTe(bo, ro) + sT) + rTr(-xs, -ys, -zs);

            } else if ((code == FLUX_DAY_VIS) || (code == FLUX_QUAD_DAY_VIS)) {

                // The solution vector is *just* the reflected light solution vector.
                sT = illuminate(b, theta, sT);

            } else if ((code == FLUX_NIGHT_VIS) || (code == FLUX_QUAD_NIGHT_VIS)) {

                sT = illuminate(b, theta, sTe(bo, ro) - sT);

            } else {

                // ?!
                throw std::runtime_error("Unexpected branch in `sT`.");

            }

        }

    }

};

} // namespace occultation
} // namespace reflected
} // namespace starry

#endif