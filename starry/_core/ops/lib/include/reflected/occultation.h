
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

    // Misc
    int deg;
    int N2;
    int N1;
    Eigen::SparseMatrix<Scalar> A1;
    Eigen::SparseMatrix<Scalar> AInv;
    Eigen::SparseMatrix<Scalar> A2;
    Eigen::SparseMatrix<Scalar> A2Inv;
    Matrix<T> I;
    Vector<T> kappa;
    Vector<T> lam;
    Vector<T> xi;
    Vector<T> PIntegral;
    Vector<T> QIntegral;
    Vector<T> TIntegral;
    RowVector<T> PQT;

    // Angles
    T costheta;
    T sintheta;
    Vector<T> cosnt;
    Vector<T> sinnt;
    Vector<T> cosmt;
    Vector<T> sinmt;

    // Helper solvers
    phasecurve::PhaseCurve<T> R;
    solver::Solver<T, true> G;

    /**
    
        Compute the change of basis matrix `A2` and its inverse.

    */
    void computeA2() {

        int i, n, l, m, mu, nu;
        Matrix<Scalar> A2InvDense = Matrix<Scalar>::Zero(N2, N2);
        n = 0;
        for (l = 0; l < deg + 2; ++l) {
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
            args << "N2 = " << N2;
            throw StarryException(
                "Error computing the change of basis matrix `A2Inv`.",
                "reflected/occultation.h",
                "Occultation.computeA2",
                args.str()
            );
        }
        Eigen::SparseMatrix<Scalar> Id = Matrix<Scalar>::Identity(N2, N2).sparseView();
        A2 = solver.solve(Id);
        if (solver.info() != Eigen::Success) {
            std::stringstream args;
            args << "N2 = " << N2;
            throw StarryException(
                "Error computing the change of basis matrix `A2`.",
                "reflected/occultation.h",
                "Occultation.computeA2",
                args.str()
            );
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
        T y0 = sqrt(1 - b * b);
        T x = -y0 * sin(theta);
        T y = y0 * cos(theta);
        T z = -b;
        Vector<T> p(4);
        p << 0, x, z, y;
        
        // Populate the matrix
        int n1 = 0;
        int n2 = 0;
        int l, n;
        bool odd1;
        for (int l1 = 0; l1 < deg + 1; ++l1) {
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

    */
    inline RowVector<T> sTr(const T& b, const T& theta) {
        
        // Compute the reflection solution in the terminator frame
        R.compute(b);
        
        // Transform to ylms and rotate into the occultor frame
        RowVector<T> rTA1 = R.rT * A1;
        RowVector<T> rTA1R(N1);
        cosnt(1) = cos(theta);
        sinnt(1) = sin(-theta);
        for (int n = 2; n < deg + 1; ++n) {
            cosnt(n) = 2.0 * cosnt(n - 1) * cosnt(1) - cosnt(n - 2);
            sinnt(n) = 2.0 * sinnt(n - 1) * cosnt(1) - sinnt(n - 2);
        }
        int n = 0;
        for (int l = 0; l < deg + 1; ++l) {
            for (int m = -l; m < 0; ++m) {
                cosmt(n) = cosnt(-m);
                sinmt(n) = -sinnt(-m);
                ++n;
            }
            for (int m = 0; m < l + 1; ++m) {
                cosmt(n) = cosnt(m);
                sinmt(n) = sinnt(m);
                ++n;
            }
            for (int j = 0; j < 2 * l + 1; ++j) {
                rTA1R(l * l + j) = rTA1(l * l + j) * cosmt(l * l + j) + rTA1(l * l + 2 * l - j) * sinmt(l * l + j);
            }
        }

        // Transform back to Green's polynomials
        return rTA1R * AInv;
    }

public:

    int code;
    RowVector<T> sT;

    explicit Occultation(int deg, const Eigen::SparseMatrix<Scalar>& A1) : 
        deg(deg),
        N2((deg + 2) * (deg + 2)),
        N1((deg + 1) * (deg + 1)),
        A1(A1),
        I(N2, N1),
        PIntegral(N2),
        QIntegral(N2),
        TIntegral(N2),
        PQT(N2),
        R(deg),
        G(deg + 1),
        sT(N1)
    {

        // Compute the change of basis matrix (constant)
        computeA2();

        // Compute AInv (constant)
        Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> solver;
        solver.compute(A1);
        if (solver.info() != Eigen::Success) {
            std::stringstream args;
            args << "N2 = " << N2 << ", " << "N1 = " << N1;
            throw StarryException(
                "Error computing the change of basis matrix `A1`.",
                "reflected/occultation.h",
                "Occultation",
                args.str()
            );
        }
        AInv = solver.solve(A2Inv);

        // Rotation vectors
        cosnt.resize(max(2, deg + 1));
        cosnt(0) = 1.0;
        sinnt.resize(max(2, deg + 1));
        sinnt(0) = 0.0;
        cosmt.resize(N1);
        sinmt.resize(N1);

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
            sT.setZero(N1);

        } else if (code == FLUX_SIMPLE_OCC) {

            // The occultor is blocking all of the nightside 
            // and some dayside flux
            sT = illuminate(b, theta, sTe(bo, ro));

        } else if (code == FLUX_SIMPLE_REFL) {

            // The total flux is the full dayside flux
            sT = sTr(b, theta);

        } else if (code == FLUX_SIMPLE_OCC_REFL) {

            // The occultor is only blocking dayside flux
            sT = illuminate(b, theta, sTe(bo, ro)) + sTr(-b, theta + pi<T>());

        } else if (code == FLUX_NOON) {

            // The substellar point is the center of the disk, so this is 
            // analytically equivalent to the linear limb darkening solution
            sT = illuminate(b, theta, sTe(bo, ro));

        } else {

            // These cases require us to solve incomplete
            // elliptic integrals.

            // Compute the primitive integrals
            computeP(deg + 1, bo, ro, kappa, PIntegral);
            computeQ(deg + 1, lam, QIntegral);
            computeT(deg + 1, b, theta, xi, TIntegral);
            PQT = (PIntegral + QIntegral + TIntegral).transpose();

            if ((code == FLUX_DAY_OCC) || (code == FLUX_TRIP_DAY_OCC)) {

                //
                sT = sTr(b, theta) - illuminate(b, theta, PQT);

            } else if ((code == FLUX_NIGHT_OCC) || (code == FLUX_TRIP_NIGHT_OCC)) {

                //
                sT = illuminate(b, theta, sTe(bo, ro) + PQT) + sTr(-b, theta + pi<T>());

            } else if ((code == FLUX_DAY_VIS) || (code == FLUX_QUAD_DAY_VIS)) {

                // The solution vector is *just* the reflected light solution vector.
                sT = illuminate(b, theta, PQT);

            } else if ((code == FLUX_NIGHT_VIS) || (code == FLUX_QUAD_NIGHT_VIS)) {

                //
                sT = illuminate(b, theta, sTe(bo, ro) - PQT);

            } else {

                // ?!
                std::stringstream args;
                args << "b = " << b << ", " 
                     << "theta = " << theta << ", "  
                     << "bo = " << bo << ", "  
                     << "ro = " << ro;
                throw StarryException(
                    "Unexpected branch.",
                    "reflected/occultation.h",
                    "Occultation.compute",
                    args.str()
                );

            }

        }

    }

};

} // namespace occultation
} // namespace reflected
} // namespace starry

#endif