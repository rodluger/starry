/**
Solutions to the surface integral over the visible region of a spherical 
harmonic map during a single-body occultation using Green's theorem.

*/

#ifndef _STARRY_SOLVER_H_
#define _STARRY_SOLVER_H_

#include "utils.h"
#include "ellip.h"
#include "errors.h"

namespace starry2 {
namespace solver {

    using namespace starry2::utils;

    /**
    Greens integration housekeeping data

    */
    template <class Scalar>
    class Greens
    {

    protected:

        // Indices
        int lmax;
        int N;

        // Autodiff stuff
        ADScalar<Scalar, 2> b_g;
        ADScalar<Scalar, 2> r_g;
        RowVector<ADScalar<Scalar, 2>> s_g;

        // Internal methods
        template <typename T>
        inline void compute_ (
            const T& b,
            const T& r,
            RowVector<T>& sT
        );

        template <bool GRADIENT=false>
        inline void computeS0AndS2_ (
            const Scalar& b,
            const Scalar& r
        );

    public:

        // Solutions
        RowVector<Scalar> sT;
        RowVector<Scalar> dsTdb;
        RowVector<Scalar> dsTdr;

        // Constructor
        explicit Greens(
            int lmax
        ) :
            lmax(lmax),
            N((lmax + 1) * (lmax + 1)),
            b_g(0.0, Vector<Scalar>::Unit(2, 0)),
            r_g(0.0, Vector<Scalar>::Unit(2, 1)),
            s_g(N),
            sT(RowVector<Scalar>::Zero(N)),
            dsTdb(RowVector<Scalar>::Zero(N)),
            dsTdr(RowVector<Scalar>::Zero(N))
        {

        }

        // Methods
        template <bool GRADIENT=false>
        inline void compute (
            const Scalar& b,
            const Scalar& r
        );

    };

    template <class Scalar>
    template <bool GRADIENT>
    inline void Greens<Scalar>::computeS0AndS2_ (
        const Scalar& b, 
        const Scalar& r
    ) {

        // TODO: Dummy calculations for now!

        // s0
        sT(0) = sqrt(b) + sqrt(r);
        if (GRADIENT) {
            dsTdb(0) = 0.5 * pow(b, -0.5);
            dsTdr(0) = 0.5 * pow(r, -0.5);
        }

        // s2
        sT(2) = b * b + r * r;
        if (GRADIENT) {
            dsTdb(2) = 2 * b;
            dsTdr(2) = 2 * r;
        }

    }

    /**
    Compute the `s^T` occultation solution vector;
    internal templated function for scalar/ADScalar
    capability.

    */
    template <class Scalar>
    template <typename T>
    inline void Greens<Scalar>::compute_ (
        const T& b, 
        const T& r,
        RowVector<T>& sT
    ) {

        // TODO: Dummy calculations for now!
        if (N > 0) {
            sT(1) = b + r;
        }
        for (int n = 3; n < N; ++n) {
            sT(n) = pow(b, n) + n * r;
        }

    }

    /**
    Compute the `s^T` occultation solution vector
    with or without the gradient.

    */
    template <class Scalar>
    template <bool GRADIENT>
    inline void Greens<Scalar>::compute (
        const Scalar& b, 
        const Scalar& r
    ) {
        // Compute the special terms
        computeS0AndS2_<GRADIENT>(b, r);
        
        // Compute the rest
        if (!GRADIENT) {
            compute_(b, r, sT);
        } else {
            b_g.value() = b;
            r_g.value() = r;
            compute_(b_g, r_g, s_g);
            if (N > 0) {
                sT(1) = s_g(1).value();
                dsTdb(1) = s_g(1).derivatives()(0);
                dsTdr(1) = s_g(1).derivatives()(1);
            }
            for (int n = 3; n < N; ++n) {
                sT(n) = s_g(n).value();
                dsTdb(n) = s_g(n).derivatives()(0);
                dsTdr(n) = s_g(n).derivatives()(1);
            }
        }
    }

} // namespace solver
} // namespace starry2

#endif