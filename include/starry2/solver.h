/**
Solutions to the surface integral over the visible region of a spherical 
harmonic map during a single-body occultation using Green's theorem.

*/

#ifndef _STARRY_SOLVER_H_
#define _STARRY_SOLVER_H_

#include "utils.h"

namespace starry2 {
namespace solver {

    using namespace starry2::utils;


    /**
    Greens integration housekeeping data

    */
    template <class T>
    class Greens
    {

    protected:

        // Indices
        int lmax;
        int N;

        // Autodiff stuff
        ADScalar<T, 2> b_g;
        ADScalar<T, 2> r_g;
        RowVector<ADScalar<T, 2>> s_g;

        // Internal methods
        template <typename T1, typename T2>
        inline void compute_ (
            const T1& b,
            const T1& r,
            T2& s
        );

    public:

        // Solutions
        RowVector<T> s;
        RowVector<T> dsdb;
        RowVector<T> dsdr;

        // Constructor
        explicit Greens(
            int lmax
        ) :
            lmax(lmax),
            N((lmax + 1) * (lmax + 1)),
            b_g(0.0, Vector<T>::Unit(2, 0)),
            r_g(0.0, Vector<T>::Unit(2, 1)),
            s_g(N),
            s(RowVector<T>::Zero(N)),
            dsdb(RowVector<T>::Zero(N)),
            dsdr(RowVector<T>::Zero(N))
        {

        }

        // Methods
        inline void compute (
            const T& b,
            const T& r,
            bool gradient=false
        );

    };

    /**
    Compute the `s^T` occultation solution vector;
    internal templated function for scalar/ADScalar
    capability.

    */
    template <class T>
    template <typename T1, typename T2>
    inline void Greens<T>::compute_ (
        const T1& b, 
        const T1& r,
        T2& s
    ) {

        // TODO
        // Dummy calculations for now!
        for (int n = 0; n < N; ++n) {
            s(n) = pow(b, n) + n * r;
        }

    }

    /**
    Compute the `s^T` occultation solution vector
    with or without the gradient.

    */
    template <class T>
    inline void Greens<T>::compute (
        const T& b, 
        const T& r,
        bool gradient
    ) {
        if (!gradient) {
            compute_(b, r, s);
        } else {
            b_g.value() = b;
            r_g.value() = r;
            compute_(b_g, r_g, s_g);
            for (int n = 0; n < N; ++n) {
                s(n) = s_g(n).value();
                dsdb(n) = s_g(n).derivatives()(0);
                dsdr(n) = s_g(n).derivatives()(1);
            }
        }
    }

} // namespace solver
} // namespace starry2

#endif