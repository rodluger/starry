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

        // Internal variables
        Scalar third;
        Scalar k;
        Scalar ksq;
        Scalar kc;
        Scalar kcsq;
        Scalar kkc;
        Scalar invksq;
        Scalar kap0;
        Scalar kap1;
        Scalar Eofk;
        Scalar Em1mKdm;

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
            third = Scalar(1.0) / Scalar(3.0);
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

        // Initialize some useful variables
        Scalar b2 = b * b;
        Scalar r2 = r * r;
        Scalar invr = Scalar(1.0) / r;
        Scalar invb = Scalar(1.0) / b;
        Scalar bmr = b - r;
        Scalar bpr = b + r;
        Scalar fourbr = 4 * b * r;
        Scalar invfourbr = 0.25 * invr * invb;
        Scalar onembmr2 = (1.0 + bmr) * (1.0 - bmr);
        Scalar onembmr2inv = Scalar(1.0) / onembmr2; 
        Scalar onembpr2 = (1.0 + bpr) * (1.0 - bpr);
        Scalar sqonembmr2 = sqrt(onembmr2);
        //Scalar b2mr22 = (b2 - r2) * (b2 - r2);
        //Scalar onemr2mb2 = (1.0 - r) * (1.0 + r) - b2;
        Scalar sqbr = sqrt(b * r); 

        // Compute the kite area and the k^2 variables
        Scalar p0 = 1.0, 
               p1 = b, 
               p2 = r;
        if (p0 < p1) swap(p0, p1);
        if (p1 < p2) swap(p1, p2);
        if (p0 < p1) swap(p0, p1);
        Scalar sqarea = (p0 + (p1 + p2)) * (p2 - (p0 - p1)) *
                        (p2 + (p0 - p1)) * (p0 + (p1 - p2));
        Scalar kite_area2 = sqrt(max(Scalar(0.0), sqarea));

        // Compute s0 and its derivatives
        if (unlikely((b == 0) || (r == 0))) {
            ksq = Scalar(INFINITY);
            k = Scalar(INFINITY);
            kc = 1;
            kcsq = 1;
            kkc = Scalar(INFINITY);
            invksq = 0;
            kap0 = 0; // Not used!
            kap1 = 0; // Not used!
            sT(0) = pi<Scalar>() * (1 - r2);
            if (GRADIENT) {
                dsTdb(0) = 0;
                dsTdr(0) = -2 * pi<Scalar>() * r;
            }
        } else {
            ksq = onembpr2 * invfourbr + 1.0;
            invksq = Scalar(1.0) / ksq;
            k = sqrt(ksq);
            if (ksq > 1) {
                kcsq = onembpr2 * onembmr2inv;
                kc = sqrt(kcsq);
                kkc = k * kc;
                kap0 = 0; // Not used!
                kap1 = 0; // Not used!
                sT(0) = pi<Scalar>() * (1 - r2);
                if (GRADIENT) {
                    dsTdb(0) = 0;
                    dsTdr(0) = -2 * pi<Scalar>() * r;
                }
            } else {
                kcsq = -onembpr2 * invfourbr;
                kc = sqrt(kcsq);
                kkc = kite_area2 * invfourbr;
                kap0 = atan2(kite_area2, (r - 1) * (r + 1) + b2);
                kap1 = atan2(kite_area2, (1 - r) * (1 + r) + b2);
                Scalar Alens = kap1 + r2 * kap0 - kite_area2 * 0.5;
                sT(0) = pi<Scalar>() - Alens;
                if (GRADIENT) {
                    dsTdb(0) = kite_area2 * invb;
                    dsTdr(0) = -2.0 * r * kap0;
                }
            }
        }

        // If the map is constant, stop here
        if (unlikely(N == 0)) return;

        // Compute s2 and its derivatives
        Scalar Lambda1 = 0;
        if ((b >= 1.0 + r) ||  (r == 0.0)) {
            // No occultation (Case 1)
            Lambda1 = 0;
            if (GRADIENT) {
                dsTdb(2) = 0;
                dsTdr(2) = 0;
            }
            Eofk = 0; // Check
            Em1mKdm = 0; // Check
        } else if (b <= r - 1.0) {
            // Full occultation (Case 11)
            Lambda1 = 0;
            if (GRADIENT) {
                dsTdb(2) = 0;
                dsTdr(2) = 0;
            }
            Eofk = 0; // Check
            Em1mKdm = 0; // Check
        } else {
            if (unlikely(b == 0)) {
                // Case 10
                Scalar sqrt1mr2 = sqrt(1.0 - r2);
                Lambda1 = -2.0 * pi<Scalar>() * sqrt1mr2 * sqrt1mr2 * sqrt1mr2; 
                Eofk = 0.5 * pi<Scalar>();
                Em1mKdm = 0.25 * pi<Scalar>();
                if (GRADIENT) {
                    dsTdb(2) = 0;
                    dsTdr(2) = -2.0 * pi<Scalar>() * r * sqrt1mr2;
                }
            } else if (unlikely(b == r)) {
                if (unlikely(r == 0.5)) {
                    // Case 6
                    Lambda1 = pi<Scalar>() - 4.0 * third;
                    Eofk = 1.0;
                    Em1mKdm = 1.0;
                    if (GRADIENT) {
                        dsTdb(2) = 2.0 * third;
                        dsTdr(2) = -2.0;
                    }
                } else if (r < 0.5) {
                    // Case 5
                    Scalar m = 4 * r2;
                    Eofk = ellip::CEL(m, Scalar(1.0), Scalar(1.0), 
                                      Scalar(1.0 - m));
                    Em1mKdm = ellip::CEL(m, Scalar(1.0), Scalar(1.0), 
                                         Scalar(0.0));
                    Lambda1 = pi<Scalar>() + 2.0 * third * 
                              ((2 * m - 3) * Eofk - m * Em1mKdm);
                    if (GRADIENT) {
                        dsTdb(2) = -4.0 * r * third * (Eofk - 2 * Em1mKdm);
                        dsTdr(2) = -4.0 * r * Eofk;
                    }
                } else {
                    // Case 7
                    Scalar m = 4 * r2; 
                    Scalar minv = Scalar(1.0) / m; 
                    Eofk = ellip::CEL(minv, Scalar(1.0), Scalar(1.0), 
                                      Scalar(1.0 - minv));
                    Em1mKdm = ellip::CEL(minv, Scalar(1.0), Scalar(1.0), 
                                         Scalar(0.0));
                    Lambda1 = pi<Scalar>() + third * invr * 
                              (-m * Eofk + (2 * m - 3) * Em1mKdm);
                    if (GRADIENT) {
                        dsTdb(2) = 2 * third * (2 * Eofk - Em1mKdm);
                        dsTdr(2) = -2 * Em1mKdm;
                    }
                }
            } else { 
                if (ksq < 1) {
                    // Case 2, Case 8
                    Scalar sqbrinv = Scalar(1.0) / sqbr;
                    Scalar Piofk;
                    ellip::CEL(ksq, kc, Scalar(bmr * bmr * kcsq), Scalar(0.0), 
                               Scalar(1.0), Scalar(1.0), 
                               Scalar(3 * kcsq * bmr * bpr), 
                               kcsq, Scalar(0.0), Piofk, Eofk, Em1mKdm);
                    Lambda1 = onembmr2 * (Piofk + (-3 + 6 * r2 + 2 * b * r) * 
                              Em1mKdm - fourbr * Eofk) * sqbrinv * third;
                    if (GRADIENT) {
                        dsTdb(2) = 2 * r * onembmr2 * (-Em1mKdm + 2 * Eofk) * 
                                sqbrinv * third;
                        dsTdr(2) = -2 * r * onembmr2 * Em1mKdm * sqbrinv;
                    }
                } else if (ksq > 1) {
                    // Case 3, Case 9
                    Scalar bmrdbpr = bmr / bpr; 
                    Scalar mu = 3 * bmrdbpr * onembmr2inv;
                    Scalar p = bmrdbpr * bmrdbpr * onembpr2 * onembmr2inv;
                    Scalar Piofk;
                    ellip::CEL(invksq, kc, p, Scalar(1 + mu), Scalar(1.0), 
                               Scalar(1.0), Scalar(p + mu), kcsq, Scalar(0.0), 
                               Piofk, Eofk, Em1mKdm);
                    Lambda1 = 2 * sqonembmr2 * 
                              (onembpr2 * Piofk - (4 - 7 * r2 - b2) * Eofk) * 
                              third;
                    if (GRADIENT) {
                        dsTdb(2) = -4 * r * third * sqonembmr2 * 
                                (Eofk - 2 * Em1mKdm);
                        dsTdr(2) = -4 * r * sqonembmr2 * Eofk;
                    }
                } else {
                    // Case 4
                    Scalar rootr1mr = sqrt(r * (1 - r));
                    Lambda1 = 2 * acos(1.0 - 2.0 * r) - 4 * third * 
                              (3 + 2 * r - 8 * r2) * 
                              rootr1mr - 2 * pi<Scalar>() * int(r > 0.5);
                    Eofk = 1.0;
                    Em1mKdm = 1.0;
                    if (GRADIENT) {
                        dsTdr(2) = -8 * r * rootr1mr;
                        dsTdb(2) = -dsTdr(1) * third;
                    }
                }
            }
        }
        sT(2) = ((1.0 - int(r > b)) * 2 * pi<Scalar>() - Lambda1) * third;

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