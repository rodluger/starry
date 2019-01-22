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

    // DEBUG DEBUG DEBUG
    double A(int i, int j, int k) {return 0.0;}
    double I(int i) {return 0.0;}
    double J(int i) {return 0.0;}
    double H(int i, int j) {return 0.0;}

    template <typename T>
    inline void computeKVariables (
        const T& b,
        const T& r,
        T& ksq,
        T& k,
        T& kc,
        T& kcsq,
        T& kkc,
        T& invksq,   
        T& kite_area2,     
        T& kap0,
        T& kap1
    ) {
        // Initialize some useful quantities
        T invr = T(1.0) / r;
        T invb = T(1.0) / b;
        T bpr = b + r;
        T invfourbr = 0.25 * invr * invb;
        T onembpr2 = (T(1.0) + bpr) * (T(1.0) - bpr); 

        // Compute the kite area and the k^2 variables
        if (unlikely((b == 0) || (r == 0))) {
            ksq = T(INFINITY);
            k = T(INFINITY);
            kc = 1;
            kcsq = 1;
            kkc = T(INFINITY);
            invksq = 0;
            kite_area2 = 0; // Not used!
            kap0 = 0; // Not used!
            kap1 = 0; // Not used!
        } else {
            ksq = onembpr2 * invfourbr + T(1.0);
            invksq = T(1.0) / ksq;
            k = sqrt(ksq);
            if (ksq > 1) {
                T bmr = b - r;
                T onembmr2 = (T(1.0) + bmr) * (T(1.0) - bmr);
                T onembmr2inv = T(1.0) / onembmr2; 
                kcsq = onembpr2 * onembmr2inv;
                kc = sqrt(kcsq);
                kkc = k * kc;
                kite_area2 = 0; // Not used!
                kap0 = 0; // Not used!
                kap1 = 0; // Not used!
            } else {
                T b2 = b * b;
                T p0 = T(1.0), 
                       p1 = b, 
                       p2 = r;
                if (p0 < p1) swap(p0, p1);
                if (p1 < p2) swap(p1, p2);
                if (p0 < p1) swap(p0, p1);
                T sqarea = (p0 + (p1 + p2)) * (p2 - (p0 - p1)) *
                                (p2 + (p0 - p1)) * (p0 + (p1 - p2));
                kite_area2 = sqrt(max(T(0.0), sqarea));
                kcsq = -onembpr2 * invfourbr;
                kc = sqrt(kcsq);
                kkc = kite_area2 * invfourbr;
                kap0 = atan2(kite_area2, (r - T(1.0)) * (r + T(1.0)) + b2);
                kap1 = atan2(kite_area2, (T(1.0) - r) * (T(1.0) + r) + b2);
            }
        }
    }

    template <class Scalar, bool GRADIENT=false>
    inline void computeS0_ (
        const Scalar& b,
        const Scalar& r,
        const Scalar& ksq, 
        const Scalar& kite_area2, 
        const Scalar& kap0, 
        const Scalar& kap1,  
        Scalar& s0,
        Scalar& ds0db,
        Scalar& ds0dr
    )  {
        if (unlikely((b == 0) || (r == 0))) {
            s0 = pi<Scalar>() * (1 - r * r);
            if (GRADIENT) {
                ds0db = 0;
                ds0dr = -2 * pi<Scalar>() * r;
            }
        } else {
            if (ksq > 1) {
                s0 = pi<Scalar>() * (1 - r * r);
                if (GRADIENT) {
                    ds0db = 0;
                    ds0dr = -2 * pi<Scalar>() * r;
                }
            } else {
                Scalar Alens = kap1 + r * r * kap0 - kite_area2 * 0.5;
                s0 = pi<Scalar>() - Alens;
                if (GRADIENT) {
                    ds0db = kite_area2 / b;
                    ds0dr = -2.0 * r * kap0;
                }
            }
        }
    }

    // TODO: Compute elliptic derivs
    template <class Scalar, bool GRADIENT=false>
    inline void computeS2_ (
        const Scalar& b,
        const Scalar& r,
        const Scalar& ksq,
        const Scalar& kc,
        const Scalar& kcsq,
        const Scalar& invksq,
        const Scalar& third,        
        Scalar& s2,
        Scalar& Eofk,
        Scalar& Em1mKdm,
        Scalar& ds2db,
        Scalar& ds2dr,
        Scalar& dEofkdb,
        Scalar& dEofkdr,
        Scalar& dEm1mKdmdb,
        Scalar& dEm1mKdmdr
    ) {
        // Initialize some useful quantities
        Scalar r2 = r * r;
        Scalar bmr = b - r;
        Scalar bpr = b + r;
        Scalar onembmr2 = (Scalar(1.0) + bmr) * (Scalar(1.0) - bmr);
        Scalar onembmr2inv = Scalar(1.0) / onembmr2; 
        
        // Compute s2 and its derivatives
        Scalar Lambda1 = 0;
        if ((b >= 1.0 + r) ||  (r == 0.0)) {
            // No occultation (Case 1)
            Lambda1 = 0;
            Eofk = 0;
            Em1mKdm = 0;
            if (GRADIENT) {
                ds2db = 0;
                ds2dr = 0;
            }
        } else if (b <= r - 1.0) {
            // Full occultation (Case 11)
            Lambda1 = 0;
            Eofk = 0;
            Em1mKdm = 0;
            if (GRADIENT) {
                ds2db = 0;
                ds2dr = 0;
            }
        } else {
            if (unlikely(b == 0)) {
                // Case 10
                Scalar sqrt1mr2 = sqrt(1.0 - r2);
                Lambda1 = -2.0 * pi<Scalar>() * sqrt1mr2 * sqrt1mr2 * sqrt1mr2; 
                Eofk = 0.5 * pi<Scalar>();
                Em1mKdm = 0.25 * pi<Scalar>();
                if (GRADIENT) {
                    ds2db = 0;
                    ds2dr = -2.0 * pi<Scalar>() * r * sqrt1mr2;
                }
            } else if (unlikely(b == r)) {
                if (unlikely(r == 0.5)) {
                    // Case 6
                    Lambda1 = pi<Scalar>() - 4.0 * third;
                    Eofk = 1.0;
                    Em1mKdm = 1.0;
                    if (GRADIENT) {
                        ds2db = 2.0 * third;
                        ds2dr = -2.0;
                    }
                } else if (r < 0.5) {
                    // Case 5
                    Scalar m = 4 * r2;
                    Eofk = ellip::CEL(m, Scalar(1.0), Scalar(1.0), Scalar(1.0 - m));
                    Em1mKdm = ellip::CEL(m, Scalar(1.0), Scalar(1.0), Scalar(0.0));
                    Lambda1 = pi<Scalar>() + 2.0 * third * 
                              ((2 * m - 3) * Eofk - m * Em1mKdm);
                    if (GRADIENT) {
                        ds2db = -4.0 * r * third * (Eofk - 2 * Em1mKdm);
                        ds2dr = -4.0 * r * Eofk;
                    }
                } else {
                    // Case 7
                    Scalar m = 4 * r2; 
                    Scalar minv = Scalar(1.0) / m; 
                    Eofk = ellip::CEL(minv, Scalar(1.0), Scalar(1.0), Scalar(1.0 - minv));
                    Em1mKdm = ellip::CEL(minv, Scalar(1.0), Scalar(1.0), Scalar(0.0));
                    Lambda1 = pi<Scalar>() + third / r * 
                              (-m * Eofk + (2 * m - 3) * Em1mKdm);
                    if (GRADIENT) {
                        ds2db = 2 * third * (2 * Eofk - Em1mKdm);
                        ds2dr = -2 * Em1mKdm;
                    }
                }
            } else { 
                if (ksq < 1) {
                    // Case 2, Case 8
                    Scalar fourbr = 4 * b * r;
                    Scalar sqbrinv = Scalar(1.0) / sqrt(b * r);
                    Scalar Piofk;
                    ellip::CEL(ksq, kc, Scalar((b - r) * (b - r) * kcsq), Scalar(0.0), 
                               Scalar(1.0), Scalar(1.0), Scalar(3 * kcsq * (b - r) * (b + r)), 
                               kcsq, Scalar(0.0), Piofk, Eofk, Em1mKdm);
                    Lambda1 = onembmr2 * (Piofk + (-3 + 6 * r2 + 2 * b * r) * 
                              Em1mKdm - fourbr * Eofk) * sqbrinv * third;
                    if (GRADIENT) {
                        ds2db = 2 * r * onembmr2 * (-Em1mKdm + 2 * Eofk) * 
                                sqbrinv * third;
                        ds2dr = -2 * r * onembmr2 * Em1mKdm * sqbrinv;
                    }
                } else if (ksq > 1) {
                    // Case 3, Case 9
                    Scalar onembpr2 = (Scalar(1.0) + bpr) * (Scalar(1.0) - bpr);
                    Scalar sqonembmr2 = sqrt(onembmr2);
                    Scalar b2 = b * b;
                    Scalar bmrdbpr = (b - r) / (b + r); 
                    Scalar mu = 3 * bmrdbpr * onembmr2inv;
                    Scalar p = bmrdbpr * bmrdbpr * onembpr2 * onembmr2inv;
                    Scalar Piofk;
                    ellip::CEL(invksq, kc, p, Scalar(1 + mu), Scalar(1.0), Scalar(1.0), 
                               Scalar(p + mu), kcsq, Scalar(0.0), Piofk, Eofk, Em1mKdm);
                    Lambda1 = 2 * sqonembmr2 * 
                              (onembpr2 * Piofk - (4 - 7 * r2 - b2) * Eofk) * 
                              third;
                    if (GRADIENT) {
                        ds2db = -4 * r * third * sqonembmr2 * 
                                (Eofk - 2 * Em1mKdm);
                        ds2dr = -4 * r * sqonembmr2 * Eofk;
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
                        ds2dr = -8 * r * rootr1mr;
                        ds2db = -ds2dr * third;
                    }
                }
            }
        }
        s2 = ((1.0 - int(r > b)) * 2 * pi<Scalar>() - Lambda1) * third;
    }

    template <class T, bool AUTODIFF>
    class Solver 
    {
    
    public:

        // Indices
        int lmax;
        int N;

        // Variables
        T b;
        T r;
        T k;
        T ksq;
        T kc;
        T kcsq;
        T kkc;
        T invksq;
        T kite_area2;
        T kap0;
        T kap1;
        T Eofk;
        T Em1mKdm;

        // Miscellaneous
        T third;
        T dummy;

        // The solution vector
        RowVector<T> sT;

        explicit Solver(
            int lmax
        ) :
            lmax(lmax),
            N((lmax + 1) * (lmax + 1)),
            sT(RowVector<T>::Zero(N))
        { 
            third = T(1.0) / T(3.0);
            dummy = 0.0;
        }

        /**
        The helper primitive integral K_{u,v}.
        TODO: This should be made into a dot product.

        */
        inline T K (
            int u, 
            int v
        ) {
            T res(0.0);
            for (int i = 0; i < u + v + 1; ++i)
                res += A(i, u, v) * I(i + u);
            return res;
        }

        /**
        The helper primitive integral L_{u,v}^(t).
        TODO: This should be made into a dot product.

        */
        inline T L (
            int u, 
            int v, 
            int t
        ) {
            T res(0.0);
            for (int i = 0; i < u + v + 1; ++i)
                res += A(i, u, v) * J(i + u + t);
            return res;
        }

        /**
        Compute s(0) for a Scalar type.

        */
        template <bool A=AUTODIFF>
        inline typename std::enable_if<!A, void>::type computeS0 () {
            computeS0_<T, false>(
                b, r, ksq, kite_area2, kap0, kap1, sT(0), dummy, dummy
            );
        }

        /**
        Compute s(0) and its gradient for an AutoDiffScalar type.
        We know how to compute the gradient analytically, so we need
        to override AutoDiff.

        */
        template <bool A=AUTODIFF>
        inline typename std::enable_if<A, void>::type computeS0 () {
            computeS0_<typename T::Scalar, true>(
                b.value(), r.value(), ksq.value(), kite_area2.value(), 
                kap0.value(), kap1.value(), sT(0).value(), 
                sT(0).derivatives()(0), sT(0).derivatives()(1)
            );
        }

        /**
        Compute s(2) for a Scalar type.

        */
        template <bool A=AUTODIFF>
        inline typename std::enable_if<!A, void>::type computeS2 () {
            computeS2_<T, false>(
                b, r, ksq, kc, kcsq, invksq, third, sT(2), Eofk, 
                Em1mKdm, dummy, dummy, dummy, dummy, dummy, dummy
            );
        }

        /**
        Compute s(2) and its gradient for an AutoDiffScalar type.
        We know how to compute the gradient analytically, so we need
        to override AutoDiff.
        
        */
        template <bool A=AUTODIFF>
        inline typename std::enable_if<A, void>::type computeS2 () {
            computeS2_<typename T::Scalar, true>(
                b.value(), r.value(), ksq.value(), kc.value(), 
                kcsq.value(), invksq.value(), third.value(), 
                sT(2).value(), Eofk.value(), Em1mKdm.value(), 
                sT(2).derivatives()(0), sT(2).derivatives()(1),
                Eofk.derivatives()(0), Eofk.derivatives()(1),
                Em1mKdm.derivatives()(0), Em1mKdm.derivatives()(1)
            );
        }

        /**
        Compute the `s^T` occultation solution vector.

        */
        inline void compute (
            const T& b_,
            const T& r_
        ) {

            // Some basic variables
            b = b_;
            r = r_;
            T twor = 2 * r;
            T tworlp2 = (2.0 * r) * (2.0 * r) * (2.0 * r);
            
            // Compute the family of k^2 variables
            computeKVariables(b, r, ksq, k, kc, kcsq, kkc, invksq, 
                              kite_area2, kap0, kap1);

            // Compute the constant term
            computeS0();

            // Break if lmax = 0
            if (unlikely(N == 0)) return;

            // The l = 1, m = -1 is zero by symmetry
            sT(1) = 0;
            
            // Compute the linear limb darkening term
            computeS2();

            // The l = 1, m = 1 term
            sT(3) = H(2, 1) - 2 * tworlp2 * K(1, 1); // TODO: Check this

            // Break if lmax = 1
            if (N == 4) return;

            // Some more basic variables
            T Q, P;
            T bmr = b - r;
            T lfac = pow(1 - bmr * bmr, 1.5);
            bool qcond = ((abs(T(1.0) - r) >= b) || (bmr >= T(1.0)));

            // Compute the othr terms of the solution vector
            int n = 4;
            for (int l = 2; l < lmax + 1; ++l) {
                
                // Update the pre-factors
                tworlp2 *= twor; 
                lfac *= twor;

                for (int m = -l; m < l + 1; ++m) {

                    int mu = l - m;
                    int nu = l + m;

                    // These terms are zero because they are proportional to
                    // odd powers of x, so we don't need to compute them!
                    if ((is_even(mu - 1)) && (!is_even((mu - 1) / 2))) {

                        sT(n) = 0;

                    // These terms are also zero for the same reason
                    } else if ((is_even(mu)) && (!is_even(mu / 2))) {

                        sT(n) = 0;

                    // We need to compute the integral...
                    } else {
                        
                        // The Q integral
                        if ((qcond) && (!is_even(mu, 2) || !is_even(nu, 2)))
                            Q = 0.0;
                        else if (!is_even(mu, 2))
                            Q = 0.0;
                        else
                            Q = H((mu + 4) / 2, nu / 2);

                        // The P integral
                        if (is_even(mu, 2))
                            P = 2 * tworlp2 * K((mu + 4) / 4, nu / 2);
                        else if ((mu == 1) && is_even(l))
                            P = lfac * (L((l - 2) / 2, 0, 0) - 
                                        2 * L((l - 2) / 2, 0, 1));
                        else if ((mu == 1) && !is_even(l))
                            P = lfac * (L((l - 3) / 2, 1, 0) - 
                                        2 * L((l - 3) / 2, 1, 1));
                        else if (is_even(mu - 1, 2))
                            P = 2 * lfac * L((mu - 1) / 4, (nu - 1) / 2, 0);
                        else
                            P = 0.0;

                        // The term of the solution vector
                        sT(n) = Q - P;

                    }

                    ++n;
                    
                }

            }

        }

    };

    /**
    Greens integral solver wrapper class.

    */
    template <class Scalar>
    class Greens
    {

    protected:

        using ADType = ADScalar<Scalar, 2>;

        // Indices
        int lmax;
        int N;

        // All variables
        Solver<Scalar, false> ScalarSolver;
        Solver<ADType, true> ADTypeSolver;

    public:

        // Solutions
        RowVector<Scalar>& sT;
        RowVector<Scalar> dsTdb;
        RowVector<Scalar> dsTdr;

        // Constructor
        explicit Greens(
            int lmax
        ) :
            lmax(lmax),
            N((lmax + 1) * (lmax + 1)),
            ScalarSolver(lmax),
            ADTypeSolver(lmax),
            sT(ScalarSolver.sT),
            dsTdb(RowVector<Scalar>::Zero(N)),
            dsTdr(RowVector<Scalar>::Zero(N))
        {
        }

        /**
        Compute the `s^T` occultation solution vector
        with or without the gradient.

        */
        template <bool GRADIENT=false>
        inline void compute (
            const Scalar& b,
            const Scalar& r
        ) {

            if (!GRADIENT) {

                ScalarSolver.compute(b, r);

            } else {

                ADTypeSolver.compute(b, r);
                for (int n = 0; n < N; ++n) {
                    sT(n) = ADTypeSolver.sT(n).value();
                    dsTdb(n) = ADTypeSolver.sT(n).derivatives()(0);
                    dsTdr(n) = ADTypeSolver.sT(n).derivatives()(1);
                }

            }

        }

    };

} // namespace solver
} // namespace starry2

#endif