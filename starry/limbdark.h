/**
Limb darkening utilities from Agol & Luger (2018).

*/

#ifndef _STARRY_LIMBDARK_H_
#define _STARRY_LIMBDARK_H_

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include "ellip.h"
#include "errors.h"
#include "solver.h"
#include "lld.h"

namespace limbdark {

    using std::abs;
    using std::swap;
    using namespace utils;


    /**
    Greens integration housekeeping data

    */
    template <class T>
    class Greens {

        public:

            // Indices
            int lmax;

            // Basic variables
            T b;
            T r;
            T k;
            T kc;
            T kkc;
            T kap0;
            T invksq;

            // Powers of basic variables
            solver::Power<T> ksq;
            solver::Power<T> fourbr;
            solver::Power<T> two;

            // Primitive matrices/vectors
            solver::Elliptic<T> ELL;
            solver::I<T> I_P;
            solver::J<T> J_P;

            // Coefficient vectors
            Vector<T> a;
            Vector<T> c;

            // The solution vector
            VectorT<T> S;

            // Constructor
            explicit Greens(int lmax) :
                   lmax(lmax),
                   ksq(T(0.0)),
                   fourbr(T(0.0)),
                   two(T(2.0)),
                   ELL((*this).ksq, (*this).invksq),
                   I_P(lmax, (*this).ksq, (*this).k, (*this).kc, (*this).kkc,
                       (*this).kap0),
                   J_P(lmax, (*this).ELL, (*this).ksq, (*this).two, (*this).k,
                       (*this).kc, (*this).invksq),
                   a(Vector<T>::Zero(lmax + 1)),
                   c(Vector<T>::Zero(lmax + 1)),
                   S(VectorT<T>::Zero(lmax + 1)) { }

            inline void computeS(const T& b_, const T& r_);
            inline void computeC(const Vector<T>& u);
            inline T computeFlux(const T& b_, const T& r_, const Vector<T>& u);

    };

    /**
    Transform the u_n coefficients to c_n, which are coefficients
    of the basis in which the P(G_n) functions are computed.

    */
    template <class T>
    inline void Greens<T>::computeC(const Vector<T>& u) {
        T bcoeff;
        size_t N = u.size();

        // Compute the a_n coefficients
        a.setZero();
        a(0) = 1.0;
        for (int i = 1; i < N; ++i) {
            bcoeff = 1.0;
            int sgn = 1;
            for (int j = 0; j <= i; ++j) {
                a(j) -= u(i) * bcoeff * sgn;
                sgn *= -1;
                bcoeff *= (T(i - j) / (j + 1));
            }
        }

        // Now, compute the c_n coefficients
        for (int j = N - 1; j >= 2; --j) {
            if (j >= N - 2)
                c(j) = a(j) / (j + 2);
            else
                c(j) = a(j) / (j + 2) + c(j + 2);
        }
        if (N >= 4)
            c(1) = a(1) + 3 * c(3);
        else
            c(1) = a(1);
        if (N >= 3)
            c(0) = a(0) + 2 * c(2);
        else
            c(0) = a(0);
    }

    /**
    Compute the `s^T` occultation solution vector

    */
    template <class T>
    inline void Greens<T>::computeS(const T& b_, const T& r_) {

        // Initialize the basic variables
        S.setZero();
        b = b_;
        r = r_;
        T ksq_;
        if ((b == 0) || (r == 0)) {
            ksq_ = T(INFINITY);
            k = T(INFINITY);
            kc = 1;
            kkc = T(INFINITY);
            invksq = 0;
            kap0 = 0;
            S(0) = pi<T>() * (1 - r * r);
        } else {
            ksq_ = (1 - (b - r)) * (1 + (b - r)) / (4 * b * r);
            invksq = (4 * b * r) / ((1 - (b - r)) * (1 + (b - r)));
            k = sqrt(ksq_);
            if (ksq_ > 1) {
                kc = sqrt(1 - invksq);
                kkc = k * kc;
                kap0 = 0; // Not used!
                S(0) = pi<T>() * (1 - r * r);
            } else {
                kc = sqrt(abs(((b + r) * (b + r) - 1) / (4 * b * r)));
                // Eric Agol's "kite" method to compute a stable
                // version of k * kc and I_0 = kap0
                // Used to be
                //   G.kkc = G.k * G.kc;
                //   G.kap0 = 2 * acos(G.kc);
                T p0 = T(1.0), p1 = b, p2 = r;
                if (p0 < p1) swap(p0, p1);
                if (p1 < p2) swap(p1, p2);
                if (p0 < p1) swap(p0, p1);
                T kite_area2 = sqrt((p0 + (p1 + p2)) * (p2 - (p0 - p1)) *
                                    (p2 + (p0 - p1)) * (p0 + (p1 - p2)));
                kkc = kite_area2 / (4 * b * r);
                kap0 = atan2(kite_area2, (r - 1) * (r + 1) + b * b);
                T kap1 = atan2(kite_area2, (1 - r) * (1 + r) + b * b);
                T Alens = kap1 + r * r * kap0 - kite_area2 * 0.5;
                S(0) = pi<T>() - Alens;

            }
        }
        ksq.reset(ksq_);
        fourbr.reset(4 * b * r);
        I_P.reset(ksq_ < 0.5);
        J_P.reset((ksq_ < 0.5) || (ksq_ > 2));

        // Linear limb darkening term
        S(1) = lld::s2(b, r, ksq(1), ELL.K(), ELL.E());

        // Even higher order terms
        int sgn, n0;
        for (int n = 2; n < lmax + 1; n += 2) {
            n0 = n / 2;
            if (is_even(n0))
                sgn = 1;
            else
                sgn = -1;
            for (int i = 0; i <= n0; ++i) {
                S(n) += sgn * tables::choose<T>(n0, i) * ksq(i) *
                        ((r - b) * I_P(n0 - i) + 2 * b * I_P(n0 - i + 1));
                sgn *= -1;
            }
            S(n) *= -2 * r * fourbr(n0);
        }

        // Odd higher order terms
        for (int n = 3; n < lmax + 1; n += 2) {
            n0 = (n - 3) / 2;
            if (is_even(n0))
                sgn = 1;
            else
                sgn = -1;
            for (int i = 0; i <= n0; ++i) {
                S(n) += sgn * tables::choose<T>(n0, i) * ksq(i) *
                        ((r - b) * J_P(n0 - i) + 2 * b * J_P(n0 - i + 1));
                sgn *= -1;
            }
            S(n) *= -2 * r * pow(1 - (b - r) * (b - r), 1.5) * fourbr(n0);
        }

    }

    /**
    Compute the flux in occultation.

    */
    template <class T>
    inline T Greens<T>::computeFlux(const T& b_, const T& r_, const Vector<T>& u) {
        // Really simple!
        computeC(u);
        computeS(b_, r_);
        return S.dot(c) / (pi<T>() * (c(0) + 2.0 * c(1) / 3.0));
    }

} // namespace limbdark

#endif
