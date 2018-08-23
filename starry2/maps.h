/**
Defines the surface map class.

TODO: Could speed up limb-darkened map rotations, since
      the effective degree of the map is lower. Won't
      work when computing gradients.

TODO: (In progress) Binding temporary references in each function
      to make code faster and easier to read.

TODO: Move linalg stuff from utils.h to linalg.h

*/

#ifndef _STARRY_MAPS_H_
#define _STARRY_MAPS_H_

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <type_traits>
#include <vector>
#include "rotation.h"
#include "basis.h"
#include "errors.h"
#include "utils.h"
#include "solver.h"

#define CACHE_NONE 0
#define CACHE_FLUX 1
#define CACHE_EVAL 2

namespace maps {

    using namespace utils;
    using std::abs;
    using std::max;
    using std::string;
    using std::to_string;
    using rotation::Wigner;
    using basis::Basis;
    using basis::polymul;
    using solver::Greens;
    using solver::Power;

    // Forward-declare some stuff
    template <class T> class Map;
    template <class T> std::string info(const Map<T>& map);
    template <> std::string info(const Map<Vector<Multi>>& map);
    template <> std::string info(const Map<Matrix<double>>& map);

    template <class T>
    class Temporary {

            static constexpr int TLEN = 4;
            static constexpr int RLEN = 4;
            static constexpr int CLEN = 1;
            static constexpr int VLEN = 1;
            static constexpr int VTLEN = 5;
            static constexpr int MLEN = 2;
            static constexpr int VTMLEN = 1;
            static constexpr int ALEN = 2;
            static constexpr int VTALEN = 1;
            static constexpr int PLEN = 2;
            static constexpr int PALEN = 2;

        public:

            T tmpT[TLEN];
            Row<T> tmpRow[RLEN];
            Column<T> tmpColumn[CLEN];
            Vector<Scalar<T>> tmpColumnVector[VLEN];
            VectorT<Scalar<T>> tmpRowVector[VTLEN];
            Matrix<Scalar<T>> tmpMatrix[MLEN];
            VectorT<Matrix<Scalar<T>>> tmpRowVectorOfMatrices[VTMLEN];
            ADScalar<Scalar<T>, 2> tmpADScalar2[ALEN];
            VectorT<ADScalar<Scalar<T>, 2>> tmpRowVectorOfADScalar2[VTALEN];
            Power<Scalar<T>> tmpPower[PLEN];
            Power<ADScalar<Scalar<T>, 2>> tmpPowerOfADScalar2[PALEN];

            explicit Temporary(int N, int nwav) {
                // Allocate memory for the Map-like variables
                // NOTE: All other containers need to be allocated elsewhere
                for (int n = 0; n < TLEN; ++n) resize(tmpT[n], N, nwav);
                for (int n = 0; n < RLEN; ++n) resize(tmpRow[n], N, nwav);
                for (int n = 0; n < CLEN; ++n) resize(tmpColumn[n], N, nwav);
            }

    };

    /**
    The main surface map class.

    */
    template <class T>
    class Map {

        public:

            const int lmax;                                                     /**< The highest degree of the map */
            const int N;                                                        /**< The number of map coefficients */
            const int nwav;                                                     /**< The number of wavelengths */
            T dI;                                                               /**< Gradient of the intensity */
            std::vector<string> dI_names;                                       /**< Names of each of the params in the intensity gradient */
            T dF;                                                               /**< Gradient of the flux */
            std::vector<string> dF_names;                                       /**< Names of each of the params in the flux gradient */

        private:

            T y;                                                                /**< The map coefficients in the spherical harmonic basis */
            T p;                                                                /**< The map coefficients in the polynomial basis */
            T g;                                                                /**< The map coefficients in the Green's basis */
            T u;                                                                /**< The limb darkening coefficients */
            T p_u;                                                              /**< The limb darkening coefficients in the polynomial basis */
            T g_u;                                                              /**< The limb darkening coefficients in the Green's basis */
            int y_deg;                                                          /**< Highest degree set by the user in the spherical harmonic vector */
            int u_deg;                                                          /**< Highest degree set by the user in the limb darkening vector */
            Vector<Matrix<Scalar<T>>> dp_udu;                                   /**< Deriv of limb darkening polynomial w/ respect to limb darkening coeffs */
            UnitVector<Scalar<T>> axis;                                         /**< The axis of rotation for the map */
            Basis<Scalar<T>> B;                                                 /**< Basis transform stuff */
            Wigner<T> W;                                                        /**< The class controlling rotations */
            Greens<Scalar<T>> G;                                                /**< The occultation integral solver class */
            Greens<ADScalar<Scalar<T>, 2>> G_grad;                              /**< The occultation integral solver class w/ AutoDiff capability */
            Scalar<T> tol;                                                      /**< Machine epsilon */

            // Temporaries and cache
            Temporary<T> tmp;
            int cache_oper;                                                     /**< Cached operation identifier */
            Scalar<T> cache_theta;                                              /**< Cached rotation angle */
            T cache_p;                                                          /**< Cached polynomial map */
            T cache_y;                                                          /**< Cached Ylm map */

            // Remove this stuff
            Vector<Scalar<T>> vtmp;                                             /**< A temporary surface map vector */
            VectorT<Scalar<T>> pT;                                              /**< The polynomial basis vector */
            VectorT<Scalar<T>> pTA1;                                            /**< Polynomial basis dotted into change of basis matrix */
            ADScalar<Scalar<T>, 2> x0_grad;                                     /**< x position AD type for map evaluation */
            ADScalar<Scalar<T>, 2> y0_grad;                                     /**< y position AD type for map evaluation */
            VectorT<ADScalar<Scalar<T>, 2>> pT_grad;                            /**< Polynomial basis AD type */
            ADScalar<Scalar<T>, 2> b_grad;                                      /**< Occultor impact parameter AD type for flux evaluation */
            ADScalar<Scalar<T>, 2> ro_grad;                                     /**< Occultor radius AD type for flux evaluation */
            VectorT<ADScalar<Scalar<T>, 2>> sT_grad;                            /**< Occultation solution vector AD type */
            VectorT<Scalar<T>> sTA;                                             /**< The solution vector in the sph harm basis */
            VectorT<Scalar<T>> sTAR;                                            /**< The solution vector in the rotated sph harm basis */
            VectorT<Scalar<T>> sTAdRdtheta;                                     /**< The derivative of `sTAR` with respect to `theta` */
            T mtmp;                                                             /**< A temporary surface map vector */
            T mtmp2;                                                            /**< A temporary surface map vector */
            T p_uy;                                                             /**< The instantaneous limb-darkened map in the polynomial basis */
            T g_uy;                                                             /**< The instantaneous limb-darkened map in the Green's basis */
            T Ry;                                                               /**< The rotated spherical harmonic vector */
            T dRdthetay;                                                        /**< Derivative of `Ry` with respect to `theta` */
            T* ptr_A1Ry;                                                        /**< Pointer to rotated polynomial vector */
            T* ptr_Ry;                                                          /**< Pointer to rotated spherical harmonic vector */
            T* ptr_RRy;                                                         /**< Pointer to rotated spherical harmonic vector */
            T ARRy;                                                             /**< The `ARRy` term in `s^TARRy` */
            Row<T> dFdb;                                                        /**< Gradient of the flux with respect to the impact parameter */
            Row<T> rtmp;                                                        /**< A temporary surface map row vector */
            Row<T> rtmp2;                                                       /**< A temporary surface map row vector */
            Row<T> rtmp3;                                                       /**< A temporary surface map row vector */
            Column<T> ctmp;                                                     /**< A temporary surface map col vector */
            Power<Scalar<T>> xpow_scalar;                                       /**< Powers of x for map evaluation */
            Power<Scalar<T>> ypow_scalar;                                       /**< Powers of y for map evaluation */
            Power<ADScalar<Scalar<T>, 2>> xpow_grad;                            /**< Powers of x for gradient map evaluation */
            Power<ADScalar<Scalar<T>, 2>> ypow_grad;                            /**< Powers of y for gradient map evaluation */
            VectorT<Matrix<Scalar<T>>> grad_p1;                                 /**< Derivative of a polynomial product (gross) */
            VectorT<Matrix<Scalar<T>>> grad_p2;                                 /**< Derivative of a polynomial product (gross) */
            Matrix<Scalar<T>> mattmp;                                           /**< Temporary matrix */
            Matrix<Scalar<T>> mattmp2;                                          /**< Temporary matrix */

            // External info function
            template <class U>
            friend std::string info(const Map<U>& map);

            // Private methods
            inline void check_degree();
            inline void clear_cache();
            inline void update_y();
            inline void update_u();
            inline void limb_darken(const T& poly, T& poly_ld, bool gradient=false);
            template <typename U>
            inline void poly_basis(Power<U>& xpow, Power<U>& ypow, VectorT<U>& basis);
            inline Row<T> evaluate_with_gradient(const Scalar<T>& theta_deg,
                                                 const Scalar<T>& x_,
                                                 const Scalar<T>& y_);
            inline Row<T> flux_with_gradient(const Scalar<T>& theta_deg,
                                             const Scalar<T>& xo_,
                                             const Scalar<T>& yo_,
                                             const Scalar<T>& ro_);
            inline Row<T> flux_ld(const Scalar<T>& xo_,
                                  const Scalar<T>& yo_,
                                  const Scalar<T>& ro_);
            inline Row<T> flux_ld_with_gradient(const Scalar<T>& xo_,
                                                const Scalar<T>& yo_,
                                                const Scalar<T>& ro_);

        public:

            /**
            Instantiate a `Map`.

            */
            explicit Map(int lmax=2, int nwav=1) :
                lmax(lmax),
                N((lmax + 1) * (lmax + 1)),
                nwav(nwav),
                dI_names({"theta", "x", "y"}),
                dF_names({"theta", "xo", "yo", "ro"}),
                dp_udu(nwav),
                B(lmax),
                W(lmax, nwav, (*this).y, (*this).axis),
                G(lmax),
                G_grad(lmax),
                tol(mach_eps<Scalar<T>>()),
                tmp(N, nwav) {

                // Populate the map gradient names
                for (int l = 0; l < lmax + 1; l++) {
                    for (int m = -l; m < l + 1; m++) {
                        dI_names.push_back(string("Y_{" + to_string(l) +
                                           "," + to_string(m) + "}"));
                        dF_names.push_back(string("Y_{" + to_string(l) +
                                           "," + to_string(m) + "}"));
                    }
                }
                for (int l = 1; l < lmax + 1; l++) {
                    dI_names.push_back(string("u_{" + to_string(l) + "}"));
                    dF_names.push_back(string("u_{" + to_string(l) + "}"));
                }

                // Initialize the map vectors
                axis = yhat<Scalar<T>>();
                resize(y, N, nwav);
                resize(p, N, nwav);
                resize(g, N, nwav);
                resize(u, lmax + 1, nwav);
                resize(p_u, N, nwav);
                resize(g_u, N, nwav);
                resize(dI, 3 + N + lmax, nwav);
                resize(dF, 4 + N + lmax, nwav);

                // TODO: Get rid of this stuff
                vtmp = Vector<Scalar<T>>::Zero(N);
                pT = VectorT<Scalar<T>>::Zero(N);
                pTA1 = VectorT<Scalar<T>>::Zero(N);
                x0_grad = ADScalar<Scalar<T>, 2>(0, Vector<Scalar<T>>::Unit(2, 0));
                y0_grad = ADScalar<Scalar<T>, 2>(0, Vector<Scalar<T>>::Unit(2, 1));
                pT_grad = VectorT<ADScalar<Scalar<T>, 2>>::Zero(N);
                b_grad = ADScalar<Scalar<T>, 2>(0, Vector<Scalar<T>>::Unit(2, 0));
                ro_grad = ADScalar<Scalar<T>, 2>(0, Vector<Scalar<T>>::Unit(2, 1));
                sT_grad = VectorT<ADScalar<Scalar<T>, 2>>::Zero(N);
                sTA = VectorT<Scalar<T>>::Zero(N);
                sTAR = VectorT<Scalar<T>>::Zero(N);
                sTAdRdtheta = VectorT<Scalar<T>>::Zero(N);
                resize(mtmp, N, nwav);
                resize(mtmp2, N, nwav);
                resize(Ry, N, nwav);
                resize(dRdthetay, N, nwav);
                resize(ARRy, N, nwav);
                resize(dFdb, N, nwav);
                resize(rtmp, N, nwav);
                resize(rtmp2, N, nwav);
                resize(rtmp3, N, nwav);
                resize(ctmp, N, nwav);

                // Reset & update the map coeffs
                reset();

            }

            // Housekeeping functions
            void update();
            void reset();

            // I/O functions
            void setY(const T& y_);
            void setY(int l, int m, const Row<T>& coeff);
            T getY() const;
            Row<T> getY(int l, int m) const;
            T getU() const;
            Row<T> getU(int l) const;
            void setU(const T& u_);
            void setU(int l, const Row<T>& coeff);
            T getP() const;
            T getG() const;
            VectorT<Scalar<T>> getR() const;
            VectorT<Scalar<T>> getS() const;
            void setAxis(const UnitVector<Scalar<T>>& axis_);
            UnitVector<Scalar<T>> getAxis() const;
            std::string __repr__();

            // Rotate the base map
            void rotate(const Scalar<T>&  theta_);

            // Evaluate the intensity at a point
            inline Row<T> evaluate(const Scalar<T>& theta_=0,
                                   const Scalar<T>& x_=0,
                                   const Scalar<T>& y_=0,
                                   bool gradient=false);

            // Compute the flux
            inline Row<T> flux(const Scalar<T>& theta_=0,
                               const Scalar<T>& xo_=0,
                               const Scalar<T>& yo_=0,
                               const Scalar<T>& ro_=0,
                               bool gradient=false);

    };

    /* ---------------- */
    /*   HOUSEKEEPING   */
    /* ---------------- */

    /**
    Check if the total degree of the map is valid

    */
    template <class T>
    inline void Map<T>::check_degree() {
        if (y_deg + u_deg > lmax) {
            setZero(u);
            setRow(u, 0, Scalar<T>(-1.0));
            u_deg = 0;
            throw errors::ValueError("Degree of the limb-darkened "
                                     "map exceeds `lmax`. Limb darkening "
                                     "coefficients have been reset.");
        }
    }

    /**
    Clear the cache

    */
    template <class T>
    inline void Map<T>::clear_cache() {
        cache_oper = CACHE_NONE;
        cache_theta = NAN;
    }

    /**
    Update the Ylm map after the coefficients changed
    TODO: This method needs to be made as fast as possible.

    */
    template <class T>
    inline void Map<T>::update_y() {
        // Check the map degree is valid
        check_degree();

        // Update the polynomial and Green's map coefficients
        p = B.A1 * y;
        g = B.A * y;

        // Update the rotation matrix
        W.update();

        // Clear the cache
        clear_cache();
    }

    /**
    Update the limb darkening map after the coefficients changed
    TODO: This method needs to be made as fast as possible.

    */
    template <class T>
    inline void Map<T>::update_u() {
        // Bind references to temporaries for speed
        Row<T>& Y00(tmp.tmpRow[0]);
        Row<T>& rTp_u(tmp.tmpRow[1]);
        Row<T>& ld_norm(tmp.tmpRow[2]);

        // Check the map degree is valid
        check_degree();

        // Update the limb darkening polynomial map
        p_u = B.U1 * u;

        // Compute the normalization that preserves
        // the disk-integrated intensity
        Y00 = getRow(y, 0);
        rTp_u = dot(B.rT, p_u);
        ld_norm = cwiseQuotient(Y00, rTp_u);

        // Compute the derivative of the LD polynomial
        // with respect to the LD coefficients
        for (int n = 0; n < nwav; ++n) {
            dp_udu(n) = -getColumn(p_u, n) * B.rTU1;
            dp_udu(n) /= dot(B.rT, getColumn(p_u, n));
            dp_udu(n) += B.U1;
            dp_udu(n) *= getColumn(ld_norm, n);
        }

        // Apply the normalization
        p_u = colwiseProduct(p_u, ld_norm);

        // Update the limb darkening Green's map
        g_u = B.A2 * p_u;

        // Clear the cache
        clear_cache();
    }


    /**
    Update the two maps after the coefficients changed

    */
    template <class T>
    inline void Map<T>::update() {
        update_y();
        update_u();
    }

    /**
    Reset the map

    */
    template <class T>
    void Map<T>::reset() {
        setZero(y);
        setRow(y, 0, Scalar<T>(1.0));
        y_deg = 0;
        setZero(u);
        u_deg = 0;
        setRow(u, 0, Scalar<T>(-1.0));
        axis = yhat<Scalar<T>>();
        update();
    }


    /* ---------------- */
    /*        I/O       */
    /* ---------------- */

    /**
    Set the spherical harmonic vector

    */
    template <class T>
    void Map<T>::setY(const T& y_) {
        if ((y_.rows() == y.rows()) && (y_.cols() == y.cols())) {
            y = y_;
            y_deg = 0;
            for (int l = lmax; l >= 0; --l) {
                if ((y.block(l * l, 0, 2 * l + 1, nwav).array() != 0.0).any()) {
                    y_deg = l;
                    break;
                }
            }
            // Note that we must implicitly call `update_u()` as well
            // because its normalization depends on Y_{0,0}!
            update();
        } else {
            throw errors::ValueError("Dimension mismatch in `y`.");
        }
    }

    /**
    Set the (l, m) spherical harmonic coefficient

    */
    template <class T>
    void Map<T>::setY(int l, int m, const Row<T>& coeff) {
        if ((0 <= l) && (l <= lmax) && (-l <= m) && (m <= l)) {
            int n = l * l + l + m;
            setRow(y, n, coeff);
            if (allZero(coeff)) {
                // If coeff is zero, we need to re-compute y_deg
                for (y_deg = l - 1; y_deg >= 0; --y_deg) {
                    for (int m = -y_deg; m < y_deg + 1; ++m){
                        if (!allZero(getRow(y, y_deg * y_deg + y_deg + m))) {
                            update();
                            return;
                        }
                    }
                }
            } else {
                y_deg = max(y_deg, l);
            }
            // Note that we must implicitly call `update_u()` as well
            // because its normalization depends on Y_{0,0}!
            update();
        } else {
            throw errors::IndexError("Invalid value for `l` and/or `m`.");
        }
    }

    /**
    Get the spherical harmonic vector

    */
    template <class T>
    T Map<T>::getY() const {
        return y;
    }

    /**
    Get the (l, m) spherical harmonic coefficient

    */
    template <class T>
    Row<T> Map<T>::getY(int l, int m) const {
        if ((0 <= l) && (l <= lmax) && (-l <= m) && (m <= l))
            return getRow(y, l * l + l + m);
        else
            throw errors::IndexError("Invalid value for `l` and/or `m`.");
    }

    /**
    Set the limb darkening vector

    */
    template <class T>
    void Map<T>::setU(const T& u_) {
        if ((u_.rows() == u.rows() - 1) && (u_.cols() == u.cols())) {
            u.block(1, 0, lmax, nwav) = u_;
            u_deg = 0;
            for (int l = lmax; l > 0; --l) {
                if (!allZero(getRow(u, l))) {
                    u_deg = l;
                    break;
                }
            }
            update_u();
        } else {
            throw errors::ValueError("Dimension mismatch in `u`.");
        }
    }

    /**
    Set the `l`th limb darkening coefficient

    */
    template <class T>
    void Map<T>::setU(int l, const Row<T>& coeff) {
        if ((1 <= l) && (l <= lmax)) {
            setRow(u, l, coeff);
            if (allZero(coeff)) {
                // If coeff is zero, we need to re-compute u_deg
                for (u_deg = l - 1; u_deg >= 0; --u_deg) {
                    if (!allZero(getRow(u, u_deg))) {
                        break;
                    }
                }
            } else {
                u_deg = max(u_deg, l);
            }
            update_u();
        } else {
            throw errors::IndexError("Invalid value for `l`.");
        }
    }

    /**
    Get the limb darkening vector

    */
    template <class T>
    T Map<T>::getU() const {
        return u.block(1, 0, lmax, nwav);
    }

    /**
    Get the `l`th limb darkening coefficient

    */
    template <class T>
    Row<T> Map<T>::getU(int l) const {
        if ((1 <= l) && (l <= lmax))
            return getRow(u,l);
        else
            throw errors::IndexError("Invalid value for `l`.");
    }

    /**
    Get the polynomial vector

    */
    template <class T>
    T Map<T>::getP() const {
        return p;
    }

    /**
    Get the Green's vector

    */
    template <class T>
    T Map<T>::getG() const {
        return g;
    }

    /**
    Get the rotation solution vector

    */
    template <class T>
    VectorT<Scalar<T>> Map<T>::getR() const {
        return B.rT;
    }

    /**
    Get the occultation solution vector

    */
    template <class T>
    VectorT<Scalar<T>> Map<T>::getS() const {
        return G.sT;
    }

    /**
    Set the axis

    */
    template <class T>
    void Map<T>::setAxis(const UnitVector<Scalar<T>>& axis_) {

        // Set it and normalize it
        axis(0) = axis_(0);
        axis(1) = axis_(1);
        axis(2) = axis_(2);
        axis = axis / sqrt(axis(0) * axis(0) +
                           axis(1) * axis(1) +
                           axis(2) * axis(2));

        // Update the rotation matrix
        W.update();

        // Reset the cache
        cache_theta = NAN;

    }

    /**
    Return a copy of the axis

    */
    template <class T>
    UnitVector<Scalar<T>> Map<T>::getAxis() const {
        return axis;
    }

    /**
    Return a human-readable map string

    */
    template <class T>
    std::string Map<T>::__repr__() {
        return info(*this);
    }


    /* -------------- */
    /*   OPERATIONS   */
    /* -------------- */

    /**
    Rotate the base map in-place given `theta` in **degrees**

    */
    template <class T>
    void Map<T>::rotate(const Scalar<T>& theta_) {
        Scalar<T> theta = theta_ * (pi<Scalar<T>>() / 180.);
        W.rotate(cos(theta), sin(theta), y);
        update();
    }


    /**
    Limb-darken a polynomial map, and optionally compute the
    gradient of the resulting map with respect to the input
    polynomial map and the input limb-darkening map.

    */
    template <class T>
    inline void Map<T>::limb_darken(const T& poly, T& poly_ld, bool gradient) {
        // Bind references to temporaries for speed
        Row<T>& rTp(tmp.tmpRow[0]);
        Row<T>& rTp_ld(tmp.tmpRow[1]);
        Row<T>& norm(tmp.tmpRow[2]);
        Matrix<Scalar<T>>& outer_inner(tmp.tmpMatrix[0]);
        Matrix<Scalar<T>>& grad_norm(tmp.tmpMatrix[1]);

        // Multiply a polynomial map by the LD polynomial
        if (gradient) {
            polymul(y_deg, poly, u_deg, p_u, lmax, poly_ld, grad_p1, grad_p2);
        } else
            polymul(y_deg, poly, u_deg, p_u, lmax, poly_ld);

        // Compute the normalization by enforcing that limb darkening does not
        // change the total disk-integrated flux.
        rTp = dot(B.rT, poly);
        if (hasZero(rTp))
            throw errors::ValueError("The visible map has zero net flux "
                                     "and cannot be limb-darkened.");
        rTp_ld = dot(B.rT, poly_ld);
        norm = cwiseQuotient(rTp, rTp_ld);

        // We need to do a little chain rule to propagate
        // the normalization to the gradient. There may be
        // a faster way that avoids the for loop...
        if (gradient) {
            for (int n = 0; n < nwav; ++n) {
                outer_inner = getColumn(poly_ld, n) * B.rT;
                outer_inner /= getColumn(rTp_ld, n);
                grad_norm = grad_p1(n) * getColumn(norm, n);
                grad_p1(n) = grad_norm + outer_inner - outer_inner * grad_norm;
                grad_norm = grad_p2(n) * getColumn(norm, n);
                grad_p2(n) = grad_norm - outer_inner * grad_norm;
            }
        }

        // Finally, apply the normalization
        poly_ld = colwiseProduct(poly_ld, norm);

    }


    /* ------------- */
    /*   INTENSITY   */
    /* ------------- */

    /**
    Compute the polynomial basis at a point; templated for AD capability

    */
    template <class T> template <typename U>
    inline void Map<T>::poly_basis(Power<U>& xpow, Power<U>& ypow,
                                   VectorT<U>& basis) {
        int l, m, mu, nu, n = 0;
        U z = sqrt(1.0 - xpow() * xpow() - ypow() * ypow());
        for (l=0; l<lmax+1; ++l) {
            for (m=-l; m<l+1; ++m) {
                mu = l - m;
                nu = l + m;
                if ((nu % 2) == 0) {
                    if ((mu > 0) && (nu > 0))
                        basis(n) = xpow(mu / 2) * ypow(nu / 2);
                    else if (mu > 0)
                        basis(n) = xpow(mu / 2);
                    else if (nu > 0)
                        basis(n) = ypow(nu / 2);
                    else
                        basis(n) = 1;
                } else {
                    if ((mu > 1) && (nu > 1))
                        basis(n) = xpow((mu - 1) / 2) *
                                   ypow((nu - 1) / 2) * z;
                    else if (mu > 1)
                        basis(n) = xpow((mu - 1) / 2) * z;
                    else if (nu > 1)
                        basis(n) = ypow((nu - 1) / 2) * z;
                    else
                        basis(n) = z;
                }
                n++;
            }
        }
    }

    /**
    Evaluate the map at a given (x0, y0) coordinate

    */
    template <class T>
    inline Row<T> Map<T>::evaluate(const Scalar<T>& theta_,
                                   const Scalar<T>& x_,
                                   const Scalar<T>& y_,
                                   bool gradient) {

        // If we're computing the gradient,
        // call the specialized function instead
        if (gradient)
            return evaluate_with_gradient(theta_, x_, y_);

        // Bind references to temporaries for speed
        Row<T>& result(tmp.tmpRow[0]);
        T& Ry(tmp.tmpT[0]);
        T& A1Ry(tmp.tmpT[1]);
        Power<Scalar<T>>& xpow(tmp.tmpPower[0]);
        Power<Scalar<T>>& ypow(tmp.tmpPower[1]);
        VectorT<Scalar<T>>& pT(tmp.tmpRowVector[0]);
        pT.resize(N);

        // Convert to internal types
        Scalar<T> x0 = x_;
        Scalar<T> y0 = y_;
        Scalar<T> theta = theta_ * (pi<Scalar<T>>() / 180.);

        // Disable rotation for constant maps
        if (y_deg == 0) theta = 0;

        // Check if outside the sphere
        if (x0 * x0 + y0 * y0 > 1.0) {
            result *= NAN;
            return result;
        }

        if ((theta == cache_theta) && (cache_oper == CACHE_EVAL)) {

            // We use the cached version of the polynomial map
            A1Ry = cache_p;

        } else {

            // Rotate the spherical harmonic map into view
            if (y_deg > 0) {
                W.rotate(cos(theta), sin(theta), Ry);
                A1Ry = B.A1 * Ry;
            } else {
                A1Ry = p;
            }

            if (u_deg > 0) {
                // Apply limb darkening
                limb_darken(A1Ry, p_uy);
                A1Ry = p_uy;
            }

            // Cache the polynomial map
            cache_oper = CACHE_EVAL;
            cache_theta = theta;
            cache_p = A1Ry;

        }

        // Compute the polynomial basis
        xpow.reset(x0);
        ypow.reset(y0);
        poly_basis(xpow, ypow, pT);

        // Dot the coefficients in to our polynomial map
        return pT * A1Ry;

    }

    /**
    Evaluate the map at a given (x0, y0) coordinate and compute the gradient

    TODO: We should really look into rotation & limb darkening
          caching for this function. This is currently VERY SLOW
          for cases where the map is evaluated over a grid at
          constant theta and limb darkening.

    */
    template <class T>
    inline Row<T> Map<T>::evaluate_with_gradient(const Scalar<T>& theta_,
                                                 const Scalar<T>& x_,
                                                 const Scalar<T>& y_) {

        // Convert to internal type
        Scalar<T> x0 = x_;
        Scalar<T> y0 = y_;
        Scalar<T> theta = theta_ * (pi<Scalar<T>>() / 180.);

        // Bind references to temporaries for speed
        Row<T>& result(tmp.tmpRow[0]);
        T& Ry(tmp.tmpT[0]);
        T& A1Ry(tmp.tmpT[1]);
        T& dRdthetay(tmp.tmpT[2]);
        VectorT<Scalar<T>>& pT(tmp.tmpRowVector[0]);
        pT.resize(N);
        VectorT<Scalar<T>>& pTA1(tmp.tmpRowVector[1]);
        pTA1.resize(N);
        VectorT<Scalar<T>>& pTA1R(tmp.tmpRowVector[2]);
        pTA1R.resize(N);
        ADScalar<Scalar<T>, 2>& x0_grad(tmp.tmpADScalar2[0]);
        ADScalar<Scalar<T>, 2>& y0_grad(tmp.tmpADScalar2[1]);
        Power<ADScalar<Scalar<T>, 2>>& xpow(tmp.tmpPowerOfADScalar2[0]);
        Power<ADScalar<Scalar<T>, 2>>& ypow(tmp.tmpPowerOfADScalar2[1]);
        VectorT<ADScalar<Scalar<T>, 2>>& pT_grad(tmp.tmpRowVectorOfADScalar2[0]);
        pT_grad.resize(N);
        VectorT<Scalar<T>>& dIdu(tmp.tmpRowVector[3]);
        VectorT<Scalar<T>>& dIdp_u(tmp.tmpRowVector[4]);

        // Check if outside the sphere
        if (x0 * x0 + y0 * y0 > 1.0) {
            dI *= NAN;
            result *= NAN;
            return result;
        }

        // Rotate the map into view
        // Note that there's no skipping this, even for constant
        // maps, so we get the right derivatives below
        W.compute(cos(theta), sin(theta));
        if (theta == 0) {
            A1Ry = p;
        } else {
            for (int l = 0; l < lmax + 1; l++)
                Ry.block(l * l, 0, 2 * l + 1, nwav) =
                    W.R[l] * y.block(l * l, 0, 2 * l + 1, nwav);
            A1Ry = B.A1 * Ry;
        }

        // Apply limb-darkening and compute its derivative
        // Note that we do this even if there's no limb
        // darkening set, to get the correct derivs
        limb_darken(A1Ry, p_uy, true);
        A1Ry = p_uy;

        // Compute the polynomial basis and its x and y derivs
        x0_grad.value() = x0;
        x0_grad.derivatives() = Vector<Scalar<T>>::Unit(2, 0);
        y0_grad.value() = y0;
        y0_grad.derivatives() = Vector<Scalar<T>>::Unit(2, 1);
        xpow.reset(x0_grad);
        ypow.reset(y0_grad);
        poly_basis(xpow, ypow, pT_grad);
        setRow(dI, 1, Scalar<T>(0.0));
        setRow(dI, 2, Scalar<T>(0.0));
        for (int i = 0; i < N; i++) {
            setRow(dI, 1, Row<T>(getRow(dI, 1) +
                                 pT_grad(i).derivatives()(0) *
                                 getRow(A1Ry, i)));
            setRow(dI, 2, Row<T>(getRow(dI, 2) +
                                 pT_grad(i).derivatives()(1) *
                                 getRow(A1Ry, i)));
            pT(i) = pT_grad(i).value();
        }

        // Compute dR/dtheta . y
        for (int l = 0; l < lmax + 1; ++l)
            dRdthetay.block(l * l, 0, 2 * l + 1, nwav) =
                W.dRdtheta[l] * y.block(l * l, 0, 2 * l + 1, nwav);

        // Compute the map derivs and the theta deriv
        // dI / dtheta = p^T . grad_p1 . A1 . d(R . y) / dtheta
        // dI / dy = p^T . grad_p1 . A1 . R
        if (u_deg > 0) {
            for (int n = 0; n < nwav; ++n) {
                pTA1 = pT * grad_p1(n) * B.A1;
                if (theta == 0) {
                    for (int i = 0; i < N; i++)
                        dI(3 + i, n) = pTA1(i);
                } else {
                    for (int l = 0; l < lmax + 1; l++)
                        pTA1R.segment(l * l, 2 * l + 1) =
                            pTA1.segment(l * l, 2 * l + 1) * W.R[l];
                    for (int i = 0; i < N; i++)
                        dI(3 + i, n) = pTA1R(i);
                }
                dI(0, n) = dot(pTA1, getColumn(dRdthetay, n))
                           * (pi<Scalar<T>>() / 180.);
            }
        } else {
            pTA1 = pT * B.A1;
            if (theta == 0) {
                for (int i = 0; i < N; i++)
                    setRow(dI, 3 + i, pTA1(i));
            } else {
                for (int l = 0; l < lmax + 1; l++)
                    pTA1R.segment(l * l, 2 * l + 1) =
                        pTA1.segment(l * l, 2 * l + 1) * W.R[l];
                for (int i = 0; i < N; i++)
                    setRow(dI, 3 + i, pTA1R(i));
            }
            setRow(dI, 0, Row<T>(dot(pTA1, dRdthetay)
                          * (pi<Scalar<T>>() / 180.)));
        }

        // Compute the derivs with respect to the limb darkening coeffs
        // dI / du = p^T . grad_p2 . dp_u / du
        for (int n = 0; n < nwav; ++n) {
            dIdp_u = pT * grad_p2(n);
            dIdu = dIdp_u * dp_udu(n);
            dI.block(3 + N, n, lmax, 1) = dIdu.segment(1, lmax).transpose();
        }

        // Dot the coefficients in to our polynomial map
        return dot(pT, A1Ry);

    }


    /* ------------- */
    /*      FLUX     */
    /* ------------- */

    /**
    Compute the flux during or outside of an occultation

    */
    template <class T>
    inline Row<T> Map<T>::flux(const Scalar<T>& theta_,
                               const Scalar<T>& xo_,
                               const Scalar<T>& yo_,
                               const Scalar<T>& ro_,
                               bool gradient) {

        if (gradient) {
            // If we're computing the gradient as well,
            // call the specialized functions
            if (y_deg == 0)
                return flux_ld_with_gradient(xo_, yo_, ro_);
            else
                return flux_with_gradient(theta_, xo_, yo_, ro_);
        } else if (y_deg == 0) {
            // If only the Y_{0,0} term is set, call the
            // faster method for pure limb-darkening
            return flux_ld(xo_, yo_, ro_);
        }

        // Bind references to temporaries for speed
        Row<T>& result(tmp.tmpRow[0]);
        T& Ry(tmp.tmpT[0]);
        T& A1Ry(tmp.tmpT[1]);
        T& RRy(tmp.tmpT[2]);
        T& ARRy(tmp.tmpT[3]);

        // Convert to internal types
        Scalar<T> xo = xo_;
        Scalar<T> yo = yo_;
        Scalar<T> ro = ro_;
        Scalar<T> theta = theta_ * (pi<Scalar<T>>() / 180.);

        // Impact parameter
        Scalar<T> b = sqrt(xo * xo + yo * yo);

        // Check for complete occultation
        if (b <= ro - 1) {
            setZero(result);
            return result;
        }

        // Rotate the map into view
        if (y_deg > 0) {
            W.rotate(cos(theta), sin(theta), Ry);
        } else {
            Ry = y;
        }

        // No occultation
        if ((b >= 1 + ro) || (ro == 0)) {

            // Easy. Note that limb-darkening does not
            // affect the total disk-integrated flux!
            return (B.rTA1 * Ry);

        // Occultation
        } else {

            // Apply limb darkening
            if (u_deg > 0) {

                if ((theta == cache_theta) && (cache_oper == CACHE_FLUX)) {

                    // Easy. Use the cached rotated map
                    Ry = cache_y;

                } else {

                    // Transform into the polynomial basis
                    A1Ry = B.A1 * Ry;

                    // Limb-darken it
                    limb_darken(A1Ry, p_uy);

                    // Back to the spherical harmonic basis
                    Ry = B.A1Inv * p_uy;

                    // Cache the map
                    cache_oper = CACHE_FLUX;
                    cache_theta = theta;
                    cache_y = Ry;

                }

            }

            // Rotate the map to align the occultor with the +y axis
            // Change basis to Green's polynomials
            if ((b > 0) && ((xo != 0) || (yo < 0))) {
                W.rotatez(yo / b, xo / b, Ry, RRy);
                ARRy = B.A * RRy;
            } else {
                ARRy = B.A * Ry;
            }

            // Compute the sT vector (sparsely)
            for (int n = 0; n < N; ++n)
                G.skip(n) = !(ARRy.block(n, 0, 1, nwav).array() != 0.0).any();
            G.compute(b, ro);

            // Dot the result in and we're done
            return G.sT * ARRy;

        }

    }

    /**
    Compute the flux during or outside of an occultation
    for a pure limb-darkened map (Y_{l,m} = 0 for l > 0).

    */
    template <class T>
    inline Row<T> Map<T>::flux_ld(const Scalar<T>& xo_,
                                  const Scalar<T>& yo_,
                                  const Scalar<T>& ro_) {

        // Convert to internal types
        Scalar<T> xo = xo_;
        Scalar<T> yo = yo_;
        Scalar<T> ro = ro_;

        // Impact parameter
        Scalar<T> b = sqrt(xo * xo + yo * yo);

        // Check for complete occultation
        if (b <= ro - 1) {
            setZero(ctmp);
            return ctmp;
        }

        // No occultation
        if ((b >= 1 + ro) || (ro == 0)) {

            // Easy: the disk-integrated intensity
            // is just the Y_{0,0} coefficient
            return getRow(y, 0);

        // Occultation
        } else {

            // Compute the sT vector (sparsely)
            for (int n = 0; n < N; ++n)
                G.skip(n) = !(g_u.block(n, 0, 1, nwav).array() != 0.0).any();
            G.compute(b, ro);

            // Dot the result in and we're done
            return G.sT * g_u;

        }

    }

    /**
    Compute the flux during or outside of an occultation
    for a pure limb-darkened map (Y_{l,m} = 0 for l > 0).

    TODO: Add limb darkening.

    */
    template <class T>
    inline Row<T> Map<T>::flux_ld_with_gradient(const Scalar<T>& xo_,
                                                const Scalar<T>& yo_,
                                                const Scalar<T>& ro_) {

        throw errors::ToDoError("Implement derivative of limb-darkened flux.");

        // Convert to internal types
        Scalar<T> xo = xo_;
        Scalar<T> yo = yo_;
        Scalar<T> ro = ro_;

        // Impact parameter
        Scalar<T> b = sqrt(xo * xo + yo * yo);

        // Check for complete occultation
        if (b <= ro - 1) {
            setZero(ctmp);
            return ctmp;
        }

        // No occultation
        if ((b >= 1 + ro) || (ro == 0)) {

            // Easy: the disk-integrated intensity
            // is just the Y_{0,0} coefficient
            return getRow(y, 0);

        // Occultation
        } else {

            // Compute the sT vector (sparsely)
            for (int n = 0; n < N; ++n)
                G.skip(n) = !(g_u.block(n, 0, 1, nwav).array() != 0.0).any();
            G.compute(b, ro);

            // Dot the result in and we're done
            return G.sT * g_u;

        }

    }

    /**
    Compute the flux during or outside of an occultation and its gradient

    TODO: Add limb darkening

    */

    template <class T>
    inline Row<T> Map<T>::flux_with_gradient(const Scalar<T>& theta_deg,
                                             const Scalar<T>& xo_,
                                             const Scalar<T>& yo_,
                                             const Scalar<T>& ro_) {

        // Convert to internal type
        Scalar<T> xo = xo_;
        Scalar<T> yo = yo_;
        Scalar<T> ro = ro_;
        Scalar<T> theta = theta_deg * (pi<Scalar<T>>() / 180.);

        // Impact parameter
        Scalar<T> b = sqrt(xo * xo + yo * yo);

        // Check for complete occultation
        if (b <= ro - 1) {
            setZero(dF);
            setZero(ctmp);
            return ctmp;
        }

        // Rotate the map into view
        W.compute(cos(theta), sin(theta));
        if (theta == 0) {
            ptr_Ry = &y;
        } else {
            for (int l = 0; l < lmax + 1; l++)
                Ry.block(l * l, 0, 2 * l + 1, nwav) =
                    W.R[l] * y.block(l * l, 0, 2 * l + 1, nwav);
            mtmp = Ry;
            ptr_Ry = &mtmp;
        }

        // No occultation
        if ((b >= 1 + ro) || (ro == 0)) {

            // Compute the theta deriv
            for (int l = 0; l < lmax + 1; l++)
                dRdthetay.block(l * l, 0, 2 * l + 1, nwav) =
                    W.dRdtheta[l] * y.block(l * l, 0, 2 * l + 1, nwav);
            setRow(dF, 0, Row<T>(dot(B.rTA1, dRdthetay) * (pi<Scalar<T>>() / 180.)));

            // The x, y, and r derivs are trivial
            setRow(dF, 1, Scalar<T>(0.0));
            setRow(dF, 2, Scalar<T>(0.0));
            setRow(dF, 3, Scalar<T>(0.0));

            // Compute the map derivs
            pTA1 = pT * B.A1;
            if (theta == 0) {
                for (int i = 0; i < N; i++)
                    setRow(dF, 4 + i, B.rTA1(i));
            } else {
                for (int l = 0; l < lmax + 1; l++)
                    vtmp.segment(l * l, 2 * l + 1) =
                        B.rTA1.segment(l * l, 2 * l + 1) * W.R[l];
                for (int i = 0; i < N; i++)
                    setRow(dF, 4 + i, vtmp(i));
            }

            // We're done!
            return (B.rTA1 * (*ptr_Ry));

        // Occultation
        } else {

            // Align occultor with the +y axis
            Scalar<T> xo_b = xo / b;
            Scalar<T> yo_b = yo / b;
            if ((b > 0) && ((xo != 0) || (yo < 0))) {
                W.rotatez(yo_b, xo_b, *ptr_Ry, mtmp2);
                ptr_RRy = &mtmp2;
            } else {
                W.cosmt = Vector<Scalar<T>>::Constant(N, 1.0);
                W.sinmt = Vector<Scalar<T>>::Constant(N, 0.0);
                ptr_RRy = ptr_Ry;
            }

            // Perform the rotation + change of basis
            ARRy = B.A * (*ptr_RRy);

            // Compute the sT vector using AutoDiff
            b_grad.value() = b;
            ro_grad.value() = ro;
            G_grad.compute(b_grad, ro_grad);

            // Compute the b and ro derivs
            setZero(dFdb);
            setRow(dF, 3, Scalar<T>(0.0));
            for (int i = 0; i < N; i++) {

                // b deriv
                dFdb += G_grad.sT(i).derivatives()(0) * getRow(ARRy, i);

                // ro deriv
                setRow(dF, 3, Row<T>(getRow(dF, 3) +
                                     G_grad.sT(i).derivatives()(1) * getRow(ARRy, i)));

                // Store the value of s^T
                G.sT(i) = G_grad.sT(i).value();

            }

            // Solution vector in spherical harmonic basis
            sTA = G.sT * B.A;

            // Compute stuff involving the Rprime rotation matrix
            int m;
            for (int l = 0; l < lmax + 1; l++) {
                for (int j = 0; j < 2 * l + 1; j++) {
                    m = j - l;
                    sTAR(l * l + j) = sTA(l * l + j) * W.cosmt(l * l + j) +
                                      sTA(l * l + 2 * l - j) * W.sinmt(l * l + j);
                    sTAdRdtheta(l * l + j) = sTA(l * l + 2 * l - j) * m * W.cosmt(l * l + j) -
                                             sTA(l * l + j) * m * W.sinmt(l * l + j);
                }
            }

            // Compute the xo and yo derivs using the chain rule
            rtmp = dot(sTAdRdtheta, *ptr_Ry) / b;
            setRow(dF, 1, Row<T>((xo_b * dFdb) + (yo_b * rtmp)));
            setRow(dF, 2, Row<T>((yo_b * dFdb) - (xo_b * rtmp)));

            // Compute the theta deriv
            for (int l = 0; l < lmax + 1; l++)
                dRdthetay.block(l * l, 0, 2 * l + 1, nwav) =
                    W.dRdtheta[l] * y.block(l * l, 0, 2 * l + 1, nwav);
            setRow(dF, 0, Row<T>(dot(sTAR, dRdthetay) * (pi<Scalar<T>>() / 180.)));

            // Compute the map derivs
            if (theta == 0) {
                for (int i = 0; i < N; i++)
                    setRow(dF, 4 + i, sTAR(i));
            } else {
                for (int l = 0; l < lmax + 1; l++)
                    vtmp.segment(l * l, 2 * l + 1) =
                        sTAR.segment(l * l, 2 * l + 1) * W.R[l];
                for (int i = 0; i < N; i++)
                    setRow(dF, 4 + i, vtmp(i));
            }

            // Dot the result in and we're done
            return G.sT * ARRy;

        }

    }

    //! Human-readable name of the map
    template <class T>
    std::string info(const Map<T>& map) {
        std::ostringstream os;
        os << "<"
           << "Map of "
           << "degree " << map.y_deg << " "
           << "with ";
        if (map.u_deg == 0)
            os << "no ";
        else if (map.u_deg == 1)
            os << "1st order ";
        else if (map.u_deg == 2)
            os << "2nd order ";
        else if (map.u_deg == 3)
            os << "3rd order ";
        else
            os << map.u_deg << "th order ";
        os << "limb darkening"
           << ">";
        return std::string(os.str());
    }

    //! Human-readable name of the map (multi)
    template <>
    std::string info(const Map<Vector<Multi>>& map) {
        std::ostringstream os;
        os << "<"
           << STARRY_NMULTI << "-digit precision "
           << "map of "
           << "degree " << map.y_deg << " "
           << "with ";
        if (map.u_deg == 0)
            os << "no ";
        else if (map.u_deg == 1)
            os << "1st order ";
        else if (map.u_deg == 2)
            os << "2nd order ";
        else if (map.u_deg == 3)
            os << "3rd order ";
        else
            os << map.u_deg << "th order ";
        os << "limb darkening"
           << ">";
        return std::string(os.str());
    }

    //! Human-readable name of the map (spectral)
    template <>
    std::string info(const Map<Matrix<double>>& map) {
        std::ostringstream os;
        os << "<"
           << "Map of "
           << "degree " << map.y_deg << " "
           << "with ";
        if (map.nwav == 1)
            os << "one wavelength bin and ";
        else
            os << map.nwav << " wavelength bins and ";
        if (map.u_deg == 0)
            os << "no ";
        else if (map.u_deg == 1)
            os << "1st order ";
        else if (map.u_deg == 2)
            os << "2nd order ";
        else if (map.u_deg == 3)
            os << "3rd order ";
        else
            os << map.u_deg << "th order ";
        os << "limb darkening"
           << ">";
        return std::string(os.str());
    }


}; // namespace maps

#undef CACHE_NONE
#undef CACHE_FLUX
#undef CACHE_EVAL

#endif
