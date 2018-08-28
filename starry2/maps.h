/**
Defines the surface map class.

TODO: Get rid of evaluation gradients
      Map evaluations only via the operator() on a Map.

TODO: Macro for loops involving nwav and specialize for nwav = 1?
      Essentially we'd replace

        for (int n = 0; n < nwav; ++n) {
            ...
        }

      with

        WAVELENGTH_LOOP(n)
            ...
        END_WAVELENGTH_LOOP

      and just do {int n = 0; ...} for nwav = 1.

TODO: Speed up limb-darkened map rotations, since
      the effective degree of the map is lower.
      This can easily be implemented in W.rotate
      if we pass y_deg along. Don't implement it in
      W.compute, since we need gradients.

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

    /**
    Stores cached Map variables.

    */
    template <class T>
    class Cache {

        public:

            static const int NONE = 0;                                          /**< Cached operation identifier (empty cache)*/
            static const int FLUX = 1;                                          /**< Cached operation identifier (flux) */
            static const int EVAL = 2;                                          /**< Cached operation identifier (evaluation) */
            int oper;                                                           /**< Cached operation identifier */
            Scalar<T> theta;                                                    /**< Cached rotation angle */
            T p;                                                                /**< Cached polynomial map */
            T y;                                                                /**< Cached Ylm map */

            //! Default constructor
            Cache() {
                clear();
            }

            //! Clear the cache
            inline void clear() {
                oper = NONE;
                theta = NAN;
            }


    };

    /**
    Temporary vector/matrix/tensor storage class.

    */
    template <class T>
    class Temporary {

            static constexpr int TLEN = 6;
            static constexpr int RLEN = 4;
            static constexpr int CLEN = 1;
            static constexpr int VLEN = 1;
            static constexpr int VTLEN = 8;
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
                // All other containers need to be allocated elsewhere
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

            // Map stuff
            T y;                                                                /**< The map coefficients in the spherical harmonic basis */
            T p;                                                                /**< The map coefficients in the polynomial basis */
            T g;                                                                /**< The map coefficients in the Green's basis */
            int y_deg;                                                          /**< Highest degree set by the user in the spherical harmonic vector */
            UnitVector<Scalar<T>> axis;                                         /**< The axis of rotation for the map */
            Basis<Scalar<T>> B;                                                 /**< Basis transform stuff */
            Wigner<T> W;                                                        /**< The class controlling rotations */
            Greens<Scalar<T>> G;                                                /**< The occultation integral solver class */
            Greens<ADScalar<Scalar<T>, 2>> G_grad;                              /**< The occultation integral solver class w/ AutoDiff capability */
            Scalar<T> tol;                                                      /**< Machine epsilon */

            // Limb darkening
            T u;                                                                /**< The limb darkening coefficients */
            T p_u;                                                              /**< The limb darkening coefficients in the polynomial basis */
            T g_u;                                                              /**< The limb darkening coefficients in the Green's basis */
            int u_deg;                                                          /**< Highest degree set by the user in the limb darkening vector */
            Vector<Matrix<Scalar<T>>> dp_udu;                                   /**< Deriv of limb darkening polynomial w/ respect to limb darkening coeffs */
            Vector<Matrix<Scalar<T>>> dg_udu;                                   /**< Deriv of limb darkening Green's polynomials w/ respect to limb darkening coeffs */
            VectorT<Matrix<Scalar<T>>> dLDdp;                                   /**< Derivative of the limb-darkened polynomial w.r.t `p` */
            VectorT<Matrix<Scalar<T>>> dLDdp_u;                                 /**< Derivative of the limb-darkened polynomial w.r.t `p_u` */
            T p_uy;                                                             /**< The instantaneous limb-darkened map in the polynomial basis */

            // Temporaries and cache
            Temporary<T> tmp;
            Cache<T> cache;

            // External info function
            template <class U>
            friend std::string info(const Map<U>& map);

            // Private methods
            inline void check_degree();
            inline void update_y();
            inline void update_u();
            inline void limb_darken(const T& poly, T& poly_ld,
                                    bool gradient=false);
            template <typename U>
            inline void poly_basis(Power<U>& xpow, Power<U>& ypow,
                                   VectorT<U>& basis);
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
                B(lmax),
                W(lmax, nwav, (*this).y, (*this).axis),
                G(lmax),
                G_grad(lmax),
                tol(mach_eps<Scalar<T>>()),
                dp_udu(nwav),
                dg_udu(nwav),
                tmp(N, nwav),
                cache() {

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
                y.resize(N, nwav);
                p.resize(N, nwav);
                g.resize(N, nwav);
                u.resize(lmax + 1, nwav);
                p_u.resize(N, nwav);
                g_u.resize(N, nwav);
                dI.resize(3 + N + lmax, nwav);
                dF.resize(4 + N + lmax, nwav);

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
        cache.clear();
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

        // Compute the derivative of the LD polynomials
        // with respect to the LD coefficients
        for (int n = 0; n < nwav; ++n) {
            dp_udu(n) = -getColumn(p_u, n) * B.rTU1;
            dp_udu(n) /= dot(B.rT, getColumn(p_u, n));
            dp_udu(n) += B.U1;
            dp_udu(n) *= getColumn(ld_norm, n);
            dg_udu(n) = B.A2 * dp_udu(n);
        }

        // Apply the normalization
        p_u = colwiseProduct(p_u, ld_norm);

        // Update the limb darkening Green's map
        g_u = B.A2 * p_u;

        // Clear the cache
        cache.clear();
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

        // Clear the cache
        cache.clear();

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
            polymul(y_deg, poly, u_deg, p_u, lmax, poly_ld, dLDdp, dLDdp_u);
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
        // the normalization to the gradient. There may
        // exist faster way that avoids the for loop.
        if (gradient) {
            for (int n = 0; n < nwav; ++n) {
                outer_inner = getColumn(poly_ld, n) * B.rT;
                outer_inner /= getColumn(rTp_ld, n);
                grad_norm = dLDdp(n) * getColumn(norm, n);
                dLDdp(n) = grad_norm + outer_inner - outer_inner * grad_norm;
                grad_norm = dLDdp_u(n) * getColumn(norm, n);
                dLDdp_u(n) = grad_norm - outer_inner * grad_norm;
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

        if ((theta == cache.theta) && (cache.oper == cache.EVAL)) {

            // We use the cached version of the polynomial map
            A1Ry = cache.p;

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
            cache.oper = cache.EVAL;
            cache.theta = theta;
            cache.p = A1Ry;

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

    TODO: Remove this function.

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
        VectorT<Scalar<T>>& dIdu(tmp.tmpRowVector[3]);
        VectorT<Scalar<T>>& dIdp_u(tmp.tmpRowVector[4]);
        ADScalar<Scalar<T>, 2>& x0_grad(tmp.tmpADScalar2[0]);
        ADScalar<Scalar<T>, 2>& y0_grad(tmp.tmpADScalar2[1]);
        Power<ADScalar<Scalar<T>, 2>>& xpow(tmp.tmpPowerOfADScalar2[0]);
        Power<ADScalar<Scalar<T>, 2>>& ypow(tmp.tmpPowerOfADScalar2[1]);
        VectorT<ADScalar<Scalar<T>, 2>>& pT_grad(tmp.tmpRowVectorOfADScalar2[0]);
        pT_grad.resize(N);

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
        // dI / dtheta = p^T . dLDdp . A1 . d(R . y) / dtheta
        // dI / dy = p^T . dLDdp . A1 . R
        if (u_deg > 0) {
            for (int n = 0; n < nwav; ++n) {
                pTA1 = pT * dLDdp(n) * B.A1;
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
        // dI / du = p^T . dLDdp_u . dp_u / du
        for (int n = 0; n < nwav; ++n) {
            dIdp_u = pT * dLDdp_u(n);
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

    TODO: Five flux functions:

        - flux
        - flux_ld
        - flux_with_gradient
        - flux_ld_with_gradient
            (This one sets dF/dy to NAN)
        - flux_sph_with_gradient (Copy from https://github.com/rodluger/starry/blob/082deebe9783bfe87a938e40956ede6f3d6d6707/starry2/maps.h#L1214)
            (This one sets dF/du to NAN)
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

                if ((theta == cache.theta) && (cache.oper == cache.FLUX)) {

                    // Easy. Use the cached rotated map
                    Ry = cache.y;

                } else {

                    // Transform into the polynomial basis
                    A1Ry = B.A1 * Ry;

                    // Limb-darken it
                    limb_darken(A1Ry, p_uy);

                    // Back to the spherical harmonic basis
                    Ry = B.A1Inv * p_uy;

                    // Cache the map
                    cache.oper = cache.FLUX;
                    cache.theta = theta;
                    cache.y = Ry;

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

        // Bind references to temporaries for speed
        Row<T>& result(tmp.tmpRow[0]);

        // Convert to internal types
        Scalar<T> xo = xo_;
        Scalar<T> yo = yo_;
        Scalar<T> ro = ro_;

        // Impact parameter
        Scalar<T> b = sqrt(xo * xo + yo * yo);

        // Check for complete occultation
        if (b <= ro - 1) {
            setZero(result);
            return result;
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

    TODO: Add a note to the docs about why dF/dY_{l,m} = NAN
          for l > 0, and explain how to override this.

    */
    template <class T>
    inline Row<T> Map<T>::flux_ld_with_gradient(const Scalar<T>& xo_,
                                                const Scalar<T>& yo_,
                                                const Scalar<T>& ro_) {

        // Bind references to temporaries for speed
        Row<T>& result(tmp.tmpRow[0]);
        Row<T>& dFdb(tmp.tmpRow[1]);
        VectorT<Scalar<T>>& dFdu(tmp.tmpRowVector[0]);
        ADScalar<Scalar<T>, 2>& b_grad(tmp.tmpADScalar2[0]);
        ADScalar<Scalar<T>, 2>& ro_grad(tmp.tmpADScalar2[1]);

        // Convert to internal types
        Scalar<T> xo = xo_;
        Scalar<T> yo = yo_;
        Scalar<T> ro = ro_;

        // Impact parameter
        Scalar<T> b = sqrt(xo * xo + yo * yo);

        // Check for complete occultation
        if (b <= ro - 1) {
            setZero(dF);
            setZero(result);
            return result;
        }

        // We intentionally don't compute the spherical
        // harmonic coeff derivs above Y_{0,0}, since that
        // would make this function slow. If users *really*
        // need them, set one of the l > 0 coeffs to a very
        // small (~1e-14) value to force the code to use the
        // generalized `flux` method.
        for (int i = 1; i < N; ++i)
            setRow(dF, 4 + i, Scalar<T>(NAN));

        // The theta deriv is always zero, since
        // pure limb-darkened maps can't be rotated.
        setRow(dF, 0, Scalar<T>(0.0));

        // No occultation
        if ((b >= 1 + ro) || (ro == 0)) {

            // The x, y, and r derivs are trivial
            setRow(dF, 1, Scalar<T>(0.0));
            setRow(dF, 2, Scalar<T>(0.0));
            setRow(dF, 3, Scalar<T>(0.0));

            // Compute the Y_{0,0} deriv, which is trivial
            setRow(dF, 4, Scalar<T>(1.0));

            // The limb darkening derivs are zero, since
            // they don't affect the total flux!
            for (int i = 0; i < lmax; ++i)
                setRow(dF, 4 + N + i, Scalar<T>(0.0));

            // Easy: the disk-integrated intensity
            // is just the Y_{0,0} coefficient
            return getRow(y, 0);

        // Occultation
        } else {

            // Compute the sT vector using AutoDiff
            b_grad.value() = b;
            b_grad.derivatives() = Vector<Scalar<T>>::Unit(2, 0);
            ro_grad.value() = ro;
            ro_grad.derivatives() = Vector<Scalar<T>>::Unit(2, 1);
            G_grad.compute(b_grad, ro_grad);

            // Compute the b and ro derivs
            setZero(dFdb);
            setRow(dF, 3, Scalar<T>(0.0));
            for (int i = 0; i < N; ++i) {
                // dF / db = dsT / db . g_u
                dFdb += G_grad.sT(i).derivatives()(0) * getRow(g_u, i);
                // dF / dro = dsT / dro . g_u
                setRow(dF, 3, Row<T>(getRow(dF, 3) +
                                     G_grad.sT(i).derivatives()(1) *
                                     getRow(g_u, i)));
                // Store the value of s^T
                G.sT(i) = G_grad.sT(i).value();
            }

            // Compute the resulting flux
            result = dot(G.sT, g_u);

            // Compute the x and y derivs (straighforward chain rule)
            setRow(dF, 1, Row<T>(dFdb * xo / b));
            setRow(dF, 2, Row<T>(dFdb * yo / b));

            // Compute the Y_{0, 0} deriv (straightforward since it's linear)
            setRow(dF, 4, cwiseQuotient(result, getRow(y, 0)));

            // Compute the derivs with respect to the limb darkening coeffs
            // dF / du = s^T . dg_u / du
            for (int n = 0; n < nwav; ++n) {
                dFdu = G.sT * dg_udu(n);
                dF.block(4 + N, n, lmax, 1) = dFdu.segment(1, lmax).transpose();
            }

            return result;

        }

    }

    /**
    Compute the flux during or outside of an occultation and its gradient

    TODO: Add limb darkening derivs

    */

    template <class T>
    inline Row<T> Map<T>::flux_with_gradient(const Scalar<T>& theta_deg,
                                             const Scalar<T>& xo_,
                                             const Scalar<T>& yo_,
                                             const Scalar<T>& ro_) {

        // Bind references to temporaries for speed
        Row<T>& result(tmp.tmpRow[0]);
        Row<T>& dFdb(tmp.tmpRow[1]);
        Row<T>& sTAdRdthetaLDRy_b(tmp.tmpRow[2]);
        T& Ry(tmp.tmpT[0]);
        T& RLDRy(tmp.tmpT[1]);
        T& ARLDRy(tmp.tmpT[2]);
        T& A1Ry(tmp.tmpT[3]);
        T& LDRy(tmp.tmpT[4]);
        T& dLDRydtheta(tmp.tmpT[5]);
        VectorT<Scalar<T>>& sTA(tmp.tmpRowVector[0]);
        VectorT<Scalar<T>>& sTAR(tmp.tmpRowVector[1]);
        sTAR.resize(N);
        VectorT<Scalar<T>>& sTARdLDdpA1(tmp.tmpRowVector[2]);
        VectorT<Scalar<T>>& sTARdLDdpA1R(tmp.tmpRowVector[3]);
        sTARdLDdpA1R.resize(N);
        VectorT<Scalar<T>>& sTAdRdtheta(tmp.tmpRowVector[4]);
        sTAdRdtheta.resize(N);
        VectorT<Scalar<T>>& rTA1R(tmp.tmpRowVector[5]);
        rTA1R.resize(N);

        VectorT<Scalar<T>>& dFdu(tmp.tmpRowVector[6]);
        VectorT<Scalar<T>>& dFdp_u(tmp.tmpRowVector[7]);

        ADScalar<Scalar<T>, 2>& b_grad(tmp.tmpADScalar2[0]);
        ADScalar<Scalar<T>, 2>& ro_grad(tmp.tmpADScalar2[1]);
        VectorT<Matrix<Scalar<T>>>& A1dLDdpA1(tmp.tmpRowVectorOfMatrices[0]);
        A1dLDdpA1.resize(nwav);
        Matrix<Scalar<T>>& A1dLDdpA1dRdtheta(tmp.tmpMatrix[0]);
        A1dLDdpA1dRdtheta.resize(N, N);

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
            setZero(result);
            return result;
        }

        // Rotate the map into view
        // TODO: Cache this operation *here*
        //       Save W.R, W.dRdtheta, Ry
        W.compute(cos(theta), sin(theta));
        if (theta == 0) {
            Ry = y;
        } else {
            for (int l = 0; l < lmax + 1; l++)
                Ry.block(l * l, 0, 2 * l + 1, nwav) =
                    W.R[l] * y.block(l * l, 0, 2 * l + 1, nwav);
        }

        // No occultation
        if ((b >= 1 + ro) || (ro == 0)) {

            // Compute d(R . y) / dtheta
            // Since limb darkening doesn't change the total flux,
            // we don't have to apply it here.
            for (int n = 0; n < nwav; ++n)
                for (int l = 0; l < lmax + 1; ++l)
                    dLDRydtheta.block(l * l, n, 2 * l + 1, 1) =
                        W.dRdtheta[l] *
                        y.block(l * l, n, 2 * l + 1, 1);

            // Compute the theta deriv
            setRow(dF, 0, Row<T>(dot(B.rTA1, dLDRydtheta) *
                                (pi<Scalar<T>>() / 180.)));

            // The x, y, and r derivs are trivial
            setRow(dF, 1, Scalar<T>(0.0));
            setRow(dF, 2, Scalar<T>(0.0));
            setRow(dF, 3, Scalar<T>(0.0));

            // Compute the map derivs
            if (theta == 0) {
                for (int i = 0; i < N; ++i)
                    setRow(dF, 4 + i, B.rTA1(i));
            } else {
                for (int l = 0; l < lmax + 1; ++l)
                    rTA1R.segment(l * l, 2 * l + 1) =
                        B.rTA1.segment(l * l, 2 * l + 1) * W.R[l];
                for (int i = 0; i < N; ++i)
                    setRow(dF, 4 + i, rTA1R(i));
            }

            // The limb darkening derivs are zero, since
            // they don't affect the total flux!
            for (int i = 0; i < lmax; ++i)
                setRow(dF, 4 + N + i, Scalar<T>(0.0));

            // We're done!
            return (B.rTA1 * Ry);

        // Occultation
        } else {

            // Apply limb darkening
            // Note that we do this even if there's no limb
            // darkening set, to get the correct derivs
            // TODO: Cache these operations for fixed theta. Seriously
            A1Ry = B.A1 * Ry;
            limb_darken(A1Ry, p_uy, true);
            LDRy = B.A1Inv * p_uy;
            for (int n = 0; n < nwav; ++n)
                A1dLDdpA1(n) = B.A1Inv * dLDdp(n) * B.A1;

            // Compute the theta deriv of the limb-darkened, rotated map
            // dLDRy / dtheta = A1^-1 . dLDdp . d(A1 . R . y) / dtheta
            //                = A1dLDdpA1 . dR / dtheta . y
            for (int n = 0; n < nwav; ++n) {
                for (int l = 0; l < lmax + 1; ++l)
                    A1dLDdpA1dRdtheta.block(0, l * l, N, 2 * l + 1) =
                        A1dLDdpA1(n).block(0, l * l, N, 2 * l + 1)
                        * W.dRdtheta[l];
                dLDRydtheta.col(n) = A1dLDdpA1dRdtheta * getColumn(y, n);
            }

            // Align occultor with the +y axis
            Scalar<T> xo_b = xo / b;
            Scalar<T> yo_b = yo / b;
            if ((b > 0) && ((xo != 0) || (yo < 0))) {
                W.rotatez(yo_b, xo_b, LDRy, RLDRy);
            } else {
                setOnes(W.cosmt);
                setZero(W.sinmt);
                RLDRy = LDRy;
            }

            // Perform the rotation + change of basis
            ARLDRy = B.A * RLDRy;

            // Compute the sT vector using AutoDiff
            b_grad.value() = b;
            b_grad.derivatives() = Vector<Scalar<T>>::Unit(2, 0);
            ro_grad.value() = ro;
            ro_grad.derivatives() = Vector<Scalar<T>>::Unit(2, 1);
            G_grad.compute(b_grad, ro_grad);

            // Compute the b and ro derivs
            setZero(dFdb);
            setRow(dF, 3, Scalar<T>(0.0));
            for (int i = 0; i < N; i++) {

                // b deriv
                dFdb += G_grad.sT(i).derivatives()(0) * getRow(ARLDRy, i);

                // ro deriv
                setRow(dF, 3, Row<T>(getRow(dF, 3) +
                                     G_grad.sT(i).derivatives()(1) *
                                     getRow(ARLDRy, i)));

                // Store the value of s^T
                G.sT(i) = G_grad.sT(i).value();

            }

            // Solution vector in spherical harmonic basis
            sTA = G.sT * B.A;

            // Compute stuff involving the Rprime rotation matrix
            int m;
            for (int l = 0; l < lmax + 1; ++l) {
                for (int j = 0; j < 2 * l + 1; ++j) {
                    m = j - l;
                    sTAR(l * l + j) = sTA(l * l + j) *
                                      W.cosmt(l * l + j) +
                                      sTA(l * l + 2 * l - j) *
                                      W.sinmt(l * l + j);
                    sTAdRdtheta(l * l + j) = sTA(l * l + 2 * l - j) * m *
                                             W.cosmt(l * l + j) -
                                             sTA(l * l + j) * m *
                                             W.sinmt(l * l + j);
                }
            }

            // Compute the xo and yo derivs using the chain rule
            sTAdRdthetaLDRy_b = dot(sTAdRdtheta, LDRy) / b;
            setRow(dF, 1, Row<T>((xo_b * dFdb) + (yo_b * sTAdRdthetaLDRy_b)));
            setRow(dF, 2, Row<T>((yo_b * dFdb) - (xo_b * sTAdRdthetaLDRy_b)));

            // Compute the theta deriv
            setRow(dF, 0, Row<T>(dot(sTAR, dLDRydtheta) *
                                (pi<Scalar<T>>() / 180.)));

            // Compute the map derivs
            // dF / dy = s^T . A . R' . (A1^-1 . dLDdp . A1) . R
            for (int n = 0; n < nwav; ++n) {
                sTARdLDdpA1 = sTAR * A1dLDdpA1(n);
                if (theta == 0) {
                    for (int i = 0; i < N; ++i)
                        dF(4 + i, n) = sTARdLDdpA1(i);
                } else {
                    for (int l = 0; l < lmax + 1; ++l)
                        sTARdLDdpA1R.segment(l * l, 2 * l + 1) =
                            sTARdLDdpA1.segment(l * l, 2 * l + 1) * W.R[l];
                    for (int i = 0; i < N; ++i)
                        dF(4 + i, n) = sTARdLDdpA1R(i);
                }
            }

            // TODO: Can be sped up
            // Compute the derivs with respect to the limb darkening coeffs
            // dF / du = s^T . A . R' . A1^-1 . dLDdp_u . dp_udu
            for (int n = 0; n < nwav; ++n) {
                dFdp_u = sTAR * B.A1Inv * dLDdp_u(n);
                dFdu = dFdp_u * dp_udu(n);
                dF.block(4 + N, n, lmax, 1) = dFdu.segment(1, lmax).transpose();
            }

            // Dot the result in and we're done
            return G.sT * ARLDRy;

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

#endif
