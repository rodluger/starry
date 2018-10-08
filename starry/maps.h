/**
Defines the surface map class.

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
#include "limbdark.h"
#include "sturm.h"
#include "minimize.h"
#include "numeric.h"

namespace kepler {
    template <class T>
    class Body;
    template <class T>
    class System;
}

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
    using limbdark::GreensLimbDark;
    using limbdark::computeC;
    using limbdark::normC;
    using solver::Power;
    using minimize::Minimizer;

    // Forward-declare some stuff
    template <class T> class Map;

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
    Check that `Map` is instantiated with the right type: Vector<T>.

    */
    template <class T>
    typename std::enable_if<!std::is_base_of<Eigen::EigenBase<T>, T>::value, bool>::type
    checkType(const Map<Vector<T>>& map) {
        if (map.nwav != 1) {
            throw errors::ValueError("Multi-wavelength support is "
                                     "available only for `Matrix` types.");
        } else {
            return true;
        }
    }

    /**
    Check that `Map` is instantiated with the right type: Matrix<T>.

    */
    template <class T>
    typename std::enable_if<!std::is_base_of<Eigen::EigenBase<T>, T>::value, bool>::type
    checkType(const Map<Matrix<T>>& map) {
        if (map.nwav < 1) {
            throw errors::ValueError("Invalid number of wavelength bins.");
        } else {
            return true;
        }
    }

    /**
    The main surface map class.

    */
    template <class T>
    class Map {

        friend class kepler::Body<T>;
        friend class kepler::System<T>;

        public:

            const int lmax;                                                     /**< The highest degree of the map */
            const int N;                                                        /**< The number of map coefficients */
            const int nwav;                                                     /**< The number of wavelengths */

        protected:

            T dF;                                                               /**< Gradient of the flux */
            std::vector<string> dF_names;                                       /**< Names of each of the params in the flux gradient */

            // Sanity checks
            const bool type_valid;                                              /**< Is the type of the Map valid? */

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
            GreensLimbDark<Scalar<T>> L;                                        /**< The occultation integral solver class (optimized for limb darkening) */
            GreensLimbDark<ADScalar<Scalar<T>, 2>> L_grad;                      /**< The occultation integral solver class (opt. for LD) w/ AutoDiff capability */
            Minimizer<T> M;                                                     /**< Map minimization class */
            Scalar<T> tol;                                                      /**< Machine epsilon */
            std::vector<string> dF_orbital_names;                               /**< Names of each of the orbital params in the flux gradient */
            std::vector<string> dF_ylm_names;                                   /**< Names of each of the Ylm params in the flux gradient */
            std::vector<string> dF_ul_names;                                    /**< Names of each of the limb darkening params in the flux gradient */
            int dF_n_ylm;                                                       /**< Number of current Ylm gradients */
            int dF_n_ul;                                                        /**< Number of current limb darkening gradients */

            // Limb darkening
            T u;                                                                /**< The limb darkening coefficients */
            T p_u;                                                              /**< The limb darkening coefficients in the polynomial basis */
            T g_u;                                                              /**< The limb darkening coefficients in the Green's basis */
            T agol_c;                                                           /**< The Agol `c` limb darkening coefficients */
            Row<T> agol_norm;                                                   /**< The Agol normalization */
            int u_deg;                                                          /**< Highest degree set by the user in the limb darkening vector */
            Vector<Matrix<Scalar<T>>> dp_udu;                                   /**< Deriv of limb darkening polynomial w/ respect to limb darkening coeffs */
            Vector<Matrix<Scalar<T>>> dg_udu;                                   /**< Deriv of limb darkening Green's polynomials w/ respect to limb darkening coeffs */
            Vector<Matrix<Scalar<T>>> dagol_cdu;                                /**< Deriv of Agol `c` coeffs w/ respect to the limb darkening ceoffs */
            VectorT<Matrix<Scalar<T>>> dLDdp;                                   /**< Derivative of the limb-darkened polynomial w.r.t `p` */
            VectorT<Matrix<Scalar<T>>> dLDdp_u;                                 /**< Derivative of the limb-darkened polynomial w.r.t `p_u` */
            T p_uy;                                                             /**< The instantaneous limb-darkened map in the polynomial basis */
            Row<T> ld_norm;

            // Temporaries and cache
            Temporary<T> tmp;
            Cache<T> cache;
            bool update_p_u_derivs;
            bool update_c_basis;

            // Private methods
            void update();
            inline void resizeGradient(const int n_ylm, const int n_ul);
            inline void checkDegree();
            inline void updateY();
            inline void updateU();
            inline void limbDarken(const T& poly, T& poly_ld,
                bool gradient=false);
            template <typename U>
            inline void polyBasis(Power<U>& xpow, Power<U>& ypow,
                VectorT<U>& basis);
            inline Row<T> fluxWithGradient(const Scalar<T>& theta_deg,
                const Scalar<T>& xo_,
                const Scalar<T>& yo_,
                const Scalar<T>& ro_);
            inline Row<T> fluxLD(const Scalar<T>& xo_,
                const Scalar<T>& yo_,
                const Scalar<T>& ro_);
            inline Row<T> fluxLDWithGradient(const Scalar<T>& xo_,
                const Scalar<T>& yo_,
                const Scalar<T>& ro_);
            inline Row<T> fluxYlmWithGradient(const Scalar<T>& theta_deg,
                const Scalar<T>& xo_,
                const Scalar<T>& yo_,
                const Scalar<T>& ro_);
            inline Row<T> fluxConstantWithGradient(const Scalar<T>& xo_,
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
                type_valid(checkType(*this)),
                B(lmax),
                W(lmax, nwav, (*this).y, (*this).axis),
                G(lmax),
                G_grad(lmax),
                L(lmax),
                L_grad(lmax),
                M(lmax),
                tol(mach_eps<Scalar<T>>()),
                dp_udu(nwav),
                dg_udu(nwav),
                dagol_cdu(nwav),
                tmp(N, nwav),
                cache() {

                // Populate the map gradient names
                dF_orbital_names.push_back("theta");
                dF_orbital_names.push_back("xo");
                dF_orbital_names.push_back("yo");
                dF_orbital_names.push_back("ro");
                for (int l = 0; l < lmax + 1; l++) {
                    for (int m = -l; m < l + 1; m++) {
                        dF_ylm_names.push_back(string("y"));
                    }
                }
                for (int l = 1; l < lmax + 1; l++) {
                    dF_ul_names.push_back(string("u"));
                }
                dF_n_ul = 0;
                dF_n_ylm = 0;
                resizeGradient(N, lmax);

                // Initialize the map vectors
                axis = yhat<Scalar<T>>();
                y.resize(N, nwav);
                p.resize(N, nwav);
                g.resize(N, nwav);
                u.resize(lmax + 1, nwav);
                agol_c.resize(lmax + 1, nwav);
                resize(agol_norm, 0, nwav);
                p_u.resize(N, nwav);
                g_u.resize(N, nwav);

                // Reset & update the map coeffs
                reset();

            }

            // Housekeeping and I/O
            void reset();
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
            virtual VectorT<Scalar<T>> getR() const;
            virtual VectorT<Scalar<T>> getS() const;
            void setAxis(const UnitVector<Scalar<T>>& axis_);
            UnitVector<Scalar<T>> getAxis() const;
            virtual std::string info();
            inline void resizeGradient();
            const T& getGradient() const;
            const std::vector<std::string>& getGradientNames() const;

            // Rotate the base map
            void rotate(const Scalar<T>&  theta_);

            // Evaluate the intensity at a point
            inline Row<T> operator()(const Scalar<T>& theta_=0,
                const Scalar<T>& x_=0,
                const Scalar<T>& y_=0);

            // Compute the flux
            inline Row<T> flux(const Scalar<T>& theta_=0,
                const Scalar<T>& xo_=0,
                const Scalar<T>& yo_=0,
                const Scalar<T>& ro_=0,
                bool gradient=false,
                bool numerical=false);

            // Is the map physical?
            inline RowBool<T> isPhysical(const Scalar<T>& epsilon=1.e-6,
                const int max_iterations=100);

    };

    /* ---------------- */
    /*   HOUSEKEEPING   */
    /* ---------------- */

    /**
    Check if the total degree of the map is valid

    */
    template <class T>
    inline void Map<T>::checkDegree() {
        if (y_deg + u_deg > lmax) {
            setZero(y);
            setRow(y, 0, Scalar<T>(1.0));
            y_deg = 0;
            setZero(u);
            u_deg = 0;
            setRow(u, 0, Scalar<T>(-1.0));
            update();
            throw errors::ValueError("Degree of the limb-darkened "
                                     "map exceeds `lmax`. All "
                                     "coefficients have been reset.");
        }
    }

    /**
    Update the Ylm map after the coefficients changed
    TODO: This method needs to be made as fast as possible.

    */
    template <class T>
    inline void Map<T>::updateY() {
        // Check the map degree is valid
        checkDegree();

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
    inline void Map<T>::updateU() {
        // Bind references to temporaries for speed
        Row<T>& Y00(tmp.tmpRow[0]);
        Row<T>& rTp_u(tmp.tmpRow[1]);

        // Check the map degree is valid
        checkDegree();

        // Update the limb darkening polynomial map
        p_u = B.U1 * u;

        // Compute the normalization that preserves
        // the disk-integrated intensity
        Y00 = getRow(y, 0);
        rTp_u = dot(B.rT, p_u);
        ld_norm = cwiseQuotient(Y00, rTp_u);

        // Apply the normalization
        p_u = colwiseProduct(p_u, ld_norm);

        // Update the limb darkening Green's map
        g_u = B.A2 * p_u;

        // Set flags
        update_c_basis = true;
        update_p_u_derivs = true;

        // Clear the cache
        cache.clear();
    }


    /**
    Update the two maps after the coefficients changed

    */
    template <class T>
    inline void Map<T>::update() {
        updateY();
        updateU();
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
        update_p_u_derivs = false;
        update_c_basis = false;
        update();
    }

    /**
    Public workaround to resize the gradients
    prior to a flux evaluation, so that we know
    externally how many gradients to expect. This
    is used in the pybind11 interface.

    */
    template <class T>
    inline void Map<T>::resizeGradient() {
        if ((u_deg > 0) && (y_deg == 0))
            resizeGradient(1, lmax);
        else if ((y_deg > 0) && (u_deg == 0))
            resizeGradient(N, 0);
        else if ((y_deg > 0) && (u_deg > 0))
            resizeGradient(N, lmax);
        else
            resizeGradient(1, 0);
    }


    /* ---------------- */
    /*        I/O       */
    /* ---------------- */

    //! Get the gradient of the flux
    template <class T>
    const T& Map<T>::getGradient() const {
        return dF;
    }

    //! Get the names of the gradient params
    template <class T>
    const std::vector<std::string>& Map<T>::getGradientNames() const {
        return dF_names;
    }

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
            // Note that we must implicitly call `updateU()` as well
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
            // Note that we must implicitly call `updateU()` as well
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
            updateU();
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
            updateU();
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
    Resize the gradient vector and set the string vector of gradient names

    */
    template <class T>
    inline void Map<T>::resizeGradient(const int n_ylm, const int n_ul) {

        // Do we need to do anything?
        if ((n_ylm == dF_n_ylm) && (n_ul == dF_n_ul))
            return;

        // Resize the gradient vector
        dF.resize(4 + n_ylm + n_ul, nwav);

        // Reset the vector of names and start from scratch
        dF_names.clear();
        dF_names.reserve(dF_orbital_names.size() + n_ylm + n_ul);
        dF_names.insert(dF_names.end(), dF_orbital_names.begin(),
                        dF_orbital_names.end());
        dF_names.insert(dF_names.end(), dF_ylm_names.begin(),
                        dF_ylm_names.begin() + n_ylm);
        dF_names.insert(dF_names.end(), dF_ul_names.begin(),
                        dF_ul_names.begin() + n_ul);
        dF_n_ylm = n_ylm;
        dF_n_ul = n_ul;

    }

    /**
    Return a human-readable map string

    */
    template <class T>
    std::string Map<T>::info() {
        std::ostringstream os;
        std::string multi;
        if (isMulti(Scalar<T>(0.)))
            multi = "True";
        else
            multi = "False";
        os << "<starry.Map("
           << "lmax=" << lmax << ", "
           << "nwav=" << nwav << ", "
           << "multi=" << multi
           << ")>";
        return std::string(os.str());
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
    inline void Map<T>::limbDarken(const T& poly, T& poly_ld, bool gradient) {
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

    /**
    Check whether the map is physical: the spherical harmonic
    component and the limb darkening components must
    independently be positive semi-definite, and the limb darkening
    component must be a monotonically decreasing function toward
    the limb.

    To ensure positive semi-definiteness...

    To ensure monotonicity, note that the radial profile is

         I = 1 - (1 - mu)^1 u1 - (1 - mu)^2 u2 - ...
           = x^0 c0 + x^1 c1 + x^2 c2 + ...

    where x = (1 - mu), c = -u, c(0) = 1. We want dI/dx < 0
    everywhere, so we require the polynomial

         P = x^0 c1 + 2x^1 c2 + 3x^2 c3 + ...

    to have zero roots in the interval [0, 1]. We use Sturm's
    theorem for this.

    */
    template <class T>
    inline RowBool<T> Map<T>::isPhysical(const Scalar<T>& epsilon,
                                         const int max_iterations) {
        RowBool<T> physical(nwav);
        Row<T> center, limb;
        if (u_deg > 0) {
            center = (*this)(0, 0, 0);
            limb = (*this)(0, 1, 0);
        }

        for (int n = 0; n < nwav; ++n) {

            // 1. Check if the polynomial map is PSD
            if (y_deg == 0) {

                // Trivial case
                Row<T> y00 = getRow(y, 0);
                for (int n = 0; n < nwav; ++n) {
                    setIndex(physical, n, getColumn(y00, n) >= 0);
                }

            } else if (y_deg == 1) {

                // Dipole: analytic
                Row<T> y00 = getRow(y, 0);
                Row<T> y1m1 = getRow(y, 1);
                Row<T> y10 = getRow(y, 2);
                Row<T> y1p1 = getRow(y, 3);
                setIndex(physical, n,
                         getColumn(y1m1, n) * getColumn(y1m1, n) +
                         getColumn(y10, n) * getColumn(y10, n) +
                         getColumn(y1p1, n) * getColumn(y1p1, n)
                         <= getColumn(y00, n) / 3.);

            } else {

                // Higher degrees are solved numerically
                setIndex(physical, n, M.psd(getColumn(p, n),
                                            epsilon, max_iterations));

            }

            // 2. Check if the LD map is PSD and monotonic
            if (u_deg > 0) {

                // First of all, ensure the function is *decreasing*
                // toward the limb
                if (getIndex(center, n) - getIndex(limb, n) < 0) {
                    setIndex(physical, n, false);
                    continue;
                }

                // Sturm's theorem on the intensity to get PSD
                Vector<Scalar<T>> c = -getColumn(u, n).reverse();
                c(c.size() - 1) = 1;
                // Hack: DFM's Sturm routine doesn't behave when
                // the linear term is zero. Not sure why.
                if (c(c.size() - 2) == 0)
                    c(c.size() - 2) = -mach_eps<Scalar<T>>();
                int nroots = sturm::polycountroots(c);
                if (nroots != 0) {
                    setIndex(physical, n, false);
                    continue;
                }

                // Sturm's theorem on the deriv to get monotonicity
                Vector<Scalar<T>> du = getColumn(u, n).segment(1, lmax);
                // Another hack
                if (du(0) == 0)
                    du(0) = mach_eps<Scalar<T>>();
                for (int i = 0; i < lmax; ++i)
                    du(i) *= (i + 1);
                c = -du.reverse();
                nroots = sturm::polycountroots(c);
                if (nroots != 0) {
                    setIndex(physical, n, false);
                    continue;
                }

            }

        }

        return physical;

    }


    /* ------------- */
    /*   INTENSITY   */
    /* ------------- */

    /**
    Compute the polynomial basis at a point; templated for AD capability

    */
    template <class T> template <typename U>
    inline void Map<T>::polyBasis(Power<U>& xpow, Power<U>& ypow,
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
    Evaluate the map at a given (x, y) coordinate

    */
    template <class T>
    inline Row<T> Map<T>::operator()(const Scalar<T>& theta_,
                                     const Scalar<T>& x_,
                                     const Scalar<T>& y_) {

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
                limbDarken(A1Ry, p_uy);
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
        polyBasis(xpow, ypow, pT);

        // Dot the coefficients in to our polynomial map
        return pT * A1Ry;

    }


    /* ------------- */
    /*      FLUX     */
    /* ------------- */

    /**
    Compute the flux during or outside of an occultation
    for the general case of a spherical harmonic map
    with or without limb darkening.

    */
    template <class T>
    inline Row<T> Map<T>::flux(const Scalar<T>& theta_,
                               const Scalar<T>& xo_,
                               const Scalar<T>& yo_,
                               const Scalar<T>& ro_,
                               bool gradient,
                               bool numerical) {

        if (gradient) {
            if (numerical)
                throw errors::NotImplementedError("Numerical gradients of the "
                                                  "flux have not been implemented.");
            // If we're computing the gradient as well,
            // call the specialized functions
            if ((u_deg > 0) && (y_deg == 0))
                return fluxLDWithGradient(xo_, yo_, ro_);
            else if ((y_deg > 0) && (u_deg == 0))
                return fluxYlmWithGradient(theta_, xo_, yo_, ro_);
            else if ((y_deg > 0) && (u_deg > 0))
                return fluxWithGradient(theta_, xo_, yo_, ro_);
            else
                return fluxConstantWithGradient(xo_, yo_, ro_);
        } else if (y_deg == 0) {
            // If only the Y_{0,0} term is set, call the
            // faster method for pure limb-darkening
            return fluxLD(xo_, yo_, ro_);
        }

        // Bind references to temporaries for speed
        Row<T>& result(tmp.tmpRow[0]);
        T& Ry(tmp.tmpT[0]);
        T& A1Ry(tmp.tmpT[1]);
        T& RRy(tmp.tmpT[2]);
        T& ARRy(tmp.tmpT[3]);
        Vector<Scalar<T>>& A1Ryn(tmp.tmpColumnVector[0]);

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
                    limbDarken(A1Ry, p_uy);

                    // Back to the spherical harmonic basis
                    Ry = B.A1Inv * p_uy;

                    // Cache the map
                    cache.oper = cache.FLUX;
                    cache.theta = theta;
                    cache.y = Ry;

                }

            }

            // Compute the flux numerically.
            // NOTE: This is VERY SLOW and used exclusively for debugging!
            if (numerical) {
                const Scalar<T> tol = 1e-5;
                for (int n = 0; n < nwav; ++n) {
                    A1Ryn =  B.A1 * getColumn(Ry, n);
                    setIndex(result, n,
                             numeric::flux(xo, yo, ro, lmax, A1Ryn, tol));
                }
                return result;
            }

            // Rotate the map to align the occultor with the +y axis
            // Change basis to Green's polynomials
            if ((y_deg > 0) && (b > 0) && ((xo != 0) || (yo < 0))) {
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

    NOTE: This uses the fast parameterization from
          Agol & Luger (2018).

    */
    template <class T>
    inline Row<T> Map<T>::fluxLD(const Scalar<T>& xo_,
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

            if ((u_deg <= 2) && (ro < 1)) {

                // Skip the overhead for quadratic limb darkening
                G.quad(b, ro);
                if (u_deg == 0)
                    return G.sT(0) * getRow(g_u, 0);
                else if (u_deg == 1)
                    return G.sT(0) * getRow(g_u, 0) +
                           G.sT(2) * getRow(g_u, 2);
                else
                    return G.sT(0) * getRow(g_u, 0) +
                           G.sT(2) * getRow(g_u, 2) +
                           G.sT(8) * getRow(g_u, 8);

            } else {

                // Compute the Agol S vector
                L.compute(b, ro);

                // Compute the Agol `c` basis
                if (update_c_basis) {
                    for (int n = 0; n < nwav; ++n) {
                        agol_c.col(n) = computeC(getColumn(u, n), dagol_cdu(n));
                        setIndex(agol_norm, n, normC(getColumn(agol_c, n)));
                    }
                    update_c_basis = false;
                }

                // Dot the result in and we're done
                return L.S * colwiseProduct(agol_c, agol_norm);

            }

        }

    }

    /**
    Compute the flux during or outside of an occultation
    for a pure limb-darkened map (Y_{l,m} = 0 for l > 0).
    Also compute the gradient with respect to the orbital
    parameters and the limb darkening coefficients.

    NOTE: This uses the fast parameterization from
          Agol & Luger (2018).

    */
    template <class T>
    inline Row<T> Map<T>::fluxLDWithGradient(const Scalar<T>& xo_,
                                             const Scalar<T>& yo_,
                                             const Scalar<T>& ro_) {

        // Bind references to temporaries for speed
        Row<T>& result(tmp.tmpRow[0]);
        resize(result, 0, nwav);
        Row<T>& dFdb(tmp.tmpRow[1]);
        resize(dFdb, 0, nwav);
        Row<T>& dFdro(tmp.tmpRow[2]);
        resize(dFdro, 0, nwav);
        T& dFdu(tmp.tmpT[0]);
        dFdu.resize(lmax, nwav);
        T& dFdc(tmp.tmpT[1]);
        dFdc.resize(lmax + 1, nwav);
        VectorT<Scalar<T>>& dSdb(tmp.tmpRowVector[0]);
        dSdb.resize(lmax + 1);
        VectorT<Scalar<T>>& dSdro(tmp.tmpRowVector[1]);
        dSdro.resize(lmax + 1);
        ADScalar<Scalar<T>, 2>& b_grad(tmp.tmpADScalar2[0]);
        ADScalar<Scalar<T>, 2>& ro_grad(tmp.tmpADScalar2[1]);

        // Resize the gradients
        resizeGradient(1, lmax);

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
                setRow(dF, 5 + i, Scalar<T>(0.0));

            // Easy: the disk-integrated intensity
            // is just the Y_{0,0} coefficient
            return getRow(y, 0);

        // Occultation
        } else {

            // Compute the Agol `c` basis
            if (update_c_basis) {
                for (int n = 0; n < nwav; ++n) {
                    agol_c.col(n) = computeC(getColumn(u, n), dagol_cdu(n));
                    setIndex(agol_norm, n, normC(getColumn(agol_c, n)));
                }
                update_c_basis = false;
            }

            // Compute the sT vector using AutoDiff
            b_grad.value() = b;
            b_grad.derivatives() = Vector<Scalar<T>>::Unit(2, 0);
            ro_grad.value() = ro;
            ro_grad.derivatives() = Vector<Scalar<T>>::Unit(2, 1);

            // Compute S, dS / db, and dS / dr
            L_grad.compute(b_grad, ro_grad);

            // Store the value of the S vector and its derivatives
            for (int i = 0; i <= lmax; ++i) {
                L.S(i) = L_grad.S(i).value();
                dSdb(i) = L_grad.S(i).derivatives()(0);
                dSdro(i) = L_grad.S(i).derivatives()(1);
            }

            // Compute the value of the flux and its derivatives
            for (int n = 0; n < nwav; ++n) {

                // F, dF / db and dF / dr
                setIndex(result, n, L.S.dot(getColumn(agol_c, n)) *
                                    getIndex(agol_norm, n));
                setIndex(dFdb, n, dSdb.dot(getColumn(agol_c, n)) *
                                  getIndex(agol_norm, n));
                setIndex(dFdro, n, dSdro.dot(getColumn(agol_c, n)) *
                                   getIndex(agol_norm, n));

                // Compute dF / dc
                dFdc.block(0, n, lmax + 1, 1) = L.S.transpose() *
                                                getIndex(agol_norm, n);
                dFdc(0, n) -= getIndex(result, n) * getIndex(agol_norm, n) *
                              pi<Scalar<T>>();
                dFdc(1, n) -= 2.0 * pi<Scalar<T>>() / 3.0 *
                              getIndex(result, n) * getIndex(agol_norm, n);

                // Chain rule to get dF / du
                dFdu.block(0, n, lmax, 1).transpose() =
                    dFdc.block(0, n, lmax + 1, 1).transpose() *
                    dagol_cdu(n).block(0, 1, lmax + 1, lmax);

            }

            // Update the user-facing derivs
            setRow(dF, 1, Row<T>(dFdb * xo / b));
            setRow(dF, 2, Row<T>(dFdb * yo / b));
            setRow(dF, 3, dFdro);
            setRow(dF, 4, cwiseQuotient(result, getRow(y, 0)));
            for (int i = 0; i < lmax; ++i)
                setRow(dF, i + 5, getRow(dFdu, i));

            // Return the flux
            return result;

        }

    }

    /**
    Compute the flux during or outside of an occultation
    for a pure spherical harmonic map (u_{l} = 0 for l > 0).
    Also compute the gradient with respect to the orbital
    parameters and the spherical harmonic map coefficients.

    */
    template <class T>
    inline Row<T> Map<T>::fluxYlmWithGradient(const Scalar<T>& theta_deg,
                                                 const Scalar<T>& xo_,
                                                 const Scalar<T>& yo_,
                                                 const Scalar<T>& ro_) {

        // Bind references to temporaries for speed
        Row<T>& result(tmp.tmpRow[0]);
        Row<T>& dFdb(tmp.tmpRow[1]);
        Row<T>& sTAdRdthetaRy_b(tmp.tmpRow[2]);
        T& Ry(tmp.tmpT[0]);
        T& RRy(tmp.tmpT[1]);
        T& ARRy(tmp.tmpT[2]);
        T& dRdthetay(tmp.tmpT[3]);
        VectorT<Scalar<T>>& sTA(tmp.tmpRowVector[0]);
        VectorT<Scalar<T>>& sTAR(tmp.tmpRowVector[1]);
        sTAR.resize(N);
        VectorT<Scalar<T>>& sTARR(tmp.tmpRowVector[2]);
        sTARR.resize(N);
        VectorT<Scalar<T>>& sTAdRdtheta(tmp.tmpRowVector[3]);
        sTAdRdtheta.resize(N);
        VectorT<Scalar<T>>& rTA1R(tmp.tmpRowVector[4]);
        rTA1R.resize(N);
        ADScalar<Scalar<T>, 2>& b_grad(tmp.tmpADScalar2[0]);
        ADScalar<Scalar<T>, 2>& ro_grad(tmp.tmpADScalar2[1]);

        // Resize the gradients
        resizeGradient(N, 0);

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

            // Compute the theta deriv
            for (int l = 0; l < lmax + 1; l++)
                dRdthetay.block(l * l, 0, 2 * l + 1, nwav) =
                    W.dRdtheta[l] * y.block(l * l, 0, 2 * l + 1, nwav);
            setRow(dF, 0, Row<T>(dot(B.rTA1, dRdthetay) *
                                (pi<Scalar<T>>() / 180.)));

            // The x, y, and r derivs are trivial
            setRow(dF, 1, Scalar<T>(0.0));
            setRow(dF, 2, Scalar<T>(0.0));
            setRow(dF, 3, Scalar<T>(0.0));

            // Compute the map derivs
            if (theta == 0) {
                for (int i = 0; i < N; i++)
                    setRow(dF, 4 + i, B.rTA1(i));
            } else {
                for (int l = 0; l < lmax + 1; l++)
                    rTA1R.segment(l * l, 2 * l + 1) =
                        B.rTA1.segment(l * l, 2 * l + 1) * W.R[l];
                for (int i = 0; i < N; i++)
                    setRow(dF, 4 + i, rTA1R(i));
            }

            // We're done!
            return (B.rTA1 * Ry);

        // Occultation
        } else {

            // Align occultor with the +y axis
            Scalar<T> xo_b = xo / b;
            Scalar<T> yo_b = yo / b;
            if ((b > 0) && ((xo != 0) || (yo < 0))) {
                W.rotatez(yo_b, xo_b, Ry, RRy);
            } else {
                setOnes(W.cosmt);
                setZero(W.sinmt);
                RRy = Ry;
            }

            // Perform the rotation + change of basis
            ARRy = B.A * RRy;

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
                dFdb += G_grad.sT(i).derivatives()(0) * getRow(ARRy, i);

                // ro deriv
                setRow(dF, 3, Row<T>(getRow(dF, 3) +
                                     G_grad.sT(i).derivatives()(1) *
                                     getRow(ARRy, i)));

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
            sTAdRdthetaRy_b = dot(sTAdRdtheta, Ry) / b;
            setRow(dF, 1, Row<T>((xo_b * dFdb) + (yo_b * sTAdRdthetaRy_b)));
            setRow(dF, 2, Row<T>((yo_b * dFdb) - (xo_b * sTAdRdthetaRy_b)));

            // Compute the theta deriv
            for (int l = 0; l < lmax + 1; l++)
                dRdthetay.block(l * l, 0, 2 * l + 1, nwav) =
                    W.dRdtheta[l] * y.block(l * l, 0, 2 * l + 1, nwav);
            setRow(dF, 0, Row<T>(dot(sTAR, dRdthetay) *
                                (pi<Scalar<T>>() / 180.)));

            // Compute the map derivs
            if (theta == 0) {
                for (int i = 0; i < N; i++)
                    setRow(dF, 4 + i, sTAR(i));
            } else {
                for (int l = 0; l < lmax + 1; l++)
                    sTARR.segment(l * l, 2 * l + 1) =
                        sTAR.segment(l * l, 2 * l + 1) * W.R[l];
                for (int i = 0; i < N; i++)
                    setRow(dF, 4 + i, sTARR(i));
            }

            // Dot the result in and we're done
            return G.sT * ARRy;

        }

    }

    /**
    Compute the flux during or outside of an occultation
    for a pure spherical harmonic map (u_{l} = 0 for l > 0).
    Also compute the gradient with respect to the orbital
    parameters and the spherical harmonic map coefficients.

    */
    template <class T>
    inline Row<T> Map<T>::fluxConstantWithGradient(const Scalar<T>& xo_,
                                                   const Scalar<T>& yo_,
                                                   const Scalar<T>& ro_) {

        // Bind references to temporaries for speed
        Row<T>& result(tmp.tmpRow[0]);
        Row<T>& dFdb(tmp.tmpRow[1]);
        T& ARRy(tmp.tmpT[2]);
        ADScalar<Scalar<T>, 2>& b_grad(tmp.tmpADScalar2[0]);
        ADScalar<Scalar<T>, 2>& ro_grad(tmp.tmpADScalar2[1]);

        // Resize the gradients
        resizeGradient(1, 0);

        // Convert to internal type
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

        // No occultation
        if ((b >= 1 + ro) || (ro == 0)) {

            // Theta deriv is zero for constant maps
            setRow(dF, 0, Scalar<T>(0.0));

            // The x, y, and r derivs are trivial
            setRow(dF, 1, Scalar<T>(0.0));
            setRow(dF, 2, Scalar<T>(0.0));
            setRow(dF, 3, Scalar<T>(0.0));

            // Compute the consant coeff deriv
            setRow(dF, 4, Scalar<T>(1.0));

            // We're done!
            return (B.rTA1 * y);

        // Occultation
        } else {

            // Perform the rotation + change of basis
            ARRy = B.A * y;

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
                dFdb += G_grad.sT(i).derivatives()(0) * getRow(ARRy, i);

                // ro deriv
                setRow(dF, 3, Row<T>(getRow(dF, 3) +
                                     G_grad.sT(i).derivatives()(1) *
                                     getRow(ARRy, i)));

                // Store the value of s^T
                G.sT(i) = G_grad.sT(i).value();

            }

            // Compute the theta deriv (trivial)
            setRow(dF, 0, Scalar<T>(0.0));

            // Compute the xo and yo derivs using the chain rule
            setRow(dF, 1, Row<T>(dFdb * xo / b));
            setRow(dF, 2, Row<T>(dFdb * yo / b));

            // Compute the map deriv
            setRow(dF, 4, cwiseQuotient(result, getRow(y, 0)));

            // Dot the result in and we're done
            return G.sT * ARRy;

        }

    }

    /**
    Compute the flux during or outside of an occultation for the
    general case of a limb-darkened spherical harmonic map. Also
    compute the gradient with respect to the orbital parameters,
    the spherical harmonic map coefficients, and the limb darkening
    coefficients.

    */
    template <class T>
    inline Row<T> Map<T>::fluxWithGradient(const Scalar<T>& theta_deg,
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

        // Resize the gradients
        resizeGradient(N, lmax);

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
            limbDarken(A1Ry, p_uy, true);
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

            // Compute the derivative of the LD polynomials
            // with respect to the LD coefficients
            if (update_p_u_derivs) {
                for (int n = 0; n < nwav; ++n) {
                    dp_udu(n) = -getColumn(p_u, n) * B.rTU1;
                    dp_udu(n) /= dot(B.rT, getColumn(p_u, n));
                    dp_udu(n) += B.U1;
                    dp_udu(n) *= getColumn(ld_norm, n);
                    dg_udu(n) = B.A2 * dp_udu(n);
                }
                update_p_u_derivs = false;
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

} // namespace maps

#endif
