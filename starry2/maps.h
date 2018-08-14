/**
Defines the surface map class.

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
    using std::string;
    using std::to_string;
    using rotation::Wigner;
    using basis::Basis;
    using solver::Greens;

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
            UnitVector<Scalar<T>> axis;                                         /**< The axis of rotation for the map */
            Basis<Scalar<T>> B;                                                 /**< Basis transform stuff */
            Wigner<T> W;                                                        /**< The class controlling rotations */
            Greens<Scalar<T>> G;                                                /**< The occultation integral solver class */
            Greens<ADScalar<Scalar<T>, 2>> G_grad;                              /**< The occultation integral solver class w/ AutoDiff capability */
            Scalar<T> tol;                                                      /**< Machine epsilon */

            // Temporary vectors
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
            T Ry;                                                               /**< The rotated spherical harmonic vector */
            T dRdthetay;                                                        /**< Derivative of `Ry` with respect to `theta` */
            T* ptr_A1Ry;                                                        /**< Pointer to rotated polynomial vector */
            T* ptr_Ry;                                                          /**< Pointer to rotated spherical harmonic vector */
            T* ptr_RRy;                                                         /**< Pointer to rotated spherical harmonic vector */
            T ARRy;                                                             /**< The `ARRy` term in `s^TARRy` */
            Row<T> dFdb;                                                        /**< Gradient of the flux with respect to the impact parameter */
            Row<T> rtmp;                                                        /**< A temporary surface map row vector */
            Column<T> ctmp;                                                     /**< A temporary surface map col vector */

            // Private methods
            template <typename V>
            inline void poly_basis(const V& x0, const V& y0, VectorT<V>& basis);
            inline Column<T> evaluate_with_gradient(const Scalar<T>& theta_deg,
                                                    const Scalar<T>& x_,
                                                    const Scalar<T>& y_);
            inline Column<T> flux_with_gradient(const Scalar<T>& theta_deg,
                                                const Scalar<T>& xo_,
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
                tol(mach_eps<Scalar<T>>()) {

                // Populate the gradient names
                for (int l = 0; l < lmax + 1; l++) {
                    for (int m = -l; m < l + 1; m++) {
                        dI_names.push_back(string("Y_{" + to_string(l) +
                                           "," + to_string(m) + "}"));
                        dF_names.push_back(string("Y_{" + to_string(l) +
                                           "," + to_string(m) + "}"));
                    }
                }

                // Initialize all the vectors
                setZero(y, N, nwav);
                setZero(u, lmax + 1, nwav);
                axis = yhat<Scalar<T>>();
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
                setZero(dI, 3 + N, nwav);
                setZero(dF, 4 + N, nwav);
                setZero(mtmp, N, nwav);
                setZero(mtmp2, N, nwav);
                setZero(Ry, N, nwav);
                setZero(dRdthetay, N, nwav);
                setZero(ARRy, N, nwav);
                setZero(dFdb, N, nwav);
                setZero(rtmp, N, nwav);
                setZero(ctmp, N, nwav);

                // Reset & update the map coeffs
                reset();

            }

            // Housekeeping functions
            void update();
            void reset();

            // I/O functions
            void setYlm(int l, int m, const Row<T>& coeff);
            Row<T> getYlm(int l, int m) const;
            void setUl(int l, const Row<T>& coeff);
            Row<T> getUl(int l) const;
            void setAxis(const UnitVector<Scalar<T>>& axis_);
            UnitVector<Scalar<T>> getAxis() const;
            T getY() const;
            void setY(const T& y_);
            T getU() const;
            void setU(const T& u_);
            T getP() const;
            T getG() const;
            VectorT<Scalar<T>> getR() const;
            VectorT<Scalar<T>> getS() const;
            std::string __repr__();

            // Rotate the base map
            void rotate(const Scalar<T>&  theta_);

            // Evaluate the intensity at a point
            inline Column<T> evaluate(const Scalar<T>& theta_=0,
                                      const Scalar<T>& x_=0,
                                      const Scalar<T>& y_=0,
                                      bool gradient=false);

            // Compute the flux
            inline Column<T> flux(const Scalar<T>& theta_=0,
                                  const Scalar<T>& xo_=0,
                                  const Scalar<T>& yo_=0,
                                  const Scalar<T>& ro_=0,
                                  bool gradient=false);

    };

    /* ---------------- */
    /*   HOUSEKEEPING   */
    /* ---------------- */


    /**
    Update the maps after the coefficients changed

    */
    template <class T>
    void Map<T>::update() {

        // Update the polynomial and Green's map coefficients
        p = B.A1 * y;
        g = B.A * y;

        // Update the rotation matrix
        W.update();

        // Update the limb darkening polynomial map
        Scalar<T> norm = 0.0;
        for (int l = 0; l < lmax + 1; ++l)
            norm -= 2.0 * u(l) / ((l + 1) * (l + 2));
        norm *= pi<Scalar<T>>();
        mtmp = B.U * u;
        setZero(mtmp2, N, nwav);
        for (int l = 0; l < lmax + 1; ++l)
            mtmp2(l * (l + 1)) = mtmp(l) / norm;
        p_u = B.A1 * mtmp2;

    }

    /**
    Reset the map

    */
    template <class T>
    void Map<T>::reset() {
        y.setZero(N, nwav);
        u.setZero(lmax + 1, nwav);
        setRow(u, 0, Scalar<T>(-1.0));
        axis = yhat<Scalar<T>>();
        update();
    }


    /* ---------------- */
    /*        I/O       */
    /* ---------------- */


    /**
    Set the (l, m) spherical harmonic coefficient

    */
    template <class T>
    void Map<T>::setYlm(int l, int m, const Row<T>& coeff) {
        if ((0 <= l) && (l <= lmax) && (-l <= m) && (m <= l)) {
            int n = l * l + l + m;
            setCoeff(y, n, coeff);
            update();
        } else {
            throw errors::IndexError("Invalid value for `l` and/or `m`.");
        }
    }

    /**
    Get the (l, m) spherical harmonic coefficient

    */
    template <class T>
    Row<T> Map<T>::getYlm(int l, int m) const {
        if ((0 <= l) && (l <= lmax) && (-l <= m) && (m <= l)) {
            return getCoeff(y, l * l + l + m);
        } else
            throw errors::IndexError("Invalid value for `l` and/or `m`.");
    }

    /**
    Set the `l`th limb darkening coefficient

    */
    template <class T>
    void Map<T>::setUl(int l, const Row<T>& coeff) {
        if ((1 <= l) && (l <= lmax)) {
            setCoeff(u, l, coeff);
            update();
        } else {
            throw errors::IndexError("Invalid value for `l`.");
        }
    }

    /**
    Get the `l`th limb darkening coefficient

    */
    template <class T>
    Row<T> Map<T>::getUl(int l) const {
        if ((1 <= l) && (l <= lmax)) {
            return getCoeff(u,l);
        } else
            throw errors::IndexError("Invalid value for `l`.");
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

    }

    /**
    Return a copy of the axis

    */
    template <class T>
    UnitVector<Scalar<T>> Map<T>::getAxis() const {
        return axis;
    }

    /**
    Get the spherical harmonic vector

    */
    template <class T>
    T Map<T>::getY() const {
        return y;
    }

    /**
    Set the spherical harmonic vector

    */
    template <class T>
    void Map<T>::setY(const T& y_) {
        if ((y_.rows() == y.rows()) && (y_.cols() == y.cols())) {
            y = y_;
            update();
        } else {
            throw errors::ValueError("Dimension mismatch in `y`.");
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
    Set the limb darkening vector

    */
    template <class T>
    void Map<T>::setU(const T& u_) {
        if ((u_.rows() == u.rows() - 1) && (u_.cols() == u.cols())) {
            u.block(1, 0, lmax, nwav) = u_;
            update();
        } else {
            throw errors::ValueError("Dimension mismatch in `u`.");
        }
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
    Return a human-readable map string

    TODO: Show spectral and limb-darkening information.

    */
    template <class T>
    std::string Map<T>::__repr__() {
        int n = 0;
        int nterms = 0;
        char buf[30];
        std::ostringstream os;
        os << "<STARRY Map: ";
        for (int l = 0; l < lmax + 1; l++) {
            for (int m = -l; m < l + 1; m++) {
                if (abs(getFirstCoeff(y, n)) > 10 * tol){
                    // Separator
                    if ((nterms > 0) && (getFirstCoeff(y, n) > 0)) {
                        os << " + ";
                    } else if ((nterms > 0) && (getFirstCoeff(y, n) < 0)){
                        os << " - ";
                    } else if ((nterms == 0) && (getFirstCoeff(y, n) < 0)){
                        os << "-";
                    }
                    // Term
                    if ((getFirstCoeff(y, n) == 1) || (getFirstCoeff(y, n) == -1)) {
                        sprintf(buf, "Y_{%d,%d}", l, m);
                        os << buf;
                    } else if (fmod(abs(getFirstCoeff(y, n)), 1.0) < 10 * tol) {
                        sprintf(buf, "%d Y_{%d,%d}", (int)abs(getFirstCoeff(y, n)), l, m);
                        os << buf;
                    } else if (fmod(abs(getFirstCoeff(y, n)), 1.0) >= 0.01) {
                        sprintf(buf, "%.2f Y_{%d,%d}", double(abs(getFirstCoeff(y, n))), l, m);
                        os << buf;
                    } else {
                        sprintf(buf, "%.2e Y_{%d,%d}", double(abs(getFirstCoeff(y, n))), l, m);
                        os << buf;
                    }
                    nterms++;
                }
                n++;
            }
        }
        if (nterms == 0)
            os << "Null";
        os << ">";
        return std::string(os.str());
    }


    /* ------------- */
    /*   ROTATIONS   */
    /* ------------- */


    /**
    Rotate the base map in-place given `theta` in **degrees**

    */
    template <class T>
    void Map<T>::rotate(const Scalar<T>& theta_) {
        Scalar<T> theta = theta_ * (pi<Scalar<T>>() / 180.);
        W.rotate(cos(theta), sin(theta), y);
        update();
    }


    /* ------------- */
    /*   INTENSITY   */
    /* ------------- */

    /**
    Compute the polynomial basis at a point; templated for AD capability

    */
    template <class T> template <typename U>
    inline void Map<T>::poly_basis(const U& x0, const U& y0, VectorT<U>& basis) {
        int l, m, mu, nu, n = 0;
        U z0 = sqrt(1.0 - x0 * x0 - y0 * y0);
        for (l=0; l<lmax+1; l++) {
            for (m=-l; m<l+1; m++) {
                mu = l - m;
                nu = l + m;
                if ((nu % 2) == 0) {
                    if ((mu > 0) && (nu > 0))
                        basis(n) = pow(x0, mu / 2) * pow(y0, nu / 2);
                    else if (mu > 0)
                        basis(n) = pow(x0, mu / 2);
                    else if (nu > 0)
                        basis(n) = pow(y0, nu / 2);
                    else
                        basis(n) = 1;
                } else {
                    if ((mu > 1) && (nu > 1))
                        basis(n) = pow(x0, (mu - 1) / 2) *
                                       pow(y0, (nu - 1) / 2) * z0;
                    else if (mu > 1)
                        basis(n) = pow(x0, (mu - 1) / 2) * z0;
                    else if (nu > 1)
                        basis(n) = pow(y0, (nu - 1) / 2) * z0;
                    else
                        basis(n) = z0;
                }
                n++;
            }
        }
    }

    /**
    Evaluate the map at a given (x0, y0) coordinate

    */
    template <class T>
    inline Column<T> Map<T>::evaluate(const Scalar<T>& theta_,
                                      const Scalar<T>& x_,
                                      const Scalar<T>& y_,
                                      bool gradient) {

        // If we're computing the gradient as well,
        // call the specialized function
        if (gradient)
            return evaluate_with_gradient(theta_, x_, y_);

        // Convert to internal types
        Scalar<T> x0 = x_;
        Scalar<T> y0 = y_;
        Scalar<T> theta = theta_ * (pi<Scalar<T>>() / 180.);

        // Rotate the map into view
        if (theta == 0) {
            ptr_A1Ry = &p;
        } else {
            W.rotate(cos(theta), sin(theta), Ry);
            mtmp = B.A1 * Ry;
            ptr_A1Ry = &mtmp;
        }

        // Check if outside the sphere
        if (x0 * x0 + y0 * y0 > 1.0) {
            setZero(ctmp, N, nwav);
            return ctmp * NAN;
        }

        // Compute the polynomial basis
        poly_basis(x0, y0, pT);

        // Dot the coefficients in to our polynomial map
        return pT * (*ptr_A1Ry);

    }

    /**
    Evaluate the map at a given (x0, y0) coordinate and compute the gradient

    */
    template <class T>
    inline Column<T> Map<T>::evaluate_with_gradient(const Scalar<T>& theta_,
                                                    const Scalar<T>& x_,
                                                    const Scalar<T>& y_) {

        // Convert to internal type
        Scalar<T> x0 = x_;
        Scalar<T> y0 = y_;
        Scalar<T> theta = theta_ * (pi<Scalar<T>>() / 180.);

        // Rotate the map into view
        W.compute(cos(theta), sin(theta));
        if (theta == 0) {
            ptr_A1Ry = &p;
        } else {
            for (int l = 0; l < lmax + 1; l++)
                Ry.block(l * l, 0, 2 * l + 1, nwav) =
                    W.R[l] * y.block(l * l, 0, 2 * l + 1, nwav);
            mtmp = B.A1 * Ry;
            ptr_A1Ry = &mtmp;
        }

        // Check if outside the sphere
        if (x0 * x0 + y0 * y0 > 1.0) {
            setZero(dI, 3 + N, nwav);
            dI = dI * NAN;
            setZero(ctmp, N, nwav);
            return ctmp * NAN;
        }

        // Compute the polynomial basis and its x and y derivs
        x0_grad.value() = x0;
        y0_grad.value() = y0;
        poly_basis(x0_grad, y0_grad, pT_grad);
        setRow(dI, 1, Scalar<T>(0.0));
        setRow(dI, 2, Scalar<T>(0.0));
        for (int i = 0; i < N; i++) {
            setRow(dI, 1, Row<T>(getRow(dI, 1) +
                                 pT_grad(i).derivatives()(0) * getRow(*ptr_A1Ry, i)));
            setRow(dI, 2, Row<T>(getRow(dI, 2) +
                                 pT_grad(i).derivatives()(1) * getRow(*ptr_A1Ry, i)));
            pT(i) = pT_grad(i).value();
        }

        // Compute the map derivs
        pTA1 = pT * B.A1;
        if (theta == 0) {
            for (int i = 0; i < N; i++)
                setRow(dI, 3 + i, pTA1(i));
        } else {
            for (int l = 0; l < lmax + 1; l++)
                vtmp.segment(l * l, 2 * l + 1) =
                    pTA1.segment(l * l, 2 * l + 1) * W.R[l];
            for (int i = 0; i < N; i++)
                setRow(dI, 3 + i, vtmp(i));
        }

        // Compute the theta deriv
        for (int l = 0; l < lmax + 1; l++)
            dRdthetay.block(l * l, 0, 2 * l + 1, nwav) =
                W.dRdtheta[l] * y.block(l * l, 0, 2 * l + 1, nwav);
        setRow(dI, 0, Row<T>(dot(pTA1, dRdthetay) * (pi<Scalar<T>>() / 180.)));

        // Dot the coefficients in to our polynomial map
        return pT * (*ptr_A1Ry);

    }


    /* ------------- */
    /*      FLUX     */
    /* ------------- */


    /**
    Compute the flux during or outside of an occultation

    */
    template <class T>
    inline Column<T> Map<T>::flux(const Scalar<T>& theta_,
                                  const Scalar<T>& xo_,
                                  const Scalar<T>& yo_,
                                  const Scalar<T>& ro_,
                                  bool gradient) {

        // If we're computing the gradient as well,
        // call the specialized function
        if (gradient)
            return flux_with_gradient(theta_, xo_, yo_, ro_);

        // Convert to internal types
        Scalar<T> xo = xo_;
        Scalar<T> yo = yo_;
        Scalar<T> ro = ro_;
        Scalar<T> theta = theta_ * (pi<Scalar<T>>() / 180.);

        // Impact parameter
        Scalar<T> b = sqrt(xo * xo + yo * yo);

        // Check for complete occultation
        if (b <= ro - 1) {
            setZero(ctmp, N, nwav);
            return ctmp;
        }

        // Rotate the map into view
        if (theta == 0) {
            ptr_Ry = &y;
        } else {
            W.rotate(cos(theta), sin(theta), Ry);
            mtmp = Ry;
            ptr_Ry = &mtmp;
        }

        // No occultation
        if ((b >= 1 + ro) || (ro == 0)) {

            // This is very easy!
            return (B.rTA1 * (*ptr_Ry));

        // Occultation
        } else {

            // Align occultor with the +y axis
            if ((b > 0) && ((xo != 0) || (yo < 0))) {
                W.rotatez(yo / b, xo / b, *ptr_Ry, mtmp);
                ptr_Ry = &mtmp;
            }

            // Perform the rotation + change of basis
            ARRy = B.A * (*ptr_Ry);

            // Compute the sT vector (sparsely)
            for (int n = 0; n < N; ++n) {
                for (int i = 0; i < nwav; ++i) {
                    G.skip(n) = true;
                    if (abs(ARRy(n, i)) > tol) {
                        G.skip(n) = false;
                        break;
                    }
                }
            }
            G.compute(b, ro);

            // Dot the result in and we're done
            return G.sT * ARRy;

        }

    }

    /**
    Compute the flux during or outside of an occultation and its gradient

    */

    template <class T>
    inline Column<T> Map<T>::flux_with_gradient(const Scalar<T>& theta_deg,
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
            setZero(dF, N, nwav);
            setZero(ctmp, N, nwav);
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
            setZero(dFdb, N, nwav);
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

}; // namespace maps

#endif
