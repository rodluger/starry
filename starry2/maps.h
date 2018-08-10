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

    //! Check our `Map` types: catch-all
    template <class T1, class T2, class T3>
    void check_types(int N, int nwav, tag<T1>, tag<T2>, tag<T3>) {
        throw errors::NotImplementedError("The `Map` class is not implemented for this type.");
    }

    //! Check our `Map` types: Matrix<T> specialization
    template <class T>
    void check_types(int N, int nwav, tag<Matrix<T>>, tag<Vector<T>>, tag<VectorT<T>>) {
        // All is good!
    }

    //! Check our `Map` types: Vector<T> specialization
    template <class T>
    void check_types(int N, int nwav, tag<Vector<T>>, tag<T>, tag<T>) {
        // Check that our wavelength dimension size is 1
        if (nwav != 1)
            throw errors::NotImplementedError("Spectral mode is disabled for this `Map` type.");
    }

    //! Check our `Map` types
    template <class T1, class T2, class T3>
    void check_types(int N, int nwav) {
        return check_types(N, nwav, tag<T1>(), tag<T2>(), tag<T3>());
    }

    //! Set a vector map coefficient
    template <class T>
    void setCoeff(Vector<T>& y, int n, const T& coeff) {
        y(n) = coeff;
    }

    //! Set a matrix map coefficient
    template <class T>
    void setCoeff(Matrix<T>& y, int n, const VectorT<T>& coeff) {
        if (coeff.size() != y.cols())
            throw errors::ValueError("Size mismatch in the wavelength dimension.");
        y.row(n) = coeff;
    }

    //! Get a vector map coefficient
    template <class T>
    T getCoeff(const Vector<T>& y, int n) {
        return y(n);
    }

    //! Get a matrix map coefficient
    template <class T>
    VectorT<T> getCoeff(const Matrix<T>& y, int n) {
        return y.row(n);
    }

    //! Get the vector map coefficient at index `n`
    template <class T>
    T getFirstCoeff(const Vector<T>& y, int n) {
        return y(n);
    }

    //! Get the matrix map coefficient at index `(n, 0)`
    template <class T>
    T getFirstCoeff(const Matrix<T>& y, int n) {
        return y(n, 0);
    }

    /**
    The main surface map class.

    */
    template <class MapType, class CoeffType, class CoeffTypeT>
    class Map {

        using T = typename MapType::Scalar;

        public:

            const int lmax;                                                     /**< The highest degree of the map */
            const int N;                                                        /**< The number of map coefficients */
            const int nwav;                                                     /**< The number of wavelengths */
            MapType dI;                                                         /**< Gradient of the intensity */
            std::vector<string> dI_names;                                       /**< Names of each of the params in the intensity gradient */
            MapType dF;                                                         /**< Gradient of the flux */
            std::vector<string> dF_names;                                       /**< Names of each of the params in the flux gradient */

        private:

            MapType y;                                                          /**< The map coefficients in the spherical harmonic basis */
            MapType p;                                                          /**< The map coefficients in the polynomial basis */
            MapType g;                                                          /**< The map coefficients in the Green's basis */
            UnitVector<T> axis;                                                 /**< The axis of rotation for the map */
            Basis<T> B;                                                         /**< Basis transform stuff */
            Wigner<MapType> W;                                                  /**< The class controlling rotations */
            Greens<T> G;                                                        /**< The occultation integral solver class */
            Greens<ADScalar<T, 2>> G_grad;                                      /**< The occultation integral solver class w/ AutoDiff capability */
            T tol;                                                              /**< Machine epsilon */

            // Temporary vectors
            Vector<T> vtmp;                                                     /**< A temporary surface map vector */
            VectorT<T> vTtmp;                                                   /**< A temporary surface map vector */
            VectorT<T> pT;                                                      /**< The polynomial basis vector */
            VectorT<T> pTA1;                                                    /**< Polynomial basis dotted into change of basis matrix */
            ADScalar<T, 2> x0_grad;                                             /**< x position AD type for map evaluation */
            ADScalar<T, 2> y0_grad;                                             /**< y position AD type for map evaluation */
            VectorT<ADScalar<T, 2>> pT_grad;                                    /**< Polynomial basis AD type */
            ADScalar<T, 2> b_grad;                                              /**< Occultor impact parameter AD type for flux evaluation */
            ADScalar<T, 2> ro_grad;                                             /**< Occultor radius AD type for flux evaluation */
            VectorT<ADScalar<T, 2>> sT_grad;                                    /**< Occultation solution vector AD type */
            VectorT<T> sTA;                                                     /**< The solution vector in the sph harm basis */
            VectorT<T> sTAR;                                                    /**< The solution vector in the rotated sph harm basis */
            VectorT<T> sTAdRdtheta;                                             /**< The derivative of `sTAR` with respect to `theta` */
            MapType mtmp;                                                       /**< A temporary surface map vector */
            MapType mtmp2;                                                      /**< A temporary surface map vector */
            MapType Ry;                                                         /**< The rotated spherical harmonic vector */
            MapType dRdthetay;                                                  /**< Derivative of `Ry` with respect to `theta` */
            MapType* ptr_A1Ry;                                                  /**< Pointer to rotated polynomial vector */
            MapType* ptr_Ry;                                                    /**< Pointer to rotated spherical harmonic vector */
            MapType* ptr_RRy;                                                   /**< Pointer to rotated spherical harmonic vector */
            MapType ARRy;                                                       /**< The `ARRy` term in `s^TARRy` */
            CoeffTypeT dFdb;                                                    /**< Gradient of the flux with respect to the impact parameter */

            // Private methods
            template <typename V>
            inline void poly_basis(const V& x0, const V& y0, VectorT<V>& basis);
            inline CoeffType evaluate_with_gradient(const T& theta_deg,
                                                    const T& x0_, const T& y0_);
            inline CoeffType flux_with_gradient(const T& theta_deg,
                                                const T& xo_, const T& yo_,
                                                const T& ro_);

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
                tol(mach_eps<T>()) {

                // Check that the map types are valid & consistent
                check_types<MapType, CoeffType, CoeffTypeT>(N, nwav);

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
                axis = yhat<T>();
                vtmp = Vector<T>::Zero(N);
                vTtmp = VectorT<T>::Zero(N);
                pT = VectorT<T>::Zero(N);
                pTA1 = VectorT<T>::Zero(N);
                x0_grad = 0, Vector<T>::Unit(2, 0);
                y0_grad = 0, Vector<T>::Unit(2, 1);
                pT_grad = VectorT<ADScalar<T, 2>>::Zero(N);
                b_grad = 0, Vector<T>::Unit(2, 0);
                ro_grad = 0, Vector<T>::Unit(2, 1);
                sT_grad = VectorT<ADScalar<T, 2>>::Zero(N);
                sTA = VectorT<T>::Zero(N);
                sTAR = VectorT<T>::Zero(N);
                sTAdRdtheta = VectorT<T>::Zero(N);
                setZero(dI, 3 + N, nwav);
                setZero(dF, 4 + N, nwav);
                setZero(mtmp, N, nwav);
                setZero(mtmp2, N, nwav);
                setZero(Ry, N, nwav);
                setZero(dRdthetay, N, nwav);
                setZero(ARRy, N, nwav);
                setZero(dFdb, N, nwav);

                // Reset & update the map coeffs
                reset();

            }

            // Housekeeping functions
            void update();
            void reset();

            // I/O functions
            void setYlm(int l, int m, const CoeffTypeT& coeff);
            CoeffTypeT getYlm(int l, int m) const;
            void setAxis(const UnitVector<T>& axis_);
            UnitVector<T> getAxis() const;
            MapType getY() const;
            void setY(const MapType& y_);
            MapType getP() const;
            MapType getG() const;
            VectorT<T> getR() const;
            VectorT<T> getS() const;
            std::string __repr__();

            // Rotate the base map
            void rotate(const T& theta_);

            // Evaluate the intensity at a point
            inline CoeffType evaluate(const T& theta_=0, const T& x_=0,
                                      const T& y_=0, bool gradient=false);

            // Compute the flux
            inline CoeffType flux(const T& theta_=0, const T& xo_=0,
                                  const T& yo_=0, const T& ro_=0,
                                  bool gradient=false);

    };

    /* ---------------- */
    /*   HOUSEKEEPING   */
    /* ---------------- */


    /**
    Update the maps after the coefficients changed

    */
    template <class MapType, class CoeffType, class CoeffTypeT>
    void Map<MapType, CoeffType, CoeffTypeT>::update() {

        // Update the polynomial and Green's map coefficients
        p = B.A1 * y;
        g = B.A * y;

        // Update the rotation matrix
        W.update();

    }

    /**
    Reset the map

    */
    template <class MapType, class CoeffType, class CoeffTypeT>
    void Map<MapType, CoeffType, CoeffTypeT>::reset() {
        y.setZero(N, nwav);
        axis = yhat<typename MapType::Scalar>();
        update();
    }


    /* ---------------- */
    /*        I/O       */
    /* ---------------- */


    /**
    Set the (l, m) spherical harmonic coefficient

    */
    template <class MapType, class CoeffType, class CoeffTypeT>
    void Map<MapType, CoeffType, CoeffTypeT>::setYlm(int l, int m, const CoeffTypeT& coeff) {
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
    template <class MapType, class CoeffType, class CoeffTypeT>
    CoeffTypeT Map<MapType, CoeffType, CoeffTypeT>::getYlm(int l, int m) const {
        if ((0 <= l) && (l <= lmax) && (-l <= m) && (m <= l)) {
            return getCoeff(y, l * l + l + m);
        } else
            throw errors::IndexError("Invalid value for `l` and/or `m`.");
    }

    /**
    Set the axis

    */
    template <class MapType, class CoeffType, class CoeffTypeT>
    void Map<MapType, CoeffType, CoeffTypeT>::setAxis(const UnitVector<T>& axis_) {

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
    template <class MapType, class CoeffType, class CoeffTypeT>
    UnitVector<typename MapType::Scalar> Map<MapType, CoeffType, CoeffTypeT>::getAxis() const {
        return axis;
    }

    /**
    Get the spherical harmonic vector

    */
    template <class MapType, class CoeffType, class CoeffTypeT>
    MapType Map<MapType, CoeffType, CoeffTypeT>::getY() const {
        return y;
    }

    /**
    Set the spherical harmonic vector

    */
    template <class MapType, class CoeffType, class CoeffTypeT>
    void Map<MapType, CoeffType, CoeffTypeT>::setY(const MapType& y_) {
        if ((y_.rows() == y.rows()) && (y_.cols() == y.cols())) {
            y = y_;
            update();
        } else {
            throw errors::ValueError("Dimension mismatch in `y`.");
        }
    }

    /**
    Get the polynomial vector

    */
    template <class MapType, class CoeffType, class CoeffTypeT>
    MapType Map<MapType, CoeffType, CoeffTypeT>::getP() const {
        return p;
    }

    /**
    Get the Green's vector

    */
    template <class MapType, class CoeffType, class CoeffTypeT>
    MapType Map<MapType, CoeffType, CoeffTypeT>::getG() const {
        return g;
    }

    /**
    Get the rotation solution vector

    */
    template <class MapType, class CoeffType, class CoeffTypeT>
    VectorT<typename MapType::Scalar> Map<MapType, CoeffType, CoeffTypeT>::getR() const {
        return B.rT;
    }

    /**
    Get the occultation solution vector

    */
    template <class MapType, class CoeffType, class CoeffTypeT>
    VectorT<typename MapType::Scalar> Map<MapType, CoeffType, CoeffTypeT>::getS() const {
        return G.sT;
    }

    /**
    Return a human-readable map string

    */
    template <class MapType, class CoeffType, class CoeffTypeT>
    std::string Map<MapType, CoeffType, CoeffTypeT>::__repr__() {
        int n = 0;
        int nterms = 0;
        char buf[30];
        std::ostringstream os;
        os << "<STARRY Map: ";
        for (int l = 0; l < lmax + 1; l++) {
            for (int m = -l; m < l + 1; m++) {
                if (abs(getFirstCoeff(y, n)) > 10 * mach_eps<T>()){
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
                    } else if (fmod(abs(getFirstCoeff(y, n)), 1.0) < 10 * mach_eps<T>()) {
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
    template <class MapType, class CoeffType, class CoeffTypeT>
    void Map<MapType, CoeffType, CoeffTypeT>::rotate(const T& theta_) {
        using T = typename MapType::Scalar;
        T theta = theta_ * (pi<T>() / 180.);
        W.rotate(cos(theta), sin(theta), y);
        update();
    }


    /* ------------- */
    /*   INTENSITY   */
    /* ------------- */

    /**
    Compute the polynomial basis at a point; templated for AD capability

    */
    template <class MapType, class CoeffType, class CoeffTypeT> template <typename U>
    inline void Map<MapType, CoeffType, CoeffTypeT>::poly_basis(const U& x0,
                                                                const U& y0,
                                                                VectorT<U>& basis) {
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
    template <class MapType, class CoeffType, class CoeffTypeT>
    inline CoeffType Map<MapType, CoeffType, CoeffTypeT>::evaluate(const T& theta_,
                                                                   const T& x_,
                                                                   const T& y_,
                                                                   bool gradient) {

        using T = typename MapType::Scalar;

        // If we're computing the gradient as well,
        // call the specialized function
        // TODO
        // if (gradient)
        //    return evaluate_with_gradient(theta_, x_, y_);

        // Convert to internal types
        T x0 = x_;
        T y0 = y_;
        T theta = theta_ * (pi<T>() / 180.);

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
            CoeffType res;
            setZero(res, N, nwav);
            return res * NAN;
        }

        // Compute the polynomial basis
        poly_basis(x0, y0, pT);

        // Dot the coefficients in to our polynomial map
        return pT * (*ptr_A1Ry);

    }

    /**
    Evaluate the map at a given (x0, y0) coordinate and compute the gradient

    */
    /* TODO
    template <class T, class MapType, class CoeffType, class CoeffTypeT>
    inline CoeffType<T> Map<T, MapType, CoeffType, CoeffTypeT>::evaluate_with_gradient(const T& theta_,
                                                    const T& x_,
                                                    const T& y_) {

        // Convert to internal type
        T x0 = x_;
        T y0 = y_;
        T theta = theta_ * (pi<T>() / 180.);

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
            dI = MapType_Zero * NAN;
            return CoeffType_Zero * NAN;
        }

        // Compute the polynomial basis and its x and y derivs
        x0_grad.value() = x0;
        y0_grad.value() = y0;
        poly_basis(x0_grad, y0_grad, pT_grad);
        dI.row(1) = VectorT<T>::Constant(nwav, 0);
        dI.row(2) = VectorT<T>::Constant(nwav, 0);
        for (int i = 0; i < N; i++) {
            dI.row(1) += pT_grad(i).derivatives()(0) * (*ptr_A1Ry).row(i);
            dI.row(2) += pT_grad(i).derivatives()(1) * (*ptr_A1Ry).row(i);
            pT(i) = pT_grad(i).value();
        }

        // Compute the map derivs
        pTA1 = pT * B.A1;
        if (theta == 0) {
            for (int i = 0; i < N; i++)
                dI.row(3 + i) = VectorT<T>::Constant(nwav, pTA1(i));
        } else {
            for (int l = 0; l < lmax + 1; l++)
                vtmp.segment(l * l, 2 * l + 1) =
                    pTA1.segment(l * l, 2 * l + 1) * W.R[l];
            for (int i = 0; i < N; i++)
                dI.row(3 + i) = VectorT<T>::Constant(nwav, vtmp(i));
        }

        // Compute the theta deriv
        for (int l = 0; l < lmax + 1; l++)
            dRdthetay.block(l * l, 0, 2 * l + 1, nwav) =
                W.dRdtheta[l] * y.block(l * l, 0, 2 * l + 1, nwav);
        dI.row(0) = pTA1 * dRdthetay * (pi<T>() / 180.);

        // Dot the coefficients in to our polynomial map
        return pT * (*ptr_A1Ry);

    }
    */

    /* ------------- */
    /*      FLUX     */
    /* ------------- */


    /**
    Compute the flux during or outside of an occultation
    */
    template <class MapType, class CoeffType, class CoeffTypeT>
    inline CoeffType Map<MapType, CoeffType, CoeffTypeT>::flux(const T& theta_,
                                                               const T& xo_,
                                                               const T& yo_,
                                                               const T& ro_,
                                                               bool gradient) {

        using T = typename MapType::Scalar;

        // If we're computing the gradient as well,
        // call the specialized function
        // TODO
        //if (gradient)
        //    return flux_with_gradient(theta_, xo_, yo_, ro_);

        // Convert to internal types
        T xo = xo_;
        T yo = yo_;
        T ro = ro_;
        T theta = theta_ * (pi<T>() / 180.);

        // Impact parameter
        T b = sqrt(xo * xo + yo * yo);

        // Check for complete occultation
        if (b <= ro - 1) {
            CoeffType res;
            setZero(res, N, nwav);
            return res;
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

    /*
    TODO
    template <class T, class MapType, class CoeffType, class CoeffTypeT>
    inline Vector<T> Map<T, MapType, CoeffType, CoeffTypeT>::flux_with_gradient(const T& theta_deg,
                                                const T& xo_,
                                                const T& yo_, const T& ro_) {

        // Convert to internal type
        T xo = xo_;
        T yo = yo_;
        T ro = ro_;
        T theta = theta_deg * (pi<T>() / 180.);

        // Impact parameter
        T b = sqrt(xo * xo + yo * yo);

        // Check for complete occultation
        if (b <= ro - 1) {
            dF = MapType::Constant(N, nwav, NAN);
            return Vector<T>::Constant(nwav, 0.0);
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
            dF.row(0) = B.rTA1 * dRdthetay * (pi<T>() / 180.);

            // The x, y, and r derivs are trivial
            dF.row(1) = Vector<T>::Zero(nwav);
            dF.row(2) = Vector<T>::Zero(nwav);
            dF.row(3) = Vector<T>::Zero(nwav);

            // Compute the map derivs
            pTA1 = pT * B.A1;
            if (theta == 0) {
                for (int i = 0; i < N; i++)
                    dF.row(4 + i) = VectorT<T>::Constant(nwav, B.rTA1(i));
            } else {
                for (int l = 0; l < lmax + 1; l++)
                    vtmp.segment(l * l, 2 * l + 1) =
                        B.rTA1.segment(l * l, 2 * l + 1) * W.R[l];
                for (int i = 0; i < N; i++)
                    dF.row(4 + i) = VectorT<T>::Constant(nwav, vtmp(i));
            }

            // We're done!
            return (B.rTA1 * (*ptr_Ry));

        // Occultation
        } else {

            // Align occultor with the +y axis
            T xo_b = xo / b;
            T yo_b = yo / b;
            if ((b > 0) && ((xo != 0) || (yo < 0))) {
                W.rotatez(yo_b, xo_b, *ptr_Ry, mtmp2);
                ptr_RRy = &mtmp2;
            } else {
                W.cosmt = Vector<T>::Constant(N, 1.0);
                W.sinmt = Vector<T>::Constant(N, 0.0);
                ptr_RRy = ptr_Ry;
            }

            // Perform the rotation + change of basis
            ARRy = B.A * (*ptr_RRy);

            // Compute the sT vector using AutoDiff
            b_grad.value() = b;
            ro_grad.value() = ro;
            G_grad.compute(b_grad, ro_grad);

            // Compute the b and ro derivs
            dFdb = VectorT<T>::Constant(nwav, 0);
            dF.row(3) = VectorT<T>::Constant(nwav, 0);
            for (int i = 0; i < N; i++) {

                // b deriv
                dFdb += G_grad.sT(i).derivatives()(0) * ARRy.row(i);

                // ro deriv
                dF.row(3) += G_grad.sT(i).derivatives()(1) * ARRy.row(i);

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
            vTtmp = (sTAdRdtheta * (*ptr_Ry)) / b;
            dF.row(1) = xo_b * dFdb + yo_b * vTtmp;
            dF.row(2) = yo_b * dFdb - xo_b * vTtmp;

            // Compute the theta deriv
            for (int l = 0; l < lmax + 1; l++)
                dRdthetay.block(l * l, 0, 2 * l + 1, nwav) =
                    W.dRdtheta[l] * y.block(l * l, 0, 2 * l + 1, nwav);
            dF.row(0) = sTAR * dRdthetay * (pi<T>() / 180.);

            // Compute the map derivs
            if (theta == 0) {
                for (int i = 0; i < N; i++)
                    dF.row(4 + i) = VectorT<T>::Constant(nwav, sTAR(i));
            } else {
                for (int l = 0; l < lmax + 1; l++)
                    vtmp.segment(l * l, 2 * l + 1) =
                        sTAR.segment(l * l, 2 * l + 1) * W.R[l];
                for (int i = 0; i < N; i++)
                    dF.row(4 + i) = VectorT<T>::Constant(nwav, vtmp(i));
            }

            // Dot the result in and we're done
            return G.sT * ARRy;

        }

    }
    */

}; // namespace maps

#endif
