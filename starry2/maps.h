/**
Defines the surface map class.

*/

#ifndef _STARRY_MAPS_H_
#define _STARRY_MAPS_H_

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include "constants.h"
#include "rotation.h"
#include "basis.h"
#include "errors.h"
#include "utils.h"

namespace maps {

    using std::abs;
    using std::max;
    using std::string;
    using rotation::Wigner;
    using rotation::computeR;

    // Constant matrices/vectors
    template <class T>
    class ConstantMatrices {

        public:

            const int lmax;
            Eigen::SparseMatrix<T> A1;
            Eigen::SparseMatrix<T> A;
            VectorT<T> rTA1;
            VectorT<T> rT;
            Matrix<T> U;

            // Constructor: compute the matrices
            ConstantMatrices(int lmax) : lmax(lmax) {
                basis::computeA1(lmax, A1);
                basis::computeA(lmax, A1, A);
                //solver::computerT(lmax, rT);
                //rTA1 = rT * A1;
                basis::computeU(lmax, U);
            }

    };

    // ****************************
    // ----------------------------
    //
    //    The surface map class
    //
    // ----------------------------
    // ****************************
    template <class T>
    class Map {

        public:

            const int lmax;                             /**< The highest degree of the map */
            const int N;                                /**< The number of map coefficients */
            Vector<T> y;                                /**< The map coefficients in the spherical harmonic basis */
            Vector<T> p;                                /**< The map coefficients in the polynomial basis */
            Vector<T> g;                                /**< The map coefficients in the Green's basis */
            UnitVector<T> axis;                         /**< The axis of rotation for the map */
            bool Y00_is_unity;                          /**< Flag: are we fixing the constant map coeff at unity? */
            std::map<string, Vector<double>> gradient;  /**< Dictionary of derivatives */
            Vector<T> dFdy;                             /**< Derivative of the flux w/ respect to the map coeffs */
            ConstantMatrices<T> C;                      /**< Constant matrices used throughout the code */
            //Greens<T> G;                                /**< Green's theorem integration stuff */

        protected:

            // Cached and temporary variables
            Vector<T> yzeta_rot;                        /**< The rotated spherical harmonic map in the zeta frame */
            Vector<T> y_rot;                            /**< The rotated spherical harmonic map in the base frame */
            Vector<T> y_rotz;                           /**< The rotated spherical harmonic map in the base frame, rotated to align the occultor with `yhat` */
            Vector<T> p_rot;                            /**< The rotated polynomial map in the base frame */
            T theta_y_rot;                              /**< The angle by which we rotated `y` to get `y_rot` (used for caching rotations) */
            Vector<T>* ptry;                            /**< Pointer to the actual spherical harmonic map we're using in the flux computation */
            Vector<T>* ptrp;                            /**< Pointer to the actual polynomial map we're using in the intensity computation */
            Vector<T> ARRy;                             /**< Everything but the solution vector in `s^TAR'Ry`; a handy temporary variable */
            VectorT<T> sTA;                             /**< The solution vector dotted into the change of basis matrix; handy for Grad stuff */
            VectorT<T> pTA;                             /**< The polynomial basis dotted into the change of basis matrix; handy for Grad stuff */
            Wigner<T> R;                                /**< The R rotation matrix in `s^TAR'Ry`; handy for Grad stuff */
            Wigner<T> RR;                               /**< The product of the two rotation matrices in `s^TAR'Ry`; handy for Grad stuff */
            VectorT<T> basis;                           /**< Polynomial basis, used to evaluate gradients */

            // Zeta transformation parameters
            T coszeta, sinzeta;                         /**< Angle between the axis of rotation and `zhat` */
            UnitVector<T> axzeta;                       /**< Axis of rotation to align the rotation axis with `zhat` */
            Wigner<T> Rzeta, RzetaInv;                  /**< The rotation matrix into the `zeta` frame and its inverse */
            Vector<T> yzeta;                            /**< The base map in the `zeta` frame */
            Vector<T> cosnt;                            /**< Vector of cos(n theta) values */
            Vector<T> sinnt;                            /**< Vector of sin(n theta) values */
            Vector<T> cosmt;                            /**< Vector of cos(m theta) values */
            Vector<T> sinmt;                            /**< Vector of sin(m theta) values */
            Vector<T> yrev;                             /**< Degree-wise reverse of the spherical harmonic map */

        public:

            // Constructor
            Map(int lmax=2) : lmax(lmax), N((lmax + 1) * (lmax + 1)),
                              C(lmax),
                              //G(lmax),
                              R(lmax),
                              RR(lmax), Rzeta(lmax), RzetaInv(lmax)
                              //M(*this)
                              {

                // Initialize all our vectors
                y = Vector<T>::Zero(N);
                p = Vector<T>::Zero(N);
                g = Vector<T>::Zero(N);
                axis(0) = 0;
                axis(1) = 1;
                axis(2) = 0;
                yzeta.resize(N);
                sTA = VectorT<T>::Zero(N);
                pTA = VectorT<T>::Zero(N);
                ARRy = Vector<T>::Zero(N);
                dFdy = Vector<T>::Zero(N);
                cosnt.resize(max(2, lmax + 1));
                cosnt(0) = 1.0;
                sinnt.resize(max(2, lmax + 1));
                sinnt(0) = 0.0;
                cosmt.resize(N);
                sinmt.resize(N);
                yrev.resize(N);
                y_rot.resize(N);
                y_rotz.resize(N);
                basis.resize(N);
                Y00_is_unity = false;
                update();

            }

            // Housekeeping functions
            void update();
            void reset();

            // I/O functions
            void setCoeff(int l, int m, T coeff);
            T getCoeff(int l, int m);
            std::string repr();

            // Various ways of rotating maps
            void rotate(const T& theta);
            void rotate(const T& costheta, const T& sintheta);
            void rotate(const T& theta, Vector<T>& yout);
            void rotate(const T& costheta, const T& sintheta, Vector<T>& yout);
            void rotatez(const T& costheta, const T& sintheta, const Vector<T>& yin, Vector<T>& yout);

            // Get the intensity of the map at a point
            T evaluate(const T& theta=0, const T& x0=0, const T& y0=0);

    };


    /* ---------------- */
    /*   HOUSEKEEPING   */
    /* ---------------- */


    // Update the maps after the coefficients changed
    // or after a base rotation was applied
    template <class T>
    void Map<T>::update() {

        // Update the polynomial and Green's map coefficients
        p = C.A1 * y;
        g = C.A * y;

        // Normalize the rotation axis
        axis = axis / sqrt(axis(0) * axis(0) + axis(1) * axis(1) + axis(2) * axis(2));

        // Compute the rotation transformation into and out of the `zeta` frame
        coszeta = axis(2);
        sinzeta = sqrt(1 - axis(2) * axis(2));
        T norm = sqrt(axis(0) * axis(0) + axis(1) * axis(1));
        if (abs(norm) < 10 * mach_eps<T>()) {
            // The rotation axis is zhat, so our zeta transform
            // is just the identity matrix.
            for (int l = 0; l < lmax + 1; l++) {
                Rzeta.Real[l] = Matrix<T>::Identity(2 * l + 1, 2 * l + 1);
                RzetaInv.Real[l] = Matrix<T>::Identity(2 * l + 1, 2 * l + 1);
            }
            yzeta = y;
        } else {
            // We need to compute the actual Wigner matrices
            axzeta(0) = axis(1) / norm;
            axzeta(1) = -axis(0) / norm;
            axzeta(2) = 0;
            computeR(lmax, axzeta, coszeta, sinzeta, Rzeta.Complex, Rzeta.Real);
            for (int l = 0; l < lmax + 1; l++) {
                yzeta.segment(l * l, 2 * l + 1) = Rzeta.Real[l] * y.segment(l * l, 2 * l + 1);
                RzetaInv.Real[l] = Rzeta.Real[l].transpose();
            }
        }

        // Reset our cache
        theta_y_rot = NAN;

    }

    // Reset the map
    template <class T>
    void Map<T>::reset() {
        y.setZero(N);
        if (Y00_is_unity) y(0) = 1;
        axis(0) = 0;
        axis(1) = 1;
        axis(2) = 0;
        update();
    }


    /* ---------------- */
    /*        I/O       */
    /* ---------------- */


    // Set the (l, m) coefficient
    template <class T>
    void Map<T>::setCoeff(int l, int m, T coeff) {
        if ((l == 0) && (Y00_is_unity) && (coeff != 1)) throw errors::Y00IsUnity();
        if ((0 <= l) && (l <= lmax) && (-l <= m) && (m <= l)) {
            int n = l * l + l + m;
            y(n) = coeff;
            update();
        } else throw errors::BadLM();
    }

    // Get the (l, m) coefficient
    template <class T>
    T Map<T>::getCoeff(int l, int m) {
        if ((0 <= l) && (l <= lmax) && (-l <= m) && (m <= l))
            return y(l * l + l + m);
        else throw errors::BadLM();
    }

    // Return a human-readable map string
    template <class T>
    std::string Map<T>::repr() {
        int n = 0;
        int nterms = 0;
        char buf[30];
        std::ostringstream os;
        os << "<STARRY Map: ";
        for (int l = 0; l < lmax + 1; l++) {
            for (int m = -l; m < l + 1; m++) {
                if (abs(y(n)) > 10 * mach_eps<T>()){
                    // Separator
                    if ((nterms > 0) && (y(n) > 0)) {
                        os << " + ";
                    } else if ((nterms > 0) && (y(n) < 0)){
                        os << " - ";
                    } else if ((nterms == 0) && (y(n) < 0)){
                        os << "-";
                    }
                    // Term
                    if ((y(n) == 1) || (y(n) == -1)) {
                        sprintf(buf, "Y_{%d,%d}", l, m);
                        os << buf;
                    } else if (fmod(abs(y(n)), 1.0) < 10 * mach_eps<T>()) {
                        sprintf(buf, "%d Y_{%d,%d}", (int)abs(y(n)), l, m);
                        os << buf;
                    } else if (fmod(abs(y(n)), 1.0) >= 0.01) {
                        sprintf(buf, "%.2f Y_{%d,%d}", abs(y(n)), l, m);
                        os << buf;
                    } else {
                        sprintf(buf, "%.2e Y_{%d,%d}", abs(y(n)), l, m);
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


    // Rotate the base map given `costheta` and `sintheta`
    template <class T>
    inline void Map<T>::rotate(const T& costheta, const T& sintheta, Vector<T>& yout) {
        // Rotate yzeta about zhat and store in yzeta_rot;
        rotatez(costheta, sintheta, yzeta, yzeta_rot);
        // Rotate out of the `zeta` frame
        for (int l = 0; l < lmax + 1; l++) {
            yout.segment(l * l, 2 * l + 1) = RzetaInv.Real[l] * yzeta_rot.segment(l * l, 2 * l + 1);
        }
    }

    // Shortcut to rotate the base map given just `theta`
    template <class T>
    inline void Map<T>::rotate(const T& theta, Vector<T>& yout) {
        rotate(cos(theta), sin(theta), yout);
    }

    // Rotate the base map in-place given `costheta` and `sintheta`
    template <class T>
    void Map<T>::rotate(const T& costheta, const T& sintheta) {
        // Do the rotation
        rotate(costheta, sintheta, y);
        // Update auxiliary variables
        update();
    }

    // Shortcut to rotate the base map in-place given just `theta`
    template <class T>
    void Map<T>::rotate(const T& theta) {
        rotate(cos(theta), sin(theta));
    }

    // Fast rotation about the z axis, skipping the Wigner matrix computation
    // See https://github.com/rodluger/starry/issues/137#issuecomment-405975092
    template <class T>
    inline void Map<T>::rotatez(const T& costheta, const T& sintheta, const Vector<T>& yin, Vector<T>& yout) {
        cosnt(1) = costheta;
        sinnt(1) = sintheta;
        for (int n = 2; n < lmax + 1; n++) {
            cosnt(n) = 2.0 * cosnt(n - 1) * cosnt(1) - cosnt(n - 2);
            sinnt(n) = 2.0 * sinnt(n - 1) * cosnt(1) - sinnt(n - 2);
        }
        int n = 0;
        for (int l = 0; l < lmax + 1; l++) {
            for (int m = -l; m < 0; m++) {
                cosmt(n) = cosnt(-m);
                sinmt(n) = -sinnt(-m);
                yrev(n) = yin(l * l + l - m);
                n++;
            }
            for (int m = 0; m < l + 1; m++) {
                cosmt(n) = cosnt(m);
                sinmt(n) = sinnt(m);
                yrev(n) = yin(l * l + l - m);
                n++;
            }
        }
        yout = cosmt.cwiseProduct(yin) - sinmt.cwiseProduct(yrev);
    }


    /* ------------- */
    /*   INTENSITY   */
    /* ------------- */


    // Evaluate our map at a given (x0, y0) coordinate
    template <class T>
    T Map<T>::evaluate(const T& theta, const T& x0, const T& y0) {

        // Rotate the map into view
        if (theta == 0) {
            ptrp = &p;
        } else {
            if (theta != theta_y_rot) rotate(theta, y_rot);
            theta_y_rot = theta;
            p_rot = C.A1 * y_rot;
            ptrp = &p_rot;
        }

        // Check if outside the sphere
        if (x0 * x0 + y0 * y0 > 1.0) return NAN * x0;

        int l, m, mu, nu, n = 0;
        T z0 = sqrt(1.0 - x0 * x0 - y0 * y0);

        // Evaluate each harmonic
        T res = 0;
        for (l=0; l<lmax+1; l++) {
            for (m=-l; m<l+1; m++) {
                if (abs((*ptrp)(n)) > 10 * mach_eps<T>()) {
                    mu = l - m;
                    nu = l + m;
                    if ((nu % 2) == 0) {
                        if ((mu > 0) && (nu > 0))
                            res += (*ptrp)(n) * pow(x0, mu / 2) * pow(y0, nu / 2);
                        else if (mu > 0)
                            res += (*ptrp)(n) * pow(x0, mu / 2);
                        else if (nu > 0)
                            res += (*ptrp)(n) * pow(y0, nu / 2);
                        else
                            res += (*ptrp)(n);
                    } else {
                        if ((mu > 1) && (nu > 1))
                            res += (*ptrp)(n) * pow(x0, (mu - 1) / 2) * pow(y0, (nu - 1) / 2) * z0;
                        else if (mu > 1)
                            res += (*ptrp)(n) * pow(x0, (mu - 1) / 2) * z0;
                        else if (nu > 1)
                            res += (*ptrp)(n) * pow(y0, (nu - 1) / 2) * z0;
                        else
                            res += (*ptrp)(n) * z0;
                    }
                }
                n++;
            }

        }
        return res;

    }

}; // namespace maps

#endif
