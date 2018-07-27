/**
Defines the surface map class.

TODO: - Make everything protected and implement friend functions!
      - All I/O should happen through set and get methods.
      - Be careful when setting the `axis` within C++.
      - Put vectorization in here! Handle grad stuff in here, too.
*/

#ifndef _STARRY_MAPS_H_
#define _STARRY_MAPS_H_

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include "constants.h"
#include "rotation.h"
#include "basis.h"
#include "solver.h"
#include "numeric.h"
#include "errors.h"
#include "utils.h"
#include "sturm.h"
#include "minimize.h"

namespace maps {

    using std::abs;
    using std::max;
    using std::string;
    using rotation::Wigner;
    using rotation::computeR;
    using solver::Greens;
    using minimize::Minimizer;

    /* TODO: Forward-declare our friends!
    namespace orbital {
        template <class T> class Body;
        template <class T> class System;
        template <class T> class Star;
        template <class T> class Planet;
    }
    */

    // Constant matrices/vectors
    template <class T>
    class ConstantMatrices {

        public:

            int lmax;
            Eigen::SparseMatrix<T> A1;
            Eigen::SparseMatrix<T> A;
            VectorT<T> rTA1;
            VectorT<T> rT;
            Matrix<T> U;

            // Constructor: compute the matrices
            ConstantMatrices(int lmax) : lmax(lmax) {
                basis::computeA1(lmax, A1);
                basis::computeA(lmax, A1, A);
                solver::computerT(lmax, rT);
                rTA1 = rT * A1;
                basis::computeU(lmax, U);
            }

    };

    // No need to autodifferentiate these, since they are constant!
    template <>
    class ConstantMatrices<Grad> {

            Eigen::SparseMatrix<double> D_A1;
            Eigen::SparseMatrix<double> D_A;
            VectorT<double> D_rTA1;
            VectorT<double> D_rT;
            Matrix<double> D_U;

        public:

            int lmax;
            Eigen::SparseMatrix<Grad> A1;
            Eigen::SparseMatrix<Grad> A;
            VectorT<Grad> rTA1;
            VectorT<Grad> rT;
            Matrix<Grad> U;

            // Constructor: compute the matrices
            ConstantMatrices(int lmax) : lmax(lmax) {
                // Do things in double
                basis::computeA1(lmax, D_A1);
                basis::computeA(lmax, D_A1, D_A);
                solver::computerT(lmax, D_rT);
                D_rTA1 = D_rT * D_A1;
                basis::computeU(lmax, D_U);
                // Cast to Grad
                A1 = D_A1.cast<Grad>();
                A = D_A.cast<Grad>();
                rTA1 = D_rTA1.cast<Grad>();
                rT = D_rT.cast<Grad>();
                U = D_U.cast<Grad>();
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

            /* TODO: Let's make C and G protected
               and expose only rT and sT to the outside. 
               More protected stuff, perhaps Y00_is_unity?
               Add a setter and getter for the axis.
            friend class orbital::Body<T>;
            friend class orbital::System<T>;
            friend class orbital::Star<T>;
            friend class orbital::Planet<T>;
            */

        public:

            const int lmax;                             /**< The highest degree of the map */
            const int N;                                /**< The number of map coefficients */
            Vector<T> y;                                /**< The map coefficients in the spherical harmonic basis */
            Vector<T> p;                                /**< The map coefficients in the polynomial basis */
            Vector<T> g;                                /**< The map coefficients in the Green's basis */
            UnitVector<T> axis;                         /**< The axis of rotation for the map */
            bool Y00_is_unity;                          /**< Flag: are we fixing the constant map coeff at unity? */
            std::map<string, Vector<double>> derivs;    /**< Dictionary of derivatives */
            Vector<T> dFdy;                             /**< Derivative of the flux w/ respect to the map coeffs */
            ConstantMatrices<T> C;                      /**< Constant matrices used throughout the code */
            Greens<T> G;                                /**< Green's theorem integration stuff */

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

            // Classes
            Minimizer<T> M;                             /**< Surface map minimization stuff */

        public:

            // Constructor
            Map(int lmax=2) : lmax(lmax), N((lmax + 1) * (lmax + 1)),
                              C(lmax), G(lmax), R(lmax),
                              RR(lmax), Rzeta(lmax), RzetaInv(lmax),
                              M(*this) {

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
            void random(double beta=0);
            std::string repr();

            // Various ways of rotating maps
            void rotate(const T& theta);
            void rotate(const T& costheta, const T& sintheta);
            void rotate(const T& theta, Vector<T>& yout);
            void rotate(const T& costheta, const T& sintheta, Vector<T>& yout);
            void rotatez(const T& costheta, const T& sintheta, const Vector<T>& yin, Vector<T>& yout);

            // Get the intensity of the map at a point
            T evaluate(const T& theta=0, const T& x0=0, const T& y0=0);

            // Get the flux of the map during or outside of an occultation
            T flux_numerical(const T& theta=0, const T& xo=0, const T& yo=0, const T& ro=0, double tol=1e-4);
            T flux(const T& theta=0, const T& xo=0, const T& yo=0, const T& ro=0);

            // Map minimization routines
            bool psd(double epsilon=1e-6, int max_iterations=100);

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
            set_value(y(n), coeff);
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

    // Generate a random map with a given power spectrum power index `beta`
    // TODO: Not tested, not yet exposed to Python
    template <class T>
    void Map<T>::random(double beta) {
        int l, m, n;
        double norm;
        Vector<double> coeffs;
        setCoeff(0, 0, 1);
        for (l = 1; l < lmax + 1; l++) {
            coeffs = Vector<double>::Random(2 * l + 1);
            norm = pow(l, beta) / coeffs.squaredNorm();
            n = 0;
            for (m = -l; m < l + 1; m++) {
                setCoeff(l, m, coeffs(n) * norm);
                n++;
            }
        }
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
                if (abs(get_value(y(n))) > 10 * mach_eps<T>()){
                    // Separator
                    if ((nterms > 0) && (get_value(y(n)) > 0)) {
                        os << " + ";
                    } else if ((nterms > 0) && (get_value(y(n)) < 0)){
                        os << " - ";
                    } else if ((nterms == 0) && (get_value(y(n)) < 0)){
                        os << "-";
                    }
                    // Term
                    if ((get_value(y(n)) == 1) || (get_value(y(n)) == -1)) {
                        sprintf(buf, "Y_{%d,%d}", l, m);
                        os << buf;
                    } else if (fmod(abs(get_value(y(n))), 1.0) < 10 * mach_eps<T>()) {
                        sprintf(buf, "%d Y_{%d,%d}", (int)abs(get_value(y(n))), l, m);
                        os << buf;
                    } else if (fmod(abs(get_value(y(n))), 1.0) >= 0.01) {
                        sprintf(buf, "%.2f Y_{%d,%d}", abs(get_value(y(n))), l, m);
                        os << buf;
                    } else {
                        sprintf(buf, "%.2e Y_{%d,%d}", abs(get_value(y(n))), l, m);
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


    // Rotate the base map in-place given `costheta` and `sintheta`
    template <class T>
    void Map<T>::rotate(const T& costheta, const T& sintheta) {

        // Rotate yzeta in-place about zhat
        rotatez(costheta, sintheta, yzeta, yzeta);

        // Rotate out of the `zeta` frame
        for (int l = 0; l < lmax + 1; l++) {
            y.segment(l * l, 2 * l + 1) = RzetaInv.Real[l] * yzeta.segment(l * l, 2 * l + 1);
        }

        // Update auxiliary variables
        update();
    }

    // Shortcut to rotate the base map in-place given just `theta`
    template <class T>
    void Map<T>::rotate(const T& theta) {
        rotate(cos(theta), sin(theta));
    }

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

    // Evaluate our map at a given (x0, y0) coordinate
    template <>
    Grad Map<Grad>::evaluate(const Grad& theta, const Grad& x0, const Grad& y0) {

        // Rotate the map into view
        if (theta == 0) {
            ptrp = &p;

            // Compute the rotation matrix explicitly
            for (int l = 0; l < lmax + 1; l++)
                R.Real[l] = Matrix<Grad>::Identity(2 * l + 1, 2 * l + 1);

        } else {
            rotate(theta, y_rot);
            p_rot = C.A1 * y_rot;
            ptrp = &p_rot;

            // We need to explicitly compute the rotation matrix to get
            // the derivatives below. See the explanation in `flux`.
            for (int l = 0; l < lmax + 1; l++) {
                for (int j = 0; j < 2 * l + 1; j++)
                    R.Real[l].col(j) = RzetaInv.Real[l].col(j) * cosmt(l * l + j) +
                                       RzetaInv.Real[l].col(2 * l - j) * sinmt(l * l + j);
                R.Real[l] = R.Real[l] * Rzeta.Real[l];
            }

        }

        // Check if outside the sphere
        if (x0 * x0 + y0 * y0 > 1.0) return NAN * x0;

        int l, m, mu, nu, n = 0;
        Grad z0 = sqrt(1.0 - x0 * x0 - y0 * y0);

        // Evaluate each harmonic
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
                        basis(n) = pow(x0, (mu - 1) / 2) * pow(y0, (nu - 1) / 2) * z0;
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

        // Compute the map derivs
        if (theta == 0) {

            dFdy = basis * C.A1;

        } else {

            pTA = basis * C.A1;
            for (int l = 0; l < lmax + 1; l++)
                dFdy.segment(l * l, 2 * l + 1) = pTA.segment(l * l, 2 * l + 1) * R.Real[l];

        }

        // Dot the coefficients in to our polynomial map
        return basis.dot(*ptrp);

    }


    /* ------------- */
    /*      FLUX     */
    /* ------------- */


    // Compute the total flux during or outside of an occultation numerically
    template <class T>
    T Map<T>::flux_numerical(const T& theta, const T& xo, const T& yo, const T& ro, double tol) {

        // Impact parameter
        T b = sqrt(xo * xo + yo * yo);

        // Check for complete occultation
        if (b <= ro - 1) return 0;

        // Rotate the map into view
        if (theta == 0) {
            ptrp = &p;
        } else {
            if (theta != theta_y_rot) rotate(theta, y_rot);
            theta_y_rot = theta;
            p_rot = C.A1 * y_rot;
            ptrp = &p_rot;
        }

        // Compute the flux numerically
        return numeric::flux(xo, yo, ro, lmax, *ptrp, tol);

    }

    // Compute the total flux during or outside of an occultation
    template <class T>
    T Map<T>::flux(const T& theta, const T& xo, const T& yo, const T& ro) {

        // Impact parameter
        T b = sqrt(xo * xo + yo * yo);

        // Check for complete occultation
        if (b <= ro - 1) return 0;

        // Rotate the map into view
        if (theta == 0) {
            ptry = &y;
        } else {
            if (theta != theta_y_rot) rotate(theta, y_rot);
            theta_y_rot = theta;
            ptry = &y_rot;
        }

        // No occultation: cake
        if ((b >= 1 + ro) || (ro == 0)) {

            return C.rTA1 * (*ptry);

        // Occultation
        } else {

            // Align occultor with the +y axis
            if ((b > 0) && ((xo != 0) || (yo < 0))) {
                rotatez(yo / b, xo / b, *ptry, y_rotz);
                ptry = &y_rotz;
            }

            // Perform the rotation + change of basis
            ARRy = C.A * (*ptry);

            // Compute the sT vector
            solver::computesT(G, b, ro, ARRy);

            // Dot the result in and we're done
            return G.sT * ARRy;

        }

    }

    /*
    Compute the total flux & its derivatives during or outside of an occultation.
    Note that we compute map derivs manually for speed.
    We've factored stuff so that the flux is

        f = (s^T . A) . (Rz' . RzetaInv . Rz . Rzeta) . y

    The derivative is therefore

        df / dy = (s^T . A) . R

    where

        R = (Rz' . RzetaInv . Rz . Rzeta)
    */
    template <>
    Grad Map<Grad>::flux(const Grad& theta, const Grad& xo, const Grad& yo_, const Grad& ro) {

        // Local copy so we can nudge it away from the unstable point
        Grad yo = yo_;

        // Impact parameter
        Grad b = sqrt(xo * xo + yo * yo);

        // Nudge away from point instabilities
        if (b == 0) {
            yo += mach_eps<double>();
            b = sqrt(xo * xo + yo * yo);
        } else if (b == (1 - ro)) {
            b -= mach_eps<double>();
        } else if (b == 1 + ro) {
            b += mach_eps<double>();
        }

        // TODO: There are still deriv instabilities in the limits
        // b --> |1 - r| and b --> 1 + r. They are not terrible
        // (and only present in dF/db) but should be fixed.

        // Check for complete occultation
        if (b <= ro - 1) {
            dFdy = Vector<Grad>::Zero(N);
            return 0;
        }

        // Rotate the map into view
        rotate(theta, y_rot);
        ptry = &y_rot;

        // Here we compute R = RzetaInv * Rz * Rzeta so we can
        // analytically compute the map derivs below.
        // Note that the Rz matrix looks like this:
        /*
            ...                             ...
                C3                      S3
                    C2              S2
                        C1      S1
                            1
                       -S1      C1
                   -S2              C2
               -S3                      C3
            ...                             ...
        */
        // where CX = cos(X theta) and SX = sin(X theta).
        // This is the sum of a diagonal and an anti-diagonal
        // matrix, so the dot product of RzetaInv and Rz can
        // be computed efficiently by doing row-wise and col-wise
        // operations on RzetaInv.
        for (int l = 0; l < lmax + 1; l++) {
            for (int j = 0; j < 2 * l + 1; j++)
                R.Real[l].col(j) = RzetaInv.Real[l].col(j) * cosmt(l * l + j) +
                                   RzetaInv.Real[l].col(2 * l - j) * sinmt(l * l + j);
            R.Real[l] = R.Real[l] * Rzeta.Real[l];
        }

        // No occultation: cake
        if ((b >= 1 + ro) || (ro == 0)) {

            // Compute map derivs
            for (int l = 0; l < lmax + 1; l++)
                dFdy.segment(l * l, 2 * l + 1) = C.rTA1.segment(l * l, 2 * l + 1) * R.Real[l];

            return C.rTA1 * (*ptry);

        // Occultation
        } else {

            // Align occultor with the +y axis
            // Even if xo = 0, we compute the prime rotation matrix
            // to correctly propagate all the derivs.
            if (b > 0) {

                // Rotate
                rotatez(yo / b, xo / b, *ptry, y_rotz);
                ptry = &y_rotz;

                // Update the rotation matrix for the map derivs
                // Since we rotated the map, we need to dot Rz' into R
                // See our note above on what's going on here.
                for (int l = 0; l < lmax + 1; l++) {
                    for (int j = 0; j < 2 * l + 1; j++)
                        RR.Real[l].row(j) = R.Real[l].row(j) * cosmt(l * l + j) -
                                            R.Real[l].row(2 * l - j) * sinmt(l * l + j);
                }

            }

            // Perform the rotation + change of basis
            ARRy = C.A * (*ptry);

            // Compute the sT vector
            solver::computesT(G, b, ro, ARRy);

            // Compute the derivatives w.r.t. the map coefficients
            sTA = G.sT * C.A;
            for (int l = 0; l < lmax + 1; l++)
                dFdy.segment(l * l, 2 * l + 1) = sTA.segment(l * l, 2 * l + 1) * RR.Real[l];

            // Dot the result in to get the flux
            return G.sT * ARRy;

        }

    }


    /* ---------------- */
    /*   MINIMIZATION   */
    /* ---------------- */


    // Check whether the map is positive semi-definite
    template <class T>
    bool Map<T>::psd(double epsilon, int max_iterations) {
        if (lmax == 0) {
            // Trivial case
            return y(0) >= 0;
        } else if (lmax == 1) {
            // Dipole case
            return y(1) * y(1) + y(2) * y(2) + y(3) * y(3) <= y(0) / 3;
        } else {
            // Not analytic! For maps of type `double`, we can
            // run our numerical search for the minimum (see below)
            throw errors::MinimumIsNotAnalytic();
        }
    }

    // Check whether the map is positive semi-definite
    // Double override: for l > 1, we do this numerically
    template <>
    bool Map<double>::psd(double epsilon, int max_iterations) {
        if (lmax == 0) {
            // Trivial case
            return y(0) >= 0;
        } else if (lmax == 1) {
            // Dipole case
            return y(1) * y(1) + y(2) * y(2) + y(3) * y(3) <= y(0) / 3;
        } else {
            // We need to solve this numerically
            return M.psd(epsilon, max_iterations);
        }
    }


    // ****************************
    // ----------------------------
    //
    // The limb-darkened map class
    //
    // ----------------------------
    // ****************************
    template <class T>
    class LimbDarkenedMap {

        protected:

            // Temporary variables
            Vector<T> tmpvec;
            Vector<T> tmpy;
            VectorT<T> sTA;

        public:

            // The map vectors
            Vector<T> y;
            Vector<T> p;
            Vector<T> g;
            Vector<T> u;

            // Map order
            int N;
            int lmax;

            // Derivatives dictionary
            std::map<string, Eigen::VectorXd> derivs;
            Vector<T> dFdu;
            Vector<T> dndu;
            Matrix<T> dydu;
            Matrix<T> dpdu;
            Matrix<T> dgdu;

            // Constant matrices
            ConstantMatrices<T> C;

            // Greens data
            solver::Greens<T> G;

            // Constructor: initialize map to zeros
            LimbDarkenedMap(int lmax=2) :
                  lmax(lmax), C(lmax),
                  G(lmax) {
                N = (lmax + 1) * (lmax + 1);
                y = Vector<T>::Zero(N);
                p = Vector<T>::Zero(N);
                g = Vector<T>::Zero(N);
                u = Vector<T>::Zero(lmax + 1);
                tmpy = Vector<T>::Zero(lmax + 1);
                tmpvec = Vector<T>::Zero(N);
                sTA = VectorT<T>::Zero(N);
                dFdu = Vector<T>::Zero(lmax + 1);
                dndu = Vector<T>::Zero(lmax + 1);
                dydu = Matrix<T>::Zero(N, lmax + 1);
                dpdu = Matrix<T>::Zero(N, lmax + 1);
                dgdu = Matrix<T>::Zero(N, lmax + 1);
                reset();
            }

            // Public methods
            T evaluate(const T& x0=0, const T& y0=0);
            void update();
            bool psd();
            bool mono();
            void setCoeff(int l, T coeff);
            T getCoeff(int l);
            void reset();
            T flux_numerical(const T& xo=0, const T& yo=0, const T& ro=0, double tol=1e-4);
            T flux(const T& xo=0, const T& yo=0, const T& ro=0);
            std::string repr();

    };

    // Update the maps after the coefficients changed
    template <class T>
    void LimbDarkenedMap<T>::update() {

        // Update our map vectors
        T norm;
        y.setZero(N);

        // Fast relations for constant, linear, and quad limb darkening
        if (lmax == 0) {
            norm = PI<T>();
            y(0) = 2 * SQRT_PI<T>() / norm;
            p.setZero(N);
            g.setZero(N);
            p(0) = 1 / norm;
            g(0) = p(0);

        } else if (lmax == 1) {
            norm = PI<T>() * (1 - u(1) / 3.);
            y(0) = (2. / norm) * SQRT_PI<T>() / 3. * (3 - 3 * u(1));
            y(2) = (2. / norm) * SQRT_PI<T>() / sqrt(3.) * u(1);
            p.setZero(N);
            g.setZero(N);
            p(0) = (1 - u(1)) / norm;
            p(2) = u(1) / norm;
            g(0) = p(0);
            g(2) = p(2);

        } else if (lmax == 2) {
            norm = PI<T>() * (1 - u(1) / 3. - u(2) / 6.);
            y(0) = (2. / norm) * SQRT_PI<T>() / 3. * (3 - 3 * u(1) - 4 * u(2));
            y(2) = (2. / norm) * SQRT_PI<T>() / sqrt(3.) * (u(1) + 2 * u(2));
            y(6) = (-4. / 3.) * SQRT_PI<T>() / sqrt(5.) * u(2) / norm;
            p.setZero(N);
            g.setZero(N);
            p(0) = (1 - u(1) - 2 * u(2)) / norm;
            p(2) = (u(1) + 2 * u(2)) / norm;
            p(4) = u(2) / norm;
            p(8) = u(2) / norm;
            g(0) = p(0);
            g(2) = p(2);
            g(4) = p(4) / 3.;
            g(8) = p(8);

        } else {
            norm = 1;
            for (int l = 1; l < lmax + 1; l++)
                norm -= 2.0 * u(l) / ((l + 1) * (l + 2));
            norm *= PI<T>();
            tmpy = C.U * u;

            int n = 0;
            for (int l = 0; l < lmax + 1; l++)
                y(l * (l + 1)) = tmpy(n++) / norm;

            p = C.A1 * y;
            g = C.A * y;
        }
    }

    // Update the maps after the coefficients changed
    // **Overload for autodiff of map coeffs**
    template <>
    void LimbDarkenedMap<Grad>::update() {

        // Update our map vectors
        Grad norm = 1;
        y.setZero(N);
        dndu.setZero(lmax + 1);
        for (int l = 1; l < lmax + 1; l++) {
            norm -= 2 * u(l) / ((l + 1) * (l + 2));
            dndu(l) -= 2.0 / ((l + 1) * (l + 2));
        }
        norm *= PI<double>();
        dndu *= PI<double>();
        tmpy = C.U * u;

        int n = 0;
        dydu.setZero(N, lmax + 1);
        for (int l = 0; l < lmax + 1; l++) {
            y(l * (l + 1)) = tmpy(n) / norm;
            dydu.row(l * (l + 1)) = ((C.U.row(n).transpose() * norm - tmpy(n) * dndu) / (norm * norm));
            n++;
        }

        p = C.A1 * y;
        g = C.A * y;
        dpdu = C.A1 * dydu;
        dgdu = C.A * dydu;

    }

    // Check whether the map is positive semi-definite
    // using Sturm's theorem
    template <class T>
    bool LimbDarkenedMap<T>::psd() {
        Vector<T> c = -u.reverse();
        c(c.size() - 1) = 1;
        int nroots = sturm::polycountroots(c);
        if (nroots == 0)
            return true;
        else
            return false;
    }

    // Check whether the map is monotonically decreasing
    // toward the limb using Sturm's theorem on the derivative
    template <class T>
    bool LimbDarkenedMap<T>::mono() {
        // The radial profile is
        //      I = 1 - (1 - mu)^1 u1 - (1 - mu)^2 u2 - ...
        //        = x^0 c0 + x^1 c1 + x^2 c2 + ...
        // where x = (1 - mu), c = -u, c(0) = 1
        // We want dI/dx < 0 everywhere, so we want the polynomial
        //      P = x^0 c1 + 2x^1 c2 + 3x^2 c3 + ...
        // to have zero roots in the interval [0, 1].
        Vector<T> du = u.segment(1, lmax);
        for (int i=0; i<lmax; i++) du(i) *= (i + 1);
        Vector<T> c = -du.reverse();
        int nroots = sturm::polycountroots(c);
        if (nroots == 0)
            return true;
        else
            return false;
    }

    // Evaluate our map at a given (x0, y0) coordinate
    template <class T>
    T LimbDarkenedMap<T>::evaluate(const T& x0, const T& y0) {

        // Check if outside the sphere
        if (x0 * x0 + y0 * y0 > 1.0) return NAN * x0;

        int l, m, mu, nu, n = 0;
        T z0 = sqrt(1.0 - x0 * x0 - y0 * y0);

        // Evaluate each harmonic
        T res = 0;
        for (l=0; l<lmax+1; l++) {
            for (m=-l; m<l+1; m++) {
                if (abs(p(n)) > 10 * mach_eps<T>()) {
                    mu = l - m;
                    nu = l + m;
                    if ((nu % 2) == 0)
                        res += p(n) * pow(x0, mu / 2) * pow(y0, nu / 2);
                    else
                        res += p(n) * pow(x0, (mu - 1) / 2) *
                                              pow(y0, (nu - 1) / 2) * z0;
                }
                n++;
            }
        }
        return res;

    }

    // Evaluate our map at a given (x0, y0) coordinate
    // **Gradient over-ride: compute map derivs manually for speed**
    template <>
    Grad LimbDarkenedMap<Grad>::evaluate(const Grad& x0, const Grad& y0) {

        // Check if outside the sphere
        if (x0 * x0 + y0 * y0 > 1.0) return NAN * x0;

        int l, m, mu, nu, n = 0;
        Grad z0 = sqrt(1.0 - x0 * x0 - y0 * y0);

        // Evaluate each harmonic
        Vector<Grad> basis;
        basis.resize(N);
        for (l=0; l<lmax+1; l++) {
            for (m=-l; m<l+1; m++) {
                mu = l - m;
                nu = l + m;
                if ((nu % 2) == 0)
                    basis(n) = pow(x0, mu / 2) * pow(y0, nu / 2);
                else
                    basis(n) = pow(x0, (mu - 1) / 2) * pow(y0, (nu - 1) / 2) * z0;
                n++;
            }
        }

        dFdu = basis.transpose() * dpdu;
        return basis.dot(p);

    }

    // Compute the total flux during or outside of an occultation numerically
    template <class T>
    T LimbDarkenedMap<T>::flux_numerical(const T& xo, const T& yo, const T& ro, double tol) {

        // Impact parameter
        T b = sqrt(xo * xo + yo * yo);

        // Check for complete occultation
        if (b <= ro - 1) return 0;

        // Compute it numerically
        tmpvec = C.A1 * y;
        return numeric::flux(xo, yo, ro, lmax, tmpvec, tol);

    }

    // Compute the total flux during or outside of an occultation
    template <class T>
    T LimbDarkenedMap<T>::flux(const T& xo, const T& yo, const T& ro) {

        // Impact parameter
        T b = sqrt(xo * xo + yo * yo);

        // Check for complete occultation
        if (b <= ro - 1) return 0;

        // If we're doing quadratic limb darkening, let's skip all the overhead
        if ((lmax <= 2) && (ro < 1)) {
            if ((b >= 1 + ro) || (ro == 0))
                return 1.0;
            else {
                T s0, s2, s8;
                if (lmax == 0)
                    solver::QuadLimbDark<T>(G, b, ro, g(0), 0, 0, s0, s2, s8);
                else if (lmax == 1)
                    solver::QuadLimbDark<T>(G, b, ro, g(0), g(2), 0, s0, s2, s8);
                else
                    solver::QuadLimbDark<T>(G, b, ro, g(0), g(2), g(8), s0, s2, s8);
                return s0 * g(0) + s2 * g(2) + s8 * g(8);
            }
        }

        // No occultation: cake
        if ((b >= 1 + ro) || (ro == 0)) {

            return 1.0;

        // Occultation
        } else {

            // Compute the sT vector
            solver::computesT(G, b, ro, g);

            // Dot the result in and we're done
            return G.sT * g;

        }

    }

    // Compute the total flux during or outside of an occultation
    // **Gradient over-ride: compute map derivs manually for speed**
    template <>
    Grad LimbDarkenedMap<Grad>::flux(const Grad& xo, const Grad& yo_, const Grad& ro) {

        // Local copy so we can nudge it away from the
        // unstable point
        Grad yo = yo_;

        // Impact parameter
        Grad b = sqrt(xo * xo + yo * yo);

        // Nudge away from point instabilities
        if (b == 0) {
            yo += mach_eps<double>();
            b = sqrt(xo * xo + yo * yo);
        } else if (b == (1 - ro)) {
            b -= mach_eps<double>();
        } else if (b == 1 + ro) {
            b += mach_eps<double>();
        }

        // TODO: There are still instabilities in the limits
        // b --> |1 - r| and b --> 1 + r. They are not terrible
        // (and only present in dF/db) but should be fixed.

        // Check for complete occultation
        if (b <= ro - 1) {
            dFdu = Vector<Grad>::Zero(lmax + 1);
            return 0;
        }

        // If we're doing quadratic limb darkening, let's skip all the overhead
        if ((lmax <= 2) && (ro < 1)) {
            if ((b >= 1 + ro) || (ro == 0)) {
                dFdu = Vector<Grad>::Zero(lmax + 1);
                return 1.0;
            } else {
                Grad s0, s2, s8;
                if (lmax == 0)
                    solver::QuadLimbDark<Grad>(G, b, ro, g(0), 0, 0, s0, s2, s8);
                else if (lmax == 1)
                    solver::QuadLimbDark<Grad>(G, b, ro, g(0), g(2), 0, s0, s2, s8);
                else
                    solver::QuadLimbDark<Grad>(G, b, ro, g(0), g(2), g(8), s0, s2, s8);
                dFdu = (s0 * dgdu.row(0) + s2 * dgdu.row(2) + s8 * dgdu.row(8)).transpose();
                return s0 * g(0) + s2 * g(2) + s8 * g(8);
            }
        }

        // No occultation: cake
        if ((b >= 1 + ro) || (ro == 0)) {

            dFdu = Vector<Grad>::Zero(lmax + 1);
            return 1.0;

        // Occultation
        } else {

            // Compute the sT vector
            solver::computesT(G, b, ro, g);

            // Compute the map derivs
            dFdu = G.sT * dgdu;

            // Dot the result in
            return G.sT * g;

        }

    }

    // Set a limb darkening coefficient
    template <class T>
    void LimbDarkenedMap<T>::setCoeff(int l, T u_l) {
        if ((l <= 0) || (l > lmax)) {
            throw errors::BadIndex();
        }

        // Set the limb darkening coefficient
        set_value(u(l), u_l);

        // Update all the vectors
        update();

    }

    // Get a limb darkening coefficient
    template <class T>
    T LimbDarkenedMap<T>::getCoeff(int l) {
        if ((l <= 0) || (l > lmax)) {
            throw errors::BadIndex();
        } else {
            return u(l);
        }
    }

    // Reset the map
    template <class T>
    void LimbDarkenedMap<T>::reset() {
        u.setZero(lmax + 1);
        u(0) = -1;
        update();
    }

    // Return a human-readable map string
    template <class T>
    std::string LimbDarkenedMap<T>::repr() {
        std::ostringstream os;
        char buf[30];
        os << "<STARRY LimbDarkenedMap: ";
        for (int l = 1; l < lmax + 1; l++) {
            sprintf(buf, "u%d = %.3f", l, get_value(u(l)));
            os << buf;
            if (l < lmax) os << ", ";
        }
        os << ">";
        return std::string(os.str());
    }

}; // namespace maps

#endif
