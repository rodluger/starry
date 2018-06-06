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
#include "solver.h"
#include "numeric.h"
#include "errors.h"
#include "utils.h"
#include "sturm.h"

namespace maps {

    using std::abs;
    using std::string;

    // Constant matrices/vectors
    template <class T>
    class Constants {

        public:

            int lmax;
            Eigen::SparseMatrix<T> A1;
            Eigen::SparseMatrix<T> A;
            VectorT<T> rTA1;
            VectorT<T> rT;
            Matrix<T> U;

            // Constructor: compute the matrices
            Constants(int lmax) : lmax(lmax) {
                basis::computeA1(lmax, A1);
                basis::computeA(lmax, A1, A);
                solver::computerT(lmax, rT);
                rTA1 = rT * A1;
                basis::computeU(lmax, U);
            }

    };

    // No need to autodifferentiate these, since they are constant!
    template <>
    class Constants<Grad> {

        public:

            int lmax;
            Eigen::SparseMatrix<double> A1;
            Eigen::SparseMatrix<double> A;
            VectorT<double> rTA1;
            VectorT<double> rT;
            Matrix<double> U;

            // Constructor: compute the matrices
            Constants(int lmax) : lmax(lmax) {
                basis::computeA1(lmax, A1);
                basis::computeA(lmax, A1, A);
                solver::computerT(lmax, rT);
                rTA1 = rT * A1;
                basis::computeU(lmax, U);
            }

    };

    // ****************************
    // ----------------------------
    //
    // The surface map vector class
    //
    // ----------------------------
    // ****************************
    template <class T>
    class Map {

        protected:

            Vector<T> basis;

            // Temporary variables
            Vector<T> tmpvec;
            T tmpscalar;
            T tmpu1, tmpu2, tmpu3;
            Vector<T> ARRy;

            // Private methods
            void apply_rotation(const UnitVector<T>& axis, const T& costheta, const T& sintheta,
                                const Vector<T>& yin, Vector<T>& yout);

        public:

            // The map vectors
            Vector<T> y;
            Vector<T> p;
            Vector<T> g;

            // Map order
            int N;
            int lmax;

            // Misc flags
            bool Y00_is_unity;

            // Derivatives dictionary
            std::map<string, Vector<double>> derivs;

            // Rotation matrices
            rotation::Wigner<T> R;

            // Constant matrices
            Constants<T> C;

            // Greens data
            solver::Greens<T> G;

            // Constructor: initialize map to zeros
            Map(int lmax=2) :
                  lmax(lmax), R(lmax), C(lmax),
                  G(lmax, true) {
                N = (lmax + 1) * (lmax + 1);
                y = Vector<T>::Zero(N);
                p = Vector<T>::Zero(N);
                g = Vector<T>::Zero(N);
                tmpvec = Vector<T>::Zero(N);
                ARRy = Vector<T>::Zero(N);
                tmpscalar = NAN;
                tmpu1 = 0;
                tmpu2 = 0;
                tmpu3 = 0;
                basis.resize(N, 1);
                Y00_is_unity = false;
                update();
            }

            // Public methods
            T evaluate(const UnitVector<T>& axis=yhat, const T& theta=0, const T& x0=0, const T& y0=0);
            void rotate(const UnitVector<T>& axis, const T& theta, const Vector<T>& yin, Vector<T>& yout);
            void rotate(const UnitVector<T>& axis, const T& costheta, const T& sintheta, const Vector<T>& yin, Vector<T>& yout);
            void rotate(const UnitVector<T>& axis, const T& theta);
            void rotate(const UnitVector<T>& axis, const T& costheta, const T& sintheta);
            void update();
            void random(double beta=0);
            void set_coeff(int l, int m, T coeff);
            T get_coeff(int l, int m);
            void reset();
            T flux_numerical(const UnitVector<T>& axis=yhat, const T& theta=0, const T& xo=0, const T& yo=0, const T& ro=0, double tol=1e-4);
            T flux_mp(const UnitVector<T>& axis=yhat, const T& theta=0, const T& xo=0, const T& yo=0, const T& ro=0);
            T flux(const UnitVector<T>& axis=yhat, const T& theta=0, const T& xo=0, const T& yo=0, const T& ro=0);
            std::string repr();

    };

    // Rotate a map `yin` and store the result in `yout`
    template <class T>
    void Map<T>::apply_rotation(const UnitVector<T>& axis, const T& costheta, const T& sintheta,
                                const Vector<T>& yin, Vector<T>& yout) {

        // Compute the rotation matrix R
        rotation::computeR(lmax, axis, costheta, sintheta, R.Complex, R.Real);

        // Dot R in, order by order
        for (int l = 0; l < lmax + 1; l++) {
            yout.segment(l * l, 2 * l + 1) = R.Real[l] *
                                             yin.segment(l * l, 2 * l + 1);
        }

        return;
    }

    // Update the maps after the coefficients changed
    // or after a base rotation was applied
    template <class T>
    void Map<T>::update() {
        p = C.A1 * y;
        g = C.A * y;
        tmpscalar = NAN;
        tmpu1 = 0;
        tmpu2 = 0;
        tmpu3 = 0;
        tmpvec = Vector<T>::Zero(N);
    }


    // Evaluate our map at a given (x0, y0) coordinate
    template <class T>
    T Map<T>::evaluate(const UnitVector<T>& axis, const T& theta, const T& x0, const T& y0) {

        // Get the polynomial map
        Vector<T>* ptrmap;

        if (theta == 0) {
            // We will use this.p
            ptrmap = &p;
        } else if ((theta == tmpscalar) && (axis(0) == tmpu1) && (axis(1) == tmpu2) && (axis(2) == tmpu3)) {
            // We will use this.tmpvec, which we computed last time around
            ptrmap = &tmpvec;
        } else {
            // Rotate the map into view
            rotate(axis, theta, y, tmpvec);
            tmpvec = C.A1 * tmpvec;
            ptrmap = &tmpvec;
        }

        // Save this value of theta so we don't have
        // to keep rotating the map when we vectorize
        // this function!
        tmpscalar = theta;
        tmpu1 = axis(0);
        tmpu2 = axis(1);
        tmpu3 = axis(2);

        // Check if outside the sphere
        if (x0 * x0 + y0 * y0 > 1.0) return NAN * x0;

        int l, m, mu, nu, n = 0;
        T z0 = sqrt(1.0 - x0 * x0 - y0 * y0);

        // Compute the polynomial basis where it is needed
        for (l=0; l<lmax+1; l++) {
            for (m=-l; m<l+1; m++) {
                if (abs((*ptrmap)(n)) < 10 * std::numeric_limits<T>::epsilon()) {
                    basis(n) = 0 * x0;
                } else {
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
                }
                n++;
            }

        }

        // Dot the coefficients in to our polynomial map
        return (*ptrmap).dot(basis);

    }

    // Shortcut to rotate the base map in-place given `theta`
    template <class T>
    void Map<T>::rotate(const UnitVector<T>& axis, const T& theta) {
        T costheta = cos(theta);
        T sintheta = sin(theta);
        apply_rotation(axis, costheta, sintheta, y, y);
        update();
    }

    // Shortcut to rotate the base map in-place given `costheta` and `sintheta`
    template <class T>
    void Map<T>::rotate(const UnitVector<T>& axis, const T& costheta, const T& sintheta) {
        apply_rotation(axis, costheta, sintheta, y, y);
        update();
    }

    // Shortcut to rotate an arbitrary map given `theta`
    template <class T>
    void Map<T>::rotate(const UnitVector<T>& axis, const T& theta, const Vector<T>& yin, Vector<T>& yout) {
        T costheta = cos(theta);
        T sintheta = sin(theta);
        apply_rotation(axis, costheta, sintheta, yin, yout);
    }

    // Shortcut to rotate an arbitrary map given `costheta` and `sintheta`
    template <class T>
    void Map<T>::rotate(const UnitVector<T>& axis, const T& costheta, const T& sintheta, const Vector<T>& yin, Vector<T>& yout) {
        apply_rotation(axis, costheta, sintheta, yin, yout);
    }

    // Compute the total flux during or outside of an occultation numerically
    template <class T>
    T Map<T>::flux_numerical(const UnitVector<T>& axis, const T& theta, const T& xo, const T& yo, const T& ro, double tol) {

        // Impact parameter
        T b = sqrt(xo * xo + yo * yo);

        // Check for complete occultation
        if (b <= ro - 1) return 0;

        // Pointer to the map we're integrating
        // (defaults to the base map)
        Vector<T>* ptry = &y;

        // Rotate the map into view if necessary and update our pointer
        if (theta != 0) {
            rotate(axis, theta, (*ptry), tmpvec);
            ptry = &tmpvec;
        }

        // Compute the flux numerically
        tmpvec = C.A1 * (*ptry);
        return numeric::flux(xo, yo, ro, lmax, tmpvec, tol);

    }

    // Compute the total flux during or outside of an occultation
    template <class T>
    T Map<T>::flux(const UnitVector<T>& axis, const T& theta, const T& xo, const T& yo, const T& ro) {

        // Impact parameter
        T b = sqrt(xo * xo + yo * yo);

        // Check for complete occultation
        if (b <= ro - 1) return 0;

        // Pointer to the map we're integrating
        // (defaults to the base map)
        Vector<T>* ptry = &y;

        // Rotate the map into view if necessary and update our pointer
        if (theta != 0) {
            rotate(axis, theta, (*ptry), tmpvec);
            ptry = &tmpvec;
        }

        // No occultation: cake
        if ((b >= 1 + ro) || (ro == 0)) {

            return C.rTA1 * (*ptry);

        // Occultation
        } else {

            // Align occultor with the +y axis if necessary
            if ((b > 0) && (xo != 0)) {
                UnitVector<T> zaxis({0, 0, 1});
                T yo_b(yo / b);
                T xo_b(xo / b);
                rotate(zaxis, yo_b, xo_b, (*ptry), tmpvec);
                ptry = &tmpvec;
            }

            // Perform the rotation + change of basis
            ARRy = C.A * (*ptry);

            // Compute the sT vector
            solver::computesT(G, b, ro, ARRy);

            // Dot the result in and we're done
            return G.sT * ARRy;

        }

    }

    // Set the (l, m) coefficient
    template <class T>
    void Map<T>::set_coeff(int l, int m, T coeff) {
        if ((l == 0) && (Y00_is_unity) && (coeff != 1)) throw errors::Y00IsUnity();
        if ((0 <= l) && (l <= lmax) && (-l <= m) && (m <= l)) {
            int n = l * l + l + m;
            set_value(y(n), coeff);
            update();
        } else throw errors::BadLM();
    }

    // Get the (l, m) coefficient
    template <class T>
    T Map<T>::get_coeff(int l, int m) {
        if ((0 <= l) && (l <= lmax) && (-l <= m) && (m <= l))
            return y(l * l + l + m);
        else throw errors::BadLM();
    }

    // Reset the map
    template <class T>
    void Map<T>::reset() {
        y.setZero(N);
        if (Y00_is_unity) y(0) = 1;
        update();
    }

    // Generate a random map with a given power spectrum power index `beta`
    template <class T>
    void Map<T>::random(double beta) {
        int l, m, n;
        double norm;
        Vector<double> coeffs;
        set_coeff(0, 0, 1);
        for (l = 1; l < lmax + 1; l++) {
            coeffs = Vector<double>::Random(2 * l + 1);
            norm = pow(l, beta) / coeffs.squaredNorm();
            n = 0;
            for (m = -l; m < l + 1; m++) {
                set_coeff(l, m, coeffs(n) * norm);
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
                if (abs(get_value(y(n))) > 10 * std::numeric_limits<T>::epsilon()){
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
                    } else if (fmod(abs(get_value(y(n))), 1.0) < 10 * std::numeric_limits<T>::epsilon()) {
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

            Vector<T> basis;

            // Temporary variables
            Vector<T> ARRy;
            Vector<T> tmpvec;
            Vector<T> tmpy;

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

            // Constant matrices
            Constants<T> C;

            // Greens data
            solver::Greens<T> G;

            // Constructor: initialize map to zeros
            LimbDarkenedMap(int lmax=2) :
                  lmax(lmax), C(lmax),
                  G(lmax, true) {
                N = (lmax + 1) * (lmax + 1);
                y = Vector<T>::Zero(N);
                p = Vector<T>::Zero(N);
                g = Vector<T>::Zero(N);
                u = Vector<T>::Zero(lmax + 1);
                tmpy = Vector<T>::Zero(lmax + 1);
                tmpvec = Vector<T>::Zero(N);
                ARRy = Vector<T>::Zero(N);
                basis.resize(N, 1);
                reset();
                update();
            }

            // Public methods
            T evaluate(const T& x0=0, const T& y0=0);
            void update();
            int roots();
            void set_coeff(int l, T coeff);
            T get_coeff(int l);
            void reset();
            T flux_numerical(const T& xo=0, const T& yo=0, const T& ro=0, double tol=1e-4);
            T flux_mp(const T& xo=0, const T& yo=0, const T& ro=0);
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
            norm = M_PI;
            y(0) = 2 * sqrt(M_PI) / norm;
            p.setZero(N);
            g.setZero(N);
            p(0) = 1 / norm;
            g(0) = p(0);

        } else if (lmax == 1) {
            norm = M_PI * (1 - u(1) / 3.);
            y(0) = (2. / norm) * sqrt(M_PI) / 3. * (3 - 3 * u(1));
            y(2) = (2. / norm) * sqrt(M_PI / 3.) * u(1);
            p.setZero(N);
            g.setZero(N);
            p(0) = (1 - u(1)) / norm;
            p(2) = u(1) / norm;
            g(0) = p(0);
            g(2) = p(2);

        } else if (lmax == 2) {
            norm = M_PI * (1 - u(1) / 3. - u(2) / 6.);
            y(0) = (2. / norm) * sqrt(M_PI) / 3. * (3 - 3 * u(1) - 4 * u(2));
            y(2) = (2. / norm) * sqrt(M_PI / 3.) * (u(1) + 2 * u(2));
            y(6) = (-4. / 3.) * sqrt(M_PI / 5.) * u(2) / norm;
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
                norm -= 2 * u(l) / ((l + 1) * (l + 2));
            norm *= M_PI;
            tmpy = C.U * u;

            int n = 0;
            for (int l = 0; l < lmax + 1; l++)
                y(l * (l + 1)) = tmpy(n++) / norm;

            p = C.A1 * y;
            g = C.A * y;
        }
    }

    // Count the number of zero-crossings of the specific intensity
    // For a physical positive semi-definite map, this should return zero!
    template <class T>
    int LimbDarkenedMap<T>::roots() {
        Vector<T> c = -u.reverse();
        c(c.size() - 1) = 1;
        return sturm::polycountroots(c);
    }

    // Evaluate our map at a given (x0, y0) coordinate
    template <class T>
    T LimbDarkenedMap<T>::evaluate(const T& x0, const T& y0) {

        // Check if outside the sphere
        if (x0 * x0 + y0 * y0 > 1.0) return NAN * x0;

        int l, m, mu, nu, n = 0;
        T z0 = sqrt(1.0 - x0 * x0 - y0 * y0);

        // Compute the polynomial basis where it is needed
        for (l=0; l<lmax+1; l++) {
            for (m=-l; m<l+1; m++) {
                if (abs(p(n)) < 10 * std::numeric_limits<T>::epsilon()) {
                    basis(n) = 0 * x0;
                } else {
                    mu = l - m;
                    nu = l + m;
                    if ((nu % 2) == 0)
                        basis(n) = pow(x0, mu / 2) * pow(y0, nu / 2);
                    else
                        basis(n) = pow(x0, (mu - 1) / 2) *
                                   pow(y0, (nu - 1) / 2) * z0;
                }
                n++;
            }
        }

        // Dot the coefficients in to our polynomial map
        return p.dot(basis);

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

        // AutoDiff casting hack
        T zero = 0 * ro;

        // Impact parameter
        T b = sqrt(xo * xo + yo * yo);

        // Check for complete occultation
        if (b <= ro - 1) return zero;

        // If we're doing quadratic limb darkening, let's skip all the overhead
        if ((lmax <= 2) && (ro < 1)) {
            if ((b >= 1 + ro) || (ro == 0))
                return 1.0;
            else {
                if (lmax == 0)
                    return solver::QuadLimbDark(G, b, ro, g(0), zero, zero);
                else if (lmax == 1)
                    return solver::QuadLimbDark(G, b, ro, g(0), g(2), zero);
                else
                    return solver::QuadLimbDark(G, b, ro, g(0), g(2), g(8));
            }
        }

        // No occultation: cake
        if ((b >= 1 + ro) || (ro == 0)) {

            return C.rTA1 * y;

        // Occultation
        } else {

            // Perform the change of basis
            ARRy = C.A * y;

            // Compute the sT vector
            solver::computesT<T>(G, b, ro, ARRy);

            // Dot the result in and we're done
            return G.sT * ARRy;

        }

    }

    // Set a limb darkening coefficient
    template <class T>
    void LimbDarkenedMap<T>::set_coeff(int l, T u_l) {
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
    T LimbDarkenedMap<T>::get_coeff(int l) {
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
