/**
Defines the surface map class.

*/

#ifndef _STARRY_MAPS_H_
#define _STARRY_MAPS_H_

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include "rotation.h"
#include "basis.h"
#include "solver.h"

// Rotation matrix type
template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T>
using VectorT = Eigen::Matrix<T, 1, Eigen::Dynamic>;
template <typename T>
using UnitVector = Eigen::Matrix<T, 3, 1>;

namespace maps {

    // Forward declare our main classes
    class Constants;
    template <class T>
    class Wigner;
    template <class T>
    class Map;

    // Rotation matrices
    template <class T>
    class Wigner {

        int lmax;

    public:

        Matrix<T>* Complex;
        Matrix<T>* Real;

        // Constructor: allocate the matrices
        Wigner(int lmax) : lmax(lmax) {

            Complex = new Matrix<T>[lmax + 1];
            Real = new Matrix<T>[lmax + 1];
            for (int l = 0; l < lmax + 1; l++) {
                Complex[l].resize(2 * l + 1, 2 * l + 1);
                Real[l].resize(2 * l + 1, 2 * l + 1);
            }

        }

        // Destructor: free the matrices
        ~Wigner() {
            delete [] Complex;
            delete [] Real;
        }

    };

    // Constant matrices
    class Constants {

        int lmax;

    public:

        Eigen::SparseMatrix<double> A1;
        Eigen::SparseMatrix<double> A;
        VectorT<double> rT;

        // Constructor: compute the matrices
        Constants(int lmax) : lmax(lmax) {
            basis::computeA1(lmax, A1);
            basis::computeA(lmax, A1, A);
            solver::computerT(lmax, rT);
        }

    };

    // The surface map vector class
    template <class T>
    class Map {

        int N;
        Vector<T> basis;
        bool needs_update;

        // Temporary map vector
        Vector<T> tmpvec;

        // Private methods
        void apply_rotation(UnitVector<T>& u, T costheta, T sintheta, Vector<T>& yin, Vector<T>& yout);

    public:

        // The map vectors
        Vector<T> y;
        Vector<T> p;
        Vector<T> g;

        // Map order
        int lmax;

        // Rotation matrices
        Wigner<T> R;

        // Constant matrices
        Constants C;

        // Greens data
        solver::Greens<T> G;

        // Constructor: initialize map to zeros
        Map(int lmax) : lmax(lmax), R(lmax), C(lmax), G(lmax) {
            N = (lmax + 1) * (lmax + 1);
            y = Vector<T>::Zero(N);
            p = Vector<T>::Zero(N);
            g = Vector<T>::Zero(N);
            tmpvec = Vector<T>::Zero(N);
            basis.resize(N, 1);
            update(true);
        }

        // Constructor: initialize map to array
        Map(Vector<T>& y) : y(y), lmax(floor(sqrt((double)y.size()) - 1)), R(lmax), C(lmax), G(lmax) {
            N = (lmax + 1) * (lmax + 1);
            tmpvec = Vector<T>::Zero(N);
            basis.resize(N, 1);
            update(true);
        }

        // Public methods
        T evaluate(const T& x0, const T& y0);
        void rotate(UnitVector<T>& u, T theta, Vector<T>& yin, Vector<T>& yout);
        void rotate(UnitVector<T>& u, T costheta, T sintheta, Vector<T>& yin, Vector<T>& yout);
        void rotate(UnitVector<T>& u, T theta);
        void rotate(UnitVector<T>& u, T costheta, T sintheta);
        void update(bool force=false);
        T flux(UnitVector<T>& u, T theta, T x0, T y0, T r);
        std::string repr(const double tol=1e-15);

    };

    // Update the maps
    template <class T>
    void Map<T>::update(bool force) {
        if (force || needs_update) {
            p = C.A1 * y;
            g = C.A * y;
            needs_update = false;
        }
    }


    // Evaluate our map at a given (x, y) coordinate
    template <class T>
    T Map<T>::evaluate(const T& x0, const T& y0) {

        // Update maps if necessary
        update();

        // Check if outside the sphere
        if (x0 * x0 + y0 * y0 > 1.0) return NAN;

        int l, m, mu, nu, n = 0;
        T z0 = sqrt(1.0 - x0 * x0 - y0 * y0);

        // Compute the polynomial basis
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

        // Dot the coefficients in to our polynomial map
        return p.dot(basis);

    }

    // Rotate a map `yin` and store the result in `yout`
    template <class T>
    void Map<T>::apply_rotation(UnitVector<T>& u, T costheta, T sintheta, Vector<T>& yin, Vector<T>& yout) {

        // Compute the rotation matrix R
        rotation::computeR(lmax, u, costheta, sintheta, R.Complex, R.Real);

        // Dot R in, order by order
        for (int l = 0; l < lmax + 1; l++) {
            yout.segment(l * l, 2 * l + 1) = R.Real[l] * yin.segment(l * l, 2 * l + 1);
        }

        return;
    }

    // Shortcut to rotate the base map in-place given `theta`
    template <class T>
    void Map<T>::rotate(UnitVector<T>& u, T theta) {
        apply_rotation(u, cos(theta), sin(theta), y, y);
        needs_update = true;
    }

    // Shortcut to rotate the base map in-place given `costheta` and `sintheta`
    template <class T>
    void Map<T>::rotate(UnitVector<T>& u, T costheta, T sintheta) {
        apply_rotation(u, costheta, sintheta, y, y);
        needs_update = true;
    }

    // Shortcut to rotate an arbitrary map given `theta`
    template <class T>
    void Map<T>::rotate(UnitVector<T>& u, T theta, Vector<T>& yin, Vector<T>& yout) {
        apply_rotation(u, cos(theta), sin(theta), yin, yout);
    }

    // Shortcut to rotate an arbitrary map given `costheta` and `sintheta`
    template <class T>
    void Map<T>::rotate(UnitVector<T>& u, T costheta, T sintheta, Vector<T>& yin, Vector<T>& yout) {
        apply_rotation(u, costheta, sintheta, yin, yout);
    }

    // Compute the total flux during or outside of an occultation
    template <class T>
    T Map<T>::flux(UnitVector<T>& u, T theta, T x0, T y0, T r) {

        // Pointer to the map we're integrating
        // (defaults to the base map)
        Vector<T>* ptry = &y;

        // Impact parameter
        T b = sqrt(x0 * x0 + y0 * y0);

        // Check for complete occultation
        if (b <= r - 1) return 0;

        // Rotate the map into view if necessary and update our pointer
        if (theta != 0) {
            rotate(u, theta, (*ptry), tmpvec);
            ptry = &tmpvec;
        }

        // No occultation: cake
        if (b >= 1 + r) {
            return C.rT * C.A1 * (*ptry);

        // Occultation
        } else {

            // Align occultor with the +y axis if necessary
            if (b > 0) {
                rotate(u, y0 / b, x0 / b, (*ptry), tmpvec);
                ptry = &tmpvec;
            }

            // Compute the sT vector
            solver::computesT<T>(G);

            // Dot the result in and we're done
            return G.sT * C.A * (*ptry);

        }

    }

    // Human-readable map string
    template <class T>
    std::string Map<T>::repr(const double TOL) {
        int n = 0;
        int nterms = 0;
        char buf[30];
        std::ostringstream os;
        os << "<STARRY Map: ";
        for (int l = 0; l < lmax + 1; l++) {
            for (int m = -l; m < l + 1; m++) {
                if (std::abs(y(n)) > TOL){
                    // Separator
                    if ((nterms > 0) && (y(n) > 0)) {
                        os << " + ";
                    } else if ((nterms > 0) && (y(n) < 0)){
                        os << " - ";
                    }
                    // Term
                    if ((y(n) == 1) || (y(n) == -1)) {
                        sprintf(buf, "Y_{%d,%d}", l, m);
                        os << buf;
                    } else if (fmod(std::abs(y(n)), 1) < TOL) {
                        sprintf(buf, "%d Y_{%d,%d}", (int)std::abs(y(n)), l, m);
                        os << buf;
                    } else if (fmod(std::abs(y(n)), 1) >= 0.01) {
                        sprintf(buf, "%.2f Y_{%d,%d}", std::abs(y(n)), l, m);
                        os << buf;
                    } else {
                        sprintf(buf, "%.2e Y_{%d,%d}", std::abs(y(n)), l, m);
                        os << buf;
                    }
                    nterms++;
                }
                n++;
            }
        }
        if (n == 0) {
            os << "Null map>";
            return std::string(os.str());
        } else {
            os << ">";
            return std::string(os.str());
        }
    }

}; // namespace maps

#endif
