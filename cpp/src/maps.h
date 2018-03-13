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
    class Greens;
    template <class T>
    class Wigner;
    template <class T>
    class Primitive;
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

    // Primitive integral helper matrices
    template <class T>
    class Primitive {

        int lmax;
        int N;
        Matrix<bool> set;
        Matrix<T> matrix;
        T (*setter)(Greens<T>&, int, int);
        Greens<T>& G;

    public:

        // Constructor
        Primitive(Greens<T>& G, int lmax, T (*setter)(Greens<T>&, int, int)) : lmax(lmax), setter(setter), G(G) {
            N = 2 * lmax + 1;
            set = Matrix<bool>::Zero(N, N);
            matrix.resize(N, N);
        }

        // Getter function. G is a pointer to the current Greens struct, and setter
        // is a pointer to the function that computes the (i, j) element
        // of this primitive matrix
        T value(int i, int j) {
            if (!set(i, j)) {
                matrix(i, j) = (*setter)(G, i, j);
                set(i, j) = true;
            }
            return matrix(i, j);
        }

    };

    // Compute the primitive integral helper matrix H
    template <typename T>
    T computeH(Greens<T>& G, int i, int j) {
        return G.b; // debug
    }

    // Compute the primitive integral helper matrix I
    template <typename T>
    T computeI(Greens<T>& G, int i, int j) {
        return G.b; // debug
    }

    // Compute the primitive integral helper matrix J
    template <typename T>
    T computeJ(Greens<T>& G, int i, int j) {
        return G.b; // debug
    }

    // Compute the primitive integral helper matrix M
    template <typename T>
    T computeM(Greens<T>& G, int i, int j) {
        return G.b; // debug
    }

    // Greens integration housekeeping data
    template <class T>
    class Greens {

        int lmax;

    public:

        int l;
        int m;
        int mu;
        int nu;

        T b;
        T b2;
        T br;
        T br32;
        T ksq;
        T k;
        T E;
        T K;
        T PI;
        T E1;
        T E2;

        Vector<T> r;
        Vector<T> b_r;
        Vector<T> cosphi;
        Vector<T> sinphi;
        Vector<T> coslam;
        Vector<T> sinlam;

        Primitive<T> H;
        Primitive<T> I;
        Primitive<T> J;
        Primitive<T> M;

        // Constructor
        Greens(int lmax) : lmax(lmax), H(*this, lmax, computeH), I(*this, lmax, computeI), J(*this, lmax, computeJ), M(*this, lmax, computeM) {

            // DEBUG
            b = 0.5;

            std::cout << I.value(2, 1) << std::endl;


            b = 0.75;

            std::cout << M.value(2, 1) << std::endl;

        }

    };


    // The surface map vector class
    template <class T>
    class Map {

        int N;
        Vector<T> basis;
        bool needs_update;

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
        Greens<T> G;

        // Constructor: initialize map to zeros
        Map(int lmax) : lmax(lmax), R(lmax), C(lmax), G(lmax) {
            N = (lmax + 1) * (lmax + 1);
            y = Vector<T>::Zero(N);
            p = Vector<T>::Zero(N);
            g = Vector<T>::Zero(N);
            basis.resize(N, 1);
            update(true);
        }
        // Constructor: initialize map to array
        Map(Vector<T>& y) : y(y), lmax(floor(sqrt((double)y.size()) - 1)), R(lmax), C(lmax), G(lmax) {
            N = (lmax + 1) * (lmax + 1);
            basis.resize(N, 1);
            update(true);
        }

        // Declare our methods
        T evaluate(const T& x, const T& y);
        void rotate(UnitVector<T>& u, T theta);
        void update(bool force=false);
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
    T Map<T>::evaluate(const T& x, const T& y) {

        // Update maps if necessary
        update();

        // Check if outside the sphere
        if (x * x + y * y > 1.0) return NAN;

        int l, m, mu, nu, n = 0;
        T z = sqrt(1.0 - x * x - y * y);

        // Compute the polynomial basis
        for (l=0; l<lmax+1; l++) {
            for (m=-l; m<l+1; m++) {
                mu = l - m;
                nu = l + m;
                if ((nu % 2) == 0)
                    basis(n) = pow(x, mu / 2) * pow(y, nu / 2);
                else
                    basis(n) = pow(x, (mu - 1) / 2) * pow(y, (nu - 1) / 2) * z;
                n++;
            }
        }

        // Dot the coefficients in to our polynomial map
        return p.dot(basis);

    }

    // Rotate the map in-place
    template <class T>
    void Map<T>::rotate(UnitVector<T>& u, T theta) {

        // Compute the rotation matrix R
        rotation::computeR(lmax, u, cos(theta), sin(theta), R.Complex, R.Real);

        // Dot R in, order by order
        for (int l = 0; l < lmax + 1; l++) {
            y.segment(l * l, 2 * l + 1) = R.Real[l] * y.segment(l * l, 2 * l + 1);
        }

        // Flag for updating
        needs_update = true;

        return;

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
