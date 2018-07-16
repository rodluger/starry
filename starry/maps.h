/**
Defines the surface map class.

*/

#ifndef _STARRY_MAPS_H_
#define _STARRY_MAPS_H_

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <LBFGS.h>
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
    using std::max;
    using std::string;
    using namespace LBFGSpp;

    // Forward declaration
    template <class T>
    class Map;

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

    // Misc stuff for fast map minimization
    template <class T>
    class Minimizer {

        public:

            Map<T>& map;
            int lmax;
            int npts;
            Vector<T> theta;
            Vector<T> phi;

            LBFGSParam<T> param;
            LBFGSSolver<T> solver;
            Vector<T> angles;
            std::function<T (const Vector<T>& angles, Vector<T>& grad)> functor;

            T minimum, val;
            int niter;

            // Constructor: compute the matrices
            Minimizer(Map<T>& map) : map(map), lmax(map.lmax), param(), solver(param), angles(Vector<T>::Zero(2)) {

                // A spherical harmonic of degree `l` has at most
                // `lmax^2 - lmax + 2` extrema
                // (http://adsabs.harvard.edu/abs/1992SvA....36..220K)
                // so let's have 4x this many points for good sampling
                // We will pick points on the sphere uniformly according
                // to http://mathworld.wolfram.com/SpherePointPicking.html
                npts = ceil(sqrt(4 * (lmax * lmax - lmax + 2)));
                theta.resize(npts);
                phi.resize(npts);
                for (int i = 0; i < npts; i++) {
                    theta(i) = acos(2.0 * (T(i) / (npts + 1)) - 1.0);
                    phi(i) = 2.0 * M_PI * (T(i) / (npts + 1));
                }

                // The function wrapper
                functor = std::bind(&Map<T>::objective, &map, std::placeholders::_1, std::placeholders::_2);

            }

            // Determine if the map is positive semi-definite
            bool psd(double epsilon=1e-6, int max_iterations=100) {

                // Do a coarse grid search for the global minimum
                angles(0) = 0;
                angles(1) = 0;
                minimum = map.evaluate(angles(0), angles(1));
                for (int u = 0; u < npts; u++) {
                    for (int v = 0; v < npts; v++) {
                        val = map.evaluate(theta(u), phi(v));
                        if (val < 0) {
                            // Our job is done!
                            return false;
                        } else if (val < minimum) {
                            minimum = val;
                            angles(0) = theta(u);
                            angles(1) = phi(v);
                        }
                    }
                }

                // Now refine it with gradient descent
                param.epsilon = epsilon;
                param.max_iterations = max_iterations;
                try {
                    niter = solver.minimize(functor, angles, minimum);
                } catch (const errors::MapIsNegative& e) {
                    return false;
                }
                if (minimum >= 0)
                    return true;
                else
                    return false;

            }

    };

    // No need to autodifferentiate these, since they are constant!
    template <>
    class Constants<Grad> {

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
            Constants(int lmax) : lmax(lmax) {
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
    // The surface map vector class
    //
    // ----------------------------
    // ****************************
    template <class T>
    class Map {

        protected:

            // Temporary variables
            Vector<T> tmpvec;
            VectorT<T> sTA;
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

            // Derivatives
            std::map<string, Vector<double>> derivs;
            Vector<T> dFdy;
            rotation::Wigner<T> RR;

            // Rotation matrices
            rotation::Wigner<T> R;

            // Constant matrices
            Constants<T> C;

            // Greens data
            solver::Greens<T> G;

            // Minimization stuff
            Minimizer<T> M;

            // Constructor: initialize map to zeros
            Map(int lmax=2) :
                  lmax(lmax), RR(lmax), R(lmax), C(lmax),
                  G(lmax), M(*this) {
                N = (lmax + 1) * (lmax + 1);
                y = Vector<T>::Zero(N);
                p = Vector<T>::Zero(N);
                g = Vector<T>::Zero(N);
                tmpvec = Vector<T>::Zero(N);
                sTA = VectorT<T>::Zero(N);
                ARRy = Vector<T>::Zero(N);
                dFdy = Vector<T>::Zero(N);
                tmpscalar = NAN;
                tmpu1 = 0;
                tmpu2 = 0;
                tmpu3 = 0;
                Y00_is_unity = false;
                update();
            }

            // Public methods
            T evaluate(const UnitVector<T>& axis=yhat, const T& theta=0, const T& x0=0, const T& y0=0);
            T evaluate(const T& theta=0, const T& phi=0);
            T objective(const Vector<T>& angles, Vector<T>& grad);
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
            T flux(const UnitVector<T>& axis=yhat, const T& theta=0, const T& xo=0, const T& yo=0, const T& ro=0);
            bool psd(double epsilon=1e-6, int max_iterations=100);
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
            yout.segment(l * l, 2 * l + 1) = R.Real[l] * yin.segment(l * l, 2 * l + 1);
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

        // Evaluate each harmonic
        T res = 0;
        for (l=0; l<lmax+1; l++) {
            for (m=-l; m<l+1; m++) {
                if (abs((*ptrmap)(n)) > 10 * mach_eps<T>()) {
                    mu = l - m;
                    nu = l + m;
                    if ((nu % 2) == 0) {
                        if ((mu > 0) && (nu > 0))
                            res += (*ptrmap)(n) * pow(x0, mu / 2) * pow(y0, nu / 2);
                        else if (mu > 0)
                            res += (*ptrmap)(n) * pow(x0, mu / 2);
                        else if (nu > 0)
                            res += (*ptrmap)(n) * pow(y0, nu / 2);
                        else
                            res += (*ptrmap)(n);
                    } else {
                        if ((mu > 1) && (nu > 1))
                            res += (*ptrmap)(n) * pow(x0, (mu - 1) / 2) * pow(y0, (nu - 1) / 2) * z0;
                        else if (mu > 1)
                            res += (*ptrmap)(n) * pow(x0, (mu - 1) / 2) * z0;
                        else if (nu > 1)
                            res += (*ptrmap)(n) * pow(y0, (nu - 1) / 2) * z0;
                        else
                            res += (*ptrmap)(n) * z0;
                    }
                }
                n++;
            }

        }
        return res;

    }

    // Evaluate our map at a given (x0, y0) coordinate
    template <>
    Grad Map<Grad>::evaluate(const UnitVector<Grad>& axis, const Grad& theta, const Grad& x0, const Grad& y0) {

        // Get the polynomial map
        Vector<Grad>* ptrmap;

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
        Grad z0 = sqrt(1.0 - x0 * x0 - y0 * y0);

        // Evaluate each harmonic
        VectorT<Grad> basis;
        basis.resize(N);
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

            sTA = basis * C.A1;
            for (int l = 0; l < lmax + 1; l++)
                dFdy.segment(l * l, 2 * l + 1) = sTA.segment(l * l, 2 * l + 1) * R.Real[l];

        }

        // Dot the coefficients in to our polynomial map
        return basis.dot(*ptrmap);

    }

    // Evaluate our map at a given (theta, phi) coordinate (no rotation)
    // Not user-facing; used exclusively for minimization of the map
    template <class T>
    T Map<T>::evaluate(const T& theta, const T& phi) {
        T sint = sin(theta),
          cost = cos(theta),
          sinp = sin(phi),
          cosp = cos(phi);
        int l, m, mu, nu, n = 0;
        T x0 = sint * cosp;
        T y0 = sint * sinp;
        T z0 = cost;
        T res = 0;
        for (l=0; l<lmax+1; l++) {
            for (m=-l; m<l+1; m++) {
                if (abs(p(n)) > 10 * mach_eps<T>()) {
                    mu = l - m;
                    nu = l + m;
                    if ((nu % 2) == 0) {
                        if ((mu > 0) && (nu > 0))
                            res += p(n) * pow(x0, mu / 2) * pow(y0, nu / 2);
                        else if (mu > 0)
                            res += p(n) * pow(x0, mu / 2);
                        else if (nu > 0)
                            res += p(n) * pow(y0, nu / 2);
                        else
                            res += p(n);
                    } else {
                        if ((mu > 1) && (nu > 1))
                            res += p(n) * pow(x0, (mu - 1) / 2) * pow(y0, (nu - 1) / 2) * z0;
                        else if (mu > 1)
                            res += p(n) * pow(x0, (mu - 1) / 2) * z0;
                        else if (nu > 1)
                            res += p(n) * pow(y0, (nu - 1) / 2) * z0;
                        else
                            res += p(n) * z0;
                    }
                }
                n++;
            }

        }
        return res;
    }

    // Computes the specific intensity as a function of theta and phi.
    // Used as the objective function in the minimization problem to
    // determine if a map is positive semi-definite.
    template <class T>
    T Map<T>::objective(const Vector<T>& angles, Vector<T>& grad) {
        T theta = mod2pi(angles(0)),
          phi = mod2pi(angles(1));

        // Avoid singular points in the derivative. In principle we could
        // re-parametrize, but it's not really worth it...
        T tol = 1e-12;
        if (abs(theta) < tol) theta = tol;
        if (abs(theta - M_PI) < tol) theta = M_PI + tol;
        if (abs(phi) < tol) phi = tol;
        if (abs(phi - M_PI) < tol) phi = M_PI + tol;
        T sint = sin(theta),
          cost = cos(theta),
          tant = tan(theta),
          sinp = sin(phi),
          cosp = cos(phi),
          tanp = tan(phi);
        int l, m, mu, nu, n = 0;
        T x0 = sint * cosp;
        T y0 = sint * sinp;
        T z0 = cost;
        T I = 0;
        T dIdt = 0;
        T dIdp = 0;
        T val;
        for (l=0; l<lmax+1; l++) {
            for (m=-l; m<l+1; m++) {
                if (abs(p(n)) > 10 * mach_eps<T>()) {
                    mu = l - m;
                    nu = l + m;
                    if ((nu % 2) == 0) {
                        if ((mu > 0) && (nu > 0)) {
                            val = p(n) * pow(x0, mu / 2) * pow(y0, nu / 2);
                            I += val;
                            dIdt += 0.5 * (mu + nu) / tant * val;
                            dIdp += 0.5 * (nu / tanp - mu * tanp) * val;
                        } else if (mu > 0) {
                            val = p(n) * pow(x0, mu / 2);
                            I += val;
                            dIdt += 0.5 * mu / tant * val;
                            dIdp += 0.5 * (-mu * tanp) * val;
                        } else if (nu > 0) {
                            val = p(n) * pow(y0, nu / 2);
                            I += val;
                            dIdt += 0.5 * nu / tant * val;
                            dIdp += 0.5 * (nu / tanp) * val;
                        } else {
                            val = p(n);
                            I += val;
                            dIdt += 0;
                            dIdp += 0;
                        }
                    } else {
                        if ((mu > 1) && (nu > 1)) {
                            val = p(n) * pow(x0, (mu - 1) / 2) * pow(y0, (nu - 1) / 2) * z0;
                            I += val;
                            dIdt += (0.5 * (-2 + mu + nu) / tant - tant) * val;
                            dIdp += 0.5 * (1 - mu + (-1 + nu) / (tanp * tanp)) * tanp * val;
                        } else if (mu > 1) {
                            val = p(n) * pow(x0, (mu - 1) / 2) * z0;
                            I += val;
                            dIdt += (0.5 * (-1 + mu) / tant - tant) * val;
                            dIdp += -0.5 * (-1 + mu) * tanp * val;
                        } else if (nu > 1) {
                            val = p(n) * pow(y0, (nu - 1) / 2) * z0;
                            I += val;
                            dIdt += (0.5 * (-1 + nu) / tant - tant) * val;
                            dIdp += 0.5 * (-1 + nu) / tanp * val;
                        } else {
                            val = p(n) * z0;
                            I += val;
                            dIdt += -tant * val;
                            dIdp += 0;
                        }
                    }
                }
                n++;
            }
        }

        // Throw an exception if the map is negative; this
        // will be caught in the enclosing scope
        if (I < 0) throw errors::MapIsNegative();

        // Update the gradient
        grad(0) = dIdt;
        grad(1) = dIdp;
        return I;

    }

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

    // Compute the total flux during or outside of an occultation
    // **Gradient over-ride: compute map derivs manually for speed**
    template <>
    Grad Map<Grad>::flux(const UnitVector<Grad>& axis, const Grad& theta, const Grad& xo, const Grad& yo_, const Grad& ro) {

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
            dFdy = Vector<Grad>::Zero(N);
            return 0;
        }

        // Pointer to the map we're integrating
        // (defaults to the base map)
        Vector<Grad>* ptry = &y;

        // Rotate the map into view if necessary and update our pointer
        if (theta != 0) {
            rotate(axis, theta, (*ptry), tmpvec);
            ptry = &tmpvec;
            for (int l = 0; l < lmax + 1; l++)
                RR.Real[l] = R.Real[l];
        } else {
            for (int l = 0; l < lmax + 1; l++)
                RR.Real[l] = Matrix<Grad>::Identity(2 * l + 1, 2 * l + 1);
        }

        // No occultation: cake
        if ((b >= 1 + ro) || (ro == 0)) {

            // Compute map derivs
            for (int l = 0; l < lmax + 1; l++)
                dFdy.segment(l * l, 2 * l + 1) = C.rTA1.segment(l * l, 2 * l + 1) * RR.Real[l];

            return C.rTA1 * (*ptry);

        // Occultation
        } else {

            // Align occultor with the +y axis if necessary
            if ((b > 0) && (xo != 0)) {
                UnitVector<Grad> zaxis({0, 0, 1});
                Grad yo_b(yo / b);
                Grad xo_b(xo / b);
                rotate(zaxis, yo_b, xo_b, (*ptry), tmpvec);
                ptry = &tmpvec;

                // Update the rotation matrix for the map derivs
                for (int l = 0; l < lmax + 1; l++)
                    RR.Real[l] = R.Real[l] * RR.Real[l];

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
            Grad res = G.sT * ARRy;

            return res;

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
            Constants<T> C;

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
            void set_coeff(int l, T coeff);
            T get_coeff(int l);
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
                norm -= 2.0 * u(l) / ((l + 1) * (l + 2));
            norm *= M_PI;
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
        norm *= M_PI;
        dndu *= M_PI;
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
