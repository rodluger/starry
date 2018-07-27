/**
Defines functions used to find the minimum of a map.

*/

#ifndef _STARRY_MIN_H_
#define _STARRY_MIN_H_

#include <cmath>
#include <Eigen/Core>
#include <LBFGS.h>
#include "errors.h"
#include "utils.h"

// Forward declaration
namespace maps {
    template <class T>
    class Map;
}

namespace minimize {

    using namespace LBFGSpp;
    using std::abs;

    // Misc stuff for fast map minimization
    template <class T>
    class Minimizer {

        public:

            maps::Map<T>& map;
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

            // Determine if the map is positive semi-definite
            bool psd(double epsilon=1e-6, int max_iterations=100) {

                // Do a coarse grid search for the global minimum
                angles(0) = 0;
                angles(1) = 0;
                minimum = evaluate(angles(0), angles(1));
                for (int u = 0; u < npts; u++) {
                    for (int v = 0; v < npts; v++) {
                        val = evaluate(theta(u), phi(v));
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

            // Evaluate a map at a given (theta, phi) coordinate (no rotation)
            T evaluate(const T& theta, const T& phi) {
                T sint = sin(theta),
                  cost = cos(theta),
                  sinp = sin(phi),
                  cosp = cos(phi);
                int l, m, mu, nu, n = 0;
                T x0 = sint * cosp;
                T y0 = sint * sinp;
                T z0 = cost;
                T res = 0;
                for (l = 0; l < map.lmax + 1; l++) {
                    for (m = -l; m < l + 1; m++) {
                        if (abs(map.p(n)) > 10 * mach_eps<T>()) {
                            mu = l - m;
                            nu = l + m;
                            if ((nu % 2) == 0) {
                                if ((mu > 0) && (nu > 0))
                                    res += map.p(n) * pow(x0, mu / 2) * pow(y0, nu / 2);
                                else if (mu > 0)
                                    res += map.p(n) * pow(x0, mu / 2);
                                else if (nu > 0)
                                    res += map.p(n) * pow(y0, nu / 2);
                                else
                                    res += map.p(n);
                            } else {
                                if ((mu > 1) && (nu > 1))
                                    res += map.p(n) * pow(x0, (mu - 1) / 2) * pow(y0, (nu - 1) / 2) * z0;
                                else if (mu > 1)
                                    res += map.p(n) * pow(x0, (mu - 1) / 2) * z0;
                                else if (nu > 1)
                                    res += map.p(n) * pow(y0, (nu - 1) / 2) * z0;
                                else
                                    res += map.p(n) * z0;
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
            T operator()(const Vector<T>& angles, Vector<T>& grad) {
                T theta = mod2pi(angles(0)),
                  phi = mod2pi(angles(1));

                // Avoid singular points in the derivative. In principle we could
                // re-parametrize, but it's not really worth it...
                T tol = 1e-12;
                if (abs(theta) < tol) theta = tol;
                if (abs(theta - PI<T>()) < tol) theta = PI<T>() + tol;
                if (abs(phi) < tol) phi = tol;
                if (abs(phi - PI<T>()) < tol) phi = PI<T>() + tol;
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
                for (l = 0; l < map.lmax + 1; l++) {
                    for (m = -l; m < l + 1; m++) {
                        if (abs(map.p(n)) > 10 * mach_eps<T>()) {
                            mu = l - m;
                            nu = l + m;
                            if ((nu % 2) == 0) {
                                if ((mu > 0) && (nu > 0)) {
                                    val = map.p(n) * pow(x0, mu / 2) * pow(y0, nu / 2);
                                    I += val;
                                    dIdt += 0.5 * (mu + nu) / tant * val;
                                    dIdp += 0.5 * (nu / tanp - mu * tanp) * val;
                                } else if (mu > 0) {
                                    val = map.p(n) * pow(x0, mu / 2);
                                    I += val;
                                    dIdt += 0.5 * mu / tant * val;
                                    dIdp += 0.5 * (-mu * tanp) * val;
                                } else if (nu > 0) {
                                    val = map.p(n) * pow(y0, nu / 2);
                                    I += val;
                                    dIdt += 0.5 * nu / tant * val;
                                    dIdp += 0.5 * (nu / tanp) * val;
                                } else {
                                    val = map.p(n);
                                    I += val;
                                    dIdt += 0;
                                    dIdp += 0;
                                }
                            } else {
                                if ((mu > 1) && (nu > 1)) {
                                    val = map.p(n) * pow(x0, (mu - 1) / 2) * pow(y0, (nu - 1) / 2) * z0;
                                    I += val;
                                    dIdt += (0.5 * (-2 + mu + nu) / tant - tant) * val;
                                    dIdp += 0.5 * (1 - mu + (-1 + nu) / (tanp * tanp)) * tanp * val;
                                } else if (mu > 1) {
                                    val = map.p(n) * pow(x0, (mu - 1) / 2) * z0;
                                    I += val;
                                    dIdt += (0.5 * (-1 + mu) / tant - tant) * val;
                                    dIdp += -0.5 * (-1 + mu) * tanp * val;
                                } else if (nu > 1) {
                                    val = map.p(n) * pow(y0, (nu - 1) / 2) * z0;
                                    I += val;
                                    dIdt += (0.5 * (-1 + nu) / tant - tant) * val;
                                    dIdp += 0.5 * (-1 + nu) / tanp * val;
                                } else {
                                    val = map.p(n) * z0;
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

            // Constructor: compute the matrices
            Minimizer(maps::Map<T>& map) : map(map), lmax(map.lmax), param(), solver(param), angles(Vector<T>::Zero(2)) {

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
                    phi(i) = 2.0 * PI<T>() * (T(i) / (npts + 1));
                }

                // The function wrapper
                functor = std::bind(*this, std::placeholders::_1, std::placeholders::_2);

            }

    };

}; // namespace minimize

#endif
