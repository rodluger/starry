/**
Defines functions used to find the minimum of a map.

TODO: The `pow()` calls in `evaluate()` and `operator()` are SUPER
      slow. Let's use the same trick as in `maps.h` to speed this up.

*/

#ifndef _STARRY_MIN_H_
#define _STARRY_MIN_H_

#include <cmath>
#include <Eigen/Core>
#include <LBFGS.h>
#include "errors.h"
#include "utils.h"

namespace minimize {

    using namespace utils;
    using namespace LBFGSpp;
    using std::abs;

    // Misc stuff for fast map minimization
    template <class T>
    class Minimizer {

        public:

            int lmax;
            int npts;
            Vector<Scalar<T>> theta;
            Vector<Scalar<T>> phi;
            LBFGSParam<Scalar<T>> param;
            LBFGSSolver<Scalar<T>> solver;
            Vector<Scalar<T>> angles;
            Scalar<T> minimum, val;
            int niter;
            Vector<Scalar<T>> p;

            // Determine if the map is positive semi-definite
            bool psd(const Vector<Scalar<T>>& p_new,
                     const Scalar<T>& epsilon=1e-6,
                     const int max_iterations=100) {

                // Update the polynomial map and bind the function
                p = p_new;
                std::function<Scalar<T> (const Vector<Scalar<T>>& angles,
                                         Vector<Scalar<T>>& grad)> functor =
                    std::bind(*this, std::placeholders::_1,
                                     std::placeholders::_2);

                // Do a coarse grid search for the global minimum
                angles(0) = 0;
                angles(1) = 0;
                minimum = evaluate(p, angles(0), angles(1));
                if (minimum < 0) return false;
                for (int u = 0; u < npts; u++) {
                    for (int v = 0; v < npts; v++) {
                        val = evaluate(p, theta(u), phi(v));
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
            Scalar<T> evaluate(const Vector<Scalar<T>>& p,
                               const Scalar<T>& theta, const Scalar<T>& phi) {
                Scalar<T> sint = sin(theta),
                          cost = cos(theta),
                          sinp = sin(phi),
                          cosp = cos(phi);
                int l, m, mu, nu, n = 0;
                Scalar<T> x0 = sint * cosp;
                Scalar<T> y0 = sint * sinp;
                Scalar<T> z0 = cost;
                Scalar<T> res = 0;
                for (l = 0; l < lmax + 1; l++) {
                    for (m = -l; m < l + 1; m++) {
                        if (abs(p(n)) > 10 * mach_eps<Scalar<T>>()) {
                            mu = l - m;
                            nu = l + m;
                            if ((nu % 2) == 0) {
                                if ((mu > 0) && (nu > 0))
                                    res += p(n) * pow(x0, mu / 2) *
                                           pow(y0, nu / 2);
                                else if (mu > 0)
                                    res += p(n) * pow(x0, mu / 2);
                                else if (nu > 0)
                                    res += p(n) * pow(y0, nu / 2);
                                else
                                    res += p(n);
                            } else {
                                if ((mu > 1) && (nu > 1))
                                    res += p(n) * pow(x0, (mu - 1) / 2) *
                                           pow(y0, (nu - 1) / 2) * z0;
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
            Scalar<T> operator()(const Vector<Scalar<T>>& angles,
                                 Vector<Scalar<T>>& grad) {

                // Ensure in range
                Scalar<T> theta = mod2pi(angles(0)),
                          phi = mod2pi(angles(1));

                // Avoid singular points in the derivative. In principle we
                // could re-parametrize, but it's not really worth it...
                Scalar<T> tol = 1e-12;
                if (abs(theta) < tol)
                    theta = tol;
                if (abs(theta - pi<Scalar<T>>()) < tol)
                    theta = pi<Scalar<T>>() + tol;
                if (abs(phi) < tol)
                    phi = tol;
                if (abs(phi - pi<Scalar<T>>()) < tol)
                    phi = pi<Scalar<T>>() + tol;
                Scalar<T> sint = sin(theta),
                          cost = cos(theta),
                          tant = tan(theta),
                          sinp = sin(phi),
                          cosp = cos(phi),
                          tanp = tan(phi);
                int l, m, mu, nu, n = 0;
                Scalar<T> x0 = sint * cosp;
                Scalar<T> y0 = sint * sinp;
                Scalar<T> z0 = cost;
                Scalar<T> I = 0;
                Scalar<T> dIdt = 0;
                Scalar<T> dIdp = 0;
                Scalar<T> val;
                for (l = 0; l < lmax + 1; l++) {
                    for (m = -l; m < l + 1; m++) {
                        if (abs(p(n)) > 10 * mach_eps<Scalar<T>>()) {
                            mu = l - m;
                            nu = l + m;
                            if ((nu % 2) == 0) {
                                if ((mu > 0) && (nu > 0)) {
                                    val = p(n) * pow(x0, mu / 2) *
                                          pow(y0, nu / 2);
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
                                    val = p(n) * pow(x0, (mu - 1) / 2) *
                                                 pow(y0, (nu - 1) / 2) * z0;
                                    I += val;
                                    dIdt += (0.5 * (-2 + mu + nu) / tant - tant)
                                            * val;
                                    dIdp += 0.5 * (1 - mu + (-1 + nu) /
                                                            (tanp * tanp)) *
                                                  tanp * val;
                                } else if (mu > 1) {
                                    val = p(n) * pow(x0, (mu - 1) / 2) * z0;
                                    I += val;
                                    dIdt += (0.5 * (-1 + mu) / tant - tant)
                                            * val;
                                    dIdp += -0.5 * (-1 + mu) * tanp * val;
                                } else if (nu > 1) {
                                    val = p(n) * pow(y0, (nu - 1) / 2) * z0;
                                    I += val;
                                    dIdt += (0.5 * (-1 + nu) / tant - tant)
                                            * val;
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

            // Constructor: compute the matrices
            explicit Minimizer(int lmax) : lmax(lmax), param(), solver(param),
                                           angles(Vector<Scalar<T>>::Zero(2)),
                                           p(Vector<Scalar<T>>::Zero((lmax + 1)
                                                                * (lmax + 1))){

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
                    theta(i) = acos(2.0 * (Scalar<T>(i) / (npts + 1)) - 1.0);
                    phi(i) = 2.0 * pi<Scalar<T>>() *
                                   (Scalar<T>(i) / (npts + 1));
                }

            }

    };

} // namespace minimize

#endif
