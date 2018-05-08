/**
Orbital star/planet/moon system class.

*/

#ifndef _STARRY_ORBITAL_H_
#define _STARRY_ORBITAL_H_

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include "constants.h"
#include "errors.h"
#include "maps.h"

// Shorthand
template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T>
using UnitVector = Eigen::Matrix<T, 3, 1>;
using maps::Map;
using maps::LimbDarkenedMap;
using maps::yhat;
using std::vector;
using std::max;
using std::abs;

namespace orbital {

    template <class T> class Body;
    template <class T> class System;
    template <class T> class Star;
    template <class T> class Planet;

    // Re-definition of fmod so we can define its derivative below
    double fmod(double numer, double denom) {
        return std::fmod(numer, denom);
    }

    // Derivative of the floating point modulo function,
    // based on https://math.stackexchange.com/a/1277049
    template <typename T>
    Eigen::AutoDiffScalar<T> fmod(const Eigen::AutoDiffScalar<T>& numer, const Eigen::AutoDiffScalar<T>& denom) {
        typename T::Scalar numer_value = numer.value(),
                           denom_value = denom.value(),
                           modulo_value = fmod(numer_value, denom_value);
        return Eigen::AutoDiffScalar<T>(
          modulo_value,
          numer.derivatives() +
          denom.derivatives() * (modulo_value - numer_value) / denom_value
        );
    }

    // Compute the eccentric anomaly. Adapted from
    // https://github.com/lkreidberg/batman/blob/master/c_src/_rsky.c
    double EccentricAnomaly(double& M, double& ecc, const double& eps, const int& maxiter) {
        // Initial condition
        double E = M;
        if (ecc > 0) {
            // Iterate
            for (int iter = 0; iter <= maxiter; iter++) {
                E = E - (E - ecc * sin(E) - M) / (1. - ecc * cos(E));
                if (abs(E - ecc * sin(E) - M) <= eps) return E;
            }
            // Didn't converge!
            throw errors::Kepler();
        }
        return E;
    }

    // Derivative of the eccentric anomaly
    template <typename T>
    Eigen::AutoDiffScalar<T> EccentricAnomaly(const Eigen::AutoDiffScalar<T>& M, const Eigen::AutoDiffScalar<T>& ecc, const double& eps, const int& maxiter) {
        typename T::Scalar M_value = M.value(),
                           ecc_value = ecc.value(),
                           E_value = EccentricAnomaly(M_value, ecc_value, eps, maxiter),
                           cosE_value = cos(E_value),
                           sinE_value = sin(E_value);
        return Eigen::AutoDiffScalar<T>(
          E_value,
          M.derivatives() / (1. - ecc_value * cosE_value) +
          ecc.derivatives() * sinE_value / (1. - ecc_value * cosE_value)
        );
    }

    // System class
    template <class T>
    class System {

        public:

            vector<Body<T>*> bodies;
            Vector<T> flux;
            double eps;
            int maxiter;
            bool computed;

            // Constructor
            System(vector<Body<T>*> bodies, const double& eps=1.0e-7, const int& maxiter=100) :
                bodies(bodies), eps(eps), maxiter(maxiter) {

                // Check that we have at least one body
                if (bodies.size() == 0)
                    throw errors::BadSystem();

                // Check that first body (and only first body) is a star
                if (!bodies[0]->is_star)
                    throw errors::BadSystem();

                // Propagate settings down
                for (int i = 1; i < bodies.size(); i++) {
                    if (bodies[i]->is_star)
                        throw errors::BadSystem();
                    bodies[i]->eps = eps;
                    bodies[i]->maxiter = maxiter;
                }

                // Set the flag
                computed = false;

            }

            // Methods
            void compute(const Vector<T>& time);

    };

    // Compute the light curve
    template <class T>
    void System<T>::compute(const Vector<T>& time) {

        int i, j, t;
        T xo, yo, ro;
        T tsec;
        int p, o;
        int NT = time.size();

        // Allocate arrays and check that the planet maps are physical
        for (i = 0; i < bodies.size(); i++) {
            bodies[i]->x.resize(NT);
            bodies[i]->y.resize(NT);
            bodies[i]->z.resize(NT);
            bodies[i]->flux.resize(NT);
        }

        // Loop through the timeseries
        for (t = 0; t < NT; t++){

            // Time in seconds
            tsec = time(t) * DAY;

            // Take an orbital step
            for (i = 0; i < bodies.size(); i++)
                bodies[i]->step(tsec, t);

            // Compute any occultations
            for (i = 0; i < bodies.size(); i++) {
                for (j = i + 1; j < bodies.size(); j++) {
                    // Determine the relative positions of the two bodies
                    if (bodies[j]->z(t) > bodies[i]->z(t)) {
                        o = j;
                        p = i;
                    } else {
                        o = i;
                        p = j;
                    }
                    xo = (bodies[o]->x(t) - bodies[p]->x(t)) / bodies[p]->r;
                    yo = (bodies[o]->y(t) - bodies[p]->y(t)) / bodies[p]->r;
                    ro = bodies[o]->r / bodies[p]->r;
                    // Compute the flux in occultation
                    if (sqrt(xo * xo + yo * yo) < 1 + ro) {
                        bodies[p]->getflux(tsec, t, xo, yo, ro);
                    }
                }
            }

        }

        // Add up all the fluxes
        flux = Vector<T>::Zero(NT);
        for (i = 0; i < bodies.size(); i++) {
            flux += bodies[i]->flux;
        }

        // Set the flag
        computed = true;

    }

    // Body class
    template <class T>
    class Body {

        protected:

            // Orbital solution variables
            T M;
            T E;
            T f;
            T rorb;
            T cwf;
            T swf;

            // Commonly used variables
            T M0;
            T cosi;
            T sini;
            T cosO;
            T sinO;
            T sqrtonepluse;
            T sqrtoneminuse;
            T ecc2;
            T cosOcosi;
            T sinOcosi;
            T angvelorb;
            T angvelrot;

            // Total flux at current timestep
            T totalflux;
            T norm;

        public:

            // Flag
            bool is_star;

            // Map stuff
            int lmax;
            UnitVector<T> axis;
            T prot;
            T theta0;
            T r;
            T L;
            Map<T> map;
            LimbDarkenedMap<T> ldmap;

            // Orbital elements
            T a;
            T porb;
            T inc;
            T ecc;
            T w;
            T Omega;
            T lambda0;
            T tref;

            // Settings
            double eps;
            int maxiter;

            // Orbital position
            Vector<T> x;
            Vector<T> y;
            Vector<T> z;

            // Flux
            Vector<T> flux;

            // Constructor
            Body(// Map stuff
                 int lmax,
                 const T& r,
                 const T& L,
                 const UnitVector<T>& axis,
                 const T& prot,
                 const T& theta0,
                 // Orbital stuff
                 const T& a,
                 const T& porb,
                 const T& inc,
                 const T& ecc,
                 const T& w,
                 const T& Omega,
                 const T& lambda0,
                 const T& tref,
                 bool is_star) :
                 is_star(is_star),
                 lmax(lmax),
                 axis(axis),
                 prot(prot * DAY),
                 theta0(theta0 * DEGREE),
                 r(r),
                 L(L),
                 // Don't waste time allocating maps we won't use
                 map{is_star ? Map<T>(0) : Map<T>(lmax)},
                 ldmap{is_star ? LimbDarkenedMap<T>(lmax) : LimbDarkenedMap<T>(0)},
                 a(a),
                 porb(porb * DAY),
                 inc(inc * DEGREE),
                 ecc(ecc),
                 w(w * DEGREE),
                 Omega(Omega * DEGREE),
                 lambda0(lambda0 * DEGREE),
                 tref(tref * DAY)
                 {

                     // Initialize the map to constant surface brightness
                     if (!is_star) {
                         map.set_coeff(0, 0, 1);
                         map.Y00_is_unity = true;
                     }

                     // LimbDarkenedMaps are normalized to 1 by default,
                     // but regular Maps (used by Planets) are normalized
                     // to sqrt(pi) / 2. Let's take care of that here.
                     if (is_star)
                        norm = 1;
                     else
                        norm = T(2. / sqrt(M_PI));

                     // Initialize orbital vars
                     reset();
                 }

            // Reset orbital variables and map normalization
            // whenever the corresponding body parameters change
            void reset() {
                M0 = lambda0 - Omega - w;
                cosi = cos(inc);
                sini = sin(inc);
                cosO = cos(Omega);
                sinO = sin(Omega);
                cosOcosi = cosO * cosi;
                sinOcosi = sinO * cosi;
                sqrtonepluse = sqrt(1 + ecc);
                sqrtoneminuse = sqrt(1 - ecc);
                ecc2 = ecc * ecc;
                angvelorb = T(2 * M_PI) / porb;
                angvelrot = T(2 * M_PI) / prot;
            };

            // Public methods
            T theta(const T& time);
            void step(const T& time, const int& t);
            void getflux(const T& time, const int& t, const T& xo=0, const T& yo=0, const T& ro=0);
            std::string repr();
    };

    // Rotation angle as a function of time
    template <class T>
    inline T Body<T>::theta(const T& time) {
        if ((prot == 0) || (prot == INFINITY))
            return theta0;
        else
            return fmod(T(theta0 + angvelrot * (time - tref)), T(2 * M_PI));
    }

    // Compute the flux in occultation
    template <class T>
    inline void Body<T>::getflux(const T& time, const int& t, const T& xo, const T& yo, const T& ro){
        if (L != 0) {
            if (is_star)
                flux(t) += norm * ldmap.flux(xo, yo, ro) - totalflux;
            else
                flux(t) += norm * L * map.flux(axis, theta(time), xo, yo, ro) - totalflux;
        }
    }

    // Compute the instantaneous x, y, and z positions of the
    // body with a simple Keplerian solver.
    template <class T>
    inline void Body<T>::step(const T& time, const int& t){

        // Primary is fixed at the origin in the Keplerian solver
        if (is_star) {
            x(t) = 0;
            y(t) = 0;
            z(t) = 0;
        } else {

            // Mean anomaly
            M = fmod(T(M0 + angvelorb * (time - tref)), T(2 * M_PI));

            // Eccentric anomaly
            E = EccentricAnomaly(M, ecc, eps, maxiter);

            // True anomaly
            if (ecc == 0) f = E;
            else f = 2. * atan2(sqrtonepluse * sin(E / 2.),
                                sqrtoneminuse * cos(E / 2.));

            // Orbital radius
            if (ecc > 0)
                rorb = a * (1 - ecc2) / (1. + ecc * cos(f));
            else
                rorb = a;

            // Murray and Dermott p. 51
            cwf = cos(w + f);
            swf = sin(w + f);
            x(t) = rorb * (cosO * cwf - sinOcosi * swf);
            y(t) = rorb * (sinO * cwf + cosOcosi * swf);
            z(t) = rorb * swf * sini;

        }

        // Compute total flux this timestep
        if (L == 0) {
            totalflux = 0;
        } else {
            if (is_star)
                totalflux = norm * ldmap.flux();
            else {
                T theta_time(theta(time));
                totalflux = norm * L * map.flux(axis, theta_time);
            }
        }
        flux(t) = totalflux;

        return;

    }

    // Return a human-readable string
    template <class T>
    std::string Body<T>::repr() {
        std::ostringstream os;
        os << "<STARRY Body>";
        return std::string(os.str());
    }

    // Star, Body subclass
    template <class T>
    class Star : public Body<T> {
        public:
            Star(int lmax=2) :
                 Body<T>(lmax, 1, 1, yhat, INFINITY, 0,
                         0, INFINITY, 0, 0, 0, 0, 0, 0, true) {
            }
        std::string repr();
    };

    // Return a human-readable string
    template <class T>
    std::string Star<T>::repr() {
        std::ostringstream os;
        os << "<STARRY Star>";
        return std::string(os.str());
    }

    // Planet, Body subclass
    template <class T>
    class Planet : public Body<T> {
        public:
            Planet(int lmax=2,
                   const T& r=0.1,
                   const T& L=0.,
                   const UnitVector<T>& axis=yhat,
                   const T& prot=0.,
                   const T& theta0=0,
                   const T& a=50.,
                   const T& porb=1.,
                   const T& inc=90.,
                   const T& ecc=0.,
                   const T& w=90.,
                   const T& Omega=0.,
                   const T& lambda0=90.,
                   const T& tref=0.) :
                   Body<T>(lmax, r, L, axis, prot,
                           theta0, a, porb, inc,
                           ecc, w, Omega, lambda0, tref,
                           false) {
            }
            std::string repr();
    };

    // Return a human-readable string
    template <class T>
    std::string Planet<T>::repr() {
        std::ostringstream os;
        os << "<STARRY Planet at P = " << std::setprecision(3) << get_value(this->porb) / DAY << " days>";
        return std::string(os.str());
    }

}; // namespace orbital

#endif
