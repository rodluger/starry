/**
Celestial star/planet/moon system class.

*/

#ifndef _STARRY_CELESTIAL_H_
#define _STARRY_CELESTIAL_H_

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include "constants.h"
#include "maps.h"

// Shorthand
template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T>
using UnitVector = Eigen::Matrix<T, 3, 1>;
using maps::Map;
using maps::yhat;
using std::vector;

namespace celestial {

    template <class T> class Body;
    template <class T> class System;

    // System class
    template <class T>
    class System {

        public:

            vector<Body<T>*> bodies;
            Vector<T> flux;
            double eps;
            int maxiter;

            // Constructor
            System(vector<Body<T>*> bodies, const double& eps=1.0e-7, const int& maxiter=100) :
                bodies(bodies), eps(eps), maxiter(maxiter) {

                // Propagate settings down
                for (int i = 1; i < bodies.size(); i++) {
                    bodies[i]->eps = eps;
                    bodies[i]->maxiter = maxiter;
                }

                // Set the flag for the primary
                bodies[0]->is_primary = true;

                // Compute the semi-major axes of each planet
                for (int i = 1; i < bodies.size(); i++) {
                    bodies[i]->is_primary = false;
                    bodies[i]->a = pow((bodies[i]->per * bodies[i]->per) *
                                       (BIGG * bodies[0]->mass) /
                                       (4 * M_PI * M_PI), (1. / 3.));
                }

            }

            // Methods
            void compute(const Vector<T>& time);

    };

    // Compute the light curve
    template <class T>
    void System<T>::compute(const Vector<T>& time) {

        int i, j, t;
        double xo, yo, ro;
        int p, o;
        int NT = time.size();

        // Allocate arrays
        for (i = 0; i < bodies.size(); i++) {
            bodies[i]->x.resize(NT);
            bodies[i]->y.resize(NT);
            bodies[i]->z.resize(NT);
            bodies[i]->flux.resize(NT);
        }

        // Loop through the timeseries
        for (t = 0; t < NT; t++){

            // Take an orbital step
            for (i = 0; i < bodies.size(); i++) {
                bodies[i]->step(time(t), t);
                bodies[i]->computed = false;
            }

            // Find occultations and compute the flux
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
                    xo = (bodies[o]->x(t) - bodies[p]->x(t)) / bodies[p]->radius;
                    yo = (bodies[o]->y(t) - bodies[p]->y(t)) / bodies[p]->radius;
                    ro = bodies[o]->radius / bodies[p]->radius;
                    // Compute the flux in occultation
                    if (sqrt(xo * xo + yo * yo) < 1 + ro) {
                        bodies[p]->getflux(time(t), t, xo, yo, ro);
                        bodies[p]->computed = true;
                    }
                }
            }

            // Compute the total flux for the remaining bodies
            for (i = 0; i < bodies.size(); i++) {
                if (!bodies[i]->computed) {
                    bodies[i]->getflux(time(t), t);
                    bodies[i]->computed = true;
                }
            }

        }

        // Add up all the fluxes
        flux = Vector<T>::Zero(NT);
        for (i = 0; i < bodies.size(); i++) {
            flux += bodies[i]->flux;
        }

    }

    // Default theta function
    template <typename T>
    T NO_ROTATION(T theta) {
        return 0;
    }

    // Body class
    template <class T>
    class Body {

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

            // Methods
            void computeM(const T& time);
            void computeE();
            void computef();

        public:

            // Flag
            bool is_primary;
            bool computed;

            // Map stuff
            Map<T>& map;
            UnitVector<T>& u;
            T (*theta)(T);
            T radius;

            // Orbital elements
            T a;
            T mass;
            T per;
            T inc;
            T ecc;
            T w;
            T Omega;
            T lambda0;
            T tref;

            // Settings
            int iErr;
            double eps;
            int maxiter;

            // Orbital position
            Vector<T> x;
            Vector<T> y;
            Vector<T> z;

            // Flux
            Vector<T> flux;

            // Constructor
            Body(Map<T>& map,
                 UnitVector<T>& u=yhat,
                 T (*theta)(T)=NO_ROTATION,
                 const T& radius=REARTH,
                 const T& mass=MEARTH,
                 const T& per=DAYSEC,
                 const T& inc=0.5 * M_PI,
                 const T& ecc=0,
                 const T& w=0,
                 const T& Omega=0,
                 const T& lambda0=0,
                 const T& tref=0) :
                 map(map), u(u), theta(theta), radius(radius), mass(mass),
                 per(per), inc(inc), ecc(ecc), w(w), Omega(Omega),
                 lambda0(lambda0), tref(tref) {

                // Initialize variables
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

            }

            // Methods
            void step(const T& time, const int& t);
            void getflux(const T& time, const int& t, const T& xo=0, const T& yo=0, const T& ro=0);

    };

    // Compute the visible flux
    template <class T>
    void Body<T>::getflux(const T& time, const int& t, const T& xo, const T& yo, const T& ro){
        flux(t) = map.flux(u, theta(time), xo, yo, ro);
    }

    // Compute the mean anomaly
    template <class T>
    void Body<T>::computeM(const T& time) {
        M = M0 + 2 * M_PI / per * (time - tref);
        M = fmod(M, 2 * M_PI);
    }

    // Compute the eccentric anomaly. Adapted from
    // https://github.com/lkreidberg/batman/blob/master/c_src/_rsky.c
    template <class T>
    void Body<T>::computeE() {
        // Initial condition
        E = M;

        // The trivial circular case
        if (ecc == 0.) return;

        // Iterate
        for (int iter = 0; iter <= maxiter; iter++) {
            E = E - (E - ecc * sin(E) - M) / (1. - ecc * cos(E));
            if (fabs(E - ecc * sin(E) - M) <= eps) return;
        }

        // Didn't converge: set the flag
        iErr = STARRY_ERR_KEPLER_MAXITER;
        return;
    }

    // Compute the true anomaly
    template <class T>
    void Body<T>::computef() {
        if (ecc == 0.) f = E;
        else f = 2. * atan2(sqrtonepluse * sin(E / 2.),
                            sqrtoneminuse * cos(E / 2.));
    }

    // Compute the instantaneous x, y, and z positions of the
    // body with a simple Keplerian solver.
    template <class T>
    void Body<T>::step(const T& time, const int& t){

        // Primary is fixed at the origin in the Keplerian solver
        if (is_primary) {
            x(t) = 0;
            y(t) = 0;
            z(t) = 0;
            return;
        }

        // Mean anomaly
        computeM(time);

        // Eccentric anomaly
        computeE();

        // True anomaly
        computef();

        // Orbital radius
        rorb = a * (1 - ecc2) / (1. + ecc * cos(f));

        // Murray and Dermott p. 51
        cwf = cos(w + f);
        swf = sin(w + f);
        x(t) = rorb * (cosO * cwf - sinOcosi * swf);
        y(t) = rorb * (sinO * cwf + cosOcosi * swf);
        z(t) = rorb * swf * sini;

        return;

    }

    // DEBUG! Let's test this thing.
    Vector<double> test(){

        // M dwarf
        Map<double> map1(2);
        map1.limbdark(0.40, 0.26);
        Body<double> star(map1, yhat, NO_ROTATION, 0.1 * RSUN, 0.1 * MSUN, 0);

        // Hot Jupiter, 1 day period
        Map<double> map2(2);
        map2.set_coeff(0, 0, 1e-1);
        map2.set_coeff(1, 0, 1e-1);
        Body<double> hot_jupiter(map2, yhat, NO_ROTATION, 12 * REARTH, 0, DAYSEC);

        // Hot Saturn, 2 day period
        Map<double> map3(2);
        map3.set_coeff(0, 0, 1e-2);
        map3.set_coeff(1, 0, 1e-2);
        Body<double> hot_saturn(map3, yhat, NO_ROTATION, 8 * REARTH, 0, 2 * DAYSEC);

        // The vector of body pointers
        vector<Body<double>*> bodies;
        bodies.push_back(&star);
        bodies.push_back(&hot_jupiter);
        bodies.push_back(&hot_saturn);

        // The system
        System<double> system(bodies);

        // The time array
        Vector<double> time = Vector<double>::LinSpaced(10000, 0., 5. * DAYSEC);
        system.compute(time);

        return system.flux;

    }

}; // namespace celestial

#endif
