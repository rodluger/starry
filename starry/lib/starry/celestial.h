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
using maps::Map;
using std::vector;

namespace celestial {

    template <class T> class Primary;
    template <class T> class Secondary;
    template <class T> class System;

    // System class
    template <class T>
    class System {

        public:

            Primary<T>& primary;
            vector<Secondary<T>*> secondaries;

            // Constructor
            System(Primary<T>& primary, vector<Secondary<T>*> secondaries) :
                primary(primary), secondaries(secondaries) {

            }

            // Methods
            void compute(const Vector<T>& time);

    };

    // Compute the light curve
    template <class T>
    void System<T>::compute(const Vector<T>& time) {

        // Allocate arrays
        int N = time.size();
        primary.x.resize(N);
        primary.y.resize(N);
        primary.z.resize(N);
        primary.flux.resize(N);
        for (int i = 0; i < secondaries.size(); i++) {
            secondaries[i]->x.resize(N);
            secondaries[i]->y.resize(N);
            secondaries[i]->z.resize(N);
            secondaries[i]->flux.resize(N);
        }

        // Loop through the timeseries
        int i;
        for (int n = 0; n < N; n++){

            // Take an orbital step
            for (i = 0; i < secondaries.size(); i++)
                secondaries[i]->Kepler(time(n), n);

            // TODO: flux

        }

    }

    // Primary body (star) class
    template <class T>
    class Primary {

        public:

            T radius;
            const Map<T>& map;
            T mass;

            // Orbital position
            Vector<T> x;
            Vector<T> y;
            Vector<T> z;

            // Flux
            Vector<T> flux;

            // Constructor
            Primary(const T& radius, const Map<T>& map, const T& mass) :
                radius(radius), map(map), mass(mass) {

            }

    };

    // Secondary body (planet) class
    template <class T>
    class Secondary {

            // Reference to the primary body
            const Primary<T>& primary;

            // Size of arrays
            int N;

            // Orbital solution variables
            int iter;
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
            T aoneminuse2;
            T cosOcosi;
            T sinOcosi;

        public:

            // Map
            T radius;
            const Map<T>& map;

            // Orbital elements
            T a;
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
            Secondary(const T& radius, const Map<T>& map,
                 const Primary<T>& primary,
                 const T& per, const T& inc=0.5 * M_PI, const T& ecc=0,
                 const T& w=0, const T& Omega=0, const T& lambda0=0, const T& tref=0,
                 const double& eps=1.0e-7, const int& maxiter=100) :
                 primary(primary),
                 radius(radius), map(map),
                 per(per), inc(inc), ecc(ecc), w(w), Omega(Omega),
                 lambda0(lambda0), tref(tref),
                 eps(eps), maxiter(maxiter) {

                // Compute the semi-major axis
                a = (per * per) * (BIGG * primary.mass) / (4 * M_PI * M_PI);

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
                aoneminuse2 = a * (1 - ecc * ecc);

            }

            // Methods
            void computeM(const T& time);
            void computeE();
            void computef();
            void Kepler(const T& time, const int& i);

    };


    // Compute the mean anomaly
    template <class T>
    void Secondary<T>::computeM(const T& time) {
        M = M0 + 2 * M_PI / per * (time - tref);
        M = fmod(M, 2 * M_PI);
    }

    // Compute the eccentric anomaly. Adapted from
    // https://github.com/lkreidberg/batman/blob/master/c_src/_rsky.c
    template <class T>
    void Secondary<T>::computeE() {
        // Initial condition
        E = M;

        // The trivial circular case
        if (ecc == 0.) return;

        // Iterate
        for (iter = 0; iter <= maxiter; iter++) {
            E = E - (E - ecc * sin(E) - M) / (1. - ecc * cos(E));
            if (fabs(E - ecc * sin(E) - M) <= eps) return;
        }

        // Didn't converge: set the flag
        iErr = STARRY_ERR_KEPLER_MAXITER;
        return;
    }

    // Compute the true anomaly
    template <class T>
    void Secondary<T>::computef() {
        if (ecc == 0.) f = E;
        else f = 2. * atan2(sqrtonepluse * sin(E / 2.),
                            sqrtoneminuse * cos(E / 2.));
    }

    // Compute the instantaneous x, y, and z positions of the
    // secondary with a simple Keplerian solver.
    template <class T>
    void Secondary<T>::Kepler(const T& time, const int& i){

        // Mean anomaly
        computeM(time);

        // Eccentric anomaly
        computeE();

        // True anomaly
        computef();

        // Orbital radius
        rorb = aoneminuse2 / (1. + ecc * cos(f));

        // Murray and Dermott p. 51
        cwf = cos(w + f);
        swf = sin(w + f);
        x(i) = rorb * (cosO * cwf - sinOcosi * swf);
        y(i) = rorb * (sinO * cwf + cosOcosi * swf);
        z(i) = rorb * swf * sini;
        return;

    }

    // DEBUG! Let's test this thing.
    Vector<double> test(){

        // M dwarf
        Map<double> map(2);
        map.limbdark(0.40, 0.26);
        Primary<double> star(0.1 * RSUN, map, 0.1 * MSUN);

        // Hot Jupiter, 1 day period
        map.reset();
        map.set_coeff(0, 0, 1);
        map.set_coeff(1, 0, 1);
        Secondary<double> hot_jupiter(12 * REARTH, map, star, 1.0);

        // Hot Saturn, 2 day period
        map.reset();
        map.set_coeff(0, 0, 1);
        map.set_coeff(1, 0, 1);
        Secondary<double> hot_saturn(8 * REARTH, map, star, 2.0);

        // The vector of planet pointers
        vector<Secondary<double>*> planets;
        planets.push_back(&hot_jupiter);
        planets.push_back(&hot_saturn);

        // The system
        System<double> system(star, planets);

        // The time array
        Vector<double> time = Vector<double>::LinSpaced(100, 0., 2.);
        system.compute(time);

        return hot_jupiter.x;

    }

}; // namespace celestial

#endif
