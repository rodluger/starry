/**
Orbital star/planet/moon system class.

*/

#ifndef _STARRY_ORBITAL_H_
#define _STARRY_ORBITAL_H_

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <string>
#include <vector>
#include "constants.h"
#include "errors.h"
#include "maps.h"
#include "utils.h"
#include "rotation.h"


namespace orbital {

    // Shorthand
    using maps::Map;
    using maps::LimbDarkenedMap;
    using std::vector;
    using std::max;
    using std::abs;
    using std::string;
    using std::to_string;
    using std::fmod;

    template <class T> class Body;
    template <class T> class System;
    template <class T> class Star;
    template <class T> class Planet;

    // Re-definition of fmod so we can define its derivative below
    template <typename T>
    T mod2pi(const T& numer) {
        return fmod(numer, T(2 * M_PI));
    }

    // Derivative of the floating point modulo function,
    // based on https://math.stackexchange.com/a/1277049
    template <typename T>
    Eigen::AutoDiffScalar<T> mod2pi(const Eigen::AutoDiffScalar<T>& numer) {
        typename T::Scalar numer_value = numer.value(),
                           modulo_value = mod2pi(numer_value);
        return Eigen::AutoDiffScalar<T>(
          modulo_value,
          numer.derivatives()
        );
    }

    // Compute the eccentric anomaly. Adapted from
    // https://github.com/lkreidberg/batman/blob/master/c_src/_rsky.c
    template <typename T>
    T EccentricAnomaly(T& M, T& ecc) {
        // Initial condition
        T E = M;
        T tol = 10 * mach_eps<T>();
        if (ecc > 0) {
            // Iterate
            for (int iter = 0; iter <= STARRY_KEPLER_MAX_ITER; iter++) {
                E = E - (E - ecc * sin(E) - M) / (1. - ecc * cos(E));
                if (abs(E - ecc * sin(E) - M) <= tol) return E;
            }
            // Didn't converge!
            throw errors::Kepler();
        }
        return E;
    }

    // Derivative of the eccentric anomaly
    template <typename T>
    Eigen::AutoDiffScalar<T> EccentricAnomaly(const Eigen::AutoDiffScalar<T>& M, const Eigen::AutoDiffScalar<T>& ecc) {
        typename T::Scalar M_value = M.value(),
                           ecc_value = ecc.value(),
                           E_value = EccentricAnomaly(M_value, ecc_value),
                           cosE_value = cos(E_value),
                           sinE_value = sin(E_value),
                           norm1 = 1./ (1. - ecc_value * cosE_value),
                           norm2 = sinE_value * norm1;
        if (M.derivatives().size() && ecc.derivatives().size())
            return Eigen::AutoDiffScalar<T>(E_value, M.derivatives() * norm1 + ecc.derivatives() * norm2);
        else if (M.derivatives().size())
            return Eigen::AutoDiffScalar<T>(E_value, M.derivatives() * norm1);
        else if (ecc.derivatives().size())
            return Eigen::AutoDiffScalar<T>(E_value, ecc.derivatives() * norm2);
        else
            return Eigen::AutoDiffScalar<T>(E_value, M.derivatives());
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
            T ecw;
            T esw;
            T angvelorb;
            T angvelrot;
            T vamp;
            T aamp;

            // Total flux at current timestep
            T totalflux;
            T norm;

        public:

            // Flag: is this a star?
            bool is_star;

            // Speed of light in units of Rstar/s
            T clight;

            // Map stuff
            int lmax;
            UnitVector<T> axis;
            T prot;
            T theta0;
            T r;
            T L;
            Map<T> map;
            LimbDarkenedMap<T> ldmap;

            // Axis of rotation and surface map
            // in the *sky* coordinates
            UnitVector<T> axis_sky;
            Map<T> map_sky;

            // Orbital elements
            T a;
            T porb;
            T inc;
            T ecc;
            T w;
            T Omega;
            T lambda0;
            T tref;

            // Derivatives dictionary
            std::map<string, Eigen::VectorXd> derivs;

            // Orbital position
            Vector<T> x;
            T x_;
            Vector<T> y;
            T y_;
            Vector<T> z;
            T z_;

            // Retarded position
            T z0;
            T vz_;
            T az_;
            T dt_;

            // Flux
            Vector<T> flux;
            T flux_;

            // Constructor
            Body(// Map stuff
                 int lmax,
                 const double& r,
                 const double& L,
                 const UnitVector<double>& axis,
                 const double& prot,
                 // Orbital stuff
                 const double& a,
                 const double& porb,
                 const double& inc,
                 const double& ecc,
                 const double& w,
                 const double& Omega,
                 const double& lambda0,
                 const double& tref,
                 bool is_star):
                 is_star(is_star),
                 lmax(lmax),
                 axis(norm_unit(axis).template cast<T>()),
                 prot(prot * DAY),
                 r(r),
                 L(L),
                 // Don't waste time allocating maps we won't use
                 map{is_star ? Map<T>(0) : Map<T>(lmax)},
                 ldmap{is_star ? LimbDarkenedMap<T>(lmax) : LimbDarkenedMap<T>(0)},
                 // Map in the sky coordinates
                 axis_sky(norm_unit(axis).template cast<T>()),
                 map_sky{is_star ? Map<T>(0) : Map<T>(lmax)},
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
                         map_sky.set_coeff(0, 0, 1);
                         map_sky.Y00_is_unity = true;
                     }

                     // LimbDarkenedMaps are normalized to 1 by default,
                     // but regular Maps (used by Planets) are normalized
                     // to sqrt(pi) / 2. Let's take care of that here.
                     if (is_star)
                        norm = 1;
                     else
                        norm = 2. / sqrt(M_PI);

                     // Initialize orbital vars
                     clight = INFINITY;
                     reset();
                     sync_maps();

                 }

            // Reset orbital variables and map normalization
            // whenever the corresponding body parameters change
            void reset() {
                M0 = lambda0 - w;
                cosi = cos(inc);
                sini = sin(inc);
                cosO = cos(Omega);
                sinO = sin(Omega);
                cosOcosi = cosO * cosi;
                sinOcosi = sinO * cosi;
                sqrtonepluse = sqrt(1 + ecc);
                sqrtoneminuse = sqrt(1 - ecc);
                ecc2 = ecc * ecc;
                ecw = ecc * cos(w);
                esw = ecc * sin(w);
                angvelorb = (2 * M_PI) / porb;
                angvelrot = (2 * M_PI) / prot;
                vamp = angvelorb * a / sqrt(1 - ecc2);

                // Light travel time delay parameters
                // `z0` is the reference point (the barycenter,
                // assuming massless planets).
                z0 = 0;
                dt_ = 0;

                // Initial map rotation angle. The map is defined at the
                // eclipsing configuration (full dayside as seen by an
                // observer viewing the system edge-on), so let's find the
                // angle by which we need to rotate the map initially to
                // make this happen.
                T f_eclipse = 1.5 * M_PI - w;
                T E_eclipse = atan2(sqrt(1 - ecc2) * sin(f_eclipse), ecc + cos(f_eclipse));
                T M_eclipse = E_eclipse - ecc * sin(E_eclipse);
                theta0 = -(porb / prot) * (M_eclipse - M0);

            };

            // Public methods
            void sync_maps();
            T theta(const T& time);
            void step(const T& time);
            void getflux(const T& time, const T& xo=0, const T& yo=0, const T& ro=0);
            std::string repr();
    };

    // Sync the map in the orbital plane (the user-facing one)
    // and the map in the sky plane (the one used internally to compute the flux)
    template <class T>
    inline void Body<T>::sync_maps() {
        if (!is_star){
            // Sync the two maps
            map_sky.y = map.y;
            map_sky.update();
            map.G.sT = map_sky.G.sT;
            map.C.rT = map_sky.C.rT;
            axis_sky = axis;
            // If there's inclination or rotation of the orbital plane,
            // we need to rotate the sky map as well as the rotation axis
            if ((Omega != 0) || (sini < 1. - 2 * mach_eps<T>())) {
                UnitVector<T> axis1 = xhat.template cast<T>();
                UnitVector<T> axis2 = zhat.template cast<T>();
                map_sky.rotate(axis1, M_PI_2 - inc);
                map_sky.rotate(axis2, Omega);
                axis_sky = rotation::AxisAngle(axis2, Omega) * (rotation::AxisAngle(axis1, T(M_PI_2 - inc)) * axis);
            }
        }
    }

    // Rotation angle as a function of (retarded) time
    template <class T>
    inline T Body<T>::theta(const T& time) {
        if ((prot == 0) || (prot == INFINITY))
            return theta0;
        else
            return mod2pi(T(theta0 + angvelrot * (time - tref - dt_)));
    }

    // Compute the flux in occultation
    template <class T>
    inline void Body<T>::getflux(const T& time, const T& xo, const T& yo, const T& ro){
        if (L != 0) {
            if (is_star)
                flux_ += norm * ldmap.flux(xo, yo, ro) - totalflux;
            else
                flux_ += norm * L * map_sky.flux(axis, theta(time), xo, yo, ro) - totalflux;
        }
    }

    // Compute the instantaneous x, y, and z positions of the
    // body with a simple Keplerian solver.
    template <class T>
    inline void Body<T>::step(const T& time){

        // Primary is fixed at the origin in the Keplerian solver
        if (is_star) {
            x_ = 0;
            y_ = 0;
            z_ = 0;
        } else {

            // Mean anomaly
            M = mod2pi(T(M0 + angvelorb * (time - tref)));

            // True anomaly and orbital radius
            if (ecc == 0) {
                f = M;
                rorb = a;
            } else {
                E = EccentricAnomaly(M, ecc);
                f = (2. * atan2(sqrtonepluse * sin(E / 2.),
                                 sqrtoneminuse * cos(E / 2.)));
                rorb = a * (1. - ecc2) / (1. + ecc * cos(f));
            }

            // Murray and Dermott p. 51, except x and y have the opposite sign
            // This makes the orbits prograde!
            cwf = cos(w + f);
            swf = sin(w + f);
            x_ = -rorb * (cosO * cwf - sinOcosi * swf);
            y_ = -rorb * (sinO * cwf + cosOcosi * swf);
            z_ = rorb * swf * sini;

            // Compute the light travel time delay
            if (!isinf(get_value(clight))) {

                // Component of the velocity out of the sky
                // Obtained by differentiating the expressions above
                vz_ = vamp * sini * (ecw + cwf);

                // Component of the acceleration out of the sky
                az_ = -angvelorb * angvelorb * a * a * a / (rorb * rorb * rorb) * z_;

                // Compute the time delay at the **retarded** position, accounting for
                // the instantaneous velocity and acceleration of the body.
                // This is slightly better than doing
                //
                //          dt_ = (z0 - z_) / clight
                //
                // which is actually the time delay at the **current** position.
                // But the photons left the planet from the **retarded** position,
                // so if the planet has motion in the `z` direction the two quantities
                // will be slightly different. In practice this doesn't matter too much,
                // though. See the derivation at https://github.com/rodluger/starry/issues/66
                if (abs(az_) < 1e-10)
                    dt_ = (z0 - z_) / (clight + vz_);
                else
                    dt_ = (clight / az_) *
                          ((1 + vz_ / clight)
                           - sqrt((1 + vz_ / clight) * (1 + vz_ / clight)
                                  - 2 * az_ * (z0 - z_) / (clight * clight)));

                // Re-compute Kepler's equation, this time solving for the **retarded** position
                M = mod2pi(T(M0 + angvelorb * (time - dt_ - tref)));
                if (ecc > 0) {
                    E = EccentricAnomaly(M, ecc);
                    f = (2. * atan2(sqrtonepluse * sin(E / 2.), sqrtoneminuse * cos(E / 2.)));
                    rorb = a * (1. - ecc2) / (1. + ecc * cos(f));
                } else {
                    f = M;
                    rorb = a;
                }
                cwf = cos(w + f);
                swf = sin(w + f);
                x_ = -rorb * (cosO * cwf - sinOcosi * swf);
                y_ = -rorb * (sinO * cwf + cosOcosi * swf);
                z_ = rorb * swf * sini;

            }

        }

        // Compute total flux this timestep
        if (L == 0) {
            totalflux = 0;
        } else {
            if (is_star)
                totalflux = norm * ldmap.flux();
            else {
                T theta_time(theta(time));
                totalflux = norm * L * map_sky.flux(axis, theta_time);
            }
        }
        flux_ = totalflux;

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
                         0, INFINITY, 0, 0, 0, 0, 0, true) {
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
                   const double& r=0.1,
                   const double& L=0.,
                   const UnitVector<double>& axis=yhat,
                   const double& prot=0.,
                   const double& a=50.,
                   const double& porb=1.,
                   const double& inc=90.,
                   const double& ecc=0.,
                   const double& w=90.,
                   const double& Omega=0.,
                   const double& lambda0=90.,
                   const double& tref=0.) :
                   Body<T>(lmax, r, L, axis, prot,
                           a, porb, inc,
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

    // System class
    template <class T>
    class System {

        public:

            vector<Body<T>*> bodies;
            Vector<T> flux;
            bool computed;
            T zero;
            T clight;

            T exptol;
            T exptime;
            int expmaxdepth;

            // Current time index
            int t;

            // Derivatives dictionary
            std::map<string, Eigen::VectorXd> derivs;

            // Constructor
            System(vector<Body<T>*> bodies, const double& scale=0, const double& exptime=0, const double& exptol=1e-8, const int& expmaxdepth=4) :
                bodies(bodies),
                clight(CLIGHT / (scale * RSUN)),
                exptol(exptol),
                exptime(exptime * DAY), // Convert to seconds
                expmaxdepth(expmaxdepth) {

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
                    bodies[i]->clight = clight;
                }

                // Set the flag
                computed = false;
                t = 0;

            }

            // Methods
            void compute(const Vector<T>& time);
            inline Vector<T> compute(const T& time, bool store_xyz=true);
            inline void compute_instantaneous(const Vector<T>& time);
            Vector<T> integrate (const Vector<T>& f1, const Vector<T>& f2, const T& t1, const T& t2, int depth);
            inline Vector<T> integrate (const T& time);
            std::string repr();

    };

    // Return a human-readable string
    template <class T>
    std::string System<T>::repr() {
        std::ostringstream os;
        os << "<STARRY " << (bodies.size() - 1) << "-planet system>";
        return std::string(os.str());
    }

    // Recursive exposure time integration function (single iteration)
    template <class T>
    Vector<T> System<T>::integrate (const Vector<T>& f1, const Vector<T>& f2, const T& t1, const T& t2, int depth) {
        T tmid = 0.5 * (t1 + t2);
        // If this is the first time we're recursing,
        // store the xyz position of the bodies
        Vector<T> fmid = compute(tmid, depth == 0),
                  fapprox = 0.5 * (f1 + f2),
                  d = fmid - fapprox;
        if (depth < expmaxdepth) {
            for (int i = 0; i < d.size(); i++) {
                if (abs(d(i)) > exptol) {
                    Vector<T> a = integrate(f1, fmid, t1, tmid, depth + 1),
                              b = integrate(fmid, f2, tmid, t2, depth + 1);
                    return a + b;
                }
            }
        }
        return fapprox * (t2 - t1);
    };

    // Recursive exposure time integration function
    template <class T>
    inline Vector<T> System<T>::integrate (const T& time) {
        if (exptime > 0.0) {
            T dt = 0.5 * exptime,
              t1 = time - dt,
              t2 = time + dt;
            return integrate(compute(t1), compute(t2), t1, t2, 0) / (t2 - t1);
        }
        // No integration
        return compute(time, true);
    };

    // Compute one cadence of the light curve
    template <class T>
    inline Vector<T> System<T>::compute(const T& time, bool store_xyz) {

        int i, j;
        T xo, yo, ro;
        int p, o;
        int NB = bodies.size();
        Vector<T> fluxes = Vector<T>::Zero(NB);

        // Take an orbital step
        for (i = 0; i < NB; i++) {
            bodies[i]->step(time);
            if (store_xyz) {
                bodies[i]->x(t) = bodies[i]->x_;
                bodies[i]->y(t) = bodies[i]->y_;
                bodies[i]->z(t) = bodies[i]->z_;
            }
        }

        // Compute any occultations
        for (i = 0; i < NB; i++) {
            for (j = i + 1; j < NB; j++) {
                // Determine the relative positions of the two bodies
                if (bodies[j]->z_ > bodies[i]->z_) {
                    o = j;
                    p = i;
                } else {
                    o = i;
                    p = j;
                }
                xo = (bodies[o]->x_ - bodies[p]->x_) / bodies[p]->r;
                yo = (bodies[o]->y_ - bodies[p]->y_) / bodies[p]->r;
                ro = (bodies[o]->r / bodies[p]->r);
                // Compute the flux in occultation
                if (sqrt(xo * xo + yo * yo) < 1 + ro) {
                    bodies[p]->getflux(time, xo, yo, ro);
                }
            }
        }

        // Return the flux from each of the bodies
        for (i = 0; i < NB; i++) fluxes(i) = bodies[i]->flux_;
        return fluxes;

    }

    // Compute the light curve
    template <class T>
    inline void System<T>::compute(const Vector<T>& time) {

        // Optimized version of this function with no exposure time integration
        if (exptime == 0.0)
            return compute_instantaneous(time);

        int i;
        T tsec;
        int NT = time.size();
        int NB = bodies.size();
        Vector<T> fluxes = Vector<T>::Zero(NB);
        flux = Vector<T>::Zero(NT);

        // Allocate arrays and check that the planet maps are physical
        // Propagate the speed of light to all the bodies
        // Sync the orbital and sky maps
        for (i = 0; i < NB; i++) {
            bodies[i]->x.resize(NT);
            bodies[i]->y.resize(NT);
            bodies[i]->z.resize(NT);
            bodies[i]->flux.resize(NT);
            bodies[i]->clight = clight;
            bodies[i]->sync_maps();
        }

        // Loop through the timeseries
        for (t = 0; t < NT; t++){

            // Time in seconds
            tsec = time(t) * DAY;

            // Take an orbital step and compute the fluxes
            fluxes = integrate(tsec);

            // Update the body vectors
            for (i = 0; i < NB; i++) {
                bodies[i]->flux(t) = fluxes(i);
                flux(t) += fluxes(i);
            }

        }

        // Set the flag
        computed = true;

    }

    // Compute the light curve. Special case w/ no exposure time integration
    // optimized for speed.
    template <class T>
    inline void System<T>::compute_instantaneous(const Vector<T>& time) {

        int i, j;
        T xo, yo, ro;
        T tsec;
        int p, o;
        int NT = time.size();
        int NB = bodies.size();
        flux = Vector<T>::Zero(NT);

        // Allocate arrays and check that the planet maps are physical
        // Propagate the speed of light to all the bodies
        // Sync the orbital and sky maps
        for (i = 0; i < NB; i++) {
            bodies[i]->x.resize(NT);
            bodies[i]->y.resize(NT);
            bodies[i]->z.resize(NT);
            bodies[i]->flux.resize(NT);
            bodies[i]->clight = clight;
            bodies[i]->sync_maps();
        }

        // Loop through the timeseries
        for (t = 0; t < NT; t++){

            // Time in seconds
            tsec = time(t) * DAY;

            // Take an orbital step
            for (i = 0; i < NB; i++)
                bodies[i]->step(tsec);

            // Compute any occultations
            for (i = 0; i < NB; i++) {
                for (j = i + 1; j < NB; j++) {
                    // Determine the relative positions of the two bodies
                    if (bodies[j]->z_ > bodies[i]->z_) {
                        o = j;
                        p = i;
                    } else {
                        o = i;
                        p = j;
                    }
                    xo = (bodies[o]->x_ - bodies[p]->x_) / bodies[p]->r;
                    yo = (bodies[o]->y_ - bodies[p]->y_) / bodies[p]->r;
                    ro = (bodies[o]->r / bodies[p]->r);
                    // Compute the flux in occultation
                    if (sqrt(xo * xo + yo * yo) < 1 + ro) {
                        bodies[p]->getflux(tsec, xo, yo, ro);
                    }
                }
            }

            // Update the body vectors
            for (i = 0; i < NB; i++) {
                bodies[i]->x(t) = bodies[i]->x_;
                bodies[i]->y(t) = bodies[i]->y_;
                bodies[i]->z(t) = bodies[i]->z_;
                bodies[i]->flux(t) = bodies[i]->flux_;
                flux(t) += bodies[i]->flux_;
            }

        }

        // Set the flag
        computed = true;

    }

    // Grad specialization: compute the light curve and the derivs
    template <>
    void System<Grad>::compute(const Vector<Grad>& time) {

        int i, j, n, k, l, m;
        Grad xo, yo, ro;
        Grad tsec;
        int NT = time.size();
        int NB = bodies.size();
        Vector<Grad> fluxes = Vector<Grad>::Zero(NB);
        vector<Vector<double>> tmpder;
        flux = Vector<Grad>::Zero(NT);

        // List of gradient names
        vector<string> names {"time"};
        for (l = 1; l < bodies[0]->ldmap.lmax + 1; l++) {
            names.push_back(string("star.u_" + to_string(l)));
        }
        for (i = 1; i < NB; i++) {
            names.push_back(string("planet" + to_string(i) + ".r"));
            names.push_back(string("planet" + to_string(i) + ".L"));
            names.push_back(string("planet" + to_string(i) + ".axis_x"));
            names.push_back(string("planet" + to_string(i) + ".axis_y"));
            names.push_back(string("planet" + to_string(i) + ".axis_z"));
            names.push_back(string("planet" + to_string(i) + ".prot"));
            names.push_back(string("planet" + to_string(i) + ".a"));
            names.push_back(string("planet" + to_string(i) + ".porb"));
            names.push_back(string("planet" + to_string(i) + ".inc"));
            names.push_back(string("planet" + to_string(i) + ".ecc"));
            names.push_back(string("planet" + to_string(i) + ".w"));
            names.push_back(string("planet" + to_string(i) + ".Omega"));
            names.push_back(string("planet" + to_string(i) + ".lambda0"));
            names.push_back(string("planet" + to_string(i) + ".tref"));
            for (l = 0; l < bodies[i]->map_sky.lmax + 1; l++) {
                for (m = -l; m < l + 1; m++) {
                    names.push_back(string("planet" + to_string(i) + ".Y_{" + to_string(l) + "," + to_string(m) + "}"));
                }
            }
        }

        // Check that our derivative vectors are large enough
        int ngrad = names.size();
        if (ngrad > STARRY_NGRAD) throw errors::TooManyDerivs(ngrad);

        // Allocate arrays and derivs
        // Propagate the speed of light to all the bodies
        // Sync the orbital and sky maps
        for (i = 0; i < NB; i++) {
            bodies[i]->x.resize(NT);
            bodies[i]->y.resize(NT);
            bodies[i]->z.resize(NT);
            bodies[i]->flux.resize(NT);
            bodies[i]->derivs.clear();
            for (n = 0; n < ngrad; n++) {
                if (i == 0)
                    bodies[i]->derivs[names[n]].resize(NT);
                else
                    bodies[i]->derivs[names[n]].resize(NT);
            }
            bodies[i]->clight = clight;
            bodies[i]->sync_maps();
        }
        for (n = 0; n < ngrad; n++) {
            derivs[names[n]].resize(NT);
        }

        // Loop through the timeseries
        for (t = 0; t < NT; t++){

            // Allocate the derivatives
            n = 0;
            tsec = time(t) * DAY;
            tsec.derivatives() = Vector<double>::Unit(STARRY_NGRAD, n++);

            // Convert the derivative back to days
            tsec.derivatives()(0) *= DAY;

            // Star derivs (map only)
            for (k = 1; k < bodies[0]->ldmap.lmax + 1; k++)
                bodies[0]->ldmap.u(k).derivatives() = Vector<double>::Unit(STARRY_NGRAD, n++);

            // The following ensures the derivatives of `u` are correctly
            // propagated to the `g` vector, which is what we use in the
            // flux calculation for limb-darkened bodies.
            if (t == 0) {
                bodies[0]->ldmap.update();
                for (i = 0; i < bodies[0]->ldmap.g.size(); i += 2)
                    tmpder.push_back(bodies[0]->ldmap.g(i).derivatives());
            } else {
                j = 0;
                for (i = 0; i < bodies[0]->ldmap.g.size(); i += 2)
                    bodies[0]->ldmap.g(i).derivatives() = tmpder[j++];
            }

            // Planet derivs
            for (i = 1; i < NB; i++) {

                // Orbital derivs
                bodies[i]->r.derivatives() = Vector<double>::Unit(STARRY_NGRAD, n++);
                bodies[i]->L.derivatives() = Vector<double>::Unit(STARRY_NGRAD, n++);
                bodies[i]->axis(0).derivatives() = Vector<double>::Unit(STARRY_NGRAD, n++);
                bodies[i]->axis(1).derivatives() = Vector<double>::Unit(STARRY_NGRAD, n++);
                bodies[i]->axis(2).derivatives() = Vector<double>::Unit(STARRY_NGRAD, n++);
                bodies[i]->prot.derivatives() = Vector<double>::Unit(STARRY_NGRAD, n++) * DAY;
                bodies[i]->a.derivatives() = Vector<double>::Unit(STARRY_NGRAD, n++);
                bodies[i]->porb.derivatives() = Vector<double>::Unit(STARRY_NGRAD, n++) * DAY;
                bodies[i]->inc.derivatives() = Vector<double>::Unit(STARRY_NGRAD, n++) * DEGREE;
                bodies[i]->ecc.derivatives() = Vector<double>::Unit(STARRY_NGRAD, n++);
                bodies[i]->w.derivatives() = Vector<double>::Unit(STARRY_NGRAD, n++) * DEGREE;
                bodies[i]->Omega.derivatives() = Vector<double>::Unit(STARRY_NGRAD, n++) * DEGREE;
                bodies[i]->lambda0.derivatives() = Vector<double>::Unit(STARRY_NGRAD, n++) * DEGREE;
                bodies[i]->tref.derivatives() = Vector<double>::Unit(STARRY_NGRAD, n++) * DAY;

                // Propagate derivs to the helper variables
                bodies[i]->reset();

                // Map derivs
                for (k = 0; k < bodies[i]->map_sky.N; k++)
                    bodies[i]->map_sky.y(k).derivatives() = Vector<double>::Unit(STARRY_NGRAD, n++);
            }

            // Take an orbital step and compute the fluxes
            fluxes = integrate(tsec);

            // Get the total system flux and update body vectors
            for (i = 0; i < NB; i++) {
                bodies[i]->flux(t) = fluxes(i);
                flux(t) += fluxes(i);

                // Store the derivs
                for (n = 0; n < ngrad; n++) {
                    (bodies[i]->derivs[names[n]])(t) = bodies[i]->flux(t).derivatives()(n);
                    if (i == 0)
                        (derivs[names[n]])(t) = bodies[i]->flux(t).derivatives()(n);
                    else
                        (derivs[names[n]])(t) += bodies[i]->flux(t).derivatives()(n);
                }
            }

        }

        // Set the flag
        computed = true;

    }

}; // namespace orbital

#endif
