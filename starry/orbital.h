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
using std::string;
using std::to_string;

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
    Eigen::AutoDiffScalar<T> fmod(const Eigen::AutoDiffScalar<T>& numer, double denom) {
        typename T::Scalar numer_value = numer.value(),
                           modulo_value = fmod(numer_value, denom);
        return Eigen::AutoDiffScalar<T>(
          modulo_value,
          numer.derivatives()
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

            // Derivatives dictionary
            std::map<string, Eigen::VectorXd> derivs;

            // Settings
            double eps;
            int maxiter;

            // Orbital position
            Vector<T> x;
            T x_;
            Vector<T> y;
            T y_;
            Vector<T> z;
            T z_;

            // Flux
            Vector<T> flux;
            T flux_;

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
                        norm = 2. / sqrt(M_PI);

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
                angvelorb = (2 * M_PI) / porb;
                angvelrot = (2 * M_PI) / prot;
            };

            // Public methods
            T theta(const T& time);
            void step(const T& time);
            void getflux(const T& time, const T& xo=0, const T& yo=0, const T& ro=0);
            std::string repr();
    };

    // Rotation angle as a function of time
    template <class T>
    inline T Body<T>::theta(const T& time) {
        if ((prot == 0) || (prot == INFINITY))
            return theta0;
        else
            return fmod(theta0 + angvelrot * (time - tref), 2 * M_PI);
    }

    // Compute the flux in occultation
    template <class T>
    inline void Body<T>::getflux(const T& time, const T& xo, const T& yo, const T& ro){
        if (L != 0) {
            if (is_star)
                flux_ += norm * ldmap.flux(xo, yo, ro) - totalflux;
            else
                flux_ += norm * L * map.flux(axis, theta(time), xo, yo, ro) - totalflux;
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
            M = fmod(M0 + angvelorb * (time - tref), 2 * M_PI);

            // Eccentric anomaly
            E = EccentricAnomaly(M, ecc, eps, maxiter);

            // True anomaly
            if (ecc == 0) f = E;
            else f = (2. * atan2(sqrtonepluse * sin(E / 2.),
                                 sqrtoneminuse * cos(E / 2.)));

            // Orbital radius
            if (ecc > 0)
                rorb = a * (1. - ecc2) / (1. + ecc * cos(f));
            else
                rorb = a;

            // Murray and Dermott p. 51
            cwf = cos(w + f);
            swf = sin(w + f);
            x_ = rorb * (cosO * cwf - sinOcosi * swf);
            y_ = rorb * (sinO * cwf + cosOcosi * swf);
            z_ = rorb * swf * sini;

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

    // System class
    template <class T>
    class System {

        public:

            vector<Body<T>*> bodies;
            Vector<T> flux;
            double eps;
            int maxiter;
            bool computed;
            T zero;

            T exptol;
            T exptime;
            int expmaxdepth;

            // Current time index
            int t;

            // Derivatives dictionary
            std::map<string, Eigen::VectorXd> derivs;

            // Constructor
            System(vector<Body<T>*> bodies, const double& eps=1.0e-7, const int& maxiter=100,
                   const double& exptime=0, const double& exptol=1e-8, const int& expmaxdepth=4) :
                bodies(bodies),
                eps(eps),
                maxiter(maxiter),
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
                    bodies[i]->eps = eps;
                    bodies[i]->maxiter = maxiter;
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
        if (exptime == 0.0) return compute_instantaneous(time);

        int i;
        T tsec;
        int NT = time.size();
        int NB = bodies.size();
        Vector<T> fluxes = Vector<T>::Zero(NB);
        flux = Vector<T>::Zero(NT);

        // Allocate arrays and check that the planet maps are physical
        for (i = 0; i < NB; i++) {
            bodies[i]->x.resize(NT);
            bodies[i]->y.resize(NT);
            bodies[i]->z.resize(NT);
            bodies[i]->flux.resize(NT);
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
        for (i = 0; i < NB; i++) {
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
            names.push_back(string("planet" + to_string(i) + ".theta0"));
            names.push_back(string("planet" + to_string(i) + ".a"));
            names.push_back(string("planet" + to_string(i) + ".porb"));
            names.push_back(string("planet" + to_string(i) + ".inc"));
            names.push_back(string("planet" + to_string(i) + ".ecc"));
            names.push_back(string("planet" + to_string(i) + ".w"));
            names.push_back(string("planet" + to_string(i) + ".Omega"));
            names.push_back(string("planet" + to_string(i) + ".lambda0"));
            names.push_back(string("planet" + to_string(i) + ".tref"));
            for (l = 0; l < bodies[i]->map.lmax + 1; l++) {
                for (m = -l; m < l + 1; m++) {
                    names.push_back(string("planet" + to_string(i) + ".Y_{" + to_string(l) + "," + to_string(m) + "}"));
                }
            }
        }

        // Check that our derivative vectors are large enough
        int ngrad = names.size();
        if (ngrad > STARRY_NGRAD) throw errors::TooManyDerivs(ngrad);

        // Allocate arrays and derivs
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

            // Convert the deriv back to days!
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
                bodies[i]->prot.derivatives() = Vector<double>::Unit(STARRY_NGRAD, n++);
                bodies[i]->theta0.derivatives() = Vector<double>::Unit(STARRY_NGRAD, n++);
                bodies[i]->a.derivatives() = Vector<double>::Unit(STARRY_NGRAD, n++);
                bodies[i]->porb.derivatives() = Vector<double>::Unit(STARRY_NGRAD, n++);
                bodies[i]->inc.derivatives() = Vector<double>::Unit(STARRY_NGRAD, n++);
                bodies[i]->ecc.derivatives() = Vector<double>::Unit(STARRY_NGRAD, n++);
                bodies[i]->w.derivatives() = Vector<double>::Unit(STARRY_NGRAD, n++);
                bodies[i]->Omega.derivatives() = Vector<double>::Unit(STARRY_NGRAD, n++);
                bodies[i]->lambda0.derivatives() = Vector<double>::Unit(STARRY_NGRAD, n++);
                bodies[i]->tref.derivatives() = Vector<double>::Unit(STARRY_NGRAD, n++);

                // Propagate derivs to the helper variables
                bodies[i]->reset();

                // Map derivs
                for (k = 0; k < bodies[i]->map.N; k++)
                    bodies[i]->map.y(k).derivatives() = Vector<double>::Unit(STARRY_NGRAD, n++);
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
