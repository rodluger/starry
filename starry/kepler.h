/**
Keplerian star/planet/moon system class.

TODO: Many of the gradient methods here can still be optimized for speed.
      Common variables can be pre-computed; use references to make the
      application of the chain rule more legible.

TODO: The biggest speedup may come from only computing the total flux
      when there are no occultations or when there are more than one.
      For exactly one occultation, the occultation flux
      is all we need!

*/

#ifndef _STARRY_ORBITAL_H_
#define _STARRY_ORBITAL_H_

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>
#include <string>
#include <vector>
#include "errors.h"
#include "maps.h"
#include "utils.h"
#include "rotation.h"


namespace units {

    const double DayToSeconds = 86400.0;                                        /**< One day in seconds */
    const double SpeedOfLight = 299792458.;                                     /**< Speed of light in m / s */
    const double SolarRadius = 6.95700e8;                                       /**< Radius of the sun in m */

} // namespace units

namespace kepler {

    using namespace utils;
    using maps::Map;
    using rotation::Wigner;
    using std::abs;
    using utils::isInfinite;

    // Forward declare our classes
    template <class T> class Body;
    template <class T> class Primary;
    template <class T> class Secondary;
    template <class T> class System;

    // Gradient labels
    static const std::vector<std::string> PRIMARY_GRAD_NAMES({"prot", "tref"});
    static const std::vector<std::string> SECONDARY_GRAD_NAMES({"r", "L",
        "prot", "a", "porb", "inc", "ecc", "w", "Omega", "lambda0", "tref"});


    /* ---------------- */
    /*     FUNCTIONS    */
    /* ---------------- */

    /**
    Compute the eccentric anomaly. Adapted from
    https://github.com/lkreidberg/batman/blob/master/c_src/_rsky.c

    */
    template <typename T>
    T EccentricAnomaly(const T& M, const T& ecc) {
        // Initial condition
        T E = M;
        T tol = 10 * mach_eps<T>();
        T delta;
        if (ecc > 0) {
            // Iterate
            for (int iter = 0; iter <= STARRY_KEPLER_MAX_ITER; iter++) {
                delta = ecc * sin(E) + M;
                E = E - (E - delta) / (1. - ecc * cos(E));
                if (abs(E - delta) <= tol) return E;
            }
            // Didn't converge!
            throw errors::ConvergenceError("The Kepler solver "
                                           "did not converge.");
        }
        return E;
    }

    //! Manual derivative of the eccentric anomaly
    template <typename T>
    Eigen::AutoDiffScalar<T> EccentricAnomaly(const Eigen::AutoDiffScalar<T>& M,
            const Eigen::AutoDiffScalar<T>& ecc) {
        typename T::Scalar M_value = M.value(),
                           ecc_value = ecc.value(),
                           E_value = EccentricAnomaly(M_value, ecc_value),
                           cosE_value = cos(E_value),
                           sinE_value = sin(E_value),
                           norm1 = 1./ (1. - ecc_value * cosE_value),
                           norm2 = sinE_value * norm1;
        if (M.derivatives().size() && ecc.derivatives().size())
            return Eigen::AutoDiffScalar<T>(E_value, M.derivatives() * norm1 +
                                            ecc.derivatives() * norm2);
        else if (M.derivatives().size())
            return Eigen::AutoDiffScalar<T>(E_value, M.derivatives() * norm1);
        else if (ecc.derivatives().size())
            return Eigen::AutoDiffScalar<T>(E_value, ecc.derivatives() * norm2);
        else
            return Eigen::AutoDiffScalar<T>(E_value, M.derivatives());
    }

    /**
    Compute the light travel time delay and apply
    it to the current (x, y, z) position of the body.
    Templated so we can autodiff it!

    */
    template <class U>
    void applyLightDelay(const U& time,
        const U& a, const U& ecc, const U& ecc2,
        const U& sqrtonepluse, const U& sqrtoneminuse,
        const U& w, const U& angvelorb, const U& tref,
        const U& M0, const U& cosO, const U& sinO, const U& sini,
        const U& cosOcosi, const U& sinOcosi,
        const U& vamp, const U& ecw, const U& z0, const U& c,
        U& cwf, U& rorb, U& x, U& y, U& z, U& delay) {

        // Component of the velocity out of the sky
        // Obtained by differentiating the Kepler solution
        U vz = vamp * sini * (ecw + cwf);

        // Component of the acceleration out of the sky
        U az = -angvelorb * angvelorb * a * a * a /
               (rorb * rorb * rorb) * z;

        // Compute the time delay at the **retarded** position, accounting
        // for the instantaneous velocity and acceleration of the body.
        // This is slightly better than doing
        //
        //          dt = (z0 - z) / c
        //
        // which is actually the time delay at the **current** position.
        // But the photons left the planet from the **retarded** position,
        // so if the planet has motion in the `z` direction the two values
        // will be slightly different. In practice it doesn't really matter
        // that much, though. See the derivation at
        // https://github.com/rodluger/starry/issues/66
        if (abs(az) < 1e-10)
            delay = (z0 - z) / (c + vz);
        else
            delay = (c / az) * ((1 + vz / c)
                    - sqrt((1 + vz / c) * (1 + vz / c)
                    - 2 * az * (z0 - z) / (c * c)));

        // Re-compute Kepler's equation, this time
        // solving for the **retarded** position
        U M = mod2pi(M0 + angvelorb * (time - delay - tref));
        U f;
        if (ecc > 0) {
            U E = EccentricAnomaly(M, ecc);
            f = (2. * atan2(sqrtonepluse * sin(E / 2.),
                            sqrtoneminuse * cos(E / 2.)));
            rorb = a * (1. - ecc2) / (1. + ecc * cos(f));
        } else {
            f = M;
            rorb = a;
        }
        cwf = cos(w + f);
        U swf = sin(w + f);
        x = -rorb * (cosO * cwf - sinOcosi * swf);
        y = -rorb * (sinO * cwf + cosOcosi * swf);
        z = rorb * swf * sini;

    }

    /**
    Compute the instantaneous x, y, and z positions of the
    body with a simple Keplerian solver. Templated for
    AD capability.

    */
    template <class U>
    void keplerStep(const U& time,
        const U& a, const U& ecc, const U& ecc2,
        const U& sqrtonepluse, const U& sqrtoneminuse,
        const U& w, const U& angvelorb, const U& tref,
        const U& M0, const U& cosO, const U& sinO, const U& sini,
        const U& cosOcosi, const U& sinOcosi,
        const U& vamp, const U& ecw, const U& z0, const U& c,
        U& x, U& y, U& z, U& delay) {

        // Mean anomaly
        U M = mod2pi(M0 + angvelorb * (time - tref));

        // True anomaly and orbital radius
        U f, rorb;
        if (ecc == 0) {
            f = M;
            rorb = a;
        } else {
            U E = EccentricAnomaly(M, ecc);
            f = (2. * atan2(sqrtonepluse * sin(E / 2.),
                            sqrtoneminuse * cos(E / 2.)));
            rorb = a * (1. - ecc2) / (1. + ecc * cos(f));
        }

        // See Murray and Dermott p. 51, except x and y
        // have the opposite sign here.
        // This ensures the orbits are prograde!
        U cwf = cos(w + f);
        U swf = sin(w + f);
        x = -rorb * (cosO * cwf - sinOcosi * swf);
        y = -rorb * (sinO * cwf + cosOcosi * swf);
        z = rorb * swf * sini;

        // Compute the light travel time delay
        if (!isInfinite(c))
            applyLightDelay(time, a, ecc, ecc2, sqrtonepluse, sqrtoneminuse,
                            w, angvelorb, tref, M0, cosO, sinO, sini,
                            cosOcosi, sinOcosi, vamp, ecw, z0, c, cwf, rorb,
                            x, y, z, delay);

    }

    /**
    Flux container for exposure time integration.

    */
    template <class T>
    class Exposure {

        public:

            std::vector<Row<T>> flux;
            std::vector<T> gradient;
            size_t nsec;
            bool grad;

            explicit Exposure(size_t nsec, bool grad) :
                    nsec(nsec), grad(grad) {
                flux.resize(nsec + 1);
                if (grad) gradient.resize(nsec + 1);
            }

            inline Exposure<T> operator+(const Exposure<T>& exposure) const {
                Exposure<T> result(*this);
                for (size_t n = 0; n < nsec + 1; ++n) {
                    result.flux[n] += exposure.flux[n];
                    if (grad)
                        result.gradient[n] += exposure.gradient[n];
                }
                return result;
            }

            inline Exposure<T> operator-(const Exposure<T>& exposure) const {
                Exposure<T> result(*this);
                for (size_t n = 0; n < nsec + 1; ++n) {
                    result.flux[n] -= exposure.flux[n];
                    if (grad)
                        result.gradient[n] -= exposure.gradient[n];
                }
                return result;
            }

            inline Exposure<T> operator*(const Scalar<T>& mult) const {
                Exposure<T> result(*this);
                for (size_t n = 0; n < nsec + 1; ++n) {
                    result.flux[n] *= mult;
                    if (grad)
                        result.gradient[n] *= mult;
                }
                return result;
            }

    };


    /* ---------------- */
    /*        AD        */
    /* ---------------- */

    /**
    A container for AutoDiffScalar variables used in the
    Kepler solver.

    */
    template <class T, int N=10>
    class AutoDiffKepler {

        public:

            using S = Scalar<T>;
            using A = ADScalar<S, N>;
            using V = Vector<S>;

            // Input variables
            A time;
            A a;
            A ecc;
            A M0;
            A tref;
            A porb;
            A w;
            A Omega;
            A inc;

            // Output variables
            A x;
            A y;
            A z;
            A delay;

            // Computed variables / temporaries
            A ecc2;
            A sqrtonepluse;
            A sqrtoneminuse;
            A angvelorb;
            A cosO;
            A sinO;
            A cosi;
            A sini;
            A cosOcosi;
            A sinOcosi;
            A vamp;
            A ecw;
            A z0;
            A c;

            /**
            Seed the input variable derivatives.

            */
            void reset(const S& time_=0, const S& a_=0, const S& ecc_=0,
                       const S& M0_=0, const S& tref_=0, const S& porb_=0,
                       const S& w_=0, const S& Omega_=0, const S& inc_=0) {
                time = A(time_, V::Unit(N, 0));
                a = A(a_, V::Unit(N, 1));
                ecc = A(ecc_, V::Unit(N, 2));
                M0 = A(M0_, V::Unit(N, 3));
                tref = A(tref_, V::Unit(N, 4));
                porb = A(porb_, V::Unit(N, 5));
                w = A(w_, V::Unit(N, 6));
                Omega = A(Omega_, V::Unit(N, 7));
                inc = A(inc_, V::Unit(N, 8));
            }

            /**
            Default constructor

            */
            explicit AutoDiffKepler() {
                reset();
                x = 0;
                y = 0;
                z = 0;
                delay = 0;
                ecc2 = 0;
                sqrtonepluse = 0;
                sqrtoneminuse = 0;
                angvelorb = 0;
                cosO = 0;
                sinO = 0;
                sini = 0;
                cosOcosi = 0;
                sinOcosi = 0;
                vamp = 0;
                ecw = 0;
                z0 = 0;
                c = 0;
            }

    };


    /* ---------------- */
    /*       BODY       */
    /* ---------------- */

    /**
    Generic body class, a subclass of Map with added orbital features.
    This class cannot be instantiated; instead, users should use the
    Primary and Secondary subclasses.

    */
    template <class T>
    class Body : public Map<T> {

        friend class System<T>;

        protected:

            using S = Scalar<T>;                                                /**< Shorthand for the scalar type (double, Multi, ...) */
            using Map<T>::dF;
            using Map<T>::dF_names;

            S r;                                                                /**< Body radius in units of primary radius */
            Row<T> L;                                                           /**< Body luminosity in units of primary luminosity */
            S prot;                                                             /**< Body rotation period in seconds */
            S tref;                                                             /**< Reference time in seconds */
            S theta0_deg;                                                       /**< Body initial rotation angle in degrees */
            S angvelrot_deg;                                                    /**< Body rotational angular velocity in degrees / second */
            S z0;                                                               /**< Reference point for retarded time calculation (the primary, assuming massless secondaries) */
            S delay;                                                            /**< The light travel time delay in seconds */
            Row<T> flux_cur;                                                    /**< Current flux visible from the body */
            Row<T> flux_tot;                                                    /**< Total flux from the body */
            T dflux_tot;                                                        /**< Gradient of the total flux from the body */
            T dflux_cur;                                                        /**< Gradient of the current flux from the body */
            size_t ngrad;                                                       /** Number of derivatives to compute */
            bool computed;                                                      /**< Did the user call `compute()`? */

            Matrix<Scalar<T>> lightcurve;                                       /**< The body's full light curve */
            Vector<T> dL;                                                       /**< The gradient of the body's light curve */
            std::vector<std::string> dL_names;                                  /**< Names of each of the params in the light curve gradient */

            // Private methods
            inline S theta_deg(const S& time);
            void computeTotal(const S& time, bool gradient, bool numerical);
            void occult(const S& time, const S& xo, const S& yo, const S& ro,
                        bool gradient, bool numerical);

            //! Wrapper to get the flux from the map (overriden in Secondary)
            virtual inline Row<T> getFlux(const S& theta_deg, const S& xo,
                const S& yo, const S& ro, bool gradient, bool numerical) {
                return this->flux(theta_deg, xo, yo, ro, gradient, numerical);
            }

            //! Compute the initial rotation angle (overriden in Secondary)
            virtual void computeTheta0() {
                theta0_deg = 0;
            }

            //! Constructor
            explicit Body(int lmax=2, int nwav=1) :
                Map<T>(lmax, nwav),
                flux_cur(nwav),
                flux_tot(nwav)  {

                // Initialize some stuff
                setZero(flux_cur);
                setZero(flux_tot);
                z0 = 0;
                delay = 0;
                computeTheta0();

                // Set the orbital variables to default values
                setRadius(1.0);
                Row<T> defaultL;
                resize(defaultL, 1, nwav);
                setOnes(defaultL);
                setLuminosity(defaultL);
                setRotPer(0.0);
                setRefTime(0.0);

                computed = false;
            }

        public:

            // I/O
            using Map<T>::nwav;
            void setRadius(const S& r_);
            virtual S getRadius() const;
            void setLuminosity(const Row<T>& L_);
            virtual Row<T> getLuminosity() const;
            void setRotPer(const S& prot_);
            S getRotPer() const;
            void setRefTime(const S& tref_);
            S getRefTime() const;
            const Matrix<Scalar<T>>& getLightcurve() const;
            const Vector<T>& getLightcurveGradient() const;
            const std::vector<std::string>& getLightcurveGradientNames() const;

    };


    /* ---------------------- */
    /*    BODY: OPERATIONS    */
    /* ---------------------- */

    //! Rotation angle in degrees as a function of (retarded) time
    template <class T>
    inline Scalar<T> Body<T>::theta_deg(const Scalar<T>& time) {
        if (prot == INFINITY)
            return theta0_deg;
        else
            return mod360(theta0_deg + angvelrot_deg * (time - tref - delay));
    }

    //! Compute the total flux from the body
    template <class T>
    inline void Body<T>::computeTotal(const Scalar<T>& time, bool gradient, bool numerical) {
        if (!allZero(L)) {
            flux_tot = cwiseProduct(L, getFlux(theta_deg(time), 0, 0, 0, gradient, numerical));
        } else {
            setZero(flux_tot);
        }
        flux_cur = flux_tot;
    }

    //! Occult the body and update the current flux
    template <class T>
    inline void Body<T>::occult(const Scalar<T>& time, const Scalar<T>& xo,
                                const Scalar<T>& yo, const Scalar<T>& ro,
                                bool gradient, bool numerical) {
        flux_cur += cwiseProduct(L, getFlux(theta_deg(time), xo, yo, ro, gradient, numerical))
                    - flux_tot;
    }


    /* ------------------ */
    /*     BODY: I/O      */
    /* ------------------ */

    //! Set the body's radius
    template <class T>
    void Body<T>::setRadius(const Scalar<T>& r_) {
        if (r_ > 0) r = r_;
        else throw errors::ValueError("Body's radius must be positive.");
    }

    //! Get the body's radius
    template <class T>
    Scalar<T> Body<T>::getRadius() const {
        return r;
    }

    //! Set the body's luminosity
    template <class T>
    void Body<T>::setLuminosity(const Row<T>& L_) {
        L = L_;
        for (int n = 0; n < nwav; ++n) {
            if (getIndex(L_, n) < 0)
                throw errors::ValueError("Body's luminosity cannot be negative.");
        }
    }

    //! Get the body's luminosity
    template <class T>
    Row<T> Body<T>::getLuminosity() const {
        return L;
    }

    //! Set the body's rotation period
    template <class T>
    void Body<T>::setRotPer(const Scalar<T>& prot_) {
        if (prot_ > 0) prot = prot_ * units::DayToSeconds;
        else if (prot_ == 0) prot = INFINITY;
        else throw errors::ValueError("Body's rotation period "
                                      "must be positive.");
        angvelrot_deg = 360.0 / prot;
        computeTheta0();
    }

    //! Get the body's rotation period
    template <class T>
    Scalar<T> Body<T>::getRotPer() const {
        return prot / units::DayToSeconds;
    }

    //! Set the reference time
    template <class T>
    void Body<T>::setRefTime(const Scalar<T>& tref_) {
        tref = tref_ * units::DayToSeconds;
    }

    //! Get the reference time
    template <class T>
    Scalar<T> Body<T>::getRefTime() const {
        return tref / units::DayToSeconds;
    }

    //! Get the body's full light curve
    template <class T>
    const Matrix<Scalar<T>>& Body<T>::getLightcurve() const {
        if (!computed)
            throw errors::ValueError("Please call the `compute` method first.");
        return lightcurve;
    }

    //! Get the gradient of the body's light curve
    template <class T>
    const Vector<T>& Body<T>::getLightcurveGradient() const {
        if (!computed)
            throw errors::ValueError("Please call the `compute` method first.");
        return dL;
    }

    //! Get the names of the params in the gradient
    template <class T>
    const std::vector<std::string>& Body<T>::getLightcurveGradientNames() const {
        return dL_names;
    }


    /* ---------------- */
    /*     PRIMARY      */
    /* ---------------- */

    /**
    Primary class, a subclass of Body that simply sits
    quietly at the origin. Its radius and luminosity are
    both fixed at unity.

    */
    template <class T>
    class Primary : public Body<T> {

        friend class System<T>;

        protected:

            using S = Scalar<T>;                                                /**< Shorthand for the scalar type (double, Multi, ...) */
            using Body<T>::lightcurve;
            using Body<T>::setRotPer;
            using Body<T>::setRefTime;
            using Body<T>::L;
            using Body<T>::dflux_tot;
            using Body<T>::dflux_cur;
            using Body<T>::flux_tot;
            using Body<T>::dF;
            using Body<T>::angvelrot_deg;
            using Body<T>::prot;
            using Body<T>::tref;
            using Body<T>::theta0_deg;
            using Body<T>::delay;
            using Body<T>::y_deg;
            using Body<T>::u_deg;
            S r_meters;                                                         /**< Radius of the body in meters */
            S c_light;                                                          /**< Speed of light in units of primary radius / s */

        public:

            //! Constructor
            explicit Primary(int lmax=2, int nwav=1) :

                // Call the `Body` constructor
                Body<T>(lmax, nwav)
            {

                setRadiusInMeters(0.0);
                setRadius(1.0);
                Row<T> defaultL;
                resize(defaultL, 1, nwav);
                setOnes(defaultL);
                setLuminosity(defaultL);
                setRotPer(0.0);
                setRefTime(0.0);

            }

            // I/O
            void setRadius(const S& r_);
            void setLuminosity(const Row<T>& L_);
            void setRadiusInMeters(const S& r_);
            S getRadiusInMeters() const;
            std::string info();

    };


    /* --------------------- */
    /*      PRIMARY: I/O     */
    /* --------------------- */

    //! Set the physical radius
    template <class T>
    void Primary<T>::setRadiusInMeters(const Scalar<T>& r_) {
        if (r_ > 0) {
            r_meters = r_;
            c_light = units::SpeedOfLight / r_meters;
        } else if (r_ == 0) {
            r_meters = 0;
            c_light = INFINITY;
        } else {
            throw errors::ValueError("The radius cannot be negative.");
        }
    }

    //! Get the physical radius
    template <class T>
    Scalar<T> Primary<T>::getRadiusInMeters() const {
        return r_meters;
    }

    //! Set the body's radius
    template <class T>
    void Primary<T>::setRadius(const Scalar<T>& r_) {
        if (r_ != 1.0)
            throw errors::NotImplementedError("The radius of the primary body "
                                              "is fixed at unity.");
    }

    //! Set the body's luminosity
    template <class T>
    void Primary<T>::setLuminosity(const Row<T>& L_) {
        for (int n = 0; n < this->nwav; ++n) {
            if (getIndex(L_, n) != 1)
                throw errors::NotImplementedError("The luminosity of the primary "
                                                  "body is fixed at unity.");
        }
    }

    //! Human-readable string
    template <class T>
    std::string Primary<T>::info() {
        std::ostringstream os;
        std::string multi;
        if (isMulti(Scalar<T>(0.)))
            multi = "True";
        else
            multi = "False";
        os << "<starry.kepler.Primary("
           << "lmax=" << this->lmax << ", "
           << "nwav=" << this->nwav << ", "
           << "multi=" << multi
           << ")>";
        return std::string(os.str());
    }

    /* ----------------- */
    /*      SECONDARY    */
    /* ----------------- */

    /**
    Secondary class, a subclass of Body that
    moves around the Primary in a Keplerian orbit.

    */
    template <class T>
    class Secondary : public Body<T> {

        friend class System<T>;

        protected:

            using S = Scalar<T>;                                                /**< Shorthand for the scalar type (double, Multi, ...) */
            using Body<T>::theta0_deg;
            using Body<T>::prot;
            using Body<T>::y;
            using Body<T>::u;
            using Body<T>::G;
            using Body<T>::B;
            using Body<T>::lmax;
            using Body<T>::N;
            using Body<T>::nwav;
            using Body<T>::y_deg;
            using Body<T>::u_deg;
            using Body<T>::tref;
            using Body<T>::z0;
            using Body<T>::delay;
            using Body<T>::lightcurve;
            using Body<T>::dF;
            using Body<T>::axis;
            using Body<T>::L;
            using Body<T>::dflux_tot;
            using Body<T>::dflux_cur;
            using Body<T>::flux_tot;
            using Body<T>::angvelrot_deg;

            // Computed values
            Vector<Scalar<T>> xvec;                                             /**< The body's Cartesian x position vector */
            Vector<Scalar<T>> yvec;                                             /**< The body's Cartesian y position vector */
            Vector<Scalar<T>> zvec;                                             /**< The body's Cartesian z position vector */

            // Sky projection stuff
            Map<T> skyMap;                                                      /**< An internal copy of the map, rotated into the sky plane */
            T skyY;                                                             /**< The skyMap spherical harmonic vector of coefficients */
            UnitVector<S> axis1;                                                /**< Instance of the xhat unit vector */
            UnitVector<S> axis2;                                                /**< Instance of the zhat unit vector */
            Wigner<T> W1;                                                       /**< First sky transform (xhat) */
            Wigner<T> W2;                                                       /**< Second sky transform (zhat) */
            Matrix<S>* RSky;                                                    /**< The rotation matrix into the sky plane */

            // The orbital elements
            S a;                                                                /**< The semi-major axis in units of the primary radius */
            S porb;                                                             /**< The orbital period in seconds */
            S inc;                                                              /**< The inclination in radians */
            S ecc;                                                              /**< The orbital eccentricity */
            S w;                                                                /**< The longitude of pericenter (varpi) in radians */
            S Omega;                                                            /**< The longitude of ascending node in radians */
            S lambda0;                                                          /**< The mean longitude at the reference time in radians */

            // Keplerian solution variables
            S* c_light;                                                         /**< Pointer to the speed of light in units of primary radius / s */
            S x_cur;                                                            /**< Current Cartesian x position */
            S y_cur;                                                            /**< Current Cartesian y position */
            S z_cur;                                                            /**< Current Cartesian z position */

            // Auxiliary orbital vars
            S M0;                                                               /**< Value of the mean anomaly at the reference time */
            S cosi;                                                             /**< cos(inc) */
            S sini;                                                             /**< sin(inc) */
            S cosO;                                                             /**< cos(Omega) */
            S sinO;                                                             /**< sin(Omega) */
            S sqrtonepluse;                                                     /**< sqrt(1 + ecc) */
            S sqrtoneminuse;                                                    /**< sqrt(1 - ecc) */
            S ecc2;                                                             /**< ecc * ecc */
            S cosOcosi;                                                         /**< cos(Omega) * cos(inc) */
            S sinOcosi;                                                         /**< sin(Omega) * cos(inc) */
            S ecw;                                                              /**< ecc * cos(w) */
            S esw;                                                              /**< ecc * sin(w) */
            S angvelorb;                                                        /**< Orbital angular velocity in radians / second */
            S vamp;                                                             /**< Orbital velocity amplitude for time delay expansion */
            S aamp;                                                             /**< Orbital acceleration amplitude for time delay expansion */
            S dtheta0_degdw;                                                    /**< Derivative of theta0 with respect to varpi in degrees/radian */
            S dtheta0_degde;                                                    /**< Derivative of theta0 with respect to the eccentricity in degrees */

            // AutoDiff
            AutoDiffKepler<T> AD;                                               /**< AutoDiff container */
            size_t g0;                                                          /**< Starting index of this body's derivs */

            // Private methods
            inline Row<T> getFlux(const S& theta_deg, const S& xo,
                const S& yo, const S& ro, bool gradient, bool numerical);
            void computeTheta0();
            inline void syncSkyMap();
            inline void computeXYZ(const S& time, bool gradient);
            inline void applyLightDelay(const S& time);

        public:

            // I/O
            VectorT<Scalar<T>> getR() const;
            VectorT<Scalar<T>> getS() const;
            void setSemi(const S& a_);
            S getSemi() const;
            void setOrbPer(const S& porb_);
            S getOrbPer() const;
            void setInc(const S& inc_);
            S getInc() const;
            void setEcc(const S& ecc_);
            S getEcc() const;
            void setVarPi(const S& w_);
            S getVarPi() const;
            void setOmega(const S& Om_);
            S getOmega() const;
            void setLambda0(const S& lambda0_);
            S getLambda0() const;
            const Vector<Scalar<T>>& getXVector() const;
            const Vector<Scalar<T>>& getYVector() const;
            const Vector<Scalar<T>>& getZVector() const;
            std::string info();

            //! Constructor
            explicit Secondary(int lmax=2, int nwav=1) :

                // Call the `Body` constructor
                Body<T>(lmax, nwav),

                // Initialize our sky map
                skyMap(lmax, nwav),
                skyY(N, nwav),
                axis1(xhat<S>()),
                axis2(zhat<S>()),
                W1(lmax, nwav, y, axis1),
                W2(lmax, nwav, y, axis2),
                AD()

            {

                // Set the orbital variables to default values
                this->setRadius(0.1);
                Row<T> defaultL;
                resize(defaultL, 1, nwav);
                setZero(defaultL);
                this->setLuminosity(defaultL);
                this->setRotPer(0.0);
                this->setRefTime(0.0);
                setSemi(50.0);
                setOrbPer(1.0);
                setInc(90.0);
                setEcc(0.0);
                setVarPi(90.0);
                setOmega(0.0);
                setLambda0(90.0);

                // Initialize the sky rotation matrix
                RSky = new Matrix<S>[lmax + 1];
                for (int l = 0; l < lmax + 1; ++l)
                    RSky[l].resize(2 * l + 1, 2 * l + 1);

                // Sync the maps
                syncSkyMap();

            }

            //! Destructor
            ~Secondary() {
                delete [] RSky;
            }

    };


    /* -------------------------- */
    /*    SECONDARY: OPERATIONS   */
    /* -------------------------- */

    /**
    Sync the map in the orbital plane (the user-facing one)
    and the map in the sky plane (the one used internally to compute the flux)

    */
    template <class T>
    inline void Secondary<T>::syncSkyMap() {

        // Sync the axis of rotation
        skyMap.setAxis(axis);

        // Sync the limb darkening
        skyMap.setU(u.block(1, 0, u.rows() - 1, nwav));

        // If there's any inclination or rotation of the orbital plane,
        // we need to rotate the sky map as well as the rotation axis
        if ((Omega != 0) || (sini < 1. - 2 * mach_eps<Scalar<T>>())) {

            // Let's store the rotation matrices: we'll need them to correctly
            // transform the derivatives of the map back to the user coordinates
            W1.update();
            W1.compute(sini, cosi);
            W2.update();
            W2.compute(cosO, sinO);
            for (int l = 0; l < lmax + 1; ++l) {
                RSky[l] = W1.R[l] * W2.R[l];
                skyY.block(l * l, 0, 2 * l + 1, nwav) =
                    RSky[l] * y.block(l * l, 0, 2 * l + 1, nwav);
            }

            // Update the sky map
            skyMap.setY(skyY);

        } else {

            // The transformation is the identity matrix
            for (int l = 0; l < lmax + 1; ++l)
                RSky[l] = Matrix<Scalar<T>>::Identity(2 * l + 1, 2 * l + 1);

            // Update the sky map
            skyMap.setY(y);

        }

    }

    /**
    Return the flux from the sky-projected map. This
    overrides `getFlux` in the Body class.

    */
    template <class T>
    inline Row<T> Secondary<T>::getFlux(const Scalar<T>& theta_deg,
                                        const Scalar<T>& xo,
                                        const Scalar<T>& yo,
                                        const Scalar<T>& ro,
                                        bool gradient, bool numerical) {
        // Compute the flux
        Row<T> F = skyMap.flux(theta_deg, xo, yo, ro, gradient, numerical);

        // Carry over the derivatives from the sky map
        auto sky_dF = skyMap.getGradient();
        dF.block(0, 0, 4, nwav) = sky_dF.block(0, 0, 4, nwav);
        if (u_deg > 0) {
            if (y_deg == 0)
                dF.block(5, 0, lmax, nwav) = sky_dF.block(5, 0, lmax, nwav);
            else
                dF.block(4 + N, 0, lmax, nwav) =
                    sky_dF.block(4 + N, 0, lmax, nwav);
        }

        // We need to transform from the derivatives of `skyMap`
        // to the derivatives of the user-facing map via a rotation
        if ((gradient) && (y_deg > 0)) {
            // dF / d{y} = dF / d{ysky} * d{ysky} / d{y}
            // And since ysky = R y, we have d{ysky} / d{y} = R
            for (int l = 0; l < lmax + 1; ++l) {
                dF.block(4 + l * l, 0, 2 * l + 1, nwav) =
                    RSky[l].transpose() *
                    sky_dF.block(4 + l * l, 0, 2 * l + 1, nwav);
            }
        }

        return F;
    }

    /**
    Map rotation angle in degrees at the reference time.
    The map is defined at the
    eclipsing configuration (full dayside as seen by an
    observer viewing the system edge-on), so let's find the
    angle by which we need to rotate the map initially to
    make this happen. This overrides `computeTheta0` in
    the Body class.

    */
    template <class T>
    void Secondary<T>::computeTheta0() {
        if (prot == INFINITY) {
            theta0_deg = 0.0;
            dtheta0_degdw = 0.0;
            dtheta0_degde = 0.0;
        } else {
            using S = Scalar<T>;
            using A = ADScalar<S, 2>;
            using V = Vector<S>;
            A w_ = A(w, V::Unit(2, 0));
            A ecc_ = A(ecc, V::Unit(2, 1));
            A M0_ = lambda0 - w_;
            // Values @ eclipsing configuration
            A f_ = 1.5 * pi<Scalar<T>>() - w_;
            A E_ = atan2(sqrt(1 - ecc_ * ecc_) * sin(f_), ecc_ + cos(f_));
            A M_ = E_ - ecc_ * sin(E_);
            A theta0_deg_ = -(porb / prot) * (M_ - M0_) *
                            180.0 / pi<Scalar<T>>();
            // Store the value & the derivs for later
            theta0_deg = theta0_deg_.value();
            dtheta0_degdw = theta0_deg_.derivatives()(0);
            dtheta0_degde = theta0_deg_.derivatives()(1);
        }
    }

    template <class T>
    void Secondary<T>::computeXYZ(const Scalar<T>& time, bool gradient) {
        if (!gradient) {
            keplerStep(time, a, ecc, ecc2, sqrtonepluse, sqrtoneminuse,
                       w, angvelorb, tref, M0, cosO, sinO, sini, cosOcosi,
                       sinOcosi, vamp, ecw, z0, *c_light,
                       x_cur, y_cur, z_cur, delay);
        } else {

            AD.reset(time, a, ecc, M0, tref, porb, w, Omega, inc);

            // TODO: These need to be pre-computed @ input time
            AD.ecc2 = AD.ecc * AD.ecc;
            AD.sqrtonepluse = sqrt(1 + AD.ecc);
            AD.sqrtoneminuse = sqrt(1 - AD.ecc);
            AD.angvelorb = 2 * pi<S>() / AD.porb;
            AD.cosO = cos(AD.Omega);
            AD.sinO = sin(AD.Omega);
            AD.cosi = cos(AD.inc);
            AD.sini = sin(AD.inc);
            AD.cosOcosi = AD.cosO * AD.cosi;
            AD.sinOcosi = AD.sinO * AD.cosi;
            AD.vamp = AD.angvelorb * AD.a / sqrt(1 - AD.ecc2);
            AD.ecw = AD.ecc * cos(AD.w);
            AD.z0 = z0;
            AD.c = *c_light;

            // Take the step
            keplerStep(AD.time, AD.a, AD.ecc, AD.ecc2, AD.sqrtonepluse,
                       AD.sqrtoneminuse, AD.w, AD.angvelorb, AD.tref,
                       AD.M0, AD.cosO, AD.sinO, AD.sini, AD.cosOcosi,
                       AD.sinOcosi, AD.vamp, AD.ecw, AD.z0, AD.c,
                       AD.x, AD.y, AD.z, AD.delay);

            // Store the values
            x_cur = AD.x.value();
            y_cur = AD.y.value();
            z_cur = AD.z.value();
            delay = AD.delay.value();

        }
    }


    /* --------------------- */
    /*     SECONDARY: I/O    */
    /* --------------------- */

    //! Get the rotation solution vector from the sky-projected map
    template <class T>
    VectorT<Scalar<T>> Secondary<T>::getR() const {
        return skyMap.getR();
    }

    //! Get the occultation solution vector from the sky-projected map
    template <class T>
    VectorT<Scalar<T>> Secondary<T>::getS() const {
        return skyMap.getS();
    }

    //! Set the semi-major axis
    template <class T>
    void Secondary<T>::setSemi(const Scalar<T>& a_) {
        if (a_ > 0) a = a_;
        else throw errors::ValueError("Semi-major axis must be positive.");
        vamp = angvelorb * a / sqrt(1 - ecc2);
    }

    //! Get the semi-major axis
    template <class T>
    Scalar<T> Secondary<T>::getSemi() const {
        return a;
    }

    //! Set the orbital period
    template <class T>
    void Secondary<T>::setOrbPer(const Scalar<T>& porb_) {
        if (porb_ > 0) porb = porb_ * units::DayToSeconds;
        else throw errors::ValueError("Orbital period must be "
                                      "greater than zero.");
        angvelorb = (2 * pi<Scalar<T>>()) / porb;
        vamp = angvelorb * a / sqrt(1 - ecc2);
        computeTheta0();
    }

    //! Get the orbital period
    template <class T>
    Scalar<T> Secondary<T>::getOrbPer() const {
        return porb / units::DayToSeconds;
    }

    //! Set the inclination
    template <class T>
    void Secondary<T>::setInc(const Scalar<T>& inc_) {
        if ((inc_ >= 0) && (inc_ <= 180.0)) inc = inc_ * pi<Scalar<T>>() / 180.0;
        else throw errors::ValueError("Inclination must be "
                                      "in the range [0, 180].");
        cosi = cos(inc);
        sini = sin(inc);
        cosOcosi = cosO * cosi;
        sinOcosi = sinO * cosi;
    }

    //! Get the inclination
    template <class T>
    Scalar<T> Secondary<T>::getInc() const {
        return inc * 180.0 / pi<Scalar<T>>();
    }

    //! Set the eccentricity
    template <class T>
    void Secondary<T>::setEcc(const Scalar<T>& ecc_) {
        if ((ecc_ >= 0) && (ecc_ < 1)) ecc = ecc_;
        else throw errors::ValueError("Eccentricity must be "
                                      "in the range [0, 1).");
        sqrtonepluse = sqrt(1 + ecc);
        sqrtoneminuse = sqrt(1 - ecc);
        ecc2 = ecc * ecc;
        ecw = ecc * cos(w);
        esw = ecc * sin(w);
        vamp = angvelorb * a / sqrt(1 - ecc2);
        computeTheta0();
    }

    //! Get the eccentricity
    template <class T>
    Scalar<T> Secondary<T>::getEcc() const {
        return ecc;
    }

    //! Set the longitude of pericenter
    template <class T>
    void Secondary<T>::setVarPi(const Scalar<T>& w_) {
        w = mod2pi(w_ * pi<Scalar<T>>() / 180.0);
        M0 = lambda0 - w;
        ecw = ecc * cos(w);
        esw = ecc * sin(w);
        computeTheta0();
    }

    //! Get the longitude of pericenter
    template <class T>
    Scalar<T> Secondary<T>::getVarPi() const {
        return w * 180.0 / pi<Scalar<T>>();
    }

    //! Set the longitude of ascending node
    template <class T>
    void Secondary<T>::setOmega(const Scalar<T>& Om_) {
        Omega = mod2pi(Om_ * pi<Scalar<T>>() / 180.0);
        cosO = cos(Omega);
        sinO = sin(Omega);
        cosOcosi = cosO * cosi;
        sinOcosi = sinO * cosi;
    }

    //! Get the longitude of ascending node
    template <class T>
    Scalar<T> Secondary<T>::getOmega() const {
        return Omega * 180.0 / pi<Scalar<T>>();
    }

    //! Set the mean longitude at the reference time
    template <class T>
    void Secondary<T>::setLambda0(const Scalar<T>& lambda0_) {
        lambda0 = mod2pi(lambda0_ * pi<Scalar<T>>() / 180.0);
        M0 = lambda0 - w;
        computeTheta0();
    }

    //! Get the mean longitude at the reference time
    template <class T>
    Scalar<T> Secondary<T>::getLambda0() const {
        return lambda0 * 180.0 / pi<Scalar<T>>();
    }

    //! Get the body's x position vector
    template <class T>
    const Vector<Scalar<T>>& Secondary<T>::getXVector() const {
        return xvec;
    }

    //! Get the body's y position vector
    template <class T>
    const Vector<Scalar<T>>& Secondary<T>::getYVector() const {
        return yvec;
    }

    //! Get the body's z position vector
    template <class T>
    const Vector<Scalar<T>>& Secondary<T>::getZVector() const {
        return zvec;
    }

    //! Human-readable string
    template <class T>
    std::string Secondary<T>::info() {
        std::ostringstream os;
        std::string multi;
        if (isMulti(Scalar<T>(0.)))
            multi = "True";
        else
            multi = "False";
        os << "<starry.kepler.Secondary("
           << "lmax=" << this->lmax << ", "
           << "nwav=" << this->nwav << ", "
           << "multi=" << multi
           << ")>";
        return std::string(os.str());
    }

    /* ----------------- */
    /*       SYSTEM      */
    /* ----------------- */

    //! Keplerian system class
    template <class T>
    class System {

        protected:

            using S = Scalar<T>;                                                /**< Shorthand for the scalar type (double, Multi, ...) */
            Matrix<Scalar<T>> lightcurve;                                       /**< The full system light curve */
            Vector<T> dL;                                                       /**< The gradient of the system light curve */
            std::vector<std::string> dL_names;                                  /**< The names of each of the derivatives in the gradient */
            Scalar<T> exptime;                                                  /**< Exposure time in days */
            Scalar<T> exptol;                                                   /**< Exposure integration tolerance */
            int expmaxdepth;                                                    /**< Maximum recursion depth in the exposure integration */
            size_t t;                                                           /** The current index in the time array */
            size_t ngrad;                                                       /** Number of derivatives to compute */
            size_t g;                                                           /** The current gradient index */
            bool computed;                                                      /** Did the user call `compute()` yet? */

            // Protected methods
            inline void step(const S& time_cur, bool gradient, bool numerical);
            Exposure<T> step(const S& time_cur, bool gradient, bool numerical, bool store_xyz);
            Exposure<T> integrate(const Exposure<T>& f1, const Exposure<T>& f2,
                                  const S& t1, const S& t2,
                                  int depth, bool gradient, bool numerical);
            inline void integrate(const S& time_cur, bool gradient, bool numerical);

            inline void computePrimaryTotalGradient(const S& time_cur);
            inline void computeSecondaryTotalGradient(const S& time_cur,
                Secondary<T>* secondary);
            inline void computePrimaryOccultationGradient(const S& time_cur,
                Secondary<T>* occultor);
            inline void computeSecondaryOccultationGradient(const S& time_cur,
                Secondary<T>* secondary);
            inline void computeSecondaryOccultationGradient(const S& time_cur,
                Secondary<T>* secondary, Secondary<T>* occultor);

        public:

            Primary<T>* primary;                                                /**< Pointer to the primary body */
            std::vector<Secondary<T>*> secondaries;                             /**< Vector of pointer(s) to the secondary body(ies) */

            //! Constructor: single secondary
            explicit System(Primary<T>* primary,
                            Secondary<T>* secondary) :
                            primary(primary)
            {
                secondaries.push_back(secondary);
                setExposureTime(0.0);
                setExposureTol(sqrt(mach_eps<Scalar<T>>()));
                setExposureMaxDepth(4);
                computed = false;
            }

            //! Constructor: multiple secondaries
            explicit System(Primary<T>* primary,
                            std::vector<Secondary<T>*> secondaries) :
                            primary(primary),
                            secondaries(secondaries)
            {
                setExposureTime(0.0);
                setExposureTol(sqrt(mach_eps<Scalar<T>>()));
                setExposureMaxDepth(4);
                computed = false;
            }

            // Public methods
            void compute(const Vector<S>& time, bool gradient=false, bool numerical=false);
            const Matrix<S>& getLightcurve() const;
            const Vector<T>& getLightcurveGradient() const;
            const std::vector<std::string>& getLightcurveGradientNames() const;
            std::string info();
            void setExposureTime(const S& t_);
            S getExposureTime() const;
            void setExposureTol(const S& t_);
            S getExposureTol() const;
            void setExposureMaxDepth(const int d_);
            int getExposureMaxDepth() const;

    };

    //! Set the exposure time in days
    template <class T>
    void System<T>::setExposureTime(const Scalar<T>& t_) {
        exptime = t_ * units::DayToSeconds;
    }

    //! Get the exposure time in days
    template <class T>
    Scalar<T> System<T>::getExposureTime() const {
        return exptime / units::DayToSeconds;
    }

    //! Set the exposure tolerance
    template <class T>
    void System<T>::setExposureTol(const Scalar<T>& t_) {
        exptol = t_;
    }

    //! Get the exposure tolerance
    template <class T>
    Scalar<T> System<T>::getExposureTol() const {
        return exptol;
    }

    //! Set the maximum exposure depth
    template <class T>
    void System<T>::setExposureMaxDepth(const int d_) {
        expmaxdepth = d_;
    }

    //! Get the maximum exposure depth
    template <class T>
    int System<T>::getExposureMaxDepth() const {
        return expmaxdepth;
    }

    //! Return a human-readable info string
    template <class T>
    std::string System<T>::info() {
        return "<starry.kepler.System()>";
    }

    //! Return the full system light curve
    template <class T>
    const Matrix<Scalar<T>>& System<T>::getLightcurve() const {
        if (!computed)
            throw errors::ValueError("Please call the `compute` method first.");
        return lightcurve;
    }

    //! Return the gradient of the light curve
    template <class T>
    const Vector<T>& System<T>::getLightcurveGradient() const {
        if (!computed)
            throw errors::ValueError("Please call the `compute` method first.");
        return dL;
    }

    //! Get the names of the params in the gradient
    template <class T>
    const std::vector<std::string>& System<T>::getLightcurveGradientNames()
            const {
        return dL_names;
    }

    /**
    Recursive exposure time integration function (single iteration)

    */
    template <class T>
    Exposure<T> System<T>::integrate(const Exposure<T>& f1,
                                     const Exposure<T>& f2,
                                     const Scalar<T>& t1,
                                     const Scalar<T>& t2,
                                     int depth, bool gradient,
                                     bool numerical) {
        Scalar<T> tmid = (t1 + t2) * 0.5;
        // If this is the first time we're recursing (depth == 0),
        // store the xyz position of the bodies in the output vectors
        Exposure<T> fmid = step(tmid, gradient, numerical, depth == 0);
        Exposure<T> fapprox = (f1 + f2) * 0.5;
        Exposure<T> d = fmid - fapprox;
        Exposure<T> a(secondaries.size(), gradient);
        Exposure<T> b(secondaries.size(), gradient);
        if (depth < expmaxdepth) {
            for (size_t i = 0; i < secondaries.size() + 1; ++i) {
                for (int n = 0; n < primary->nwav; ++n) {
                    if (abs(getIndex(d.flux[i], n)) > exptol) {
                        a = integrate(f1, fmid, t1, tmid, depth + 1, gradient, numerical);
                        b = integrate(fmid, f2, tmid, t2, depth + 1, gradient, numerical);
                        return a + b;
                    }
                }
            }
        }
        return fapprox * (t2 - t1);
    }

    /**
    Recursive exposure time integration function

    */
    template <class T>
    inline void System<T>::integrate(const Scalar<T>& time_cur, bool gradient, bool numerical) {
        Exposure<T> exposure(secondaries.size(), gradient);
        Scalar<T> dt = 0.5 * exptime,
                  t1 = time_cur - dt,
                  t2 = time_cur + dt,
                  invdt = 1. / (t2 - t1);
        exposure = integrate(step(t1, gradient, numerical, false),
                             step(t2, gradient, numerical, false),
                             t1, t2, 0, gradient, numerical) * invdt;
        primary->flux_cur = exposure.flux[0];
        if (gradient)
            primary->dflux_cur = exposure.gradient[0];
        for (size_t i = 0; i < secondaries.size(); ++i) {
            secondaries[i]->flux_cur = exposure.flux[i + 1];
            if (gradient)
                secondaries[i]->dflux_cur = exposure.gradient[i + 1];
        }
    }

    /**
    Take a single orbital + photometric step.

    */
    template <class T>
    inline void System<T>::step(const Scalar<T>& time_cur, bool gradient, bool numerical) {

        Scalar<T> xo, yo, ro;
        size_t NS = secondaries.size();
        size_t o, p;

        // Compute fluxes and take an orbital step
        primary->computeTotal(time_cur, gradient, numerical);
        if (gradient)
            computePrimaryTotalGradient(time_cur);
        for (auto secondary : secondaries) {
            secondary->computeXYZ(time_cur, gradient);
            secondary->computeTotal(time_cur, gradient, numerical);
            if (gradient)
                computeSecondaryTotalGradient(time_cur, secondary);
        }

        // Compute occultations involving the primary
        for (auto secondary : secondaries) {
            Scalar<T> bsq = secondary->x_cur * secondary->x_cur +
                            secondary->y_cur * secondary->y_cur;
            if (bsq < (1 + secondary->r) * (1 + secondary->r)) {
                if (secondary->z_cur > 0) {
                    primary->occult(time_cur, secondary->x_cur,
                                    secondary->y_cur, secondary->r,
                                    gradient, numerical);
                    if (gradient)
                        computePrimaryOccultationGradient(time_cur, secondary);
                } else if (!allZero(secondary->L)) {
                    ro = 1. / secondary->r;
                    secondary->occult(time_cur, -ro * secondary->x_cur,
                                      -ro * secondary->y_cur, ro,
                                      gradient, numerical);
                    if (gradient)
                        computeSecondaryOccultationGradient(time_cur,
                                                            secondary);
                }
            }
        }

        // Compute occultations among the secondaries
        for (size_t i = 0; i < NS; i++) {
            for (size_t j = i + 1; j < NS; j++) {
                if (secondaries[j]->z_cur > secondaries[i]->z_cur) {
                    o = j;
                    p = i;
                } else {
                    o = i;
                    p = j;
                }
                if (!allZero(secondaries[p]->L)) {
                    ro = 1. / secondaries[p]->r;
                    xo = ro * (secondaries[o]->x_cur - secondaries[p]->x_cur);
                    yo = ro * (secondaries[o]->y_cur - secondaries[p]->y_cur);
                    ro = ro * secondaries[o]->r;
                    if (xo * xo + yo * yo < (1 + ro) * (1 + ro)) {
                        secondaries[p]->occult(time_cur, xo, yo, ro, gradient, numerical);
                        if (gradient)
                            computeSecondaryOccultationGradient(time_cur,
                                                                secondaries[p],
                                                                secondaries[o]);
                    }
                }
            }
        }

    }

    /**
    Take a single orbital + photometric step
    (exposure time integration overload).

    */
    template <class T>
    inline Exposure<T> System<T>::step(const Scalar<T>& time, bool gradient,
                                       bool numerical, bool store_xyz) {

        // Take the step
        step(time, gradient, numerical);

        // Collect the current values of the flux for each body
        // We compare them to the previous iteration to determine
        // whether we need to recurse.
        size_t NS = secondaries.size();
        Exposure<T> exposure(NS, gradient);
        exposure.flux[0] = primary->flux_cur;
        if (gradient)
            exposure.gradient[0] = primary->dflux_cur;
        for (size_t n = 0; n < NS; ++n) {
            if (store_xyz) {
                // If this is the midpoint of the integration,
                // store the cartesian position of the bodies
                secondaries[n]->xvec(t) = secondaries[n]->x_cur;
                secondaries[n]->yvec(t) = secondaries[n]->y_cur;
                secondaries[n]->zvec(t) = secondaries[n]->z_cur;
            }
            exposure.flux[n + 1] = secondaries[n]->flux_cur;
            if (gradient)
                exposure.gradient[n + 1] = secondaries[n]->dflux_cur;
        }

        // Return the flux from each of the bodies
        return exposure;

    }

    /**
    Compute the full system light curve.

    */
    template <class T>
    void System<T>::compute(const Vector<Scalar<T>>& time_, bool gradient, bool numerical) {

        size_t NT = time_.size();
        Vector<Scalar<T>> time = time_ * units::DayToSeconds;
        int iletter = 98;                                                       // This is the ASCII code for 'b'
        std::string letter;                                                     // The secondary letter designation
        computed = true;

        // Primary:
        // - Allocate arrays
        // - Figure out the number and names of derivatives
        lightcurve.resize(NT, primary->nwav);
        primary->lightcurve.resize(NT, primary->nwav);
        primary->computed = true;
        if (gradient) {
            // Resize the outer arrays and
            // populate the primary gradient names
            dL.resize(NT);
            primary->resizeGradient();
            dL_names.clear();
            dL_names.push_back("time");
            for (std::string name : PRIMARY_GRAD_NAMES)
                dL_names.push_back("A." + name);
            // Note that we skip the Y_{0,0} deriv
            for (size_t n=5; n < primary->dF_names.size(); ++n)
                dL_names.push_back("A." + primary->dF_names[n]);
            ngrad = dL_names.size();
        } else {
            ngrad = 0;
        }
        if (!allOnes(primary->getY(0, 0)))
            throw errors::ValueError("Bodies instantiated via the Kepler "
                                     "module must all have Y_{0, 0} = 1. "
                                     "You probably want to change the "
                                     "body's luminosity instead.");

        // Secondaries:
        // - Allocate arrays
        // - Sync the orbital and sky maps
        // - Sync the speed of light across all secondaries
        // - Figure out the number and names of derivatives
        for (auto secondary : secondaries) {
            if (secondary->nwav != primary->nwav)
                throw errors::ValueError("All bodies must have the same "
                                         "wavelength grid.");
            if (!allOnes(secondary->getY(0, 0)))
                throw errors::ValueError("Bodies instantiated via the Kepler "
                                         "module must all have Y_{0, 0} = 1. "
                                         "You probably want to change the "
                                         "body's luminosity instead.");
            secondary->xvec.resize(NT);
            secondary->yvec.resize(NT);
            secondary->zvec.resize(NT);
            secondary->lightcurve.resize(NT, primary->nwav);
            secondary->syncSkyMap();
            secondary->c_light = &(primary->c_light);
            secondary->computed = true;
            if (gradient) {
                // Resize the outer arrays
                // and populate the secondary gradient names
                letter = (char) iletter++;
                secondary->resizeGradient();
                for (std::string name : SECONDARY_GRAD_NAMES)
                    dL_names.push_back(letter + "." + name);
                // Note that we skip the Y_{0,0} deriv
                for (size_t n=5; n < secondary->dF_names.size(); ++n)
                    dL_names.push_back(letter + "." + secondary->dF_names[n]);
                secondary->g0 = ngrad;
                ngrad = dL_names.size();
            }
        }

        // Sync the derivs across all bodies
        if (gradient) {
            primary->dL.resize(NT);
            primary->dL_names = dL_names;
            primary->ngrad = ngrad;
            for (auto secondary : secondaries) {
                secondary->dL.resize(NT);
                secondary->dL_names = dL_names;
                secondary->ngrad = ngrad;
            }
        }

        // Loop through the timeseries
        for (t = 0; t < NT; ++t){

            // Take an orbital step and compute the fluxes
            if (exptime == 0)
                step(time(t), gradient, numerical);
            else
                integrate(time(t), gradient, numerical);

            // Update the light curves and orbital positions
            for (int n = 0; n < primary->nwav; ++n) {
                primary->lightcurve(t, n) = getColumn(primary->flux_cur, n);
                lightcurve(t, n) = getColumn(primary->flux_cur, n);
            }
            if (gradient) {
                primary->dL(t) = primary->dflux_cur;
                dL(t) = primary->dL(t);
            }
            for (auto secondary : secondaries) {
                if (exptime == 0) {
                    secondary->xvec(t) = secondary->x_cur;
                    secondary->yvec(t) = secondary->y_cur;
                    secondary->zvec(t) = secondary->z_cur;
                }
                for (int n = 0; n < primary->nwav; ++n) {
                    secondary->lightcurve(t, n) =
                        getColumn(secondary->flux_cur, n);
                    lightcurve(t, n) += getColumn(secondary->flux_cur, n);
                }
                if (gradient) {
                    secondary->dL(t) = secondary->dflux_cur;
                    dL(t) += secondary->dL(t);
                }
            }

        }
    }

    /**
    Compute the gradient of the primary's total flux.

    */
    template <class T>
    inline void System<T>::computePrimaryTotalGradient(const S& time_cur) {

        // Allocate memory; reset the gradient index
        primary->dflux_tot.setZero(ngrad, primary->nwav);
        g = 0;

        // dF / dt from dtheta / dt
        setRow(primary->dflux_tot, g++,
               Row<T>(getRow(primary->dF, 0) *
                      primary->angvelrot_deg * units::DayToSeconds));

        // dF / dprot from dtheta / dt
        setRow(primary->dflux_tot, g++,
               Row<T>(-getRow(primary->dF, 0) *
               (primary->angvelrot_deg *
               (time_cur - primary->tref - primary->delay) / primary->prot +
               primary->theta0_deg / primary->prot) * units::DayToSeconds));

        // dF / dtref from dtheta / dt
        setRow(primary->dflux_tot, g++,
            Row<T>(-getRow(primary->dF, 0) *
                   primary->angvelrot_deg * units::DayToSeconds));

        // dF / d{y} and dF / d{u} from the map derivs
        // Note that we skip the Y_{0,0} deriv
        int sz = primary->dF.rows() - 5;
        primary->dflux_tot.block(g, 0, sz, primary->nwav) =
            primary->dF.block(5, 0, sz, primary->nwav);

        // Update current flux derivative
        primary->dflux_cur = primary->dflux_tot;

    }

    /**
    Compute the gradient of the secondary's total flux.

    */
    template <class T>
    inline void System<T>::computeSecondaryTotalGradient(const S& time_cur,
            Secondary<T>* secondary) {

        // Allocate memory for this secondary
        secondary->dflux_tot.setZero(ngrad, secondary->nwav);

        // NOTE: If L = 0, we don't actually call the `flux()` routine,
        // so we set all derivs to zero. In reality, dF / dL is nonzero,
        // so this is *technically* incorrect. But computing `flux()`
        // just to get this single derivative (which will likely never
        // be used) seems silly. To override this, users can just set the
        // body's luminosity to a very small nonzero value (~1e-15).
        if (!allZero(secondary->L)) {

            // Time delay corrections to the derivatives from
            // the dependence of `theta` on `delay`
            Row<T> corr = -cwiseProduct(secondary->L, getRow(secondary->dF, 0)) *
                          secondary->angvelrot_deg;

            // Starting index for the secondary's derivs
            g = secondary->g0;

            // dF / dt from dtheta / dt
            setRow(secondary->dflux_tot, 0,
                   Row<T>(cwiseProduct(secondary->L, getRow(secondary->dF, 0)) *
                          secondary->angvelrot_deg * units::DayToSeconds +
                          corr * secondary->AD.delay.derivatives()(0) *
                          units::DayToSeconds));

            // dF / dr
            setRow(secondary->dflux_tot, g++, 0.0);

            // dF / dL
            setRow(secondary->dflux_tot, g++,
                   Row<T>(cwiseQuotient(secondary->flux_tot, secondary->L)));

            // dF / dprot from dtheta / dt
            setRow(secondary->dflux_tot, g++,
                   Row<T>(-cwiseProduct(secondary->L, getRow(secondary->dF, 0)) *
                         (secondary->angvelrot_deg *
                         (time_cur - secondary->tref - secondary->delay) /
                          secondary->prot +
                          secondary->theta0_deg / secondary->prot) *
                          units::DayToSeconds));

            // dF / da
            setRow(secondary->dflux_tot, g++,
                   Row<T>(corr * secondary->AD.delay.derivatives()(1)));

            // dF / dporb
            setRow(secondary->dflux_tot, g++,
                   Row<T>(cwiseProduct(secondary->L, getRow(secondary->dF, 0)) *
                          secondary->theta0_deg / secondary->porb *
                          units::DayToSeconds +
                          corr * secondary->AD.delay.derivatives()(5) *
                          units::DayToSeconds));

            // dF / dinc
            // Tricky because the sky rotation transform depends on inc
            // NOTE: There's a more intelligent way of doing this; see
            // `dF / dinc` in `computeSecondaryOccultationGradient()`
            if (secondary->y_deg > 0) {
                T y_transf(secondary->N, secondary->nwav);
                for (int n = 0; n < secondary->nwav; ++n) {
                    for (int l = 0; l < secondary->lmax + 1; ++l) {
                        y_transf.block(l * l, n, 2 * l + 1, 1) =
                            secondary->skyMap.W.R[l] *
                            secondary->W1.dRdtheta[l] *
                            secondary->W2.R[l] *
                            secondary->y.block(l * l, n, 2 * l + 1, 1);
                    }
                }
                setRow(secondary->dflux_tot, g++,
                       Row<T>(cwiseProduct(secondary->L, dot(secondary->B.rTA1, y_transf)) *
                              (-pi<Scalar<T>>() / 180.) +
                              corr * secondary->AD.delay.derivatives()(8) *
                              pi<Scalar<T>>() / 180.0));
            }

            // dF / decc
            setRow(secondary->dflux_tot, g++,
                   Row<T>(cwiseProduct(secondary->L, getRow(secondary->dF, 0)) *
                          secondary->dtheta0_degde +
                          corr * secondary->AD.delay.derivatives()(2)));

            // dF / dw; note that we must account for d(delay) / dM0(w)
            setRow(secondary->dflux_tot, g++,
                   Row<T>(cwiseProduct(secondary->L, getRow(secondary->dF, 0)) *
                          secondary->dtheta0_degdw * pi<Scalar<T>>() / 180.0 +
                          corr * (secondary->AD.delay.derivatives()(6) -
                                  secondary->AD.delay.derivatives()(3)) *
                          pi<Scalar<T>>() / 180.0));

            // dF / dOmega; note that time delay correction is always zero
            // Tricky because the sky rotation transform depends on Omega
            // NOTE: There's a more intelligent way of doing this; see
            // `dF / dOmega` in `computeSecondaryOccultationGradient()`
            if (secondary->y_deg > 0) {
                T y_transf(secondary->N, secondary->nwav);
                for (int n = 0; n < secondary->nwav; ++n) {
                    for (int l = 0; l < secondary->lmax + 1; ++l) {
                        y_transf.block(l * l, n, 2 * l + 1, 1) =
                            secondary->skyMap.W.R[l] *
                            secondary->W1.R[l] *
                            secondary->W2.dRdtheta[l] *
                            secondary->y.block(l * l, n, 2 * l + 1, 1);
                    }
                }
                setRow(secondary->dflux_tot, g++,
                       Row<T>(cwiseProduct(secondary->L, dot(secondary->B.rTA1, y_transf)) *
                              (pi<Scalar<T>>() / 180.)));
            }

            // dF / dlambda0
            setRow(secondary->dflux_tot, g++,
                   Row<T>(cwiseProduct(secondary->L, getRow(secondary->dF, 0)) *
                          secondary->porb / secondary->prot +
                          corr * secondary->AD.delay.derivatives()(3) *
                          pi<Scalar<T>>() / 180.0));

            // dF / dtref from dtheta / dt
            setRow(secondary->dflux_tot, g++,
                   Row<T>(-cwiseProduct(secondary->L, getRow(secondary->dF, 0)) *
                          secondary->angvelrot_deg * units::DayToSeconds +
                          corr * secondary->AD.delay.derivatives()(4) *
                          units::DayToSeconds));

            // dF / d{y} and dF / d{u} from the map derivs
            // Note that we skip the Y_{0,0} deriv
            int sz = secondary->dF.rows() - 5;
            for (int n = 0; n < secondary->nwav; ++n) {
                secondary->dflux_tot.block(g, n, sz, n + 1) = getIndex(secondary->L, n) * secondary->dF.block(5, n, sz, n + 1);
            }
        }

        // Update current flux derivative
        secondary->dflux_cur = secondary->dflux_tot;

    }

    /**
    Compute the gradient of the primary's flux in occultation.
    This function computes

        dflux_cur += L * dF - dflux_tot

    */
    template <class T>
    inline void System<T>::computePrimaryOccultationGradient(const S& time_cur,
        Secondary<T>* occultor) {

        // ** First, derivs with respect to the primary's own properties **
        // t, prot, tref, and map coeffs

        // dF / dt
        g = 0;
        setRow(primary->dflux_cur, g, Row<T>(
               getRow(primary->dflux_cur, g) -
               getRow(primary->dflux_tot, g) +
               (getRow(primary->dF, 0) * primary->angvelrot_deg +               // dtheta / dt
                getRow(primary->dF, 1) * occultor->AD.x.derivatives()(0) +      // dxo / dt
                getRow(primary->dF, 2) * occultor->AD.y.derivatives()(0)) *     // dyo / dt
               units::DayToSeconds));
        g++;

        // dF / prot
        setRow(primary->dflux_cur, g, Row<T>(
               getRow(primary->dflux_cur, g) -
               getRow(primary->dflux_tot, g) -
               getRow(primary->dF, 0) *
               (primary->angvelrot_deg *
               (time_cur - primary->tref - primary->delay) / primary->prot +
               primary->theta0_deg / primary->prot) * units::DayToSeconds));
        g++;

        // dF / dtref
        setRow(primary->dflux_cur, g, Row<T>(
               getRow(primary->dflux_cur, g) -
               getRow(primary->dflux_tot, g) -
               getRow(primary->dF, 0) *
               primary->angvelrot_deg * units::DayToSeconds));
        g++;

        // dF / d{y} and dF / d{u}
        // Note that we skip the Y_{0,0} deriv
        int sz = primary->dF.rows() - 5;
        primary->dflux_cur.block(g, 0, sz, primary->nwav) +=
            primary->dF.block(5, 0, sz, primary->nwav) -
            primary->dflux_tot.block(g, 0, sz, primary->nwav);

        // ** Now the derivs with respect to the occultor's properties **
        // r, L, prot, a, porb, inc, ecc, w, Omega, lambda0, tref

        // Starting index for the secondary's derivs
        g = occultor->g0;

        // dF / dr
        setRow(primary->dflux_cur, g, Row<T>(
               getRow(primary->dflux_cur, g) -
               getRow(primary->dflux_tot, g) +
               getRow(primary->dF, 3)));
        g += 3;                                                                 // dF / dL and dF / dprot are zero

        // dF / da
        setRow(primary->dflux_cur, g, Row<T>(
               getRow(primary->dflux_cur, g) -
               getRow(primary->dflux_tot, g) +
               (getRow(primary->dF, 1) * occultor->AD.x.derivatives()(1) +      // dxo / da
                getRow(primary->dF, 2) * occultor->AD.y.derivatives()(1))));    // dyo / da
        g++;

        // dF / dporb
        setRow(primary->dflux_cur, g, Row<T>(
               getRow(primary->dflux_cur, g) -
               getRow(primary->dflux_tot, g) +
               (getRow(primary->dF, 1) * occultor->AD.x.derivatives()(5) +      // dxo / dporb
                getRow(primary->dF, 2) * occultor->AD.y.derivatives()(5)) *     // dyo / dporb
               units::DayToSeconds));
        g++;

        // dF / dinc
        setRow(primary->dflux_cur, g, Row<T>(
               getRow(primary->dflux_cur, g) -
               getRow(primary->dflux_tot, g) +
               (getRow(primary->dF, 1) * occultor->AD.x.derivatives()(8) +      // dxo / dinc
                getRow(primary->dF, 2) * occultor->AD.y.derivatives()(8)) *     // dyo / dinc
               pi<Scalar<T>>() / 180.0));
        g++;

        // dF / decc
        setRow(primary->dflux_cur, g, Row<T>(
               getRow(primary->dflux_cur, g) -
               getRow(primary->dflux_tot, g) +
               (getRow(primary->dF, 1) * occultor->AD.x.derivatives()(2) +      // dxo / decc
                getRow(primary->dF, 2) * occultor->AD.y.derivatives()(2))));    // dyo / decc
        g++;

        // dF / dw
        setRow(primary->dflux_cur, g, Row<T>(
               getRow(primary->dflux_cur, g) -
               getRow(primary->dflux_tot, g) +
               (getRow(primary->dF, 1) * (occultor->AD.x.derivatives()(6) -
                                          occultor->AD.x.derivatives()(3)) +    // dxo / dw
                getRow(primary->dF, 2) * (occultor->AD.y.derivatives()(6) -
                                          occultor->AD.y.derivatives()(3))) *   // dyo / dw
               pi<Scalar<T>>() / 180.0));
        g++;

        // dF / dOmega
        setRow(primary->dflux_cur, g, Row<T>(
               getRow(primary->dflux_cur, g) -
               getRow(primary->dflux_tot, g) +
               (getRow(primary->dF, 1) * occultor->AD.x.derivatives()(7) +      // dxo / dOmega
                getRow(primary->dF, 2) * occultor->AD.y.derivatives()(7)) *     // dyo / dOmega
               pi<Scalar<T>>() / 180.0));
        g++;

        // dF / dlambda0
        setRow(primary->dflux_cur, g, Row<T>(
               getRow(primary->dflux_cur, g) -
               getRow(primary->dflux_tot, g) +
               (getRow(primary->dF, 1) * occultor->AD.x.derivatives()(3) +      // dxo / dlambda0
                getRow(primary->dF, 2) * occultor->AD.y.derivatives()(3)) *     // dyo / dlambda0
               pi<Scalar<T>>() / 180.0));
        g++;

        // dF / dtref
        setRow(primary->dflux_cur, g, Row<T>(
               getRow(primary->dflux_cur, g) -
               getRow(primary->dflux_tot, g) +
               (getRow(primary->dF, 1) * occultor->AD.x.derivatives()(4) +      // dxo / dtref
                getRow(primary->dF, 2) * occultor->AD.y.derivatives()(4)) *     // dyo / dtref
               units::DayToSeconds));
        g++;

    }

    /**
    Compute the gradient of the secondary's flux during an occultation by
    the primary.

    This function computes

        dflux_cur += L * dF - dflux_tot

    via EXTENSIVE use of the chain rule.

    */
    template <class T>
    inline void System<T>::computeSecondaryOccultationGradient(const S& time_cur,
            Secondary<T>* secondary) {

        // See note in `computeSecondaryTotalGradient()`
        if (!allZero(secondary->L)) {

            // ** Pre-compute some stuff **

            // Occultor Radius
            Scalar<T> ro = 1. / secondary->r;

            // Starting index for the secondary's derivs
            g = secondary->g0;

            // ** First, derivs with respect to the secondary's own props **
            // t, r, L, prot, a, porb, inc, ecc, w, Omega, lambda0, tref

            // dF / dt
            setRow(secondary->dflux_cur, 0, Row<T>(
                   getRow(secondary->dflux_cur, 0) -
                   getRow(secondary->dflux_tot, 0) +
                   cwiseProduct(secondary->L, Row<T>
                   (getRow(secondary->dF, 0) * secondary->angvelrot_deg -
                    getRow(secondary->dF, 0) * secondary->angvelrot_deg *
                        secondary->AD.delay.derivatives()(0) -
                    ro * getRow(secondary->dF, 1) *
                        secondary->AD.x.derivatives()(0) -
                    ro * getRow(secondary->dF, 2) *
                        secondary->AD.y.derivatives()(0))) *
                   units::DayToSeconds));

            // dF / dr
            setRow(secondary->dflux_cur, g, Row<T>(
                   getRow(secondary->dflux_cur, g) -
                   getRow(secondary->dflux_tot, g) +
                   ro * ro *
                   cwiseProduct(secondary->L, Row<T>
                   (secondary->x_cur * getRow(secondary->dF, 1) +
                    secondary->y_cur * getRow(secondary->dF, 2) -
                    getRow(secondary->dF, 3)))));
            g++;

            // dF / dL
            setRow(secondary->dflux_cur, g, Row<T>(
                   getRow(secondary->dflux_cur, g) -
                   getRow(secondary->dflux_tot, g) +
                   cwiseQuotient(secondary->flux_cur, secondary->L)));
            g++;

            // dF / dprot
            setRow(secondary->dflux_cur, g, Row<T>(
                   getRow(secondary->dflux_cur, g) -
                   getRow(secondary->dflux_tot, g) -
                   cwiseProduct(secondary->L, getRow(secondary->dF, 0)) *
                   (secondary->angvelrot_deg *
                   (time_cur - secondary->tref - secondary->delay) / secondary->prot +
                   secondary->theta0_deg / secondary->prot) * units::DayToSeconds));
            g++;

            // dF / da
            setRow(secondary->dflux_cur, g, Row<T>(
                   getRow(secondary->dflux_cur, g) -
                   getRow(secondary->dflux_tot, g) -
                   cwiseProduct(secondary->L, Row<T>(
                       getRow(secondary->dF, 0) * secondary->angvelrot_deg *
                            secondary->AD.delay.derivatives()(1) +
                       getRow(secondary->dF, 1) *
                            secondary->AD.x.derivatives()(1) * ro +
                       getRow(secondary->dF, 2) *
                            secondary->AD.y.derivatives()(1) * ro))));
            g++;

            // dF / porb
            setRow(secondary->dflux_cur, g, Row<T>(
                   getRow(secondary->dflux_cur, g) -
                   getRow(secondary->dflux_tot, g) -
                   units::DayToSeconds *
                   cwiseProduct(secondary->L, Row<T>(
                       getRow(secondary->dF, 0) * secondary->angvelrot_deg *
                            secondary->AD.delay.derivatives()(5) -
                       getRow(secondary->dF, 0) *
                            secondary->theta0_deg / secondary->porb +
                       getRow(secondary->dF, 1) *
                            secondary->AD.x.derivatives()(5) * ro +
                       getRow(secondary->dF, 2) *
                            secondary->AD.y.derivatives()(5) * ro))));
            g++;

            // dF / dinc
            // Tricky because the sky rotation transform depends on I
            // TODO: This can be sped up a ton
            if (secondary->y_deg > 0) {
                Row<T> dFdinc;
                resize(dFdinc, secondary->N, secondary->nwav);
                for (int n = 0; n < secondary->nwav; ++n) {
                    Vector<Scalar<T>> dyskydinc(secondary->N);
                    VectorT<Scalar<T>> dFdysky =
                        secondary->skyMap.getGradient().
                            block(4, n, secondary->N, 1).transpose();
                    for (int l = 0; l < secondary->lmax + 1; ++l) {
                        dyskydinc.segment(l * l, 2 * l + 1) =
                            -secondary->W1.dRdtheta[l] *
                            secondary->W2.R[l] *
                            secondary->y.block(l * l, n, 2 * l + 1, 1);
                    }
                    setIndex(dFdinc, n, dFdysky.dot(dyskydinc));
                }
                setRow(secondary->dflux_cur, g, Row<T>(
                       getRow(secondary->dflux_cur, g) -
                       getRow(secondary->dflux_tot, g) +
                       pi<Scalar<T>>() / 180. *
                       cwiseProduct(secondary->L, Row<T>(
                       -getRow(secondary->dF, 0) *
                               secondary->angvelrot_deg *
                               secondary->AD.delay.derivatives()(8) +
                       dFdinc
                       - getRow(secondary->dF, 1) *
                            secondary->AD.x.derivatives()(8) * ro
                       - getRow(secondary->dF, 2) *
                            secondary->AD.y.derivatives()(8) * ro)))
                   );
            }
            g++;

            // dF / decc
            setRow(secondary->dflux_cur, g, Row<T>(
                   getRow(secondary->dflux_cur, g) -
                   getRow(secondary->dflux_tot, g) -
                   cwiseProduct(secondary->L, Row<T>(
                       getRow(secondary->dF, 0) * secondary->angvelrot_deg *
                            secondary->AD.delay.derivatives()(2) -
                       getRow(secondary->dF, 0) *
                            secondary->dtheta0_degde +
                       getRow(secondary->dF, 1) *
                            secondary->AD.x.derivatives()(2) * ro +
                       getRow(secondary->dF, 2) *
                            secondary->AD.y.derivatives()(2) * ro))));
            g++;

            // dF / dw; note that we must account for d / dM0(w)
            setRow(secondary->dflux_cur, g, Row<T>(
                   getRow(secondary->dflux_cur, g) -
                   getRow(secondary->dflux_tot, g) -
                   pi<Scalar<T>>() / 180.0 *
                   cwiseProduct(secondary->L, Row<T>(
                       getRow(secondary->dF, 0) * secondary->angvelrot_deg *
                            (secondary->AD.delay.derivatives()(6) -
                             secondary->AD.delay.derivatives()(3)) -
                       getRow(secondary->dF, 0) *
                            secondary->dtheta0_degdw +
                       getRow(secondary->dF, 1) *
                            (secondary->AD.x.derivatives()(6) -
                             secondary->AD.x.derivatives()(3)) * ro +
                       getRow(secondary->dF, 2) *
                            (secondary->AD.y.derivatives()(6) -
                             secondary->AD.y.derivatives()(3)) * ro))));
            g++;


            // dF / dOmega
            // Tricky because the sky rotation transform depends on Omega
            // TODO: This can be sped up a ton
            if (secondary->y_deg > 0) {
                Row<T> dFdOmega;
                resize(dFdOmega, secondary->N, secondary->nwav);
                for (int n = 0; n < secondary->nwav; ++n) {
                    Vector<Scalar<T>> dyskydOmega(secondary->N);
                    VectorT<Scalar<T>> dFdysky =
                        secondary->skyMap.getGradient().
                            block(4, n, secondary->N, 1).transpose();
                    for (int l = 0; l < secondary->lmax + 1; ++l) {
                        dyskydOmega.segment(l * l, 2 * l + 1) =
                            secondary->W1.R[l] *
                            secondary->W2.dRdtheta[l] *
                            secondary->y.block(l * l, n, 2 * l + 1, 1);
                    }
                    setIndex(dFdOmega, n, dFdysky.dot(dyskydOmega));
                }
                setRow(secondary->dflux_cur, g, Row<T>(
                       getRow(secondary->dflux_cur, g) -
                       getRow(secondary->dflux_tot, g) +
                       pi<Scalar<T>>() / 180. *
                       cwiseProduct(secondary->L, Row<T>(
                       dFdOmega
                       - getRow(secondary->dF, 1) *
                            secondary->AD.x.derivatives()(7) * ro
                       - getRow(secondary->dF, 2) *
                            secondary->AD.y.derivatives()(7) * ro)))
                   );
            }
            g++;

            // dF / dlambda0
            setRow(secondary->dflux_cur, g, Row<T>(
                   getRow(secondary->dflux_cur, g) -
                   getRow(secondary->dflux_tot, g) +
                   pi<Scalar<T>>() / 180.0 *
                   cwiseProduct(secondary->L, Row<T>(
                       getRow(secondary->dF, 0) *
                            (180.0 / pi<Scalar<T>>() *
                            secondary->porb / secondary->prot -
                            secondary->angvelrot_deg *
                            secondary->AD.delay.derivatives()(3)) -
                       getRow(secondary->dF, 1) *
                            secondary->AD.x.derivatives()(3) * ro -
                       getRow(secondary->dF, 2) *
                            secondary->AD.y.derivatives()(3) * ro))));
            g++;

            // dF / dtref
            setRow(secondary->dflux_cur, g, Row<T>(
                   getRow(secondary->dflux_cur, g) -
                   getRow(secondary->dflux_tot, g) -
                   units::DayToSeconds *
                   cwiseProduct(secondary->L, Row<T>(
                       getRow(secondary->dF, 0) *
                            secondary->angvelrot_deg *
                            (1 + secondary->AD.delay.derivatives()(4)) +
                       getRow(secondary->dF, 1) *
                            secondary->AD.x.derivatives()(4) * ro +
                       getRow(secondary->dF, 2) *
                            secondary->AD.y.derivatives()(4) * ro))));
            g++;

            // dF / d{y} and dF / d{u}
            // Note that we skip the Y_{0,0} deriv
            int sz = secondary->dF.rows() - 5;
            for (int n = 0; n < secondary->nwav; ++n) {
                secondary->dflux_cur.block(g, n, sz, n + 1) +=
                getIndex(secondary->L, n) * secondary->dF.block(5, n, sz, n + 1) -
                secondary->dflux_tot.block(g, n, sz, n + 1);
            }

        }
    }

    /**
    Compute the gradient of the secondary's flux during a secondary-secondary
    occultation.

    */
    template <class T>
    inline void System<T>::computeSecondaryOccultationGradient(const S& time_cur,
            Secondary<T>* secondary, Secondary<T>* occultor) {

        if (!allZero(secondary->L)) {

            // ** Pre-compute some stuff **

            // Radius normalization
            Scalar<T> rb = 1 / secondary->r;

            // Starting index for the secondary's derivs
            g = secondary->g0;

            // ** First, derivs with respect to the secondary's own props **
            // t, r, L, prot, a, porb, inc, ecc, w, Omega, lambda0, tref

            // dF / dt
            setRow(secondary->dflux_cur, 0, Row<T>(
                   getRow(secondary->dflux_cur, 0) -
                   getRow(secondary->dflux_tot, 0) +
                   cwiseProduct(secondary->L, Row<T>
                   (getRow(secondary->dF, 0) * secondary->angvelrot_deg -
                    getRow(secondary->dF, 0) * secondary->angvelrot_deg *
                        secondary->AD.delay.derivatives()(0) +
                    rb * getRow(secondary->dF, 1) *
                        (occultor->AD.x.derivatives()(0) - secondary->AD.x.derivatives()(0)) +
                    rb * getRow(secondary->dF, 2) *
                        (occultor->AD.y.derivatives()(0) - secondary->AD.y.derivatives()(0)))) *
                   units::DayToSeconds));

               // dF / dr
               setRow(secondary->dflux_cur, g, Row<T>(
                      getRow(secondary->dflux_cur, g) -
                      getRow(secondary->dflux_tot, g) +
                      rb * rb *
                      cwiseProduct(secondary->L, Row<T>
                      ((secondary->x_cur - occultor->x_cur) * getRow(secondary->dF, 1) +
                       (secondary->y_cur - occultor->y_cur) * getRow(secondary->dF, 2) -
                       occultor->r * getRow(secondary->dF, 3)))));
               g++;

               // dF / dL
               setRow(secondary->dflux_cur, g, Row<T>(
                      getRow(secondary->dflux_cur, g) -
                      getRow(secondary->dflux_tot, g) +
                      cwiseQuotient(secondary->flux_cur, secondary->L)));
               g++;

               // dF / dprot
               setRow(secondary->dflux_cur, g, Row<T>(
                      getRow(secondary->dflux_cur, g) -
                      getRow(secondary->dflux_tot, g) -
                      cwiseProduct(secondary->L, getRow(secondary->dF, 0)) *
                      (secondary->angvelrot_deg *
                      (time_cur - secondary->tref - secondary->delay) / secondary->prot +
                      secondary->theta0_deg / secondary->prot) * units::DayToSeconds));
               g++;

               // dF / da
               setRow(secondary->dflux_cur, g, Row<T>(
                      getRow(secondary->dflux_cur, g) -
                      getRow(secondary->dflux_tot, g) -
                      cwiseProduct(secondary->L, Row<T>(
                          getRow(secondary->dF, 0) * secondary->angvelrot_deg *
                               secondary->AD.delay.derivatives()(1) +
                          getRow(secondary->dF, 1) *
                               secondary->AD.x.derivatives()(1) * rb +
                          getRow(secondary->dF, 2) *
                               secondary->AD.y.derivatives()(1) * rb))));
               g++;

               // dF / porb
               setRow(secondary->dflux_cur, g, Row<T>(
                      getRow(secondary->dflux_cur, g) -
                      getRow(secondary->dflux_tot, g) -
                      units::DayToSeconds *
                      cwiseProduct(secondary->L, Row<T>(
                          getRow(secondary->dF, 0) * secondary->angvelrot_deg *
                               secondary->AD.delay.derivatives()(5) -
                          getRow(secondary->dF, 0) *
                               secondary->theta0_deg / secondary->porb +
                          getRow(secondary->dF, 1) *
                               secondary->AD.x.derivatives()(5) * rb +
                          getRow(secondary->dF, 2) *
                               secondary->AD.y.derivatives()(5) * rb))));
               g++;

               // dF / dinc
               // Tricky because the sky rotation transform depends on I
               // TODO: This can be sped up a ton
               if (secondary->y_deg > 0) {
                   Row<T> dFdinc;
                   resize(dFdinc, secondary->N, secondary->nwav);
                   for (int n = 0; n < secondary->nwav; ++n) {
                       Vector<Scalar<T>> dyskydinc(secondary->N);
                       VectorT<Scalar<T>> dFdysky =
                           secondary->skyMap.getGradient().
                               block(4, n, secondary->N, 1).transpose();
                       for (int l = 0; l < secondary->lmax + 1; ++l) {
                           dyskydinc.segment(l * l, 2 * l + 1) =
                               -secondary->W1.dRdtheta[l] *
                               secondary->W2.R[l] *
                               secondary->y.block(l * l, n, 2 * l + 1, 1);
                       }
                       setIndex(dFdinc, n, dFdysky.dot(dyskydinc));
                   }
                   setRow(secondary->dflux_cur, g, Row<T>(
                          getRow(secondary->dflux_cur, g) -
                          getRow(secondary->dflux_tot, g) +
                          pi<Scalar<T>>() / 180. *
                          cwiseProduct(secondary->L, Row<T>(
                          -getRow(secondary->dF, 0) *
                                  secondary->angvelrot_deg *
                                  secondary->AD.delay.derivatives()(8) +
                          dFdinc
                          - getRow(secondary->dF, 1) *
                               secondary->AD.x.derivatives()(8) * rb
                          - getRow(secondary->dF, 2) *
                               secondary->AD.y.derivatives()(8) * rb)))
                      );
               }
               g++;

               // dF / decc
               setRow(secondary->dflux_cur, g, Row<T>(
                      getRow(secondary->dflux_cur, g) -
                      getRow(secondary->dflux_tot, g) -
                      cwiseProduct(secondary->L, Row<T>(
                          getRow(secondary->dF, 0) * secondary->angvelrot_deg *
                               secondary->AD.delay.derivatives()(2) -
                          getRow(secondary->dF, 0) *
                               secondary->dtheta0_degde +
                          getRow(secondary->dF, 1) *
                               secondary->AD.x.derivatives()(2) * rb +
                          getRow(secondary->dF, 2) *
                               secondary->AD.y.derivatives()(2) * rb))));
               g++;

               // dF / dw; note that we must account for d / dM0(w)
               setRow(secondary->dflux_cur, g, Row<T>(
                      getRow(secondary->dflux_cur, g) -
                      getRow(secondary->dflux_tot, g) -
                      pi<Scalar<T>>() / 180.0 *
                      cwiseProduct(secondary->L, Row<T>(
                          getRow(secondary->dF, 0) * secondary->angvelrot_deg *
                               (secondary->AD.delay.derivatives()(6) -
                                secondary->AD.delay.derivatives()(3)) -
                          getRow(secondary->dF, 0) *
                               secondary->dtheta0_degdw +
                          getRow(secondary->dF, 1) *
                               (secondary->AD.x.derivatives()(6) -
                                secondary->AD.x.derivatives()(3)) * rb +
                          getRow(secondary->dF, 2) *
                               (secondary->AD.y.derivatives()(6) -
                                secondary->AD.y.derivatives()(3)) * rb))));
               g++;


               // dF / dOmega
               // Tricky because the sky rotation transform depends on Omega
               // TODO: This can be sped up a ton
               if (secondary->y_deg > 0) {
                   Row<T> dFdOmega;
                   resize(dFdOmega, secondary->N, secondary->nwav);
                   for (int n = 0; n < secondary->nwav; ++n) {
                       Vector<Scalar<T>> dyskydOmega(secondary->N);
                       VectorT<Scalar<T>> dFdysky =
                           secondary->skyMap.getGradient().
                               block(4, n, secondary->N, 1).transpose();
                       for (int l = 0; l < secondary->lmax + 1; ++l) {
                           dyskydOmega.segment(l * l, 2 * l + 1) =
                               secondary->W1.R[l] *
                               secondary->W2.dRdtheta[l] *
                               secondary->y.block(l * l, n, 2 * l + 1, 1);
                       }
                       setIndex(dFdOmega, n, dFdysky.dot(dyskydOmega));
                   }
                   setRow(secondary->dflux_cur, g, Row<T>(
                          getRow(secondary->dflux_cur, g) -
                          getRow(secondary->dflux_tot, g) +
                          pi<Scalar<T>>() / 180. *
                          cwiseProduct(secondary->L, Row<T>(
                          dFdOmega
                          - getRow(secondary->dF, 1) *
                               secondary->AD.x.derivatives()(7) * rb
                          - getRow(secondary->dF, 2) *
                               secondary->AD.y.derivatives()(7) * rb)))
                      );
               }
               g++;

               // dF / dlambda0
               setRow(secondary->dflux_cur, g, Row<T>(
                      getRow(secondary->dflux_cur, g) -
                      getRow(secondary->dflux_tot, g) +
                      pi<Scalar<T>>() / 180.0 *
                      cwiseProduct(secondary->L, Row<T>(
                          getRow(secondary->dF, 0) *
                               (180.0 / pi<Scalar<T>>() *
                               secondary->porb / secondary->prot -
                               secondary->angvelrot_deg *
                               secondary->AD.delay.derivatives()(3)) -
                          getRow(secondary->dF, 1) *
                               secondary->AD.x.derivatives()(3) * rb -
                          getRow(secondary->dF, 2) *
                               secondary->AD.y.derivatives()(3) * rb))));
               g++;

               // dF / dtref
               setRow(secondary->dflux_cur, g, Row<T>(
                      getRow(secondary->dflux_cur, g) -
                      getRow(secondary->dflux_tot, g) -
                      units::DayToSeconds *
                      cwiseProduct(secondary->L, Row<T>(
                          getRow(secondary->dF, 0) *
                               secondary->angvelrot_deg *
                               (1 + secondary->AD.delay.derivatives()(4)) +
                          getRow(secondary->dF, 1) *
                               secondary->AD.x.derivatives()(4) * rb +
                          getRow(secondary->dF, 2) *
                               secondary->AD.y.derivatives()(4) * rb))));
               g++;

               // dF / d{y} and dF / d{u}
               // Note that we skip the Y_{0,0} deriv
               int sz = secondary->dF.rows() - 5;
               for (int n = 0; n < secondary->nwav; ++n) {
                   secondary->dflux_cur.block(g, n, sz, n + 1) +=
                       getIndex(secondary->L, n) * secondary->dF.block(5, n, sz, n + 1) -
                       secondary->dflux_tot.block(g, n, sz, n + 1);
               }


               // ** Now the derivs with respect to the occultor's properties **
               // r, L, prot, a, porb, inc, ecc, w, Omega, lambda0, tref

               // Starting index for the occultor's derivs
               g = occultor->g0;

               // dF / dr
               setRow(secondary->dflux_cur, g, Row<T>(
                      getRow(secondary->dflux_cur, g) -
                      getRow(secondary->dflux_tot, g) +
                      rb *
                      cwiseProduct(secondary->L, getRow(secondary->dF, 3))));
               g += 3;                                                          // dF / dL and dF / dprot are zero

               // dF / da
               setRow(secondary->dflux_cur, g, Row<T>(
                      getRow(secondary->dflux_cur, g) -
                      getRow(secondary->dflux_tot, g) +
                      rb *
                      cwiseProduct(secondary->L, Row<T>(
                      getRow(secondary->dF, 1) * occultor->AD.x.derivatives()(1) +
                      getRow(secondary->dF, 2) * occultor->AD.y.derivatives()(1)))));
               g++;

               // dF / dporb
               setRow(secondary->dflux_cur, g, Row<T>(
                      getRow(secondary->dflux_cur, g) -
                      getRow(secondary->dflux_tot, g) +
                      rb *
                      cwiseProduct(secondary->L, Row<T>
                      (getRow(secondary->dF, 1) * occultor->AD.x.derivatives()(5) +
                       getRow(secondary->dF, 2) * occultor->AD.y.derivatives()(5))) *
                      units::DayToSeconds));
               g++;

               // dF / dinc
               setRow(secondary->dflux_cur, g, Row<T>(
                      getRow(secondary->dflux_cur, g) -
                      getRow(secondary->dflux_tot, g) +
                      rb *
                      cwiseProduct(secondary->L, Row<T>
                      (getRow(secondary->dF, 1) * occultor->AD.x.derivatives()(8) +
                       getRow(secondary->dF, 2) * occultor->AD.y.derivatives()(8))) *
                      pi<Scalar<T>>() / 180.0));
               g++;


               // dF / decc
               setRow(secondary->dflux_cur, g, Row<T>(
                      getRow(secondary->dflux_cur, g) -
                      getRow(secondary->dflux_tot, g) +
                      rb * cwiseProduct(secondary->L, Row<T>
                      (getRow(secondary->dF, 1) * occultor->AD.x.derivatives()(2) +
                       getRow(secondary->dF, 2) * occultor->AD.y.derivatives()(2)))));
               g++;

               // dF / dw
               setRow(secondary->dflux_cur, g, Row<T>(
                      getRow(secondary->dflux_cur, g) -
                      getRow(secondary->dflux_tot, g) +
                      rb * cwiseProduct(secondary->L, Row<T>
                      (getRow(secondary->dF, 1) * (occultor->AD.x.derivatives()(6) -
                                                 occultor->AD.x.derivatives()(3)) +
                       getRow(secondary->dF, 2) * (occultor->AD.y.derivatives()(6) -
                                                 occultor->AD.y.derivatives()(3)))) *
                      pi<Scalar<T>>() / 180.0));
               g++;

               // dF / dOmega
               setRow(secondary->dflux_cur, g, Row<T>(
                      getRow(secondary->dflux_cur, g) -
                      getRow(secondary->dflux_tot, g) +
                      rb * cwiseProduct(secondary->L, Row<T>
                      (getRow(secondary->dF, 1) * occultor->AD.x.derivatives()(7) +
                       getRow(secondary->dF, 2) * occultor->AD.y.derivatives()(7))) *
                      pi<Scalar<T>>() / 180.0));
               g++;

               // dF / dlambda0
               setRow(secondary->dflux_cur, g, Row<T>(
                      getRow(secondary->dflux_cur, g) -
                      getRow(secondary->dflux_tot, g) +
                      rb * cwiseProduct(secondary->L, Row<T>
                      (getRow(secondary->dF, 1) * occultor->AD.x.derivatives()(3) +
                       getRow(secondary->dF, 2) * occultor->AD.y.derivatives()(3))) *
                      pi<Scalar<T>>() / 180.0));
               g++;

               // dF / dtref
               setRow(secondary->dflux_cur, g, Row<T>(
                      getRow(secondary->dflux_cur, g) -
                      getRow(secondary->dflux_tot, g) +
                      rb * cwiseProduct(secondary->L, Row<T>
                      (getRow(secondary->dF, 1) * occultor->AD.x.derivatives()(4) +
                       getRow(secondary->dF, 2) * occultor->AD.y.derivatives()(4))) *
                      units::DayToSeconds));
               g++;


        }
    }

} // namespace kepler

#endif
