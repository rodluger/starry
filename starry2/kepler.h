/**
Keplerian star/planet/moon system class.

TODO: gradients!

*/

#ifndef _STARRY_ORBITAL_H_
#define _STARRY_ORBITAL_H_

#include <iostream>
#include <cmath>
#include <Eigen/Core>
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

}; // namespace units

namespace kepler {

    using namespace utils;
    using maps::Map;
    using rotation::Wigner;
    using std::abs;
    using std::isinf;

    // Forward declare our classes
    template <class T> class Body;
    template <class T> class Primary;
    template <class T> class Secondary;
    template <class T> class System;


    /* ---------------- */
    /*     FUNCTIONS    */
    /* ---------------- */

    /**
    Compute the eccentric anomaly. Adapted from
    https://github.com/lkreidberg/batman/blob/master/c_src/_rsky.c

    */
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
            throw errors::ConvergenceError("The Kepler solver "
                                           "did not converge.");
        }
        return E;
    }

    /**
    Manual override of the derivative of the eccentric anomaly

    */
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
            return Eigen::AutoDiffScalar<T>(E_value,
                                            M.derivatives() * norm1 +
                                            ecc.derivatives() * norm2);
        else if (M.derivatives().size())
            return Eigen::AutoDiffScalar<T>(E_value, M.derivatives() * norm1);
        else if (ecc.derivatives().size())
            return Eigen::AutoDiffScalar<T>(E_value, ecc.derivatives() * norm2);
        else
            return Eigen::AutoDiffScalar<T>(E_value, M.derivatives());
    }


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
            S r;                                                                /**< Body radius in units of primary radius */
            S L;                                                                /**< Body luminosity in units of primary luminosity */
            S prot;                                                             /**< Body rotation period in seconds */
            S tref;                                                             /**< Reference time in seconds */
            S theta0_deg;                                                       /**< Body initial rotation angle in degrees */
            S angvelrot_deg;                                                    /**< Body rotational angular velocity in degrees / second */
            S z0;                                                               /**< Reference point for retarded time calculation (the primary, assuming massless secondaries) */
            S delay;                                                            /**< The light travel time delay in seconds */
            Row<T> flux_cur;                                                    /**< Current flux visible from the body */
            Row<T> flux_tot;                                                    /**< Total flux from the body */

            // Private methods
            inline S theta_deg(const S& time);
            void computeTotal(const S& time);
            void occult(const S& time, const S& xo, const S& yo, const S& ro);

            //! Wrapper to get the flux from the map (overriden in Secondary)
            virtual inline Row<T> getFlux(const S& theta_deg, const S& xo,
                const S& yo, const S& ro, bool gradient) {
                return this->flux(theta_deg, xo, yo, ro, gradient);
            }

            //! Compute the initial rotation angle (overriden in Secondary)
            virtual void computeTheta0() {
                theta0_deg = 0;
            }

            //! Constructor
            explicit Body(int lmax=2, int nwav=1) :
                Map<T>(lmax, nwav),
                flux_cur(nwav),
                flux_tot(nwav) {

                // Initialize some stuff
                setZero(flux_cur);
                setZero(flux_tot);
                z0 = 0;
                delay = 0;
                computeTheta0();

                // Set the orbital variables to default values
                setRadius(1.0);
                setLuminosity(1.0);
                setRotPer(0.0);
                setRefTime(0.0);

            }

        public:

            // I/O
            void setRadius(const S& r_);
            virtual S getRadius() const;
            void setLuminosity(const S& L_);
            virtual S getLuminosity() const;
            void setRotPer(const S& prot_);
            S getRotPer() const;
            void setRefTime(const S& tref_);
            S getRefTime() const;

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
    inline void Body<T>::computeTotal(const Scalar<T>& time) {
        if (L != 0)
            flux_tot = L * getFlux(theta_deg(time), 0, 0, 0, false);
        else
            setZero(flux_tot);
        flux_cur = flux_tot;
    }

    //! Occult the body and update the current flux
    template <class T>
    inline void Body<T>::occult(const Scalar<T>& time, const Scalar<T>& xo,
                                const Scalar<T>& yo, const Scalar<T>& ro) {
        if ((L != 0) && (xo * xo + yo * yo < (1 + ro) * (1 + ro)))
            flux_cur += L * getFlux(theta_deg(time), xo, yo, ro, false)
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
    void Body<T>::setLuminosity(const Scalar<T>& L_) {
        if (L_ >= 0) L = L_;
        else throw errors::ValueError("Body's luminosity cannot be negative.");
    }

    //! Get the body's luminosity
    template <class T>
    Scalar<T> Body<T>::getLuminosity() const {
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

    //!


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
            S r_meters;                                                         /**< Radius of the body in meters */
            S c_light;                                                          /**< Speed of light in units of primary radius / s */

        public:

            Matrix<Scalar<T>> lightcurve;                                       /**< The body's full light curve */

            //! Constructor
            explicit Primary(int lmax=2, int nwav=1) :

                // Call the `Body` constructor
                Body<T>(lmax, nwav)
            {

                setRadiusInMeters(0.0);
                setRadius(1.0);
                setLuminosity(1.0);
                this->setRotPer(0.0);
                this->setRefTime(0.0);

            }

            // I/O
            void setRadius(const S& r_);
            void setLuminosity(const S& L_);
            void setRadiusInMeters(const S& r_);
            S getRadiusInMeters() const;
            const Matrix<Scalar<T>>& getLightcurve() const;

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

    //! Get the body's full light curve
    template <class T>
    const Matrix<Scalar<T>>& Primary<T>::getLightcurve() const {
        return lightcurve;
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
    void Primary<T>::setLuminosity(const Scalar<T>& L_) {
        if (L_ != 1.0)
            throw errors::NotImplementedError("The luminosity of the primary "
                                              "body is fixed at unity.");
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
            using Body<T>::G;
            using Body<T>::B;
            using Body<T>::lmax;
            using Body<T>::N;
            using Body<T>::nwav;
            using Body<T>::tref;
            using Body<T>::z0;
            using Body<T>::delay;
            /*
            using Body<T>::r;
            using Body<T>::L;
            using Body<T>::angvelrot_deg;
            using Body<T>::theta_deg;
            */

            // Computed values
            Matrix<Scalar<T>> lightcurve;                                       /**< The body's full light curve */
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
            S M;                                                                /**< Mean anomaly in radians */
            S E;                                                                /**< Eccentric anomaly in radians */
            S f;                                                                /**< True anomaly in radians */
            S rorb;                                                             /**< Instantaneous orbital radius in units of the primary radius */
            S cwf;                                                              /**< cos(w + f) */
            S swf;                                                              /**< sin(w + f) */
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

            // Private methods
            inline Row<T> getFlux(const S& theta_deg, const S& xo,
                const S& yo, const S& ro, bool gradient);
            void computeTheta0();
            inline void syncSkyMap();
            inline void orbitStep(const S& time);
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
            const Matrix<Scalar<T>>& getLightcurve() const;
            const Vector<Scalar<T>>& getXVector() const;
            const Vector<Scalar<T>>& getYVector() const;
            const Vector<Scalar<T>>& getZVector() const;

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
                W2(lmax, nwav, y, axis2)

            {

                // Set the orbital variables to default values
                this->setRadius(0.1);
                this->setLuminosity(0.0);
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

        // If there's any inclination or rotation of the orbital plane,
        // we need to rotate the sky map as well as the rotation axis
        if ((Omega != 0) || (sini < 1. - 2 * mach_eps<Scalar<T>>())) {

            // Let's store the rotation matrices: we'll need them to correctly
            // transform the derivatives of the map back to the user coordinates
            W1.compute(sini, cosi);
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
                                        bool gradient) {
        return skyMap.flux(theta_deg, xo, yo, ro, gradient);
    }

    /**
    Initial map rotation angle in degrees. The map is defined at the
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
        } else {
            Scalar<T> f_eclipse = 1.5 * pi<Scalar<T>>() - w;
            Scalar<T> E_eclipse = atan2(sqrt(1 - ecc2) * sin(f_eclipse),
                                        ecc + cos(f_eclipse));
            Scalar<T> M_eclipse = E_eclipse - ecc * sin(E_eclipse);
            theta0_deg = -(porb / prot) * (M_eclipse - M0);
            theta0_deg *= 180.0 / pi<Scalar<T>>();
        }
    }

    /**
    Compute the instantaneous x, y, and z positions of the
    body with a simple Keplerian solver.

    */
    template <class T>
    void Secondary<T>::orbitStep(const Scalar<T>& time) {

        // Mean anomaly
        M = mod2pi(M0 + angvelorb * (time - tref));

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

        // See Murray and Dermott p. 51, except x and y
        // have the opposite sign here.
        // This ensures the orbits are prograde!
        cwf = cos(w + f);
        swf = sin(w + f);
        x_cur = -rorb * (cosO * cwf - sinOcosi * swf);
        y_cur = -rorb * (sinO * cwf + cosOcosi * swf);
        z_cur = rorb * swf * sini;

        // Compute the light travel time delay
        if (!isinf(*c_light)) applyLightDelay(time);

    }

    /**
    Compute the light travel time delay and apply
    it to the current (x, y, z) position of the body.

    */
    template <class T>
    void Secondary<T>::applyLightDelay(const Scalar<T>& time) {

        // Component of the velocity out of the sky
        // Obtained by differentiating the expressions above
        Scalar<T> vz = vamp * sini * (ecw + cwf);

        // Component of the acceleration out of the sky
        Scalar<T> az = -angvelorb * angvelorb * a * a * a /
                       (rorb * rorb * rorb) * z_cur;

        // Compute the time delay at the **retarded** position, accounting
        // for the instantaneous velocity and acceleration of the body.
        // This is slightly better than doing
        //
        //          dt_ = (z0 - z_) / c_light
        //
        // which is actually the time delay at the **current** position.
        // But the photons left the planet from the **retarded** position,
        // so if the planet has motion in the `z` direction the two values
        // will be slightly different. In practice it doesn't really matter
        // that much, though. See the derivation at
        // https://github.com/rodluger/starry/issues/66
        if (abs(az) < 1e-10)
            delay = (z0 - z_cur) / (*c_light + vz);
        else
            delay = (*c_light / az) * ((1 + vz / *c_light)
                    - sqrt((1 + vz / *c_light) * (1 + vz / *c_light)
                    - 2 * az * (z0 - z_cur) / (*c_light * *c_light)));

        // Re-compute Kepler's equation, this time
        // solving for the **retarded** position
        M = mod2pi(M0 + angvelorb * (time - delay - tref));
        if (ecc > 0) {
            E = EccentricAnomaly(M, ecc);
            f = (2. * atan2(sqrtonepluse * sin(E / 2.),
                            sqrtoneminuse * cos(E / 2.)));
            rorb = a * (1. - ecc2) / (1. + ecc * cos(f));
        } else {
            f = M;
            rorb = a;
        }
        cwf = cos(w + f);
        swf = sin(w + f);
        x_cur = -rorb * (cosO * cwf - sinOcosi * swf);
        y_cur = -rorb * (sinO * cwf + cosOcosi * swf);
        z_cur = rorb * swf * sini;

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
        return porb;
    }

    //! Set the inclination
    template <class T>
    void Secondary<T>::setInc(const Scalar<T>& inc_) {
        if ((inc_ >= 0) && (inc_ < 180.0)) inc = inc_ * pi<Scalar<T>>() / 180.0;
        else throw errors::ValueError("Inclination must be "
                                      "in the range [0, 180).");
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

    //! Get the body's full light curve
    template <class T>
    const Matrix<Scalar<T>>& Secondary<T>::getLightcurve() const {
        return lightcurve;
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

    /* ----------------- */
    /*       SYSTEM      */
    /* ----------------- */

    //! Keplerian system class
    template <class T>
    class System {

        protected:

            using S = Scalar<T>;                                                /**< Shorthand for the scalar type (double, Multi, ...) */
            Matrix<Scalar<T>> lightcurve;                                       /**< The body's full light curve */
            Scalar<T> exptime;                                                  /**< Exposure time in days */
            Scalar<T> exptol;
            int expmaxdepth;
            size_t t;

            // Protected methods
            inline void step(const S& time_cur);
            T step(const S& time_cur, bool store_xyz);
            T integrate(const T& f1, const T& f2,
                        const S& t1, const S& t2, int depth);
            inline T integrate (const S& time_cur);

        public:

            Primary<T>* primary;
            std::vector<Secondary<T>*> secondaries;

            //! Constructor: single secondary
            explicit System(Primary<T>* primary,
                            Secondary<T>* secondary) :
                            primary(primary)
            {
                secondaries.push_back(secondary);
                setExposureTime(0.0);
                setExposureTol(sqrt(mach_eps<Scalar<T>>()));
                setExposureMaxDepth(4);
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
            }

            // Public methods
            void compute(const Vector<S>& time);
            const Matrix<S>& getLightcurve() const;
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
        return "<Keplerian " + std::to_string(secondaries.size() + 1) +
               "-body system>";
    }

    //! Return the full system light curve
    template <class T>
    const Matrix<Scalar<T>>& System<T>::getLightcurve() const {
        return lightcurve;
    }

    /**
    Recursive exposure time integration function (single iteration)

    */
    template <class T>
    T System<T>::integrate(const T& f1,
                           const T& f2,
                           const Scalar<T>& t1,
                           const Scalar<T>& t2, int depth) {
        Scalar<T> tmid = 0.5 * (t1 + t2);
        // If this is the first time we're recursing (depth == 0),
        // store the xyz position of the bodies in the output vectors
        T fmid = step(tmid, depth == 0),
          fapprox = 0.5 * (f1 + f2),
          d = fmid - fapprox;

        // DEBUG
        //std::cout << depth << ": " << d.transpose() << std::endl;

        if (depth < expmaxdepth) {
            for (int i = 0; i < d.rows(); ++i) {
                for (int n = 0; n < primary->nwav; ++n) {
                    if (abs(d(i, n)) > exptol) {
                        T a = integrate(f1, fmid, t1, tmid, depth + 1),
                          b = integrate(fmid, f2, tmid, t2, depth + 1);
                        return a + b;
                    }
                }
            }
        }
        return fapprox * (t2 - t1);
    };

    /**
    Recursive exposure time integration function

    */
    template <class T>
    inline T System<T>::integrate(const Scalar<T>& time_cur) {
        Scalar<T> dt = 0.5 * exptime,
                  t1 = time_cur - dt,
                  t2 = time_cur + dt;

        return integrate(step(t1, false), step(t2, false), t1, t2, 0) /
                   (t2 - t1);

    };

    /**
    Take a single orbital + photometric step.

    */
    template <class T>
    inline void System<T>::step(const Scalar<T>& time_cur) {

        Scalar<T> xo, yo, ro;
        size_t NS = secondaries.size();
        size_t o, p;

        // Compute fluxes and take an orbital step
        primary->computeTotal(time_cur);
        for (auto secondary : secondaries) {
            secondary->orbitStep(time_cur);
            secondary->computeTotal(time_cur);
        }

        // Compute occultations involving the primary
        for (auto secondary : secondaries) {
            if (secondary->z_cur > 0) {
                primary->occult(time_cur, secondary->x_cur,
                                secondary->y_cur, secondary->r);
            } else {
                ro = 1. / secondary->r;
                secondary->occult(time_cur, -ro * secondary->x_cur,
                                  -ro * secondary->y_cur, ro);
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
                ro = 1. / secondaries[p]->r;
                xo = ro * (secondaries[o]->x_cur - secondaries[p]->x_cur);
                yo = ro * (secondaries[o]->y_cur - secondaries[p]->y_cur);
                ro = ro * secondaries[o]->r;
                secondaries[p]->occult(time_cur, xo, yo, ro);
            }
        }

    }

    /**
    Take a single orbital + photometric step
    (exposure time integration overload).

    */
    template <class T>
    inline T System<T>::step(const Scalar<T>& time, bool store_xyz) {

        // Take the step
        step(time);

        // Get the current values of the flux for each body
        size_t NS = secondaries.size();
        T fluxes(NS + 1, primary->nwav);
        setRow(fluxes, 0, primary->flux_cur);
        for (size_t n = 0; n < NS; ++n) {
            if (store_xyz) {
                // If this is the midpoint of the integration,
                // store the cartesian position of the bodies
                secondaries[n]->xvec(t) = secondaries[n]->x_cur;
                secondaries[n]->yvec(t) = secondaries[n]->y_cur;
                secondaries[n]->zvec(t) = secondaries[n]->z_cur;
            }
            setRow(fluxes, n + 1, secondaries[n]->flux_cur);
        }

        // Return the flux from each of the bodies
        return fluxes;

    }

    /**
    Compute the full system light curve. Special case w/ no exposure
    time integration, optimized for speed.

    */
    template <class T>
    void System<T>::compute(const Vector<Scalar<T>>& time_) {

        size_t NT = time_.size();
        Vector<Scalar<T>> time = time_ * units::DayToSeconds;

        // Allocate arrays
        // Sync the orbital and sky maps
        // Sync the speed of light across all secondaries
        lightcurve.resize(NT, primary->nwav);
        primary->lightcurve.resize(NT, primary->nwav);
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
        }

        // Loop through the timeseries
        for (t = 0; t < NT; ++t){

            // Take an orbital step and compute the fluxes
            if (exptime == 0) {

                // Advance the system
                step(time(t));

                // Update the light curves and orbital positions
                for (int n = 0; n < primary->nwav; ++n) {
                    primary->lightcurve(t, n) = getColumn(primary->flux_cur, n);
                    lightcurve(t, n) = getColumn(primary->flux_cur, n);
                }
                for (auto secondary : secondaries) {
                    secondary->xvec(t) = secondary->x_cur;
                    secondary->yvec(t) = secondary->y_cur;
                    secondary->zvec(t) = secondary->z_cur;
                    for (int n = 0; n < primary->nwav; ++n) {
                        secondary->lightcurve(t, n) =
                            getColumn(secondary->flux_cur, n);
                        lightcurve(t, n) += getColumn(secondary->flux_cur, n);
                    }
                }

            } else {

                // Integrate over the exposure time
                T fluxes = integrate(time(t));

                // Update the light curves
                for (int n = 0; n < primary->nwav; ++n) {
                    primary->lightcurve(t, n) = fluxes(0, n);
                    lightcurve(t, n) = fluxes(0, n);
                }
                for (size_t i = 0; i < secondaries.size(); ++i) {
                    for (int n = 0; n < primary->nwav; ++n) {
                        secondaries[i]->lightcurve(t, n) = fluxes(i + 1, n);
                        lightcurve(t, n) += fluxes(i + 1, n);
                    }
                }

            }

        }

    }

}; // namespace kepler

#endif
