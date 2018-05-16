/**
Docstrings for the Python functions.

*/

#ifndef _STARRY_DOCS_H_
#define _STARRY_DOCS_H_

#include <stdlib.h>
using namespace std;

namespace docstrings {

    const char * starry =
    R"pbdoc(
        starry
        ------

        .. contents::
            :local:

        Introduction
        ============

        This page documents the :py:mod:`starry` API, which is coded
        in C++ with a :py:mod:`pybind11` Python interface. The API consists
        of a :py:class:`Map` class, which houses all of the surface map photometry
        stuff, and the :py:class:`Star`, :py:class:`Planet`, and :py:class:`System`
        classes, which facilitate the generation of light curves for actual
        stellar and planetary systems. There are two broad ways in which users can access
        the core :py:mod:`starry` functionality:

            - Users can instantiate a :py:class:`Map` class to compute phase curves
              and occultation light curves by directly specifying the rotational state
              of the object and (optionally) the position and size of an occultor.
              Users can also instantiate a :py:class:`LimbDarkenedMap` class for
              radially-symmetric stellar surfaces. Both cases
              may be particularly useful for users who wish to integrate :py:mod:`starry`
              with their own dynamical code or for users wishing to compute simple light
              curves without any orbital solutions.

            - Users can instantiate a :py:class:`Star` and one or more :py:class:`Planet`
              objects and feed them into a :py:class:`System` instance for integration
              with the Keplerian solver. All :py:class:`Star` and :py:class:`Planet`
              instances have a :py:obj:`map <>` attribute that allows users to customize
              the surface map prior to computing the system light curve.

        At present, :py:mod:`starry` uses a simple Keplerian solver to compute orbits, so
        the second approach listed above is limited to systems with low mass planets that
        do not exhibit transit timing variations. The next version will include integration
        with an N-body solver, so stay tuned!


        The Map classes
        ===============
        .. autoclass:: Map(lmax=2)
        .. autoclass:: LimbDarkenedMap(lmax=2)


        The orbital classes
        ===================
        .. autoclass:: Star()
        .. autoclass:: Planet(lmax=2, r=0.1, L=0, axis=(0, 1, 0), prot=0, theta0=0, a=50, porb=1, inc=90, ecc=0, w=90, Omega=0, lambda0=90, tref=0)
        .. autoclass:: System(bodies, kepler_tol=1.0e-7, kepler_max_iter=100)

    )pbdoc";

    const char * NotImplemented =
    R"pbdoc(
        Method or attribute not implemented for this class.
    )pbdoc";

    const char * mp_digits =
    R"pbdoc(
        Number of digits used to perform multi-precision calculations.
        Double precision roughly corresponds to 16, and quadrupole
        precision (default) roughly corresponds 32.
        This is a compile-time constant. If you wish to change it, you'll
        have to re-compile :py:obj:`starry` by executing

        .. code-block:: bash

            STARRY_MP_DIGITS=XX pip install --force-reinstall --ignore-installed --no-binary :all: starry
    )pbdoc";

    namespace Map {

        const char * Map =
        R"pbdoc(
                Instantiate a :py:mod:`starry` surface map. Maps instantiated in this fashion
                are *orthonormalized*, so the total integrated luminosity of the map is
                $2\sqrt{\pi} Y_{0,0}$.

                Args:
                    lmax (int): Largest spherical harmonic degree in the surface map. Default 2.

                .. autoattribute:: optimize
                .. automethod:: evaluate(axis=(0, 1, 0), theta=0, x=0, y=0)
                .. automethod:: rotate(axis=(0, 1, 0), theta=0)
                .. automethod:: flux_numerical(axis=(0, 1, 0), theta=0, xo=0, yo=0, ro=0, tol=1.e-4)
                .. automethod:: flux_mp(axis=(0, 1, 0), theta=0, xo=0, yo=0, ro=0)
                .. automethod:: flux(axis=(0, 1, 0), theta=0, xo=0, yo=0, ro=0)
                .. automethod:: get_coeff(l, m)
                .. automethod:: set_coeff(l, m, coeff)
                .. automethod:: reset()
                .. autoattribute:: lmax
                .. autoattribute:: y
                .. autoattribute:: p
                .. autoattribute:: g
                .. autoattribute:: s
                .. autoattribute:: s_mp
                .. autoattribute:: r
                .. automethod:: minimum()
                .. automethod:: add_gaussian()
                .. automethod:: load_image(image)
                .. automethod:: load_healpix(image)
                .. automethod:: show(cmap='plasma', res=300)
                .. automethod:: animate(axis=(0, 1, 0), cmap='plasma', res=150, frames=50)
                .. autoattribute:: mp_digits

            )pbdoc";

        const char * get_coeff =
        R"pbdoc(
            Return the (:py:obj:`l`, :py:obj:`m`) coefficient of the map.

            .. note:: Users can also retrieve a coefficient by accessing the \
                      [:py:obj:`l`, :py:obj:`m`] index of the map as if it \
                      were a 2D array.

            Args:
                l (int): The spherical harmonic degree, ranging from 0 to :py:attr:`lmax`.
                m (int): The spherical harmonic order, ranging from -:py:obj:`l` to :py:attr:`l`.
        )pbdoc";

        const char * set_coeff =
        R"pbdoc(
            Set the (:py:obj:`l`, :py:obj:`m`) coefficient of the map.

            .. note:: Users can also set a coefficient by setting the \
                      [:py:obj:`l`, :py:obj:`m`] index of the map as if it \
                      were a 2D array.

            Args:
                l (int): The spherical harmonic degree, ranging from 0 to :py:attr:`lmax`.
                m (int): The spherical harmonic order, ranging from -:py:obj:`l` to :py:attr:`l`.
                coeff (float): The value of the coefficient.
        )pbdoc";

        const char * reset =
        R"pbdoc(
            Set all of the map coefficients to zero.
        )pbdoc";

        const char * lmax =
        R"pbdoc(
            The highest spherical harmonic order of the map. *Read-only.*
        )pbdoc";

        const char * y =
        R"pbdoc(
            The spherical harmonic map vector. *Read-only.*
        )pbdoc";

        const char * p =
        R"pbdoc(
            The polynomial map vector. *Read-only.*
        )pbdoc";

        const char * g =
        R"pbdoc(
            The Green's polynomial map vector. *Read-only.*
        )pbdoc";

        const char * s =
        R"pbdoc(
            The current solution vector `s`. *Read-only.*
        )pbdoc";

        const char * r =
        R"pbdoc(
            The current solution vector `r`. *Read-only.*
        )pbdoc";

        const char * s_mp =
        R"pbdoc(
            The current multi-precision solution vector `s`. Only available after :py:meth:`flux_mp` has been called. *Read-only.*
        )pbdoc";

        const char * optimize =
        R"pbdoc(
            Set to :py:obj:`False` to disable Taylor expansions of the primitive integrals when \
            computing occultation light curves. This is in general not something you should do! \
            Default :py:obj:`True`.
        )pbdoc";

        const char * evaluate =
        R"pbdoc(
            Return the specific intensity at a point (`x`, `y`) on the map.

            Users may optionally provide a rotation state. Note that this does
            not rotate the base map.

            Args:
                axis (ndarray): *Normalized* unit vector specifying the body's axis of rotation. Default :math:`\hat{y} = (0, 1, 0)`.
                theta (float or ndarray): Angle of rotation in radians. Default 0.
                x (float or ndarray): Position scalar, vector, or matrix.
                y (float or ndarray): Position scalar, vector, or matrix.

            Returns:
                The specific intensity at (`x`, `y`).
        )pbdoc";

        const char * flux =
        R"pbdoc(
            Return the total flux received by the observer.

            Computes the total flux received by the observer from the
            map during or outside of an occultation.

            Args:
                axis (ndarray): *Normalized* unit vector specifying the body's axis of rotation. Default :math:`\hat{y} = (0, 1, 0)`.
                theta (float or ndarray): Angle of rotation. Default 0.
                xo (float or ndarray): The `x` position of the occultor (if any). Default 0.
                yo (float or ndarray): The `y` position of the occultor (if any). Default 0.
                ro (float): The radius of the occultor in units of this body's radius. Default 0 (no occultation).

            Returns:
                The flux received by the observer (a scalar or a vector).
        )pbdoc";

        const char * flux_mp =
        R"pbdoc(
            Return the total flux received by the observer, computed using multi-precision.

            Computes the total flux received by the observer from the
            map during or outside of an occultation. By default, this method
            performs all occultation calculations using 128-bit (quadruple) floating point
            precision, corresponding to 32 significant digits. Users can increase this to any
            number of digits (RAM permitting) by setting the :py:obj:`STARRY_MP_DIGITS=XX` flag
            at compile time. Note, importantly, that run times are **much** slower for multi-precision
            calculations.

            Args:
                axis (ndarray): *Normalized* unit vector specifying the body's axis of rotation. Default :math:`\hat{y} = (0, 1, 0)`.
                theta (float or ndarray): Angle of rotation. Default 0.
                xo (float or ndarray): The `x` position of the occultor (if any). Default 0.
                yo (float or ndarray): The `y` position of the occultor (if any). Default 0.
                ro (float): The radius of the occultor in units of this body's radius. Default 0 (no occultation).

            Returns:
                The flux received by the observer (a scalar or a vector).
        )pbdoc";

        const char * flux_numerical =
        R"pbdoc(
            Return the total flux received by the observer, computed numerically.

            Computes the total flux received by the observer from the
            map during or outside of an occultation. The flux is computed
            numerically using an adaptive radial mesh.

            Args:
                axis (ndarray): *Normalized* unit vector specifying the body's axis of rotation. Default :math:`\hat{y} = (0, 1, 0)`.
                theta (float or ndarray): Angle of rotation. Default 0.
                xo (float or ndarray): The `x` position of the occultor (if any). Default 0.
                yo (float or ndarray): The `y` position of the occultor (if any). Default 0.
                ro (float): The radius of the occultor in units of this body's radius. Default 0 (no occultation).
                tol (float): Tolerance of the numerical solver. Default `1.e-4`

            Returns:
                The flux received by the observer (a scalar or a vector).
            )pbdoc";

        const char * rotate =
        R"pbdoc(
            Rotate the base map an angle :py:obj:`theta` about :py:obj:`axis`.

            This performs a permanent rotation to the base map. Subsequent
            rotations and calculations will be performed relative to this
            rotational state.

            Args:
                axis (ndarray): *Normalized* unit vector specifying the body's axis of rotation. Default :math:`\hat{y} = (0, 1, 0)`.
                theta (float or ndarray): Angle of rotation in radians. Default 0.

        )pbdoc";

        const char * minimum =
        R"pbdoc(
            Find the global minimum of the map.

            This routine wraps :py:class:`scipy.optimize.minimize` to find
            the global minimum of the surface map. This is useful for ensuring
            that the surface map is nonnegative everywhere.

            .. note:: Because this routine wraps a Python wrapper of a C function \
                      to perform a non-linear optimization in three dimensions, it is \
                      **slow** and should probably not be used repeatedly when fitting \
                      a map to data!
        )pbdoc";

        const char * add_gaussian =
        R"pbdoc(
            Add the spherical harmonic expansion of a gaussian to the current map.

            This routine adds a gaussian-like feature to the surface map by computing
            the spherical harmonic expansion of a 3D gaussian constrained to the surface
            of the sphere. This is useful for, say, modeling star spots or other discrete,
            localized features on a body's surface.

            .. note:: Because this routine wraps a Python function, \
                      it is **slow** and should probably not be used repeatedly when fitting \
                      a map to data!

            Args:
                sigma (float): The standard deviation of the gaussian. Default 0.1
                amp (float): The amplitude. Default 1.0, resulting in a gaussian whose \
                             integral over the sphere is unity.
                lat (float): The latitude of the center of the gaussian in radians. Default 0.
                lon (float): The longitude of the center of the gaussian in radians. Default 0.
        )pbdoc";

        const char * load_image =
        R"pbdoc(
            Load an image from file.

            This routine loads an image file, computes its spherical harmonic
            expansion up to degree :py:attr:`lmax`, and sets the map vector.

            Args:
                image (str): The full path to the image file.

            .. todo:: The map is currently unnormalized; the max/min will depend \
                      on the colorscale of the input image. This will be fixed \
                      soon.

        )pbdoc";

        const char * load_healpix =
        R"pbdoc(
            Load a healpix image array.

            This routine loads a :py:obj:`healpix` array, computes its
            spherical harmonic
            expansion up to degree :py:attr:`lmax`, and sets the map vector.

            Args:
                image (ndarray): The ring-ordered :py:obj:`healpix` array.

            .. todo:: This routine has not yet been tested!
        )pbdoc";

        const char * show =
        R"pbdoc(
            Convenience routine to quickly display the body's surface map.

            Args:
                cmap (str): The :py:mod:`matplotlib` colormap name. Default `plasma`.
                res (int): The resolution of the map in pixels on a side. Default 300.
        )pbdoc";

        const char * animate =
        R"pbdoc(
            Convenience routine to animate the body's surface map as it rotates.

            Args:
                axis (ndarray): *Normalized* unit vector specifying the axis of rotation. Default :math:`\hat{y} = (0, 1, 0)`.
                cmap (str): The :py:mod:`matplotlib` colormap name. Default `plasma`.
                res (int): The resolution of the map in pixels on a side. Default 150.
                frames (int): The number of frames in the animation. Default 50.
        )pbdoc";

    } // namespace Map

    namespace LimbDarkenedMap {

        const char * LimbDarkenedMap =
        R"pbdoc(
                Instantiate a :py:mod:`starry` limb-darkened surface map.

                This differs from the base :py:class:`Map` class in that maps
                instantiated this way are radially symmetric: only the radial (:py:obj:`m = 0`)
                coefficients of the map are available. Users edit the map by directly
                specifying the polynomial limb darkening coefficients :py:obj:`u`, starting
                with $u_1$ (linear limb darkening). The coefficient $u_0$ is fixed to enforce
                the correct normalization.

                .. warning:: Unlike :py:class:`Map`, maps instantiated this \
                             way are normalized so that the integral of the specific intensity over the \
                             visible disk is unity. This is convenient for using this map to model \
                             stars: the unocculted flux from the star is equal to one, regardless of the limb-darkening \
                             coefficients!

                Args:
                    lmax (int): Largest spherical harmonic degree in the surface map. Default 2.

                .. autoattribute:: optimize
                .. automethod:: evaluate(x=0, y=0)
                .. automethod:: flux_numerical(xo=0, yo=0, ro=0, tol=1.e-4)
                .. automethod:: flux_mp(xo=0, yo=0, ro=0)
                .. automethod:: flux(xo=0, yo=0, ro=0)
                .. automethod:: get_coeff(l)
                .. automethod:: set_coeff(l, coeff)
                .. automethod:: reset()
                .. autoattribute:: lmax
                .. autoattribute:: y
                .. autoattribute:: p
                .. autoattribute:: g
                .. autoattribute:: u
                .. autoattribute:: s
                .. autoattribute:: s_mp
                .. automethod:: show(cmap='plasma', res=300)
                .. autoattribute:: mp_digits

        )pbdoc";

        const char * get_coeff =
        R"pbdoc(
            Return the limb darkening coefficient of order :py:obj:`l`.

            .. note:: Users can also retrieve a coefficient by accessing the \
                      [:py:obj:`l`] index of the map as if it were an array.

            Args:
                l (int): The limb darkening order (> 0).
        )pbdoc";

        const char * set_coeff =
        R"pbdoc(
            Set the limb darkening coefficient of order :py:obj:`l`.

            .. note:: Users can also set a coefficient by setting the \
                      [:py:obj:`l`] index of the map as if it \
                      were an array.

            Args:
                l (int): The limb darkening order (> 0).
                coeff (float): The value of the coefficient.
        )pbdoc";

        const char * reset =
        R"pbdoc(
            Set all of the map coefficients to zero.
        )pbdoc";

        const char * lmax =
        R"pbdoc(
            The highest spherical harmonic order of the map. *Read-only.*
        )pbdoc";

        const char * y =
        R"pbdoc(
            The spherical harmonic map vector. *Read-only.*
        )pbdoc";

        const char * p =
        R"pbdoc(
            The polynomial map vector. *Read-only.*
        )pbdoc";

        const char * g =
        R"pbdoc(
            The Green's polynomial map vector. *Read-only.*
        )pbdoc";

        const char * s =
        R"pbdoc(
            The current solution vector `s`. *Read-only.*
        )pbdoc";

        const char * u =
        R"pbdoc(
            The limb darkening coefficient vector. *Read-only.*
        )pbdoc";

        const char * s_mp =
        R"pbdoc(
            The current multi-precision solution vector `s`. Only available after :py:meth:`flux_mp` has been called. *Read-only.*
        )pbdoc";

        const char * optimize =
        R"pbdoc(
            Set to :py:obj:`False` to disable Taylor expansions of the primitive integrals when \
            computing occultation light curves. This is in general not something you should do! \
            Default :py:obj:`True`.
        )pbdoc";

        const char * evaluate =
        R"pbdoc(
            Return the specific intensity at a point (`x`, `y`) on the map.

            Args:
                x (float or ndarray): Position scalar, vector, or matrix.
                y (float or ndarray): Position scalar, vector, or matrix.

            Returns:
                The specific intensity at (`x`, `y`).
        )pbdoc";

        const char * flux =
        R"pbdoc(
            Return the total flux received by the observer.

            Computes the total flux received by the observer from the
            map during or outside of an occultation.

            Args:
                xo (float or ndarray): The `x` position of the occultor (if any). Default 0.
                yo (float or ndarray): The `y` position of the occultor (if any). Default 0.
                ro (float): The radius of the occultor in units of this body's radius. Default 0 (no occultation).

            Returns:
                The flux received by the observer (a scalar or a vector).
        )pbdoc";

        const char * flux_mp =
        R"pbdoc(
            Return the total flux received by the observer, computed using multi-precision.

            Computes the total flux received by the observer from the
            map during or outside of an occultation. By default, this method
            performs all occultation calculations using 128-bit (quadruple) floating point
            precision, corresponding to 32 significant digits. Users can increase this to any
            number of digits (RAM permitting) by setting the :py:obj:`STARRY_MP_DIGITS=XX` flag
            at compile time. Note, importantly, that run times are **much** slower for multi-precision
            calculations.

            Args:
                xo (float or ndarray): The `x` position of the occultor (if any). Default 0.
                yo (float or ndarray): The `y` position of the occultor (if any). Default 0.
                ro (float): The radius of the occultor in units of this body's radius. Default 0 (no occultation).

            Returns:
                The flux received by the observer (a scalar or a vector).
        )pbdoc";

        const char * flux_numerical =
        R"pbdoc(
            Return the total flux received by the observer, computed numerically.

            Computes the total flux received by the observer from the
            map during or outside of an occultation. The flux is computed
            numerically using an adaptive radial mesh.

            Args:
                xo (float or ndarray): The `x` position of the occultor (if any). Default 0.
                yo (float or ndarray): The `y` position of the occultor (if any). Default 0.
                ro (float): The radius of the occultor in units of this body's radius. Default 0 (no occultation).
                tol (float): Tolerance of the numerical solver. Default `1.e-4`

            Returns:
                The flux received by the observer (a scalar or a vector).
            )pbdoc";

        const char * show =
        R"pbdoc(
            Convenience routine to quickly display the body's surface map.

            Args:
                cmap (str): The :py:mod:`matplotlib` colormap name. Default `plasma`.
                res (int): The resolution of the map in pixels on a side. Default 300.
        )pbdoc";

    } // namespace LimbDarkenedMap

    namespace System {

        const char * System =
        R"pbdoc(
                Instantiate an orbital system.

                Args:
                    bodies (list): List of bodies in the system, with the primary (usually the star) listed first.
                    kepler_tol (float): Kepler solver tolerance. Default `1e-7`.
                    kepler_max_iter (int): Maximum number of iterations in the Kepler solver. Default `100`.
                    exposure_time (float): The exposure time of the observations in days. If nonzero, the flux will \
                                           be integrated over this exposure time. Default `0`.
                    exposure_tol (float): Tolerance of the recursive method for integrating the flux over the exposure time. Default `1e-8`.
                    exposure_maxdepth (int): Maximum recursion depth for the exposure calculation. Default `4`.

                .. automethod:: compute(time)
                .. autoattribute:: flux
                .. autoattribuet:: exposure_time
                .. autoattribuet:: exposure_tol
                .. autoattribuet:: exposure_max_depth
                .. autoattribuet:: kepler_tol
                .. autoattribuet:: kepler_max_iter
        )pbdoc";

        const char * compute =
        R"pbdoc(
            Compute the system light curve analytically.

            Compute the full system light curve at the times
            given by the :py:obj:`time <>` array and store the result
            in :py:attr:`flux`. The light curve for each body in the
            system is stored in the body's :py:attr:`flux` attribute.

            Args:
                time (ndarray): Time array, measured in days.
        )pbdoc";

        const char * flux =
        R"pbdoc(
            The computed system light curve. Must run :py:meth:`compute` first.
        )pbdoc";

        const char * exposure_time =
        R"pbdoc(
            The exposure time of the observations in days. If nonzero, the flux will
            be integrated over this exposure time. Default `0`.
        )pbdoc";

        const char * exposure_tol =
        R"pbdoc(
            Tolerance of the recursive method for integrating the flux over the exposure time. Default `1e-8`.
        )pbdoc";

        const char * exposure_max_depth =
        R"pbdoc(
            Maximum recursion depth for the exposure calculation. Default `4`.
        )pbdoc";

        const char * kepler_max_iter =
        R"pbdoc(
            Maximum number of iterations in the Kepler solver. Default `100`.
        )pbdoc";

        const char * kepler_tol =
        R"pbdoc(
            Kepler solver tolerance. Default `1e-7`.
        )pbdoc";

    } // namespace System

    namespace Body {

        const char * map =
        R"pbdoc(
            The body's surface map.
        )pbdoc";

        const char * flux =
        R"pbdoc(
            The body's computed light curve.
        )pbdoc";

        const char * x =
        R"pbdoc(
            The `x` position of the body in AU.
        )pbdoc";

        const char * y =
        R"pbdoc(
            The `y` position of the body in AU.
        )pbdoc";

        const char * z =
        R"pbdoc(
            The `z` position of the body in AU.
        )pbdoc";

        const char * r =
        R"pbdoc(
            Body radius in units of stellar radius.
        )pbdoc";

        const char * L =
        R"pbdoc(
            Body luminosity in units of stellar luminosity.
        )pbdoc";

        const char * axis =
        R"pbdoc(
            *Normalized* unit vector specifying the body's axis of rotation.
        )pbdoc";

        const char * prot =
        R"pbdoc(
            Rotation period in days.
        )pbdoc";

        const char * theta0 =
        R"pbdoc(
            Rotation phase at time :py:obj:`tref` in degrees.
        )pbdoc";

        const char * a =
        R"pbdoc(
            Body semi-major axis in units of stellar radius.
        )pbdoc";

        const char * porb =
        R"pbdoc(
            Orbital period in days.
        )pbdoc";

        const char * inc =
        R"pbdoc(
            Orbital inclination in degrees.
        )pbdoc";

        const char * ecc =
        R"pbdoc(
            Orbital eccentricity.
        )pbdoc";

        const char * w =
        R"pbdoc(
            Longitude of pericenter in degrees.
        )pbdoc";

        const char * Omega =
        R"pbdoc(
            Longitude of ascending node in degrees.
        )pbdoc";

        const char * lambda0 =
        R"pbdoc(
            Mean longitude at time :py:obj:`tref` in degrees.
        )pbdoc";

        const char * tref =
        R"pbdoc(
            Reference time in days.
        )pbdoc";

    } // namespace Body

    namespace Star {

        const char * Star =
        R"pbdoc(
            Instantiate a stellar :py:class:`Body` object.

            The star's radius and luminosity are fixed at unity.

            Args:
                lmax (int): Largest spherical harmonic degree in body's surface map. Default 2.

            .. autoattribute:: map
            .. autoattribute:: flux
            .. autoattribute:: r
            .. autoattribute:: L
        )pbdoc";

        const char * map =
        R"pbdoc(
            The star's surface map, a :py:class:`LimbDarkenedMap` instance.
        )pbdoc";

        const char * r =
        R"pbdoc(
            The star's radius, fixed to unity. *Read-only.*
        )pbdoc";

        const char * L =
        R"pbdoc(
            The star's luminosity, fixed to unity. *Read-only.*
        )pbdoc";

    } // namespace Star

    namespace Planet {

        const char * Planet =
        R"pbdoc(
            Instantiate a planetary :py:class:`Body` object.

            Instantiate a planet. At present, :py:mod:`starry` computes orbits with a simple
            Keplerian solver, so the planet is assumed to be massless.

            Args:
                lmax (int): Largest spherical harmonic degree in body's surface map. Default 2.
                r (float): Body radius in stellar radii. Default 0.1
                L (float): Body luminosity in units of the stellar luminosity. Default 0.
                axis (ndarray): A *normalized* unit vector specifying the body's axis of rotation. Default :math:`\hat{y} = (0, 1, 0)`.
                prot (float): Rotation period in days. Default no rotation.
                theta0 (float): Rotation phase at time :py:obj:`tref` in degrees. Default 0.
                a (float): Semi-major axis in stellar radii. Default 50.
                porb (float): Orbital period in days. Default 1.
                inc (float): Orbital inclination in degrees. Default 90.
                ecc (float): Orbital eccentricity. Default 0.
                w (float): Longitude of pericenter in degrees. Default 90.
                Omega (float): Longitude of ascending node in degrees. Default 0.
                lambda0 (float): Mean longitude at time :py:obj:`tref` in degrees. Default 90.
                tref (float): Reference time in days. Default 0.

            .. autoattribute:: map
            .. autoattribute:: flux
            .. autoattribute:: x
            .. autoattribute:: y
            .. autoattribute:: z
            .. autoattribute:: r
            .. autoattribute:: L
            .. autoattribute:: axis
            .. autoattribute:: prot
            .. autoattribute:: theta0
            .. autoattribute:: a
            .. autoattribute:: porb
            .. autoattribute:: inc
            .. autoattribute:: ecc
            .. autoattribute:: w
            .. autoattribute:: Omega
            .. autoattribute:: lambda0
            .. autoattribute:: tref
        )pbdoc";

    } // namespace Planet

} // namespace docstrings

namespace docstrings_grad {

    const char * starry =
    R"pbdoc(
        starry.grad
        -----------

        .. contents::
            :local:

        Introduction
        ============

        This page documents the :py:mod:`starry.grad` API, which is coded
        in C++ with a :py:mod:`pybind11` Python interface. This API is
        identical in nearly all respects to the :py:mod:`starry` API, except
        that its methods compute gradients with respect to the input parameters,
        in addition to the actual return values. For instance, consider the
        following code block:

        .. code-block:: python

            >>> import starry
            >>> m = starry.Map()
            >>> m[1, 0] = 1
            >>> m.flux(axis=(0, 1, 0), theta=0.3, xo=0.1, yo=0.1, ro=0.1)
            0.9626882655504516

        Here's the same code executed using the :py:obj:`Map()` class in :py:mod:`starry.grad`:

        .. code-block:: python

            >>> import starry
            >>> m = starry.grad.Map()
            >>> m[1, 0] = 1
            >>> m.flux(axis=(0, 1, 0), theta=0.3, xo=0.1, yo=0.1, ro=0.1)
            0.9626882655504516

        So far, they look identical. However, in the second case :py:obj:`starry`
        has also computed the gradient of the flux with respect to each of the
        input parameters (including the map coefficients):

        .. code-block:: python

            >>> m.gradient
            {'Y_{0,0}': array([0.]),
             'Y_{1,-1}': array([-0.00153499]),
             'Y_{1,0}': array([0.96268827]),
             'Y_{1,1}': array([-0.29940113]),
             'Y_{2,-1}': array([0.]),
             'Y_{2,-2}': array([0.]),
             'Y_{2,0}': array([0.]),
             'Y_{2,1}': array([0.]),
             'Y_{2,2}': array([0.]),
             'axis_x': array([0.00045362]),
             'axis_y': array([0.]),
             'axis_z': array([-6.85580453e-05]),
             'ro': array([-0.29791067]),
             'theta': array([-0.29940113]),
             'xo': array([-0.00304715]),
             'yo': array([0.00148905])}

        The :py:attr:`gradient` attribute can be accessed like any Python
        dictionary:

        .. code-block:: python

            >>> m.gradient["ro"]
            array([-0.29791067])
            >>> m.gradient["theta"]
            array([-0.29940113])

        In case :py:obj:`flux` is called with vector arguments, :py:attr:`gradient`
        is also vectorized:

        .. code-block:: python

            >>> import starry
            >>> m = starry.grad.Map()
            >>> m[1, 0] = 1
            >>> m.flux(axis=(0, 1, 0), theta=0.3, xo=[0.1, 0.2, 0.3, 0.4], yo=0.1, ro=0.1)
            array([[0.96268827],
                   [0.96245977],
                   [0.96238958],
                   [0.96249153]])
            >>> m.gradient["ro"]
            array([-0.29791067, -0.30245629, -0.30381564, -0.30170352])
            >>> m.gradient["theta"]
            array([-0.29940113, -0.3009372 , -0.30252224, -0.30416053])

        Note, importantly, that the derivatives in this module are all
        computed **analytically** using autodifferentiation, so their evaluation is fast
        and numerically stable. However, runtimes will in general be slower than those
        in :py:mod:`starry`.

        .. note:: If the degree of the map is large, you may run into a \
                  :py:obj:`RuntimeError` saying too many derivatives were requested. \
                  The :py:obj:`STARRY_NGRAD` compiler flag determines the size of the \
                  gradient vector and can be changed by setting an environment variable \
                  of the same name prior to compiling :py:obj:`starry`. You can do this \
                  by executing \
                  :py:obj:`STARRY_NGRAD=56 pip install --force-reinstall --ignore-installed --no-binary :all: starry`

        As in :py:mod:`starry`, the API consists of a :py:class:`Map` class,
        which houses all of the surface map photometry
        stuff, and the :py:class:`Star`, :py:class:`Planet`, and :py:class:`System`
        classes, which facilitate the generation of light curves for actual
        stellar and planetary systems. There are two broad ways in which users can access
        the core :py:mod:`starry` functionality:

            - Users can instantiate a :py:class:`Map` class to compute phase curves
              and occultation light curves by directly specifying the rotational state
              of the object and (optionally) the position and size of an occultor.
              Users can also instantiate a :py:class:`LimbDarkenedMap` class for
              radially-symmetric stellar surfaces. Both cases
              may be particularly useful for users who wish to integrate :py:mod:`starry`
              with their own dynamical code or for users wishing to compute simple light
              curves without any orbital solutions.

            - Users can instantiate a :py:class:`Star` and one or more :py:class:`Planet`
              objects and feed them into a :py:class:`System` instance for integration
              with the Keplerian solver. All :py:class:`Star` and :py:class:`Planet`
              instances have a :py:obj:`map <>` attribute that allows users to customize
              the surface map prior to computing the system light curve.

        At present, :py:mod:`starry` uses a simple Keplerian solver to compute orbits, so
        the second approach listed above is limited to systems with low mass planets that
        do not exhibit transit timing variations. The next version will include integration
        with an N-body solver, so stay tuned!


        The Map classes
        ===============
        .. autoclass:: Map(lmax=2)
        .. autoclass:: LimbDarkenedMap(lmax=2)


        The orbital classes
        ===================
        .. autoclass:: Star()
        .. autoclass:: Planet(lmax=2, r=0.1, L=0, axis=(0, 1, 0), prot=0, theta0=0, a=50, porb=1, inc=90, ecc=0, w=90, Omega=0, lambda0=90, tref=0)
        .. autoclass:: System(bodies, kepler_tol=1.0e-7, kepler_max_iter=100)
    )pbdoc";

    const char * NotImplemented = docstrings::NotImplemented;

    const char * mp_digits = docstrings::mp_digits;

    const char * ngrad =
    R"pbdoc(
        Length of the gradient vector.
        This is a compile-time constant. If you get errors saying this
        value is too small, you'll need to re-compile :py:obj:`starry` by executing

        .. code-block:: bash

            STARRY_NGRAD=XX pip install --force-reinstall --ignore-installed --no-binary :all: starry
    )pbdoc";

    namespace Map {

        const char * Map =
        R"pbdoc(
                Instantiate a :py:mod:`starry` surface map. Maps instantiated in this fashion
                are *orthonormalized*, so the total integrated luminosity of the map is
                $2\sqrt{\pi} Y_{0,0}$.

                Args:
                    lmax (int): Largest spherical harmonic degree in the surface map. Default 2.

                .. autoattribute:: optimize
                .. automethod:: evaluate(axis=(0, 1, 0), theta=0, x=0, y=0)
                .. automethod:: rotate(axis=(0, 1, 0), theta=0)
                .. automethod:: flux(axis=(0, 1, 0), theta=0, xo=0, yo=0, ro=0)
                .. automethod:: get_coeff(l, m)
                .. automethod:: set_coeff(l, m, coeff)
                .. automethod:: reset()
                .. autoattribute:: gradient
                .. autoattribute:: lmax
                .. autoattribute:: y
                .. autoattribute:: p
                .. autoattribute:: g
                .. autoattribute:: s
                .. autoattribute:: r
                .. automethod:: minimum()
                .. automethod:: load_image(image)
                .. automethod:: load_healpix(image)
                .. automethod:: show(cmap='plasma', res=300)
                .. automethod:: animate(axis=(0, 1, 0), cmap='plasma', res=150, frames=50)
                .. autoattribute:: mp_digits
                .. autoattribute:: ngrad
        )pbdoc";

        const char * get_coeff = docstrings::Map::get_coeff;

        const char * set_coeff = docstrings::Map::set_coeff;

        const char * reset = docstrings::Map::reset;

        const char * gradient =
        R"pbdoc(
            A dictionary of derivatives for all model parameters, populated on
            calls to :py:meth:`flux` and :py:meth:`evaluate`.
        )pbdoc";

        const char * lmax = docstrings::Map::lmax;

        const char * y = docstrings::Map::y;

        const char * p = docstrings::Map::p;

        const char * g = docstrings::Map::g;

        const char * s = docstrings::Map::s;

        const char * r = docstrings::Map::r;

        const char * add_gaussian = docstrings::Map::add_gaussian;

        const char * optimize = docstrings::Map::optimize;

        const char * evaluate = docstrings::Map::evaluate;

        const char * flux = docstrings::Map::flux;

        const char * rotate = docstrings::Map::rotate;

        const char * minimum = docstrings::Map::minimum;

        const char * load_image = docstrings::Map::load_image;

        const char * load_healpix = docstrings::Map::load_healpix;

        const char * show = docstrings::Map::show;

        const char * animate = docstrings::Map::animate;

    } // namespace Map

    namespace LimbDarkenedMap {

        const char * LimbDarkenedMap =
        R"pbdoc(
                Instantiate a :py:mod:`starry` limb-darkened surface map.

                This differs from the base :py:class:`Map` class in that maps
                instantiated this way are radially symmetric: only the radial (:py:obj:`m = 0`)
                coefficients of the map are available. Users edit the map by directly
                specifying the polynomial limb darkening coefficients :py:obj:`u`, starting
                with $u_1$ (linear limb darkening). The coefficient $u_0$ is fixed to enforce
                the correct normalization.

                .. warning:: Unlike :py:class:`Map`, maps instantiated this \
                             way are normalized so that the integral of the specific intensity over the \
                             visible disk is unity. This is convenient for using this map to model \
                             stars: the unocculted flux from the star is equal to one, regardless of the limb-darkening \
                             coefficients!

                Args:
                    lmax (int): Largest spherical harmonic degree in the surface map. Default 2.

                .. autoattribute:: optimize
                .. automethod:: evaluate(x=0, y=0)
                .. automethod:: flux(xo=0, yo=0, ro=0)
                .. automethod:: get_coeff(l)
                .. automethod:: set_coeff(l, coeff)
                .. automethod:: reset()
                .. autoattribute:: gradient
                .. autoattribute:: lmax
                .. autoattribute:: y
                .. autoattribute:: p
                .. autoattribute:: g
                .. autoattribute:: u
                .. autoattribute:: s
                .. automethod:: show(cmap='plasma', res=300)
                .. autoattribute:: mp_digits
                .. autoattribute:: ngrad

        )pbdoc";

        const char * get_coeff = docstrings::LimbDarkenedMap::get_coeff;

        const char * set_coeff = docstrings::LimbDarkenedMap::set_coeff;

        const char * reset = docstrings::LimbDarkenedMap::reset;

        const char * gradient =
        R"pbdoc(
            A dictionary of derivatives for all model parameters, populated on
            calls to :py:meth:`flux` and :py:meth:`evaluate`.
        )pbdoc";

        const char * lmax = docstrings::LimbDarkenedMap::lmax;

        const char * y = docstrings::LimbDarkenedMap::y;

        const char * p = docstrings::LimbDarkenedMap::p;

        const char * g = docstrings::LimbDarkenedMap::g;

        const char * s = docstrings::LimbDarkenedMap::s;

        const char * u = docstrings::LimbDarkenedMap::u;

        const char * optimize = docstrings::LimbDarkenedMap::optimize;

        const char * evaluate = docstrings::LimbDarkenedMap::evaluate;

        const char * flux = docstrings::LimbDarkenedMap::flux;

        const char * show = docstrings::LimbDarkenedMap::show;

    } // namespace LimbDarkenedMap

    namespace System {

        const char * System =
        R"pbdoc(
                Instantiate an orbital system.

                Args:
                    bodies (list): List of bodies in the system, with the primary (usually the star) listed first.
                    kepler_tol (float): Kepler solver tolerance. Default `1e-7`.
                    kepler_max_iter (int): Maximum number of iterations in the Kepler solver. Default `100`.
                    exposure_time (float): The exposure time of the observations in days. If nonzero, the flux will \
                                           be integrated over this exposure time. Default `0`.
                    exposure_tol (float): Tolerance of the recursive method for integrating the flux over the exposure time. Default `1e-8`.
                    exposure_maxdepth (int): Maximum recursion depth for the exposure calculation. Default `4`.

                .. automethod:: compute(time)
                .. autoattribute:: flux
                .. autoattribuet:: exposure_time
                .. autoattribuet:: exposure_tol
                .. autoattribuet:: exposure_max_depth
                .. autoattribuet:: kepler_tol
                .. autoattribuet:: kepler_max_iter

        )pbdoc";

        const char * compute = docstrings::System::compute;

        const char * flux = docstrings::System::flux;

        const char * exposure_time = docstrings::System::exposure_time;

        const char * exposure_tol = docstrings::System::exposure_tol;

        const char * exposure_max_depth = docstrings::System::exposure_max_depth;

        const char * kepler_max_iter = docstrings::System::kepler_max_iter;

        const char * kepler_tol = docstrings::System::kepler_tol;

        const char * gradient =
        R"pbdoc(
            A dictionary of derivatives of the system flux with respect to
            all model parameters, populated on calls to :py:meth:`compute`.

            .. note:: This dictionary is similar to the :py:obj:`gradient` \
                      attribute of a :py:obj:`Map` instance, but the keys in \
                      the dictionary are prepended by either `star.` (for \
                      the star) or `planetX` (where `X` is the planet number,
                      starting with 1). For instance, the gradient of the \
                      system flux with respect to the second planet's eccentricity \
                      is :py:obj:`gradient['planet2.ecc']`.
        )pbdoc";

    } // namespace System

    namespace Body {

        const char * map = docstrings::Body::map;

        const char * flux = docstrings::Body::flux;

        const char * gradient =
        R"pbdoc(
            A dictionary of derivatives of the body's flux with respect to
            all model parameters, populated on calls to :py:meth:`System.compute`.

            .. note:: This dictionary is similar to the :py:obj:`gradient` \
                      attribute of a :py:obj:`Map` instance, but the keys in \
                      the dictionary are prepended by either `star.` (for \
                      the star) or `planetX` (where `X` is the planet number,
                      starting with 1). For instance, the gradient of this body's \
                      flux with respect to the second planet's eccentricity \
                      is :py:obj:`gradient['planet2.ecc']`.
        )pbdoc";

        const char * x = docstrings::Body::x;

        const char * y = docstrings::Body::y;

        const char * z = docstrings::Body::z;

        const char * r = docstrings::Body::r;

        const char * L = docstrings::Body::L;

        const char * axis = docstrings::Body::axis;

        const char * prot = docstrings::Body::prot;

        const char * theta0 = docstrings::Body::theta0;

        const char * a = docstrings::Body::a;

        const char * porb = docstrings::Body::porb;

        const char * inc = docstrings::Body::inc;

        const char * ecc = docstrings::Body::ecc;

        const char * w = docstrings::Body::w;

        const char * Omega = docstrings::Body::Omega;

        const char * lambda0 = docstrings::Body::lambda0;

        const char * tref = docstrings::Body::tref;

    } // namespace Body

    namespace Star {

        const char * Star =
        R"pbdoc(
            Instantiate a stellar :py:class:`Body` object.

            The star's radius and luminosity are fixed at unity.

            Args:
                lmax (int): Largest spherical harmonic degree in body's surface map. Default 2.

            .. autoattribute:: map
            .. autoattribute:: flux
            .. autoattribute:: gradient
            .. autoattribute:: r
            .. autoattribute:: L
        )pbdoc";

        const char * map = docstrings::Star::map;

        const char * r = docstrings::Star::r;

        const char * L = docstrings::Star::L;

    } // namespace Star

    namespace Planet {

        const char * Planet =
        R"pbdoc(
            Instantiate a planetary :py:class:`Body` object.

            Instantiate a planet. At present, :py:mod:`starry` computes orbits with a simple
            Keplerian solver, so the planet is assumed to be massless.

            Args:
                lmax (int): Largest spherical harmonic degree in body's surface map. Default 2.
                r (float): Body radius in stellar radii. Default 0.1
                L (float): Body luminosity in units of the stellar luminosity. Default 0.
                axis (ndarray): A *normalized* unit vector specifying the body's axis of rotation. Default :math:`\hat{y} = (0, 1, 0)`.
                prot (float): Rotation period in days. Default no rotation.
                theta0 (float): Rotation phase at time :py:obj:`tref` in degrees. Default 0.
                a (float): Semi-major axis in stellar radii. Default 50.
                porb (float): Orbital period in days. Default 1.
                inc (float): Orbital inclination in degrees. Default 90.
                ecc (float): Orbital eccentricity. Default 0.
                w (float): Longitude of pericenter in degrees. Default 90.
                Omega (float): Longitude of ascending node in degrees. Default 0.
                lambda0 (float): Mean longitude at time :py:obj:`tref` in degrees. Default 90.
                tref (float): Reference time in days. Default 0.

            .. autoattribute:: map
            .. autoattribute:: flux
            .. autoattribute:: gradient
            .. autoattribute:: x
            .. autoattribute:: y
            .. autoattribute:: z
            .. autoattribute:: r
            .. autoattribute:: L
            .. autoattribute:: axis
            .. autoattribute:: prot
            .. autoattribute:: theta0
            .. autoattribute:: a
            .. autoattribute:: porb
            .. autoattribute:: inc
            .. autoattribute:: ecc
            .. autoattribute:: w
            .. autoattribute:: Omega
            .. autoattribute:: lambda0
            .. autoattribute:: tref
        )pbdoc";

    } // namespace Planet

} // namespace docstrings_grad

#endif
