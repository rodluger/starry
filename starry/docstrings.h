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

    namespace Map {

        const char * Map =
        R"pbdoc(
                Instantiate a :py:mod:`starry` surface map.

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
                .. automethod:: load_image(image)
                .. automethod:: load_healpix(image)
                .. automethod:: show(cmap='plasma', res=300)
                .. automethod:: animate(axis=(0, 1, 0), cmap='plasma', res=150, frames=50)
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
            The current multi-precision solution vector `s`. Only available after :py:method:`flux_mp` has been called. *Read-only.*
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
                specifying the polynomial limb darkening coefficients :py:obj:`u`.

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
            The current multi-precision solution vector `s`. Only available after :py:method:`flux_mp` has been called. *Read-only.*
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
        that its methods return gradients with respect to the input parameters,
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
            >>> m = starry.Map()
            >>> m[1, 0] = 1
            >>> m.flux(axis=(0, 1, 0), theta=0.3, xo=0.1, yo=0.1, ro=0.1)
            array([[ 9.62688266e-01,  4.53620580e-04,  0.00000000e+00,
                    -6.85580453e-05, -2.99401131e-01, -3.04715096e-03,
                    1.48905485e-03, -2.97910667e-01]])

        The :py:obj:`flux()` method now returns a vector, where the first value is the
        actual flux and the remaining seven values are the derivatives of the flux
        with respect to each of the input parameters
        :py:obj:`(axis[0], axis[1], axis[2], theta, xo, yo, ro)`. Note that as in
        :py:mod:`starry`, many of the functions in :py:mod:`starry.grad` are
        vectorizable, meaning that vectors can be provided as inputs to compute,
        say, the light curve for an entire timeseries. In this case, the return
        values are **matrices**, with one vector of :py:obj:`(value, derivs)` per row.

        Note, importantly, that the derivatives in this module are all
        computed **analytically** using autodifferentiation, so their evaluation is fast
        and numerically stable. However, runtimes will in general be slower than those
        in :py:mod:`starry`.

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

    namespace Map {

        const char * Map =
        R"pbdoc(
                Instantiate a :py:mod:`starry` surface map.

                Args:
                    lmax (int): Largest spherical harmonic degree in the surface map. Default 2.

                .. autoattribute:: optimize
                .. automethod:: evaluate(axis=(0, 1, 0), theta=0, x=0, y=0)
                .. automethod:: rotate(axis=(0, 1, 0), theta=0)
                .. automethod:: flux(axis=(0, 1, 0), theta=0, xo=0, yo=0, ro=0)
                .. automethod:: get_coeff(l, m)
                .. automethod:: set_coeff(l, m, coeff)
                .. automethod:: reset()
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
        )pbdoc";

        const char * get_coeff = docstrings::Map::get_coeff;

        const char * set_coeff = docstrings::Map::set_coeff;

        const char * reset = docstrings::Map::reset;

        const char * lmax = docstrings::Map::lmax;

        const char * y = docstrings::Map::y;

        const char * p = docstrings::Map::p;

        const char * g = docstrings::Map::g;

        const char * s = docstrings::Map::s;

        const char * r = docstrings::Map::r;

        const char * optimize = docstrings::Map::optimize;

        const char * evaluate = docstrings::Map::evaluate;

        const char * flux = docstrings::Map::flux;

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

            .. note:: This function only returns the **value** of the numerical flux, and **not** its \
                      derivatives. Autodifferentiation of numerical integration is \
                      simply a terrible idea!
        )pbdoc";

        const char * rotate = docstrings::Map::rotate;

        const char * minimum = docstrings::Map::minimum;

        const char * load_image = docstrings::Map::load_image;

        const char * load_healpix = docstrings::Map::load_healpix;

        const char * show = docstrings::Map::show;

        const char * animate = docstrings::Map::animate;

        const char * map_gradients =
        R"pbdoc(
            Compute gradients with respect to the map coefficients?
            Default :py:obj:`False`, in which case only gradients
            with respect to the orbital parameters are computed. If
            Default :py:obj:`True`, the map gradients are appended to
            the end of the vector returned by the functions
            :py:method:`flux` and :py:method:`flux`.
        )pbdoc";

    } // namespace Map

    namespace LimbDarkenedMap {

        const char * LimbDarkenedMap =
        R"pbdoc(
                Instantiate a :py:mod:`starry` limb-darkened surface map.

                This differs from the base :py:class:`Map` class in that maps
                instantiated this way are radially symmetric: only the radial (:py:obj:`m = 0`)
                coefficients of the map are available. Users edit the map by directly
                specifying the polynomial limb darkening coefficients :py:obj:`u`.

                Args:
                    lmax (int): Largest spherical harmonic degree in the surface map. Default 2.

                .. autoattribute:: optimize
                .. automethod:: evaluate(x=0, y=0)
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
                .. automethod:: show(cmap='plasma', res=300)
        )pbdoc";

        const char * get_coeff = docstrings::LimbDarkenedMap::get_coeff;

        const char * set_coeff = docstrings::LimbDarkenedMap::set_coeff;

        const char * reset = docstrings::LimbDarkenedMap::reset;

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

        const char * map_gradients =
        R"pbdoc(
            Compute gradients with respect to the map coefficients?
            Default :py:obj:`False`, in which case only gradients
            with respect to the orbital parameters are computed. If
            Default :py:obj:`True`, the map gradients are appended to
            the end of the vector returned by the functions
            :py:method:`flux` and :py:method:`flux`.
        )pbdoc";

    } // namespace LimbDarkenedMap

} // namespace docstrings_grad

#endif
