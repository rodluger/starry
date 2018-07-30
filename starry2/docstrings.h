/**
Docstrings for the Python functions.

*/

#ifndef _STARRY_DOCS_H_
#define _STARRY_DOCS_H_
#include <stdlib.h>
#include "utils.h"

namespace docstrings {

    using namespace std;

    template <class T>
    class Map_ {
    public:
        const char * doc;
        const char * get_coeff;
        const char * set_coeff;
        const char * reset;
        const char * lmax;
        const char * y;
        const char * p;
        const char * g;
        const char * s;
        const char * r;
        const char * axis;
        const char * reparam;
        const char * evaluate;
        const char * flux;
        const char * flux_numerical;
        const char * rotate;
        const char * psd;
        const char * add_gaussian;
        const char * load_array;
        const char * load_image;
        const char * load_healpix;
        const char * show;
        const char * animate;
        const char * gradient;
        void add_extras() {};

        Map_(){

            get_coeff = R"pbdoc(
                Return the (:py:obj:`l`, :py:obj:`m`) coefficient of the map.

                .. note:: Users can also retrieve a coefficient by accessing the \
                          [:py:obj:`l`, :py:obj:`m`] index of the map as if it \
                          were a 2D array. Single slice indexing is also allowed.

                Args:
                    l (int): The spherical harmonic degree, ranging from 0 to :py:attr:`lmax`.
                    m (int): The spherical harmonic order, ranging from -:py:obj:`l` to :py:attr:`l`.
            )pbdoc";

            set_coeff = R"pbdoc(
                Set the (:py:obj:`l`, :py:obj:`m`) coefficient of the map.

                .. note:: Users can also set a coefficient by setting the \
                          [:py:obj:`l`, :py:obj:`m`] index of the map as if it \
                          were a 2D array. Single slice indexing is also allowed.

                Args:
                    l (int): The spherical harmonic degree, ranging from 0 to :py:attr:`lmax`.
                    m (int): The spherical harmonic order, ranging from -:py:obj:`l` to :py:attr:`l`.
                    coeff (float): The value of the coefficient.
            )pbdoc";

            reset = R"pbdoc(
                Set all of the map coefficients to zero.
            )pbdoc";

            lmax = R"pbdoc(
                The highest spherical harmonic order of the map. *Read-only.*
            )pbdoc";

            y = R"pbdoc(
                The spherical harmonic map vector. *Read-only.*
            )pbdoc";

            p = R"pbdoc(
                The polynomial map vector. *Read-only.*
            )pbdoc";

            g = R"pbdoc(
                The Green's polynomial map vector. *Read-only.*
            )pbdoc";

            s = R"pbdoc(
                The current solution vector `s`. *Read-only.*
            )pbdoc";

            r = R"pbdoc(
                The current solution vector `r`. *Read-only.*
            )pbdoc";

            axis = R"pbdoc(
                *Normalized* unit vector specifying the body's axis of rotation. Default :math:`\hat{y} = (0, 1, 0)`.
            )pbdoc";

            reparam = R"pbdoc(
                Set to :py:obj:`False` to disable reparametrization of the primitive integrals when \
                computing occultation light curves for large occultors. This is in general not something you should do! \
                Default :py:obj:`True`.
            )pbdoc";

            evaluate = R"pbdoc(
                Return the specific intensity at a point (`x`, `y`) on the map.
                Users may optionally provide a rotation state. Note that this does
                not rotate the base map.

                Args:
                    theta (float or ndarray): Angle of rotation in degrees. Default 0.
                    x (float or ndarray): Position scalar, vector, or matrix.
                    y (float or ndarray): Position scalar, vector, or matrix.

                Returns:
                    The specific intensity at (`x`, `y`).
            )pbdoc";

            flux = R"pbdoc(
                Return the total flux received by the observer.
                Computes the total flux received by the observer from the
                map during or outside of an occultation.

                Args:
                    theta (float or ndarray): Angle of rotation. Default 0.
                    xo (float or ndarray): The `x` position of the occultor (if any). Default 0.
                    yo (float or ndarray): The `y` position of the occultor (if any). Default 0.
                    ro (float): The radius of the occultor in units of this body's radius. Default 0 (no occultation).

                Returns:
                    The flux received by the observer (a scalar or a vector).
            )pbdoc";

            rotate = R"pbdoc(
                Rotate the base map an angle :py:obj:`theta` about :py:obj:`axis`.
                This performs a permanent rotation to the base map. Subsequent
                rotations and calculations will be performed relative to this
                rotational state.

                Args:
                    theta (float or ndarray): Angle of rotation in degrees. Default 0.
            )pbdoc";

            psd = R"pbdoc(
                Check whether the map is positive semi-definite. Returns :py:obj:`True`
                if it is positive semi-definite and :py:obj:`False` otherwise.
                For maps of degree `l = 0` and `l = 1`, this is analytic and fast to
                compute, but for maps of higher degree a numerical solution is employed.

                Args:
                    epsilon (float): Numerical solver tolerance. Default `1e-6`
                    max_iterations (int): Maximum number of iterations for the numerical solver. Default `100`
            )pbdoc";

            add_gaussian = R"pbdoc(
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
                    lat (float): The latitude of the center of the gaussian in degrees. Default 0.
                    lon (float): The longitude of the center of the gaussian in degrees. Default 0.
            )pbdoc";

            load_array = R"pbdoc(
                Load a lat-lon image array.
                This routine loads a 2D :py:obj:`numpy` array, computes its
                spherical harmonic expansion up to degree :py:attr:`lmax`,
                and sets the map vector.

                Args:
                    image (ndarray): The 2D :py:obj:`numpy` lat-lon array.
            )pbdoc";

            load_image = R"pbdoc(
                Load an image from file.
                This routine loads an image file, computes its spherical harmonic
                expansion up to degree :py:attr:`lmax`, and sets the map vector.

                Args:
                    image (str): The full path to the image file.
            )pbdoc";

            load_healpix = R"pbdoc(
                Load a healpix image array.
                This routine loads a :py:obj:`healpix` array, computes its
                spherical harmonic
                expansion up to degree :py:attr:`lmax`, and sets the map vector.

                Args:
                    image (ndarray): The ring-ordered :py:obj:`healpix` array.

                .. todo:: This routine has not been tested. If you have any \
                          problems with it, please submit an issue on GitHub.
            )pbdoc";

            show = R"pbdoc(
                Convenience routine to quickly display the body's surface map.

                Args:
                    cmap (str): The :py:mod:`matplotlib` colormap name. Default `plasma`.
                    res (int): The resolution of the map in pixels on a side. Default 300.
            )pbdoc";

            animate = R"pbdoc(
                Convenience routine to animate the body's surface map as it rotates.

                Args:
                    cmap (str): The :py:mod:`matplotlib` colormap name. Default `plasma`.
                    res (int): The resolution of the map in pixels on a side. Default 150.
                    frames (int): The number of frames in the animation. Default 50.
            )pbdoc";

            add_extras();

        }
    };

    template <>
    void Map_<double>::add_extras() {

        doc = R"pbdoc(
                Instantiate a :py:mod:`starry` surface map. Maps instantiated in this fashion
                are *orthonormalized*, so the total integrated luminosity of the map is
                :math:`2\sqrt{\pi} Y_{0,0}`.

                Args:
                    lmax (int): Largest spherical harmonic degree in the surface map. Default 2.

                .. automethod:: evaluate(theta=0, x=0, y=0)
                .. automethod:: rotate(theta=0)
                .. automethod:: flux(theta=0, xo=0, yo=0, ro=0)
                .. automethod:: get_coeff(l, m)
                .. automethod:: set_coeff(l, m, coeff)
                .. automethod:: reset()
                .. autoattribute:: lmax
                .. autoattribute:: y
                .. autoattribute:: p
                .. autoattribute:: g
                .. autoattribute:: s
                .. autoattribute:: r
                .. automethod:: psd()
                .. automethod:: add_gaussian()
                .. automethod:: load_array(image)
                .. automethod:: load_image(image)
                .. automethod:: load_healpix(image)
                .. automethod:: show(cmap='plasma', res=300)
                .. automethod:: animate(cmap='plasma', res=150, frames=50)
            )pbdoc";

        flux_numerical = R"pbdoc(
            Return the total flux received by the observer, computed numerically.
            Computes the total flux received by the observer from the
            map during or outside of an occultation. The flux is computed
            numerically using an adaptive radial mesh.

            Args:
                theta (float or ndarray): Angle of rotation. Default 0.
                xo (float or ndarray): The `x` position of the occultor (if any). Default 0.
                yo (float or ndarray): The `y` position of the occultor (if any). Default 0.
                ro (float): The radius of the occultor in units of this body's radius. Default 0 (no occultation).
                tol (float): Tolerance of the numerical solver. Default `1.e-4`

            Returns:
                The flux received by the observer (a scalar or a vector).
            )pbdoc";

    };

    template <>
    void Map_<utils::Multi>::add_extras() {

        doc = R"pbdoc()pbdoc";

    };

    template <class T>
    class docs {
    public:

        const char * doc;
        const char * NotImplemented;
        const char * nmulti;
        const char * ngrad;
        void add_extras() { };
        Map_<T> Map;

        docs() : Map() {

            NotImplemented = R"pbdoc(
                Method or attribute not implemented for this class.
            )pbdoc";

            add_extras();

        }
    };

    template <>
    void docs<double>::add_extras() {

        doc = R"pbdoc(
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
            .. autoclass:: Planet(lmax=2, r=0.1, L=0, axis=(0, 1, 0), prot=0, a=50, porb=1, inc=90, ecc=0, w=90, Omega=0, lambda0=90, tref=0)
            .. autoclass:: System(bodies, scale=0, exposure_time=0, exposure_tol=1e-8, exposure_max_depth=4)
        )pbdoc";

    };

    template <>
    void docs<utils::Multi>::add_extras() {

        doc = R"pbdoc()pbdoc";

    };

}

#endif
