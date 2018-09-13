/**
Docstrings for the Python functions.

TODO: The docs need a lot of work still.

TODO: Add a note to the docs about when we don't compute
      dF/dy or dF/du, and explain how to override this. I.e.:

        Note that we intentionally don't compute the spherical
        harmonic coeff derivs above Y_{0,0}, since that
        would make this function slow. If users *really*
        need them, set one of the l > 0 coeffs to a very
        small (~1e-14) value to force the code to use the
        generalized `flux` method.

*/

#ifndef _STARRY_DOCS_H_
#define _STARRY_DOCS_H_
#include <stdlib.h>
#include "utils.h"

namespace docstrings {

    using namespace std;

    namespace starry {

        const char* doc = R"pbdoc()pbdoc";

    }

    namespace Map {

        const char* doc = R"pbdoc(
            Instantiate a :py:mod:`starry` surface map.

            Args:
                lmax (int): Largest spherical harmonic degree \
                    in the surface map. Default 2.
                nwav (int): Number of map wavelength bins. Default 1.
                multi (bool): Use multi-precision to perform all \
                    calculations? Default :py:obj:`False`. If :py:obj:`True`, \
                    defaults to 32-digit (approximately 128-bit) floating point \
                    precision. This can be adjusted by changing the \
                    :py:obj:`STARRY_NMULTI` compiler macro.

            .. automethod:: __call__(theta=0, x=0, y=0)
            .. automethod:: flux(theta=0, xo=0, yo=0, ro=0, gradient=False)
            .. automethod:: rotate(theta=0)
            .. automethod:: show(cmap='plasma', res=300)
            .. automethod:: animate(cmap='plasma', res=150, frames=50, interval=75, gif='')
            .. automethod:: reset()
            .. autoattribute:: lmax
            .. autoattribute:: nwav
            .. autoattribute:: N
            .. autoattribute:: y
            .. autoattribute:: u
            .. autoattribute:: p
            .. autoattribute:: g
            .. autoattribute:: r
            .. autoattribute:: s
            .. autoattribute:: axis
        )pbdoc";

        const char* reset = R"pbdoc(
            Set all of the map coefficients to zero, except for :math:`Y_{0,0}`, \
            which is set to unity.
        )pbdoc";

        const char* lmax = R"pbdoc(
            The highest spherical harmonic degree of the map. *Read-only.*
        )pbdoc";

        const char* N = R"pbdoc(
            The number of map coefficients, equal to :math:`(l + 1)^2`. *Read-only.*
        )pbdoc";

        const char* precision = R"pbdoc(
            The floating-point precision of the map in digits. *Read-only.*
        )pbdoc";

        const char* nwav = R"pbdoc(
            The number of wavelength bins. *Read-only.*
        )pbdoc";

        const char* y = R"pbdoc(
            The spherical harmonic map vector. This is a vector of the coefficients \
            of the spherical harmonics :math:`Y_{0,0}, Y_{1,-1}, Y_{1,0}, Y_{1,1}, ...`. \
            *Read-only.*
        )pbdoc";

        const char* u = R"pbdoc(
            The limb darkening map vector. This is a vector of the limb darkening \
            coefficients :math:`u_1, u_2, u_3, ...`. *Read-only.*
        )pbdoc";

        const char* p = R"pbdoc(
            The polynomial map vector. This is a vector of the coefficients of the \
            polynomial basis :math:`1, x, z, y, x^2, xz, ...`. \
            *Read-only.*
        )pbdoc";

        const char* g = R"pbdoc(
            The Green's polynomial map vector. This is a vector of the coefficients of the \
            polynomial basis :math:`1, 2x, z, y, 3x^2, -3xz, ...`. \
            *Read-only.*
        )pbdoc";

        const char* r = R"pbdoc(
            The current rotation solution vector `r`. Each term in this vector corresponds \
            to the total flux observed from each of the terms in the polynomial basis. \
            *Read-only.*
        )pbdoc";

        const char* s = R"pbdoc(
            The current occultation solution vector `s`. Each term in this vector corresponds to the \
            total flux observeed from each of the terms in the Green's basis. *Read-only.*
            
            .. note:: For pure linear and quadratic limb darkening, the full solution vector is **not** \
                computed, so this vector will not necessarily reflect the true solution coefficients.
        )pbdoc";

        const char* axis = R"pbdoc(
            A *normalized* unit vector specifying the default axis of rotation for the map. \
            Default :math:`\hat{y} = (0, 1, 0)`.
        )pbdoc";

        const char* evaluate = R"pbdoc(
            Return the specific intensity at a point :py:obj:`(x, y)` on the map.
            Users may optionally provide a rotation state. Note that this does
            not rotate the base map.

            Args:
                theta (float or ndarray): Angle of rotation in degrees. Default 0.
                x (float or ndarray): Position scalar or vector.
                y (float or ndarray): Position scalar or vector.

            Returns:
                The specific intensity at :py:obj:`(x, y)`.
        )pbdoc";

        const char* flux = R"pbdoc(
            Return the total flux received by the observer from the map.
            Computes the total flux received by the observer
            during or outside of an occultation.

            Args:
                theta (float or ndarray): Angle of rotation. Default 0.
                xo (float or ndarray): The :py:obj:`x` position of the occultor (if any). Default 0.
                yo (float or ndarray): The :py:obj:`y` position of the occultor (if any). Default 0.
                ro (float): The radius of the occultor in units of this body's radius. Default 0 (no occultation).
                gradient (bool): Compute and return the gradient of the flux as well? Default :py:obj:`False`.

            Returns:
                The flux received by the observer (a scalar or a vector). \
                If :py:obj:`gradient` is :py:obj:`True`, \
                returns the tuple :py:obj:`(F, dF)`, where :py:obj:`F` is the flux and :py:obj:`dF` is \
                a dictionary containing the derivatives with respect to each of the input parameters \
                and each of the map coefficients.
        )pbdoc";

        const char* rotate = R"pbdoc(
            Rotate the base map an angle :py:obj:`theta` about :py:obj:`axis`.
            This performs a permanent rotation to the base map. Subsequent
            rotations and calculations will be performed relative to this
            rotational state.

            Args:
                theta (float or ndarray): Angle of rotation in degrees. Default 0.
        )pbdoc";

        const char* is_physical = R"pbdoc(
            Check whether the map is positive semi-definite (PSD). Returns :py:obj:`True`
            if the map is PSD, :py:obj:`False` otherwise. For pure limb-darkened maps,
            this routine uses Sturm's theorem to find the number of roots. For pure
            spherical harmonic maps up to :py:obj:`l = 1`, the solution is analytic. For all
            other cases, this routine attempts to find the global minimum numerically
            and checks if it is negative. For maps with :py:obj:`nwav > 1`, this routine
            returns an array of boolean values, one per wavelength bin.

            Args:
                epsilon (float): Numerical tolerance. Default :math:`10^{-6}`
                max_iterations (int): Maximum number of iterations for the \
                    numerical solver. Default 100
        )pbdoc";

        const char* show = R"pbdoc(
            Convenience routine to quickly display the body's surface map.

            Args:
                cmap (str): The :py:mod:`matplotlib` colormap name. Default :py:obj:`plasma`.
                res (int): The resolution of the map in pixels on a side. Default 300.

            .. note:: For maps with :py:obj:`nwav > 1`, this method displays an
                animated sequence of frames, one per wavelength bin. Users can save
                this to disk by specifying a :py:obj:`gif` keyword with the name of the
                GIF image to save to.

        )pbdoc";

        const char* animate = R"pbdoc(
            Convenience routine to animate the body's surface map as it rotates.

            Args:
                cmap (str): The :py:mod:`matplotlib` colormap name. Default :py:obj:`plasma`.
                res (int): The resolution of the map in pixels on a side. Default 150.
                frames (int): The number of frames in the animation. Default 50.
                interval (int): Interval in milliseconds between frames. Default 75.
                gif (str): The name of the `.gif` file to save the animation to. \
                           Requires `ImageMagick` to be installed. If set, does not \
                           show the animation. Default :py:obj:`None`.
        )pbdoc";

        const char* load_image = R"pbdoc(
            Load an image from file.
            This routine loads an image file, computes its spherical harmonic
            expansion up to degree :py:attr:`lmax`, and sets the map vector.

            Args:
                image (str): The full path to the image file.
                lmax (int): The maximum degree of the spherical harmonic expansion \
                            of the image. Default :py:attr:`lmax`.

            .. note:: For maps with :py:obj:`nwav > 1`, users may specify a
                :py:obj:`nwav` keyword argument indicating the wavelength bin
                into which the image will be loaded.

        )pbdoc";

    }

    namespace kepler {

        const char* doc = R"pbdoc()pbdoc";

    }

    namespace Body {

        const char* r = R"pbdoc()pbdoc";

        const char* L = R"pbdoc()pbdoc";

        const char* tref = R"pbdoc()pbdoc";

        const char* prot = R"pbdoc()pbdoc";

        const char* lightcurve = R"pbdoc()pbdoc";

        const char* gradient = R"pbdoc()pbdoc";

        // TODO

    }

    namespace Primary {

        using namespace Body;

        const char* doc = R"pbdoc(

            .. autoattribute:: r
            .. autoattribute:: L
            .. autoattribute:: tref
            .. autoattribute:: prot
            .. autoattribute:: lightcurve
            .. autoattribute:: gradient
            .. autoattribute:: r_m

        )pbdoc";

        const char* r_m = R"pbdoc()pbdoc";

        // TODO

    }

    namespace Secondary {

        using namespace Body;

        const char* doc = R"pbdoc(

            .. autoattribute:: r
            .. autoattribute:: L
            .. autoattribute:: tref
            .. autoattribute:: prot
            .. autoattribute:: lightcurve
            .. autoattribute:: gradient
            .. autoattribute:: a
            .. autoattribute:: porb
            .. autoattribute:: inc
            .. autoattribute:: ecc
            .. autoattribute:: w
            .. autoattribute:: Omega
            .. autoattribute:: lambda0
            .. autoattribute:: X
            .. autoattribute:: Y
            .. autoattribute:: Z

        )pbdoc";

        const char* a = R"pbdoc()pbdoc";

        const char* porb = R"pbdoc()pbdoc";

        const char* inc = R"pbdoc()pbdoc";

        const char* ecc = R"pbdoc()pbdoc";

        const char* w = R"pbdoc()pbdoc";

        const char* Omega = R"pbdoc()pbdoc";

        const char* lambda0 = R"pbdoc()pbdoc";

        const char* X = R"pbdoc()pbdoc";

        const char* Y = R"pbdoc()pbdoc";

        const char* Z = R"pbdoc()pbdoc";

        // TODO

    }

    namespace System {

        const char* doc = R"pbdoc(

            .. automethod:: compute
            .. autoattribute:: lightcurve
            .. autoattribute:: gradient
            .. autoattribute:: exposure_time
            .. autoattribute:: exposure_tol
            .. autoattribute:: exposure_max_depth

        )pbdoc";

        const char* compute = R"pbdoc()pbdoc";

        const char* lightcurve = R"pbdoc()pbdoc";

        const char* gradient = R"pbdoc()pbdoc";

        const char* exposure_time = R"pbdoc()pbdoc";

        const char* exposure_tol = R"pbdoc()pbdoc";

        const char* exposure_max_depth = R"pbdoc()pbdoc";

        // TODO

    }
}

#endif
