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

            .. automethod:: __call__(theta=0, x=0, y=0)
            .. automethod:: flux(theta=0, xo=0, yo=0, ro=0)
            .. automethod:: rotate(theta=0)
            .. automethod:: show()
            .. automethod:: animate()
            .. automethod:: reset()
            .. autoattribute:: lmax
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
            Set all of the map coefficients to zero.
        )pbdoc";

        const char* lmax = R"pbdoc(
            The highest spherical harmonic order of the map. *Read-only.*
        )pbdoc";

        const char* N = R"pbdoc(
            The number of map coefficients, equal to `(l + 1) ** 2`. *Read-only.*
        )pbdoc";

        const char* precision = R"pbdoc(
            The floating-point precision of the map. *Read-only.*
        )pbdoc";

        const char* nwav = R"pbdoc(
            The number of wavelength bins. *Read-only.*
        )pbdoc";

        const char* y = R"pbdoc(
            The spherical harmonic map vector. *Read-only.*
        )pbdoc";

        const char* u = R"pbdoc(
            The limb darkening map vector. *Read-only.*
        )pbdoc";

        const char* p = R"pbdoc(
            The polynomial map vector. *Read-only.*
        )pbdoc";

        const char* g = R"pbdoc(
            The Green's polynomial map vector. *Read-only.*
        )pbdoc";

        const char* r = R"pbdoc(
            The current solution vector `r`. *Read-only.*
        )pbdoc";

        const char* s = R"pbdoc(
            The current solution vector `s`. *Read-only.*
        )pbdoc";

        const char* axis = R"pbdoc(
            *Normalized* unit vector specifying the body's axis of rotation. Default :math:`\hat{y} = (0, 1, 0)`.
        )pbdoc";

        const char* evaluate = R"pbdoc(
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

        const char* flux = R"pbdoc(
            Return the total flux received by the observer.
            Computes the total flux received by the observer from the
            map during or outside of an occultation.

            Args:
                theta (float or ndarray): Angle of rotation. Default 0.
                xo (float or ndarray): The `x` position of the occultor (if any). Default 0.
                yo (float or ndarray): The `y` position of the occultor (if any). Default 0.
                ro (float): The radius of the occultor in units of this body's radius. Default 0 (no occultation).
                gradient (bool): Compute and return the gradient of the flux as well? Default :py:obj:`False`.

            Returns:
                The flux received by the observer (a scalar or a vector). \
                If :py:obj:`gradient` is :py:obj:`True`, \
                returns the tuple `(F, dF)`, where `F` is the flux and `dF` is \
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
            spherical harmonic maps up to `l = 1`, the solution is analytic. For all
            other cases, this routine attempts to find the global minimum numerically
            and checks if it is negative. For maps with `nwav > 1`, this routine
            returns an array of boolean values, one per wavelength bin.

            Args:
                epsilon (float): Numerical tolerance. Default 1.e-6
                max_iterations (int): Maximum number of iterations for the \
                    numerical solver. Default 100
        )pbdoc";

        const char* show = R"pbdoc(
            Convenience routine to quickly display the body's surface map.
            Args:
                cmap (str): The :py:mod:`matplotlib` colormap name. Default `plasma`.
                res (int): The resolution of the map in pixels on a side. Default 300.
        )pbdoc";

        const char* animate = R"pbdoc(
            Convenience routine to animate the body's surface map as it rotates.
            Args:
                cmap (str): The :py:mod:`matplotlib` colormap name. Default `plasma`.
                res (int): The resolution of the map in pixels on a side. Default 150.
                frames (int): The number of frames in the animation. Default 50.
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
                            of the image. Default :py:attr:`map.lmax`.
        )pbdoc";

    }

    namespace Body {

        const char* doc = R"pbdoc()pbdoc";

        const char* r = R"pbdoc()pbdoc";

        const char* L = R"pbdoc()pbdoc";

        const char* tref = R"pbdoc()pbdoc";

        const char* prot = R"pbdoc()pbdoc";

        const char* lightcurve = R"pbdoc()pbdoc";

        const char* gradient = R"pbdoc()pbdoc";

        // TODO

    }

    namespace Primary {

        const char* doc = R"pbdoc()pbdoc";

        const char* r = R"pbdoc()pbdoc";

        const char* L = R"pbdoc()pbdoc";

        const char* r_m = R"pbdoc()pbdoc";

        // TODO

    }

    namespace Secondary {

        const char* doc = R"pbdoc()pbdoc";

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

        const char* doc = R"pbdoc()pbdoc";

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
