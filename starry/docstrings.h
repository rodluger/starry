/**
Docstrings for the Python functions.

*/

#ifndef _STARRY_DOCS_H_
#define _STARRY_DOCS_H_

#include <stdlib.h>
using namespace std;

namespace docstrings {

    namespace map {

        const char * map =
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

        } // namespace map

} // namespace docs

namespace docstrings_grad {

    namespace map {

        const char * map =
        R"pbdoc(
                Instantiate a :py:mod:`starry` surface map.

                Args:
                    lmax (int): Largest spherical harmonic degree in the surface map. Default 2.

                .. autoattribute:: optimize
                .. automethod:: evaluate(axis=(0, 1, 0), theta=0, x=0, y=0)
                .. automethod:: rotate(axis=(0, 1, 0), theta=0)
                .. automethod:: flux_numerical(axis=(0, 1, 0), theta=0, xo=0, yo=0, ro=0, tol=1.e-4)
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

        const char * optimize = docstrings::map::optimize;

        const char * evaluate = docstrings::map::evaluate;

        const char * flux = docstrings::map::flux;

    } // namespace map

} // namespace docs_grad

#endif
