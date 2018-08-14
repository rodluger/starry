/**
Docstrings for the Python functions.

*/

#ifndef _STARRY_DOCS_H_
#define _STARRY_DOCS_H_
#include <stdlib.h>
#include "utils.h"

#define STARRY_MODULE_MAIN     0
#define STARRY_MODULE_MULTI    1
#define STARRY_MODULE_SPECTRAL 2

namespace docstrings {

    using namespace std;


    /**
    Python documentation for the `Map` class.

    */
    template <int Module>
    class Map_ {
    public:
        const char * doc;
        const char * reset;
        const char * lmax;
        const char * N;
        const char * nwav;
        const char * y;
        const char * u;
        const char * p;
        const char * g;
        const char * r;
        const char * s;
        const char * axis;
        const char * evaluate;
        const char * flux;
        const char * rotate;
        const char * show;
        const char * animate;
        void add_extras() { doc = R"pbdoc()pbdoc"; };

        Map_() {

            reset = R"pbdoc(
                Set all of the map coefficients to zero.
            )pbdoc";

            lmax = R"pbdoc(
                The highest spherical harmonic order of the map. *Read-only.*
            )pbdoc";

            N = R"pbdoc(
                The number of map coefficients, equal to `(l + 1) ** 2`. *Read-only.*
            )pbdoc";

            nwav = R"pbdoc(
                The number of wavelength bins in `spectral` mode. *Read-only.*
            )pbdoc";

            y = R"pbdoc(
                The spherical harmonic map vector.
            )pbdoc";

            u = R"pbdoc(
                The limb darkening map vector.
            )pbdoc";

            p = R"pbdoc(
                The polynomial map vector. *Read-only.*
            )pbdoc";

            g = R"pbdoc(
                The Green's polynomial map vector. *Read-only.*
            )pbdoc";

            r = R"pbdoc(
                The current solution vector `r`. *Read-only.*
            )pbdoc";

            s = R"pbdoc(
                The current solution vector `s`. *Read-only.*
            )pbdoc";

            axis = R"pbdoc(
                *Normalized* unit vector specifying the body's axis of rotation. Default :math:`\hat{y} = (0, 1, 0)`.
            )pbdoc";

            evaluate = R"pbdoc(
                Return the specific intensity at a point (`x`, `y`) on the map.
                Users may optionally provide a rotation state. Note that this does
                not rotate the base map.

                Args:
                    theta (float or ndarray): Angle of rotation in degrees. Default 0.
                    x (float or ndarray): Position scalar, vector, or matrix.
                    y (float or ndarray): Position scalar, vector, or matrix.
                    gradient (bool): Compute and return the gradient of the intensity as well? Default :py:obj:`False`.

                Returns:
                    The specific intensity at (`x`, `y`). If :py:obj:`gradient` is :py:obj:`True`, \
                    returns the tuple `(I, dI)`, where `I` is the specific intensity and `dI` is \
                    a dictionary containing the derivatives with respect to each of the input parameters \
                    and each of the map coefficients.
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
                    gradient (bool): Compute and return the gradient of the flux as well? Default :py:obj:`False`.

                Returns:
                    The flux received by the observer (a scalar or a vector). \
                    If :py:obj:`gradient` is :py:obj:`True`, \
                    returns the tuple `(F, dF)`, where `F` is the flux and `dF` is \
                    a dictionary containing the derivatives with respect to each of the input parameters \
                    and each of the map coefficients.

            )pbdoc";

            rotate = R"pbdoc(
                Rotate the base map an angle :py:obj:`theta` about :py:obj:`axis`.
                This performs a permanent rotation to the base map. Subsequent
                rotations and calculations will be performed relative to this
                rotational state.

                Args:
                    theta (float or ndarray): Angle of rotation in degrees. Default 0.
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

    /**
    Documentation specific to the `starry.map` class.

    */
    template <>
    void Map_<STARRY_MODULE_MAIN>::add_extras() {

        doc = R"pbdoc(
                Instantiate a :py:mod:`starry` surface map. Maps instantiated in this fashion
                are *orthonormalized*, so the total integrated luminosity of the map is
                :math:`2\sqrt{\pi} Y_{0,0}`.

                Args:
                    lmax (int): Largest spherical harmonic degree in the surface map. Default 2.

                .. automethod:: evaluate(theta=0, x=0, y=0)
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

    };

    /**
    Python documentation for `starry`.

    */
    template <int Module>
    class docs {
    public:

        const char * doc;
        const char * nmulti;
        void add_extras() { doc = R"pbdoc()pbdoc"; };
        Map_<Module> Map;

        docs() : Map() {

            add_extras();

        }
    };

}

#endif
