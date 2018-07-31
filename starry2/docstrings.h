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
        const char * reset;
        const char * lmax;
        const char * y;
        const char * p;
        const char * g;
        const char * r;
        const char * axis;
        const char * evaluate;
        const char * rotate;
        void add_extras() { doc = R"pbdoc()pbdoc"; };

        Map_(){

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

            r = R"pbdoc(
                The current solution vector `r`. *Read-only.*
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

                Returns:
                    The specific intensity at (`x`, `y`).
            )pbdoc";

            rotate = R"pbdoc(
                Rotate the base map an angle :py:obj:`theta` about :py:obj:`axis`.
                This performs a permanent rotation to the base map. Subsequent
                rotations and calculations will be performed relative to this
                rotational state.

                Args:
                    theta (float or ndarray): Angle of rotation in degrees. Default 0.
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
                .. automethod:: reset()
                .. autoattribute:: lmax
                .. autoattribute:: y
                .. autoattribute:: p
                .. autoattribute:: g
                .. autoattribute:: r

            )pbdoc";

    };

    template <class T>
    class docs {
    public:

        const char * doc;
        const char * nmulti;
        void add_extras() { doc = R"pbdoc()pbdoc"; };
        Map_<T> Map;

        docs() : Map() {

            add_extras();

        }
    };

}

#endif
