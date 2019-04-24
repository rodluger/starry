/**
\file docstrings.h
\brief Docstrings for the extensions module.

*/

#ifndef _STARRY_DOCS_EXT_H_
#define _STARRY_DOCS_EXT_H_
#include <stdlib.h>

namespace docstrings {

using namespace std;

namespace extensions {

    const char* RAxisAngle = R"pbdoc(
        Return the 3x3 Cartesian rotation matrix for rotation
        through an angle :py:obj:`theta` about the unit vector :py:obj:`axis`.

        Args:
            axis (ndarray): A Cartesian unit vector.
            angle (scalar): The angle of rotation in degrees.
    )pbdoc";

}
}

#endif