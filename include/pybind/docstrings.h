/**
\file docstrings.h
\brief Docstrings for the python interface.

*/

#ifndef _STARRY_DOCS_H_
#define _STARRY_DOCS_H_
#include <stdlib.h>

namespace docstrings {

using namespace std;

namespace Map {

    const char* ydeg = R"pbdoc(
        The highest spherical harmonic degree of the map. *Read-only.*
    )pbdoc";

    const char* udeg = R"pbdoc(
        The highest degree of the limb darkening filter. *Read-only.*
    )pbdoc";

    const char* fdeg = R"pbdoc(
        The highest degree of the spherical harmonic filter. *Read-only.*
    )pbdoc";

    const char* N = R"pbdoc(
        The total number of map coefficients after applying the filters
        (if applicable). *Read-only.*
    )pbdoc";

    const char* Ny = R"pbdoc(
        The total number of spherical harmonic coefficients, including
        the :math:`Y_{0,0}` term. *Read-only.*
    )pbdoc";

    const char* Nu = R"pbdoc(
        The total number of limb darkening coefficients, including
        the :math:`u_{0}` term. *Read-only.*
    )pbdoc";

    const char* Nf = R"pbdoc(
        The total number of spherical harmonic filter coefficients, including
        the :math:`Y_{0,0}` term. *Read-only.*
    )pbdoc";

    const char* nt = R"pbdoc(
        The number of temporal bins. *Read-only.*
    )pbdoc";

    const char* nw = R"pbdoc(
        The number of wavelength bins. *Read-only.*
    )pbdoc";

    const char* multi = R"pbdoc(
        Are calculations done using multi-precision? *Read-only.*
    )pbdoc";

    const char* setitem = R"pbdoc(
        Set a spherical harmonic or limb darkening coefficient or
        array of coefficients. Users may set these coefficients
        multiple different ways. For example, for a scalar map:

        .. code-block:: python

            map[3, 1] = 0.5            # Set the Y_{3,1} coefficient to a scalar
            map[3, :] = 0.5            # Set all Y_{3,m} coefficients to 0.5
            map[1:, 1] = 0.5           # Set all Y_{l,1} coefficients to 0.5
            map[1:, :] = [...]         # Set all map coefficients

        .. code-block:: python

            map[1] = 0.5               # Set the u_1 limb darkening coefficient
            map[1:] = [...]            # Set all limb darkening coefficients
        
        For a spectral or temporal map, an extra index must be provided.
        For example:

        .. code-block:: python

            map[3, 1, 0] = 0.5         # Set the Y_{3,1,0} coefficient to a scalar
            map[3, 1, :] = 0.5         # Set all Y_{3,1} coefficients to 0.5
            map[1:, 1, :] = 0.5        # Set all Y_{l,1} coefficients to 0.5
            map[1:, :, :] = [...]      # Set all map coefficients

        .. code-block:: python

            map[1, 0] = 0.5            # Set the u_{1,0} limb darkening coefficient
            map[1:, :] = [...]         # Set all limb darkening coefficients

    )pbdoc";

    const char* getitem = R"pbdoc(
        Retrieve a spherical harmonic or limb darkening coefficient or
        array of coefficients. Indexing is the same as in the :py:meth:`__setitem__`
        method.

        Returns:
            A spherical harmonic or limb darkening coefficient, or an array \
            of coefficients.
    )pbdoc";

    const char* reset = R"pbdoc(
        Reset all of the spherical harmonic, limb darkening, and filter
        coefficients. Also reset the rotation axis.
    )pbdoc";

    const char* y = R"pbdoc(
        The spherical harmonic map coefficients. For scalar maps, 
        this is a vector of the coefficients of the spherical harmonics
        :math:`\{Y_{0,0}, Y_{1,-1}, Y_{1,0}, Y_{1,1}, ...\}`.
        For spectral maps, this is a *matrix*, where each column
        is the spherical harmonic vector at a particular wavelength.
        For temporal maps, this is also a *flattened* matrix, where the 
        vectors corresponding to each temporal component are stacked
        on top of each other. *Read-only.*
    )pbdoc";

    const char* u = R"pbdoc(
        The limb darkening map coefficients. For scalar maps, 
        this is a vector of the limb darkening coefficients 
        :math:`\{u_1, u_2, u_3, ...\}`. For spectral maps, this is a *matrix*, 
        where each column is the limb darkening vector at a particular 
        wavelength. *Read-only.*
    )pbdoc";

    const char* f = R"pbdoc(
        The spherical harmonic filter coefficients.
        This is a vector of the coefficients of the spherical harmonics
        :math:`\{Y_{0,0}, Y_{1,-1}, Y_{1,0}, Y_{1,1}, ...\}`. Whenever
        intensities and fluxes are computed, the filter is applied 
        multiplicatively to the map. Filters are the extension of limb darkening
        to non-radially symmetric modifications to the surface intensity.
        *Read-only.*
    )pbdoc";

    const char* axis = R"pbdoc(
        A *normalized* unit vector specifying the default axis of
        rotation for the map. Default :math:`\hat{y} = (0, 1, 0)`.
    )pbdoc";

    const char* inc = R"pbdoc(
        The inclination of the map in degrees. 
        Setting this value overrides :py:attr:`axis`. Default :math:`90^\circ`.
    )pbdoc";

    const char* obl = R"pbdoc(
        The obliquity of the map in degrees. 
        Setting this value overrides :py:attr:`axis`. Default :math:`0^\circ`.
    )pbdoc";

    const char* rotate = R"pbdoc(
        Rotate the base map an angle :py:obj:`theta` about :py:obj:`axis`.
        This performs a permanent rotation to the base map. Subsequent
        rotations and calculations will be performed relative to this
        rotational state.
        
        Arguments:
            theta (float): Angle of rotation in degrees. \
                Default 0.
            axis (ndarray): Axis of rotation. \
                Default is the current map axis.
    )pbdoc";

    const char* add_spot = R"pbdoc(
        Add the spherical harmonic expansion of a gaussian to the current
        map. This routine adds a gaussian-like feature to the surface map
        by computing the spherical harmonic expansion of a 3D gaussian
        constrained to the surface of the sphere. This is useful for, say,
        modeling star spots or other discrete, localized features on a
        body's surface.

        Arguments:
            amp (float or ndarray): The amplitude. Default 1.0, resulting \
                in a gaussian whose integral over the sphere is unity. For \
                spectral and temporal maps, this should be a vector \
                corresponding to the amplitude in each map column.
        
        Keyword Arguments:
            sigma (float): The standard deviation of the gaussian. \
                Default 0.1
            lat (float): The latitude of the center of the gaussian \
                in degrees. Default 0.
            lon (float): The longitude of the center of the gaussian \
                in degrees. Default 0.
        )pbdoc";

    const char* random = R"pbdoc(
        Draw a map from an isotropic distribution with a given power
        spectrum in :math:`l` and set the map coefficients.

        Arguments:
            power (ndarray): The power at each degree, starting at :code:`l=0`.
            
        Keyword Arguments:
            seed (int): Randomizer seed. Default :py:obj:`None`.
            col (int): The map column into which the random map will be placed.\
                Default :py:obj:`None` (in which case the map is replicated into all \
                columns). *Spectral / temporal maps only.*

    )pbdoc";

    const char* compile_flags = R"pbdoc(
        A dictionary of informational flags about the ``Map`` set at compile time.
    )pbdoc";

}

}

#endif
