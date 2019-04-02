/**
\file docstrings.h
\brief Docstrings for the python interface.

*/

#ifndef _STARRY_DOCS_H_
#define _STARRY_DOCS_H_
#include <stdlib.h>

namespace docstrings {

using namespace std;

namespace starry {

    const char* doc = R"pbdoc(
        A code to compute analytic occultation light curves in C++,
        with a sleek Python interface.
    )pbdoc";

}

namespace Map {

    const char* doc = R"pbdoc(
        Instantiate a :py:mod:`starry` surface map. The map is described
        as an expansion in spherical harmonics, with optional arbitrary
        order limb darkening. Users can set the spherical harmonic
        coefficients by direct assignment to the :py:obj:`l, m` index of
        the map instance. If :py:obj:`map` is an instance of this class,

        .. code-block:: python

            map[1, 0] = 0.5

        sets the coefficient of the :math:`Y_{1,0}` harmonic to
        :math:`\frac{1}{2}`. Users can set limb darkening coefficients
        by direct assignment to the :py:obj:`l` index of the map:

        .. code-block:: python

            map[1] = 0.4

        sets the first order limb darkening coefficient :math:`u_1` to
        :math:`0.4`. Note that the zeroth-order limb darkening coefficient
        cannot be set, as it enforces the normalization of the map.

        By default, map calculations are performed using double (64 bit)
        precision. Users may specify :code:`multi=True` to perform
        calculations using multi-precision (usually 32 digits of precision,
        roughly corresponding to 128 bits, unless the compiler macro 
        :py:obj:`STARRY_NMULTI` was set).

        Also by default, surface map instances have scalar coefficients.
        :py:mod:`starry` also supports vectorized coefficients, which are
        convenient for maps whose intensity is a function of wavelength.
        Users can enable spectral maps by setting the :py:obj:`nw` keyword 
        argument, corresponding to the number of wavelength bins.
        If this argument is set to a value greater than one, light curves 
        will be computed for each wavelength bin of the map. All coefficients 
        at a given value of :py:obj:`l, m` and all limb darkening coefficients 
        at a given value of :py:obj:`l` are now *vectors*, with one value per 
        wavelength bin. Functions like :py:meth:`flux()` and 
        :py:meth:`__call__()` will return one value per wavelength bin, and 
        gradients will be computed at every wavelength.

        Finally, :py:mod:`starry` also supports time-dependent maps, which
        can be used to model stellar variability, cloud variability, or any
        process that causes the map to look different at different points in
        time. Users can enable this feature by setting the :py:obj:`nt` keyword 
        argument, corresponding to the number of temporal bins. The *nth*
        temporal bin corresponds to the *nth* derivative of the map with 
        respect to time. The specific intensity at time :math:`t` is therefore
        just the Taylor expansion of the map coefficient matrix. Note, 
        importantly, that temporal variability is only enabled for
        the spherical harmonic coefficients, and *not* the limb darkening
        coefficients.

        .. note:: Map instances are normalized such that the
            **average disk-integrated intensity is equal to the coefficient
            of the** :math:`Y_{0,0}` **term**, which defaults to unity. The
            total luminosity over all :math:`4\pi` steradians is therefore
            four times the :math:`Y_{0,0}` coefficient. This normalization
            is particularly convenient for constant or purely limb-darkened
            maps, whose disk-integrated intensity is always equal to unity.

        .. note:: The total degree of the map can never
            exceed :py:obj:`lmax`. For example, if :py:obj:`lmax=5` and
            you set spherical harmonic coefficients up to :py:obj:`lmax=3`,
            you may only set limb darkening coefficients up to second
            order.

        Args:
            lmax (int): Largest spherical harmonic degree in the surface map. Default 2.
            nw (int): Number of map wavelength bins. Default 1.
            nt (int): Number of map temporal bins. Default 1.
            multi (bool): Use multi-precision to perform all \
                calculations? Default :py:obj:`False`. If :py:obj:`True`, \
                defaults to 32-digit (approximately 128-bit) floating \
                point precision. This can be adjusted by changing the \
                :py:obj:`STARRY_NMULTI` compiler macro.

        .. automethod:: __call__(t=0, theta=0, x=0, y=0)
        .. automethod:: __getitem__(index, value)
        .. automethod:: __setitem__(index, value)
        .. automethod:: add_spot(amp=1, sigma=0.1, lat=0, lon=0, lmax=-1)
        .. automethod:: flux(t=0, theta=0, xo=0, yo=0, ro=0, gradient=False)
        .. automethod:: linear_model(theta=0, xo=0, yo=0, ro=0, gradient=False)
        .. automethod:: load_image(image, lmax=-1, col=-1, normalize=True, sampling_factor=8)
        .. automethod:: random(power, seed=None, col=-1)
        .. automethod:: render(t=0, theta=0, res=300)
        .. automethod:: reset()
        .. automethod:: rotate(theta=0)
        .. automethod:: show(t=0, theta=0, cmap='plasma', res=300, interval=75, gif='')
        .. autoattribute:: __compile_flags__
        .. autoattribute:: axis
        .. autoattribute:: inc
        .. autoattribute:: obl
        .. autoattribute:: lmax
        .. autoattribute:: multi
        .. autoattribute:: N
        .. autoattribute:: ncolu
        .. autoattribute:: ncoly
        .. autoattribute:: nt
        .. autoattribute:: nw
        .. autoattribute:: u
        .. autoattribute:: y
    )pbdoc";

    const char* reset = R"pbdoc(
        Set all of the map coefficients to zero, except for :math:`Y_{0,0}`,
        which is set to unity.
    )pbdoc";

    const char* lmax = R"pbdoc(
        The highest spherical harmonic degree of the map. *Read-only.*
    )pbdoc";

    const char* N = R"pbdoc(
        The number of map coefficients, equal to :math:`(l + 1)^2`.
        *Read-only.*
    )pbdoc";

    const char* nw = R"pbdoc(
        The number of wavelength bins. *Read-only.*
    )pbdoc";

    const char* nt = R"pbdoc(
        The number of temporal bins. *Read-only.*
    )pbdoc";

    const char* ncoly = R"pbdoc(
        The number of columns in the spherical harmonic coefficient matrix. *Read-only.*
    )pbdoc";

    const char* ncolu = R"pbdoc(
        The number of columns in the limb darkening coefficient matrix. *Read-only.*
    )pbdoc";

    const char* multi = R"pbdoc(
        Are calculations done using multi-precision? *Read-only.*
    )pbdoc";

    const char* y = R"pbdoc(
        The spherical harmonic map coefficients. In the default case, 
        this is a vector of the coefficients of the spherical harmonics
        :math:`\{Y_{0,0}, Y_{1,-1}, Y_{1,0}, Y_{1,1}, ...\}`.
        For spectral maps, this is a *matrix*, where each column
        is the spherical harmonic vector at a particular wavelength.
        For temporal maps, this is also a matrix, where the column of index
        :math:`n` corresponds to the *nth* order derivative of the map
        with respect to time.
        *Read-only.*
    )pbdoc";

    const char* u = R"pbdoc(
        The limb darkening map coefficients. In the default and temporal cases, 
        this is a vector of the limb darkening coefficients 
        :math:`\{u_1, u_2, u_3, ...\}`. For spectral maps, this is a *matrix*, 
        where each column is the limb darkening vector at a particular 
        wavelength. *Read-only.*
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

    const char* compile_flags = R"pbdoc(
        A dictionary of flags set at compile time.
    )pbdoc";

    const char* setitem = R"pbdoc(
        Set a spherical harmonic or limb darkening coefficient or
        array of coefficients. Users may set these coefficients
        multiple different ways. For example:

        .. code-block:: python

            map[3, 1] = 0.5         # Set the Y_{3,1} coefficient to a scalar
            map[3, 1] = [0.5, 0.6]  # Set it to a vector (:code:`nw = 2` or :code:`nt = 2`)
            map[3, :] = 0.5         # Set all Y_{3,m} coefficients to 0.5
            map[:, 1] = 0.5         # Set all Y_{l,1} coefficients to 0.5
            map[:, :] = [...]       # Set all map coefficients at once

        .. code-block:: python

            map[1] = 0.5            # Set the u_1 limb darkening coefficient
            map[1] = [0.5, 0.6]     # Set it to a vector (:code:`nw = 2` or :code:`nt = 2`)
            map[:] = [...]          # Set all limb darkening coefficients
    )pbdoc";

    const char* getitem = R"pbdoc(
        Retrieve a spherical harmonic or limb darkening coefficient or
        array of coefficients. Indexing is the same as in the :py:meth:`__setitem__`
        method above.

        Returns:
            A spherical harmonic or limb darkening coefficient, or an array \
            of coefficients.
    )pbdoc";

    const char* call = R"pbdoc(
        Return the specific intensity at a point :py:obj:`(x, y)` on the
        map. Users may optionally provide a rotation angle and/or a time
        (in the case of a temporal map). Note that applying a rotation
        here does not rotate the base map.

        Args:
            t (float or ndarray): The time at which to evaluate the map \
                (temporal maps only). Default 0.
            theta (float or ndarray): Angle of rotation in degrees. \
                Default 0.
            x (float or ndarray): Position scalar or vector.
            y (float or ndarray): Position scalar or vector.

        Returns:
            The specific intensity at :py:obj:`(x, y)`.
    )pbdoc";

    const char* add_spot = R"pbdoc(
        Add the spherical harmonic expansion of a gaussian to the current
        map. This routine adds a gaussian-like feature to the surface map
        by computing the spherical harmonic expansion of a 3D gaussian
        constrained to the surface of the sphere. This is useful for, say,
        modeling star spots or other discrete, localized features on a
        body's surface.

        Args:
            amp (float or ndarray): The amplitude. Default 1.0, resulting \
                in a gaussian whose integral over the sphere is unity. For \
                spectral and temporal maps, this should be a vector \
                corresponding to the amplitude in each map column.
            sigma (float): The standard deviation of the gaussian. \
                Default 0.1
            lat (float): The latitude of the center of the gaussian \
                in degrees. Default 0.
            lon (float): The longitude of the center of the gaussian \
                in degrees. Default 0.
        )pbdoc";

    const char* flux = R"pbdoc(
        Return the total flux received by the observer from the map.
        Computes the total flux received by the observer
        during or outside of an occultation.

        Args:
            t (float or ndarray): Time (temporal maps only). Default 0.
            theta (float or ndarray): Angle of rotation. Default 0.
            xo (float or ndarray): The :py:obj:`x` position of the \
                occultor (if any). Default 0.
            yo (float or ndarray): The :py:obj:`y` position of the \
                occultor (if any). Default 0.
            ro (float): The radius of the occultor in units of this \
                body's radius. Default 0 (no occultation).
            gradient (bool): Compute and return the gradient of the \
                flux as well? Default :py:obj:`False`.

        Returns:
            The flux received by the observer (a scalar or a vector). \
            If :py:obj:`gradient` is :py:obj:`True`, \
            returns the tuple :py:obj:`(F, dF)`, where :py:obj:`F` is \
            the flux and :py:obj:`dF` is \
            a dictionary containing the derivatives with respect to \
            each of the input parameters \
            and each of the map coefficients.
    )pbdoc";

    const char* linear_model = R"pbdoc(
        Return the `starry` linear model.

        ..note:: This routine is currently only implemented for \
            `Default`, `double`-precision maps.
            
        Args:
            theta (float or ndarray): Angle of rotation. Default 0.
            xo (float or ndarray): The :py:obj:`x` position of the \
                occultor (if any). Default 0.
            yo (float or ndarray): The :py:obj:`y` position of the \
                occultor (if any). Default 0.
            ro (float): The radius of the occultor in units of this \
                body's radius. Default 0 (no occultation).
            gradient (bool): Compute and return the gradient of the \
                modeel as well? Default :py:obj:`False`.

        Returns:
            A matrix `M`. When `M` is dotted into a spherical harmonic \
            vector `y`, the result is the light curve predicted by the \
            model.

    )pbdoc";

    const char* rotate = R"pbdoc(
        Rotate the base map an angle :py:obj:`theta` about :py:obj:`axis`.
        This performs a permanent rotation to the base map. Subsequent
        rotations and calculations will be performed relative to this
        rotational state.

        Args:
            theta (float or ndarray): Angle of rotation in degrees. \
                Default 0.
    )pbdoc";

    const char* show = R"pbdoc(
        Display an image of the body's surface map. If either the time 
        :py:obj:`t` (in the case of a temporal map) or the angle of rotation 
        :py:obj:`theta` are vectors, this routine shows an animation of the
        map. If the map is spectral (:py:obj:`nw > 1`), this method displays an
        animated sequence of frames, one per wavelength bin.

        Args:
            t (float or ndarray): Time (temporal maps only). Default 0.
            theta (float or ndarray): Angle of rotation in degrees. Default 0.
            cmap (str): The :py:mod:`matplotlib` colormap name. \
                Default :py:obj:`plasma`.
            res (int): The resolution of the map in pixels on a side. \
                Default 300.
            interval (int): Interval in milliseconds between frames. \
                Default 75.
            gif (str): The name of the `.gif` file to save the animation \
                to. If set, does not show the animation. \
                Default :py:obj:`None`.
    )pbdoc";

    const char* load_image = R"pbdoc(
        Load an array or an image from file.
        This routine loads an image file, computes its spherical harmonic
        expansion up to degree :py:attr:`lmax`, and sets the map vector.

        Args:
            image (str): The full path to the image file.
            lmax (int): The maximum degree of the spherical harmonic \
                expansion of the image. Default :py:attr:`lmax`.
            col (int): For spectral or temporal maps, indicates the bin \
                into which the image will be loaded. Default -1 (loads the \
                image into *all* bins).
            normalize (bool): Normalize the image so that the coefficient of \
                the :math:`Y_{0,0}` harmonic is unity? Default :py:obj:`True`.
            sampling_factor (int): Sampling factor used in the conversion of \
                the image to a :py:obj:`healpix` map. Default 8.
    )pbdoc";

    const char* render = R"pbdoc(
        Render the visible map on a square cartesian grid.

        Args:
            t (float): Time (temporal maps only). Default 0.
            theta (float or ndarray): Angle of rotation in degrees. Default 0.
            res (int): The resolution of the map in pixels on a side. \
                Default 300.

        Returns:
            A 2D array (default or temporal maps) or a 3D array (spectral maps)
            corresponding to the intensity of the map evaluated on a 
            :code:`res x res` pixel grid spanning the visible image of the map.
            Points outside the projected disk are set to :py:obj:`NaN`.
    )pbdoc";

    const char* random = R"pbdoc(
        Draw a map from an isotropic distribution with a given power
        spectrum in :math:`l` and set the map coefficients.

        Args:
            power (ndarray): The power at each degree, starting at :code:`l=0`.
            seed (int): Randomizer seed. Default :py:obj:`None`.
            col (int): The map column into which the random map will be placed.\
                Default -1 (in which case the map is replicated into all \
                columns).
    )pbdoc";


}

}

#endif
