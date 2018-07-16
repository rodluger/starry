/**
Docstrings for the Python functions.

*/

#ifndef _STARRY_DOCS_H_
#define _STARRY_DOCS_H_
#include <stdlib.h>

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
                    axis (ndarray): *Normalized* unit vector specifying the body's axis of rotation. Default :math:`\hat{y} = (0, 1, 0)`.
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
                    axis (ndarray): *Normalized* unit vector specifying the body's axis of rotation. Default :math:`\hat{y} = (0, 1, 0)`.
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
                    axis (ndarray): *Normalized* unit vector specifying the body's axis of rotation. Default :math:`\hat{y} = (0, 1, 0)`.
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
                    axis (ndarray): *Normalized* unit vector specifying the axis of rotation. Default :math:`\hat{y} = (0, 1, 0)`.
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
                .. automethod:: psd()
                .. automethod:: add_gaussian()
                .. automethod:: load_array(image)
                .. automethod:: load_image(image)
                .. automethod:: load_healpix(image)
                .. automethod:: show(cmap='plasma', res=300)
                .. automethod:: animate(axis=(0, 1, 0), cmap='plasma', res=150, frames=50)
            )pbdoc";

        flux_numerical = R"pbdoc(
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

    };

    template <>
    void Map_<Multi>::add_extras() {

        doc = R"pbdoc(
                Instantiate a :py:mod:`starry` surface map. Maps instantiated in this fashion
                are *orthonormalized*, so the total integrated luminosity of the map is
                :math:`2\sqrt{\pi} Y_{0,0}`.

                Args:
                    lmax (int): Largest spherical harmonic degree in the surface map. Default 2.

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
                .. automethod:: psd()
                .. automethod:: add_gaussian()
                .. automethod:: load_array(image)
                .. automethod:: load_image(image)
                .. automethod:: load_healpix(image)
                .. automethod:: show(cmap='plasma', res=300)
                .. automethod:: animate(axis=(0, 1, 0), cmap='plasma', res=150, frames=50)
            )pbdoc";

    };

    template <>
    void Map_<Grad>::add_extras() {

        doc = R"pbdoc(
                Instantiate a :py:mod:`starry` surface map. Maps instantiated in this fashion
                are *orthonormalized*, so the total integrated luminosity of the map is
                :math:`2\sqrt{\pi} Y_{0,0}`.

                Args:
                    lmax (int): Largest spherical harmonic degree in the surface map. Default 2.

                .. automethod:: evaluate(axis=(0, 1, 0), theta=0, x=0, y=0)
                .. automethod:: rotate(axis=(0, 1, 0), theta=0)
                .. automethod:: flux(axis=(0, 1, 0), theta=0, xo=0, yo=0, ro=0)
                .. automethod:: get_coeff(l, m)
                .. automethod:: set_coeff(l, m, coeff)
                .. automethod:: reset()
                .. autoattribute:: gradient
                .. autoattribute:: lmax
                .. autoattribute:: y
                .. autoattribute:: p
                .. autoattribute:: g
                .. autoattribute:: s
                .. autoattribute:: r
                .. automethod:: psd()
                .. automethod:: load_array(image)
                .. automethod:: load_image(image)
                .. automethod:: load_healpix(image)
                .. automethod:: show(cmap='plasma', res=300)
                .. automethod:: animate(axis=(0, 1, 0), cmap='plasma', res=150, frames=50)
        )pbdoc";

        gradient = R"pbdoc(
            A dictionary of derivatives for all model parameters, populated on
            calls to :py:meth:`flux` and :py:meth:`evaluate`.
        )pbdoc";

    };

    template <class T>
    class LimbDarkenedMap_ {
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
        const char * u;
        const char * reparam;
        const char * evaluate;
        const char * flux;
        const char * flux_numerical;
        const char * psd;
        const char * mono;
        const char * show;
        const char * gradient;
        void add_extras() {};

        LimbDarkenedMap_(){

            get_coeff = R"pbdoc(
                Return the limb darkening coefficient of order :py:obj:`l`.

                .. note:: Users can also retrieve a coefficient by accessing the \
                          [:py:obj:`l`] index of the map as if it were an array. \
                          Single slice indexing is also allowed.

                Args:
                    l (int): The limb darkening order (> 0).
            )pbdoc";

            set_coeff = R"pbdoc(
                Set the limb darkening coefficient of order :py:obj:`l`.

                .. note:: Users can also set a coefficient by setting the \
                          [:py:obj:`l`] index of the map as if it \
                          were an array. Single slice indexing is also allowed.

                Args:
                    l (int): The limb darkening order (> 0).
                    coeff (float): The value of the coefficient.
            )pbdoc";

            psd = R"pbdoc(
                Check whether the map is positive semi-definite. Returns :py:obj:`True`
                if it is positive semi-definite and :py:obj:`False` otherwise. This routine
                uses Sturm's theorem to count the number of roots of the
                specific intensity polynomial.
            )pbdoc";

            mono = R"pbdoc(
                Check whether the map is monotonically decreasing toward the limb. Returns :py:obj:`True`
                if it this is the case and :py:obj:`False` otherwise. This routine
                uses Sturm's theorem to count the number of roots of the derivative of the
                specific intensity polynomial.
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

            u = R"pbdoc(
                The limb darkening coefficient vector. *Read-only.*
            )pbdoc";

            reparam = R"pbdoc(
                Set to :py:obj:`False` to disable reparametrization of the primitive integrals when \
                computing occultation light curves for large occultors. This is in general not something you should do! \
                Default :py:obj:`True`.
            )pbdoc";

            evaluate = R"pbdoc(
                Return the specific intensity at a point (`x`, `y`) on the map.

                Args:
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
                    xo (float or ndarray): The `x` position of the occultor (if any). Default 0.
                    yo (float or ndarray): The `y` position of the occultor (if any). Default 0.
                    ro (float): The radius of the occultor in units of this body's radius. Default 0 (no occultation).

                Returns:
                    The flux received by the observer (a scalar or a vector).
            )pbdoc";

            show = R"pbdoc(
                Convenience routine to quickly display the body's surface map.

                Args:
                    cmap (str): The :py:mod:`matplotlib` colormap name. Default `plasma`.
                    res (int): The resolution of the map in pixels on a side. Default 300.
            )pbdoc";

            add_extras();
        }

    };

    template <>
    void LimbDarkenedMap_<double>::add_extras() {

        doc = R"pbdoc(
                Instantiate a :py:mod:`starry` limb-darkened surface map.
                This differs from the base :py:class:`Map` class in that maps
                instantiated this way are radially symmetric: only the radial (:py:obj:`m = 0`)
                coefficients of the map are available. Users edit the map by directly
                specifying the polynomial limb darkening coefficients :py:obj:`u`, starting
                with :math:`u_1` (linear limb darkening). The coefficient :math:`u_0` is fixed to enforce
                the correct normalization.

                .. warning:: Unlike :py:class:`Map`, maps instantiated this \
                             way are normalized so that the integral of the specific intensity over the \
                             visible disk is unity. This is convenient for using this map to model \
                             stars: the unocculted flux from the star is equal to one, regardless of the limb-darkening \
                             coefficients!

                Args:
                    lmax (int): Largest spherical harmonic degree in the surface map. Default 2.

                .. automethod:: evaluate(x=0, y=0)
                .. automethod:: flux(xo=0, yo=0, ro=0)
                .. automethod:: get_coeff(l)
                .. automethod:: set_coeff(l, coeff)
                .. automethod:: reset()
                .. automethod:: psd()
                .. automethod:: mono()
                .. autoattribute:: lmax
                .. autoattribute:: y
                .. autoattribute:: p
                .. autoattribute:: g
                .. autoattribute:: u
                .. autoattribute:: s
                .. automethod:: show(cmap='plasma', res=300)
        )pbdoc";

        flux_numerical = R"pbdoc(
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

    };

    template <>
    void LimbDarkenedMap_<Multi>::add_extras() {

        doc = R"pbdoc(
                Instantiate a :py:mod:`starry` limb-darkened surface map.
                This differs from the base :py:class:`Map` class in that maps
                instantiated this way are radially symmetric: only the radial (:py:obj:`m = 0`)
                coefficients of the map are available. Users edit the map by directly
                specifying the polynomial limb darkening coefficients :py:obj:`u`, starting
                with :math:`u_1` (linear limb darkening). The coefficient :math:`u_0` is fixed to enforce
                the correct normalization.

                .. warning:: Unlike :py:class:`Map`, maps instantiated this \
                             way are normalized so that the integral of the specific intensity over the \
                             visible disk is unity. This is convenient for using this map to model \
                             stars: the unocculted flux from the star is equal to one, regardless of the limb-darkening \
                             coefficients!

                Args:
                    lmax (int): Largest spherical harmonic degree in the surface map. Default 2.

                .. automethod:: evaluate(x=0, y=0)
                .. automethod:: flux(xo=0, yo=0, ro=0)
                .. automethod:: get_coeff(l)
                .. automethod:: set_coeff(l, coeff)
                .. automethod:: reset()
                .. automethod:: psd()
                .. automethod:: mono()
                .. autoattribute:: lmax
                .. autoattribute:: y
                .. autoattribute:: p
                .. autoattribute:: g
                .. autoattribute:: u
                .. autoattribute:: s
                .. automethod:: show(cmap='plasma', res=300)
        )pbdoc";

    };

    template <>
    void LimbDarkenedMap_<Grad>::add_extras() {

        doc = R"pbdoc(
                Instantiate a :py:mod:`starry` limb-darkened surface map.
                This differs from the base :py:class:`Map` class in that maps
                instantiated this way are radially symmetric: only the radial (:py:obj:`m = 0`)
                coefficients of the map are available. Users edit the map by directly
                specifying the polynomial limb darkening coefficients :py:obj:`u`, starting
                with :math:`u_1` (linear limb darkening). The coefficient :math:`u_0` is fixed to enforce
                the correct normalization.

                .. warning:: Unlike :py:class:`Map`, maps instantiated this \
                             way are normalized so that the integral of the specific intensity over the \
                             visible disk is unity. This is convenient for using this map to model \
                             stars: the unocculted flux from the star is equal to one, regardless of the limb-darkening \
                             coefficients!

                Args:
                    lmax (int): Largest spherical harmonic degree in the surface map. Default 2.

                .. automethod:: evaluate(x=0, y=0)
                .. automethod:: flux(xo=0, yo=0, ro=0)
                .. automethod:: get_coeff(l)
                .. automethod:: set_coeff(l, coeff)
                .. automethod:: reset()
                .. automethod:: psd()
                .. automethod:: mono()
                .. autoattribute:: gradient
                .. autoattribute:: lmax
                .. autoattribute:: y
                .. autoattribute:: p
                .. autoattribute:: g
                .. autoattribute:: u
                .. autoattribute:: s
                .. automethod:: show(cmap='plasma', res=300)
        )pbdoc";

        gradient = R"pbdoc(
            A dictionary of derivatives for all model parameters, populated on
            calls to :py:meth:`flux` and :py:meth:`evaluate`.
        )pbdoc";

    };

    template <class T>
    class System_ {
    public:
        const char * doc;
        const char * compute;
        const char * flux;
        const char * scale;
        const char * exposure_time;
        const char * exposure_tol;
        const char * exposure_max_depth;
        const char * gradient;
        void add_extras();

        System_(){

            compute = R"pbdoc(
                Compute the system light curve analytically.
                Compute the full system light curve at the times
                given by the :py:obj:`time <>` array and store the result
                in :py:attr:`flux`. The light curve for each body in the
                system is stored in the body's :py:attr:`flux` attribute.

                Args:
                    time (ndarray): Time array, measured in days.
            )pbdoc";

            flux = R"pbdoc(
                The computed system light curve. Must run :py:meth:`compute` first. *Read-only*.
            )pbdoc";

            scale = R"pbdoc(
                This parameter sets the lengthscale for computing the light travel time delay
                and is simply equal to the radius of the star in :math:`R_\odot`. If zero, the light
                travel time delay is not computed, corresponding to an effectively infinite
                speed of light.
            )pbdoc";

            exposure_time = R"pbdoc(
                The exposure time of the observations in days. If nonzero, the flux will
                be integrated over this exposure time.
            )pbdoc";

            exposure_tol = R"pbdoc(
                Tolerance of the recursive method for integrating the flux over the exposure time.
            )pbdoc";

            exposure_max_depth = R"pbdoc(
                Maximum recursion depth for the exposure calculation.
            )pbdoc";

            add_extras();

        }

    };

    template <typename T>
    void System_<T>::add_extras() {

        doc = R"pbdoc(
                Instantiate an orbital system.

                Args:
                    bodies (list): List of bodies in the system, with the primary (usually the star) listed first.
                    scale (float): This parameter sets the lengthscale for computing the light travel time delay \
                                   and is simply equal to the radius of the star in :math:`R_\odot`. Default `0`, meaning \
                                   the light travel time effect is not computed.
                    exposure_time (float): The exposure time of the observations in days. If nonzero, the flux will \
                                           be integrated over this exposure time. Note that setting this will result \
                                           in slower run times, since the integrated flux is computed numerically. Default `0`.
                    exposure_tol (float): Tolerance of the recursive method for integrating the flux over the exposure time. Default `1e-8`.
                    exposure_maxdepth (int): Maximum recursion depth for the exposure calculation. Default `4`.

                .. automethod:: compute(time)
                .. autoattribute:: flux
                .. autoattribute:: exposure_time
                .. autoattribute:: exposure_tol
                .. autoattribute:: exposure_max_depth
        )pbdoc";

    };

    template <>
    void System_<Grad>::add_extras() {

        doc = R"pbdoc(
                Instantiate an orbital system.

                Args:
                    bodies (list): List of bodies in the system, with the primary (usually the star) listed first.
                    scale (float): This parameter sets the lengthscale for computing the light travel time delay \
                                   and is simply equal to the radius of the star in :math:`R_\odot`. Default `0`, meaning \
                                   the light travel time effect is not computed.
                    exposure_time (float): The exposure time of the observations in days. If nonzero, the flux will \
                                           be integrated over this exposure time. Default `0`.
                    exposure_tol (float): Tolerance of the recursive method for integrating the flux over the exposure time. Default `1e-8`.
                    exposure_maxdepth (int): Maximum recursion depth for the exposure calculation. Default `4`.

                .. automethod:: compute(time)
                .. autoattribute:: flux
                .. autoattribute:: gradient
                .. autoattribute:: scale
                .. autoattribute:: exposure_time
                .. autoattribute:: exposure_tol
                .. autoattribute:: exposure_max_depth
        )pbdoc";

        gradient = R"pbdoc(
            A dictionary of derivatives of the system flux with respect to
            all model parameters, populated on calls to :py:meth:`compute`.

            .. note:: This dictionary is similar to the :py:obj:`gradient` \
                      attribute of a :py:obj:`Map` instance, but the keys in \
                      the dictionary are prepended by either `star.` (for \
                      the star) or `planetX` (where `X` is the planet number,
                      starting with 1). For instance, the gradient of the \
                      system flux with respect to the second planet's eccentricity \
                      is :py:obj:`gradient['planet2.ecc']`.
        )pbdoc";

    };

    template <class T>
    class Body_ {
    public:
        const char * map;
        const char * flux;
        const char * x;
        const char * y;
        const char * z;
        const char * r;
        const char * L;
        const char * axis;
        const char * prot;
        const char * a;
        const char * porb;
        const char * inc;
        const char * ecc;
        const char * w;
        const char * Omega;
        const char * lambda0;
        const char * tref;
        const char * gradient;
        void add_extras();

        Body_(){

            map = R"pbdoc(
                The body's surface map.
            )pbdoc";

            flux = R"pbdoc(
                The body's computed light curve. *Read-only*.
            )pbdoc";

            x = R"pbdoc(
                The `x` position of the body in stellar radii. *Read-only*.
            )pbdoc";

            y = R"pbdoc(
                The `y` position of the body in stellar radii. *Read-only*.
            )pbdoc";

            z = R"pbdoc(
                The `z` position of the body in stellar radii. *Read-only*.
            )pbdoc";

            r = R"pbdoc(
                Body radius in units of stellar radius.
            )pbdoc";

            L = R"pbdoc(
                Body luminosity in units of stellar luminosity.
            )pbdoc";

            axis = R"pbdoc(
                *Normalized* unit vector specifying the body's axis of rotation.
            )pbdoc";

            prot = R"pbdoc(
                Rotation period in days.
            )pbdoc";

            a = R"pbdoc(
                Body semi-major axis in units of stellar radius.
            )pbdoc";

            porb = R"pbdoc(
                Orbital period in days.
            )pbdoc";

            inc = R"pbdoc(
                Orbital inclination in degrees.
            )pbdoc";

            ecc = R"pbdoc(
                Orbital eccentricity.
            )pbdoc";

            w = R"pbdoc(
                Longitude of pericenter in degrees. This is usually denoted :math:`\varpi`.
                See the `Wikipedia <https://en.wikipedia.org/wiki/Longitude_of_the_periapsis>`_ entry.
            )pbdoc";

            Omega = R"pbdoc(
                Longitude of ascending node in degrees.
            )pbdoc";

            lambda0 = R"pbdoc(
                Mean longitude at time :py:obj:`tref` in degrees.
            )pbdoc";

            tref = R"pbdoc(
                Reference time in days.
            )pbdoc";

            add_extras();

        }

    };

    template <typename T>
    void Body_<T>::add_extras() {

    };

    template <>
    void Body_<Grad>::add_extras() {

        gradient = R"pbdoc(
            A dictionary of derivatives of the body's flux with respect to
            all model parameters, populated on calls to :py:meth:`System.compute`.

            .. note:: This dictionary is similar to the :py:obj:`gradient` \
                      attribute of a :py:obj:`Map` instance, but the keys in \
                      the dictionary are prepended by either `star.` (for \
                      the star) or `planetX` (where `X` is the planet number,
                      starting with 1). For instance, the gradient of this body's \
                      flux with respect to the second planet's eccentricity \
                      is :py:obj:`gradient['planet2.ecc']`.
        )pbdoc";

    };

    template <class T>
    class Star_ {
    public:
        const char * doc;
        const char * map;
        const char * r;
        const char * L;
        void add_extras();

        Star_(){

            map = R"pbdoc(
                The star's surface map, a :py:class:`LimbDarkenedMap` instance.
            )pbdoc";

            r = R"pbdoc(
                The star's radius, fixed to unity. *Read-only.*
            )pbdoc";

            L = R"pbdoc(
                The star's luminosity, fixed to unity. *Read-only.*
            )pbdoc";

            add_extras();

        }

    };

    template <typename T>
    void Star_<T>::add_extras() {

        doc = R"pbdoc(
           Instantiate a stellar :py:class:`Body` object.
           The star's radius and luminosity are fixed at unity.

           Args:
               lmax (int): Largest spherical harmonic degree in body's surface map. Default 2.

           .. autoattribute:: map
           .. autoattribute:: flux
           .. autoattribute:: r
           .. autoattribute:: L
        )pbdoc";

    };

    template <>
    void Star_<Grad>::add_extras() {

        doc = R"pbdoc(
            Instantiate a stellar :py:class:`Body` object.
            The star's radius and luminosity are fixed at unity.

            Args:
                lmax (int): Largest spherical harmonic degree in body's surface map. Default 2.

            .. autoattribute:: map
            .. autoattribute:: flux
            .. autoattribute:: gradient
            .. autoattribute:: r
            .. autoattribute:: L
        )pbdoc";

    };

    template <class T>
    class Planet_ {
    public:
        const char * doc;
        void add_extras();

        Planet_(){

            add_extras();

        }

    };

    template <typename T>
    void Planet_<T>::add_extras() {

        doc = R"pbdoc(
            Instantiate a planetary :py:class:`Body` object.
            Instantiate a planet. At present, :py:mod:`starry` computes orbits with a simple
            Keplerian solver, so the planet is assumed to be massless.

            Args:
                lmax (int): Largest spherical harmonic degree in body's surface map. Default 2.
                r (float): Body radius in stellar radii. Default 0.1
                L (float): Body luminosity in units of the stellar luminosity. Default 0.
                axis (ndarray): A *normalized* unit vector specifying the body's axis of rotation. Default :math:`\hat{y} = (0, 1, 0)`.
                prot (float): Rotation period in days. Default no rotation.
                a (float): Semi-major axis in stellar radii. Default 50.
                porb (float): Orbital period in days. Default 1.
                inc (float): Orbital inclination in degrees. Default 90.
                ecc (float): Orbital eccentricity. Default 0.
                w (float): Longitude of pericenter in degrees. Default 90.
                Omega (float): Longitude of ascending node in degrees. Default 0.
                lambda0 (float): Mean longitude at time :py:obj:`tref` in degrees. Default 90.
                tref (float): Reference time in days. Default 0.

            .. autoattribute:: map
            .. autoattribute:: flux
            .. autoattribute:: x
            .. autoattribute:: y
            .. autoattribute:: z
            .. autoattribute:: r
            .. autoattribute:: L
            .. autoattribute:: axis
            .. autoattribute:: prot
            .. autoattribute:: a
            .. autoattribute:: porb
            .. autoattribute:: inc
            .. autoattribute:: ecc
            .. autoattribute:: w
            .. autoattribute:: Omega
            .. autoattribute:: lambda0
            .. autoattribute:: tref
        )pbdoc";

    };

    template <>
    void Planet_<Grad>::add_extras() {

        doc = R"pbdoc(
            Instantiate a planetary :py:class:`Body` object.
            Instantiate a planet. At present, :py:mod:`starry` computes orbits with a simple
            Keplerian solver, so the planet is assumed to be massless.

            Args:
                lmax (int): Largest spherical harmonic degree in body's surface map. Default 2.
                r (float): Body radius in stellar radii. Default 0.1
                L (float): Body luminosity in units of the stellar luminosity. Default 0.
                axis (ndarray): A *normalized* unit vector specifying the body's axis of rotation. Default :math:`\hat{y} = (0, 1, 0)`.
                prot (float): Rotation period in days. Default no rotation.
                a (float): Semi-major axis in stellar radii. Default 50.
                porb (float): Orbital period in days. Default 1.
                inc (float): Orbital inclination in degrees. Default 90.
                ecc (float): Orbital eccentricity. Default 0.
                w (float): Longitude of pericenter in degrees. Default 90.
                Omega (float): Longitude of ascending node in degrees. Default 0.
                lambda0 (float): Mean longitude at time :py:obj:`tref` in degrees. Default 90.
                tref (float): Reference time in days. Default 0.

            .. autoattribute:: map
            .. autoattribute:: flux
            .. autoattribute:: gradient
            .. autoattribute:: x
            .. autoattribute:: y
            .. autoattribute:: z
            .. autoattribute:: r
            .. autoattribute:: L
            .. autoattribute:: axis
            .. autoattribute:: prot
            .. autoattribute:: a
            .. autoattribute:: porb
            .. autoattribute:: inc
            .. autoattribute:: ecc
            .. autoattribute:: w
            .. autoattribute:: Omega
            .. autoattribute:: lambda0
            .. autoattribute:: tref
        )pbdoc";

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
        LimbDarkenedMap_<T> LimbDarkenedMap;
        System_<T> System;
        Body_<T> Body;
        Star_<T> Star;
        Planet_<T> Planet;

        docs() : Map(), LimbDarkenedMap(), System(), Body(), Star(), Planet() {

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
    void docs<Multi>::add_extras() {

        doc = R"pbdoc(
            starry
            ------

            .. contents::
                :local:

            Introduction
            ============

            This page documents the :py:mod:`starry.multi` API, which is coded
            in C++ with a :py:mod:`pybind11` Python interface. This API is
            identical in nearly all respects to the :py:mod:`starry` API, except
            that all internal computations are done using multi-precision floating
            point arithmetic. By default, :py:mod:`starry.multi` performs calculations
            using 32 digits, which roughly corresponds to 128-bit (quadruple) precision.

            .. note:: The :py:obj:`STARRY_NMULTI` compiler flag determines the number of significant \
                      digits to use in multi-precision calculations and can be changed by setting an environment variable \
                      of the same name prior to compiling :py:obj:`starry`.
                      See :doc:`install` for more information.

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

        nmulti = R"pbdoc(
            Number of digits used to perform multi-precision calculations.
            Double precision roughly corresponds to 16, and quadruple
            precision (default) roughly corresponds 32.
            This is a compile-time constant. If you wish to change it, you'll
            have to re-compile :py:obj:`starry`. See :doc:`install` for more information.
        )pbdoc";

    };

    template <>
    void docs<Grad>::add_extras() {

        doc = R"pbdoc(
            starry.grad
            -----------

            .. contents::
                :local:

            Introduction
            ============

            This page documents the :py:mod:`starry.grad` API, which is coded
            in C++ with a :py:mod:`pybind11` Python interface. This API is
            identical in nearly all respects to the :py:mod:`starry` API, except
            that its methods compute gradients with respect to the input parameters,
            in addition to the actual return values. For instance, consider the
            following code block:

            .. code-block:: python

                >>> import starry
                >>> m = starry.Map()
                >>> m[1, 0] = 1
                >>> m.flux(axis=(0, 1, 0), theta=30, xo=0.1, yo=0.1, ro=0.1)
                0.8723336063428014

            Here's the same code executed using the :py:obj:`Map()` class in :py:mod:`starry.grad`:

            .. code-block:: python

                >>> import starry
                >>> m = starry.grad.Map()
                >>> m[1, 0] = 1
                >>> m.flux(axis=(0, 1, 0), theta=30, xo=0.1, yo=0.1, ro=0.1)
                0.8723336063428014

            So far, they look identical. However, in the second case :py:obj:`starry`
            has also computed the gradient of the flux with respect to each of the
            input parameters (including the map coefficients):

            .. code-block:: python

                >>> m.gradient
                {'Y_{0,0}': array([0.]),
                 'Y_{1,-1}': array([-0.00153499]),
                 'Y_{1,0}': array([0.87233361]),
                 'Y_{1,1}': array([-0.5054145]),
                 'Y_{2,-1}': array([0.]),
                 'Y_{2,-2}': array([0.]),
                 'Y_{2,0}': array([0.]),
                 'Y_{2,1}': array([0.]),
                 'Y_{2,2}': array([0.]),
                 'axis_x': array([0.0007675]),
                 'axis_y': array([8.52090655e-20]),
                 'axis_z': array([-0.00020565]),
                 'ro': array([-0.27718567]),
                 'theta': array([-0.00882115]),
                 'xo': array([-0.0063251]),
                 'yo': array([0.00134985])}

            The :py:attr:`gradient` attribute can be accessed like any Python
            dictionary:

            .. code-block:: python

                >>> m.gradient["ro"]
                array([-0.27718567])
                >>> m.gradient["theta"]
                array([-0.00882115])

            In case :py:obj:`flux` is called with vector arguments, :py:attr:`gradient`
            is also vectorized:

            .. code-block:: python

                >>> import starry
                >>> m = starry.grad.Map()
                >>> m[1, 0] = 1
                >>> m.flux(axis=(0, 1, 0), theta=30, xo=[0.1, 0.2, 0.3, 0.4], yo=0.1, ro=0.1)
                array([[0.87233361],
                       [0.87177019],
                       [0.87135028],
                       [0.87108642]])
                >>> m.gradient["ro"]
                array([-0.27718567, -0.28843198, -0.29678989, -0.30200085])
                >>> m.gradient["theta"]
                array([-0.00882115, -0.0088464 , -0.00887311, -0.00890139])

            Note, importantly, that the derivatives in this module are all
            computed **analytically** using autodifferentiation, so their evaluation is fast
            and numerically stable. However, runtimes will in general be slower than those
            in :py:mod:`starry`.

            .. note:: If the degree of the map is large, you may run into a \
                      :py:obj:`RuntimeError` saying too many derivatives were requested. \
                      The :py:obj:`STARRY_NGRAD` compiler flag determines the size of the \
                      gradient vector and can be changed by setting an environment variable \
                      of the same name prior to compiling :py:obj:`starry`.
                      See :doc:`install` for more information.

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
            .. autoclass:: Planet(lmax=2, r=0.1, L=0, axis=(0, 1, 0), prot=0, a=50, porb=1, inc=90, ecc=0, w=90, Omega=0, lambda0=90, tref=0)
            .. autoclass:: System(bodies, scale=0, exposure_time=0, exposure_tol=1e-8, exposure_max_depth=4)
        )pbdoc";

        ngrad = R"pbdoc(
            Length of the gradient vector.
            This is a compile-time constant. If you get errors saying this
            value is too small, you'll need to re-compile :py:obj:`starry`.
            See :doc:`install` for more information.
        )pbdoc";

    };

}

#endif
