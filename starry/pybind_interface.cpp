#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <cmath>
#include <stdlib.h>
#include "constants.h"
#include "ellip.h"
#include "maps.h"
#include "basis.h"
#include "fact.h"
#include "sqrtint.h"
#include "rotation.h"
#include "solver.h"
#include "orbital.h"

using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;
template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T>
using VectorT = Eigen::Matrix<T, 1, Eigen::Dynamic>;
template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using UnitVector = Eigen::Matrix<T, 3, 1>;
using std::vector;


// Ensure we are passing the flux back to Python by reference. (I think we are).
// Check out PYBIND11_MAKE_OPAQUE(vector<orbital::Body<double>*>);


PYBIND11_MODULE(starry, m) {

    // Disable auto signatures
    py::options options;
    options.disable_function_signatures();

    m.doc() = R"pbdoc(
        API
        ---

        .. contents::
            :local:

        Introduction
        ============

        This page documents the :py:mod:`starry` API, which is coded
        in C++ and wrapped in Python using :py:mod:`pybind11`. The API consists
        of a :py:class:`Map` class, which houses all of the surface map photometry
        stuff, and the :py:class:`Star`, :py:class:`Planet`, and :py:class:`System`
        classes, which facilitate the generation of light curves for actual
        stellar and planetary systems.

        There are two broad ways in which users can access the core :py:mod:`starry`
        functionality:

            - Users can instantiate a :py:class:`Map` class to compute phase curves
              and occultation light curves by directly specifying the rotational state
              of the object and (optionally) the position and size of an occultor. This
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


        The Map class
        =============
        .. autoclass:: Map(lmax=2)


        The orbital classes
        ===================
        .. autoclass:: Star(r=1, L=1, m=1)
        .. autoclass:: Planet(lmax=2, r=1, L=0, u=(0, 1, 0), prot=0, theta0=0, porb=1, inc=90, ecc=0, w=90, Omega=0, lambda0=90, tref=0)
        .. autoclass:: System(bodies, kepler_tol=1.0e-7, kepler_max_iter=100)

    )pbdoc";

    // Orbital system class
    py::class_<orbital::System<double>>(m, "System", R"pbdoc(
            Instantiate an orbital system.

            Args:
                bodies (list): List of bodies in the system, with the primary (usually the star) listed first.
                kepler_tol (float): Kepler solver tolerance.
                kepler_max_iter (int): Maximum number of iterations in the Kepler solver.

            .. automethod:: compute(time)
            .. autoattribute:: flux
        )pbdoc")

        .def(py::init<vector<orbital::Body<double>*>, double, int>(),
            "bodies"_a, "kepler_tol"_a=1.0e-7, "kepler_max_iter"_a=100)

        .def("compute", &orbital::System<double>::compute,
            R"pbdoc(
                Compute the system light curve analytically.

                Compute the full system light curve at the times
                given by the :py:obj:`time <>` array and store the result
                in :py:attr:`flux`. The light curve for each body in the
                system is stored in the body's :py:attr:`flux` attribute.

                Args:
                    time (ndarray): Time array, measured in days.
            )pbdoc", "time"_a)

        .def_property_readonly("flux", [](orbital::System<double> &system){return system.flux;},
            R"pbdoc(
                The computed system light curve. Must run :py:meth:`compute` first.
            )pbdoc");

     // Body class
     py::class_<orbital::Body<double>> PyBody(m, "Body", R"pbdoc(
             Celestial body base class.

             .. note:: This is a base class for :py:class:`Star` and :py:class:`Body`. Users should \
                       instantiate body objects via those two subclasses rather than instantiating this class \
                       directly.

             Args:
                 lmax (int): Largest spherical harmonic degree in body's surface map.
                 r (float): Body radius in units of :py:obj:`UNIT_RADIUS`.
                 L (float): Body luminosity in units of :py:obj:`UNIT_LUMINOSITY`.
                 u (ndarray): A unit vector specifying the body's axis of rotation.
                 prot (float): Rotation period in days.
                 theta0 (float): Rotation phase at time :py:obj:`tref` in degrees.
                 m (float): Body mass in units of :py:obj:`UNIT_MASS`.
                 porb (float): Orbital period in days.
                 inc (float): Orbital inclination in degrees.
                 ecc (float): Orbital eccentricity.
                 w (float): Longitude of pericenter in degrees.
                 Omega (float): Longitude of ascending node in degrees.
                 lambda0 (float): Mean longitude at time :py:obj:`tref` in degrees.
                 tref (float): Reference time in days.
                 UNIT_RADIUS (float): Radius conversion factor in meters.
                 UNIT_MASS (float): Mass conversion factor for this body in kg.
                 UNIT_LUMINOSITY (float): Luminosity conversion factor for this body in W / m^2.

             .. autoattribute:: map
             .. autoattribute:: flux
             .. autoattribute:: x
             .. autoattribute:: y
             .. autoattribute:: z
             .. autoattribute:: r
             .. autoattribute:: L
             .. autoattribute:: u
             .. autoattribute:: prot
             .. autoattribute:: theta0
             .. autoattribute:: m
             .. autoattribute:: porb
             .. autoattribute:: inc
             .. autoattribute:: ecc
             .. autoattribute:: w
             .. autoattribute:: Omega
             .. autoattribute:: lambda0
             .. autoattribute:: tref
         )pbdoc");

     PyBody.def(py::init<int, const double&, const double&,
                         Eigen::Matrix<double, 3, 1>&,
                         const double&, const double&,
                         const double&, const double&,
                         const double&, const double&,
                         const double&, const double&,
                         const double&, const double&,
                         const double&, const double&,
                         const double&>(),
                         "lmax"_a, "r"_a, "L"_a, "u"_a,
                         "prot"_a, "theta0"_a, "m"_a, "porb"_a,
                         "inc"_a, "ecc"_a, "w"_a, "Omega"_a,
                         "lambda0"_a, "tref"_a, "UNIT_RADIUS"_a,
                         "UNIT_MASS"_a, "UNIT_LUMINOSITY"_a)

        .def_property_readonly("map", [](orbital::Body<double> &body){return &body.map;},
            R"pbdoc(
                The body's surface map.
            )pbdoc")

        .def_property_readonly("flux", [](orbital::Body<double> &body){return &body.flux;},
            R"pbdoc(
                The body's computed light curve.
            )pbdoc")

        .def_property_readonly("x", [](orbital::Body<double> &body){return body.x * AU;},
            R"pbdoc(
                The `x` position of the body in AU.
            )pbdoc")

        .def_property_readonly("y", [](orbital::Body<double> &body){return body.y * AU;},
            R"pbdoc(
                The `y` position of the body in AU.
            )pbdoc")

        .def_property_readonly("z", [](orbital::Body<double> &body){return body.z * AU;},
            R"pbdoc(
                The `z` position of the body in AU.
            )pbdoc")

        .def_property("r", [](orbital::Body<double> &body){return body.r / body.UNIT_RADIUS;},
                           [](orbital::Body<double> &body, double r){body.r = r * body.UNIT_RADIUS;},
            R"pbdoc(
                Body radius in units of :py:obj:`UNIT_RADIUS`.
            )pbdoc")

        .def_property("L", [](orbital::Body<double> &body){return body.L / body.UNIT_LUMINOSITY;},
                           [](orbital::Body<double> &body, double L){body.L = L * body.UNIT_LUMINOSITY; body.reset();},
            R"pbdoc(
                Body luminosity in units of :py:obj:`UNIT_LUMINOSITY`.
            )pbdoc")

        .def_property("u", [](orbital::Body<double> &body){return body.u;},
                           [](orbital::Body<double> &body, UnitVector<double> u){body.u = u;},
            R"pbdoc(
                Unit vector specifying the body's axis of rotation.
            )pbdoc")

        .def_property("prot", [](orbital::Body<double> &body){return body.prot / DAY;},
                              [](orbital::Body<double> &body, double prot){body.prot = prot * DAY; body.reset();},
            R"pbdoc(
                Rotation period in days.
            )pbdoc")

        .def_property("theta0", [](orbital::Body<double> &body){return body.theta0 / DEGREE;},
                                [](orbital::Body<double> &body, double theta0){body.theta0 = theta0 * DEGREE;},
            R"pbdoc(
                Rotation phase at time :py:obj:`tref` in degrees.
            )pbdoc")

        .def_property("m", [](orbital::Body<double> &body){return body.m / body.UNIT_MASS;},
                           [](orbital::Body<double> &body, double m){body.m = m * body.UNIT_MASS;},
            R"pbdoc(
                Body mass in units of :py:obj:`UNIT_MASS`.
            )pbdoc")

        .def_property("porb", [](orbital::Body<double> &body){return body.porb / DAY;},
                              [](orbital::Body<double> &body, double porb){body.porb = porb * DAY; body.reset();},
            R"pbdoc(
                Orbital period in days.
            )pbdoc")

        .def_property("inc", [](orbital::Body<double> &body){return body.inc / DEGREE;},
                             [](orbital::Body<double> &body, double inc){body.inc = inc * DEGREE; body.reset();},
            R"pbdoc(
                Orbital inclination in degrees.
            )pbdoc")

        .def_property("ecc", [](orbital::Body<double> &body){return body.ecc;},
                             [](orbital::Body<double> &body, double ecc){body.ecc = ecc; body.reset();},
            R"pbdoc(
                Orbital eccentricity.
            )pbdoc")

        .def_property("w", [](orbital::Body<double> &body){return body.w / DEGREE;},
                           [](orbital::Body<double> &body, double w){body.w = w * DEGREE; body.reset();},
            R"pbdoc(
                Longitude of pericenter in degrees.
            )pbdoc")

        .def_property("Omega", [](orbital::Body<double> &body){return body.Omega / DEGREE;},
                               [](orbital::Body<double> &body, double Omega){body.Omega = Omega * DEGREE; body.reset();},
            R"pbdoc(
                Longitude of ascending node in degrees.
            )pbdoc")

        .def_property("lambda0", [](orbital::Body<double> &body){return body.lambda0 / DEGREE;},
                                 [](orbital::Body<double> &body, double lambda0){body.lambda0 = lambda0 * DEGREE; body.reset();},
            R"pbdoc(
                Mean longitude at time :py:obj:`tref` in degrees.
            )pbdoc")

        .def_property("tref", [](orbital::Body<double> &body){return body.tref / DAY;},
                              [](orbital::Body<double> &body, double tref){body.tref = tref * DAY;},
            R"pbdoc(
                Reference time in days.
            )pbdoc")

        .def("__repr__", [](orbital::Body<double> &body) -> string {return body.repr();});

    // Star class
    py::class_<orbital::Star<double>>(m, "Star", PyBody, R"pbdoc(
            Instantiate a stellar :py:class:`Body` object.

            Instantiate a star by specifying its
            radius, mass, and luminosity, which default to solar values. The
            degree of the surface map is fixed at `l = 2` to allow for
            quadratic limb-darkending. This class has the following units:

                - :py:attr:`UNIT_RADIUS`: Solar radius
                - :py:attr:`UNIT_MASS`: Solar mass
                - :py:attr:`UNIT_LUMINOSITY`: Solar luminosity

            Args:
                r (float): Stellar radius in solar radii. Default 1.
                L (float): Stellar luminosity in units of LSUN. Default 1.
                m (float): Stellar mass in solar masses. Default 1.

            .. autoattribute:: map
            .. autoattribute:: flux
            .. autoattribute:: r
            .. autoattribute:: L
            .. autoattribute:: m
        )pbdoc")

        .def(py::init<const double&, const double&, const double&>(),
            "r"_a=1, "L"_a=1, "m"_a=1)

        .def("__repr__", [](orbital::Star<double> &star) -> string {return star.repr();});

    // Planet class
    py::class_<orbital::Planet<double>>(m, "Planet", PyBody, R"pbdoc(
            Instantiate a planetary :py:class:`Body` object.

            Instantiate a planet. At present, :py:mod:`starry` computes orbits with a simple
            Keplerian solver, so the planet is assumed to be massless.
            This class has the following units:

                - :py:attr:`UNIT_RADIUS`: Earth radius
                - :py:attr:`UNIT_MASS`: Earth mass
                - :py:attr:`UNIT_LUMINOSITY`: 1.e-9 solar luminosity

            Args:
                lmax (int): Largest spherical harmonic degree in body's surface map. Default 2.
                r (float): Body radius in Earth radii. Default 1.
                L (float): Body luminosity in units of LSUN. Default 0.
                u (ndarray): A unit vector specifying the body's axis of rotation. Default :math:`\hat{y} = (0, 1, 0)`.
                prot (float): Rotation period in days. Default no rotation.
                theta0 (float): Rotation phase at time :py:obj:`tref` in degrees. Default 0.
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
            .. autoattribute:: u
            .. autoattribute:: prot
            .. autoattribute:: theta0
            .. autoattribute:: porb
            .. autoattribute:: inc
            .. autoattribute:: ecc
            .. autoattribute:: w
            .. autoattribute:: Omega
            .. autoattribute:: lambda0
            .. autoattribute:: tref
        )pbdoc")

        .def(py::init<int, const double&, const double&,
                      Eigen::Matrix<double, 3, 1>&,
                      const double&, const double&,
                      const double&, const double&,
                      const double&, const double&,
                      const double&, const double&,
                      const double&>(),
                      "lmax"_a=2, "r"_a=1, "L"_a=0., "u"_a=maps::yhat,
                      "prot"_a=0, "theta0"_a=0, "porb"_a=1,
                      "inc"_a=90., "ecc"_a=0, "w"_a=90, "Omega"_a=0,
                      "lambda0"_a=90, "tref"_a=0)

        .def("__repr__", [](orbital::Planet<double> &planet) -> string {return planet.repr();});

    // Surface map class
    py::class_<maps::Map<double>>(m, "Map", R"pbdoc(
            Instantiate a :py:mod:`starry` surface map.

            Args:
                lmax (int): Largest spherical harmonic degree in the surface map. Default 2.

            .. autoattribute:: use_mp
            .. autoattribute:: taylor
            .. automethod:: evaluate(u=(0, 1, 0), theta=0, x=0, y=0)
            .. automethod:: rotate(u=(0, 1, 0), theta=0)
            .. automethod:: flux(u=(0, 1, 0), theta=0, xo=0, yo=0, ro=0, numerical=False, tol=1.e-4)
            .. automethod:: get_coeff(l, m)
            .. automethod:: set_coeff(l, m, coeff)
            .. automethod:: reset()
            .. automethod:: limbdark(u1, u2)
            .. autoattribute:: lmax
            .. autoattribute:: y
            .. autoattribute:: p
            .. autoattribute:: g
            .. automethod:: minimum()
            .. automethod:: load_image(image)
            .. automethod:: load_healpix(image)
            .. automethod:: show(cmap='plasma', res=300)
            .. automethod:: animate(u=(0, 1, 0), cmap='plasma', res=150, frames=50)

        )pbdoc")

        .def(py::init<int>(), "lmax"_a=2)

        .def_property("use_mp", [](maps::Map<double> &map){return map.use_mp;},
                                [](maps::Map<double> &map, bool use_mp){map.use_mp = use_mp;},
            R"pbdoc(
                Set to :py:obj:`True` to turn on multi-precision mode. By default, this \
                will perform all occultation calculations using 128-bit (quadruple) floating point \
                precision, corresponding to 32 significant digits. Users can increase this to any \
                number of digits (RAM permitting) by setting the :py:obj:`STARRY_MP_DIGITS=XX` flag \
                at compile time. Note, importantly, that run times are **much** slower when multi-precision \
                is enabled. Default :py:obj:`False`.
            )pbdoc")

        .def_property("taylor", [](maps::Map<double> &map){return map.G.taylor;},
                                [](maps::Map<double> &map, bool taylor){map.G.taylor = taylor;},
            R"pbdoc(
                Set to :py:obj:`False` to disable Taylor expansions of the primitive integrals when \
                computing occultation light curves. This is in general not something you should do! \
                Default :py:obj:`True`.
            )pbdoc")

        .def("evaluate", py::vectorize(&maps::Map<double>::evaluate),
            R"pbdoc(
                Return the specific intensity at a point (`x`, `y`) on the map.

                Users may optionally provide a rotation state. Note that this does
                not rotate the base map.

                Args:
                    u (ndarray): Unit vector specifying the body's axis of rotation. Default :math:`\hat{y} = (0, 1, 0)`.
                    theta (float or ndarray): Angle of rotation in radians. Default 0.
                    x (float or ndarray): Position scalar, vector, or matrix.
                    y (float or ndarray): Position scalar, vector, or matrix.

                Returns:
                    The specific intensity at (`x`, `y`).
            )pbdoc", "u"_a=maps::yhat, "theta"_a=0, "x"_a=0, "y"_a=0)

        .def("rotate", [](maps::Map<double> &map, UnitVector<double>& u,
                          double theta){return map.rotate(u, theta);},
            R"pbdoc(
                Rotate the base map an angle :py:obj:`theta` about :py:obj:`u`.

                This performs a permanent rotation to the base map. Subsequent
                rotations and calculations will be performed relative to this
                rotational state.

                Args:
                    u (ndarray): Unit vector specifying the body's axis of rotation. Default :math:`\hat{y} = (0, 1, 0)`.
                    theta (float or ndarray): Angle of rotation in radians. Default 0.

            )pbdoc", "u"_a=maps::yhat, "theta"_a=0)

        .def("flux", py::vectorize(&maps::Map<double>::flux),
            R"pbdoc(
                Return the total flux received by the observer.

                Computes the total flux received by the observer from the
                map during or outside of an occultation.

                Args:
                    u (ndarray): Unit vector specifying the body's axis of rotation. Default :math:`\hat{y} = (0, 1, 0)`.
                    theta (float or ndarray): Angle of rotation. Default 0.
                    xo (float or ndarray): The `x` position of the occultor (if any). Default 0.
                    yo (float or ndarray): The `y` position of the occultor (if any). Default 0.
                    ro (float): The radius of the occultor in units of this body's radius. Default 0 (no occultation).
                    numerical (bool): Compute the flux numerically using an adaptive mesh? Default :py:obj:`False`.
                    tol (float): Tolerance of the numerical solver. Default `1.e-4`

                Returns:
                    The flux received by the observer (a scalar or a vector).
            )pbdoc", "u"_a=maps::yhat, "theta"_a=0, "xo"_a=0,
                     "yo"_a=0, "ro"_a=0, "numerical"_a=false,
                     "tol"_a=1e-4)

        .def("get_coeff", &maps::Map<double>::get_coeff,
            R"pbdoc(
                Return the (:py:obj:`l`, :py:obj:`m`) coefficient of the map.

                .. note:: Users can also retrieve a coefficient by accessing the \
                          [:py:obj:`l`, :py:obj:`m`] index of the map as if it \
                          were a 2D array.

                Args:
                    l (int): The spherical harmonic degree, ranging from 0 to :py:attr:`lmax`.
                    m (int): The spherical harmonic order, ranging from -:py:obj:`l` to :py:attr:`l`.
            )pbdoc", "l"_a, "m"_a)

        .def("set_coeff", &maps::Map<double>::set_coeff,
            R"pbdoc(
                Set the (:py:obj:`l`, :py:obj:`m`) coefficient of the map.

                .. note:: Users can also set a coefficient by setting the \
                          [:py:obj:`l`, :py:obj:`m`] index of the map as if it \
                          were a 2D array.

                Args:
                    l (int): The spherical harmonic degree, ranging from 0 to :py:attr:`lmax`.
                    m (int): The spherical harmonic order, ranging from -:py:obj:`l` to :py:attr:`l`.
                    coeff (float): The value of the coefficient.
            )pbdoc", "l"_a, "m"_a, "coeff"_a)

        .def("reset", &maps::Map<double>::reset,
            R"pbdoc(
                Set all of the map coefficients to zero.
            )pbdoc")

        .def("limbdark", &maps::Map<double>::limbdark,
            R"pbdoc(
                Set the linear and quadratic limb darkening coefficients.

                .. note:: This will overwrite all existing coefficients.

                Args:
                    u1 (float): The linear limb darkening coefficient.
                    u2 (float): The quadratic limb darkening coefficient.

            )pbdoc", "u1"_a, "u2"_a)

        .def_property_readonly("lmax", [](maps::Map<double> &map){return map.lmax;},
            R"pbdoc(
                The highest spherical harmonic order of the map. *Read-only.*
            )pbdoc")

        .def_property_readonly("y", [](maps::Map<double> &map){map.update(true); return map.y;},
            R"pbdoc(
                The spherical harmonic map vector. *Read-only.*
            )pbdoc")

        .def_property_readonly("p", [](maps::Map<double> &map){map.update(true); return map.p;},
            R"pbdoc(
                The polynomial map vector. *Read-only.*
            )pbdoc")

        .def_property_readonly("g", [](maps::Map<double> &map){map.update(true); return map.g;},
            R"pbdoc(
                The Green's polynomial map vector. *Read-only.*
            )pbdoc")

        .def("__setitem__", [](maps::Map<double>& map, vector<int> lm, double coeff){
            if (lm.size() == 1) {
                int l = (int)floor(sqrt(lm[0]));
                int m = lm[0] - l * l - l;
                map.set_coeff(l, m, coeff);
            } else if (lm.size() == 2) {
                map.set_coeff(lm[0], lm[1], coeff);
            } else {
                std::cout << "ERROR: Invalid spherical harmonic index." << std::endl;
            }
        })

        .def("__getitem__", [](maps::Map<double>& map, vector<int> lm){
            if (lm.size() == 1) {
                int l = (int)floor(sqrt(lm[0]));
                int m = lm[0] - l * l - l;
                return map.get_coeff(l, m);
            } else if (lm.size() == 2) {
                return map.get_coeff(lm[0], lm[1]);
            } else {
                std::cout << "ERROR: Invalid spherical harmonic index." << std::endl;
                return double(0.);
            }
        })

        .def("__repr__", [](maps::Map<double> &map) -> string {return map.repr();})

        //
        // This is where things go nuts: Let's call Python from C++
        //

        .def("minimum", [](maps::Map<double> &map) -> double {
            map.update();
            py::object minimize = py::module::import("starry_maps").attr("minimize");
            return minimize(map.p).cast<double>();
        },
        R"pbdoc(
            Find the global minimum of the map.

            This routine wraps :py:class:`scipy.optimize.minimize` to find
            the global minimum of the surface map. This is useful for ensuring
            that the surface map is nonnegative everywhere. If a negative value
            is returned, add this value to the `l = 0, m = 0` coefficient of the
            map to force the map to not go negative.

            .. note:: Because this routine wraps a Python wrapper of a C function \
                      to perform a non-linear optimization in three dimensions, it is \
                      **slow** and should probably not be used repeatedly when fitting \
                      a map to data!
        )pbdoc")

        .def("load_image", [](maps::Map<double> &map, string& image) {
            py::object load_map = py::module::import("starry_maps").attr("load_map");
            Vector<double> y = load_map(image, map.lmax).cast<Vector<double>>();
            int n = 0;
            for (int l = 0; l < map.lmax + 1; l++) {
                for (int m = -l; m < l + 1; m++) {
                    map.set_coeff(l, m, y(n));
                    n++;
                }
            }
            // We need to apply some rotations to get
            // to the desired orientation
            map.rotate(maps::xhat, M_PI / 2.);
            map.rotate(maps::zhat, M_PI);
            map.rotate(maps::yhat, M_PI / 2);
        },
        R"pbdoc(
            Load an image from file.

            This routine loads an image file, computes its spherical harmonic
            expansion up to degree :py:attr:`lmax`, and sets the map vector.

            Args:
                image (str): The full path to the image file.

        )pbdoc", "image"_a)

        .def("load_healpix", [](maps::Map<double> &map, Matrix<double>& image) {
            py::object load_map = py::module::import("starry_maps").attr("load_map");
            Vector<double> y = load_map(image, map.lmax).cast<Vector<double>>();
            int n = 0;
            for (int l = 0; l < map.lmax + 1; l++) {
                for (int m = -l; m < l + 1; m++) {
                    map.set_coeff(l, m, y(n));
                    n++;
                }
            }
            // We need to apply some rotations to get
            // to the desired orientation
            map.rotate(maps::xhat, M_PI / 2.);
            map.rotate(maps::zhat, M_PI);
            map.rotate(maps::yhat, M_PI / 2);
        },
        R"pbdoc(
            Load a healpix image array.

            This routine loads a :py:obj:`healpix` array, computes its
            spherical harmonic
            expansion up to degree :py:attr:`lmax`, and sets the map vector.

            Args:
                image (ndarray): The ring-ordered :py:obj:`healpix` array.
        )pbdoc", "image"_a)

        .def("show", [](maps::Map<double> &map, string cmap, int res) {
            py::object show = py::module::import("starry_maps").attr("show");
            Matrix<double> I;
            I.resize(res, res);
            Vector<double> x;
            x = Vector<double>::LinSpaced(res, -1, 1);
            for (int i = 0; i < res; i++){
                for (int j = 0; j < res; j++){
                    I(j, i) = map.evaluate(maps::yhat, 0, x(i), x(j));
                }
            }
            show(I, "cmap"_a=cmap, "res"_a=res);
        },
        R"pbdoc(
            Convenience routine to quickly display the body's surface map.

            Args:
                cmap (str): The :py:mod:`matplotlib` colormap name. Default `plasma`.
                res (int): The resolution of the map in pixels on a side. Default 300.
        )pbdoc", "cmap"_a="plasma", "res"_a=300)

        .def("animate", [](maps::Map<double> &map, UnitVector<double>& u, string cmap, int res, int frames) {
            std::cout << "Rendering animation..." << std::endl;
            py::object animate = py::module::import("starry_maps").attr("animate");
            vector<Matrix<double>> I;
            Vector<double> x, theta;
            x = Vector<double>::LinSpaced(res, -1, 1);
            theta = Vector<double>::LinSpaced(frames, 0, 2 * M_PI);
            for (int t = 0; t < frames; t++){
                I.push_back(Matrix<double>::Zero(res, res));
                for (int i = 0; i < res; i++){
                    for (int j = 0; j < res; j++){
                        I[t](j, i) = map.evaluate(u, theta(t), x(i), x(j));
                    }
                }
            }
            animate(I, u, "cmap"_a=cmap, "res"_a=res);
        },
        R"pbdoc(
            Convenience routine to animate the body's surface map as it rotates.

            Args:
                u (ndarray): Unit vector specifying the axis of rotation. Default :math:`\hat{y} = (0, 1, 0)`.
                cmap (str): The :py:mod:`matplotlib` colormap name. Default `plasma`.
                res (int): The resolution of the map in pixels on a side. Default 150.
                frames (int): The number of frames in the animation. Default 50.
        )pbdoc", "u"_a=maps::yhat, "cmap"_a="plasma", "res"_a=150, "frames"_a=50);


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

}
