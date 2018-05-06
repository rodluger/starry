/**
This defines the main Python interface to the code.
Note that this file is #include'd *twice*, once
for `starry` and once for `starry.grad` so we don't
have to duplicate any code. Yes, this is super hacky.

TODO: Make usage of & consistent in input arguments!
*/

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>
#include <cmath>
#include <stdlib.h>
#include <vector>
#include "maps.h"
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


// Home-built vectorization wrapper to replace py::vectorize when using autodiff
#ifndef _STARRY_VECTORIZE_
#define _STARRY_VECTORIZE_
Eigen::VectorXd vectorize(py::object& obj, int& size){
    Eigen::VectorXd res;
    if (py::isinstance<py::float_>(obj) || py::isinstance<py::int_>(obj)) {
        res = Eigen::VectorXd::Constant(size, py::cast<double>(obj));
        return res;
    } else if (py::isinstance<py::array>(obj) || py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
        res = py::cast<Eigen::VectorXd>(obj);
        if ((size == 0) || (res.size() == size)) {
            size = res.size();
            return res;
        } else {
            throw invalid_argument("Mismatch in argument dimensions.");
        }
    } else {
        throw invalid_argument("Incorrect type for one or more of the arguments.");
    }
}
#endif


/**
Define our map type (double or AutoDiffScalar)
Apologies to sticklers for the indented #define's.

*/
#ifndef STARRY_AUTODIFF

    #undef MapType
    #define MapType                         double

#else

    #define STARRY_NGRAD                    21
    #define STARRY_NGRAD_MAP_EVALUATE       6
    #define STARRY_NGRAD_MAP_FLUX           7
    #define STARRY_NGRAD_LDMAP_EVALUATE     2
    #define STARRY_NGRAD_LDMAP_FLUX         3
    #undef MapType
    #define MapType                         Eigen::AutoDiffScalar<Eigen::Matrix<double, STARRY_NGRAD, 1>> // Eigen::AutoDiffScalar<Eigen::VectorXd>

#endif


/**
Instantiate the `starry` module with or without autodiff
*/
#ifndef STARRY_AUTODIFF
void add_starry(py::module &m) {

    m.doc() = R"pbdoc(
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
        stellar and planetary systems.)pbdoc"

#else
void add_starry_grad(py::module &m) {

    m.doc() = R"pbdoc(
        starry.grad
        -----------

        .. contents::
            :local:

        Introduction
        ============

        This page documents the :py:mod:`starry.grad` API, which is coded
        in C++ with a :py:mod:`pybind11` Python interface. This API is
        identical in nearly all respects to the :py:mod:`starry` API, except
        that its methods return gradients with respect to the input parameters,
        in addition to the actual return values. For instance, consider the
        following code block:

        .. code-block:: python

            >>> import starry
            >>> m = starry.Map()
            >>> m[1, 0] = 1
            >>> m.flux(axis=(0, 1, 0), theta=0.3, xo=0.1, yo=0.1, ro=0.1)
            0.9626882655504516

        Here's the same code executed using the :py:obj:`Map()` class in :py:mod:`starry.grad`:

        .. code-block:: python

            >>> import starry
            >>> m = starry.Map()
            >>> m[1, 0] = 1
            >>> m.flux(axis=(0, 1, 0), theta=0.3, xo=0.1, yo=0.1, ro=0.1)
            array([[ 9.62688266e-01,  4.53620580e-04,  0.00000000e+00,
                    -6.85580453e-05, -2.99401131e-01, -3.04715096e-03,
                    1.48905485e-03, -2.97910667e-01]])

        The `flux()` method now returns a vector, where the first value is the
        actual flux and the remaining seven values are the derivatives of the flux
        with respect to each of the input parameters
        :py:obj:`(axis[0], axis[1], axis[2], theta, xo, yo, ro)`. Note that as in
        :py:mod:`starry`, many of the functions in :py:mod:`starry.grad` are
        vectorizable, meaning that vectors can be provided as inputs to compute,
        say, the light curve for an entire timeseries. In this case, the return
        values are **matrices**, with one vector of :py:obj:`(value, derivs)` per row.

        Note, importantly, that the derivatives in this module are all
        computed **analytically** using autodifferentiation, so their evaluation is fast
        and numerically stable. However, runtimes will in general be slower than those
        in :py:mod:`starry`.

        As in :py:mod:`starry`, the API consists of a :py:class:`Map` class,
        which houses all of the surface map photometry
        stuff, and the :py:class:`Star`, :py:class:`Planet`, and :py:class:`System`
        classes, which facilitate the generation of light curves for actual
        stellar and planetary systems.)pbdoc"

#endif

        R"pbdoc(There are two broad ways in which users can access
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
        .. autoclass:: Planet(lmax=2, r=0.1, L=0, axis=(0, 1, 0), prot=0, theta0=0, a=50, porb=1, inc=90, ecc=0, w=90, Omega=0, lambda0=90, tref=0)
        .. autoclass:: System(bodies, kepler_tol=1.0e-7, kepler_max_iter=100)

    )pbdoc";

#ifndef STARRY_AUTODIFF
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

     // Body class (not user-facing, just a base class)
     py::class_<orbital::Body<double>> PyBody(m, "Body");

     PyBody.def(py::init<int, const double&, const double&,
                         Eigen::Matrix<double, 3, 1>&,
                         const double&, const double&,
                         const double&, const double&,
                         const double&, const double&,
                         const double&, const double&,
                         const double&, const double&,
                         bool>(),
                         "lmax"_a, "r"_a, "L"_a, "axis"_a,
                         "prot"_a, "theta0"_a, "a"_a, "porb"_a,
                         "inc"_a, "ecc"_a, "w"_a, "Omega"_a,
                         "lambda0"_a, "tref"_a, "is_star"_a)

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

        .def_property("r", [](orbital::Body<double> &body){return body.r;},
                           [](orbital::Body<double> &body, double r){body.r = r;},
            R"pbdoc(
                Body radius in units of stellar radius.
            )pbdoc")

        .def_property("L", [](orbital::Body<double> &body){return body.L;},
                           [](orbital::Body<double> &body, double L){body.L = L; body.reset();},
            R"pbdoc(
                Body luminosity in units of stellar luminosity.
            )pbdoc")

        .def_property("axis", [](orbital::Body<double> &body){return body.axis;},
                           [](orbital::Body<double> &body, UnitVector<double> axis){body.axis = axis;},
            R"pbdoc(
                *Normalized* unit vector specifying the body's axis of rotation.
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

        .def_property("a", [](orbital::Body<double> &body){return body.a;},
                           [](orbital::Body<double> &body, double a){body.a = a;},
            R"pbdoc(
                Body semi-major axis in units of stellar radius.
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
#else
        // TODO
        py::class_<orbital::System<MapType>>(m, "System", R"pbdoc(Not yet implemented.)pbdoc");
#endif

#ifndef STARRY_AUTODIFF
    // Star class
    py::class_<orbital::Star<double>>(m, "Star", PyBody, R"pbdoc(
            Instantiate a stellar :py:class:`Body` object.

            The star's radius and luminosity are fixed at unity.

            Args:
                lmax (int): Largest spherical harmonic degree in body's surface map. Default 2.

            .. autoattribute:: map
            .. autoattribute:: flux
        )pbdoc")

        .def(py::init<int>(), "lmax"_a=2)
        .def_property_readonly("map", [](orbital::Body<double> &body){return &body.ldmap;},
            R"pbdoc(
                The star's surface map, a :py:class:`LimbDarkenedMap` instance.
            )pbdoc")
        .def_property_readonly("r", [](orbital::Star<double> &star){return star.r;})
        .def_property_readonly("L", [](orbital::Star<double> &star){return star.L;})
        .def_property_readonly("axis", [](orbital::Star<double> &star){return star.axis;})
        .def_property_readonly("prot", [](orbital::Star<double> &star){return star.prot;})
        .def_property_readonly("theta0", [](orbital::Star<double> &star){return star.theta0;})
        .def_property_readonly("a", [](orbital::Star<double> &star){return star.a;})
        .def_property_readonly("porb", [](orbital::Star<double> &star){return star.porb;})
        .def_property_readonly("inc", [](orbital::Star<double> &star){return star.inc;})
        .def_property_readonly("ecc", [](orbital::Star<double> &star){return star.ecc;})
        .def_property_readonly("w", [](orbital::Star<double> &star){return star.w;})
        .def_property_readonly("Omega", [](orbital::Star<double> &star){return star.Omega;})
        .def_property_readonly("lambda0", [](orbital::Star<double> &star){return star.lambda0;})
        .def_property_readonly("tref", [](orbital::Star<double> &star){return star.tref;})
        .def("__setitem__", [](orbital::Star<double> &body, py::object index, double coeff) {
            if (py::isinstance<py::tuple>(index)) {
                throw errors::BadIndex();
            } else {
                int l = py::cast<int>(index);
                body.ldmap.set_coeff(l, coeff);
            }
        })
        .def("__getitem__", [](orbital::Star<double> &body, py::object index) {
            if (py::isinstance<py::tuple>(index)) {
                throw errors::BadIndex();
            } else {
                int l = py::cast<int>(index);
                return body.ldmap.get_coeff(l);
            }
        })
        .def("__repr__", [](orbital::Star<double> &star) -> string {return star.repr();});
#else
        // TODO
        py::class_<orbital::Star<MapType>>(m, "Star", R"pbdoc(Not yet implemented.)pbdoc");
#endif

#ifndef STARRY_AUTODIFF
    // Planet class
    py::class_<orbital::Planet<double>>(m, "Planet", PyBody, R"pbdoc(
            Instantiate a planetary :py:class:`Body` object.

            Instantiate a planet. At present, :py:mod:`starry` computes orbits with a simple
            Keplerian solver, so the planet is assumed to be massless.

            Args:
                lmax (int): Largest spherical harmonic degree in body's surface map. Default 2.
                r (float): Body radius in stellar radii. Default 0.1
                L (float): Body luminosity in units of the stellar luminosity. Default 0.
                axis (ndarray): A *normalized* unit vector specifying the body's axis of rotation. Default :math:`\hat{y} = (0, 1, 0)`.
                prot (float): Rotation period in days. Default no rotation.
                theta0 (float): Rotation phase at time :py:obj:`tref` in degrees. Default 0.
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
            .. autoattribute:: theta0
            .. autoattribute:: a
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
                      const double&, const double&>(),
                      "lmax"_a=2, "r"_a=0.1, "L"_a=0., "axis"_a=maps::yhat,
                      "prot"_a=0, "theta0"_a=0, "a"_a=50., "porb"_a=1,
                      "inc"_a=90., "ecc"_a=0, "w"_a=90, "Omega"_a=0,
                      "lambda0"_a=90, "tref"_a=0)
        .def("__setitem__", [](orbital::Planet<double> &body, py::object index, double coeff) {
            if (py::isinstance<py::tuple>(index)) {
                // This is a (l, m) tuple
                py::tuple lm = index;
                int l = py::cast<int>(lm[0]);
                int m = py::cast<int>(lm[1]);
                body.map.set_coeff(l, m, coeff);
            } else {
                throw errors::BadIndex();
            }
        })
        .def("__getitem__", [](orbital::Planet<double> &body, py::object index) {
            if (py::isinstance<py::tuple>(index)) {
                // This is a (l, m) tuple
                py::tuple lm = index;
                int l = py::cast<int>(lm[0]);
                int m = py::cast<int>(lm[1]);
                return body.map.get_coeff(l, m);
            } else {
                throw errors::BadIndex();
            }
        })
        .def("__repr__", [](orbital::Planet<double> &planet) -> string {return planet.repr();});
#else
        // TODO
        py::class_<orbital::Planet<MapType>>(m, "Planet", R"pbdoc(Not yet implemented.)pbdoc");
#endif

    // Surface map class
    py::class_<maps::Map<MapType>>(m, "Map", R"pbdoc(
            Instantiate a :py:mod:`starry` surface map.

            Args:
                lmax (int): Largest spherical harmonic degree in the surface map. Default 2.

            .. autoattribute:: optimize
            .. automethod:: evaluate(axis=(0, 1, 0), theta=0, x=0, y=0)
            .. automethod:: rotate(axis=(0, 1, 0), theta=0)
            .. automethod:: flux_numerical(axis=(0, 1, 0), theta=0, xo=0, yo=0, ro=0, tol=1.e-4))pbdoc"
    #ifndef STARRY_AUTODIFF
            R"pbdoc(
            .. automethod:: flux_mp(axis=(0, 1, 0), theta=0, xo=0, yo=0, ro=0))pbdoc"
    #endif
            R"pbdoc(
            .. automethod:: flux(axis=(0, 1, 0), theta=0, xo=0, yo=0, ro=0)
            .. automethod:: get_coeff(l, m)
            .. automethod:: set_coeff(l, m, coeff)
            .. automethod:: reset()
            .. autoattribute:: lmax
            .. autoattribute:: y
            .. autoattribute:: p
            .. autoattribute:: g
            .. autoattribute:: s)pbdoc"
    #ifndef STARRY_AUTODIFF
            R"pbdoc(
            .. autoattribute:: s_mp)pbdoc"
    #endif
            R"pbdoc(
            .. autoattribute:: r
            .. automethod:: minimum()
            .. automethod:: load_image(image)
            .. automethod:: load_healpix(image)
            .. automethod:: show(cmap='plasma', res=300)
            .. automethod:: animate(axis=(0, 1, 0), cmap='plasma', res=150, frames=50)

        )pbdoc")

        .def(py::init<int>(), "lmax"_a=2)

        .def_property("optimize", [](maps::Map<MapType> &map){return map.G.taylor;},
                                  [](maps::Map<MapType> &map, bool taylor){map.G.taylor = taylor;},
            R"pbdoc(
                Set to :py:obj:`False` to disable Taylor expansions of the primitive integrals when \
                computing occultation light curves. This is in general not something you should do! \
                Default :py:obj:`True`.
            )pbdoc")

        .def("evaluate",
    #ifndef STARRY_AUTODIFF
            py::vectorize(&maps::Map<MapType>::evaluate),
    #else
            [](maps::Map<MapType>& map, UnitVector<double>& axis, py::object theta, py::object x, py::object y){
                // Vectorize the inputs
                int size = 0;
                Eigen::VectorXd theta_v, x_v, y_v;
                if (py::hasattr(theta, "__len__")) {
                    theta_v = vectorize(theta, size);
                    x_v = vectorize(x, size);
                    y_v = vectorize(y, size);
                } else if (py::hasattr(x, "__len__")) {
                    x_v = vectorize(x, size);
                    y_v = vectorize(y, size);
                    theta_v = vectorize(theta, size);
                } else if (py::hasattr(y, "__len__")) {
                    y_v = vectorize(y, size);
                    theta_v = vectorize(theta, size);
                    x_v = vectorize(x, size);
                } else {
                    size = 1;
                    theta_v = vectorize(theta, size);
                    x_v = vectorize(x, size);
                    y_v = vectorize(y, size);
                }

                // Declare the result matrix
                Eigen::MatrixXd result(theta_v.size(), STARRY_NGRAD_MAP_EVALUATE + 1);

                // Declare our gradient types
                MapType axis_x(axis(0), STARRY_NGRAD, 0);
                MapType axis_y(axis(1), STARRY_NGRAD, 1);
                MapType axis_z(axis(2), STARRY_NGRAD, 2);
                MapType theta_g(0., STARRY_NGRAD, 3);
                MapType x_g(0., STARRY_NGRAD, 4);
                MapType y_g(0., STARRY_NGRAD, 5);
                UnitVector<MapType> axis_g({axis_x, axis_y, axis_z});
                MapType tmp;

                // Compute the flux at each cadence
                for (int i = 0; i < theta_v.size(); i++) {
                    theta_g.value() = theta_v(i);
                    x_g.value() = x_v(i);
                    y_g.value() = y_v(i);
                    tmp = map.evaluate(axis_g, theta_g, x_g, y_g);
                    result(i, 0) = tmp.value();
                    result.block<1, STARRY_NGRAD_MAP_EVALUATE>(i, 1) = tmp.derivatives().head<STARRY_NGRAD_MAP_EVALUATE>();
                }
                return result;
            },
    #endif
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
            )pbdoc", "axis"_a=maps::yhat, "theta"_a=0, "x"_a=0, "y"_a=0)

        .def("rotate",
    #ifndef STARRY_AUTODIFF
            [](maps::Map<MapType> &map, UnitVector<double>& axis, double theta){
                return map.rotate(axis, theta);
            },
    #else
            [](maps::Map<MapType> &map, UnitVector<double>& axis, double theta){
                MapType axis_x(axis(0), STARRY_NGRAD, 0);
                MapType axis_y(axis(1), STARRY_NGRAD, 1);
                MapType axis_z(axis(2), STARRY_NGRAD, 2);
                MapType theta_g(theta, STARRY_NGRAD, 3);
                UnitVector<MapType> axis_g({axis_x, axis_y, axis_z});
                return map.rotate(axis_g, theta_g);
            },
    #endif
            R"pbdoc(
                Rotate the base map an angle :py:obj:`theta` about :py:obj:`axis`.

                This performs a permanent rotation to the base map. Subsequent
                rotations and calculations will be performed relative to this
                rotational state.

                Args:
                    axis (ndarray): *Normalized* unit vector specifying the body's axis of rotation. Default :math:`\hat{y} = (0, 1, 0)`.
                    theta (float or ndarray): Angle of rotation in radians. Default 0.

            )pbdoc", "axis"_a=maps::yhat, "theta"_a=0)

        .def("flux_numerical",
    #ifndef STARRY_AUTODIFF
            py::vectorize(&maps::Map<MapType>::flux_numerical),
    #else
            [](maps::Map<MapType> &map, UnitVector<double>& axis, py::object theta, py::object xo, py::object yo, py::object ro, double tol){
                // Vectorize the inputs
                int size = 0;
                Eigen::VectorXd theta_v, xo_v, yo_v, ro_v;
                if (py::hasattr(theta, "__len__")) {
                    theta_v = vectorize(theta, size);
                    xo_v = vectorize(xo, size);
                    yo_v = vectorize(yo, size);
                    ro_v = vectorize(ro, size);
                } else if (py::hasattr(xo, "__len__")) {
                    xo_v = vectorize(xo, size);
                    yo_v = vectorize(yo, size);
                    ro_v = vectorize(ro, size);
                    theta_v = vectorize(theta, size);
                } else if (py::hasattr(yo, "__len__")) {
                    yo_v = vectorize(yo, size);
                    ro_v = vectorize(ro, size);
                    theta_v = vectorize(theta, size);
                    xo_v = vectorize(xo, size);
                } else if (py::hasattr(ro, "__len__")) {
                    ro_v = vectorize(ro, size);
                    theta_v = vectorize(theta, size);
                    xo_v = vectorize(xo, size);
                    yo_v = vectorize(yo, size);
                } else {
                    size = 1;
                    theta_v = vectorize(theta, size);
                    xo_v = vectorize(xo, size);
                    yo_v = vectorize(yo, size);
                    ro_v = vectorize(ro, size);
                }

                // Declare the result vector
                Eigen::VectorXd result(theta_v.size());

                // Declare our gradient types, although I simply *will not*
                // take automatic derivatives of a numerically computed function!
                MapType axis_x(axis(0));
                MapType axis_y(axis(1));
                MapType axis_z(axis(2));
                MapType theta_g(0.);
                MapType xo_g(0.);
                MapType yo_g(0.);
                MapType ro_g(0.);
                UnitVector<MapType> axis_g({axis_x, axis_y, axis_z});
                MapType tmp;

                // Compute the flux at each cadence
                for (int i = 0; i < theta_v.size(); i++) {
                    theta_g.value() = theta_v(i);
                    xo_g.value() = xo_v(i);
                    yo_g.value() = yo_v(i);
                    ro_g.value() = ro_v(i);
                    tmp = map.flux_numerical(axis_g, theta_g, xo_g, yo_g, ro_g, tol);
                    result(i) = tmp.value();
                }
                return result;
            },
    #endif
            R"pbdoc(
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
                )pbdoc"
    #ifndef STARRY_AUTODIFF
                R"pbdoc()pbdoc",
    #else
                R"pbdoc(
                .. note:: This function only returns the **value** of the numerical flux, and **not** its \
                          derivatives. Autodifferentiation of numerical integration is \
                          simply a terrible idea!
                )pbdoc",
    #endif
            "axis"_a=maps::yhat, "theta"_a=0, "xo"_a=0, "yo"_a=0, "ro"_a=0, "tol"_a=1e-4)

    #ifndef STARRY_AUTODIFF
        // NOTE: No autograd implementation of this function.
        .def("flux_mp",

            py::vectorize(&maps::Map<MapType>::flux_mp),
            R"pbdoc(
                Return the total flux received by the observer, computed using multi-precision.

                Computes the total flux received by the observer from the
                map during or outside of an occultation. By default, this method
                performs all occultation calculations using 128-bit (quadruple) floating point
                precision, corresponding to 32 significant digits. Users can increase this to any
                number of digits (RAM permitting) by setting the :py:obj:`STARRY_MP_DIGITS=XX` flag
                at compile time. Note, importantly, that run times are **much** slower for multi-precision
                calculations.

                Args:
                    axis (ndarray): *Normalized* unit vector specifying the body's axis of rotation. Default :math:`\hat{y} = (0, 1, 0)`.
                    theta (float or ndarray): Angle of rotation. Default 0.
                    xo (float or ndarray): The `x` position of the occultor (if any). Default 0.
                    yo (float or ndarray): The `y` position of the occultor (if any). Default 0.
                    ro (float): The radius of the occultor in units of this body's radius. Default 0 (no occultation).

                Returns:
                    The flux received by the observer (a scalar or a vector).
            )pbdoc", "axis"_a=maps::yhat, "theta"_a=0, "xo"_a=0, "yo"_a=0, "ro"_a=0)
    #endif

        .def("flux",
    #ifndef STARRY_AUTODIFF
            py::vectorize(&maps::Map<MapType>::flux),
    #else
            [](maps::Map<MapType> &map, UnitVector<double> axis, py::object theta, py::object xo, py::object yo, py::object ro){
                // Vectorize the inputs
                int size = 0;
                Eigen::VectorXd theta_v, xo_v, yo_v, ro_v;
                if (py::hasattr(theta, "__len__")) {
                    theta_v = vectorize(theta, size);
                    xo_v = vectorize(xo, size);
                    yo_v = vectorize(yo, size);
                    ro_v = vectorize(ro, size);
                } else if (py::hasattr(xo, "__len__")) {
                    xo_v = vectorize(xo, size);
                    yo_v = vectorize(yo, size);
                    ro_v = vectorize(ro, size);
                    theta_v = vectorize(theta, size);
                } else if (py::hasattr(yo, "__len__")) {
                    yo_v = vectorize(yo, size);
                    ro_v = vectorize(ro, size);
                    theta_v = vectorize(theta, size);
                    xo_v = vectorize(xo, size);
                } else if (py::hasattr(ro, "__len__")) {
                    ro_v = vectorize(ro, size);
                    theta_v = vectorize(theta, size);
                    xo_v = vectorize(xo, size);
                    yo_v = vectorize(yo, size);
                } else {
                    size = 1;
                    theta_v = vectorize(theta, size);
                    xo_v = vectorize(xo, size);
                    yo_v = vectorize(yo, size);
                    ro_v = vectorize(ro, size);
                }

                // Declare the result matrix
                Eigen::MatrixXd result(theta_v.size(), STARRY_NGRAD_MAP_FLUX + 1);

                // Declare our gradient types
                // TODO: This is how we should be defining active vectors!
                UnitVector<MapType> axis_g;
                axis_g(0).value() = axis(0);
                axis_g(0).derivatives() = Eigen::VectorXd::Unit(STARRY_NGRAD, 0);
                axis_g(1).value() = axis(1);
                axis_g(1).derivatives() = Eigen::VectorXd::Unit(STARRY_NGRAD, 1);
                axis_g(2).value() = axis(2);
                axis_g(2).derivatives() = Eigen::VectorXd::Unit(STARRY_NGRAD, 2);
                MapType theta_g(0., STARRY_NGRAD, 3);
                MapType xo_g(0., STARRY_NGRAD, 4);
                MapType yo_g(0., STARRY_NGRAD, 5);
                MapType ro_g(0., STARRY_NGRAD, 6);
                MapType tmp;

                /*
                TODO:
                I think this is how I would go about requesting derivs w/ respect
                to the map coeffs:

                    map.y(n).derivatives() = Eigen::VectorXd::Unit(STARRY_NGRAD, derNumber);

                */

                // Compute the flux at each cadence
                for (int i = 0; i < theta_v.size(); i++) {
                    theta_g.value() = theta_v(i);
                    xo_g.value() = xo_v(i);
                    yo_g.value() = yo_v(i);
                    ro_g.value() = ro_v(i);
                    tmp = map.flux(axis_g, theta_g, xo_g, yo_g, ro_g);
                    result(i, 0) = tmp.value();
                    result.block<1, STARRY_NGRAD_MAP_FLUX>(i, 1) = tmp.derivatives().head<STARRY_NGRAD_MAP_FLUX>();
                }
                return result;
            },
    #endif
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
            )pbdoc", "axis"_a=maps::yhat, "theta"_a=0, "xo"_a=0, "yo"_a=0, "ro"_a=0)

        .def("get_coeff",
    #ifndef STARRY_AUTODIFF
            &maps::Map<MapType>::get_coeff,
    #else
            [](maps::Map<MapType> &map, int l, int m){
                return map.get_coeff(l, m).value();
            },
    #endif
            R"pbdoc(
                Return the (:py:obj:`l`, :py:obj:`m`) coefficient of the map.

                .. note:: Users can also retrieve a coefficient by accessing the \
                          [:py:obj:`l`, :py:obj:`m`] index of the map as if it \
                          were a 2D array.

                Args:
                    l (int): The spherical harmonic degree, ranging from 0 to :py:attr:`lmax`.
                    m (int): The spherical harmonic order, ranging from -:py:obj:`l` to :py:attr:`l`.
            )pbdoc", "l"_a, "m"_a)

        .def("set_coeff",
    #ifndef STARRY_AUTODIFF
            &maps::Map<MapType>::set_coeff,
    #else
            [](maps::Map<MapType> &map, int l, int m, double coeff){
                map.set_coeff(l, m, MapType(coeff));
            },
    #endif
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

        .def("reset", &maps::Map<MapType>::reset,
            R"pbdoc(
                Set all of the map coefficients to zero.
            )pbdoc")

        .def_property_readonly("lmax", [](maps::Map<MapType> &map){return map.lmax;},
            R"pbdoc(
                The highest spherical harmonic order of the map. *Read-only.*
            )pbdoc")

        .def_property_readonly("y", [](maps::Map<MapType> &map){
            map.update(true);
    #ifndef STARRY_AUTODIFF
            return map.y;
    #else
            Eigen::VectorXd vec;
            vec.resize(map.N);
            for (int n = 0; n < map.N + 1; n++) {
                vec(n) = map.y(n).value();
            }
            return vec;
    #endif
        },
            R"pbdoc(
                The spherical harmonic map vector. *Read-only.*
            )pbdoc")

        .def_property_readonly("p", [](maps::Map<MapType> &map){
            map.update(true);
    #ifndef STARRY_AUTODIFF
            return map.p;
    #else
            Eigen::VectorXd vec;
            vec.resize(map.N);
            for (int n = 0; n < map.N + 1; n++) {
                vec(n) = map.p(n).value();
            }
            return vec;
    #endif
            },
            R"pbdoc(
                The polynomial map vector. *Read-only.*
            )pbdoc")

        .def_property_readonly("g", [](maps::Map<MapType> &map){
            map.update(true);
    #ifndef STARRY_AUTODIFF
            return map.g;
    #else
            Eigen::VectorXd vec;
            vec.resize(map.N);
            for (int n = 0; n < map.N + 1; n++) {
                vec(n) = map.g(n).value();
            }
            return vec;
    #endif
        },
            R"pbdoc(
                The Green's polynomial map vector. *Read-only.*
            )pbdoc")

        .def_property_readonly("s", [](maps::Map<MapType> &map){
    #ifndef STARRY_AUTODIFF
            return map.G.sT;
    #else
            Eigen::VectorXd vec;
            vec.resize(map.N);
            for (int n = 0; n < map.N + 1; n++) {
                vec(n) = map.G.sT(n).value();
            }
            return vec;
    #endif
        },
            R"pbdoc(
                The current solution vector `s`. *Read-only.*
            )pbdoc")

        .def_property_readonly("r", [](maps::Map<MapType> &map){
            return map.C.rT;
        },
            R"pbdoc(
                The current solution vector `r`. *Read-only.*
            )pbdoc")

    #ifndef STARRY_AUTODIFF
        // NOTE: No autograd implementation of this attribute.
        .def_property_readonly("s_mp", [](maps::Map<MapType> &map){
            VectorT<double> sT = map.mpG.sT.template cast<double>();
            return sT;
        },
            R"pbdoc(
                The current multi-precision solution vector `s`. Only available after `flux_mp` has been called. *Read-only.*
            )pbdoc")
    #endif

        .def("__setitem__", [](maps::Map<MapType>& map, py::object index, double coeff) {
            if (py::isinstance<py::tuple>(index)) {
                // This is a (l, m) tuple
                py::tuple lm = index;
                int l = py::cast<int>(lm[0]);
                int m = py::cast<int>(lm[1]);
                map.set_coeff(l, m, MapType(coeff));
            } else {
                throw errors::BadIndex();
            }
        })

        .def("__getitem__", [](maps::Map<MapType>& map, py::object index) -> py::object {
            if (py::isinstance<py::tuple>(index)) {
                // This is a (l, m) tuple
                py::tuple lm = index;
                int l = py::cast<int>(lm[0]);
                int m = py::cast<int>(lm[1]);
    #ifndef STARRY_AUTODIFF
                return py::cast(map.get_coeff(l, m));
    #else
                return py::cast(map.get_coeff(l, m).value());
    #endif
            } else {
                throw errors::BadIndex();
            }
        })

        .def("__repr__", [](maps::Map<MapType> &map) -> string {
    #ifndef STARRY_AUTODIFF
            return map.repr();
    #else
            int n = 0;
            int nterms = 0;
            char buf[30];
            double yn;
            ostringstream os;
            os << "<STARRY AutoDiff Map: ";
            for (int l = 0; l < map.lmax + 1; l++) {
                for (int m = -l; m < l + 1; m++) {
                    yn = map.y(n).value();
                    if (abs(yn) > STARRY_MAP_TOLERANCE) {
                        // Separator
                        if ((nterms > 0) && (yn > 0)) {
                            os << " + ";
                        } else if ((nterms > 0) && (yn < 0)) {
                            os << " - ";
                        } else if ((nterms == 0) && (yn < 0)) {
                            os << "-";
                        }
                        // Term
                        if ((yn == 1) || (yn == -1)) {
                            sprintf(buf, "Y_{%d,%d}", l, m);
                            os << buf;
                        } else if (fmod(abs(yn), 1.0) < STARRY_MAP_TOLERANCE) {
                            sprintf(buf, "%d Y_{%d,%d}", (int)abs(yn), l, m);
                            os << buf;
                        } else if (fmod(abs(yn), 1.0) >= 0.01) {
                            sprintf(buf, "%.2f Y_{%d,%d}", abs(yn), l, m);
                            os << buf;
                        } else {
                            sprintf(buf, "%.2e Y_{%d,%d}", abs(yn), l, m);
                            os << buf;
                        }
                        nterms++;
                    }
                    n++;
                }
            }
            if (nterms == 0)
                os << "Null";
            os << ">";
            return std::string(os.str());
    #endif
        })

        //
        // This is where things go nuts: Let's call Python from C++
        //

        .def("minimum", [](maps::Map<MapType> &map) -> double {
            map.update();
            py::object minimize = py::module::import("starry_maps").attr("minimize");
    #ifndef STARRY_AUTODIFF
            return minimize(map.p).cast<double>();
    #else
            Eigen::VectorXd vec;
            vec.resize(map.N);
            for (int n = 0; n < map.N + 1; n++) {
                vec(n) = map.p(n).value();
            }
            return minimize(vec).cast<double>();
    #endif
        },
        R"pbdoc(
            Find the global minimum of the map.

            This routine wraps :py:class:`scipy.optimize.minimize` to find
            the global minimum of the surface map. This is useful for ensuring
            that the surface map is nonnegative everywhere.

            .. note:: Because this routine wraps a Python wrapper of a C function \
                      to perform a non-linear optimization in three dimensions, it is \
                      **slow** and should probably not be used repeatedly when fitting \
                      a map to data!
        )pbdoc")

        /*
        // TODO: I need to give this function a little more thought.
        .def("random", [] (maps::Map<MapType>& map, double beta=0, bool nonnegative=true) {
            py::object minimize = py::module::import("starry_maps").attr("minimize");
            double minval, c00;

            // Generate random coefficients
            map.random(beta);

            // Ensure non-negative
            if (nonnegative) {
                map.update();
                c00 = map.get_coeff(0, 0);
                minval = minimize(map.p).cast<double>();
                if (minval < 0)
                    map.set_coeff(0, 0, c00 - sqrt(4 * M_PI) * minval);
            }
        },
        R"pbdoc(
            Generate a random map with a power spectrum given by the power law index `beta`.

            Args:
                beta (float): Power law index. The sum of the squares of all the coefficients at \
                              degree `l` is proportional to `l ** beta`. Default 0 (white spectrum).
                nonnegative (bool): Force map to be non-negative everywhere? Default :py:obj:`True`.
            )pbdoc", "beta"_a=0., "nonnegative"_a=true)
        */

        .def("load_image", [](maps::Map<MapType> &map, string& image) {
            py::object load_map = py::module::import("starry_maps").attr("load_map");
            Vector<double> y = load_map(image, map.lmax).cast<Vector<double>>();
            double y_normed;
            int n = 0;
            for (int l = 0; l < map.lmax + 1; l++) {
                for (int m = -l; m < l + 1; m++) {
                    y_normed = y(n) / y(0);
                    map.set_coeff(l, m, MapType(y_normed));
                    n++;
                }
            }
            // We need to apply some rotations to get
            // to the desired orientation
            UnitVector<MapType> xhat(maps::xhat);
            UnitVector<MapType> yhat(maps::yhat);
            UnitVector<MapType> zhat(maps::zhat);
            MapType Pi(M_PI);
            MapType PiOver2(M_PI / 2.);
            map.rotate(xhat, PiOver2);
            map.rotate(zhat, Pi);
            map.rotate(yhat, PiOver2);
        },
        R"pbdoc(
            Load an image from file.

            This routine loads an image file, computes its spherical harmonic
            expansion up to degree :py:attr:`lmax`, and sets the map vector.

            Args:
                image (str): The full path to the image file.

            .. todo:: The map is currently unnormalized; the max/min will depend \
                      on the colorscale of the input image. This will be fixed \
                      soon.

        )pbdoc", "image"_a)

        .def("load_healpix", [](maps::Map<MapType> &map, Matrix<double>& image) {
            py::object load_map = py::module::import("starry_maps").attr("load_map");
            Vector<double> y = load_map(image, map.lmax).cast<Vector<double>>();
            double y_normed;
            int n = 0;
            for (int l = 0; l < map.lmax + 1; l++) {
                for (int m = -l; m < l + 1; m++) {
                    y_normed = y(n) / y(0);
                    map.set_coeff(l, m, MapType(y_normed));
                    n++;
                }
            }
            // We need to apply some rotations to get
            // to the desired orientation
            UnitVector<MapType> xhat(maps::xhat);
            UnitVector<MapType> yhat(maps::yhat);
            UnitVector<MapType> zhat(maps::zhat);
            MapType Pi(M_PI);
            MapType PiOver2(M_PI / 2.);
            map.rotate(xhat, PiOver2);
            map.rotate(zhat, Pi);
            map.rotate(yhat, PiOver2);
        },
        R"pbdoc(
            Load a healpix image array.

            This routine loads a :py:obj:`healpix` array, computes its
            spherical harmonic
            expansion up to degree :py:attr:`lmax`, and sets the map vector.

            Args:
                image (ndarray): The ring-ordered :py:obj:`healpix` array.
        )pbdoc", "image"_a)

        .def("show", [](maps::Map<MapType> &map, string cmap, int res) {
            py::object show = py::module::import("starry_maps").attr("show");
            Matrix<double> I;
            I.resize(res, res);
            Vector<double> x;
            UnitVector<MapType> yhat(maps::yhat);
            x = Vector<double>::LinSpaced(res, -1, 1);
            MapType Zero = 0.;
    #ifdef STARRY_AUTODIFF
            MapType tmp1, tmp2;
    #endif
            for (int i = 0; i < res; i++){
                for (int j = 0; j < res; j++){
    #ifndef STARRY_AUTODIFF
                    I(j, i) = map.evaluate(yhat, Zero, x(i), x(j));
    #else
                    tmp1.value() = x(i);
                    tmp2.value() = x(j);
                    I(j, i) = map.evaluate(yhat, Zero, tmp1, tmp2).value();
    #endif
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

        .def("animate", [](maps::Map<MapType> &map, UnitVector<double>& axis, string cmap, int res, int frames) {
            std::cout << "Rendering animation..." << std::endl;
            py::object animate = py::module::import("starry_maps").attr("animate");
            vector<Matrix<double>> I;
            Vector<double> x, theta;
            x = Vector<double>::LinSpaced(res, -1, 1);
            theta = Vector<double>::LinSpaced(frames, 0, 2 * M_PI);
            UnitVector<MapType> MapType_axis(axis);
    #ifdef STARRY_AUTODIFF
            MapType tmp1, tmp2, tmp3;
    #endif
            for (int t = 0; t < frames; t++){
                I.push_back(Matrix<double>::Zero(res, res));
                for (int i = 0; i < res; i++){
                    for (int j = 0; j < res; j++){
    #ifndef STARRY_AUTODIFF
                        I[t](j, i) = map.evaluate(axis, theta(t), x(i), x(j));
    #else
                        tmp1.value() = theta(t);
                        tmp2.value() = x(i);
                        tmp3.value() = x(j);
                        I[t](j, i) = map.evaluate(MapType_axis, tmp1, tmp2, tmp3).value();
    #endif
                    }
                }
            }
            animate(I, axis, "cmap"_a=cmap, "res"_a=res);
        },
        R"pbdoc(
            Convenience routine to animate the body's surface map as it rotates.

            Args:
                axis (ndarray): *Normalized* unit vector specifying the axis of rotation. Default :math:`\hat{y} = (0, 1, 0)`.
                cmap (str): The :py:mod:`matplotlib` colormap name. Default `plasma`.
                res (int): The resolution of the map in pixels on a side. Default 150.
                frames (int): The number of frames in the animation. Default 50.
        )pbdoc", "axis"_a=maps::yhat, "cmap"_a="plasma", "res"_a=150, "frames"_a=50);



    // Limb-darkened surface map class
    py::class_<maps::LimbDarkenedMap<MapType>>(m, "LimbDarkenedMap", R"pbdoc(
            Instantiate a :py:mod:`starry` limb-darkened surface map.

            This differs from the base :py:class:`Map` class in that maps
            instantiated this way are radially symmetric: only the radial (`m = 0`)
            coefficients of the map are available. Users edit the map by directly
            specifying the polynomial limb darkening coefficients `u`.

            Args:
                lmax (int): Largest spherical harmonic degree in the surface map. Default 2.

            .. autoattribute:: optimize
            .. automethod:: evaluate(x=0, y=0)
            .. automethod:: flux_numerical(xo=0, yo=0, ro=0, tol=1.e-4)pbdoc"
    #ifndef STARRY_AUTODIFF
            R"pbdoc(
            .. automethod:: flux_mp(xo=0, yo=0, ro=0))pbdoc"
    #endif
            R"pbdoc(
            .. automethod:: flux(xo=0, yo=0, ro=0)
            .. automethod:: get_coeff(l)
            .. automethod:: set_coeff(l, coeff)
            .. automethod:: reset()
            .. autoattribute:: lmax
            .. autoattribute:: y
            .. autoattribute:: p
            .. autoattribute:: g
            .. autoattribute:: u
            .. autoattribute:: s)pbdoc"
    #ifndef STARRY_AUTODIFF
            R"pbdoc(
            .. autoattribute:: s_mp)pbdoc"
    #endif
            R"pbdoc(
            .. automethod:: show(cmap='plasma', res=300)

        )pbdoc")

        .def(py::init<int>(), "lmax"_a=2)

        .def_property("optimize", [](maps::LimbDarkenedMap<double> &map){return map.G.taylor;},
                                  [](maps::LimbDarkenedMap<double> &map, bool taylor){map.G.taylor = taylor;},
            R"pbdoc(
                Set to :py:obj:`False` to disable Taylor expansions of the primitive integrals when \
                computing occultation light curves. This is in general not something you should do! \
                Default :py:obj:`True`.
            )pbdoc")

        .def("evaluate",
    #ifndef STARRY_AUTODIFF
            py::vectorize(&maps::LimbDarkenedMap<MapType>::evaluate),
    #else
            [](maps::LimbDarkenedMap<MapType>& map, py::object x, py::object y){
                // Vectorize the inputs
                int size = 0;
                Eigen::VectorXd x_v, y_v;
                if (py::hasattr(x, "__len__")) {
                    x_v = vectorize(x, size);
                    y_v = vectorize(y, size);
                } else if (py::hasattr(y, "__len__")) {
                    y_v = vectorize(y, size);
                    x_v = vectorize(x, size);
                } else {
                    size = 1;
                    x_v = vectorize(x, size);
                    y_v = vectorize(y, size);
                }

                // Declare the result matrix
                Eigen::MatrixXd result(x_v.size(), STARRY_NGRAD_LDMAP_EVALUATE + 1);

                // Declare our gradient types
                MapType x_g(0., STARRY_NGRAD, 0);
                MapType y_g(0., STARRY_NGRAD, 1);
                MapType tmp;

                // Compute the flux at each cadence
                for (int i = 0; i < x_v.size(); i++) {
                    x_g.value() = x_v(i);
                    y_g.value() = y_v(i);
                    tmp = map.evaluate(x_g, y_g);
                    result(i, 0) = tmp.value();
                    result.block<1, STARRY_NGRAD_LDMAP_EVALUATE>(i, 1) = tmp.derivatives().head<STARRY_NGRAD_LDMAP_EVALUATE>();
                }

                return result;
            },
    #endif
            R"pbdoc(
                Return the specific intensity at a point (`x`, `y`) on the map.

                Users may optionally provide a rotation state. Note that this does
                not rotate the base map.

                Args:
                    x (float or ndarray): Position scalar, vector, or matrix.
                    y (float or ndarray): Position scalar, vector, or matrix.

                Returns:
                    The specific intensity at (`x`, `y`).
            )pbdoc", "x"_a=0, "y"_a=0)

        .def("flux_numerical",
    #ifndef STARRY_AUTODIFF
            py::vectorize(&maps::LimbDarkenedMap<MapType>::flux_numerical),
    #else
            [](maps::LimbDarkenedMap<MapType> &map, py::object xo, py::object yo, py::object ro, double tol){
                // Vectorize the inputs
                int size = 0;
                Eigen::VectorXd xo_v, yo_v, ro_v;
                if (py::hasattr(xo, "__len__")) {
                    xo_v = vectorize(xo, size);
                    yo_v = vectorize(yo, size);
                    ro_v = vectorize(ro, size);
                } else if (py::hasattr(yo, "__len__")) {
                    yo_v = vectorize(yo, size);
                    ro_v = vectorize(ro, size);
                    xo_v = vectorize(xo, size);
                } else if (py::hasattr(ro, "__len__")) {
                    ro_v = vectorize(ro, size);
                    xo_v = vectorize(xo, size);
                    yo_v = vectorize(yo, size);
                } else {
                    size = 1;
                    xo_v = vectorize(xo, size);
                    yo_v = vectorize(yo, size);
                    ro_v = vectorize(ro, size);
                }

                // Declare the result vector
                Eigen::VectorXd result(xo_v.size());

                // Declare our gradient types, although I simply *will not*
                // take automatic derivatives of a numerically computed function!
                MapType xo_g(0.);
                MapType yo_g(0.);
                MapType ro_g(0.);
                MapType tmp;

                // Compute the flux at each cadence
                for (int i = 0; i < xo_v.size(); i++) {
                    xo_g.value() = xo_v(i);
                    yo_g.value() = yo_v(i);
                    ro_g.value() = ro_v(i);
                    tmp = map.flux_numerical(xo_g, yo_g, ro_g, tol);
                    result(i) = tmp.value();
                }
                return result;
            },
    #endif
            R"pbdoc(
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
                )pbdoc"
    #ifndef STARRY_AUTODIFF
                R"pbdoc()pbdoc",
    #else
                R"pbdoc(
                .. note:: This function only returns the **value** of the numerical flux, and **not** its \
                          derivatives. Autodifferentiation of numerical integration is \
                          simply a terrible idea!
                )pbdoc",
    #endif
            "xo"_a=0, "yo"_a=0, "ro"_a=0, "tol"_a=1e-4)

    #ifndef STARRY_AUTODIFF
        // NOTE: No autograd implementation of this function.
        .def("flux_mp", py::vectorize(&maps::LimbDarkenedMap<MapType>::flux_mp),
            R"pbdoc(
                Return the total flux received by the observer, computed using multi-precision.

                Computes the total flux received by the observer from the
                map during or outside of an occultation. By default, this method
                performs all occultation calculations using 128-bit (quadruple) floating point
                precision, corresponding to 32 significant digits. Users can increase this to any
                number of digits (RAM permitting) by setting the :py:obj:`STARRY_MP_DIGITS=XX` flag
                at compile time. Note, importantly, that run times are **much** slower for multi-precision
                calculations.

                Args:
                    xo (float or ndarray): The `x` position of the occultor (if any). Default 0.
                    yo (float or ndarray): The `y` position of the occultor (if any). Default 0.
                    ro (float): The radius of the occultor in units of this body's radius. Default 0 (no occultation).

                Returns:
                    The flux received by the observer (a scalar or a vector).
            )pbdoc", "xo"_a=0, "yo"_a=0, "ro"_a=0)
    #endif

        .def("flux",
    #ifndef STARRY_AUTODIFF
            py::vectorize(&maps::LimbDarkenedMap<MapType>::flux),
    #else
            [](maps::LimbDarkenedMap<MapType> &map, py::object xo, py::object yo, py::object ro){
                // Vectorize the inputs
                int size = 0;
                Eigen::VectorXd xo_v, yo_v, ro_v;
                if (py::hasattr(xo, "__len__")) {
                    xo_v = vectorize(xo, size);
                    yo_v = vectorize(yo, size);
                    ro_v = vectorize(ro, size);
                } else if (py::hasattr(yo, "__len__")) {
                    yo_v = vectorize(yo, size);
                    ro_v = vectorize(ro, size);
                    xo_v = vectorize(xo, size);
                } else if (py::hasattr(ro, "__len__")) {
                    ro_v = vectorize(ro, size);
                    xo_v = vectorize(xo, size);
                    yo_v = vectorize(yo, size);
                } else {
                    size = 1;
                    xo_v = vectorize(xo, size);
                    yo_v = vectorize(yo, size);
                    ro_v = vectorize(ro, size);
                }

                // Declare the result matrix
                Eigen::MatrixXd result(xo_v.size(), STARRY_NGRAD_LDMAP_FLUX + 1);

                // Declare our gradient types
                MapType xo_g(0., STARRY_NGRAD, 0);
                MapType yo_g(0., STARRY_NGRAD, 1);
                MapType ro_g(0., STARRY_NGRAD, 2);
                MapType tmp;

                // Compute the flux at each cadence
                for (int i = 0; i < xo_v.size(); i++) {
                    xo_g.value() = xo_v(i);
                    yo_g.value() = yo_v(i);
                    ro_g.value() = ro_v(i);
                    tmp = map.flux(xo_g, yo_g, ro_g);
                    result(i, 0) = tmp.value();
                    result.block<1, STARRY_NGRAD_LDMAP_FLUX>(i, 1) = tmp.derivatives().head<STARRY_NGRAD_LDMAP_FLUX>();
                }

                return result;
            },
    #endif
            R"pbdoc(
                Return the total flux received by the observer.

                Computes the total flux received by the observer from the
                map during or outside of an occultation.

                Args:
                    xo (float or ndarray): The `x` position of the occultor (if any). Default 0.
                    yo (float or ndarray): The `y` position of the occultor (if any). Default 0.
                    ro (float): The radius of the occultor in units of this body's radius. Default 0 (no occultation).

                Returns:
                    The flux received by the observer (a scalar or a vector).
            )pbdoc", "xo"_a=0, "yo"_a=0, "ro"_a=0)

        .def("get_coeff",
    #ifndef STARRY_AUTODIFF
            &maps::LimbDarkenedMap<MapType>::get_coeff,
    #else
            [](maps::LimbDarkenedMap<MapType> &map, int l){
                return map.get_coeff(l).value();
            },
    #endif
            R"pbdoc(
                Return the limb darkening coefficient of order :py:obj:`l`.

                .. note:: Users can also retrieve a coefficient by accessing the \
                          [:py:obj:`l`] index of the map as if it were an array.

                Args:
                    l (int): The limb darkening order (> 0).
            )pbdoc", "l"_a)

        .def("set_coeff",
    #ifndef STARRY_AUTODIFF
            &maps::LimbDarkenedMap<MapType>::set_coeff,
    #else
            [](maps::LimbDarkenedMap<MapType> &map, int l, double coeff){
                map.set_coeff(l, MapType(coeff));
            },
    #endif
            R"pbdoc(
                Set the limb darkening coefficient of order :py:obj:`l`.

                .. note:: Users can also set a coefficient by setting the \
                          [:py:obj:`l`] index of the map as if it \
                          were an array.

                Args:
                    l (int): The limb darkening order (> 0).
                    coeff (float): The value of the coefficient.
            )pbdoc", "l"_a, "coeff"_a)

        .def("reset", &maps::LimbDarkenedMap<MapType>::reset,
            R"pbdoc(
                Set all of the map coefficients to zero.
            )pbdoc")

        .def_property_readonly("lmax", [](maps::LimbDarkenedMap<MapType> &map){return map.lmax;},
            R"pbdoc(
                The highest spherical harmonic order of the map. *Read-only.*
            )pbdoc")

        .def_property_readonly("y", [](maps::LimbDarkenedMap<MapType> &map){
            map.update(true);
    #ifndef STARRY_AUTODIFF
            return map.y;
    #else
            Eigen::VectorXd vec;
            vec.resize(map.N);
            for (int n = 0; n < map.N + 1; n++) {
                vec(n) = map.y(n).value();
            }
            return vec;
    #endif
        },
            R"pbdoc(
                The spherical harmonic map vector. *Read-only.*
            )pbdoc")

        .def_property_readonly("p", [](maps::LimbDarkenedMap<MapType> &map){
            map.update(true);
    #ifndef STARRY_AUTODIFF
            return map.p;
    #else
            Eigen::VectorXd vec;
            vec.resize(map.N);
            for (int n = 0; n < map.N + 1; n++) {
                vec(n) = map.p(n).value();
            }
            return vec;
    #endif
            },
            R"pbdoc(
                The polynomial map vector. *Read-only.*
            )pbdoc")

        .def_property_readonly("g", [](maps::LimbDarkenedMap<MapType> &map){
            map.update(true);
    #ifndef STARRY_AUTODIFF
            return map.g;
    #else
            Eigen::VectorXd vec;
            vec.resize(map.N);
            for (int n = 0; n < map.N + 1; n++) {
                vec(n) = map.g(n).value();
            }
            return vec;
    #endif
        },
            R"pbdoc(
                The Green's polynomial map vector. *Read-only.*
            )pbdoc")

        .def_property_readonly("s", [](maps::LimbDarkenedMap<MapType> &map){
    #ifndef STARRY_AUTODIFF
            return map.G.sT;
    #else
            Eigen::VectorXd vec;
            vec.resize(map.N);
            for (int n = 0; n < map.N + 1; n++) {
                vec(n) = map.G.sT(n).value();
            }
            return vec;
    #endif
        },
            R"pbdoc(
                The current solution vector `s`. *Read-only.*
            )pbdoc")

    #ifndef STARRY_AUTODIFF
        // NOTE: No autograd implementation of this attribute.
        .def_property_readonly("s_mp", [](maps::LimbDarkenedMap<MapType> &map){
            VectorT<double> sT = map.mpG.sT.template cast<double>();
            return sT;
        },
            R"pbdoc(
                The current multi-precision solution vector `s`. Only available after `flux_mp` has been called. *Read-only.*
            )pbdoc")
    #endif

        .def_property_readonly("u", [](maps::LimbDarkenedMap<MapType> &map) {
    #ifndef STARRY_AUTODIFF
            return map.u;
    #else
            Eigen::VectorXd vec;
            vec.resize(map.lmax + 1);
            for (int n = 0; n < map.lmax + 1; n++) {
                vec(n) = map.u(n).value();
            }
            return vec;
    #endif
        },
            R"pbdoc(
                The limb darkening coefficients. *Read-only.*
            )pbdoc")

        .def("__setitem__", [](maps::LimbDarkenedMap<MapType>& map, py::object index, double coeff) {
            if (py::isinstance<py::tuple>(index)) {
                throw errors::BadIndex();
            } else {
                // This is a limb darkening index
                int l = py::cast<int>(index);
                map.set_coeff(l, MapType(coeff));
            }
        })

        .def("__getitem__", [](maps::LimbDarkenedMap<MapType>& map, py::object index) {
            if (py::isinstance<py::tuple>(index)) {
                throw errors::BadIndex();
            } else {
                int l = py::cast<int>(index);
    #ifndef STARRY_AUTODIFF
                return map.get_coeff(l);
    #else
                return map.get_coeff(l).value();
    #endif
            }
        })

        .def("__repr__", [](maps::LimbDarkenedMap<MapType> &map) -> string {
    #ifndef STARRY_AUTODIFF
            return map.repr();
    #else
            ostringstream os;
            os << "<STARRY AutoDiff LimbDarkenedMap: ";
            Eigen::VectorXd u;
            u.resize(map.lmax + 1);
            for (int n = 0; n < map.lmax + 1; n++)
                u(n) = map.u(n).value();
            os << u.transpose();
            os << ">";
            return std::string(os.str());
    #endif
        })

        //
        // This is where things go nuts: Let's call Python from C++
        //

        .def("show", [](maps::LimbDarkenedMap<MapType> &map, string cmap, int res) {
            py::object show = py::module::import("starry_maps").attr("show");
            Matrix<double> I;
            I.resize(res, res);
            Vector<double> x;
            x = Vector<double>::LinSpaced(res, -1, 1);
            for (int i = 0; i < res; i++){
                for (int j = 0; j < res; j++){
    #ifndef STARRY_AUTODIFF
                    I(j, i) = map.evaluate(x(i), x(j));
    #else
                    I(j, i) = map.evaluate(MapType(x(i)), MapType(x(j))).value();
    #endif
                }
            }
            show(I, "cmap"_a=cmap, "res"_a=res);
        },
        R"pbdoc(
            Convenience routine to quickly display the body's surface map.

            Args:
                cmap (str): The :py:mod:`matplotlib` colormap name. Default `plasma`.
                res (int): The resolution of the map in pixels on a side. Default 300.
        )pbdoc", "cmap"_a="plasma", "res"_a=300);

}
