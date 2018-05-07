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
#include "docstrings.h"
#include "vect.h"
#include "utils.h"

using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;
using namespace vect;


/**
Define our map type (double or AutoDiffScalar)
and some other type-specific stuff.

*/
#ifndef STARRY_AUTODIFF

#undef MAPTYPE
#define MAPTYPE                         double
#undef DOCS
#define DOCS                            docstrings
#undef ADD_MODULE
#define ADD_MODULE                      add_starry

#else

#undef MAPTYPE
#define MAPTYPE                         Eigen::AutoDiffScalar<Vector<double>>
#undef DOCS
#define DOCS                            docstrings_grad
#undef ADD_MODULE
#define ADD_MODULE                      add_starry_grad

#endif

/**
Instantiate the `starry` module with or without autodiff
*/
void ADD_MODULE(py::module &m) {

    m.doc() = DOCS::starry;

    // Surface map class
    py::class_<maps::Map<MAPTYPE>>(m, "Map", DOCS::map::map)

        .def(py::init<int>(), "lmax"_a=2)

        .def("__setitem__", [](maps::Map<MAPTYPE>& map, py::object index, double coeff) {
                if (py::isinstance<py::tuple>(index)) {
                    py::tuple lm = index;
                    int l = py::cast<int>(lm[0]);
                    int m = py::cast<int>(lm[1]);
                    map.set_coeff(l, m, MAPTYPE(coeff));
                } else {
                    throw errors::BadIndex();
                }
            })

        .def("__getitem__", [](maps::Map<MAPTYPE>& map, py::object index) -> py::object {
                if (py::isinstance<py::tuple>(index)) {
                    py::tuple lm = index;
                    int l = py::cast<int>(lm[0]);
                    int m = py::cast<int>(lm[1]);
                    return py::cast(get_value(map.get_coeff(l, m)));
                } else {
                    throw errors::BadIndex();
                }
            })

        .def("get_coeff", [](maps::Map<MAPTYPE> &map, int l, int m){
                return get_value(map.get_coeff(l, m));
            }, DOCS::map::get_coeff, "l"_a, "m"_a)

        .def("set_coeff", [](maps::Map<MAPTYPE> &map, int l, int m, double coeff){
                map.set_coeff(l, m, MAPTYPE(coeff));
            }, DOCS::map::set_coeff, "l"_a, "m"_a, "coeff"_a)

        .def("reset", &maps::Map<MAPTYPE>::reset, DOCS::map::reset)

        .def_property_readonly("lmax", [](maps::Map<MAPTYPE> &map){return map.lmax;}, DOCS::map::lmax)

        .def_property_readonly("y", [](maps::Map<MAPTYPE> &map){
                map.update(true);
                return get_value(map.y);
            }, DOCS::map::y)

        .def_property_readonly("p", [](maps::Map<MAPTYPE> &map){
                map.update(true);
                return get_value(map.p);
            }, DOCS::map::p)

        .def_property_readonly("g", [](maps::Map<MAPTYPE> &map){
                map.update(true);
                return get_value(map.g);
            }, DOCS::map::g)

        .def_property_readonly("s", [](maps::Map<MAPTYPE> &map){
                map.update(true);
                return get_value((Vector<MAPTYPE>)map.G.sT);
            }, DOCS::map::s)

        .def_property_readonly("r", [](maps::Map<MAPTYPE> &map){
                map.update(true);
                return get_value((Vector<MAPTYPE>)map.C.rT);
            }, DOCS::map::r)

        .def_property("optimize", [](maps::Map<MAPTYPE> &map){return map.G.taylor;},
            [](maps::Map<MAPTYPE> &map, bool taylor){map.G.taylor = taylor;}, DOCS::map::optimize)

        .def("evaluate", [](maps::Map<MAPTYPE>& map, UnitVector<double>& axis, py::object& theta, py::object& x, py::object& y) {
                return vectorize(axis, theta, x, y, &maps::Map<MAPTYPE>::evaluate, map);
            }, DOCS::map::evaluate, "axis"_a=maps::yhat, "theta"_a=0, "x"_a=0, "y"_a=0)

        .def("flux", [](maps::Map<MAPTYPE>& map, UnitVector<double>& axis, py::object& theta, py::object& xo, py::object& yo, py::object& ro) {
                return vectorize(axis, theta, xo, yo, ro, &maps::Map<MAPTYPE>::flux, map);
            }, DOCS::map::flux, "axis"_a=maps::yhat, "theta"_a=0, "xo"_a=0, "yo"_a=0, "ro"_a=0)

        .def("rotate", [](maps::Map<MAPTYPE> &map, UnitVector<double>& axis, double theta){
                map.rotate(UnitVector<MAPTYPE>(axis), MAPTYPE(theta));
            }, DOCS::map::rotate, "axis"_a=maps::yhat, "theta"_a=0)

        //
        // This is where things go nuts: Let's call Python from C++
        //

        .def("minimum", [](maps::Map<MAPTYPE> &map) -> double {
                map.update();
                py::object minimize = py::module::import("starry_maps").attr("minimize");
                Vector<double> p = get_value(map.p);
                return minimize(p).cast<double>();
            }, DOCS::map::minimum)

        .def("load_image", [](maps::Map<MAPTYPE> &map, string& image) {
                py::object load_map = py::module::import("starry_maps").attr("load_map");
                Vector<double> y = load_map(image, map.lmax).cast<Vector<double>>();
                double y_normed;
                int n = 0;
                for (int l = 0; l < map.lmax + 1; l++) {
                    for (int m = -l; m < l + 1; m++) {
                        y_normed = y(n) / y(0);
                        map.set_coeff(l, m, MAPTYPE(y_normed));
                        n++;
                    }
                }
                // We need to apply some rotations to get
                // to the desired orientation
                UnitVector<MAPTYPE> xhat(maps::xhat);
                UnitVector<MAPTYPE> yhat(maps::yhat);
                UnitVector<MAPTYPE> zhat(maps::zhat);
                MAPTYPE Pi(M_PI);
                MAPTYPE PiOver2(M_PI / 2.);
                map.rotate(xhat, PiOver2);
                map.rotate(zhat, Pi);
                map.rotate(yhat, PiOver2);
            }, DOCS::map::load_image, "image"_a)

        .def("load_healpix", [](maps::Map<MAPTYPE> &map, Matrix<double>& image) {
                py::object load_map = py::module::import("starry_maps").attr("load_map");
                Vector<double> y = load_map(image, map.lmax).cast<Vector<double>>();
                double y_normed;
                int n = 0;
                for (int l = 0; l < map.lmax + 1; l++) {
                    for (int m = -l; m < l + 1; m++) {
                        y_normed = y(n) / y(0);
                        map.set_coeff(l, m, MAPTYPE(y_normed));
                        n++;
                    }
                }
                // We need to apply some rotations to get
                // to the desired orientation
                UnitVector<MAPTYPE> xhat(maps::xhat);
                UnitVector<MAPTYPE> yhat(maps::yhat);
                UnitVector<MAPTYPE> zhat(maps::zhat);
                MAPTYPE Pi(M_PI);
                MAPTYPE PiOver2(M_PI / 2.);
                map.rotate(xhat, PiOver2);
                map.rotate(zhat, Pi);
                map.rotate(yhat, PiOver2);
            }, DOCS::map::load_healpix, "image"_a)

        .def("show", [](maps::Map<MAPTYPE> &map, string cmap, int res) {
                py::object show = py::module::import("starry_maps").attr("show");
                Matrix<double> I;
                I.resize(res, res);
                Vector<double> x;
                UnitVector<MAPTYPE> yhat(maps::yhat);
                x = Vector<double>::LinSpaced(res, -1, 1);
                for (int i = 0; i < res; i++){
                    for (int j = 0; j < res; j++){
                        I(j, i) = get_value(map.evaluate(yhat, MAPTYPE(0), MAPTYPE(x(i)), MAPTYPE(x(j))));
                    }
                }
                show(I, "cmap"_a=cmap, "res"_a=res);
            }, DOCS::map::show, "cmap"_a="plasma", "res"_a=300)

        .def("animate", [](maps::Map<MAPTYPE> &map, UnitVector<double>& axis, string cmap, int res, int frames) {
            std::cout << "Rendering animation..." << std::endl;
            py::object animate = py::module::import("starry_maps").attr("animate");
            vector<Matrix<double>> I;
            Vector<double> x, theta;
            x = Vector<double>::LinSpaced(res, -1, 1);
            theta = Vector<double>::LinSpaced(frames, 0, 2 * M_PI);
            UnitVector<MAPTYPE> MapType_axis(axis);
            for (int t = 0; t < frames; t++){
                I.push_back(Matrix<double>::Zero(res, res));
                for (int i = 0; i < res; i++){
                    for (int j = 0; j < res; j++){
                        I[t](j, i) = get_value(map.evaluate(axis, MAPTYPE(theta(t)), MAPTYPE(x(i)), MAPTYPE(x(j))));
                    }
                }
            }
            animate(I, axis, "cmap"_a=cmap, "res"_a=res);
        }, DOCS::map::animate, "axis"_a=maps::yhat, "cmap"_a="plasma", "res"_a=150, "frames"_a=50)

#ifndef STARRY_AUTODIFF

        // Methods and attributes only in `starry.Map()``

        .def_property_readonly("s_mp", [](maps::Map<MAPTYPE> &map){
                VectorT<double> sT = map.mpG.sT.template cast<double>();
                return sT;
            }, DOCS::map::s_mp)

        .def("flux_mp", [](maps::Map<MAPTYPE>& map, UnitVector<double>& axis, py::object& theta, py::object& xo, py::object& yo, py::object& ro) {
                return vectorize(axis, theta, xo, yo, ro, &maps::Map<MAPTYPE>::flux_mp, map);
            }, DOCS::map::flux, "axis"_a=maps::yhat, "theta"_a=0, "xo"_a=0, "yo"_a=0, "ro"_a=0)

        .def("flux_numerical", [](maps::Map<MAPTYPE>& map, UnitVector<double>& axis, py::object& theta, py::object& xo, py::object& yo, py::object& ro, double tol) {
                return vectorize(axis, theta, xo, yo, ro, tol, &maps::Map<MAPTYPE>::flux_numerical, map);
            }, DOCS::map::flux_numerical, "axis"_a=maps::yhat, "theta"_a=0, "xo"_a=0, "yo"_a=0, "ro"_a=0, "tol"_a=1e-4)

#else

        // Methods and attributes only in `starry.grad.Map()`

        .def_property("map_gradients", [](maps::Map<MAPTYPE> &map){return map.map_gradients;},
            [](maps::Map<MAPTYPE> &map, bool map_gradients){map.map_gradients = map_gradients;}, DOCS::map::map_gradients)

#endif

        .def("__repr__", [](maps::Map<MAPTYPE> &map) -> string {return map.repr();});


/* DEBUG --->

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

<-- DEBUG */



/** DEBUG --->

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

<--- DEBUG */

}
