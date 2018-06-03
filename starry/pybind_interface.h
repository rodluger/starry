/**
This defines the main Python interface to the code.
Note that this file is #include'd several times,
once for each variable type (double, AutoDiffScalar, ...)

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
#include "errors.h"

using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;
using namespace vect;

/**
Define our map type and some other type-specific stuff.
*/
#if MODULE == MODULE_STARRY

#undef MAPTYPE
#define MAPTYPE                         double
#undef DOCS
#define DOCS                            docstrings
#undef ADD_MODULE
#define ADD_MODULE                      add_starry

#elif MODULE == MODULE_STARRY_GRAD

#undef MAPTYPE
#define MAPTYPE                         Grad
#undef DOCS
#define DOCS                            docstrings_grad
#undef ADD_MODULE
#define ADD_MODULE                      add_starry_grad

#endif

/**
Instantiate the `starry` module
*/
void ADD_MODULE(py::module &m) {

    m.doc() = DOCS::starry;

    // Surface map class
    py::class_<maps::Map<MAPTYPE>>(m, "Map", DOCS::Map::Map)

        .def(py::init<int>(), "lmax"_a=2)

        .def("__setitem__", [](maps::Map<MAPTYPE>& map, py::object index, py::object& coeff) {
                if (py::isinstance<py::tuple>(index)) {
                    // User provided a (l, m) tuple
                    py::tuple lm = index;
                    int l, m;
                    double value = py::cast<double>(coeff);
                    try {
                        l = py::cast<int>(lm[0]);
                        m = py::cast<int>(lm[1]);
                    } catch (const char* msg) {
                        throw errors::BadLMIndex();
                    }
                    map.set_coeff(l, m, MAPTYPE(value));
                } else if (py::isinstance<py::slice>(index)) {
                    // User provided a slice of some sort
                    size_t start, stop, step, slicelength;
                    Vector<double> values;
                    py::slice slice = py::cast<py::slice>(index);
                    if(!slice.compute(map.N, &start, &stop, &step, &slicelength))
                        throw pybind11::error_already_set();
                    if (py::isinstance<py::float_>(coeff) || py::isinstance<py::int_>(coeff)) {
                        // Set all indices to the same value
                        values = Vector<double>::Constant(slicelength, py::cast<double>(coeff));
                    } else {
                        // Set the indices to a vector of values
                        values = py::cast<Vector<double>>(coeff);
                        if (values.size() != slicelength)
                            throw errors::BadSliceLength();
                    }
                    for (size_t i = 0; i < slicelength; ++i) {
                        map.y(start) = MAPTYPE(values(i));
                        start += step;
                    }
                    map.update();
                } else {
                    throw errors::BadIndex();
                }
            })

        .def("__getitem__", [](maps::Map<MAPTYPE>& map, py::object index) -> py::object {
                if (py::isinstance<py::tuple>(index)) {
                    py::tuple lm = index;
                    int l, m;
                    try {
                        l = py::cast<int>(lm[0]);
                        m = py::cast<int>(lm[1]);
                    } catch (const char* msg) {
                        throw errors::BadLMIndex();
                    }
                    return py::cast(get_value(map.get_coeff(l, m)));
                } else if (py::isinstance<py::slice>(index)) {
                    // User provided a slice of some sort
                    size_t start, stop, step, slicelength;
                    Vector<double> values;
                    py::slice slice = py::cast<py::slice>(index);
                    if(!slice.compute(map.N, &start, &stop, &step, &slicelength))
                        throw pybind11::error_already_set();
                    Vector<double> res(slicelength);
                    for (size_t i = 0; i < slicelength; ++i) {
                        res[i] = get_value(map.y(start));
                        start += step;
                    }
                    return py::cast(res);
                } else {
                    throw errors::BadIndex();
                }
            })

        .def("get_coeff", [](maps::Map<MAPTYPE> &map, int l, int m){
                return get_value(map.get_coeff(l, m));
            }, DOCS::Map::get_coeff, "l"_a, "m"_a)

        .def("set_coeff", [](maps::Map<MAPTYPE> &map, int l, int m, double coeff){
                map.set_coeff(l, m, MAPTYPE(coeff));
            }, DOCS::Map::set_coeff, "l"_a, "m"_a, "coeff"_a)

        .def("reset", &maps::Map<MAPTYPE>::reset, DOCS::Map::reset)

        .def_property_readonly("mp_digits", [](maps::Map<MAPTYPE> &map){return STARRY_MP_DIGITS;}, DOCS::mp_digits)

        .def_property_readonly("lmax", [](maps::Map<MAPTYPE> &map){return map.lmax;}, DOCS::Map::lmax)

        .def_property_readonly("y", [](maps::Map<MAPTYPE> &map){
                return get_value(map.y);
            }, DOCS::Map::y)

        .def_property_readonly("p", [](maps::Map<MAPTYPE> &map){
                return get_value(map.p);
            }, DOCS::Map::p)

        .def_property_readonly("g", [](maps::Map<MAPTYPE> &map){
                return get_value(map.g);
            }, DOCS::Map::g)

        .def_property_readonly("s", [](maps::Map<MAPTYPE> &map){
                return get_value((Vector<MAPTYPE>)map.G.sT);
            }, DOCS::Map::s)

        .def_property_readonly("r", [](maps::Map<MAPTYPE> &map){
                return get_value((Vector<MAPTYPE>)map.C.rT);
            }, DOCS::Map::r)

        .def_property("optimize", [](maps::Map<MAPTYPE> &map){return map.G.taylor;},
            [](maps::Map<MAPTYPE> &map, bool taylor){map.G.taylor = taylor;}, DOCS::Map::optimize)

        .def("evaluate", [](maps::Map<MAPTYPE>& map, UnitVector<double>& axis, py::object& theta, py::object& x, py::object& y) {
                UnitVector<double> axis_norm = norm_unit(axis);
                return vectorize_map_evaluate(axis_norm, theta, x, y, map);
            }, DOCS::Map::evaluate, "axis"_a=maps::yhat, "theta"_a=0, "x"_a=0, "y"_a=0)

        .def("flux", [](maps::Map<MAPTYPE>& map, UnitVector<double>& axis, py::object& theta, py::object& xo, py::object& yo, py::object& ro) {
                UnitVector<double> axis_norm = norm_unit(axis);
                return vectorize_map_flux(axis_norm, theta, xo, yo, ro, map);
            }, DOCS::Map::flux, "axis"_a=maps::yhat, "theta"_a=0, "xo"_a=0, "yo"_a=0, "ro"_a=0)

        .def("rotate", [](maps::Map<MAPTYPE> &map, UnitVector<double>& axis, double theta){
                UnitVector<MAPTYPE> axis_norm = UnitVector<MAPTYPE>(norm_unit(axis));
                map.rotate(axis_norm, theta * DEGREE);
            }, DOCS::Map::rotate, "axis"_a=maps::yhat, "theta"_a=0)

        //
        // This is where things go nuts: Let's call Python from C++
        //

        .def("minimum", [](maps::Map<MAPTYPE> &map) -> double {
                py::object minimize = py::module::import("starry_maps").attr("minimize");
                Vector<double> p = get_value(map.p);
                return minimize(p).cast<double>();
            }, DOCS::Map::minimum)

        .def("load_array", [](maps::Map<MAPTYPE> &map, Matrix<double>& image) {
                py::object load_map = py::module::import("starry_maps").attr("load_map");
                Vector<double> y = load_map(image, map.lmax, false).cast<Vector<double>>();
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
            }, DOCS::Map::load_array, "image"_a)

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
            }, DOCS::Map::load_image, "image"_a)

        .def("load_healpix", [](maps::Map<MAPTYPE> &map, Matrix<double>& image) {
                py::object load_map = py::module::import("starry_maps").attr("load_map");
                Vector<double> y = load_map(image, map.lmax, true).cast<Vector<double>>();
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
            }, DOCS::Map::load_healpix, "image"_a)

        .def("add_gaussian", [](maps::Map<MAPTYPE> &map, double sigma, double amp, double lat, double lon) {
                py::object gaussian = py::module::import("starry_maps").attr("gaussian");
                Vector<double> y = gaussian(sigma, map.lmax).cast<Vector<double>>();
                int n = 0;
                // Create a temporary map and add the gaussian
                maps::Map<double> tmpmap(map.lmax);
                for (int l = 0; l < tmpmap.lmax + 1; l++) {
                    for (int m = -l; m < l + 1; m++) {
                        tmpmap.set_coeff(l, m, amp * y(n));
                        n++;
                    }
                }
                // Rotate it to the sub-observer point
                UnitVector<double> xhat(maps::xhat);
                UnitVector<double> yhat(maps::yhat);
                UnitVector<double> zhat(maps::zhat);
                tmpmap.rotate(xhat, M_PI / 2.);
                tmpmap.rotate(zhat, M_PI);
                tmpmap.rotate(yhat, M_PI / 2.);
                // Now rotate it to where the user wants it
                tmpmap.rotate(xhat, -lat * DEGREE);
                tmpmap.rotate(yhat, lon * DEGREE);
                // Add it to the current map
                for (int l = 0; l < map.lmax + 1; l++) {
                    for (int m = -l; m < l + 1; m++) {
                        map.set_coeff(l, m, get_value(map.get_coeff(l, m)) + tmpmap.get_coeff(l, m));
                    }
                }
            }, DOCS::Map::add_gaussian, "sigma"_a=0.1, "amp"_a=1, "lat"_a=0, "lon"_a=0)

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
            }, DOCS::Map::show, "cmap"_a="plasma", "res"_a=300)

        .def("animate", [](maps::Map<MAPTYPE> &map, UnitVector<double>& axis, string cmap, int res, int frames) {
            std::cout << "Rendering animation..." << std::endl;
            py::object animate = py::module::import("starry_maps").attr("animate");
            vector<Matrix<double>> I;
            Vector<double> x, theta;
            x = Vector<double>::LinSpaced(res, -1, 1);
            theta = Vector<double>::LinSpaced(frames, 0, 2 * M_PI);
            UnitVector<MAPTYPE> MapType_axis(norm_unit(axis));
            for (int t = 0; t < frames; t++){
                I.push_back(Matrix<double>::Zero(res, res));
                for (int i = 0; i < res; i++){
                    for (int j = 0; j < res; j++){
                        I[t](j, i) = get_value(map.evaluate(MapType_axis, MAPTYPE(theta(t)), MAPTYPE(x(i)), MAPTYPE(x(j))));
                    }
                }
            }
            animate(I, axis, "cmap"_a=cmap, "res"_a=res);
        }, DOCS::Map::animate, "axis"_a=maps::yhat, "cmap"_a="plasma", "res"_a=150, "frames"_a=50)

#if MODULE == MODULE_STARRY

        // Methods and attributes only in `starry.Map()``

        .def_property_readonly("s_mp", [](maps::Map<MAPTYPE> &map){
                VectorT<double> sT = map.mpG.sT.template cast<double>();
                return sT;
            }, DOCS::Map::s_mp)

        .def("flux_mp", [](maps::Map<MAPTYPE>& map, UnitVector<double>& axis, py::object& theta, py::object& xo, py::object& yo, py::object& ro) {
                UnitVector<double> axis_norm = norm_unit(axis);
                return vectorize_map_flux_mp(axis_norm, theta, xo, yo, ro, map);
            }, DOCS::Map::flux, "axis"_a=maps::yhat, "theta"_a=0, "xo"_a=0, "yo"_a=0, "ro"_a=0)

        .def("flux_numerical", [](maps::Map<MAPTYPE>& map, UnitVector<double>& axis, py::object& theta, py::object& xo, py::object& yo, py::object& ro, double tol) {
                UnitVector<double> axis_norm = norm_unit(axis);
                return vectorize_map_flux_numerical(axis_norm, theta, xo, yo, ro, tol, map);
            }, DOCS::Map::flux_numerical, "axis"_a=maps::yhat, "theta"_a=0, "xo"_a=0, "yo"_a=0, "ro"_a=0, "tol"_a=1e-4)

#elif MODULE == MODULE_STARRY_GRAD

        // Methods and attributes only in `starry.grad.Map()`

        .def_property_readonly("gradient", [](maps::Map<MAPTYPE> &map){
                return py::cast(map.derivs);
            }, DOCS::Map::gradient)

        .def_property_readonly("ngrad", [](maps::Map<MAPTYPE> &map){return STARRY_NGRAD;}, DOCS::ngrad)

#endif

        .def("__repr__", [](maps::Map<MAPTYPE> &map) -> string {return map.repr();});


    // Limb-darkened surface map class
    py::class_<maps::LimbDarkenedMap<MAPTYPE>>(m, "LimbDarkenedMap", DOCS::LimbDarkenedMap::LimbDarkenedMap)

        .def(py::init<int>(), "lmax"_a=2)

        .def("__setitem__", [](maps::LimbDarkenedMap<MAPTYPE>& map, py::object index, py::object& coeff) {
            if (py::isinstance<py::int_>(index)) {
                // User provided a single index
                int l = py::cast<int>(index);
                double value = py::cast<double>(coeff);
                map.set_coeff(l, MAPTYPE(value));
            } else if (py::isinstance<py::slice>(index)) {
                // User provided a slice of some sort
                size_t start, stop, step, slicelength;
                Vector<double> values;
                py::slice slice = py::cast<py::slice>(index);
                if(!slice.compute(map.lmax, &start, &stop, &step, &slicelength))
                    throw pybind11::error_already_set();
                if (py::isinstance<py::float_>(coeff) || py::isinstance<py::int_>(coeff)) {
                    // Set all indices to the same value
                    values = Vector<double>::Constant(slicelength, py::cast<double>(coeff));
                } else {
                    // Set the indices to a vector of values
                    values = py::cast<Vector<double>>(coeff);
                    if (values.size() != slicelength)
                        throw errors::BadSliceLength();
                }
                for (size_t i = 0; i < slicelength; ++i) {
                    map.u(start + 1) = MAPTYPE(values(i));
                    start += step;
                }
                map.update();
            } else {
                throw errors::BadIndex();
            }
        })

        .def("__getitem__", [](maps::LimbDarkenedMap<MAPTYPE>& map, py::object index) -> py::object {
                if (py::isinstance<py::int_>(index)) {
                    // User provided a single index
                    int l = py::cast<int>(index);
                    return py::cast(get_value(map.get_coeff(l)));
                } else if (py::isinstance<py::slice>(index)) {
                    // User provided a slice of some sort
                    size_t start, stop, step, slicelength;
                    Vector<double> values;
                    py::slice slice = py::cast<py::slice>(index);
                    if(!slice.compute(map.lmax, &start, &stop, &step, &slicelength))
                        throw pybind11::error_already_set();
                    Vector<double> res(slicelength);
                    for (size_t i = 0; i < slicelength; ++i) {
                        res[i] = get_value(map.u(start + 1));
                        start += step;
                    }
                    return py::cast(res);
                } else {
                    throw errors::BadIndex();
                }
        })

        .def("get_coeff", [](maps::LimbDarkenedMap<MAPTYPE> &map, int l){
                return get_value(map.get_coeff(l));
            }, DOCS::LimbDarkenedMap::get_coeff, "l"_a)

        .def("set_coeff", [](maps::LimbDarkenedMap<MAPTYPE> &map, int l, double coeff){
                map.set_coeff(l, MAPTYPE(coeff));
            }, DOCS::LimbDarkenedMap::set_coeff, "l"_a, "coeff"_a)

        .def("reset", &maps::LimbDarkenedMap<MAPTYPE>::reset, DOCS::LimbDarkenedMap::reset)

        .def("roots", &maps::LimbDarkenedMap<MAPTYPE>::roots, DOCS::LimbDarkenedMap::roots)

        .def_property_readonly("lmax", [](maps::LimbDarkenedMap<MAPTYPE> &map){return map.lmax;}, DOCS::LimbDarkenedMap::lmax)

        .def_property_readonly("mp_digits", [](maps::LimbDarkenedMap<MAPTYPE> &map){return STARRY_MP_DIGITS;}, DOCS::mp_digits)

        .def_property_readonly("y", [](maps::LimbDarkenedMap<MAPTYPE> &map){
                return get_value(map.y);
            }, DOCS::LimbDarkenedMap::y)

        .def_property_readonly("p", [](maps::LimbDarkenedMap<MAPTYPE> &map){
                return get_value(map.p);
            }, DOCS::LimbDarkenedMap::p)

        .def_property_readonly("g", [](maps::LimbDarkenedMap<MAPTYPE> &map){
                return get_value(map.g);
            }, DOCS::LimbDarkenedMap::g)

        .def_property_readonly("s", [](maps::LimbDarkenedMap<MAPTYPE> &map){
                return get_value((Vector<MAPTYPE>)map.G.sT);
            }, DOCS::LimbDarkenedMap::s)

        .def_property_readonly("u", [](maps::LimbDarkenedMap<MAPTYPE> &map){
                // Hide u_0, since it's never used!
                Vector<double> u(map.lmax);
                for (int i = 0; i < map.lmax; i++)
                    u(i) = get_value(map.u(i + 1));
                return u;
            }, DOCS::LimbDarkenedMap::u)

        .def_property("optimize", [](maps::LimbDarkenedMap<MAPTYPE> &map){return map.G.taylor;},
            [](maps::LimbDarkenedMap<MAPTYPE> &map, bool taylor){map.G.taylor = taylor;}, DOCS::LimbDarkenedMap::optimize)

        .def("evaluate", [](maps::LimbDarkenedMap<MAPTYPE>& map, py::object& x, py::object& y) {
                return vectorize_ldmap_evaluate(x, y, map);
            }, DOCS::LimbDarkenedMap::evaluate, "x"_a=0, "y"_a=0)

        .def("flux", [](maps::LimbDarkenedMap<MAPTYPE>& map, py::object& xo, py::object& yo, py::object& ro) {
                return vectorize_ldmap_flux(xo, yo, ro, map);
            }, DOCS::LimbDarkenedMap::flux, "xo"_a=0, "yo"_a=0, "ro"_a=0)

        .def("show", [](maps::LimbDarkenedMap<MAPTYPE> &map, string cmap, int res) {
                py::object show = py::module::import("starry_maps").attr("show");
                Matrix<double> I;
                I.resize(res, res);
                Vector<double> x;
                UnitVector<MAPTYPE> yhat(maps::yhat);
                x = Vector<double>::LinSpaced(res, -1, 1);
                for (int i = 0; i < res; i++){
                    for (int j = 0; j < res; j++){
                        I(j, i) = get_value(map.evaluate(MAPTYPE(x(i)), MAPTYPE(x(j))));
                    }
                }
                show(I, "cmap"_a=cmap, "res"_a=res);
            }, DOCS::LimbDarkenedMap::show, "cmap"_a="plasma", "res"_a=300)

#if MODULE == MODULE_STARRY

        // Methods and attributes only in `starry.LimbDarkenedMap()``

        .def_property_readonly("s_mp", [](maps::LimbDarkenedMap<MAPTYPE> &map){
                VectorT<double> sT = map.mpG.sT.template cast<double>();
                return sT;
            }, DOCS::LimbDarkenedMap::s_mp)

        .def("flux_mp", [](maps::LimbDarkenedMap<MAPTYPE>& map, py::object& xo, py::object& yo, py::object& ro) {
                return vectorize_ldmap_flux_mp(xo, yo, ro, map);
            }, DOCS::LimbDarkenedMap::flux, "xo"_a=0, "yo"_a=0, "ro"_a=0)

        .def("flux_numerical", [](maps::LimbDarkenedMap<MAPTYPE>& map, py::object& xo, py::object& yo, py::object& ro, double tol) {
                return vectorize_ldmap_flux_numerical(xo, yo, ro, tol, map);
            }, DOCS::LimbDarkenedMap::flux_numerical, "xo"_a=0, "yo"_a=0, "ro"_a=0, "tol"_a=1e-4)

#elif MODULE == MODULE_STARRY_GRAD

        // Methods and attributes only in `starry.grad.LimbDarkenedMap()`

        .def_property_readonly("gradient", [](maps::LimbDarkenedMap<MAPTYPE> &map){
                return py::cast(map.derivs);
            }, DOCS::LimbDarkenedMap::gradient)

        .def_property_readonly("ngrad", [](maps::LimbDarkenedMap<MAPTYPE> &map){return STARRY_NGRAD;}, DOCS::ngrad)

#endif

        .def("__repr__", [](maps::LimbDarkenedMap<MAPTYPE> &map) -> string {return map.repr();});


    // Orbital system class
    py::class_<orbital::System<MAPTYPE>>(m, "System", DOCS::System::System)

        .def(py::init<vector<orbital::Body<MAPTYPE>*>, double, double, int, double, double, int>(),
            "bodies"_a, "scale"_a=0, "kepler_tol"_a=1.0e-7, "kepler_max_iter"_a=100, "exposure_time"_a=0, "exposure_tol"_a=1e-8, "exposure_max_depth"_a=4)

        .def("compute", [](orbital::System<MAPTYPE> &system, Vector<double>& time){system.compute((Vector<MAPTYPE>)time);},
            DOCS::System::compute, "time"_a)

        .def_property_readonly("flux", [](orbital::System<MAPTYPE> &system){return get_value(system.flux);},
            DOCS::System::flux)

        .def_property("scale", [](orbital::System<MAPTYPE> &system){return CLIGHT / (system.clight * RSUN);},
            [](orbital::System<MAPTYPE> &system, double scale){
                if (scale == 0)
                    system.clight = INFINITY;
                else
                    system.clight = CLIGHT / (scale * RSUN);
            }, DOCS::System::scale)

        .def_property("kepler_tol", [](orbital::System<MAPTYPE> &system){return system.eps;},
            [](orbital::System<MAPTYPE> &system, double eps){system.eps = eps;}, DOCS::System::kepler_tol)

        .def_property("kepler_max_iter", [](orbital::System<MAPTYPE> &system){return system.maxiter;},
            [](orbital::System<MAPTYPE> &system, int maxiter){system.maxiter = maxiter;}, DOCS::System::kepler_max_iter)

        .def_property("exposure_time", [](orbital::System<MAPTYPE> &system){return system.exptime / DAY;},
            [](orbital::System<MAPTYPE> &system, double exptime){system.exptime = exptime * DAY;}, DOCS::System::exposure_time)

        .def_property("exposure_tol", [](orbital::System<MAPTYPE> &system){return system.exptol;},
            [](orbital::System<MAPTYPE> &system, double exptol){system.exptol = exptol;}, DOCS::System::exposure_tol)

        .def_property("exposure_max_depth", [](orbital::System<MAPTYPE> &system){return system.expmaxdepth;},
            [](orbital::System<MAPTYPE> &system, int expmaxdepth){system.expmaxdepth = expmaxdepth;}, DOCS::System::exposure_max_depth)

#if MODULE == MODULE_STARRY_GRAD

        .def_property_readonly("gradient", [](orbital::System<MAPTYPE> &system){return py::cast(system.derivs);},
            DOCS::System::gradient)

#endif

        .def("__repr__", [](orbital::System<MAPTYPE> &system) -> string {return system.repr();});

     // Body class (not user-facing, just a base class)
     py::class_<orbital::Body<MAPTYPE>> PyBody(m, "Body");

     PyBody.def(py::init<int, const double&, const double&,
                         Eigen::Matrix<double, 3, 1>&,
                         const double&, const double&,
                         const double&, const double&,
                         const double&, const double&,
                         const double&, const double&,
                         const double&, bool>(),
                         "lmax"_a, "r"_a, "L"_a, "axis"_a,
                         "prot"_a, "a"_a, "porb"_a,
                         "inc"_a, "ecc"_a, "w"_a, "Omega"_a,
                         "lambda0"_a, "tref"_a, "is_star"_a)

        // NOTE: & is necessary in the return statement so we pass a reference back to Python!
        .def_property_readonly("map", [](orbital::Body<MAPTYPE> &body){return &body.map;}, DOCS::Body::map)

        .def_property_readonly("flux", [](orbital::Body<MAPTYPE> &body){return get_value(body.flux);}, DOCS::Body::flux)

#if MODULE == MODULE_STARRY_GRAD

        .def_property_readonly("gradient", [](orbital::Body<MAPTYPE> &body){return py::cast(body.derivs);},
            DOCS::Body::gradient)

#endif

        .def_property_readonly("x", [](orbital::Body<MAPTYPE> &body){return get_value(body.x);}, DOCS::Body::x)

        .def_property_readonly("y", [](orbital::Body<MAPTYPE> &body){return get_value(body.y);}, DOCS::Body::y)

        .def_property_readonly("z", [](orbital::Body<MAPTYPE> &body){return get_value(body.z);}, DOCS::Body::z)

        .def_property("r", [](orbital::Body<MAPTYPE> &body){return get_value(body.r);},
            [](orbital::Body<MAPTYPE> &body, double r){body.r = r;}, DOCS::Body::r)

        .def_property("L", [](orbital::Body<MAPTYPE> &body){return get_value(body.L);},
            [](orbital::Body<MAPTYPE> &body, double L){body.L = L; body.reset();}, DOCS::Body::L)

        .def_property("axis", [](orbital::Body<MAPTYPE> &body){return get_value((Vector<MAPTYPE>)body.axis);},
            [](orbital::Body<MAPTYPE> &body, UnitVector<double> axis){body.axis = (Vector<MAPTYPE>)norm_unit(axis);}, DOCS::Body::axis)

        .def_property("prot", [](orbital::Body<MAPTYPE> &body){return get_value(body.prot) / DAY;},
            [](orbital::Body<MAPTYPE> &body, double prot){body.prot = prot * DAY; body.reset();}, DOCS::Body::prot)

        .def_property("a", [](orbital::Body<MAPTYPE> &body){return get_value(body.a);},
            [](orbital::Body<MAPTYPE> &body, double a){body.a = a; body.reset();}, DOCS::Body::a)

        .def_property("porb", [](orbital::Body<MAPTYPE> &body){return get_value(body.porb) / DAY;},
            [](orbital::Body<MAPTYPE> &body, double porb){body.porb = porb * DAY; body.reset();}, DOCS::Body::porb)

        .def_property("inc", [](orbital::Body<MAPTYPE> &body){return get_value(body.inc) / DEGREE;},
            [](orbital::Body<MAPTYPE> &body, double inc){body.inc = inc * DEGREE; body.reset();}, DOCS::Body::inc)

        .def_property("ecc", [](orbital::Body<MAPTYPE> &body){return get_value(body.ecc);},
            [](orbital::Body<MAPTYPE> &body, double ecc){body.ecc = ecc; body.reset();}, DOCS::Body::ecc)

        .def_property("w", [](orbital::Body<MAPTYPE> &body){return get_value(body.w) / DEGREE;},
            [](orbital::Body<MAPTYPE> &body, double w){body.w = w * DEGREE; body.reset();}, DOCS::Body::w)

        .def_property("Omega", [](orbital::Body<MAPTYPE> &body){return get_value(body.Omega) / DEGREE;},
            [](orbital::Body<MAPTYPE> &body, double Omega){body.Omega = Omega * DEGREE; body.reset();}, DOCS::Body::Omega)

        .def_property("lambda0", [](orbital::Body<MAPTYPE> &body){return get_value(body.lambda0) / DEGREE;},
            [](orbital::Body<MAPTYPE> &body, double lambda0){body.lambda0 = lambda0 * DEGREE; body.reset();}, DOCS::Body::lambda0)

        .def_property("tref", [](orbital::Body<MAPTYPE> &body){return get_value(body.tref) / DAY;},
            [](orbital::Body<MAPTYPE> &body, double tref){body.tref = tref * DAY;}, DOCS::Body::tref)

        .def("__repr__", [](orbital::Body<MAPTYPE> &body) -> string {return body.repr();});


    // Star class
    py::class_<orbital::Star<MAPTYPE>>(m, "Star", PyBody, DOCS::Star::Star)

        .def(py::init<int>(), "lmax"_a=2)

        // NOTE: & is necessary in the return statement so we pass a reference back to Python!
        .def_property_readonly("map", [](orbital::Body<MAPTYPE> &body){return &body.ldmap;}, DOCS::Star::map)

        .def_property_readonly("r", [](orbital::Star<MAPTYPE> &star){return 1.;}, DOCS::Star::r)

        .def_property_readonly("L", [](orbital::Star<MAPTYPE> &star){return 1.;}, DOCS::Star::L)

        .def_property_readonly("axis", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, DOCS::NotImplemented)

        .def_property_readonly("prot", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, DOCS::NotImplemented)

        .def_property_readonly("a", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, DOCS::NotImplemented)

        .def_property_readonly("porb", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, DOCS::NotImplemented)

        .def_property_readonly("inc", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, DOCS::NotImplemented)

        .def_property_readonly("ecc", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, DOCS::NotImplemented)

        .def_property_readonly("w", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, DOCS::NotImplemented)

        .def_property_readonly("Omega", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, DOCS::NotImplemented)

        .def_property_readonly("lambda0", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, DOCS::NotImplemented)

        .def_property_readonly("tref", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, DOCS::NotImplemented)

        .def("__setitem__", [](orbital::Star<MAPTYPE> &star, py::object index, py::object& coeff) {
            if (py::isinstance<py::int_>(index)) {
                // User provided a single index
                int l = py::cast<int>(index);
                double value = py::cast<double>(coeff);
                star.ldmap.set_coeff(l, MAPTYPE(value));
            } else if (py::isinstance<py::slice>(index)) {
                // User provided a slice of some sort
                size_t start, stop, step, slicelength;
                Vector<double> values;
                py::slice slice = py::cast<py::slice>(index);
                if(!slice.compute(star.ldmap.lmax, &start, &stop, &step, &slicelength))
                    throw pybind11::error_already_set();
                if (py::isinstance<py::float_>(coeff) || py::isinstance<py::int_>(coeff)) {
                    // Set all indices to the same value
                    values = Vector<double>::Constant(slicelength, py::cast<double>(coeff));
                } else {
                    // Set the indices to a vector of values
                    values = py::cast<Vector<double>>(coeff);
                    if (values.size() != slicelength)
                        throw errors::BadSliceLength();
                }
                for (size_t i = 0; i < slicelength; ++i) {
                    star.ldmap.u(start + 1) = MAPTYPE(values(i));
                    start += step;
                }
                star.ldmap.update();
            } else {
                throw errors::BadIndex();
            }
        })

        .def("__getitem__", [](orbital::Star<MAPTYPE> &star, py::object index) -> py::object {
                if (py::isinstance<py::int_>(index)) {
                    // User provided a single index
                    int l = py::cast<int>(index);
                    return py::cast(get_value(star.ldmap.get_coeff(l)));
                } else if (py::isinstance<py::slice>(index)) {
                    // User provided a slice of some sort
                    size_t start, stop, step, slicelength;
                    Vector<double> values;
                    py::slice slice = py::cast<py::slice>(index);
                    if(!slice.compute(star.ldmap.lmax, &start, &stop, &step, &slicelength))
                        throw pybind11::error_already_set();
                    Vector<double> res(slicelength);
                    for (size_t i = 0; i < slicelength; ++i) {
                        res[i] = get_value(star.ldmap.u(start + 1));
                        start += step;
                    }
                    return py::cast(res);
                } else {
                    throw errors::BadIndex();
                }
        })

        .def("__repr__", [](orbital::Star<MAPTYPE> &star) -> string {return star.repr();});


    // Planet class
    py::class_<orbital::Planet<MAPTYPE>>(m, "Planet", PyBody, DOCS::Planet::Planet)

        .def(py::init<int, const double&, const double&,
                      Eigen::Matrix<double, 3, 1>&,
                      const double&, const double&,
                      const double&, const double&,
                      const double&, const double&,
                      const double&, const double&,
                      const double&>(),
                      "lmax"_a=2, "r"_a=0.1, "L"_a=0., "axis"_a=maps::yhat,
                      "prot"_a=0, "a"_a=50., "porb"_a=1,
                      "inc"_a=90., "ecc"_a=0, "w"_a=90, "Omega"_a=0,
                      "lambda0"_a=90, "tref"_a=0)

        .def("__setitem__", [](orbital::Planet<MAPTYPE>& planet, py::object index, py::object& coeff) {
            if (py::isinstance<py::tuple>(index)) {
                // User provided a (l, m) tuple
                py::tuple lm = index;
                int l, m;
                double value = py::cast<double>(coeff);
                try {
                   l = py::cast<int>(lm[0]);
                   m = py::cast<int>(lm[1]);
                } catch (const char* msg) {
                   throw errors::BadLMIndex();
                }
                planet.map.set_coeff(l, m, MAPTYPE(value));
            } else if (py::isinstance<py::slice>(index)) {
                // User provided a slice of some sort
                size_t start, stop, step, slicelength;
                Vector<double> values;
                py::slice slice = py::cast<py::slice>(index);
                if(!slice.compute(planet.map.N, &start, &stop, &step, &slicelength))
                    throw pybind11::error_already_set();
                if (py::isinstance<py::float_>(coeff) || py::isinstance<py::int_>(coeff)) {
                    // Set all indices to the same value
                    values = Vector<double>::Constant(slicelength, py::cast<double>(coeff));
                } else {
                    // Set the indices to a vector of values
                    values = py::cast<Vector<double>>(coeff);
                    if (values.size() != slicelength)
                        throw errors::BadSliceLength();
                }
                for (size_t i = 0; i < slicelength; ++i) {
                    planet.map.y(start) = MAPTYPE(values(i));
                    start += step;
                }
                planet.map.update();
            } else {
                throw errors::BadIndex();
            }
        })

        .def("__getitem__", [](orbital::Planet<MAPTYPE>& planet, py::object index) -> py::object {
              if (py::isinstance<py::tuple>(index)) {
                  py::tuple lm = index;
                  int l, m;
                  try {
                      l = py::cast<int>(lm[0]);
                      m = py::cast<int>(lm[1]);
                  } catch (const char* msg) {
                      throw errors::BadLMIndex();
                  }
                  return py::cast(get_value(planet.map.get_coeff(l, m)));
              } else if (py::isinstance<py::slice>(index)) {
                  // User provided a slice of some sort
                  size_t start, stop, step, slicelength;
                  Vector<double> values;
                  py::slice slice = py::cast<py::slice>(index);
                  if(!slice.compute(planet.map.N, &start, &stop, &step, &slicelength))
                      throw pybind11::error_already_set();
                  Vector<double> res(slicelength);
                  for (size_t i = 0; i < slicelength; ++i) {
                      res[i] = get_value(planet.map.y(start));
                      start += step;
                  }
                  return py::cast(res);
              } else {
                  throw errors::BadIndex();
              }
          })

        .def("__repr__", [](orbital::Planet<MAPTYPE> &planet) -> string {return planet.repr();});

} // ADD_MODULE
