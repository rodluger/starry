/**
This defines the main Python interface to the code.

*/

#ifndef _STARRY_PYBIND_H_
#define _STARRY_PYBIND_H_

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

template <typename MAPTYPE>
void add_Map_extras(py::class_<maps::Map<MAPTYPE>>& PyMap, const docstrings::docs<MAPTYPE>& docs) { }

template <>
void add_Map_extras<double>(py::class_<maps::Map<double>>& PyMap, const docstrings::docs<double>& docs) {

    PyMap

        .def("_flux_numerical", [](maps::Map<double>& map, UnitVector<double>& axis, py::object& theta, py::object& xo, py::object& yo, py::object& ro, double tol) {
                UnitVector<double> axis_norm = norm_unit(axis);
                return vectorize_map_flux_numerical(axis_norm, theta, xo, yo, ro, tol, map);
            }, docs.Map.flux_numerical, "axis"_a=yhat, "theta"_a=0, "xo"_a=0, "yo"_a=0, "ro"_a=0, "tol"_a=1e-4);
}

template <>
void add_Map_extras<Grad>(py::class_<maps::Map<Grad>>& PyMap, const docstrings::docs<Grad>& docs) {

    PyMap

        .def_property_readonly("gradient", [](maps::Map<Grad> &map){
                return py::cast(map.derivs);
            }, docs.Map.gradient);
}

template <>
void add_Map_extras<Multi>(py::class_<maps::Map<Multi>>& PyMap, const docstrings::docs<Multi>& docs) {

    PyMap

        // Numerically computed derivative of the flux with respect to `yo`. Used
        // exclusively for stability tests of the double-precision derivatives in
        // `starry.grad`. Not user-facing, not documented.
        .def("_dfluxdyo", [](maps::Map<Multi>& map, UnitVector<double>& axis, py::object& theta, py::object& xo, py::object& yo, py::object& ro) {

                // Vectorize the inputs
                UnitVector<Multi> axis_v = norm_unit(axis).cast<Multi>();
                Vector<double> theta_v, xo_v, yo_v, ro_v;
                vectorize_args(theta, xo, yo, ro, theta_v, xo_v, yo_v, ro_v);

                // Step size
                Multi eps = sqrt(mach_eps<Multi>());

                // Compute the function for each vector index
                Multi f2, f1;
                Vector<double> deriv(theta_v.size());
                for (int i = 0; i < theta_v.size(); i++) {
                    f2 = map.flux(axis_v, theta_v(i) * DEGREE, xo_v(i), Multi(yo_v(i) + eps), ro_v(i));
                    f1 = map.flux(axis_v, theta_v(i) * DEGREE, xo_v(i), Multi(yo_v(i) - eps), ro_v(i));
                    deriv(i) = (double)((f2 - f1) / (2 * eps));
                }
                return deriv;

            }, "axis"_a=yhat, "theta"_a=0, "xo"_a=0, "yo"_a=0, "ro"_a=0)

        // Numerically computed derivative of the flux with respect to `ro`. Used
        // exclusively for stability tests of the double-precision derivatives in
        // `starry.grad`. Not user-facing, not documented.
        .def("_dfluxdro", [](maps::Map<Multi>& map, UnitVector<double>& axis, py::object& theta, py::object& xo, py::object& yo, py::object& ro) {

                // Vectorize the inputs
                UnitVector<Multi> axis_v = norm_unit(axis).cast<Multi>();
                Vector<double> theta_v, xo_v, yo_v, ro_v;
                vectorize_args(theta, xo, yo, ro, theta_v, xo_v, yo_v, ro_v);

                // Step size
                Multi eps = sqrt(mach_eps<Multi>());

                // Compute the function for each vector index
                Multi f2, f1;
                Vector<double> deriv(theta_v.size());
                for (int i = 0; i < theta_v.size(); i++) {
                    f2 = map.flux(axis_v, theta_v(i) * DEGREE, xo_v(i), yo_v(i), Multi(ro_v(i) + eps));
                    f1 = map.flux(axis_v, theta_v(i) * DEGREE, xo_v(i), yo_v(i), Multi(ro_v(i) - eps));
                    deriv(i) = (double)((f2 - f1) / (2 * eps));
                }
                return deriv;

            }, "axis"_a=yhat, "theta"_a=0, "xo"_a=0, "yo"_a=0, "ro"_a=0);

}

template <typename MAPTYPE>
void add_Map(py::class_<maps::Map<MAPTYPE>>& PyMap, const docstrings::docs<MAPTYPE>& docs) {

    PyMap

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
                        if (size_t(values.size()) != slicelength)
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
            }, docs.Map.get_coeff, "l"_a, "m"_a)

        .def("set_coeff", [](maps::Map<MAPTYPE> &map, int l, int m, double coeff){
                map.set_coeff(l, m, MAPTYPE(coeff));
            }, docs.Map.set_coeff, "l"_a, "m"_a, "coeff"_a)

        .def("reset", &maps::Map<MAPTYPE>::reset, docs.Map.reset)

        .def_property_readonly("lmax", [](maps::Map<MAPTYPE> &map){return map.lmax;}, docs.Map.lmax)

        .def_property_readonly("y", [](maps::Map<MAPTYPE> &map){
                return get_value(map.y);
            }, docs.Map.y)

        .def_property_readonly("p", [](maps::Map<MAPTYPE> &map){
                return get_value(map.p);
            }, docs.Map.p)

        .def_property_readonly("g", [](maps::Map<MAPTYPE> &map){
                return get_value(map.g);
            }, docs.Map.g)

        .def_property_readonly("s", [](maps::Map<MAPTYPE> &map){
                return get_value((Vector<MAPTYPE>)map.G.sT);
            }, docs.Map.s)

        .def_property_readonly("r", [](maps::Map<MAPTYPE> &map){
                return get_value((Vector<MAPTYPE>)map.C.rT);
            }, docs.Map.r)

        .def("evaluate", [](maps::Map<MAPTYPE>& map, UnitVector<double>& axis, py::object& theta, py::object& x, py::object& y) {
                UnitVector<double> axis_norm = norm_unit(axis);
                return vectorize_map_evaluate(axis_norm, theta, x, y, map);
            }, docs.Map.evaluate, "axis"_a=yhat, "theta"_a=0, "x"_a=0, "y"_a=0)

        .def("flux", [](maps::Map<MAPTYPE>& map, UnitVector<double>& axis, py::object& theta, py::object& xo, py::object& yo, py::object& ro) {
                UnitVector<double> axis_norm = norm_unit(axis);
                return vectorize_map_flux(axis_norm, theta, xo, yo, ro, map);
            }, docs.Map.flux, "axis"_a=yhat, "theta"_a=0, "xo"_a=0, "yo"_a=0, "ro"_a=0)

        .def("rotate", [](maps::Map<MAPTYPE> &map, UnitVector<double>& axis, double theta){
                //UnitVector<MAPTYPE> axis_norm = UnitVector<MAPTYPE>(norm_unit(axis));
                UnitVector<MAPTYPE> axis_norm = norm_unit(axis).template cast<MAPTYPE>();
                map.rotate(axis_norm, theta * DEGREE);
            }, docs.Map.rotate, "axis"_a=yhat, "theta"_a=0)

        .def("psd", &maps::Map<MAPTYPE>::psd, docs.Map.psd, "epsilon"_a=1.e-6, "max_iterations"_a=100)

        .def("load_array", [](maps::Map<MAPTYPE> &map, Matrix<double>& image) {
                py::object load_map = py::module::import("starry.maps").attr("load_map");
                Vector<double> y = load_map(image, map.lmax, false).template cast<Vector<double>>();
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
                UnitVector<MAPTYPE> M_xhat = xhat.template cast<MAPTYPE>();
                UnitVector<MAPTYPE> M_yhat = yhat.template cast<MAPTYPE>();
                UnitVector<MAPTYPE> M_zhat = zhat.template cast<MAPTYPE>();
                MAPTYPE Pi(M_PI);
                MAPTYPE PiOver2(M_PI / 2.);
                map.rotate(M_xhat, PiOver2);
                map.rotate(M_zhat, Pi);
                map.rotate(M_yhat, PiOver2);
            }, docs.Map.load_array, "image"_a)

        .def("load_image", [](maps::Map<MAPTYPE> &map, string& image) {
                py::object load_map = py::module::import("starry.maps").attr("load_map");
                Vector<double> y = load_map(image, map.lmax).template cast<Vector<double>>();
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
                UnitVector<MAPTYPE> M_xhat = xhat.template cast<MAPTYPE>();
                UnitVector<MAPTYPE> M_yhat = yhat.template cast<MAPTYPE>();
                UnitVector<MAPTYPE> M_zhat = zhat.template cast<MAPTYPE>();
                MAPTYPE Pi(M_PI);
                MAPTYPE PiOver2(M_PI / 2.);
                map.rotate(M_xhat, PiOver2);
                map.rotate(M_zhat, Pi);
                map.rotate(M_yhat, PiOver2);
            }, docs.Map.load_image, "image"_a)

        .def("load_healpix", [](maps::Map<MAPTYPE> &map, Matrix<double>& image) {
                py::object load_map = py::module::import("starry.maps").attr("load_map");
                Vector<double> y = load_map(image, map.lmax, true).template cast<Vector<double>>();
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
                UnitVector<MAPTYPE> M_xhat = xhat.template cast<MAPTYPE>();
                UnitVector<MAPTYPE> M_yhat = yhat.template cast<MAPTYPE>();
                UnitVector<MAPTYPE> M_zhat = zhat.template cast<MAPTYPE>();
                MAPTYPE Pi(M_PI);
                MAPTYPE PiOver2(M_PI / 2.);
                map.rotate(M_xhat, PiOver2);
                map.rotate(M_zhat, Pi);
                map.rotate(M_yhat, PiOver2);
            }, docs.Map.load_healpix, "image"_a)

        .def("add_gaussian", [](maps::Map<MAPTYPE> &map, double sigma, double amp, double lat, double lon) {
                py::object gaussian = py::module::import("starry.maps").attr("gaussian");
                Vector<double> y = gaussian(sigma, map.lmax).template cast<Vector<double>>();
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
                UnitVector<double> D_xhat(xhat);
                UnitVector<double> D_yhat(yhat);
                UnitVector<double> D_zhat(zhat);
                tmpmap.rotate(D_xhat, M_PI / 2.);
                tmpmap.rotate(D_zhat, M_PI);
                tmpmap.rotate(D_yhat, M_PI / 2.);
                // Now rotate it to where the user wants it
                tmpmap.rotate(D_xhat, -lat * DEGREE);
                tmpmap.rotate(D_yhat, lon * DEGREE);
                // Add it to the current map
                for (int l = 0; l < map.lmax + 1; l++) {
                    for (int m = -l; m < l + 1; m++) {
                        map.set_coeff(l, m, get_value(map.get_coeff(l, m)) + tmpmap.get_coeff(l, m));
                    }
                }
            }, docs.Map.add_gaussian, "sigma"_a=0.1, "amp"_a=1, "lat"_a=0, "lon"_a=0)

        .def("show", [](maps::Map<MAPTYPE> &map, string cmap, int res) {
                py::object show = py::module::import("starry.maps").attr("show");
                Matrix<double> I;
                I.resize(res, res);
                Vector<double> x;
                UnitVector<MAPTYPE> M_yhat = yhat.template cast<MAPTYPE>();
                x = Vector<double>::LinSpaced(res, -1, 1);
                for (int i = 0; i < res; i++){
                    for (int j = 0; j < res; j++){
                        I(j, i) = get_value(map.evaluate(M_yhat, MAPTYPE(0), MAPTYPE(x(i)), MAPTYPE(x(j))));
                    }
                }
                show(I, "cmap"_a=cmap, "res"_a=res);
            }, docs.Map.show, "cmap"_a="plasma", "res"_a=300)

        .def("animate", [](maps::Map<MAPTYPE> &map, UnitVector<double>& axis, string cmap, int res, int frames) {
            std::cout << "Rendering animation..." << std::endl;
            py::object animate = py::module::import("starry.maps").attr("animate");
            vector<Matrix<double>> I;
            Vector<double> x, theta;
            x = Vector<double>::LinSpaced(res, -1, 1);
            theta = Vector<double>::LinSpaced(frames, 0, 2 * M_PI);
            UnitVector<MAPTYPE> MapType_axis = norm_unit(axis).template cast<MAPTYPE>();
            for (int t = 0; t < frames; t++){
                I.push_back(Matrix<double>::Zero(res, res));
                for (int i = 0; i < res; i++){
                    for (int j = 0; j < res; j++){
                        I[t](j, i) = get_value(map.evaluate(MapType_axis, MAPTYPE(theta(t)), MAPTYPE(x(i)), MAPTYPE(x(j))));
                    }
                }
            }
            animate(I, axis, "cmap"_a=cmap, "res"_a=res);
        }, docs.Map.animate, "axis"_a=yhat, "cmap"_a="plasma", "res"_a=150, "frames"_a=50)

        .def("__repr__", [](maps::Map<MAPTYPE> &map) -> string {return map.repr();});

    add_Map_extras(PyMap, docs);

}

template <typename MAPTYPE>
void add_LimbDarkenedMap_extras(py::class_<maps::LimbDarkenedMap<MAPTYPE>>& PyLimbDarkenedMap, const docstrings::docs<MAPTYPE>& docs) { }

template <>
void add_LimbDarkenedMap_extras<double>(py::class_<maps::LimbDarkenedMap<double>>& PyLimbDarkenedMap, const docstrings::docs<double>& docs) {

    PyLimbDarkenedMap

        .def("_flux_numerical", [](maps::LimbDarkenedMap<double>& map, py::object& xo, py::object& yo, py::object& ro, double tol) {
                return vectorize_ldmap_flux_numerical(xo, yo, ro, tol, map);
            }, docs.LimbDarkenedMap.flux_numerical, "xo"_a=0, "yo"_a=0, "ro"_a=0, "tol"_a=1e-4);

}

template <>
void add_LimbDarkenedMap_extras<Grad>(py::class_<maps::LimbDarkenedMap<Grad>>& PyLimbDarkenedMap, const docstrings::docs<Grad>& docs) {

    PyLimbDarkenedMap

        .def_property_readonly("gradient", [](maps::LimbDarkenedMap<Grad> &map){
                return py::cast(map.derivs);
            }, docs.LimbDarkenedMap.gradient);
}

template <typename MAPTYPE>
void add_LimbDarkenedMap(py::class_<maps::LimbDarkenedMap<MAPTYPE>>& PyLimbDarkenedMap, const docstrings::docs<MAPTYPE>& docs) {

    PyLimbDarkenedMap

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
                    if (size_t(values.size()) != slicelength)
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
            }, docs.LimbDarkenedMap.get_coeff, "l"_a)

        .def("set_coeff", [](maps::LimbDarkenedMap<MAPTYPE> &map, int l, double coeff){
                map.set_coeff(l, MAPTYPE(coeff));
            }, docs.LimbDarkenedMap.set_coeff, "l"_a, "coeff"_a)

        .def("reset", &maps::LimbDarkenedMap<MAPTYPE>::reset, docs.LimbDarkenedMap.reset)

        .def("psd", &maps::LimbDarkenedMap<MAPTYPE>::psd, docs.LimbDarkenedMap.psd)

        .def("mono", &maps::LimbDarkenedMap<MAPTYPE>::mono, docs.LimbDarkenedMap.mono)

        .def_property_readonly("lmax", [](maps::LimbDarkenedMap<MAPTYPE> &map){return map.lmax;}, docs.LimbDarkenedMap.lmax)

        .def_property_readonly("y", [](maps::LimbDarkenedMap<MAPTYPE> &map){
                return get_value(map.y);
            }, docs.LimbDarkenedMap.y)

        .def_property_readonly("p", [](maps::LimbDarkenedMap<MAPTYPE> &map){
                return get_value(map.p);
            }, docs.LimbDarkenedMap.p)

        .def_property_readonly("g", [](maps::LimbDarkenedMap<MAPTYPE> &map){
                return get_value(map.g);
            }, docs.LimbDarkenedMap.g)

        .def_property_readonly("s", [](maps::LimbDarkenedMap<MAPTYPE> &map){
                return get_value((Vector<MAPTYPE>)map.G.sT);
            }, docs.LimbDarkenedMap.s)

        .def_property_readonly("u", [](maps::LimbDarkenedMap<MAPTYPE> &map){
                // Hide u_0, since it's never used!
                Vector<double> u(map.lmax);
                for (int i = 0; i < map.lmax; i++)
                    u(i) = get_value(map.u(i + 1));
                return u;
            }, docs.LimbDarkenedMap.u)

        .def("evaluate", [](maps::LimbDarkenedMap<MAPTYPE>& map, py::object& x, py::object& y) {
                return vectorize_ldmap_evaluate(x, y, map);
            }, docs.LimbDarkenedMap.evaluate, "x"_a=0, "y"_a=0)

        .def("flux", [](maps::LimbDarkenedMap<MAPTYPE>& map, py::object& xo, py::object& yo, py::object& ro) {
                return vectorize_ldmap_flux(xo, yo, ro, map);
            }, docs.LimbDarkenedMap.flux, "xo"_a=0, "yo"_a=0, "ro"_a=0)

        .def("show", [](maps::LimbDarkenedMap<MAPTYPE> &map, string cmap, int res) {
                py::object show = py::module::import("starry.maps").attr("show");
                Matrix<double> I;
                I.resize(res, res);
                Vector<double> x;
                x = Vector<double>::LinSpaced(res, -1, 1);
                for (int i = 0; i < res; i++){
                    for (int j = 0; j < res; j++){
                        I(j, i) = get_value(map.evaluate(MAPTYPE(x(i)), MAPTYPE(x(j))));
                    }
                }
                show(I, "cmap"_a=cmap, "res"_a=res);
            }, docs.LimbDarkenedMap.show, "cmap"_a="plasma", "res"_a=300)

        .def("__repr__", [](maps::LimbDarkenedMap<MAPTYPE> &map) -> string {return map.repr();});

    add_LimbDarkenedMap_extras(PyLimbDarkenedMap, docs);

}

template <typename MAPTYPE>
void add_System_extras(py::class_<orbital::System<MAPTYPE>>& PySystem, const docstrings::docs<MAPTYPE>& docs) { }

template <>
void add_System_extras<Grad>(py::class_<orbital::System<Grad>>& PySystem, const docstrings::docs<Grad>& docs) {

    PySystem

        .def_property_readonly("gradient", [](orbital::System<Grad> &system){return py::cast(system.derivs);},
                docs.System.gradient);

}

template <typename MAPTYPE>
void add_System(py::class_<orbital::System<MAPTYPE>>& PySystem, const docstrings::docs<MAPTYPE>& docs) {

    PySystem

        .def(py::init<vector<orbital::Body<MAPTYPE>*>, double, double, double, int>(),
            "bodies"_a, "scale"_a=0, "exposure_time"_a=0, "exposure_tol"_a=1e-8, "exposure_max_depth"_a=4)

        .def("compute", [](orbital::System<MAPTYPE> &system, Vector<double>& time){system.compute(time.template cast<MAPTYPE>());},
            docs.System.compute, "time"_a)

        .def_property_readonly("flux", [](orbital::System<MAPTYPE> &system){return get_value(system.flux);},
            docs.System.flux)

        .def_property("scale", [](orbital::System<MAPTYPE> &system){return CLIGHT / (system.clight * RSUN);},
            [](orbital::System<MAPTYPE> &system, double scale){
                if (scale == 0)
                    system.clight = MAPTYPE(INFINITY);
                else
                    system.clight = MAPTYPE(CLIGHT / (scale * RSUN));
            }, docs.System.scale)

        .def_property("exposure_time", [](orbital::System<MAPTYPE> &system){return system.exptime / DAY;},
            [](orbital::System<MAPTYPE> &system, double exptime){system.exptime = exptime * DAY;}, docs.System.exposure_time)

        .def_property("exposure_tol", [](orbital::System<MAPTYPE> &system){return system.exptol;},
            [](orbital::System<MAPTYPE> &system, double exptol){system.exptol = exptol;}, docs.System.exposure_tol)

        .def_property("exposure_max_depth", [](orbital::System<MAPTYPE> &system){return system.expmaxdepth;},
            [](orbital::System<MAPTYPE> &system, int expmaxdepth){system.expmaxdepth = expmaxdepth;}, docs.System.exposure_max_depth)

        .def("__repr__", [](orbital::System<MAPTYPE> &system) -> string {return system.repr();});

    add_System_extras(PySystem, docs);

}

template <typename MAPTYPE>
void add_Body_extras(py::class_<orbital::Body<MAPTYPE>>& PyBody, const docstrings::docs<MAPTYPE>& docs) { }

template <>
void add_Body_extras<Grad>(py::class_<orbital::Body<Grad>>& PyBody, const docstrings::docs<Grad>& docs) {

    PyBody

        .def_property_readonly("gradient", [](orbital::Body<Grad> &body){return py::cast(body.derivs);},
            docs.Body.gradient);

}

template <typename MAPTYPE>
void add_Body(py::class_<orbital::Body<MAPTYPE>>& PyBody, const docstrings::docs<MAPTYPE>& docs) {

    PyBody

        .def(py::init<int, const double&, const double&,
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
        .def_property_readonly("map", [](orbital::Body<MAPTYPE> &body){return &body.map;}, docs.Body.map)

        .def_property_readonly("flux", [](orbital::Body<MAPTYPE> &body){return get_value(body.flux);}, docs.Body.flux)

        .def_property_readonly("x", [](orbital::Body<MAPTYPE> &body){return get_value(body.x);}, docs.Body.x)

        .def_property_readonly("y", [](orbital::Body<MAPTYPE> &body){return get_value(body.y);}, docs.Body.y)

        .def_property_readonly("z", [](orbital::Body<MAPTYPE> &body){return get_value(body.z);}, docs.Body.z)

        .def_property("r", [](orbital::Body<MAPTYPE> &body){return get_value(body.r);},
            [](orbital::Body<MAPTYPE> &body, double r){body.r = MAPTYPE(r);}, docs.Body.r)

        .def_property("L", [](orbital::Body<MAPTYPE> &body){return get_value(body.L);},
            [](orbital::Body<MAPTYPE> &body, double L){body.L = MAPTYPE(L); body.reset();}, docs.Body.L)

        .def_property("axis", [](orbital::Body<MAPTYPE> &body){return get_value((Vector<MAPTYPE>)body.axis);},
            [](orbital::Body<MAPTYPE> &body, UnitVector<double> axis){body.axis = norm_unit(axis).template cast<MAPTYPE>();}, docs.Body.axis)

        .def_property("prot", [](orbital::Body<MAPTYPE> &body){return get_value(body.prot) / DAY;},
            [](orbital::Body<MAPTYPE> &body, double prot){body.prot = MAPTYPE(prot * DAY); body.reset();}, docs.Body.prot)

        .def_property("a", [](orbital::Body<MAPTYPE> &body){return get_value(body.a);},
            [](orbital::Body<MAPTYPE> &body, double a){body.a = MAPTYPE(a); body.reset();}, docs.Body.a)

        .def_property("porb", [](orbital::Body<MAPTYPE> &body){return get_value(body.porb) / DAY;},
            [](orbital::Body<MAPTYPE> &body, double porb){body.porb = MAPTYPE(porb * DAY); body.reset();}, docs.Body.porb)

        .def_property("inc", [](orbital::Body<MAPTYPE> &body){return get_value(body.inc) / DEGREE;},
            [](orbital::Body<MAPTYPE> &body, double inc){body.inc = MAPTYPE(inc * DEGREE); body.reset();}, docs.Body.inc)

        .def_property("ecc", [](orbital::Body<MAPTYPE> &body){return get_value(body.ecc);},
            [](orbital::Body<MAPTYPE> &body, double ecc){body.ecc = MAPTYPE(ecc); body.reset();}, docs.Body.ecc)

        .def_property("w", [](orbital::Body<MAPTYPE> &body){return get_value(body.w) / DEGREE;},
            [](orbital::Body<MAPTYPE> &body, double w){body.w = MAPTYPE(w * DEGREE); body.reset();}, docs.Body.w)

        .def_property("Omega", [](orbital::Body<MAPTYPE> &body){return get_value(body.Omega) / DEGREE;},
            [](orbital::Body<MAPTYPE> &body, double Omega){body.Omega = MAPTYPE(Omega * DEGREE); body.reset();}, docs.Body.Omega)

        .def_property("lambda0", [](orbital::Body<MAPTYPE> &body){return get_value(body.lambda0) / DEGREE;},
            [](orbital::Body<MAPTYPE> &body, double lambda0){body.lambda0 = MAPTYPE(lambda0 * DEGREE); body.reset();}, docs.Body.lambda0)

        .def_property("tref", [](orbital::Body<MAPTYPE> &body){return get_value(body.tref) / DAY;},
            [](orbital::Body<MAPTYPE> &body, double tref){body.tref = MAPTYPE(tref * DAY);}, docs.Body.tref)

        .def("__repr__", [](orbital::Body<MAPTYPE> &body) -> string {return body.repr();});

    add_Body_extras(PyBody, docs);

}

template <typename MAPTYPE>
void add_Star_extras(py::class_<orbital::Star<MAPTYPE>>& PyStar, const docstrings::docs<MAPTYPE>& docs) { }

template <typename MAPTYPE>
void add_Star(py::class_<orbital::Star<MAPTYPE>>& PyStar, const docstrings::docs<MAPTYPE>& docs) {

    PyStar

        .def(py::init<int>(), "lmax"_a=2)

        // NOTE: & is necessary in the return statement so we pass a reference back to Python!
        .def_property_readonly("map", [](orbital::Star<MAPTYPE> &star){return &star.ldmap;}, docs.Star.map)

        .def_property_readonly("r", [](orbital::Star<MAPTYPE> &star){return 1.;}, docs.Star.r)

        .def_property_readonly("L", [](orbital::Star<MAPTYPE> &star){return 1.;}, docs.Star.L)

        .def_property_readonly("axis", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, docs.NotImplemented)

        .def_property_readonly("prot", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, docs.NotImplemented)

        .def_property_readonly("a", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, docs.NotImplemented)

        .def_property_readonly("porb", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, docs.NotImplemented)

        .def_property_readonly("inc", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, docs.NotImplemented)

        .def_property_readonly("ecc", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, docs.NotImplemented)

        .def_property_readonly("w", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, docs.NotImplemented)

        .def_property_readonly("Omega", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, docs.NotImplemented)

        .def_property_readonly("lambda0", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, docs.NotImplemented)

        .def_property_readonly("tref", [](orbital::Star<MAPTYPE> &star){throw errors::NotImplemented();}, docs.NotImplemented)

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
                    if (size_t(values.size()) != slicelength)
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

    add_Star_extras(PyStar, docs);

}

template <typename MAPTYPE>
void add_Planet_extras(py::class_<orbital::Planet<MAPTYPE>>& PyPlanet, const docstrings::docs<MAPTYPE>& docs) { }

template <typename MAPTYPE>
void add_Planet(py::class_<orbital::Planet<MAPTYPE>>& PyPlanet, const docstrings::docs<MAPTYPE>& docs) {

    PyPlanet

        .def(py::init<int, const double&, const double&,
                      Eigen::Matrix<double, 3, 1>&,
                      const double&, const double&,
                      const double&, const double&,
                      const double&, const double&,
                      const double&, const double&,
                      const double&>(),
                      "lmax"_a=2, "r"_a=0.1, "L"_a=0., "axis"_a=yhat,
                      "prot"_a=INFINITY, "a"_a=50., "porb"_a=1,
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
                    if (size_t(values.size()) != slicelength)
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

    add_Planet_extras(PyPlanet, docs);

}

template <typename MAPTYPE>
void add_extras(py::module& m, const docstrings::docs<MAPTYPE>& docs) { }

template <>
void add_extras(py::module& m, const docstrings::docs<Multi>& docs) {
    m.attr("NMULTI") = STARRY_NMULTI;
}

template <>
void add_extras(py::module& m, const docstrings::docs<Grad>& docs) {
    m.attr("NGRAD") = STARRY_NGRAD;
}

template <typename MAPTYPE>
void add_starry(py::module& m, const docstrings::docs<MAPTYPE>& docs) {

    // Main docs
    m.doc() = docs.doc;

    // Type-specific stuff
    add_extras(m, docs);

    // Surface map class
    py::class_<maps::Map<MAPTYPE>> PyMap(m, "Map", docs.Map.doc);
    add_Map(PyMap, docs);

    // Limb-darkened map class
    py::class_<maps::LimbDarkenedMap<MAPTYPE>> PyLimbDarkenedMap(m, "LimbDarkenedMap", docs.LimbDarkenedMap.doc);
    add_LimbDarkenedMap(PyLimbDarkenedMap, docs);

    // System class
    py::class_<orbital::System<MAPTYPE>> PySystem(m, "System", docs.System.doc);
    add_System(PySystem, docs);

    // Body class (not user-facing)
    py::class_<orbital::Body<MAPTYPE>> PyBody(m, "Body");
    add_Body(PyBody, docs);

    // Star class
    py::class_<orbital::Star<MAPTYPE>> PyStar(m, "Star", PyBody, docs.Star.doc);
    add_Star(PyStar, docs);

    // Planet class
    py::class_<orbital::Planet<MAPTYPE>> PyPlanet(m, "Planet", PyBody, docs.Planet.doc);
    add_Planet(PyPlanet, docs);

}

#endif
