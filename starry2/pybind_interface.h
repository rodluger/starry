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
#include "docstrings.h"
#include "utils.h"
#include "errors.h"

using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;

template <typename MAPTYPE>
void add_Map_extras(py::class_<maps::Map<MAPTYPE>>& PyMap, const docstrings::docs<MAPTYPE>& docs) { }

template <>
void add_Map_extras<double>(py::class_<maps::Map<double>>& PyMap, const docstrings::docs<double>& docs) { }

template <>
void add_Map_extras<Grad>(py::class_<maps::Map<Grad>>& PyMap, const docstrings::docs<Grad>& docs) {

    PyMap

        .def_property_readonly("gradient", [](maps::Map<Grad> &map){
                return py::cast(map.derivs);
            }, docs.Map.gradient);

}

template <>
void add_Map_extras<Multi>(py::class_<maps::Map<Multi>>& PyMap, const docstrings::docs<Multi>& docs) { }

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
                    map.setCoeff(l, m, MAPTYPE(value));
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
                    return py::cast(get_value(map.getCoeff(l, m)));
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
                return get_value(map.getCoeff(l, m));
            }, docs.Map.get_coeff, "l"_a, "m"_a)

        .def("set_coeff", [](maps::Map<MAPTYPE> &map, int l, int m, double coeff){
                map.setCoeff(l, m, MAPTYPE(coeff));
            }, docs.Map.set_coeff, "l"_a, "m"_a, "coeff"_a)

        .def_property("axis",
            [](maps::Map<MAPTYPE> &map) {
                UnitVector<double> axis;
                axis(0) = get_value(map.axis(0));
                axis(1) = get_value(map.axis(1));
                axis(2) = get_value(map.axis(2));
                return axis;
            },
            [](maps::Map<MAPTYPE> &map, UnitVector<double>& axis){
                map.axis(0) = axis(0);
                map.axis(1) = axis(1);
                map.axis(2) = axis(2);
                map.update();
            }, docs.Map.axis)

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

        /*
        .def("evaluate", [](maps::Map<MAPTYPE>& map, py::object& theta, py::object& x, py::object& y) {
                return vectorize_map_evaluate(theta, x, y, map);
            }, docs.Map.evaluate, "theta"_a=0, "x"_a=0, "y"_a=0)
        */

        .def("rotate", [](maps::Map<MAPTYPE> &map, double theta){
                map.rotate(theta * DEGREE);
            }, docs.Map.rotate, "theta"_a=0)

        .def("__repr__", [](maps::Map<MAPTYPE> &map) -> string {return map.repr();});

    add_Map_extras(PyMap, docs);

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
    py::class_<maps::Map<MAPTYPE>> PyMap(m, "SurfaceMap", docs.Map.doc);
    add_Map(PyMap, docs);

}

#endif
