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

namespace pybind_interface {

    using namespace std;
    using namespace utils;
    using namespace pybind11::literals;
    namespace py = pybind11;

    template <typename T>
    void add_Map_extras(py::class_<maps::Map<T, double>>& PyMap, const docstrings::docs<T>& docs) { }

    template <>
    void add_Map_extras<double>(py::class_<maps::Map<double, double>>& PyMap, const docstrings::docs<double>& docs) { }


    template <>
    void add_Map_extras<Multi>(py::class_<maps::Map<Multi, double>>& PyMap, const docstrings::docs<Multi>& docs) { }

    template <typename T>
    void add_Map(py::class_<maps::Map<T, double>>& PyMap, const docstrings::docs<T>& docs) {

        PyMap

            .def(py::init<int>(), "lmax"_a=2)

            .def("__setitem__", [](maps::Map<T, double>& map, py::object index, py::object& coeff) {
                    if (py::isinstance<py::tuple>(index)) {
                        // User provided a (l, m) tuple
                        py::tuple lm = index;
                        int l, m;
                        double value = py::cast<double>(coeff);
                        try {
                            l = py::cast<int>(lm[0]);
                            m = py::cast<int>(lm[1]);
                        } catch (const char* msg) {
                            throw errors::IndexError("Invalid value for `l` and/or `m`.");
                        }
                        map.setCoeff(l, m, value);
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
                                throw errors::IndexError("Invalid slice length.");
                        }
                        Vector<int> inds(values.size());
                        for (size_t i = 0; i < slicelength; ++i) {
                            inds(i) = start;
                            start += step;
                        }
                        map.setCoeff(inds, values);
                    } else {
                        throw errors::IndexError("Invalid value for `l` and/or `m`.");
                    }
                })

            .def("__getitem__", [](maps::Map<T, double>& map, py::object index) -> py::object {
                    if (py::isinstance<py::tuple>(index)) {
                        py::tuple lm = index;
                        int l, m;
                        try {
                            l = py::cast<int>(lm[0]);
                            m = py::cast<int>(lm[1]);
                        } catch (const char* msg) {
                            throw errors::IndexError("Invalid value for `l` and/or `m`.");
                        }
                        return py::cast(map.getCoeff(l, m));
                    } else if (py::isinstance<py::slice>(index)) {
                        // User provided a slice of some sort
                        size_t start, stop, step, slicelength;
                        py::slice slice = py::cast<py::slice>(index);
                        if(!slice.compute(map.N, &start, &stop, &step, &slicelength))
                            throw pybind11::error_already_set();
                        Vector<int> inds(slicelength);
                        for (size_t i = 0; i < slicelength; ++i) {
                            inds(i) = start;
                            start += step;
                        }
                        Vector<double> res = map.getCoeff(inds);
                        return py::cast(res);
                    } else {
                        throw errors::IndexError("Invalid value for `l` and/or `m`.");
                    }
                })

            .def_property("axis",
                [](maps::Map<T, double> &map) {return map.getAxis();},
                [](maps::Map<T, double> &map, UnitVector<double>& axis){map.setAxis(axis);},
                docs.Map.axis)

            .def("reset", &maps::Map<T, double>::reset, docs.Map.reset)

            .def_property_readonly("lmax", [](maps::Map<T, double> &map){return map.lmax;}, docs.Map.lmax)

            .def_property_readonly("y", [](maps::Map<T, double> &map){return map.getY();}, docs.Map.y)

            .def_property_readonly("p", [](maps::Map<T, double> &map){return map.getP();}, docs.Map.p)

            .def_property_readonly("g", [](maps::Map<T, double> &map){return map.getG();}, docs.Map.g)

            .def_property_readonly("r", [](maps::Map<T, double> &map){return map.getR();}, docs.Map.r)

            /*
            .def_property_readonly("s", [](maps::Map<T, double> &map){
                    return get_value((Vector<T>)map.G.sT);
                }, docs.Map.s)
            */

            .def("evaluate", py::vectorize(&maps::Map<T, double>::evaluate), docs.Map.evaluate, "theta"_a=0.0, "x"_a=0.0, "y"_a=0.0)

            .def("rotate", &maps::Map<T, double>::rotate, docs.Map.rotate, "theta"_a=0)

            .def("__repr__", &maps::Map<T, double>::__repr__);


        add_Map_extras(PyMap, docs);

    }

    template <typename T>
    void add_extras(py::module& m, const docstrings::docs<T>& docs) { }

    template <>
    void add_extras(py::module& m, const docstrings::docs<Multi>& docs) {
        m.attr("NMULTI") = STARRY_NMULTI;
    }

    template <typename T>
    void add_starry(py::module& m, const docstrings::docs<T>& docs) {

        // Main docs
        m.doc() = docs.doc;

        // Type-specific stuff
        add_extras(m, docs);

        // Surface map class
        py::class_<maps::Map<T, double>> PyMap(m, "Map", docs.Map.doc);
        add_Map(PyMap, docs);

    }

}; // namespace pybind_interface

#endif
