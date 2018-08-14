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
#include "pybind_vectorize.h"


namespace pybind_interface {

    using namespace std;
    using namespace utils;
    using namespace pybind11::literals;
    namespace py = pybind11;
    namespace vectorize = pybind_vectorize;


    template <typename T, int Module>
    void add_Map_extras(py::class_<maps::Map<T>>& PyMap,
                        const docstrings::docs<Module>& docs) { }

    template <>
    void add_Map_extras(py::class_<maps::Map<Vector<double>>>& PyMap,
                        const docstrings::docs<STARRY_MODULE_MAIN>& docs) {

        PyMap

            .def(py::init<int>(), "lmax"_a=2)

            .def("__setitem__", [](maps::Map<Vector<double>>& map,
                                   py::tuple lm, double& coeff) {
                int l, m;
                try {
                    l = py::cast<int>(lm[0]);
                    m = py::cast<int>(lm[1]);
                } catch (const char* msg) {
                    throw errors::IndexError("Invalid value for `l` and/or `m`.");
                }
                map.setYlm(l, m, coeff);
            })

            .def("__getitem__", [](maps::Map<Vector<double>>& map,
                                   py::tuple lm) -> double {
                int l, m;
                try {
                    l = py::cast<int>(lm[0]);
                    m = py::cast<int>(lm[1]);
                } catch (const char* msg) {
                    throw errors::IndexError("Invalid value for `l` and/or `m`.");
                }
                return map.getYlm(l, m);
            });

    }

    template <>
    void add_Map_extras(py::class_<maps::Map<Vector<Multi>>>& PyMap,
                        const docstrings::docs<STARRY_MODULE_MULTI>& docs) {

        PyMap

            .def(py::init<int>(), "lmax"_a=2)

            .def("__setitem__", [](maps::Map<Vector<Multi>>& map,
                                   py::tuple lm, double& coeff) {
                int l, m;
                try {
                    l = py::cast<int>(lm[0]);
                    m = py::cast<int>(lm[1]);
                } catch (const char* msg) {
                    throw errors::IndexError("Invalid value for `l` and/or `m`.");
                }
                map.setYlm(l, m, static_cast<Multi>(coeff));
            })

            .def("__getitem__", [](maps::Map<Vector<Multi>>& map,
                                   py::tuple lm) -> double {
                int l, m;
                try {
                    l = py::cast<int>(lm[0]);
                    m = py::cast<int>(lm[1]);
                } catch (const char* msg) {
                    throw errors::IndexError("Invalid value for `l` and/or `m`.");
                }
                return static_cast<double>(map.getYlm(l, m));
            });

    }

    template <>
    void add_Map_extras(py::class_<maps::Map<Matrix<double>>>& PyMap,
                        const docstrings::docs<STARRY_MODULE_SPECTRAL>& docs) {

        PyMap

            .def(py::init<int, int>(), "lmax"_a=2, "nwav"_a=1)

            .def_property_readonly("nwav",
                [](maps::Map<Matrix<double>> &map){
                    return map.nwav;
                }, docs.Map.nwav)

            .def("__setitem__", [](maps::Map<Matrix<double>>& map,
                                   py::tuple lm, Vector<double>& coeff) {
                int l, m;
                try {
                    l = py::cast<int>(lm[0]);
                    m = py::cast<int>(lm[1]);
                } catch (const char* msg) {
                    throw errors::IndexError("Invalid value for `l` and/or `m`.");
                }
                map.setYlm(l, m, coeff);
            })

            .def("__getitem__", [](maps::Map<Matrix<double>>& map,
                                   py::tuple lm) -> VectorT<double> {
                int l, m;
                try {
                    l = py::cast<int>(lm[0]);
                    m = py::cast<int>(lm[1]);
                } catch (const char* msg) {
                    throw errors::IndexError("Invalid value for `l` and/or `m`.");
                }
                return map.getYlm(l, m);
            });

    }

    template <typename T, int Module>
    void add_Map(py::class_<maps::Map<T>>& PyMap,
                 const docstrings::docs<Module>& docs) {

        PyMap

            .def_property("axis",
                [](maps::Map<T> &map) -> UnitVector<double> {
                        return map.getAxis().template cast<double>();
                    },
                [](maps::Map<T> &map, UnitVector<double>& axis){
                        map.setAxis(axis.template cast<Scalar<T>>());
                    },
                docs.Map.axis)

            .def("reset", &maps::Map<T>::reset, docs.Map.reset)

            .def_property_readonly("lmax", [](maps::Map<T> &map){
                    return map.lmax;
                }, docs.Map.lmax)

            .def_property_readonly("N", [](maps::Map<T> &map){
                    return map.N;
                }, docs.Map.N)

            // TODO: Currently, slice indexing works for the getter,
            // but not for the setter. Unfortunately, it doesn't throw
            // an error -- it just simply never calls this function,
            // so the array does not get set.
            .def_property("y", [](maps::Map<T> &map) -> Vector<double>{
                        return map.getY().template cast<double>();
                    },
                [](maps::Map<T> &map, Vector<double>& y){
                        map.setY(y.template cast<Scalar<T>>());
                    },
                docs.Map.y)

            .def_property_readonly("p", [](maps::Map<T> &map) -> Vector<double>{
                    return map.getP().template cast<double>();
                }, docs.Map.p)

            .def_property_readonly("g", [](maps::Map<T> &map) -> Vector<double>{
                    return map.getG().template cast<double>();
                }, docs.Map.g)

            .def_property_readonly("r", [](maps::Map<T> &map) -> VectorT<double>{
                    return map.getR().template cast<double>();
                }, docs.Map.r)

            .def_property_readonly("s", [](maps::Map<T> &map) -> VectorT<double>{
                    return map.getS().template cast<double>();
                }, docs.Map.s)

            .def("evaluate", [](maps::Map<T> &map,
                                py::array_t<double>& theta,
                                py::array_t<double>& x,
                                py::array_t<double>& y,
                                bool gradient)
                                -> py::object {
                    return vectorize::evaluate(map, theta, x, y, gradient);
                }, docs.Map.evaluate, "theta"_a=0.0, "x"_a=0.0, "y"_a=0.0,
                                      "gradient"_a=false)

            .def("flux", [](maps::Map<T> &map,
                            py::array_t<double>& theta,
                            py::array_t<double>& xo,
                            py::array_t<double>& yo,
                            py::array_t<double>& ro,
                            bool gradient)
                            -> py::object {
                    return vectorize::flux(map, theta, xo, yo, ro, gradient);
                }, docs.Map.flux, "theta"_a=0.0, "xo"_a=0.0, "yo"_a=0.0,
                                   "ro"_a=0.0, "gradient"_a=false)

            .def("rotate", [](maps::Map<T> &map, double theta) {
                    map.rotate(static_cast<Scalar<T>>(theta));
            }, docs.Map.rotate, "theta"_a=0)

            .def("__repr__", &maps::Map<T>::__repr__);

        add_Map_extras(PyMap, docs);

    }

    template <int Module>
    void add_extras(py::module& m, const docstrings::docs<Module>& docs) { }

    template <>
    void add_extras(py::module& m,
                    const docstrings::docs<STARRY_MODULE_MULTI>& docs) {
        m.attr("NMULTI") = STARRY_NMULTI;
    }

    template <typename T, int Module>
    void add_starry(py::module& m, const docstrings::docs<Module>& docs) {

        // Main docs
        m.doc() = docs.doc;

        // Type-specific stuff
        add_extras(m, docs);

        // Surface map class
        py::class_<maps::Map<T>> PyMap(m, "Map", docs.Map.doc);
        add_Map(PyMap, docs);

    }

}; // namespace pybind_interface

#endif
