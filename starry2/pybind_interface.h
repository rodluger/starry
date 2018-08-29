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
#include "pybind_utils.h"


namespace pybind_interface {

    namespace py = pybind11;
    namespace vectorize = pybind_vectorize;
    using namespace utils;
    using namespace pybind11::literals;
    using pybind_utils::get_Ylm_inds;
    using pybind_utils::get_Ul_inds;

    /**
    Add type-specific features to the Map class.

    */
    template <typename T, int Module>
    void add_Map_extras(py::class_<maps::Map<T>>& PyMap,
                        const docstrings::docs<Module>& docs) {

        PyMap

            .def(py::init<int>(), "lmax"_a=2)

            .def("show", [](maps::Map<T> &map, std::string cmap, int res) {
                py::object show = py::module::import("starry2.maps").attr("show");
                Matrix<double> I;
                I.resize(res, res);
                Vector<Scalar<T>> x;
                x = Vector<Scalar<T>>::LinSpaced(res, -1, 1);
                for (int i = 0; i < res; i++){
                    for (int j = 0; j < res; j++){
                        I(j, i) = static_cast<double>(
                                  map(0.0, x(i), x(j)));
                    }
                }
                show(I, "cmap"_a=cmap, "res"_a=res);
            }, docs.Map.show, "cmap"_a="plasma", "res"_a=300)

            .def("animate", [](maps::Map<T> &map, std::string cmap, int res,
                               int frames, int interval, std::string& gif) {
                std::cout << "Rendering..." << std::endl;
                py::object animate = py::module::import("starry2.maps").attr("animate");
                std::vector<Matrix<double>> I;
                Vector<Scalar<T>> x, theta;
                x = Vector<Scalar<T>>::LinSpaced(res, -1, 1);
                theta = Vector<Scalar<T>>::LinSpaced(frames, 0, 360);
                for (int t = 0; t < frames; t++){
                    I.push_back(Matrix<double>::Zero(res, res));
                    for (int i = 0; i < res; i++){
                        for (int j = 0; j < res; j++){
                            I[t](j, i) = static_cast<double>(
                                         map(theta(t), x(i), x(j)));
                        }
                    }
                }
                animate(I, "cmap"_a=cmap, "res"_a=res, "gif"_a=gif,
                        "interval"_a=interval);
             }, docs.Map.animate, "cmap"_a="plasma", "res"_a=150,
                                  "frames"_a=50, "interval"_a=75, "gif"_a="")

             .def("load_image", [](maps::Map<T> &map, std::string& image, int lmax) {
                 py::object load_map = py::module::import("starry.maps").attr("load_map");
                 if (lmax == -1)
                    lmax = map.lmax;
                 Vector<double> y_double = load_map(image, map.lmax).template cast<Vector<double>>();
                 T y = y_double.template cast<Scalar<T>>();
                 Scalar<T> y_normed;
                 int n = 0;
                 for (int l = 0; l < lmax + 1; ++l) {
                     for (int m = -l; m < l + 1; ++m) {
                         y_normed = y(n) / y(0);
                         map.setY(l, m, y_normed);
                         ++n;
                     }
                 }
                 // We need to apply some rotations to get to the
                 // desired orientation, where the center of the image
                 // is projected onto the sub-observer point
                 auto map_axis = map.getAxis();
                 map.setAxis(xhat<Scalar<T>>());
                 map.rotate(90.0);
                 map.setAxis(zhat<Scalar<T>>());
                 map.rotate(180.0);
                 map.setAxis(yhat<Scalar<T>>());
                 map.rotate(90.0);
                 map.setAxis(map_axis);
             }, docs.Map.load_image, "image"_a, "lmax"_a=-1);

    }

    /**
    Add type-specific features to the Map class: spectral specialization.

    TODO: Add a "load_image" method to load images at specific wavelengths.

    */
    template <>
    void add_Map_extras(py::class_<maps::Map<Matrix<double>>>& PyMap,
                        const docstrings::docs<STARRY_MODULE_SPECTRAL>& docs) {

        PyMap

            .def(py::init<int, int>(), "lmax"_a=2, "nwav"_a=1)

            .def_property_readonly("nwav",
                [](maps::Map<Matrix<double>> &map){
                    return map.nwav;
                }, docs.Map.nwav)

            .def("show", [](maps::Map<Matrix<double>> &map, std::string cmap,
                            int res, std::string& gif) {
                std::cout << "Rendering..." << std::endl;
                py::object animate = py::module::import("starry2.maps").attr("animate");
                std::vector<Matrix<double>> I;
                Vector<double> x;
                VectorT<double> row;
                std::vector<std::string> labels;
                x = Vector<double>::LinSpaced(res, -1, 1);
                int interval = static_cast<int>(75 * (50.0 / map.nwav));
                if (interval < 50)
                    interval = 50;
                else if (interval > 500)
                    interval = 500;
                for (int t = 0; t < map.nwav; t++) {
                    labels.push_back(std::string("Wavelength Bin #") +
                                     std::to_string(t + 1));
                    I.push_back(Matrix<double>::Zero(res, res));
                }
                for (int i = 0; i < res; i++){
                    for (int j = 0; j < res; j++){
                        row = map(0, x(i), x(j));
                        for (int t = 0; t < map.nwav; t++) {
                            I[t](j, i) = row(t);
                        }
                    }
                }
                animate(I, "cmap"_a=cmap, "res"_a=res, "gif"_a=gif,
                        "labels"_a=labels, "interval"_a=interval);
            }, docs.Map.show, "cmap"_a="plasma", "res"_a=150, "gif"_a="");

    }

    /**
    The pybind wrapper for the Map class.

    */
    template <typename T, int Module>
    void add_Map(py::class_<maps::Map<T>>& PyMap,
                 const docstrings::docs<Module>& docs) {

        PyMap

            // Set one or more spherical harmonic coefficients to the same value
            .def("__setitem__", [](maps::Map<T>& map, py::tuple lm, RowDouble<T>& coeff) {
                auto inds = get_Ylm_inds(map.lmax, lm);
                auto y = map.getY();
                for (auto n : inds)
                    setRow(y, n, coeff);
                map.setY(y);
            })

            // Set one or more spherical harmonic coefficients to an array of values
            .def("__setitem__", [](maps::Map<T>& map, py::tuple lm, MapDouble<T>& coeff) {
                auto inds = get_Ylm_inds(map.lmax, lm);
                if (coeff.rows() != static_cast<long>(inds.size()))
                    throw errors::ValueError("Mismatch in slice length and coefficient array size.");
                auto y = map.getY();
                int i = 0;
                Row<T> row;
                for (auto n : inds) {
                    row = getRow(coeff, i++);
                    setRow(y, n, row);
                }
                map.setY(y);
            })

            // Retrieve one or more spherical harmonic coefficients
            .def("__getitem__", [](maps::Map<T>& map, py::tuple lm) -> py::object {
                auto inds = get_Ylm_inds(map.lmax, lm);
                auto y = map.getY();
                MapDouble<T> res;
                resize(res, inds.size(), map.nwav);
                int i = 0;
                Row<T> row;
                for (auto n : inds) {
                    row = getRow(y, n);
                    setRow(res, i++, row);
                }
                if (inds.size() == 1)
                    return py::cast<RowDouble<T>>(getRow(res, 0));
                else
                    return py::cast<MapDouble<T>>(res);
            })

            // Set one or more limb darkening coefficients to the same value
            .def("__setitem__", [](maps::Map<T>& map, py::object l, RowDouble<T>& coeff) {
                auto inds = get_Ul_inds(map.lmax, l);
                auto u = map.getU();
                for (auto n : inds)
                    setRow(u, n - 1, coeff);
                map.setU(u);
            })

            // Set one or more limb darkening coefficients to an array of values
            .def("__setitem__", [](maps::Map<T>& map, py::object l, MapDouble<T>& coeff) {
                auto inds = get_Ul_inds(map.lmax, l);
                if (coeff.rows() != static_cast<long>(inds.size()))
                    throw errors::ValueError("Mismatch in slice length and coefficient array size.");
                auto u = map.getU();
                int i = 0;
                Row<T> row;
                for (auto n : inds) {
                    row = getRow(coeff, i++);
                    setRow(u, n - 1, row);
                }
                map.setU(u);
            })

            // Retrieve one or more limb darkening coefficients
            .def("__getitem__", [](maps::Map<T>& map, py::object l) -> py::object {
                auto inds = get_Ul_inds(map.lmax, l);
                auto u = map.getU();
                MapDouble<T> res;
                resize(res, inds.size(), map.nwav);
                int i = 0;
                Row<T> row;
                for (auto n : inds) {
                    row = getRow(u, n - 1);
                    setRow(res, i++, row);
                }
                if (inds.size() == 1)
                    return py::cast<RowDouble<T>>(getRow(res, 0));
                else
                    return py::cast<MapDouble<T>>(res);
            })

            // Evaluate the map intensity at a point
            .def("__call__", [](maps::Map<T> &map,
                                py::array_t<double>& theta,
                                py::array_t<double>& x,
                                py::array_t<double>& y)
                                -> py::object {
                    return vectorize::evaluate(map, theta, x, y);
                }, docs.Map.evaluate, "theta"_a=0.0, "x"_a=0.0, "y"_a=0.0)

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

            .def_property_readonly("y", [](maps::Map<T> &map) -> MapDouble<T>{
                        return map.getY().template cast<double>();
                    }, docs.Map.y)

            .def_property_readonly("u", [](maps::Map<T> &map) -> MapDouble<T>{
                        return map.getU().template cast<double>();
                    }, docs.Map.u)

            .def_property_readonly("p", [](maps::Map<T> &map) -> MapDouble<T>{
                    return map.getP().template cast<double>();
                }, docs.Map.p)

            .def_property_readonly("g", [](maps::Map<T> &map) -> MapDouble<T>{
                    return map.getG().template cast<double>();
                }, docs.Map.g)

            .def_property_readonly("r", [](maps::Map<T> &map) -> VectorT<double>{
                    return map.getR().template cast<double>();
                }, docs.Map.r)

            .def_property_readonly("s", [](maps::Map<T> &map) -> VectorT<double>{
                    return map.getS().template cast<double>();
                }, docs.Map.s)

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
