/**
This defines the main Python interface to the code.

TODO: There is a lot of ugly looping and copying in the
      routines that transform the gradients into python
      dictionaries. There are four of them here (two in
      Body, two in System) and two in `pybind_vectorize.h`.
      These need to be sped up.

TODO: Add wavelength-dependent radius support
      Two options: arbitrary r(lambda), full computation
      or linear expansion about mean radius using autodiff.

TODO: Add the `add_gaussian` method to spectral starry.

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
#include <type_traits>
#include "maps.h"
#include "docstrings.h"
#include "utils.h"
#include "errors.h"
#include "kepler.h"
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
    Add type-specific features to the Map class: single-wavelength starry.

    */
    template <typename T>
    typename std::enable_if<!std::is_base_of<Eigen::EigenBase<Row<T>>,
                                             Row<T>>::value, void>::type
    addMapExtras(py::class_<maps::Map<T>>& Map) {

        Map

            .def("show", [](maps::Map<T> &map, std::string cmap, int res) {
                py::object show =
                    py::module::import("starry.maps").attr("show");
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
            }, docstrings::Map::show, "cmap"_a="plasma", "res"_a=300)

            .def("animate", [](maps::Map<T> &map, std::string cmap, int res,
                               int frames, int interval, std::string& gif)
                               -> py::object{
                std::cout << "Rendering..." << std::endl;
                py::object animate =
                    py::module::import("starry.maps").attr("animate");
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
                return animate(I, "cmap"_a=cmap, "res"_a=res, "gif"_a=gif,
                               "interval"_a=interval);
             }, docstrings::Map::animate, "cmap"_a="plasma", "res"_a=150,
                                  "frames"_a=50, "interval"_a=75, "gif"_a="")

             .def("load_image", [](maps::Map<T> &map, std::string& image,
                                   int lmax) {
                 py::object load_map =
                    py::module::import("starry.maps").attr("load_map");
                 if (lmax == -1)
                    lmax = map.lmax;
                 Vector<double> y_double =
                    load_map(image, map.lmax).template cast<Vector<double>>();
                 T y = y_double.template cast<Scalar<T>>();
                 y /= y(0);
                 int n = 0;
                 for (int l = 0; l < lmax + 1; ++l) {
                     for (int m = -l; m < l + 1; ++m) {
                         map.setY(l, m, y(n));
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
             }, docstrings::Map::load_image, "image"_a, "lmax"_a=-1)

             .def("load_image", [](maps::Map<T> &map,
                                   const Matrix<double>& image,
                                   int lmax) {
                 py::object load_map =
                    py::module::import("starry.maps").attr("load_map");
                 if (lmax == -1)
                    lmax = map.lmax;
                 Vector<double> y_double =
                    load_map(image, map.lmax).template cast<Vector<double>>();
                 T y = y_double.template cast<Scalar<T>>();
                 y /= y(0);
                 int n = 0;
                 for (int l = 0; l < lmax + 1; ++l) {
                     for (int m = -l; m < l + 1; ++m) {
                         map.setY(l, m, y(n));
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
             }, docstrings::Map::load_image, "image"_a, "lmax"_a=-1)

             .def("load_healpix", [](maps::Map<T> &map,
                                     const Vector<double>& image,
                                     int lmax) {
                 py::object load_map =
                    py::module::import("starry.maps").attr("load_map");
                 if (lmax == -1)
                    lmax = map.lmax;
                 Vector<double> y_double =
                    load_map(image, map.lmax, true).template cast<Vector<double>>();
                 T y = y_double.template cast<Scalar<T>>();
                 y /= y(0);
                 int n = 0;
                 for (int l = 0; l < lmax + 1; ++l) {
                     for (int m = -l; m < l + 1; ++m) {
                         map.setY(l, m, y(n));
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
             }, docstrings::Map::load_healpix, "image"_a, "lmax"_a=-1)

             .def("add_gaussian", [](maps::Map<T> &map, const double& sigma,
                                     const double& amp, const double& lat,
                                     const double& lon, int lmax) {
                py::object gaussian =
                    py::module::import("starry.maps").attr("gaussian");
                if (lmax == -1)
                   lmax = map.lmax;
                Vector<double> y = amp *
                    gaussian(sigma, map.lmax).template cast<Vector<double>>();

                // Create a temporary map and add the gaussian
                maps::Map<Vector<double>> tmpmap(map.lmax);
                tmpmap.setY(y);

                // Rotate it to the sub-observer point
                tmpmap.setAxis(xhat<double>());
                tmpmap.rotate(90.0);
                tmpmap.setAxis(zhat<double>());
                tmpmap.rotate(180.0);
                tmpmap.setAxis(yhat<double>());
                tmpmap.rotate(90.0);

                // Now rotate it to where the user wants it
                tmpmap.setAxis(xhat<double>());
                tmpmap.rotate(-lat);
                tmpmap.setAxis(yhat<double>());
                tmpmap.rotate(lon);

                // Add it to the current map
                for (int l = 0; l < lmax + 1; ++l) {
                    for (int m = -l; m < l + 1; ++m) {
                        map.setY(l, m, map.getY(l, m) + tmpmap.getY(l, m));
                    }
                }
            }, docstrings::Map::add_gaussian, "sigma"_a=0.1, "amp"_a=1,
                "lat"_a=0, "lon"_a=0, "lmax"_a=-1);

    }

    /**
    Add type-specific features to the Map class: spectral starry.

    */
    template <typename T>
    typename std::enable_if<std::is_base_of<Eigen::EigenBase<Row<T>>,
                                            Row<T>>::value, void>::type
    addMapExtras(py::class_<maps::Map<T>>& Map) {

        Map

            .def("show", [](maps::Map<T> &map, std::string cmap,
                            int res, std::string& gif, bool show_labels) -> py::object {
                std::cout << "Rendering..." << std::endl;
                py::object animate =
                    py::module::import("starry.maps").attr("animate");
                std::vector<Matrix<double>> I;
                Vector<Scalar<T>> x;
                VectorT<Scalar<T>> row;
                std::vector<std::string> labels;
                x = Vector<Scalar<T>>::LinSpaced(res, -1, 1);
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
                            I[t](j, i) = static_cast<double>(row(t));
                        }
                    }
                }
                if (show_labels)
                    return animate(I, "cmap"_a=cmap, "res"_a=res, "gif"_a=gif,
                                      "labels"_a=labels, "interval"_a=interval);
                else
                    return animate(I, "cmap"_a=cmap, "res"_a=res, "gif"_a=gif,
                                      "interval"_a=interval);
            }, docstrings::Map::show, "cmap"_a="plasma",
               "res"_a=150, "gif"_a="", "show_labels"_a=true)

            .def("load_image", [](maps::Map<T> &map,
                                  const Matrix<double>& image,
                                  int nwav, int lmax) {
                py::object load_map =
                    py::module::import("starry.maps").attr("load_map");
                if (lmax == -1)
                    lmax = map.lmax;
                // Below, we rotate the entire map to get it to the
                // right orientation after loading the image. In order
                // to not screw up the map at other wavelengths, we can
                // pre-apply the opposite transformation.
                // NOTE: I can think of far better ways of doing this, but
                // I don't think there's a pressing need to optimize this
                // function.
                auto map_axis = map.getAxis();
                map.setAxis(yhat<Scalar<T>>());
                map.rotate(-90.0);
                map.setAxis(zhat<Scalar<T>>());
                map.rotate(-180.0);
                map.setAxis(xhat<Scalar<T>>());
                map.rotate(-90.0);
                map.setAxis(map_axis);
                Vector<double> y_double =
                    load_map(image, map.lmax).template cast<Vector<double>>();
                T y = y_double.template cast<Scalar<T>>();
                Row<T> row;
                int n = 0;
                for (int l = 0; l < lmax + 1; ++l) {
                    for (int m = -l; m < l + 1; ++m) {
                        row = map.getY(l, m);
                        row(nwav) = y(n) / y(0);
                        map.setY(l, m, row);
                        ++n;
                    }
                }
                // We need to apply some rotations to get to the
                // desired orientation, where the center of the image
                // is projected onto the sub-observer point
                map.setAxis(xhat<Scalar<T>>());
                map.rotate(90.0);
                map.setAxis(zhat<Scalar<T>>());
                map.rotate(180.0);
                map.setAxis(yhat<Scalar<T>>());
                map.rotate(90.0);
                map.setAxis(map_axis);
            }, docstrings::Map::load_image, "image"_a, "nwav"_a=0, "lmax"_a=-1)

            .def("load_image", [](maps::Map<T> &map, std::string& image,
                                  int nwav, int lmax) {
                py::object load_map =
                    py::module::import("starry.maps").attr("load_map");
                if (lmax == -1)
                    lmax = map.lmax;
                // Below, we rotate the entire map to get it to the
                // right orientation after loading the image. In order
                // to not screw up the map at other wavelengths, we can
                // pre-apply the opposite transformation.
                // TODO: This is unnecessarily slow b/c of all the rotations.
                // I can think of far better ways of doing this.
                auto map_axis = map.getAxis();
                map.setAxis(yhat<Scalar<T>>());
                map.rotate(-90.0);
                map.setAxis(zhat<Scalar<T>>());
                map.rotate(-180.0);
                map.setAxis(xhat<Scalar<T>>());
                map.rotate(-90.0);
                map.setAxis(map_axis);
                Vector<double> y_double =
                    load_map(image, map.lmax).template cast<Vector<double>>();
                T y = y_double.template cast<Scalar<T>>();
                Row<T> row;
                int n = 0;
                for (int l = 0; l < lmax + 1; ++l) {
                    for (int m = -l; m < l + 1; ++m) {
                        row = map.getY(l, m);
                        row(nwav) = y(n) / y(0);
                        map.setY(l, m, row);
                        ++n;
                    }
                }
                // We need to apply some rotations to get to the
                // desired orientation, where the center of the image
                // is projected onto the sub-observer point
                map.setAxis(xhat<Scalar<T>>());
                map.rotate(90.0);
                map.setAxis(zhat<Scalar<T>>());
                map.rotate(180.0);
                map.setAxis(yhat<Scalar<T>>());
                map.rotate(90.0);
                map.setAxis(map_axis);
            }, docstrings::Map::load_image, "image"_a, "nwav"_a=0, "lmax"_a=-1)

            .def("load_healpix", [](maps::Map<T> &map, const Vector<double>& image,
                                  int nwav, int lmax) {
                py::object load_map =
                    py::module::import("starry.maps").attr("load_map");
                if (lmax == -1)
                    lmax = map.lmax;
                // Below, we rotate the entire map to get it to the
                // right orientation after loading the image. In order
                // to not screw up the map at other wavelengths, we can
                // pre-apply the opposite transformation.
                // TODO: This is unnecessarily slow b/c of all the rotations.
                // I can think of far better ways of doing this.
                auto map_axis = map.getAxis();
                map.setAxis(yhat<Scalar<T>>());
                map.rotate(-90.0);
                map.setAxis(zhat<Scalar<T>>());
                map.rotate(-180.0);
                map.setAxis(xhat<Scalar<T>>());
                map.rotate(-90.0);
                map.setAxis(map_axis);
                Vector<double> y_double =
                    load_map(image, map.lmax, true).template cast<Vector<double>>();
                T y = y_double.template cast<Scalar<T>>();
                Row<T> row;
                int n = 0;
                for (int l = 0; l < lmax + 1; ++l) {
                    for (int m = -l; m < l + 1; ++m) {
                        row = map.getY(l, m);
                        row(nwav) = y(n) / y(0);
                        map.setY(l, m, row);
                        ++n;
                    }
                }
                // We need to apply some rotations to get to the
                // desired orientation, where the center of the image
                // is projected onto the sub-observer point
                map.setAxis(xhat<Scalar<T>>());
                map.rotate(90.0);
                map.setAxis(zhat<Scalar<T>>());
                map.rotate(180.0);
                map.setAxis(yhat<Scalar<T>>());
                map.rotate(90.0);
                map.setAxis(map_axis);
            }, docstrings::Map::load_healpix, "image"_a, "nwav"_a=0, "lmax"_a=-1)

            .def("add_gaussian", [](maps::Map<T> &map, py::args args,
                                    py::kwargs kwargs) {
                throw errors::NotImplementedError("This routine has not yet been implemented for maps with nwav > 1.");
           });

    }

    /**
    The pybind wrapper for the Map class.

    */
    template <typename T>
    py::class_<maps::Map<T>> bindMap(py::module& m, const char* name) {

        // Declare the class
        py::class_<maps::Map<T>> Map(m, name, docstrings::Map::doc);

        // Add generic attributes & methods
        Map

            // Constructor
            .def(py::init<int, int>(), "lmax"_a=2, "nwav"_a=1)

            // Number of wavelength bins
            .def_property_readonly("nwav",
                [](maps::Map<T> &map){
                    return map.nwav;
            }, docstrings::Map::nwav)

            // Is multiprecision enabled?
            .def_property_readonly("multi",
                [](maps::Map<T> &map){
                    Scalar<T> foo;
                    return isMulti(foo);
            }, docstrings::Map::multi)

            // Floating point precision
            .def_property_readonly("precision",
                [](maps::Map<T> &map){
                    return precision<Scalar<T>>();
            }, docstrings::Map::precision)

            // Set one or more spherical harmonic coeffs to the same value
            .def("__setitem__", [](maps::Map<T>& map, py::tuple lm,
                                   RowDouble<T>& coeff) {
                auto inds = get_Ylm_inds(map.lmax, lm);
                auto y = map.getY();
                for (auto n : inds)
                    setRow(y, n, coeff);
                map.setY(y);
            })

            // Set one or more spherical harmonic coeffs to an array of values
            .def("__setitem__", [](maps::Map<T>& map, py::tuple lm,
                                   MapDouble<T>& coeff_) {
                auto inds = get_Ylm_inds(map.lmax, lm);
                T coeff = coeff_.template cast<Scalar<T>>();
                if (coeff.rows() != static_cast<long>(inds.size()))
                    throw errors::ValueError("Mismatch in slice length and "
                                             "coefficient array size.");
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
            .def("__getitem__", [](maps::Map<T>& map,
                                   py::tuple lm) -> py::object {
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
            .def("__setitem__", [](maps::Map<T>& map, py::object l,
                                   RowDouble<T>& coeff) {
                auto inds = get_Ul_inds(map.lmax, l);
                auto u = map.getU();
                for (auto n : inds)
                    setRow(u, n - 1, coeff);
                map.setU(u);
            })

            // Set one or more limb darkening coefficients to an array of values
            .def("__setitem__", [](maps::Map<T>& map, py::object l,
                                   MapDouble<T>& coeff_) {
                auto inds = get_Ul_inds(map.lmax, l);
                T coeff = coeff_.template cast<Scalar<T>>();
                if (coeff.rows() != static_cast<long>(inds.size()))
                    throw errors::ValueError("Mismatch in slice length and "
                                             "coefficient array size.");
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
            .def("__getitem__", [](maps::Map<T>& map,
                                   py::object l) -> py::object {
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
                }, docstrings::Map::evaluate, "theta"_a=0.0,
                   "x"_a=0.0, "y"_a=0.0)

            .def_property("axis",
                [](maps::Map<T> &map) -> UnitVector<double> {
                        return map.getAxis().template cast<double>();
                    },
                [](maps::Map<T> &map, UnitVector<double>& axis){
                        map.setAxis(axis.template cast<Scalar<T>>());
                    },
                docstrings::Map::axis)

            .def("reset", &maps::Map<T>::reset, docstrings::Map::reset)

            .def_property_readonly("lmax", [](maps::Map<T> &map){
                    return map.lmax;
                }, docstrings::Map::lmax)

            .def_property_readonly("N", [](maps::Map<T> &map){
                    return map.N;
                }, docstrings::Map::N)

            .def_property_readonly("y", [](maps::Map<T> &map) -> MapDouble<T>{
                        return map.getY().template cast<double>();
                    }, docstrings::Map::y)

            .def_property_readonly("u", [](maps::Map<T> &map) -> MapDouble<T>{
                        return map.getU().template cast<double>();
                    }, docstrings::Map::u)

            .def_property_readonly("p", [](maps::Map<T> &map) -> MapDouble<T>{
                    return map.getP().template cast<double>();
                }, docstrings::Map::p)

            .def_property_readonly("g", [](maps::Map<T> &map) -> MapDouble<T>{
                    return map.getG().template cast<double>();
                }, docstrings::Map::g)

            .def_property_readonly("r", [](maps::Map<T> &map)
                                        -> VectorT<double>{
                    return map.getR().template cast<double>();
                }, docstrings::Map::r)

            .def_property_readonly("s", [](maps::Map<T> &map)
                                        -> VectorT<double>{
                    return map.getS().template cast<double>();
                }, docstrings::Map::s)

            .def("flux", [](maps::Map<T> &map,
                            py::array_t<double>& theta,
                            py::array_t<double>& xo,
                            py::array_t<double>& yo,
                            py::array_t<double>& ro,
                            bool gradient,
                            bool numerical)
                            -> py::object {
                    return vectorize::flux(map, theta, xo, yo, ro,
                                           gradient, numerical);
                }, docstrings::Map::flux, "theta"_a=0.0, "xo"_a=0.0, "yo"_a=0.0,
                                   "ro"_a=0.0, "gradient"_a=false,
                                   "numerical"_a=false)

            .def("rotate", [](maps::Map<T> &map, double theta) {
                    map.rotate(static_cast<Scalar<T>>(theta));
            }, docstrings::Map::rotate, "theta"_a=0)

            .def("is_physical", [](maps::Map<T> &map, double epsilon,
                                   int max_iterations) {
                    return map.isPhysical(static_cast<Scalar<T>>(epsilon),
                                   max_iterations);
            }, docstrings::Map::is_physical, "epsilon"_a=1.e-6,
               "max_iterations"_a=100)

            .def("__repr__", &maps::Map<T>::info);

        // Add type-specific attributes & methods
        addMapExtras(Map);

        return Map;

    }

    /**
    Add type-specific features to the Body class: single-wavelength starry.

    */
    template <typename T>
    typename std::enable_if<!std::is_base_of<Eigen::EigenBase<Row<T>>,
                                             Row<T>>::value, void>::type
    addBodyExtras(py::class_<kepler::Body<T>>& Body) {

        Body

            // Luminosity in units of primary luminosity
            .def_property("L",
                [](kepler::Body<T> &body) {
                    return static_cast<double>(body.getLuminosity());
                },
                [](kepler::Body<T> &body, const double& L){
                    body.setLuminosity(Scalar<T>(L));
                }, docstrings::Body::L)

            // The gradient of the light curve: a dictionary of matrices/vectors
            // NOTE: This may be slow because we need to swap some axes here:
            //      dL(NT)(ngrad, nwav) --> gradient(ngrad)(NT, nwav)
            // I haven't figured out a way of avoiding this yet...
            .def_property_readonly("gradient", [](kepler::Body<T> &body) -> py::object {
                const Vector<T>& dL = body.getLightcurveGradient();
                const std::vector<std::string> dL_names = body.getLightcurveGradientNames();
                size_t sz = dL.size();
                std::map<std::string, Matrix<double>> gradient;
                std::string param;
                int n;

                // Allocate memory for non-map coeffs
                for (auto name : dL_names) {
                    if ((name.substr(1, 2) != ".y") && (name.substr(1, 2) != ".u"))
                        gradient[name].resize(sz, 1);
                }

                // Allocate memory for the Ylms and Uls
                for (auto var : {".y", ".u"}) {
                    n = 0;
                    for (auto name : dL_names) {
                        if (name.substr(1, 2) == var) {
                            param = name;
                            ++n;
                        } else if (n > 0) {
                            gradient[param].resize(sz, n);
                            n = 0;
                        }
                    }
                    if (n > 0)
                        gradient[param].resize(sz, n);
                }

                // Populate the dictionary
                // Loop over all times
                for (size_t t = 0; t < sz; ++t) {
                    // Loop over all params
                    n = 0;
                    for (size_t i = 0; i < dL_names.size(); ++i) {
                        if (dL_names[i].substr(1, 2) == ".y") {
                            if ((i > 0) && (dL_names[i - 1].substr(1, 2) != ".y"))
                                n = 0;
                            gradient[dL_names[i]](t, n++) = static_cast<double>(dL(t)(i));
                        } else if (dL_names[i].substr(1, 2) == ".u") {
                            if ((i > 0) && (dL_names[i - 1].substr(1, 2) != ".u"))
                                n = 0;
                            gradient[dL_names[i]](t, n++) = static_cast<double>(dL(t)(i));
                        } else {
                            gradient[dL_names[i]](t, 0) = static_cast<double>(dL(t)(i));
                        }
                    }
                }

                // Convert to an actual python dictionary
                // Necessary because we're mixing vectors and matrices
                // among the dictionary items.
                // NOTE: All this copying could be slow: not ideal.
                auto pygrad = py::dict();
                for (auto& entry : gradient) {
                    std::string const& name = entry.first;
                    if ((name.substr(1, 2) != ".y") && (name.substr(1, 2) != ".u"))
                        pygrad[name.c_str()] = gradient[name].col(0);
                    else
                        pygrad[name.c_str()] = gradient[name].transpose();
                }
                return pygrad;

            }, docstrings::Body::gradient);

    }

    /**
    Add type-specific features to the Body class: spectral starry.

    */
    template <typename T>
    typename std::enable_if<std::is_base_of<Eigen::EigenBase<Row<T>>,
                                            Row<T>>::value, void>::type
    addBodyExtras(py::class_<kepler::Body<T>>& Body) {

        Body

            // Luminosity in units of primary luminosity
            .def_property("L",
                [](kepler::Body<T> &body) {
                    return body.getLuminosity().template cast<double>();
                },
                [](kepler::Body<T> &body, const Vector<double>& L){
                    body.setLuminosity(L.template cast<Scalar<T>>());
                }, docstrings::Body::L)

            // The gradient of the light curve: a dictionary of matrices/vectors
            // NOTE: This may be slow because we need to swap some axes here:
            //      dL(NT)(ngrad, nwav) --> gradient(ngrad)(NT, nwav)
            // I haven't figured out a way of avoiding this yet...
            .def_property_readonly("gradient", [](kepler::Body<T> &body) -> py::object {
                const Vector<T>& dL = body.getLightcurveGradient();
                const std::vector<std::string> dL_names = body.getLightcurveGradientNames();
                size_t sz = dL.size();
                std::map<std::string, std::vector<Matrix<double>>> gradient;
                int n;

                // Allocate memory
                for (auto name : dL_names) {
                    gradient[name].push_back(Matrix<double>(sz, body.nwav));
                }

                // Populate the dictionary
                // Loop over all times
                for (size_t t = 0; t < sz; ++t) {
                    // Loop over all params
                    n = 0;
                    for (size_t i = 0; i < dL_names.size(); ++i) {
                        if (dL_names[i].substr(1, 2) == ".y") {
                            if ((i > 0) && (dL_names[i - 1].substr(1, 2) != ".y"))
                                n = 0;
                            gradient[dL_names[i]][n++].row(t) = dL(t).row(i).template cast<double>();
                        } else if (dL_names[i].substr(1, 2) == ".u") {
                            if ((i > 0) && (dL_names[i - 1].substr(1, 2) != ".u"))
                                n = 0;
                            gradient[dL_names[i]][n++].row(t) = dL(t).row(i).template cast<double>();
                        } else {
                            gradient[dL_names[i]][0].row(t) = dL(t).row(i).template cast<double>();
                        }
                    }
                }

                // Convert to an actual python dictionary. Necessary so
                // we can transform 1-element lists of matrices to
                // straight up matrices
                // NOTE: All this copying could be slow: not ideal.
                auto pygrad = py::dict();
                for (auto& entry : gradient) {
                    std::string const& name = entry.first;
                    if ((name.substr(1, 2) != ".y") && (name.substr(1, 2) != ".u"))
                        pygrad[name.c_str()] = gradient[name][0];
                    else
                        pygrad[name.c_str()] = gradient[name];
                }
                return pygrad;

            }, docstrings::Body::gradient);

    }

    /**
    The pybind wrapper for the Body class.

    */
    template <typename T>
    py::class_<kepler::Body<T>> bindBody(py::module& m,
                                         py::class_<maps::Map<T>> Map,
                                         const char* name) {

        // Declare the class
        py::class_<kepler::Body<T>> Body(m, name, Map);

        // Add generic attributes & methods
        Body

            // Radius in units of primary radius
            .def_property("r",
                [](kepler::Body<T> &body) {
                    return static_cast<double>(body.getRadius());
                },
                [](kepler::Body<T> &body, const double& r){
                    body.setRadius(r);
                }, docstrings::Body::r)

            // Rotation period in days
            .def_property("prot",
                [](kepler::Body<T> &body) {
                    return static_cast<double>(body.getRotPer());
                },
                [](kepler::Body<T> &body, const double& prot){
                    body.setRotPer(prot);
                }, docstrings::Body::prot)

            // Reference time in days
            .def_property("tref",
                [](kepler::Body<T> &body) {
                    return static_cast<double>(body.getRefTime());
                },
                [](kepler::Body<T> &body, const double& tref){
                    body.setRefTime(tref);
                }, docstrings::Body::tref)

            // The computed light curve: a matrix or a vector
            .def_property_readonly("lightcurve", [](kepler::Body<T> &body)
                    -> py::object{
                if (body.nwav == 1) {
                    return py::cast(getColumn(
                            body.getLightcurve(), 0).template cast<double>());
                } else {
                    return py::cast(
                            body.getLightcurve().template cast<double>());
                }
            }, docstrings::Body::lightcurve);

        // Add type-specific attributes & methods
        addBodyExtras(Body);

        return Body;
    }

    /**
    Add type-specific features to the Body class: single-wavelength starry.

    */
    template <typename T>
    typename std::enable_if<!std::is_base_of<Eigen::EigenBase<Row<T>>,
                                            Row<T>>::value, void>::type
    addPrimaryExtras(py::class_<kepler::Primary<T>>& Primary) {

        Primary

            // Luminosity in units of primary luminosity
            .def_property("L",
                [](kepler::Primary<T> &body) {
                    return static_cast<double>(body.getLuminosity());
                },
                [](kepler::Primary<T> &body, const double& L){
                    body.setLuminosity(Scalar<T>(L));
                }, docstrings::Primary::L);

    }

    /**
    Add type-specific features to the Body class: spectral starry.

    */
    template <typename T>
    typename std::enable_if<std::is_base_of<Eigen::EigenBase<Row<T>>,
                                            Row<T>>::value, void>::type
    addPrimaryExtras(py::class_<kepler::Primary<T>>& Primary) {

        Primary

            // Luminosity in units of primary luminosity
            .def_property("L",
                [](kepler::Primary<T> &body) {
                    return body.getLuminosity().template cast<double>();
                },
                [](kepler::Primary<T> &body, const Vector<double>& L){
                    body.setLuminosity(L.template cast<Scalar<T>>());
                }, docstrings::Primary::L);

    }

    /**
    The pybind wrapper for the Primary class.

    */
    template <typename T>
    py::class_<maps::Map<T>> bindPrimary(py::module& m,
                                         py::class_<kepler::Body<T>> Body,
                                         const char* name) {

        // Declare the class
        py::class_<kepler::Primary<T>> Primary(m, name, Body,
                                               docstrings::Primary::doc);

        // Add generic attributes & methods
        Primary

            // Constructor
            .def(py::init<int, int>(), "lmax"_a=2, "nwav"_a=1)

            // Radius in units of primary radius
            .def_property("r",
                [](kepler::Primary<T> &body) {
                    return static_cast<double>(body.getRadius());
                },
                [](kepler::Primary<T> &body, const double& r){
                    body.setRadius(r);
                }, docstrings::Primary::r)

            // Radius in meters (sets a scale for light travel time delay)
            .def_property("r_m",
                [](kepler::Primary<T> &body) {
                    return static_cast<double>(body.getRadiusInMeters());
                },
                [](kepler::Primary<T> &body, const double& r_m){
                    body.setRadiusInMeters(r_m);
                }, docstrings::Primary::r_m);

        // Add type-specific attributes & methods
        addPrimaryExtras(Primary);

        return Primary;

    }

    /**
    The pybind wrapper for the Secondary class.

    */
    template <typename T>
    py::class_<maps::Map<T>> bindSecondary(py::module& m,
                                           py::class_<kepler::Body<T>> Body,
                                           const char* name) {

        // Declare the class
        py::class_<kepler::Secondary<T>> Secondary(m, name, Body,
                                                   docstrings::Secondary::doc);

        // Add generic attributes & methods
        Secondary

            // Constructor
            .def(py::init<int, int>(), "lmax"_a=2, "nwav"_a=1)

            // Semi-major axis in units of primary radius
            .def_property("a",
                [](kepler::Secondary<T> &sec) {
                    return static_cast<double>(sec.getSemi());
                },
                [](kepler::Secondary<T> &sec, const double& a){
                    sec.setSemi(a);
                }, docstrings::Secondary::a)

            // Orbital period in days
            .def_property("porb",
                [](kepler::Secondary<T> &sec) {
                    return static_cast<double>(sec.getOrbPer());
                },
                [](kepler::Secondary<T> &sec, const double& p){
                    sec.setOrbPer(p);
                }, docstrings::Secondary::porb)

            // Semi-major axis
            .def_property("inc",
                [](kepler::Secondary<T> &sec) {
                    return static_cast<double>(sec.getInc());
                },
                [](kepler::Secondary<T> &sec, const double& i){
                    sec.setInc(i);
                }, docstrings::Secondary::inc)

            // Eccentricity
            .def_property("ecc",
                [](kepler::Secondary<T> &sec) {
                    return static_cast<double>(sec.getEcc());
                },
                [](kepler::Secondary<T> &sec, const double& ecc){
                    sec.setEcc(ecc);
                }, docstrings::Secondary::ecc)

            // Longitude of pericenter (varpi) in degrees
            .def_property("w",
                [](kepler::Secondary<T> &sec) {
                    return static_cast<double>(sec.getVarPi());
                },
                [](kepler::Secondary<T> &sec, const double& w){
                    sec.setVarPi(w);
                }, docstrings::Secondary::w)

            // Longitude of ascending node in degrees
            .def_property("Omega",
                [](kepler::Secondary<T> &sec) {
                    return static_cast<double>(sec.getOmega());
                },
                [](kepler::Secondary<T> &sec, const double& Om){
                    sec.setOmega(Om);
                }, docstrings::Secondary::Omega)

            // Mean longitude at the reference time in degrees
            .def_property("lambda0",
                [](kepler::Secondary<T> &sec) {
                    return static_cast<double>(sec.getLambda0());
                },
                [](kepler::Secondary<T> &sec, const double& l0){
                    sec.setLambda0(l0);
                }, docstrings::Secondary::lambda0)

            // Cartesian x position vector
            .def_property_readonly("X",
                [](kepler::Secondary<T> &sec) {
                    return sec.getXVector().template cast<double>();
                }, docstrings::Secondary::X)

            // Cartesian y position vector
            .def_property_readonly("Y",
                [](kepler::Secondary<T> &sec) {
                    return sec.getYVector().template cast<double>();
                }, docstrings::Secondary::Y)

            // Cartesian z position vector
            .def_property_readonly("Z",
                [](kepler::Secondary<T> &sec) {
                    return sec.getZVector().template cast<double>();
                }, docstrings::Secondary::Z);

        return Secondary;

    }

    /**
    Add type-specific features to the System class: single-wavelength starry.

    */
    template <typename T>
    typename std::enable_if<!std::is_base_of<Eigen::EigenBase<Row<T>>,
                                             Row<T>>::value, void>::type
    addSystemExtras(py::class_<kepler::System<T>>& System) {

        System

            // The gradient of the light curve: a dictionary of matrices/vectors
            // NOTE: This may be slow because we need to swap some axes here:
            //      dL(NT)(ngrad, nwav) --> gradient(ngrad)(NT, nwav)
            // I haven't figured out a way of avoiding this yet...
            .def_property_readonly("gradient", [](kepler::System<T> &sys) -> py::object {
                const Vector<T>& dL = sys.getLightcurveGradient();
                const std::vector<std::string> dL_names = sys.getLightcurveGradientNames();
                size_t sz = dL.size();
                std::map<std::string, Matrix<double>> gradient;
                std::string param;
                int n;

                // Allocate memory for non-map coeffs
                for (auto name : dL_names) {
                    if ((name.substr(1, 2) != ".y") && (name.substr(1, 2) != ".u"))
                        gradient[name].resize(sz, 1);
                }

                // Allocate memory for the Ylms and Uls
                for (auto var : {".y", ".u"}) {
                    n = 0;
                    for (auto name : dL_names) {
                        if (name.substr(1, 2) == var) {
                            param = name;
                            ++n;
                        } else if (n > 0) {
                            gradient[param].resize(sz, n);
                            n = 0;
                        }
                    }
                    if (n > 0)
                        gradient[param].resize(sz, n);
                }

                // Populate the dictionary
                // Loop over all times
                for (size_t t = 0; t < sz; ++t) {
                    // Loop over all params
                    n = 0;
                    for (size_t i = 0; i < dL_names.size(); ++i) {
                        if (dL_names[i].substr(1, 2) == ".y") {
                            if ((i > 0) && (dL_names[i - 1].substr(1, 2) != ".y"))
                                n = 0;
                            gradient[dL_names[i]](t, n++) = static_cast<double>(dL(t)(i));
                        } else if (dL_names[i].substr(1, 2) == ".u") {
                            if ((i > 0) && (dL_names[i - 1].substr(1, 2) != ".u"))
                                n = 0;
                            gradient[dL_names[i]](t, n++) = static_cast<double>(dL(t)(i));
                        } else {
                            gradient[dL_names[i]](t, 0) = static_cast<double>(dL(t)(i));
                        }
                    }
                }

                // Convert to an actual python dictionary
                // Necessary because we're mixing vectors and matrices
                // among the dictionary items.
                // NOTE: All this copying could be slow: not ideal.
                auto pygrad = py::dict();
                for (auto& entry : gradient) {
                    std::string const& name = entry.first;
                    if ((name.substr(1, 2) != ".y") && (name.substr(1, 2) != ".u"))
                        pygrad[name.c_str()] = gradient[name].col(0);
                    else
                        pygrad[name.c_str()] = gradient[name].transpose();
                }
                return pygrad;

            }, docstrings::System::gradient);

    }

    /**
    Add type-specific features to the System class: spectral starry.

    */
    template <typename T>
    typename std::enable_if<std::is_base_of<Eigen::EigenBase<Row<T>>,
                                            Row<T>>::value, void>::type
    addSystemExtras(py::class_<kepler::System<T>>& System) {

        System

            // The gradient of the light curve: a dictionary of matrices/vectors
            // NOTE: This may be slow because we need to swap some axes here:
            //      dL(NT)(ngrad, nwav) --> gradient(ngrad)(NT, nwav)
            // I haven't figured out a way of avoiding this yet...
            .def_property_readonly("gradient", [](kepler::System<T> &sys) -> py::object {
                const Vector<T>& dL = sys.getLightcurveGradient();
                const std::vector<std::string> dL_names = sys.getLightcurveGradientNames();
                size_t sz = dL.size();
                std::map<std::string, std::vector<Matrix<double>>> gradient;
                std::string param;
                int n;

                // Allocate memory
                for (auto name : dL_names) {
                    gradient[name].push_back(Matrix<double>(sz, sys.primary->nwav));
                }

                // Populate the dictionary
                // Loop over all times
                for (size_t t = 0; t < sz; ++t) {
                    // Loop over all params
                    n = 0;
                    for (size_t i = 0; i < dL_names.size(); ++i) {
                        if (dL_names[i].substr(1, 2) == ".y") {
                            if ((i > 0) && (dL_names[i - 1].substr(1, 2) != ".y"))
                                n = 0;
                            gradient[dL_names[i]][n++].row(t) = dL(t).row(i).template cast<double>();
                        } else if (dL_names[i].substr(1, 2) == ".u") {
                            if ((i > 0) && (dL_names[i - 1].substr(1, 2) != ".u"))
                                n = 0;
                            gradient[dL_names[i]][n++].row(t) = dL(t).row(i).template cast<double>();
                        } else {
                            gradient[dL_names[i]][0].row(t) = dL(t).row(i).template cast<double>();
                        }
                    }
                }

                // Convert to an actual python dictionary. Necessary so
                // we can transform 1-element lists of matrices to
                // straight up matrices
                // NOTE: All this copying could be slow: not ideal.
                auto pygrad = py::dict();
                for (auto& entry : gradient) {
                    std::string const& name = entry.first;
                    if ((name.substr(1, 2) != ".y") && (name.substr(1, 2) != ".u"))
                        pygrad[name.c_str()] = gradient[name][0];
                    else
                        pygrad[name.c_str()] = gradient[name];
                }
                return pygrad;

            }, docstrings::System::gradient);

    }

    /**
    The pybind wrapper for the System class.

    */
    template <typename T>
    py::class_<maps::Map<T>> bindSystem(py::module& m,
                                        const char* name) {

        // Declare the class
        py::class_<kepler::System<T>> System(m, name, docstrings::System::doc);

        // Add generic attributes & methods
        System

            // Constructor: one secondary
            .def(py::init<kepler::Primary<T>*, kepler::Secondary<T>*>(),
                 py::keep_alive<1, 2>(), py::keep_alive<1, 3>())

            // Constructor: multiple secondaries
            .def(py::init<kepler::Primary<T>*,
                          std::vector<kepler::Secondary<T>*>>(),
                 py::keep_alive<1, 2>(), py::keep_alive<1, 3>())

            .def_property_readonly("primary", [](kepler::System<T> &system) {
                return system.primary;
            }, docstrings::System::primary)

            .def_property_readonly("secondaries", [](kepler::System<T> &system) {
                return system.secondaries;
            }, docstrings::System::secondaries)

            // Compute the light curve
            .def("compute", [](kepler::System<T> &system,
                               const Vector<double>& time,
                               bool gradient, bool numerical) {
                system.compute(time.template cast<Scalar<T>>(), gradient, numerical);
            }, docstrings::System::compute, "time"_a, "gradient"_a=false, "numerical"_a=false)

            // Exposure time in days
            .def_property("exposure_time",
                [](kepler::System<T> &sys) {
                    return static_cast<double>(sys.getExposureTime());
                },
                [](kepler::System<T> &sys, const double& t){
                    sys.setExposureTime(t);
                }, docstrings::System::exposure_time)

            // Exposure tolerance
            .def_property("exposure_tol",
                [](kepler::System<T> &sys) {
                    return static_cast<double>(sys.getExposureTol());
                },
                [](kepler::System<T> &sys, const double& t){
                    sys.setExposureTol(t);
                }, docstrings::System::exposure_tol)

            // Exposure max recursion depth
            .def_property("exposure_max_depth",
                [](kepler::System<T> &sys) {
                    return sys.getExposureMaxDepth();
                },
                [](kepler::System<T> &sys, int d){
                    sys.setExposureMaxDepth(d);
                }, docstrings::System::exposure_max_depth)

            // The computed light curve: a matrix or a vector
            .def_property_readonly("lightcurve", [](kepler::System<T> &system)
                    -> py::object{
                if (system.primary->nwav == 1) {
                    return py::cast(getColumn(
                            system.getLightcurve(), 0).template cast<double>());
                } else {
                    return py::cast(
                            system.getLightcurve().template cast<double>());
                }
            }, docstrings::System::lightcurve)

            .def("__repr__", &kepler::System<T>::info);

        // Add type-specific attributes & methods
        addSystemExtras(System);

        return System;

    }

} // namespace pybind_interface

#endif
