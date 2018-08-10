/**
Miscellaneous stuff used throughout the code.

*/

#ifndef _STARRY_VECTORIZE_H_
#define _STARRY_VECTORIZE_H_

#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "utils.h"
#include "errors.h"

namespace pybind_vectorize {

    using namespace utils;
    namespace py = pybind11;

    // Vectorize a single python object
    inline Vector<double> vectorize_arg(py::object& obj, int& size){
        Vector<double> res;
        if (py::isinstance<py::float_>(obj) || py::isinstance<py::int_>(obj)) {
            res = Vector<double>::Constant(size, py::cast<double>(obj));
            return res;
        } else if (py::isinstance<py::array>(obj) ||
                   py::isinstance<py::list>(obj) ||
                   py::isinstance<py::tuple>(obj)) {
            res = py::cast<Vector<double>>(obj);
            if ((size == 0) || (res.size() == size)) {
                size = res.size();
                return res;
            } else {
                throw errors::ValueError("Mismatch in argument dimensions.");
            }
        } else {
            throw errors::ValueError("Incorrect type for one or more of the arguments.");
        }
    }

    // Vectorize function of three args
    inline void vectorize_args(py::object& arg1, py::object& arg2, py::object& arg3,
                               Vector<double>& arg1_v, Vector<double>& arg2_v,
                               Vector<double>& arg3_v) {
        int size = 0;
        if (py::hasattr(arg1, "__len__")) {
            // arg1 is a vector
            arg1_v = vectorize_arg(arg1, size);
            arg2_v = vectorize_arg(arg2, size);
            arg3_v = vectorize_arg(arg3, size);
        } else if (py::hasattr(arg2, "__len__")) {
            // arg2 is a vector
            arg2_v = vectorize_arg(arg2, size);
            arg3_v = vectorize_arg(arg3, size);
            arg1_v = vectorize_arg(arg1, size);
        } else if (py::hasattr(arg3, "__len__")) {
            // arg3 is a vector
            arg3_v = vectorize_arg(arg3, size);
            arg1_v = vectorize_arg(arg1, size);
            arg2_v = vectorize_arg(arg2, size);
        } else {
            // no arg is a vector
            size = 1;
            arg1_v = vectorize_arg(arg1, size);
            arg2_v = vectorize_arg(arg2, size);
            arg3_v = vectorize_arg(arg3, size);
        }
    }

    // Vectorize function of four args
    inline void vectorize_args(py::object& arg1, py::object& arg2, py::object& arg3,
                               py::object& arg4, Vector<double>& arg1_v,
                               Vector<double>& arg2_v, Vector<double>& arg3_v,
                               Vector<double>& arg4_v) {
        int size = 0;
        if (py::hasattr(arg1, "__len__")) {
            // arg1 is a vector
            arg1_v = vectorize_arg(arg1, size);
            arg2_v = vectorize_arg(arg2, size);
            arg3_v = vectorize_arg(arg3, size);
            arg4_v = vectorize_arg(arg4, size);
        } else if (py::hasattr(arg2, "__len__")) {
            // arg2 is a vector
            arg2_v = vectorize_arg(arg2, size);
            arg3_v = vectorize_arg(arg3, size);
            arg4_v = vectorize_arg(arg4, size);
            arg1_v = vectorize_arg(arg1, size);
        } else if (py::hasattr(arg3, "__len__")) {
            // arg3 is a vector
            arg3_v = vectorize_arg(arg3, size);
            arg4_v = vectorize_arg(arg4, size);
            arg1_v = vectorize_arg(arg1, size);
            arg2_v = vectorize_arg(arg2, size);
        } else if (py::hasattr(arg4, "__len__")) {
            // arg4 is a vector
            arg4_v = vectorize_arg(arg4, size);
            arg1_v = vectorize_arg(arg1, size);
            arg2_v = vectorize_arg(arg2, size);
            arg3_v = vectorize_arg(arg3, size);
        } else {
            // no arg is a vector
            size = 1;
            arg1_v = vectorize_arg(arg1, size);
            arg2_v = vectorize_arg(arg2, size);
            arg3_v = vectorize_arg(arg3, size);
            arg4_v = vectorize_arg(arg4, size);
        }
    }


    template <typename T1, typename T2, typename T3>
    py::object flux(maps::Map<T1, T2, T3> &map,
                    py::object& theta,
                    py::object& xo,
                    py::object& yo,
                    py::object& ro,
                    bool gradient){

        if (gradient) {

            /* TODO

            // Initialize a dictionary of derivatives
            size_t n = max(theta.size(), max(xo.size(),
                           max(yo.size(), ro.size())));
            std::map<string, Vector<double>> grad;
            for (auto name : map.dF_names)
                grad[name].resize(n);

            // Nested lambda function; https://github.com/pybind/pybind11/issues/761#issuecomment-288818460
            int t = 0;
            auto F = py::vectorize([&map, &grad, &t](double theta, double xo, double yo, double ro) {
                // Evaluate the function
                double res = map.flux(theta, xo, yo, ro, true);
                // Gather the derivatives
                for (int j = 0; j < map.dF.size(); j++)
                    grad[map.dF_names[j]](t) = map.dF(j);
                t++;
                return res;
            })(theta, xo, yo, ro);

            // Return a tuple of (F, dict(dF))
            return py::make_tuple(F, grad);

            */

            return py::cast(0.0);

        } else {

            // Easy! We'll just return F
            return py::vectorize([&map](double theta, double xo, double yo, double ro) {
                        return static_cast<double>(map.flux(theta, xo, yo, ro, false));
                   })(theta, xo, yo, ro);

        }

    }

    template <>
    py::object flux(maps::Map<Matrix<double>, Vector<double>, VectorT<double>> &map,
                    py::object& theta,
                    py::object& xo,
                    py::object& yo,
                    py::object& ro,
                    bool gradient){

        if (gradient) {

            /* TODO */
            return py::cast(0.0);

        } else {

            // Vectorize the arguments manually
            Vector<double> theta_v, xo_v, yo_v, ro_v;
            vectorize_args(theta, xo, yo, ro,
                           theta_v, xo_v, yo_v, ro_v);

            // Iterate through the timeseries
            size_t sz = theta_v.size();
            Matrix<double> F(sz, map.nwav);
            for (size_t i = 0; i < sz; ++i) {
                F.row(i) = map.flux(theta_v(i), xo_v(i), yo_v(i), ro_v(i), false);
            }

            // Cast to python object
            return py::cast(F);

        }

    }

    template <typename T1, typename T2, typename T3>
    py::object evaluate(maps::Map<T1, T2, T3> &map,
                        py::object& theta,
                        py::object& x,
                        py::object& y,
                        bool gradient){

        if (gradient) {

            /* TODO

            // Initialize a dictionary of derivatives
            size_t n = max(theta.size(), max(x.size(), y.size()));
            std::map<string, Vector<double>> grad;
            for (auto name : map.dI_names)
                grad[name].resize(n);

            // Nested lambda function;
            // https://github.com/pybind/pybind11/issues/761#issuecomment-288818460
            int t = 0;
            auto I = py::vectorize([&map, &grad, &t](double theta, double x, double y) {
                // Evaluate the function
                double res = map.evaluate(theta, x, y, true);
                // Gather the derivatives
                for (int j = 0; j < map.dI.size(); j++)
                    grad[map.dI_names[j]](t) = map.dI(j);
                t++;
                return res;
            })(theta, x, y);

            // Return a tuple of (I, dict(dI))
            return py::make_tuple(I, grad);

            */

            return py::cast(0.0);

        } else {

            // Easy! We'll just return I
            return py::vectorize([&map](double theta, double x, double y) {
                return static_cast<double>(map.evaluate(theta, x, y, false));
            })(theta, x, y);

        }

    }

    template <>
    py::object evaluate(maps::Map<Matrix<double>, Vector<double>, VectorT<double>> &map,
                        py::object& theta,
                        py::object& x,
                        py::object& y,
                        bool gradient){

        if (gradient) {

            /* TODO */
            return py::cast(0.0);

        } else {

            // Vectorize the arguments manually
            Vector<double> theta_v, x_v, y_v;
            vectorize_args(theta, x, y,
                           theta_v, x_v, y_v);

            // Iterate through the timeseries
            size_t sz = theta_v.size();
            Matrix<double> I(sz, map.nwav);
            for (size_t i = 0; i < sz; ++i) {
                I.row(i) = map.evaluate(theta_v(i), x_v(i), y_v(i), false);
            }

            // Cast to python object
            return py::cast(I);

        }

    }

}; // namespace vectorize

#endif
