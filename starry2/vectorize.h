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

namespace vectorize {

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

}; // namespace vectorize

#endif
