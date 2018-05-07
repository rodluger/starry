/**
Home-built vectorization wrappers to replace py::vectorize when using autodiff.

These are not particularly elegant, and there's a lot of code duplication below.
If anyone would like to try templating some of this stuff to make it more efficient,
please go for it!

*/

#ifndef _STARRY_VECT_H_
#define _STARRY_VECT_H_

#include <iostream>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <stdlib.h>
#include <utils.h>
#include <maps.h>

using namespace std;
namespace py = pybind11;

namespace vect {

    inline Vector<double> vectorize_arg(py::object& obj, int& size){
        Vector<double> res;
        if (py::isinstance<py::float_>(obj) || py::isinstance<py::int_>(obj)) {
            res = Vector<double>::Constant(size, py::cast<double>(obj));
            return res;
        } else if (py::isinstance<py::array>(obj) || py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
            res = py::cast<Vector<double>>(obj);
            if ((size == 0) || (res.size() == size)) {
                size = res.size();
                return res;
            } else {
                throw invalid_argument("Mismatch in argument dimensions.");
            }
        } else {
            throw invalid_argument("Incorrect type for one or more of the arguments.");
        }
    }

    inline void vectorize_args(py::object& arg1, py::object& arg2, py::object& arg3, Vector<double>& arg1_v, Vector<double>& arg2_v, Vector<double>& arg3_v) {
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

    inline void vectorize_args(py::object& arg1, py::object& arg2, py::object& arg3, py::object& arg4, Vector<double>& arg1_v, Vector<double>& arg2_v, Vector<double>& arg3_v, Vector<double>& arg4_v) {
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

    /**
    Vectorize a class member function for class type double.
    Based on https://stackoverflow.com/a/12662950. This function
    takes 5 arguments, the first of which is a constant unit vector.
    The remaining arguments are vectorized.

    This is used specifically to vectorize `starry.Map.flux()`.
    */
    inline Vector<double> vectorize(
        UnitVector<double>& arg1, py::object& arg2, py::object& arg3, py::object& arg4, py::object& arg5,
        double (maps::Map<double>::*func)(const UnitVector<double>&, const double&, const double&, const double&, const double&),
        maps::Map<double>& map
    ) {
        // Vectorize the inputs
        Vector<double> arg2_v, arg3_v, arg4_v, arg5_v;
        vectorize_args(arg2, arg3, arg4, arg5, arg2_v, arg3_v, arg4_v, arg5_v);

        // Compute the function for each vector index
        Vector<double> result(arg2_v.size());
        for (int i = 0; i < arg2_v.size(); i++)
            result(i) = (map.*func)(arg1, arg2_v(i), arg3_v(i), arg4_v(i), arg5_v(i));

        // Return an array
        return result;
    }

    /**
    Overloaded definition of `vectorize()` for function taking 4 arguments, the
    first of which is a constant unit vector.

    This is used specifically to vectorize `starry.Map.evaluate()`.
    */
    inline Vector<double> vectorize(
        UnitVector<double>& arg1, py::object& arg2, py::object& arg3, py::object& arg4,
        double (maps::Map<double>::*func)(const UnitVector<double>&, const double&, const double&, const double&),
        maps::Map<double>& map
    ) {
        Vector<double> arg2_v, arg3_v, arg4_v;
        vectorize_args(arg2, arg3, arg4, arg2_v, arg3_v, arg4_v);
        Vector<double> result(arg2_v.size());
        for (int i = 0; i < arg2_v.size(); i++)
            result(i) = (map.*func)(arg1, arg2_v(i), arg3_v(i), arg4_v(i));
        return result;
    }

    /**
    Overloaded definition of `vectorize()` for function taking 6 arguments, the
    first of which is a constant unit vector, and the last of which is a constant double.

    This is used specifically to vectorize `starry.Map.flux_numerical()`.
    */
    inline Vector<double> vectorize(
        UnitVector<double>& arg1, py::object& arg2, py::object& arg3, py::object& arg4, py::object& arg5, double& arg6,
        double (maps::Map<double>::*func)(const UnitVector<double>&, const double&, const double&, const double&, const double&, double),
        maps::Map<double>& map
    ) {
        Vector<double> arg2_v, arg3_v, arg4_v, arg5_v;
        vectorize_args(arg2, arg3, arg4, arg5, arg2_v, arg3_v, arg4_v, arg5_v);
        Vector<double> result(arg2_v.size());
        for (int i = 0; i < arg2_v.size(); i++)
            result(i) = (map.*func)(arg1, arg2_v(i), arg3_v(i), arg4_v(i), arg5_v(i), arg6);
        return result;
    }

    /**
    Vectorize a class member function for class type Grad.
    Based on https://stackoverflow.com/a/12662950. This function
    takes 5 arguments, the first of which is a constant unit vector.
    The remaining arguments are vectorized.

    This is used specifically to vectorize `starry.grad.Map.flux()`.
    */
    inline Matrix<double> vectorize(
        UnitVector<double>& arg1, py::object& arg2, py::object& arg3, py::object& arg4, py::object& arg5,
        Grad (maps::Map<Grad>::*func)(const UnitVector<Grad>&, const Grad&, const Grad&, const Grad&, const Grad&),
        maps::Map<Grad>& map
    ) {

        // Vectorize only the inputs of type double
        int nder = 7 + map.N * (int)map.map_gradients;
        Vector<double> arg2_v, arg3_v, arg4_v, arg5_v;
        vectorize_args(arg2, arg3, arg4, arg5, arg2_v, arg3_v, arg4_v, arg5_v);

        // Declare our gradient types
        Grad arg1_x(arg1(0), nder, 0);
        Grad arg1_y(arg1(1), nder, 1);
        Grad arg1_z(arg1(2), nder, 2);
        Grad arg2_g(0., nder, 3);
        Grad arg3_g(0., nder, 4);
        Grad arg4_g(0., nder, 5);
        Grad arg5_g(0., nder, 6);
        UnitVector<Grad> arg1_g({arg1_x, arg1_y, arg1_z});
        Grad tmp;

        // Compute gradients w/ respect to the map coefficients?
        if (map.map_gradients) {
            for (int n = 0; n < map.N; n++)
                map.y(n).derivatives() = Vector<double>::Unit(nder, 7 + n);
        }

        // Compute the function at each index
        Matrix<double> result(arg2_v.size(), nder + 1);
        for (int i = 0; i < arg2_v.size(); i++) {
            arg2_g.value() = arg2_v(i);
            arg3_g.value() = arg3_v(i);
            arg4_g.value() = arg4_v(i);
            arg5_g.value() = arg5_v(i);
            tmp = (map.*func)(arg1_g, arg2_g, arg3_g, arg4_g, arg5_g);
            result(i, 0) = tmp.value();
            for (int j = 0; j < nder; j++)
                result(i, j + 1) = tmp.derivatives()(j);
        }

        // Return a matrix of function value, derivatives at each index
        return result;

    }

    /**
    Overloaded definition of `vectorize()` for function taking 4 arguments, the
    first of which is a constant unit vector.

    This is used specifically to vectorize `starry.grad.Map.evaluate()`.
    */
    inline Matrix<double> vectorize(
        UnitVector<double>& arg1, py::object& arg2, py::object& arg3, py::object& arg4,
        Grad (maps::Map<Grad>::*func)(const UnitVector<Grad>&, const Grad&, const Grad&, const Grad&),
        maps::Map<Grad>& map
    ) {

            int nder = 6 + map.N * (int)map.map_gradients;
            Vector<double> arg2_v, arg3_v, arg4_v;
            vectorize_args(arg2, arg3, arg4, arg2_v, arg3_v, arg4_v);
            Grad arg1_x(arg1(0), nder, 0);
            Grad arg1_y(arg1(1), nder, 1);
            Grad arg1_z(arg1(2), nder, 2);
            Grad arg2_g(0., nder, 3);
            Grad arg3_g(0., nder, 4);
            Grad arg4_g(0., nder, 5);
            UnitVector<Grad> arg1_g({arg1_x, arg1_y, arg1_z});
            Grad tmp;
            if (map.map_gradients) {
                for (int n = 0; n < map.N; n++)
                    map.y(n).derivatives() = Vector<double>::Unit(nder, 6 + n);
            }
            Matrix<double> result(arg2_v.size(), nder + 1);
            for (int i = 0; i < arg2_v.size(); i++) {
                arg2_g.value() = arg2_v(i);
                arg3_g.value() = arg3_v(i);
                arg4_g.value() = arg4_v(i);
                tmp = (map.*func)(arg1_g, arg2_g, arg3_g, arg4_g);
                result(i, 0) = tmp.value();
                for (int j = 0; j < nder; j++)
                    result(i, j + 1) = tmp.derivatives()(j);
            }
            return result;

    }

} // namespace vect

#endif
