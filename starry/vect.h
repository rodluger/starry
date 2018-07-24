/**
Home-built vectorization wrappers to replace py::vectorize when using autodiff.

These are not particularly elegant, and there's a lot of code duplication below.
If anyone would like to try templating some of this stuff to make it more efficient,
please go for it!

NOTE: I'm converting the angles from degrees to radians here before they get passed
      to `Map`. This is not the best way to be doing this...

*/

#ifndef _STARRY_VECT_H_
#define _STARRY_VECT_H_

#include <iostream>
#include <cmath>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <stdlib.h>
#include "utils.h"
#include "maps.h"
#include "constants.h"
#include "errors.h"
namespace py = pybind11;

namespace vect {

    using namespace std;

    // Vectorize a single python object
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

    // Vectorize function of two args
    inline void vectorize_args(py::object& arg1, py::object& arg2, Vector<double>& arg1_v, Vector<double>& arg2_v) {
        int size = 0;
        if (py::hasattr(arg1, "__len__")) {
            // arg1 is a vector
            arg1_v = vectorize_arg(arg1, size);
            arg2_v = vectorize_arg(arg2, size);
        } else if (py::hasattr(arg2, "__len__")) {
            // arg2 is a vector
            arg2_v = vectorize_arg(arg2, size);
            arg1_v = vectorize_arg(arg1, size);
        } else {
            // no arg is a vector
            size = 1;
            arg1_v = vectorize_arg(arg1, size);
            arg2_v = vectorize_arg(arg2, size);
        }
    }

    // Vectorize function of three args
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

    // Vectorize function of four args
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

    /* --------------------------------------------- */

    // Instantiate a Grad type with or without derivatives
    Grad new_grad(const string& name, const double& value, const vector<string>& gradients, int& n, const int& ngrad) {
        if(find(gradients.begin(), gradients.end(), name) != gradients.end()) {
            return Grad(value, ngrad, n++);
        } else {
            return Grad(value);
        }
    }

    /* --------------------------------------------- */

    // Vectorize `starry.Map.flux()`.
    inline Vector<double> vectorize_map_flux(
            py::object& theta, py::object& xo, py::object& yo, py::object& ro,
            maps::Map<double>& map) {

        // Vectorize the inputs
        Vector<double> theta_v, xo_v, yo_v, ro_v;
        vectorize_args(theta, xo, yo, ro, theta_v, xo_v, yo_v, ro_v);

        // Compute the function for each vector index
        Vector<double> result(theta_v.size());
        for (int i = 0; i < theta_v.size(); i++)
            result(i) = map.flux(theta_v(i) * DEGREE, xo_v(i), yo_v(i), ro_v(i));

        // Return an array
        return result;
    }

    // Vectorize `starry.Map.flux_numerical()`.
    inline Vector<double> vectorize_map_flux_numerical(
            py::object& theta, py::object& xo, py::object& yo, py::object& ro, double& tol,
            maps::Map<double>& map) {

        // Vectorize the inputs
        Vector<double> theta_v, xo_v, yo_v, ro_v;
        vectorize_args(theta, xo, yo, ro, theta_v, xo_v, yo_v, ro_v);

        // Compute the function for each vector index
        Vector<double> result(theta_v.size());
        for (int i = 0; i < theta_v.size(); i++)
            result(i) = map.flux_numerical(theta_v(i) * DEGREE, xo_v(i), yo_v(i), ro_v(i), tol);

        // Return an array
        return result;
    }

    // Vectorize `starry.Map.evaluate()`.
    inline Vector<double> vectorize_map_evaluate(
            py::object& theta, py::object& x, py::object& y,
            maps::Map<double>& map) {

        // Vectorize the inputs
        Vector<double> theta_v, x_v, y_v;
        vectorize_args(theta, x, y, theta_v, x_v, y_v);

        // Compute the function for each vector index
        Vector<double> result(theta_v.size());
        for (int i = 0; i < theta_v.size(); i++)
            result(i) = map.evaluate(theta_v(i) * DEGREE, x_v(i), y_v(i));

        // Return an array
        return result;
    }

    // Vectorize `starry.LimbDarkenedMap.flux()`.
    inline Vector<double> vectorize_ldmap_flux(
            py::object& xo, py::object& yo, py::object& ro,
            maps::LimbDarkenedMap<double>& map) {

        // Vectorize the inputs
        Vector<double> xo_v, yo_v, ro_v;
        vectorize_args(xo, yo, ro, xo_v, yo_v, ro_v);

        // Compute the function for each vector index
        Vector<double> result(xo_v.size());
        for (int i = 0; i < xo_v.size(); i++)
            result(i) = map.flux(xo_v(i), yo_v(i), ro_v(i));

        // Return an array
        return result;
    }

    // Vectorize `starry.LimbDarkenedMap.flux_numerical()`.
    inline Vector<double> vectorize_ldmap_flux_numerical(
            py::object& xo, py::object& yo, py::object& ro, double& tol,
            maps::LimbDarkenedMap<double>& map) {

        // Vectorize the inputs
        Vector<double> xo_v, yo_v, ro_v;
        vectorize_args(xo, yo, ro, xo_v, yo_v, ro_v);

        // Compute the function for each vector index
        Vector<double> result(xo_v.size());
        for (int i = 0; i < xo_v.size(); i++)
            result(i) = map.flux_numerical(xo_v(i), yo_v(i), ro_v(i), tol);

        // Return an array
        return result;
    }

    // Vectorize `starry.LimbDarkenedMap.evaluate()`.
    inline Vector<double> vectorize_ldmap_evaluate(
            py::object& x, py::object& y,
            maps::LimbDarkenedMap<double>& map) {

        // Vectorize the inputs
        Vector<double> x_v, y_v;
        vectorize_args(x, y, x_v, y_v);

        // Compute the function for each vector index
        Vector<double> result(x_v.size());
        for (int i = 0; i < x_v.size(); i++)
            result(i) = map.evaluate(x_v(i), y_v(i));

        // Return an array
        return result;
    }

    /* --------------------------------------------- */

    // Vectorize `starry.multi.Map.flux()`.
    inline Vector<double> vectorize_map_flux(
            py::object& theta, py::object& xo, py::object& yo, py::object& ro,
            maps::Map<Multi>& map) {

        // Vectorize the inputs
        Vector<double> theta_v, xo_v, yo_v, ro_v;
        vectorize_args(theta, xo, yo, ro, theta_v, xo_v, yo_v, ro_v);

        // Compute the function for each vector index
        Vector<double> result(theta_v.size());
        for (int i = 0; i < theta_v.size(); i++)
            result(i) = (double)map.flux(theta_v(i) * DEGREE, xo_v(i), yo_v(i), ro_v(i));

        // Return an array
        return result;
    }

    // Vectorize `starry.multi.Map.evaluate()`.
    inline Vector<double> vectorize_map_evaluate(
            py::object& theta, py::object& x, py::object& y,
            maps::Map<Multi>& map) {

        // Vectorize the inputs
        Vector<double> theta_v, x_v, y_v;
        vectorize_args(theta, x, y, theta_v, x_v, y_v);

        // Compute the function for each vector index
        Vector<double> result(theta_v.size());
        for (int i = 0; i < theta_v.size(); i++)
            result(i) = (double)map.evaluate(theta_v(i) * DEGREE, x_v(i), y_v(i));

        // Return an array
        return result;
    }

    // Vectorize `starry.multi.LimbDarkenedMap.flux()`.
    inline Vector<double> vectorize_ldmap_flux(
            py::object& xo, py::object& yo, py::object& ro,
            maps::LimbDarkenedMap<Multi>& map) {

        // Vectorize the inputs
        Vector<double> xo_v, yo_v, ro_v;
        vectorize_args(xo, yo, ro, xo_v, yo_v, ro_v);

        // Compute the function for each vector index
        Vector<double> result(xo_v.size());
        for (int i = 0; i < xo_v.size(); i++)
            result(i) = (double)map.flux(xo_v(i), yo_v(i), ro_v(i));

        // Return an array
        return result;
    }

    // Vectorize `starry.multi.LimbDarkenedMap.evaluate()`.
    inline Vector<double> vectorize_ldmap_evaluate(
            py::object& x, py::object& y,
            maps::LimbDarkenedMap<Multi>& map) {

        // Vectorize the inputs
        Vector<double> x_v, y_v;
        vectorize_args(x, y, x_v, y_v);

        // Compute the function for each vector index
        Vector<double> result(x_v.size());
        for (int i = 0; i < x_v.size(); i++)
            result(i) = (double)map.evaluate(x_v(i), y_v(i));

        // Return an array
        return result;
    }

    /* --------------------------------------------- */

    // Vectorize `starry.grad.Map.flux()`.
    inline Vector<double> vectorize_map_flux(
            py::object& theta, py::object& xo, py::object& yo, py::object& ro,
            maps::Map<Grad>& map) {

        int l, m, n, i, j;

        // Check that our derivative vectors are large enough
        int ngrad = 4;
        if (ngrad > STARRY_NGRAD) throw errors::TooManyDerivs(ngrad);

        // Vectorize only the inputs of type double
        Vector<double> theta_v, xo_v, yo_v, ro_v;
        vectorize_args(theta, xo, yo, ro, theta_v, xo_v, yo_v, ro_v);

        // Declare our gradient types
        vector<string> names {"theta", "xo", "yo", "ro"};
        Grad theta_g(0., STARRY_NGRAD, 0);
        Grad xo_g(0., STARRY_NGRAD, 1);
        Grad yo_g(0., STARRY_NGRAD, 2);
        Grad ro_g(0., STARRY_NGRAD, 3);
        Grad tmp;
        Vector<double> result(theta_v.size());

        // Allocate memory for the derivs
        map.derivs.clear();
        for (n = 0; n < ngrad; n++) {
            map.derivs[names[n]].resize(theta_v.size());
        }

        // Treat the map derivs separately (we compute them manually)
        for (l = 0; l < map.lmax + 1; l++) {
            for (m = -l; m < l + 1; m++) {
                names.push_back(string("Y_{" + to_string(l) + "," + to_string(m) + "}"));
                map.derivs[names[n]].resize(theta_v.size());
                n++;
            }
        }

        // Populate the result vector and the gradients
        for (i = 0; i < theta_v.size(); i++) {
            theta_g.value() = theta_v(i);
            xo_g.value() = xo_v(i);
            yo_g.value() = yo_v(i);
            ro_g.value() = ro_v(i);
            tmp = map.flux(theta_g * DEGREE, xo_g, yo_g, ro_g);
            result(i) = tmp.value();
            for (n = 0; n < ngrad; n++) {
                (map.derivs[names[n]])(i) = tmp.derivatives()(n);
            }
            for (j = 0; j < map.N; j++) {
                (map.derivs[names[n + j]])(i) = map.dFdy(j).value();
            }
        }

        // Return an array
        return result;

    }

    // Vectorize `starry.grad.Map.evaluate()`.
    inline Vector<double> vectorize_map_evaluate(
            py::object& theta, py::object& x, py::object& y,
            maps::Map<Grad>& map) {

        int l, m, n, i, j;

        // Check that our derivative vectors are large enough
        int ngrad = 3;
        if (ngrad > STARRY_NGRAD) throw errors::TooManyDerivs(ngrad);

        // Vectorize only the inputs of type double
        Vector<double> theta_v, x_v, y_v;
        vectorize_args(theta, x, y, theta_v, x_v, y_v);

        // Declare our gradient types
        vector<string> names {"theta", "x", "y"};
        Grad theta_g(0., STARRY_NGRAD, 0);
        Grad x_g(0., STARRY_NGRAD, 1);
        Grad y_g(0., STARRY_NGRAD, 2);
        Grad tmp;
        Vector<double> result(theta_v.size());

        // Allocate memory for the derivs
        map.derivs.clear();
        for (n = 0; n < ngrad; n++) {
            map.derivs[names[n]].resize(theta_v.size());
        }

        // Treat the map derivs separately (we compute them manually)
        for (l = 0; l < map.lmax + 1; l++) {
            for (m = -l; m < l + 1; m++) {
                names.push_back(string("Y_{" + to_string(l) + "," + to_string(m) + "}"));
                map.derivs[names[n]].resize(theta_v.size());
                n++;
            }
        }

        // Populate the result vector and the gradients
        for (i = 0; i < theta_v.size(); i++) {
            theta_g.value() = theta_v(i);
            x_g.value() = x_v(i);
            y_g.value() = y_v(i);
            tmp = map.evaluate(theta_g * DEGREE, x_g, y_g);
            result(i) = tmp.value();
            for (n = 0; n < ngrad; n++) {
                (map.derivs[names[n]])(i) = tmp.derivatives()(n);
            }
            for (j = 0; j < map.N; j++) {
                (map.derivs[names[n + j]])(i) = map.dFdy(j).value();
            }
        }

        // Return an array
        return result;

    }

    // Vectorize `starry.grad.LimbDarkenedMap.flux()`.
    inline Vector<double> vectorize_ldmap_flux(
            py::object& xo, py::object& yo, py::object& ro,
            maps::LimbDarkenedMap<Grad>& map) {

        int l, n, i;

        // Check that our derivative vectors are large enough
        int ngrad = 3;
        if (ngrad > STARRY_NGRAD) throw errors::TooManyDerivs(ngrad);

        // Vectorize only the inputs of type double
        Vector<double> xo_v, yo_v, ro_v;
        vectorize_args(xo, yo, ro, xo_v, yo_v, ro_v);

        // Declare our gradient types
        vector<string> names {"xo", "yo", "ro"};
        Grad xo_g(0., STARRY_NGRAD, 0);
        Grad yo_g(0., STARRY_NGRAD, 1);
        Grad ro_g(0., STARRY_NGRAD, 2);
        Grad tmp;
        Vector<double> result(xo_v.size());

        // Allocate memory for the derivs
        map.derivs.clear();
        for (n = 0; n < ngrad; n++) {
            map.derivs[names[n]].resize(xo_v.size());
        }

        // Treat the map derivs separately
        for (l = 1; l < map.lmax + 1; l++) {
            names.push_back(string("u_" + to_string(l)));
            map.derivs[names[n]].resize(xo_v.size());
            n++;
        }

        // Populate the result vector and the gradients
        for (i = 0; i < xo_v.size(); i++) {
            xo_g.value() = xo_v(i);
            yo_g.value() = yo_v(i);
            ro_g.value() = ro_v(i);
            tmp = map.flux(xo_g, yo_g, ro_g);
            result(i) = tmp.value();
            for (n = 0; n < ngrad; n++) {
                (map.derivs[names[n]])(i) = tmp.derivatives()(n);
            }
            for (l = 1; l < map.lmax + 1; l++) {
                (map.derivs[names[n + l - 1]])(i) = map.dFdu(l).value();
            }
        }

        // Return an array
        return result;

    }

    // Vectorize `starry.grad.LimbDarkenedMap.evaluate()`.
    inline Vector<double> vectorize_ldmap_evaluate(
            py::object& x, py::object& y,
            maps::LimbDarkenedMap<Grad>& map) {

        int l, n, i;

        // Check that our derivative vectors are large enough
        int ngrad = 2;
        if (ngrad > STARRY_NGRAD) throw errors::TooManyDerivs(ngrad);

        // Vectorize only the inputs of type double
        Vector<double> x_v, y_v;
        vectorize_args(x, y, x_v, y_v);

        // Declare our gradient types
        vector<string> names {"x", "y"};
        Grad x_g(0., STARRY_NGRAD, 0);
        Grad y_g(0., STARRY_NGRAD, 1);
        Grad tmp;
        Vector<double> result(x_v.size());

        // Allocate memory for the derivs
        map.derivs.clear();
        for (n = 0; n < ngrad; n++) {
            map.derivs[names[n]].resize(x_v.size());
        }

        // Treat the map derivs separately
        for (l = 1; l < map.lmax + 1; l++) {
            names.push_back(string("u_" + to_string(l)));
            map.derivs[names[n]].resize(x_v.size());
            n++;
        }

        // Populate the result vector and the gradients
        for (i = 0; i < x_v.size(); i++) {
            x_g.value() = x_v(i);
            y_g.value() = y_v(i);
            tmp = map.evaluate(x_g, y_g);
            result(i) = tmp.value();
            for (n = 0; n < ngrad; n++) {
                (map.derivs[names[n]])(i) = tmp.derivatives()(n);
            }
            for (l = 1; l < map.lmax + 1; l++) {
                (map.derivs[names[n + l - 1]])(i) = map.dFdu(l).value();
            }
        }

        // Return an array
        return result;

    }

} // namespace vect

#endif
