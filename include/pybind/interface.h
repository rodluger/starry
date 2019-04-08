/**
\file interface.h
\brief Miscellaneous utilities used for the `pybind` interface.

*/

#ifndef _STARRY_PYBIND_UTILS_H_
#define _STARRY_PYBIND_UTILS_H_

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <stdlib.h>
#include <vector>
#include <starry/utils.h>
#include <starry/maps.h>

#ifdef _STARRY_MULTI_
#   define ENSURE_DOUBLE(X)               static_cast<double>(X)
#   define ENSURE_DOUBLE_ARR(X)           X.template cast<double>()
#   define PYOBJECT_CAST(X)               py::cast(static_cast<double>(X))
#   define PYOBJECT_CAST_ARR(X)           py::cast(X.template cast<double>())
#else
#   define ENSURE_DOUBLE(X)               X
#   define ENSURE_DOUBLE_ARR(X)           X
#   define PYOBJECT_CAST(X)               py::cast(X)
#   define PYOBJECT_CAST_ARR(X)           py::cast(&X)
#endif

#define PY_ARRAY py::array_t<double, py::array::c_style | py::array::forcecast>

#define MAKE_READ_ONLY(X) \
    reinterpret_cast<py::detail::PyArray_Proxy*>(X.ptr())->flags &= \
        ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;

#ifdef _STARRY_DOUBLE_
#   define MAP_TO_EIGEN_VECTOR(PYX, X, T, N) \
        Eigen::Map<Vector<T>> X(NULL, N, 1); \
        Vector<T> tmp##X; \
        auto buf##X = PYX.request(); \
        double* ptr##X = (double *) buf##X.ptr; \
        if (buf##X.ndim == 0) { \
            tmp##X = ptr##X[0] * Vector<T>::Ones(N); \
            new (&X) Eigen::Map<Vector<T>>(&tmp##X(0), N, 1); \
        } else if ((buf##X.ndim == 1) && (buf##X.size == N)) { \
            new (&X) Eigen::Map<Vector<T>>(ptr##X, N, 1); \
        } else { \
            throw std::length_error("Argument `" #X "` has the wrong shape."); \
        }
#   define MAP_TO_EIGEN_ROW_MATRIX(PYX, X, T) \
        py::buffer_info buf##X = PYX.request(); \
        double *ptr##X = (double *) buf##X.ptr; \
        Eigen::Map<RowMatrix<T>> X(NULL, \
                                   buf##X.ndim > 0 ? buf##X.shape[0] : 1, \
                                   buf##X.ndim > 1 ? buf##X.shape[1] : 1); \
        RowMatrix<T> tmp##X; \
        if (buf##X.ndim == 0) { \
            tmp##X = ptr##X[0] * RowMatrix<T>::Ones(1, 1); \
            new (&X) Eigen::Map<RowMatrix<T>>(&tmp##X(0), 1, 1); \
        } else if (buf##X.ndim == 1) { \
            new (&X) Eigen::Map<RowMatrix<T>>(ptr##X, buf##X.shape[0], 1); \
        } else if (buf##X.ndim == 2) { \
            new (&X) Eigen::Map<RowMatrix<T>>(ptr##X, buf##X.shape[0], buf##X.shape[1]); \
        } else { \
            throw std::invalid_argument("Argument `" #X "` has the wrong shape."); \
        }
#   define MAP_TO_EIGEN_ROW_MATRIX_OF_UNIT_VECTORS(PYX, X, T, N) \
        Eigen::Map<RowMatrix<T>> X(NULL, N, 3); \
        RowMatrix<T> tmp##X; \
        auto buf##X = PYX.request(); \
        double* ptr##X = (double *) buf##X.ptr; \
        if ((buf##X.ndim == 1) && (buf##X.size == 3)) { \
            tmp##X = (Eigen::Map<RowMatrix<T>>(ptr##X, 1, 3)).replicate(nt, 1);\
            new (&X) Eigen::Map<RowMatrix<T>>(&tmp##X(0), N, 3); \
        } else if ((buf##X.ndim == 2) && (buf##X.size == 3 * N)) { \
            new (&X) Eigen::Map<RowMatrix<T>>(ptr##X, N, 3); \
        } else { \
            throw std::length_error("Argument `" #X "` has the wrong shape."); \
        }
#else
#   define MAP_TO_EIGEN_VECTOR(PYX, X, T, N) \
        Eigen::Map<Vector<T>> X(NULL, N, 1); \
        Vector<T> tmp##X; \
        auto buf##X = PYX.request(); \
        double* ptr##X = (double *) buf##X.ptr; \
        if (buf##X.ndim == 0) { \
            tmp##X = ptr##X[0] * Vector<T>::Ones(N); \
            new (&X) Eigen::Map<Vector<T>>(&tmp##X(0), N, 1); \
        } else if ((buf##X.ndim == 1) && (buf##X.size == N)) { \
            tmp##X = (py::cast<Vector<double>>(PYX)).template cast<T>(); \
            new (&X) Eigen::Map<Vector<T>>(&tmp##X(0), N, 1); \
        } else { \
            throw std::length_error("Argument `" #X "` has the wrong shape."); \
        }
#   define MAP_TO_EIGEN_ROW_MATRIX(PYX, X, T) \
        py::buffer_info buf##X = PYX.request(); \
        double *ptr##X = (double *) buf##X.ptr; \
        Eigen::Map<RowMatrix<T>> X(NULL, \
                                   buf##X.ndim > 0 ? buf##X.shape[0] : 1, \
                                   buf##X.ndim > 1 ? buf##X.shape[1] : 1); \
        RowMatrix<T> tmp##X; \
        if (buf##X.ndim == 0) { \
            tmp##X = ptr##X[0] * RowMatrix<T>::Ones(1, 1); \
            new (&X) Eigen::Map<RowMatrix<T>>(&tmp##X(0), 1, 1); \
        } else if (buf##X.ndim == 1) { \
            tmp##X = (py::cast<RowMatrix<double>>(PYX)).template cast<T>(); \
            new (&X) Eigen::Map<RowMatrix<T>>(&tmp##X(0), buf##X.shape[0], 1); \
        } else if (buf##X.ndim == 2) { \
            tmp##X = (py::cast<RowMatrix<double>>(PYX)).template cast<T>(); \
            new (&X) Eigen::Map<RowMatrix<T>>(&tmp##X(0), buf##X.shape[0], buf##X.shape[1]); \
        } else { \
            throw std::invalid_argument("Argument `" #X "` has the wrong shape."); \
        }
#   define MAP_TO_EIGEN_ROW_MATRIX_OF_UNIT_VECTORS(PYX, X, T, N) \
        Eigen::Map<RowMatrix<T>> X(NULL, N, 3); \
        RowMatrix<T> tmp##X; \
        auto buf##X = PYX.request(); \
        if ((buf##X.ndim == 1) && (buf##X.size == 3)) { \
            tmp##X = ((py::cast<RowMatrix<double>>(PYX)).template cast<T>()).replicate(nt, 1);\
            new (&X) Eigen::Map<RowMatrix<T>>(&tmp##X(0), N, 3); \
        } else if ((buf##X.ndim == 2) && (buf##X.size == 3 * N)) { \
            tmp##X = (py::cast<RowMatrix<double>>(PYX)).template cast<T>();\
            new (&X) Eigen::Map<RowMatrix<T>>(&tmp##X(0), N, 3); \
        } else { \
            throw std::length_error("Argument `" #X "` has the wrong shape."); \
        }
#endif

namespace interface {

//! Misc stuff we need
#include <Python.h>
using namespace pybind11::literals;
using namespace starry;
using namespace starry::utils;
using starry::maps::Map;

/**
Re-interpret the `start`, `stop`, and `step` attributes of a `py::slice`,
allowing for *actual* negative indices. This allows the user to provide
something like `map[3, -3:0]` to get the `l = 3, m = {-3, -2, -1}` indices
of the spherical harmonic map. Pretty sneaky stuff.

*/
void reinterpret_slice (
    const py::slice& slice, 
    const int smin,
    const int smax, 
    int& start, 
    int& stop, 
    int& step
) {
    PySliceObject *r = (PySliceObject*)(slice.ptr());
    if (r->start == Py_None)
        start = smin;
    else
        start = PyLong_AsSsize_t(r->start);
    if (r->stop == Py_None)
        stop = smax;
    else
        stop = PyLong_AsSsize_t(r->stop) - 1;
    if ((r->step == Py_None) || (PyLong_AsSsize_t(r->step) == 1))
        step = 1;
    else
        throw std::invalid_argument("Slices with steps different from "
                                 "one are not supported.");
}

/**
Parse a user-provided `(l, m)` tuple into spherical harmonic map indices.

*/
std::vector<int> get_Ylm_inds (
    const int lmax, 
    const py::tuple& lm
) {
    auto integer = py::module::import("numpy").attr("integer");
    int N = (lmax + 1) * (lmax + 1);
    int n;
    if (lm.size() != 2)
        throw std::out_of_range("Invalid `l`, `m` tuple.");
    std::vector<int> inds;
    if ((py::isinstance<py::int_>(lm[0]) || py::isinstance(lm[0], integer)) && 
        (py::isinstance<py::int_>(lm[1]) || py::isinstance(lm[1], integer))) {
        // User provided `(l, m)`
        int l = py::cast<int>(lm[0]);
        int m = py::cast<int>(lm[1]);
        n = l * l + l + m;
        if ((n < 0) || (n >= N) || (m > l) || (m < -l))
            throw std::out_of_range("Invalid value for `l` and/or `m`.");
        inds.push_back(n);
        return inds;
    } else if ((py::isinstance<py::slice>(lm[0])) && 
               (py::isinstance<py::slice>(lm[1]))) {
        // User provided `(slice, slice)`
        auto lslice = py::cast<py::slice>(lm[0]);
        auto mslice = py::cast<py::slice>(lm[1]);
        int lstart, lstop, lstep;
        int mstart, mstop, mstep;
        reinterpret_slice(lslice, 0, lmax, lstart, lstop, lstep);
        if ((lstart < 0) || (lstart > lmax))
            throw std::out_of_range("Invalid value for `l`.");
        for (int l = lstart; l < lstop + 1; l += lstep) {
            reinterpret_slice(mslice, -l, l, mstart, mstop, mstep);
            if (mstart < -l)
                mstart = -l;
            if (mstop > l)
                mstop = l;
            for (int m = mstart; m < mstop + 1; m += mstep) {
                n = l * l + l + m;
                if ((n < 0) || (n >= N) || (m > l) || (m < -l))
                    throw std::out_of_range(
                        "Invalid value for `l` and/or `m`.");
                inds.push_back(n);
            }
        }
        return inds;
    } else if ((py::isinstance<py::int_>(lm[0]) || 
                py::isinstance(lm[0], integer)) && 
               (py::isinstance<py::slice>(lm[1]))) {
        // User provided `(l, slice)`
        int l = py::cast<int>(lm[0]);
        auto mslice = py::cast<py::slice>(lm[1]);
        int mstart, mstop, mstep;
        reinterpret_slice(mslice, -l, l, mstart, mstop, mstep);
        if (mstart < -l)
            mstart = -l;
        if (mstop > l)
            mstop = l;
        for (int m = mstart; m < mstop + 1; m += mstep) {
            n = l * l + l + m;
            if ((n < 0) || (n >= N) || (m > l) || (m < -l))
                throw std::out_of_range("Invalid value for `l` and/or `m`.");
            inds.push_back(n);
        }
        return inds;
    } else if ((py::isinstance<py::slice>(lm[0])) && 
               (py::isinstance<py::int_>(lm[1]) || 
                py::isinstance(lm[1], integer))) {
        // User provided `(slice, m)`
        int m = py::cast<int>(lm[1]);
        auto lslice = py::cast<py::slice>(lm[0]);
        int lstart, lstop, lstep;
        reinterpret_slice(lslice, 0, lmax, lstart, lstop, lstep);
        if ((lstart < 0) || (lstart > lmax))
            throw std::out_of_range("Invalid value for `l`.");
        for (int l = lstart; l < lstop + 1; l += lstep) {
            if ((m < -l) || (m > l))
                continue;
            n = l * l + l + m;
            if ((n < 0) || (n >= N) || (m > l) || (m < -l))
                throw std::out_of_range("Invalid value for `l` and/or `m`.");
            inds.push_back(n);
        }
        return inds;
    } else {
        // User provided something silly
        throw std::out_of_range("Unsupported input type for `l` and/or `m`.");
    }
}

/**
Parse a user-provided `(l, m, t)` tuple into spherical harmonic map indices.

*/
std::tuple<std::vector<int>, int> get_Ylmt_inds (
    const int lmax, 
    const int Nt,
    const py::tuple& lmt
) {
    auto integer = py::module::import("numpy").attr("integer");
    int Ny = (lmax + 1) * (lmax + 1);
    if (lmt.size() == 3) {
        std::vector<int> inds0 = get_Ylm_inds(lmax, py::make_tuple(lmt[0], lmt[1]));
        std::vector<int> inds;
        if ((py::isinstance<py::int_>(lmt[2]) || py::isinstance(lmt[2], integer))) {
            // User provided an integer time
            int t = py::cast<int>(lmt[2]);
            if ((t < 0) || (t >= Nt))
                throw std::out_of_range("Invalid value for `t`.");
            for (int n: inds0)
                inds.push_back(n + t * Ny);
            return std::make_tuple(inds, 1);
        } else if (py::isinstance<py::slice>(lmt[2])) {
            // User provided a time slice
            py::slice slice = py::cast<py::slice>(lmt[2]);
            ssize_t start, stop, step, slicelength;
            if(!slice.compute(Nt,
                              reinterpret_cast<size_t*>(&start),
                              reinterpret_cast<size_t*>(&stop),
                              reinterpret_cast<size_t*>(&step),
                              reinterpret_cast<size_t*>(&slicelength)))
                throw pybind11::error_already_set();
            if ((start < 0) || (start >= Nt)) {
                throw std::out_of_range("Invalid value for `t`.");
            } else if (step < 0) {
                throw std::invalid_argument(
                    "Slices with negative steps are not supported.");
            }
            for (int n: inds0) {
                for (ssize_t t = start; t < stop; t += step) {
                    inds.push_back(n + t * Ny);
                }
            }
            int ncols = 0;
            for (ssize_t t = start; t < stop; t += step) ++ncols;
            return std::make_tuple(inds, ncols);
        } else {
            // User provided something silly
            throw std::out_of_range("Unsupported input type for `t`.");
        }
    } else {
        throw std::out_of_range("Invalid `l`, `m`, `t` tuple.");
    }
}

/**
Parse a user-provided `(l, m, w)` tuple into spherical harmonic map indices.

*/
std::tuple<std::vector<int>, std::vector<int>> get_Ylmw_inds (
    const int lmax, 
    const int Nw,
    const py::tuple& lmw
) {
    auto integer = py::module::import("numpy").attr("integer");
    if (lmw.size() == 3) {
        std::vector<int> rows = get_Ylm_inds(lmax, py::make_tuple(lmw[0], lmw[1]));
        std::vector<int> cols;
        if ((py::isinstance<py::int_>(lmw[2]) || py::isinstance(lmw[2], integer))) {
            // User provided an integer wavelength bin
            int w = py::cast<int>(lmw[2]);
            if ((w < 0) || (w >= Nw))
                throw std::out_of_range("Invalid value for `w`.");
            cols.push_back(w);
            return std::make_tuple(rows, cols);
        } else if (py::isinstance<py::slice>(lmw[2])) {
            // User provided a wavelength slice
            py::slice slice = py::cast<py::slice>(lmw[2]);
            ssize_t start, stop, step, slicelength;
            if(!slice.compute(Nw,
                              reinterpret_cast<size_t*>(&start),
                              reinterpret_cast<size_t*>(&stop),
                              reinterpret_cast<size_t*>(&step),
                              reinterpret_cast<size_t*>(&slicelength)))
                throw pybind11::error_already_set();
            if ((start < 0) || (start >= Nw)) {
                throw std::out_of_range("Invalid value for `w`.");
            } else if (step < 0) {
                throw std::invalid_argument(
                    "Slices with negative steps are not supported.");
            }
            for (ssize_t w = start; w < stop; w += step) {
                cols.push_back(w);
            }
            return std::make_tuple(rows, cols);
        } else {
            // User provided something silly
            throw std::out_of_range("Unsupported input type for `w`.");
        }
    } else {
        throw std::out_of_range("Invalid `l`, `m`, `w` tuple.");
    }
}

/**
Parse a user-provided `l` into limb darkening map indices.

*/
std::vector<int> get_Ul_inds (
    int lmax, 
    const py::object& l
) {
    auto integer = py::module::import("numpy").attr("integer");
    int n;
    std::vector<int> inds;
    if (py::isinstance<py::int_>(l) || py::isinstance(l, integer)) {
        n = py::cast<int>(l);
        if ((n < 0) || (n > lmax))
            throw std::out_of_range("Invalid value for `l`.");
        inds.push_back(n);
        return inds;
    } else if (py::isinstance<py::slice>(l)) {
        py::slice slice = py::cast<py::slice>(l);
        ssize_t start, stop, step, slicelength;
        if(!slice.compute(lmax + 1,
                          reinterpret_cast<size_t*>(&start),
                          reinterpret_cast<size_t*>(&stop),
                          reinterpret_cast<size_t*>(&step),
                          reinterpret_cast<size_t*>(&slicelength)))
            throw pybind11::error_already_set();
        if ((start < 0) || (start > lmax)) {
            throw std::out_of_range("Invalid value for `l`.");
        } else if (step < 0) {
            throw std::invalid_argument(
                "Slices with negative steps are not supported.");
        }
        std::vector<int> inds;
        for (ssize_t i = start; i < stop; i += step) {
            inds.push_back(i);
        }
        return inds;
    } else {
        // User provided something silly
        throw std::out_of_range("Unsupported input type for `l`.");
    }
}

/**
Parse a user-provided `(l, w)` tuple into limb darkening map indices.

*/
std::tuple<std::vector<int>, std::vector<int>> get_Ulw_inds (
    const int lmax, 
    const int Nw,
    const py::tuple& lw
) {
    auto integer = py::module::import("numpy").attr("integer");
    if (lw.size() == 2) {
        std::vector<int> rows = get_Ul_inds(lmax, lw[0]);
        std::vector<int> cols;
        if ((py::isinstance<py::int_>(lw[1]) || py::isinstance(lw[1], integer))) {
            // User provided an integer wavelength bin
            int w = py::cast<int>(lw[1]);
            if ((w < 0) || (w >= Nw))
                throw std::out_of_range("Invalid value for `w`.");
            cols.push_back(w);
            return std::make_tuple(rows, cols);
        } else if (py::isinstance<py::slice>(lw[1])) {
            // User provided a wavelength slice
            py::slice slice = py::cast<py::slice>(lw[1]);
            ssize_t start, stop, step, slicelength;
            if(!slice.compute(Nw,
                              reinterpret_cast<size_t*>(&start),
                              reinterpret_cast<size_t*>(&stop),
                              reinterpret_cast<size_t*>(&step),
                              reinterpret_cast<size_t*>(&slicelength)))
                throw pybind11::error_already_set();
            if ((start < 0) || (start >= Nw)) {
                throw std::out_of_range("Invalid value for `w`.");
            } else if (step < 0) {
                throw std::invalid_argument(
                    "Slices with negative steps are not supported.");
            }
            for (ssize_t w = start; w < stop; w += step) {
                cols.push_back(w);
            }
            return std::make_tuple(rows, cols);
        } else {
            // User provided something silly
            throw std::out_of_range("Unsupported input type for `w`.");
        }
    } else {
        throw std::out_of_range("Invalid `l`, `w` tuple.");
    }
}

/**
Set one or more spherical harmonic coefficients

*/
template <typename T>
void set_Ylm(
    Map<T>& map, 
    const py::tuple& lm,
    py::array_t<double>& coeff_
) {
    using Scalar = typename T::Scalar;
    // Figure out the indices we're setting
#   if defined(_STARRY_TEMPORAL_)
        std::vector<int> rows = 
            std::get<0>(get_Ylmt_inds(map.ydeg, map.Nt, lm));
        std::vector<int> cols(1, 0);
#   elif defined(_STARRY_SPECTRAL_)
        auto inds = get_Ylmw_inds(map.ydeg, map.Nw, lm);
        std::vector<int> rows = std::get<0>(inds);
        std::vector<int> cols = std::get<1>(inds);
#   else
        std::vector<int> rows = get_Ylm_inds(map.ydeg, lm);
        std::vector<int> cols(1, 0);
#   endif

    // Reshape coeff into (rows, cols)
    py::buffer_info buf = coeff_.request();
    double *ptr = (double *) buf.ptr;
    Matrix<Scalar> coeff(rows.size(), cols.size());
    if (buf.ndim == 0) {
        // Set an array of indices (or rows/columns) to the same value
        coeff.setConstant(ptr[0]);
    } else if (buf.ndim == 1) {
        if (cols.size() == 1) {
            // Set an array of indices to an array of values
            coeff = py::cast<Matrix<double>>(coeff_).template 
                        cast<Scalar>();
        } else {
            if (rows.size() == 1) {
                // Set a row to an array of values
                coeff = (py::cast<Matrix<double>>(coeff_).template 
                            cast<Scalar>()).transpose();
            } else {
                // ?
                throw std::length_error("Invalid coefficient "
                                        "array shape.");
            }
        }
    } else if (buf.ndim == 2) {
        // Set a matrix of (row, column) indices to a matrix of values
        coeff = py::cast<Matrix<double>>(coeff_).template 
                    cast<Scalar>();
    } else {
        // ?
        throw std::length_error("Invalid coefficient array shape.");
    }

#   if defined(_STARRY_TEMPORAL_)
        // Flatten the input array if needed
        Matrix<Scalar> tmpcoeff = coeff.transpose();
        coeff = Eigen::Map<Matrix<Scalar>>(tmpcoeff.data(), 
                                            coeff.size(), 1);
#   endif

    // Check shape
    if (!((size_t(coeff.rows()) == size_t(rows.size())) && 
            (size_t(coeff.cols()) == size_t(cols.size()))))
        throw std::length_error("Mismatch in index array and " 
                                "coefficient array sizes.");

    // Grab the map vector and update it term by term
    auto y = map.getY();
    int i = 0;
    for (int row : rows) {
        int j = 0;
        for (int col : cols) {
            y(row, col) = static_cast<Scalar>(coeff(i, j));
            ++j;
        }
        ++i;
    }
    map.setY(y);
}

/** 
Set one or more limb darkening coefficients

*/
template <typename T>
void set_Ul(
    Map<T>& map, 
    const py::object& l,
    py::array_t<double>& coeff_
) {
    using Scalar = typename T::Scalar;
    // Figure out the indices we're setting
#   if defined(_STARRY_SPECTRAL_) && defined(_STARRY_LD_)
        auto inds = get_Ulw_inds(map.udeg, map.Nw, l);
        std::vector<int> rows = std::get<0>(inds);
        std::vector<int> cols = std::get<1>(inds);
#   else
        std::vector<int> rows = get_Ul_inds(map.udeg, l);
        std::vector<int> cols(1, 0);
#   endif

    // Reshape coeff if necessary
    py::buffer_info buf = coeff_.request();
    double *ptr = (double *) buf.ptr;
    Matrix<Scalar> coeff(rows.size(), cols.size());
    if (buf.ndim == 0) {
        // Set an array of indices (or rows/columns) to the same value
        coeff.setConstant(ptr[0]);
    } else if (buf.ndim == 1) {
        if (cols.size() == 1) {
            // Set an array of indices to an array of values
            coeff = py::cast<Matrix<double>>(coeff_).template 
                        cast<Scalar>();
        } else {
            if (rows.size() == 1) {
                // Set a row to an array of values
                coeff = (py::cast<Matrix<double>>(coeff_).template 
                            cast<Scalar>()).transpose();
            } else {
                // ?
                throw std::length_error("Invalid coefficient "
                                        "array shape.");
            }
        }
    } else if (buf.ndim == 2) {
        // Set a matrix of (row, column) indices to a matrix of values
        coeff = py::cast<Matrix<double>>(coeff_).template 
                    cast<Scalar>();
    } else {
        // ?
        throw std::length_error("Invalid coefficient array shape.");
    }

    // Check shape
    if (!((size_t(coeff.rows()) == size_t(rows.size())) && 
            (size_t(coeff.cols()) == size_t(cols.size()))))
        throw std::length_error("Mismatch in index array and " 
                                "coefficient array sizes.");

    // Grab the map vector and update it term by term
    auto u = map.getU();
    int i = 0;
    for (int row : rows) {
        int j = 0;
        for (int col : cols) {
            u(row, col) = static_cast<Scalar>(coeff(i, j));
            ++j;
        }
        ++i;
    }
    map.setU(u);
}

/** 
Retrieve one or more spherical harmonic coefficients

*/
template <typename T>
py::object get_Ylm (
    Map<T>& map,
    const py::tuple& lm
) {
    // Figure out the indices we're accessing
#   if defined(_STARRY_TEMPORAL_)
        auto rows_ncols = get_Ylmt_inds(map.ydeg, map.Nt, lm);
        std::vector<int> rows = std::get<0>(rows_ncols);
        int ncols = std::get<1>(rows_ncols);
        std::vector<int> cols(1, 0);
        Matrix<double> coeff_(rows.size(), cols.size());
#   elif defined(_STARRY_SPECTRAL_)
        auto inds = get_Ylmw_inds(map.ydeg, map.Nw, lm);
        std::vector<int> rows = std::get<0>(inds);
        std::vector<int> cols = std::get<1>(inds);
        Matrix<double> coeff_(rows.size(), cols.size());
#   else
        std::vector<int> rows = get_Ylm_inds(map.ydeg, lm);
        std::vector<int> cols(1, 0);
        Vector<double> coeff_(rows.size());
#   endif

    // Grab the map vector and update the output vector term by term
    auto y = map.getY();
    int i = 0;
    for (int row : rows) {
        int j = 0;
        for (int col : cols) {
            coeff_(i, j) = static_cast<double>(y(row, col));
            ++j;
        }
        ++i;
    }

#   if defined(_STARRY_TEMPORAL_)
        // Reshape the coefficients into a matrix
        Matrix<double> tmpcoeff = coeff_;
        coeff_ = Eigen::Map<Matrix<double>>(tmpcoeff.data(), 
                    ncols, coeff_.size() / ncols).transpose();
#   endif

    // Squeeze the output and cast to a py::array
    if (coeff_.size() == 1) {
#       if defined(_STARRY_TEMPORAL_) || defined(_STARRY_SPECTRAL_) 
            auto coeff = py::cast(coeff_.row(0));
            MAKE_READ_ONLY(coeff);
            return coeff;
#       else
            return py::cast<double>(coeff_(0));
#       endif
    } else {
        auto coeff = py::cast(coeff_);
        MAKE_READ_ONLY(coeff);
        return coeff;
    }
}

/**
Retrieve one or more limb darkening coefficients

*/
template <typename T>
py::object get_Ul(
    Map<T>& map, 
    const py::object& l
) {
    // Figure out the indices we're accessing
#   if defined(_STARRY_SPECTRAL_) && defined(_STARRY_LD_)
        auto inds = get_Ulw_inds(map.udeg, map.Nw, l);
        std::vector<int> rows = std::get<0>(inds);
        std::vector<int> cols = std::get<1>(inds);
        Matrix<double> coeff_(rows.size(), cols.size());
#   else
        std::vector<int> rows = get_Ul_inds(map.udeg, l);
        std::vector<int> cols(1, 0);
        Vector<double> coeff_(rows.size());
#   endif

    // Grab the map vector and update the output term by term
    auto u = map.getU();
    int i = 0;
    for (int row : rows) {
        int j = 0;
        for (int col : cols) {
            coeff_(i, j) = static_cast<double>(u(row, col));
            ++j;
        }
        ++i;
    }

    // Squeeze the output and cast to a py::array
    if (coeff_.size() == 1) {
#   if defined(_STARRY_SPECTRAL_) && defined(_STARRY_LD_)
            auto coeff = py::cast(coeff_.row(0));
            MAKE_READ_ONLY(coeff);
            return coeff;
#       else
            return py::cast<double>(coeff_(0));
#       endif
    } else {
        auto coeff = py::cast(coeff_);
        MAKE_READ_ONLY(coeff);
        return coeff;
    }
}

/**
Return a lambda function to compute the linear intensity model
on a grid.

*/
template <typename T>
std::function<py::object(
        Map<T> &, 
#       ifdef _STARRY_TEMPORAL_
            PY_ARRAY&, 
#       endif
        PY_ARRAY&,
        PY_ARRAY&, 
#       ifdef _STARRY_REFLECTED_
            PY_ARRAY&,
            PY_ARRAY& 
#       else
            PY_ARRAY&
#       endif
    )> linear_intensity_model () 
{
    return []
    (
        Map<T> &map, 
#ifdef _STARRY_TEMPORAL_
        PY_ARRAY& t_, 
#endif
        PY_ARRAY& theta_, 
        PY_ARRAY& x_, 
#ifdef _STARRY_REFLECTED_
        PY_ARRAY& y_,
        PY_ARRAY& source_
#else
        PY_ARRAY& y_
#endif
    ) -> py::object {
        using Scalar = typename T::Scalar;

        // Get Eigen references to the Python arrays
        MAP_TO_EIGEN_ROW_MATRIX(x_, x, Scalar);
        MAP_TO_EIGEN_ROW_MATRIX(y_, y, Scalar);

        // Ensure their shapes match
        assert((x.rows() == y.rows()) && (x.cols() == y.cols()));

        // Figure out the length of the timeseries
        std::vector<long> v{
#           ifdef _STARRY_TEMPORAL_
                t_.request().size,
#           endif
            theta_.request().size
        };
        py::ssize_t nt = *std::max_element(v.begin(), v.end());
#       ifdef _STARRY_REFLECTED_
            auto buf = source_.request();
            if ((buf.ndim == 2) && (buf.shape[0] > nt))
                nt = buf.shape[0];
#       endif

        // Get Eigen references to the Python arrays
#       ifdef _STARRY_TEMPORAL_
            MAP_TO_EIGEN_VECTOR(t_, t, Scalar, nt);
#       endif
        MAP_TO_EIGEN_VECTOR(theta_, theta, Scalar, nt);
#       ifdef _STARRY_REFLECTED_
            MAP_TO_EIGEN_ROW_MATRIX_OF_UNIT_VECTORS(source_, source, Scalar, nt);
#       endif

        // Compute the model
        map.computeLinearIntensityModel(
#           ifdef _STARRY_TEMPORAL_
                t,
#           endif
            theta, 
            x, 
            y, 
#           ifdef _STARRY_REFLECTED_
                source,
#           endif
            map.data.X
        );

        return PYOBJECT_CAST_ARR(map.data.X);

    };
}

/**
Return a lambda function to compute the linear flux model. 
Optionally compute and return the gradient.

*/
template <typename T>
std::function<py::object (
        Map<T> &, 
#       ifdef _STARRY_TEMPORAL_
            PY_ARRAY&,
#       endif
        PY_ARRAY&, 
        PY_ARRAY&, 
        PY_ARRAY&, 
        PY_ARRAY&,
        PY_ARRAY&,
#       ifdef _STARRY_REFLECTED_
            PY_ARRAY&,
#       endif
        bool
    )> linear_flux_model () 
{
    return []
    (
        Map<T> &map, 
#       ifdef _STARRY_TEMPORAL_
            PY_ARRAY& t_,
#       endif
        PY_ARRAY& theta_, 
        PY_ARRAY& xo_, 
        PY_ARRAY& yo_, 
        PY_ARRAY& zo_,
        PY_ARRAY& ro_,
#       ifdef _STARRY_REFLECTED_
            PY_ARRAY& source_,
#       endif
        bool compute_gradient
    ) -> py::object 
    {
        using Scalar = typename T::Scalar;
        auto reshape = py::module::import("numpy").attr("reshape");

        // Figure out the length of the timeseries
        std::vector<long> v {
#           ifdef _STARRY_TEMPORAL_
                t_.request().size,
#           endif
            theta_.request().size,
            xo_.request().size,
            yo_.request().size,
            zo_.request().size,
            ro_.request().size
        };
        py::ssize_t nt = *std::max_element(v.begin(), v.end());
#       ifdef _STARRY_REFLECTED_
            auto buf = source_.request();
            if ((buf.ndim == 2) && (buf.shape[0] > nt))
                nt = buf.shape[0];
#       endif

        // Get Eigen references to the Python arrays
#       ifdef _STARRY_TEMPORAL_
            MAP_TO_EIGEN_VECTOR(t_, t, Scalar, nt);
#       endif
        MAP_TO_EIGEN_VECTOR(theta_, theta, Scalar, nt);
        MAP_TO_EIGEN_VECTOR(xo_, xo, Scalar, nt);
        MAP_TO_EIGEN_VECTOR(yo_, yo, Scalar, nt);
        MAP_TO_EIGEN_VECTOR(zo_, zo, Scalar, nt);
        MAP_TO_EIGEN_VECTOR(ro_, ro, Scalar, nt);
#       ifdef _STARRY_REFLECTED_
            MAP_TO_EIGEN_ROW_MATRIX_OF_UNIT_VECTORS(source_, source, Scalar, nt);
#       endif

        if (!compute_gradient) {

            // Compute the model and return
            map.computeLinearFluxModel(
#               ifdef _STARRY_TEMPORAL_
                    t, 
#               endif
                theta, 
                xo, 
                yo, 
                zo, 
                ro, 
#               ifdef _STARRY_REFLECTED_
                    source, 
#               endif
                map.data.X
            );

            return PYOBJECT_CAST_ARR(map.data.X);

        } else {

            // Compute the model + gradient
            map.computeLinearFluxModel(
#               ifdef _STARRY_TEMPORAL_
                    t,
#               endif
                theta, 
                xo, 
                yo, 
                zo, 
                ro,
#               ifdef _STARRY_REFLECTED_
                    source, 
#               endif 
                map.data.X, 
#               ifdef _STARRY_TEMPORAL_
                    map.data.DXDt,
#               endif
                map.data.DXDtheta, 
                map.data.DXDxo, 
                map.data.DXDyo, 
                map.data.DXDro,
#               ifdef _STARRY_REFLECTED_
                    map.data.DXDsource, 
#               endif   
                map.data.DXDu, 
                map.data.DXDinc,
                map.data.DXDobl
            );

            // Get Eigen references to the arrays, as these
            // are automatically passed by ref to the Python side
#           ifdef _STARRY_TEMPORAL_
                auto Dt = Ref<RowMatrix<Scalar>>(map.data.DXDt);
#           endif
            auto Dtheta = Ref<RowMatrix<Scalar>>(map.data.DXDtheta);
            auto Dxo = Ref<RowMatrix<Scalar>>(map.data.DXDxo);
            auto Dyo = Ref<RowMatrix<Scalar>>(map.data.DXDyo);
            auto Dro = Ref<RowMatrix<Scalar>>(map.data.DXDro);
#           ifdef _STARRY_REFLECTED_
                auto Dsource = Ref<RowMatrix<Scalar>>(map.data.DXDsource);
#           endif
            auto Du = Ref<RowMatrix<Scalar>>(map.data.DXDu);
            auto Dinc = Ref<RowMatrix<Scalar>>(map.data.DXDinc);
            auto Dobl = Ref<RowMatrix<Scalar>>(map.data.DXDobl);

            // Construct a dictionary
            py::dict gradient = py::dict(
#               ifdef _STARRY_TEMPORAL_
                    "t"_a=ENSURE_DOUBLE_ARR(Dt),
#               endif
                "theta"_a=ENSURE_DOUBLE_ARR(Dtheta),
                "xo"_a=ENSURE_DOUBLE_ARR(Dxo),
                "yo"_a=ENSURE_DOUBLE_ARR(Dyo),
                "ro"_a=ENSURE_DOUBLE_ARR(Dro),
#               ifdef _STARRY_REFLECTED_
                    "source"_a=ENSURE_DOUBLE_ARR(Dsource),
#               endif    
                "u"_a=reshape(ENSURE_DOUBLE_ARR(Du), 
                              py::make_tuple(map.Nu - 1, nt, map.Ny * map.Nt)),
                "inc"_a=ENSURE_DOUBLE_ARR(Dinc),
                "obl"_a=ENSURE_DOUBLE_ARR(Dobl)           
            );

            // Return
            return py::make_tuple(
                ENSURE_DOUBLE_ARR(map.data.X), 
                gradient
            );

        }

    };
}

/**
Return a lambda function to compute the limb-darkened flux.
Optionally compute and return the gradient.

*/
template <typename T>
std::function<py::object (
        Map<T> &,
        PY_ARRAY&, 
        PY_ARRAY&, 
        PY_ARRAY&, 
        bool
    )> ld_flux () 
{
    return []
    (
        Map<T> &map, 
        PY_ARRAY& b_, 
        PY_ARRAY& zo_,
        PY_ARRAY& ro_,
        bool compute_gradient
    ) -> py::object 
    {
        using Scalar = typename T::Scalar;
        auto reshape = py::module::import("numpy").attr("reshape");

        // Figure out the length of the timeseries
        std::vector<long> v {
            b_.request().size,
            zo_.request().size,
            ro_.request().size
        };
        py::ssize_t nt = *std::max_element(v.begin(), v.end());

        // Get Eigen references to the Python arrays
        MAP_TO_EIGEN_VECTOR(b_, b, Scalar, nt);
        MAP_TO_EIGEN_VECTOR(zo_, zo, Scalar, nt);
        MAP_TO_EIGEN_VECTOR(ro_, ro, Scalar, nt);

        if (!compute_gradient) {

            // Compute the model and return
            map.computeLimbDarkenedFlux(
                b, 
                zo, 
                ro, 
#               if defined(_STARRY_SPECTRAL_)
                    map.data.flux_spectral
#               else
                    map.data.flux
#               endif
            );

#           if defined(_STARRY_SPECTRAL_)
                return PYOBJECT_CAST_ARR(map.data.flux_spectral);
#           else
                return PYOBJECT_CAST_ARR(map.data.flux);
#           endif
            
        } else {

            // Compute the model + gradient
            map.computeLimbDarkenedFlux(
                b, 
                zo, 
                ro,
#               if defined(_STARRY_SPECTRAL_)
                    map.data.flux_spectral,
                    map.data.DfDb_spectral, 
                    map.data.DfDro_spectral,
#               else
                    map.data.flux,
                    map.data.DfDb, 
                    map.data.DfDro,
#               endif
                map.data.DfDu
            );

            // Get Eigen references to the arrays, as these
            // are automatically passed by ref to the Python side
#           if defined(_STARRY_SPECTRAL_)
                auto Db = Ref<Matrix<Scalar>>(map.data.DfDb_spectral);
                auto Dro = Ref<Matrix<Scalar>>(map.data.DfDro_spectral);
#           else
                auto Db = Ref<Vector<Scalar>>(map.data.DfDb);
                auto Dro = Ref<Vector<Scalar>>(map.data.DfDro);
#           endif
            auto Du = Ref<Matrix<Scalar>>(map.data.DfDu);

            // Construct a dictionary
            py::dict gradient = py::dict(
                "b"_a=ENSURE_DOUBLE_ARR(Db),
                "ro"_a=ENSURE_DOUBLE_ARR(Dro),  
#           if defined(_STARRY_SPECTRAL_)
                "u"_a=reshape(ENSURE_DOUBLE_ARR(Du), 
                              py::make_tuple(map.udeg, nt, map.Nw))
#           else
                "u"_a=ENSURE_DOUBLE_ARR(Du)   
#           endif
            );

            // Return
            return py::make_tuple(
#               if defined(_STARRY_SPECTRAL_)
                    ENSURE_DOUBLE_ARR(map.data.flux_spectral),
#               else
                    ENSURE_DOUBLE_ARR(map.data.flux),
#               endif
                gradient
            );

        }

    };
}

} // namespace interface

#endif



