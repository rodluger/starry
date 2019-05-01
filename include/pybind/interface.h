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

#define MAKE_READ_ONLY(X) \
    reinterpret_cast<py::detail::PyArray_Proxy*>(X.ptr())->flags &= \
        ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;

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
Set one or more filter coefficients

*/
template <typename T>
void set_Flm(
    Map<T>& map, 
    const py::tuple& lm,
    py::array_t<double>& coeff_
) {
    using Scalar = typename T::Scalar;
    std::vector<int> rows = get_Ylm_inds(map.fdeg, lm);
    std::vector<int> cols(1, 0);

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

    // Check shape
    if (!((size_t(coeff.rows()) == size_t(rows.size())) && 
            (size_t(coeff.cols()) == size_t(cols.size()))))
        throw std::length_error("Mismatch in index array and " 
                                "coefficient array sizes.");

    // Grab the map vector and update it term by term
    auto f = map.getF();
    int i = 0;
    for (int row : rows) {
        int j = 0;
        for (int col : cols) {
            f(row, col) = static_cast<Scalar>(coeff(i, j));
            ++j;
        }
        ++i;
    }
    map.setF(f);
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
Retrieve one or more filter coefficients

*/
template <typename T>
py::object get_Flm (
    Map<T>& map,
    const py::tuple& lm
) {
    // Figure out the indices we're accessing
    std::vector<int> rows = get_Ylm_inds(map.fdeg, lm);
    std::vector<int> cols(1, 0);
    Vector<double> coeff_(rows.size());

    // Grab the map vector and update the output vector term by term
    auto f = map.getF();
    int i = 0;
    for (int row : rows) {
        int j = 0;
        for (int col : cols) {
            coeff_(i, j) = static_cast<double>(f(row, col));
            ++j;
        }
        ++i;
    }

    // Squeeze the output and cast to a py::array
    if (coeff_.size() == 1) {
        return py::cast<double>(coeff_(0));
    } else {
        auto coeff = py::cast(coeff_);
        MAKE_READ_ONLY(coeff);
        return coeff;
    }
}

} // namespace interface

#endif



