/**
Miscellaneous utilities used for the pybind interface.

*/

#ifndef _STARRY_PYBIND_UTILS_H_
#define _STARRY_PYBIND_UTILS_H_

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <stdlib.h>
#include <vector>
#include "errors.h"

namespace pybind_utils {

    #include <Python.h>
    namespace py = pybind11;

    /**
    Re-interpret the `start`, `stop`, and `step` attributes of a `py::slice`,
    allowing for *actual* negative indices. This allows the user to provide
    something like `map[3, -3:0]` to get the `l = 3, m = {-3, -2, -1}` indices
    of the spherical harmonic map. Pretty sneaky stuff.

    */
    void reinterpret_slice(const py::slice& slice, const int smin,
                           const int smax, int& start, int& stop, int& step) {
        PySliceObject *r = (PySliceObject*)(slice.ptr());
        if (r->start == Py_None)
            start = smin;
        else
            start = PyLong_AsSsize_t(r->start);
        if (r->stop == Py_None)
            stop = smax;
        else
            stop = PyLong_AsSsize_t(r->stop);
        if ((r->step == Py_None) || (PyLong_AsSsize_t(r->step) == 1))
            step = 1;
        else
            throw errors::ValueError("Slices with steps different from "
                                     "one are not supported.");
    }

    /**
    Parse a user-provided `(l, m)` tuple into spherical harmonic map indices.

    */
    std::vector<int> get_Ylm_inds(const int lmax, const py::tuple& lm) {
        int N = (lmax + 1) * (lmax + 1);
        int n;
        if (lm.size() != 2)
            throw errors::IndexError("Invalid `l`, `m` tuple.");
        std::vector<int> inds;
        if ((py::isinstance<py::int_>(lm[0])) && (py::isinstance<py::int_>(lm[1]))) {
            // User provided `(l, m)`
            int l = py::cast<int>(lm[0]);
            int m = py::cast<int>(lm[1]);
            n = l * l + l + m;
            if ((n < 0) || (n >= N))
                throw errors::IndexError("Invalid value for `l` and/or `m`.");
            inds.push_back(n);
            return inds;
        } else if ((py::isinstance<py::slice>(lm[0])) && (py::isinstance<py::slice>(lm[1]))) {
            // User provided `(slice, slice)`
            auto lslice = py::cast<py::slice>(lm[0]);
            auto mslice = py::cast<py::slice>(lm[1]);
            int lstart, lstop, lstep;
            int mstart, mstop, mstep;
            reinterpret_slice(lslice, 0, lmax, lstart, lstop, lstep);
            if ((lstart < 0) || (lstart > lmax))
                throw errors::IndexError("Invalid value for `l`.");
            for (int l = lstart; l < lstop + 1; l += lstep) {
                reinterpret_slice(mslice, -l, l, mstart, mstop, mstep);
                if (mstart < -l)
                    mstart = -l;
                if (mstop > l)
                    mstop = l;
                for (int m = mstart; m < mstop + 1; m += mstep) {
                    n = l * l + l + m;
                    if ((n < 0) || (n >= N))
                        throw errors::IndexError("Invalid value for `l` and/or `m`.");
                    inds.push_back(n);
                }
            }
            return inds;
        } else if ((py::isinstance<py::int_>(lm[0])) && (py::isinstance<py::slice>(lm[1]))) {
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
                if ((n < 0) || (n >= N))
                    throw errors::IndexError("Invalid value for `l` and/or `m`.");
                inds.push_back(n);
            }
            return inds;
        } else if ((py::isinstance<py::slice>(lm[0])) && (py::isinstance<py::int_>(lm[1]))) {
            // User provided `(slice, m)`
            int m = py::cast<int>(lm[1]);
            auto lslice = py::cast<py::slice>(lm[0]);
            int lstart, lstop, lstep;
            reinterpret_slice(lslice, 0, lmax, lstart, lstop, lstep);
            if ((lstart < 0) || (lstart > lmax))
                throw errors::IndexError("Invalid value for `l`.");
            for (int l = lstart; l < lstop + 1; l += lstep) {
                if ((m < -l) || (m > l))
                    continue;
                n = l * l + l + m;
                if ((n < 0) || (n >= N))
                    throw errors::IndexError("Invalid value for `l` and/or `m`.");
                inds.push_back(n);
            }
            return inds;
        } else {
            // User provided something silly
            throw errors::IndexError("Unsupported input type for `l` and/or `m`.");
        }
    }

    /**
    Parse a user-provided `l` into limb darkening map indices.

    */
    std::vector<int> get_Ul_inds(int lmax, const py::object& l) {
        int n;
        std::vector<int> inds;
        if (py::isinstance<py::int_>(l)) {
            n = py::cast<int>(l);
            if ((n < 1) || (n > lmax))
                throw errors::IndexError("Invalid value for `l`.");
            inds.push_back(n);
            return inds;
        } else if (py::isinstance<py::slice>(l)) {
            py::slice slice = py::cast<py::slice>(l);
            ssize_t start, stop, step, slicelength;
            if(!slice.compute(lmax,
                              reinterpret_cast<size_t*>(&start),
                              reinterpret_cast<size_t*>(&stop),
                              reinterpret_cast<size_t*>(&step),
                              reinterpret_cast<size_t*>(&slicelength)))
                throw pybind11::error_already_set();
            if ((start < 0) || (start > lmax)) {
                throw errors::IndexError("Invalid value for `l`.");
            } else if (step < 0) {
                throw errors::ValueError("Slices with negative steps are not supported.");
            } else if (start == 0) {
                // Let's give the user the benefit of the doubt here
                start = 1;
            }
            std::vector<int> inds;
            for (ssize_t i = start; i < stop + 1; i += step) {
                inds.push_back(i);
            }
            return inds;
        } else {
            // User provided something silly
            throw errors::IndexError("Unsupported input type for `l`.");
        }
    }

} // namespace pybind_utils

#endif
