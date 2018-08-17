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

    namespace py = pybind11;

    /**
    Re-interpret the `start`, `stop`, and `step` attributes of a `py::slice`,
    allowing for *actual* negative indices.

    */
    void reinterpret_slice(const py::slice& slice, const int smin,
                           const int smax, int& start, int& stop, int& step) {
        // NOTE: This is super hacky. Because `m` indices can be negative, we need
        // to re-interpret what a slice with negative indices actually
        // means. Casting to an actual Python slice and running
        // `compute` interprets negative indices as indices counting
        // backwards from the end, which is not what we want. There
        // doesn't seem to be a way to reconstruct the original arguments
        // to `slice(start, stop, step)` that works in *all* cases (I've tried!)
        // so for now we'll parse the string representation of the slice.
        //
        // NOTE: This is likely slow, so a hack that digs into the actual
        // CPython backend and recovers the `start`, `stop`, and `step`
        // attributes of a `py::slice` object would be better. Suggestions welcome!
        std::ostringstream os;
        os << slice;
        std::string str_slice = std::string(os.str());
        size_t pos = 0;
        std::string str_start, str_stop, str_step;
        pos = str_slice.find(", ");
        str_start = str_slice.substr(6, pos - 6);
        str_slice = str_slice.substr(pos + 2, str_slice.size() - pos);
        pos = str_slice.find(", ");
        str_stop = str_slice.substr(0, pos);
        str_step = str_slice.substr(pos + 2, str_slice.size() - pos - 3);
        if (str_start == "None")
            start = smin;
        else
            start = stoi(str_start);
        if (str_stop == "None")
            stop = smax;
        else
            stop = stoi(str_stop);
        if (str_step == "None")
            step = 1;
        else
            step = stoi(str_step);
        if (step < 0)
            throw errors::ValueError("Slices with negative steps are not supported.");
    }

    /**
    Parse a user-provided `(l, m)` tuple into spherical harmonic map indices.

    */
    std::vector<int> get_Ylm_inds(const int lmax, const py::tuple& lm) {
        if (lm.size() != 2)
            throw errors::IndexError("Invalid `l`, `m` tuple.");
        std::vector<int> inds;
        if ((py::isinstance<py::int_>(lm[0])) && (py::isinstance<py::int_>(lm[1]))) {
            // User provided `(l, m)`
            int l = py::cast<int>(lm[0]);
            int m = py::cast<int>(lm[1]);
            inds.push_back(l * l + l + m);
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
                    inds.push_back(l * l + l + m);
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
                inds.push_back(l * l + l + m);
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
                inds.push_back(l * l + l + m);
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
        std::vector<int> inds;
        if (py::isinstance<py::int_>(l)) {
            inds.push_back(py::cast<int>(l));
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
