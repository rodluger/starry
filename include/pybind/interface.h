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
#include <starry/errors.h>
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

#define MAKE_READ_ONLY(X) \
    reinterpret_cast<py::detail::PyArray_Proxy*>(X.ptr())->flags &= \
        ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;

#ifdef _STARRY_DOUBLE_
#   define MAP_TO_EIGEN(PYX, X, T, N) \
        Eigen::Map<Vector<T>> X(NULL, N, 1); \
        Vector<T> tmp##X; \
        buf = PYX.request(); \
        ptr = (double *) buf.ptr; \
        if (buf.ndim == 0) { \
            tmp##X = ptr[0] * Vector<T>::Ones(N); \
            new (&X) Eigen::Map<Vector<T>>(&tmp##X(0), N, 1); \
        } else if ((buf.ndim == 1) && (buf.size == N)) { \
            new (&X) Eigen::Map<Vector<T>>(ptr, N, 1); \
        } else { \
            throw errors::ShapeError("A vector has the incorrect shape."); \
        }
#else
#   define MAP_TO_EIGEN(PYX, X, T, N) \
        Eigen::Map<Vector<T>> X(NULL, N, 1); \
        Vector<T> tmp##X; \
        buf = PYX.request(); \
        ptr = (double *) buf.ptr; \
        if (buf.ndim == 0) { \
            tmp##X = ptr[0] * Vector<T>::Ones(N); \
            new (&X) Eigen::Map<Vector<T>>(&tmp##X(0), N, 1); \
        } else if ((buf.ndim == 1) && (buf.size == N)) { \
            tmp##X = (py::cast<Vector<double>>(PYX)).template cast<T>(); \
            new (&X) Eigen::Map<Vector<T>>(&tmp##X(0), N, 1); \
        } else { \
            throw errors::ShapeError("A vector has the incorrect shape."); \
        }
#endif

namespace interface {

//! Misc stuff we need
#include <Python.h>
using namespace pybind11::literals;
using namespace starry;
using namespace starry::utils;
using starry::maps::Map;
static const auto integer = py::module::import("numpy").attr("integer");


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
        throw errors::ValueError("Slices with steps different from "
                                 "one are not supported.");
}

/**
Parse a user-provided `(l, m)` tuple into spherical harmonic map indices.

*/
std::vector<int> get_Ylm_inds (
    const int lmax, 
    const py::tuple& lm
) {
    int N = (lmax + 1) * (lmax + 1);
    int n;
    if (lm.size() != 2)
        throw errors::IndexError("Invalid `l`, `m` tuple.");
    std::vector<int> inds;
    if ((py::isinstance<py::int_>(lm[0]) || 
         py::isinstance(lm[0], integer)) && 
        (py::isinstance<py::int_>(lm[1]) || 
         py::isinstance(lm[1], integer))) {
        // User provided `(l, m)`
        int l = py::cast<int>(lm[0]);
        int m = py::cast<int>(lm[1]);
        n = l * l + l + m;
        if ((n < 0) || (n >= N))
            throw errors::IndexError("Invalid value for `l` and/or `m`.");
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
                    throw errors::IndexError(
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
            if ((n < 0) || (n >= N))
                throw errors::IndexError("Invalid value for `l` and/or `m`.");
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
std::vector<int> get_Ul_inds (
    int lmax, 
    const py::object& l
) {
    int n;
    std::vector<int> inds;
    if (py::isinstance<py::int_>(l) || py::isinstance(l, integer)) {
        n = py::cast<int>(l);
        if ((n < 1) || (n > lmax))
            throw errors::IndexError("Invalid value for `l`.");
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
            throw errors::IndexError("Invalid value for `l`.");
        } else if (step < 0) {
            throw errors::ValueError(
                "Slices with negative steps are not supported.");
        } else if (start == 0) {
            // Let's give the user the benefit of the doubt here
            start = 1;
        }
        std::vector<int> inds;
        for (ssize_t i = start; i < stop; i += step) {
            inds.push_back(i);
        }
        return inds;
    } else {
        // User provided something silly
        throw errors::IndexError("Unsupported input type for `l`.");
    }
}

/**
Return a lambda function to compute the intensity at a point 
or a vector of points.

*/
template <typename T>
std::function<py::object(
        Map<T> &, 
#       ifdef _STARRY_TEMPORAL_
            py::array_t<double>&, 
#       endif
        py::array_t<double>&, 
        py::array_t<double>&, 
#       ifdef _STARRY_REFLECTED_
            py::array_t<double>&,
            py::array_t<double>& 
#       else
            py::array_t<double>&
#       endif
    )> intensity () 
{
    return []
    (
        Map<T> &map, 
#ifdef _STARRY_TEMPORAL_
        py::array_t<double>& t, 
#endif
        py::array_t<double>& theta, 
        py::array_t<double>& x, 
#ifdef _STARRY_REFLECTED_
        py::array_t<double>& y,
        py::array_t<double>& source_
#else
        py::array_t<double>& y
#endif
    ) -> py::object {
        using Scalar = typename T::Scalar;
#       ifdef _STARRY_REFLECTED_
            // Pick out the columns of the `source_` numpy array so we 
            // can vectorize it easily.
            py::buffer_info buf = source_.request();
            double *ptr = (double *) buf.ptr;
            assert(
                ((buf.ndim == 1) && (buf.size == 3)) ||
                ((buf.ndim == 2) && (buf.shape[1] == 3))
            );
            auto sx = py::array_t<double>(buf.size / 3);
            py::buffer_info bufx = sx.request();
            double *ptrx = (double *) bufx.ptr;
            auto sy = py::array_t<double>(buf.size / 3);
            py::buffer_info bufy = sy.request();
            double *ptry = (double *) bufy.ptr;
            auto sz = py::array_t<double>(buf.size / 3);
            py::buffer_info bufz = sz.request();
            double *ptrz = (double *) bufz.ptr;
            for (int i = 0; i < sx.size(); ++i) {
                ptrx[i] = ptr[3 * i];
                ptry[i] = ptr[3 * i + 1];
                ptrz[i] = ptr[3 * i + 2];
            }
#       endif
#       ifdef _STARRY_TEMPORAL_
#           ifdef _STARRY_EMITTED_
                std::vector<long> v{t.size(), theta.size(), 
                                    x.size(), y.size()};
#           else
                std::vector<long> v{t.size(), theta.size(), x.size(), 
                                    y.size(), sx.size()};
#           endif
#       else
#           ifdef _STARRY_EMITTED_
                std::vector<long> v{theta.size(), x.size(), y.size()};
#           else
                std::vector<long> v{theta.size(), x.size(), 
                                    y.size(), sx.size()};
#           endif
#       endif
        size_t nt = *std::max_element(v.begin(), v.end());
        size_t n = 0;
#       ifdef _STARRY_SPECTRAL_
            RowMatrix<Scalar> intensity(nt, map.nflx);
#       else
            Vector<Scalar> intensity(nt);
#       endif
        py::vectorize([&map, &intensity, &n](
#           ifdef _STARRY_TEMPORAL_
                double t,
#           endif
            double theta, 
            double x, 
#           ifdef _STARRY_REFLECTED_
                double y,
                double sx, 
                double sy, 
                double sz
#           else
                double y
#           endif
        ) {
#           ifdef _STARRY_REFLECTED_
                UnitVector<Scalar> source(3);
                source << sx, sy, sz;
#           endif
            map.computeIntensity(
#               ifdef _STARRY_TEMPORAL_
                    static_cast<Scalar>(t), 
#               endif
                static_cast<Scalar>(theta), 
                static_cast<Scalar>(x), 
                static_cast<Scalar>(y), 
#               ifdef _STARRY_REFLECTED_
                    source,
#               endif
                intensity.row(n)
            );
            ++n;
            return 0;
        })(
#           ifdef _STARRY_TEMPORAL_
                t, 
#           endif
            theta, 
            x, 
#           ifdef _STARRY_REFLECTED_
                y,
                sx,
                sy,
                sz
#           else
                y
#           endif
        );
        if (nt > 1) {
            return py::cast(intensity.template cast<double>());
        } else {
#           ifdef _STARRY_SPECTRAL_
                RowVector<double> f = intensity.row(0).template cast<double>();
                return py::cast(f);
#           else
                return py::cast(static_cast<double>(intensity(0)));
#           endif
        }
    };
}

/**
Return a lambda function to compute the linear model at a point 
or a vector of points. Optionally compute and return 
the gradient.

\todo Implement this for reflected types

*/
template <typename T>
std::function<py::object (
        Map<T> &, 
#       ifdef _STARRY_TEMPORAL_
            py::array_t<double>&,
#       endif
        py::array_t<double>&, 
        py::array_t<double>&, 
        py::array_t<double>&, 
        py::array_t<double>&,
        py::array_t<double>&,
        bool
    )> linear_flux_model () 
{
    return []
    (
        Map<T> &map, 
#       ifdef _STARRY_TEMPORAL_
            py::array_t<double>& t_,
#       endif
        py::array_t<double>& theta_, 
        py::array_t<double>& xo_, 
        py::array_t<double>& yo_, 
        py::array_t<double>& zo_,
        py::array_t<double>& ro_,
        bool compute_gradient
    ) -> py::object 
    {
        using Scalar = typename T::Scalar;

        // Figure out the length of the timeseries
        std::vector<long> v{
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
        py::buffer_info buf;
        double *ptr;

        // Get Eigen references to the Python arrays
#       ifdef _STARRY_TEMPORAL_
            MAP_TO_EIGEN(t_, t, Scalar, nt);
#       endif
        MAP_TO_EIGEN(theta_, theta, Scalar, nt);
        MAP_TO_EIGEN(xo_, xo, Scalar, nt);
        MAP_TO_EIGEN(yo_, yo, Scalar, nt);
        MAP_TO_EIGEN(zo_, zo, Scalar, nt);
        MAP_TO_EIGEN(ro_, ro, Scalar, nt);

        if (!compute_gradient) {

            // Compute the model and return
#           ifdef _STARRY_TEMPORAL_
                map.computeLinearFluxModel(t, theta, xo, yo, zo, ro, map.data.A);
#           else
                map.computeLinearFluxModel(theta, xo, yo, zo, ro, map.data.A);
#           endif
            return PYOBJECT_CAST_ARR(map.data.A);

        } else {

            // Compute the model + gradient
            map.computeLinearFluxModel(
#               ifdef _STARRY_TEMPORAL_
                    t,
#               endif
                theta, xo, yo, zo, ro, map.data.A, 
#               ifdef _STARRY_TEMPORAL_
                    map.data.DADt,
#               endif
                map.data.DADtheta, map.data.DADxo, 
                map.data.DADyo, map.data.DADro
            );

            // Get Eigen references to the arrays, as these
            // are automatically passed by ref to the Python side
#           ifdef _STARRY_TEMPORAL_
                auto Dt = Ref<RowMatrix<Scalar>>(map.data.DADt);
#           endif
            auto Dtheta = Ref<RowMatrix<Scalar>>(map.data.DADtheta);
            auto Dxo = Ref<RowMatrix<Scalar>>(map.data.DADxo);
            auto Dyo = Ref<RowMatrix<Scalar>>(map.data.DADyo);
            auto Dro = Ref<RowMatrix<Scalar>>(map.data.DADro);

            // Construct a dictionary
            py::dict gradient = py::dict(
#               ifdef _STARRY_TEMPORAL_
                    "t"_a=ENSURE_DOUBLE_ARR(Dt),
#               endif
                "theta"_a=ENSURE_DOUBLE_ARR(Dtheta),
                "xo"_a=ENSURE_DOUBLE_ARR(Dxo),
                "yo"_a=ENSURE_DOUBLE_ARR(Dyo),
                "ro"_a=ENSURE_DOUBLE_ARR(Dro)
            );

            // Return
            return py::make_tuple(
                ENSURE_DOUBLE_ARR(map.data.A), 
                gradient
            );

        }

    };
}

} // namespace interface

#endif



