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
#define ENSURE_DOUBLE(X)               static_cast<double>(X)
#define ENSURE_DOUBLE_ARR(X)           X.template cast<double>()
#define PYOBJECT_CAST(X)               py::cast(static_cast<double>(X))
#define PYOBJECT_CAST_ARR(X)           py::cast(X.template cast<double>())
#else
#define ENSURE_DOUBLE(X)               X
#define ENSURE_DOUBLE_ARR(X)           X
#define PYOBJECT_CAST(X)               py::cast(X)
#define PYOBJECT_CAST_ARR(X)           py::cast(&X)
#endif

#define MAKE_READ_ONLY(X)              reinterpret_cast<py::detail::PyArray_Proxy*>(X.ptr())->flags &= \
                                       ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;

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
#ifdef _STARRY_TEMPORAL_
        py::array_t<double>&, 
#endif
        py::array_t<double>&, 
        py::array_t<double>&, 
        py::array_t<double>&
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
        py::array_t<double>& y
    ) -> py::object {
        using Scalar = typename T::Scalar;
#ifdef _STARRY_TEMPORAL_
        std::vector<long> v{t.size(), theta.size(), x.size(), y.size()};
#else
        std::vector<long> v{theta.size(), x.size(), y.size()};
#endif
        size_t nt = *std::max_element(v.begin(), v.end());
        size_t n = 0;
#ifdef _STARRY_SPECTRAL_
        RowMatrix<Scalar> intensity(nt, map.nflx);
#else
        Vector<Scalar> intensity(nt);
#endif
        py::vectorize([&map, &intensity, &n](
#ifdef _STARRY_TEMPORAL_
            double t,
#endif
            double theta, 
            double x, 
            double y
        ) {
            map.computeIntensity(
#ifdef _STARRY_TEMPORAL_
                static_cast<Scalar>(t), 
#endif
                static_cast<Scalar>(theta), 
                static_cast<Scalar>(x), 
                static_cast<Scalar>(y), 
                intensity.row(n)
            );
            ++n;
            return 0;
        })(
#ifdef _STARRY_TEMPORAL_
            t, 
#endif
            theta, 
            x, 
            y
        );
        if (nt > 1) {
            return py::cast(intensity.template cast<double>());
        } else {
#ifdef _STARRY_SPECTRAL_
            RowVector<double> f = intensity.row(0).template cast<double>();
            return py::cast(f);
#else
            return py::cast(static_cast<double>(intensity(0)));
#endif
        }
    };
}

/**
Return a lambda function to compute the flux at a point 
or a vector of points. Optionally compute and return 
the gradient.

*/
template <typename T>
std::function<py::object(
        Map<T> &, 
#ifdef _STARRY_TEMPORAL_
        py::array_t<double>&, 
#endif
        py::array_t<double>&, 
        py::array_t<double>&, 
        py::array_t<double>&, 
        py::array_t<double>&,
        py::array_t<double>&,
#if defined(_STARRY_REFLECTED_)
        py::array_t<double>&, 
#endif
        bool
    )> flux () 
{
    return []
    (
        Map<T> &map, 
#ifdef _STARRY_TEMPORAL_
        py::array_t<double>& t, 
#endif
        py::array_t<double>& theta, 
        py::array_t<double>& xo, 
        py::array_t<double>& yo, 
        py::array_t<double>& zo,
        py::array_t<double>& ro,
#if defined(_STARRY_REFLECTED_)
        py::array_t<double>& source_, 
#endif
        bool compute_gradient
    ) -> py::object 
    {
        using Scalar = typename T::Scalar;
        using TSType = typename T::TSType;

#if defined(_STARRY_SPECTRAL_) || defined(_STARRY_TEMPORAL_)
        // We need our old friend numpy to reshape
        // matrices into 3-tensors on the Python side
        auto numpy = py::module::import("numpy");
        auto reshape = numpy.attr("reshape");
#endif

#if defined(_STARRY_REFLECTED_)
        // Convert the `source` to an Eigen unit vector
        py::buffer_info buf = source_.request();
        assert(buf.ndim == 1);
        assert(buf.size == 3);
        double *ptr = (double *) buf.ptr;
        UnitVector<Scalar> source(3);
        for (int i = 0; i < 3; ++i)
            source(i) = ptr[i];
#endif

#ifdef _STARRY_TEMPORAL_
        std::vector<long> v{t.size(), theta.size(), xo.size(), yo.size(), zo.size(), ro.size()};
#else
        std::vector<long> v{theta.size(), xo.size(), yo.size(), zo.size(), ro.size()};
#endif
        size_t nt = *std::max_element(v.begin(), v.end());
        size_t n = 0;

        // Allocate space for the flux
        map.cache.pb_flux.resize(nt, map.nflx);

        if (compute_gradient) {

            // Allocate space for the gradient
            map.cache.pb_Dtheta.resize(nt, map.nflx);
            map.cache.pb_Dxo.resize(nt, map.nflx);
            map.cache.pb_Dyo.resize(nt, map.nflx);
            map.cache.pb_Dro.resize(nt, map.nflx);

            // The y and u derivs have variable shapes
            int ny, nu;
            if (map.getYDeg() == 0) {
                ny = 1;
                nu = map.lmax + STARRY_DFDU_DELTA;
            } else if (map.getUDeg() == 0) {
                ny = map.N;
                nu = 0;
            } else {
                ny = map.N;
                nu = map.lmax + STARRY_DFDU_DELTA;
            } 

#if defined(_STARRY_DEFAULT_)
            map.cache.pb_Dy.resize(ny, nt);
            map.cache.pb_Du.resize(nu, nt);
#elif defined(_STARRY_SPECTRAL_)
            map.cache.pb_Dy.resize(ny * nt, map.ncoly);
            map.cache.pb_Du.resize(nu * nt, map.ncolu);
#elif defined(_STARRY_TEMPORAL_)
            map.cache.pb_Dt.resize(nt, map.nflx);
            map.cache.pb_Dy.resize(ny * nt, map.ncoly);
            map.cache.pb_Du.resize(nu, nt);
#endif

#if defined(_STARRY_REFLECTED_)
            map.cache.pb_Dsource.resize(3, nt);
#endif

            // Vectorize the computation
            py::vectorize([&map, 
#if defined(_STARRY_REFLECTED_)
                           &source,
#endif            
                           &n, &ny, &nu](

#ifdef _STARRY_TEMPORAL_
                double t, 
#endif
                double theta, 
                double xo, 
                double yo, 
                double zo,
                double ro
            ) {
                map.computeFlux(
#ifdef _STARRY_TEMPORAL_
                    static_cast<Scalar>(t),
#endif
                    static_cast<Scalar>(theta),
                    static_cast<Scalar>(xo),
                    static_cast<Scalar>(yo),
                    static_cast<Scalar>(zo),
                    static_cast<Scalar>(ro),
#ifdef _STARRY_REFLECTED_
                    source,
#endif
                    map.cache.pb_flux.row(n),
#ifdef _STARRY_TEMPORAL_
                    map.cache.pb_Dt.row(n),
#endif
                    map.cache.pb_Dtheta.row(n),
                    map.cache.pb_Dxo.row(n),
                    map.cache.pb_Dyo.row(n),
                    map.cache.pb_Dro.row(n),
#if defined(_STARRY_DEFAULT_)
                    map.cache.pb_Dy.col(n),
                    map.cache.pb_Du.col(n)
#elif defined(_STARRY_SPECTRAL_)
                    map.cache.pb_Dy.block(n * ny, 0, ny, map.ncoly),
                    map.cache.pb_Du.block(n * nu, 0, nu, map.ncolu)
#elif defined(_STARRY_TEMPORAL_)
                    map.cache.pb_Dy.block(n * ny, 0, ny, map.ncoly),
                    map.cache.pb_Du.col(n)
#endif
#if defined(_STARRY_REFLECTED_)
                    , map.cache.pb_Dsource.col(n)
#endif
                );
                ++n;
                return 0;
            })(
#ifdef _STARRY_TEMPORAL_
                t,
#endif
                theta, 
                xo, 
                yo, 
                zo,
                ro
            );

            // Construct the gradient dictionary and
            // return a tuple of (flux, gradient)
            if (nt > 1) {

                // Get Eigen references to the arrays, as these
                // are automatically passed by ref to the Python side
                auto flux = Ref<TSType>(map.cache.pb_flux);
                auto Dtheta = Ref<TSType>(map.cache.pb_Dtheta);
                auto Dxo = Ref<TSType>(map.cache.pb_Dxo);
                auto Dyo = Ref<TSType>(map.cache.pb_Dyo);
                auto Dro = Ref<TSType>(map.cache.pb_Dro);
                auto Dy = Ref<RowMatrix<Scalar>>(map.cache.pb_Dy);
                auto Du = Ref<RowMatrix<Scalar>>(map.cache.pb_Du);

#if defined(_STARRY_REFLECTED_)
                auto Dsource = Ref<RowMatrix<Scalar>>(map.cache.pb_Dsource);
#endif

#if defined(_STARRY_DEFAULT_)
                py::dict gradient = py::dict(
                    "theta"_a=ENSURE_DOUBLE_ARR(Dtheta),
                    "xo"_a=ENSURE_DOUBLE_ARR(Dxo),
                    "yo"_a=ENSURE_DOUBLE_ARR(Dyo),
                    "ro"_a=ENSURE_DOUBLE_ARR(Dro),
                    "y"_a=ENSURE_DOUBLE_ARR(Dy),
                    "u"_a=ENSURE_DOUBLE_ARR(Du)
#if defined(_STARRY_REFLECTED_)
                    , "source"_a=ENSURE_DOUBLE_ARR(Dsource)
#endif
                );
#elif defined(_STARRY_SPECTRAL_)
                auto dy_shape = py::make_tuple(ny, nt, map.ncoly);
                auto dy_reshaped = reshape(ENSURE_DOUBLE_ARR(Dy), dy_shape);
                auto du_shape = py::make_tuple(nu, nt, map.ncolu);
                auto du_reshaped = reshape(ENSURE_DOUBLE_ARR(Du), du_shape);
                py::dict gradient = py::dict(
                    "theta"_a=ENSURE_DOUBLE_ARR(Dtheta),
                    "xo"_a=ENSURE_DOUBLE_ARR(Dxo),
                    "yo"_a=ENSURE_DOUBLE_ARR(Dyo),
                    "ro"_a=ENSURE_DOUBLE_ARR(Dro),
                    "y"_a=dy_reshaped,
                    "u"_a=du_reshaped
#if defined(_STARRY_REFLECTED_)
                    , "source"_a=ENSURE_DOUBLE_ARR(Dsource)
#endif
                );
#elif defined(_STARRY_TEMPORAL_)
                auto Dt = Ref<TSType>(map.cache.pb_Dt);
                auto dy_shape = py::make_tuple(ny, nt, map.ncoly);
                auto dy_reshaped = reshape(ENSURE_DOUBLE_ARR(Dy), dy_shape);
                py::dict gradient = py::dict(
                    "t"_a=ENSURE_DOUBLE_ARR(Dt),
                    "theta"_a=ENSURE_DOUBLE_ARR(Dtheta),
                    "xo"_a=ENSURE_DOUBLE_ARR(Dxo),
                    "yo"_a=ENSURE_DOUBLE_ARR(Dyo),
                    "ro"_a=ENSURE_DOUBLE_ARR(Dro),
                    "y"_a=dy_reshaped,
                    "u"_a=ENSURE_DOUBLE_ARR(Du)
#if defined(_STARRY_REFLECTED_)
                    , "source"_a=ENSURE_DOUBLE_ARR(Dsource)
#endif
                );
#endif

                return py::make_tuple(
                    ENSURE_DOUBLE_ARR(flux), 
                    gradient
                );

            } else {
#if defined(_STARRY_DEFAULT_)
                py::dict gradient = py::dict(
                    "theta"_a=ENSURE_DOUBLE(map.cache.pb_Dtheta(0)),
                    "xo"_a=ENSURE_DOUBLE(map.cache.pb_Dxo(0)),
                    "yo"_a=ENSURE_DOUBLE(map.cache.pb_Dyo(0)),
                    "ro"_a=ENSURE_DOUBLE(map.cache.pb_Dro(0)),
                    "y"_a=ENSURE_DOUBLE_ARR(map.cache.pb_Dy.col(0)),
                    "u"_a=ENSURE_DOUBLE_ARR(map.cache.pb_Du.col(0))
#if defined(_STARRY_REFLECTED_)
                    , "source"_a=ENSURE_DOUBLE_ARR(map.cache.pb_Dsource.col(0))
#endif
                );
#elif defined(_STARRY_SPECTRAL_)
                py::dict gradient = py::dict(
                    "theta"_a=ENSURE_DOUBLE_ARR(map.cache.pb_Dtheta.row(0)),
                    "xo"_a=ENSURE_DOUBLE_ARR(map.cache.pb_Dxo.row(0)),
                    "yo"_a=ENSURE_DOUBLE_ARR(map.cache.pb_Dyo.row(0)),
                    "ro"_a=ENSURE_DOUBLE_ARR(map.cache.pb_Dro.row(0)),
                    "y"_a=ENSURE_DOUBLE_ARR(map.cache.pb_Dy),
                    "u"_a=ENSURE_DOUBLE_ARR(map.cache.pb_Du)
#if defined(_STARRY_REFLECTED_)
                    , "source"_a=ENSURE_DOUBLE_ARR(map.cache.pb_Dsource.col(0))
#endif
                );
#elif defined(_STARRY_TEMPORAL_)
                py::dict gradient = py::dict(
                    "t"_a=ENSURE_DOUBLE(map.cache.pb_Dt(0)),
                    "theta"_a=ENSURE_DOUBLE(map.cache.pb_Dtheta(0)),
                    "xo"_a=ENSURE_DOUBLE(map.cache.pb_Dxo(0)),
                    "yo"_a=ENSURE_DOUBLE(map.cache.pb_Dyo(0)),
                    "ro"_a=ENSURE_DOUBLE(map.cache.pb_Dro(0)),
                    "y"_a=ENSURE_DOUBLE_ARR(map.cache.pb_Dy),
                    "u"_a=ENSURE_DOUBLE_ARR(map.cache.pb_Du.col(0))
#if defined(_STARRY_REFLECTED_)
                    , "source"_a=ENSURE_DOUBLE_ARR(map.cache.pb_Dsource.col(0))
#endif
                );
#endif

                return py::make_tuple(
#ifdef _STARRY_SPECTRAL_
                    ENSURE_DOUBLE_ARR(map.cache.pb_flux.row(0)), 
#else
                    ENSURE_DOUBLE(map.cache.pb_flux(0)), 
#endif
                    gradient
                );
            }

        } else {
            
            // Trivial!
            py::vectorize([&map, 
#if defined(_STARRY_REFLECTED_)
                           &source,
#endif              
                           &n](
#ifdef _STARRY_TEMPORAL_
                double t, 
#endif
                double theta, 
                double xo, 
                double yo, 
                double zo,
                double ro
            ) {
                map.computeFlux(
#ifdef _STARRY_TEMPORAL_
                    static_cast<Scalar>(t), 
#endif
                    static_cast<Scalar>(theta), 
                    static_cast<Scalar>(xo), 
                    static_cast<Scalar>(yo), 
                    static_cast<Scalar>(zo),
                    static_cast<Scalar>(ro), 
#if defined(_STARRY_REFLECTED_)
                    source,
#endif
                    map.cache.pb_flux.row(n)
                );
                ++n;
                return 0;
            })(
#ifdef _STARRY_TEMPORAL_
                t,
#endif
                theta, 
                xo, 
                yo, 
                zo,
                ro
            );
            if (nt > 1) {
                return PYOBJECT_CAST_ARR(map.cache.pb_flux);
            } else {
#ifdef _STARRY_SPECTRAL_
                return py::cast(ENSURE_DOUBLE_ARR(map.cache.pb_flux.row(0)));
#else
                return PYOBJECT_CAST(map.cache.pb_flux(0));
#endif
            }

        }

    };
}

} // namespace interface

#endif
