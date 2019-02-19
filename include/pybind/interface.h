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
#ifdef _STARRY_REFLECTED_
        , py::array_t<double>& 
#endif
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
#ifdef _STARRY_REFLECTED_
        , py::array_t<double>& source_
#endif
    ) -> py::object {
        using Scalar = typename T::Scalar;
#ifdef _STARRY_REFLECTED_
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
#endif
#ifdef _STARRY_TEMPORAL_
#  ifdef _STARRY_EMITTED_
        std::vector<long> v{t.size(), theta.size(), x.size(), y.size()};
#  else
        std::vector<long> v{t.size(), theta.size(), x.size(), y.size(), sx.size()};
#  endif
#else
#  ifdef _STARRY_EMITTED_
        std::vector<long> v{theta.size(), x.size(), y.size()};
#  else
        std::vector<long> v{theta.size(), x.size(), y.size(), sx.size()};
#  endif
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
#ifdef _STARRY_REFLECTED_
            , double sx, 
            double sy, 
            double sz
#endif
        ) {
#ifdef _STARRY_REFLECTED_
            UnitVector<Scalar> source(3);
            source << sx, sy, sz;
#endif
            map.computeIntensity(
#ifdef _STARRY_TEMPORAL_
                static_cast<Scalar>(t), 
#endif
                static_cast<Scalar>(theta), 
                static_cast<Scalar>(x), 
                static_cast<Scalar>(y), 
#ifdef _STARRY_REFLECTED_
                source,
#endif
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
#ifdef _STARRY_REFLECTED_
            , sx,
            sy,
            sz
#endif
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
#ifdef _STARRY_REFLECTED_
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
#ifdef _STARRY_REFLECTED_
        py::array_t<double>& source_, 
#endif
        bool compute_gradient
    ) -> py::object 
    {
        using Scalar = typename T::Scalar;
        using TSType = typename T::TSType;

#ifdef _STARRY_REFLECTED_
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
#endif

#if defined(_STARRY_SPECTRAL_) || defined(_STARRY_TEMPORAL_)
        // We need our old friend numpy to reshape
        // matrices into 3-tensors on the Python side
        auto numpy = py::module::import("numpy");
        auto reshape = numpy.attr("reshape");
#endif

#ifdef _STARRY_TEMPORAL_
#  ifdef _STARRY_EMITTED_
        std::vector<long> v{t.size(), theta.size(), xo.size(), yo.size(), zo.size(), ro.size()};
#  else
        std::vector<long> v{t.size(), theta.size(), xo.size(), yo.size(), zo.size(), ro.size(), sx.size()};
#  endif
#else
#  ifdef _STARRY_EMITTED_
        std::vector<long> v{theta.size(), xo.size(), yo.size(), zo.size(), ro.size()};
#  else
        std::vector<long> v{theta.size(), xo.size(), yo.size(), zo.size(), ro.size(), sx.size()};
#  endif
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
            py::vectorize([&map, &n, &ny, &nu](

#ifdef _STARRY_TEMPORAL_
                double t, 
#endif
                double theta, 
                double xo, 
                double yo, 
                double zo,
                double ro
#ifdef _STARRY_REFLECTED_
                , double sx,
                double sy,
                double sz 
#endif
            ) {
#ifdef _STARRY_REFLECTED_
                    UnitVector<Scalar> source(3);
                    source << sx, sy, sz;
#endif
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
#ifdef _STARRY_REFLECTED_
                , sx,
                sy,
                sz
#endif
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
            py::vectorize([&map, &n](
#ifdef _STARRY_TEMPORAL_
                double t, 
#endif
                double theta, 
                double xo, 
                double yo, 
                double zo,
                double ro
#if defined(_STARRY_REFLECTED_)
                , double sx,
                double sy,
                double sz
#endif              
            ) {
#if defined(_STARRY_REFLECTED_)
                UnitVector<Scalar> source(3);
                source << sx, sy, sz;
#endif
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
#ifdef _STARRY_REFLECTED_
                , sx,
                sy,
                sz
#endif
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

/**
Return a lambda function to compute the linear model at a point 
or a vector of points. Optionally compute and return 
the gradient.

\todo Implement this for temporal & all reflected types

*/
template <typename T>
std::function<py::object (
        Map<T> &, 
        py::array_t<double>&, 
        py::array_t<double>&, 
        py::array_t<double>&, 
        py::array_t<double>&,
        py::array_t<double>&,
        bool
    )> linear_model () 
{
    return []
    (
        Map<T> &map, 
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
        std::vector<long> v{theta_.request().size,
                            xo_.request().size,
                            yo_.request().size,
                            zo_.request().size,
                            ro_.request().size};
        py::ssize_t nt = *std::max_element(v.begin(), v.end());
        py::buffer_info buf;
        double *ptr;

        // Map `theta` to an Eigen vector
        Eigen::Map<Vector<Scalar>> theta(NULL, nt, 1);
        Vector<Scalar> tmp_theta;
        buf = theta_.request();
        ptr = (double *) buf.ptr;
        if (buf.ndim == 0) {
            tmp_theta = ptr[0] * Vector<Scalar>::Ones(nt);
            new (&theta) Eigen::Map<Vector<Scalar>>(&tmp_theta(0), nt, 1);
        } else if ((buf.ndim == 1) && (buf.size == nt)) {
#ifdef _STARRY_DOUBLE_
            new (&theta) Eigen::Map<Vector<Scalar>>(ptr, nt, 1);
#else
            // We need to make a copy
            tmp_theta = (py::cast<Vector<double>>(theta_)).template cast<Scalar>();
            new (&theta) Eigen::Map<Vector<Scalar>>(&tmp_theta(0), nt, 1);
#endif
        } else {
            throw errors::ShapeError("Vector `theta` has the incorrect shape.");
        }

        // Map `xo` to an Eigen vector
        Eigen::Map<Vector<Scalar>> xo(NULL, nt, 1);
        Vector<Scalar> tmp_xo;
        buf = xo_.request();
        ptr = (double *) buf.ptr;
        if (buf.ndim == 0) {
            tmp_xo = ptr[0] * Vector<Scalar>::Ones(nt);
            new (&xo) Eigen::Map<Vector<Scalar>>(&tmp_xo(0), nt, 1);
        } else if ((buf.ndim == 1) && (buf.size == nt)) {
#ifdef _STARRY_DOUBLE_
            new (&xo) Eigen::Map<Vector<Scalar>>(ptr, nt, 1);
#else
            // We need to make a copy
            tmp_xo = (py::cast<Vector<double>>(xo_)).template cast<Scalar>();
            new (&xo) Eigen::Map<Vector<Scalar>>(&tmp_xo(0), nt, 1);
#endif
        } else {
            throw errors::ShapeError("Vector `xo` has the incorrect shape.");
        }

        // Map `yo` to an Eigen vector
        Eigen::Map<Vector<Scalar>> yo(NULL, nt, 1);
        Vector<Scalar> tmp_yo;
        buf = yo_.request();
        ptr = (double *) buf.ptr;
        if (buf.ndim == 0) {
            tmp_yo = ptr[0] * Vector<Scalar>::Ones(nt);
            new (&yo) Eigen::Map<Vector<Scalar>>(&tmp_yo(0), nt, 1);
        } else if ((buf.ndim == 1) && (buf.size == nt)) {
#ifdef _STARRY_DOUBLE_
            new (&yo) Eigen::Map<Vector<Scalar>>(ptr, nt, 1);
#else
            // We need to make a copy
            tmp_yo = (py::cast<Vector<double>>(yo_)).template cast<Scalar>();
            new (&yo) Eigen::Map<Vector<Scalar>>(&tmp_yo(0), nt, 1);
#endif
        } else {
            throw errors::ShapeError("Vector `yo` has the incorrect shape.");
        }

        // Map `zo` to an Eigen vector
        Eigen::Map<Vector<Scalar>> zo(NULL, nt, 1);
        Vector<Scalar> tmp_zo;
        buf = zo_.request();
        ptr = (double *) buf.ptr;
        if (buf.ndim == 0) {
            tmp_zo = ptr[0] * Vector<Scalar>::Ones(nt);
            new (&zo) Eigen::Map<Vector<Scalar>>(&tmp_zo(0), nt, 1);
        } else if ((buf.ndim == 1) && (buf.size == nt)) {
#ifdef _STARRY_DOUBLE_
            new (&zo) Eigen::Map<Vector<Scalar>>(ptr, nt, 1);
#else
            // We need to make a copy
            tmp_zo = (py::cast<Vector<double>>(zo_)).template cast<Scalar>();
            new (&zo) Eigen::Map<Vector<Scalar>>(&tmp_zo(0), nt, 1);
#endif
        } else {
            throw errors::ShapeError("Vector `zo` has the incorrect shape.");
        }

        // Map `ro` to an Eigen vector
        Eigen::Map<Vector<Scalar>> ro(NULL, nt, 1);
        Vector<Scalar> tmp_ro;
        buf = ro_.request();
        ptr = (double *) buf.ptr;
        if (buf.ndim == 0) {
            tmp_ro = ptr[0] * Vector<Scalar>::Ones(nt);
            new (&ro) Eigen::Map<Vector<Scalar>>(&tmp_ro(0), nt, 1);
        } else if ((buf.ndim == 1) && (buf.size == nt)) {
#ifdef _STARRY_DOUBLE_
            new (&ro) Eigen::Map<Vector<Scalar>>(ptr, nt, 1);
#else
            // We need to make a copy
            tmp_ro = (py::cast<Vector<double>>(ro_)).template cast<Scalar>();
            new (&ro) Eigen::Map<Vector<Scalar>>(&tmp_ro(0), nt, 1);
#endif
        } else {
            throw errors::ShapeError("Vector `ro` has the incorrect shape.");
        }

        // The linear model matrix
        RowMatrix<Scalar> A;

        if (!compute_gradient) {

            // Compute and return
            map.computeLinearModel(theta, xo, yo, zo, ro, A);
            return PYOBJECT_CAST_ARR(A);

        } else {

            // Allocate space for the gradient
            map.cache.pb_DADtheta.resize(nt, map.N);
            map.cache.pb_DADxo.resize(nt, map.N);
            map.cache.pb_DADyo.resize(nt, map.N);
            map.cache.pb_DADro.resize(nt, map.N);

            // Compute the model + gradient
            map.computeLinearModel(
                theta, xo, yo, zo, ro, A, 
                map.cache.pb_DADtheta, map.cache.pb_DADxo, 
                map.cache.pb_DADyo, map.cache.pb_DADro
            );

            // Get Eigen references to the arrays, as these
            // are automatically passed by ref to the Python side
            auto Dtheta = Ref<RowMatrix<Scalar>>(map.cache.pb_DADtheta);
            auto Dxo = Ref<RowMatrix<Scalar>>(map.cache.pb_DADxo);
            auto Dyo = Ref<RowMatrix<Scalar>>(map.cache.pb_DADyo);
            auto Dro = Ref<RowMatrix<Scalar>>(map.cache.pb_DADro);

            // Construct a dictionary
            py::dict gradient = py::dict(
                "theta"_a=ENSURE_DOUBLE_ARR(Dtheta),
                "xo"_a=ENSURE_DOUBLE_ARR(Dxo),
                "yo"_a=ENSURE_DOUBLE_ARR(Dyo),
                "ro"_a=ENSURE_DOUBLE_ARR(Dro)
            );

            // Return
            return py::make_tuple(
                ENSURE_DOUBLE_ARR(A), 
                gradient
            );

        }

    };
}

} // namespace interface

#endif
