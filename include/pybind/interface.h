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
#include <starry2/errors.h>
#include <starry2/utils.h>
#include <starry2/maps.h>


#ifdef STARRY_MULTI
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

#define MAXSIZE3(A, B, C) max(max(A.size(), B.size()), C.size())
#define MAXSIZE4(A, B, C, D) max(max(max(A.size(), B.size()), C.size()), D.size())
#define MAXSIZE5(A, B, C, D, E) max(max(max(max(A.size(), B.size()), C.size()), D.size()), E.size())


namespace interface {

//! Misc stuff we need
#include <Python.h>
using namespace pybind11::literals;
using namespace starry2::utils;
using starry2::maps::Map;
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
#ifdef STARRY_TEMPORAL
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
#ifdef STARRY_TEMPORAL
        py::array_t<double>& t, 
#endif
        py::array_t<double>& theta, 
        py::array_t<double>& x, 
        py::array_t<double>& y
    ) -> py::object {
        using Scalar = typename T::Scalar;
#ifdef STARRY_TEMPORAL
        size_t nt = MAXSIZE4(t, theta, x, y);
#else
        size_t nt = MAXSIZE3(theta, x, y);
#endif
        size_t n = 0;
#ifdef STARRY_SPECTRAL
        RowMatrix<Scalar> intensity(nt, map.ncol);
#else
        Vector<Scalar> intensity(nt);
#endif
        py::vectorize([&map, &intensity, &n](
#ifdef STARRY_TEMPORAL
            double t,
#endif
            double theta, 
            double x, 
            double y
        ) {
            map.computeIntensity(
#ifdef STARRY_TEMPORAL
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
#ifdef STARRY_TEMPORAL
            t, 
#endif
            theta, 
            x, 
            y
        );
        if (nt > 1) {
            return py::cast(intensity.template cast<double>());
        } else {
#ifdef STARRY_SPECTRAL
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
#ifdef STARRY_TEMPORAL
        py::array_t<double>&, 
#endif
        py::array_t<double>&, 
        py::array_t<double>&, 
        py::array_t<double>&, 
        py::array_t<double>&,
        bool
    )> flux () 
{
    return []
    (
        Map<T> &map, 
#ifdef STARRY_TEMPORAL
        py::array_t<double>& t, 
#endif
        py::array_t<double>& theta, 
        py::array_t<double>& xo, 
        py::array_t<double>& yo, 
        py::array_t<double>& ro,
        bool compute_gradient
    ) -> py::object 
    {
        using Scalar = typename T::Scalar;
#ifdef STARRY_TEMPORAL
        size_t nt = MAXSIZE5(t, theta, xo, yo, ro); 
#else
        size_t nt = MAXSIZE4(theta, xo, yo, ro);
#endif
        size_t n = 0;

        // Allocate space for the flux
        map.cache.pb_flux.resize(nt, map.nflx);

        if (compute_gradient) {

            // Allocate space for the gradient
            map.cache.pb_theta.resize(nt, map.nflx);
            map.cache.pb_xo.resize(nt, map.nflx);
            map.cache.pb_yo.resize(nt, map.nflx);
            map.cache.pb_ro.resize(nt, map.nflx);

            // The y and u derivs have variable shapes
            int ny, nu;
            if (map.getYDeg_() == 0) {
                ny = 1;
                nu = map.lmax + STARRY_DFDU_DELTA;
            } else if (map.getUDeg_() == 0) {
                ny = map.N;
                nu = 1;
            } else {
                ny = map.N;
                nu = map.lmax + STARRY_DFDU_DELTA;
            } 

#if defined(STARRY_DEFAULT)
            map.cache.pb_y.resize(nt, ny);
            map.cache.pb_u.resize(nt, nu);
#elif defined(STARRY_SPECTRAL)
            map.cache.pb_y.resize(nt * ny, map.ncol);
            map.cache.pb_u.resize(nt * nu, map.ncol);
#elif defined(STARRY_TEMPORAL)
            map.cache.pb_time.resize(nt, map.nflx);
            map.cache.pb_y.resize(nt * ny, map.ncol);
            map.cache.pb_u.resize(nt, nu);
#endif

            // Vectorize the computation
            py::vectorize([&map, &n, &ny, &nu](
#ifdef STARRY_TEMPORAL
                double t, 
#endif
                double theta, 
                double xo, 
                double yo, 
                double ro
            ) {
                map.computeFlux(
#ifdef STARRY_TEMPORAL
                    static_cast<Scalar>(t),
#endif
                    static_cast<Scalar>(theta),
                    static_cast<Scalar>(xo),
                    static_cast<Scalar>(yo),
                    static_cast<Scalar>(ro),
                    map.cache.pb_flux.row(n),
#ifdef STARRY_TEMPORAL
                    map.cache.pb_time.row(n),
#endif
                    map.cache.pb_theta.row(n),
                    map.cache.pb_xo.row(n),
                    map.cache.pb_yo.row(n),
                    map.cache.pb_ro.row(n),
#if defined(STARRY_DEFAULT)
                    map.cache.pb_y.row(n).transpose(),
                    map.cache.pb_u.row(n).transpose()
#elif defined(STARRY_SPECTRAL)
                    map.cache.pb_y.block(n * ny, 0, ny, map.ncol),
                    map.cache.pb_u.block(n * nu, 0, nu, map.ncol)
#elif defined(STARRY_TEMPORAL)
                    map.cache.pb_y.row(n), // TODO
                    map.cache.pb_u.row(n).transpose()
#endif
                );
                ++n;
                return 0;
            })(
#ifdef STARRY_TEMPORAL
                t,
#endif
                theta, 
                xo, 
                yo, 
                ro
            );

            // Construct the gradient dictionary and
            // return a tuple of (flux, gradient)
            if (nt > 1) {
                
#if defined(STARRY_DEFAULT)
                auto flux = Ref<Vector<Scalar>>(map.cache.pb_flux);
                auto dtheta = Ref<Vector<Scalar>>(map.cache.pb_theta);
                auto dxo = Ref<Vector<Scalar>>(map.cache.pb_xo);
                auto dyo = Ref<Vector<Scalar>>(map.cache.pb_yo);
                auto dro = Ref<Vector<Scalar>>(map.cache.pb_ro);
                auto dy = Ref<RowMatrix<Scalar>>(map.cache.pb_y);
                auto du = Ref<RowMatrix<Scalar>>(map.cache.pb_u);
                py::dict gradient = py::dict(
                    "theta"_a=ENSURE_DOUBLE_ARR(dtheta),
                    "xo"_a=ENSURE_DOUBLE_ARR(dxo),
                    "yo"_a=ENSURE_DOUBLE_ARR(dyo),
                    "ro"_a=ENSURE_DOUBLE_ARR(dro),
                    "y"_a=ENSURE_DOUBLE_ARR(dy),
                    "u"_a=ENSURE_DOUBLE_ARR(du)
                );
#elif defined(STARRY_SPECTRAL)
                // TODO
                auto flux = Ref<RowMatrix<Scalar>>(map.cache.pb_flux);
                auto dtheta = Ref<RowMatrix<Scalar>>(map.cache.pb_theta);
                py::dict gradient = py::dict(
                    "theta"_a=ENSURE_DOUBLE_ARR(dtheta)
                );
#elif defined(STARRY_TEMPORAL)
                auto flux = Ref<Vector<Scalar>>(map.cache.pb_flux);
                auto dt = Ref<Vector<Scalar>>(map.cache.pb_time);
                auto dtheta = Ref<Vector<Scalar>>(map.cache.pb_theta);
                auto dxo = Ref<Vector<Scalar>>(map.cache.pb_xo);
                auto dyo = Ref<Vector<Scalar>>(map.cache.pb_yo);
                auto dro = Ref<Vector<Scalar>>(map.cache.pb_ro);
                auto dy = Ref<RowMatrix<Scalar>>(map.cache.pb_y);
                auto du = Ref<RowMatrix<Scalar>>(map.cache.pb_u);
                py::dict gradient = py::dict(
                    "t"_a=ENSURE_DOUBLE_ARR(dt),
                    "theta"_a=ENSURE_DOUBLE_ARR(dtheta),
                    "xo"_a=ENSURE_DOUBLE_ARR(dxo),
                    "yo"_a=ENSURE_DOUBLE_ARR(dyo),
                    "ro"_a=ENSURE_DOUBLE_ARR(dro),
                    "y"_a=ENSURE_DOUBLE_ARR(dy), // TODO
                    "u"_a=ENSURE_DOUBLE_ARR(du)
                );
#endif

                return py::make_tuple(
                    ENSURE_DOUBLE_ARR(flux), 
                    gradient
                );

            } else {
#if defined(STARRY_DEFAULT)
                py::dict gradient = py::dict(
                    "theta"_a=ENSURE_DOUBLE(map.cache.pb_theta(0)),
                    "xo"_a=ENSURE_DOUBLE(map.cache.pb_xo(0)),
                    "yo"_a=ENSURE_DOUBLE(map.cache.pb_yo(0)),
                    "ro"_a=ENSURE_DOUBLE(map.cache.pb_ro(0)),
                    "y"_a=ENSURE_DOUBLE_ARR(map.cache.pb_y.row(0)),
                    "u"_a=ENSURE_DOUBLE_ARR(map.cache.pb_u.row(0))
                );
#elif defined(STARRY_SPECTRAL)
                // TODO
                py::dict gradient = py::dict(
                    "theta"_a=ENSURE_DOUBLE(map.cache.pb_theta.row(0))
                );
#elif defined(STARRY_TEMPORAL)
                py::dict gradient = py::dict(
                    "t"_a=ENSURE_DOUBLE_ARR(map.cache.pb_time(0)),
                    "theta"_a=ENSURE_DOUBLE(map.cache.pb_theta(0)),
                    "xo"_a=ENSURE_DOUBLE(map.cache.pb_xo(0)),
                    "yo"_a=ENSURE_DOUBLE(map.cache.pb_yo(0)),
                    "ro"_a=ENSURE_DOUBLE(map.cache.pb_ro(0)),
                    "y"_a=ENSURE_DOUBLE_ARR(map.cache.pb_y.row(0)), // TODO
                    "u"_a=ENSURE_DOUBLE_ARR(map.cache.pb_u.row(0))
                );
#endif

                return py::make_tuple(
#ifdef STARRY_SPECTRAL
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
#ifdef STARRY_TEMPORAL
                double t, 
#endif
                double theta, 
                double xo, 
                double yo, 
                double ro
            ) {
                map.computeFlux(
#ifdef STARRY_TEMPORAL
                    static_cast<Scalar>(t), 
#endif
                    static_cast<Scalar>(theta), 
                    static_cast<Scalar>(xo), 
                    static_cast<Scalar>(yo), 
                    static_cast<Scalar>(ro), 
                    map.cache.pb_flux.row(n)
                );
                ++n;
                return 0;
            })(
#ifdef STARRY_TEMPORAL
                t,
#endif
                theta, 
                xo, 
                yo, 
                ro
            );
            if (nt > 1) {
                return PYOBJECT_CAST_ARR(map.cache.pb_flux);
            } else {
#ifdef STARRY_SPECTRAL
                return PYOBJECT_CAST(ENSURE_DOUBLE_ARR(map.cache.pb_flux.row(0)));
#else
                return PYOBJECT_CAST(map.cache.pb_flux(0));
#endif
            }

        }

    };
}

} // namespace interface

#endif
