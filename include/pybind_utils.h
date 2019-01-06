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
#include "utils.h"
#include "maps.h"

namespace pybind_utils {

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
or a vector of points for a static, single-wavelength map.

*/
template <typename T, IsDefault<T>* = nullptr>
std::function<py::object(
        Map<T> &, 
        py::array_t<double>&, 
        py::array_t<double>&, 
        py::array_t<double>&
    )> intensity () 
{
    return []
    (
        Map<T> &map, 
        py::array_t<double>& theta, 
        py::array_t<double>& x, 
        py::array_t<double>& y
    ) -> py::object {
        using Scalar = typename T::Scalar;
        size_t nt = max(max(theta.size(), x.size()), y.size());
        size_t n = 0;
        Vector<Scalar> intensity(nt);

        py::vectorize([&map, &intensity, &n](
            double theta, 
            double x, 
            double y
        ) {
            map.computeIntensity(static_cast<Scalar>(theta), 
                                 static_cast<Scalar>(x), 
                                 static_cast<Scalar>(y), 
                                 intensity.row(n));
            ++n;
            return 0;
        })(theta, x, y);
        if (nt > 1)
            return py::cast(intensity.template cast<double>());
        else
            return py::cast(static_cast<double>(intensity(0)));
    };
}

/**
Return a lambda function to compute the intensity at a point 
or a vector of points for a spectral map.

*/
template <typename T, IsSpectral<T>* = nullptr>
std::function<py::object(
        Map<T> &, 
        py::array_t<double>&, 
        py::array_t<double>&, 
        py::array_t<double>&
    )> intensity () 
{
    return []
    (
        Map<T> &map, 
        py::array_t<double>& theta, 
        py::array_t<double>& x, 
        py::array_t<double>& y
    ) -> py::object {
        using Scalar = typename T::Scalar;
        size_t nt = max(max(theta.size(), x.size()), y.size());
        size_t n = 0;
        RowMatrix<Scalar> intensity(nt, map.ncol);

        py::vectorize([&map, &intensity, &n](
            double theta, 
            double x, 
            double y
        ) {
            map.computeIntensity(static_cast<Scalar>(theta), 
                                 static_cast<Scalar>(x), 
                                 static_cast<Scalar>(y), 
                                 intensity.row(n));
            ++n;
            return 0;
        })(theta, x, y);
        if (nt > 1)
            return py::cast(intensity.template cast<double>());
        else {
            RowVector<double> f = intensity.row(0).template cast<double>();
            return py::cast(f);
        }
    };
}

/**
Return a lambda function to compute the intensity at a point 
or a vector of points for a temporal map.

*/
template <typename T, IsTemporal<T>* = nullptr>
std::function<py::object(
        Map<T> &, 
        py::array_t<double>&, 
        py::array_t<double>&, 
        py::array_t<double>&,
        py::array_t<double>&
    )> intensity () 
{
    return []
    (
        Map<T> &map, 
        py::array_t<double>& t,
        py::array_t<double>& theta, 
        py::array_t<double>& x, 
        py::array_t<double>& y
    ) -> py::object {
        using Scalar = typename T::Scalar;
        size_t nt = max(max(max(theta.size(), x.size()), y.size()), t.size());
        size_t n = 0;
        Vector<Scalar> intensity(nt);

        py::vectorize([&map, &intensity, &n](
            double t,
            double theta, 
            double x, 
            double y
        ) {
            map.computeIntensity(static_cast<Scalar>(t), 
                                 static_cast<Scalar>(theta), 
                                 static_cast<Scalar>(x), 
                                 static_cast<Scalar>(y), 
                                 intensity.row(n));
            ++n;
            return 0;
        })(t, theta, x, y);
        if (nt > 1)
            return py::cast(intensity.template cast<double>());
        else
            return py::cast(static_cast<double>(intensity(0)));
    };
}


/**
Return a lambda function to compute the flux at a point 
or a vector of points for a static, single-wavelength map. Optionally
compute and return the gradient.

*/
template <typename T, IsDefault<T>* = nullptr>
std::function<py::object(
        Map<T> &, 
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
        py::array_t<double>& theta, 
        py::array_t<double>& xo, 
        py::array_t<double>& yo, 
        py::array_t<double>& ro,
        bool compute_gradient
    ) -> py::object 
    {
        using Scalar = typename T::Scalar;
        size_t nt = max(max(max(theta.size(), xo.size()), 
                            yo.size()), ro.size());
        size_t n = 0;

        // Allocate the arrays
        map.cache.pb_flux.resize(nt);
        map.cache.pb_theta.resize(nt);
        map.cache.pb_xo.resize(nt);
        map.cache.pb_yo.resize(nt);
        map.cache.pb_ro.resize(nt);
        if (map.getYDeg_() == 0) {
            map.cache.pb_y.resize(nt, 1);
            map.cache.pb_u.resize(nt, map.lmax);
        } else if (map.getUDeg_() == 0) {
            map.cache.pb_y.resize(nt, map.N);
            map.cache.pb_u.resize(nt, 1);
        } else {
            map.cache.pb_y.resize(nt, map.N);
            map.cache.pb_u.resize(nt, map.lmax);
        }

        if (compute_gradient) {

            // Vectorize the computation
            py::vectorize([&map, &n](
                double theta, 
                double xo, 
                double yo, 
                double ro
            ) {
                map.computeFlux(
                    static_cast<Scalar>(theta),
                    static_cast<Scalar>(xo),
                    static_cast<Scalar>(yo),
                    static_cast<Scalar>(ro),
                    map.cache.pb_flux.row(n),
                    map.cache.pb_theta.row(n),
                    map.cache.pb_xo.row(n),
                    map.cache.pb_yo.row(n),
                    map.cache.pb_ro.row(n),
                    map.cache.pb_y.row(n),
                    map.cache.pb_u.row(n).transpose()
                );
                ++n;
                return 0;
            })(theta, xo, yo, ro);

            // Construct the gradient dictionary and
            // return a tuple of (flux, gradient)
            if (nt > 1) {
                py::dict gradient_dict = py::dict(
                    "theta"_a=map.cache.pb_theta.template cast<double>(),
                    "xo"_a=map.cache.pb_xo.template cast<double>(),
                    "yo"_a=map.cache.pb_yo.template cast<double>(),
                    "ro"_a=map.cache.pb_ro.template cast<double>(),
                    "y"_a=map.cache.pb_y.template cast<double>(),
                    "u"_a=map.cache.pb_u.template cast<double>()
                );
                return py::make_tuple(
                    map.cache.pb_flux.template cast<double>(), 
                    gradient_dict
                );
            } else {
                py::dict gradient_dict = py::dict(
                    "theta"_a=static_cast<double>(map.cache.pb_theta(0)),
                    "xo"_a=static_cast<double>(map.cache.pb_xo(0)),
                    "yo"_a=static_cast<double>(map.cache.pb_yo(0)),
                    "ro"_a=static_cast<double>(map.cache.pb_ro(0)),
                    "y"_a=map.cache.pb_y.row(0).template cast<double>(),
                    "u"_a=map.cache.pb_u.row(0).template cast<double>()
                );
                return py::make_tuple(
                    static_cast<double>(map.cache.pb_flux(0)), 
                    gradient_dict
                );
            }

        } else {
            
            // Trivial!
            py::vectorize([&map, &n](
                double theta, 
                double xo, 
                double yo, 
                double ro
            ) {
                map.computeFlux(
                    static_cast<Scalar>(theta), 
                    static_cast<Scalar>(xo), 
                    static_cast<Scalar>(yo), 
                    static_cast<Scalar>(ro), 
                    map.cache.pb_flux.row(n)
                );
                ++n;
                return 0;
            })(theta, xo, yo, ro);
            if (nt > 1)
                return py::cast(map.cache.pb_flux.template cast<double>());
            else
                return py::cast(static_cast<double>(map.cache.pb_flux(0)));

        }

    };
}

/**
Return a lambda function to compute the flux at a point 
or a vector of points for a spectral map. Optionally
compute and return the gradient.

*/
template <typename T, IsSpectral<T>* = nullptr>
std::function<py::object(
        Map<T> &, 
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
        py::array_t<double>& theta, 
        py::array_t<double>& xo, 
        py::array_t<double>& yo, 
        py::array_t<double>& ro,
        bool compute_gradient
    ) -> py::object {
        using Scalar = typename T::Scalar;
        size_t nt = max(max(max(theta.size(), xo.size()), yo.size()), 
                        ro.size());
        size_t n = 0;
        RowMatrix<Scalar> flux(nt, map.ncol);

        // Numpy hacks
        //auto numpy = py::module::import("numpy");
        //auto swapaxes = numpy.attr("swapaxes");
        //auto reshape = numpy.attr("reshape");

        if (compute_gradient) {
            
            // Allocate storage for the gradient
            // TODO: use cache for each deriv

            // Vectorize the computation
            py::vectorize([&map, &flux, &n](
                double theta, 
                double xo, 
                double yo, 
                double ro
            ) {
                // TODO
                // map.computeFlux(...)
                ++n;
                return 0;
            })(theta, xo, yo, ro);

            // Construct the gradient dictionary and
            // return a tuple of (flux, gradient)
            if (nt > 1) {
                py::dict gradient_dict = py::dict(
                    // TODO
                    "theta"_a=0,
                    "xo"_a=0,
                    "yo"_a=0,
                    "ro"_a=0,
                    "y"_a=0,
                    "u"_a=0
                );
                return py::make_tuple(flux.template cast<double>(), 
                                      gradient_dict);
            } else {
                py::dict gradient_dict = py::dict(
                    // TODO
                    "theta"_a=0,
                    "xo"_a=0,
                    "yo"_a=0,
                    "ro"_a=0,
                    "y"_a=0,
                    "u"_a=0
                );
                return py::make_tuple(static_cast<double>(flux(0)), 
                                      gradient_dict);
            }

        } else {

            // Trivial!
            py::vectorize([&map, &flux, &n](
                double theta, 
                double xo, 
                double yo, 
                double ro
            ) {
                map.computeFlux(static_cast<Scalar>(theta), 
                                static_cast<Scalar>(xo), 
                                static_cast<Scalar>(yo), 
                                static_cast<Scalar>(ro), 
                                flux.row(n));
                ++n;
                return 0;
            })(theta, xo, yo, ro);
            if (nt > 1)
                return py::cast(flux.template cast<double>());
            else {
                RowVector<double> f = flux.row(0).template cast<double>();
                return py::cast(f);
            }
        }

    };
}

/**
Return a lambda function to compute the flux at a point 
or a vector of points for a temporal map. Optionally
compute and return the gradient.

*/
template <typename T, IsTemporal<T>* = nullptr>
std::function<py::object(
        Map<T> &, 
        py::array_t<double>&, 
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
        py::array_t<double>& t,
        py::array_t<double>& theta, 
        py::array_t<double>& xo, 
        py::array_t<double>& yo, 
        py::array_t<double>& ro, 
        bool compute_gradient
    ) -> py::object 
    {
        using Scalar = typename T::Scalar;
        size_t nt = max(max(max(max(theta.size(), xo.size()), 
                            yo.size()), ro.size()), t.size());
        size_t n = 0;
        Vector<Scalar> flux(nt);

        if (compute_gradient) {

            // Allocate storage for the gradient
            // TODO: use cache for each deriv

            // Vectorize the computation
            py::vectorize([&map, &flux, &n](
                double t,
                double theta, 
                double xo, 
                double yo, 
                double ro
            ) {
                // TODO
                // map.computeFlux(...)
                ++n;
                return 0;
            })(t, theta, xo, yo, ro);

            // Construct the gradient dictionary and
            // return a tuple of (flux, gradient)
            if (nt > 1) {
                py::dict gradient_dict = py::dict(
                    // TODO
                    "t"_a=0,
                    "theta"_a=0,
                    "xo"_a=0,
                    "yo"_a=0,
                    "ro"_a=0,
                    "y"_a=0,
                    "u"_a=0
                );
                return py::make_tuple(flux.template cast<double>(), 
                                      gradient_dict);
            } else {
                py::dict gradient_dict = py::dict(
                    // TODO
                    "t"_a=0,
                    "theta"_a=0,
                    "xo"_a=0,
                    "yo"_a=0,
                    "ro"_a=0,
                    "y"_a=0,
                    "u"_a=0
                );
                return py::make_tuple(static_cast<double>(flux(0)), 
                                      gradient_dict);
            }

        } else {
            
            // Trivial!
            py::vectorize([&map, &flux, &n](
                double t,
                double theta, 
                double xo, 
                double yo, 
                double ro
            ) {
                map.computeFlux(static_cast<Scalar>(t), 
                                static_cast<Scalar>(theta), 
                                static_cast<Scalar>(xo), 
                                static_cast<Scalar>(yo), 
                                static_cast<Scalar>(ro), 
                                flux.row(n));
                ++n;
                return 0;
            })(t, theta, xo, yo, ro);
            if (nt > 1)
                return py::cast(flux.template cast<double>());
            else
                return py::cast(static_cast<double>(flux(0)));

        }

    };
}

} // namespace pybind_utils

#endif
