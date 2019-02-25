/**
\file interface.cpp
\brief Defines the entry point for the C++ API.

*/

// Enable debug mode?
#ifdef STARRY_DEBUG
#   undef NDEBUG
#endif

// Enable the Python interface
#ifndef STARRY_ENABLE_PYTHON_INTERFACE
#   define STARRY_ENABLE_PYTHON_INTERFACE
#endif

// Select which module to build
#if defined(_STARRY_DEFAULT_DOUBLE_) || defined(_STARRY_DEFAULT_REFL_DOUBLE_)
#   define _STARRY_DEFAULT_
#   define _STARRY_DOUBLE_
#   define _STARRY_STATIC_
#   define _STARRY_SINGLECOL_
#   if defined(_STARRY_DEFAULT_DOUBLE_)
#       define _STARRY_NAME_ _starry_default_double
#       define _STARRY_TYPE_ Default<double, false>
#       define _STARRY_EMITTED_
#   else
#       define _STARRY_NAME_ _starry_default_refl_double
#       define _STARRY_TYPE_ Default<double, true>
#       define _STARRY_REFLECTED_
#   endif
#elif defined(_STARRY_DEFAULT_MULTI_) || defined(_STARRY_DEFAULT_REFL_MULTI_)
#   define _STARRY_DEFAULT_
#   define _STARRY_MULTI_
#   define _STARRY_STATIC_
#   define _STARRY_SINGLECOL_
#   define STARRY_ENABLE_BOOST
#   if defined(_STARRY_DEFAULT_MULTI_)
#       define _STARRY_NAME_ _starry_default_multi
#       define _STARRY_TYPE_ Default<Multi, false>
#       define _STARRY_EMITTED_
#   else
#       define _STARRY_NAME_ _starry_default_refl_multi
#       define _STARRY_TYPE_ Default<Multi, true>
#       define _STARRY_REFLECTED_
#   endif
#elif defined(_STARRY_SPECTRAL_DOUBLE_) || defined(_STARRY_SPECTRAL_REFL_DOUBLE_)
#   define _STARRY_SPECTRAL_
#   define _STARRY_DOUBLE_
#   define _STARRY_STATIC_
#   define _STARRY_MULTI_COL
#   if defined(_STARRY_SPECTRAL_DOUBLE_)
#       define _STARRY_NAME_ _starry_spectral_double
#       define _STARRY_TYPE_ Spectral<double, false>
#       define _STARRY_EMITTED_
#   else
#       define _STARRY_NAME_ _starry_spectral_refl_double
#       define _STARRY_TYPE_ Spectral<double, true>
#       define _STARRY_REFLECTED_
#   endif
#elif defined(_STARRY_SPECTRAL_MULTI_) || defined(_STARRY_SPECTRAL_REFL_MULTI_)
#   define _STARRY_SPECTRAL_
#   define _STARRY_MULTI_
#   define _STARRY_STATIC_
#   define _STARRY_MULTI_COL
#   define STARRY_ENABLE_BOOST
#   if defined(_STARRY_SPECTRAL_MULTI_)
#       define _STARRY_NAME_ _starry_spectral_multi
#       define _STARRY_TYPE_ Spectral<Multi, false>
#       define _STARRY_EMITTED_
#   else
#       define _STARRY_NAME_ _starry_spectral_refl_multi
#       define _STARRY_TYPE_ Spectral<Multi, true>
#       define _STARRY_REFLECTED_
#   endif
#elif defined(_STARRY_TEMPORAL_DOUBLE_) || defined(_STARRY_TEMPORAL_REFL_DOUBLE_)
#   define _STARRY_TEMPORAL_
#   define _STARRY_DOUBLE_
#   define _STARRY_MULTI_COL
#   if defined(_STARRY_TEMPORAL_DOUBLE_)
#       define _STARRY_NAME_ _starry_temporal_double
#       define _STARRY_TYPE_ Temporal<double, false>
#       define _STARRY_EMITTED_
#   else
#       define _STARRY_NAME_ _starry_temporal_refl_double
#       define _STARRY_TYPE_ Temporal<double, true>
#       define _STARRY_REFLECTED_
#   endif
#elif defined(_STARRY_TEMPORAL_MULTI_) || defined(_STARRY_TEMPORAL_REFL_MULTI_)
#   define _STARRY_TEMPORAL_
#   define _STARRY_MULTI_
#   define _STARRY_MULTI_COL
#   define STARRY_ENABLE_BOOST
#   if defined(_STARRY_TEMPORAL_MULTI_)
#       define _STARRY_NAME_ _starry_temporal_multi
#       define _STARRY_TYPE_ Temporal<Multi, false>
#       define _STARRY_EMITTED_
#   else
#       define _STARRY_NAME_ _starry_temporal_refl_multi
#       define _STARRY_TYPE_ Temporal<Multi, true>
#       define _STARRY_REFLECTED_
#endif
#else
    static_assert(false, "Invalid or missing `starry` module type.");
#endif

// Includes
#include <pybind11/embed.h>
#include "interface.h"
#include "docstrings.h"
using namespace interface;

// Register the Python module
PYBIND11_MODULE(
    _STARRY_NAME_, 
    m
) {

    // Module docs
    py::options options;
    options.disable_function_signatures();
    m.doc() = docstrings::starry::doc;

    // Current Map Type
    using T = _STARRY_TYPE_;

    // Declare the Map class
    py::class_<Map<T>> PyMap(m, "Map", docstrings::Map::doc);

    // Constructor
#   if defined(_STARRY_SINGLECOL_) 
        PyMap.def(py::init<int, int>(), "ydeg"_a=2, "udeg"_a=2);
#   else
        PyMap.def(py::init<int, int, int>(), "ydeg"_a=2, "udeg"_a=2, "ncol"_a=1);
#   endif

    // String representation of the map
    PyMap.def("__repr__", &Map<T>::info);

    // Number of Ylm map columns
    PyMap.def_property_readonly(
        "ncoly", [] (
            Map<T> &map
        ) {
            return map.ncoly;
    }, docstrings::Map::ncoly);

    // Number of limb darkening map columns
#   ifdef _STARRY_EMITTED_
        PyMap.def_property_readonly(
            "ncolu", [] (
                Map<T> &map
            ) {
                return map.ncolu;
        }, docstrings::Map::ncolu);
#   endif

    // Number of wavelength bins
    PyMap.def_property_readonly(
        "nw", [] (
            Map<T> &map
        ) {
#           if defined(_STARRY_SPECTRAL_)
                return map.ncoly;
#           else
                return 1;
#           endif
    }, docstrings::Map::nw);

    // Number of temporal bins
    PyMap.def_property_readonly(
        "nt", [] (
            Map<T> &map
        ) {
#           if defined(_STARRY_TEMPORAL_)
                return map.ncoly;
#           else
                return 1;
#           endif
    }, docstrings::Map::nt);

    // Highest degree of the map
    PyMap.def_property_readonly(
        "ydeg", [] (
            Map<T> &map
        ) {
            return map.ydeg;
    });

    // Highest degree of the map
    PyMap.def_property_readonly(
        "udeg", [] (
            Map<T> &map
        ) {
            return map.udeg;
    });

    // Number of spherical harmonic coefficients
    PyMap.def_property_readonly(
        "N", [] (
            Map<T> &map
        ) {
            return map.N;
    });

    // Number of spherical harmonic coefficients
    PyMap.def_property_readonly(
        "Ny", [] (
            Map<T> &map
        ) {
            return map.Ny;
    });

    // Number of spherical harmonic coefficients
    PyMap.def_property_readonly(
        "Nu", [] (
            Map<T> &map
        ) {
            return map.Nu;
    });

    // Multiprecision enabled?
    PyMap.def_property_readonly(
        "multi", [] (
            Map<T> &map
        ) {
#           if defined(_STARRY_MULTI_)
                return true;
#           else
                return false;
#           endif
    }, docstrings::Map::multi);

    // Set one or more spherical harmonic coefficients to the same scalar value
    PyMap.def(
        "__setitem__", [](
            Map<T>& map, 
            py::tuple lm,
            const double& coeff
        ) {
            auto inds = get_Ylm_inds(map.ydeg, lm);
            auto y = map.getY();
            for (auto n : inds)
                y.row(n).setConstant(static_cast<typename T::Scalar>(coeff));
            map.setY(y);
    }, docstrings::Map::setitem);

    // Set one or more spherical harmonic coefficients to the same vector value
    PyMap.def(
        "__setitem__", [](
            Map<T>& map, 
            py::tuple lm,
            const typename T::Double::YCoeffType& coeff
        ) {
            auto inds = get_Ylm_inds(map.ydeg, lm);
            auto y = map.getY();
            for (auto n : inds)
                y.row(n) = coeff.template cast<typename T::Scalar>();
            map.setY(y);
    });

    // Set multiple spherical harmonic coefficients at once
    PyMap.def(
        "__setitem__", [](
            Map<T>& map, 
            py::tuple lm,
            const typename T::Double::YType& coeff_
        ) {
            auto inds = get_Ylm_inds(map.ydeg, lm);
            typename T::YType coeff = coeff_.template cast<typename T::Scalar>();
            if (coeff.rows() != static_cast<long>(inds.size()))
                throw errors::ValueError("Mismatch in slice length and "
                                         "coefficient array size.");
            auto y = map.getY();
            int i = 0;
            for (auto n : inds)
                y.row(n) = coeff.row(i++);
            map.setY(y);
    });

    // Retrieve one or more spherical harmonic coefficients
    PyMap.def(
        "__getitem__", [](
            Map<T>& map,
            py::tuple lm
        ) -> py::object {
            auto inds = get_Ylm_inds(map.ydeg, lm);
            auto y = map.getY();
            typename T::Double::YType res;
            res.resize(inds.size(), map.ncoly);
            int i = 0;
            for (auto n : inds)
                res.row(i++) = y.row(n).template cast<double>();
            if (inds.size() == 1) {
#               ifdef _STARRY_SINGLECOL_
                    return py::cast<double>(res(0));
#               else
                    auto coeff = py::cast(res.row(0).template cast<double>());
                    MAKE_READ_ONLY(coeff);
                    return coeff;
#               endif
            } else {
                auto coeff = py::cast(res.template cast<double>());
                MAKE_READ_ONLY(coeff);
                return coeff;
            }
    }, docstrings::Map::getitem);

    // Limb darkening I/O
#   ifdef _STARRY_EMITTED_
        // Set one or more limb darkening coefficients to the same scalar value
        PyMap.def(
            "__setitem__", [](
                Map<T>& map, 
                py::object l,
                const double& coeff
            ) {
                auto inds = get_Ul_inds(map.udeg, l);
                auto u = map.getU();
                for (auto n : inds)
                    u.row(n - 1).setConstant(static_cast<typename T::Scalar>(coeff));
                map.setU(u);
        });

        // Set one or more limb darkening coefficients to the same vector value
        PyMap.def(
            "__setitem__", [](
                Map<T>& map, 
                py::object l,
                const typename T::Double::UCoeffType& coeff
            ) {
                auto inds = get_Ul_inds(map.udeg, l);
                auto u = map.getU();
                for (auto n : inds)
                    u.row(n - 1) = coeff.template cast<typename T::Scalar>();
                map.setU(u);
        });

        // Set multiple limb darkening coefficients at once
        PyMap.def(
            "__setitem__", [](
                Map<T>& map, 
                py::object l,
                const typename T::Double::UType& coeff_
            ) {
                auto inds = get_Ul_inds(map.udeg, l);
                typename T::UType coeff = coeff_.template cast<typename T::Scalar>();
                if (coeff.rows() != static_cast<long>(inds.size()))
                    throw errors::ValueError("Mismatch in slice length and "
                                            "coefficient array size.");
                auto u = map.getU();
                int i = 0;
                for (auto n : inds)
                    u.row(n - 1) = coeff.row(i++);
                map.setU(u);
        });

        // Retrieve one or more limb darkening coefficients
        PyMap.def(
            "__getitem__", [](
                Map<T>& map,
                py::object l
            ) -> py::object {
                auto inds = get_Ul_inds(map.udeg, l);
                auto u = map.getU();
                typename T::Double::UType res;
                res.resize(inds.size(), map.ncolu);
                int i = 0;
                for (auto n : inds)
                    res.row(i++) = u.row(n - 1).template cast<double>();
                if (inds.size() == 1) {
#                   if defined(_STARRY_DEFAULT_) || defined(_STARRY_TEMPORAL_)
                        return py::cast<double>(res(0));
#                   else
                        auto coeff = py::cast(res.row(0).template cast<double>());
                        MAKE_READ_ONLY(coeff);
                        return coeff;
#                   endif
                } else {
                    auto coeff = py::cast(res.template cast<double>());
                    MAKE_READ_ONLY(coeff);
                    return coeff;
                }
        });

#   endif

    // Reset the map
    PyMap.def("reset", &Map<T>::reset, docstrings::Map::reset);
    
    // Vector of spherical harmonic coefficients
    PyMap.def_property_readonly(
        "y", [] (
            Map<T> &map
        ) {
            auto y = py::cast(map.getY().template cast<double>());
            MAKE_READ_ONLY(y);
            return y;
    }, docstrings::Map::y);

    // Vector of limb darkening coefficients
#   ifdef _STARRY_EMITTED_
        PyMap.def_property_readonly(
            "u", [] (
                Map<T> &map
            ) {
                auto u = py::cast(map.getU().template cast<double>());
                MAKE_READ_ONLY(u);
                return u;
        }, docstrings::Map::u);
#   endif

    // Get/set the rotation axis
    PyMap.def_property(
        "axis", [] (
            Map<T> &map
        ) -> UnitVector<double> {
            return map.getAxis().template cast<double>();
        }, [] (
            Map<T> &map, 
            UnitVector<double>& axis
        ) {
            map.setAxis(axis.template cast<typename T::Scalar>());
    }, docstrings::Map::axis);

    // Rotate the base map
    PyMap.def("rotate", &Map<T>::rotate, "theta"_a=0.0, docstrings::Map::rotate);

    // Add a gaussian spot with a vector amplitude
#   if defined(_STARRY_SINGLECOL_)
        PyMap.def(
            "add_spot", [](
                Map<T>& map,
                const double& amp,
                const double& sigma,
                const double& lat,
                const double& lon,
                const int lmax
            ) {
                typename T::YCoeffType amp_;
                amp_(0) = amp;
                map.addSpot(amp_, sigma, lat, lon, lmax);
            }, 
            docstrings::Map::add_spot,
            "amp"_a, "sigma"_a=0.1, "lat"_a=0.0, "lon"_a=0.0, "lmax"_a=-1);
#   else
        PyMap.def(
            "add_spot", [](
                Map<T>& map,
                const typename T::Double::YCoeffType& amp,
                const double& sigma,
                const double& lat,
                const double& lon,
                const int lmax
            ) {
                map.addSpot(amp.template cast<typename T::Scalar>(), 
                            sigma, lat, lon, lmax);
            }, 
            docstrings::Map::add_spot,
            "amp"_a, "sigma"_a=0.1, "lat"_a=0.0, "lon"_a=0.0, "lmax"_a=-1);
#   endif

    // Generate a random map
#   if defined(_STARRY_SINGLECOL_)
        PyMap.def(
            "random", [](
                Map<T>& map,
                const Vector<double>& power,
                py::object seed_
            ) {
                if (seed_.is(py::none())) {
                    // \todo Find a better, more thread-safe randomizer seed
                    auto seed = std::chrono::system_clock::now()
                                .time_since_epoch().count();
                    map.random(power.template cast<typename T::Scalar>(), seed);
                } else {
                    double seed = py::cast<double>(seed_);
                    map.random(power.template cast<typename T::Scalar>(), seed);
                }
            }, 
            docstrings::Map::random,
            "power"_a, "seed"_a=py::none());
#   else
        PyMap.def(
            "random", [](
                Map<T>& map,
                const Vector<double>& power,
                py::object seed_,
                int col
            ) {
                if (seed_.is(py::none())) {
                    // \todo Find a better, more thread-safe randomizer seed
                    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
                    map.random(power.template cast<typename T::Scalar>(), seed, col);
                } else {
                    double seed = py::cast<double>(seed_);
                    map.random(power.template cast<typename T::Scalar>(), seed, col);
                }
            }, 
            docstrings::Map::random,
            "power"_a, "seed"_a=py::none(), "col"_a=-1);
#   endif

/* \todo
    // Compute the intensity
#   if defined(_STARRY_STATIC_)
#       if defined(_STARRY_EMITTED_)
            PyMap.def("__call__", intensity<T>(), docstrings::Map::call, 
                      "theta"_a=0.0, "x"_a=0.0, "y"_a=0.0);
#       else
            PyMap.def("__call__", intensity<T>(), docstrings::Map::call, 
                      "theta"_a=0.0, "x"_a=0.0, "y"_a=0.0, 
                      "source"_a=-xhat<double>());
#       endif
#   else
#       if defined(_STARRY_EMITTED_)
            PyMap.def("__call__", intensity<T>(),  docstrings::Map::call, 
                      "t"_a=0.0, "theta"_a=0.0, "x"_a=0.0, "y"_a=0.0);
#       else
            PyMap.def("__call__", intensity<T>(),  docstrings::Map::call, 
                      "t"_a=0.0, "theta"_a=0.0, "x"_a=0.0, "y"_a=0.0, 
                      "source"_a=-xhat<double>());
#       endif
#   endif
*/

// Compute the flux
#   if defined(_STARRY_STATIC_)
#       if defined(_STARRY_EMITTED_)
            PyMap.def("linear_flux_model", linear_flux_model<T>(), 
                      "theta"_a=0.0, "xo"_a=0.0, 
                      "yo"_a=0.0, "zo"_a=1.0, "ro"_a=0.0, "gradient"_a=false);
#       else
            // \todo Implement linear model
#       endif
#   else
#       if defined(_STARRY_EMITTED_)
            PyMap.def("linear_flux_model", linear_flux_model<T>(), 
                      "t"_a=0.0, 
                      "theta"_a=0.0, "xo"_a=0.0, "yo"_a=0.0, "zo"_a=1.0, 
                      "ro"_a=0.0, "gradient"_a=false);
#       else
            // \todo Implement linear model
#       endif
#   endif

    // Code version
#   ifdef VERSION_INFO
        m.attr("__version__") = VERSION_INFO;
#   else
        m.attr("__version__") = "dev";
#   endif

    // A dictionary of all compiler flags
    PyMap.def_property_readonly(
        "__compile_flags__", [] (
            Map<T> &map
        ) -> py::dict {
            auto flags = py::dict();
            flags["STARRY_NMULTI"] = STARRY_NMULTI;
            flags["STARRY_ELLIP_MAX_ITER"] = STARRY_ELLIP_MAX_ITER;
            flags["STARRY_MAX_LMAX"] = STARRY_MAX_LMAX;
            flags["STARRY_BCUT"] = STARRY_BCUT;
            flags["STARRY_MN_MAX_ITER"] = STARRY_MN_MAX_ITER;
#           ifdef STARRY_KEEP_DFDU_AS_DFDG
                flags["STARRY_KEEP_DFDU_AS_DFDG"] = STARRY_KEEP_DFDU_AS_DFDG;
#           else
                flags["STARRY_KEEP_DFDU_AS_DFDG"] = 0;
#           endif
#           ifdef STARRY_O
                flags["STARRY_O"] = STARRY_O;
#           else
                flags["STARRY_O"] = py::none();
#           endif
#           ifdef STARRY_DEBUG
                flags["STARRY_DEBUG"] = STARRY_DEBUG;
#           else
                flags["STARRY_DEBUG"] = 0;
#           endif
            return flags;
    }, docstrings::Map::compile_flags);

}
