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
#if defined(_STARRY_DEFAULT_YLM_DOUBLE_) || defined(_STARRY_DEFAULT_REFLECTED_DOUBLE_)
#   define _STARRY_DOUBLE_
#   define _STARRY_YDIM_ 2
#   define _STARRY_UDIM_ 1
#   if defined(_STARRY_DEFAULT_YLM_DOUBLE_)
#       define _STARRY_NAME_ _starry_default_ylm_double
#       define _STARRY_TYPE_ MapType<double, false, false, false, false>
#   else
#       define _STARRY_NAME_ _starry_default_reflected_double
#       define _STARRY_TYPE_ MapType<double, false, false, true, false>
#       define _STARRY_REFLECTED_
#   endif
#elif defined(_STARRY_DEFAULT_YLM_MULTI_) || defined(_STARRY_DEFAULT_REFLECTED_MULTI_)
#   define _STARRY_MULTI_
#   define STARRY_ENABLE_BOOST
#   define _STARRY_YDIM_ 2
#   define _STARRY_UDIM_ 1
#   if defined(_STARRY_DEFAULT_YLM_MULTI_)
#       define _STARRY_NAME_ _starry_default_ylm_multi
#       define _STARRY_TYPE_ MapType<Multi, false, false, false, false>
#   else
#       define _STARRY_NAME_ _starry_default_reflected_multi
#       define _STARRY_TYPE_ MapType<Multi, false, false, true, false>
#       define _STARRY_REFLECTED_
#   endif
#elif defined(_STARRY_SPECTRAL_YLM_DOUBLE_) || defined(_STARRY_SPECTRAL_REFLECTED_DOUBLE_)
#   define _STARRY_SPECTRAL_
#   define _STARRY_DOUBLE_
#   define _STARRY_YDIM_ 3
#   define _STARRY_UDIM_ 1
#   if defined(_STARRY_SPECTRAL_YLM_DOUBLE_)
#       define _STARRY_NAME_ _starry_spectral_ylm_double
#       define _STARRY_TYPE_ MapType<double, true, false, false, false>
#   else
#       define _STARRY_NAME_ _starry_spectral_reflected_double
#       define _STARRY_TYPE_ MapType<double, true, false, true, false>
#       define _STARRY_REFLECTED_
#   endif
#elif defined(_STARRY_SPECTRAL_YLM_MULTI_) || defined(_STARRY_SPECTRAL_REFLECTED_MULTI_)
#   define _STARRY_SPECTRAL_
#   define _STARRY_MULTI_
#   define STARRY_ENABLE_BOOST
#   define _STARRY_YDIM_ 3
#   define _STARRY_UDIM_ 1
#   if defined(_STARRY_SPECTRAL_YLM_MULTI_)
#       define _STARRY_NAME_ _starry_spectral_ylm_multi
#       define _STARRY_TYPE_ MapType<Multi, true, false, false, false>
#   else
#       define _STARRY_NAME_ _starry_spectral_reflected_multi
#       define _STARRY_TYPE_ MapType<Multi, true, false, true, false>
#       define _STARRY_REFLECTED_
#   endif
#elif defined(_STARRY_TEMPORAL_YLM_DOUBLE_) || defined(_STARRY_TEMPORAL_REFLECTED_DOUBLE_)
#   define _STARRY_TEMPORAL_
#   define _STARRY_DOUBLE_
#   define _STARRY_YDIM_ 3
#   define _STARRY_UDIM_ 1
#   if defined(_STARRY_TEMPORAL_YLM_DOUBLE_)
#       define _STARRY_NAME_ _starry_temporal_ylm_double
#       define _STARRY_TYPE_ MapType<double, false, true, false, false>
#   else
#       define _STARRY_NAME_ _starry_temporal_reflected_double
#       define _STARRY_TYPE_ MapType<double, false, true, true, false>
#       define _STARRY_REFLECTED_
#   endif
#elif defined(_STARRY_TEMPORAL_YLM_MULTI_) || defined(_STARRY_TEMPORAL_REFLECTED_MULTI_)
#   define _STARRY_TEMPORAL_
#   define _STARRY_MULTI_
#   define STARRY_ENABLE_BOOST
#   define _STARRY_YDIM_ 3
#   define _STARRY_UDIM_ 1
#   if defined(_STARRY_TEMPORAL_YLM_MULTI_)
#       define _STARRY_NAME_ _starry_temporal_ylm_multi
#       define _STARRY_TYPE_ MapType<Multi, false, true, false, false>
#   else
#       define _STARRY_NAME_ _starry_temporal_reflected_multi
#       define _STARRY_TYPE_ MapType<Multi, false, true, true, false>
#       define _STARRY_REFLECTED_
#   endif
#elif defined(_STARRY_DEFAULT_LD_DOUBLE_)
#   define _STARRY_LD_
#   define _STARRY_DOUBLE_
#   define _STARRY_YDIM_ 0
#   define _STARRY_UDIM_ 1
#   define _STARRY_NAME_ _starry_default_ld_double
#   define _STARRY_TYPE_ MapType<double, false, false, false, true>
#elif defined(_STARRY_DEFAULT_LD_MULTI_)
#   define _STARRY_LD_
#   define _STARRY_MULTI_
#   define STARRY_ENABLE_BOOST
#   define _STARRY_YDIM_ 0
#   define _STARRY_UDIM_ 1
#   define _STARRY_NAME_ _starry_default_ld_multi
#   define _STARRY_TYPE_ MapType<Multi, false, false, false, true>
#elif defined(_STARRY_SPECTRAL_LD_DOUBLE_)
#   define _STARRY_SPECTRAL_
#   define _STARRY_LD_
#   define _STARRY_DOUBLE_
#   define _STARRY_YDIM_ 0
#   define _STARRY_UDIM_ 2
#   define _STARRY_NAME_ _starry_spectral_ld_double
#   define _STARRY_TYPE_ MapType<double, true, false, false, true>
#elif defined(_STARRY_SPECTRAL_LD_MULTI_)
#   define _STARRY_SPECTRAL_
#   define _STARRY_LD_
#   define _STARRY_MULTI_
#   define STARRY_ENABLE_BOOST
#   define _STARRY_YDIM_ 0
#   define _STARRY_UDIM_ 2
#   define _STARRY_NAME_ _starry_spectral_ld_multi
#   define _STARRY_TYPE_ MapType<Multi, true, false, false, true>
#else
    static_assert(false, "Invalid or missing `starry` module type.");
#endif

// Includes
#include <pybind11/embed.h>
#include "interface.h"
#include "docstrings.h"
using namespace interface;

class Filter {
};

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
    using Scalar = typename T::Scalar;

    // Declare the Map class
    py::class_<Map<T>> PyMap(m, "Map", docstrings::Map::doc);

    // Constructor
#   if defined(_STARRY_LD_)
#       if defined(_STARRY_SPECTRAL_)
            PyMap.def(py::init<int, int>(), "udeg"_a=2, "nterms"_a=1);
#       else
            PyMap.def(py::init<int>(), "udeg"_a=2);
#       endif
#   elif defined(_STARRY_TEMPORAL_) || defined(_STARRY_SPECTRAL_) 
        PyMap.def(py::init<int, int, int, int>(), 
                  "ydeg"_a=2, "udeg"_a=0, "fdeg"_a=0, "nterms"_a=1);
#   else
        PyMap.def(py::init<int, int, int>(), "ydeg"_a=2, "udeg"_a=0, "fdeg"_a=0);
#   endif

    // String representation of the map
    PyMap.def("__repr__", &Map<T>::info);

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

    // Total number of spherical harmonic coefficients after limb-darkening
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

    // Number of limb darkening coefficients
    PyMap.def_property_readonly(
        "Nu", [] (
            Map<T> &map
        ) {
            return map.Nu;
    });

    // Number of temporal components
    PyMap.def_property_readonly(
        "nt", [] (
            Map<T> &map
        ) {
            return map.Nt;
    });

    // Number of spectral components
    PyMap.def_property_readonly(
        "nw", [] (
            Map<T> &map
        ) {
            return map.Nw;
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

    // Item setter
    PyMap.def(
        "__setitem__", [](
            Map<T>& map,
            const py::object& inds,
            py::array_t<double>& coeff
        ) {
            int size;
            if (py::isinstance<py::tuple>(inds))
                size = py::cast<py::tuple>(inds).size();
            else
                size = 1;
            if (size == _STARRY_UDIM_)
                return set_Ul(map, inds, coeff);
            else if (size == _STARRY_YDIM_)
                return set_Ylm(map, inds, coeff);
            else
                throw std::invalid_argument(
                    "Incorrect coefficient index shape for this type of map."
                );
    }, docstrings::Map::setitem);

    // Item getter
    PyMap.def(
        "__getitem__", [](
            Map<T>& map,
            const py::object& inds
        ) -> py::object {
            int size;
            if (py::isinstance<py::tuple>(inds))
                size = py::cast<py::tuple>(inds).size();
            else
                size = 1;
            if (size == _STARRY_UDIM_)
                return get_Ul(map, inds);
            else if (size == _STARRY_YDIM_)
                return get_Ylm(map, inds);
            else
                throw std::invalid_argument(
                    "Incorrect coefficient index shape for this type of map."
                );
    }, docstrings::Map::getitem);

    // Filter setter
    PyMap.def(
        "_set_filter", [](
            Map<T>& map,
            const py::object& inds,
            py::array_t<double>& coeff
        ) {
            int size;
            if (py::isinstance<py::tuple>(inds))
                size = py::cast<py::tuple>(inds).size();
            else
                size = 1;
            if (size == 2)
                return set_Flm(map, inds, coeff);
            else
                throw std::invalid_argument(
                    "Incorrect coefficient index shape for this type of map."
                );
    });

    // Filter getter
    PyMap.def(
        "_get_filter", [](
            Map<T>& map,
            const py::object& inds
        ) -> py::object {
            int size;
            if (py::isinstance<py::tuple>(inds))
                size = py::cast<py::tuple>(inds).size();
            else
                size = 1;
            if (size == 2)
                return get_Flm(map, inds);
            else
                throw std::invalid_argument(
                    "Incorrect coefficient index shape for this type of map."
                );
    });

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
    PyMap.def_property_readonly(
        "u", [] (
            Map<T> &map
        ) {
            auto u = py::cast(map.getU().template cast<double>());
            MAKE_READ_ONLY(u);
            return u;
    }, docstrings::Map::u);

    // Vector of filter spherical harmonic coefficients
    PyMap.def_property_readonly(
        "f", [] (
            Map<T> &map
        ) {
            auto f = py::cast(map.getF().template cast<double>());
            MAKE_READ_ONLY(f);
            return f;
    });

// Ylm map methods
#   if !defined(_STARRY_LD_)

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
                map.setAxis(axis.template cast<Scalar>());
        }, docstrings::Map::axis);

        // Get/set the map inclination
        PyMap.def_property(
            "inc", [] (
                Map<T> &map
            ) -> double {
                return static_cast<double>(map.getInclination());
            }, [] (
                Map<T> &map, 
                double& inc
            ) {
                map.setInclination(static_cast<Scalar>(inc));
        }, docstrings::Map::inc);

        // Get/set the map obliquity
        PyMap.def_property(
            "obl", [] (
                Map<T> &map
            ) -> double {
                return static_cast<double>(map.getObliquity());
            }, [] (
                Map<T> &map, 
                double& obl
            ) {
                map.setObliquity(static_cast<Scalar>(obl));
        }, docstrings::Map::obl);

        // Rotate the base map
        PyMap.def(
            "rotate", [](
                Map<T>& map,
                const double& theta
            ) {
                map.rotate(theta);
        }, "theta"_a=0.0, docstrings::Map::rotate);

        // Add a gaussian spot
#       if defined(_STARRY_TEMPORAL_) || defined(_STARRY_SPECTRAL_) 
            PyMap.def(
                "add_spot", [](
                    Map<T>& map,
                    const RowVector<double>& amp,
                    const double& sigma,
                    const double& lat,
                    const double& lon,
                    const int lmax
                ) {
                    map.addSpot(amp.template cast<Scalar>(), 
                                sigma, lat, lon, lmax);
                }, 
                docstrings::Map::add_spot,
                "amp"_a, "sigma"_a=0.1, "lat"_a=0.0, "lon"_a=0.0, "lmax"_a=-1);
#      else
            PyMap.def(
                "add_spot", [](
                    Map<T>& map,
                    const double& amp,
                    const double& sigma,
                    const double& lat,
                    const double& lon,
                    const int lmax
                ) {
                    RowVector<Scalar> amp_(1);
                    amp_(0) = amp;
                    map.addSpot(amp_, sigma, lat, lon, lmax);
                }, 
                docstrings::Map::add_spot,
                "amp"_a, "sigma"_a=0.1, "lat"_a=0.0, "lon"_a=0.0, "lmax"_a=-1);
#      endif

        // Generate a random map
#      if defined(_STARRY_TEMPORAL_) || defined(_STARRY_SPECTRAL_) 
            PyMap.def(
                "random", [](
                    Map<T>& map,
                    const Vector<double>& power,
                    py::object seed_,
                    int col
                ) {
                    if (seed_.is(py::none())) {
                        // \todo Find a better, more thread-safe randomizer seed
                        auto seed = std::chrono::system_clock::now()
                                    .time_since_epoch().count();
                        map.random(power.template cast<Scalar>(), seed, col);
                    } else {
                        double seed = py::cast<double>(seed_);
                        map.random(power.template cast<Scalar>(), seed, col);
                    }
                }, 
                docstrings::Map::random,
                "power"_a, "seed"_a=py::none(), "col"_a=-1);
#      else
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
                        map.random(power.template cast<Scalar>(), seed);
                    } else {
                        double seed = py::cast<double>(seed_);
                        map.random(power.template cast<Scalar>(), seed);
                    }
                }, 
                docstrings::Map::random,
                "power"_a, "seed"_a=py::none());
#      endif

        // Compute the linear intensity model
#      if defined(_STARRY_TEMPORAL_)
#          if defined(_STARRY_REFLECTED_)
                PyMap.def("linear_intensity_model", linear_intensity_model<T>(),
                        "t"_a=0.0, "theta"_a=0.0, "x"_a=0.0, "y"_a=0.0, 
                        "source"_a=-xhat<double>());
                PyMap.def("intensity", linear_intensity_model<T, true>(),
                          "t"_a=0.0, "theta"_a=0.0, "x"_a=0.0, "y"_a=0.0, 
                          "source"_a=-xhat<double>());
#          else
                PyMap.def("linear_intensity_model", linear_intensity_model<T>(),
                          "t"_a=0.0, "theta"_a=0.0, "x"_a=0.0, "y"_a=0.0);
                PyMap.def("intensity", linear_intensity_model<T, true>(),
                          "t"_a=0.0, "theta"_a=0.0, "x"_a=0.0, "y"_a=0.0);
#          endif
#      else
#          if defined(_STARRY_REFLECTED_)
                PyMap.def("linear_intensity_model", linear_intensity_model<T>(),
                          "theta"_a=0.0, "x"_a=0.0, "y"_a=0.0, 
                          "source"_a=-xhat<double>());
                PyMap.def("intensity", linear_intensity_model<T, true>(),
                          "theta"_a=0.0, "x"_a=0.0, "y"_a=0.0, 
                          "source"_a=-xhat<double>());
#          else
                PyMap.def("linear_intensity_model", linear_intensity_model<T>(), 
                          "theta"_a=0.0, "x"_a=0.0, "y"_a=0.0);
                PyMap.def("intensity", linear_intensity_model<T, true>(), 
                          "theta"_a=0.0, "x"_a=0.0, "y"_a=0.0);
#          endif
#      endif

        // Compute the linear flux model
#      if defined(_STARRY_TEMPORAL_)
#          if defined(_STARRY_REFLECTED_)
                PyMap.def("linear_flux_model", linear_flux_model<T>(), 
                        "t"_a=0.0, 
                        "theta"_a=0.0, "xo"_a=0.0, "yo"_a=0.0, "zo"_a=1.0, 
                        "ro"_a=0.0, "source"_a=-xhat<double>(), 
                        "gradient"_a=false);
#          else
                PyMap.def("linear_flux_model", linear_flux_model<T>(), 
                        "t"_a=0.0, 
                        "theta"_a=0.0, "xo"_a=0.0, "yo"_a=0.0, "zo"_a=1.0, 
                        "ro"_a=0.0, "gradient"_a=false);
#          endif
#      else
#          if defined(_STARRY_REFLECTED_)
                PyMap.def("linear_flux_model", linear_flux_model<T>(), 
                        "theta"_a=0.0, "xo"_a=0.0, 
                        "yo"_a=0.0, "zo"_a=1.0, "ro"_a=0.0, 
                        "source"_a=-xhat<double>(), "gradient"_a=false);
#          else
                PyMap.def("linear_flux_model", linear_flux_model<T>(), 
                        "theta"_a=0.0, "xo"_a=0.0, 
                        "yo"_a=0.0, "zo"_a=1.0, "ro"_a=0.0, "gradient"_a=false);
#          endif
#      endif

// Limb darkened map methods
#else

    PyMap.def("flux", ld_flux<T>(),
              "b"_a=0.0, "zo"_a=1.0, "ro"_a=0.0, "gradient"_a=false);

#endif

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
