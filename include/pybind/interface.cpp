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
#   define _STARRY_DOUBLE_
#   if defined(_STARRY_DEFAULT_DOUBLE_)
#       define _STARRY_NAME_ _starry_default_double
#       define _STARRY_TYPE_ MapType<double, false, false, false, false>
#   else
#       define _STARRY_NAME_ _starry_default_refl_double
#       define _STARRY_TYPE_ MapType<double, false, false, true, false>
#       define _STARRY_REFLECTED_
#   endif
#elif defined(_STARRY_DEFAULT_MULTI_) || defined(_STARRY_DEFAULT_REFL_MULTI_)
#   define _STARRY_MULTI_
#   define STARRY_ENABLE_BOOST
#   if defined(_STARRY_DEFAULT_MULTI_)
#       define _STARRY_NAME_ _starry_default_multi
#       define _STARRY_TYPE_ MapType<Multi, false, false, false, false>
#   else
#       define _STARRY_NAME_ _starry_default_refl_multi
#       define _STARRY_TYPE_ MapType<Multi, false, false, true, false>
#       define _STARRY_REFLECTED_
#   endif
#elif defined(_STARRY_SPECTRAL_DOUBLE_) || defined(_STARRY_SPECTRAL_REFL_DOUBLE_)
#   define _STARRY_SPECTRAL_
#   define _STARRY_DOUBLE_
#   if defined(_STARRY_SPECTRAL_DOUBLE_)
#       define _STARRY_NAME_ _starry_spectral_double
#       define _STARRY_TYPE_ MapType<double, true, false, false, false>
#   else
#       define _STARRY_NAME_ _starry_spectral_refl_double
#       define _STARRY_TYPE_ MapType<double, true, false, true, false>
#       define _STARRY_REFLECTED_
#   endif
#elif defined(_STARRY_SPECTRAL_MULTI_) || defined(_STARRY_SPECTRAL_REFL_MULTI_)
#   define _STARRY_SPECTRAL_
#   define _STARRY_MULTI_
#   define STARRY_ENABLE_BOOST
#   if defined(_STARRY_SPECTRAL_MULTI_)
#       define _STARRY_NAME_ _starry_spectral_multi
#       define _STARRY_TYPE_ MapType<Multi, true, false, false, false>
#   else
#       define _STARRY_NAME_ _starry_spectral_refl_multi
#       define _STARRY_TYPE_ MapType<Multi, true, false, true, false>
#       define _STARRY_REFLECTED_
#   endif
#elif defined(_STARRY_TEMPORAL_DOUBLE_) || defined(_STARRY_TEMPORAL_REFL_DOUBLE_)
#   define _STARRY_TEMPORAL_
#   define _STARRY_DOUBLE_
#   if defined(_STARRY_TEMPORAL_DOUBLE_)
#       define _STARRY_NAME_ _starry_temporal_double
#       define _STARRY_TYPE_ MapType<double, false, true, false, false>
#   else
#       define _STARRY_NAME_ _starry_temporal_refl_double
#       define _STARRY_TYPE_ MapType<double, false, true, true, false>
#       define _STARRY_REFLECTED_
#   endif
#elif defined(_STARRY_TEMPORAL_MULTI_) || defined(_STARRY_TEMPORAL_REFL_MULTI_)
#   define _STARRY_TEMPORAL_
#   define _STARRY_MULTI_
#   define STARRY_ENABLE_BOOST
#   if defined(_STARRY_TEMPORAL_MULTI_)
#       define _STARRY_NAME_ _starry_temporal_multi
#       define _STARRY_TYPE_ MapType<Multi, false, true, false, false>
#   else
#       define _STARRY_NAME_ _starry_temporal_refl_multi
#       define _STARRY_TYPE_ MapType<Multi, false, true, true, false>
#       define _STARRY_REFLECTED_
#   endif
#elif defined(_STARRY_LIMBDARKENED_DOUBLE_)
#   define _STARRY_LIMBDARKENED_
#   define _STARRY_DOUBLE_
#   define _STARRY_NAME_ _starry_limbdarkened_double
#   define _STARRY_TYPE_ MapType<double, false, false, false, true>
#elif defined(_STARRY_LIMBDARKENED_MULTI_)
#   define _STARRY_LIMBDARKENED_
#   define _STARRY_MULTI_
#   define _STARRY_NAME_ _starry_limbdarkened_multi
#   define _STARRY_TYPE_ MapType<Multi, false, false, false, true>
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
    using Scalar = typename T::Scalar;

    // Declare the Map class
    py::class_<Map<T>> PyMap(m, "Map", docstrings::Map::doc);

    // Constructor
#   if defined(_STARRY_LIMBDARKENED_)
        PyMap.def(py::init<int>(), "udeg"_a=2);
#   elif defined(_STARRY_TEMPORAL_) || defined(_STARRY_SPECTRAL_) 
        PyMap.def(py::init<int, int, int>(), 
                  "ydeg"_a=2, "udeg"_a=0, "nterms"_a=1);
#   else
        PyMap.def(py::init<int, int>(), "ydeg"_a=2, "udeg"_a=0);
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

    // Set one or more spherical harmonic coefficients
    PyMap.def(
        "__setitem__", [](
            Map<T>& map, 
            py::tuple lm,
            py::array_t<double>& coeff_
        ) {
            // Figure out the indices we're setting
#           if defined(_STARRY_TEMPORAL_)
                std::vector<int> rows = 
                    std::get<0>(get_Ylmt_inds(map.ydeg, map.Nt, lm));
                std::vector<int> cols(1, 0);
#           elif defined(_STARRY_SPECTRAL_)
                auto inds = get_Ylmw_inds(map.ydeg, map.Nw, lm);
                std::vector<int> rows = std::get<0>(inds);
                std::vector<int> cols = std::get<1>(inds);
#           else
                std::vector<int> rows = get_Ylm_inds(map.ydeg, lm);
                std::vector<int> cols(1, 0);
#           endif

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

#           if defined(_STARRY_TEMPORAL_)
                // Flatten the input array if needed
                Matrix<Scalar> tmpcoeff = coeff.transpose();
                coeff = Eigen::Map<Matrix<Scalar>>(tmpcoeff.data(), 
                                                   coeff.size(), 1);
#           endif

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
    }, docstrings::Map::setitem);

    // Retrieve one or more spherical harmonic coefficients
    PyMap.def(
        "__getitem__", [](
            Map<T>& map,
            py::tuple lm
        ) -> py::object {
            // Figure out the indices we're accessing
#           if defined(_STARRY_TEMPORAL_)
                auto rows_ncols = get_Ylmt_inds(map.ydeg, map.Nt, lm);
                std::vector<int> rows = std::get<0>(rows_ncols);
                int ncols = std::get<1>(rows_ncols);
                std::vector<int> cols(1, 0);
                Matrix<double> coeff_(rows.size(), cols.size());
#           elif defined(_STARRY_SPECTRAL_)
                auto inds = get_Ylmw_inds(map.ydeg, map.Nw, lm);
                std::vector<int> rows = std::get<0>(inds);
                std::vector<int> cols = std::get<1>(inds);
                Matrix<double> coeff_(rows.size(), cols.size());
#           else
                std::vector<int> rows = get_Ylm_inds(map.ydeg, lm);
                std::vector<int> cols(1, 0);
                Vector<double> coeff_(rows.size());
#           endif

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

#           if defined(_STARRY_TEMPORAL_)
                // Reshape the coefficients into a matrix
                Matrix<double> tmpcoeff = coeff_;
                coeff_ = Eigen::Map<Matrix<double>>(tmpcoeff.data(), 
                            ncols, coeff_.size() / ncols).transpose();
#           endif

            // Squeeze the output and cast to a py::array
            if (coeff_.size() == 1) {
#               if defined(_STARRY_TEMPORAL_) || defined(_STARRY_SPECTRAL_) 
                    auto coeff = py::cast(coeff_.row(0));
                    MAKE_READ_ONLY(coeff);
                    return coeff;
#               else
                    return py::cast<double>(coeff_(0));
#               endif
            } else {
                auto coeff = py::cast(coeff_);
                MAKE_READ_ONLY(coeff);
                return coeff;
            }
    }, docstrings::Map::getitem);

    // Set one or more limb darkening coefficients
    PyMap.def(
        "__setitem__", [](
            Map<T>& map, 
            py::object l,
            py::array_t<double>& coeff_
        ) {
            // Figure out the indices we're setting
            std::vector<int> rows = get_Ul_inds(map.udeg, l);

            // Reshape coeff if necessary
            py::buffer_info buf = coeff_.request();
            double *ptr = (double *) buf.ptr;
            Vector<Scalar> coeff(rows.size());
            if (buf.ndim == 0) {
                // Set an array of indices (or rows/columns) to same value
                coeff.setConstant(ptr[0]);
            } else if (buf.ndim == 1) {
                // Set an array of indices to an array of values
                coeff = py::cast<Vector<double>>(coeff_).template 
                            cast<Scalar>();
            } else {
                // ?
                throw std::length_error("Invalid coefficient "
                                        "array shape.");
            }

            // Check shape
            if (!(size_t(coeff.rows()) == size_t(rows.size())))
                throw std::length_error("Mismatch in index array and " 
                                        "coefficient array sizes.");

            // Grab the map vector and update it term by term
            auto u = map.getU();
            int i = 0;
            for (int row : rows) {
                u(row) = static_cast<Scalar>(coeff(i));
                ++i;
            }
            map.setU(u);
    }, docstrings::Map::setitem);

    // Retrieve one or more limb darkening coefficients
    PyMap.def(
        "__getitem__", [](
            Map<T>& map,
            py::object l
        ) -> py::object {
            // Figure out the indices we're accessing
            std::vector<int> rows = get_Ul_inds(map.udeg, l);
            Vector<double> coeff_(rows.size());

            // Grab the map vector and update the output term by term
            auto u = map.getU();
            int i = 0;
            for (int row : rows) {
                coeff_(i) = static_cast<double>(u(row));
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
    }, docstrings::Map::getitem);

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

// Ylm map methods
#   if !defined(_STARRY_LIMBDARKENED_)

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
#          else
                PyMap.def("linear_intensity_model", linear_intensity_model<T>(),
                        "t"_a=0.0, "theta"_a=0.0, "x"_a=0.0, "y"_a=0.0);
#          endif
#      else
#          if defined(_STARRY_REFLECTED_)
                PyMap.def("linear_intensity_model", linear_intensity_model<T>(),
                        "theta"_a=0.0, "x"_a=0.0, "y"_a=0.0, 
                        "source"_a=-xhat<double>());
#          else
                PyMap.def("linear_intensity_model", linear_intensity_model<T>(), 
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
