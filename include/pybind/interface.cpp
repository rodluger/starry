/**
\file interface.cpp
\brief Defines the entry point for the C++ API.

*/

// Enable debug mode?
#ifdef STARRY_DEBUG
#undef NDEBUG
#endif

// Enable the Python interface
#ifndef STARRY_ENABLE_PYTHON_INTERFACE
#define STARRY_ENABLE_PYTHON_INTERFACE
#endif

// Select which module to build
#if defined(_STARRY_DEFAULT_DOUBLE_)
#define _STARRY_NAME_ _starry_default_double
#define _STARRY_DEFAULT_
#define _STARRY_DOUBLE_
#define _STARRY_STATIC_
#define _STARRY_SINGLECOL_
#define _STARRY_TYPE_ Default<double>

#elif defined(_STARRY_DEFAULT_MULTI_)
#define _STARRY_NAME_ _starry_default_multi
#define _STARRY_DEFAULT_
#define _STARRY_MULTI_
#define _STARRY_STATIC_
#define _STARRY_SINGLECOL_
#define STARRY_ENABLE_BOOST
#define _STARRY_TYPE_ Default<Multi>

#elif defined(_STARRY_SPECTRAL_DOUBLE_)
#define _STARRY_NAME_ _starry_spectral_double
#define _STARRY_SPECTRAL_
#define _STARRY_DOUBLE_
#define _STARRY_STATIC_
#define _STARRY_MULTI_COL
#define _STARRY_TYPE_ Spectral<double>

#elif defined(_STARRY_SPECTRAL_MULTI_)
#define _STARRY_NAME_ _starry_spectral_multi
#define _STARRY_SPECTRAL_
#define _STARRY_MULTI_
#define _STARRY_STATIC_
#define _STARRY_MULTI_COL
#define STARRY_ENABLE_BOOST
#define _STARRY_TYPE_ Spectral<Multi>

#elif defined(_STARRY_TEMPORAL_DOUBLE_)
#define _STARRY_NAME_ _starry_temporal_double
#define _STARRY_TEMPORAL_
#define _STARRY_DOUBLE_
#define _STARRY_MULTI_COL
#define _STARRY_TYPE_ Temporal<double>

#elif defined(_STARRY_TEMPORAL_MULTI_)
#define _STARRY_NAME_ _starry_temporal_multi
#define _STARRY_TEMPORAL_
#define _STARRY_MULTI_
#define _STARRY_MULTI_COL
#define STARRY_ENABLE_BOOST
#define _STARRY_TYPE_ Temporal<Multi>

#else
static_assert(false, "Invalid or missing STARRY module type.");
#endif

//! Includes
#include <pybind11/embed.h>
#include "interface.h"
#include "docstrings.h"
using namespace interface;

//! Register the Python module
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

#if defined(_STARRY_SINGLECOL_) 
    // Constructor for vector maps
    PyMap.def(py::init<int>(), "lmax"_a=2);
#else
    // Constructor for matrix maps
    PyMap.def(py::init<int, int>(), "lmax"_a=2, "ncol"_a=1);
#endif

    // String representation of the map
    PyMap.def("__repr__", &Map<T>::info);

    // Number of Ylm map columns
    PyMap.def_property_readonly(
        "ncoly", [] (
            Map<T> &map
        ) {
            return map.ncoly;
    }, docstrings::Map::ncoly);

    // Number of Ul map columns
    PyMap.def_property_readonly(
        "ncolu", [] (
            Map<T> &map
        ) {
            return map.ncolu;
    }, docstrings::Map::ncolu);

    // Number of wavelength bins
    PyMap.def_property_readonly(
        "nw", [] (
            Map<T> &map
        ) {
#if defined(_STARRY_SPECTRAL_)
            return map.ncoly;
#else
            return 1;
#endif
    }, docstrings::Map::nw);

    // Number of temporal bins
    PyMap.def_property_readonly(
        "nt", [] (
            Map<T> &map
        ) {
#if defined(_STARRY_TEMPORAL_)
            return map.ncoly;
#else
            return 1;
#endif
    }, docstrings::Map::nt);

    // Highest degree of the map
    PyMap.def_property_readonly(
        "lmax", [] (
            Map<T> &map
        ) {
            return map.lmax;
    }, docstrings::Map::lmax);

    // Number of spherical harmonic coefficients
    PyMap.def_property_readonly(
        "N", [] (
            Map<T> &map
        ) {
            return map.N;
    }, docstrings::Map::N);

    // Multiprecision enabled?
    PyMap.def_property_readonly(
        "multi", [] (
            Map<T> &map
        ) {
#if defined(_STARRY_MULTI_)
            return true;
#else
            return false;
#endif
    }, docstrings::Map::multi);

    // Set one or more spherical harmonic coefficients to the same scalar value
    PyMap.def(
        "__setitem__", [](
            Map<T>& map, 
            py::tuple lm,
            const double& coeff
        ) {
            auto inds = get_Ylm_inds(map.lmax, lm);
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
            auto inds = get_Ylm_inds(map.lmax, lm);
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
            auto inds = get_Ylm_inds(map.lmax, lm);
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
            auto inds = get_Ylm_inds(map.lmax, lm);
            auto y = map.getY();
            typename T::Double::YType res;
            res.resize(inds.size(), map.ncoly);
            int i = 0;
            for (auto n : inds)
                res.row(i++) = y.row(n).template cast<double>();
            if (inds.size() == 1) {
#ifdef _STARRY_SINGLECOL_
                return py::cast<double>(res(0));
#else
                auto coeff = py::cast(res.row(0).template cast<double>());
                MAKE_READ_ONLY(coeff);
                return coeff;
#endif
            } else {
                auto coeff = py::cast(res.template cast<double>());
                MAKE_READ_ONLY(coeff);
                return coeff;
            }
    }, docstrings::Map::getitem);

    // Set one or more limb darkening coefficients to the same scalar value
    PyMap.def(
        "__setitem__", [](
            Map<T>& map, 
            py::object l,
            const double& coeff
        ) {
            auto inds = get_Ul_inds(map.lmax, l);
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
            auto inds = get_Ul_inds(map.lmax, l);
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
            auto inds = get_Ul_inds(map.lmax, l);
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
            auto inds = get_Ul_inds(map.lmax, l);
            auto u = map.getU();
            typename T::Double::UType res;
            res.resize(inds.size(), map.ncolu);
            int i = 0;
            for (auto n : inds)
                res.row(i++) = u.row(n - 1).template cast<double>();
            if (inds.size() == 1) {
#if defined(_STARRY_DEFAULT_) || defined(_STARRY_TEMPORAL_)
                return py::cast<double>(res(0));
#else
                auto coeff = py::cast(res.row(0).template cast<double>());
                MAKE_READ_ONLY(coeff);
                return coeff;
#endif
            } else {
                auto coeff = py::cast(res.template cast<double>());
                MAKE_READ_ONLY(coeff);
                return coeff;
            }
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

#if defined(_STARRY_SINGLECOL_)
    // Add a gaussian spot with a scalar amplitude
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
            map.addSpot(amp_, 
                        sigma, lat, lon, lmax);
        }, 
        docstrings::Map::add_spot,
        "amp"_a, "sigma"_a=0.1, "lat"_a=0.0, "lon"_a=0.0, "lmax"_a=-1);
#else
    // Add a gaussian spot with a vector amplitude
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
#endif

#if defined(_STARRY_SINGLECOL_)
    // Generate a random map
    PyMap.def(
        "random", [](
            Map<T>& map,
            const Vector<double>& power,
            py::object seed_
        ) {
            if (seed_.is(py::none())) {
                // TODO: We need a better, more thread-safe randomizer seed
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
#else
    // Generate a random map
    PyMap.def(
        "random", [](
            Map<T>& map,
            const Vector<double>& power,
            py::object seed_,
            int col
        ) {
            if (seed_.is(py::none())) {
                // TODO: We need a better, more thread-safe randomizer seed
                auto seed = std::chrono::system_clock::now().time_since_epoch().count();
                map.random(power.template cast<typename T::Scalar>(), seed, col);
            } else {
                double seed = py::cast<double>(seed_);
                map.random(power.template cast<typename T::Scalar>(), seed, col);
            }
        }, 
        docstrings::Map::random,
        "power"_a, "seed"_a=py::none(), "col"_a=-1);
#endif

#if defined(_STARRY_STATIC_)
    // Show an image/animation of the map
    PyMap.def(
        "show", [](
            Map<T>& map,
            py::array_t<double> theta_,
            std::string cmap,
            int res,
            int interval,
            std::string gif
        ) -> py::object {
            auto theta = py::cast<Vector<double>>(theta_);
            if (theta.size() == 0) {
                return map.show(0.0, cmap, res);
            } else if (theta.size() == 1) {
                return map.show(theta(0), cmap, res);
            } else {
                return map.show(theta.template cast<typename T::Scalar>(), 
                                cmap, res, interval, gif);
            }
        }, 
        docstrings::Map::show,
        "theta"_a=py::array_t<double>(), "cmap"_a="plasma", "res"_a=300,
        "interval"_a=75, "gif"_a="");
#else
    // Show an image/animation of the map
    PyMap.def(
        "show", [](
            Map<T>& map,
            py::array_t<double> t_,
            py::array_t<double> theta_,
            std::string cmap,
            int res,
            int interval,
            std::string gif
        ) -> py::object {
            auto atleast_1d = py::module::import("numpy").attr("atleast_1d");
            Vector<double> t = py::cast<Vector<double>>(atleast_1d(t_));
            Vector<double> theta = py::cast<Vector<double>>(atleast_1d(theta_));
            int sz = max(t.size(), theta.size());
            if ((t.size() == 0) || (theta.size() == 0)) {
                throw errors::ValueError(
                    "Invalid dimensions for `t` and/or `theta`.");
            } else if (t.size() == 1) {
                t.setConstant(sz, t(0));
            } else if (theta.size() == 1) {
                theta.setConstant(sz, theta(0));
            } else if (t.size() != theta.size()){
                throw errors::ValueError(
                    "Invalid dimensions for `t` and/or `theta`.");
            }
            return map.show(t.template cast<typename T::Scalar>(), 
                            theta.template cast<typename T::Scalar>(), 
                            cmap, res, interval, gif);
        }, 
        docstrings::Map::show,
        "t"_a=0.0, "theta"_a=0.0, 
        "cmap"_a="plasma", "res"_a=300, "interval"_a=75, "gif"_a="");
#endif

#if defined(_STARRY_DEFAULT_)
    // Render the visible map on a square grid
    PyMap.def(
        "render", [](
            Map<T>& map,
            double theta,
            int res
        ) -> py::object {
            auto reshape = py::module::import("numpy").attr("reshape");
            Matrix<typename T::Scalar> intensity;
            map.renderMap(theta, res, intensity);
            return reshape(intensity.template cast<double>(), 
                           py::make_tuple(res, res));

        }, 
        docstrings::Map::render,
        "theta"_a=0.0, "res"_a=300);
#elif defined(_STARRY_SPECTRAL_)
    // Render the visible map on a square grid
    PyMap.def(
        "render", [](
            Map<T>& map,
            double theta,
            int res
        ) -> py::object {
            auto reshape = py::module::import("numpy").attr("reshape");
            Matrix<typename T::Scalar> intensity;
            map.renderMap(theta, res, intensity);
            return reshape(intensity.template cast<double>(), 
                           py::make_tuple(res, res, map.nflx));
        }, 
        docstrings::Map::render,
        "theta"_a=0.0, "res"_a=300);
#elif defined(_STARRY_TEMPORAL_)
    // Render the visible map on a square grid
    PyMap.def(
        "render", [](
            Map<T>& map,
            double t,
            double theta,
            int res
        ) -> py::object {
            auto reshape = py::module::import("numpy").attr("reshape");
            Matrix<typename T::Scalar> intensity;
            map.renderMap(t, theta, res, intensity);
            return reshape(intensity.template cast<double>(), 
                           py::make_tuple(res, res));
        }, 
        docstrings::Map::render,
        "t"_a=0.0, "theta"_a=0.0, "res"_a=300);
#endif

#if defined(_STARRY_SINGLECOL_)
    // Load an image from a file
    PyMap.def(
        "load_image", [](
            Map<T>& map,
            std::string image,
            int lmax,
            bool normalize,
            int sampling_factor
        ) {
            map.loadImage(image, lmax, normalize, sampling_factor);
        },
        docstrings::Map::load_image,
        "image"_a, "lmax"_a=-1, "normalize"_a=true, "sampling_factor"_a=8);
#else
    // Load an image from a file
    PyMap.def(
        "load_image", [](
            Map<T>& map,
            std::string image,
            int lmax,
            int col,
            bool normalize,
            int sampling_factor
        ) {
            map.loadImage(image, lmax, col, normalize, sampling_factor);
        },
        docstrings::Map::load_image,
        "image"_a, "lmax"_a=-1, "col"_a=-1, "normalize"_a=true, "sampling_factor"_a=8);
#endif

#if defined(_STARRY_STATIC_)
    // Compute the intensity
    PyMap.def("__call__", intensity<T>(), docstrings::Map::call, 
              "theta"_a=0.0, "x"_a=0.0, "y"_a=0.0);
#else
    // Compute the intensity
    PyMap.def("__call__", intensity<T>(),  docstrings::Map::call, "t"_a=0.0, 
              "theta"_a=0.0, "x"_a=0.0, "y"_a=0.0);
#endif

#if defined(_STARRY_STATIC_)
    // Compute the flux
    PyMap.def("flux", flux<T>(), docstrings::Map::flux, "theta"_a=0.0, "xo"_a=0.0, 
              "yo"_a=0.0, "zo"_a=1.0, "ro"_a=0.0, "gradient"_a=false);
#else
    // Compute the flux
    PyMap.def("flux", flux<T>(), docstrings::Map::flux, "t"_a=0.0,"theta"_a=0.0, "xo"_a=0.0, 
              "yo"_a=0.0, "zo"_a=1.0, "ro"_a=0.0, "gradient"_a=false);
#endif

#if defined(_STARRY_DEFAULT_) && defined(_STARRY_DOUBLE_)
    // Compute the MAP map coefficients
    PyMap.def(
        "MAP", [](
            Map<T>& map,
            py::array_t<double>& flux_,
            py::array_t<double>& flux_err_,
            py::array_t<double>& theta_,
            py::array_t<double>& xo_,
            py::array_t<double>& yo_,
            py::array_t<double>& zo_,
            py::array_t<double>& ro_,
            py::array_t<double>& L_
        ) {
            // Map the flux to an Eigen type and figure
            // out the size of our vectors
            py::buffer_info buf = flux_.request();
            assert(buf.ndim == 1);
            py::ssize_t nt = buf.size;
            double *ptr = (double *) buf.ptr;
            Eigen::Map<Vector<double>> flux(ptr, nt, 1);

            // The remaining vectors/matrices
            Eigen::Map<Vector<double>> flux_err(NULL, nt, 1);
            Vector<double> tmp_flux_err;
            buf = flux_err_.request();
            ptr = (double *) buf.ptr;
            if (buf.ndim == 0) {
                tmp_flux_err = ptr[0] * Vector<double>::Ones(nt);
                new (&flux_err) Eigen::Map<Vector<double>>(&tmp_flux_err(0), nt, 1);
            } else if ((buf.ndim == 1) && (buf.size == nt)) {
                new (&flux_err) Eigen::Map<Vector<double>>(ptr, nt, 1);
            } else {
                throw errors::ShapeError("Vector `flux_err` has the incorrect shape.");
            }

            Eigen::Map<Vector<double>> theta(NULL, nt, 1);
            Vector<double> tmp_theta;
            buf = theta_.request();
            ptr = (double *) buf.ptr;
            if (buf.ndim == 0) {
                tmp_theta = ptr[0] * Vector<double>::Ones(nt);
                new (&theta) Eigen::Map<Vector<double>>(&tmp_theta(0), nt, 1);
            } else if ((buf.ndim == 1) && (buf.size == nt)) {
                new (&theta) Eigen::Map<Vector<double>>(ptr, nt, 1);
            } else {
                throw errors::ShapeError("Vector `theta` has the incorrect shape.");
            }

            Eigen::Map<Vector<double>> xo(NULL, nt, 1);
            Vector<double> tmp_xo;
            buf = xo_.request();
            ptr = (double *) buf.ptr;
            if (buf.ndim == 0) {
                tmp_xo = ptr[0] * Vector<double>::Ones(nt);
                new (&xo) Eigen::Map<Vector<double>>(&tmp_xo(0), nt, 1);
            } else if ((buf.ndim == 1) && (buf.size == nt)) {
                new (&xo) Eigen::Map<Vector<double>>(ptr, nt, 1);
            } else {
                throw errors::ShapeError("Vector `xo` has the incorrect shape.");
            }

            Eigen::Map<Vector<double>> yo(NULL, nt, 1);
            Vector<double> tmp_yo;
            buf = yo_.request();
            ptr = (double *) buf.ptr;
            if (buf.ndim == 0) {
                tmp_yo = ptr[0] * Vector<double>::Ones(nt);
                new (&yo) Eigen::Map<Vector<double>>(&tmp_yo(0), nt, 1);
            } else if ((buf.ndim == 1) && (buf.size == nt)) {
                new (&yo) Eigen::Map<Vector<double>>(ptr, nt, 1);
            } else {
                throw errors::ShapeError("Vector `yo` has the incorrect shape.");
            }

            Eigen::Map<Vector<double>> zo(NULL, nt, 1);
            Vector<double> tmp_zo;
            buf = zo_.request();
            ptr = (double *) buf.ptr;
            if (buf.ndim == 0) {
                tmp_zo = ptr[0] * Vector<double>::Ones(nt);
                new (&zo) Eigen::Map<Vector<double>>(&tmp_zo(0), nt, 1);
            } else if ((buf.ndim == 1) && (buf.size == nt)) {
                new (&zo) Eigen::Map<Vector<double>>(ptr, nt, 1);
            } else {
                throw errors::ShapeError("Vector `zo` has the incorrect shape.");
            }

            Eigen::Map<Vector<double>> ro(NULL, nt, 1);
            Vector<double> tmp_ro;
            buf = ro_.request();
            ptr = (double *) buf.ptr;
            if (buf.ndim == 0) {
                tmp_ro = ptr[0] * Vector<double>::Ones(nt);
                new (&ro) Eigen::Map<Vector<double>>(&tmp_ro(0), nt, 1);
            } else if ((buf.ndim == 1) && (buf.size == nt)) {
                new (&ro) Eigen::Map<Vector<double>>(ptr, nt, 1);
            } else {
                throw errors::ShapeError("Vector `ro` has the incorrect shape.");
            }

            Eigen::Map<Matrix<double>> L(NULL, map.N, map.N);
            Matrix<double> tmp_L;
            buf = L_.request();
            ptr = (double *) buf.ptr;
            if (buf.ndim == 0) {
                tmp_L = ptr[0] * Matrix<double>::Identity(map.N, map.N);
                new (&L) Eigen::Map<Matrix<double>>(&tmp_L(0), map.N, map.N);
            } else if ((buf.ndim == 1) && (buf.size == map.N)) {
                Eigen::Map<Vector<double>> L_diag(ptr, nt, 1);
                tmp_L = Matrix<double>(L_diag.asDiagonal());
                new (&L) Eigen::Map<Matrix<double>>(ptr, nt, 1);
            } else if ((buf.ndim == 2) && (buf.shape[0] == map.N) && (buf.shape[1] == map.N)) {
                new (&L) Eigen::Map<Matrix<double>>(ptr, map.N, map.N);
            } else {
                throw errors::ShapeError("Matrix `L` has the incorrect shape.");
            }

            //
            Vector<double> yhat(map.N, 1);
            Matrix<double> yvar(map.N, map.N);
            map.computeMaxLikeMap(flux, flux_err, theta, xo, yo, zo, ro, L, yhat, yvar);

        },
        "flux"_a, "flux_err"_a, "theta"_a=0.0, "xo"_a=0.0, "yo"_a=0.0, 
        "zo"_a=0.0, "ro"_a=0.0, "L"_a=0.0);
#endif

// Code version
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

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

#ifdef STARRY_KEEP_DFDU_AS_DFDG
            flags["STARRY_KEEP_DFDU_AS_DFDG"] = STARRY_KEEP_DFDU_AS_DFDG;
#else
            flags["STARRY_KEEP_DFDU_AS_DFDG"] = 0;
#endif

#ifdef STARRY_O
            flags["STARRY_O"] = STARRY_O;
#else
            flags["STARRY_O"] = py::none();
#endif

#ifdef STARRY_DEBUG
            flags["STARRY_DEBUG"] = STARRY_DEBUG;
#else
            flags["STARRY_DEBUG"] = 0;
#endif

            return flags;

    }, docstrings::Map::compile_flags);

}
