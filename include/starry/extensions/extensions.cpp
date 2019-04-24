/**
\file extensions.cpp
\brief Python bindings for custom C++ extensions to the code.

*/

// Enable debug mode?
#ifdef STARRY_DEBUG
#undef NDEBUG
#endif

// Enable the Python interface
#ifndef STARRY_ENABLE_PYTHON_INTERFACE
#define STARRY_ENABLE_PYTHON_INTERFACE
#endif

//! Import pybind11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
using namespace pybind11::literals;

//! Import starry
#include <starry/starry.h>
#include <starry/wigner.h>
#include "extensions.h"
#include "docstrings.h"
using namespace starry::utils;
using namespace starry::extensions;

//! Register the Python module
PYBIND11_MODULE(
    _starry_extensions, 
    m
) {

    // Disable docstring function signatures
    py::options options;
    options.disable_function_signatures();

    // Axis-angle rotation matrix in 3-space
    m.def(
        "RAxisAngle", [](
            const UnitVector<double>& axis,
            const double& angle
        ) {
            return starry::wigner::AxisAngle(axis.normalized(), 
                                             pi<double>() / 180. * angle);
        }, 
        "axis"_a, "angle"_a, docstrings::extensions::RAxisAngle);

}