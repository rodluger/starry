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
#include <starry2/starry2.h>
#include "extensions.h"
using namespace starry2::utils;
using namespace starry2::extensions;

//! Register the Python module
PYBIND11_MODULE(
    _starry_extensions, 
    m
) {

    // Add bindings for custom extensions here.

}