#ifdef STARRY_DEBUG
#undef NDEBUG
#endif

// Default
#define STARRY_NAME _starry_mono_64
#define STARRY_TYPE Vector<double>

// Monochromatic, double precision
#ifdef STARRY_MONO_64
#undef STARRY_NAME
#undef STARRY_TYPE
#define STARRY_NAME _starry_mono_64
#define STARRY_TYPE Vector<double>
#endif

// Monochromatic, quadruple precision
#ifdef STARRY_MONO_128
#undef STARRY_NAME
#undef STARRY_TYPE
#define STARRY_NAME _starry_mono_128
#define STARRY_TYPE Vector<Multi>
#endif

// Spectral, double precision
#ifdef STARRY_SPECTRAL_64
#undef STARRY_NAME
#undef STARRY_TYPE
#define STARRY_NAME _starry_spectral_64
#define STARRY_TYPE Matrix<double>
#endif

// Spectral, quadruple precision
#ifdef STARRY_SPECTRAL_128
#undef STARRY_NAME
#undef STARRY_TYPE
#define STARRY_NAME _starry_spectral_128
#define STARRY_TYPE Matrix<Multi>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdlib.h>
#include "utils.h"
#include "pybind_interface.h"
#include "docstrings.h"
#include "kepler.h"
#include "maps.h"
namespace py = pybind11;


PYBIND11_MODULE(STARRY_NAME, m) {

    using utils::Matrix;
    using utils::Vector;
    using utils::Multi;
    using pybind_interface::bindMap;
    using pybind_interface::bindBody;
    using pybind_interface::bindPrimary;
    using pybind_interface::bindSecondary;
    using pybind_interface::bindSystem;
    using namespace pybind11::literals;

    py::options options;
    options.disable_function_signatures();
    m.doc() = docstrings::starry::doc;

    auto Map = bindMap<STARRY_TYPE>(m, "Map");
    auto mk = m.def_submodule("kepler", docstrings::kepler::doc);
    auto Body = bindBody<STARRY_TYPE>(mk, Map, "Body");
    auto Primary1 = bindPrimary<STARRY_TYPE>(mk, Body, "Primary");
    auto Secondary = bindSecondary<STARRY_TYPE>(mk, Body, "Secondary");
    auto System = bindSystem<STARRY_TYPE>(mk, "System");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

}
