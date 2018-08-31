#ifdef STARRY_DEBUG
#undef NDEBUG
#endif

#include <pybind11/pybind11.h>
#include <stdlib.h>
#include "utils.h"
#include "pybind_interface.h"
#include "docstrings.h"
namespace py = pybind11;

PYBIND11_MODULE(_starry2, m) {

    using utils::Matrix;
    using utils::Vector;
    using utils::VectorT;
    using utils::Multi;
    using pybind_interface::bindMap;
    using namespace pybind11::literals;

    // Disable auto signatures
    py::options options;
    options.disable_function_signatures();

    // Add the top-level doc
    m.doc() = docstrings::starry::doc;

    // Create the four possible Map classes
    auto MapDoubleMono = bindMap<Vector<double>>(m, "MapDoubleMono");
    auto MapMultiMono = bindMap<Vector<Multi>>(m, "MapMultiMono");
    auto MapDoubleSpectral = bindMap<Matrix<double>>(m, "MapDoubleSpectral");
    auto MapMultiSpectral = bindMap<Matrix<Multi>>(m, "MapMultiSpectral");

    // The user-facing Map class factory
    m.def("Map",
          [MapDoubleMono, MapMultiMono, MapDoubleSpectral, MapMultiSpectral]
          (const int lmax=2, const int nwav=1, const bool multi=false) {
        if ((nwav == 1) && (!multi)) {
            return MapDoubleMono(lmax, nwav);
        } else if ((nwav == 1) && (multi)) {
            return MapMultiMono(lmax, nwav);
        } else if ((nwav > 1) && (!multi)) {
            return MapDoubleSpectral(lmax, nwav);
        } else if ((nwav > 1) && (multi)) {
            return MapMultiSpectral(lmax, nwav);
        } else {
            throw errors::ValueError("Invalid argument(s) to `Map`.");
        }
    }, docstrings::Map::doc, "lmax"_a=2, "nwav"_a=1, "multi"_a=false);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

}
