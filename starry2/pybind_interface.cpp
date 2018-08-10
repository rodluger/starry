// This throws Assertion errors if we've messed up
// any Eigen operations. Uncomment for debugging.
#undef NDEBUG
#include <pybind11/pybind11.h>
#include <stdlib.h>
#include "utils.h"
#include "pybind_interface.h"
#include "docstrings.h"
namespace py = pybind11;

PYBIND11_MODULE(starry2, m) {

    using utils::Matrix;
    using utils::Vector;
    using utils::VectorT;
    using utils::Multi;

    // Disable auto signatures
    py::options options;
    options.disable_function_signatures();

    // starry
    docstrings::docs<STARRY_MODULE_MAIN> docs_starry;
    pybind_interface::add_starry<Vector<double>, double, double, STARRY_MODULE_MAIN>(m, docs_starry);

    // starry.multi
    docstrings::docs<STARRY_MODULE_MULTI> docs_multi;
    auto mmulti = m.def_submodule("multi");
    pybind_interface::add_starry<Vector<Multi>, Multi, Multi, STARRY_MODULE_MULTI>(mmulti, docs_multi);

    // starry.spectral
    docstrings::docs<STARRY_MODULE_SPECTRAL> docs_spectral;
    auto mspectral = m.def_submodule("spectral");
    pybind_interface::add_starry<Matrix<double>, Vector<double>, VectorT<double>, STARRY_MODULE_SPECTRAL>(mspectral, docs_spectral);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

}
