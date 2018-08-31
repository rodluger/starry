#ifdef STARRY_DEBUG
#undef NDEBUG
#endif

#include <pybind11/pybind11.h>
#include <stdlib.h>
#include "utils.h"
#include "pybind_interface.h"
namespace py = pybind11;

PYBIND11_MODULE(_starry2, m) {

    using utils::Matrix;
    using utils::Vector;
    using utils::VectorT;
    using utils::Multi;

    // Disable auto signatures
    py::options options;
    options.disable_function_signatures();

    // starry
    pybind_interface::add_starry<Vector<double>>(m);

    // starry.multi
    auto mmulti = m.def_submodule("multi");
    pybind_interface::add_starry<Vector<Multi>>(mmulti);

    // starry.spectral
    auto mspectral = m.def_submodule("spectral");
    pybind_interface::add_starry<Matrix<double>>(mspectral);

    // starry.multispectral
    auto mmultispectral = m.def_submodule("multispectral");
    pybind_interface::add_starry<Matrix<Multi>>(mmultispectral);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

}
