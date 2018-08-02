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

    // Disable auto signatures
    py::options options;
    options.disable_function_signatures();

    // starry
    docstrings::docs<double> docs_starry;
    pybind_interface::add_starry(m, docs_starry);

    // starry.multi
    docstrings::docs<utils::Multi> docs_multi;
    auto mmulti = m.def_submodule("multi");
    pybind_interface::add_starry(mmulti, docs_multi);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

}
