// DEBUG: This will throw Assertion errors if we've messed up
// any Eigen operations. Remove this once the code is stable.
#undef NDEBUG
#include <pybind11/pybind11.h>
#include <stdlib.h>

// MAGIC: Include our starry interface several times
#define MODULE_STARRY               1
#define MODULE_STARRY_GRAD          2
#define MODULE_STARRY_MULTI         3

// starry
#undef MODULE
#define MODULE                      MODULE_STARRY
#include "pybind_interface.h"

// starry.grad
#undef MODULE
#define MODULE                      MODULE_STARRY_GRAD
#include "pybind_interface.h"

using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;

PYBIND11_MODULE(starry, m) {

    // Disable auto signatures
    py::options options;
    options.disable_function_signatures();

    // starry
    add_starry(m);

    // starry.grad
    auto mgrad = m.def_submodule("grad");
    add_starry_grad(mgrad);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
