// This throws Assertion errors if we've messed up
// any Eigen operations. Uncomment for debugging.
// #undef NDEBUG
#include <pybind11/pybind11.h>
#include <stdlib.h>
#include "utils.h"
#include "pybind_interface.h"
#include "docstrings.h"

using namespace std;
using namespace pybind11::literals;
using namespace docstrings;
namespace py = pybind11;

PYBIND11_MODULE(multi, m) {

    // Disable auto signatures
    py::options options;
    options.disable_function_signatures();

    // starry.grad
    docs<Multi> docs_multi;
    add_starry(m, docs_multi);

}
