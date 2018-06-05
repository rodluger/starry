// DEBUG: This will throw Assertion errors if we've messed up
// any Eigen operations. Remove this once the code is stable.
#undef NDEBUG
#include <pybind11/pybind11.h>
#include <stdlib.h>
#include "utils.h"
#include "pybind_interface.h"
#include "docstrings.h"

using namespace std;
using namespace pybind11::literals;
using namespace docstrings;
namespace py = pybind11;
MathUtils math;

PYBIND11_MODULE(starry, m) {

    // Disable auto signatures
    py::options options;
    options.disable_function_signatures();

    // starry
    docs<double> docs_starry;
    add_starry(m, docs_starry);

    // starry.grad
    docs<Grad> docs_grad;
    auto mgrad = m.def_submodule("grad");
    add_starry(mgrad, docs_grad);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

}
