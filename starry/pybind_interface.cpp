#include <pybind11/pybind11.h>
#include <stdlib.h>

// MAGIC: Include our starry interface twice,
// once with no derivs and once with derivs.
#undef STARRY_AUTODIFF
//#include "pybind_interface.h"
#define STARRY_AUTODIFF
#include "pybind_interface.h"

using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;


PYBIND11_MODULE(starry, m) {

    // Disable auto signatures
    py::options options;
    options.disable_function_signatures();

    // starry
    //add_starry(m);

    // starry.grad
    auto mgrad = m.def_submodule("grad");
    add_starry_grad(mgrad);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
