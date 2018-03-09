#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include "ellip.h"
#include "basis.h"

using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;

PYBIND11_MODULE(starry, m) {
    m.doc() = R"pbdoc(
        elliptic functions
        ------------------

        .. currentmodule:: starry

        .. autosummary::
           :toctree: _generate

           K
           E
           PI

    )pbdoc";

    m.def("K", [] (double ksq) { return ellip::K(ksq); },
    R"pbdoc(
        Complete elliptic integral of the first kind.
    )pbdoc", "ksq"_a);

    m.def("E", [] (double ksq) { return ellip::E(ksq); },
    R"pbdoc(
        Complete elliptic integral of the second kind.
    )pbdoc", "ksq"_a);

    m.def("PI", [] (double n, double ksq) { return ellip::PI(n, ksq); },
    R"pbdoc(
        Complete elliptic integral of the third kind.
    )pbdoc", "n"_a, "ksq"_a);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
