#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>
#include <iostream>
#include "ellip.h"
#include "basis.h"
#include "fact.h"

using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;

struct ndarray {
    ndarray(py::array_t<double, py::array::c_style>& arr) {
        auto buf = arr.request();
        if (buf.ndim != 1) throw std::runtime_error("invalid array");
        size = buf.size;
        ptr = (double*)buf.ptr;
    }
    int size = 0;
    double* ptr = NULL;
};

PYBIND11_MODULE(starry, m) {
    m.doc() = R"pbdoc(
        starry
        ------

        .. currentmodule:: starry

        .. autosummary::
           :toctree: _generate

           K
           E
           PI
           poly

    )pbdoc";

    // Utilities
    m.def("factorial", [] (int n) { return fact::factorial(n); },
    R"pbdoc(
        Factorial of a non-negative integer.
    )pbdoc", "n"_a);

    m.def("half_factorial", [] (int n) { return fact::half_factorial(n); },
    R"pbdoc(
        Factorial of `n` / 2.
    )pbdoc", "n"_a);

    // Elliptic functions

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

    // Basis functions

    py::class_<basis::Map<double>>(m, "Map")
        .def(py::init<Eigen::Matrix<double, Eigen::Dynamic, 1>&>())
        .def(py::init<int>())
        .def("evaluate", &basis::Map<double>::evaluate);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
