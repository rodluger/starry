#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>
#include "ellip.h"
#include "maps.h"
#include "basis.h"
#include "fact.h"
#include "sqrtint.h"
#include "rotation.h"
#include "solver.h"

using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;

PYBIND11_MODULE(starry, m) {
    m.doc() = R"pbdoc(
        The starry C++ library.
    )pbdoc";

    // Core Map class
    py::class_<maps::Map<double>>(m, "Map")
        .def(py::init<int>())
        .def("evaluate", py::vectorize(&maps::Map<double>::evaluate))
        .def("rotate", [](maps::Map<double> &map, Eigen::Matrix<double, 3, 1>& u, double theta){return map.rotate(u, theta);})
        .def("flux", py::vectorize(&maps::Map<double>::flux),
            R"pbdoc(
                Return the total flux received by the observer.
            )pbdoc", "u"_a=maps::yhat, "theta"_a=0, "x0"_a=-INFINITY,
                     "y0"_a=-INFINITY, "r"_a=1, "numerical"_a=false,
                     "tol"_a=1e-4)
        .def("get_coeff", &maps::Map<double>::get_coeff)
        .def("set_coeff", &maps::Map<double>::set_coeff)
        .def("reset", &maps::Map<double>::reset)
        .def_property_readonly("y", [](const maps::Map<double> &map){return map.y;})
        .def_property_readonly("p", [](maps::Map<double> &map){map.update(true); return map.p;})
        .def_property_readonly("g", [](maps::Map<double> &map){map.update(true); return map.g;})
        .def("__repr__", [](maps::Map<double> &map) -> string {return map.repr();});

    // Utilities
    py::module m_utils = m.def_submodule("utils");

        m_utils.def("factorial", [] (int n) { return fact::factorial(n); },
        R"pbdoc(
            Factorial of a non-negative integer.
        )pbdoc", "n"_a);

        m_utils.def("half_factorial", [] (int n) { return fact::half_factorial(n); },
        R"pbdoc(
            Factorial of `n` / 2.
        )pbdoc", "n"_a);

        m_utils.def("sqrt_int", [] (int n) { return sqrtint::sqrt_int(n); },
        R"pbdoc(
            Square root of `n`.
        )pbdoc", "n"_a);

        m_utils.def("invsqrt_int", [] (int n) { return sqrtint::invsqrt_int(n); },
        R"pbdoc(
            Inverse of the square root of `n`.
        )pbdoc", "n"_a);

    // Elliptic functions
    py::module m_ellip = m.def_submodule("elliptic");

        m_ellip.def("K", [] (double ksq) { return ellip::K(ksq); },
        R"pbdoc(
            Complete elliptic integral of the first kind.
        )pbdoc", "ksq"_a);

        m_ellip.def("E", [] (double ksq) { return ellip::E(ksq); },
        R"pbdoc(
            Complete elliptic integral of the second kind.
        )pbdoc", "ksq"_a);

        m_ellip.def("PI", [] (double n, double ksq) { return ellip::PI(n, ksq); },
        R"pbdoc(
            Complete elliptic integral of the third kind.
        )pbdoc", "n"_a, "ksq"_a);


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

}
