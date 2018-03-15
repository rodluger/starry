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

PYBIND11_MODULE(_starry, m) {
    m.doc() = R"pbdoc(
        The starry C++ library.
    )pbdoc";

    // Core Map class
    py::class_<maps::Map<double>>(m, "Map")
        .def(py::init<int>())
        .def("evaluate", py::vectorize(&maps::Map<double>::evaluate),
            R"pbdoc(
                Return the specific intensity at a point on the map.
            )pbdoc", "u"_a=maps::yhat, "theta"_a=0, "x"_a=0, "y"_a=0)
        .def("rotate", [](maps::Map<double> &map, Eigen::Matrix<double, 3, 1>& u,
                          double theta){return map.rotate(u, theta);})
        .def("flux", py::vectorize(&maps::Map<double>::flux),
            R"pbdoc(
                Return the total flux received by the observer.
            )pbdoc", "u"_a=maps::yhat, "theta"_a=0, "xo"_a=-INFINITY,
                     "yo"_a=-INFINITY, "ro"_a=1, "numerical"_a=false,
                     "tol"_a=1e-4)
        .def("get_coeff", &maps::Map<double>::get_coeff)
        .def("set_coeff", &maps::Map<double>::set_coeff)
        .def("reset", &maps::Map<double>::reset)
        .def_property_readonly("lmax", [](maps::Map<double> &map){return map.lmax;})
        .def_property_readonly("y", [](maps::Map<double> &map){return map.y;})
        .def_property_readonly("p", [](maps::Map<double> &map)
                               {map.update(true); return map.p;})
        .def_property_readonly("g", [](maps::Map<double> &map)
                               {map.update(true); return map.g;})
        .def("__repr__", [](maps::Map<double> &map) -> string {return map.repr();});

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

}
