#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include "ellip.h"
#include <iostream>
#include <iomanip>
using namespace std;

namespace py = pybind11;

double K(double ksq){
    return starry::K(ksq);
}

double E(double ksq){
    return starry::E(ksq);
}

double PI(double n, double ksq){
    return starry::PI(n, ksq);
}

double dKdk2(double ksq){
    typedef Eigen::Matrix<double, 1, 1> DerType;
    Eigen::AutoDiffScalar<DerType> z(ksq, 1, 0);
    return starry::K(z).derivatives().value();
}

double dEdk2(double ksq){
    typedef Eigen::Matrix<double, 1, 1> DerType;
    Eigen::AutoDiffScalar<DerType> z(ksq, 1, 0);
    return starry::E(z).derivatives().value();
}

PYBIND11_MODULE(starry, m) {
    m.doc() = R"pbdoc(
        starry
        ------

        .. currentmodule:: starry

        .. autosummary::
           :toctree: _generate

           K

    )pbdoc";

    m.def("K", &K,
    R"pbdoc(
        Complete elliptic integral of the first kind.
    )pbdoc");

    m.def("E", &E,
    R"pbdoc(
        Complete elliptic integral of the second kind.
    )pbdoc");

    m.def("PI", &PI,
    R"pbdoc(
        Complete elliptic integral of the third kind.
    )pbdoc");

    m.def("dKdk2", &dKdk2,
    R"pbdoc(
        Derivative of the complete elliptic integral of the first kind.
    )pbdoc");

    m.def("dEdk2", &dEdk2,
    R"pbdoc(
        Derivative of the complete elliptic integral of the second kind.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
