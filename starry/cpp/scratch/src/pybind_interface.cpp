#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>
#include <iostream>
#include <iomanip>
#include "ellip.h"
#include "basis.h"

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
        elliptic functions
        ------------------

        .. currentmodule:: starry

        .. autosummary::
           :toctree: _generate

           K
           E
           PI
           poly

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

    m.def("poly",
        [] (
            const Eigen::Matrix<double, Eigen::Dynamic, 1>& p,
            const double& x, const double& y
        ) {
            int lmax = floor(sqrt((double)p.size()) - 1);
            return basis::poly(lmax, p, x, y);
        },
    R"pbdoc(
        Evaluate a polynomial vector `p` at a given (x, y) coordinate.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
