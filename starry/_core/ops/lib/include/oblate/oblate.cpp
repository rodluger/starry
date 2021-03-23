// Enable debug mode?
#ifdef STARRY_DEBUG
#undef NDEBUG
#endif

// Includes
#include "../utils.h"
#include "occultation.h"
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

// Register the Python module
PYBIND11_MODULE(_c_ops, m) {
  using namespace starry::utils;
  using namespace starry::oblate::ellip;
  using namespace starry::oblate::occultation;
  m.def("E", [](const double &k2, const Pair<double> &phi) {
    using A = ADScalar<double, 3>;
    A k2_ad;
    k2_ad.value() = k2;
    k2_ad.derivatives() = Vector<double>::Unit(3, 0);
    Pair<A> phi_ad;
    phi_ad(0).value() = phi(0);
    phi_ad(0).derivatives() = Vector<double>::Unit(3, 1);
    phi_ad(1).value() = phi(1);
    phi_ad(1).derivatives() = Vector<double>::Unit(3, 2);
    auto integrals = IncompleteEllipticIntegrals<double, 3>(k2_ad, phi_ad);
    return py::make_tuple(integrals.E.value(), integrals.E.derivatives());
  });

  m.def("sT", [](const int &deg, const double &bo_, const double &ro_,
                 const double &f_, const double &theta_, const double &phi1_,
                 const double &phi2_, const double &xi1_, const double &xi2_,
                 const int &nruns) {
    using A = ADScalar<double, 0>;
    A bo, ro, f, theta, phi1, phi2, xi1, xi2;
    bo.value() = bo_;
    ro.value() = ro_;
    f.value() = f_;
    theta.value() = theta_;
    phi1.value() = phi1_;
    phi2.value() = phi2_;
    xi1.value() = xi1_;
    xi2.value() = xi2_;
    auto occ = Occultation<double, 0>(deg);

    for (int n = 0; n < nruns; ++n)
      occ.compute(bo, ro, f, theta, phi1, phi2, xi1, xi2);

    Vector<double> sT_value(occ.sT.size());
    for (int n = 0; n < occ.sT.size(); ++n)
      sT_value(n) = occ.sT(n).value();
    return sT_value;
  });
}