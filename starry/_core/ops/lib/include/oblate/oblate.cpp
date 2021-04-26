// Enable debug mode?
#ifdef STARRY_DEBUG
#undef NDEBUG
#endif

// Includes
#include "../utils.h"
#include "geometry.h"
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
  using namespace starry::oblate::occultation;
  using namespace starry::oblate::geometry;

  m.def("sT", [](const int &deg, const double &bo_, const double &ro_,
                 const double &f_, const double &theta_, const int &nruns) {
    using A = ADScalar<double, 0>;
    A bo, ro, f, theta;
    bo.value() = bo_;
    ro.value() = ro_;
    f.value() = f_;
    theta.value() = theta_;
    auto occ = Occultation<double, 0>(deg);

    for (int n = 0; n < nruns; ++n)
      occ.compute(bo, ro, f, theta);

    Vector<double> sT_value(occ.sT.size());
    for (int n = 0; n < occ.sT.size(); ++n)
      sT_value(n) = occ.sT(n).value();
    return sT_value;
  });

  m.def("dsT", [](const int &deg, const double &bo_, const double &ro_,
                  const double &f_, const double &theta_, const int &nruns) {
    using A = ADScalar<double, 4>;
    A bo, ro, f, theta;
    bo.value() = bo_;
    bo.derivatives() = Vector<double>::Unit(4, 0);
    ro.value() = ro_;
    ro.derivatives() = Vector<double>::Unit(4, 1);
    f.value() = f_;
    f.derivatives() = Vector<double>::Unit(4, 2);
    theta.value() = theta_;
    theta.derivatives() = Vector<double>::Unit(4, 3);
    auto occ = Occultation<double, 4>(deg);

    for (int n = 0; n < nruns; ++n)
      occ.compute(bo, ro, f, theta);

    Vector<double> sT_value(occ.sT.size());
    Vector<double> Dbo(occ.sT.size());
    Vector<double> Dro(occ.sT.size());
    Vector<double> Df(occ.sT.size());
    Vector<double> Dtheta(occ.sT.size());
    for (int n = 0; n < occ.sT.size(); ++n) {
      Dbo(n) = occ.sT(n).derivatives()(0);
      Dro(n) = occ.sT(n).derivatives()(1);
      Df(n) = occ.sT(n).derivatives()(2);
      Dtheta(n) = occ.sT(n).derivatives()(3);
    }
    return py::make_tuple(Dbo, Dro, Df, Dtheta);
  });

  m.def("angles", [](const double &bo_, const double &ro_, const double &f_,
                     const double &theta_) {
    using A = ADScalar<double, 0>;
    A bo, ro, f, theta, phi1, phi2, xi1, xi2;
    bo.value() = bo_;
    ro.value() = ro_;
    f.value() = f_;
    theta.value() = theta_;
    get_angles(bo, ro, f, theta, phi1, phi2, xi1, xi2);
    return py::make_tuple(phi1.value(), phi2.value(), xi1.value(), xi2.value());
  });
}