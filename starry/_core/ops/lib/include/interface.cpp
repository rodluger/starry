/**
\file interface.cpp
\brief Defines the entry point for the C++ API.

*/

// Enable debug mode?
#ifdef STARRY_DEBUG
#undef NDEBUG
#endif

// Includes
#include "basis.h"
#include "ops.h"
#include "reflected/scatter.h"
#include "sturm.h"
#include "utils.h"
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

// Multiprecision?
#if STARRY_NDIGITS > 16
#define STARRY_MULTI
using Scalar = Multi;
#else
using Scalar = double;
#endif

// Register the Python module
PYBIND11_MODULE(_c_ops, m) {
  // Import some useful stuff
  using namespace starry::utils;

  // Declare the Ops class
  py::class_<starry::Ops<Scalar>> Ops(m, "Ops");

  // Constructor
  Ops.def(py::init<int, int, int>());

  // Enable pickling
  Ops.def(py::pickle(
      [](const starry::Ops<Scalar> &ops) {
        // __getstate__
        return py::make_tuple(ops.ydeg, ops.udeg, ops.fdeg);
      },
      [](py::tuple t) {
        // __setstate__
        if (t.size() != 3)
          throw std::runtime_error("Invalid state!");
        starry::Ops<Scalar> ops(t[0].cast<int>(), t[1].cast<int>(),
                                t[2].cast<int>());
        return ops;
      }));

  // Map dimensions
  Ops.def_property_readonly("ydeg",
                            [](starry::Ops<Scalar> &ops) { return ops.ydeg; });
  Ops.def_property_readonly("Ny",
                            [](starry::Ops<Scalar> &ops) { return ops.Ny; });
  Ops.def_property_readonly("udeg",
                            [](starry::Ops<Scalar> &ops) { return ops.udeg; });
  Ops.def_property_readonly("Nu",
                            [](starry::Ops<Scalar> &ops) { return ops.Nu; });
  Ops.def_property_readonly("fdeg",
                            [](starry::Ops<Scalar> &ops) { return ops.fdeg; });
  Ops.def_property_readonly("Nf",
                            [](starry::Ops<Scalar> &ops) { return ops.Nf; });
  Ops.def_property_readonly("deg",
                            [](starry::Ops<Scalar> &ops) { return ops.deg; });
  Ops.def_property_readonly("N",
                            [](starry::Ops<Scalar> &ops) { return ops.N; });

  // Occultation solution in emitted light
  Ops.def("sT", [](starry::Ops<Scalar> &ops, const Vector<double> &b,
                   const double &r) {
    size_t npts = size_t(b.size());
    Matrix<double, RowMajor> sT(npts, ops.N);
    for (size_t n = 0; n < npts; ++n) {
      ops.G.compute(static_cast<Scalar>(b(n)), static_cast<Scalar>(r));
      sT.row(n) = ops.G.sT.template cast<double>();
    }
    return sT;
  });

  // Gradient of occultation solution in emitted light
  Ops.def("sT", [](starry::Ops<Scalar> &ops, const Vector<double> &b,
                   const double &r, const Matrix<double, RowMajor> &bsT) {
    size_t npts = size_t(b.size());
    Vector<double> bb(npts);
    double br = 0.0;
    for (size_t n = 0; n < npts; ++n) {
      ops.G.template compute<true>(static_cast<Scalar>(b(n)),
                                   static_cast<Scalar>(r));
      bb(n) = static_cast<double>(
          ops.G.dsTdb.dot(bsT.row(n).template cast<Scalar>()));
      br += static_cast<double>(
          ops.G.dsTdr.dot(bsT.row(n).template cast<Scalar>()));
    }
    return py::make_tuple(bb, br);
  });

  // Change of basis matrix: Ylm to poly
  Ops.def_property_readonly("A1", [](starry::Ops<Scalar> &ops) {
#ifdef STARRY_MULTI
    return (ops.B.A1.template cast<double>()).eval();
#else
    return ops.B.A1;
#endif
  });

  // Augmented change of basis matrix: Ylm to poly
  Ops.def_property_readonly("A1Big", [](starry::Ops<Scalar> &ops) {
#ifdef STARRY_MULTI
    return (ops.B.A1_big.template cast<double>()).eval();
#else
    return ops.B.A1_big;
#endif
  });

  // Augmented change of basis matrix: poly to Ylm
  Ops.def_property_readonly("A1Inv", [](starry::Ops<Scalar> &ops) {
#ifdef STARRY_MULTI
    return (ops.B.A1Inv.template cast<double>()).eval();
#else
    return ops.B.A1Inv;
#endif
  });

  // Change of basis matrix: Ylm to greens
  Ops.def_property_readonly("A", [](starry::Ops<Scalar> &ops) {
#ifdef STARRY_MULTI
    return (ops.B.A.template cast<double>()).eval();
#else
    return ops.B.A;
#endif
  });

  // Rotation solution in emitted light
  Ops.def_property_readonly("rT", [](starry::Ops<Scalar> &ops) {
    return ops.B.rT.template cast<double>();
  });

  // Phase curve in reflected light (w/ fwd gradient)
  // NOTE: This vector is already weighted by the illumination.
  Ops.def("rTReflected", [](starry::Ops<Scalar> &ops, const Vector<double> &b_,
                            const double &sigr_) {
    // Total number of terms in `r^T`
    int K = b_.size();

    // Seed the derivatives. We'll compute them using forward
    // diff and return them for the backprop call.
    ADScalar<Scalar, 2> b, sigr;
    b.derivatives() = Vector<Scalar>::Unit(2, 0);
    sigr.derivatives() = Vector<Scalar>::Unit(2, 1);
    sigr.value() = sigr_;

    // The output
    Matrix<double> result(K, ops.N);
    Matrix<double> ddb(K, ops.N);
    Matrix<double> ddsigr(K, ops.N);

    // Loop through the timeseries
    for (int k = 0; k < K; ++k) {

      // Hack: deriv undefined for b = +/- 1 (not a numerical issue)
      if (b_(k) >= 1.0 - 1e-15) {
        b.value() = Scalar(1.0) - Scalar(1e-15);
      } else if (b_(k) <= -1.0 + 1e-15) {
        b.value() = Scalar(-1.0) + Scalar(1e-15);
      } else {
        b.value() = static_cast<Scalar>(b_(k));
      }

      // Compute rT for this timestep
      ops.RP.compute(b, sigr);

      // Process the ADScalar
      for (int n = 0; n < ops.N; ++n) {
        result(k, n) = static_cast<double>(ops.RP.rT(n).value());
        ddb(k, n) = static_cast<double>(ops.RP.rT(n).derivatives()(0));
        ddsigr(k, n) = static_cast<double>(ops.RP.rT(n).derivatives()(1));
      }
    }

    // Return the value & the forward derivs
    return py::make_tuple(result, ddb, ddsigr);
  });

  // Occultation in reflected light (w/ fwd gradient)
  // NOTE: This vector is already weighted by the illumination.
  Ops.def("sTReflected", [](starry::Ops<Scalar> &ops, const Vector<double> &b_,
                            const Vector<double> &theta_,
                            const Vector<double> &bo_, const double &ro_,
                            const double &sigr_) {
    // Total number of terms in `s^T`
    int K = b_.size();

    // Seed the derivatives. We'll compute them using forward
    // diff and return them for the backprop call.
    ADScalar<Scalar, 5> b, theta, bo, ro, sigr;
    b.derivatives() = Vector<Scalar>::Unit(5, 0);
    theta.derivatives() = Vector<Scalar>::Unit(5, 1);
    bo.derivatives() = Vector<Scalar>::Unit(5, 2);
    ro.derivatives() = Vector<Scalar>::Unit(5, 3);
    sigr.derivatives() = Vector<Scalar>::Unit(5, 4);
    ro.value() = ro_;
    sigr.value() = sigr_;

    // The output
    Matrix<double> result(K, ops.N);
    Matrix<double> ddb(K, ops.N);
    Matrix<double> ddtheta(K, ops.N);
    Matrix<double> ddbo(K, ops.N);
    Matrix<double> ddro(K, ops.N);
    Matrix<double> ddsigr(K, ops.N);

    // Loop through the timeseries
    for (int k = 0; k < K; ++k) {

      // Hack: deriv undefined for b = +/- 1 (not a numerical issue)
      if (b_(k) >= 1.0 - 1e-15) {
        b.value() = Scalar(1.0) - Scalar(1e-15);
      } else if (b_(k) <= -1.0 + 1e-15) {
        b.value() = Scalar(-1.0) + Scalar(1e-15);
      } else {
        b.value() = static_cast<Scalar>(b_(k));
      }

      theta.value() = static_cast<Scalar>(theta_(k));
      bo.value() = static_cast<Scalar>(bo_(k));

      // Compute sT for this timestep
      ops.RO.compute(b, theta, bo, ro, sigr);

      // Process the ADScalar
      for (int n = 0; n < ops.N; ++n) {
        result(k, n) = static_cast<double>(ops.RO.sT(n).value());
        ddb(k, n) = static_cast<double>(ops.RO.sT(n).derivatives()(0));
        ddtheta(k, n) = static_cast<double>(ops.RO.sT(n).derivatives()(1));
        ddbo(k, n) = static_cast<double>(ops.RO.sT(n).derivatives()(2));
        ddro(k, n) = static_cast<double>(ops.RO.sT(n).derivatives()(3));
        ddsigr(k, n) = static_cast<double>(ops.RO.sT(n).derivatives()(4));
      }
    }

    // Return the value & the forward derivs
    return py::make_tuple(result, ddb, ddtheta, ddbo, ddro, ddsigr);
  });

  // Rotation solution in emitted light dotted into Ylm space
  Ops.def_property_readonly("rTA1", [](starry::Ops<Scalar> &ops) {
    return ops.B.rTA1.template cast<double>();
  });

  // Polynomial basis at a vector of points
  Ops.def("pT", [](starry::Ops<Scalar> &ops, const int deg,
                   const RowVector<double> &x, const RowVector<double> &y,
                   const RowVector<double> &z) {
    ops.B.computePolyBasis(deg, x.template cast<Scalar>(),
                           y.template cast<Scalar>(),
                           z.template cast<Scalar>());
    return ops.B.pT.template cast<double>();
  });

  // Rotation dot product operator (vectors)
  Ops.def("dotR", [](starry::Ops<Scalar> &ops, const RowVector<double> &M,
                     const double &x, const double &y, const double &z,
                     const double &theta) {
    ops.W.dotR(M.template cast<Scalar>(), static_cast<Scalar>(x),
               static_cast<Scalar>(y), static_cast<Scalar>(z),
               static_cast<Scalar>(theta));
    return ops.W.dotR_result.template cast<double>();
  });

  // Rotation dot product operator (matrices)
  Ops.def("dotR",
          [](starry::Ops<Scalar> &ops, const Matrix<double> &M, const double &x,
             const double &y, const double &z, const double &theta) {
            ops.W.dotR(M.template cast<Scalar>(), static_cast<Scalar>(x),
                       static_cast<Scalar>(y), static_cast<Scalar>(z),
                       static_cast<Scalar>(theta));
            return ops.W.dotR_result.template cast<double>();
          });

  // Gradient of rotation dot product operator (vectors)
  Ops.def("dotR", [](starry::Ops<Scalar> &ops, const RowVector<double> &M,
                     const double &x, const double &y, const double &z,
                     const double &theta, const Matrix<double> &bMR) {
    ops.W.dotR(M.template cast<Scalar>(), static_cast<Scalar>(x),
               static_cast<Scalar>(y), static_cast<Scalar>(z),
               static_cast<Scalar>(theta), bMR.template cast<Scalar>());
    return py::make_tuple(ops.W.dotR_bM.template cast<double>(),
                          static_cast<double>(ops.W.dotR_bx),
                          static_cast<double>(ops.W.dotR_by),
                          static_cast<double>(ops.W.dotR_bz),
                          static_cast<double>(ops.W.dotR_btheta));
  });

  // Gradient of rotation dot product operator (matrices)
  Ops.def("dotR", [](starry::Ops<Scalar> &ops, const Matrix<double> &M,
                     const double &x, const double &y, const double &z,
                     const double &theta, const Matrix<double> &bMR) {
    ops.W.dotR(M.template cast<Scalar>(), static_cast<Scalar>(x),
               static_cast<Scalar>(y), static_cast<Scalar>(z),
               static_cast<Scalar>(theta), bMR.template cast<Scalar>());
    return py::make_tuple(ops.W.dotR_bM.template cast<double>(),
                          static_cast<double>(ops.W.dotR_bx),
                          static_cast<double>(ops.W.dotR_by),
                          static_cast<double>(ops.W.dotR_bz),
                          static_cast<double>(ops.W.dotR_btheta));
  });

  // Z rotation operator (vectors)
  Ops.def("tensordotRz", [](starry::Ops<Scalar> &ops,
                            const RowVector<double> &M,
                            const Vector<double> &theta) {
    ops.W.tensordotRz(M.template cast<Scalar>(), theta.template cast<Scalar>());
    return ops.W.tensordotRz_result.template cast<double>();
  });

  // Z rotation operator (matrices)
  Ops.def("tensordotRz", [](starry::Ops<Scalar> &ops, const Matrix<double> &M,
                            const Vector<double> &theta) {
    ops.W.tensordotRz(M.template cast<Scalar>(), theta.template cast<Scalar>());
    return ops.W.tensordotRz_result.template cast<double>();
  });

  // Gradient of Z rotation matrix (vectors)
  Ops.def("tensordotRz", [](starry::Ops<Scalar> &ops,
                            const RowVector<double> &M,
                            const Vector<double> &theta,
                            const Matrix<double> &bMRz) {
    ops.W.tensordotRz(M.template cast<Scalar>(), theta.template cast<Scalar>(),
                      bMRz.template cast<Scalar>());
    return py::make_tuple(ops.W.tensordotRz_bM.template cast<double>(),
                          ops.W.tensordotRz_btheta.template cast<double>());
  });

  // Gradient of Z rotation matrix (matrices)
  Ops.def("tensordotRz", [](starry::Ops<Scalar> &ops, const Matrix<double> &M,
                            const Vector<double> &theta,
                            const Matrix<double> &bMRz) {
    ops.W.tensordotRz(M.template cast<Scalar>(), theta.template cast<Scalar>(),
                      bMRz.template cast<Scalar>());
    return py::make_tuple(ops.W.tensordotRz_bM.template cast<double>(),
                          ops.W.tensordotRz_btheta.template cast<double>());
  });

  // Filter operator
  Ops.def("F", [](starry::Ops<Scalar> &ops, const Vector<double> &u,
                  const Vector<double> &f) {
    ops.F.computeF(u.template cast<Scalar>(), f.template cast<Scalar>());
    return ops.F.F.template cast<double>();
  });

  // Gradient of filter operator
  Ops.def("F", [](starry::Ops<Scalar> &ops, const Vector<double> &u,
                  const Vector<double> &f, const Matrix<double> &bF) {
    ops.F.computeF(u.template cast<Scalar>(), f.template cast<Scalar>(),
                   bF.template cast<Scalar>());
    return py::make_tuple(ops.F.bu.template cast<double>(),
                          ops.F.bf.template cast<double>());
  });

  // Compute the Ylm expansion of a gaussian spot
  Ops.def("spotYlm", [](starry::Ops<Scalar> &ops, const RowVector<Scalar> &amp,
                        const Scalar &sigma, const Scalar &lat,
                        const Scalar &lon) {
    return ops
        .spotYlm(amp.template cast<double>(), static_cast<Scalar>(sigma),
                 static_cast<Scalar>(lat), static_cast<Scalar>(lon))
        .template cast<double>();
  });

  // Gradient of the Ylm expansion of a gaussian spot
  Ops.def("spotYlm", [](starry::Ops<Scalar> &ops, const RowVector<Scalar> &amp,
                        const Scalar &sigma, const Scalar &lat,
                        const Scalar &lon, const Matrix<double> &by) {
    ops.spotYlm(amp.template cast<double>(), static_cast<Scalar>(sigma),
                static_cast<Scalar>(lat), static_cast<Scalar>(lon),
                by.template cast<Scalar>());
    return py::make_tuple(
        ops.bamp.template cast<double>(), static_cast<double>(ops.bsigma),
        static_cast<double>(ops.blat), static_cast<double>(ops.blon));
  });

  // Oren-Nayar (1994) illumination polynomial (reflected light)
  Ops.def("OrenNayarPolynomial",
          [](starry::Ops<Scalar> &ops, const Vector<double> &b,
             const Vector<double> &theta, const double &sigr) {
            int N = (STARRY_OREN_NAYAR_DEG + 1) * (STARRY_OREN_NAYAR_DEG + 1);
            Matrix<double> p(N, b.size());
            for (int i = 0; i < b.size(); ++i) {
              p.col(i) =
                  starry::reflected::scatter::OrenNayarPolynomial(
                      static_cast<Scalar>(b(i)), static_cast<Scalar>(theta(i)),
                      static_cast<Scalar>(sigr), ops.B)
                      .template cast<double>();
            }
            return p;
          });

  // Export the degree for access in theano
  m.attr("STARRY_OREN_NAYAR_DEG") = py::int_(STARRY_OREN_NAYAR_DEG);

  // Sturm's theorem to get number of poly roots between `a` and `b`
  m.def("nroots",
        [](const Vector<double> &p, const double &a, const double &b) {
          return starry::sturm::polycountroots(p.template cast<Scalar>(),
                                               static_cast<Scalar>(a),
                                               static_cast<Scalar>(b));
        });
}