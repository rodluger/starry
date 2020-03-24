/**
\file interface.cpp
\brief Defines the entry point for the C++ API.

*/

// Enable debug mode?
#ifdef STARRY_DEBUG
#undef NDEBUG
#endif

// Includes
#include <pybind11/eigen.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include "ops.h"
#include "sturm.h"
#include "utils.h"
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
  Ops.def(py::init<int, int, int, int>());

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
  Ops.def_property_readonly(
      "drorder", [](starry::Ops<Scalar> &ops) { return ops.drorder; });

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

  // Rotation solution in reflected light
  Ops.def("rTReflected",
          [](starry::Ops<Scalar> &ops, const Vector<double> &bterm) {
            size_t npts = size_t(bterm.size());
            Matrix<double, RowMajor> rT(npts, ops.N);
            for (size_t n = 0; n < npts; ++n) {
              ops.RP.compute(static_cast<Scalar>(bterm(n)));
              rT.row(n) = ops.RP.rT.template cast<double>();
            }
            return rT;
          });

  // Gradient of rotation solution in reflected light
  Ops.def("rTReflected", [](starry::Ops<Scalar> &ops,
                            const Vector<double> &bterm,
                            const Matrix<double, RowMajor> &brT) {
    size_t npts = size_t(bterm.size());
    Vector<double> bb(npts);
    for (size_t n = 0; n < npts; ++n) {
      bb(n) = static_cast<double>(ops.RP.compute(
          static_cast<Scalar>(bterm(n)), brT.row(n).template cast<Scalar>()));
    }
    return bb;
  });

  // Occultation in reflected light
  Ops.def("sTReflected", [](starry::Ops<Scalar> &ops, const Vector<double>& b_, 
                            const Vector<double>& theta_, const Vector<double>& bo_, 
                            const double& ro_) {
      
      // Total number of terms in `s^T`
      int K = b_.size();
      int N = (ops.ydeg + 1) * (ops.ydeg + 1);
      
      // Seed the derivatives. We'll compute them using forward
      // diff and store them for the backprop call.
      // TODO: Code up the backprop gradients for these occultations.
      // This will likely be quite hard, but should speed this up
      // quite a bit.
      ADScalar<Scalar, 4> b, theta, bo, ro;
      b.derivatives() = Vector<Scalar>::Unit(4, 0);
      theta.derivatives() = Vector<Scalar>::Unit(4, 1);
      bo.derivatives() = Vector<Scalar>::Unit(4, 2);
      ro.derivatives() = Vector<Scalar>::Unit(4, 3);
      ro.value() = ro_;
      
      // The output
      Matrix<double> result(K, N);
      ops.RO.ddb.resize(K, N);
      ops.RO.ddtheta.resize(K, N);
      ops.RO.ddbo.resize(K, N);
      ops.RO.ddro.resize(K, N);
      
      // Loop through the timeseries
      for (int k = 0; k < K; ++k) {
        
        // Compute sT for this timestep
        b.value() = static_cast<Scalar>(b_(k));
        theta.value() = static_cast<Scalar>(theta_(k));
        bo.value() = static_cast<Scalar>(bo_(k));
        ops.RO.compute(b, theta, bo, ro);

        // Process the ADScalar
        for (int n = 0; n < N; ++n) {
            result(k, n) = static_cast<double>(ops.RO.sT(n).value());
            ops.RO.ddb(k, n) = static_cast<double>(ops.RO.sT(n).derivatives()(0));
            ops.RO.ddtheta(k, n) = static_cast<double>(ops.RO.sT(n).derivatives()(1));
            ops.RO.ddbo(k, n) = static_cast<double>(ops.RO.sT(n).derivatives()(2));
            ops.RO.ddro(k, n) = static_cast<double>(ops.RO.sT(n).derivatives()(3));
        }

      }

      // Return just the value
      return result;

  });

  // Occultation in reflected light (gradient)
  Ops.def("sTReflected", [](starry::Ops<Scalar> &ops, const Vector<double>& b_, 
                            const Vector<double>& theta_, const Vector<double>& bo_, 
                            const double& ro_, const Matrix<double>& bsT) {

      // Total number of terms in `s^T`
      int K = b_.size();
      int N = (ops.ydeg + 1) * (ops.ydeg + 1);

      // The output
      Vector<double> bb(K), btheta(K), bbo(K);
      bb.setZero();
      btheta.setZero();
      bbo.setZero();
      double bro = 0.0;

      // Check that we've called the forward pass
      if ((ops.RO.ddb.rows() != K) || (ops.RO.ddb.cols() != N)) {
        throw std::runtime_error("Must call the forward pass of `sTReflected` first!");
      }

      // Loop through the timeseries
      for (int k = 0; k < K; ++k) {
        
        // Process the ADScalar
        for (int n = 0; n < N; ++n) {
            bb(k) += ops.RO.ddb(k, n) * bsT(k, n);
            btheta(k) += ops.RO.ddtheta(k, n) * bsT(k, n);
            bbo(k) += ops.RO.ddbo(k, n) * bsT(k, n);
            bro += ops.RO.ddro(k, n) * bsT(k, n);
        }

      }

      // Return the derivs
      return py::make_tuple(bb, btheta, bbo, bro);

  });

  // Rotation solution in emitted light dotted into Ylm space
  Ops.def_property_readonly("rTA1", [](starry::Ops<Scalar> &ops) {
    return ops.B.rTA1.template cast<double>();
  });

  // Polynomial basis at a vector of points
  Ops.def("pT", [](starry::Ops<Scalar> &ops, const RowVector<double> &x,
                   const RowVector<double> &y, const RowVector<double> &z) {
    ops.B.computePolyBasis(x.template cast<Scalar>(), y.template cast<Scalar>(),
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
  Ops.def(
      "spotYlm", [](starry::Ops<Scalar> &ops, const RowVector<Scalar> &amp,
                    const Scalar &sigma, const Scalar &lat, const Scalar &lon) {
        return ops
            .spotYlm(amp.template cast<double>(), static_cast<Scalar>(sigma),
                     static_cast<Scalar>(lat), static_cast<Scalar>(lon))
            .template cast<double>();
      });

  // Gradient of the Ylm expansion of a gaussian spot
  Ops.def(
      "spotYlm", [](starry::Ops<Scalar> &ops, const RowVector<Scalar> &amp,
                    const Scalar &sigma, const Scalar &lat, const Scalar &lon, 
                    const Matrix<double> &by) {
        ops.spotYlm(amp.template cast<double>(), static_cast<Scalar>(sigma),
                     static_cast<Scalar>(lat), static_cast<Scalar>(lon), 
                     by.template cast<Scalar>());
        return py::make_tuple(ops.bamp.template cast<double>(),
                              static_cast<double>(ops.bsigma),
                              static_cast<double>(ops.blat),
                              static_cast<double>(ops.blon));
      });

  // Differential rotation operator (matrices)
  Ops.def("tensordotD", [](starry::Ops<Scalar> &ops, const Matrix<double> &M,
                           const Vector<double> &wta) {
    ops.D.tensordotD(M.template cast<Scalar>(), wta.template cast<Scalar>());
    return ops.D.tensordotD_result.template cast<double>();
  });

  // Differential rotation operator (vectors)
  Ops.def("tensordotD", [](starry::Ops<Scalar> &ops, const RowVector<double> &M,
                           const Vector<double> &wta) {
    ops.D.tensordotD(M.template cast<Scalar>(), wta.template cast<Scalar>());
    return ops.D.tensordotD_result.template cast<double>();
  });

  // Gradient of differential rotation operator (vectors)
  Ops.def(
      "tensordotD", [](starry::Ops<Scalar> &ops, const RowVector<double> &M,
                       const Vector<double> &wta, const Matrix<double> &bMD) {
        ops.D.tensordotD(M.template cast<Scalar>(), wta.template cast<Scalar>(),
                         bMD.template cast<Scalar>());
        return py::make_tuple(ops.D.tensordotD_bM.template cast<double>(),
                              ops.D.tensordotD_bwta.template cast<double>());
      });

  // Gradient of differential rotation operator (matrices)
  Ops.def(
      "tensordotD", [](starry::Ops<Scalar> &ops, const Matrix<double> &M,
                       const Vector<double> &wta, const Matrix<double> &bMD) {
        ops.D.tensordotD(M.template cast<Scalar>(), wta.template cast<Scalar>(),
                         bMD.template cast<Scalar>());
        return py::make_tuple(ops.D.tensordotD_bM.template cast<double>(),
                              ops.D.tensordotD_bwta.template cast<double>());
      });

  // Sturm's theorem to get number of poly roots between `a` and `b`
  m.def("nroots",
        [](const Vector<double> &p, const double &a, const double &b) {
          return starry::sturm::polycountroots(p.template cast<Scalar>(),
                                               static_cast<Scalar>(a),
                                               static_cast<Scalar>(b));
        });
}