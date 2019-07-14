/**
\file interface.cpp
\brief Defines the entry point for the C++ API.

*/

// Enable debug mode?
#ifdef STARRY_DEBUG
#   undef NDEBUG
#endif

// Includes
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include "utils.h"
#include "ops.h"
namespace py = pybind11;

// Multiprecision?
#if STARRY_NDIGITS > 16
#   define STARRY_MULTI
    using Scalar = Multi;
#else
    using Scalar = double;
#endif

// Register the Python module
PYBIND11_MODULE(
    _c_ops, 
    m
) {

    // Import some useful stuff
    using namespace starry::utils;

    // Declare the Ops class
    py::class_<starry::Ops<Scalar>> Ops(m, "Ops");

    // Constructor
    Ops.def(py::init<int, int, int>());

    // Map dimensions
    Ops.def_property_readonly("ydeg", [](starry::Ops<Scalar>& ops){return ops.ydeg;});
    Ops.def_property_readonly("Ny", [](starry::Ops<Scalar>& ops){return ops.Ny;});
    Ops.def_property_readonly("udeg", [](starry::Ops<Scalar>& ops){return ops.udeg;});
    Ops.def_property_readonly("Nu", [](starry::Ops<Scalar>& ops){return ops.Nu;});
    Ops.def_property_readonly("fdeg", [](starry::Ops<Scalar>& ops){return ops.fdeg;});
    Ops.def_property_readonly("Nf", [](starry::Ops<Scalar>& ops){return ops.Nf;});
    Ops.def_property_readonly("deg", [](starry::Ops<Scalar>& ops){return ops.deg;});
    Ops.def_property_readonly("N", [](starry::Ops<Scalar>& ops){return ops.N;});

    // Occultation solution in emitted light
    Ops.def(
        "sT", [](
            starry::Ops<Scalar>& ops,
            const Vector<double>& b,
            const double& r
        )
    {
        size_t npts = size_t(b.size());
        Matrix<double, RowMajor> sT(npts, ops.N);
        for (size_t n = 0; n < npts; ++n) {
            ops.G.compute(static_cast<Scalar>(b(n)), static_cast<Scalar>(r));
            sT.row(n) = ops.G.sT.template cast<double>();
        }
        return sT;
    });

    // Gradient of occultation solution in emitted light
    Ops.def(
        "sT", [](
            starry::Ops<Scalar>& ops,
            const Vector<double>& b,
            const double& r,
            const Matrix<double, RowMajor>& bsT
        )
    {
        size_t npts = size_t(b.size());
        Vector<double> bb(npts);
        double br = 0.0;
        for (size_t n = 0; n < npts; ++n) {
            ops.G.template compute<true>(static_cast<Scalar>(b(n)), static_cast<Scalar>(r));
            bb(n) = static_cast<double>(ops.G.dsTdb.dot(bsT.row(n).template cast<Scalar>()));
            br += static_cast<double>(ops.G.dsTdr.dot(bsT.row(n).template cast<Scalar>()));
        }
        return py::make_tuple(bb, br);
    });

    // Change of basis matrix: Ylm to poly
    Ops.def_property_readonly(
        "A1", [](
            starry::Ops<Scalar>& ops
        )
    {   
#       ifdef STARRY_MULTI
            return (ops.B.A1.template cast<double>()).eval();
#       else
            return ops.B.A1;
#       endif
    });

    // Augmented change of basis matrix: poly to Ylm
    Ops.def_property_readonly(
        "A1Inv", [](
            starry::Ops<Scalar>& ops
        )
    {
#       ifdef STARRY_MULTI
            return (ops.B.A1Inv.template cast<double>()).eval();
#       else
            return ops.B.A1Inv;
#       endif
    });

    // Change of basis matrix: Ylm to greens
    Ops.def_property_readonly(
        "A", [](
            starry::Ops<Scalar>& ops
        )
    {
#       ifdef STARRY_MULTI
            return (ops.B.A.template cast<double>()).eval();
#       else
            return ops.B.A;
#       endif
    });

    // Rotation solution in emitted light
    Ops.def_property_readonly(
        "rT", [](
            starry::Ops<Scalar>& ops
        )
    {
        return ops.B.rT.template cast<double>();
    });

    // Rotation solution in reflected light
    Ops.def(
        "rTReflected", [](
            starry::Ops<Scalar>& ops,
            const Vector<double>& bterm
        )
    {
        size_t npts = size_t(bterm.size());
        Matrix<double, RowMajor> rT(npts, ops.N);
        for (size_t n = 0; n < npts; ++n) {
            ops.GRef.compute(static_cast<Scalar>(bterm(n)));
            rT.row(n) = ops.GRef.rT.template cast<double>();
        }
        return rT;
    });

    // Gradient of rotation solution in reflected light
    Ops.def(
        "rTReflected", [](
            starry::Ops<Scalar>& ops,
            const Vector<double>& bterm,
            const Matrix<double, RowMajor>& brT
        )
    {
        size_t npts = size_t(bterm.size());
        Vector<double> bb(npts);
        for (size_t n = 0; n < npts; ++n) {
            bb(n) = static_cast<double>(ops.GRef.compute(static_cast<Scalar>(bterm(n)), 
                                        brT.row(n).template cast<Scalar>()));
        }
        return bb;
    });

    // Rotation solution in emitted light dotted into Ylm space
    Ops.def_property_readonly(
        "rTA1", [](
            starry::Ops<Scalar>& ops
        )
    {
        return ops.B.rTA1.template cast<double>();
    });

    // Polynomial basis at a vector of points
    Ops.def(
        "pT", [](
            starry::Ops<Scalar>& ops,
            const RowVector<double>& x,
            const RowVector<double>& y,
            const RowVector<double>& z
        )
    {
        ops.B.computePolyBasis(x.template cast<Scalar>(), 
                               y.template cast<Scalar>(), 
                               z.template cast<Scalar>());
        return ops.B.pT.template cast<double>();
    });

    // XY rotation operator (vectors)
    Ops.def(
        "dotRxy", [](
            starry::Ops<Scalar>& ops,
            const RowVector<double>& M,
            const double& inc,
            const double& obl
        )
    {
        ops.W.dotRxy(M.template cast<Scalar>(), static_cast<Scalar>(inc), 
                     static_cast<Scalar>(obl));
        return ops.W.dotRxy_result.template cast<double>();
    });

    // XY rotation operator (matrices)
    Ops.def(
        "dotRxy", [](
            starry::Ops<Scalar>& ops,
            const Matrix<double>& M,
            const double& inc,
            const double& obl
        )
    {
        ops.W.dotRxy(M.template cast<Scalar>(), static_cast<Scalar>(inc), 
                     static_cast<Scalar>(obl));
        return ops.W.dotRxy_result.template cast<double>();
    });

    // Gradient of XY rotation matrix (vectors)
    Ops.def(
        "dotRxy", [](
            starry::Ops<Scalar>& ops,
            const RowVector<double>& M,
            const double& inc,
            const double& obl,
            const Matrix<double>& bMRxy
        )
    {
        ops.W.dotRxy(M.template cast<Scalar>(), static_cast<Scalar>(inc), 
                     static_cast<Scalar>(obl), bMRxy.template cast<Scalar>());
        return py::make_tuple(ops.W.dotRxy_bM.template cast<double>(), 
                              static_cast<double>(ops.W.dotRxy_binc), 
                              static_cast<double>(ops.W.dotRxy_bobl));
    });

    // Gradient of XY rotation matrix (matrices)
    Ops.def(
        "dotRxy", [](
            starry::Ops<Scalar>& ops,
            const Matrix<double>& M,
            const double& inc,
            const double& obl,
            const Matrix<double>& bMRxy
        )
    {
        ops.W.dotRxy(M.template cast<Scalar>(), static_cast<Scalar>(inc), 
                     static_cast<Scalar>(obl), bMRxy.template cast<Scalar>());
        return py::make_tuple(ops.W.dotRxy_bM.template cast<double>(), 
                              static_cast<double>(ops.W.dotRxy_binc), 
                              static_cast<double>(ops.W.dotRxy_bobl));
    });

    // Transpose of XY rotation operator (vectors)
    Ops.def(
        "dotRxyT", [](
            starry::Ops<Scalar>& ops,
            const RowVector<double>& M,
            const double& inc,
            const double& obl
        )
    {
        ops.W.dotRxyT(M.template cast<Scalar>(), static_cast<Scalar>(inc), 
                      static_cast<Scalar>(obl));
        return ops.W.dotRxyT_result.template cast<double>();
    });

    // Transpose of XY rotation operator (matrices)
    Ops.def(
        "dotRxyT", [](
            starry::Ops<Scalar>& ops,
            const Matrix<double>& M,
            const double& inc,
            const double& obl
        )
    {
        ops.W.dotRxyT(M.template cast<Scalar>(), static_cast<Scalar>(inc), 
                      static_cast<Scalar>(obl));
        return ops.W.dotRxyT_result.template cast<double>();
    });

    // Gradient of transpose of XY rotation matrix (vectors)
    Ops.def(
        "dotRxyT", [](
            starry::Ops<Scalar>& ops,
            const RowVector<double>& M,
            const double& inc,
            const double& obl,
            const Matrix<double>& bMRxyT
        )
    {
        ops.W.dotRxyT(M.template cast<Scalar>(), static_cast<Scalar>(inc), 
                      static_cast<Scalar>(obl), bMRxyT.template cast<Scalar>());
        return py::make_tuple(ops.W.dotRxyT_bM.template cast<double>(), 
                              static_cast<double>(ops.W.dotRxyT_binc), 
                              static_cast<double>(ops.W.dotRxyT_bobl));
    });

    // Gradient of transpose of XY rotation matrix (matrices)
    Ops.def(
        "dotRxyT", [](
            starry::Ops<Scalar>& ops,
            const Matrix<double>& M,
            const double& inc,
            const double& obl,
            const Matrix<double>& bMRxyT
        )
    {
        ops.W.dotRxyT(M.template cast<Scalar>(), static_cast<Scalar>(inc), 
                      static_cast<Scalar>(obl), bMRxyT.template cast<Scalar>());
        return py::make_tuple(ops.W.dotRxyT_bM.template cast<double>(), 
                              static_cast<double>(ops.W.dotRxyT_binc), 
                              static_cast<double>(ops.W.dotRxyT_bobl));
    });

    // Z rotation operator (vectors)
    Ops.def(
        "dotRz", [](
            starry::Ops<Scalar>& ops,
            const RowVector<double>& M,
            const Vector<double>& theta
        )
    {
        ops.W.dotRz(M.template cast<Scalar>(), theta.template cast<Scalar>());
        return ops.W.dotRz_result.template cast<double>();
    });

    // Z rotation operator (matrices)
    Ops.def(
        "dotRz", [](
            starry::Ops<Scalar>& ops,
            const Matrix<double>& M,
            const Vector<double>& theta
        )
    {
        ops.W.dotRz(M.template cast<Scalar>(), theta.template cast<Scalar>());
        return ops.W.dotRz_result.template cast<double>();
    });

    // Gradient of Z rotation matrix (vectors)
    Ops.def(
        "dotRz", [](
            starry::Ops<Scalar>& ops,
            const RowVector<double>& M,
            const Vector<double>& theta,
            const Matrix<double>& bMRz
        )
    {
        ops.W.dotRz(M.template cast<Scalar>(), theta.template cast<Scalar>(), 
                    bMRz.template cast<Scalar>());
        return py::make_tuple(ops.W.dotRz_bM.template cast<double>(), 
                              ops.W.dotRz_btheta.template cast<double>());
    });

    // Gradient of Z rotation matrix (matrices)
    Ops.def(
        "dotRz", [](
            starry::Ops<Scalar>& ops,
            const Matrix<double>& M,
            const Vector<double>& theta,
            const Matrix<double>& bMRz
        )
    {
        ops.W.dotRz(M.template cast<Scalar>(), theta.template cast<Scalar>(), 
                    bMRz.template cast<Scalar>());
        return py::make_tuple(ops.W.dotRz_bM.template cast<double>(), 
                              ops.W.dotRz_btheta.template cast<double>());
    });

    // Filter operator
    Ops.def(
        "F", [](
            starry::Ops<Scalar>& ops,
            const Vector<double>& u,
            const Vector<double>& f
        )
    {
        ops.F.compute(u.template cast<Scalar>(), f.template cast<Scalar>());
        return ops.F.F.template cast<double>();
    });

    // Gradient of filter operator
    Ops.def(
        "F", [](
            starry::Ops<Scalar>& ops,
            const Vector<double>& u,
            const Vector<double>& f,
            const Matrix<double>& bF
        )
    {
        ops.F.compute(u.template cast<Scalar>(), f.template cast<Scalar>(), 
                      bF.template cast<Scalar>());
        return py::make_tuple(ops.F.bu.template cast<double>(), 
                              ops.F.bf.template cast<double>());
    });

    // Compute the Ylm expansion of a gaussian spot
    Ops.def(
        "spotYlm", [](
            starry::Ops<Scalar>& ops,
            const RowVector<Scalar>& amp,
            const Scalar& sigma,
            const Scalar& lat,
            const Scalar& lon,
            const Scalar& inc,
            const Scalar& obl
        )
    {
        return ops.spotYlm(amp.template cast<double>(), 
                           static_cast<Scalar>(sigma), 
                           static_cast<Scalar>(lat), 
                           static_cast<Scalar>(lon),
                           static_cast<Scalar>(inc), 
                           static_cast<Scalar>(obl)
                           ).template cast<double>();
    });

    // Rotate a Ylm map given an axis `u` and angle `theta`
    Ops.def(
        "rotate", [](
            starry::Ops<Scalar>& ops,
            const UnitVector<Scalar>& u,
            const Scalar& theta,
            const Matrix<double>& y
        )
    {
        ops.W.rotate(static_cast<Scalar>(u(0)), 
                     static_cast<Scalar>(u(1)), 
                     static_cast<Scalar>(u(2)), 
                     static_cast<Scalar>(theta), 
                     y.template cast<Scalar>());
        return ops.W.rotate_result.template cast<double>();
    });

    // Return the actual Wigner matrix `R`
    Ops.def(
        "R", [](
            starry::Ops<Scalar>& ops,
            const UnitVector<Scalar>& u,
            const Scalar& theta
        )
    {
        ops.W.compute(static_cast<Scalar>(u(0)), 
                      static_cast<Scalar>(u(1)), 
                      static_cast<Scalar>(u(2)), 
                      static_cast<Scalar>(theta));
        // NOTE: this will *fail* for multiprecision types
        return ops.W.R;
    });

}