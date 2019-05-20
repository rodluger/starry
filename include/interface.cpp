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
    using Scalar = Multi;
    // TODO: Cast things as needed below.
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
        Matrix<Scalar, RowMajor> sT(npts, ops.N);
        for (size_t n = 0; n < npts; ++n) {
            ops.G.compute(b(n), r);
            sT.row(n) = ops.G.sT;
        }
        return sT;
    });

    // Gradient of occultation solution in emitted light
    Ops.def(
        "sT", [](
            starry::Ops<Scalar>& ops,
            const Vector<double>& b,
            const Scalar& r,
            const Matrix<double, RowMajor>& bsT
        )
    {
        size_t npts = size_t(b.size());
        Vector<Scalar> bb(npts);
        Scalar br = 0.0;
        for (size_t n = 0; n < npts; ++n) {
            ops.G.template compute<true>(b(n), r);
            bb(n) = ops.G.dsTdb.dot(bsT.row(n));
            br += ops.G.dsTdr.dot(bsT.row(n));
        }
        return py::make_tuple(bb, br);
    });

    // Change of basis matrix: Ylm to poly
    Ops.def_property_readonly(
        "A1", [](
            starry::Ops<Scalar>& ops
        )
    {
        return ops.B.A1;
    });

    // Augmented change of basis matrix: poly to Ylm
    Ops.def_property_readonly(
        "A1Inv", [](
            starry::Ops<Scalar>& ops
        )
    {
        return ops.B.A1Inv;
    });

    // Change of basis matrix: Ylm to greens
    Ops.def_property_readonly(
        "A", [](
            starry::Ops<Scalar>& ops
        )
    {
        return ops.B.A;
    });

    // Rotation solution in emitted light
    Ops.def_property_readonly(
        "rT", [](
            starry::Ops<Scalar>& ops
        )
    {
        return ops.B.rT;
    });

    // Rotation solution in emitted light dotted into Ylm space
    Ops.def_property_readonly(
        "rTA1", [](
            starry::Ops<Scalar>& ops
        )
    {
        return ops.B.rTA1;
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
        ops.B.computePolyBasis(x, y, z);
        return ops.B.pT;
    });

    // Polynomial basis at a point
    Ops.def(
        "pT", [](
            starry::Ops<Scalar>& ops,
            const double& x,
            const double& y,
            const double& z
        )
    {
        ops.B.computePolyBasis(x, y, z);
        return ops.B.pT_point;
    });

    // Gradient of the polynomial basis at a point
    Ops.def(
        "pT", [](
            starry::Ops<Scalar>& ops,
            const double& x,
            const double& y,
            const double& z,
            const RowVector<double>& bpT
        )
    {
        ops.B.computePolyBasis(x, y, z, bpT);
        return py::make_tuple(ops.B.bx, ops.B.by, ops.B.bz);
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
        ops.W.dotRxy(M, inc, obl);
        return ops.W.dotRxy_result;
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
        ops.W.dotRxy(M, inc, obl);
        return ops.W.dotRxy_result;
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
        ops.W.dotRxy(M, inc, obl, bMRxy);
        return py::make_tuple(ops.W.dotRxy_bM, ops.W.dotRxy_binc, ops.W.dotRxy_bobl);
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
        ops.W.dotRxy(M, inc, obl, bMRxy);
        return py::make_tuple(ops.W.dotRxy_bM, ops.W.dotRxy_binc, ops.W.dotRxy_bobl);
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
        ops.W.dotRxyT(M, inc, obl);
        return ops.W.dotRxyT_result;
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
        ops.W.dotRxyT(M, inc, obl);
        return ops.W.dotRxyT_result;
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
        ops.W.dotRxyT(M, inc, obl, bMRxyT);
        return py::make_tuple(ops.W.dotRxyT_bM, ops.W.dotRxyT_binc, ops.W.dotRxyT_bobl);
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
        ops.W.dotRxyT(M, inc, obl, bMRxyT);
        return py::make_tuple(ops.W.dotRxyT_bM, ops.W.dotRxyT_binc, ops.W.dotRxyT_bobl);
    });

    // Z rotation operator (vectors)
    Ops.def(
        "dotRz", [](
            starry::Ops<Scalar>& ops,
            const RowVector<double>& M,
            const Vector<double>& theta
        )
    {
        ops.W.dotRz(M, theta);
        return ops.W.dotRz_result;
    });

    // Z rotation operator (matrices)
    Ops.def(
        "dotRz", [](
            starry::Ops<Scalar>& ops,
            const Matrix<double>& M,
            const Vector<double>& theta
        )
    {
        ops.W.dotRz(M, theta);
        return ops.W.dotRz_result;
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
        ops.W.dotRz(M, theta, bMRz);
        return py::make_tuple(ops.W.dotRz_bM, ops.W.dotRz_btheta);
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
        ops.W.dotRz(M, theta, bMRz);
        return py::make_tuple(ops.W.dotRz_bM, ops.W.dotRz_btheta);
    });

    // Filter operator
    Ops.def(
        "F", [](
            starry::Ops<Scalar>& ops,
            const Vector<double>& u,
            const Vector<double>& f
        )
    {
        ops.F.compute(u, f);
        return ops.F.F;
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
        ops.F.compute(u, f, bF);
        return py::make_tuple(ops.F.bu, ops.F.bf);
    });

}