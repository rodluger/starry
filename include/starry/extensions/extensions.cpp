/**
\file extensions.cpp
\brief Python bindings for custom C++ extensions to the code.

*/

// Enable debug mode?
#ifdef STARRY_DEBUG
#undef NDEBUG
#endif

// Enable the Python interface
#ifndef STARRY_ENABLE_PYTHON_INTERFACE
#define STARRY_ENABLE_PYTHON_INTERFACE
#endif

//! Import pybind11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
using namespace pybind11::literals;

//! Import starry
#include <starry/starry.h>
#include "extensions.h"
using namespace starry::utils;
using namespace starry::extensions;

//! Register the Python module
PYBIND11_MODULE(
    _starry_extensions, 
    m
) {

    // Add bindings for custom extensions here.

    // \todo Compute the MAP map coefficients
    m.def(
        "MAP", [](
            const Matrix<double>& A,
            py::array_t<double>& flux_,
            py::array_t<double>& C_,
            py::array_t<double>& L_
        ) {
            // Check the dimensions of A
            py::ssize_t nt = A.rows();
            assert(A.cols() == map.N);

            // Map the flux to an Eigen type
            py::buffer_info buf = flux_.request();
            assert(buf.ndim == 1);
            assert(buf.size == nt);
            double *ptr = (double *) buf.ptr;
            Eigen::Map<Vector<double>> flux(ptr, nt, 1);

            // Map the covariance to an Eigen type
            Eigen::Map<Vector<double>> C(NULL, nt, 1);
            Vector<double> tmp_C;
            buf = C_.request();
            ptr = (double *) buf.ptr;
            if (buf.ndim == 0) {
                tmp_C = ptr[0] * Vector<double>::Ones(nt);
                new (&C) Eigen::Map<Vector<double>>(&tmp_C(0), nt, 1);
            } else if ((buf.ndim == 1) && (buf.size == nt)) {
                new (&C) Eigen::Map<Vector<double>>(ptr, nt, 1);
            } else {
                throw errors::ShapeError("Vector `C` has the incorrect shape.");
            }

            // Map the prior weights to an Eigen type
            Eigen::Map<Vector<double>> L(NULL, map.N, 1);
            Vector<double> tmp_L;
            buf = L_.request();
            ptr = (double *) buf.ptr;
            if (buf.ndim == 0) {
                tmp_L = ptr[0] * Vector<double>::Ones(map.N);
                new (&L) Eigen::Map<Vector<double>>(&tmp_L(0), map.N, 1);
            } else if ((buf.ndim == 1) && (buf.size == map.N)) {
                new (&L) Eigen::Map<Vector<double>>(ptr, map.N, 1);
            } else {
                throw errors::ShapeError("Vector `L` has the incorrect shape.");
            }

            /* This is how we would handle 2D L and C:
            Eigen::Map<Matrix<double>> L(NULL, map.N, map.N);
            Matrix<double> tmp_L;
            buf = L_.request();
            ptr = (double *) buf.ptr;
            if (buf.ndim == 0) {
                tmp_L = ptr[0] * Matrix<double>::Identity(map.N, map.N);
                new (&L) Eigen::Map<Matrix<double>>(&tmp_L(0), map.N, map.N);
            } else if ((buf.ndim == 1) && (buf.size == map.N)) {
                Eigen::Map<Vector<double>> L_diag(ptr, nt, 1);
                tmp_L = Matrix<double>(L_diag.asDiagonal());
                new (&L) Eigen::Map<Matrix<double>>(ptr, nt, 1);
            } else if ((buf.ndim == 2) && (buf.shape[0] == map.N) && (buf.shape[1] == map.N)) {
                new (&L) Eigen::Map<Matrix<double>>(ptr, map.N, map.N);
            } else {
                throw errors::ShapeError("Matrix `L` has the incorrect shape.");
            }
            */

            // Compute the max like model
            Vector<double> yhat(map.N, 1);
            Matrix<double> yvar(map.N, map.N);
            map.computeMaxLikeMap(A, flux, C, L, yhat, yvar);

            // Set the map coefficients
            map.setU(Vector<typename T::Scalar>::Zero(map.lmax));
            map.setY(yhat);

            // Return the variance
            return yvar;

        }, "A"_a, "flux"_a, "C"_a, "L"_a=INFINITY);

}