/**
This defines the main Python interface to the code.
Note that this file is #include'd *twice*, once
for `starry` and once for `starry.grad` so we don't
have to duplicate any code. Yes, this is super hacky.

TODO: Make usage of & consistent in input arguments!
*/

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>
#include <cmath>
#include <stdlib.h>
#include <vector>

using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;
template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T>
using VectorT = Eigen::Matrix<T, 1, Eigen::Dynamic>;
template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using UnitVector = Eigen::Matrix<T, 3, 1>;

/**
Define our map type (double or AutoDiffScalar)
Apologies to sticklers for the indented #define's.

*/
#ifndef STARRY_AUTODIFF

    void add_starry(py::module &m) { }

#else

    #define STARRY_NGRAD                    7
    //#define MapType                         Eigen::AutoDiffScalar<Eigen::Matrix<double, STARRY_NGRAD, 1>>
    #define MapType                         Eigen::AutoDiffScalar<Eigen::VectorXd>

    #include "rotation.h"

    void add_starry_grad(py::module &m) {

        m.def("test", []() {

            // Declare our gradient types
            UnitVector<MapType> axis_g;
            axis_g(0).value() = 1. / sqrt(3.);
            axis_g(0).derivatives() = Eigen::VectorXd::Unit(STARRY_NGRAD, 0);
            axis_g(1).value() = 1. / sqrt(3.);
            axis_g(1).derivatives() = Eigen::VectorXd::Unit(STARRY_NGRAD, 1);
            axis_g(2).value() = 1. / sqrt(3.);
            axis_g(2).derivatives() = Eigen::VectorXd::Unit(STARRY_NGRAD, 2);
            MapType theta_g(0.5, STARRY_NGRAD, 3);
            MapType xo_g(0.1, STARRY_NGRAD, 4);
            MapType yo_g(0.1, STARRY_NGRAD, 5);
            MapType ro_g(0.1, STARRY_NGRAD, 6);
            MapType tmp;

            int lmax = 2;
            rotation::Wigner<MapType> R(lmax);
            MapType costheta = cos(theta_g);
            MapType sintheta = sin(theta_g);
            rotation::computeR(lmax, axis_g, costheta, sintheta, R.Complex, R.Real);

            tmp = R.Real[1](0, 0);

            cout << tmp.value() << endl;
            cout << tmp.derivatives().transpose() << endl;


        });

    }

#endif
