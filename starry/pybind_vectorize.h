/**
Vectorization wrappers for the `Map` methods.

*/

#ifndef _STARRY_VECTORIZE_H_
#define _STARRY_VECTORIZE_H_

#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdlib.h>
#include "utils.h"
#include "errors.h"

namespace pybind_vectorize {

    using std::max;
    using std::string;
    using namespace utils;
    namespace py = pybind11;

    //! Vectorize a single python object
    inline Vector<double> vectorize_arg(const py::array_t<double>& arg,
                                        int& size){
        Vector<double> res;
        if (arg.size() == 1) {
            res = Vector<double>::Constant(size, py::cast<double>(arg));
            return res;
        } else {
            res = py::cast<Vector<double>>(arg);
            if ((size == 0) || (res.size() == size)) {
                size = res.size();
                return res;
            } else {
                throw errors::ValueError("Mismatch in argument dimensions.");
            }
        }
    }

    //! Vectorize function of three args
    inline void vectorize_args(const py::array_t<double>& arg1,
                               const py::array_t<double>& arg2,
                               const py::array_t<double>& arg3,
                               Vector<double>& arg1_v,
                               Vector<double>& arg2_v,
                               Vector<double>& arg3_v) {
        int size = 0;
        if (arg1.size() > 1) {
            // arg1 is a vector
            arg1_v = vectorize_arg(arg1, size);
            arg2_v = vectorize_arg(arg2, size);
            arg3_v = vectorize_arg(arg3, size);
        } else if (arg2.size() > 1) {
            // arg2 is a vector
            arg2_v = vectorize_arg(arg2, size);
            arg3_v = vectorize_arg(arg3, size);
            arg1_v = vectorize_arg(arg1, size);
        } else if (arg3.size() > 1) {
            // arg3 is a vector
            arg3_v = vectorize_arg(arg3, size);
            arg1_v = vectorize_arg(arg1, size);
            arg2_v = vectorize_arg(arg2, size);
        } else {
            // no arg is a vector
            size = 1;
            arg1_v = vectorize_arg(arg1, size);
            arg2_v = vectorize_arg(arg2, size);
            arg3_v = vectorize_arg(arg3, size);
        }
    }

    //! Vectorize function of four args
    inline void vectorize_args(const py::array_t<double>& arg1,
                               const py::array_t<double>& arg2,
                               const py::array_t<double>& arg3,
                               const py::array_t<double>& arg4,
                               Vector<double>& arg1_v,
                               Vector<double>& arg2_v,
                               Vector<double>& arg3_v,
                               Vector<double>& arg4_v) {
        int size = 0;
        if (arg1.size() > 1) {
            // arg1 is a vector
            arg1_v = vectorize_arg(arg1, size);
            arg2_v = vectorize_arg(arg2, size);
            arg3_v = vectorize_arg(arg3, size);
            arg4_v = vectorize_arg(arg4, size);
        } else if (arg2.size() > 1) {
            // arg2 is a vector
            arg2_v = vectorize_arg(arg2, size);
            arg3_v = vectorize_arg(arg3, size);
            arg4_v = vectorize_arg(arg4, size);
            arg1_v = vectorize_arg(arg1, size);
        } else if (arg3.size() > 1) {
            // arg3 is a vector
            arg3_v = vectorize_arg(arg3, size);
            arg4_v = vectorize_arg(arg4, size);
            arg1_v = vectorize_arg(arg1, size);
            arg2_v = vectorize_arg(arg2, size);
        } else if (arg4.size() > 1) {
            // arg4 is a vector
            arg4_v = vectorize_arg(arg4, size);
            arg1_v = vectorize_arg(arg1, size);
            arg2_v = vectorize_arg(arg2, size);
            arg3_v = vectorize_arg(arg3, size);
        } else {
            // no arg is a vector
            size = 1;
            arg1_v = vectorize_arg(arg1, size);
            arg2_v = vectorize_arg(arg2, size);
            arg3_v = vectorize_arg(arg3, size);
            arg4_v = vectorize_arg(arg4, size);
        }
    }

    //! Vectorized `flux` method: single-wavelength starry
    template <typename T>
    typename std::enable_if<!std::is_base_of<Eigen::EigenBase<Row<T>>,
                                             Row<T>>::value, py::object>::type
    flux(maps::Map<T> &map, py::array_t<double>& theta, py::array_t<double>& xo,
         py::array_t<double>& yo, py::array_t<double>& ro, bool gradient,
         bool numerical){

        if (gradient) {

            // Initialize a dictionary of derivatives
            size_t n = max(theta.size(), max(xo.size(),
                           max(yo.size(), ro.size())));
            std::map<string, Matrix<double>> grad;
            map.resizeGradient();
            auto dF_names = map.getGradientNames();
            int n_ylm = 0, n_ul = 0;
            for (auto name : dF_names) {
                if (name == "y")
                    ++n_ylm;
                else if (name == "u")
                    ++n_ul;
                else
                    grad[name].resize(n, 1);
            }
            grad["y"].resize(n, n_ylm);
            grad["u"].resize(n, n_ul);

            // Nested lambda function;
            // https://github.com/pybind/pybind11/issues/
            // 761#issuecomment-288818460
            int t = 0;
            auto F = py::vectorize([&map, &grad, &t, &dF_names, &numerical]
                        (double theta, double xo, double yo, double ro) {
                // Evaluate the function
                double res = static_cast<double>(map.flux(theta, xo, yo,
                                                          ro, true,
                                                          numerical));
                // Gather the derivatives
                auto dF = map.getGradient();
                int n_ylm = 0, n_ul = 0;
                for (int j = 0; j < dF.size(); j++) {
                    if (dF_names[j] == "y") {
                        grad["y"](t, n_ylm++) = static_cast<double>(dF(j));
                    } else if (dF_names[j] == "u") {
                        grad["u"](t, n_ul++) = static_cast<double>(dF(j));
                    } else {
                        grad[dF_names[j]](t, 0) = static_cast<double>(dF(j));
                    }
                }
                t++;
                return res;
            })(theta, xo, yo, ro);

            // Convert to an actual python dictionary
            // Necessary because we're mixing vectors and matrices
            // among the dictionary items.
            // NOTE: All this copying could be slow: not ideal.
            auto pygrad = py::dict();
            for (std::string name : dF_names) {
                if ((name != "y") && (name != "u")) {
                    pygrad[name.c_str()] = grad[name].col(0);
                }
            }
            pygrad["y"] = grad["y"].transpose();
            pygrad["u"] = grad["u"].transpose();

            // Return a tuple of (F, dict(dF))
            return py::make_tuple(F, pygrad);

        } else {

            // Easy! We'll just return F
            return py::vectorize([&map, &numerical](double theta, double xo,
                                        double yo, double ro) {
                return static_cast<double>(map.flux(theta, xo, yo, ro, false,
                                           numerical));
            })(theta, xo, yo, ro);

        }

    }

    //! Vectorized `flux` method: spectral starry
    template <typename T>
    typename std::enable_if<std::is_base_of<Eigen::EigenBase<Row<T>>,
                                            Row<T>>::value, py::object>::type
    flux(maps::Map<T> &map, py::array_t<double>& theta, py::array_t<double>& xo,
         py::array_t<double>& yo, py::array_t<double>& ro, bool gradient,
         bool numerical){

        // Vectorize the arguments manually
        Vector<double> theta_v, xo_v, yo_v, ro_v;
        vectorize_args(theta, xo, yo, ro,
                       theta_v, xo_v, yo_v, ro_v);
        size_t sz = theta_v.size();

        if (gradient) {

            // Initialize the gradient matrix
            std::map<string, Matrix<double>> grad;
            map.resizeGradient();
            auto dF_names = map.getGradientNames();
            int n_ylm = 0, n_ul = 0;
            for (auto name : dF_names) {
                if (name == "y")
                    ++n_ylm;
                else if (name == "u")
                    ++n_ul;
                else
                    grad[name].resize(sz, map.nwav);
            }

            // Initialize the map coefficient gradients
            std::vector<Matrix<double>> grad_y(n_ylm);
            for (int i = 0; i < n_ylm; ++i)
                grad_y[i].resize(sz, map.nwav);
            std::vector<Matrix<double>> grad_u(n_ul);
            for (int i = 0; i < n_ul; ++i)
                grad_u[i].resize(sz, map.nwav);

            // Iterate through the timeseries
            Matrix<double> F(sz, map.nwav);
            for (size_t i = 0; i < sz; ++i) {

                // Function value
                F.row(i) = map.flux(theta_v(i), xo_v(i),
                           yo_v(i), ro_v(i), true,
                           numerical).template cast<double>();

                // Gradient
                auto dF = map.getGradient();
                int ky = 0, ku = 0;
                for (int j = 0; j < dF.rows(); ++j) {
                    if (dF_names[j] == "y") {
                        grad_y[ky++].row(i) = dF.row(j).template cast<double>();
                    } else if (dF_names[j] == "u") {
                        grad_u[ku++].row(i) = dF.row(j).template cast<double>();
                    } else {
                        grad[dF_names[j]].row(i) = dF.row(j).template cast<double>();
                    }
                }

            }

            // Convert to a python dictionary
            auto pygrad = py::dict();
            for (std::string name : dF_names) {
                if ((name != "y") && (name != "u")) {
                    pygrad[name.c_str()] = grad[name];
                }
            }
            pygrad["y"] = grad_y;
            pygrad["u"] = grad_u;

            // Cast to python object
            return py::make_tuple(F, pygrad);

        } else {

            // Iterate through the timeseries
            Matrix<double> F(sz, map.nwav);
            for (size_t i = 0; i < sz; ++i) {
                F.row(i) = map.flux(theta_v(i), xo_v(i),
                           yo_v(i), ro_v(i), false,
                           numerical).template cast<double>();
            }

            // Cast to python object
            return py::cast(F);

        }

    }

    //! Vectorized `evaluate` method: single-wavelength starry
    template <typename T>
    typename std::enable_if<!std::is_base_of<Eigen::EigenBase<Row<T>>,
                                             Row<T>>::value, py::object>::type
    evaluate(maps::Map<T> &map, py::array_t<double>& theta,
             py::array_t<double>& x, py::array_t<double>& y){

        // Easy! We'll just return I
        return py::vectorize([&map](double theta, double x, double y) {
            return static_cast<double>(map(theta, x, y));
        })(theta, x, y);

    }

    //! Vectorized `evaluate` method: spectral starry
    template <typename T>
    typename std::enable_if<std::is_base_of<Eigen::EigenBase<Row<T>>,
                                            Row<T>>::value, py::object>::type
    evaluate(maps::Map<T> &map, py::array_t<double>& theta,
             py::array_t<double>& x, py::array_t<double>& y){

        // Vectorize the arguments manually
        Vector<double> theta_v, x_v, y_v;
        vectorize_args(theta, x, y, theta_v, x_v, y_v);
        size_t sz = theta_v.size();

        // Iterate through the timeseries
        Matrix<double> I(sz, map.nwav);
        for (size_t i = 0; i < sz; ++i) {
            I.row(i) = map(theta_v(i), x_v(i), y_v(i)).template cast<double>();
        }

        // Cast to python object
        return py::cast(I);

    }

} // namespace vectorize

#endif
