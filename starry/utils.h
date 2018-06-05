/**
Miscellaneous stuff used throughout the code.

*/

#ifndef _STARRY_UTILS_H_
#define _STARRY_UTILS_H_

#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>
#include <iostream>
#include <limits>
#include <vector>
#include "constants.h"
#include "errors.h"

// Multiprecision datatype
#include <boost/multiprecision/cpp_dec_float.hpp>
typedef boost::multiprecision::cpp_dec_float<STARRY_NMULTI> mp_backend;
typedef boost::multiprecision::number<mp_backend, boost::multiprecision::et_off> bigdouble;
#define BIGPI                                   bigdouble("3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068")

// Our custom vector types
template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T>
using VectorT = Eigen::Matrix<T, 1, Eigen::Dynamic>;
template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using UnitVector = Eigen::Matrix<T, 3, 1>;
using Grad = Eigen::AutoDiffScalar<Eigen::Matrix<double, STARRY_NGRAD, 1>>;

// Some useful unit vectors
static const UnitVector<double> xhat({1, 0, 0});
static const UnitVector<double> yhat({0, 1, 0});
static const UnitVector<double> zhat({0, 0, 1});

// Return the value of a scalar MapType variable
inline double get_value(double x) { return x; }
inline double get_value(Grad x) { return x.value(); }
inline double get_value(bigdouble x) { return (double)x; }
inline Vector<double> get_value(Vector<double> x) { return x; }
inline Vector<double> get_value(Vector<bigdouble> x) {
    Vector<double> vec;
    vec.resize(x.size());
    for (int n = 0; n < x.size(); n++) {
        vec(n) = (double)x(n);
    }
    return vec;
}
inline Vector<double> get_value(Vector<Grad> x) {
    Vector<double> vec;
    vec.resize(x.size());
    for (int n = 0; n < x.size(); n++) {
        vec(n) = x(n).value();
    }
    return vec;
}

// Set the value of a MapType variable
template <typename T>
inline void set_value(T& x, T& y) { x = y; }
template <>
inline void set_value(Grad& x, Grad& y) { x.value() = y.value(); }

// Print the derivatives of a MapType variable for debugging
template <typename T>
void print_derivs(T x) { std::cout << "None" << std::endl; }
template <>
void print_derivs(Grad x) { std::cout << x.derivatives().transpose() << std::endl; }

// Zero out the derivatives of a MapType variable
template <typename T>
inline void set_derivs_to_zero(T& x) { }
template <>
inline void set_derivs_to_zero(Grad& x) { x.derivatives().setZero(x.derivatives().size()); }

// Normalize a unit vector
template <typename T>
inline UnitVector<T> norm_unit(const UnitVector<T>& vec) {
    UnitVector<T> result = vec / sqrt(vec(0) * vec(0) + vec(1) * vec(1) + vec(2) * vec(2));
    return result;
}

// Helper function to figure out if we're using multiprecision
template <typename T>
inline bool is_bigdouble(T x) {
    return false;
}

// Helper function to figure out if we're using multiprecision
template <>
inline bool is_bigdouble(bigdouble x) {
    return true;
}

// Check if number is even (or doubly, triply, quadruply... even)
inline bool is_even(int n, int ntimes=1) {
    for (int i = 0; i < ntimes; i++) {
        if ((n % 2) != 0) return false;
        n /= 2;
    }
    return true;
}

#endif
