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
typedef boost::multiprecision::cpp_dec_float<STARRY_MP_DIGITS> mp_backend;
typedef boost::multiprecision::number<mp_backend, boost::multiprecision::et_off> bigdouble;

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

// Fast square roots of integers
template <class T>
class SqrtInt {
        std::vector<T> vec;
    public:
        SqrtInt() { vec.push_back(T(0.0)); }
        inline T value(const int& n) {
            if (n < 0)  throw errors::BadIndex();
            while (n >= vec.size()) vec.push_back(T(sqrt(T(vec.size()))));
            return vec[n];
        }
        inline T operator() (int n) { return value(n); }
};

// Fast inverse square roots of integers
template <class T>
class InvSqrtInt {
        std::vector<T> vec;
    public:
        InvSqrtInt() { vec.push_back(T(INFINITY)); }
        inline T value(const int& n) {
            if (n < 0)  throw errors::BadIndex();
            while (n >= vec.size()) vec.push_back(T(T(1.) / T(sqrt(vec.size()))));
            return vec[n];
        }
        inline T operator() (int n) { return value(n); }
};

// Fast factorials of integers
template <class T>
class Factorial {
        std::vector<T> vec;
    public:
        Factorial() { vec.push_back(T(1.0)); }
        inline T value(const int& n) {
            if (n < 0) throw errors::BadIndex();
            int sz = vec.size();
            while (n >= vec.size()) {
                vec.push_back(T(T(sz) * vec[sz - 1]));
                sz++;
            }
            return vec[n];
        }
        inline T operator() (int n) { return value(n); }
};

// Fast factorials of (n / 2)
template <class T>
class HalfFactorial {
        std::vector<T> pos;
        std::vector<T> neg;
    public:
        HalfFactorial() {
            pos.push_back(0.5 * T(sqrt(T(acos(T(-1))))));
            neg.push_back(T(sqrt(T(acos(T(-1))))));
        }
        inline T value(const int& n) {
            if (is_even(n)) return T(INFINITY);
            int i, sz;
            if (n < 0) {
                i = (-n - 1) / 2;
                sz = neg.size();
                while (i >= neg.size()) {
                    neg.push_back(T(neg[sz - 1]) / T(0.5 - sz));
                    sz++;
                }
                return neg[i];
            } else {
                i = (n - 1) / 2;
                sz = pos.size();
                while (i >= pos.size()) {
                    pos.push_back(T(T(sz + 0.5) * pos[sz - 1]));
                    sz++;
                }
                return pos[i];
            }
        }
        inline T operator() (int n) { return value(n); }
};

// Miscellaneous math utilities for fast calculations in all
// three data types used by `starry`
class MathUtils {

    public:

        // Integer functions w/ caching
        Factorial<double> D_factorial;
        Factorial<bigdouble> B_factorial;
        Factorial<Grad> G_factorial;
        HalfFactorial<double> D_half_factorial;
        HalfFactorial<bigdouble> B_half_factorial;
        HalfFactorial<Grad> G_half_factorial;
        SqrtInt<double> D_sqrt_int;
        SqrtInt<bigdouble> B_sqrt_int;
        SqrtInt<Grad> G_sqrt_int;
        InvSqrtInt<double> D_invsqrt_int;
        InvSqrtInt<bigdouble> B_invsqrt_int;
        InvSqrtInt<Grad> G_invsqrt_int;

        // Constants
        double D_PI;
        bigdouble B_PI;
        Grad G_PI;

        MathUtils() :
            D_factorial(), B_factorial(), G_factorial(),
            D_half_factorial(), B_half_factorial(), G_half_factorial(),
            D_sqrt_int(), B_sqrt_int(), G_sqrt_int(),
            D_invsqrt_int(), B_invsqrt_int(), G_invsqrt_int() {

                D_PI = acos((double)(-1.0));
                B_PI = acos((bigdouble)(-1.0));
                G_PI = acos((Grad)(-1.0));
                G_PI.derivatives().setZero(G_PI.derivatives().size());

        }

        // The value of PI
        template <typename T>
        T PI();

        // Factorial of n
        template <typename T>
        T factorial(const int& n);

        // Factorial of n / 2
        template <typename T>
        T half_factorial(const int& n);

        // Square root of n
        template <typename T>
        T sqrt(const int& n);

        // One over the square root of n
        template <typename T>
        T invsqrt(const int& n);

        // Binomial coefficient
        template <typename T>
        inline T choose(int n, int k) {
            return factorial<T>(n) / (factorial<T>(k) * factorial<T>(n - k));
        }

        // Gamma function
        template <typename T>
        inline T gamma(int n) {
            return factorial<T>(n - 1);
        }

        // Gamma of n + 1/2
        template <typename T>
        inline T gamma_sup(int n) {
            return half_factorial<T>(2 * n - 1);
        }

};

template <>
inline double MathUtils::PI(){
    return D_PI;
}

template <>
inline bigdouble MathUtils::PI(){
    return B_PI;
}

template <>
inline Grad MathUtils::PI(){
    return G_PI;
}

template <>
inline double MathUtils::factorial(const int& n){
    return D_factorial(n);
}

template <>
inline bigdouble MathUtils::factorial(const int& n){
    return B_factorial(n);
}

template <>
inline Grad MathUtils::factorial(const int& n){
    return G_factorial(n);
}

template <>
inline double MathUtils::half_factorial(const int& n) {
    if (is_even(n)) {
        if (n < 0) return INFINITY;
        else return D_factorial(n / 2);
    } else {
        return D_half_factorial(n);
    }
}

template <>
inline bigdouble MathUtils::half_factorial(const int& n) {
    if (is_even(n)) {
        if (n < 0) return bigdouble(INFINITY);
        else return B_factorial(n / 2);
    } else {
        return B_half_factorial(n);
    }
}

template <>
inline Grad MathUtils::half_factorial(const int& n) {
    if (is_even(n)) {
        if (n < 0) return Grad(INFINITY);
        else return G_factorial(n / 2);
    } else {
        return G_half_factorial(n);
    }
}

template <>
inline double MathUtils::sqrt(const int& n){
    return D_sqrt_int(n);
}

template <>
inline bigdouble MathUtils::sqrt(const int& n){
    return B_sqrt_int(n);
}

template <>
inline Grad MathUtils::sqrt(const int& n){
    return G_sqrt_int(n);
}

template <>
inline double MathUtils::invsqrt(const int& n){
    return D_invsqrt_int(n);
}

template <>
inline bigdouble MathUtils::invsqrt(const int& n){
    return B_invsqrt_int(n);
}

template <>
inline Grad MathUtils::invsqrt(const int& n){
    return G_invsqrt_int(n);
}

// We can now access this anywhere in the code!
extern MathUtils math;

#endif
