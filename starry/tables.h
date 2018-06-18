/**
Tables of square roots and factorials for fast `double` math
using good old template metaprogramming (>= C++14)

*/

#ifndef _STARRY_TABLES_H_
#define _STARRY_TABLES_H_

#include <cmath>
#include <boost/math/special_functions/factorials.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include "errors.h"

// Square root of pi at double precision
#define SQRTPI 1.772453850905516027298167483341

// Largest factorial computable at double precision
#define MAXFACT 170

// Largest double factorial computable at double precision
#define MAXDOUBLEFACT 300

// Largest square root we're willing to tabulate
#define MAXSQRT 300

namespace tables {

    double constexpr sqrt_rec(double x, double curr, double prev) {
        return curr == prev ? curr : sqrt_rec(x, 0.5 * (curr + x / curr), curr);
    }

    template <typename T>
    double constexpr sqrt_(T x) {
        return x > 0 ? sqrt_rec((double)x, (double)x, 0) : NAN;
    }

    double constexpr factorial_(int x) {
        return x > 0 ?
               (x <= MAXFACT ? x * factorial_(x - 1) : INFINITY):
               1.0;
    }

    double constexpr double_factorial_(int x) {
        return x > 0 ?
               (x <= MAXDOUBLEFACT ? x * double_factorial_(x - 2) : INFINITY):
               1.0;
    }

    double constexpr half_factorial_pos_(int x) {
        return x > 1 ?
               (x % 2 == 0 ? factorial_(x / 2) : half_factorial_pos_(x - 2) * (x / 2.0)) :
               (x == 0 ? 1 : 0.5 * SQRTPI);
    }

    double constexpr half_factorial_neg_(int x) {
        return x > 1 ?
               (x % 2 == 0 ? INFINITY : half_factorial_neg_(x - 2) / (1.0 - x / 2.0)) :
               (x == 0 ? 1 : SQRTPI);
    }

    // The table of values, coded using the (prettier) C++14 syntax for `constexpr`.
    // If needed, we could re-code this for C++11...
    struct Table {

        double sqrt_int[MAXSQRT + 1];
        double invsqrt_int[MAXSQRT + 1];
        double factorial[MAXFACT + 1];
        double double_factorial[MAXDOUBLEFACT + 1];
        double half_factorial_pos[2 * MAXFACT + 1];
        double half_factorial_neg[2 * MAXFACT + 1];

        constexpr Table() : sqrt_int(), invsqrt_int(), factorial(), double_factorial(),
                half_factorial_pos(), half_factorial_neg() {
            for (auto i = 0; i <= MAXSQRT; ++i) {
                sqrt_int[i] = sqrt_(i);
                invsqrt_int[i] = i > 0 ? 1. / sqrt_int[i] : INFINITY;
            }
            for (auto i = 0; i <= MAXFACT; ++i) {
                factorial[i] = factorial_(i);
            }
            for (auto i = 0; i <= MAXDOUBLEFACT; ++i) {
                double_factorial[i] = double_factorial_(i);
            }
            for (auto i = 0; i <= 2 * MAXFACT; ++i) {
                half_factorial_pos[i] = half_factorial_pos_(i);
                half_factorial_neg[i] = half_factorial_neg_(i);
            }
        }

    };

    // Instantiate the table
    constexpr auto table = Table();

    // Square root of n
    template <typename T>
    T sqrt_int(int n) {
        if (n < 0)
            throw errors::SqrtNegativeNumber();
        else if (n > MAXSQRT)
            return sqrt(T(n));
        else
            return T(table.sqrt_int[n]);
    }

    template <>
    Multi sqrt_int(int n) {
        if (n < 0)
            throw errors::SqrtNegativeNumber();
        else
            return sqrt(Multi(n));
    }

    // Inverse of the square root of n
    template <typename T>
    T invsqrt_int(int n) {
        if (n < 0)
            throw errors::SqrtNegativeNumber();
        else if (n > MAXSQRT)
            return 1.0 / sqrt(T(n));
        else
            return T(table.invsqrt_int[n]);
    }

    template <>
    Multi invsqrt_int(int n) {
        if (n < 0)
            throw errors::SqrtNegativeNumber();
        else
            return 1.0 / sqrt(Multi(n));
    }

    // Factorial of n
    template <typename T>
    inline T factorial(int n) {
        if (n < 0)
            return T(INFINITY);
        else if (n > MAXFACT)
            return T(INFINITY);
        else
            return T(table.factorial[n]);
    }

    template <>
    inline Multi factorial(int n) {
        if (n < 0)
            return Multi(INFINITY);
        else
            return boost::math::factorial<Multi>(n);
    }

    // Double factorial of n
    template <typename T>
    inline T double_factorial(int n) {
        if (n < 0) {
            if ((-n) % 2 == 0)
                return T(INFINITY);
            else if (n == -1)
                return T(1);
            else
                return pow(-1, (-n - 1) / 2) / double_factorial<T>(-2 - n);
        } else if (n > MAXDOUBLEFACT)
            return T(INFINITY);
        else
            return T(table.double_factorial[n]);
    }

    template <>
    inline Multi double_factorial(int n) {
        if (n < 0) {
            if ((-n) % 2 == 0)
                return Multi(INFINITY);
            else if (n == -1)
                return Multi(1);
            else
                return pow(-1, (-n - 1) / 2) / double_factorial<Multi>(-2 - n);
        } else
            return boost::math::double_factorial<Multi>(n);
    }

    // Factorial of (n / 2)
    template <typename T>
    inline T half_factorial(int n) {
        if (n > 2 * MAXFACT)
            return T(INFINITY);
        else {
            if (n < 0)
                return T(table.half_factorial_neg[-n]);
            else
                return T(table.half_factorial_pos[n]);
        }
    }

    template <>
    inline Multi half_factorial(int n) {
        if (n % 2 == 0) {
            if (n < 0)
                return Multi(INFINITY);
            else if (n == 0)
                return 1;
            else
                return boost::math::factorial<Multi>(n / 2);
        } else {
            return boost::math::tgamma<Multi>(1.0 + n / 2.0);
        }
    }

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

}

#endif
