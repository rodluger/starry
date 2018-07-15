/**
Tables of square roots and factorials for fast `double` math
using good old template metaprogramming (>= C++14)

*/

#ifndef _STARRY_TABLES_H_
#define _STARRY_TABLES_H_

#include <cmath>
#include <boost/math/special_functions/factorials.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <cstddef>
#include <array>
#include "errors.h"

// Square root of pi at double precision
#define SQRTPI 1.772453850905516027298167483341

// Largest factorial computable at double precision
#define MAXFACT 170

// Largest double factorial computable at double precision
#define MAXDOUBLEFACT 300

// Largest square root we're willing to tabulate
#define MAXSQRT 300

// Compile-time-generated tables of square roots and factorials
// closely following this thread: https://stackoverflow.com/a/19016627
namespace const_tables {

    template<std::size_t... Is>
    struct seq{};

    template<std::size_t N, std::size_t... Is>
    struct gen_seq : gen_seq<N-1, N-1, Is...>{};

    template<std::size_t... Is>
    struct gen_seq<0, Is...> : seq<Is...>{};

    template<class Generator, std::size_t... Is>
    constexpr auto generate_array_helper(Generator g, seq<Is...>)
      -> std::array<decltype(g(std::size_t{}, sizeof...(Is))), sizeof...(Is)> {
      return {{g(Is, sizeof...(Is))...}};
    }

    template<std::size_t tcount, class Generator>
    constexpr auto generate_array(Generator g)
      -> decltype( generate_array_helper(g, gen_seq<tcount>{}) ) {
      return generate_array_helper(g, gen_seq<tcount>{});
    }

    double constexpr sqrt_rec(double x, double curr, double prev) {
        return curr == prev ? curr : sqrt_rec(x, 0.5 * (curr + x / curr), curr);
    }

    double constexpr sqrt_(std::size_t curr, std::size_t total) {
        return curr > 0 ? sqrt_rec((double)curr, (double)curr, 0) : 0;
    }

    // NOTE: Since infinity is not a constexpr, invsqrt_(0) = 0
    // We override this behavior in `invsqrt()` below.
    double constexpr invsqrt_(std::size_t curr, std::size_t total) {
        return curr > 0 ?
               1. / sqrt_rec((double)curr, (double)curr, 0):
               0;
    }

    // NOTE: Since infinity is not a constexpr, factorial_(MAXFACT) = 0
    // We override this behavior in `factorial()` below.
    double constexpr factorial_(std::size_t curr, std::size_t total) {
        return curr > 0 ?
               (curr <= MAXFACT ? curr * factorial_(curr - 1, total) : 0):
               1.0;
    }

    // NOTE: Since infinity is not a constexpr, double_factorial_(MAXFACT) = 0
    // We override this behavior in `double_factorial()` below.
    double constexpr double_factorial_(std::size_t curr, std::size_t total) {
        return curr > 0 ?
               (curr <= MAXDOUBLEFACT ? curr * double_factorial_(curr - 2, total) : 0):
               1.0;
    }

    double constexpr half_factorial_pos_(std::size_t curr, std::size_t total) {
        return curr > 1 ?
               (curr % 2 == 0 ? factorial_(curr / 2, total) : half_factorial_pos_(curr - 2, total) * (curr / 2.0)) :
               (curr == 0 ? 1 : 0.5 * SQRTPI);
    }

    double constexpr half_factorial_neg_(std::size_t curr, std::size_t total) {
        return curr > 1 ?
               (curr % 2 == 0 ? INFINITY : half_factorial_neg_(curr - 2, total) / (1.0 - curr / 2.0)) :
               (curr == 0 ? 1 : SQRTPI);
    }

    // The compile-time tabulated arrays
    constexpr auto sqrt_int = generate_array<MAXSQRT + 1>(sqrt_);
    constexpr auto invsqrt_int = generate_array<MAXSQRT + 1>(invsqrt_);
    constexpr auto factorial = generate_array<MAXFACT + 1>(factorial_);
    constexpr auto double_factorial = generate_array<MAXDOUBLEFACT + 1>(double_factorial_);
    constexpr auto half_factorial_pos = generate_array<2 * MAXFACT + 1>(half_factorial_pos_);
    constexpr auto half_factorial_neg = generate_array<2 * MAXFACT + 1>(half_factorial_neg_);

}

namespace tables {

    // Square root of n
    template <typename T>
    inline T sqrt_int(int n) {
        if (n < 0)
            throw errors::SqrtNegativeNumber();
        else if (n > MAXSQRT)
            return sqrt(T(n));
        else
            return T(const_tables::sqrt_int[n]);
    }

    template <>
    inline Multi sqrt_int(int n) {
        if (n < 0)
            throw errors::SqrtNegativeNumber();
        else
            return sqrt(Multi(n));
    }

    // Inverse of the square root of n
    template <typename T>
    inline T invsqrt_int(int n) {
        if (n < 0)
            throw errors::SqrtNegativeNumber();
        else if (n > MAXSQRT)
            return 1.0 / sqrt(T(n));
        else if (n == 0)
            return INFINITY;
        else
            return T(const_tables::invsqrt_int[n]);
    }

    template <>
    inline Multi invsqrt_int(int n) {
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
            return T(const_tables::factorial[n]);
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
            return T(const_tables::double_factorial[n]);
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
                return T(const_tables::half_factorial_neg[-n]);
            else
                return T(const_tables::half_factorial_pos[n]);
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
