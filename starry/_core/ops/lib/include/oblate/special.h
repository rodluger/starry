/**
\file special.h
\brief Special functions.

*/

#ifndef _STARRY_OBLATE_SPECIAL_H_
#define _STARRY_OBLATE_SPECIAL_H_

#include "../quad.h"
#include "../utils.h"
#include "constants.h"

namespace starry {
namespace oblate {
namespace special {

using std::abs;
using namespace utils;
using namespace quad;

/**
  The Gauss hypergeometric function 2F1.

*/
template <typename T>
inline T hyp2f1(const T &a_, const T &b_, const T &c_, const T &z) {

  // Compute the value
  T a = a_;
  T b = b_;
  T c = c_;
  T term = a * b * z / c;
  T value = 1.0 + term;
  int n = 1;
  while ((abs(term) > STARRY_2F1_MAXTOL) && (n < STARRY_2F1_MAXITER)) {
    a += 1;
    b += 1;
    c += 1;
    n += 1;
    term *= a * b * z / c / n;
    value += term;
  }
  if ((n == STARRY_2F1_MAXITER) && (abs(term) > STARRY_2F1_MINTOL)) {
    std::stringstream args;
    args << "a_ = " << a_ << ", "
         << "b_ = " << b_ << ", "
         << "c_ = " << c_ << ", "
         << "z = " << z;
    throw StarryException("Series for 2F1 did not converge.",
                          "oblate/special.h", "hyp2f1", args.str());
  }
  return value;
}

/**
  The Gauss hypergeometric function 2F1 and its z derivative.

*/
template <typename T, int N>
inline ADScalar<T, N> hyp2f1(const T &a, const T &b, const T &c,
                             const ADScalar<T, N> &z) {
  ADScalar<T, N> F;
  F.value() = hyp2f1(a, b, c, z.value());
  F.derivatives() =
      z.derivatives() * a * b / c * hyp2f1(a + 1, b + 1, c + 1, z.value());
  return F;
}

} // namespace special
} // namespace oblate
} // namespace starry

#endif
