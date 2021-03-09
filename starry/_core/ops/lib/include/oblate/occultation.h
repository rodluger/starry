
/**
\file occultation.h
\brief Solver for occultations of oblate bodies.

*/

#ifndef _STARRY_OBLATE_OCCULTATION_H_
#define _STARRY_OBLATE_OCCULTATION_H_

#include "../quad.h"
#include "../utils.h"
#include "constants.h"
#include "ellip.h"

namespace starry {
namespace oblate {
namespace occultation {

using namespace utils;

template <class T> class Occultation {

  using Scalar = typename T::Scalar;

protected:
  // Misc
  int deg;
  int N;

  // Numerical integration
  quad::Quad<Scalar> QUAD;

public:
  RowVector<T> sT;

  explicit Occultation(int deg) : deg(deg), N((deg + 1) * (deg + 1)) {}

  /**
      Compute the full solution vector s^T.

  */
  inline void compute(const T &f, const T &theta, const T &bo, const T &ro) {

    // TODO
  }
};

} // namespace occultation
} // namespace oblate
} // namespace starry

#endif