/**
\file scatter.h
\brief Illumination and scattering transforms.

*/

#ifndef _STARRY_REFLECTED_SCATTER_H_
#define _STARRY_REFLECTED_SCATTER_H_

#include "../utils.h"
#include "constants.h"

namespace starry {
namespace reflected {
namespace scatter {

using namespace utils;

template <typename T>
inline void computeI(Eigen::SparseMatrix<T> &I, const T &b, const T &theta) {

  // Compute illumination x, y, z coefficients
  T y0, x, y, z;
  if (abs(b) < 1 - STARRY_B_ONE_TOL) {
    y0 = sqrt(1 - b * b);
    x = -y0 * sin(theta);
    y = y0 * cos(theta);
  } else {
    x = 0;
    y = 0;
  }
  z = -b;
  Vector<T> p(3);

  // Illumination function in the polynomial basis
  p << x, z, y;

  // Input degree and (empirical) approximate
  // number of coefficients in the matrix
  int deg_in = (int)(sqrt(I.cols()) - 1);
  int nillum = max(4, int(ceil(5.5 * deg_in * deg_in + 9.5 * deg_in + 3)));

  // Populate the triplets
  typedef Eigen::Triplet<T> T3;
  std::vector<T3> tripletList;
  tripletList.reserve(nillum);
  int n1 = 0;
  int n2 = 0;
  int l, n;
  bool odd1;
  for (int l1 = 0; l1 < deg_in + 1; ++l1) {
    for (int m1 = -l1; m1 < l1 + 1; ++m1) {
      if (is_even(l1 + m1))
        odd1 = false;
      else
        odd1 = true;
      n2 = 0;
      for (int m2 = -1; m2 < 2; ++m2) {
        l = l1 + 1;
        n = l * l + l + m1 + m2;
        if (odd1 && (is_even(m2))) {
          tripletList.push_back(T3(n - 4 * l + 2, n1, p(n2)));
          tripletList.push_back(T3(n - 2, n1, -p(n2)));
          tripletList.push_back(T3(n + 2, n1, -p(n2)));
        } else {
          tripletList.push_back(T3(n, n1, p(n2)));
        }
        n2 += 1;
      }
      n1 += 1;
    }
  }

  // Create the sparse illumination matrix
  I.setFromTriplets(tripletList.begin(), tripletList.end());
}

} // namespace scatter
} // namespace reflected
} // namespace starry

#endif
