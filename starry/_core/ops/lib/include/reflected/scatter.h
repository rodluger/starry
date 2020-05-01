/**
\file scatter.h
\brief Illumination and scattering transforms.

*/

#ifndef _STARRY_REFLECTED_SCATTER_H_
#define _STARRY_REFLECTED_SCATTER_H_

#include "../basis.h"
#include "../utils.h"
#include "constants.h"
#include "oren_nayar.h"

namespace starry {
namespace reflected {
namespace scatter {

using namespace utils;

template <typename T>
inline void computeI_Lambertian(const int deg, Eigen::SparseMatrix<T> &I,
                                const T &b, const T &theta) {

  int N_out = (deg + 2) * (deg + 2);
  int N_in = (deg + 1) * (deg + 1);
  I.resize(N_out, N_in);

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

  // Empirical number of coefficients in the matrix (approximate)
  int nillum = max(4, int(ceil(5.5 * deg * deg + 9.5 * deg + 3)));

  // Populate the triplets
  typedef Eigen::Triplet<T> T3;
  std::vector<T3> tripletList;
  tripletList.reserve(nillum);
  int n1 = 0;
  int n2 = 0;
  int l, n;
  bool odd1;
  for (int l1 = 0; l1 < deg + 1; ++l1) {
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

template <typename T, typename Scalar>
inline Vector<T> OrenNayarPolynomial(const T &b, const T &theta, const T &sigr,
                                     const basis::Basis<Scalar> &B) {

  /*
    This is the function

        \cos\theta_i
        \mathrm{max}\left(0, \cos(\phi_r - \phi_i)\right)
        \sin\alpha \tan\beta

    from Equation (30) in Oren & Nayar (1994)
  */
  Vector<T> f(STARRY_OREN_NAYAR_N);
  f.setZero();

  // Oren-Nayar roughness coefficients
  T sig2 = sigr * sigr;
  T cA = 1.0 - 0.5 * sig2 / (sig2 + 0.33);
  T cB = 0.45 * sig2 / (sig2 + 0.09);

  // Terminator complement
  T bc = sqrt(1 - b * b);

  // The function `f` is zero everywhere if `b >= 0`,
  // since in this case `cos(phi_r - phi_i) <= 0`.
  if (b < 0) {

    // Construct the bbc basis
    // NOTE: The lowest power of `b` is *ONE*, since
    // we need `f = 0` eveywhere when `b = 0` for
    // a smooth transition to Lambertian at crescent
    // phase.
    Vector<T> bbc(STARRY_OREN_NAYAR_NB * STARRY_OREN_NAYAR_NB);
    bbc(0) = b;
    for (int l = 1; l < STARRY_OREN_NAYAR_NB; ++l) {
      bbc(l) = bc * bbc(l - 1);
    }
    for (int k = 1; k < STARRY_OREN_NAYAR_NB; ++k) {
      bbc(k * STARRY_OREN_NAYAR_NB) = b * bbc((k - 1) * STARRY_OREN_NAYAR_NB);
      for (int l = 1; l < STARRY_OREN_NAYAR_NB; ++l) {
        bbc(k * STARRY_OREN_NAYAR_NB + l) =
            bc * bbc(k * STARRY_OREN_NAYAR_NB + l - 1);
      }
    }

    // Sum over the bbc basis to obtain the Oren-Nayar
    // coefficients in the polynomial basis
    for (int n = 0; n < STARRY_OREN_NAYAR_N; ++n) {
      for (int m = 0; m < STARRY_OREN_NAYAR_NB * STARRY_OREN_NAYAR_NB; ++m) {
        f(n) += STARRY_OREN_NAYAR_COEFFS[m +
                                         STARRY_OREN_NAYAR_NB *
                                             STARRY_OREN_NAYAR_NB * n] *
                bbc(m);
      }
    }
  }

  // Account for the roughness and add in the Lambertian term
  Vector<T> p = cB * f;
  p(2) -= cA * b;
  p(3) += cA * bc;

  // Rotate the polynomial to the correct orientation on the sky
  if (theta != 0.0) {

    Vector<T> cosnt(STARRY_OREN_NAYAR_DEG + 1);
    Vector<T> sinnt(STARRY_OREN_NAYAR_DEG + 1);
    Vector<T> cosmt(STARRY_OREN_NAYAR_N);
    Vector<T> sinmt(STARRY_OREN_NAYAR_N);
    cosnt(0) = 1.0;
    sinnt(0) = 0.0;

    // Transform to ylms
    Vector<T> A1Invp = B.A1Inv_Reflected.block(0, 0, STARRY_OREN_NAYAR_N,
                                               STARRY_OREN_NAYAR_N) *
                       p;

    // Rotate on the sky
    Vector<T> RA1Invp(STARRY_OREN_NAYAR_N);
    cosnt(1) = cos(theta);
    sinnt(1) = sin(-theta);
    for (int n = 2; n < STARRY_OREN_NAYAR_DEG + 1; ++n) {
      cosnt(n) = 2.0 * cosnt(n - 1) * cosnt(1) - cosnt(n - 2);
      sinnt(n) = 2.0 * sinnt(n - 1) * cosnt(1) - sinnt(n - 2);
    }
    int n = 0;
    for (int l = 0; l < STARRY_OREN_NAYAR_DEG + 1; ++l) {
      for (int m = -l; m < 0; ++m) {
        cosmt(n) = cosnt(-m);
        sinmt(n) = -sinnt(-m);
        ++n;
      }
      for (int m = 0; m < l + 1; ++m) {
        cosmt(n) = cosnt(m);
        sinmt(n) = sinnt(m);
        ++n;
      }
      for (int j = 0; j < 2 * l + 1; ++j) {
        RA1Invp(l * l + j) = A1Invp(l * l + j) * cosmt(l * l + j) +
                             A1Invp(l * l + 2 * l - j) * sinmt(l * l + j);
      }
    }

    // Transform back to polynomials
    p = B.A1_Reflected.block(0, 0, STARRY_OREN_NAYAR_N, STARRY_OREN_NAYAR_N) *
        RA1Invp;
  }

  return p;
};

template <typename T, typename Scalar>
inline void computeI_OrenNayar(const int deg, Eigen::SparseMatrix<T> &I,
                               const T &b, const T &theta, const T &sigr,
                               const basis::Basis<Scalar> &B) {

  // Get the polynomial illumination function
  Vector<T> p = OrenNayarPolynomial(b, theta, sigr, B);

  // Declare the output
  int N_out =
      (deg + 1 + STARRY_OREN_NAYAR_DEG) * (deg + 1 + STARRY_OREN_NAYAR_DEG);
  int N_in = (deg + 1) * (deg + 1);
  I.resize(N_out, N_in);

  // Populate the triplets
  typedef Eigen::Triplet<T> T3;
  std::vector<T3> tripletList;
  tripletList.reserve(
      int(1.25 * STARRY_OREN_NAYAR_N * N_in)); // very approximate!
  int n1 = 0;
  int n2 = 0;
  int l, n;
  bool odd1;
  for (int l1 = 0; l1 < deg + 1; ++l1) {
    for (int m1 = -l1; m1 < l1 + 1; ++m1) {
      odd1 = !is_even(l1 + m1);
      n2 = 0;
      for (int l2 = 0; l2 < STARRY_OREN_NAYAR_DEG + 1; ++l2) {
        for (int m2 = -l2; m2 < l2 + 1; ++m2) {
          l = l1 + l2;
          n = l * l + l + m1 + m2;
          if (odd1 && (!is_even(l2 + m2))) {
            tripletList.push_back(T3(n - 4 * l + 2, n1, p(n2)));
            tripletList.push_back(T3(n - 2, n1, -p(n2)));
            tripletList.push_back(T3(n + 2, n1, -p(n2)));
          } else {
            tripletList.push_back(T3(n, n1, p(n2)));
          }
          n2 += 1;
        }
      }
      n1 += 1;
    }
  }

  // Create the sparse illumination matrix
  I.setFromTriplets(tripletList.begin(), tripletList.end());
}

template <typename T, typename Scalar>
inline void computeI(const int deg, Eigen::SparseMatrix<T> &I, const T &b,
                     const T &theta, const T &sigr,
                     const basis::Basis<Scalar> &B) {
  if (sigr > 0.0)
    computeI_OrenNayar(deg, I, b, theta, sigr, B);
  else
    computeI_Lambertian(deg, I, b, theta);
}

} // namespace scatter
} // namespace reflected
} // namespace starry

#endif
