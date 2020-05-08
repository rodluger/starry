/**
\file primitive.h
\brief Primitive integrals for reflected light occultations.

*/

#ifndef _STARRY_PRIMITIVE_H_
#define _STARRY_PRIMITIVE_H_

#include "../utils.h"
#include "constants.h"
#include "ellip.h"
#include "special.h"

namespace starry {
namespace reflected {
namespace primitive {

using namespace utils;
using namespace special;
using namespace ellip;
using namespace quad;

/**
    Vieta's theorem coefficient A_{i,u,v}

*/
template <class T> class Vieta {
protected:
  int umax;
  int vmax;
  T res;
  T u_choose_j1;
  T v_choose_c0;
  T fac;
  Vector<T> delta;
  Matrix<bool> set;
  Matrix<Vector<T>> vec;

  //! Compute the double-binomial coefficient A_{i,u,v}
  inline void compute(int u, int v) {
    int j1 = u;
    int j2 = u;
    int c0 = v;
    int sgn0 = 1;
    u_choose_j1 = 1.0;
    v_choose_c0 = 1.0;
    for (int i = 0; i < u + v + 1; ++i) {
      res = 0;
      int c = c0;
      fac = sgn0 * u_choose_j1 * v_choose_c0;
      for (int j = j1; j < j2 + 1; ++j) {
        res += fac * delta(c);
        --c;
        fac *= -((u - j) * (c + 1.0)) / ((j + 1.0) * (v - c));
      }
      if (i >= v)
        --j2;
      if (i < u) {
        --j1;
        sgn0 *= -1;
        u_choose_j1 *= (j1 + 1.0) / (u - j1);
      } else {
        --c0;
        if (c0 < v)
          v_choose_c0 *= (c0 + 1.0) / (v - c0);
        else
          v_choose_c0 = 1.0;
      }
      vec(u, v)(i) = res;
    }
    set(u, v) = true;
  }

  //! Getter function
  inline Vector<T> &get_value(int u, int v) {
    CHECK_BOUNDS(u, 0, umax);
    CHECK_BOUNDS(v, 0, vmax);
    if (set(u, v)) {
      return vec(u, v);
    } else {
      compute(u, v);
      return vec(u, v);
    }
  }

public:
  //! Constructor
  explicit Vieta(int lmax)
      : umax(is_even(lmax) ? (lmax + 2) / 2 : (lmax + 3) / 2),
        vmax(lmax > 0 ? lmax : 1), delta(vmax + 1), set(umax + 1, vmax + 1),
        vec(umax + 1, vmax + 1) {
    delta(0) = 1.0;
    set.setZero();
    for (int u = 0; u < umax + 1; ++u) {
      for (int v = 0; v < vmax + 1; ++v) {
        vec(u, v).resize(u + v + 1);
      }
    }
  }

  //! Overload () to get the function value without calling `get_value()`
  inline Vector<T> &operator()(int u, int v) { return get_value(u, v); }

  //! Resetter
  void reset(const T &delta_) {
    set.setZero();
    for (int v = 1; v < vmax + 1; ++v) {
      delta(v) = delta(v - 1) * delta_;
    }
  }
};

/**

    Given s1 = sin(0.5 * kappa), compute the integral of

        cos(x) sin^{2v + 1}(x)

    from 0.5 * kappa1 to 0.5 * kappa2 recursively and return an array
    containing the values of this function from v = 0 to v = vmax.

*/
template <typename T> inline Vector<T> U(const int vmax, const Vector<T> &s2) {
  Vector<T> result(vmax + 1);
  Vector<T> term = s2;
  for (int v = 0; v < vmax + 1; ++v) {
    result(v) = pairdiff(term) / (2 * v + 2);
    term.array() *= s2.array();
  }
  return result;
}

/**
    Compute the helper integral I by upward and/or downward recursion.

    TODO: This code currently can't handle cases where the (lo, hi)
          limit pairs in the kappa vector belong to different regimes.
          These regimes are

                sin(kappa / 2) > 0.5 --> upward recursion
                sin(kappa / 2) <= 0.5 --> downward recursion

          Cases where the two angles are close to 0.5 are *fine*. Issues
          will occur when sin(kappa_0 / 2) ~ 0 and sin(kappa_1 / 2) ~ 1,
          in which case I don't know how to stably compute the result.
          I don't *think* these cases are encountered in practice, but
          we should look into it.

*/
template <typename T>
inline Vector<T> I(const int nmax, const Vector<T> &kappa_,
                   const Vector<T> &s1_, const Vector<T> &c1_,
                   Quad<typename T::Scalar> &QUAD) {
  Vector<T> result(nmax + 1);
  result.setZero();
  Vector<T> indef(nmax + 1);
  Vector<T> kappa(2), s1(2), c1(2), s2(2);

  // Loop through the pairs of limits
  size_t K = kappa_.size();
  for (size_t i = 0; i < K; i += 2) {

    // Current limits
    kappa = kappa_.segment(i, 2);
    s1 = s1_.segment(i, 2);
    c1 = c1_.segment(i, 2);
    s2.array() = s1.array() * s1.array();

    // Upward recursion; stability criterion from the
    // original starry paper, where s2 = k^2
    if ((abs(s2(0)) > 0.5) || (abs(s2(1)) > 0.5)) {

      indef(0) = 0.5 * pairdiff(kappa);
      Vector<T> term(2);
      term.array() = s1.array() * c1.array();
      for (int v = 1; v < nmax + 1; ++v) {
        indef(v) =
            (1.0 / (2.0 * v)) * ((2 * v - 1) * indef(v - 1) - pairdiff(term));
        term.array() *= s2.array();
      }

      // Downward recursion
    } else {

      // Compute the trig part upwards...
      Vector<T> term(nmax + 1);
      T termlo = s1(0) * c1(0);
      T termhi = s1(1) * c1(1);
      term(0) = termhi - termlo;
      for (int v = 1; v < nmax + 1; ++v) {
        termlo *= s2(0);
        termhi *= s2(1);
        term(v) = termhi - termlo;
      }

      // Evaluate numerically
      indef(nmax) = I_numerical(nmax, kappa, QUAD);

      // Recurse down
      for (int v = nmax - 1; v > -1; --v) {
        indef(v) = ((2.0 * v + 2.0) * indef(v + 1) + term(v)) / (2.0 * v + 1);
      }
    }

    // Definite integral
    result += indef;
  }

  return result;
}

/**
    Compute the expression

        0.5 * s^(2n + 2) (3 / (n + 1) * 2F1(-1/2, n + 1, n + 2, 1 - q^2) + 2q^3)
   / (2n
   + 5)

    evaluated at n = [0 .. nmax], where

        s = sin(1/2 kappa)
        q = (1 - s^2 / k^2)^1/2

    by either upward recursion (stable for |1 - q^2| > 1/2) or downward
    recursion (always stable).

*/
template <typename T>
inline Vector<T> W_indef(const int nmax, const T &s2_, const T &q2,
                         const T &q3) {
  Vector<T> result(nmax + 1);

  // TODO: Is this instability encountered in practice?
  // If so, find the limiting value of W when s2 = 0.
  T s2 = s2_;
  if (abs(s2) < STARRY_MIN_SIN_ALPHA)
    s2 = (s2 > 0) ? T(STARRY_MIN_SIN_ALPHA) : T(-STARRY_MIN_SIN_ALPHA);

  if (abs(1 - q2) < 0.5) {

    // Setup
    T invs2 = 1 / s2;
    T z = (1.0 - q2) * invs2;
    T s2nmax = pow(s2, nmax);
    T x = q2 * q3 * s2nmax;

    // Upper boundary condition
    result(nmax) =
        (s2 * s2nmax * (3.0 / (nmax + 1.0) *
                            hyp2f1(-0.5, nmax + 1.0, nmax + 2.0, T(1.0 - q2)) +
                        2 * q3) /
         (2.0 * nmax + 5.0));

    // Recurse down
    T f, A, B;
    for (int b = nmax - 1; b > -1; --b) {
      f = 1.0 / (b + 1.0);
      A = z * (1.0 + 2.5 * f);
      B = x * f;
      result(b) = A * result(b + 1) + B;
      x *= invs2;
    }

  } else {

    // Setup
    T z = s2 / (1.0 - q2);
    T x = -2.0 * q3 * (z - s2) * s2;

    // Lower boundary condition
    result(0) = (2.0 / 5.0) * (z * (1.0 - q3) + s2 * q3);

    // Recurse up
    T f, A, B;
    for (int b = 1; b < nmax + 1; ++b) {
      f = 1.0 / (2.0 * b + 5);
      A = z * (2.0 * b) * f;
      B = x * f;
      result(b) = A * result(b - 1) + B;
      x *= s2;
    }
  }

  return 0.5 * result;
}

/**
    Compute the definite helper integral W from W_indef.

*/
template <typename T>
inline Vector<T> W(const int nmax, const Vector<T> &s2, const Vector<T> &q2,
                   const Vector<T> &q3) {
  size_t K = s2.size();
  Vector<T> result(nmax + 1);
  result.setZero();
  for (size_t i = 0; i < K; i += 2) {
    result += W_indef(nmax, s2(i + 1), q2(i + 1), q3(i + 1)) -
              W_indef(nmax, s2(i), q2(i), q3(i));
  }
  return result;
}

/**
    Compute the helper integral J.

    Returns the array J[0 .. nmax], computed recursively using
    a tridiagonal solver and a lower boundary condition
    (analytic in terms of elliptic integrals) and an upper
    boundary condition (computed numerically).

*/
template <typename T>
inline Vector<T>
J(const int nmax, const T &k2, const T &km2, const Vector<T> &kappa,
  const Vector<T> &s1, const Vector<T> &s2, const Vector<T> &c1,
  const Vector<T> &q2, const T &F, const T &E, Quad<typename T::Scalar> &QUAD) {

  // Useful variables
  size_t K = kappa.size();
  Vector<T> z(K);
  Vector<T> sqrtq2(K);
  sqrtq2.array() = (q2.array() > 0).select(sqrt(q2.array()), 0.0);
  z.array() = s1.array() * c1.array() * sqrtq2.array();

#if (STARRY_USE_INCOMPLETE_INTEGRALS)
  // Lower boundary: analytic
  T f0 = (1.0 / 3.0) * (2 * (2 - km2) * E + (km2 - 1) * F + km2 * pairdiff(z));
#else
  // Lower boundary: numerical
  T f0 = J_numerical(0, k2, kappa, QUAD);
#endif

  // Upper boundary
  T fN = J_numerical(nmax, k2, kappa, QUAD);

  // Set up the tridiagonal problem
  Vector<T> a(nmax - 1), b(nmax - 1), c(nmax - 1);
  Vector<T> term(K);
  term.array() = k2 * z.array() * q2.array() * q2.array();
  T amp;
  int i = 0;
  for (int v = 2; v < nmax + 1; ++v) {
    amp = 1.0 / (2 * v + 3);
    a(i) = -2 * (v + (v - 1) * k2 + 1) * amp;
    b(i) = (2 * v - 3) * k2 * amp;
    c(i) = pairdiff(term) * amp;
    term.array() *= s2.array();
    ++i;
  }

  // Add the boundary conditions
  c(0) -= b(0) * f0;
  c(nmax - 2) -= fN;

  // Construct the tridiagonal matrix
  // TODO: We should probably use a sparse solve here!
  Matrix<T> A(nmax - 1, nmax - 1);
  A.setZero();
  A.diagonal(0) = a;
  A.diagonal(-1) = b.segment(1, nmax - 2);
  A.diagonal(1).setOnes();

  // Solve
  Vector<T> soln = A.lu().solve(c);

  // Append lower and upper boundary conditions
  Vector<T> result(nmax + 1);
  result(0) = f0;
  result(nmax) = fN;
  result.segment(1, nmax - 1) = soln;

  // We're done
  return result;
}

/**
    Compute the helper integral K.

*/
template <typename T>
inline T K(Vieta<T> &A, const Vector<T> &I, const int u, const int v) {
  return A(u, v).dot(I.segment(u, u + v + 1));
}

/**
    Compute the helper integral L.
    Note: factor of k^3 moved to parent scope.

*/
template <typename T>
inline T L(Vieta<T> &A, const Vector<T> &J, const int u, const int v,
           const int t) {
  return A(u, v).dot(J.segment(u + t, u + v + 1));
}

/**
    Compute the helper integral H.
    The forward derivatives are used in the recursion, so we compute them
   manually.

*/
template <typename T, int N>
inline Matrix<ADScalar<T, N>> H(const int uvmax,
                                const Vector<ADScalar<T, N>> &xi) {

  Matrix<ADScalar<T, N>> f(uvmax + 1, uvmax + 1);
  f.setZero();
  size_t K = xi.size();
  int sgn;

  // Split xi into its value and derivs
  Vector<T> xi_value(K);
  Matrix<T> dxi(N, K);
  for (size_t i = 0; i < K; ++i) {
    xi_value(i) = xi(i).value();
    dxi.col(i) = xi(i).derivatives();
  }

  // Helper vars
  Vector<T> c(K), s(K), cs(K), cc(K), ss(K);
  c.array() = cos(xi_value.array());
  s.array() = sin(xi_value.array());
  cs.array() = c.array() * s.array();
  cc.array() = c.array() * c.array();
  ss.array() = s.array() * s.array();

  // Compute H and dH / dxi
  Matrix<Vector<T>> df(uvmax + 1, uvmax + 1);

  // Lower boundary
  f(0, 0).value() = pairdiff(xi_value);
  df(0, 0).setOnes(K);
  f(1, 0).value() = pairdiff(s);
  df(1, 0) = c;
  f(0, 1).value() = -pairdiff(c);
  df(0, 1) = s;
  f(1, 1).value() = -0.5 * pairdiff(cc);
  df(1, 1) = cs;
  for (int u = 0; u < 2; ++u) {
    for (int v = 0; v < 2; ++v) {
      sgn = -1;
      for (size_t i = 0; i < K; ++i) {
        f(u, v).derivatives() += sgn * df(u, v)(i) * xi(i).derivatives();
        sgn *= -1;
      }
    }
  }

  // Recurse upward
  for (int u = 0; u < 2; ++u) {
    for (int v = 2; v < uvmax + 1 - u; ++v) {
      f(u, v).value() = (-pairdiff(Vector<T>(df(u, v - 2).cwiseProduct(cs))) +
                         (v - 1) * f(u, v - 2).value()) /
                        (u + v);
      df(u, v) = df(u, v - 2).cwiseProduct(ss);
      sgn = -1;
      for (size_t i = 0; i < K; ++i) {
        f(u, v).derivatives() += sgn * df(u, v)(i) * xi(i).derivatives();
        sgn *= -1;
      }
    }
  }
  for (int u = 2; u < uvmax + 1; ++u) {
    for (int v = 0; v < uvmax + 1 - u; ++v) {
      f(u, v).value() = (pairdiff(Vector<T>(df(u - 2, v).cwiseProduct(cs))) +
                         (u - 1) * f(u - 2, v).value()) /
                        (u + v);
      df(u, v) = df(u - 2, v).cwiseProduct(cc);
      sgn = -1;
      for (size_t i = 0; i < K; ++i) {
        f(u, v).derivatives() += sgn * df(u, v)(i) * xi(i).derivatives();
        sgn *= -1;
      }
    }
  }

  return f;
}

/**
    Compute the primitive integral T[2].

    Note that these expressions are only valid for b >= 0.

*/
template <typename T> inline T T2_indef(const T &b, const T &xi) {

  // Helper vars
  T c = cos(xi);
  T s = sin(xi);
  T bc = sqrt(1 - b * b);
  T bbc = b * bc;

  // Special cases
  if (abs(xi - 0) < STARRY_ANGLE_TOL)
    return -(arctan(T((2 * b * b - 1) / (2 * bbc))) + bbc) / 3;
  else if (abs(xi - 0.5 * pi<T>()) < STARRY_ANGLE_TOL)
    return (0.5 * pi<T>() - arctan(T(b / bc))) / 3;
  else if (abs(xi - pi<T>()) < STARRY_ANGLE_TOL)
    return (0.5 * pi<T>() + bbc) / 3;
  else if (abs(xi - 1.5 * pi<T>()) < STARRY_ANGLE_TOL)
    return (0.5 * pi<T>() + arctan(T(b / bc)) + 2 * bbc) / 3;

  // Figure out the offset
  T delta;
  if (xi < 0.5 * pi<T>())
    delta = 0;
  else if (xi < pi<T>())
    delta = pi<T>();
  else if (xi < 1.5 * pi<T>())
    delta = 2 * bbc;
  else
    delta = pi<T>() + 2 * bbc;

  // We're done
  T term = arctan(T(b * s / c));
  int sgn = s > 0 ? 1 : s < 0 ? -1 : 0;
  return (term -
          sgn * (arctan(T(((s / (1 + c)) * (s / (1 + c)) + 2 * b * b - 1) /
                          (2 * bbc))) +
                 bbc * c) +
          delta) /
         3.0;
}

/**
    Compute the primitive integral T.

*/
template <typename S>
inline void computeT(const int ydeg, const S &b_, const S &theta,
                     const Vector<S> &xi, Vector<S> &T) {

  // Check for trivial result
  T.setZero((ydeg + 1) * (ydeg + 1));
  if (xi.size() == 0)
    return;

  // Pre-compute H
  Matrix<S> HIntegral = H(ydeg + 2, xi);

  // HACK: We should derive exact expressions in this limit.
  S b = b_;
  if (abs(b_) < STARRY_B_ZERO_TOL) {
    b = STARRY_B_ZERO_TOL;
    if (b_ < 0)
      b *= -1;
  }

  // Vars
  int jmax, kmax, mu, nu, l, j, k, n1, n3, n4, n5, p, q;
  S Z, Z0, Z1, Z2, Z_1, Z_5, fac;
  size_t K = xi.size();
  S ct = cos(theta);
  S st = sin(theta);
  S ttinvb = st / (b * ct);
  S invbtt = ct / (b * st);
  S b32 = (1 - b * b) * sqrt(1 - b * b);
  S bct = b * ct;
  S bst = b * st;

  // Case 2 (special
  int sgn = b > 0 ? -1 : b < 0 ? 1 : 0;
  for (size_t i = 0; i < K; ++i) {
    T(2) += sgn * T2_indef(S(abs(b)), xi(i));
    sgn *= -1;
  }

  // Special limit: sin(theta) = 0
  // TODO: eliminate the calls to pow in favor of recursion
  if (abs(st) < STARRY_T_TOL) {

    int sgnct = ct > 0 ? 1 : ct < 0 ? -1 : 0;
    int n = 0;
    for (int l = 0; l < ydeg + 1; ++l) {
      for (int m = -l; m < l + 1; ++m) {
        int mu = l - m;
        int nu = l + m;
        if (nu % 2 == 0) {
          T(n) = pow(sgnct, l) * pow(b, (1 + nu / 2)) *
                 HIntegral((mu + 4) / 2, nu / 2);
        } else {
          if (mu == 1) {
            if (l % 2 == 0)
              T(n) = -sgnct * b32 * HIntegral(l - 2, 4);
            else if (l > 1)
              T(n) = -b * b32 * HIntegral(l - 3, 5);
          } else {
            T(n) =
                pow(sgnct, (l - 1)) * (b32 * pow(b, ((nu + 1) / 2)) *
                                       HIntegral((mu - 1) / 2, (nu + 5) / 2));
          }
        }
        ++n;
      }
    }
    return;

  }

  // Special limit: cos(theta) = 0
  // TODO: eliminate the calls to pow in favor of recursion
  else if (abs(ct) < STARRY_T_TOL) {

    int sgnst = st > 0 ? 1 : st < 0 ? -1 : 0;
    int n = 0;
    for (int l = 0; l < ydeg + 1; ++l) {
      for (int m = -l; m < l + 1; ++m) {
        int mu = l - m;
        int nu = l + m;
        if (nu % 2 == 0) {
          T(n) = pow(b, ((mu + 2) / 2)) * HIntegral(nu / 2, (mu + 4) / 2);
          if (sgnst == 1)
            T(n) *= pow(-1, (mu / 2));
          else
            T(n) *= pow(-1, (nu / 2));
        } else {
          if (mu == 1) {
            if (l % 2 == 0) {
              T(n) = (pow(-sgnst, l - 1) * pow(b, l - 1) * b32 *
                      HIntegral(1, l + 1));
            } else if (l > 1) {
              T(n) = pow(b, l - 2) * b32 * HIntegral(2, l);
              if (sgnst == 1)
                T(n) *= pow(-1, l);
              else
                T(n) *= -1;
            }
          } else {
            T(n) = (b32 * pow(b, (mu - 3) / 2) *
                    HIntegral((nu - 1) / 2, (mu + 5) / 2));
            if (sgnst == 1)
              T(n) *= pow(-1, (mu - 1) / 2);
            else
              T(n) *= pow(-1, (nu - 1) / 2);
          }
        }
        ++n;
      }
    }
    return;
  }

  // Cases 1 and 5
  Z0 = 1.0;
  jmax = 0;
  for (nu = 0; nu < 2 * ydeg + 1; nu += 2) {
    kmax = 0;
    Z1 = Z0;
    for (mu = 0; mu < 2 * ydeg + 1 - nu; mu += 2) {
      l = (mu + nu) / 2;
      n1 = l * l + nu;
      n5 = (l + 2) * (l + 2) + nu + 1;
      Z2 = Z1;
      for (j = 0; j < jmax + 1; ++j) {
        Z_1 = -bst * Z2;
        Z_5 = b32 * Z2;
        for (k = 0; k < kmax + 1; ++k) {
          p = j + k;
          q = l + 1 - (j + k);
          fac = -invbtt / (k + 1.0);
          T(n1) += Z_1 * (bct * HIntegral(p + 1, q) - st * HIntegral(p, q + 1));
          Z_1 *= (kmax + 1 - k) * fac;
          if (n5 < (ydeg + 1) * (ydeg + 1)) {
            T(n5) += Z_5 *
                     (bct * HIntegral(p + 1, q + 2) - st * HIntegral(p, q + 3));
            Z_5 *= (kmax - k) * fac;
          }
        }
        T(n1) +=
            Z_1 * (bct * HIntegral(p + 2, q - 1) - st * HIntegral(p + 1, q));
        Z2 *= (jmax - j) / (j + 1.0) * ttinvb;
      }
      kmax += 1;
      Z1 *= -bst;
    }
    jmax += 1;
    Z0 *= bct;
  }

  // Cases 3 and 4
  Z0 = b32;
  kmax = 0;
  for (l = 2; l < ydeg + 1; l += 2) {
    n3 = l * l + 2 * l - 1;
    n4 = (l + 1) * (l + 1) + 2 * l + 1;
    Z = Z0;
    for (k = 0; k < kmax + 1; ++k) {
      p = k;
      q = l + 1 - k;
      T(n3) -= Z * (bst * HIntegral(p + 1, q) + ct * HIntegral(p, q + 1));
      if (l < ydeg) {
        T(n4) -= Z * (bst * st * HIntegral(p + 2, q) +
                      bct * ct * HIntegral(p, q + 2) +
                      (1 + b * b) * st * ct * HIntegral(p + 1, q + 1));
      }
      Z *= -(kmax - k) / (k + 1.0) * invbtt;
    }
    kmax += 2;
    Z0 *= bst * bst;
  }

  return;
}

/**
    Compute the primitive integral Q.

*/
template <typename T>
inline void computeQ(const int ydeg, const Vector<T> &lam, Vector<T> &Q) {

  // Allocate
  Q.setZero((ydeg + 1) * (ydeg + 1));

  // Check for trivial result
  if (lam.size() == 0)
    return;

  // Pre-compute H
  Matrix<T> HIntegral = H(ydeg + 2, lam);

  // Note that the linear term is special
  Q(2) = pairdiff(lam) / 3.0;

  // Easy!
  int n = 0;
  int mu, nu;
  for (int l = 0; l < ydeg + 1; ++l) {
    for (int m = -l; m < l + 1; ++m) {
      mu = l - m;
      nu = l + m;
      if (nu % 2 == 0) {
        Q(n) = HIntegral((mu + 4) / 2, nu / 2);
      }
      ++n;
    }
  }
}

/**
    Compute the primitive integral P.

*/
template <typename T>
inline void computeP(const int ydeg, const T &bo_, const T &ro_,
                     const Vector<T> &kappa, Vector<T> &P,
                     Quad<typename T::Scalar> &QUAD) {

  // Check for trivial result
  P.resize((ydeg + 1) * (ydeg + 1));
  if (kappa.size() == 0) {
    P.setZero();
    return;
  }

  // Nudge away from instability at bo = ro
  T bo = bo_;
  T ro = ro_;
  if (abs(bo - ro) < STARRY_BO_EQUALS_RO_TOL)
    bo = ro + (bo > ro ? STARRY_BO_EQUALS_RO_TOL : -STARRY_BO_EQUALS_RO_TOL);

  // Basic variables
  T delta = (bo - ro) / (2 * ro);
  T k2 = (1 - ro * ro - bo * bo + 2 * bo * ro) / (4 * bo * ro);
  T km2 = 1.0 / k2;
  T k3fourbr15 = pow(1 - ro * ro - bo * bo + 2 * bo * ro, 1.5);
  Vector<T> tworo(ydeg + 4);
  tworo(0) = 1.0;
  for (int i = 1; i < ydeg + 4; ++i) {
    tworo(i) = tworo(i - 1) * 2 * ro;
  }
  Vieta<T> A(ydeg);

  // Pre-compute the helper integrals
  size_t M = kappa.size();
  Vector<T> x(M), s1(M), s2(M), c1(M), q2(M), q3(M);
  x = 0.5 * kappa;
  s1.array() = sin(x.array());
  s2.array() = s1.array() * s1.array();
  c1.array() = cos(x.array());
  q2.array() = (s2.array() * km2 < 1.0).select(1.0 - s2.array() * km2, 0.0);
  q3.array() = (q2.array() > 0).select(q2.array() * sqrt(q2.array()), 0.0);
  Vector<T> UIntegral = U(ydeg + 2, s2);
  Vector<T> IIntegral = I(ydeg + 3, kappa, s1, c1, QUAD);
  Vector<T> WIntegral = W(ydeg, s2, q2, q3);
  A.reset(delta);

// Compute the elliptic integrals?
#if (STARRY_USE_INCOMPLETE_INTEGRALS)
  auto integrals =
      IncompleteEllipticIntegrals<typename T::Scalar>(bo, ro, kappa);
  T F = integrals.F;
  T E = integrals.E;
  T PIp = integrals.PIp;
#else
  T F = 0;
  T E = 0;
  T PIp = 0;
#endif

  // Compute J
  Vector<T> JIntegral;
  if (km2 > 0.0) {
    // Compute by recursion
    JIntegral = J(ydeg + 1, k2, km2, kappa, s1, s2, c1, q2, F, E, QUAD);
  } else {
    // Special limit, k2 -> inf
    JIntegral = IIntegral.head(ydeg + 2);
  }

  // Now populate the P array
  int n = 0;
  int mu, nu;
  for (int l = 0; l < ydeg + 1; ++l) {
    for (int m = -l; m < l + 1; ++m) {

      mu = l - m;
      nu = l + m;

      if (is_even(mu, 2)) {

        // CASE 1: Same as in starry
        P(n) = 2 * tworo(l + 2) * K(A, IIntegral, (mu + 4) / 4, nu / 2);

      } else if (mu == 1) {

        if (l == 1) {

          // CASE 2: Same as in starry, but using expression from Pal (2012)
          P(2) = P2(bo, ro, k2, kappa, s1, s2, c1, F, E, PIp, QUAD);

        } else if (is_even(l)) {

          // CASE 3: Same as in starry
          P(n) = (tworo(l - 1) * k3fourbr15 *
                  (L(A, JIntegral, (l - 2) / 2, 0, 0) -
                   2 * L(A, JIntegral, (l - 2) / 2, 0, 1)));

        } else {

          // CASE 4: Same as in starry
          P(n) = (tworo(l - 1) * k3fourbr15 *
                  (L(A, JIntegral, (l - 3) / 2, 1, 0) -
                   2 * L(A, JIntegral, (l - 3) / 2, 1, 1)));
        }

      } else if (is_even(mu - 1, 2)) {

        // CASE 5: Same as in starry
        P(n) = (2 * tworo(l - 1) * k3fourbr15 *
                L(A, JIntegral, (mu - 1) / 4, (nu - 1) / 2, 0));

      } else {

        /*
        A note about these cases. In the original starry code, these integrals
        are always zero because the integrand is antisymmetric about the
        midpoint. Now, however, the integration limits are different, so
        there's no cancellation in general.

        The cases below are just the first and fourth cases in equation (D25)
        of the starry paper. We can re-write them as the first and fourth cases
        in (D32) and (D35), respectively, but note that we pick up a factor
        of `sgn(cos(phi))`, since the power of the cosine term in the integrand
        is odd.

        The other thing to note is that `u` in the call to `K(u, v)` is now
        a half-integer, so our Vieta trick (D36, D37) doesn't work out of the
        box.
        */

        if (is_even(nu)) {

          // CASE 6
          int u = (mu + 2) / 4;
          int v = nu / 2;
          T res = A(u, v).dot(UIntegral.segment(u, u + v + 1));
          P(n) = 2 * tworo(l + 2) * res;

        } else {

          // CASE 7
          int u = (mu - 1) / 4;
          int v = (nu - 1) / 2;
          T res = A(u, v).dot(WIntegral.segment(u, u + v + 1));
          P(n) = 2 * tworo(l - 1) * k3fourbr15 * res;
        }
      }

      ++n;
    }
  }
}

} // namespace primitive
} // namespace reflected
} // namespace starry

#endif
