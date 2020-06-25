/**
\file solver.h
\brief Integration of the occulted portion of the map.

Solutions to the surface integral over the visible region of a spherical
harmonic map during a single-body occultation using Green's theorem.

*/

#ifndef _STARRY_SOLVER_H_
#define _STARRY_SOLVER_H_

#include "ellip.h"
#include "quad.h"
#include "utils.h"

namespace starry {
namespace solver {

using namespace starry::utils;
using namespace starry::quad;

template <class T>
inline T K_integrand(const int u, const int v, const T &delta, const T &phi) {
  T s2 = sin(phi);
  s2 *= s2;
  return pow(s2 * (1 - s2), u) * pow(delta + s2, v);
}

template <class T>
inline T K_numerical(const int u, const int v, const T &delta, const T &kappa,
                     Quad<T> &QUAD) {
  std::function<T(T)> f = [u, v, delta](T phi) {
    return K_integrand(u, v, delta, phi);
  };
  return QUAD.integrate(-0.5 * kappa, 0.5 * kappa, f);
}

template <class T>
inline T L_integrand(const int u, const int v, const int t, const T &delta,
                     const T &ksq, const T &phi) {
  T s2 = sin(phi);
  s2 *= s2;
  T term = pow(1 - s2 / ksq, 1.5);
  return pow(s2, t) * pow(s2 * (1 - s2), u) * pow(delta + s2, v) * term;
}

template <class T>
inline T L_numerical(const int u, const int v, const int t, const T &delta,
                     const T &kappa, const T &ksq, Quad<T> &QUAD) {
  std::function<T(T)> f = [u, v, t, delta, ksq](T phi) {
    return L_integrand(u, v, t, delta, ksq, phi);
  };
  return QUAD.integrate(-0.5 * kappa, 0.5 * kappa, f);
}

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
The helper primitive integral H_{u,v}.

*/
template <class T> class HIntegral {
protected:
  int umax;
  int vmax;
  Matrix<bool> set;
  Matrix<T> value;
  Vector<T> pow_coslam;
  Vector<T> pow_sinlam;
  bool coslam_is_zero;

  //! Getter function, templated so we can optimize out
  //! most of the branching at compile time.
  template <bool COSLAMZERO = false, bool UODD = false, bool VODD = false,
            bool ULT2 = false>
  inline T get_value(int u, int v) {
    CHECK_BOUNDS(u, 0, umax);
    CHECK_BOUNDS(v, 0, vmax);
    if (set(u, v)) {
      return value(u, v);
    } else if ((UODD) || (!is_even(u))) {
      return T(0.0);
    } else if ((COSLAMZERO) || (coslam_is_zero)) {
      if ((VODD) || (!is_even(v))) {
        return T(0.0);
      } else {
        if ((ULT2) || (u < 2))
          value(u, v) =
              (v - 1) * get_value<true, false, false, true>(u, v - 2) / (u + v);
        else
          value(u, v) = (u - 1) *
                        get_value<true, false, false, false>(u - 2, v) /
                        (u + v);
        set(u, v) = true;
        return value(u, v);
      }
    } else {
      if ((ULT2) || (u < 2))
        value(u, v) =
            (-2.0 * pow_coslam(u + 1) * pow_sinlam(v - 1) +
             (v - 1) * get_value<false, false, false, true>(u, v - 2)) /
            (u + v);
      else
        value(u, v) =
            (2.0 * pow_coslam(u - 1) * pow_sinlam(v + 1) +
             (u - 1) * get_value<false, false, false, false>(u - 2, v)) /
            (u + v);
      set(u, v) = true;
      return value(u, v);
    }
  }

public:
  //! Constructor
  explicit HIntegral(int lmax)
      : umax(lmax + 2), vmax(max(1, lmax)), set(umax + 1, vmax + 1),
        value(umax + 1, vmax + 1), pow_coslam(umax + 2), pow_sinlam(vmax + 2) {
    set.setZero();
    pow_coslam(0) = 1.0;
    pow_sinlam(0) = 1.0;
    coslam_is_zero = false;
  }

  //! Reset flags and compute `H_00` and `H_01`
  inline void reset(const T &coslam, const T &sinlam) {
    set.setZero();
    if (coslam == 0) {
      coslam_is_zero = true;
      value(0, 0) = 2.0 * pi<T>();
      value(0, 1) = 0.0;
    } else {
      coslam_is_zero = false;
      for (int u = 1; u < umax + 2; ++u) {
        pow_coslam(u) = pow_coslam(u - 1) * coslam;
      }
      for (int v = 1; v < vmax + 2; ++v) {
        pow_sinlam(v) = pow_sinlam(v - 1) * sinlam;
      }
      if (sinlam < 0.5)
        value(0, 0) = 2.0 * asin(sinlam) + pi<T>();
      else
        value(0, 0) = 2.0 * acos(coslam) + pi<T>();
      value(0, 1) = -2.0 * coslam;
    }
    set(0, 0) = true;
    set(0, 1) = true;
  }

  //! Overload () to get the function value without calling `get_value()`
  inline T operator()(int u, int v) { return get_value(u, v); }
};

template <typename T>
inline void computeKVariables(const T &b, const T &r, T &ksq, T &k, T &kc,
                              T &kcsq, T &kkc, T &invksq, T &kite_area2,
                              T &kap0, T &kap1, T &invb, T &invr, T &coslam,
                              T &sinlam, bool &qcond) {
  // Initialize some useful quantities
  invr = T(1.0) / r;
  invb = T(1.0) / b;
  T bmr = b - r;
  T bpr = b + r;
  T invfourbr = 0.25 * invr * invb;
  T onembpr2 = (T(1.0) + bpr) * (T(1.0) - bpr);

  // Compute cos(lambda) and sin(lambda)
  qcond = ((abs(T(1.0) - r) >= b) || (bmr >= T(1.0)));
  if (qcond) {
    sinlam = 1.0;
    coslam = 0.0;
  } else {
    sinlam = 0.5 * (invb + bmr * (T(1.0) + r * invb));
    if (sinlam > 0.5) {
      T del = T(1.0) - (b + r);
      T eps = del * invb * (r + (0.5 * del));
      sinlam = T(1.0) + eps;
      coslam = sqrt(-eps * (T(2.0) + eps));
    } else {
      coslam = sqrt(T(1.0) - sinlam * sinlam);
    }
  }

  // Compute the kite area and the k^2 variables
  if (unlikely((b == 0) || (r == 0))) {
    ksq = T(INFINITY);
    k = T(INFINITY);
    kc = 1;
    kcsq = 1;
    kkc = T(INFINITY);
    invksq = 0;
    kite_area2 = 0; // Not used!
    kap0 = 0;       // Not used!
    kap1 = 0;       // Not used!
  } else {
    ksq = onembpr2 * invfourbr + T(1.0);
    invksq = T(1.0) / ksq;
    k = sqrt(ksq);
    if (ksq > 1) {
      T bmr = b - r;
      T onembmr2 = (T(1.0) + bmr) * (T(1.0) - bmr);
      T onembmr2inv = T(1.0) / onembmr2;
      kcsq = onembpr2 * onembmr2inv;
      kc = sqrt(kcsq);
      kkc = k * kc;
      kite_area2 = 0; // Not used!
      kap0 = 0;       // Not used!
      kap1 = 0;       // Not used!
    } else {
      T b2 = b * b;
      T p0 = T(1.0), p1 = b, p2 = r;
      if (p0 < p1)
        swap(p0, p1);
      if (p1 < p2)
        swap(p1, p2);
      if (p0 < p1)
        swap(p0, p1);
      T sqarea = (p0 + (p1 + p2)) * (p2 - (p0 - p1)) * (p2 + (p0 - p1)) *
                 (p0 + (p1 - p2));
      kite_area2 = sqrt(max(T(0.0), sqarea));
      kcsq = -onembpr2 * invfourbr;
      kc = sqrt(kcsq);
      kkc = kite_area2 * invfourbr;
      kap0 = atan2(kite_area2, (r - T(1.0)) * (r + T(1.0)) + b2);
      kap1 = atan2(kite_area2, (T(1.0) - r) * (T(1.0) + r) + b2);
    }
  }
}

template <class Scalar, bool GRADIENT = false>
inline void computeS0_(const Scalar &b, const Scalar &r, const Scalar &ksq,
                       const Scalar &kite_area2, const Scalar &kap0,
                       const Scalar &kap1, const Scalar &invb, Scalar &s0,
                       Scalar &ds0db, Scalar &ds0dr) {
  if (unlikely((b == 0) || (r == 0))) {
    s0 = pi<Scalar>() * (1 - r * r);
    if (GRADIENT) {
      ds0db = 0;
      ds0dr = -2 * pi<Scalar>() * r;
    }
  } else {
    if (ksq > 1) {
      s0 = pi<Scalar>() * (1 - r * r);
      if (GRADIENT) {
        ds0db = 0;
        ds0dr = -2 * pi<Scalar>() * r;
      }
    } else {
      Scalar Alens = kap1 + r * r * kap0 - kite_area2 * 0.5;
      s0 = pi<Scalar>() - Alens;
      if (GRADIENT) {
        ds0db = kite_area2 * invb;
        ds0dr = -2.0 * r * kap0;
      }
    }
  }
}

template <class Scalar, bool GRADIENT = false>
inline void computeS2_(const Scalar &b, const Scalar &r, const Scalar &ksq,
                       const Scalar &kc, const Scalar &kcsq,
                       const Scalar &invksq, const Scalar &third, Scalar &s2,
                       Scalar &EllipticE, Scalar &EllipticEK, Scalar &ds2db,
                       Scalar &ds2dr, Scalar &dEllipticEdm,
                       Scalar &dEllipticEKdm) {
  // Initialize some useful quantities
  Scalar r2 = r * r;
  Scalar bmr = b - r;
  Scalar bpr = b + r;
  Scalar onembmr2 = (Scalar(1.0) + bmr) * (Scalar(1.0) - bmr);
  Scalar onembmr2inv = Scalar(1.0) / onembmr2;

  // Compute s2 and its derivatives
  Scalar Lambda1 = 0;
  if ((b >= 1.0 + r) || (r == 0.0)) {
    // No occultation (Case 1)
    Lambda1 = 0;
    EllipticE = 0;
    EllipticEK = 0;
    if (GRADIENT) {
      ds2db = 0;
      ds2dr = 0;
      dEllipticEdm = 0;
      dEllipticEKdm = 0;
    }
  } else if (b <= r - 1.0) {
    // Full occultation (Case 11)
    Lambda1 = 0;
    EllipticE = 0;
    EllipticEK = 0;
    if (GRADIENT) {
      ds2db = 0;
      ds2dr = 0;
      dEllipticEdm = 0;
      dEllipticEKdm = 0;
    }
  } else {
    if (unlikely(b == 0)) {
      // Case 10
      Scalar sqrt1mr2 = sqrt(1.0 - r2);
      Lambda1 = -2.0 * pi<Scalar>() * sqrt1mr2 * sqrt1mr2 * sqrt1mr2;
      EllipticE = 0.5 * pi<Scalar>();
      EllipticEK = 0.25 * pi<Scalar>();
      if (GRADIENT) {
        ds2db = 0;
        ds2dr = -2.0 * pi<Scalar>() * r * sqrt1mr2;
        dEllipticEdm = 0;
        dEllipticEKdm = 0;
      }
    } else if (unlikely(b == r)) {
      if (unlikely(r == 0.5)) {
        // Case 6
        Lambda1 = pi<Scalar>() - 4.0 * third;
        EllipticE = 1.0;
        EllipticEK = 1.0;
        if (GRADIENT) {
          ds2db = 2.0 * third;
          ds2dr = -2.0;
          dEllipticEdm = 0;
          dEllipticEKdm = Scalar(INFINITY);
        }
      } else if (r < 0.5) {
        // Case 5
        Scalar m = 4 * r2;
        EllipticE = ellip::CEL(m, Scalar(1.0), Scalar(1.0), Scalar(1.0 - m));
        EllipticEK = ellip::CEL(m, Scalar(1.0), Scalar(1.0), Scalar(0.0));
        Lambda1 = pi<Scalar>() +
                  2.0 * third * ((2 * m - 3) * EllipticE - m * EllipticEK);
        if (GRADIENT) {
          ds2db = -4.0 * r * third * (EllipticE - 2 * EllipticEK);
          ds2dr = -4.0 * r * EllipticE;
          dEllipticEdm = 0.5 * (EllipticEK - EllipticE) / (1 - m);
          dEllipticEKdm = 0.5 * ((m - Scalar(2.0)) * EllipticEK + EllipticE) /
                          (m * (1.0 - Scalar(m)));
        }
      } else {
        // Case 7
        Scalar m = 4 * r2;
        Scalar minv = Scalar(1.0) / m;
        EllipticE =
            ellip::CEL(minv, Scalar(1.0), Scalar(1.0), Scalar(1.0 - minv));
        EllipticEK = ellip::CEL(minv, Scalar(1.0), Scalar(1.0), Scalar(0.0));
        Lambda1 = pi<Scalar>() +
                  third / r * (-m * EllipticE + (2 * m - 3) * EllipticEK);
        if (GRADIENT) {
          ds2db = 2 * third * (2 * EllipticE - EllipticEK);
          ds2dr = -2 * EllipticEK;
          dEllipticEdm = 0.5 * (EllipticEK - EllipticE) / (1 - minv);
          dEllipticEKdm = 0.5 *
                          ((minv - Scalar(2.0)) * EllipticEK + EllipticE) /
                          (minv * (1.0 - Scalar(minv)));
        }
      }
    } else {
      if (ksq < 1) {
        // Case 2, Case 8
        Scalar fourbr = 4 * b * r;
        Scalar sqbrinv = Scalar(1.0) / sqrt(b * r);
        Scalar Piofk;
        ellip::CEL(ksq, kc, Scalar((b - r) * (b - r) * kcsq), Scalar(0.0),
                   Scalar(1.0), Scalar(1.0),
                   Scalar(3 * kcsq * (b - r) * (b + r)), kcsq, Scalar(0.0),
                   Piofk, EllipticE, EllipticEK);
        Lambda1 = onembmr2 * (Piofk + (-3 + 6 * r2 + 2 * b * r) * EllipticEK -
                              fourbr * EllipticE) *
                  sqbrinv * third;
        if (GRADIENT) {
          ds2db = 2 * r * onembmr2 * (-EllipticEK + 2 * EllipticE) * sqbrinv *
                  third;
          ds2dr = -2 * r * onembmr2 * EllipticEK * sqbrinv;
          dEllipticEdm = 0.5 * (EllipticEK - EllipticE) / (1 - ksq);
          dEllipticEKdm = 0.5 * ((ksq - Scalar(2.0)) * EllipticEK + EllipticE) /
                          (ksq * (1.0 - Scalar(ksq)));
        }
      } else if (ksq > 1) {
        // Case 3, Case 9
        Scalar onembpr2 = (Scalar(1.0) + bpr) * (Scalar(1.0) - bpr);
        Scalar sqonembmr2 = sqrt(onembmr2);
        Scalar b2 = b * b;
        Scalar bmrdbpr = (b - r) / (b + r);
        Scalar mu = 3 * bmrdbpr * onembmr2inv;
        Scalar p = bmrdbpr * bmrdbpr * onembpr2 * onembmr2inv;
        Scalar Piofk;
        ellip::CEL(invksq, kc, p, Scalar(1 + mu), Scalar(1.0), Scalar(1.0),
                   Scalar(p + mu), kcsq, Scalar(0.0), Piofk, EllipticE,
                   EllipticEK);
        Lambda1 = 2 * sqonembmr2 *
                  (onembpr2 * Piofk - (4 - 7 * r2 - b2) * EllipticE) * third;
        if (GRADIENT) {
          ds2db = -4 * r * third * sqonembmr2 * (EllipticE - 2 * EllipticEK);
          ds2dr = -4 * r * sqonembmr2 * EllipticE;
          dEllipticEdm = 0.5 * (EllipticEK - EllipticE) / (1 - invksq);
          dEllipticEKdm = 0.5 *
                          ((invksq - Scalar(2.0)) * EllipticEK + EllipticE) /
                          (invksq * (1.0 - Scalar(invksq)));
        }
      } else {
        // Case 4
        Scalar rootr1mr = sqrt(r * (1 - r));
        Lambda1 = 2 * acos(1.0 - 2.0 * r) -
                  4 * third * (3 + 2 * r - 8 * r2) * rootr1mr -
                  2 * pi<Scalar>() * int(r > 0.5);
        EllipticE = 1.0;
        EllipticEK = 1.0;
        if (GRADIENT) {
          ds2dr = -8 * r * rootr1mr;
          ds2db = -ds2dr * third;
          dEllipticEdm = 0;
          dEllipticEKdm = Scalar(INFINITY);
        }
      }
    }
  }
  s2 = ((1.0 - int(r > b)) * 2 * pi<Scalar>() - Lambda1) * third;
}

template <class T, bool AUTODIFF> class Solver {
public:
  // Indices
  int lmax;
  int N;
  int ivmax;
  int jvmax;

  // Variables
  T b;
  T r;
  T delta;
  T k;
  T ksq;
  T kc;
  T kcsq;
  T kkc;
  T invksq;
  T kite_area2;
  T kap0;
  T kap1;
  T invb;
  T invr;
  T coslam;
  T sinlam;
  T EllipticE;
  T EllipticEK;

  // Miscellaneous
  T third;
  T dummy;
  bool qcond;
  Vector<T> pow_ksq;
  Vector<T> cjlow;
  Vector<T> cjhigh;
  std::vector<int> jvseries;

  // Integrals
  Vieta<T> A;
  HIntegral<T> H;
  Vector<T> I;
  Vector<T> IGamma;
  Vector<T> J;

  // Numerical integration
  Quad<T> QUAD;

  // The solution vector
  RowVector<T> sT;

  explicit Solver(int lmax)
      : lmax(lmax), N((lmax + 1) * (lmax + 1)), ivmax(lmax + 2),
        jvmax(lmax > 0 ? lmax - 1 : 0), pow_ksq(ivmax + 1),
        cjlow(Vector<T>::Zero(jvmax + 2)), cjhigh(Vector<T>::Zero(jvmax + 2)),
        A(lmax), H(lmax), I(ivmax + 1), IGamma(ivmax + 1), J(jvmax + 1),
        sT(RowVector<T>::Zero(N)) {
    third = T(1.0) / T(3.0);
    dummy = 0.0;
    pow_ksq(0) = 1.0;
    precomputeIGamma();
    precomputeJCoeffs();
  }

#ifdef STARRY_ENABLE_BOOST

  /**
  The helper primitive integral I_{v} when k^2 >= 1.
  This is pre-computed when the class is instantiated.
  AutoDiff specialization.

  */
  template <typename U = T, bool A = AUTODIFF>
  inline typename std::enable_if<A, void>::type precomputeIGamma() {
    for (int v = 0; v <= ivmax; v++) {
      IGamma(v) =
          root_pi<T>() *
          boost::math::tgamma_delta_ratio<typename U::Scalar>(v + 0.5, 0.5);
    }
  }

  /**
  The helper primitive integral I_{v} when k^2 >= 1.
  This is pre-computed when the class is instantiated.
  Scalar specialization.

  */
  template <typename U = T, bool A = AUTODIFF>
  inline typename std::enable_if<!A, void>::type precomputeIGamma() {
    for (int v = 0; v <= ivmax; v++) {
      IGamma(v) =
          root_pi<T>() * boost::math::tgamma_delta_ratio<U>(v + 0.5, 0.5);
    }
  }

#else

  /**
  The helper primitive integral I_{v} when k^2 >= 1.
  This is pre-computed when the class is instantiated.

  */
  inline void precomputeIGamma() {
    T term;
    for (int v = 0; v <= ivmax; v++) {
      term = pi<T>();
      for (int i = 1; i < v; ++i)
        term *= (i - T(0.5)) / (i + T(1.0));
      for (int i = max(1, v); i < v + 1; ++i)
        term *= i - T(0.5);
      IGamma(v) = term;
    }
  }

#endif

  /**
  Pre-compute some useful coefficients in the series
  expansion of the high-degree J terms.

  */
  inline void precomputeJCoeffs() {
    // This vector contains the indices of `J` for which
    // we explicitly compute the integral via the series
    // expression when doing downward recursion. The top
    // index is the only one we *have* to compute, but I've
    // found that the recursion loses precision every ~25 degrees
    // or so. Therefore we force the series evaluation every
    // `STARRY_REFINE_J_AT` indices by including the indices in
    // this vector.
    for (int v = jvmax > 1 ? jvmax : 1; v >= 1; v -= STARRY_REFINE_J_AT) {
      jvseries.push_back(v);
    }

    // Pre-compute the factors we'll need for the series evaluation
    for (int vtop : jvseries) {
      for (int v = vtop; v > vtop - 2; --v) {
        T term0 = 3 * pi<T>();
        T term1 = T(8.0);
        for (int i = 1; i <= v; ++i) {
          term1 *= 2.0 * (i + 2);
          term0 *= (2.0 * i - 1);
        }
        cjlow(v) = term0 / term1;
        term0 = pi<T>();
        for (int i = 1; i <= v; ++i)
          term0 *= (T(1.0) - T(0.5) / i);
        cjhigh(v) = term0;
      }
    }
  }

  /**
  The helper primitive integral I_{v}, computed
  by downward recursion.

  */
  inline void computeIDownward() {
    // Track the error
    T tol = mach_eps<T>() * ksq;
    T error = T(INFINITY);

    // Computing leading coefficient
    T coeff = T(2.0) / T(2.0 * ivmax + 1.0);
    T res = coeff;

    // Compute higher order terms
    int n = 1;
    while ((n < STARRY_IJ_MAX_ITER) && (abs(error) > tol)) {
      coeff *= (2.0 * n - 1.0) * 0.5 * T(2 * n + 2 * ivmax - 1) /
               T(n * (2.0 * n + 2.0 * ivmax + 1)) * ksq;
      error = coeff;
      res += coeff;
      ++n;
    }
    if (unlikely(n == STARRY_IJ_MAX_ITER))
      throw std::runtime_error("Primitive integral `I` did not converge.");

    // This is I_{ivmax}
    I(ivmax) = pow_ksq(ivmax) * k * res;

    // Now compute the remaining terms
    for (int v = ivmax - 1; v >= 0; --v) {
      I(v) =
          T(2.0) / T(2.0 * v + 1.0) * ((v + 1.0) * I(v + 1) + pow_ksq(v) * kkc);
    }
  }

  /**
  The helper primitive integral I_{v}, computed
  by upward recursion.

  */
  inline void computeIUpward() {
    I(0) = kap0;
    for (int v = 1; v < ivmax + 1; ++v) {
      I(v) = (0.5 * (2.0 * v - 1.0) * I(v - 1) - pow_ksq(v - 1) * kkc) / v;
    }
  }

  /**
  The helper primitive integral J_{v}, computed
  by downward recursion.

  */
  template <bool KSQLESSTHANONE> inline void computeJDownward() {
    // Track the error
    T tol;
    if (KSQLESSTHANONE)
      tol = mach_eps<T>() * ksq;
    else
      tol = mach_eps<T>() * invksq;
    T coeff, res, error;
    T f1, f2, f3;
    int vtop, vbot;

    // Compute our initial terms via the series expansion
    for (size_t i = 0; i < jvseries.size(); ++i) {
      vtop = jvseries[i];
      // Top two terms
      for (int v = vtop; v > vtop - 2; --v) {
        error = T(INFINITY);
        if (KSQLESSTHANONE)
          coeff = cjlow(v);
        else
          coeff = cjhigh(v);
        res = coeff;
        int n = 1;
        while ((n < STARRY_IJ_MAX_ITER) && (abs(error) > tol)) {
          if (KSQLESSTHANONE)
            coeff *= (2.0 * n - 1.0) * (2.0 * (n + v) - 1.0) * 0.25 /
                     T(n * (n + v + 2.0)) * ksq;
          else
            coeff *=
                (T(1.0) - T(2.5 / n)) * (T(1.0) - T(0.5 / (n + v))) * invksq;
          error = coeff;
          res += coeff;
          ++n;
        }
        if (unlikely(n == STARRY_IJ_MAX_ITER))
          throw std::runtime_error("Primitive integral `J` did not converge.");
        if (KSQLESSTHANONE)
          J(v) = pow_ksq(v) * k * res;
        else
          J(v) = res;
      }
      // Recurse downward
      if (i < jvseries.size() - 1)
        vbot = jvseries[i + 1];
      else
        vbot = -1;
      for (int v = vtop - 2; v > vbot; --v) {
        if (KSQLESSTHANONE) {
          f2 = T(1.0) / (ksq * (2 * v + 1));
          f1 = 2 * (T(3 + v) + ksq * (1 + v)) * f2;
          f3 = T(2 * v + 7) * f2;
          J(v) = f1 * J(v + 1) - f3 * J(v + 2);
        } else {
          f3 = T(1.0) / T(2 * v + 1);
          f2 = T(2 * v + 7) * f3 * invksq;
          f1 = 2.0 * f3 * ((3 + v) * invksq + T(1 + v));
          J(v) = f1 * J(v + 1) - f2 * J(v + 2);
        }
      }
    }
  }

  /**
  The helper primitive integral J_{v}, computed
  by upward recursion.

  */
  template <bool KSQLESSTHANONE> inline void computeJUpward() {
    T f1, f2;
    if (KSQLESSTHANONE) {
      T fac = 2.0 * third / k;
      J(0) = fac * (EllipticE + (3.0 * ksq - T(2.0)) * EllipticEK);
      J(1) = 0.2 * fac * ((T(4.0) - 3.0 * ksq) * EllipticE +
                          (9.0 * ksq - T(8.0)) * EllipticEK);
    } else {
      J(0) = 2.0 * third *
             ((T(3.0) - 2.0 * invksq) * EllipticE + invksq * EllipticEK);
      J(1) = 0.4 * third * ((T(9.0) - 8.0 * invksq) * EllipticE +
                            (4.0 * invksq - T(3.0)) * EllipticEK);
    }
    for (int v = 2; v < jvmax + 1; ++v) {
      f1 = 2.0 * (T(v + 1) + (v - 1) * ksq);
      f2 = ksq * (2 * v - 3);
      J(v) = (f1 * J(v - 1) - f2 * J(v - 2)) / T(2 * v + 3);
    }
  }

  /**
  The helper primitive integral K_{u,v}.

  */
  inline T K(int u, int v) {

#if defined(STARRY_DEBUG) || defined(STARRY_KL_NUMERICAL)
    // HACK: Fix numerical instabilities at high l
    if (lmax > 15) {
      if (ksq >= 1)
        return K_numerical(u, v, delta, pi<T>(), QUAD);
      else
        return K_numerical(u, v, delta, kap0, QUAD);
    }
#endif

    if (ksq >= 1)
      return A(u, v).dot(IGamma.segment(u, u + v + 1));
    else
      return A(u, v).dot(I.segment(u, u + v + 1));
  }

  /**
  The helper primitive integral L_{u,v}^(t).

  */
  inline T L(int u, int v, int t) {

#if defined(STARRY_DEBUG) || defined(STARRY_KL_NUMERICAL)
    // HACK: Fix numerical instabilities at high l
    if (lmax > 15) {
      if (ksq >= 1)
        return L_numerical(u, v, t, delta, pi<T>(), ksq, QUAD);
      else
        return L_numerical(u, v, t, delta, kap0, ksq, QUAD);
    }
#endif

    return A(u, v).dot(J.segment(u + t, u + v + 1));
  }

  /**
  Compute s(0) for a Scalar type.

  */
  template <bool A = AUTODIFF>
  inline typename std::enable_if<!A, void>::type computeS0() {
    computeS0_<T, false>(b, r, ksq, kite_area2, kap0, kap1, invb, sT(0), dummy,
                         dummy);
  }

  /**
  Compute s(0) and its gradient for an AutoDiffScalar type.
  We know how to compute the gradient analytically, so we need
  to override AutoDiff.

  */
  template <bool A = AUTODIFF>
  inline typename std::enable_if<A, void>::type computeS0() {
    typename T::Scalar ds0db, ds0dr;
    computeS0_<typename T::Scalar, true>(
        b.value(), r.value(), ksq.value(), kite_area2.value(), kap0.value(),
        kap1.value(), invb.value(), sT(0).value(), ds0db, ds0dr);
    sT(0).derivatives() = ds0db * b.derivatives() + ds0dr * r.derivatives();
  }

  /**
  Compute s(2) for a Scalar type.

  */
  template <bool A = AUTODIFF>
  inline typename std::enable_if<!A, void>::type computeS2() {
    computeS2_<T, false>(b, r, ksq, kc, kcsq, invksq, third, sT(2), EllipticE,
                         EllipticEK, dummy, dummy, dummy, dummy);
  }

  /**
  Compute s(2) and its gradient for an AutoDiffScalar type.
  We know how to compute the gradient analytically, so we need
  to override AutoDiff.

  */
  template <bool A = AUTODIFF>
  inline typename std::enable_if<A, void>::type computeS2() {
    typename T::Scalar ds2db, ds2dr;
    typename T::Scalar dEdksq, dEKdksq;
    computeS2_<typename T::Scalar, true>(
        b.value(), r.value(), ksq.value(), kc.value(), kcsq.value(),
        invksq.value(), third.value(), sT(2).value(), EllipticE.value(),
        EllipticEK.value(), ds2db, ds2dr, dEdksq, dEKdksq);
    sT(2).derivatives() = ds2db * b.derivatives() + ds2dr * r.derivatives();
    if (ksq < 1) {
      EllipticE.derivatives() = dEdksq * ksq.derivatives();
      EllipticEK.derivatives() = dEKdksq * ksq.derivatives();
    } else {
      EllipticE.derivatives() = dEdksq * invksq.derivatives();
      EllipticEK.derivatives() = dEKdksq * invksq.derivatives();
    }
  }

  /**
  Compute the `s^T` occultation solution vector.

  */
  inline void compute(const T &b_, const T &r_) {
    // Initialize b and r
    b = b_;
    r = r_;

    // HACK: Fix an instability that exists *really* close to b = r = 0.5
    if (unlikely(abs(b - r) < 5 * mach_eps<T>())) {
      if (unlikely(abs(r - T(0.5)) < 5 * mach_eps<T>())) {
        b += 5 * mach_eps<T>();
      }
    }

    // Special case: complete occultation
    if (unlikely(b < r - 1)) {
      sT.setZero();
      return;
    }

    // Special case: no occultation
    if (unlikely(r == 0) || (b > r + 1)) {
      throw std::runtime_error(
          "No occultation, but occultation routine was called.");
    }

    // Special case: negative radius
    if (unlikely(r < 0)) {
      throw std::runtime_error("Occultor radius is negative. Aborting.");
    }

    // Compute the k^2 terms and angular variables
    computeKVariables(b, r, ksq, k, kc, kcsq, kkc, invksq, kite_area2, kap0,
                      kap1, invb, invr, coslam, sinlam, qcond);

    // Some useful quantities
    T twor = 2 * r;
    T bmr = b - r;
    T tworlp2 = twor * twor * twor;
    delta = 0.5 * bmr * invr;

    // Compute the constant term
    computeS0();

    // Break if lmax = 0
    if (unlikely(N == 1))
      return;

    // The l = 1, m = -1 is zero by symmetry
    sT(1) = 0;

    // Compute the linear limb darkening term
    // and the elliptic integrals
    computeS2();

    // The l = 1, m = 1 term, written out explicitly for speed
    T K11;
    if (ksq >= 1) {
      K11 = pi<T>() * (2 * delta + T(1.0)) / 16.;
    } else {
      T fac = T(3.0) + 6 * delta;
      K11 =
          0.0625 * third *
          (2.0 * kkc * (2.0 * ksq * (6.0 * delta + 4.0 * ksq - T(1.0)) - fac) +
           kap0 * fac);
    }
    sT(3) = -2.0 * third * coslam * coslam * coslam - 2 * tworlp2 * K11;

    // Break if lmax = 1
    if (N == 4)
      return;

    // Compute powers of ksq
    for (int v = 1; v < ivmax + 1; ++v)
      pow_ksq(v) = pow_ksq(v - 1) * ksq;

    // Compute the helper integrals
    A.reset(delta);
    H.reset(coslam, sinlam);
    if (ksq < 0.5)
      computeIDownward();
    else if (ksq < 1.0)
      computeIUpward();
    // else we use `IGamma`

    if (ksq < 1.0) {
      if (unlikely(ksq == 0))
        J = IGamma;
      else if (ksq < 0.5)
        computeJDownward<true>();
      else
        computeJUpward<true>();
    } else {
      if (ksq > 2.0)
        computeJDownward<false>();
      else
        computeJUpward<false>();
    }

    // Some more basic variables
    T Q, P;
    T lfac = pow(1 - bmr * bmr, 1.5);

    // Compute the other terms of the solution vector
    int n = 4;
    for (int l = 2; l < lmax + 1; ++l) {
      // Update the pre-factors
      tworlp2 *= twor;
      lfac *= twor;

      for (int m = -l; m < l + 1; ++m) {
        int mu = l - m;
        int nu = l + m;

        // These terms are zero because they are proportional to
        // odd powers of x, so we don't need to compute them!
        if ((is_even(mu - 1)) && (!is_even((mu - 1) / 2))) {
          sT(n) = 0;

          // These terms are also zero for the same reason
        } else if ((is_even(mu)) && (!is_even(mu / 2))) {
          sT(n) = 0;

          // We need to compute the integral...
        } else {
          // The Q integral
          if ((qcond) && (!is_even(mu, 2) || !is_even(nu, 2)))
            Q = 0.0;
          else if (!is_even(mu, 2))
            Q = 0.0;
          else
            Q = H((mu + 4) / 2, nu / 2);

          // The P integral
          if (is_even(mu, 2))
            P = 2 * tworlp2 * K((mu + 4) / 4, nu / 2);
          else if ((mu == 1) && is_even(l))
            P = lfac * (L((l - 2) / 2, 0, 0) - 2 * L((l - 2) / 2, 0, 1));
          else if ((mu == 1) && !is_even(l))
            P = lfac * (L((l - 3) / 2, 1, 0) - 2 * L((l - 3) / 2, 1, 1));
          else if (is_even(mu - 1, 2))
            P = 2 * lfac * L((mu - 1) / 4, (nu - 1) / 2, 0);
          else
            P = 0.0;

          // The term of the solution vector
          sT(n) = Q - P;
        }

        ++n;
      }
    }
  }
};

/**
Greens integral solver wrapper class.
Emitted light specialization.

*/
template <class Scalar> class Greens {
protected:
  using ADType = ADScalar<Scalar, 2>;

  // Indices
  int lmax;
  int N;

  // Solvers
  Solver<Scalar, false> ScalarSolver;
  Solver<ADType, true> ADTypeSolver;

  // AutoDiff
  ADType b_ad;
  ADType r_ad;

public:
  // Solutions
  RowVector<Scalar> &sT;
  RowVector<Scalar> dsTdb;
  RowVector<Scalar> dsTdr;

  // Constructor
  explicit Greens(int lmax)
      : lmax(lmax), N((lmax + 1) * (lmax + 1)), ScalarSolver(lmax),
        ADTypeSolver(lmax), b_ad(ADType(0.0, Vector<Scalar>::Unit(2, 0))),
        r_ad(ADType(0.0, Vector<Scalar>::Unit(2, 1))), sT(ScalarSolver.sT),
        dsTdb(RowVector<Scalar>::Zero(N)), dsTdr(RowVector<Scalar>::Zero(N)) {}

  /**
  Compute the `s^T` occultation solution vector
  with or without the gradient.

  */
  template <bool GRADIENT = false>
  inline void compute(const Scalar &b, const Scalar &r) {
    if (!GRADIENT) {
      ScalarSolver.compute(b, r);

    } else {
      b_ad.value() = b;
      r_ad.value() = r;
      ADTypeSolver.compute(b_ad, r_ad);
      for (int n = 0; n < N; ++n) {
        sT(n) = ADTypeSolver.sT(n).value();
        dsTdb(n) = ADTypeSolver.sT(n).derivatives()(0);
        dsTdr(n) = ADTypeSolver.sT(n).derivatives()(1);
      }
    }
  }
};

} // namespace solver
} // namespace starry

#endif
