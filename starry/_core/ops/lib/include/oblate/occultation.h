
/**
\file occultation.h
\brief Solver for occultations of oblate bodies.

*/

#ifndef _STARRY_OBLATE_OCCULTATION_H_
#define _STARRY_OBLATE_OCCULTATION_H_

#include "../quad.h"
#include "../utils.h"
#include "ellip.h"
#include "geometry.h"
#include "special.h"

namespace starry {
namespace oblate {
namespace occultation {

using namespace utils;
using namespace ellip;
using namespace special;
using namespace geometry;
using std::min, std::max;

template <class Scalar, int N> class Occultation {

  using A = ADScalar<Scalar, N>;

protected:
  // Misc
  int deg;
  int ncoeff;

  // Inputs
  A bo;
  A ro;
  A f;
  A theta;

  // Transformed inputs
  A k2;
  A kc2;
  A invkc2;
  A costheta;
  A sintheta;
  A tantheta;
  A invtantheta;
  A gamma;
  A sqrtgamma;
  A w2;

  // Integrals
  Vector<A> W;
  Vector<A> V;
  Matrix<A> J;
  Matrix<A> J32;
  Matrix<A> J12;
  Matrix<A> Lp;
  Matrix<A> Lt;
  Matrix<A> M;
  Matrix<A> I;
  Matrix<A> K;
  Matrix<A> H;

  // Helper matrices
  Matrix<A> BL;
  Matrix<A> BR;
  Matrix<A> S;
  Matrix<A> C;
  Eigen::Matrix<A, 2, 3, ColMajor> D;

  // Numerical integration
  quad::Quad<Scalar> QUAD;

  /**
   *
   * Solve a recursion relation of the form
   *
   *    f(n) = A(n) * f(n - 1) + B(n) * f (n - 2) + C(n)
   *
   * given a lower boundary condition f(0) and
   * an upper boundary condition f(N).
   *
   */
  inline void solve(const A &f0, const A &fN, std::function<A(int)> funcA,
                    std::function<A(int)> funcB, std::function<A(int)> funcC,
                    const int &nmax, Vector<A> &result) {

    // Set up the tridiagonal problem
    Vector<A> a(nmax - 1), b(nmax - 1), c(nmax - 1);
    for (int n = 2; n < nmax + 1; ++n) {
      a(n - 2) = -funcA(n);
      b(n - 2) = -funcB(n);
      c(n - 2) = funcC(n);
    }

    // Add the boundary conditions
    c(0) -= b(0) * f0;
    c(nmax - 2) -= fN;

    // Construct the tridiagonal matrix
    // TODO: We should probably use a sparse solve here!
    Matrix<A> M(nmax - 1, nmax - 1);
    M.setZero();
    M.diagonal(0) = a;
    M.diagonal(-1) = b.segment(1, nmax - 2);
    M.diagonal(1).setOnes();

    // Solve
    Vector<A> soln = M.lu().solve(c);

    // Append lower and upper boundary conditions
    result.resize(nmax + 1);
    result(0) = f0;
    result(nmax) = fN;
    result.segment(1, nmax - 1) = soln;
  }

  /**
   *
   * Compute the vector of `W` integrals, computed
   * recursively given a lower boundary condition
   * (analytic in terms of elliptic integrals) and an upper
   * boundary condition (computed numerically).
   *
   * The term `W_j` is the solution to the integral of
   *
   *    sin(u)^(2 * j) * sqrt(1 - sin(u)^2 / (1 - k^2))
   *
   * from u = u1 to u = u2, where
   *
   *    u = (pi - 2 * phi) / 4
   *
   */
  inline void compute_W() {

    int nmax = 2 * deg + 4;
    Scalar invkc2_value = invkc2.value();

    // Transformed angles
    Pair<A> u;
    u(0) = 0.25 * (pi<A>() - 2 * phi1);
    u(1) = 0.25 * (pi<A>() - 2 * phi2);
    Pair<A> sinu;
    sinu.array() = sin(u.array());
    Pair<A> cosu;
    cosu.array() = cos(u.array());

    // Delta^3 parameter from Gradshteyn & Ryzhik
    Pair<A> D1, D2, D3;
    D2.array() = abs(1 - sinu.array() * sinu.array() * invkc2);
    D1.array() = pow(D2.array(), 0.5);
    D3.array() = D1.array() * D2.array();

    // Lower boundary
    A f0;
    if (k2 >= 1) {

      auto Elliptic = IncompleteEllipticIntegrals<Scalar, N>(invkc2, u);
      f0 = Elliptic.E;

    } else {

      // Complex trickery: D is actually imaginary, so D^3 is negative
      // in this branch. The extra factor of `i` cancels with the
      // imaginary part of the elliptic integral below to yield a real
      // result.
      D3 *= -1;

      // We need to compute the *imaginary* part of the elliptic integrals
      // in this branch. Ideally we could just do
      //
      //  auto Elliptic = IncompleteEllipticIntegrals<Scalar, N>(invkc2, u);
      //  f0 = imag(Elliptic.E);
      //
      // But our `el2` algorithm only computes the real part. So we need
      // to reparametrize.
      Pair<A> term, v, C, sinv;
      int n0, n1, sgn0, sgn1;
      n0 = std::floor((u(0).value() + 0.5 * pi<Scalar>()) / pi<Scalar>());
      n1 = std::floor((u(1).value() + 0.5 * pi<Scalar>()) / pi<Scalar>());
      sgn0 = is_even(n0) ? 1 : -1;
      sgn1 = is_even(n1) ? 1 : -1;
      term = D1 * sqrt(1 / k2 - 1);
      v(0) = n0 * pi<Scalar>() + sgn0 * asin(term(0) / sinu(0));
      v(1) = n1 * pi<Scalar>() + sgn1 * asin(term(1) / sinu(1));
      auto Elliptic = IncompleteEllipticIntegrals<Scalar, N>(k2, v);
      sinv.array() = sin(v.array());
      C.array() = sinv.array() * cos(v.array()) /
                  sqrt(1 - k2 * sinv.array() * sinv.array());
      f0 = -sqrt(kc2) * (Elliptic.F - (Elliptic.E - k2 * (C(1) - C(0))) / kc2);
    }

    // Upper boundary
    // TODO: Find a series solution so we don't have to integrate
    std::function<Scalar(Scalar)> funcN = [nmax, invkc2_value](Scalar x) {
      Scalar sinx2 = sin(x);
      sinx2 *= sinx2;
      return pow(sinx2, nmax) * sqrt(abs(1 - sinx2 * invkc2_value));
    };
    A fN;
    fN.value() = QUAD.integrate(u(0).value(), u(1).value(), funcN);
    if (N > 0) {
      std::function<Scalar(Scalar)> dfuncNdinvkc2 = [nmax,
                                                     invkc2_value](Scalar x) {
        Scalar sinx2 = sin(x);
        sinx2 *= sinx2;
        return -0.5 * pow(sinx2, nmax) * sinx2 *
               pow(abs(1 - sinx2 * invkc2_value), -0.5);
      };
      fN.derivatives() =
          (funcN(u(1).value()) * u(1).derivatives() +
           funcN(u(0).value()) * u(0).derivatives() +
           QUAD.integrate(u(0).value(), u(1).value(), dfuncNdinvkc2) *
               invkc2.derivatives());
    }

    // Recursion coefficients
    A kc2_alias = kc2;
    std::function<A(int)> funcA = [kc2_alias](int n) {
      return 2.0 * (n + (n - 1.0) * kc2_alias) / (2.0 * n + 1.0);
    };
    std::function<A(int)> funcB = [kc2_alias](int n) {
      return -(2.0 * n - 3.0) / (2.0 * n + 1.0) * kc2_alias;
    };
    std::function<A(int)> funcC = [kc2_alias, sinu, cosu, D3](int n) {
      return (pow(sinu(1), (2.0 * n - 3.0)) * cosu(1) * D3(1) -
              pow(sinu(0), (2.0 * n - 3.0)) * cosu(0) * D3(0)) *
             kc2_alias / (2.0 * n + 1.0);
    };

    // Solve the tridiagonal problem
    solve(f0, fN, funcA, funcB, funcC, nmax, W);
  }

  /**
   *
   * Compute the vector of `V` integrals, computed
   * recursively given a lower boundary condition
   * (trivial) and an upper boundary condition
   * (computed from `2F1`).
   *
   * The term `V_i` is the solution to the integral of
   *
   *    cos(phi) * sin(phi)^i * sqrt(1 - w^2 sin(phi))
   *
   * from phi = phi1 to phi = phi2, where
   *
   *    w^2 = 1 / (2 * k^2 - 1)
   *
   */
  inline void compute_V() {

    int nmax = 2 * deg + 4;
    A invw2 = 2 * k2 - 1;
    A sinphi, x, xbar;
    A f0, fN, term, H;
    std::function<A(int)> funcA, funcB, funcC;
    Vector<A> V2, V1;

    // -- Upper limit --

    // Useful quantities
    sinphi = sin(phi2);
    x = sinphi / invw2;
    xbar = abs(1 - x);
    xbar = xbar < 0 ? 0 : xbar;

    // Boundary conditions
    if (k2 > 0.5) {
      f0 = (2.0 / 3.0) * (1 - xbar * sqrt(xbar)) * invw2;
      H = hypspecial1(nmax, x);
      fN = pow(sinphi, (nmax + 1.0)) / (nmax + 1.0) * H;
    } else {
      // When k^2 < 1/2, sqrt(gamma) is imaginary,
      // so we need to compute the *imaginary* part
      // of the integral here!
      term = (2.0 / 3.0) * xbar * sqrt(xbar);
      f0 = term * invw2;
      H = hypspecial2(nmax, A(-xbar));
      fN = term * pow(sinphi, (nmax + 1.0)) * H;
    }

    // Recursion coefficients
    funcA = [x, invw2](int n) {
      return (2 * n + (2 * n + 1) * x) * invw2 / (2 * n + 3);
    };
    funcB = [sinphi, invw2](int n) {
      return -2 * (n - 1) * sinphi * invw2 / (2 * n + 3);
    };
    funcC = [](int n) { return 0.0; };

    // Solve the tridiagonal problem
    solve(f0, fN, funcA, funcB, funcC, nmax, V2);

    // -- Lower limit --

    // Useful quantities
    sinphi = sin(phi1);
    x = sinphi / invw2;
    xbar = abs(1 - x);
    xbar = xbar < 0 ? 0 : xbar;

    // Boundary conditions
    if (k2 > 0.5) {
      f0 = (2.0 / 3.0) * (1 - xbar * sqrt(xbar)) * invw2;
      H = hypspecial1(nmax, x);
      fN = pow(sinphi, (nmax + 1.0)) / (nmax + 1.0) * H;
    } else {
      // When k^2 < 1/2, sqrt(gamma) is imaginary,
      // so we need to compute the *imaginary* part
      // of the integral here!
      term = (2.0 / 3.0) * xbar * sqrt(xbar);
      f0 = term * invw2;
      H = hypspecial2(nmax, A(-xbar));
      fN = term * pow(sinphi, (nmax + 1.0)) * H;
    }

    // Recursion coefficients
    funcA = [x, invw2](int n) {
      return (2 * n + (2 * n + 1) * x) * invw2 / (2 * n + 3);
    };
    funcB = [sinphi, invw2](int n) {
      return -2 * (n - 1) * sinphi * invw2 / (2 * n + 3);
    };
    funcC = [](int n) { return 0.0; };

    // Solve the tridiagonal problem
    solve(f0, fN, funcA, funcB, funcC, nmax, V1);

    // Definite integral
    V = V2 - V1;
  }

  /**
   *
   * Compute the matrix of `J` integrals.
   *
   * The term `J_{i,j}` is the solution to the integral of
   *
   *    cos(phi)^i * sin(phi)^j * sqrt(1 - w^2 sin(phi))
   *
   * from phi = phi1 to phi = phi2, where
   *
   *    w^2 = 1 / (2 * k^2 - 1)
   *
   */
  inline void compute_J() {

    //
    int nmax = deg + 3;
    A c1 = -2.0 * sqrt(abs(1 - w2));
    A term;
    Scalar fac, amp;
    J.setZero(nmax, nmax);

    // Compute the helper integral vectors
    compute_W();
    compute_V();

    // Compute all `J`
    for (int s = 0; s < nmax / 2; ++s) {
      for (int q = 0; q < nmax; ++q) {
        fac = 1.0;
        for (int i = 0; i < s + 1; ++i) {
          term = 0.0;
          amp = 1.0;
          for (int j = 0; j < 2 * i + q + 1; ++j) {
            term += amp * W(j);
            amp *= -2 * (2 * i + q - j) / (j + 1.0);
          }
          J(2 * s, q) += c1 * fac * term;
          J(2 * s + 1, q) += fac * V(2 * i + q);
          fac *= (i - s) / (i + 1.0);
        }
      }
    }
  }

  /**
   *
   * Compute the matrix of `L` integrals.
   *
   * The term `L_{i,j}` is the solution to the integral of
   *
   *    cos(phi)^i * sin(phi)^j
   *
   * from phi = phip1 to phi = phip2, where
   *
   *    phip = phi - theta
   *
   * (pT integral) or
   *
   *    phip = xi
   *
   * (tT integral).
   *
   */
  inline void compute_L(const A &phip1, const A &phip2, Matrix<A> &L) {

    //
    int nmax = deg + 3;
    A cp1 = cos(phip1);
    A cp2 = cos(phip2);
    A sp1 = sin(phip1);
    A sp2 = sin(phip2);
    L.resize(nmax, nmax);

    // Lower boundary
    L(0, 0) = phip2 - phip1;
    L(1, 0) = sp2 - sp1;
    L(0, 1) = cp1 - cp2;
    L(1, 1) = 0.5 * (cp1 * cp1 - cp2 * cp2);

    // Recursion coeffs
    A fac, A1, B1, C1, D1;
    A A0 = cp1 * sp1;
    A B0 = cp2 * sp2;
    A C0 = cp2 * sp2;
    A D0 = cp1 * sp1;

    // Recurse
    for (int u = 0; u < nmax; ++u) {
      A1 = A0;
      B1 = B0;
      C1 = C0;
      D1 = D0;
      for (int v = 2; v < nmax; ++v) {
        fac = 1.0 / (u + v);
        L(u, v) = fac * (A1 - B1 + (v - 1) * L(u, v - 2));
        L(v, u) = fac * (C1 - D1 + (v - 1) * L(v - 2, u));
        A1 *= sp1;
        B1 *= sp2;
        C1 *= cp2;
        D1 *= cp1;
      }
      A0 *= cp1;
      B0 *= cp2;
      C0 *= sp2;
      D0 *= sp1;
    }
  }

  /**
   *
   * Compute the matrix of `L` integrals for the special case
   * phi1 = 0, phi2 = 2 * pi (no occultation).
   *
   */
  inline void compute_L0(Matrix<A> &L) {

    //
    int nmax = deg + 3;
    L.setZero(nmax, nmax);

    // Lower boundary
    L(0, 0) = 2 * pi<Scalar>();

    // Recurse
    A fac;
    for (int u = 0; u < nmax; u += 2) {
      for (int v = 2; v < nmax; v += 2) {
        fac = (v - 1.0) / (u + v);
        L(u, v) = fac * L(u, v - 2);
        L(v, u) = fac * L(v - 2, u);
      }
    }
  }

public:
  RowVector<A> sT;
  RowVector<A> sT0;
  A phi1, phi2, xi1, xi2;

  explicit Occultation(int deg) : deg(deg), ncoeff((deg + 1) * (deg + 1)) {
    compute_phase();
  }

  /**
      Compute the solution vector in the case
      of no occultation. The phase curve solution
      vector for any value of `f` is just

        sT = (1 - f) * sT0

  */
  inline void compute_phase() {
    compute_L0(Lt);
    sT0.setZero(ncoeff);
    int mu, nu, n = 0;
    for (int l = 0; l < deg + 1; ++l) {
      for (int m = -l; m < l + 1; ++m) {
        mu = l - m;
        nu = l + m;
        if (is_even(nu)) {
          // Case 1
          sT0(n) = Lt(mu / 2 + 2, nu / 2);
        } else if ((l == 1) && (m == 0)) {
          // Case 2
          sT0(n) = 2 * pi<Scalar>() / 3;
        }
        ++n;
      }
    }
  }

  /**
      Compute the full solution vector s^T.

      TODO: Unit tests for the gradients (never tested!)

  */
  inline void compute(const A &bo_, const A &ro_, const A &f_,
                      const A &theta_) {
    // Make local copies of the inputs
    bo = bo_;
    ro = ro_;
    f = f_;
    theta = theta_;
    costheta = cos(theta);
    sintheta = sin(theta);

    // Nudge the inputs away from singular points
    if (abs(bo - ro) < STARRY_BO_EQUALS_RO_TOL)
      bo = ro + (bo > ro ? STARRY_BO_EQUALS_RO_TOL : -STARRY_BO_EQUALS_RO_TOL);
    if ((abs(bo - ro) < STARRY_BO_EQUALS_RO_EQUALS_HALF_TOL) &&
        (abs(ro - 0.5) < STARRY_BO_EQUALS_RO_EQUALS_HALF_TOL))
      bo = ro + STARRY_BO_EQUALS_RO_EQUALS_HALF_TOL;
    if (abs(bo) < STARRY_BO_EQUALS_ZERO_TOL)
      bo = STARRY_BO_EQUALS_ZERO_TOL;
    if ((ro > 0) && (ro < STARRY_RO_EQUALS_ZERO_TOL))
      ro = STARRY_RO_EQUALS_ZERO_TOL;
    if (abs(1 - bo - ro) < STARRY_BO_EQUALS_ONE_MINUS_RO_TOL)
      bo = 1 - ro + STARRY_BO_EQUALS_ONE_MINUS_RO_TOL;
    if (abs(theta - 0.5 * pi<Scalar>()) < STARRY_ROOT_TOL_THETA_PI_TWO) {
      theta += (theta > 0.5 * pi<Scalar>() ? 1.0 : -1.0) *
               STARRY_ROOT_TOL_THETA_PI_TWO;
      costheta = cos(theta);
      sintheta = sin(theta);
    } else if (abs(theta + 0.5 * pi<Scalar>()) < STARRY_ROOT_TOL_THETA_PI_TWO) {
      theta += (theta > -0.5 * pi<Scalar>() ? 1.0 : -1.0) *
               STARRY_ROOT_TOL_THETA_PI_TWO;
      costheta = cos(theta);
      sintheta = sin(theta);
    } else if (abs(sintheta) < STARRY_T_TOL) {
      sintheta = sintheta > 0 ? STARRY_T_TOL : -STARRY_T_TOL;
      theta = costheta > 0 ? 0 : pi<Scalar>();
    }
    if (f < STARRY_MIN_F)
      f = STARRY_MIN_F;

    // Compute the angles of intersection
    if (ro == 0) {

      // No occultation
      sT = (1 - f) * sT0;
      return;

    } else {

      get_angles(bo, ro, f, theta, phi1, phi2, xi1, xi2);
    }

    // Special cases
    if ((phi1 == 0.0) && (phi2 == 0.0) && (xi1 == 0.0) && (xi2 == 0.0)) {

      // Complete occultation
      sT.setZero(ncoeff);
      return;

    } else if ((phi1 == 0.0) && (phi2 == 0.0) && (xi1 == 0.0) &&
               (xi2 == 2 * pi<Scalar>())) {

      // No occultation
      sT = (1 - f) * sT0;
      return;
    }

    // Useful variables
    gamma = 1 - bo * bo - ro * ro;
    sqrtgamma = sqrt(abs(gamma));
    k2 = (gamma + 2 * bo * ro) / (4 * bo * ro);
    kc2 = 1 - k2;
    w2 = 1.0 / (2 * k2 - 1);
    invkc2 = 1.0 / kc2;
    tantheta = sintheta / costheta;
    invtantheta = 1.0 / tantheta;

    // Compute the binomial helper matrices
    S.setZero(deg + 3, deg + 3);
    C.setZero(deg + 3, deg + 3);
    BL.setZero(deg + 3, deg + 3);
    BR.setZero(deg + 3, deg + 3);
    A fac, fac0, facc, facs, fac0l, fac0r, facl, facr, facb;
    fac0 = 1.0;
    fac0l = 1.0;
    fac0r = 1.0;
    for (int i = 0; i < deg + 3; ++i) {
      facs = ro * fac0;
      facc = fac0;
      facl = fac0l;
      facr = fac0r;
      for (int j = 0; j < deg + 3; ++j) {
        if (j < i + 1) {
          S(i, deg + 2 + j - i) = facs;
          C(deg + 2 + j - i, i) = facc;
          fac = bo * (i - j) / (ro * (j + 1.0));
          facs *= fac * sintheta;
          facc *= fac * costheta;
        }
        BL(i, j) = facl;
        BR(i, j) = facr;
        facb = (i - j) / (j + 1.0);
        facl *= facb * tantheta;
        facr *= -facb * invtantheta;
      }
      fac0 *= ro;
      fac0l *= costheta;
      fac0r *= -sintheta;
    }

    // Compute the first `f` derivative matrix
    A cos2theta = cos(2 * theta);
    D << -1.5 * (bo * bo + ro * ro + (bo + ro) * (bo - ro) * cos2theta),
        -6 * bo * ro * costheta * costheta, -3 * ro * ro * cos2theta,
        6 * bo * ro * costheta * sintheta, 6 * ro * ro * costheta * sintheta, 0;

    // Compute the M integral
    compute_L(phi1 - theta, phi2 - theta, Lp);
    M = S.topRightCorner(deg + 2, deg + 2) * Lp.reverse().topRows(deg + 2) *
        C.leftCols(deg + 2);

    // Compute the I integral
    compute_J();
    J32 = gamma * sqrtgamma *
          (J.topLeftCorner(deg + 2, deg + 2) -
           w2 * J.topRightCorner(deg + 2, deg + 2));
    J12 = sqrtgamma * J.topLeftCorner(deg + 2, deg + 2);
    I.setZero(deg + 2, deg + 2);
    A J0, J1, T;
    for (int i = 0; i < deg + 1; ++i) {
      int jmax = min(deg, i + 1);
      for (int j = 0; j < jmax; ++j) {
        J0 = J32(i - j, j);
        J1 = (D.cwiseProduct(J12.block(i - j, j, 2, 3))).sum();
        T = J0 + f * J1;
        for (int k = 0; k < i + 1; ++k) {
          int imk = i - k;
          int lmin = max(0, j - imk);
          int lmax = min(k, j);
          for (int l = lmin; l < lmax + 1; ++l) {
            I(deg + 1 - imk, deg + 1 - k) += BL(imk, j - l) * T * BR(k, l);
          }
        }
      }
    }

    // Compute the K, H integrals
    K = S.topRightCorner(deg + 2, deg + 1) * I.topRows(deg + 1) *
        C.bottomLeftCorner(deg + 2, deg + 2);
    H = S.topRightCorner(deg + 2, deg + 2) * I.leftCols(deg + 1) *
        C.bottomLeftCorner(deg + 1, deg + 2);

    // Compute L (t integral)
    compute_L(xi1, xi2, Lt);

    // Go through the cases
    A pT, tT;
    sT.resize(ncoeff);
    int mu, nu, n = 0;
    for (int l = 0; l < deg + 1; ++l) {
      for (int m = -l; m < l + 1; ++m) {
        mu = l - m;
        nu = l + m;

        // Compute the pT and tT integrals
        if (is_even(nu)) {
          // Case 1
          pT = pow(1 - f, -(nu / 2)) * M((mu + 2) / 2, nu / 2);
          tT = (1 - f) * Lt(mu / 2 + 2, nu / 2);
        } else if ((l == 1) && (m == 0)) {
          // Case 2
          pT = p2_numerical(bo, ro, f, theta, phi1, phi2, QUAD);
          tT = (1 - f) * (xi2 - xi1) / 3;
        } else if (is_even(l) && (mu == 1)) {
          // Case 3
          pT = -(1 - f) * H(l - 2, 0);
          tT = 0.0;
        } else if (!is_even(l) && (mu == 1)) {
          // Case 4
          pT = -H(l - 3, 1);
          tT = 0.0;
        } else {
          // Case 5
          pT = pow(1 - f, (1 - nu) / 2) * K((mu - 3) / 2, (nu - 1) / 2);
          tT = 0.0;
        }

        // The surface integral is just their sum
        sT(n) = pT + tT;
        ++n;
      }
    }
  }
};

} // namespace occultation
} // namespace oblate
} // namespace starry

#endif