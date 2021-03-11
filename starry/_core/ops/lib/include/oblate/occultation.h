
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
using namespace ellip;

template <class Scalar, int N> class Occultation {

  using A = ADScalar<Scalar, N>;

protected:
  // Misc
  int deg;

  // Inputs
  A bo;
  A ro;
  A f;
  A theta;
  A phi1;
  A phi2;
  A xi1;
  A xi2;

  // Transformed inputs
  A k2;
  A kc2;
  A invkc2;

  // Integrals
  Vector<A> W;

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

    // Transformed angles
    Pair<A> u;
    u(0) = 0.25 * (pi<A>() - 2 * phi1);
    u(1) = 0.25 * (pi<A>() - 2 * phi2);
    Pair<A> sinu;
    sinu.array() = sin(u.array());
    Pair<A> cosu;
    cosu.array() = cos(u.array());

    // Delta^3 parameter from Gradshteyn & Ryzhik
    Pair<A> D3;
    D3.array() = pow(1 - sinu.array() * sinu.array() * invkc2, 1.5);

    // Lower boundary
    auto Elliptic = IncompleteEllipticIntegrals<Scalar, N>(invkc2, u);
    A f0 = Elliptic.E;

    // Upper boundary
    int nmax = 2 * deg + 4;
    Scalar invkc2_value = invkc2.value();
    std::function<Scalar(Scalar)> func = [nmax, invkc2_value](Scalar x) {
      Scalar sinx2 = sin(x);
      sinx2 *= sinx2;
      return pow(sinx2, nmax) * sqrt(1 - sinx2 * invkc2_value);
    };
    A fN = QUAD.integrate(u(0).value(), u(1).value(), func);

    // Recursion coefficients
    A kc2_alias = kc2;
    std::function<A(int)> funcA = [kc2_alias](int n) {
      return 2 * (n + (n - 1) * kc2_alias) / (2 * n + 1);
    };
    std::function<A(int)> funcB = [kc2_alias](int n) {
      return -(2 * n - 3) / (2 * n + 1) * kc2_alias;
    };
    std::function<A(int)> funcC = [kc2_alias, sinu, cosu, D3](int n) {
      return (pow(sinu(1), (2 * n - 3)) * cosu(1) * D3(1) -
              pow(sinu(0), (2 * n - 3)) * cosu(0) * D3(0)) *
             kc2_alias / (2 * n + 1);
    };

    // Solve the tridiagonal problem
    solve(f0, fN, funcA, funcB, funcC, nmax, W);
  }

public:
  RowVector<A> sT;

  explicit Occultation(int deg) : deg(deg) {}

  /**
      Compute the full solution vector s^T.

  */
  inline void compute(const A &bo_, const A &ro_, const A &f_, const A &theta_,
                      const A &phi1_, const A &phi2_, const A &xi1_,
                      const A &xi2_) {
    bo = bo_;
    ro = ro_;
    f = f_;
    theta = theta_;
    phi1 = phi1_;
    phi2 = phi2_;
    xi1 = xi1_;
    xi2 = xi2_;

    k2 = (1 - bo * bo - ro * ro + 2 * bo * ro) / (4 * bo * ro);
    kc2 = 1 - k2;
    invkc2 = 1.0 / kc2;

    // TODO
    compute_W();
    sT = W.transpose();
  }
};

} // namespace occultation
} // namespace oblate
} // namespace starry

#endif