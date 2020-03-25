/**
\file misc.h
\brief Miscelaneous map operations.

*/

#ifndef _STARRY_MISC_H_
#define _STARRY_MISC_H_

#include "utils.h"
#include "wigner.h"

namespace starry {
namespace misc {

using namespace utils;

/**
Compute the Ylm expansion of a spot at a given latitude/longitude on the map.

*/
template <class Scalar>
inline Matrix<Scalar> spotYlm(const RowVector<Scalar> &amp, const Scalar &sigma,
                              const Scalar &lat, const Scalar &lon, int l,
                              wigner::Wigner<Scalar> &W) {
  // Compute the integrals recursively
  Vector<Scalar> IP(l + 1);
  Vector<Scalar> ID(l + 1);
  Matrix<Scalar> y((l + 1) * (l + 1), amp.cols());
  y.setZero();

  // Constants
  Scalar a = 1.0 / (2 * sigma * sigma);
  Scalar sqrta = sqrt(a);
  Scalar erfa = erf(2 * sqrta);
  Scalar term = exp(-4 * a);

  // Seeding values
  IP(0) = root_pi<Scalar>() / (2 * sqrta) * erfa;
  IP(1) = (root_pi<Scalar>() * sqrta * erfa + term - 1) / (2 * a);
  ID(0) = 0;
  ID(1) = IP(0);

  // Recurse
  int sgn = -1;
  for (int n = 2; n < l + 1; ++n) {
    IP(n) = (2.0 * n - 1.0) / (2.0 * n * a) * (ID(n - 1) + sgn * term - 1.0) +
            (2.0 * n - 1.0) / n * IP(n - 1) - (n - 1.0) / n * IP(n - 2);
    ID(n) = (2.0 * n - 1.0) * IP(n - 1) + ID(n - 2);
    sgn *= -1;
  }

  // Compute the coefficients of the expansion
  for (int n = 0; n < l + 1; ++n)
    y.row(n * n + n) = amp * sqrt(2 * n + 1) * (IP(n) / IP(0));

  // Rotate the spot to the correct lat/lon
  // We are computing the compound rotation matrix
  //
  //         R = R(yhat, lon) . R(xhat, -lat)
  //
  // in one go.
  Scalar tol = 10 * mach_eps<Scalar>();
  if ((abs(lat) > tol) || (abs(lon) > tol)) {
    Scalar clat = cos(lat);
    Scalar clon = cos(lon);
    Scalar slat = sin(lat);
    Scalar slon = sin(lon);
    Scalar costheta = 0.5 * (clon + clat + clon * clat - 1);
    UnitVector<Scalar> u;
    u << -slat * (1 + clon), slon * (1 + clat), slon * slat;
    Scalar normu = u.norm();
    u /= normu;
    Scalar sintheta = 0.5 * normu;
    Scalar theta = atan2(sintheta, costheta);
    W.dotR(y.transpose(), u(0), u(1), u(2), -theta);
    y = W.dotR_result.transpose();
  }

  return y;
}

/**
Compute the gradient of the Ylm expansion of a spot at a
given latitude/longitude on the map.

This routine uses forward diff pretty inefficiently and
can be sped up if we think about it more carefully.

*/
template <class Scalar>
inline void spotYlm(const RowVector<Scalar> &amp, const Scalar &sigma_,
                    const Scalar &lat, const Scalar &lon,
                    const Matrix<double> &by, int l, wigner::Wigner<Scalar> &W,
                    RowVector<Scalar> &bamp, Scalar &bsigma, Scalar &blat,
                    Scalar &blon) {

  // Forward diff for sigma
  // TODO: Compute the backprop expression
  using ADType = ADScalar<Scalar, 1>;
  ADType sigma = sigma_;
  sigma.derivatives() = Vector<Scalar>::Unit(1, 0);

  // Compute the integrals recursively
  Vector<ADType> IP(l + 1);
  Vector<ADType> ID(l + 1);
  Vector<Scalar> y((l + 1) * (l + 1));
  y.setZero();
  ADType yn;
  Vector<Scalar> dydsigma((l + 1) * (l + 1));
  dydsigma.setZero();

  // Constants
  ADType a = 1.0 / (2 * sigma * sigma);
  ADType sqrta = sqrt(a);
  ADType erfa = erf(2 * sqrta.value());
  erfa.derivatives()(0) =
      -sqrt(32 / pi<Scalar>()) * exp(-4 * a.value()) * a.value();
  ADType term = exp(-4 * a);

  // Seeding values
  IP(0) = root_pi<Scalar>() / (2 * sqrta) * erfa;
  IP(1) = (root_pi<Scalar>() * sqrta * erfa + term - 1) / (2 * a);
  ID(0) = 0;
  ID(1) = IP(0);

  // Recurse
  int sgn = -1;
  for (int n = 2; n < l + 1; ++n) {
    IP(n) = (2.0 * n - 1.0) / (2.0 * n * a) * (ID(n - 1) + sgn * term - 1.0) +
            (2.0 * n - 1.0) / n * IP(n - 1) - (n - 1.0) / n * IP(n - 2);
    ID(n) = (2.0 * n - 1.0) * IP(n - 1) + ID(n - 2);
    sgn *= -1;
  }

  // Compute the coefficients of the expansion (w/o the amplitude)
  for (int n = 0; n < l + 1; ++n) {
    yn = sqrt(2 * n + 1) * (IP(n) / IP(0));
    y(n * n + n) = yn.value();
    dydsigma(n * n + n) = yn.derivatives()(0);
  }

  // Rotate the spot to the correct lat/lon
  Scalar tol = 10 * mach_eps<Scalar>();
  UnitVector<Scalar> u;
  Scalar normu;
  Scalar theta;
  Scalar clat = cos(lat);
  Scalar clon = cos(lon);
  Scalar slat = sin(lat);
  Scalar slon = sin(lon);

  // Axis & angle of rotation
  if ((abs(lat) > tol) || (abs(lon) > tol)) {
    Scalar costheta = 0.5 * (clon + clat + clon * clat - 1);
    u << -slat * (1 + clon), slon * (1 + clat), slon * slat;
    normu = u.norm();
    u /= normu;
    Scalar sintheta = 0.5 * normu;
    theta = atan2(sintheta, costheta);
  } else {
    theta = 0.0;
    u << 0, 1, 0;
    normu = 1;
  }

  // Gradient w/ respect to lat, lon, and sigma
  Matrix<Scalar> y_amp = y * amp;
  W.dotR(y_amp.transpose(), u(0), u(1), u(2), -theta, by.transpose());

  // lat
  Scalar termz = (clat * (1 + clon) * (1 + clon) * slat - slat * slon * slon) /
                 (normu * normu * normu);
  Scalar dxdl = -clat * (1 + clon) / normu + (1 + clon) * slat * termz;
  Scalar dydl = -slat * slon / normu - (1 + clat) * slon * termz;
  Scalar dzdl = clat * slon / normu - slat * slon * termz;
  Scalar dthetadl = -u(0);
  blat = (dxdl * W.dotR_bx + dydl * W.dotR_by + dzdl * W.dotR_bz -
          dthetadl * W.dotR_btheta);

  // lon
  termz = (clon * (1 + clat) * (1 + clat) * slon - slon * slat * slat) /
          (normu * normu * normu);
  dxdl = slat * slon / normu + (1 + clon) * slat * termz;
  dydl = (1 + clat) * clon / normu - (1 + clat) * slon * termz;
  dzdl = clon * slat / normu - slat * slon * termz;
  dthetadl = u(1);
  blon = (dxdl * W.dotR_bx + dydl * W.dotR_by + dzdl * W.dotR_bz -
          dthetadl * W.dotR_btheta);

  // sigma
  W.dotR(dydsigma.transpose(), u(0), u(1), u(2), -theta);
  dydsigma = W.dotR_result.transpose();
  bsigma = (by.transpose() * dydsigma).dot(amp);

  // Compute the actual result (w/o amplitude)
  W.dotR(y.transpose(), u(0), u(1), u(2), -theta);
  y = W.dotR_result.transpose();

  // Gradient of amplitude
  bamp = y.transpose() * by;
}

} // namespace misc
} // namespace starry
#endif