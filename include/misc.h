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
inline Matrix<Scalar> spotYlm(
    const RowVector<Scalar>& amp,
    const Scalar& sigma,
    const Scalar& lat,
    const Scalar& lon,
    int l,
    wigner::Wigner<Scalar>& W
) {

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
    // normalized so the integral over the sphere is `amp`
    for (int n = 0; n < l + 1; ++n)
        y.row(n * n + n) = 0.25 * amp * sqrt(2 * n + 1) * (IP(n) / IP(0));

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
        W.rotate(u(0), u(1), u(2), atan2(sintheta, costheta), y);
        y = W.rotate_result;
    }

    return y;
}

} // namespace misc
} // namespace starry
#endif