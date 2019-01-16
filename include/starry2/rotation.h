/**
Spherical harmonic rotation matrices. These are adapted from the Fortran
code of

    Alvarez Collado  et al. (1989) "Rotation of real spherical harmonics".
    compute Physics Communications 52, 3.
    https://doi.org/10.1016/0010-4655(89)90107-0

who computed the Euleriean rotation matrices for real spherical harmonics
from the Wigner-D matrices for complex spherical harmonics.

*/

#ifndef _STARRY_ROT_H_
#define _STARRY_ROT_H_

#include "utils.h"

namespace starry2 {
namespace rotation {

using namespace starry2::utils;

/**
Axis-angle rotation matrix, used to rotate Cartesian
vectors around in 3D space.

*/
template <typename T>
Matrix<T> AxisAngle (
    const UnitVector<T>& u, 
    const T& theta
) {
    Matrix<T> R(3, 3);
    T cost = cos(theta);
    T sint = sin(theta);
    R(0, 0) = cost + u(0) * u(0) * (1 - cost);
    R(0, 1) = u(0) * u(1) * (1 - cost) - u(2) * sint;
    R(0, 2) = u(0) * u(2) * (1 - cost) + u(1) * sint;
    R(1, 0) = u(1) * u(0) * (1 - cost) + u(2) * sint;
    R(1, 1) = cost + u(1) * u(1) * (1 - cost);
    R(1, 2) = u(1) * u(2) * (1 - cost) - u(0) * sint;
    R(2, 0) = u(2) * u(0) * (1 - cost) - u(1) * sint;
    R(2, 1) = u(2) * u(1) * (1 - cost) + u(0) * sint;
    R(2, 2) = cost + u(2) * u(2) * (1 - cost);
    return R;
}

/**
Compute the Wigner d matrices.

*/
template <class Scalar>
inline void dlmn (
    int l, 
    const Scalar& s1,
    const Scalar& c1,
    const Scalar& c2,
    const Scalar& tgbet2,
    const Scalar& s3,
    const Scalar& c3,
    std::vector<Matrix<Scalar>>& D,
    std::vector<Matrix<Scalar>>& R
) {
    int iinf = 1 - l;
    int isup = -iinf;
    int m, mp;
    int al, al1, tal1, amp, laux, lbux, am, lauz, lbuz;
    int sign;
    Scalar ali, auz, aux, cux, fact, term, cuz;
    Scalar cosaux, cosmal, sinmal, cosag, sinag, cosagm, sinagm, cosmga, sinmga;
    Scalar d1, d2;

    // Compute the D[l;m',m) matrix.
    // First row by recurrence (Eq. 19 and 20 in Alvarez Collado et al.)
    D[l](2 * l, 2 * l) = 0.5 * D[l - 1](isup + l - 1, isup + l - 1) 
                                 * (1. + c2);
    D[l](2 * l, 0) = 0.5 * D[l - 1](isup + l - 1, -isup + l - 1) 
                             * (1. - c2);
    for (m = isup; m > iinf - 1; --m)
        D[l](2 * l, m + l) = -tgbet2 * sqrt((l + m + 1.0) / (l - m)) 
                                         * D[l](2 * l, m + 1 + l);

    // The rows of the upper quarter triangle of the D[l;m',m) matrix
    // (Eq. 21 in Alvarez Collado et al.)
    al = l;
    al1 = al - 1;
    tal1 = al + al1;
    ali = 1.0 / al1;
    cosaux = c2 * al * al1;
    for (mp = l - 1; mp > -1; --mp) {
        amp = mp;
        laux = l + mp;
        lbux = l - mp;
        aux = ali / sqrt(laux * lbux);
        cux = sqrt((laux - 1) * (lbux - 1)) * al;
        for (m = isup; m > iinf - 1; --m) {
            am = m;
            lauz = l + m;
            lbuz = l - m;
            auz = 1.0 / sqrt(lauz * lbuz);
            fact = aux * auz;
            term = tal1 * (cosaux - am * amp) 
                        * D[l - 1](mp + l - 1, m + l - 1);
            if ((lbuz != 1) && (lbux != 1)) {
                cuz = sqrt((lauz - 1) * (lbuz - 1));
                term = term - D[l - 2](mp + l - 2, m + l - 2) * cux * cuz;
            }
            D[l](mp + l, m + l) = fact * term;
        }
        ++iinf;
        --isup;
    }

    // The remaining elements of the D[l;m',m) matrix are calculated
    // using the corresponding symmetry relations:
    // reflection ---> ((-1)**(m-m')) D[l;m,m') = D[l;m',m), m'<=m
    // inversion ---> ((-1)**(m-m')) D[l;-m',-m) = D[l;m',m)

    // Reflection
    sign = 1;
    iinf = -l;
    isup = l - 1;
    for (m = l; m > 0; --m) {
        for (mp = iinf; mp < isup + 1; ++mp) {
            D[l](mp + l, m + l) = sign * D[l](m + l, mp + l);
            sign *= -1;
        }
        ++iinf;
        --isup;
    }

    // Inversion
    iinf = -l;
    isup = iinf;
    for (m = l - 1; m > -(l + 1); --m) {
        sign = -1;
        for (mp = isup; mp > iinf - 1; --mp) {
            D[l](mp + l, m + l) = sign * D[l](-mp + l, -m + l);
            sign *= -1;
        }
        ++isup;
    }

    // Compute the real rotation matrices R from the complex ones D
    R[l](0 + l, 0 + l) = D[l](0 + l, 0 + l);
    cosmal = c1;
    sinmal = s1;
    sign = -1;
    Scalar root_two = sqrt(2.0);
    for (mp = 1; mp < l + 1; ++mp) {
        cosmga = c3;
        sinmga = s3;
        aux = root_two * D[l](0 + l, mp + l);
        R[l](mp + l, 0 + l) = aux * cosmal;
        R[l](-mp + l, 0 + l) = aux * sinmal;
        for (m = 1; m < l + 1; ++m) {
            aux = root_two * D[l](m + l, 0 + l);
            R[l](l, m + l) = aux * cosmga;
            R[l](l, -m + l) = -aux * sinmga;
            d1 = D[l](-mp + l, -m + l);
            d2 = sign * D[l](mp + l, -m + l);
            cosag = cosmal * cosmga - sinmal * sinmga;
            cosagm = cosmal * cosmga + sinmal * sinmga;
            sinag = sinmal * cosmga + cosmal * sinmga;
            sinagm = sinmal * cosmga - cosmal * sinmga;
            R[l](mp + l, m + l) = d1 * cosag + d2 * cosagm;
            R[l](mp + l, -m + l) = -d1 * sinag + d2 * sinagm;
            R[l](-mp + l, m + l) = d1 * sinag + d2 * sinagm;
            R[l](-mp + l, -m + l) = d1 * cosag - d2 * cosagm;
            aux = cosmga * c3 - sinmga * s3;
            sinmga = sinmga * c3 + cosmga * s3;
            cosmga = aux;
        }
        sign *= -1;
        aux = cosmal * c1 - sinmal * s1;
        sinmal = sinmal * c1 + cosmal * s1;
        cosmal = aux;
    }
}

/**
Compute the Wigner D matrices.

*/
template <class Scalar>
inline void rotar (
    const int lmax,
    const Scalar& c1,
    const Scalar& s1,
    const Scalar& c2,
    const Scalar& s2,
    const Scalar& c3,
    const Scalar& s3,
    const Scalar& tol,
    std::vector<Matrix<Scalar>>& D,
    std::vector<Matrix<Scalar>>& R
) {
    Scalar cosag, cosamg, sinag, sinamg, tgbet2;
    Scalar root_two = sqrt(2.0);

    // Compute the initial matrices D0, R0, D1 and R1
    D[0](0, 0) = 1.0;
    R[0](0, 0) = 1.0;
    D[1](2, 2) = 0.5 * (1.0 + c2);
    D[1](2, 1) = -s2 / root_two;
    D[1](2, 0) = 0.5 * (1.0 - c2);
    D[1](1, 2) = -D[1](2, 1);
    D[1](1, 1) = D[1](2, 2) - D[1](2, 0);
    D[1](1, 0) = D[1](2, 1);
    D[1](0, 2) = D[1](2, 0);
    D[1](0, 1) = D[1](1, 2);
    D[1](0, 0) = D[1](2, 2);
    cosag = c1 * c3 - s1 * s3;
    cosamg = c1 * c3 + s1 * s3;
    sinag = s1 * c3 + c1 * s3;
    sinamg = s1 * c3 - c1 * s3;
    R[1](1, 1) = D[1](1, 1);
    R[1](2, 1) = root_two * D[1](1, 2) * c1;
    R[1](0, 1) = root_two * D[1](1, 2) * s1;
    R[1](1, 2) = root_two * D[1](2, 1) * c3;
    R[1](1, 0) = -root_two * D[1](2, 1) * s3;
    R[1](2, 2) = D[1](2, 2) * cosag - D[1](2, 0) * cosamg;
    R[1](2, 0) = -D[1](2, 2) * sinag - D[1](2, 0) * sinamg;
    R[1](0, 2) = D[1](2, 2) * sinag - D[1](2, 0) * sinamg;
    R[1](0, 0) = D[1](2, 2) * cosag + D[1](2, 0) * cosamg;

    // The remaining matrices are calculated using 
    // symmetry and and recurrence relations
    if (abs(s2) < tol)
        tgbet2 = s2; // = 0
    else
        tgbet2 = (1.0 - c2) / s2;

    for (int l = 2; l < lmax + 1; ++l)
        dlmn(l, s1, c1, c2, tgbet2, s3, c3, D, R);

    return;
}

/**

*/
template <typename Scalar>
inline void axisAngleToEuler (
    const Scalar& axis_x,
    const Scalar& axis_y,
    const Scalar& costheta,
    const Scalar& sintheta,
    const Scalar& tol,
    Scalar& cosalpha,
    Scalar& sinalpha,
    Scalar& cosbeta,
    Scalar& sinbeta,
    Scalar& cosgamma,
    Scalar& singamma
) {
    // Construct the axis-angle rotation matrix R_A
    Scalar RA01 = axis_x * axis_y * (1 - costheta);
    Scalar RA02 = axis_y * sintheta;
    Scalar RA11 = costheta + axis_y * axis_y * (1 - costheta);
    Scalar RA12 = -axis_x * sintheta;
    Scalar RA20 = -axis_y * sintheta;
    Scalar RA21 = axis_x * sintheta;
    Scalar RA22 = costheta;

    // Determine the Euler angles
    Scalar norm1, norm2;
    if ((RA22 < -1 + tol) && (RA22 > -1 - tol)) {
        cosbeta = RA22; // = -1
        sinbeta = 1 + RA22; // = 0
        cosgamma = RA11;
        singamma = RA01;
        cosalpha = -RA22; // = 1
        sinalpha = 1 + RA22; // = 0
    } else if ((RA22 < 1 + tol) && (RA22 > 1 - tol)) {
        cosbeta = RA22; // = 1
        sinbeta = 1 - RA22; // = 0
        cosgamma = RA11;
        singamma = -RA01;
        cosalpha = RA22; // = 1
        sinalpha = 1 - RA22; // = 0
    } else {
        cosbeta = RA22;
        sinbeta = sqrt(1 - cosbeta * cosbeta);
        norm1 = sqrt(RA20 * RA20 + RA21 * RA21);
        norm2 = sqrt(RA02 * RA02 + RA12 * RA12);
        cosgamma = -RA20 / norm1;
        singamma = RA21 / norm1;
        cosalpha = RA02 / norm2;
        sinalpha = RA12 / norm2;
    }
}

/**
Rotation matrix class for the spherical harmonics.

*/
template <class MapType>
class Wigner {

protected:

    using Scalar = typename MapType::Scalar;

    const int lmax;                                                            /**< Highest degree of the map */
    const int N;                                                               /**< Number of map coefficients */
    const int ncol;                                                            /**< Number of map columns */
    const int nflx;                                                            /**< Number of contracted map columns */
    const Scalar tol;                                                          /**< Numerical tolerance used to prevent division-by-zero errors */

    // References to the base map and the rotation axis
    MapType& y;                                                                /**< Reference to the spherical harmonic map to be rotated */
    UnitVector<Scalar>& axis;                                                  /**< Reference to the rotation axis */

    std::vector<Matrix<Scalar>> DZeta;                                         /**< The complex Wigner matrix in the `zeta` frame */
    std::vector<Matrix<Scalar>> RZeta;                                         /**< The real Wigner matrix in the `zeta` frame */
    std::vector<Matrix<Scalar>> RZetaInv;                                      /**< The inverse of the real Wigner matrix in the `zeta` frame */
    MapType y_zeta;                                                            /**< The base map in the `zeta` frame */
    
    // Temporaries
    MapType y_zeta_rot;                                                        /**< The base map in the `zeta` frame after a `zhat` rotation */
    MapType y_rev;                                                             /**< Degree-wise reverse of the spherical harmonic map */
    Matrix<Scalar> y_rev_ctr;                                                  /**< Degree-wise reverse of the contracted spherical harmonic map */
    Vector<Scalar> cosmt;                                                      /**< Vector of cos(m theta) values */
    Vector<Scalar> sinmt;                                                      /**< Vector of sin(m theta) values */
    Vector<Scalar> cosnt;                                                      /**< Vector of cos(n theta) values */
    Vector<Scalar> sinnt;                                                      /**< Vector of sin(n theta) values */

    // Methods
    inline void computeZeta (
        const Scalar& axis_x,
        const Scalar& axis_y,
        const Scalar& costheta,
        const Scalar& sintheta
    );

    inline void rotatez (
        const Scalar& costheta, 
        const Scalar& sintheta,
        const MapType& yin, 
        MapType& yout
    );

public:

    std::vector<Matrix<Scalar>> R;                                             /**< The full rotation matrix for real spherical harmonics */
    std::vector<Matrix<Scalar>> DRDtheta;                                      /**< The derivative of the rotation matrix with respect to theta */
    
    inline void updateZeta ();

    inline void updateYZeta ();

    inline void rotate (
        const Scalar& costheta, 
        const Scalar& sintheta
    );

    inline void rotate (
        const Scalar& costheta, 
        const Scalar& sintheta,
        MapType& yout
    );

    template <typename T>
    inline void rotate_about_z (
        const Scalar& costheta, 
        const Scalar& sintheta,
        const MatrixBase<T>& yin, 
        MatrixBase<T>& yout
    );

    template <typename T>
    inline void rotate_about_z (
        const Scalar& costheta,
        const Scalar& sintheta,
        const MatrixBase<T>& yin, 
        MatrixBase<T>& yout,
        const RowVector<Scalar>& v,
        RowVector<Scalar>& vR,
        RowVector<Scalar>& vDRDtheta
    );

    inline void compute (
        const Scalar& costheta, 
        const Scalar& sintheta
    );

    Wigner(
        int lmax, 
        int ncol, 
        int nflx,
        MapType& y, 
        UnitVector<Scalar>& axis
    ) : 
        lmax(lmax), 
        N((lmax + 1) * (lmax + 1)), 
        ncol(ncol),
        nflx(nflx),
        tol(10 * mach_eps<Scalar>()), 
        y(y), 
        axis(axis),
        DZeta(lmax + 1),
        RZeta(lmax + 1),
        RZetaInv(lmax + 1),
        y_zeta(N, ncol),
        R(lmax + 1),
        DRDtheta(lmax + 1)
    {
        // Allocate the Wigner matrices
        for (int l = 0; l < lmax + 1; ++l) {
            int sz = 2 * l + 1;
            DZeta[l].resize(sz, sz);
            RZeta[l].resize(sz, sz);
            RZetaInv[l].resize(sz, sz);
            R[l].resize(sz, sz);
            DRDtheta[l].resize(sz, sz);
        }

        // Initialize our z rotation vectors
        cosnt.resize(max(2, lmax + 1));
        cosnt(0) = 1.0;
        sinnt.resize(max(2, lmax + 1));
        sinnt(0) = 0.0;
        cosmt.resize(N);
        sinmt.resize(N);
        y_rev.resize(N, ncol);
        y_rev_ctr.resize(N, nflx);
    }

};

/**
Rotate the base map about the current axis
given `costheta` and `sintheta`

*/
template <class MapType>
inline void Wigner<MapType>::rotate (
    const Scalar& costheta,
    const Scalar& sintheta,
    MapType& yout
) {
    // Rotate `yzeta` about `zhat` and store in `yzeta_rot`
    rotatez(costheta, sintheta, y_zeta, y_zeta_rot);

    // Rotate out of the `zeta` frame
    for (int l = 0; l < lmax + 1; l++) {
        yout.block(l * l, 0, 2 * l + 1, ncol) =
            RZetaInv[l] * y_zeta_rot.block(l * l, 0, 2 * l + 1, ncol);
    }
}

/**
Rotate the base map *in place* about the current axis
given `costheta` and `sintheta`.

*/
template <class MapType>
inline void Wigner<MapType>::rotate (
    const Scalar& costheta,
    const Scalar& sintheta
) {
    rotate(costheta, sintheta, y);
    for (int l = 0; l < lmax + 1; ++l) {
        y_zeta.block(l * l, 0, 2 * l + 1, ncol) =
            RZeta[l] * y.block(l * l, 0, 2 * l + 1, ncol);
    }
}


/**
Perform a fast rotation about the z axis, skipping the Wigner matrix computation.
See https://github.com/rodluger/starry/issues/137#issuecomment-405975092
This is the user-facing version of the `computez` method below.

*/
template <class MapType>
template <typename T>
inline void Wigner<MapType>::rotate_about_z (
    const Scalar& costheta,
    const Scalar& sintheta,
    const MatrixBase<T>& yin, 
    MatrixBase<T>& yout
) {
    cosnt(1) = costheta;
    sinnt(1) = sintheta;
    for (int n = 2; n < lmax + 1; ++n) {
        cosnt(n) = 2.0 * cosnt(n - 1) * cosnt(1) - cosnt(n - 2);
        sinnt(n) = 2.0 * sinnt(n - 1) * cosnt(1) - sinnt(n - 2);
    }
    int n = 0;
    for (int l = 0; l < lmax + 1; ++l) {
        for (int m = -l; m < 0; ++m) {
            cosmt(n) = cosnt(-m);
            sinmt(n) = -sinnt(-m);
            y_rev_ctr.row(n) = yin.row(l * l + l - m);
            ++n;
        }
        for (int m = 0; m < l + 1; ++m) {
            cosmt(n) = cosnt(m);
            sinmt(n) = sinnt(m);
            y_rev_ctr.row(n) = yin.row(l * l + l - m);
            ++n;
        }
    }
    yout = (
        yin.transpose().array().rowwise() * cosmt.array().transpose() -
        y_rev_ctr.transpose().array().rowwise() * sinmt.array().transpose()
    ).transpose();
}


/**
Compute the same fast rotation about the z axis as above.
Additionally, pre-multiply the rotation matrix `R`
by a row vector `v` and return `v . R`. Also compute the derivative
of `R` with respect to `theta` and return `v . dR / dtheta`. This 
is handy for computing gradients of the occultation flux.

*/
template <class MapType>
template <typename T>
inline void Wigner<MapType>::rotate_about_z (
    const Scalar& costheta,
    const Scalar& sintheta,
    const MatrixBase<T>& yin, 
    MatrixBase<T>& yout,
    const RowVector<Scalar>& v,
    RowVector<Scalar>& vR,
    RowVector<Scalar>& vDRDtheta
) {
    rotate_about_z(costheta, sintheta, yin, yout);
    for (int l = 0; l < lmax + 1; ++l) {
        for (int j = 0; j < 2 * l + 1; ++j) {
            int m = j - l;
            vR(l * l + j) = v(l * l + j) * cosmt(l * l + j) +
                            v(l * l + 2 * l - j) * sinmt(l * l + j);
            vDRDtheta(l * l + j) = v(l * l + 2 * l - j) * m * cosmt(l * l + j) -
                                   v(l * l + j) * m * sinmt(l * l + j);
        }
    }
}

/**
Explicitly compute the full rotation matrix and its derivative.
The full rotation matrix is factored as

    R = RZetaInv . Rz . RZeta

where Rz has the form

        ...                             ...
            C3                      S3
                C2              S2
                    C1      S1
                         1
                    -S1      C1
                -S2              C2
            -S3                      C3
        ...                             ...

with CX = cos(X theta) and SX = sin(X theta). The derivative of R with
respect to theta is

    dR/Dtheta = RZetaInv . dRz/Dtheta . RZeta

where dRz/Dtheta has the form

        ...                                 ...
            -3 S3                      3 C3
                -2 S2             2 C2
                      -S1      C1
                           0
                    -C1      -S1
                -2 C2             -2 S2
            -3 C3                      -3 S3
        ...                                 ...

*/
template <class MapType>
inline void Wigner<MapType>::compute (
    const Scalar& costheta,
    const Scalar& sintheta
) {

    // Compute the cos and sin vectors for the zhat rotation
    cosnt(1) = costheta;
    sinnt(1) = sintheta;
    for (int n = 2; n < lmax + 1; ++n) {
        cosnt(n) = 2.0 * cosnt(n - 1) * cosnt(1) - cosnt(n - 2);
        sinnt(n) = 2.0 * sinnt(n - 1) * cosnt(1) - sinnt(n - 2);
    }
    int n = 0;
    for (int l = 0; l < lmax + 1; ++l) {
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
    }

    // Now compute the full rotation matrix
    int m;
    for (int l = 0; l < lmax + 1; ++l) {
        for (int j = 0; j < 2 * l + 1; ++j) {
            m = j - l;
            R[l].col(j) = RZetaInv[l].col(j) * cosmt(l * l + j) +
                          RZetaInv[l].col(2 * l - j) * sinmt(l * l + j);
            DRDtheta[l].col(j) = RZetaInv[l].col(2 * l - j) * m * cosmt(l * l + j) -
                                 RZetaInv[l].col(j) * m * sinmt(l * l + j);
        }
        R[l] = R[l] * RZeta[l];
        DRDtheta[l] = DRDtheta[l] * RZeta[l];
    }

}

/**
Update the zeta rotation matrix.

*/
template <class MapType>
inline void Wigner<MapType>::updateZeta () 
{
    // Compute the rotation transformation into and out of the `zeta` frame
    Scalar cos_zeta = axis(2);
    Scalar sin_zeta = sqrt(1 - axis(2) * axis(2));
    Scalar norm = sqrt(axis(0) * axis(0) + axis(1) * axis(1));
    if (abs(norm) < tol) {
        // The rotation axis is zhat, so our zeta transform
        // is just the identity matrix.
        for (int l = 0; l < lmax + 1; l++) {
            if (axis(2) > 0) {
                RZeta[l] = Matrix<Scalar>::Identity(2 * l + 1, 2 * l + 1);
                RZetaInv[l] = Matrix<Scalar>::Identity(2 * l + 1, 2 * l + 1);
            } else {
                RZeta[l] = -Matrix<Scalar>::Identity(2 * l + 1, 2 * l + 1);
                RZetaInv[l] = -Matrix<Scalar>::Identity(2 * l + 1, 2 * l + 1);
            }
        }
    } else {
        // We need to compute the actual Wigner matrices
        Scalar axis_x = axis(1) / norm;
        Scalar axis_y = -axis(0) / norm;
        computeZeta(axis_x, axis_y, cos_zeta, sin_zeta);
    }
}

/**
Update the base map in the zeta frame.

*/
template <class MapType>
inline void Wigner<MapType>::updateYZeta () 
{
    // Update the map in the `zeta` frame
    for (int l = 0; l < lmax + 1; ++l) {
        y_zeta.block(l * l, 0, 2 * l + 1, ncol) =
            RZeta[l] * y.block(l * l, 0, 2 * l + 1, ncol);
    }
}

/**
Perform a fast rotation about the z axis, skipping the Wigner matrix computation.
See https://github.com/rodluger/starry/issues/137#issuecomment-405975092

*/
template <class MapType>
inline void Wigner<MapType>::rotatez (
    const Scalar& costheta,
    const Scalar& sintheta,
    const MapType& yin, 
    MapType& yout
) {
    cosnt(1) = costheta;
    sinnt(1) = sintheta;
    for (int n = 2; n < lmax + 1; ++n) {
        cosnt(n) = 2.0 * cosnt(n - 1) * cosnt(1) - cosnt(n - 2);
        sinnt(n) = 2.0 * sinnt(n - 1) * cosnt(1) - sinnt(n - 2);
    }
    int n = 0;
    for (int l = 0; l < lmax + 1; ++l) {
        for (int m = -l; m < 0; ++m) {
            cosmt(n) = cosnt(-m);
            sinmt(n) = -sinnt(-m);
            y_rev.row(n) = yin.row(l * l + l - m);
            ++n;
        }
        for (int m = 0; m < l + 1; ++m) {
            cosmt(n) = cosnt(m);
            sinmt(n) = sinnt(m);
            y_rev.row(n) = yin.row(l * l + l - m);
            ++n;
        }
    }
    yout = (
        yin.transpose().array().rowwise() * cosmt.array().transpose() -
        y_rev.transpose().array().rowwise() * sinmt.array().transpose()
    ).transpose();
}

/**
Compute the axis-angle rotation matrix for real spherical 
harmonics up to order lmax.

*/
template <class MapType>
inline void Wigner<MapType>::computeZeta (
    const Scalar& axis_x,
    const Scalar& axis_y,
    const Scalar& costheta,
    const Scalar& sintheta
) {
    // Trivial case
    if (lmax == 0) {
        RZeta[0](0, 0) = 1;
        RZetaInv[0](0, 0) = 1;
        return;
    }

    // Get Euler angles
    Scalar cosalpha, sinalpha, cosbeta, sinbeta, cosgamma, singamma;
    axisAngleToEuler(axis_x, axis_y, costheta, sintheta, tol,
                     cosalpha, sinalpha, cosbeta, sinbeta, 
                     cosgamma, singamma);

    // Call the eulerian rotation function
    rotar(lmax, cosalpha, sinalpha, cosbeta, sinbeta, 
          cosgamma, singamma, tol, DZeta, RZeta);

    // Compute the inverse transform (trivial!)
    for (int l = 0; l < lmax + 1; ++l)
        RZetaInv[l] = RZeta[l].transpose();
}

} // namespace rotation
} // namespace starry2

#endif
