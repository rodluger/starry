/**
\file wigner.h
\brief Spherical harmonic rotation utilities.

*/

#ifndef _STARRY_WIGNER_H_
#define _STARRY_WIGNER_H_

#include "utils.h"

namespace starry { 
namespace wigner {

using namespace utils;

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
                             * (Scalar(1.0) + c2);
    D[l](2 * l, 0) = 0.5 * D[l - 1](isup + l - 1, -isup + l - 1) 
                         * (Scalar(1.0) - c2);
    for (m = isup; m > iinf - 1; --m)
        D[l](2 * l, m + l) = -tgbet2 * sqrt(Scalar(l + m + 1) / (l - m)) 
                                         * D[l](2 * l, m + 1 + l);

    // The rows of the upper quarter triangle of the D[l;m',m) matrix
    // (Eq. 21 in Alvarez Collado et al.)
    al = l;
    al1 = al - 1;
    tal1 = al + al1;
    ali = Scalar(1.0) / al1;
    cosaux = c2 * al * al1;
    for (mp = l - 1; mp > -1; --mp) {
        amp = mp;
        laux = l + mp;
        lbux = l - mp;
        aux = ali / sqrt(Scalar(laux * lbux));
        cux = sqrt(Scalar((laux - 1) * (lbux - 1))) * al;
        for (m = isup; m > iinf - 1; --m) {
            am = m;
            lauz = l + m;
            lbuz = l - m;
            auz = Scalar(1.0) / sqrt(Scalar(lauz * lbuz));
            fact = aux * auz;
            term = tal1 * (cosaux - Scalar(am * amp)) 
                        * D[l - 1](mp + l - 1, m + l - 1);
            if ((lbuz != 1) && (lbux != 1)) {
                cuz = sqrt(Scalar((lauz - 1) * (lbuz - 1)));
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
    Scalar root_two = sqrt(Scalar(2.0));
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
    const int ydeg,
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
    Scalar root_two = sqrt(Scalar(2.0));

    // Compute the initial matrices D0, R0, D1 and R1
    D[0](0, 0) = 1.0;
    R[0](0, 0) = 1.0;
    D[1](2, 2) = 0.5 * (Scalar(1.0) + c2);
    D[1](2, 1) = -s2 / root_two;
    D[1](2, 0) = 0.5 * (Scalar(1.0) - c2);
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
        tgbet2 = (Scalar(1.0) - c2) / s2;

    for (int l = 2; l < ydeg + 1; ++l)
        dlmn(l, s1, c1, c2, tgbet2, s3, c3, D, R);

    return;
}

/**
Compute the Euler angles from an axis and an angle.

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
    Scalar RA01 = axis_x * axis_y * (Scalar(1.0) - costheta);
    Scalar RA02 = axis_y * sintheta;
    Scalar RA11 = costheta + axis_y * axis_y * (Scalar(1.0) - costheta);
    Scalar RA12 = -axis_x * sintheta;
    Scalar RA20 = -axis_y * sintheta;
    Scalar RA21 = axis_x * sintheta;
    Scalar RA22 = costheta;

    // Determine the Euler angles
    Scalar norm1, norm2;
    if ((RA22 < Scalar(-1.0) + tol) && (RA22 > Scalar(-1.0) - tol)) {
        cosbeta = RA22; // = -1
        sinbeta = Scalar(1.0) + RA22; // = 0
        cosgamma = RA11;
        singamma = RA01;
        cosalpha = -RA22; // = 1
        sinalpha = Scalar(1.0) + RA22; // = 0
    } else if ((RA22 < Scalar(1.0) + tol) && (RA22 > Scalar(1.0) - tol)) {
        cosbeta = RA22; // = 1
        sinbeta = Scalar(1.0) - RA22; // = 0
        cosgamma = RA11;
        singamma = -RA01;
        cosalpha = RA22; // = 1
        sinalpha = Scalar(1.0) - RA22; // = 0
    } else {
        cosbeta = RA22;
        sinbeta = sqrt(Scalar(1.0) - cosbeta * cosbeta);
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
template <class Scalar>
class Wigner {

protected:

    // Sizes
    const int ydeg;
    const int Ny;                                                              /**< Number of spherical harmonic `(l, m)` coefficients */
    const int udeg;
    const int Nu;                                                              /**< Number of limb darkening coefficients */
    const int fdeg;
    const int Nf;                                                              /**< Number of filter `(l, m)` coefficients */
    const int deg;
    const int N;

    // Helper variables
    Matrix<Scalar> cosmt;                                                      /**< Matrix of cos(m theta) values */
    Matrix<Scalar> sinmt;                                                      /**< Matrix of sin(m theta) values */
    Matrix<Scalar> cosnt;                                                      /**< Matrix of cos(n theta) values */
    Matrix<Scalar> sinnt;                                                      /**< Matrix of sin(n theta) values */
    Vector<Scalar> tmp_c, tmp_s;
    Vector<Scalar> theta_Rz_cache, costheta, sintheta;
    Scalar obl_cache, inc_cache;
    Scalar tol;

    // The complex rotation matrix
    std::vector<Matrix<Scalar>> D;

public:

    // The real rotation matrix
    std::vector<Matrix<Scalar>> R;

protected:

    // The full XY rotation matrices
    std::vector<Matrix<Scalar>> Dxy;                                           /**< The complex Wigner matrix in the `xy` frame */
    std::vector<Matrix<Scalar>> Rxy;                                           /**< The real Wigner matrix in the `xy` frame */
    using ADType = ADScalar<Scalar, 2>;                                        /**< AutoDiffScalar type for derivs w.r.t. the rotation axis */
    std::vector<Matrix<ADType>> Dxy_ad;                                        /**< [AutoDiffScalar] The complex Wigner matrix in the `xy` frame */
    std::vector<Matrix<ADType>> Rxy_ad;                                        /**< [AutoDiffScalar] The real Wigner matrix in the `xy` frame */
    std::vector<Matrix<Scalar>> DRxyDinc;
    std::vector<Matrix<Scalar>> DRxyDobl;

public:

    // Full rotation results
    Matrix<Scalar> rotate_result;

    // Z rotation results
    Matrix<Scalar> dotRz_result;
    Vector<Scalar> dotRz_btheta;
    Matrix<Scalar> dotRz_bM;

    // XY rotation results
    Matrix<Scalar> dotRxy_result;
    Scalar dotRxy_binc;
    Scalar dotRxy_bobl;
    Matrix<Scalar> dotRxy_bM;

    Matrix<Scalar> dotRxyT_result;
    Scalar dotRxyT_binc;
    Scalar dotRxyT_bobl;
    Matrix<Scalar> dotRxyT_bM;

    Wigner(
        int ydeg, 
        int udeg,
        int fdeg
    ) :
        ydeg(ydeg), 
        Ny((ydeg + 1) * (ydeg + 1)),
        udeg(udeg),
        Nu(udeg + 1),
        fdeg(fdeg),
        Nf((fdeg + 1) * (fdeg + 1)),
        deg(ydeg + udeg + fdeg),
        N((deg + 1) * (deg + 1)),
        theta_Rz_cache(0),
        obl_cache(NAN),
        inc_cache(NAN)
    {

        // Allocate the Wigner matrices
        D.resize(ydeg + 1);
        R.resize(ydeg + 1);
        Dxy.resize(ydeg + 1);
        Rxy.resize(ydeg + 1);
        Dxy_ad.resize(ydeg + 1);
        Rxy_ad.resize(ydeg + 1);
        DRxyDinc.resize(ydeg + 1);
        DRxyDobl.resize(ydeg + 1);
        for (int l = 0; l < ydeg + 1; ++l) {
            int sz = 2 * l + 1;
            D[l].resize(sz, sz);
            R[l].resize(sz, sz);
            Dxy[l].resize(sz, sz);
            Rxy[l].resize(sz, sz);
            Dxy_ad[l].resize(sz, sz);
            Rxy_ad[l].resize(sz, sz);
            DRxyDobl[l].resize(sz, sz);
            DRxyDinc[l].resize(sz, sz);
        }

        // Misc
        tol = 10 * mach_eps<Scalar>();

    }

    /**
    Compute the full rotation matrix R.
    \todo: Compute gradients here as well.

    */
    inline void compute (
        const Scalar& axis_x,
        const Scalar& axis_y,
        const Scalar& axis_z,
        const Scalar& theta
    ) {
        Scalar costheta = cos(theta);
        Scalar sintheta = sin(theta);
        Scalar RA01 = axis_x * axis_y * (1 - costheta) - axis_z * sintheta;
        Scalar RA02 = axis_x * axis_z * (1 - costheta) + axis_y * sintheta;
        Scalar RA11 = costheta + axis_y * axis_y * (1 - costheta);
        Scalar RA12 = axis_y * axis_z * (1 - costheta) - axis_x * sintheta;
        Scalar RA20 = axis_z * axis_x * (1 - costheta) - axis_y * sintheta;
        Scalar RA21 = axis_z * axis_y * (1 - costheta) + axis_x * sintheta;
        Scalar RA22 = costheta + axis_z * axis_z * (1 - costheta);

        // Determine the Euler angles
        Scalar cosalpha, sinalpha, cosbeta, sinbeta, cosgamma, singamma;
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

        // Call the Eulerian rotation function
        rotar(ydeg, cosalpha, sinalpha, cosbeta, sinbeta, 
                cosgamma, singamma, tol, D, R);
    }

    /**
    Rotate a Ylm matrix or vector ``M`` given an axis of rotation
    and an angle and store the result in ``rotate_result``. This
    method doesn't do any caching and does not propagate gradients.
    It is used strictly to rotate maps *outside* of any flux
    calculations.

    */
    template <typename T1, bool M_IS_ROW_VECTOR=(T1::RowsAtCompileTime==1)>
    inline void rotate (
        const Scalar& axis_x,
        const Scalar& axis_y,
        const Scalar& axis_z,
        const Scalar& theta,
        const MatrixBase<T1>& M
    ) {

        // Compute the rotation transformation into and out of the `xy` frame
        if ((ydeg == 0) || (abs(theta) < tol)) {

            // Trivial case, no rotation
            rotate_result = M;

        } else if (abs(axis_z) > 1 - tol) {
            
            // The axis of rotation is zhat, so this is easy
            size_t nw = M.cols();

            // Compute the sin & cos matrices
            // Recall that `computeRz` effectively computes
            // the *transpose* of R_z, so we need to negate `theta`.
            OneByOne<Scalar> theta_v;
            theta_v(0) = -theta;
            if (axis_z < 0) theta_v(0) *= -1;
            computeRz(theta_v); 

            // Init result
            rotate_result.resize(Ny, nw);

            // Dot them in
            for (int l = 0; l < ydeg + 1; ++l) {
                for (int j = 0; j < 2 * l + 1; ++j) {
                        rotate_result.row(l * l + j) = 
                            M.row(l * l + j) * cosmt(0, l * l + j) + 
                            M.row(l * l + 2 * l - j) * sinmt(0, l * l + j);
                }
            }
            
        } else {

            // Construct the axis-angle rotation matrix R_A
            Scalar costheta = cos(theta);
            Scalar sintheta = sin(theta);
            Scalar RA02 = axis_x * axis_z * (1 - costheta) + axis_y * sintheta;
            Scalar RA12 = axis_y * axis_z * (1 - costheta) - axis_x * sintheta;
            Scalar RA20 = axis_z * axis_x * (1 - costheta) - axis_y * sintheta;
            Scalar RA21 = axis_z * axis_y * (1 - costheta) + axis_x * sintheta;
            Scalar RA22 = costheta + axis_z * axis_z * (1 - costheta);

            // Determine the Euler angles
            Scalar cosalpha, sinalpha, cosbeta, sinbeta, cosgamma, singamma;
            Scalar norm1, norm2;
            cosbeta = RA22;
            sinbeta = sqrt(1 - cosbeta * cosbeta);
            norm1 = sqrt(RA20 * RA20 + RA21 * RA21);
            norm2 = sqrt(RA02 * RA02 + RA12 * RA12);
            cosgamma = -RA20 / norm1;
            singamma = RA21 / norm1;
            cosalpha = RA02 / norm2;
            sinalpha = RA12 / norm2;

            // Call the Eulerian rotation function
            rotar(ydeg, cosalpha, sinalpha, cosbeta, sinbeta, 
                  cosgamma, singamma, tol, D, R);

            // Shape checks
            size_t nw = M.cols();

            // Init result
            rotate_result.resize(Ny, nw);

            // Dot the matrix in
            for (int l = 0; l < ydeg + 1; ++l) {
                rotate_result.block(l * l, 0, 2 * l + 1, nw) =
                    R[l] * M.block(l * l, 0, 2 * l + 1, nw);
            }

        }
    }

    /**
    Compute the ``Rxy`` rotation matrix.
    
    */
    inline void computeRxy(
        const Scalar& inc_,
        const Scalar& obl_
    ) {
        
        // Check the cache
        if ((inc_ == inc_cache) && (obl_ == obl_cache)) {
            return;
        }
        inc_cache = inc_;
        obl_cache = obl_;

        // Convert to ADType
        ADType inc = inc_;
        ADType obl = obl_;
        inc.derivatives() = Vector<Scalar>::Unit(2, 0);
        obl.derivatives() = Vector<Scalar>::Unit(2, 1);
        ADType sini = sin(inc),
               cosi = cos(inc),
               neg_sino = -sin(obl),
               coso = cos(obl);

        // Compute the rotation transformation into and out of the `xy` frame
        if (ydeg == 0) {

            // Trivial case
            Rxy[0](0, 0) = 1;
            DRxyDinc[0].setZero();
            DRxyDobl[0].setZero();

        } else if (abs(sini.value()) < tol) {
            
            // The rotation axis is about +/- zhat
            if (cosi.value() > 0) {

                // Trivial: the xy frame is the current frame
                for (int l = 0; l < ydeg + 1; l++) {
                    Rxy[l] = Matrix<Scalar>::Identity(2 * l + 1, 2 * l + 1);
                } 

            } else {

                // We must flip the body around, since the axis is -zhat
                rotar(ydeg, Scalar(1.0), Scalar(0.0), Scalar(-1.0), 
                      Scalar(0.0), Scalar(1.0), Scalar(0.0), tol, Dxy, Rxy);

            }

            // All derivs are zero
            for (int l = 0; l < ydeg + 1; l++) {
                DRxyDinc[l].setZero();
                DRxyDobl[l].setZero();
            }

        } else {

            // We need to compute the actual Wigner matrices
            ADType tol_ad = tol;
            ADType cosalpha, sinalpha, cosbeta, sinbeta, cosgamma, singamma;
            axisAngleToEuler(coso, neg_sino, cosi, sini, tol_ad,
                             cosalpha, sinalpha, cosbeta, sinbeta, 
                             cosgamma, singamma);

            // Call the Eulerian rotation function
            rotar(ydeg, cosalpha, sinalpha, cosbeta, sinbeta, 
                  cosgamma, singamma, tol_ad, Dxy_ad, Rxy_ad);

            // Extract the matrices and their derivatives
            for (int l = 0; l < ydeg + 1; ++l) {
                // \todo This data copy is *very* slow
                for (int i = 0; i < 2 * l + 1; ++i) {
                    for (int j = 0; j < 2 * l + 1; ++j) {
                        Rxy[l](i, j) = Rxy_ad[l](i, j).value();
                        DRxyDinc[l](i, j) = Rxy_ad[l](i, j).derivatives()(0);
                        DRxyDobl[l](i, j) = Rxy_ad[l](i, j).derivatives()(1);
                    }
                }
            }

        }
    }

    /**
    Compute the ``Rz`` rotation matrix.

    */
    inline void computeRz(
        const Vector<Scalar>& theta
    ) {

        // Length of timeseries
        size_t npts = theta.size();

        // Check the cache
        if ((npts == size_t(theta_Rz_cache.size())) && (theta == theta_Rz_cache)) {
            return;
        } else if (npts == 0) {
            return;
        }
        theta_Rz_cache = theta;

        // Compute sin & cos
        costheta = theta.array().cos();
        sintheta = theta.array().sin();

        // Initialize our z rotation vectors
        cosnt.resize(npts, max(2, deg + 1));
        cosnt.col(0).setOnes();
        sinnt.resize(npts, max(2, deg + 1));
        sinnt.col(0).setZero();
        cosmt.resize(npts, N);
        sinmt.resize(npts, N);

        // Compute the cos and sin vectors for the zhat rotation
        cosnt.col(1) = costheta;
        sinnt.col(1) = sintheta;
        for (int n = 2; n < deg + 1; ++n) {
            cosnt.col(n) = 2.0 * cosnt.col(n - 1).cwiseProduct(cosnt.col(1)) 
                           - cosnt.col(n - 2);
            sinnt.col(n) = 2.0 * sinnt.col(n - 1).cwiseProduct(cosnt.col(1)) 
                           - sinnt.col(n - 2);
        }
        int n = 0;
        for (int l = 0; l < deg + 1; ++l) {
            for (int m = -l; m < 0; ++m) {
                cosmt.col(n) = cosnt.col(-m);
                sinmt.col(n) = -sinnt.col(-m);
                ++n;
            }
            for (int m = 0; m < l + 1; ++m) {
                cosmt.col(n) = cosnt.col(m);
                sinmt.col(n) = sinnt.col(m);
                ++n;
            }
        }
    }

    /* 
    Computes the dot product M . Rxy(theta).

    */
    template <typename T1, bool M_IS_ROW_VECTOR=(T1::RowsAtCompileTime==1)>
    inline void dotRxy (
        const MatrixBase<T1>& M, 
        const Scalar& inc,
        const Scalar& obl
    ) {
        
        // Shape checks
        size_t npts = M.rows();

        // Compute the Wigner matrices
        computeRxy(inc, obl);

        // Init result
        dotRxy_result.resize(npts, Ny);
        if (unlikely(npts == 0)) return;

        // Dot them in
        for (int l = 0; l < ydeg + 1; ++l) {
            dotRxy_result.block(0, l * l, npts, 2 * l + 1) =
                M.block(0, l * l, npts, 2 * l + 1) * Rxy[l];
        }

    }

    /* 
    Computes the gradient of the dot product M . Rxy(theta).

    */
    template <typename T1, bool M_IS_ROW_VECTOR=(T1::RowsAtCompileTime==1)>
    inline void dotRxy (
        const MatrixBase<T1>& M, 
        const Scalar& inc,
        const Scalar& obl,
        const Matrix<Scalar>& bMRxy
    ) {
        
        // Shape checks
        size_t npts = M.rows();

        // Compute the Wigner matrices
        computeRxy(inc, obl);

        // Init grads
        dotRxy_binc = 0.0;
        dotRxy_bobl = 0.0;
        dotRxy_bM.setZero(npts, Ny);
        if (unlikely(npts == 0)) return;

        // Dot them in
        // \todo: There must be a more efficient way of doing this.
        for (int l = 0; l < ydeg + 1; ++l) {

            // d / dinc & d / dobl
            dotRxy_binc += (M.block(0, l * l, npts, 2 * l + 1) * DRxyDinc[l])
                            .cwiseProduct(bMRxy.block(0, l * l, npts, 2 * l + 1))
                            .sum();
            dotRxy_bobl+= (M.block(0, l * l, npts, 2 * l + 1) * DRxyDobl[l])
                            .cwiseProduct(bMRxy.block(0, l * l, npts, 2 * l + 1))
                            .sum();

            // d / dM
            dotRxy_bM.block(0, l * l, npts, 2 * l + 1) =
                bMRxy.block(0, l * l, npts, 2 * l + 1) * Rxy[l].transpose();

        }

    }

    /* 
    Computes the dot product M . Rxy^T(theta).

    */
    template <typename T1, bool M_IS_ROW_VECTOR=(T1::RowsAtCompileTime==1)>
    inline void dotRxyT (
        const MatrixBase<T1>& M, 
        const Scalar& inc,
        const Scalar& obl
    ) {
        
        // Shape checks
        size_t npts = M.rows();

        // Compute the Wigner matrices
        computeRxy(inc, obl);

        // Init result
        dotRxyT_result.resize(npts, Ny);
        if (unlikely(npts == 0)) return;

        // Dot them in
        for (int l = 0; l < ydeg + 1; ++l) {
            dotRxyT_result.block(0, l * l, npts, 2 * l + 1) =
                M.block(0, l * l, npts, 2 * l + 1) * Rxy[l].transpose();
        }

    }

    /* 
    Computes the gradient of the dot product M . Rxy^T(theta).

    */
    template <typename T1, bool M_IS_ROW_VECTOR=(T1::RowsAtCompileTime==1)>
    inline void dotRxyT (
        const MatrixBase<T1>& M, 
        const Scalar& inc,
        const Scalar& obl,
        const Matrix<Scalar>& bMRxyT
    ) {
        
        // Shape checks
        size_t npts = M.rows();

        // Compute the Wigner matrices
        computeRxy(inc, obl);

        // Init grads
        dotRxyT_binc = 0.0;
        dotRxyT_bobl = 0.0;
        dotRxyT_bM.setZero(npts, M.cols());
        if (unlikely(npts == 0)) return;

        // Dot them in
        // TODO: There must be a more efficient way of doing this.
        for (int l = 0; l < ydeg + 1; ++l) {

            // d / dinc & d / dobl
            dotRxyT_binc += (M.block(0, l * l, npts, 2 * l + 1) * DRxyDinc[l].transpose())
                            .cwiseProduct(bMRxyT.block(0, l * l, npts, 2 * l + 1))
                            .sum();
            dotRxyT_bobl+= (M.block(0, l * l, npts, 2 * l + 1) * DRxyDobl[l].transpose())
                            .cwiseProduct(bMRxyT.block(0, l * l, npts, 2 * l + 1))
                            .sum();

            // d / dM
            dotRxyT_bM.block(0, l * l, npts, 2 * l + 1) =
                bMRxyT.block(0, l * l, npts, 2 * l + 1) * Rxy[l];

        }

    }

    /* 
    Computes the dot product M . Rz(theta).

    */
    template <typename T1, bool M_IS_ROW_VECTOR=(T1::RowsAtCompileTime==1)>
    inline void dotRz (
        const MatrixBase<T1>& M, 
        const Vector<Scalar>& theta
    ) {
        
        // Shape checks
        size_t npts = theta.size();
        size_t Nr = M.cols();
        int degr = sqrt(Nr) - 1;

        // Compute the sin & cos matrices
        computeRz(theta);

        // Init result
        dotRz_result.resize(npts, Nr);
        if (unlikely(npts == 0)) return;

        // Dot them in
        for (int l = 0; l < degr + 1; ++l) {
            for (int j = 0; j < 2 * l + 1; ++j) {
                if (M_IS_ROW_VECTOR) {
                    dotRz_result.col(l * l + j) = 
                        M(l * l + j) * cosmt.col(l * l + j) +
                        M(l * l + 2 * l - j) * sinmt.col(l * l + j);
                } else {
                    dotRz_result.col(l * l + j) = 
                        M.col(l * l + j).cwiseProduct(cosmt.col(l * l + j)) +
                        M.col(l * l + 2 * l - j).cwiseProduct(sinmt.col(l * l + j));
                }
            }
        }

    }

    /* 
    Computes the gradient of the dot product M . Rz(theta).

    */
    template <typename T1, bool M_IS_ROW_VECTOR=(T1::RowsAtCompileTime==1)>
    inline void dotRz (
        const MatrixBase<T1>& M, 
        const Vector<Scalar>& theta,
        const Matrix<Scalar>& bMRz
    ) {
        
        // Shape checks
        size_t npts = theta.size();
        size_t Nr = M.cols();
        int degr = sqrt(Nr) - 1;

        // Compute the sin & cos matrices
        computeRz(theta);

        // Init grads
        dotRz_btheta.setZero(npts);
        dotRz_bM.setZero(M.rows(), Nr);
        if (unlikely((npts == 0) || (M.rows() == 0))) return;

        // Dot the sines and cosines in
        for (int l = 0; l < degr + 1; ++l) {
            for (int j = 0; j < 2 * l + 1; ++j) {

                // Pre-compute these guys
                tmp_c = bMRz.col(l * l + j).cwiseProduct(cosmt.col(l * l + j));
                tmp_s = bMRz.col(l * l + j).cwiseProduct(sinmt.col(l * l + j));

                // d / dtheta
                if (M_IS_ROW_VECTOR) {
                    dotRz_btheta += (j - l) * (
                            M(l * l + 2 * l - j) * tmp_c -
                            M(l * l + j) * tmp_s
                    );
                } else {
                    dotRz_btheta += (j - l) * (
                        M.col(l * l + 2 * l - j).cwiseProduct(tmp_c) -
                        M.col(l * l + j).cwiseProduct(tmp_s)
                    );
                }
                
                // d / dM
                if (M_IS_ROW_VECTOR) {
                    dotRz_bM(l * l + 2 * l - j) += tmp_s.sum();
                    dotRz_bM(l * l + j) += tmp_c.sum();
                } else {
                    dotRz_bM.col(l * l + 2 * l - j) += tmp_s;
                    dotRz_bM.col(l * l + j) += tmp_c;
                }

            }
        }

    }

};

} // namespace wigner
} // namespace starry

#endif