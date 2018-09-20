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

#include <cmath>
#include <Eigen/Core>
#include "utils.h"
#include "tables.h"

namespace rotation {

    using namespace utils;
    using std::abs;
    using std::max;

    /**
    Axis-angle rotation matrix, used to rotate Cartesian
    vectors around in 3D space.

    */
    template <typename T>
    Matrix<T> AxisAngle(const UnitVector<T>& u, const T& theta) {
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
    Rotation matrix class for the spherical harmonics.

    */
    template <class MapType>
    class Wigner {

        using T = typename MapType::Scalar;

        const int lmax;                                                         /**< Highest degree of the map */
        const int N;                                                            /**< Number of map coefficients */
        const int NW;                                                           /**< Number of wavelengths */
        const T tol;                                                            /**< Numerical tolerance used to prevent division-by-zero errors */

        // References to the base map and the rotation axis
        MapType& y;                                                             /**< Reference to the spherical harmonic map to be rotated */
        UnitVector<T>& axis;                                                    /**< Reference to the rotation axis */

        // Cached transforms
        T cache_costheta;                                                       /**< Last value of cos(theta) used */
        T cache_sintheta;                                                       /**< Last value of sin(theta) used */
        MapType cache_y;                                                        /**< Last value of the rotated map coefficients */

        // The actual Wigner matrices
        Matrix<T>* DZeta;                                                       /**< The complex Wigner matrix in the `zeta` frame */
        Matrix<T>* RZeta;                                                       /**< The real Wigner matrix in the `zeta` frame */
        Matrix<T>* RZetaInv;                                                    /**< The inverse of the real Wigner matrix in the `zeta` frame */

        // `zhat` rotation params
        Vector<T> cosnt;                                                        /**< Vector of cos(n theta) values */
        Vector<T> sinnt;                                                        /**< Vector of sin(n theta) values */
        MapType yrev;                                                           /**< Degree-wise reverse of the spherical harmonic map */

        // `zeta` transform params
        T cos_zeta, sin_zeta;                                                   /**< Angle between the axis of rotation and `zhat` */
        UnitVector<T> axis_zeta;                                                /**< Axis of rotation to align the rotation axis with `zhat` */
        MapType y_zeta;                                                         /**< The base map in the `zeta` frame */
        MapType y_zeta_rot;                                                     /**< The base map in the `zeta` frame after a `zhat` rotation */

        // Methods
        inline void rotar(T& c1, T& s1, T& c2, T& s2, T& c3, T& s3);
        inline void dlmn(int l, T& s1, T& c1, T& c2, T& tgbet2, T& s3, T& c3);
        inline void computeZeta(const UnitVector<T>& axis, const T& costheta,
                                const T& sintheta);
        inline void rotate(const T& costheta, const T& sintheta);

    public:

        Matrix<T>* R;                                                           /**< The full rotation matrix for real spherical harmonics */
        Matrix<T>* dRdtheta;                                                    /**< The derivative of the rotation matrix with respect to theta */
        Vector<T> cosmt;                                                        /**< Vector of cos(m theta) values */
        Vector<T> sinmt;                                                        /**< Vector of sin(m theta) values */

        // These methods are accessed by the `Map` class
        inline void update();
        inline void rotate(const T& costheta, const T& sintheta,
                           MapType& yout);
        inline void compute(const T& costheta, const T& sintheta);
        inline void rotatez(const T& costheta, const T& sintheta,
                            const MapType& yin, MapType& yout);

        // Constructor: allocate the matrices
        Wigner(int lmax, int nwav, MapType& y, UnitVector<T>& axis) :
            lmax(lmax), N((lmax + 1) * (lmax + 1)), NW(nwav),
            tol(10 * mach_eps<T>()), y(y), axis(axis) {

            // Allocate the Wigner matrices
            DZeta = new Matrix<T>[lmax + 1];
            RZeta = new Matrix<T>[lmax + 1];
            RZetaInv = new Matrix<T>[lmax + 1];
            R = new Matrix<T>[lmax + 1];
            dRdtheta = new Matrix<T>[lmax + 1];
            for (int l = 0; l < lmax + 1; l++) {
                DZeta[l].resize(2 * l + 1, 2 * l + 1);
                RZeta[l].resize(2 * l + 1, 2 * l + 1);
                RZetaInv[l].resize(2 * l + 1, 2 * l + 1);
                R[l].resize(2 * l + 1, 2 * l + 1);
                dRdtheta[l].resize(2 * l + 1, 2 * l + 1);
            }

            // Initialize our z rotation vectors
            cosnt.resize(max(2, lmax + 1));
            cosnt(0) = 1.0;
            sinnt.resize(max(2, lmax + 1));
            sinnt(0) = 0.0;
            cosmt.resize(N);
            sinmt.resize(N);
            resize(yrev, N, NW);

            // The base map in the `zeta` frame
            resize(y_zeta, N, NW);

            // The cached rotated map
            resize(cache_y, N, NW);

        }

        // Destructor: free the matrices
        ~Wigner() {
            delete [] DZeta;
            delete [] RZeta;
            delete [] RZetaInv;
            delete [] R;
            delete [] dRdtheta;
        }

    };

    /**
    Rotate the base map given `costheta` and `sintheta`

    */
    template <class MapType>
    inline void Wigner<MapType>::rotate(const typename MapType::Scalar& costheta,
                                        const typename MapType::Scalar& sintheta,
                                        MapType& yout) {

        // Return the cached result?
        if ((costheta == cache_costheta) && (sintheta == cache_sintheta)) {
            yout = cache_y;
            return;
        }

        // Rotate `yzeta` about `zhat` and store in `yzeta_rot`;
        rotatez(costheta, sintheta, y_zeta, y_zeta_rot);

        // Rotate out of the `zeta` frame
        for (int l = 0; l < lmax + 1; l++) {
            cache_y.block(l * l, 0, 2 * l + 1, NW) =
                RZetaInv[l] * y_zeta_rot.block(l * l, 0, 2 * l + 1, NW);
        }

        // Export the result and cache the angles
        yout = cache_y;
        cache_costheta = costheta;
        cache_sintheta = sintheta;

    }

    /**
    Explicitly compute the full rotation matrix and its derivative.
    The full rotation matrix is

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

        dR/dtheta = RZetaInv . dRz/dtheta . RZeta

    where dRz/dtheta has the form

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
    inline void Wigner<MapType>::compute(const typename MapType::Scalar& costheta,
                                         const typename MapType::Scalar& sintheta) {

        // Compute the cos and sin vectors for the zhat rotation
        cosnt(1) = costheta;
        sinnt(1) = sintheta;
        for (int n = 2; n < lmax + 1; n++) {
            cosnt(n) = 2.0 * cosnt(n - 1) * cosnt(1) - cosnt(n - 2);
            sinnt(n) = 2.0 * sinnt(n - 1) * cosnt(1) - sinnt(n - 2);
        }
        int n = 0;
        for (int l = 0; l < lmax + 1; l++) {
            for (int m = -l; m < 0; m++) {
                cosmt(n) = cosnt(-m);
                sinmt(n) = -sinnt(-m);
                n++;
            }
            for (int m = 0; m < l + 1; m++) {
                cosmt(n) = cosnt(m);
                sinmt(n) = sinnt(m);
                n++;
            }
        }

        // Now compute the full rotation matrix
        int m;
        for (int l = 0; l < lmax + 1; l++) {
            for (int j = 0; j < 2 * l + 1; j++) {
                m = j - l;
                R[l].col(j) = RZetaInv[l].col(j) * cosmt(l * l + j) +
                              RZetaInv[l].col(2 * l - j) * sinmt(l * l + j);
                dRdtheta[l].col(j) = RZetaInv[l].col(2 * l - j) * m * cosmt(l * l + j) -
                                     RZetaInv[l].col(j) * m * sinmt(l * l + j);
            }
            R[l] = R[l] * RZeta[l];
            dRdtheta[l] = dRdtheta[l] * RZeta[l];
        }

    }

    /**
    Update the zeta rotation matrix and the base map
    in the zeta frame whenever the map coeffs or the axis change.

    */
    template <class MapType>
    inline void Wigner<MapType>::update() {

        // Compute the rotation transformation into and out of the `zeta` frame
        cos_zeta = axis(2);
        sin_zeta = sqrt(1 - axis(2) * axis(2));
        typename MapType::Scalar norm = sqrt(axis(0) * axis(0) + axis(1) * axis(1));
        if (abs(norm) < tol) {
            // The rotation axis is zhat, so our zeta transform
            // is just the identity matrix.
            for (int l = 0; l < lmax + 1; l++) {
               if (axis(2) > 0) {
                   RZeta[l] = Matrix<T>::Identity(2 * l + 1, 2 * l + 1);
                   RZetaInv[l] = Matrix<T>::Identity(2 * l + 1, 2 * l + 1);
               } else {
                   RZeta[l] = -Matrix<T>::Identity(2 * l + 1, 2 * l + 1);
                   RZetaInv[l] = -Matrix<T>::Identity(2 * l + 1, 2 * l + 1);
               }
            }
        } else {
            // We need to compute the actual Wigner matrices
            axis_zeta(0) = axis(1) / norm;
            axis_zeta(1) = -axis(0) / norm;
            axis_zeta(2) = 0;
            computeZeta(axis_zeta, cos_zeta, sin_zeta);
        }

        // Update the map in the `zeta` frame
        for (int l = 0; l < lmax + 1; l++) {
            y_zeta.block(l * l, 0, 2 * l + 1, NW) =
                RZeta[l] * y.block(l * l, 0, 2 * l + 1, NW);
        }

        // Reset the cache
        cache_costheta = NAN;
        cache_sintheta = NAN;

    }

    /**
    Perform a fast rotation about the z axis, skipping the Wigner matrix computation.
    See https://github.com/rodluger/starry/issues/137#issuecomment-405975092

    */
    template <class MapType>
    inline void Wigner<MapType>::rotatez(const typename MapType::Scalar& costheta,
                                         const typename MapType::Scalar& sintheta,
                                         const MapType& yin, MapType& yout) {
        cosnt(1) = costheta;
        sinnt(1) = sintheta;
        for (int n = 2; n < lmax + 1; n++) {
            cosnt(n) = 2.0 * cosnt(n - 1) * cosnt(1) - cosnt(n - 2);
            sinnt(n) = 2.0 * sinnt(n - 1) * cosnt(1) - sinnt(n - 2);
        }
        int n = 0;
        for (int l = 0; l < lmax + 1; l++) {
            for (int m = -l; m < 0; m++) {
                cosmt(n) = cosnt(-m);
                sinmt(n) = -sinnt(-m);
                yrev.row(n) = yin.row(l * l + l - m);
                n++;
            }
            for (int m = 0; m < l + 1; m++) {
                cosmt(n) = cosnt(m);
                sinmt(n) = sinnt(m);
                yrev.row(n) = yin.row(l * l + l - m);
                n++;
            }
        }

        // TODO. This may be faster if MapType is Vector:
        // yout = cosmt.cwiseProduct(yin) - sinmt.cwiseProduct(yrev);
        yout = (yin.transpose().array().rowwise() * cosmt.array().transpose() -
                yrev.transpose().array().rowwise() * sinmt.array().transpose()).transpose();

    }

    /**
    Compute the axis-angle rotation matrix for real spherical harmonics up to order lmax.

    */
    template <class MapType>
    inline void Wigner<MapType>::computeZeta(const UnitVector<typename MapType::Scalar>& axis,
                                             const typename MapType::Scalar& costheta,
                                             const typename MapType::Scalar& sintheta) {

        using T = typename MapType::Scalar;

        // Trivial case
        if (lmax == 0) {
            RZeta[0](0, 0) = 1;
            RZetaInv[0](0, 0) = 1;
            return;
        }

        // Construct the axis-angle rotation matrix R_A
        T RA01 = axis(0) * axis(1) * (1 - costheta) - axis(2) * sintheta;
        T RA02 = axis(0) * axis(2) * (1 - costheta) + axis(1) * sintheta;
        T RA11 = costheta + axis(1) * axis(1) * (1 - costheta);
        T RA12 = axis(1) * axis(2) * (1 - costheta) - axis(0) * sintheta;
        T RA20 = axis(2) * axis(0) * (1 - costheta) - axis(1) * sintheta;
        T RA21 = axis(2) * axis(1) * (1 - costheta) + axis(0) * sintheta;
        T RA22 = costheta + axis(2) * axis(2) * (1 - costheta);

        // Determine the Euler angles
        T cosalpha, sinalpha, cosbeta, sinbeta, cosgamma, singamma;
        T norm1, norm2;
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

        // Call the eulerian rotation function
        rotar(cosalpha, sinalpha, cosbeta, sinbeta, cosgamma, singamma);

        // Set the inverse transform
        for (int l = 0; l < lmax + 1; l++)
            RZetaInv[l] = RZeta[l].transpose();

        return;

    }

    /**
    Compute the eulerian rotation matrix for real spherical
    harmonics up to order lmax.

    */
    template <class MapType>
    inline void Wigner<MapType>::rotar(typename MapType::Scalar& c1,
                                       typename MapType::Scalar& s1,
                                       typename MapType::Scalar& c2,
                                       typename MapType::Scalar& s2,
                                       typename MapType::Scalar& c3,
                                       typename MapType::Scalar& s3) {

        using T = typename MapType::Scalar;
        T cosag, COSAMG, sinag, SINAMG, tgbet2;

        // Compute the initial matrices D0, R0, D1 and R1
        DZeta[0](0, 0) = 1.;
        RZeta[0](0, 0) = 1.;
        DZeta[1](2, 2) = 0.5 * (1. + c2);
        DZeta[1](2, 1) = -s2 * tables::invsqrt_int<T>(2);
        DZeta[1](2, 0) = 0.5 * (1. - c2);
        DZeta[1](1, 2) = -DZeta[1](2, 1);
        DZeta[1](1, 1) = DZeta[1](2, 2) - DZeta[1](2, 0);
        DZeta[1](1, 0) = DZeta[1](2, 1);
        DZeta[1](0, 2) = DZeta[1](2, 0);
        DZeta[1](0, 1) = DZeta[1](1, 2);
        DZeta[1](0, 0) = DZeta[1](2, 2);
        cosag = c1 * c3 - s1 * s3;
        COSAMG = c1 * c3 + s1 * s3;
        sinag = s1 * c3 + c1 * s3;
        SINAMG = s1 * c3 - c1 * s3;
        RZeta[1](1, 1) = DZeta[1](1, 1);
        RZeta[1](2, 1) = tables::sqrt_int<T>(2) * DZeta[1](1, 2) * c1;
        RZeta[1](0, 1) = tables::sqrt_int<T>(2) * DZeta[1](1, 2) * s1;
        RZeta[1](1, 2) = tables::sqrt_int<T>(2) * DZeta[1](2, 1) * c3;
        RZeta[1](1, 0) = -tables::sqrt_int<T>(2) * DZeta[1](2, 1) * s3;
        RZeta[1](2, 2) = DZeta[1](2, 2) * cosag - DZeta[1](2, 0) * COSAMG;
        RZeta[1](2, 0) = -DZeta[1](2, 2) * sinag - DZeta[1](2, 0) * SINAMG;
        RZeta[1](0, 2) = DZeta[1](2, 2) * sinag - DZeta[1](2, 0) * SINAMG;
        RZeta[1](0, 0) = DZeta[1](2, 2) * cosag + DZeta[1](2, 0) * COSAMG;

        // The remaining matrices are calculated using symmetry and
        // and recurrence relations
        if (abs(s2) < tol)
            tgbet2 = s2; // = 0
        else
            tgbet2 = (1. - c2) / s2;

        for (int l=2; l<lmax+1; l++)
            dlmn(l, s1, c1, c2, tgbet2, s3, c3);

        return;

    }

    /**
    Compute the Wigner D matrices.

    */
    template <class MapType>
    inline void Wigner<MapType>::dlmn(int l, typename MapType::Scalar& s1,
                                             typename MapType::Scalar& c1,
                                             typename MapType::Scalar& c2,
                                             typename MapType::Scalar& tgbet2,
                                             typename MapType::Scalar& s3,
                                             typename MapType::Scalar& c3) {

        using T = typename MapType::Scalar;
        int iinf = 1 - l;
        int isup = -iinf;
        int m, mp;
        int al, al1, tal1, amp, laux, lbux, am, lauz, lbuz;
        int sign;
        T ali, auz, aux, cux, fact, term, cuz;
        T cosaux, cosmal, sinmal, cosag, sinag, cosagm, sinagm, cosmga, sinmga;
        T d1, d2;

        // Compute the DZeta[l;m',m) matrix.
        // First row by recurrence (Eq. 19 and 20 in Alvarez Collado et al.)
        DZeta[l](2 * l, 2 * l) = DZeta[l - 1](isup + l - 1, isup + l - 1) * (1. + c2) / 2.;
        DZeta[l](2 * l, 0) = DZeta[l - 1](isup + l - 1, -isup + l - 1) * (1. - c2) / 2.;
        for (m=isup; m>iinf-1; m--)
            DZeta[l](2 * l, m + l) = -tgbet2 * tables::sqrt_int<T>(l + m + 1) *
                                  tables::invsqrt_int<T>(l - m) * DZeta[l](2 * l, m + 1 + l);

        // The rows of the upper quarter triangle of the DZeta[l;m',m) matrix
        // (Eq. 21 in Alvarez Collado et al.)
        al = l;
        al1 = al - 1;
        tal1 = al + al1;
        ali = (1. / (T) al1);
        cosaux = c2 * al * al1;
        for (mp=l-1; mp>-1; mp--) {
            amp = mp;
            laux = l + mp;
            lbux = l - mp;
            aux = tables::invsqrt_int<T>(laux * lbux) * ali;
            cux = tables::sqrt_int<T>((laux - 1) * (lbux - 1)) * al;
            for (m=isup; m>iinf-1; m--) {
                am = m;
                lauz = l + m;
                lbuz = l - m;
                auz = tables::invsqrt_int<T>(lauz * lbuz);
                fact = aux * auz;
                term = tal1 * (cosaux - am * amp) * DZeta[l - 1](mp + l - 1, m + l - 1);
                if ((lbuz != 1) && (lbux != 1)) {
                    cuz = tables::sqrt_int<T>((lauz - 1) * (lbuz - 1));
                    term = term - DZeta[l - 2](mp + l - 2, m + l - 2) * cux * cuz;
                }
                DZeta[l](mp + l, m + l) = fact * term;
            }
            iinf = iinf + 1;
            isup = isup - 1;
        }

        // The remaining elements of the DZeta[l;m',m) matrix are calculated
        // using the corresponding symmetry relations:
        // reflection ---> ((-1)**(m-m')) DZeta[l;m,m') = DZeta[l;m',m), m'<=m
        // inversion ---> ((-1)**(m-m')) DZeta[l;-m',-m) = DZeta[l;m',m)

        // Reflection
        sign = 1;
        iinf = -l;
        isup = l - 1;
        for (m=l; m>0; m--) {
            for (mp=iinf; mp<isup+1; mp++) {
                DZeta[l](mp + l, m + l) = sign * DZeta[l](m + l, mp + l);
                sign = -sign;
            }
            iinf = iinf + 1;
            isup = isup - 1;
        }

        // Inversion
        iinf = -l;
        isup = iinf;
        for (m=l-1; m>-(l+1); m--) {
            sign = -1;
            for (mp=isup; mp>iinf-1; mp--) {
                DZeta[l](mp + l, m + l) = sign * DZeta[l](-mp + l, -m + l);
                sign = -sign;
            }
            isup = isup + 1;
        }

        // Compute the real rotation matrices R from the complex ones D
        RZeta[l](0 + l, 0 + l) = DZeta[l](0 + l, 0 + l);
        cosmal = c1;
        sinmal = s1;
        sign = -1;
        for (mp=1; mp<l+1; mp++) {
            cosmga = c3;
            sinmga = s3;
            aux = tables::sqrt_int<T>(2) * DZeta[l](0 + l, mp + l);
            RZeta[l](mp + l, 0 + l) = aux * cosmal;
            RZeta[l](-mp + l, 0 + l) = aux * sinmal;
            for (m=1; m<l+1; m++) {
                aux = tables::sqrt_int<T>(2) * DZeta[l](m + l, 0 + l);
                RZeta[l](l, m + l) = aux * cosmga;
                RZeta[l](l, -m + l) = -aux * sinmga;
                d1 = DZeta[l](-mp + l, -m + l);
                d2 = sign * DZeta[l](mp + l, -m + l);
                cosag = cosmal * cosmga - sinmal * sinmga;
                cosagm = cosmal * cosmga + sinmal * sinmga;
                sinag = sinmal * cosmga + cosmal * sinmga;
                sinagm = sinmal * cosmga - cosmal * sinmga;
                RZeta[l](mp + l, m + l) = d1 * cosag + d2 * cosagm;
                RZeta[l](mp + l, -m + l) = -d1 * sinag + d2 * sinagm;
                RZeta[l](-mp + l, m + l) = d1 * sinag + d2 * sinagm;
                RZeta[l](-mp + l, -m + l) = d1 * cosag - d2 * cosagm;
                aux = cosmga * c3 - sinmga * s3;
                sinmga = sinmga * c3 + cosmga * s3;
                cosmga = aux;
            }
            sign = -sign;
            aux = cosmal * c1 - sinmal * s1;
            sinmal = sinmal * c1 + cosmal * s1;
            cosmal = aux;
        }

        return;
    }

} // namespace rotation

#endif
