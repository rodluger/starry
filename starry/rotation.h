/**
Spherical harmonic rotation matrices. These are adapted from the Fortran
code of

    Alvarez Collado  et al. (1989) "Rotation of real spherical harmonics".
    Computer Physics Communications 52, 3.
    https://doi.org/10.1016/0010-4655(89)90107-0

who computed the Euleriean rotation matrices for real spherical harmonics
from the Wigner-D matrices for complex spherical harmonics.
*/

#ifndef _STARRY_ROT_H_
#define _STARRY_ROT_H_

#include <cmath>
#include <Eigen/Core>
#include "constants.h"
#include "utils.h"
#include "tables.h"

namespace rotation {

    using std::abs;

    /**
    Axis-angle rotation matrix, used to rotate vectors around.

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
    Compute the Wigner D matrices.

    */
    template <typename T>
    void dlmn(int l, T& s1, T& c1, T& c2, T& tgbet2, T& s3, T& c3, Matrix<T>* D, Matrix<T>* R) {
        int iinf = 1 - l;
        int isup = -iinf;
        int m, mp;
        int al, al1, tal1, amp, laux, lbux, am, lauz, lbuz;
        int sign;
        T ali, auz, aux, cux, fact, term, cuz;
        T cosaux, cosmal, sinmal, cosag, sinag, cosagm, sinagm, cosmga, sinmga;
        T d1, d2;

        // Compute the D[l;m',m) matrix.
        // First row by recurrence (Eq. 19 and 20 in Alvarez Collado et al.)
        D[l](2 * l, 2 * l) = D[l - 1](isup + l - 1, isup + l - 1) * (1. + c2) / 2.;
        D[l](2 * l, 0) = D[l - 1](isup + l - 1, -isup + l - 1) * (1. - c2) / 2.;
        for (m=isup; m>iinf-1; m--)
            D[l](2 * l, m + l) = -tgbet2 * tables::sqrt_int<T>(l + m + 1) *
                                  tables::invsqrt_int<T>(l - m) * D[l](2 * l, m + 1 + l);

        // The rows of the upper quarter triangle of the D[l;m',m) matrix
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
            aux = tables::invsqrt_int<T>(laux) * tables::invsqrt_int<T>(lbux) * ali;
            cux = tables::sqrt_int<T>(laux - 1) * tables::sqrt_int<T>(lbux - 1) * al;
            for (m=isup; m>iinf-1; m--) {
                am = m;
                lauz = l + m;
                lbuz = l - m;
                auz = tables::invsqrt_int<T>(lauz) * tables::invsqrt_int<T>(lbuz);
                fact = aux * auz;
                term = tal1 * (cosaux - am * amp) * D[l - 1](mp + l - 1, m + l - 1);
                if ((lbuz != 1) && (lbux != 1)) {
                    cuz = tables::sqrt_int<T>(lauz - 1) * tables::sqrt_int<T>(lbuz - 1);
                    term = term - D[l - 2](mp + l - 2, m + l - 2) * cux * cuz;
                }
                D[l](mp + l, m + l) = fact * term;
            }
            iinf = iinf + 1;
            isup = isup - 1;
        }

        // The remaining elements of the D[l;m',m) matrix are calculated
        // using the corresponding symmetry relations:
        // reflection ---> ((-1)**(m-m')) D[l;m,m') = D[l;m',m), m'<=m
        // inversion ---> ((-1)**(m-m')) D[l;-m',-m) = D[l;m',m)

        // Reflection
        sign = 1;
        iinf = -l;
        isup = l - 1;
        for (m=l; m>0; m--) {
            for (mp=iinf; mp<isup+1; mp++) {
                D[l](mp + l, m + l) = sign * D[l](m + l, mp + l);
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
                D[l](mp + l, m + l) = sign * D[l](-mp + l, -m + l);
                sign = -sign;
            }
            isup = isup + 1;
        }

        // Compute the real rotation matrices R from the complex ones D
        R[l](0 + l, 0 + l) = D[l](0 + l, 0 + l);
        cosmal = c1;
        sinmal = s1;
        sign = -1;
        for (mp=1; mp<l+1; mp++) {
            cosmga = c3;
            sinmga = s3;
            aux = tables::sqrt_int<T>(2) * D[l](0 + l, mp + l);
            R[l](mp + l, 0 + l) = aux * cosmal;
            R[l](-mp + l, 0 + l) = aux * sinmal;
            for (m=1; m<l+1; m++) {
                aux = tables::sqrt_int<T>(2) * D[l](m + l, 0 + l);
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
            sign = -sign;
            aux = cosmal * c1 - sinmal * s1;
            sinmal = sinmal * c1 + cosmal * s1;
            cosmal = aux;
        }

        return;
    }

    /**
    Compute the eulerian rotation matrix for real spherical
    harmonics up to order lmax.

    */
    template <typename T>
    void rotar(int lmax, T& c1, T& s1, T& c2, T& s2, T& c3, T& s3, Matrix<T>* D, Matrix<T>* R, T tol=10 * mach_eps<T>()) {
        T cosag, COSAMG, sinag, SINAMG, tgbet2;

        // Compute the initial matrices D0, R0, D1 and R1
        D[0](0, 0) = 1.;
        R[0](0, 0) = 1.;
        D[1](2, 2) = (1. + c2) / 2.;
        D[1](2, 1) = -s2 / tables::sqrt_int<T>(2);
        D[1](2, 0) = (1. - c2) / 2.;
        D[1](1, 2) = -D[1](2, 1);
        D[1](1, 1) = D[1](2, 2) - D[1](2, 0);
        D[1](1, 0) = D[1](2, 1);
        D[1](0, 2) = D[1](2, 0);
        D[1](0, 1) = D[1](1, 2);
        D[1](0, 0) = D[1](2, 2);
        cosag = c1 * c3 - s1 * s3;
        COSAMG = c1 * c3 + s1 * s3;
        sinag = s1 * c3 + c1 * s3;
        SINAMG = s1 * c3 - c1 * s3;
        R[1](1, 1) = D[1](1, 1);
        R[1](2, 1) = tables::sqrt_int<T>(2) * D[1](1, 2) * c1;
        R[1](0, 1) = tables::sqrt_int<T>(2) * D[1](1, 2) * s1;
        R[1](1, 2) = tables::sqrt_int<T>(2) * D[1](2, 1) * c3;
        R[1](1, 0) = -tables::sqrt_int<T>(2) * D[1](2, 1) * s3;
        R[1](2, 2) = D[1](2, 2) * cosag - D[1](2, 0) * COSAMG;
        R[1](2, 0) = -D[1](2, 2) * sinag - D[1](2, 0) * SINAMG;
        R[1](0, 2) = D[1](2, 2) * sinag - D[1](2, 0) * SINAMG;
        R[1](0, 0) = D[1](2, 2) * cosag + D[1](2, 0) * COSAMG;

        // The remaining matrices are calculated using symmetry and
        // and recurrence relations
        if (abs(s2) < tol)
            tgbet2 = s2; // = 0
        else
            tgbet2 = (1. - c2) / s2;

        for (int l=2; l<lmax+1; l++)
            dlmn(l, s1, c1, c2, tgbet2, s3, c3, D, R);

        return;

    }

    /**
    Compute the axis-angle rotation matrix for real spherical
    harmonics up to order lmax.

    */
    template <typename T>
    void computeR(int lmax, const Eigen::Matrix<T, 3, 1>& axis, const T& costheta, const T& sintheta, Matrix<T>* D, Matrix<T>* R, T tol=10 * mach_eps<T>()) {

        // Trivial case
        if (lmax == 0) {
            R[0](0, 0) = 1;
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
        rotar(lmax, cosalpha, sinalpha, cosbeta,
              sinbeta, cosgamma, singamma, D, R, tol);

        return;

    }

    // Rotation matrix class
    template <class T>
    class Wigner {

        int lmax;

    public:

        Matrix<T>* Complex;
        Matrix<T>* Real;

        // Constructor: allocate the matrices
        Wigner(int lmax) : lmax(lmax) {

            Complex = new Matrix<T>[lmax + 1];
            Real = new Matrix<T>[lmax + 1];
            for (int l = 0; l < lmax + 1; l++) {
                Complex[l].resize(2 * l + 1, 2 * l + 1);
                Real[l].resize(2 * l + 1, 2 * l + 1);
            }

        }

        // Destructor: free the matrices
        ~Wigner() {
            delete [] Complex;
            delete [] Real;
        }

    };

}; // namespace rotation

#endif
