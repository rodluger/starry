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
Rotation matrix class for the spherical harmonics.

*/
template <class Scalar>
class Wigner {

protected:

    Matrix<Scalar> cosmt;                                                      /**< Matrix of cos(m theta) values */
    Matrix<Scalar> sinmt;                                                      /**< Matrix of sin(m theta) values */
    Matrix<Scalar> cosnt;                                                      /**< Matrix of cos(n theta) values */
    Matrix<Scalar> sinnt;                                                      /**< Matrix of sin(n theta) values */

public:

    const int ydeg;
    const int Ny;                                                              /**< Number of spherical harmonic `(l, m)` coefficients */
    const int udeg;
    const int Nu;                                                              /**< Number of limb darkening coefficients */
    const int fdeg;
    const int Nf;                                                              /**< Number of filter `(l, m)` coefficients */
    const int deg;
    const int N;

    // Z rotation
    Vector<Scalar> btheta;
    Matrix<Scalar> bM;
    Vector<Scalar> tmp_c, tmp_s;

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
        N((deg + 1) * (deg + 1))
    {

        //

    }


    /* 
    Computes the dot product ARz = A . Rz.

    */
    template <bool GRADIENT=false, int StorageOrder=ColMajor>
    inline void dotRz (
        const Matrix<Scalar, StorageOrder>& M, 
        const Vector<Scalar>& costheta,
        const Vector<Scalar>& sintheta,
        Matrix<Scalar, StorageOrder>& MRz,
        const Matrix<Scalar, StorageOrder>& bMRz=Matrix<Scalar, StorageOrder>()
    ) {
        // Length of timeseries
        size_t npts = costheta.size();

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
            cosnt.col(n) = 2.0 * cosnt.col(n - 1).cwiseProduct(cosnt.col(1)) - cosnt.col(n - 2);
            sinnt.col(n) = 2.0 * sinnt.col(n - 1).cwiseProduct(cosnt.col(1)) - sinnt.col(n - 2);
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

        // Init grads
        if (GRADIENT) {
            btheta.setZero(costheta.size());
            bM.setZero(npts, M.cols());
        }

        // Dot them in
        for (int l = 0; l < deg + 1; ++l) {
            for (int j = 0; j < 2 * l + 1; ++j) {
                MRz.col(l * l + j) = 
                    M.col(l * l + j).cwiseProduct(cosmt.col(l * l + j)) +
                    M.col(l * l + 2 * l - j).cwiseProduct(sinmt.col(l * l + j));
                if (GRADIENT) {
                    tmp_c = bMRz.col(l * l + j).cwiseProduct(cosmt.col(l * l + j));
                    tmp_s = bMRz.col(l * l + j).cwiseProduct(sinmt.col(l * l + j));

                    // d/dtheta
                    btheta += (j - l) * (
                            M.col(l * l + 2 * l - j).cwiseProduct(tmp_c) -
                            M.col(l * l + j).cwiseProduct(tmp_s)
                        );
                    
                    // d/dM
                    bM.col(l * l + 2 * l - j) += tmp_s;
                    bM.col(l * l + j) += tmp_c;

                }
            }
        }
    }

};

} // namespace wigner
} // namespace starry

#endif