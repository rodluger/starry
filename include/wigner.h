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

    const int ydeg;
    const int Ny;                                                              /**< Number of spherical harmonic `(l, m)` coefficients */
    const int udeg;
    const int Nu;                                                              /**< Number of limb darkening coefficients */
    const int fdeg;
    const int Nf;                                                              /**< Number of filter `(l, m)` coefficients */
    const int deg;
    const int N;

    Matrix<Scalar> cosmt;                                                      /**< Matrix of cos(m theta) values */
    Matrix<Scalar> sinmt;                                                      /**< Matrix of sin(m theta) values */
    Matrix<Scalar> cosnt;                                                      /**< Matrix of cos(n theta) values */
    Matrix<Scalar> sinnt;                                                      /**< Matrix of sin(n theta) values */

    Vector<Scalar> tmp_c, tmp_s;
    Vector<Scalar> theta_Rz_cache, theta_rad, costheta, sintheta;

    Scalar radian, degree;

public:

    // Z rotation
    Matrix<Scalar> dotRz_result;
    Vector<Scalar> dotRz_btheta;
    Matrix<Scalar> dotRz_bM;

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
        theta_Rz_cache(0)
    {

        //
        radian = (pi<Scalar>() / 180.0);
        degree = 1.0 / radian;

    }

    inline void computeRz(
        const Vector<Scalar>& theta
    ) {

        // Length of timeseries
        size_t npts = theta.size();

        // Check the cache
        if ((npts == size_t(theta_Rz_cache.size())) && (theta == theta_Rz_cache)) {
            return;
        }
        theta_Rz_cache = theta;

        // Compute sin & cos
        theta_rad = theta * radian;
        costheta = theta_rad.array().cos();
        sintheta = theta_rad.array().sin();

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
    Computes the dot product M . Rz(theta).

    */
    inline void dotRz (
        const Matrix<Scalar>& M, 
        const Vector<Scalar>& theta
    ) {
        
        // Shape checks
        size_t npts = theta.size();
        CHECK_SHAPE(M, npts, N);

        // Compute the sin & cos matrices
        computeRz(theta);

        // Init result
        dotRz_result.resize(npts, N);

        // Dot them in
        for (int l = 0; l < deg + 1; ++l) {
            for (int j = 0; j < 2 * l + 1; ++j) {
                dotRz_result.col(l * l + j) = 
                    M.col(l * l + j).cwiseProduct(cosmt.col(l * l + j)) +
                    M.col(l * l + 2 * l - j).cwiseProduct(sinmt.col(l * l + j));
            }
        }

    }

    /* 
    Computes the gradient of the dot product M . Rz(theta).

    */
    inline void dotRz (
        const Matrix<Scalar>& M, 
        const Vector<Scalar>& theta,
        const Matrix<Scalar>& bMRz
    ) {
        
        // Shape checks
        size_t npts = theta.size();
        CHECK_SHAPE(M, npts, N);

        // Compute the sin & cos matrices
        computeRz(theta);

        // Init grads
        dotRz_btheta.setZero(npts);
        dotRz_bM.setZero(npts, N);

        // Dot the sines and cosines in
        for (int l = 0; l < deg + 1; ++l) {
            for (int j = 0; j < 2 * l + 1; ++j) {

                // Pre-compute these guys
                tmp_c = bMRz.col(l * l + j).cwiseProduct(cosmt.col(l * l + j));
                tmp_s = bMRz.col(l * l + j).cwiseProduct(sinmt.col(l * l + j));

                // d / dtheta
                dotRz_btheta += (j - l) * (
                        M.col(l * l + 2 * l - j).cwiseProduct(tmp_c) -
                        M.col(l * l + j).cwiseProduct(tmp_s)
                    );
                
                // d / dM
                dotRz_bM.col(l * l + 2 * l - j) += tmp_s;
                dotRz_bM.col(l * l + j) += tmp_c;

            }
        }

        // Unit change for theta
        dotRz_btheta *= radian;

    }

};

} // namespace wigner
} // namespace starry

#endif