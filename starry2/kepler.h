/**
Keplerian star/planet/moon system class.

*/

#ifndef _STARRY_ORBITAL_H_
#define _STARRY_ORBITAL_H_

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <string>
#include <vector>
#include "errors.h"
#include "maps.h"
#include "utils.h"
#include "rotation.h"

namespace units {

    const double DayToSeconds = 86400.0;

}; // namespace units

namespace kepler {

    using namespace utils;

    /**
    Compute the eccentric anomaly. Adapted from
    https://github.com/lkreidberg/batman/blob/master/c_src/_rsky.c

    */
    template <typename T>
    T EccentricAnomaly(T& M, T& ecc) {
        // Initial condition
        T E = M;
        T tol = 10 * mach_eps<T>();
        if (ecc > 0) {
            // Iterate
            for (int iter = 0; iter <= STARRY_KEPLER_MAX_ITER; iter++) {
                E = E - (E - ecc * sin(E) - M) / (1. - ecc * cos(E));
                if (abs(E - ecc * sin(E) - M) <= tol) return E;
            }
            // Didn't converge!
            throw errors::ConvergenceError("The Kepler solver did not converge.");
        }
        return E;
    }

    /**
    Manual override of the derivative of the eccentric anomaly

    */
    template <typename T>
    Eigen::AutoDiffScalar<T> EccentricAnomaly(const Eigen::AutoDiffScalar<T>& M,
        const Eigen::AutoDiffScalar<T>& ecc) {
        typename T::Scalar M_value = M.value(),
                           ecc_value = ecc.value(),
                           E_value = EccentricAnomaly(M_value, ecc_value),
                           cosE_value = cos(E_value),
                           sinE_value = sin(E_value),
                           norm1 = 1./ (1. - ecc_value * cosE_value),
                           norm2 = sinE_value * norm1;
        if (M.derivatives().size() && ecc.derivatives().size())
            return Eigen::AutoDiffScalar<T>(E_value,
                                            M.derivatives() * norm1 +
                                            ecc.derivatives() * norm2);
        else if (M.derivatives().size())
            return Eigen::AutoDiffScalar<T>(E_value, M.derivatives() * norm1);
        else if (ecc.derivatives().size())
            return Eigen::AutoDiffScalar<T>(E_value, ecc.derivatives() * norm2);
        else
            return Eigen::AutoDiffScalar<T>(E_value, M.derivatives());
    }

    /**
    Generic body class, a subclass of Map with added orbital features.

    */
    template <class T>
    class Body : public maps::Map<T> {

        protected:

            // Hide the Map's flux function from the user,
            // as we compute lightcurves via the System class
            Row<T> flux();

            // Shorthand for the scalar type (double, Multi)
            using S = Scalar<T>;

            S r;                                                                /**< Body radius in units of primary radius */
            S L;                                                                /**< Body luminosity in units of primary luminosity */
            S prot;                                                             /**< Body rotation period in seconds */
            S theta0;                                                           /**< Body initial rotation angle in radians */

            // Map attributes we need access to within this class
            using maps::Map<T>::lmax;
            using maps::Map<T>::N;
            using maps::Map<T>::nwav;

        public:

            explicit Body(S r=1, S L=1, S prot=0, S theta0=0,
                          int lmax=2, int nwav=1) :
                maps::Map<T>(lmax, nwav),
                r(r),
                L(L),
                prot(prot * units::DayToSeconds),
                theta0(theta0 * pi<S>()/ 180.) {

            }

    };

}; // namespace kepler

#endif
