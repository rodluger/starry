/**
Miscellaneous stuff used throughout the code.

*/

#ifndef _STARRY_UTILS_H_
#define _STARRY_UTILS_H_

#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/constants/constants.hpp>
#include <iostream>
#include <limits>
#include "errors.h"

namespace utils {

    //! Type-independent PI
    using boost::math::constants::pi;
    //! Type-independent sqrt of PI
    using boost::math::constants::root_pi;

    //! Multiprecision datatype backend
    typedef boost::multiprecision::cpp_dec_float<STARRY_NMULTI> mp_backend;
    //! Multiprecision datatype
    typedef boost::multiprecision::number<mp_backend, boost::multiprecision::et_off> Multi;

    //! A generic row vector
    template <typename T>
    using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    template <typename T>
    //! A generic column vector
    using VectorT = Eigen::Matrix<T, 1, Eigen::Dynamic>;
    template <typename T>
    //! A generic matrix
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    template <typename T>
    //! A generic 3-component unit vector
    using UnitVector = Eigen::Matrix<T, 3, 1>;

    //! A custom AutoDiffScalar type
    template <typename T, int N>
    using ADScalar = Eigen::AutoDiffScalar<Eigen::Matrix<T, N, 1>>;

    // Some useful unit vectors
    static const UnitVector<double> xhat_double({1, 0, 0});
    static const UnitVector<double> yhat_double({0, 1, 0});
    static const UnitVector<double> zhat_double({0, 0, 1});
    //! Unit vector in the xhat direction
    template <typename T> inline UnitVector<T> xhat(){ return xhat_double.template cast<T>(); }
    //! Unit vector in the yhat direction
    template <typename T> inline UnitVector<T> yhat(){ return yhat_double.template cast<T>(); }
    //! Unit vector in the zhat direction
    template <typename T> inline UnitVector<T> zhat(){ return zhat_double.template cast<T>(); }


    // Below we define the machine precision for an arbitrary type.
    // We need to be careful with AutoDiffScalar specialization.
    // See https://stackoverflow.com/a/36209847

    //! Tag forwarding hack
    template<class T> struct tag{};
    //! Machine precision for current type
    template<class T> T mach_eps(tag<T>) { return std::numeric_limits<T>::epsilon(); }
    //! Machine precision for current type (AutoDiffScalar specialization)
    template<class T> Eigen::AutoDiffScalar<T> mach_eps(tag<Eigen::AutoDiffScalar<T>>) {
        return std::numeric_limits<typename T::Scalar>::epsilon();
    }
    //! Machine precision for current type
    template<class T> T mach_eps() { return mach_eps(tag<T>()); }

    /**
    Check if a number is even (or doubly, triply, quadruply... even)

    */
    inline bool is_even(int n, int ntimes=1) {
        for (int i = 0; i < ntimes; i++) {
            if ((n % 2) != 0) return false;
            n /= 2;
        }
        return true;
    }

}; // namespace utils

#endif
