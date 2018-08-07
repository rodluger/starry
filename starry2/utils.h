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

    //! Multiprecision datatype backend
    typedef boost::multiprecision::cpp_dec_float<STARRY_NMULTI> mp_backend;

    //! Multiprecision datatype
    typedef boost::multiprecision::number<mp_backend, boost::multiprecision::et_off> Multi;

    //! A generic row vector
    template <typename T>
    using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    //! A generic column vector
    template <typename T>
    using VectorT = Eigen::Matrix<T, 1, Eigen::Dynamic>;

    //! A generic matrix
    template <typename T>
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    //! A generic 3-component unit vector
    template <typename T>
    using UnitVector = Eigen::Matrix<T, 3, 1>;

    //! A custom AutoDiffScalar type
    template <typename T, int N>
    using ADScalar = Eigen::AutoDiffScalar<Eigen::Matrix<T, N, 1>>;

    // -- Tag forwarding hacks --

    //! Tag forwarding struct
    template <class T> struct tag{};

    //! Pi for current type (tag forwarding)
    template <class T> inline T pi(tag<T>) { return boost::math::constants::pi<T>(); }

    //! Pi for current type (AutoDiffScalar specialization)
    template <class T> inline Eigen::AutoDiffScalar<T> pi(tag<Eigen::AutoDiffScalar<T>>) {
        return boost::math::constants::pi<typename T::Scalar>();
    }

    //! Pi for current type
    template <class T> inline T pi() { return pi(tag<T>()); }

    //! Square root of pi for current type (tag forwarding)
    template <class T> inline T root_pi(tag<T>) { return boost::math::constants::root_pi<T>(); }

    //! Square root of pi for current type (AutoDiffScalar specialization)
    template <class T> inline Eigen::AutoDiffScalar<T> root_pi(tag<Eigen::AutoDiffScalar<T>>) {
        return boost::math::constants::root_pi<typename T::Scalar>();
    }

    //! Square root of pi for current type
    template <class T> inline T root_pi() { return root_pi(tag<T>()); }

    //! Machine precision for current type
    template<class T> inline T mach_eps(tag<T>) { return std::numeric_limits<T>::epsilon(); }

    //! Machine precision for current type (AutoDiffScalar specialization)
    template<class T> inline Eigen::AutoDiffScalar<T> mach_eps(tag<Eigen::AutoDiffScalar<T>>) {
        return std::numeric_limits<typename T::Scalar>::epsilon();
    }

    //! Machine precision for current type
    template<class T> inline T mach_eps() { return mach_eps(tag<T>()); }

    // -- --

    // Some useful unit vectors
    static const UnitVector<double> xhat_double({1, 0, 0});
    static const UnitVector<double> yhat_double({0, 1, 0});
    static const UnitVector<double> zhat_double({0, 0, 1});

    //! Unit vector in the xhat direction
    template <typename T> inline UnitVector<T> xhat(){
        return xhat_double.template cast<T>();
    }

    //! Unit vector in the yhat direction
    template <typename T> inline UnitVector<T> yhat(){
        return yhat_double.template cast<T>();
    }

    //! Unit vector in the zhat direction
    template <typename T> inline UnitVector<T> zhat(){
        return zhat_double.template cast<T>();
    }

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
