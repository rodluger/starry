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

//! Number of digits for the multiprecision type in starry.multi
#ifndef STARRY_NMULTI
#define STARRY_NMULTI                           32
#endif

//! Max iterations in elliptic integrals
#ifndef STARRY_ELLIP_MAX_ITER
#define STARRY_ELLIP_MAX_ITER                   200
#endif

//! Max iterations in computation of I_v and J_v
#ifndef STARRY_IJ_MAX_ITER
#define STARRY_IJ_MAX_ITER                      200
#endif

//! Max iterations in Kepler solver
#ifndef STARRY_KEPLER_MAX_ITER
#define STARRY_KEPLER_MAX_ITER                  100
#endif

//! Re-parameterize solution vector when
//! abs(b - r) < STARRY_EPS_BMR_ZERO
#ifndef STARRY_EPS_BMR_ZERO
#define STARRY_EPS_BMR_ZERO                     1e-2
#endif

//! Re-parameterize solution vector when
//! 1 - STARRY_EPS_BMR_ONE < abs(b - r) < 1 + STARRY_EPS_BMR_ONE
#ifndef STARRY_EPS_BMR_ONE
#define STARRY_EPS_BMR_ONE                      1e-5
#endif

//! Re-parameterize solution vector when
//! 1 - STARRY_EPS_BMR_ONE < abs(b + r) < 1 + STARRY_EPS_BPR_ONE
#ifndef STARRY_EPS_BPR_ONE
#define STARRY_EPS_BPR_ONE                      1e-5
#endif

//! Re-parameterize solution vector when
//! abs(b) < STARRY_EPS_B_ZERO
#ifndef STARRY_EPS_B_ZERO
#define STARRY_EPS_B_ZERO                       1e-1
#endif

namespace utils {

    // --------------------------
    // --------- Aliases --------
    // --------------------------


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


    // --------------------------
    // -- Tag forwarding hacks --
    // --------------------------


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


    // --------------------------
    // ------ Unit Vectors ------
    // --------------------------


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


    // --------------------------
    // ----- Misc utilities -----
    // --------------------------


    //! Check if a number is even (or doubly, triply, quadruply... even)
    inline bool is_even(int n, int ntimes=1) {
        for (int i = 0; i < ntimes; i++) {
            if ((n % 2) != 0) return false;
            n /= 2;
        }
        return true;
    }

    //! Figure out the dimensions of the coefficients of a map.
    namespace types {

        template <typename T>
        struct TypeSelector{ };

        template <typename T>
        struct TypeSelector <Matrix<T>>{
            using Column = Vector<T>;
            using Row = VectorT<T>;
            using Scalar = T;
        };

        template <typename T>
        struct TypeSelector <Vector<T>>{
            using Column = T;
            using Row = T;
            using Scalar = T;
        };

    }

    //! The type of a `Map` row (vector^T or scalar)
    template <class MapType>
    using Row = typename types::TypeSelector<MapType>::Row;

    //! The type of a `Map` column (vector or scalar)
    template <class MapType>
    using Column = typename types::TypeSelector<MapType>::Column;

    //! The scalar type of a `Map` (essentially the same as `MapType::Scalar`)
    template <class MapType>
    using Scalar = typename types::TypeSelector<MapType>::Scalar;


    // --------------------------
    // -- Map coefficient utils -
    // --------------------------


    //! Set a map vector/matrix to zero
    template <class T>
    void setZero(T& obj, int N, int NW) {
        obj = 0;
    }

    //! Set a map vector/matrix to zero
    template <class T>
    void setZero(Matrix<T>& obj, int N, int NW) {
        obj = Matrix<T>::Zero(N, NW);
    }

    //! Set a map vector/matrix to zero
    template <class T>
    void setZero(Vector<T>& obj, int N, int NW) {
        obj = Vector<T>::Zero(N);
    }

    //! Set a map vector/matrix to zero
    template <class T>
    void setZero(VectorT<T>& obj, int N, int NW) {
        obj = VectorT<T>::Zero(NW);
    }

    //! Set a vector map coefficient
    template <class T>
    inline void setCoeff(Vector<T>& y, int n, const T& coeff) {
        y(n) = coeff;
    }

    //! Set a matrix map coefficient
    template <class T>
    inline void setCoeff(Matrix<T>& y, int n, const VectorT<T>& coeff) {
        if (coeff.size() != y.cols())
            throw errors::ValueError("Size mismatch in the wavelength dimension.");
        y.row(n) = coeff;
    }

    //! Get a vector map coefficient
    template <class T>
    inline T getCoeff(const Vector<T>& y, int n) {
        return y(n);
    }

    //! Get a matrix map coefficient
    template <class T>
    inline VectorT<T> getCoeff(const Matrix<T>& y, int n) {
        return y.row(n);
    }

    //! Get the vector map coefficient at index `n`
    template <class T>
    inline T getFirstCoeff(const Vector<T>& y, int n) {
        return y(n);
    }

    //! Get the matrix map coefficient at index `(n, 0)`
    template <class T>
    inline T getFirstCoeff(const Matrix<T>& y, int n) {
        return y(n, 0);
    }

    //! Set a row in a map vector
    template <class T>
    inline void setRow(Vector<T>& vec, int row, T val) {
        vec(row) = val;
    }

    //! Set a row in a map vector
    template <class T>
    inline void setRow(Matrix<T>& vec, int row, const VectorT<T>& val) {
        vec.row(row) = val;
    }

    //! Set a row in a map vector
    template <class T>
    inline void setRow(Matrix<T>& vec, int row, T val) {
        vec.row(row) = VectorT<T>::Constant(vec.cols(), val);
    }

    //! Return a row in a map vector
    template <class T>
    inline T getRow(const Vector<T>& vec, int row) {
        return vec(row);
    }

    //! Return a row in a map vector
    template <class T>
    inline Vector<T> getRow(const Matrix<T>& vec, int row) {
        return vec.row(row);
    }

    //! Vector-vector dot product
    template <typename T>
    T dot(const VectorT<T>& vT, const Vector<T>& u) {
        return vT.dot(u);
    }

    //! Vector-matrix dot product
    template <typename T>
    VectorT<T> dot(const VectorT<T>& vT, const Matrix<T>& U) {
        return vT * U;
    }

}; // namespace utils

#endif
