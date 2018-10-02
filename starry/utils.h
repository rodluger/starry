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
#include <type_traits>
#include "errors.h"

//! Compiler branching optimizations
#define likely(x)    __builtin_expect (!!(x), 1)
#define unlikely(x)  __builtin_expect (!!(x), 0)

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


    using std::isinf;
    using std::fmod;


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


    //! @private
    template <class T> struct tag{};

    //! @private
    template <class T> inline T pi(tag<T>) { return boost::math::constants::pi<T>(); }

    //! @private
    template <class T> inline Eigen::AutoDiffScalar<T> pi(tag<Eigen::AutoDiffScalar<T>>) {
        return boost::math::constants::pi<typename T::Scalar>();
    }

    //! Pi for current type
    template <class T> inline T pi() { return pi(tag<T>()); }

    //! @private
    template <class T> inline T root_pi(tag<T>) { return boost::math::constants::root_pi<T>(); }

    //! @private
    template <class T> inline Eigen::AutoDiffScalar<T> root_pi(tag<Eigen::AutoDiffScalar<T>>) {
        return boost::math::constants::root_pi<typename T::Scalar>();
    }

    //! Square root of pi for current type
    template <class T> inline T root_pi() { return root_pi(tag<T>()); }

    //! @private
    template<class T> inline T mach_eps(tag<T>) { return std::numeric_limits<T>::epsilon(); }

    //! @private
    template<class T> inline Eigen::AutoDiffScalar<T> mach_eps(tag<Eigen::AutoDiffScalar<T>>) {
        return std::numeric_limits<typename T::Scalar>::epsilon();
    }

    //! Machine precision for current type
    template<class T> inline T mach_eps() { return mach_eps(tag<T>()); }

    //! @private
    template<class T> inline std::string precision(tag<T>) {
        return "unknown";
    }

    //! @private
    template<> inline std::string precision(tag<double>) {
        return "double";
    }

    //! @private
    template<> inline std::string precision(tag<Multi>) {
        return std::to_string(STARRY_NMULTI) + " digits";
    }

    //! Scalar type precision descriptor
    template<class T> inline std::string precision() { return precision(tag<T>()); }


    //! @private
    template<class T> inline bool isMulti(const T& scalar) {
        return false;
    }

    //! @private
    template<> inline bool isMulti(const Multi& scalar) {
        return true;
    }

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

    // Normalize a unit vector
    template <typename T>
    inline UnitVector<T> norm_unit(const UnitVector<T>& vec) {
        UnitVector<T> result = vec / sqrt(vec(0) * vec(0) +
                                          vec(1) * vec(1) +
                                          vec(2) * vec(2));
        return result;
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

    //! Return the modulo of an angle by 2 pi
    template <typename T>
    T mod2pi(const T& numer) {
        return fmod(numer, T(2 * pi<T>()));
    }

    //! Manual override of the derivative of `mod2pi`
    template <typename T>
    Eigen::AutoDiffScalar<T> mod2pi(const Eigen::AutoDiffScalar<T>& numer) {
        typename T::Scalar numer_value = numer.value(),
                           modulo_value = mod2pi(numer_value);
        return Eigen::AutoDiffScalar<T>(
          modulo_value,
          numer.derivatives()
        );
    }

    //! Return the modulo of an angle by 360 degrees
    template <typename T>
    T mod360(const T& numer) {
        return fmod(numer, T(360.0));
    }

    //! Check whether a number is infinite
    template <typename T>
    inline bool isInfinite(const T& number) {
        return isinf(number);
    }

    //! Check whether an AutoDiffScalar's value is infinite
    template <typename T>
    inline bool isInfinite(const Eigen::AutoDiffScalar<T>& number) {
        return isinf(number.value());
    }

    //! Figure out the dimensions and types of the coefficients of a map.
    namespace types {

        template <typename T>
        struct TypeSelector{ };

        template <typename T>
        struct TypeSelector <Matrix<T>>{
            using Column = Vector<T>;
            using Row = VectorT<T>;
            using Scalar = T;
            using MapDouble = Matrix<double>;
            using ColumnDouble = Vector<double>;
            using RowDouble = VectorT<double>;
            using RowBool = VectorT<bool>;
        };

        template <typename T>
        struct TypeSelector <Vector<T>>{
            using Column = T;
            using Row = T;
            using Scalar = T;
            using MapDouble = Vector<double>;
            using ColumnDouble = double;
            using RowDouble = double;
            using RowBool = bool;
        };

    }

    //! The type of a `Map` row (Vector^T or scalar)
    template <class MapType>
    using Row = typename types::TypeSelector<MapType>::Row;

    //! The type of a `Map` column (Vector or scalar)
    template <class MapType>
    using Column = typename types::TypeSelector<MapType>::Column;

    //! The scalar type of a `Map` (essentially the same as `MapType::Scalar`)
    template <class MapType>
    using Scalar = typename types::TypeSelector<MapType>::Scalar;

    //! The type of a `Map` row cast to double (Vector^T or scalar)
    template <class MapType>
    using RowDouble = typename types::TypeSelector<MapType>::RowDouble;

    //! The type of a `Map` column cast to double (Vector or scalar)
    template <class MapType>
    using ColumnDouble = typename types::TypeSelector<MapType>::ColumnDouble;

    //! The type of a `Map` cast to double (Matrix or vector)
    template <class MapType>
    using MapDouble = typename types::TypeSelector<MapType>::MapDouble;

    //! The type of a `Map` row cast to bool (Vector or scalar)
    template <class MapType>
    using RowBool = typename types::TypeSelector<MapType>::RowBool;


    // --------------------------
    // -- Map coefficient utils -
    // --------------------------


    //! Resize a map: Matrix specialization
    template <class T>
    inline void resize(Matrix<T>& obj, int N, int NW) {
        obj.resize(N, NW);
    }

    //! Resize a map: Vector specialization
    template <class T>
    inline void resize(Vector<T>& obj, int N, int NW) {
        obj.resize(N);
    }

    //! Resize a map: VectorT specialization
    template <class T>
    inline void resize(VectorT<T>& obj, int N, int NW) {
        obj.resize(NW);
    }

    //! Resize a map: Scalar specialization (does nothing!)
    template <class T>
    inline typename std::enable_if<!std::is_base_of<Eigen::EigenBase<T>, T>::value, void>::type
    resize(T& obj, int N, int NW) { }

    //! Zero out a map: specialization for all Eigen types
    template <class T>
    inline typename std::enable_if<std::is_base_of<Eigen::EigenBase<T>, T>::value, void>::type
    setZero(T& obj) {
        obj.setZero();
    }

    //! Zero out a map: Scalar specialization
    template <class T>
    inline typename std::enable_if<!std::is_base_of<Eigen::EigenBase<T>, T>::value, void>::type
    setZero(T& obj) {
        obj = 0;
    }

    //! Set a map to one: specialization for all Eigen types
    template <class T>
    inline typename std::enable_if<std::is_base_of<Eigen::EigenBase<T>, T>::value, void>::type
    setOnes(T& obj) {
        obj.setOnes();
    }

    //! Set a map to one: Scalar specialization
    template <class T>
    inline typename std::enable_if<!std::is_base_of<Eigen::EigenBase<T>, T>::value, void>::type
    setOnes(T& obj) {
        obj = 1;
    }

    //! Set a row in a map: Vector specialization
    template <class T, class U>
    inline void setRow(Vector<T>& vec, int row, U val) {
        vec(row) = static_cast<T>(val);
    }

    //! Set a row in a map: Matrix specialization
    template <class T, class U>
    inline void setRow(Matrix<T>& vec, int row, const VectorT<U>& val) {
        if (val.size() != vec.cols())
            throw errors::ValueError("Size mismatch in the wavelength dimension.");
        vec.row(row) = val.template cast<T>();
    }

    //! Set a row in a map to a constant value: specialization for all eigen types
    template <class T, class U>
    inline typename std::enable_if<!std::is_base_of<Eigen::EigenBase<U>, U>::value, void>::type
    setRow(Matrix<T>& vec, int row, U val) {
        vec.row(row) = VectorT<T>::Constant(vec.cols(), static_cast<T>(val));
    }

    //! Return a row in a map: Vector specialization
    template <class T>
    inline T getRow(const Vector<T>& vec, int row) {
        return vec(row);
    }

    //! Return a row in a map: VectorT specialization
    template <class T>
    inline VectorT<T> getRow(const Matrix<T>& vec, int row) {
        return vec.row(row);
    }

    //! Set an index of a vector: Row vector specialization
    template <class T>
    inline void setIndex(VectorT<T>& vec, int col, T val) {
        vec(col) = val;
    }

    //! Set an index of a vector: Column vector specialization
    template <class T>
    inline void setIndex(Vector<T>& vec, int row, T val) {
        vec(row) = val;
    }

    //! Set an index of a vector: Scalar specialization
    template <class T>
    inline void setIndex(T& scalar, int col, T val) {
        if (col == 0)
            scalar = val;
        else
            throw errors::IndexError("Attempting to index a scalar variable.");
    }

    //! Get an index of a vector: Row vector specialization
    template <class T>
    inline T getIndex(const VectorT<T>& vec, int col) {
        return vec(col);
    }

    //! Get an index of a vector: Column vector specialization
    template <class T>
    inline T getIndex(const Vector<T>& vec, int row) {
        return vec(row);
    }

    //! Get an index of a vector: Scalar specialization
    template <class T>
    inline T getIndex(const T& scalar, int col) {
        if (col == 0)
            return scalar;
        else
            throw errors::IndexError("Attempting to index a scalar variable.");
    }

    //! Return the value at an index: Vector specialization
    template <class T>
    inline Vector<T> getColumn(const Vector<T>& vec, int col) {
        if (col == 0)
            return vec;
        else
            throw errors::IndexError("Attempting to index a scalar variable.");
    }

    //! Return the value at an index: VectorT specialization
    template <class T>
    inline T getColumn(const VectorT<T>& vec, int col) {
        return vec(col);
    }

    //! Return the value at an index: Matrix specialization
    template <class T>
    inline Vector<T> getColumn(const Matrix<T>& vec, int col) {
        return vec.col(col);
    }

    //! Return the value at an index: Scalar specialization
    template <class T>
    inline typename std::enable_if<!std::is_base_of<Eigen::EigenBase<T>, T>::value, T>::type
    getColumn(const T& vec, int col) {
        if (col == 0)
            return vec;
        else
            throw errors::IndexError("Attempting to index a scalar variable.");
    }

    //! Does a map tensor have any zero elements? Specialization for all Eigen types
    template <typename T>
    inline typename std::enable_if<std::is_base_of<Eigen::EigenBase<T>, T>::value, bool>::type
    hasZero(const T& v) {
        return (v.array() == 0.0).any();
    }

    //! Does a map tensor have any zero elements? Specialization for Scalar
    template <typename T>
    inline typename std::enable_if<!std::is_base_of<Eigen::EigenBase<T>, T>::value, bool>::type
    hasZero(const T& v) {
        return v == 0.0;
    }

    //! Does a map tensor have *all* zero elements? Specialization for all Eigen types
    template <typename T>
    inline typename std::enable_if<std::is_base_of<Eigen::EigenBase<T>, T>::value, bool>::type
    allZero(const T& v) {
        return (v.array() == 0.0).all();
    }

    //! Does a map tensor have *all* zero elements? Specialization for Scalar
    template <typename T>
    inline typename std::enable_if<!std::is_base_of<Eigen::EigenBase<T>, T>::value, bool>::type
    allZero(const T& v) {
        return v == 0.0;
    }

    //! Does a map tensor have *all* unit elements? Specialization for all Eigen types
    template <typename T>
    inline typename std::enable_if<std::is_base_of<Eigen::EigenBase<T>, T>::value, bool>::type
    allOnes(const T& v) {
        return (v.array() == 1.0).all();
    }

    //! Does a map tensor have *all* unit elements? Specialization for Scalar
    template <typename T>
    inline typename std::enable_if<!std::is_base_of<Eigen::EigenBase<T>, T>::value, bool>::type
    allOnes(const T& v) {
        return v == 1.0;
    }

    //! VectorT-Vector dot product
    template <typename T>
    T dot(const VectorT<T>& vT, const Vector<T>& u) {
        return vT.dot(u);
    }

    //! VectorT-Matrix dot product
    template <typename T>
    VectorT<T> dot(const VectorT<T>& vT, const Matrix<T>& U) {
        return vT * U;
    }

    //! Matrix-Vector dot product
    template <typename T>
    VectorT<T> dot(const Matrix<T>& U, const Vector<T>& v) {
        return U * v;
    }

    //! Eigen-Eigen coeff-wise product
    template <typename T>
    inline typename std::enable_if<std::is_base_of<Eigen::EigenBase<T>, T>::value, T>::type
    cwiseProduct(const T& v, const T& u) {
        return v.cwiseProduct(u);
    }

    //! Scalar-scalar product
    template <typename T>
    inline typename std::enable_if<!std::is_base_of<Eigen::EigenBase<T>, T>::value, T>::type
    cwiseProduct(const T& v, const T& u) {
        return v * u;
    }

    //! Eigen-Eigen coeff-wise quotient
    template <typename T>
    inline typename std::enable_if<std::is_base_of<Eigen::EigenBase<T>, T>::value, T>::type
    cwiseQuotient(const T& v, const T& u) {
        return v.cwiseQuotient(u);
    }

    //! Scalar-scalar quotient
    template <typename T>
    inline typename std::enable_if<!std::is_base_of<Eigen::EigenBase<T>, T>::value, T>::type
    cwiseQuotient(const T& v, const T& u) {
        return v / u;
    }

    //! Column-wise Matrix-VectorT multiplication
    template <typename T>
    inline Matrix<T> colwiseProduct(Matrix<T>& mat, const VectorT<T>& vec) {
        Matrix<T> out = mat;
        for (int n = 0; n < mat.cols(); ++n)
            out.col(n) *= vec(n);
        return out;
    }

    //! Column-wise Vector-Scalar multiplication
    template <typename T>
    inline Vector<T> colwiseProduct(Vector<T>& vec, const T& scal) {
        return vec * scal;
    }


} // namespace utils

#endif
