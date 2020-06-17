/**
\file utils.h
\brief Miscellaneous utilities and definitions used throughout the code.

*/

#ifndef _STARRY_UTILS_H_
#define _STARRY_UTILS_H_

// Includes
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseLU>
#include <chrono>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdarg.h>
#include <stdlib.h>
#include <unsupported/Eigen/AutoDiff>
#include <vector>

//! Number of digits (16 = double)
#ifndef STARRY_NDIGITS
#define STARRY_NDIGITS 16
#else
#if STARRY_NDIGITS > 16
#define STARRY_ENABLE_BOOST
#endif
#endif
#ifdef STARRY_ENABLE_BOOST
#include <boost/math/special_functions/gamma.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
typedef boost::multiprecision::cpp_dec_float<STARRY_NDIGITS> mp_backend;
typedef boost::multiprecision::number<mp_backend, boost::multiprecision::et_off>
    Multi;
#else
typedef void Multi;
#endif

//! Compiler branching optimizations
#ifdef STARRY_BRANCHING_DISABLE_OPTIM
#define likely(x) x
#define unlikely(x) x
#else
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

//! Number of Gaussian-Legendre quadrature points for numerical integration
#ifndef STARRY_QUAD_POINTS
#define STARRY_QUAD_POINTS 100
#endif

//! Max iterations in elliptic integrals
#ifndef STARRY_ELLIP_MAX_ITER
#define STARRY_ELLIP_MAX_ITER 200
#endif

//! Max iterations in computing the M & N integrals
#ifndef STARRY_MN_MAX_ITER
#define STARRY_MN_MAX_ITER 100
#endif

//! Max iterations in computing the I & J integrals
#ifndef STARRY_IJ_MAX_ITER
#define STARRY_IJ_MAX_ITER 200
#endif

//! Refine the downward recursion in the J integral at this index
#ifndef STARRY_REFINE_J_AT
#define STARRY_REFINE_J_AT 25
#endif

//! Cutoff value for `b` below which we reparametrize LD evaluation
#ifndef STARRY_BCUT
#define STARRY_BCUT 1.0e-3
#endif

//! Things currently go numerically unstable in our bases for high `l`
#ifndef STARRY_MAX_LMAX
#define STARRY_MAX_LMAX 50
#endif

//! The value of `pi` in double precision
#ifndef M_PI
#define M_PI 3.14159265358979323846264338328
#endif

//! Square root of `pi` in double precision
#ifndef M_SQRTPI
#define M_SQRTPI 1.77245385090551602729816748334
#endif

// Bounds checks
#ifdef STARRY_DEBUG
//! Check matrix shape (debug mode only)
#define CHECK_SHAPE(MATRIX, ROWS, COLS)                                        \
  assert((static_cast<size_t>(MATRIX.cols()) == static_cast<size_t>(COLS)) &&  \
         (static_cast<size_t>(MATRIX.rows()) == static_cast<size_t>(ROWS)))
//! Check matrix columns (debug mode only)
#define CHECK_COLS(MATRIX, COLS)                                               \
  assert(static_cast<size_t>(MATRIX.cols()) == static_cast<size_t>(COLS))
//! Check matrix rows (debug mode only)
#define CHECK_ROWS(MATRIX, ROWS)                                               \
  assert(static_cast<size_t>(MATRIX.rows()) == static_cast<size_t>(ROWS))
//! Check index bounds (debug mode only)
#define CHECK_BOUNDS(INDEX, IMIN, IMAX)                                        \
  assert((static_cast<size_t>(INDEX) >= static_cast<size_t>(IMIN)) &&          \
         (static_cast<size_t>(INDEX) <= static_cast<size_t>(IMAX)))
#else
//! Check matrix shape (debug mode only)
#define CHECK_SHAPE(MATRIX, ROWS, COLS)                                        \
  do {                                                                         \
  } while (0)
//! Check matrix columns (debug mode only)
#define CHECK_COLS(MATRIX, COLS)                                               \
  do {                                                                         \
  } while (0)
//! Check matrix rows (debug mode only)
#define CHECK_ROWS(MATRIX, ROWS)                                               \
  do {                                                                         \
  } while (0)
//! Check index bounds (debug mode only)
#define CHECK_BOUNDS(INDEX, IMIN, IMAX)                                        \
  do {                                                                         \
  } while (0)

#endif

namespace starry {
namespace utils {

//! Commonly used stuff throughout starry
using std::abs;
using std::isinf;
using std::max;
using std::swap;

//! This is an alias for `enable_if_t`
template <bool B, class T = void>
using EnableIf = typename std::enable_if<B, T>::type;

// --------------------------
// ----- Linear Algebra -----
// --------------------------

//! Matrices
using Eigen::ColMajor;
using Eigen::MatrixBase;
using Eigen::Ref;
using Eigen::RowMajor;
template <typename T> using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T> using UnitVector = Eigen::Matrix<T, 3, 1>;
template <typename T> using RowVector = Eigen::Matrix<T, 1, Eigen::Dynamic>;
template <typename T> using OneByOne = Eigen::Matrix<T, 1, 1>;
template <typename T, int StorageOrder = ColMajor>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, StorageOrder>;
template <typename T, int N>
using ADScalar = Eigen::AutoDiffScalar<Eigen::Matrix<T, N, 1>>;

// --------------------------
// -------- Constants -------
// --------------------------

// Tag forwarding hack
template <class T> struct tag {};

//! Pi for current type
#ifdef STARRY_ENABLE_BOOST
template <class T> inline T pi(tag<T>) {
  return boost::math::constants::pi<T>();
}
template <class T>
inline Eigen::AutoDiffScalar<T> pi(tag<Eigen::AutoDiffScalar<T>>) {
  return boost::math::constants::pi<typename T::Scalar>();
}
template <class T> inline T pi() { return pi(tag<T>()); }
#else
template <class T> inline T pi() { return static_cast<T>(M_PI); }
#endif

//! Square root of pi for current type
#ifdef STARRY_ENABLE_BOOST
template <class T> inline T root_pi(tag<T>) {
  return boost::math::constants::root_pi<T>();
}
template <class T>
inline Eigen::AutoDiffScalar<T> root_pi(tag<Eigen::AutoDiffScalar<T>>) {
  return boost::math::constants::root_pi<typename T::Scalar>();
}
template <class T> inline T root_pi() { return root_pi(tag<T>()); }
#else
template <class T> inline T root_pi() { return static_cast<T>(M_SQRTPI); }
#endif

//! Machine precision for current type
template <class T> inline T mach_eps(tag<T>) {
  return std::numeric_limits<T>::epsilon();
}
template <class T>
inline Eigen::AutoDiffScalar<T> mach_eps(tag<Eigen::AutoDiffScalar<T>>) {
  return std::numeric_limits<typename T::Scalar>::epsilon();
}
template <class T> inline T mach_eps() { return mach_eps(tag<T>()); }

// --------------------------
// ----- Utility Funcs ------
// --------------------------

//! Check if a number is even (or doubly, triply, quadruply... even)
inline bool is_even(int n, int ntimes = 1) {
  for (int i = 0; i < ntimes; i++) {
    if ((n % 2) != 0)
      return false;
    n /= 2;
  }
  return true;
}

//! Modulo for angles.
// Equivalent to the Python expression `result = angle % max_angle`
template <typename T, typename Scalar = double>
inline T angle(const T &x, const Scalar max_angle = 2 * M_PI) {
  T result = x;
  while (result < 0)
    result += max_angle;
  while (result > max_angle)
    result -= max_angle;
  return result;
}

// --------------------------
// ------ Unit Vectors ------
// --------------------------

// Some useful unit vectors
static const UnitVector<double> xhat_double({1, 0, 0});
static const UnitVector<double> yhat_double({0, 1, 0});
static const UnitVector<double> zhat_double({0, 0, 1});

//! Unit vector in the xhat direction
template <typename T> inline UnitVector<T> xhat() {
  return xhat_double.template cast<T>();
}

//! Unit vector in the yhat direction
template <typename T> inline UnitVector<T> yhat() {
  return yhat_double.template cast<T>();
}

//! Unit vector in the zhat direction
template <typename T> inline UnitVector<T> zhat() {
  return zhat_double.template cast<T>();
}

// Normalize a unit vector
template <typename T> inline UnitVector<T> norm_unit(const UnitVector<T> &vec) {
  UnitVector<T> result =
      vec / sqrt(vec(0) * vec(0) + vec(1) * vec(1) + vec(2) * vec(2));
  return result;
}

// --------------------------
// -------- Debugging -------
// --------------------------

template <class T> inline void print_scalar(const T &x) {
  std::cout << x << std::endl;
}

template <class T> inline void print_scalar(const Eigen::AutoDiffScalar<T> &x) {
  std::cout << x << ", " << x.derivatives().transpose() << std::endl;
}

class StarryException : public std::exception {

  std::string m_msg;

  std::string bold(const char *msg) {
    std::stringstream boldmsg;
    boldmsg << "\e[1m" << msg << "\e[0m";
    return boldmsg.str();
  }

  std::string url(const char *msg) {
    std::stringstream urlmsg;
    urlmsg << "\e[1m\e[34m" << msg << "\e[0m\e[39m";
    return urlmsg.str();
  }

public:
  StarryException(const std::string &msg, const std::string &file,
                  const std::string &function, const std::string &args)
      : m_msg(std::string("Something went wrong in starry! \n\n") +
              bold("Error: ") + msg + std::string("\n") + bold("File: ") +
              file + std::string("\n") + bold("Function: ") + function +
              std::string("\n") + bold("Arguments: ") + args +
              std::string("\n") +
              std::string(
                  "If you believe this is a bug, please open an issue at ") +
              url("https://github.com/rodluger/starry/issues/new. ") +
              std::string("Include the information above and a minimum working "
                          "example. \n")) {}

  virtual const char *what() const throw() { return m_msg.c_str(); }
};

} // namespace utils
} // namespace starry
#endif