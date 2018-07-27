/**
Miscellaneous stuff used throughout the code.

*/

#ifndef _STARRY_UTILS_H_
#define _STARRY_UTILS_H_

#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>
#include <iostream>
#include <limits>
#include <vector>
#include "constants.h"
#include "errors.h"

// Physical constants
#define BIGG                                    6.67428e-11                     // Gravitational constant in m^3/kg/s^2
#define DAY                                     86400.                          // Number of seconds in one day
#define CLIGHT                                  299792458.                      // Speed of light in m / s
#define REARTH                                  6.3781e6                        // Radius of the Earth in m
#define PARSEC                                  3.086e16                        // Meters in 1 parsec
#define MEARTH                                  (3.986004418e14 / BIGG)         // Mass of Earth in kg (from GM)
#define MSUN                                    (1.32712440018e20 / BIGG)       // Mass of the sun in kg (from GM)
#define AU                                      149597870700.                   // Astronomical unit in m
#define RSUN                                    6.957e8                         // Radius of the Sun in m
#define LSUN                                    3.828e26                        // Solar luminosity in W/m^2
#define RJUP                                    7.1492e7                        // Radius of Jupiter in m
#define DEGREE                                  (M_PI / 180.)                   // One degree in radians

// Multiprecision datatype
#include <boost/multiprecision/cpp_dec_float.hpp>
typedef boost::multiprecision::cpp_dec_float<STARRY_NMULTI> mp_backend;
typedef boost::multiprecision::number<mp_backend, boost::multiprecision::et_off> Multi;
#if STARRY_NMULTI > 150
#error "Currently, PI is computed to a maximum of 150 digits of precision. If you **really** need `STARRY_NMULTI` > 150, you will need to re-define PI in `utils.h`."
#endif

// Some frequently used constants
static const double PI_DOUBLE = M_PI;
static const Multi PI_MULTI = Multi("3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128");
template <typename T> inline T PI(){ return T(PI_DOUBLE); }
template <> inline Multi PI(){ return PI_MULTI; }

static const double SQRT_PI_DOUBLE = sqrt(PI<double>());
static const Multi SQRT_PI_MULTI = sqrt(PI<Multi>());
template <typename T> inline T SQRT_PI(){ return T(SQRT_PI_DOUBLE); }
template <> inline Multi SQRT_PI(){ return SQRT_PI_MULTI; }

static const double TWO_OVER_SQRT_PI_DOUBLE = 2.0 / sqrt(PI<double>());
static const Multi TWO_OVER_SQRT_PI_MULTI = 2.0 / sqrt(PI<Multi>());
template <typename T> inline T TWO_OVER_SQRT_PI(){ return T(TWO_OVER_SQRT_PI_DOUBLE); }
template <> inline Multi TWO_OVER_SQRT_PI(){ return TWO_OVER_SQRT_PI_MULTI; }

// Our custom vector types
template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T>
using VectorT = Eigen::Matrix<T, 1, Eigen::Dynamic>;
template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using UnitVector = Eigen::Matrix<T, 3, 1>;
using Grad = Eigen::AutoDiffScalar<Eigen::Matrix<double, STARRY_NGRAD, 1>>;
using Grad0 = Eigen::AutoDiffScalar<Eigen::Matrix<double, 0, 1>>;


// Some useful unit vectors
static const UnitVector<double> xhat({1, 0, 0});
static const UnitVector<double> yhat({0, 1, 0});
static const UnitVector<double> zhat({0, 0, 1});

// Return the value of a scalar MapType variable
inline double get_value(double x) { return x; }
inline double get_value(Grad x) { return x.value(); }
inline double get_value(Multi x) {
    if ((x > 1e308) || (x < -1e308))
        return INFINITY;
    else
        return (double)x;
}
inline Vector<double> get_value(Vector<double> x) { return x; }
inline Vector<double> get_value(Vector<Multi> x) {
    Vector<double> vec;
    vec.resize(x.size());
    for (int n = 0; n < x.size(); n++) {
        if ((x(n) > 1e308) || (x(n) < -1e308))
            vec(n) = INFINITY;
        else
            vec(n) = (double)x(n);
    }
    return vec;
}
inline Vector<double> get_value(Vector<Grad> x) {
    Vector<double> vec;
    vec.resize(x.size());
    for (int n = 0; n < x.size(); n++) {
        vec(n) = x(n).value();
    }
    return vec;
}

// Set the value of a MapType variable
template <typename T>
inline void set_value(T& x, T& y) { x = y; }
template <>
inline void set_value(Grad& x, Grad& y) { x.value() = y.value(); }

// Print the derivatives of a MapType variable for debugging
template <typename T>
void print_derivs(T x) { std::cout << "None" << std::endl; }
template <>
void print_derivs(Grad x) { std::cout << x.derivatives().transpose() << std::endl; }

// Zero out the derivatives of a MapType variable
template <typename T>
inline void set_derivs_to_zero(T& x) { }
template <>
inline void set_derivs_to_zero(Grad& x) { x.derivatives().setZero(x.derivatives().size()); }

// Normalize a unit vector
template <typename T>
inline UnitVector<T> norm_unit(const UnitVector<T>& vec) {
    UnitVector<T> result = vec / sqrt(vec(0) * vec(0) + vec(1) * vec(1) + vec(2) * vec(2));
    return result;
}

// Helper function to figure out if we're using multiprecision
template <typename T>
inline bool is_Multi(T x) {
    return false;
}

// Helper function to figure out if we're using multiprecision
template <>
inline bool is_Multi(Multi x) {
    return true;
}

// Helper function to figure out if we're using autodiff
template <typename T>
inline bool is_Grad(T x) {
    return false;
}

// Helper function to figure out if we're using autodiff
template <>
inline bool is_Grad(Grad x) {
    return true;
}

// Check if number is even (or doubly, triply, quadruply... even)
inline bool is_even(int n, int ntimes=1) {
    for (int i = 0; i < ntimes; i++) {
        if ((n % 2) != 0) return false;
        n /= 2;
    }
    return true;
}

// Machine precision at current type
template <typename T>
inline T mach_eps() {
    return std::numeric_limits<T>::epsilon();
}

template <>
inline Grad mach_eps() {
    return Grad(mach_eps<double>());
}

// Re-definition of fmod so we can define its derivative below
using std::fmod;
template <typename T>
T mod2pi(const T& numer) {
    return fmod(numer, T(2 * PI<T>()));
}

// Derivative of the floating point modulo function
template <typename T>
Eigen::AutoDiffScalar<T> mod2pi(const Eigen::AutoDiffScalar<T>& numer) {
    typename T::Scalar numer_value = numer.value(),
                       modulo_value = mod2pi(numer_value);
    return Eigen::AutoDiffScalar<T>(
      modulo_value,
      numer.derivatives()
    );
}

#endif
