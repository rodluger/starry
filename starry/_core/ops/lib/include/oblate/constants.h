/**
\file constants.h
\brief Hard-coded constants

*/

#ifndef _STARRY_OBLATE_CONSTANTS_H_
#define _STARRY_OBLATE_CONSTANTS_H_

// --------------------------
// ------ User-tunable ------
// --------------------------

//! Maximum number of iterations when computing `el2`
#ifndef STARRY_EL2_MAX_ITER
#define STARRY_EL2_MAX_ITER 100
#endif

// --------------------------
// ---------- Fixed ---------
// --------------------------

// Square root of the desired precision in `el2`
#ifndef STARRY_EL2_CA
#define STARRY_EL2_CA 1e-8
#endif

// Replace `inf` with this value in argument to `el2`
#ifndef STARRY_HUGE_TAN
#define STARRY_HUGE_TAN 1e15
#endif

// Nudge k^2 away from 1 when it gets this close
#ifndef STARRY_K2_ONE_TOL
#define STARRY_K2_ONE_TOL 1e-12
#endif

// Maximum number of iterationswhen computing 2F1
#ifndef STARRY_2F1_MAXITER
#define STARRY_2F1_MAXITER 200
#endif

// Tolerance when computing 2F1
#ifndef STARRY_2F1_MAXTOL
#define STARRY_2F1_MAXTOL 1e-15
#endif

// Tolerance when computing 2F1
#ifndef STARRY_2F1_MINTOL
#define STARRY_2F1_MINTOL 1e-12
#endif

#endif