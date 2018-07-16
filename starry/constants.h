/**
Defines compiler constants used throughout the code.

*/

#ifndef _STARRY_CONSTS_H_
#define _STARRY_CONSTS_H_

// Gradient size in starry.grad
#ifndef STARRY_NGRAD
#define STARRY_NGRAD                            13
#endif

// Number of digits for the multiprecision type in starry.multi
#ifndef STARRY_NMULTI
#define STARRY_NMULTI                           32
#endif

// Max iterations in elliptic integrals
#ifndef STARRY_ELLIP_MAX_ITER
#define STARRY_ELLIP_MAX_ITER                   200
#endif

// Max iterations in computation of I_v and J_v
#ifndef STARRY_IJ_MAX_ITER
#define STARRY_IJ_MAX_ITER                      200
#endif

// Max iterations in Kepler solver
#ifndef STARRY_KEPLER_MAX_ITER
#define STARRY_KEPLER_MAX_ITER                  100
#endif

// Re-parameterize solution vector when abs(b - r) < STARRY_EPS_BMR_ZERO
#ifndef STARRY_EPS_BMR_ZERO
#define STARRY_EPS_BMR_ZERO                     1e-2
#endif

// Re-parameterize solution vector when 1 - STARRY_EPS_BMR_ONE < abs(b - r) < 1 + STARRY_EPS_BMR_ONE
#ifndef STARRY_EPS_BMR_ONE
#define STARRY_EPS_BMR_ONE                      1e-5
#endif

// Re-parameterize solution vector when 1 - STARRY_EPS_BMR_ONE < abs(b + r) < 1 + STARRY_EPS_BPR_ONE
#ifndef STARRY_EPS_BPR_ONE
#define STARRY_EPS_BPR_ONE                      1e-5
#endif

// Re-parameterize solution vector when abs(b) < STARRY_EPS_B_ZERO
#ifndef STARRY_EPS_B_ZERO
#define STARRY_EPS_B_ZERO                       1e-1
#endif

#endif
