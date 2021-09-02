/**
\file macros.h
\brief Compile-time-tunable macros

*/

#ifndef _STARRY_CONSTANTS_H_
#define _STARRY_CONSTANTS_H_

//! Number of digits of precision (16 = double)
#ifndef STARRY_NDIGITS
#define STARRY_NDIGITS 16
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

//! Use the incomplete elliptic integrals to compute P?
#ifndef STARRY_USE_INCOMPLETE_INTEGRALS
#define STARRY_USE_INCOMPLETE_INTEGRALS 0
#endif

//! Maximum number of iterations when computing `el2`
#ifndef STARRY_EL2_MAX_ITER
#define STARRY_EL2_MAX_ITER 100
#endif

//! Square root of the desired precision in `el2`
#ifndef STARRY_EL2_CA
#define STARRY_EL2_CA 1e-8
#endif

//! Replace `inf` with this value in argument to `el2`
#ifndef STARRY_HUGE_TAN
#define STARRY_HUGE_TAN 1e15
#endif

//! Nudge k^2 away from 1 when it gets this close
#ifndef STARRY_K2_ONE_TOL
#define STARRY_K2_ONE_TOL 1e-12
#endif

//! Maximum number of iterations when computing 2F1
#ifndef STARRY_2F1_MAXITER
#define STARRY_2F1_MAXITER 200
#endif

//! Tolerance when computing 2F1
#ifndef STARRY_2F1_MAXTOL
#define STARRY_2F1_MAXTOL 1e-15
#endif

//! Tolerance when computing 2F1
#ifndef STARRY_2F1_MINTOL
#define STARRY_2F1_MINTOL 1e-12
#endif

//! Things currently go numerically unstable in our bases for high `l`
#ifndef STARRY_MAX_LMAX
#define STARRY_MAX_LMAX 50
#endif

//! If |sin(theta)| or |cos(theta)| is less than this, set  0
#ifndef STARRY_T_TOL
#define STARRY_T_TOL 1e-12
#endif

//! Tolerance for quartic root polishing
#ifndef STARRY_ROOT_TOL_LOW
#define STARRY_ROOT_TOL_LOW 1e-2
#endif

//! Tolerance for theta = +/- pi/2 in oblate case
#ifndef STARRY_ROOT_TOL_THETA_PI_TWO
#define STARRY_ROOT_TOL_THETA_PI_TWO 1e-5
#endif

//! Tolerance for quartic root polishing
#ifndef STARRY_ROOT_TOL_MED
#define STARRY_ROOT_TOL_MED 1e-10
#endif

//! Tolerance for quartic root polishing
#ifndef STARRY_ROOT_TOL_HIGH
#define STARRY_ROOT_TOL_HIGH 1e-15
#endif

//! Tolerance for duplicate roots
#ifndef STARRY_ROOT_TOL_DUP
#define STARRY_ROOT_TOL_DUP 1e-7
#endif

//! Tolerance for quartic root polishing
#ifndef STARRY_ROOT_TOL_FINAL
#define STARRY_ROOT_TOL_FINAL 1e-13
#endif

//! Maximum number of root polishing iterations
#ifndef STARRY_ROOT_MAX_ITER
#define STARRY_ROOT_MAX_ITER 30
#endif

//! If |b| is less than this value, set equal to 0
#define STARRY_B_ZERO_TOL 1e-8

//! If |b| is within this value of 1, set equal to 1
#ifndef STARRY_B_ONE_TOL
#define STARRY_B_ONE_TOL 1e-6
#endif

//! Minimum value of the oblateness `f`
#ifndef STARRY_MIN_F
#define STARRY_MIN_F 1e-6
#endif

//! Tolerance at singular point `bo ~ 1 - ro`
#ifndef STARRY_BO_EQUALS_ONE_MINUS_RO_TOL
#define STARRY_BO_EQUALS_ONE_MINUS_RO_TOL 1e-6
#endif

//! Tolerance for various functions that calculate phi, xi, and lam
#ifndef STARRY_ANGLE_TOL
#define STARRY_ANGLE_TOL 1e-13
#endif

//! Nudge bo away from ro when it gets this close
#ifndef STARRY_BO_EQUALS_RO_TOL
#define STARRY_BO_EQUALS_RO_TOL 1e-8
#endif

//! Nudge bo away from ro when bo ~ ro ~ 0.5
#ifndef STARRY_BO_EQUALS_RO_EQUALS_HALF_TOL
#define STARRY_BO_EQUALS_RO_EQUALS_HALF_TOL 1e-5
#endif

//! Nudge bo away from zero when it gets this close (oblate case)
#ifndef STARRY_BO_EQUALS_ZERO_TOL
#define STARRY_BO_EQUALS_ZERO_TOL 1e-12
#endif

//! Nudge ro away from zero when it gets this close (oblate case)
#ifndef STARRY_RO_EQUALS_ZERO_TOL
#define STARRY_RO_EQUALS_ZERO_TOL 1e-12
#endif

//! Nudge theta away from pi / 2 when ro = 1
#ifndef STARRY_THETA_UNIT_RADIUS_TOL
#define STARRY_THETA_UNIT_RADIUS_TOL 1e-5
#endif

//! Smallest value of sin(kappa / 2) in computing W
#ifndef STARRY_MIN_SIN_ALPHA
#define STARRY_MIN_SIN_ALPHA 1e-8
#endif

//! Tolerance for complete occultations
#ifndef STARRY_COMPLETE_OCC_TOL
#define STARRY_COMPLETE_OCC_TOL 1e-8
#endif

//! Tolerance for no occultation
#ifndef STARRY_NO_OCC_TOL
#define STARRY_NO_OCC_TOL 1e-8
#endif

//! Tolerance for grazing occultations
#ifndef STARRY_GRAZING_TOL
#define STARRY_GRAZING_TOL 1e-7
#endif

//! Tolerance for the Pal (2012) solver
#ifndef STARRY_CRJ_MAX_ITER
#define STARRY_CRJ_MAX_ITER 100
#endif

//! Tolerance for the Pal (2012) solver
#ifndef STARRY_CRJ_LO_LIM
#define STARRY_CRJ_LO_LIM 2e-26
#endif

//! Tolerance for the Pal (2012) solver
#ifndef STARRY_CRJ_HI_LIM
#define STARRY_CRJ_HI_LIM 3e24
#endif

//! Tolerance for the Pal (2012) solver
#ifndef STARRY_CRJ_TOL
#define STARRY_CRJ_TOL 2e-2
#endif

//! Tolerance for the Pal (2012) solver
#ifndef STARRY_PAL_BO_EQUALS_RO_TOL
#define STARRY_PAL_BO_EQUALS_RO_TOL 1e-3
#endif

//! Tolerance for the Pal (2012) solver
#ifndef STARRY_PAL_BO_EQUALS_RO_MINUS_ONE_TOL
#define STARRY_PAL_BO_EQUALS_RO_MINUS_ONE_TOL 1e-3
#endif

//! Tolerance for the Pal (2012) solver
#ifndef STARRY_PAL_BO_EQUALS_ONE_MINUS_RO_TOL
#define STARRY_PAL_BO_EQUALS_ONE_MINUS_RO_TOL 1e-3
#endif

#endif