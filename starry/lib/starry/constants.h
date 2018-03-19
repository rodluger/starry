/**
Defines constants used throughout the code.

*/


// Smallest allowable map coefficient
#ifndef STARRY_MAP_TOLERANCE
#define STARRY_MAP_TOLERANCE                    1.0e-14
#endif

// For impact parameters below this value,
// we Taylor expand the J primitive integral
#ifndef STARRY_BMIN
#define STARRY_BMIN                             1.0e-1
#endif

// Largest tabulated integer square root
#ifndef STARRY_MAX_SQRT
#define STARRY_MAX_SQRT                         201
#endif

// Largest tabulated factorial
#ifndef STARRY_MAX_FACT
#define STARRY_MAX_FACT                         170
#endif

// Largest tabulated half factorial
#ifndef STARRY_MAX_HALF_FACT
#define STARRY_MAX_HALF_FACT                    341
#endif

// Elliptic integral convergence tolerance
#ifndef STARRY_ELLIP_CONV_TOL
#define STARRY_ELLIP_CONV_TOL                   1.0e-8
#endif

// Elliptic integral maximum iterations
#ifndef STARRY_ELLIP_MAX_ITER
#define STARRY_ELLIP_MAX_ITER                   200
#endif

// Uncomment this to disable autodiff
/*
#ifndef STARRY_NO_AUTODIFF
#define STARRY_NO_AUTODIFF                      1
#endif
*/
