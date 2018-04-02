/**
Defines constants used throughout the code.

*/


// Smallest allowable map coefficient
#ifndef STARRY_MAP_TOLERANCE
#define STARRY_MAP_TOLERANCE                    1.0e-14
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

// Physical constants
#ifndef STARRY_PHYSICAL_CONSTANTS
#define STARRY_PHYSICAL_CONSTANTS
#define BIGG                                    6.67428e-11                     // Gravitational constant in m^3/kg/s^2
#define DAY                                     86400.                          // Number of seconds in one day
#define CLIGHT                                  2.998e8                         // Speed of light in m / s
#define REARTH                                  6.3781e6                        // Radius of the Earth in m
#define PARSEC                                  3.086e16                        // Meters in 1 parsec
#define MEARTH                                  (3.986004418e14 / BIGG)         // Mass of Earth in kg (from GM)
#define MSUN                                    (1.32712440018e20 / BIGG)       // Mass of the sun in kg (from GM)
#define AU                                      149597870700.                   // Astronomical unit in m
#define RSUN                                    6.957e8                         // Radius of the Sun in m
#define LSUN                                    3.828e26                        // Solar luminosity in W/m^2
#define RJUP                                    7.1492e7                        // Radius of Jupiter in m
#define DEGREE                                  (M_PI / 180.)                   // One degree in radians
#endif
