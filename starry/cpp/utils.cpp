#include "starry.h"

// Factorial for integers
double factorial(int n) {
    if (n < 0)
        return INFINITY;
    return
        gsl_sf_fact(n);
}

// Factorial for floats
double factorial(double n) {
    if (roundf(n) == n)
        return factorial((int)n);
    else
        return gsl_sf_gamma(n + 1);
}
