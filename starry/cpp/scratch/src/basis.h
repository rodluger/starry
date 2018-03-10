/**
Spherical harmonic, polynomial, and Green's basis utilities.

*/

#ifndef _STARRY_BASIS_H_
#define _STARRY_BASIS_H_

#include <cmath>
#include "fact.h"

namespace basis {

    // Contraction coefficient for the Ylms
    double C(int p, int q, int k) {
        if ((p > k) && ((p - k) % 2 == 0)) {
            return 0;
        } else if ((q > p) && ((q - p) % 2 == 0)) {
            return 0;
        } else {
            return fact::half_factorial(k) /
                        (fact::half_factorial(q) *
                         fact::half_factorial(k - p) *
                         fact::half_factorial(p - q));
        }
    }


}; // namespace basis

#endif
