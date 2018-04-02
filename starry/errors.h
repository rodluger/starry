/**
Custom exceptions for starry.

*/

#ifndef _STARRY_ERRORS_H_
#define _STARRY_ERRORS_H_

#include <iostream>
#include <exception>

namespace errors {

    using namespace std;

    struct Kepler : public exception {
    	const char * what () const throw (){
        	return "The Kepler solver failed to converge when computing the eccentric anomaly.";
        }
    };

    struct BadY00 : public exception {
    	const char * what () const throw (){
        	return "The coefficient of Y_{0,0} must be positive for all bodies.";
        }
    };

    struct Elliptic : public exception {
    	const char * what () const throw (){
        	return "Elliptic integral did not converge.";
        }
    };

    struct BadTaylor : public exception {
        const char * what () const throw (){
            return "Expression order exceeds the order of the tabulated Taylor expansions.";
        }
    };

    struct LargeOccultorsUnstable : public exception {
        const char * what () const throw (){
            return "Expressions for large occultors are numerically unstable for l > 10. Please enable multi-precision.";
        }
    };


}; // namespace errors

#endif
