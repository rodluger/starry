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

}; // namespace errors

#endif
