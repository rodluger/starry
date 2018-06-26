/**
Custom exceptions for starry.

*/

#ifndef _STARRY_ERRORS_H_
#define _STARRY_ERRORS_H_

#include <iostream>
#include <exception>
#include <string>

namespace errors {

    using namespace std;

    class TooManyDerivs : public exception {
        string m_msg;
    public:
        TooManyDerivs(const int& ngrad) :
            m_msg(string("Too many derivatives requested. Either decrease the degree of the map or re-compile starry with compiler flag STARRY_NGRAD >= " + to_string(ngrad) + ".")) { }
        virtual const char* what() const throw() {
            return m_msg.c_str();
        }
    };

    struct MinimumIsNotAnalytic : public exception {
        const char * what () const throw (){
            return "The minimum of the map cannot be found analytically. To enable numerical searches, instantiate a map from the main `starry` module.";
        }
    };

    struct Kepler : public exception {
    	const char * what () const throw (){
        	return "The Kepler solver failed to converge when computing the eccentric anomaly.";
        }
    };

    struct MapIsNegative : public exception {
    	const char * what () const throw (){
        	return "The map is not positive semi-definite.";
        }
    };

    struct TODO : public exception {
    	const char * what () const throw (){
        	return "TODO!";
        }
    };

    class Elliptic : public exception {
        string m_msg;
    public:
        Elliptic(const string& name) :
            m_msg(string("Elliptic integral " + name + " did not converge.")) { }
        virtual const char* what() const throw() {
            return m_msg.c_str();
        }
    };

    class Primitive : public exception {
        string m_msg;
    public:
        Primitive(const string& name) :
            m_msg(string("Primitive integral " + name + " did not converge.")) { }
        virtual const char* what() const throw() {
            return m_msg.c_str();
        }
    };

    struct BadIndex : public exception {
        const char * what () const throw (){
            return "Invalid index.";
        }
    };

    struct Recursion : public exception {
        const char * what () const throw (){
            return "Error in recursion.";
        }
    };

    struct SqrtNegativeNumber : public exception {
        const char * what () const throw (){
            return "Attempt to take square root of a negative number.";
        }
    };

    struct BadLMIndex : public exception {
        const char * what () const throw (){
            return "Invalid (`l`, `m`) index.";
        }
    };

    struct BadSliceLength : public exception {
        const char * what () const throw (){
            return "Mismatch between slice length and array length.";
        }
    };

    struct BadSystem : public exception {
        const char * what () const throw (){
            return "The first body (and only the first body) must be a `Star`.";
        }
    };

    struct SparseFail : public exception {
        const char * what () const throw (){
            return "Sparse solve failed for matrix `A`.";
        }
    };

    struct BadLM : public exception {
        const char * what () const throw (){
            return "Invalid value for `l` and/or `m`.";
        }
    };

    struct NoLimbDark : public exception {
        const char * what () const throw (){
            return "The map is not currently limb-darkened.";
        }
    };

    struct NotImplemented : public exception {
        const char * what () const throw (){
            return "Function, method, or attribute not implemented.";
        }
    };

    struct Y00IsUnity : public exception {
        const char * what () const throw (){
            return "The Y_{0,0} coefficient is fixed at unity. You probably want to change the body's luminosity instead.";
        }
    };

}; // namespace errors

#endif
