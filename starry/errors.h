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

    /**
    Raised when an operation or function receives an argument that has
    an inappropriate value, and the situation is not described by a more
    precise exception such as IndexError.

    */
    class ValueError : public exception {
        string m_msg;
    public:
        ValueError(const string& message) :
            m_msg(string(message)) { }
        virtual const char* what() const throw() {
            return m_msg.c_str();
        }
    };

    /**
    Raised when a value has the wrong type.

    */
    class TypeError : public exception {
        string m_msg;
    public:
        TypeError(const string& message) :
            m_msg(string(message)) { }
        virtual const char* what() const throw() {
            return m_msg.c_str();
        }
    };

    /**
    Raised when a deprecated operation or function is used.

    */
    class DeprecationError : public exception {
        string m_msg;
    public:
        DeprecationError(const string& message) :
            m_msg(string(message)) { }
        virtual const char* what() const throw() {
            return m_msg.c_str();
        }
    };

    /**
    Raised when an operation or function that is not
    implemented is used.

    */
    class NotImplementedError : public exception {
        string m_msg;
    public:
        NotImplementedError(const string& message) :
            m_msg(string(message)) { }
        virtual const char* what() const throw() {
            return m_msg.c_str();
        }
    };

    /**
    Raised when something hasn't been coded yet;
    for use in development mode only!

    */
    class ToDoError : public exception {
        string m_msg;
    public:
        ToDoError(const string& message) :
            m_msg(string(message)) { }
        virtual const char* what() const throw() {
            return m_msg.c_str();
        }
    };

    /**
    Raised when a sequence subscript is out of range.

    */
    class IndexError : public exception {
        string m_msg;
    public:
        IndexError(const string& message) :
            m_msg(string(message)) { }
        virtual const char* what() const throw() {
            return m_msg.c_str();
        }
    };

    /**
    Raised when a linear algebra operation, such as
    matrix inversion, fails.

    */
    class LinearAlgebraError : public exception {
        string m_msg;
    public:
        LinearAlgebraError(const string& message) :
            m_msg(string(message)) { }
        virtual const char* what() const throw() {
            return m_msg.c_str();
        }
    };

    /**
    Raised when an algorithm fails to converge.

    */
    class ConvergenceError : public exception {
        string m_msg;
    public:
        ConvergenceError(const string& message) :
            m_msg(string(message)) { }
        virtual const char* what() const throw() {
            return m_msg.c_str();
        }
    };

    /**
    Raised and caught internally when a map is not PSD.

    */
    struct MapIsNegative : public exception {
    	const char * what () const throw (){
        	return "The map is not positive semi-definite.";
        }
    };

} // namespace errors

#endif
