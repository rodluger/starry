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

    class ValueError : public exception {
        string m_msg;
    public:
        ValueError(const string& message) :
            m_msg(string(message)) { }
        virtual const char* what() const throw() {
            return m_msg.c_str();
        }
    };

    class DeprecationError : public exception {
        string m_msg;
    public:
        DeprecationError(const string& message) :
            m_msg(string(message)) { }
        virtual const char* what() const throw() {
            return m_msg.c_str();
        }
    };

    class NotImplementedError : public exception {
        string m_msg;
    public:
        NotImplementedError(const string& message) :
            m_msg(string(message)) { }
        virtual const char* what() const throw() {
            return m_msg.c_str();
        }
    };

    class ToDoError : public exception {
        string m_msg;
    public:
        ToDoError(const string& message) :
            m_msg(string(message)) { }
        virtual const char* what() const throw() {
            return m_msg.c_str();
        }
    };

    class IndexError : public exception {
        string m_msg;
    public:
        IndexError(const string& message) :
            m_msg(string(message)) { }
        virtual const char* what() const throw() {
            return m_msg.c_str();
        }
    };

    class LinearAlgebraError : public exception {
        string m_msg;
    public:
        LinearAlgebraError(const string& message) :
            m_msg(string(message)) { }
        virtual const char* what() const throw() {
            return m_msg.c_str();
        }
    };

}; // namespace errors

#endif
