/**
Defines vectorization wrappers for functions.

*/

#ifndef _STARRY_VECT_H_
#define _STARRY_VECT_H_

#include <stdlib.h>
#include <Eigen/Core>

namespace vectorize {

    // Shorthand for the scalar type of a vector
    template <typename T>
    using Scalar = typename T::Scalar;

    /**
    Vectorize a function of one argument.

    */
    template <class T>
    class Vec1 {

    public:

        std::function<Scalar<T>(const Scalar<T>&)> function;

        Vec1(std::function<Scalar<T>(const Scalar<T>&)> function) :
            function(function) { }

        // 0
        inline Scalar<T> operator()(const Scalar<T>& arg1) {
            return function(arg1);
        }

        // 1
        inline T operator()(const T& arg1) {
            T res(arg1.size());
            for (size_t i = 0; i < arg1.size(); i++) res(i) = function(arg1(i));
            return res;
        }

    };

    /**
    Vectorize a function of two arguments.

    */
    template <class T>
    class Vec2 {

    public:

        std::function<Scalar<T>(const Scalar<T>&, const Scalar<T>&)> function;

        Vec2(std::function<Scalar<T>(const Scalar<T>&, const Scalar<T>&)> function) :
            function(function) { }

        // 00
        inline Scalar<T> operator()(const Scalar<T>& arg1, const Scalar<T>& arg2) {
            return function(arg1, arg2);
        }

        // 01
        inline T operator()(const Scalar<T>& arg1, const T& arg2) {
            T res(arg2.size());
            for (size_t i = 0; i < arg2.size(); i++) res(i) = function(arg1, arg2(i));
            return res;
        }

        // 10
        inline T operator()(const T& arg1, const Scalar<T>& arg2) {
            T res(arg1.size());
            for (size_t i = 0; i < arg1.size(); i++) res(i) = function(arg1(i), arg2);
            return res;
        }

        // 11
        inline T operator()(const T& arg1, const T& arg2) {
            T res(arg1.size());
            for (size_t i = 0; i < arg1.size(); i++) res(i) = function(arg1(i), arg2(i));
            return res;
        }

    };

    /**
    Vectorize a function of three arguments.

    */
    template <class T>
    class Vec3 {

    public:

        std::function<Scalar<T>(const Scalar<T>&, const Scalar<T>&, const Scalar<T>&)> function;

        Vec3(std::function<Scalar<T>(const Scalar<T>&, const Scalar<T>&, const Scalar<T>&)> function) :
            function(function) { }

        // 000
        inline Scalar<T> operator()(const Scalar<T>& arg1, const Scalar<T>& arg2, const Scalar<T>& arg3) {
            return function(arg1, arg2, arg3);
        }

        // 001
        inline T operator()(const Scalar<T>& arg1, const Scalar<T>& arg2, const T& arg3) {
            T res(arg3.size());
            for (size_t i = 0; i < arg3.size(); i++) res(i) = function(arg1, arg2, arg3(i));
            return res;
        }

        // 010
        inline T operator()(const Scalar<T>& arg1, const T& arg2, const Scalar<T>& arg3) {
            T res(arg2.size());
            for (size_t i = 0; i < arg2.size(); i++) res(i) = function(arg1, arg2(i), arg3);
            return res;
        }

        // 011
        inline T operator()(const Scalar<T>& arg1, const T& arg2, const T& arg3) {
            T res(arg2.size());
            for (size_t i = 0; i < arg2.size(); i++) res(i) = function(arg1, arg2(i), arg3(i));
            return res;
        }

        // 100
        inline T operator()(const T& arg1, const Scalar<T>& arg2, const Scalar<T>& arg3) {
            T res(arg1.size());
            for (size_t i = 0; i < arg1.size(); i++) res(i) = function(arg1(i), arg2, arg3);
            return res;
        }

        // 101
        inline T operator()(const T& arg1, const Scalar<T>& arg2, const T& arg3) {
            T res(arg1.size());
            for (size_t i = 0; i < arg1.size(); i++) res(i) = function(arg1(i), arg2, arg3(i));
            return res;
        }

        // 110
        inline T operator()(const T& arg1, const T& arg2, const Scalar<T>& arg3) {
            T res(arg1.size());
            for (size_t i = 0; i < arg1.size(); i++) res(i) = function(arg1(i), arg2(i), arg3);
            return res;
        }

        // 111
        inline T operator()(const T& arg1, const T& arg2, const T& arg3) {
            T res(arg1.size());
            for (size_t i = 0; i < arg1.size(); i++) res(i) = function(arg1(i), arg2(i), arg3(i));
            return res;
        }

    };

    /**
    Vectorize a function of four arguments.

    */
    template <class T>
    class Vec4 {

    public:

        std::function<Scalar<T>(const Scalar<T>&, const Scalar<T>&, const Scalar<T>&, const Scalar<T>&)> function;

        Vec4(std::function<Scalar<T>(const Scalar<T>&, const Scalar<T>&, const Scalar<T>&, const Scalar<T>&)> function) :
            function(function) { }

        // 0000
        inline Scalar<T> operator()(const Scalar<T>& arg1, const Scalar<T>& arg2, const Scalar<T>& arg3, const Scalar<T>& arg4) {
            return function(arg1, arg2, arg3, arg4);
        }

        // 0001
        inline T operator()(const Scalar<T>& arg1, const Scalar<T>& arg2, const Scalar<T>& arg3, const T& arg4) {
            T res(arg4.size());
            for (size_t i = 0; i < arg4.size(); i++) res(i) = function(arg1, arg2, arg3, arg4(i));
            return res;
        }

        // 0010
        inline T operator()(const Scalar<T>& arg1, const Scalar<T>& arg2, const T& arg3, const Scalar<T>& arg4) {
            T res(arg3.size());
            for (size_t i = 0; i < arg3.size(); i++) res(i) = function(arg1, arg2, arg3(i), arg4);
            return res;
        }

        // 0011
        inline T operator()(const Scalar<T>& arg1, const Scalar<T>& arg2, const T& arg3, const T& arg4) {
            T res(arg3.size());
            for (size_t i = 0; i < arg3.size(); i++) res(i) = function(arg1, arg2, arg3(i), arg4(i));
            return res;
        }

        // 0100
        inline T operator()(const Scalar<T>& arg1, const T& arg2, const Scalar<T>& arg3, const Scalar<T>& arg4) {
            T res(arg2.size());
            for (size_t i = 0; i < arg2.size(); i++) res(i) = function(arg1, arg2(i), arg3, arg4);
            return res;
        }

        // 0101
        inline T operator()(const Scalar<T>& arg1, const T& arg2, const Scalar<T>& arg3, const T& arg4) {
            T res(arg2.size());
            for (size_t i = 0; i < arg2.size(); i++) res(i) = function(arg1, arg2(i), arg3, arg4(i));
            return res;
        }

        // 0110
        inline T operator()(const Scalar<T>& arg1, const T& arg2, const T& arg3, const Scalar<T>& arg4) {
            T res(arg2.size());
            for (size_t i = 0; i < arg2.size(); i++) res(i) = function(arg1, arg2(i), arg3(i), arg4);
            return res;
        }

        // 0111
        inline T operator()(const Scalar<T>& arg1, const T& arg2, const T& arg3, const T& arg4) {
            T res(arg2.size());
            for (size_t i = 0; i < arg2.size(); i++) res(i) = function(arg1, arg2(i), arg3(i), arg4(i));
            return res;
        }

        // 1000
        inline T operator()(const T& arg1, const Scalar<T>& arg2, const Scalar<T>& arg3, const Scalar<T>& arg4) {
            T res(arg1.size());
            for (size_t i = 0; i < arg1.size(); i++) res(i) = function(arg1(i), arg2, arg3, arg4);
            return res;
        }

        // 0001
        inline T operator()(const T& arg1, const Scalar<T>& arg2, const Scalar<T>& arg3, const T& arg4) {
            T res(arg1.size());
            for (size_t i = 0; i < arg1.size(); i++) res(i) = function(arg1(i), arg2, arg3, arg4(i));
            return res;
        }

        // 0010
        inline T operator()(const T& arg1, const Scalar<T>& arg2, const T& arg3, const Scalar<T>& arg4) {
            T res(arg1.size());
            for (size_t i = 0; i < arg1.size(); i++) res(i) = function(arg1(i), arg2, arg3(i), arg4);
            return res;
        }

        // 0011
        inline T operator()(const T& arg1, const Scalar<T>& arg2, const T& arg3, const T& arg4) {
            T res(arg1.size());
            for (size_t i = 0; i < arg1.size(); i++) res(i) = function(arg1(i), arg2, arg3(i), arg4(i));
            return res;
        }

        // 0100
        inline T operator()(const T& arg1, const T& arg2, const Scalar<T>& arg3, const Scalar<T>& arg4) {
            T res(arg1.size());
            for (size_t i = 0; i < arg1.size(); i++) res(i) = function(arg1(i), arg2(i), arg3, arg4);
            return res;
        }

        // 0101
        inline T operator()(const T& arg1, const T& arg2, const Scalar<T>& arg3, const T& arg4) {
            T res(arg1.size());
            for (size_t i = 0; i < arg1.size(); i++) res(i) = function(arg1(i), arg2(i), arg3, arg4(i));
            return res;
        }

        // 0110
        inline T operator()(const T& arg1, const T& arg2, const T& arg3, const Scalar<T>& arg4) {
            T res(arg1.size());
            for (size_t i = 0; i < arg1.size(); i++) res(i) = function(arg1(i), arg2(i), arg3(i), arg4);
            return res;
        }

        // 0111
        inline T operator()(const T& arg1, const T& arg2, const T& arg3, const T& arg4) {
            T res(arg1.size());
            for (size_t i = 0; i < arg1.size(); i++) res(i) = function(arg1(i), arg2(i), arg3(i), arg4(i));
            return res;
        }

    };

} // namespace vectorize

#endif
