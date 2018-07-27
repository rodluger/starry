#include <stdlib.h>
#include <Eigen/Core>
#include <iostream>
#include <unsupported/Eigen/AutoDiff>
#include <type_traits>
#include <limits>
#include "vectorize.h"

template <typename T>
using Vector = Eigen::Matrix<T, 1, Eigen::Dynamic>;

template<typename T>
T add_(const T& first) {
  return first;
}

template<typename T, typename... Args>
T add_(const T& first, Args... args) {
  return first + add_(args...);
}

// Test the vectorization of functions of up to 4 arguments
void test_vectorization() {

    using namespace vectorize;

    // A scalar and a vector
    double s = 1;
    Vector<double> v = Vector<double>::Ones(5);

    // 1 argument
    Vec1<Vector<double>> add1(add_);
    std::cout << add1(s) << std::endl;
    std::cout << add1(v) << std::endl;

    // 2 arguments
    Vec2<Vector<double>> add2(add_);
    std::cout << add2(s, s) << std::endl;
    std::cout << add2(s, v) << std::endl;
    std::cout << add2(v, s) << std::endl;
    std::cout << add2(v, v) << std::endl;

    // 3 arguments
    Vec3<Vector<double>> add3(add_);
    std::cout << add3(s, s, s) << std::endl;
    std::cout << add3(s, s, v) << std::endl;
    std::cout << add3(s, v, s) << std::endl;
    std::cout << add3(s, v, v) << std::endl;
    std::cout << add3(v, s, s) << std::endl;
    std::cout << add3(v, s, v) << std::endl;
    std::cout << add3(v, v, s) << std::endl;
    std::cout << add3(v, v, v) << std::endl;

    // 4 arguments
    Vec4<Vector<double>> add4(add_);
    std::cout << add4(s, s, s, s) << std::endl;
    std::cout << add4(s, s, s, v) << std::endl;
    std::cout << add4(s, s, v, s) << std::endl;
    std::cout << add4(s, s, v, v) << std::endl;
    std::cout << add4(s, v, s, s) << std::endl;
    std::cout << add4(s, v, s, v) << std::endl;
    std::cout << add4(s, v, v, s) << std::endl;
    std::cout << add4(s, v, v, v) << std::endl;
    std::cout << add4(v, s, s, s) << std::endl;
    std::cout << add4(v, s, s, v) << std::endl;
    std::cout << add4(v, s, v, s) << std::endl;
    std::cout << add4(v, s, v, v) << std::endl;
    std::cout << add4(v, v, s, s) << std::endl;
    std::cout << add4(v, v, s, v) << std::endl;
    std::cout << add4(v, v, v, s) << std::endl;
    std::cout << add4(v, v, v, v) << std::endl;

}






int main() {

    using Grad = Eigen::AutoDiffScalar<Eigen::Matrix<double, 10, 1>>;

    std::cout << mach_eps<double>() << std::endl;
    std::cout << mach_eps<Grad>() << std::endl;
}
