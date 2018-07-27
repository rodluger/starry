#include <stdlib.h>
#include <Eigen/Core>
#include <iostream>
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


int main() {

    using namespace vectorize;

    // A scalar and a vector
    double s = 1;
    Vector<double> v = Vector<double>::Ones(5);

    // Define our vectorization wrapper
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
