#include <stdlib.h>
#include <Eigen/Core>
#include <iostream>
#include <unsupported/Eigen/AutoDiff>
#include <type_traits>
#include <limits>
#include "vectorize.h"

template <typename T>
using Vector = Eigen::Matrix<T, 1, Eigen::Dynamic>;

class Test {

public:

    int z;
    double func1_(const double& x) {return x + z;}
    double func2_(const double& x, const double& y) {return x + y + z;}

    Test() : z(1) {

        vectorize::Vec1<Vector<double>> func1(std::bind(&Test::func1_, this, std::placeholders::_1));
        vectorize::Vec2<Vector<double>> func2(std::bind(&Test::func2_, this, std::placeholders::_1, std::placeholders::_2));

        Vector<double> foo(3);
        foo(0) = 1;
        foo(1) = 2;
        foo(2) = 3;

        std::cout << func1(foo) << std::endl;
        std::cout << func2(1, foo) << std::endl;

    }

};


int main() {

    Test test;

}
