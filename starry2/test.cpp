#include <stdlib.h>
#include <iostream>
#include "utils.h"
#include "basis.h"
using namespace utils;
using namespace basis;


int main() {


    Matrix<ADScalar<double, 8>> p1;
    p1 = Matrix<ADScalar<double, 8>>::Zero(4, 2);
    // p1 = {1, 1} + {1, 1} x
    p1(0, 0) = 2;
    p1(0, 1) = 2;
    p1(3, 0) = 3;
    p1(3, 1) = 3;

    for (int n = 0; n < 4; ++n)


    Matrix<ADScalar<double, 8>> p2;
    p2 = Matrix<ADScalar<double, 8>>::Zero(4, 2);
    // p2 = {1, 1} z
    p2(2, 0) = 5;
    p2(2, 1) = 5;

    Matrix<ADScalar<double, 8>> p1p2;
    // p1p2 = {1, 1} z + {1, 1} xz
    polymul(1, p1, 1, p2, 2, p1p2);


    std::cout << p1 << std::endl << std::endl;
    std::cout << p2 << std::endl << std::endl;
    std::cout << p1p2 << std::endl;
}
