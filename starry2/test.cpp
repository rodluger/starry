#include <stdlib.h>
#include <iostream>
#include "utils.h"
#include "basis.h"
using namespace utils;
using namespace basis;


int main() {

    Matrix<double> p1(4, 2); // z
    p1 << 0, 0,
          0, 0,
          1, 2,
          0, 0;
    Matrix<double> p2(4, 2); // y
    p2 << 0, 0,
          0, 0,
          0, 0,
          3, 4;

    Matrix<double> p1p2;
    Matrix<Vector<double>> grad_p1;
    Matrix<Vector<double>> grad_p2;

    polymul(1, p1, 1, p2, 2, p1p2, grad_p1, grad_p2);

    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 2; ++j) {
            std::cout << "f_{" << i << "," << j << "} = " << p1p2(i, j) << std::endl;
            for (int k = 0; k < 4; ++k) {
                std::cout << "df_{" << i << "," << j << "} / dp1_{" << k << "," << j << "} = "
                          << grad_p1(i, j)(k) << std::endl;
            }
            std::cout << std::endl;
        }
    }

}
