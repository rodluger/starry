#include <stdlib.h>
#include <iostream>
#include "utils.h"
#include "basis.h"
using namespace utils;
using namespace basis;


int main() {

    Matrix<double> p1(4, 2);
    Matrix<double> p2(4, 2);
    Matrix<double> p1p2;
    Matrix<Vector<double>> grad_p1;
    Matrix<Vector<double>> grad_p2;
    Matrix<double> eps1(4, 2);
    Matrix<double> eps2(4, 2);

    for (int n = 0; n < 8; ++n) {

        eps1 = Map<Matrix<double>>(Vector<double>::Unit(8, n).data()).transpose();

        std::cout << eps1 << std::endl;

        /*
        p1 << 2.34, 3.10,
              5.4, -3.4,
              1.12, 2.01,
              -0.14, -5.2;

        p1 += eps1;

        p2 << 0.12, 4.91,
              1.38, -0.44,
              2.40, 3.08,
              -3.11, 4.46;

        p2 += eps2;

        polymul(1, p1, 1, p2, 2, p1p2, grad_p1, grad_p2);
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 4; ++k) {
                    grad_p1(i, j)(k)
                }
            }
        }

        */
    }

}
