#include <stdlib.h>
#include <iostream>
#include "utils.h"
#include "basis.h"
using namespace utils;
using namespace basis;


int main() {

    Matrix<double> p10(4, 2);
    Matrix<double> p1(4, 2);
    Matrix<double> p2(4, 2);
    Matrix<double> p1p2;
    Matrix<Vector<double>> grad_p1;
    Matrix<Vector<double>> grad_p2;
    Matrix<double> f1, f2;
    p10 << 2.34, 3.10,
           5.4, -3.4,
           1.12, 2.01,
           -0.14, -5.2;
    p2 << 0.12, 4.91,
          1.38, -0.44,
          2.40, 3.08,
          -3.11, 4.46;


    for (int a = 0; a < 4; ++a) {

        // Analytically
        p1 = p10;
        polymul(1, p1, 1, p2, 2, p1p2, grad_p1, grad_p2);
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 9; ++i) {
                std::cout << grad_p1(i, j)(a) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        // Numerically
        for (int b = 0; b < 2; ++b) {
            p1 = p10;
            p1(a, b) -= 1.e-8;
            polymul(1, p1, 1, p2, 2, f1);
            p1 = p10;
            p1(a, b) += 1.e-8;
            polymul(1, p1, 1, p2, 2, f2);
            std::cout << ((f2 - f1) / (2.e-8)).transpose().row(b) << std::endl;

        }
        std::cout << std::endl << std::endl;

    }

}
