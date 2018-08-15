#include <stdlib.h>
#include <iostream>
#include "maps.h"
#include "utils.h"
using namespace utils;


int main() {

    using T = Vector<double>;
    int lmax = 10;
    int res = 150;
    int frames = 50;
    
    maps::Map<T> map(lmax);
    for (int l = 0; l < lmax + 1; ++l) {
        for (int m = -l; m < l + 1; ++m) {
            map.setYlm(l, m, 1.0);
        }
    }
    std::vector<Matrix<double>> I;
    Vector<Scalar<T>> x, theta;
    x = Vector<Scalar<T>>::LinSpaced(res, -1, 1);
    theta = Vector<Scalar<T>>::LinSpaced(frames, 0, 360);
    for (int t = 0; t < frames; t++){
        I.push_back(Matrix<double>::Zero(res, res));
        for (int i = 0; i < res; i++){
            for (int j = 0; j < res; j++){
                I[t](j, i) = static_cast<double>(
                             map.evaluate(theta(t), x(i), x(j)));
            }
        }
    }

    /*
    int lmax = 2;
    double theta = 30;
    double x = 0.1;
    double y = 0.3;
    double r = 0.1;

    // Default
    maps::Map<Vector<double>> map2(lmax);
    for (int l = 0; l < lmax + 1; ++l) {
        for (int m = -l; m < l + 1; ++m) {
            map2.setYlm(l, m, 1.0);
        }
    }
    std::cout << "Default" << std::endl;
    std::cout << map2.evaluate(theta, x, y, false) << std::endl;
    std::cout << std::endl;
    std::cout << map2.evaluate(theta, x, y, true) << std::endl;
    std::cout << std::endl;
    std::cout << map2.flux(theta, x, y, r, false) << std::endl;
    std::cout << std::endl;
    std::cout << map2.flux(theta, x, y, r, true) << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;

    // Multi
    maps::Map<Vector<Multi>> map3(lmax);
    for (int l = 0; l < lmax + 1; ++l) {
        for (int m = -l; m < l + 1; ++m) {
            map3.setYlm(l, m, Multi(1.0));
        }
    }
    std::cout << "Multi" << std::endl;
    std::cout << map3.evaluate(theta, x, y, false) << std::endl;
    std::cout << std::endl;
    std::cout << map3.evaluate(theta, x, y, true) << std::endl;
    std::cout << std::endl;
    std::cout << map3.flux(theta, x, y, r, false) << std::endl;
    std::cout << std::endl;
    std::cout << map3.flux(theta, x, y, r, true) << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;

    // Spectral
    int nwav = 2;
    Vector<double> coeff(2);
    coeff(0) = 1.0;
    coeff(1) = 2.0;
    maps::Map<Matrix<double>> map(lmax, nwav);
    for (int l = 0; l < lmax + 1; ++l) {
        for (int m = -l; m < l + 1; ++m) {
            map.setYlm(l, m, coeff);
        }
    }
    std::cout << "Spectral" << std::endl;
    std::cout << map.evaluate(theta, x, y, false) << std::endl;
    std::cout << std::endl;
    std::cout << map.evaluate(theta, x, y, true) << std::endl;
    std::cout << std::endl;
    std::cout << map.flux(theta, x, y, r, false) << std::endl;
    std::cout << std::endl;
    std::cout << map.flux(theta, x, y, r, true) << std::endl;
    */

}
