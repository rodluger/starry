#include <stdlib.h>
#include <iostream>
#include "maps.h"
#include "utils.h"
using namespace utils;


int main() {

    using T = Vector<double>;
    maps::Map<T> map(10);

    map.setYlm(5, 5, 1.0);
    std::cout << map.__repr__() << std::endl;

    map.setYlm(4, 4, 1.0);
    std::cout << map.__repr__() << std::endl;

    map.setYlm(5, 5, 0.0);
    std::cout << map.__repr__() << std::endl;

    map.setUl(1, 1.0);
    std::cout << map.__repr__() << std::endl;

    map.setUl(3, 1.0);
    std::cout << map.__repr__() << std::endl;

    map.setUl(3, 0.0);
    std::cout << map.__repr__() << std::endl;

    map.setUl(1, 0.0);
    std::cout << map.__repr__() << std::endl;

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
