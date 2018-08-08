#include <stdlib.h>
#include <iostream>
#include "maps.h"
#include "utils.h"

int main() {

    int lmax = 1;
    int nwav = 2;
    maps::Map<double, double> map(lmax, nwav);

    // First wavelength is Y_{1,0}
    // Second wavelength is Y_{1,1}
    utils::VectorT<double> coeff(nwav);
    coeff << 1, 0;
    map.setCoeff(1, 0, coeff);
    coeff << 0, 1;
    map.setCoeff(1, 1, coeff);

    std::cout << map.evaluate(0.01, 0.3, 0.5, true).transpose() << std::endl;
    std::cout << std::endl;
    std::cout << map.dI << std::endl;

    std::cout << std::endl;
    std::cout << map.flux(30, 1.3, 0.5, 0., false).transpose() << std::endl;
}
