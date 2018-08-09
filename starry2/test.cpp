#include <stdlib.h>
#include <iostream>
#include "maps.h"
#include "utils.h"

int main() {

    int lmax = 1;
    int nwav = 2;
    maps::Map<double> map(lmax, nwav);

    // First wavelength is Y_{1,0}
    // Second wavelength is Y_{1,1}
    utils::VectorT<double> coeff(nwav);
    coeff << 1, 0;
    map.setYlm(1, 0, coeff);
    coeff << 0, 1;
    map.setYlm(1, 1, coeff);


    std::cout << map.flux(0.001, 1.3, 0.5, 0., true).transpose() << std::endl;

    std::cout << map.dF << std::endl;

}
