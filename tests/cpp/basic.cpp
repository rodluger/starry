/**
Instantiate a simple map and compute the occultation flux.

*/

#include <stdlib.h>
#include <iostream>
#include "starry2.h"
#include <Eigen/Core>

using namespace starry2;

int main() {

    Map<Default<double>> map(2);
    
    Eigen::VectorXd u(2);
    u(0) = 0.4;
    u(1) = 0.26;
    map.setU(u);

    Eigen::VectorXd b(1000); 
    b = Eigen::VectorXd::LinSpaced(1000, 0.0, 1.1);
    Eigen::VectorXd flux(b.size());
    for (int k = 0; k < b.size(); ++k)
        map.computeFlux(0.0, b(k), 0.0, 0.1, flux.row(k));

}
