#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <vector>
#include "kepler.h"
#include "utils.h"
using namespace utils;
using namespace kepler;

int main() {

    using T = Vector<double>;

    Primary<T> star(2, 1);
    Secondary<T> b(2, 1);
    System<T> system(&star, &b);
    Vector<double> time(12);
    time << 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1;

    Vector<double> y10(2);
    y10 << 0.57, -0.57;

    star.setY(1, 0, 0.57);
    star.setRotPer(1);

    system.compute(time);

    std::cout << star.getLightcurve() << std::endl << std::endl;

}
