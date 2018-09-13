// TODO: I get slightly different values here than when calling the Python
// version. Investigate!

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

    Vector<double> time(12);
    time << 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1;

    Primary<T> star(2, 1);
    star.setY(1, 0, 0.1);
    star.setRotPer(1);

    Secondary<T> b(2, 1);
    b.setY(1, 0, 0.5);
    b.setLuminosity(0.001);
    b.setRotPer(1.);
    b.setRefTime(0.3);

    Secondary<T> c(2, 1);
    c.setY(1, 0, 0.5);
    c.setLuminosity(0.001);
    c.setRotPer(1.5);
    c.setRefTime(0.7);

    std::vector<Secondary<T>*> planets{&b, &c};
    System<T> system(&star, planets);
    system.compute(time);
    std::cout << system.getLightcurve() << std::endl << std::endl;

    std::vector<Secondary<T>*> planets_r{&c, &b};
    System<T> system_r(&star, planets_r);
    system_r.compute(time);
    std::cout << system_r.getLightcurve() << std::endl << std::endl;


}
