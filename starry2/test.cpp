#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <vector>
#include "kepler.h"
#include "utils.h"
using namespace utils;
using namespace kepler;

// TODO: Why is the starting position x = -49.9685?

int main() {

    using T = Vector<double>;

    Primary<T> star{};
    Secondary<T> b{};
    Secondary<T> c{};
    std::vector<Secondary<T>*> planets{&b, &c};

    System<T> system(&star, &b);

    Vector<double> time(5);
    time << 0, 0.1, 0.2, 0.3, 0.4;

    system.compute(time);

    std::cout << b.getXVector().transpose() << std::endl;
    std::cout << b.getYVector().transpose() << std::endl;
    std::cout << b.getZVector().transpose() << std::endl;


}
