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

    Primary<T> star{};
    Secondary<T> b{};
    Secondary<T> c{};
    std::vector<Body<T>*> bodies{&star, &b, &c};
    System<T> system(bodies);

    b.setRadius(10.0);
    std::cout << system.secondaries[0]->getRadius() << std::endl;
    system.secondaries[0]->setRadius(20.0);
    std::cout << b.getRadius() << std::endl;

}
