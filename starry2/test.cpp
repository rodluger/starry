#include <stdlib.h>
#include <iostream>
#include <cmath>
#include "kepler.h"
#include "utils.h"
using namespace utils;
using namespace kepler;

int main() {

    Body<Vector<double>> map{};
    map.setY(1, 0, 1.0);
    std::cout << map.getY(1, 0) << std::endl;

}
