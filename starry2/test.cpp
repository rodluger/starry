#include <stdlib.h>
#include <iostream>
#include <cmath>
#include "kepler.h"
#include "utils.h"
using namespace utils;
using namespace kepler;

int main() {

    Primary<Vector<Multi>> primary{};
    primary.setU(1, 0.4);
    primary.setU(2, 0.26);
    std::cout << primary() << std::endl;

}
