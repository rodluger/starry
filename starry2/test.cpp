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
    std::vector<Secondary<T>*> planets{&b, &c};
    System<T> system(&star, &b);



}
