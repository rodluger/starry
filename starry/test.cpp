// TODO: I get slightly different values here than when calling the Python
// version. Investigate!

#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <vector>
#include "utils.h"
#include "limbdark.h"
#include "solver.h"
#include "maps.h"

int main() {

    using namespace utils;
    using T = double;

    int lmax = 5;
    T b = 0.5; //0.1; //3.5;
    T r = 0.1; // 0.3; // 3;
    Vector<T> u(lmax + 1);
    u(0) = NAN;
    u(1) = 0.2;
    u(2) = 0.3;
    u(3) = 0.4;
    u(4) = 0.5;
    u(5) = 0.6;

    // Agol
    limbdark::Greens<T> L(lmax);
    std::cout << std::setprecision(16) << L.computeFlux(b, r, u) << std::endl;

    // Luger
    /*maps::Map<Vector<T>> map(lmax);
    map.setU(u.segment(1, lmax));
    std::cout << map.flux(0, b, 0, r) << std::endl;*/
}
