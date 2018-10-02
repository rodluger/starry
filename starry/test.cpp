#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <vector>
#include "utils.h"
#include "limbdark.h"
#include <chrono>

using namespace utils;

int main() {

    int lmax = 50;
    limbdark::GreensLimbDark<double> L(lmax);
    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < 100000; ++i)
        L.compute(0.5, 0.1);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cout << diff.count() << std::endl;

}
