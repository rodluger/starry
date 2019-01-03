#include "test.h"
#include "default.h"
#include "spectral.h"
#include "temporal.h"

int main() {

    int nerr = 0;

    std::cout << "Testing default map...." << std::endl;
    nerr += test_default::test();

    std::cout << "Testing spectral map...." << std::endl;
    nerr += test_spectral::test();

    std::cout << "Testing temporal map...." << std::endl;
    nerr += test_temporal::test();

    std::cout << "Completed with " << nerr << " error(s)." << std::endl;
    return nerr;

}