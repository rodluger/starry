#include <stdlib.h>
#include <iostream>
#include <cmath>
#include "maps.h"
#include "utils.h"

using namespace utils;

template <class T>
class Body : public maps::Map<T> {

    protected:

        // Hide the Map's flux function from the user
        Row<T> flux();

    public:

        using maps::Map<T>::N;

        Body(int lmax=2, int nwav=1) : maps::Map<T>(lmax, nwav) {


        }

};


int main() {

    Body<Matrix<double>> map(2);
    //map.setY(1, 0, 1.0);
    //std::cout << map.flux() << std::endl;

}
