#include <stdlib.h>
#include <iostream>
#include "maps.h"
#include "utils.h"
using namespace utils;

int main() {

    int lmax = 2;
    int nwav = 1;
    maps::Map<double> map(lmax, nwav);

    for (int l = 0; l < lmax + 1; ++l) {
        for (int m = -l; m < l + 1; ++m) {
            map.setYlm(l, m, Vector<double>::Constant(1, 1));
        }
    }

    int nt = 100000;
    Vector<double> theta = Vector<double>::LinSpaced(nt, 0, 45);
    Vector<double> xo = Vector<double>::LinSpaced(nt, -1.2, 1.15);
    Vector<double> yo = Vector<double>::LinSpaced(nt, -0.2, 0.33);
    double ro = 0.1;

    for (size_t t = 0; t < nt; ++t) {
        map.flux(0, xo(t), yo(t), ro, false);
    }

}
