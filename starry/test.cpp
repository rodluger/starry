#include <stdlib.h>
#include "maps.h"
#include "utils.h"

int main() {

  int lmax = 5;
  double theta, xo, yo, ro;

  maps::Map<double> map(lmax);
  for (int l = 0; l < lmax + 1; l++) {
    for (int m = -l; m < l + 1; m++) {
      map.setCoeff(l, m, 1.0);
    }
  }


  theta = 0;
  xo = 0.3;
  yo = 0.3;
  ro = 0.1;
  for (int i = 0; i < 1000; i++)
    map.flux(theta, xo, yo, ro);

}
