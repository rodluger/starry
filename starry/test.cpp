#include <stdlib.h>
#include "maps.h"
#include "utils.h"

int main() {

  int lmax = 5;
  double theta, xo, yo, ro;

  maps::Map<double> map(lmax);
  for (int l = 0; l < lmax + 1; l++) {
    for (int m = -l; m < l + 1; m++) {
      map.set_coeff(l, m, 1.0);
    }
  }


  theta = 0;
  xo = 0.3;
  yo = 0.3;
  ro = 0.1;
  for (int i = 0; i < 100; i++)
    map.flux(yhat, theta, xo, yo, ro);

}
