#define STARRY_NO_AUTODIFF 1
#include <iostream>
#include <Eigen/Core>
#include "ellip.h"
#include "maps.h"
#include "basis.h"
#include "fact.h"
#include "sqrtint.h"
#include "rotation.h"
#include "solver.h"

using namespace std;

int main() {

    // Generate a map
    maps::Map<double> y = maps::Map<double>(2);
    y.set_coeff(2, 1, 1);
    cout << y.repr() << endl;

    // Compute the occultation flux
    int npts = 20;
    double r = 0.25;
    double y0 = 0.25;
    Eigen::VectorXd x0 = Eigen::VectorXd::LinSpaced(npts, -1.5, 1.5);
    Eigen::VectorXd flux = Eigen::VectorXd::Zero(npts);

    for (int i = 0; i < npts; i ++) {
        flux(i) = y.flux(maps::yhat, 0, x0(i), y0, r);
    }

    cout << flux << endl;

    return 0;
}
