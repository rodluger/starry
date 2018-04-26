// Disable autodiff for these tests
#ifndef STARRY_NO_AUTODIFF
#define STARRY_NO_AUTODIFF                      1
#endif

#include <iostream>
#include <iomanip>
#include <Eigen/Core>
#include "constants.h"
#include "ellip.h"
#include "maps.h"
#include "basis.h"
#include "fact.h"
#include "sqrtint.h"
#include "rotation.h"
#include "solver.h"
#include "orbital.h"

using namespace std;

int main() {

    // Instantiate a star and a planet, just to check the constructors
    orbital::Planet<double> planet = orbital::Planet<double>(8);
    orbital::Star<double> star = orbital::Star<double>(8);

    // Manipulate the planet's map
    maps::Map<double>& y = planet.map;
    y.set_coeff(2, 0, 1);
    cout << y.repr() << endl;

    // Compute the occultation flux
    int npts = 20;
    double r = 0.25;
    double y0 = 0.25;
    double diff;
    Eigen::VectorXd x0 = Eigen::VectorXd::LinSpaced(npts, -1.5, 1.5);
    Eigen::VectorXd flux = Eigen::VectorXd::Zero(npts);
    Eigen::VectorXd numflux = Eigen::VectorXd::Zero(npts);

    cout << endl;
    cout << setw(12) << "Analytic" << "     "
         << setw(12) << "Numerical" << "    "
         << setw(12) << "Difference" << endl;
         cout << setw(12) << "--------" << "     "
              << setw(12) << "---------" << "    "
              << setw(12) << "----------" << endl;
    for (int i = 0; i < npts; i ++) {
        flux(i) = y.flux(maps::yhat, 0, x0(i), y0, r);
        numflux(i) = y.flux(maps::yhat, 0, x0(i), y0, r, true, 1e-4);
        diff = (flux(i) - numflux(i));
        cout << setw(12) << flux(i) << "     "
             << setw(12) << numflux(i) << "    "
             << setw(12) << diff << endl;
    }

    return 0;
}
