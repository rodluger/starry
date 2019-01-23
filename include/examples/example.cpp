#include <stdlib.h>
#include <iostream>

// Eigen stuff
#include <Eigen/Core>
template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T>
using RowVector = Eigen::Matrix<T, 1, Eigen::Dynamic>;
template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

// Import starry
#include "starry2.h"
using namespace starry2;

void compute_default() {

    // Instantiate a default map
    int lmax = 2;
    Map<Default<double>> map(lmax);
    
    // Give the star unit flux
    map.setY(0, 0, 1.0);

    // Set the linear and quadratic
    // limb darkening coefficients
    map.setU(1, 0.4);
    map.setU(2, 0.26);

    // Inputs
    int nt = 10;
    double theta = 0.0;
    Vector<double> xo = Vector<double>::LinSpaced(nt, -1.5, 1.5);
    double yo = 0.0;
    double ro = 0.1;

    // Outputs
    Vector<double> flux(nt);
    Vector<double> Dtheta(nt);
    Vector<double> Dxo(nt);
    Vector<double> Dyo(nt);
    Vector<double> Dro(nt);
    Matrix<double> Dy(nt, (lmax + 1) * (lmax + 1));
    Matrix<double> Du(nt, lmax);

    // Compute
    for (int t = 0; t < nt; ++t)
        map.computeFlux(theta, xo(t), yo, ro, flux.row(t),
                        Dtheta.row(t), Dxo.row(t), Dyo.row(t), 
                        Dro.row(t), 
                        Dy.row(t).transpose(), 
                        Du.row(t).transpose());

    // Print
    std::cout << flux.transpose() << std::endl;

}

int main() {
    compute_default();
}