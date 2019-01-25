#include <stdlib.h>
#include <iostream>

// Import eigen and define some useful aliases
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

/**
Compute the light curve for a transit across a
body whose surface is described by a 5-th degree
spherical harmonic that varies in time.

*/
void spherical_harmonic_transit() {

    // Instantiate a temporal map of spherical harmonic degree
    // 5 with 2 time columns: the base map and its first time
    // derivative.
    int lmax = 5;
    int nt = 2;
    Map<Temporal<double>> map(lmax, nt);
    
    // Give the planet a random isotropic base map with unit power 
    // at all scales. Do the same thing for the map derivatives,
    // but weighted by 0.1 so the base map contribution dominates. 
    int seed1 = 42, 
        seed2 = 17;
    Vector<double> power = Vector<double>::Ones(lmax + 1);
    map.random(power, seed1, 0);
    map.random(0.1 * power, seed2, 1);

    // Inputs
    int npts = 5;                                                              /**< Number of light curve points */
    Vector<double> time = Vector<double>::LinSpaced(npts, 0, 1);               /**< Vary the time (in arbitrary units) from 0 to 1 over the observation window */
    Vector<double> theta = Vector<double>::LinSpaced(npts, 0, 30);             /**< The occulted body rotates from 0 to 30 degrees over the observation window */
    Vector<double> xo = Vector<double>::LinSpaced(npts, -1.2, 1.2);            /**< x position of occultor normalized to radius of occulted body */
    double yo = 0.0;                                                           /**< y position of occultor normalized to radius of occulted body */
    double ro = 0.1;                                                           /**< Radius of occultor normalized to radius of occulted body */

    // Outputs
    Vector<double> flux(npts);                                                 /**< The flux for each value of `xo` */
    Vector<double> Dtime(npts);                                                /**< The derivatives of the flux with respect to each of the inputs */
    Vector<double> Dtheta(npts);
    Vector<double> Dxo(npts);
    Vector<double> Dyo(npts);
    Vector<double> Dro(npts);
    Matrix<double> Dy(npts * (lmax + 1) * (lmax + 1), nt);                     /**< Note the shape! */
    Matrix<double> Du(npts, 0);                                                /**< We don't care about the derivs with respect to limb darkening */

    // NOTE: The derivative with respect to the map coefficients `y`
    // is now a rank-3 tensor of dimensions 
    //      (# of points, # of coeffs, # of time components)
    // which we need to collapse to a 2D matrix, hence the wonky
    // shape `(npts * (lmax + 1) * (lmax + 1), nt)` above.

    // Compute the light curve and the gradient.
    for (int t = 0; t < npts; ++t)
        map.computeFlux(
            time(t),
            theta(t), 
            xo(t), 
            yo, 
            ro, 
            flux.row(t),
            Dtime.row(t),
            Dtheta.row(t), 
            Dxo.row(t), 
            Dyo.row(t), 
            Dro.row(t), 
            Dy.block(                                                          
                t * (lmax + 1) * (lmax + 1),                                   // We need to pass in a **matrix** to starry at each timestep
                0,                                                             // of shape (number of coefficients, number of time components)
                (lmax + 1) * (lmax + 1),                                       // so this requires a little data reshaping.
                nt
            ),
            Du.row(t).transpose()
        );

    // Print the light curve
    std::cout << "Spherical harmonic transit:" << std::endl;
    std::cout << "f:" << std::endl << flux.transpose() << std::endl;

    // Print some of the derivatives
    std::cout << "df/dt:" << std::endl << Dtime.transpose() << std::endl;
    std::cout << "df/dtheta:" << std::endl << Dtheta.transpose() << std::endl;
    std::cout << "df/dxo:" << std::endl << Dxo.transpose() << std::endl;
    std::cout << "df/dro:" << std::endl << Dro.transpose() << std::endl;
    std::cout << "df/dy:" << std::endl;
    for (int t = 0; t < npts; ++t)
        std::cout << Dy.block(t * (lmax + 1) * (lmax + 1), 0, 
            (lmax + 1) * (lmax + 1), nt) << std::endl << std::endl;

}

int main() {
    spherical_harmonic_transit();
}