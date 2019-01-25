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
simple quadratically limb-darkened star.

*/
void limb_darkened_transit() {

    // Instantiate a spectral map
    // with 3 wavelength bins
    int lmax = 2;
    int nw = 3;
    Map<Spectral<double>> map(lmax, nw);
    
    // Give the star unit flux
    // in all three wavelength bins
    map.setY(0, 0, Vector<double>::Ones(nw));                                  /**< This is important! Otherwise there will be no transit. */

    // Set the linear and quadratic
    // limb darkening coefficients
    Vector<double> u1(nw), u2(nw);
    u1 << 0.4, 0.3, 0.2;
    u2 << 0.26, 0.16, 0.06;
    map.setU(1, u1);
    map.setU(2, u2);

    // Inputs
    int npts = 5;                                                              /**< Number of light curve points */
    double theta = 0.0;                                                        /**< Angular phase of the map; doesn't matter for a purely limb-darkened body! */
    Vector<double> xo = Vector<double>::LinSpaced(npts, -1.2, 1.2);            /**< x position of occultor normalized to radius of occulted body */
    double yo = 0.0;                                                           /**< y position of occultor normalized to radius of occulted body */
    double ro = 0.1;                                                           /**< Radius of occultor normalized to radius of occulted body */

    // Outputs
    Matrix<double> flux(npts, nw);                                             /**< The flux for each value of `xo` in each wavelength bin */  
    Matrix<double> Dtheta(npts, nw);                                           /**< The flux derivatives */
    Matrix<double> Dxo(npts, nw);
    Matrix<double> Dyo(npts, nw);
    Matrix<double> Dro(npts, nw);
    Matrix<double> Dy(npts * 1, nw);
    Matrix<double> Du(npts * lmax, nw);

    // Compute the light curve and the gradient.
    for (int t = 0; t < npts; ++t)
        map.computeFlux(
            theta, 
            xo(t), 
            yo, 
            ro, 
            flux.row(t),
            Dtheta.row(t), 
            Dxo.row(t), 
            Dyo.row(t), 
            Dro.row(t), 
            Dy.row(t), 
            Du.block(t * lmax, 0, lmax, nw)
        );

    // Print the light curve
    std::cout << "Limb-darkened transit:" << std::endl;
    std::cout << "f:" << std::endl << flux.transpose() << std::endl;

    // Print some of the derivatives
    std::cout << "df/dxo:" << std::endl << Dxo.transpose() << std::endl;
    std::cout << "df/dro:" << std::endl << Dro.transpose() << std::endl;
    std::cout << "df/du:" << std::endl;
    for (int t = 0; t < npts; ++t)
        std::cout << Du.block(t * lmax, 0, lmax, nw) << std::endl << std::endl;
    
}

/**
Compute the light curve for a transit across a
body whose surface is described by a 5-th degree
spherical harmonic.

*/
void spherical_harmonic_transit() {

    // Instantiate a spectral map 
    // with 3 wavelength bins
    int lmax = 5;
    int nw = 3;
    Map<Spectral<double>> map(lmax, nw);
    
    // Give the planet a random isotropic
    // map with unit power at all scales.
    // We'll set completely different
    // coefficients at each wavelength bin.
    int seed1 = 42, 
        seed2 = 17,
        seed3 = 12;
    Vector<double> power = Vector<double>::Ones(lmax + 1);
    map.random(power, seed1, 0);
    map.random(power, seed2, 1);
    map.random(power, seed3, 2);

    // Inputs
    int npts = 5;                                                              /**< Number of light curve points */
    Vector<double> theta = Vector<double>::LinSpaced(npts, 0, 30);             /**< The occulted body rotates from 0 to 30 degrees over the observation window */
    Vector<double> xo = Vector<double>::LinSpaced(npts, -1.2, 1.2);            /**< x position of occultor normalized to radius of occulted body */
    double yo = 0.0;                                                           /**< y position of occultor normalized to radius of occulted body */
    double ro = 0.1;                                                           /**< Radius of occultor normalized to radius of occulted body */

    // Outputs
    Matrix<double> flux(npts, nw);                                             /**< The flux for each value of `xo` in each wavelength bin */  
    Matrix<double> Dtheta(npts, nw);                                           /**< The flux derivatives */
    Matrix<double> Dxo(npts, nw);
    Matrix<double> Dyo(npts, nw);
    Matrix<double> Dro(npts, nw);
    Matrix<double> Dy(npts * (lmax + 1) * (lmax + 1), nw);
    Matrix<double> Du(npts, nw);

    // Compute the light curve and the gradient.
    for (int t = 0; t < npts; ++t)
        map.computeFlux(
            theta(t), 
            xo(t), 
            yo, 
            ro, 
            flux.row(t),
            Dtheta.row(t), 
            Dxo.row(t), 
            Dyo.row(t), 
            Dro.row(t), 
            Dy.block(
                t * (lmax + 1) * (lmax + 1), 
                0, 
                (lmax + 1) * (lmax + 1), 
                nw
            ), 
            Du.row(t)
        );

    // Print the light curve
    std::cout << "Spherical harmonic transit:" << std::endl;
    std::cout << "f:" << std::endl << flux.transpose() << std::endl;

    // Print some of the derivatives
    std::cout << "df/dtheta:" << std::endl << Dtheta.transpose() << std::endl;
    std::cout << "df/dxo:" << std::endl << Dxo.transpose() << std::endl;
    std::cout << "df/dro:" << std::endl << Dro.transpose() << std::endl;
    std::cout << "df/dy:" << std::endl;
    for (int t = 0; t < npts; ++t)
        std::cout << Dy.block(t * (lmax + 1) * (lmax + 1), 0, 
            (lmax + 1) * (lmax + 1), nw) << std::endl << std::endl;

}

int main() {
    limb_darkened_transit();
    spherical_harmonic_transit();
}