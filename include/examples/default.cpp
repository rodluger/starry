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

    // Instantiate a default map
    int lmax = 2;
    Map<Default<double>> map(lmax);
    
    // Give the star unit flux
    map.setY(0, 0, 1.0);                                                       /**< This is important! Otherwise there will be no transit. */

    // Set the linear and quadratic
    // limb darkening coefficients
    map.setU(1, 0.4);
    map.setU(2, 0.26);

    // Inputs
    int npts = 5;                                                              /**< Number of light curve points */
    double theta = 0.0;                                                        /**< Angular phase of the map; doesn't matter for a purely limb-darkened body! */
    Vector<double> xo = Vector<double>::LinSpaced(npts, -1.2, 1.2);            /**< x position of occultor normalized to radius of occulted body */
    double yo = 0.0;                                                           /**< y position of occultor normalized to radius of occulted body */
    double ro = 0.1;                                                           /**< Radius of occultor normalized to radius of occulted body */

    // Outputs
    Vector<double> flux(npts);                                                 /**< The flux for each value of `xo` */
    Vector<double> Dtheta(npts);                                               /**< The derivatives of the flux with respect to each of the inputs */
    Vector<double> Dxo(npts);
    Vector<double> Dyo(npts);
    Vector<double> Dro(npts);
    Matrix<double> Dy(npts, 1);
    Matrix<double> Du(npts, lmax);

    // NOTE: `df/dy` and `df/du` are **matrices**, whose rows are the 
    // derivatives of the flux with respect to each of the coefficients in `y` 
    // and `u`, respectively. HOWEVER, for purely limb-darkend maps, starry 
    // does not compute all of the terms in `df/dy`, since the user probably 
    // doesn't care about them (and we can save lots of CPU cycles this way). 
    // So while `Du` has `lmax` rows (the number of limb darkening 
    // coefficients), we construct `Dy` with only a *single* row: the
    // derivative with respect to the constant `Y_{0,0}` term (which starry 
    // **always** computes). In the example below, where we actually care about 
    // the `Y_{l,m}` coefficients, `Dy` has `(lmax + 1) * (lmax + 1)` rows.

    // Compute the light curve. Note that if you don't 
    // care about the gradients, you can simply omit 
    // all of them from the function call. This will 
    // lead to slightly faster run time.
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
            Dy.row(t).transpose(), 
            Du.row(t).transpose()
        );

    // Print the light curve
    std::cout << "Limb-darkened transit:" << std::endl;
    std::cout << "f:" << std::endl << flux.transpose() << std::endl;

    // Print some of the derivatives
    std::cout << "df/dxo:" << std::endl << Dxo.transpose() << std::endl;
    std::cout << "df/dro:" << std::endl << Dro.transpose() << std::endl;
    std::cout << "df/du:" << std::endl << Du.transpose() << std::endl;
    std::cout << std::endl;
    
}

/**
Compute the light curve for a transit across a
body whose surface is described by a 5-th degree
spherical harmonic.

*/
void spherical_harmonic_transit() {

    // Instantiate a default map
    int lmax = 5;
    Map<Default<double>> map(lmax);
    
    // Give the planet a random isotropic
    // map with unit power at all scales.
    // Note that we could also set all the 
    // Y_{l,m} coefficients individually
    // via `map.setY(l, m, value)`.
    int seed = 42;
    Vector<double> power = Vector<double>::Ones(lmax + 1);
    map.random(power, seed);

    // Inputs
    int npts = 5;                                                              /**< Number of light curve points */
    Vector<double> theta = Vector<double>::LinSpaced(npts, 0, 30);             /**< The occulted body rotates from 0 to 30 degrees over the observation window */
    Vector<double> xo = Vector<double>::LinSpaced(npts, -1.2, 1.2);            /**< x position of occultor normalized to radius of occulted body */
    double yo = 0.0;                                                           /**< y position of occultor normalized to radius of occulted body */
    double ro = 0.1;                                                           /**< Radius of occultor normalized to radius of occulted body */

    // Outputs
    Vector<double> flux(npts);                                                 /**< The flux for each value of `xo` */
    Vector<double> Dtheta(npts);                                               /**< The derivatives of the flux with respect to each of the inputs */
    Vector<double> Dxo(npts);
    Vector<double> Dyo(npts);
    Vector<double> Dro(npts);
    Matrix<double> Dy(npts, (lmax + 1) * (lmax + 1));                          /**< Note the shape! */
    Matrix<double> Du(npts, 0);                                                /**< Note the shape! */

    // NOTE: See comment in `limb_darkened_transit()` above. Here we
    // construct `Dy` with as many rows as there are spherical harmonic
    // coefficients, and construct `Du` with **zero** rows, since starry
    // dos not compute any derivs with respect to `u` if none of the 
    // coefficients are set. We **could** construct `Du` with `lmax`
    // rows as above, but starry will just set them to zero (which can be
    // a tiny speed hit).

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
            Dy.row(t).transpose(), 
            Du.row(t).transpose()
        );

    // Print the light curve
    std::cout << "Spherical harmonic transit:" << std::endl;
    std::cout << "f:" << std::endl << flux.transpose() << std::endl;

    // Print some of the derivatives
    std::cout << "df/dtheta:" << std::endl << Dtheta.transpose() << std::endl;
    std::cout << "df/dxo:" << std::endl << Dxo.transpose() << std::endl;
    std::cout << "df/dro:" << std::endl << Dro.transpose() << std::endl;
    std::cout << "df/dy:" << std::endl << Dy.transpose() << std::endl;

}

int main() {
    limb_darkened_transit();
    spherical_harmonic_transit();
}