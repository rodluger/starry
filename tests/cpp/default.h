/**
Instantiate a default map and compute the flux
during an occultation.

*/

#ifndef _TEST_DEFAULT_H_
#define _TEST_DEFAULT_H_

#include "test.h"

namespace test_default {

    /**
    Compare the flux in double precision to
    the flux using multiprecision.

    */
    int test_flux(int nt=100) {

        using namespace starry2;

        // Instantiate the default map
        int lmax = 2;
        Map<Default<double>> map(lmax);
        
        // Give the star unit flux
        map.setY(0, 0, 1.0);

        // Set the linear and quadratic
        // limb darkening coefficients
        map.setU(1, 0.4);
        map.setU(2, 0.26);

        // Compute the flux
        Vector<double> b(nt); 
        b = Vector<double>::LinSpaced(nt, -1.5, 1.5);
        Vector<double> flux(nt);
        for (int t = 0; t < nt; ++t)
            map.computeFlux(0.0, b(t), 0.0, 0.1, flux.row(t));

        // -- Now do the same thing in multiprecision --

        // Instantiate the default map
        Map<Default<Multi>> map_multi(lmax);
        
        // Give the star unit flux
        map_multi.setY(0, 0, 1.0);

        // Set the linear and quadratic
        // limb darkening coefficients
        map_multi.setU(1, 0.4);
        map_multi.setU(2, 0.26);

        // Compute the flux
        Vector<Multi> flux_multi(nt);
        for (int t = 0; t < nt; ++t)
            map_multi.computeFlux(0.0, Multi(b(t)), 0.0, 0.1, flux_multi.row(t));

        // Compare
        int nerr = 0;
        if (!flux.isApprox(flux_multi.template cast<double>())) {
            std::cout << "Wrong flux in `test_flux`." << std::endl;
            ++nerr;
        }
        return nerr;
    }

    /**
    Compare the flux in double precision to
    the flux using multiprecision. Also
    compute and compare derivatives.

    */
    int test_flux_with_grad(int nt=100) {

        using namespace starry2;

        // Instantiate the default map
        int lmax = 2;
        Map<Default<double>> map(lmax);
        
        // Give the star unit flux
        map.setY(0, 0, 1.0);

        // Set the linear and quadratic
        // limb darkening coefficients
        map.setU(1, 0.4);
        map.setU(2, 0.26);

        // Compute the flux and the derivatives
        Vector<double> b(nt); 
        b = Vector<double>::LinSpaced(nt, -1.5, 1.5);
        Vector<double> flux(nt);
        Vector<double> dtheta(nt);
        Vector<double> dxo(nt);
        Vector<double> dyo(nt);
        Vector<double> dro(nt);
        Vector<double> dy(nt);
        Matrix<double> du(nt, lmax);
        for (int t = 0; t < nt; ++t)
            map.computeFlux(0.0, b(t), 0.0, 0.1, flux.row(t),
                            dtheta.row(t), dxo.row(t), dyo.row(t), 
                            dro.row(t), dy.row(t), 
                            du.row(t).transpose());

        // -- Now compute the derivatives numerically --

        // Instantiate the default map
        Map<Default<Multi>> map_multi(lmax);
        
        // Give the star unit flux
        map_multi.setY(0, 0, 1.0);

        // Set the linear and quadratic
        // limb darkening coefficients
        map_multi.setU(1, 0.4);
        map_multi.setU(2, 0.26);

        // Compute the derivatives
        Multi eps = 1.e-15;
        Vector<Multi> f1(1), f2(1);
        Vector<Multi> dtheta_multi(nt),
                    dxo_multi(nt),
                    dyo_multi(nt),
                    dro_multi(nt),
                    dy_multi(nt);
        Matrix<Multi> du_multi(nt, lmax);
        for (int t = 0; t < nt; ++t) {
            map_multi.computeFlux(0.0 - eps, Multi(b(t)), 0.0, 0.1, f1);
            map_multi.computeFlux(0.0 + eps, Multi(b(t)), 0.0, 0.1, f2);
            dtheta_multi(t) = (f2(0) - f1(0)) / (2 * eps);

            map_multi.computeFlux(0.0, Multi(b(t)) - eps, 0.0, 0.1, f1);
            map_multi.computeFlux(0.0, Multi(b(t)) + eps, 0.0, 0.1, f2);
            dxo_multi(t) = (f2(0) - f1(0)) / (2 * eps);

            map_multi.computeFlux(0.0, Multi(b(t)), 0.0 - eps, 0.1, f1);
            map_multi.computeFlux(0.0, Multi(b(t)), 0.0 + eps, 0.1, f2);
            dyo_multi(t) = (f2(0) - f1(0)) / (2 * eps);

            map_multi.computeFlux(0.0, Multi(b(t)), 0.0, 0.1 - eps, f1);
            map_multi.computeFlux(0.0, Multi(b(t)), 0.0, 0.1 + eps, f2);
            dro_multi(t) = (f2(0) - f1(0)) / (2 * eps);
        
            map_multi.setY(0, 0, 1.0 - eps);
            map_multi.computeFlux(0.0, Multi(b(t)), 0.0, 0.1, f1);
            map_multi.setY(0, 0, 1.0 + eps);
            map_multi.computeFlux(0.0, Multi(b(t)), 0.0, 0.1, f2);
            map_multi.setY(0, 0, 1.0);
            dy_multi(t) = (f2(0) - f1(0)) / (2 * eps);

            map_multi.setU(1, 0.4 - eps);
            map_multi.computeFlux(0.0, Multi(b(t)), 0.0, 0.1, f1);
            map_multi.setU(1, 0.4 + eps);
            map_multi.computeFlux(0.0, Multi(b(t)), 0.0, 0.1, f2);
            map_multi.setU(1, 0.4);
            du_multi(t, 0) = (f2(0) - f1(0)) / (2 * eps);

            map_multi.setU(2, 0.26 - eps);
            map_multi.computeFlux(0.0, Multi(b(t)), 0.0, 0.1, f1);
            map_multi.setU(2, 0.26 + eps);
            map_multi.computeFlux(0.0, Multi(b(t)), 0.0, 0.1, f2);
            map_multi.setU(2, 0.26);
            du_multi(t, 1) = (f2(0) - f1(0)) / (2 * eps);
        }

        // Compare
        int nerr = 0;
        if (!dtheta.isApprox(dtheta_multi.template cast<double>())) {
            std::cout << "Wrong theta deriv in `test_flux_with_grad`." << std::endl;
            ++nerr;
        }
        if (!dxo.isApprox(dxo_multi.template cast<double>())) {
            std::cout << "Wrong xo deriv in `test_flux_with_grad`." << std::endl;
            ++nerr;
        }
        if (!dyo.isApprox(dyo_multi.template cast<double>())) {
            std::cout << "Wrong yo deriv in `test_flux_with_grad`." << std::endl;
            ++nerr;
        }
        if (!dro.isApprox(dro_multi.template cast<double>())) {
            std::cout << "Wrong ro deriv in `test_flux_with_grad`." << std::endl;
            ++nerr;
        }
        if (!dy.isApprox(dy_multi.template cast<double>())) {
            std::cout << "Wrong y deriv in `test_flux_with_grad`." << std::endl;
            ++nerr;
        }
        if (!du.isApprox(du_multi.template cast<double>())) {
            std::cout << "Wrong u deriv in `test_flux_with_grad`." << std::endl;
            ++nerr;
        }
        return nerr;
    }

    int test () {
        return test_flux() + 
               test_flux_with_grad();
    }

} // namespace test_default
#endif