/**
Instantiate a default map and compute the flux
during an occultation.

*/

#ifndef _TEST_DEFAULT_H_
#define _TEST_DEFAULT_H_

#include "test.h"

namespace test_default {

    /**
    Compare the limb-darkened flux in double precision to
    the flux using multiprecision.

    */
    int test_flux_ld(int nt=100) {

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

        // Compute the occultation flux
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
            std::cout << "Wrong flux in `test_flux_ld`." << std::endl;
            ++nerr;
        }
        return nerr;
    }

    /**
    Compare the spherical harmonic flux in double precision to
    the flux using multiprecision.

    TODO: Compute the *occultation* flux as well.

    */
    int test_flux_ylm(int nt=100) {

        using namespace starry2;

        // Instantiate the default map
        int lmax = 2;
        Map<Default<double>> map(lmax);
        
        // Set all of the Ylm coefficients
        Vector<double> y = Vector<double>::Ones((lmax + 1) * (lmax + 1));
        map.setY(y);

        // Compute the phase curve 
        Vector<double> theta = Vector<double>::LinSpaced(nt, 0, 360);
        Vector<double> flux(nt);
        for (int t = 0; t < nt; ++t)
            map.computeFlux(theta(t), -1.0, -1.0, 0.1, flux.row(t));

        // -- Now do the same thing in multiprecision --

        // Instantiate the default map
        Map<Default<Multi>> map_multi(lmax);
        
        // Set the coefficients
        map_multi.setY(y.template cast<Multi>());

        // Compute the flux
        Vector<Multi> flux_multi(nt);
        for (int t = 0; t < nt; ++t)
            map_multi.computeFlux(Multi(theta(t)), -1.0, -1.0, 0.1, flux_multi.row(t));

        // Compare
        int nerr = 0;
        if (!flux.isApprox(flux_multi.template cast<double>())) {
            std::cout << "Wrong flux in `test_flux_ylm`." << std::endl;
            ++nerr;
        }
        return nerr;
    }

    /**
    Compare the flux in double precision for a limb-darkened map to
    the flux using multiprecision. Also compute and compare derivatives.

    */
    int test_grad_ld(int nt=100) {

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
        Vector<double> Dtheta(nt);
        Vector<double> Dxo(nt);
        Vector<double> Dyo(nt);
        Vector<double> Dro(nt);
        Vector<double> Dy(nt);
        Matrix<double> Du(nt, lmax);
        for (int t = 0; t < nt; ++t)
            map.computeFlux(0.0, b(t), 0.0, 0.1, flux.row(t),
                            Dtheta.row(t), Dxo.row(t), Dyo.row(t), 
                            Dro.row(t), Dy.row(t), 
                            Du.row(t).transpose());

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
        if (!Dtheta.isApprox(dtheta_multi.template cast<double>())) {
            std::cout << "Wrong theta deriv in `test_grad_ld`." << std::endl;
            ++nerr;
        }
        if (!Dxo.isApprox(dxo_multi.template cast<double>())) {
            std::cout << "Wrong xo deriv in `test_grad_ld`." << std::endl;
            ++nerr;
        }
        if (!Dyo.isApprox(dyo_multi.template cast<double>())) {
            std::cout << "Wrong yo deriv in `test_grad_ld`." << std::endl;
            ++nerr;
        }
        if (!Dro.isApprox(dro_multi.template cast<double>())) {
            std::cout << "Wrong ro deriv in `test_grad_ld`." << std::endl;
            ++nerr;
        }
        if (!Dy.isApprox(dy_multi.template cast<double>())) {
            std::cout << "Wrong y deriv in `test_grad_ld`." << std::endl;
            ++nerr;
        }
        if (!Du.isApprox(du_multi.template cast<double>())) {
            std::cout << "Wrong u deriv in `test_grad_ld`." << std::endl;
            ++nerr;
        }
        return nerr;
    }

    /**
    Compare the flux in double precision for a spherical harmonic map to
    the flux using multiprecision. Also compute and compare derivatives.

    TODO: Also compute the occultation flux.

    */
    int test_grad_ylm(int nt=100) {

        using namespace starry2;

        // Instantiate the default map
        int lmax = 2;
        Map<Default<double>> map(lmax);
        
        // Set all of the Ylm coefficients
        Vector<double> y = Vector<double>::Ones((lmax + 1) * (lmax + 1));
        map.setY(y);

        // Compute the flux and the derivatives
        Vector<double> theta = Vector<double>::LinSpaced(nt, 0, 360);
        Vector<double> flux(nt);
        Vector<double> Dtheta(nt);
        Vector<double> Dxo(nt);
        Vector<double> Dyo(nt);
        Vector<double> Dro(nt);
        Vector<double> Du(nt);
        Matrix<double> Dy(nt, (lmax + 1) * (lmax + 1));
        for (int t = 0; t < nt; ++t)
            map.computeFlux(theta(t), -1.0, -1.0, 0.1, flux.row(t),
                            Dtheta.row(t), Dxo.row(t), Dyo.row(t), 
                            Dro.row(t), Dy.row(t).transpose(), 
                            Du.row(t));

        // -- Now compute the derivatives numerically --

        // Instantiate the default map
        Map<Default<Multi>> map_multi(lmax);
        
        // Set the coefficients
        map_multi.setY(y.template cast<Multi>());

        // Compute the derivatives
        Multi eps = 1.e-15;
        Vector<Multi> f1(1), f2(1);
        Vector<Multi> dtheta_multi(nt),
                      dxo_multi(nt),
                      dyo_multi(nt),
                      dro_multi(nt);
        Matrix<Multi> dy_multi(nt, (lmax + 1) * (lmax + 1));
        for (int t = 0; t < nt; ++t) {
            map_multi.computeFlux(Multi(theta(t) - eps), -1.0, -1.0, 0.1, f1);
            map_multi.computeFlux(Multi(theta(t) + eps), -1.0, -1.0, 0.1, f2);
            dtheta_multi(t) = (f2(0) - f1(0)) / (2 * eps);

            map_multi.computeFlux(Multi(theta(t)), -1.0 - eps, -1.0, 0.1, f1);
            map_multi.computeFlux(Multi(theta(t)), -1.0 + eps, -1.0, 0.1, f2);
            dxo_multi(t) = (f2(0) - f1(0)) / (2 * eps);

            map_multi.computeFlux(Multi(theta(t)), -1.0, -1.0 - eps, 0.1, f1);
            map_multi.computeFlux(Multi(theta(t)), -1.0, -1.0 + eps, 0.1, f2);
            dyo_multi(t) = (f2(0) - f1(0)) / (2 * eps);

            map_multi.computeFlux(Multi(theta(t)), -1.0, -1.0, 0.1 - eps, f1);
            map_multi.computeFlux(Multi(theta(t)), -1.0, -1.0, 0.1 + eps, f2);
            dro_multi(t) = (f2(0) - f1(0)) / (2 * eps);

            int n = 0;
            for (int l = 0; l < lmax + 1; ++l) {
                for (int m = -l; m < l + 1; ++m) {
                    map_multi.setY(l, m, 1.0 - eps);
                    map_multi.computeFlux(Multi(theta(t)), -1.0, -1.0, 0.1, f1);
                    map_multi.setY(l, m, 1.0 + eps);
                    map_multi.computeFlux(Multi(theta(t)), -1.0, -1.0, 0.1, f2);
                    map_multi.setY(l, m, 1.0);
                    dy_multi(t, n) = (f2(0) - f1(0)) / (2 * eps);
                    ++n;
                }
            }
            
        }

        // Compare
        int nerr = 0;
        if (!Dtheta.isApprox(dtheta_multi.template cast<double>())) {
            std::cout << "Wrong theta deriv in `test_grad_ylm`." << std::endl;
            ++nerr;
        }
        if (!Dxo.isApprox(dxo_multi.template cast<double>())) {
            std::cout << "Wrong xo deriv in `test_grad_ylm`." << std::endl;
            ++nerr;
        }
        if (!Dyo.isApprox(dyo_multi.template cast<double>())) {
            std::cout << "Wrong yo deriv in `test_grad_ylm`." << std::endl;
            ++nerr;
        }
        if (!Dro.isApprox(dro_multi.template cast<double>())) {
            std::cout << "Wrong ro deriv in `test_grad_ylm`." << std::endl;
            ++nerr;
        }
        if (!Dy.isApprox(dy_multi.template cast<double>())) {
            std::cout << "Wrong y deriv in `test_grad_ylm`." << std::endl;
            ++nerr;
        }
        return nerr;
    }

    int test () {
        return test_flux_ld() + 
               test_grad_ld() +
               test_flux_ylm() + 
               test_grad_ylm();
    }

} // namespace test_default
#endif