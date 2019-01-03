/**
Instantiate a spectral map and compute the flux
during an occultation.

*/

#ifndef _TEST_SPECTRAL_H_
#define _TEST_SPECTRAL_H_

#include "test.h"

namespace test_spectral {

    /**
    Compare the flux in double precision to
    the flux using multiprecision.

    */
    int test_flux(int nt=100) {

        using namespace starry2;

        // Instantiate a spectral map
        // with 3 wavelength bins
        int nw = 3;
        int lmax = 2;
        Map<Spectral<double>> map(lmax, nw);
        
        // Give the star unit flux at 
        // all wavelengths
        map.setY(0, 0, 1.0);

        // Set the linear and quadratic
        // limb darkening coefficients
        Vector<double> u1(nw), u2(nw);
        u1 << 1.0, 0.0, 0.4;
        u2 << 0.0, 1.0, 0.26;
        map.setU(1, u1);
        map.setU(2, u2);

        // Compute the flux
        Vector<double> b(nt); 
        b = Vector<double>::LinSpaced(nt, -1.5, 1.5);
        Matrix<double> flux(nt, nw);
        for (int t = 0; t < nt; ++t)
            map.computeFlux(0.0, b(t), 0.0, 0.1, flux.row(t));

        // -- Now do the same thing in multiprecision --

        // Instantiate the default map
        Map<Spectral<Multi>> map_multi(lmax, nw);
        
        // Give the star unit flux
        map_multi.setY(0, 0, 1.0);

        // Set the linear and quadratic
        // limb darkening coefficients
        map_multi.setU(1, u1.template cast<Multi>());
        map_multi.setU(2, u2.template cast<Multi>());

        // Compute the flux
        Matrix<Multi> flux_multi(nt, nw);
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

        // Instantiate a spectral map
        // with 3 wavelength bins
        int nw = 3;
        int lmax = 2;
        Map<Spectral<double>> map(lmax, nw);
        
        // Give the star unit flux
        map.setY(0, 0, 1.0);

        // Set the linear and quadratic
        // limb darkening coefficients
        Vector<double> u1(nw), u2(nw);
        u1 << 1.0, 0.0, 0.4;
        u2 << 0.0, 1.0, 0.26;
        map.setU(1, u1);
        map.setU(2, u2);

        // Compute the flux and the derivatives
        Vector<double> b(nt); 
        b = Vector<double>::LinSpaced(nt, -1.5, 1.5);
        Matrix<double> flux(nt, nw);
        Matrix<double> dtheta(nt, nw);
        Matrix<double> dxo(nt, nw);
        Matrix<double> dyo(nt, nw);
        Matrix<double> dro(nt, nw);
        Matrix<double> dy(nt, nw);
        Matrix<double> du(nt * lmax, nw);
        for (int t = 0; t < nt; ++t)
            map.computeFlux(0.0, b(t), 0.0, 0.1, flux.row(t),
                            dtheta.row(t), dxo.row(t), dyo.row(t), 
                            dro.row(t), dy.row(t), 
                            du.block(t * lmax, 0, lmax, nw));

        // -- Now compute the derivatives numerically --

        // Instantiate the default map
        Map<Spectral<Multi>> map_multi(lmax, nw);
        
        // Give the star unit flux
        map_multi.setY(0, 0, 1.0);

        // Set the linear and quadratic
        // limb darkening coefficients
        map_multi.setU(1, u1.template cast<Multi>());
        map_multi.setU(2, u2.template cast<Multi>());

        // Compute the derivatives
        Multi eps = 1.e-15;
        Vector<Multi> eps3(3);
        eps3.setConstant(eps);
        RowVector<Multi> f1(nw), f2(nw);
        Matrix<Multi> dtheta_multi(nt, nw),
                    dxo_multi(nt, nw),
                    dyo_multi(nt, nw),
                    dro_multi(nt, nw),
                    dy_multi(nt, nw);
        Matrix<Multi> du_multi(nt * lmax, nw);
        for (int t = 0; t < nt; ++t) {
            map_multi.computeFlux(0.0 - eps, Multi(b(t)), 0.0, 0.1, f1);
            map_multi.computeFlux(0.0 + eps, Multi(b(t)), 0.0, 0.1, f2);
            dtheta_multi.row(t) = (f2 - f1) / (2 * eps);

            map_multi.computeFlux(0.0, Multi(b(t)) - eps, 0.0, 0.1, f1);
            map_multi.computeFlux(0.0, Multi(b(t)) + eps, 0.0, 0.1, f2);
            dxo_multi.row(t) = (f2 - f1) / (2 * eps);

            map_multi.computeFlux(0.0, Multi(b(t)), 0.0 - eps, 0.1, f1);
            map_multi.computeFlux(0.0, Multi(b(t)), 0.0 + eps, 0.1, f2);
            dyo_multi.row(t) = (f2 - f1) / (2 * eps);

            map_multi.computeFlux(0.0, Multi(b(t)), 0.0, 0.1 - eps, f1);
            map_multi.computeFlux(0.0, Multi(b(t)), 0.0, 0.1 + eps, f2);
            dro_multi.row(t) = (f2 - f1) / (2 * eps);
        
            map_multi.setY(0, 0, 1.0 - eps);
            map_multi.computeFlux(0.0, Multi(b(t)), 0.0, 0.1, f1);
            map_multi.setY(0, 0, 1.0 + eps);
            map_multi.computeFlux(0.0, Multi(b(t)), 0.0, 0.1, f2);
            map_multi.setY(0, 0, 1.0);
            dy_multi.row(t) = (f2 - f1) / (2 * eps);

            map_multi.setU(1, u1.template cast<Multi>() - eps3);
            map_multi.computeFlux(0.0, Multi(b(t)), 0.0, 0.1, f1);
            map_multi.setU(1, u1.template cast<Multi>() + eps3);
            map_multi.computeFlux(0.0, Multi(b(t)), 0.0, 0.1, f2);
            map_multi.setU(1, u1.template cast<Multi>());
            du_multi.block(t * lmax, 0, 1, nw) = (f2 - f1) / (2 * eps);

            map_multi.setU(2, u2.template cast<Multi>() - eps3);
            map_multi.computeFlux(0.0, Multi(b(t)), 0.0, 0.1, f1);
            map_multi.setU(2, u2.template cast<Multi>() + eps3);
            map_multi.computeFlux(0.0, Multi(b(t)), 0.0, 0.1, f2);
            map_multi.setU(2, u2.template cast<Multi>());
            du_multi.block(t * lmax + 1, 0, 1, nw) = (f2 - f1) / (2 * eps);
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

} // namespace test_spectral
#endif