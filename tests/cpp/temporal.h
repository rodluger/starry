/**
Instantiate a temporal map and compute the flux
during an occultation.

*/

#ifndef _TEST_TEMPORAL_H_
#define _TEST_TEMPORAL_H_

#include "test.h"

namespace test_temporal {

    /**
    Compare the limb-darkened flux in double precision to
    the flux using multiprecision.

    */
    int test_flux_ld(int nb=100) {

        using namespace starry2;

        // Instantiate a temporal map with 3 time columns
        int lmax = 2;
        int nt = 3;
        Map<Temporal<double>> map(lmax, nt);
        
        // Give the star a time-variable total flux
        Vector<double> y00(nt);
        y00 << 1.0, -0.5, 0.25;
        map.setY(0, 0, y00);

        // Set the linear and quadratic
        // limb darkening coefficients;
        // these are constant in time
        map.setU(1, 0.4);
        map.setU(2, 0.26);

        // Compute the flux at t = 0.5, but for a varying
        // impact parameter
        double t = 0.5;
        Vector<double> b(nb); 
        b = Vector<double>::LinSpaced(nb, -1.5, 1.5);
        Vector<double> flux(nb);
        for (int i = 0; i < nb; ++i)
            map.computeFlux(t, 0.0, b(i), 0.0, 0.1, flux.row(i));

        // -- Compare to a static map --

        // At time t, the y00 coefficient should be...
        double y00t = y00(0) + y00(1) * t + 0.5 * y00(2) * t * t;

        // Let's check that we get the same flux from a static map:
        Map<Default<double>> map_static(lmax);
        map_static.setY(0, 0, y00t);
        map_static.setU(1, 0.4);
        map_static.setU(2, 0.26);
        Vector<double> flux_static(nb);
        for (int i = 0; i < nb; ++i)
            map_static.computeFlux(0.0, b(i), 0.0, 0.1, flux_static.row(i));

        // -- Now evaluate stuff in multiprecision --

        // Instantiate the default map
        Map<Temporal<Multi>> map_multi(lmax, nt);
        
        // Give the star unit flux
        map_multi.setY(0, 0, y00.template cast<Multi>());

        // Set the linear and quadratic
        // limb darkening coefficients
        map_multi.setU(1, 0.4);
        map_multi.setU(2, 0.26);

        // Compute the flux
        Vector<Multi> flux_multi(nb);
        for (int i = 0; i < nb; ++i)
            map_multi.computeFlux(Multi(t), 0.0, Multi(b(i)), 0.0, 0.1, flux_multi.row(i));

        // Compare
        int nerr = 0;
        if (!flux.isApprox(flux_static.template cast<double>())) {
            std::cout << "Flux does not match static flux in `test_flux`." << std::endl;
            ++nerr;
        }
        if (!flux.isApprox(flux_multi.template cast<double>())) {
            std::cout << "Flux does not match Multi flux in `test_flux`." << std::endl;
            ++nerr;
        }
        return nerr;
    }

    /**
    Compare the spherical harmonic flux in double precision to
    the flux using multiprecision.

    TODO: Also compute the occultation flux.

    */
    int test_flux_ylm(int nb=100) {

        using namespace starry2;

        // Instantiate a temporal map with 3 time columns
        int lmax = 2;
        int nt = 3;
        Map<Temporal<double>> map(lmax, nt);
        
        // Give the star a time-variable Ylm flux
        Matrix<double> y((lmax + 1) * (lmax + 1), nt);
        y << 1.0, -0.5, 0.25,
               1.5, -0.3, 0.15,
               -0.1, 0.1, 0.3,
               0.3, 0.25, -0.1,
               1.0, -0.5, 0.25,
               1.7, -0.3, 0.15,
               -0.1, 0.1, 0.2,
               0.4, -0.15, 0.1,
               0.2, 0.15, -0.1;
        map.setY(y);

        // Compute the flux at t = 0.5, but for a varying theta
        double t = 0.5;
        Vector<double> theta = Vector<double>::LinSpaced(nb, 0, 360);
        Vector<double> flux(nb);
        for (int i = 0; i < nb; ++i)
            map.computeFlux(t, theta(i), -1.0, -1.0, 0.1, flux.row(i));

        // -- Compare to a static map --

        // At time t, the ylm vector should be...
        Vector<double> yt = y.col(0) + y.col(1) * t + 0.5 * y.col(2) * t * t;

        // Let's check that we get the same flux from a static map:
        Map<Default<double>> map_static(lmax);
        map_static.setY(yt);
        Vector<double> flux_static(nb);
        for (int i = 0; i < nb; ++i)
            map_static.computeFlux(theta(i), -1.0, -1.0, 0.1, flux_static.row(i));

        // -- Now evaluate stuff in multiprecision --

        // Instantiate the default map
        Map<Temporal<Multi>> map_multi(lmax, nt);
        
        // Set the coeffs
        map_multi.setY(y.template cast<Multi>());

        // Compute the flux
        Vector<Multi> flux_multi(nb);
        for (int i = 0; i < nb; ++i)
            map_multi.computeFlux(Multi(t), Multi(theta(i)), -1.0, -1.0, 0.1, flux_multi.row(i));

        // Compare
        int nerr = 0;
        if (!flux.isApprox(flux_static.template cast<double>())) {
            std::cout << "Flux does not match static flux in `test_flux`." << std::endl;
            ++nerr;
        }
        if (!flux.isApprox(flux_multi.template cast<double>())) {
            std::cout << "Flux does not match Multi flux in `test_flux`." << std::endl;
            ++nerr;
        }
        return nerr;
    }

    /**
    Compare the limb-darkened flux in double precision to
    the flux using multiprecision. Also compute and compare derivatives.

    */
    int test_grad_ld(int nb=100) {

        using namespace starry2;

        // Instantiate a temporal map with 3 time columns
        int lmax = 2;
        int nt = 3;
        Map<Temporal<double>> map(lmax, nt);
        
        // Give the star a time-variable total flux
        Vector<double> y00(nt);
        y00 << 1.0, -0.5, 0.25;
        map.setY(0, 0, y00);

        // Set the linear and quadratic
        // limb darkening coefficients
        map.setU(1, 0.4);
        map.setU(2, 0.26);

        // Compute the flux and the derivatives at t = 0.5
        double t = 0.5;
        Vector<double> b(nb); 
        b = Vector<double>::LinSpaced(nb, -2.5, 2.5);
        Vector<double> flux(nb);
        Vector<double> Dt(nb);
        Vector<double> Dtheta(nb);
        Vector<double> Dxo(nb);
        Vector<double> Dyo(nb);
        Vector<double> Dro(nb);
        Matrix<double> Dy(nb, nt);
        Matrix<double> Du(nb, lmax);
        for (int i = 0; i < nb; ++i)
            map.computeFlux(t, 0.0, b(i), 0.0, 0.1, flux.row(i), Dt.row(i),
                            Dtheta.row(i), Dxo.row(i), Dyo.row(i), 
                            Dro.row(i), Dy.row(i), 
                            Du.row(i).transpose());

        // -- Now compute the derivatives numerically --

        // Instantiate the default map
        Map<Temporal<Multi>> map_multi(lmax, nt);
        
        // Give the star unit flux
        map_multi.setY(0, 0, y00.template cast<Multi>());

        // Set the linear and quadratic
        // limb darkening coefficients
        map_multi.setU(1, 0.4);
        map_multi.setU(2, 0.26);

        // Compute the derivatives
        Multi eps = 1.e-15;
        Vector<Multi> eps0(3), 
                      eps1(3), 
                      eps2(3);
        eps0 << eps, 0, 0;
        eps1 << 0, eps, 0;
        eps2 << 0, 0, eps;
        Vector<Multi> f1(1), f2(1);
        Vector<Multi> dt_multi(nb),
                    dtheta_multi(nb),
                    dxo_multi(nb),
                    dyo_multi(nb),
                    dro_multi(nb);
        Matrix<Multi> dy_multi(nb, nt);
        Matrix<Multi> du_multi(nb, lmax);
        for (int i = 0; i < nb; ++i) {
            map_multi.computeFlux(t - eps, 0.0, Multi(b(i)), 0.0, 0.1, f1);
            map_multi.computeFlux(t + eps, 0.0, Multi(b(i)), 0.0, 0.1, f2);
            dt_multi(i) = (f2(0) - f1(0)) / (2 * eps);
            
            map_multi.computeFlux(t, 0.0 - eps, Multi(b(i)), 0.0, 0.1, f1);
            map_multi.computeFlux(t, 0.0 + eps, Multi(b(i)), 0.0, 0.1, f2);
            dtheta_multi(i) = (f2(0) - f1(0)) / (2 * eps);

            map_multi.computeFlux(t, 0.0, Multi(b(i)) - eps, 0.0, 0.1, f1);
            map_multi.computeFlux(t, 0.0, Multi(b(i)) + eps, 0.0, 0.1, f2);
            dxo_multi(i) = (f2(0) - f1(0)) / (2 * eps);

            map_multi.computeFlux(t, 0.0, Multi(b(i)), 0.0 - eps, 0.1, f1);
            map_multi.computeFlux(t, 0.0, Multi(b(i)), 0.0 + eps, 0.1, f2);
            dyo_multi(i) = (f2(0) - f1(0)) / (2 * eps);

            map_multi.computeFlux(t, 0.0, Multi(b(i)), 0.0, 0.1 - eps, f1);
            map_multi.computeFlux(t, 0.0, Multi(b(i)), 0.0, 0.1 + eps, f2);
            dro_multi(i) = (f2(0) - f1(0)) / (2 * eps);

            map_multi.setY(0, 0, y00.template cast<Multi>() - eps0);
            map_multi.computeFlux(t, 0.0, Multi(b(i)), 0.0, 0.1, f1);
            map_multi.setY(0, 0, y00.template cast<Multi>() + eps0);
            map_multi.computeFlux(t, 0.0, Multi(b(i)), 0.0, 0.1, f2);
            map_multi.setY(0, 0, y00.template cast<Multi>());
            dy_multi(i, 0) = (f2(0) - f1(0)) / (2 * eps);

            map_multi.setY(0, 0, y00.template cast<Multi>() - eps1);
            map_multi.computeFlux(t, 0.0, Multi(b(i)), 0.0, 0.1, f1);
            map_multi.setY(0, 0, y00.template cast<Multi>() + eps1);
            map_multi.computeFlux(t, 0.0, Multi(b(i)), 0.0, 0.1, f2);
            map_multi.setY(0, 0, y00.template cast<Multi>());
            dy_multi(i, 1) = (f2(0) - f1(0)) / (2 * eps);

            map_multi.setY(0, 0, y00.template cast<Multi>() - eps2);
            map_multi.computeFlux(t, 0.0, Multi(b(i)), 0.0, 0.1, f1);
            map_multi.setY(0, 0, y00.template cast<Multi>() + eps2);
            map_multi.computeFlux(t, 0.0, Multi(b(i)), 0.0, 0.1, f2);
            map_multi.setY(0, 0, y00.template cast<Multi>());
            dy_multi(i, 2) = (f2(0) - f1(0)) / (2 * eps);

            map_multi.setU(1, 0.4 - eps);
            map_multi.computeFlux(t, 0.0, Multi(b(i)), 0.0, 0.1, f1);
            map_multi.setU(1, 0.4 + eps);
            map_multi.computeFlux(t, 0.0, Multi(b(i)), 0.0, 0.1, f2);
            map_multi.setU(1, 0.4);
            du_multi(i, 0) = (f2(0) - f1(0)) / (2 * eps);

            map_multi.setU(2, 0.26 - eps);
            map_multi.computeFlux(t, 0.0, Multi(b(i)), 0.0, 0.1, f1);
            map_multi.setU(2, 0.26 + eps);
            map_multi.computeFlux(t, 0.0, Multi(b(i)), 0.0, 0.1, f2);
            map_multi.setU(2, 0.26);
            du_multi(i, 1) = (f2(0) - f1(0)) / (2 * eps);
        }

        // Compare
        int nerr = 0;
        if (!Dt.isApprox(dt_multi.template cast<double>())) {
            std::cout << "Wrong t deriv in `test_flux_with_grad`." << std::endl;
            ++nerr;
        }
        if (!Dtheta.isApprox(dtheta_multi.template cast<double>())) {
            std::cout << "Wrong theta deriv in `test_flux_with_grad`." << std::endl;
            ++nerr;
        }
        if (!Dxo.isApprox(dxo_multi.template cast<double>())) {
            std::cout << "Wrong xo deriv in `test_flux_with_grad`." << std::endl;
            ++nerr;
        }
        if (!Dyo.isApprox(dyo_multi.template cast<double>())) {
            std::cout << "Wrong yo deriv in `test_flux_with_grad`." << std::endl;
            ++nerr;
        }
        if (!Dro.isApprox(dro_multi.template cast<double>())) {
            std::cout << "Wrong ro deriv in `test_flux_with_grad`." << std::endl;
            ++nerr;
        }
        if (!Dy.isApprox(dy_multi.template cast<double>())) {
            std::cout << "Wrong y deriv in `test_flux_with_grad`." << std::endl;
            
            std::cout << Dy << std::endl << std::endl;
            std::cout << dy_multi << std::endl;

            ++nerr;
        }
        if (!Du.isApprox(du_multi.template cast<double>())) {
            std::cout << "Wrong u deriv in `test_flux_with_grad`." << std::endl;
            ++nerr;
        }
        return nerr;
    }

    /**
    Compare the spherical harmonic flux in double precision to
    the flux using multiprecision. Also compute and compare derivatives.

    TODO: Also compute the occultation flux.

    */
    int test_grad_ylm(int nb=100) {

        using namespace starry2;

        // Instantiate a temporal map with 3 time columns
        int lmax = 2;
        int nt = 3;
        Map<Temporal<double>> map(lmax, nt);
        
        // Give the star a time-variable Ylm flux
        Matrix<double> y((lmax + 1) * (lmax + 1), nt);
        y << 1.0, -0.5, 0.25,
               1.5, -0.3, 0.15,
               -0.1, 0.1, 0.3,
               0.3, 0.25, -0.1,
               1.0, -0.5, 0.25,
               1.7, -0.3, 0.15,
               -0.1, 0.1, 0.2,
               0.4, -0.15, 0.1,
               0.2, 0.15, -0.1;
        map.setY(y);

        // Compute the flux and the derivatives at t = 0.5
        double t = 0.5;
        Vector<double> theta = Vector<double>::LinSpaced(nb, 0, 360);
        Vector<double> flux(nb);
        Vector<double> Dt(nb);
        Vector<double> Dtheta(nb);
        Vector<double> Dxo(nb);
        Vector<double> Dyo(nb);
        Vector<double> Dro(nb);
        Matrix<double> Dy(nb * (lmax + 1) * (lmax + 1), nt);
        Vector<double> Du(nb);
        for (int i = 0; i < nb; ++i)
            map.computeFlux(t, theta(i), -1.0, -1.0, 0.1, flux.row(i), Dt.row(i),
                            Dtheta.row(i), Dxo.row(i), Dyo.row(i), 
                            Dro.row(i), 
                            Dy.block(i * (lmax + 1) * (lmax + 1), 0, 
                                     (lmax + 1) * (lmax + 1), nt), 
                            Du.row(i));

        // -- Now compute the derivatives numerically --

        // Instantiate the default map
        Map<Temporal<Multi>> map_multi(lmax, nt);
        
        // Give the star unit flux
        map_multi.setY(y.template cast<Multi>());

        // Compute the derivatives
        Multi eps = 1.e-15;
        Matrix<Multi> epsm((lmax + 1) * (lmax + 1), nt);
        Vector<Multi> f1(1), f2(1);
        Vector<Multi> dt_multi(nb),
                    dtheta_multi(nb),
                    dxo_multi(nb),
                    dyo_multi(nb),
                    dro_multi(nb),
                    du_multi(nb);
        Matrix<Multi> dy_multi(nb * (lmax + 1) * (lmax + 1), nt);
        for (int i = 0; i < nb; ++i) {
            map_multi.computeFlux(t - eps, Multi(theta(i)), -1.0, -1.0, 0.1, f1);
            map_multi.computeFlux(t + eps, Multi(theta(i)), -1.0, -1.0, 0.1, f2);
            dt_multi(i) = (f2(0) - f1(0)) / (2 * eps);
            
            map_multi.computeFlux(t, Multi(theta(i)) - eps, -1.0, -1.0, 0.1, f1);
            map_multi.computeFlux(t, Multi(theta(i)) + eps, -1.0, -1.0, 0.1, f2);
            dtheta_multi(i) = (f2(0) - f1(0)) / (2 * eps);

            map_multi.computeFlux(t, Multi(theta(i)), -1.0 - eps, -1.0, 0.1, f1);
            map_multi.computeFlux(t, Multi(theta(i)), -1.0 + eps, -1.0, 0.1, f2);
            dxo_multi(i) = (f2(0) - f1(0)) / (2 * eps);

            map_multi.computeFlux(t, Multi(theta(i)), -1.0, -1.0 - eps, 0.1, f1);
            map_multi.computeFlux(t, Multi(theta(i)), -1.0, -1.0 + eps, 0.1, f2);
            dyo_multi(i) = (f2(0) - f1(0)) / (2 * eps);

            map_multi.computeFlux(t, Multi(theta(i)), -1.0, -1.0, 0.1 - eps, f1);
            map_multi.computeFlux(t, Multi(theta(i)), -1.0, -1.0, 0.1 + eps, f2);
            dro_multi(i) = (f2(0) - f1(0)) / (2 * eps);

            int n = 0;
            for (int l = 0; l < lmax + 1; ++l) {
                for (int m = -l; m < l + 1; ++m) {
                    for (int j = 0; j < nt; ++j) {
                        epsm.setZero();
                        epsm(n, j) = eps;
                        map_multi.setY(y.template cast<Multi>() - epsm);
                        map_multi.computeFlux(t, Multi(theta(i)), -1.0, -1.0, 0.1, f1);
                        map_multi.setY(y.template cast<Multi>() + epsm);
                        map_multi.computeFlux(t, Multi(theta(i)), -1.0, -1.0, 0.1, f2);
                        map_multi.setY(y.template cast<Multi>());
                        dy_multi(i * (lmax + 1) * (lmax + 1) + n, j) = 
                            (f2(0) - f1(0)) / (2 * eps);
                    }
                    ++n;
                }
            }
        }

        // Compare
        int nerr = 0;
        if (!Dt.isApprox(dt_multi.template cast<double>())) {
            std::cout << "Wrong t deriv in `test_flux_with_grad`." << std::endl;

                        
            std::cout << Dt.transpose() << std::endl << std::endl;
            std::cout << dt_multi.transpose() << std::endl;


            ++nerr;
        }
        if (!Dtheta.isApprox(dtheta_multi.template cast<double>())) {
            std::cout << "Wrong theta deriv in `test_flux_with_grad`." << std::endl;
            ++nerr;
        }
        if (!Dxo.isApprox(dxo_multi.template cast<double>())) {
            std::cout << "Wrong xo deriv in `test_flux_with_grad`." << std::endl;
            ++nerr;
        }
        if (!Dyo.isApprox(dyo_multi.template cast<double>())) {
            std::cout << "Wrong yo deriv in `test_flux_with_grad`." << std::endl;
            ++nerr;
        }
        if (!Dro.isApprox(dro_multi.template cast<double>())) {
            std::cout << "Wrong ro deriv in `test_flux_with_grad`." << std::endl;
            ++nerr;
        }
        if (!Dy.isApprox(dy_multi.template cast<double>())) {
            std::cout << "Wrong y deriv in `test_flux_with_grad`." << std::endl;
            ++nerr;
        }
        if (!Du.isApprox(du_multi.template cast<double>())) {
            std::cout << "Wrong u deriv in `test_flux_with_grad`." << std::endl;
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

} // namespace test_temporal
#endif