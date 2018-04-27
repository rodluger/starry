#include <iostream>
#include <iomanip>
#include <Eigen/Core>
#include <cmath>
#include "solver.h"

using namespace std;

int main() {

    // User inputs
    double b = 0.3;
    double r = 0.1;
    double s2;

    // Initialize the G class
    solver::Greens<double> G(2);
    G.br = b * r;
    G.br32 = pow(G.br, 1.5);
    G.b.reset(b);
    G.r.reset(r);
    G.b_r.reset(b / r);
    if (r <= 1)
        G.ksq.reset((1 - G.r(2) - G.b(2) + 2 * G.br) / (4 * G.br));
    else
        G.ksq.reset((1 - (b - r)) * (1 + (b - r)) / (4 * G.br));
    G.k = sqrt(G.ksq());
    if ((abs(1 - r) < b) && (b < 1 + r)) {
        if (r <= 1) {
            G.sinphi.reset((1 - G.r(2) - G.b(2)) / (2 * G.br));
            G.cosphi.reset(sqrt(1 - G.sinphi() * G.sinphi()));
            G.sinlam.reset((1 - G.r(2) + G.b(2)) / (2 * G.b()));
            G.coslam.reset(sqrt(1 - G.sinlam() * G.sinlam()));
            G.phi = asin(G.sinphi());
            G.lam = asin(G.sinlam());
        } else {
            G.sinphi.reset(2 * (G.ksq() - 0.5));
            G.cosphi.reset(2 * G.k * sqrt(1 - G.ksq()));
            G.sinlam.reset(0.5 * ((1. / b) + (b - r) * (1. + r / b)));
            G.coslam.reset(sqrt(1 - G.sinlam() * G.sinlam()));
            G.phi = asin(G.sinphi());
            G.lam = asin(G.sinlam());
        }
    } else {
        G.sinphi.reset(1);
        G.cosphi.reset(0);
        G.sinlam.reset(1);
        G.coslam.reset(0);
        G.phi = 0.5 * G.pi;
        G.lam = 0.5 * G.pi;
    }
    G.H.reset();
    G.I.reset();
    G.J.reset();
    G.M.reset();
    G.ELL.reset();

    // Ok, compute s2
    s2 = solver::s2(G);

    // Print it
    cout << s2 << endl;

    return 0;
}
