#include "starry.h"

/**
Return the n^th term of the *r* phase curve solution vector.

*/
double rn(int mu, int nu) {
        if (((mu % 2) == 0) && ((mu / 2) % 2 == 0) && ((nu / 2) % 2 == 0))
            return gsl_sf_gamma(0.25 * mu + 0.5) *
                   gsl_sf_gamma(0.25 * nu + 0.5) /
                   gsl_sf_gamma(0.25 * (mu + nu) + 2);
        else if ((((mu - 1) % 2) == 0) && (((mu - 1) / 2) % 2 == 0) && (((nu - 1) / 2) % 2 == 0))
            return gsl_sf_gamma(0.25 * mu + 0.25) *
                   gsl_sf_gamma(0.25 * nu + 0.25) /
                   gsl_sf_gamma(0.25 * (mu + nu) + 2) /
                   M_2_SQRTPI;
        else
            return 0;

}

/**
Compute the *r* phase curve solution vector.

*/
void r(int lmax, double* vector) {
    int l, m, mu, nu;
    int n = 0;
    for (l=0; l<lmax+1; l++) {
        for (m=-l; m<l+1; m++) {
            mu = l - m;
            nu = l + m;
            vector[n] = rn(mu, nu);
            n++;
        }
    }
    return;
}


/**
Compute the n=2 term of the *s* occultation solution vector.
This is the Mandel & Agol solution for linear limb darkening.

*/
double s2(double b, double r, double ksq, double K, double E) {
    double Lambda;
    double r2 = r * r;
    double b2 = b * b;
    double br = b * r;
    double xi = 2 * br * (4 - 7 * r2 - b2);
    double PI;

    if (b == 0) {
        Lambda = -(2. / 3.) * pow(1 - r2, 1.5);
    } else if (ksq < 1) {
        PI = ellipPI(1 - 1. / ((b - r) * (b - r)), ksq);
        Lambda = (1. / (8 * M_PI * sqrt(br))) *
                 ((-3 + 12 * r2 - 10 * b2 * r2 - 6 * r2 * r2 + xi) * K -
                  2 * xi * E +
                  3 * (b + r) / (b - r) * PI);
    } else {
        PI = ellipPI(1. / (ksq - 1. / (4 * br)), 1. / ksq);
        Lambda = (1. / (9 * M_PI * sqrt((1 - b + r) * (1 + b - r)))) *
                 (2 * (1 - 5 * b2 + r2 + (r2 - b2) * (r2 - b2)) * K -
                  2 * xi * ksq * E +
                  3 * (b + r) / (b - r) * PI);
    }

    return (2. / (3. * M_PI)) * (1 - 1.5 * Lambda - step(r - b));
}

/**
The primitive integral P.
TODO!

*/
double P(int l, int m, double b, double r){
    return 0;
}

/**
The primitive integral Q.
TODO!

*/
double Q(int l, int m, double b, double r){
    return 0;
}

/**
Compute the *s* occultation solution vector.

*/
void s(int lmax, double b, double r, double* vector) {
    int l, m;
    int n = 0;
    double ksq;
    double K, E;

    // Sanity check
    if ((b <= r - 1) || (b >= 1 + r)) {
        cout << "ERROR: Domain error in function s()." << endl;
        exit(1);
    }

    // Pre-compute the elliptic integrals
    ksq = (1 - r * r - b * b + 2 * b * r) / (4 * b * r);
    K = ellipK(ksq);
    E = ellipE(ksq);

    // Populate the vector
    for (l=0; l<lmax+1; l++) {
        for (m=-l; m<l+1; m++) {
            if ((l == 1) && (m == 0))
                vector[n] = s2(b, r, ksq, K, E);
            else
                vector[n] = Q(l, m, b, r) - P(l, m, b, r);
            n++;
        }
    }
    return;
}
