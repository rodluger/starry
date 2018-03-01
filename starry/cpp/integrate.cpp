#include "starry.h"

/**

*/
typedef struct {
    int l;
    int m;
    int mu;
    int nu;
    int lmax;
    double b;
    double b2;
    double br;
    double* r;
    double* b_r;
    double* cosphi;
    double* sinphi;
    double* coslam;
    double* sinlam;
    double ksq;
    double k;
    double E;
    double K;
    double PI;
    double E1;
    double E2;
    double** HPHI;
    bool** HPHISET;
    double** HLAM;
    bool** HLAMSET;
} GREENS;

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
void rT(int lmax, double* vector) {
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
double s2(GREENS* G) {
    double Lambda;
    double xi = 2 * G->br * (4 - 7 * G->r[2] - G->b2);

    if (G->b == 0) {
        Lambda = -(2. / 3.) * pow(1 - G->r[2], 1.5);
    } else if (G->ksq < 1) {
        Lambda = (1. / (8 * M_PI * sqrt(G->br))) *
                 ((-3 + 12 * G->r[2] - 10 * G->b2 * G->r[2] - 6 * G->r[4] + xi) * G->K -
                  2 * xi * G->E +
                  3 * (G->b + G->r[1]) / (G->b - G->r[1]) * G->PI);
    } else {
        Lambda = (2. / (9 * M_PI * sqrt((1 - G->b + G->r[1]) * (1 + G->b - G->r[1])))) *
                 ((1 - 5 * G->b2 + G->r[2] + (G->r[2] - G->b2) * (G->r[2] - G->b2)) * G->K -
                  2 * xi * G->ksq * G->E +
                  3 * (G->b + G->r[1]) / (G->b - G->r[1]) * G->PI);
    }

    return (2. / (3. * M_PI)) * (1 - 1.5 * Lambda - step(G->r[1] - G->b));
}

/**
The helper primitive integral H(phi) for odd nu.

    TODO!

*/
double HODDPHI(int u, int n, GREENS* G) {
    if (!G->HPHISET[u][n]) {
        G->HPHISET[u][n] = true;
        if (1) {
            G->HPHI[u][n] = 0;
        } else if (0) {

        } else if (0) {

        } else if (0) {

        } else if (0) {

        } else {

        }

    }
    return G->HPHI[u][n];
}

/**
The helper primitive integral H(phi) for even nu.

*/
double HEVENPHI(int u, int n, GREENS* G) {
    if (!G->HPHISET[u][n]) {
        G->HPHISET[u][n] = true;
        if (u % 2 == 1) {
            G->HPHI[u][n] = 0;
        } else if ((u == 0) && (n == 0)) {
            G->HPHI[u][n] = 2 * asin(G->sinphi[1]) + M_PI;
        } else if ((u == 0) && (n == 1)) {
            G->HPHI[u][n] = -2 * G->cosphi[1];
        } else if (u >= 2) {
            G->HPHI[u][n] = (2 * G->cosphi[u - 1] * G->sinphi[n + 1] +
                            (u - 1) * HEVENPHI(u - 2, n, G)) / (u + n);
        } else {
            G->HPHI[u][n] = (-2 * G->cosphi[u + 1] * G->sinphi[n - 1] +
                            (n - 1) * HEVENPHI(u, n - 2, G)) / (u + n);
        }
    }
    return G->HPHI[u][n];
}

/**
The helper primitive integral H(lambda) for even nu.

*/
double HEVENLAM(int u, int n, GREENS* G) {
    if (!G->HLAMSET[u][n]) {
        G->HLAMSET[u][n] = true;
        if (u % 2 == 1) {
            G->HLAM[u][n] = 0;
        } else if ((u == 0) && (n == 0)) {
            G->HLAM[u][n] = 2 * asin(G->sinlam[1]) + M_PI;
        } else if ((u == 0) && (n == 1)) {
            G->HLAM[u][n] = -2 * G->coslam[1];
        } else if (u >= 2) {
            G->HLAM[u][n] = (2 * G->coslam[u - 1] * G->sinlam[n + 1] +
                            (u - 1) * HEVENLAM(u - 2, n, G)) / (u + n);
        } else {
            G->HLAM[u][n] = (-2 * G->coslam[u + 1] * G->sinlam[n - 1] +
                            (n - 1) * HEVENLAM(u, n - 2, G)) / (u + n);
        }
    }
    return G->HLAM[u][n];
}

/**
The helper primitive integral I(phi).

*/
double IPHI(int u, int v, GREENS* G) {
    double res = 0;
    for (int n=0; n<v+1; n++) {
        if (G->nu % 2 == 0)
            res += gsl_sf_choose(v, n) * G->b_r[v - n] * HEVENPHI(u, n, G);
        else
            res += gsl_sf_choose(v, n) * G->b_r[v - n] * HODDPHI(u, n, G);
    }
    return res;
}

/**
The primitive integral P.

*/
double P(GREENS* G){
    if (G->nu % 2 == 0)
        return G->r[G->l + 2] * IPHI((G->mu + 4) / 2, G->nu / 2, G);
    else if ((G->mu == 1) && (G->l % 2 == 0))
        return G->b * G->r[G->l - 2] * IPHI(G->l - 2, 0, G) -
               G->r[G->l - 1] * IPHI(G->l - 2, 1, G);
    else if ((G->mu == 1) && (G->l % 2 == 0))
        return G->b * G->r[G->l - 2] * IPHI(G->l - 3, 1, G) -
               G->r[G->l - 1] * IPHI(G->l - 3, 2, G);
    else
        return G->r[G->l - 1] * IPHI((G->mu - 1) / 2, (G->nu - 1) / 2, G);
}

/**
The primitive integral Q.

*/
double Q(GREENS* G){
    if (G->nu % 2 == 0)
        return HEVENLAM((G->mu + 4) / 2, G->nu / 2, G);
    else
        return 0;
}

/**
Compute the *s* occultation solution vector.

*/
void sT(int lmax, double b, double r, double* vector) {
    int l, m;
    int n = 0;
    int N = lmax + 3;
    double b_r = b / r;
    double cosphi, sinphi, coslam, sinlam;

    // Sanity check
    if ((b <= r - 1) || (b >= 1 + r)) {
        cout << "ERROR: Domain error in function s()." << endl;
        exit(1);
    }

    // TODO: Check if b = 0 (special case)

    // Instantiate a GREENS struct to hold useful quantities
    GREENS G;
    G.lmax = lmax;
    G.b = b;
    G.b2 = b * b;
    G.br = b * r;

    // Compute the powers of r
    G.r = new double[N];
    G.r[0] = 1;
    for (l=1; l<N; l++)
        G.r[l] = r * G.r[l - 1];

    // Compute the powers of (b / r)
    G.b_r = new double[N];
    G.b_r[0] = 1;
    for (l=1; l<N; l++)
        G.b_r[l] = b_r * G.b_r[l - 1];

    // Compute the sine and cosine of the angles
    if ((abs(1 - r) < b) && (b < 1 + r)) {
        // sin(arcsin(x)) = x
        // cos(arcsin(x)) = sqrt(1 - x * x)
        sinphi = (1 - G.r[2] - G.b2) / (2 * G.br);
        cosphi = sqrt(1 - sinphi * sinphi);
        sinlam = (1 - G.r[2] + G.b2) / (2 * G.b);
        coslam = sqrt(1 - sinlam * sinlam);
    } else {
        sinphi = 1;
        cosphi = 0;
        sinlam = 1;
        coslam = 0;
    }

    // Compute the sine and cosine powers
    G.cosphi = new double[N + 1];
    G.sinphi = new double[N + 1];
    G.coslam = new double[N + 1];
    G.sinlam = new double[N + 1];
    G.cosphi[0] = 1;
    G.sinphi[0] = 1;
    G.coslam[0] = 1;
    G.sinlam[0] = 1;
    for (l=1; l<N+1; l++) {
        G.cosphi[l] = cosphi * G.cosphi[l - 1];
        G.sinphi[l] = sinphi * G.sinphi[l - 1];
        G.coslam[l] = coslam * G.coslam[l - 1];
        G.sinlam[l] = sinlam * G.sinlam[l - 1];
    }

    // Compute the elliptic integrals
    G.ksq = (1 - G.r[2] - G.b * G.b + 2 * G.br) / (4 * G.br);
    G.k = sqrt(G.ksq);
    if (G.ksq < 1) {
        G.K = ellipK(G.ksq);
        G.E = ellipE(G.ksq);
        G.PI = ellipPI(1 - 1. / ((G.b - G.r[1]) * (G.b - G.r[1])), G.ksq);
        G.E1 = (1 - G.ksq) * G.K;
        G.E2 = G.E;
    } else {
        G.K = ellipK(1. / G.ksq);
        G.E = ellipE(1. / G.ksq);
        G.PI = ellipPI(1. / (G.ksq - 1. / (4 * G.br)), 1. / G.ksq);
        G.E1 = (1 - G.ksq) / G.k * G.K;
        G.E2 = G.k * G.E + (1 - G.ksq) / G.k * G.K;
    }

    // Allocate the H integrals
    G.HPHI = new double*[N];
    G.HLAM = new double*[N];
    G.HPHISET = new bool*[N];
    G.HLAMSET = new bool*[N];
    for (l=0; l<N; l++) {
        G.HPHI[l] = new double[N];
        G.HLAM[l] = new double[N];
        G.HPHISET[l] = new bool[N];
        G.HLAMSET[l] = new bool[N];
        for (m=0; m<N; m++) {
            G.HPHISET[l][m] = false;
            G.HLAMSET[l][m] = false;
        }
    }

    // Populate the vector
    for (l=0; l<lmax+1; l++) {
        G.l = l;
        for (m=-l; m<l+1; m++) {
            G.m = m;
            G.mu = l - m;
            G.nu = l + m;
            if ((l == 1) && (m == 0))
                vector[n] = s2(&G);
            else
                vector[n] = Q(&G) - P(&G);
            n++;
        }
    }

    // Free the memory
    delete [] G.r;
    delete [] G.b_r;
    delete [] G.cosphi;
    delete [] G.sinphi;
    delete [] G.coslam;
    delete [] G.sinlam;
    for (l=0; l<N; l++) {
        delete [] G.HPHI[l];
        delete [] G.HLAM[l];
        delete [] G.HPHISET[l];
        delete [] G.HLAMSET[l];
    }
    delete [] G.HPHI;
    delete [] G.HLAM;
    delete [] G.HPHISET;
    delete [] G.HLAMSET;

    return;
}
