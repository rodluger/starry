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
    double br32;
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
    double** H;
    bool** bH;
    double** I;
    bool** bI;
    double** J;
    bool** bJ;
    double** M;
    bool** bM;
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

    return (2. * M_PI / 3.) * (1 - 1.5 * Lambda - step(G->r[1] - G->b));
}


/**
The helper primitive integral H_{u,v}

*/
double H(int u, int v, GREENS* G) {
    if (!G->bH[u][v]) {
        G->bH[u][v] = true;
        if (u % 2 == 1) {
            G->H[u][v] = 0;
        } else if ((u == 0) && (v == 0)) {
            G->H[u][v] = 2 * asin(G->sinlam[1]) + M_PI;
        } else if ((u == 0) && (v == 1)) {
            G->H[u][v] = -2 * G->coslam[1];
        } else if (u >= 2) {
            G->H[u][v] = (2 * G->coslam[u - 1] * G->sinlam[v + 1] +
                            (u - 1) * H(u - 2, v, G)) / (u + v);
        } else {
            G->H[u][v] = (-2 * G->coslam[u + 1] * G->sinlam[v - 1] +
                            (v - 1) * H(u, v - 2, G)) / (u + v);
        }
    }
    return G->H[u][v];
}

/**
The helper primitive integral I_{u,v}

*/
double I(int u, int v, GREENS* G) {
    if (!G->bI[u][v]) {
        G->bI[u][v] = true;
        if (u % 2 == 1) {
            G->I[u][v] = 0;
        } else if ((u == 0) && (v == 0)) {
            G->I[u][v] = 2 * asin(G->sinphi[1]) + M_PI;
        } else if ((u == 0) && (v == 1)) {
            G->I[u][v] = -2 * G->cosphi[1];
        } else if (u >= 2) {
            G->I[u][v] = (2 * G->cosphi[u - 1] * G->sinphi[v + 1] +
                            (u - 1) * I(u - 2, v, G)) / (u + v);
        } else {
            G->I[u][v] = (-2 * G->cosphi[u + 1] * G->sinphi[v - 1] +
                            (v - 1) * I(u, v - 2, G)) / (u + v);
        }
    }
    return G->I[u][v];
}

/**
The helper primitive integral M_{p,q}

*/
double M(int p, int q, GREENS* G) {
    if (!G->bM[p][q]) {
        G->bM[p][q] = true;
        if ((p % 2 == 1) || (q % 2 == 1)) {
            G->M[p][q] = 0;
        } else if ((p == 0) && (q == 0)) {
            G->M[p][q] = ((8 - 12 * G->ksq) * G->E1 + (-8 + 16 * G->ksq) * G->E2) / 3.;
        } else if ((p == 0) && (q == 2)) {
            G->M[p][q] = ((8 - 24 * G->ksq) * G->E1 + (-8 + 28 * G->ksq + 12 * G->ksq * G->ksq) * G->E2) / 15.;
        } else if ((p == 2) && (q == 0)) {
            G->M[p][q] = ((32 - 36 * G->ksq) * G->E1 + (-32 + 52 * G->ksq - 12 * G->ksq * G->ksq) * G->E2) / 15.;
        } else if ((p == 2) && (q == 2)) {
            G->M[p][q] = ((32 - 60 * G->ksq + 12 * G->ksq * G->ksq) * G->E1 + (-32 + 76 * G->ksq - 36 * G->ksq * G->ksq + 24 * G->ksq * G->ksq * G->ksq) * G->E2) / 105.;
        } else if (q >= 4) {
            double d1, d2;
            d1 = q + 2 + (p + q - 2) * (1 - G->ksq);
            d2 = (3 - q) * (1 - G->ksq);
            G->M[p][q] = (d1 * M(p, q - 2, G) + d2 * M(p, q - 4, G)) / (p + q + 3);
        } else if (p >= 4) {
            double d3, d4;
            d3 = 2 * p + q - (p + q - 2) * (1 - G->ksq);
            d4 = (3 - p) + (p - 3) * (1 - G->ksq);
            G->M[p][q] = (d3 * M(p - 2, q, G) + d4 * M(p - 4, q, G)) / (p + q + 3);
        } else {
            cout << "ERROR: Domain error in function M()." << endl;
            exit(1);
        }
    }
    return G->M[p][q];
}

/**
The helper primitive integral J_{u,v}

*/
double J(int u, int v, GREENS* G) {
    if (!G->bJ[u][v]) {
        G->bJ[u][v] = true;
        G->J[u][v] = 0;
        for (int i=0; i<v+1; i++)
            if ((i - v - u) % 2 == 0)
                G->J[u][v] += gsl_sf_choose(v, i) * M(u + 2 * i, u + 2 * v - 2 * i, G);
            else
                G->J[u][v] -= gsl_sf_choose(v, i) * M(u + 2 * i, u + 2 * v - 2 * i, G);
        G->J[u][v] *= pow(2, u + 3) * (G->br32);
    }
    return G->J[u][v];
}

/**
The helper primitive integral K_{u,v}.

*/
double K(int u, int v, GREENS* G) {
    double res = 0;
    for (int i=0; i<v+1; i++)
        res += gsl_sf_choose(v, i) * G->b_r[v - i] * I(u, i, G);
    return res;
}

/**
The helper primitive integral L_{u,v}.

*/
double L(int u, int v, GREENS* G) {
    double res = 0;
    for (int i=0; i<v+1; i++)
        res += gsl_sf_choose(v, i) * G->b_r[v - i] * J(u, i, G);
    return res;
}

/**
The primitive integral P(G_n).

*/
double P(GREENS* G){
    if (G->nu % 2 == 0)
        return G->r[G->l + 2] * K((G->mu + 4) / 2, G->nu / 2, G);
    else if ((G->mu == 1) && (G->l % 2 == 0))
        return G->b * G->r[G->l - 2] * L(G->l - 2, 0, G) -
               G->r[G->l - 1] * L(G->l - 2, 1, G);
    else if ((G->mu == 1) && (G->l % 2 == 1))
        return G->b * G->r[G->l - 2] * L(G->l - 3, 1, G) -
               G->r[G->l - 1] * L(G->l - 3, 2, G);
    else
        return G->r[G->l - 1] * L((G->mu - 1) / 2, (G->nu - 1) / 2, G);
}

/**
The primitive integral Q(G_n).

*/
double Q(GREENS* G){
    if (G->nu % 2 == 0)
        return H((G->mu + 4) / 2, G->nu / 2, G);
    else
        return 0;
}

/**
Compute the *s* occultation solution vector.

*/
void sT(int lmax, double b, double r, double* vector) {
    int l, m;
    int n = 0;
    int N = 2 * lmax + 1;
    if (N < 2) N = 2;
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
    G.br32 = pow(G.br, 1.5);

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
    G.cosphi = new double[N];
    G.sinphi = new double[N];
    G.coslam = new double[N];
    G.sinlam = new double[N];
    G.cosphi[0] = 1;
    G.sinphi[0] = 1;
    G.coslam[0] = 1;
    G.sinlam[0] = 1;
    for (l=1; l<N; l++) {
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

    // Allocate the integral matrices
    G.H = new double*[N];
    G.I = new double*[N];
    G.J = new double*[N];
    G.M = new double*[N];
    G.bH = new bool*[N];
    G.bI = new bool*[N];
    G.bJ = new bool*[N];
    G.bM = new bool*[N];
    for (l=0; l<N; l++) {
        G.H[l] = new double[N];
        G.I[l] = new double[N];
        G.J[l] = new double[N];
        G.M[l] = new double[N];
        G.bH[l] = new bool[N];
        G.bI[l] = new bool[N];
        G.bJ[l] = new bool[N];
        G.bM[l] = new bool[N];
        for (m=0; m<N; m++) {
            G.bH[l][m] = false;
            G.bI[l][m] = false;
            G.bJ[l][m] = false;
            G.bM[l][m] = false;
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
        delete [] G.H[l];
        delete [] G.I[l];
        delete [] G.J[l];
        delete [] G.M[l];
        delete [] G.bH[l];
        delete [] G.bI[l];
        delete [] G.bJ[l];
        delete [] G.bM[l];
    }
    delete [] G.H;
    delete [] G.I;
    delete [] G.J;
    delete [] G.M;
    delete [] G.bH;
    delete [] G.bI;
    delete [] G.bJ;
    delete [] G.bM;

    return;
}
