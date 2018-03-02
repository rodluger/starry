#include "starry.h"

/**
Initialize a CONSTANTS struct.

*/
void init_constants(int lmax, CONSTANTS* C) {
    int i, j;
    C->lmax = lmax;
    C->N = (lmax + 1) * (lmax + 1);

    // Allocate the matrices
    C->A = new double*[C->N];
    C->A1 = new double*[C->N];
    for (i=0; i<C->N; i++) {
        C->A[i] = new double[C->N];
        C->A1[i] = new double[C->N];
        for (j=0; j<C->N; j++) {
            C->A[i][j] = 0;
            C->A1[i][j] = 0;
        }
    }

    // Allocate the vector
    C->rT = new double[C->N];
    for (i=0; i<C->N; i++) {
        C->rT[i] = 0;
    }

    // Compute A1
    A1(C->lmax, C->A1);

    // Compute A (from A1 for speed)
    double **tmp = new double*[C->N];
    for (i=0; i<C->N; i++)
        tmp[i] = new double[C->N];
    A2(C->lmax, tmp);
    dot(C->N, tmp, C->A1, C->A);
    for(i = 0; i<C->N; i++)
        delete [] tmp[i];
    delete [] tmp;

    // Compute r
    rT(C->lmax, C->rT);

}


/**
Compute the flux from a STARRY map.

*/
void flux(int NT, double* y, double u[3], double* theta, double* x0,
          double* y0, double r, CONSTANTS* C, double* result) {
    int i;
    double b;
    double zhat[3] = {0., 0., 1.};
    double** ROT1;
    double** ROT2;
    ROT1 = new double*[C->N];
    ROT2 = new double*[C->N];
    for (i=0; i<C->N; i++) {
        ROT1[i] = new double[C->N];
        ROT2[i] = new double[C->N];
    }
    double* tmp1 = new double[C->N];
    double* tmp2 = new double[C->N];
    double costheta, sintheta;
    double cosomega, sinomega;

    for (int t=0; t<NT; t++) {
        // Impact parameter
        b = sqrt(x0[t] * x0[t] + y0[t] * y0[t]);

        // Check for complete occultation
        if (b <= r - 1) {
            result[t] = 0;
            continue;
        }

        // Rotate the map into view
        if (theta[t] == 0) {
            // No need to rotate the map
            for(i = 0; i<C->N; i++)
                tmp1[i] = y[i];
        } else if ((t > 0) && (theta[t] == theta[t - 1])) {
            // We just computed this last step,
            // so we can re-use it!
            dot(C->N, ROT1, y, tmp1);
        } else {
            // Rotate the map
            costheta = cos(theta[t]);
            sintheta = sin(theta[t]);
            R(C->lmax, u, costheta, sintheta, ROT1);
            dot(C->N, ROT1, y, tmp1);
        }

        // No occultation
        if (b >= 1 + r) {

            // Change basis to polynomials
            dot(C->N, C->A1, tmp1, tmp2);

            // Dot into solution vector r^T
            result[t] = dot(C->N, C->rT, tmp2);

        // Occultation
        } else {

            if (b > 0) {
                // Align occultor with the +y axis
                sinomega = x0[t] / b;
                cosomega = y0[t] / b;
                R(C->lmax, zhat, cosomega, sinomega, ROT2);
                dot(C->N, ROT2, tmp1, tmp2);
            } else {
                for(i = 0; i<C->N; i++)
                    tmp2[i] = tmp1[i];
            }

            // Change basis to Green's
            dot(C->N, C->A, tmp2, tmp1);

            // Dot into solution vector s^T
            sT(C->lmax, b, r, tmp2);
            result[t] = dot(C->N, tmp2, tmp1);

        }

    }

    // Free the memory
    for(i = 0; i<C->N; i++) {
        delete [] ROT1[i];
        delete [] ROT2[i];
    }
    delete [] ROT1;
    delete [] ROT2;
    delete [] tmp1;
    delete [] tmp2;

}

/**
Render a STARRY map on a grid.
*/
void render(double* y, double u[3], double theta,
            CONSTANTS* C, int res, double** result) {
    int i, j;
    double* x = new double[res];
    double* tmp = new double[C->N];
    double* p = new double[C->N];
    double** ROT1;
    ROT1 = new double*[C->N];
    for (i=0; i<C->N; i++)
        ROT1[i] = new double[C->N];

    // Rotate the map into view
    if (theta == 0) {
        for(i = 0; i<C->N; i++)
            tmp[i] = y[i];
    } else {
        R(C->lmax, u, cos(theta), sin(theta), ROT1);
        dot(C->N, ROT1, y, tmp);
    }

    // Convert it to a polynomial
    dot(C->N, C->A1, tmp, p);

    // Render it
    for (i=0; i<res; i++) {
        x[i] = -1. + 2. * i / (res - 1.);
    }
    for (i=0; i<res; i++) {
        for (j=0; j<res; j++) {
            // Are we inside the body?
            if (x[i] * x[i] + x[j] * x[j] < 1) {
                result[j][i] = poly(C->lmax, p, x[i], x[j]);
            } else {
                result[j][i] = NAN;
            }
        }
    }

    // Free
    for(i = 0; i<C->N; i++)
        delete [] ROT1[i];
    delete [] ROT1;
    delete [] tmp;
    delete [] p;
    delete [] x;

}

/**
Free the arrays in a CONSTANTS struct.

*/
void free_constants(int lmax, CONSTANTS *C) {
    int i;
    for(i = 0; i<C->N; i++) {
        delete [] C->A[i];
        delete [] C->A1[i];
    }
    delete [] C->A;
    delete [] C->A1;
    delete [] C->rT;
}
