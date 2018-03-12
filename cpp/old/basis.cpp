#include "starry.h"

/**
Evaluate a polynomial vector at a given (x, y) coordinate.
*/
double poly(int lmax, double* p, double x, double y) {
    int N = (lmax + 1) * (lmax + 1);
    double z = sqrt(1 - x * x - y * y);
    double* basis = new double[N];
    int l, m, mu, nu;
    int n = 0;
    double res;

    // Compute the basis
    for (l=0; l<lmax+1; l++) {
        for (m=-l; m<l+1; m++) {
            mu = l - m;
            nu = l + m;
            if ((nu % 2) == 0)
                basis[n] = pow(x, mu / 2) * pow(y, nu / 2);
            else
                basis[n] = pow(x, (mu - 1) / 2) * pow(y, (nu - 1) / 2) * z;
            n++;
        }
    }

    // Dot the coefficients in
    res = dot(N, p, basis);
    delete [] basis;

    return res;
}

/**
Contraction coefficient for the Ylms (Equation 10)

*/
double C(int p, int q, int k) {
    return factorial(0.5 * k) / (factorial(0.5 * q) *
                                 factorial(0.5 * (k - p)) *
                                 factorial(0.5 * (p - q)));
}

/**
Return the normalization constant A for a Ylm (Equation 5)

*/
double Norm(int l, int m) {
    return sqrt((1. / (4 * M_PI)) *
                (2 - (m == 0)) *
                (2 * l + 1) *
                factorial(l - abs(m)) /
                factorial(l + abs(m)));
}

/**
Return the B coefficient for a Ylm (Equation 8)

*/
double B(int l, int m, int j, int k) {
    double two_l = 1;
    for (int i=0; i < l; i++) two_l *= 2;
    return (two_l * factorial(m) * factorial(0.5 * (l + m + k - 1))) /
           (factorial(j) * factorial(k) * factorial(m - j) *
            factorial(l - m - k) * factorial(0.5 * (-l + m + k - 1)));
}

/**
Return the ijk tensor element of the spherical harmonic Ylm

*/
double Lijk(int l, int m, int i, int j, int k) {
    if ((i == abs(m) + k) && (j <= abs(m))) {
        if ((m >= 0) && (j % 2 == 0)) {
            if ((j / 2) % 2 == 0)
                return Norm(l, m) * B(l, m, j, k);
            else
                return -Norm(l, m) * B(l, m, j, k);
        } else if ((m < 0) && (j % 2 == 1)) {
            if (((j - 1) / 2) % 2 == 0)
                return Norm(l, -m) * B(l, -m, j, k);
            else
                return -Norm(l, -m) * B(l, -m, j, k);
        } else {
            return 0;
        }
    } else {
        return 0;
    }
}

/**
Compute a vector of polynomial coefficients corresponding to a Ylm

*/
void Y(int l, int m, double* y) {
    int i, j, k, p, q, n;
    double coeff;
    double Ylm[l + 1][l + 1][2];
    for (i=0;i<l+1;i++) {
        for (j=0;j<l+1;j++){
            Ylm[i][j][0] = 0.;
            Ylm[i][j][1] = 0.;
        }
    }

    // Compute the contracted polynomial tensor
    for (k=0; k<l+1; k++) {
        for (i=k; i<l+1; i++) {
            for (j=0; j<i-k+1; j++) {
                coeff = Lijk(l, m, i, j, k);
                if (coeff) {
                    if ((k == 0) || (k == 1)) {
                        // 1 or z
                        Ylm[i][j][k] += coeff;
                    } else if ((k % 2) == 0) {
                        // Even power of z
                        for (p=0; p<k+1; p+=2) {
                            for (q=0; q<p+1; q+=2) {
                                if ((p / 2) % 2 == 0)
                                    Ylm[i - k + p][j + q][0] +=
                                        C(p, q, k) * coeff;
                                else
                                    Ylm[i - k + p][j + q][0] -=
                                        C(p, q, k) * coeff;
                            }
                        }
                    } else {
                        // Odd power of z
                        for (p=0; p<k+1; p+=2) {
                            for (q=0; q<p+1; q+=2) {
                                if ((p / 2) % 2 == 0)
                                    Ylm[i - k + p + 1][j + q][1] +=
                                        C(p, q, k - 1) * coeff;
                                else
                                    Ylm[i - k + p + 1][j + q][1] -=
                                        C(p, q, k - 1) * coeff;
                            }
                        }
                    }
                }
            }
        }
    }

    // Now we further contract the tensor down to a vector
    n = 0;
    for (i=0; i<l+1; i++) {
        for (j=0; j<i+1; j++) {
            y[n] = Ylm[i][j][0];
            n++;
            if (j < i) {
                y[n] = Ylm[i][j][1];
                n++;
            }
        }
    }

    return;
}

/**
Compute the first change of basis matrix, A_1

*/
void A1(int lmax, double** matrix) {
    int i, j, l, m;
    int N = (lmax + 1) * (lmax + 1);
    int n = 0;
    double tmp[N][N];
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++)
            tmp[i][j] = 0;
    }

    // Compute the spherical harmonic vectors
    for (l=0; l<lmax+1; l++) {
        for (m=-l; m<l+1; m++) {
            Y(l, m, tmp[n]);
            n++;
        }
    }

    // Transpose the result
    for (i = 0; i < N; i++){
        for (j = 0; j < N; j++){
            matrix[j][i]= tmp[i][j];
        }
    }

    return;
}

/**
Compute the second change of basis matrix, A_2.

*/
void A2(int lmax, double **matrix) {
    int i, j, n, l, m, mu, nu;
    int N = (lmax + 1) * (lmax + 1);

    // Let's compute the inverse of the matrix, since it's easier
    double **invmat = new double*[N];
    for (i=0; i<N; i++) {
        invmat[i] = new double[N];
        for (j=0; j<N; j++)
            invmat[i][j] = 0;
    }

    // Loop over the columns
    n = 0;
    for (l=0; l<lmax+1; l++) {
        for (m=-l; m<l+1; m++){
            mu = l - m;
            nu = l + m;
            if (nu % 2 == 0) {
                // x^(mu/2) y^(nu/2)
                invmat[n][n] = (mu + 2) / 2;
            } else if ((l == 1) && (m == 0)) {
                // z
                invmat[n][n] = 1;
            } else if ((mu == 1) && (l % 2 == 0)) {
                // x^(l-2) y z
                i = l * l + 3;
                invmat[i][n] = 3;
            } else if ((mu == 1) && (l % 2 == 1)) {
                // x^(l-3) z
                i = 1 + (l - 2) * (l - 2);
                invmat[i][n] = -1;
                // x^(l-1) z
                i = l * l + 1;
                invmat[i][n] = 1;
                // x^(l-3) y^2 z
                i = l * l + 5;
                invmat[i][n] = 4;
            } else {
                if (mu != 3) {
                    // x^((mu - 5)/2) y^((nu - 1)/2)
                    i = nu + ((mu - 4 + nu) * (mu - 4 + nu)) / 4;
                    invmat[i][n] = (mu - 3) / 2;
                    // x^((mu - 5)/2) y^((nu + 3)/2)
                    i = nu + 4 + ((mu + nu) * (mu + nu)) / 4;
                    invmat[i][n] = -(mu - 3) / 2;
                }
                // x^((mu - 1)/2) y^((nu - 1)/2)
                i = nu + (mu + nu) * (mu + nu) / 4;
                invmat[i][n] = -(mu + 3) / 2;
            }
            n++;
        }
    }

    // Compute the inverse
    invert(N, invmat, matrix);

    // Free
    for(i = 0; i<N; i++)
        delete [] invmat[i];
    delete [] invmat;

    return;
}

/**
Compute the full change of basis matrix, A.

*/
void A(int lmax, double **matrix) {
    int i;
    int N = (lmax + 1) * (lmax + 1);

    // Initialize an empty matrix
    double **a1 = new double*[N];
    double **a2 = new double*[N];
    for (i=0; i<N; i++) {
        a1[i] = new double[N];
        a2[i] = new double[N];
    }

    // Compute A1 and A2
    A1(lmax, a1);
    A2(lmax, a2);

    // Dot them
    dot(N, a2, a1, matrix);

    // Free
    for (i=0; i<N; i++) {
        delete [] a1[i];
        delete [] a2[i];
    }
    delete [] a1;
    delete [] a2;

    return;
}
