#include "starry.h"

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
