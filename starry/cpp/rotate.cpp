#include "starry.h"

/**
Root of an integer.
TODO: Tabulate these.

*/
double root(int i){
    return sqrt((double)i);
}

/**
Inverse of the root of an integer.
TODO: Tabulate these.

*/
double rooti(int i){
    return (1. / sqrt((double)i));
}

/**
Compute the Wigner D matrices.

*/
void dlmn(int L, double s1, double c1, double c2, double TGBET2, double s3, double c3, double*** DL, double*** RL) {
    int IINF = 1 - L;
    int ISUP = -IINF;
    int M, MP;
    int AL, AL1, TAL1, AMP, LAUX, LBUX, AM, LAUZ, LBUZ;
    int SIGN;
    double ALI, AUZ, AUX, CUX, FACT, TERM, CUZ;
    double COSAUX, COSMAL, SINMAL, COSAG, SINAG, COSAGM, SINAGM, COSMGA, SINMGA;
    double D1, D2;

    // COMPUTATION OF THE DL[L;M',M) MATRIX, MP IS M' AND M IS M.
    // FIRST ROW BY RECURRENCE (SEE EQUATIONS 19 AND 20)
    DL[L][L][L] = DL[ISUP][ISUP][L - 1] * (1. + c2) / 2.;
    DL[L][-L][L] = DL[ISUP][-ISUP][L - 1] * (1. - c2) / 2.;
    for (M=ISUP; M>IINF-1; M--)
        DL[L][M][L] = -TGBET2 * root(L + M + 1) * rooti(L - M) * DL[L][M + 1][L];

    // THE ROWS OF THE UPPER QUARTER TRIANGLE OF THE DL[L;M',M) MATRIX
    // (SEE EQUATION 21)
    AL = L;
    AL1 = AL - 1;
    TAL1 = AL + AL1;
    ALI = (1. / (double) AL1);
    COSAUX = c2 * AL * AL1;
    for (MP=L-1; MP>-1; MP--) {
        AMP = MP;
        LAUX = L + MP;
        LBUX = L - MP;
        AUX = rooti(LAUX) * rooti(LBUX) * ALI;
        CUX = root(LAUX - 1) * root(LBUX - 1) * AL;
        for (M=ISUP; M>IINF-1; M--) {
            AM = M;
            LAUZ = L + M;
            LBUZ = L - M;
            AUZ = rooti(LAUZ) * rooti(LBUZ);
            FACT = AUX * AUZ;
            TERM = TAL1 * (COSAUX - AM * AMP) * DL[MP][M][L - 1];
            if ((LBUZ != 1) && (LBUX != 1)) {
                CUZ = root(LAUZ - 1) * root(LBUZ - 1);
                TERM = TERM - DL[MP][M][L - 2] * CUX * CUZ;
            }
            DL[MP][M][L] = FACT * TERM;
        }
        IINF = IINF + 1;
        ISUP = ISUP - 1;
    }

    // THE REMAINING ELEMENTS OF THE DL[L;M',M) MATRIX ARE CALCULATED
    // USING THE CORRESPONDING SYMMETRY RELATIONS:
    // REFLEXION ---> ((-1)**(M-M')) DL[L;M,M') = DL[L;M',M), M'<=M
    // INVERSION ---> ((-1)**(M-M')) DL[L;-M',-M) = DL[L;M',M)

    // REFLEXION
    SIGN = 1;
    IINF = -L;
    ISUP = L - 1;
    for (M=L; M>0; M--) {
        for (MP=IINF; MP<ISUP+1; MP++) {
            DL[MP][M][L] = SIGN * DL[M][MP][L];
            SIGN = -SIGN;
        }
        IINF = IINF + 1;
        ISUP = ISUP - 1;
    }
    // INVERSION
    IINF = -L;
    ISUP = IINF;
    for (M=L-1; M>-(L+1); M--) {
        SIGN = -1;
        for (MP=ISUP; MP>IINF-1; MP--) {
            DL[MP][M][L] = SIGN * DL[-MP][-M][L];
            SIGN = -SIGN;
        }
        ISUP = ISUP + 1;
    }

    // COMPUTATION OF THE ROTATION MATRICES RL[L;M',M) FOR REAL SPHERICAL
    // HARMONICS USING THE MATRICES DL[L;M',M) FOR COMPLEX SPHERICAL
    // HARMONICS (SEE EQUATIONS 10 TO 18)
    RL[0][0][L] = DL[0][0][L];
    COSMAL = c1;
    SINMAL = s1;
    SIGN = -1;
    for (MP=1; MP<L+1; MP++) {
        COSMGA = c3;
        SINMGA = s3;
        AUX = root(2) * DL[0][MP][L];
        RL[MP][0][L] = AUX * COSMAL;
        RL[-MP][0][L] = AUX * SINMAL;
        for (M=1; M<L+1; M++) {
            AUX = root(2) * DL[M][0][L];
            RL[0][M][L] = AUX * COSMGA;
            RL[0][-M][L] = -AUX * SINMGA;
            D1 = DL[-MP][-M][L];
            D2 = SIGN * DL[MP][-M][L];
            COSAG = COSMAL * COSMGA - SINMAL * SINMGA;
            COSAGM = COSMAL * COSMGA + SINMAL * SINMGA;
            SINAG = SINMAL * COSMGA + COSMAL * SINMGA;
            SINAGM = SINMAL * COSMGA - COSMAL * SINMGA;
            RL[MP][M][L] = D1 * COSAG + D2 * COSAGM;
            RL[MP][-M][L] = -D1 * SINAG + D2 * SINAGM;
            RL[-MP][M][L] = D1 * SINAG + D2 * SINAGM;
            RL[-MP][-M][L] = D1 * COSAG - D2 * COSAGM;
            AUX = COSMGA * c3 - SINMGA * s3;
            SINMGA = SINMGA * c3 + COSMGA * s3;
            COSMGA = AUX;
        }
        SIGN = -SIGN;
        AUX = COSMAL * c1 - SINMAL * s1;
        SINMAL = SINMAL * c1 + COSMAL * s1;
        COSMAL = AUX;
    }

    return;
}

/**
Compute the eulerian rotation matrix for real spherical
harmonics up to order lmax.

*/
void rotar(int lmax, double c1, double s1, double c2, double s2, double c3, double s3, double** matrix, double tol) {
    double COSAG, COSAMG, SINAG, SINAMG, TGBET2;
    int i, j, k, l, n;

    // Declare our RL and DL matrices
    double*** RL = new double**[2 * lmax + 1];
    double*** DL = new double**[2 * lmax + 1];
    for (i=0; i<2*lmax+1; i++) {
        RL[i] = new double*[2 * lmax + 1];
        DL[i] = new double*[2 * lmax + 1];
        for (j=0; j<2*lmax+1; j++) {
            RL[i][j] = new double[lmax + 1];
            DL[i][j] = new double[lmax + 1];
            for (k=0; k<lmax+1; k++) {
                RL[i][j][k] = 0;
                DL[i][j][k] = 0;
            }
        }
    }

    // Shift the pointers so the indices range from [-l, l][-l, l][0, l]
    RL = RL + lmax;
    DL = DL + lmax;
    for (i=-lmax; i<lmax+1; i++) {
        RL[i] = RL[i] + lmax;
        DL[i] = DL[i] + lmax;
    }

    // COMPUTATION OF THE INITIAL MATRICES D0, R0, D1 AND R1
    DL[0][0][0] = 1.;
    RL[0][0][0] = 1.;
    DL[1][1][1] = (1. + c2) / 2.;
    DL[1][0][1] = -s2 / root(2);
    DL[1][-1][1] = (1. - c2) / 2.;
    DL[0][1][1] = -DL[1][0][1];
    DL[0][0][1] = DL[1][1][1] - DL[1][-1][1];
    DL[0][-1][1] = DL[1][0][1];
    DL[-1][1][1] = DL[1][-1][1];
    DL[-1][0][1] = DL[0][1][1];
    DL[-1][-1][1] = DL[1][1][1];
    COSAG = c1 * c3 - s1 * s3;
    COSAMG = c1 * c3 + s1 * s3;
    SINAG = s1 * c3 + c1 * s3;
    SINAMG = s1 * c3 - c1 * s3;
    RL[0][0][1] = DL[0][0][1];
    RL[1][0][1] = root(2) * DL[0][1][1] * c1;
    RL[-1][0][1] = root(2) * DL[0][1][1] * s1;
    RL[0][1][1] = root(2) * DL[1][0][1] * c3;
    RL[0][-1][1] = -root(2) * DL[1][0][1] * s3;
    RL[1][1][1] = DL[1][1][1] * COSAG - DL[1][-1][1] * COSAMG;
    RL[1][-1][1] = -DL[1][1][1] * SINAG - DL[1][-1][1] * SINAMG;
    RL[-1][1][1] = DL[1][1][1] * SINAG - DL[1][-1][1] * SINAMG;
    RL[-1][-1][1] = DL[1][1][1] * COSAG + DL[1][-1][1] * COSAMG;

    // THE REMAINING MATRICES ARE CALCULATED USING SYMMETRY AND
    // RECURRENCE RELATIONS BY MEANS OF THE SUBROUTINE DLMN.
    if (abs(s2) < tol)
        TGBET2 = 0.;
    else
        TGBET2 = (1. - c2) / s2;

    for (l=2; l<lmax+1; l++)
        dlmn(l, s1, c1, c2, TGBET2, s3, c3, DL, RL);

    // Flatten RL into a block-diagonal matrix, matrix
    // matrix has dimensions [(lmax + 1)^2, (lmax + 1)^2]
    for (l=0; l<lmax+1; l++) {
        n = l * l;
        for (i=0; i<2*l + 1; i++) {
            for (j=0; j<2*l + 1; j++) {
                matrix[i + n][j + n] = RL[i - l][j - l][l];
            }
        }
    }

    // Unshift the matrix pointers
    for (i=-lmax; i<lmax+1; i++) {
        RL[i] = RL[i] - lmax;
        DL[i] = DL[i] - lmax;
    }
    RL = RL - lmax;
    DL = DL - lmax;

    // Free the matrices
    for(i=0; i<2*lmax+1; i++){
        for(j=0; j<2*lmax+1; j++){
            delete [] RL[i][j];
            delete [] DL[i][j];
        }
        delete [] RL[i];
        delete [] DL[i];
    }
    delete [] RL;
    delete [] DL;

    return;

}

/**
Compute the axis-angle rotation matrix for real spherical
harmonics up to order lmax.

*/
void R(int lmax, double u[3], double theta, double** matrix, double tol) {
    // Trivial case
    if (lmax == 0) {
        matrix[0][0] = 1;
        return;
    }

    // Construct the axis-angle rotation matrix R_A
    double costheta = cos(theta);
    double sintheta = sin(theta);
    double ux = u[0];
    double uy = u[1];
    double uz = u[2];
    double RA01 = ux * uy * (1 - costheta) - uz * sintheta;
    double RA02 = ux * uz * (1 - costheta) + uy * sintheta;
    double RA11 = costheta + uy * uy * (1 - costheta);
    double RA12 = uy * uz * (1 - costheta) - ux * sintheta;
    double RA20 = uz * ux * (1 - costheta) - uy * sintheta;
    double RA21 = uz * uy * (1 - costheta) + ux * sintheta;
    double RA22 = costheta + uz * uz * (1 - costheta);

    // Determine the Euler angles
    double cosalpha, sinalpha, cosbeta, sinbeta, cosgamma, singamma;
    double norm1, norm2;
    if ((RA22 < -1 + tol) && (RA22 > -1 - tol)) {
        cosbeta = -1;
        sinbeta = 0;
        cosgamma = RA11;
        singamma = RA01;
        cosalpha = 1;
        sinalpha = 0;
    } else if ((RA22 < 1 + tol) && (RA22 > 1 - tol)) {
        cosbeta = 1;
        sinbeta = 0;
        cosgamma = RA11;
        singamma = -RA01;
        cosalpha = 1;
        sinalpha = 0;
    } else {
        cosbeta = RA22;
        sinbeta = sqrt(1 - cosbeta * cosbeta);
        norm1 = sqrt(RA20 * RA20 + RA21 * RA21);
        norm2 = sqrt(RA02 * RA02 + RA12 * RA12);
        cosgamma = -RA20 / norm1;
        singamma = RA21 / norm1;
        cosalpha = RA02 / norm2;
        sinalpha = RA12 / norm2;
    }

    // Call the eulerian rotation function
    rotar(lmax, cosalpha, sinalpha, cosbeta,
          sinbeta, cosgamma, singamma, matrix, tol);

    return;

}
