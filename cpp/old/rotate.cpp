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
void dlmn(int l, double s1, double c1, double c2, double tgbet2, double s3,
          double c3, double*** DL, double*** RL) {
    int iinf = 1 - l;
    int isup = -iinf;
    int m, mp;
    int al, al1, tal1, amp, laux, lbux, am, lauz, lbuz;
    int sign;
    double ali, auz, aux, cux, fact, term, cuz;
    double cosaux, cosmal, sinmal, cosag, sinag, cosagm, sinagm, cosmga, sinmga;
    double D1, D2;

    // Compute the DL[l;m',m) matrix.
    // First row by recurrence (Eq. 19 and 20 in Alvarez Collado et al.)
    DL[2 * l][2 * l][l] = DL[isup + l - 1][isup + l - 1][l - 1] * (1. + c2) / 2.;
    DL[2 * l][0][l] = DL[isup + l - 1][-isup + l - 1][l - 1] * (1. - c2) / 2.;
    for (m=isup; m>iinf-1; m--)
        DL[2 * l][m + l][l] = -tgbet2 * root(l + m + 1) *
                              rooti(l - m) * DL[2 * l][m + 1 + l][l];

    // The rows of the upper quarter triangle of the DL[l;m',m) matrix
    // (Eq. 21 in Alvarez Collado et al.)
    al = l;
    al1 = al - 1;
    tal1 = al + al1;
    ali = (1. / (double) al1);
    cosaux = c2 * al * al1;
    for (mp=l-1; mp>-1; mp--) {
        amp = mp;
        laux = l + mp;
        lbux = l - mp;
        aux = rooti(laux) * rooti(lbux) * ali;
        cux = root(laux - 1) * root(lbux - 1) * al;
        for (m=isup; m>iinf-1; m--) {
            am = m;
            lauz = l + m;
            lbuz = l - m;
            auz = rooti(lauz) * rooti(lbuz);
            fact = aux * auz;
            term = tal1 * (cosaux - am * amp) * DL[mp + l - 1][m + l - 1][l - 1];
            if ((lbuz != 1) && (lbux != 1)) {
                cuz = root(lauz - 1) * root(lbuz - 1);
                term = term - DL[mp + l - 2][m + l - 2][l - 2] * cux * cuz;
            }
            DL[mp + l][m + l][l] = fact * term;
        }
        iinf = iinf + 1;
        isup = isup - 1;
    }

    // The remaining elements of the DL[l;m',m) matrix are calculated
    // using the corresponding symmetry relations:
    // reflection ---> ((-1)**(m-m')) DL[l;m,m') = DL[l;m',m), m'<=m
    // inversion ---> ((-1)**(m-m')) DL[l;-m',-m) = DL[l;m',m)

    // Reflection
    sign = 1;
    iinf = -l;
    isup = l - 1;
    for (m=l; m>0; m--) {
        for (mp=iinf; mp<isup+1; mp++) {
            DL[mp + l][m + l][l] = sign * DL[m + l][mp + l][l];
            sign = -sign;
        }
        iinf = iinf + 1;
        isup = isup - 1;
    }

    // Inversion
    iinf = -l;
    isup = iinf;
    for (m=l-1; m>-(l+1); m--) {
        sign = -1;
        for (mp=isup; mp>iinf-1; mp--) {
            DL[mp + l][m + l][l] = sign * DL[-mp + l][-m + l][l];
            sign = -sign;
        }
        isup = isup + 1;
    }

    // Compute the real rotation matrices RL from the complex ones DL
    RL[0 + l][0 + l][l] = DL[0 + l][0 + l][l];
    cosmal = c1;
    sinmal = s1;
    sign = -1;
    for (mp=1; mp<l+1; mp++) {
        cosmga = c3;
        sinmga = s3;
        aux = root(2) * DL[0 + l][mp + l][l];
        RL[mp + l][0 + l][l] = aux * cosmal;
        RL[-mp + l][0 + l][l] = aux * sinmal;
        for (m=1; m<l+1; m++) {
            aux = root(2) * DL[m + l][0 + l][l];
            RL[l][m + l][l] = aux * cosmga;
            RL[l][-m + l][l] = -aux * sinmga;
            D1 = DL[-mp + l][-m + l][l];
            D2 = sign * DL[mp + l][-m + l][l];
            cosag = cosmal * cosmga - sinmal * sinmga;
            cosagm = cosmal * cosmga + sinmal * sinmga;
            sinag = sinmal * cosmga + cosmal * sinmga;
            sinagm = sinmal * cosmga - cosmal * sinmga;
            RL[mp + l][m + l][l] = D1 * cosag + D2 * cosagm;
            RL[mp + l][-m + l][l] = -D1 * sinag + D2 * sinagm;
            RL[-mp + l][m + l][l] = D1 * sinag + D2 * sinagm;
            RL[-mp + l][-m + l][l] = D1 * cosag - D2 * cosagm;
            aux = cosmga * c3 - sinmga * s3;
            sinmga = sinmga * c3 + cosmga * s3;
            cosmga = aux;
        }
        sign = -sign;
        aux = cosmal * c1 - sinmal * s1;
        sinmal = sinmal * c1 + cosmal * s1;
        cosmal = aux;
    }

    return;
}

/**
Compute the eulerian rotation matrix for real spherical
harmonics up to order lmax.

*/
void rotar(int lmax, double c1, double s1, double c2, double s2,
           double c3, double s3, double** matrix, double tol) {
    double cosag, COSAMG, sinag, SINAMG, tgbet2;
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

    // Compute the initial matrices D0, R0, D1 and R1
    DL[0][0][0] = 1.;
    RL[0][0][0] = 1.;
    DL[2][2][1] = (1. + c2) / 2.;
    DL[2][1][1] = -s2 / root(2);
    DL[2][0][1] = (1. - c2) / 2.;
    DL[1][2][1] = -DL[2][1][1];
    DL[1][1][1] = DL[2][2][1] - DL[2][0][1];
    DL[1][0][1] = DL[2][1][1];
    DL[0][2][1] = DL[2][0][1];
    DL[0][1][1] = DL[1][2][1];
    DL[0][0][1] = DL[2][2][1];
    cosag = c1 * c3 - s1 * s3;
    COSAMG = c1 * c3 + s1 * s3;
    sinag = s1 * c3 + c1 * s3;
    SINAMG = s1 * c3 - c1 * s3;
    RL[1][1][1] = DL[1][1][1];
    RL[2][1][1] = root(2) * DL[1][2][1] * c1;
    RL[0][1][1] = root(2) * DL[1][2][1] * s1;
    RL[1][2][1] = root(2) * DL[2][1][1] * c3;
    RL[1][0][1] = -root(2) * DL[2][1][1] * s3;
    RL[2][2][1] = DL[2][2][1] * cosag - DL[2][0][1] * COSAMG;
    RL[2][0][1] = -DL[2][2][1] * sinag - DL[2][0][1] * SINAMG;
    RL[0][2][1] = DL[2][2][1] * sinag - DL[2][0][1] * SINAMG;
    RL[0][0][1] = DL[2][2][1] * cosag + DL[2][0][1] * COSAMG;

    // The remaining matrices are calculated using symmetry and
    // and recurrence relations
    if (abs(s2) < tol)
        tgbet2 = 0.;
    else
        tgbet2 = (1. - c2) / s2;

    for (l=2; l<lmax+1; l++)
        dlmn(l, s1, c1, c2, tgbet2, s3, c3, DL, RL);

    // Flatten RL into a block-diagonal matrix, matrix
    // matrix has dimensions [(lmax + 1)^2, (lmax + 1)^2]
    for (l=0; l<lmax+1; l++) {
        n = l * l;
        for (i=0; i<2*l + 1; i++) {
            for (j=0; j<2*l + 1; j++) {
                matrix[i + n][j + n] = RL[i][j][l];
            }
        }
    }

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
void R(int lmax, double u[3], double costheta, double sintheta, double** matrix, double tol) {
    // Trivial case
    if (lmax == 0) {
        matrix[0][0] = 1;
        return;
    }

    // Zero out the matrix
    int N = (lmax + 1) * (lmax + 1);
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++)
            matrix[i][j] = 0;
    }

    // Construct the axis-angle rotation matrix R_A
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
