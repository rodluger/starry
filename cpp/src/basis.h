/**
Spherical harmonic, polynomial, and Green's basis utilities.

*/

#ifndef _STARRY_BASIS_H_
#define _STARRY_BASIS_H_

#include <cmath>
#include <Eigen/Core>
#include "fact.h"
#include "sqrtint.h"

namespace basis {

    // Contraction coefficient for the Ylms
    double C(int p, int q, int k) {
        if ((p > k) && ((p - k) % 2 == 0)) {
            return 0;
        } else if ((q > p) && ((q - p) % 2 == 0)) {
            return 0;
        } else {
            return fact::half_factorial(k) /
                        (fact::half_factorial(q) *
                         fact::half_factorial(k - p) *
                         fact::half_factorial(p - q));
        }
    }

    // Return the normalization constant A for a Ylm
    double Norm(int l, int m) {
        return sqrt((1. / (4 * M_PI)) *
                    (2 - (int)(m == 0)) *
                    (2 * l + 1) *
                    fact::factorial(l - abs(m)) /
                    fact::factorial(l + abs(m)));
    }

    // Return the B coefficient for a Ylm
    double B(int l, int m, int j, int k) {

        // Is it zero?
        int i1 = l + m + k - 1;
        int i2 = -l + m + k - 1;
        if ((i1 < 0) && (i1 % 2 == 0)) return INFINITY;
        else if ((i2 < 0) && (i2 % 2 == 0)) return 0;
        else if (m - j < 0) return 0;
        else if (l - m - k < 0) return 0;

        // Ok, let's calculate it
        double two_l = 1;
        for (int i=0; i < l; i++)
            two_l *= 2;
        double a = fact::factorial(m);
        double b = fact::half_factorial(i1);
        double c = fact::factorial(j);
        double d = fact::factorial(k);
        double e = fact::factorial(m - j);
        double f = fact::factorial(l - m - k);
        double g = fact::half_factorial(i2);
        return two_l * a * b / (c * d * e * f * g);
    }

    // Return the ijk tensor element of the spherical harmonic Ylm
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

    // Compute the first change of basis matrix, A_1
    void computeA1(int lmax, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& A1) {
        int l, m;
        int n = 0;
        int i, j, k, p, q, v;
        double coeff;
        double Ylm[lmax + 1][lmax + 1][2];

        // Iterate over the spherical harmonic orders and degrees
        for (l=0; l<lmax+1; l++) {
            for (m=-l; m<l+1; m++) {

                // Compute the contracted polynomial tensor
                for (i=0;i<l+1;i++) {
                    for (j=0;j<l+1;j++){
                        Ylm[i][j][0] = 0.;
                        Ylm[i][j][1] = 0.;
                    }
                }
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

                // Now contract the tensor down to a vector
                v = 0;
                for (i=0; i<l+1; i++) {
                    for (j=0; j<i+1; j++) {
                        A1(v, n) = Ylm[i][j][0];
                        v++;
                        if (j < i) {
                            A1(v, n) = Ylm[i][j][1];
                            v++;
                        }
                    }
                }

                // Next term in the vector
                n++;
            }
        }

        return;
    }

}; // namespace basis

#endif
