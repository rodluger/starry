#define _USE_MATH_DEFINES
#include <iostream>
#include <iomanip>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_sf_ellint.h>
#include <cmath>
using namespace std;

// utils.cpp
double factorial(int n);
double factorial(double n);
double step(double x);
double ellipK(double ksq, gsl_mode_t mode=GSL_PREC_DOUBLE);
double ellipE(double ksq, gsl_mode_t mode=GSL_PREC_DOUBLE);
double ellipPI(double nsq, double ksq, gsl_mode_t mode=GSL_PREC_DOUBLE);
void invert(int N, double** invmat, double **matrix, double tol=1e-10);

// basis.cpp
void A1(int lmax, double** matrix);
void A2(int lmax, double** matrix);
void A(int lmax, double** matrix);

// rotate.cpp
void R(int lmax, double u[3], double theta, double** matrix, double tol=1e-15);

// integrate.cpp
void r(int lmax, double* vector);
void s(int lmax, double b, double r, double* vector);
