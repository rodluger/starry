#define _USE_MATH_DEFINES
#include <iostream>
#include <iomanip>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_linalg.h>
#include <cmath>
using namespace std;

// utils.cpp
double factorial(int n);
double factorial(double n);
void invert(int N, double** invmat, double **matrix, double tol=1e-10);

// basis.cpp
void A1(int lmax, double** matrix);
void A2(int lmax, double** matrix);
void A(int lmax, double** matrix);

// rotate.cpp
void R(int lmax, double u[3], double theta, double** matrix, double tol=1e-15);
