#define _USE_MATH_DEFINES
#include <iostream>
#include <iomanip>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_sf_ellint.h>
#include <cmath>
#include <pybind11/pybind11.h>
using namespace std;
namespace py = pybind11;

/**
The GREENS recurrence struct.

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
Constant vectors and matrices

*/
typedef struct {
    int lmax;
    int N;
    double** A;
    double** A1;
    double* rT;
} CONSTANTS;

// starry.cpp
void init_constants(int lmax, CONSTANTS* C);
void free_constants(int lmax, CONSTANTS* C);
void flux(int NT, double* y, double u[3], double* theta, double* x0, double* y0, double r, CONSTANTS* C, double* result);

// utils.cpp
double factorial(int n);
double factorial(double n);
double step(double x);
void dot(int N, double** A, double** B, double** AB);
void dot(int N, double** A, double* b, double* Ab);
double dot(int N, double* a, double* b);
double ellipK(double ksq, gsl_mode_t mode=GSL_PREC_DOUBLE);
double ellipE(double ksq, gsl_mode_t mode=GSL_PREC_DOUBLE);
double ellipPI(double n, double ksq, gsl_mode_t mode=GSL_PREC_DOUBLE);
void invert(int N, double** invmat, double **matrix, double tol=1e-10);

// basis.cpp
void A1(int lmax, double** matrix);
void A2(int lmax, double** matrix);
void A(int lmax, double** matrix);

// rotate.cpp
void R(int lmax, double u[3], double costheta, double sintheta, double** matrix, double tol=1e-15);

// integrate.cpp
void rT(int lmax, double* vector);
void sT(int lmax, double b, double r, double* vector);
