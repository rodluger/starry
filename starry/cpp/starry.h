#define _USE_MATH_DEFINES
#include <iostream>
#include <iomanip>
#include <gsl/gsl_sf_gamma.h>
#include <cmath>
using namespace std;

// utils.cpp
double factorial(int n);
double factorial(double n);

// basis.cpp
void A1(int lmax, double** matrix);
