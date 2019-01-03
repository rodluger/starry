#ifndef _TEST_H_
#define _TEST_H_

#include <stdlib.h>
#include <iostream>
#include <Eigen/Core>

// Shorthand for vectors and matrices
template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T>
using RowVector = Eigen::Matrix<T, 1, Eigen::Dynamic>;
template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

// Enable boost functionality for multiprecision stuff
#ifndef STARRY_ENABLE_BOOST
#define STARRY_ENABLE_BOOST
#endif

// Turn on debug mode
#ifndef STARRY_DEBUG
#define STARRY_DEBUG
#endif

// Import starry
#include "starry2.h"

#endif