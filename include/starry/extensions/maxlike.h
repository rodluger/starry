/**
\file maxlike.h
\brief This is a work in progress.

*/

#ifndef _STARRY_MAXLIKEE_H_
#define _STARRY_MAXLIKEE_H_

#include "../utils.h"
#include "../basis.h"
#include "../rotation.h"

namespace starry { 
namespace extensions {

using namespace starry::utils;

/**
\todo Compute the maximum likelihood map coefficients.

*/
template <typename T1, typename T2, typename T3>
inline void computeMaxLikeMapInternal (
    const Matrix<T1>& A,
    const Vector<T1>& flux, 
    const MatrixBase<T1>& C, 
    const MatrixBase<T2>& L,
    Vector<T1>& yhat,
    Matrix<T1>& yvar
) {
    // todo
}

} // namespace extensions
} // namespace starry

#endif