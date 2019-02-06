/**
Compute the linear Ylm model. Default / Spectral specialization.

*/
template <typename U=S, typename=IsDefaultOrSpectral<U>>
inline void computeLinearModel (
    const Vector<Scalar>& theta, 
    const Vector<Scalar>& xo, 
    const Vector<Scalar>& yo, 
    const Vector<Scalar>& zo, 
    const Vector<Scalar>& ro, 
    Matrix<Scalar>& A
) {
    computeLinearModelInternal(theta, xo, yo, zo, ro, A);
}

/**
Compute the linear Ylm model. Temporal specialization.

*/
template <typename U=S, typename=IsTemporal<U>>
inline void computeLinearModel (
    const Vector<Scalar>& t,
    const Vector<Scalar>& theta, 
    const Vector<Scalar>& xo, 
    const Vector<Scalar>& yo, 
    const Vector<Scalar>& zo, 
    const Vector<Scalar>& ro, 
    Matrix<Scalar>& A
) {
    throw errors::NotImplementedError(
        "`computeLinearModel` not yet implemented for `Spectral` or `Temporal` maps.");
}

/**
Compute the max like map coefficients. Default specialization.

*/
template <typename U=S>
inline IsDefault<U, void> computeMaxLikeMap (
    const Matrix<Scalar>& A,
    const Vector<Scalar>& flux, 
    const Vector<Scalar>& C, 
    const Vector<Scalar>& L,
    Vector<Scalar>& yhat,
    Matrix<Scalar>& yvar
) {
    computeMaxLikeMapInternal(A, flux, C, L, yhat, yvar);
}

/**
Compute the max like map coefficients. Spectral / Temporal specialization.

*/
template <typename U=S>
inline IsSpectralOrTemporal<U, void> computeMaxLikeMap (
    const Matrix<Scalar>& A,
    const Vector<Scalar>& flux, 
    const Vector<Scalar>& C, 
    const Vector<Scalar>& L,
    Vector<Scalar>& yhat,
    Matrix<Scalar>& yvar
) {
    throw errors::NotImplementedError(
        "`computeMaxLikeMap` not yet implemented for `Spectral` or `Temporal` maps.");
}