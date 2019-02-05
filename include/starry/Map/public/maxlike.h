/**
Compute the max like map coefficients. Static specialization.

*/
template <typename U=S, typename=IsDefault<U>>
inline void computeMaxLikeMap (
    const Vector<Scalar>& flux, 
    const Vector<Scalar>& flux_err, 
    const Vector<Scalar>& theta, 
    const Vector<Scalar>& xo, 
    const Vector<Scalar>& yo, 
    const Vector<Scalar>& zo, 
    const Vector<Scalar>& ro, 
    const Matrix<Scalar>& L,
    Vector<Scalar>& yhat,
    Matrix<Scalar>& yvar
) {
    computeMaxLikeMapInternal(flux, flux_err, theta, xo, yo, zo, ro, L, yhat, yvar);
}

/**
Compute the max like map coefficients. Spectral / Temporal specialization.

*/
template <typename U=S, typename=IsSpectralOrTemporal<U>>
inline void computeMaxLikeMap (
    const Vector<Scalar>& flux, 
    const Vector<Scalar>& flux_err, 
    const Vector<Scalar>& t,
    const Vector<Scalar>& theta, 
    const Vector<Scalar>& xo, 
    const Vector<Scalar>& yo, 
    const Vector<Scalar>& zo, 
    const Vector<Scalar>& ro, 
    const Matrix<Scalar>& L,
    Vector<Scalar>& yhat,
    Matrix<Scalar>& yvar
) {
    throw errors::NotImplementedError(
        "`computeMaxLikeMap` not yet implemented for `Spectral` or `Temporal` maps.");
}