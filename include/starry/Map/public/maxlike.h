/**
Compute the max like map coefficients. Static specialization.

*/
template <typename U=S, typename=IsDefault<U>>
inline void computeMaxLikeMap (
    const Vector<Scalar>& flux, 
    const Vector<Scalar>& C, 
    const Vector<Scalar>& L,
    const Vector<Scalar>& theta, 
    const Vector<Scalar>& xo, 
    const Vector<Scalar>& yo, 
    const Vector<Scalar>& zo, 
    const Vector<Scalar>& ro, 
    Matrix<Scalar>& A,
    Vector<Scalar>& yhat,
    Matrix<Scalar>& yvar
) {
    computeMaxLikeMapInternal(flux, C, L, theta, xo, yo, zo, ro, A, yhat, yvar);
}

/**
Compute the max like map coefficients. Spectral / Temporal specialization.

*/
template <typename U=S, typename=IsSpectralOrTemporal<U>>
inline void computeMaxLikeMap (
    const Vector<Scalar>& flux, 
    const Vector<Scalar>& C, 
    const Vector<Scalar>& L,
    const Vector<Scalar>& t,
    const Vector<Scalar>& theta, 
    const Vector<Scalar>& xo, 
    const Vector<Scalar>& yo, 
    const Vector<Scalar>& zo, 
    const Vector<Scalar>& ro,
    Matrix<Scalar>& A,
    Vector<Scalar>& yhat,
    Matrix<Scalar>& yvar
) {
    throw errors::NotImplementedError(
        "`computeMaxLikeMap` not yet implemented for `Spectral` or `Temporal` maps.");
}