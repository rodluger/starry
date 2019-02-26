/**
Compute the linear Ylm model. Default / Spectral specialization.

*/
template <
    typename U=S, 
    typename=IsDefaultOrSpectral<U>, 
    typename=IsEmitted<U>
>
inline void computeLinearIntensityModel (
    const Scalar& theta, 
    const RowMatrix<Scalar>& x, 
    const RowMatrix<Scalar>& y,
    RowMatrix<Scalar>& X
) {
    computeLinearIntensityModelInternal(theta, x, y, X);
}

/**
Compute the linear Ylm model. Temporal specialization.

*/
template <
    typename U=S, 
    typename=IsTemporal<U>, 
    typename=IsEmitted<U>
>
inline void computeLinearIntensityModel (
    const Scalar& t,
    const Scalar& theta, 
    const RowMatrix<Scalar>& x, 
    const RowMatrix<Scalar>& y,
    RowMatrix<Scalar>& X
) {
    computeTaylor(t);
    computeLinearIntensityModelInternal(theta, x, y, X);
}

/**
Compute the linear Ylm model. Default / Spectral specialization.

*/
template <
    typename U=S, 
    typename=IsDefaultOrSpectral<U>, 
    typename=IsReflected<U>
>
inline void computeLinearIntensityModel (
    const Scalar& theta, 
    const RowMatrix<Scalar>& x, 
    const RowMatrix<Scalar>& y,
    const UnitVector<Scalar>& source,
    RowMatrix<Scalar>& X
) {
    computeLinearIntensityModelInternal(theta, x, y, source, X);
}

/**
Compute the linear Ylm model. Temporal specialization.

*/
template <
    typename U=S, 
    typename=IsTemporal<U>, 
    typename=IsReflected<U>
>
inline void computeLinearIntensityModel (
    const Scalar& t,
    const Scalar& theta, 
    const RowMatrix<Scalar>& x, 
    const RowMatrix<Scalar>& y,
    const UnitVector<Scalar>& source,
    RowMatrix<Scalar>& X
) {
    computeTaylor(t);
    computeLinearIntensityModelInternal(theta, x, y, source, X);
}