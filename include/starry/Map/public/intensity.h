/**
Compute the linear Ylm model. Default / Spectral specialization.

*/
template <
    typename U=S, 
    typename=IsDefaultOrSpectral<U>, 
    typename=IsEmitted<U>
>
inline void computeLinearIntensityModel (
    const Vector<Scalar>& theta, 
    const RowMatrix<Scalar>& x, 
    const RowMatrix<Scalar>& y,
    RowMatrix<Scalar>& X
) {
    RowMatrix<Scalar> source; // dummy
    computeLinearIntensityModelInternal(theta, x, y, source, X);
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
    const Vector<Scalar>& t,
    const Vector<Scalar>& theta, 
    const RowMatrix<Scalar>& x, 
    const RowMatrix<Scalar>& y,
    RowMatrix<Scalar>& X
) {
    RowMatrix<Scalar> source; // dummy
    computeTaylor(t);
    computeLinearIntensityModelInternal(theta, x, y, source, X);
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
    const Vector<Scalar>& theta, 
    const RowMatrix<Scalar>& x, 
    const RowMatrix<Scalar>& y,
    const RowMatrix<Scalar>& source,
    RowMatrix<Scalar>& X
) {
    computeLinearIntensityModelInternal(theta, x, y, source.rowwise().normalized(), X);
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
    const Vector<Scalar>& t,
    const Vector<Scalar>& theta, 
    const RowMatrix<Scalar>& x, 
    const RowMatrix<Scalar>& y,
    const RowMatrix<Scalar>& source,
    RowMatrix<Scalar>& X
) {
    computeTaylor(t);
    computeLinearIntensityModelInternal(theta, x, y, source.rowwise().normalized(), X);
}