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
    UnitVector<Scalar> source; // dummy
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
    const Scalar& t,
    const Scalar& theta, 
    const RowMatrix<Scalar>& x, 
    const RowMatrix<Scalar>& y,
    RowMatrix<Scalar>& X
) {
    UnitVector<Scalar> source; // dummy
    Vector<Scalar> tvec(1);
    tvec << t;
    computeTaylor(tvec);
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
    Vector<Scalar> tvec(1);
    tvec << t;
    computeTaylor(tvec);
    computeLinearIntensityModelInternal(theta, x, y, source, X);
}