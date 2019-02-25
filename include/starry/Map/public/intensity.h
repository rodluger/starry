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
    const Vector<Scalar>& x, 
    const Vector<Scalar>& y,
    RowMatrix<Scalar>& A
) {
    computeLinearIntensityModelInternal(theta, x, y, A);
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
    const Vector<Scalar>& x, 
    const Vector<Scalar>& y,
    RowMatrix<Scalar>& A
) {
    computeTaylor(t);
    computeLinearIntensityModelInternal(theta, x, y, A);
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
    const Vector<Scalar>& x, 
    const Vector<Scalar>& y,
    const UnitVector<Scalar>& source,
    RowMatrix<Scalar>& A
) {
    computeLinearIntensityModelInternal(theta, x, y, source, A);
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
    const Vector<Scalar>& x, 
    const Vector<Scalar>& y,
    const UnitVector<Scalar>& source,
    RowMatrix<Scalar>& A
) {
    computeTaylor(t);
    computeLinearIntensityModelInternal(theta, x, y, source, A);
}

/**
Render the visible map on a square cartesian grid at given
resolution. Static specialization.

*/
template <
    typename U=S, 
    typename=IsDefaultOrSpectral<U>, 
    typename=IsEmitted<U>, 
    typename T1
>
inline void renderMap (
    const Scalar& theta,
    int res,
    MatrixBase<T1> const & intensity
){
    // \todo renderMapInternal(theta, res, intensity);
}

/**
Render the visible map on a square cartesian grid at given
resolution. Temporal specialization.

*/
template <
    typename U=S, 
    typename=IsDefaultOrSpectral<U>, 
    typename=IsEmitted<U>, 
    typename T1
>
inline void renderMap (
    const Scalar& t,
    const Scalar& theta,
    int res,
    MatrixBase<T1> const & intensity
){
    // computeTaylor(t);
    // \todo renderMapInternal(theta, res, intensity);
}