/**
Compute the reflected flux. Static specialization.

*/
template <typename U=S, typename=IsDefaultOrSpectral<U>, 
          typename=IsReflected<U>, typename T1>
inline void computeFlux (
    const Scalar& theta, 
    const Scalar& xo, 
    const Scalar& yo, 
    const Scalar& zo, 
    const Scalar& ro, 
    const UnitVector<Scalar>& source,
    MatrixBase<T1> const & flux
) {
    computeReflectedFluxInternal(theta, xo, yo, zo, ro, source.normalized(), flux);
}

/**
Compute the reflected flux and its gradient. Static specialization.

*/
template <typename U=S, typename=IsDefaultOrSpectral<U>, 
          typename=IsReflected<U>,
          typename T1, typename T2, typename T3, typename T4, 
          typename T5, typename T6, typename T7, typename T8>
inline void computeFlux (
    const Scalar& theta, 
    const Scalar& xo, 
    const Scalar& yo, 
    const Scalar& zo, 
    const Scalar& ro, 
    const UnitVector<Scalar>& source,
    MatrixBase<T1> const & flux,
    MatrixBase<T2> const & Dtheta,
    MatrixBase<T3> const & Dxo,
    MatrixBase<T4> const & Dyo,
    MatrixBase<T5> const & Dro,
    MatrixBase<T6> const & Dy,
    MatrixBase<T7> const & Du,
    MatrixBase<T8> const & Dsource
) {
    // \todo Implement reflectance gradient
    throw errors::NotImplementedError("Gradient of reflectance not yet implemented.");
}

/**
Compute the reflected flux. Temporal specialization.

*/
template <typename U=S, typename=IsTemporal<U>, 
          typename=IsReflected<U>, typename T1>
inline void computeFlux (
    const Scalar& t,
    const Scalar& theta, 
    const Scalar& xo, 
    const Scalar& yo, 
    const Scalar& zo, 
    const Scalar& ro, 
    const UnitVector<Scalar>& source,
    MatrixBase<T1> const & flux
) {
    computeTaylor(t);
    computeReflectedFluxInternal(theta, xo, yo, zo, ro, source.normalized(), flux);
}

/**
Compute the reflected flux and its gradient. Temporal specialization.

*/
template <typename U=S, typename=IsTemporal<U>, 
          typename=IsReflected<U>,
          typename T1, typename T2, typename T3, typename T4, 
          typename T5, typename T6, typename T7, typename T8,
          typename T9>
inline void computeFlux (
    const Scalar& t,
    const Scalar& theta, 
    const Scalar& xo, 
    const Scalar& yo, 
    const Scalar& zo, 
    const Scalar& ro, 
    const UnitVector<Scalar>& source,
    MatrixBase<T1> const & flux, 
    MatrixBase<T2> const & Dt,
    MatrixBase<T3> const & Dtheta,
    MatrixBase<T4> const & Dxo,
    MatrixBase<T5> const & Dyo,
    MatrixBase<T6> const & Dro,
    MatrixBase<T7> const & Dy,
    MatrixBase<T8> const & Du,
    MatrixBase<T9> const & Dsource
) {
    // \todo Implement gradient of reflectance for temporal maps
    throw errors::NotImplementedError("Gradient of reflectance not yet implemented.");
}

/**
Evaluate the reflected map at a given (theta, x, y) coordinate.
Static specialization.

*/  
template <typename U=S, typename=IsDefaultOrSpectral<U>, 
          typename=IsReflected<U>, typename T1>
inline void computeIntensity (
    const Scalar& theta,
    const Scalar& x_,
    const Scalar& y_,
    const UnitVector<Scalar>& source,
    MatrixBase<T1> const & intensity
){
    computeReflectedIntensityInternal(theta, x_, y_, source.normalized(), intensity);
}

/**
Evaluate the reflected map at a given (theta, x, y) coordinate.
Temporal specialization.

*/  
template <typename U=S, typename=IsTemporal<U>, 
          typename=IsReflected<U>, typename T1>
inline void computeIntensity (
    const Scalar& t,
    const Scalar& theta,
    const Scalar& x_,
    const Scalar& y_,
    const UnitVector<Scalar>& source,
    MatrixBase<T1> const & intensity
){
    computeTaylor(t);
    computeReflectedIntensityInternal(theta, x_, y_, source.normalized(), intensity);
}

/**
Render the reflected map on a square cartesian grid at given
resolution. Static specialization.

*/
template <typename U=S, typename=IsDefaultOrSpectral<U>, 
          typename=IsReflected<U>, typename T1>
inline void renderMap (
    const Scalar& theta,
    const UnitVector<Scalar>& source,
    int res,
    MatrixBase<T1> const & intensity
){
    renderReflectedMapInternal(theta, source.normalized(), res, intensity);
}

/**
Render the reflected map on a square cartesian grid at given
resolution. Temporal specialization.

*/
template <typename U=S, typename=IsTemporal<U>, 
          typename=IsReflected<U>, typename T1>
inline void renderMap (
    const Scalar& t,
    const Scalar& theta,
    const UnitVector<Scalar>& source,
    int res,
    MatrixBase<T1> const & intensity
){
    computeTaylor(t);
    renderReflectedMapInternal(theta, source.normalized(), res, intensity);
}