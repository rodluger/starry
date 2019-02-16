/**
Compute the reflectance. Static specialization.

*/
template <typename U=S, typename=IsDefaultOrSpectral<U>, typename T1, 
          bool E=EMISSION, typename=typename std::enable_if<!E>::type>
inline void computeFlux (
    const Scalar& theta, 
    const Scalar& xo, 
    const Scalar& yo, 
    const Scalar& zo, 
    const Scalar& ro, 
    const UnitVector<Scalar>& source,
    MatrixBase<T1> const & flux
) {
    computeReflectanceInternal(theta, xo, yo, zo, ro, source, flux);
}

/**
Compute the flux and its gradient. Static specialization.

*/
template <typename U=S, typename=IsDefaultOrSpectral<U>, 
          bool E=EMISSION, typename=typename std::enable_if<!E>::type,
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
Compute the flux. Temporal specialization.

*/
template <typename U=S, typename=IsTemporal<U>, typename T1, 
          bool E=EMISSION, typename=typename std::enable_if<!E>::type>
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
    computeReflectanceInternal(theta, xo, yo, zo, ro, source, flux);
}

/**
Compute the flux and its gradient. Temporal specialization.

*/
template <typename U=S, typename=IsTemporal<U>, 
          bool E=EMISSION, typename=typename std::enable_if<!E>::type,
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