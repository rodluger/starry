/**
Compute the flux. Static specialization.

*/
template <typename U=S, typename=IsDefaultOrSpectral<U>, 
          typename=IsEmitted<U>, typename T1>
inline void computeFlux (
    const Scalar& theta, 
    const Scalar& xo, 
    const Scalar& yo, 
    const Scalar& zo, 
    const Scalar& ro, 
    MatrixBase<T1> const & flux
) {
    computeFluxInternal(theta, xo, yo, zo, ro, flux);
}

/**
Compute the flux and its gradient. Static specialization.

*/
template <typename U=S, typename=IsDefaultOrSpectral<U>,
          typename=IsEmitted<U>,
          typename T1, typename T2, typename T3, typename T4, 
          typename T5, typename T6, typename T7>
inline void computeFlux (
    const Scalar& theta, 
    const Scalar& xo, 
    const Scalar& yo, 
    const Scalar& zo, 
    const Scalar& ro, 
    MatrixBase<T1> const & flux,
    MatrixBase<T2> const & Dtheta,
    MatrixBase<T3> const & Dxo,
    MatrixBase<T4> const & Dyo,
    MatrixBase<T5> const & Dro,
    MatrixBase<T6> const & Dy,
    MatrixBase<T7> const & Du
) {
    Matrix<Scalar> Dt(1, nflx);
    computeFluxInternal(theta, xo, yo, zo, ro, flux, 
                    Dt, Dtheta, Dxo, Dyo, Dro, Dy, Du);
}

/**
Compute the flux. Temporal specialization.

*/
template <typename U=S, typename=IsTemporal<U>, 
          typename=IsEmitted<U>,
          typename T1>
inline void computeFlux (
    const Scalar& t,
    const Scalar& theta, 
    const Scalar& xo, 
    const Scalar& yo, 
    const Scalar& zo, 
    const Scalar& ro, 
    MatrixBase<T1> const & flux
) {
    computeTaylor(t);
    computeFluxInternal(theta, xo, yo, zo, ro, flux);
}

/**
Compute the flux and its gradient. Temporal specialization.

*/
template <typename U=S, typename=IsTemporal<U>,
          typename=IsEmitted<U>,
          typename T1, typename T2, typename T3, typename T4, 
          typename T5, typename T6, typename T7, typename T8>
inline void computeFlux (
    const Scalar& t,
    const Scalar& theta, 
    const Scalar& xo, 
    const Scalar& yo, 
    const Scalar& zo, 
    const Scalar& ro, 
    MatrixBase<T1> const & flux, 
    MatrixBase<T2> const & Dt,
    MatrixBase<T3> const & Dtheta,
    MatrixBase<T4> const & Dxo,
    MatrixBase<T5> const & Dyo,
    MatrixBase<T6> const & Dro,
    MatrixBase<T7> const & Dy,
    MatrixBase<T8> const & Du
) {
    computeTaylor(t);
    computeFluxInternal(theta, xo, yo, zo, ro, flux,
                    Dt, Dtheta, Dxo, Dyo, Dro, Dy, Du);
}