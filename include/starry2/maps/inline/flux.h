/**
Evaluate the map at a given (theta, x, y) coordinate
for a static map.

*/  
template<typename U=S, typename=IsStatic<U>, 
         typename T1>
inline void computeIntensity (
    const Scalar& theta,
    const Scalar& x_,
    const Scalar& y_,
    MatrixBase<T1> const & intensity
){
    computeIntensity_(theta, x_, y_, intensity);
}

/**
Render the visible map on a square cartesian grid at given
resolution for a temporal map.

*/
template<typename U=S, typename=IsTemporal<U>,
         typename T1>
inline void renderMap (
    const Scalar& t,
    const Scalar& theta,
    int res,
    MatrixBase<T1> const & intensity
){
    computeTaylor(t);
    renderMap_(theta, res, intensity);
}

/**
Evaluate the map at a given (theta, x, y) coordinate
for a temporal map.

*/  
template<typename U=S, typename=IsTemporal<U>, 
         typename T1>
inline void computeIntensity (
    const Scalar& t,
    const Scalar& theta,
    const Scalar& x_,
    const Scalar& y_,
    MatrixBase<T1> const & intensity
){
    computeTaylor(t);
    computeIntensity_(theta, x_, y_, intensity);
}

/**
Render the visible map on a square cartesian grid at given
resolution for a static map.

*/
template<typename U=S, typename=IsStatic<U>,
         typename T1>
inline void renderMap (
    const Scalar& theta,
    int res,
    MatrixBase<T1> const & intensity
){
    renderMap_(theta, res, intensity);
}

/**
Compute the flux for a static map.

*/
template<typename U=S, typename=IsStatic<U>,
         typename T1>
inline void computeFlux (
    const Scalar& theta, 
    const Scalar& xo, 
    const Scalar& yo, 
    const Scalar& ro, 
    MatrixBase<T1> const & flux
) {
    computeFlux_(theta, xo, yo, ro, flux);
}

/**
Compute the flux and its gradient for a static map.

*/
template<typename U=S, typename=IsStatic<U>,
         typename T1, typename T2, typename T3, typename T4, 
         typename T5, typename T6, typename T7>
inline void computeFlux (
    const Scalar& theta, 
    const Scalar& xo, 
    const Scalar& yo, 
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
    computeFlux_(theta, xo, yo, ro, flux, 
                 Dt, Dtheta, Dxo, Dyo, Dro, Dy, Du);
}

/**
Compute the flux for a temporal map.

*/
template<typename U=S, typename=IsTemporal<U>,
         typename T1>
inline void computeFlux (
    const Scalar& t,
    const Scalar& theta, 
    const Scalar& xo, 
    const Scalar& yo, 
    const Scalar& ro, 
    MatrixBase<T1> const & flux
) {
    computeTaylor(t);
    computeFlux_(theta, xo, yo, ro, flux);
}

/**
Compute the flux and its gradient for a temporal map.

*/
template<typename U=S, typename=IsTemporal<U>,
         typename T1, typename T2, typename T3, typename T4, 
         typename T5, typename T6, typename T7, typename T8>
inline void computeFlux (
    const Scalar& t,
    const Scalar& theta, 
    const Scalar& xo, 
    const Scalar& yo, 
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
    computeFlux_(theta, xo, yo, ro, flux,
                 Dt, Dtheta, Dxo, Dyo, Dro, Dy, Du);
}