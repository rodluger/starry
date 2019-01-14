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
    MatrixBase<T2> const & dtheta,
    MatrixBase<T3> const & dxo,
    MatrixBase<T4> const & dyo,
    MatrixBase<T5> const & dro,
    MatrixBase<T6> const & dy,
    MatrixBase<T7> const & du
) {
    Matrix<Scalar> dt(1, nflx);
    computeFlux_(theta, xo, yo, ro, flux, 
                 dt, dtheta, dxo, dyo, dro, dy, du);
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
    MatrixBase<T2> const & dt,
    MatrixBase<T3> const & dtheta,
    MatrixBase<T4> const & dxo,
    MatrixBase<T5> const & dyo,
    MatrixBase<T6> const & dro,
    MatrixBase<T7> const & dy,
    MatrixBase<T8> const & du
) {
    computeTaylor(t);
    computeFlux_(theta, xo, yo, ro, flux,
                 dt, dtheta, dxo, dyo, dro, dy, du);
}