/**
Evaluate the map at a given (theta, x, y) coordinate.
Static specialization.

*/  
template <typename U=S, typename=IsDefaultOrSpectral<U>, typename T1>
inline void computeIntensity (
    const Scalar& theta,
    const Scalar& x_,
    const Scalar& y_,
    MatrixBase<T1> const & intensity
){
    computeIntensity_(theta, x_, y_, intensity);
}

/**
Evaluate the map at a given (theta, x, y) coordinate.
Temporal specialization.

*/  
template <typename U=S, typename=IsTemporal<U>, typename T1>
inline void computeIntensity (
    const Scalar& t,
    const Scalar& theta,
    const Scalar& x_,
    const Scalar& y_,
    MatrixBase<T1> const & intensity
){
    computeTaylor_(t);
    computeIntensity_(theta, x_, y_, intensity);
}

/**
Render the visible map on a square cartesian grid at given
resolution. Static specialization.

*/
template <typename U=S, typename=IsDefaultOrSpectral<U>, typename T1>
inline void renderMap (
    const Scalar& theta,
    int res,
    MatrixBase<T1> const & intensity
){
    renderMap_(theta, res, intensity);
}

/**
Render the visible map on a square cartesian grid at given
resolution. Temporal specialization.

*/
template <typename U=S, typename=IsTemporal<U>, typename T1>
inline void renderMap (
    const Scalar& t,
    const Scalar& theta,
    int res,
    MatrixBase<T1> const & intensity
){
    computeTaylor_(t);
    renderMap_(theta, res, intensity);
}