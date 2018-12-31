/**
Evaluate the map at a given (theta, x, y) coordinate
for a static map.

*/  
template<typename U=S, typename=IsStatic<U>>
inline void computeIntensity (
    const Scalar& theta,
    const Scalar& x_,
    const Scalar& y_,
    Ref<FluxType> intensity
){
    computeIntensity_(0, theta, x_, y_, intensity);
};

/**
Render the visible map on a square cartesian grid at given
resolution for a temporal map.

*/
template<typename Derived, typename U=S, typename=IsTemporal<U>>
inline void renderMap (
    const Scalar& t,
    const Scalar& theta,
    int res,
    MatrixBase<Derived> const & intensity
){
    renderMap_(t, theta, res, intensity);
};

/**
Evaluate the map at a given (theta, x, y) coordinate
for a temporal map.

*/  
template<typename U=S, typename=IsTemporal<U>>
inline void computeIntensity (
    const Scalar& t,
    const Scalar& theta,
    const Scalar& x_,
    const Scalar& y_,
    Ref<FluxType> intensity
){
    computeIntensity_(t, theta, x_, y_, intensity);
};

/**
Render the visible map on a square cartesian grid at given
resolution for a static map.

*/
template<typename Derived, typename U=S, typename=IsStatic<U>>
inline void renderMap (
    const Scalar& theta,
    int res,
    MatrixBase<Derived> const & intensity
){
    renderMap_(0, theta, res, intensity);
};

/**
Compute the flux for a static map.

*/
template<typename U=S, typename=IsStatic<U>>
inline void computeFlux (
    const Scalar& theta, 
    const Scalar& xo, 
    const Scalar& yo, 
    const Scalar& ro, 
    Ref<FluxType> flux
) {
    computeFlux_(0, theta, xo, yo, ro, flux);
};

/**
Compute the flux and its gradient for a static map.

*/
template<typename U=S, typename=IsStatic<U>>
inline void computeFlux (
    const Scalar& theta, 
    const Scalar& xo, 
    const Scalar& yo, 
    const Scalar& ro, 
    Ref<FluxType> flux, 
    Ref<GradType> gradient
) {
    computeFlux_(0, theta, xo, yo, ro, flux, gradient);
};

/**
Compute the flux for a temporal map.

*/
template<typename U=S, typename=IsTemporal<U>>
inline void computeFlux (
    const Scalar& t,
    const Scalar& theta, 
    const Scalar& xo, 
    const Scalar& yo, 
    const Scalar& ro, 
    Ref<FluxType> flux
) {
    computeFlux_(t, theta, xo, yo, ro, flux);
};

/**
Compute the flux and its gradient for a temporal map.

*/
template<typename U=S, typename=IsTemporal<U>>
inline void computeFlux (
    const Scalar& t,
    const Scalar& theta, 
    const Scalar& xo, 
    const Scalar& yo, 
    const Scalar& ro, 
    Ref<FluxType> flux, 
    Ref<GradType> gradient
) {
    computeFlux_(t, theta, xo, yo, ro, flux, gradient);
};