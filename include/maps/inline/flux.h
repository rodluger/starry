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
}

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
}

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
}

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
}

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
}

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
}

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
}

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
}

/**
Compute the derivatives of the limb-darkened flux
with respect to the limb darkening coefficients
from the derivatives with respect to the Agol
Green's coefficients. Default map specialization.

NOTE: The largest bottleneck in computing limb 
darkening derivatives is applying the change of basis 
matrix from g -> u. The flag `STARRY_KEEP_DFDU_AS_DFDG`
allows the user to request derivatives with respect to 
the Green's coefficients `g`, skipping the costly matrix 
math. The change of basis can then be applied directly 
to the gradient of the *likelihood* when doing inference, 
saving a *lot* of compute time. See Agol et al. (2019)
for more info.

*/
template<typename U=S>
inline IsDefault<U, void> computeDfDu (
    Ref<FluxType> flux, 
    Ref<GradType> gradient
) {
#ifdef STARRY_KEEP_DFDU_AS_DFDG
    gradient.segment(idx.u, lmax) = L.s.transpose();
    gradient(idx.u) -= pi<Scalar>() * flux(0);
    if (lmax > 0)
        gradient(idx.u + 1) -= (2.0 / 3.0) * pi<Scalar>() * flux(0);
#else
    Vector<Scalar> dFdc = L.s.transpose();
    dFdc(0) -= pi<Scalar>() * flux(0);
    if (lmax > 0)
        dFdc(1) -= (2.0 / 3.0) * pi<Scalar>() * flux(0);
    gradient.segment(idx.u, lmax) = cache.dcdu * dFdc;
#endif
}

/**
Compute the derivatives of the limb-darkened flux
with respect to the limb darkening coefficients
from the derivatives with respect to the Agol
Green's coefficients. Spectral map specialization.

See note above.

*/
template<typename U=S>
inline IsSpectral<U, void> computeDfDu (
    Ref<FluxType> flux, 
    Ref<GradType> gradient
) {
    Scalar twothirdspi = (2.0 / 3.0) * pi<Scalar>();
#ifdef STARRY_KEEP_DFDU_AS_DFDG
    Vector<Scalar> dFdc = L.s.transpose();
    for (int n = 0; n < ncol; ++n) {
        dFdc(0) = L.s(0) - pi<Scalar>() * flux(n);
        if (lmax > 0)
            dFdc(1) = L.s(1) - twothirdspi * flux(n);
        gradient.block(idx.u, n, lmax, 1) = dFdc;
    }
#else
    Vector<Scalar> dFdc = L.s.transpose();
    for (int n = 0; n < ncol; ++n) {
        dFdc(0) = L.s(0) - pi<Scalar>() * flux(n);
        if (lmax > 0)
            dFdc(1) = L.s(1) - twothirdspi * flux(n);
        gradient.block(idx.u, n, lmax, 1) = 
            cache.dcdu.block(n * lmax, 0, lmax, lmax + 1) * dFdc;
    }
#endif
}

/**
Compute the derivatives of the limb-darkened flux
with respect to the limb darkening coefficients
from the derivatives with respect to the Agol
Green's coefficients. Temporal map specialization.

*/
template<typename U=S>
inline IsTemporal<U, void> computeDfDu (
    Ref<FluxType> flux, 
    Ref<GradType> gradient
) {
    // TODO!
    throw errors::NotImplementedError("Temporal limb darkening not yet working.");
}