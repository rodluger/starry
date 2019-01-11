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

/**
Compute the derivatives of the limb-darkened flux
with respect to the limb darkening coefficients
from the derivatives with respect to the Agol
Green's coefficients. Single wavelength map specialization.

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
template<typename U=S, typename T1, typename T2>
inline IsSingleWavelength<U, void> computeDfDu (
    MatrixBase<T1> const & flux,
    MatrixBase<T2> const & du,
    const UCoeffType & norm
) {
    if (likely(lmax > 0)) {
#ifdef STARRY_KEEP_DFDU_AS_DFDG
        MBCAST(du, T2) = L.s.transpose();
        MBCAST(du, T2)(0) -= pi<Scalar>() * flux(0);
        MBCAST(du, T2)(1) -= (2.0 / 3.0) * pi<Scalar>() * flux(0);
        MBCAST(du, T2) = du * norm(0);
#else
        Vector<Scalar> dFdAgolG = L.s.transpose();
        dFdAgolG(0) -= pi<Scalar>() * flux(0);
        dFdAgolG(1) -= (2.0 / 3.0) * pi<Scalar>() * flux(0);
        MBCAST(du, T2) = cache.dAgolGdu * dFdAgolG * norm(0);
#endif
    }
}

/**
Compute the derivatives of the limb-darkened flux
with respect to the limb darkening coefficients
from the derivatives with respect to the Agol
Green's coefficients. Spectral map specialization.

See note above.

*/
template<typename U=S, typename T1, typename T2>
inline IsSpectral<U, void> computeDfDu (
    MatrixBase<T1> const & flux,
    MatrixBase<T2> const & du,
    const UCoeffType & norm
) {
    if (likely(lmax > 0)) {
        Scalar twothirdspi = (2.0 / 3.0) * pi<Scalar>();
#ifdef STARRY_KEEP_DFDU_AS_DFDG
        Vector<Scalar> dFdAgolG = L.s.transpose();
        for (int n = 0; n < ncoly; ++n) {
            dFdAgolG(0) = L.s(0) - pi<Scalar>() * flux(n);
            dFdAgolG(1) = L.s(1) - twothirdspi * flux(n);
            MBCAST(du, T2).col(n) = dFdAgolG * norm(n);
        }
#else
        Vector<Scalar> dFdAgolG = L.s.transpose();
        for (int n = 0; n < ncoly; ++n) {
            dFdAgolG(0) = L.s(0) - pi<Scalar>() * flux(n);
            dFdAgolG(1) = L.s(1) - twothirdspi * flux(n);
            MBCAST(du, T2).col(n) = 
                cache.dAgolGdu.block(n * lmax, 0, lmax, lmax + 1) * 
                dFdAgolG * norm(n);
        }
#endif
    }
}