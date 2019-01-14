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
inline IsSingleWavelength<U, void> computeDfDuLDOccultation (
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
inline IsSpectral<U, void> computeDfDuLDOccultation (
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

/**
The derivative of the flux with respect to time
outside of an occultation for a spherical harmonic map. 
Static specialization.

*/
template<typename U=S, typename T1>
inline IsStatic<U, void> computeDfDtYlmNoOccultation (
    MatrixBase<T1> const & dt,
    const YCoeffType & flux0
){
}

/**
The derivative of the flux with respect to time
outside of an occultation for a spherical harmonic map. 
Temporal specialization.

*/
template<typename U=S, typename T1>
inline IsTemporal<U, void> computeDfDtYlmNoOccultation (
    MatrixBase<T1> const & dt,
    const YCoeffType & flux0
){
    MBCAST(dt, T1) = flux0.segment(1, ncoly - 1) * taylor.segment(0, ncoly - 1);
}

/**
The derivative of the flux with respect to time
for a limb-darkened map outside of an occultation. 
Static specialization.

*/
template<typename U=S, typename T1>
inline IsStatic<U, void> computeDfDtLDNoOccultation (
    MatrixBase<T1> const & dt
){
}

/**
The derivative of the flux with respect to time
for a limb-darkened map outside of an occultation. 
Temporal specialization.

*/
template<typename U=S, typename T1>
inline IsTemporal<U, void> computeDfDtLDNoOccultation (
    MatrixBase<T1> const & dt
){
    MBCAST(dt, T1) = y.block(0, 1, 1, ncoly - 1) * taylor.segment(0, ncoly - 1);
}

/**
The derivative of the flux with respect to time
for a limb-darkened map during an occultation. 
Static specialization.

*/
template<typename U=S, typename T1>
inline IsStatic<U, void> computeDfDtLDOccultation (
    MatrixBase<T1> const & dt,
    const UCoeffType & flux0
){
}

/**
The derivative of the flux with respect to time
for a limb-darkened map during an occultation. 
Temporal specialization.

*/
template<typename U=S, typename T1>
inline IsTemporal<U, void> computeDfDtLDOccultation (
    MatrixBase<T1> const & dt,
    const UCoeffType & flux0
){
    UCoeffType norm_deriv = y.block(0, 1, 1, ncoly - 1) * 
                            taylor.segment(0, ncoly - 1);
    MBCAST(dt, T1) = flux0.cwiseProduct(norm_deriv);
}

/**
The derivative of the limb-darkened flux with respect
to the spherical harmonic coefficients outside of an
occultation. Static case.

NOTE: We only compute the derivative with respect to
the Y_{0,0} coefficient. The other Ylm derivs are not 
necessarily zero, but we explicitly don't compute them 
for purely limb-darkened maps to ensure computational 
efficiency. See the docs for more information.

*/
template<typename U=S, typename T1>
inline IsStatic<U, void> computeDfDyLDNoOccultation (
    MatrixBase<T1> const & dy
){
    MBCAST(dy, T1).setZero();
    MBCAST(dy, T1).row(0).setOnes();
}

/**
The derivative of the limb-darkened flux with respect
to the spherical harmonic coefficients outside of an
occultation. Temporal case.

NOTE: We only compute the derivative with respect to
the Y_{0,0} coefficient. The other Ylm derivs are not 
necessarily zero, but we explicitly don't compute them 
for purely limb-darkened maps to ensure computational 
efficiency. See the docs for more information.

*/
template<typename U=S, typename T1>
inline IsTemporal<U, void> computeDfDyLDNoOccultation (
    MatrixBase<T1> const & dy
){
    MBCAST(dy, T1).setZero();
    MBCAST(dy, T1).row(0) = taylor.transpose();
}

/**
The derivative of the limb-darkened flux with respect
to the spherical harmonic coefficients during an
occultation. Static case.

NOTE: We only compute the derivative with respect to
the Y_{0,0} coefficient. The other Ylm derivs are not 
necessarily zero, but we explicitly don't compute them 
for purely limb-darkened maps to ensure computational 
efficiency. See the docs for more information.

*/
template<typename U=S, typename T1>
inline IsStatic<U, void> computeDfDyLDOccultation (
    MatrixBase<T1> const & dy, 
    const UCoeffType & flux0
){
    MBCAST(dy, T1).setZero();
    MBCAST(dy, T1).row(0) = flux0;
}

/**
The derivative of the limb-darkened flux with respect
to the spherical harmonic coefficients during an occultation. 
Temporal case.

NOTE: We only compute the derivative with respect to
the Y_{0,0} coefficient. The other Ylm derivs are not 
necessarily zero, but we explicitly don't compute them 
for purely limb-darkened maps to ensure computational 
efficiency. See the docs for more information.

*/
template<typename U=S, typename T1>
inline IsTemporal<U, void> computeDfDyLDOccultation (
    MatrixBase<T1> const & dy, 
    const UCoeffType & flux0
){
    MBCAST(dy, T1).setZero();
    MBCAST(dy, T1).row(0) = flux0(0) * taylor.transpose();
}

/**
The derivative of the flux with respect to the spherical harmonic coefficients
when there is no occultation. Default case.

*/
template<typename U=S, typename T1>
inline IsDefault<U, void> computeDfDyYlmNoOccultation (
    MatrixBase<T1> const & dy,
    const Scalar& theta
){
    if (theta == 0) {
        MBCAST(dy, T1) = B.rTA1.transpose();
    } else {
        for (int l = 0; l < lmax + 1; ++l)
            MBCAST(dy, T1).segment(l * l, 2 * l + 1) = 
                (B.rTA1.segment(l * l, 2 * l + 1) * W.R[l]).transpose();
    }
}

/**
The derivative of the flux with respect to the spherical harmonic coefficients
when there is no occultation. Spectral case.

*/
template<typename U=S, typename T1>
inline IsSpectral<U, void> computeDfDyYlmNoOccultation (
    MatrixBase<T1> const & dy,
    const Scalar& theta
){
    if (theta == 0) {
        MBCAST(dy, T1) = B.rTA1.transpose().replicate(1, ncoly);
    } else {
        for (int l = 0; l < lmax + 1; ++l)
            MBCAST(dy, T1).block(l * l, 0, 2 * l + 1, ncoly) = 
                (B.rTA1.segment(l * l, 2 * l + 1) * W.R[l])
                .transpose().replicate(1, ncoly);
    }
}

/**
The derivative of the flux with respect to the spherical harmonic coefficients
when there is no occultation. Temporal case.

*/
template<typename U=S, typename T1>
inline IsTemporal<U, void> computeDfDyYlmNoOccultation (
    MatrixBase<T1> const & dy,
    const Scalar& theta
){
    if (theta == 0) {
        MBCAST(dy, T1) = (taylor * B.rTA1).transpose();
    } else {
        for (int l = 0; l < lmax + 1; ++l)
            MBCAST(dy, T1).block(l * l, 0, 2 * l + 1, ncoly) = 
                (taylor * (B.rTA1.segment(l * l, 2 * l + 1) * W.R[l]))
                .transpose();
    }
}

/**
The derivative of the flux with respect to the spherical harmonic 
coefficients for a limb-darkened spherical harmonic map.
Default case.

*/
template<typename U=S, typename T1>
inline IsDefault<U, void> computeDfDyYlmLDNoOccultation (
    MatrixBase<T1> const & dy
) {
    MBCAST(dy, T1) = (B.rT * cache.dLDdy[0]).transpose();
}

/**
The derivative of the flux with respect to the spherical harmonic 
coefficients for a limb-darkened spherical harmonic map.
Spectral case.

*/
template<typename U=S, typename T1>
inline IsSpectral<U, void> computeDfDyYlmLDNoOccultation (
    MatrixBase<T1> const & dy
) {
    for (int i = 0; i < ncoly; ++i)
        MBCAST(dy, T1).col(i) = B.rT * cache.dLDdy[i];
}

/**
The derivative of the flux with respect to the spherical harmonic 
coefficients for a limb-darkened spherical harmonic map.
Temporal case.

*/
template<typename U=S, typename T1>
inline IsTemporal<U, void> computeDfDyYlmLDNoOccultation (
    MatrixBase<T1> const & dy
){
    // TODO
    throw errors::ToDoError("TODO: Temporal dfdy for Ylm + LD.");
}

/**
The derivative of the flux with respect to the limb darkening
coefficients for a limb-darkened spherical harmonic map.
Default case.

*/
template<typename U=S, typename T1>
inline IsDefault<U, void> computeDfDuYlmLDNoOccultation (
    MatrixBase<T1> const & du
) {
    MBCAST(du, T1) = (B.rT * cache.dLDdu[0]).segment(1, lmax).transpose();
}

/**
The derivative of the flux with respect to the limb darkening
coefficients for a limb-darkened spherical harmonic map.
Spectral case.

*/
template<typename U=S, typename T1>
inline IsSpectral<U, void> computeDfDuYlmLDNoOccultation (
    MatrixBase<T1> const & du
) {
    for (int i = 0; i < ncolu; ++i)
        MBCAST(du, T1).col(i) = (B.rT * cache.dLDdu[i]).segment(1, lmax);
}

/**
The derivative of the flux with respect to the limb darkening
coefficients for a limb-darkened spherical harmonic map.
Temporal case.

*/
template<typename U=S, typename T1>
inline IsTemporal<U, void> computeDfDuYlmLDNoOccultation (
    MatrixBase<T1> const & du
){
    MBCAST(du, T1) = (B.rT * cache.dLDdu[0]).segment(1, lmax).transpose();
}