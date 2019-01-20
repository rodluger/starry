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
inline IsDefaultOrTemporal<U, void> computeDfDuLDOccultation (
    MatrixBase<T1> const & flux,
    MatrixBase<T2> const & Du,
    const UCoeffType & norm
) {
    if (likely(lmax > 0)) {
#ifdef STARRY_KEEP_DFDU_AS_DFDG
        MBCAST(Du, T2) = L.sT.transpose();
        MBCAST(Du, T2)(0) -= pi<Scalar>() * flux(0);
        MBCAST(Du, T2)(1) -= (2.0 / 3.0) * pi<Scalar>() * flux(0);
        MBCAST(Du, T2) = Du * norm(0);
#else
        Vector<Scalar> DfDg = L.sT.transpose();
        DfDg(0) -= pi<Scalar>() * flux(0);
        DfDg(1) -= (2.0 / 3.0) * pi<Scalar>() * flux(0);
        MBCAST(Du, T2) = cache.DgDu * DfDg * norm(0);
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
    MatrixBase<T2> const & Du,
    const UCoeffType & norm
) {
    if (likely(lmax > 0)) {
        Scalar twothirdspi = (2.0 / 3.0) * pi<Scalar>();
#ifdef STARRY_KEEP_DFDU_AS_DFDG
        Vector<Scalar> DfDg = L.sT.transpose();
        for (int n = 0; n < ncoly; ++n) {
            DfDg(0) = L.sT(0) - pi<Scalar>() * flux(n);
            DfDg(1) = L.sT(1) - twothirdspi * flux(n);
            MBCAST(Du, T2).col(n) = DfDg * norm(n);
        }
#else
        Vector<Scalar> DfDg = L.sT.transpose();
        for (int n = 0; n < ncoly; ++n) {
            DfDg(0) = L.sT(0) - pi<Scalar>() * flux(n);
            DfDg(1) = L.sT(1) - twothirdspi * flux(n);
            MBCAST(Du, T2).col(n) = 
                cache.DgDu.block(n * lmax, 0, lmax, lmax + 1) * 
                DfDg * norm(n);
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
inline IsDefaultOrSpectral<U, void> computeDfDtYlmNoOccultation (
    MatrixBase<T1> const & Dt
){
}

/**
The derivative of the flux with respect to time
outside of an occultation for a spherical harmonic map. 
Temporal specialization.

*/
template<typename U=S, typename T1>
inline IsTemporal<U, void> computeDfDtYlmNoOccultation (
    MatrixBase<T1> const & Dt
){
    MBCAST(Dt, T1) = (B.rTA1 * cache.RY).block(0, 1, 1, ncoly - 1) 
                     * taylor.segment(0, ncoly - 1);
}

/**
The derivative of the flux with respect to time
for a limb-darkened map outside of an occultation. 
Static specialization.

*/
template<typename U=S, typename T1>
inline IsDefaultOrSpectral<U, void> computeDfDtLDNoOccultation (
    MatrixBase<T1> const & Dt
){
}

/**
The derivative of the flux with respect to time
for a limb-darkened map outside of an occultation. 
Temporal specialization.

*/
template<typename U=S, typename T1>
inline IsTemporal<U, void> computeDfDtLDNoOccultation (
    MatrixBase<T1> const & Dt
){
    MBCAST(Dt, T1) = y.block(0, 1, 1, ncoly - 1) * taylor.segment(0, ncoly - 1);
}

/**
The derivative of the flux with respect to time
for a limb-darkened map during an occultation. 
Static specialization.

*/
template<typename U=S, typename T1>
inline IsDefaultOrSpectral<U, void> computeDfDtLDOccultation (
    MatrixBase<T1> const & Dt,
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
    MatrixBase<T1> const & Dt,
    const UCoeffType & flux0
){
    UCoeffType norm_deriv = y.block(0, 1, 1, ncoly - 1) * 
                            taylor.segment(0, ncoly - 1);
    MBCAST(Dt, T1) = flux0.cwiseProduct(norm_deriv);
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
inline IsDefaultOrSpectral<U, void> computeDfDyLDNoOccultation (
    MatrixBase<T1> const & Dy
){
    MBCAST(Dy, T1).setZero();
    MBCAST(Dy, T1).row(0).setOnes();
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
    MatrixBase<T1> const & Dy
){
    MBCAST(Dy, T1).setZero();
    MBCAST(Dy, T1).row(0) = taylor.transpose();
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
inline IsDefaultOrSpectral<U, void> computeDfDyLDOccultation (
    MatrixBase<T1> const & Dy, 
    const UCoeffType & flux0
){
    MBCAST(Dy, T1).setZero();
    MBCAST(Dy, T1).row(0) = flux0;
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
    MatrixBase<T1> const & Dy, 
    const UCoeffType & flux0
){
    MBCAST(Dy, T1).setZero();
    MBCAST(Dy, T1).row(0) = flux0(0) * taylor.transpose();
}

/**
The derivative of the flux with respect to the spherical harmonic coefficients
when there is no occultation. Default case.

*/
template<typename U=S, typename T1>
inline IsDefault<U, void> computeDfDyYlmNoOccultation (
    MatrixBase<T1> const & Dy,
    const Scalar& theta
){
    if (theta != 0) {
        for (int l = 0; l < lmax + 1; ++l)
            MBCAST(Dy, T1).segment(l * l, 2 * l + 1) = 
                (B.rTA1.segment(l * l, 2 * l + 1) * W.R[l]).transpose();
    } else {
        MBCAST(Dy, T1) = B.rTA1.transpose();
    }
}

/**
The derivative of the flux with respect to the spherical harmonic coefficients
when there is no occultation. Spectral case.

*/
template<typename U=S, typename T1>
inline IsSpectral<U, void> computeDfDyYlmNoOccultation (
    MatrixBase<T1> const & Dy,
    const Scalar& theta
){
    if (theta != 0) {
        for (int l = 0; l < lmax + 1; ++l)
            MBCAST(Dy, T1).block(l * l, 0, 2 * l + 1, ncoly) = 
                (B.rTA1.segment(l * l, 2 * l + 1) * W.R[l])
                .transpose().replicate(1, ncoly);
    } else {
        MBCAST(Dy, T1) = B.rTA1.transpose().replicate(1, ncoly);
    }
}

/**
The derivative of the flux with respect to the spherical harmonic coefficients
when there is no occultation. Temporal case.

*/
template<typename U=S, typename T1>
inline IsTemporal<U, void> computeDfDyYlmNoOccultation (
    MatrixBase<T1> const & Dy,
    const Scalar& theta
){
    if (theta != 0) {
        for (int l = 0; l < lmax + 1; ++l)
            MBCAST(Dy, T1).block(l * l, 0, 2 * l + 1, ncoly) = 
                (taylor * (B.rTA1.segment(l * l, 2 * l + 1) * W.R[l]))
                .transpose();
    } else {
        MBCAST(Dy, T1) = (taylor * B.rTA1).transpose();
    }
}

/**
The derivative of the flux with respect to the spherical harmonic 
coefficients for a limb-darkened spherical harmonic map.
Default case.

*/
template<typename U=S, typename T1>
inline IsDefault<U, void> computeDfDyYlmLDNoOccultation (
    MatrixBase<T1> const & Dy
) {
    MBCAST(Dy, T1) = cache.vTDpupyDy;
}

/**
The derivative of the flux with respect to the spherical harmonic 
coefficients for a limb-darkened spherical harmonic map.
Spectral case.

*/
template<typename U=S, typename T1>
inline IsSpectral<U, void> computeDfDyYlmLDNoOccultation (
    MatrixBase<T1> const & Dy
) {
    MBCAST(Dy, T1) = cache.vTDpupyDy;
}

/**
The derivative of the flux with respect to the spherical harmonic 
coefficients for a limb-darkened spherical harmonic map.
Temporal case.

*/
template<typename U=S, typename T1>
inline IsTemporal<U, void> computeDfDyYlmLDNoOccultation (
    MatrixBase<T1> const & Dy
){
    MBCAST(Dy, T1) = cache.vTDpupyDy.array().rowwise() * 
                     taylor.transpose().array();
}

/**
The derivative of the flux with respect to the limb darkening
coefficients for a limb-darkened spherical harmonic map.
Default case.

*/
template<typename U=S, typename T1>
inline IsDefault<U, void> computeDfDuYlmLDNoOccultation (
    MatrixBase<T1> const & Du
) {
    MBCAST(Du, T1) = cache.vTDpupyDu.block(1, 0, lmax, 1);
}

/**
The derivative of the flux with respect to the limb darkening
coefficients for a limb-darkened spherical harmonic map.
Spectral case.

*/
template<typename U=S, typename T1>
inline IsSpectral<U, void> computeDfDuYlmLDNoOccultation (
    MatrixBase<T1> const & Du
) {
    MBCAST(Du, T1) = cache.vTDpupyDu.block(1, 0, lmax, ncolu);
}

/**
The derivative of the flux with respect to the limb darkening
coefficients for a limb-darkened spherical harmonic map.
Temporal case.

*/
template<typename U=S, typename T1>
inline IsTemporal<U, void> computeDfDuYlmLDNoOccultation (
    MatrixBase<T1> const & Du
){
    MBCAST(Du, T1) = cache.vTDpupyDu.block(1, 0, lmax, 1);
}

/**
The derivative of the flux with respect to theta
for a limb-darkened spherical harmonic map.
Single-wavelength case.

*/
template<typename U=S, typename T1>
inline IsDefaultOrTemporal<U, void> computeDfDthetaYlmLDNoOccultation (
    MatrixBase<T1> const & Dtheta
){
    MBCAST(Dtheta, T1) = cache.vTDpupyDpy * (B.A1 * cache.DRDthetay);
    MBCAST(Dtheta, T1) *= radian;
}

/**
The derivative of the flux with respect to theta
for a limb-darkened spherical harmonic map.
Spectral case.

*/
template<typename U=S, typename T1>
inline IsSpectral<U, void> computeDfDthetaYlmLDNoOccultation (
    MatrixBase<T1> const & Dtheta
){
    // This is a little nasty because `DpupyDpy` is a tensor
    // and we're doing a sneaky contraction
    MBCAST(Dtheta, T1) = (cache.vTDpupyDpy.transpose()
                            .cwiseProduct(B.A1 * cache.DRDthetay)
                         ).colwise().sum();
    MBCAST(Dtheta, T1) *= radian;
}

/**
The derivative of the flux with respect to time
outside of an occultation for a limb-darkened
spherical harmonic map. 
Static specialization.

*/
template<typename U=S, typename T1>
inline IsDefaultOrSpectral<U, void> computeDfDtYlmLDNoOccultation (
    MatrixBase<T1> const & Dt
){
}

/**
The derivative of the flux with respect to time
outside of an occultation for a limb-darkened 
spherical harmonic map. 
Temporal specialization.

*/
template<typename U=S, typename T1>
inline IsTemporal<U, void> computeDfDtYlmLDNoOccultation (
    MatrixBase<T1> const & Dt
){
    MBCAST(Dt, T1).setZero();
    auto A1Ry = B.A1 * cache.RY; // TODO CACHE THIS
    for (int i = 0; i < ncoly - 1; ++i) {
        limbDarken(A1Ry.col(i + 1), cache.pupy);
        MBCAST(Dt, T1) += OneByOne<Scalar>(B.rT * cache.pupy) * taylor(i);
    }
}

/**
The derivative of the flux with respect to time
during an occultation for a spherical harmonic map. 
Static specialization.

*/
template<typename U=S, typename T1>
inline IsDefaultOrSpectral<U, void> computeDfDtYlmOccultation (
    MatrixBase<T1> const & Dt
){
}

/**
The derivative of the flux with respect to time
during an occultation for a spherical harmonic map. 
Temporal specialization.

*/
template<typename U=S, typename T1>
inline IsTemporal<U, void> computeDfDtYlmOccultation (
    MatrixBase<T1> const & Dt
){
    MBCAST(Dt, T1) = (cache.sTAR * cache.RY)
                     .block(0, 1, 1, ncoly - 1) 
                     * taylor.segment(0, ncoly - 1);
}

/**
The derivative of the flux with respect to the spherical harmonic 
coefficients for a spherical harmonic map during an occultation.
Default case.

*/
template<typename U=S, typename T1>
inline IsDefault<U, void> computeDfDyYlmOccultation (
    MatrixBase<T1> const & Dy,
    const Scalar& theta
) {
    if (theta != 0) {
        for (int l = 0; l < lmax + 1; ++l)
            MBCAST(Dy, T1).segment(l * l, 2 * l + 1) = 
                (cache.sTAR.segment(l * l, 2 * l + 1) * W.R[l]);
    } else {
        MBCAST(Dy, T1) = cache.sTAR;
    }
}

/**
The derivative of the flux with respect to the spherical harmonic 
coefficients for a spherical harmonic map during an occultation.
Spectral case.

*/
template<typename U=S, typename T1>
inline IsSpectral<U, void> computeDfDyYlmOccultation (
    MatrixBase<T1> const & Dy,
    const Scalar& theta
) {
    if (theta != 0) {
        for (int l = 0; l < lmax + 1; ++l)
            MBCAST(Dy, T1).block(l * l, 0, 2 * l + 1, ncoly) = 
                (cache.sTAR.segment(l * l, 2 * l + 1) * W.R[l])
                .transpose().replicate(1, ncoly);
    } else {
        MBCAST(Dy, T1) = cache.sTAR.transpose().replicate(1, ncoly);
    }
}

/**
The derivative of the flux with respect to the spherical harmonic 
coefficients for a spherical harmonic map during an occultation.
Temporal case.

*/
template<typename U=S, typename T1>
inline IsTemporal<U, void> computeDfDyYlmOccultation (
    MatrixBase<T1> const & Dy,
    const Scalar& theta
){
    if (theta != 0) {
        for (int l = 0; l < lmax + 1; ++l)
            MBCAST(Dy, T1).block(l * l, 0, 2 * l + 1, ncoly) = 
                (taylor * (cache.sTAR.segment(l * l, 2 * l + 1) * W.R[l]))
                .transpose();
    } else {
        MBCAST(Dy, T1) = (taylor * cache.sTAR).transpose();
    }
}

/**
The derivative of the flux with respect to theta
for a limb-darkened spherical harmonic map.
Single-wavelength case.

*/
template<typename U=S, typename T1>
inline IsDefaultOrTemporal<U, void> computeDfDthetaYlmLDOccultation (
    MatrixBase<T1> const & Dtheta
){
    W.leftMultiplyRz(cache.vTDpupyDpyA1, cache.vTDpupyDpyA1R);
    MBCAST(Dtheta, T1) = cache.vTDpupyDpyA1R * cache.DRDthetay;
    MBCAST(Dtheta, T1) *= radian;
}

/**
The derivative of the flux with respect to theta
for a limb-darkened spherical harmonic map.
Spectral case.

*/
template<typename U=S, typename T1>
inline IsSpectral<U, void> computeDfDthetaYlmLDOccultation (
    MatrixBase<T1> const & Dtheta
){
    // This is a little nasty because `DpupyDpy` is a tensor
    // and we're doing a sneaky contraction
    W.leftMultiplyRz(cache.vTDpupyDpyA1, cache.vTDpupyDpyA1R);
    MBCAST(Dtheta, T1) = (cache.vTDpupyDpyA1R.transpose()
                            .cwiseProduct(cache.DRDthetay)
                         ).colwise().sum();
    MBCAST(Dtheta, T1) *= radian;
}

/**
The derivative of the flux with respect to the limb darkening
coefficients for a limb-darkened spherical harmonic map.
Default case.

*/
template<typename U=S, typename T1>
inline IsDefault<U, void> computeDfDuYlmLDOccultation (
    MatrixBase<T1> const & Du
) {
    MBCAST(Du, T1) = cache.vTDpupyDu.block(1, 0, lmax, 1);
}

/**
The derivative of the flux with respect to the limb darkening
coefficients for a limb-darkened spherical harmonic map.
Spectral case.

*/
template<typename U=S, typename T1>
inline IsSpectral<U, void> computeDfDuYlmLDOccultation (
    MatrixBase<T1> const & Du
) {
    MBCAST(Du, T1) = cache.vTDpupyDu.block(1, 0, lmax, ncolu);
}

/**
The derivative of the flux with respect to the limb darkening
coefficients for a limb-darkened spherical harmonic map.
Temporal case.

*/
template<typename U=S, typename T1>
inline IsTemporal<U, void> computeDfDuYlmLDOccultation (
    MatrixBase<T1> const & Du
){
    MBCAST(Du, T1) = cache.vTDpupyDu.block(1, 0, lmax, 1);
}

/**
The derivative of the flux with respect to the spherical harmonic 
coefficients for a limb-darkened spherical harmonic map.
Default case.

*/
template<typename U=S, typename T1>
inline IsDefault<U, void> computeDfDyYlmLDOccultation (
    MatrixBase<T1> const & Dy
) {
    MBCAST(Dy, T1) = cache.vTDpupyDy;
}

/**
The derivative of the flux with respect to the spherical harmonic 
coefficients for a limb-darkened spherical harmonic map.
Spectral case.

*/
template<typename U=S, typename T1>
inline IsSpectral<U, void> computeDfDyYlmLDOccultation (
    MatrixBase<T1> const & Dy
) {
    MBCAST(Dy, T1) = cache.vTDpupyDy;
}

/**
The derivative of the flux with respect to the spherical harmonic 
coefficients for a limb-darkened spherical harmonic map.
Temporal case.

*/
template<typename U=S, typename T1>
inline IsTemporal<U, void> computeDfDyYlmLDOccultation (
    MatrixBase<T1> const & Dy
){
    MBCAST(Dy, T1) = cache.vTDpupyDy.array().rowwise() * 
                     taylor.transpose().array();
}

/**
The derivative of the flux with respect to time
during an occultation for a limb-darkened
spherical harmonic map. 
Static specialization.

*/
template<typename U=S, typename T1>
inline IsDefaultOrSpectral<U, void> computeDfDtYlmLDOccultation (
    MatrixBase<T1> const & Dt,
    const Scalar& xo_b,
    const Scalar& yo_b
){
}

/**
The derivative of the flux with respect to time
during an occultation for a limb-darkened 
spherical harmonic map. 
Temporal specialization.

*/
template<typename U=S, typename T1>
inline IsTemporal<U, void> computeDfDtYlmLDOccultation (
    MatrixBase<T1> const & Dt,
    const Scalar& xo_b,
    const Scalar& yo_b
){
    // TODO: Speed this up a bit
    MBCAST(Dt, T1).setZero();
    for (int i = 0; i < ncoly - 1; ++i) {
        W.rotateAboutZ(yo_b, xo_b, cache.RY.col(i + 1), cache.RRy);
        cache.A1Ry = B.A1 * cache.RRy;
        limbDarken(cache.A1Ry, cache.pupy);
        MBCAST(Dt, T1) += OneByOne<Scalar>(cache.sTA2 * cache.pupy) * taylor(i);
    }
}