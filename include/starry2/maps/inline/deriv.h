/**
The derivative of the flux with respect to time
outside of an occultation. Static specialization.

*/
template<typename U=S, typename T1>
inline IsStatic<U, void> computeDfDtNoOccultation (
    MatrixBase<T1> const & dt,
    const YCoeffType & flux0
){
}

/**
The derivative of the flux with respect to time
for a limb-darkened map outside of an occultation. 
Static specialization.

*/
template<typename U=S, typename T1>
inline IsStatic<U, void> computeDfDtLimbDarkenedNoOccultation (
    MatrixBase<T1> const & dt
){
}

/**
The derivative of the flux with respect to time
for a limb-darkened map during an occultation. 
Static specialization.

*/
template<typename U=S, typename T1>
inline IsStatic<U, void> computeDfDtLimbDarkenedOccultation (
    MatrixBase<T1> const & dt,
    const UCoeffType & flux0
){
}

/**
The derivative of the flux with respect to time
outside of an occultation. Temporal specialization.

*/
template<typename U=S, typename T1>
inline IsTemporal<U, void> computeDfDtNoOccultation (
    MatrixBase<T1> const & dt,
    const YCoeffType & flux0
){
    MBCAST(dt, T1) = flux0.segment(1, ncol - 1) * taylor.segment(0, ncol - 1);
}

/**
The derivative of the flux with respect to time
for a limb-darkened map outside of an occultation. 
Temporal specialization.

*/
template<typename U=S, typename T1>
inline IsTemporal<U, void> computeDfDtLimbDarkenedNoOccultation (
    MatrixBase<T1> const & dt
){
    MBCAST(dt, T1) = y.block(0, 1, 1, ncol - 1) * taylor.segment(0, ncol - 1);
}

/**
The derivative of the flux with respect to time
for a limb-darkened map during an occultation. 
Temporal specialization.

*/
template<typename U=S, typename T1>
inline IsTemporal<U, void> computeDfDtLimbDarkenedOccultation (
    MatrixBase<T1> const & dt,
    const UCoeffType & flux0
){
    UCoeffType norm_deriv = y.block(0, 1, 1, ncol - 1) * 
                            taylor.segment(0, ncol - 1);
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
inline IsStatic<U, void> computeDfDyLimbDarkenedNoOccultation (
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
inline IsTemporal<U, void> computeDfDyLimbDarkenedNoOccultation (
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
inline IsStatic<U, void> computeDfDyLimbDarkenedOccultation (
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
inline IsTemporal<U, void> computeDfDyLimbDarkenedOccultation (
    MatrixBase<T1> const & dy, 
    const UCoeffType & flux0
){
    MBCAST(dy, T1).setZero();
    MBCAST(dy, T1).row(0) = flux0(0) * taylor.transpose();
}

/**
The derivative of the flux with respect
to the spherical harmonic coefficients
when there is no occultation.
Default case.

*/
template<typename U=S, typename T1>
inline IsDefault<U, void> computeDfDyNoOccultation (
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
The derivative of the flux with respect
to the spherical harmonic coefficients. 
Spectral case.

*/
template<typename U=S, typename T1>
inline IsSpectral<U, void> computeDfDyNoOccultation (
    MatrixBase<T1> const & dy,
    const Scalar& theta
){
    if (theta == 0) {
        MBCAST(dy, T1) = B.rTA1.transpose().replicate(1, ncol);
    } else {
        for (int l = 0; l < lmax + 1; ++l)
            MBCAST(dy, T1).block(l * l, 0, 2 * l + 1, ncol) = 
                (B.rTA1.segment(l * l, 2 * l + 1) * W.R[l])
                .transpose().replicate(1, ncol);
    }
}

/**
The derivative of the flux with respect
to the spherical harmonic coefficients. 
Temporal case.

*/
template<typename U=S, typename T1>
inline IsTemporal<U, void> computeDfDyNoOccultation (
    MatrixBase<T1> const & dy,
    const Scalar& theta
){
    if (theta == 0) {
        MBCAST(dy, T1) = (taylor * B.rTA1).transpose();
    } else {
        for (int l = 0; l < lmax + 1; ++l)
            MBCAST(dy, T1).block(l * l, 0, 2 * l + 1, ncol) = 
                (taylor * (B.rTA1.segment(l * l, 2 * l + 1) * W.R[l]))
                .transpose();
    }
}