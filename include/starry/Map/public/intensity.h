/**
Compute the linear Ylm model. Basic / Spectral specialization. Emitted light.

*/
template <typename U=S>
inline EnableIf<!U::Reflected && !U::Temporal, void> computeLinearIntensityModel (
    const Vector<Scalar>& theta, 
    const RowMatrix<Scalar>& x, 
    const RowMatrix<Scalar>& y,
    RowMatrix<Scalar>& X
) {
    RowMatrix<Scalar> source; // dummy
    computeLinearIntensityModelInternal(theta, x, y, source, X);
}

/**
Compute the linear Ylm model. Temporal specialization. Emitted light.

*/
template <typename U=S>
inline EnableIf<!U::Reflected && U::Temporal, void> computeLinearIntensityModel (
    const Vector<Scalar>& t,
    const Vector<Scalar>& theta, 
    const RowMatrix<Scalar>& x, 
    const RowMatrix<Scalar>& y,
    RowMatrix<Scalar>& X
) {
    RowMatrix<Scalar> source; // dummy
    computeTaylor(t);
    computeLinearIntensityModelInternal(theta, x, y, source, X);
}

/**
Compute the linear Ylm model. Basic / Spectral specialization. Reflected light.

*/
template <typename U=S>
inline EnableIf<U::Reflected && !U::Temporal, void> computeLinearIntensityModel (
    const Vector<Scalar>& theta, 
    const RowMatrix<Scalar>& x, 
    const RowMatrix<Scalar>& y,
    const RowMatrix<Scalar>& source,
    RowMatrix<Scalar>& X
) {
    computeLinearIntensityModelInternal(theta, x, y, source.rowwise().normalized(), X);
}

/**
Compute the linear Ylm model. Temporal specialization. Reflected light.

*/
template <typename U=S>
inline EnableIf<U::Reflected && U::Temporal, void> computeLinearIntensityModel (
    const Vector<Scalar>& t,
    const Vector<Scalar>& theta, 
    const RowMatrix<Scalar>& x, 
    const RowMatrix<Scalar>& y,
    const RowMatrix<Scalar>& source,
    RowMatrix<Scalar>& X
) {
    computeTaylor(t);
    computeLinearIntensityModelInternal(theta, x, y, source.rowwise().normalized(), X);
}