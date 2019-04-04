/**
Compute the linear Ylm model in emitted light. 
Basic / Spectral specialization.

*/
template <typename U=S>
inline EnableIf<!U::Reflected && !U::Temporal, void> computeLinearFluxModel (
    const Vector<Scalar>& theta, 
    const Vector<Scalar>& xo, 
    const Vector<Scalar>& yo, 
    const Vector<Scalar>& zo, 
    const Vector<Scalar>& ro, 
    RowMatrix<Scalar>& X
) {
    computeLinearFluxModelInternal(theta, xo, yo, zo, ro, X);
}

/**
Compute the linear Ylm model in emitted light. Temporal specialization.

*/
template <typename U=S>
inline EnableIf<!U::Reflected && U::Temporal, void> computeLinearFluxModel (
    const Vector<Scalar>& t,
    const Vector<Scalar>& theta, 
    const Vector<Scalar>& xo, 
    const Vector<Scalar>& yo, 
    const Vector<Scalar>& zo, 
    const Vector<Scalar>& ro, 
    RowMatrix<Scalar>& X
) {
    computeTaylor(t);
    computeLinearFluxModelInternal(theta, xo, yo, zo, ro, X);
}

/**
Compute the linear Ylm model in reflected light. 
Basic / Spectral specialization.

*/
template <typename U=S>
inline EnableIf<U::Reflected && !U::Temporal, void> computeLinearFluxModel (
    const Vector<Scalar>& theta, 
    const Vector<Scalar>& xo, 
    const Vector<Scalar>& yo, 
    const Vector<Scalar>& zo, 
    const Vector<Scalar>& ro, 
    const RowMatrix<Scalar>& source,
    RowMatrix<Scalar>& X
) {
    computeLinearFluxModelInternal(theta, xo, yo, zo, 
                                   ro, source.rowwise().normalized(), X);
}

/**
Compute the linear Ylm model in reflected light. Temporal specialization.

*/
template <typename U=S>
inline EnableIf<U::Reflected && U::Temporal, void> computeLinearFluxModel (
    const Vector<Scalar>& t,
    const Vector<Scalar>& theta, 
    const Vector<Scalar>& xo, 
    const Vector<Scalar>& yo, 
    const Vector<Scalar>& zo, 
    const Vector<Scalar>& ro, 
    const RowMatrix<Scalar>& source,
    RowMatrix<Scalar>& X
) {
    computeTaylor(t);
    computeLinearFluxModelInternal(theta, xo, yo, zo, ro, 
                                   source.rowwise().normalized(), X);
}

/**
Compute the linear Ylm model in emitted light and its gradient. 
Basic / Spectral specialization.

*/
template <typename U=S>
inline EnableIf<!U::Reflected && !U::Temporal, void> computeLinearFluxModel (
    const Vector<Scalar>& theta, 
    const Vector<Scalar>& xo, 
    const Vector<Scalar>& yo, 
    const Vector<Scalar>& zo, 
    const Vector<Scalar>& ro, 
    RowMatrix<Scalar>& X,
    RowMatrix<Scalar>& Dtheta,
    RowMatrix<Scalar>& Dxo,
    RowMatrix<Scalar>& Dyo,
    RowMatrix<Scalar>& Dro,
    RowMatrix<Scalar>& Du,
    RowMatrix<Scalar>& Dinc,
    RowMatrix<Scalar>& Dobl
) {
    RowMatrix<Scalar> Dt; // Dummy!
    computeLinearFluxModelInternal(
        theta, xo, yo, zo, ro, X, Dt, Dtheta, Dxo, Dyo, Dro, Du, Dinc, Dobl
    );
}

/**
Compute the linear Ylm model in emitted light and its gradient. 
Temporal specialization.

*/
template <typename U=S>
inline EnableIf<!U::Reflected && U::Temporal, void> computeLinearFluxModel (
    const Vector<Scalar>& t, 
    const Vector<Scalar>& theta, 
    const Vector<Scalar>& xo, 
    const Vector<Scalar>& yo, 
    const Vector<Scalar>& zo, 
    const Vector<Scalar>& ro, 
    RowMatrix<Scalar>& X,
    RowMatrix<Scalar>& Dt,
    RowMatrix<Scalar>& Dtheta,
    RowMatrix<Scalar>& Dxo,
    RowMatrix<Scalar>& Dyo,
    RowMatrix<Scalar>& Dro,
    RowMatrix<Scalar>& Du,
    RowMatrix<Scalar>& Dinc,
    RowMatrix<Scalar>& Dobl
) {
    computeTaylor(t);
    computeLinearFluxModelInternal(
        theta, xo, yo, zo, ro, X, Dt, Dtheta, Dxo, Dyo, Dro, Du, Dinc, Dobl
    );
}

/**
Compute the linear Ylm model in reflected light and its gradient. 
Basic / Spectral specialization.

*/
template <typename U=S>
inline EnableIf<U::Reflected && !U::Temporal, void> computeLinearFluxModel ( 
    const Vector<Scalar>& theta, 
    const Vector<Scalar>& xo, 
    const Vector<Scalar>& yo, 
    const Vector<Scalar>& zo, 
    const Vector<Scalar>& ro, 
    const RowMatrix<Scalar>& source,
    RowMatrix<Scalar>& X,
    RowMatrix<Scalar>& Dtheta,
    RowMatrix<Scalar>& Dxo,
    RowMatrix<Scalar>& Dyo,
    RowMatrix<Scalar>& Dro,
    RowMatrix<Scalar>& Dsource,
    RowMatrix<Scalar>& Du,
    RowMatrix<Scalar>& Dinc,
    RowMatrix<Scalar>& Dobl
) {
    RowMatrix<Scalar> Dt; // Dummy!
    computeLinearFluxModelInternal(
        theta, xo, yo, zo, ro, source.rowwise().normalized(), X, Dt, 
        Dtheta, Dxo, Dyo, Dro, Dsource, Du, Dinc, Dobl
    );
}

/**
Compute the linear Ylm model in reflected light and its gradient. 
Temporal specialization.

*/
template <typename U=S>
inline EnableIf<U::Reflected && U::Temporal, void> computeLinearFluxModel (
    const Vector<Scalar>& t, 
    const Vector<Scalar>& theta, 
    const Vector<Scalar>& xo, 
    const Vector<Scalar>& yo, 
    const Vector<Scalar>& zo, 
    const Vector<Scalar>& ro, 
    const RowMatrix<Scalar>& source,
    RowMatrix<Scalar>& X,
    RowMatrix<Scalar>& Dt,
    RowMatrix<Scalar>& Dtheta,
    RowMatrix<Scalar>& Dxo,
    RowMatrix<Scalar>& Dyo,
    RowMatrix<Scalar>& Dro,
    RowMatrix<Scalar>& Dsource,
    RowMatrix<Scalar>& Du,
    RowMatrix<Scalar>& Dinc,
    RowMatrix<Scalar>& Dobl
) {
    computeTaylor(t);
    computeLinearFluxModelInternal(
        theta, xo, yo, zo, ro, source.rowwise().normalized(), X, Dt, 
        Dtheta, Dxo, Dyo, Dro, Dsource, Du, Dinc, Dobl
    );
}

/**
Compute the flux from a purely limb-darkened map.

*/
template <typename U=S>
inline EnableIf<U::LimbDarkened, void> computeLimbDarkenedFlux (
    const Vector<Scalar>& b, 
    const Vector<Scalar>& zo, 
    const Vector<Scalar>& ro, 
    FType& flux
) {
    if (ydeg > 0)
        throw std::runtime_error(
            "This method is for purely limb-darkened maps only."
        );
    computeLimbDarkenedFluxInternal(b, zo, ro, flux);
}

/**
Compute the flux from a purely limb-darkened map.
Also compute the gradient.

*/
template <typename U=S>
inline EnableIf<U::LimbDarkened, void> computeLimbDarkenedFlux (
    const Vector<Scalar>& b, 
    const Vector<Scalar>& zo, 
    const Vector<Scalar>& ro, 
    FType& flux,
    FType& Db,
    FType& Dro,
    Matrix<Scalar>& Du
) {
    if (ydeg > 0)
        throw std::runtime_error(
            "This method is for purely limb-darkened maps only."
        );
    computeLimbDarkenedFluxInternal(b, zo, ro, flux, Db, Dro, Du);
}