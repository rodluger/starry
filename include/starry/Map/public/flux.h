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
    const Scalar& ro, 
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
    const Scalar& ro, 
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
    const Scalar& ro, 
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
    const Scalar& ro, 
    const RowMatrix<Scalar>& source,
    RowMatrix<Scalar>& X
) {
    computeTaylor(t);
    computeLinearFluxModelInternal(theta, xo, yo, zo, ro, 
                                   source.rowwise().normalized(), X);
}

/**
Compute the linear Ylm model in emitted light and its (backprop) gradient. 
Basic / Spectral specialization.

*/
template <typename U=S>
inline EnableIf<!U::Reflected && !U::Temporal, void> computeLinearFluxModel (
    const Vector<Scalar>& theta, 
    const Vector<Scalar>& xo, 
    const Vector<Scalar>& yo, 
    const Vector<Scalar>& zo, 
    const Scalar& ro, 
    RowMatrix<Scalar>& X,
    const RowMatrix<Scalar>& bX,
    Vector<Scalar>& btheta, 
    Vector<Scalar>& bxo,
    Vector<Scalar>& byo,
    Scalar& bro,
    UType& bu,
    Vector<Scalar>& bf,
    Scalar& binc,
    Scalar& bobl
) {
    Vector<Scalar> bt; // Dummy!
    computeLinearFluxModelInternal(
        theta, xo, yo, zo, ro, X, bX, bt, btheta, 
        bxo, byo, bro, bu, bf, binc, bobl
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
    const Scalar& ro, 
    RowMatrix<Scalar>& X,
    const RowMatrix<Scalar>& bX,
    Vector<Scalar>& bt, 
    Vector<Scalar>& btheta, 
    Vector<Scalar>& bxo,
    Vector<Scalar>& byo,
    Scalar& bro,
    UType& bu,
    Vector<Scalar>& bf,
    Scalar& binc,
    Scalar& bobl
) {
    computeTaylor(t);
    // DEBUG TODO
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
    const Scalar& ro, 
    const RowMatrix<Scalar>& source,
    RowMatrix<Scalar>& X,
    const RowMatrix<Scalar>& bX,
    Vector<Scalar>& btheta, 
    Vector<Scalar>& bxo,
    Vector<Scalar>& byo,
    Scalar& bro,
    RowMatrix<Scalar>& bsource,
    UType& bu,
    Vector<Scalar>& bf,
    Scalar& binc,
    Scalar& bobl
) {
    Vector<Scalar> bt; // Dummy!
    // DEBUG TODO
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
    const Scalar& ro, 
    const RowMatrix<Scalar>& source,
    RowMatrix<Scalar>& X,
    const RowMatrix<Scalar>& bX,
    Vector<Scalar>& bt, 
    Vector<Scalar>& btheta, 
    Vector<Scalar>& bxo,
    Vector<Scalar>& byo,
    Scalar& bro,
    RowMatrix<Scalar>& bsource,
    UType& bu,
    Vector<Scalar>& bf,
    Scalar& binc,
    Scalar& bobl
) {
    computeTaylor(t);
    // DEBUG TODO
}

/**
Compute the flux from a purely limb-darkened map.

*/
template <typename U=S>
inline EnableIf<U::LimbDarkened, void> computeLimbDarkenedFlux (
    const Vector<Scalar>& b, 
    const Vector<Scalar>& zo, 
    const Scalar& ro, 
    FType& flux
) {
    computeLimbDarkenedFluxInternal(b, zo, ro, flux);
}

/**
Compute the flux from a purely limb-darkened map.
Also compute the (backprop) gradient.

*/
template <typename U=S>
inline EnableIf<U::LimbDarkened, void> computeLimbDarkenedFlux (
    const Vector<Scalar>& b, 
    const Vector<Scalar>& zo, 
    const Scalar& ro, 
    FType& flux,
    const FType& bf,
    Vector<Scalar>& bb,
    Scalar& bro,
    UType& bu
) {
    computeLimbDarkenedFluxInternal(b, zo, ro, flux, 
                                    bf, bb, bro, bu);
}