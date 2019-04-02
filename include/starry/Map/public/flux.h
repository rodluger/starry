/**
Compute the linear Ylm model. Default / Spectral specialization.

*/
template <
    typename U=S, 
    typename=IsDefaultOrSpectral<U>, 
    typename=IsEmitted<U>
>
inline void computeLinearFluxModel (
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
Compute the linear Ylm model. Temporal specialization.

*/
template <
    typename U=S, 
    typename=IsTemporal<U>, 
    typename=IsEmitted<U>
>
inline void computeLinearFluxModel (
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
Compute the linear Ylm model. Default / Spectral specialization.

*/
template <
    typename U=S, 
    typename=IsDefaultOrSpectral<U>, 
    typename=IsReflected<U>
>
inline void computeLinearFluxModel (
    const Vector<Scalar>& theta, 
    const Vector<Scalar>& xo, 
    const Vector<Scalar>& yo, 
    const Vector<Scalar>& zo, 
    const Vector<Scalar>& ro, 
    const RowMatrix<Scalar>& source,
    RowMatrix<Scalar>& X
) {
    computeLinearFluxModelInternal(theta, xo, yo, zo, ro, source.rowwise().normalized(), X);
}

/**
Compute the linear Ylm model. Temporal specialization.

*/
template <
    typename U=S, 
    typename=IsTemporal<U>, 
    typename=IsReflected<U>
>
inline void computeLinearFluxModel (
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
    computeLinearFluxModelInternal(theta, xo, yo, zo, ro, source.rowwise().normalized(), X);
}

/**
Compute the linear Ylm model and its gradient. Default / Spectral specialization.

*/
template <
    typename U=S, 
    typename=IsDefaultOrSpectral<U>, 
    typename=IsEmitted<U>
>
inline void computeLinearFluxModel (
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
Compute the linear Ylm model and its gradient. Temporal specialization.

*/
template <
    typename U=S, 
    typename=IsTemporal<U>, 
    typename=IsEmitted<U>
>
inline void computeLinearFluxModel (
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
Compute the linear Ylm model and its gradient. Default / Spectral specialization.

*/
template <
    typename U=S, 
    typename=IsDefaultOrSpectral<U>, 
    typename=IsReflected<U>
>
inline void computeLinearFluxModel ( 
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
        theta, xo, yo, zo, ro, source.rowwise().normalized(), X, Dt, Dtheta, Dxo, Dyo, Dro, Dsource, Du, Dinc, Dobl
    );
}

/**
Compute the linear Ylm model and its gradient. Temporal specialization.

*/
template <
    typename U=S, 
    typename=IsTemporal<U>, 
    typename=IsReflected<U>
>
inline void computeLinearFluxModel (
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
        theta, xo, yo, zo, ro, source.rowwise().normalized(), X, Dt, Dtheta, Dxo, Dyo, Dro, Dsource, Du, Dinc, Dobl
    );
}