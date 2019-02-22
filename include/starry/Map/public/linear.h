/**
Compute the linear Ylm model. Default / Spectral specialization.

*/
template <
    typename U=S, 
    typename=IsDefaultOrSpectral<U>, 
    typename=IsEmitted<U>
>
inline void computeLinearModel (
    const Vector<Scalar>& theta, 
    const Vector<Scalar>& xo, 
    const Vector<Scalar>& yo, 
    const Vector<Scalar>& zo, 
    const Vector<Scalar>& ro, 
    RowMatrix<Scalar>& A
) {
    computeLinearModelInternal(theta, xo, yo, zo, ro, A);
}

/**
Compute the linear Ylm model and its gradient. Default / Spectral specialization.

*/
template <
    typename U=S, 
    typename=IsDefaultOrSpectral<U>, 
    typename=IsEmitted<U>
>
inline void computeLinearModel (
    const Vector<Scalar>& theta, 
    const Vector<Scalar>& xo, 
    const Vector<Scalar>& yo, 
    const Vector<Scalar>& zo, 
    const Vector<Scalar>& ro, 
    RowMatrix<Scalar>& A,
    RowMatrix<Scalar>& Dtheta,
    RowMatrix<Scalar>& Dxo,
    RowMatrix<Scalar>& Dyo,
    RowMatrix<Scalar>& Dro
) {
    RowMatrix<Scalar> Dt; // Dummy!
    computeLinearModelInternal(theta, xo, yo, zo, ro, A, Dt, Dtheta, Dxo, Dyo, Dro);
}

/**
Compute the linear Ylm model. Temporal specialization.

*/
template <
    typename U=S, 
    typename=IsTemporal<U>, 
    typename=IsEmitted<U>
>
inline void computeLinearModel (
    const Vector<Scalar>& t,
    const Vector<Scalar>& theta, 
    const Vector<Scalar>& xo, 
    const Vector<Scalar>& yo, 
    const Vector<Scalar>& zo, 
    const Vector<Scalar>& ro, 
    RowMatrix<Scalar>& A
) {
    computeTaylorMatrix(t);
    computeLinearModelInternal(theta, xo, yo, zo, ro, A);
}

/**
Compute the linear Ylm model and its gradient. Temporal specialization.

*/
template <
    typename U=S, 
    typename=IsTemporal<U>, 
    typename=IsEmitted<U>
>
inline void computeLinearModel (
    const Vector<Scalar>& t, 
    const Vector<Scalar>& theta, 
    const Vector<Scalar>& xo, 
    const Vector<Scalar>& yo, 
    const Vector<Scalar>& zo, 
    const Vector<Scalar>& ro, 
    RowMatrix<Scalar>& A,
    RowMatrix<Scalar>& Dt,
    RowMatrix<Scalar>& Dtheta,
    RowMatrix<Scalar>& Dxo,
    RowMatrix<Scalar>& Dyo,
    RowMatrix<Scalar>& Dro
) {
    computeTaylorMatrix(t);
    computeLinearModelInternal(theta, xo, yo, zo, ro, A, Dt, Dtheta, Dxo, Dyo, Dro);
}