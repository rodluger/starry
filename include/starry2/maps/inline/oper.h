/**
Generate a random isotropic map with a given power spectrum.

*/
template<typename V, typename U=S, typename=IsSingleColumn<U>>
inline void random (
    const Vector<Scalar>& power,
    const V& seed
) {
    random_(power, seed, 0);
}

/**
Generate a random isotropic map with a given power spectrum.
NOTE: If `col = -1`, sets all columns to the same map.

*/
template<typename V, typename U=S, typename=IsMultiColumn<U>>
inline void random (
    const Vector<Scalar>& power,
    const V& seed,
    int col=-1
) {
    random_(power, seed, col);
}

/**
Compute the Taylor expansion basis at a point in time.
Static specialization (does nothing).

*/
template<typename U=S>
inline IsStatic<U, void> computeTaylor (
    const Scalar & t
) {
}

/**
Compute the Taylor expansion basis at a point in time.
Temporal specialization.

*/
template<typename U=S>
inline IsTemporal<U, void> computeTaylor (
    const Scalar & t
) {
    if (t != cache.taylort) {
        for (int n = 1; n < ncoly; ++n)
            taylor(n) = taylor(n - 1) * t / n;
        cache.taylort = t;
    }
}

/**
Temporal contraction operation for static maps: 
does nothing, and returns a reference to the 
original map.

*/
template<typename U=S, typename T1>
inline IsStatic<U, MatrixBase<T1>&> contract (
    MatrixBase<T1> const & mat
) {
    return MBCAST(mat, T1);
}

/**
Contracts a temporal map by dotting the map matrix with the
Taylor expansion basis.

*/
template<typename U=S>
inline IsTemporal<U, Vector<Scalar>> contract (
    const Matrix<Scalar> & mat
) {
    return mat * taylor;
}

/**
Set the zeroth order limb darkening coefficient.
This is a **constant** whose value ensures that
I(mu = 0) / I0 = 1.

*/
template<typename U=S>
inline IsStatic<U, void> setU0 () {
    u.row(0).setConstant(-1.0);
}

/**
Set the zeroth order limb darkening coefficient
for a temporal map. All derivatives are set to
zero.

*/
template<typename U=S>
inline IsTemporal<U, void> setU0 () {
    u.row(0).setZero();
    u(0, 0) = -1.0;
}

/**
Normalize the Agol C basis and its derivatives. 
Default map specialization.

*/
template<typename U=S, typename T1, typename T2>
inline IsDefault<U, void> normalizeC (
    MatrixBase<T1> const & c,
    MatrixBase<T2> const & dcdu
) {
    // The total flux is given by `y00 * (s . c)`
    Scalar norm = Scalar(1.0) / (pi<Scalar>() * (c(0) + 2.0 * c(1) / 3.0));
    MBCAST(c, T1) = c * norm;
    MBCAST(dcdu, T2) = dcdu * norm;
}

/**
Normalize the Agol C basis and its derivatives. 
Spectral map specialization.

*/
template<typename U=S, typename T1, typename T2>
inline IsSpectral<U, void> normalizeC (
    MatrixBase<T1> const & c,
    MatrixBase<T2> const & dcdu
) {
    // The total flux is given by `y00 * (s . c)`
    for (int n = 0; n < ncoly; ++n) {
        Scalar norm = Scalar(1.0) / (pi<Scalar>() * (c(0, n) + 2.0 * c(1, n) / 3.0));
        MBCAST(c, T1).col(n) = c.col(n) * norm;
        MBCAST(dcdu, T2).block(n * lmax, 0, lmax, lmax + 1) = 
            dcdu.block(n * lmax, 0, lmax, lmax + 1) * norm;
    }
}

/**
Normalize the Agol C basis and its derivatives. 
Temporal map specialization.

*/
template<typename U=S, typename T1, typename T2>
inline IsTemporal<U, void> normalizeC (
    MatrixBase<T1> const & c,
    MatrixBase<T2> const & dcdu
) {
    // The total flux is given by `y00 * (s . c)`
    Scalar norm = Scalar(1.0) / (pi<Scalar>() * (c(0) + 2.0 * c(1) / 3.0));
    MBCAST(c, T1) = c * norm;
    MBCAST(dcdu, T2) = dcdu * norm;

}