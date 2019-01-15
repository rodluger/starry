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
Rotate the map by an angle `theta` and store
the result in `cache.Ry`. Optionally compute
and cache the Wigner rotation matrices and
their derivatives. Static map specialization.

*/
template<typename U=S>
inline IsStatic<U, void> rotateIntoCache (
    const Scalar& theta,
    bool compute_matrices=false
) 
{
    Scalar theta_rad = theta * radian;
    computeWigner();
    if ((!compute_matrices) && (cache.theta != theta)) {
        W.rotate(cos(theta_rad), sin(theta_rad), cache.Ry);
        cache.theta = theta;
    } else if (compute_matrices && (cache.theta_with_grad != theta)) {
        W.compute(cos(theta_rad), sin(theta_rad));
        for (int l = 0; l < lmax + 1; ++l) {
            cache.Ry.block(l * l, 0, 2 * l + 1, ncoly) =
                W.R[l] * y.block(l * l, 0, 2 * l + 1, ncoly);
            cache.dRdthetay.block(l * l, 0, 2 * l + 1, ncoly) =
                W.dRdtheta[l] * y.block(l * l, 0, 2 * l + 1, ncoly);
        }
        cache.theta_with_grad = theta;
    }
}

/**
Rotate the map by an angle `theta` and store
the result in `cache.Ry`. Optionally compute
and cache the Wigner rotation matrices and
their derivatives. Temporal map specialization.

*/
template<typename U=S>
inline IsTemporal<U, void> rotateIntoCache (
    const Scalar& theta,
    bool compute_matrices=false
) 
{

    // TODO: Caching is broken below for same theta, different t
    throw errors::ToDoError("TODO");
    
    Scalar theta_rad = theta * radian;
    computeWigner();
    if ((!compute_matrices) && (cache.theta != theta)) {
        W.rotate(cos(theta_rad), sin(theta_rad), cache.RyUncontracted);
        cache.theta = theta;
        cache.Ry = contract(cache.RyUncontracted);
    } else if (compute_matrices && (cache.theta_with_grad != theta)) {
        W.compute(cos(theta_rad), sin(theta_rad));
        for (int l = 0; l < lmax + 1; ++l) {
            cache.RyUncontracted.block(l * l, 0, 2 * l + 1, ncoly) =
                W.R[l] * y.block(l * l, 0, 2 * l + 1, ncoly);
            cache.Ry = contract(cache.RyUncontracted);
            cache.dRdthetay.block(l * l, 0, 2 * l + 1, nflx) =
                contract(W.dRdtheta[l] * y.block(l * l, 0, 2 * l + 1, ncoly));
        }
        cache.theta_with_grad = theta;
    }
}

/**
Normalize the Agol g basis and its derivatives. 
Default map specialization.

*/
template<typename U=S, typename T1, typename T2>
inline IsDefault<U, void> normalizeAgolG (
    MatrixBase<T1> const & g,
    MatrixBase<T2> const & dAgolGdu
) {
    // The total flux is given by `y00 * (s . g)`
    Scalar norm = Scalar(1.0) / (pi<Scalar>() * (g(0) + 2.0 * g(1) / 3.0));
    MBCAST(g, T1) = g * norm;
    MBCAST(dAgolGdu, T2) = dAgolGdu * norm;
}

/**
Normalize the Agol g basis and its derivatives. 
Spectral map specialization.

*/
template<typename U=S, typename T1, typename T2>
inline IsSpectral<U, void> normalizeAgolG (
    MatrixBase<T1> const & g,
    MatrixBase<T2> const & dAgolGdu
) {
    // The total flux is given by `y00 * (s . g)`
    for (int n = 0; n < ncoly; ++n) {
        Scalar norm = Scalar(1.0) / (pi<Scalar>() * (g(0, n) + 2.0 * g(1, n) / 3.0));
        MBCAST(g, T1).col(n) = g.col(n) * norm;
        MBCAST(dAgolGdu, T2).block(n * lmax, 0, lmax, lmax + 1) = 
            dAgolGdu.block(n * lmax, 0, lmax, lmax + 1) * norm;
    }
}

/**
Normalize the Agol g basis and its derivatives. 
Temporal map specialization.

*/
template<typename U=S, typename T1, typename T2>
inline IsTemporal<U, void> normalizeAgolG (
    MatrixBase<T1> const & g,
    MatrixBase<T2> const & dAgolGdu
) {
    // The total flux is given by `y00 * (s . c)`
    Scalar norm = Scalar(1.0) / (pi<Scalar>() * (g(0) + 2.0 * g(1) / 3.0));
    MBCAST(g, T1) = g * norm;
    MBCAST(dAgolGdu, T2) = dAgolGdu * norm;

}