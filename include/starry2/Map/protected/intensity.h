/**
Evaluate the map at a given (theta, x, y) coordinate.
Internal method.

*/
template <typename T1>
inline void computeIntensityInternal (
    const Scalar& theta,
    const Scalar& x_,
    const Scalar& y_,
    MatrixBase<T1> const & intensity
) {
    // Shape checks
    CHECK_SHAPE(intensity, 1, nflx);

    // Check if outside the sphere
    if (x_ * x_ + y_ * y_ > 1.0) {
        MBCAST(intensity, T1).setConstant(NAN);
        return;
    }

    // Rotate the map into view
    rotateIntoCache(theta);

    // Change basis to polynomials
    cache.A1Ry = B.A1 * cache.Ry;

    // Apply limb darkening
    computeDegreeU();
    if (u_deg > 0) {
        limbDarken(cache.A1Ry, cache.pupy);
        cache.A1Ry = cache.pupy;
    }

    // Compute the polynomial basis
    B.computePolyBasis(x_, y_, cache.pT);

    // Dot the coefficients in to our polynomial map
    MBCAST(intensity, T1) = cache.pT * cache.A1Ry;
}

/**
Render the visible map on a square cartesian grid at given
resolution. Internal method.

*/
template <typename T1>
inline void renderMapInternal (
    const Scalar& theta,
    int res,
    MatrixBase<T1> const & intensity
) {
    // Shape checks
    CHECK_SHAPE(intensity, res * res, nflx);

    // Compute the pixelization matrix
    computeP(res);

    // Rotate the map into view
    rotateIntoCache(theta);

    // Change basis to polynomials
    cache.A1Ry = B.A1 * cache.Ry;

    // Apply limb darkening
    computeDegreeU();
    if (u_deg > 0) {
        limbDarken(cache.A1Ry, cache.pupy);
        cache.A1Ry = cache.pupy;
    }

    // Apply the basis transform
    MBCAST(intensity, T1) = cache.P * cache.A1Ry;
}