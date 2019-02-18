/**
Compute the reflected flux. Internal method.

*/
template <typename T1>
inline void computeReflectedFluxInternal (
    const Scalar& theta, 
    const Scalar& xo, 
    const Scalar& yo, 
    const Scalar& zo, 
    const Scalar& ro, 
    const UnitVector<Scalar>& source,
    MatrixBase<T1> const & flux
) {
    // Shape checks
    CHECK_SHAPE(flux, 1, nflx);

    // Impact parameter
    Scalar b = sqrt(xo * xo + yo * yo);

    // Check for complete occultation
    if (b <= ro - 1) {
        MBCAST(flux, T1).setZero();
        return;
    }

    // Rotate the map into view
    rotateIntoCache(theta);

    // No occultation
    if ((zo < 0) || (b >= 1 + ro) || (ro <= 0.0)) {

        // \todo Implement phase curves in reflected light
        throw errors::NotImplementedError("Phase curves in reflected light not yet implemented.");

    // Occultation
    } else {

        // \todo Implement occultations in reflected light
        throw errors::NotImplementedError("Occultations in reflected light not yet implemented.");

    }
}

/**
Evaluate the reflected map at a given (theta, x, y) coordinate.
Internal method.

*/
template <typename T1>
inline void computeReflectedIntensityInternal (
    const Scalar& theta,
    const Scalar& x_,
    const Scalar& y_,
    const UnitVector<Scalar>& source,
    MatrixBase<T1> const & intensity
) {
    // Shape checks
    CHECK_SHAPE(intensity, 1, nflx);

    // Check if outside the sphere
    Scalar r2_ = x_ * x_ + y_ * y_;
    if (r2_ > 1.0) {
        MBCAST(intensity, T1).setConstant(NAN);
        return;
    }

    // Get the source vector components
    Scalar sx = source(0);
    Scalar sy = source(1);
    Scalar sz = source(2);

    // Compute the terminator curve
    // and check if we're on the night side
    Scalar b = -sz;
    Scalar yrot;
    if (unlikely((sx == 0) && (sy == 0))) {
        if (sz < 0) {
            MBCAST(intensity, T1).setZero();
            return;
        }
        yrot = y_;
    } else {
        Scalar invsr = Scalar(1.0) / sqrt(sx * sx + sy * sy);
        Scalar cosw = sy * invsr;
        Scalar sinw = -sx * invsr;
        Scalar xrot = x_ * cosw + y_ * sinw;
        Scalar yterm = b * sqrt(Scalar(1.0) - Scalar(xrot * xrot));
        yrot = -x_ * sinw + y_ * cosw;
        if (yrot < yterm) {
            MBCAST(intensity, T1).setZero();
            return;
        }
    }

    // Rotate the map into view
    rotateIntoCache(theta);

    // Change basis to polynomials
    cache.A1Ry = B.A1 * cache.Ry;

    // Compute the polynomial basis
    B.computePolyBasis(x_, y_, cache.pT);

    // Dot the coefficients in to our polynomial map
    // and multiply by the illumination
    Scalar z_ = sqrt(Scalar(1.0) - r2_);
    Scalar illum = sqrt(Scalar(1.0) - Scalar(b * b)) * yrot - b * z_;
    MBCAST(intensity, T1) = illum * cache.pT * cache.A1Ry;

}

/**
Render the reflected map on a square cartesian grid at given
resolution. Internal method.

*/
template <typename T1>
inline void renderReflectedMapInternal (
    const Scalar& theta,
    const UnitVector<Scalar>& source,
    int res,
    MatrixBase<T1> const & intensity
) {
    // Shape checks
    CHECK_SHAPE(intensity, res * res, nflx);

    // Compute the pixelization matrix
    computeP(res);

    // Compute the illumination matrix
    computeI(res, source);

    // Rotate the map into view
    rotateIntoCache(theta);

    // Change basis to polynomials
    cache.A1Ry = B.A1 * cache.Ry;

    // Apply the basis transform
    MBCAST(intensity, T1) = (cache.P * cache.A1Ry).cwiseProduct(cache.I);

}