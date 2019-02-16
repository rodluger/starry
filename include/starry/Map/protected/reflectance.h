/**
Compute the reflected flux. Internal method.

*/
template <typename T1>
inline void computeReflectanceInternal (
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

        // Easy
        MBCAST(flux, T1) = B.rTA1 * cache.Ry;

    // Occultation
    } else {

        // \todo Implement occultations in reflected light
        throw errors::NotImplementedError("Occultations in reflected light not yet implemented.");

    }
}

