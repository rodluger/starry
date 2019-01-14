// --------------------------
// ------- Intensity --------
// --------------------------


/**
Evaluate the map at a given (theta, x, y) coordinate.

*/
template <class S>
template <typename T1>
inline void Map<S>::computeIntensity_(
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
        limbDarken(cache.A1Ry, cache.p_uy);
        cache.A1Ry = cache.p_uy;
    }

    // Compute the polynomial basis
    B.computePolyBasis(x_, y_, cache.pT);

    // Dot the coefficients in to our polynomial map
    auto result = cache.pT * cache.A1Ry;

    // Contract the map if needed
    MBCAST(intensity, T1) = contract(result);
}

/**
Render the visible map on a square cartesian grid at given
resolution. 

*/
template <class S>
template <typename T1>
inline void Map<S>::renderMap_(
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
        limbDarken(cache.A1Ry, cache.p_uy);
        cache.A1Ry = cache.p_uy;
    }

    // Apply the basis transform
    auto result = cache.P * cache.A1Ry;
    MBCAST(intensity, T1) = contract(result);
}

// --------------------------
// ---------- Flux ----------
// --------------------------


/**

*/
template <class S>
template <typename T1>
inline void Map<S>::computeFlux_(
    const Scalar& theta, 
    const Scalar& xo, 
    const Scalar& yo, 
    const Scalar& ro, 
    MatrixBase<T1> const & flux
) {
    // Shape checks
    CHECK_SHAPE(flux, 1, nflx);

    // Figure out the degree of the map
    computeDegreeU();
    computeDegreeY();

    // Impact parameter
    Scalar b = sqrt(xo * xo + yo * yo);

    // Check for complete occultation
    if (b <= ro - 1) {
        MBCAST(flux, T1).setZero();
        return;
    }

    // Compute the flux
    if (y_deg == 0) {
        computeFluxLD(b, ro, flux);
    } else if (u_deg == 0) {
        computeFluxYlm(theta, xo, yo, b, ro, flux);
    } else {
        computeFluxYlmLD(theta, xo, yo, b, ro, flux);
    }
}

template <class S>
template <typename T1>
inline void Map<S>::computeFluxLD(
    const Scalar& b, 
    const Scalar& ro, 
    MatrixBase<T1> const & flux
) {
    // No occultation
    if ((b >= 1 + ro) || (ro == 0.0)) {

        // Easy: the disk-integrated intensity
        // is just the Y_{0,0} coefficient
        MBCAST(flux, T1) = contract(y.row(0));

    // Occultation
    } else {

        // Compute the Agol `s` vector
        L.compute(b, ro);

        // Compute the Agol `g` basis
        computeAgolGBasis();

        // Normalize by y00
        UCoeffType norm = contract(y.row(0));

        // Dot the integral solution in, and we're done!
        MBCAST(flux, T1) = (L.s * cache.agol_g).cwiseProduct(norm);

    }
}

template <class S>
template <typename T1>
inline void Map<S>::computeFluxYlm(
    const Scalar& theta,
    const Scalar& xo,
    const Scalar& yo,  
    const Scalar& b, 
    const Scalar& ro, 
    MatrixBase<T1> const & flux
) {
    // Rotate the map into view
    rotateIntoCache(theta);

    // No occultation
    if (1) { // DEBUG ((b >= 1 + ro) || (ro == 0.0)) {

        // Easy
        MBCAST(flux, T1) = contract(B.rTA1 * cache.Ry);

    // Occultation
    } else {

        // TODO!
        throw errors::NotImplementedError("Not yet implemented.");

    }
}

template <class S>
template <typename T1>
inline void Map<S>::computeFluxYlmLD(
    const Scalar& theta,
    const Scalar& xo,
    const Scalar& yo,  
    const Scalar& b, 
    const Scalar& ro, 
    MatrixBase<T1> const & flux
) {
    // Rotate the map into view
    rotateIntoCache(theta);

    // No occultation
    if (1) { // DEBUG ((b >= 1 + ro) || (ro == 0.0)) {

        // Change basis to polynomials
        cache.A1Ry = B.A1 * cache.Ry;

        // Apply limb darkening
        limbDarken(cache.A1Ry, cache.p_uy);
        MBCAST(flux, T1) = contract(B.rT * cache.p_uy);

    // Occultation
    } else {

        // TODO!
        throw errors::NotImplementedError("Not yet implemented.");

    }
}

// --------------------------
// ---- Flux + gradients ----
// --------------------------


template <class S>
template <typename T1, typename T2, typename T3, typename T4, 
          typename T5, typename T6, typename T7, typename T8>
inline void Map<S>::computeFlux_(
    const Scalar& theta, 
    const Scalar& xo, 
    const Scalar& yo, 
    const Scalar& ro, 
    MatrixBase<T1> const & flux, 
    MatrixBase<T2> const & dt,
    MatrixBase<T3> const & dtheta,
    MatrixBase<T4> const & dxo,
    MatrixBase<T5> const & dyo,
    MatrixBase<T6> const & dro,
    MatrixBase<T7> const & dy,
    MatrixBase<T8> const & du
) {
    // Shape checks
    CHECK_SHAPE(flux, 1, nflx);
    CHECK_SHAPE(dt, 1, nflx);
    CHECK_SHAPE(dtheta, 1, nflx);
    CHECK_SHAPE(dxo, 1, nflx);
    CHECK_SHAPE(dyo, 1, nflx);
    CHECK_SHAPE(dro, 1, nflx);
    CHECK_COLS(dy, ncoly);
    CHECK_COLS(du, ncolu);

    // Figure out the degree of the map
    computeDegreeU();
    computeDegreeY();

    // Impact parameter
    Scalar b = sqrt(xo * xo + yo * yo);

    // Check for complete occultation
    if (b <= ro - 1) {
        MBCAST(flux, T1).setZero();
        MBCAST(dt, T2).setZero();
        MBCAST(dtheta, T3).setZero();
        MBCAST(dxo, T4).setZero();
        MBCAST(dyo, T5).setZero();
        MBCAST(dro, T6).setZero();
        MBCAST(dy, T7).setZero();
        MBCAST(du, T8).setZero();
        return;
    }

    // Compute the flux
    if (y_deg == 0) {
        CHECK_ROWS(du, lmax + STARRY_DFDU_DELTA);
        computeFluxLD(xo, yo, b, ro, flux, dt, 
                      dtheta, dxo, dyo, dro, dy, du);
    } else if (u_deg == 0) {
        CHECK_ROWS(dy, N);
        computeFluxYlm(theta, xo, yo, b, ro, flux, dt, 
                       dtheta, dxo, dyo, dro, dy, du);
    } else {
        CHECK_ROWS(dy, N);
        CHECK_ROWS(du, lmax + STARRY_DFDU_DELTA);
        computeFluxYlmLD(theta, xo, yo, b, ro, flux, dt, 
                         dtheta, dxo, dyo, dro, dy, du);
    }
}

template <class S>
template <typename T1, typename T2, typename T3, typename T4, 
          typename T5, typename T6, typename T7, typename T8>
inline void Map<S>::computeFluxLD(
    const Scalar& xo, 
    const Scalar& yo, 
    const Scalar& b, 
    const Scalar& ro, 
    MatrixBase<T1> const & flux, 
    MatrixBase<T2> const & dt,
    MatrixBase<T3> const & dtheta,
    MatrixBase<T4> const & dxo,
    MatrixBase<T5> const & dyo,
    MatrixBase<T6> const & dro,
    MatrixBase<T7> const & dy,
    MatrixBase<T8> const & du
) {

    // No occultation
    if ((b >= 1 + ro) || (ro == 0.0)) {

        // Most of the derivs are zero
        MBCAST(dtheta, T3).setZero();
        MBCAST(dxo, T4).setZero();
        MBCAST(dyo, T5).setZero();
        MBCAST(dro, T6).setZero();
        MBCAST(du, T8).setZero();

        // dF / dy
        computeDfDyLDNoOccultation(dy);

        // dF / dt
        computeDfDtLDNoOccultation(dt);
        
        // The disk-integrated intensity
        // is just the Y_{0,0} coefficient
        MBCAST(flux, T1) = contract(y.row(0));

    // Occultation
    } else {

        // Compute the Agol `s` vector and its derivs
        L.compute(b, ro, true);

        // Compute the Agol `g` basis
        computeAgolGBasis();

        // Compute the normalization
        UCoeffType norm = contract(y.row(0));

        // Compute the flux
        UCoeffType flux0 = (L.s * cache.agol_g);
        MBCAST(flux, T1) = flux0.cwiseProduct(norm);

        // The theta deriv is always zero
        MBCAST(dtheta, T3).setConstant(0.0);

        // dF / db  ->  dF / dx, dF / dy
        if (likely(b > 0)) {
            MBCAST(dxo, T4) = (L.dsdb * cache.agol_g);
            MBCAST(dxo, T4) /= b;
            MBCAST(dyo, T5) = dxo;
            MBCAST(dxo, T4) *= xo;
            MBCAST(dyo, T5) *= yo;
            MBCAST(dxo, T4) = dxo.cwiseProduct(norm);
            MBCAST(dyo, T5) = dyo.cwiseProduct(norm);
        } else {
            MBCAST(dxo, T4) = L.dsdb * cache.agol_g;
            MBCAST(dxo, T4) = dxo.cwiseProduct(norm);
            MBCAST(dyo, T5) = dxo;
        }

        // dF / dr
        MBCAST(dro, T6) = (L.dsdr * cache.agol_g);
        MBCAST(dro, T6) = dro.cwiseProduct(norm);

        // dF / dy
        computeDfDyLDOccultation(dy, flux0);

        // dF / dt
        computeDfDtLDOccultation(dt, flux0);

        // dF / du from dF / dc
        computeDfDuLDOccultation(flux0, du, norm);

    }

}

template <class S>
template <typename T1, typename T2, typename T3, typename T4, 
          typename T5, typename T6, typename T7, typename T8>
inline void Map<S>::computeFluxYlm (
    const Scalar& theta,
    const Scalar& xo,
    const Scalar& yo,  
    const Scalar& b, 
    const Scalar& ro, 
    MatrixBase<T1> const & flux, 
    MatrixBase<T2> const & dt,
    MatrixBase<T3> const & dtheta,
    MatrixBase<T4> const & dxo,
    MatrixBase<T5> const & dyo,
    MatrixBase<T6> const & dro,
    MatrixBase<T7> const & dy,
    MatrixBase<T8> const & du
) {

    // Rotate the map into view and explicitly
    // compute the Wigner matrices
    rotateIntoCache(theta, true);

    // No occultation
    if (1) { // DEBUG ((b >= 1 + ro) || (ro == 0.0)) {

        // Compute the theta deriv
        MBCAST(dtheta, T3) = contract(B.rTA1 * cache.dRdthetay);
        MBCAST(dtheta, T3) *= radian;

        // The xo, yo, and ro derivs are trivial
        MBCAST(dxo, T4).setZero();
        MBCAST(dyo, T5).setZero();
        MBCAST(dro, T6).setZero();
        
        // Compute derivs with respect to y
        computeDfDyYlmNoOccultation(dy, theta);

        // Note that we do not compute limb darkening 
        // derivatives in this case; see the docs.
        MBCAST(du, T8).setZero();

        // Compute the flux
        auto flux0 = B.rTA1 * cache.Ry;
        MBCAST(flux, T1) = contract(flux0);

        // Compute the time deriv
        computeDfDtYlmNoOccultation(dt, flux0);

    // Occultation
    } else {

        // TODO!
        throw errors::NotImplementedError("Not yet implemented.");

    }

}

template <class S>
template <typename T1, typename T2, typename T3, typename T4, 
          typename T5, typename T6, typename T7, typename T8>
inline void Map<S>::computeFluxYlmLD(
    const Scalar& theta,
    const Scalar& xo,
    const Scalar& yo,  
    const Scalar& b, 
    const Scalar& ro, 
    MatrixBase<T1> const & flux, 
    MatrixBase<T2> const & dt,
    MatrixBase<T3> const & dtheta,
    MatrixBase<T4> const & dxo,
    MatrixBase<T5> const & dyo,
    MatrixBase<T6> const & dro,
    MatrixBase<T7> const & dy,
    MatrixBase<T8> const & du
) {

    // Rotate the map into view and explicitly
    // compute the Wigner matrices
    rotateIntoCache(theta, true);

    // No occultation
    if (1) { // DEBUG ((b >= 1 + ro) || (ro == 0.0)) {

        // Change basis to polynomials
        cache.A1Ry = B.A1 * cache.Ry;

        // Apply limb darkening
        limbDarken(cache.A1Ry, cache.p_uy, true);
        auto flux0 = B.rTA1 * cache.p_uy;
        MBCAST(flux, T1) = contract(flux0);

        // Compute the map derivs
        computeDfDyYlmLDNoOccultation(dy);
        computeDfDuYlmLDNoOccultation(du);

        // The xo, yo, and ro derivs are trivial
        MBCAST(dxo, T4).setZero();
        MBCAST(dyo, T5).setZero();
        MBCAST(dro, T6).setZero();

        // Compute the theta deriv
        computeDfDthetaYlmLDNoOccultation(dtheta);

        // Compute the time deriv
        computeDfDtYlmLDNoOccultation(dt, flux0);

    // Occultation
    } else {

        // TODO!
        throw errors::NotImplementedError("Not yet implemented.");

    }

}