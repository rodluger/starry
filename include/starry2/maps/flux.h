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
        limbDarken(cache.A1Ry, cache.pupy);
        cache.A1Ry = cache.pupy;
    }

    // Apply the basis transform
    MBCAST(intensity, T1) = cache.P * cache.A1Ry;
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
        MBCAST(flux, T1) = (L.sT * cache.g).cwiseProduct(norm);

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
    if ((b >= 1 + ro) || (ro == 0.0)) {

        // Easy
        MBCAST(flux, T1) = B.rTA1 * cache.Ry;

    // Occultation
    } else {

        // Compute the solution vector
        G.compute(b, ro);

        // Rotate the occultor onto the y axis if needed
        // and dot the solution vector into the map
        // Recall that f = sTARRy!
        if (likely((b > 0) && ((xo != 0) || (yo < 0)))) {
            W.rotate_about_z(yo / b, xo / b, cache.Ry, cache.RRy);
            MBCAST(flux, T1) = G.sT * B.A * cache.RRy;
        } else {
            MBCAST(flux, T1) = G.sT * B.A * cache.Ry;
        }

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
    if ((b >= 1 + ro) || (ro == 0.0)) {

        // Change basis to polynomials
        cache.A1Ry = B.A1 * cache.Ry;

        // Apply limb darkening
        limbDarken(cache.A1Ry, cache.pupy);

        // Dot the rotation solution vector in
        MBCAST(flux, T1) = B.rT * cache.pupy;

    // Occultation
    } else {

        // Compute the solution vector
        G.compute(b, ro);

        // Rotate the occultor onto the y axis if needed
        // and change basis to polynomials so we can
        // apply the limb darkening
        if (likely((b > 0) && ((xo != 0) || (yo < 0)))) {
            W.rotate_about_z(yo / b, xo / b, cache.Ry, cache.RRy);
            cache.A1Ry = B.A1 * cache.RRy;
        } else {
            cache.A1Ry = B.A1 * cache.Ry;
        }

        // Apply limb darkening
        limbDarken(cache.A1Ry, cache.pupy);

        // Transform to the Greens basis and dot the solution in
        MBCAST(flux, T1) = G.sT * B.A2 * cache.pupy;

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
        UCoeffType flux0 = (L.sT * cache.g);
        MBCAST(flux, T1) = flux0.cwiseProduct(norm);

        // The theta deriv is always zero
        MBCAST(dtheta, T3).setConstant(0.0);

        // dF / db  ->  dF / dx, dF / dy
        if (likely(b > 0)) {
            MBCAST(dxo, T4) = (L.dsTdb * cache.g);
            MBCAST(dxo, T4) /= b;
            MBCAST(dyo, T5) = dxo;
            MBCAST(dxo, T4) *= xo;
            MBCAST(dyo, T5) *= yo;
            MBCAST(dxo, T4) = dxo.cwiseProduct(norm);
            MBCAST(dyo, T5) = dyo.cwiseProduct(norm);
        } else {
            MBCAST(dxo, T4) = L.dsTdb * cache.g;
            MBCAST(dxo, T4) = dxo.cwiseProduct(norm);
            MBCAST(dyo, T5) = dxo;
        }

        // dF / dr
        MBCAST(dro, T6) = (L.dsTdr * cache.g);
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
    if ((b >= 1 + ro) || (ro == 0.0)) {

        // Compute the theta deriv
        MBCAST(dtheta, T3) = B.rTA1 * cache.DRDthetay;
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
        MBCAST(flux, T1) = B.rTA1 * cache.Ry;

        // Compute the time deriv
        computeDfDtYlmNoOccultation(dt);

    // Occultation
    } else {

        // TODO!
        throw errors::NotImplementedError("computeFluxYlm(gradient=true) not yet implemented.");

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
    if ((b >= 1 + ro) || (ro == 0.0)) {

        // Change basis to polynomials
        cache.A1Ry = B.A1 * cache.Ry;

        // Apply limb darkening
        limbDarken(cache.A1Ry, cache.pupy, true);
        MBCAST(flux, T1) = B.rT * cache.pupy;

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
        computeDfDtYlmLDNoOccultation(dt);

    // Occultation
    } else {

        // TODO!
        throw errors::NotImplementedError("computeFluxYlmLD(gradient=true) not yet implemented.");

    }

}