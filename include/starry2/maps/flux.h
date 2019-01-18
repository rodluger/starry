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
            W.rotateAboutZ(yo / b, xo / b, cache.Ry, cache.RRy);
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
            W.rotateAboutZ(yo / b, xo / b, cache.Ry, cache.RRy);
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
    MatrixBase<T2> const & Dt,
    MatrixBase<T3> const & Dtheta,
    MatrixBase<T4> const & Dxo,
    MatrixBase<T5> const & Dyo,
    MatrixBase<T6> const & Dro,
    MatrixBase<T7> const & Dy,
    MatrixBase<T8> const & Du
) {
    // Shape checks
    CHECK_SHAPE(flux, 1, nflx);
    CHECK_SHAPE(Dt, 1, nflx);
    CHECK_SHAPE(Dtheta, 1, nflx);
    CHECK_SHAPE(Dxo, 1, nflx);
    CHECK_SHAPE(Dyo, 1, nflx);
    CHECK_SHAPE(Dro, 1, nflx);
    CHECK_COLS(Dy, ncoly);
    CHECK_COLS(Du, ncolu);

    // Figure out the degree of the map
    computeDegreeU();
    computeDegreeY();

    // Impact parameter
    Scalar b = sqrt(xo * xo + yo * yo);

    // Check for complete occultation
    if (b <= ro - 1) {
        MBCAST(flux, T1).setZero();
        MBCAST(Dt, T2).setZero();
        MBCAST(Dtheta, T3).setZero();
        MBCAST(Dxo, T4).setZero();
        MBCAST(Dyo, T5).setZero();
        MBCAST(Dro, T6).setZero();
        MBCAST(Dy, T7).setZero();
        MBCAST(Du, T8).setZero();
        return;
    }

    // Compute the flux
    if (y_deg == 0) {
        CHECK_ROWS(Du, lmax + STARRY_DFDU_DELTA);
        computeFluxLD(xo, yo, b, ro, flux, Dt, 
                      Dtheta, Dxo, Dyo, Dro, Dy, Du);
    } else if (u_deg == 0) {
        CHECK_ROWS(Dy, N);
        computeFluxYlm(theta, xo, yo, b, ro, flux, Dt, 
                       Dtheta, Dxo, Dyo, Dro, Dy, Du);
    } else {
        CHECK_ROWS(Dy, N);
        CHECK_ROWS(Du, lmax + STARRY_DFDU_DELTA);
        computeFluxYlmLD(theta, xo, yo, b, ro, flux, Dt, 
                         Dtheta, Dxo, Dyo, Dro, Dy, Du);
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
    MatrixBase<T2> const & Dt,
    MatrixBase<T3> const & Dtheta,
    MatrixBase<T4> const & Dxo,
    MatrixBase<T5> const & Dyo,
    MatrixBase<T6> const & Dro,
    MatrixBase<T7> const & Dy,
    MatrixBase<T8> const & Du
) {

    // No occultation
    if ((b >= 1 + ro) || (ro == 0.0)) {

        // Most of the derivs are zero
        MBCAST(Dtheta, T3).setZero();
        MBCAST(Dxo, T4).setZero();
        MBCAST(Dyo, T5).setZero();
        MBCAST(Dro, T6).setZero();
        MBCAST(Du, T8).setZero();

        // dF / Dy
        computeDfDyLDNoOccultation(Dy);

        // dF / Dt
        computeDfDtLDNoOccultation(Dt);
        
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
        MBCAST(Dtheta, T3).setConstant(0.0);

        // dF / db  ->  dF / dx, dF / Dy
        if (likely(b > 0)) {
            MBCAST(Dxo, T4) = (L.dsTdb * cache.g);
            MBCAST(Dxo, T4) /= b;
            MBCAST(Dyo, T5) = Dxo;
            MBCAST(Dxo, T4) *= xo;
            MBCAST(Dyo, T5) *= yo;
            MBCAST(Dxo, T4) = Dxo.cwiseProduct(norm);
            MBCAST(Dyo, T5) = Dyo.cwiseProduct(norm);
        } else {
            MBCAST(Dxo, T4) = L.dsTdb * cache.g;
            MBCAST(Dxo, T4) = Dxo.cwiseProduct(norm);
            MBCAST(Dyo, T5) = Dxo;
        }

        // dF / dr
        MBCAST(Dro, T6) = (L.dsTdr * cache.g);
        MBCAST(Dro, T6) = Dro.cwiseProduct(norm);

        // dF / Dy
        computeDfDyLDOccultation(Dy, flux0);

        // dF / Dt
        computeDfDtLDOccultation(Dt, flux0);

        // dF / Du from dF / dc
        computeDfDuLDOccultation(flux0, Du, norm);

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
    MatrixBase<T2> const & Dt,
    MatrixBase<T3> const & Dtheta,
    MatrixBase<T4> const & Dxo,
    MatrixBase<T5> const & Dyo,
    MatrixBase<T6> const & Dro,
    MatrixBase<T7> const & Dy,
    MatrixBase<T8> const & Du
) {

    // Rotate the map into view and explicitly
    // compute the Wigner matrices
    rotateIntoCache(theta, true);

    // No occultation
    if ((b >= 1 + ro) || (ro == 0.0)) {

        // Compute the theta deriv
        MBCAST(Dtheta, T3) = B.rTA1 * cache.DRDthetay;
        MBCAST(Dtheta, T3) *= radian;

        // The xo, yo, and ro derivs are trivial
        MBCAST(Dxo, T4).setZero();
        MBCAST(Dyo, T5).setZero();
        MBCAST(Dro, T6).setZero();
        
        // Compute derivs with respect to y
        computeDfDyYlmNoOccultation(Dy, theta);

        // Note that we do not compute limb darkening 
        // derivatives in this case; see the docs.
        MBCAST(Du, T8).setZero();

        // Compute the flux
        MBCAST(flux, T1) = B.rTA1 * cache.Ry;

        // Compute the time deriv
        computeDfDtYlmNoOccultation(Dt);

    // Occultation
    } else {

        // Compute the solution vector and its gradient
        G.compute(b, ro, true);

        // Transform the solution vector into Ylms
        cache.sTA = G.sT * B.A;

        // The normalized occultor position
        Scalar xo_b = xo / b,
               yo_b = yo / b;

        // Align the occultor with the y axis
        if (likely((b > 0) && ((xo != 0) || (yo < 0)))) {
            // Compute the rotation matrix and its derivative
            W.rotateAboutZ(yo_b, xo_b, cache.Ry, cache.RRy);

            // Dot sTA into R and dRdphi 
            W.leftMultiplyRz(cache.sTA, cache.sTAR);
            W.leftMultiplyDRz(cache.sTA, cache.sTADRDphi);
        
            // The Green's polynomial of the rotated map
            cache.ARRy = B.A * cache.RRy;

            // Compute the contribution to the xo and yo
            // derivs from the occultor rotation matrix
            cache.sTADRDphiRy_b = cache.sTADRDphi * cache.Ry;
            cache.sTADRDphiRy_b /= b;
            MBCAST(Dxo, T4) = yo_b * cache.sTADRDphiRy_b;
            MBCAST(Dyo, T4) = -xo_b * cache.sTADRDphiRy_b;
        } else {
            cache.sTAR = cache.sTA;
            cache.ARRy = B.A * cache.Ry;
            MBCAST(Dxo, T4).setZero();
            MBCAST(Dyo, T5).setZero();
        }   

        // Compute the contribution to the xo and yo
        // derivs from the solution vector
        if (likely(b > 0)) {
            cache.dFdb = G.dsTdb * cache.ARRy;
            MBCAST(Dxo, T4) += cache.dFdb * xo_b;
            MBCAST(Dyo, T5) += cache.dFdb * yo_b;
        }

        // Compute the flux
        MBCAST(flux, T1) = G.sT * cache.ARRy;
    
        // Theta derivative
        MBCAST(Dtheta, T3) = cache.sTAR * cache.DRDthetay;
        MBCAST(Dtheta, T3) *= radian;

        // Occultor radius derivative
        MBCAST(Dro, T5) = G.dsTdr * cache.ARRy;

        // Compute derivs with respect to y
        computeDfDyYlmOccultation(Dy, theta);

        // Note that we do not compute limb darkening 
        // derivatives in this case; see the docs.
        MBCAST(Du, T8).setZero();

        // Compute the time deriv
        computeDfDtYlmOccultation(Dt);

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
    MatrixBase<T2> const & Dt,
    MatrixBase<T3> const & Dtheta,
    MatrixBase<T4> const & Dxo,
    MatrixBase<T5> const & Dyo,
    MatrixBase<T6> const & Dro,
    MatrixBase<T7> const & Dy,
    MatrixBase<T8> const & Du
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
        computeDfDyYlmLDNoOccultation(Dy);
        computeDfDuYlmLDNoOccultation(Du);

        // The xo, yo, and ro derivs are trivial
        MBCAST(Dxo, T4).setZero();
        MBCAST(Dyo, T5).setZero();
        MBCAST(Dro, T6).setZero();

        // Compute the theta deriv
        computeDfDthetaYlmLDNoOccultation(Dtheta);

        // Compute the time deriv
        computeDfDtYlmLDNoOccultation(Dt);

    // Occultation
    } else {

        // TODO!
        throw errors::NotImplementedError("computeFluxYlmLD(gradient=true) not yet implemented.");

        // WIP
        
#if 0
        // Transform to the Greens basis and dot the solution in
        //MBCAST(flux, T1) = G.sT * B.A2 * cache.pupy;


        // Compute the solution vector and its gradient
        G.compute(b, ro, true);

        // Transform the solution vector into polynomials
        RowVector<Scalar> cache_sTA2 = G.sT * B.A2; // TODO: Add to cache

        // The normalized occultor position
        Scalar xo_b = xo / b,
               yo_b = yo / b;

        // Align the occultor with the y axis
        // and limb darken it
        if (likely((b > 0) && ((xo != 0) || (yo < 0)))) {
            // Compute the rotation matrix and its derivative
            W.rotateAboutZ(yo_b, xo_b, cache.Ry, cache.RRy);
            
            // W.rightMultiplyR(); // TODO
            // cache.sTA2, cache.sTAR, cache.sTADRDphi); // TODO: Check this
            W.leftMultiplyRz(cache.sTA, cache.sTAR);
            W.leftMultiplyDRz(cache.sTA, cache.sTADRDphi);

            // Apply the limb darkening
            cache.A1Ry = B.A1 * cache.RRy;
            limbDarken(cache.A1Ry, cache.pupy, true);

            // The Green's polynomial of the rotated map
            cache.ARRy = B.A2 * cache.pupy;

            // Compute the contribution to the xo and yo
            // derivs from the occultor rotation matrix
            // TODO Template this and clean it up
            RowVector<Scalar> foo = (cache_sTA2 * cache.DpupyDpy[0]) * B.A1;
            RowVector<Scalar> bar(N);
            W.leftMultiplyDRz(foo, bar);
            FluxType cache_DfDphi = bar * cache.Ry;

            MBCAST(Dxo, T4) = cache_DfDphi * yo_b / b;
            MBCAST(Dyo, T5) = -cache_DfDphi * xo_b / b;
        } else {
            cache.ARRy = B.A * cache.Ry;
            MBCAST(Dxo, T4).setZero();
            MBCAST(Dyo, T5).setZero();
        }   


        // Compute the contribution to the xo and yo
        // derivs from the solution vector
        if (likely(b > 0)) {
            cache.dFdb = G.dsTdb * cache.ARRy;
            MBCAST(Dxo, T4) += cache.dFdb * xo_b;
            MBCAST(Dyo, T5) += cache.dFdb * yo_b;
        }

        // Compute the flux
        MBCAST(flux, T1) = G.sT * cache.ARRy;
    
        // Theta derivative (TODO)
        MBCAST(Dtheta, T3).setZero();
        MBCAST(Dtheta, T3) *= radian;

        // Occultor radius derivative
        MBCAST(Dro, T5) = G.dsTdr * cache.ARRy;

        // Compute derivs with respect to y (TODO)
        computeDfDyYlmOccultation(Dy, theta);

        // Compute derivs with respect to u (TODO)
        MBCAST(Du, T8).setZero();

        // Compute the time deriv (TODO)
        computeDfDtYlmOccultation(Dt);

#endif

    }

}