// --------------------------
// ------- Intensity --------
// --------------------------


/**
Evaluate the map at a given (theta, x, y) coordinate.

*/
template <class S>
template <typename T1>
inline void Map<S>::computeIntensity_(
    const Scalar& t,
    const Scalar& theta,
    const Scalar& x_,
    const Scalar& y_,
    MatrixBase<T1> const & intensity
) {
    // Shape checks
    check_shape(intensity, 1, nflx);

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
    MBCAST(intensity, T1) = contract(result, t);
}

/**
Render the visible map on a square cartesian grid at given
resolution. 

*/
template <class S>
template <typename T1>
inline void Map<S>::renderMap_(
    const Scalar& t,
    const Scalar& theta,
    int res,
    MatrixBase<T1> const & intensity
) {
    // Shape checks
    check_shape(intensity, res * res, nflx);

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
    MBCAST(intensity, T1) = contract(result, t);
}

// --------------------------
// ---------- Flux ----------
// --------------------------


/**

*/
template <class S>
template <typename T1>
inline void Map<S>::computeFlux_(
    const Scalar& t,
    const Scalar& theta, 
    const Scalar& xo, 
    const Scalar& yo, 
    const Scalar& ro, 
    MatrixBase<T1> const & flux
) {
    // Shape checks
    check_shape(flux, 1, nflx);

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
        computeFluxLD(t, b, ro, flux);
    } else if (u_deg == 0) {
        computeFluxYlm(t, theta, xo, yo, b, ro, flux);
    } else {
        computeFluxYlmLD(t, theta, xo, yo, b, ro, flux);
    }
}

template <class S>
template <typename T1>
inline void Map<S>::computeFluxLD(
    const Scalar& t,
    const Scalar& b, 
    const Scalar& ro, 
    MatrixBase<T1> const & flux
) {
    // No occultation
    if ((b >= 1 + ro) || (ro == 0)) {

        // Easy: the disk-integrated intensity
        // is just the Y_{0,0} coefficient
        MBCAST(flux, T1) = contract(y.row(0), t);

    // Occultation
    } else {

        // Compute the Agol `S` vector
        L.compute(b, ro);

        // Compute the Agol `c` basis
        computeC();

        // Normalize by y00
        UCoeffType norm = contract(y.row(0), t);

        // Dot the integral solution in, and we're done!
        MBCAST(flux, T1) = (L.s * cache.c).cwiseProduct(norm);

    }
}

template <class S>
template <typename T1>
inline void Map<S>::computeFluxYlm(
    const Scalar& t,
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
    if ((b >= 1 + ro) || (ro == 0)) {

        // Easy
        MBCAST(flux, T1) = contract(B.rTA1 * cache.Ry, t);

    // Occultation
    } else {

        // TODO!
        throw errors::NotImplementedError("Not yet implemented.");

    }
}

template <class S>
template <typename T1>
inline void Map<S>::computeFluxYlmLD(
    const Scalar& t,
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
    if ((b >= 1 + ro) || (ro == 0)) {

        // Change basis to polynomials
        cache.A1Ry = B.A1 * cache.Ry;

        // Apply limb darkening
        limbDarken(cache.A1Ry, cache.p_uy);
        MBCAST(flux, T1) = contract(B.rT * cache.p_uy, t);

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
    const Scalar& t,
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
    check_shape(flux, 1, nflx);
    check_shape(dt, 1, nflx);
    check_shape(dtheta, 1, nflx);
    check_shape(dxo, 1, nflx);
    check_shape(dyo, 1, nflx);
    check_shape(dro, 1, nflx);
    check_cols(dy, ncol);
    check_cols(du, nflx);

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
#ifdef STARRY_KEEP_DFDU_AS_DFDG
        check_rows(du, lmax + 1);
#else
        check_rows(du, lmax);
#endif
        computeFluxLD(t, xo, yo, b, ro, flux, dt, 
                      dtheta, dxo, dyo, dro, dy, du);
    } else if (u_deg == 0) {
        check_rows(dy, N);
        computeFluxYlm(t, theta, xo, yo, b, ro, flux, dt, 
                       dtheta, dxo, dyo, dro, dy, du);
    } else {
        check_rows(dy, N);
#ifdef STARRY_KEEP_DFDU_AS_DFDG
        check_rows(du, lmax + 1);
#else
        check_rows(du, lmax);
#endif
        computeFluxYlmLD(t, theta, xo, yo, b, ro, flux, dt, 
                         dtheta, dxo, dyo, dro, dy, du);
    }
}

template <class S>
template <typename T1, typename T2, typename T3, typename T4, 
          typename T5, typename T6, typename T7, typename T8>
inline void Map<S>::computeFluxLD(
    const Scalar& t,
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
    if ((b >= 1 + ro) || (ro == 0)) {

        // Most of the derivs are zero
        MBCAST(dtheta, T3).setZero();
        MBCAST(dxo, T4).setZero();
        MBCAST(dyo, T5).setZero();
        MBCAST(dro, T6).setZero();
        MBCAST(du, T8).setZero();

        // The Y_{0,0} deriv is unity for static maps and equal to
        // {1, t, 1/2 t^2 ...} for temporal maps.
        // The other Ylm derivs are not necessarily zero, but we
        // explicitly don't compute them for purely limb-darkened
        // maps to ensure computational efficiency. See the docs
        // for more information.
        MBCAST(dy, T7).setZero();
        UCoeffType flux0(ncol);
        flux0.setOnes();
        MBCAST(dy, T7).row(0) = dfdy0(flux0, t);

        // dF / dt
        if (std::is_same<S, Temporal<Scalar>>::value) {
            MBCAST(dt, T2) = contract_deriv(y.row(0), t);
        }
        
        // The flux is easy: the disk-integrated intensity
        // is just the Y_{0,0} coefficient
        MBCAST(flux, T1) = contract(y.row(0), t);

    // Occultation
    } else {

        // Compute the Agol `s` vector and its derivs
        L.compute(b, ro, true);

        // Compute the Agol `c` basis
        computeC();

        // Compute the normalization
        UCoeffType norm = contract(y.row(0), t);

        // Compute the flux
        UCoeffType flux0 = (L.s * cache.c);
        MBCAST(flux, T1) = flux0.cwiseProduct(norm);

        // The theta deriv is always zero
        MBCAST(dtheta, T3).setConstant(0.0);

        // dF / db  ->  dF / dx, dF / dy
        if (likely(b > 0)) {
            MBCAST(dxo, T4) = (L.dsdb * cache.c);
            MBCAST(dxo, T4) /= b;
            MBCAST(dyo, T5) = dxo;
            MBCAST(dxo, T4) *= xo;
            MBCAST(dyo, T5) *= yo;
            MBCAST(dxo, T4) = dxo.cwiseProduct(norm);
            MBCAST(dyo, T5) = dyo.cwiseProduct(norm);
        } else {
            MBCAST(dxo, T4) = L.dsdb * cache.c;
            MBCAST(dxo, T4) = dxo.cwiseProduct(norm);
            MBCAST(dyo, T5) = dxo;
        }

        // dF / dr
        MBCAST(dro, T6) = (L.dsdr * cache.c);

        // Derivs with respect to the Ylms (see note above)
        MBCAST(dy, T7).setZero();
        MBCAST(dy, T7).row(0) = dfdy0(flux0, t);

        // dF / dt
        // TODO: Template this?
        if (std::is_same<S, Temporal<Scalar>>::value) {
            UCoeffType norm_deriv = contract_deriv(y.row(0), t);
            MBCAST(dt, T2) = flux0.cwiseProduct(norm_deriv);
        }

        // Derivs with respect to the limb darkening coeffs from dF / dc
        computeDfDu(flux0, du, norm);

    }

}

template <class S>
template <typename T1, typename T2, typename T3, typename T4, 
          typename T5, typename T6, typename T7, typename T8>
inline void Map<S>::computeFluxYlm (
    const Scalar& t,
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
    if ((b >= 1 + ro) || (ro == 0)) {

        // Compute the theta deriv
        MBCAST(dtheta, T3) = contract(B.rTA1 * cache.dRdthetay, t);
        MBCAST(dtheta, T3) *= radian;

        // The xo, yo, and ro derivs are trivial
        MBCAST(dxo, T4).setZero();
        MBCAST(dyo, T5).setZero();
        MBCAST(dro, T6).setZero();

        // Compute the Ylm derivs
        if (theta == 0) {
            MBCAST(dy, T7) = B.rTA1.transpose().replicate(1, ncol);
        } else {
            for (int l = 0; l < lmax + 1; ++l)
                MBCAST(dy, T7).block(l * l, 0, 2 * l + 1, ncol) = 
                    (B.rTA1.segment(l * l, 2 * l + 1) * W.R[l])
                    .transpose().replicate(1, ncol);
        }

        // Note that we do not compute limb darkening 
        // derivatives in this case; see the docs.
        MBCAST(du, T8).setZero();

        // The flux and its time derivative
        auto flux_ = B.rTA1 * cache.Ry;
        MBCAST(dt, T2) = contract_deriv(flux_, t);
        MBCAST(flux, T1) = contract(flux_, t);

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
    const Scalar& t,
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
    if ((b >= 1 + ro) || (ro == 0)) {

        // TODO!
        // NOTE: Recall that with our new normalization,
        // limb darkening CAN affect the total flux!
        throw errors::NotImplementedError("Not yet implemented.");

    // Occultation
    } else {

        // TODO!
        throw errors::NotImplementedError("Not yet implemented.");

    }

}

