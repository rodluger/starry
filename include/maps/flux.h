// --------------------------
// ------- Intensity --------
// --------------------------


/**
Evaluate the map at a given (theta, x, y) coordinate.

*/
template <class S>
inline void Map<S>::computeIntensity_(
    const Scalar& t,
    const Scalar& theta,
    const Scalar& x_,
    const Scalar& y_,
    Ref<FluxType> intensity
) {
    // Check if outside the sphere
    if (x_ * x_ + y_ * y_ > 1.0) {
        intensity *= NAN;
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
    intensity = contract(result, t);
}

/**
Render the visible map on a square cartesian grid at given
resolution. 

*/
template <class S>
template <typename Derived>
inline void Map<S>::renderMap_(
    const Scalar& t,
    const Scalar& theta,
    int res,
    MatrixBase<Derived> const& intensity
) {
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

    // NOTE: The following line is the hack recommended here:
    // https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html.
    // In general, the `intensity` matrix to which we are assigning will either
    // be an Eigen Matrix or a **block** of an Eigen Matrix. It is notoriously
    // difficult to pass a BlockXpr by reference to a function and still allow
    // it to act as an l-value. The recommended hack is to declare it as const
    // and cast the `const` qualifier away:
    const_cast<MatrixBase<Derived>&>(intensity) = contract(result, t);
}

// --------------------------
// ---------- Flux ----------
// --------------------------


/**

*/
template <class S>
inline void Map<S>::computeFlux_(
    const Scalar& t,
    const Scalar& theta, 
    const Scalar& xo, 
    const Scalar& yo, 
    const Scalar& ro, 
    Ref<FluxType> flux
) {
    // Figure out the degree of the map
    computeDegreeU();
    computeDegreeY();

    // Impact parameter
    Scalar b = sqrt(xo * xo + yo * yo);

    // Check for complete occultation
    if (b <= ro - 1) {
        flux.setZero();
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
inline void Map<S>::computeFluxLD(
    const Scalar& t,
    const Scalar& b, 
    const Scalar& ro, 
    Ref<FluxType> flux
) {
    // No occultation
    if ((b >= 1 + ro) || (ro == 0)) {

        // Easy: the disk-integrated intensity
        // is just the Y_{0,0} coefficient
        flux = contract(y.row(0), t);

    // Occultation
    } else {

        // Compute the Agol `S` vector
        L.compute(b, ro);

        // Compute the Agol `c` basis
        computeC();
       
        // Dot the integral solution in, and we're done!
        flux = contract(L.s * cache.c, t);

    }
}

template <class S>
inline void Map<S>::computeFluxYlm(
    const Scalar& t,
    const Scalar& theta,
    const Scalar& xo,
    const Scalar& yo,  
    const Scalar& b, 
    const Scalar& ro, 
    Ref<FluxType> flux
) {
    // Rotate the map into view
    rotateIntoCache(theta);

    // No occultation
    if ((b >= 1 + ro) || (ro == 0)) {

        // Easy
        flux = contract(B.rTA1 * cache.Ry, t);

    // Occultation
    } else {

        // TODO!
        throw errors::NotImplementedError("Not yet implemented.");

    }
}

template <class S>
inline void Map<S>::computeFluxYlmLD(
    const Scalar& t,
    const Scalar& theta,
    const Scalar& xo,
    const Scalar& yo,  
    const Scalar& b, 
    const Scalar& ro, 
    Ref<FluxType> flux
) {
    // Rotate the map into view
    rotateIntoCache(theta);

    // No occultation
    if ((b >= 1 + ro) || (ro == 0)) {

        // Change basis to polynomials
        cache.A1Ry = B.A1 * cache.Ry;

        // Apply limb darkening
        limbDarken(cache.A1Ry, cache.p_uy);
        flux = contract(B.rT * cache.p_uy, t);

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
inline void Map<S>::computeFlux_(
    const Scalar& t,
    const Scalar& theta, 
    const Scalar& xo, 
    const Scalar& yo, 
    const Scalar& ro, 
    Ref<FluxType> flux,
    Ref<GradType> gradient
) {
    // Figure out the degree of the map
    computeDegreeU();
    computeDegreeY();

    // Impact parameter
    Scalar b = sqrt(xo * xo + yo * yo);

    // Check for complete occultation
    if (b <= ro - 1) {
        flux.setZero();
        gradient.setZero();
        return;
    }

    // Compute the flux
    if (y_deg == 0) {
        computeFluxLD(t, xo, yo, b, ro, flux, gradient);
    } else if (u_deg == 0) {
        computeFluxYlm(t, theta, xo, yo, b, ro, flux, gradient);
    } else {
        computeFluxYlmLD(t, theta, xo, yo, b, ro, flux, gradient);
    }
}

template <class S>
inline void Map<S>::computeFluxLD(
    const Scalar& t,
    const Scalar& xo, 
    const Scalar& yo, 
    const Scalar& b, 
    const Scalar& ro, 
    Ref<FluxType> flux, 
    Ref<GradType> gradient
) {

    // No occultation
    if ((b >= 1 + ro) || (ro == 0)) {

        // Most of the derivs are zero
        gradient.setZero();

        // The Y_{0,0} deriv is unity
        gradient.row(idx.y).setConstant(1.0);

        // The other Ylm derivs are not necessarily zero, but we
        // explicitly don't compute them for purely limb-darkened
        // maps to ensure computational efficiency. See the docs
        // for more information.

        // The time derivative of the flux
        gradient.row(idx.t) = contract_deriv(y.row(0), t);

        // The flux is easy: the disk-integrated intensity
        // is just the Y_{0,0} coefficient
        flux = contract(y.row(0), t);

    // Occultation
    } else {

        // The theta deriv is always zero
        gradient.row(idx.theta).setConstant(0.0);

        // Compute the Agol `S` vector and its derivs
        L.compute(b, ro, true);

        // Compute the Agol `c` basis
        computeC();

        // Compute the flux
        auto flux_ = L.s * cache.c;

        // dF / db  ->  dF / dx, dF / dy
        FluxType dFdb_b(contract(L.dsdb * cache.c, t));
        dFdb_b /= b;
        gradient.row(idx.xo) = dFdb_b * xo;
        gradient.row(idx.yo) = dFdb_b * yo;

        // dF / dr
        gradient.row(idx.ro) = contract(L.dsdr * cache.c, t);

        // Derivs with respect to the Ylms (see note above)
        gradient.row(idx.y) = contract(flux_.cwiseQuotient(y.row(0)), t);

        // The flux and its time derivative
        gradient.row(idx.t) = contract_deriv(flux_, t);
        flux = contract(flux_, t);

        // Derivs with respect to the limb darkening coeffs from dF / dc
        computeDfDu(flux, gradient);

    }
}

template <class S>
inline void Map<S>::computeFluxYlm (
    const Scalar& t,
    const Scalar& theta,
    const Scalar& xo,
    const Scalar& yo,  
    const Scalar& b, 
    const Scalar& ro, 
    Ref<FluxType> flux,
    Ref<GradType> gradient
) {

    // Rotate the map into view and explicitly
    // compute the Wigner matrices
    rotateIntoCache(theta, true);

    // No occultation
    if ((b >= 1 + ro) || (ro == 0)) {

        // Compute the theta deriv
        gradient.row(idx.theta) = contract(B.rTA1 * cache.dRdthetay, t);
        gradient.row(idx.theta) *= radian;

        // The xo, yo, and ro derivs are trivial
        gradient.block(idx.xo, 0, 3, ncol).setZero();

        // Compute the Ylm derivs
        if (theta == 0) {
            gradient.block(idx.y, 0, idx.ny, ncol) = 
                B.rTA1.transpose().replicate(1, ncol);
        } else {
            for (int l = 0; l < lmax + 1; ++l)
                gradient.block(idx.y + l * l, 0, 2 * l + 1, ncol) = 
                    (B.rTA1.segment(l * l, 2 * l + 1) * W.R[l])
                    .transpose().replicate(1, ncol);
        }

        // Note that we do not compute limb darkening 
        // derivatives in this case; see the docs.

        // The flux and its time derivative
        auto flux_ = B.rTA1 * cache.Ry;
        gradient.row(idx.t) = contract_deriv(flux_, t);
        flux = contract(flux_, t);

    // Occultation
    } else {

        // TODO!
        throw errors::NotImplementedError("Not yet implemented.");

    }

}

template <class S>
inline void Map<S>::computeFluxYlmLD(
    const Scalar& t,
    const Scalar& theta,
    const Scalar& xo,
    const Scalar& yo,  
    const Scalar& b, 
    const Scalar& ro, 
    Ref<FluxType> flux,
    Ref<GradType> gradient
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

