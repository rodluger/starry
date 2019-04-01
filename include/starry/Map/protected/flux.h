/**
Compute the linear spherical harmonic model for default / spectral maps
in emitted light. Internal method.

*/
template <
    typename U=S, 
    typename=IsEmitted<U>
>
inline void computeLinearFluxModelInternal (
    const Vector<Scalar>& theta, 
    const Vector<Scalar>& xo, 
    const Vector<Scalar>& yo, 
    const Vector<Scalar>& zo, 
    const Vector<Scalar>& ro,
    RowMatrix<Scalar>& X
) {

    // Shape checks
    size_t nt = theta.rows();
    CHECK_SHAPE(xo, nt, 1);
    CHECK_SHAPE(yo, nt, 1);
    CHECK_SHAPE(zo, nt, 1);
    CHECK_SHAPE(ro, nt, 1);
    if (std::is_same<S, Temporal<Scalar, S::Reflected>>::value) {
        CHECK_ROWS(taylor, nt);
    }

    // Convert to radians
    Vector<Scalar> theta_rad = theta * radian;

    // \todo: Cache these
    RowVector<Scalar> rTA1RZetaInv(Ny);
    RowVector<Scalar> rTA1RZetaInvRz(Ny);
    RowVector<Scalar> sTA(Ny);
    RowVector<Scalar> sTARz(Ny);
    RowVector<Scalar> sTARzRZetaInv(Ny);
    RowVector<Scalar> sTARzRZetaInvRz(Ny);
    Matrix<Scalar> LA1;
    Eigen::SparseMatrix<Scalar> A2LA1;
    RowVector<Scalar> rTLA1;

    // Pre-compute the limb darkening operator (\todo: cache it)
    if (udeg > 0) {
        UType tmp = B.U1 * u;
        Vector<Scalar> pu = tmp * pi<Scalar>() * 
            (B.rT.segment(0, (udeg + 1) * (udeg + 1)) * tmp).cwiseInverse();
        Matrix<Scalar> L;
        Vector<Matrix<Scalar>> dLdp; // not used
        computePolynomialProductMatrix<false>(udeg, pu, L, dLdp);
        LA1 = (L * B.A1);
        A2LA1 = (B.A2 * LA1).sparseView();
        rTLA1 = B.rT * LA1;
    }
    Eigen::SparseMatrix<Scalar>& A = (udeg > 0) ? A2LA1 : B.A;
    RowVector<Scalar>& rTA1 = (udeg > 0) ? rTLA1 : B.rTA1;

    // Pre-compute the rotation
    W.leftMultiplyRZetaInv(rTA1, rTA1RZetaInv);

    // Our model matrix, f = X . y
    X.resize(nt, Ny * Nt);

    // Loop over the timeseries and compute the model
    Scalar b, invb;
    Scalar theta_cache = NAN,
           theta_occ_cache = NAN;
    for (size_t n = 0; n < nt; ++n) {

        // Impact parameter
        b = sqrt(xo(n) * xo(n) + yo(n) * yo(n));

        // Complete occultation?
        if (b <= ro(n) - 1) {

            X.row(n).setZero();

        // No occultation
        } else if ((zo(n) < 0) || (b >= 1 + ro(n)) || (ro(n) <= 0.0)) {

            // Rotate the map
            if (theta_rad(n) != theta_cache) {
                theta_cache = theta_rad(n);
                W.compute(cos(theta_rad(n)), sin(theta_rad(n)));
                W.leftMultiplyRz(rTA1RZetaInv, rTA1RZetaInvRz);
                W.leftMultiplyRZeta(rTA1RZetaInvRz, X.block(n, 0, 1, Ny));
            }

            // Apply the Taylor expansion
            if (std::is_same<S, Temporal<Scalar, S::Reflected>>::value) {
                for (int i = 1; i < Nt; ++i) {
                    X.block(n, i * Ny, 1, Ny) = 
                        X.block(n, 0, 1, Ny) * taylor(n, i);
                }
            }

        // Occultation
        } else {

            // Compute the solution vector
            G.compute(b, ro(n));

            // Compute the occultor rotation matrix Rz
            if (likely(b != 0)) {
                invb = Scalar(1.0) / b;
                W.compute(yo(n) * invb, xo(n) * invb);
            } else {
                W.compute(1.0, 0.0);
            }

            // Dot stuff in
            sTA = G.sT * A;
            W.leftMultiplyRz(sTA, sTARz);
            W.leftMultiplyRZetaInv(sTARz, sTARzRZetaInv);

            // Rotate the map
            if (theta_rad(n) != theta_occ_cache) {
                theta_occ_cache = theta_rad(n);
                W.compute(cos(theta_rad(n)), sin(theta_rad(n)));
                W.leftMultiplyRz(sTARzRZetaInv, sTARzRZetaInvRz);
                W.leftMultiplyRZeta(sTARzRZetaInvRz, X.block(n, 0, 1, Ny));
            }

            // Apply the Taylor expansion
            if (std::is_same<S, Temporal<Scalar, S::Reflected>>::value) {
                for (int i = 1; i < Nt; ++i) {
                    X.block(n, i * Ny, 1, Ny) = 
                        X.block(n, 0, 1, Ny) * taylor(n, i);
                }
            }

        }
    }

}

/**
Compute the linear spherical harmonic model for default / spectral maps
in reflected light. Internal method.

*/
template <
    typename U=S, 
    typename=IsReflected<U>
>
inline void computeLinearFluxModelInternal (
    const Vector<Scalar>& theta, 
    const Vector<Scalar>& xo, 
    const Vector<Scalar>& yo, 
    const Vector<Scalar>& zo, 
    const Vector<Scalar>& ro, 
    const RowMatrix<Scalar>& source,
    RowMatrix<Scalar>& X
) {

    // Shape checks
    size_t nt = theta.rows();
    CHECK_SHAPE(xo, nt, 1);
    CHECK_SHAPE(yo, nt, 1);
    CHECK_SHAPE(zo, nt, 1);
    CHECK_SHAPE(ro, nt, 1);
    CHECK_SHAPE(source, nt, 3);
    if (std::is_same<S, Temporal<Scalar, S::Reflected>>::value) {
        CHECK_ROWS(taylor, nt);
    }

    // Convert to radians
    Vector<Scalar> theta_rad = theta * radian;

    // \todo: Cache these
    RowVector<Scalar> rTA1(Ny);
    RowVector<Scalar> rTA1Rz(Ny);
    RowVector<Scalar> rTA1RzRZetaInv(Ny);
    RowVector<Scalar> rTA1RzRZetaInvRz(Ny);
    Eigen::SparseMatrix<Scalar> LA1;

    // Pre-compute the limb darkening operator (\todo: cache it)
    if (udeg > 0) {
        UType tmp = B.U1 * u;
        Vector<Scalar> pu = tmp * pi<Scalar>() * 
            (B.rT.segment(0, (udeg + 1) * (udeg + 1)) * tmp).cwiseInverse();
        Matrix<Scalar> L;
        Vector<Matrix<Scalar>> dLdp; // not used
        computePolynomialProductMatrix<false>(udeg, pu, L, dLdp);
        LA1 = (L * B.A1).sparseView();
    } else {
        LA1 = B.A1;
    }

    // Our model matrix, f = X . y
    X.resize(nt, Ny * Nt);

    // Loop over the timeseries
    Scalar sx_cache = NAN,
           sy_cache = NAN,
           sz_cache = NAN;
    Scalar norm, cosw, sinw;
    Scalar b;
    for (size_t n = 0; n < nt; ++n) {

        // Impact parameter
        b = sqrt(xo(n) * xo(n) + yo(n) * yo(n));

        // Complete occultation?
        if (b <= ro(n) - 1) {

            X.row(n).setZero();

        // No occultation
        } else if ((zo(n) < 0) || (b >= 1 + ro(n)) || (ro(n) <= 0.0)) {

            // Compute the reflectance integrals
            if ((source(n, 0) != sx_cache) ||
                (source(n, 1) != sy_cache) ||
                (source(n, 2) != sz_cache))  
            {
                
                // The semi-minor axis of the terminator ellipse
                Scalar bterm = -source(n, 2);

                // Compute the phase curve integrals and
                // transform them into the polynomial basis
                if (source(n, 2) != sz_cache) {
                    G.compute(bterm);
                    rTA1 = G.rT * LA1;
                }

                // Rotate into the correct frame on the sky plane
                if (likely(abs(bterm) != 1.0)) {
                    norm = Scalar(1.0) / sqrt(source(n, 0) * source(n, 0) + 
                                              source(n, 1) * source(n, 1));
                    cosw = source(n, 1) * norm;
                    sinw = source(n, 0) * norm;
                } else {
                    cosw = 1.0;
                    sinw = 0.0;
                }
                W.compute(cosw, sinw);
                W.leftMultiplyRz(rTA1, rTA1Rz);

                // Cache the source position
                sx_cache = source(n, 0);
                sy_cache = source(n, 1);
                sz_cache = source(n, 2);

            }

            // Rotate to the correct phase
            W.compute(cos(theta_rad(n)), sin(theta_rad(n)));
            W.leftMultiplyRZetaInv(rTA1Rz, rTA1RzRZetaInv);
            W.leftMultiplyRz(rTA1RzRZetaInv, rTA1RzRZetaInvRz);
            W.leftMultiplyRZeta(rTA1RzRZetaInvRz, X.block(n, 0, 1, Ny));

            // Apply the Taylor expansion
            if (std::is_same<S, Temporal<Scalar, S::Reflected>>::value) {
                for (int i = 1; i < Nt; ++i) {
                    X.block(n, i * Ny, 1, Ny) = 
                        X.block(n, 0, 1, Ny) * taylor(n, i);
                }
            }

        // Occultation
        } else {

            // \todo Implement occultations in reflected light
            throw std::runtime_error (
                "Occultations in reflected light not yet implemented."
            );

        }

    }

}


/**
\todo Compute the linear spherical harmonic model and its gradient. Internal method.

*/
template <
    typename U=S, 
    typename=IsEmitted<U>
>
inline void computeLinearFluxModelInternal (
    const Vector<Scalar>& theta, 
    const Vector<Scalar>& xo, 
    const Vector<Scalar>& yo, 
    const Vector<Scalar>& zo, 
    const Vector<Scalar>& ro, 
    RowMatrix<Scalar>& X,
    RowMatrix<Scalar>& Dt,
    RowMatrix<Scalar>& Dtheta,
    RowMatrix<Scalar>& Dxo,
    RowMatrix<Scalar>& Dyo,
    RowMatrix<Scalar>& Dro,
    RowMatrix<Scalar>& Du,
    RowMatrix<Scalar>& Daxis
) {
    // \todo
    throw std::runtime_error("Gradients not yet implemented.");
}

/**
\todo Compute the linear spherical harmonic model and its gradient. Internal method.

*/
template <
    typename U=S, 
    typename=IsReflected<U>
>
inline void computeLinearFluxModelInternal (
    const Vector<Scalar>& theta, 
    const Vector<Scalar>& xo, 
    const Vector<Scalar>& yo, 
    const Vector<Scalar>& zo, 
    const Vector<Scalar>& ro, 
    const RowMatrix<Scalar>& source,
    RowMatrix<Scalar>& X,
    RowMatrix<Scalar>& Dt,
    RowMatrix<Scalar>& Dtheta,
    RowMatrix<Scalar>& Dxo,
    RowMatrix<Scalar>& Dyo,
    RowMatrix<Scalar>& Dro,
    RowMatrix<Scalar>& Dsource,
    RowMatrix<Scalar>& Du,
    RowMatrix<Scalar>& Daxis
) {
    // \todo
    throw std::runtime_error("Gradients not yet implemented.");
}

/* \todo

    // Shape checks
    size_t nt = theta.rows();
    CHECK_SHAPE(xo, nt, 1);
    CHECK_SHAPE(yo, nt, 1);
    CHECK_SHAPE(zo, nt, 1);
    CHECK_SHAPE(ro, nt, 1);

    // Convert to radians
    Vector<Scalar> theta_rad = theta * radian;



    // \todo: Cache these
    RowVector<Scalar> rTA1RZetaInv(Ny);
    RowVector<Scalar> rTA1RZetaInvRz(Ny);
    RowVector<Scalar> rTA1RZetaInvDRzDtheta(Ny);
    RowVector<Scalar> sTA(Ny);
    RowVector<Scalar> dsTdrA(Ny);
    RowVector<Scalar> dsTdbA(Ny);
    RowVector<Scalar> sTARz(Ny);
    RowVector<Scalar> dsTdrARz(Ny);
    RowVector<Scalar> dsTdbARz(Ny);
    RowVector<Scalar> sTADRzDw(Ny);
    RowVector<Scalar> sTARzRZetaInv(Ny);
    RowVector<Scalar> dsTdrARzRZetaInv(Ny);
    RowVector<Scalar> dsTdbARzRZetaInv(Ny);
    RowVector<Scalar> sTADRzDwRZetaInv(Ny);
    RowVector<Scalar> sTARzRZetaInvRz(Ny);
    RowVector<Scalar> sTARzRZetaInvDRzDtheta(Ny);
    RowVector<Scalar> dsTdrARzRZetaInvRz(Ny);
    RowVector<Scalar> dsTdbARzRZetaInvRz(Ny);
    RowVector<Scalar> sTADRzDwRZetaInvRz(Ny);
    RowVector<Scalar> sTADRzDwRZetaInvRzRZeta(Ny);

    // Pre-compute the limb darkening matrix
    UType tmp = B.U1 * u;
    Scalar norm = pi<Scalar>() / B.rT.dot(tmp);
    Vector<Scalar> pu = tmp * norm;
    Matrix<Scalar> L;
    Vector<Matrix<Scalar>> dLdp;
    computePolynomialProductMatrix<true>(udeg, pu, L, dLdp);
    Matrix<Scalar> LA1 = (L * B.A1).block(0, 0, Ny, Ny);
    Matrix<Scalar> A2LA1 = B.A2 * LA1;

    // Limb darkening derivatives (TODO)
    //Matrix<Scalar> dLdu1 = (dLdp[0] * B.U);
    //RowMatrix<Scalar> Du1;
    //Du1.resize(nt, Ny);

    // Pre-compute the W matrices
    if (udeg == 0)
        W.leftMultiplyRZetaInv(B.rTA1, rTA1RZetaInv);
    else
        W.leftMultiplyRZetaInv((B.rT * LA1).eval(), rTA1RZetaInv);

    // Our model matrix, f = A . y, and its derivatives
    X.resize(nt, Ny);
    Dtheta.resize(nt, Ny);
    Dxo.resize(nt, Ny);
    Dyo.resize(nt, Ny);
    Dro.resize(nt, Ny);

    // 
    if (std::is_same<S, Temporal<Scalar, S::Reflected>>::value)
        Dt.resize(nt, Ny);

    // Loop over the timeseries and compute the model
    Scalar b, invb;
    for (size_t n = 0; n < nt; ++n) {

        // Impact parameter
        b = sqrt(xo(n) * xo(n) + yo(n) * yo(n));

        // Complete occultation?
        if (b <= ro(n) - 1) {

            X.row(n).setZero();
            Dtheta.row(n).setZero();
            Dxo.row(n).setZero();
            Dyo.row(n).setZero();
            Dro.row(n).setZero();

        // No occultation
        } else if ((zo(n) < 0) || (b >= 1 + ro(n)) || (ro(n) <= 0.0)) {

            // Compute the Rz rotation matrix
            W.compute(cos(theta_rad(n)), sin(theta_rad(n)));

            // Dot it in
            W.leftMultiplyRz(rTA1RZetaInv, rTA1RZetaInvRz);

            // Transform back to the sky plane
            W.leftMultiplyRZeta(rTA1RZetaInvRz, X.block(n, 0, 1, Ny));

            // Theta deriv
            W.leftMultiplyDRz(rTA1RZetaInv, rTA1RZetaInvDRzDtheta);
            W.leftMultiplyRZeta(rTA1RZetaInvDRzDtheta, Dtheta.block(n, 0, 1, Ny));
            Dtheta.block(n, 0, 1, Ny) *= radian;

            // Apply the Taylor expansion?
            if (std::is_same<S, Temporal<Scalar, S::Reflected>>::value) {
                Dt.block(n, 0, 1, Ny).setZero();
                for (int i = 1; i < ncoly; ++i) {
                    X.block(n, i * Ny, 1, Ny) = X.block(n, 0, 1, Ny) * taylor(n, i);
                    Dt.block(n, i * Ny, 1, Ny) = X.block(n, 0, 1, Ny) * taylor(n, i - 1);
                    Dtheta.block(n, i * Ny, 1, Ny) = Dtheta.block(n, 0, 1, Ny) * taylor(n, i);
                } 
            }

            // Occultor derivs are zero
            Dxo.row(n).setZero();
            Dyo.row(n).setZero();
            Dro.row(n).setZero();
            
        // Occultation
        } else {

            // Compute the solution vector
            G.template compute<true>(b, ro(n));
            if (udeg == 0) {
                sTA = G.sT * B.A;
                dsTdrA = G.dsTdr * B.A;
                dsTdbA = G.dsTdb * B.A;
            } else {
                sTA = G.sT * A2LA1;
                dsTdrA = G.dsTdr * A2LA1;
                dsTdbA = G.dsTdb * A2LA1;
            }
            invb = Scalar(1.0) / b;

            // Compute the occultor rotation matrix Rz
            W.compute(yo(n) * invb, xo(n) * invb);

            // Dot stuff in
            W.leftMultiplyRz(sTA, sTARz);
            W.leftMultiplyRZetaInv(sTARz, sTARzRZetaInv);
            W.leftMultiplyRz(dsTdrA, dsTdrARz);
            W.leftMultiplyRZetaInv(dsTdrARz, dsTdrARzRZetaInv);
            W.leftMultiplyRz(dsTdbA, dsTdbARz);
            W.leftMultiplyRZetaInv(dsTdbARz, dsTdbARzRZetaInv);
            W.leftMultiplyDRz(sTA, sTADRzDw);
            W.leftMultiplyRZetaInv(sTADRzDw, sTADRzDwRZetaInv);

            // DEBUG
            //RowVector<Scalar> foo(Ny), bar(Ny);
            //foo = G.sT * B.A2 * dLdu1 * B.A1;
            //W.leftMultiplyRz(foo, bar);
            //W.leftMultiplyRZetaInv(bar, foo);

            // Compute the Rz rotation matrix
            W.compute(cos(theta_rad(n)), sin(theta_rad(n)));

            // DEBUG
            //W.leftMultiplyRz(foo, bar);
            //W.leftMultiplyRZeta(bar, Du1.block(n, 0, 1, Ny));
            //std::cout << Du1.block(n, 0, 1, Ny) * y << std::endl;

            // Dot it in
            W.leftMultiplyRz(sTARzRZetaInv, sTARzRZetaInvRz);

            // Transform back to the sky plane
            W.leftMultiplyRZeta(sTARzRZetaInvRz, X.block(n, 0, 1, Ny));

            // Theta deriv
            W.leftMultiplyDRz(sTARzRZetaInv, sTARzRZetaInvDRzDtheta);
            W.leftMultiplyRZeta(sTARzRZetaInvDRzDtheta, Dtheta.block(n, 0, 1, Ny));
            Dtheta.block(n, 0, 1, Ny) *= radian;

            // Radius deriv
            W.leftMultiplyRz(dsTdrARzRZetaInv, dsTdrARzRZetaInvRz);
            W.leftMultiplyRZeta(dsTdrARzRZetaInvRz, Dro.block(n, 0, 1, Ny));

            // xo and yo derivatives
            W.leftMultiplyRz(dsTdbARzRZetaInv, dsTdbARzRZetaInvRz);
            W.leftMultiplyRZeta(dsTdbARzRZetaInvRz, Dxo.block(n, 0, 1, Ny));
            Dyo.block(n, 0, 1, Ny) = Dxo.block(n, 0, 1, Ny);
            Dxo.block(n, 0, 1, Ny) *= xo(n) * invb;
            Dyo.block(n, 0, 1, Ny) *= yo(n) * invb;
            W.leftMultiplyRz(sTADRzDwRZetaInv, sTADRzDwRZetaInvRz);
            W.leftMultiplyRZeta(sTADRzDwRZetaInvRz, sTADRzDwRZetaInvRzRZeta);
            sTADRzDwRZetaInvRzRZeta *= invb * invb;
            Dxo.block(n, 0, 1, Ny) += yo(n) * sTADRzDwRZetaInvRzRZeta;
            Dyo.block(n, 0, 1, Ny) -= xo(n) * sTADRzDwRZetaInvRzRZeta;

            // Apply the Taylor expansion?
            if (std::is_same<S, Temporal<Scalar, S::Reflected>>::value) {
                Dt.block(n, 0, 1, Ny).setZero();
                for (int i = 1; i < ncoly; ++i) {
                    X.block(n, i * Ny, 1, Ny) = X.block(n, 0, 1, Ny) * taylor(n, i);
                    Dt.block(n, i * Ny, 1, Ny) = X.block(n, 0, 1, Ny) * taylor(n, i - 1);
                    Dtheta.block(n, i * Ny, 1, Ny) = Dtheta.block(n, 0, 1, Ny) * taylor(n, i);
                    Dxo.block(n, i * Ny, 1, Ny) = Dxo.block(n, 0, 1, Ny) * taylor(n, i);
                    Dyo.block(n, i * Ny, 1, Ny) = Dyo.block(n, 0, 1, Ny) * taylor(n, i);
                    Dro.block(n, i * Ny, 1, Ny) = Dro.block(n, 0, 1, Ny) * taylor(n, i);
                } 
            }

        }
    }
    
    */