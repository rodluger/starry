/**
Compute the linear spherical harmonic model for default / spectral maps
in emitted light. Internal method.

*/
template <typename U=S>
inline EnableIf<!U::Reflected && !U::LimbDarkened, void> computeLinearFluxModelInternal (
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
    if (S::Temporal) {
        CHECK_ROWS(taylor, nt);
    }

    // Convert to radians
    Vector<Scalar> theta_rad = theta * radian;

    // Pre-compute the limb darkening / filter operator
    if ((udeg > 0) || (fdeg > 0)) {
        
        // Compute the two polynomials
        Vector<Scalar> tmp = B.U1 * u;
        Scalar norm = Scalar(1.0) / B.rT.segment(0, (udeg + 1) * (udeg + 1)).dot(tmp);
        Vector<Scalar> pu = tmp * norm * pi<Scalar>();
        Vector<Scalar> pf = B.A1.block(0, 0, Nf, Nf) * f;
        
        // Multiply them
        Vector<Scalar> p;
        Matrix<Scalar> dpdpu; // not used
        Matrix<Scalar> dpdpf; // not used
        if (fdeg == 0)
            p = pu;
        else if (udeg == 0)
            p = pf;
        else if (udeg > fdeg) 
            computePolynomialProduct<false>(udeg, pu, fdeg, pf, p, dpdpu, dpdpf);
        else
            computePolynomialProduct<false>(fdeg, pf, udeg, pu, p, dpdpf, dpdpu);

        // Compute the operator
        Matrix<Scalar> L;
        Vector<Matrix<Scalar>> dLdp; // not used
        computePolynomialProductMatrix<false>(udeg + fdeg, p, L, dLdp);
        LA1 = (L * B.A1.block(0, 0, Ny, Ny));
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
            } else {
                 X.block(n, 0, 1, Ny) =  X.block(n - 1, 0, 1, Ny);
            }

            // Apply the Taylor expansion
            if (S::Temporal) {
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
            }
            W.leftMultiplyRz(sTARzRZetaInv, sTARzRZetaInvRz);
            W.leftMultiplyRZeta(sTARzRZetaInvRz, X.block(n, 0, 1, Ny));

            // Apply the Taylor expansion
            if (S::Temporal) {
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
template <typename U=S>
inline EnableIf<U::Reflected && !U::LimbDarkened, void> computeLinearFluxModelInternal (
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
    if (S::Temporal) {
        CHECK_ROWS(taylor, nt);
    }

    // Convert to radians
    Vector<Scalar> theta_rad = theta * radian;

    // Pre-compute the limb darkening / filter operator
    if ((udeg > 0) || (fdeg > 0)) {

        // Compute the two polynomials
        Vector<Scalar> tmp = B.U1 * u;
        Scalar norm = Scalar(1.0) / B.rT.segment(0, (udeg + 1) * (udeg + 1)).dot(tmp);
        Vector<Scalar> pu = tmp * norm * pi<Scalar>();
        Vector<Scalar> pf = B.A1.block(0, 0, Nf, Nf) * f;
        
        // Multiply them
        Vector<Scalar> p;
        Matrix<Scalar> dpdpu; // not used
        Matrix<Scalar> dpdpf; // not used
        if (fdeg == 0)
            p = pu;
        else if (udeg == 0)
            p = pf;
        else if (udeg > fdeg) 
            computePolynomialProduct<false>(udeg, pu, fdeg, pf, p, dpdpu, dpdpf);
        else
            computePolynomialProduct<false>(fdeg, pf, udeg, pu, p, dpdpf, dpdpu);

        // Compute the operator
        Matrix<Scalar> L;
        Vector<Matrix<Scalar>> dLdp; // not used
        computePolynomialProductMatrix<false>(udeg + fdeg, p, L, dLdp);
        LA1_ = (L * B.A1.block(0, 0, Ny, Ny)).sparseView();
    }
    Eigen::SparseMatrix<Scalar>& LA1 = (udeg > 0) ? LA1_ : B.A1;

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
            if (S::Temporal) {
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
Compute the linear spherical harmonic model and its gradient. Internal method.

*/
template <typename U=S>
inline EnableIf<!U::Reflected && !U::LimbDarkened, void> computeLinearFluxModelInternal (
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
    RowMatrix<Scalar>& Dinc,
    RowMatrix<Scalar>& Dobl
) {

    // Shape checks
    size_t nt = theta.rows();
    CHECK_SHAPE(xo, nt, 1);
    CHECK_SHAPE(yo, nt, 1);
    CHECK_SHAPE(zo, nt, 1);
    CHECK_SHAPE(ro, nt, 1);
    if (S::Temporal) {
        CHECK_ROWS(taylor, nt);
    }

    // Convert to radians
    Vector<Scalar> theta_rad = theta * radian;

    // Pre-compute the limb darkening / filter operator
    if ((udeg > 0) || (fdeg > 0)) {
        // Compute the two polynomials
        Vector<Scalar> tmp = B.U1 * u;
        Scalar norm = Scalar(1.0) / B.rT.segment(0, (udeg + 1) * (udeg + 1)).dot(tmp);
        Vector<Scalar> pu = tmp * norm * pi<Scalar>();
        Vector<Scalar> pf = B.A1.block(0, 0, Nf, Nf) * f;

        // Multiply them
        Vector<Scalar> p;
        Matrix<Scalar> dpdpu; // todo
        Matrix<Scalar> dpdpf; // todo
        if (fdeg == 0)
            p = pu;
        else if (udeg == 0)
            p = pf;
        else if (udeg > fdeg) 
            computePolynomialProduct<true>(udeg, pu, fdeg, pf, p, dpdpu, dpdpf);
        else
            computePolynomialProduct<true>(fdeg, pf, udeg, pu, p, dpdpf, dpdpu);

        // Compute the operator
        Matrix<Scalar> L;
        Vector<Matrix<Scalar>> dLdp;
        computePolynomialProductMatrix<true>(udeg + fdeg, p, L, dLdp);
        LA1 = (L * B.A1.block(0, 0, Ny, Ny));
        A2LA1 = (B.A2 * LA1).sparseView();
        rTLA1 = B.rT * LA1;

        throw std::runtime_error("TODO: Implement filter derivatives.");

        // Pre-compute its derivatives
        Matrix<Scalar> DpDu = pi<Scalar>() * norm * B.U1 - 
            pu * B.rT.segment(0, (udeg + 1) * (udeg + 1)) * B.U1 * norm;
        for (int l = 0; l < udeg + 1; ++l) {
            dLdu(l).setZero(N, Ny);
        }
        for (int j = 0; j < Np; ++j) {
            for (int l = 0; l < udeg + 1; ++l) {
                dLdu(l) += dLdp(j) * DpDu(j, l);
                rTdLduA1.row(l) = (B.rT * dLdu(l)) * B.A1.block(0, 0, Ny, Ny);
            }
        }
        Du.resize((Nu - 1) * nt, Ny * Nt);
    }
    Eigen::SparseMatrix<Scalar>& A = (udeg > 0) ? A2LA1 : B.A;
    RowVector<Scalar>& rTA1 = (udeg > 0) ? rTLA1 : B.rTA1;

    // Pre-compute the rotation
    W.leftMultiplyRZetaInv(rTA1, rTA1RZetaInv);

    // Our model matrix, f = X . y, and its derivatives
    X.resize(nt, Ny * Nt);
    Dtheta.resize(nt, Ny * Nt);
    Dxo.resize(nt, Ny * Nt);
    Dyo.resize(nt, Ny * Nt);
    Dro.resize(nt, Ny * Nt);
    Dinc.resize(nt, Ny * Nt);
    Dobl.resize(nt, Ny * Nt);
    if (S::Temporal)
        Dt.resize(nt, Ny * Nt);

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
            Dinc.row(n).setZero();
            Dobl.row(n).setZero();
            if (S::Temporal)
                Dt.row(n).setZero();

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
            W.leftMultiplyRZeta(rTA1RZetaInvDRzDtheta, 
                                Dtheta.block(n, 0, 1, Ny));
            Dtheta.block(n, 0, 1, Ny) *= radian;

            // Axis derivs
            W.leftMultiplyDRZetaInvDInc(rTA1, rTA1DRZetaInvDAngle);
            W.leftMultiplyRz(rTA1DRZetaInvDAngle, rTA1DRZetaInvDAngleRz);
            W.leftMultiplyRZeta(rTA1DRZetaInvDAngleRz, 
                                rTA1DRZetaInvDAngleRzRZeta);
            W.leftMultiplyDRZetaDInc(rTA1RZetaInvRz, 
                                     rTA1RZetaInvRzDRZetaDAngle);
            Dinc.block(n, 0, 1, Ny) = (rTA1DRZetaInvDAngleRzRZeta + 
                                       rTA1RZetaInvRzDRZetaDAngle) * radian;
            W.leftMultiplyDRZetaInvDObl(rTA1, rTA1DRZetaInvDAngle);
            W.leftMultiplyRz(rTA1DRZetaInvDAngle, rTA1DRZetaInvDAngleRz);
            W.leftMultiplyRZeta(rTA1DRZetaInvDAngleRz, 
                                rTA1DRZetaInvDAngleRzRZeta);
            W.leftMultiplyDRZetaDObl(rTA1RZetaInvRz, 
                                     rTA1RZetaInvRzDRZetaDAngle);
            Dobl.block(n, 0, 1, Ny) = (rTA1DRZetaInvDAngleRzRZeta + 
                                       rTA1RZetaInvRzDRZetaDAngle) * radian;

            // Limb darkening derivs
            if (udeg > 0) {
                for (int l = 1; l < udeg + 1; ++l) {
                    W.leftMultiplyR(rTdLduA1.row(l), 
                                    Du.block((l - 1) * nt + n, 0, 1, Ny));
                } 
            }

            // Apply the Taylor expansion?
            if (S::Temporal) {
                Dt.block(n, 0, 1, Ny).setZero();
                for (int i = 1; i < Nt; ++i) {
                    X.block(n, i * Ny, 1, Ny) = 
                        X.block(n, 0, 1, Ny) * taylor(n, i);
                    Dt.block(n, i * Ny, 1, Ny) = 
                        X.block(n, 0, 1, Ny) * taylor(n, i - 1);
                    Dtheta.block(n, i * Ny, 1, Ny) = 
                        Dtheta.block(n, 0, 1, Ny) * taylor(n, i);
                    Dinc.block(n, i * Ny, 1, Ny) = 
                        Dinc.block(n, 0, 1, Ny) * taylor(n, i);
                    Dobl.block(n, i * Ny, 1, Ny) = 
                        Dobl.block(n, 0, 1, Ny) * taylor(n, i);
                    if (udeg > 0) {
                        for (int l = 1; l < udeg + 1; ++l) {
                            Du.block((l - 1) * nt + n, i * Ny, 1, Ny) = 
                                Du.block((l - 1) * nt + n, 0, 1, Ny) * 
                                taylor(n, i);
                        } 
                    }
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
            sTA = G.sT * A;
            dsTdrA = G.dsTdr * A;
            dsTdbA = G.dsTdb * A;
            
            // Compute the occultor rotation matrix Rz
            if (likely(b != 0)) {
                invb = Scalar(1.0) / b;
                W.compute(yo(n) * invb, xo(n) * invb);
            } else {
                W.compute(1.0, 0.0);
            }

            // Dot stuff in
            W.leftMultiplyRz(sTA, sTARz);
            W.leftMultiplyRZetaInv(sTARz, sTARzRZetaInv);
            W.leftMultiplyRz(dsTdrA, dsTdrARz);
            W.leftMultiplyRZetaInv(dsTdrARz, dsTdrARzRZetaInv);
            W.leftMultiplyRz(dsTdbA, dsTdbARz);
            W.leftMultiplyRZetaInv(dsTdbARz, dsTdbARzRZetaInv);
            W.leftMultiplyDRz(sTA, sTADRzDw);
            W.leftMultiplyRZetaInv(sTADRzDw, sTADRzDwRZetaInv);
            if (udeg > 0) {
                for (int j = 0; j < Np; ++j) {
                    for (int l = 0; l < udeg + 1; ++l) {
                        sTA2dLduA1.row(l) = ((G.sT * B.A2) * dLdu(l)) * 
                                             B.A1.block(0, 0, Ny, Ny);
                    }
                }
                W.leftMultiplyRz(sTA2dLduA1, sTA2dLduA1Rz);
            }

            // Compute the Rz rotation matrix
            W.compute(cos(theta_rad(n)), sin(theta_rad(n)));
            
            // Dot Rz in
            W.leftMultiplyRz(sTARzRZetaInv, sTARzRZetaInvRz);

            // Transform back to the sky plane
            W.leftMultiplyRZeta(sTARzRZetaInvRz, X.block(n, 0, 1, Ny));

            // Theta deriv
            W.leftMultiplyDRz(sTARzRZetaInv, sTARzRZetaInvDRzDtheta);
            W.leftMultiplyRZeta(sTARzRZetaInvDRzDtheta, 
                                Dtheta.block(n, 0, 1, Ny));
            Dtheta.block(n, 0, 1, Ny) *= radian;

            // Radius deriv
            W.leftMultiplyRz(dsTdrARzRZetaInv, dsTdrARzRZetaInvRz);
            W.leftMultiplyRZeta(dsTdrARzRZetaInvRz, Dro.block(n, 0, 1, Ny));

            // xo and yo derivatives            
            if (likely(b != 0)) {    
                W.leftMultiplyRz(dsTdbARzRZetaInv, dsTdbARzRZetaInvRz);
                W.leftMultiplyRZeta(dsTdbARzRZetaInvRz, Dxo.block(n, 0, 1, Ny));
                Dyo.block(n, 0, 1, Ny) = Dxo.block(n, 0, 1, Ny);
                Dxo.block(n, 0, 1, Ny) *= xo(n) * invb;
                Dyo.block(n, 0, 1, Ny) *= yo(n) * invb;
                W.leftMultiplyRz(sTADRzDwRZetaInv, sTADRzDwRZetaInvRz);
                W.leftMultiplyRZeta(sTADRzDwRZetaInvRz, 
                                    sTADRzDwRZetaInvRzRZeta);
                sTADRzDwRZetaInvRzRZeta *= invb * invb;
                Dxo.block(n, 0, 1, Ny) += yo(n) * sTADRzDwRZetaInvRzRZeta;
                Dyo.block(n, 0, 1, Ny) -= xo(n) * sTADRzDwRZetaInvRzRZeta;
            } else {
                // \todo Need to compute these in the limit b-->0
                Dxo.block(n, 0, 1, Ny).setConstant(NAN);
                Dyo.block(n, 0, 1, Ny).setConstant(NAN);
            }

            // Axis derivs
            W.leftMultiplyDRZetaInvDInc(sTARz, sTARzDRZetaInvDAngle);
            W.leftMultiplyRz(sTARzDRZetaInvDAngle, sTARzDRZetaInvDAngleRz);
            W.leftMultiplyRZeta(sTARzDRZetaInvDAngleRz, 
                                sTARzDRZetaInvDAngleRzRZeta);
            W.leftMultiplyDRZetaDInc(sTARzRZetaInvRz, 
                                     sTARzRZetaInvRzDRZetaDAngle);
            Dinc.block(n, 0, 1, Ny) = (sTARzDRZetaInvDAngleRzRZeta + 
                                       sTARzRZetaInvRzDRZetaDAngle) * radian;
            W.leftMultiplyDRZetaInvDObl(sTARz, sTARzDRZetaInvDAngle);
            W.leftMultiplyRz(sTARzDRZetaInvDAngle, sTARzDRZetaInvDAngleRz);
            W.leftMultiplyRZeta(sTARzDRZetaInvDAngleRz, 
                                sTARzDRZetaInvDAngleRzRZeta);
            W.leftMultiplyDRZetaDObl(sTARzRZetaInvRz, 
                                     sTARzRZetaInvRzDRZetaDAngle);
            Dobl.block(n, 0, 1, Ny) = (sTARzDRZetaInvDAngleRzRZeta + 
                                       sTARzRZetaInvRzDRZetaDAngle) * radian;

            // Limb darkening derivs
            if (udeg > 0) {
                for (int l = 1; l < udeg + 1; ++l) {
                    W.leftMultiplyR(sTA2dLduA1Rz.row(l), 
                                    Du.block((l - 1) * nt + n, 0, 1, Ny));
                }
            }

            // Apply the Taylor expansion?
            if (S::Temporal) {
                Dt.block(n, 0, 1, Ny).setZero();
                for (int i = 1; i < Nt; ++i) {
                    X.block(n, i * Ny, 1, Ny) = 
                        X.block(n, 0, 1, Ny) * taylor(n, i);
                    Dt.block(n, i * Ny, 1, Ny) = 
                        X.block(n, 0, 1, Ny) * taylor(n, i - 1);
                    Dtheta.block(n, i * Ny, 1, Ny) = 
                        Dtheta.block(n, 0, 1, Ny) * taylor(n, i);
                    Dxo.block(n, i * Ny, 1, Ny) = 
                        Dxo.block(n, 0, 1, Ny) * taylor(n, i);
                    Dyo.block(n, i * Ny, 1, Ny) = 
                        Dyo.block(n, 0, 1, Ny) * taylor(n, i);
                    Dro.block(n, i * Ny, 1, Ny) = 
                        Dro.block(n, 0, 1, Ny) * taylor(n, i);
                    Dinc.block(n, i * Ny, 1, Ny) = 
                        Dinc.block(n, 0, 1, Ny) * taylor(n, i);
                    Dobl.block(n, i * Ny, 1, Ny) = 
                        Dobl.block(n, 0, 1, Ny) * taylor(n, i);
                    if (udeg > 0) {
                        for (int l = 1; l < udeg + 1; ++l) {
                            Du.block((l - 1) * nt + n, i * Ny, 1, Ny) = 
                                Du.block((l - 1) * nt + n, 0, 1, Ny) * 
                                taylor(n, i);
                        } 
                    }
                } 
            }

        }
    }
}

/**
\todo Compute the linear spherical harmonic model and its gradient. 
Internal method.

*/
template <typename U=S>
inline EnableIf<U::Reflected && !U::LimbDarkened, void> computeLinearFluxModelInternal (
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
    RowMatrix<Scalar>& Dinc,
    RowMatrix<Scalar>& Dobl
) {
    throw std::runtime_error("Gradients not yet implemented in reflected light.");
}

/**
Compute the flux from a purely limb-darkened map.

*/
template <typename U=S>
inline EnableIf<U::LimbDarkened, void> computeLimbDarkenedFluxInternal (
    const Vector<Scalar>& b, 
    const Vector<Scalar>& zo,
    const Vector<Scalar>& ro, 
    FType& flux
) {
    // Shape checks
    size_t nt = b.rows();
    CHECK_SHAPE(zo, nt, 1);
    CHECK_SHAPE(ro, nt, 1);
    flux.resize(nt, Nw);
    
    // Compute the Agol `g` basis
    L.computeBasis(u);

    // Loop through the timeseries
    for (size_t n = 0; n < nt; ++n) {

        // No occultation
        if ((zo(n) < 0) || (b(n) >= 1 + ro(n)) || (ro(n) <= 0.0)) {

            // Easy!
            flux.row(n).setOnes();

        // Occultation
        } else {

            // Compute the Agol `s` vector
            L.compute(b(n), ro(n));

            // Dot the integral solution in, and we're done!
            flux.row(n) = (L.sT * L.g).cwiseProduct(L.I0);

        }

    }

}

/**
Compute the flux from a purely limb-darkened map.
Also compute the gradient.

*/
template <typename U=S>
inline EnableIf<U::LimbDarkened, void> computeLimbDarkenedFluxInternal (
    const Vector<Scalar>& b, 
    const Vector<Scalar>& zo,
    const Vector<Scalar>& ro,
    FType& flux,
    FType& Db,
    FType& Dro,
    Matrix<Scalar>& Du
) {

    // Shape checks
    size_t nt = b.rows();
    CHECK_SHAPE(zo, nt, 1);
    CHECK_SHAPE(ro, nt, 1);
    flux.resize(nt, Nw);
    Db.resize(nt, Nw);
    Dro.resize(nt, Nw);
    Du.resize(udeg, nt * Nw);
    Matrix<Scalar> Dg(udeg + 1, nt * Nw);

    // Compute the Agol `g` basis
    L.computeBasis(u);

    // Loop through the timeseries
    for (size_t n = 0; n < nt; ++n) {

        // No occultation
        if ((zo(n) < 0) || (b(n) >= 1 + ro(n)) || (ro(n) <= 0.0)) {

            // Most of the derivs are zero
            Db.row(n).setZero();
            Dro.row(n).setZero();
            for (int w = 0; w < Nw; ++w)
                Dg.col(Nw * n + w).setZero();
            flux.row(n).setOnes();

        // Occultation
        } else {

            // Compute the Agol `s` vector and its derivs
            L.template compute<true>(b(n), ro(n));

            // Compute the flux
            flux.row(n) = (L.sT * L.g).cwiseProduct(L.I0);

            // b and ro derivs
            Db.row(n) = (L.dsTdb * L.g).cwiseProduct(L.I0);
            Dro.row(n) = (L.dsTdr * L.g).cwiseProduct(L.I0);

            // Compute df / dg
            if (likely(udeg > 0)) {
                for (int w = 0; w < Nw; ++w) {
                    Dg.col(Nw * n + w) = L.sT * L.I0(w);
                    Dg(0, Nw * n + w) -= pi<Scalar>() * flux(n, w) * L.I0(w);
                    Dg(1, Nw * n + w) -= (2.0 / 3.0) * pi<Scalar>() * flux(w) * L.I0(w);
                }
            }

        }

    }

    // Change basis to `u`
    if (likely(udeg > 0))
        Du = L.DgDu * Dg;

}