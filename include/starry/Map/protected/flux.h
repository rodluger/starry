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
    Matrix<Scalar> L;
    Vector<Matrix<Scalar>> DLDp; // not used
    Matrix<Scalar> DpDpu; // not used
    Matrix<Scalar> DpDpf; // not used
    bool apply_filter = (udeg > 0) || (fdeg > 0);
    if (apply_filter) {
        
        // Compute the two polynomials
        Vector<Scalar> tmp = B.U1 * u;
        Scalar norm = Scalar(1.0) / B.rT.segment(0, (udeg + 1) * (udeg + 1)).dot(tmp);
        Vector<Scalar> pu = tmp * norm * pi<Scalar>();
        Vector<Scalar> pf;
        pf = B.A1.block(0, 0, Nf, Nf) * f;

        // Multiply them
        Vector<Scalar> p;
        if (udeg > fdeg) {
            computePolynomialProduct<false>(udeg, pu, fdeg, pf, p, DpDpu, DpDpf);
        } else {
            computePolynomialProduct<false>(fdeg, pf, udeg, pu, p, DpDpf, DpDpu);
        }

        // Compute the polynomial filter operator
        computePolynomialProductMatrix<false>(udeg + fdeg, p, L, DLDp);

        // Compute the phase curve integral operator
        LA1 = (L * B.A1.block(0, 0, Ny, Ny)).sparseView();
        rTLA1 = B.rT * LA1;
        
        // Rotate the filter operator fully into Ylm space
        L = B.A1Inv * LA1;

    }
    // Pre-compute the rotation
    if (apply_filter)
        W.leftMultiplyRZetaInv(rTLA1, rTLA1RZetaInv);
    else
        W.leftMultiplyRZetaInv(B.rTA1, rTLA1RZetaInv);

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
                W.leftMultiplyRz(rTLA1RZetaInv, rTLA1RZetaInvRz);
                W.leftMultiplyRZeta(rTLA1RZetaInvRz, X.block(n, 0, 1, Ny));
            } else {
                W.leftMultiplyRZeta(rTLA1RZetaInvRz, X.block(n, 0, 1, Ny));
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
            sTA = G.sT * B.A;

            // Compute the occultor rotation matrix Rz
            if (likely(b != 0)) {
                invb = Scalar(1.0) / b;
                W.compute(yo(n) * invb, xo(n) * invb);
            } else {
                W.compute(1.0, 0.0);
            }

            // Rotate & apply the filter
            if (apply_filter) {
                W.leftMultiplyRzAugmented(sTA, sTARz);
                sTARzL = sTARz * L;
            } else {
                W.leftMultiplyRz(sTA, sTARzL);
            }

            // Rotate the map
            W.leftMultiplyRZetaInv(sTARzL, sTARzLRZetaInv);
            if (theta_rad(n) != theta_occ_cache) {
                theta_occ_cache = theta_rad(n);
                W.compute(cos(theta_rad(n)), sin(theta_rad(n)));
            }
            W.leftMultiplyRz(sTARzLRZetaInv, sTARzLRZetaInvRz);
            W.leftMultiplyRZeta(sTARzLRZetaInvRz, X.block(n, 0, 1, Ny));

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
    Matrix<Scalar> L;
    Vector<Matrix<Scalar>> DLDp; // not used
    Matrix<Scalar> DpDpu; // not used
    Matrix<Scalar> DpDpf; // not used
    bool apply_filter = (udeg > 0) || (fdeg > 0);
    if (apply_filter) {

        // Compute the two polynomials
        Vector<Scalar> tmp = B.U1 * u;
        Scalar norm = Scalar(1.0) / B.rT.segment(0, (udeg + 1) * (udeg + 1)).dot(tmp);
        Vector<Scalar> pu = tmp * norm * pi<Scalar>();
        Vector<Scalar> pf;
        pf = B.A1.block(0, 0, Nf, Nf) * f;

        // Multiply them
        Vector<Scalar> p;
        if (udeg > fdeg) {
            computePolynomialProduct<false>(udeg, pu, fdeg, pf, p, DpDpu, DpDpf);
        } else {
            computePolynomialProduct<false>(fdeg, pf, udeg, pu, p, DpDpf, DpDpu);
        }

        // Compute the operator
        computePolynomialProductMatrix<false>(udeg + fdeg, p, L, DLDp);
        LA1 = (L * B.A1.block(0, 0, Ny, Ny)).sparseView();

        // Rotate the filter operator fully into Ylm space
        L = B.A1Inv * LA1;
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
                    rTA1 = G.rT * B.A1;
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

                // Rotate & apply the filter
                if (apply_filter) {
                    W.leftMultiplyRzAugmented(rTA1, rTA1Rz);
                    rTA1RzL = rTA1Rz * L;
                } else {
                    W.leftMultiplyRz(rTA1, rTA1RzL);
                }

                // Cache the source position
                sx_cache = source(n, 0);
                sy_cache = source(n, 1);
                sz_cache = source(n, 2);

            }

            // Rotate to the correct phase
            W.compute(cos(theta_rad(n)), sin(theta_rad(n)));
            W.leftMultiplyRZetaInv(rTA1RzL, rTA1RzLRZetaInv);
            W.leftMultiplyRz(rTA1RzLRZetaInv, rTA1RzLRZetaInvRz);
            W.leftMultiplyRZeta(rTA1RzLRZetaInvRz, X.block(n, 0, 1, Ny));

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
    const RowMatrix<Scalar>& bX,
    Vector<Scalar>& bt,
    Vector<Scalar>& btheta, 
    Vector<Scalar>& bxo,
    Vector<Scalar>& byo,
    Vector<Scalar>& bro,
    UType& bu,
    Vector<Scalar>& bf,
    Scalar& binc,
    Scalar& bobl
) {

    // TODO: PHASE THESE OUT
    RowMatrix<Scalar> Dt;
    RowMatrix<Scalar> Dtheta;
    RowMatrix<Scalar> Dxo;
    RowMatrix<Scalar> Dyo;
    RowMatrix<Scalar> Dro;
    RowMatrix<Scalar> Du;
    RowMatrix<Scalar> Df;
    RowMatrix<Scalar> Dinc;
    RowMatrix<Scalar> Dobl;

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
    Matrix<Scalar> L;
    Vector<Matrix<Scalar>> DLDp;
    Matrix<Scalar> DpDpu;
    Matrix<Scalar> DpDpf;
    bool apply_filter = (udeg > 0) || (fdeg > 0);
    if (apply_filter) {
        
        // Compute the two polynomials
        Vector<Scalar> tmp = B.U1 * u;
        Scalar norm = Scalar(1.0) / B.rT.segment(0, (udeg + 1) * (udeg + 1)).dot(tmp);
        Vector<Scalar> pu = tmp * norm * pi<Scalar>();
        Vector<Scalar> pf;
        pf = B.A1.block(0, 0, Nf, Nf) * f;

        // Multiply them
        Vector<Scalar> p;
        if (udeg > fdeg) {
            computePolynomialProduct<true>(udeg, pu, fdeg, pf, p, DpDpu, DpDpf);
        } else {
            computePolynomialProduct<true>(fdeg, pf, udeg, pu, p, DpDpf, DpDpu);
        }

        // Compute the polynomial filter operator
        computePolynomialProductMatrix<true>(udeg + fdeg, p, L, DLDp);

        // Compute the phase curve integral operator
        LA1 = (L * B.A1.block(0, 0, Ny, Ny)).sparseView();
        rTLA1 = B.rT * LA1;
        
        // Rotate the filter operator fully into Ylm space
        L = B.A1Inv * LA1;

        // Pre-compute the limb darkening derivatives
        Matrix<Scalar> DpuDu = pi<Scalar>() * norm * B.U1 - 
            pu * B.rT.segment(0, (udeg + 1) * (udeg + 1)) * B.U1 * norm;
        for (int l = 0; l < udeg + 1; ++l) {
            DLDu(l).setZero(N, Ny);
        }
        Matrix<Scalar> DpDu = DpDpu * DpuDu;
        for (int j = 0; j < (udeg + fdeg + 1) * (udeg + fdeg + 1); ++j) {
            for (int l = 0; l < udeg + 1; ++l) {
                DLDu(l) += DLDp(j) * DpDu(j, l);
                rTDLDuA1.row(l) = (B.rT * DLDu(l)) * B.A1.block(0, 0, Ny, Ny);
            }
        }
        // Rotate DLDu fully into Ylm space
        for (int l = 0; l < udeg + 1; ++l) {
            DLDu(l) = B.A1Inv * DLDu(l) * B.A1.block(0, 0, Ny, Ny);
        }
        Du.resize((Nu - 1) * nt, Ny * Nt);

        // Pre-compute the filter derivatives
        Matrix<Scalar> DpfDf = B.A1.block(0, 0, Nf, Nf);
        for (int l = 0; l < Nf; ++l) {
            DLDf(l).setZero(N, Ny);
        }
        Matrix<Scalar> DpDf = DpDpf * DpfDf;
        for (int j = 0; j < (udeg + fdeg + 1) * (udeg + fdeg + 1); ++j) {
            for (int l = 0; l < Nf; ++l) {
                DLDf(l) += DLDp(j) * DpDf(j, l);
                rTDLDfA1.row(l) = (B.rT * DLDf(l)) * B.A1.block(0, 0, Ny, Ny);
            }
        }
        // Rotate DLDf fully into Ylm space
        for (int l = 0; l < Nf; ++l) {
            DLDf(l) = B.A1Inv * DLDf(l) * B.A1.block(0, 0, Ny, Ny);
        }
        Df.resize(Nf * nt, Ny * Nt);

    } else {

        // Life is so much easier!
        rTLA1 = B.rTA1.segment(0, Ny);
        Du.resize(0, 0);
        Df.resize(0, 0);

    }

    // Pre-compute the rotation
    W.leftMultiplyRZetaInv(rTLA1, rTLA1RZetaInv);

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
            W.leftMultiplyRz(rTLA1RZetaInv, rTLA1RZetaInvRz);

            // Transform back to the sky plane
            W.leftMultiplyRZeta(rTLA1RZetaInvRz, X.block(n, 0, 1, Ny));

            // Theta deriv
            W.leftMultiplyDRz(rTLA1RZetaInv, rTLA1RZetaInvDRzDtheta);
            W.leftMultiplyRZeta(rTLA1RZetaInvDRzDtheta, 
                                Dtheta.block(n, 0, 1, Ny));
            Dtheta.block(n, 0, 1, Ny) *= radian;

            // Axis derivs
            W.leftMultiplyDRZetaInvDInc(rTLA1, rTLA1DRZetaInvDAngle);
            W.leftMultiplyRz(rTLA1DRZetaInvDAngle, rTLA1DRZetaInvDAngleRz);
            W.leftMultiplyRZeta(rTLA1DRZetaInvDAngleRz, 
                                rTLA1DRZetaInvDAngleRzRZeta);
            W.leftMultiplyDRZetaDInc(rTLA1RZetaInvRz, 
                                     rTLA1RZetaInvRzDRZetaDAngle);
            Dinc.block(n, 0, 1, Ny) = (rTLA1DRZetaInvDAngleRzRZeta + 
                                       rTLA1RZetaInvRzDRZetaDAngle) * radian;
            W.leftMultiplyDRZetaInvDObl(rTLA1, rTLA1DRZetaInvDAngle);
            W.leftMultiplyRz(rTLA1DRZetaInvDAngle, rTLA1DRZetaInvDAngleRz);
            W.leftMultiplyRZeta(rTLA1DRZetaInvDAngleRz, 
                                rTLA1DRZetaInvDAngleRzRZeta);
            W.leftMultiplyDRZetaDObl(rTLA1RZetaInvRz, 
                                     rTLA1RZetaInvRzDRZetaDAngle);
            Dobl.block(n, 0, 1, Ny) = (rTLA1DRZetaInvDAngleRzRZeta + 
                                       rTLA1RZetaInvRzDRZetaDAngle) * radian;

            // Limb darkening derivs
            if (udeg > 0) {
                for (int l = 1; l < udeg + 1; ++l) {
                    W.leftMultiplyR(rTDLDuA1.row(l), 
                                    Du.block((l - 1) * nt + n, 0, 1, Ny));
                } 
            }

            // Filter derivs
            if (fdeg > 0) {
                for (int l = 0; l < Nf; ++l) {
                    W.leftMultiplyR(rTDLDfA1.row(l), 
                                    Df.block(l * nt + n, 0, 1, Ny));
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
                    if (fdeg > 0) {
                        for (int l = 0; l < Nf; ++l) {
                            Df.block(l * nt + n, i * Ny, 1, Ny) = 
                                Df.block(l * nt + n, 0, 1, Ny) * 
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
            sTA = G.sT * B.A;
            DsTDrA = G.dsTdr * B.A;
            DsTDbA = G.dsTdb * B.A;
            
            // Compute the occultor rotation matrix Rz
            if (likely(b != 0)) {
                invb = Scalar(1.0) / b;
                W.compute(yo(n) * invb, xo(n) * invb);
            } else {
                invb = INFINITY;
                W.compute(1.0, 0.0);
            }

            // Dot stuff in
            if (apply_filter) {
                W.leftMultiplyRzAugmented(sTA, sTARz);
                sTARzL = sTARz * L;
                W.leftMultiplyRzAugmented(DsTDrA, DsTDrARz);
                DsTDrARzL = DsTDrARz * L;
                W.leftMultiplyRzAugmented(DsTDbA, DsTDbARz);
                DsTDbARzL = DsTDbARz * L;
                W.leftMultiplyDRzAugmented(sTA, sTADRzDw);
                sTADRzDwL = sTADRzDw * L;
            } else {
                W.leftMultiplyRz(sTA, sTARzL);
                W.leftMultiplyRz(DsTDrA, DsTDrARzL);
                W.leftMultiplyRz(DsTDbA, DsTDbARzL);
                W.leftMultiplyDRz(sTA, sTADRzDwL);
            }
            W.leftMultiplyRZetaInv(sTARzL, sTARzLRZetaInv);
            W.leftMultiplyRZetaInv(DsTDrARzL, DsTDrARzLRZetaInv);
            W.leftMultiplyRZetaInv(DsTDbARzL, DsTDbARzLRZetaInv);
            W.leftMultiplyRZetaInv(sTADRzDwL, sTADRzDwLRZetaInv);
            if (udeg > 0) {
                for (int l = 0; l < udeg + 1; ++l) {
                    sTARzDLDu.row(l) = sTARz * DLDu(l);
                }
            }
            if (fdeg > 0) {
                for (int l = 0; l < Nf; ++l) {
                    sTARzDLDf.row(l) = sTARz * DLDf(l);
                }
            }

            // Compute the Rz rotation matrix in the zeta frame
            W.compute(cos(theta_rad(n)), sin(theta_rad(n)));
            
            // Dot Rz in
            W.leftMultiplyRz(sTARzLRZetaInv, sTARzLRZetaInvRz);

            // Transform back to the sky plane
            W.leftMultiplyRZeta(sTARzLRZetaInvRz, X.block(n, 0, 1, Ny));

            // Theta deriv
            W.leftMultiplyDRz(sTARzLRZetaInv, sTARzLRZetaInvDRzDtheta);
            W.leftMultiplyRZeta(sTARzLRZetaInvDRzDtheta, 
                                Dtheta.block(n, 0, 1, Ny));
            Dtheta.block(n, 0, 1, Ny) *= radian;

            // Radius deriv
            W.leftMultiplyRz(DsTDrARzLRZetaInv, DsTDrARzLRZetaInvRz);
            W.leftMultiplyRZeta(DsTDrARzLRZetaInvRz, Dro.block(n, 0, 1, Ny));

            // xo and yo derivatives            
            if (likely(b != 0)) {    
                W.leftMultiplyRz(DsTDbARzLRZetaInv, DsTDbARzLRZetaInvRz);
                W.leftMultiplyRZeta(DsTDbARzLRZetaInvRz, Dxo.block(n, 0, 1, Ny));
                Dyo.block(n, 0, 1, Ny) = Dxo.block(n, 0, 1, Ny);
                Dxo.block(n, 0, 1, Ny) *= xo(n) * invb;
                Dyo.block(n, 0, 1, Ny) *= yo(n) * invb;
                W.leftMultiplyRz(sTADRzDwLRZetaInv, sTADRzDwLRZetaInvRz);
                W.leftMultiplyRZeta(sTADRzDwLRZetaInvRz, 
                                    sTADRzDwLRZetaInvRzRZeta);
                sTADRzDwLRZetaInvRzRZeta *= invb * invb;
                Dxo.block(n, 0, 1, Ny) += yo(n) * sTADRzDwLRZetaInvRzRZeta;
                Dyo.block(n, 0, 1, Ny) -= xo(n) * sTADRzDwLRZetaInvRzRZeta;
            } else {
                // \todo Need to compute these in the limit b-->0
                Dxo.block(n, 0, 1, Ny).setConstant(NAN);
                Dyo.block(n, 0, 1, Ny).setConstant(NAN);
            }

            // Axis derivs
            W.leftMultiplyDRZetaInvDInc(sTARzL, sTARzLDRZetaInvDAngle);
            W.leftMultiplyRz(sTARzLDRZetaInvDAngle, sTARzLDRZetaInvDAngleRz);
            W.leftMultiplyRZeta(sTARzLDRZetaInvDAngleRz, 
                                sTARzLDRZetaInvDAngleRzRZeta);
            W.leftMultiplyDRZetaDInc(sTARzLRZetaInvRz, 
                                     sTARzLRZetaInvRzDRZetaDAngle);
            Dinc.block(n, 0, 1, Ny) = (sTARzLDRZetaInvDAngleRzRZeta + 
                                       sTARzLRZetaInvRzDRZetaDAngle) * radian;
            W.leftMultiplyDRZetaInvDObl(sTARzL, sTARzLDRZetaInvDAngle);
            W.leftMultiplyRz(sTARzLDRZetaInvDAngle, sTARzLDRZetaInvDAngleRz);
            W.leftMultiplyRZeta(sTARzLDRZetaInvDAngleRz, 
                                sTARzLDRZetaInvDAngleRzRZeta);
            W.leftMultiplyDRZetaDObl(sTARzLRZetaInvRz, 
                                     sTARzLRZetaInvRzDRZetaDAngle);
            Dobl.block(n, 0, 1, Ny) = (sTARzLDRZetaInvDAngleRzRZeta + 
                                       sTARzLRZetaInvRzDRZetaDAngle) * radian;

            // Limb darkening derivs
            if (udeg > 0) {
                for (int l = 1; l < udeg + 1; ++l) {
                    W.leftMultiplyR(sTARzDLDu.row(l), 
                                    Du.block((l - 1) * nt + n, 0, 1, Ny));
                }
            }

            // Filter derivs
            if (fdeg > 0) {
                for (int l = 0; l < Nf; ++l) {
                    W.leftMultiplyR(sTARzDLDf.row(l), 
                                    Df.block(l * nt + n, 0, 1, Ny));
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
                    if (fdeg > 0) {
                        for (int l = 0; l < Nf; ++l) {
                            Df.block(l * nt + n, i * Ny, 1, Ny) = 
                                Df.block(l * nt + n, 0, 1, Ny) * 
                                taylor(n, i);
                        } 
                    }
                } 
            }

        }
    }

    // Backprop (TODO: compute these directly)
    bu.resize(udeg);
    if (udeg > 0) {
        for (int l = 0; l < udeg; ++l) {
            bu(l) = Du.block(l * nt, 0, nt, Ny).cwiseProduct(bX).sum();
        }
    }
    if (fdeg > 0) {
        bf.resize(Nf);
        for (int l = 0; l < udeg; ++l) {
            bf(l) = Df.block(l * nt, 0, nt, Ny).cwiseProduct(bX).sum();
        }
    } else {
        bf.resize(0);
    }
    binc = Dinc.cwiseProduct(bX).sum();
    bobl = Dobl.cwiseProduct(bX).sum();
    btheta = Dtheta.cwiseProduct(bX).rowwise().sum();
    bxo = Dxo.cwiseProduct(bX).rowwise().sum();
    byo = Dyo.cwiseProduct(bX).rowwise().sum();
    bro = Dro.cwiseProduct(bX).rowwise().sum();
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
    RowMatrix<Scalar>& Df,
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
        if ((zo(n) < 0) || (abs(b(n)) >= 1 + ro(n)) || (ro(n) <= 0.0)) {

            // Easy!
            flux.row(n).setOnes();

        // Occultation
        } else {

            // Compute the Agol `s` vector
            L.compute(abs(b(n)), ro(n));

            // Dot the integral solution in, and we're done!
            flux.row(n) = (L.sT * L.g).cwiseProduct(L.I0);

        }

    }

}

/**
Compute the flux from a purely limb-darkened map.
Also compute the (backprop) gradient.

*/
template <typename U=S>
inline EnableIf<U::LimbDarkened, void> computeLimbDarkenedFluxInternal (
    const Vector<Scalar>& b, 
    const Vector<Scalar>& zo,
    const Vector<Scalar>& ro,
    FType& flux,
    const FType& bf,
    Vector<Scalar>& bb,
    Vector<Scalar>& bro,
    UType& bu
) {

    // Shape checks
    size_t nt = b.rows();
    CHECK_SHAPE(zo, nt, 1);
    CHECK_SHAPE(ro, nt, 1);
    CHECK_SHAPE(bf, nt, Nw);
    flux.resize(nt, Nw);
    
    // Initialize derivs
    bb.resize(nt);
    bro.resize(nt);
    UType bg(udeg + 1, Nw);
    bg.setZero();
    RowVector<Scalar> I0bf, piI0fbf;

    // Compute the Agol `g` basis
    L.computeBasis(u);

    // Loop through the timeseries
    for (size_t n = 0; n < nt; ++n) {

        // No occultation
        if ((zo(n) < 0) || (abs(b(n)) >= 1 + ro(n)) || (ro(n) <= 0.0)) {

            flux.row(n).setOnes();

        // Occultation
        } else {

            // Compute the Agol `s` vector and its derivs
            L.template compute<true>(abs(b(n)), ro(n));

            // Compute the flux
            flux.row(n) = (L.sT * L.g).cwiseProduct(L.I0);

            // b and ro derivs
            I0bf = (L.I0).cwiseProduct(bf.row(n));
            bb(n) = (L.dsTdb * L.g).dot(I0bf);
            bro(n) = (L.dsTdr * L.g).dot(I0bf);

            // Compute df / dg
            if (likely(udeg > 0)) {
                bg += L.sT.transpose() * I0bf;
                piI0fbf = pi<Scalar>() * I0bf.cwiseProduct(flux.row(n));
                bg.row(0) -= piI0fbf;
                bg.row(1) -= (2.0 / 3.0) * piI0fbf;
            }

        }

    }

    // Change basis to `u`
    if (likely(udeg > 0))
        bu = L.DgDu * bg;
    else
        bu.setZero(udeg, Nw);

}