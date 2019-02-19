/**
Compute the linear spherical harmonic model. Internal method.

\todo We need to think about caching some of these rotations.


*/
inline void computeLinearModelInternal (
    const Vector<Scalar>& theta, 
    const Vector<Scalar>& xo, 
    const Vector<Scalar>& yo, 
    const Vector<Scalar>& zo, 
    const Vector<Scalar>& ro, 
    RowMatrix<Scalar>& A
) {

    // Shape checks
    size_t nt = theta.rows();
    CHECK_SHAPE(xo, nt, 1);
    CHECK_SHAPE(yo, nt, 1);
    CHECK_SHAPE(zo, nt, 1);
    CHECK_SHAPE(ro, nt, 1);

    // Pre-compute the Wigner matrices
    computeWigner();
    RowVector<Scalar> rTA1RZetaInv(N);
    RowVector<Scalar> rTA1RZetaInvRz(N);
    RowVector<Scalar> sTARz(N);
    RowVector<Scalar> sTARzRZetaInv(N);
    RowVector<Scalar> sTARzRZetaInvRz(N);

    W.leftMultiplyRZetaInv(B.rTA1, rTA1RZetaInv);
    Vector<Scalar> theta_rad = theta * radian;

    // Our model matrix, f = A . y
    A.resize(nt, N);

    // Loop over the timeseries and compute the model
    Scalar b;
    Scalar theta_cache = NAN;
    for (size_t n = 0; n < nt; ++n) {

        // Impact parameter
        b = sqrt(xo(n) * xo(n) + yo(n) * yo(n));

        // Complete occultation?
        if (b <= ro(n) - 1) {

            A.row(n).setZero();

        // No occultation
        } else if ((zo(n) < 0) || (b >= 1 + ro(n)) || (ro(n) <= 0.0)) {

            // Compute the Rz rotation matrix
            W.computeFourierTerms(cos(theta_rad(n)), sin(theta_rad(n)));

            // Dot everything together to get the model
            W.leftMultiplyRz(rTA1RZetaInv, rTA1RZetaInvRz);
            W.leftMultiplyRZeta(rTA1RZetaInvRz, A.row(n));
            theta_cache = theta_rad(n);
            
        // Occultation
        } else {

            // Compute the solution vector
            G.compute(b, ro(n));

            // Compute the occultor rotation matrix Rz
            W.computeFourierTerms(yo(n) / b, xo(n) / b);

            // Dot stuff in
            W.leftMultiplyRz(G.sT * B.A, sTARz);
            W.leftMultiplyRZetaInv(sTARz, sTARzRZetaInv);

            // Compute the Rz rotation matrix
            W.computeFourierTerms(cos(theta_rad(n)), sin(theta_rad(n)));

            // Dot everything together to get the model
            W.leftMultiplyRz(sTARzRZetaInv, sTARzRZetaInvRz);
            W.leftMultiplyRZeta(sTARzRZetaInvRz, A.row(n));

        }
    }
}

/**
Compute the linear spherical harmonic model and its gradient. Internal method.

\todo Occultation gradients

*/
inline void computeLinearModelInternal (
    const Vector<Scalar>& theta, 
    const Vector<Scalar>& xo, 
    const Vector<Scalar>& yo, 
    const Vector<Scalar>& zo, 
    const Vector<Scalar>& ro, 
    RowMatrix<Scalar>& A,
    RowMatrix<Scalar>& Dtheta,
    RowMatrix<Scalar>& Dxo,
    RowMatrix<Scalar>& Dyo,
    RowMatrix<Scalar>& Dro
) {

    // Shape checks
    size_t nt = theta.rows();
    CHECK_SHAPE(xo, nt, 1);
    CHECK_SHAPE(yo, nt, 1);
    CHECK_SHAPE(zo, nt, 1);
    CHECK_SHAPE(ro, nt, 1);

    // Pre-compute the Wigner matrices
    computeWigner();
    RowVector<Scalar> rTA1RZetaInv(N);
    RowVector<Scalar> rTA1RZetaInvRz(N);
    RowVector<Scalar> rTA1RZetaInvDRzDtheta(N);
    RowVector<Scalar> sTA(N);
    RowVector<Scalar> sTARz(N);
    RowVector<Scalar> dsTdrARz(N);
    RowVector<Scalar> dsTdbARz(N);
    RowVector<Scalar> sTADRzDw(N);
    RowVector<Scalar> sTARzRZetaInv(N);
    RowVector<Scalar> dsTdrARzRZetaInv(N);
    RowVector<Scalar> dsTdbARzRZetaInv(N);
    RowVector<Scalar> sTADRzDwRZetaInv(N);
    RowVector<Scalar> sTARzRZetaInvRz(N);
    RowVector<Scalar> sTARzRZetaInvDRzDtheta(N);
    RowVector<Scalar> dsTdrARzRZetaInvRz(N);
    RowVector<Scalar> dsTdbARzRZetaInvRz(N);
    RowVector<Scalar> sTADRzDwRZetaInvRz(N);
    RowVector<Scalar> sTADRzDwRZetaInvRzRZeta(N);

    W.leftMultiplyRZetaInv(B.rTA1, rTA1RZetaInv);
    Vector<Scalar> theta_rad = theta * radian;

    // Our model matrix, f = A . y
    A.resize(nt, N);

    // Loop over the timeseries and compute the model
    Scalar b, invb;
    for (size_t n = 0; n < nt; ++n) {

        // Impact parameter
        b = sqrt(xo(n) * xo(n) + yo(n) * yo(n));

        // Complete occultation?
        if (b <= ro(n) - 1) {

            A.row(n).setZero();
            Dtheta.row(n).setZero();
            Dxo.row(n).setZero();
            Dyo.row(n).setZero();
            Dro.row(n).setZero();

        // No occultation
        } else if ((zo(n) < 0) || (b >= 1 + ro(n)) || (ro(n) <= 0.0)) {

            // Compute the Rz rotation matrix
            W.computeFourierTerms(cos(theta_rad(n)), sin(theta_rad(n)));

            // Dot everything together to get the model
            W.leftMultiplyRz(rTA1RZetaInv, rTA1RZetaInvRz);
            W.leftMultiplyRZeta(rTA1RZetaInvRz, A.row(n));

            // Theta deriv
            W.leftMultiplyDRz(rTA1RZetaInv, rTA1RZetaInvDRzDtheta);
            W.leftMultiplyRZeta(rTA1RZetaInvDRzDtheta, Dtheta.row(n));
            Dtheta.row(n) *= radian;

            // Occultor derivs are zero
            Dxo.row(n).setZero();
            Dyo.row(n).setZero();
            Dro.row(n).setZero();
            
        // Occultation
        } else {

            // Compute the solution vector
            G.template compute<true>(b, ro(n));
            sTA = G.sT * B.A;
            invb = Scalar(1.0) / b;

            // Compute the occultor rotation matrix Rz
            W.computeFourierTerms(yo(n) * invb, xo(n) * invb);

            // Dot stuff in
            W.leftMultiplyRz(sTA, sTARz);
            W.leftMultiplyRZetaInv(sTARz, sTARzRZetaInv);
            W.leftMultiplyRz(G.dsTdr * B.A, dsTdrARz);
            W.leftMultiplyRZetaInv(dsTdrARz, dsTdrARzRZetaInv);
            W.leftMultiplyRz(G.dsTdb * B.A, dsTdbARz);
            W.leftMultiplyRZetaInv(dsTdbARz, dsTdbARzRZetaInv);
            W.leftMultiplyDRz(sTA, sTADRzDw);
            W.leftMultiplyRZetaInv(sTADRzDw, sTADRzDwRZetaInv);

            // Compute the Rz rotation matrix
            W.computeFourierTerms(cos(theta_rad(n)), sin(theta_rad(n)));

            // Dot everything together to get the model
            W.leftMultiplyRz(sTARzRZetaInv, sTARzRZetaInvRz);
            W.leftMultiplyRZeta(sTARzRZetaInvRz, A.row(n));

            // Theta deriv
            W.leftMultiplyDRz(sTARzRZetaInv, sTARzRZetaInvDRzDtheta);
            W.leftMultiplyRZeta(sTARzRZetaInvDRzDtheta, Dtheta.row(n));
            Dtheta.row(n) *= radian;

            // Radius deriv
            W.leftMultiplyRz(dsTdrARzRZetaInv, dsTdrARzRZetaInvRz);
            W.leftMultiplyRZeta(dsTdrARzRZetaInvRz, Dro.row(n));

            // xo and yo derivatives
            W.leftMultiplyRz(dsTdbARzRZetaInv, dsTdbARzRZetaInvRz);
            W.leftMultiplyRZeta(dsTdbARzRZetaInvRz, Dxo.row(n));
            Dyo.row(n) = Dxo.row(n);
            Dxo.row(n) *= xo(n) * invb;
            Dyo.row(n) *= yo(n) * invb;
            W.leftMultiplyRz(sTADRzDwRZetaInv, sTADRzDwRZetaInvRz);
            W.leftMultiplyRZeta(sTADRzDwRZetaInvRz, sTADRzDwRZetaInvRzRZeta);
            sTADRzDwRZetaInvRzRZeta *= invb * invb;
            Dxo.row(n) += yo(n) * sTADRzDwRZetaInvRzRZeta;
            Dyo.row(n) -= xo(n) * sTADRzDwRZetaInvRzRZeta;

        }
    }
}