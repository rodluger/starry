/**
Compute the max like map coefficients. Internal method.

\todo We need to think about caching some of these rotations.
\todo We should eventually allow for off-diagonal elements in C and L


*/
inline void computeMaxLikeMapInternal (
    const Vector<Scalar>& flux, 
    const Vector<Scalar>& C, 
    const Vector<Scalar>& L,
    const Vector<Scalar>& theta, 
    const Vector<Scalar>& xo, 
    const Vector<Scalar>& yo, 
    const Vector<Scalar>& zo, 
    const Vector<Scalar>& ro, 
    Matrix<Scalar>& A,
    Vector<Scalar>& yhat,
    Matrix<Scalar>& yvar
) {

    // Shape checks
    size_t nt = flux.rows();
    CHECK_SHAPE(C, nt, 1);
    CHECK_SHAPE(L, N, 1);
    CHECK_SHAPE(theta, nt, 1);
    CHECK_SHAPE(xo, nt, 1);
    CHECK_SHAPE(yo, nt, 1);
    CHECK_SHAPE(zo, nt, 1);
    CHECK_SHAPE(ro, nt, 1);
    CHECK_SHAPE(yhat, N, 1);
    CHECK_SHAPE(yvar, N, N);

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


    // Now compute the MAP matrix,
    //    M = (A^T . C^-1 . A + L^-1)^-1 . A^T . C^-1
    Eigen::HouseholderQR<Matrix<double>> solver(N, N);
    Vector<double> CInv = C.cwiseInverse();
    Matrix<double> K = A.transpose() * CInv.asDiagonal() * A;
    Matrix<double> KReg = K;
    KReg.diagonal() += L.cwiseInverse();
    solver.compute(KReg);
    Matrix<double> M = solver.solve(A.transpose() * CInv.asDiagonal());

    // The MAP map coefficients are then just
    //    y = M . f
    yhat = M * flux;

    // The MAP variance is just
    //   var(y) = K^-1
    solver.compute(K);
    yvar = solver.solve(Matrix<double>::Identity(N, N));
}