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
    Matrix<Scalar>& A
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
Compute the maximum likelihood map coefficients. Internal method.
This method computes the MAP matrix,

   M = W^-1 . A^T . C^-1

where

  W^-1 = (A^T . C^-1 . A + L^-1)^-1

The MAP map coefficients are then just

   y = M . f

and the variance is

   var(y) = M . A . W^-1
          = M . (W^-1 . A^T)^T


\todo The expression for the variance above is not what we want; we
      want the uncertainty on the coefficients, not the variance
      of the MAP estimator!
\todo We should allow for off-diagonal elements in C and L. Use MatrixBase

*/
inline void computeMaxLikeMapInternal (
    const Matrix<Scalar>& A,
    const Vector<Scalar>& flux, 
    const Vector<Scalar>& C, 
    const Vector<Scalar>& L,
    Vector<Scalar>& yhat,
    Matrix<Scalar>& yvar
) {
    Eigen::HouseholderQR<Matrix<double>> solver(N, N);
    Vector<double> CInv = C.cwiseInverse();
    Matrix<double> W = A.transpose() * CInv.asDiagonal() * A;
    W.diagonal() += L.cwiseInverse();
    solver.compute(W);
    Matrix<double> WInvAT = solver.solve(A.transpose());
    Matrix<double> M = WInvAT * CInv.asDiagonal();
    yhat = M * flux;
    yvar = M * WInvAT.transpose();
}