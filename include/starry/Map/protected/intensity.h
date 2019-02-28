/**

*/
inline void computeLinearIntensityModelInternal (
    const Vector<Scalar>& theta, 
    const RowMatrix<Scalar>& x_, 
    const RowMatrix<Scalar>& y_,
    const RowMatrix<Scalar>& source,
    RowMatrix<Scalar>& X
) {

    // Shape checks
    CHECK_SHAPE(y_, x_.rows(), x_.cols());
    if (S::Reflected) {
        CHECK_SHAPE(source, theta.rows(), 3);
    }
    if (std::is_same<S, Temporal<Scalar, S::Reflected>>::value) {
        CHECK_ROWS(taylor, theta.rows());
    }

    // Flatten x and y
    Eigen::Map<const RowVector<Scalar>> x(x_.data(), x_.size());
    Eigen::Map<const RowVector<Scalar>> y(y_.data(), y_.size());
    size_t npts = x.size();

    // Number of time points
    size_t ntimes = theta.size();

    // Our model matrix for the first timestep, f0 = X0 . y
    RowMatrix<Scalar> X0;
    
    if (udeg == 0) {

        // Compute the polynomial basis matrix
        RowMatrix<Scalar> Xp(npts, Ny);
        B.computePolyBasis(x, y, Xp);

        // Rotate it into Ylm land
        X0 = Xp * B.A1;

    } else {

        // Compute the limb darkening operator
        UType tmp = B.U1 * u;
        Vector<Scalar> pu = tmp * pi<Scalar>() * 
            (B.rT.segment(0, (udeg + 1) * (udeg + 1)) * tmp).cwiseInverse();
        Matrix<Scalar> L;
        Vector<Matrix<Scalar>> dLdp; // not used
        computePolynomialProductMatrix<false>(udeg, pu, L, dLdp);

        // Compute the polynomial basis matrix
        RowMatrix<Scalar> XLD(npts, N);
        B.computePolyBasis(x, y, XLD);

        // Rotate it into Ylm land
        X0 = XLD * L * B.A1.block(0, 0, Ny, Ny);

    }

    // Our full matrix, f = X . y
    X.resize(npts * ntimes, Ny * Nt);

    // Apply the time evolution (moving source, changing theta, or Taylor)
    Scalar theta_cache = NAN;
    Scalar sx_cache = NAN,
           sy_cache = NAN,
           sz_cache = NAN;
    RowVector<Scalar> xrot, xrot2, yrot, yterm, z, I, Ones;
    if (S::Reflected) {
        Ones = RowVector<Scalar>::Ones(x.cols());
        RowVector<Scalar> x2 = x.array().square();
        RowVector<Scalar> y2 = y.array().square();
        z = (Ones - x2 - y2).cwiseSqrt();
    }
    for (size_t n = 0; n < ntimes; ++n) {

        // Rotate the map
        RowMatrix<Scalar> XR(npts, Ny);
        RowMatrix<Scalar> X0R(npts, Ny);
        RowMatrix<Scalar> X0RR(npts, Ny);
        W.leftMultiplyRZetaInv(X0, X0R);
        Scalar theta_rad = theta(n) * radian;
        if (theta_rad != theta_cache) {
            theta_cache = theta_rad;
            W.compute(cos(theta_rad), sin(theta_rad));
            W.leftMultiplyRz(X0R, X0RR);
            W.leftMultiplyRZeta(X0RR, XR);
        }
        X.block(npts * n, 0, npts, Ny) = XR;

        // Illuminate the map
        if (S::Reflected) {

            // Get the source vector components
            Scalar sx = source(n, 0);
            Scalar sy = source(n, 1);
            Scalar sz = source(n, 2);

            if ((sx != sx_cache) || (sy != sy_cache) || (sz != sz_cache)) {

                // Update the cache
                sx_cache = sx;
                sy_cache = sy;
                sz_cache = sz;

                // Compute the terminator curve
                Scalar b = -sz;
                Scalar invsr = Scalar(1.0) / sqrt(sx * sx + sy * sy);
                Scalar cosw = sy * invsr;
                Scalar sinw = -sx * invsr;
                xrot = x * cosw + y * sinw;
                xrot2 = xrot.array().square();
                yrot = -x * sinw + y * cosw;
                yterm = b * (Ones - xrot2).cwiseSqrt();

                // Compute the illumination
                I = sqrt(Scalar(1.0) - Scalar(b * b)) * yrot - b * z;
                I = (yrot.array() > yterm.array()).select(I, 0.0);

            }
            X.block(npts * n, 0, npts, Ny) = 
                X.block(npts * n, 0, npts, Ny).array().colwise() * 
                    I.transpose().array();
        }

        // Apply the Taylor expansion
        if (std::is_same<S, Temporal<Scalar, S::Reflected>>::value) {
            for (int i = 0; i < Nt; ++i) {
                X.block(npts * n, i * Ny, npts, Ny) = 
                    X.block(npts * n, 0, npts, Ny) * taylor(n, i);
            }
        }

    }

}