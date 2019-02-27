/**

*/
inline void computeLinearIntensityModelInternal (
    const Scalar& theta, 
    const RowMatrix<Scalar>& x_, 
    const RowMatrix<Scalar>& y_,
    const UnitVector<Scalar>& source,
    RowMatrix<Scalar>& X
) {

    // Shape checks
    size_t nrows = x_.rows();
    size_t ncols = x_.cols();
    CHECK_SHAPE(y_, nrows, ncols);

    // Flatten x and y
    Eigen::Map<const RowVector<Scalar>> x(x_.data(), x_.size());
    Eigen::Map<const RowVector<Scalar>> y(y_.data(), y_.size());
    size_t npts = x.size();

    // Our model matrix, f = X . y
    X.resize(npts, Ny);
    
    if (udeg == 0) {

        // Compute the polynomial basis matrix
        B.computePolyBasis(x, y, X);

        // Rotate it into Ylm land
        X = X * B.A1;

    } else {

        // Compute the limb darkening operator
        UType tmp = B.U1 * u;
        Vector<Scalar> pu = tmp * pi<Scalar>() * (B.rT.segment(0, (udeg + 1) * (udeg + 1)) * tmp).cwiseInverse();
        Matrix<Scalar> L;
        Vector<Matrix<Scalar>> dLdp; // not used
        computePolynomialProductMatrix<false>(udeg, pu, L, dLdp);

        // Compute the polynomial basis matrix
        RowMatrix<Scalar> XLD(npts, N);
        B.computePolyBasis(x, y, XLD);

        // Rotate it into Ylm land
        X = XLD * L * B.A1.block(0, 0, Ny, Ny);

    }
    
    // Illuminate the map if we're modeling reflected light
    if (S::Reflected) {

        // Get the source vector components
        Scalar sx = source(0);
        Scalar sy = source(1);
        Scalar sz = source(2);

        // Compute the terminator curve
        Scalar b = -sz;
        Scalar invsr = Scalar(1.0) / sqrt(sx * sx + sy * sy);
        Scalar cosw = sy * invsr;
        Scalar sinw = -sx * invsr;
        RowVector<Scalar> xrot = x * cosw + y * sinw;
        RowVector<Scalar> yrot = -x * sinw + y * cosw;
        RowVector<Scalar> Ones = RowVector<Scalar>::Ones(nrows, ncols);
        RowVector<Scalar> yterm = b * (Ones - (xrot * xrot).eval()).cwiseSqrt();
        
        // Compute the illumination matrix
        RowVector<Scalar> z = (Ones - (x * x).eval() + (y * y).eval()).cwiseSqrt();
        RowVector<Scalar> I = sqrt(Scalar(1.0) - Scalar(b * b)) * yrot - b * z;
        I = (yrot.array() > yterm.array()).select(I, 0.0);

        // Multiply it in
        X = X.array().colwise() * I.transpose().array();

    }

    // Dot it into the rotation matrix
    if (theta != 0) {
        RowMatrix<Scalar> XR(npts, Ny);
        Scalar theta_rad = theta * radian;
        W.compute(cos(theta_rad), sin(theta_rad));
        W.leftMultiplyR(X, XR);
        X = XR;
    }

    // Apply the Taylor expansion if we're modeling temporal variability
    if (std::is_same<S, Temporal<Scalar, S::Reflected>>::value) {
        RowMatrix<Scalar> X0 = X;
        X.resize(npts, Ny * Nt);
        for (int i = 0; i < Nt; ++i) {
            X.block(0, i * Ny, npts, Ny) = X0 * taylor(0, i);
        }
    }

}