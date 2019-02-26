/**

*/
template <
    typename U=S, 
    typename=IsEmitted<U>
>
inline void computeLinearIntensityModelInternal (
    const Scalar& theta, 
    const RowMatrix<Scalar>& x_, 
    const RowMatrix<Scalar>& y_,
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
    X.resize(npts, Ny * Nt);
    
    if (udeg == 0) {

        // Compute the polynomial basis matrix
        B.computePolyBasis(x, y, X);

        // Rotate it into Ylm land
        X = X * B.A1;

    } else {

        // Compute the limb darkening operator
        UType tmp = B.U1 * u;
        Scalar norm = pi<Scalar>() / B.rT.segment(0, (udeg + 1) * (udeg + 1)).dot(tmp);
        Vector<Scalar> pu = tmp * norm;
        Matrix<Scalar> L;
        Vector<Matrix<Scalar>> dLdp; // not used
        computePolynomialProductMatrix<false>(udeg, pu, L, dLdp);

        // Compute the polynomial basis matrix
        RowMatrix<Scalar> XLD(npts, N);
        B.computePolyBasis(x, y, XLD);

        // Rotate it into Ylm land
        X = XLD * L * B.A1.block(0, 0, Ny, Ny);

    }
    
    // Dot it into the rotation matrix
    if (theta != 0) {
        RowMatrix<Scalar> XR(npts, Ny);
        Scalar theta_rad = theta * radian;
        W.compute(cos(theta_rad), sin(theta_rad));
        W.leftMultiplyR(X, XR);
        X = XR;
    }

}

/**

*/
template <
    typename U=S, 
    typename=IsReflected<U>
>
inline void computeLinearIntensityModelInternal (
    const Scalar& theta, 
    const Vector<Scalar>& x, 
    const Vector<Scalar>& y,
    const UnitVector<Scalar>& source,
    RowMatrix<Scalar>& A
) {

    // \todo

}