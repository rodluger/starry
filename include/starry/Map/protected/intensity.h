/**

*/
template <bool CONTRACT_Y=false>
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
    if (S::Temporal) {
        CHECK_ROWS(taylor, theta.rows());
    }

    // Flatten x and y
    Eigen::Map<const RowVector<Scalar>> x(x_.data(), x_.size());
    Eigen::Map<const RowVector<Scalar>> y(y_.data(), y_.size());
    size_t npts = x.size();

    // Convert to radians
    Vector<Scalar> theta_rad = theta * radian;

    // Number of time points
    size_t ntimes = theta.size();

    // Compute the polynomial basis matrix
    if ((Xp.rows() != long(npts)) || (x - x_cache).any() || (y - y_cache).any()) {
        Xp.resize(npts, Ny);
        B.computePolyBasis(x, y, Xp);
        X0 = Xp * B.A1;
        x_cache = x;
        y_cache = y;
    }
    
    // Apply limb darkening
    if (udeg > 0) {

        // Compute the limb darkening operator
        UType tmp = B.U1 * u;
        Scalar norm = Scalar(1.0) / B.rT.segment(0, (udeg + 1) * (udeg + 1)).dot(tmp);
        UType pu = tmp * norm * pi<Scalar>();
        Matrix<Scalar> L;
        Vector<Matrix<Scalar>> dLdp; // not used
        computePolynomialProductMatrix<false>(udeg, pu, L, dLdp);
        
        // Rotate it into Ylm land
        X0 = Xp * L * B.A1.block(0, 0, Ny, Ny);
    }

    if (CONTRACT_Y) {
        // The design matrix
        X.resize(npts * ntimes, Nw);
    } else {
        // The actual intensity
        X.resize(npts * ntimes, Ny * Nt);
    }

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
    RowMatrix<Scalar> XR, X0R, X0RR; 
    YType Ry;
    RowMatrix<Scalar> X0Ry;
    if (CONTRACT_Y) {
        Ry.resize(Ny, Nw);
        X0Ry.resize(npts, Nw);
    } else {
        XR.resize(npts, Ny);
        X0R.resize(npts, Ny);
        X0RR.resize(npts, Ny);
    }
    for (size_t n = 0; n < ntimes; ++n) {

        if (CONTRACT_Y) {
            // Rotate the map
            if (theta_rad(n) != theta_cache) { // todo broken for temporal
                theta_cache = theta_rad(n);
                W.compute(cos(theta_rad(n)), sin(theta_rad(n)));
                if (!S::Temporal) {
                    W.leftMultiplyR(y.transpose(), Ry.transpose());
                    X0Ry = X0 * Ry;
                }
            }
            // Temporal expansion
            if (S::Temporal) {
                X0Ry.setZero();
                for (int i = 0; i < Nt; ++i) {
                    W.leftMultiplyR(y.block(i * Ny, 0, Ny, Nw).transpose(), Ry.transpose());
                    X0Ry += X0 * Ry * taylor(n, i);
                }
            }
            X.block(npts * n, 0, npts, Nw) = X0Ry;
        } else {
            // Rotate the design matrix
            W.leftMultiplyRZetaInv(X0, X0R);
            if (theta_rad(n) != theta_cache) {
                theta_cache = theta_rad(n);
                W.compute(cos(theta_rad(n)), sin(theta_rad(n)));
                W.leftMultiplyRz(X0R, X0RR);
                W.leftMultiplyRZeta(X0RR, XR);
            }
            X.block(npts * n, 0, npts, Ny) = XR;
        }

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
            if (CONTRACT_Y) {
                // TODO!
                throw std::runtime_error("Not yet implemented."); 
            } else {
                X.block(npts * n, 0, npts, Ny) = 
                    X.block(npts * n, 0, npts, Ny).array().colwise() * 
                        I.transpose().array();
            }
        }

        // Apply the Taylor expansion
        if ((!CONTRACT_Y) && (S::Temporal)) {
            for (int i = 0; i < Nt; ++i) {
                X.block(npts * n, i * Ny, npts, Ny) = 
                    X.block(npts * n, 0, npts, Ny) * taylor(n, i);
            }
        }

    }

}