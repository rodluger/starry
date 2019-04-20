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
    Eigen::Map<const RowVector<Scalar>> xv(x_.data(), x_.size());
    Eigen::Map<const RowVector<Scalar>> yv(y_.data(), y_.size());
    size_t npts = xv.size();

    // Convert to radians
    Vector<Scalar> theta_rad = theta * radian;

    // Number of time points
    size_t ntimes = theta.size();

    // Compute the polynomial basis matrix
    if ((Xp.rows() != long(npts)) || (xv - x_cache).any() || (yv - y_cache).any()) {
        Xp.resize(npts, N);
        B.computePolyBasis(xv, yv, Xp);
        x_cache = xv;
        y_cache = yv;
        if ((udeg == 0) && (fdeg == 0))
            X0 = Xp * B.A1.block(0, 0, N, Ny);
    }
    
    // Apply limb darkening / filter
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

        // Rotate it into Ylm land
        X0 = Xp * L * B.A1.block(0, 0, Ny, Ny);

    }

    if (CONTRACT_Y) {
        // The actual intensity
        X.resize(npts * ntimes, Nw);
    } else {
        // The design matrix
        X.resize(npts * ntimes, Ny * Nt);
    }

    // Apply the time evolution (moving source, changing theta, or Taylor)
    Scalar theta_cache = NAN;
    Scalar sx_cache = NAN,
           sy_cache = NAN,
           sz_cache = NAN;
    RowVector<Scalar> xrot, xrot2, yrot, yterm, z, I, Ones;
    if (S::Reflected) {
        Ones = RowVector<Scalar>::Ones(xv.cols());
        RowVector<Scalar> x2 = xv.array().square();
        RowVector<Scalar> y2 = yv.array().square();
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
            if (theta_rad(n) != theta_cache) {
                theta_cache = theta_rad(n);
                // Note that we're computing the transpose of the rotated map,
                // so we need to do theta --> -theta!
                W.compute(cos(theta_rad(n)), -sin(theta_rad(n)));
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

                Scalar b = -sz;
                if (likely(abs(b) < 1.0)) {
                    // Compute the terminator curve
                    Scalar invsr = Scalar(1.0) / sqrt(sx * sx + sy * sy);
                    Scalar cosw = sy * invsr;
                    Scalar sinw = -sx * invsr;
                    xrot = xv * cosw + yv * sinw;
                    xrot2 = xrot.array().square();
                    yrot = -xv * sinw + yv * cosw;
                    yterm = b * (Ones - xrot2).cwiseSqrt();
                    // Compute the illumination
                    I = sqrt(Scalar(1.0) - Scalar(b * b)) * yrot - b * z;
                    I = (yrot.array() > yterm.array()).select(I, 0.0);
                } else if (b < 0) {
                    // Noon
                    I = z;
                } else {
                    // Midnight
                    I = z * 0.0;
                }

            }
            if (CONTRACT_Y) {
                X.block(npts * n, 0, npts, Nw) = 
                    X.block(npts * n, 0, npts, Nw).array().colwise() * 
                        I.transpose().array();
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