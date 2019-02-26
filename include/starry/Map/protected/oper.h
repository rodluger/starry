/**
Check if the total degree of the map is valid.

*/


//! Compute the illumination matrix
inline void computeI (
    int res, 
    const UnitVector<Scalar>& source
) {
    // \todo B.computeIlluminationMatrix(res, source, I);
}

/**
Rotate an arbitrary map vector in place
given an axis and an angle. If `col = -1`,
rotate all columns of the map, otherwise
rotate only the column with index `col`.

*/
inline void rotateByAxisAngle (
    const UnitVector<Scalar>& axis_,
    const Scalar& costheta,
    const Scalar& sintheta,
    YType& y_,
    int col=-1
) {
    Scalar tol = 10 * mach_eps<Scalar>();
    Scalar cosalpha, sinalpha, cosbeta, sinbeta, cosgamma, singamma;
    wigner::axisAngleToEuler(
        axis_(0), axis_(1), costheta, sintheta, tol,
        cosalpha, sinalpha, cosbeta, sinbeta, 
        cosgamma, singamma);
    wigner::rotar(
        ydeg, cosalpha, sinalpha, 
        cosbeta, sinbeta, 
        cosgamma, singamma, tol, 
        data.EulerD, data.EulerR);
    if (col == -1) {
        for (int l = 0; l < ydeg + 1; ++l) {
            y_.block(l * l, 0, 2 * l + 1, Nw) =
                data.EulerR[l] * y_.block(l * l, 0, 2 * l + 1, Nw);
        }
    } else {
        for (int l = 0; l < ydeg + 1; ++l) {
            y_.block(l * l, col, 2 * l + 1, 1) =
                data.EulerR[l] * y_.block(l * l, col, 2 * l + 1, 1);
        }
    }
}

/**
Generate a random isotropic map with a given power spectrum.

*/
template <class V>
inline void randomInternal (
    const Vector<Scalar>& power,
    const V& seed,
    int col
) {
    int lmax_ = power.size() - 1;
    if (lmax_ > ydeg) 
        lmax_ = ydeg;
    int N_ = (lmax_ + 1) * (lmax_ + 1);

    // Generate N_ standard normal variables
    std::mt19937 gen(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    Vector<Scalar> vec(N_);
    for (int n = 0; n < N_; ++n)
        vec(n) = static_cast<Scalar>(normal(gen));

    // Weight them by sqrt(power) at each l
    for (int l = 0; l < lmax_ + 1; ++l) {
        vec.segment(l * l, 2 * l + 1) *= sqrt(abs(power(l)));
    }

    // Set the vector
    if (col == -1) {
        y.block(0, 0, N_, Nw) = vec.replicate(1, Nw);
    } else {
        y.block(0, col, N_, 1) = vec;
    }

}


/**
Compute the Taylor expansion basis at a vector of times.
Temporal specialization.

*/
template <typename U=S>
inline IsTemporal<U, void> computeTaylor (
    const Vector<Scalar> & t
) {
    taylor.resize(t.rows(), Nt);
    taylor.col(0).setOnes();
    for (int i = 1; i < Nt; ++i)
        taylor.col(i) = taylor.col(i - 1).cwiseProduct(t) / i;
}

/**
Set the zeroth order limb darkening coefficient.
This is a **constant** whose value ensures that
I(mu = 0) / I0 = 1.

*/
template <typename U=S>
inline IsDefaultOrSpectral<U, void> setU0 () {
    u.row(0).setConstant(-1.0);
}

/**
Set the zeroth order limb darkening coefficient
for a temporal map. All derivatives are set to
zero.

*/
template <typename U=S>
inline IsTemporal<U, void> setU0 () {
    u.row(0).setZero();
    u(0, 0) = -1.0;
}

/**


*/
template <bool GRADIENT=false>
inline void computePolynomialProductMatrix (
    const int plmax, 
    const Vector<Scalar>& p,
    Matrix<Scalar>& M,
    Vector<Matrix<Scalar>>& dMdp
) {
    bool odd1;
    int l, n;
    int n1 = 0, n2 = 0;
    M.setZero((plmax + ydeg + 1) * (plmax + ydeg + 1), Ny);
    if (GRADIENT) {
        dMdp.resize((plmax + 1) * (plmax + 1));
        for (n = 0; n < (plmax + 1) * (plmax + 1); ++n)
            dMdp(n).setZero((plmax + ydeg + 1) * (plmax + ydeg + 1), Ny);
    }
    for (int l1 = 0; l1 < ydeg + 1; ++l1) {
        for (int m1 = -l1; m1 < l1 + 1; ++m1) {
            odd1 = (l1 + m1) % 2 == 0 ? false : true;
            n2 = 0;
            for (int l2 = 0; l2 < plmax + 1; ++l2) {
                for (int m2 = -l2; m2 < l2 + 1; ++m2) {
                    if (p(n2)) {
                        l = l1 + l2;
                        n = l * l + l + m1 + m2;
                        if (odd1 && ((l2 + m2) % 2 != 0)) {
                            M(n - 4 * l + 2, n1) += p(n2);
                            M(n - 2, n1) -= p(n2);
                            M(n + 2, n1) -= p(n2);
                            if (GRADIENT) {
                                dMdp[n2](n - 4 * l + 2, n1) += 1;
                                dMdp[n2](n - 2, n1) -= 1;
                                dMdp[n2](n + 2, n1) -= 1;
                            }
                        } else {
                            M(n, n1) += p(n2);
                            if (GRADIENT) {
                                dMdp[n2](n, n1) += 1;
                            }
                        }
                    }
                    ++n2;
                }
            }
            ++n1;
        }
    }  
}