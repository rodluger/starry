/**
Check if the total degree of the map is valid.

*/

/**
Rotate an arbitrary ylm vector in place
given an axis and an angle.

*/
template <typename T1>
inline void rotateByAxisAngle (
    const UnitVector<Scalar>& axis_,
    const Scalar& costheta,
    const Scalar& sintheta,
    MatrixBase<T1>& y_
) {
    Scalar tol = 10 * mach_eps<Scalar>();
    Scalar cosalpha, sinalpha, cosbeta, sinbeta, cosgamma, singamma;
    wigner::axisAngleToEuler(
        axis_(0), axis_(1), costheta, sintheta, tol,
        cosalpha, sinalpha, cosbeta, sinbeta, 
        cosgamma, singamma
    );
    wigner::rotar(
        ydeg, cosalpha, sinalpha, 
        cosbeta, sinbeta, 
        cosgamma, singamma, tol, 
        data.EulerD, data.EulerR
    );
    for (int l = 0; l < ydeg + 1; ++l) {
        y_.block(l * l, 0, 2 * l + 1, y_.cols()) =
            data.EulerR[l] * y_.block(l * l, 0, 2 * l + 1, y_.cols());
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
inline EnableIf<U::Temporal, void> computeTaylor (
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
inline void setU0 () {
    u.row(0).setConstant(-1.0);
}

/**
Set the zeroth order spherical harmonic coefficient.
This is a **constant** fixed at unity.

*/
template <typename U=S>
inline EnableIf<!U::Temporal, void> setY00 () {
    y.row(0).setConstant(1.0);
}

/**
Set the zeroth order spherical harmonic coefficient
for a temporal map. All derivatives are set to
zero.

*/
template <typename U=S>
inline EnableIf<U::Temporal, void> setY00 () {
    y(0) = 1.0;
    for (int i = 1; i < Nt; ++i) {
        y(i * Ny) = 0.0;
    }
}

/**
Set the zeroth order filter spherical harmonic coefficient.
This is a **constant** fixed at unity.

*/
inline void setF00 () {
    f.row(0).setConstant(1.0);
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

template <bool GRADIENT=false>
inline void computePolynomialProduct(
    const int lmax1, 
    const Vector<Scalar>& p1, 
    const int lmax2,
    const Vector<Scalar>& p2, 
    Vector<Scalar>& p1p2,
    Matrix<Scalar>& grad_p1,
    Matrix<Scalar>& grad_p2
) {
    int n1, n2, l1, m1, l2, m2, l, n;
    bool odd1;
    int N1 = (lmax1 + 1) * (lmax1 + 1);
    int N2 = (lmax2 + 1) * (lmax2 + 1);
    int N12 = (lmax1 + lmax2 + 1) * (lmax1 + lmax2 + 1);
    p1p2.setZero(N);
    Scalar mult;
    n1 = 0;
    if (GRADIENT) {
        grad_p1.setZero(N12, N1);
        grad_p2.setZero(N12, N2);
    }
    for (l1 = 0; l1 < lmax1 + 1; ++l1) {
        for (m1 = -l1; m1 < l1 + 1; ++m1) {
            odd1 = (l1 + m1) % 2 == 0 ? false : true;
            n2 = 0;
            for (l2 = 0; l2 < lmax2 + 1; ++l2) {
                for (m2 = -l2; m2 < l2 + 1; ++m2) {
                    l = l1 + l2;
                    n = l * l + l + m1 + m2;
                    mult = p1(n1) * p2(n2);
                    if (odd1 && ((l2 + m2) % 2 != 0)) {
                        p1p2(n - 4 * l + 2) += mult;
                        p1p2(n - 2) -= mult;
                        p1p2(n + 2) -= mult;
                        if (GRADIENT) {
                            grad_p1(n - 4 * l + 2, n1) += p2(n2);
                            grad_p2(n - 4 * l + 2, n2) += p1(n1);
                            grad_p1(n - 2, n1) -= p2(n2);
                            grad_p2(n - 2, n2) -= p1(n1);
                            grad_p1(n + 2, n1) -= p2(n2);
                            grad_p2(n + 2, n2) -= p1(n1);
                        }  
                    } else {
                        p1p2(n) += mult;
                        if (GRADIENT) {
                            grad_p1(n, n1) += p2(n2);
                            grad_p2(n, n2) += p1(n1);
                        }
                    }
                    ++n2;
                }
            }
            ++n1;
        }
    }
}