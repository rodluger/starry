/**
Check if the total degree of the map is valid.

*/
inline void checkDegree_ () 
{
    if (y_deg + u_deg > lmax) {
        cache.reset();
        y.setZero();
        y_deg = 0;
        u.setZero();
        setU0_();
        u_deg = 0;
        throw errors::ValueError("Degree of the limb-darkened "
                                 "map exceeds `lmax`. All "
                                 "coefficients have been reset.");
    }
}

//! Compute the degree of the Ylm map.
inline void computeDegreeY_ () 
{
    if (cache.compute_degree_y) {
        y_deg = 0;
        for (int l = lmax; l >= 0; --l) {
            if ((y.block(l * l, 0, 2 * l + 1, ncoly).array() 
                    != 0.0).any()) {
                y_deg = l;
                break;
            }
        }
        checkDegree_();
        cache.compute_degree_y = false;
    }
}

//! Compute the degree of the Ul map.
inline void computeDegreeU_ () 
{
    if (cache.compute_degree_u) {
        u_deg = 0;
        for (int l = lmax; l > 0; --l) {
            if (u.row(l).any()) {
                u_deg = l;
                break;
            }
        }
        checkDegree_();
        cache.compute_degree_u = false;
    }
}

//! Compute the change of basis matrix from Ylms to pixels
inline void computeP_ (int res) {
    if (cache.compute_P || (cache.res != res)) {
        B.computePolyMatrix(res, cache.P);
        cache.res = res;
        cache.compute_P = false;
    }
}

//! Compute the zeta frame transform for Ylm rotations
inline void computeWigner_ () {
    if (cache.compute_Zeta) {
        W.updateZeta();
        W.updateYZeta();
        cache.compute_Zeta = false;
        cache.compute_YZeta = false;
    } else if (cache.compute_YZeta) {
        W.updateYZeta();
        cache.compute_YZeta = false;
    }
}

/**
Compute the Agol `c` basis and its derivative.
These are both normalized such that the total
unobscured flux is **unity**.

*/
inline void computeAgolGBasis_ () {
    if (cache.compute_g) {
        limbdark::computeAgolGBasis_(u, cache.g, cache.DgDu);
        normalizeAgolGBasis_(cache.g, cache.DgDu);
        cache.compute_g = false;
    }
}

/**
Rotate an arbitrary map vector in place
given an axis and an angle. If `col = -1`,
rotate all columns of the map, otherwise
rotate only the column with index `col`.

*/
inline void rotateByAxisAngle_ (
    const UnitVector<Scalar>& axis_,
    const Scalar& costheta,
    const Scalar& sintheta,
    YType& y_,
    int col=-1
) {
    Scalar tol = 10 * mach_eps<Scalar>();
    Scalar cosalpha, sinalpha, cosbeta, sinbeta, cosgamma, singamma;
    rotation::axisAngleToEuler(
        axis_(0), axis_(1), costheta, sintheta, tol,
        cosalpha, sinalpha, cosbeta, sinbeta, 
        cosgamma, singamma);
    rotation::rotar(
        lmax, cosalpha, sinalpha, 
        cosbeta, sinbeta, 
        cosgamma, singamma, tol, 
        cache.EulerD, cache.EulerR);
    if (col == -1) {
        for (int l = 0; l < lmax + 1; ++l) {
            y_.block(l * l, 0, 2 * l + 1, ncoly) =
                cache.EulerR[l] * y_.block(l * l, 0, 2 * l + 1, ncoly);
        }
    } else {
        for (int l = 0; l < lmax + 1; ++l) {
            y_.block(l * l, col, 2 * l + 1, 1) =
                cache.EulerR[l] * y_.block(l * l, col, 2 * l + 1, 1);
        }
    }
}

/**
Generate a random isotropic map with a given power spectrum.

*/
template <class V>
inline void random_ (
    const Vector<Scalar>& power,
    const V& seed,
    int col
) {
    int lmax_ = power.size() - 1;
    if (lmax_ > lmax) 
        lmax_ = lmax;
    int N_ = (lmax_ + 1) * (lmax_ + 1);

    // Generate N_ standard normal variables
    std::mt19937 gen(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    Vector<Scalar> vec(N_);
    for (int n = 0; n < N_; ++n)
        vec(n) = static_cast<Scalar>(normal(gen));

    // Zero degree
    vec(0) = sqrt(abs(power(0)));

    // Higher degrees
    for (int l = 1; l < lmax_ + 1; ++l) {
        vec.segment(l * l, 2 * l + 1) *=
            sqrt(abs(power(l)) / vec.segment(l * l, 2 * l + 1).squaredNorm());
    }

    // Set the vector
    if (col == -1) {
        y.block(0, 0, N_, ncoly) = vec.replicate(1, ncoly);
    } else {
        y.block(0, col, N_, 1) = vec;
    }

    cache.yChanged();
}

/**
Compute the Taylor expansion basis at a point in time.
Static specialization (does nothing).

*/
template <typename U=S>
inline IsDefaultOrSpectral<U, void> computeTaylor_ (
    const Scalar & t
) {
}

/**
Compute the Taylor expansion basis at a point in time.
Temporal specialization.

*/
template <typename U=S>
inline IsTemporal<U, void> computeTaylor_ (
    const Scalar & t
) {
    if (t != cache.taylort) {
        for (int n = 1; n < ncoly; ++n)
            taylor(n) = taylor(n - 1) * t / n;
        cache.taylort = t;
    }
}

/**
Temporal contraction operation for static maps: 
does nothing, and returns a reference to the 
original map.

*/
template <typename U=S, typename T1>
inline IsDefaultOrSpectral<U, MatrixBase<T1>&> contract_ (
    MatrixBase<T1> const & mat
) {
    return MBCAST(mat, T1);
}

/**
Contracts a temporal map by dotting the map matrix with the
Taylor expansion basis.

*/
template <typename U=S>
inline IsTemporal<U, Vector<Scalar>> contract_ (
    const Matrix<Scalar> & mat
) {
    return mat * taylor;
}

/**
Set the zeroth order limb darkening coefficient.
This is a **constant** whose value ensures that
I(mu = 0) / I0 = 1.

*/
template <typename U=S>
inline IsDefaultOrSpectral<U, void> setU0_ () {
    u.row(0).setConstant(-1.0);
}

/**
Set the zeroth order limb darkening coefficient
for a temporal map. All derivatives are set to
zero.

*/
template <typename U=S>
inline IsTemporal<U, void> setU0_ () {
    u.row(0).setZero();
    u(0, 0) = -1.0;
}

/**
Rotate the map by an angle `theta` and store
the result in `cache.Ry`. Optionally compute
and cache the Wigner rotation matrices and
their derivatives. Static map specialization.

*/
template <typename U=S>
inline IsDefaultOrSpectral<U, void> rotateIntoCache_ (
    const Scalar& theta,
    bool compute_matrices=false
) 
{
    Scalar theta_rad = theta * radian;
    computeWigner_();
    if ((!compute_matrices) && (cache.theta != theta)) {
        W.rotate(cos(theta_rad), sin(theta_rad), cache.Ry);
        cache.theta = theta;
    } else if (compute_matrices && (cache.theta_with_grad != theta)) {
        W.compute(cos(theta_rad), sin(theta_rad));
        for (int l = 0; l < lmax + 1; ++l) {
            cache.Ry.block(l * l, 0, 2 * l + 1, ncoly) =
                W.R[l] * y.block(l * l, 0, 2 * l + 1, ncoly);
            cache.DRDthetay.block(l * l, 0, 2 * l + 1, ncoly) =
                W.DRDtheta[l] * y.block(l * l, 0, 2 * l + 1, ncoly);
        }
        cache.theta_with_grad = theta;
    }
}

/**
Rotate the map by an angle `theta` and store
the result in `cache.Ry`. Optionally compute
and cache the Wigner rotation matrices and
their derivatives. Temporal map specialization.

*/
template <typename U=S>
inline IsTemporal<U, void> rotateIntoCache_ (
    const Scalar& theta,
    bool compute_matrices=false
) 
{    
    Scalar theta_rad = theta * radian;
    computeWigner_();
    if (!compute_matrices) {
        if (cache.theta != theta) {
            W.rotate(cos(theta_rad), sin(theta_rad), cache.RY);
            cache.theta = theta;
        }
        cache.Ry = contract_(cache.RY);
    } else { 
        if (cache.theta_with_grad != theta) {
            W.compute(cos(theta_rad), sin(theta_rad));
            for (int l = 0; l < lmax + 1; ++l) {
                cache.RY.block(l * l, 0, 2 * l + 1, ncoly) =
                    W.R[l] * y.block(l * l, 0, 2 * l + 1, ncoly);
            }
            cache.theta_with_grad = theta;
        }
        cache.Ry = contract_(cache.RY);
        for (int l = 0; l < lmax + 1; ++l) {
            cache.DRDthetay.block(l * l, 0, 2 * l + 1, nflx) =
                contract_(W.DRDtheta[l] * y.block(l * l, 0, 2 * l + 1, ncoly));
        }
    }
}

/**
Normalize the Agol g basis and its derivatives. 
Default map specialization.

*/
template <typename U=S, typename T1, typename T2>
inline IsDefault<U, void> normalizeAgolGBasis_ (
    MatrixBase<T1> const & g,
    MatrixBase<T2> const & DgDu
) {
    // The total flux is given by `y00 * (s . g)`
    Scalar norm;
    if (likely(lmax > 0))
       norm = Scalar(1.0) / (pi<Scalar>() * (g(0) + 2.0 * g(1) / 3.0));
    else
        norm = Scalar(1.0) / (pi<Scalar>() * g(0));
    MBCAST(g, T1) = g * norm;
    MBCAST(DgDu, T2) = DgDu * norm;
}

/**
Normalize the Agol g basis and its derivatives. 
Spectral map specialization.

*/
template <typename U=S, typename T1, typename T2>
inline IsSpectral<U, void> normalizeAgolGBasis_ (
    MatrixBase<T1> const & g,
    MatrixBase<T2> const & DgDu
) {
    // The total flux is given by `y00 * (s . g)`
    Scalar norm;
    for (int n = 0; n < ncoly; ++n) {
        if (likely(lmax > 0))
            norm = Scalar(1.0) / (pi<Scalar>() * (g(0, n) + 2.0 * g(1, n) / 3.0));
        else
            norm = Scalar(1.0) / (pi<Scalar>() * g(0, n));
        MBCAST(g, T1).col(n) = g.col(n) * norm;
        MBCAST(DgDu, T2).block(n * lmax, 0, lmax, lmax + 1) = 
            DgDu.block(n * lmax, 0, lmax, lmax + 1) * norm;
    }
}

/**
Normalize the Agol g basis and its derivatives. 
Temporal map specialization.

*/
template <typename U=S, typename T1, typename T2>
inline IsTemporal<U, void> normalizeAgolGBasis_ (
    MatrixBase<T1> const & g,
    MatrixBase<T2> const & DgDu
) {
    // The total flux is given by `y00 * (s . c)`
    Scalar norm;
    if (likely(lmax > 0))
       norm = Scalar(1.0) / (pi<Scalar>() * (g(0) + 2.0 * g(1) / 3.0));
    else
        norm = Scalar(1.0) / (pi<Scalar>() * g(0));
    MBCAST(g, T1) = g * norm;
    MBCAST(DgDu, T2) = DgDu * norm;
}

/** 
Compute the limb darkening polynomial `p` and optionally
propagate gradients. Default specialization.

*/
template <bool GRADIENT=false, typename U=S>
inline IsDefault<U, void> computeLDPolynomial_ () {
    if (!GRADIENT) {
        if (cache.compute_p) {
            UType tmp = B.U1 * u;
            Scalar norm = pi<Scalar>() * y(0) / B.rT.dot(tmp);
            cache.p = tmp * norm;
            cache.compute_p = false;
        }
    } else {
        if (cache.compute_p_grad) {
            UType tmp = B.U1 * u;
            Scalar rTU1u = B.rT.dot(tmp);
            Scalar norm0 = pi<Scalar>() / rTU1u;
            Scalar norm = norm0 * y(0);
            RowVector<Scalar> dnormdu = -(norm / rTU1u) * B.rTU1;
            cache.p = tmp * norm;
            cache.DpuDu = B.U1 * norm + tmp * dnormdu;
            cache.DpuDy0 = tmp * norm0;
            cache.compute_p_grad = false;
        }
    }
}

/** 
Compute the limb darkening polynomial `p` and optionally
propagate gradients. Spectral specialization.

*/
template <bool GRADIENT=false, typename U=S>
inline IsSpectral<U, void> computeLDPolynomial_ () {
    if (!GRADIENT) {
        if (cache.compute_p) {
            UType tmp = B.U1 * u;
            UCoeffType norm = pi<Scalar>() * 
                              y.row(0).cwiseQuotient(B.rT * tmp);
            cache.p = tmp.array().rowwise() * norm.array();
            cache.compute_p = false;
        }
    } else {
        if (cache.compute_p_grad) {
            UType tmp = B.U1 * u;
            UCoeffType rTU1u = B.rT * tmp;
            for (int i = 0; i < ncolu; ++i) {
                Scalar norm0 = pi<Scalar>() / rTU1u(i);
                Scalar norm = norm0 * y(0, i);
                RowVector<Scalar> dnormdu = -(norm / rTU1u(i)) * B.rTU1;
                cache.p.col(i) = tmp.col(i) * norm;
                cache.DpuDu[i] = B.U1 * norm + tmp.col(i) * dnormdu;
                cache.DpuDy0.col(i) = tmp.col(i) * norm0;
            }
            cache.compute_p_grad = false;
        }
    }
}

/** 
Compute the limb darkening polynomial `p` and optionally
propagate gradients. Temporal specialization.

*/
template <bool GRADIENT=false, typename U=S>
inline IsTemporal<U, void> computeLDPolynomial_ () {
    if (!GRADIENT) {
        if (cache.compute_p) {
            UType tmp = B.U1 * u;
            Scalar norm = pi<Scalar>() * contract_(y.row(0))(0) / B.rT.dot(tmp);
            cache.p = tmp * norm;
            cache.compute_p = false;
        }
    } else {
        if (cache.compute_p_grad) {
            UType tmp = B.U1 * u;
            Scalar rTU1u = B.rT.dot(tmp);
            Scalar norm0 = pi<Scalar>() / rTU1u;
            Scalar norm = norm0 * contract_(y.row(0))(0);
            RowVector<Scalar> dnormdu = -(norm / rTU1u) * B.rTU1;
            cache.p = tmp * norm;
            cache.DpuDu = B.U1 * norm + tmp * dnormdu;
            cache.DpuDy0 = tmp * norm0;
            cache.compute_p_grad = false;
        }
    }
}

/**
Limb-darken a polynomial map. Static specialization.

*/
template <typename U=S>
inline IsDefaultOrSpectral<U, void> limbDarken_ (
    const YType& poly, 
    YType& poly_ld
) {
    computeLDPolynomial_();
    basis::polymul(y_deg, poly, u_deg, cache.p, lmax, poly_ld);
}

/**
Limb-darken a polynomial map. Temporal specialization;
here we limb-darken the *contracted* map.

*/
template <typename U=S>
inline IsTemporal<U, void> limbDarken_ (
    const Vector<Scalar>& poly, 
    Vector<Scalar>& poly_ld
) {
    computeLDPolynomial_();
    basis::polymul(y_deg, poly, u_deg, cache.p, lmax, poly_ld);
}

/**
Limb-darken a polynomial map and compute the
gradient of the resulting map with respect to the input
polynomial map and the input limb-darkening map.
Default specialization.

*/
template <bool OCCULTATION=false, typename U=S>
inline IsDefault<U, void> limbDarkenWithGradient_ (
    const YType& poly, 
    YType& poly_ld
) {
    // Compute the limb darkening polynomial
    computeLDPolynomial_<true>();

    // Multiply the polynomials
    if (OCCULTATION)
        basis::polymul(y_deg, poly, u_deg, cache.p, lmax, poly_ld, cache.sTA2,
                       cache.vTDpupyDpy, cache.vTDpupyDpu);
    else
        basis::polymul(y_deg, poly, u_deg, cache.p, lmax, poly_ld, B.rT,
                       cache.vTDpupyDpy, cache.vTDpupyDpu);

    // Propagate the gradient to d(polynomial) / du
    // and d(polynomial) / dy 
    cache.vTDpupyDu = cache.vTDpupyDpu * cache.DpuDu;
    cache.vTDpupyDpyA1 = cache.vTDpupyDpy * B.A1;

    if (OCCULTATION) {
        W.leftMultiplyRz(cache.vTDpupyDpyA1, cache.vTDpupyDpyA1R);
        for (int l = 0; l < lmax + 1; ++l)
            cache.vTDpupyDpyA1RR.segment(l * l, 2 * l + 1) = 
                cache.vTDpupyDpyA1R.segment(l * l, 2 * l + 1) * W.R[l];
        cache.vTDpupyDy = cache.vTDpupyDpyA1RR.transpose();
    } else {
        for (int l = 0; l < lmax + 1; ++l)
            cache.vTDpupyDpyA1R.segment(l * l, 2 * l + 1) =
                cache.vTDpupyDpyA1.segment(l * l, 2 * l + 1) * W.R[l];
        cache.vTDpupyDy = cache.vTDpupyDpyA1R.transpose();
    }
    
    cache.vTDpupyDy(0) += (cache.vTDpupyDpu * cache.DpuDy0)(0);
}

/**
Limb-darken a polynomial map and compute the
gradient of the resulting map with respect to the input
polynomial map and the input limb-darkening map.
Spectral specialization.

*/
template <bool OCCULTATION=false, typename U=S>
inline IsSpectral<U, void> limbDarkenWithGradient_ (
    const YType& poly, 
    YType& poly_ld
) {
    // Compute the limb darkening polynomial
    computeLDPolynomial_<true>();

    // Multiply the polynomials
    if (OCCULTATION)
        basis::polymul(y_deg, poly, u_deg, cache.p, lmax, poly_ld, cache.sTA2,
                       cache.vTDpupyDpy, cache.vTDpupyDpu);
    else
        basis::polymul(y_deg, poly, u_deg, cache.p, lmax, poly_ld, B.rT,
                       cache.vTDpupyDpy, cache.vTDpupyDpu);

    // Propagate the gradient to d(polynomial) / du
    // and d(polynomial) / dy 
    for (int i = 0; i < ncolu; ++i)
        cache.vTDpupyDu.col(i) = cache.vTDpupyDpu.row(i) * cache.DpuDu[i];
    cache.vTDpupyDpyA1 = cache.vTDpupyDpy * B.A1;
    
    if (OCCULTATION) {
        W.leftMultiplyRz(cache.vTDpupyDpyA1, cache.vTDpupyDpyA1R);
        for (int l = 0; l < lmax + 1; ++l)
            cache.vTDpupyDpyA1RR.block(0, l * l, nflx, 2 * l + 1) =
                cache.vTDpupyDpyA1R.block(0, l * l, nflx, 2 * l + 1) * W.R[l];
        for (int i = 0; i < ncoly; ++i) {
            cache.vTDpupyDy.col(i) = cache.vTDpupyDpyA1RR.row(i);
            cache.vTDpupyDy(0, i) += (cache.vTDpupyDpu.row(i) * 
                                        cache.DpuDy0.col(i))(0);
        }
    } else {
        for (int l = 0; l < lmax + 1; ++l)
            cache.vTDpupyDpyA1R.block(0, l * l, nflx, 2 * l + 1) =
                cache.vTDpupyDpyA1.block(0, l * l, nflx, 2 * l + 1) * W.R[l];
        for (int i = 0; i < ncoly; ++i) {
            cache.vTDpupyDy.col(i) = cache.vTDpupyDpyA1R.row(i);
            cache.vTDpupyDy(0, i) += (cache.vTDpupyDpu.row(i) * 
                                        cache.DpuDy0.col(i))(0);
        }
    }
    
}

/**
Limb-darken a polynomial map and compute the
gradient of the resulting map with respect to the input
polynomial map and the input limb-darkening map.
Temporal specialization.

*/
template <bool OCCULTATION=false, typename U=S>
inline IsTemporal<U, void> limbDarkenWithGradient_ (
    const Vector<Scalar>& poly, 
    Vector<Scalar>& poly_ld
) {
    // Compute the limb darkening polynomial
    computeLDPolynomial_<true>();

    // Multiply the polynomials
    if (OCCULTATION)
        basis::polymul(y_deg, poly, u_deg, cache.p, lmax, poly_ld, cache.sTA2,
                       cache.vTDpupyDpy, cache.vTDpupyDpu);
    else
        basis::polymul(y_deg, poly, u_deg, cache.p, lmax, poly_ld, B.rT,
                       cache.vTDpupyDpy, cache.vTDpupyDpu);
                    
    // Propagate the gradient to d(polynomial) / du
    // and d(polynomial) / dy 
    cache.vTDpupyDu = cache.vTDpupyDpu * cache.DpuDu;
    cache.vTDpupyDpyA1 = cache.vTDpupyDpy * B.A1;
    
    if (OCCULTATION) {
        W.leftMultiplyRz(cache.vTDpupyDpyA1, cache.vTDpupyDpyA1R);
        for (int l = 0; l < lmax + 1; ++l)
            cache.vTDpupyDpyA1RR.segment(l * l, 2 * l + 1) =
                cache.vTDpupyDpyA1R.segment(l * l, 2 * l + 1) * W.R[l];
        cache.vTDpupyDy = cache.vTDpupyDpyA1RR.replicate(ncoly, 1).transpose();
    } else {
        for (int l = 0; l < lmax + 1; ++l)
            cache.vTDpupyDpyA1R.segment(l * l, 2 * l + 1) =
                cache.vTDpupyDpyA1.segment(l * l, 2 * l + 1) * W.R[l];
        cache.vTDpupyDy = cache.vTDpupyDpyA1R.replicate(ncoly, 1).transpose();
    }
    
    cache.vTDpupyDy.row(0) += (cache.vTDpupyDpu * cache.DpuDy0)
                                .replicate(ncoly, 1).transpose();
}