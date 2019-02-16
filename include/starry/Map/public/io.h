/**
Return a human-readable map string.

*/
std::string info () {
    std::ostringstream os;
    if (std::is_same<S, Default<Scalar, S::Reflected>>::value) {
        os << "<starry.Map("
            << "lmax=" << lmax << ", "
            << "reflected=" << S::Reflected << ", "
            << "multi=" << !std::is_same<Scalar, double>::value
            << ")>";
    } else if (std::is_same<S, Spectral<Scalar, S::Reflected>>::value) {
        os << "<starry.Map("
            << "lmax=" << lmax << ", "
            << "nw=" << ncoly << ", "
            << "reflected=" << S::Reflected << ", "
            << "multi=" << !std::is_same<Scalar, double>::value
            << ")>";
    } else if (std::is_same<S, Temporal<Scalar, S::Reflected>>::value) {
        os << "<starry.Map("
            << "lmax=" << lmax << ", "
            << "nt=" << ncoly << ", "
            << "reflected=" << S::Reflected << ", "
            << "multi=" << !std::is_same<Scalar, double>::value
            << ")>";
    } else {
        // ??
        os << "<starry.Map>";
    }
    return std::string(os.str());
}

/**
Set the full spherical harmonic vector/matrix.

*/
inline void setY (
    const YType& y_
) {
    cache.yChanged();
    if ((y_.rows() == y.rows()) && (y_.cols() == y.cols()))
        y = y_;
    else
        throw errors::ValueError("Dimension mismatch in `y`.");
}

/**
Set the (l, m) row of the spherical harmonic coefficient *matrix* to an
array of coefficients.

*/
inline void setY (
    int l, 
    int m, 
    const Ref<const YCoeffType>& coeff
) {
    cache.yChanged();
    if ((0 <= l) && (l <= lmax) && (-l <= m) && (m <= l))
        y.row(l * l + l + m) = coeff;
    else
        throw errors::IndexError("Invalid value for `l` and/or `m`.");
}

/**
Set the (l, m) index of the spherical harmonic coefficient vector, or
the entire (l, m) row of the spherical harmonic coefficient matrix, to a
single value.

*/
inline void setY (
    int l, 
    int m, 
    const Scalar& coeff
) {
    cache.yChanged();
    if ((0 <= l) && (l <= lmax) && (-l <= m) && (m <= l))
        y.row(l * l + l + m).setConstant(coeff);
    else
        throw errors::IndexError("Invalid value for `l` and/or `m`.");
}

/**
Get the full spherical harmonic vector/matrix

*/
inline const YType getY () const {
    return y;
}

/**
Set the full limb darkening vector/matrix.

*/
inline void setU (
    const UType& u_
) 
{
    cache.uChanged();
    if ((u_.rows() == u.rows() - 1) && (u_.cols() == u.cols()))
        u.block(1, 0, lmax, u.cols()) = u_;
    else
        throw errors::ValueError("Dimension mismatch in `u`.");
}

/**
Set the `l`th index of the limb darkening coefficient *matrix* to an
array of coefficients.

*/
inline void setU (
    int l, 
    const Ref<const UCoeffType>& coeff
) {
    cache.uChanged();
    if ((1 <= l) && (l <= lmax))
        u.row(l) = coeff;
    else
        throw errors::IndexError("Invalid value for `l`.");
}

/**
Set the `l`th index of the limb darkening coefficient vector, or
the entire `l`th row of the limb darkening coefficient matrix, to a
single value.

*/
inline void setU (
    int l, 
    const Scalar& coeff
) {
    cache.uChanged();
    if ((1 <= l) && (l <= lmax))
        u.row(l).setConstant(coeff);
    else
        throw errors::IndexError("Invalid value for `l`.");
}

/**
Get the full limb darkening vector/matrix.

*/
inline const UType getU () const {
    return u.block(1, 0, lmax, u.cols());
}

/**
Set the axis of rotation for the map.

*/
inline void setAxis (
    const UnitVector<Scalar>& axis_
) {
    cache.axisChanged();
    axis(0) = axis_(0);
    axis(1) = axis_(1);
    axis(2) = axis_(2);
    axis = axis / sqrt(axis(0) * axis(0) +
                       axis(1) * axis(1) +
                       axis(2) * axis(2));
}

/**
Return a copy of the axis.

*/
inline const UnitVector<Scalar> getAxis () const {
    return axis;
}

/**
Get the (l, m) row of the spherical harmonic coefficient *matrix*

*/
template <typename U=S, typename=IsSpectralOrTemporal<U>>
inline YCoeffType getY (
    int l,
    int m
) const {
    if ((0 <= l) && (l <= lmax) && (-l <= m) && (m <= l))
        return y.row(l * l + l + m);
    else
        throw errors::IndexError("Invalid value for `l` and/or `m`.");
}

/**
Get the (l, m) index of the spherical harmonic coefficient *vector*

*/
template <typename U=S, typename=IsDefault<U>>
inline Scalar getY (
    int l, 
    int m
) const {
    if ((0 <= l) && (l <= lmax) && (-l <= m) && (m <= l))
        return y(l * l + l + m);
    else
        throw errors::IndexError("Invalid value for `l` and/or `m`.");
}

/**
Get the `l`th row of the limb darkening coefficient *matrix*

*/
template <typename U=S, typename=IsSpectral<U>>
inline UCoeffType getU (
    int l
) const {
    if ((1 <= l) && (l <= lmax))
        return u.row(l);
    else
        throw errors::IndexError("Invalid value for `l`.");
}

/**
Get the `l`th index of the limb darkening coefficient *vector*

*/
template <typename U=S, typename=IsDefaultOrTemporal<U>>
inline Scalar getU (
    int l
) const {
    if ((1 <= l) && (l <= lmax))
        return u(l);
    else
        throw errors::IndexError("Invalid value for `l`.");
}