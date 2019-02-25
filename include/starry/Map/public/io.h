/**
Return a human-readable map string.

*/
std::string info () {
    std::ostringstream os;
    if (std::is_same<S, Default<Scalar, S::Reflected>>::value) {
        os << "<starry.Map("
            << "ydeg=" << ydeg << ", "
            << "udeg=" << udeg << ", "
            << "reflected=" << S::Reflected << ", "
            << "multi=" << !std::is_same<Scalar, double>::value
            << ")>";
    } else if (std::is_same<S, Spectral<Scalar, S::Reflected>>::value) {
        os << "<starry.Map("
            << "ydeg=" << ydeg << ", "
            << "udeg=" << udeg << ", "
            << "nw=" << ncoly << ", "
            << "reflected=" << S::Reflected << ", "
            << "multi=" << !std::is_same<Scalar, double>::value
            << ")>";
    } else if (std::is_same<S, Temporal<Scalar, S::Reflected>>::value) {
        os << "<starry.Map("
            << "ydeg=" << ydeg << ", "
            << "udeg=" << udeg << ", "
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
    if ((0 <= l) && (l <= ydeg) && (-l <= m) && (m <= l))
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
    if ((0 <= l) && (l <= ydeg) && (-l <= m) && (m <= l))
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
    if ((u_.rows() == u.rows() - 1) && (u_.cols() == u.cols()))
        u.block(1, 0, udeg, u.cols()) = u_;
    else
        throw errors::ValueError("Dimension mismatch in `u`.");
}

/**
Set the `l`th index of the limb darkening coefficient *matrix* to an
array of coefficients.

*/
template <typename U=S, typename=IsEmitted<U>>
inline void setU (
    int l, 
    const Ref<const UCoeffType>& coeff
) {
    if ((1 <= l) && (l <= udeg))
        u.row(l) = coeff;
    else
        throw errors::IndexError("Invalid value for `l`.");
}

/**
Set the `l`th index of the limb darkening coefficient vector, or
the entire `l`th row of the limb darkening coefficient matrix, to a
single value.

*/
template <typename U=S, typename=IsEmitted<U>>
inline void setU (
    int l, 
    const Scalar& coeff
) {
    if ((1 <= l) && (l <= udeg))
        u.row(l).setConstant(coeff);
    else
        throw errors::IndexError("Invalid value for `l`.");
}

/**
Get the full limb darkening vector/matrix.

*/
template <typename U=S, typename=IsEmitted<U>>
inline const UType getU () const {
    return u.block(1, 0, udeg, u.cols());
}

/**
Set the axis of rotation for the map.

*/
inline void setAxis (
    const UnitVector<Scalar>& axis_
) {
    axis = axis_.normalized();
    W.updateAxisAndGradient(axis);
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
    if ((0 <= l) && (l <= ydeg) && (-l <= m) && (m <= l))
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
    if ((0 <= l) && (l <= ydeg) && (-l <= m) && (m <= l))
        return y(l * l + l + m);
    else
        throw errors::IndexError("Invalid value for `l` and/or `m`.");
}

/**
Get the `l`th row of the limb darkening coefficient *matrix*

*/
template <typename U=S, typename=IsSpectral<U>, typename=IsEmitted<U>>
inline UCoeffType getU (
    int l
) const {
    if ((1 <= l) && (l <= ydeg))
        return u.row(l);
    else
        throw errors::IndexError("Invalid value for `l`.");
}

/**
Get the `l`th index of the limb darkening coefficient *vector*

*/
template <typename U=S, typename=IsDefaultOrTemporal<U>, typename=IsEmitted<U>>
inline Scalar getU (
    int l
) const {
    if ((1 <= l) && (l <= ydeg))
        return u(l);
    else
        throw errors::IndexError("Invalid value for `l`.");
}