/**
Get the (l, m) row of the spherical harmonic coefficient *matrix*

*/
template<typename U=S, typename=IsMultiColumn<U>>
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
template<typename U=S, typename=IsSingleColumn<U>>
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
template<typename U=S, typename=IsSpectral<U>>
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
template<typename U=S, typename=IsSingleWavelength<U>>
inline Scalar getU (
    int l
) const {
    if ((1 <= l) && (l <= lmax))
        return u(l);
    else
        throw errors::IndexError("Invalid value for `l`.");
}