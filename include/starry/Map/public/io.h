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
            << "nw=" << Nw << ", "
            << "reflected=" << S::Reflected << ", "
            << "multi=" << !std::is_same<Scalar, double>::value
            << ")>";
    } else if (std::is_same<S, Temporal<Scalar, S::Reflected>>::value) {
        os << "<starry.Map("
            << "ydeg=" << ydeg << ", "
            << "udeg=" << udeg << ", "
            << "nt=" << Nt << ", "
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
Set the full spherical harmonic vector.

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
Get the full spherical harmonic vector.

*/
inline const YType getY () const {
    return y;
}

/**
Set the full limb darkening vector.

*/
inline void setU (
    const UType& u_
) 
{
    if ((u_.rows() == u.rows()) && (u_.cols() == u.cols()))
        u = u_;
    else
        throw errors::ValueError("Dimension mismatch in `u`.");
}

/**
Get the full limb darkening vector.

*/
template <typename U=S, typename=IsEmitted<U>>
inline const UType getU () const {
    return u;
}

/**
Set the axis of rotation for the map.

*/
inline void setAxis (
    const UnitVector<Scalar>& axis_
) {
    axis = axis_.normalized();
    W.updateAxis(axis);
}

/**
Return a copy of the axis.

*/
inline const UnitVector<Scalar> getAxis () const {
    return axis;
}