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
        throw std::length_error("Dimension mismatch in `y`.");
    // Check that y(0) == 1
    if (!(y.row(0) == RowVector<Scalar>::Ones(y.row(0).size()))) {
        y.row(0).setConstant(1.0);
        throw std::invalid_argument("The coefficient of the Y_{0,0} " 
                                    "term must be fixed at unity.");
    }
    // Check that the derivatives of y(0) == 0
    if (std::is_same<S, Temporal<Scalar, S::Reflected>>::value) {
        for (int i = 1; i < Nt; ++i) {
            if (y(i * Ny) != 0)
                throw std::invalid_argument("The Y_{0,0} term cannot "
                                            "have time dependence.");
        }
    }
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
        throw std::length_error("Dimension mismatch in `u`.");
    // Check that u(0) == -1
    if (!(u.row(0) == -RowVector<Scalar>::Ones(u.row(0).size()))) {
        u.row(0).setConstant(-1.0);
        throw std::invalid_argument("The coefficient of the u_0 " 
                                    "term must be fixed at -1.0.");
    }
}

/**
Get the full limb darkening vector.

*/
inline const UType getU () const {
    return u;
}

/**
Set the axis of rotation for the map and update
the pre-computed Wigner matrices.

\todo There are singularities here!

*/
inline void setAxis (
    const UnitVector<Scalar>& axis_
) {
    UnitVector<Scalar> axis = axis_.normalized();
    obl = atan2(axis(0), axis(1)) * 180.0 / pi<Scalar>();
    inc = atan2(axis(0) / sin(obl * pi<Scalar>() / 180.0), axis(2)) * 180.0 / pi<Scalar>();
    W.updateAxis(inc, obl);
}

/**
Return a copy of the axis.

*/
inline const UnitVector<Scalar> getAxis () const {
    UnitVector<Scalar> axis;
    axis << sin(obl * pi<Scalar>() / 180.) * sin(inc * pi<Scalar>() / 180.),
            cos(obl * pi<Scalar>() / 180.) * sin(inc * pi<Scalar>() / 180.),
            cos(inc * pi<Scalar>() / 180.);
    return axis;
}

/**
Set the inclination the map and update
the pre-computed Wigner matrices.

*/
inline void setInclination (
    const Scalar& inc_
) {
    if ((inc_ < 0) || (inc_ > 180))
        throw std::out_of_range("Inclination must be between 0 and 180 degrees.");
    inc = inc_;
    W.updateAxis(inc, obl);
}

/**
Return the inclination of the map.

*/
inline const Scalar getInclination () const {
    return inc;
}

/**
Set the obliquity the map and update
the pre-computed Wigner matrices.

*/
inline void setObliquity (
    const Scalar& obl_
) {
    if ((obl_ < -180) || (obl_ > 180))
        throw std::out_of_range("Obliquity must be between -180 and 180 degrees.");
    obl = obl_;
    W.updateAxis(inc, obl);
}

/**
Return the obliquity of the map.

*/
inline const Scalar getObliquity () const {
    return obl;
}