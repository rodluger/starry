/**
Return a human-readable map string.

*/
std::string info () {
    std::ostringstream os;
    if (S::Spectral) {
        os << "<starry.Map("
            << "ydeg=" << ydeg << ", "
            << "udeg=" << udeg << ", "
            << "fdeg=" << fdeg << ", "
            << "nw=" << Nw << ", "
            << "reflected=" << S::Reflected << ", "
            << "multi=" << !std::is_same<Scalar, double>::value
            << ")>";
    } else if (S::Temporal) {
        os << "<starry.Map("
            << "ydeg=" << ydeg << ", "
            << "udeg=" << udeg << ", "
            << "fdeg=" << fdeg << ", "
            << "nt=" << Nt << ", "
            << "reflected=" << S::Reflected << ", "
            << "multi=" << !std::is_same<Scalar, double>::value
            << ")>";
    } else if (S::LimbDarkened) {
        os << "<starry.Map("
            << "udeg=" << udeg << ", "
            << "multi=" << !std::is_same<Scalar, double>::value
            << ")>";
    } else {
        os << "<starry.Map("
            << "ydeg=" << ydeg << ", "
            << "udeg=" << udeg << ", "
            << "fdeg=" << fdeg << ", "
            << "reflected=" << S::Reflected << ", "
            << "multi=" << !std::is_same<Scalar, double>::value
            << ")>";
    }
    return std::string(os.str());
}

/**
Set the full spherical harmonic vector.

*/
template <typename T1>
inline void setY (
    const MatrixBase<T1>& y_
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
    if (S::Temporal) {
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
inline auto getY () -> const decltype(y) {
    return y;
}

/**
Set the full limb darkening vector.

*/
template <typename T1>
inline void setU (
    const MatrixBase<T1>& u_
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
inline auto getU () -> const decltype(u) {
    return u;
}

/**
Set the full filter spherical harmonic vector.

*/
template <typename T1>
inline void setF (
    const MatrixBase<T1>& f_
) {
    if ((f_.rows() == f.rows()) && (f_.cols() == f.cols()))
        f = f_;
    else
        throw std::length_error("Dimension mismatch in `f`.");
    // Check that f(0) == 1
    if (!(f.row(0) == RowVector<Scalar>::Ones(f.row(0).size()))) {
        f.row(0).setConstant(1.0);
        throw std::invalid_argument("The coefficient of the constant " 
                                    "term of the filter must be fixed at unity.");
    }
}

/**
Get the full filter spherical harmonic vector.

*/
inline auto getF () -> const decltype(f) {
    return f;
}

/**
Set the axis of rotation for the map and update
the pre-computed Wigner matrices.

*/
inline void setAxis (
    const UnitVector<Scalar>& axis_
) {
    UnitVector<Scalar> axis = axis_.normalized();
    obl = atan2(axis(0), axis(1)) * 180.0 / pi<Scalar>();
    Scalar sino = sin(obl * pi<Scalar>() / 180.0);
    if (abs(sino) < 1e-10) {
        Scalar coso = cos(obl * pi<Scalar>() / 180.0);
        inc = atan2(axis(1) / coso, axis(2)) * 180.0 / pi<Scalar>();
    } else {
        inc = atan2(axis(0) / sino, axis(2)) * 180.0 / pi<Scalar>();
    }
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