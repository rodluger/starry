#ifdef STARRY_ENABLE_PYTHON_INTERFACE

/**
Display the map (static case).

*/  
template <typename U=S, typename=IsDefaultOrSpectral<U>, typename=IsEmitted<U>>
inline py::object show (
    const Scalar& theta=0.0,
    std::string cmap="plasma",
    size_t res=300
) {
    return showInternal(0, theta, cmap, res);
}

/**
Display an animation of the map as it rotates (static case).

*/  
template <typename U=S, typename=IsDefaultOrSpectral<U>, typename=IsEmitted<U>>
inline py::object show (
    const Vector<Scalar>& theta,
    std::string cmap="plasma",
    size_t res=300,
    int interval=75,
    std::string gif=std::string()
) {
    Vector<Scalar> t = Vector<Scalar>::Zero(theta.size());
    return showInternal(t, theta, cmap, res, interval, gif);
}

/**
Display the map (temporal case).

*/  
template <typename U=S, typename=IsTemporal<U>, typename=IsEmitted<U>>
inline py::object show (
    const Scalar& t=0.0,
    const Scalar& theta=0.0,
    std::string cmap="plasma",
    size_t res=300,
    int interval=75,
    std::string gif=std::string()
) {
    return showInternal(t, theta, cmap, res, interval, gif);
}

/**
Display an animation of the map as it rotates (temporal case).

*/  
template <typename U=S, typename=IsTemporal<U>, typename=IsEmitted<U>>
inline py::object show (
    const Vector<Scalar>& t,
    const Vector<Scalar>& theta,
    std::string cmap="plasma",
    size_t res=300,
    int interval=75,
    std::string gif=std::string()
) {
    return showInternal(t, theta, cmap, res, interval, gif);
}

/**
Display the map (static case).

*/  
template <typename U=S, typename=IsDefaultOrSpectral<U>, typename=IsReflected<U>>
inline py::object show (
    const Scalar& theta=0.0,
    const UnitVector<Scalar>& source=-xhat<Scalar>(),
    std::string cmap="plasma",
    size_t res=300
) {
    return showInternal(0, theta, source.normalized(), cmap, res);
}

/**
Display an animation of the map as it rotates (static case).

*/  
template <typename U=S, typename=IsDefaultOrSpectral<U>, typename=IsReflected<U>>
inline py::object show (
    const Vector<Scalar>& theta,
    const Matrix<Scalar>& source,
    std::string cmap="plasma",
    size_t res=300,
    int interval=75,
    std::string gif=std::string()
) {
    Vector<Scalar> t = Vector<Scalar>::Zero(theta.size());
    return showInternal(t, theta, source, cmap, res, interval, gif);
}

/**
Display the map (temporal case).

*/  
template <typename U=S, typename=IsTemporal<U>, typename=IsReflected<U>>
inline py::object show (
    const Scalar& t=0.0,
    const Scalar& theta=0.0,
    const UnitVector<Scalar>& source=-xhat<Scalar>(),
    std::string cmap="plasma",
    size_t res=300,
    int interval=75,
    std::string gif=std::string()
) {
    return showInternal(t, theta, source.normalized(), cmap, res, interval, gif);
}

/**
Display an animation of the map as it rotates (temporal case).

*/  
template <typename U=S, typename=IsTemporal<U>, typename=IsReflected<U>>
inline py::object show (
    const Vector<Scalar>& t,
    const Vector<Scalar>& theta,
    const Matrix<Scalar>& source,
    std::string cmap="plasma",
    size_t res=300,
    int interval=75,
    std::string gif=std::string()
) {
    return showInternal(t, theta, source, cmap, res, interval, gif);
}

/**
Load an image (single-column case).

*/  
template <typename U=S, typename=IsDefault<U>>
void loadImage (
    std::string image,
    int l=-1,
    bool normalize=true,
    int sampling_factor=8
) {
    loadImageInternal(image, l, 0, normalize, sampling_factor);
}

/**
Load an image (multi-column case).

*/  
template <typename U=S, typename=IsSpectralOrTemporal<U>>
void loadImage (
    std::string image,
    int l=-1,
    int col=-1,
    bool normalize=true,
    int sampling_factor=8
) {
    loadImageInternal(image, l, col, normalize, sampling_factor);
}

#endif