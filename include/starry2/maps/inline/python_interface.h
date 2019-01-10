#ifdef STARRY_ENABLE_PYTHON_INTERFACE

protected:

py::object show_ (
    const Scalar& t=0.0,
    const Scalar& theta=0.0,
    std::string cmap="plasma",
    size_t res=300,
    int interval=75,
    std::string gif=std::string()
);

py::object show_ (
    const Vector<Scalar>& t,
    const Vector<Scalar>& theta,
    std::string cmap="plasma",
    size_t res=300,
    int interval=75,
    std::string gif=std::string()
);

void loadImage_ (
    std::string image,
    int l=-1,
    int col=-1,
    bool normalize=true,
    int sampling_factor=8
);

public:

/**
Display the map (static case).

*/  
template<typename U=S, typename=IsStatic<U>>
inline py::object show (
    const Scalar& theta=0.0,
    std::string cmap="plasma",
    size_t res=300
) {
    return show_(0, theta, cmap, res);
}

/**
Display an animation of the map as it rotates (static case).

*/  
template<typename U=S, typename=IsStatic<U>>
inline py::object show (
    const Vector<Scalar>& theta,
    std::string cmap="plasma",
    size_t res=300,
    int interval=75,
    std::string gif=std::string()
) {
    if (ncol > 1) {
        throw errors::NotImplementedError("Spectral maps cannot be animated.");
    } else {
        Vector<Scalar> t = Vector<Scalar>::Zero(theta.size());
        return show_(t, theta, cmap, res, interval, gif);
    }
}

/**
Display the map (temporal case).

*/  
template<typename U=S, typename=IsTemporal<U>>
inline py::object show (
    const Scalar& t=0.0,
    const Scalar& theta=0.0,
    std::string cmap="plasma",
    size_t res=300,
    int interval=75,
    std::string gif=std::string()
) {
    return show_(t, theta, cmap, res, interval, gif);
}

/**
Display an animation of the map as it rotates (temporal case).

*/  
template<typename U=S, typename=IsTemporal<U>>
inline py::object show (
    const Vector<Scalar>& t,
    const Vector<Scalar>& theta,
    std::string cmap="plasma",
    size_t res=300,
    int interval=75,
    std::string gif=std::string()
) {
    return show_(t, theta, cmap, res, interval, gif);
}

/**
Load an image (single-column case).

*/  
template<typename U=S, typename=IsSingleColumn<U>>
void loadImage (
    std::string image,
    int l=-1,
    bool normalize=true,
    int sampling_factor=8
) {
    loadImage_(image, l, 0, normalize, sampling_factor);
}

/**
Load an image (multi-column case).

*/  
template<typename U=S, typename=IsMultiColumn<U>>
void loadImage (
    std::string image,
    int l=-1,
    int col=-1,
    bool normalize=true,
    int sampling_factor=8
) {
    loadImage_(image, l, col, normalize, sampling_factor);
}

#endif