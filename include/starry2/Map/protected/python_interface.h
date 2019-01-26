#ifdef STARRY_ENABLE_PYTHON_INTERFACE

/**
 
*/
py::object show_ (
    const Scalar& t=0.0,
    const Scalar& theta=0.0,
    std::string cmap="plasma",
    size_t res=300,
    int interval=75,
    std::string gif=std::string()
) {
    py::object fshow;
    if (std::is_same<S, Spectral<Scalar>>::value)
        fshow = py::module::import("starry2._plotting").attr("show_spectral");
    else
        fshow = py::module::import("starry2._plotting").attr("show");
    if (res < 1)
        throw errors::ValueError("Invalid value for `res`.");
    Matrix<Scalar> intensity(res * res, nflx);
    computeTaylor_(t);
    renderMap_(theta, res, intensity);
    return fshow(intensity.template cast<double>(), res, cmap, gif, interval);
}

/**
 
*/
py::object show_ (
    const Vector<Scalar>& t,
    const Vector<Scalar>& theta,
    std::string cmap="plasma",
    size_t res=300,
    int interval=75,
    std::string gif=std::string()
) {
    if (res < 1)
        throw errors::ValueError("Invalid value for `res`.");
    size_t res2 = res * res;
    int frames = theta.size();
    Matrix<Scalar> intensity(res2 * frames, nflx);
    int n = 0;
    for (int j = 0; j < frames; ++j) {
        computeTaylor_(t(j));
        renderMap_(theta(j), res, intensity.block(n, 0, res2, nflx));
        n += res2;
    }
    py::object fshow = py::module::import("starry2._plotting").attr("animate");
    return fshow(intensity.template cast<double>(), res, cmap, gif, interval);
}

/**
NOTE: If `l = -1`, computes the expansion up to `lmax`.
NOTE: If `col = -1`, loads the image into all columns.

*/
void loadImage_ (
    std::string image,
    int l=-1,
    int col=-1,
    bool normalize=true,
    int sampling_factor=8
) {
    py::object fload = py::module::import("starry2._healpy").attr("load_map");
    if ((l == -1) || (l > lmax))
        l = lmax;
    if (col > ncoly)
        throw errors::ValueError("Invalid value for `col`.");
    auto y_double = py::cast<Vector<double>>(fload(image, l, sampling_factor));
    if (normalize)
        y_double /= y_double(0);
    if (ncoly == 1) {
        y.block(0, 0, (l + 1) * (l + 1), 1) = y_double.cast<Scalar>();
    } else if (col == -1) {
        y.block(0, 0, (l + 1) * (l + 1), ncoly) = 
            y_double.cast<Scalar>().replicate(1, ncoly);
    } else {
        y.block(0, col, (l + 1) * (l + 1), 1) = y_double.cast<Scalar>();
    }
    rotateByAxisAngle_(xhat<Scalar>(), 0.0, 1.0, y, col);
    rotateByAxisAngle_(zhat<Scalar>(), -1.0, 0.0, y, col);
    rotateByAxisAngle_(yhat<Scalar>(), 0.0, -1.0, y, col);
    cache.yChanged();
}

#endif