#ifdef STARRY_ENABLE_PYTHON_INTERFACE

/**
 
*/
template <typename S>
py::object Map<S>::show_ (
    const Scalar& t,
    const Scalar& theta,
    std::string cmap,
    size_t res,
    int interval,
    std::string gif
) {
    py::object fshow;
    if (ncol == 1)
        fshow = py::module::import("starry2._plotting").attr("show");
    else
        fshow = py::module::import("starry2._plotting").attr("show_spectral");
    MapType intensity;
    renderMap_(t, theta, res, intensity);
    return fshow(intensity.template cast<double>(), res, cmap, gif, interval);
}

/**
 
*/
template <typename S>
py::object Map<S>::show_ (
    const Vector<Scalar>& t,
    const Vector<Scalar>& theta,
    std::string cmap,
    size_t res,
    int interval,
    std::string gif
) {
    size_t res2 = res * res;
    int frames = theta.size();
    MapType intensity(res2 * frames, nflx);
    int n = 0;
    for (int j = 0; j < frames; ++j) {
        renderMap_(t(j), theta(j), res, intensity.block(n, 0, res2, nflx));
        n += res2;
    }
    py::object fshow = py::module::import("starry2._plotting").attr("animate");
    return fshow(intensity.template cast<double>(), res, cmap, gif, interval);
}

/**
NOTE: If `l = -1`, computes the expansion up to `lmax`.
NOTE: If `col = -1`, loads the image into all columns.

*/
template <typename S>
void Map<S>::loadImage_ (
    std::string image,
    int l,
    int col,
    bool normalize,
    int sampling_factor
) {
    py::object fload = py::module::import("starry2._healpy").attr("load_map");
    if (l == -1)
        l = lmax;
    auto y_double = py::cast<Vector<double>>(fload(image, l, sampling_factor));
    if (normalize)
        y_double /= y_double(0);
    if (ncol == 1) {
        y.block(0, 0, (l + 1) * (l + 1), 1) = y_double.cast<Scalar>();
    } else if (col == -1) {
        y.block(0, 0, (l + 1) * (l + 1), ncol) = 
            y_double.cast<Scalar>().replicate(1, ncol);
    } else {
        y.block(0, col, (l + 1) * (l + 1), 1) = y_double.cast<Scalar>();
    }
    rotateByAxisAngle(xhat<Scalar>(), 0.0, 1.0, y);
    rotateByAxisAngle(zhat<Scalar>(), -1.0, 0.0, y);
    rotateByAxisAngle(yhat<Scalar>(), 0.0, -1.0, y);
    cache.yChanged();
}

#endif