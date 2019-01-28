/**
Reset the map coefficients, the axis, 
and all cached variables.

*/
inline void reset () 
{
    // Reset the cache
    cache.reset();

    // Reset Ylms
    y.setZero(N, ncoly);
    y_deg = 0;

    // Reset limb darkening
    u.setZero(lmax + 1, ncolu);
    setU0();
    u_deg = 0;

    // Reset the axis
    axis = yhat<Scalar>();
}

/**
Return the current highest spherical 
harmonic degree of the map.

*/
inline int getYDeg ()
{
    computeDegreeY();
    return y_deg;
}

/**
Return the current highest limb darkening
degree of the map.

*/
inline int getUDeg ()
{
    computeDegreeU();
    return u_deg;
}


/**
Rotate the map *in place* by an angle `theta`.

*/
inline void rotate (
    const Scalar& theta
) 
{
    Scalar theta_rad = theta * radian;
    computeWigner();
    W.rotate(cos(theta_rad), sin(theta_rad));
    cache.mapRotated();
}

/**
Add a gaussian spot at a given latitude/longitude on the map.

*/
inline void addSpot (
    const YCoeffType& amp,
    const Scalar& sigma,
    const Scalar& lat=0,
    const Scalar& lon=0,
    int l=-1
) {
    // Default degree is max degree
    if (l < 0) 
        l = lmax;
    if (l > lmax) 
        throw errors::ValueError("Invalid value for `l`.");

    // Compute the integrals recursively
    Vector<Scalar> IP(l + 1);
    Vector<Scalar> ID(l + 1);
    YType coeff(N, ncoly);
    coeff.setZero();

    // Constants
    Scalar a = 1.0 / (2 * sigma * sigma);
    Scalar sqrta = sqrt(a);
    Scalar erfa = erf(2 * sqrta);
    Scalar term = exp(-4 * a);

    // Seeding values
    IP(0) = root_pi<Scalar>() / (2 * sqrta) * erfa;
    IP(1) = (root_pi<Scalar>() * sqrta * erfa + term - 1) / (2 * a);
    ID(0) = 0;
    ID(1) = IP(0);

    // Recurse
    int sgn = -1;
    for (int n = 2; n < l + 1; ++n) {
        IP(n) = (2.0 * n - 1.0) / (2.0 * n * a) * (ID(n - 1) + sgn * term - 1.0) +
                (2.0 * n - 1.0) / n * IP(n - 1) - (n - 1.0) / n * IP(n - 2);
        ID(n) = (2.0 * n - 1.0) * IP(n - 1) + ID(n - 2);
        sgn *= -1;
    }

    // Compute the coefficients of the expansion
    // normalized so the integral over the sphere is `amp`
    for (int n = 0; n < l + 1; ++n)
        coeff.row(n * n + n) = 0.25 * amp * sqrt(2 * n + 1) * (IP(n) / IP(0));

    // Rotate the spot to the correct lat/lon
    // TODO: Speed this up with a single compound rotation
    Scalar lat_rad = lat * radian;
    Scalar lon_rad = lon * radian;
    rotateByAxisAngle(xhat<Scalar>(), cos(lat_rad), -sin(lat_rad), coeff);
    rotateByAxisAngle(yhat<Scalar>(), cos(lon_rad), sin(lon_rad), coeff);

    // Add this to the map
    cache.yChanged();
    y += coeff;
}

/**
Generate a random isotropic map with a given power spectrum.

*/
template <typename V, typename U=S, typename=IsDefault<U>>
inline void random (
    const Vector<Scalar>& power,
    const V& seed
) {
    randomInternal(power, seed, 0);
}

/**
Generate a random isotropic map with a given power spectrum.
NOTE: If `col = -1`, sets all columns to the same map.

*/
template <typename V, typename U=S, typename=IsSpectralOrTemporal<U>>
inline void random (
    const Vector<Scalar>& power,
    const V& seed,
    int col=-1
) {
    randomInternal(power, seed, col);
}