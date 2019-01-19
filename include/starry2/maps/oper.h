template <class S>
inline void Map<S>::reset () 
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
template <class S>
inline int Map<S>::getYDeg_ ()
{
    computeDegreeY();
    return y_deg;
}

/**
Return the current highest limb darkening
degree of the map.

*/
template <class S>
inline int Map<S>::getUDeg_ ()
{
    computeDegreeU();
    return u_deg;
}

/**
Check if the total degree of the map is valid.

*/
template <class S>
inline void Map<S>::checkDegree () 
{
    if (y_deg + u_deg > lmax) {
        cache.reset();
        y.setZero();
        y_deg = 0;
        u.setZero();
        setU0();
        u_deg = 0;
        throw errors::ValueError("Degree of the limb-darkened "
                                 "map exceeds `lmax`. All "
                                 "coefficients have been reset.");
    }
}

//! Compute the degree of the Ylm map.
template <class S>
inline void Map<S>::computeDegreeY () 
{
    if (cache.compute_degree_y) {
        y_deg = 0;
        for (int l = lmax; l >= 0; --l) {
            if ((y.block(l * l, 0, 2 * l + 1, ncoly).array() 
                    != 0.0).any()) {
                y_deg = l;
                break;
            }
        }
        checkDegree();
        cache.compute_degree_y = false;
    }
}

//! Compute the degree of the Ul map.
template <class S>
inline void Map<S>::computeDegreeU () 
{
    if (cache.compute_degree_u) {
        u_deg = 0;
        for (int l = lmax; l > 0; --l) {
            if (u.row(l).any()) {
                u_deg = l;
                break;
            }
        }
        checkDegree();
        cache.compute_degree_u = false;
    }
}

//! Compute the change of basis matrix from Ylms to pixels
template <class S>
inline void Map<S>::computeP (int res) {
    if (cache.compute_P || (cache.res != res)) {
        B.computePolyMatrix(res, cache.P);
        cache.res = res;
        cache.compute_P = false;
    }
}

//! Compute the zeta frame transform for Ylm rotations
template <class S>
inline void Map<S>::computeWigner () {
    if (cache.compute_Zeta) {
        W.updateZeta();
        W.updateYZeta();
        cache.compute_Zeta = false;
        cache.compute_YZeta = false;
    } else if (cache.compute_YZeta) {
        W.updateYZeta();
        cache.compute_YZeta = false;
    }
}

/**
Compute the Agol `c` basis and its derivative.
These are both normalized such that the total
unobscured flux is **unity**.

*/
template<class S>
inline void Map<S>::computeAgolGBasis () {
    if (cache.compute_g) {
        limbdark::computeAgolGBasis(u, cache.g, cache.DgDu);
        normalizeAgolGBasis(cache.g, cache.DgDu);
        cache.compute_g = false;
    }
}

/**
Rotate the map *in place* by an angle `theta`.

*/
template <class S>
inline void Map<S>::rotate (
    const Scalar& theta
) 
{
    Scalar theta_rad = theta * radian;
    computeWigner();
    W.rotate(cos(theta_rad), sin(theta_rad));
    cache.mapRotated();
}

/**
Rotate an arbitrary map vector in place
given an axis and an angle. If `col = -1`,
rotate all columns of the map, otherwise
rotate only the column with index `col`.

*/
template <class S>
inline void Map<S>::rotateByAxisAngle (
    const UnitVector<Scalar>& axis_,
    const Scalar& costheta,
    const Scalar& sintheta,
    YType& y_,
    int col
) {
    Scalar tol = 10 * mach_eps<Scalar>();
    Scalar cosalpha, sinalpha, cosbeta, sinbeta, cosgamma, singamma;
    rotation::axisAngleToEuler(
        axis_(0), axis_(1), costheta, sintheta, tol,
        cosalpha, sinalpha, cosbeta, sinbeta, 
        cosgamma, singamma);
    rotation::rotar(
        lmax, cosalpha, sinalpha, 
        cosbeta, sinbeta, 
        cosgamma, singamma, tol, 
        cache.EulerD, cache.EulerR);
    if (col == -1) {
        for (int l = 0; l < lmax + 1; ++l) {
            y_.block(l * l, 0, 2 * l + 1, ncoly) =
                cache.EulerR[l] * y_.block(l * l, 0, 2 * l + 1, ncoly);
        }
    } else {
        for (int l = 0; l < lmax + 1; ++l) {
            y_.block(l * l, col, 2 * l + 1, 1) =
                cache.EulerR[l] * y_.block(l * l, col, 2 * l + 1, 1);
        }
    }
}

/**
Add a gaussian spot at a given latitude/longitude on the map.

*/
template <class S>
inline void Map<S>::addSpot (
    const YCoeffType& amp,
    const Scalar& sigma,
    const Scalar& lat,
    const Scalar& lon,
    int l
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
template <class S>
template <class V>
inline void Map<S>::random_ (
    const Vector<Scalar>& power,
    const V& seed,
    int col
) {
    int lmax_ = power.size() - 1;
    if (lmax_ > lmax) 
        lmax_ = lmax;
    int N_ = (lmax_ + 1) * (lmax_ + 1);

    // Generate N_ standard normal variables
    std::mt19937 gen(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    Vector<Scalar> vec(N_);
    for (int n = 0; n < N_; ++n)
        vec(n) = static_cast<Scalar>(normal(gen));

    // Zero degree
    vec(0) = sqrt(abs(power(0)));

    // Higher degrees
    for (int l = 1; l < lmax_ + 1; ++l) {
        vec.segment(l * l, 2 * l + 1) *=
            sqrt(abs(power(l)) / vec.segment(l * l, 2 * l + 1).squaredNorm());
    }

    // Set the vector
    if (col == -1) {
        y.block(0, 0, N_, ncoly) = vec.replicate(1, ncoly);
    } else {
        y.block(0, col, N_, 1) = vec;
    }

    cache.yChanged();
}