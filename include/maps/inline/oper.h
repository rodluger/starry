/**
Generate a random isotropic map with a given power spectrum.

*/
template<typename V, typename U=S, typename=IsSingleColumn<U>>
inline void random (
    const Vector<Scalar>& power,
    const V& seed
) {
    random_(power, seed, 0);
}

/**
Generate a random isotropic map with a given power spectrum.
NOTE: If `col = -1`, sets all columns to the same map.

*/
template<typename V, typename U=S, typename=IsMultiColumn<U>>
inline void random (
    const Vector<Scalar>& power,
    const V& seed,
    int col=-1
) {
    random_(power, seed, col);
}

/**
Temporal contraction operation for static maps: effectively does nothing,
and returns the original map.

*/
template<typename V, typename U=S>
inline IsStatic<U, V&> contract(const V& mat, const Scalar& t) {
    return const_cast<V&>(mat);
}

/**
Contracts a temporal map by dotting the map matrix with the
Taylor expansion basis.

*/
template<typename V, typename U=S>
inline IsTemporal<U, Vector<Scalar>> contract(const V& mat, const Scalar& t) {
    if (expansion == STARRY_EXPANSION_TAYLOR) {
        for (int n = 1; n < ncol; ++n)
            tbasis(n) = tbasis(n - 1) * t;
    } else if (expansion == STARRY_EXPANSION_FOURIER) {
        // TODO
        throw errors::NotImplementedError(
                "Fourier expansion not yet implemented.");
    } else {
        throw errors::ValueError("Invalid temporal expansion type.");
    }
    return mat * tbasis;
}

/**
Derivative of the contraction operation for static maps: returns zero.

*/
template<typename V, typename U=S>
inline IsStatic<U, Matrix<Scalar>> contract_deriv(const V& mat, const Scalar& t) {
    return mat * 0.0;
}

/**
Contracts a temporal map by dotting the map matrix with the
derivative of the Taylor expansion basis.

*/
template<typename V, typename U=S>
inline IsTemporal<U, Vector<Scalar>> contract_deriv(const V& mat, const Scalar& t) {
    if (ncol > 1) {
        if (expansion == STARRY_EXPANSION_TAYLOR) {
            dtbasis(1) = 1;
            for (int n = 2; n < ncol; ++n)
                dtbasis(n) = dtbasis(n - 1) * t * (n / (n - 1.0));
        } else if (expansion == STARRY_EXPANSION_FOURIER) {
            // TODO
            throw errors::NotImplementedError(
                    "Fourier expansion not yet implemented.");
        } else {
            throw errors::ValueError("Invalid temporal expansion type.");
        }
    }
    return mat * dtbasis;
}

/**
Compute the Agol `c` basis and its derivative.
These are both normalized to the Y_{0,0} coefficient,
a scalar in this case.

*/
template<typename U=S>
inline IsSingleColumn<U, void> computeC() {
    if (cache.compute_c) {
        limbdark::computeC(u, cache.c, cache.dcdu, y(0));
        cache.compute_c = false;
    }
}

/**
Compute the Agol `c` basis and its derivative.
These are both normalized to the Y_{0,0} coefficient,
a row vector in this case.

*/
template<typename U=S>
inline IsMultiColumn<U, void> computeC() {
    if (cache.compute_c) {
        limbdark::computeC(u, cache.c, cache.dcdu, y.row(0));
        cache.compute_c = false;
    }
}