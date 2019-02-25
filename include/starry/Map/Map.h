//! The Map class
template <class S=Default<double, false>>
class Map 
{

public:

    // Types
    using Scalar = typename S::Scalar;                                         /**< The scalar type of the map */
    using YType = typename S::YType;                                           /**< The type of the spherical harmonic coefficient object (vector/matrix) */
    using YCoeffType = typename S::YCoeffType;                                 /**< The type of the spherical harmonic coefficients (scalar/row vector) */
    using UType = typename S::UType;                                           /**< The type of the limb darkening coefficient object (vector/matrix) */
    using UCoeffType = typename S::UCoeffType;                                 /**< The type of the limb darkening coefficients (scalar/row vector) */
    using FluxType = typename S::FluxType;                                     /**< The type of the output flux (scalar/row vector) */

    // Public variables
    const int ydeg;                                                            /**< Maximum degree of the spherical harmonic map */
    const int udeg;                                                            /**< Maximum degree of the limb darkening map */
    const int Ny;                                                              /**< Number of spherical harmonic `(l, m)` coefficients */
    const int Nu;                                                              /**< Number of limb darkening coefficients */
    const int N;                                                               /**< Total number of spherical harmonic `(l, m)` coefficients after limb darkening */
    const int ncoly;                                                           /**< Number of columns in the `y` matrix */
    const int ncolu;                                                           /**< Number of columns in the `u` matrix */
    const int nflx;
    Data<Scalar> data;                                                         /**< Internal storage class */

protected:

    // Internal methods
    #include "protected/intensity.h"
    #include "protected/flux.h"
    #include "protected/oper.h"

    // Internal variables
    YType y;                                                                   /**< Vector/matrix of spherical harmonic coefficients */
    UType u;                                                                   /**< Vector/matrix of limb darkening coefficients */
    UnitVector<Scalar> axis;                                                   /**< The axis of rotation for the map */
    basis::Basis<Scalar> B;                                                    /**< Basis transform stuff */
    wigner::Wigner<Scalar> W;                                                  /**< Ylm rotation stuff */
    solver::Greens<Scalar, S::Reflected> G;                                    /**< The occultation integral solver class */
    limbdark::GreensLimbDark<Scalar, S::Reflected> L;                          /**< The occultation integral solver class (optimized for limb darkening) */
    Matrix<Scalar> taylor;
    Scalar radian;                                                             /**< Conversion factor from degrees to radians */
     
    //! Constructor for all map types
    explicit Map (
        int ydeg,
        int udeg,
        int ncoly,
        int ncolu,
        int nflx
    ) :
        ydeg(ydeg), 
        udeg(udeg),
        Ny((ydeg + 1) * (ydeg + 1)), 
        Nu(udeg + 1),
        N((ydeg + udeg + 1) * (ydeg + udeg + 1)),
        ncoly(ncoly), 
        ncolu(ncolu),
        nflx(nflx),
        data(ydeg),
        y(Ny, ncoly),
        u(Nu, ncolu),
        axis(yhat<Scalar>()),
        B(ydeg, udeg),
        W(ydeg, axis),
        G(ydeg + udeg),
        L(udeg)
    {
        if ((ydeg < 0) || (ydeg > STARRY_MAX_LMAX))
            throw errors::ValueError(
                "Spherical harmonic degree out of range.");
        if ((udeg < 0) || (udeg > STARRY_MAX_LMAX))
            throw errors::ValueError(
                "Limb darkening degree out of range.");
        if ((ydeg + udeg > STARRY_MAX_LMAX))
            throw errors::ValueError(
                "Total map degree out of range.");
        if ((ncoly < 1) || (ncolu < 1))
            throw errors::ValueError(
                "The number of map columns must be positive.");
        radian = pi<Scalar>() / 180.;
        reset();
    };

public:

    //! Constructor for single-column maps
    template <typename U=S, typename=IsDefault<U>>
    explicit Map (
        int ydeg,
        int udeg
    ) : Map(ydeg, udeg, 1, 1, 1) {}

    //! Constructor for multi-column maps
    template <typename U=S, typename=IsSpectralOrTemporal<U>>
    explicit Map (
        int ydeg,
        int udeg,
        int ncol
    ) : Map(ydeg, udeg, ncol, 
            std::is_same<U, Spectral<Scalar, S::Reflected>>::value ? ncol : 1,
            std::is_same<U, Spectral<Scalar, S::Reflected>>::value ? ncol : 1) {}

    // Public methods
    #include "public/io.h"
    #include "public/oper.h"
    #include "public/intensity.h"
    #include "public/flux.h"

}; // class Map




