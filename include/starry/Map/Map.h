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

    // Public variables
    const int ydeg;                                                            /**< Maximum degree of the spherical harmonic map */
    const int udeg;                                                            /**< Maximum degree of the limb darkening map */
    const int deg;                                                             /**< Maximum degree of the combined Ylm + limb darkening map */
    const int Ny;                                                              /**< Number of spherical harmonic `(l, m)` coefficients */
    const int Nu;                                                              /**< Number of limb darkening coefficients in the `u` basis */
    const int Np;                                                              /**< Number of limb darkening coefficients in the `p` basis*/
    const int Nw;                                                              /**< Number of spectral components */
    const int Nt;                                                              /**< Number of temporal components */
    const int Nf;                                                              /**< Number of flux components per timestep */
    const int N;                                                               /**< Total number of spherical harmonic `(l, m)` coefficients after limb darkening */
    Data<Scalar> data;                                                         /**< Internal storage class */

protected:

    // Internal methods
    #include "protected/intensity.h"
    #include "protected/flux.h"
    #include "protected/oper.h"

    // Internal variables
    YType y;                                                                   /**< Vector of spherical harmonic coefficients */
    UType u;                                                                   /**< Vector of limb darkening coefficients */
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
        int Nw,
        int Nt,
        int Nf
    ) :
        ydeg(ydeg), 
        udeg(udeg),
        deg(ydeg + udeg),
        Ny((ydeg + 1) * (ydeg + 1)), 
        Nu(udeg + 1),
        Np((udeg + 1) * (udeg + 1)),
        Nw(Nw),
        Nt(Nt),
        Nf(Nf),
        N((deg + 1) * (deg + 1)),
        data(ydeg),
        y(Ny * Nt, Nw),
        u(Nu),
        axis(yhat<Scalar>()),
        B(ydeg, udeg),
        W(ydeg, axis),
        G(ydeg),
        L(udeg)
    {
        if ((ydeg < 0) || (ydeg > STARRY_MAX_LMAX))
            throw std::out_of_range(
                "Spherical harmonic degree out of range.");
        if ((udeg < 0) || (udeg > STARRY_MAX_LMAX))
            throw std::out_of_range(
                "Limb darkening degree out of range.");
        if ((deg > STARRY_MAX_LMAX))
            throw std::out_of_range(
                "Total map degree out of range.");
        if ((Nw < 1) || (Nt < 1))
            throw std::out_of_range(
                "The number of temporal / spectral terms must be positive.");
        radian = pi<Scalar>() / 180.;
        reset();
    };

public:

    //! Constructor for the default map
    template <typename U=S, typename=IsDefault<U>>
    explicit Map (
        int ydeg,
        int udeg
    ) : Map(ydeg, udeg, 1, 1, 1) {}

    //! Constructor for spectral & temporal maps
    template <typename U=S, typename=IsSpectralOrTemporal<U>>
    explicit Map (
        int ydeg,
        int udeg,
        int nterms
    ) : Map(ydeg, udeg, 
            std::is_same<U, Spectral<Scalar, S::Reflected>>::value ? nterms : 1,
            std::is_same<U, Temporal<Scalar, S::Reflected>>::value ? nterms : 1,
            std::is_same<U, Spectral<Scalar, S::Reflected>>::value ? nterms : 1) {}

    // Public methods
    #include "public/io.h"
    #include "public/oper.h"
    #include "public/intensity.h"
    #include "public/flux.h"

}; // class Map




