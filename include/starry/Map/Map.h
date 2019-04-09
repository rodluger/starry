//! The Map class
template <class S=MapType<double, false, false, false, false>>
class Map 
{

public:

    // Types
    using Scalar = typename S::Scalar;                                         /**< The scalar type of the map */
    using YType = typename S::YType;
    using UType = typename S::UType;
    using FType = typename S::FType;
    
    // Public variables
    const int ydeg;                                                            /**< Maximum degree of the spherical harmonic map */
    const int udeg;                                                            /**< Maximum degree of the limb darkening map */
    const int fdeg;                                                            /**< Maximum degree of the multiplicative filter */
    const int deg;                                                             /**< Maximum degree of the combined Ylm + limb darkening map */
    const int Ny;                                                              /**< Number of spherical harmonic `(l, m)` coefficients */
    const int Nu;                                                              /**< Number of limb darkening coefficients in the `u` basis */
    const int Nf;                                                              /**< Number of filter coefficients */
    const int Np;                                                              /**< Number of limb darkening coefficients in the `p` basis*/
    const int Nw;                                                              /**< Number of spectral components */
    const int Nt;                                                              /**< Number of temporal components */
    const int N;                                                               /**< Total number of spherical harmonic `(l, m)` coefficients after limb darkening */
    Data<Scalar> data;                                                         /**< Internal storage class */

protected:

    // Internal methods
    #include "protected/intensity.h"
    #include "protected/flux.h"
    #include "protected/oper.h"
    #include "protected/arrays.h"

    // Internal variables
    YType y;                                                                   /**< Vector of spherical harmonic coefficients */
    UType u;                                                                   /**< Vector of limb darkening coefficients */
    Vector<Scalar> f;                                                          /**< Vector of multiplicative filter spherical harmonic coefficients */
    bool filter_on;                                                            /**< Filter enabled? */
    Scalar inc;                                                                /**< Inclination of the rotation axis in degrees */
    Scalar obl;                                                                /**< Obliquity of the rotation axis in degrees */
    basis::Basis<Scalar> B;                                                    /**< Basis transform stuff */
    wigner::Wigner<Scalar> W;                                                  /**< Ylm rotation stuff */
    solver::Greens<Scalar, S::Reflected> G;                                    /**< The occultation integral solver class */
    limbdark::GreensLimbDark<Scalar, S::Reflected> L;                          /**< The occultation integral solver class (optimized for limb darkening) */
    Matrix<Scalar> taylor;                                                     /**< Temporal expansion basis */
    Scalar radian;                                                             /**< Conversion factor from degrees to radians */
    
    //! Constructor for all map types
    explicit Map (
        int ydeg,
        int udeg,
        int fdeg,
        int Nw,
        int Nt
    ) :
        ydeg(ydeg), 
        udeg(udeg),
        fdeg(fdeg),
        deg(ydeg + udeg + fdeg),
        Ny((ydeg + 1) * (ydeg + 1)), 
        Nu(udeg + 1),
        Nf((fdeg + 1) * (fdeg + 1)),
        Np((udeg + 1) * (udeg + 1)),
        Nw(Nw),
        Nt(Nt),
        N((deg + 1) * (deg + 1)),
        data(ydeg),
        y(Ny * Nt, S::LimbDarkened ? 1 : Nw),
        u(Nu, S::LimbDarkened ? Nw : 1),
        f(Nf),
        filter_on(fdeg > 0),
        inc(90.0),
        obl(0.0),
        B(ydeg, udeg, fdeg),
        W(ydeg, inc, obl),
        G(deg),
        L(udeg, Nw)
    {
        // Bounds checks
        if ((ydeg < 0) || (ydeg > STARRY_MAX_LMAX))
            throw std::out_of_range(
                "Spherical harmonic degree out of range.");
        if ((udeg < 0) || (udeg > STARRY_MAX_LMAX))
            throw std::out_of_range(
                "Limb darkening degree out of range.");
        if ((fdeg < 0) || (fdeg > STARRY_MAX_LMAX))
            throw std::out_of_range(
                "Filter degree out of range.");
        if ((deg > STARRY_MAX_LMAX))
            throw std::out_of_range(
                "Total map degree out of range.");
        if ((Nw < 1) || (Nt < 1))
            throw std::out_of_range(
                "The number of temporal / spectral terms must be positive.");
        
        // \todo There appear to be flux normalization issues for reflected-light
        // maps with non-limb darkening filters. Look into these before enabling
        // this feature.
        if ((S::Reflected) && (fdeg > 0))
            throw std::runtime_error(
                "Filters are not yet available for reflected light maps.");
        
        radian = pi<Scalar>() / 180.;
        reset();
        resize_arrays();
    };

public:

    //! Constructor for the default map
    template <
        typename U=S, 
        typename=EnableIf<!(U::Spectral || U::Temporal || U::LimbDarkened)>
    >
    explicit Map (
        int ydeg,
        int udeg,
        int fdeg
    ) : Map(ydeg, udeg, fdeg, 1, 1) {}

    //! Constructor for spectral & temporal maps
    template <
        typename U=S, 
        typename=EnableIf<(U::Spectral || U::Temporal) && !U::LimbDarkened>
    >
    explicit Map (
        int ydeg,
        int udeg,
        int fdeg,
        int nterms
    ) : Map(ydeg, udeg, fdeg,
            U::Spectral ? nterms : 1,
            U::Temporal ? nterms : 1) {}

    //! Constructor for the single-wavelength limb-darkened map
    template <
        typename U=S, 
        typename=EnableIf<U::LimbDarkened && !U::Spectral>
    >
    explicit Map (
        int udeg
    ) : Map(0, udeg, 0, 1, 1) {}

    //! Constructor for the spectral limb-darkened map
    // Note that we need to hack SFINAE a little differently
    // to avoid re-declaration of the <int, int> specialization
    template <typename U=S>
    explicit Map (
        int udeg,
        int nterms,
        EnableIf<U::LimbDarkened && U::Spectral>* = 0
    ) : Map(0, udeg, 0, nterms, 1) {}

    // Public methods
    #include "public/io.h"
    #include "public/oper.h"
    #include "public/intensity.h"
    #include "public/flux.h"

}; // class Map




