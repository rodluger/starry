//! Gradient indices
struct GradientIndices {
    const int t;
    const int theta;
    const int xo;
    const int yo;
    const int ro;
    const int nx;
    int y;
    int ny;
    int u;
    int nu;
    int ndim;
    explicit GradientIndices(const int lmax) :
        t(0),
        theta(1),
        xo(2),
        yo(3),
        ro(4),
        nx(5),
        y(nx),
        ny((lmax + 1) * (lmax + 1)),
        u(y + ny),
        nu(lmax),
        ndim(nx + ny + nu) {}
};

//! The Map class
template <class S>
class Map 
{

public:

    // Types
    using Scalar = typename S::Scalar;
    using MapType = typename S::MapType;
    using CoeffType = typename S::CoeffType;
    using FluxType = typename S::FluxType;
    using GradType = typename S::GradType;

    // Public variables
    const int lmax;
    const int N;
    const int ncol;
    const int nflx;
    const int expansion;
    Cache<S> cache;
    GradientIndices idx;

protected:

    // Internal methods
    inline void computeIntensity_ (
        const Scalar& t,
        const Scalar& theta,
        const Scalar& x_,
        const Scalar& y_,
        Ref<FluxType> intensity
    );

    template <typename Derived>
    inline void renderMap_ (
        const Scalar& t,
        const Scalar& theta,
        int res,
        MatrixBase<Derived> const & intensity
    );

    inline void computeFlux_ (
        const Scalar& t,
        const Scalar& theta, 
        const Scalar& xo, 
        const Scalar& yo, 
        const Scalar& ro, 
        Ref<FluxType> flux
    );

    inline void computeFlux_ (
        const Scalar& t,
        const Scalar& theta, 
        const Scalar& xo, 
        const Scalar& yo, 
        const Scalar& ro, 
        Ref<FluxType> flux, 
        Ref<GradType> gradient
    );

    template <class U>
    inline void random_ (
        const Vector<Scalar>& power,
        const U& seed,
        int col
    );

    inline void computeFluxLD (
        const Scalar& t,
        const Scalar& b, 
        const Scalar& ro, 
        Ref<FluxType> flux
    );

    inline void computeFluxYlm (
        const Scalar& t,
        const Scalar& theta,
        const Scalar& xo,
        const Scalar& yo,   
        const Scalar& b, 
        const Scalar& ro, 
        Ref<FluxType> flux
    );

    inline void computeFluxYlmLD (
        const Scalar& t,
        const Scalar& theta,
        const Scalar& xo,
        const Scalar& yo,   
        const Scalar& b, 
        const Scalar& ro, 
        Ref<FluxType> flux
    );

    inline void computeFluxLD (
        const Scalar& t,
        const Scalar& xo, 
        const Scalar& yo, 
        const Scalar& b, 
        const Scalar& ro, 
        Ref<FluxType> flux, 
        Ref<GradType> gradient
    );

    inline void computeFluxYlm (
        const Scalar& t,
        const Scalar& theta,
        const Scalar& xo,
        const Scalar& yo,   
        const Scalar& b, 
        const Scalar& ro, 
        Ref<FluxType> flux,
        Ref<GradType> gradient
    );

    inline void computeFluxYlmLD (
        const Scalar& t,
        const Scalar& theta,
        const Scalar& xo,
        const Scalar& yo,   
        const Scalar& b, 
        const Scalar& ro, 
        Ref<FluxType> flux,
        Ref<GradType> gradient
    );

    inline void checkDegree ();

    inline void computeDegreeY ();

    inline void computeDegreeU ();

    inline void computeP (int res);

    inline void computeWigner ();

    inline void computeLDPolynomial ();

    inline void rotateIntoCache (
        const Scalar& theta,
        bool compute_matrices=false
    );

    inline void rotateByAxisAngle (
        const UnitVector<Scalar>& axis_,
        const Scalar& costheta,
        const Scalar& sintheta,
        MapType& y_
    );

    inline void limbDarken (
        const MapType& poly, 
        MapType& poly_ld, 
        bool gradient=false
    );

    // Internal variables
    MapType y;
    MapType u;
    UnitVector<Scalar> axis;                                                   /**< The axis of rotation for the map */
    basis::Basis<Scalar> B;                                                    /**< Basis transform stuff */
    rotation::Wigner<MapType> W;                                               /**< Ylm rotation stuff */
    limbdark::GreensLimbDark<Scalar> L;                                        /**< The occultation integral solver class (optimized for limb darkening) */
    Vector<Scalar> tbasis;
    Vector<Scalar> dtbasis;
    int u_deg;                                                                 /**< Highest degree set by the user in the limb darkening vector */
    int y_deg;
    Scalar radian;
     
    //! Constructor for all map types
    explicit Map (
        int lmax,
        int ncol,
        int nflx,
        int expansion
    ) :
        lmax(lmax), 
        N((lmax + 1) * (lmax + 1)), 
        ncol(ncol), 
        nflx(nflx),
        expansion(expansion),
        cache(lmax, ncol),
        idx(lmax),
        B(lmax),
        W(lmax, ncol, (*this).y, (*this).axis),
        L(lmax),
        tbasis(ncol),
        dtbasis(ncol)
    {
        if (ncol < 1) throw errors::ValueError(
            "The number of map columns must be positive.");
        radian = pi<Scalar>() / 180.;
        tbasis(0) = 1.0;
        dtbasis(0) = 0.0;
        reset();
    };

public:

    //! Constructor for default maps
    template<typename U=S, typename=IsDefault<U>>
    explicit Map (
        int lmax
    ) : Map(lmax, 1, 1, STARRY_EXPANSION_NONE) {}

    //! Constructor for spectral maps
    template<typename U=S, typename=IsSpectral<U>>
    explicit Map (
        int lmax,
        int ncol
    ) : Map(lmax, ncol, ncol, STARRY_EXPANSION_NONE) {}

    //! Constructor for temporal maps
    template<typename U=S, typename=IsTemporal<U>>
    explicit Map (
        int lmax,
        int ncol,
        int expansion=STARRY_EXPANSION_TAYLOR
    ) : Map(lmax, ncol, 1, expansion) {}

    // Inline declarations. These are methods whose
    // call signature depends on the map type or on
    // macros, and must be declared inline.
    #include "inline/io.h"
    #include "inline/oper.h"
    #include "inline/flux.h"
    #include "inline/python_interface.h"

    inline void updateIndices_ () {
        computeDegreeY();
        computeDegreeU();
    }

    inline void setY (
        const MapType& y_
    );

    inline void setY (
        int l, 
        int m,
        const Ref<const CoeffType>& coeff
    );

    inline void setY (
        int l, 
        int m,
        const Scalar& coeff_
    );

    inline MapType getY () const;

    inline void setU (
        const MapType& u_
    );

    inline void setU (
        int l, 
        const Ref<const CoeffType>& coeff
    );

    inline void setU (
        int l, 
        const Scalar& coeff
    );

    inline MapType getU () const;

    inline void setAxis (
        const UnitVector<Scalar>& axis_
    );
    
    inline UnitVector<Scalar> getAxis () const;

    inline void reset ();

    inline void addSpot (
        const CoeffType& amp,
        const Scalar& sigma,
        const Scalar& lat=0,
        const Scalar& lon=0,
        int l=-1
    );

    inline void rotate (
        const Scalar& theta
    );

    std::string info ();

}; // class Map




