//! The Map class
template <class S=Default<double>>
class Map 
{

public:

    // Types
    using Scalar = typename S::Scalar;
    using YType = typename S::YType;
    using YCoeffType = typename S::YCoeffType;
    using UType = typename S::UType;
    using UCoeffType = typename S::UCoeffType;
    using CtrYType = typename S::CtrYType;

    // Public variables
    const int lmax;
    const int N;
    const int ncoly;
    const int ncolu;
    const int nflx;
    Cache<S> cache;

protected:

    // Internal methods
    template <typename T1>
    inline void computeIntensity_ (
        const Scalar& theta,
        const Scalar& x_,
        const Scalar& y_,
        MatrixBase<T1> const & intensity
    );

    template <typename T1>
    inline void renderMap_ (
        const Scalar& theta,
        int res,
        MatrixBase<T1> const & intensity
    );

    template <typename T1>
    inline void computeFlux_ (
        const Scalar& theta, 
        const Scalar& xo, 
        const Scalar& yo, 
        const Scalar& ro, 
        MatrixBase<T1> const & flux
    );

    template <typename T1, typename T2, typename T3, typename T4, 
              typename T5, typename T6, typename T7, typename T8>
    inline void computeFlux_ (
        const Scalar& theta, 
        const Scalar& xo, 
        const Scalar& yo, 
        const Scalar& ro, 
        MatrixBase<T1> const & flux, 
        MatrixBase<T2> const & dt,
        MatrixBase<T3> const & dtheta,
        MatrixBase<T4> const & dxo,
        MatrixBase<T5> const & dyo,
        MatrixBase<T6> const & dro,
        MatrixBase<T7> const & dy,
        MatrixBase<T8> const & du
    );

    template <class U>
    inline void random_ (
        const Vector<Scalar>& power,
        const U& seed,
        int col
    );

    template <typename T1>
    inline void computeFluxLD (
        const Scalar& b, 
        const Scalar& ro, 
        MatrixBase<T1> const & flux
    );

    template <typename T1>
    inline void computeFluxYlm (
        const Scalar& theta,
        const Scalar& xo,
        const Scalar& yo,   
        const Scalar& b, 
        const Scalar& ro, 
        MatrixBase<T1> const & flux
    );

    template <typename T1>
    inline void computeFluxYlmLD (
        const Scalar& theta,
        const Scalar& xo,
        const Scalar& yo,   
        const Scalar& b, 
        const Scalar& ro, 
        MatrixBase<T1> const & flux
    );

    template <typename T1, typename T2, typename T3, typename T4, 
              typename T5, typename T6, typename T7, typename T8>
    inline void computeFluxLD (
        const Scalar& xo, 
        const Scalar& yo, 
        const Scalar& b, 
        const Scalar& ro, 
        MatrixBase<T1> const & flux, 
        MatrixBase<T2> const & dt,
        MatrixBase<T3> const & dtheta,
        MatrixBase<T4> const & dxo,
        MatrixBase<T5> const & dyo,
        MatrixBase<T6> const & dro,
        MatrixBase<T7> const & dy,
        MatrixBase<T8> const & du
    );

    template <typename T1, typename T2, typename T3, typename T4, 
              typename T5, typename T6, typename T7, typename T8>
    inline void computeFluxYlm (
        const Scalar& theta,
        const Scalar& xo,
        const Scalar& yo,   
        const Scalar& b, 
        const Scalar& ro, 
        MatrixBase<T1> const & flux, 
        MatrixBase<T2> const & dt,
        MatrixBase<T3> const & dtheta,
        MatrixBase<T4> const & dxo,
        MatrixBase<T5> const & dyo,
        MatrixBase<T6> const & dro,
        MatrixBase<T7> const & dy,
        MatrixBase<T8> const & du
    );

    template <typename T1, typename T2, typename T3, typename T4, 
              typename T5, typename T6, typename T7, typename T8>
    inline void computeFluxYlmLD (
        const Scalar& theta,
        const Scalar& xo,
        const Scalar& yo,   
        const Scalar& b, 
        const Scalar& ro, 
        MatrixBase<T1> const & flux, 
        MatrixBase<T2> const & dt,
        MatrixBase<T3> const & dtheta,
        MatrixBase<T4> const & dxo,
        MatrixBase<T5> const & dyo,
        MatrixBase<T6> const & dro,
        MatrixBase<T7> const & dy,
        MatrixBase<T8> const & du
    );

    inline void checkDegree ();

    inline void computeDegreeY ();

    inline void computeDegreeU ();

    inline void computeP (int res);

    inline void computeWigner ();

    inline void computeLDPolynomial (
        bool gradient=false
    );

    inline void computeAgolGBasis ();

    inline void rotateByAxisAngle (
        const UnitVector<Scalar>& axis_,
        const Scalar& costheta,
        const Scalar& sintheta,
        YType& y_
    );

    inline void limbDarken (
        const CtrYType& poly, 
        CtrYType& poly_ld, 
        bool gradient=false
    );

    // Internal variables
    YType y;
    UType u;
    UnitVector<Scalar> axis;                                                   /**< The axis of rotation for the map */
    basis::Basis<Scalar> B;                                                    /**< Basis transform stuff */
    rotation::Wigner<YType> W;                                                 /**< Ylm rotation stuff */
    limbdark::GreensLimbDark<Scalar> L;                                        /**< The occultation integral solver class (optimized for limb darkening) */
    Vector<Scalar> taylor;
    int u_deg;                                                                 /**< Highest degree set by the user in the limb darkening vector */
    int y_deg;
    Scalar radian;
     
    //! Constructor for all map types
    explicit Map (
        int lmax,
        int ncoly,
        int ncolu,
        int nflx
    ) :
        lmax(lmax), 
        N((lmax + 1) * (lmax + 1)), 
        ncoly(ncoly), 
        ncolu(ncolu),
        nflx(nflx),
        cache(lmax, ncoly, ncolu, nflx),
        B(lmax),
        W(lmax, ncoly, (*this).y, (*this).axis),
        L(lmax),
        taylor(ncoly)
    {
        if ((lmax < 0) || (lmax > STARRY_MAX_LMAX))
            throw errors::ValueError(
                "Spherical harmonic degree out of range.");
        if ((ncoly < 1) || (ncolu < 1))
            throw errors::ValueError(
                "The number of map columns must be positive.");
        radian = pi<Scalar>() / 180.;
        taylor(0) = 1.0;
        reset();
    };

public:

    //! Constructor for single-column maps
    template<typename U=S, typename=IsSingleColumn<U>>
    explicit Map (
        int lmax
    ) : Map(lmax, 1, 1, 1) {}

    //! Constructor for multi-column maps
    template<typename U=S, typename=IsMultiColumn<U>>
    explicit Map (
        int lmax,
        int ncol
    ) : Map(lmax, ncol, 
            std::is_same<U, Spectral<Scalar>>::value ? ncol : 1,
            std::is_same<U, Spectral<Scalar>>::value ? ncol : 1) {}

    // Inline declarations. These are methods whose
    // call signature depends on the map type or on
    // macros, and must be declared inline.
    #include "inline/io.h"
    #include "inline/oper.h"
    #include "inline/deriv.h"
    #include "inline/flux.h"
    #include "inline/python_interface.h"

    // Public methods not really meant for user access
    inline void updateIndices_ () {
        computeDegreeY();
        computeDegreeU();
    }

    inline int getYDeg_ ();

    inline int getUDeg_ ();

    // I/O
    inline void setY (
        const YType& y_
    );

    inline void setY (
        int l, 
        int m,
        const Ref<const YCoeffType>& coeff
    );

    inline void setY (
        int l, 
        int m,
        const Scalar& coeff_
    );

    inline YType getY () const;

    inline void setU (
        const UType& u_
    );

    inline void setU (
        int l, 
        const Ref<const UCoeffType>& coeff
    );

    inline void setU (
        int l, 
        const Scalar& coeff
    );

    inline UType getU () const;

    inline void setAxis (
        const UnitVector<Scalar>& axis_
    );
    
    inline UnitVector<Scalar> getAxis () const;

    std::string info ();

    // Miscellaneous operations
    inline void reset ();

    inline void rotate (
        const Scalar& theta
    );

    inline void addSpot (
        const YCoeffType& amp,
        const Scalar& sigma,
        const Scalar& lat=0,
        const Scalar& lon=0,
        int l=-1
    );

}; // class Map




