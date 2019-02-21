/**
\file cache.h
\brief Defines classes used for caching of temporary map variables.

*/

template <class S, typename T=void> 
class Cache_;

template <class S> 
class Cache_<S, IsDefault<S>> 
{
public:

    using Scalar = typename S::Scalar;

    // Map vectors
    Vector<Scalar> Ry;
    Vector<Scalar> RRy;
    Vector<Scalar> A1Ry;
    Vector<Scalar> ARRy;
    Vector<Scalar> pupy;

    // Derivatives
    Vector<Scalar> DRDthetay;
    Matrix<Scalar> DpuDu;
    Vector<Scalar> DpuDy0;
    Vector<Scalar> vTDpupyDy;
    Vector<Scalar> vTDpupyDu;
    RowVector<Scalar> vTDpupyDpu;
    RowVector<Scalar> vTDpupyDpy;
    RowVector<Scalar> vTDpupyDpyA1;
    RowVector<Scalar> vTDpupyDpyA1R;
    RowVector<Scalar> vTDpupyDpyA1RR;
    RowVector<Scalar> vTDpupyDpyA1DRDomega;

    Cache_ (
        int lmax,
        int ncoly,
        int ncolu,
        int nflx
    ) {
        int N = (lmax + 1) * (lmax + 1);
        Ry.resize(N);
        RRy.resize(N);
        A1Ry.resize(N);
        ARRy.resize(N);
        pupy.resize(N);
        DRDthetay.resize(N);
        DpuDu.resize(lmax + 1, N);
        DpuDy0.resize(N);
        vTDpupyDy.resize(N);
        vTDpupyDu.resize(lmax + 1);
        vTDpupyDpu.resize(N);
        vTDpupyDpy.resize(N);
        vTDpupyDpyA1.resize(N);
        vTDpupyDpyA1R.resize(N);
        vTDpupyDpyA1RR.resize(N);
        vTDpupyDpyA1DRDomega.resize(N);
    }

};

template <class S> 
class Cache_<S, IsSpectral<S>> 
{
public:

    using Scalar = typename S::Scalar;

    // Map matrices
    Matrix<Scalar> Ry;
    Matrix<Scalar> RRy;
    Matrix<Scalar> A1Ry;
    Matrix<Scalar> ARRy;
    Matrix<Scalar> pupy;

    // Derivatives
    Matrix<Scalar> DRDthetay;
    std::vector<Matrix<Scalar>> DpuDu;
    Matrix<Scalar> DpuDy0;
    Matrix<Scalar> vTDpupyDy;
    Matrix<Scalar> vTDpupyDu;
    Matrix<Scalar> vTDpupyDpu;
    Matrix<Scalar> vTDpupyDpy;
    Matrix<Scalar> vTDpupyDpyA1;
    Matrix<Scalar> vTDpupyDpyA1R;
    Matrix<Scalar> vTDpupyDpyA1RR;
    Matrix<Scalar> vTDpupyDpyA1DRDomega;

    Cache_ (
        int lmax,
        int ncoly,
        int ncolu,
        int nflx
    ) {
        int N = (lmax + 1) * (lmax + 1);
        Ry.resize(N, ncoly);
        RRy.resize(N, ncoly);
        A1Ry.resize(N, ncoly);
        ARRy.resize(N, ncoly);
        pupy.resize(N, ncoly);
        DRDthetay.resize(N, ncoly);
        DpuDu.resize(ncolu);
        for (int i = 0; i < ncolu; ++i) {
            DpuDu[i].resize(lmax + 1, N);
        }
        DpuDy0.resize(N, ncolu);
        vTDpupyDy.resize(N, ncoly);
        vTDpupyDu.resize(lmax + 1, ncolu);
        vTDpupyDpu.resize(ncolu, N);
        vTDpupyDpy.resize(ncoly, N);
        vTDpupyDpyA1.resize(ncoly, N); 
        vTDpupyDpyA1R.resize(ncoly, N);   
        vTDpupyDpyA1RR.resize(ncoly, N); 
        vTDpupyDpyA1DRDomega.resize(ncoly, N);
    }

};

template <class S> 
class Cache_<S, IsTemporal<S>> 
{
public:

    using Scalar = typename S::Scalar;

    // Contracted map vectors
    Vector<Scalar> Ry;
    Matrix<Scalar> RY;                                                         /**< Uncontracted R . Y matrix */
    Vector<Scalar> RRy;
    Vector<Scalar> A1Ry;
    Matrix<Scalar> A1RY;
    Vector<Scalar> ARRy;
    Vector<Scalar> pupy;

    // Derivatives
    Vector<Scalar> DRDthetay;
    Matrix<Scalar> DpuDu;
    Vector<Scalar> DpuDy0;
    Matrix<Scalar> vTDpupyDy;
    Vector<Scalar> vTDpupyDu;
    RowVector<Scalar> vTDpupyDpu;
    RowVector<Scalar> vTDpupyDpy;
    RowVector<Scalar> vTDpupyDpyA1;
    RowVector<Scalar> vTDpupyDpyA1R;
    RowVector<Scalar> vTDpupyDpyA1RR;
    RowVector<Scalar> vTDpupyDpyA1DRDomega;

    Cache_ (
        int lmax,
        int ncoly,
        int ncolu,
        int nflx
    ) {
        int N = (lmax + 1) * (lmax + 1);
        Ry.resize(N);
        RY.resize(N, ncoly);
        RRy.resize(N);
        A1Ry.resize(N);
        A1RY.resize(N, ncoly);
        ARRy.resize(N);
        pupy.resize(N);
        DRDthetay.resize(N);
        DpuDu.resize(lmax + 1, N);
        DpuDy0.resize(N);
        vTDpupyDy.resize(N, ncoly);
        vTDpupyDu.resize(lmax + 1);
        vTDpupyDpu.resize(N);
        vTDpupyDpy.resize(N);
        vTDpupyDpyA1.resize(N);
        vTDpupyDpyA1R.resize(N);
        vTDpupyDpyA1RR.resize(N);
        vTDpupyDpyA1DRDomega.resize(N);
    }

};

template <class S>
class Cache : public Cache_<S>
{
protected:

    // Types
    using Scalar = typename S::Scalar;
    using YType = typename S::YType;
    using YCoeffType = typename S::YCoeffType;
    using UType = typename S::UType;
    using TSType = typename S::TSType;
    using FluxType = typename S::FluxType;

public:
    
    int lmax;
    int ncoly;
    int ncolu;
    int nflx;
    int N;

    // Flags
    bool compute_g;
    bool compute_Zeta;
    bool compute_YZeta;
    bool compute_degree_y;
    bool compute_degree_u;
    bool compute_P;
    bool compute_p;
    bool compute_p_grad;

    // Cached variables
    int res;
    Scalar taylort;
    Scalar theta;
    Scalar theta_with_grad;
    Matrix<Scalar> P;                                                          /**< The change of basis matrix from Ylms to pixels */
    Vector<Scalar> I;                                                          /**< The illumination matrix (reflecte light maps only) */
    Scalar sx;
    Scalar sy;
    Scalar sz;
    UType g;
    Matrix<Scalar> DgDu;                                                       /**< Deriv of Agol `g` coeffs w/ respect to the limb darkening coeffs */
    UType p;
    FluxType sTADRDphiRy_b;
    FluxType dFdb;
    RowVector<Scalar> pT;
    RowVector<Scalar> sTA;
    RowVector<Scalar> sTA2;
    RowVector<Scalar> sTAR;
    RowVector<Scalar> sTADRDphi;
    std::vector<Matrix<Scalar>> EulerD;
    std::vector<Matrix<Scalar>> EulerR;

    // Pybind cache
    TSType pb_flux;
    TSType pb_Dt;
    TSType pb_Dtheta;
    TSType pb_Dxo;
    TSType pb_Dyo;
    TSType pb_Dro;
    RowMatrix<Scalar> pb_Dy;
    RowMatrix<Scalar> pb_Du;
    RowMatrix<Scalar> pb_Dsource;
    RowMatrix<Scalar> pb_DADt;
    RowMatrix<Scalar> pb_DADtheta;
    RowMatrix<Scalar> pb_DADxo;
    RowMatrix<Scalar> pb_DADyo;
    RowMatrix<Scalar> pb_DADro;
    RowMatrix<Scalar> pb_A;

    //
    inline void yChanged () {
        compute_YZeta = true;
        compute_degree_y = true;
        compute_P = true;
        theta = NAN;
        theta_with_grad = NAN;
        // Recall that the normalization of the LD
        // polynomial depends on Y_{0,0}
        compute_p = true;
        compute_p_grad = true;
        compute_g = true;
    }

    inline void uChanged () {
        compute_degree_u = true;
        compute_p = true;
        compute_p_grad = true;
        compute_g = true;
    }

    inline void axisChanged () {
        compute_Zeta = true;
        compute_YZeta = true;
        theta = NAN;
        theta_with_grad = NAN;
    }

    inline void mapRotated () {
        compute_P = true;
        theta = NAN;
        theta_with_grad = NAN;
    }

    //! Reset all flags
    inline void reset () 
    {
        compute_g = true;
        compute_Zeta = true;
        compute_YZeta = true;
        compute_degree_y = true;
        compute_degree_u = true;
        compute_P = true;
        compute_p = true;
        compute_p_grad = true;
        res = -1;
        sx = 0;
        sy = 0;
        sz = 0;
        taylort = NAN;
        theta = NAN;
        theta_with_grad = NAN;
    };

    //! Constructor
    Cache (
        int lmax,
        int ncoly,
        int ncolu,
        int nflx
    ) :
        Cache_<S>(lmax, ncoly, ncolu, nflx),
        lmax(lmax),
        ncoly(ncoly),
        ncolu(ncolu),
        nflx(nflx),
        N((lmax + 1) * (lmax + 1)),
        g(lmax + 1, ncolu),
        DgDu(lmax * ncolu, lmax + 1),
        p(N, ncolu),        
        sTADRDphiRy_b(nflx),
        dFdb(nflx),
        pT(N),
        sTA(N),
        sTA2(N),
        sTAR(N),
        sTADRDphi(N),
        EulerD(lmax + 1),
        EulerR(lmax + 1)
    {
        for (int l = 0; l < lmax + 1; ++l) {
            int sz = 2 * l + 1;
            EulerD[l].resize(sz, sz);
            EulerR[l].resize(sz, sz);
        }
        reset();
    };

}; // class Cache