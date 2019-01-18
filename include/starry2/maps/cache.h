template <class S>
class Cache
{
protected:

    // Types
    using Scalar = typename S::Scalar;
    using YType = typename S::YType;
    using YCoeffType = typename S::YCoeffType;
    using UType = typename S::UType;
    using TSType = typename S::TSType;
    using CtrYType = typename S::CtrYType;
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
    UType g;
    Matrix<Scalar> DgDu;                                                       /**< Deriv of Agol `g` coeffs w/ respect to the limb darkening coeffs */
    UType p;
    YType RyUncontracted;
    CtrYType Ry;
    CtrYType RRy;
    CtrYType A1Ry;
    CtrYType ARRy;
    CtrYType DRDthetay;
    CtrYType pupy;
    FluxType sTADRDphiRy_b;
    FluxType dFdb;
    RowVector<Scalar> pT;
    RowVector<Scalar> sTA;
    RowVector<Scalar> sTAR;
    RowVector<Scalar> sTADRDphi;
    std::vector<Matrix<Scalar>> EulerD;
    std::vector<Matrix<Scalar>> EulerR;


// TODO template these and maybe transpose them
#if defined(_STARRY_DEFAULT_)
    Matrix<Scalar> DpuDu;
    Vector<Scalar> DpuDy0;
    Vector<Scalar> rTDpupyDy;
    Vector<Scalar> rTDpupyDu;
    RowVector<Scalar> rTDpupyDpu;
    RowVector<Scalar> rTDpupyDpy;
    RowVector<Scalar> rTDpupyDpyA1R;
#elif defined(_STARRY_SPECTRAL_)
    std::vector<Matrix<Scalar>> DpuDu;
    Matrix<Scalar> DpuDy0;
    Matrix<Scalar> rTDpupyDy;
    Matrix<Scalar> rTDpupyDu;
    Matrix<Scalar> rTDpupyDpu;
    Matrix<Scalar> rTDpupyDpy;
    Matrix<Scalar> rTDpupyDpyA1R;
#else
    Matrix<Scalar> DpuDu;
    Vector<Scalar> DpuDy0;
    Matrix<Scalar> rTDpupyDy;
    Vector<Scalar> rTDpupyDu;
    RowVector<Scalar> rTDpupyDpu;
    RowVector<Scalar> rTDpupyDpy;
    RowVector<Scalar> rTDpupyDpyA1R;

#endif

    // Pybind cache
    TSType pb_flux;
    TSType pb_Dt;
    TSType pb_Dtheta;
    TSType pb_Dxo;
    TSType pb_Dyo;
    TSType pb_Dro;
    RowMatrix<Scalar> pb_Dy;
    RowMatrix<Scalar> pb_Du;

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
        lmax(lmax),
        ncoly(ncoly),
        ncolu(ncolu),
        nflx(nflx),
        N((lmax + 1) * (lmax + 1)),
        g(lmax + 1, ncolu),
        DgDu(lmax * ncolu, lmax + 1),
        p(N, ncolu),
        RyUncontracted(N, ncoly),
        Ry(N, nflx),
        RRy(N, nflx),
        A1Ry(N, nflx),
        ARRy(N, nflx),
        DRDthetay(N, nflx),
        pupy(N, nflx),
        sTADRDphiRy_b(nflx),
        dFdb(nflx),
        pT(N),
        sTA(N),
        sTAR(N),
        sTADRDphi(N),
        EulerD(lmax + 1),
        EulerR(lmax + 1),

// TODO template these
#if defined(_STARRY_DEFAULT_)
        DpuDu(lmax + 1, N),
        DpuDy0(N),
        rTDpupyDy(N),
        rTDpupyDu(lmax + 1),
        rTDpupyDpu(N),
        rTDpupyDpy(N),
        rTDpupyDpyA1R(N)
#elif defined(_STARRY_SPECTRAL_)
        DpuDu(ncolu),
        DpuDy0(N, ncolu),
        rTDpupyDy(N, ncoly),
        rTDpupyDu(lmax + 1, ncolu),
        rTDpupyDpu(ncolu, N),
        rTDpupyDpy(ncoly, N),
        rTDpupyDpyA1R(ncoly, N)
#else
        DpuDu(lmax + 1, N),
        DpuDy0(N),
        rTDpupyDy(N, ncoly),
        rTDpupyDu(lmax + 1),
        rTDpupyDpu(N),
        rTDpupyDpy(N),
        rTDpupyDpyA1R(N)
#endif
    {
        for (int l = 0; l < lmax + 1; ++l) {
            int sz = 2 * l + 1;
            EulerD[l].resize(sz, sz);
            EulerR[l].resize(sz, sz);
        }

#if defined(_STARRY_SPECTRAL_)
        for (int i = 0; i < ncolu; ++i) {
            DpuDu[i].resize(lmax + 1, N);
        }
#endif

        reset();
    };

}; // class Cache