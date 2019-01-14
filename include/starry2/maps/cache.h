template <class S>
class Cache
{
protected:

    // Types
    using Scalar = typename S::Scalar;
    using YType = typename S::YType;
    using YCoeffType = typename S::YCoeffType;
    using UType = typename S::UType;
    using FType = typename S::FType;
    using CtrYType = typename S::CtrYType;

public:
    
    int lmax;
    int ncoly;
    int ncolu;
    int nflx;
    int N;

    // Flags
    bool compute_agol_g;
    bool compute_Zeta;
    bool compute_YZeta;
    bool compute_degree_y;
    bool compute_degree_u;
    bool compute_P;
    bool compute_agol_p;
    bool compute_agol_p_grad;

    // Cached variables
    int res;
    Scalar taylort;
    Scalar theta;
    Scalar theta_with_grad;
    Matrix<Scalar> P;                                                          /**< The change of basis matrix from Ylms to pixels */
    UType agol_g;
    Matrix<Scalar> dAgolGdu;                                                   /**< Deriv of Agol `g` coeffs w/ respect to the limb darkening coeffs */
    UType agol_p;
    YType RyUncontracted;
    CtrYType Ry;
    CtrYType A1Ry;
    CtrYType dRdthetay;
    CtrYType p_uy;
    RowVector<Scalar> pT;
    std::vector<Matrix<Scalar>> EulerD;
    std::vector<Matrix<Scalar>> EulerR;
    std::vector<Matrix<Scalar>> dLDdp;
    std::vector<Matrix<Scalar>> dLDdagol_p;
    std::vector<Matrix<Scalar>> dAgolPdu;
    std::vector<Matrix<Scalar>> dAgolPdy; // TODO: This is sparse

    std::vector<Matrix<Scalar>> dLDdu;
    std::vector<Matrix<Scalar>> dLDdy;

    // Pybind cache
    FType pb_flux;
    FType pb_time;
    FType pb_theta;
    FType pb_xo;
    FType pb_yo;
    FType pb_ro;
    RowMatrix<Scalar> pb_y;
    RowMatrix<Scalar> pb_u;

    //
    inline void yChanged () {
        compute_YZeta = true;
        compute_degree_y = true;
        compute_P = true;
        theta = NAN;
        theta_with_grad = NAN;
        // Recall that the normalization of the LD
        // polynomial depends on Y_{0,0}
        compute_agol_p = true;
        compute_agol_p_grad = true;
        compute_agol_g = true;
    }

    inline void uChanged () {
        compute_degree_u = true;
        compute_agol_p = true;
        compute_agol_p_grad = true;
        compute_agol_g = true;
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
        compute_agol_g = true;
        compute_Zeta = true;
        compute_YZeta = true;
        compute_degree_y = true;
        compute_degree_u = true;
        compute_P = true;
        compute_agol_p = true;
        compute_agol_p_grad = true;
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
        agol_g(lmax + 1, ncolu),
        dAgolGdu(lmax * ncolu, lmax + 1),
        agol_p(N, ncolu),
        RyUncontracted(N, ncoly),
        Ry(N, nflx),
        A1Ry(N, nflx),
        dRdthetay(N, nflx),
        p_uy(N, nflx),
        pT(N),
        EulerD(lmax + 1),
        EulerR(lmax + 1),
        dLDdp(ncoly),
        dLDdagol_p(ncolu),
        dAgolPdu(ncolu),
        dAgolPdy(ncolu),
        dLDdu(ncolu),
        dLDdy(ncoly)
    {
        for (int l = 0; l < lmax + 1; ++l) {
            int sz = 2 * l + 1;
            EulerD[l].resize(sz, sz);
            EulerR[l].resize(sz, sz);
        }
        for (int i = 0; i < ncoly; ++i) {
            dLDdp[i].resize(N, N);
            dLDdy[i].resize(N, N);
        }
        for (int i = 0; i < ncolu; ++i) {
            // TODO: CHECK THESE SHAPES
            dLDdagol_p[i].resize(N, N);
            dAgolPdu[i].resize(N, N);
            dAgolPdy[i].resize(N, N);
            dLDdu[i].resize(N, N);
        }
        reset();
    };

}; // class Cache