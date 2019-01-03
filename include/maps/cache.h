template <class S>
class Cache
{
protected:

    // Types
    using Scalar = typename S::Scalar;
    using YType = typename S::YType;
    using UType = typename S::UType;

public:
    
    int lmax;
    int ncol;
    int nflx;
    int N;

    // Flags
    bool compute_c;
    bool compute_Zeta;
    bool compute_YZeta;
    bool compute_degree_y;
    bool compute_degree_u;
    bool compute_P;
    bool compute_p_u;

    // Cached variables
    int res;
    Scalar taylort;
    Scalar theta;
    Scalar theta_with_grad;
    Matrix<Scalar> P;                                                          /**< The change of basis matrix from Ylms to pixels */
    UType c;
    Matrix<Scalar> dcdu;                                                       /**< Deriv of Agol `c` coeffs w/ respect to the limb darkening coeffs */
    UType p_u;
    RowMatrix<Scalar> gradient;
    YType Ry;
    YType A1Ry;
    YType dRdthetay;
    YType p_uy;
    RowVector<Scalar> pT;
    std::vector<Matrix<Scalar>> EulerD;
    std::vector<Matrix<Scalar>> EulerR;

    //
    inline void yChanged () {
        compute_YZeta = true;
        compute_degree_y = true;
        compute_P = true;
        theta = NAN;
        theta_with_grad = NAN;
        // Recall that the normalization of the LD
        // polynomial depends on Y_{0,0}
        compute_p_u = true;
        compute_c = true;
    }

    inline void uChanged () {
        compute_degree_u = true;
        compute_p_u = true;
        compute_c = true;
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
        compute_c = true;
        compute_Zeta = true;
        compute_YZeta = true;
        compute_degree_y = true;
        compute_degree_u = true;
        compute_P = true;
        compute_p_u = true;
        res = -1;
        taylort = NAN;
        theta = NAN;
        theta_with_grad = NAN;
    };

    //! Constructor
    Cache (
        int lmax,
        int ncol,
        int nflx
    ) :
        lmax(lmax),
        ncol(ncol),
        nflx(nflx),
        N((lmax + 1) * (lmax + 1)),
        c(lmax + 1, nflx),
        dcdu(lmax * nflx, lmax + 1),
        p_u(N, nflx),
        Ry(N, ncol),
        A1Ry(N, ncol),
        dRdthetay(N, ncol),
        p_uy(N, ncol),
        pT(N),
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