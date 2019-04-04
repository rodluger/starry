/**
\file data.h
\brief Defines classes used for storing arrays and matrices.

*/

template <class Scalar>
class Data
{
protected:

public:
    
    // Pybind cache
    RowMatrix<Scalar> DXDt;
    RowMatrix<Scalar> DXDtheta;
    RowMatrix<Scalar> DXDxo;
    RowMatrix<Scalar> DXDyo;
    RowMatrix<Scalar> DXDro;
    RowMatrix<Scalar> DXDsource;
    RowMatrix<Scalar> DXDu;
    RowMatrix<Scalar> DXDinc;
    RowMatrix<Scalar> DXDobl;
    RowMatrix<Scalar> X;

    Vector<Scalar> flux;
    Vector<Scalar> DfDb;
    Vector<Scalar> DfDro;
    Matrix<Scalar> DfDu;

    Matrix<Scalar> flux_spectral;
    Matrix<Scalar> DfDb_spectral;
    Matrix<Scalar> DfDro_spectral;

    // Cache
    std::vector<Matrix<Scalar>> EulerD;
    std::vector<Matrix<Scalar>> EulerR;

    //! Constructor
    Data (
        const int ydeg
    ) :
        EulerD(ydeg + 1),
        EulerR(ydeg + 1)
    {
        for (int l = 0; l < ydeg + 1; ++l) {
            int sz = 2 * l + 1;
            EulerD[l].resize(sz, sz);
            EulerR[l].resize(sz, sz);
        }
    };

}; // class Data