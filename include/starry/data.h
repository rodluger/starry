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
    RowMatrix<Scalar> DADt;
    RowMatrix<Scalar> DADtheta;
    RowMatrix<Scalar> DADxo;
    RowMatrix<Scalar> DADyo;
    RowMatrix<Scalar> DADro;
    RowMatrix<Scalar> A;

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