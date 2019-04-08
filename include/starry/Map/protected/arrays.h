/**
Temporary vectors and matrices used throughout the code.

*/
RowVector<Scalar> rTA1RZetaInv;
RowVector<Scalar> rTA1RZetaInvRz;
RowVector<Scalar> sTA;
RowVector<Scalar> sTARz;
RowVector<Scalar> sTARzRZetaInv;
RowVector<Scalar> sTARzRZetaInvRz;
Matrix<Scalar> LA1;
Eigen::SparseMatrix<Scalar> A2LA1;
RowVector<Scalar> rTLA1;
RowVector<Scalar> rTA1;
RowVector<Scalar> rTA1Rz;
RowVector<Scalar> rTA1RzRZetaInv;
RowVector<Scalar> rTA1RzRZetaInvRz;
Eigen::SparseMatrix<Scalar> LA1_;
RowVector<Scalar> rTA1RZetaInvDRzDtheta;
RowVector<Scalar> dsTdrA;
RowVector<Scalar> dsTdbA;
RowVector<Scalar> dsTdrARz;
RowVector<Scalar> dsTdbARz;
RowVector<Scalar> sTADRzDw;
RowVector<Scalar> dsTdrARzRZetaInv;
RowVector<Scalar> dsTdbARzRZetaInv;
RowVector<Scalar> sTADRzDwRZetaInv;
RowVector<Scalar> sTARzRZetaInvDRzDtheta;
RowVector<Scalar> dsTdrARzRZetaInvRz;
RowVector<Scalar> dsTdbARzRZetaInvRz;
RowVector<Scalar> sTADRzDwRZetaInvRz;
RowVector<Scalar> sTADRzDwRZetaInvRzRZeta;
RowVector<Scalar> rTA1DRZetaInvDAngle;
RowVector<Scalar> rTA1DRZetaInvDAngleRz;
RowVector<Scalar> rTA1DRZetaInvDAngleRzRZeta;
RowVector<Scalar> rTA1RZetaInvRzDRZetaDAngle;
RowVector<Scalar> sTARzDRZetaInvDAngle;
RowVector<Scalar> sTARzDRZetaInvDAngleRz;
RowVector<Scalar> sTARzDRZetaInvDAngleRzRZeta;
RowVector<Scalar> sTARzRZetaInvRzDRZetaDAngle;
Vector<Matrix<Scalar>> dLdu;
Matrix<Scalar> rTdLduA1;
Matrix<Scalar> sTA2dLduA1;
Matrix<Scalar> sTA2dLduA1Rz;
RowMatrix<Scalar> X0;
RowMatrix<Scalar> Xp;
RowVector<Scalar> x_cache;
RowVector<Scalar> y_cache;

/**
Allocate the temporary vectors and matrices 
used throughout the code.

*/
void resize_arrays() {
    rTA1RZetaInv.resize(Ny);
    rTA1RZetaInvRz.resize(Ny);
    sTA.resize(Ny);
    sTARz.resize(Ny);
    sTARzRZetaInv.resize(Ny);
    sTARzRZetaInvRz.resize(Ny);
    LA1.resize(N, Ny);
    A2LA1.resize(Ny, Ny);
    rTLA1.resize(Ny);
    rTA1.resize(Ny);
    rTA1Rz.resize(Ny);
    rTA1RzRZetaInv.resize(Ny);
    rTA1RzRZetaInvRz.resize(Ny);
    LA1_.resize(N, Ny);
    rTA1RZetaInvDRzDtheta.resize(Ny);
    dsTdrA.resize(Ny);
    dsTdbA.resize(Ny);
    dsTdrARz.resize(Ny);
    dsTdbARz.resize(Ny);
    sTADRzDw.resize(Ny);
    dsTdrARzRZetaInv.resize(Ny);
    dsTdbARzRZetaInv.resize(Ny);
    sTADRzDwRZetaInv.resize(Ny);
    sTARzRZetaInvDRzDtheta.resize(Ny);
    dsTdrARzRZetaInvRz.resize(Ny);
    dsTdbARzRZetaInvRz.resize(Ny);
    sTADRzDwRZetaInvRz.resize(Ny);
    sTADRzDwRZetaInvRzRZeta.resize(Ny);
    rTA1DRZetaInvDAngle.resize(Ny);
    rTA1DRZetaInvDAngleRz.resize(Ny);
    rTA1DRZetaInvDAngleRzRZeta.resize(Ny);
    rTA1RZetaInvRzDRZetaDAngle.resize(Ny);
    sTARzDRZetaInvDAngle.resize(Ny);
    sTARzDRZetaInvDAngleRz.resize(Ny);
    sTARzDRZetaInvDAngleRzRZeta.resize(Ny);
    sTARzRZetaInvRzDRZetaDAngle.resize(Ny);
    dLdu.resize(Nu);
    rTdLduA1.resize(Nu, Ny);
    sTA2dLduA1.resize(Nu, Ny);
    sTA2dLduA1Rz.resize(Nu, Ny);
    X0.resize(0, Ny);
    Xp.resize(0, Ny);
}