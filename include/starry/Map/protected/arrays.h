/**
Temporary vectors and matrices used throughout the code.

*/
RowMatrix<Scalar> X0;
RowMatrix<Scalar> Xp;
RowVector<Scalar> x_cache;
RowVector<Scalar> y_cache;
RowVector<Scalar> rTLA1RZetaInv;
RowVector<Scalar> rTLA1RZetaInvRz;
RowVector<Scalar> sTA;
RowVector<Scalar> sTARz;
RowVector<Scalar> sTARzL;
RowVector<Scalar> sTARzLRZetaInv;
RowVector<Scalar> sTARzLRZetaInvRz;
Eigen::SparseMatrix<Scalar> LA1;
RowVector<Scalar> rTLA1;
RowVector<Scalar> rTA1;
RowVector<Scalar> rTA1Rz;
RowVector<Scalar> rTA1RzL;
RowVector<Scalar> rTA1RzLRZetaInv;
RowVector<Scalar> rTA1RzLRZetaInvRz;
Vector<Matrix<Scalar>> DLDu;
Matrix<Scalar> rTDLDuA1;
RowVector<Scalar> rTLA1RZetaInvDRzDtheta;
RowVector<Scalar> rTLA1DRZetaInvDAngle;
RowVector<Scalar> rTLA1DRZetaInvDAngleRz;
RowVector<Scalar> rTLA1DRZetaInvDAngleRzRZeta;
RowVector<Scalar> rTLA1RZetaInvRzDRZetaDAngle;
RowVector<Scalar> DsTDrA;
RowVector<Scalar> DsTDbA;
RowVector<Scalar> DsTDrARz;
RowVector<Scalar> DsTDrARzL;
RowVector<Scalar> DsTDbARz;
RowVector<Scalar> DsTDbARzL;
RowVector<Scalar> sTADRzDw;
RowVector<Scalar> sTADRzDwL;
RowVector<Scalar> DsTDrARzLRZetaInv;
RowVector<Scalar> DsTDbARzLRZetaInv;
RowVector<Scalar> sTADRzDwLRZetaInv;
RowVector<Scalar> sTARzLRZetaInvDRzDtheta;
RowVector<Scalar> DsTDrARzLRZetaInvRz;
RowVector<Scalar> DsTDbARzLRZetaInvRz;
RowVector<Scalar> sTADRzDwLRZetaInvRz;
RowVector<Scalar> sTADRzDwLRZetaInvRzRZeta;
RowVector<Scalar> sTARzLDRZetaInvDAngle;
RowVector<Scalar> sTARzLDRZetaInvDAngleRz;
RowVector<Scalar> sTARzLDRZetaInvDAngleRzRZeta;
RowVector<Scalar> sTARzLRZetaInvRzDRZetaDAngle;
Matrix<Scalar> sTARzDLDu;

/**
Allocate the temporary vectors and matrices 
used throughout the code.

*/
void resize_arrays() {
    X0.resize(0, Ny);
    Xp.resize(0, N);
    rTLA1RZetaInv.resize(Ny);
    rTLA1RZetaInvRz.resize(Ny);
    sTA.resize(N);
    sTARz.resize(N);
    sTARzL.resize(Ny);
    sTARzLRZetaInv.resize(Ny);
    sTARzLRZetaInvRz.resize(Ny);
    LA1.resize(N, Ny);
    rTLA1.resize(Ny);
    rTA1.resize(N);
    rTA1Rz.resize(N);
    rTA1RzL.resize(Ny);
    rTA1RzLRZetaInv.resize(Ny);
    rTA1RzLRZetaInvRz.resize(Ny);
    DLDu.resize(Nu);
    rTDLDuA1.resize(Nu, Ny);
    rTLA1RZetaInvDRzDtheta.resize(Ny);
    rTLA1DRZetaInvDAngle.resize(Ny);
    rTLA1DRZetaInvDAngleRz.resize(Ny);
    rTLA1DRZetaInvDAngleRzRZeta.resize(Ny);
    rTLA1RZetaInvRzDRZetaDAngle.resize(Ny);
    DsTDrA.resize(N);
    DsTDbA.resize(N);
    DsTDrARz.resize(N);
    DsTDrARzL.resize(Ny);
    DsTDbARz.resize(N);
    DsTDbARzL.resize(Ny);
    sTADRzDw.resize(N);
    sTADRzDwL.resize(Ny);
    DsTDrARzLRZetaInv.resize(Ny);
    DsTDbARzLRZetaInv.resize(Ny);
    sTADRzDwLRZetaInv.resize(Ny);
    sTARzLRZetaInvDRzDtheta.resize(Ny);
    DsTDrARzLRZetaInvRz.resize(Ny);
    DsTDbARzLRZetaInvRz.resize(Ny);
    sTADRzDwLRZetaInvRz.resize(Ny);
    sTADRzDwLRZetaInvRzRZeta.resize(Ny);
    sTARzLDRZetaInvDAngle.resize(Ny);
    sTARzLDRZetaInvDAngleRz.resize(Ny);
    sTARzLDRZetaInvDAngleRzRZeta.resize(Ny);
    sTARzLRZetaInvRzDRZetaDAngle.resize(Ny);


    sTARzDLDu.resize(Nu, Ny);

    /*
    dsTdrARzRZetaInvRz.resize(Ny);
    dsTdbARzRZetaInvRz.resize(Ny);
    sTADRzDwRZetaInvRz.resize(Ny);
    sTADRzDwRZetaInvRzRZeta.resize(Ny);
    sTA2dLduA1.resize(Nu, Ny);
    sTA2dLduA1Rz.resize(Nu, Ny);
    */
}