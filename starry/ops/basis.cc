#section support_code_struct

starry_theano::basis::Basis<DTYPE_OUTPUT_0>* APPLY_SPECIFIC(B);

#section init_code_struct

{
    APPLY_SPECIFIC(B) = NULL;
}

#section cleanup_code_struct

if (APPLY_SPECIFIC(B) != NULL) {
    delete APPLY_SPECIFIC(B);
}

#section support_code_struct

int APPLY_SPECIFIC(basis)(
    PyArrayObject** output0,  // rT
    PyArrayObject** output1,  // rTA1
    PARAMS_TYPE* params
) {
    typedef DTYPE_OUTPUT_0 Scalar;

    // Map degree
    int ydeg = params->ydeg;
    int udeg = params->udeg;
    int fdeg = params->fdeg;
    int N = params->N;
    int Ny = params->Ny;

    // Set up the op; if it exists, reuse it
    if (APPLY_SPECIFIC(B) == NULL || 
        APPLY_SPECIFIC(B)->ydeg != ydeg ||
        APPLY_SPECIFIC(B)->udeg != udeg ||
        APPLY_SPECIFIC(B)->fdeg != fdeg) {
        if (APPLY_SPECIFIC(B) != NULL)
            delete APPLY_SPECIFIC(B);
        APPLY_SPECIFIC(B) = new starry_theano::basis::Basis<Scalar>(ydeg, udeg, fdeg);
    }

    // Access the output data
    std::vector<npy_intp> shapeN{N};
    auto rT = starry_theano::allocate_output<DTYPE_OUTPUT_0>(
        1, &(shapeN[0]), TYPENUM_OUTPUT_0, output0
    );
    if (rT == NULL) {
        PyErr_Format(PyExc_RuntimeError, "`rT` is NULL");
        return 2;
    }

    std::vector<npy_intp> shapeNy{Ny};
    auto rTA1 = starry_theano::allocate_output<DTYPE_OUTPUT_1>(
        1, &(shapeNy[0]), TYPENUM_OUTPUT_1, output1
    );
    if (rTA1 == NULL) {
        PyErr_Format(PyExc_RuntimeError, "`rTA1` is NULL");
        return 2;
    }
    
    // Copy the matrices over
    for (npy_intp n = 0; n < N; ++n) {
        rT[n] = APPLY_SPECIFIC(B)->rT(n);
    }

    for (npy_intp n = 0; n < Ny; ++n) {
        rTA1[n] = APPLY_SPECIFIC(B)->rTA1(n);
    }

    return 0;
}