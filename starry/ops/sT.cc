#section support_code_struct

starry_theano::solver::Greens<DTYPE_OUTPUT_0>* APPLY_SPECIFIC(G);

#section init_code_struct

{
    APPLY_SPECIFIC(G) = NULL;
}

#section cleanup_code_struct

if (APPLY_SPECIFIC(G) != NULL) {
    delete APPLY_SPECIFIC(G);
}

#section support_code_struct

int APPLY_SPECIFIC(sT)(
    PyArrayObject* input0,    // b
    PyArrayObject* input1,    // r
    PyArrayObject** output0,  // s
    PARAMS_TYPE* params
) {
    typedef DTYPE_OUTPUT_0 Scalar;

    // Map degree
    int deg = params->deg;

    // Get pointers to the input data
    npy_intp nb = -1, 
             nr = 1, 
             ns = -1;
    auto b = starry_theano::get_flat_input<DTYPE_INPUT_0>(input0, &nb);
    auto r_ = starry_theano::get_flat_input<DTYPE_INPUT_1>(input1, &nr);
    if (b == NULL || r_ == NULL) {
        PyErr_Format(PyExc_RuntimeError, "either `b` or `r` is NULL");
        return 1;
    }
    auto r = r_[0];

    // Set up the op; if it exists, reuse it
    if (APPLY_SPECIFIC(G) == NULL || APPLY_SPECIFIC(G)->deg != deg) {
        if (APPLY_SPECIFIC(G) != NULL)
            delete APPLY_SPECIFIC(G);
        APPLY_SPECIFIC(G) = new starry_theano::solver::Greens<Scalar>(deg);
    }
    int N = APPLY_SPECIFIC(G)->N;

    // Access the output data
    auto ndim = PyArray_NDIM(input0);
    auto dims = PyArray_DIMS(input0);
    std::vector<npy_intp> shape(ndim + 1);
    for (npy_intp i = 0; i < ndim; ++i) 
        shape[i] = dims[i];
    shape[ndim] = N;
    auto s = starry_theano::allocate_output<DTYPE_OUTPUT_0>(
        ndim + 1, &(shape[0]), TYPENUM_OUTPUT_0, output0
    );
    if (s == NULL) {
        PyErr_Format(PyExc_RuntimeError, "`s` is NULL");
        return 2;
    }
    
    // Do the computation
    for (npy_intp n = 0; n < nb; ++n) {
        try {
            APPLY_SPECIFIC(G)->compute(
                std::abs(b[n]), 
                r,
                &(s[n * N])
            );
        } catch (std::exception& e) {
            PyErr_Format(PyExc_RuntimeError, "starry compute failed");
            return 3;
        }
    }

    return 0;
}