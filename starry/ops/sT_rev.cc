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

int APPLY_SPECIFIC(sT_rev)(
    PyArrayObject* input0,    // b
    PyArrayObject* input1,    // r
    PyArrayObject* input2,    // bs
    PyArrayObject** output0,  // bb
    PyArrayObject** output1,  // br
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
    auto bs = starry_theano::get_flat_input<DTYPE_INPUT_2>(input2, &ns);
    if (b == NULL || r_ == NULL || bs == NULL) {
        PyErr_Format(PyExc_RuntimeError, "either `b`, `r`, or `bs` is `NULL`");
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
    auto bb = starry_theano::allocate_output<DTYPE_OUTPUT_0>(
        ndim, &(shape[0]), TYPENUM_OUTPUT_0, output0
    );
    if (bb == NULL) {
        PyErr_Format(PyExc_RuntimeError, "`bb` is NULL");
        return 2;
    }
    auto br = starry_theano::allocate_output<DTYPE_OUTPUT_1>(
        0, &nr, TYPENUM_OUTPUT_1, output1
    );
    if (br == NULL) {
        PyErr_Format(PyExc_RuntimeError, "`br` is NULL");
        return 2;
    }
    
    // Do the computation
    *br = 0.0;
    for (npy_intp n = 0; n < nb; ++n) {
        bb[n] = 0.0;
        try {
            APPLY_SPECIFIC(G)->compute(
                std::abs(b[n]), 
                r, 
                &(bs[n * N]),
                bb[n], 
                *br
            );
        } catch (std::exception& e) {
            PyErr_Format(PyExc_RuntimeError, "starry compute failed");
            return 3;
        }
    }

    return 0;
}