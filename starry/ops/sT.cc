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
    PyArrayObject* input2,    // bs
    PyArrayObject** output0,  // s
    PyArrayObject** output1,  // bb
    PyArrayObject** output2,  // br
    PARAMS_TYPE* params
) {
    typedef DTYPE_OUTPUT_0 Scalar;

    // Map degree
    int deg = params->deg;

    // Get pointers to the input data
    npy_intp size = -1, one = 1, bs_size = -1;
    auto b = starry_theano::get_flat_input<DTYPE_INPUT_0>(input0, &size);
    auto r = starry_theano::get_flat_input<DTYPE_INPUT_1>(input1, &one);
    auto bs = starry_theano::get_flat_input<DTYPE_INPUT_2>(input2, &bs_size);
    if (b == NULL || r == NULL || bs == NULL) {
        return 1;
    }

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
        return 2;
    }
    auto bb = starry_theano::allocate_output<DTYPE_OUTPUT_1>(
        ndim, &(shape[0]), TYPENUM_OUTPUT_1, output1
    );
    if (bb == NULL) {
        return 2;
    }
    auto br = starry_theano::allocate_output<DTYPE_OUTPUT_2>(
        0, &one, TYPENUM_OUTPUT_2, output2
    );
    if (br == NULL) {
        return 2;
    }
    *br = 0.0;

    // Do the computation
    auto r0 = r[0];
    for (npy_intp n = 0; n < size; ++n) {
        bb[n] = 0.0;

        // DEBUG
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> s_vec(N);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> bs_vec(N);
        for (npy_intp j = 0; j < N; ++j) {
            bs_vec(j) = bs[n * N + j];
        }

        try {
            APPLY_SPECIFIC(G)->compute(std::abs(b[n]), r0, bs_vec, s_vec, bb[n], *br);
        } catch (std::exception& e) {
            PyErr_Format(PyExc_RuntimeError, "starry compute failed");
            return 3;
        }

        for (npy_intp j = 0; j < N; ++j) {
            s[n * N + j] = s_vec(j);
        }

    }

    return 0;
}