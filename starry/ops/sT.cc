#section support_code_struct

starry::Ops<DTYPE_OUTPUT_0>* APPLY_SPECIFIC(ops);

#section init_code_struct

{
  APPLY_SPECIFIC(ops) = NULL;
}

#section cleanup_code_struct

if (APPLY_SPECIFIC(ops) != NULL) {
  delete APPLY_SPECIFIC(ops);
}

#section support_code_struct

int APPLY_SPECIFIC(sT)(
    PyArrayObject* input0,    // b
    PyArrayObject* input1,    // r
    PyArrayObject** output0,  // sT
    PARAMS_TYPE* params
  )
{
  typedef DTYPE_OUTPUT_0 Scalar;

  int ydeg = params->ydeg;
  int udeg = params->udeg;
  int fdeg = params->fdeg;

  // Get pointers to the input data
  npy_intp size = -1, one = 1;
  auto b = starry_theano::get_flat_input<DTYPE_INPUT_0>(input0, &size);
  auto r = starry_theano::get_flat_input<DTYPE_INPUT_1>(input1, &one);
  if (b == NULL || r == NULL) {
    return 1;
  }

  // Set up the op; if it exists, reuse it
  if (APPLY_SPECIFIC(ops) == NULL || APPLY_SPECIFIC(ops)->ydeg != ydeg || APPLY_SPECIFIC(ops)->udeg != udeg || APPLY_SPECIFIC(ops)->fdeg != fdeg) {
    if (APPLY_SPECIFIC(ops) != NULL) delete APPLY_SPECIFIC(ops);
    APPLY_SPECIFIC(ops) = new starry::Ops<Scalar>(ydeg, udeg, fdeg);
  }
  int N = APPLY_SPECIFIC(ops)->N;

  // Access the output data
  auto ndim = PyArray_NDIM(input0);
  auto dims = PyArray_DIMS(input0);
  std::vector<npy_intp> shape(ndim + 1);
  for (npy_intp i = 0; i < ndim; ++i) shape[i] = dims[i];
  shape[ndim] = N;
  auto s = starry_theano::allocate_output<DTYPE_OUTPUT_0>(ndim+1, &(shape[0]), TYPENUM_OUTPUT_0, output0);
  if (s == NULL) {
    return 2;
  }

  // Do the computation
  auto r0 = r[0];
  auto G = APPLY_SPECIFIC(ops)->G;
  for (npy_intp n = 0; n < size; ++n) {
    try {
      G.compute(std::abs(b[n]), r0);
    } catch (std::exception& e) {
      PyErr_Format(PyExc_RuntimeError, "starry compute failed");
      return 3;
    }

    auto sTval = G.sT;
    for (npy_intp j = 0; j < N; ++j) {
      s[n * N + j] = sTval(j);
    }
  }

  return 0;
}