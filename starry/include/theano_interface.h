#ifndef _STARRY_THEANO_INTERFACE_
#define _STARRY_THEANO_INTERFACE_

#include <cmath>
#include <exception>
#include <Python.h>
#include <numpy/arrayobject.h>

#include "greens.h"

namespace starry_theano {

  template <typename Scalar>
  Scalar* get_flat_input (PyArrayObject* input, npy_intp* size = NULL) {

    // Check to make sure that the input is given and C contiguous
    if (input == NULL || !PyArray_CHKFLAGS(input, NPY_ARRAY_C_CONTIGUOUS)) {
      PyErr_Format(PyExc_ValueError, "input must be C contiguous");
      return NULL;
    }

    // Get the dimensions of the array
    auto actual_size = PyArray_SIZE(input);

    // Check the dimensions if the expectation is provided
    if (size != NULL && *size >= 0) {
      if (actual_size != *size) {
        PyErr_Format(PyExc_ValueError, "size mismatch; expected %d got %d", *size, actual_size);
        return NULL;
      }
    }

    // Save the dimensions
    *size = actual_size;

    return (Scalar*)PyArray_DATA(input);
  }

  template <typename Scalar>
  Scalar* allocate_output(int ndim, npy_intp* shape, int typenum, PyArrayObject** output) {

    // See if the output exists and has the right dimensions
    bool flag = true;
    if (*output != NULL && PyArray_NDIM(*output) == ndim) {
      for (int n = 0; n < ndim; ++n) {
        if (PyArray_DIMS(*output)[n] != shape[n]) {
          flag = false;
          break;
        }
      }
    } else {
      flag = false;
    }

    // If not, allocate the memory
    if (!flag || !PyArray_CHKFLAGS(*output, NPY_ARRAY_C_CONTIGUOUS)) {
      Py_XDECREF(*output);
      *output = (PyArrayObject*)PyArray_EMPTY(ndim, shape, typenum, 0);
      if (!*output) {
        PyErr_Format(PyExc_ValueError, "Could not allocate output storage");
        return NULL;
      }
    }

    return (Scalar*)PyArray_DATA(*output);
  }


}

#endif  // _STARRY_THEANO_INTERFACE_