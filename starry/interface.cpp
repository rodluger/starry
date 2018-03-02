#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "starry.h"

namespace py = pybind11;

struct ndarray {
  ndarray(py::array_t<double, py::array::c_style>& arr) {
    auto buf = arr.request();
    if (buf.ndim != 1) throw std::runtime_error("invalid array");
    size = buf.size;
    ptr = (double*)buf.ptr;
  }
  int size = 0;
  double* ptr = NULL;
};

PYBIND11_MODULE(interface, m) {

  m.doc() = R"delim(
    This is the module docstring
  )delim";

  auto doc = R"delim(
    This is a method docstring
  )delim";

  m.def("flux",
    [](
      py::array_t<double, py::array::c_style> y_in,
      py::array_t<double, py::array::c_style> u_in,
      py::array_t<double, py::array::c_style> theta_in,
      py::array_t<double, py::array::c_style> x0_in,
      py::array_t<double, py::array::c_style> y0_in,
      double r,
      int lmax
    ) {
      // Parse the input arrays
      ndarray y(y_in), u(u_in), theta(theta_in), x0(x0_in), y0(y0_in);
      int NT = theta.size;

      // Check the dimensions
      if (x0.size != NT || y0.size != NT || u.size != 3)
        throw std::runtime_error("dimension mismatch");

      // Set up the constants
      CONSTANTS C;
      init_constants(lmax, &C);
      if (y.size != C.N) {
        free_constants(lmax, &C);
        throw std::runtime_error("dimension mismatch");
      }

      // Allocate memory for the results
      auto result_out = py::array_t<double>(NT);
      auto result = result_out.request();

      // Run the code
      flux(NT, y.ptr, u.ptr, theta.ptr, x0.ptr, y0.ptr, r, &C, (double*)result.ptr);

      // Clean up
      free_constants(lmax, &C);

      return result;
    }, doc
  );

}
