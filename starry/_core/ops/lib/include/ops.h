/**
\file ops.h
\brief Class housing all map operations.

*/

#include "basis.h"
#include "filter.h"
#include "misc.h"
#include "reflected/occultation.h"
#include "reflected/phasecurve.h"
#include "solver.h"
#include "utils.h"
#include "wigner.h"

namespace starry {

using namespace utils;

//! The Ops class
template <class Scalar> class Ops {
public:
  const int ydeg;
  const int Ny; /**< Number of spherical harmonic `(l, m)` coefficients */
  const int udeg;
  const int Nu; /**< Number of limb darkening coefficients */
  const int fdeg;
  const int Nf; /**< Number of filter `(l, m)` coefficients */
  const int deg;
  const int N;

  basis::Basis<Scalar> B;
  wigner::Wigner<Scalar> W;
  solver::Greens<Scalar> G; /**< The occultation integral solver class */
  reflected::phasecurve::PhaseCurve<ADScalar<Scalar, 2>> RP;
  reflected::occultation::Occultation<ADScalar<Scalar, 5>> RO;
  filter::Filter<Scalar> F;

  // Spot gradients
  RowVector<Scalar> bamp;
  Scalar bsigma;
  Scalar blat;
  Scalar blon;

  // Constructor
  explicit Ops(int ydeg, int udeg, int fdeg)
      : ydeg(ydeg), Ny((ydeg + 1) * (ydeg + 1)), udeg(udeg), Nu(udeg + 1),
        fdeg(fdeg), Nf((fdeg + 1) * (fdeg + 1)), deg(ydeg + udeg + fdeg),
        N((deg + 1) * (deg + 1)), B(ydeg, udeg, fdeg), W(ydeg, udeg, fdeg),
        G(deg), RP(deg, B), RO(deg, B), F(B) {
    // Bounds checks
    if ((ydeg < 0) || (ydeg > STARRY_MAX_LMAX))
      throw std::out_of_range("Spherical harmonic degree out of range.");
    if ((deg > STARRY_MAX_LMAX))
      throw std::out_of_range("Total degree out of range.");
  };

  // Compute the Ylm expansion of a gaussian spot at a
  // given latitude/longitude on the map.
  inline Matrix<Scalar> spotYlm(const RowVector<Scalar> &amp,
                                const Scalar &sigma, const Scalar &lat = 0,
                                const Scalar &lon = 0) {
    return misc::spotYlm(amp, sigma, lat, lon, ydeg, W);
  }

  // Compute the gradient of the Ylm expansion of a gaussian spot at a
  // given latitude/longitude on the map.
  inline void spotYlm(const RowVector<Scalar> &amp, const Scalar &sigma,
                      const Scalar &lat, const Scalar &lon,
                      const Matrix<double> &by) {
    misc::spotYlm(amp, sigma, lat, lon, by, ydeg, W, bamp, bsigma, blat, blon);
  }

}; // class Ops

} // namespace starry