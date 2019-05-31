/**
\file ops.h
\brief Class housing all map operations.

*/

#include "solver_emitted.h"
#include "solver_reflected.h"
#include "basis.h"
#include "wigner.h"
#include "filter.h"

namespace starry {

//! The Ops class
template <class Scalar>
class Ops 
{

public:
    
    const int ydeg;
    const int Ny;                                                              /**< Number of spherical harmonic `(l, m)` coefficients */
    const int udeg;
    const int Nu;                                                              /**< Number of limb darkening coefficients */
    const int fdeg;
    const int Nf;                                                              /**< Number of filter `(l, m)` coefficients */
    const int deg;
    const int N;

    basis::Basis<Scalar> B;
    wigner::Wigner<Scalar> W;
    solver::GreensEmitted<Scalar> G;                                           /**< The occultation integral solver class */
    solver::GreensReflected<Scalar> GRef;
    filter::Filter<Scalar> F;

    // Constructor
    explicit Ops (
        int ydeg,
        int udeg,
        int fdeg
    ) :
        ydeg(ydeg), 
        Ny((ydeg + 1) * (ydeg + 1)),
        udeg(udeg),
        Nu(udeg + 1),
        fdeg(fdeg),
        Nf((fdeg + 1) * (fdeg + 1)),
        deg(ydeg + udeg + fdeg),
        N((deg + 1) * (deg + 1)),
        B(ydeg, udeg, fdeg),
        W(ydeg, udeg, fdeg),
        G(deg),
        GRef(deg),
        F(B)
    {
        // Bounds checks
        if ((ydeg < 0) || (ydeg > STARRY_MAX_LMAX))
            throw std::out_of_range(
                "Spherical harmonic degree out of range."
            );
        if ((deg > STARRY_MAX_LMAX))
            throw std::out_of_range(
                "Total degree out of range."
            );
    };

}; // class Ops

} // namespace starry