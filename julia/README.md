4/26/2018

Julia implementation of the STARRY equations, with
automatic differentiation.

So far the s_n(r,b) vector has been computed and
auto-diffed to give the jacobian of s_n with respect
to r and b up to l_max.  To run this from the Julia prompt,

julia> include("test_sn_jacobian.jl")
