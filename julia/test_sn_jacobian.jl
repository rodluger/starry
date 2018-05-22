# Tests automatic differentiation on sn.jl:
include("sn.jl")

using ForwardDiff
using DiffResults

function sn_jac(l_max::Int64,r::T,b::T) where {T <: Real}
  # Computes the derivative of s_n(r,b) with respect to r, b.
  # Create a vector for use with ForwardDiff
  x=[r,b]
  # Compute the length of the vector s_n:
  n_max = l_max^2+2*l_max
  # Allocate an array for s_n:
  sn = zeros(typeof(r),n_max+1)

  # Now, define a wrapper of s_n! for use with ForwardDiff:
  function diff_sn(x::Array{T,1}) where {T <: Real}
  # x should be a two-element vector with values [r,b]
  r,b = x
  sn = zeros(typeof(r),n_max+1)
  s_n!(l_max,r,b,sn)
  return sn
  end

  # Set up a type to store s_n and it's Jacobian with respect to x:
  out = DiffResults.JacobianResult(sn,x) 
  # Compute the Jacobian (and value):
  out = ForwardDiff.jacobian!(out,diff_sn,x)
  # Place the value in the s_n vector:
  sn = DiffResults.value(out)
  # And, place the Jacobian in an array:
  sn_jacobian = DiffResults.jacobian(out)
return sn,sn_jacobian
end

#r = 0.1; b= 0.95
#r = 0.1; b= 1.0-r
#r = 0.1; b= r
r = 100.0; b=100.5
l_max = 10
sn,sn_jacobian= sn_jac(l_max,r,b)

# Now, carry out finite-difference:
dq = big(1e-18)

n_max = l_max^2+2*l_max
# Allocate an array for s_n:
sn_big = zeros(BigFloat,n_max+1)
# Make BigFloat versions of r & b:
r_big = big(r); b_big = big(b)
# Compute s_n to BigFloat precision:
s_n!(l_max,r_big,b_big,sn_big)
# Now, compute finite differences:
sn_jac_big= zeros(BigFloat,n_max+1,2)
sn_plus = copy(sn_big)
s_n!(l_max,r_big+dq,b_big,sn_plus)
sn_minus = copy(sn_big)
s_n!(l_max,r_big-dq,b_big,sn_minus)
sn_jac_big[:,1] = (sn_plus-sn_minus)*.5/dq
s_n!(l_max,r_big,b_big+dq,sn_plus)
s_n!(l_max,r_big,b_big-dq,sn_minus)
sn_jac_big[:,2] = (sn_plus-sn_minus)*.5/dq

#convert(Array{Float64,2},sn_jac_big)
#sn_jacobian
convert(Array{Float64,2},sn_jac_big)-sn_jacobian
