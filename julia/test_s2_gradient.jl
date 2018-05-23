# Tests automatic differentiation on s2.jl:
include("s2_stable.jl")

using ForwardDiff
using DiffResults

function s2_grad(r::T,b::T) where {T <: Real}
  # Computes the derivative of s_n(r,b) with respect to r, b.
  # Create a vector for use with ForwardDiff
  x=[r,b]
  # Now, define a wrapper of s2 for use with ForwardDiff:
  function diff_s2(x::Array{T,1}) where {T <: Real}
  # x should be a two-element vector with values [r,b]
  r,b = x
  return s2(r,b)
  end

  # Set up a type to store s_n and it's Jacobian with respect to x:
  out = DiffResults.GradientResult(x) 
  # Compute the Jacobian (and value):
  out = ForwardDiff.gradient!(out,diff_s2,x)
  # Place the value in the s_2 vector:
  s_2 = DiffResults.value(out)
  # And, place the Jacobian in an array:
  s2_gradient= DiffResults.gradient(out)
return s_2,s2_gradient
end

#r = 0.1; b= 0.95
#r = 0.1; b= 1.0-r
#r = 0.1; b= r
r = 0.5; b= 0.5
#r = 100.0; b=100.5
s_2,s2_gradient= s2_grad(r,b)

# Now, carry out finite-difference:
dq = big(1e-18)

# Allocate an array for s_2:
# Make BigFloat versions of r & b:
r_big = big(r); b_big = big(b)
# Compute s_n to BigFloat precision:
s2_big = s2(r_big,b_big)
# Now, compute finite differences:
s2_grad_big= zeros(BigFloat,2)
s2_plus = s2(r_big+dq,b_big)
s2_minus = s2(r_big-dq,b_big)
s2_grad_big[1] = (s2_plus-s2_minus)*.5/dq
s2_plus=s2(r_big,b_big+dq)
s2_minus=s2(r_big,b_big-dq)
s2_grad_big[2] = (s2_plus-s2_minus)*.5/dq

convert(Array{Float64,1},s2_grad_big)-s2_gradient
