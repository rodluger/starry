# Automatic differentiation on sn.jl:
include("sn_bigr.jl")

using ForwardDiff
using DiffResults

function sn_jac(l_max::Int64,r::T,b::T) where {T <: Real}
  # Computes the derivative of s_n(r,b) with respect to r, b.
  # Replace s2 jacobian with analytic result
  s2_grad = zeros(typeof(r),2)
  s_2=s2!(r,b,s2_grad)
  # Create a vector for use with ForwardDiff
  x=[r,b]
  # Compute the length of the vector s_n:
  n_max = l_max^2+2*l_max
  # Allocate an array for s_n:
  sn = zeros(typeof(r),n_max+1)

  # Now, define a wrapper of s_n! for use with ForwardDiff:
  function diff_sn(x::Array{T,1}) where {T <: Real}
  # x should be a two-element vector with values [r,b]
  r0,b0 = x
  sn = zeros(typeof(r0),n_max+1)
#  s_n!(l_max,r,b,sn)
  s_n_bigr!(l_max,r0,b0,sn)
  return sn
  end

  # Set up a type to store s_n and it's Jacobian with respect to x:
  out = DiffResults.JacobianResult(sn,x) 
  # Compute the Jacobian (and value):
  out = ForwardDiff.jacobian!(out,diff_sn,x)
  # Place the value in the s_n vector:
  sn = DiffResults.value(out)
#  println("type of sn: ",typeof(sn)," type of s_2: ",typeof(s_2))
  # And, place the Jacobian in an array:
  sn_jacobian = DiffResults.jacobian(out)
#  println("type of sn_jacobian: ",typeof(sn_jacobian)," type of s2_grad: ",typeof(s2_grad))
  # Add in the s_2 values:
  sn[3]=s_2
  sn_jacobian[3,:]=s2_grad
return sn,sn_jacobian
end

# Here are some example lines of how to use it:
#r = 100.0; b=100.5
#l_max = 10
#sn,sn_jacobian= sn_jac(l_max,r,b)
