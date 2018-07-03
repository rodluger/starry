# Automatic differentiation on sn.jl:
include("transit_poly.jl")

using ForwardDiff
using DiffResults

function transit_poly_grad(r::T,b::T,u_n::Array{T,1}) where {T <: Real}
  # Computes the derivative of transit_poly(r,b,u_n) with respect to r, b, u_n.
  # Create a vector for use with ForwardDiff:
  x=[r;b;u_n]
  # Now, define a wrapper of transit_poly for use with ForwardDiff:
  function diff_transit_poly(x::Array{T,1}) where {T <: Real}
  # x should be a n+2-element vector with values [r; b; u_n]
  return transit_poly(x[1],x[2],x[3:length(x)])
  end
  dfdp = zeros(typeof(r),length(u_n)+2)
  # Set up a type to store s_n and it's Jacobian with respect to x:
  out = DiffResults.GradientResult(x) 
  # Compute the Jacobian (and value):
  out = ForwardDiff.gradient!(out,diff_transit_poly,x)
  # Place the value in the s_n vector:
  flux = DiffResults.value(out)
  # And, place the Jacobian in an array:
  dfdp = DiffResults.gradient(out)
return flux,dfdp
end

function transit_poly_grad_c(r::T,b::T,c_n::Array{T,1}) where {T <: Real}
  # Computes the derivative of transit_poly_c(r,b,c_n) with respect to r, b, c_n.
  # Create a vector for use with ForwardDiff:
  x=[r;b;c_n]
  # Now, define a wrapper of transit_poly for use with ForwardDiff:
  function diff_transit_poly_c(x::Array{T,1}) where {T <: Real}
  # x should be a n+2-element vector with values [r; b; c_n]
  return transit_poly_c(x[1],x[2],x[3:length(x)])
  end
  dfdp = zeros(typeof(r),length(c_n)+2)
  # Set up a type to store s_n and it's Jacobian with respect to x:
  out = DiffResults.GradientResult(x) 
  # Compute the Jacobian (and value):
  out = ForwardDiff.gradient!(out,diff_transit_poly_c,x)
  # Place the value in the s_n vector:
  flux = DiffResults.value(out)
  # And, place the Jacobian in an array:
  dfdp = DiffResults.gradient(out)
return flux,dfdp
end

# Here are some example lines of how to use it:
#r = 100.0; b=100.5
