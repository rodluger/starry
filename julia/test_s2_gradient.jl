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
#r = 100.0; b=100.5

function s2_grad_num(r::T,b::T) where {T <: Real}
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
return convert(Array{Float64,1},s2_grad_big)
end

# In the following lines I test the derivatives for the few special
# cases needed for computing s_2.  Note that in the paper I report
# \Lambda, while this gets multiplied by -\pi to produce s_2.

# Try r=b=1/2 special case:
r = 0.5; b= 0.5
s_2,s2_gradient= s2_grad(r,b)
s2_grad_numeric = s2_grad_num(r,b)
diff = s2_grad_numeric-s2_gradient
println("b : ",b," r: ",r," diff: ",diff)
println("gradient: ",s2_gradient," expected: ",-[2,-2./3.])

# Try b=0 special case:
r = 0.3; b= 0.0
s_2,s2_gradient= s2_grad(r,b)
diff = s2_grad_num(r,b)-s2_gradient
println("b : ",b," r: ",r," diff: ",diff)
println("gradient: ",s2_gradient," expected: ",-pi*[2*r*sqrt(1.-r^2),0.])

# Try r+b=1 special case:
r = 0.2; b= 0.8
s_2,s2_gradient= s2_grad(r,b)
s2_grad_numeric = s2_grad_num(r,b)
diff = s2_grad_numeric-s2_gradient
println("b : ",b," r: ",r," diff: ",diff)
println("gradient: ",s2_gradient," expected: ",-8r*sqrt(r*(1-r))*[1,-1/3])

# Try r=b <1/2 special case:
r = 0.3; b= 0.3
s_2,s2_gradient= s2_grad(r,b)
s2_grad_numeric = s2_grad_num(r,b)
diff = s2_grad_numeric-s2_gradient
println("b : ",b," r: ",r," diff: ",diff)
println("gradient: ",s2_gradient," expected: ",-4*r*[cel_bulirsch(4*r^2,1.,1.,1-4*r^2),cel_bulirsch(4*r^2,1.,-1.,1.-4*r^2)/3.])

# Try r=b > 1/2 special case:
r = 3.0; b= 3.0
s_2,s2_gradient= s2_grad(r,b)
s2_grad_numeric = s2_grad_num(r,b)
diff = s2_grad_numeric-s2_gradient
println("b : ",b," r: ",r," diff: ",diff)
println("gradient: ",s2_gradient," expected: ",-2*[cel_bulirsch(.25/r^2,1.,1.,0.),-cel_bulirsch(.25/r^2,1.,1.,2*(1.-.25/r^2))/3.])

# Now, try a random case with b+r < 1:
b=r=2.0
while b+r > 1
  r = 2rand(); b= 2rand()
end
s_2,s2_gradient= s2_grad(r,b)
s2_grad_numeric = s2_grad_num(r,b)
diff = s2_grad_numeric-s2_gradient
println("Test b+r < 1:")
println("b : ",b," r: ",r," diff: ",diff)

# Now, try a random case with b+r > 1:
b=r=0.
while b+r < 1
  r = 2rand(); b= 2rand()
end
s_2,s2_gradient= s2_grad(r,b)
s2_grad_numeric = s2_grad_num(r,b)
diff = s2_grad_numeric-s2_gradient
println("Test b+r > 1:")
println("b : ",b," r: ",r," diff: ",diff)
