# Trying out derivatives of cel_bulirsch with respect to
# k_c, p, a, b:
include("cel_bulirsch.jl")

using ForwardDiff
using DiffRules

dcel_dn = m -> ForwardDiff.derivative(cel_bulirsch,m)

function cel_bad_grad(r::T,b::T) where {T <: Real}
x=[r,b]
function diff_cel(x::Array{T,1}) where {T <: Real}
r,b=x
onembpr2 = (1-b-r)*(1+b+r); onembmr2=(r-b+1)*(b-r+1); fourbr = 4b*r
k2 = onembpr2/fourbr+1
k2c = -onembpr2/fourbr; kc = sqrt(b+r-1)*sqrt(b+r+1)/sqrt(4*b*r)
#p=(b-r)^2*k2c; a=0.0; b_cel=3*k2c*(b-r)*(b+r)
p=(b-r)^2*k2c; a=0.0; b_cel=3*(b+r)*(b-r)*k2c
return cel_bulirsch(k2,kc,p,zero(b),b_cel)
end
out = DiffResults.GradientResult(x)
# Compute the Jacobian (and value):
out = ForwardDiff.gradient!(out,diff_cel,x)
# Place the value in the cel_b vector:
cel_b = DiffResults.value(out)
# And, place the Jacobian in an array:
cel_gradient= DiffResults.gradient(out)
return cel_b,cel_gradient
end


function cel_grad(kc::T,p::T,a::T,b::T) where {T <: Real}
  # Computes the derivative of cel(k_c,p,a,b) with respect to (k_c,p,a,b)
  # Create a vector for use with ForwardDiff
  x=[kc,p,a,b]
  # Now, define a wrapper of cel_bulirsch for use with ForwardDiff:
  function diff_cel(x::Array{T,1}) where {T <: Real}
  # x should be a four-element vector with values [kc,p,a,b]
  kc,p,a,b = x
  return cel_bulirsch(1.0-kc*kc,kc,p,a,b)
  end

  # Set up a type to store s_n and it's Jacobian with respect to x:
  out = DiffResults.GradientResult(x)
  # Compute the Jacobian (and value):
  out = ForwardDiff.gradient!(out,diff_cel,x)
  # Place the value in the cel_b vector:
  cel_b = DiffResults.value(out)
  # And, place the Jacobian in an array:
  cel_gradient= DiffResults.gradient(out)
return cel_b,cel_gradient
end

function cel_grad_anal(kc::T,p::T,a::T,b::T) where {T <: Real}
# Now, compute the gradients analytically:
kc2 = kc*kc
dcel_dkc = -kc/(p-kc2)*(cel_bulirsch(1-kc*kc,kc,kc2,a,b)-cel_b)
#dcel_dp = (2*p*b-p*a-b)/(2*p*(1-p))*cel_bulirsch(1-kc*kc,kc,p,zero(p),one(p))+
#  (b-a*p)/(2*p*(1-p)*(p-kc2))*cel_bulirsch(1-kc*kc,kc,one(p),one(p),kc2) -
#  (b-a*p)/(2*(1-p)*(p-kc2))*cel_bulirsch(1-kc*kc,kc,p,one(p),one(p))
dcel_dp = ((kc2*(b+a*p-2*b*p)+p*(3*b*p-a*p^2-2*b))*cel_bulirsch(1-kc*kc,kc,p,zero(p),one(p))+
  (b-a*p)*cel_bulirsch(1-kc*kc,kc,one(p),1-p,kc2-p))/(2*p*(1-p)*(p-kc2))
dcel_da = cel_bulirsch(1-kc*kc,kc,p,one(p),zero(p))
dcel_db = cel_bulirsch(1-kc*kc,kc,p,zero(p),one(p))
return [dcel_dkc,dcel_dp,dcel_da,dcel_db]
end

function cel_grad_num(kc::T,p::T,a::T,b::T) where {T <: Real}
# Now, compute the gradients numerically:
epsilon = 1e-24
dcel_dkc = convert(Float64,(cel_bulirsch(big(1)-(big(kc)+big(epsilon))^2,big(kc)+big(epsilon),big(p),big(a),big(b))-
  cel_bulirsch(big(1)-(big(kc)-big(epsilon))^2,big(kc)-big(epsilon),big(p),big(a),big(b)))/(2*big(epsilon)))
dcel_dp = convert(Float64,(cel_bulirsch(big(1)-big(kc)^2,big(kc),big(p)+big(epsilon),big(a),big(b))-
  cel_bulirsch(big(1)-big(kc)^2,big(kc),big(p)-big(epsilon),big(a),big(b)))/(2*big(epsilon)))
dcel_da = convert(Float64,(cel_bulirsch(big(1)-big(kc)^2,big(kc),big(p),big(a)+big(epsilon),big(b))-
  cel_bulirsch(big(1)-big(kc)^2,big(kc),big(p),big(a)-big(epsilon),big(b)))/(2*big(epsilon)))
dcel_db = convert(Float64,(cel_bulirsch(big(1)-big(kc)^2,big(kc),big(p),big(a),big(b)+big(epsilon))-
  cel_bulirsch(big(1)-big(kc)^2,big(kc),big(p),big(a),big(b)-big(epsilon)))/(2*big(epsilon)))
return [dcel_dkc,dcel_dp,dcel_da,dcel_db]
end

r = 100.0; b = 100.0-1e-12
onembpr2 = (1-b-r)*(1+b+r); onembmr2=(r-b+1)*(b-r+1); fourbr = 4b*r
k2 = onembpr2/fourbr+1
k2c = -onembpr2/fourbr; kc = sqrt(k2c)
p=(b-r)^2*k2c; a=0.0; b_cel=3*k2c*(b-r)*(b+r)

#kc = rand(); p = rand(); a=rand(); b= rand()
cel_b,cel_gradient= cel_grad(kc,p,a,b_cel)
dcel_anal = cel_grad_anal(kc,p,a,b_cel)
dcel_num  = cel_grad_num(kc,p,a,b_cel)
println("d cel/d kc: ",dcel_anal[1]," ",cel_gradient[1]," ",dcel_num[1]," diff 1-2: ",dcel_anal[1]-cel_gradient[1]," ",dcel_num[1]-cel_gradient[1])
println("d cel/d p: ",dcel_anal[2]," ",cel_gradient[2]," ",dcel_num[2]," diff 1-2: ",dcel_anal[2]-cel_gradient[2]," ",dcel_num[2]-cel_gradient[2])
println("d cel/d a: ",dcel_anal[3]," ",cel_gradient[3]," ",dcel_num[3]," diff 1-2: ",dcel_anal[3]-cel_gradient[3]," ",dcel_num[3]-cel_gradient[3])
println("d cel/d b: ",dcel_anal[4]," ",cel_gradient[4]," ",dcel_num[4]," diff 1-2: ",dcel_anal[4]-cel_gradient[4]," ",dcel_num[4]-cel_gradient[4])

function cel_bad_anal(r::T,b::T) where {T <: Real}
onembpr2 = (1-b-r)*(1+b+r); onembmr2=(r-b+1)*(b-r+1); fourbr = 4b*r
k2 = onembpr2/fourbr+1
k2c = -onembpr2/fourbr; kc = sqrt(k2c)
p=(b-r)^2*k2c; a=zero(r); b_cel=3*k2c*(b-r)*(b+r)
dcel_anal = cel_grad_anal(kc,p,a,b_cel)
dceldr = (dcel_anal[1]*(1-(b-r)*(b+r))/(2*kc)+dcel_anal[2]*(1-3r^2-b^2)*(b-r)*(b+r)-dcel_anal[4]*3*(3*r^4+b^4-b^2-r^2+4*b*r^3))/(4*b*r^2)
dceldb = (dcel_anal[1]*(1+(b-r)*(b+r))/(2*kc)+dcel_anal[2]*(r^2+3b^2-1)*(b-r)*(b+r)+dcel_anal[4]*3*(3*b^4+r^4-b^2-r^2+4*b^3*r))/(4*b^2*r)
return [dceldr,dceldb]
end

dcel_bad_anal = cel_bad_anal(big(r),big(b))
println("dcel/dr: ",dcel_bad_anal[1]," dcel/db: ",dcel_bad_anal[2])
dcel_bad_anal = cel_bad_anal(r,b)
println("dcel/dr: ",dcel_bad_anal[1]," dcel/db: ",dcel_bad_anal[2])
cel_bad,cel_bad_gradient=cel_bad_grad(big(r),big(b))
println("dcel/dr: ",cel_bad_gradient[1]," dcel/db: ",cel_bad_gradient[2])
cel_bad,cel_bad_gradient=cel_bad_grad(r,b)
println("dcel/dr: ",cel_bad_gradient[1]," dcel/db: ",cel_bad_gradient[2])
