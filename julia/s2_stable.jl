# Uses new formulation from limbdark paper.
include("cel_bulirsch.jl")

function s2(r::T,b::T) where {T <: Real}
# For now, just compute linear component:
Lambda1 = zero(typeof(b))
if b >= 1.0+r ||  r == 0.0
  # No occultation:
  Lambda1 = zero(b)  # Case 1
elseif b <= r-1.0
  # Full occultation:
  Lambda1 = zero(b)  # Case 11
else 
  if b == 0 
#    Lambda1 = -2/3*sqrt(1.0-r^2)^3 # Case 10
    Lambda1 = -2/3*sqrt(1.0-r^2)^3 # Case 10
  elseif b==r
    if r == 0.5
      Lambda1 = 1/3-4/(9pi) - 2*(b-0.5)/(3*pi) +2*(r-0.5)/pi # Case 6; I've added in analytic first derivaties.
    elseif r < 0.5
      m = 4r^2
      Lambda1 = 1/3+2/(9pi)*cel_bulirsch(m,one(r),m-3,(1-m)*(2m-3)) + # Case 5
        (b-r)*4*r/(3pi)*cel_bulirsch(m,one(r),-one(r),1-m)  # Adding in first derivative
    else
      m = 4r^2; minv = inv(m)
      Lambda1 = 1/3+1/(9pi*r)*cel_bulirsch(minv,one(r),m-3,1-m) - # Case 7
        (b-r)*2/(3pi)*cel_bulirsch(minv,one(r),one(r),2*(1-minv)) # Adding in first derivative
    end
  else
#    onembpr2 = 1-(b+r)^2; onembmr2=1-(b-r)^2; fourbr = 4b*r
    onembpr2 = (1-b-r)*(1+b+r); onembmr2=(r-b+1)*(b-r+1); fourbr = 4b*r
    k2 = onembpr2/fourbr+1
#    k2 = onembmr2/fourbr
    if (b+r) > 1.0 # k^2 < 1, Case 2, Case 8
      k2c = -onembpr2/fourbr; kc = sqrt(k2c); sqbr=sqrt(b*r)
      Lambda1 = onembmr2*(cel_bulirsch(k2,kc,(b-r)^2*k2c,zero(b),3*k2c*(b-r)*(b+r))+
          cel_bulirsch(k2,kc,one(b),-3+6r^2-2*b*r,onembpr2))/(9*pi*sqrt(b*r))
    elseif (b+r) < 1.0  # k^2 > 1, Case 3, Case 9
      k2inv = inv(k2); k2c =onembpr2/onembmr2; kc = sqrt(k2c)
      Eofk = cel_bulirsch(k2inv,kc,one(b),one(b),k2c) # Complete elliptic integral of second kind
      bmrdbpr = (b-r)/(b+r); 
      mu = 3bmrdbpr/onembmr2
      p = bmrdbpr^2*onembpr2/onembmr2
      Lambda1 = 2*sqrt(onembmr2)*(onembpr2*cel_bulirsch(k2inv,kc,p,1.0+mu,p+mu)
             -(4-7r^2-b^2)*Eofk)/(9*pi)
    else
      # b+r = 1 or k^2=1, Case 4 (extending r up to 1)
      Lambda1 = 2/(3pi)*acos(1.-2.*r)-4/(9pi)*(3+2r-8r^2)*sqrt(r*(1-r))-2/3*convert(typeof(b),r>.5) -
          8/(3pi)*(r+b-1)*r*sqrt(r*(1-r)) # Adding in first derivatives
    end
  end
end
flux = 1.0-1.5*Lambda1-convert(typeof(b),r>b)
return flux*2pi/3
end

function s2!(r::T,b::T,s2_grad::Array{T,1}) where {T <: Real}
# Computes the linear limb-darkening case, as well as the gradient,
# s2_grad=[ds_2/dr,ds_2/db] is a pre-allocated two-element array.
# For now, just compute linear component:
Lambda1 = zero(typeof(b))
fill!(s2_grad,zero(r))
if b >= 1.0+r ||  r == 0.0
  # No occultation:
  Lambda1 = zero(b)  # Case 1
elseif b <= r-1.0
  # Full occultation:
  Lambda1 = zero(b)  # Case 11
else 
  if b == 0 
#    Lambda1 = -2/3*sqrt(1.0-r^2)^3 # Case 10
    sqrt1mr2 = sqrt(1.0-r^2)
    Lambda1 = -2/3*sqrt1mr2^3 # Case 10
    s2_grad[1] = -2pi*r*sqrt1mr2 # dLambda/dr (dLambda/db= 0)
  elseif b==r
    if r == 0.5
      Lambda1 = 1/3-4/(9pi) # Case 6; added in analytic first derivaties.
      s2_grad[1] = -2      # dLambda/dr
      s2_grad[2] =  2/3  # dLambda/db
    elseif r < 0.5
      m = 4r^2
      Lambda1 = 1/3+2/(9pi)*cel_bulirsch(m,2r,one(r),m-3,(1-m)*(2m-3))  # Case 5
      s2_grad[1] = -4*pi*r*cel_bulirsch(m,2r,one(r),one(r),1-m)      # Adding in first derivative dLambda/dr
      s2_grad[2] = -4*r/3*cel_bulirsch(m,one(r),-one(r),1-m) # Adding in first derivative dLambda/db
    else
      m = 4r^2; minv = inv(m); kc = sqrt(1.-minv)
      Lambda1 = 1/3+1/(9pi*r)*cel_bulirsch(minv,kc,one(r),m-3,1-m)  # Case 7
      s2_grad[1] = -2*cel_bulirsch(minv,kc,one(r),one(r),zero(r)) # dLambda/dr
      s2_grad[2] =  2/3*cel_bulirsch(minv,kc,one(r),one(r),2*(1-minv)) # dLambda/db
    end
  else
#    onembpr2 = 1-(b+r)^2; onembmr2=1-(b-r)^2; fourbr = 4b*r
    onembpr2 = (1-b-r)*(1+b+r); onembmr2=(r-b+1)*(b-r+1); fourbr = 4b*r
    k2 = onembpr2/fourbr+1
#    k2 = onembmr2/fourbr
    if (b+r) > 1.0 # k^2 < 1, Case 2, Case 8
      k2c = -onembpr2/fourbr; kc = sqrt(k2c); sqbr=sqrt(b*r)
      Lambda1 = onembmr2*(cel_bulirsch(k2,kc,(b-r)^2*k2c,zero(b),3*k2c*(b-r)*(b+r))+
          cel_bulirsch(k2,kc,one(b),-3+6r^2-2*b*r,onembpr2))/(9*pi*sqrt(b*r))
      s2_grad[1] = -cel_bulirsch(k2,kc,one(r),2r*(1-(b-r)^2),zero(r))/(sqrt(b*r))
      s2_grad[2] = -(1-(b-r)^2)*cel_bulirsch(k2,kc,one(r),-2r,(1-(b+r)^2)/b)/(3*sqrt(b*r))
    elseif (b+r) < 1.0  # k^2 > 1, Case 3, Case 9
      k2inv = inv(k2); k2c =onembpr2/onembmr2; kc = sqrt(k2c)
      Eofk = cel_bulirsch(k2inv,kc,one(b),one(b),k2c) # Complete elliptic integral of second kind
      bmrdbpr = (b-r)/(b+r); 
      mu = 3bmrdbpr/onembmr2
      p = bmrdbpr^2*onembpr2/onembmr2
      Lambda1 = 2*sqrt(onembmr2)*(onembpr2*cel_bulirsch(k2inv,kc,p,1.0+mu,p+mu)
             -(4-7r^2-b^2)*Eofk)/(9*pi)
      s2_grad[1] = -4*r*sqrt(1-(b-r)^2)*cel_bulirsch(k2inv,kc,one(r),one(r),k2c)
      s2_grad[2] = -4*r/3*sqrt(1-(b-r)^2)*cel_bulirsch(k2inv,kc,one(r),-one(r),k2c)
    else
      # b+r = 1 or k^2=1, Case 4 (extending r up to 1)
      Lambda1 = 2/(3pi)*acos(1.-2.*r)-4/(9pi)*(3+2r-8r^2)*sqrt(r*(1-r))-2/3*convert(typeof(b),r>.5) 
      s2_grad[1] = 8*r/pi*sqrt(r*(1-r))
      s2_grad[2] = -s2_grad[1]/3
    end
  end
end
flux = 1.0-1.5*Lambda1-convert(typeof(b),r>b)
return flux*2pi/3
end
