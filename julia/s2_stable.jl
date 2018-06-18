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
    onembpr2 = 1-(b+r)^2; onembmr2=1-(b-r)^2; fourbr = 4b*r
    k2 = onembpr2/fourbr+1
    if (b+r) > 1.0 # k^2 < 1, Case 2, Case 8
      k2c = -onembpr2/fourbr; kc = sqrt(k2c)
      Eofk = cel_bulirsch(k2,kc,one(b),one(b),k2c) # Complete elliptic integral of second kind
      Lambda1 = onembmr2*(k2c*(b-r)*(b+r)*cel_bulirsch(k2,kc,(b-r)^2*k2c,zero(b),3one(b))
          +cel_bulirsch(k2,kc,one(b),-3+6r^2-2*b*r,onembpr2))/(9*pi*sqrt(b*r))
#      Lambda1 = onembmr2*(cel_bulirsch(k2,kc,(b-r)^2*k2c,zero(b),3*k2c*(b-r)*(b+r))
#          -(3-6*r^2-2*b*r)*cel_bulirsch(k2,kc,one(b),one(b),zero(b))-fourbr*Eofk)/(9*pi*sqrt(b*r))
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
