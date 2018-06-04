# Computes the function cel(kc,p,a,b) from Bulirsch (1969).
function cel_bulirsch(k2::T,p::T,a::T,b::T) where {T <: Real}
@assert (k2 <= 1.0)
ca = sqrt(eps(k2))
# Avoid undefined k2=1 case:
if k2 != 1.0
  kc = sqrt(1.0-k2)
else
  kc = eps(k2)
end
# Initialize values:
ee = kc; m=1.0
if p > 0.0
  p = sqrt(p); b /= p
else
  q=k2; g=1.0-p; f = g-k2
  q *= (b-a*p); p=sqrt(f/g); a=(a-b)/g
  b = -q/(g*g*p)+a*p
end
# Compute recursion:
f=a; a += b/p; g=ee/p; b += f*g; b +=b; p +=g; g=m; m += kc
while abs(g-kc) > g*ca
  kc=sqrt(ee); kc += kc; ee = kc*m
  f=a; a += b/p; g=ee/p; b += f*g; b +=b; p +=g; g=m; m += kc
end
return pi/2*(a*m+b)/(m*(m+p))
end

# Version called with kc (this is to improve precision of computation):
function cel_bulirsch(k2::T,kc::T,p::T,a::T,b::T) where {T <: Real}
@assert (k2 <= 1.0)
ca = sqrt(eps(k2))
# Avoid undefined k2=1 case:
if k2 == 1.0
  kc = eps(k2)
end
# Initialize values:
ee = kc; m=1.0
if p > 0.0
  p = sqrt(p); b /= p
else
  q=k2; g=1.0-p; f = g-k2
  q *= (b-a*p); p=sqrt(f/g); a=(a-b)/g
  b = -q/(g*g*p)+a*p
end
# Compute recursion:
f=a; a += b/p; g=ee/p; b += f*g; b +=b; p +=g; g=m; m += kc
while abs(g-kc) > g*ca
  kc=sqrt(ee); kc += kc; ee = kc*m
  f=a; a += b/p; g=ee/p; b += f*g; b +=b; p +=g; g=m; m += kc
end
return pi/2*(a*m+b)/(m*(m+p))
end
