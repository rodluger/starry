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
iter = 0; itmax = 50
while abs(g-kc) > g*ca && iter < itmax
  kc=sqrt(ee); kc += kc; ee = kc*m
  f=a; a += b/p; g=ee/p; b += f*g; b +=b; p +=g; g=m; m += kc; iter+=1
end
if iter == itmax
  println("k2 ",k2," kc ",kc,"abs(g-kc) ",abs(g-kc)," g*ca ",g*ca)
end
return pi/2*(a*m+b)/(m*(m+p))
end

# Version called with kc (this is to improve precision of computation):
function cel_bulirsch(k2::T,kc::T,p::T,a::T,b::T) where {T <: Real}
#println("cel ",k2," ",kc," ",p," ",a," ",b)
@assert (k2 <= 1.0)
ca = sqrt(eps(k2))
# Avoid undefined k2=1 case:
if k2 == 1.0 || kc == 0.0
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
iter = 0; itmax = 50
while abs(g-kc) > g*ca && iter < itmax
  kc=sqrt(ee); kc += kc; ee = kc*m
  f=a; a += b/p; g=ee/p; b += f*g; b +=b; p +=g; g=m; m += kc; iter +=1
end
if iter == itmax
  println("k2 ",k2," kc ",kc," abs(g-kc) ",abs(g-kc)," g*ca ",g*ca," cel ",pi/2*(a*m+b)/(m*(m+p)))
end
return pi/2*(a*m+b)/(m*(m+p))
end
