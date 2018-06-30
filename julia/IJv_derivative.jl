# Computes I_v(k) and I_v(k) vectors from Luger et al. (2018), along 
# the derivatives with respect to k using recursion.

#using GSL

function Iv_series(k2::T,v::Int64) where {T <: Real}
# Use series expansion to compute I_v:
nmax = 100
n = 1; error = Inf; if k2 < 1; tol = eps(k2); else; tol = eps(inv(k2)); end
# Computing leading coefficient (n=0):
coeff = 2/(2v+1)
# Add leading term to I_v:
Iv = one(k2)*coeff
# Now, compute higher order terms until desired precision is reached:
while n < nmax && abs(error) > tol
  coeff *= (2.0*n-1.0)*.5*(2n+2v-1)/(n*(2n+2v+1))*k2
  error = coeff
  Iv += coeff
  n +=1
end
return Iv*k2^v*sqrt(k2)
end

# Compute I_v with hypergeometric function (this requires GSL library,
# which can't use BigFloat or ForwardDiff types):
function Iv_hyp(k2::T,v::Int64) where {T <: Real}
a = 0.5*one(k2); b=v+0.5*one(k2); c=v+1.5*one(k2);  fac = 2/(1+2v)
return sqrt(k2)*k2^v*fac*hypergeom([a,b],c,k2)
end

# Compute J_v with hypergeometric function:
function Jv_hyp(k2::T,v::Int64) where {T <: Real}
if k2 < 1
  a = 0.5; b=v+0.5; c=v+3.0;  fac = 3pi/(4*(v+1)*(v+2))
  for i=1:v
    fac *= (1.-.5/i)
  end
  return sqrt(k2)*k2^v*fac*hypergeom([a,b],c,k2)
else # k^2 >=1
  k2inv = inv(k2)
  # Found a simpler expression than the one in paper (and perhaps more stable for large k^2):
  return  sqrt(pi)*gamma(v+.5)*(sf_hyperg_2F1_renorm(-.5,v+.5,v+1.,k2inv)-(.5+v)*k2inv*sf_hyperg_2F1_renorm(-.5,v+1.5,v+2.,k2inv))
end
end

function Jv_series(k2::T,v::Int64) where {T <: Real}
# Use series expansion to compute J_v:
nmax = 100
n = 1; error = Inf; if k2 < 1; tol = eps(k2); else; tol = eps(inv(k2)); end
# Computing leading coefficient (n=0):
#coeff = 3pi/(2^(2+v)*factorial(v+2))
if k2 < 1
  coeff = 3pi/(2^(2+v)*exp(lfact(v+2)))
# multiply by (2v-1)!!
  for i=1:v
    coeff *= 2.*i-1
  end
# Add leading term to J_v:
  Jv = one(k2)*coeff
# Now, compute higher order terms until desired precision is reached:
  while n < nmax && abs(error) > tol
    coeff *= (2.0*n-1.0)*(2.0*(n+v)-1.0)*.25/(n*(n+v+2))*k2
    error = coeff
    Jv += coeff
    n +=1
  end
  return Jv*k2^v*sqrt(k2)
else # k^2 >= 1
  coeff = pi
  # Compute (2v-1)!!/(2^v v!):
  for i=1:v
    coeff *= 1.-.5/i
  end
  Jv = one(k2)*coeff; n=1
  while n < nmax && abs(error) > tol
    coeff *= (1.-2.5/n)*(1.-.5/(n+v))/k2
    error = coeff
    Jv += coeff
    n +=1
  end
  return Jv
end
end

function IJv_raise!(v_max::Int64,k2::T,kc::T,Iv::Array{T,1},Jv::Array{T,1})  where {T <: Real}
# This function needs debugging. [ ]
# Compute I_v, J_v for 0 <= v <= v_max = l_max+2
# Define k:
k = sqrt(k2)
# Iterate upwards in v:
v = v_max
# Compute I_v via upward iteration on v:
if k2 < 1
# First, compute value for v=0:
  if k2 < 0.5
    Iv[1] = 2*asin(sqrt(k2))
#  Iv[1] = acos(1.-2k2)
  else
    Iv[1] = 2*acos(kc)
  end
# Try something else:
#  Iv[1] = asin(2*k*kc)
# Next, iterate upwards in v:
#  f0 = kc/k
  f0 = kc*k
  v = 1
# Loop over v, computing I_v and J_v from higher v:
  while v <= v_max
    Iv[v+1]=((2v-1)*Iv[v]/2-f0)/v
    f0 *= k2
    v += 1
  end
else # k^2 >= 1
  # Compute v=0
  Iv[1] = pi
  for v=1:v_max
    Iv[v+1]=Iv[v]*(1.0-0.5/v)
  end
end
# Need to compute J_v for v=0, 1:
v= 0
if k2 < 1
  # Use cel_bulirsch:
  if k2 > 0
    Jv[v+1]=2/(3k2*k)*cel_bulirsch(k2,kc,one(k2),k2*(3k2-1),k2*(1-k2))
    Jv[v+2]= 2/(15k2*k)*cel_bulirsch(k2,kc,one(k2),2k2*(3k2-2),k2*(4-7k2+3k2*k2))
  else
    Jv[v+1]= 0.0
    Jv[v+2]= 0.0
  end
else # k^2 >=1
  k2inv = inv(k2)
  Jv[v+1]=2/3*cel_bulirsch(k2inv,kc,one(k2),3-k2inv,3-5k2inv+2k2inv^2)
  Jv[v+2]=cel_bulirsch(k2inv,kc,one(k2),12-8*k2inv,2*(9-8k2inv)*(1-k2inv))/15
end
v=2
while v <= v_max
  Jv[v+1] = (2*(v+1+(v-1)*k2)*Jv[v]-k2*(2v-3)*Jv[v-1])/(2v+3)
  v += 1
end
return
end

function IJv_tridiag!(v_max::Int64,k2::T,kc::T,Iv::Array{T,1},Jv::Array{T,1})  where {T <: Real}
# Compute I_v, J_v for 0 <= v <= v_max = l_max+2
# Define k:
k = sqrt(k2)
# Iterate downwards in v:
v = v_max
# Add in k2 > 1 cases [ ]
# First, compute approximation for large v:
#Iv[v+1]=Iv_hyp(k2,v)
if k2 < 1
  Iv[v+1]=Iv_series(k2,v)
# Next, iterate downwards in v:
  f0 = k2^(v-1)*k*kc
# Loop over v, computing I_v and J_v from higher v:
  while v >= 1
    Iv[v] = 2/(2v-1)*(v*Iv[v+1]+f0)
    f0 /= k2
    v -= 1
  end
else # k^2 >= 1
  # Compute v=0 (no need to iterate downwards in this case):
  Iv[1] = pi
  for v=1:v_max
    Iv[v+1]=Iv[v]*(1-.5/v)
  end
end
# Try tri-diagonal solver
# Need to compute J_v for v=0 and v=v_max:
if k2 < 1
  # Use cel_bulirsch:
#  println("k2: ",k2)
  if k2 > 0
#    fe = 2*(2k2-1); fk = (1-k2)*(2-3k2)
#    Jv[1] = 2/(3k2*k)*cel_bulirsch(k2,kc,one(k2),fk+fe,fk+fe*(1-k2))
    Jv[1] = 2/(3k2*k)*cel_bulirsch(k2,kc,one(k2),k2*(3k2-1),k2*(1-k2))
  else
    Jv[1] = 0.0
  end
else # k^2 >=1
  k2inv = inv(k2)
#  fe = 2*(2-k2inv); fk=-1+k2inv
#  Jv[1]=2/3*cel_bulirsch(k2inv,kc,one(k2),fk+fe,fk+fe*(1-k2inv))
  Jv[1]=2/3*cel_bulirsch(k2inv,kc,one(k2),3-k2inv,3-5k2inv+2k2inv^2)
end
Jv[v_max+1]=Jv_series(k2,v_max)
# Now, implement tridiagonal algorithm:
c = zeros(typeof(k2),v_max-1)
d = zeros(typeof(k2),v_max-1)
# Iterate upwards in v (lower):
v = 2
fac = 2*((v+1)+(v-1)*k2)
c[1] = -(2v+3)/fac
d[1] =  (2v-3)*k2/fac*Jv[1]
for v=3:v_max-1
#  f2 = k2*(2v-3); f1 = 2*(v+1+(v-1)*k2)/f2; f3 = (2v+3)/f2
  fac = 2*((v+1)+(v-1)*k2); den = fac + (2v-3)*k2*c[v-2]
  c[v-1] = -(2v+3)/den
  d[v-1] =  (2v-3)*k2*d[v-2]/den
end
v = v_max
fac = 2*((v+1)+(v-1)*k2); den = fac + (2v-3)*k2*c[v-2]
d[v_max-1]=((2v+3)*Jv[v_max+1]+(2v-3)*k2*d[v-2])/den
# Now, back-substitution:
Jv[v_max]=d[v_max-1]
for v=v_max-1:-1:2
  Jv[v]=d[v-1]-c[v-1]*Jv[v+1]
end
return
end

function IJv_lower!(v_max::Int64,k2::T,kc::T,Iv::Array{T,1},Jv::Array{T,1})  where {T <: Real}
# Compute I_v, J_v for 0 <= v <= v_max = l_max+2
# Define k:
k = sqrt(k2)
# Iterate downwards in v:
v = v_max
# Add in k2 > 1 cases [ ]
# First, compute approximation for large v:
#Iv[v+1]=Iv_hyp(k2,v)
if k2 < 1
  Iv[v+1]=Iv_series(k2,v)
# Next, iterate downwards in v:
  f0 = k2^(v-1)*k*kc
# Loop over v, computing I_v and J_v from higher v:
  while v >= 1
    Iv[v] = 2/(2v-1)*(v*Iv[v+1]+f0)
    f0 /= k2
    v -= 1
  end
else # k^2 >= 1
  # Compute v=0 (no need to iterate downwards in this case):
  Iv[1] = pi
  for v=1:v_max
    Iv[v+1]=Iv[v]*(1-.5/v)
  end
end
v= v_max
# Need to compute top two for J_v:
#if typeof(k2) == BigFloat
  Jv[v]=Jv_series(k2,v-1); Jv[v+1]=Jv_series(k2,v)
#else
#  Jv[v]=Jv_hyp(k2,v-1); Jv[v+1]=Jv_hyp(k2,v)
#end
#if typeof(k2) == Float64
#  println("v ",v," k2 ",k2," Jv_ser ",convert(Float64,Jv_series(big(k2),v-1))," ",convert(Float64,Jv_series(big(k2),v))," Jv_hyp ",Jv_hyp(k2,v-1)," ",Jv_hyp(k2,v))
#end
# Iterate downwards in v (lower):
while v >= 2
  f2 = k2*(2v-3); f1 = 2*(v+1+(v-1)*k2)/f2; f3 = (2v+3)/f2
  Jv[v-1] = f1*Jv[v]-f3*Jv[v+1]
  v -= 1
end
# Compute first two exactly:
v= 0
if k2 < 1
#  # Use cel_bulirsch:
#  if k2 > 0
#    Jv[v+1]=2/(3*k2*k)*cel_bulirsch(k2,kc,one(k2),k2*(3k2-1),k2*(1-k2))
#    Jv[v+2]=2/(15*k2*k)*cel_bulirsch(k2,kc,one(k2),2*k2*(3k2-2),k2*(4-7k2+3k2*k2))
##    Jv[v+3] =-2/(35*k)*cel_bulirsch(k2,kc,one(k2),8-11k2+k2*k2,(-8+5k2+2k2*k2)*(1-k2))
#  else
#    Jv[v+1]= 0.0
#    Jv[v+2]= 0.0
#  end
else # k^2 >=1
  k2inv = inv(k2)
  Jv[v+1]=2/3*cel_bulirsch(k2inv,kc,one(k2),3-k2inv,3-5k2inv+2k2inv^2)
  Jv[v+2]=cel_bulirsch(k2inv,kc,one(k2),12-8*k2inv,2*(9-8k2inv)*(1-k2inv))/15
#  Jv[v+3]=2/(35*k2)*cel_bulirsch(k2inv,kc,one(k2),-k2^2+11k2-8,(k2^2+16*k2-16)*(1-k2inv))
end
return
end

Jv_check = false # Checks values of J_v

#println("v_max: ",v_max)
Iv = zeros(typeof(k2),v_max+1); Jv = zeros(typeof(k2),v_max+1)
# This computes I_v for the largest v, and then works down to smaller values:
if k2 > 0
  if k2 < 0.5 || k2 > 2.0
    IJv_lower!(v_max,k2,kc,Iv,Jv)
  else
    IJv_raise!(v_max,k2,kc,Iv,Jv)
  end
#  IJv_tridiag!(v_max,k2,kc,Iv,Jv)
end
if Jv_check && typeof(k2) == Float64
# We can't compute Jv_hyp for values of k2 close to one and large values of v.
  for v=0:8
     Jv_tmp = Jv_hyp(k2,v)
     if abs(Jv[v+1]-Jv_tmp) > 1e-5
       println("v: ",v," k2: ",k2," Jv: ",Jv[v+1]," Jv_hyp: ",Jv_tmp)
     end
  end
end
