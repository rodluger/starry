# Computes I_v(k) and I_v(k) vectors from Luger et al. (2018), along 
# the derivatives with respect to k using recursion.
include("cel_bulirsch.jl")

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

function dJv_seriesdk(k2::T,v::Int64) where {T <: Real}
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
  dJvdk = one(k2)*coeff*(2v+1)
# Now, compute higher order terms until desired precision is reached:
  while n < nmax && abs(error) > tol
    coeff *= (2.0*n-1.0)*(2.0*(n+v)-1.0)*.25/(n*(n+v+2))*k2
    error = coeff
    Jv += coeff
    dJvdk += coeff*(2*(n+v)+1)
    n +=1
  end
  dJvdk *= k2^v
  return Jv*k2^v*sqrt(k2),dJvdk
else # k^2 >= 1
  coeff = pi
  # Compute (2v-1)!!/(2^v v!):
  for i=1:v
    coeff *= 1.-.5/i
  end
  Jv = one(k2)*coeff; n=1
  dJvdk = zero(k2)
  while n < nmax && abs(error) > tol
    coeff *= (1.-2.5/n)*(1.-.5/(n+v))/k2
    error = coeff
    Jv += coeff
    dJvdk -= 2*n*coeff
    n +=1
  end
  dJvdk /= sqrt(k2)
  return Jv,dJvdk
end
end

function dIJv_raise_dk!(v_max::Int64,k2::T,kc::T,Iv::Array{T,1},Jv::Array{T,1},dIvdk::Array{T,1},dJvdk::Array{T,1})  where {T <: Real}
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
  else
    Iv[1] = 2*acos(kc)
  end
# Try something else:
# Next, iterate upwards in v:
  f0 = kc*k
  v = 1
# Loop over v, computing I_v and J_v from higher v:
  while v <= v_max
    Iv[v+1]=((2v-1)*Iv[v]/2-f0)/v
    f0 *= k2
    v += 1
  end
  # Now compute compute derivatives:
  dIvdk[1] = 2/kc
  for v=1:v_max
    dIvdk[v+1] = k2*dIvdk[v]
  end
else # k^2 >= 1
  # Compute v=0
  Iv[1] = pi
  for v=1:v_max
    Iv[v+1]=Iv[v]*(1.0-0.5/v)
  end
  # Derivatives of I_v are zero:
  fill!(dIvdk,zero(k2))
end
# Need to compute J_v for v=0, 1:
v= 0
if k2 < 1
  # Use cel_bulirsch:
  if k2 > 0
    Jv[v+1]=2/(3k2*k)*cel_bulirsch(k2,kc,one(k2),k2*(3k2-1),k2*(1-k2))
    Jv[v+2]= 2/(15k2*k)*cel_bulirsch(k2,kc,one(k2),2k2*(3k2-2),k2*(4-7k2+3k2*k2))
    dJvdk[v+1] = 2*cel_bulirsch(k2,kc,one(k2),inv(k2),(1-inv(k2)))
    dJvdk[v+2] = -3*Jv[v+2]/k+k2*dJvdk[v+1]
  else
    Jv[v+1]= 0.0
    Jv[v+2]= 0.0
    dJvdk[v+1] = 0.0
    dJvdk[v+2] = 0.0
  end
else # k^2 >=1
  k2inv = inv(k2)
  Jv[v+1] = 2/3*cel_bulirsch(k2inv,kc,one(k2),3-k2inv,3-5k2inv+2k2inv^2)
  Jv[v+2] = cel_bulirsch(k2inv,kc,one(k2),12-8*k2inv,2*(9-8k2inv)*(1-k2inv))/15
  dJvdk[v+1] = 2/(k2*k)*cel_bulirsch(k2inv,kc,one(k2),one(k2),2*(1-k2inv))
  dJvdk[v+2] = -3*Jv[v+2]/k+k2*dJvdk[v+1]
end
v=2
while v <= v_max
  Jv[v+1] = (2*(v+1+(v-1)*k2)*Jv[v]-k2*(2v-3)*Jv[v-1])/(2v+3)
  dJvdk[v+1] = -3*Jv[v+1]/k+k2*dJvdk[v]
  v += 1
end
return
end

function dIJv_lower_dk!(v_max::Int64,k2::T,kc::T,Iv::Array{T,1},Jv::Array{T,1},dIvdk::Array{T,1},dJvdk::Array{T,1})  where {T <: Real}
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
  # Now compute compute derivatives:
  dIvdk[1] = 2/kc
  for v=1:v_max
    dIvdk[v+1] = k2*dIvdk[v]
  end
else # k^2 >= 1
  # Compute v=0 (no need to iterate downwards in this case):
  Iv[1] = pi
  for v=1:v_max
    Iv[v+1]=Iv[v]*(1-.5/v)
  end
  # Derivatives of I_v are zero:
  fill!(dIvdk,zero(k2))
end
v= v_max
# Need to compute top two for J_v:
#if typeof(k2) == BigFloat
  dJvdk0 = zero(typeof(k2)); dJvdk1 = zero(typeof(k2))
  Jv[v],dJvdk0 = dJv_seriesdk(k2,v-1); Jv[v+1],dJvdk1=dJv_seriesdk(k2,v)
  dJvdk[v] = dJvdk0; dJvdk[v+1] = dJvdk1
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
  dJvdk[v-1] = (dJvdk[v]+3/k*Jv[v])/k2
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
