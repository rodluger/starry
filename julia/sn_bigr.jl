# Computes s_n vector from Luger et al. (2018) to ~machine
# precision for b+r > 1:
include("s2_stable.jl")

#using GSL

function vector_sum(x::Array{T,1},y::Array{T,1},n::Int64) where {T <: Real}
# Trying to stabilize arithmetic:
sum_pos = zero(typeof(x[1]))
sum_neg = zero(typeof(x[1]))
# Sort from smallest to largest:
ind = sortperm(abs.(x.*y))
#println(x[ind].*y[ind])
# Carry out sum from smallest to largest (keeping track of positive and negative
# terms separately):
for i=1:n
#  term = x[i]*y[i]
  term = x[ind[i]]*y[ind[i]]
  if term > 0.0
    sum_pos += term
  else
    sum_neg += term
  end
end
# Add together the results:
return sum_pos+sum_neg
end

function aiuv(delta::T,u::Int64,v::Int64) where {T <: Real}
# Computes the double-binomial coefficients A_{i,u,v}:
a=zeros(typeof(delta),u+v+1)
for i=0:u+v
  j1 = maximum([0,u-i])
#  coeff = binomial(u,j1)*binomial(v,u+v-i-j1)*(-1.)^(u+j1)*delta^(v+u-i-j1)
#  a[i+1] = coeff
  for j=j1:minimum([u+v-i,u])
#  for j=j1+1:minimum([u+v-i,u])
#    coeff *= -(u-j+1)*(u+v-i-j+1)/(j*(i+j-u)*delta)
    a[i+1] += binomial(u,j)*binomial(v,u+v-i-j)*(-1.)^(u+j)*delta^(v+u-i-j)
#    a[i+1] += coeff
  end
end
return a
end

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
  return  sqrt(pi)*gamma(v+.5)*(sf_hyperg_2F1_renorm(-.5,v+.5,v+1,k2inv)-(.5+v)*k2inv*sf_hyperg_2F1_renorm(-.5,v+1.5,v+2,k2inv))
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
    coeff *= 1-.5/i
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

function IJv_raise!(l_max::Int64,k2::T,kc::T,Iv::Array{T,1},Jv::Array{T,1})  where {T <: Real}
# This function needs debugging. [ ]
# Compute I_v, J_v for 0 <= v <= v_max = l_max+2
# Define k:
k = sqrt(k2)
# Iterate upwards in v:
v_max = l_max+3; v = v_max
# Compute I_v via upward iteration on v:
if k2 < 1
# First, compute value for v=0:
  Iv[1] = 2*asin(sqrt(k2))
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
#  fe = 2*(2k2-1); fk = (1-k2)*(2-3k2)
#  Jv[v+1]=2/(3k2*k)*cel_bulirsch(k2,kc,one(k2),fk+fe,fk+fe*(1-k2))
  if k2 > 0
#    println("k2: ",k2)
    Jv[v+1]=2/(3k2*k)*cel_bulirsch(k2,kc,one(k2),k2*(3k2-1),k2*(1-k2))
    fe = -3k2*k2+13k2-8; fk = (1-k2)*(8-9k2)
    Jv[v+2]= 2/(15k2*k)*cel_bulirsch(k2,kc,one(k2),2k2*(3k2-2),k2*(4-7k2+3k2*k2))
  else
    Jv[v+1]= 0.0
    Jv[v+2]= 0.0
  end
else # k^2 >=1
  k2inv = inv(k2)
#  fe = 2*(2-k2inv); fk=-1+k2inv
#  Jv[v+1]=2/3*cel_bulirsch(k2inv,kc,one(k2),fk+fe,fk+fe*(1-k2inv))
  Jv[v+1]=2/3*cel_bulirsch(k2inv,kc,one(k2),3-k2inv,3-5k2inv+2k2inv^2)
#  fe = -6k2+26-16k2inv; fk=2*(1-k2inv)*(3k2-4)
#  Jv[v+2]=cel_bulirsch(k2inv,kc,one(k2),fk+fe,fk+fe*(1-k2inv))/15
  Jv[v+2]=cel_bulirsch(k2inv,kc,one(k2),2-8*k2inv,8*(1-2k2inv)*(1-k2inv))/15
end
v=2
while v <= v_max
  Jv[v+1] = (2*(v+1+(v-1)*k2)*Jv[v]-k2*(2v-3)*Jv[v-1])/(2v+3)
  v += 1
end
return
end

function IJv_tridiag!(l_max::Int64,k2::T,kc::T,Iv::Array{T,1},Jv::Array{T,1})  where {T <: Real}
# Compute I_v, J_v for 0 <= v <= v_max = l_max+2
# Define k:
k = sqrt(k2)
# Iterate downwards in v:
v_max = l_max+3; v = v_max
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
c = zeros(v_max-1)
d = zeros(v_max-1)
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

function IJv_lower!(l_max::Int64,k2::T,kc::T,Iv::Array{T,1},Jv::Array{T,1})  where {T <: Real}
# Compute I_v, J_v for 0 <= v <= v_max = l_max+2
# Define k:
k = sqrt(k2)
# Iterate downwards in v:
v_max = l_max+3; v = v_max
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
return
end

function Huv_down(l_max,kappa)
# Iterate downwards in v and upwards in u:
Huv = zeros(typeof(kappa),l_max+3,3l_max+3)
kap2 = 0.5*kappa
ckap = cos(kap2); skap = sin(kap2)
skap2 = skap*skap
ckap2 = 1.0-skap2; ckapn = ckap; skapn = skap
u=0
if isodd(l_max)
  v = 3l_max+1
else
  v = 3l_max+2
end
# First, compute approximation for large v:
Huv[u+1,v+1]=kap2^(v+1)*(2/(v+1)-(u+v/3)/(v+3)*kap2^2+
          (5*(3u+v)^2-30u-2v)/360*kap2^4)
#println("Huv: ",Huv[u+1,v+1]," error: ",last_term)
println("Huv: ",Huv[u+1,v+1])
# Next, iterate downwards in v:
while v >= 2
  Huv[u+1,v-1] = (u+v)/(v-1)*(Huv[u+1,v+1]+2/(u+v)*skap^(v-1)*ckap^(u+1))
  v -= 2
end
# Then, iterate upwards in u:
for u=2:2:l_max+2
  skapn = skap
  v = 0
  while v <= 3l_max+2
    Huv[u+1,v+1]= (2*ckapn*skapn+(u-1)*Huv[u-1,v+1])/(u+v)
    # We only need even terms:
    skapn *= skap2
    v +=2
  end
  ckapn *= ckap2
end
return Huv
end

function s_n_bigr!(l_max::Int64,r::T,b::T,sn::Array{T,1}) where {T <: Real}
@assert(r > 0.0) # if r=0, then no occultation - can just use phase curve term.
# Computes the s_n terms up to l_max
# Find n_max:
n_max = l_max^2+2*l_max
u_max = floor(Int64,l_max/2+1)
v_max = l_max + 3
if r >= 1+b
  # full obscuration - return zeros
  return sn
end
if b == 0.0
  # Annular eclipse - integrate around the full boundary of both bodies:
  lam = pi/2; slam = one(r); clam = zero(r); clam2 = zero(r)
  phi = pi/2
  k2 = Inf; kc = 0.0
else
  if b > abs(1.0-r) && b < 1.0+r
#    lam = asin((1.0-r^2+b^2)/(2*b)); slam = (1.0-r^2+b^2)/(2*b); clam = sqrt((1-(b-r)^2)*((b+r)^2-1))/(2b)
#    slam = (1.0-r^2+b^2)/(2*b); lam = asin(slam);  clam = sqrt((1-(b-r)^2)*((b+r)^2-1))/(2b)
#    slam = (1.0-r^2+b^2)/(2*b);  clam = sqrt((1-(b-r)^2)*((b+r)^2-1))/(2b);  clam2 = (1-(b-r)^2)*((b+r)^2-1)/(4b^2); lam = acos(clam); if lam > pi/2; lam -= pi; end
    slam = ((1.0-r)*(1.0+r)+b^2)/(2*b);  clam = sqrt((1-b+r)*(1+b-r)*(b+r-1)*(b+r+1))/(2b);  clam2 = (1-b+r)*(1+b-r)*(b+r-1)*(b+r+1)/(4b^2); lam = acos(clam); if lam > pi/2; lam -= pi; end
    phi = asin((1.0-r^2-b^2)/(2*b*r))
  else
    lam=pi/2; phi=pi/2; slam = one(r); clam = zero(r); clam2 = zero(r)
  end
# Next, compute k^2 = m:
  k2 = (1.0-(b-r)^2)/(4b*r); kc = sqrt(abs(((b+r)^2-1))/(4*b*r))
  if k2 > 1
    kc = sqrt(abs(((b+r)^2-1)/((b-r)^2-1)))
  end
end

function Hv_raise!(l_max::Int64,k2::T,kc::T,Hv::Array{T,1})  where {T <: Real}
# This Iv function is defined for Q(G_n)/H_uv calculation:
#Iv = zeros(typeof(k2),v_max+1)
# Compute H_uv,  0 <= v <= v_max = l_max+2
# Define k:
k = sqrt(k2)
# Iterate upwards in v:
v_max = l_max+3; v = v_max
# Compute I_v via upward iteration on v:
if k2 < 1
# First, compute value for v=0:
  Hv[1] = 2*asin(k)
# Next, iterate upwards in v:
#  f0 = kc/k
  f0 = kc*k
  v = 1
# Loop over v, computing I_v and J_v from higher v:
  while v <= v_max
    Hv[v+1]=((2v-1)*Hv[v]/2-f0)/v
    f0 *= k2
    v += 1
  end
else # k^2 >= 1
  # Compute v=0
  Hv[1] = pi
  for v=1:v_max
    Hv[v+1]=Hv[v]*(1-.5/v)
  end
end
return
end

function Hv_lower!(l_max::Int64,k2::T,kc::T,Hv::Array{T,1})  where {T <: Real}
# Compute H_v for 0 <= v <= v_max = l_max+2
# Define k:
k = sqrt(k2)
# Iterate downwards in v:
v_max = l_max+3; v = v_max
# Add in k2 > 1 cases [ ]
# First, compute approximation for large v:
if k2 < 1
  Hv[v+1]=Iv_series(k2,v)
# Next, iterate downwards in v:
  f0 = k2^(v-1)*k*kc
# Loop over v, computing I_v and J_v from higher v:
  while v >= 1
    Hv[v] = 2/(2v-1)*(v*Hv[v+1]+f0)
    f0 /= k2
    v -= 1
  end
else # k^2 >= 1
  # Compute v=0 (no need to iterate downwards in this case):
  Hv[1] = pi
  for v=1:v_max
    Hv[v+1]=Hv[v]*(1-.5/v)
  end
end
return
end

# First, compute Huv:
#Hv = zeros(typeof(r),v_max+1)
#k2H = ((b+1)^2-r^2)/(4b)
#if k2H < 0.5 
#  Hv_lower!(l_max,k2H,sqrt(abs((r^2-(1-b)^2))/(4b)),Hv)
#else
#  Hv_raise!(l_max,k2H,sqrt(abs((r^2-(1-b)^2))/(4b)),Hv)
#end
Huv = zeros(typeof(r),l_max+3,l_max+1)
##clam = cos(lam); slam = sin(lam)
#clam2 = clam*clam; clamn = clam; slamn = slam
#for u=0:2:l_max+2
#  if u == 0
#    Huv[1,1]=  2*lam+pi
#    Huv[1,2]= -2*clam
#    slamn = slam
#    v=2
#    while v <= l_max
#      Huv[1,v+1]= (-2*clam*slamn+(v-1)*Huv[1,v-1])/(u+v)
#      slamn *= slam
#      v+=1
#    end
#  else
#    slamn = slam
#    v = 0
##    Huv[u+1,v+1]= (2*clamn*slamn+(u-1)*Huv[u-1,v+1])/(u+v)
##    slamn *= slam
##    v = 1
##    Huv[u+1,v+1]= (2*clamn*slamn+(u-1)*Huv[u-1,v+1])/(u+v)
##    slamn *= slam
##    v = 2
#    while v <= l_max
##      Huv[u+1,v+1]= (-2*clam*slamn+(v-1)*Huv[u+1,v-1])/(u+v)
#      Huv[u+1,v+1]= (2*clamn*slamn+(u-1)*Huv[u-1,v+1])/(u+v)
#      slamn *= slam
#      v+=1
#    end
#    clamn *= clam2
#  end
#end
clam2 = clam*clam; clamn = clam; slamn = slam
for u=0:2:l_max+2
  if u == 0
    Huv[1,1]=  2*lam+pi
    Huv[1,2]= -2*clam
    slamn = slam
    v=2
    while v <= l_max
      Huv[1,v+1]= (-2*clam*slamn+(v-1)*Huv[1,v-1])/(u+v)
      slamn *= slam
      v+=1
    end
  else
    slamn = slam
    v=0
    Huv[u+1,v+1]= (2*clamn*slamn+(u-1)*Huv[u-1,v+1])/(u+v)
    slamn *= slam
    v=1
#    Huv[u+1,v+1]= (2*clamn*slamn+(u-1)*Huv[u-1,v+1])/(u+v)
    clamn *= clam2
    Huv[u+1,v+1] = -2*clamn/(u+1)
#    println("u: ",u," v: ",v," Huv[u+1,v+1] ",Huv[u+1,v+1]," -2*clamn/(u+1): ", -2*clamn/(u+1))
    slamn = slam
    v=2
    while v <= l_max
      Huv[u+1,v+1]= (-2*clamn*slamn+(v-1)*Huv[u+1,v-1])/(u+v)
      slamn *= slam
      v+=1
    end
  end
end


#println("gamma: ",asin(-clam))
#Huv = Huv_down(l_max,acos(-slam))

Iv = zeros(typeof(k2),v_max+1); Jv = zeros(typeof(k2),v_max+1)
# This computes I_v for the largest v, and then works down to smaller values:
if k2 > 0
  if k2 < 0.5 || k2 > 2.0
    IJv_lower!(l_max,k2,kc,Iv,Jv)
  else
    IJv_raise!(l_max,k2,kc,Iv,Jv)
  end
end
#println("Iv: ",Iv," Jv: ",Jv)
#IJv_raise!(l_max,k2,kc,Iv,Jv)
#IJv_tridiag!(l_max,k2,kc,Iv,Jv)
#Ivr = zeros(typeof(k2),v_max+1); Jvr = zeros(typeof(k2),v_max+1)
#IJv_raise!(l_max,k2,kc,Ivr,Jvr)
#println("Jv lower: ",Jv," Jv raise: ",Jvr," diff: ",Jv-Jvr)
#Ivt = zeros(typeof(k2),v_max+1); Jvt = zeros(typeof(k2),v_max+1)
#IJv_tridiag!(l_max,k2,kc,Ivt,Jvt)
#println("Jv lower: ",Jv," Jv tridiag: ",Jvt," diff: ",Jv-Jvt)
Kuv = zeros(typeof(r),u_max+1,v_max+1)
Luv = zeros(typeof(r),u_max+1,v_max+1,2)
delta = (b-r)/(2r)
l = 0; n = 0; m = 0; pofgn = zero(typeof(r)); qofgn = zero(typeof(r))
#  k^3*(4br)^(3/2) = (1-(2r\delta)^2)^{3/2}:
#Lfac = (1-(2r*delta)^2)^1.5
Lfac = (1-(b-r)^2)^1.5
while n <= n_max
  if n == 2
    sn[n+1] = s2(r,b)
  else
    mu = l-m; nu = l+m; u=0; v=0
    pofgn = zero(typeof(r)); qofgn = zero(typeof(r))
    # Equation for P(G_n) and Q(G_n):
    if (isodd(mu) && isodd(round(Int64,(mu-1)/2))) || (iseven(mu) && isodd(round(Int64,mu/2)))
      # These cases are zero
    else
      # First, get values of u and v:
      if mod(mu,4) == 0
        u = convert(Int64,mu/4)+1; v= convert(Int64,nu/2)
      elseif iseven(l) && mu == 1
        u = convert(Int64,l/2)-1; v=0
      elseif isodd(l) && mu == 1
        u = convert(Int64,(l-1)/2)-1; v=1
      else
        u=convert(Int64,(mu-1)/4); v=convert(Int64,(nu-1)/2)
      end
      # If they haven't been computed yet, compute Kuv, Luv:
      if Kuv[u+1,v+1] == 0.0 && Luv[u+1,v+1,1] == 0.0 && Luv[u+1,v+1,2] == 0.0
        # First, compute double-binomial coefficients:
        a=aiuv(delta,u,v)
#        Kuv[u+1,v+1]   = sum(a[1:u+v+1].*Iv[u+1:2u+v+1])
        Kuv[u+1,v+1]   = vector_sum(a[1:u+v+1],Iv[u+1:2u+v+1],u+v+1)
#        println("Kuv 1 ",sum(a[1:u+v+1].*Iv[u+1:2u+v+1])," Kuv 2 ",Kuv[u+1,v+1])
#        Luv[u+1,v+1,1] = Lfac*sum(a[1:u+v+1].*Jv[u+1:v+2u+1])
        Luv[u+1,v+1,1] = Lfac*vector_sum(a[1:u+v+1],Jv[u+1:v+2u+1],u+v+1)
        if v <= 1
#          Luv[u+1,v+1,2] = Lfac*sum(a[1:u+v+1].*Jv[u+2:v+2u+2])
          Luv[u+1,v+1,2] = Lfac*vector_sum(a[1:u+v+1],Jv[u+2:v+2u+2],u+v+1)
        end
      end
      # Now, compute P(Gn) & Q(Gn):
      if mod(mu,4) == 0
        pofgn = 2*(2r)^(l+2)*Kuv[u+1,v+1]
# Use alternate form of Kuv (6/11/2018 notes):
#        Kuv_alt = zero(r)
#        coeff = exp(2*lgamma(u+.5)-lgamma(2u+1))
#        Kuv_alt = coeff*delta^v
#        for i=1:v
#          coeff *= (u+i-.5)/(2u+i)
#          Kuv_alt += binomial(v,i)*delta^(v-i)*coeff
#        end
#        println("u ",u," v ",v," Kuv ",Kuv[u+1,v+1]," Kuv_alt ",Kuv_alt)
##        pofgn = 2*(2r)^(l+2)*Kuv_alt
        if iseven(v) || k2 <= 1  # Q is zero for odd v & k^2 > 1
          qofgn = Huv[2u+1,v+1]
#          a=aiuv(-convert(typeof(k2),0.5),u,v)
#          qofgn = 2^(2u+v+1)*sum(a[1:u+v+1].*Hv[u+1:2u+v+1])
#          qofgn = 2^(2u+v+1)*vector_sum(a[1:u+v+1],Hv[u+1:2u+v+1],u+v+1)
#          println("u ",u," v ",v," Huv ",Huv[2u+1,v+1]," Qnew ",qofgn)
        end
      else
        pofgn = Luv[u+1,v+1,1]
        if mu == 1 
          pofgn -= 2*Luv[u+1,v+1,2]
        else
          pofgn *= 2
        end
        pofgn *= (2r)^(l-1)
      end
#      elseif iseven(l) && mu == 1
#        pofgn = -(2r)^(l-1)*(2*Luv[u+1,v+1,2]-Luv[u+1,v+1,1])
#      elseif isodd(l) && mu == 1
#        pofgn = -(2r)^(l-1)*(2*Luv[u+1,v+1,2]-Luv[u+1,v+1,1])
#      else
#        pofgn = 2*(2r)^(l-1)*Luv[u+1,v+1,1]
    end
    sn[n+1] = qofgn-pofgn
#    sn[n+1] = -pofgn
#    sn[n+1] = qofgn
  end
  m +=1
  if m > l
    l += 1
    m = -l
  end
  n += 1
end
# Return the vector of coefficients:
#return sn
return
end
