#function area_triangle(a,b,c)
#a,b,c=reverse(sort([a,b,c]))
#area = .25*sqrt((a+(b+c))*(c-(a-b))*(c+(a-b))*(a+(b-c)))
#return area
#end


# Computes a limb-darkened transit light curve with the dependence:
# I(\mu) = 1-\sum_{n=1}^N u_n (1-\mu)^n
# where \mu = \cos{\theta} = z is the cosine of the angle
# angle from the sub-stellar point (or equivalently the
# height on the star relative to the sky plane if the radius 
# of the star is unity.

include("sn_bigr.jl")
include("IJv_derivative.jl")
include("area_triangle.jl")

function transit_poly(r::T,b::T,u_n::Array{T,1}) where {T <: Real}
# Transform the u_n coefficients to c_n, which are coefficients
# of the basis in which the P(G_n) functions are computed.
n = length(u_n)
c_n = zeros(typeof(r),n+3)
a_n = zeros(typeof(r),n+1)
a_n[1] = one(r)  # Add in the first constant coefficient term
for i=1:n
  # Compute the contribution to a_n*\mu^n
  for j=0:i
    a_n[j+1] -= u_n[i]*binomial(i,j)*(-1)^j
#    println("i: ",i," j: ",j," a_i: ",a_n[j+1])
  end
end
# Now, compute the c_n coefficients:
for j=n:-1:2
  c_n[j+1] = a_n[j+1]/(j+2)+c_n[j+3]
end
c_n[2] = a_n[2]+3*c_n[4]
c_n[1] = a_n[1]+2*c_n[3]
#println("u_n: ",u_n)
#println("a_n: ",a_n)
#if typeof(r) == Float64
#  println("c_n: ",c_n)
#end
return transit_poly_c(r,b,c_n[1:n+1])
end

function transit_poly_c(r::T,b::T,c_n::Array{T,1}) where {T <: Real}
# Number of limb-darkening components to include (beyond 0 and 1):
N_c = length(c_n)-1
# We are parameterizing these with the function:
# g_n = c_n [(n+2) z^n - n z^{n-2}] for n >= 2
# while g_{0} = c_0 z^0 (uniform source) and g_{1} = c_1 z^1 (linear limb-darkening)
# which gives a Green's integral of:
# P(G_n) = \int_{\pi-\phi}^{2\pi+\phi} (1-r^2-b^2-2br s_\varphi)^{n/2} (r+b s_\varphi) d\varphi
# for which we have a solution in terms of I_v (for even n) and J_v (for odd n).

# Set up a vector for storing results of P(G_n)-Q(G_n); note that
# this is a different vector than the Starry case:
sn = zeros(typeof(r),N_c+1)

# Check for different cases:
if b >= 1+r
  # unobscured - return one:
  return one(r)
end
if r >= 1+b
  # full obscuration - return zero:
  return zero(r)
end
if b == 0.0
  # Annular eclipse - integrate around the full boundary of both bodies:
  flux = zero(r); sqrt1mr2 = sqrt(1-r^2)
  flux = (c_n[1]*(1-r^2)+2/3*c_n[2]*sqrt1mr2^3)
  fac= 2r^2*(1-r^2)
  for i=2:N_c
    flux += -c_n[i+1]*fac
    fac *= sqrt1mr2
  end
  return flux/(c_n[1]+2*c_n[2]/3)
else
# Next, compute k^2 = m:
  onembmr2=(r-b+1)*(b-r+1); fourbr = 4b*r
  k2 = onembmr2/fourbr
  if k2 > 1
    if k2 > 2.0
      kc = sqrt(1.-inv(k2))
    else
      kc2 = (1-(b+r)^2)/(1-(b-r)^2)
      kc = sqrt(kc2)
    end
  else
    if k2 > 0.5
      kc2 = ((b+r)^2-1)/(4*b*r)
      kc = sqrt(kc2)
    else
      kc = sqrt(1.-k2)
    end
  end
end

# Compute the highest value of v in J_v or I_v that we need:
if iseven(N_c)
  v_max = round(Int64,N_c/2)+2
else
  v_max = round(Int64,(N_c-1)/2)+2
end
# Compute the J_v and I_v functions:
Iv = zeros(typeof(k2),v_max+1); Jv = zeros(typeof(k2),v_max+1)
if k2 > 0
  if k2 < 0.5 || k2 > 2.0
# This computes I_v,J_v for the largest v, and then works down to smaller values:
    IJv_lower!(v_max,k2,kc,Iv,Jv)
  else
# This computes I_0,J_0,J_1, and then works upward to larger v:
    IJv_raise!(v_max,k2,kc,Iv,Jv)
  end
end

#nphi = 1000; dphi=2pi/nphi; phigrid = linspace(.5*dphi,1-.5*dphi,nphi)
# Next, loop over the Green's function components:
for n=2:N_c
  pofgn = zero(r)
  if iseven(n)
# For even values of n, sum over I_v:
    n0 = convert(Int64,n/2)
    coeff = (-fourbr)^n0
    # Compute i=0 term
    pofgn = coeff*((r-b)*Iv[n0+1]+2b*Iv[n0+2])
# For even n, compute coefficients for the sum over I_v:
#    println("n0: ",n0," i: ",0," coeff: ",coeff)
    for i=1:n0
      coeff *= -(n0-i+1)/i*k2
#      println("n0: ",n0," i: ",i," coeff: ",coeff)
      pofgn += coeff*((r-b)*Iv[n0-i+1]+2b*Iv[n0-i+2])
    end
    pofgn *= 2r
  else
# Now do the same for odd N_c in sum over J_v:
    n0 = convert(Int64,(n-3)/2)
    coeff = (-fourbr)^n0
    # Compute i=0 term
    pofgn = coeff*((r-b)*Jv[n0+1]+2b*Jv[n0+2])
#    println("n0: ",n0," i: ",0," coeff: ",coeff)
# For even n, compute coefficients for the sum over I_v:
    for i=1:n0
      coeff *= -(n0-i+1)/i*k2
#      println("n0: ",n0," i: ",i," coeff: ",coeff)
      pofgn += coeff*((r-b)*Jv[n0-i+1]+2b*Jv[n0-i+2])
    end
    pofgn *= 2r*onembmr2^1.5
  end
#  pofgn_num = sum(sqrt.(1-r^2-b^2-2*b*r*sin.(phigrid)).^n.*(r+b.*sin.(phigrid))*r*dphi)
#  println("n: ",n," P(G_n): ",pofgn," P(G_n),num: ",pofgn_num)
# Q(G_n) is zero in this case since on limb of star z^n = 0 at the stellar
# boundary for n > 0.
# Compute sn[n]:
  #println("n: ",n," P(G_n): ",pofgn)
  sn[n+1] = -pofgn
end
# Just compute sn[1] and sn[2], and then we're done. [ ]
if b <= 1-r
  lam = pi*r^2
  sn[1] = pi-lam
else
  k=sqrt(k2)
  if k2 < 0.5
    kap = 2*asin(k)
  else
    kap = 2*acos(kc)
  end
  slam = ((1.0-r)*(1.0+r)+b^2)/(2*b);  clam = sqrt((1-b+r)*(1+b-r)*(b+r-1)*(b+r+1))/(2b);  lam = acos(clam); if slam < 0.; lam = -lam; end
  sn[1] = lam+pi/2+clam*slam-r^2*kap -4r^2*kc*k*(k2-.5)
#  sn[1] = lam+pi/2+clam*slam-8*r^2*(Iv[2]-Iv[3])
# These lines gave poor precision (based on Mandel & Agol 2002):
#  lam = r^2*acos((r^2+b^2-1)/(2*b*r))+acos((1-r^2+b^2)/(2*b))-sqrt(b^2-.25*(1+b^2-r^2)^2)
#  sn[1] = pi-lam
end
sn[2] = s2(r,b)
#if typeof(r) == Float64
#  println("r: ",r," b: ",b," s2 error: ",convert(Float64,s2(big(r),big(b)))-sn[2])
#end
# That's it!
#println("s_n: ",sn)
#println("c_n*s_n: ",c_n.*sn)
flux = sum(c_n.*sn)/(pi*(c_n[1]+2*c_n[2]/3))  # for c_2 and above, the flux is zero.
return flux
end

function transit_poly!(r::T,b::T,u_n::Array{T,1},dfdrbu::Array{T,1}) where {T <: Real}
# Transform the u_n coefficients to c_n, which are coefficients
# of the basis in which the P(G_n) functions are computed.
# Compute the derivatives of the flux with respect to the u coefficients.
n = length(u_n)
# We define c_n with two extra elements which are zero:
c_n = zeros(typeof(r),n+3)
dfdrbc = zeros(typeof(r),n+3)
a_n = zeros(typeof(r),n+1)
dadu = zeros(typeof(r),n+1,n)
dcdu = zeros(typeof(r),n+3,n)
a_n[1] = one(r)  # Add in the first constant coefficient term
for i=1:n
  # Compute the contribution to a_n*\mu^n
  for j=0:i
    a_n[j+1] -= u_n[i]*binomial(i,j)*(-1)^j
    dadu[j+1,i] -= binomial(i,j)*(-1)^j
#    println("i: ",i," j: ",j," a_i: ",a_n[j+1])
  end
end
# Now, compute the c_n coefficients and propagate derivatives:
for j=n:-1:2
  c_n[j+1] = a_n[j+1]/(j+2)+c_n[j+3]
  for i=1:n
    dcdu[j+1,i] = dadu[j+1,i]/(j+2) + dcdu[j+3,i]
  end
end
c_n[2] = a_n[2]+3*c_n[4]
for i=1:n
  dcdu[2,i] = dadu[2,i] + 3*dcdu[4,i]
end
c_n[1] = a_n[1]+2*c_n[3]
for i=1:n
  dcdu[1,i] = dadu[1,i] + 2*dcdu[3,i]
end
#println("a_n: ",a_n)
if typeof(r) == Float64
  println("u_n: ",u_n)
  println("c_n: ",c_n)
  println("dcdu: ",dcdu)
end
# Pass c_n (without last two dummy values):
flux = transit_poly_c!(r,b,c_n[1:n+1],dfdrbc)
# Now, transform derivaties from c to u:
dfdrbu[1] = dfdrbc[1]  # r derivative
dfdrbu[2] = dfdrbc[2]  # b derivative
# u_n derivatives:
for i=1:n, j=0:n
  dfdrbu[i+2] += dfdrbc[j+3]*dcdu[j+1,i]
end
return flux
end

function transit_poly_c!(r::T,b::T,c_n::Array{T,1},dfdrbc::Array{T,1}) where {T <: Real}
@assert((length(c_n)+2) == length(dfdrbc))
# Number of limb-darkening components to include (beyond 0 and 1):
N_c = length(c_n)-1
# We are parameterizing these with the function:
# g_n = c_n [(n+2) z^n - n z^{n-2}] for n >= 2
# while g_{0} = c_0 z^0 (uniform source) and g_{1} = c_1 z^1 (linear limb-darkening)
# which gives a Green's integral of:
# P(G_n) = \int_{\pi-\phi}^{2\pi+\phi} (1-r^2-b^2-2br s_\varphi)^{n/2} (r+b s_\varphi) d\varphi
# for which we have a solution in terms of I_v (for even n) and J_v (for odd n).
# Compute the derivative of the flux with respect to the different coefficients.

# Set up a vector for storing results of P(G_n)-Q(G_n); note that
# this is a different vector than the Starry case:
sn = zeros(typeof(r),N_c+1)
dsndr = zeros(typeof(r),N_c+1)
dsndb = zeros(typeof(r),N_c+1)
fill!(dfdrbc,zero(r))
# Check for different cases:
if b >= 1+r
  # unobscured - return one:
  return one(r)
end
if r >= 1+b
  # full obscuration - return zero:
  return zero(r)
end
if b == 0.0
  # Annular eclipse - integrate around the full boundary of both bodies:
  flux = zero(r); sqrt1mr2 = sqrt(1-r^2)
  flux = (c_n[1]*(1-r^2)+2/3*c_n[2]*sqrt1mr2^3)
  fac= 2r^2*(1-r^2)
  for i=2:N_c
    flux += -c_n[i+1]*fac
    fac *= sqrt1mr2
  end
  # Also need to compute derivatives [ ]
  return flux/(c_n[1]+2*c_n[2]/3)
else
# Next, compute k^2 = m:
  onembmr2=(r-b+1)*(b-r+1); fourbr = 4b*r
  k2 = onembmr2/fourbr; k = sqrt(k2)
  dkdr = (b^2-r^2-1)/(8*k*b*r^2)
  dkdb = (r^2-b^2-1)/(8*k*b^2*r)
  if k2 > 1
    if k2 > 2.0
      kc = sqrt(1.-inv(k2))
    else
#      kc2 = (1-(b+r)^2)/(1-(b-r)^2)
      kc2 = (1-b-r)*(1+b+r)/(1-b+r)/(1-r+b)
      kc = sqrt(kc2)
    end
  else
    if k2 > 0.5
#      kc2 = ((b+r)^2-1)/(4*b*r)
      kc2 = (b+r-1)*(b+r+1)/(4*b*r)
      kc = sqrt(kc2)
    else
      kc = sqrt(1.-k2)
    end
  end
end

# Compute the highest value of v in J_v or I_v that we need:
if iseven(N_c)
  v_max = round(Int64,N_c/2)+2
else
  v_max = round(Int64,(N_c-1)/2)+2
end
println("v_max: ",v_max," N_c: ",N_c)
# Compute the J_v and I_v functions:
Iv = zeros(typeof(k2),v_max+1); Jv = zeros(typeof(k2),v_max+1)
# And their derivatives with respect to k:
dIvdk = zeros(typeof(k2),v_max+1); dJvdk = zeros(typeof(k2),v_max+1)
if k2 > 0
  if k2 < 0.5 || k2 > 2.0
# This computes I_v,J_v for the largest v, and then works down to smaller values:
    dIJv_lower_dk!(v_max,k2,kc,Iv,Jv,dIvdk,dJvdk)
  else
# This computes I_0,J_0,J_1, and then works upward to larger v:
    dIJv_raise_dk!(v_max,k2,kc,Iv,Jv,dIvdk,dJvdk)
  end
end

#nphi = 1000; dphi=2pi/nphi; phigrid = linspace(.5*dphi,1-.5*dphi,nphi)
# Next, loop over the Green's function components:
for n=2:N_c
  pofgn = zero(r)
  dpdr = zero(r)
  dpdb = zero(r)
  dpdk = zero(r)
  if iseven(n)
# For even values of n, sum over I_v:
    n0 = convert(Int64,n/2)
    coeff = (-fourbr)^n0
    # Compute i=0 term
    pofgn = coeff*((r-b)*Iv[n0+1]+2b*Iv[n0+2])
    dpdr = coeff*Iv[n0+1]
    dpdb = coeff*(-Iv[n0+1]+2*Iv[n0+2])
    dpdk = coeff*((r-b)*dIvdk[n0+1]+2b*dIvdk[n0+2])
# For even n, compute coefficients for the sum over I_v:
#    println("n0: ",n0," i: ",0," coeff: ",coeff)
    for i=1:n0
      coeff *= -(n0-i+1)/i*k2
#      println("n0: ",n0," i: ",i," coeff: ",coeff)
      pofgn += coeff*((r-b)*Iv[n0-i+1]+2b*Iv[n0-i+2])
      dpdr += coeff*Iv[n0-i+1]
      dpdb += coeff*(-Iv[n0-i+1]+2*Iv[n0-i+2])
      dpdk += coeff*((r-b)*dIvdk[n0-i+1]+2b*dIvdk[n0-i+2])
      dpdk += coeff*2*i/k*((r-b)*Iv[n0-i+1]+2b*Iv[n0-i+2])
    end
    pofgn *= 2r
    dpdr *= 2r
    dpdr += (n0+1)*pofgn/r
    dpdb *= 2r
    dpdb += n0*pofgn/b
    dpdk *= 2r
  else
# Now do the same for odd N_c in sum over J_v:
    n0 = convert(Int64,(n-3)/2)
    coeff = (-fourbr)^n0
    # Compute i=0 term
    pofgn = coeff*((r-b)*Jv[n0+1]+2b*Jv[n0+2])
    dpdr = coeff*Jv[n0+1]
    dpdb = coeff*(-Jv[n0+1]+2*Jv[n0+2])
    dpdk = coeff*((r-b)*dJvdk[n0+1]+2b*dJvdk[n0+2])
#    println("n0: ",n0," i: ",0," coeff: ",coeff)
# For even n, compute coefficients for the sum over I_v:
    for i=1:n0
      coeff *= -(n0-i+1)/i*k2
#      println("n0: ",n0," i: ",i," coeff: ",coeff)
      pofgn += coeff*((r-b)*Jv[n0-i+1]+2b*Jv[n0-i+2])
      dpdr  +=  coeff*Jv[n0-i+1]
      dpdb  +=  coeff*(-Jv[n0-i+1]+2*Jv[n0-i+2])
      dpdk  += coeff*((r-b)*dJvdk[n0-i+1]+2b*dJvdk[n0-i+2])
      dpdk  += coeff*2*i/k*((r-b)*Jv[n0-i+1]+2b*Jv[n0-i+2])
    end
    pofgn *= 2r*onembmr2^1.5
    dpdr *= 2r*onembmr2^1.5
    dpdr += ((n0+1)/r+3*(b-r)/onembmr2)*pofgn
    dpdb *= 2r*onembmr2^1.5
    dpdb += (n0/b-3*(b-r)/onembmr2)*pofgn
    dpdk *= 2r*onembmr2^1.5
  end
#  pofgn_num = sum(sqrt.(1-r^2-b^2-2*b*r*sin.(phigrid)).^n.*(r+b.*sin.(phigrid))*r*dphi)
#  println("n: ",n," P(G_n): ",pofgn," P(G_n),num: ",pofgn_num)
# Q(G_n) is zero in this case since on limb of star z^n = 0 at the stellar
# boundary for n > 0.
# Compute sn[n]:
  #println("n: ",n," P(G_n): ",pofgn)
  sn[n+1] = -pofgn
  dsndr[n+1] = -(dpdr+dpdk*dkdr)
  dsndb[n+1] = -(dpdb+dpdk*dkdb)
end
# Just compute sn[1] and sn[2], and then we're done. [ ]
if b <= 1-r
  lam = pi*r^2
  sn[1] = pi-lam
  dsndr[1] = -2*pi*r
  dsndb[1] = 0.
else
  k=sqrt(k2)
  if k2 < 0.5
#    kap = 2*asin(k)
    kap = 2*atan2(sqrt((1-b+r)*(1+b-r)),sqrt((b+r-1)*(b+r+1)))
  else
    kap = 2*acos(kc)
  end
#  slam = ((1.0-r)*(1.0+r)+b^2)/(2*b);  clam = sqrt((1-b+r)*(1+b-r)*(b+r-1)*(b+r+1))/(2b);  lam = acos(clam); if slam < 0.; lam = -lam; end
  slam = ((1.0-r)*(1.0+r)+b^2)/(2*b);  clam = 2*area_triangle(1.,b,r)/b;  lam = acos(clam); if slam < 0.; lam = -lam; end
#  sn[1] = lam+pi/2+clam*slam-r^2*kap -4r^2*kc*k*(k2-.5)
  dsndr[1]= -2*r*kap
#  dsndr[1]= -r*(pi+2*asin((1-r^2-b^2)/(2*b*r)))
  dsndb[1]= 2*clam
  sn[1] = lam+pi/2+clam*slam-8*r^2*(Iv[2]-Iv[3])
# These lines gave poor precision (based on Mandel & Agol 2002):
#  lam = r^2*acos((r^2+b^2-1)/(2*b*r))+acos((1-r^2+b^2)/(2*b))-sqrt(b^2-.25*(1+b^2-r^2)^2)
#  sn[1] = pi-lam
end
s2_grad = zeros(typeof(r),2)
sn[2] = s2!(r,b,s2_grad)
dsndr[2] = s2_grad[1]
dsndb[2] = s2_grad[2]
#if typeof(r) == Float64
#  println("r: ",r," b: ",b," s2 error: ",convert(Float64,s2(big(r),big(b)))-sn[2])
#end
# That's it!
#println("s_n: ",sn)
#println("c_n*s_n: ",c_n.*sn)
# Compute derivatives with respect to the coefficients:
den = inv(pi*(c_n[1]+2*c_n[2]/3))
flux = zero(r)
dfdrbc[1]=zero(r)  # Derivative with respect to r
dfdrbc[2]=zero(r)  # Derivative with respect to b
for n=0:N_c
  # derivatives with respect to the coefficients:
  dfdrbc[n+3]= sn[n+1]*den
  # total flux:
  flux += c_n[n+1]*dfdrbc[n+3]
  # derivatives with respect to r and b:
  dfdrbc[1] += c_n[n+1]*dsndr[n+1]*den
  dfdrbc[2] += c_n[n+1]*dsndb[n+1]*den
end
# Include derivatives with respect to first two c_n parameters:
dfdrbc[3] -= flux*den*pi
dfdrbc[4] -= flux*den*2pi/3
#flux = sum(c_n.*sn)*den   # for c_2 and above, the flux integrated over the star is zero.
return flux
end
