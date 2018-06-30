# Computes a limb-darkened transit light curve with the dependence:
# I(\mu) = 1-\sum_{n=1}^N u_n (1-\mu)^n
# where \mu = \cos{\theta} = z is the cosine of the angle
# angle from the sub-stellar point (or equivalently the
# height on the star relative to the sky plane if the radius 
# of the star is unity.

include("sn_bigr.jl")

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
  v_max = round(Int64,N_c/2)+1
else
  v_max = round(Int64,(N_c-1)/2)+1
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
else
  lam = r^2*acos((r^2+b^2-1)/(2*b*r))+acos((1-r^2+b^2)/(2*b))-sqrt(b^2-.25*(1+b^2-r^2)^2)
end
sn[1] = pi-lam
sn[2] = s2(r,b)
if typeof(r) == Float64
  println("r: ",r," b: ",b," s2 error: ",convert(Float64,s2(big(r),big(b)))-sn[2])
end
# That's it!
#println("s_n: ",sn)
#println("c_n*s_n: ",c_n.*sn)
flux = sum(c_n.*sn)/(pi*(c_n[1]+2*c_n[2]/3))  # for c_2 and above, the flux is zero.
return flux
end
