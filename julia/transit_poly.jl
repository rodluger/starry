# Computes a limb-darkened transit light curve with the dependence:
# I(\mu) = 1-\sum_{n=1}^N u_n (1-\mu)^n
# where \mu = \cos{\theta} = z is the cosine of the angle
# angle from the sub-stellar point (or equivalently the
# height on the star relative to the sky plane if the radius 
# of the star is unity.

include("sn_bigr.jl")

function transit_poly(r::T,b::T,c_n::Array{T,1}) where {T <: Real}
# Number of limb-darkening components to include (beyond 0 and 1):
N_c = length(c_n)-1
# We are parameterizing these with the function:
# g_n = c_n [(n+2) z^n - n z^{n-2}] for n >= 2
# while g_{-2} = c_0 z^0 (uniform source) and g_{-1} = c_1 z^1 (linear limb-darkening)
# which gives a Green's integral of:
# P(G_n) = \int_{\pi-\phi}^{2\pi+\phi} (1-r^2-b^2-2br s_\varphi)^{n/2} (r+b s_\varphi) d\varphi
# for which we have a solution in terms of I_v (for even n) and J_v (for odd n).

# Set up a vector for storing results of P(G_n)-Q(G_n); note that
# this is a different vector than the Starry case:
sn = zeros(N_c+1)

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
  k2 = Inf; kc = 0.0
else
# Next, compute k^2 = m:
  k2 = (1.0-(b-r)^2)/(4b*r);
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

# Next, loop over the Green's function components:
for n=2:N_c
  pofgn = zero(r)
  if iseven(n)
# For even values of n, sum over I_v:
    n0 = convert(Int64,n/2)
    coeff = (-1)^n0
    # Compute i=0 term
    pofgn = coeff*((r-b)*Iv[n0+1]+2b*Iv[n0+2])
    kn = one(r)
# For even n, compute coefficients for the sum over I_v:
    println("n0: ",n0," i: ",0," coeff: ",coeff)
    for i=1:n0
      coeff *= -(n0-i+1)/i
      println("n0: ",n0," i: ",i," coeff: ",coeff)
      kn *= k2
      pofgn += coeff*kn*((r-b)*Iv[n0-i+1]+2b*Iv[n0-i+2])
    end
    pofgn *= 2r*(4*b*r)^n0
  else
# Now do the same for odd N_c in sum over J_v:
    n0 = convert(Int64,(n-3)/2)
    coeff = (-1)^n0
    # Compute i=0 term
    pofgn = coeff*((r-b)*Jv[n0+1]+2b*Jv[n0+2])
    kn = one(r)
    println("n0: ",n0," i: ",0," coeff: ",coeff)
# For even n, compute coefficients for the sum over I_v:
    for i=1:n0
      coeff *= -(n0-i+1)/i
      println("n0: ",n0," i: ",i," coeff: ",coeff)
      kn *= k2
      pofgn += coeff*kn*((r-b)*Jv[n0-i+1]+2b*Jv[n0-i+2])
    end
    pofgn *= 2r*(1-(b-r)^2)^1.5*(4*b*r)^n0
  end
# Q(G_n) is zero in this case since on limb of star z^{n+2} = 0
# Compute sn[n]:
  println("n: ",n," P(G_n): ",pofgn)
  sn[n+1] = pofgn
end
# Just compute sn[1] and sn[2], and then we're done. [ ]
if b <= 1-r
  lam = pi*r^2
else
  lam = r^2*acos((r^2+b^2-1)/(2*b*r))+acos((1-r^2+b^2)/(2*b))-sqrt(b^2-.25*(1+b^2-r^2)^2)
end
sn[1] = pi-lam
sn[2] = s2(r,b)
# That's it!
println("s_n: ",sn)
flux = sum(c_n.*sn)/(pi*(c_n[1]+2*c_n[2]/3))  # for c_2 and above, the flux is zero.
return flux
end
