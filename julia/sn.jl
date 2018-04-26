# Computes the s_n terms from STARRY:

include("mpq.jl")
include("ellk_bulirsch.jl")
include("ellec_bulirsch.jl")
include("s2.jl")

function s_n!(l_max::Int64,r::T,b::T,sn::Array{T,1}) where {T <: Real}
@assert(r > 0.0) # if r=0, then no occultation - can just use phase curve term.
# Computes the s_n terms up to l_max
# Find n_max:
n_max = l_max^2+2*l_max
# sn = zeros(typeof(r),n_max+1); 
Kofk = zero(typeof(b)); Eofk = zero(typeof(b))
# First, compute lambda and phi:
if b == 0.0
  if r >= 1.0
    # full obscuration - return zeros
    return sn     
  else
    # Annular eclipse - integrate around the full boundary of both bodies:
    lam = pi/2
    phi = pi/2
  end
  k2 = Inf
  # Elliptic integrals K(0) & E(0):
  Kofk = pi/2
  Eofk = pi/2
else
  if b > abs(1.0-r) && b < 1.0+r
    lam = asin((1.0-r^2+b^2)/(2*b))
    phi = asin((1.0-r^2-b^2)/(2*b*r))
  else
    lam=pi/2; phi=pi/2
  end
# Next, compute k^2 = m:
  k2 = (1.0-(r-b)^2)/(4*b*r)
# Compute elliptic integrals:
  if k2 < 1.0
    Kofk = ellk_bulirsch(k2)
    Eofk = ellec_bulirsch(k2)
  elseif k2 > 1.0
    # Need to compute inverse of k2:
    k2inv = 1.0/k2
    Kofk = ellk_bulirsch(k2inv)
    Eofk = ellec_bulirsch(k2inv)
  else
    # For k2=1.0, K(k2) diverges, E(k2) is 1:
    Kofk = Inf
    Eofk = 1.0
  end
end

#println("K(m): ",Kofk," E(m): ",Eofk)
#println("phi: ",phi," lam: ",lam)

# Second, pre-compute the I, H & J functions:
# mu goes from 0 to 2*l_max, so u needs to go from 0 up to l_max+2.
# nu goes up to 2*l_max, so v goes from 0 up to l_max.
Iuv = zeros(typeof(r),l_max+3,l_max+1)
Huv = zeros(typeof(r),l_max+3,l_max+1)
clam = cos(lam); slam = sin(lam)
clam2 = clam*clam; clamn = clam; slamn = slam
cphi = cos(phi); sphi = sin(phi)
cphi2 = cphi*cphi; cphin = cphi
for u=0:2:l_max+2
  if u == 0
    Huv[1,1]=  2*lam+pi
    Huv[1,2]= -2*clam
    Iuv[1,1]=  2*phi+pi
    Iuv[1,2]= -2*cphi
    slamn = slam
    sphin = sphi
    v=2
    while v <= l_max
      Huv[1,v+1]= (-2*clam*slamn+(v-1)*Huv[1,v-1])/(u+v)
      Iuv[1,v+1]= (-2*cphi*sphin+(v-1)*Iuv[1,v-1])/(u+v)
      slamn *= slam
      sphin *= sphi
      v+=1
    end
  else
    slamn = slam
    sphin = sphi
    v = 0
    while v <= l_max
      Huv[u+1,v+1]= (2*clamn*slamn+(u-1)*Huv[u-1,v+1])/(u+v)
      Iuv[u+1,v+1]= (2*cphin*sphin+(u-1)*Iuv[u-1,v+1])/(u+v)
      slamn *= slam
      sphin *= sphi
      v+=1
    end
    clamn *= clam2
    cphin *= cphi2
  end
end
#println("Huv: ",Huv)
#println("Iuv: ",Iuv)

# Next, compute Juv:
Juv = zeros(typeof(r),l_max+3,l_max+1)
# p and q go up to u+2v = 3*l_max+2
if b == 0.0
  Juv .= (1.0-r^2)^1.5*Iuv
else
  mpq = mpq_of(k2,3*l_max+2,3*l_max+2,Kofk,Eofk,true)
  #println("M_pq: ",mpq)
  factor = 8*(b*r)^1.5
  for u=0:2:l_max+2
    for v=0:l_max, i=0:v
      Juv[u+1,v+1] += factor * binomial(v,i)*(-1)^(i-v-u)*mpq[u+2*i+1,u+2*(v-i)+1]
    end
    factor *= 4
  end
end
#println("Juv: ",Juv)
# Next, compute the K and L functions:
Kuv = zeros(typeof(r),l_max+3,l_max+1)
Luv = zeros(typeof(r),l_max+3,l_max+1)
bonr = b/r
for u=0:2:l_max+2
  for v=0:l_max
    for i=0:v
      fac = binomial(v,i)*bonr^(v-i)
      Kuv[u+1,v+1] += fac*Iuv[u+1,i+1]
      Luv[u+1,v+1] += fac*Juv[u+1,i+1]
    end
  end
end
#println("Kuv: ",Kuv)
#println("Luv: ",Luv)

l = 0; n = 0; m = 0; pofgn = zero(typeof(r)); qofgn = zero(typeof(r))
while n <= n_max
  if n == 2
    sn[n+1] = s2(b,r,Kofk,Eofk)
#    println("l: ",l," m: ",m," mu: ",l-m," nu: ",l+m," n: ",n," s_n: ",sn[n+1])
  else
    mu = l-m; nu = l+m
    pofgn = zero(typeof(r)); qofgn = zero(typeof(r))
    # Equation for P(G_n) and Q(G_n):
    if iseven(nu)
      i1 = convert(Int64,mu/2)+2+1; i2 = convert(Int64,nu/2)+1
      pofgn = r^(l+2)*Kuv[i1,i2]
      qofgn = Huv[i1,i2]
    elseif iseven(l) && mu == 1
      pofgn = -r^(l-1)*Juv[l-2+1,2]
    elseif mu == 1
      pofgn = -r^(l-2)*(b*Juv[l-3+1,2]+r*Juv[l-3+1,3])
    else
      pofgn = r^(l-1)*Luv[convert(Int64,(mu-1)/2)+1,convert(Int64,(nu-1)/2)+1]
    end
    sn[n+1] = qofgn-pofgn
#    if n == 7 || n == 10 || n == 14
#      println("l: ",l," m: ",m," mu: ",mu," nu: ",nu," n: ",n," s_n: ",sn[n+1]," Q: ",qofgn," P: ",pofgn)
#    end
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
