# Computes the M_{p,q} coefficients defined in (D37) for
# the r0 >> 1 limit in which k2 << 1 and r0-1 < b < r0+1
# (this is a huge occultor).

using PyPlot

function eps3_of(k2::T,Kofk::T,Eofk::T) where {T <: Real}
eps3 = Kofk-Eofk-0.5*k2*Kofk
return eps3
end

function eps3_series(k2::Real)
# Series evaluation of eps3 for k2 < 1:
@assert(k2 < 1)
tol = eps(k2)
eps3 = zero(k2)
# Compute n=1 term
term = pi/32
eps3 += term
n=2
while abs(term) > tol*abs(eps3)
  term *= (2*n-1)^2*k2/4/(n^2-1)
  eps3 += term
  n +=1
end
return eps3*k2*k2
end

function mpq_init(k2::Real)
# Initialize the M_{p,q} computation:
k4 = k2*k2
m001 = (8.-16.*k2)/3.  # eps3 term
m002 = 4k4/3  # K(m) term
m021 = (8.-28k2-12k4)/15.
m022 = (22.-6k2)*k4/15
m201 = (32.-52k2+12k4)/15.
m202 = (-2.+6k2)*k4/15
m221 = (32.-76k2+36k4-24k4*k2)/105.
m222 = (-2.+30k2-12k4)*k4/105.
return m001,m002,m021,m022,m201,m202,m221,m222
end

function mpq_of(k2::T,pmax::Int64,qmax::Int64,Kofk::T,Eofk::T,alt::Bool) where {T <: Real}
mpq = zeros(typeof(k2),pmax+1,qmax+1)
if k2 < 1 && alt
  mpq1 = zeros(typeof(k2),pmax+1,qmax+1)
  mpq2 = zeros(typeof(k2),pmax+1,qmax+1)
  m001 = 0.; m021=0.; m201=0.; m221=0.
  m002 = 0.; m022=0.; m202=0.; m222=0.
  eps3 = zero(typeof(k2))
# Alternate form:
  if k2 < 0.1
    eps3 = eps3_series(k2)
  else
    eps3=eps3_of(k2,Kofk,Eofk)
  end
  m001,m002,m021,m022,m201,m202,m221,m222 = mpq_init(k2)
  mpq[1,1]=m001*eps3+m002*Kofk; mpq[1,3]=m021*eps3+m022*Kofk
  mpq[3,1]=m201*eps3+m202*Kofk; mpq[3,3]=m221*eps3+m222*Kofk
#  println("m221: ",m221," m222: ",m222," eps3: ",eps3," K(m): ",Kofk)
else 
  k4  = k2^2
  if k2 < 1
    eps1 = (1-k2)*Kofk; eps2 = Eofk
  elseif k2 > 1
  # In this case, no special handling is required as there
  # is not cancellation as there is in k2 < 1 case.
  # However, for k2 >> 1, are the recursion relations stable?
    k2inv = 1.0/k2; k = sqrt(k2)
    eps1 = (1.0-k2)/k*Kofk; eps2 = k*Eofk+eps1
  else
    eps1 = zero(k2); eps2= one(k2)
  end
  mpq[1,1] = (8-12k2)*eps1/3+(-8+16*k2)*eps2/3
  mpq[1,3] = (8-24k2)*eps1/15+(-8+28*k2+12*k4)*eps2/15
  mpq[3,1] = (32-36k2)*eps1/15+(-32+52*k2-12k4)*eps2/15
  mpq[3,3] = (32-60k2+12k4)*eps1/105+(-32+76k2-36k4+24k4*k2)*eps2/105
end
#println("p,q: ",0," ",0," M_{pq}: ",mpq[1,1])
#println("p,q: ",0," ",2," M_{pq}: ",mpq[1,3])
#println("p,q: ",2," ",0," M_{pq}: ",mpq[3,1])
#println("p,q: ",2," ",2," M_{pq}: ",mpq[3,3])

# Now, use recursion relation to compute M_{p,q}:
for p=0:2:pmax
  if p > 2
    for q = 0:2:2
      d3 = 2p+q-(p+q-2)*(1.-k2)
      d4 = (3-p)*k2
      mpq[p+1,q+1] = (d3*mpq[p-1,q+1]+d4*mpq[p-3,q+1])/(p+q+3)
#      println("p,q: ",p," ",q," M_{pq}: ",mpq[p+1,q+1])
    end
  end
  for q=4:2:qmax
    d1 = q+2+(p+q-2)*(1.-k2)
    d2 = (3-q)*(1.-k2)
    mpq[p+1,q+1]=(d1*mpq[p+1,q-1]+d2*mpq[p+1,q-3])/(p+q+3)
#    println("p,q: ",p," ",q," M_{pq}: ",mpq[p+1,q+1])
  end
end
return mpq
end
