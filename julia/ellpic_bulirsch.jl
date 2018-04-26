function ellpic_bulirsch(n::T,k2::T) where {T <: Real}
# Computes the complete elliptical integral of the third kind using
# the algorithm of Bulirsch (1965):
kc = sqrt(1.0-k2)
ca = sqrt(eps(n))
if (kc*(n+1.0)) == 0
  ellpic = NaN
else
  ee = kc
  m0 = 1.0
  if n > -1.0
    c = 1.0
    p = sqrt(n+1.0) 
    d = 1.0/p 
  else
    g = -n
    f = -k2-n
    p = sqrt(f/g)
    d = -k2/(g*p)
    c = 0.0
  end
  
  f = copy(c)
  c += d/p
  g = ee/p 
  d = (f*g+d)*2.0
  p += g
  g = m0 
  m0 += kc
  while abs(g-kc) > (ca*g)
    kc = 2.0*sqrt(ee)
    ee = kc*m0
    f = copy(c)
    c += d/p
    g = ee/p 
    d = (f*g+d)*2.0
    p += g
    g = m0 
    m0 += kc
  end
  ellpic= 0.5*pi*(c*m0+d)/(m0*(m0+p))
end
return ellpic
end
