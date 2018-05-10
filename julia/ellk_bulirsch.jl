function ellk_bulirsch(k2::Real)
ca = sqrt(eps(k2))
kc=sqrt(1.0-k2)
h=1.0
m=1.0
while abs(h-kc) > (ca*h)
  h  = m 
  m += kc
  kc = sqrt(h*kc) 
  m  = 0.5*m
#  println(h,' ',m,' ',kc,' ',h-kc)
end
h=m 
m+=kc
return pi/m
end
