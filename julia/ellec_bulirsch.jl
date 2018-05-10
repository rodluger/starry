function ellec_bulirsch(k2::Real)
ca = sqrt(eps(k2))
if k2 == 1.0
  return 1.0
end
m=1.0
a=1.0
b=1.0-k2
kc=sqrt(b)
c=a 
a+=b

b=2.0*(c*kc+b)
c=a
m0=m 
m+=kc
a+=b/m
while abs(m0-kc) > (ca*m0)
  kc=2.0*sqrt(kc*m0)
  b=2.0*(c*kc+b)
  c=a
  m0=m 
  m+=kc
  a+=b/m
#  println(a,' ',b,' ',c,' ',m0,' ',kc)
end
return pi*0.25*a/m
end
