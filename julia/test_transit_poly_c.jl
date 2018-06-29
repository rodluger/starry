include("transit_poly.jl")
nc = 2+ceil(Int64,rand()*20)
r=rand(); b=rand()*(1+r); c = rand(nc)
#r=1.2*rand(); b=0.0; c = rand(nc)
#r = rand(); b = rand(); c = [1.0,1.0,10.0]
#r=0.7821623254303613; b= 0.5; c = [0.0,1.0]

flux = transit_poly_c(r,b,c)
@time flux = transit_poly_c(r,b,c)

# Now, integrate by hand:
function transit_poly_int_c(r,b,c)
s_1 = maximum([0.0,b-r]); s_2=minimum([1.0,b+r])
ns = 500; ds=(s_2-s_1)/ns
s = linspace(s_1+.5*ds,s_2-.5*ds,ns)
fobs = zero(r)
for j=1:ns
  if s[j] < r-b
    dphi = 2pi
  else
    dphi = 2*acos((s[j]^2+b^2-r^2)/(2*b*s[j]))
  end
  z = sqrt(1-s[j]^2)
  imu = c[1] + c[2]*z
  for n=2:length(c)-1
    imu += c[n+1]*(z^n*(n+2)-n*z^(n-2))
  end
  fobs += s[j]*ds*dphi*imu
end
fobs /= pi*(c[1]+2/3*c[2])
fobs = 1-fobs
#println("r: ",r," b: ",b,"f_an: ",flux," flux: ",flux0," f_num: ",fobs," f_exp: ",pi*(c[1]+2/3*c[2]))
return fobs
end
f_num = transit_poly_int_c(r,b,c)
@time f_num = transit_poly_int_c(r,b,c)
println("r: ",r," b: ",b," f_an: ",flux," f_num: ",f_num)

#nphi = 500; ns = 500; dphi = 2pi/nphi; ds=1/ns
#phi = linspace(.5*dphi,2pi-.5*dphi,nphi); s = linspace(.5*ds,1-.5*ds,ns)
#flux0 = zero(r); fobs = zero(r)
#for i=1:nphi, j=1:ns
#  x=s[j]*cos(phi[i]); y = s[j]*sin(phi[i])
#  z = sqrt(1-s[j]^2)
#  imu = c[1]+c[2]*z
#  for k=2:length(c)-1
#    imu += c[k+1]*(z^k*(k+2)-k*z^(k-2))
#  end
#  flux0 += s[j]*ds*dphi*imu
#  if (y-b)^2+x^2 > r^2
#    fobs += s[j]*ds*dphi*imu
#  end
#end
##println("r: ",r," b: ",b,"f_an: ",flux," flux: ",flux0," f_num: ",fobs/flux0," f_exp: ",pi*(c[1]+2/3*c[2]))
#println("r: ",r," b: ",b," f_an: ",flux," f_num: ",fobs/flux0)
