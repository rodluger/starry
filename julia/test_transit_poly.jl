include("transit_poly.jl")
nc = round(Int64,rand()*20)
r=rand(); b=rand()*(1+r); c = rand(nc)

flux = transit_poly(r,b,c)

# Now, integrate by hand:
nphi = 500; ns = 500; dphi = 2pi/nphi; ds=1/ns
phi = linspace(.5*dphi,2pi-.5*dphi,nphi); s = linspace(.5*ds,1-.5*ds,ns)
flux0 = zero(r); fobs = zero(r)
for i=1:nphi, j=1:ns
  x=s[j]*cos(phi[i]); y = s[j]*sin(phi[i])
  z = sqrt(1-s[j]^2)
  imu = c[1]+c[2]*z
  for k=2:length(c)-1
    imu += c[k+1]*(z^k*(k+2)-k*z^(k-2))
  end
  flux0 += s[j]*ds*dphi*imu
  if (y-b)^2+x^2 > r^2
    fobs += s[j]*ds*dphi*imu
  end
end
#println("r: ",r," b: ",b,"f_an: ",flux," flux: ",flux0," f_num: ",fobs/flux0," f_exp: ",pi*(c[1]+2/3*c[2]))
println("r: ",r," b: ",b," f_an: ",flux," f_num: ",fobs/flux0)
