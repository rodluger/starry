include("sn.jl")
include("sn_bigr.jl")

#r=10.
#r=100.
#r=100.
r=0.1
l_max = 20
n_max = l_max^2+2*l_max
sn_big = zeros(BigFloat,n_max+1);
snew = zeros(typeof(r),n_max+1);
snew_big = zeros(BigFloat,n_max+1);
#b=sqrt((r+.5)^2+0.1^2)
#b=1e-8
b=0.55
#b=100.
for i=1:length(b)
  s_n_bigr!(l_max,r,b[i],snew)
  s_n_bigr!(l_max,big(r),big(b[i]),snew_big)
  s_n!(l_max,big(r),big(b[i]),sn_big)
  l = 0; n = 0; m = 0
  for n=0:n_max
    mu = l-m; nu = l+m
    if snew[n+1] != 0.0
      println("n: ",n," l: ",l," m: ",m," mu: ",mu," nu: ",nu," s: ",snew[n+1],
#       " s_old: ",convert(Float64,sn_big[n+1])," d_old: ",snew[n+1]/convert(Float64,sn_big[n+1])-1.,
       " s_big: ",convert(Float64,snew_big[n+1])," d_old: ",snew[n+1]/convert(Float64,sn_big[n+1])-1.,
       " d_big: ",snew[n+1]/convert(Float64,snew_big[n+1])-1.)
    end
    m +=1
    if m > l
      l += 1
      m = -l
    end
  end
end
