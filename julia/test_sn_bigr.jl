include("sn.jl")
include("sn_bigr.jl")

#r=10.
r=100.
#r=110.
#r=1.1
#r=0.01
l_max = 20
n_max = l_max^2+2*l_max
sn_big = zeros(BigFloat,n_max+1);
snew = zeros(typeof(r),n_max+1);
snew_big = zeros(BigFloat,n_max+1);
#b=sqrt((r+.5)^2+0.1^2)
#b=1e-3
#b=0.99
#b=0.9
#b=0.55
b = 1+r-1e-8
#b = 1-r+1e-8
#b = 110.99
#b = 1.0
#b=100.
#b = r+1-1e-18
diff_frac = zeros(n_max+1)
diff_rel  = zeros(n_max+1)
for i=1:length(b)
  s_n_bigr!(l_max,r,b[i],snew)
  s_n_bigr!(l_max,big(r),big(b[i]),snew_big)
  s_n!(l_max,big(r),big(b[i]),sn_big)
  l = 0; n = 0; m = 0
  for n=0:n_max
    mu = l-m; nu = l+m
    if snew[n+1] != 0.0
      diff = abs(snew[n+1]/convert(Float64,snew_big[n+1])-1.)
      diff_frac[n+1]=diff
      diff_rel[n+1]=abs(snew[n+1]-convert(Float64,snew_big[n+1]))
#      if diff > 0.01
      println("n: ",n," l: ",l," m: ",m," mu: ",mu," nu: ",nu," s: ",snew[n+1],
#       " s_old: ",convert(Float64,sn_big[n+1])," d_old: ",snew[n+1]/convert(Float64,sn_big[n+1])-1.,
       " s_old: ",convert(Float64,sn_big[n+1]),
       " s_big: ",convert(Float64,snew_big[n+1])," d_old: ",snew[n+1]/convert(Float64,sn_big[n+1])-1.,
       " d_big: ",snew[n+1]/convert(Float64,snew_big[n+1])-1.)
#      end
    end
    m +=1
    if m > l
      l += 1
      m = -l
    end
  end
end

using PyPlot
clf()
semilogy(diff_frac,"o",label="fractional",c="blue")
semilogy(diff_rel,"o",label="relative",c="orange")
legend(loc="upper left")
xlabel("n")
ylabel("Error")
axis([-5,n_max,1e-25,1e-2])
