using PyPlot
include("/Users/ericagol/Students/Luger/Starry/starry/julia/s2_stable.jl")

r0 = [0.01,100.]
nb = 1000
fig,axes = subplots(1,2)

epsilon = 1e-12; delta = 1e-3
for i=1:2
#  r=r0[i]
#  ax = axes[i]
#  b = [linspace(r-1,r-1+epsilon,nb); linspace(r-1+epsilon,r-1+1e-3,nb); linspace(r-1+1e-3,r-1e-3,nb); linspace(r-1e-3,r-epsilon,nb); linspace(r-epsilon,r+epsilon,nb); linspace(r+epsilon,r+1e-3,nb);  linspace(r+1e-3,r+1-1e-3,nb); linspace(r+1-1e-3,r+1-epsilon,nb); linspace(r+1-epsilon,r+1,nb)]
  ax=axes[i]
  r=r0[i]
  if r < 1.0
    b = [linspace(1e-15,epsilon,nb); linspace(epsilon,delta,nb); linspace(delta,r-delta,nb);
     r-logspace(log10(delta),log10(epsilon),nb); linspace(r-epsilon,r+epsilon,nb); r+logspace(log10(epsilon),log10(delta),nb);
     linspace(r+delta,1-r-delta,nb); 1-r-logspace(log10(delta),log10(epsilon),nb); linspace(1-r-epsilon,1-r+epsilon,nb);
     1-r+logspace(log10(epsilon),log10(delta),nb); linspace(1-r+delta,1+r-delta,nb); 1+r-logspace(log10(delta),log10(epsilon),nb);linspace(1+r-epsilon,1+r-1e-15,nb)]
     nticks = 14
     xticknames=[L"$10^{-15}$",L"$10^{-12}$",L"$10^{-3}$",L"$r-10^{-3}$",L"$r-10^{-12}$",L"$r+10^{-12}$",L"$r+10^{-3}$",
     L"$1-r-10^{-3}$",L"$1-r-10^{-12}$",L"$1-r+10^{-12}$",L"$1-r+10^{-3}$",L"$1+r-10^{-3}$",L"$1+r-10^{-12}$",L"$1+r-10^{-15}$"]
  else
    b = [r-1+logspace(log10(epsilon),log10(delta),nb); linspace(r-1+delta,r-delta,nb);
     r-logspace(log10(delta),log10(epsilon),nb); linspace(r-epsilon,r+epsilon,nb); r+logspace(log10(epsilon),log10(delta),nb);
     linspace(r+delta,r+1-delta,nb); r+1-logspace(log10(delta),log10(epsilon),nb)]
     nticks = 8
     xticknames=[L"$r-1+10^{-12}$",L"$r-1+10^{-3}$",L"$r-10^{-3}$",L"$r-10^{-12}$",L"$r+10^{-12}$",L"$r+10^{-3}$",
     L"$r+1-10^{-3}$",L"$r+1-10^{-12}$"]
  end

  s2_grid = s2.(r,b)
  s2_big = convert(Array{Float64,1},s2.(big(r),big.(b)))
  diff = s2_grid-s2_big
#  ax[:plot](b,diff)
  ax[:plot](diff)
  ax[:set_xlabel]("b")
  ax[:set_ylabel]("s2-s2(big)")
#  ax[:axis]([0,length(b),1e-16,1])
  ax[:set_xticks](nb*linspace(0,nticks-1,nticks))
  ax[:set_xticklabels](xticknames,rotation=45)
  if i==1
    ax[:set_title]("r = 0.01")
  else
    ax[:set_title]("r = 100")
  end
end
