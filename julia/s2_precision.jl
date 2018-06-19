using PyPlot
include("/Users/ericagol/Students/Luger/Starry/starry/julia/s2_stable.jl")

r0 = [0.01,100.]
nb = 1000
fig,axes = subplots(2,2)

for i=1:2
  r=r0[i]
  b = [linspace(r-1+1e-8,r-1+1e-6,nb); linspace(r-1+1e-6,r-1+1e-3,nb); linspace(r-1+1e-3,r-1e-3,nb); linspace(r-1e-3,r-1e-6,nb); linspace(r-1e-6,r+1e-6,nb); linspace(r+1e-6,r+1e-3,nb);  linspace(r+1e-3,r+1-1e-3,nb); linspace(r+1-1e-3,r+1-1e-6,nb); linspace(r+1-1e-6,r+1-1e-8,nb)]

  s2_grid = s2.(r,b)
  s2_big = convert(Array{Float64,1},s2.(big(r),big.(b)))
  diff = s2_grid-s2_big
  ax[:plot](b,diff)
  ax[:xlabel]("b")
  ax[:ylabel]("s2-s2(big)")
return
end
