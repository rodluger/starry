# Tests automatic differentiation on transit_poly.jl:
include("transit_poly_gradient.jl")
using PyPlot

function test_transit_poly_gradient(u_n)
#r0 = [0.01,100.0]; n_u = 5; u_n = ones(n_u)/n_u
r0 = [0.01,100.0]; n_u = length(u_n)
nb = 50

# Now, carry out finite-difference derivative computation:
function transit_poly_grad_num(r::T,b::T,u_n::Array{T,1}) where {T <: Real}
  dq = big(1e-18)
# Make BigFloat versions of r, b & u_n:
  r_big = big(r); b_big = big(b); u_big = big.(u_n)
# Compute flux to BigFloat precision:
  tp=transit_poly(r_big,b_big,u_big)
  println("r: ",r," b: ",b," tp error: ",convert(Float64,tp) - transit_poly(r,b,u_n))
# Now, compute finite differences:
  tp_grad_big= zeros(BigFloat,2+length(u_n))
  tp_plus = transit_poly(r_big+dq,b_big,u_big)
  tp_minus = transit_poly(r_big-dq,b_big,u_big)
  tp_grad_big[1] = (tp_plus-tp_minus)*.5/dq
  tp_plus = transit_poly(r_big,b_big+dq,u_big)
  tp_minus = transit_poly(r_big,b_big-dq,u_big)
  tp_grad_big[2] = (tp_plus-tp_minus)*.5/dq
  for i=1:length(u_n)
    u_tmp = copy(u_big); u_tmp[i] += dq
    tp_plus = transit_poly(r_big,b_big,u_tmp)
    u_tmp[i] -= 2dq
    tp_minus = transit_poly(r_big,b_big,u_tmp)
    tp_grad_big[2+i] = (tp_plus-tp_minus)*.5/dq
  end
return convert(Float64,tp),convert(Array{Float64,1},tp_grad_big)
end

epsilon = 1e-12; delta = 1e-3
for i=1:2
  fig,axes = subplots(1,1)
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
  tp_grad_grid = zeros(length(b),n_u+2)
  tp_grad_array= zeros(n_u+2)
  tp_grid = zeros(length(b))
  tp_grid_big = zeros(length(b))
  tp_grad_grid_num = zeros(length(b),n_u+2)
  for j=1:length(b)
    println("r: ",r," b: ",b[j])
    tp,tp_grad_array= transit_poly_grad(r,b[j],u_n)
    transit_poly(r,b[j],u_n)
    tp_grid[j,:]=tp
    tp_grad_grid[j,:,:]=tp_grad_array
    # Now compute with BigFloat finite difference:
    tp,tp_grad_array =  transit_poly_grad_num(r,b[j],u_n)
    tp_grad_grid_num[j,:,:]=tp_grad_array
    tp_grid_big[j,:]=tp
  end
# Now, make plots:
  ax = axes
  ax[:semilogy](abs.(asinh.(tp_grid)-asinh.(tp_grid_big)),lw=1)
  for n=1:n_u+2
#    ax[:semilogy](abs.(asinh.(tp_grad_grid[:,n])-asinh.(tp_grad_grid_num[:,n])),lw=1)
  end
  ax[:legend](loc="upper right",fontsize=6)
  ax[:set_xlabel]("b values")
  ax[:set_ylabel]("Derivative Error")
  ax[:axis]([0,length(b),1e-16,1])
  ax[:set_xticks](nb*linspace(0,nticks-1,nticks))
  ax[:set_xticklabels](xticknames,rotation=45)
  if i==1 
    ax[:set_title]("r = 0.01")
  else
    ax[:set_title]("r = 100")
  end
  read(STDIN,Char)

### Loop over n and see where the differences between the finite-difference
### and AutoDiff are greater than the derivative value: 
#l=0; m=0
#for n=0:n_max
#  diff = sn_jac_grid[:,n+1,:]-sn_jac_grid_num[:,n+1,:]
#  mask = (abs.(diff) .> 1e-3*abs.(sn_jac_grid[:,n+1,:])) .& (abs.(sn_jac_grid[:,n+1,:]) .> 1e-5)
#  if sum(mask) > 0 || mod(l-m,4) == 0
#    println("n: ",n," max dsn/dr: ",maximum(abs.(diff)))
#    println("b: ",b[mask[:,1]],b[mask[:,2]]," k^2_H: ",k2H[mask[:,1]],k2H[mask[:,2]])
#    clf()
#    plot(b,sn_grid[:,n+1])
#    plot(b,sn_jac_grid[:,n+1,1])
#    plot(b[mask[:,1]],sn_jac_grid[mask[:,1],n+1,1],"o")
#    plot(b,sn_jac_grid[:,n+1,2])
#    plot(b[mask[:,2]],sn_jac_grid[mask[:,2],n+1,2],"o")
#    plot(b,sn_jac_grid_num[:,n+1,1],linestyle="--",linewidth=2)
#    plot(b,sn_jac_grid_num[:,n+1,2],linestyle="--",linewidth=2)
#    println("n: ",n," l: ",l," m: ",m," mu: ",l-m," nu: ",l+m)
#    read(STDIN,Char)
#  end
#  m +=1
#  if m > l
#    l += 1
#    m = -l
#  end
#end
##
##
###end
##
###loglog(abs.(reshape(sn_jac_grid,length(b)*(n_max+1)*2)),abs.(reshape(sn_jac_grid-sn_jac_grid_num,length(b)*(n_max+1)*2)),".")
###loglog(abs.(reshape(sn_jac_grid,length(b)*(n_max+1)*2)),abs.(reshape(sn_jac_grid_num,length(b)*(n_max+1)*2)),".")
end
return
end
