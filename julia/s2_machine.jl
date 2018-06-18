# Uses new formulation from limbdark paper.

include("s2_stable.jl")

s2diff(b,r) = s2(r,b)-convert(Float64,s2(big(r),big(b)))

# Run some tests:

using PyPlot

nb = 1024; nr = 1024
fig,axes = subplots(3,2)
# Compute r = b+-\epsilon
epsilon = 1e-8
fepsp= zeros(Float64,nb)
fepsbigp= zeros(Float64,nb)
fepsm= zeros(Float64,nb)
fepsbigm= zeros(Float64,nb)
b=linspace(epsilon,2.0,nr)
for ib=1:nb
  fepsbigp[ib] = float(s2(big(b[ib]+epsilon),big(b[ib])))
  fepsp[ib] = float(s2(b[ib]+epsilon,b[ib]))
  fepsbigm[ib] = float(s2(big(b[ib]-epsilon),big(b[ib])))
  fepsm[ib] = float(s2(b[ib]-epsilon,b[ib]))
end
ax = axes[1]
ax[:plot](b,fepsp,label=L"$s_2(b,b+10^{-8})$")
ax[:plot](b,fepsm,label=L"$s_2(b,b-10^{-8})$")
ax[:plot](b,s2.(b,b),label=L"$s_2(b,b)$")
ax[:legend](loc="upper right",fontsize=10)
ax = axes[2]
ax[:plot](b,(fepsp-s2.(b,b))/epsilon,label=L"$s_2(b,b+10^{-8})-s_2(b,b)$")
ax[:plot](b,(fepsbigp-s2.(b,b))/epsilon,linewidth=2,linestyle="--",label=L"$s_2(b,b+10^{-8})-s_2(b,b)$, big")
ax[:plot](b,(fepsm-s2.(b,b))/epsilon,label=L"$s_2(b,b-10^{-8})-s_2(b,b)$")
ax[:plot](b,(fepsbigm-s2.(b,b))/epsilon,linewidth=2,linestyle="--",label=L"$s_2(b,b-10^{-8})-s_2(b,b)$, big")
ax[:legend](loc="right",fontsize=5)
ax = axes[3]
ax[:plot](b,s2diff.(b,b+epsilon),label=L"$s_2(b,b+10^{-8})-$ big")
ax[:plot](b,s2diff.(b,b-epsilon),linewidth=2,linestyle="--",label=L"$s_2(b,b-10^{-8})-$ big")
#ax[:plot](b,fepsp-fepsbigp,label=L"$s_2(b,b+10^{-8})-$ big")
#ax[:plot](b,fepsm-fepsbigm,linewidth=2,linestyle="--",label=L"$s_2(b,b-10^{-8})-$ big")
ax[:legend](loc="lower right",fontsize=6)

# Compute b+r = 1+-\epsilon
epsilon = 1e-8
b=linspace(0.0,2.0,nr)
for ib=1:nr
  fepsbigp[ib] = float(s2(big(1.0-b[ib]+epsilon),big(b[ib])))
  fepsp[ib] = float(s2(1.0-b[ib]+epsilon,b[ib]))
  fepsbigm[ib] = float(s2(big(1.0-b[ib]-epsilon),big(b[ib])))
  fepsm[ib] = float(s2(1.0-b[ib]-epsilon,b[ib]))
end
ax = axes[4]
ax[:plot](b,fepsp,label=L"$s_2(b,1-b+10^{-8})$")
ax[:plot](b,fepsm,label=L"$s_2(b,1-b-10^{-8})$")
ax[:plot](b,s2.(1.0-b,b),label=L"$s_2(b,1-b)$")
ax[:legend](loc="lower right",fontsize=10)
ax = axes[5]
ax[:plot](b,(fepsp-s2.(1.0-b,b))/epsilon,label=L"$s_2(b,1-b+10^{-8})-s_2(b,1-b)$")
ax[:plot](b,(fepsbigp-s2.(1.0-b,b))/epsilon,linewidth=2,linestyle="--",label=L"$s_2(b,1-b+10^{-8})-s_2(b,1-b)$, big")
ax[:plot](b,(fepsm-s2.(1.0-b,b))/epsilon,label=L"$s_2(b,1-b-10^{-8})-s_2(b,1-b)$")
ax[:plot](b,(fepsbigm-s2.(1.0-b,b))/epsilon,linewidth=2,linestyle="--",linestyle="--",label=L"$s_2(b,1-b-10^{-8})-s_2(b,1-b)$, big")
ax[:legend](loc="lower right",fontsize=5)
ax = axes[6]
ax[:plot](b,s2diff.(1.0-b+epsilon,b),label=L"$s_2(b,1-b+10^{-8})-$ big")
ax[:plot](b,s2diff.(1.0-b-epsilon,b),linewidth=2,linestyle="--",label=L"$s_2(b,1-b+10^{-8})-$ big")
ax[:legend](loc="upper right",fontsize=6)
savefig("s2_machine.pdf", bbox_inches="tight")
