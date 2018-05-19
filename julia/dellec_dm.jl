# Computes dE(m)/dm = 2(E(m)-K(m)):

include("ellec_bulirsch.jl")
include("ellk_bulirsch.jl")

using ForwardDiff

dellec_dm = m -> ForwardDiff.derivative(ellec_bulirsch,m)

m = linspace(0.0,1.0,1000)

using PyPlot

dell = zeros(1000)
for i=1:1000
  dell[i] = dellec_dm(m[i])
end
plot(m,dell)
plot(m,.5*(ellec_bulirsch.(m)-ellk_bulirsch.(m))./m)
