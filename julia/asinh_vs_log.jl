using PyPlot
x=logspace(-2,2,1000)
clf()
plot(x,log.(2x),label=L"$f(x)=\log{(2x)}$")
loglog(x,asinh.(x),label=L"$f(x)=\sinh^{-1}{x}$")
loglog(x,x,label=L"$f(x)=x$")
legend(loc="upper left")
ylabel(L"$f(x)$")
xlabel(L"$x$")
