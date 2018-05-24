# Tests light-travel time delay: planets with
# semi-major axis as follows:
a = [0.232456517,1.162282585,2.32456517]
# with time offset between secondary eclipse
# and transit of:
dt = [0.5027,0.51337,0.52643]-0.5
# Convert to minutes:
dt *= 24.*60.
# for an orbital period of a day.

# Define some constants:

C  = 2.99792458e10
AU = 1.49598e13

# Expected time delay (for circular orbit) is 2a/c:
dt_mod = 2*a*AU/C/60.   # In minutes

using PyPlot
clf()
loglog(a,dt_mod,linewidth=3,label=L"$\Delta t = 2a/c$ ",c="r")
scatter(a,dt,s=50.,label=L"$\Delta t$ from starry")
legend(loc="upper left")
xlabel("Semi-major axis [AU]")
ylabel("Time delay [min]")
axis([.2,3,3.,40.])

println(dt - dt_mod)
