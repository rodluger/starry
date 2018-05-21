"""
Starry SPIDERMAN comparison speed tests.
"""
import starry
import timeit
try:
    import builtins
except ImportError:
    import __builtin__ as builtins

import matplotlib.pyplot as plt
import numpy as np
import spiderman as sp
import time


def comparison():

    ### Define SPIDERMAN model parameters ###

    # Parameters adapted from https://github.com/tomlouden/SPIDERMAN/blob/master/examples/Brightness%20maps.ipynb
    spider_params = sp.ModelParams(brightness_model='spherical')

    spider_params.n_layers = 100 # Will be reset later

    spider_params.t0 = 0               # Central time of PRIMARY transit [days]
    # Large orbital and rotational period so planet is effectively fixed
    spider_params.per = 1000.81347753       # Period [days]
    spider_params.a_abs = 0.01526        # The absolute value of the semi-major axis [AU]
    spider_params.inc = 90            # Inclination [degrees]
    spider_params.ecc = 0.0              # Eccentricity
    spider_params.w = 0                 # Argument of periastron
    spider_params.rp = 0.1594            # Planet to star radius ratio
    spider_params.a = 4.855              # Semi-major axis scaled by stellar radius
    spider_params.p_u1 = 0               # Planetary limb darkening parameter
    spider_params.p_u2 = 0               # Planetary limb darkening parameter

    # SPIDERMAN spherical harmonic parameters
    ratio = 1.0e-3
    spider_params.sph = [ratio, 0, 0, ratio/2]      # vector of spherical harmonic weights
    spider_params.degree = 2
    spider_params.la0 = 0.0
    spider_params.lo0 = 0.0

    ### Define starry model parameters to match SPIDERMAN system ###

    # Define star
    star = starry.Star()

    # Define planet with lambda0=90 (start a middle of primary transit)
    planet = starry.Planet(lmax=2,
                           lambda0=90.0,
                           theta0=180.0,
                           w=spider_params.w,
                           r=spider_params.rp,
                           L=1.0e-3 * np.pi,
                           inc=spider_params.inc,
                           a=spider_params.a,
                           porb=spider_params.per,
                           tref=spider_params.t0,
                           prot=spider_params.per,
                           ecc=0.0)

    # Define spherical harmonic coefficients
    planet.map[0,0] = 1.0
    planet.map[1,-1] = 0.0
    planet.map[1,0] = 0.5
    planet.map[1,1] = 0.0

    # Make a system
    system = starry.System([star,planet])

    ### Speed test! ###

    # Number of time array points
    ns = np.array([10, 50, 100, 200, 500, 1000], dtype=int)

    # SPIDERMAN grid resolution points
    ngrid = np.array([5, 10, 20, 50, 100], dtype=int)

    n_repeats = 3
    t_starry = np.nan + np.zeros(len(ns))
    t_spiderman = np.nan + np.zeros((len(ns),len(ngrid)))

    for ii, n in enumerate(ns):
        print("Current time grid size: %d." % n)

        # New time array of length n
        time_arr = np.linspace(0, spider_params.per, n)

        # Repeat calculation a few times and pick best one
        best_starry = np.inf
        for _ in range(n_repeats):

            start = time.time()
            system.compute(time_arr)
            flux = np.array(system.flux)
            dt = time.time() - start

            if dt < best_starry:
                best_starry = dt

        # Save fastest time
        t_starry[ii] = best_starry

        # Time batman (for all grid resolutions)
        for jj, ng in enumerate(ngrid):

            # New number of layers
            spider_params.n_layers = ng

            # Repeat calculation a few times
            best_spiderman = np.inf
            for _ in range(n_repeats):

                start = time.time()
                lc = spider_params.lightcurve(time_arr, use_phase=False)
                dt = time.time() - start

                if dt < best_spiderman:
                    best_spiderman = dt

            # Save fastest time
            t_spiderman[ii,jj] = best_spiderman

    ### Generate the figure! ###

    fig, ax = plt.subplots(figsize=(9,8))

    ax.plot(ns, t_starry, "o-", lw=3, label="starry")

    # Loop over all grid resolutions
    for jj, ng in enumerate(ngrid):
        ax.plot(ns, t_spiderman[:,jj], "o-", lw=3, label="SPIDERMAN nlayers=%d" % ng)

    # Format
    ax.set_xlabel("Number of points")
    ax.set_ylabel("Time [s]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="upper left")

    fig.savefig("spidercomp.png", bbox_inches="tight")
# Done!

# Run it!
if __name__ == "__main__":
    print("starry version:",starry.__version__)
    comparison()
