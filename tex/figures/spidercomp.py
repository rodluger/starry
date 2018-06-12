"""
Starry-SPIDERMAN comparisons and speed tests.
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


# Marker size is proportional to log diff
def ms(diff):
    return 18 + np.log10(diff)

# Main comparison function
def comparison():

    ### Define SPIDERMAN model parameters ###

    # Parameters adapted from https://github.com/tomlouden/SPIDERMAN/blob/master/examples/Brightness%20maps.ipynb
    spider_params = sp.ModelParams(brightness_model='spherical')

    spider_params.n_layers = 10        # Will be reset later

    spider_params.t0 = 0                # Central time of PRIMARY transit [days]
    spider_params.per = 0.81347753      # Period [days]
    spider_params.a_abs = 1.0e-30       # Nearly 0 a to ignore light travel effects
    spider_params.inc = 90.0            # Inclination [degrees]
    spider_params.ecc = 0.0             # Eccentricity
    spider_params.w = 0.0               # Argument of periastron
    spider_params.rp = 0.1594           # Planet to star radius ratio
    spider_params.a = 4.855             # Semi-major axis scaled by stellar radius
    spider_params.p_u1 = 0.0            # Planetary limb darkening parameter
    spider_params.p_u2 = 0.0            # Planetary limb darkening parameter

    # SPIDERMAN spherical harmonics parameters
    ratio = 1.0e-3                                  # Planet-star flux ratio
    spider_params.sph = [ratio, ratio/2, 0, 0]      # vector of spherical harmonic weights
    spider_params.degree = 2
    spider_params.la0 = 0.0
    spider_params.lo0 = 0.0

    ### Define starry model parameters to match SPIDERMAN system ###

    # Define star
    star = starry.Star()

    # Define planet
    planet = starry.Planet(lmax=2,
                           lambda0=90.0,
                           w=spider_params.w,
                           r=spider_params.rp,
                           L=1.0e-3 * np.pi, # Factor of pi to match SPIDERMAN normalization
                           inc=spider_params.inc,
                           a=spider_params.a,
                           porb=spider_params.per,
                           tref=spider_params.t0,
                           prot=spider_params.per,
                           ecc=spider_params.ecc)

    # Define spherical harmonic coefficients
    planet.map[0,0] = 1.0
    planet.map[1,-1] = 0.0
    planet.map[1,0] = 0.0
    planet.map[1,1] = 0.5

    # Make a system
    system = starry.System([star,planet])

    ### Speed test! ###

    # Number of time array points
    ns = np.array([20, 100, 500, 1000], dtype=int)

    # SPIDERMAN grid resolution points
    ngrid = np.array([5, 10, 20, 50, 100], dtype=int)

    n_repeats = 3
    t_starry = np.nan + np.zeros(len(ns))
    t_spiderman = np.nan + np.zeros((len(ns),len(ngrid)))
    diff = np.nan + np.zeros_like(t_spiderman)
    flux_comp = []

    # Loop over time grid sizes
    for ii, n in enumerate(ns):

        # New time array of length n just around the occulation and some phase curve
        time_arr = np.linspace(0.4*spider_params.per, 0.6*spider_params.per, n)

        # Repeat calculation a few times and pick fastest one
        best_starry = np.inf
        for _ in range(n_repeats):

            start = time.time()
            system.compute(time_arr)
            flux = np.array(system.flux)
            dt = time.time() - start

            if dt < best_starry:
                best_starry = dt
                best_starry_flux = flux

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
                    best_spiderman_flux = lc

            # Save fastest time
            t_spiderman[ii,jj] = best_spiderman

            # Save log maximum relative error
            diff[ii,jj] = np.max(np.fabs((best_starry_flux - best_spiderman_flux)/best_starry_flux))

            # For highest time resolution, compute differences between the predictions
            if n == ns[-1]:
                flux_comp.append(np.fabs(best_starry_flux - best_spiderman_flux))

    ### Generate the figures! ###

    # First figure: relative error

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot2grid((3, 1), (0, 0), colspan=1, rowspan=1)
    ax2 = plt.subplot2grid((3, 1), (1, 0), colspan=1, rowspan=2)

    time_arr = np.linspace(0.4*spider_params.per, 0.6*spider_params.per, ns[-1])

    # Flux panel
    ax.plot(time_arr, best_starry_flux)
    ax.get_xaxis().set_ticklabels([])
    ax.set_xlim(time_arr.min(), time_arr.max())
    ax.set_ylabel("Flux")

    # Error panel
    for kk in range(len(flux_comp)):
           ax2.plot(time_arr, flux_comp[kk],
                   label="n$_{\mathrm{layers}}$=%d" % ngrid[kk])

    ax2.set_ylim(1.0e-11, 1.0e-4)
    ax2.set_xlim(time_arr.min(), time_arr.max())
    ax2.set_yscale("log")
    ax2.set_ylabel("Relative error")
    ax2.set_xlabel("Time [d]")
    ax2.legend(loc="best", framealpha=0.0)
    fig.savefig("spidercomp_flux.pdf", bbox_inches="tight")

    # Second figure: speed comparison

    fig = plt.figure(figsize=(7, 4))
    ax = plt.subplot2grid((2, 5), (0, 0), colspan=4, rowspan=2)
    axleg1 = plt.subplot2grid((2, 5), (0, 4))
    axleg2 = plt.subplot2grid((2, 5), (1, 4))
    axleg1.axis('off')
    axleg2.axis('off')
    ax.set_xlabel('Number of points', fontsize=14, fontweight='bold')
    for tick in ax.get_xticklabels():
        tick.set_fontsize(12)
    ax.set_ylabel('Evaluation time [seconds]', fontsize=14, fontweight='bold')
    ax.set_yscale("log")
    ax.set_xscale("log")
    for tick in ax.get_yticklabels():
        tick.set_fontsize(12)

    # Starry, loop over all points
    for ii in range(len(ns)):
        ax.plot(ns[ii], t_starry[ii], "o", lw=2, color="C0", ms=ms(1.0e-16))
    ax.plot(ns, t_starry, "-", lw=1.5, color="C0", alpha=0.25)

    # Loop over all grid resolutions
    for jj, ng in enumerate(ngrid):
        ax.plot(ns, t_spiderman[:,jj], "-", color="C%d" % (jj+1), alpha=0.25, lw=1.5)
        for kk in range(len(ns)):
            ax.plot(ns[kk], t_spiderman[kk,jj], "o", ms=ms(diff[kk,jj]),
                    color="C%d" % (jj+1))

    # Legend 1
    axleg1.plot([0, 1], [0, 1], color='C0', label='starry')
    # Loop over all grid resolutions
    for jj, ng in enumerate(ngrid):
        axleg1.plot([0, 1], [0, 1], color="C%d" % (jj+1), label="n$_{\mathrm{layers}}$=%d" % ng)
    axleg1.set_xlim(2, 3)
    axleg1.legend(loc='center', frameon=False, title=r'\textbf{method}')

    for logerr in [-16, -12, -8, -4, 0]:
        axleg2.plot([0, 1], [0, 1], 'o', color='gray',
                    ms=ms(10 ** logerr),
                    label=r'$%3d$' % logerr)
    axleg2.set_xlim(2, 3)
    leg = axleg2.legend(loc='center', labelspacing=1, frameon=False,
                        title=r'\textbf{log error}')

    fig.savefig("spidercomp.pdf", bbox_inches="tight")
# Done!

# Run it!
if __name__ == "__main__":
    comparison()
