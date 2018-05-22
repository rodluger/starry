"""
Starry SPIDERMAN phase curve comparison speed tests.
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


def phase_comparison():

    ### Define SPIDERMAN model parameters ###

    # Parameters adapted from https://github.com/tomlouden/SPIDERMAN/blob/master/examples/Brightness%20maps.ipynb
    spider_params = sp.ModelParams(brightness_model='spherical')

    spider_params.n_layers = 100 # Will be reset later

    spider_params.t0 = 0                # Central time of PRIMARY transit [days]
    spider_params.per = 0.81347753      # Period [days]
    spider_params.a_abs = 0.01526       # The absolute value of the semi-major axis [AU]
    spider_params.inc = 90.0             # Inclination [degrees]
    spider_params.ecc = 0.0             # Eccentricity
    spider_params.w = 0                 # Argument of periastron
    spider_params.rp = 0.1594           # Planet to star radius ratio
    spider_params.a = 4.855             # Semi-major axis scaled by stellar radius
    spider_params.p_u1 = 0              # Planetary limb darkening parameter
    spider_params.p_u2 = 0              # Planetary limb darkening parameter

    # SPIDERMAN spherical harmonics parameters
    ratio = 1.0e-3
    spider_params.sph = [ratio, 0, 0, ratio/2]
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
                           L=1.0e-3 * np.pi, # To match SPIDERMAN normalization
                           inc=spider_params.inc,
                           a=spider_params.a,
                           porb=spider_params.per,
                           tref=spider_params.t0,
                           prot=spider_params.per,
                           ecc=spider_params.ecc)

    # Define spherical harmonic coefficients
    planet.map[0,0] = 1.0
    planet.map[1,-1] = 0.0
    planet.map[1,0] = 0.5
    planet.map[1,1] = 0.0

    # Make a system
    system = starry.System([star,planet])

    ### Speed test! ###

    # Number of time array points
    ns = np.array([50, 100, 200, 500, 1000], dtype=int)

    # SPIDERMAN grid resolution points
    ngrid = np.array([5, 10, 20, 50, 100], dtype=int)

    n_repeats = 5
    t_starry = np.nan + np.zeros(len(ns))
    t_spiderman = np.nan + np.zeros((len(ns),len(ngrid)))
    diff = np.nan + np.zeros_like(t_spiderman)

    for ii, n in enumerate(ns):
        print("Current time grid size: %d." % n)

        # New time array of length n
        time_arr = np.linspace(0.1*spider_params.per, 0.9*spider_params.per, n)

        # Repeat calculation a few times and pick best one
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
            # Mask out transit, secondary eclipse for phasecurve comparison
            mask = (time_arr > 0.45*spider_params.per) & (time_arr < 0.55*spider_params.per)
            diff[ii,jj] = np.max(np.fabs((best_starry_flux[~mask] - best_spiderman_flux[~mask])/best_starry_flux[~mask]))

    ### Generate the figure! ###
    # Plot it
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

    # Starry
    ax.plot(ns, t_starry, "o", lw=2, color="C0")
    ax.plot(ns, t_starry, "-", lw=2, color="C0", alpha=0.25)

    # Loop over all grid resolutions
    for jj, ng in enumerate(ngrid):
        ax.plot(ns, t_spiderman[:,jj], "-", color="C%d" % (jj+1))
        for kk in range(len(ns)):
            ax.plot(ns[kk], t_spiderman[kk,jj], "o", ms=ms(diff[kk,jj]),
                    color="C%d" % (jj+1))

    # label="SPIDERMAN nlayers=%d" % ng

    # Legend 1
    axleg1.plot([0, 1], [0, 1], color='C0', label='starry')
    # Loop over all grid resolutions
    for jj, ng in enumerate(ngrid):
        axleg1.plot([0, 1], [0, 1], color="C%d" % (jj+1), label="Spiderman n$_{\mathrm{layers}}$=%d" % ng)
    axleg1.set_xlim(2, 3)
    axleg1.legend(loc='center', frameon=False, title=r'\textbf{method}')

    for logerr in [-16, -12, -8, -4, 0]:
        axleg2.plot([0, 1], [0, 1], 'o', color='gray',
                    ms=ms(10 ** logerr),
                    label=r'$%3d$' % logerr)
    axleg2.set_xlim(2, 3)
    leg = axleg2.legend(loc='center', labelspacing=1, frameon=False,
                        title=r'\textbf{log error}')

    fig.savefig("spiderphase.png", bbox_inches="tight")
# Done!

# Run it!
if __name__ == "__main__":
    phase_comparison()
