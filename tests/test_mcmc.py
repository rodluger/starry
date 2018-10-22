"""Test a quick MCMC inference run."""
from starry.kepler import Primary, Secondary, System
import emcee
import numpy as np
np.random.seed(42)
from tqdm import tqdm


def set_coeffs(p, planet):
    """Set the coefficients of the planet map."""
    planet[1, :] = p


def gen_coeffs():
    """Generate random initial conditions."""
    y1m1 = np.random.randn()
    y10 = np.random.randn()
    y11 = np.random.randn()
    return [y1m1, y10, y11]


def lnprior(p):
    """Loosely informative log prior probability."""
    if np.any(p < -5) or np.any(p > 5):
        return -np.inf
    else:
        return 0


def lnlike(p, time, y, yerr, system, planet):
    """Log likelihood."""
    ll = lnprior(p)
    if np.isinf(ll):
        return ll

    # Set the coeffs and compute the flux
    set_coeffs(p, planet)
    system.compute(time)

    # Compute the chi-squared
    chisq = np.sum((y - system.lightcurve) ** 2) / yerr ** 2
    ll += -0.5 * chisq

    return ll


def generate(x, tstart=1, tend=5.3, npts=100, ning=100, neg=100):
    """Generate a synthetic light curve."""
    # Instantiate the star (params known exactly)
    star = Primary()
    star[1] = 0.4
    star[2] = 0.26

    # Instantiate the planet
    planet = Secondary(lmax=1)
    planet.lambda0 = 270
    planet.r = 0.0916
    planet.L = 5e-3
    planet.inc = 87
    planet.a = 11.12799
    planet.prot = 4.3
    planet.porb = 4.3
    planet.tref = 2.0

    # Instantiate the system
    system = System(star, planet)

    # Set the map coeffs
    set_coeffs(x, planet)

    # Time array w/ extra resolution at ingress/egress
    ingress = (1.94, 1.96)
    egress = (2.04, 2.06)
    time = np.linspace(tstart, tend, npts)
    if ingress is not None:
        t = np.linspace(ingress[0], ingress[1], ning)
        time = np.append(time, t)
    if egress is not None:
        t = np.linspace(egress[0], egress[1], neg)
        time = np.append(time, t)
    time = time[np.argsort(time)]

    # Compute and plot the starry flux
    system.compute(time)
    flux = np.array(system.lightcurve)

    # Noise it
    yerr = 1e-4 * np.nanmedian(flux)
    y = flux + yerr * np.random.randn(len(flux))

    # Compute the flux at hi res for plotting
    time_hires = np.linspace(tstart, tend, npts * 100)
    system.compute(time_hires)
    flux_hires = np.array(system.lightcurve)

    return time, y, yerr, star, planet, system, time_hires, flux_hires


def test_mcmc():
    """Run the tests."""
    # These are the values we're going to try to recover
    y1m1 = 0.20
    y10 = 0.30
    y11 = 0.20
    truth = np.array([y1m1, y10, y11])

    # Generate synthetic data
    time, y, yerr, star, planet, system, \
        time_hires, flux_hires = generate(truth)

    # Set the initial conditions
    nburn = 500
    nsteps = 1000
    nwalk = 30
    ndim = len(truth)
    p0 = [gen_coeffs() for k in range(nwalk)]

    # Run our MCMC chain
    sampler = emcee.EnsembleSampler(nwalk, ndim, lnlike,
                                    args=[time, y, yerr, system, planet])
    for i in tqdm(sampler.sample(p0, iterations=nsteps), total=nsteps):
        pass

    # Check that our posteriors are consistent with the truth at 3 sigma
    c1 = sampler.chain[:, nburn:, 0].flatten()
    c2 = sampler.chain[:, nburn:, 1].flatten()
    c3 = sampler.chain[:, nburn:, 2].flatten()
    assert np.abs((y1m1 - np.mean(c1)) / np.std(c1) ** 1) < 3
    assert np.abs((y10 - np.mean(c2)) / np.std(c2) ** 1) < 3
    assert np.abs((y11 - np.mean(c3)) / np.std(c3) ** 1) < 3


if __name__ == "__main__":
    test_mcmc()
