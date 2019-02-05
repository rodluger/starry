import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import exoplanet as xo
from starry.ops import LightCurve


# Compute a limb-darkened light curve using starry
lmax = 2
t = np.linspace(-0.1, 0.1, 1000)
np.random.seed(41)
y_true = 0.1 * np.random.randn((lmax + 1) ** 2)
y_true[0] = 1.0
op = LightCurve(lmax)

# Randomize an orbit
np.random.seed(123)
period_true = np.random.uniform(5, 20)
t0_true = period_true * np.random.rand()
t = np.arange(0, 80, 0.02)
ferr = 5e-4

with pm.Model() as model:

    # The baseline flux
    mean = pm.Normal("mean", mu=0.0, sd=1.0)
    
    # The time of a reference transit for each planet
    t0 = pm.Normal("t0", mu=t0_true, sd=1.0)
    
    # The log period; also tracking the period itself
    logP = pm.Normal("logP", mu=np.log(period_true), sd=0.1)
    period = pm.Deterministic("period", pm.math.exp(logP))
    
    # Normal distributions for the map coeffs
    y = pm.Normal("y", mu=y_true, sd=1.0, shape=len(y_true))

    # Normal distributions for r and b
    r = pm.Normal("r", mu=0.06, sd=0.001)
    b = pm.Normal("b", mu=0.4, sd=0.03)
    
    # Set up a Keplerian orbit for the planets
    orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b)
    
    # Compute the model light curve using starry
    light_curve = op.get_light_curve(orbit=orbit, r=r, t=t, y=y) + mean
    
    # Here we track the value of the model light curve for plotting
    # purposes
    pm.Deterministic("light_curve", light_curve)
    
    # In this line, we simulate the dataset that we will fit
    flux = xo.eval_in_model(light_curve)
    flux += ferr * np.random.randn(len(flux))
    
    # The likelihood function assuming known Gaussian uncertainty
    pm.Normal("obs", mu=light_curve, sd=ferr, observed=flux)
    
    # Fit for the maximum a posteriori parameters given the simuated
    # dataset
    map_soln = pm.find_MAP(start=model.test_point)


np.random.seed(42)
sampler = xo.PyMC3Sampler(window=100, finish=200)
with model:
    burnin = sampler.tune(tune=2500, start=map_soln, step_kwargs=dict(target_accept=0.9))
    trace = sampler.sample(draws=3000)