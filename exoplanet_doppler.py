import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import exoplanet as xo
import starry
import theano
import theano.tensor as tt
np.random.seed(42)


# True values
truths = {
    "inc":      75.0,
    "obl":      30.0,
    "alpha":    0.40,
    "veq":      3.00,
    "period":   10.0,
    "t0":       0.0,
    "b":        0.25,
    "r":        0.1,
    "u":        [0.4, 0.26],
    "rv_err":   0.0005
}

# Generate a synthetic dataset
t = np.linspace(-0.15, 0.15, 1000)
udeg = len(truths["u"])
map = starry.DopplerMap(udeg=udeg)
map[1:] = truths["u"]
map.inc = truths["inc"]
map.obl = truths["obl"]
map.alpha = truths["alpha"]
map.veq = truths["veq"]
orbit = xo.orbits.KeplerianOrbit(period=truths["period"], 
                                 t0=truths["t0"], b=truths["b"])
coords = orbit.get_relative_position(t)
x = coords[0].eval()
y = coords[1].eval()
z = -coords[2].eval()
truths["rv"] = map.rv(xo=x, yo=y, zo=z, ro=truths["r"])
rv = truths["rv"] + truths["rv_err"] * np.random.randn(len(t))

# Define our model
starry_op = starry.ops.DopplerMapOp(udeg=udeg)
with pm.Model() as model:

    # Wide priors
    inc = pm.Uniform("inc", 0, 180)
    obl = pm.Uniform("obl", -180, 180)
    alpha = pm.Uniform("alpha", 0, 1)
    veq = pm.Uniform("veq", 0, 10)
    period = pm.Uniform("period", 1.0, 100.0, testval=5.0)
    t0 = pm.Uniform("t0", -0.1, 0.1)
    u = xo.distributions.QuadLimbDark("u", testval=np.array([0.3, 0.2]))
    r = pm.Uniform("r", 0.01, 0.25)
    b = pm.Uniform("b", -1.25, 1.25)

    # We're not fitting for theta
    theta = np.zeros_like(t)
    
    # The map Ylm degree is zero, so there are no Ylms to fit
    y = np.empty(0)

    # Vectorize the occultor radius
    rs = np.ones_like(t) * r
    rs.name = 'r'

    # Set up a Keplerian orbit for the planet
    orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b)
    coords = orbit.get_relative_position(t)
    _x = coords[0]
    _y = coords[1]
    _z = -coords[2]

    # Compute the model light curve using starry
    rv_model = starry_op(
        y,
        u,
        inc,
        obl,
        veq, 
        alpha,
        theta,
        _x,
        _y,
        _z, 
        rs
    )

    # Here we track the value of the model light curve for plotting later
    pm.Deterministic("rv_model", rv_model)

    # Save our initial guess
    model_guess = xo.eval_in_model(rv_model)

    # The likelihood function assuming known Gaussian uncertainty
    pm.Normal("obs", mu=rv_model, sd=truths["rv_err"], observed=rv)

    # Fit for the maximum a posteriori parameters
    map_soln = xo.optimize(start=model.test_point)


# Plot it
fig, ax = plt.subplots(1, figsize=(8, 4))
ax.plot(t, rv, 'k.', alpha=0.3, ms=3)
ax.set_ylim(*ax.get_ylim())
ax.plot(t, truths["rv"], label="True")
ax.plot(t, model_guess, label="Initial")
ax.plot(t, map_soln["rv_model"], label="Optimized")
ax.legend()
plt.show()

# Sample it
#sampler = xo.PyMC3Sampler(window=100, finish=200)