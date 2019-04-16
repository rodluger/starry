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

# Noise it
rv = truths["rv"] + truths["rv_err"] * np.random.randn(len(t))

# Plot it
plt.plot(t, rv, 'k.', alpha=0.3, ms=3)
plt.plot(t, truths["rv"])
plt.show()

# Sample it
sampler = xo.PyMC3Sampler(window=100, finish=200)

starry_op = starry.ops.DopplerMapOp(udeg=udeg)

with pm.Model() as model:

    inc = pm.Uniform("inc", 0, 180)
    obl = pm.Uniform("obl", -180, 180)
    alpha = pm.Uniform("alpha", 0, 1)
    veq = pm.Uniform("veq", 0, 10)
    period = pm.Uniform("period", 1.0, 100.0)
    t0 = pm.Uniform("t0", -1, 1)
    u = xo.distributions.QuadLimbDark("u", testval=np.array([0.3, 0.2]))
    r = pm.Uniform("r", 0.01, 0.25)
    b = pm.Uniform("b", 0, 1.25)

    # We're not fitting for theta
    # @dfm: How do I prevent pymc3 from fitting for it?
    theta = np.ones_like(t) * pm.Uniform("theta", 0, 1)
    
    # The map Ylm degree is zero, so there are no Ylms to fit
    y = tt.as_tensor_variable([], name='y')
    y.name = 'y'

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
        tt.as_tensor_variable(y),
        tt.as_tensor_variable(u),
        tt.as_tensor_variable(inc),
        tt.as_tensor_variable(obl),
        tt.as_tensor_variable(veq), 
        tt.as_tensor_variable(alpha),
        tt.as_tensor_variable(theta),
        tt.as_tensor_variable(_x),
        tt.as_tensor_variable(_y),
        tt.as_tensor_variable(_z), 
        tt.as_tensor_variable(rs)
    )

    # Here we track the value of the model light curve for plotting later
    pm.Deterministic("rv_model", rv_model)

    # The likelihood function assuming known Gaussian uncertainty
    pm.Normal("obs", mu=rv_model, sd=truths["rv_err"], observed=rv)

    # Fit for the maximum a posteriori parameters
    map_soln = xo.optimize(start=model.test_point)