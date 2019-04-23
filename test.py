import numpy as np
import pymc3 as pm
import exoplanet as xo
import starry
import theano
import theano.tensor as tt
np.random.seed(42)

# True values
truths = {
    #
    # Star
    #
    "A_u":        [0.4, 0.26],
    "A_inc":      75.0,         # deg
    "A_obl":      30.0,         # deg
    "A_alpha":    0.40,
    "A_veq":      5000.0,       # m/s
    "A_m":        0.7,          # Solar
    "A_r":        1.0,          # Solar
    #
    # Planet b
    #
    "b_period":   3.12,         # days
    "b_t0":       2.00,         # days
    "b_inc":      89.5,         # deg
    "b_ecc":      0.10,
    "b_w":        70.0,         # deg
    "b_r":        0.02,         # Solar
    "b_m":        3e-5,         # Solar
    #
    # Dataset
    #
    "tstart":     0.0,          # days
    "tend":       4.0,          # days
    "tcad":       1./24.,   # days 
    "rv_err":     0.1           # m/s
}

# Instantiate the starry `Map`
udeg = len(truths["A_u"])
map = starry.DopplerMap(udeg=udeg)
map[1:] = truths["A_u"]
map.inc = truths["A_inc"]
map.obl = truths["A_obl"]
map.alpha = truths["A_alpha"]
map.veq = truths["A_veq"]

# Instantiate an `orbit` object
orbit = xo.orbits.KeplerianOrbit(
    period=truths["b_period"], 
    t0=truths["b_t0"], 
    incl=truths["b_inc"] * np.pi / 180.,
    m_planet=truths["b_m"],
    m_star=truths["A_m"],
    r_star=truths["A_r"]
)

# Compute the RV curve
t = np.arange(truths["tstart"], truths["tend"], truths["tcad"])
truths["rv_orbit"] = orbit.get_radial_velocity(t).eval()

# Compute the R-M signature
truths["rv_rm"] = map.rv_op(t=t, orbit=orbit, ro=truths["b_r"] / truths["A_r"]).eval()

# Join them and add noise
truths["rv"] = truths["rv_orbit"] + truths["rv_rm"]
rv = truths["rv"] + truths["rv_err"] * np.random.randn(len(t))

# Define our model
map = starry.DopplerMap(udeg=udeg)
with pm.Model() as model:
    
    # Priors on stellar parameters
    A_u = xo.distributions.QuadLimbDark("A_u", testval=np.array([0.3, 0.2]))
    A_inc = pm.Uniform("A_inc", 0.0, 180.0, testval=90.)
    pm.Potential("A_inc_prior", pm.math.sin(A_inc * np.pi / 180.0))
    A_obl = pm.Uniform("A_obl", -180.0, 180.0, testval=0.0)
    A_alpha = pm.Uniform("A_alpha", 0.0, 1.0, testval=0.0)
    A_m = pm.Normal("A_m", 0.7, 0.01)
    A_r = pm.Normal("A_r", 1.0, 0.01)
    
    # Priors on planet parameters
    b_period = pm.Normal("b_period", 3.12, 0.001)
    b_t0 = pm.Normal("b_t0", 2.00, 0.01)
    b_inc = pm.Normal("b_inc", 89.5, 0.1)
    b_ecc = pm.Uniform("b_ecc", 0.0, 1.0, testval=0.0)
    b_w = pm.Uniform("b_w", 0.0, 360.0, testval=0.0)
    b_r = pm.Normal("b_r", 0.02, 0.001)
    b_m = pm.Normal("b_m", 3.0e-5, 1.0e-5)
    
    # Set up a Keplerian orbit for the planet
    orbit = xo.orbits.KeplerianOrbit(
        period=b_period, 
        t0=b_t0, 
        incl=b_inc * np.pi / 180.,
        m_planet=b_m,
        m_star=A_m,
        r_star=A_r
    )

    # Compute the model light curve using starry
    rv_model = map.rv_op(
        t=t, 
        orbit=orbit, 
        u=A_u,
        inc=A_inc,
        obl=A_obl,
        alpha=A_alpha,
    )

    # Here we track the value of the model light curve for plotting later
    pm.Deterministic("rv_model", rv_model)

    # Save our initial guess
    model_guess = xo.eval_in_model(rv_model)

    # The likelihood function assuming known Gaussian uncertainty
    pm.Normal("obs", mu=rv_model, sd=truths["rv_err"], observed=rv)

    # Fit for the maximum a posteriori parameters
    map_soln = xo.optimize(start=model.test_point)