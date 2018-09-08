import starry2
import numpy as np

names = ["time", "A.prot", "A.tref", "A.Y_{1,-1}", "A.Y_{1,0}", "A.Y_{1,1}",
         "A.u_{1}", "b.r", "b.L", "b.prot", "b.a", "b.porb", "b.inc", "b.ecc",
         "b.w", "b.Omega", "b.lambda0", "b.tref", "b.Y_{1,-1}", "b.Y_{1,0}",
         "b.Y_{1,1}", "b.u_{1}"]


def lightcurve(eps=np.zeros(22), gradient=False, delay=False, transit=False):
    """Compute the light curve."""
    # Evaluate the flux mid-transit (manually tuned)
    if transit and delay:
        time = [0.5] + eps[0]
    elif transit:
        time = [0.56] + eps[0]
    else:
        time = [0.3] + eps[0]

    star = starry2.kepler.Primary(multi=True)
    if delay:
        star.r_m = 3e11
    star.prot = 1.23 + eps[1]
    star.tref = 0.1 + eps[2]
    star[1, -1] = 0.11 + eps[3]
    star[1, 0] = 0.23 + eps[4]
    star[1, 1] = -0.15 + eps[5]
    star[1] = 0.3 + eps[6]
    star.axis = [1, 3, 2]

    b = starry2.kepler.Secondary(multi=True)
    b.r = 0.1 + eps[7]
    b.L = 1 + eps[8]
    b.prot = 1.4 + eps[9]
    b.a = 9.3 + eps[10]
    b.porb = 1.2 + eps[11]
    # Adjust inclination so planet transits if needed
    if transit:
        b.inc = 89.9 + eps[12]
    else:
        b.inc = 77.3 + eps[12]
    b.ecc = 0.3 + eps[13]
    b.w = 3.5 + eps[14]
    b.Omega = 4.2 + eps[15]
    b.lambda0 = 7.1 + eps[16]
    b.tref = -0.8 + eps[17]
    b[1, -1] = 0.21 + eps[18]
    b[1, 0] = 0.17 + eps[19]
    b[1, 1] = -0.2 + eps[20]
    b[1] = 0.1 + eps[21]
    b.axis = [1, 2, 3]

    if gradient:
        sys = starry2.kepler.System(star, b)
        sys.compute(time, gradient=True)
        star_grad = np.zeros(22)
        b_grad = np.zeros(22)
        for i, name in enumerate(names):
            star_grad[i] = star.gradient[name]
            b_grad[i] = b.gradient[name]
        return star.lightcurve[0], star_grad, \
               b.lightcurve[0], b_grad
    else:
        sys = starry2.kepler.System(star, b)
        sys.compute(time)
        return star.lightcurve[0], b.lightcurve[0]


def run(epsilon=1e-8, delay=False, transit=False):
    """Verify the gradient against numerical derivatives."""
    star_num_grad = np.zeros(22)
    b_num_grad = np.zeros(22)
    fstar1, star_grad, fb1, b_grad = lightcurve(gradient=True, delay=delay,
                                                transit=transit)
    for i, name in enumerate(names):
        eps = np.zeros(22)
        eps[i] = epsilon
        fstar2, fb2 = lightcurve(eps, delay=delay, transit=transit)
        star_num_grad[i] = (fstar2 - fstar1) / epsilon
        b_num_grad[i] = (fb2 - fb1) / epsilon

        if not np.isclose(star_grad[i], star_num_grad[i]):
            print("A - %10s: %11.8f %11.8f" % (name, star_grad[i], star_num_grad[i]))
        if not np.isclose(b_grad[i], b_num_grad[i]):
           print("b - %10s: %11.8f %11.8f" % (name, b_grad[i], b_num_grad[i]))


def test_phase_curve():
    """Test the phase curve derivatives."""
    run(delay=False, transit=False)
    run(delay=True, transit=False)


def test_light_curve():
    """Test the phase curve derivatives."""
    run(delay=False, transit=True)
    run(delay=True, transit=True)


if __name__ == "__main__":
    test_phase_curve()
    test_light_curve()
