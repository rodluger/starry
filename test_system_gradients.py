import starry2
import numpy as np

names = ["time", "A.prot", "A.tref", "A.Y_{1,-1}", "A.Y_{1,0}", "A.Y_{1,1}",
         "A.u_{1}", "b.r", "b.L", "b.prot", "b.a", "b.porb", "b.inc", "b.ecc",
         "b.w", "b.Omega", "b.lambda0", "b.tref", "b.Y_{1,-1}", "b.Y_{1,0}",
         "b.Y_{1,1}", "b.u_{1}"]


def lightcurve(x, eps=np.zeros(22), gradient=False, delay=False, transit=False):
    """Compute the light curve."""
    star = starry2.kepler.Primary(multi=True)
    b = starry2.kepler.Secondary(multi=True)
    if delay:
        star.r_m = 3e11
    else:
        star.r_m = 0
    star.axis = [1, 3, 2]
    b.axis = [1, 2, 3]
    time = [x[0]] + eps[0]
    star.prot = x[1] + eps[1]
    star.tref = x[2] + eps[2]
    star[1, -1] = x[3] + eps[3]
    star[1, 0] = x[4] + eps[4]
    star[1, 1] = x[5] + eps[5]
    star[1] = x[6] + eps[6]
    b.r = x[7] + eps[7]
    b.L = x[8] + eps[8]
    b.prot = x[9] + eps[9]
    b.a = x[10] + eps[10]
    b.porb = x[11] + eps[11]
    b.inc = x[12] + eps[12]
    b.ecc = x[13] + eps[13]
    b.w = x[14] + eps[14]
    b.Omega = x[15] + eps[15]
    b.lambda0 = x[16] + eps[16]
    b.tref = x[17] + eps[17]
    b[1, -1] = x[18] + eps[18]
    b[1, 0] = x[19] + eps[19]
    b[1, 1] = x[20] + eps[20]
    b[1] = x[21] + eps[21]
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


def run(delay=False, transit=False):
    """Verify the gradient against numerical derivatives."""
    x = np.array([0.5, 1.23, 0.1, 0.11, 0.23, -0.15, 0.3, 0.1, 1.0,
                  1.4, 9.3, 1.2, 89.9, 0.3, 39.5, 67.2, 43.1, -0.8,
                  0.21, 0.17, -0.2, 0.1])
    if transit and delay:
        x[0] = 0.5
    elif transit:
        x[0] = 0.56
    else:
        x[0] = 0.3
    if transit:
        x[12] = 89.9
    else:
        x[12] = 77.3
    epsilon = x * 1e-8
    star_num_grad = np.zeros(22)
    b_num_grad = np.zeros(22)
    fstar1, star_grad, fb1, b_grad = lightcurve(x, gradient=True, delay=delay,
                                                transit=transit)
    for i, name in enumerate(names):
        eps = np.zeros(22)
        eps[i] = epsilon[i]
        fstar2, fb2 = lightcurve(x, eps, delay=delay, transit=transit)
        star_num_grad[i] = (fstar2 - fstar1) / eps[i]
        b_num_grad[i] = (fb2 - fb1) / eps[i]
        assert np.isclose(star_grad[i], star_num_grad[i])
        assert np.isclose(b_grad[i], b_num_grad[i])


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
