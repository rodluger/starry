import starry2
import numpy as np

names = ["time", "A.prot", "A.tref", "A.Y_{1,-1}", "A.Y_{1,0}", "A.Y_{1,1}",
         "A.u_{1}", "b.r", "b.L", "b.prot", "b.a", "b.porb", "b.inc", "b.ecc",
         "b.w", "b.Omega", "b.lambda0", "b.tref", "b.Y_{1,-1}", "b.Y_{1,0}",
         "b.Y_{1,1}", "b.u_{1}"]

def lightcurve(eps=np.zeros(22), gradient=False):
    """Compute the light curve."""
    time = [0.3] + eps[0]

    star = starry2.kepler.Primary()
    star.prot = 1.23 + eps[1]
    star.tref = 0.1 + eps[2]
    star[1, -1] = 0.11 + eps[3]
    star[1, 0] = 0.23 + eps[4]
    star[1, 1] = -0.15 + eps[5]
    star[1] = 0.3 + eps[6]

    b = starry2.kepler.Secondary()
    b.r = 0.1 + eps[7]
    b.L = 0.001 + eps[8]
    b.prot = 1.4 + eps[9]
    b.a = 30.3 + eps[10]
    b.porb = 1.2 + eps[11]
    b.inc = 77.3 + eps[12] # no transit!
    b.ecc = 0.3 + eps[13]
    b.w = 20.5 + eps[14]
    b.Omega = 30.2 + eps[15]
    b.lambda0 = 52.1 + eps[16]
    b.tref = -0.8 + eps[17]
    b[1, -1] = 0.21 + eps[18]
    b[1, 0] = 0.17 + eps[19]
    b[1, 1] = -0.08 + eps[20]
    b[1] = 0.1 + eps[21]
    b.axis = [1, 2, 3]

    if gradient:
        sys = starry2.kepler.System(star, b)
        sys.compute(time, gradient=True)
        grad = np.zeros(22)
        for i, name in enumerate(names):
            grad[i] = sys.gradient[name]
        return sys.lightcurve[0], grad
    else:
        sys = starry2.kepler.System(star, b)
        sys.compute(time)
        return sys.lightcurve[0]


def run():
    """Verify the gradient against numerical derivatives."""
    num_grad = np.zeros(22)
    f1, grad = lightcurve(gradient=True)
    for i, name in enumerate(names):
        eps = np.zeros(22)
        eps[i] = 1e-8
        f2 = lightcurve(eps)
        num_grad[i] = (f2 - f1) / 1e-8
        if np.isclose(grad[i], num_grad[i]):
            status = ""
        else:
            status = "X"
        print("%10s: %8.5f %8.5f %s" % (name, grad[i], num_grad[i], status))


if __name__ == "__main__":
    run()
