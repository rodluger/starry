from theano.tests.unittest_tools import verify_grad
import starry
import numpy as np


def test_sT(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    map = starry.Map(ydeg=2)
    verify_grad(map.ops.sT, (np.linspace(0.01, 1.09, 30), 0.1), 
                abs_tol=abs_tol, rel_tol=rel_tol, eps=eps)


def test_dotRz(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    map = starry.Map(ydeg=2)
    theta = np.array([0., 15., 30., 45., 60., 75., 90.])

    # Matrix M
    M = np.ones((7, 9))
    verify_grad(map.ops.dotRz, (M, theta), 
                abs_tol=abs_tol, rel_tol=rel_tol, eps=eps)

    # Vector M
    M = np.ones((1, 9))
    verify_grad(map.ops.dotRz, (M, theta), 
                abs_tol=abs_tol, rel_tol=rel_tol, eps=eps)


def test_dotRxy(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    map = starry.Map(ydeg=2)
    inc = 85.0
    obl = 30.0

    # Matrix M
    M = np.ones((7, 9))
    verify_grad(map.ops.dotRxy, (M, inc, obl), 
                abs_tol=abs_tol, rel_tol=rel_tol, eps=eps)

    # Vector M
    M = np.ones((1, 9))
    verify_grad(map.ops.dotRxy, (M, inc, obl), 
                abs_tol=abs_tol, rel_tol=rel_tol, eps=eps)


def test_dotRxyT(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    map = starry.Map(ydeg=2)
    inc = 85.0
    obl = 30.0

    # Matrix M
    M = np.ones((7, 9))
    verify_grad(map.ops.dotRxyT, (M, inc, obl), 
                abs_tol=abs_tol, rel_tol=rel_tol, eps=eps)

    # Vector M
    M = np.ones((1, 9))
    verify_grad(map.ops.dotRxyT, (M, inc, obl), 
                abs_tol=abs_tol, rel_tol=rel_tol, eps=eps)


def test_flux(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    map = starry.Map(ydeg=2)
    map.inc = 85.0
    map.obl = 30.0
    theta = np.linspace(0, 30, 10)
    xo = np.linspace(-1.5, 1.5, len(theta))
    yo = np.ones_like(xo) * 0.3
    ro = 0.1

    verify_grad(map.flux, (theta, xo, yo, ro), 
                abs_tol=abs_tol, rel_tol=rel_tol, eps=eps)


if __name__ == "__main__":
    import theano
    print(theano.config.optimizer)
    test_flux()
    quit()
    test_dotRxyT()
    test_dotRxy()
    test_dotRz()
    test_sT()