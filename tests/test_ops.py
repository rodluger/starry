from theano.tests.unittest_tools import verify_grad
import theano.tensor as tt
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
    inc = 85.0 * np.pi / 180.
    obl = 30.0 * np.pi / 180.

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
    inc = 85.0 * np.pi / 180.
    obl = 30.0 * np.pi / 180.

    # Matrix M
    M = np.ones((7, 9))
    verify_grad(map.ops.dotRxyT, (M, inc, obl), 
                abs_tol=abs_tol, rel_tol=rel_tol, eps=eps)

    # Vector M
    M = np.ones((1, 9))
    verify_grad(map.ops.dotRxyT, (M, inc, obl), 
                abs_tol=abs_tol, rel_tol=rel_tol, eps=eps)


def test_F(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    map = starry.Map(ydeg=2, udeg=2, doppler=True)
    np.random.seed(11)
    u = np.random.randn(3)
    u[0] = -1
    f = np.random.randn(16)
    verify_grad(map.ops.F, (u, f), abs_tol=abs_tol, rel_tol=rel_tol, eps=eps)


def test_pT(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    map = starry.Map(ydeg=2)
    map[1:, :] = 1
    x = 0.13
    y = 0.25
    z = np.sqrt(1 - x ** 2 - y ** 2)
    verify_grad(map.ops.pT, (x, y, z), abs_tol=abs_tol, rel_tol=rel_tol, eps=eps)


def test_flux(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    map = starry.Map(ydeg=2)
    theta = np.linspace(0, 30, 10)
    xo = np.linspace(-1.5, 1.5, len(theta))
    yo = np.ones_like(xo) * 0.3
    zo = 1.0
    ro = 0.1
    inc = 85.0 * np.pi / 180.
    obl = 30.0 * np.pi / 180.
    y = np.ones(9)
    u = [-1.0]
    f = [np.pi]

    func = lambda *args: tt.dot(map.ops.X(*args), y)

    # Just rotation
    verify_grad(func, (theta, xo, yo, zo, 0.0, inc, obl, u, f), 
                abs_tol=abs_tol, rel_tol=rel_tol, eps=eps)

    # Just occultation
    verify_grad(func, (theta, xo / 3, yo, zo, ro, inc, obl, u, f), 
                abs_tol=abs_tol, rel_tol=rel_tol, eps=eps)

    # Rotation + occultation
    verify_grad(func, (theta, xo, yo, zo, ro, inc, obl, u, f), 
                abs_tol=abs_tol, rel_tol=rel_tol, eps=eps)


if __name__ == "__main__":
    test_pT()
    test_flux()
    test_dotRxyT()
    test_dotRxy()
    test_dotRz()
    test_sT()
    test_F()