from theano.tests.unittest_tools import verify_grad
import starry
import numpy as np


def test_sT(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    map = starry.Map(ydeg=2)
    verify_grad(map._ops.sT, (np.linspace(0.01, 1.09, 30), 0.1), 
                abs_tol=abs_tol, rel_tol=rel_tol, eps=eps)


def test_dotRz(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    map = starry.Map(ydeg=2)
    theta = np.array([0., 15., 30., 45., 60., 75., 90.])

    # Matrix M
    M = np.ones((7, 9))
    verify_grad(map._ops.dotRz, (M, theta), 
                abs_tol=abs_tol, rel_tol=rel_tol, eps=eps)

    # Vector M
    M = np.ones((1, 9))
    verify_grad(map._ops.dotRz, (M, theta), 
                abs_tol=abs_tol, rel_tol=rel_tol, eps=eps)


def test_dotRxy(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    map = starry.Map(ydeg=2)
    inc = 85.0
    obl = 30.0

    # Matrix M
    M = np.ones((7, 9))
    verify_grad(map._ops.dotRxy, (M, inc, obl), 
                abs_tol=abs_tol, rel_tol=rel_tol, eps=eps)

    # Vector M
    M = np.ones((1, 9))
    verify_grad(map._ops.dotRxy, (M, inc, obl), 
                abs_tol=abs_tol, rel_tol=rel_tol, eps=eps)


def test_dotRxyT(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    map = starry.Map(ydeg=2)
    inc = 85.0
    obl = 30.0

    # Matrix M
    M = np.ones((7, 9))
    verify_grad(map._ops.dotRxyT, (M, inc, obl), 
                abs_tol=abs_tol, rel_tol=rel_tol, eps=eps)

    # Vector M
    M = np.ones((1, 9))
    verify_grad(map._ops.dotRxyT, (M, inc, obl), 
                abs_tol=abs_tol, rel_tol=rel_tol, eps=eps)


if __name__ == "__main__":
    test_dotRxyT()
    test_dotRxy()
    test_dotRz()
    test_sT()