from theano.tests.unittest_tools import verify_grad
import starry
import numpy as np


def test_sT(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    map = starry.Map(ydeg=2)
    verify_grad(map._ops.sT, (np.linspace(0.01, 1.09, 30), 0.1), 
                abs_tol=abs_tol, rel_tol=rel_tol, eps=eps)


def test_dotRz(abs_tol=1e-5, rel_tol=1e-5, eps=1e-7):
    map = starry.Map(ydeg=2)
    M = np.ones((3, 9))
    theta = np.array([0.1, 0.3, 0.5])
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    verify_grad(map._ops.dotRz, (M, theta), 
                abs_tol=abs_tol, rel_tol=rel_tol, eps=eps)


if __name__ == "__main__":
    test_dotRz()