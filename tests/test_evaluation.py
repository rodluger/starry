"""Test the map evaluation."""
from starry2 import Map
import numpy as np


def numerical_gradient(lmax, yvec, axis, theta, x, y, eps=1e-8):
    """Return the gradient computed numerically."""
    map = Map(lmax)
    map.axis = axis
    map[:] = yvec
    grad = {}

    # x
    F1 = map.evaluate(x=x - eps, y=y, theta=theta)
    F2 = map.evaluate(x=x + eps, y=y, theta=theta)
    grad['x'] = [(F2 - F1) / (2 * eps)]

    # y
    F1 = map.evaluate(x=x, y=y - eps, theta=theta)
    F2 = map.evaluate(x=x, y=y + eps, theta=theta)
    grad['y'] = [(F2 - F1) / (2 * eps)]

    # theta
    F1 = map.evaluate(x=x, y=y, theta=theta - eps)
    F2 = map.evaluate(x=x, y=y, theta=theta + eps)
    grad['theta'] = [(F2 - F1) / (2 * eps)]

    # map coeffs
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            map[l, m] -= eps
            F1 = map.evaluate(x=x, y=y, theta=theta)
            map[l, m] += 2 * eps
            F2 = map.evaluate(x=x, y=y, theta=theta)
            map[l, m] -= eps
            grad['Y_{%d,%d}' % (l, m)] = [(F2 - F1) / (2 * eps)]

    return grad


def test_evaluation():
    """Test the map evaluation against some benchmarks."""
    # Instantiate
    lmax = 2
    map = Map(lmax)
    map.axis = [0, 1, 0]
    map[:] = 1

    # No arguments
    I = map.evaluate()
    assert np.allclose(I, 1.4014804341818383)

    # Scalar evaluation
    I = map.evaluate(x=0.1, y=0.1)
    assert np.allclose(I, 1.7026057774431276)

    # Scalar evaluation
    I = map.evaluate(x=0.1, y=0.1, theta=30)
    assert np.allclose(I, 0.7736072493369371)

    # Vector evaluation
    I = map.evaluate(x=[0.1, 0.2, 0.3], y=[0.1, 0.2, 0.3], theta=30)
    assert np.allclose(I, [0.7736072493369371,
                           1.0432785526935853,
                           1.318434613210305])
    # Rotation caching
    I = map.evaluate(x=0.1, y=0.1, theta=[0, 30, 30, 0])
    assert np.allclose(I, [1.7026057774431276,
                           0.7736072493369371,
                           0.7736072493369371,
                           1.7026057774431276])


def test_evaluation_with_gradients():
    """Test the map evaluation with gradients."""
    # Instantiate
    lmax = 2
    map = Map(lmax)
    map.axis = [0, 1, 0]
    map[:] = 1

    # No arguments
    I = map.evaluate()
    I_grad, dI = map.evaluate(gradient=True)
    assert np.allclose(I, I_grad)
    dI_num = numerical_gradient(lmax, map[:], map.axis, 0, 0, 0)
    for key in dI.keys():
        assert np.allclose(dI[key], dI_num[key])

    # Scalar evaluation
    I = map.evaluate(x=0.1, y=0.1)
    I_grad, dI = map.evaluate(x=0.1, y=0.1, gradient=True)
    assert np.allclose(I, I_grad)
    dI_num = numerical_gradient(lmax, map[:], map.axis, 0, 0.1, 0.1)
    for key in dI.keys():
        assert np.allclose(dI[key], dI_num[key])

    # Scalar evaluation
    I = map.evaluate(x=0.1, y=0.1, theta=30)
    I_grad, dI = map.evaluate(x=0.1, y=0.1, theta=30, gradient=True)
    assert np.allclose(I, I_grad)
    dI_num = numerical_gradient(lmax, map[:], map.axis, 30, 0.1, 0.1)
    for key in dI.keys():
        assert np.allclose(dI[key], dI_num[key])

    # Vector evaluation
    I = map.evaluate(x=0.1, y=0.1, theta=[0, 30])
    I_grad, dI = map.evaluate(x=0.1, y=0.1, theta=[0, 30], gradient=True)
    assert np.allclose(I, I_grad)
    dI_num1 = numerical_gradient(lmax, map[:], map.axis, 0, 0.1, 0.1)
    dI_num2 = numerical_gradient(lmax, map[:], map.axis, 30, 0.1, 0.1)
    for key in dI.keys():
        assert np.allclose(dI[key][0], dI_num1[key])
        assert np.allclose(dI[key][1], dI_num2[key])


if __name__ == "__main__":
    test_evaluation()
    test_evaluation_with_gradients()
