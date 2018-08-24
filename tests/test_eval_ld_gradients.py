"""Test gradients of limb-darkened maps."""
import starry2
import numpy as np
norm = 0.5 * np.sqrt(np.pi)


def num_grad_eval(Map, lmax, y_deg, yvec, uvec, axis, theta, x, y, eps=1e-8):
    """Return the gradient computed numerically."""
    map = Map(lmax)
    map.axis = axis
    map[:, :] = yvec
    map[:] = uvec
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
    for l in range(y_deg + 1):
        for m in range(-l, l + 1):
            map[l, m] -= eps
            F1 = map.evaluate(x=x, y=y, theta=theta)
            map[l, m] += 2 * eps
            F2 = map.evaluate(x=x, y=y, theta=theta)
            map[l, m] -= eps
            grad['Y_{%d,%d}' % (l, m)] = [(F2 - F1) / (2 * eps)]

    # ld coeffs
    for l in range(1, lmax - y_deg + 1):
        map[l] -= eps
        F1 = map.evaluate(x=x, y=y, theta=theta)
        map[l] += 2 * eps
        F2 = map.evaluate(x=x, y=y, theta=theta)
        map[l] -= eps
        grad['u_{%d}' % (l)] = [(F2 - F1) / (2 * eps)]

    return grad


def run_evaluate(Map):
    """Compare the gradients to numerical derivatives."""

    # Instantiate
    lmax = 3
    y_deg = 1
    map = Map(lmax)
    map.axis = [0, 1, 0]
    map[:y_deg, :] = 1
    map[1] = 1

    # No arguments
    I = map.evaluate()
    I_grad, dI = map.evaluate(gradient=True)
    assert np.allclose(I, I_grad, atol=1e-7)
    dI_num = num_grad_eval(Map, lmax, y_deg, map.y, map.u, map.axis, 0, 0, 0)
    for key in dI.keys():
        if key in dI_num.keys():
            assert np.allclose(dI[key], dI_num[key], atol=1e-7)

    # Scalar evaluation
    I = map.evaluate(x=0.1, y=0.1)
    I_grad, dI = map.evaluate(x=0.1, y=0.1, gradient=True)
    assert np.allclose(I, I_grad, atol=1e-7)
    dI_num = num_grad_eval(Map, lmax, y_deg, map.y, map.u,
                           map.axis, 0, 0.1, 0.1)
    for key in dI.keys():
        if key in dI_num.keys():
            assert np.allclose(dI[key], dI_num[key], atol=1e-7)

    # Scalar evaluation
    I = map.evaluate(x=0.1, y=0.1, theta=30)
    I_grad, dI = map.evaluate(x=0.1, y=0.1, theta=30, gradient=True)
    assert np.allclose(I, I_grad, atol=1e-7)
    dI_num = num_grad_eval(Map, lmax, y_deg, map.y, map.u,
                           map.axis, 30, 0.1, 0.1)
    for key in dI.keys():
        if key in dI_num.keys():
            assert np.allclose(dI[key], dI_num[key], atol=1e-7)

    # Vector evaluation
    I = map.evaluate(x=0.1, y=0.1, theta=[0, 30])
    I_grad, dI = map.evaluate(x=0.1, y=0.1, theta=[0, 30], gradient=True)
    assert np.allclose(I, I_grad, atol=1e-7)
    dI_num1 = num_grad_eval(Map, lmax, y_deg, map.y, map.u,
                            map.axis, 0, 0.1, 0.1)
    dI_num2 = num_grad_eval(Map, lmax, y_deg, map.y, map.u,
                            map.axis, 30, 0.1, 0.1)
    for key in dI.keys():
        if key in dI_num.keys():
            assert np.allclose(dI[key][0], dI_num1[key], atol=1e-7)
            assert np.allclose(dI[key][1], dI_num2[key], atol=1e-7)


def test_ld_evaluate_with_gradients_double():
    """Test the map evaluation with gradients [double]."""
    return run_evaluate(starry2.Map)


def test_ld_evaluate_with_gradients_multi():
    """Test the map evaluation with gradients [multi]."""
    return run_evaluate(starry2.multi.Map)


if __name__ == "__main__":
    test_ld_evaluate_with_gradients_double()
    test_ld_evaluate_with_gradients_multi()
