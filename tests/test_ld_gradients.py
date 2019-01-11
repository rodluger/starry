"""Test gradients of limb-darkened maps."""
import starry2
import numpy as np


def num_grad_flux(multi, lmax, y_deg, yvec, uvec, axis, theta, xo, yo, ro,
                  eps=1e-8):
    """Return the gradient computed numerically."""
    map = starry2.Map(lmax, multi=multi)
    map.axis = axis
    map[:, :] = yvec
    map[:] = uvec
    grad = {}

    # x
    F1 = map.flux(xo=xo - eps, yo=yo, ro=ro, theta=theta)
    F2 = map.flux(xo=xo + eps, yo=yo, ro=ro, theta=theta)
    grad['xo'] = [(F2 - F1) / (2 * eps)]

    # y
    F1 = map.flux(xo=xo, yo=yo - eps, ro=ro, theta=theta)
    F2 = map.flux(xo=xo, yo=yo + eps, ro=ro, theta=theta)
    grad['yo'] = [(F2 - F1) / (2 * eps)]

    # ro
    # Let's not call starry with a negative radius!
    F1 = map.flux(xo=xo, yo=yo, ro=ro, theta=theta)
    F2 = map.flux(xo=xo, yo=yo, ro=ro + eps, theta=theta)
    grad['ro'] = [(F2 - F1) / (eps)]

    # theta
    F1 = map.flux(xo=xo, yo=yo, ro=ro, theta=theta - eps)
    F2 = map.flux(xo=xo, yo=yo, ro=ro, theta=theta + eps)
    grad['theta'] = [(F2 - F1) / (2 * eps)]

    # map coeffs
    grad['y'] = []
    for l in range(y_deg + 1):
        for m in range(-l, l + 1):
            map[l, m] -= eps
            F1 = map.flux(xo=xo, yo=yo, ro=ro, theta=theta)
            map[l, m] += 2 * eps
            F2 = map.flux(xo=xo, yo=yo, ro=ro, theta=theta)
            map[l, m] -= eps
            grad['y'] += [(F2 - F1) / (2 * eps)]

    # ld coeffs
    grad['u'] = []
    for l in range(1, lmax - y_deg + 1):
        map[l] -= eps
        F1 = map.flux(xo=xo, yo=yo, ro=ro, theta=theta)
        map[l] += 2 * eps
        F2 = map.flux(xo=xo, yo=yo, ro=ro, theta=theta)
        map[l] -= eps
        grad['u'] += [(F2 - F1) / (2 * eps)]

    return grad


def run_flux(multi=False, case="ld"):
    """Compare the gradients to numerical derivatives."""
    # Instantiate
    lmax = 4
    map = starry2.Map(lmax, multi=multi)
    map.axis = [0, 1, 0]
    if case == "ld":
        # Limb darkening only
        y_deg = 0
        map[0, 0] = 1
        map[1] = 0.4
        map[2] = 0.26
        map[3] = 0.5
        map[4] = -0.3
    elif case == "sph":
        # Spherical harmonics only
        y_deg = 4
        map[:, :] = 1
    else:
        # Both!
        y_deg = 3
        map[1] = 0.4
        map[:y_deg, :] = 1

    # Scalar evaluation
    I = map.flux(theta=30)
    I_grad, dI = map.flux(theta=30, gradient=True)
    assert np.allclose(I, I_grad, atol=1e-7)
    dI_num = num_grad_flux(multi, lmax, y_deg, map.y, map.u,
                           map.axis, 30, 0, 0, 0)
    for key in dI.keys():
        if (key in dI_num.keys()):
            assert np.allclose(dI[key], dI_num[key], atol=1e-6)

    # Scalar evaluation
    I = map.flux(xo=0.1, yo=0.1, ro=0.1)
    I_grad, dI = map.flux(xo=0.1, yo=0.1, ro=0.1, gradient=True)
    assert np.allclose(I, I_grad, atol=1e-7)
    dI_num = num_grad_flux(multi, lmax, y_deg, map.y, map.u,
                           map.axis, 0, 0.1, 0.1, 0.1)

    for key in dI.keys():
        if (key in dI_num.keys()):
            assert np.allclose(dI[key], dI_num[key], atol=1e-6)

    # Scalar evaluation
    I = map.flux(xo=0.1, yo=0.1, ro=0.1, theta=30)
    I_grad, dI = map.flux(xo=0.1, yo=0.1, ro=0.1, theta=30, gradient=True)
    assert np.allclose(I, I_grad, atol=1e-7)
    dI_num = num_grad_flux(multi, lmax, y_deg, map.y, map.u,
                           map.axis, 30, 0.1, 0.1, 0.1)
    for key in dI.keys():
        if (key in dI_num.keys()):
            assert np.allclose(dI[key], dI_num[key], atol=1e-6)

    # Vector evaluation
    I = map.flux(xo=0.1, yo=0.1, ro=0.1, theta=[0, 30])
    I_grad, dI = map.flux(xo=0.1, yo=0.1, ro=0.1,
                              theta=[0, 30], gradient=True)
    assert np.allclose(I, I_grad, atol=1e-7)
    dI_num1 = num_grad_flux(multi, lmax, y_deg, map.y, map.u,
                            map.axis, 0, 0.1, 0.1, 0.1)
    dI_num2 = num_grad_flux(multi, lmax, y_deg, map.y, map.u,
                            map.axis, 30, 0.1, 0.1, 0.1)
    for key in dI.keys():
        if (key in dI_num.keys()):
            assert np.allclose(np.squeeze(dI[key]).transpose()[0], dI_num1[key], atol=1e-6)
            assert np.allclose(np.squeeze(dI[key]).transpose()[1], dI_num2[key], atol=1e-6)


def test_ld_flux_with_gradients_double():
    """Test the flux with gradients [double]."""
    for case in ["ld"]: # TODO, "sph", "ld+sph"]:
        run_flux(multi=False, case=case)


def test_ld_flux_with_gradients_multi():
    """Test the flux with gradients [multi]."""
    for case in ["ld"]: # TODO, "sph", "ld+sph"]:
        run_flux(multi=True, case=case)


if __name__ == "__main__":
    test_ld_flux_with_gradients_double()
    test_ld_flux_with_gradients_multi()
