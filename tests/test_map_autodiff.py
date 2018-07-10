"""Test autodiff with the `Map` class."""
from starry.grad import Map
import numpy as np

# Default system settings
ro = 0.1
xo = np.linspace(-1.5, 1.5, 30)
yo = np.linspace(-0.3, 0.3, 30)
theta = np.linspace(0, 30, 30)
axis = np.array([1.0, 1.0, 1.0])
y = [1, -0.3, -0.5, 0.3, 0.25, -0.25, 0.1, -0.3, 0.05]


def numerical_gradient(dxo=0, dyo=0, dro=0, dtheta=0,
                       dY1m1=0, dY10=0, dY11=0, dY2m2=0,
                       dY2m1=0, dY20=0, dY21=0, dY22=0):
    """Return the gradient computed numerically."""
    # Compute F(x - dx)
    map = Map()
    map[:] = y - np.array([0, dY1m1, dY10, dY11, dY2m2,
                           dY2m1, dY20, dY21, dY22])
    F1 = map.flux(axis=axis, theta=theta - dtheta, xo=xo - dxo,
                  yo=yo - dyo, ro=ro - dro)

    # Compute F(x + dx)
    map = Map()
    map[:] = y + np.array([0, dY1m1, dY10, dY11, dY2m2,
                           dY2m1, dY20, dY21, dY22])
    F2 = map.flux(axis=axis, theta=theta + dtheta, xo=xo + dxo,
                  yo=yo + dyo, ro=ro + dro)

    return (F2 - F1) / (2 * (dxo + dyo + dro + dtheta +
                             dY1m1 + dY10 + dY11 + dY2m2 + dY2m1 +
                             dY20 + dY21 + dY22))


def test_singularities():
    """Test singular points in the derivatives."""
    map = Map()
    map[:] = 1
    map.flux(xo=0, yo=0, ro=0.1)
    assert not np.isnan(map.gradient['xo'])


def test_map():
    """Test map light curves with autodiff."""
    # Star
    map = Map()
    map[:] = y
    map.flux(axis=axis, theta=theta, xo=xo, yo=yo, ro=ro)

    # Compare the gradient w/ respect to select parameters
    # to a numerical version
    def compare(name, delta, tol=1e-6):
        assert np.max(np.abs(map.gradient[name] -
                             numerical_gradient(**delta))) < tol

    compare('xo', dict(dxo=1e-8))
    compare('yo', dict(dyo=1e-8))
    compare('ro', dict(dro=1e-8))
    compare('theta', dict(dtheta=1e-8))
    compare('Y_{1,-1}', dict(dY1m1=1e-8))
    compare('Y_{1,0}', dict(dY10=1e-8))
    compare('Y_{1,1}', dict(dY11=1e-8))
    compare('Y_{2,-2}', dict(dY2m2=1e-8))
    compare('Y_{2,-1}', dict(dY2m1=1e-8))
    compare('Y_{2,0}', dict(dY20=1e-8))
    compare('Y_{2,1}', dict(dY21=1e-8))
    compare('Y_{2,2}', dict(dY22=1e-8))


if __name__ == "__main__":
    test_map()
    # TODO Currently failing!
    # test_singularities()
