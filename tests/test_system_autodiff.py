"""Test autodiff with the `System` class."""
from starry.grad import Star, Planet, System
import numpy as np


def numerical_gradient(du1=0, du2=0, dr=0, dporb=0, decc=0, dw=0, dY10=0,
                       dtref=0, dinc=0):
    """Return the gradient computed numerically."""
    # Compute F(x - dx)
    time = np.linspace(-0.35, 0.2, 25)
    star = Star()
    star.map[1] = 0.4 - du1
    star.map[2] = 0.26 - du2
    planet = Planet(lmax=2,
                    r=0.1 - dr,
                    a=60,
                    inc=89.5 - dinc,
                    porb=50 - dporb,
                    prot=0.5,
                    lambda0=90,
                    ecc=0.3 - decc,
                    w=89 - dw,
                    L=1e-3,
                    tref=0 - dtref)
    planet.map[1, 0] = -0.5 - dY10
    system = System([star, planet])
    system.compute(time)
    F1 = np.array(system.flux)

    # Compute F(x + dx)
    star = Star()
    star.map[1] = 0.4 + du1
    star.map[2] = 0.26 + du2
    planet = Planet(lmax=2,
                    r=0.1 + dr,
                    a=60,
                    inc=89.5 + dinc,
                    porb=50 + dporb,
                    prot=0.5,
                    lambda0=90,
                    ecc=0.3 + decc,
                    w=89 + dw,
                    L=1e-3,
                    tref=0 + dtref)
    planet.map[1, 0] = -0.5 + dY10
    system = System([star, planet])
    system.compute(time)
    F2 = np.array(system.flux)

    return (F2 - F1) / (2 * (du1 + du2 + dr + dporb +
                             decc + dw + dY10 + dtref + dinc))


def test_system():
    """Test transit light curves with autodiff."""
    # Time array
    time = np.linspace(-0.35, 0.2, 25)

    # Star
    star = Star()
    star.map[1] = 0.4
    star.map[2] = 0.26

    # Planet
    planet = Planet(lmax=2,
                    r=0.1,
                    a=60,
                    inc=89.5,
                    porb=50,
                    prot=0.5,
                    lambda0=90,
                    ecc=0.3,
                    w=89,
                    L=1e-3,
                    tref=0)
    planet.map[1, 0] = -0.5

    # Compute the starry flux
    system = System([star, planet])
    system.compute(time)

    # Compare the gradient w/ respect to select parameters
    # to a numerical version
    def compare(name, delta, tol=1e-6):
        assert np.max(np.abs(system.gradient[name] -
                             numerical_gradient(**delta))) < tol
    compare('star.u_1', dict(du1=1e-8))
    compare('star.u_2', dict(du2=1e-8))
    compare('planet1.r', dict(dr=1e-8))
    compare('planet1.porb', dict(dporb=1e-8))
    compare('planet1.ecc', dict(decc=1e-8))
    compare('planet1.w', dict(dw=1e-8))
    compare('planet1.Y_{1,0}', dict(dY10=1e-8))
    compare('planet1.tref', dict(dtref=1e-8))
    compare('planet1.inc', dict(dinc=1e-8))
    compare('planet1.porb', dict(dporb=1e-8))


if __name__ == "__main__":
    test_system()
