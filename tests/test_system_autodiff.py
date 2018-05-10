"""Test autodiff with the `System` class."""
from starry.grad import Star, Planet, System
import numpy as np


def test_system():
    """Test transit light curves with autodiff."""
    # Time array
    time = np.linspace(-0.35, 0.2, 2500)

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
                    L=1e-3)
    planet.map[1, 0] = -0.5

    # Compute the starry flux
    system = System([star, planet])
    system.compute(time)

    # TODO: Checks
    

if __name__ == "__main__":
    test_system()
