"""Python interface to the C++ starry code."""
import _starry
from .maps import image2map, healpix2map
import numpy as np
import matplotlib.pyplot as pl
__version__ = _starry.__version__


class Map(_starry.Map):
    """The main starry Python interface."""

    def __init__(self, *args, **kwargs):
        """Initialize a starry Map."""
        # Python-only kwargs
        image = kwargs.pop('image', None)

        # Call the C++ constructor
        super().__init__(*args, **kwargs)

        # Allow the user to specify an image or healpix map
        if image is not None:
            if type(image) is str:
                y = image2map(image, lmax=self.lmax)
            # Or is this a healpix array?
            elif type(image) is np.ndarray:
                y = healpix2map(image, lmax=self.lmax)
            else:
                raise ValueError("Invalid `image` value.")
            # Set the coefficients of the base map
            n = 0
            for l in range(self.lmax + 1):
                for m in range(-l, l + 1):
                    self.set_coeff(l, m, y[n])
                    n += 1
            # We need to apply some rotations to get
            # to the desired orientation
            self.rotate([1, 0, 0], np.pi / 2)
            self.rotate([0, 0, 1], np.pi)
            self.rotate([0, 1, 0], np.pi / 2)

    def __getitem__(self, lm_or_n):
        """Allow users to access elements using their `l` and `m` indices."""
        if hasattr(lm_or_n, "__len__") and len(lm_or_n) == 2:
            if type(lm_or_n[0]) is slice or type(lm_or_n[1]) is slice:
                raise ValueError("Slice indexing only supported "
                                 "when accessing items by their `n` index.")
            else:
                return self.get_coeff(lm_or_n[0], lm_or_n[1])
        elif not hasattr(lm_or_n, "__len__"):
            if type(lm_or_n) is slice:
                return self.y[lm_or_n]
            else:
                l = int(np.floor(np.sqrt(lm_or_n)))
                m = lm_or_n - l ** 2 - l
            return self.get_coeff(l, m)
        else:
            raise ValueError("Invalid spherical harmonic index.")

    def __setitem__(self, lm_or_n, val):
        """Allow users to set elements using their `l` and `m` indices."""
        if hasattr(lm_or_n, "__len__") and len(lm_or_n) == 2:
            if type(lm_or_n[0]) is slice or type(lm_or_n[1]) is slice:
                raise ValueError("Slice indexing only supported "
                                 "when accessing items by their `n` index.")
            else:
                return self.set_coeff(lm_or_n[0], lm_or_n[1], val)
        elif not hasattr(lm_or_n, "__len__"):
            if type(lm_or_n) is slice:
                y = np.array(self.y)
                y[lm_or_n] = val
                n = 0
                for l in range(self.lmax + 1):
                    for m in range(-l, l + 1):
                        self.set_coeff(l, m, y[n])
                        n += 1
            else:
                l = int(np.floor(np.sqrt(lm_or_n)))
                m = lm_or_n - l ** 2 - l
                return self.set_coeff(l, m, val)
        else:
            raise ValueError("Invalid spherical harmonic index.")

    def show(self, cmap='plasma', u=[0, 1, 0], theta=0, res=300):
        """Show the rendered map using `imshow`."""
        x, y = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
        F = self.evaluate(u=u, theta=theta, x=x, y=y)
        fig, ax = pl.subplots(1, figsize=(3, 3))
        ax.imshow(F, origin="lower", interpolation="none", cmap=cmap)
        ax.axis('off')
        pl.show()
