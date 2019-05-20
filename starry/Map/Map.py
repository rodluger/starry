from ..ops import Ops, vectorize, to_tensor, is_theano
from .indices import get_ylm_inds, get_ul_inds
import numpy as np
import theano
import theano.tensor as tt
import theano.sparse as ts


class Map(object):
    """

    """

    def __init__(self, ydeg=0, udeg=0, doppler=False):
        """

        """
        # Doppler filter?
        self._doppler = doppler
        if self._doppler:
            fdeg = 3
        else:
            fdeg = 0

        # Instantiate the Theano ops class
        self.ops = Ops(ydeg, udeg, fdeg)

        # Dimensions
        self.ydeg = ydeg
        self.Ny = (ydeg + 1) ** 2 
        self.udeg = udeg
        self.Nu = udeg + 1
        self.fdeg = fdeg
        self.Nf = (fdeg + 1) ** 2
        self.deg = ydeg + udeg + fdeg
        self.N = (ydeg + udeg + fdeg + 1) ** 2

        # Initialize
        self.reset()

    def __getitem__(self, idx):
        """

        """
        if isinstance(idx, tuple) and len(idx) == 2:
            # User is accessing a Ylm index
            inds = get_ylm_inds(self.ydeg, idx[0], idx[1])
            return self._y[inds]
        elif isinstance(idx, (int, np.int)):
            # User is accessing a limb darkening index
            inds = get_ul_inds(self.udeg, idx)
            return self._u[inds]
        else:
            raise ValueError("Invalid map index.")

    def __setitem__(self, idx, val):
        """

        """
        if isinstance(idx, tuple) and len(idx) == 2:
            # User is accessing a Ylm index
            inds = get_ylm_inds(self.ydeg, idx[0], idx[1])
            if 0 in inds:
                raise ValueError("The Y_{0,0} coefficient cannot be set.")
            self._y = tt.set_subtensor(self._y[inds], val * tt.ones(len(inds)))
        elif isinstance(idx, (int, np.int, slice)):
            # User is accessing a limb darkening index
            inds = get_ul_inds(self.udeg, idx)
            if 0 in inds:
                raise ValueError("The u_0 coefficient cannot be set.")
            self._u = tt.set_subtensor(self._u[inds], val * tt.ones(len(inds)))
        else:
            raise ValueError("Invalid map index.")

    @property
    def y(self):
        """

        """
        return self._y

    @property
    def u(self):
        """

        """
        return self._u

    @property
    def inc(self):
        """

        """
        return self._inc

    @inc.setter
    def inc(self, value):
        self._inc = to_tensor(value)

    @property
    def obl(self):
        """

        """
        return self._obl
    
    @obl.setter
    def obl(self, value):
        self._obl = to_tensor(value)

    def reset(self):
        """

        """
        y = np.zeros(self.Ny)
        y[0] = 1.0
        self._y = to_tensor(y)

        u = np.zeros(self.Nu)
        u[0] = -1.0
        self._u = to_tensor(u)

        f = np.zeros(self.Nf)
        f[0] = np.pi
        self._f = to_tensor(f)

        self._inc = to_tensor(90.0)
        self._obl = to_tensor(0.0)

    def X(self, **kwargs):
        """
        Compute and return the light curve design matrix.

        """
        # Orbital kwargs
        theta = kwargs.pop("theta", 0.0)
        xo = kwargs.pop("xo", 0.0)
        yo = kwargs.pop("yo", 0.0)
        zo = kwargs.pop("zo", 1.0)
        ro = kwargs.pop("ro", 0.0)
        theta, xo, yo, zo = vectorize(theta, xo, yo, zo)
        theta, xo, yo, zo, ro = to_tensor(theta, xo, yo, zo, ro)

        # To radians
        theta *= np.pi / 180.

        # Compute & return
        return self.ops.X(theta, xo, yo, zo, ro, 
                          self._inc, self._obl, self._u, self._f)

    def flux(self, **kwargs):
        """
        Compute and return the light curve.
        
        """
        # Compute the design matrix
        X = self.X(**kwargs)

        # Dot it into the map to get the flux
        return tt.dot(X, self.y)