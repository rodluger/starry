from ..ops import Ops, vectorize, to_tensor, is_theano
from .indices import get_ylm_inds, get_ul_inds
import numpy as np
import theano
import theano.tensor as tt
import theano.sparse as ts


class Map(object):
    """

    """

    def __init__(self, ydeg=0, udeg=0, fdeg=0):
        """

        """

        # Instantiate the Theano ops class
        self.ops = Ops(ydeg, udeg, fdeg)

        # Dimensions
        self.ydeg = ydeg
        self.Ny = (self.ydeg + 1) ** 2 
        self.udeg = udeg
        self.Nu = self.udeg + 1
        self.fdeg = fdeg
        self.Nf = (self.fdeg + 1) ** 2
        self.deg = self.ydeg + self.udeg + self.fdeg
        self.N = (self.deg + 1) ** 2

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
            self._y[inds] = val
            self._check_y0()
        elif isinstance(idx, (int, np.int, slice)):
            # User is accessing a limb darkening index
            inds = get_ul_inds(self.udeg, idx)
            self._u[inds] = val
            self._check_u0()
        else:
            raise ValueError("Invalid map index.")

    def _set_y0(self):
        self._y[0] = 1.0

    def _check_y0(self):
        assert self._y[0] == 1.0, \
            "The coefficient of the `Y_{0,0}` harmonic is fixed at unity."

    def _set_u0(self):
        self._u[0] = -1.0

    def _check_u0(self):
        assert self._u[0] == -1.0, \
            "The limb darkening coefficient `u_0` cannot be set."

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

    def reset(self):
        """

        """

        self._y = np.zeros(self.Ny)
        self._set_y0()
        self._u = np.zeros(self.Nu)
        self._set_u0()
        self.inc = 90.0
        self.obl = 0.0

    def X(self, **kwargs):
        """
        Compute and return the light curve design matrix.

        """

        # Should we eval at the end?
        if is_theano(*kwargs.values()):
            evaluate = False
        else:
            evaluate = True

        # Orbital kwargs
        theta = kwargs.pop("theta", 0.0)
        xo = kwargs.pop("xo", 0.0)
        yo = kwargs.pop("yo", 0.0)
        zo = kwargs.pop("zo", 1.0)
        ro = kwargs.pop("ro", 0.0)
        theta, xo, yo, zo = vectorize(theta, xo, yo, zo)

        # Other kwargs
        inc = to_tensor(kwargs.pop("inc", self.inc))
        obl = to_tensor(kwargs.pop("obl", self.obl))

        # Compute & return
        X = self.ops.X(theta, xo, yo, zo, ro, inc, obl)
        if evaluate:
            return X.eval()
        else:
            return X

    def flux(self, *args, **kwargs):
        """
        Compute and return the light curve.
        
        """

        # Should we eval at the end?
        if is_theano(*kwargs.values()):
            evaluate = False
        else:
            evaluate = True

        # Did the user provide map coefficients?
        y = to_tensor(kwargs.pop("y", self.y))

        # Compute & return
        flux = tt.dot(self.X(*args, **kwargs), y)
        if evaluate:
            return flux.eval()
        else:
            return flux