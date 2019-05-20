from ..ops import Ops, vectorize, to_tensor, is_theano
from .indices import get_ylm_inds, get_ul_inds
import numpy as np
import theano
import theano.tensor as tt
import theano.sparse as ts
radian = np.pi / 180.0


__all__ = ["Map"]


class YlmBase(object):
    """

    """

    def __init__(self, ydeg, udeg, fdeg):
        """

        """
        # Instantiate the Theano ops class
        self.ops = Ops(ydeg, udeg, fdeg)

        # Dimensions
        self._ydeg = ydeg
        self._Ny = (ydeg + 1) ** 2 
        self._udeg = udeg
        self._Nu = udeg + 1
        self._fdeg = fdeg
        self._Nf = (fdeg + 1) ** 2
        self._deg = ydeg + udeg + fdeg
        self._N = (ydeg + udeg + fdeg + 1) ** 2

        # Initialize
        self.reset()

    @property
    def ydeg(self):
        """
        
        """
        return self._ydeg
    
    @property
    def Ny(self):
        """
        
        """
        return self._Ny

    @property
    def udeg(self):
        """
        
        """
        return self._udeg
    
    @property
    def Nu(self):
        """
        
        """
        return self._Nu

    @property
    def fdeg(self):
        """
        
        """
        return self._fdeg
    
    @property
    def Nf(self):
        """
        
        """
        return self._Nf

    @property
    def deg(self):
        """
        
        """
        return self._deg
    
    @property
    def N(self):
        """
        
        """
        return self._N

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

        # Convert angles radians
        inc = self._inc * radian
        obl = self._obl * radian
        theta *= radian

        # Compute & return
        return self.ops.X(theta, xo, yo, zo, ro, 
                        inc, obl, self._u, self._f)

    def flux(self, **kwargs):
        """
        Compute and return the light curve.
        
        """
        # Compute the design matrix
        X = self.X(**kwargs)

        # Dot it into the map to get the flux
        return tt.dot(X, self.y)


class DopplerBase(object):
    """

    """

    def reset(self):
        """

        """
        super(DopplerBase, self).reset()
        self._alpha = to_tensor(0.0)
        self._veq = to_tensor(0.0)

    @property
    def alpha(self):
        """
        The rotational shear coefficient, a float in the range ``[0, 1]``.
        
        The parameter :math:`\\alpha` is used to model linear differential
        rotation. The angular velocity at a given latitude :math:`\\theta`
        is

        :math:`\\omega = \\omega_{eq}(1 - \\alpha \\sin^2\\theta)`

        where :math:`\\omega_{eq}` is the equatorial angular velocity of
        the object.
        """
        return self._alpha
    
    @alpha.setter
    def alpha(self, value):
        self._alpha = to_tensor(value)

    @property
    def veq(self):
        """The equatorial velocity of the object in arbitrary units."""
        return self._veq
    
    @veq.setter
    def veq(self, value):
        self._veq = to_tensor(value)

    def _unset_doppler_filter(self):
        f = np.zeros(self.Nf)
        f[0] = np.pi
        self._f = to_tensor(f)

    def _set_doppler_filter(self):
        # Define some angular quantities
        cosi = tt.cos(self._inc * radian)
        sini = tt.sin(self._inc * radian)
        cosl = tt.cos(self._obl * radian)
        sinl = tt.sin(self._obl * radian)
        A = sini * cosl
        B = -sini * sinl
        C = cosi

        # Compute the Ylm expansion of the RV field
        self._f = tt.reshape([
             0,
             self._veq * np.sqrt(3) * B * 
                (-A ** 2 * self._alpha - B ** 2 * self._alpha - 
                 C ** 2 * self._alpha + 5) / 15,
             0,
             self._veq * np.sqrt(3) * A * 
                (-A ** 2 * self._alpha - B ** 2 * self._alpha - 
                 C ** 2 * self._alpha + 5) / 15,
             0,
             0,
             0,
             0,
             0,
             self._veq * self._alpha * np.sqrt(70) * B * 
                (3 * A ** 2 - B ** 2) / 70,
             self._veq * self._alpha * 2 * np.sqrt(105) * C * 
                (-A ** 2 + B ** 2) / 105,
             self._veq * self._alpha * np.sqrt(42) * B * 
                (A ** 2 + B ** 2 - 4 * C ** 2) / 210,
             0,
             self._veq * self._alpha * np.sqrt(42) * A * 
                (A ** 2 + B ** 2 - 4 * C ** 2) / 210,
             self._veq * self._alpha * 4 * np.sqrt(105) * A * B * C / 105,
             self._veq * self._alpha * np.sqrt(70) * A * 
                (A ** 2 - 3 * B ** 2) / 70
            ], [-1]) * np.pi

    def rv(self, **kwargs):
        """
        Compute the net radial velocity one would measure from the object.

        The radial velocity is computed as the ratio

            :math:`\\Delta RV = \\frac{\\int Iv \\mathrm{d}A}{\\int I \\mathrm{d}A}`

        where both integrals are taken over the visible portion of the 
        projected disk. :math:`I` is the intensity field (described by the
        spherical harmonic and limb darkening coefficients) and :math:`v`
        is the radial velocity field (computed based on the equatorial velocity
        of the star, its orientation, etc.)
        """
        # Compute the velocity-weighted intensity
        self._set_doppler_filter()
        Iv = self.flux(**kwargs)

        # Compute the inverse of the intensity
        self._unset_doppler_filter()
        invI = np.array([1.0]) / self.flux(**kwargs)
        invI = tt.where(tt.isinf(invI), 0.0, invI)

        # The RV signal is just the product        
        return Iv * invI


def Map(ydeg=0, udeg=0, doppler=False):
    """

    """

    Bases = (YlmBase,)

    if doppler:
        Bases = (DopplerBase,) + Bases
        fdeg = 3
    else:
        fdeg = 0

    class Map(*Bases): 
        pass

    return Map(ydeg, udeg, fdeg)