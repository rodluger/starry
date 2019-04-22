# -*- coding: utf-8 -*-
import numpy as np
from . import rvderivs
from ..ops import DopplerMapOp
import theano.tensor as tt


__all__ = ["DopplerBase"]


class DopplerBase(object):
    """
    Doppler map base.

    """
    def __init__(self, *args, **kwargs):
        super(DopplerBase, self).__init__(*args, **kwargs)
        self._alpha = 0.0
        self._veq = 0.0
        self._unset_rv_filter()
        self._op = DopplerMapOp(self)

    def _unset_rv_filter(self):
        """Remove the RV filter."""
        coeffs = np.zeros(self.Nf)
        coeffs[0] = 1.0
        self._set_filter((slice(None, None, None), slice(None, None, None)), coeffs)
        self.DfDinc = None
        self.DfDobl = None
        self.DfDalpha = None
        self.DfDveq = None

    def _set_rv_filter(self, gradient=False):
        """Set the filter coefficients to the RV field of the star."""
        # Parameters
        obl = self.obl
        alpha = self.alpha
        
        # Define some angular quantities
        cosi = np.cos(self.inc * np.pi / 180)
        sini = np.sin(self.inc * np.pi / 180)
        cosl = np.cos(self.obl * np.pi / 180)
        sinl = np.sin(self.obl * np.pi / 180)
        A = sini * cosl
        B = -sini * sinl
        C = cosi

        # Compute the Ylm expansion of the RV field
        # NOTE: We implicitly apply the `starry` Ylm normalization in the equation
        # below, so the factors of `pi` all go away!
        if self.alpha == 0:
            f = np.zeros(16)
            f[1] = self.veq * np.sqrt(3) * B / 3
            f[3] = self.veq * np.sqrt(3) * A / 3
        else:
            f = np.array([
                0,
                np.sqrt(3) * B * (-A ** 2 - B ** 2 - C ** 2 + 5 / self.alpha) / 15,
                0,
                np.sqrt(3) * A * (-A ** 2 - B ** 2 - C ** 2 + 5 / self.alpha) / 15,
                0,
                0,
                0,
                0,
                0,
                np.sqrt(70) * B * (3 * A ** 2 - B ** 2) / 70,
                2 * np.sqrt(105) * C * (-A ** 2 + B ** 2) / 105,
                np.sqrt(42) * B * (A ** 2 + B ** 2 - 4 * C ** 2) / 210,
                0,
                np.sqrt(42) * A * (A ** 2 + B ** 2 - 4 * C ** 2) / 210,
                4 * np.sqrt(105) * A * B * C / 105,
                np.sqrt(70) * A * (A ** 2 - 3 * B ** 2) / 70
            ]) * self.veq * self.alpha

        # Compute the derivs
        if gradient:
            DADi = cosi * cosl
            DADl = -sini * sinl
            DBDi = -cosi * sinl
            DBDl = -sini * cosl
            DCDi = -sini
            DfDA = rvderivs.DfDA(f, A, B, C, self.alpha, self.veq)
            DfDB = rvderivs.DfDB(f, A, B, C, self.alpha, self.veq)
            DfDC = rvderivs.DfDC(f, A, B, C, self.alpha, self.veq)
            self.DfDinc = (DfDA * DADi + DfDB * DBDi + DfDC * DCDi) * np.pi / 180
            self.DfDobl = (DfDA * DADl + DfDB * DBDl) * np.pi / 180
            self.DfDalpha = rvderivs.DfDalpha(f, A, B, C, self.alpha, self.veq)
            self.DfDveq = rvderivs.DfDveq(f, A, B, C, self.alpha, self.veq)

        # Set the filter
        self._set_filter((slice(None, None, None), slice(None, None, None)), f)

    @property
    def alpha(self):
        """The rotational shear coefficient."""
        return self._alpha
    
    @alpha.setter
    def alpha(self, val):
        """The rotational shear coefficient."""
        assert (val >= 0) and (val <= 1), \
            "The rotational shear coefficient must be between 0 and 1."
        self._alpha = val

    @property
    def veq(self):
        """The equatorial velocity in arbitrary units."""
        return self._veq
    
    @veq.setter
    def veq(self, val):
        """The equatorial velocity in arbitrary units."""
        assert (val >= 0), "The equatorial velocity must be non-negative."
        self._veq = val

    def rv(self, *args, gradient=False, **kwargs):
        """Compute the net RV of the star."""
        if gradient:
            # Compute the integral of Iv
            self._set_rv_filter(True)
            DfDalpha = np.array(self.DfDalpha)
            DfDveq= np.array(self.DfDveq)
            DfDinc = np.array(self.DfDinc)
            DfDobl = np.array(self.DfDobl)
            Iv, Iv_grad = np.array(self.flux(*args, gradient=True, **kwargs))
            # Compute the integral of I
            self._unset_rv_filter()
            I, I_grad = np.array(self.flux(*args, gradient=True, **kwargs))
            # Chain rule for the gradient
            grad = {}
            for key in Iv_grad.keys():
                if key == "f":
                    continue
                grad[key] = (Iv_grad[key] * I - Iv * I_grad[key]) / I ** 2
            # Compute RV field gradients
            grad["alpha"] = np.dot(Iv_grad["f"].T, DfDalpha) / I
            grad["veq"] = np.dot(Iv_grad["f"].T, DfDveq) / I
            grad["inc"] += np.dot(Iv_grad["f"].T, DfDinc) / I
            grad["obl"] += np.dot(Iv_grad["f"].T, DfDobl) / I
            return Iv / I, grad
        else:
            self._set_rv_filter(False)
            Iv = np.array(self.flux(*args, **kwargs))
            self._unset_rv_filter()
            I = np.array(self.flux(*args, **kwargs))
            return Iv / I
    
    def render(self, rv=True, *args, **kwargs):
        """Render the image of the star, optionally weighted by the RV."""
        if rv:
            self._set_rv_filter()
        res = super(DopplerBase, self).render(*args, **kwargs)
        if rv:
            self._unset_rv_filter()
        return res
    
    def rv_op(self, y=None, u=None, inc=None, obl=None, veq=None, alpha=None,
              theta=0, orbit=None, t=None, xo=None, yo=None, zo=1, ro=0.1):
        """
        
        """
        # TODO: Implement this op for spectral and temporal types.
        if self._spectral or self._temporal:
            raise NotImplementedError(
                "Op not yet implemented for this map type."
            )

        # Map coefficients. If not set, default to the
        # values of the Map instance itself.
        if y is None:
            y = np.array(self.y[1:])
        if u is None:
            u = np.array(self.u[1:])

        # Misc properties. If not set, default to the
        # values of the Map instance itself.
        if inc is None:
            inc = self.inc
        if obl is None:
            obl = self.obl
        if veq is None:
            veq = self.veq
        if alpha is None:
            alpha = self.alpha

        # Orbital coords.
        if orbit is not None:

            # Compute the orbit
            assert t is not None, \
                "Please provide a set of times `t` at which to compute the orbit."
            try:
                npts = len(t)
            except TypeError:
                npts = tt.as_tensor(t).shape.eval()[0]
            coords = orbit.get_relative_position(t)
            xo = coords[0]
            yo = coords[1]
            # Note that `exoplanet` uses a slightly different coord system!
            zo = -coords[2]

            # Vectorize `theta` and `ro`
            theta = tt.as_tensor_variable(theta)
            if (theta.ndim == 0):
                theta = tt.ones(npts) * theta
            ro = tt.as_tensor_variable(ro)
            if (ro.ndim == 0):
                ro = tt.ones(npts) * ro

        else:

            if (xo is None) or (yo is None) or (zo is None) or (ro is None):

                # No occultation
                theta = tt.as_tensor_variable(theta)
                if (theta.ndim == 0):
                    npts = 1
                else:
                    npts = theta.shape.eval()[0]
                theta = tt.ones(npts) * theta
                xo = tt.zeros(npts)
                yo = tt.zeros(npts)
                zo = tt.zeros(npts)
                ro = tt.zeros(npts)
            
            else:

                # Occultation with manually specified coords
                xo = tt.as_tensor_variable(xo)
                yo = tt.as_tensor_variable(yo)
                zo = tt.as_tensor_variable(zo)
                ro = tt.as_tensor_variable(ro)
                theta = tt.as_tensor_variable(theta)

                # Figure out the length of the timeseries
                if (xo.ndim != 0):
                    npts = xo.shape.eval()[0]
                elif (yo.ndim != 0):
                    npts = yo.shape.eval()[0]
                elif (zo.ndim != 0):
                    npts = zo.shape.eval()[0]
                elif (ro.ndim != 0):
                    npts = ro.shape.eval()[0]
                elif (theta.ndim != 0):
                    npts = theta.shape.eval()[0]
                else:
                    npts = 1 

                # Vectorize everything
                if (xo.ndim == 0):
                    xo = tt.ones(npts) * xo
                if (yo.ndim == 0):
                    yo = tt.ones(npts) * yo
                if (zo.ndim == 0):
                    zo = tt.ones(npts) * zo
                if (ro.ndim == 0):
                    ro = tt.ones(npts) * ro
                if (theta.ndim == 0):
                    theta = tt.ones(npts) * theta

        # Now ensure everything is `floatX`.
        # This is necessary because Theano will try to cast things
        # to float32 if they can be exactly represented with 32 bits.
        args = [y, u, inc, obl, veq, alpha, theta, xo, yo, zo, ro]
        for i, arg in enumerate(args):
            if hasattr(arg, 'astype'):
                args[i] = arg.astype(tt.config.floatX)
            else:
                args[i] = getattr(np, tt.config.floatX)(arg)

        # Call the op
        return self._op(*args)