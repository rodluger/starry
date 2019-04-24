# -*- coding: utf-8 -*-
import numpy as np
from . import rvderivs
from ..ops import DopplerMapOp
import theano.tensor as tt


__all__ = ["DopplerBase"]


class DopplerBase(object):
    """
    .. autoattribute:: alpha
    .. autoattribute:: veq
    .. automethod:: rv(*args, **kwargs)
    .. automethod:: rv_op(y=None, u=None, inc=None, obl=None, veq=None, alpha=None, \
            theta=0, orbit=None, t=None, xo=None, yo=None, zo=1, ro=0.1)
    """

    @staticmethod
    def __descr__():
        return (
            "Instantiate a :py:mod:`starry` Doppler map. This map behaves the same " +
            "as a regular :py:mod:`starry.Map` instance, except it implements the custom " +
            ":py:meth:`rv()` and :py:meth:`rv_op()` methods for computing the net radial velocity " +
            "imparted by occultations or the darkening due to surface features such as spots. It " +
            "also implements new attributes, including :py:attr:`alpha()` and :py:meth:`veq()` " +
            "for specifying additional map properties.\n\n" +
            "Args:\n" +
            "    ydeg (int): Largest spherical harmonic degree of the surface map.\n" +
            "    udeg (int): Largest limb darkening degree of the surface map. Default 0.\n" +
            "    fdeg (int): Largest spherical harmonic filter degree. Default 0.\n" +
            "    nw (int): Number of map wavelength bins. Default :py:obj:`None`.\n" +
            "    nt (int): Number of map temporal bins. Default :py:obj:`None`.\n" +
            "    multi (bool): Use multi-precision to perform all " +
            "        calculations? Default :py:obj:`False`. If :py:obj:`True`, " +
            "        defaults to 32-digit (approximately 128-bit) floating " +
            "        point precision. This can be adjusted by changing the " +
            "        :py:obj:`STARRY_NMULTI` compiler macro.\n\n")

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
        r"""
        Compute the net radial velocity one would measure from the object.

        The radial velocity is computed as the ratio

            :math:`\Delta RV = \frac{\int Iv \mathrm{d}A}{\int I \mathrm{d}A}`

        where both integrals are taken over the visible portion of the 
        projected disk. :math:`I` is the intensity field (described by the
        spherical harmonic and limb darkening coefficients) and :math:`v`
        is the radial velocity field (computed based on the equatorial velocity
        of the star, its orientation, etc.)

        Args:
            t (float or ndarray): Time at which to evaluate. Default 0. \
                *Temporal maps only.*
            theta (float or ndarray): Angle of rotation. Default 0. \
                *Not available for pure limb darkening.*
            xo (float or ndarray): The :py:obj:`x` position of the \
                occultor (if any). Default 0. \
                *Not available for pure limb darkening.*
            yo (float or ndarray): The :py:obj:`y` position of the \
                occultor (if any). Default 0. \
                *Not available for pure limb darkening.*
            zo (float or ndarray): The :py:obj:`z` position of the \
                occultor (if any). Default 1.0 (on the side closest to \
                the observer).
            b (float or ndarray): The impact parameter of the \
                occultor. Default 0. *Purely limb-darkened maps only.*
            ro (float): The radius of the occultor in units of this \
                body's radius. Default 0 (no occultation).
            gradient (bool): Compute and return the gradient of the \
                flux as well? Default :py:obj:`False`.
            source (ndarray): The source position, a unit vector or a
                vector of unit vectors. Default :math:`-\hat{x} = (-1, 0, 0)`.
                *Reflected light maps only.*

        Returns:
            A vector of radial velocities. If :py:obj:`gradient` is enabled, also returns a \
            dictionary whose keys are the derivatives of `rv` with respect \
            to all model parameters.

        """
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
            invI = 1.0 / I
            invI[np.isinf(invI)] = 0.0
            # Chain rule for the gradient
            grad = {}
            for key in Iv_grad.keys():
                if key == "f":
                    continue
                grad[key] = (Iv_grad[key] * I - Iv * I_grad[key]) * invI ** 2
            # Compute RV field gradients
            grad["alpha"] = np.dot(Iv_grad["f"].T, DfDalpha) * invI
            grad["veq"] = np.dot(Iv_grad["f"].T, DfDveq) * invI
            grad["inc"] += np.dot(Iv_grad["f"].T, DfDinc) * invI
            grad["obl"] += np.dot(Iv_grad["f"].T, DfDobl) * invI
            return Iv * invI, grad
        else:
            self._set_rv_filter(False)
            Iv = np.array(self.flux(*args, **kwargs))
            self._unset_rv_filter()
            I = np.array(self.flux(*args, **kwargs))
            invI = 1.0 / I
            invI[np.isinf(invI)] = 0.0
            return Iv * invI
    
    def render(self, rv=True, *args, **kwargs):
        """
        Render the map on a grid and return the pixel intensities (or
        velocity-weighted intensities) as a 
        two-dimensional array (with time as an optional third dimension).

        Args:
            rv (bool): Compute the intensity-weighted radial velocity field of \
                the map? Default :py:obj:`True`. If :py:obj:`False`, computes just the \
                intensity map.
            theta (float): Angle of rotation of the map in degrees. Default 0.
            res (int): Map resolution, corresponding to the number of pixels \
                on a side (for the orthographic projection) or the number of \
                pixels in latitude (for the rectangular projection; the number \
                of pixels in longitude is twice this value). Default 300.
            projection (str): One of "orthographic" or "rectangular". The former \
                results in a map of the disk as seen on the plane of the sky, \
                padded by :py:obj:`NaN` outside of the disk. The latter results \
                in an equirectangular (geographic, equidistant cylindrical) \
                view of the entire surface of the map in latitude-longitude space. \
                Default "orthographic".
            t (float or ndarray): The time(s) at which to evaluate the map. \
                *Temporal maps only*. Default 0.
            source (ndarray): A unit vector corresponding to the direction to the \
                light source. This may optionally be a vector of unit vectors. \
                *Reflected light maps only*. Default :math:`-\hat{x}`.

        .. note:: If :py:obj:`rv = True`, the :py:obj:`projection` kwarg is \
            ignored and the map can only be plotted in the orthographic projection.

        """
        if rv:
            kwargs.pop("projection", None)
            self._set_rv_filter()
        res = super(DopplerBase, self).render(*args, **kwargs)
        if rv:
            self._unset_rv_filter()
        return res
    
    def show(self, *args, **kwargs):
        """
        Render and plot an image of the map in either intensity or radial
        velocity; optionally display an animation.

        If running in a Jupyter Notebook, animations will be displayed
        in the notebook using javascript.
        Refer to the docstring of :py:meth:`render` for additional kwargs
        accepted by this method.

        Args:
            Z (ndarray): The array of pixel intensities returned by a call \
                to :py:meth:`render`. Default :py:obj:`None`, in which case \
                this routine will call :py:meth:`render` with any additional \
                kwargs provided by the user.
            rv (bool): Plot the intensity-weighted radial velocity field of \
                the map? Default :py:obj:`True`. If :py:obj:`False`, plots just the \
                intensity map.
            cmap: The colormap used for plotting (a string or a \
                :py:obj:`matplotlib` colormap object). Default "plasma".
            grid (bool): Overplot static grid lines? Default :py:obj:`True`.
            interval (int): Interval in ms between frames (animated maps only). \
                Default 75.
            mp4 (str): Name of the mp4 file to save the animation to \
                (animated maps only). Default :py:obj:`None`.
            kwargs: Any additional kwargs accepted by :py:meth:`render`.

        .. note:: If :py:obj:`rv = True`, the :py:obj:`projection` kwarg is \
            ignored and the map can only be plotted in the orthographic projection.

        """
        # Override the `projection` kwarg if we're
        # plotting the radial velocity.
        if kwargs.get("rv", True):
            kwargs.pop("projection", None)
        return super(DopplerBase, self).show(*args, **kwargs)

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
            xo = coords[0] / orbit.r_star
            yo = coords[1] / orbit.r_star
            # Note that `exoplanet` uses a slightly different coord system!
            zo = -coords[2] / orbit.r_star

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