# -*- coding: utf-8 -*-
import numpy as np
from ..utils import is_theano, to_tensor, vectorize
import theano.tensor as tt


__all__ = ["DopplerBase"]


class DopplerBase(object):
    """
    .. autoattribute:: alpha
    .. autoattribute:: veq
    .. automethod:: rv
    """

    @staticmethod
    def __descr__():
        return r"""
        This class implements the custom
        :py:meth:`rv()` method for computing the net radial velocity
        imparted by occultations or the darkening due to surface features such as spots. It
        also implements new attributes, including :py:attr:`alpha()` and :py:meth:`veq()`
        for specifying additional map properties.
        """

    def __init__(self, *args, **kwargs):
        super(DopplerBase, self).__init__(*args, **kwargs)
        self._alpha = 0.0
        self._veq = 0.0
        self._unset_rv_filter()

    def _unset_rv_filter(self):
        """Remove the RV filter."""
        coeffs = np.zeros(self.Nf)
        coeffs[0] = np.pi
        self._set_filter((slice(None), slice(None)), coeffs)

    def _set_rv_filter(self):
        """Set the filter coefficients to the RV field of the star."""
        coeffs = self._get_rv_filter(self.inc, self.obl, self.alpha, self.veq)
        self._set_filter((slice(None), slice(None)), coeffs)

    def _get_rv_filter(self, inc, obl, alpha, veq):
        """

        """
        # Theano or numpy?
        if is_theano(inc, obl, veq, alpha):
            math = tt
        else:
            math = np

        # Define some angular quantities
        rad = np.pi / 180
        cosi = math.cos(inc * rad)
        sini = math.sin(inc * rad)
        cosl = math.cos(obl * rad)
        sinl = math.sin(obl * rad)
        A = sini * cosl
        B = -sini * sinl
        C = cosi

        # Compute the Ylm expansion of the RV field
        f = math.reshape([
             0,
             veq * np.sqrt(3) * B * (-A ** 2 * alpha - B ** 2 * alpha - C ** 2 * alpha + 5) / 15,
             0,
             veq * np.sqrt(3) * A * (-A ** 2 * alpha - B ** 2 * alpha - C ** 2 * alpha + 5) / 15,
             0,
             0,
             0,
             0,
             0,
             veq * alpha * np.sqrt(70) * B * (3 * A ** 2 - B ** 2) / 70,
             veq * alpha * 2 * np.sqrt(105) * C * (-A ** 2 + B ** 2) / 105,
             veq * alpha * np.sqrt(42) * B * (A ** 2 + B ** 2 - 4 * C ** 2) / 210,
             0,
             veq * alpha * np.sqrt(42) * A * (A ** 2 + B ** 2 - 4 * C ** 2) / 210,
             veq * alpha * 4 * np.sqrt(105) * A * B * C / 105,
             veq * alpha * np.sqrt(70) * A * (A ** 2 - 3 * B ** 2) / 70
            ], [-1]) * np.pi
        return f

    @property
    def alpha(self):
        r"""
        The rotational shear coefficient, a float in the range [0, 1].
        
        The parameter :math:`\alpha` is used to model linear differential
        rotation. The angular velocity at a given latitude :math:`\theta`
        is

        :math:`\omega = \omega_{eq}(1 - \alpha \sin^2\theta)`

        where :math:`\omega_{eq}` is the equatorial angular velocity.
        """
        return self._alpha
    
    @alpha.setter
    def alpha(self, val):
        assert (val >= 0) and (val <= 1), \
            "The rotational shear coefficient must be between 0 and 1."
        self._alpha = val

    @property
    def veq(self):
        """The equatorial velocity of the object in arbitrary units."""
        return self._veq
    
    @veq.setter
    def veq(self, val):
        assert (val >= 0), "The equatorial velocity must be non-negative."
        self._veq = val

    def render(self, rv=True, **kwargs):
        r"""
        Render the map on a grid and return the pixel intensities (or
        velocity-weighted intensities) as a two-dimensional array 
        (with time as an optional third dimension).

        Kwargs:
            rv (bool): Compute the intensity-weighted radial velocity field of \
                the map? Default :py:obj:`True`. If :py:obj:`False`, computes \
                just the intensity map.
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
        
        Kwargs (temporal maps):
            t (float or ndarray): The time(s) at which to evaluate the map. \
                Default 0.
        
        .. note:: If :py:obj:`rv = True`, the :py:obj:`projection` kwarg is \
                ignored and the map can only be plotted in the orthographic \
                projection.
        """
        if rv:
            kwargs.pop("projection", None)
            self._set_rv_filter()
        res = super(DopplerBase, self).render(**kwargs)
        if rv:
            self._unset_rv_filter()
        return res
    
    def show(self, **kwargs):
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
                ignored and the map can only be plotted in the orthographic \
                projection.
        """
        # Override the `projection` kwarg if we're
        # plotting the radial velocity.
        if kwargs.get("rv", True):
            kwargs.pop("projection", None)
        return super(DopplerBase, self).show(**kwargs)

    def rv(self, **kwargs):
        r"""
        Compute the net radial velocity one would measure from the object.

        The radial velocity is computed as the ratio

            :math:`\Delta RV = \frac{\int Iv \mathrm{d}A}{\int I \mathrm{d}A}`

        where both integrals are taken over the visible portion of the 
        projected disk. :math:`I` is the intensity field (described by the
        spherical harmonic and limb darkening coefficients) and :math:`v`
        is the radial velocity field (computed based on the equatorial velocity
        of the star, its orientation, etc.)

        Kwargs:
            theta (float or ndarray): Angle of rotation. Default 0.
            xo (float or ndarray): The :py:obj:`x` position of the \
                occultor (if any). Default 0.
            yo (float or ndarray): The :py:obj:`y` position of the \
                occultor (if any). Default 0.
            zo (float or ndarray): The :py:obj:`z` position of the \
                occultor (if any). Default 1.0 (on the side closest to \
                the observer).
            ro (float): The radius of the occultor in units of this \
                body's radius. Default 0 (no occultation).
        
        Kwargs (temporal maps or if :py:obj:`orbit` is provided):
            t: Time at which to evaluate the map and/or orbit. Default 0.

        Additional kwargs accepted by this method:
            y: The vector of spherical harmonic coefficients. Default \
                is the map's current spherical harmonic vector.
            u: The vector of limb darkening coefficients. Default \
                is the map's current limb darkening vector.
            inc: The map inclination in degrees. Default is the map's current \
                inclination.
            obl: The map obliquity in degrees. Default is the map's current \
                obliquity. 
            veq: The equatorial velocity of the object in arbitrary units. \
                Default is the map's current velocity.
            alpha: The rotational shear. Default is the map's current shear.
            orbit: And :py:obj:`exoplanet.orbits.KeplerianOrbit` instance. \
                This will override the :py:obj:`b` and :py:obj:`zo` keywords \
                above as long as a time vector :py:obj:`t` is also provided \
                (see above). Default :py:obj:`None`.

        Returns:
            The radial velocity timeseries.
        """
        # Ingest op-specific kwargs
        # Other kwargs get ingested in call to `flux` below
        inc = kwargs.get("inc", None)
        obl = kwargs.get("obl", None)
        veq = kwargs.pop("veq", None)
        alpha = kwargs.pop("alpha", None)
        if inc is None:
            inc = self.inc
        elif not is_theano(inc):
            self.inc = inc
        if obl is None:
            obl = self.obl
        elif not is_theano(obl):
            self.obl = obl
        if veq is None:
            veq = self.veq
        elif not is_theano(veq):
            self.veq = veq
        if alpha is None:
            alpha = self.alpha
        elif not is_theano(alpha):
            self.alpha = alpha

        # Compute the velocity-weighted intensity
        kwargs["f"] = self._get_rv_filter(inc, obl, alpha, veq)
        Iv = self.flux(**kwargs)

        # Compute the inverse of the intensity
        kwargs["f"] = np.append([np.pi], np.zeros(self.Nf - 1))
        invI = np.array([1.0]) / self.flux(**kwargs)
        if is_theano(invI):
            invI = tt.where(tt.isinf(invI), 0.0, invI)
        else:
            invI[np.isinf(invI)] = 0.0

        # The RV signal is just the product        
        return Iv * invI