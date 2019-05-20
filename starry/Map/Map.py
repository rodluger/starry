from ..ops import Ops, vectorize, to_tensor, is_theano
from .indices import get_ylm_inds, get_ul_inds
from .utils import get_ortho_latitude_lines, get_ortho_longitude_lines
from .sht import image2map, healpix2map, array2map
import numpy as np
import theano
import theano.tensor as tt
import theano.sparse as ts
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
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

    def intensity(self, **kwargs):
        """
        Compute and return the intensity of the map.
        
        """
        theta = kwargs.pop("theta", 0.0)
        x = kwargs.pop("x", 0.0)
        y = kwargs.pop("y", 0.0)
        x, y = vectorize(x, y)
        x, y = to_tensor(x, y)
        theta = tt.reshape(vectorize(theta), [-1])

        # Convert angles radians
        inc = self._inc * radian
        obl = self._obl * radian
        theta *= radian

        # Compute & return
        return self.ops.intensity(theta, x, y, inc, obl, 
                                  self._y, self._u, self._f)

    def render(self, **kwargs):
        """
        Compute and return the sky-projected intensity of 
        the map on a square Cartesian grid.
        
        """
        res = kwargs.pop("res", 300)
        projection = kwargs.pop("projection", "ortho")
        theta = kwargs.pop("theta", 0.0)
        theta = tt.reshape(vectorize(theta), [-1])

        # Convert angles radians
        inc = self._inc * radian
        obl = self._obl * radian
        theta *= radian

        # Compute & return
        return self.ops.render(res, projection, theta, inc, obl, 
                               self._y, self._u, self._f)

    def show(self, **kwargs):
        """
        Display an image of the map, with optional animation.

        """
        # Get kwargs
        cmap = kwargs.pop("cmap", "plasma")
        projection = kwargs.get("projection", "ortho")
        grid = kwargs.pop("grid", True)
        interval = kwargs.pop("interval", 75)
        mp4 = kwargs.pop("mp4", None)

        # Render the map
        image = kwargs.pop("image", self.render(**kwargs))
        if is_theano(image):
            image = image.eval()
        if len(image.shape) == 3:
            nframes = image.shape[-1]
        else:
            nframes = 1
            image = np.reshape(image, image.shape + (1,))

        # Animation
        animated = (nframes > 1)

        if projection == "rect":
            # Set up the plot
            fig, ax = plt.subplots(1, figsize=(7, 3.75))
            extent = (-180, 180, -90, 90)

            # Grid lines
            if grid:
                latlines = np.linspace(-90, 90, 7)[1:-1]
                lonlines = np.linspace(-180, 180, 13)
                for lat in latlines:
                    ax.axhline(lat, color="k", lw=0.5, alpha=0.5, zorder=100)
                for lon in lonlines:
                    ax.axvline(lon, color="k", lw=0.5, alpha=0.5, zorder=100)
            ax.set_xticks(lonlines)
            ax.set_yticks(latlines)
            ax.set_xlabel("Longitude [deg]")
            ax.set_ylabel("Latitude [deg]")

        else:
            # Set up the plot
            fig, ax = plt.subplots(1, figsize=(3, 3))
            ax.axis('off')
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            extent = (-1, 1, -1, 1)

            # Grid lines
            if grid:
                x = np.linspace(-1, 1, 10000)
                y = np.sqrt(1 - x ** 2)
                ax.plot(x, y, 'k-', alpha=1, lw=1)
                ax.plot(x, -y, 'k-', alpha=1, lw=1)
                inc = self.inc.eval()
                obl = self.obl.eval()
                lat_lines = get_ortho_latitude_lines(inc=inc, obl=obl)
                lon_lines = get_ortho_longitude_lines(inc=inc, obl=obl)
                for x, y in lat_lines + lon_lines:
                    ax.plot(x, y, 'k-', lw=0.5, alpha=0.5, zorder=100)

        # Plot the first frame of the image
        img = ax.imshow(image[:, :, 0], origin="lower", 
                        extent=extent, cmap=cmap,
                        interpolation="none",
                        vmin=np.nanmin(image), vmax=np.nanmax(image), 
                        animated=animated)

        # Display or save the image / animation
        if animated:
            interval = kwargs.pop("interval", 75)
            mp4 = kwargs.pop("mp4", None)
            
            def updatefig(i):
                img.set_array(image[:, :, i])
                return img,

            ani = FuncAnimation(fig, updatefig, interval=interval,
                                blit=False, frames=image.shape[-1])

            # Business as usual
            if (mp4 is not None) and (mp4 != ""):
                if mp4.endswith(".mp4"):
                    mp4 = mp4[:-4]
                ani.save('%s.mp4' % mp4, writer='ffmpeg')
                plt.close()
            else:
                try:
                    if 'zmqshell' in str(type(get_ipython())):
                        plt.close()
                        display(HTML(ani.to_jshtml()))
                    else:
                        raise NameError("")
                except NameError:
                    plt.show()
                    plt.close()
        else:
            plt.show()

    def load(self, image, healpix=False, **kwargs):
        """
        Load an image, array, or ``healpix`` map. 
        
        This routine uses various routines in ``healpix`` to compute the spherical
        harmonic expansion of the input image and sets the map's :py:attr:`y`
        coefficients accordingly.

        Args:
            image: A path to an image file, a two-dimensional ``numpy`` 
                array, or a ``healpix`` map array (if ``healpix==True``).
        
        Keyword arguments:
            healpix (bool): Treat ``image`` as a ``healpix`` array? 
                Default ``False``.
            sampling_factor (int): Oversampling factor when computing the 
                ``healpix`` representation of an input image or array. 
                Default 8. Increasing this number may improve the fidelity of 
                the expanded map, but the calculation will take longer.
            sigma (float): If not ``None``, apply gaussian smoothing 
                with standard deviation ``sigma`` to smooth over 
                spurious ringing features. Smoothing is performed with 
                the ``healpix.sphtfunc.smoothalm`` method. 
                Default ``None``.
        """
        # Is this a file name?
        if type(image) is str:
            y = image2map(image, lmax=self.ydeg, **kwargs)
        # or is it an array?
        elif (type(image) is np.ndarray):
            if healpix:
                y = healpix2map(image, lmax=self.ydeg, **kwargs)
            else:
                y = array2map(image, lmax=self.ydeg, **kwargs)
        else:
            raise ValueError("Invalid `image` value.")
        
        # Ingest the coefficients
        self._y = to_tensor(y)


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