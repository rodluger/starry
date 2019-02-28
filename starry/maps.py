import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .extensions import RAxisAngle

class PythonMapBase(object):
    """

    """

    def show(self, theta=0, res=300, cmap="plasma", projection="ortho", **kwargs):
        """

        """        
        # Type-specific kwargs
        if projection.lower().startswith("rect"):
            projection = "rect"
            npts = 1
            model_kwargs = dict()
        elif projection.lower().startswith("ortho"):
            projection = "ortho"
            if hasattr(theta, "__len__"):
                npts = len(theta)
            else:
                npts = 1
            model_kwargs = dict(theta=theta)
        else:
            raise ValueError("Invalid projection. Allowed projections are " +
                             "`rectangular` and `orthographic` (default).")

        # Are we modeling time variability?
        if self._temporal:
            t = kwargs.pop("t", 0.0)
            if hasattr(t, "__len__"):
                npts = max(npts, len(t))
            model_kwargs["t"] = t

        # Are we modeling reflected light?
        if self._reflected:
            source = np.array(kwargs.pop("source", [-1.0, 0.0, 0.0]))
            if len(source.shape) == 2:
                npts = max(npts, len(source))
            model_kwargs["source"] = source

        # Are we doing wavelength dependence?
        if self._spectral:
            animated = True
            assert npts == 1, "Spectral map rotation cannot be animated."
        else:
            animated = (npts > 1)

        if projection == "rect":

            # Generate the lat/lon grid for one hemisphere
            lon = np.linspace(-np.pi, np.pi, res)
            lat = np.linspace(1e-3, np.pi / 2, res // 2)
            lon, lat = np.meshgrid(lon, lat)
            x = np.sin(np.pi / 2 - lat) * np.cos(lon - np.pi / 2)
            y = np.sin(np.pi / 2 - lat) * np.sin(lon - np.pi / 2)

            # Rotate so we're looking down the north pole
            map_axis = np.array(self.axis)
            sinalpha = np.sqrt(self.axis[0] ** 2 + self.axis[1] ** 2)
            cosalpha = self.axis[2]
            u = np.array([self.axis[1], self.axis[0], 0]) / sinalpha
            alpha = (180 / np.pi) * np.arctan2(sinalpha, cosalpha)
            self.axis = u
            self.rotate(alpha)

            # We need to rotate the light source as well
            if self._reflected:
                R = RAxisAngle(u, alpha)
                source = np.atleast_2d(model_kwargs["source"])
                for i in range(len(source)):
                    source[i] = np.dot(R, source[i])
                model_kwargs["source"] = source

            # Compute the linear model
            X = self.linear_intensity_model(x=x, y=y, **model_kwargs)

            # Compute the northern hemisphere map
            if self._spectral:
                Z_north = np.dot(X, self.y).reshape(res, res // 2, self.nw)
                Z_north = np.moveaxis(Z_north, -1, 0)
                Z_north = np.rot90(Z_north, axes=(1, 2), k=3)
                Z_north = np.roll(Z_north, -res // 4, axis=2)
            else:
                Z_north = np.dot(X, self.y).reshape(npts, res // 2, res)

            # Flip the planet around
            self.axis = [1, 0, 0]
            self.rotate(180)

            # We need to rotate the light source as well
            # (and re-compute the model)
            if self._reflected:
                R = RAxisAngle([1, 0, 0], 180)
                source = np.atleast_2d(model_kwargs["source"])
                for i in range(len(source)):
                    source[i] = np.dot(R, source[i])
                model_kwargs["source"] = source
                X = self.linear_intensity_model(x=x, y=y, **model_kwargs)

            # Compute the southern hemisphere map
            if self._spectral:
                Z_south = np.dot(X, self.y).reshape(res, res // 2, self.nw)
                Z_south = np.moveaxis(Z_south, -1, 0)
                Z_south = np.flip(Z_south, axis=(1, 2))
                Z_south = np.rot90(Z_south, axes=(1, 2), k=3)
                Z_south = np.roll(Z_south, -res // 4, axis=2)
            else:
                Z_south = np.dot(X, self.y).reshape(npts, res // 2, res)
                Z_south = np.flip(Z_south, axis=(1, 2))
                Z_south = np.roll(Z_south, -res // 2, axis=2)

            # Join them
            Z = np.concatenate((Z_south, Z_north), axis=1)

            # Undo all the rotations
            self.rotate(-180)
            self.axis = u
            self.rotate(-alpha)
            self.axis = map_axis

            # Set up the plot
            fig, ax = plt.subplots(1, figsize=(6, 3))
            extent = (-180, 180, -90, 90)

        else:

            # Create a grid of X and Y and construct the linear model
            x, y = np.meshgrid(np.linspace(-1, 1, res), 
                               np.linspace(-1, 1, res))
            X = self.linear_intensity_model(x=x, y=y, **model_kwargs)
            if self._spectral:
                Z = np.moveaxis(
                        np.dot(X, self.y).reshape(res, res, self.nw), 
                        -1, 0)
            else:
                Z = np.dot(X, self.y).reshape(npts, res, res)

            # Set up the plot
            fig, ax = plt.subplots(1, figsize=(3, 3))
            ax.axis('off')
            extent = (-1, 1, -1, 1)

        # Plot the first frame of the image
        img = ax.imshow(Z[0], origin="lower", 
                        extent=extent, cmap=cmap,
                        interpolation="none",
                        vmin=np.nanmin(Z), vmax=np.nanmax(Z), 
                        animated=animated)
        
        # Display or save the image / animation
        if animated:
            interval = kwargs.pop("interval", 75)
            gif = kwargs.pop("gif", None)
            
            def updatefig(i):
                img.set_array(Z[i])
                return img,

            ani = FuncAnimation(fig, updatefig, interval=interval,
                                blit=False, frames=len(Z))

            # TODO: Jupyter override

            # Business as usual
            if (gif is not None) and (gif != ""):
                if gif.endswith(".gif"):
                    gif = gif[:-4]
                ani.save('%s.gif' % gif, writer='imagemagick')
            else:
                plt.show()
            plt.close()
        else:
            plt.show()


def Map(*args, **kwargs):
    """
    Figures out which `Map` class the user wants and instantiates it.

    """
    # Figure out the correct base class
    multi = kwargs.pop('multi', False)
    reflected = kwargs.pop('reflected', False)
    nw = kwargs.pop('nw', None)
    nt = kwargs.pop('nt', None)
    if (nw is None):
        if (nt is None):
            if (not reflected):
                if (not multi):
                    try:
                        from ._starry_default_double import Map as CMapBase
                    except ModuleNotFoundError:
                        raise ModuleNotFoundError("Requested module not found. " + 
                            "Please re-compile `starry` with bit 1 enabled.")
                else:
                    try:
                        from ._starry_default_multi import Map as CMapBase
                    except ModuleNotFoundError:
                        raise ModuleNotFoundError("Requested module not found. " + 
                            "Please re-compile `starry` with bit 2 enabled.")
            else:
                if (not multi):
                    try:
                        from ._starry_default_refl_double import Map as CMapBase
                    except ModuleNotFoundError:
                        raise ModuleNotFoundError("Requested module not found. " + 
                            "Please re-compile `starry` with bit 64 enabled.")
                else:
                    try:
                        from ._starry_default_refl_multi import Map as CMapBase
                    except ModuleNotFoundError:
                        raise ModuleNotFoundError("Requested module not found. " + 
                            "Please re-compile `starry` with bit 128 enabled.")
        else:
            kwargs['nterms'] = nt
            if (not reflected):
                if (not multi):
                    try:
                        from ._starry_temporal_double import Map as CMapBase
                    except ModuleNotFoundError:
                        raise ModuleNotFoundError("Requested module not found. " + 
                            "Please re-compile `starry` with bit 16 enabled.")
                else:
                    try:
                        from ._starry_temporal_multi import Map as CMapBase
                    except ModuleNotFoundError:
                        raise ModuleNotFoundError("Requested module not found. " + 
                            "Please re-compile `starry` with bit 32 enabled.")
            else:
                if (not multi):
                    try:
                        from ._starry_temporal_refl_double import Map as CMapBase
                    except ModuleNotFoundError:
                        raise ModuleNotFoundError("Requested module not found. " + 
                            "Please re-compile `starry` with bit 1024 enabled.")
                else:
                    try:
                        from ._starry_temporal_refl_multi import Map as CMapBase
                    except ModuleNotFoundError:
                        raise ModuleNotFoundError("Requested module not found. " + 
                            "Please re-compile `starry` with bit 2048 enabled.")
    else:
        if (nt is None):
            kwargs['nterms'] = nw
            if (not reflected):
                if (not multi):
                    try:
                        from ._starry_spectral_double import Map as CMapBase
                    except ModuleNotFoundError:
                        raise ModuleNotFoundError("Requested module not found. " + 
                            "Please re-compile `starry` with bit 4 enabled.")
                else:
                    try:
                        from ._starry_spectral_multi import Map as CMapBase
                    except ModuleNotFoundError:
                        raise ModuleNotFoundError("Requested module not found. " + 
                            "Please re-compile `starry` with bit 8 enabled.")
            else:
                if (not multi):
                    try:
                        from ._starry_spectral_refl_double import Map as CMapBase
                    except ModuleNotFoundError:
                        raise ModuleNotFoundError("Requested module not found. " + 
                            "Please re-compile `starry` with bit 256 enabled.")
                else:
                    try:
                        from ._starry_spectral_refl_multi import Map as CMapBase
                    except ModuleNotFoundError:
                        raise ModuleNotFoundError("Requested module not found. " + 
                            "Please re-compile `starry` with bit 512 enabled.")
        else:
            raise ValueError("Spectral maps cannot have temporal variability.")


    # Subclass it
    class Map(CMapBase, PythonMapBase):
        __doc__ = CMapBase.__doc__
        def __init__(self, *init_args, **init_kwargs):
            self._multi = multi
            self._reflected = reflected
            self._temporal = (nt is not None)
            self._spectral = (nw is not None)
            super(Map, self).__init__(*init_args, **init_kwargs)

    # Hack this function's docstring
    __doc__ = Map.__doc__

    # Return an instance
    return Map(*args, **kwargs)