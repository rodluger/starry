import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class PythonMapBase(object):
    """

    """

    def show(self, theta=0, res=300, cmap="plasma", **kwargs):
        """

        """
        # Create a grid of X and Y
        x, y = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
        
        # Check input dimensions
        if hasattr(theta, "__len__"):
            npts = len(theta)
        else:
            npts = 1
        
        # Type-specific kwargs
        model_kwargs = {}

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

        # Construct the linear model
        if self._spectral:
            animated = True
            assert npts == 1, "Spectral map rotation cannot be animated."
            X = self.linear_intensity_model(theta=theta, x=x, y=y, **model_kwargs)
            Z = np.moveaxis(np.dot(X, self.y).reshape(res, res, self.nw), -1, 0)
        else:
            animated = (npts > 1)
            X = self.linear_intensity_model(theta=theta, x=x, y=y, **model_kwargs)
            Z = np.dot(X, self.y).reshape(npts, res, res)

        # Set up the plot
        vmin = np.nanmin(Z)
        vmax = np.nanmax(Z)
        fig, ax = plt.subplots(1, figsize=(3, 3))
        img = ax.imshow(Z[0], origin="lower", 
                        extent=(-1, 1, -1, 1), cmap=cmap,
                        interpolation="none",
                        vmin=vmin, vmax=vmax, animated=animated)
        ax.axis('off')

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