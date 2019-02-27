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
        theta = np.atleast_1d(theta)
        assert len(theta.shape) == 1, "Invalid shape for keyword `theta`."
        
        # Are we modeling time variability?
        if self._temporal:
            t = np.atleast_1d(kwargs.pop("t", 0.0))
            assert len(t.shape) == 1, "Invalid shape for keyword `t`."
            if len(t) == 1:
                t = np.ones_like(theta) * t[0]
            else:
                if len(theta) == 1:
                    theta = np.ones_like(t) * theta[0]
                assert len(t) == len(theta), "Invalid size for keyword `t`."

        # Construct the linear model
        if self._spectral:
            animated = True
            Z = np.empty((self.nw, res, res))
            assert len(theta) == 1, "Spectral map rotation cannot be animated."
            X = self.linear_intensity_model(theta=theta[0], x=x, y=y)
            for i in range(self.nw):
                Z[i] = np.dot(X, self.y[:, i]).reshape(res, res)
        else:
            animated = (len(theta) > 1)
            Z = np.empty((len(theta), res, res))
            for i in range(len(theta)):
                if self._temporal:
                    X = self.linear_intensity_model(t=t[i], theta=theta[i], x=x, y=y)
                else:
                    X = self.linear_intensity_model(theta=theta[i], x=x, y=y)
                Z[i] = np.dot(X, self.y).reshape(res, res)

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
                    from ._starry_default_double import Map as CMapBase
                else:
                    from ._starry_default_multi import Map as CMapBase
            else:
                if (not multi):
                    from ._starry_default_refl_double import Map as CMapBase
                else:
                    from ._starry_default_refl_multi import Map as CMapBase
        else:
            kwargs['nterms'] = nt
            if (not reflected):
                if (not multi):
                    from ._starry_temporal_double import Map as CMapBase
                else:
                    from ._starry_temporal_multi import Map as CMapBase
            else:
                if (not multi):
                    from ._starry_temporal_refl_double import Map as CMapBase
                else:
                    from ._starry_temporal_refl_multi import Map as CMapBase
    else:
        if (nt is None):
            kwargs['nterms'] = nw
            if (not reflected):
                if (not multi):
                    from ._starry_spectral_double import Map as CMapBase
                else:
                    from ._starry_spectral_multi import Map as CMapBase
            else:
                if (not multi):
                    from ._starry_spectral_refl_double import Map as CMapBase
                else:
                    from ._starry_spectral_refl_multi import Map as CMapBase
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