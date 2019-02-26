import numpy as np
import matplotlib.pyplot as plt


class PythonMapBase(object):
    """

    """
    
    def show(self, theta=0, res=300, cmap="plasma"):
        x, y = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
        X = self.linear_intensity_model(theta=theta, x=x, y=y)
        img = np.dot(X, self.y).reshape(res, res)
        fig, ax = plt.subplots(1, figsize=(3, 3))
        ax.imshow(img, origin="lower", 
                  extent=(-1, 1, -1, 1), cmap=cmap,
                  interpolation="none")
        ax.axis('off')
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
            kwargs['ncol'] = nt
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
            kwargs['ncol'] = nw
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
            super(Map, self).__init__(*init_args, **init_kwargs)

    # Hack this function's docstring
    __doc__ = Map.__doc__

    # Return an instance
    return Map(*args, **kwargs)