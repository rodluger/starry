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
                    from ._starry_default_double import Map as MapBase
                else:
                    from ._starry_default_multi import Map as MapBase
            else:
                if (not multi):
                    from ._starry_default_refl_double import Map as MapBase
                else:
                    from ._starry_default_refl_multi import Map as MapBase
        else:
            kwargs['ncol'] = nt
            if (not reflected):
                if (not multi):
                    from ._starry_temporal_double import Map as MapBase
                else:
                    from ._starry_temporal_multi import Map as MapBase
            else:
                if (not multi):
                    from ._starry_temporal_refl_double import Map as MapBase
                else:
                    from ._starry_temporal_refl_multi import Map as MapBase
    else:
        if (nt is None):
            kwargs['ncol'] = nw
            if (not reflected):
                if (not multi):
                    from ._starry_spectral_double import Map as MapBase
                else:
                    from ._starry_spectral_multi import Map as MapBase
            else:
                if (not multi):
                    from ._starry_spectral_refl_double import Map as MapBase
                else:
                    from ._starry_spectral_refl_multi import Map as MapBase
        else:
            raise ValueError("Spectral maps cannot have temporal variability.")


    # Subclass it
    class Map(MapBase):
        __doc__ = MapBase.__doc__
        def __init__(self, *init_args, **init_kwargs):
            super(Map, self).__init__(*init_args, **init_kwargs)

        # Add custom attributes/methods here

    # Hack this function's docstring
    __doc__ = Map.__doc__

    # Return an instance
    return Map(*args, **kwargs)