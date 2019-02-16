"""Defines the Python Map interface."""
__all__ = ["Map"]
__mapdoc__ = None
__version__ = None


# List of all available modules
_modules = {
    1: dict(
            name="_starry_default_double",
            condition="(nw is None) and (nt is None) and (not multi) and (not reflected)",
            map_args="lmax"
    ),
    2: dict(
            name="_starry_default_multi",
            condition="(nw is None) and (nt is None) and (multi) and (not reflected)",
            map_args="lmax"
    ),
    4: dict(
            name="_starry_spectral_double",
            condition="(nw is not None) and (nw > 0) and (nt is None) and (not multi) and (not reflected)",
            map_args="lmax, nw"
    ),
    8: dict(
            name="_starry_spectral_multi",
            condition="(nw is not None) and (nw > 0) and (nt is None) and (multi) and (not reflected)",
            map_args="lmax, nw"
    ),
    16: dict(
            name="_starry_temporal_double",
            condition="(nt is not None) and (nt > 0) and (nw is None) and (not multi) and (not reflected)",
            map_args="lmax, nt"
    ),
    32: dict(
            name="_starry_temporal_multi",
            condition="(nt is not None) and (nt > 0) and (nw is None) and (multi) and (not reflected)",
            map_args="lmax, nt"
    ),
    64: dict(
            name="_starry_default_refl_double",
            condition="(nw is None) and (nt is None) and (not multi) and (reflected)",
            map_args="lmax"
    ),
    128: dict(
            name="_starry_default_refl_multi",
            condition="(nw is None) and (nt is None) and (multi) and (reflected)",
            map_args="lmax"
    ),
    256: dict(
            name="_starry_spectral_refl_double",
            condition="(nw is not None) and (nw > 0) and (nt is None) and (not multi) and (reflected)",
            map_args="lmax, nw"
    ),
    512: dict(
            name="_starry_spectral_refl_multi",
            condition="(nw is not None) and (nw > 0) and (nt is None) and (multi) and (reflected)",
            map_args="lmax, nw"
    ),
    1024: dict(
            name="_starry_temporal_refl_double",
            condition="(nt is not None) and (nt > 0) and (nw is None) and (not multi) and (reflected)",
            map_args="lmax, nt"
    ),
    2048: dict(
            name="_starry_temporal_refl_multi",
            condition="(nt is not None) and (nt > 0) and (nw is None) and (multi) and (reflected)",
            map_args="lmax, nt"
    )
}


# Load each of the modules
_loader = \
"""try:
    from . import {module}
    if __version__ is None:
        from .{module} import __version__
        __mapdoc__ = {module}.Map.__doc__
except ImportError:
    {module} = None
"""

for bit in _modules.keys():
    exec(_loader.format(module=_modules[bit]["name"]))


# Class factory
def Map(lmax=2, nw=None, nt=None, multi=False, reflected=False):
    for bit in _modules.keys():
        module = _modules[bit]["name"]
        condition = _modules[bit]["condition"]
        map_args = _modules[bit]["map_args"]
        if eval(condition):
            if eval(module) is not None:
                return eval("{module}.Map({map_args})".format(
                    module=module, map_args=map_args))
            else:
                raise ImportError("Module not available. Please " + 
                    "compile `starry` with module bit {bit}.".format(bit=bit))


# Hack the docstring
Map.__doc__ = __mapdoc__