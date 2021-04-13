# -*- coding: utf-8 -*-
import nbsphinx
import starry
import re
from sphinx.ext import autodoc


# Hack `nbsphinx` to enable us to hide certain input cells in the
# jupyter notebooks. This works with nbsphinx==0.5.0
nbsphinx.RST_TEMPLATE = nbsphinx.RST_TEMPLATE.replace(
    "{% block input -%}",
    '{% block input -%}\n{%- if not "hide_input" in cell.metadata.tags %}',
)
nbsphinx.RST_TEMPLATE = nbsphinx.RST_TEMPLATE.replace(
    "{% endblock input %}", "{% endif %}\n{% endblock input %}"
)

# Hack `nbsphinx` to prevent fixed-height images, which look
# terrible when the window is resized!
nbsphinx.RST_TEMPLATE = re.sub(
    r"\{%- if height %\}.*?{% endif %}",
    "",
    nbsphinx.RST_TEMPLATE,
    flags=re.DOTALL,
)

# Hack the docstrings of the different base maps. This allows us
# to have a different docs page for radial velocity maps, reflected
# light maps, etc, even though those classes are instantiated via
# a class factory and don't actually exist in the starry namespace.
class CustomBase(object):
    @property
    def amp(self):
        """
        The overall amplitude of the map in arbitrary units. This factor
        multiplies the intensity and the flux and is thus proportional to the
        luminosity of the object. For multi-wavelength maps, this is a vector
        corresponding to the amplitude of each wavelength bin.
        For reflected light maps, this is the average spherical albedo
        of the body.
        """
        pass


class _SphericalHarmonicMap(
    CustomBase, starry.maps.YlmBase, starry.maps.MapBase
):
    __doc__ = starry.maps.YlmBase.__doc__


class _LimbDarkenedMap(
    CustomBase, starry.maps.LimbDarkenedBase, starry.maps.MapBase
):
    __doc__ = starry.maps.LimbDarkenedBase.__doc__


class _ReflectedLightMap(
    CustomBase, starry.maps.ReflectedBase, starry.maps.MapBase
):
    __doc__ = starry.maps.ReflectedBase.__doc__


class _RadialVelocityMap(CustomBase, starry.maps.RVBase, starry.maps.MapBase):
    __doc__ = starry.maps.RVBase.__doc__


class _OblateMap(CustomBase, starry.maps.OblateBase, starry.maps.MapBase):
    __doc__ = starry.maps.OblateBase.__doc__


starry._SphericalHarmonicMap = _SphericalHarmonicMap
starry._LimbDarkenedMap = _LimbDarkenedMap
starry._ReflectedLightMap = _ReflectedLightMap
starry._RadialVelocityMap = _RadialVelocityMap
starry._OblateMap = _OblateMap

replace = {
    "_SphericalHarmonicMap": "Map",
    "_LimbDarkenedMap": "Map",
    "_ReflectedLightMap": "Map",
    "_RadialVelocityMap": "Map",
    "_OblateMap": "Map",
}


def format_name(self) -> str:
    name = ".".join(self.objpath) or self.modname
    for old_name, new_name in replace.items():
        name = name.replace(old_name, new_name)
    return name


# This hack works with sphinx==2.2.0
autodoc.Documenter.format_name = format_name
