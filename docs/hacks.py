# -*- coding: utf-8 -*-
import nbsphinx
import os
import sys
import starry
import re
import packaging
import urllib


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
        """
        pass


class _Map(CustomBase, starry.maps.YlmBase, starry.maps.MapBase):
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


starry._Map = _Map
starry._LimbDarkenedMap = _LimbDarkenedMap
starry._ReflectedLightMap = _ReflectedLightMap
starry._RadialVelocityMap = _RadialVelocityMap


# Hack to figure out if we are the latest version
url = "https://raw.githubusercontent.com/rodluger/starry/gh-pages/versions.txt"
all_versions = []
for line in urllib.request.urlopen(url):
    version_string = line.decode("utf-8").replace("\n", "").strip()
    all_versions.append(packaging.version.parse(version_string))
all_versions = sorted(all_versions)
current_version = packaging.version.parse(starry.__version__)
is_latest = (current_version.is_devrelease) and (
    current_version.base_version >= all_versions[-1].base_version
)
