The starry Python API
=====================

.. contents:: :local:


The starry Map
--------------

This is the main interface to ``starry``. Instances returned by
``Map`` are spherical bodies whose surfaces are 
described by spherical harmonics. In the default case, a vector
of spherical harmonic coefficients describes the specific intensity
everywhere on the surface, although it may also describe the
albedo (for maps in reflected light) or the brightness-weighted
radial velocity (for Doppler maps). ``Map`` allows
users to easily and efficiently manipulate the surface representation
and compute intensities and fluxes (light curves) as the object
rotates and becomes occulted by other spherical bodies.

The call sequence for the ``Map`` is as follows:

.. py:function:: starry.Map(ydeg=0, udeg=0, fdeg=0, nt=None, nw=None, reflected=False, doppler=False, multi=False)

    **Keyword Arguments:**

        - **ydeg** (*int*): The highest spherical harmonic degree of the map.
        - **udeg** (*int*): The highest limb darkening degree of the map.
        - **fdeg** (*int*): The highest degree of the custom multiplicative filter applied to the map.
        - **nt** (*int*): The number of temporal components in the map. Cannot be set simultaneously with ``nw``.
        - **nw** (*int*): The number of spectral components in the map. Cannot be set simultaneously with ``nt``.
        - **reflected** (*bool*): If ``True``, performs all calculations in reflected light. The spherical harmonic expansion now corresponds to the *albedo* of the surface
        - **doppler** (*bool*): If ``True``, enables Doppler mode. See :py:class:`DopplerMap` for details.
        - **multi** (*bool*): If ``True``, performs all calculations using multi-precision floating point arithmetic. The number of digits of the multi-precision type is controlled by the ``STARRY_NMULTI`` compile-time constant.

Depending on the values of these arguments, one of three map types
will be instantiated: a :py:class:`SphericalHarmonicMap`, a
:py:class:`LimbDarkenedMap`, or a :py:class:`DopplerMap`. See below
for details on each one of these.


Spherical Harmonic Maps
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: starry.SphericalHarmonicMap


Limb Darkened Maps
~~~~~~~~~~~~~~~~~~

.. autoclass:: starry.LimbDarkenedMap


Doppler Maps
~~~~~~~~~~~~

.. autoclass:: starry.DopplerMap


The extensions module
---------------------

.. automodule:: starry.extensions