New in version 1.0.0
====================

The first release version of ``starry`` differs quite a bit from the beta
(``0.3.0``) version, particularly in how gradients are computed and propagated.
The main difference is that ``starry`` now uses ``theano`` under the hood for all
calculations, which allows it to interface efficiently with ``exoplanet`` and
``pymc3``. From the user's perspective the only major change is that calculations
are performed **lazily** by default. What that means is that whenever the user
calls a method like ``flux()`` or ``intensity()``, the code will return a node in
a graph rather than an actual numerical value. Operations can then be done on
these nodes (such as computing the likelihood in an inference problem), with
gradients automatically (back-)propagated along the way. This is exactly how
``exoplanet`` and ``pymc3`` work, so the whole inference problem is made much easier.

Significant changes
~~~~~~~~~~~~~~~~~~~

Lazy evaluation
^^^^^^^^^^^^^^^
By default, all operations are done lazily, and all map properties are stored as
``theano`` nodes rather than numerical values. This behavior can be changed by
setting ``starry.config.lazy = False`` as soon as ``starry`` is imported
(before any maps have been instantiated). Doing so will automatically compile
all functions behind the scenes, and the code will function similarly to version
``0.3.0``.

The frame in which the coefficients are defined
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Spherical harmonic coefficients are now defined in a static frame that is
independent of the orientation of the map. Changing the rotational axis of the
map (which is now done by specifying the inclination and obliquity of the object) 
no longer changes the :math:`Y_{l,m}` coefficients. Another way to think about this 
is that changing the inclination and obliquity corresponds to moving the 
observer around, *not* rotating the map.

Normalization of limb-darkened maps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The normalization of limb-darkened spherical harmonic maps has changed. 
The normalization is such that the total luminosity of the map remains unchanged 
when limb darkening is applied or when the limb darkening coefficients are 
modified. Note that this could mean that the flux at any given viewing angle 
will be different from the non-limb-darkened version of the map. This is 
significantly different from the beta version of the code, which normalized 
it such that the flux remained unchanged. That normalization was almost certainly 
unphysical.

Linearized computations
^^^^^^^^^^^^^^^^^^^^^^^
All flux and intensity computations are now explicitly linear in the spherical 
harmonic coefficients. This allows users to easily access the ``design_matrix`` 
that transforms from a vector of spherical harmonic coefficients to a light 
curve. This matrix is extremely useful, as it lets you tackle the inverse 
problem analytically!

Fixed :math:`Y_{0,0}` coefficient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The constant (:math:`Y_{0,0}`) coefficient is now fixed at unity. The previous 
version allowed the user to change this, which effectively changed the 
luminosity of the map. There is now an explcit luminosity ``L`` attribute 
that can be changed instead.

Smaller changes
~~~~~~~~~~~~~~~

Deprecation of the ``axis`` attribute
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The axis of rotation is now specified by the inclination ``inc`` and obliquity 
``obl`` of the map (see above).

Deprecation of ``load_image`` in favor of ``load``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Just a name change.

Deprecation of ``add_gaussian`` in favor of ``add_spot``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Also a name change, but the algorithm used to compute the spherical harmonic 
expansion of the spot is now analytic and much, much faster.

Deprecation of the ``animate`` method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
All map visualizations are now performed via the ``show`` method. If you want
to animate the map as it rotates, pass a vector-valued ``theta``.

Deprecation of the ``__call__`` method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the previous version, users could evaluate the intensity at a point on the
surface of a map by calling the instance directly: ``map(x=0, y=0)``. This is
now done via the ``intensity`` method, whose arguments are the latitude 
``lat`` and longitude ``lon`` of the point(s) rather than their Cartesian
coordinates.

Distinction between ``ydeg`` and ``udeg``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Previously, maps were instantiated for a given maximum degree ``lmax``. 
This has been deprecated in favor of a maximum :math:`Y_{l,m}` degree ``ydeg`` 
and a maximum limb darkening degree ``udeg``.

Custom units
^^^^^^^^^^^^
Users can now pass custom ``astropy`` units to override the defaults.

Faster linear algebra
^^^^^^^^^^^^^^^^^^^^^
The change of basis matrices between spherical harmonics and polynomials are 
now computed recursively, rather than via the horrendous nested sums presented 
in the paper. The new method is more elegant and more stable.

New stuff
~~~~~~~~~

Reflected light maps
^^^^^^^^^^^^^^^^^^^^
Maps can now be modelled in reflected light. The spherical harmonic 
coefficients now correspond to the surface albedo, rather than its specific 
intensity.

Doppler maps
^^^^^^^^^^^^
The ``Map`` class can now also model radial velocity observations. This is 
useful for modeling the Rossiter-McLaughlin effect, for example.

Differential rotation (experimental)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Users can now specify the differential rotation parameter ``alpha``
to model weak differential rotation over short timescales. Unfortunately,
differential rotation is not a linear operation on the spherical harmonic
coefficient vector, since the shearing induces higher order modes that
grow strongly with time. The version implemented in ``starry`` is just a 
low-order approximation to differential rotation that works in cases where
both ``alpha`` and the number of rotations are small.