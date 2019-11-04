The starry Python API
=====================

This page describes the two main ways of interfacing with the ``starry`` code.

The ``starry`` Map_ class allows users to quickly model surface maps described
by spherical harmonics. This class inherits from one of the following three base classes:

    - YlmBase_: Spherical harmonic maps of the surface intensity field
    - RVBase_: Spherical harmonic maps of the radial velocity field
    - ReflectedBase_: Spherical harmonic maps in reflected light

The ``starry`` Keplerian module allows users to model systems of objects, each
with their own surface Map_. Users can instantiate the following objects:

    - Primary_: Primary Keplerian objects
    - Secondary_: Secondary (orbiting) Keplerian objects
    - System_: Systems of Keplerian objects

The Map class
-------------

.. _Map:

.. autofunction:: starry.Map

.. _YlmBase:

.. autoclass:: starry.maps.YlmBase()
    :members:

.. _RVBase:

.. autoclass:: starry.maps.RVBase()
    :members:

.. _ReflectedBase:

.. autoclass:: starry.maps.ReflectedBase()
    :members:

The Keplerian module
--------------------

.. _Primary:

.. autoclass:: starry.Primary()
    :members:
    :inherited-members:

.. _Secondary:

.. autoclass:: starry.Secondary()
    :members:
    :inherited-members:

.. _System:

.. autoclass:: starry.System()
    :members:
