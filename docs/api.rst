API
===

.. hlist::
    :columns: 1

    * Intro_
    * Map_
    * Primary_
    * Secondary_
    * System_

.. _Intro:

Introduction
------------

This page documents the :py:mod:`starry` API, which is coded
in C++ with a :py:mod:`pybind11` Python interface. The API consists
of a :py:class:`Map` class, which houses all of the surface map photometry
stuff, and a :py:class:`kepler` module, which connects photometry to
dynamics via a simple (multi-)Keplerian solver.

There are two broad ways in which users can access
the core :py:mod:`starry` functionality:

    - Users can instantiate a :py:class:`Map` class to compute phase curves
      and occultation light curves by directly specifying the rotational state
      of the object and (optionally) the position and size of an occultor.
      Users can specify spherical harmonic coefficients, limb
      darkening coefficients, or both. There is also support for both
      monochromatic and wavelenght-dependent maps and light curves.
      This class may be particularly useful for users who wish to
      integrate :py:mod:`starry` with their own dynamical code or for users
      wishing to compute simple light curves without any orbital solutions.

    - Users can instantiate a :py:class:`kepler.Primary` and one or more
      :py:class:`kepler.Secondary` objects and feed them into a
      :py:class:`kepler.System` instance for integration
      with the Keplerian solver. The :py:class:`kepler.Primary`
      and :py:class:`kepler.Secondary` classes inherit from :py:class:`Map`,
      so users have access to all of the functionality of the previous
      method.

At present, :py:mod:`starry` uses a simple Keplerian solver to compute orbits,
so the second approach listed above is limited to two-body systems or systems
where the secondary masses are negligible. Stay tuned for the next release
of the code, which will incorporate an N-body solver.

.. _Map:

.. autoclass:: starry.Map

.. _Primary:

.. autoclass:: starry.kepler.Primary

.. _Secondary:

.. autoclass:: starry.kepler.Secondary

.. _System:

.. autoclass:: starry.kepler.System
