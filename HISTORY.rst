.. :changelog:

1.1.0 (2021-02-25)
++++++++++++++++++

- Implements phase curves and occultations in reflected light
- Fixes several compatibility issues with ``theano`` and ``pymc3``
- Removes all dependencies on ``healpy``; now computing spherical
  harmonic transforms in-house
- Deprecates the ``add_spot`` method in favor of a new ``spot`` method
  adapted from the algorithm developed in ``starry_process``
- Makes ``starry`` ``Map`` objects picklable
- Allows the ``show`` method to be called within a ``pymc3`` model
- Fixes an issue with RVs for multi-planet systems
- Fixes several small bugs, closes several GitHub issues

1.0.0 (2019-12-26)
++++++++++++++++++

- First official release of the code.
- Major changes to the backend, which is now mostly written on top
  of Theano and uses lazy evaluation by default.
- Easy integration with `pymc3` and `exoplanet` for inference problems.
- New features, such as radial velocity and reflected light modeling.
- Lots of other stuff; see the docs for details.

0.3.0 (2018-12-30)
++++++++++++++++++

- Release of the code for the published version of the paper.
- Sped up gradient I/O for limb-darkened light curves.
- Fixed bug that led to huge (> 100 MB) binary sizes on Linux
  machines.
- Small miscellaneous bugfixes.

0.2.2 (2018-10-14)
++++++++++++++++++

- Third release of the code for submission to the arXiv.
- Implemented the faster equations for polynomial limb-darkening
  from Agol & Luger (2018, in prep).
- Implemented gradients of planet-planet occultation light curves
  and added complete benchmarking of all derivatives in the `System`
  class.
- Fixed memory issue in the `System` class that de-allocated the
  `Primary` and `Secondary` objects if they went out of scope on
  the Python side.
- Small miscellaneous bugfixes.

0.2.1 (2018-10-01)
++++++++++++++++++

- Second release of the code for resubmission to AJ.
- Major code re-write. Redesigned user interface, easier to use,
  faster, and more flexible.
- Maps are now instantiated via a single `Map` object that supports
  both spherical harmonic coefficients and limb darkening coefficients,
  as well as arbitrary combinations of both.
- The `Map` class now accepts a `nwav` keyword specifying the number of
  wavelength bins (default 1). This allows users to easily and efficiently
  compute wavelength-dependent light curves from spectral surface maps.
- The `Primary` and `Secondary` classes replaced the old `Star` and
  `Planet` classes for increased generality. These are now actual subclasses
  of `Map`, making them even easier to use.
- Several modifications to the computation of the light curves were made to
  increase speed. The largest speed improvement is in the computation of
  gradients. The code is no longer limited by the `STARRY_NGRAD` parameter,
  and arbitrary number of gradients can now be taken efficiently.
- Derivatives with respect to the map coefficients are no longer labeled
  as `Y_{0,0}`, `Y_{1,-1}`, ... Instead, they are labeled by a single key
  `y`, whose value is a vector of derivatives corresponding to each element
  in the spherical harmonic vector `y`. The same applies to the limb
  darkening coefficients, whose derivatives are stored in they key `u`.
- Minor changes to the names of some keywords and class properties, and
  to some call sequences.
- Minor bug fixes here and there.


0.1.2 (2018-07-15)
++++++++++++++++++

- Initial stable beta release of the code for submission to AJ.
- Implements simple `Map` and `LimbDarkenedMap` surface maps for arbitrary
  surface maps and radial (limb-darkened) surface maps, respectively. Also
  implements a simple `System` class for modeling light curves of
  Keplerian star-planet systems.
- Supports gradient computation via autodifferentiation in the `grad` module
  and multi-precision floating point calculations in the `multi` module.
