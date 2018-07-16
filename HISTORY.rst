.. :changelog:

0.1.2 (2018-07-15)
++++++++++++++++++

- Initial stable beta release of the code for submission to ApJ.
- Implements simple `Map` and `LimbDarkenedMap` surface maps for arbitrary
  surface maps and radial (limb-darkened) surface maps, respectively. Also
  implements a simple `System` class for modeling light curves of
  Keplerian star-planet systems.
- Supports gradient computation via autodifferentiation in the `grad` module
  and multi-precision floating point calculations in the `multi` module.
