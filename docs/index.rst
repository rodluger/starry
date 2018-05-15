.. raw:: html

   <div align="center">
   <img src="_images/starry.png" width="400px">
   </img>
   <br/>
   </div>
   <br/><br/>


Documentation
=============

Welcome to the :py:mod:`starry` documentation. The :py:mod:`starry` code package
enables the computation of light curves for various applications in
astronomy: transits and secondary eclipses of exoplanets, light curves of
eclipsing binaries, rotational phase curves of exoplanets, light curves of
planet-planet and planet-moon occultations, and more. By modeling celestial
body surface maps as sums of spherical harmonics, :py:mod:`starry` does all
this **analytically** and is therefore fast, stable, and differentiable. Coded in
C++ but wrapped in Python, :py:mod:`starry` is easy to install and use.

.. raw:: html

   <div align="center">
   <img src="_images/earthoccultation.jpg" width="600px">
   </img>
   <br/>
   <span style="font-weight:bold">Figure 1.</span> Occultation of the Earth by
   the Moon computed analytically using starry.
   </div>
   <br/><br/>

For a more detailed introduction to :py:mod:`starry`, including the theory behind the
equations and information about how to use the code in your own research,
check out the
`latest draft of the paper <https://github.com/rodluger/starry/raw/master-pdf/tex/starry.pdf>`_.


Index
=====

Check out the links below for examples, tutorials, detailed documentation of the
API, and more.

.. toctree::
   :maxdepth: 1

   Installation <install>
   Examples & tutorials <tutorials>
   API <api>
   Github <https://github.com/rodluger/starry>
   Submit an issue <https://github.com/rodluger/starry/issues>
   Read the paper <https://github.com/rodluger/starry/raw/master-pdf/tex/starry.pdf>
