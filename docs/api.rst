The starry API
==============

Below you'll find links to the :py:mod:`starry` API and all functionality
accessible to the user. There are three flavors of the API: plain vanilla
:py:mod:`starry`, which computes light curves, phase curves, and all the
other good stuff; :py:mod:`starry.grad`, which
does all that **and** computes gradients with respect to the input
parameters using (analytical) autodifferentiation; and
:py:mod:`starry.multi`, which computes everything using multi-precision
floating point arithmetic (and is almost certainly overkill).

.. toctree::
   :maxdepth: 2

   starry <starry>
   starry.grad <starry_grad>
   starry.multi <starry_multi>
