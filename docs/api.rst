The starry API
==============

.. toctree::
   :maxdepth: 1

   starry <starry>


.. raw:: html

    <p style="padding-left: 30px;">
    This is the main starry module. Here you can compute light curves,
    phase curves, and all the other good stuff. This is probably where you want to
    start poking around.
    </p>


.. toctree::
   :maxdepth: 1

   starry.grad <starry_grad>


.. raw:: html

    <p style="padding-left: 30px;">
    This is the autodifferentation-enabled starry interface. It does
    everything the regular starry module does, but it also analytically
    computes gradients of the light curve with respect to all of the input
    parameters. It is in general slower by about an order of magnitude, but can
    greatly speed up inference and optimization problems that require knowledge
    of the gradient of your model.
    </p>


.. toctree::
   :maxdepth: 1

   starry.multi <starry_multi>


.. raw:: html

    <p style="padding-left: 30px;">
    This is the multiprecision-enabled starry interface. It also does
    everything the regular starry module does, except it computes everything
    using multi-precision floating point arithmetic (and is almost certainly overkill).
    It is also slower than the regular interface by (at least) an order of magnitude.
    In general this module is used for stability tests and debugging, or if you have
    a <span style="font-weight:bold;">really</span> high degree map you need to compute light curves for.
    </p>
