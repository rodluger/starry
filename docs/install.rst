Installation
============

.. warning:: The code has not yet been uploaded to PyPI, so :py:obj:`pip` won't work just yet.

Using pip
---------

The easiest way to install :py:obj:`starry` is using the
`pip <https://pip.pypa.io/en/stable/installing/>`_ command:

.. code-block:: bash

    pip install starry

This will download, compile, and install :py:obj:`starry`. Depending on your machine,
compiling may take a couple minutes. Once that's done, open a :py:obj:`python` terminal,
:py:obj:`import starry`, and hack away!


Development version
-------------------

Alternatively, you can install the development version of :py:obj:`starry` directly
from `GitHub <https://github.com/rodluger/starry>`_:

.. code-block:: bash

    git clone https://github.com/rodluger/starry.git
    cd starry
    python setup.py develop


Custom builds
-------------

Some users may want to change the values of certain compile-time constants.
For instance, by default :py:obj:`starry` computes the gradient of the flux
with respect to at most 43 parameters. This is almost certainly overkill
for most applications, but if you really need more derivatives, you'll have
to change the :py:obj:`STARRY_NGRAD` compiler flag and re-build the code.
To do this, you'll need to clone the development version from github:

.. code-block:: bash

        git clone https://github.com/rodluger/starry.git
        cd starry
        STARRY_NGRAD=XX python setup.py install


where :py:obj:`XX` is the number of gradients you wish to
compute. Keep in mind that the more gradients you ask :py:obj:`starry` to
compute, the slower the code will run. If you really need performance in
autodiff mode, you could potentially ask :py:obj:`starry` to compute fewer
gradients (provided your map is low-degree and you only have one or maybe
two planets in your system).
