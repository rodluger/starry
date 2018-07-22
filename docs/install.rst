Installation
============

Using conda
-----------

The easiest way to install :py:obj:`starry` is using 
`conda <https://conda.io/>`_ via `conda-forge <https://conda-forge.org/>`_:

.. code-block:: bash

    conda install -c conda-forge starry

Using pip
---------

The next easiest way to install :py:obj:`starry` is using the
`pip <https://pip.pypa.io/en/stable/installing/>`_ command:

.. code-block:: bash

    pip install starry

This will download, compile, and install :py:obj:`starry`. Depending on your machine,
compiling may take a couple minutes. Once that's done, open a :py:obj:`python` terminal,
:py:obj:`import starry`, and hack away!


Development version
-------------------

You can install the development version of :py:obj:`starry` directly
from `GitHub <https://github.com/rodluger/starry>`_:

.. code-block:: bash

    git clone https://github.com/rodluger/starry.git
    cd starry
    python setup.py develop


Custom builds
-------------

Some users may want to change the values of certain compile-time constants.
For instance, by default :py:obj:`starry` computes the gradient of the flux
with respect to at most 13 parameters. This should be plenty
for most applications, but if you really need more derivatives, you'll have
to change the :py:obj:`STARRY_NGRAD` compiler flag and re-build the code.
To do this, you'll need to clone the development version from github:

.. code-block:: bash

        git clone https://github.com/rodluger/starry.git
        cd starry
        STARRY_NGRAD=XX python setup.py install


where :py:obj:`XX` is the number of gradients you wish to
compute. Keep in mind that the more gradients you ask :py:obj:`starry` to
compute, the slower the code will run.
