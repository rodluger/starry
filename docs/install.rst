Installation
============

Using conda
-----------

The easiest way to install :py:obj:`starry` is using
`conda <https://conda.io/>`_ via `conda-forge <https://conda-forge.org/>`_.
First, ensure you have the :py:obj:`conda-forge` channel enabled:

.. code-block:: bash

    conda config --add channels conda-forge

Once that's done, installing :py:obj:`starry` is easy:

.. code-block:: bash

    conda install starry


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
