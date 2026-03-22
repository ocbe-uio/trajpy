Installation
============

Version |release| is the current stable release.

TrajPy
------

TrajPy is hosted on PyPI. Install it with pip:

.. code-block:: bash

   pip install trajpy

or with `uv <https://github.com/astral-sh/uv>`_:

.. code-block:: bash

   uv add trajpy

Graphical User Interface (trajpy-ui)
-------------------------------------

An optional graphical user interface is available as a separate package:

.. code-block:: bash

   pip install trajpy-ui

or with uv:

.. code-block:: bash

   uv add trajpy-ui

Development version
-------------------

To install the latest development version, clone the repository and install
in editable mode:

.. code-block:: bash

   git clone https://github.com/ocbe-uio/trajpy
   cd trajpy
   uv sync

Dependencies
------------

Core runtime dependencies are listed in ``pyproject.toml``.
The docs extras (Sphinx, nbsphinx, etc.) can be installed with:

.. code-block:: bash

   uv sync --group docs
