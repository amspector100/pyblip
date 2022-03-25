Getting Started
===============

Installation
------------

To install pyblip, use pip:

``pip install pyblip``

What if installation fails?
---------------------------

knockpy relies on heavy-duty linear algebra routines which sometimes fail on non-Linux environments.

1. To start, install a lightweight version of knockpy using
``pip install knockpy``. This should install correctly on all devices, and contains nearly all of the functionality of the prior installation. However, the algorithms for computing optimal distributions for Gaussian knockoffs, such as `minimum reconstructability knockoffs`_ or `SDP knockoffs`_, may be an order of magnitude slower.

.. _minimum reconstructability knockoffs: https://arxiv.org/abs/2011.14625
.. _SDP knockoffs: https://arxiv.org/abs/1610.02351

2. [Optional] To speed up computation for minimum reconstructability knockoffs (the default knockoff type), follow these instructions.
    (a) Run
        ``pip install cython>=0.29.14``.
        If the installation fails, likely due to the incorrect configuration of a C compiler, you have three options. First, the conda_ package manager includes a compiler, so the command
        ``conda install cython``
        should work on all platforms. Second, on Windows, you can install precompiled `binaries for cython`_. Lastly, on all platforms, the `documentation here`_ describes how to properly configure a C compiler during installation.
    (b) Run
        ``pip install git+git://github.com/jcrudy/choldate.git``.

.. _conda: https://docs.anaconda.com/anaconda/user-guide/tasks/install-packages/
.. _binaries for cython: https://www.lfd.uci.edu/~gohlke/pythonlibs/
.. _documentation here: https://cython.readthedocs.io/en/latest/src/quickstart/install.html

3. [Optional] To speed up computation for (non-default) SDP knockoffs, you will need to install ``scikit-dsdp``. This can be challenging on non-Linux environments. We hope to provide more explicit instructions for installation of this package in the future.

