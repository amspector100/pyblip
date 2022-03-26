Getting Started
===============

Installation
------------

To install pyblip, use pip:

``pip install pyblip``

Performance improvement with CBC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BLiP uses `cvxpy`_ to efficiently solve linear programs. ``pyblip`` will install and run correctly using the default ``cvxpy`` solvers, but to improve performance, we also recommend installing the Cbc solver.  See `this link`_ for system-specific instructions to install Cbc.

.. _cvxpy: https://www.cvxpy.org/install/index.html
.. _this link: https://www.cvxpy.org/install/index.html#install-with-cbc-clp-cgl-support

Installation issues
~~~~~~~~~~~~~~~~~~~

``pip`` should automatically install ``pyblip``'s dependencies. However, if installation fails, it is likely for one of three reasons:

1. You are having trouble installing ``cvxpy``.
    
    By default, installing ``cvxpy`` also installs several heavy-duty convex solvers, and installation of these solvers can fail. However, ``pyblip`` only requires a few solvers. As a result, it can be easier to install a lightweight version of ``cvxpy`` with the following command:

    ``pip install cvxpy-base``
    
    and then install the ``SCS`` or ``CBC`` solvers. Please see the `cvxpy installation instructions`_ for more details.

2. You are having trouble installing ``cython``.

    The Bayesian samplers in ``pyblip`` are written in cython to improve performance. If your system is having trouble installing cython, see the `cython website`_ for instructions.

3. You are having trouble installing ``cvxopt``.

    ``pyblip`` requires ``cvxopt`` because installing ``cvxopt`` is often the easiest way to get access to a solver for mixed-integer linear programs. If you are having trouble installing ``cvxopt``, you can avoid this problem by specifying ``deterministic=False`` whenever running BLiP, and you will not need a mixed-integer solver. Alternatively, you can follow the instructions on the `cvxpy website`_ to install any other mixed-integer solver. For example, if you install ``CBC`` as described earlier, you do not need ``cvxopt``.

.. _cvxpy installation instructions: https://www.cvxpy.org/install/index.html
.. _cython website: https://cython.org/
.. _cvxpy website: https://www.cvxpy.org/install/index.html

Minimal example
---------------

Here, we apply BLiP to perform variable selection in a sparse linear regression problem. The first step is to generate synthetic data and fit the Bayesian model.

.. code-block:: python

    # Synthetic regression data
    import pyblip
    X, y, _ = pyblip.utilities.generate_regression_data(
        n=100, p=200, sparsity=0.05,
    )

    # Step 1: fit sparse Bayesian regression model
    lm = pyblip.linear.LinearSpikeSlab(
        X=X, y=y, 
    )
    lm.sample(N=1000, chains=10)

The second step is to apply BLiP directly on top of the posterior samples of the linear coefficients:

.. code-block:: python

    # Step 2: run BLiP
    detections = pyblip.blip.BLiP(
        samples=lm.betas,
        q=0.1,
        error='fdr'
    )
    for x in detections:
        print("BLiP has detected a signal among {x.group}!")
