API Reference
=============

BLiP
----
.. automodule:: pyblip.blip
  :members:

Creating Candidate Groups
-------------------------

Variable selection / discrete locations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: pyblip.create_groups
  :members:

Continuous sets of locations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pyblip.create_groups_cts
  :members:

Edge clique cover
~~~~~~~~~~~~~~~~~

.. automodule:: pyblip.ecc
  :members:

Weight functions
~~~~~~~~~~~~~~~~

.. automodule:: pyblip.weight_fns
  :members:

Bayesian Samplers
-----------------

BLiP can wrap on top of nearly any Bayesian model or algorithm. However,
for convenience, ``pyblip`` includes a few custom MCMC samplers. 

Linear Spike and Slab
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pyblip.linear.linear
  :members:


Probit Spike and Slab
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pyblip.probit.probit
  :members:


Neuronized Priors
~~~~~~~~~~~~~~~~~

.. automodule:: pyblip.nprior.nprior
  :members:

Changepoint Detection
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pyblip.changepoint
  :members:
