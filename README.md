# pyblip: A python implementation of BLiP

## Introduction

In many applications, we can tell that a signal of interest exists but cannot perfectly "localize" it. For example, when regressing an outcome *Y* on highly correlated covariates (X<sub>1</sub>, X<sub>2</sub>), the data may suggest that *at least* one of (X<sub>1</sub>, X<sub>2</sub>) influences *Y*, but it may be challenging to tell which of (X<sub>1</sub>, X<sub>2</sub>) is important. Likewise, in genetic fine-mapping, biologists may have high confidence that a gene influences a disease without knowing precisely which genetic variants cause the disease. Similar problems arise in many settings with spatial or temporal structure, including change-point detection and astronomical point-source detection.

*Bayesian Linear Programming* (BLiP) is a method which jointly detects as many signals as possible while localizing them as precisely as possible. BLiP can wrap on top of nearly any Bayesian model or algorithm, and it will return a set of regions which each contain at least one signal with high confidence. For example, in regression problems, BLiP might return the region (X<sub>1</sub>, X<sub>2</sub>), which suggests that at least one of (X<sub>1</sub>, X<sub>2</sub>) is an important variable. BLiP controls the false discovery rate while also making these regions as narrow as possible, meaning that (roughly speaking) it will perfectly localize signals whenever this is possible! 

``pyblip`` is an efficient python implementation of BLiP, which is designed so that BLiP can wrap on top of the Bayesian model in only one or two lines of code. For convenience, it also includes fast Bayesian samplers for linear regression, probit regresion, and change point detection, since BLiP can wrap on top of any of these methods.

## Installation

You can install ``pyblip`` using pip:

	``pip install pyblip``

### Performance improvement

BLiP uses [cvxpy](https://www.cvxpy.org/install/index.html) to efficiently solve linear programs. ``pyblip`` will install and run correctly using the default ``cvxpy`` solvers, but to improve performance, we also recommend installing the Cbc solver. See [this link](https://www.cvxpy.org/install/index.html#install-with-cbc-clp-cgl-support) for system-specific instructions to install Cbc.

### Installation issues

``pip`` should automatically install ``pyblip``'s dependencies. However, if installation fails, it is likely for one of three reasons:

1. You are having trouble installing ``cvxpy``
	
	By default, installing ``cvxpy`` also installs several heavy-duty convex solvers, and installation of these solvers can fail. However, ``pyblip`` only requires a few solvers. As a result, it can be easier to install a lightweight version of ``cvxpy`` with the following command:
		``pip install cvxpy-base``
	and then install the ``SCS`` or ``CBC`` solvers. Please see the [cvxpy installation instructions](https://www.cvxpy.org/install/index.html) for more details.

2. You are having trouble installing ``cython``.

	The Bayesian samplers in ``pyblip`` are written in cython to improve performance. If your system is having trouble installing cython, see the [cython website](https://cython.org/) for instructions.

3. You are having trouble installing ``cvxopt``.

	``pyblip`` requires ``cvxopt`` because installing ``cvxopt`` is often the easiest way to get access to a solver for mixed-integer linear programs. If you are having trouble installing ``cvxopt``, you can avoid this problem by specifying ``deterministic=False`` whenever running BLiP, and you will not need a mixed-integer solver. Alternatively, you can follow the instructions on the [cvxpy website](https://www.cvxpy.org/install/index.html) to install any other mixed-integer solver. For example, if you install ``CBC`` as described earlier, you do not need ``cvxopt``.

## Quick start

Here, we apply BLiP to perform variable selection in a sparse linear regression problem. The first step is to generate synthetic data and fit the Bayesian model.

```
	# Synthetic regression data with AR1 design matrix
	import numpy as np
	import scipy.linalg
	n, p, nsignals, rho = 100, 500, 20, 0.95
	c = np.cumsum(np.zeros(p) + np.log(rho)) - np.log(rho)
	cov = scipy.linalg.toeplitz(np.exp(c))
	X = np.dot(np.random.randn(n, p), np.linalg.cholesky(cov).T)

	# Sparse coefficients for linear model
	beta = np.zeros(p)
	signal_locations = np.random.choice(np.arange(p), nsignals)
	beta[signal_locations] = np.random.randn(nsignals)
	y = np.dot(X, beta) + np.random.randn(n)

	# Spike-and-slab bayesian sampling
	import pyblip
	lm = pyblip.linear.LinearSpikeSlab(
		X=X, y=y, 
	)
	lm.sample(N=1000, chains=10)
```

The second step is to apply BLiP directly on top of the posterior samples of the linear coefficients:

```
	detections = pyblip.blip.BLiP(
		samples=lm.betas,
		q=0.1,
		error='fdr'
	)
	for x in detections:
		print("BLiP has detected a signal among {x.group}!")
```

Please see [amspector100.github.io/pyblip/usage.html](https://amspector100.github.io/pyblip/usage.html) for examples ranging from variable selection to change-point detection to astronomical point-source detection. 

## Documentation

Documentation is available at [amspector100.github.io/pyblip](https://amspector100.github.io/pyblip).

## Reference

If you use ``pyblip`` or BLiP in an academic publication, please consider citing Spector and Janson (2022). The bibtex entry is below:


```
@article{AS-LJ:2022,
  title={Controlled Discovery and Localization of Signals via Bayesian Linear Programming},
  author={Spector, Asher and Janson, Lucas},
  journal={arXiv preprint arXiv:2203.17208},
  url={https://arxiv.org/abs/2203.17208},
  year={2022}
}
```