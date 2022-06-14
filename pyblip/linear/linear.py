import time
import numpy as np
import scipy as sp
from scipy import linalg
from scipy import stats
from ..utilities import apply_pool
from ._linear import _sample_spikeslab
from ._linear_multi import _sample_spikeslab_multi

class LinearSpikeSlab():
	"""

	Spike-and-slab model for linear regression.

	Parameters
	----------
	X : np.array
		``(n,p)``-shaped design matrix.
	y : np.array
		``n``-length array of responses.
	p0 : float
		Prior probability that any coefficient equals zero.
	update_p0 : bool
		If True, updates ``p0`` using a Beta hyperprior on ``p0``.
		Else, the value of ``p0`` is fixed.
	p0_a0 : float
		If ``update_p0`` is True, ``p0`` has a
		Beta(``p0_a0``, ``p0_b0``, ``min_p0``) hyperprior.
	p0_b0 : float
		If ``update_p0`` is True, ``p0`` has a
		TruncBeta(``p0_a0``, ``p0_b0``, ``min_p0``) hyperprior.
	min_p0 : float
		Minimum value for ``p0`` as specified by the prior.
	sigma2 : float
		Variance of y given X.
	update_sigma2 : bool
		If True, imposes an InverseGamma hyperprior on ``sigma2``.
		Else, the value of ``sigma2`` is fixed.
	sigma2_a0 : float
		If ``update_sigma2`` is True, ``sigma2`` has an
		InvGamma(``sigma2_a0``, ``sigma2_b0``) hyperprior.
	sigma2_b0 : float
		If ``update_sigma2`` is True, ``sigma2`` has an
		InvGamma(``sigma2_a0``, ``sigma2_b0``) hyperprior.
	tau2 : float
		Prior variance on nonzero coefficients.
	update_tau2 : bool
		If True, imposes an InverseGamma hyperprior on ``tau2``.
		Else, the value of ``tau2`` is fixed.
	tau2_a0 : float
		If ``update_sigma2`` is True, ``tau2`` has an
		InvGamma(``tau2_a0``, ``tau2_b0``) hyperprior.
	tau2_b0 : float
		If ``update_sigma2`` is True, ``tau2`` has an
		InvGamma(``tau2_a0``, ``tau2_b0``) hyperprior.

	Methods
	-------
	sample:
		Samples from the posterior using Gibbs sampling.
	"""

	def __init__(
		self,
		X,
		y,
		p0=0.9,
		p0_a0=1,
		p0_b0=1,
		update_p0=True,
		min_p0=0,
		sigma2=1,
		update_sigma2=True,
		sigma2_a0=2,
		sigma2_b0=1,
		tau2=1,
		tau2_a0=2,
		tau2_b0=1,
		update_tau2=True,
	):
		self.X = X
		# ensure contiguous
		if not self.X.flags['C_CONTIGUOUS']:
			self.X = np.ascontiguousarray(self.X)
		self.y = y
		# sigma2
		self.sigma2 = sigma2
		self.sigma2_a0 = sigma2_a0
		self.sigma2_b0 = sigma2_b0
		self.update_sigma2 = update_sigma2
		# tau2
		self.tau2 = tau2
		self.tau2_a0 = tau2_a0
		self.tau2_b0 = tau2_b0
		self.update_tau2 = update_tau2
		# p0
		self.p0 = p0
		self.p0_a0 = p0_a0
		self.p0_b0 = p0_b0
		self.update_p0 = update_p0
		self.min_p0 = min_p0

	def sample(
		self,
		N,
		burn=100, 
		chains=1, 
		num_processes=1, 
		bsize=1,
		max_signals_per_block=None,
	):
		"""
		N : int
			Number of samples per chain
		burn : int
			Number of samples to burn per chain
		chains : int
			Number of chains to run
		num_processes : int
			How many processes to use
		bsize : int
			Maximum block size within gibbs sampling. Default: 1.
		max_signals_per_block : int
			Maximum number of signals allowed per block. Default: None
			(this places no restrictions on the number of signals per block).
			The default is highly recommended.
		"""
		z = np.zeros(1).astype(int) # dummy variable
		constant_inputs=dict(
			X=self.X,
			y=self.y,
			z=z,
			probit=False,
			tau2=self.tau2,
			update_tau2=self.update_tau2,
			tau2_a0=self.tau2_a0,
			tau2_b0=self.tau2_b0,
			sigma2=self.sigma2,
			update_sigma2=self.update_sigma2,
			sigma2_a0=self.sigma2_a0,
			sigma2_b0=self.sigma2_b0,
			p0=self.p0,
			update_p0=self.update_p0,
			min_p0=self.min_p0,
			p0_a0=self.p0_a0,
			p0_b0=self.p0_b0,
		)
		# Add block size in and decide underlying function call
		bsize = min(bsize, self.X.shape[1])
		if bsize > 1:
			fn = _sample_spikeslab_multi
			constant_inputs['bsize'] = bsize
			if max_signals_per_block is None:
				max_signals_per_block = 0
			constant_inputs['max_signals_per_block'] = max_signals_per_block
		else:
			fn = _sample_spikeslab

		out = apply_pool(
			fn,
			constant_inputs=constant_inputs,
			N=[N+burn for _ in range(chains)],
			num_processes=num_processes
		)
		self.betas = np.concatenate([x['betas'][burn:] for x in out])
		self.p0s = np.concatenate([x['p0s'][burn:] for x in out])
		self.tau2s = np.concatenate([x['tau2s'][burn:] for x in out])
		self.sigma2s = np.concatenate([x['sigma2s'][burn:] for x in out])


