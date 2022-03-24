import time
import numpy as np
import scipy as sp
from scipy import linalg
from scipy import stats
from knockpy.utilities import apply_pool # TODO deal with this later

from ._nprior import _nprior_sample


class NPrior():
	"""
	Implements Neuronized Prior sampler for spike-and-slab
	regression.

	Parameters
	----------

	X : np.array
		``(n,p)``-shaped design matrix
	y : np.array
		``(n,)``-shaped vector of response data
	p0 : float
		The initial parameter for the proportion of nulls.
		Defaults to 1 - min(0.01, 1/p).
	update_p0 : bool
		If true, will update p0 throughout MCMC sampling using
		a uniform hyperprior.
	min_p0 : float
		If updating p0 throughout uniform sampling, will force 
		p0 to be above ``min_p0``. This can dramatically speed
		up computation in very high-dimensional settings.
	sigma_prior_type : integer
		If 0, assumes sigma2 is conditionally independent of the
		coefficients given the residuals.
	tauw2 : float
		prior variance of the weight parameter
	a0 : float
		sigma2 has an inverse-gamma prior with parameters a0, b0
	b0 : float
		sigma2 has an inverse-gamma prior with parameters a0, b0

	Notes
	-----

	See https://arxiv.org/pdf/1810.00141.pdf.
	"""

	def __init__(
		self,
		X,
		y, 
		tauw2,
		p0=None,
		update_p0=True,
		min_p0=1e-10,
		sigma_prior_type=0,
		sigma_a0=5,
		sigma_b0=1,
		alpha0_a0=1,
		alpha0_b0=1
	):

		# Save parameters and data
		self.n = X.shape[0]
		self.p = X.shape[1]
		self.X = X
		self.y = y
		self.tauw2 = tauw2
		self.sigma_prior_type = sigma_prior_type
		self.sigma_a0 = sigma_a0
		self.sigma_b0 = sigma_b0
		self.alpha0_a0 = alpha0_a0
		self.alpha0_b0 = alpha0_b0
		if p0 is None:
			p0 = 1 - min(0.01, 1/self.p)
		self.p0_init = p0
		self.update_p0 = update_p0
		self.min_p0 = min_p0

		# Pre-initialization
		self.XT = X.T
		self.Xl2s = np.power(X, 2).sum(axis=0)

	def sample(
		self,
		N=100,
		burn=10,
		chains=1,
		num_processes=1,
		joint_sample_W=True,
		group_alpha_update=True,
		log_interval=None
	):
		"""
		Parameters
		----------
		N : int
			The number of samples to draw from the chain
		burn : int
			The burn-in period for each chain.
		chains : int
			The number of independent MCMC chains to run.
		num_processes : int
			The number of processes to run the chains.
		joint_sample_W : bool
			If true, will jointly sample the "W" variables
			at each iteration before individually resampling
			alpha and W. This can improve sample efficiency
			but is a computational bottleneck in high dimensions.
		group_alpha_update : bool
			If true, does a joint group-move update to estimate 
			the sparsity. Else, uses the standard conjugacy
			rules for a Uniform prior on the sparsity.
		log_interval : int
			Will log progress after ``log_interval`` iterations. 
			Defaults to None (no logging).
		"""
		## Previous implementation
		##
		# self.alphas = out['alphas']
		# self.ws = out['ws']
		# self.betas = out['betas']
		# self.sigma2s = out['sigma2s']
		# self.alpha0s = out['alpha0s']
		# self.p0s = out['p0s']
		time0 = time.time()
		if log_interval is None:
			log_interval = N + burn + 1


		out = apply_pool(
			_nprior_sample,
			constant_inputs=dict(
				X=self.X,
				y=self.y,
				tauw2=self.tauw2,
				p0_init=self.p0_init,
				min_p0=self.min_p0,
				update_p0=self.update_p0,
				sigma_a0=self.sigma_a0,
				sigma_b0=self.sigma_b0,
				alpha0_a0=self.alpha0_a0,
				alpha0_b0=self.alpha0_b0,
				sigma_prior_type=self.sigma_prior_type,
				joint_sample_W=joint_sample_W,
				group_alpha_update=group_alpha_update,
				log_interval=log_interval,
				time0=time0
			),
			N=[N+burn for _ in range(chains)],
			num_processes=num_processes
		)
		self.alphas = np.concatenate([x['alphas'][burn:] for x in out]) 
		self.ws = np.concatenate([x['ws'][burn:] for x in out])
		self.betas = np.concatenate([x['betas'][burn:] for x in out])
		self.sigma2s = np.concatenate([x['sigma2s'][burn:] for x in out])
		self.alpha0s = np.concatenate([x['alpha0s'][burn:] for x in out])
		self.p0s = np.concatenate([x['p0s'][burn:] for x in out])