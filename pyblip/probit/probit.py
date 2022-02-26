import time
import numpy as np
import scipy as sp
from scipy import linalg
from scipy import stats
from knockpy.utilities import apply_pool

from ..linear._linear import _sample_linear_spikeslab
from ._probit import _sample_probit_spikeslab

class ProbitSpikeSlab():

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
		self, N, burn=100, chains=1, num_processes=1, bsize=1
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
		"""
		# Note that in the inner sampling function,
		# y is always continuous linear regression responses
		# and z is always the 0/1 probit responses
		y_latent = np.zeros(self.y.shape[0])
		constant_inputs=dict(
			X=self.X,
			y=y_latent,
			z=self.y.astype(int),
			probit=1,
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

		out = apply_pool(
			_sample_linear_spikeslab,
			constant_inputs=constant_inputs,
			N=[N+burn for _ in range(chains)],
			num_processes=num_processes
		)
		self.betas = np.concatenate([x['betas'][burn:] for x in out])
		self.p0s = np.concatenate([x['p0s'][burn:] for x in out])
		self.tau2s = np.concatenate([x['tau2s'][burn:] for x in out])
		self.sigma2s = np.concatenate([x['sigma2s'][burn:] for x in out])
		self.Z = np.concatenate([x['y_latent'][burn:] for x in out])
		#self.y_latent = np.concatenate([x['y_latent'][burn:] for x in out])