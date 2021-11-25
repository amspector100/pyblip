import time
import numpy as np
import scipy as sp
from scipy import linalg
from scipy import stats
from knockpy.utilities import apply_pool

from ._probit import _sample_probit_spikeslab

class ProbitSpikeSlab():

	def __init__(
		self,
		X,
		y,
		sigma2=1,
		tau2=1,
		p0=0.9,
		update_p0=True
	):
		self.X = X
		self.y = y.astype(int)
		self.sigma2 = sigma2
		self.tau2 = tau2
		self.p0 = p0
		self.update_p0 = update_p0

	def sample(
		self, N, burn=100, chains=1, num_processes=1
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
		out = apply_pool(
			_sample_probit_spikeslab,
			constant_inputs=dict(
				X=self.X,
				y=self.y,
				sigma2=self.sigma2,
				tau2=self.tau2,
				p0=self.p0,
				update_p0=self.update_p0,
			),
			N=[N+burn for _ in range(chains)],
			num_processes=num_processes
		)
		self.betas = np.concatenate([x['betas'][burn:] for x in out])
		self.Z = np.concatenate([x['Z'][burn:] for x in out])
		self.p0s = np.concatenate([x['p0s'][burn:] for x in out])


