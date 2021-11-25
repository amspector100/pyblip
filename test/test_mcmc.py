import time
import numpy as np
import scipy as sp
from scipy import stats
import unittest
import pytest
from . import context
from .context import pyblip
from pyblip import linear, probit, nprior

class TestMCMC(unittest.TestCase):

	def test_linear_sparsity_estimation(self):

		# Sample data
		np.random.seed(123)
		coeff_size = 3
		X, y, beta = context.generate_regression_data(
			a=5, b=1, y_dist='linear', p=500, n=100, sparsity=0.05
		)

		# Fit linear model
		lm = pyblip.linear.LinearSpikeSlab(X=X, y=y)
		lm.sample(N=500, chains=5)
		nlm = pyblip.nprior.NPrior(X=X, y=y, tauw2=1)
		nlm.sample(N=500, chains=5)
		print(lm.betas.shape)
		print(nlm.betas.shape)
		raise ValueError()

if __name__ == "__main__":
	unittest.main()
