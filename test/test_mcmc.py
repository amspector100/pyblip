import time
import numpy as np
import scipy as sp
from scipy import stats
import unittest
import pytest
from . import context
from .context import pyblip

class TestMCMC(unittest.TestCase):

	def test_conjugate_case(self):

		np.random.seed(123)
		n = 20
		p = 3
		tau2 = 1
		sigma2 = 1
		M = 10000
		X, y, beta = context.generate_regression_data(
			a=1, b=1, coeff_size=tau2, p=p, n=n, sparsity=1, coeff_dist='normal'
		)
		X = X - X.mean(axis=0)
		# linear spike slab
		lm = pyblip.linear.LinearSpikeSlab(
			X=X, y=y, tau2=tau2, update_tau2=False, sigma2=sigma2, update_sigma2=False,
			p0=0.000000001, update_p0=False
		)
		# Calculate analytical solution
		Sigma11 = sigma2 * np.eye(n) + tau2 * np.dot(X, X.T)
		Sigma12 = tau2 * X
		Sigma22 = tau2 * np.eye(p)
		Sigma = np.concatenate(
			[np.concatenate([Sigma11, Sigma12], axis=1),
			 np.concatenate([Sigma12.T, Sigma22], axis=1)],
			axis=0
		)
		S11I = np.linalg.inv(Sigma11)
		cond_mean = np.dot(np.dot(Sigma12.T, S11I), y)
		cond_var = Sigma22 - np.dot(Sigma12.T, np.dot(S11I, Sigma12))

		for bsize in [1, 2]:
			lm.sample(N=M, bsize=bsize, burn=M)
			post_mean = lm.betas.mean(axis=0)
			np.testing.assert_array_almost_equal(
				post_mean,
				cond_mean,
				decimal=2,
				err_msg=f"post_mean={cond_mean} but posterior samples had mean {post_mean}."
			)
			post_var = np.cov(lm.betas.T)
			np.testing.assert_array_almost_equal(
				cond_var,
				post_var,
				decimal=2,
				err_msg=f"post_var={post_var}, post samples had cov {post_var}"
			)

	def test_update_hparams(self):

		np.random.seed(123)
		n = 1000
		p = 4
		p0_a0 = 1
		p0_b0 = 1
		X, y, beta = context.generate_regression_data(
			a=1, b=10, coeff_size=5, n=n, p=p, sparsity=0.5, coeff_dist='normal'
		)
		lm = pyblip.linear.LinearSpikeSlab(
			X=X, y=y, p0=0.5, p0_a0=p0_a0, p0_b0=p0_b0, min_p0=0, 
			update_p0=True, update_sigma2=True, update_tau2=True, 
		)
		# Posterior of p0 should be Beta(1+2, 1+2)
		for bsize in [4,1]:
			lm.sample(N=5000, bsize=bsize, burn=1000)
			nactive = np.sum(lm.betas != 0, axis=1)
			post_means = (p0_a0 + p - nactive) / (p0_a0 + p0_b0 + p)
			expected_mean = np.mean(post_means)
			p0_post_mean = np.mean(lm.p0s)
			np.testing.assert_array_almost_equal(
				p0_post_mean,
				expected_mean,
				decimal=2,
				err_msg=f"expected p0 mean={expected_mean}, bsize={bsize}, but posterior samples had mean {p0_post_mean}."
			)

	def test_coeff_estimation(self):

		# Sample data with high SNR
		np.random.seed(123)
		X, y, beta_linear = context.generate_regression_data(
			a=1, b=1, y_dist='linear', p=100, n=1000, sparsity=0.25, coeff_dist='uniform'
		)
		# Probit data
		Xp, yp, beta_probit = context.generate_regression_data(
			a=0.1, b=1, y_dist='probit', p=50, n=1000, sparsity=0.25, coeff_dist='uniform'
		)

		# Initialize models models
		lm = pyblip.linear.LinearSpikeSlab(X=X, y=y)#, p0=0.25, update_p0=False)
		nlm = pyblip.nprior.NPrior(X=X, y=y, tauw2=1)
		probit = pyblip.probit.ProbitSpikeSlab(X=Xp, y=yp)
		for name, model, beta in zip(
			['linspikeslab', 'nprior', 'probitspikeslab'],
			[lm, nlm, probit],
			[beta_linear, beta_linear, beta_probit],
		):
			sample_kwargs = dict(N=1500, chains=1, burn=500)
			model.sample(**sample_kwargs)
			# Test inclusions
			pips = (model.betas != 0).mean(axis=0)
			m_nn_pip = np.min(pips[beta != 0])
			self.assertTrue(
				m_nn_pip > 0.9,
				f"With n >= 10 p, min non-null PIP is {m_nn_pip} > 0.9 for {name}"
			)
			# Test estimation of beta for linear models
			# (all are slightly misspecified and this makes a difference for probit)
			if name != 'probitspikeslab':
				hatbeta = model.betas.mean(axis=0)
				hatbeta_nn = hatbeta[beta != 0]
				hatbeta_null = hatbeta[beta == 0]
				np.testing.assert_almost_equal(
					np.power(hatbeta_nn, 2).mean(),
					np.power(beta[beta != 0], 2).mean(),
					decimal=2,
					err_msg=f"Average l2 norm of non-null est vs. true values differs for {name}"
				)
				np.testing.assert_almost_equal(
					np.power(hatbeta_null, 2).mean(),
					0,
					decimal=1,
					err_msg=f"Average l2 norm of null coeffs is too large"
				)

	# def test_probit_estimation(self):

	# 	# Sample data with high SNR
	# 	np.random.seed(123)
	# 	X, y, beta = context.generate_regression_data(
	# 		a=1, b=1, y_dist='probit', p=50, n=1000, sparsity=0.25, coeff_dist='uniform'
	# 	)

	# 	# Fit probit model
	# 	probit = pyblip.probit.ProbitSpikeSlab(X=X, y=y)
	# 	probit.sample(N=1000, burn=200, chains=1)

	# def test_linear_sparsity_estimation(self):

	# 	# Sample data
	# 	np.random.seed(123)
	# 	N = 200
	# 	chains = 5
	# 	coeff_size = 1
	# 	for sparsity in [0.01, 0.05]:
	# 		X, y, beta = context.generate_regression_data(
	# 			a=5, b=1, y_dist='linear', p=500, n=200, sparsity=sparsity,
	# 			coeff_size=coeff_size, coeff_dist='normal'
	# 		)

	# 		# Fit linear model
	# 		p0 = 1 - sparsity
	# 		lm = pyblip.linear.LinearSpikeSlab(X=X, y=y)
	# 		lm.sample(N=N, chains=chains)
	# 		self.assertTrue(
	# 			np.abs(np.mean(lm.p0s) - p0) < 0.1,
	# 			f"Est p0 for linspikeslab is {np.mean(lm.p0s)}, true p0={p0}"
	# 		)
	# 		nlm = pyblip.nprior.NPrior(X=X, y=y, tauw2=1)
	# 		nlm.sample(N=N, chains=chains)

	# 		self.assertTrue(
	# 			np.abs(np.mean(nlm.p0s) - p0) < 0.1,
	# 			f"Est p0 for nprior is {np.mean(nlm.p0s)}, true p0={p0}"
	# 		)

if __name__ == "__main__":
	unittest.main()
TestMCMC