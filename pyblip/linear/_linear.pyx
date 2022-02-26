# cython: profile=False

import time 
cimport cython
import numpy as np
import scipy.stats
cimport numpy as np
from numpy cimport PyArray_ZEROS
import scipy.linalg
cimport scipy.linalg.cython_blas as blas
cimport scipy.linalg.cython_lapack as lapack

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log, exp, fabs, sqrt, fmax, erfc

# Fast uniform sampling
from ..cython_utils._truncnorm import random_uniform
from ..cython_utils._update_hparams import _update_hparams

# Blas commonly used parameters
cdef double zero = 0, one = 1, neg1 = -1
cdef int inc_0 = 0;
cdef int inc_1 = 1
cdef char* trans_t = 'T'
cdef char* trans_n = 'N'
cdef char* triang_u = 'U'
cdef char* triang_l = 'L'
cdef double M_SQRT1_2 = sqrt(0.5)

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def _sample_linear_spikeslab(
	int N,
	double[:, ::1] X,
	double[::1] y,
	double tau2,
	int update_tau2,
	double tau2_a0,
	double tau2_b0,
	double sigma2,
	int update_sigma2,
	double sigma2_a0,
	double sigma2_b0,
	double p0,
	int update_p0,
	double min_p0,
	double p0_a0,
	double p0_b0
):
	# Initialize outputs
	cdef:
		# Useful constants
		int n = X.shape[0]
		int p = X.shape[1]
		int i, it, j

		# Initialize outputs
		np.ndarray[long, ndim=1] inds = np.arange(p)
		np.ndarray[double, ndim=1] p0s_arr = np.zeros((N,))
		double[::1] p0s = p0s_arr
		np.ndarray[double, ndim=2] betas_arr = np.zeros((N, p))
		double[:, ::1] betas = betas_arr
		np.ndarray[double, ndim=1] sigma2s_arr = np.zeros(N,)
		double[::1] sigma2s = sigma2s_arr
		np.ndarray[double, ndim=1] tau2s_arr = np.zeros(N,)
		double[::1] tau2s = tau2s_arr

		# Precompute useful quantities 
		double[:, ::1] XT = np.ascontiguousarray(X.T)
		double[::1] Xl2 = np.power(X, 2).sum(axis=0)
		double[::1] logdets = np.zeros((p, ))
		double logodds = log(p0) - log(1 - p0)
		double[::1] post_vars = np.zeros((p,))

		# scratch
		double old_betaj = 0
		double neg_betaj
		double XjTr, log_ratio_num, log_ratio_denom, log_det
		double u, ratio, kappa, delta
		int num_active

		# Proposals, only used if min_p0 > 0
		int max_nprop = 100
		np.ndarray[double, ndim=1] p0_proposal_arr = np.zeros(max_nprop,)
		double[::1] p0_proposals = p0_proposal_arr

		# for sigma2 updates
		double r2, sigma_b

		# for tau2 updates
		double sample_var

		# Initialize mu (predictions) and r (residuals)
		# np.ndarray[double, ndim=1] mu_arr = np.dot(X, betas_arr[0])
		# double[::1] mu = mu_arr
		np.ndarray[double, ndim=1] r_arr = y - np.dot(X, betas_arr[0])
		double[::1] r = r_arr

	# precompute sigma2 posterior variance
	cdef double sigma_a = n / 2.0 + sigma2_a0
	cdef np.ndarray [double, ndim=1] invgammas = scipy.stats.invgamma(sigma_a).rvs(N) 

	# initialize
	sigma2s[0] = sigma2
	tau2s[0] = tau2
	p0s[0] = p0

	for i in range(N):
		# precompute log determinants / posterior variances
		for j in range(p):
			logdets[j] = log(1.0 + tau2s[i] * Xl2[j] / sigma2s[i]) / 2.0
			post_vars[j] = 1.0 / (1.0 / tau2s[i] + Xl2[j] / sigma2s[i])
		# update beta
		np.random.shuffle(inds)
		for j in inds:
			old_betaj = betas[i, j]
			# possibly reset residuals to zero out betaj
			if old_betaj != 0:
				blas.daxpy(&n, &betas[i,j], &XT[j,0], &inc_1, &r[0], &inc_1)
			# compute log ratio P(betaj = 0) / P(betaj != 0)
			XjTr = blas.ddot(
				&n,
				&r[0],
				&inc_1,
				&XT[j, 0],
				&inc_1
			)
			log_ratio_num = tau2s[i] / sigma2s[i] * XjTr * XjTr
			log_ratio_denom = 2 * (sigma2s[i] + tau2s[i] * Xl2[j]) 
			ratio = exp(logodds - log_ratio_num / log_ratio_denom + logdets[j])
			kappa = ratio / (1.0 + ratio)
			# Reset betaj
			u = random_uniform()
			if u <= kappa:
				betas[i, j] = 0
			else:
				post_mean = post_vars[j] * XjTr / sigma2s[i]
				#print(f"i={i}, j={j}, kappa={kappa}, XjTr={XjTr}, post_mean={post_mean}, post_var={post_vars[j]}")
				#post_var * np.sum(r_arr)
				betas[i, j] = np.sqrt(post_vars[j]) * np.random.randn() + post_mean
			# update mu
			# delta = betas[i, j] - old_betaj # note if delta = 0, old_betaj = 0
			# if delta != 0:
			# 	blas.daxpy(
			# 		&n,
			# 		&delta,
			# 		&XT[j,0],
			# 		&inc_1,
			# 		&mu[0],
			# 		&inc_1
			# 	)
			if betas[i, j] != 0:
				neg_betaj = -1*betas[i,j]
				blas.daxpy(
					&n,
					&neg_betaj,
					&XT[j,0],
					&inc_1,
					&r[0],
					&inc_1
				)

		# Update hyperparams
		_update_hparams(
			i=i, 
			n=n,
			p=p,
			max_nprop=max_nprop,
			betas=betas,
			r=r,
			p0s=p0s,
			sigma2s=sigma2s,
			tau2s=tau2s,
			invgamma=invgammas[i],
			p0_proposals=p0_proposals,
			update_tau2=update_tau2,
			tau2_a0=tau2_a0,
			tau2_b0=tau2_b0,
			update_sigma2=update_sigma2,
			sigma2_a0=sigma2_a0,
			sigma2_b0=sigma2_b0,
			update_p0=update_p0,
			min_p0=min_p0,
			p0_a0=p0_a0,
			p0_b0=p0_b0,
		)

		# Recompute logodds
		logodds = log(p0s[i]) - log(1 - p0s[i])

		# Set new betas, p0s to be old values (temporarily)
		if i != N - 1:
			betas[i+1] = betas[i]
			p0s[i+1] = p0s[i]
			sigma2s[i+1] = sigma2s[i]
			tau2s[i+1] = tau2s[i]


	return {"betas":betas_arr, "p0s":p0s_arr, "tau2s":tau2s_arr, "sigma2s":sigma2s_arr}




	
