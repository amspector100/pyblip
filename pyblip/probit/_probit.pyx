# cython: profile=True

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

# Blas commonly used parameters
cdef double zero = 0, one = 1, neg1 = -1
cdef int inc_0 = 0;
cdef int inc_1 = 1
cdef char* trans_t = 'T'
cdef char* trans_n = 'N'
cdef char* triang_u = 'U'
cdef char* triang_l = 'L'
cdef double M_SQRT1_2 = sqrt(0.5)

# Truncated normal sampling
from ..cython_utils._truncnorm import random_uniform, sample_truncnorm

def _sample_probit_spikeslab(
	int N,
	double[:, ::1] X,
	long[::1] y,
	double tau2,
	double sigma2,
	double p0,
	int update_p0
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
		np.ndarray[double, ndim=2] Z_arr = np.zeros((N, n))
		double[:, ::1] Z = Z_arr

		# Precompute useful quantities 
		double[:, ::1] XT = np.ascontiguousarray(X.T)
		double[::1] Xl2 = np.power(X, 2).sum(axis=0)
		double[::1] logdets = np.zeros((p, ))
		double logodds = log(p0) - log(1 - p0)
		double[::1] post_vars = np.zeros((p,))

		# scratch
		double old_betaj = 0
		double XjTr, log_ratio_num, log_ratio_denom, log_det
		double u, ratio, kappa, delta, post_mean
		int num_active

		# Initialize mu (predictions) and r (residuals)
		np.ndarray[double, ndim=1] mu_arr = np.dot(X, betas_arr[0])
		double[::1] mu = mu_arr
		np.ndarray[double, ndim=1] r_arr = np.zeros((n,))
		double[::1] r = r_arr

	# precompute log determinants / posterior variances
	for j in range(p):
		logdets[j] = log(1 + tau2 * Xl2[j] / sigma2) / 2
		post_vars[j] = 1 / (1 / tau2 + Xl2[j] / sigma2)

	# Initialize Z, mu, r
	for it in range(n):
		Z[0, it] = sample_truncnorm(
			mean=mu[it], var=sigma2, b=0, lower_interval=y[it] 
		)
		r[it] = Z[0, it] - mu[it]


	for i in range(N):
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
			log_ratio_num = tau2 / sigma2 * XjTr * XjTr
			log_ratio_denom = 2 * (sigma2 + tau2 * Xl2[j]) 
			ratio = exp(logodds - log_ratio_num / log_ratio_denom + logdets[j])
			kappa = ratio / (1.0 + ratio)
			# Reset betaj
			u = random_uniform()
			if u <= kappa:
				betas[i, j] = 0
			else:
				post_mean = post_vars[j] * XjTr / sigma2
				#print(f"i={i}, j={j}, kappa={kappa}, XjTr={XjTr}, post_mean={post_mean}, post_var={post_vars[j]}")
				#post_var * np.sum(r_arr)
				betas[i, j] = np.sqrt(post_vars[j]) * np.random.randn() + post_mean
			# Update Z, r, mu
			delta = betas[i, j] - old_betaj # note if delta = 0, old_betaj = 0
			if delta != 0:
				# update mu
				blas.daxpy(
					&n,
					&delta,
					&XT[j,0],
					&inc_1,
					&mu[0],
					&inc_1
				)
				# make sure sgn(z) matches y
				for it in range(n):
					Z[i, it] = sample_truncnorm(
						mean=mu[it], var=sigma2, b=0, lower_interval=y[it] 
					)
					r[it] = Z[i, it] - mu[it]
					# if y[it] == 0:
					# 	assert Z[i, it] > 0
					# if y[it] == 1:
					# 	assert Z[i, it] < 0

		# Assert mu = np.dot(X, beta)
		#mu2 = np.dot(X, betas_arr[i])
		#r2 = Z_arr[i] - mu_arr
		#print(f"mu - mu2: {np.abs(mu_arr - mu2).mean()}")
		#print(f"r - r2: {np.abs(r_arr - r2).mean()}")

		# Resample p0s
		if update_p0 == 1:
			# Calculate number of active variables
			num_active = 0
			for j in range(p):
				if betas[i,j] != 0:
					num_active += 1
			# sample 0
			p0s[i] = np.random.beta(
				1 + p - num_active, 1 + num_active
			)
			logodds = log(p0s[i]) - log(1 - p0s[i])

		# Set new betas, Z to be old values (temporarily)
		if i != N - 1:
			betas[i+1] = betas[i]
			Z[i+1] = Z[i]
			p0s[i+1] = p0s[i]


	return {"Z":Z_arr, "betas":betas_arr, "p0s":p0s_arr}




	
