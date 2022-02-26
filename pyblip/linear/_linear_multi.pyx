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

# Fast uniform/truncated normal sampling
from ..cython_utils._truncnorm import random_uniform, sample_truncnorm
from ..cython_utils._update_hparams import _update_hparams

from itertools import combinations

# Blas commonly used parameters
cdef double zero = 0, one = 1, neg1 = -1
cdef int inc_0 = 0;
cdef int inc_1 = 1
cdef char* trans_t = 'T'
cdef char* trans_n = 'N'
cdef char* triang_u = 'U'
cdef char* triang_l = 'L'
cdef char* triang_r = 'R'
cdef double M_SQRT1_2 = sqrt(0.5)

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef int _zero_XAT(
	double[:, ::1] XAT,
	int bsize,
	int n
):
	"""
	Fills XAT with zeros.
	"""
	cdef int nb = n * bsize
	blas.daxpy(
		&nb,
		&neg1,
		&XAT[0,0],
		&inc_1,
		&XAT[0,0],
		&inc_1
	)
	return 0

cdef int _weighted_choice(
	double[::1] probs,
	double[::1] log_probs,
	int nmodel
):
	cdef int ii
	cdef double max_log_prob = 0.0
	cdef double denom = 0.0
	cdef double cumsum = 0.0
	# Avoid overflow
	for ii in range(nmodel):
		max_log_prob = fmax(log_probs[ii], max_log_prob)
	# Compute probabilities
	for ii in range(nmodel):
		probs[ii] = exp(log_probs[ii] - max_log_prob)
		denom += probs[ii]
	# Choose integer
	cdef double u = random_uniform()
	for ii in range(nmodel):
		cumsum += probs[ii] / denom
		if cumsum >= u:
			return ii


cdef int _precompute_suff_stats(
	double[:, ::1] XT,
	double[:, ::1] XAT,
	double[:, ::1] XTX,
	double[:, ::1] XATXA_cache,
	double[::1] r,
	double[::1] XATr_cache,
	long[:, ::1] blocks,
	int bi,
	int n,
	int bsize,
):
	"""
	Computes XATXA and XATr for the full model.
	These are the only O(n) operations required
	in the full loop, 
	"""
	cdef int ii, jj, jj2, j, j2, INFO
	cdef int b2 = bsize * bsize

	# 1. Loop through fill XAT. Note
	# XAT is initialized to all zeros.
	ii = 0
	for jj in range(bsize):
		j = blocks[bi, jj]
		# Set XA[:, ii] = X[:, j]
		blas.daxpy(
			&n,
			&one,
			&XT[j,0],
			&inc_1,
			&XAT[ii,0],
			&inc_1
		)
		ii += 1

	# Step 2. Set XATXA_cache = np.dot(XAT, XA)
	for jj in range(bsize):
		j = blocks[bi, jj]
		for jj2 in range(bsize):
			j2 = blocks[bi, jj2]
			XATXA_cache[jj, jj2] = XTX[j, j2]

	# blas.dgemm(
	# 	trans_t, # transA
	# 	trans_n, # transB
	# 	&bsize, # M = op(A).shape[0]
	# 	&bsize, # N = op(B).shape[1]
	# 	&n, # K = op(A).shape[1] = op(B).shape[0]
	# 	&one, # alpha
	# 	&XAT[0,0], # A
	# 	&n, # LDA first dim of A in calling program
	# 	&XAT[0,0], # B
	# 	&n, # LDB first dim of B in calling program
	# 	&zero, # beta
	# 	&XATXA_cache[0,0], # C
	# 	&bsize, # first dim of C in calling program
	# )

	# Step 3. Set XATr_cache = np.dot(XAT, r)
	blas.dgemm(
		trans_t, # transA
		trans_n, # transB
		&bsize, # M = op(A).shape[0]
		&inc_1, # N = op(B).shape[1]
		&n, # K = op(A).shape[1] = op(B).shape[0]
		&one, # alpha
		&XAT[0,0], # A
		&n, # LDA first dim of A in calling program
		&r[0], # B
		&n, # LDB first dim of B in calling program
		&zero, # beta
		&XATr_cache[0], # C
		&bsize, # first dim of C in calling program
	)

	return 0



cdef double _compute_QA(
	double[:, ::1] XATXA,
	double[:, ::1] XATXA_cache,
	double[::1] mu,
	double[::1] XATr,
	double[::1] XATr_cache,
	double[:, ::1] QA,
	tuple model_comb,
	int bi,
	int bsize,
	double tau2,
	double sigma2,
):
	"""
	Fills XATr, XATXA based on the indices from blocks.
	Also fills mu and computes QA.
	"""
	cdef int msize = len(model_comb)
	# Step 1. Assemble XA
	cdef int ii, jj, ii2, jj2, j, INFO
	cdef int b2 = bsize * bsize
	cdef double cm_scale

	# 1. Fill XATr based on XATr_cache
	ii = 0
	for jj in model_comb:
		XATr[ii] = XATr_cache[jj]
		ii += 1

	# 2. Fill XATXA based on XATXA_cache
	ii = 0
	for jj in model_comb:
		ii2 = 0
		for jj2 in model_comb:
			XATXA[ii, ii2] = XATXA_cache[jj, jj2]
			ii2 += 1
		ii += 1

	# Set QA to the identity
	for ii in range(msize):
		for jj in range(msize):
			if ii == jj:
				QA[ii, jj] = 1.0
			else:
				QA[ii, jj] = 0.0
	# Set QA = I + tau2 / sigma2 XATXA
	cm_scale = tau2 / sigma2
	blas.daxpy(
		&b2,
		&cm_scale,
		&XATXA[0,0],
		&inc_1,
		&QA[0,0],
		&inc_1,
	)

	# Step 3: QA = cholesky decomp of QA
	lapack.dpotrf(
		triang_l, # UPLO, upper vs. lower triangle of A
		&msize, # dim of matrix
		&QA[0, 0], # matrix to perform cholesky decomp on
		&bsize, # LDA = leading dim of QA in calling program
		&INFO # error output
	)
	### FOR DEBUGGING DELETE LATER
	if INFO != 0:
		raise RuntimeError(f"dpotrf exited with INFO={INFO}. Try setting bsize=1.")

	# Calculate log determinant term of (original QA)
	cdef double log_det_QA = 0.0
	for jj in range(msize):
		log_det_QA += 2.0*log(QA[jj, jj])

	# Step 4: QA = inverse of itself
	lapack.dtrtri(
		triang_l, # UPLO, upper vs. lower triangular
		trans_n, # diag, n means not diagonal
		&msize, # N dimension
		&QA[0,0], # A
		&bsize, # LDA = leading dim of QA in calling program
		&INFO # error output
	)

	# ### FOR DEBUGGING DELETE LATER
	# if INFO != 0:
	# 	raise RuntimeError(f"dtrtri exited with INFO={INFO}. Try setting bsize=1.")

	# Step 6: set mu = np.dot(QA, XATr)
	for jj in range(msize):
		mu[jj] = XATr[jj]
	blas.dtrmm(
		triang_l, # 'L' (left: op(A) @ B) or 'R' (right: B @ op(A))
		triang_l, # upper or lower for triang matrix
		trans_n, # transA, transpose A or not (I think no)
		trans_n, # diag, n means not unit triangular
		&msize, # M = B.shape[0]
		&inc_1, # N = op(B).shape[1]
		&one, # alpha
		&QA[0,0], # A
		&bsize, # LDA first dim of A in calling program
		&mu[0], # B
		&bsize, # LDB first dim of B in calling program
	)

	return log_det_QA


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def _sample_spikeslab_multi(
	int N,
	int bsize,
	double[:, ::1] X,
	double[::1] y, # outcomes for linear regression, not used in probit model
	long[::1] z, # censored outcomes for probit model, not used in linear regression
	int probit,
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
		int nblock = round((p + bsize - 1) / bsize)
		int nmodel = int(2**bsize)
		int i, ii, it, bi, bj, j, jj, mj, msize
		int b2 = bsize * bsize
		int INFO = 0

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

		#### scratch for block gibbs sampling
		list model_combs
		tuple model_comb
		np.ndarray[long, ndim=2] blocks_arr = -1*np.ones((nblock, bsize)).astype(int)
		long[:, ::1] blocks = blocks_arr
		double[::1] model_probs = np.zeros(nmodel,) # P(A = J0) in notation of paper
		double[::1] model_lprobs = np.zeros(nmodel,) # log P(A = J0) in notation of paper
		np.ndarray[long, ndim=1] model_inds = np.arange(bsize)
		np.ndarray[double, ndim=2] XAT_arr = np.zeros((bsize, n))
		double[:, ::1] XAT = XAT_arr

		# Sufficient statistics. The caches store precomputed values for the full bsize model
		# and then the other arrays get filled with submodels.
		np.ndarray[double, ndim=2] XATXA_c = np.zeros((bsize, bsize))
		double[:, ::1] XATXA_cache = XATXA_c
		np.ndarray[double, ndim=2] XATXA_arr = np.zeros((bsize, bsize))
		double[:, ::1] XATXA = XATXA_arr
		np.ndarray[double, ndim=1] XATr_c = np.zeros(bsize)
		double[::1] XATr_cache = XATr_c
		np.ndarray[double, ndim=1] XATr_arr = np.zeros(bsize)
		double[::1] XATr = XATr_arr

		# matrix QA from paper
		np.ndarray[double, ndim=2] QA_arr = np.zeros((bsize, bsize)) # can delete after debugging done
		double[:, ::1] QA = QA_arr
		# The conditional mean / variance / related vectors
		np.ndarray[double, ndim=1] mu_arr = np.zeros(bsize,)
		double[::1] mu = mu_arr
		double[::1] mu_copy = np.zeros(bsize,) # because there's no in-place dgemm
		np.ndarray[double, ndim=2] V_arr = np.zeros((bsize, bsize)) # conditional covariance
		double[:, ::1] V = V_arr
		np.ndarray[double, ndim=2] Vs_arr = np.zeros((bsize, bsize)) # scratch for V
		double[:, ::1] Vs = Vs_arr # just scratch
		# Store updates for beta as contiguous array
		np.ndarray[double, ndim=1] beta_next_arr = np.zeros(bsize,)
		double[::1] beta_next = beta_next_arr

		# Misc. scratch scalars
		double log_det_QA, exp_term, cm_scale
		double old_betaj = 0
		double neg_betaj
		int num_active

		# Precompute useful quantities 
		double[:, ::1] XT = np.ascontiguousarray(X.T)
		double logp0 = log(p0)
		double log1p0 = log(1 - p0)

		# Proposals, only used if min_p0 > 0
		int max_nprop = 100
		np.ndarray[double, ndim=1] p0_proposal_arr = np.zeros(max_nprop,)
		double[::1] p0_proposals = p0_proposal_arr

		# for sigma2 /tau2 updates
		double r2, sigma_b, sample_var

		# Initialize mu (predictions) and r (residuals)
		np.ndarray[double, ndim=1] mu_pred_arr = np.dot(X, betas_arr[0])
		double[::1] mu_pred = mu_pred_arr
		np.ndarray[double, ndim=1] r_arr = np.zeros(n)# = y - np.dot(X, betas_arr[0])
		double[::1] r = r_arr
		
		# Initialize latent variables Y (only used in probit regression)
		np.ndarray[double, ndim=2] Y_latent_arr = np.zeros((N, n))
		double[:, ::1] Y_latent = Y_latent_arr

	# Cache XTX
	cdef double[:, ::1] XTX = np.dot(X.T, X)

	# precompute sigma2 posterior variance
	cdef double sigma_a = n / 2.0 + sigma2_a0
	cdef np.ndarray [double, ndim=1] invgammas = scipy.stats.invgamma(sigma_a).rvs(N) 

	# initialize
	sigma2s[0] = sigma2
	tau2s[0] = tau2
	p0s[0] = p0

	# Initialize residuals and, when probit=1, latent variables
	if probit == 1:
		for it in range(n):
			Y_latent[0, it] = sample_truncnorm(
				mean=mu_pred[it], var=sigma2, b=0, lower_interval=z[it] 
			)
			y[it] = Y_latent[0, it]
	for it in range(n):
		r[it] = y[it] - mu_pred[it]

	# Initialize blocks from which to do Gibbs sampling
	j = 0
	for bi in range(nblock):
		for bj in range(bsize):
			blocks[bi, bj] = j
			j += 1
			# Ensure that the last block is full
			# even if the blocks aren't fully disjoint
			if j == p:
				j = 0

	# Create model combinations
	model_combs = []
	for msize in range(1, bsize+1):
		model_combs.extend(combinations(model_inds, msize))

	for i in range(N):
		# shuffle blocks and shift by one
		np.random.shuffle(blocks_arr)
		blocks_arr = (blocks_arr + 1) % p
		for bi in range(nblock):
			# Reset residuals to zero out betas in this block
			for bj in range(bsize):
				j = blocks[bi, bj] # the feature we are working with
				if j == -1: # should not happen given our "partition"
					print(f"ERROR at i={i}, bj={bj}, j={j}=-1.")
				old_betaj = betas[i, j]
				if old_betaj != 0:
					blas.daxpy(&n, &old_betaj, &XT[j,0], &inc_1, &r[0], &inc_1)
					# Update predictions for probit case
					if probit == 1:
						neg_betaj = -1*old_betaj
						blas.daxpy(&n, &neg_betaj, &XT[j,0], &inc_1, &mu_pred[0], &inc_1)
				# Set beta to zero
				betas[i, j] = 0

			#print("I claim sum(abs(XAT_arr))==0", np.sum(np.abs(XAT_arr)))
			_zero_XAT(XAT, bsize, n)

			# Precompute sufficient statistics
			_precompute_suff_stats(
				XT=XT,
				XAT=XAT,
				XTX=XTX,
				XATXA_cache=XATXA_cache,
				r=r,
				XATr_cache=XATr_cache,
				blocks=blocks,
				bi=bi,
				n=n,
				bsize=bsize,
			)

			# Loop through models to compute P(A = J).
			# First model (all zeros) is easy
			model_lprobs[0] = bsize * logp0
			# Loop through model choices
			for mj in range(1, nmodel):
				model_comb = model_combs[mj-1]
				msize = len(model_comb)
				log_det_QA = _compute_QA(
					XATXA=XATXA,
					XATXA_cache=XATXA_cache,
					QA=QA,
					mu=mu,
					XATr=XATr,
					XATr_cache=XATr_cache,
					model_comb=model_comb,
					bi=bi,
					bsize=bsize,
					tau2=tau2s[i],
					sigma2=sigma2s[i],
				)

				# Compute exp term
				exp_term = blas.dnrm2(&msize, &mu[0], &inc_1)
				exp_term = tau2s[i] * exp_term * exp_term / (2 * sigma2s[i] * sigma2s[i])

				# Compute overall unnormalized probability
				model_lprobs[mj] = exp_term - log_det_QA / 2
				model_lprobs[mj] += (bsize - msize) * logp0
				model_lprobs[mj] += msize * log1p0

				# ### Debugging
				# QA_guess = tau2s[i] / sigma2s[i] * np.dot(XAT_arr, XAT_arr.T) + np.eye(bsize)
				# QA_guess = QA_guess[0:msize][:, 0:msize]
				# sign, ldet_qa_guess = np.linalg.slogdet(QA_guess)
				# print(f"sign={sign}, det_term={log_det_QA}, confirm={ldet_qa_guess}")
				# QAI = np.linalg.inv(QA_guess)
				# XATr_guess = np.dot(XAT_arr, r_arr)[0:msize]
				# exp_term2 = np.dot(np.dot(XATr_guess, QAI), XATr_guess)
				# exp_term2 *= tau2s[i] / (2 * sigma2s[i] * sigma2s[i])

				# # ### FOR DEBUGGING DELETE LATER
				# QA_guess = tau2s[i] / sigma2s[i] * np.dot(XAT_arr, XAT_arr.T) + np.eye(bsize)
				# QA_guess = QA_guess[0:msize][:, 0:msize]
				# LA_guess = np.linalg.cholesky(QA_guess)
				# LAI = np.linalg.inv(LA_guess)
				# mu_guess = np.dot(LAI, XATr_guess)
				# #print(LAI - QA_arr[0:msize][:, 0:msize].T)
				# print(f"mu={mu_arr[0:msize]}, mudiff={mu_arr[0:msize]-mu_guess}")
				# exp_term3 = np.dot(mu_guess, mu_guess) * tau2s[i] / (2 * sigma2s[i] * sigma2s[i])
				# print(f"msize={msize}, exp_term={exp_term}, exp_term2={exp_term2}, exp_term3={exp_term3}")


				# ### FOR DEBUGGING DELETE LATER
				minds = [blocks[bi,jj] for jj in model_comb]
				# for ii, j in enumerate(minds):
				# 	print("If XAT is right, this should be zero", np.abs(
				# 		XAT_arr[ii,:]-X.T[j,:]).sum()
				# 	)

				# ### FOR DEBUGGING DELETE LATER (check QA)
				# XATXA_guess = np.dot(XAT_arr, XAT_arr.T)
				# print(f"msize={msize}, XATXA_guess-XATXA_arr={XATXA_guess-XATXA_arr}")

				# ### FOR DEBUGGING DELETE LATER (check cholesky)
				# print(f"msize={msize}")
				# print("QAR", QA_arr)
				# print("my cholesky", np.linalg.cholesky(QA_guess[0:msize][:, 0:msize]))
				# print(f"msize={len(model_combs[mj-1])}, xatrdiff={XATr_arr-np.dot(XAT_arr, r_arr)}")

			# Choose model
			mj = _weighted_choice(
				probs=model_probs, log_probs=model_lprobs, nmodel=nmodel
			)
			# Fill coefficients beta. We only need to do this if mj != 0.
			if mj > 0:
				model_comb = model_combs[mj-1]
				msize = len(model_comb)
				# Compute inv cholesky of QA, XA, etc.
				log_det_QA = _compute_QA(
					XATXA=XATXA,
					XATXA_cache=XATXA_cache,
					QA=QA,
					mu=mu,
					XATr=XATr,
					XATr_cache=XATr_cache,
					model_comb=model_comb,
					bi=bi,
					bsize=bsize,
					tau2=tau2s[i],
					sigma2=sigma2s[i],
				)
				# conditional mean (i). Using paper notation, this sets
				# mu = QA^{-1}  XA^T r.
				blas.dtrmm(
					triang_l, # 'L' (left: op(A) @ B) or 'R' (right: B @ op(A))
					triang_l, # upper or lower for triang matrix
					trans_t, # transA, transpose A or not
					trans_n, # diag, n means not unit triangular
					&msize, # M = B.shape[0]
					&inc_1, # N = op(B).shape[1]
					&one, # alpha
					&QA[0,0], # A
					&bsize, # LDA first dim of A in calling program
					&mu[0], # B
					&bsize, # LDB first dim of B in calling program
				)

				# # ### Debugging
				# QA_guess = tau2s[i] / sigma2s[i] * np.dot(XAT_arr, XAT_arr.T) + np.eye(bsize)
				# QA_guess = QA_guess[0:msize][:, 0:msize]
				# QAI = np.linalg.inv(QA_guess)
				# expected = np.dot(QAI, XATr_arr[0:msize])
				# print(f"H1 mu-expected={mu_arr[0:msize] - expected}, mu={mu_arr[0:msize]}")

				# conditional mean (ii). Using paper notation, this sets
				# mu = tau2 / sigma2 * XATXA QA^{-1} XA^T r
				# As programmed, sets mu = tau2 / sigma2 * (XATXA @ mu)
				cm_scale = tau2s[i] / sigma2s[i]
				for ii in range(msize):
					mu_copy[ii] = mu[ii]
				blas.dgemm(
					trans_n, # transA
					trans_n, # transB
					&msize, # M = op(A).shape[0]
					&inc_1, # N = op(B).shape[1]
					&msize, # K = op(A).shape[1] = op(B).shape[0]
					&cm_scale, # alpha
					&XATXA[0,0], # A
					&bsize, # LDA first dim of A in calling program
					&mu_copy[0], # B
					&bsize, # LDB first dim of B in calling program
					&zero, # beta
					&mu[0], # C
					&bsize, # LDC first dim of C in calling program
				)

				# ### Debugging
				# QA_guess = tau2s[i] / sigma2s[i] * np.dot(XAT_arr, XAT_arr.T) + np.eye(bsize)
				# QA_guess = QA_guess[0:msize][:, 0:msize]
				# QAI = np.linalg.inv(QA_guess)
				# expected = cm_scale * np.dot(
				# 	XATXA_arr[0:msize][:, 0:msize],
				# 	np.dot(QAI, XATr_arr[0:msize])
				# )
				# print(XATXA_arr[0:msize][:, 0:msize])
				# print(XATXA[0,0])
				# print(f"H2 mu-expected={mu_arr[0:msize] - expected}, mu={mu_arr[0:msize]}, expected={expected}")

				# conditional mean (iii). In the paper's notation, this sets
				# mu = - XA^T r + tau2 / sigma2 XATXA QA^{-1} XA^T r
				blas.daxpy(
					&msize,
					&neg1,
					&XATr[0],
					&inc_1,
					&mu[0],
					&inc_1
				)
				# finish conditional mean by multiplying by -tau2 / sigma2
				for ii in range(msize):
					mu[ii] *= -1 * tau2s[i] / sigma2s[i]

				# # # ### Debugging
				# sigma2 = sigma2s[i]
				# tau2 = tau2s[i]
				# XA = XAT_arr.T[:, 0:msize]
				# k = msize
				# Sigma11 = sigma2 * np.eye(n) + tau2 * np.dot(XA, XA.T)
				# Sigma12 = tau2 * XA
				# Sigma22 = tau2 * np.eye(k)
				# Sigma = np.concatenate(
				# 	[np.concatenate([Sigma11, Sigma12], axis=1),
				# 	np.concatenate([Sigma12.T, Sigma22], axis=1)],
				# 	axis=0
				# )
				# expected = np.dot(np.dot(Sigma12.T, np.linalg.inv(Sigma11)), r_arr)
				# # QA_guess = tau2s[i] / sigma2s[i] * np.dot(XAT_arr, XAT_arr.T) + np.eye(bsize)
				# # QA_guess = QA_guess[0:msize][:, 0:msize]
				# # QAI = np.linalg.inv(QA_guess)
				# # t1 = tau2s[i] * XATr_arr[0:msize] / sigma2s[i]
				# # t2 = np.dot(
				# # 	XATXA_arr[0:msize][:, 0:msize],
				# # 	np.dot(QAI, XATr_arr[0:msize])
				# # )
				# # expected = t1 - ((tau2s[i] / sigma2s[i])**2) * t2
				# print(f"H3 mu-expected={mu_arr[0:msize] - expected}, mu={mu_arr[0:msize]}, expected={expected}")

				# Now compute conditional covariance
				# 0. Set V = XATXA
				for ii in range(msize):
					for jj in range(msize):
						V[ii, jj] = XATXA[ii, jj]
				# print(f"C0: diff={XATXA_arr[0:msize][:, 0:msize] - V_arr[0:msize][:, 0:msize]}")

				# 1. Set V = LA^{-1} XATXA
				blas.dtrmm(
					triang_l, # 'L' (left: op(A) @ B) or 'R' (right: B @ op(A))
					triang_l, # upper or lower for triang matrix
					trans_n, # transA, transpose A or not (I think no)
					trans_n, # diag, n means not unit triangular
					&msize, # M = B.shape[0]
					&msize, # N = op(B).shape[1]
					&one, # alpha
					&QA[0,0], # A
					&bsize, # LDA first dim of A in calling program
					&V[0,0], # B
					&bsize, # LDB first dim of B in calling program
				)
				# ### DEBUGGING delete later
				# QA_guess = tau2s[i] / sigma2s[i] * np.dot(XAT_arr, XAT_arr.T) + np.eye(bsize)
				# QA_guess = QA_guess[0:msize][:, 0:msize]
				# QAI = np.linalg.inv(np.linalg.cholesky(QA_guess))
				# expected = np.dot(QAI.T, XATXA_arr[0:msize][:, 0:msize])
				# result = V_arr[0:msize][:, 0:msize]
				# print(f"C1 V-expected={result - expected}\n V={result}\n expected={expected}\n")
				
				# 2. Set V = QA^{-1} XATXA
				blas.dtrmm(
					triang_l, # 'L' (left: op(A) @ B) or 'R' (right: B @ op(A))
					triang_l, # upper or lower for triang matrix
					trans_t, # transA, transpose A or not (I think no)
					trans_n, # diag, n means not unit triangular
					&msize, # M = B.shape[0]
					&msize, # N = op(B).shape[1]
					&one, # alpha
					&QA[0,0], # A
					&bsize, # LDA first dim of A in calling program
					&V[0,0], # B
					&bsize, # LDB first dim of B in calling program
				)
				# # ### DEBUGGING delete later
				# QA_guess = tau2s[i] / sigma2s[i] * np.dot(XAT_arr, XAT_arr.T) + np.eye(bsize)
				# QA_guess = QA_guess[0:msize][:, 0:msize]
				# QAI = np.linalg.inv(QA_guess)
				# expected = np.dot(QAI, XATXA_arr[0:msize][:, 0:msize])
				# result = V_arr[0:msize][:, 0:msize]
				# print(f"C2 Vs-expected=\n{result - expected}\n Vs={result}, expected={expected}")

				# 3. Set Vs = XATXA QA^{-1} XATXA
				blas.dgemm(
					trans_n, # transA
					trans_n, # transB
					&msize, # M = op(A).shape[0]
					&msize, # N = op(B).shape[1]
					&msize, # K = op(A).shape[1] = op(B).shape[0]
					&one, # alpha
					&XATXA[0,0], # A
					&bsize, # LDA first dim of A in calling program
					&V[0,0], # B
					&bsize, # LDB first dim of B in calling program
					&zero, # beta
					&Vs[0,0], # C
					&bsize, # first dim of C in calling program
				)
				# ### DEBUGGING delete later
				# QA_guess = tau2s[i] / sigma2s[i] * np.dot(XAT_arr, XAT_arr.T) + np.eye(bsize)
				# QA_guess = QA_guess[0:msize][:, 0:msize]
				# QAI = np.linalg.inv(QA_guess)
				# XATXA_g = XATXA_arr[0:msize][:, 0:msize]
				# expected = np.dot(XATXA_g, np.dot(QAI, XATXA_g))
				# result = Vs_arr[0:msize][:, 0:msize]
				# print(f"C3 Vs-expected=\n{result - expected}\n Vs={result}, expected={expected}")

				# 4. Set V to be the conditional variance:
				# tau2 I_{|A|} - tau2/sigma4 XATXA + tau4/sigma6 XATXA QA^{-1} XATXA
				b2 = bsize * bsize
				blas.daxpy(
					&b2,
					&neg1,
					&V[0,0],
					&inc_1,
					&V[0,0],
					&inc_1,
				)
				for ii in range(msize):
					V[ii, ii] = tau2s[i]
				cm_scale = -1 * tau2s[i] * tau2s[i] / sigma2s[i]
				blas.daxpy(
					&b2,
					&cm_scale,
					&XATXA[0,0],
					&inc_1,
					&V[0,0],
					&inc_1,
				)
				cm_scale = tau2s[i] * tau2s[i] * tau2s[i] / (sigma2s[i] * sigma2s[i])
				blas.daxpy(
					&b2,
					&cm_scale,
					&Vs[0,0],
					&inc_1,
					&V[0,0],
					&inc_1,
				)
				# ### DEBUGGING delete later
				# sigma2 = sigma2s[i]
				# tau2 = tau2s[i]
				# XA = XAT_arr.T[:, 0:msize]
				# k = msize
				# Sigma11 = sigma2 * np.eye(n) + tau2 * np.dot(XA, XA.T)
				# Sigma12 = tau2 * XA
				# Sigma22 = tau2 * np.eye(k)
				# Sigma = np.concatenate(
				# 	[np.concatenate([Sigma11, Sigma12], axis=1),
				# 	np.concatenate([Sigma12.T, Sigma22], axis=1)],
				# 	axis=0
				# )
				# expected = Sigma22 - np.dot(np.dot(Sigma12.T, np.linalg.inv(Sigma11)), Sigma12)
				# # QA = np.eye(k) + tau2 / sigma2 * np.dot(XA.T, XA)
				# # QAI = np.linalg.inv(QA)
				# # XATXA = np.dot(XA.T, XA)
				# # QA_guess = tau2s[i] / sigma2s[i] * np.dot(XAT_arr, XAT_arr.T) + np.eye(bsize)
				# # QA_guess = QA_guess[0:msize][:, 0:msize]
				# # QAI = np.linalg.inv(QA_guess)
				# # XATXA_g = XATXA_arr[0:msize][:, 0:msize]
				# # expected = tau2s[i] * np.eye(msize) - tau2s[i] / (sigma2s[i]**2) * XATXA_g
				# # expected = expected + cm_scale * np.dot(XATXA_g, np.dot(QAI, XATXA_g))
				# result = V_arr[0:msize][:, 0:msize]
				# print(f"C4 V-expected=\n{result - expected}")

				# 5. Set V to be cholesky decomp of cond. variance for cholesky sampling
				lapack.dpotrf(
					triang_l, # UPLO, upper vs. lower triangle of A
					&msize, # dim of matrix
					&V[0, 0], # matrix to perform cholesky decomp on
					&bsize, # LDA = leading dim of QA in calling program
					&INFO # error output
				)
				### FOR DEBUGGING DELETE LATER
				if INFO != 0:
					raise RuntimeError(f"dpotrf exited with INFO={INFO}. Try setting bsize=1.")
				# ### DEBUGGING delete later
				# QA_guess = tau2s[i] / sigma2s[i] * np.dot(XAT_arr, XAT_arr.T) + np.eye(bsize)
				# QA_guess = QA_guess[0:msize][:, 0:msize]
				# QAI = np.linalg.inv(QA_guess)
				# XATXA_g = XATXA_arr[0:msize][:, 0:msize]
				# cond_var = tau2s[i] * np.eye(msize) - tau2s[i]*tau2s[i] / (sigma2s[i]) * XATXA_g
				# cond_var = cond_var + cm_scale * np.dot(XATXA_g, np.dot(QAI, XATXA_g))
				# print(np.linalg.eigh(cond_var)[0].min())
				# expected = np.linalg.cholesky(cond_var)
				# result = V_arr[0:msize][:, 0:msize]
				# print(f"C5 V-expected=\n{result.T - expected}")

				### Sample i.i.d. std normals
				for jj in range(msize):
					beta_next[jj] = np.random.randn()
				
				### DEBUGGING ONLY save this
				beta_ns = beta_next.copy()

				# Apply V to beta_next
				blas.dtrmm(
					triang_l, # 'L' (left: op(A) @ B) or 'R' (right: B @ op(A))
					triang_l, # upper or lower for triang matrix
					trans_n, # transA, transpose A or not (I think no)
					trans_n, # diag, n means not unit triangular
					&msize, # M = B.shape[0]
					&inc_1, # N = op(B).shape[1]
					&one, # alpha
					&V[0,0], # A
					&bsize, # LDA first dim of A in calling program
					&beta_next[0], # B
					&bsize, # LDB first dim of B in calling program
				)

				# Add mu
				blas.daxpy(
					&msize,
					&one,
					&mu[0],
					&inc_1,
					&beta_next[0],
					&inc_1
				)

				### DEBUGGING ONLY 
				# QA_guess = tau2s[i] / sigma2s[i] * np.dot(XAT_arr, XAT_arr.T) + np.eye(bsize)
				# QA_guess = QA_guess[0:msize][:, 0:msize]
				# QAI = np.linalg.inv(QA_guess)
				# XATXA_g = XATXA_arr[0:msize][:, 0:msize]
				# cond_var = tau2s[i] * np.eye(msize) - tau2s[i]*tau2s[i] / (sigma2s[i]) * XATXA_g
				# cond_var = cond_var + cm_scale * np.dot(XATXA_g, np.dot(QAI, XATXA_g))
				# EV = np.linalg.cholesky(cond_var)
				# expected = np.dot(EV, beta_ns[0:msize]) + mu_arr[0:msize]
				# result = beta_next_arr[0:msize]
				# print(f"result-expected, {result-expected}, result={result}, expected={expected}")


				# Loop through and reset beta
				ii = 0
				for jj in model_comb:
					j = blocks[bi, jj]
					betas[i, j] = beta_next[ii]
					# Update residuals for lin. reg.
					if probit != 1:
						neg_betaj = -1 * beta_next[ii]
						blas.daxpy(
							&n,
							&neg_betaj,
							&XT[j,0],
							&inc_1,
							&r[0],
							&inc_1
						)
					# For probit, update mu_pred
					else:
						blas.daxpy(
							&n,
							&betas[i, j],
							&XT[j,0],
							&inc_1,
							&mu_pred[0],
							&inc_1
						)
					ii += 1

				# For probit, update latents
				if probit == 1:
					for it in range(n):
						Y_latent[i, it] = sample_truncnorm(
							mean=mu_pred[it], var=sigma2, b=0, lower_interval=z[it] 
						)
						y[it] = Y_latent[i, it]
						r[it] = y[it] - mu_pred[it]

				# Check residuals are correct
				#r_t = y - np.dot(X, betas_arr[i])
				#diff = np.abs(r_arr - r_t).mean()
				#print(f"msize={len(model_comb)}, bext_next={beta_next_arr[0:msize]}, i={i}, mean resid diff={diff}")

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

		# Reset logodds
		logp0 = log(p0s[i])
		log1p0 = log(1 - p0s[i])

		# Set new betas, p0s to be old values (temporarily)
		if i != N - 1:
			betas[i+1] = betas[i]
			p0s[i+1] = p0s[i]
			sigma2s[i+1] = sigma2s[i]
			tau2s[i+1] = tau2s[i]


	# Return output, including latents for probit
	output = {"betas":betas_arr, "p0s":p0s_arr, "tau2s":tau2s_arr, "sigma2s":sigma2s_arr}
	if probit == 1:
		output['y_latent'] = Y_latent_arr
	return output

	
