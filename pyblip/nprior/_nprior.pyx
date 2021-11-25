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
from ..cython_utils._truncnorm import rationalapprox, random_uniform, sample_truncnorm

cdef double prof_exp(double x):
	return exp(x)

cdef double prof_log(double x):
	return log(x)

# log normal cdf
cdef double normcdf(double z):
	return 0.5 * erfc(-z * M_SQRT1_2)

@cython.nonecheck(False)
@cython.cdivision(True)
cdef double norm_quantile_apprx(double p):
	"""
	approximates normal quantile.
	Absolute value of error should be < 4.5 e-4.
	"""
	if (p < 0.5):
		return -1*rationalapprox(sqrt(-2.0*log(p)))
	else:
		return rationalapprox(sqrt(-2.0*log(1-p)))

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def _nprior_sample(
	int N,
	double[:, ::1] X,
	double[::1] y,
	double tauw2,
	double p0_init,
	double min_p0,
	int update_p0,
	double sigma_a0,
	double sigma_b0,
	double alpha0_a0,
	double alpha0_b0,
	int sigma_prior_type,
	int joint_sample_W,
	int group_alpha_update,
	int log_interval, # how often to log
	double time0 # start time
):
	# Precomputation is done in numpy
	# 1. Collect useful quantities
	if time0 is None:
		time0 = time.time()
	cdef int n = X.shape[0]
	cdef int p = X.shape[1]
	cdef double[:, ::1] XT = np.ascontiguousarray(X.T)
	cdef double[::1] Xl2s = np.power(
		X, 2
	).sum(axis=0)
	assert Xl2s.shape[0] == p
	cdef double min_alpha0 = norm_quantile_apprx(min_p0)
	cdef double min_delta # equals min_alpha0 - alpha0
	
	# We shuffle this using numpy so not a memview
	cdef np.ndarray [long, ndim=1] inds = np.arange(p)

	# 2. We know how many uniforms/invGammas we need
	# so we allocate memory / do samplign in advance
	cdef double sigma_a = n / 2.0 + sigma_a0
	if sigma_prior_type != 0:
		sigma_a += p / 2.0
	cdef np.ndarray [double, ndim=1] invgammas = scipy.stats.invgamma(sigma_a).rvs(N)

	# 3. Initialize final results
	cdef double delta = 0  # for group move of alpha0s
	cdef np.ndarray [double, ndim=2] alphas_arr = 1 / sqrt(2*log(p)) * np.random.randn(N, p)
	cdef double[:, ::1] alphas = alphas_arr
	cdef np.ndarray[double, ndim=2] ws_arr = 0.1 * np.random.randn(N, p)
	cdef double[:, ::1] ws = ws_arr
	cdef np.ndarray [double, ndim=2] betas_arr = np.zeros((N, p))
	cdef double[:, ::1] betas = betas_arr
	cdef np.ndarray[double, ndim=1] sigma2s_arr = np.ones((N,))
	cdef double[::1] sigma2s = sigma2s_arr
	cdef np.ndarray[double, ndim=1] alpha0s_arr = np.zeros((N,))
	cdef double[::1] alpha0s = alpha0s_arr
	cdef np.ndarray[double, ndim=1] p0s_arr = np.zeros((N,))
	cdef double[::1] p0s = p0s_arr

	# 4. Initialize residuals
	cdef np.ndarray [double, ndim=1] r = np.zeros(n)
	cdef double[::1] r_v = r # the memview of r

	# Initialize alphas and sigma
	sigma2s[0] = np.random.randn()**2
	p0s[0] = p0_init
	alpha0s[0] = norm_quantile_apprx(p0_init)

	# Iterate
	cdef int i, j, jn
	cdef double objective
	for i in range(N):
		# Note that at this point in time,
		# alphas[i] has been initialized to equal
		# alphas[i-1] but we'll resample again.
		# same holds for alpha0s, p0s, sigma2s

		# Step 1: possibly jointly sample w conditional on y, alpha, sigma2
		if joint_sample_W == 1:
			ws[i] = _sample_w(
				XT=XT,
				y=y,
				alpha=alphas[i],
				sigma2=sigma2s[i],
				alpha0=alpha0s[i],
				tauw2=tauw2
			)

		# Create new beta and residuals
		for j in range(p):
			betas[i, j] = ws[i, j] * fmax(alphas[i, j] - alpha0s[i], 0)
		# Equivalent python: r = y - np.dot(X, betas[i])
		for jn in range(n):
			r_v[jn] = y[jn]
		blas.dgemv(
			trans_t, #TRANS
			&p, # M
			&n, # N
			&neg1, # ALPHA
			&X.T[0, 0], # A (this may need to be fortran-contiguous)
			&p, # LDA
			&betas[i, 0], # X
			&inc_1, # INCX
			&one, # BETA
			&r_v[0], #Y
			&inc_1 #INCY
		)

		# Randomize order of indices
		np.random.shuffle(inds)
		# Resample alphas and sequentially
		for j in inds:
			# Reset residual to ignore feature j
			# equivalent: 
			blas.daxpy(&n, &betas[i,j], &XT[j,0], &inc_1, &r_v[0], &inc_1)

			# Resample alphaj
			alphas[i, j] = _update_coordinate_relu(
				j=j,
				r_v=r_v, 
				wj=ws[i,j],
				sigma2=sigma2s[i],
				p0=p0s[i],
				alpha0=alpha0s[i],
				XT=XT,
				Xj_l2norm=Xl2s[j]
			)

			# Resample wj
			ws[i, j] = _update_wj(
				j=j,
				r_v=r_v,
				XT=XT,
				alphaj=alphas[i, j], 
				sigma2=sigma2s[i],
				Xj_l2norm=Xl2s[j],
				alpha0=alpha0s[i],
				tauw2=tauw2,
				sigma_prior_type=sigma_prior_type
			)

			# Reset residual to account for feature j
			betas[i, j] = -1 * ws[i, j] * fmax(alphas[i, j] - alpha0s[i], 0)
			blas.daxpy(&n, &betas[i,j], &XT[j,0], &inc_1, &r_v[0], &inc_1)
			betas[i, j] = -1 * betas[i, j]

		# print(f"About to check the assertion at it={i}...")
		# rreal = y - np.dot(X, betas[i])
		# absdiff = np.around(np.abs(rreal - r), 5)
		# if not np.all(absdiff < 1e-5):
		# 	print(absdiff)
		# 	raise ValueError("sad")

		# Resample scalar parameters
		sigma2s[i] = _sample_sigma2(
			w=ws[i],
			r_v=r_v,
			sigma_b0=sigma_b0,
			tauw2=tauw2, 
			invgamma=invgammas[i],
			sigma_prior_type=sigma_prior_type
		)
		if update_p0 != 0:
			### option 1: update p0 directly
			if group_alpha_update == 0:
				p0s[i] = _sample_p0(
					alphas=alphas[i],
					alpha0=alpha0s[i]
				)
				p0s[i] = fmax(min_p0, p0s[i])
				alpha0s[i] = norm_quantile_apprx(p0s[i])
			### option 2: group update 
			else:
				min_delta = min_alpha0 - alpha0s[i]
				delta = _sample_delta(
					alphas=alphas[i],
					alpha0=alpha0s[i]
				)
				delta = fmax(min_delta, delta)
				alpha0s[i] += delta
				p0s[i] = normcdf(alpha0s[i])
				for j in range(p):
					alphas[i, j] += delta

		# Initialize next set of alphas, sigmas
		if i < N - 1:
			alphas[i+1] = alphas[i]
			alpha0s[i+1] = alpha0s[i]
			p0s[i+1] = p0s[i]
			sigma2s[i+1] = sigma2s[i]

		# Possibly log progress
		if i % log_interval == 0 and log_interval < N:
			num_active = np.sum(betas_arr[i] != 0)
			msg = f"Starting it={i+1} at time={np.around(time.time()-time0, 2)}"
			msg += f", num_active={num_active}, p0={p0s[i]}"
			print(msg)


	return dict(
		alphas=alphas_arr,
		ws=ws_arr,
		betas=betas_arr,
		sigma2s=sigma2s_arr,
		alpha0s=alpha0s_arr,
		p0s=p0s_arr
	)

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double[::1] _sample_w(
	double[:, ::1] XT,
	double[::1] y,
	double[::1] alpha,
	double sigma2,
	double alpha0,
	double tauw2
):
	"""
	Samples W conditional on y, alpha, sigma2.
	Runtime is O(min(n^2p, A^3 + nA)) where A 
	is the number of active variables.

	use_bcm16 : bool
		If True, uses the algorithm from 
		https://arxiv.org/pdf/1506.04778.pdf
		to run in O(n^2 p). By default checks which
		has a faster (asymptotic) runtime.
	"""
	cdef:
		int jwp, act_ind # iterators
		int p = XT.shape[0]
		int n = XT.shape[1]
		int num_active = 0

	# sample independent normals for non-active vars
	cdef double[::1] w_out = np.random.randn(p) / sqrt(sigma2 / tauw2)

	# Find active coordinates (T is flags for active coords)
	cdef double[::1] T = np.zeros((p,))
	for jwp in range(p):
		T[jwp] = fmax(alpha[jwp] - alpha0, 0)
		if T[jwp] > 0:
			num_active += 1

	# If no active coordinates, we're done :)
	if num_active == 0:
		return w_out

	# Construct Phi = X * ReLU(alphas - alpha0)
	cdef np.ndarray[double, ndim=2] Phi_arr = np.zeros((num_active, n))
	cdef double[:, ::1] Phi = Phi_arr
	act_ind = 0 # counts which active ind we are on
	for jwp in range(p): # iterates through initial indexes
		if T[jwp] > 0:
			# Equivalent to: Phi[act_ind] = T[jwp] * XT[jwp] 
			blas.daxpy(
				&n,
				&T[jwp],
				&XT[jwp, 0],
				&inc_1,
				&Phi[act_ind, 0],
				&inc_1
			)
			act_ind += 1

	# Equivalent: PTP = np.dot(Phi, Phi.T)
	cdef np.ndarray[double, ndim=2] PTP_arr = np.zeros((num_active, num_active))
	cdef double [:, ::1] PTP = PTP_arr
	blas.dgemm(
		trans_t, # TRANSA 
		trans_n, # TRANSB
		&num_active, # M = #rows of final result 
		&num_active, # N = #cols of final result
		&n, # K = num cols of op(A) 
		&one, # alpha (scale)
		&Phi.T[0,0], # A
		&n, # LDA first dim of A
		&Phi.T[0,0], # B
		&n, # LDB first dim of B 
		&zero, # beta scale 
		&PTP.T[0,0], # output C
		&num_active # first dim of C
	)
	# cdef np.ndarray[double, ndim=2] PTP2 = np.dot(
	# 	Phi_arr, Phi_arr.T
	# )
	# print(np.abs(PTP2 - PTP_arr).mean())

	# Equivalent: PTP = PTP + np.eye(num_active) / tauw2)
	for act_ind in range(num_active):
		PTP[act_ind, act_ind] += 1 / tauw2
	
	# Now: PTP is the precision matrix
	cdef int info = 0
	# L is lower-triangular and C-contiguous
	cdef np.ndarray[double, ndim=2] L_arr = scipy.linalg.cholesky(PTP_arr).T
	#L_arr0 = L_arr.copy()
	cdef double[:, ::1] L = L_arr
	# equiv: L = np.linalg.inv(L)
	# L is still lower-triangular and C-contiguous
	lapack.dtrtri(
		triang_u, # UPLO: make this upper because L is lower and c-contig
		trans_n, # diag, n means not diagonal
		&num_active, # N dimension
		&L[0,0], # A
		&num_active, # LDA
		&info # output
	)
	if info != 0:
		raise ValueError(f"inversion of triangular matrix L failed, info={info}")
	#print("here0", np.abs(np.linalg.inv(L_arr0) - L_arr).mean())
	# L_arr2 = L_arr.copy()

	# Calculate tmu, the mean of the multivariate normal
	# equiv: tmu = np.dot(inv(PTP), np.dot(Phi, y))
	# step 1: equiv tmu = np.dot(Phi, y)
	cdef double[::1] tmu = np.zeros(num_active)
	blas.dgemv(
		trans_t, #TRANS
		&n, # M
		&num_active, # N
		&one, # ALPHA
		&Phi.T[0,0], # A
		&n, # LDA
		&y[0], # X
		&inc_1, # INCX
		&zero, # BETA
		&tmu[0], #Y
		&inc_1 #INCY
	)
	# tmu1 = np.dot(Phi, y)
	# print("here1", np.mean(np.abs(tmu - tmu1)))
	# step 2: equiv: tmu = np.dot(L, tmu)
	# tmu2 = np.dot(L_arr2, tmu)
	blas.dtrmv(
		triang_u, #UPLO
		trans_t, # TRANS
		trans_n, # diag (not unit)
		&num_active, # N
		&L.T[0,0], # A
		&num_active, # LDA
		&tmu[0], #X
		&inc_1 #INCX
	)
	# print("here2", np.mean(np.abs(tmu - tmu2)))
	# step 3: equiv: tmu = np.dot(L.T, tmu)
	# tmu3 = np.dot(L_arr2.T, tmu)
	blas.dtrmv(
		triang_u, #UPLO
		trans_n, # TRANS
		trans_n, # diag (not unit)
		&num_active, # N
		&L.T[0,0], # A
		&num_active, # LDA
		&tmu[0], #X
		&inc_1 #INCX
	)

	# sample from MVN now that we have mean/cholesky decopm of cov
	cdef double[::1] wt = np.random.randn(num_active)
	# equivalent to wt = np.dot(L.T, wt)
	# we could allocate a bit less memory and save one dtrmv call
	# if we added the noise to tmu before applying L.T, but 
	# this code is (slightly) more interpretable and not much
	# slower 
	blas.dtrmv(
		triang_u, #UPLO
		trans_n, # TRANS
		trans_n, # diag (not unit)
		&num_active, # N
		&L.T[0,0], # A
		&num_active, # LDA
		&wt[0], #X
		&inc_1 #INCX
	)
	# equiv: wt = wt + tmu
	#wt2 = np.asarray(wt) + np.asarray(tmu)
	#print(np.abs(wt2 - wt).mean())
	blas.daxpy(&num_active, &one, &tmu[0], &inc_1, &wt[0], &inc_1)

	# fill in active coordinates of w_out
	act_ind = 0
	for jwp in range(p):
		if T[jwp] > 0:
			w_out[jwp] = wt[act_ind]
			act_ind += 1

	return w_out


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef _sample_p0(
	double [::1] alphas,
	double alpha0
):
	cdef:
		int j # iterator
		int p = alphas.shape[0]
		int num_active = 0
		double new_p0
	# Count number of active coordinates
	for j in range(p):
		if alphas[j] > alpha0:
			num_active += 1
	# sample from beta
	new_p0 = np.random.beta(
			a=1 + p - num_active, #alpha0_a0
			b=1 + num_active # alpha0_b0
	)
	return new_p0


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _sample_delta(
	double [::1] alphas,
	double alpha0
):
	""" group-move of alpha and alpha0.
	Note this assumes the prior on the sparsity
	is uniform."""
	cdef:
		int jdp # iterator
		double mean_alpha0 = 0# conditional mean
		int p = alphas.shape[0] # dimension
		double pf = p # avoid integer division errors
		double sd_alpha0 = 1 / sqrt(pf + 1) # conditional sd
		#double delta, alpha_l2
	# Calculate mean which is np.dot(alpha, alpha) + alpha0
	#alpha_l2 = blas.dnrm2(&p, &alphas[0], &inc_1)
	#print(np.dot(alphas, alphas), alpha_l2)
	for jdp in range(p):
		mean_alpha0 += alphas[jdp]
	mean_alpha0 = -1*(mean_alpha0 + alpha0) / (pf + 1)

	#mean_alpha0 = (alpha_l2 * alpha_l2 + alpha0) / (pf + 1)
	#print(f"alphamax={np.asarray(alphas).max()}, pf={pf}")
	#print(f"mean_alpha0={mean_alpha0}")
	# Sample and return
	delta = sd_alpha0 * np.random.randn() + mean_alpha0
	return delta



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double _sample_sigma2(
	double[::1] w,
	double[::1] r_v,
	double sigma_b0,
	double tauw2,
	double invgamma,
	int sigma_prior_type
):
	cdef:
		double b = 1# scale parameter for gamma
		int n = r_v.shape[0]
		int p = w.shape[0]
		double r2 # ||r_v||_2^2
		double w2 # ||w||_2^2

	# Calculate l2 norms of r, w
	r2 = blas.dnrm2(&n, &r_v[0], &inc_1)
	r2 = r2 * r2
	if sigma_prior_type != 0:
		w2 = blas.dnrm2(&p, &w[0], &inc_1)
		w2 = w2 * w2

	# Compute b parameter and rescale
	b = r2 / 2 + sigma_b0
	if sigma_prior_type != 0:
		b = b + w2 / (2.0*tauw2)

	return b * invgamma

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double _update_wj(
	int j,
	double[::1] r_v,
	double[:, ::1] XT,
	double alphaj, 
	double sigma2,
	double Xj_l2norm,
	double alpha0,
	double tauw2,
	int sigma_prior_type
):
	"""z
	ReLU nprior update for Wj
	computational complexity: O(n)
	"""
	cdef int n = r_v.shape[0]
	cdef double Tj = fmax(alphaj - alpha0, 0)

	# Calculate inverse variance, slightly depends
	# on whether prior for sigma and tauw2 are related
	cdef double sigma02 = sigma2
	if sigma_prior_type == 0:
		sigma02 = 1.0
	cdef double vj = Xj_l2norm * Tj * Tj + (sigma02 / tauw2)

	# Calculate mean
	# equivalent to np.dot(XT[j], r_v) but faster
	cdef double mj
	if Tj > 0:
		mj = blas.ddot(
			&n,
			&r_v[0],
			&inc_1,
			&XT[j, 0],
			&inc_1
		)
		mj = mj * Tj / vj
	else:
		mj = 0

	# Sample
	cdef double z = np.random.randn()
	z = sqrt(sigma2 / vj) * z + mj
	return z


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double _update_coordinate_relu(
	int j,
	double[:] r_v, 
	double wj,
	double sigma2,
	double p0,
	double alpha0,
	double[:, ::1] XT,
	double Xj_l2norm
):
	""" 
	Samples alphaj from a mixture of truncated normals
	Complexity: O(n)
	"""
	cdef: 
		double r2
		double rplus2
		double log_kappa_num
		double log_kappa_denom
		double ratio # kappa = ratio / (ratio + 1)
		double kappa
		int n = r_v.shape[0]

	# 1. preliminaries
	# l2 norm of r
	r2 = blas.dnrm2(&n, &r_v[0], &inc_1)
	r2 = r2 * r2
	# r = r + Xj * alpha0 * wj
	cdef double r_scale = alpha0 * wj
	if r_scale != 0:
		blas.daxpy(&n, &r_scale, &XT[j, 0], &inc_1, &r_v[0], &inc_1)

	# 2. Compute mixture parameters
	cdef double t_alphaj_denom = wj*wj*Xj_l2norm + sigma2
	cdef double t_sigma2 = sigma2 / t_alphaj_denom
	cdef double t_alphaj = blas.ddot(
		&n,
		&r_v[0],
		&inc_1,
		&XT[j, 0],
		&inc_1
	)
	t_alphaj = wj * t_alphaj / t_alphaj_denom

	# 3. Compute mixture weights
	rplus2 = blas.dnrm2(&n, &r_v[0], &inc_1)
	rplus2 = rplus2 * rplus2
	log_kappa_num = prof_log(p0)
	log_kappa_num -= r2/(2*sigma2)

	# split log_kappa_denom into two terms
	log_kappa_denom = prof_log(1-normcdf((alpha0 - t_alphaj)/sqrt(t_sigma2)))
	log_kappa_denom += prof_log(t_sigma2)/2 + (
		t_alphaj*t_alphaj/(2*t_sigma2) - rplus2/(2*sigma2)
	)

	# 4. reset residuals r
	if r_scale != 0:
		r_scale = - 1 * r_scale
		blas.daxpy(&n, &r_scale, &XT[j, 0], &inc_1, &r_v[0], &inc_1)


	# 5. Select mixture and sample from truncnormal
	ratio = prof_exp(log_kappa_num - log_kappa_denom)
	kappa = ratio / (one + ratio) ### TO FIX
	#print(f"for j={j}, wj={wj}, kappa={kappa}, alpha0={alpha0}")
	cdef double u = random_uniform()
	cdef double new_alphaj
	if u < kappa:
		new_alphaj = sample_truncnorm(
			mean=zero,
			var=one,
			b=alpha0,
			lower_interval=inc_1
		)
		#assert new_alphaj <= alpha0
	else:
		#print(f"cond_mean in update_relu={t_alphaj}")
		new_alphaj = sample_truncnorm(
			mean=t_alphaj,
			var=t_sigma2,
			b=alpha0,
			lower_interval=inc_0
		)
		#assert new_alphaj >= alpha0
	return new_alphaj