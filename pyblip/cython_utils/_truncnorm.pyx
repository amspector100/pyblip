# cython: profile=False

######################################################################
# Custom (faster) truncated normal sampler.
# See http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.26.6892
######################################################################
cimport cython
import numpy as np
cimport numpy as np

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log, exp, fabs, sqrt

@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double random_uniform():
	cdef double r = rand()
	return r / RAND_MAX

@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double rationalapprox(double t):
	"""
	a helper function for approximation
	of the normal CDF.
	See
	https://www.johndcook.com/blog/cpp_phi_inverse/
	"""
	cdef double c2, c1, c0, d2, d1, d0, frac
	c0 = 2.515517
	c1 = 0.802853
	c2 = 0.010328
	d0 = 1.432788
	d1 = 0.189269
	d2 = 0.001308
	frac = (c2*t + c1)*t + c0
	frac = frac / (((d2*t + d1)*t + d0)*t + 1.0)
	return t - frac

@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _expo_tn_sampler(double a):
	"""
	Samples Z \sim N(0,1) | Z \in [a,infty).
	"""
	cdef double rho
	cdef double u 
	cdef double y # ~ expo(a)
	while True:
		y = random_uniform()
		y = -1*log(y) / a
		rho = exp(-0.5*y*y)
		u = random_uniform()
		if u <= rho:
			return y + a

@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _norm_tn_sampler(double a):
	"""
	Samples Z \sim N(0,1) | Z \in [a,infty)
	"""
	cdef double z;
	while True:
		z = np.random.randn()
		if a >= 0:
			z = fabs(z) 
		if z >= a:
			return z

@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _sample_truncnorm_std(double a):
	"""
	Samples Z \sim N(0,1) | Z \in [a, \infty)
	efficiencitly.
	"""

	# constants from the paper
	cdef double t1 = 0.150
	if a >= t1:
		return _expo_tn_sampler(a)
	else:
		return _norm_tn_sampler(a)

@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double sample_truncnorm(
	double mean,
	double var,
	double b,
	int lower_interval
):
	"""
	if lower_interval == 1, samples from
	Z \sim N(mean, var) | Z \in (-infty, b)
	else, samples from 
	Z \sim N(mean, var) | Z \in (b, infty)
	"""
	scale = sqrt(var)
	cdef double a = (b - mean) / scale
	cdef double z;
	if lower_interval == 0:
		z = _sample_truncnorm_std(a)
	else:
		z = -1 * _sample_truncnorm_std(-1*a)
	return mean + scale * z