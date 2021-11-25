import os
import sys

# Add path to allow import of code
file_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.split(file_directory)[0]
sys.path.insert(0, os.path.abspath(parent_directory))

# Import the actual stuff
import pyblip

# for generating synthetic data
import numpy as np
from scipy import stats


def generate_regression_data(
	n=100,
	p=500,
	y_dist='gaussian', # one of gaussian, probit, binomial
	coeff_size=1,
	coeff_dist='normal',
	sparsity=0.05,
	a=5,
	b=1,
	max_corr=0.99,
	permute=False,
):
	# Generate X-data
	rhos = stats.beta.rvs(size=p-1, a=a, b=b)
	rhos = np.minimum(rhos, max_corr)
	X = np.random.randn(n, p)
	for j in range(1, p):
		X[:, j] = rhos[j-1]*X[:, j-1] + np.sqrt(1 - rhos[j-1]**2) * X[:, j]
	
	# Possibly permute to make slightly more realistic
	if permute:
		perminds = np.arange(p)
		np.random.shuffle(perminds)
		X = np.ascontiguousarray(X[:, perminds])

	# Create sparse coefficients,
	beta = np.zeros(p)
	k = np.around(sparsity * p).astype(int)
	nonnull_coefs = np.sqrt(coeff_size) * np.random.randn(k)
	beta[np.random.choice(np.arange(p), k, replace=False)] = nonnull_coefs
	print(np.mean(beta != 0), sparsity, p)

	# Create Y
	mu = np.dot(X, beta)
	if y_dist == 'gaussian' or y_dist=='linear':
		y = mu + np.random.randn(n)
	elif y_dist == 'probit':
		y = ((mu + np.random.randn(n)) < 0).astype(float)
	elif y_dist == 'binomial':
		probs = np.exp(mu)
		probs = probs / (1.0 + probs)
		y = np.random.binomial(1, probs)
	else:
		raise ValueError(f"unrecognized y_dist=={y_dist}")

	return X, y, beta

