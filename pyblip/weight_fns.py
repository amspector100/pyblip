"""
A few default choices of weight function to define power.
"""
import numpy as np

def inverse_size(cand_group):
	"""
	Weights candidate groups by 1 / len(cand_group.group)

	Parameters
	----------
	cand_group : CandidateGroup
		The candidate group

	Returns
	-------
	weight : float
		The associated weight when calculating power.
	"""
	return 1.0 / len(cand_group.group)

def log_inverse_size(cand_group):
	"""
	Weights candidate groups by 1 / (1 + log2(len(cand_group.group)))

	Parameters
	----------
	cand_group : CandidateGroup
		The candidate group

	Returns
	-------
	weight : float
		The associated weight when calculating power.
	"""
	return 1.0 / (1 + np.log2(len(cand_group.group)))

def inverse_radius_weight(cand_group):
	"""
	Weights candidate groups by the inverse of the radius.

	Parameters
	----------
	cand_group : CandidateGroup
		The candidate group

	Returns
	-------
	weight : float
		The associated weight when calculating power.
	"""
	return 1.0 / cand_group.data['radius']

def inverse_area_weight(cand_group):
	"""
	Weights candidate groups by the inverse of their area.

	Parameters
	----------
	cand_group : CandidateGroup
		The candidate group

	Returns
	-------
	weight : float
		The associated weight when calculating power.
	"""
	return 1.0 / (cand_group.data['radius']**2)
