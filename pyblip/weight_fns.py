"""
A few default choices of weight function to define power.
"""
import numpy as np

def inverse_size(cand_group):
	return 1.0 / len(cand_group.group)

def log_inverse_size(cand_group):
	return 1.0 + np.log2(inverse_size(cand_group))

def inverse_radius_weight(cand_group):
	return 1.0 / cand_group.data['radius']

def inverse_area_weight(cand_group):
	return 1.0 / (cand_group.data['radius']**2)
