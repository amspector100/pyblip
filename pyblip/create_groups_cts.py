""" 
Creates candidate groups when signals could appear anywhere in a
continuous d-dimensional space. 
"""
import time
import numpy as np
import networkx as nx

import warnings
import itertools
from .utilities import elapsed
from .create_groups import CandidateGroup
from . import create_groups

def normalize_locs(locs):
	"""
	Paramters
	---------
	locs : np.array
		A (N, num_disc, d)-dimensional array. Here, N is the
		number of samples from the posterior, d is the number
		of dimensions of the space, and each point corresponds
		to a signal in a particular posterior sample.

	Returns
	-------
	locs : np.array
		locs, but normalized such that all values lie in [0,1].
		NAs are converted to -1.
	shifts : np.array
		``d``-dimensional array corresponding to the shifts applied
		in the normalizatin. 
	scales : np.array
		``d``-dimension array corresponding to the scales in the 
		normalization.
	"""
	min_val = np.nanmin(np.nanmin(locs, axis=0), axis=0)
	max_val = np.nanmax(np.nanmax(locs, axis=0), axis=0)
	shifts = min_val
	scales = max_val - min_val
	norm_locs = (locs - shifts) / scales
	return norm_locs, shifts, scales


def grid_peps(
	locs,
	grid_sizes,
	extra_centers=None,
	max_pep=0.25,
	log_interval=None,
	time0=None
):
	"""
	Parameters
	----------
	locs : np.array
		A (N, num_disc, d)-dimensional array. Here, N is the
		number of samples from the posterior, d is the number
		of dimensions of the space, and each point corresponds
		to a signal in a particular posterior sample. 
	grid_sizes : list or np.ndarray
		List of grid-sizes to split up the locations.
	extra_centers : np.ndarray
		A (ncenters, d)-dimensional array. At each resolution,
		a candidate groups will be computed with centers at these
		locations.

	Returns
	-------
	cand_groups : list
		A list of CandidateGroup objects.
	"""
	if time0 is None:
		time0 = time.time()

	if np.nanmin(locs) < -1e10 or np.nanmax(locs) > 1 + 1e-10:
		raise ValueError(
			f"locs are not normalized: apply create_groups_cts.normalize_locs first."
		)

	# Create PIPs
	pips = dict()
	N = locs.shape[0]
	ndisc = locs.shape[1]
	d = locs.shape[2]
	if extra_centers is not None:
		n_extra_cent = extra_centers.shape[0]

	for j in range(N):
		# Ignore dummy discoveries
		active = ~np.any(np.isnan(locs[j]), axis=1)
		samples = locs[j, active]
		# Loop through grid sizes and find centers
		for gsize in grid_sizes:
			# sum(active) x d
			centers = np.floor(
				samples * gsize
			)
			centers = centers.astype(float) / gsize

			# Eliminate duplicates
			centers = list(set(tuple(list(c)) for c in centers))
			for c in centers:
				key = c + (gsize,)
				if key not in pips:
					pips[key] = 1 / N
				else:
					pips[key] += 1 / N

		# Repeat for manual centers
		if extra_centers is not None and n_extra_cent > 0 and samples.shape[0] > 0:
			dists = extra_centers.reshape(n_extra_cent, 1, d) - samples.reshape(1, -1, d)
			min_dists = np.abs(dists).max(axis=2).min(axis=1)
			for gsize in grid_sizes:
				radius = 1 / (2*gsize)
				for nc in np.where(min_dists <= radius)[0]:
					extra_corner = np.around(extra_centers[nc] - radius, 10)
					key = tuple(extra_corner) + (gsize,)
					if key not in pips:
						pips[key] = 1 / N
					else:
						pips[key] += 1 / N

		if log_interval is not None:
			if j % log_interval == 0:
				print(f"Computing PEPs: finished with {j} / {N} posterior samples at {elapsed(time0)}.")

	# Filter
	filtered_peps = {}
	for key in pips.keys():
		pep = 1 - pips[key]
		if pep <= max_pep:
			filtered_peps[key] = pep

	return filtered_peps

def calculate_overlaps(cent, radii):
	"""
	cent : p-length array of centers of boxes
	radii : p-length array of radius of box
	"""
	dists = np.abs(cent.reshape(-1, 1) - cent.reshape(1, -1))
	deltas = radii.reshape(-1, 1) + radii.reshape(1, -1)
	return dists < deltas

def grid_peps_to_cand_groups(
	filtered_peps, 
	time0=None,
	max_blip_size=1000,
	verbose=False
):
	"""
	Turns the output of the ``grid_peps`` function into
	a list of list of CandidateGroups. Each sub-list corresponds
	to a list of completely disconnected CandidateGroups which
	can be fed to BLiP separately (this saves computation).

	filtered_peps : dict
		An output of the ``grid_peps`` function.
	time0 : float
		The initial time the analysis started, useful for logging.
	max_blip_size : int
		Maximum size of a problem that can be fed into BLiP.
	verbose: bool
		If True, will report progress over time. Default: False.
	"""
	if time0 is None:
		time0 = time.time()

	# Step 1: compute adjacency matrix
	ngroups = len(filtered_peps)
	if ngroups == 0:
		return [], []
	if ngroups > 50000:
		warnings.warn(f"Computing adjacency matrix may be too inefficient for {ngroups} candidate groups.")
	keys = sorted(filtered_peps.keys())
	d = len(keys[0]) - 1 # dimensionality of problem

	if verbose:
		print(f"Constructing constraint matrix with ngroups={ngroups} at {elapsed(time0)}")
	constraints = np.ones((ngroups, ngroups)).astype(bool)
	centers = np.zeros((d, ngroups))
	centers[:] = np.nan
	radii = np.array([1.0 / float(k[-1]) for k in keys]) / 2
	for j in range(d):
		# Extract corners/centers of hypothesis
		corners = np.array([k[j] for k in keys])
		centers[j] = corners + radii
		if centers[j].max() > 1 or centers[j].min() < 0:
			print(f"{centers[j].max()}, {centers[j].min()}")
			raise ValueError("centers must be between 0 and 1 but this is not true")
		# Find overlaps and add to constraints
		constraints = constraints & calculate_overlaps(centers[j], radii)

	# Step 2: Split problem into connected components
	if verbose:
		print(f"Isolating connected components at {elapsed(time0)}")
	G = nx.Graph(constraints)
	components = list(nx.algorithms.components.connected_components(G))
	merged_components = [[]]
	for c in components:
		if len(merged_components[-1]) + len(c) > max_blip_size:
			merged_components.append([])
		merged_components[-1].extend(list(c))
		
	# Step 3: construct cand_groups for BLiP
	all_cand_groups = []
	peps_arr = np.array([filtered_peps[k] for k in keys])
	for component in merged_components:
		component_cand_groups = []
		for j in component:
			group = set(np.where(constraints[j])[0].tolist())
			data_dict = dict(radius=radii[j])
			for k in range(d):
				data_dict[f'dim{k}'] = centers[k, j]
			component_cand_groups.append(
				CandidateGroup(
					group=group,
					pep=peps_arr[j],
					data=data_dict
				)
			)
		all_cand_groups.append(component_cand_groups)

	# Return
	return all_cand_groups, merged_components