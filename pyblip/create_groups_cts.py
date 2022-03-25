""" 
Creates candidate groups when signals could appear anywhere in a
continuous d-dimensional space. 
"""
import time
import copy
import numpy as np
import networkx as nx
import warnings

from .utilities import elapsed
from .create_groups import CandidateGroup
from .ecc import edge_clique_cover
from collections import Counter

TOL = 1e-10

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

def _additional_circle_centers(
	samples, centers, center_list, radius, gsize
):
	N, d = samples.shape
	radius2 = np.power(radius, 2)
	# Check upper/lower/left/right
	for j in range(d):
		for offset in [-1, 1]:
			centers_new = centers.copy()
			centers_new[:, j] += offset / gsize
			# Check if points are in the offset centers
			#print(samples.shape, centers.shape, centers_new.shape)
			included = np.power(samples - centers_new, 2).sum(axis=1) <= radius2
			included = included & (np.max(centers_new, axis=1) < 1 + TOL)
			included = included & (np.min(centers_new, axis=1) > -TOL)
			center_list.extend([
				tuple(list(c)) for c in centers_new[included]
			])

	return center_list


def _find_centers(
	samples, gsize, shape
):
	# Find central boxes containing samples
	N, d = samples.shape
	# lower-left corner (comments in 2d for intuition)
	corners = np.floor(
		samples * gsize
	)
	corners = corners.astype(float) / gsize

	# Adjust to make centers and check for unexpected shapes
	centers = corners + 1 / (2 * gsize)
	center_list = [tuple(list(c)) for c in centers]
	if shape == 'circle':
		# Radius of Eucliean balls
		if d != 2:
			raise NotImplementedError(
				"Shape=circle only implemented for 2-d data, but detected {d} dimensions"
			)
		radius = np.sqrt(d) / (2 * gsize)
	elif shape == 'square':
		radius = 1 / (2 * gsize)
	else:
		raise ValueError(f"Unrecognized shape={shape}, must be one of 'square', 'circle'")

	# Add extra centers for circles only
	if shape == 'circle' and N > 0:
		center_list = _additional_circle_centers(
			samples, centers, center_list, radius, gsize 
		)

	return [tuple(np.around(center, 10)) + (radius,) for center in center_list]

def grid_peps(
	locs,
	grid_sizes,
	count_signals=False,
	extra_centers=None,
	max_pep=0.25,
	log_interval=None,
	time0=None,
	shape='square'
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
	count_signals : bool
		If True, the number of signals in each group is counted.
	extra_centers : np.ndarray
		A (ncenters, d)-dimensional array. At each resolution,
		a candidate groups will be computed with centers at these
		locations.
	shape : string
		One of ``square`` or ``circle``

	Returns
	-------
	cand_groups : list
		A list of CandidateGroup objects.
	"""
	if time0 is None:
		time0 = time.time()

	if np.nanmin(locs) < -TOL or np.nanmax(locs) > 1 + TOL:
		raise ValueError(
			f"locs are not normalized: apply create_groups_cts.normalize_locs first."
		)

	# Create PIPs
	pips = dict()
	N = locs.shape[0]
	d = locs.shape[2]
	if extra_centers is not None:
		n_extra_cent = extra_centers.shape[0]

	for j in range(N):
		# Ignore dummy discoveries
		active = ~np.any(np.isnan(locs[j]), axis=1)
		samples = locs[j, active]
		# Loop through grid sizes and find centers
		all_centers = []
		for gsize in grid_sizes:
			all_centers.extend(_find_centers(
				samples=samples, gsize=gsize, shape=shape
			))

		# Repeat for manually added centers
		all_extra_centers = []
		if extra_centers is not None and n_extra_cent > 0 and samples.shape[0] > 0:
			dists = extra_centers.reshape(n_extra_cent, 1, d) - samples.reshape(1, -1, d)
			if shape == 'square':
				dists = np.abs(dists).max(axis=2)
			else:
				dists = np.sqrt(np.power(dists, 2).sum(axis=2))
			min_dists = dists.min(axis=1)
			for gsize in grid_sizes:
				radius = 1 / (2*gsize)
				if shape == 'circle':
					radius = np.sqrt(d) * radius
				for nc in np.where(min_dists <= radius)[0]:
					key = tuple(extra_centers[nc]) + (radius,)
					# Just add this once if we are looking for the presence of a signal
					if not count_signals:
						all_extra_centers.append(key)
					# else count num. signals in this region
					else:
						nsignals = np.sum(dists[nc] <= radius)
						all_extra_centers.extend([key for _ in range(nsignals)])

		# Update PIPs
		final_centers = all_centers + all_extra_centers
		if not count_signals:
			final_centers = set(final_centers)
			for key in final_centers:
				if key not in pips:
					pips[key] = 1 / N
				else:
					pips[key] += 1 / N
		else:
			counter = Counter(final_centers)
			for key in counter:
				count = counter[key]
				if key not in pips:
					pips[key] = {count: 1 / N, 'pip': 1 / N}
				else:
					if count not in pips[key]:
						pips[key][count] = 1 / N
					else:
						pips[key][count] += 1 / N
					pips[key]['pip'] += 1 / N

		if log_interval is not None:
			if j % log_interval == 0:
				print(f"Computing PEPs: finished with {j+1} / {N} posterior samples at {elapsed(time0)}.")

	# Filter
	filtered_peps = {}
	for key in pips.keys():
		if not count_signals:
			pep = 1 - pips[key]
			if pep <= max_pep:
				filtered_peps[key] = pep
		else:
			pep = 1 - pips[key]['pip']
			if pep <= max_pep:
				filtered_peps[key] = pips[key]

	return filtered_peps

# def _postprocess_peps_count_signals(
# 	filtered_peps,
# 	max_pep=0.5,
# 	cutoff=0.95,
# ):
# 	"""
# 	Given the output of grid_peps with count_signals=True,
# 	postprocesses the result. This is a helper function for
# 	grid_peps_to_cand_groups.
# 	"""
# 	output = dict()
# 	for key in filtered_peps.keys():
# 		nsignals = [x for x in filtered_peps[key].keys() if x != 'pip']
# 		props = [filtered_peps[key][n] for n in nsignals]
# 		inds = np.argsort(props)




def grid_peps_to_cand_groups(
	filtered_peps, 
	time0=None,
	max_blip_size=1000,
	verbose=False,
	shape='square',
	max_pep=1,
	min_pep=0.001,
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
	shape : string
		One of ``square`` or ``circle``
	max_pep : float
		The maximum pep for candidate groups. Default: 1.
	min_pep : float
		Once we achieve a pep of this level for candidate groups,
		we do not search to find peps any lower. Default: 0.001.
	"""
	if time0 is None:
		time0 = time.time()

	# Step 0: determine whether filtered_peps counts the number of signals or not
	ngroups = len(filtered_peps)
	if ngroups == 0:
		return [], []
	keys = sorted(filtered_peps.keys())
	if isinstance(filtered_peps[keys[0]], dict):
		count_signals = True
		peps_arr = 1 - np.array([filtered_peps[k]['pip'] for k in keys])
	else:
		count_signals = False
		peps_arr = np.array([filtered_peps[k] for k in keys])


	# Step 1: compute adjacency matrix
	if ngroups > 50000:
		warnings.warn(f"Computing adjacency matrix may be too inefficient for {ngroups} candidate groups.")
	d = len(keys[0]) - 1 # dimensionality of problem

	if verbose:
		print(f"Constructing constraint matrix with ngroups={ngroups} at {elapsed(time0)}")
	constraints = np.ones((ngroups, ngroups)).astype(bool)
	centers = np.zeros((d, ngroups))
	centers[:] = np.nan
	radii = np.array([float(k[-1]) for k in keys])
	for j in range(d):
		# Extract centers of groups
		centers[j] = np.array([k[j] for k in keys])
		if centers[j].max() > 1 or centers[j].min() < 0:
			print(f"centers max = {centers[j].max()}, centers min = {centers[j].min()}")
			raise ValueError("centers must be between 0 and 1 but this is not true")
	# Find overlaps and add to constraints
	radii = radii.astype(np.float32)
	centers = centers.astype(np.float32)
	deltas = radii.reshape(-1, 1) + radii.reshape(1, -1)
	if shape == 'square':
		for j in range(d):
			constraints = constraints & (
				np.abs(centers[j].reshape(-1, 1) - centers[j].reshape(1, -1)) < deltas
			)
	elif shape == 'circle':
		dists = np.sqrt(np.power(
			centers.reshape(d, ngroups, 1) - centers.reshape(d, 1, ngroups), 2
		).sum(axis=0))
		constraints = dists < deltas
	else:
		raise ValueError(f"Unrecognized shape={shape}, must be one of 'square', 'circle'")


	# Step 2: Split problem into connected components
	if verbose:
		print(f"Isolating connected components at {elapsed(time0)}. Graph has {np.sum(constraints)} edges.")
	G = nx.Graph(constraints)
	components = list(nx.algorithms.components.connected_components(G))
	merged_components = [[]]
	for c in components:
		if len(merged_components[-1]) + len(c) > max_blip_size:
			merged_components.append([])
		merged_components[-1].extend(list(c))
	del G # save memory
		
	# Step 3: construct cand_groups for BLiP
	all_cand_groups = []
	for compnum, component in enumerate(merged_components):
		component_cand_groups = []
		component_groups = [[] for _ in component]
		subG = nx.Graph(constraints[component, :][:, component])
		# Construct groups for BLiP using as few locations as possible.
		# This is related to the problem of listing the maximal cliques of
		# subG. 
		if verbose:
			print(f"Finding list of cliques for component {compnum} / {len(merged_components)} with dim {len(component)} at {elapsed(time0)}")
		#cliques = nx.algorithms.clique.find_cliques(subG)
		cliques = edge_clique_cover(subG)#
		for cliquenum, clique in enumerate(cliques):
			for j in clique:
				component_groups[j].append(cliquenum)
		if verbose:
			print(f"Finished finding cliques, constructing cand_groups at {elapsed(time0)}")

		for ii, j in enumerate(component):
			group = set(component_groups[ii])
			data_dict = dict(radius=radii[j])
			for k in range(d):
				data_dict[f'dim{k}'] = centers[k, j]
			data_dict['center'] = centers[:, j].tolist()
			# Add many candidate groups with varying confidence regions
			# for the number of signals
			if count_signals:
				key = keys[j]
				nsignals = np.array([
					x for x in filtered_peps[key].keys() if x != 'pip'
				])
				props = np.array([filtered_peps[key][n] for n in nsignals])
				inds = np.argsort(-1*props)
				nsignals = nsignals[inds]
				cumprops = np.cumsum(props[inds])
				for ell in range(len(nsignals)):
					if cumprops[ell] >= 1 - max_pep:
						nsignal_ci = set(nsignals[0:(ell+1)].tolist())
						data_dict['nsignals'] = nsignal_ci
						component_cand_groups.append(
							CandidateGroup(
								group=group,
								pep=max(0, 1-cumprops[ell]),
								data=copy.deepcopy(data_dict)
							)
						)
					if cumprops[ell] >= 1 - min_pep:
						break
						
			else:
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