""" Functions for creating candidate groups from posterior samples"""
import numpy as np
from tqdm import tqdm
# Tree methods from scipy
import scipy.cluster.hierarchy as hierarchy
import scipy.spatial.distance as ssd
### todo : a generic function which performs prenarrowing

class CandidateGroup():
	"""
	A single candidate group as an input to BLiP.
	Rejecting the 

	Parameters
	----------
	group : list or set
		Set of locations in the group.
	pep : float
		Posterior error probability (1 - PIP) for the group.
	data : dict
		Miscallaneous other attributes to associate with the group.

	Notes: The class attributes are the same as the parameters
	(group, pep, and data).
	"""
	def __init__(self, group, pep, data=dict()):
		self.group = group
		self.pep = pep
		self.data = data

	def __str__(self):
		return f"CandidateGroup(group={self.group}, pep={self.pep}, data={self.data})"

	def __repr__(self):
		return self.__str__()

	def to_dict(self):
		"""
		Converts self into a dictionary for saving as JSON.
		"""
		out = dict()
		out['group'] = list(group)
		out['pep'] = pep
		for key in self.data:
			if isinstance(set, data[key]):
				out[key] = list(self.data[key])
			else:
				out[key] = self.data[key]
		return out

def sequential_groups(
		inclusions, 
		q=0,
		max_pep=1,
		max_size=25,
		prenarrow=True
	):
	"""
	Calculate peps for all sequential groups 
	of size less than max_size.

	Parameters
	----------
	inclusions : np.ndarray
		An ``(N, p)``-shaped array of posterior samples,
		where a nonzero value indicates the presence of a signal.
	q : float
		The nominal level at which to control the error rate.
	max_pep : float
		The maximum posterior error probability allowed in
		a candidate group. Default is 1.
	max_size : float
		Maximum size of a group. Default is 25.
	prenarrow : bool
		If true, "prenarrows" the candidate groups
		as described in the paper. Defaults to True.
	"""
	inclusions = inclusions != 0 # make boolean
	N = inclusions.shape[0]
	p = inclusions.shape[1]
	max_size = min(max_size, p)
	cum_incs = np.zeros((N, p+1))
	cum_incs[:, 1:(p+1)] = np.cumsum(inclusions, axis=1)

	# Compute successive groups of size m
	all_PEPs = {}
	for m in list(range(max_size)):
		cum_diffs = cum_incs[:, (m+1):(p+1)] - cum_incs[:, :int(p-m)]
		all_PEPs[m] = np.mean(cum_diffs == 0, axis=0)
	#print(all_PEPs[1])
	#print(np.mean(inclusions, axis=0))

	# the index is the first (smallest) variable in the group which has size m
	active_inds = {}
	for m in range(max_size):
		active_inds[m] = np.where(all_PEPs[m] < max_pep)[0]

	# Lucas's trick
	# This iteratively updates the list elim_inds so that
	# when consider the set of groups of size m+1, 
	# elim_inds are all the indices that are redundant
	if prenarrow:
		elim_inds = set(np.where(all_PEPs[0] < q)[0].tolist())
		for m in range(1, max_size):
			# If index j is eliminated for level m-1, indexes j and j-1
			# are eliminated for index m
			elim_inds = elim_inds.union(set([x-1 for x in elim_inds]))
			# Update active_inds[m]
			update = set(active_inds[m].tolist()) - elim_inds
			active_inds[m] = np.array(list(update))
			# At this level, account for groups with PEP < q
			elim_inds = elim_inds.union(set(
				np.where(all_PEPs[m] < q)[0].tolist()
			))

	# Step 3: create candidate groups
	cand_groups = []
	for m in range(max_size):
		for ind in active_inds[m]:
			group = set(list(range(ind, ind+m+1)))
			pep = all_PEPs[m][ind]
			cand_groups.append(CandidateGroup(
				group=group, pep=pep
			))

	return cand_groups

def _extract_groups(root):
	"""
	Extracts the set of all groups from a scipy hierarchical
	clustering tree.
	"""
	output = [root.pre_order()]
	if root.left is not None:
		output.extend(_extract_groups(root.left))
	if root.right is not None:
		output.extend(_extract_groups(root.right))
	return output

def _dedup_list_of_lists(x):
	return list(set(tuple(i) for i in x))

def _dist_matrix_to_groups(
	dist_matrix
):
	"""
	Creates groups based on corr_matrix using
	single, average, and hierarchical clustering.
	"""
	# prevent numerical errors
	dist_matrix -= np.diag(np.diag(dist_matrix))
	dist_matrix = (dist_matrix.T + dist_matrix) / 2
	# turn into scipy format
	condensed_dist_matrix = ssd.squareform(dist_matrix)
	# Run hierarchical clustering
	all_groups = []
	for cluster_func in [hierarchy.single, hierarchy.average, hierarchy.complete]:
		link = cluster_func(condensed_dist_matrix)
		groups = _extract_groups(hierarchy.to_tree(link))
		all_groups.extend(groups)
		# deduplicate
		all_groups = _dedup_list_of_lists(all_groups)
	return all_groups

def hierarchical_groups(
	inclusions,
	dist_matrix=None,
	max_pep=1,
	max_size=25,
	filter_sequential=False,
):
	"""
	Parameters
	----------
	inclusions : np.ndarray
		An ``(N, p)``-shaped array of posterior samples,
		where a nonzero value indicates the presence of a signal.
	dist_matrix : np.ndarray
		square numpy array corresponding to distances between locations.
		This is used to hierarchically cluster the groups.
	max_pep : float
		The maximum posterior error probability allowed in
		a candidate group. Default is 1.
	max_size : float
		Maximum size of a group. Default is 25.
	filter_sequential : bool
		If True, does not calculate PEPs for sequential (contiguous)
		groups of variables to avoid duplicates.
	"""
	# Initial values
	p = inclusions.shape[1]
	inclusions = inclusions != 0
	# Trivial case where there is only one feature
	if p == 1:
		pep = 1 - inclusions.mean()
		return [CandidateGroup(group=set([0]), pep=pep)]

	# Estimate cov matrix from inclusions if 
	# concatenations ensure no inclusions are all zero or one
	if dist_matrix is None:
		precorr = np.concatenate(
			[
				inclusions,
				np.zeros((1, p)),
				np.ones((1, p))
			],
			axis=0
		)
		corr_matrix = np.corrcoef(precorr.T)
		dist_matrix = corr_matrix + 1

	# Create groups
	groups = _dist_matrix_to_groups(dist_matrix)
	# Create candidate group objects
	cand_groups = []
	for group in groups:
		gsize = len(group)
		if gsize > max_size:
			continue
		# Possibly filter out contiguous groups
		if filter_sequential:
			mingroup = min(group)
			maxgroup = max(group)
			if maxgroup - mingroup == gsize - 1:
				continue

		pep = 1 - np.any(inclusions[:, group], axis=1).mean()
		if pep < max_pep:
			cand_groups.append(
				CandidateGroup(group=set(group), pep=pep)
			)

	return cand_groups


def _elim_redundant_features(cand_groups):
	"""
	After prefiltering groups, some features/locations may not
	appear in any candidate groups. When this happens, this
	function reindexes the locations to improve the efficiency
	of the BLiP solver.

	Parameters
	----------
	cand_groups : list
		A list of CandidateGroup objects.

	Returns
	-------
	cand_groups : list
		A list of CandidateGroup objects, but with a "blip-group" 
		attribute that reindexes the features to avoid redundancy.
	nrel : int
		The number relevant features.
	"""
	# Step 1: find relevant features
	active_features = set()
	for cand_group in cand_groups:
		group = set(cand_group.group)
		active_features = active_features.union(group)

	# Step 2: change feature inds to save computation
	nrel = len(active_features)
	new2orig = np.zeros((nrel,))
	orig2new = {}
	for i, j in enumerate(list(active_features)):
		new2orig[i] = j
		orig2new[j] = i
	for cand_group in cand_groups:
		group = [orig2new[x] for x in cand_group.group]
		cand_group.data['blip-group'] = set(group)
	# return
	return cand_groups, nrel




