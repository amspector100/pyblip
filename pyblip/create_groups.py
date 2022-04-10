""" Functions for creating candidate groups from posterior samples"""
import copy
import numpy as np
# Tree methods from scipy
import scipy.cluster.hierarchy as hierarchy
import scipy.spatial.distance as ssd
import networkx as nx
from .ecc import edge_clique_cover

MIN_PEP = 1e-15 # for numerical stability

class CandidateGroup():
	"""
	A single candidate group as an input to BLiP.
	Discovering a group asserts there is at least 
	one signal in the group.

	Parameters
	----------
	group : list or set
		Set of locations in the group.
	pep : float
		Posterior error probability (1 - PIP) for the group.
	data : dict
		Miscallaneous other attributes to associate with the group.
	"""
	def __init__(self, group, pep, data=None):
		self.group = set(group)
		self.pep = pep
		if data is None:
			data = dict()
		self.data = data

	def __str__(self):
		return f"CandidateGroup(group={self.group}, pep={self.pep}, data={self.data})"

	def __repr__(self):
		return self.__str__()

	def to_dict(self):
		"""
		Converts ``CandidateGroup`` into a dictionary.
		"""
		out = dict()
		out['group'] = list(self.group)
		out['pep'] = self.pep
		for key in self.data:
			if isinstance(self.data[key], set):
				out[key] = list(self.data[key])
			else:
				out[key] = self.data[key]
		return out


def all_cand_groups(
	samples,
	X=None,
	q=0,
	max_pep=0.25,
	max_size=25,
	prenarrow=True,
	prefilter_thresholds=[0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2]
):
	"""
	Creates many candidate groups by prefiltering locations at
	various thresholds and then creating sequential and hierarchical
	candidate groups.

	Parameters
	----------
	samples : np.ndarray
		An ``(N, p)``-shaped array of posterior samples,
		where a nonzero value indicates the presence of a signal.
	X : np.array
		The n x p deisgn matrix. Defaults to ``None.``
		If provided, adds hierarchical groups based 
		on a correlation cluster of X.
	q : float
		The nominal level at which to control the error rate.
	max_pep : float
		The maximum posterior error probability allowed in
		a candidate group. Default is 1.
	max_size : float
		Maximum size of a group. Default is 25.
	prenarrow : bool
		If true, "prenarrows" the candidate groups
		as described in the paper. Defaults to False.
	prefilter_thresholds : list
		List of thresholds at which to prefilter the locations.

	Returns
	-------
	cand_groups : list
		A list of ``CandidateGroup`` objects.
	"""
	samples = samples != 0
	marg_pips = np.mean(samples, axis=0)

	# pre-filter features at various levels
	all_groups = set()
	all_cgs = []
	for thresh in prefilter_thresholds:
		rel_features = np.where(marg_pips > thresh)[0]
		if len(rel_features) == 0:
			continue

		# Sequential groups
		cgs = sequential_groups(
			samples[:, rel_features],
			q=q,
			max_pep=max_pep,
			max_size=max_size,
			prenarrow=prenarrow
		)
		# Distance matrices
		dms = [_samples_dist_matrix(samples[:, rel_features])]
		if X is not None:
			dms.append(np.abs(1 - np.corrcoef(X[:, rel_features].T)))
		cgs.extend(hierarchical_groups(
			samples[:, rel_features],
			dist_matrix=dms,
			max_pep=max_pep,
			max_size=max_size,
			filter_sequential=True,
		))
		# Correct group indices and add to all cgs
		groups = []
		for cg in cgs:
			group = tuple(sorted(rel_features[list(cg.group)].tolist()))
			if group not in all_groups:
				cg.group = set(group)
				all_cgs.append(cg)
				groups.append(group)
		all_groups = all_groups.union(groups)

	return all_cgs


def susie_groups(
	alphas,
	X,
	q,
	max_pep=0.25,
	max_size=25,
	prenarrow=False,
	purity_threshold=0.0,
):
	"""
	Creates candidate groups based on a SuSiE fit.

	Parameters
	----------
	alphas : np.array
		An ``(L, p)``-shaped matrix of alphas from a SuSiE object,
		``L`` is the number of SuSiE iterations and ``p``
		is the number of covariates.
	X : np.array
		The n x p deisgn matrix. Defaults to ``None.``
		If provided, adds hierarchical groups based 
		on a correlation cluster of X.
	q : float
		The level at which to control the error rate
	max_pep : float
		The maximum posterior error probability allowed in
		a candidate group. Default is 1.
	max_size : float
		Maximum size of a group. Default is 25.
	prenarrow : bool
		If true, "prenarrows" the candidate groups
		as described in the paper. Defaults to False.
	purity_threshold : float
		When computing PIPs, ignores SuSiE iterations which
		do not pass this purity threshold, as in the original
		SusiE paper.

	Returns
	-------
	cand_groups : list
		A list of ``CandidateGroup`` objects.
	"""
	L, p = alphas.shape
	np.random.seed(1)
	# Preprocessing
	if 

	# Add groups discovered by susie
	groups_to_add = []
	for j in range(L):
		if np.sum(alphas[j]) >= 1 - q:
			inds = np.argsort(-1*alphas[j])
			k = np.min(np.where(np.cumsum(alphas[j,inds]) >= 1 - q))
			group = inds[0:(k+1)].tolist()
			groups_to_add.append(group)

	# sequential groups
	cand_groups = sequential_groups(
		susie_alphas=alphas, 
		q=q,
		max_pep=max_pep,
		max_size=max_size,
		prenarrow=prenarrow,
	)
	# Add hierarchical groups
	if X is not None:
		dist_matrix = np.abs(1 - np.corrcoef(X.T))
		groups_to_add.extend(_dist_matrices_to_groups(dist_matrix))

	# Add these to cand_groups
	groups_to_add = _dedup_list_of_lists(groups_to_add)
	for g in groups_to_add:
		if len(g) > max_size:
			continue
		if np.max(g) - np.min(g) == len(g) - 1:
			continue
		iter_peps = 1 - alphas[:,g].sum(axis=1)
		iter_peps[iter_peps < MIN_PEP] = MIN_PEP # for numerical stability
		pep = np.exp(np.log(iter_peps).sum())
		if pep < max_pep:
			cand_groups.append(CandidateGroup(
				group=set(g), pep=pep
			))

	return cand_groups

def sequential_groups(
	samples=None, 
	susie_alphas=None,
	q=0,
	max_pep=0.25,
	max_size=25,
	prenarrow=False
):
	"""
	Calculates all sequential candidate groups below max_size.

	Parameters
	----------
	samples : np.ndarray
		An ``(N, p)``-shaped array of posterior samples,
		where a nonzero value indicates the presence of a signal.
	susie_alphas : np.ndarray
		As an alternative to posterior samples, users may specify an
		L x p matrix of alphas from a SuSiE object. However, calling
		``susie_groups`` is recommended instead.
	q : float
		The nominal level at which to control the error rate.
	max_pep : float
		The maximum posterior error probability allowed in
		a candidate group. Default is 1.
	max_size : float
		Maximum size of a group. Default is 25.
	prenarrow : bool
		If true, "prenarrows" the candidate groups
		as described in the paper. Defaults to False.

	Returns
	-------
	cand_groups : list
		A list of ``CandidateGroup`` objects.
	"""

	if samples is not None:
		samples = samples != 0 # make boolean
		N, p = samples.shape
		max_size = min(max_size, p)
		cum_incs = np.zeros((N, p+1))
		cum_incs[:, 1:(p+1)] = np.cumsum(samples, axis=1)

		# Compute successive groups of size m
		all_PEPs = {}
		for m in range(max_size):
			cum_diffs = cum_incs[:, (m+1):(p+1)] - cum_incs[:, :int(p-m)]
			all_PEPs[m] = np.mean(cum_diffs == 0, axis=0)
	elif susie_alphas is not None:
		L, p = susie_alphas.shape
		max_size = min(max_size, p)
		cumalphas = np.zeros((L, p + 1))
		cumalphas[:, 1:(p+1)] = np.cumsum(susie_alphas, axis=1)
		# Compute successive groups of size m
		all_PEPs = {}
		for m in range(max_size):
			cumdiffs = 1 - (cumalphas[:, (m+1):(p+1)] - cumalphas[:, :int(p-m)])
			cumdiffs[cumdiffs < MIN_PEP] = MIN_PEP
			all_PEPs[m] = np.exp(np.log(cumdiffs).sum(axis=0))
	else:
		raise ValueError("Either samples or susie_alphas must be specified.")

	# the index is the first (smallest) variable in the group which has size m
	active_inds = {}
	for m in range(max_size):
		active_inds[m] = np.where(all_PEPs[m] < max_pep)[0]

	# prenarrowing
	# This iteratively updates the list elim_inds so that
	# when consider the set of groups of size m+1, 
	# elim_inds are all the indices that are redundant
	if prenarrow:
		elim_inds = set(np.where(all_PEPs[0] < q / 2)[0].tolist())
		for m in range(1, max_size):
			# If index j is eliminated for level m-1, indexes j and j-1
			# are eliminated for level m
			elim_inds = elim_inds.union(set([x-1 for x in elim_inds]))
			# Update active_inds[m]
			update = set(active_inds[m].tolist()) - elim_inds
			active_inds[m] = np.array(list(update))
			# At this level, account for groups with PEP < q
			elim_inds = elim_inds.union(set(
				np.where(all_PEPs[m] < q / 2)[0].tolist()
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

def hierarchical_groups(
	samples,
	dist_matrix=None,
	max_pep=0.25,
	max_size=25,
	filter_sequential=False,
	**kwargs
):
	"""
	Creates candidate groups by hierarchical clustering.

	Parameters
	----------
	samples : np.ndarray
		An ``(N, p)``-shaped array of posterior samples,
		where a nonzero value indicates the presence of a signal.
	dist_matrix : np.ndarray or list
		A square numpy array corresponding to distances between locations.
		Can also be a list of different distance matrices. The default
		is to use a correlation matrix based on ``samples.``
	max_pep : float
		The maximum posterior error probability allowed in
		a candidate group. Default is 1.
	max_size : float
		Maximum size of a group. Default is 25.
	filter_sequential : bool
		If True, does not calculate PEPs for sequential (contiguous)
		groups of variables to avoid duplicates.

	Returns
	-------
	cand_groups : list
		A list of ``CandidateGroup`` objects.
	"""
	# Initial values
	p = samples.shape[1]
	samples = samples != 0
	# Trivial case where there is only one feature
	if p == 1:
		pep = 1 - samples.mean()
		return [CandidateGroup(group=set([0]), pep=pep)]

	# Estimate cov matrix from samples if 
	# concatenations ensure no samples are all zero or one
	if dist_matrix is None:
		dist_matrix = _samples_dist_matrix(samples)


	# Create groups
	groups = _dist_matrices_to_groups(dist_matrix, **kwargs)
	# Create candidate group objects
	cand_groups = []
	for group in groups:
		gsize = len(group)
		if gsize > max_size:
			continue
		# Possibly filter out contiguous groups
		if filter_sequential:
			if np.max(group) - np.min(group) == gsize - 1:
				continue

		pep = 1 - np.any(samples[:, group], axis=1).mean()
		if pep < max_pep:
			cand_groups.append(
				CandidateGroup(group=set(group), pep=pep)
			)

	return cand_groups

def _extract_groups(root, p):
	"""
	Extracts the set of all groups from a scipy hierarchical
	clustering tree.
	"""
	output = []
	queue = []
	queue.append(root)
	while len(queue) > 0:
		node = queue.pop(0)
		if node.left is not None:
			queue.append(node.left)
		if node.right is not None:
			queue.append(node.right)
		output.append(node.pre_order())
	return output

def _dedup_list_of_lists(x):
	return list(set(tuple(sorted(i)) for i in x))

def _dist_matrices_to_groups(
	dist_matrices,
	cluster_funcs=None,
):
	"""
	Creates groups based on corr_matrix using
	single, average, and hierarchical clustering.
	"""
	if isinstance(dist_matrices, np.ndarray):
		dist_matrices = [dist_matrices]
	p = dist_matrices[0].shape[0]
	all_groups = []
	if cluster_funcs is None:
		cluster_funcs = [hierarchy.single, hierarchy.average, hierarchy.complete]
	for dist_matrix in dist_matrices:
		# prevent numerical errors
		dist_matrix -= np.diag(np.diag(dist_matrix))
		#lower_inds = np.tril_indices(p, -1)
		#dist_matrix[lower_inds] = dist_matrix.T[lower_inds]
		dist_matrix = (dist_matrix.T + dist_matrix) / 2
		# turn into scipy format
		condensed_dist_matrix = ssd.squareform(dist_matrix)
		# Run hierarchical clustering
		all_groups = []
		for cluster_func in cluster_funcs:
			link = cluster_func(condensed_dist_matrix)
			groups = _extract_groups(hierarchy.to_tree(link), p=p)
			all_groups.extend(groups)
			# deduplicate
			all_groups = _dedup_list_of_lists(all_groups)
	return all_groups

def _samples_dist_matrix(samples):
	p = samples.shape[1]
	precorr = np.concatenate(
		[
			samples,
			np.zeros((1, p)),
			np.ones((1, p))
		],
		axis=0
	)
	corr_matrix = np.corrcoef(precorr.T)
	dist_matrix = corr_matrix + 1
	return dist_matrix

def _prefilter(cand_groups, max_pep):
	"""
	Returns the subset of cand_groups with a pep below max_pep.
	"""
	return [
		x for x in cand_groups if x.pep < max_pep
	]



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
	active_features = set([j for x in cand_groups for j in x.group])

	# Step 2: change feature inds to save computation
	nrel = len(active_features)
	orig2new = {}
	for i, j in enumerate(list(active_features)):
		orig2new[j] = i
	for cand_group in cand_groups:
		blip_group = [orig2new[x] for x in cand_group.group]
		cand_group.data['blip-group'] = set(blip_group)

	# return
	return cand_groups, nrel


def _ecc_reduction(cand_groups):
	"""
	Given a set of ``CandidateGroup``s with finite group
	attributes, performs an ECC reduction to reduce
	the number of locations.

	Parameters
	----------
	cand_groups : list
		A list of CandidateGroup objects.

	Returns
	-------
	cand_groups : list
		The same list of CandidateGroup objects but with
		an updated 'blip-group' attribute.
	nrel : int
		Number of locations in reduced problem.


	Notes
	-----
	This function is intended as a preprocessing step
	for certain discrete problems. For continuous problems,
	use the functions in the ``pyblip.create_groups_cts``
	module.

	It is not yet efficient and is largely used to reduce very
	small problems to avoid solver error. 
	"""
	hasblipgroup = ['blip-group' in cg.data for cg in cand_groups]
	if not np.all(np.array(hasblipgroup)):
		for cg in cand_groups:
			cg.data['blip-group'] = cg.group

	# O(|G|^2) algorithm to find intersections
	m = len(cand_groups)
	G = np.eye(m).astype(bool)
	for i, cg1 in enumerate(cand_groups):
		g1 = cg1.data['blip-group']
		for j in range(i):
			g2 = cand_groups[j].data['blip-group']
			if len(g1.intersection(g2)) > 0:
				G[i, j] = True
				G[j, i] = True

	# edge clique cover
	G = nx.Graph(G)
	cliques = edge_clique_cover(G)
	# reset blip-group
	for cg in cand_groups:
		cg.data['blip-group'] = []
	for cliquenum, clique in enumerate(cliques):
		for i in clique:
			cand_groups[i].data['blip-group'].append(cliquenum)
	for cg in cand_groups:
		cg.data['blip-group'] = set(cg.data['blip-group'])

	return cand_groups, len(cliques)
