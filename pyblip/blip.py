import time
import warnings

import numpy as np
import pandas as pd

import cvxpy as cp ## faster for linear programs

from . import create_groups, create_groups_cts, weight_fns


# Find default solver
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
if 'GUROBI' in INSTALLED_SOLVERS:
	DEFAULT_SOLVER = 'GUROBI'
elif 'CBC' in INSTALLED_SOLVERS:
	DEFAULT_SOLVER = 'CBC'
else:
	warnings.warn("Using default (slightly slower) solver ECOS")
	DEFAULT_SOLVER = 'ECOS'

WEIGHT_FNS = {
	'inverse_size':weight_fns.inverse_size,
	'log_inverse_size':weight_fns.log_inverse_size,
}
ERROR_OPTIONS = ['fdr', 'local_fdr', 'fwer', 'pfer']
BINARY_TOL = 1e-3

def BLiP(
	inclusions=None,
	cand_groups=None,
	weight_fn='inverse_size',
	error='fdr',
	q=0.1,
	max_pep=0.5,
	how_binarize='sample',
	verbose=True,
	perturb=False,
	max_iters=50,
	search_method='binary',
	**kwargs
):
	"""
	Given samples from a posterior and/or a set of candidate
	groups, performs resolution-adaptive signal detection
	to maximize power while controlling the FWER, FDR, local FDR,
	or PFER.

	Note: when working with image data or a continuous set
	of locations, use ``BLiP_cts``.

	Paramters
	---------
	inclusions : np.array
		An ``(N, p)``-shaped array of posterior samples,
		where a nonzero value indicates the presence of a signal.
		Defaults to `None`.
	cand_groups : list
		A list of CandidateGroups for BLiP to optimize over. 
		Defaults to `None`. Note either ``cand_groups`` or 
		``inclusions`` must be provided.
	weight_fn : string or function
		A function which takes a CandidateGroup as an input and
		returns a (nonnegative) weight as an output. This defines
		the expected power function. Alternatively, can be
		'inverse_size' or 'log_inverse_size'.
	error : string
		The type of error rate to control. Must be one of 
		'fdr', 'local_fdr', 'fwer', or 'pfer'.
	q : float
		The level at which to control the error rate.
	max_pep : float
		BLiP automatically filters out candidate groups with
		a posterior error probability (PEP) above ``max_pep``.
		Default: 0.5.
	how_binarize : string
		How to round the solutions to the linear program (LP) to
		integers. Must equal either 'sample' (random sampling) 
		or 'intlp' (integer LP).
	verbose : bool
		If True, gives occasional progress reports.
	max_iters : int
		Maximum number of line-search iterations for FDR problem.
		Defaults to 50.
	search_method : string
		For FWER control, how to find the optimal parameter
		for the LP. Either "none" or "binary_search." Defaults
		to "binary". 
	perturb : bool
		If True, will perturb the weight function 
	"""
	# Parse arguments
	time0 = time.time()
	error = str(error).lower()
	if error not in ERROR_OPTIONS:
		raise ValueError(f"error type {error} must be one of {ERROR_OPTIONS}")
	if cand_groups is None and inclusions is None:
		raise ValueError("At least one of cand_groups and inclusions must be provided.")
	if error != 'fdr':
		max_pep = min(max_pep, q) # this is optimal for all but error == 'fdr'

	# Create cand groups if necessary
	if cand_groups is None:
		cand_groups = create_groups.sequential_groups(
			inclusions=inclusions,
			q=q,
			max_pep=max_pep,
			prenarrow=True,
		)
		cand_groups.extend(
			create_groups.hierarchical_groups(
				inclusions=inclusions,
				max_pep=max_pep,
				filter_sequential=True
			)
		)

	# Else, prefilter
	cand_groups = create_groups._prefilter(cand_groups, max_pep=max_pep)
	# Edge case where nothing is below max_pep
	if len(cand_groups) == 0:
		return []
	# Construct a re-indexing which does not include redundant features
	cand_groups, nrel = create_groups._elim_redundant_features(cand_groups)

	# Weights for each candidate group
	ngroups = len(cand_groups)
	if isinstance(weight_fn, str):
		weight_fn = weight_fn.lower()
		if weight_fn not in WEIGHT_FNS:
			raise ValueError(f"Unrecognized weight_fn={weight_fn}, must be one of {list(WEIGHT_FNS.keys())}")
		weight_fn = WEIGHT_FNS[weight_fn]
	weights = np.array([weight_fn(x) for x in cand_groups])
	# perturb to avoid degeneracy in some cases
	if perturb:
		weights = np.array([
			w*(1 + 0.001*np.random.uniform()) for w in weights]
		)

	# Extract peps
	peps = np.array([x.pep for x in cand_groups])

	# Constraints to ensure selected groups are disjoint
	if verbose:
		print(f"BLiP problem has {ngroups} groups in contention, with {nrel} active features/locations")
	group_constraints = np.zeros((ngroups, nrel), dtype=bool)
	for gj, cand_group in enumerate(cand_groups):
		for feature in cand_group.data['blip-group']:
			group_constraints[gj, feature] = 1

	# Concatenate constraints, variables
	A = group_constraints
	x = cp.Variable(ngroups)
	x.value = np.zeros(ngroups)
	selections = x.value
	b = np.ones(nrel)
	v_current_cp = cp.Parameter()

	# We perform a binary search for FDR to find the optimal v. 
	# We can also do this for FWER if ``inclusions`` is available.
	if error == 'fdr':
		binary_search = True
	elif inclusions is not None and error == 'fwer' and search_method == 'binary':
		binary_search = True
	else:
		binary_search = False
	if binary_search:
		# upper bnd in binary search (e.g., min val not controlling FWER)
		v_upper = np.around(q * nrel)
		# lower bnd in binary search (e.g., max val not controlling FDR)
		v_lower = q	if error == 'fdr' else 0
		v_current_cp.value = (v_upper + v_lower)/2
	else:
		v_current_cp.value = q

	# Construct problem
	constraints = [
		x >= 0,
		x <= 1,
		A.T @ x <= b,
		x @ peps <= v_current_cp,
	]
	# Last constraint is redundant for local_fdr
	if error == 'local_fdr':
		constraints = constraints[0:-1]
	# Extra constraint for fdr
	if error == 'fdr':
		constraints += [cp.sum(x) >= v_current_cp / q]

	# Create problem
	objective = cp.Maximize(weights @ x)
	prev_selections = x.value.copy()
	problem = cp.Problem(objective=objective, constraints=constraints)

	# Solve problem once if we do not need to search over v
	if not binary_search:
		problem.solve(solver=DEFAULT_SOLVER)
		selections = x.value

	# Solve FWER/FDR through binary search
	elif binary_search:
		power_lower = 0 # for FDR control only
		problem.solve(solver=DEFAULT_SOLVER, warm_start=True)
		inclusions = inclusions != 0
		for niter in range(max_iters):
			
			# Change parametrized constraint
			v_current = (v_upper + v_lower)/2
			v_current_cp.value = v_current 

			# Solve
			problem.solve(solver=DEFAULT_SOLVER, warm_start=True)
			selections = x.value

			# Binary search
			if error == 'fdr':
				if problem.status == 'infeasible':
					power = 0
				else:
					power = np.sum(selections * (1-peps) * weights)
				if power > power_lower:
					v_lower = v_current
				else:
					v_upper = v_current
			# FWER search
			elif error == 'fwer':
				# Round selections---could do something smarter, TODO
				selections = np.around(selections)
				# Compute exact FWER
				false_disc = np.zeros(inclusions.shape[0]).astype(bool)
				for gj in np.where(selections > BINARY_TOL)[0]:
					group = list(cand_groups[gj].group)
					false_disc = false_disc | np.all(inclusions[:, group] == 0, axis=1)
				fwer = false_disc.mean()
				if fwer > q:
					v_upper = v_current
				else:
				 	v_lower = v_current
			# Possibly break
			if v_upper - v_lower < 1e-3:
				break

		# Solve with v_lower for final solution
		v_current_cp.value = v_lower
		problem.solve(solver=DEFAULT_SOLVER, warm_start=True)
		if problem.status == 'infeasible':
			selections = np.zeros(ngroups)
		else:
			selections = x.value

		# TODO could do something smarter for FWER
		if error == 'fwer':
			selections = np.around(selections)

	for cand_group, sprob, weight in zip(cand_groups, selections, weights):
		cand_group.data['sprob'] = sprob
		cand_group.data['weight'] = weight

	return binarize_selections(
		cand_groups=cand_groups,
		p=nrel,
		v_opt=v_current_cp.value,
		error=error,
		how_binarize=how_binarize
	)


def binarize_selections(
	cand_groups,
	p,
	v_opt,
	error,
	how_binarize='sample',
	tol=1e-3,
):
	"""
	Parameters
	----------
	cand_groups : list
		list of candidate groups. 
	p : int
		dimensionality
	v_opt : int
		Level for PFER control.
	how_binarize : str 
		how to binarize the selections.

	Returns
	-------
	detections : list
		List of candidate groups which have been detected.
	"""
	output = []

	# Prune options with zero selection prob
	# and add options which have selection prob of 1. 
	nontriv_cand_groups = []
	for cand_group in cand_groups:
		if cand_group.data['sprob'] < tol:
			continue
		elif cand_group.data['sprob'] > 1 - tol:
			output.append(cand_group)
		else:
			nontriv_cand_groups.append(cand_group)

	# The easy cases...
	ngroups = len(nontriv_cand_groups)
	if ngroups == 0:
		return output
	if ngroups == 1:
		if how_binarize == 'sample' and np.random.uniform() < nontriv_cand_groups[0].data['sprob']:
			output.append(nontriv_cand_groups[0])
		return output

	# Sampling method
	if how_binarize == 'sample':
		# Create dataframe
		ndf = pd.DataFrame(columns = ['prob'] + list(range(p)))
		for i, cand_group in enumerate(nontriv_cand_groups):
			ndf.loc[i] = np.zeros((p+1),)
			ndf.loc[i, 'prob'] = cand_group.data['sprob']
			for j in cand_group.data['blip-group']:
				ndf.loc[i, j] = 1

		 # Sort features in terms of marg. prob of selection
		ind_probs = []
		for x in range(p):
			ind_probs.append(ndf.loc[ndf[x] == 1, 'prob'].sum())
		inds = np.argsort(-1*np.array(ind_probs))
		
		# Initialize
		eliminated_groups = np.zeros((ngroups,))
		selected_groups = []
		for x in inds:

			if eliminated_groups.sum() == ngroups:
				break
			
			# Subset of available options to sample from
			subset = ndf.loc[(ndf[x] == 1) & (eliminated_groups == 0)]
			if subset.shape[0] == 0:
				continue
			
			# Scale up probabilities
			prev_eliminated = ndf.loc[(ndf[x] == 1) & (eliminated_groups == 1)]
			scale = 1 - prev_eliminated['prob'].sum()
			new_probs = subset['prob'].values / scale
			
			# Select nothing with some probability
			if np.random.uniform() <= 1 - new_probs.sum():
				eliminated_groups[subset.index] = 1
				continue
			
			# Else, select something according to new_probs
			selected_group = np.where(
				np.random.multinomial(1, new_probs / new_probs.sum()) != 0
			)[0][0]
			selected_group = subset.index[selected_group]
			
			# Save this group 
			selected_groups.append(selected_group)
			
			# Eliminate all mutually exclusive groups
			row = ndf.loc[selected_group]
			features = [x for x in row.index if row[x] == 1]
			for feature in features:
				eliminated_groups[ndf.loc[ndf[feature] == 1].index] = 1

		if ndf.loc[selected_groups, inds].sum(axis='index').max() > 1:
			raise RuntimeError("A feature was selected in multiple groups...")
		output.extend([nontriv_cand_groups[sg] for sg in selected_groups])

	elif how_binarize == 'intlp':
		
		# Constraints to ensure selected groups are disjoint
		A = np.zeros((ngroups, p))
		for gj, cand_group in enumerate(nontriv_cand_groups):
			for feature in cand_group.data['blip-group']:
				A[gj, feature] = 1
		A = A[:, A.sum(axis=0) > 0] # eliminate irrelevant features

		# Concatenate constraints, variables
		x = cp.Variable(ngroups, integer=True)
		x.value = np.zeros(ngroups)
		selections = x.value
		b = np.ones(A.shape[1])
		v_prime = v_opt - sum([n.pep for n in output])
		peps = [cand_group.pep for cand_group in nontriv_cand_groups]
		weights = [cand_group.data['weight'] for cand_group in nontriv_cand_groups]

		# Construct problem
		constraints = [
			A.T @ x <= b,
			x >= 0,
			x <= 1,
			x @ peps <= v_prime,
		]
		if error == 'local_fdr':
			constraints = constraints[0:-1]

		objective = cp.Maximize(weights @ x)
		problem = cp.Problem(objective=objective, constraints=constraints)
		problem.solve()

		for binprob, x in zip(x.value, nontriv_cand_groups):
			if binprob > 1 - tol:
				output.append(x)

	return output