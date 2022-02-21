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
	DEFAULT_SOLVER = 'ECOS'
# Solver for binarization (very small-scale mixed-integer LP)
# GLPK is a good solver in general, but it has known 
# bugs in very small problems like this one. See
# https://github.com/cvxpy/cvxpy/issues/1112
if 'CBC' in INSTALLED_SOLVERS:
	BIN_SOLVER = 'CBC'
else:
	BIN_SOLVER = None # default

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
	deterministic=True,
	verbose=True,
	perturb=True,
	max_iters=100,
	search_method='binary',
	return_problem_status=False,
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
		'inverse_size', 'log_inverse_size', or 'prespecified', 
		in which case each cand_group must have a weight key
		in its data dictionary. 
		Defaults to 'inverse_size'.
	error : string
		The type of error rate to control. Must be one of 
		'fdr', 'local_fdr', 'fwer', 'pfer'.
	q : float
		The level at which to control the error rate.
	max_pep : float
		BLiP automatically filters out candidate groups with
		a posterior error probability (PEP) above ``max_pep``.
		Default: 0.5.
	deterministic : bool
		If True, gives a deterministic solution. Otherwise BLiP
		may employ a randomized solution for slightly more power.
	verbose : bool
		If True, gives occasional progress reports.
	max_iters : int
		Maximum number of line-search iterations for FDR problem.
		Defaults to 100.
	search_method : string
		For FWER control, how to find the optimal parameter
		for the LP. Either "none" or "binary." Defaults to "binary". 
	perturb : bool
		If True, will perturb the weight function 
	return_problem_status : False
		If True, will return extra information about the problem.

	Returns
	-------
	detections : list
		A list of CandidateGroups consisting of the detected signals
		and the regions containing them.
	problem_status : dict
		A dict containing information about the BLiP optimization problem.
		Only returned if ``return_problem_status==True``.	
	"""
	# Parse arguments
	time0 = time.time()
	error = str(error).lower()
	if error not in ERROR_OPTIONS:
		raise ValueError(f"error type {error} must be one of {ERROR_OPTIONS}")
	if cand_groups is None and inclusions is None:
		raise ValueError("At least one of cand_groups and inclusions must be provided.")
	if error in ['fwer', 'pfer', 'local_fdr']:
		max_pep = min(max_pep, q) # this is optimal for all but error == 'fdr'
	solver = kwargs.get("solver", DEFAULT_SOLVER)
	if solver == 'ECOS' and verbose:
		warnings.warn("Using ECOS solver, which will be slightly slower. Consider installing CBC.")

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
	if weight_fn == 'prespecified':
		weights = np.array([x.data['weight'] for x in cand_groups])
	else:
		if 'weight' in cand_groups[0].data:
			warnings.warn(
				"Cand groups have a 'weight' attribute, do you mean to set weight_fn='prespecified'?"
			)
		if isinstance(weight_fn, str):
			weight_fn = weight_fn.lower()
			if weight_fn not in WEIGHT_FNS:
				raise ValueError(f"Unrecognized weight_fn={weight_fn}, must be one of {list(WEIGHT_FNS.keys())}")
			weight_fn = WEIGHT_FNS[weight_fn]
		# Get weights
		weights = np.array([weight_fn(x) for x in cand_groups])

	# perturb to avoid degeneracy in some cases
	if perturb:
		weights = np.array([
			w*(1 + 0.0001*np.random.uniform()) for w in weights]
		)

	# Extract peps
	peps = np.array([x.pep for x in cand_groups])

	# Constraints to ensure selected groups are disjoint
	if verbose:
		print(f"BLiP problem has {ngroups} groups in contention, with {nrel} active features/locations")
	A = np.zeros((ngroups, nrel), dtype=bool)
	for gj, cand_group in enumerate(cand_groups):
		for feature in cand_group.data['blip-group']:
			A[gj, feature] = 1

	# Assemble constraints, variables
	x = cp.Variable(ngroups)
	x.value = np.zeros(ngroups)
	b = np.ones(nrel)
	v_param = cp.Parameter()
	v_param.value = q
	v_var = cp.Variable(pos=True) # for FDR only

	# We perform a binary for FWER to find optimal v.
	if inclusions is not None and error == 'fwer' and search_method == 'binary':
		binary_search = True
	else:
		binary_search = False

	# Construct problem
	constraints = [
		x >= 0,
		x <= 1,
		A.T @ x <= b
	]
	# Last constraint is redundant for local_fdr
	if error in ['pfer', 'fwer']:
		constraints += [
			x @ peps <= v_param
		]
	# Extra constraints for fdr
	elif error == 'fdr':
		constraints += [
			x @ peps <= v_var,
			cp.sum(x) >= v_var / q,
			v_var >= 0,
			v_var <= q * nrel + 1
		]

	# Create problem
	objective = cp.Maximize(((1-peps) * weights) @ x)
	problem = cp.Problem(objective=objective, constraints=constraints)

	# Solve problem once if we do not need to search over v
	if not binary_search:
		problem.solve(solver=solver)
		selections = x.value

	# Solve FWER through binary search
	elif binary_search:
		v_upper = q * nrel + 1 # upper bnd in search (min val not controlling FWER)
		v_lower = 0 # lower bnd in search (max val controlling FWER)
		inclusions = inclusions != 0
		for niter in range(max_iters):
			# Change parametrized constraint
			v_current = (v_upper + v_lower)/2
			v_param.value = v_current 
			# Solve
			problem.solve(solver=solver, warm_start=True)
			selections = x.value

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
			if v_upper - v_lower < 1e-4:
				break

		# Solve with v_lower for final solution
		v_param.value = v_lower
		problem.solve(solver=solver, warm_start=True)
		if problem.status == 'infeasible':
			selections = np.zeros(ngroups)
		else:
			selections = x.value

		# TODO could do something smarter for FWER bin search
		if error == 'fwer' and binary_search:
			selections = np.around(selections)

	for cand_group, sprob, weight in zip(cand_groups, selections, weights):
		cand_group.data['sprob'] = sprob
		cand_group.data['weight'] = weight

	# Diagnostics to (optionally) return
	problem_status = dict(
		ngroups=ngroups,
		nrel=nrel,
		lp_bound=np.dot(selections, (1-peps) * weights),
		backtracking_iter=0,
		deterministic=deterministic,
	)

	return binarize_selections(
		cand_groups=cand_groups,
		q=q,
		error=error,
		deterministic=deterministic,
		problem_status=problem_status,
		return_problem_status=return_problem_status,
	)

def BLiP_cts(
	locs,
	grid_sizes,
	weight_fn=weight_fns.inverse_radius_weight,
	max_pep=0.25,
	max_blip_size=1500,
	**kwargs
):
	"""
	BLiP when the set of locations is continuous, e.g. when
	working with image data.

	Parameters
	----------
	locs : np.ndarray
		A (N, num_disc, d)-dimensional array. Here, N is the
		number of samples from the posterior, d is the number
		of dimensions of the space, and each point corresponds
		to a signal in a particular posterior sample. NANs do not
		count as signals (this is useful in case there are different
		numbers of signals at each posterior iteration.)
	grid_sizes : list or np.ndarray
		List of grid-sizes to split up the locations.
	weight_fn : string or function
			A function which takes a CandidateGroup as an input and
			returns a (nonnegative) weight as an output. This defines
			the expected power function. Alternatively, can be
			'inverse_size' or 'log_inverse_size'. Defaults to
			'inverse_size'.
	kwargs : dict
		Additional arguments to pass to the underlying BLiP call.

	Returns
	-------
	detections : list
		A list of CandidateGroups consisting of the detected signals
		and the regions containing them.
	"""

	# Normalize locations
	norm_locs, shifts, scales = create_groups_cts.normalize_locs(locs)

	# 1. Calculate filtered PEPs
	peps = create_groups_cts.grid_peps(
		locs=norm_locs, grid_sizes=grid_sizes, max_pep=max_pep
	)

	# 2. Calculate nodes, components, and so on
	all_cand_groups, _ = create_groups_cts.grid_peps_to_cand_groups(
		peps, max_blip_size=max_blip_size
	)

	# 3. Run BLiP
	all_rej = []
	for i, cand_groups in enumerate(all_cand_groups):
		rej = BLiP(
			cand_groups=cand_groups,
			weiht_fn=weight_fn,
			max_pep=max_pep,
			**kwargs,
		)
		all_rej.extend(rej)

	# 4. Renormalize locations
	d = locs.shape[2] # dimensionality
	for cand_group in all_rej:
		center = np.zeros(d)
		radius = cand_group.data.pop('radius') * scales + shifts
		for k in range(d):
			center[k] = cand_group.data.pop(f'dim{k}') * scales[k] + shifts[k]
		cand_group.data['center'] = center
		cand_group.data['radii'] = radius

	return all_rej


def binarize_selections(
	cand_groups,
	q,
	error,
	deterministic,
	problem_status=None,
	return_problem_status=False,
	tol=1e-3,
):
	"""
	Parameters
	----------
	cand_groups : list
		list of candidate groups. 
	q : float
		Level at which to control the error rate.
	error : string
		The error to control: one of 'fdr', 'fwer', 'pfer', 'local_fdr'
	deterministic : bool 
		If True, will not use a randomized solution.

	Returns
	-------
	detections : list
		List of candidate groups which have been detected.
	"""
	output = []
	if problem_status is None:
		problem_status = dict(backtracking_iter=0)

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
	problem_status['ngroups_nonint'] = ngroups
	if ngroups == 0 or ngroups == 1:
		if ngroups == 1:
			if not deterministic and np.random.uniform() < nontriv_cand_groups[0].data['sprob']:
				output.append(nontriv_cand_groups[0])
		if return_problem_status:
			return output, problem_status
		return output

	# Constraints to ensure selected groups are disjoint
	nontriv_cand_groups, nrel = create_groups._elim_redundant_features(nontriv_cand_groups)
	A = np.zeros((ngroups, nrel), dtype=bool)
	for gj, cand_group in enumerate(nontriv_cand_groups):
		for feature in cand_group.data['blip-group']:
			A[gj, feature] = 1

	# Sampling method
	if not deterministic:
		sprobs = np.array([cg.data['sprob'] for cg in nontriv_cand_groups])

		# Sort features in order of marg. prob of selection
		marg_probs = np.zeros(nrel)
		for j in range(nrel):
			marg_probs[j] = sprobs[A[:,j] == 1].sum()
		inds = np.argsort(-1*marg_probs)

		# Initialize
		eliminated_groups = np.zeros(ngroups).astype(bool)
		selected_groups = []
		# Loop through features and sample
		for feature in inds:
			if np.all(eliminated_groups):
				break
			# Subset of available groups which contain the feature
			available_flags = (A[:,feature] == 1) & (~eliminated_groups)
			if np.any(available_flags):
				# Scale up conditional probabilities
				prev_elim = (A[:,feature] == 1) & (eliminated_groups)
				scale = 1 - sprobs[prev_elim].sum()
				new_probs = sprobs[available_flags] / scale
				# select nothing with some probability
				if np.random.uniform() <= 1 - new_probs.sum():
					eliminated_groups[A[:,feature] == 1] = True
					continue
				# else select one of the groups containing feature
				selected_group = np.where(
					np.random.multinomial(1, new_probs / new_probs.sum()) != 0
				)[0][0]
				selected_group = np.where(available_flags)[0][selected_group]
				selected_groups.append(selected_group)
				# eliminate all mutually exclusive features
				group_features = np.where(A[selected_group]==1)[0]
				new_elim_groups = np.sum(A[:, group_features], axis=1) != 0
				eliminated_groups[new_elim_groups] = True

		output.extend([nontriv_cand_groups[sg] for sg in selected_groups])

	else:
		# Construct integer linear program 
		peps = np.array([cand_group.pep for cand_group in nontriv_cand_groups])
		weights = np.array([
			cand_group.data['weight'] for cand_group in nontriv_cand_groups
		])
		# Assemble constraints, variables
		x = cp.Variable(ngroups, boolean=True)
		x.value = np.zeros(ngroups)
		b = np.ones(nrel)
		# Account for discoveries already made
		ndisc_out = len(output)
		v_output = sum([cg.pep for cg in output])
		# Construct problem
		constraints = [
			A.T @ x <= b
		]
		objective = cp.Maximize(((1-peps) * weights) @ x)
		if error in ['pfer', 'fwer']:
			if error == 'pfer':
				v_opt = q
			else:
				v_opt = sum([cg.pep * cg.data['sprob'] for cg in cand_groups])
			v_new = v_opt - v_output
			constraints += [
				x @ peps <= v_new
			]
			problem = cp.Problem(objective=objective, constraints=constraints)
			problem.solve(solver=BIN_SOLVER)
		elif error == 'fdr':
			# Create output
			output = sorted(output, key=lambda x: x.pep)
			# Iteratively try to solve problem and then backtrack if infeasible
			# (backtracking is extremely rare)
			while len(output) >= 0:
				v_var = cp.Variable(pos=True)
				constraints_fdr = constraints + [
					x @ peps <= v_var,
					cp.sum(x) >= (v_var + v_output) / q - ndisc_out
				]
				problem = cp.Problem(objective=objective, constraints=constraints_fdr)
				try:
					problem.solve(solver='GLPK_MI')
				except cp.error.SolverError as e:
					problem.solve(solver=BIN_SOLVER)
				if problem.status != 'infeasible':
					break
				else:
					# This should never be triggered (is mathematically impossible
					if len(output) == 0:
						raise RuntimeError(
							f"Backtracking for FDR control failed"
						)
					# Backtrack by getting rid of last group
					problem_status['backtracking_iter'] += 1
					v_output -= output[-1].pep
					ndisc_out -= 1
					output = output[0:-1]

		if x.value is None:
			print(f"stats={problem.status}, nontriv_cand_groups={nontriv_cand_groups}")

		for binprob, x in zip(x.value, nontriv_cand_groups):
			if binprob > 1 - tol:
				output.append(x)

	if return_problem_status:
		return output, problem_status
	return output