import time
import numpy as np
import scipy as sp
import scipy.linalg
from scipy import stats
import unittest
import pytest
from .context import pyblip
from collections import Counter
from pyblip import create_groups, create_groups_cts
from pyblip.create_groups import CandidateGroup

def AR1_corr(p):
	# create ar1 corr matrix 
	inds = np.arange(p)
	diffs = np.abs(inds.reshape(-1, 1) - inds.reshape(1, -1))
	diffs = np.minimum(diffs, 50)
	return np.exp(-1*diffs/25)

class CheckCandGroups(unittest.TestCase):
	"""
	Helper functions for testing candidate groups.
	"""
	def check_unique(self, cand_groups):
		groups = [list(x.group) for x in cand_groups]
		dedup_groups = create_groups._dedup_list_of_lists(groups)
		self.assertTrue(
			len(groups)==len(dedup_groups),
			f"groups={groups} appears to contain duplicates with len(dedup_groups)={len(dedup_groups)}"
		)

	def check_peps_correct(self, peps, x, y, radius, locs, shape, count_signals=False):
		if shape != 'circle':
			raise NotImplementedError("Only works for circles right now")
		dists = np.sqrt(np.power(
			locs - np.array([x, y]).reshape(1, 1, 2), 2
		).sum(axis=2))
		pip = np.mean(np.any(dists <= radius, axis=1))
		expected_pep = 1 - pip
		if expected_pep != 1:
			key = (x, y, radius)
			if not count_signals:
				observed_pep = peps[key]
			else:
				observed_pep = 1 - peps[key]['pip']
			np.testing.assert_almost_equal(
				observed_pep, 
				expected_pep,
				decimal=5,
				err_msg=f"PEP at {key} should be {expected_pep} but is {observed_pep}"
			)
			if count_signals:
				signal_counts = np.sum(dists <= radius, axis=1)
				counter = Counter(signal_counts.tolist())
				for nsignal in counter:
					if nsignal == 0:
						continue
					expected_prop = counter[nsignal] / len(signal_counts)
					observed_prop = peps[key][nsignal]
					np.testing.assert_almost_equal(
						expected_prop, 
						observed_prop,
						decimal=5,
						err_msg=f"Prop. nsignals={nsignal} at {key} should be {expected_prop} but is {observed_prop}"
					)

	def check_blip_groups_correct(self, cand_groups):
		for j1, cg1 in enumerate(cand_groups):
			g1 = cg1.group
			bg1 = cg1.data['blip-group']
			for j2 in range(j1):
				g2 = cand_groups[j2].group
				bg2 = cand_groups[j2].data['blip-group']
				if len(g1.intersection(g2)) > 0:
					self.assertTrue(
						len(bg1.intersection(bg2)) > 0,
						f'Groups {g1}, {g2} overlap but BLiP groups {bg1}, {bg2} do not. cgs={cand_groups}.'
					)
				else:
					self.assertTrue(
						len(bg1.intersection(bg2)) == 0,
						f'Groups {g1}, {g2} do not overlap but BLiP groups {bg1}, {bg2} do. cgs={cand_groups}.'
					)
		
	def check_cand_groups_correct(self, cand_groups, shape):
		if shape != 'circle':
			raise NotImplementedError("Only works for circles currently")
		# Check that overlap calculations are correct
		for j1, cg1 in enumerate(cand_groups):
			g1 = cg1.group
			for j2, cg2 in enumerate(cand_groups):
				if j2 >= j1:
					continue
				g2 = cg2.group
				c1 = np.array(cg1.data['center'])
				c2 = np.array(cg2.data['center'])
				dist = np.sqrt(np.sum(np.power(c1-c2, 2)))
				r1, r2 =  cg1.data['radius'], cg2.data['radius']
				if len(g1.intersection(g2)) > 0:
					self.assertTrue(
						dist < r1 + r2,
						f"""
						BLiP posits c1, c2 groups ({g1}, {g2}) overlap. Centers= {c1} and {c2} with dist={dist}
						and r1={r1}, r2={r2}, j1={j1}, j2={j2}.
						"""
					)
				else:
					self.assertTrue(
						dist >= r1 + r2 + 1e-5,
						f"""
						BLiP posits c1, c2 groups ({g1}, {g2}) do not overlap. Centers= {c1} and {c2}  with dist={dist}
						and r1={r1}, r2={r2}, j1={j1}, j2={j2}.
						"""
					)

class TestCandGroups(CheckCandGroups):
	"""
	Tests creation of candidate groups when the set of locations
	is discrete, e.g. when the locations correspond to features
	in variable selection problems.
	"""
	def test_deduplication(self):
		# Simple test to make sure _dedup is working
		groups = [[0,1], [1,0], np.array([0,1]), np.array([1,0]), (0,1)]
		dedup = create_groups._dedup_list_of_lists(groups)
		self.assertEqual(
			len(dedup),
			1,
			"_dedup_list_of_lists fails to accurately deduplicate"
		)

	def test_ecc_reduction(self):
		cand_groups1 = [
			CandidateGroup(pep=0, group=set([0,1])),
			CandidateGroup(pep=0, group=set([1,2])),
			CandidateGroup(pep=0, group=set([0,2]))
		]
		ncg = 100
		p = 50
		cand_groups2 = []
		for j in range(ncg):
			cand_groups2.append(
				CandidateGroup(
					pep=0, 
					group=set(np.random.choice(np.arange(p), 5))
				)
			)
		for i, cand_groups in enumerate([cand_groups1, cand_groups2]):
			cand_groups, nrel = pyblip.create_groups._ecc_reduction(cand_groups)
			self.check_blip_groups_correct(cand_groups)
			if i == 0:
				self.assertTrue(
					nrel==1, f"For simple problem nrel={nrel}, should be 1, cgs={cand_groups}."
				)

	def test_sequential_groups(self):
		
		# Generate fake data
		samples = np.array([
			[0, 1, 0, 0, 1],
			[1, 1, 0, 1, 1],
			[1, 0, 0, 0, 1],
			[0, 1, 0, 1, 1],
			[1, 0, 0, 1, 1],
		])

		cand_groups = create_groups.sequential_groups(
			samples, max_size=4, max_pep=1
		)
		expected_length = int(5+4+3+2) - 1 # ignores columm with all zeros
		self.assertTrue(
			len(cand_groups)==expected_length,
			f"len(cand_groups)={len(cand_groups)} != expected length={expected_length}"
		)
		# Check that max_peps works
		for max_pep in [0.2, 0.4, 0.6, 0.8, 1]:
			cand_groups = create_groups.sequential_groups(
				samples, max_size=4, max_pep=max_pep
			)
			max_pep_obs = max([x.pep for x in cand_groups])
			self.assertTrue(
				max_pep_obs < max_pep,
				f"max pep in cand_groups is {max_pep_obs} > max_pep={max_pep}"
			)
			# Check for duplicates
			self.check_unique(cand_groups)

		# Check that prenarrowing works
		q = 0.9
		max_pep = 0.5
		cand_groups = create_groups.sequential_groups(
			samples, prenarrow=True, q=q, max_pep=max_pep
		)
		self.check_unique(cand_groups)
		groups = [x.group for x in cand_groups]
		self.assertTrue(
			set([0]) in groups,
			f"For q={q}, group=(0) should be in groups={groups}"
		)
		self.assertTrue(
			set([0,1]) not in groups,
			f"For q={q}, group=(0,1) was not prenarrowed out of groups={groups}"
		)

		# Finally, check that we can successfully eliminate the last redundant feature
		cand_groups, nrel = create_groups._elim_redundant_features(
			cand_groups
		)
		self.assertTrue(
			nrel==4,
			"number of relevant features is should be 4 but equals {nrel}"
		)

	def test_all_groups(self):
		# generate fake samples
		n = 1000
		p = 500
		probs = np.random.beta(0.4, 0.4, size=(p,))
		samples = np.random.binomial(
			1, probs, size=(n,p)
		)
		# AR1 corr matrix
		C = AR1_corr(p)
		X = np.random.randn(n,p) @ np.linalg.cholesky(C).T
		# get all groups
		cgs = create_groups.all_cand_groups(
			samples=samples, 
			X=X,
			prenarrow=False,
			max_pep=1
		)
		# Check that no groups are repeated
		self.check_unique(cgs)

		# test that the min_purity arg works
		min_purity = 0.8
		hatC = np.abs(np.corrcoef(X.T))
		cgs = create_groups.all_cand_groups(
			samples=samples, 
			X=X,
			prenarrow=True,
			max_pep=0.1,
			min_purity=min_purity,
		)
		for cg in cgs:
			if len(cg.group) == 1:
				continue
			group = np.array(list(cg.group))
			purity = hatC[group][:, group].min()
			self.assertTrue(
				purity >= min_purity,
				f"outputted cg with purity={purity} but min_purity={min_purity}"
			)
		self.check_unique(cgs)

		## check if there are issues with no nonnulls
		samples = np.zeros((n, p))
		cgs = create_groups.all_cand_groups(
			samples=samples, 
			X=X,
			prenarrow=False,
			max_pep=0.5
		)
		self.assertTrue(len(cgs) == 0, f"with all PEPs=1, len(cgs)={len(cgs)} but should be 0")

	def test_hierarchical_groups(self):
		
		# Generate fake data
		samples = np.array([
			[0, 1, 0, 0, 1],
			[1, 1, 0, 1, 1],
			[1, 0, 0, 0, 1],
			[0, 1, 0, 1, 1],
			[1, 0, 0, 1, 1],
		])

		# Check that the default includes the expected groups
		cand_groups = create_groups.hierarchical_groups(
			samples, filter_sequential=False, max_pep=1
		)
		groups = [x.group for x in cand_groups]
		expected = [set([i]) for i in [0,1,3,4]] + [set([0,1]), set([0,1,2,3,4])]
		for expect in expected:
			self.assertTrue(
				expect in groups,
				f"groups={expect} was unexpectedly not in groups={groups}"
			)

		# Check that max_peps works
		for max_pep in [0.2, 0.4, 0.6, 0.8, 1]:
			cand_groups = create_groups.hierarchical_groups(
				samples, max_size=4, max_pep=max_pep
			)
			max_pep_obs = max([x.pep for x in cand_groups])
			self.assertTrue(
				max_pep_obs < max_pep,
				f"max pep in cand_groups is {max_pep_obs} > max_pep={max_pep}"
			)
			# Check for duplicates
			self.check_unique(cand_groups)

	def test_susie_groups(self):
		# susie alphas, L = 3, p = 5
		alphas = np.array([
			[0.89, 0.07, 0.04, 0.0, 0.0],
			[0.05, 0.25, 0.02, 0.03, 0.65],
			[0.85, 0.01, 0.01, 0.01, 0.12],
			[0.01, 0.01, 0.01, 0.01, 0.96]
		])
		n = 20
		L, p = alphas.shape
		# Create groups
		cand_groups = create_groups.susie_groups(
			alphas=alphas,
			X=np.random.randn(n, p),
			q=0.1,
			max_size=5,
			max_pep=1,
			prenarrow=False
		)
		self.check_unique(cand_groups)
		# Check existence of several groups
		groups = [x.group for x in cand_groups]
		expected = [set([i]) for i in [0,1,2,3,4]] + [set([0,1]), set([1,4]), set([0,4])]
		for expect in expected:
			self.assertTrue(
				expect in groups,
				f"groups={expect} was unexpectedly not in groups={groups}"
			)
		# Check peps are correct
		for g in [[0], [0,1], [1,4], [0,4]]:
			pep = np.exp(np.log(1 - alphas[:,g].sum(axis=1)).sum())
			cgs = [x for x in cand_groups if x.group == set(g)]
			cg = cgs[0]
			np.testing.assert_almost_equal(
				cg.pep, 
				pep,
				decimal=10,
				err_msg=f'susie_groups computed the wrong pep for group={g}, pep={cg.pep}, expected={pep}'
			)

	def test_susie_groups_purity(self):
		# susie alphas, L = 3, p = 5
		p = 5
		n = 10000
		rho = 0.9

		# Correlation matrix
		c = np.cumsum(np.zeros(p) + np.log(rho)) - np.log(rho)
		cov = scipy.linalg.toeplitz(np.exp(c))
		X = np.dot(np.random.randn(n, p), np.linalg.cholesky(cov).T)

		# susie alphas
		alphas = np.array([
			[0.1, 0.5, 0.4, 0.0, 0.0],
			[0.5, 0.0, 0.0, 0.0, 0.5],
			[0.2, 0.2, 0.2, 0.2, 0.2],
		])
		pt = 0.7
		cand_groups = create_groups.susie_groups(
			alphas, 
			X=X, 
			purity_threshold=pt,
			q=0.1,
			max_pep=1
		)
		for cg in cand_groups:
			expected = 1 - alphas[0, list(cg.group)].sum()
			np.testing.assert_almost_equal(
				expected, cg.pep, 3,
				f"PEP {cg.pep} != expected {expected} for susie group={cg.group} with purity threshold={pt}"

			)



class TestCtsPEPs(CheckCandGroups):
	"""
	Tests creation of candidate groups when the set of locations
	is continuous.
	"""
	def test_normalize_locs(self):
		# Create unnormalized locs
		N = 100
		n_disc = 10
		d = 5
		locs = np.random.randn(N, n_disc, d)
		locs[12, 5, :] = np.nan

		# Normalize
		norm_locs, shifts, scales = create_groups_cts.normalize_locs(locs)

		# Test NANs / max / min values
		self.assertTrue(
			np.all(np.isnan(norm_locs[np.isnan(locs)])),
			f"norm_locs is not always nan where locs is nan"
		)
		self.assertTrue(
			np.all(~np.isnan(norm_locs[~np.isnan(locs)])),
			f"norm_locs is sometimes nan where locs is not"
		)
		np.testing.assert_almost_equal(
			np.nanmin(norm_locs),
			0,
			decimal=6,
			err_msg=f"Min of norm_locs is {np.nanmin(norm_locs)} != 0"
		)
		np.testing.assert_almost_equal(
			np.nanmax(norm_locs),
			1,
			decimal=6,
			err_msg=f"Max of norm_locs is {np.nanmax(norm_locs)} != 1"
		)

		# Test that we can recreate original locs
		np.testing.assert_array_almost_equal(
			norm_locs * scales +  shifts,
			locs,
			decimal=6,
			err_msg=f"Cannot recreate locs from norm_locs"
		)

	def test_circular_groups_cts(self):

		# 2d ex with 2 discoveries
		locs = np.array([
			[[0.13054,0.410234], [np.nan, np.nan]],
			[[0.12958,0.406639], [0.46009, 0.459]],
			[[0.46001, 0.45119], [0.1302, 0.4126]],
			[[np.nan, np.nan], [0.95, 0.899]],
			[[np.nan, np.nan], [np.nan, np.nan]]
		])
		grid_sizes = [10, 100, 1000]
		peps = create_groups_cts.grid_peps(
			locs, 
			grid_sizes=grid_sizes,
			log_interval=1,
			max_pep=1,
			shape='circle',
		)

		r1 = np.sqrt(2) / 20
		r2 = np.sqrt(2) / 200
		for x, y, radius in zip(
			[0.45, 0.135, 0.135, 0.95, 0.95, 0.455],
			[0.45, 0.405, 0.415, 0.95, 0.85, 0.465],
			[r1, r2, r2, r1, r1, r2]
		):
			self.check_peps_correct(
				peps=peps,
				locs=locs,
				x=x,
				y=y,
				radius=radius,
				shape='circle',
			)

		# Compute BLiP nodes
		all_cgroups, components = create_groups_cts.grid_peps_to_cand_groups(
			peps, verbose=True, shape='circle'
		)
		self.assertTrue(
			len(components) == 1,
			f"In tiny problem, number of components is {len(components)} > 1."
		)

		# Check that cand groups are right
		for cand_group1 in all_cgroups[0]:
			rad1 = cand_group1.data['radius']
			cent1 = np.array(cand_group1.data['center'])
			for cand_group2 in all_cgroups[0]:
				rad2 = cand_group2.data['radius']
				cent2 = np.array(cand_group2.data['center'])
				dist = np.sqrt(np.power(cent1 - cent2, 2).sum())
				if dist < rad1 + rad2:
					print(cand_group1.data, cand_group2.data)
					self.assertTrue(
						len(cand_group1.group.intersection(cand_group2.group)) > 0,
						f"cand_group1 and cand_group2 should overlap but don't: {cand_group1.data}, {cand_group2.data}"
					)


	def test_square_groups_cts(self):

		# Simple 2d example with 2 discoveries
		locs = np.array([
			[[0.5502, 0.4502], [0.30000001, 0.2000001]],
			[[0.549, 0.451], [0.305, 0.201]],
			[[0.553, 0.456], [np.nan, np.nan]]
		])

		# Run with one manual center added
		xc1 = 0.55012
		yc1 = 0.45012
		peps = create_groups_cts.grid_peps(
			locs, 
			grid_sizes=[10, 100, 1000],
			extra_centers=np.array([[xc1, yc1]]),
			log_interval=1, 
			max_pep=1,
			shape='square'
		)
		# Check PEPs are right
		pep1 = peps[(0.55, 0.45, 1/20)]
		self.assertTrue(
			pep1 == 0,
			f"PEP at (0.55, 0.45, 1/20) should be 0 but is {pep1}"
		)
		pep2 = peps[(0.555, 0.455, 1/200)]
		np.testing.assert_almost_equal(
			pep2, 
			1/3,
			decimal=5,
			err_msg=f"PEP at (0.555, 0.455, 1/200) should be 1/3 but is {pep2}"
		)
		pep3 = peps[(0.3005, 0.2005, 1/2000)]
		np.testing.assert_almost_equal(
			pep3,
			2/3,
			decimal=5,
			err_msg=f"PEP at (0.3005, 0.2005, 1/2000) should be 2/3 but is {pep3}"
		)
		key4 = (np.around(xc1, 8), np.around(yc1, 8), 1/20)
		pep4 = peps[key4]
		np.testing.assert_almost_equal(
			pep4,
			0,
			decimal=5,
			err_msg=f"PEP at {key4} should be 0 but is {pep4}"
		)
		# Compute BLiP nodes
		all_cgroups, components = create_groups_cts.grid_peps_to_cand_groups(
			peps, verbose=True, shape='square'
		)
		self.assertTrue(
			len(components) == 1,
			f"In tiny problem, number of components is {len(components)} > 1."
		)

		# Check that constraints are enforced
		cent_flag = False # check that the correct center exists
		for cand_group1 in all_cgroups[0]:
			for cand_group2 in all_cgroups[0]:
				overlap_flag = True
				rad1 = cand_group1.data['radius']
				rad2 = cand_group2.data['radius']
				for j in range(2):
					if np.abs(cand_group1.data[f'dim{j}'] - cand_group2.data[f'dim{j}']) > rad1 + rad2:
						overlap_flag = False
				if overlap_flag:
					print(cand_group1.data, cand_group2.data)
					self.assertTrue(
						len(cand_group1.group.intersection(cand_group2.group)) > 0,
						f"cand_group1 and cand_group2 should overlap but don't: {cand_group1.data}, {cand_group2.data}"
					)
			x1 = np.around(cand_group1.data['dim0'], 5)
			y1 = np.around(cand_group1.data['dim1'], 5)
			if np.abs(x1 - 0.555) < 1e-5 and np.abs(y1 - 0.455) < 1e-5:
				if np.abs(cand_group1.pep - 1/3) < 1e-5:
					if np.abs(cand_group1.data['radius'] - 0.005) < 1e-5:
						cent_flag = True

		self.assertTrue(
			cent_flag,
			f'Final nodes are missing a node with the correct center/PEP'
		)

	def test_count_signals(self):

		# Simple 2d example with 2 discoveries
		locs = np.array([
			[[0.5502, 0.4502], [0.5452, 0.4452]],
			[[0.5502, 0.4502], [np.nan, np.nan]],
			[[0.5502, 0.4502], [np.nan, np.nan]],
			[[0.6082, 0.1502], [0.6082, 0.1503]],
		])
		# Run with one manual center added
		xc1 = 0.55012
		yc1 = 0.45012
		peps = create_groups_cts.grid_peps(
			locs, 
			grid_sizes=[10, 100, 1000],
			count_signals=True,
			extra_centers=np.array([[xc1, yc1]]),
			log_interval=1, 
			max_pep=1,
			shape='circle'
		)
		r1 = np.sqrt(2) / 20
		r2 = np.sqrt(2) / 200
		for x, y, radius in zip(
			[0.55, 0.555, 0.65, 0.605,],
			[0.45, 0.455, 0.15, 0.155,],
			[r1, r2,r1, r2]
		):
			self.check_peps_correct(
				peps=peps,
				locs=locs,
				x=x,
				y=y,
				radius=radius,
				shape='circle',
				count_signals=True,
			)

	def test_grid_peps_to_cand_groups(self):

		# Notes:
		# (1) The first two overlap if they are rectangles, but not if
		# they are circles.
		# (2) The second two do overlap
		# (3) The last two do not overlap

		# (center_x, center_y, key) --> pep
		peps1 = {
			(0.12, 0.15, 0.01):0.113,
			(0.13, 0.16, 0.001):0.514, 
			(0.25, 0.23, 0.011):0.153,
			(0.25, 0.23, 0.053):0.0513,
			(0.421, 0.493, 0.01414):0.0999,
			(0.419, 0.522, 0.01):0.01,
			(0.419, 0.522, 0.05):0.001,
			(0.419, 0.522, 0.10):0.0,
		}

		# Similar example when counting the number of signals
		peps2 = {
			(0.12, 0.15, 0.01):{'pip':0.993, 1:0.5, 2:0.4, 3:0.01},
			(0.13, 0.16, 0.0001):{'pip':0.134, 1:0.0213, 2:0.09234, 3:0.01},
			(0.25, 0.23, 0.011):{'pip':0.368, 1:0.1267, 2:0.4, 3:0.2},
			(0.25, 0.23, 0.053):{'pip':0.456, 1:0.1674, 2:0.168, 3:0.9},
			(0.421, 0.493, 0.01414):{'pip':0.345, 1:0.2347, 2:0.4, 3:0.4},
		}

		# Random example to really test things
		peps3 = dict()
		for j in range(100):
			key = tuple(np.around(np.random.uniform(size=(3,)), 5))
			pep = np.random.uniform()
			peps3[key] = pep

		# Test all of these examples
		for peps in [peps1, peps2, peps3]:
			cand_groups, _ = create_groups_cts.grid_peps_to_cand_groups(
				filtered_peps=peps, verbose=True, shape='circle'
			)
			self.assertTrue(
				len(cand_groups) == 1,
				f"In simple ex, there are {len(cand_groups)} components, but should only be one."
			)
			self.check_cand_groups_correct(
				cand_groups[0], shape='circle'
			)

		# Lastly, test that this finds the optimal number of locations
		peps1 = {
			(0.12, 0.15, 0.01):0,
			(0.12, 0.15, 0.02):0,
			(0.12, 0.15, 0.03):0,
			(0.12, 0.15, 0.04):0,
			(0.12, 0.15, 0.05):0,
			(0.12, 0.15, 0.06):0,
		}
		peps2 = {
			(0.12, 0.15, 0.01):0,
			(0.12, 0.15, 0.02):0,
			(0.12, 0.15, 0.05):0,
			(0.20, 0.20, 0.05):0,
			(0.20, 0.20, 0.001):0,
			(0.20, 0.20, 0.002):0,
		}
		opt_nlocs = [1, 3]
		for peps, opt_nloc  in zip([peps1, peps2], opt_nlocs):
			cand_groups, _ = create_groups_cts.grid_peps_to_cand_groups(
				filtered_peps=peps, verbose=True, shape='circle'
			)
			# Run previous tests for good measure
			self.assertTrue(
				len(cand_groups) == 1,
				f"In simple ex, there are {len(cand_groups)} components, but should only be one."
			)
			self.check_cand_groups_correct(
				cand_groups[0], shape='circle'
			)
			# Check number of locations
			nloc = len(set([j for x in cand_groups[0] for j in x.group]))
			self.assertTrue(
				nloc == opt_nloc,
				f"In simple ex, there are {nloc} locations, but optimal number is {opt_nloc}."
			)


if __name__ == "__main__":
	unittest.main()
