import time
import numpy as np
import scipy as sp
from scipy import stats
import unittest
import pytest
from .context import pyblip
from pyblip import create_groups, create_groups_cts

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

	def check_peps_correct(self, peps, x, y, radius, locs, shape):
		if shape != 'circle':
			raise NotImplementedError("Only works for circles right now")
		dists = np.sqrt(np.power(
			locs - np.array([x, y]).reshape(1, 1, 2), 2
		).sum(axis=2))
		pip = np.mean(np.any(dists <= radius, axis=1))
		expected_pep = 1 - pip
		if expected_pep != 1:
			key = (x, y, radius)
			observed_pep = peps[key]
			np.testing.assert_almost_equal(
				observed_pep, 
				expected_pep,
				decimal=5,
				err_msg=f"PEP at {key} should be {expected_pep} but is {observed_pep}"
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

	def test_sequential_groups(self):
		
		# Generate fake data
		inclusions = np.array([
			[0, 1, 0, 0, 1],
			[1, 1, 0, 1, 1],
			[1, 0, 0, 0, 1],
			[0, 1, 0, 1, 1],
			[1, 0, 0, 1, 1],
		])

		cand_groups = create_groups.sequential_groups(
			inclusions, max_size=4
		)
		expected_length = int(5+4+3+2) - 1 # ignores columm with all zeros
		self.assertTrue(
			len(cand_groups)==expected_length,
			f"len(cand_groups)={len(cand_groups)} != expected length={expected_length}"
		)
		# Check that max_peps works
		for max_pep in [0.2, 0.4, 0.6, 0.8, 1]:
			cand_groups = create_groups.sequential_groups(
				inclusions, max_size=4, max_pep=max_pep
			)
			max_pep_obs = max([x.pep for x in cand_groups])
			self.assertTrue(
				max_pep_obs < max_pep,
				f"max pep in cand_groups is {max_pep_obs} > max_pep={max_pep}"
			)
			# Check for duplicates
			self.check_unique(cand_groups)

		# Check that prenarrowing works
		q = 0.5
		max_pep = 0.5
		cand_groups = create_groups.sequential_groups(
			inclusions, prenarrow=True, q=q, max_pep=max_pep
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

	def test_hierarchical_groups(self):
		
		# Generate fake data
		inclusions = np.array([
			[0, 1, 0, 0, 1],
			[1, 1, 0, 1, 1],
			[1, 0, 0, 0, 1],
			[0, 1, 0, 1, 1],
			[1, 0, 0, 1, 1],
		])

		# Check that the default includes the expected groups
		cand_groups = create_groups.hierarchical_groups(
			inclusions, filter_sequential=False
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
				inclusions, max_size=4, max_pep=max_pep
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
			peps, verbose=True, shape='square'
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




if __name__ == "__main__":
	unittest.main()
