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

class TestCandGroups(CheckCandGroups):
	"""
	Tests creation of candidate groups when the set of locations
	is discrete, e.g. when the locations correspond to features
	in variable selection problems.
	"""

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



class TestCtsPEPs(unittest.TestCase):
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


	def test_square_groups_cts(self):

		# Simple 2d example with 2 discoveries
		locs = np.array([
			[[0.5502, 0.4502], [0.3, 0.2]],
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
		print(peps)
		pep1 = peps[(0.5, 0.4, 1/20)]
		self.assertTrue(
			pep1 == 0,
			f"PEP at (0.5, 0.4, 1/20) should be 0 but is {pep1}"
		)
		pep2 = peps[(0.55, 0.45, 1/200)]
		np.testing.assert_almost_equal(
			pep2, 
			1/3,
			decimal=5,
			err_msg=f"PEP at (0.5, 0.4, 1/200) should be 1/3 but is {pep2}"
		)
		pep3 = peps[(0.3, 0.2, 1/2000)]
		np.testing.assert_almost_equal(
			pep3,
			1/3,
			decimal=5,
			err_msg=f"PEP at (0.3, 0.2, 1/2000) should be 1/3 but is {pep3}"
		)
		pep4 = peps[(np.around(xc1 - 0.05, 8), np.around(yc1 - 0.05, 8), 10)]
		np.testing.assert_almost_equal(
			pep4,
			0,
			decimal=5,
			err_msg=f"PEP at {(xc1 - 0.05, yc1 - 0.05, 10)} should be 0 but is {pep4}"
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
			if x1 == 0.555 and y1 == 0.455:
				if np.abs(cand_group1.pep - 1/3) < 1e-5:
					if np.abs(cand_group1.data['radius'] - 0.005) < 1e-5:
						cent_flag = True

		self.assertTrue(
			cent_flag,
			f'Final nodes are missing a node with the correct center/PEP'
		)




if __name__ == "__main__":
	unittest.main()
