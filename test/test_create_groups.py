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



# class TestCtsPEPs(unittest.TestCase):
#     """ Tests sparse bayesian linear reg """

#     def test_cts_peps(self):

#         # Simple 2d example with 2 discoveries
#         locs = np.array([
#             [[0.5502, 0.4502], [0.3, 0.2]],
#             [[0.549, 0.451], [0.305, 0.201]],
#             [[0.553, 0.456], [-1, -1]]
#         ])
        
#         # Run with one manual center added
#         xc1 = 0.55012
#         yc1 = 0.45012
#         peps = calc_peps_cts.grid_peps(
#             locs, 
#             grid_sizes=[10, 100, 1000],
#             extra_centers=np.array([[xc1, yc1]]),
#             log_interval=1, 
#             max_pep=1
#         )
#         # Check PEPs are right
#         pep1 = peps[(0.5, 0.4, 10)]
#         self.assertTrue(
#             pep1 == 0,
#             f"PEP at (0.5, 0.4, 10) should be 0 but is {pep1}"
#         )
#         pep2 = peps[(0.55, 0.45, 100)]
#         np.testing.assert_almost_equal(
#             pep2, 
#             1/3,
#             decimal=5,
#             err_msg=f"PEP at (0.5, 0.4, 100) should be 1/3 but is {pep2}"
#         )
#         pep3 = peps[(0.3, 0.2, 10)]
#         np.testing.assert_almost_equal(
#             pep3,
#             1/3,
#             decimal=5,
#             err_msg=f"PEP at (0.3, 0.2, 10) should be 1/3 but is {pep3}"
#         )
#         pep4 = peps[(np.around(xc1 - 0.05, 8), np.around(yc1 - 0.05, 8), 10)]
#         np.testing.assert_almost_equal(
#             pep4,
#             0,
#             decimal=5,
#             err_msg=f"PEP at {(xc1 - 0.05, yc1 - 0.05, 10)} should be 0 but is {pep4}"
#         )
#         # Compute BLiP nodes
#         all_nodes, components = calc_peps_cts.grid_peps_to_nodes(peps, verbose=True)
#         self.assertTrue(
#             len(components) == 1,
#             f"In tiny problem, number of components is {len(components)} > 1."
#         )

#         # Check that constraints are enforced
#         cent_flag = False # check that the correct center exists
#         for node1 in all_nodes[0]:
#             for node2 in all_nodes[0]:
#                 overlap_flag = True
#                 rad1 = node1.data['radius']
#                 rad2 = node2.data['radius']
#                 for j in range(2):
#                     if np.abs(node1.data[f'dim{j}'] - node2.data[f'dim{j}']) > rad1 + rad2:
#                         overlap_flag = False
#                 if overlap_flag:
#                     print(node1.data, node2.data)
#                     self.assertTrue(
#                         len(node1.data['group'].intersection(node2.data['group'])) > 0,
#                         f"node1 and node2 should overlap but don't: {node1.data}, {node2.data}"
#                     )
#             x1 = np.around(node1.data['dim0'], 5)
#             y1 = np.around(node1.data['dim1'], 5)
#             if x1 == 0.555 and y1 == 0.455:
#                 if np.abs(node1.data['pep'] - 1/3) < 1e-5:
#                     if np.abs(node1.data['radius'] - 0.005) < 1e-5:
#                         cent_flag = True

#         self.assertTrue(
#             cent_flag,
#             f'Final nodes are missing a node with the correct center/PEP'
#         )




if __name__ == "__main__":
    unittest.main()
