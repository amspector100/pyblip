import time
import numpy as np
import scipy as sp
from scipy import stats
import unittest
import pytest
import sys
try:
	from . import context
	from .context import pyblip
# For profiling
except ImportError:
	import context
	from context import pyblip
from pyblip import blip, create_groups
from pyblip.create_groups import CandidateGroup

TOL = 1e-5

class CheckDetections(unittest.TestCase):
	"""
	Helper methods to make sure the BLiP output is right.
	"""
	def check_error_control(self, detections, error, q, **kwargs):
		error = str(error).lower()
		if len(detections) == 0:
			return True
		if error == 'pfer':
			return self._check_pfer_control(detections, q)
		elif error == 'local_fdr':
			return self._check_local_fdr_control(detections, q)
		elif error == 'fwer':
			return self._check_fwer_control(detections, q, **kwargs)
		elif error == 'fdr':
			return self._check_fdr_control(detections, q)
		else:
			raise ValueError(f"Unrecognized error = {error}")

	def _check_pfer_control(self, detections, q):
		pfer = sum([x.pep for x in detections])
		self.assertTrue(
			pfer <= q + TOL, 
			f"PFER={pfer} > q={q} for detections={detections}"
		)

	def _check_fdr_control(self, detections, q):
		fdr = np.mean([x.pep for x in detections])
		self.assertTrue(
			fdr <= q + TOL,  
			f"FDR={fdr} > q={q} for detections={detections}"
		)


	def _check_local_fdr_control(self, detections, q):
		lfdr = np.max([x.pep for x in detections])
		self.assertTrue(
			lfdr <= q + TOL, 
			f"local_fdr={lfdr} > q={q} for detections={detections}"
		)

	def _check_fwer_control(self, detections, q, samples=None):
		if samples is None:
			self._check_pfer_control(detections, q)
		else:
			false_disc = np.zeros(samples.shape[0]).astype(bool)
			for cand_group in detections:
				false_disc = false_disc | np.all(
					samples[:, list(cand_group.group)] == 0, axis=1
				)
			fwer = false_disc.mean()
			self.assertTrue(
				fwer <= q + TOL,
				f"FWER={fwer} > q={q} for detections={detections}"
			)

	def check_disjoint(self, detections):
		groups = [x.group for x in detections]
		all_locs = set()
		for group in groups:
			self.assertTrue(
				len(all_locs.intersection(group)) == 0,
				f"all_locs={all_locs} has nonzero intersection with group={group}"
			)
			all_locs = all_locs.union(group)

class TestBinarizeSelections(CheckDetections):

	def test_pfer_simple(self):

		# Generate two randomized pairs and run the intlp
		cand_groups = [
			CandidateGroup(group=[0,], pep=0.19, data=dict(weight=0.5)),
			CandidateGroup(group=[0,1], pep=0.0, data=dict(weight=0.25)),
			CandidateGroup(group=[2], pep=0.19, data=dict(weight=1)),
			CandidateGroup(group=[2,3],pep=0.0, data=dict(weight=0.5))
		]
		for cand_group in cand_groups:
			cand_group.data['blip-group'] = cand_group.group
			cand_group.data['sprob'] = 0.5

		# Check that results are correct for intlp
		p = 4
		q = 0.2
		detections = blip.binarize_selections(
			cand_groups, q=q, error='pfer', deterministic=True
		)
		detected_groups = set([tuple(x.group) for x in detections])
		expected = set([(2,), (0,1)])
		self.assertTrue(
			detected_groups==expected,
			f"In simple example, intlp soln {detected_groups} != expected {expected}"
		)

		# Repeat for sampling method
		reps = 10
		all_detections = []
		for _ in range(reps):
			detections = blip.binarize_selections(
				cand_groups, q=q, error='pfer', deterministic=False
			)
			self.check_disjoint(detections)
			self.check_error_control(detections, error='pfer', q=q)

	def test_error_control_cand_groups(self):

		# Test a much more complicated example for deterministic=False
		np.random.seed(123)
		cand_groups = []
		p = 30
		inds = np.arange(p)
		for j in range(20):
			sprob = 0.01 + 0.98 *np.random.uniform()
			pep = 0.2 * np.random.uniform()
			group = set(np.random.choice(inds, j+1, replace=False).tolist())
			cand_groups.append(CandidateGroup(
				group=group,
				pep=pep,
				data={
					"weight":1/len(group),
					"sprob":sprob,
					"blip-group":group,
				}
			))
		qs = [0.1, 0.2, 0.5, 1.5, 2]
		reps = 2 # for randomized solution
		for q in qs:
			for error in ['pfer', 'fdr', 'local_fdr', 'fwer']:
				# Deterministic solution
				print(f"Not randomized, error={error}, q={q}")
				detections = blip.binarize_selections(
					cand_groups, q=q, error=error, deterministic=True
				)
				self.check_disjoint(detections)
				self.check_error_control(detections=detections, error=error, q=q)
				# Randomized solution
				for _ in range(reps):
					print(f"Randomized, error={error}, q={q}")
					detections = blip.binarize_selections(
						cand_groups, q=q, error=error, deterministic=False
					)
					self.check_disjoint(detections)
					self.check_error_control(detections=detections, error=error, q=q)

class TestBLiP(CheckDetections):

	def test_blip_regression(self):

		# Sample data
		np.random.seed(110)
		X, y, beta = context.generate_regression_data(
			a=5, b=1, y_dist='linear', p=200, n=100, sparsity=0.05
		)

		# Fit linear model
		lm = pyblip.linear.LinearSpikeSlab(X=X, y=y)
		lm.sample(N=500, chains=2)
		samples1 = lm.betas != 0

		# Just make random samples
		p2 = 500
		p0s = np.random.beta(a=1, b=3, size=p2)
		samples2 = np.random.binomial(1, p0s, size=(1000,p2))
		cand_groups2 = pyblip.create_groups.sequential_groups(
			samples=samples2
		)

		# Test Bayesian error control.
		for kwargs, samples in zip(
			[dict(samples=samples1), dict(cand_groups=cand_groups2)],
			[samples1, samples2],
		):
			for q in [0.01, 0.05, 0.1, 0.2]:
				for error, weight_fn in zip(
					['local_fdr', 'fdr', 'fwer', 'pfer'],
					['log_inverse_size', 'inverse_size', 'inverse_size', 'log_inverse_size'], 
				):
					for deterministic in [True, False]:
						detections = pyblip.blip.BLiP(
							error=error,
							q=q,
							search_method='binary',
							max_pep=2*q,
							deterministic=deterministic,
							**kwargs
						)
						self.check_disjoint(detections)
						self.check_error_control(
							detections=detections, error=error, samples=samples, q=q
						)

	def test_backtracking(self):
		# Cand groups created to require backtracking
		q = 0.1
		cand_groups = [
			CandidateGroup(group=[0,1], pep=0.0, data=dict(weight=1)),
			CandidateGroup(group=[1,2], pep=0.0, data=dict(weight=1)),
			CandidateGroup(group=[2,0], pep=0.0, data=dict(weight=1)),
			CandidateGroup(group=[3],pep=0.25, data=dict(weight=1))
		]
		# Try to control FDR
		detections, status = pyblip.blip.BLiP(
			cand_groups=cand_groups,
			error='fdr',
			weight_fn='prespecified',
			q=q,
			deterministic=True,
			return_problem_status=True,
		)
		bfdr = np.mean([x.pep for x in detections])
		self.assertTrue(
			bfdr <= q,
			f"BLiP violates FDR control (fdr={bfdr}, detections={detections}) for ex. requiring backtracking"
		)
		self.assertTrue(
			status['backtracking_iter'] == 1,
			f"BLiP runs the wrong number of backtracking iter (status={status}, should be 1 iter)"
		)
		expected = 1.5 + 0.75
		self.assertTrue(
			abs(status['lp_bound'] - expected) < 1e-3,
			f"LP bound for backtracking example is wrong (status={status}, should be {expected})"  
		)

		# Repeat and make sure we get the right answer
		q = 0.1
		cand_groups = [
			CandidateGroup(group=[0,1], pep=0.0, data=dict(weight=1.1)),
			CandidateGroup(group=[1,2], pep=0.0, data=dict(weight=1)),
			CandidateGroup(group=[2,0], pep=0.0, data=dict(weight=1)),
			CandidateGroup(group=[3],pep=0.1, data=dict(weight=1)),
			CandidateGroup(group=[4],pep=0.21, data=dict(weight=1)),
		]
		# Try to control FDR
		detections, status = pyblip.blip.BLiP(
			cand_groups=cand_groups,
			error='fdr',
			weight_fn='prespecified',
			q=q,
			deterministic=True,
			return_problem_status=True,
		)
		groups = set([tuple(x.group) for x in detections])
		expected = set([(0,1), (3,)])
		self.assertEqual(
			groups, expected, f"FDR solution for backtracking example #2 is wrong"
		)
		self.assertTrue(
			status['backtracking_iter'] == 1,
			f"BLiP runs the wrong number of backtracking iter (status={status}, should be 1 iter)"
		)



	def test_fdr_good_soln(self):
		# Cand groups created to be adversarially tricky
		cand_groups = [
			CandidateGroup(group=[0,], pep=0.05, data=dict(weight=1)),
			CandidateGroup(group=[1], pep=0.1, data=dict(weight=1)),
			CandidateGroup(group=[2], pep=0.05, data=dict(weight=1/100)),
			CandidateGroup(group=[3],pep=0.05, data=dict(weight=1/200))
		]
		# FDR
		detections, status = pyblip.blip.BLiP(
			cand_groups=cand_groups,
			error='fdr',
			weight_fn='prespecified',
			q=0.05,
			deterministic=True,
			return_problem_status=True,
		)
		groups = set([tuple(x.group) for x in detections])
		expected = set([(0,), (2,), (3,)])
		self.assertEqual(
			groups, expected, f"FDR solution for adversarial example #1 is wrong"
		)

		# Another adversarial example where higher PFER is bad
		cand_groups = [
			CandidateGroup(group=[0,1], pep=0.001, data=dict(weight=0.05)),
			CandidateGroup(group=[1,2], pep=0.05, data=dict(weight=1)),
			CandidateGroup(group=[2,], pep=0.0001, data=dict(weight=0.05)),
		]
		# BLiP for FDR control
		detections = pyblip.blip.BLiP(
			cand_groups=cand_groups,
			error='fdr',
			weight_fn='prespecified',
			q=0.05
		)
		groups = set([tuple(x.group) for x in detections])
		expected = set([(1,2)])
		self.assertEqual(
			groups, expected, f"FDR solution for adversarial example #2 is wrong"
		)



if __name__ == "__main__":
	# Run all tests---useful if using cprofilev
	if sys.argv[0] == 'test/test_blip.py':
		time0 = time.time()
		context.run_all_tests([TestBLiP(), TestBinarizeSelections()])
		elapsed = np.around(time.time() - time0, 2)
		print(f"Finished running all tests at time={elapsed}")

	# Else let unittest handle this for us
	else:
		unittest.main()


