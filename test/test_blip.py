import time
import numpy as np
import scipy as sp
from scipy import stats
import unittest
import pytest
from . import context
from .context import pyblip
from pyblip import blip, create_groups
from pyblip.create_groups import CandidateGroup
from knockpy import dgp

class CheckDetections(unittest.TestCase):
	"""
	Helper methods to make sure the BLiP output is right.
	"""
	def check_disjoint(self, detections):
		groups = [x.group for x in detections]
		all_locs = set()
		for group in groups:
			self.assertTrue(
				len(all_locs.intersection(group)) == 0,
				f"all_locs={all_locs} has nonzero intersection with group={group}"
			)
			all_locs = all_locs.union(group)

	def check_pfer_control(self, detections, v):
		if isinstance(detections[0], CandidateGroup):
			detections = [detections]
		pfer = 0
		for d in detections:
			pfer += sum([x.pep for x in d]) / len(detections)
		self.assertTrue(
			pfer <= v,
			f"average pfer {pfer} is larger than nominal level {v}"
		)

	def check_fwer_control(self, detections, inclusions, q):
		if isinstance(detections[0], CandidateGroup):
			detections = [detections]
		fwer = 0
		for d in detections:
			false_disc = np.zeros(inclusions.shape[0]).astype(bool)
			for cand_group in d:
				false_disc = false_disc | np.all(
					inclusions[:, list(cand_group.group)] == 0, axis=1
				)
			fwer += false_disc.mean() / len(detections)
		self.assertTrue(
			fwer <= q,
			f"average fwer {fwer} is larger than nominal level {q}"
		)

	def check_fdr_control(self, detections, q):
		if isinstance(detections[0], CandidateGroup):
			detections = [detections]
		fdr = 0
		for d in detections:
			fdr += np.mean([x.pep for x in d]) / len(detections)
		self.assertTrue(
			fdr <= q,
			f"average fdr {fdr} is larger than nominal level {q}"
		)


class TestBinarizeSelections(CheckDetections):

	def test_pfer_simple(self):

		# Generate two randomized pairs and run the intlp
		cand_groups = [
			CandidateGroup(group=[0,], pep=0.2, data=dict(weight=0.5)),
			CandidateGroup(group=[0,1], pep=0.0, data=dict(weight=0.25)),
			CandidateGroup(group=[2], pep=0.2, data=dict(weight=1)),
			CandidateGroup(group=[2,3],pep=0.0, data=dict(weight=0.5))
		]
		for cand_group in cand_groups:
			cand_group.data['blip-group'] = cand_group.group
			cand_group.data['sprob'] = 0.5

		# Check that results are correct for intlp
		p = 4
		v = 0.2
		detections = blip.binarize_selections(
			cand_groups, p=p, v_opt=v, error='pfer', how_binarize='intlp'
		)
		detected_groups = set([tuple(x.group) for x in detections])
		expected = set([(2,), (0,1)])
		self.assertTrue(
			detected_groups==expected,
			f"In simple example, intlp soln {detected_groups} != expected {expected}"
		)

		# Repeat for sampling method
		reps = 128
		all_detections = []
		for _ in range(reps):
			detections = blip.binarize_selections(
				cand_groups, p=p, v_opt=v, error='pfer', how_binarize='sample'
			)
			self.check_disjoint(detections)
			all_detections.append(detections)

		self.check_pfer_control(all_detections, v=v + 0.25 / np.sqrt(reps))

	def test_pfer_complex(self):

		# Test a much more complicated example
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
		vs = [0.1, 0.2, 0.5, 1.5, 2]
		for v in vs:
			detections = blip.binarize_selections(
				cand_groups, p=p, v_opt=v, error='pfer', how_binarize='intlp'
			)
			self.check_disjoint(detections)
			self.check_pfer_control(detections=detections, v=v)
			# For smallest v
			if v == vs[0]:
				reps = 64
				all_detections = []
				for _ in range(reps):
					detections = blip.binarize_selections(
						cand_groups, p=p, v_opt=v, error='pfer', how_binarize='sample'
					)
					self.check_disjoint(detections)
					all_detections.append(detections)
				self.check_pfer_control(all_detections, v=v + 0.25 / np.sqrt(reps))


class TestBLiP(CheckDetections):

	def test_blip_regression(self):

		# Sample data
		np.random.seed(110)
		X, y, beta = context.generate_regression_data(
			a=5, b=1, y_dist='linear', p=500, n=100, sparsity=0.05
		)
		print(np.where(beta != 0))

		# Fit linear model
		lm = pyblip.linear.LinearSpikeSlab(X=X, y=y)
		lm.sample(N=500, chains=5)
		inclusions = lm.betas != 0
		print(np.mean(inclusions, axis=0))

		# Test FDR, FWER, PFER, local FDR control.
		# Note this just test that the Bayesian FDR is controlled
		# assuming the model is well-specified, it doesn't check
		# the frequentist FDR (which would be expensive).
		for q in [0.01, 0.05, 0.1, 0.2]:
			detections = pyblip.blip.BLiP(
				inclusions=inclusions,
				error='fwer',
				q=q,
				search_method='binary'
			)
			self.check_disjoint(detections)
			self.check_fwer_control(detections=detections, inclusions=inclusions, q=q)

			# Test FDR control
			detections = pyblip.blip.BLiP(
				inclusions=inclusions,
				error='fdr',
				q=q,
				how_binarize='intlp'
			)
			self.check_disjoint(detections)
			self.check_fdr_control(detections, q=q)
			#print(f"for q={q}, realized FDR={np.mean([x.pep for x in detections])}")

			# PFER control
			detections = pyblip.blip.BLiP(
				inclusions=inclusions,
				weight_fn='log_inverse_size',
				error='pfer',
				q=q,
				how_binarize='intlp'
			)
			self.check_disjoint(detections)
			self.check_pfer_control(detections, v=q)

if __name__ == "__main__":
	unittest.main()
