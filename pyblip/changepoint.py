import numpy as np
from . import create_groups, blip

def changepoint_cand_groups(model, **kwargs):
	# Create inclusions
	inclusions = model.betas != 0
	inclusions[:, 0] = 0 # don't want to discovery time = 0
	# Sequential cand groups
	return create_groups.sequential_groups(
		inclusions=inclusions, **kwargs
	)

def detect_changepoints(Y, q=0.1, lm_kwargs={}, sample_kwargs={}, blip_kwargs={}):
	"""
	Changepoint detection with BLiP and LSS gibbs sampler.
	
	Parameters
	----------
	Y : np.array
		Array of observations in order they were observed.
	q : float
		Level at which to control the FDR.
	**kwargs : dict
		Optional inputs to linear spike slab model.
	**sample_kwargs : dict
		Optional inputs to ``sample" method of 
		linear spike slab model.
	**blip_kwargs : dict
		Optional inputs to BLiP.
	"""
	T = Y.shape[0]
	# Dummy X for regression
	X = np.ones((T, T))
	for j in range(T):
		X[0:j, j] = 0
	# Create model
	lm = pyblip.linear.LinearSpikeSlab(
		X=X, y=Y, **lm_kwargs
	)
	lm.sample(**sample_kwargs)
	# Create inclusions
	cand_groups = changepoint_cand_groups(lm)
	# Run BLiP
	return blip.BLiP(
		cand_groups=cand_groups, q=q, **blip_kwargs
	)