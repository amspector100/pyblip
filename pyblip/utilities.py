import time
import numpy as np

# Multiprocessing tools
from multiprocessing import Pool
from functools import partial

def elapsed(time0):
    return np.around(time.time() - time0, 2)

def min_eigval(cov):
    """
    eigsh is faster for super high dimensions,
    but often fails to converge
    """
    return np.linalg.eigh(cov)[0].min()

def shift_until_PSD(
    cov, 
    tol, 
    n_iter=8,
    init_gamma=None,
    conv_tol=1 / (2**10),
):
    """
    Finds the minimum value gamma such that 
    cov*gamma + (1 - gamma) * I is PSD.
    """
    p = cov.shape[0]
    if p < 7500:
        mineig = min_eigval(cov)
        if mineig < tol:
            gamma = (tol - mineig) / (1 - mineig)
        else:
            gamma = 0
        return cov * (1-gamma) + gamma * np.eye(p)
    else:
        ugamma = 0.2 # min gamma controlling eig bound, if > 0.3 this is really bad
        lgamma = 0 # max gamma violating eig bound
        for j in range(n_iter):
            if init_gamma is not None and j == 0:
                gamma = init_gamma
            else:
                gamma = (ugamma + lgamma) / 2
            try:
                np.linalg.cholesky(cov * (1 - gamma) + (gamma - tol) * np.eye(p))
                ugamma = gamma
            except np.linalg.LinAlgError:
                lgamma = gamma
            if ugamma - lgamma < conv_tol:
                break
        return cov * (1 - ugamma) + ugamma * np.eye(p)

# Multiprocessing helpers
def _one_arg_function(list_of_inputs, args, func, kwargs):
	"""
	Globally-defined helper function for pickling in multiprocessing.
    
    Parameters
    ----------
	list_of_inputs: list
        List of inputs to a function
	args : list
        Names/args for those inputs
	func : function
        Function to apply
	kwargs : dict
        Other kwargs to pass to ``func``. 
	"""
	new_kwargs = {}
	for i, inp in enumerate(list_of_inputs):
		new_kwargs[args[i]] = inp
	return func(**new_kwargs, **kwargs)

def apply_pool(func, constant_inputs={}, num_processes=1, **kwargs):
	"""
	Spawns num_processes processes to apply func to many different arguments.
	This wraps the multiprocessing.pool object plus the functools partial function. 
	
	Parameters
	----------
	func : function
		An arbitrary function
	constant_inputs : dictionary
		A dictionary of arguments to func which do not change in each
		of the processes spawned, defaults to {}.
	num_processes : int
		The maximum number of processes spawned, defaults to 1.
	kwargs : dict
		Each key should correspond to an argument to func and should
		map to a list of different arguments.
	Returns
	-------
	outputs : list
		List of outputs for each input, in the order of the inputs.
	Examples
	--------
	If we are varying inputs 'a' and 'b', we might have
	``apply_pool(
		func=my_func, a=[1,3,5], b=[2,4,6]
	)``
	which would return ``[my_func(a=1, b=2), my_func(a=3,b=4), my_func(a=5,b=6)]``.
	"""

	# Construct input sequence
	args = sorted(kwargs.keys())
	num_inputs = len(kwargs[args[0]])
	for arg in args:
		if len(kwargs[arg]) != num_inputs:
			raise ValueError(f"Number of inputs differs for {args[0]} and {arg}")
	inputs = [[] for _ in range(num_inputs)]
	for arg in args:
		for j in range(num_inputs):
			inputs[j].append(kwargs[arg][j])

	# Construct partial function
	partial_func = partial(
		_one_arg_function, args=args, func=func, kwargs=constant_inputs,
	)

	# Don't use the pool object if num_processes=1
	num_processes = min(num_processes, len(inputs))
	if num_processes == 1:
		all_outputs = []
		for inp in inputs:
			all_outputs.append(partial_func(inp))
	else:
		with Pool(num_processes) as thepool:
			all_outputs = thepool.map(partial_func, inputs)

	return all_outputs