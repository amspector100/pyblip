import time
import numpy as np

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