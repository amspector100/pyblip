import time
import networkx as nx
import numpy as np
import scipy as sp
from scipy import stats
import unittest
import pytest
from . import context
from .context import pyblip
from pyblip import ecc

class CheckEdgeCliqueCover(unittest.TestCase):
    """
    Helper methods to make sure the ECC output is a valid
    Edge Clique Cover.
    """
    def check_edge_clique_cover(self, am, cliques):
        """
        This method checks if the cliques in cliques are a valid
        edge clique cover of the graph represented by the adjacency
        matrix am.

        Parameters
        ----------
        am : np.narray
            Adjacency matrix of the graph. Must be a dense 
            matrix.
        cliques : list
            List of list of proposed edge clique cover.
        """
        am2 = am.copy()
        for j in range(am.shape[0]):
            am2[j, j] = 0
        for c in cliques:
            lc = list(c)
            lc_inds = np.ix_(lc, lc)
            am2[lc_inds] = 0
            self.assertTrue(
                np.all(am[lc][:, lc]),
                f"Clique {c} is not a valid clique."
            )
        
        xs, ys = np.where(am2 != 0)
        uncovered_edges = [xy for xy in zip(xs, ys)]
        self.assertTrue(
            np.all(am2 == 0),
            f"The cliques {cliques} are not a valid edge cover. Uncovered edges={uncovered_edges}"
        )
        
        

class TestECCSolver(CheckEdgeCliqueCover):

    def test_ecc(self):
        """
        Test the ECC solver.
        """
        G1 = nx.karate_club_graph() # simple networkx example

        # Example more relevant to us (local edges)
        thresh = 0.5
        p = 500
        X, _, _ = context.generate_regression_data(
            p=p, n=int(2*p)
        )
        V = np.corrcoef(X.T)
        flags = V >= thresh
        for j in range(p):
            flags[j, j] = 0
        G2 = nx.Graph(flags)

        # For each example, test that ECC gives the correct solution
        for G in [G1, G2]:
            cliques = ecc.edge_clique_cover(G)
            am = nx.adjacency_matrix(G).todense()
            for j in range(len(list(G.nodes))):
                am[j, j] = True
            self.check_edge_clique_cover(am, cliques)

            # Check that this is a good solution
            nmc = len(list(nx.find_cliques(G)))
            self.assertTrue(
                len(cliques) <= nmc,
                f"The clique edge cover has {len(cliques)} cliques, which is greater than the number of maximal cliques in the graph {nmc}."
            )


if __name__ == "__main__":
	unittest.main()
