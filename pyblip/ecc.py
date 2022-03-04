""" 
Polynomial-time function to find a (heuristically) minimal edge clique cover
"""

import numpy as np
import networkx as nx

def edge_clique_cover(G):
    """
    Parameters
    ----------
    G : networkx Graph
        Undirected graph. Should not have self-loops.
    
    Returns
    -------
    cliques : list
        List of cliques. Each clique is a list of ints where
        the integers correspond to the nodes of G.
    """
    am = nx.adjacency_matrix(G) # original adj.matrix
    am2 = am.copy() # new adj. matrix after deleting edges
    nodes = list(G.nodes)
    n = len(G.nodes)
    degrees = np.array([G.degree(node) for node in nodes])
    cliques = []
    for e in G.edges:
        (v1, v2) = e
        if am2[v1, v2] == 0:
            continue
        else:
            am2[v1, v2] = 0
            am2[v2, v1] = 0
            degrees[v1] -= 1
            degrees[v2] -= 1
            
        c = list(e) # the clique to add
        neighbors = set(G.neighbors(nodes[v1])) # common neighbors of c
        neighbors = neighbors.intersection(set(G.neighbors(nodes[v2])))
        if np.any(degrees[list(neighbors)] > 0):
            while len(neighbors) > 0:
                # Find the neighbor with the highest degree in the updated graph
                ln = sorted(list(neighbors))
                vnew = ln[np.argmax(degrees[ln])]
                # Update am2 and degrees
                for vold in c:
                    if am2[vnew, vold] != 0 or am2[vold, vnew] != 0:
                        am2[vnew, vold] = 0
                        am2[vold, vnew] = 0
                        degrees[vold] -= 1
                        degrees[vnew] -=1
                # Add vnew to c and update neighbors
                neighbors = neighbors.intersection(set(G.neighbors(nodes[vnew])))
                c.append(vnew)
        cliques.append(set(c))
    return cliques