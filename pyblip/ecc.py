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
        Undirected graph.
    
    Returns
    -------
    cliques : list
        List of cliques. Each clique is a list of ints where
        the integers correspond to the nodes of G.
    """ 
    # Add self-edges
    G.add_edges_from(nx.selfloop_edges(G))

    G2 = G.copy() # new Graph after deleting edges
    nodes = list(G.nodes)
    n = len(G.nodes)
    degrees = np.array([G.degree(node) for node in nodes])
    cliques = []
    for e in G.edges:
        (v1, v2) = e
        if not G2.has_edge(v1, v2):
            continue
        else:
            G2.remove_edge(v1, v2)
            degrees[v1] -= 1
            degrees[v2] -= 1
            for v in [v1, v2]:
                if G2.has_edge(v, v):
                    G2.remove_edge(v, v)
                    degrees[v] -= 1
            
        c = list(e) # the clique to add
        neighbors = set(G.neighbors(nodes[v1])) # common neighbors of c
        neighbors = neighbors.intersection(set(G.neighbors(nodes[v2])))
        neighbors = neighbors - set([v1, v2])
        if np.any(degrees[list(neighbors)] > 0):
            while len(neighbors) > 0:
                # Find the neighbor with the highest degree in the updated graph
                ln = sorted(list(neighbors))
                vnew = ln[np.argmax(degrees[ln])]
                c.append(vnew)
                # Update am2 and degrees
                for vold in c:
                    if G2.has_edge(vold, vnew):
                        G2.remove_edge(vold, vnew)
                        degrees[vold] -= 1
                        if vnew != vold:
                            degrees[vnew] -=1

                # Add vnew to c and update neighbors
                neighbors = neighbors.intersection(set(G.neighbors(nodes[vnew])))
                neighbors = neighbors - set([vnew])
        cliques.append(set(c))

    return cliques