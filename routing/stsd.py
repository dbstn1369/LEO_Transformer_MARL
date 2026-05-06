"""
STSD – Static Topology Shortest Delay benchmark.

Selects paths based solely on propagation latency using a snapshot
of the current topology (no queuing delay, no dynamic updates).
Reference: [20] in the paper (Section V-A).
"""

import networkx as nx
from typing import List, Optional, Dict

from environment.constellation import C_LIGHT
import config as cfg


class STSD:
    """
    Static Topology Shortest-Delay routing.

    At each time slot, builds a graph with propagation-delay edge
    weights and runs Dijkstra.  Does NOT consider:
      – queuing delay
      – link congestion / buffer occupancy
      – future link availability
    """

    def __init__(self, n_sats: int):
        self.n_sats = n_sats

    def route(
        self,
        adj_avail,         # (n, n) bool ndarray
        adj_dist,          # (n, n) float ndarray  [m]
        sessions,          # list of Session objects
    ) -> Dict[int, Optional[List[int]]]:
        """
        Compute paths for all sessions.

        Returns
        -------
        paths : dict  session_id → list of node IDs (or None)
        """
        G = self._build_graph(adj_avail, adj_dist)
        paths = {}
        for sess in sessions:
            try:
                p = nx.shortest_path(G, sess.src, sess.dst, weight="prop_delay")
                if len(p) - 1 > cfg.HOP_LIMIT:
                    paths[sess.sid] = None
                else:
                    paths[sess.sid] = p
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                paths[sess.sid] = None
        return paths

    def _build_graph(self, adj_avail, adj_dist) -> nx.Graph:
        G = nx.Graph()
        n = self.n_sats
        for u in range(n):
            G.add_node(u)
        for u in range(n):
            for v in range(u + 1, n):
                if adj_avail[u, v]:
                    prop = float(adj_dist[u, v]) / C_LIGHT
                    G.add_edge(u, v, prop_delay=prop)
        return G
