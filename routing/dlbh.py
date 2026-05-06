"""
DLBH – Demand-Aware Load Balancing Heuristic benchmark.

Distributes traffic based on instantaneous queue loads but does NOT
consider future link availability or topology evolution.
Reference: [14] in the paper (Section V-A).
"""

import networkx as nx
import numpy as np
from typing import List, Optional, Dict

from environment.constellation import C_LIGHT
import config as cfg


class DLBH:
    """
    Demand-Aware Load Balancing Heuristic.

    Edge weight combines:
      w(u,v) = α · prop_delay(u,v) + (1-α) · queue_load(v)

    where queue_load(v) = Q_v / Q_max.
    """

    def __init__(self, n_sats: int, alpha: float = 0.5):
        self.n_sats = n_sats
        self.alpha  = alpha

    def route(
        self,
        adj_avail,        # (n, n) bool ndarray
        adj_dist,         # (n, n) float ndarray [m]
        queues,           # (n,)   float ndarray [pkts]
        sessions,         # list of Session objects
    ) -> Dict[int, Optional[List[int]]]:
        """
        Compute paths for all sessions.

        Returns
        -------
        paths : dict  session_id → list of node IDs (or None)
        """
        G = self._build_graph(adj_avail, adj_dist, queues)
        paths = {}
        for sess in sessions:
            try:
                p = nx.shortest_path(G, sess.src, sess.dst, weight="weight")
                if len(p) - 1 > cfg.HOP_LIMIT:
                    paths[sess.sid] = None
                else:
                    paths[sess.sid] = p
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                paths[sess.sid] = None
        return paths

    def _build_graph(self, adj_avail, adj_dist, queues) -> nx.Graph:
        G = nx.Graph()
        n = self.n_sats
        max_d = float(adj_dist.max()) if adj_dist.max() > 0 else 1.0
        max_q = float(cfg.BUFFER_SIZE_PKTS)

        for u in range(n):
            G.add_node(u)
        for u in range(n):
            for v in range(u + 1, n):
                if not adj_avail[u, v]:
                    continue
                prop  = float(adj_dist[u, v]) / C_LIGHT
                # normalised queue load at the destination node v
                ql_v  = float(queues[v]) / max_q
                w     = self.alpha * prop + (1.0 - self.alpha) * ql_v
                G.add_edge(u, v, weight=w)
        return G
