"""
Poisson traffic generator for LEO satellite network sessions.
Section III-A: each session k is (src, dst, lambda_k, L_p).
"""

import math
import numpy as np
from typing import List, Tuple, Dict

import config as cfg


class Session:
    """One traffic session (source → destination)."""

    def __init__(self, sid: int, src: int, dst: int, rate_pps: float):
        self.sid     = sid
        self.src     = src
        self.dst     = dst
        self.rate_pps = rate_pps   # λ_k [packets/s]


class TrafficGenerator:
    """
    Generates a set of foreground Poisson video sessions and
    background flows.
    """

    def __init__(
        self,
        n_sats: int,
        n_fg_sessions: int = cfg.N_GROUND_PAIRS,
        fg_rate_mbps: float = cfg.VIDEO_RATE_MBPS,
        n_bg_flows: int = cfg.N_BG_FLOWS,
        rng: np.random.Generator = None,
    ):
        self.n_sats       = n_sats
        self.n_fg         = n_fg_sessions
        self.fg_rate_pps  = fg_rate_mbps * 1e6 / cfg.PKT_SIZE_BITS
        self.n_bg         = n_bg_flows
        self.rng          = rng if rng is not None else np.random.default_rng(cfg.SEED)

    def sample_sessions(
        self,
        gateway_pairs: List[Tuple[int, int]] = None,
    ) -> List[Session]:
        """
        Sample foreground sessions.  If gateway_pairs is provided, use them;
        otherwise pick random satellite pairs.
        """
        sessions = []
        if gateway_pairs is not None:
            pairs = [(s, d) for s, d in gateway_pairs if s >= 0 and d >= 0]
        else:
            nodes = self.rng.choice(self.n_sats, size=(self.n_fg, 2), replace=True)
            pairs = [(int(r[0]), int(r[1])) for r in nodes if r[0] != r[1]]

        for i, (src, dst) in enumerate(pairs[: self.n_fg]):
            sessions.append(Session(i, src, dst, self.fg_rate_pps))
        return sessions

    def poisson_arrivals(self, rate_pps: float, dt: float) -> int:
        """Sample Poisson arrivals in one slot of length dt [s]."""
        lam = rate_pps * dt
        return int(self.rng.poisson(lam))

    def background_pairs(self) -> List[Tuple[int, int]]:
        """Random source-destination pairs for background flows."""
        pairs = []
        nodes = self.rng.choice(self.n_sats, size=(self.n_bg, 2), replace=True)
        for r in nodes:
            if r[0] != r[1]:
                pairs.append((int(r[0]), int(r[1])))
        return pairs
