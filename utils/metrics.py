"""
Metrics tracker for comparative evaluation.

Tracks per-episode / per-slot:
  - Average E2E delay
  - Throughput (Mbps)
  - Packet delivery ratio (PDR)
  - Path stability (switching frequency)
  - Jitter (delay variance)
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class EpisodeStats:
    scheme_name   : str
    delays        : List[float] = field(default_factory=list)   # E2E delay per pkt [s]
    throughputs   : List[float] = field(default_factory=list)   # Mbps per slot
    delivered     : int = 0
    dropped       : int = 0
    path_switches : int = 0     # number of route changes
    n_slots       : int = 0

    def record_slot(
        self,
        delay_s      : float,
        n_delivered  : int,
        n_dropped    : int,
        throughput_mbps: float,
        path_switched: bool = False,
    ):
        if delay_s > 0:
            self.delays.append(delay_s)
        self.throughputs.append(throughput_mbps)
        self.delivered    += n_delivered
        self.dropped      += n_dropped
        self.path_switches += int(path_switched)
        self.n_slots      += 1

    @property
    def mean_delay_ms(self) -> float:
        return float(np.mean(self.delays)) * 1e3 if self.delays else 0.0

    @property
    def mean_throughput_mbps(self) -> float:
        return float(np.mean(self.throughputs)) if self.throughputs else 0.0

    @property
    def pdr(self) -> float:
        total = self.delivered + self.dropped
        return self.delivered / max(total, 1)

    @property
    def jitter_ms(self) -> float:
        if len(self.delays) < 2:
            return 0.0
        return float(np.std(self.delays)) * 1e3

    @property
    def switch_rate(self) -> float:
        return self.path_switches / max(self.n_slots, 1)


class MetricsTracker:
    """Aggregate metrics across multiple episodes for multiple schemes."""

    def __init__(self, scheme_names: List[str]):
        self.scheme_names = scheme_names
        # scheme → list of EpisodeStats
        self.history: Dict[str, List[EpisodeStats]] = {s: [] for s in scheme_names}
        self._current: Dict[str, Optional[EpisodeStats]] = {s: None for s in scheme_names}

    def start_episode(self, scheme: str):
        self._current[scheme] = EpisodeStats(scheme_name=scheme)

    def record(
        self,
        scheme       : str,
        delay_s      : float,
        n_delivered  : int,
        n_dropped    : int,
        throughput_mbps: float,
        path_switched: bool = False,
    ):
        if self._current[scheme] is None:
            self.start_episode(scheme)
        self._current[scheme].record_slot(
            delay_s, n_delivered, n_dropped, throughput_mbps, path_switched
        )

    def end_episode(self, scheme: str):
        if self._current[scheme] is not None:
            self.history[scheme].append(self._current[scheme])
            self._current[scheme] = None

    def summary(self) -> Dict[str, Dict]:
        out = {}
        for scheme in self.scheme_names:
            eps = self.history[scheme]
            if not eps:
                continue
            out[scheme] = {
                "mean_delay_ms"      : np.mean([e.mean_delay_ms  for e in eps]),
                "std_delay_ms"       : np.std( [e.mean_delay_ms  for e in eps]),
                "mean_throughput_mbps": np.mean([e.mean_throughput_mbps for e in eps]),
                "std_throughput_mbps" : np.std( [e.mean_throughput_mbps for e in eps]),
                "mean_pdr"           : np.mean([e.pdr            for e in eps]),
                "mean_jitter_ms"     : np.mean([e.jitter_ms      for e in eps]),
                "mean_switch_rate"   : np.mean([e.switch_rate    for e in eps]),
            }
        return out

    def delay_series(self, scheme: str) -> np.ndarray:
        """Per-slot mean delay across all episodes."""
        all_delays = [e.delays for e in self.history[scheme] if e.delays]
        if not all_delays:
            return np.array([])
        min_len = min(len(d) for d in all_delays)
        return np.mean([d[:min_len] for d in all_delays], axis=0)

    def throughput_series(self, scheme: str) -> np.ndarray:
        """Per-slot throughput across all episodes."""
        all_tp = [e.throughputs for e in self.history[scheme] if e.throughputs]
        if not all_tp:
            return np.array([])
        min_len = min(len(t) for t in all_tp)
        return np.mean([t[:min_len] for t in all_tp], axis=0)
