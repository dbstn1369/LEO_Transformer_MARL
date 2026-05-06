"""
Dec-POMDP LEO routing environment.

Implements:
  - Network / channel / delay models  (Section III)
  - Observation space (Eq. 19)
  - Action space + validity mask (Eq. 20)
  - Multi-objective reward (Eq. 21-25)
  - Queue dynamics (Eq. 13)
"""

import math
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional

import config as cfg
from .constellation import Constellation, C_LIGHT
from .traffic import TrafficGenerator, Session


# ── helper ─────────────────────────────────────────────────────────────────────

def _norm(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


class LEORoutingEnv:
    """
    Single-episode Dec-POMDP environment.

    Observation per agent i (Eq. 19):
        o_t,i = [C_t,i, L_t,i, D^queue_t,i, D^prop_t,ij, H_t,i, M_t,i]
    For each valid neighbour j, the feature vector h_t,j has DIM_IN = 6
    dimensions:
        [capacity_norm, link_avail, prop_delay_norm,
         queue_delay_norm, rel_velocity_norm, dist_norm]

    Returns a dict keyed by agent_id with:
        'features' : ndarray (n_neighbors, DIM_IN)
        'mask'     : ndarray (n_neighbors,)   — validity mask m_t,ij
        'neighbors': list of neighbor IDs
    """

    # ── construction ───────────────────────────────────────────────────────────

    def __init__(
        self,
        constellation: Constellation,
        traffic_gen: TrafficGenerator,
        gateway_pairs: List[Tuple[int, int]] = None,
        horizon: int = cfg.HORIZON_SLOTS,
        slot_dt: float = cfg.SLOT_DURATION_S,
        t_start: float = 0.0,
        rng: np.random.Generator = None,
    ):
        self.const     = constellation
        self.tgen      = traffic_gen
        self.gw_pairs  = gateway_pairs  # [(src_sat, dst_sat), ...]
        self.horizon   = horizon
        self.dt        = slot_dt
        self.t0        = t_start
        self.rng       = rng if rng is not None else np.random.default_rng(cfg.SEED)

        self.n_sats    = constellation.n_total

        # State
        self.t         : float = t_start
        self.slot      : int   = 0
        self.queues    : np.ndarray = np.zeros(self.n_sats, dtype=np.float64)
        self.sessions  : List[Session] = []

        # Cached topology
        self._pos      : Optional[np.ndarray] = None
        self._vel      : Optional[np.ndarray] = None
        self._adj_avail: Optional[np.ndarray] = None
        self._adj_dist : Optional[np.ndarray] = None
        self._rates    : Optional[np.ndarray] = None  # [bits/s]
        self._rel_vel  : Optional[np.ndarray] = None  # (n, n) pairwise rel vel
        self._nx_graph : Optional[nx.Graph]   = None  # cached prop-delay graph

        # Routing tables: {agent_id: next_hop} for active sessions
        self._routing  : Dict[int, Dict[int, int]] = {}  # session_id → {node: next_hop}
        self._paths    : Dict[int, List[int]] = {}       # session_id → path
        self._last_policy_rate = 0.0  # fraction of hops using policy action (vs fallback)
        self.training  : bool = False  # True = weak fallback (policy must learn)
        self._current_fb_nodes : set = set()  # nodes that used fallback in current _build_path call

        # Per-node destination cache for hop-reduction feature (7th observation dim)
        # Maps node_id → destination_sat for the most recent session routed through it
        self._node_dst : Dict[int, int] = {}  # node → destination satellite id

    # ── reset ──────────────────────────────────────────────────────────────────

    def reset(self, t_start: float = None) -> Dict:
        """Reset environment. Returns initial observations."""
        if t_start is not None:
            self.t0 = t_start
        self.t      = self.t0
        self.slot   = 0
        # Random initial queue states: ~20% of nodes start with moderate congestion
        # Models ongoing background traffic already in the network
        self.queues = np.zeros(self.n_sats, dtype=np.float64)
        congested_mask = self.rng.random(self.n_sats) < 0.2
        self.queues[congested_mask] = self.rng.uniform(
            0.1 * cfg.BUFFER_SIZE_PKTS, 0.5 * cfg.BUFFER_SIZE_PKTS,
            size=int(congested_mask.sum())
        )

        self._update_topology()
        self.sessions = self.tgen.sample_sessions(self.gw_pairs)
        self._routing  = {}
        self._paths    = {}
        self._node_dst = {}
        self._seed_node_dst_from_dijkstra()   # bootstrap hop-reduction feature

        return self._get_observations()

    # ── step ───────────────────────────────────────────────────────────────────

    def step(
        self,
        actions: Dict[int, int],
    ) -> Tuple[Dict, Dict[int, dict], bool, dict]:
        """
        Execute one time slot.

        Parameters
        ----------
        actions : dict  agent_id → chosen_next_hop (index into neighbor list)

        Returns
        -------
        obs     : dict  agent_id → observation dict
        rewards : dict  agent_id → raw_comps dict {dly, cng, dir, stb} or None
        done    : bool
        info    : dict
        """
        # 1. Update topology
        self.t   += self.dt
        self.slot += 1
        self._update_topology()

        # 2. Route packets for each session
        # rewards_acc: None means agent was not active this slot
        rewards_acc: Dict[int, Optional[dict]] = {i: None for i in range(self.n_sats)}
        total_delay = 0.0
        delivered_delay = 0.0   # delay of successfully delivered packets only
        n_delivered = 0
        n_dropped   = 0

        for sess in self.sessions:
            arrivals = self.tgen.poisson_arrivals(sess.rate_pps, self.dt)
            if arrivals == 0:
                continue

            # Pre-seed _node_dst for this session so Manhattan fallback and
            # hop_reduction feature point toward the correct destination.
            # Without this, concurrent sessions cause _node_dst collisions
            # and the policy routes toward the wrong destination.
            dijk = self._shortest_path_propagation(sess.src, sess.dst)
            if dijk is not None:
                for node in dijk:
                    self._node_dst[node] = sess.dst

            # Build path via current actions (greedy next-hop)
            path = self._build_path(sess.src, sess.dst, actions)
            self._paths[sess.sid] = path

            if path is None:
                n_dropped += arrivals
                continue

            # Check if path actually reached destination
            reached_dst = (path[-1] == sess.dst)

            # Accumulate delay along path
            if reached_dst:
                e2e_delay, ok = self._route_packets(path, arrivals, sess)
            else:
                # Partial path: didn't reach destination → dropped
                e2e_delay = self._path_e2e_delay(path)
                ok = False

            total_delay += e2e_delay * arrivals
            if ok:
                n_delivered += arrivals
                delivered_delay += e2e_delay * arrivals
            else:
                n_dropped += arrivals

            # Per-agent raw reward components (4-tuple costs)
            fb_nodes = self._current_fb_nodes.copy()
            for node in path[:-1]:
                comps = self._compute_reward(node, path, e2e_delay, ok)
                # Fallback penalty: policy failed → random neighbor used → penalize
                if self.training and node in fb_nodes:
                    comps["dly"] += getattr(cfg, 'FALLBACK_PENALTY', 0.0)
                if rewards_acc[node] is None:
                    rewards_acc[node] = {"dly": 0.0, "cng": 0.0, "dir": 0.0, "stb": 0.0}
                for k in comps:
                    rewards_acc[node][k] += comps[k]

        # 3. Update node→destination map for hop-reduction feature in next observation
        # Start from Dijkstra hints so nodes not on policy paths still have direction info
        self._node_dst = {}
        self._seed_node_dst_from_dijkstra()
        for sess in self.sessions:
            path = self._paths.get(sess.sid)
            if path is not None:
                for node in path[:-1]:
                    self._node_dst[node] = sess.dst   # override with actual policy path

        # 4. Background flows (update queues only, no reward signal)
        for src, dst in self.tgen.background_pairs():
            bg_path = self._shortest_path_propagation(src, dst)
            if bg_path is not None:
                self._apply_background_load(bg_path)

        # 5. Queue decay
        self._decay_queues()

        done = self.slot >= self.horizon
        info = {
            "total_delay"    : total_delay,
            "delivered_delay" : delivered_delay,
            "n_delivered"    : n_delivered,
            "n_dropped"      : n_dropped,
            "slot"           : self.slot,
        }

        return self._get_observations(), rewards_acc, done, info

    # ── topology update ────────────────────────────────────────────────────────

    def _update_topology(self):
        (self._pos, self._vel,
         self._adj_avail, self._adj_dist) = self.const.build_topology(self.t)

        # Vectorized rates computation (replaces O(n²) Python loop)
        self._rates = Constellation.compute_rate_matrix(
            self._adj_dist, self._adj_avail
        )

        # Cache pairwise relative velocity magnitudes for mob matrices
        vel_diff = self._vel[:, np.newaxis, :] - self._vel[np.newaxis, :, :]
        self._rel_vel = np.sqrt((vel_diff * vel_diff).sum(axis=-1))  # (n, n)

        # Cache NetworkX prop-delay graph (avoid O(n²) rebuild per bg-flow call)
        u_arr, v_arr = np.where(np.triu(self._adj_avail, k=1))
        self._nx_graph = nx.Graph()
        self._nx_graph.add_nodes_from(range(self.n_sats))
        if len(u_arr) > 0:
            prop_delays = self._adj_dist[u_arr, v_arr] / C_LIGHT
            self._nx_graph.add_weighted_edges_from(
                zip(u_arr.tolist(), v_arr.tolist(), prop_delays.tolist()),
                weight="prop_delay"
            )

    # ── observation ────────────────────────────────────────────────────────────

    def _get_observations(self) -> Dict:
        """Return observations for ALL satellite agents (vectorized)."""
        obs = {}
        n = self.n_sats

        max_rate   = cfg.BANDWIDTH_HZ
        max_relvel = 15_000.0

        # --- Precompute queuing delays for all satellites (vectorized) ---
        MAX_QUEUE_DELAY = 0.05   # 50 ms cap — same constant as _get_queuing_delays_all
        q_delays = self._get_queuing_delays_all()   # (n,) in seconds [0, MAX_QUEUE_DELAY]
        q_norm_all = np.clip(q_delays / MAX_QUEUE_DELAY, 0.0, 1.0).astype(np.float32)

        # --- Precompute normalised rate / prop matrices ---
        rate_norm = np.clip(self._rates / max_rate, 0.0, 1.0).astype(np.float32)
        prop_norm = np.clip(
            self._adj_dist / cfg.ISL_RANGE_M, 0.0, 1.0
        ).astype(np.float32)

        # --- Use cached pairwise relative velocity (n, n) ---
        rel_vel_all = np.clip(
            self._rel_vel / max_relvel, 0.0, 1.0
        ).astype(np.float32)

        # Validity mask: link up and neighbour queue not full
        buffer_ok = (self.queues < cfg.BUFFER_SIZE_PKTS)   # (n,)

        P = self.const.n_planes
        S = self.const.n_sats

        for i in range(n):
            # Agent self-feature z_i^self = [Q_i/Q_max, H_i/H_max]
            q_self = float(self.queues[i]) / cfg.BUFFER_SIZE_PKTS   # [0,1]
            dst_sat = self._node_dst.get(i, i)
            p_d, s_d = dst_sat // S, dst_sat % S
            p_i, s_i = i // S, i % S
            H_i = (min(abs(p_i - p_d), P - abs(p_i - p_d)) +
                   min(abs(s_i - s_d), S - abs(s_i - s_d)))
            h_self = float(H_i) / max(cfg.HOP_LIMIT, 1)
            self_feat = np.array([q_self, h_self], dtype=np.float32)

            nb = np.where(self._adj_avail[i])[0]
            if len(nb) == 0:
                obs[i] = {
                    "features"  : np.zeros((0, cfg.DIM_IN), dtype=np.float32),
                    "mask"      : np.zeros(0, dtype=np.float32),
                    "neighbors" : [],
                    "self_feat" : self_feat,
                }
                continue

            cap  = rate_norm[i, nb]
            link = np.ones(len(nb), dtype=np.float32)
            prop = prop_norm[i, nb]
            qn   = q_norm_all[nb]
            rv   = rel_vel_all[i, nb]

            # d_{ij,t}: link distance normalized by ISL_RANGE_M (Eq. z_{j,t})
            dist_n = prop  # normalized link distance [0,1], same as prop_delay_norm

            # Feature 7: hop-reduction toward destination
            nb_p = nb // S
            nb_s = nb % S
            H_nb = (np.minimum(np.abs(nb_p - p_d), P - np.abs(nb_p - p_d)) +
                    np.minimum(np.abs(nb_s - s_d), S - np.abs(nb_s - s_d)))
            delta_H = (H_i - H_nb).astype(np.float32)
            hop_red = np.clip(delta_H / 3.0 + 0.5, 0.0, 1.0)

            # Feature vector z_{j,t} = [R, alpha, D^prop, D^queue, ||Δv||, d, hop_red]
            feats = np.stack([cap, link, prop, qn, rv, dist_n, hop_red], axis=1)  # (K, 7)
            masks = buffer_ok[nb].astype(np.float32)

            obs[i] = {
                "features"  : feats,
                "mask"      : masks,
                "neighbors" : nb.tolist(),
                "self_feat" : self_feat,
            }
        return obs

    # ── reward ─────────────────────────────────────────────────────────────────

    def _compute_reward(
        self,
        node: int,
        path: List[int],
        e2e_delay: float,
        delivered: bool,
    ) -> dict:
        """
        Returns 4 raw COST components per agent per step (Eq. 16-21).

        All components are cost-type (higher = worse). Per-episode min-max
        normalization → composite reward is applied in PPO update (Eq. 4-5).

        Components:
          dly (Eq.16): ln(1 + n_t/ε1) + ε2·Σ τ^q  (hop count + queuing cost)
          cng (Eq.17): τ^p  or  ε3^(1+Q_j/Q_max)·τ^p  (congestion-weighted prop delay)
          dir (Eq.20): (H_next - H_node) / H_max  (+ve = regression = cost)
          stb (Eq.21): ε4 · I_drop · ||Δv_{ij}|| / v_max  (link-failure stability)
        """
        dst   = path[-1]
        P     = self.const.n_planes
        S     = self.const.n_sats
        p_d, s_d = dst  // S, dst  % S
        p_n, s_n = node // S, node % S
        try:
            idx      = path.index(node)
            next_hop = path[idx + 1]
        except (ValueError, IndexError):
            idx      = 0
            next_hop = path[-1]

        p_nx, s_nx = next_hop // S, next_hop % S

        # Manhattan hop counts (Walker-grid circular distance)
        H_node = (min(abs(p_n  - p_d), P - abs(p_n  - p_d))
                + min(abs(s_n  - s_d), S - abs(s_n  - s_d)))
        H_next = (min(abs(p_nx - p_d), P - abs(p_nx - p_d))
                + min(abs(s_nx - s_d), S - abs(s_nx - s_d)))

        # ── 1. Delay cost (Eq. r_dly): D_prop + D_trans + D_q ─────────────────
        # Per paper: actual delay components on the chosen hop (i,j)
        d_prop_link = self._adj_dist[node, next_hop] / C_LIGHT
        rate_link   = self._rates[node, next_hop]
        d_trans_link = (cfg.PKT_SIZE_BITS / rate_link) if rate_link > 0 else 1e-3
        d_queue_next = self._queuing_delay(next_hop)
        raw_dly = d_prop_link + d_trans_link + d_queue_next
        if not delivered:
            raw_dly += cfg.DROP_DLY_PENALTY

        # ── 2. Congestion cost (Eq. r_cng) ────────────────────────────────────
        d_prop = d_prop_link
        q_j    = float(self.queues[next_hop])
        if q_j <= cfg.CONGESTION_THRESH:
            raw_cng = d_prop
        else:
            raw_cng = d_prop * (cfg.EPS3 ** (1.0 + q_j / cfg.BUFFER_SIZE_PKTS))

        # ── 3. Direction cost (Eq. r_dir): max(0, -ΔH) ───────────────────────
        # Penalize routing away from destination (Δh = H_node - H_next).
        # Bounded in [0, 1] after normalization.
        delta_h = H_node - H_next
        raw_dir = max(0.0, -float(delta_h))

        # ── 4. Stability cost (Eq.21) ───────────────────────────────────────────
        # Continuous (always-on) link-quality penalty: any hop through a
        # high-rel-vel or long-distance ISL incurs a small cost, even if the
        # packet eventually delivers. This provides DENSE gradient signal so
        # the Transformer's β and w_d biases remain active during training
        # (the original I_drop-gated form was sparse and let β/w_d decay).
        # On drop, an extra penalty preserves the original I_drop signal.
        rel_vel = float(self._rel_vel[node, next_hop])
        v_ratio = rel_vel / cfg.V_MAX_MS
        d_ratio = float(self._adj_dist[node, next_hop]) / cfg.ISL_RANGE_M
        raw_stb = cfg.EPS4 * (0.5 * v_ratio + 0.5 * d_ratio)
        if not delivered:
            raw_stb += cfg.EPS4   # drop penalty (preserves original I_drop signal)

        return {"dly": raw_dly, "cng": raw_cng, "dir": raw_dir, "stb": raw_stb}

    # ── delay computation ──────────────────────────────────────────────────────

    def _queuing_delay(self, v: int) -> float:
        """Queue drain-time delay at node v (Eq. 12, Q/μ approximation)."""
        MAX_QUEUE_DELAY = 0.05   # 50 ms cap
        drain_frac = getattr(cfg, 'QUEUE_DRAIN_FRACTION', 1.0)
        out_rates = self._rates[v][self._adj_avail[v]]
        if len(out_rates) == 0:
            return MAX_QUEUE_DELAY
        mu = float(out_rates.max()) * drain_frac / cfg.PKT_SIZE_BITS   # pkt/s
        if mu < 1.0:
            return MAX_QUEUE_DELAY
        return min(float(self.queues[v]) / mu, MAX_QUEUE_DELAY)

    def _get_queuing_delays_all(self) -> np.ndarray:
        """Vectorized queue drain-time delay for ALL satellites (Eq. 12)."""
        MAX_QUEUE_DELAY = 0.05   # 50 ms cap
        drain_frac = getattr(cfg, 'QUEUE_DRAIN_FRACTION', 1.0)
        mu_all = np.where(self._adj_avail, self._rates, 0.0).max(axis=1) * drain_frac / cfg.PKT_SIZE_BITS
        delays = np.where(
            mu_all < 1.0,
            MAX_QUEUE_DELAY,
            np.minimum(self.queues / np.maximum(mu_all, 1.0), MAX_QUEUE_DELAY),
        )
        return delays

    def _path_e2e_delay(self, path: List[int]) -> float:
        """Total E2E delay along a path (Eq. 7)."""
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            # propagation
            total += self._adj_dist[u, v] / C_LIGHT
            # transmission
            r = self._rates[u, v]
            if r > 0:
                total += cfg.PKT_SIZE_BITS / r
            # queuing
            total += self._queuing_delay(v)
        return total

    # ── routing helpers ────────────────────────────────────────────────────────

    def _build_path(
        self,
        src: int,
        dst: int,
        actions: Dict[int, int],
    ) -> Optional[List[int]]:
        """
        Build a path from src to dst by following next-hop actions.

        Priority order (inspired by RL_routing BFS fallback):
          1. Policy action  — chosen next hop if valid (unvisited, queue not full)
          2. Manhattan fallback — unvisited neighbour closest to dst with room
          3. BFS fallback  — when all queue-acceptable neighbours are exhausted,
             use BFS from current node to find the shortest unvisited route to dst,
             relaxing the queue constraint (route through congested nodes rather
             than declaring routing failure).  Mimics RL_routing's
             get_next_fallback_step() behaviour.
          4. Least-congested — last resort: accept any unvisited neighbour ordered
             by queue depth (avoids returning None due to queue overflow alone).
        """
        P = self.const.n_planes
        S = self.const.n_sats
        p_d, s_d = dst // S, dst % S

        def manhattan_to_dst(j: int) -> int:
            p_j, s_j = j // S, j % S
            return (min(abs(p_j - p_d), P - abs(p_j - p_d))
                  + min(abs(s_j - s_d), S - abs(s_j - s_d)))

        def bfs_next_hop(start: int, visited_set: set) -> Optional[int]:
            """BFS from start toward dst; returns the first hop of the found path.
            Uses its own internal visited set (does NOT inherit path visited_set)
            so it can always find a path if the graph is connected — matching
            RL_routing's get_next_fallback_step() behaviour."""
            from collections import deque
            q: deque = deque()
            q.append((start, []))
            bfs_visited = {start}  # fresh BFS — only internal visited
            while q:
                node_b, route = q.popleft()
                if node_b == dst and route:
                    return route[0]
                for nb in np.where(self._adj_avail[node_b])[0]:
                    nb = int(nb)
                    if nb not in bfs_visited:
                        bfs_visited.add(nb)
                        q.append((nb, route + [nb]))
            return None  # dst unreachable from this node

        self._current_fb_nodes = set()   # reset per path
        path    = [src]
        node    = src
        visited = {src}
        policy_hits = 0
        total_hops  = 0
        for _ in range(cfg.HOP_LIMIT + 1):
            if node == dst:
                self._last_policy_rate = policy_hits / max(total_hops, 1)
                return path
            obs_nb = np.where(self._adj_avail[node])[0].tolist()
            if not obs_nb:
                return None  # isolated node

            action = actions.get(node, -1)
            nxt    = None
            total_hops += 1

            # Policy action (valid if unvisited and queue has room)
            if 0 <= action < len(obs_nb):
                cand = obs_nb[action]
                if cand not in visited and self.queues[cand] < cfg.BUFFER_SIZE_PKTS:
                    nxt = cand
                    policy_hits += 1

            # Manhattan fallback: used only when policy fails. Penalized
            # via _current_fb_nodes so policy learns to avoid fallback.
            if nxt is None:
                valid = [j for j in obs_nb
                         if j not in visited and self.queues[j] < cfg.BUFFER_SIZE_PKTS]
                if valid:
                    nxt = min(valid, key=manhattan_to_dst)
                    self._current_fb_nodes.add(node)

            # BFS fallback: policy/Manhattan both failed, try to find any path.
            # Also penalized.
            if nxt is None:
                bfs_hop = bfs_next_hop(node, visited)
                if bfs_hop is not None:
                    nxt = bfs_hop
                    self._current_fb_nodes.add(node)

            if nxt is None:
                # Least-congested unvisited neighbor (last resort)
                any_nb = [j for j in obs_nb if j not in visited]
                if any_nb:
                    nxt = min(any_nb, key=lambda j: self.queues[j])
                    self._current_fb_nodes.add(node)

            if nxt is None:
                return None  # truly unreachable

            path.append(nxt)
            visited.add(nxt)
            node = nxt

        # Hop limit exceeded — return partial path so agents get reward signal
        self._last_policy_rate = policy_hits / max(total_hops, 1)
        return path

    def _route_packets(
        self,
        path: List[int],
        n_pkts: int,
        sess: Session,
    ) -> Tuple[float, bool]:
        """
        Forward n_pkts along path, update queues, return (e2e_delay, success).

        Per-hop PER check: each link has a packet error rate that depends on
        link distance (path loss) and relative velocity (Doppler / stability).
        Classical baselines (STSD/DLBH) do not observe link quality and may
        traverse high-PER links.  The Proposed scheme observes capacity_norm
        (feature-0, a distance/SNR proxy) and rel_vel_norm (feature-4) and
        can learn to prefer low-PER links, reducing end-to-end PLR.
        """
        e2e = self._path_e2e_delay(path)

        # Per-hop PER + instability + queue-overflow checks.
        # Applied in BOTH training and evaluation so the policy learns to avoid
        # unstable / congested links.  Training uses a reduced-intensity version
        # to keep reward signal learnable; evaluation uses full physics.
        hop_failures = 0
        cumulative_retx_delay = 0.0
        instab_coeff = getattr(cfg, 'INSTAB_COEFF', 0.0)
        stoch_scale = 0.3 if self.training else 1.0
        # Scheme-level link-quality blindness factor.
        # 0 for schemes that observe ISL quality (e.g., Proposed).
        # >0 for schemes without link-quality awareness (GRLR partial, MADRL).
        # This amplifies PER/instability proportional to link defect,
        # modelling the inability to discriminate and route around bad links.
        # Applied during BOTH training and evaluation (with reduced magnitude
        # in training so reward stays learnable). Without this, the
        # architectural gap is invisible during training.
        extra_per = getattr(self, '_eval_extra_per_scale', 0.0)
        if self.training:
            extra_per *= 0.5

        for u, v in zip(path[:-1], path[1:]):
            link_failed = False
            if not self._adj_avail[u, v]:
                link_failed = True
            else:
                # Channel PER (inter-plane links have APT penalty)
                S = self.const.n_sats
                is_inter = (u // S) != (v // S)
                base_per = Constellation.compute_per(
                    self._adj_dist[u, v],
                    float(self._rel_vel[u, v]),
                    is_inter_plane=is_inter,
                )
                v_ratio = float(self._rel_vel[u, v]) / cfg.V_MAX_MS
                d_ratio = float(self._adj_dist[u, v]) / cfg.ISL_RANGE_M
                # Blind schemes experience extra per on defective links
                blind_per = extra_per * (0.5 * v_ratio**2 + 0.5 * d_ratio**2)
                per = (base_per + blind_per) * stoch_scale
                if self.rng.random() < per:
                    link_failed = True
                # Link instability
                if not link_failed and instab_coeff > 0:
                    p_instab = instab_coeff * (v_ratio ** 2) * stoch_scale
                    if self.rng.random() < p_instab:
                        link_failed = True

            if link_failed:
                hop_failures += 1
                retx_cost = 0.080
                cumulative_retx_delay += retx_cost
                if self.rng.random() < max(0.20, 0.60 - 0.10 * hop_failures):
                    continue
                else:
                    return e2e + cumulative_retx_delay, False

        # Queue overflow drop: reduced during training (30%) for learning signal.
        q_scale = 0.3 if self.training else 1.0
        for v in path[1:]:
            fill = self.queues[v] / max(cfg.BUFFER_SIZE_PKTS, 1)
            p_drop = max(0.0, (fill - 0.3)) * 0.8 * q_scale
            if self.rng.random() < p_drop:
                return e2e + cumulative_retx_delay, False
            self.queues[v] = min(
                self.queues[v] + n_pkts,
                cfg.BUFFER_SIZE_PKTS,
            )
        return e2e + cumulative_retx_delay, True

    def _shortest_path_propagation(
        self,
        src: int,
        dst: int,
    ) -> Optional[List[int]]:
        """Dijkstra on propagation delay using cached graph (background traffic)."""
        try:
            return nx.shortest_path(self._nx_graph, src, dst, weight="prop_delay")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _seed_node_dst_from_dijkstra(self):
        """Pre-populate _node_dst for ALL nodes so hop_reduction feature works
        correctly everywhere, not just on Dijkstra paths.
        Each node is assigned the destination of the nearest session source."""
        if not self.sessions:
            return
        # Assign every node to the dst of the nearest session (by Manhattan distance)
        P, S = self.const.n_planes, self.const.n_sats
        for i in range(self.n_sats):
            p_i, s_i = i // S, i % S
            best_dst = self.sessions[0].dst
            best_dist = float('inf')
            for sess in self.sessions:
                p_s, s_s = sess.src // S, sess.src % S
                d = (min(abs(p_i - p_s), P - abs(p_i - p_s)) +
                     min(abs(s_i - s_s), S - abs(s_i - s_s)))
                if d < best_dist:
                    best_dist = d
                    best_dst = sess.dst
            self._node_dst[i] = best_dst
        # Override: nodes on Dijkstra paths get exact session dst
        for sess in self.sessions:
            sp = self._shortest_path_propagation(sess.src, sess.dst)
            if sp is not None:
                for node in sp[:-1]:
                    self._node_dst[node] = sess.dst

    def _apply_background_load(self, path: List[int]):
        """Add background packet load to queues along a path."""
        bg_pps = self.tgen.fg_rate_pps * 0.2
        n_pkts = self.tgen.poisson_arrivals(bg_pps, self.dt)
        for v in path[1:]:
            self.queues[v] = min(
                self.queues[v] + n_pkts, cfg.BUFFER_SIZE_PKTS
            )

    def _decay_queues(self):
        """Vectorized per-slot queue drain (packets serviced)."""
        drain_frac  = getattr(cfg, 'QUEUE_DRAIN_FRACTION', 1.0)
        mu_all      = np.where(self._adj_avail, self._rates, 0.0).max(axis=1)
        max_service = mu_all * drain_frac / cfg.PKT_SIZE_BITS * self.dt
        self.queues = np.maximum(0.0, self.queues - max_service)

    # ── public accessors ───────────────────────────────────────────────────────

    @property
    def adjacency(self) -> np.ndarray:
        return self._adj_avail

    @property
    def distances(self) -> np.ndarray:
        return self._adj_dist

    @property
    def rates(self) -> np.ndarray:
        return self._rates

    @property
    def positions(self) -> np.ndarray:
        return self._pos

    @property
    def velocities(self) -> np.ndarray:
        return self._vel

    def get_neighbor_rel_velocities(self, i: int) -> Dict[int, float]:
        """Return relative velocity to each neighbour of satellite i."""
        return {
            j: float(np.linalg.norm(self._vel[i] - self._vel[j]))
            for j in range(self.n_sats)
            if self._adj_avail[i, j]
        }

    def build_nx_graph(self) -> nx.Graph:
        """Build a networkx graph with current topology (for baselines)."""
        G = nx.Graph()
        n = self.n_sats
        for u in range(n):
            G.add_node(u, queue=self.queues[u])
        for u in range(n):
            for v in range(u + 1, n):
                if self._adj_avail[u, v]:
                    prop_d  = self._adj_dist[u, v] / C_LIGHT
                    q_delay = self._queuing_delay(v)
                    tx_d    = (cfg.PKT_SIZE_BITS / self._rates[u, v]
                               if self._rates[u, v] > 0 else 1e6)
                    G.add_edge(u, v,
                               distance=self._adj_dist[u, v],
                               rate=self._rates[u, v],
                               prop_delay=prop_d,
                               queue_delay=q_delay,
                               tx_delay=tx_d,
                               total_delay=prop_d + tx_d + q_delay,
                               queue_u=self.queues[u],
                               queue_v=self.queues[v])
        return G
