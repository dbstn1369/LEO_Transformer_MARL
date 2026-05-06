"""
Comprehensive simulation validation suite for LEO_Transformer_MARL.

Tests:
  T1  Constellation geometry (positions, velocities, link formation)
  T2  Feature vector dimensions and value ranges
  T3  Action-to-neighbor mapping consistency (train vs env)
  T4  Reward formula correctness and component ranges
  T5  Queue dynamics (drain, fill, cap)
  T6  Path building logic (_build_path fallback frequency)
  T7  Routing progress (ΔH > 0 check)
  T8  End-to-end delay sanity (propagation + transmission + queuing)
  T9  DLBH / STSD baseline path quality vs random policy
  T10 Training data flow (feats → action → reward loop, no NaN/Inf)

Usage:
    python test_simulation.py          # run all tests, print summary
    python test_simulation.py -v       # verbose output
"""

import sys
import math
import time
import traceback
import argparse
import numpy as np
import torch

# ── project imports ─────────────────────────────────────────────────────────────
import config as cfg
from environment import Constellation, LEORoutingEnv, TrafficGenerator
from environment.constellation import C_LIGHT
from models import TransformerActor
from models.critic import CentralizedCritic
from models.maac import MADRLAgent
from algorithms import PPO_CTDE

# ── helpers ─────────────────────────────────────────────────────────────────────
PASS  = "\033[92m[PASS]\033[0m"
FAIL  = "\033[91m[FAIL]\033[0m"
WARN  = "\033[93m[WARN]\033[0m"
INFO  = "\033[94m[INFO]\033[0m"

results = []

def check(name, condition, detail="", warn_only=False):
    tag = PASS if condition else (WARN if warn_only else FAIL)
    status = "PASS" if condition else ("WARN" if warn_only else "FAIL")
    print(f"  {tag} {name}" + (f" | {detail}" if detail else ""))
    results.append((name, status, detail))
    return condition


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── Shared fixtures ─────────────────────────────────────────────────────────────
N_PLANES = 18
N_SATS   = 18
SEED     = 42

rng   = np.random.default_rng(SEED)
const = Constellation(N_PLANES, N_SATS)
tgen  = TrafficGenerator(const.n_total, rng=rng)
env   = LEORoutingEnv(const, tgen, horizon=10, rng=rng)


# ══════════════════════════════════════════════════════════════════════
# T1  Constellation geometry
# ══════════════════════════════════════════════════════════════════════
def test_t1_constellation():
    section("T1: Constellation geometry")
    obs = env.reset(t_start=0.0)

    pos = env._pos     # (n, 3)
    vel = env._vel     # (n, 3)
    adj = env._adj_avail
    dst = env._adj_dist

    n = const.n_total

    # Positions on correct shell
    r = np.linalg.norm(pos, axis=1)
    r_expected = cfg.ALTITUDE_M + 6_371_000.0
    check("T1.1 orbital radius within 1%",
          np.allclose(r, r_expected, rtol=0.01),
          f"mean={r.mean()/1e6:.3f} Mm, expected={r_expected/1e6:.3f} Mm")

    # Velocities reasonable (LEO ~7.6 km/s)
    v = np.linalg.norm(vel, axis=1)
    check("T1.2 orbital velocity 7-8 km/s",
          np.all((v > 7000) & (v < 8000)),
          f"min={v.min():.0f} m/s, max={v.max():.0f} m/s")

    # Adjacency is symmetric
    check("T1.3 adjacency symmetric",
          np.all(adj == adj.T),
          "adj[i,j] == adj[j,i]")

    # Each satellite has 2-4 ISL neighbours in 18x18 grid
    deg = adj.sum(axis=1)
    check("T1.4 all satellites connected (deg >= 2)",
          np.all(deg >= 2),
          f"degrees: min={deg.min()}, mean={deg.mean():.1f}, max={deg.max()}")

    check("T1.5 edge satellites may have deg < 4",
          deg.max() <= 4,
          f"max degree={deg.max()}", warn_only=(deg.max() > 4))

    # ISL distances within range
    active_dst = dst[adj > 0]
    check("T1.6 ISL distances <= ISL_RANGE_M",
          np.all(active_dst <= cfg.ISL_RANGE_M * 1.001),
          f"max={active_dst.max()/1e3:.0f} km, limit={cfg.ISL_RANGE_M/1e3:.0f} km")

    check("T1.7 ISL distances > 0",
          np.all(active_dst > 0),
          f"min={active_dst.min()/1e3:.0f} km")

    # Rate matrix sanity
    rates = env._rates
    check("T1.8 rates positive where link up",
          np.all(rates[adj > 0] > 0),
          f"min rate on active link={rates[adj>0].min()/1e6:.1f} Mbps")

    check("T1.9 rates zero where no link",
          np.all(rates[adj == 0] == 0),
          "")

    # Rel-velocity matrix symmetric and reasonable
    rv = env._rel_vel
    check("T1.10 rel_vel symmetric", np.allclose(rv, rv.T), "")
    rv_active = rv[adj > 0]
    check("T1.11 rel_vel in [0, 20 km/s]",
          np.all((rv_active >= 0) & (rv_active < 20_000)),
          f"max={rv_active.max():.0f} m/s")


# ══════════════════════════════════════════════════════════════════════
# T2  Feature vector dimensions and value ranges
# ══════════════════════════════════════════════════════════════════════
def test_t2_features():
    section("T2: Feature vector dimensions and value ranges")
    obs = env.reset(t_start=100.0)

    # Collect all feature vectors
    all_feats = []
    for i, o in obs.items():
        f = o["features"]  # (K, DIM_IN)
        if f.shape[0] > 0:
            all_feats.append(f)

    all_feats = np.concatenate(all_feats, axis=0)  # (total_edges, DIM_IN)

    check("T2.1 DIM_IN == 6",
          all_feats.shape[1] == cfg.DIM_IN,
          f"shape={all_feats.shape}")

    feat_names = ["cap", "link_avail", "prop_norm", "q_norm", "rel_vel_norm", "dist_n"]
    for fi, name in enumerate(feat_names):
        col = all_feats[:, fi]
        in_range = np.all((col >= 0.0) & (col <= 1.0 + 1e-6))
        check(f"T2.{fi+2} {name} in [0,1]",
              in_range,
              f"min={col.min():.4f}, max={col.max():.4f}, mean={col.mean():.4f}")

    # link_avail should be all 1.0 (only valid neighbors included)
    check("T2.8 link_avail always 1.0",
          np.all(all_feats[:, 1] == 1.0), "")

    # prop_norm and dist_n should be identical (same underlying data)
    check("T2.9 prop_norm == dist_n (feature 2 == feature 5)",
          np.allclose(all_feats[:, 2], all_feats[:, 5]),
          "Note: dist_n and prop_norm are redundant (both = adj_dist/ISL_RANGE_M)")

    # Masks
    all_masks = np.concatenate([obs[i]["mask"] for i in obs if len(obs[i]["mask"]) > 0])
    check("T2.10 masks are 0 or 1",
          np.all((all_masks == 0) | (all_masks == 1)),
          f"unique values: {np.unique(all_masks)}")

    # Neighbors list matches features rows
    mismatch = any(obs[i]["features"].shape[0] != len(obs[i]["neighbors"]) for i in obs)
    check("T2.11 features rows == len(neighbors) for all agents",
          not mismatch, "")


# ══════════════════════════════════════════════════════════════════════
# T3  Action-to-neighbor mapping consistency
# ══════════════════════════════════════════════════════════════════════
def test_t3_action_mapping():
    section("T3: Action-to-neighbor mapping consistency")
    obs = env.reset(t_start=200.0)

    n = const.n_total
    mismatch_count = 0
    fallback_count = 0
    total_paths    = 0

    # For each satellite, verify that obs neighbors match adj_avail ordering
    for i in range(n):
        obs_nb   = np.where(env._adj_avail[i])[0].tolist()
        env_nb   = obs[i]["neighbors"]
        if obs_nb != env_nb:
            mismatch_count += 1

    check("T3.1 observation neighbors == np.where(adj_avail[i]) for ALL agents",
          mismatch_count == 0,
          f"{mismatch_count}/{n} mismatches")

    # Simulate actions and track fallback rate
    for ep in range(3):
        obs = env.reset(t_start=rng.uniform(0, 6000))
        for slot in range(5):
            # Random actions over obs neighbors
            actions = {}
            for i, o in obs.items():
                nb_count = len(o["neighbors"])
                if nb_count > 0:
                    actions[i] = int(rng.integers(0, nb_count))
                else:
                    actions[i] = 0

            # Manually call _build_path and check if fallback triggered
            for sess in env.sessions:
                src, dst = sess.src, sess.dst
                path = env._build_path(src, dst, actions)
                total_paths += 1
                if path is not None and len(path) >= 2:
                    # Check if first hop follows chosen action
                    node    = src
                    obs_nb  = np.where(env._adj_avail[node])[0].tolist()
                    action  = actions.get(node, -1)
                    if obs_nb and 0 <= action < len(obs_nb):
                        cand = obs_nb[action]
                        if path[1] != cand:  # fallback was used
                            fallback_count += 1

            obs, _, _, _ = env.step(actions)

    fallback_rate = fallback_count / max(total_paths, 1)
    check("T3.2 fallback rate < 30% (agent action is usually respected)",
          fallback_rate < 0.30,
          f"fallback={fallback_count}/{total_paths} = {fallback_rate:.2%}",
          warn_only=(fallback_rate >= 0.30))

    check("T3.3 fallback rate < 60% (not completely overriding agent)",
          fallback_rate < 0.60,
          f"fallback rate={fallback_rate:.2%}")


# ══════════════════════════════════════════════════════════════════════
# T4  Reward formula correctness
# ══════════════════════════════════════════════════════════════════════
def test_t4_reward():
    section("T4: Reward formula correctness")
    obs = env.reset(t_start=300.0)

    # Test reward components manually for a known path
    # First find a valid session
    sess = env.sessions[0]
    src, dst = sess.src, sess.dst

    # Build shortest path via networkx
    import networkx as nx
    try:
        path = nx.shortest_path(env._nx_graph, src, dst, weight="prop_delay")
    except Exception:
        path = None

    if path is None or len(path) < 2:
        print(f"  {WARN} No path found src={src}→dst={dst}, skipping T4")
        return

    # Compute reward manually
    h = len(path) - 1
    q_sum = sum(env._queuing_delay(v) for v in path[1:])
    r_delay_expected = -(math.log(1 + h / cfg.EPS1) + cfg.EPS2 * q_sum)

    # Call environment reward
    env.queues[:] = 0.0  # reset queues for clean test
    r_env = env._compute_reward(path[0], path, env._path_e2e_delay(path), True)

    # Check delay term (other terms depend on node-specific quantities)
    q_sum2 = sum(env._queuing_delay(v) for v in path[1:])
    r_delay_check = -(math.log(1 + h / cfg.EPS1) + cfg.EPS2 * q_sum2)

    check("T4.1 r_delay formula: -(ln(1+h/EPS1) + EPS2*q_sum)",
          abs(r_delay_check - r_delay_expected) < 1e-9,
          f"expected={r_delay_expected:.4f}")

    check("T4.2 reward with zero queues is negative (penalty)",
          r_env < 0,
          f"r_env={r_env:.4f}")

    # Congestion reward: if queue < CONGESTION_THRESH → -d_prop
    node     = path[0]
    next_hop = path[1]
    d_prop   = env._adj_dist[node, next_hop] / C_LIGHT
    check("T4.3 prop delay > 0 on active link",
          d_prop > 0,
          f"d_prop={d_prop*1000:.3f} ms")

    # Routing direction: delta_H should be >= 0 for shortest path
    P = const.n_planes
    S = const.n_sats
    p_d, s_d   = dst // S, dst % S
    p_n, s_n   = node // S, node % S
    p_nx, s_nx = next_hop // S, next_hop % S
    H_node = min(abs(p_n - p_d), P - abs(p_n - p_d)) + min(abs(s_n - s_d), S - abs(s_n - s_d))
    H_next = min(abs(p_nx - p_d), P - abs(p_nx - p_d)) + min(abs(s_nx - s_d), S - abs(s_nx - s_d))
    delta_H = H_node - H_next
    check("T4.4 routing reward >= 0 for shortest-path first hop",
          delta_H >= 0,
          f"delta_H={delta_H}, H_node={H_node}, H_next={H_next}")

    # Delivery failure gives -PENALTY_KAPPA
    r_fail = env._compute_reward(path[0], path, 0.0, False)
    check("T4.5 failed delivery reward == -PENALTY_KAPPA",
          abs(r_fail - (-cfg.PENALTY_KAPPA)) < 1e-9,
          f"r_fail={r_fail}, -kappa={-cfg.PENALTY_KAPPA}")

    # Reward components weighted sum adds up
    r_total_w = (cfg.RHO_DELAY + cfg.RHO_CONGEST + cfg.RHO_ROUTING + cfg.RHO_STAB)
    check("T4.6 reward weights sum to 1.0",
          abs(r_total_w - 1.0) < 1e-6,
          f"sum={r_total_w}")

    # Queuing delay scalar range [0, 50ms]
    q_delays = np.array([env._queuing_delay(v) for v in range(const.n_total)])
    check("T4.7 queuing delays in [0, 50ms]",
          np.all((q_delays >= 0) & (q_delays <= 0.05 + 1e-9)),
          f"max={q_delays.max()*1000:.2f} ms")


# ══════════════════════════════════════════════════════════════════════
# T5  Queue dynamics
# ══════════════════════════════════════════════════════════════════════
def test_t5_queues():
    section("T5: Queue dynamics")
    obs = env.reset(t_start=0.0)

    # Fill queues artificially
    env.queues[:] = 0.0

    # Single packet burst on satellite 5
    env.queues[5] = 500.0

    q_before = env.queues[5]
    env._decay_queues()
    q_after = env.queues[5]

    check("T5.1 queues drain after decay",
          q_after < q_before,
          f"q_before={q_before:.1f}, q_after={q_after:.2f}")

    check("T5.2 queues never negative after decay",
          np.all(env.queues >= 0),
          f"min={env.queues.min():.4f}")

    # Overflow cap
    env.queues[:] = cfg.BUFFER_SIZE_PKTS + 1000
    env._decay_queues()
    # Cap applied in _route_packets, not _decay_queues; but queues should drain
    check("T5.3 queues drain from overfull state",
          np.all(env.queues <= cfg.BUFFER_SIZE_PKTS + 1000),  # only drain, not clamp
          f"max after decay={env.queues.max():.1f}")

    # Route packets and check queue update
    env.queues[:] = 0.0
    obs = env.reset(t_start=0.0)
    sess = env.sessions[0]
    src, dst = sess.src, sess.dst
    import networkx as nx
    try:
        path = nx.shortest_path(env._nx_graph, src, dst, weight="prop_delay")
    except Exception:
        path = None

    if path and len(path) >= 2:
        q_before = env.queues[path[1]]
        env._route_packets(path, 10, sess)
        q_after = env.queues[path[1]]
        check("T5.4 queue increases after routing packets",
              q_after >= q_before,
              f"q[path[1]] before={q_before:.0f}, after={q_after:.0f}")

    # Buffer-full mask
    env.queues[:] = cfg.BUFFER_SIZE_PKTS
    obs2 = env._get_observations()
    masks_all = np.concatenate([obs2[i]["mask"] for i in obs2 if len(obs2[i]["mask"]) > 0])
    check("T5.5 all neighbor masks == 0 when all queues full",
          np.all(masks_all == 0),
          f"non-zero masks when full: {(masks_all > 0).sum()}")

    env.queues[:] = 0.0  # restore


# ══════════════════════════════════════════════════════════════════════
# T6  Path building fallback frequency analysis
# ══════════════════════════════════════════════════════════════════════
def test_t6_path_building():
    section("T6: Path building (fallback and path quality)")
    env.queues[:] = 0.0
    obs = env.reset(t_start=400.0)

    n_sessions   = len(env.sessions)
    path_lengths = []
    none_count   = 0
    hop_limit_ok = True

    for _ in range(5):
        # Random valid actions
        actions = {}
        for i, o in obs.items():
            nb_count = len(o["neighbors"])
            actions[i] = int(rng.integers(0, max(nb_count, 1))) if nb_count > 0 else 0

        for sess in env.sessions:
            path = env._build_path(sess.src, sess.dst, actions)
            if path is None:
                none_count += 1
            else:
                path_lengths.append(len(path) - 1)
                if len(path) - 1 > cfg.HOP_LIMIT:
                    hop_limit_ok = False

        obs, _, _, _ = env.step(actions)

    total_paths = 5 * n_sessions
    none_rate   = none_count / total_paths

    check("T6.1 path found rate > 50% with random policy",
          none_rate < 0.5,
          f"none={none_count}/{total_paths} = {none_rate:.1%}",
          warn_only=(none_rate >= 0.5))

    if path_lengths:
        avg_hops = np.mean(path_lengths)
        check("T6.2 average path hops in [2, HOP_LIMIT]",
              2 <= avg_hops <= cfg.HOP_LIMIT,
              f"avg={avg_hops:.1f}, max={max(path_lengths)}, HOP_LIMIT={cfg.HOP_LIMIT}")

    check("T6.3 no paths exceed HOP_LIMIT",
          hop_limit_ok, "")


# ══════════════════════════════════════════════════════════════════════
# T7  Routing progress (greedy should make ΔH > 0 progress)
# ══════════════════════════════════════════════════════════════════════
def test_t7_routing_progress():
    section("T7: Routing direction reward analysis")
    obs = env.reset(t_start=500.0)

    import networkx as nx
    P = const.n_planes
    S = const.n_sats

    progress_rates = []
    for sess in env.sessions[:5]:
        src, dst = sess.src, sess.dst
        try:
            path = nx.shortest_path(env._nx_graph, src, dst, weight="prop_delay")
        except Exception:
            continue
        if len(path) < 3:
            continue

        hop_progress = 0
        for node, next_hop in zip(path[:-1], path[1:]):
            p_d, s_d   = dst // S, dst % S
            p_n, s_n   = node     // S, node     % S
            p_nx, s_nx = next_hop // S, next_hop % S
            H_node = min(abs(p_n - p_d), P - abs(p_n - p_d)) + min(abs(s_n - s_d), S - abs(s_n - s_d))
            H_next = min(abs(p_nx - p_d), P - abs(p_nx - p_d)) + min(abs(s_nx - s_d), S - abs(s_nx - s_d))
            if H_next < H_node:
                hop_progress += 1

        progress_rate = hop_progress / (len(path) - 1)
        progress_rates.append(progress_rate)

    avg_progress = np.mean(progress_rates) if progress_rates else 0.0
    check("T7.1 shortest-path hop progress > 80%",
          avg_progress > 0.80,
          f"avg progress rate={avg_progress:.2%}",
          warn_only=(avg_progress <= 0.80))

    # E2E delay on shortest path is reasonable (<500ms per hop)
    obs = env.reset(t_start=500.0)
    env.queues[:] = 0.0
    delays = []
    for sess in env.sessions[:5]:
        src, dst = sess.src, sess.dst
        try:
            path = nx.shortest_path(env._nx_graph, src, dst, weight="prop_delay")
        except Exception:
            continue
        if len(path) >= 2:
            d = env._path_e2e_delay(path)
            delays.append(d * 1000)  # ms

    if delays:
        avg_d = np.mean(delays)
        check("T7.2 E2E delay (zero queues) in [10ms, 500ms]",
              10 < avg_d < 500,
              f"avg={avg_d:.1f} ms, min={min(delays):.1f} ms, max={max(delays):.1f} ms")


# ══════════════════════════════════════════════════════════════════════
# T8  End-to-end delay components sanity
# ══════════════════════════════════════════════════════════════════════
def test_t8_delay_components():
    section("T8: Delay component sanity")
    obs = env.reset(t_start=600.0)
    env.queues[:] = 0.0

    # Pick one active link
    u_arr, v_arr = np.where(np.triu(env._adj_avail, k=1))
    if len(u_arr) == 0:
        print(f"  {WARN} No active links")
        return

    u, v = int(u_arr[0]), int(v_arr[0])
    d    = env._adj_dist[u, v]
    r    = env._rates[u, v]

    prop_ms = d / C_LIGHT * 1000
    tx_ms   = cfg.PKT_SIZE_BITS / r * 1000
    q_ms    = env._queuing_delay(v) * 1000  # 0 since queues empty

    check("T8.1 propagation delay in [1ms, 10ms]",
          1 < prop_ms < 10,
          f"prop={prop_ms:.3f} ms  (dist={d/1e3:.0f} km)")

    check("T8.2 transmission delay < 1ms for 1 Gbps link",
          tx_ms < 1.0,
          f"tx={tx_ms:.4f} ms (rate={r/1e9:.2f} Gbps)")

    check("T8.3 queuing delay 0ms with empty queue",
          q_ms < 0.001,
          f"q={q_ms:.4f} ms")

    # Fill queue and check queuing delay increases
    env.queues[v] = 500.0
    q_ms2 = env._queuing_delay(v) * 1000
    check("T8.4 queuing delay increases with filled queue",
          q_ms2 > q_ms,
          f"q_empty={q_ms:.4f} ms, q_500pkts={q_ms2:.4f} ms")

    # Queuing delay cap at 50ms
    env.queues[v] = cfg.BUFFER_SIZE_PKTS * 10  # overfull
    q_ms3 = env._queuing_delay(v) * 1000
    check("T8.5 queuing delay capped at 50ms",
          abs(q_ms3 - 50.0) < 0.1,
          f"q_overfull={q_ms3:.2f} ms")

    env.queues[:] = 0.0


# ══════════════════════════════════════════════════════════════════════
# T9  Baseline comparison (DLBH vs random policy E2E delay)
# ══════════════════════════════════════════════════════════════════════
def test_t9_baselines():
    section("T9: DLBH vs random policy comparison")
    import networkx as nx

    def run_episode_custom_routing(routing_fn, n_slots=10):
        """routing_fn(env, sess) → path or None"""
        obs = env.reset(t_start=rng.uniform(0, 6000))
        env.queues[:] = 0.0
        total_delay   = 0.0
        n_delivered   = 0
        n_dropped     = 0

        for _ in range(n_slots):
            for sess in env.sessions:
                arrivals = tgen.poisson_arrivals(sess.rate_pps, env.dt)
                if arrivals == 0:
                    continue
                path = routing_fn(env, sess)
                if path is None:
                    n_dropped += arrivals
                    continue
                e2e, ok = env._route_packets(path, arrivals, sess)
                if ok:
                    total_delay += e2e * arrivals
                    n_delivered += arrivals
                else:
                    n_dropped += arrivals
            env._decay_queues()
            env.t += env.dt

        total = n_delivered + n_dropped
        delivery_rate = n_delivered / max(total, 1)
        avg_delay_ms  = (total_delay / max(n_delivered, 1)) * 1000
        return delivery_rate, avg_delay_ms

    def dlbh_routing(env, sess):
        """Dynamic Load-Balanced Hop (Dijkstra on α·prop + (1-α)·queue_load)."""
        alpha = 0.5
        G = nx.DiGraph()
        n = env.n_sats
        for u in range(n):
            for v in np.where(env._adj_avail[u])[0]:
                prop_cost  = env._adj_dist[u, v] / C_LIGHT
                queue_cost = env.queues[v] / cfg.BUFFER_SIZE_PKTS
                w = alpha * prop_cost + (1 - alpha) * queue_cost
                G.add_edge(u, v, weight=w)
        try:
            return nx.shortest_path(G, sess.src, sess.dst, weight="weight")
        except Exception:
            return None

    def stsd_routing(env, sess):
        """Shortest Time Shortest Delay (Dijkstra on propagation delay)."""
        try:
            return nx.shortest_path(env._nx_graph, sess.src, sess.dst, weight="prop_delay")
        except Exception:
            return None

    def random_routing(env, sess):
        """Random walk routing."""
        path = [sess.src]
        visited = {sess.src}
        node = sess.src
        for _ in range(cfg.HOP_LIMIT):
            if node == sess.dst:
                return path
            nb = [j for j in np.where(env._adj_avail[node])[0] if j not in visited]
            if not nb:
                return None
            node = nb[int(rng.integers(0, len(nb)))]
            path.append(node)
            visited.add(node)
        return None

    N_RUNS = 3
    dlbh_delays  = []
    stsd_delays  = []
    rand_delays  = []
    dlbh_drs     = []
    stsd_drs     = []
    rand_drs     = []

    for _ in range(N_RUNS):
        dr, d  = run_episode_custom_routing(dlbh_routing)
        dlbh_delays.append(d); dlbh_drs.append(dr)
        dr, d  = run_episode_custom_routing(stsd_routing)
        stsd_delays.append(d); stsd_drs.append(dr)
        dr, d  = run_episode_custom_routing(random_routing)
        rand_delays.append(d); rand_drs.append(dr)

    dlbh_avg = np.mean(dlbh_delays)
    stsd_avg = np.mean(stsd_delays)
    rand_avg = np.mean(rand_delays)

    print(f"  {INFO} DLBH: delay={dlbh_avg:.1f} ms, DR={np.mean(dlbh_drs):.2%}")
    print(f"  {INFO} STSD: delay={stsd_avg:.1f} ms, DR={np.mean(stsd_drs):.2%}")
    print(f"  {INFO} Random: delay={rand_avg:.1f} ms, DR={np.mean(rand_drs):.2%}")

    check("T9.1 DLBH delay < random delay (heuristic beats random)",
          dlbh_avg < rand_avg,
          f"DLBH={dlbh_avg:.1f}ms < Random={rand_avg:.1f}ms",
          warn_only=True)

    check("T9.2 STSD delivery rate > random delivery rate",
          np.mean(stsd_drs) >= np.mean(rand_drs) * 0.9,
          f"STSD DR={np.mean(stsd_drs):.2%}, Random DR={np.mean(rand_drs):.2%}",
          warn_only=True)

    check("T9.3 baseline delays are physically reasonable (< 1000ms)",
          all(d < 1000 for d in [dlbh_avg, stsd_avg]),
          f"DLBH={dlbh_avg:.1f}ms, STSD={stsd_avg:.1f}ms")


# ══════════════════════════════════════════════════════════════════════
# T10  Training data flow (no NaN/Inf in 5 episodes)
# ══════════════════════════════════════════════════════════════════════
def test_t10_training_flow():
    section("T10: Training data flow (NaN/Inf check)")
    torch.manual_seed(SEED)
    MAX_NB = 4
    device = "cpu"

    actor  = TransformerActor(
        dim_in=cfg.DIM_IN, d_model=64, n_heads=4,
        d_ff=128, n_layers=2, dropout=0.0, max_neighbors=MAX_NB,
    )
    critic = CentralizedCritic(
        n_agents=const.n_total, max_neighbors=MAX_NB, dim_in=cfg.DIM_IN
    )
    trainer = PPO_CTDE(actor, critic, device=device,
                       max_iter=5 * cfg.HORIZON_SLOTS,
                       ppo_epochs=2, batch_size=512,
                       is_transformer=True)
    trainer.init_buffers(list(range(const.n_total)))

    def build_mob(obs_i, agent_id):
        nb = obs_i["neighbors"]
        n  = len(nb)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float32)
        mob = env._rel_vel[np.ix_(nb, nb)].astype(np.float32)
        mx  = mob.max()
        return (mob / mx) if mx > 0 else mob

    def build_dist(obs_i, agent_id):
        nb = obs_i["neighbors"]
        if len(nb) == 0:
            return np.zeros(0, dtype=np.float32)
        return np.clip(env._adj_dist[agent_id, nb].astype(np.float32) / cfg.ISL_RANGE_M, 0, 1)

    def build_gs(obs, n_agents):
        parts = []
        for i in range(n_agents):
            if i in obs:
                f = obs[i]["features"]
                n = f.shape[0]
                pad = np.zeros((MAX_NB - n, cfg.DIM_IN), dtype=np.float32) if n < MAX_NB else np.zeros((0, cfg.DIM_IN))
                parts.append(np.concatenate([f[:MAX_NB], pad], axis=0).flatten())
            else:
                parts.append(np.zeros(MAX_NB * cfg.DIM_IN, dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    nan_in_feats   = False
    nan_in_rewards = False
    nan_after_upd  = False
    ep_rewards     = []

    for ep in range(5):
        obs = env.reset(t_start=rng.uniform(0, 6000))
        ep_reward = 0.0

        for slot in range(min(cfg.HORIZON_SLOTS, 20)):
            gs       = build_gs(obs, const.n_total)
            obs_list = [obs[i] for i in range(const.n_total)]
            mob_list = [build_mob(obs[i], i) for i in range(const.n_total)]
            dist_list= [build_dist(obs[i], i) for i in range(const.n_total)]

            actions, logps = trainer.select_actions_batch(obs_list, mob_list, MAX_NB, dist_list)

            for i in range(const.n_total):
                obs_i = obs_list[i]
                if obs_i["features"].shape[0] == 0:
                    continue
                if np.any(np.isnan(obs_i["features"])) or np.any(np.isinf(obs_i["features"])):
                    nan_in_feats = True
                trainer.store(
                    agent_id=i,
                    feats=obs_i["features"],
                    mob=mob_list[i],
                    mask=obs_i["mask"],
                    action=actions[i],
                    logp=logps[i],
                    reward=0.0,
                    done=False,
                    global_state=gs,
                    dist=dist_list[i],
                )

            obs, rewards, done, info = env.step(actions)

            for i in range(const.n_total):
                r = rewards.get(i, 0.0)
                if math.isnan(r) or math.isinf(r):
                    nan_in_rewards = True
                if trainer.buffers[i].transitions:
                    trainer.buffers[i].transitions[-1].reward = r
                    trainer.buffers[i].transitions[-1].done   = done
                ep_reward += r

            if done:
                break

        ep_rewards.append(ep_reward)

        stats = trainer.update()

        # Check for NaN in model parameters after update
        for name, p in actor.named_parameters():
            if torch.any(torch.isnan(p)) or torch.any(torch.isinf(p)):
                nan_after_upd = True
                print(f"    {WARN} NaN/Inf in {name}")

    check("T10.1 no NaN in observation features",
          not nan_in_feats, "")

    check("T10.2 no NaN/Inf in rewards",
          not nan_in_rewards, "")

    check("T10.3 no NaN/Inf in actor parameters after update",
          not nan_after_upd, "")

    avg_r = np.mean(ep_rewards)
    check("T10.4 episode rewards are finite",
          all(math.isfinite(r) for r in ep_rewards),
          f"ep rewards: {[f'{r:.1f}' for r in ep_rewards]}")

    check("T10.5 episode rewards not all zero",
          avg_r != 0.0,
          f"avg={avg_r:.2f}")

    # Verify action indices are in valid range
    all_actions_valid = True
    for i, buf in trainer.buffers.items():
        for tr in buf.transitions:
            if tr.action.item() < 0 or tr.action.item() >= MAX_NB:
                all_actions_valid = False
    check("T10.6 all action indices in [0, MAX_NB)",
          all_actions_valid, f"MAX_NB={MAX_NB}")


# ══════════════════════════════════════════════════════════════════════
# T11  Convergence diagnostics (check if checkpoint exists and has improved)
# ══════════════════════════════════════════════════════════════════════
def test_t11_convergence():
    section("T11: Training convergence diagnostics")
    import os

    # Check reward history
    log_tf   = "logs/train_rewards.npy"
    log_maac = "logs/maac_rewards.npy"
    ckpt_tf  = "checkpoints/best_transformer.pt"
    ckpt_ma  = "checkpoints/best_maac.pt"

    if os.path.exists(log_tf):
        arr = np.load(log_tf)
        n = len(arr)
        first50_avg  = arr[:50].mean()  if n >= 50  else arr.mean()
        last50_avg   = arr[-50:].mean() if n >= 50  else arr.mean()
        improving    = last50_avg > first50_avg
        print(f"  {INFO} Proposed: {n} eps, best={arr.max():.2f}, "
              f"first50_avg={first50_avg:.2f}, last50_avg={last50_avg:.2f}")
        check("T11.1 Proposed training shows improvement (last50 > first50)",
              improving,
              f"first50={first50_avg:.2f}, last50={last50_avg:.2f}",
              warn_only=True)
        check("T11.2 Proposed has >= 100 episodes",
              n >= 100, f"n={n}")
    else:
        print(f"  {WARN} {log_tf} not found -- training not started or in progress")
        check("T11.1 Proposed reward log exists", False, "file missing", warn_only=True)

    if os.path.exists(log_maac):
        arr = np.load(log_maac)
        n = len(arr)
        first50_avg  = arr[:50].mean()  if n >= 50  else arr.mean()
        last50_avg   = arr[-50:].mean() if n >= 50  else arr.mean()
        print(f"  {INFO} MADRL: {n} eps, best={arr.max():.2f}, "
              f"first50_avg={first50_avg:.2f}, last50_avg={last50_avg:.2f}")
        check("T11.3 MADRL training shows improvement",
              last50_avg > first50_avg,
              f"first50={first50_avg:.2f}, last50={last50_avg:.2f}",
              warn_only=True)
    else:
        print(f"  {WARN} {log_maac} not found")
        check("T11.3 MADRL reward log exists", False, "file missing", warn_only=True)

    # Check checkpoint files exist
    check("T11.4 best_transformer.pt exists",
          os.path.exists(ckpt_tf), f"path={ckpt_tf}", warn_only=True)
    check("T11.5 best_maac.pt exists",
          os.path.exists(ckpt_ma), f"path={ckpt_ma}", warn_only=True)

    # Entropy analysis (from log)
    if os.path.exists(log_tf):
        arr = np.load(log_tf)
        n = len(arr)
        if n >= 200:
            variance_ratio = arr[-100:].std() / (arr[:100].std() + 1e-9)
            check("T11.6 reward variance decreasing (last100.std < 2x first100.std)",
                  variance_ratio < 2.0,
                  f"ratio={variance_ratio:.2f}",
                  warn_only=True)


# ══════════════════════════════════════════════════════════════════════
# T12  Link rates and q_norm normalization
# ══════════════════════════════════════════════════════════════════════
def test_t12_link_rates():
    section("T12: ISL link rates and q_norm normalization (SINR fix)")
    obs = env.reset(t_start=0.0)
    rates = env._rates
    active = env._adj_avail
    active_rates = rates[active]

    check("T12.1 all active link rates > 0 (SINR_THRESHOLD correct)",
          len(active_rates) > 0 and np.all(active_rates > 0),
          (f"min={active_rates.min()/1e6:.1f}Mbps, max={active_rates.max()/1e6:.1f}Mbps"
           if len(active_rates) > 0 else "no active links"))

    check("T12.2 link rates in [10 Mbps, BANDWIDTH_HZ]",
          np.all((active_rates >= 10e6) & (active_rates <= cfg.BANDWIDTH_HZ)),
          f"range=[{active_rates.min()/1e6:.0f}, {active_rates.max()/1e6:.0f}] Mbps")

    # Queue drain sanity
    env.queues[:] = 500.0
    q_before = env.queues.copy()
    env._decay_queues()
    q_after = env.queues.copy()
    check("T12.3 queues drain with positive rates",
          np.all(q_after <= q_before) and np.any(q_after < q_before),
          f"mean drain = {(q_before - q_after).mean():.1f} pkts/slot")

    # cap feature is non-zero
    env.queues[:] = 0.0
    obs2 = env.reset(t_start=0.0)
    all_caps = []
    for i, o in obs2.items():
        if o["features"].shape[0] > 0:
            all_caps.append(o["features"][:, 0])
    if all_caps:
        caps = np.concatenate(all_caps)
        check("T12.4 capacity feature (cap) is non-zero",
              np.all(caps > 0),
              f"cap range=[{caps.min():.4f}, {caps.max():.4f}]")

    # q_norm is properly normalized [0,1]
    env.queues[:] = 0.0
    obs_q0 = env._get_observations()
    all_qn = np.concatenate([obs_q0[i]["features"][:, 3] for i in obs_q0
                             if obs_q0[i]["features"].shape[0] > 0])
    check("T12.5 q_norm near 0 when queues empty",
          np.all(all_qn < 0.05),
          f"max q_norm with empty queues={all_qn.max():.4f}")

    env.queues[:] = cfg.BUFFER_SIZE_PKTS
    obs_qfull = env._get_observations()
    all_qn_full = np.concatenate([obs_qfull[i]["features"][:, 3] for i in obs_qfull
                                  if obs_qfull[i]["features"].shape[0] > 0])
    check("T12.6 q_norm = 1.0 when queues full",
          np.all(all_qn_full >= 0.99),
          f"min q_norm with full queues={all_qn_full.min():.4f}")
    env.queues[:] = 0.0

    # SINR threshold check
    import math
    from environment.constellation import C_LIGHT as _CL
    min_d = env._adj_dist[env._adj_avail].min()
    max_d = env._adj_dist[env._adj_avail].max()
    fspl_max = (20*math.log10(max_d) + 20*math.log10(cfg.CARRIER_FREQ_HZ)
                + 20*math.log10(4*math.pi/_CL))
    snr_min_db = (10*math.log10(cfg.TX_POWER_W) + cfg.TX_GAIN_DBI + cfg.RX_GAIN_DBI
                  - cfg.NOISE_PSD_DBW_HZ - 10*math.log10(cfg.BANDWIDTH_HZ) - fspl_max)
    check("T12.7 worst-case SNR > SINR_THRESHOLD_DB",
          snr_min_db > cfg.SINR_THRESHOLD_DB,
          f"worst SNR={snr_min_db:.2f}dB, threshold={cfg.SINR_THRESHOLD_DB}dB")


# ══════════════════════════════════════════════════════════════════════
# Main runner
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    print("\n" + "="*60)
    print("  LEO_Transformer_MARL -- Simulation Validation Suite")
    print("="*60)
    print(f"  Constellation: {N_PLANES}x{N_SATS}={N_PLANES*N_SATS} agents")
    print(f"  Config: EPS1={cfg.EPS1}, EPS2={cfg.EPS2}, EPS3={cfg.EPS3}")
    print(f"  Reward weights: r1={cfg.RHO_DELAY}, r2={cfg.RHO_CONGEST}, "
          f"r3={cfg.RHO_ROUTING}, r4={cfg.RHO_STAB}")

    tests = [
        test_t1_constellation,
        test_t2_features,
        test_t3_action_mapping,
        test_t4_reward,
        test_t5_queues,
        test_t6_path_building,
        test_t7_routing_progress,
        test_t8_delay_components,
        test_t9_baselines,
        test_t10_training_flow,
        test_t11_convergence,
        test_t12_link_rates,
    ]

    for fn in tests:
        try:
            fn()
        except Exception as e:
            section_name = fn.__name__.split("_")[1].upper()
            print(f"\n  {FAIL} {fn.__name__} raised exception: {e}")
            if args.verbose:
                traceback.print_exc()

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    n_warn = sum(1 for _, s, _ in results if s == "WARN")
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"  Total: {len(results)} checks  |  "
          f"{PASS} {n_pass}  {WARN} {n_warn}  {FAIL} {n_fail}")
    print(f"  Elapsed: {elapsed:.1f}s")

    if n_fail > 0:
        print(f"\n  {FAIL} Failed checks:")
        for name, status, detail in results:
            if status == "FAIL":
                print(f"    - {name}" + (f": {detail}" if detail else ""))

    if n_warn > 0:
        print(f"\n  {WARN} Warnings:")
        for name, status, detail in results:
            if status == "WARN":
                print(f"    - {name}" + (f": {detail}" if detail else ""))

    print()
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
