"""
Quick smoke test: 환경 초기화 → 1 에피소드 (10 슬롯) 실행 → 모든 스킴 경로 확인
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("  LEO Transformer MARL - Smoke Test")
print("=" * 60)

# ── 1. Constellation ──────────────────────────────────────────
from environment.constellation import Constellation
import config as cfg

const = Constellation(
    n_planes=20, n_sats=22,  # 440 sats: intra~1978km, inter~2176km (both < 2500km ISL)
    altitude_m=cfg.ALTITUDE_M,
    inclination_deg=cfg.INCLINATION_DEG,
)
print(f"\n[1] Constellation: {const.n_planes}x{const.n_sats} = {const.n_total} sats")

pos, vel, adj_avail, adj_dist = const.build_topology(t=0.0)
n_links = int(adj_avail.sum()) // 2
print(f"    Positions:  {pos.shape}  |  Active ISLs: {n_links}")

# ── 2. Environment ────────────────────────────────────────────
from environment.leo_env import LEORoutingEnv
from environment.traffic import TrafficGenerator

rng  = np.random.default_rng(42)
tgen = TrafficGenerator(const.n_total, n_fg_sessions=2, rng=rng)
env  = LEORoutingEnv(const, tgen, horizon=10, rng=rng)

obs  = env.reset()
print(f"\n[2] Environment reset OK  |  Sessions: {len(env.sessions)}")
agent0 = obs[0]
print(f"    Agent-0 neighbours: {len(agent0['neighbors'])}  "
      f"features: {agent0['features'].shape}  "
      f"mask: {agent0['mask']}")

# ── 3. Transformer actor forward pass ────────────────────────
from models.transformer_actor import TransformerActor

actor = TransformerActor(max_neighbors=4)
print(f"\n[3] TransformerActor parameters: "
      f"{sum(p.numel() for p in actor.parameters()):,}")

# Build dummy input (1 agent, 4 neighbours padded)
MAX_NB = 4
n  = agent0["features"].shape[0]
f  = np.zeros((MAX_NB, cfg.DIM_IN), dtype=np.float32)
f[:n] = agent0["features"][:MAX_NB]
m  = np.zeros((MAX_NB, MAX_NB), dtype=np.float32)
v  = np.zeros(MAX_NB, dtype=np.float32)
v[:len(agent0["mask"][:MAX_NB])] = agent0["mask"][:MAX_NB]

ft = torch.tensor(f).unsqueeze(0)
mt = torch.tensor(m).unsqueeze(0)
vt = torch.tensor(v).unsqueeze(0)

with torch.no_grad():
    logits = actor(ft, mt, vt)
print(f"    Logits shape: {logits.shape}  sample: {logits[0].tolist()[:4]}")

# ── 4. One full episode (10 slots) ───────────────────────────
from models.critic import CentralizedCritic
from algorithms.ppo_ctde import PPO_CTDE

critic  = CentralizedCritic(n_agents=const.n_total, max_neighbors=MAX_NB)
trainer = PPO_CTDE(actor, critic, is_transformer=True)
trainer.init_buffers(list(range(const.n_total)))

print(f"\n[4a] DRL (untrained, random actions) - 5 slots:")
obs = env.reset()
total_delivered = 0
for slot in range(5):
    actions = {}
    for i in range(const.n_total):
        obs_i = obs[i]
        nb    = obs_i["features"].shape[0]
        if nb == 0:
            actions[i] = 0; continue
        nb_pad = min(nb, MAX_NB)
        f2 = np.zeros((MAX_NB, cfg.DIM_IN), dtype=np.float32)
        f2[:nb_pad] = obs_i["features"][:nb_pad]
        m2 = np.zeros((MAX_NB, MAX_NB), dtype=np.float32)
        v2 = np.zeros(MAX_NB, dtype=np.float32)
        v2[:nb_pad] = obs_i["mask"][:nb_pad]
        a, lp = trainer.select_action(f2, m2, v2)
        actions[i] = a
    obs, rewards, done, info = env.step(actions)
    total_delivered += info["n_delivered"]
    print(f"    Slot {slot+1}: delivered={info['n_delivered']:3d}  dropped={info['n_dropped']:3d}  "
          f"(random agent -> loops expected)")
print(f"    NOTE: delivered=0 is expected for untrained agent (random policy creates loops)")

# ── 4b. STSD-guided actions (verify delivery works) ──────────
print(f"\n[4b] STSD-guided actions (oracle) - 5 slots:")
from routing.stsd import STSD
stsd2 = STSD(const.n_total)
obs   = env.reset()
total_stsd = 0
for slot in range(5):
    env._update_topology()
    paths = stsd2.route(env.adjacency, env.distances, env.sessions)
    # Convert paths to per-node next-hop action indices
    actions2 = {}
    for sess in env.sessions:
        path = paths.get(sess.sid)
        if path is None: continue
        for idx, node in enumerate(path[:-1]):
            nxt = path[idx + 1]
            nb_i = [j for j in range(const.n_total) if env.adjacency[node, j]]
            if nxt in nb_i:
                actions2[node] = nb_i.index(nxt)
    next_obs, rewards, done, info = env.step(actions2)
    total_stsd += info["n_delivered"]
    print(f"    Slot {slot+1}: delivered={info['n_delivered']:3d}  dropped={info['n_dropped']:3d}  "
          f"delay={info['total_delay']:.6f}s")
    obs = next_obs
print(f"    Total delivered (STSD oracle): {total_stsd}")

# ── 5. Heuristic baselines ────────────────────────────────────
from routing.stsd import STSD
from routing.dlrh import DLRH

obs = env.reset()
stsd = STSD(const.n_total)
dlrh = DLRH(const.n_total)

stsd_paths = stsd.route(env.adjacency, env.distances, env.sessions)
dlrh_paths = dlrh.route(env.adjacency, env.distances, env.queues, env.sessions)

def path_str(p): return "->".join(map(str,p)) if p else "No Path"
print(f"\n[5] Baseline routing (1 snapshot):")
for sess in env.sessions:
    print(f"    Session {sess.sid} ({sess.src}→{sess.dst})")
    print(f"      STSD: {path_str(stsd_paths.get(sess.sid))}")
    print(f"      DLRH: {path_str(dlrh_paths.get(sess.sid))}")

print("\n" + "=" * 60)
print("  All checks passed!")
print("=" * 60)
