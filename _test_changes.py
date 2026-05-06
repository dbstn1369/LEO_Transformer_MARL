"""Quick sanity test for all code changes."""
import sys
print("Python:", sys.executable)

import config as cfg
print("DIM_IN:", cfg.DIM_IN)

from models.transformer_actor import TransformerActor, MobilityAwareAttention
print("TransformerActor imported OK")

import torch

# Test with dist
actor = TransformerActor()
f = torch.zeros(1, 4, cfg.DIM_IN)
m = torch.zeros(1, 4, 4)
v = torch.ones(1, 4)
d = torch.ones(1, 4) * 0.5
out = actor(f, m, v, d)
print("Actor output shape:", out.shape)  # should be (1, 4)

# Test without dist (backward compat)
out2 = actor(f, m, v)
print("Without dist:", out2.shape)

# Test DLBH
from routing.dlbh import DLBH
dlbh = DLBH(324)
print("DLBH OK:", dlbh.__class__.__name__)

# Test env feature vector
from environment import Constellation, LEORoutingEnv, TrafficGenerator
import numpy as np
rng = np.random.default_rng(42)
const = Constellation(6, 6)
tgen = TrafficGenerator(const.n_total, rng=rng)
env = LEORoutingEnv(const, tgen, rng=rng)
obs = env.reset()
active = [k for k in obs if obs[k]["features"].shape[0] > 0]
if active:
    i = active[0]
    feat = obs[i]["features"]
    print(f"Feature shape: {feat.shape}, DIM_IN={cfg.DIM_IN}")
    assert feat.shape[1] == cfg.DIM_IN, f"DIM_IN mismatch: {feat.shape[1]} != {cfg.DIM_IN}"
    # Feature 5 and 6 should be rel_vel and dist_norm (not hop count)
    print(f"  feat[0]: {feat[0]}")  # [cap, link, prop, queue, rv, dist]

# Test build_dist_vec
from train import build_mob_matrix, build_dist_vec
if active:
    i = active[0]
    mob = build_mob_matrix(obs[i], env, i)
    dv  = build_dist_vec(obs[i],  env, i)
    print(f"mob shape: {mob.shape}, dist_vec shape: {dv.shape}")
    print(f"dist_vec: {dv}")

print("\nALL TESTS PASSED")
