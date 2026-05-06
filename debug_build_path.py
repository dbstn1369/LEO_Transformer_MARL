"""Deep debug: verify that policy actions are actually being used + rewarded."""
import numpy as np
import torch
import config as cfg
from environment import Constellation, LEORoutingEnv, TrafficGenerator
from models import TransformerActor
from models.critic import CentralizedCritic
from algorithms import PPO_CTDE
from train import build_global_state, build_mob_vec, build_dist_vec

rng = np.random.default_rng(42)
torch.manual_seed(42)
const = Constellation(18, 18)
n = const.n_total

tgen = TrafficGenerator(n, n_fg_sessions=10, rng=rng)
env = LEORoutingEnv(const, tgen, horizon=100, rng=rng)
env.training = True

MAX_NB = 4
actor = TransformerActor(dim_self=cfg.DIM_SELF, max_neighbors=MAX_NB).to("cuda")
critic = CentralizedCritic(n_agents=n, max_neighbors=MAX_NB).to("cuda")
trainer = PPO_CTDE(actor, critic, device="cuda", is_transformer=True)

print("=== Run 1 episode, track policy action usage ===")
obs = env.reset()
policy_rates = []
fallback_counts = []
n_active_per_slot = []

for slot in range(10):
    obs_list = [obs[i] for i in range(n)]
    mob_list = [build_mob_vec(obs[i], env, i) for i in range(n)]
    dist_list = [build_dist_vec(obs[i], env, i) for i in range(n)]
    self_feat_list = [obs[i].get("self_feat", np.zeros(cfg.DIM_SELF, dtype=np.float32))
                      for i in range(n)]

    actions, logps = trainer.select_actions_batch(
        obs_list, mob_list, MAX_NB,
        dist_list=dist_list, self_feat_list=self_feat_list,
    )

    # Count policy action vs fallback usage
    # Actually observe what happens in _build_path by checking _current_fb_nodes
    next_obs, rewards, done, info = env.step(actions)

    n_active = sum(1 for v in rewards.values() if v is not None)
    policy_rates.append(env._last_policy_rate)
    fallback_counts.append(len(env._current_fb_nodes))
    n_active_per_slot.append(n_active)

    # Check reward magnitudes
    active_rewards = [v for v in rewards.values() if v is not None]
    if active_rewards and slot < 3:
        dly_vals = [r['dly'] for r in active_rewards]
        cng_vals = [r['cng'] for r in active_rewards]
        dir_vals = [r['dir'] for r in active_rewards]
        stb_vals = [r['stb'] for r in active_rewards]
        print(f"Slot {slot}: active={n_active}, policy_rate={env._last_policy_rate:.2f}, fb_nodes={len(env._current_fb_nodes)}")
        print(f"  dly: mean={np.mean(dly_vals):.3f}, std={np.std(dly_vals):.3f}")
        print(f"  cng: mean={np.mean(cng_vals):.3f}, std={np.std(cng_vals):.3f}")
        print(f"  dir: mean={np.mean(dir_vals):.3f}, std={np.std(dir_vals):.3f}")
        print(f"  stb: mean={np.mean(stb_vals):.3f}, std={np.std(stb_vals):.3f}")
    obs = next_obs

print(f"\n=== Summary over 10 slots ===")
print(f"Policy rate (fraction of decisions made by policy, not fallback):")
print(f"  mean={np.mean(policy_rates):.3f}, min={min(policy_rates):.3f}, max={max(policy_rates):.3f}")
print(f"Active agents per slot: mean={np.mean(n_active_per_slot):.1f} / {n}")
print(f"Fallback nodes per slot: mean={np.mean(fallback_counts):.1f}")

# Key diagnostic: if policy_rate is low, policy actions aren't being used
# If policy_rate is high but learning fails, reward signal is wrong
