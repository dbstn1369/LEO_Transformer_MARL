"""
Training entry point.

Trains the Transformer-MADRL agent using CTDE-PPO (Algorithm 1).
Saves checkpoints to ./checkpoints/.

Usage:
    python train.py [--episodes N] [--device cpu|cuda] [--seed S]
"""

import argparse
import os
import math
import time
import numpy as np
import torch

import config as cfg
from environment import Constellation, LEORoutingEnv, TrafficGenerator
from models import TransformerActor
from models.critic import CentralizedCritic
from algorithms import PPO_CTDE

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)


def build_global_state(obs: dict, n_agents: int, max_nb: int) -> np.ndarray:
    """
    Flatten all agents' feature observations into a single vector
    for the centralised critic.
    """
    parts = []
    for i in range(n_agents):
        if i in obs:
            f = obs[i]["features"]           # (N_i, DIM_IN)
            n = f.shape[0]
            # Pad to max_nb
            pad = np.zeros((max_nb - n, cfg.DIM_IN), dtype=np.float32) if n < max_nb else np.zeros((0, cfg.DIM_IN))
            parts.append(np.concatenate([f[:max_nb], pad], axis=0).flatten())
        else:
            parts.append(np.zeros(max_nb * cfg.DIM_IN, dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def build_mob_vec(obs_i: dict, env: LEORoutingEnv, agent_id: int) -> np.ndarray:
    """Build agent-to-neighbour relative velocity vector (normalised by V_MAX_MS).

    Paper Eq.(24): attention penalty = -β·‖Δv_{ij}‖/v_max  where i=agent, j=neighbour.
    Previous version used pairwise neighbour-to-neighbour velocity (semantically wrong).
    """
    nb = obs_i["neighbors"]
    if len(nb) == 0:
        return np.zeros(0, dtype=np.float32)
    mob = env._rel_vel[agent_id, nb].astype(np.float32) / cfg.V_MAX_MS
    return np.clip(mob, 0.0, 1.0)


# Keep alias for backward compatibility with train_madrl.py import
build_mob_matrix = build_mob_vec


def build_dist_vec(obs_i: dict, env: LEORoutingEnv, agent_id: int) -> np.ndarray:
    """Build agent-to-neighbour distance vector (normalised to [0,1])."""
    nb = obs_i["neighbors"]
    if len(nb) == 0:
        return np.zeros(0, dtype=np.float32)
    dist = env._adj_dist[agent_id, nb].astype(np.float32) / cfg.ISL_RANGE_M
    return np.clip(dist, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int,   default=cfg.N_EPISODES)
    parser.add_argument("--device",   type=str,   default="cpu")
    parser.add_argument("--seed",     type=int,   default=cfg.SEED)
    parser.add_argument("--planes",   type=int,   default=cfg.TRAIN_N_PLANES)
    parser.add_argument("--sats",     type=int,   default=cfg.TRAIN_N_SATS_PER_PLANE)
    parser.add_argument("--lr",       type=float, default=cfg.LEARNING_RATE, help="Override learning rate")
    parser.add_argument("--entropy",  type=float, default=cfg.ENTROPY_COEF,  help="Override entropy coefficient")
    parser.add_argument("--epochs",   type=int,   default=cfg.PPO_EPOCHS,    help="Override PPO epochs")
    parser.add_argument("--resume",     action="store_true", help="Resume from best checkpoint")
    parser.add_argument("--n_sessions", type=int, default=cfg.N_GROUND_PAIRS,
                        help="Number of traffic sessions (more = denser reward signal)")
    parser.add_argument("--tag",        type=str, default="",
                        help="Tag for reward log filename (e.g. lr1e-3 → logs/train_rewards_lr1e-3.npy)")
    parser.add_argument("--mini_batch", type=int, default=cfg.MINI_BATCH_SIZE,
                        help="PPO mini-batch size")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # ── Environment setup ────────────────────────────────────────────────────────
    const   = Constellation(args.planes, args.sats)
    n_sats  = const.n_total
    tgen    = TrafficGenerator(n_sats, n_fg_sessions=args.n_sessions, rng=rng)
    env     = LEORoutingEnv(const, tgen, horizon=cfg.HORIZON_SLOTS, rng=rng)
    env.training = True   # weak fallback: policy must learn routing
    # Proposed: full link-quality awareness via attention bias (no extra blindness)
    env._eval_extra_per_scale = 0.0

    MAX_NB = 4   # 4-connected Walker grid (intra ±1, inter ±1 plane)

    # ── Models ──────────────────────────────────────────────────────────────────
    actor  = TransformerActor(
        dim_in=cfg.DIM_IN,
        dim_self=cfg.DIM_SELF,
        d_model=cfg.D_MODEL,
        n_heads=cfg.N_HEADS,
        d_ff=cfg.D_FF,
        n_layers=cfg.N_LAYERS,
        dropout=cfg.DROPOUT,
        max_neighbors=MAX_NB,
    )
    critic = CentralizedCritic(
        n_agents=n_sats,
        max_neighbors=MAX_NB,
        dim_in=cfg.DIM_IN,
    )

    trainer = PPO_CTDE(
        actor,
        critic,
        lr=args.lr,
        entropy_coef=args.entropy,
        batch_size=args.mini_batch,
        device=args.device,
        max_iter=args.episodes * cfg.HORIZON_SLOTS,
        is_transformer=True,
    )
    # Override PPO epochs if specified
    trainer.ppo_epochs = args.epochs
    trainer.init_buffers(list(range(n_sats)))

    # ── Resume from checkpoint ────────────────────────────────────────────────
    reward_history = []
    best_reward    = -math.inf
    if args.resume and os.path.exists("checkpoints/best_transformer.pt"):
        ckpt = torch.load("checkpoints/best_transformer.pt", map_location=args.device)
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        best_reward = ckpt.get("reward", -math.inf)
        if os.path.exists("logs/train_rewards.npy"):
            reward_history = np.load("logs/train_rewards.npy").tolist()
        print(f"Resumed from ep {ckpt.get('episode', '?')}, best reward={best_reward:.2f}, history={len(reward_history)} eps")

    print(f"Constellation: {args.planes} planes × {args.sats} sats = {n_sats} agents")
    print(f"Training for {args.episodes} more episodes × {cfg.HORIZON_SLOTS} slots")

    # ── Training loop ────────────────────────────────────────────────────────────
    for ep in range(1, args.episodes + 1):
        t_start_ep = time.time()
        obs      = env.reset(t_start=rng.uniform(0, 6000))
        ep_reward    = 0.0
        ep_delivered = 0
        ep_dropped   = 0

        slot_gs_list = []   # one global state per slot (shared across agents)
        for slot in range(cfg.HORIZON_SLOTS):
            # Build global state once per slot and store in list (not per-agent copy)
            gs = build_global_state(obs, n_sats, MAX_NB)
            slot_gs_list.append(gs)

            # Precompute mobility vectors + distance vectors + self-features for all agents
            obs_list       = [obs[i] for i in range(n_sats)]
            mob_list       = [build_mob_vec(obs[i], env, i)                   for i in range(n_sats)]
            dist_list      = [build_dist_vec(obs[i], env, i)                  for i in range(n_sats)]
            self_feat_list = [obs[i].get("self_feat", np.zeros(cfg.DIM_SELF, dtype=np.float32))
                              for i in range(n_sats)]

            # Batched action selection (single forward pass)
            actions, logps = trainer.select_actions_batch(
                obs_list, mob_list, MAX_NB,
                dist_list=dist_list,
                self_feat_list=self_feat_list,
            )

            # Store transitions (raw_comps placeholder; updated after env.step)
            for i in range(n_sats):
                obs_i = obs_list[i]
                if obs_i["features"].shape[0] == 0:
                    continue
                trainer.store(
                    agent_id=i,
                    feats=obs_i["features"],
                    mob=mob_list[i],
                    mask=obs_i["mask"],
                    action=actions[i],
                    logp=logps[i],
                    reward=(0.0, 0.0, 0.0, 0.0),   # placeholder
                    active=False,
                    done=False,
                    slot_idx=slot,
                    dist=dist_list[i],
                    self_feat=self_feat_list[i],
                )

            # Environment step
            next_obs, rewards, done, info = env.step(actions)

            # Update raw reward components and active flag for each agent's last transition
            for i in range(n_sats):
                comps = rewards.get(i)   # None if agent was inactive
                if trainer.buffers[i].transitions:
                    tr = trainer.buffers[i].transitions[-1]
                    if comps is not None:
                        tr.reward = (comps["dly"], comps["cng"],
                                     comps["dir"], comps["stb"])
                        tr.active = True
                    tr.done = done

            ep_delivered += info["n_delivered"]
            ep_dropped  += info["n_dropped"]

            obs = next_obs
            if done:
                break

        # ep_reward: delivery rate (higher = better, scale [0, 1])
        ep_total = ep_delivered + ep_dropped
        ep_reward = ep_delivered / max(ep_total, 1)

        # ── PPO update ──────────────────────────────────────────────────────────
        stats = trainer.update(slot_gs_list=slot_gs_list)
        reward_history.append(ep_reward)

        # ── Logging ─────────────────────────────────────────────────────────────
        if ep % 10 == 0 or ep == 1:
            elapsed = time.time() - t_start_ep
            avg_r   = np.mean(reward_history[-10:])
            print(
                f"Ep {ep:4d}/{args.episodes} | "
                f"Reward {ep_reward:8.2f} | "
                f"Avg(10) {avg_r:8.2f} | "
                f"PLoss {stats.get('policy_loss', 0):.4f} | "
                f"VLoss {stats.get('value_loss', 0):.4f} | "
                f"Ent {stats.get('entropy', 0):.4f} | "
                f"{elapsed:.1f}s"
            )

        # ── Checkpoint ──────────────────────────────────────────────────────────
        ckpt_tag = f"_{args.tag}" if args.tag else ""
        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save({
                "episode"  : ep,
                "actor"    : actor.state_dict(),
                "critic"   : critic.state_dict(),
                "reward"   : best_reward,
            }, f"checkpoints/best_transformer{ckpt_tag}.pt")

        if ep % 100 == 0:
            torch.save({
                "episode"  : ep,
                "actor"    : actor.state_dict(),
                "critic"   : critic.state_dict(),
            }, f"checkpoints/transformer_ep{ep}.pt")

    # Save reward history
    tag = f"_{args.tag}" if args.tag else ""
    np.save(f"logs/train_rewards{tag}.npy", np.array(reward_history))
    print(f"\nTraining complete. Best reward: {best_reward:.2f}")
    print(f"Rewards saved to logs/train_rewards{tag}.npy")
    print("Checkpoint saved to checkpoints/best_transformer.pt")


if __name__ == "__main__":
    main()
