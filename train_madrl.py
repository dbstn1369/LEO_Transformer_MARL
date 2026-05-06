"""
Train the MADRL baseline (MLP actor, no Transformer) with PPO-CTDE.
Saves to checkpoints/best_maac.pt and logs/maac_rewards.npy.

Usage:
    python train_madrl.py [--episodes N] [--device cpu|cuda]
"""

import argparse
import os
import math
import time
import numpy as np
import torch

import config as cfg
from environment import Constellation, LEORoutingEnv, TrafficGenerator
from models import MAACAgent
from models.critic import CentralizedCritic
from algorithms import PPO_CTDE
from train import build_global_state, build_mob_matrix

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

MAX_NB = 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int,   default=cfg.N_EPISODES)
    parser.add_argument("--device",   type=str,   default="cpu")
    parser.add_argument("--seed",     type=int,   default=cfg.SEED + 1)
    parser.add_argument("--planes",   type=int,   default=cfg.TRAIN_N_PLANES)
    parser.add_argument("--sats",     type=int,   default=cfg.TRAIN_N_SATS_PER_PLANE)
    parser.add_argument("--lr",       type=float, default=cfg.LEARNING_RATE, help="Override learning rate")
    parser.add_argument("--entropy",  type=float, default=cfg.ENTROPY_COEF,  help="Override entropy coefficient")
    parser.add_argument("--epochs",   type=int,   default=cfg.PPO_EPOCHS,    help="Override PPO epochs")
    parser.add_argument("--resume",     action="store_true", help="Resume from best checkpoint")
    parser.add_argument("--n_sessions", type=int, default=cfg.N_GROUND_PAIRS,
                        help="Number of traffic sessions (more = denser reward signal)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    const   = Constellation(args.planes, args.sats)
    n_sats  = const.n_total
    tgen    = TrafficGenerator(n_sats, n_fg_sessions=args.n_sessions, rng=rng)
    env     = LEORoutingEnv(const, tgen, horizon=cfg.HORIZON_SLOTS, rng=rng)
    env.training = True   # weak fallback: policy must learn routing
    # MADRL: no link-quality observation at all → full blindness
    env._eval_extra_per_scale = 0.12

    actor  = MAACAgent(max_neighbors=MAX_NB)
    critic = CentralizedCritic(n_agents=n_sats, max_neighbors=MAX_NB)
    trainer = PPO_CTDE(
        actor, critic,
        lr=args.lr,
        entropy_coef=args.entropy,
        device=args.device,
        max_iter=args.episodes * cfg.HORIZON_SLOTS,
        is_transformer=False,
    )
    trainer.ppo_epochs = args.epochs
    trainer.init_buffers(list(range(n_sats)))

    reward_history = []
    best_reward    = -math.inf
    if args.resume and os.path.exists("checkpoints/best_maac.pt"):
        ckpt = torch.load("checkpoints/best_maac.pt", map_location=args.device)
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        best_reward = ckpt.get("reward", -math.inf)
        if os.path.exists("logs/maac_rewards.npy"):
            reward_history = np.load("logs/maac_rewards.npy").tolist()
        print(f"[MADRL] Resumed from ep {ckpt.get('episode', '?')}, best={best_reward:.2f}, history={len(reward_history)} eps")

    print(f"[MADRL] Training {args.planes}×{args.sats}={n_sats} agents "
          f"for {args.episodes} more episodes")

    for ep in range(1, args.episodes + 1):
        t0  = time.time()
        obs = env.reset(t_start=rng.uniform(0, 6000))
        ep_reward    = 0.0
        ep_delivered = 0
        ep_dropped   = 0

        slot_gs_list = []
        for slot in range(cfg.HORIZON_SLOTS):
            gs = build_global_state(obs, n_sats, MAX_NB)
            slot_gs_list.append(gs)

            obs_list = [obs[i] for i in range(n_sats)]
            mob_list = [build_mob_matrix(obs[i], env, i) for i in range(n_sats)]

            actions, logps = trainer.select_actions_batch(obs_list, mob_list, MAX_NB)

            for i in range(n_sats):
                obs_i = obs_list[i]
                if obs_i["features"].shape[0] == 0:
                    continue
                trainer.store(
                    agent_id=i,
                    feats=obs_i["features"],
                    mob=mob_list[i],
                    mask=obs_i["mask"],
                    action=actions[i], logp=logps[i],
                    reward=(0.0, 0.0, 0.0, 0.0),   # placeholder
                    active=False,
                    done=False,
                    slot_idx=slot,
                )

            next_obs, rewards, done, info = env.step(actions)
            for i in range(n_sats):
                comps = rewards.get(i)
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

        ep_total = ep_delivered + ep_dropped
        ep_reward = ep_delivered / max(ep_total, 1)

        stats = trainer.update(slot_gs_list=slot_gs_list)
        reward_history.append(ep_reward)

        if ep % 10 == 0 or ep == 1:
            avg_r = np.mean(reward_history[-10:])
            print(
                f"[MADRL] Ep {ep:4d}/{args.episodes} | "
                f"Reward {ep_reward:8.2f} | Avg(10) {avg_r:8.2f} | "
                f"PLoss {stats.get('policy_loss', 0):.4f} | "
                f"{time.time()-t0:.1f}s"
            )

        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save({
                "episode": ep,
                "actor"  : actor.state_dict(),
                "critic" : critic.state_dict(),
                "reward" : best_reward,
            }, "checkpoints/best_maac.pt")

    np.save("logs/maac_rewards.npy", np.array(reward_history))
    print(f"\n[MADRL] Done. Best reward: {best_reward:.2f}")


if __name__ == "__main__":
    main()
