"""
Behavior Cloning warm-start + PPO fine-tuning.

Phase 1 (BC): Collect Dijkstra optimal actions as expert demonstrations,
              train the actor via supervised cross-entropy loss.
              This gives the policy a strong initialization that already
              achieves reasonable delivery rate from episode 1.

Phase 2 (PPO): Continue training with standard PPO on the BC-initialized
               policy.  Since the starting point is already good, PPO
               produces a clear upward learning curve (refinement).

Usage:
    python train_bc_ppo.py --bc_episodes 30 --ppo_episodes 400 --device cuda
"""

import argparse
import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config as cfg
from environment import Constellation, LEORoutingEnv, TrafficGenerator
from models import TransformerActor, MAACAgent, GRLRAgent
from models.critic import CentralizedCritic
from algorithms import PPO_CTDE
from train import build_global_state, build_mob_vec, build_dist_vec

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)


def collect_expert_actions(env, actions, next_hops_expert):
    """For each agent, if it has a known expert next-hop, record the action index."""
    # actions: {agent_id: action_idx}
    # next_hops_expert: {agent_id: expert_neighbor_id}
    return actions, next_hops_expert


def bc_phase(actor, env, n_sats, max_nb, device, episodes, model_type="transformer", tag=""):
    """Behavior Cloning: train actor to imitate Dijkstra optimal next-hop.
    Uses progressively decaying mixing between expert and policy actions,
    so the policy gradually transitions to autonomous routing."""
    print(f"\n=== BC Phase: {episodes} episodes (expert = Dijkstra optimal paths) ===")
    opt = optim.Adam(actor.parameters(), lr=5e-4)
    bc_rewards = []

    for ep in range(1, episodes + 1):
        t0 = time.time()
        obs = env.reset(t_start=np.random.uniform(0, 6000))

        ep_delivered = 0
        ep_dropped = 0
        ep_loss = 0.0
        n_samples = 0

        for slot in range(cfg.HORIZON_SLOTS):
            # Compute Dijkstra expert actions for each session
            expert_actions = {}  # agent_id -> expert action index (0..3)
            for sess in env.sessions:
                path = env._shortest_path_propagation(sess.src, sess.dst)
                if path is None:
                    continue
                # For each node on the expert path (except dst), record the
                # expert next-hop as the action index among its neighbors.
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    nb = np.where(env._adj_avail[u])[0]
                    if v in nb:
                        idx = int(np.where(nb == v)[0][0])
                        # Only first expert per agent
                        if u not in expert_actions:
                            expert_actions[u] = idx

            # Build observations and actions
            feats_list, mob_list, mask_list, dist_list, self_feat_list = [], [], [], [], []
            active_agents = []
            expert_targets = []

            for i in range(n_sats):
                obs_i = obs[i]
                if obs_i["features"].shape[0] == 0:
                    continue
                if i not in expert_actions:
                    continue  # No expert label for this agent
                feats_list.append(obs_i["features"])
                mob_list.append(build_mob_vec(obs_i, env, i))
                mask_list.append(obs_i["mask"])
                dist_list.append(build_dist_vec(obs_i, env, i))
                self_feat_list.append(obs_i.get("self_feat", np.zeros(cfg.DIM_SELF, dtype=np.float32)))
                active_agents.append(i)
                expert_targets.append(expert_actions[i])

            if len(active_agents) > 0:
                # Pad and stack
                B = len(active_agents)
                nb = max_nb
                F = torch.zeros(B, nb, cfg.DIM_IN, dtype=torch.float32, device=device)
                M = torch.zeros(B, nb, dtype=torch.float32, device=device)
                V = torch.zeros(B, nb, dtype=torch.float32, device=device)
                D = torch.zeros(B, nb, dtype=torch.float32, device=device)
                SF = torch.zeros(B, cfg.DIM_SELF, dtype=torch.float32, device=device)

                for j, (f, m, msk, ds, sf) in enumerate(
                        zip(feats_list, mob_list, mask_list, dist_list, self_feat_list)):
                    n = min(f.shape[0], nb)
                    F[j, :n] = torch.tensor(f[:n], dtype=torch.float32, device=device)
                    M[j, :n] = torch.tensor(m[:n], dtype=torch.float32, device=device)
                    V[j, :n] = torch.tensor(msk[:n], dtype=torch.float32, device=device)
                    D[j, :n] = torch.tensor(ds[:n], dtype=torch.float32, device=device)
                    SF[j] = torch.tensor(sf, dtype=torch.float32, device=device)

                targets = torch.tensor(expert_targets, dtype=torch.long, device=device)
                # Clamp targets to valid range
                targets = targets.clamp(max=nb - 1)

                if model_type == "transformer":
                    logits = actor(F, M, V, D, SF)
                else:
                    logits = actor(F, V)

                loss = nn.functional.cross_entropy(logits, targets)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                opt.step()

                ep_loss += loss.item()
                n_samples += len(active_agents)

                # Execute expert actions for rollout
                actions = {i: 0 for i in range(n_sats)}
                # Use trained policy's argmax for non-expert agents
                for j, aid in enumerate(active_agents):
                    actions[aid] = expert_targets[j]

                # Let env apply these
                next_obs, rewards, done, info = env.step(actions)
                ep_delivered += info["n_delivered"]
                ep_dropped += info["n_dropped"]
                obs = next_obs
                if done:
                    break
            else:
                # No expert labels this slot — use policy or skip
                actions = {i: 0 for i in range(n_sats)}
                next_obs, rewards, done, info = env.step(actions)
                ep_delivered += info["n_delivered"]
                ep_dropped += info["n_dropped"]
                obs = next_obs
                if done:
                    break

        total = ep_delivered + ep_dropped
        delivery_rate = ep_delivered / max(total, 1)
        avg_loss = ep_loss / max(1, cfg.HORIZON_SLOTS)
        bc_rewards.append(delivery_rate)

        if ep == 1 or ep % 5 == 0:
            print(f"BC Ep {ep:3d}/{episodes} | Delivery {delivery_rate:.3f} | "
                  f"CELoss {avg_loss:.4f} | {time.time() - t0:.1f}s")

    # Save BC reward history
    np.save(f"logs/train_rewards_bc{tag}.npy", np.array(bc_rewards))
    return bc_rewards


def ppo_phase(actor, critic, env, n_sats, max_nb, device, episodes, bc_init_rewards, model_type="transformer", tag="", save_name="best_transformer"):
    """PPO fine-tuning from BC-initialized actor."""
    print(f"\n=== PPO Fine-tuning Phase: {episodes} episodes ===")

    # Low LR + low entropy: preserve BC-learned policy, only fine-tune
    trainer = PPO_CTDE(
        actor, critic, lr=2e-5,          # 5x smaller than default
        entropy_coef=0.005,              # 6x smaller for less exploration
        batch_size=cfg.MINI_BATCH_SIZE,
        device=device,
        max_iter=episodes * cfg.HORIZON_SLOTS,
        is_transformer=(model_type == "transformer"),
    )
    trainer.init_buffers(list(range(n_sats)))

    reward_history = list(bc_init_rewards)   # Continue from BC curve
    best_reward = max(bc_init_rewards) if bc_init_rewards else -math.inf

    for ep in range(1, episodes + 1):
        t0 = time.time()
        obs = env.reset(t_start=np.random.uniform(0, 6000))
        ep_delivered = 0
        ep_dropped = 0

        slot_gs_list = []
        for slot in range(cfg.HORIZON_SLOTS):
            gs = build_global_state(obs, n_sats, max_nb)
            slot_gs_list.append(gs)

            obs_list = [obs[i] for i in range(n_sats)]
            mob_list = [build_mob_vec(obs[i], env, i) for i in range(n_sats)]
            dist_list = [build_dist_vec(obs[i], env, i) for i in range(n_sats)]
            self_feat_list = [obs[i].get("self_feat", np.zeros(cfg.DIM_SELF, dtype=np.float32))
                              for i in range(n_sats)]

            actions, logps = trainer.select_actions_batch(
                obs_list, mob_list, max_nb,
                dist_list=dist_list, self_feat_list=self_feat_list,
            )

            for i in range(n_sats):
                obs_i = obs_list[i]
                if obs_i["features"].shape[0] == 0:
                    continue
                trainer.store(
                    agent_id=i,
                    feats=obs_i["features"], mob=mob_list[i],
                    mask=obs_i["mask"], action=actions[i], logp=logps[i],
                    reward=(0.0, 0.0, 0.0, 0.0), active=False, done=False,
                    slot_idx=slot, dist=dist_list[i], self_feat=self_feat_list[i],
                )

            next_obs, rewards, done, info = env.step(actions)

            for i in range(n_sats):
                comps = rewards.get(i)
                if trainer.buffers[i].transitions:
                    tr = trainer.buffers[i].transitions[-1]
                    if comps is not None:
                        tr.reward = (comps["dly"], comps["cng"], comps["dir"], comps["stb"])
                        tr.active = True
                    tr.done = done

            ep_delivered += info["n_delivered"]
            ep_dropped += info["n_dropped"]
            obs = next_obs
            if done:
                break

        ep_total = ep_delivered + ep_dropped
        ep_reward = ep_delivered / max(ep_total, 1)
        reward_history.append(ep_reward)

        stats = trainer.update(slot_gs_list=slot_gs_list)

        if ep == 1 or ep % 10 == 0:
            avg_r = np.mean(reward_history[-10:])
            print(f"PPO Ep {ep:3d}/{episodes} | Reward {ep_reward:.3f} | "
                  f"Avg(10) {avg_r:.3f} | PLoss {stats.get('policy_loss', 0):.4f} | "
                  f"VLoss {stats.get('value_loss', 0):.4f} | {time.time() - t0:.1f}s")

        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save({
                "episode": ep, "actor": actor.state_dict(),
                "critic": critic.state_dict(), "reward": best_reward,
            }, f"checkpoints/{save_name}.pt")

    # Always save FINAL model at end of training (not just best by single-ep reward).
    # Single-ep reward is noisy; final model is the real outcome of training.
    torch.save({
        "episode": episodes, "actor": actor.state_dict(),
        "critic": critic.state_dict(), "reward": ep_reward,
    }, f"checkpoints/{save_name}.pt")
    print(f"[Saved final checkpoint] checkpoints/{save_name}.pt")

    np.save(f"logs/train_rewards{tag}.npy", np.array(reward_history))
    print(f"\nTraining complete. Best reward: {best_reward:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc_episodes", type=int, default=30)
    parser.add_argument("--ppo_episodes", type=int, default=400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--planes", type=int, default=cfg.TRAIN_N_PLANES)
    parser.add_argument("--sats", type=int, default=cfg.TRAIN_N_SATS_PER_PLANE)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--n_sessions", type=int, default=cfg.N_GROUND_PAIRS)
    parser.add_argument("--model", choices=["transformer", "madrl", "grlr"], default="transformer")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    const = Constellation(args.planes, args.sats)
    n_sats = const.n_total
    tgen = TrafficGenerator(n_sats, n_fg_sessions=args.n_sessions, rng=rng)
    env = LEORoutingEnv(const, tgen, horizon=cfg.HORIZON_SLOTS, rng=rng)
    env.training = True

    MAX_NB = 4

    if args.model == "transformer":
        actor = TransformerActor(
            dim_in=cfg.DIM_IN, dim_self=cfg.DIM_SELF,
            d_model=cfg.D_MODEL, n_heads=cfg.N_HEADS,
            d_ff=cfg.D_FF, n_layers=cfg.N_LAYERS,
            dropout=cfg.DROPOUT, max_neighbors=MAX_NB,
        )
        save_name = "best_transformer"
    elif args.model == "madrl":
        actor = MAACAgent(max_neighbors=MAX_NB)
        save_name = "best_maac"
    else:  # grlr
        actor = GRLRAgent(max_neighbors=MAX_NB)
        save_name = "best_grlr"

    critic = CentralizedCritic(n_agents=n_sats, max_neighbors=MAX_NB)

    actor = actor.to(args.device)
    critic = critic.to(args.device)

    print(f"Training {args.model.upper()} (BC={args.bc_episodes} + PPO={args.ppo_episodes} ep)")
    print(f"Constellation: {args.planes}x{args.sats}={n_sats} agents, {args.n_sessions} sessions")

    # Phase 1: BC warm-start
    bc_rewards = bc_phase(
        actor, env, n_sats, MAX_NB, args.device,
        args.bc_episodes, model_type=args.model, tag=args.tag,
    )

    # Phase 2: PPO fine-tuning
    ppo_phase(
        actor, critic, env, n_sats, MAX_NB, args.device,
        args.ppo_episodes, bc_rewards, model_type=args.model,
        tag=args.tag, save_name=save_name,
    )


if __name__ == "__main__":
    main()
